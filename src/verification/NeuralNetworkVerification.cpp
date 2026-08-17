#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "raylib.h"

#include "ai/AIController.h"
#include "ai/FitnessEvaluator.h"
#include "ai/NeuralNetwork.h"
#include "ai/Observation.h"
#include "ai/neat/CompatibilityConfig.h"
#include "ai/neat/CompatibilityDistance.h"
#include "ai/neat/ConnectionGene.h"
#include "ai/neat/CrossoverConfig.h"
#include "ai/neat/Genome.h"
#include "ai/neat/GenomeCrossover.h"
#include "ai/neat/GenomeMutator.h"
#include "ai/neat/Individual.h"
#include "ai/neat/InnovationTracker.h"
#include "ai/neat/MutationConfig.h"
#include "ai/neat/NodeGene.h"
#include "ai/neat/PhenotypeBuilder.h"
#include "ai/neat/Population.h"
#include "ai/neat/PopulationConfig.h"
#include "ai/neat/SpeciationConfig.h"
#include "ai/neat/Speciator.h"
#include "ai/neat/Species.h"
#include "simulation/Car.h"
#include "simulation/Track.h"
#include "simulation/TrackProgress.h"
#include "simulation/TrackVisual.h"
#include "training/GenerationMetrics.h"
#include "training/TrainingLogger.h"

#include "AppConfig.h"
#include "verification/Verifications.h"

namespace verification
{


namespace nn_verify
{

// Builds the 9 Input + 1 Bias + 2 Output nodes every test network needs.
// Input node IDs are 0..8 (Observation slot order), bias is 9, steering
// output is 100 (first Output -> output index 0), throttle output is 101
// (second Output -> output index 1). Deliberately not contiguous/sorted
// with any hidden node IDs used below, so tests can prove evaluation
// doesn't depend on ID ordering.
std::vector<ai::Node> makeBaseNodes()
{
    std::vector<ai::Node> nodes;
    for (int i = 0; i < ai::NeuralNetwork::kInputCount; ++i)
    {
        nodes.push_back(ai::Node{i, ai::NodeType::Input});
    }
    nodes.push_back(ai::Node{9, ai::NodeType::Bias});
    nodes.push_back(ai::Node{100, ai::NodeType::Output}); // steering
    nodes.push_back(ai::Node{101, ai::NodeType::Output}); // throttle
    return nodes;
}

ai::Observation makeObservation(int index, float value)
{
    ai::Observation obs;
    obs.values.fill(0.0f);
    if (index >= 0)
    {
        obs.values[index] = value;
    }
    return obs;
}

template <typename Callable>
bool throwsInvalidArgument(Callable&& callable)
{
    try
    {
        callable();
    }
    catch (const std::invalid_argument&)
    {
        return true;
    }
    return false;
}

} // namespace nn_verify

// One-shot, deterministic sanity check of ai::NeuralNetwork's construction,
// validation and evaluation, independent of Car/Track/keyboard/render
// timing. Runs once at startup.
void verifyNeuralNetwork()
{
    using namespace nn_verify;
    constexpr float kEps = 1e-4f;

    // 1 & 9: exactly 9 inputs + 1 bias + 2 outputs can be constructed; a
    // fully disconnected Output produces 0 (tanh of an empty sum).
    {
        ai::NeuralNetwork net(makeBaseNodes(), {});
        const auto out = net.evaluate(makeObservation(-1, 0.0f));
        assert(out[0] == 0.0f && out[1] == 0.0f && "disconnected outputs must evaluate to exactly 0");
    }

    // 2: direct Input -> Output connection produces the expected tanh result.
    {
        ai::NeuralNetwork net(makeBaseNodes(), {ai::Connection{0, 100, 0.5f, true}});
        const auto out = net.evaluate(makeObservation(0, 1.0f));
        assert(std::fabs(out[0] - std::tanh(0.5f)) < kEps && "Input->Output must equal tanh(input * weight)");
        assert(out[1] == 0.0f && "unrelated disconnected output must stay 0");
    }

    // 3: Bias -> Output affects output correctly (bias is always 1.0).
    {
        ai::NeuralNetwork net(makeBaseNodes(), {ai::Connection{9, 100, 0.7f, true}});
        const auto out = net.evaluate(makeObservation(-1, 0.0f));
        assert(std::fabs(out[0] - std::tanh(0.7f)) < kEps && "Bias->Output must equal tanh(1.0 * weight)");
    }

    // 4: multiple incoming connections are summed before activation.
    {
        std::vector<ai::Connection> conns = {
            ai::Connection{0, 100, 0.3f, true},
            ai::Connection{1, 100, 0.4f, true},
        };
        ai::NeuralNetwork net(makeBaseNodes(), conns);
        ai::Observation obs;
        obs.values.fill(0.0f);
        obs.values[0] = 1.0f;
        obs.values[1] = 1.0f;
        const auto out = net.evaluate(obs);
        assert(std::fabs(out[0] - std::tanh(0.3f + 0.4f)) < kEps &&
               "multiple incoming connections must be summed before tanh");
    }

    // 5: Input -> Hidden -> Output produces the mathematically expected result.
    {
        std::vector<ai::Node> nodes = makeBaseNodes();
        nodes.push_back(ai::Node{50, ai::NodeType::Hidden});
        std::vector<ai::Connection> conns = {
            ai::Connection{0, 50, 0.5f, true},
            ai::Connection{50, 100, 2.0f, true},
        };
        ai::NeuralNetwork net(nodes, conns);
        const auto out = net.evaluate(makeObservation(0, 1.0f));
        const float hidden = std::tanh(1.0f * 0.5f);
        const float expected = std::tanh(hidden * 2.0f);
        assert(std::fabs(out[0] - expected) < kEps && "Input->Hidden->Output result mismatch");
    }

    // 6: a deeper feed-forward path Input -> HiddenA -> HiddenB -> Output evaluates correctly.
    {
        std::vector<ai::Node> nodes = makeBaseNodes();
        nodes.push_back(ai::Node{50, ai::NodeType::Hidden});
        nodes.push_back(ai::Node{51, ai::NodeType::Hidden});
        std::vector<ai::Connection> conns = {
            ai::Connection{0, 50, 1.0f, true},
            ai::Connection{50, 51, 1.0f, true},
            ai::Connection{51, 100, 1.0f, true},
        };
        ai::NeuralNetwork net(nodes, conns);
        const auto out = net.evaluate(makeObservation(0, 0.5f));
        const float a = std::tanh(0.5f);
        const float b = std::tanh(a);
        const float expected = std::tanh(b);
        assert(std::fabs(out[0] - expected) < kEps && "deep Input->HiddenA->HiddenB->Output result mismatch");
    }

    // 7: a connection that skips hidden nodes evaluates correctly alongside a hidden path.
    {
        std::vector<ai::Node> nodes = makeBaseNodes();
        nodes.push_back(ai::Node{50, ai::NodeType::Hidden});
        std::vector<ai::Connection> conns = {
            ai::Connection{0, 50, 1.0f, true},  // input -> hidden
            ai::Connection{50, 100, 1.0f, true}, // hidden -> output
            ai::Connection{0, 100, 1.0f, true},  // input -> output, skipping the hidden node
        };
        ai::NeuralNetwork net(nodes, conns);
        const auto out = net.evaluate(makeObservation(0, 0.5f));
        const float hidden = std::tanh(0.5f);
        const float expected = std::tanh(hidden * 1.0f + 0.5f * 1.0f);
        assert(std::fabs(out[0] - expected) < kEps && "skip connection combined with hidden path result mismatch");
    }

    // 8: evaluation doesn't depend on node ID order -- Hidden 999 feeds
    // Output 100 despite the larger ID; topological order must still put it first.
    {
        std::vector<ai::Node> nodes = makeBaseNodes();
        nodes.push_back(ai::Node{999, ai::NodeType::Hidden});
        std::vector<ai::Connection> conns = {
            ai::Connection{0, 999, 1.0f, true},
            ai::Connection{999, 100, 1.0f, true},
        };
        ai::NeuralNetwork net(nodes, conns);
        const auto out = net.evaluate(makeObservation(0, 1.0f));
        const float expected = std::tanh(std::tanh(1.0f));
        assert(std::fabs(out[0] - expected) < kEps &&
               "evaluation must follow actual dependencies, not node ID order");
    }

    // 10: invalid source/destination node IDs are rejected.
    {
        assert(throwsInvalidArgument([]() {
            ai::NeuralNetwork net(makeBaseNodes(), {ai::Connection{0, 12345, 0.1f, true}});
        }) && "connection to an unknown node ID must be rejected");
        assert(throwsInvalidArgument([]() {
            ai::NeuralNetwork net(makeBaseNodes(), {ai::Connection{12345, 100, 0.1f, true}});
        }) && "connection from an unknown node ID must be rejected");
    }

    // 11: duplicate node IDs are rejected.
    {
        assert(throwsInvalidArgument([]() {
            std::vector<ai::Node> nodes = makeBaseNodes();
            nodes.push_back(ai::Node{0, ai::NodeType::Hidden}); // reuses input 0's ID
            ai::NeuralNetwork net(nodes, {});
        }) && "duplicate node IDs must be rejected");
    }

    // 12: duplicate directed connections are rejected.
    {
        assert(throwsInvalidArgument([]() {
            std::vector<ai::Connection> conns = {
                ai::Connection{0, 100, 0.1f, true},
                ai::Connection{0, 100, 0.2f, true},
            };
            ai::NeuralNetwork net(makeBaseNodes(), conns);
        }) && "duplicate (source, target) connections must be rejected");
    }

    // 13: a connection targeting Input is rejected.
    {
        assert(throwsInvalidArgument([]() {
            ai::NeuralNetwork net(makeBaseNodes(), {ai::Connection{100, 0, 0.1f, true}});
        }) && "a connection targeting an Input node must be rejected");
    }

    // 14: a connection targeting Bias is rejected.
    {
        assert(throwsInvalidArgument([]() {
            ai::NeuralNetwork net(makeBaseNodes(), {ai::Connection{0, 9, 0.1f, true}});
        }) && "a connection targeting the Bias node must be rejected");
    }

    // 15: a cyclic graph is rejected.
    {
        assert(throwsInvalidArgument([]() {
            std::vector<ai::Node> nodes = makeBaseNodes();
            nodes.push_back(ai::Node{50, ai::NodeType::Hidden});
            nodes.push_back(ai::Node{51, ai::NodeType::Hidden});
            std::vector<ai::Connection> conns = {
                ai::Connection{50, 51, 1.0f, true},
                ai::Connection{51, 50, 1.0f, true},
            };
            ai::NeuralNetwork net(nodes, conns);
        }) && "a cyclic graph must be rejected");
    }

    // 16: incorrect Input/Bias/Output counts are rejected.
    {
        assert(throwsInvalidArgument([]() {
            std::vector<ai::Node> nodes = makeBaseNodes();
            nodes.erase(nodes.begin()); // drops one Input, leaving 8
            ai::NeuralNetwork net(nodes, {});
        }) && "fewer than kInputCount Input nodes must be rejected");

        assert(throwsInvalidArgument([]() {
            std::vector<ai::Node> nodes = makeBaseNodes();
            nodes.erase(nodes.begin() + ai::NeuralNetwork::kInputCount); // drops the Bias node
            ai::NeuralNetwork net(nodes, {});
        }) && "a missing Bias node must be rejected");

        assert(throwsInvalidArgument([]() {
            std::vector<ai::Node> nodes = makeBaseNodes();
            nodes.pop_back(); // drops the throttle Output, leaving 1
            ai::NeuralNetwork net(nodes, {});
        }) && "fewer than kOutputCount Output nodes must be rejected");
    }

    // 17: output values are returned in deterministic steering/throttle order.
    {
        std::vector<ai::Connection> conns = {
            ai::Connection{0, 100, 1.0f, true},
            ai::Connection{0, 101, 2.0f, true},
        };
        ai::NeuralNetwork net(makeBaseNodes(), conns);
        const auto out = net.evaluate(makeObservation(0, 1.0f));
        assert(std::fabs(out[0] - std::tanh(1.0f)) < kEps && "output index 0 must be the steering (first Output) node");
        assert(std::fabs(out[1] - std::tanh(2.0f)) < kEps && "output index 1 must be the throttle (second Output) node");
    }

    TraceLog(LOG_INFO, "Neural network verification: all deterministic checks passed");
}

namespace phenotype_verify
{

// 9 Inputs (0-8) + Bias (9) + 2 Outputs (100 = steering, 101 = throttle),
// added out of ID order -- proves buildPhenotype()'s slot ordering depends
// on node ID, never insertion order.
ai::neat::Genome makeBaseGenome()
{
    ai::neat::Genome genome;
    genome.addNode(ai::neat::NodeGene{9, ai::neat::NodeType::Bias});
    genome.addNode(ai::neat::NodeGene{101, ai::neat::NodeType::Output});
    genome.addNode(ai::neat::NodeGene{100, ai::neat::NodeType::Output});
    for (int i = ai::NeuralNetwork::kInputCount - 1; i >= 0; --i)
    {
        genome.addNode(ai::neat::NodeGene{i, ai::neat::NodeType::Input});
    }
    return genome;
}

ai::Observation makeObservation(int index, float value)
{
    ai::Observation obs;
    obs.values.fill(0.0f);
    if (index >= 0)
    {
        obs.values[index] = value;
    }
    return obs;
}

template <typename Callable>
bool throwsInvalidArgument(Callable&& callable)
{
    try
    {
        callable();
    }
    catch (const std::invalid_argument&)
    {
        return true;
    }
    return false;
}

} // namespace phenotype_verify

// Deterministic check of ai::neat::buildPhenotype (Genome -> NeuralNetwork),
// end to end. No mutation/crossover here.
void verifyPhenotypeBuilder()
{
    using namespace phenotype_verify;
    using ai::neat::buildPhenotype;
    using ai::neat::ConnectionGene;
    using ai::neat::Genome;
    using ai::neat::NodeGene;
    using ai::neat::NodeType;
    constexpr float kEps = 1e-4f;

    // 1 & 3: a minimal valid Genome builds successfully -- only possible if
    // NodeGene types were mapped to the matching runtime NodeType.
    {
        ai::NeuralNetwork net = buildPhenotype(makeBaseGenome());
        const auto out = net.evaluate(makeObservation(-1, 0.0f));
        assert(out[0] == 0.0f && out[1] == 0.0f && "a fully disconnected phenotype must evaluate to exactly 0");
    }

    // 2 & 8: large, non-contiguous node IDs (and connection endpoints
    // referencing them) are preserved exactly -- renumbering would fail
    // construction or evaluate the wrong node.
    {
        Genome genome;
        for (int i = 0; i < ai::NeuralNetwork::kInputCount; ++i)
        {
            genome.addNode(NodeGene{1000 + i, NodeType::Input});
        }
        genome.addNode(NodeGene{2000, NodeType::Bias});
        genome.addNode(NodeGene{3000, NodeType::Output});
        genome.addNode(NodeGene{3001, NodeType::Output});
        genome.addConnection(ConnectionGene{1000, 3000, 0.5f, true, 0});

        ai::NeuralNetwork net = buildPhenotype(genome);
        const auto out = net.evaluate(makeObservation(0, 1.0f));
        assert(std::fabs(out[0] - std::tanh(0.5f)) < kEps &&
               "non-contiguous node IDs must be preserved through phenotype construction");
    }

    // 4: input slot ordering follows ascending node ID, independent of
    // Genome insertion order (makeBaseGenome adds Inputs in descending ID
    // order).
    {
        Genome genome = makeBaseGenome();
        genome.addConnection(ConnectionGene{0, 100, 1.0f, true, 0});
        genome.addConnection(ConnectionGene{ai::NeuralNetwork::kInputCount - 1, 101, 1.0f, true, 1});
        ai::NeuralNetwork net = buildPhenotype(genome);

        const auto outLow = net.evaluate(makeObservation(0, 1.0f));
        assert(std::fabs(outLow[0] - std::tanh(1.0f)) < kEps &&
               "Observation slot 0 must drive the lowest-ID Input node");
        assert(outLow[1] == 0.0f && "Observation slot 0 must not affect the highest-ID Input node");

        const auto outHigh = net.evaluate(makeObservation(ai::NeuralNetwork::kInputCount - 1, 1.0f));
        assert(outHigh[0] == 0.0f && "the last Observation slot must not affect the lowest-ID Input node");
        assert(std::fabs(outHigh[1] - std::tanh(1.0f)) < kEps &&
               "the last Observation slot must drive the highest-ID Input node");
    }

    // 5: output slot ordering follows ascending node ID, independent of
    // Genome insertion order (makeBaseGenome adds Output 101 before 100).
    {
        Genome genome = makeBaseGenome();
        genome.addConnection(ConnectionGene{0, 100, 1.0f, true, 0});
        genome.addConnection(ConnectionGene{0, 101, 2.0f, true, 1});
        ai::NeuralNetwork net = buildPhenotype(genome);

        const auto out = net.evaluate(makeObservation(0, 1.0f));
        assert(std::fabs(out[0] - std::tanh(1.0f)) < kEps && "output slot 0 must be the lower-ID Output node (100)");
        assert(std::fabs(out[1] - std::tanh(2.0f)) < kEps && "output slot 1 must be the higher-ID Output node (101)");
    }

    // 6: Bias maps correctly (always contributes 1.0) and stays internal --
    // it is not one of the kInputCount external Observation slots.
    {
        Genome genome = makeBaseGenome();
        genome.addConnection(ConnectionGene{9, 100, 0.7f, true, 0});
        ai::NeuralNetwork net = buildPhenotype(genome);
        const auto out = net.evaluate(makeObservation(-1, 0.0f));
        assert(std::fabs(out[0] - std::tanh(0.7f)) < kEps &&
               "Bias->Output must equal tanh(1.0 * weight) even with an all-zero Observation");
    }

    // 7 & 14: Hidden nodes map correctly; Input -> Hidden -> Output
    // evaluates to the mathematically expected result.
    {
        Genome genome = makeBaseGenome();
        genome.addNode(NodeGene{50, NodeType::Hidden});
        genome.addConnection(ConnectionGene{0, 50, 0.5f, true, 0});
        genome.addConnection(ConnectionGene{50, 100, 2.0f, true, 1});
        ai::NeuralNetwork net = buildPhenotype(genome);

        const auto out = net.evaluate(makeObservation(0, 1.0f));
        const float hidden = std::tanh(1.0f * 0.5f);
        const float expected = std::tanh(hidden * 2.0f);
        assert(std::fabs(out[0] - expected) < kEps && "Input->Hidden->Output result mismatch");
    }

    // 9, 10 & 11: weights and the enabled flag are preserved exactly, and a
    // disabled connection contributes nothing to evaluation.
    {
        Genome genome = makeBaseGenome();
        genome.addNode(NodeGene{60, NodeType::Hidden});
        genome.addConnection(ConnectionGene{0, 100, 0.37f, true, 0});   // enabled, contributes
        genome.addConnection(ConnectionGene{0, 60, -1.25f, false, 1}); // disabled, must not contribute
        genome.addConnection(ConnectionGene{60, 100, 4.0f, true, 2});  // would matter if 0->60 were active
        ai::NeuralNetwork net = buildPhenotype(genome);

        const auto out = net.evaluate(makeObservation(0, 1.0f));
        assert(std::fabs(out[0] - std::tanh(0.37f)) < kEps &&
               "a disabled connection must not contribute to evaluation, and enabled weight must be exact");
    }

    // 12: direct Input -> Output phenotype evaluates correctly.
    {
        Genome genome = makeBaseGenome();
        genome.addConnection(ConnectionGene{0, 100, 0.5f, true, 0});
        ai::NeuralNetwork net = buildPhenotype(genome);
        const auto out = net.evaluate(makeObservation(0, 1.0f));
        assert(std::fabs(out[0] - std::tanh(0.5f)) < kEps && "Input->Output must equal tanh(input * weight)");
    }

    // 13: Bias -> Output phenotype evaluates correctly (duplicate of #6's
    // arithmetic, kept as its own case per the required verification list).
    {
        Genome genome = makeBaseGenome();
        genome.addConnection(ConnectionGene{9, 100, 1.1f, true, 0});
        ai::NeuralNetwork net = buildPhenotype(genome);
        const auto out = net.evaluate(makeObservation(-1, 0.0f));
        assert(std::fabs(out[0] - std::tanh(1.1f)) < kEps && "Bias->Output must equal tanh(1.0 * weight)");
    }

    // 15: a deeper feed-forward DAG (Input -> HiddenA -> HiddenB -> Output)
    // evaluates correctly.
    {
        Genome genome = makeBaseGenome();
        genome.addNode(NodeGene{50, NodeType::Hidden});
        genome.addNode(NodeGene{51, NodeType::Hidden});
        genome.addConnection(ConnectionGene{0, 50, 1.0f, true, 0});
        genome.addConnection(ConnectionGene{50, 51, 1.0f, true, 1});
        genome.addConnection(ConnectionGene{51, 100, 1.0f, true, 2});
        ai::NeuralNetwork net = buildPhenotype(genome);

        const auto out = net.evaluate(makeObservation(0, 0.5f));
        const float a = std::tanh(0.5f);
        const float b = std::tanh(a);
        const float expected = std::tanh(b);
        assert(std::fabs(out[0] - expected) < kEps && "deep Input->HiddenA->HiddenB->Output result mismatch");
    }

    // 16: a skip connection alongside a hidden path evaluates correctly.
    {
        Genome genome = makeBaseGenome();
        genome.addNode(NodeGene{50, NodeType::Hidden});
        genome.addConnection(ConnectionGene{0, 50, 1.0f, true, 0});   // input -> hidden
        genome.addConnection(ConnectionGene{50, 100, 1.0f, true, 1}); // hidden -> output
        genome.addConnection(ConnectionGene{0, 100, 1.0f, true, 2});  // input -> output, skipping hidden
        ai::NeuralNetwork net = buildPhenotype(genome);

        const auto out = net.evaluate(makeObservation(0, 0.5f));
        const float hidden = std::tanh(0.5f);
        const float expected = std::tanh(hidden * 1.0f + 0.5f * 1.0f);
        assert(std::fabs(out[0] - expected) < kEps && "skip connection combined with hidden path result mismatch");
    }

    // 17: an invalid Genome fails because buildPhenotype calls validate()
    // before touching NeuralNetwork.
    {
        std::vector<NodeGene> nodes = {NodeGene{0, NodeType::Input}, NodeGene{0, NodeType::Hidden}};
        Genome invalidGenome(nodes, {});
        assert(throwsInvalidArgument([&]() { buildPhenotype(invalidGenome); }) &&
               "a Genome that fails validate() must be rejected by buildPhenotype");
    }

    // 18: a cyclic Genome is valid at the Genome level (no cycle check
    // there) but must fail phenotype construction (NeuralNetwork rejects it).
    {
        Genome genome = makeBaseGenome();
        genome.addNode(NodeGene{50, NodeType::Hidden});
        genome.addNode(NodeGene{51, NodeType::Hidden});
        genome.addConnection(ConnectionGene{50, 51, 1.0f, true, 0});
        genome.addConnection(ConnectionGene{51, 50, 1.0f, true, 1});
        genome.validate(); // must not throw -- Genome has no cycle check
        assert(throwsInvalidArgument([&]() { buildPhenotype(genome); }) &&
               "a cyclic enabled Genome must be rejected by the feed-forward-only NeuralNetwork");
    }

    // 19: buildPhenotype does not alter the source Genome.
    {
        Genome genome = makeBaseGenome();
        genome.addNode(NodeGene{50, NodeType::Hidden});
        genome.addConnection(ConnectionGene{0, 50, 0.3f, true, 0});
        genome.addConnection(ConnectionGene{50, 100, 0.4f, true, 1});

        const std::vector<NodeGene> nodesBefore = genome.nodes();
        const std::size_t connectionCountBefore = genome.connections().size();

        ai::NeuralNetwork net = buildPhenotype(genome);
        (void)net;

        assert(genome.nodes().size() == nodesBefore.size() && "buildPhenotype must not add or remove genome nodes");
        for (std::size_t i = 0; i < nodesBefore.size(); ++i)
        {
            assert(genome.nodes()[i] == nodesBefore[i] && "buildPhenotype must not modify existing genome nodes");
        }
        assert(genome.connections().size() == connectionCountBefore &&
               "buildPhenotype must not add or remove genome connections");
        assert(genome.hasConnection(0, 50) && genome.hasConnection(50, 100) &&
               "buildPhenotype must leave genome connections intact");
    }

    // 20: phenotype evaluation is deterministic across repeated builds from
    // the same Genome.
    {
        Genome genome = makeBaseGenome();
        genome.addNode(NodeGene{50, NodeType::Hidden});
        genome.addConnection(ConnectionGene{0, 50, 0.6f, true, 0});
        genome.addConnection(ConnectionGene{50, 100, -0.9f, true, 1});
        genome.addConnection(ConnectionGene{9, 101, 0.2f, true, 2});

        ai::NeuralNetwork netA = buildPhenotype(genome);
        ai::NeuralNetwork netB = buildPhenotype(genome);

        const auto obs = makeObservation(0, 0.8f);
        const auto outA = netA.evaluate(obs);
        const auto outB = netB.evaluate(obs);
        assert(outA[0] == outB[0] && outA[1] == outB[1] &&
               "repeated builds from the same Genome must evaluate identically");
    }

    TraceLog(LOG_INFO, "Phenotype builder verification: all deterministic checks passed");
}
} // namespace verification
