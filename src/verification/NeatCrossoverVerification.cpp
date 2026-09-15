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


namespace genome_crossover_verify
{

// kInputCount Input + 1 Bias + 3 Output, no connections -- the minimal
// interface every phenotype-buildable genome in this suite needs; both
// parents share it so the child inherits it regardless of which connections
// it gets.
ai::neat::Genome makeInterfaceGenome()
{
    using ai::neat::Genome;
    using ai::neat::NodeGene;
    using ai::neat::NodeType;

    Genome genome;
    for (int i = 0; i < ai::NeuralNetwork::kInputCount; ++i)
    {
        genome.addNode(NodeGene{i, NodeType::Input});
    }
    genome.addNode(NodeGene{ai::NeuralNetwork::kInputCount, NodeType::Bias});
    genome.addNode(NodeGene{100, NodeType::Output});
    genome.addNode(NodeGene{101, NodeType::Output});
    genome.addNode(NodeGene{102, NodeType::Output});
    return genome;
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

} // namespace genome_crossover_verify

// Deterministic check of ai::neat::GenomeCrossover, end to end. No
// compatibility distance/species/population/reproduction logic here --
// only crossover.
void verifyGenomeCrossover()
{
    using namespace genome_crossover_verify;
    using ai::neat::ConnectionGene;
    using ai::neat::CrossoverConfig;
    using ai::neat::Genome;
    using ai::neat::GenomeCrossover;
    using ai::neat::NodeGene;
    using ai::neat::NodeType;

    // 1 & 2: invalid/non-finite config probabilities are rejected.
    {
        Genome a = makeInterfaceGenome();
        Genome b = makeInterfaceGenome();
        GenomeCrossover crossover(1u);

        CrossoverConfig outOfRange;
        outOfRange.matchingGeneChooseParentAProbability = 1.5f;
        assert(throwsInvalidArgument([&]() { crossover.crossover(a, 1.0f, b, 1.0f, outOfRange); }) &&
               "an out-of-range matchingGeneChooseParentAProbability must be rejected"); // 1

        CrossoverConfig negative;
        negative.disabledGeneRemainDisabledProbability = -0.1f;
        assert(throwsInvalidArgument([&]() { crossover.crossover(a, 1.0f, b, 1.0f, negative); }) &&
               "a negative disabledGeneRemainDisabledProbability must be rejected"); // 1 (continued)

        CrossoverConfig nanConfig;
        nanConfig.matchingGeneChooseParentAProbability = std::numeric_limits<float>::quiet_NaN();
        assert(throwsInvalidArgument([&]() { crossover.crossover(a, 1.0f, b, 1.0f, nanConfig); }) &&
               "a NaN config probability must be rejected"); // 2

        CrossoverConfig infConfig;
        infConfig.disabledGeneRemainDisabledProbability = std::numeric_limits<float>::infinity();
        assert(throwsInvalidArgument([&]() { crossover.crossover(a, 1.0f, b, 1.0f, infConfig); }) &&
               "a non-finite config probability must be rejected"); // 2 (continued)
    }

    // 3: non-finite parent fitness is rejected.
    {
        Genome a = makeInterfaceGenome();
        Genome b = makeInterfaceGenome();
        GenomeCrossover crossover(1u);
        CrossoverConfig config;

        assert(throwsInvalidArgument([&]() { crossover.crossover(a, std::numeric_limits<float>::quiet_NaN(), b, 1.0f, config); }) &&
               "a NaN fitnessA must be rejected");
        assert(throwsInvalidArgument([&]() { crossover.crossover(a, 1.0f, b, std::numeric_limits<float>::infinity(), config); }) &&
               "a non-finite fitnessB must be rejected");
    }

    // 4: empty (interface-only) parents produce a valid child -- all
    // mandatory nodes, no connections, still buildPhenotype()-able.
    {
        Genome a = makeInterfaceGenome();
        Genome b = makeInterfaceGenome();
        GenomeCrossover crossover(1u);
        CrossoverConfig config;

        const Genome child = crossover.crossover(a, 1.0f, b, 1.0f, config);
        assert(child.connections().empty() && "no connections in either parent must yield no connections in the child");
        assert(child.nodes().size() == static_cast<std::size_t>(ai::NeuralNetwork::kInputCount) + 4 &&
               "the child must contain exactly the kInputCount Input + 1 Bias + 3 Output nodes");

        child.validate();
        ai::NeuralNetwork net = ai::neat::buildPhenotype(child);
        const auto out = net.evaluate(makeObservation(-1, 0.0f));
        assert(out[0] == 0.0f && out[1] == 0.0f && out[2] == 0.0f &&
               "a fully disconnected child phenotype must evaluate to exactly 0");
    }

    // 5: matching genes align by innovation number, not vector index --
    // parent A stores innovation 5 before innovation 2, parent B stores
    // innovation 2 before innovation 5.
    {
        Genome parentA = makeInterfaceGenome();
        parentA.addConnection(ConnectionGene{1, 100, 0.33f, true, 5});
        parentA.addConnection(ConnectionGene{0, 100, 0.11f, true, 2});

        Genome parentB = makeInterfaceGenome();
        parentB.addConnection(ConnectionGene{0, 100, 0.22f, true, 2});
        parentB.addConnection(ConnectionGene{1, 100, 0.44f, true, 5});

        GenomeCrossover crossover(1u);
        CrossoverConfig config;
        config.matchingGeneChooseParentAProbability = 1.0f; // always A, so the result is fully deterministic here

        const Genome child = crossover.crossover(parentA, 1.0f, parentB, 1.0f, config);
        const ConnectionGene* innov2 = child.findConnection(0, 100);
        const ConnectionGene* innov5 = child.findConnection(1, 100);
        assert(innov2 != nullptr && innov2->getInnovationNumber() == 2 && innov2->getWeight() == 0.11f &&
               "innovation 2 must be matched to innovation 2, regardless of vector position");
        assert(innov5 != nullptr && innov5->getInnovationNumber() == 5 && innov5->getWeight() == 0.33f &&
               "innovation 5 must be matched to innovation 5, regardless of vector position");
    }

    // 6: matching gene can inherit parent A's weight.
    {
        Genome parentA = makeInterfaceGenome();
        parentA.addConnection(ConnectionGene{0, 100, 0.11f, true, 0});
        Genome parentB = makeInterfaceGenome();
        parentB.addConnection(ConnectionGene{0, 100, 0.99f, true, 0});

        GenomeCrossover crossover(1u);
        CrossoverConfig config;
        config.matchingGeneChooseParentAProbability = 1.0f;
        const Genome child = crossover.crossover(parentA, 1.0f, parentB, 1.0f, config);
        assert(child.findConnection(0, 100)->getWeight() == 0.11f &&
               "probability 1.0 must always inherit parent A's weight for a matching gene");
    }

    // 7: matching gene can inherit parent B's weight.
    {
        Genome parentA = makeInterfaceGenome();
        parentA.addConnection(ConnectionGene{0, 100, 0.11f, true, 0});
        Genome parentB = makeInterfaceGenome();
        parentB.addConnection(ConnectionGene{0, 100, 0.99f, true, 0});

        GenomeCrossover crossover(1u);
        CrossoverConfig config;
        config.matchingGeneChooseParentAProbability = 0.0f;
        const Genome child = crossover.crossover(parentA, 1.0f, parentB, 1.0f, config);
        assert(child.findConnection(0, 100)->getWeight() == 0.99f &&
               "probability 0.0 must always inherit parent B's weight for a matching gene");
    }

    // 8: matching innovation with conflicting endpoints between parents is
    // rejected.
    {
        Genome parentA = makeInterfaceGenome();
        parentA.addConnection(ConnectionGene{0, 100, 0.5f, true, 7});
        Genome parentB = makeInterfaceGenome();
        parentB.addConnection(ConnectionGene{1, 100, 0.5f, true, 7}); // same innovation, different source

        GenomeCrossover crossover(1u);
        CrossoverConfig config;
        assert(throwsInvalidArgument([&]() { crossover.crossover(parentA, 1.0f, parentB, 1.0f, config); }) &&
               "a matching innovation with conflicting endpoints between parents must be rejected");
    }

    // 9: a matching gene enabled in both parents stays enabled -- uses
    // probability 1.0, which would force disabled if the both-enabled
    // shortcut were broken.
    {
        Genome parentA = makeInterfaceGenome();
        parentA.addConnection(ConnectionGene{0, 100, 0.5f, true, 0});
        Genome parentB = makeInterfaceGenome();
        parentB.addConnection(ConnectionGene{0, 100, 0.6f, true, 0});

        GenomeCrossover crossover(1u);
        CrossoverConfig config;
        config.disabledGeneRemainDisabledProbability = 1.0f;
        const Genome child = crossover.crossover(parentA, 1.0f, parentB, 1.0f, config);
        assert(child.findConnection(0, 100)->isEnabled() && "a matching gene enabled in both parents must remain enabled");
    }

    // 10: a matching gene disabled in one parent can remain disabled.
    {
        Genome parentA = makeInterfaceGenome();
        parentA.addConnection(ConnectionGene{0, 100, 0.5f, false, 0});
        Genome parentB = makeInterfaceGenome();
        parentB.addConnection(ConnectionGene{0, 100, 0.6f, true, 0});

        GenomeCrossover crossover(1u);
        CrossoverConfig config;
        config.disabledGeneRemainDisabledProbability = 1.0f;
        const Genome child = crossover.crossover(parentA, 1.0f, parentB, 1.0f, config);
        assert(!child.findConnection(0, 100)->isEnabled() &&
               "probability 1.0 must keep a partially-disabled matching gene disabled");
    }

    // 11: a matching gene disabled in one parent can become enabled.
    {
        Genome parentA = makeInterfaceGenome();
        parentA.addConnection(ConnectionGene{0, 100, 0.5f, false, 0});
        Genome parentB = makeInterfaceGenome();
        parentB.addConnection(ConnectionGene{0, 100, 0.6f, true, 0});

        GenomeCrossover crossover(1u);
        CrossoverConfig config;
        config.disabledGeneRemainDisabledProbability = 0.0f;
        const Genome child = crossover.crossover(parentA, 1.0f, parentB, 1.0f, config);
        assert(child.findConnection(0, 100)->isEnabled() &&
               "probability 0.0 must always enable a partially-disabled matching gene");
    }

    // 12: a matching gene disabled in both parents follows the same
    // disabled-gene probability rule.
    {
        Genome parentA = makeInterfaceGenome();
        parentA.addConnection(ConnectionGene{0, 100, 0.5f, false, 0});
        Genome parentB = makeInterfaceGenome();
        parentB.addConnection(ConnectionGene{0, 100, 0.6f, false, 0});

        CrossoverConfig config;

        GenomeCrossover crossoverStaysDisabled(1u);
        config.disabledGeneRemainDisabledProbability = 1.0f;
        const Genome childStaysDisabled = crossoverStaysDisabled.crossover(parentA, 1.0f, parentB, 1.0f, config);
        assert(!childStaysDisabled.findConnection(0, 100)->isEnabled() &&
               "probability 1.0 must keep a both-disabled matching gene disabled");

        GenomeCrossover crossoverBecomesEnabled(1u);
        config.disabledGeneRemainDisabledProbability = 0.0f;
        const Genome childBecomesEnabled = crossoverBecomesEnabled.crossover(parentA, 1.0f, parentB, 1.0f, config);
        assert(childBecomesEnabled.findConnection(0, 100)->isEnabled() &&
               "probability 0.0 must enable a both-disabled matching gene");
    }

    // 13 & 14: the fitter parent A's non-matching genes are inherited; the
    // less-fit parent B's non-matching genes are excluded.
    {
        Genome parentA = makeInterfaceGenome();
        parentA.addConnection(ConnectionGene{2, 100, 0.5f, true, 10});
        Genome parentB = makeInterfaceGenome();
        parentB.addConnection(ConnectionGene{3, 101, 0.6f, true, 11});

        GenomeCrossover crossover(1u);
        CrossoverConfig config;
        const Genome child = crossover.crossover(parentA, 10.0f, parentB, 5.0f, config); // A fitter

        assert(child.hasConnection(2, 100) && child.findConnection(2, 100)->getInnovationNumber() == 10 &&
               "the fitter parent A's non-matching gene must be inherited"); // 13
        assert(!child.hasConnection(3, 101) && "the less-fit parent B's non-matching gene must be excluded"); // 14
    }

    // 15 & 16: symmetric case -- fitter parent B's non-matching genes are
    // inherited; less-fit parent A's non-matching genes are excluded.
    {
        Genome parentA = makeInterfaceGenome();
        parentA.addConnection(ConnectionGene{2, 100, 0.5f, true, 10});
        Genome parentB = makeInterfaceGenome();
        parentB.addConnection(ConnectionGene{3, 101, 0.6f, true, 11});

        GenomeCrossover crossover(1u);
        CrossoverConfig config;
        const Genome child = crossover.crossover(parentA, 5.0f, parentB, 10.0f, config); // B fitter

        assert(child.hasConnection(3, 101) && child.findConnection(3, 101)->getInnovationNumber() == 11 &&
               "the fitter parent B's non-matching gene must be inherited"); // 15
        assert(!child.hasConnection(2, 100) && "the less-fit parent A's non-matching gene must be excluded"); // 16
    }

    // 17: with equal fitness, non-matching genes from *both* parents may
    // appear in the child -- looped over enough seeds that both must be
    // observed at least once.
    {
        Genome parentA = makeInterfaceGenome();
        parentA.addConnection(ConnectionGene{2, 100, 0.5f, true, 20});
        Genome parentB = makeInterfaceGenome();
        parentB.addConnection(ConnectionGene{3, 101, 0.6f, true, 21});

        CrossoverConfig config;
        bool sawA = false;
        bool sawB = false;
        for (std::uint32_t seed = 1; seed <= 20; ++seed)
        {
            GenomeCrossover crossover(seed);
            const Genome child = crossover.crossover(parentA, 10.0f, parentB, 10.0f, config); // equal fitness
            if (child.hasConnection(2, 100))
            {
                sawA = true;
            }
            if (child.hasConnection(3, 101))
            {
                sawB = true;
            }
        }
        assert(sawA && sawB && "equal-fitness non-matching genes from both parents must each be inheritable");
    }

    // 18: an equal-fitness candidate that would duplicate a directed
    // connection is skipped rather than producing an invalid child.
    {
        Genome parentA = makeInterfaceGenome();
        parentA.addNode(NodeGene{50, NodeType::Hidden});
        parentA.addConnection(ConnectionGene{0, 50, 0.5f, true, 20});
        Genome parentB = makeInterfaceGenome();
        parentB.addNode(NodeGene{50, NodeType::Hidden});
        parentB.addConnection(ConnectionGene{0, 50, 0.6f, true, 21}); // same (source, target), different innovation

        CrossoverConfig config;
        for (std::uint32_t seed = 1; seed <= 30; ++seed)
        {
            GenomeCrossover crossover(seed);
            const Genome child = crossover.crossover(parentA, 10.0f, parentB, 10.0f, config); // must never throw
            const ConnectionGene* found = child.findConnection(0, 50);
            assert((found == nullptr || found->getInnovationNumber() == 20 || found->getInnovationNumber() == 21) &&
                   "at most one of the two conflicting candidates may end up in the child");
        }
    }

    // 19: an equal-fitness candidate that would create an enabled cycle is
    // skipped -- a matching 50->51 edge is always present, and a
    // non-matching 51->50 candidate would close a 2-cycle if included.
    {
        Genome parentA = makeInterfaceGenome();
        parentA.addNode(NodeGene{50, NodeType::Hidden});
        parentA.addNode(NodeGene{51, NodeType::Hidden});
        parentA.addConnection(ConnectionGene{50, 51, 0.5f, true, 30}); // matching
        parentA.addConnection(ConnectionGene{51, 50, 0.7f, true, 31}); // A-only, would close a cycle

        Genome parentB = makeInterfaceGenome();
        parentB.addNode(NodeGene{50, NodeType::Hidden});
        parentB.addNode(NodeGene{51, NodeType::Hidden});
        parentB.addConnection(ConnectionGene{50, 51, 0.5f, true, 30}); // matching

        CrossoverConfig config;
        for (std::uint32_t seed = 1; seed <= 30; ++seed)
        {
            GenomeCrossover crossover(seed);
            const Genome child = crossover.crossover(parentA, 10.0f, parentB, 10.0f, config); // equal fitness, must never throw
            assert(!child.hasConnection(51, 50) && "a candidate that would close an enabled cycle must always be skipped");
        }
    }

    // 20, 21 & 22: the required Input/Bias/Output nodes are preserved
    // regardless of connections.
    {
        Genome parentA = makeInterfaceGenome();
        Genome parentB = makeInterfaceGenome();
        GenomeCrossover crossover(1u);
        CrossoverConfig config;
        const Genome child = crossover.crossover(parentA, 1.0f, parentB, 1.0f, config);

        for (int i = 0; i < ai::NeuralNetwork::kInputCount; ++i)
        {
            assert(child.hasNode(i) && child.findNode(i)->getType() == NodeType::Input &&
                   "every Input node must be preserved"); // 20
        }
        assert(child.hasNode(ai::NeuralNetwork::kInputCount) &&
               child.findNode(ai::NeuralNetwork::kInputCount)->getType() == NodeType::Bias &&
               "the Bias node must be preserved"); // 21
        assert(child.hasNode(100) && child.findNode(100)->getType() == NodeType::Output && child.hasNode(101) &&
               child.findNode(101)->getType() == NodeType::Output && child.hasNode(102) &&
               child.findNode(102)->getType() == NodeType::Output && "every Output node must be preserved"); // 22
    }

    // 23: a Hidden node referenced by an inherited connection is preserved.
    {
        Genome parentA = makeInterfaceGenome();
        parentA.addNode(NodeGene{50, NodeType::Hidden});
        parentA.addConnection(ConnectionGene{0, 50, 0.5f, true, 10}); // A-only, references Hidden 50
        Genome parentB = makeInterfaceGenome();

        GenomeCrossover crossover(1u);
        CrossoverConfig config;
        const Genome child = crossover.crossover(parentA, 10.0f, parentB, 5.0f, config); // A fitter
        assert(child.hasNode(50) && child.findNode(50)->getType() == NodeType::Hidden &&
               "a Hidden node referenced by an inherited connection must be preserved");
    }

    // 24: an unreferenced Hidden node need not be preserved.
    {
        Genome parentA = makeInterfaceGenome();
        parentA.addNode(NodeGene{60, NodeType::Hidden}); // never referenced by any connection
        Genome parentB = makeInterfaceGenome();

        GenomeCrossover crossover(1u);
        CrossoverConfig config;
        const Genome child = crossover.crossover(parentA, 10.0f, parentB, 5.0f, config);
        assert(!child.hasNode(60) && "an unreferenced Hidden node need not be preserved");
    }

    // 25: the same NodeId with a conflicting NodeType between parents is
    // rejected.
    {
        Genome parentA = makeInterfaceGenome();
        parentA.addNode(NodeGene{50, NodeType::Hidden});
        Genome parentB = makeInterfaceGenome();
        parentB.addNode(NodeGene{50, NodeType::Output}); // conflicting type for the same ID

        GenomeCrossover crossover(1u);
        CrossoverConfig config;
        assert(throwsInvalidArgument([&]() { crossover.crossover(parentA, 1.0f, parentB, 1.0f, config); }) &&
               "the same NodeId with a conflicting NodeType between parents must be rejected");
    }

    // 26: node IDs are never renumbered, even when large/non-contiguous.
    {
        Genome parentA;
        for (int i = 0; i < ai::NeuralNetwork::kInputCount; ++i)
        {
            parentA.addNode(NodeGene{1000 + i, NodeType::Input});
        }
        parentA.addNode(NodeGene{2000, NodeType::Bias});
        parentA.addNode(NodeGene{3000, NodeType::Output});
        parentA.addNode(NodeGene{3001, NodeType::Output});
        parentA.addNode(NodeGene{3002, NodeType::Output}); // GenomeCrossover::crossover() itself
                                                             // buildPhenotype()-validates its child
        parentA.addConnection(ConnectionGene{1000, 3000, 0.5f, true, 0});

        Genome parentB;
        for (int i = 0; i < ai::NeuralNetwork::kInputCount; ++i)
        {
            parentB.addNode(NodeGene{1000 + i, NodeType::Input});
        }
        parentB.addNode(NodeGene{2000, NodeType::Bias});
        parentB.addNode(NodeGene{3000, NodeType::Output});
        parentB.addNode(NodeGene{3001, NodeType::Output});
        parentB.addNode(NodeGene{3002, NodeType::Output});
        parentB.addConnection(ConnectionGene{1000, 3000, 0.9f, true, 0}); // matching, different weight

        GenomeCrossover crossover(1u);
        CrossoverConfig config;
        const Genome child = crossover.crossover(parentA, 1.0f, parentB, 1.0f, config);
        assert(child.hasNode(1000) && child.hasNode(1008) && child.hasNode(2000) && child.hasNode(3000) &&
               child.hasNode(3001) && "node IDs must be preserved exactly, never renumbered");
        assert(child.findConnection(1000, 3000) != nullptr && child.findConnection(1000, 3000)->getInnovationNumber() == 0 &&
               "connection endpoints must reference the original, non-renumbered node IDs");
    }

    // 27: connection innovation numbers are preserved exactly.
    {
        Genome parentA = makeInterfaceGenome();
        parentA.addConnection(ConnectionGene{0, 100, 0.5f, true, 42});
        Genome parentB = makeInterfaceGenome();
        parentB.addConnection(ConnectionGene{0, 100, 0.6f, true, 42});

        GenomeCrossover crossover(1u);
        CrossoverConfig config;
        const Genome child = crossover.crossover(parentA, 1.0f, parentB, 1.0f, config);
        assert(child.findConnection(0, 100) != nullptr && child.findConnection(0, 100)->getInnovationNumber() == 42 &&
               "connection innovation numbers must be preserved exactly");
    }

    // 28 & 29: child nodes are stored ascending by node ID, and child
    // connections ascending by innovation number, regardless of either
    // parent's own storage order.
    {
        Genome parentA;
        parentA.addNode(NodeGene{ai::NeuralNetwork::kInputCount, NodeType::Bias});
        parentA.addNode(NodeGene{102, NodeType::Output}); // GenomeCrossover::crossover() itself
                                                            // buildPhenotype()-validates its child
        parentA.addNode(NodeGene{101, NodeType::Output});
        parentA.addNode(NodeGene{100, NodeType::Output});
        for (int i = ai::NeuralNetwork::kInputCount - 1; i >= 0; --i)
        {
            parentA.addNode(NodeGene{i, NodeType::Input});
        }
        parentA.addConnection(ConnectionGene{0, 101, 0.1f, true, 9});
        parentA.addConnection(ConnectionGene{1, 100, 0.2f, true, 3});

        Genome parentB = parentA; // identical structure -- every gene matches itself

        GenomeCrossover crossover(1u);
        CrossoverConfig config;
        const Genome child = crossover.crossover(parentA, 1.0f, parentB, 1.0f, config);

        for (std::size_t i = 1; i < child.nodes().size(); ++i)
        {
            assert(child.nodes()[i - 1].getId() < child.nodes()[i].getId() &&
                   "child nodes must be stored ascending by node ID"); // 28
        }
        for (std::size_t i = 1; i < child.connections().size(); ++i)
        {
            assert(child.connections()[i - 1].getInnovationNumber() < child.connections()[i].getInnovationNumber() &&
                   "child connections must be stored ascending by innovation number"); // 29
        }
    }

    // 30: duplicate connection innovation numbers inside parent A are
    // rejected (Genome::validate() alone would not catch this, since the
    // two connections use different (source, target) pairs).
    {
        std::vector<NodeGene> nodes = {NodeGene{0, NodeType::Input}, NodeGene{1, NodeType::Input},
                                        NodeGene{100, NodeType::Output}, NodeGene{101, NodeType::Output}};
        std::vector<ConnectionGene> connections = {ConnectionGene{0, 100, 0.1f, true, 5},
                                                    ConnectionGene{1, 101, 0.2f, true, 5}};
        Genome parentAWithDuplicateInnovation(nodes, connections);
        parentAWithDuplicateInnovation.validate(); // sanity: Genome::validate() itself does not reject this
        Genome parentB;

        GenomeCrossover crossover(1u);
        CrossoverConfig config;
        assert(throwsInvalidArgument(
                   [&]() { crossover.crossover(parentAWithDuplicateInnovation, 1.0f, parentB, 1.0f, config); }) &&
               "parent A with duplicate connection innovation numbers must be rejected");
    }

    // 31: symmetric case for parent B.
    {
        std::vector<NodeGene> nodes = {NodeGene{0, NodeType::Input}, NodeGene{1, NodeType::Input},
                                        NodeGene{100, NodeType::Output}, NodeGene{101, NodeType::Output}};
        std::vector<ConnectionGene> connections = {ConnectionGene{0, 100, 0.1f, true, 5},
                                                    ConnectionGene{1, 101, 0.2f, true, 5}};
        Genome parentBWithDuplicateInnovation(nodes, connections);
        Genome parentA;

        GenomeCrossover crossover(1u);
        CrossoverConfig config;
        assert(throwsInvalidArgument(
                   [&]() { crossover.crossover(parentA, 1.0f, parentBWithDuplicateInnovation, 1.0f, config); }) &&
               "parent B with duplicate connection innovation numbers must be rejected");
    }

    // 32 & 33: child.validate() and buildPhenotype() succeed for a
    // representative mixed scenario (matching genes plus fitter-only
    // genes together, including a Hidden node).
    {
        Genome parentA = makeInterfaceGenome();
        parentA.addNode(NodeGene{50, NodeType::Hidden});
        parentA.addConnection(ConnectionGene{0, 100, 0.5f, true, 0});   // matching
        parentA.addConnection(ConnectionGene{1, 50, 0.3f, true, 10});   // A-only
        parentA.addConnection(ConnectionGene{50, 101, 0.4f, true, 11}); // A-only

        Genome parentB = makeInterfaceGenome();
        parentB.addNode(NodeGene{50, NodeType::Hidden});
        parentB.addConnection(ConnectionGene{0, 100, 0.9f, true, 0}); // matching

        GenomeCrossover crossover(1u);
        CrossoverConfig config;
        const Genome child = crossover.crossover(parentA, 10.0f, parentB, 5.0f, config); // A fitter

        child.validate();                                      // 32: must not throw
        ai::NeuralNetwork net = ai::neat::buildPhenotype(child); // 33: must not throw
        (void)net;
    }

    // 34: the crossover child's phenotype evaluates deterministically.
    {
        Genome parentA = makeInterfaceGenome();
        parentA.addConnection(ConnectionGene{0, 100, 0.5f, true, 0});
        Genome parentB = makeInterfaceGenome();
        parentB.addConnection(ConnectionGene{0, 100, 0.5f, true, 0});

        GenomeCrossover crossover(1u);
        CrossoverConfig config;
        const Genome child = crossover.crossover(parentA, 1.0f, parentB, 1.0f, config);

        ai::NeuralNetwork netA = ai::neat::buildPhenotype(child);
        ai::NeuralNetwork netB = ai::neat::buildPhenotype(child);
        const auto obs = makeObservation(0, 1.0f);
        const auto outA = netA.evaluate(obs);
        const auto outB = netB.evaluate(obs);
        assert(outA[0] == outB[0] && outA[1] == outB[1] && outA[2] == outB[2] &&
               "the phenotype built from a crossover child must evaluate deterministically");
    }

    // 35: neither parent is ever modified by crossover(), even across the
    // fullest path (matching + equal-fitness non-matching genes).
    {
        Genome parentA = makeInterfaceGenome();
        parentA.addNode(NodeGene{50, NodeType::Hidden});
        parentA.addConnection(ConnectionGene{0, 100, 0.5f, true, 0}); // matching
        parentA.addConnection(ConnectionGene{1, 50, 0.3f, true, 10}); // A-only

        Genome parentB = makeInterfaceGenome();
        parentB.addNode(NodeGene{50, NodeType::Hidden});
        parentB.addConnection(ConnectionGene{0, 100, 0.9f, true, 0}); // matching, different weight
        parentB.addConnection(ConnectionGene{2, 50, 0.4f, true, 11}); // B-only

        const std::vector<NodeGene> nodesABefore = parentA.nodes();
        const std::vector<ConnectionGene> connectionsABefore = parentA.connections();
        const std::vector<NodeGene> nodesBBefore = parentB.nodes();
        const std::vector<ConnectionGene> connectionsBBefore = parentB.connections();

        GenomeCrossover crossover(1u);
        CrossoverConfig config;
        crossover.crossover(parentA, 10.0f, parentB, 10.0f, config); // equal fitness

        assert(parentA.nodes().size() == nodesABefore.size() && parentA.connections().size() == connectionsABefore.size() &&
               "crossover must not add or remove parent A's genes");
        for (std::size_t i = 0; i < nodesABefore.size(); ++i)
        {
            assert(parentA.nodes()[i] == nodesABefore[i] && "crossover must not modify parent A's nodes");
        }
        for (std::size_t i = 0; i < connectionsABefore.size(); ++i)
        {
            const ConnectionGene& before = connectionsABefore[i];
            const ConnectionGene& after = parentA.connections()[i];
            assert(after.getSourceId() == before.getSourceId() && after.getTargetId() == before.getTargetId() &&
                   after.getWeight() == before.getWeight() && after.isEnabled() == before.isEnabled() &&
                   after.getInnovationNumber() == before.getInnovationNumber() &&
                   "crossover must not modify parent A's connections");
        }

        assert(parentB.nodes().size() == nodesBBefore.size() && parentB.connections().size() == connectionsBBefore.size() &&
               "crossover must not add or remove parent B's genes");
        for (std::size_t i = 0; i < nodesBBefore.size(); ++i)
        {
            assert(parentB.nodes()[i] == nodesBBefore[i] && "crossover must not modify parent B's nodes");
        }
        for (std::size_t i = 0; i < connectionsBBefore.size(); ++i)
        {
            const ConnectionGene& before = connectionsBBefore[i];
            const ConnectionGene& after = parentB.connections()[i];
            assert(after.getSourceId() == before.getSourceId() && after.getTargetId() == before.getTargetId() &&
                   after.getWeight() == before.getWeight() && after.isEnabled() == before.isEnabled() &&
                   after.getInnovationNumber() == before.getInnovationNumber() &&
                   "crossover must not modify parent B's connections");
        }
    }

    // 36: the same seed, parents, fitnesses, and config produce an
    // identical child.
    {
        Genome parentA = makeInterfaceGenome();
        parentA.addConnection(ConnectionGene{0, 100, 0.5f, true, 0});
        parentA.addConnection(ConnectionGene{1, 100, 0.3f, true, 5});
        Genome parentB = makeInterfaceGenome();
        parentB.addConnection(ConnectionGene{0, 100, 0.9f, true, 0}); // matching, different weight
        parentB.addConnection(ConnectionGene{2, 101, 0.4f, true, 6}); // B-only

        CrossoverConfig config;

        GenomeCrossover crossoverX(777u);
        const Genome childX = crossoverX.crossover(parentA, 10.0f, parentB, 10.0f, config); // equal fitness

        GenomeCrossover crossoverY(777u);
        const Genome childY = crossoverY.crossover(parentA, 10.0f, parentB, 10.0f, config);

        assert(childX.nodes().size() == childY.nodes().size() &&
               childX.connections().size() == childY.connections().size() &&
               "identical seed/parents/fitness/config must produce a structurally identical child");
        for (std::size_t i = 0; i < childX.nodes().size(); ++i)
        {
            assert(childX.nodes()[i] == childY.nodes()[i] && "identical runs must produce identical child nodes");
        }
        for (std::size_t i = 0; i < childX.connections().size(); ++i)
        {
            const ConnectionGene& x = childX.connections()[i];
            const ConnectionGene& y = childY.connections()[i];
            assert(x.getSourceId() == y.getSourceId() && x.getTargetId() == y.getTargetId() &&
                   x.getWeight() == y.getWeight() && x.isEnabled() == y.isEnabled() &&
                   x.getInnovationNumber() == y.getInnovationNumber() &&
                   "identical runs must produce identical child connections");
        }
    }

    // 37: different seeds can select a different parent's copy for a
    // matching gene.
    {
        Genome parentA = makeInterfaceGenome();
        parentA.addConnection(ConnectionGene{0, 100, 0.11f, true, 0});
        Genome parentB = makeInterfaceGenome();
        parentB.addConnection(ConnectionGene{0, 100, 0.99f, true, 0});

        CrossoverConfig config; // default matchingGeneChooseParentAProbability = 0.5

        std::vector<float> weightsSeen;
        for (std::uint32_t seed = 1; seed <= 10; ++seed)
        {
            GenomeCrossover crossover(seed);
            const Genome child = crossover.crossover(parentA, 1.0f, parentB, 1.0f, config);
            weightsSeen.push_back(child.findConnection(0, 100)->getWeight());
        }
        const bool sawVariation =
            std::any_of(weightsSeen.begin(), weightsSeen.end(), [&](float w) { return w != weightsSeen.front(); });
        assert(sawVariation && "different seeds must be capable of selecting a different parent's copy for a matching gene");
    }

    // 38: repeated calls on the same GenomeCrossover instance advance its
    // RNG state.
    {
        Genome parentA = makeInterfaceGenome();
        parentA.addConnection(ConnectionGene{0, 100, 0.11f, true, 0});
        Genome parentB = makeInterfaceGenome();
        parentB.addConnection(ConnectionGene{0, 100, 0.99f, true, 0});

        CrossoverConfig config;
        GenomeCrossover crossover(55u);

        std::vector<float> weightsSeen;
        for (int i = 0; i < 10; ++i)
        {
            const Genome child = crossover.crossover(parentA, 1.0f, parentB, 1.0f, config);
            weightsSeen.push_back(child.findConnection(0, 100)->getWeight());
        }
        const bool sawVariation =
            std::any_of(weightsSeen.begin(), weightsSeen.end(), [&](float w) { return w != weightsSeen.front(); });
        assert(sawVariation && "repeated calls from the same GenomeCrossover instance must advance its RNG state");
    }

    // 39: no InnovationTracker state involved -- crossover()'s signature
    // takes no tracker parameter.
    // 40: no mutation is performed (reinforced by 35 -- both parents unchanged).
    // 41: no population/species logic -- crossover() takes only two Genomes and two fitness floats.
    // 42: all previous verification suites still pass.

    TraceLog(LOG_INFO, "Genome crossover verification: all deterministic checks passed");
}

// Regression test for the crossover cycle bug (see GenomeCrossover.cpp's
// class/function comments): two parent genomes that are each individually
// acyclic can combine, via crossover, into a child with an enabled cycle --
// the standard NEAT crossover pitfall. Two independent minimal reproducers,
// one per previously-unprotected code path (a real production run hit this
// at ~generation 160, almost certainly via path 2, the far more common
// unequal-fitness case):
//
//   1. Two MATCHING genes (same innovation number, present in both
//      parents) where each parent has exactly one of the pair enabled --
//      crossover's probabilistic re-enable of a disabled matching gene can
//      enable BOTH in the child even though neither parent ever had both
//      enabled at once.
//   2. A matching gene re-enabled by crossover (as in 1) combined with the
//      FITTER parent's own exclusive (aOnly) gene, inherited unconditionally
//      before this fix -- neither parent had both edges enabled
//      simultaneously, but the child did.
void verifyGenomeCrossoverCycleSafety()
{
    using namespace genome_crossover_verify;
    using ai::neat::ConnectionGene;
    using ai::neat::CrossoverConfig;
    using ai::neat::Genome;
    using ai::neat::GenomeCrossover;
    using ai::neat::NodeGene;
    using ai::neat::NodeType;

    // Scenario 1: matching-gene re-enable path.
    //
    // Parent A: hidden 103->104 ENABLED, hidden 104->103 disabled -- alone,
    // acyclic (single live edge).
    // Parent B: hidden 103->104 disabled, hidden 104->103 ENABLED -- alone,
    // also acyclic (single live edge, the opposite one).
    // Both also share one ordinary, always-enabled-in-both matching gene
    // (Bias->Steering) to confirm the fix doesn't touch unrelated genes.
    {
        Genome parentA = makeInterfaceGenome();
        parentA.addNode(NodeGene{103, NodeType::Hidden});
        parentA.addNode(NodeGene{104, NodeType::Hidden});
        parentA.addConnection(ConnectionGene{ai::NeuralNetwork::kInputCount, 100, 0.42f, true, 0}); // ordinary matching gene, both enabled
        parentA.addConnection(ConnectionGene{103, 104, 0.5f, true, 10});  // enabled in A only
        parentA.addConnection(ConnectionGene{104, 103, 0.5f, false, 11}); // disabled in A

        Genome parentB = makeInterfaceGenome();
        parentB.addNode(NodeGene{103, NodeType::Hidden});
        parentB.addNode(NodeGene{104, NodeType::Hidden});
        parentB.addConnection(ConnectionGene{ai::NeuralNetwork::kInputCount, 100, 0.42f, true, 0}); // same ordinary matching gene
        parentB.addConnection(ConnectionGene{103, 104, 0.7f, false, 10}); // disabled in B
        parentB.addConnection(ConnectionGene{104, 103, 0.7f, true, 11}); // enabled in B only

        // 1: both parents individually build a valid phenotype (each is
        // genuinely, independently acyclic).
        (void)ai::neat::buildPhenotype(parentA);
        (void)ai::neat::buildPhenotype(parentB);

        // 2: confirms the scenario is a genuine cycle, independent of
        // trusting the fix -- a genome with BOTH edges enabled (what the
        // pre-fix code would have unconditionally produced here, since
        // disabledGeneRemainDisabledProbability=0 below forces every
        // disabled-in-one-parent matching gene to be re-enabled) is invalid.
        {
            Genome bothEnabled = makeInterfaceGenome();
            bothEnabled.addNode(NodeGene{103, NodeType::Hidden});
            bothEnabled.addNode(NodeGene{104, NodeType::Hidden});
            bothEnabled.addConnection(ConnectionGene{103, 104, 0.5f, true, 10});
            bothEnabled.addConnection(ConnectionGene{104, 103, 0.5f, true, 11});
            bool threw = false;
            try
            {
                (void)ai::neat::buildPhenotype(bothEnabled);
            }
            catch (const std::invalid_argument&)
            {
                threw = true;
            }
            assert(threw && "setup: enabling both 103->104 and 104->103 must itself be a genuine cycle"); // 2
        }

        // 3 & 4: the FIXED crossover, with disabledGeneRemainDisabledProbability=0
        // (deterministically re-enable every disabled-in-one-parent matching
        // gene -- the worst case for triggering this bug), must produce an
        // acyclic child whose phenotype builds successfully.
        CrossoverConfig config;
        config.matchingGeneChooseParentAProbability = 0.5f;
        config.disabledGeneRemainDisabledProbability = 0.0f;
        GenomeCrossover crossover(4242u);
        const Genome child = crossover.crossover(parentA, 1.0f, parentB, 1.0f, config); // equal fitness -- exercises the matching-gene path regardless
        child.validate();
        (void)ai::neat::buildPhenotype(child); // 3 & 4: must not throw

        // 5: exactly one of the two matching genes ends up enabled -- never
        // both (would be a cycle) and never neither (a matching gene is
        // never dropped, only forced disabled -- see 6/7 below). Processing
        // order is ascending innovation number, so 103->104 (10) is
        // resolved before 104->103 (11): 10 keeps its re-enabled state,
        // and 11's re-enable is deterministically overridden back to
        // disabled because it would close the cycle 10 just opened.
        const ConnectionGene* geneTen = child.findConnection(103, 104);
        const ConnectionGene* geneEleven = child.findConnection(104, 103);
        assert(geneTen != nullptr && geneEleven != nullptr &&
               "6 & 7: both matching genes must still be present in the child (never dropped)");
        assert(geneTen->isEnabled() && !geneEleven->isEnabled() &&
               "5: exactly the first-processed (lowest-innovation) edge must end up enabled, the other forced "
               "disabled -- deterministically, not both, not neither");

        // 7 (continued): the unrelated, always-enabled-in-both matching
        // gene must be completely unaffected.
        const ConnectionGene* unrelated = child.findConnection(ai::NeuralNetwork::kInputCount, 100);
        assert(unrelated != nullptr && unrelated->isEnabled() &&
               "an ordinary matching gene unrelated to the cycle must be inherited normally, untouched");
    }

    // Scenario 2: matching-gene re-enable COMBINED with the fitter parent's
    // own exclusive (aOnly) gene -- the far more common unequal-fitness
    // crossover path, and the one with NO cycle protection at all before
    // this fix.
    //
    // Parent A (fitter): matching gene 104->103 disabled; exclusive
    // (aOnly) gene 103->104 enabled. Alone: acyclic (only 103->104 live).
    // Parent B (less fit): matching gene 104->103 enabled; does not have
    // the 103->104 innovation at all. Alone: acyclic (only 104->103 live).
    {
        Genome parentA = makeInterfaceGenome();
        parentA.addNode(NodeGene{103, NodeType::Hidden});
        parentA.addNode(NodeGene{104, NodeType::Hidden});
        parentA.addConnection(ConnectionGene{104, 103, 0.3f, false, 10}); // matching, disabled in A
        parentA.addConnection(ConnectionGene{103, 104, 0.4f, true, 11});  // aOnly (A's exclusive gene), enabled

        Genome parentB = makeInterfaceGenome();
        parentB.addNode(NodeGene{103, NodeType::Hidden});
        parentB.addNode(NodeGene{104, NodeType::Hidden});
        parentB.addConnection(ConnectionGene{104, 103, 0.6f, true, 10}); // matching, enabled in B

        (void)ai::neat::buildPhenotype(parentA); // 1 (scenario 2)
        (void)ai::neat::buildPhenotype(parentB);

        CrossoverConfig config;
        config.matchingGeneChooseParentAProbability = 0.5f;
        config.disabledGeneRemainDisabledProbability = 0.0f; // deterministically re-enable the matching gene
        GenomeCrossover crossover(1337u);
        // A is strictly fitter -- exercises the previously-unprotected
        // fitterIsA/aOnly branch, not the equal-fitness merge.
        const Genome child = crossover.crossover(parentA, 2.0f, parentB, 1.0f, config);
        child.validate();
        (void)ai::neat::buildPhenotype(child); // 3 & 4 (scenario 2): must not throw

        const ConnectionGene* matchingGene = child.findConnection(104, 103);
        const ConnectionGene* exclusiveGene = child.findConnection(103, 104);
        assert(matchingGene != nullptr && matchingGene->isEnabled() &&
               "the matching gene (re-enabled deterministically) must be present and enabled in the child");
        // 6 & 7 (scenario 2): the exclusive gene, unlike a matching gene,
        // is optional structure -- the fitter parent's own convention this
        // file already used for equal-fitness merges (skip, don't repair)
        // extends here: it is dropped rather than force-disabled, since
        // (unlike a matching gene) it has no shared ancestry in parent B to
        // preserve continuity for.
        assert(exclusiveGene == nullptr &&
               "the fitter parent's cycle-completing exclusive gene must be skipped, not force-added, when it "
               "would close a cycle against an already-inherited matching gene");
    }

    TraceLog(LOG_INFO, "Genome crossover cycle-safety verification: all deterministic checks passed");
}

} // namespace verification
