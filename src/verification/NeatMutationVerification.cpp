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


namespace genome_mutator_verify
{

// 9 Input + 1 Bias + 2 Output nodes plus four connections with known
// weights, one disabled (so mutation of disabled genes can be checked).
ai::neat::Genome makeTestGenome()
{
    using ai::neat::ConnectionGene;
    using ai::neat::Genome;
    using ai::neat::NodeGene;
    using ai::neat::NodeType;

    Genome genome;
    for (int i = 0; i < ai::NeuralNetwork::kInputCount; ++i)
    {
        genome.addNode(NodeGene{i, NodeType::Input});
    }
    genome.addNode(NodeGene{9, NodeType::Bias});
    genome.addNode(NodeGene{100, NodeType::Output});
    genome.addNode(NodeGene{101, NodeType::Output});

    genome.addConnection(ConnectionGene{0, 100, 0.5f, true, 0});
    genome.addConnection(ConnectionGene{1, 100, -0.3f, true, 1});
    genome.addConnection(ConnectionGene{9, 101, 0.2f, false, 2}); // disabled on purpose
    genome.addConnection(ConnectionGene{2, 101, 0.9f, true, 3});
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

// Genome with only two Output nodes: Output is never a valid add-connection
// source, so no source candidate can ever exist.
ai::neat::Genome makeOnlyOutputsGenome()
{
    using ai::neat::Genome;
    using ai::neat::NodeGene;
    using ai::neat::NodeType;

    Genome genome;
    genome.addNode(NodeGene{0, NodeType::Output});
    genome.addNode(NodeGene{1, NodeType::Output});
    return genome;
}

// Genome with only two Input nodes: Input is never a valid add-connection
// target, so no target candidate can ever exist.
ai::neat::Genome makeOnlyInputsGenome()
{
    using ai::neat::Genome;
    using ai::neat::NodeGene;
    using ai::neat::NodeType;

    Genome genome;
    genome.addNode(NodeGene{0, NodeType::Input});
    genome.addNode(NodeGene{1, NodeType::Input});
    return genome;
}

// Genome with only two Bias nodes: like makeOnlyInputsGenome(), Bias is
// never a valid target, so no target candidate can ever exist.
ai::neat::Genome makeOnlyBiasGenome()
{
    using ai::neat::Genome;
    using ai::neat::NodeGene;
    using ai::neat::NodeType;

    Genome genome;
    genome.addNode(NodeGene{0, NodeType::Bias});
    genome.addNode(NodeGene{1, NodeType::Bias});
    return genome;
}

// The only structurally possible pair is Input(0) -> Output(1): proves
// Input can be a source and Output can be a target.
ai::neat::Genome makeInputToOutputGenome()
{
    using ai::neat::Genome;
    using ai::neat::NodeGene;
    using ai::neat::NodeType;

    Genome genome;
    genome.addNode(NodeGene{0, NodeType::Input});
    genome.addNode(NodeGene{1, NodeType::Output});
    return genome;
}

// The only structurally possible pair is Bias(0) -> Output(1): proves Bias
// can be a source.
ai::neat::Genome makeBiasToOutputGenome()
{
    using ai::neat::Genome;
    using ai::neat::NodeGene;
    using ai::neat::NodeType;

    Genome genome;
    genome.addNode(NodeGene{0, NodeType::Bias});
    genome.addNode(NodeGene{1, NodeType::Output});
    return genome;
}

// The only structurally possible pair is Hidden(0) -> Hidden(1): proves
// Hidden can be both a source and a target.
ai::neat::Genome makeHiddenToHiddenGenome()
{
    using ai::neat::Genome;
    using ai::neat::NodeGene;
    using ai::neat::NodeType;

    Genome genome;
    genome.addNode(NodeGene{0, NodeType::Hidden});
    genome.addNode(NodeGene{1, NodeType::Hidden});
    return genome;
}

// A single Hidden node: the only "candidate" pair is a self-connection,
// which must never be added.
ai::neat::Genome makeSingleHiddenGenome()
{
    using ai::neat::Genome;
    using ai::neat::NodeGene;
    using ai::neat::NodeType;

    Genome genome;
    genome.addNode(NodeGene{0, NodeType::Hidden});
    return genome;
}

// Genome with exactly one structurally possible pair, already connected:
// proves a duplicate directed connection (enabled or disabled) is never
// re-added.
ai::neat::Genome makeAlreadyConnectedGenome(bool enabled)
{
    using ai::neat::ConnectionGene;
    using ai::neat::Genome;
    using ai::neat::NodeGene;
    using ai::neat::NodeType;

    Genome genome;
    genome.addNode(NodeGene{0, NodeType::Input});
    genome.addNode(NodeGene{1, NodeType::Output});
    genome.addConnection(ConnectionGene{0, 1, 0.1f, enabled, 0});
    return genome;
}

// Descending-ID edge 10 -> 3 (so ascending-ID assumptions can't sneak in).
// With existingEdgeEnabled true, the reverse 3 -> 10 closes a 2-cycle and
// must be rejected, leaving no valid pair. With it false, 10 -> 3 no
// longer counts toward the enabled graph, so 3 -> 10 must be accepted --
// proving disabled edges don't participate in cycle detection.
ai::neat::Genome makeCycleGenome(bool existingEdgeEnabled)
{
    using ai::neat::ConnectionGene;
    using ai::neat::Genome;
    using ai::neat::NodeGene;
    using ai::neat::NodeType;

    Genome genome;
    genome.addNode(NodeGene{10, NodeType::Hidden});
    genome.addNode(NodeGene{3, NodeType::Hidden});
    genome.addConnection(ConnectionGene{10, 3, 0.1f, existingEdgeEnabled, 0});
    return genome;
}

// Every valid (source, target) pair connected except Bias(1) -> Output(3):
// exercises the deterministic exhaustive fallback, since random attempts
// are unlikely to land on the single remaining pair.
ai::neat::Genome makeSparseValidPairGenome()
{
    using ai::neat::ConnectionGene;
    using ai::neat::Genome;
    using ai::neat::NodeGene;
    using ai::neat::NodeType;

    Genome genome;
    genome.addNode(NodeGene{0, NodeType::Input});
    genome.addNode(NodeGene{1, NodeType::Bias});
    genome.addNode(NodeGene{2, NodeType::Hidden});
    genome.addNode(NodeGene{3, NodeType::Output});
    genome.addConnection(ConnectionGene{0, 2, 0.1f, true, 0});
    genome.addConnection(ConnectionGene{0, 3, 0.1f, true, 1});
    genome.addConnection(ConnectionGene{1, 2, 0.1f, true, 2});
    genome.addConnection(ConnectionGene{2, 3, 0.1f, true, 3});
    return genome;
}

// Same as makeSparseValidPairGenome() but with Bias(1) -> Output(3) also
// connected -- every valid pair exists, fully saturated.
ai::neat::Genome makeSaturatedGenome()
{
    using ai::neat::ConnectionGene;
    using ai::neat::Genome;
    using ai::neat::NodeGene;
    using ai::neat::NodeType;

    Genome genome;
    genome.addNode(NodeGene{0, NodeType::Input});
    genome.addNode(NodeGene{1, NodeType::Bias});
    genome.addNode(NodeGene{2, NodeType::Hidden});
    genome.addNode(NodeGene{3, NodeType::Output});
    genome.addConnection(ConnectionGene{0, 2, 0.1f, true, 0});
    genome.addConnection(ConnectionGene{0, 3, 0.1f, true, 1});
    genome.addConnection(ConnectionGene{1, 2, 0.1f, true, 2});
    genome.addConnection(ConnectionGene{1, 3, 0.1f, true, 3});
    genome.addConnection(ConnectionGene{2, 3, 0.1f, true, 4});
    return genome;
}

} // namespace genome_mutator_verify

// Deterministic check of ai::neat::GenomeMutator's connection-weight
// mutation, end to end. No structural mutation/crossover/species/population here.
void verifyGenomeMutator()
{
    using namespace genome_mutator_verify;
    using ai::neat::ConnectionGene;
    using ai::neat::Genome;
    using ai::neat::GenomeMutator;
    using ai::neat::MutationConfig;
    using ai::neat::NodeGene;
    constexpr float kEps = 1e-5f;

    // 1: zero mutation probability changes no weights.
    {
        Genome genome = makeTestGenome();
        const std::vector<ConnectionGene> before = genome.connections();
        MutationConfig config;
        config.weightMutationProbability = 0.0f;
        GenomeMutator mutator(12345u);
        mutator.mutateWeights(genome, config);
        for (std::size_t i = 0; i < before.size(); ++i)
        {
            assert(genome.connections()[i].getWeight() == before[i].getWeight() &&
                   "zero mutation probability must leave every weight unchanged");
        }
    }

    // 2: probability 1 selects every connection -- every weight must change
    // (colliding exactly with the original value is astronomically unlikely).
    {
        Genome genome = makeTestGenome();
        const std::vector<ConnectionGene> before = genome.connections();
        MutationConfig config;
        config.weightMutationProbability = 1.0f;
        GenomeMutator mutator(1u);
        mutator.mutateWeights(genome, config);
        for (std::size_t i = 0; i < before.size(); ++i)
        {
            assert(genome.connections()[i].getWeight() != before[i].getWeight() &&
                   "mutation probability 1 must select and change every connection's weight");
        }
    }

    // 3: perturb probability 1 performs perturbation only -- every changed
    // weight stays within oldWeight +/- perturbStrength.
    {
        Genome genome = makeTestGenome();
        const std::vector<ConnectionGene> before = genome.connections();
        MutationConfig config;
        config.weightMutationProbability = 1.0f;
        config.weightPerturbProbability = 1.0f;
        config.perturbStrength = 0.3f;
        GenomeMutator mutator(7u);
        mutator.mutateWeights(genome, config);
        for (std::size_t i = 0; i < before.size(); ++i)
        {
            const float delta = genome.connections()[i].getWeight() - before[i].getWeight();
            assert(std::fabs(delta) <= config.perturbStrength + kEps &&
                   "perturb probability 1 must only ever perturb, never replace outright");
        }
    }

    // 4: perturb probability 0 performs replacement only -- every changed
    // weight lands within [replacementWeightMin, replacementWeightMax].
    {
        Genome genome = makeTestGenome();
        MutationConfig config;
        config.weightMutationProbability = 1.0f;
        config.weightPerturbProbability = 0.0f;
        config.replacementWeightMin = -2.0f;
        config.replacementWeightMax = 2.0f;
        GenomeMutator mutator(9u);
        mutator.mutateWeights(genome, config);
        for (const ConnectionGene& c : genome.connections())
        {
            assert(c.getWeight() >= config.replacementWeightMin && c.getWeight() <= config.replacementWeightMax &&
                   "perturb probability 0 must always replace within the configured range");
        }
    }

    // 5: zero perturb strength preserves the selected weight exactly (a
    // perturbation by +/-0 is a no-op).
    {
        Genome genome = makeTestGenome();
        const std::vector<ConnectionGene> before = genome.connections();
        MutationConfig config;
        config.weightMutationProbability = 1.0f;
        config.weightPerturbProbability = 1.0f;
        config.perturbStrength = 0.0f;
        GenomeMutator mutator(3u);
        mutator.mutateWeights(genome, config);
        for (std::size_t i = 0; i < before.size(); ++i)
        {
            assert(genome.connections()[i].getWeight() == before[i].getWeight() &&
                   "zero perturb strength must preserve the selected weight exactly");
        }
    }

    // 6: replacement weights remain within the configured range, checked
    // across many seeds for confidence.
    {
        MutationConfig config;
        config.weightMutationProbability = 1.0f;
        config.weightPerturbProbability = 0.0f;
        config.replacementWeightMin = -0.75f;
        config.replacementWeightMax = 1.25f;
        for (std::uint32_t seed = 0; seed < 20; ++seed)
        {
            Genome genome = makeTestGenome();
            GenomeMutator mutator(seed);
            mutator.mutateWeights(genome, config);
            for (const ConnectionGene& c : genome.connections())
            {
                assert(c.getWeight() >= config.replacementWeightMin && c.getWeight() <= config.replacementWeightMax &&
                       "replacement weights must always stay within the configured range");
            }
        }
    }

    // 7: a disabled connection's weight is also eligible for mutation.
    {
        Genome genome = makeTestGenome();
        const ConnectionGene* disabledBefore = genome.findConnection(9, 101);
        assert(disabledBefore != nullptr && !disabledBefore->isEnabled() &&
               "test genome must contain a disabled connection");
        const float disabledWeightBefore = disabledBefore->getWeight();

        MutationConfig config;
        config.weightMutationProbability = 1.0f;
        GenomeMutator mutator(11u);
        mutator.mutateWeights(genome, config);

        const ConnectionGene* disabledAfter = genome.findConnection(9, 101);
        assert(disabledAfter->getWeight() != disabledWeightBefore &&
               "a disabled connection's weight must still be eligible for mutation");
        assert(!disabledAfter->isEnabled() && "mutating weights must not change the enabled state");
    }

    // 8, 9, 10, 11 & 12: enabled states, node genes, connection endpoints,
    // innovation numbers, and node/connection counts all remain unchanged.
    {
        Genome genome = makeTestGenome();
        const std::vector<NodeGene> nodesBefore = genome.nodes();
        const std::vector<ConnectionGene> connectionsBefore = genome.connections();

        MutationConfig config;
        config.weightMutationProbability = 1.0f;
        GenomeMutator mutator(21u);
        mutator.mutateWeights(genome, config);

        assert(genome.nodes().size() == nodesBefore.size() && "mutation must not change node count");
        assert(genome.connections().size() == connectionsBefore.size() && "mutation must not change connection count");

        for (std::size_t i = 0; i < nodesBefore.size(); ++i)
        {
            assert(genome.nodes()[i] == nodesBefore[i] && "mutation must not change any node gene");
        }
        for (std::size_t i = 0; i < connectionsBefore.size(); ++i)
        {
            const ConnectionGene& before = connectionsBefore[i];
            const ConnectionGene& after = genome.connections()[i];
            assert(after.getSourceId() == before.getSourceId() && after.getTargetId() == before.getTargetId() &&
                   "mutation must not change connection endpoints");
            assert(after.getInnovationNumber() == before.getInnovationNumber() &&
                   "mutation must not change innovation numbers");
            assert(after.isEnabled() == before.isEnabled() && "mutation must not change enabled state");
        }
    }

    // 13: Genome::validate() succeeds after mutation.
    {
        Genome genome = makeTestGenome();
        MutationConfig config;
        config.weightMutationProbability = 1.0f;
        GenomeMutator mutator(33u);
        mutator.mutateWeights(genome, config);
        genome.validate(); // must not throw
    }

    // 14: identical seed + identical genome produces identical results.
    {
        Genome genomeA = makeTestGenome();
        Genome genomeB = makeTestGenome();
        MutationConfig config;
        GenomeMutator mutatorA(555u);
        GenomeMutator mutatorB(555u);
        mutatorA.mutateWeights(genomeA, config);
        mutatorB.mutateWeights(genomeB, config);
        for (std::size_t i = 0; i < genomeA.connections().size(); ++i)
        {
            assert(genomeA.connections()[i].getWeight() == genomeB.connections()[i].getWeight() &&
                   "identical seed and genome must produce identical mutation results");
        }
    }

    // 15: different seeds can produce different results.
    {
        Genome genomeA = makeTestGenome();
        Genome genomeB = makeTestGenome();
        MutationConfig config;
        GenomeMutator mutatorA(1u);
        GenomeMutator mutatorB(2u);
        mutatorA.mutateWeights(genomeA, config);
        mutatorB.mutateWeights(genomeB, config);

        bool anyDifferent = false;
        for (std::size_t i = 0; i < genomeA.connections().size(); ++i)
        {
            if (genomeA.connections()[i].getWeight() != genomeB.connections()[i].getWeight())
            {
                anyDifferent = true;
                break;
            }
        }
        assert(anyDifferent && "different seeds must be capable of producing different results");
    }

    // 16: repeated mutations advance the owned RNG state -- must not repeat the same draws.
    {
        Genome genomeA = makeTestGenome();
        Genome genomeB = makeTestGenome();
        MutationConfig config;
        GenomeMutator mutator(77u);
        mutator.mutateWeights(genomeA, config);
        mutator.mutateWeights(genomeB, config);

        bool anyDifferent = false;
        for (std::size_t i = 0; i < genomeA.connections().size(); ++i)
        {
            if (genomeA.connections()[i].getWeight() != genomeB.connections()[i].getWeight())
            {
                anyDifferent = true;
                break;
            }
        }
        assert(anyDifferent &&
               "repeated mutations from the same mutator must advance its RNG state, not repeat the same draws");
    }

    // 17: invalid probabilities are rejected.
    {
        Genome genome = makeTestGenome();
        GenomeMutator mutator(1u);

        MutationConfig tooHigh;
        tooHigh.weightMutationProbability = 1.5f;
        assert(throwsInvalidArgument([&]() { mutator.mutateWeights(genome, tooHigh); }) &&
               "a weightMutationProbability outside [0,1] must be rejected");

        MutationConfig tooLow;
        tooLow.weightPerturbProbability = -0.1f;
        assert(throwsInvalidArgument([&]() { mutator.mutateWeights(genome, tooLow); }) &&
               "a weightPerturbProbability outside [0,1] must be rejected");
    }

    // 18: negative perturb strength is rejected.
    {
        Genome genome = makeTestGenome();
        GenomeMutator mutator(1u);
        MutationConfig config;
        config.perturbStrength = -0.01f;
        assert(throwsInvalidArgument([&]() { mutator.mutateWeights(genome, config); }) &&
               "a negative perturbStrength must be rejected");
    }

    // 19: inverted replacement range is rejected.
    {
        Genome genome = makeTestGenome();
        GenomeMutator mutator(1u);
        MutationConfig config;
        config.replacementWeightMin = 1.0f;
        config.replacementWeightMax = -1.0f;
        assert(throwsInvalidArgument([&]() { mutator.mutateWeights(genome, config); }) &&
               "an inverted replacement range must be rejected");
    }

    // 20: non-finite values are rejected.
    {
        Genome genome = makeTestGenome();
        GenomeMutator mutator(1u);

        MutationConfig nanConfig;
        nanConfig.weightMutationProbability = std::numeric_limits<float>::quiet_NaN();
        assert(throwsInvalidArgument([&]() { mutator.mutateWeights(genome, nanConfig); }) &&
               "a NaN probability must be rejected");

        MutationConfig infConfig;
        infConfig.perturbStrength = std::numeric_limits<float>::infinity();
        assert(throwsInvalidArgument([&]() { mutator.mutateWeights(genome, infConfig); }) &&
               "an infinite perturbStrength must be rejected");
    }

    // 21: empty Genome mutation succeeds without error.
    {
        Genome empty;
        MutationConfig config;
        GenomeMutator mutator(1u);
        mutator.mutateWeights(empty, config); // must not throw
        assert(empty.connections().empty() && empty.nodes().empty() && "an empty genome must remain empty after mutation");
    }

    // 22 & 23: phenotype can still be built after mutation, and changed
    // weights produce a deterministic changed phenotype output.
    {
        Genome genome = makeTestGenome();
        ai::NeuralNetwork before = ai::neat::buildPhenotype(genome);

        MutationConfig config;
        config.weightMutationProbability = 1.0f;
        GenomeMutator mutator(999u);
        mutator.mutateWeights(genome, config);

        ai::NeuralNetwork after = ai::neat::buildPhenotype(genome); // must not throw

        ai::Observation obs;
        obs.values.fill(0.5f);
        const auto outBefore = before.evaluate(obs);
        const auto outAfter = after.evaluate(obs);
        assert((outBefore[0] != outAfter[0] || outBefore[1] != outAfter[1]) &&
               "mutated weights must produce a changed phenotype output");

        // Determinism: the same mutation from the same seed on a fresh
        // identical genome reproduces the same phenotype output.
        Genome genomeRepeat = makeTestGenome();
        GenomeMutator mutatorRepeat(999u);
        mutatorRepeat.mutateWeights(genomeRepeat, config);
        ai::NeuralNetwork afterRepeat = ai::neat::buildPhenotype(genomeRepeat);
        const auto outAfterRepeat = afterRepeat.evaluate(obs);
        assert(outAfter[0] == outAfterRepeat[0] && outAfter[1] == outAfterRepeat[1] &&
               "identical seed and genome must produce a deterministic phenotype output after mutation");
    }

    // 24: no structural genes are added or removed, re-confirmed across a
    // batch of different seeds.
    {
        for (std::uint32_t seed = 100; seed < 110; ++seed)
        {
            Genome genome = makeTestGenome();
            const std::size_t nodeCountBefore = genome.nodes().size();
            const std::size_t connectionCountBefore = genome.connections().size();
            MutationConfig config;
            config.weightMutationProbability = 1.0f;
            GenomeMutator mutator(seed);
            mutator.mutateWeights(genome, config);
            assert(genome.nodes().size() == nodeCountBefore && genome.connections().size() == connectionCountBefore &&
                   "no structural genes may be added or removed by weight mutation");
        }
    }

    // 25: all previous verification suites still pass -- enforced by main()
    // continuing to call every earlier verify*() function unchanged.

    TraceLog(LOG_INFO, "Genome mutator verification: all deterministic checks passed");
}

// Deterministic check of ai::neat::GenomeMutator::mutateAddConnection().
// No add-node/crossover/species/population logic here.
void verifyAddConnectionMutation()
{
    using namespace genome_mutator_verify;
    using ai::neat::ConnectionGene;
    using ai::neat::Genome;
    using ai::neat::GenomeMutator;
    using ai::neat::InnovationNumber;
    using ai::neat::InnovationTracker;
    using ai::neat::MutationConfig;
    using ai::neat::NodeGene;

    // 1: probability 0 returns false and changes nothing.
    {
        Genome genome = makeTestGenome();
        const std::vector<NodeGene> nodesBefore = genome.nodes();
        const std::vector<ConnectionGene> connectionsBefore = genome.connections();
        InnovationTracker tracker(200, 300);
        MutationConfig config;
        config.addConnectionProbability = 0.0f;
        GenomeMutator mutator(1u);

        const bool added = mutator.mutateAddConnection(genome, tracker, config);
        assert(!added && "probability 0 must never add a connection");
        assert(genome.nodes().size() == nodesBefore.size() && genome.connections().size() == connectionsBefore.size() &&
               "probability 0 must change nothing structurally");
        for (std::size_t i = 0; i < connectionsBefore.size(); ++i)
        {
            assert(genome.connections()[i].getWeight() == connectionsBefore[i].getWeight() &&
                   "probability 0 must not touch any existing connection");
        }
        assert(tracker.getNextAvailableNodeId() == 200 && tracker.getNextAvailableInnovation() == 300 &&
               "probability 0 must leave the tracker unchanged");
    }

    // 2 & 6-13: probability 1 attempts (and, given an available valid
    // pair, succeeds at) adding exactly one new, well-formed connection,
    // without touching anything that already existed.
    {
        Genome genome = makeTestGenome();
        const std::size_t nodeCountBefore = genome.nodes().size();
        const std::vector<NodeGene> nodesBefore = genome.nodes();
        const std::vector<ConnectionGene> connectionsBefore = genome.connections();

        InnovationTracker tracker(200, 300);
        MutationConfig config;
        config.addConnectionProbability = 1.0f;
        GenomeMutator mutator(42u);

        const bool added = mutator.mutateAddConnection(genome, tracker, config);
        assert(added && "probability 1 with an available valid pair must add a connection"); // 2 & 6 (part 1)

        assert(genome.nodes().size() == nodeCountBefore && "node count must remain unchanged"); // 7
        for (std::size_t i = 0; i < nodesBefore.size(); ++i)
        {
            assert(genome.nodes()[i] == nodesBefore[i] && "existing node genes must remain unchanged"); // 8 (nodes)
        }
        for (std::size_t i = 0; i < connectionsBefore.size(); ++i)
        {
            const ConnectionGene& before = connectionsBefore[i];
            const ConnectionGene& after = genome.connections()[i];
            assert(after.getSourceId() == before.getSourceId() && after.getTargetId() == before.getTargetId() &&
                   after.getWeight() == before.getWeight() && after.isEnabled() == before.isEnabled() &&
                   after.getInnovationNumber() == before.getInnovationNumber() &&
                   "existing connection genes must remain unchanged"); // 8 (connections)
        }
        assert(genome.connections().size() == connectionsBefore.size() + 1 &&
               "connection count must increase by exactly one"); // 6 (part 2)

        const ConnectionGene& newConnection = genome.connections().back();
        assert(genome.hasNode(newConnection.getSourceId()) && genome.hasNode(newConnection.getTargetId()) &&
               "new connection endpoints must reference existing nodes"); // 9
        assert(newConnection.isEnabled() && "a newly added connection must be enabled"); // 10
        assert(newConnection.getWeight() >= config.newConnectionWeightMin &&
               newConnection.getWeight() <= config.newConnectionWeightMax &&
               "new weight must lie within the configured bounds"); // 11

        // 12: innovation comes from InnovationTracker -- re-requesting the
        // same pair from the same tracker must reuse (not change) it.
        const InnovationNumber beforeReuseCheck = tracker.getNextAvailableInnovation();
        const InnovationNumber sameInnovation =
            tracker.getConnectionInnovation(newConnection.getSourceId(), newConnection.getTargetId());
        assert(sameInnovation == newConnection.getInnovationNumber() && tracker.getNextAvailableInnovation() == beforeReuseCheck &&
               "the new connection's innovation number must come from, and already be recorded in, InnovationTracker");

        // 13: tracker advances only after successful mutation -- node
        // counter untouched (add-connection never allocates a node ID),
        // innovation counter advanced by exactly one.
        assert(tracker.getNextAvailableNodeId() == 200 && "add-connection mutation must never allocate a node ID");
        assert(tracker.getNextAvailableInnovation() == 301 &&
               "a successful mutation must advance the innovation counter by exactly one");
    }

    // 14: a failed mutation (selected, but no valid candidate exists) does
    // not advance the tracker.
    {
        Genome genome = makeAlreadyConnectedGenome(/*enabled=*/true);
        InnovationTracker tracker(50, 60);
        MutationConfig config;
        config.addConnectionProbability = 1.0f;
        GenomeMutator mutator(1u);

        const bool added = mutator.mutateAddConnection(genome, tracker, config);
        assert(!added && "a genome whose only structurally possible pair already exists must fail to mutate");
        assert(tracker.getNextAvailableNodeId() == 50 && tracker.getNextAvailableInnovation() == 60 &&
               "a failed mutation must leave the tracker completely unchanged");
    }

    // 3: empty Genome returns false.
    {
        Genome empty;
        InnovationTracker tracker(0, 0);
        MutationConfig config;
        config.addConnectionProbability = 1.0f;
        GenomeMutator mutator(1u);
        assert(!mutator.mutateAddConnection(empty, tracker, config) && "an empty genome must return false");
        assert(tracker.getNextAvailableNodeId() == 0 && tracker.getNextAvailableInnovation() == 0 &&
               "an empty genome must leave the tracker unchanged");
    }

    // 4 & 18: a genome with no valid source node type (only Output nodes)
    // returns false -- Output can never be a source.
    {
        Genome genome = makeOnlyOutputsGenome();
        InnovationTracker tracker(0, 0);
        MutationConfig config;
        config.addConnectionProbability = 1.0f;
        GenomeMutator mutator(1u);
        assert(!mutator.mutateAddConnection(genome, tracker, config) &&
               "a genome with only Output nodes must have no valid source and must return false");
    }

    // 5 & 19: a genome with no valid target node type (only Input nodes)
    // returns false -- Input can never be a target.
    {
        Genome genome = makeOnlyInputsGenome();
        InnovationTracker tracker(0, 0);
        MutationConfig config;
        config.addConnectionProbability = 1.0f;
        GenomeMutator mutator(1u);
        assert(!mutator.mutateAddConnection(genome, tracker, config) &&
               "a genome with only Input nodes must have no valid target and must return false");
    }

    // 20: Bias can never be a target either.
    {
        Genome genome = makeOnlyBiasGenome();
        InnovationTracker tracker(0, 0);
        MutationConfig config;
        config.addConnectionProbability = 1.0f;
        GenomeMutator mutator(1u);
        assert(!mutator.mutateAddConnection(genome, tracker, config) &&
               "a genome with only Bias nodes must have no valid target and must return false");
    }

    // 21 & 25: Input can be a source, Output can be a target -- the only
    // structurally possible pair must be found and added.
    {
        Genome genome = makeInputToOutputGenome();
        InnovationTracker tracker(0, 0);
        MutationConfig config;
        config.addConnectionProbability = 1.0f;
        GenomeMutator mutator(7u);
        assert(mutator.mutateAddConnection(genome, tracker, config) && "Input -> Output must be found as a valid candidate");
        assert(genome.connections().size() == 1 && genome.connections()[0].getSourceId() == 0 &&
               genome.connections()[0].getTargetId() == 1 && "the added connection must be exactly Input(0) -> Output(1)");
    }

    // 22: Bias can be a source.
    {
        Genome genome = makeBiasToOutputGenome();
        InnovationTracker tracker(0, 0);
        MutationConfig config;
        config.addConnectionProbability = 1.0f;
        GenomeMutator mutator(7u);
        assert(mutator.mutateAddConnection(genome, tracker, config) && "Bias -> Output must be found as a valid candidate");
        assert(genome.connections().size() == 1 && genome.connections()[0].getSourceId() == 0 &&
               "Bias must be usable as a source");
    }

    // 23 & 24: Hidden can be both a source and a target.
    {
        Genome genome = makeHiddenToHiddenGenome();
        InnovationTracker tracker(0, 0);
        MutationConfig config;
        config.addConnectionProbability = 1.0f;
        GenomeMutator mutator(7u);
        assert(mutator.mutateAddConnection(genome, tracker, config) && "Hidden -> Hidden must be found as a valid candidate");
        assert(genome.connections().size() == 1 && genome.connections()[0].getSourceId() == 0 &&
               genome.connections()[0].getTargetId() == 1 && "Hidden must be usable as both source and target");
    }

    // 17: self-connection is never added, even when it is the only
    // structurally "available" pair.
    {
        Genome genome = makeSingleHiddenGenome();
        InnovationTracker tracker(0, 0);
        MutationConfig config;
        config.addConnectionProbability = 1.0f;
        GenomeMutator mutator(1u);
        assert(!mutator.mutateAddConnection(genome, tracker, config) && "a single node can never validly connect to itself");
        assert(genome.connections().empty() && "no self-connection may ever be added");
    }

    // 15: duplicate directed connection is never added.
    {
        Genome genome = makeAlreadyConnectedGenome(/*enabled=*/true);
        InnovationTracker tracker(0, 0);
        MutationConfig config;
        config.addConnectionProbability = 1.0f;
        GenomeMutator mutator(1u);
        assert(!mutator.mutateAddConnection(genome, tracker, config) &&
               "the only structurally possible pair already exists, so nothing may be added");
        assert(genome.connections().size() == 1 && "an already-existing connection must never be duplicated");
    }

    // 16: a disabled duplicate still blocks addition.
    {
        Genome genome = makeAlreadyConnectedGenome(/*enabled=*/false);
        InnovationTracker tracker(0, 0);
        MutationConfig config;
        config.addConnectionProbability = 1.0f;
        GenomeMutator mutator(1u);
        assert(!mutator.mutateAddConnection(genome, tracker, config) &&
               "a disabled connection between the only possible pair must still block a duplicate");
        assert(genome.connections().size() == 1 && !genome.connections()[0].isEnabled() &&
               "the existing disabled connection must not be re-enabled or duplicated");
    }

    // 26 & 28: a connection that would create an enabled cycle is rejected
    // (10 -> 3 exists, so 3 -> 10 would close a 2-cycle), regardless of node ID order.
    {
        Genome genome = makeCycleGenome(/*existingEdgeEnabled=*/true);
        InnovationTracker tracker(0, 0);
        MutationConfig config;
        config.addConnectionProbability = 1.0f;
        GenomeMutator mutator(1u);
        assert(!mutator.mutateAddConnection(genome, tracker, config) &&
               "3 -> 10 must be rejected as a cycle (10 already has an enabled path to 3), regardless of node ID order");
        assert(genome.connections().size() == 1 && "a rejected cycle candidate must not be added");
    }

    // 27: disabled edges do not participate in cycle detection -- with the
    // existing 10 -> 3 edge disabled, 3 -> 10 is no longer a cycle and must
    // be accepted.
    {
        Genome genome = makeCycleGenome(/*existingEdgeEnabled=*/false);
        InnovationTracker tracker(0, 0);
        MutationConfig config;
        config.addConnectionProbability = 1.0f;
        GenomeMutator mutator(1u);
        const bool added = mutator.mutateAddConnection(genome, tracker, config);
        assert(added && "with 10 -> 3 disabled, it is not an enabled path, so 3 -> 10 is not a cycle");
        assert(genome.connections().size() == 2 && genome.connections().back().getSourceId() == 3 &&
               genome.connections().back().getTargetId() == 10 &&
               "the new connection must be exactly 3 -> 10, proving disabled edges do not create false cycle detection");
    }

    // 29: the exhaustive fallback finds the sole valid pair even with
    // maxAttempts = 1 (overwhelmingly likely to miss it randomly) -- every
    // seed must still succeed via the fallback scan.
    {
        for (std::uint32_t seed = 1; seed <= 20; ++seed)
        {
            Genome genome = makeSparseValidPairGenome();
            InnovationTracker tracker(0, 0);
            MutationConfig config;
            config.addConnectionProbability = 1.0f;
            config.addConnectionMaxAttempts = 1;
            GenomeMutator mutator(seed);

            const bool added = mutator.mutateAddConnection(genome, tracker, config);
            assert(added && genome.connections().back().getSourceId() == 1 && genome.connections().back().getTargetId() == 3 &&
                   "the single valid pair must always be found, whether by the one random attempt or the fallback scan");
        }
    }

    // 30: a saturated valid DAG (every valid feed-forward pair already
    // exists) returns false.
    {
        Genome genome = makeSaturatedGenome();
        InnovationTracker tracker(0, 0);
        MutationConfig config;
        config.addConnectionProbability = 1.0f;
        GenomeMutator mutator(1u);
        assert(!mutator.mutateAddConnection(genome, tracker, config) &&
               "a genome where every valid pair already exists must return false");
        assert(genome.connections().size() == 5 && "a saturated genome must not gain a new connection");
    }

    // 31: identical seed + identical genome + identical tracker state
    // produces the same added pair, weight, and innovation.
    {
        MutationConfig config;
        config.addConnectionProbability = 1.0f;

        Genome genomeA = makeTestGenome();
        InnovationTracker trackerA(200, 300);
        GenomeMutator mutatorA(777u);
        mutatorA.mutateAddConnection(genomeA, trackerA, config);
        const ConnectionGene& resultA = genomeA.connections().back();

        Genome genomeB = makeTestGenome();
        InnovationTracker trackerB(200, 300);
        GenomeMutator mutatorB(777u);
        mutatorB.mutateAddConnection(genomeB, trackerB, config);
        const ConnectionGene& resultB = genomeB.connections().back();

        assert(resultA.getSourceId() == resultB.getSourceId() && resultA.getTargetId() == resultB.getTargetId() &&
               resultA.getWeight() == resultB.getWeight() && resultA.getInnovationNumber() == resultB.getInnovationNumber() &&
               "identical seed, genome and tracker state must produce identical results");
    }

    // 32: different seeds can produce different valid results.
    {
        MutationConfig config;
        config.addConnectionProbability = 1.0f;

        Genome genomeA = makeTestGenome();
        InnovationTracker trackerA(200, 300);
        GenomeMutator mutatorA(1u);
        mutatorA.mutateAddConnection(genomeA, trackerA, config);
        const ConnectionGene& resultA = genomeA.connections().back();

        Genome genomeB = makeTestGenome();
        InnovationTracker trackerB(200, 300);
        GenomeMutator mutatorB(2u);
        mutatorB.mutateAddConnection(genomeB, trackerB, config);
        const ConnectionGene& resultB = genomeB.connections().back();

        assert((resultA.getSourceId() != resultB.getSourceId() || resultA.getTargetId() != resultB.getTargetId() ||
                resultA.getWeight() != resultB.getWeight()) &&
               "different seeds must be capable of producing different results");
    }

    // 33: repeated calls advance the mutator's owned RNG state.
    {
        MutationConfig config;
        config.addConnectionProbability = 1.0f;

        Genome genomeA = makeTestGenome();
        InnovationTracker trackerA(200, 300);
        Genome genomeB = makeTestGenome();
        InnovationTracker trackerB(200, 300);

        GenomeMutator mutator(55u);
        mutator.mutateAddConnection(genomeA, trackerA, config);
        mutator.mutateAddConnection(genomeB, trackerB, config);

        const ConnectionGene& resultA = genomeA.connections().back();
        const ConnectionGene& resultB = genomeB.connections().back();
        assert((resultA.getSourceId() != resultB.getSourceId() || resultA.getTargetId() != resultB.getTargetId() ||
                resultA.getWeight() != resultB.getWeight()) &&
               "repeated calls from the same mutator must advance its RNG state, not repeat the same draws");
    }

    // 34, 35, 36 & 37: invalid configuration is rejected, and never
    // touches the tracker.
    {
        Genome genome = makeTestGenome();
        InnovationTracker tracker(200, 300);
        GenomeMutator mutator(1u);

        MutationConfig badProbability;
        badProbability.addConnectionProbability = 1.5f;
        assert(throwsInvalidArgument([&]() { mutator.mutateAddConnection(genome, tracker, badProbability); }) &&
               "an addConnectionProbability outside [0,1] must be rejected"); // 34

        MutationConfig invertedRange;
        invertedRange.newConnectionWeightMin = 1.0f;
        invertedRange.newConnectionWeightMax = -1.0f;
        assert(throwsInvalidArgument([&]() { mutator.mutateAddConnection(genome, tracker, invertedRange); }) &&
               "an inverted new-connection weight range must be rejected"); // 35

        MutationConfig infiniteBound;
        infiniteBound.newConnectionWeightMin = std::numeric_limits<float>::infinity();
        assert(throwsInvalidArgument([&]() { mutator.mutateAddConnection(genome, tracker, infiniteBound); }) &&
               "a non-finite weight bound must be rejected"); // 36

        MutationConfig nanProbability;
        nanProbability.addConnectionProbability = std::numeric_limits<float>::quiet_NaN();
        assert(throwsInvalidArgument([&]() { mutator.mutateAddConnection(genome, tracker, nanProbability); }) &&
               "a NaN probability must be rejected"); // 36 (continued)

        MutationConfig zeroAttempts;
        zeroAttempts.addConnectionMaxAttempts = 0;
        assert(throwsInvalidArgument([&]() { mutator.mutateAddConnection(genome, tracker, zeroAttempts); }) &&
               "a zero addConnectionMaxAttempts must be rejected"); // 37

        MutationConfig negativeAttempts;
        negativeAttempts.addConnectionMaxAttempts = -5;
        assert(throwsInvalidArgument([&]() { mutator.mutateAddConnection(genome, tracker, negativeAttempts); }) &&
               "a negative addConnectionMaxAttempts must be rejected"); // 37 (continued)

        assert(tracker.getNextAvailableNodeId() == 200 && tracker.getNextAvailableInnovation() == 300 &&
               "rejected configuration must never touch the tracker");
    }

    // 38, 39 & 40: Genome::validate() and buildPhenotype() succeed after
    // mutation, and the resulting phenotype is deterministic.
    {
        Genome genome = makeTestGenome();
        InnovationTracker tracker(200, 300);
        MutationConfig config;
        config.addConnectionProbability = 1.0f;
        GenomeMutator mutator(123u);

        assert(mutator.mutateAddConnection(genome, tracker, config) && "setup for phenotype checks must succeed");

        genome.validate(); // 38: must not throw

        ai::NeuralNetwork phenotype = ai::neat::buildPhenotype(genome);      // 39: must not throw
        ai::NeuralNetwork phenotypeAgain = ai::neat::buildPhenotype(genome); // 40: rebuilding must match

        ai::Observation obs;
        obs.values.fill(0.3f);
        const auto out1 = phenotype.evaluate(obs);
        const auto out2 = phenotypeAgain.evaluate(obs);
        assert(out1[0] == out2[0] && out1[1] == out2[1] && "the phenotype built from a mutated genome must be deterministic");
    }

    // 41: mutateWeights() regression check -- zero probability still changes nothing.
    {
        Genome genome = makeTestGenome();
        const std::vector<ConnectionGene> before = genome.connections();
        MutationConfig config;
        config.weightMutationProbability = 0.0f;
        GenomeMutator mutator(1u);
        mutator.mutateWeights(genome, config);
        for (std::size_t i = 0; i < before.size(); ++i)
        {
            assert(genome.connections()[i].getWeight() == before[i].getWeight() &&
                   "mutateWeights() behavior must remain unchanged: zero probability still changes nothing");
        }
    }

    // 42: no add-node behavior occurs -- node-ID counter never advances, across a batch of seeds.
    {
        for (std::uint32_t seed = 1; seed <= 5; ++seed)
        {
            Genome genome = makeTestGenome();
            InnovationTracker tracker(200, 300);
            MutationConfig config;
            config.addConnectionProbability = 1.0f;
            GenomeMutator mutator(seed);
            mutator.mutateAddConnection(genome, tracker, config);
            assert(tracker.getNextAvailableNodeId() == 200 &&
                   "add-connection mutation must never allocate a node ID (no add-node behavior)");
        }
    }

    // 43: all previous verification suites still pass.

    TraceLog(LOG_INFO, "Add-connection mutation verification: all deterministic checks passed");
}

namespace add_node_verify
{

// Minimal genome: one Input, one Output, one enabled connection. Used for
// the plain structural-invariant checks that don't need a full
// Observation-shaped (9 Input + 1 Bias + 2 Output) genome.
ai::neat::Genome makeSimpleSplitGenome()
{
    using ai::neat::ConnectionGene;
    using ai::neat::Genome;
    using ai::neat::NodeGene;
    using ai::neat::NodeType;

    Genome genome;
    genome.addNode(NodeGene{0, NodeType::Input});
    genome.addNode(NodeGene{1, NodeType::Output});
    genome.addConnection(ConnectionGene{0, 1, 0.75f, true, 0});
    return genome;
}

// A genome whose sole connection is already disabled: no eligible candidate
// exists to split.
ai::neat::Genome makeOnlyDisabledConnectionGenome()
{
    using ai::neat::ConnectionGene;
    using ai::neat::Genome;
    using ai::neat::NodeGene;
    using ai::neat::NodeType;

    Genome genome;
    genome.addNode(NodeGene{0, NodeType::Input});
    genome.addNode(NodeGene{1, NodeType::Output});
    genome.addConnection(ConnectionGene{0, 1, 0.5f, false, 0});
    return genome;
}

} // namespace add_node_verify

// Deterministic check of ai::neat::GenomeMutator::mutateAddNode(), end to
// end. No crossover/species/population logic here.
void verifyAddNodeMutation()
{
    using namespace add_node_verify;
    using namespace genome_mutator_verify;
    using ai::neat::ConnectionGene;
    using ai::neat::Genome;
    using ai::neat::GenomeMutator;
    using ai::neat::InnovationNumber;
    using ai::neat::InnovationTracker;
    using ai::neat::MutationConfig;
    using ai::neat::NodeGene;
    using ai::neat::NodeId;
    using ai::neat::NodeSplitInnovation;
    using ai::neat::NodeType;

    // 1 & 25: probability 0 returns false and changes nothing; tracker
    // unchanged.
    {
        Genome genome = makeSimpleSplitGenome();
        const std::vector<NodeGene> nodesBefore = genome.nodes();
        const std::vector<ConnectionGene> connectionsBefore = genome.connections();
        InnovationTracker tracker(50, 60);
        MutationConfig config;
        config.addNodeProbability = 0.0f;
        GenomeMutator mutator(1u);

        const bool result = mutator.mutateAddNode(genome, tracker, config);
        assert(!result && "probability 0 must never split a connection");
        assert(genome.nodes().size() == nodesBefore.size() && genome.connections().size() == connectionsBefore.size() &&
               "probability 0 must change nothing structurally");
        assert(tracker.getNextAvailableNodeId() == 50 && tracker.getNextAvailableInnovation() == 60 &&
               "probability 0 must leave the tracker unchanged");
    }

    // 3 & 26: empty genome returns false, tracker unchanged.
    {
        Genome empty;
        InnovationTracker tracker(0, 0);
        MutationConfig config;
        config.addNodeProbability = 1.0f;
        GenomeMutator mutator(1u);
        assert(!mutator.mutateAddNode(empty, tracker, config) && "an empty genome must return false");
        assert(tracker.getNextAvailableNodeId() == 0 && tracker.getNextAvailableInnovation() == 0 &&
               "an empty genome must leave the tracker unchanged");
    }

    // 4, 26 & 33: a genome with only disabled connections returns false,
    // tracker unchanged, and the disabled connection remains untouched.
    {
        Genome genome = makeOnlyDisabledConnectionGenome();
        InnovationTracker tracker(10, 20);
        MutationConfig config;
        config.addNodeProbability = 1.0f;
        GenomeMutator mutator(1u);
        assert(!mutator.mutateAddNode(genome, tracker, config) &&
               "a genome with only disabled connections must have no eligible candidate");
        assert(tracker.getNextAvailableNodeId() == 10 && tracker.getNextAvailableInnovation() == 20 &&
               "no eligible candidate must leave the tracker unchanged");
        assert(genome.connections().size() == 1 && !genome.connections()[0].isEnabled() &&
               "the sole disabled connection must remain untouched");
    }

    // 2, 5-21: probability 1 splits the sole enabled connection, producing
    // exactly the structure NEAT's add-node mutation specifies.
    {
        Genome genome = makeSimpleSplitGenome();
        InnovationTracker tracker(50, 60);
        MutationConfig config;
        config.addNodeProbability = 1.0f;
        GenomeMutator mutator(1u);

        const bool result = mutator.mutateAddNode(genome, tracker, config);
        assert(result && "probability 1 with an eligible connection must split it"); // 2

        // 5: a brand-new split advances the tracker by one node ID and two innovation numbers.
        assert(tracker.getNextAvailableNodeId() == 51 && "a successful split must allocate exactly one new node ID");
        assert(tracker.getNextAvailableInnovation() == 62 &&
               "a successful split must allocate exactly two new innovation numbers");

        assert(genome.nodes().size() == 3 && "node count must increase by exactly one"); // 5
        assert(genome.connections().size() == 3 && "connection count must increase by exactly two"); // 6

        const ConnectionGene* original = genome.findConnection(0, 1);
        assert(original != nullptr && "the original connection must remain present"); // 7
        assert(!original->isEnabled() && "the original connection must become disabled"); // 8
        assert(original->getWeight() == 0.75f && "the original connection's weight must remain unchanged"); // 9 (weight)
        assert(original->getSourceId() == 0 && original->getTargetId() == 1 &&
               "the original connection's source/target must remain unchanged"); // 9 (source/target)
        assert(original->getInnovationNumber() == 0 && "the original connection's innovation must remain unchanged"); // 9 (innovation)

        const NodeGene& newNode = genome.nodes().back();
        assert(newNode.getType() == NodeType::Hidden && "the new node must be Hidden"); // 10

        // Re-requesting the same split is idempotent, proving the new node
        // ID and both new innovations came from InnovationTracker.
        const NodeSplitInnovation split = tracker.getNodeSplitInnovation(0, 0, 1);
        assert(newNode.getId() == split.newNodeId && "the new node's ID must come from InnovationTracker"); // 11

        const ConnectionGene* incoming = genome.findConnection(0, split.newNodeId);
        assert(incoming != nullptr && "source -> newNode connection must exist");
        assert(incoming->getWeight() == 1.0f && "source -> newNode weight must be exactly 1.0"); // 12
        assert(incoming->isEnabled() && "source -> newNode connection must be enabled"); // 14 (part 1)
        assert(incoming->getInnovationNumber() == split.incomingInnovation &&
               "source -> newNode innovation must come from InnovationTracker"); // 15 (part 1)

        const ConnectionGene* outgoing = genome.findConnection(split.newNodeId, 1);
        assert(outgoing != nullptr && "newNode -> target connection must exist");
        assert(outgoing->getWeight() == 0.75f && "newNode -> target weight must equal the old connection's weight"); // 13
        assert(outgoing->isEnabled() && "newNode -> target connection must be enabled"); // 14 (part 2)
        assert(outgoing->getInnovationNumber() == split.outgoingInnovation &&
               "newNode -> target innovation must come from InnovationTracker"); // 15 (part 2)

        // 16 & 17: no unrelated node/connection exists -- exactly the three
        // nodes and three connections accounted for above.
        assert(genome.hasNode(0) && genome.hasNode(1) && genome.hasNode(split.newNodeId) &&
               "no unrelated node may appear");
        assert(genome.connections().size() == 3 && "no unrelated connection may appear");

        genome.validate(); // 18: must not throw
    }

    // 18, 19, 20 & 21: validate()/buildPhenotype() succeed after mutation
    // (phenotype stays acyclic), and disabled connections never affect evaluation.
    {
        Genome genome = makeTestGenome();
        InnovationTracker tracker(200, 300);
        MutationConfig config;
        config.addNodeProbability = 1.0f;
        GenomeMutator mutator(123u);

        assert(mutator.mutateAddNode(genome, tracker, config) && "setup for phenotype checks must succeed");

        genome.validate(); // 18: must not throw

        ai::NeuralNetwork phenotype = ai::neat::buildPhenotype(genome); // 19 & 20: must not throw (acyclic)
        ai::Observation obs;
        obs.values.fill(0.3f);
        const auto outBefore = phenotype.evaluate(obs);

        // 21: perturbing every disabled connection's weight (including the
        // split original) must not change the phenotype's evaluation.
        for (ConnectionGene& connection : genome.mutableConnections())
        {
            if (!connection.isEnabled())
            {
                connection.setWeight(connection.getWeight() + 999.0f);
            }
        }
        ai::NeuralNetwork phenotypeAfter = ai::neat::buildPhenotype(genome);
        const auto outAfter = phenotypeAfter.evaluate(obs);
        assert(outBefore[0] == outAfter[0] && outBefore[1] == outAfter[1] &&
               "disabled connections, including the split original, must never affect evaluation");
    }

    // 22: identical seed + identical genome + identical tracker state
    // produces the same split.
    {
        MutationConfig config;
        config.addNodeProbability = 1.0f;

        Genome genomeA = makeTestGenome();
        InnovationTracker trackerA(200, 300);
        GenomeMutator mutatorA(777u);
        mutatorA.mutateAddNode(genomeA, trackerA, config);

        Genome genomeB = makeTestGenome();
        InnovationTracker trackerB(200, 300);
        GenomeMutator mutatorB(777u);
        mutatorB.mutateAddNode(genomeB, trackerB, config);

        assert(genomeA.nodes().size() == genomeB.nodes().size() &&
               genomeA.connections().size() == genomeB.connections().size() &&
               "identical seed, genome and tracker state must produce a structurally identical result");
        const NodeGene& newNodeA = genomeA.nodes().back();
        const NodeGene& newNodeB = genomeB.nodes().back();
        assert(newNodeA.getId() == newNodeB.getId() && "identical runs must pick the same new node ID");
        const ConnectionGene& lastA = genomeA.connections().back();
        const ConnectionGene& lastB = genomeB.connections().back();
        assert(lastA.getSourceId() == lastB.getSourceId() && lastA.getTargetId() == lastB.getTargetId() &&
               lastA.getInnovationNumber() == lastB.getInnovationNumber() &&
               "identical runs must split the same original connection");
    }

    // 23 & 33: different seeds can select different enabled connections to
    // split (makeTestGenome() has three eligible), and the pre-existing
    // disabled connection is never selected or altered.
    {
        MutationConfig config;
        config.addNodeProbability = 1.0f;
        std::vector<std::pair<NodeId, NodeId>> splitOriginals;

        for (std::uint32_t seed = 1; seed <= 10; ++seed)
        {
            Genome genome = makeTestGenome();
            InnovationTracker tracker(200, 300);
            GenomeMutator mutator(seed);
            const bool result = mutator.mutateAddNode(genome, tracker, config);
            assert(result && "an eligible connection must always be found in makeTestGenome()");

            const ConnectionGene* preExistingDisabled = genome.findConnection(9, 101);
            assert(preExistingDisabled != nullptr && !preExistingDisabled->isEnabled() &&
                   preExistingDisabled->getWeight() == 0.2f &&
                   "a connection that was already disabled must never be selected or altered");

            for (const ConnectionGene& connection : genome.connections())
            {
                if (!connection.isEnabled() && connection.getInnovationNumber() != 2)
                {
                    splitOriginals.emplace_back(connection.getSourceId(), connection.getTargetId());
                }
            }
        }

        const bool sawMoreThanOneChoice =
            std::any_of(splitOriginals.begin(), splitOriginals.end(),
                        [&](const std::pair<NodeId, NodeId>& p) { return p != splitOriginals.front(); });
        assert(sawMoreThanOneChoice && "different seeds must be capable of selecting different eligible connections");
    }

    // 24: repeated calls advance the mutator's RNG state -- ten consecutive
    // calls (a single pair could coincidentally repeat) must show variation.
    {
        MutationConfig config;
        config.addNodeProbability = 1.0f;
        GenomeMutator mutator(55u);

        std::vector<std::pair<NodeId, NodeId>> choices;
        for (int i = 0; i < 10; ++i)
        {
            Genome genome = makeTestGenome();
            InnovationTracker tracker(200, 300);
            mutator.mutateAddNode(genome, tracker, config);
            const ConnectionGene& last = genome.connections().back();
            choices.emplace_back(last.getSourceId(), last.getTargetId());
        }

        const bool sawVariation = std::any_of(choices.begin(), choices.end(),
                                               [&](const std::pair<NodeId, NodeId>& p) { return p != choices.front(); });
        assert(sawVariation && "repeated calls from the same mutator must advance its RNG state, not repeat the same draws");
    }

    // 27, 28 & 29: the same historical split, requested for two
    // structurally equivalent (but distinct) genomes sharing one
    // InnovationTracker, reuses exactly the same new node ID and the same
    // two connection innovations -- never renumbered.
    {
        InnovationTracker sharedTracker(50, 60);
        MutationConfig config;
        config.addNodeProbability = 1.0f;

        Genome genomeA = makeSimpleSplitGenome();
        GenomeMutator mutatorA(1u);
        assert(mutatorA.mutateAddNode(genomeA, sharedTracker, config) && "first genome's split must succeed");

        // genomeB's split reuses genomeA's committed historical record, so
        // it must not advance the tracker further.
        const NodeId nodeCounterBeforeReuse = sharedTracker.getNextAvailableNodeId();
        const InnovationNumber innovationCounterBeforeReuse = sharedTracker.getNextAvailableInnovation();

        Genome genomeB = makeSimpleSplitGenome();
        GenomeMutator mutatorB(2u); // different seed, same single eligible connection either way
        assert(mutatorB.mutateAddNode(genomeB, sharedTracker, config) && "second genome's split must succeed");

        assert(sharedTracker.getNextAvailableNodeId() == nodeCounterBeforeReuse &&
               "reusing an already-recorded historical split must not advance the node ID counter"); // NEW 6
        assert(sharedTracker.getNextAvailableInnovation() == innovationCounterBeforeReuse &&
               "reusing an already-recorded historical split must not advance the innovation counter"); // NEW 6

        const NodeGene& newNodeA = genomeA.nodes().back();
        const NodeGene& newNodeB = genomeB.nodes().back();
        assert(newNodeA.getId() == newNodeB.getId() &&
               "the same historical split across two genomes must reuse the same new node ID"); // 27

        const ConnectionGene* incomingA = genomeA.findConnection(0, newNodeA.getId());
        const ConnectionGene* incomingB = genomeB.findConnection(0, newNodeB.getId());
        const ConnectionGene* outgoingA = genomeA.findConnection(newNodeA.getId(), 1);
        const ConnectionGene* outgoingB = genomeB.findConnection(newNodeB.getId(), 1);
        assert(incomingA != nullptr && incomingB != nullptr && outgoingA != nullptr && outgoingB != nullptr &&
               incomingA->getInnovationNumber() == incomingB->getInnovationNumber() &&
               outgoingA->getInnovationNumber() == outgoingB->getInnovationNumber() &&
               "the same historical split across two genomes must reuse the same two connection innovations"); // 28

        const NodeSplitInnovation split = sharedTracker.getNodeSplitInnovation(0, 0, 1);
        assert(newNodeA.getId() == split.newNodeId && newNodeB.getId() == split.newNodeId &&
               "the historical node ID must never be renumbered"); // 29
    }

    // 30: a returned new node ID that already exists in the genome as a
    // non-Hidden node is a structural inconsistency and must be rejected
    // clearly, not silently repaired or renumbered.
    {
        InnovationTracker tracker(5, 10); // will allocate node ID 5 for this split
        Genome genome;
        genome.addNode(NodeGene{0, NodeType::Input});
        genome.addNode(NodeGene{1, NodeType::Output});
        genome.addNode(NodeGene{5, NodeType::Input}); // occupies the ID the split would allocate
        genome.addConnection(ConnectionGene{0, 1, 0.5f, true, 0});

        MutationConfig config;
        config.addNodeProbability = 1.0f;
        GenomeMutator mutator(1u);

        assert(throwsInvalidArgument([&]() { mutator.mutateAddNode(genome, tracker, config); }) &&
               "a conflicting existing non-Hidden node at the split's new node ID must be rejected");
        // Detected by inspection before any allocation -- tracker unchanged.
        assert(tracker.getNextAvailableNodeId() == 5 && tracker.getNextAvailableInnovation() == 10 &&
               "a structural conflict must never alter the tracker's state");
    }

    // 31: an existing connection occupying one of the split's two directed
    // pairs, but with a different innovation number than InnovationTracker
    // just returned, is a structural inconsistency and must be rejected.
    {
        InnovationTracker tracker(5, 10);
        Genome genome;
        genome.addNode(NodeGene{0, NodeType::Input});
        genome.addNode(NodeGene{1, NodeType::Output});
        genome.addNode(NodeGene{5, NodeType::Hidden}); // correct type -- not a conflict by itself
        genome.addConnection(ConnectionGene{0, 1, 0.5f, true, 0});
        genome.addConnection(ConnectionGene{0, 5, 1.0f, true, 999}); // wrong innovation for source -> newNode

        MutationConfig config;
        config.addNodeProbability = 1.0f;
        GenomeMutator mutator(1u);

        assert(throwsInvalidArgument([&]() { mutator.mutateAddNode(genome, tracker, config); }) &&
               "an existing connection with a conflicting innovation number at the split's endpoints must be rejected");
        assert(tracker.getNextAvailableNodeId() == 5 && tracker.getNextAvailableInnovation() == 10 &&
               "a structural conflict must never alter the tracker's state");
    }

    // 32: when the selected candidate's split already fully exists in the
    // genome, the mutation does not duplicate it -- it moves on to another
    // eligible candidate instead.
    {
        InnovationTracker tracker(50, 60);

        Genome genome;
        genome.addNode(NodeGene{0, NodeType::Input});
        genome.addNode(NodeGene{1, NodeType::Output});
        genome.addNode(NodeGene{2, NodeType::Output});
        genome.addConnection(ConnectionGene{0, 1, 0.5f, true, 0}); // connA: already "split" below
        genome.addConnection(ConnectionGene{0, 2, 0.9f, true, 1}); // connB: fresh, another eligible candidate

        // Pre-populate connA's split genes as if it already happened, while
        // leaving connA itself enabled -- eligible but unsuitable. The two
        // new split connections are themselves splittable, so any of connB
        // or either of them is a valid alternative; the test only requires
        // some valid alternative gets split and connA's split is untouched.
        const NodeSplitInnovation splitA = tracker.getNodeSplitInnovation(0, 0, 1);
        genome.addNode(NodeGene{splitA.newNodeId, NodeType::Hidden});
        genome.addConnection(ConnectionGene{0, splitA.newNodeId, 1.0f, true, splitA.incomingInnovation});
        genome.addConnection(ConnectionGene{splitA.newNodeId, 1, 0.5f, true, splitA.outgoingInnovation});

        const std::size_t nodeCountBefore = genome.nodes().size();
        const std::size_t connCountBefore = genome.connections().size();

        MutationConfig config;
        config.addNodeProbability = 1.0f;

        // Try every starting seed so the search is exercised regardless of
        // which candidate is inspected first.
        for (std::uint32_t seed = 1; seed <= 10; ++seed)
        {
            Genome trial = genome;
            InnovationTracker trialTracker = tracker;
            GenomeMutator mutator(seed);

            const NodeId nodeCounterBeforeCall = trialTracker.getNextAvailableNodeId();
            const InnovationNumber innovationCounterBeforeCall = trialTracker.getNextAvailableInnovation();

            const bool result = mutator.mutateAddNode(trial, trialTracker, config);
            assert(result && "some other eligible connection must always remain available even if connA is inspected first and rejected");

            // Regardless of whether connA was inspected and skipped first,
            // the tracker advances by exactly the one selected candidate's cost.
            assert(trialTracker.getNextAvailableNodeId() == nodeCounterBeforeCall + 1 &&
                   "an inspected-but-skipped candidate must never itself allocate a node ID");
            assert(trialTracker.getNextAvailableInnovation() == innovationCounterBeforeCall + 2 &&
                   "an inspected-but-skipped candidate must never itself allocate an innovation number");

            assert(trial.nodes().size() == nodeCountBefore + 1 &&
                   "exactly one new node must be added, never a duplicate of connA's split");
            assert(trial.connections().size() == connCountBefore + 2 &&
                   "exactly two new connections must be added, never duplicates of connA's split");

            const ConnectionGene* connAAfter = trial.findConnection(0, 1);
            assert(connAAfter != nullptr && connAAfter->isEnabled() && connAAfter->getWeight() == 0.5f &&
                   connAAfter->getInnovationNumber() == 0 && "connA, already fully split, must be left completely untouched");

            const ConnectionGene* incomingAAfter = trial.findConnection(0, splitA.newNodeId);
            const ConnectionGene* outgoingAAfter = trial.findConnection(splitA.newNodeId, 1);
            assert(incomingAAfter != nullptr && incomingAAfter->getInnovationNumber() == splitA.incomingInnovation &&
                   outgoingAAfter != nullptr && outgoingAAfter->getInnovationNumber() == splitA.outgoingInnovation &&
                   "connA's existing split genes must not be duplicated or altered");
        }
    }

    // A failed add-node mutation (every eligible candidate unsuitable) must
    // never advance either tracker counter.
    {
        InnovationTracker tracker(50, 60);
        Genome genome;
        genome.addNode(NodeGene{0, NodeType::Input});
        genome.addNode(NodeGene{1, NodeType::Output});
        genome.addConnection(ConnectionGene{0, 1, 0.5f, true, 0});

        // The sole enabled connection's split already exists, so it's
        // unsuitable. The two split connections are added disabled so they
        // don't become alternative eligible candidates themselves.
        const NodeSplitInnovation split = tracker.getNodeSplitInnovation(0, 0, 1);
        genome.addNode(NodeGene{split.newNodeId, NodeType::Hidden});
        genome.addConnection(ConnectionGene{0, split.newNodeId, 1.0f, false, split.incomingInnovation});
        genome.addConnection(ConnectionGene{split.newNodeId, 1, 0.5f, false, split.outgoingInnovation});

        const NodeId nodeCounterBeforeAttempt = tracker.getNextAvailableNodeId();
        const InnovationNumber innovationCounterBeforeAttempt = tracker.getNextAvailableInnovation();

        MutationConfig config;
        config.addNodeProbability = 1.0f;
        GenomeMutator mutator(1u);

        const bool result = mutator.mutateAddNode(genome, tracker, config);
        assert(!result && "the only eligible candidate is already fully split; nothing else to try");
        assert(tracker.getNextAvailableNodeId() == nodeCounterBeforeAttempt &&
               "a failed add-node mutation must never advance the node ID counter");
        assert(tracker.getNextAvailableInnovation() == innovationCounterBeforeAttempt &&
               "a failed add-node mutation must never advance the innovation counter");
    }

    // 34 & 35: invalid addNodeProbability configuration is rejected, and
    // never touches the tracker.
    {
        Genome genome = makeSimpleSplitGenome();
        InnovationTracker tracker(50, 60);
        GenomeMutator mutator(1u);

        MutationConfig outOfRange;
        outOfRange.addNodeProbability = 1.5f;
        assert(throwsInvalidArgument([&]() { mutator.mutateAddNode(genome, tracker, outOfRange); }) &&
               "an addNodeProbability outside [0,1] must be rejected"); // 34

        MutationConfig negative;
        negative.addNodeProbability = -0.1f;
        assert(throwsInvalidArgument([&]() { mutator.mutateAddNode(genome, tracker, negative); }) &&
               "a negative addNodeProbability must be rejected"); // 34 (continued)

        MutationConfig nanProbability;
        nanProbability.addNodeProbability = std::numeric_limits<float>::quiet_NaN();
        assert(throwsInvalidArgument([&]() { mutator.mutateAddNode(genome, tracker, nanProbability); }) &&
               "a NaN addNodeProbability must be rejected"); // 35

        MutationConfig infiniteProbability;
        infiniteProbability.addNodeProbability = std::numeric_limits<float>::infinity();
        assert(throwsInvalidArgument([&]() { mutator.mutateAddNode(genome, tracker, infiniteProbability); }) &&
               "a non-finite addNodeProbability must be rejected"); // 35 (continued)

        assert(tracker.getNextAvailableNodeId() == 50 && tracker.getNextAvailableInnovation() == 60 &&
               "rejected configuration must never touch the tracker");
    }

    // 36: mutateWeights() behavior remains unchanged.
    {
        Genome genome = makeTestGenome();
        const std::vector<ConnectionGene> before = genome.connections();
        MutationConfig config;
        config.weightMutationProbability = 0.0f;
        GenomeMutator mutator(1u);
        mutator.mutateWeights(genome, config);
        for (std::size_t i = 0; i < before.size(); ++i)
        {
            assert(genome.connections()[i].getWeight() == before[i].getWeight() &&
                   "mutateWeights() behavior must remain unchanged");
        }
    }

    // 37: mutateAddConnection() behavior remains unchanged.
    {
        Genome genome = makeTestGenome();
        InnovationTracker tracker(200, 300);
        MutationConfig config;
        config.addConnectionProbability = 0.0f;
        GenomeMutator mutator(1u);
        const bool added = mutator.mutateAddConnection(genome, tracker, config);
        assert(!added &&
               "mutateAddConnection() behavior must remain unchanged: probability 0 still adds nothing");
    }

    // 38 & 39: no crossover or population API is called anywhere here.
    // 40: all previous verification suites still pass.

    TraceLog(LOG_INFO, "Add-node mutation verification: all deterministic checks passed");
}
} // namespace verification
