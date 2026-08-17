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


namespace compatibility_verify
{

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

// nodeCount nodes (IDs 0..nodeCount-1, alternating Input/Output -- type is
// irrelevant to compatibility distance), no connections.
ai::neat::Genome makeNodeGenome(int nodeCount)
{
    using ai::neat::Genome;
    using ai::neat::NodeGene;
    using ai::neat::NodeType;

    Genome genome;
    for (int i = 0; i < nodeCount; ++i)
    {
        genome.addNode(NodeGene{i, (i % 2 == 0) ? NodeType::Input : NodeType::Output});
    }
    return genome;
}

// connectionCount connections 0->1, 1->2, ... with innovations 0..N-1,
// weight 0. Two chain genomes always agree on any overlapping prefix
// (same node-index scheme), giving clean matching genes for normalization tests.
ai::neat::Genome makeChainGenome(int connectionCount)
{
    using ai::neat::ConnectionGene;
    using ai::neat::Genome;
    using ai::neat::NodeGene;
    using ai::neat::NodeType;

    Genome genome;
    for (int i = 0; i <= connectionCount; ++i)
    {
        genome.addNode(NodeGene{i, (i % 2 == 0) ? NodeType::Input : NodeType::Output});
    }
    for (int i = 0; i < connectionCount; ++i)
    {
        genome.addConnection(ConnectionGene{i, i + 1, 0.0f, true, i});
    }
    return genome;
}

} // namespace compatibility_verify

// Deterministic check of ai::neat::compatibilityDistance()/
// compatibilityBreakdown(), end to end. No species/population/reproduction
// logic here -- only distance calculation.
void verifyCompatibilityDistance()
{
    using namespace compatibility_verify;
    using ai::neat::CompatibilityBreakdown;
    using ai::neat::CompatibilityConfig;
    using ai::neat::compatibilityBreakdown;
    using ai::neat::compatibilityDistance;
    using ai::neat::ConnectionGene;
    using ai::neat::Genome;
    using ai::neat::NodeGene;
    using ai::neat::NodeType;
    constexpr float kEps = 1e-5f;

    // 1, 2 & 3: negative coefficients are rejected.
    {
        Genome a;
        Genome b;

        CompatibilityConfig negExcess;
        negExcess.excessCoefficient = -1.0f;
        assert(throwsInvalidArgument([&]() { compatibilityDistance(a, b, negExcess); }) &&
               "a negative excessCoefficient must be rejected"); // 1

        CompatibilityConfig negDisjoint;
        negDisjoint.disjointCoefficient = -1.0f;
        assert(throwsInvalidArgument([&]() { compatibilityDistance(a, b, negDisjoint); }) &&
               "a negative disjointCoefficient must be rejected"); // 2

        CompatibilityConfig negWeight;
        negWeight.weightDifferenceCoefficient = -1.0f;
        assert(throwsInvalidArgument([&]() { compatibilityDistance(a, b, negWeight); }) &&
               "a negative weightDifferenceCoefficient must be rejected"); // 3
    }

    // 4: non-finite coefficients are rejected.
    {
        Genome a;
        Genome b;

        CompatibilityConfig nanConfig;
        nanConfig.excessCoefficient = std::numeric_limits<float>::quiet_NaN();
        assert(throwsInvalidArgument([&]() { compatibilityDistance(a, b, nanConfig); }) &&
               "a NaN coefficient must be rejected");

        CompatibilityConfig infConfig;
        infConfig.disjointCoefficient = std::numeric_limits<float>::infinity();
        assert(throwsInvalidArgument([&]() { compatibilityDistance(a, b, infConfig); }) &&
               "an infinite coefficient must be rejected");
    }

    // 5: a zero normalization threshold is rejected.
    {
        Genome a;
        Genome b;
        CompatibilityConfig zeroThreshold;
        zeroThreshold.smallGenomeNormalizationThreshold = 0;
        assert(throwsInvalidArgument([&]() { compatibilityDistance(a, b, zeroThreshold); }) &&
               "a zero smallGenomeNormalizationThreshold must be rejected");
    }

    // 6: two empty connection sets give distance 0.
    {
        Genome a;
        Genome b;
        CompatibilityConfig config;
        assert(compatibilityDistance(a, b, config) == 0.0f && "two genomes with no connections must have distance 0");
    }

    // 7: identical genomes give distance 0.
    {
        Genome a = makeNodeGenome(4);
        a.addConnection(ConnectionGene{0, 1, 0.5f, true, 0});
        a.addConnection(ConnectionGene{2, 3, 0.7f, true, 1});
        Genome b = a; // structurally and weight-identical copy

        CompatibilityConfig config;
        assert(std::fabs(compatibilityDistance(a, b, config)) < kEps && "identical genomes must have distance 0");
    }

    // 8: an enabled-state difference alone gives distance 0 -- matching by
    // innovation, endpoints and weight all agree; only enabled differs.
    {
        Genome a = makeNodeGenome(2);
        a.addConnection(ConnectionGene{0, 1, 0.5f, true, 0});
        Genome b = makeNodeGenome(2);
        b.addConnection(ConnectionGene{0, 1, 0.5f, false, 0});

        CompatibilityConfig config;
        assert(std::fabs(compatibilityDistance(a, b, config)) < kEps &&
               "an enabled-state difference alone must not affect distance");
    }

    // 9: one matching gene with equal weight gives W=0.
    {
        Genome a = makeNodeGenome(2);
        a.addConnection(ConnectionGene{0, 1, 0.5f, true, 0});
        Genome b = makeNodeGenome(2);
        b.addConnection(ConnectionGene{0, 1, 0.5f, true, 0});

        CompatibilityConfig config;
        const CompatibilityBreakdown result = compatibilityBreakdown(a, b, config);
        assert(result.matching == 1 && result.averageWeightDifference == 0.0f &&
               "a matching gene with equal weight must contribute zero to W");
    }

    // 10: one matching gene with different weight gives the correct W.
    {
        Genome a = makeNodeGenome(2);
        a.addConnection(ConnectionGene{0, 1, 0.3f, true, 0});
        Genome b = makeNodeGenome(2);
        b.addConnection(ConnectionGene{0, 1, 0.8f, true, 0});

        CompatibilityConfig config;
        const CompatibilityBreakdown result = compatibilityBreakdown(a, b, config);
        assert(result.matching == 1 && "exactly one matching gene must be found");
        assert(std::fabs(result.averageWeightDifference - 0.5f) < kEps && "W must equal the single weight difference");
    }

    // 11: multiple matching genes calculate the mean absolute weight
    // difference correctly.
    {
        Genome a = makeNodeGenome(6);
        a.addConnection(ConnectionGene{0, 1, 0.0f, true, 0});
        a.addConnection(ConnectionGene{2, 3, 0.5f, true, 1});
        a.addConnection(ConnectionGene{4, 5, 1.0f, true, 2});

        Genome b = makeNodeGenome(6);
        b.addConnection(ConnectionGene{0, 1, 0.3f, true, 0}); // diff 0.3
        b.addConnection(ConnectionGene{2, 3, 0.1f, true, 1}); // diff 0.4
        b.addConnection(ConnectionGene{4, 5, 1.2f, true, 2}); // diff 0.2

        CompatibilityConfig config;
        const CompatibilityBreakdown result = compatibilityBreakdown(a, b, config);
        assert(result.matching == 3 && "three matching genes must be found");
        const float expected = (0.3f + 0.4f + 0.2f) / 3.0f;
        assert(std::fabs(result.averageWeightDifference - expected) < kEps &&
               "W must equal the mean of the individual absolute weight differences");
    }

    // 12: matching genes align by innovation number, independent of vector order.
    {
        Genome a = makeNodeGenome(4);
        a.addConnection(ConnectionGene{2, 3, 0.9f, true, 5}); // innovation 5 stored first
        a.addConnection(ConnectionGene{0, 1, 0.1f, true, 2}); // innovation 2 stored second

        Genome b = makeNodeGenome(4);
        b.addConnection(ConnectionGene{0, 1, 0.15f, true, 2}); // innovation 2 stored first
        b.addConnection(ConnectionGene{2, 3, 0.5f, true, 5});  // innovation 5 stored second

        CompatibilityConfig config;
        const CompatibilityBreakdown result = compatibilityBreakdown(a, b, config);
        assert(result.matching == 2 && "both connections must be matched, one per innovation number");
        // Innovation-aligned: |0.1-0.15|=0.05, |0.9-0.5|=0.4 -> mean 0.225.
        // A broken index-aligned pairing would instead give mean 0.575.
        assert(std::fabs(result.averageWeightDifference - 0.225f) < kEps &&
               "matching must align by innovation number, not vector position");
    }

    // 13: the same innovation number with conflicting endpoints between
    // genomes is rejected.
    {
        Genome a = makeNodeGenome(4);
        a.addConnection(ConnectionGene{0, 1, 0.5f, true, 7});
        Genome b = makeNodeGenome(4);
        b.addConnection(ConnectionGene{2, 3, 0.5f, true, 7}); // same innovation, different endpoints

        CompatibilityConfig config;
        assert(throwsInvalidArgument([&]() { compatibilityDistance(a, b, config); }) &&
               "a matching innovation with conflicting endpoints must be rejected");
    }

    // 14: a duplicate connection innovation number inside genome A is
    // rejected (Genome::validate() alone would not catch this, since the
    // two connections use different (source, target) pairs).
    {
        std::vector<NodeGene> nodes = {NodeGene{0, NodeType::Input}, NodeGene{1, NodeType::Input},
                                        NodeGene{2, NodeType::Output}, NodeGene{3, NodeType::Output}};
        std::vector<ConnectionGene> connections = {ConnectionGene{0, 2, 0.1f, true, 5},
                                                    ConnectionGene{1, 3, 0.2f, true, 5}};
        Genome aWithDuplicateInnovation(nodes, connections);
        Genome b;

        CompatibilityConfig config;
        assert(throwsInvalidArgument([&]() { compatibilityDistance(aWithDuplicateInnovation, b, config); }) &&
               "genome A with duplicate connection innovation numbers must be rejected");
    }

    // 15: symmetric case for genome B.
    {
        std::vector<NodeGene> nodes = {NodeGene{0, NodeType::Input}, NodeGene{1, NodeType::Input},
                                        NodeGene{2, NodeType::Output}, NodeGene{3, NodeType::Output}};
        std::vector<ConnectionGene> connections = {ConnectionGene{0, 2, 0.1f, true, 5},
                                                    ConnectionGene{1, 3, 0.2f, true, 5}};
        Genome bWithDuplicateInnovation(nodes, connections);
        Genome a;

        CompatibilityConfig config;
        assert(throwsInvalidArgument([&]() { compatibilityDistance(a, bWithDuplicateInnovation, config); }) &&
               "genome B with duplicate connection innovation numbers must be rejected");
    }

    // 16 & 17: an A-only innovation at/below B's max is disjoint; above it, excess.
    {
        Genome a = makeNodeGenome(12);
        a.addConnection(ConnectionGene{0, 1, 0.1f, true, 1});  // matching
        a.addConnection(ConnectionGene{2, 3, 0.2f, true, 3});  // A-only, expect disjoint (3 <= maxB=15)
        a.addConnection(ConnectionGene{4, 5, 0.3f, true, 20}); // A-only, expect excess (20 > maxB=15)

        Genome b = makeNodeGenome(12);
        b.addConnection(ConnectionGene{0, 1, 0.1f, true, 1});   // matching
        b.addConnection(ConnectionGene{6, 7, 0.4f, true, 15}); // B-only, expect disjoint (15 <= maxA=20)

        CompatibilityConfig config;
        const CompatibilityBreakdown result = compatibilityBreakdown(a, b, config);
        assert(result.matching == 1 && "exactly one matching gene");
        assert(result.disjoint == 2 && "A's innovation 3 and B's innovation 15 must both be disjoint"); // 16
        assert(result.excess == 1 && "A's innovation 20 must be excess"); // 17
    }

    // 18 & 19: symmetric case -- a B-only innovation at/below A's max is
    // disjoint; a B-only innovation above A's max is excess.
    {
        Genome a = makeNodeGenome(12);
        a.addConnection(ConnectionGene{0, 1, 0.1f, true, 1});   // matching
        a.addConnection(ConnectionGene{2, 3, 0.2f, true, 15}); // A-only, expect disjoint (15 <= maxB=20)

        Genome b = makeNodeGenome(12);
        b.addConnection(ConnectionGene{0, 1, 0.1f, true, 1});  // matching
        b.addConnection(ConnectionGene{4, 5, 0.3f, true, 3});  // B-only, expect disjoint (3 <= maxA=15)
        b.addConnection(ConnectionGene{6, 7, 0.4f, true, 20}); // B-only, expect excess (20 > maxA=15)

        CompatibilityConfig config;
        const CompatibilityBreakdown result = compatibilityBreakdown(a, b, config);
        assert(result.matching == 1 && "exactly one matching gene");
        assert(result.disjoint == 2 && "A's innovation 15 and B's innovation 3 must both be disjoint"); // 18
        assert(result.excess == 1 && "B's innovation 20 must be excess"); // 19
    }

    // 20: a mixed matching/disjoint/excess case is classified correctly in
    // a single comparison.
    {
        Genome a = makeNodeGenome(12);
        a.addConnection(ConnectionGene{0, 1, 0.1f, true, 1});  // matching
        a.addConnection(ConnectionGene{2, 3, 0.2f, true, 2});  // matching
        a.addConnection(ConnectionGene{4, 5, 0.3f, true, 4});  // A-only, disjoint (4 <= maxB=6)
        a.addConnection(ConnectionGene{6, 7, 0.4f, true, 50}); // A-only, excess (50 > maxB=6)

        Genome b = makeNodeGenome(12);
        b.addConnection(ConnectionGene{0, 1, 0.5f, true, 1}); // matching
        b.addConnection(ConnectionGene{2, 3, 0.6f, true, 2}); // matching
        b.addConnection(ConnectionGene{8, 9, 0.7f, true, 6}); // B-only, disjoint (6 <= maxA=50)

        CompatibilityConfig config;
        const CompatibilityBreakdown result = compatibilityBreakdown(a, b, config);
        assert(result.matching == 2 && result.disjoint == 2 && result.excess == 1 &&
               "a mixed comparison must classify every gene into the correct bucket");
    }

    // 21 & 22: a genome pair whose larger connection count is below the
    // (default) normalization threshold uses N=1.
    {
        // 21: larger size 5.
        {
            Genome a = makeChainGenome(5);
            Genome b = makeChainGenome(3);
            CompatibilityConfig config;
            const CompatibilityBreakdown result = compatibilityBreakdown(a, b, config);
            assert(result.normalization == 1.0f && "a small genome pair (larger size 5) must use N=1");
            // matching={0,1,2} (W=0); A-only={3,4}, maxB=2, both excess.
            assert(result.matching == 3 && result.excess == 2 && result.disjoint == 0);
            assert(std::fabs(result.distance - 2.0f) < kEps && "distance must use N=1, not the genome size, here");
        }
        // 22: larger size 19 -- exactly one below the default threshold (20).
        {
            Genome a = makeChainGenome(19);
            Genome b = makeChainGenome(15);
            CompatibilityConfig config;
            const CompatibilityBreakdown result = compatibilityBreakdown(a, b, config);
            assert(result.normalization == 1.0f && "a genome pair one below the threshold must still use N=1");
            // matching={0..14} (15 genes, W=0); A-only={15,16,17,18}, maxB=14, all excess.
            assert(result.matching == 15 && result.excess == 4 && result.disjoint == 0);
            assert(std::fabs(result.distance - 4.0f) < kEps && "distance must use N=1 here, not N=19");
        }
    }

    // 23: a genome pair whose larger connection count is exactly at the
    // threshold uses N = larger genome size (not 1).
    {
        Genome a = makeChainGenome(20);
        Genome b = makeChainGenome(15);
        CompatibilityConfig config; // default threshold = 20
        const CompatibilityBreakdown result = compatibilityBreakdown(a, b, config);
        assert(result.normalization == 20.0f && "a genome pair exactly at the threshold must use N = larger size");
        // matching={0..14} (15 genes, W=0); A-only={15..19} (5), maxB=14, all excess.
        assert(result.matching == 15 && result.excess == 5 && result.disjoint == 0);
        assert(std::fabs(result.distance - 0.25f) < kEps && "distance must divide by N=20 here"); // 5/20 = 0.25
    }

    // 24: a genome pair well above the threshold normalizes correctly.
    {
        Genome a = makeChainGenome(25);
        Genome b = makeChainGenome(10);
        CompatibilityConfig config;
        const CompatibilityBreakdown result = compatibilityBreakdown(a, b, config);
        assert(result.normalization == 25.0f && "normalization must equal the larger genome's connection count");
        // matching={0..9} (10 genes, W=0); A-only={10..24} (15), maxB=9, all excess.
        assert(result.matching == 10 && result.excess == 15 && result.disjoint == 0);
        assert(std::fabs(result.distance - 0.6f) < kEps && "distance must divide by N=25 here"); // 15/25 = 0.6
    }

    // 25, 26, 27 & 28: custom coefficients affect the result, and setting
    // any one to zero removes exactly its own contribution -- one fixture
    // with known E=1, D=2, W=0.25, N=1.
    {
        Genome a = makeNodeGenome(12);
        a.addConnection(ConnectionGene{0, 1, 0.2f, true, 0}); // matching, weight diff 0.5
        a.addConnection(ConnectionGene{2, 3, 0.1f, true, 1}); // matching, weight diff 0.0
        a.addConnection(ConnectionGene{4, 5, 0.9f, true, 2}); // A-only, disjoint (2 <= maxB=4)
        a.addConnection(ConnectionGene{6, 7, 0.9f, true, 10}); // A-only, excess (10 > maxB=4)

        Genome b = makeNodeGenome(12);
        b.addConnection(ConnectionGene{0, 1, 0.7f, true, 0}); // matching, weight diff 0.5
        b.addConnection(ConnectionGene{2, 3, 0.1f, true, 1}); // matching, weight diff 0.0
        b.addConnection(ConnectionGene{8, 9, 0.9f, true, 4}); // B-only, disjoint (4 <= maxA=10)

        // matching=2 (W = (0.5+0.0)/2 = 0.25), disjoint=2 (A's 2 + B's 4),
        // excess=1 (A's 10), larger size = 4 (< default threshold 20) -> N=1.

        CompatibilityConfig defaultConfig; // c1=1.0, c2=1.0, c3=0.4
        const float defaultDistance = compatibilityDistance(a, b, defaultConfig);
        assert(std::fabs(defaultDistance - 3.1f) < kEps && "baseline distance must equal 1*1 + 1*2 + 0.4*0.25 = 3.1");

        CompatibilityConfig doubledExcess = defaultConfig;
        doubledExcess.excessCoefficient = 2.0f;
        assert(std::fabs(compatibilityDistance(a, b, doubledExcess) - 4.1f) < kEps &&
               "doubling excessCoefficient must add exactly one extra excess contribution"); // 25

        CompatibilityConfig zeroExcess = defaultConfig;
        zeroExcess.excessCoefficient = 0.0f;
        assert(std::fabs(compatibilityDistance(a, b, zeroExcess) - 2.1f) < kEps &&
               "a zero excessCoefficient must remove exactly the excess contribution"); // 26

        CompatibilityConfig zeroDisjoint = defaultConfig;
        zeroDisjoint.disjointCoefficient = 0.0f;
        assert(std::fabs(compatibilityDistance(a, b, zeroDisjoint) - 1.1f) < kEps &&
               "a zero disjointCoefficient must remove exactly the disjoint contribution"); // 27

        CompatibilityConfig zeroWeight = defaultConfig;
        zeroWeight.weightDifferenceCoefficient = 0.0f;
        assert(std::fabs(compatibilityDistance(a, b, zeroWeight) - 3.0f) < kEps &&
               "a zero weightDifferenceCoefficient must remove exactly the weight contribution"); // 28
    }

    // 29: no matching genes at all gives W=0.
    {
        Genome a = makeNodeGenome(4);
        a.addConnection(ConnectionGene{0, 1, 0.5f, true, 0});
        a.addConnection(ConnectionGene{2, 3, 0.9f, true, 1});
        Genome b = makeNodeGenome(4);
        b.addConnection(ConnectionGene{0, 1, 0.5f, true, 5});
        b.addConnection(ConnectionGene{2, 3, 0.9f, true, 6});

        CompatibilityConfig config;
        const CompatibilityBreakdown result = compatibilityBreakdown(a, b, config);
        assert(result.matching == 0 && result.averageWeightDifference == 0.0f &&
               "with no matching genes at all, W must be exactly 0");
    }

    // 30: an empty genome vs. a non-empty one classifies every gene of the
    // non-empty genome as excess.
    {
        Genome a;
        Genome b = makeNodeGenome(4);
        b.addConnection(ConnectionGene{0, 1, 0.5f, true, 0});
        b.addConnection(ConnectionGene{2, 3, 0.9f, true, 1});

        CompatibilityConfig config;
        const CompatibilityBreakdown result = compatibilityBreakdown(a, b, config);
        assert(result.matching == 0 && result.disjoint == 0 && result.excess == 2 &&
               "every gene of a non-empty genome compared against an empty one must be excess");
    }

    // 31 & 32: distance and breakdown counts are symmetric.
    {
        Genome a = makeNodeGenome(12);
        a.addConnection(ConnectionGene{0, 1, 0.2f, true, 0});
        a.addConnection(ConnectionGene{2, 3, 0.9f, true, 2});
        a.addConnection(ConnectionGene{4, 5, 0.9f, true, 10});

        Genome b = makeNodeGenome(12);
        b.addConnection(ConnectionGene{0, 1, 0.7f, true, 0});
        b.addConnection(ConnectionGene{8, 9, 0.9f, true, 4});

        CompatibilityConfig config;
        const float distanceAB = compatibilityDistance(a, b, config);
        const float distanceBA = compatibilityDistance(b, a, config);
        assert(std::fabs(distanceAB - distanceBA) < kEps && "distance(A,B) must equal distance(B,A)"); // 31

        const CompatibilityBreakdown breakdownAB = compatibilityBreakdown(a, b, config);
        const CompatibilityBreakdown breakdownBA = compatibilityBreakdown(b, a, config);
        assert(breakdownAB.matching == breakdownBA.matching && breakdownAB.disjoint == breakdownBA.disjoint &&
               breakdownAB.excess == breakdownBA.excess &&
               std::fabs(breakdownAB.averageWeightDifference - breakdownBA.averageWeightDifference) < kEps &&
               breakdownAB.normalization == breakdownBA.normalization &&
               "breakdown counts, W and normalization must all be symmetric"); // 32
    }

    // 33: the order genes/nodes were inserted into either parent does not
    // affect the result.
    {
        Genome inOrderA = makeNodeGenome(4);
        inOrderA.addConnection(ConnectionGene{0, 1, 0.3f, true, 0});
        inOrderA.addConnection(ConnectionGene{2, 3, 0.6f, true, 1});
        Genome inOrderB = makeNodeGenome(4);
        inOrderB.addConnection(ConnectionGene{0, 1, 0.8f, true, 0});
        inOrderB.addConnection(ConnectionGene{2, 3, 0.6f, true, 1});

        CompatibilityConfig config;
        const float canonical = compatibilityDistance(inOrderA, inOrderB, config);

        // Same final gene set, built in a different order (nodes added
        // high-to-low, connections added in reverse innovation order).
        using ai::neat::Genome;
        using ai::neat::NodeGene;
        using ai::neat::NodeType;
        Genome shuffledA;
        shuffledA.addNode(NodeGene{3, NodeType::Output});
        shuffledA.addNode(NodeGene{1, NodeType::Output});
        shuffledA.addNode(NodeGene{2, NodeType::Input});
        shuffledA.addNode(NodeGene{0, NodeType::Input});
        shuffledA.addConnection(ConnectionGene{2, 3, 0.6f, true, 1});
        shuffledA.addConnection(ConnectionGene{0, 1, 0.3f, true, 0});

        Genome shuffledB;
        shuffledB.addNode(NodeGene{2, NodeType::Input});
        shuffledB.addNode(NodeGene{0, NodeType::Input});
        shuffledB.addNode(NodeGene{3, NodeType::Output});
        shuffledB.addNode(NodeGene{1, NodeType::Output});
        shuffledB.addConnection(ConnectionGene{2, 3, 0.6f, true, 1});
        shuffledB.addConnection(ConnectionGene{0, 1, 0.8f, true, 0});

        const float shuffled = compatibilityDistance(shuffledA, shuffledB, config);
        assert(std::fabs(canonical - shuffled) < kEps && "insertion order must not affect the computed distance");
    }

    // 34: node-only differences (extra unreferenced Hidden nodes) don't
    // affect the connection-based distance.
    {
        Genome baselineA = makeNodeGenome(2);
        baselineA.addConnection(ConnectionGene{0, 1, 0.4f, true, 0});
        Genome baselineB = makeNodeGenome(2);
        baselineB.addConnection(ConnectionGene{0, 1, 0.9f, true, 0});
        CompatibilityConfig config;
        const float baselineDistance = compatibilityDistance(baselineA, baselineB, config);

        Genome withExtraNodesA = makeNodeGenome(2);
        withExtraNodesA.addNode(NodeGene{50, NodeType::Hidden});
        withExtraNodesA.addConnection(ConnectionGene{0, 1, 0.4f, true, 0});
        Genome withExtraNodesB = makeNodeGenome(2);
        withExtraNodesB.addNode(NodeGene{60, NodeType::Hidden}); // different extra node ID entirely
        withExtraNodesB.addConnection(ConnectionGene{0, 1, 0.9f, true, 0});

        const float extraNodesDistance = compatibilityDistance(withExtraNodesA, withExtraNodesB, config);
        assert(std::fabs(baselineDistance - extraNodesDistance) < kEps &&
               "node-only differences must not affect the connection-based compatibility distance");
    }

    // 35 & 36: neither genome's node or connection genes are ever modified.
    {
        Genome a = makeNodeGenome(4);
        a.addNode(NodeGene{50, NodeType::Hidden});
        a.addConnection(ConnectionGene{0, 1, 0.3f, true, 0});
        a.addConnection(ConnectionGene{2, 50, 0.6f, false, 5});

        Genome b = makeNodeGenome(4);
        b.addConnection(ConnectionGene{0, 1, 0.8f, true, 0});
        b.addConnection(ConnectionGene{2, 3, 0.6f, true, 9});

        const std::vector<NodeGene> nodesABefore = a.nodes();
        const std::vector<ConnectionGene> connectionsABefore = a.connections();
        const std::vector<NodeGene> nodesBBefore = b.nodes();
        const std::vector<ConnectionGene> connectionsBBefore = b.connections();

        CompatibilityConfig config;
        compatibilityDistance(a, b, config);

        assert(a.nodes().size() == nodesABefore.size() && "compatibilityDistance must not add/remove genome A's nodes");
        for (std::size_t i = 0; i < nodesABefore.size(); ++i)
        {
            assert(a.nodes()[i] == nodesABefore[i] && "compatibilityDistance must not modify genome A's nodes"); // 35
        }
        assert(a.connections().size() == connectionsABefore.size() &&
               "compatibilityDistance must not add/remove genome A's connections");
        for (std::size_t i = 0; i < connectionsABefore.size(); ++i)
        {
            const ConnectionGene& before = connectionsABefore[i];
            const ConnectionGene& after = a.connections()[i];
            assert(after.getSourceId() == before.getSourceId() && after.getTargetId() == before.getTargetId() &&
                   after.getWeight() == before.getWeight() && after.isEnabled() == before.isEnabled() &&
                   after.getInnovationNumber() == before.getInnovationNumber() &&
                   "compatibilityDistance must not modify genome A's connections"); // 36
        }

        assert(b.nodes().size() == nodesBBefore.size() && "compatibilityDistance must not add/remove genome B's nodes");
        for (std::size_t i = 0; i < nodesBBefore.size(); ++i)
        {
            assert(b.nodes()[i] == nodesBBefore[i] && "compatibilityDistance must not modify genome B's nodes");
        }
        assert(b.connections().size() == connectionsBBefore.size() &&
               "compatibilityDistance must not add/remove genome B's connections");
        for (std::size_t i = 0; i < connectionsBBefore.size(); ++i)
        {
            const ConnectionGene& before = connectionsBBefore[i];
            const ConnectionGene& after = b.connections()[i];
            assert(after.getSourceId() == before.getSourceId() && after.getTargetId() == before.getTargetId() &&
                   after.getWeight() == before.getWeight() && after.isEnabled() == before.isEnabled() &&
                   after.getInnovationNumber() == before.getInnovationNumber() &&
                   "compatibilityDistance must not modify genome B's connections");
        }
    }

    // 37: no InnovationTracker state involved -- compatibilityDistance()'s signature takes no tracker.
    // 38: no RNG state involved -- a pure function of its three arguments.

    // 39: repeated calls with identical inputs produce an identical result.
    {
        Genome a = makeNodeGenome(4);
        a.addConnection(ConnectionGene{0, 1, 0.3f, true, 0});
        a.addConnection(ConnectionGene{2, 3, 0.6f, true, 5});
        Genome b = makeNodeGenome(4);
        b.addConnection(ConnectionGene{0, 1, 0.8f, true, 0});

        CompatibilityConfig config;
        const float first = compatibilityDistance(a, b, config);
        const float second = compatibilityDistance(a, b, config);
        const float third = compatibilityDistance(a, b, config);
        assert(first == second && second == third && "repeated calls with identical inputs must produce identical results");
    }

    // 40: all previous verification suites still pass.

    TraceLog(LOG_INFO, "Compatibility distance verification: all deterministic checks passed");
}

namespace speciation_verify
{

// A 2-node genome with connection 0->1 (innovation 0, given weight). Two
// such genomes always match with no disjoint/excess, so distance =
// 0.4 * |weightA - weightB| with the default config -- predictable
// arithmetic for driving Speciator's threshold.
ai::neat::Genome makeSimpleGenome(float weight)
{
    using namespace compatibility_verify;
    using ai::neat::ConnectionGene;
    using ai::neat::Genome;

    Genome genome = makeNodeGenome(2);
    genome.addConnection(ConnectionGene{0, 1, weight, true, 0});
    return genome;
}

} // namespace speciation_verify

// Deterministic check of ai::neat::Speciator/Species species-membership
// assignment, end to end. No Population/fitness/reproduction/mutation
// involved -- only species assignment. Persistent cross-generation
// identity/stagnation behavior has its own suite,
// verifyPersistentSpeciesAndStagnation() below (items 28 & 29 here assert
// the persistent-identity behavior specifically; every other test calls
// speciate() only once per Speciator, so persistence doesn't affect them).
void verifySpeciation()
{
    using namespace compatibility_verify;
    using namespace speciation_verify;
    using ai::neat::CompatibilityConfig;
    using ai::neat::ConnectionGene;
    using ai::neat::Genome;
    using ai::neat::NodeGene;
    using ai::neat::NodeType;
    using ai::neat::SpeciationConfig;
    using ai::neat::Speciator;
    using ai::neat::Species;

    // 1 & 2: an invalid (negative or non-finite) compatibilityThreshold is
    // rejected, and never advances the SpeciesId counter.
    {
        std::vector<Genome> genomes = {makeSimpleGenome(0.0f)};
        Speciator speciator;
        CompatibilityConfig compatConfig;

        SpeciationConfig negativeThreshold;
        negativeThreshold.compatibilityThreshold = -1.0f;
        assert(throwsInvalidArgument([&]() { speciator.speciate(genomes, compatConfig, negativeThreshold); }) &&
               "a negative compatibilityThreshold must be rejected"); // 1

        SpeciationConfig nanThreshold;
        nanThreshold.compatibilityThreshold = std::numeric_limits<float>::quiet_NaN();
        assert(throwsInvalidArgument([&]() { speciator.speciate(genomes, compatConfig, nanThreshold); }) &&
               "a NaN compatibilityThreshold must be rejected"); // 2

        SpeciationConfig infThreshold;
        infThreshold.compatibilityThreshold = std::numeric_limits<float>::infinity();
        assert(throwsInvalidArgument([&]() { speciator.speciate(genomes, compatConfig, infThreshold); }) &&
               "a non-finite compatibilityThreshold must be rejected"); // 2 (continued)

        SpeciationConfig validConfig;
        const std::vector<Species> species = speciator.speciate(genomes, compatConfig, validConfig);
        assert(species.size() == 1 && species[0].getId() == 0 &&
               "earlier rejected calls must never have advanced the SpeciesId counter");
    }

    // 3 & 4: empty input returns no species and consumes no SpeciesId.
    {
        std::vector<Genome> empty;
        Speciator speciator;
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig;

        const std::vector<Species> species = speciator.speciate(empty, compatConfig, speciationConfig);
        assert(species.empty() && "an empty genome vector must produce no species"); // 3

        std::vector<Genome> nonEmpty = {makeSimpleGenome(0.0f)};
        const std::vector<Species> species2 = speciator.speciate(nonEmpty, compatConfig, speciationConfig);
        assert(species2.size() == 1 && species2[0].getId() == 0 &&
               "an empty call must not consume a SpeciesId"); // 4
    }

    // 5, 6 & 7: one genome creates exactly one species, with the first
    // available SpeciesId, whose representative equals the founding genome.
    {
        Genome g = makeSimpleGenome(0.5f);
        std::vector<Genome> genomes = {g};
        Speciator speciator;
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig;

        const std::vector<Species> species = speciator.speciate(genomes, compatConfig, speciationConfig);
        assert(species.size() == 1 && "one genome must create exactly one species"); // 5
        assert(species[0].getId() == 0 && "the first species must get SpeciesId 0"); // 6

        const ConnectionGene* repConnection = species[0].getRepresentative().findConnection(0, 1);
        assert(repConnection != nullptr && repConnection->getWeight() == 0.5f &&
               "the representative must equal the genome that founded the species"); // 7
    }

    // 8: the representative is a copy, not a reference -- mutating the
    // original genome afterward must not affect it.
    {
        std::vector<Genome> genomes = {makeSimpleGenome(0.5f)};
        Speciator speciator;
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig;

        const std::vector<Species> species = speciator.speciate(genomes, compatConfig, speciationConfig);
        genomes[0].mutableConnections()[0].setWeight(999.0f);
        assert(species[0].getRepresentative().findConnection(0, 1)->getWeight() == 0.5f &&
               "the representative must be an independent copy, unaffected by later changes to the source genome");
    }

    // 9: identical genomes join the same species.
    {
        std::vector<Genome> genomes = {makeSimpleGenome(0.5f), makeSimpleGenome(0.5f)};
        Speciator speciator;
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig;

        const std::vector<Species> species = speciator.speciate(genomes, compatConfig, speciationConfig);
        assert(species.size() == 1 && species[0].size() == 2 && "identical genomes must join the same species");
    }

    // 10: a distance exactly equal to the threshold still qualifies.
    {
        std::vector<Genome> genomes = {makeSimpleGenome(0.0f), makeSimpleGenome(2.5f)}; // distance = 0.4*2.5 = 1.0
        Speciator speciator;
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig;
        speciationConfig.compatibilityThreshold = 1.0f;

        const std::vector<Species> species = speciator.speciate(genomes, compatConfig, speciationConfig);
        assert(species.size() == 1 && species[0].size() == 2 &&
               "a distance exactly equal to the threshold must still qualify for membership");
    }

    // 11: a distance above the threshold creates a new species.
    {
        std::vector<Genome> genomes = {makeSimpleGenome(0.0f), makeSimpleGenome(3.0f)}; // distance = 0.4*3.0 = 1.2
        Speciator speciator;
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig;
        speciationConfig.compatibilityThreshold = 1.0f;

        const std::vector<Species> species = speciator.speciate(genomes, compatConfig, speciationConfig);
        assert(species.size() == 2 && species[0].getId() == 0 && species[1].getId() == 1 &&
               "a distance above the threshold must create a new species");
    }

    // 12: sufficiently different genomes can create more than two species.
    {
        std::vector<Genome> genomes = {makeSimpleGenome(0.0f), makeSimpleGenome(10.0f), makeSimpleGenome(20.0f)};
        Speciator speciator;
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig; // default threshold 3.0

        const std::vector<Species> species = speciator.speciate(genomes, compatConfig, speciationConfig);
        assert(species.size() == 3 && "sufficiently different genomes must be able to create multiple species");
    }

    // 13 & 14: member indices are exactly the assigned genomes' original
    // indices, stored in ascending (input) order.
    {
        std::vector<Genome> genomes = {makeSimpleGenome(0.0f), makeSimpleGenome(0.1f), makeSimpleGenome(10.0f),
                                        makeSimpleGenome(0.05f)};
        Speciator speciator;
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig; // default threshold 3.0

        const std::vector<Species> species = speciator.speciate(genomes, compatConfig, speciationConfig);
        assert(species.size() == 2 && "this fixture must produce exactly two species");
        const std::vector<std::size_t> expectedSpeciesZeroMembers = {0, 1, 3};
        const std::vector<std::size_t> expectedSpeciesOneMembers = {2};
        assert(species[0].getMemberIndices() == expectedSpeciesZeroMembers &&
               "member indices must exactly match the genomes assigned to this species"); // 13
        assert(species[1].getMemberIndices() == expectedSpeciesOneMembers &&
               "member indices must exactly match the genomes assigned to this species"); // 13 (continued)
        // 14: {0,1,3} is already strictly ascending -- input-order sorted.
        for (std::size_t i = 1; i < species[0].getMemberIndices().size(); ++i)
        {
            assert(species[0].getMemberIndices()[i - 1] < species[0].getMemberIndices()[i] &&
                   "member indices must remain in ascending input-genome-index order");
        }
    }

    // 15: the returned species vector is sorted ascending by SpeciesId.
    {
        std::vector<Genome> genomes = {makeSimpleGenome(0.0f), makeSimpleGenome(10.0f), makeSimpleGenome(20.0f)};
        Speciator speciator;
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig;

        const std::vector<Species> species = speciator.speciate(genomes, compatConfig, speciationConfig);
        for (std::size_t i = 1; i < species.size(); ++i)
        {
            assert(species[i - 1].getId() < species[i].getId() &&
                   "the returned species vector must be sorted ascending by SpeciesId");
        }
    }

    // 16 & 17: first-match rule -- a genome compatible with more than one
    // existing species joins the FIRST (lowest-SpeciesId) one, even when a
    // later species is a strictly closer match; nearest-species search is
    // never used.
    {
        // rep0=0.0, rep1=10.0 (founds species1). g2=7.0 qualifies for BOTH
        // (dist to rep0=2.8, to rep1=1.2, the closer match) -- first-match
        // must still place it in species0.
        std::vector<Genome> genomes = {makeSimpleGenome(0.0f), makeSimpleGenome(10.0f), makeSimpleGenome(7.0f)};
        Speciator speciator;
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig; // default threshold 3.0

        const std::vector<Species> species = speciator.speciate(genomes, compatConfig, speciationConfig);
        assert(species.size() == 2 && "this fixture must produce exactly two species");
        const std::vector<std::size_t> expectedSpeciesZeroMembers = {0, 2};
        assert(species[0].getMemberIndices() == expectedSpeciesZeroMembers &&
               "genome 2 must join species 0 (first match), even though species 1's representative is a strictly "
               "closer match -- nearest-species search is not implemented"); // 16 & 17
        assert(species[1].getMemberIndices().size() == 1);
    }

    // 18: input genome order can change the resulting grouping under
    // first-match semantics -- the same three genomes, reordered, produce
    // a different membership split because a different genome ends up
    // founding (and fixing the representative of) the first species.
    {
        // A=0.0, B=3.0 (dist(A,B)=1.2), C=5.5 (dist(A,C)=2.2, dist(B,C)=1.0), threshold=1.5.
        const float weightA = 0.0f;
        const float weightB = 3.0f;
        const float weightC = 5.5f;
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig;
        speciationConfig.compatibilityThreshold = 1.5f;

        // [A,B,C]: species0 founded by A; B joins (dist(A,B)=1.2); C
        // compared against rep A (dist=2.2 > 1.5) -> founds species1.
        {
            std::vector<Genome> genomes = {makeSimpleGenome(weightA), makeSimpleGenome(weightB), makeSimpleGenome(weightC)};
            Speciator speciator;
            const std::vector<Species> species = speciator.speciate(genomes, compatConfig, speciationConfig);
            assert(species.size() == 2 && species[0].size() == 2 && species[1].size() == 1 &&
                   "order [A,B,C] must split C into its own species");
        }

        // [B,A,C]: species0 founded by B; A joins (symmetric); C compared
        // against rep B (dist=1.0 <= 1.5) -> also joins.
        {
            std::vector<Genome> genomes = {makeSimpleGenome(weightB), makeSimpleGenome(weightA), makeSimpleGenome(weightC)};
            Speciator speciator;
            const std::vector<Species> species = speciator.speciate(genomes, compatConfig, speciationConfig);
            assert(species.size() == 1 && species[0].size() == 3 &&
                   "order [B,A,C] must group all three into one species -- a different result from [A,B,C] for the "
                   "same genomes, purely because a different genome founded the first species");
        }
    }

    // 19: a fresh Speciator run with identical input produces an identical
    // result to another fresh Speciator run with the same input.
    {
        std::vector<Genome> genomes = {makeSimpleGenome(0.0f), makeSimpleGenome(1.0f), makeSimpleGenome(10.0f)};
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig;

        Speciator speciatorX;
        const std::vector<Species> speciesX = speciatorX.speciate(genomes, compatConfig, speciationConfig);
        Speciator speciatorY;
        const std::vector<Species> speciesY = speciatorY.speciate(genomes, compatConfig, speciationConfig);

        assert(speciesX.size() == speciesY.size() && "repeated identical runs must produce the same species count");
        for (std::size_t i = 0; i < speciesX.size(); ++i)
        {
            assert(speciesX[i].getId() == speciesY[i].getId() && speciesX[i].getMemberIndices() == speciesY[i].getMemberIndices() &&
                   "repeated identical runs must produce identical species IDs and membership");
        }
    }

    // 20: no RNG state anywhere in Speciator/Species -- Speciator's
    // constructor takes no seed; determinism above (19) follows directly.

    // 21: input genomes are never modified by speciate().
    {
        Genome g1 = makeSimpleGenome(0.5f);
        Genome g2 = makeSimpleGenome(3.5f);
        std::vector<Genome> genomes = {g1, g2};
        const std::vector<NodeGene> nodes0Before = genomes[0].nodes();
        const std::vector<ConnectionGene> connections0Before = genomes[0].connections();
        const std::vector<NodeGene> nodes1Before = genomes[1].nodes();
        const std::vector<ConnectionGene> connections1Before = genomes[1].connections();

        Speciator speciator;
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig;
        speciator.speciate(genomes, compatConfig, speciationConfig);

        assert(genomes[0].nodes() == nodes0Before && genomes[1].nodes() == nodes1Before &&
               "speciate() must not modify any input genome's nodes");
        auto connectionsMatch = [](const std::vector<ConnectionGene>& after, const std::vector<ConnectionGene>& before)
        {
            if (after.size() != before.size())
            {
                return false;
            }
            for (std::size_t i = 0; i < before.size(); ++i)
            {
                if (after[i].getSourceId() != before[i].getSourceId() || after[i].getTargetId() != before[i].getTargetId() ||
                    after[i].getWeight() != before[i].getWeight() || after[i].isEnabled() != before[i].isEnabled() ||
                    after[i].getInnovationNumber() != before[i].getInnovationNumber())
                {
                    return false;
                }
            }
            return true;
        };
        assert(connectionsMatch(genomes[0].connections(), connections0Before) &&
               connectionsMatch(genomes[1].connections(), connections1Before) &&
               "speciate() must not modify any input genome's connections");
    }

    // 22 & 23: the representative stays fixed at the founding genome's copy
    // -- a later compatible member never overwrites or blends into it.
    {
        std::vector<Genome> genomes = {makeSimpleGenome(0.0f), makeSimpleGenome(0.1f)};
        Speciator speciator;
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig;

        const std::vector<Species> species = speciator.speciate(genomes, compatConfig, speciationConfig);
        assert(species.size() == 1 && species[0].size() == 2 && "both genomes must join the same species");
        assert(species[0].getRepresentative().findConnection(0, 1)->getWeight() == 0.0f &&
               "the representative must remain the founding genome's own weight, unaffected by the later member "
               "joining"); // 22 & 23
    }

    // 24 & 25: CompatibilityConfig is honored -- custom coefficients
    // change whether two genomes are considered compatible.
    {
        std::vector<Genome> genomes = {makeSimpleGenome(0.0f), makeSimpleGenome(2.0f)}; // weight diff = 2.0
        SpeciationConfig speciationConfig;
        speciationConfig.compatibilityThreshold = 1.0f;

        CompatibilityConfig lowWeightCoefficient;
        lowWeightCoefficient.weightDifferenceCoefficient = 0.1f; // distance = 0.1*2.0 = 0.2 <= 1.0
        Speciator speciatorLow;
        const std::vector<Species> speciesLow = speciatorLow.speciate(genomes, lowWeightCoefficient, speciationConfig);
        assert(speciesLow.size() == 1 &&
               "with a low weightDifferenceCoefficient, the genomes must be compatible"); // 24

        CompatibilityConfig highWeightCoefficient;
        highWeightCoefficient.weightDifferenceCoefficient = 1.0f; // distance = 1.0*2.0 = 2.0 > 1.0
        Speciator speciatorHigh;
        const std::vector<Species> speciesHigh = speciatorHigh.speciate(genomes, highWeightCoefficient, speciationConfig);
        assert(speciesHigh.size() == 2 &&
               "with a high weightDifferenceCoefficient, the same genomes must become incompatible"); // 25
    }

    // 26: a zero threshold groups only exactly-zero-distance genomes
    // together.
    {
        std::vector<Genome> genomes = {makeSimpleGenome(0.5f), makeSimpleGenome(0.5f), makeSimpleGenome(0.6f)};
        Speciator speciator;
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig;
        speciationConfig.compatibilityThreshold = 0.0f;

        const std::vector<Species> species = speciator.speciate(genomes, compatConfig, speciationConfig);
        assert(species.size() == 2 && species[0].size() == 2 && species[1].size() == 1 &&
               "a zero threshold must only group exactly-zero-distance genomes together");
    }

    // 27: a very large threshold groups all compatible genomes into one
    // species.
    {
        std::vector<Genome> genomes = {makeSimpleGenome(0.0f), makeSimpleGenome(100.0f), makeSimpleGenome(-50.0f)};
        Speciator speciator;
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig;
        speciationConfig.compatibilityThreshold = 1000.0f;

        const std::vector<Species> species = speciator.speciate(genomes, compatConfig, speciationConfig);
        assert(species.size() == 1 && species[0].size() == 3 &&
               "a very large threshold must group all genomes into a single species");
    }

    // 28 & 29: SpeciesIds increase monotonically across speciate() calls on
    // the same Speciator, extinct ids are never reused, and a genome
    // compatible with a persistent species' representative rejoins that
    // SAME SpeciesId rather than founding a new one.
    {
        Speciator speciator;
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig;

        std::vector<Genome> firstBatch = {makeSimpleGenome(0.0f), makeSimpleGenome(10.0f), makeSimpleGenome(20.0f)};
        const std::vector<Species> firstSpecies = speciator.speciate(firstBatch, compatConfig, speciationConfig);
        assert(firstSpecies.size() == 3 && firstSpecies[0].getId() == 0 && firstSpecies[1].getId() == 1 &&
               firstSpecies[2].getId() == 2 && "the first call must allocate SpeciesIds 0, 1, 2");

        // weight 0.0 rejoins SpeciesId 0 (still compatible with its rep);
        // weight 30.0 matches none of the old reps, founds id 3. Species 1
        // and 2 go extinct this pass -- their ids are never reused.
        std::vector<Genome> secondBatch = {makeSimpleGenome(0.0f), makeSimpleGenome(30.0f)};
        const std::vector<Species> secondSpecies = speciator.speciate(secondBatch, compatConfig, speciationConfig);
        assert(secondSpecies.size() == 2 && secondSpecies[0].getId() == 0 && secondSpecies[1].getId() == 3 &&
               "a genome compatible with a persistent species must rejoin its existing SpeciesId, while a "
               "genuinely new species still receives the next monotonically increasing id"); // 28 & 29
    }

    // 30: an empty call sandwiched between non-empty calls does not
    // advance the SpeciesId counter.
    {
        Speciator speciator;
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig;

        std::vector<Genome> firstBatch = {makeSimpleGenome(0.0f)};
        const std::vector<Species> firstSpecies = speciator.speciate(firstBatch, compatConfig, speciationConfig);
        assert(firstSpecies.size() == 1 && firstSpecies[0].getId() == 0);

        std::vector<Genome> emptyBatch;
        const std::vector<Species> emptySpecies = speciator.speciate(emptyBatch, compatConfig, speciationConfig);
        assert(emptySpecies.empty());

        std::vector<Genome> secondBatch = {makeSimpleGenome(50.0f)};
        const std::vector<Species> secondSpecies = speciator.speciate(secondBatch, compatConfig, speciationConfig);
        assert(secondSpecies.size() == 1 && secondSpecies[0].getId() == 1 &&
               "an empty call between non-empty calls must not consume a SpeciesId");
    }

    // 31 & 32: size()/empty() report member counts correctly.
    {
        std::vector<Genome> genomes = {makeSimpleGenome(0.0f), makeSimpleGenome(0.1f), makeSimpleGenome(10.0f)};
        Speciator speciator;
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig;
        const std::vector<Species> species = speciator.speciate(genomes, compatConfig, speciationConfig);

        assert(species.size() == 2 && "this fixture must produce exactly two species");
        assert(species[0].size() == 2 && !species[0].empty() && "size()/empty() must reflect two members"); // 31 & 32
        assert(species[1].size() == 1 && !species[1].empty() && "size()/empty() must reflect one member");

        const Species freshlyConstructed(0, makeSimpleGenome(0.0f));
        assert(freshlyConstructed.size() == 0 && freshlyConstructed.empty() &&
               "a species with no members added yet must report empty()"); // 32 (continued)
    }

    // 33 & 35: a structural conflict compatibilityDistance() would reject
    // (conflicting matching endpoints) propagates unchanged out of
    // speciate() rather than being silently skipped.
    {
        Genome g1 = makeNodeGenome(4);
        g1.addConnection(ConnectionGene{0, 1, 0.5f, true, 7});
        Genome g2 = makeNodeGenome(4);
        g2.addConnection(ConnectionGene{2, 3, 0.5f, true, 7}); // same innovation, different endpoints
        std::vector<Genome> genomes = {g1, g2};

        Speciator speciator;
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig;
        assert(throwsInvalidArgument([&]() { speciator.speciate(genomes, compatConfig, speciationConfig); }) &&
               "a conflicting-endpoints structural error from compatibilityDistance() must propagate out of "
               "speciate(), not be silently skipped"); // 33 & 35
    }

    // 34: duplicate connection innovation numbers inside a genome
    // propagate as a rejection -- every genome is validated
    // unconditionally, whether or not it's ever compared against a
    // representative.
    {
        std::vector<NodeGene> nodes = {NodeGene{0, NodeType::Input}, NodeGene{1, NodeType::Input},
                                        NodeGene{2, NodeType::Output}, NodeGene{3, NodeType::Output}};
        std::vector<ConnectionGene> connections = {ConnectionGene{0, 2, 0.1f, true, 5},
                                                    ConnectionGene{1, 3, 0.2f, true, 5}};
        Genome genomeWithDuplicateInnovation(nodes, connections);
        std::vector<Genome> genomes = {makeSimpleGenome(0.0f), genomeWithDuplicateInnovation};

        Speciator speciator;
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig;
        assert(throwsInvalidArgument([&]() { speciator.speciate(genomes, compatConfig, speciationConfig); }) &&
               "a genome with duplicate connection innovation numbers must cause speciate() to throw once compared");
    }

    // Every genome is validated (genome.validate() plus a self
    // compatibilityDistance() innovation-uniqueness check) before it can
    // join or found a species -- including the very first genome
    // processed, which has no existing representative to compare against.

    // A single (lone) structurally invalid genome is rejected even as the
    // founding genome, with no comparison to trigger validation otherwise.
    {
        std::vector<NodeGene> nodes = {NodeGene{0, NodeType::Input}};
        std::vector<ConnectionGene> connections = {ConnectionGene{0, 99, 0.1f, true, 0}}; // target node 99 does not exist
        Genome invalidGenome(nodes, connections);
        std::vector<Genome> genomes = {invalidGenome};

        Speciator speciator;
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig;
        assert(throwsInvalidArgument([&]() { speciator.speciate(genomes, compatConfig, speciationConfig); }) &&
               "a single structurally invalid genome (Genome::validate() failure) must be rejected even as the "
               "lone founding genome"); // correction 1
    }

    // A single (lone) genome with duplicate connection innovation numbers is rejected the same way.
    {
        std::vector<NodeGene> nodes = {NodeGene{0, NodeType::Input}, NodeGene{1, NodeType::Input},
                                        NodeGene{2, NodeType::Output}, NodeGene{3, NodeType::Output}};
        std::vector<ConnectionGene> connections = {ConnectionGene{0, 2, 0.1f, true, 5},
                                                    ConnectionGene{1, 3, 0.2f, true, 5}}; // duplicate innovation 5
        Genome genomeWithDuplicateInnovation(nodes, connections);
        std::vector<Genome> genomes = {genomeWithDuplicateInnovation};

        Speciator speciator;
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig;
        assert(throwsInvalidArgument([&]() { speciator.speciate(genomes, compatConfig, speciationConfig); }) &&
               "a single genome with duplicate connection innovation numbers must be rejected even as the lone "
               "founding genome"); // correction 2
    }

    // A rejected invalid first genome must not consume a SpeciesId -- a
    // subsequent valid call must still start at SpeciesId 0.
    {
        std::vector<NodeGene> nodes = {NodeGene{0, NodeType::Input}};
        std::vector<ConnectionGene> connections = {ConnectionGene{0, 99, 0.1f, true, 0}};
        Genome invalidGenome(nodes, connections);

        Speciator speciator;
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig;

        assert(throwsInvalidArgument([&]()
                                      {
                                          std::vector<Genome> invalidBatch = {invalidGenome};
                                          speciator.speciate(invalidBatch, compatConfig, speciationConfig);
                                      }) &&
               "setup: the invalid genome must be rejected");

        std::vector<Genome> validBatch = {makeSimpleGenome(0.0f)};
        const std::vector<Species> species = speciator.speciate(validBatch, compatConfig, speciationConfig);
        assert(species.size() == 1 && species[0].getId() == 0 &&
               "a rejected invalid first genome must not have consumed a SpeciesId"); // correction 3
    }

    // An invalid LATER genome still causes speciate() to throw -- the
    // exception aborts before returning, so no partial species vector is ever observed.
    {
        Genome valid = makeSimpleGenome(0.0f);
        std::vector<NodeGene> nodes = {NodeGene{0, NodeType::Input}};
        std::vector<ConnectionGene> connections = {ConnectionGene{0, 99, 0.1f, true, 0}};
        Genome invalidGenome(nodes, connections);
        std::vector<Genome> genomes = {valid, invalidGenome};

        Speciator speciator;
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig;
        assert(throwsInvalidArgument([&]() { speciator.speciate(genomes, compatConfig, speciationConfig); }) &&
               "an invalid later genome must still cause speciate() to throw, with no partial result returned"); // correction 4
    }

    // A valid lone genome still creates exactly one species, normally.
    {
        std::vector<Genome> genomes = {makeSimpleGenome(0.5f)};
        Speciator speciator;
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig;
        const std::vector<Species> species = speciator.speciate(genomes, compatConfig, speciationConfig);
        assert(species.size() == 1 && species[0].getId() == 0 && species[0].size() == 1 &&
               "a valid lone genome must still create exactly one species"); // correction 5
    }

    // All earlier assertions in this function ran unmodified before reaching this point.
    // 36: Species holds no fitness data -- only id, representative, memberIndices.
    // 37: no genome is ever mutated by speciate() (reinforced by 21, 8/22/23 above).
    // 38: no crossover occurs -- GenomeCrossover is never referenced by Species/Speciator.
    // 39: no mutation occurs -- GenomeMutator is never referenced by Species/Speciator.
    // 40: no population/reproduction behavior -- Speciator only assigns membership.
    // 41: Speciator/Species are exercised only from this function, never from main()'s loop.
    // 42: all previous verification suites still pass.

    TraceLog(LOG_INFO, "Speciation verification: all deterministic checks passed");
}
} // namespace verification
