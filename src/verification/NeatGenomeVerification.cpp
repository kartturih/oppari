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


// Deterministic check of the NEAT gene definitions (NodeGene,
// ConnectionGene) -- pure data structures, no runtime network involved.
void verifyNeatGenes()
{
    using ai::neat::ConnectionGene;
    using ai::neat::NodeGene;
    using ai::neat::NodeType;

    auto throwsInvalidArgument = [](auto&& callable) -> bool
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
    };

    // 1: valid construction for every node type, with getters reporting back
    // exactly what was passed in.
    {
        const NodeGene input(0, NodeType::Input);
        const NodeGene bias(1, NodeType::Bias);
        const NodeGene hidden(2, NodeType::Hidden);
        const NodeGene output(3, NodeType::Output);
        assert(input.getId() == 0 && input.getType() == NodeType::Input && "Input node gene must round-trip");
        assert(bias.getId() == 1 && bias.getType() == NodeType::Bias && "Bias node gene must round-trip");
        assert(hidden.getId() == 2 && hidden.getType() == NodeType::Hidden && "Hidden node gene must round-trip");
        assert(output.getId() == 3 && output.getType() == NodeType::Output && "Output node gene must round-trip");
    }

    // 2: a negative node ID is rejected.
    {
        assert(throwsInvalidArgument([]() { NodeGene bad(-1, NodeType::Hidden); }) &&
               "a negative node ID must be rejected");
    }

    // 3: equality compares both ID and type; either differing breaks equality.
    {
        const NodeGene a(5, NodeType::Hidden);
        const NodeGene b(5, NodeType::Hidden);
        const NodeGene differentId(6, NodeType::Hidden);
        const NodeGene differentType(5, NodeType::Output);
        assert(a == b && "identical node genes must compare equal");
        assert(a != differentId && "node genes with different IDs must not compare equal");
        assert(a != differentType && "node genes with different types must not compare equal");
    }

    // 4: valid construction preserves source, target, weight, enabled flag,
    // and innovation number exactly.
    {
        const ConnectionGene c(0, 1, 0.75f, true, 3);
        assert(c.getSourceId() == 0 && c.getTargetId() == 1 && "source/target must round-trip");
        assert(c.getWeight() == 0.75f && "weight must be preserved exactly, not clamped or rounded");
        assert(c.isEnabled() && "enabled flag must round-trip");
        assert(c.getInnovationNumber() == 3 && "innovation number must round-trip");
    }

    // 5: a negative weight is preserved exactly (weights are never clamped).
    {
        const ConnectionGene c(0, 1, -2.5f, false, 0);
        assert(c.getWeight() == -2.5f && "negative weight must be preserved exactly");
        assert(!c.isEnabled() && "enabled=false at construction must be preserved");
    }

    // 6: enable()/disable() toggle the flag and nothing else.
    {
        ConnectionGene c(0, 1, 1.0f, false, 0);
        assert(!c.isEnabled() && "must start disabled");
        c.enable();
        assert(c.isEnabled() && "enable() must set the flag");
        c.disable();
        assert(!c.isEnabled() && "disable() must clear the flag");
        assert(c.getSourceId() == 0 && c.getTargetId() == 1 && c.getWeight() == 1.0f &&
               c.getInnovationNumber() == 0 && "enable()/disable() must not affect other fields");
    }

    // 7: negative source/target node IDs are rejected.
    {
        assert(throwsInvalidArgument([]() { ConnectionGene c(-1, 1, 0.0f, true, 0); }) &&
               "a negative source node ID must be rejected");
        assert(throwsInvalidArgument([]() { ConnectionGene c(0, -1, 0.0f, true, 0); }) &&
               "a negative target node ID must be rejected");
    }

    // 8: a negative innovation number is rejected.
    {
        assert(throwsInvalidArgument([]() { ConnectionGene c(0, 1, 0.0f, true, -1); }) &&
               "a negative innovation number must be rejected");
    }

    // 9: a self-connection (source == target) is rejected.
    {
        assert(throwsInvalidArgument([]() { ConnectionGene c(4, 4, 0.0f, true, 0); }) &&
               "a self-connection must be rejected");
    }

    TraceLog(LOG_INFO, "NEAT gene verification: all deterministic checks passed");
}

// Deterministic check of ai::neat::Genome -- a pure genetic container, no
// evaluation/mutation/crossover here.
void verifyGenome()
{
    using ai::neat::ConnectionGene;
    using ai::neat::Genome;
    using ai::neat::NodeGene;
    using ai::neat::NodeType;

    auto throwsInvalidArgument = [](auto&& callable) -> bool
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
    };

    // 1: a default-constructed genome is empty and already valid.
    {
        Genome genome;
        assert(genome.nodes().empty() && "a fresh genome must have no nodes");
        assert(genome.connections().empty() && "a fresh genome must have no connections");
        genome.validate(); // must not throw
    }

    // 2: addNode appends nodes and they show up in nodes().
    {
        Genome genome;
        genome.addNode(NodeGene{0, NodeType::Input});
        genome.addNode(NodeGene{1, NodeType::Bias});
        genome.addNode(NodeGene{2, NodeType::Output});
        assert(genome.nodes().size() == 3 && "addNode must append to nodes()");
        assert(genome.nodes()[0].getId() == 0 && genome.nodes()[1].getId() == 1 &&
               genome.nodes()[2].getId() == 2 && "nodes() must preserve insertion order");
    }

    // 3: adding a node with a duplicate ID is rejected, and does not modify the genome.
    {
        Genome genome;
        genome.addNode(NodeGene{0, NodeType::Input});
        assert(throwsInvalidArgument([&]() { genome.addNode(NodeGene{0, NodeType::Hidden}); }) &&
               "a duplicate node ID must be rejected");
        assert(genome.nodes().size() == 1 && "a rejected addNode must not modify the genome");
    }

    // 4: hasNode/findNode report existence and identity correctly, including for missing IDs.
    {
        Genome genome;
        genome.addNode(NodeGene{7, NodeType::Hidden});
        assert(genome.hasNode(7) && "hasNode must find an existing node ID");
        assert(!genome.hasNode(8) && "hasNode must not find a missing node ID");
        const NodeGene* found = genome.findNode(7);
        assert(found != nullptr && found->getId() == 7 && found->getType() == NodeType::Hidden &&
               "findNode must return the matching node");
        assert(genome.findNode(8) == nullptr && "findNode must return nullptr for a missing node ID");
    }

    // 5: a valid connection between two existing nodes is accepted.
    {
        Genome genome;
        genome.addNode(NodeGene{0, NodeType::Input});
        genome.addNode(NodeGene{1, NodeType::Output});
        genome.addConnection(ConnectionGene{0, 1, 0.5f, true, 0});
        assert(genome.connections().size() == 1 && "addConnection must append to connections()");
        assert(genome.connections()[0].getSourceId() == 0 && genome.connections()[0].getTargetId() == 1 &&
               "the stored connection must match what was added");
    }

    // 6: a connection with a missing source node is rejected.
    {
        Genome genome;
        genome.addNode(NodeGene{1, NodeType::Output});
        assert(throwsInvalidArgument([&]() { genome.addConnection(ConnectionGene{0, 1, 0.1f, true, 0}); }) &&
               "a connection with an unknown source node must be rejected");
        assert(genome.connections().empty() && "a rejected addConnection must not modify the genome");
    }

    // 7: a connection with a missing target node is rejected.
    {
        Genome genome;
        genome.addNode(NodeGene{0, NodeType::Input});
        assert(throwsInvalidArgument([&]() { genome.addConnection(ConnectionGene{0, 1, 0.1f, true, 0}); }) &&
               "a connection with an unknown target node must be rejected");
        assert(genome.connections().empty() && "a rejected addConnection must not modify the genome");
    }

    // 8: a duplicate directed connection is rejected, even with a different
    // weight/innovation number -- only (source, target) identity matters.
    {
        Genome genome;
        genome.addNode(NodeGene{0, NodeType::Input});
        genome.addNode(NodeGene{1, NodeType::Output});
        genome.addConnection(ConnectionGene{0, 1, 0.1f, true, 0});
        assert(throwsInvalidArgument([&]() { genome.addConnection(ConnectionGene{0, 1, 0.9f, false, 99}); }) &&
               "a duplicate (source, target) connection must be rejected regardless of weight/innovation");
        assert(genome.connections().size() == 1 && "a rejected duplicate connection must not modify the genome");
    }

    // 9: hasConnection/findConnection report existence and identity correctly,
    // including for missing (source, target) pairs, and the reverse direction
    // is treated as a distinct, absent connection.
    {
        Genome genome;
        genome.addNode(NodeGene{0, NodeType::Input});
        genome.addNode(NodeGene{1, NodeType::Output});
        genome.addConnection(ConnectionGene{0, 1, 0.42f, true, 5});

        assert(genome.hasConnection(0, 1) && "hasConnection must find an existing (source, target) pair");
        assert(!genome.hasConnection(1, 0) && "hasConnection must not treat the reverse direction as existing");

        const ConnectionGene* found = genome.findConnection(0, 1);
        assert(found != nullptr && found->getWeight() == 0.42f && found->getInnovationNumber() == 5 &&
               "findConnection must return the matching connection");
        assert(genome.findConnection(1, 0) == nullptr &&
               "findConnection must return nullptr for a missing (source, target) pair");
    }

    // 10: validate() succeeds (does not throw) on a genome built entirely
    // through the validated addNode/addConnection API.
    {
        Genome genome;
        genome.addNode(NodeGene{0, NodeType::Input});
        genome.addNode(NodeGene{1, NodeType::Bias});
        genome.addNode(NodeGene{2, NodeType::Output});
        genome.addConnection(ConnectionGene{0, 2, 1.0f, true, 0});
        genome.addConnection(ConnectionGene{1, 2, 1.0f, true, 1});
        genome.validate(); // must not throw
    }

    // 11: validate() rejects duplicate node IDs (built via the raw bulk
    // constructor, since addNode itself would reject this).
    {
        std::vector<NodeGene> nodes = {NodeGene{0, NodeType::Input}, NodeGene{0, NodeType::Hidden}};
        Genome genome(nodes, {});
        assert(throwsInvalidArgument([&]() { genome.validate(); }) &&
               "validate() must reject duplicate node IDs");
    }

    // 12: validate() rejects a genome with a connection referencing a
    // nonexistent node.
    {
        std::vector<NodeGene> nodes = {NodeGene{0, NodeType::Input}};
        std::vector<ConnectionGene> connections = {ConnectionGene{0, 99, 0.1f, true, 0}};
        Genome genome(nodes, connections);
        assert(throwsInvalidArgument([&]() { genome.validate(); }) &&
               "validate() must reject a connection with a nonexistent endpoint");
    }

    // 13: validate() rejects a genome with duplicate directed connections.
    {
        std::vector<NodeGene> nodes = {NodeGene{0, NodeType::Input}, NodeGene{1, NodeType::Output}};
        std::vector<ConnectionGene> connections = {
            ConnectionGene{0, 1, 0.1f, true, 0},
            ConnectionGene{0, 1, 0.2f, true, 1},
        };
        Genome genome(nodes, connections);
        assert(throwsInvalidArgument([&]() { genome.validate(); }) &&
               "validate() must reject duplicate directed connections");
    }

    TraceLog(LOG_INFO, "Genome verification: all deterministic checks passed");
}

namespace innovation_tracker_verify
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

template <typename Callable>
bool throwsOverflowError(Callable&& callable)
{
    try
    {
        callable();
    }
    catch (const std::overflow_error&)
    {
        return true;
    }
    return false;
}

} // namespace innovation_tracker_verify

// Deterministic check of ai::neat::InnovationTracker's historical-marking
// bookkeeping. Its API takes no Genome parameter anywhere, so it can't
// mutate one even in principle -- enforced by the type signatures themselves.
void verifyInnovationTracker()
{
    using namespace innovation_tracker_verify;
    using ai::neat::InnovationNumber;
    using ai::neat::InnovationTracker;
    using ai::neat::NodeId;
    using ai::neat::NodeSplitInnovation;

    // 1: constructor preserves initial counters.
    {
        InnovationTracker tracker(5, 20);
        assert(tracker.getNextAvailableNodeId() == 5 && tracker.getNextAvailableInnovation() == 20 &&
               "constructor must preserve the initial node ID and innovation counters");
    }

    // 2: invalid initial node ID rejected.
    {
        assert(throwsInvalidArgument([]() { InnovationTracker tracker(-1, 0); }) &&
               "a negative firstAvailableNodeId must be rejected");
    }

    // 3: invalid initial innovation rejected.
    {
        assert(throwsInvalidArgument([]() { InnovationTracker tracker(0, -1); }) &&
               "a negative firstAvailableInnovation must be rejected");
    }

    // 4-10: connection innovation behavior.
    {
        InnovationTracker tracker(5, 20);

        // 4: first new connection receives the initial innovation value.
        const InnovationNumber first = tracker.getConnectionInnovation(0, 1);
        assert(first == 20 && "the first new connection must receive the initial innovation value");

        // 5: the same directed connection reuses its innovation.
        const InnovationNumber firstAgain = tracker.getConnectionInnovation(0, 1);
        assert(firstAgain == first && tracker.getNextAvailableInnovation() == 21 &&
               "requesting the same directed connection again must reuse its innovation, not allocate a new one");

        // 6: reversed connection receives a different innovation.
        const InnovationNumber reversed = tracker.getConnectionInnovation(1, 0);
        assert(reversed != first && reversed == 21 && "the reversed connection must be a distinct structural connection");

        // 7: different connections receive unique, monotonically increasing innovations.
        const InnovationNumber c1 = tracker.getConnectionInnovation(2, 3);
        const InnovationNumber c2 = tracker.getConnectionInnovation(3, 4);
        const InnovationNumber c3 = tracker.getConnectionInnovation(4, 5);
        assert(c1 == 22 && c2 == 23 && c3 == 24 &&
               "distinct new connections must receive unique, monotonically increasing innovations");

        // 8: negative source rejected.
        assert(throwsInvalidArgument([&]() { tracker.getConnectionInnovation(-1, 0); }) &&
               "a negative source ID must be rejected");

        // 9: negative target rejected.
        assert(throwsInvalidArgument([&]() { tracker.getConnectionInnovation(0, -1); }) &&
               "a negative target ID must be rejected");

        // 10: self-connection rejected.
        assert(throwsInvalidArgument([&]() { tracker.getConnectionInnovation(2, 2); }) &&
               "a self-connection must be rejected");
    }

    // 11-18: node-split innovation behavior.
    {
        InnovationTracker tracker(100, 200);

        // 11: first node split receives the initial available node ID.
        const NodeSplitInnovation split1 = tracker.getNodeSplitInnovation(50, 1, 2);
        assert(split1.newNodeId == 100 && "the first node split must receive the initial available node ID");

        // 12: split allocates two distinct connection innovations.
        assert(split1.incomingInnovation == 200 && split1.outgoingInnovation == 201 &&
               split1.incomingInnovation != split1.outgoingInnovation &&
               "a split must allocate two distinct connection innovations");
        assert(tracker.getNextAvailableNodeId() == 101 && tracker.getNextAvailableInnovation() == 202 &&
               "a fresh split must advance both counters");

        // 13: repeated identical split reuses the complete record.
        const NodeSplitInnovation split1Again = tracker.getNodeSplitInnovation(50, 1, 2);
        assert(split1Again.newNodeId == split1.newNodeId && split1Again.incomingInnovation == split1.incomingInnovation &&
               split1Again.outgoingInnovation == split1.outgoingInnovation &&
               "requesting the same split again must reuse the complete stored record");

        // 14: repeated split does not advance counters.
        assert(tracker.getNextAvailableNodeId() == 101 && tracker.getNextAvailableInnovation() == 202 &&
               "repeating a split must not allocate anything new");

        // 15 & 16: a different split receives a different node ID and
        // distinct connection innovations.
        const NodeSplitInnovation split2 = tracker.getNodeSplitInnovation(60, 3, 4);
        assert(split2.newNodeId == 101 && split2.newNodeId != split1.newNodeId &&
               "a different split must receive a different node ID");
        assert(split2.incomingInnovation == 202 && split2.outgoingInnovation == 203 &&
               split2.incomingInnovation != split1.incomingInnovation && split2.outgoingInnovation != split1.outgoingInnovation &&
               "a different split must receive distinct connection innovations");

        // 17: inconsistent source/target for an existing split innovation is rejected.
        assert(throwsInvalidArgument([&]() { tracker.getNodeSplitInnovation(50, 1, 3); }) &&
               "requesting an already-recorded split innovation with a different target must be rejected");
        assert(throwsInvalidArgument([&]() { tracker.getNodeSplitInnovation(50, 9, 2); }) &&
               "requesting an already-recorded split innovation with a different source must be rejected");

        // 18: negative split innovation rejected.
        assert(throwsInvalidArgument([&]() { tracker.getNodeSplitInnovation(-1, 1, 2); }) &&
               "a negative splitConnectionInnovation must be rejected");
    }

    // 19 & 20: split-created connection innovations share the global
    // connection history -- direct requests for the same pairs reuse them.
    {
        InnovationTracker tracker(100, 200);
        const NodeSplitInnovation split = tracker.getNodeSplitInnovation(50, 1, 2);

        const InnovationNumber directIncoming = tracker.getConnectionInnovation(1, split.newNodeId);
        assert(directIncoming == split.incomingInnovation && tracker.getNextAvailableInnovation() == 202 &&
               "a split's incoming connection must be reusable via a direct getConnectionInnovation call, "
               "allocating nothing new");

        const InnovationNumber directOutgoing = tracker.getConnectionInnovation(split.newNodeId, 2);
        assert(directOutgoing == split.outgoingInnovation && tracker.getNextAvailableInnovation() == 202 &&
               "a split's outgoing connection must be reusable via a direct getConnectionInnovation call, "
               "allocating nothing new");
    }

    // 21: a direct connection innovation created before a split is reused
    // by a later split that needs the exact same (source, newNode) pair.
    {
        InnovationTracker tracker(10, 100);

        const InnovationNumber preExisting = tracker.getConnectionInnovation(1, 10); // 10 is what the split below will allocate
        assert(preExisting == 100 && tracker.getNextAvailableInnovation() == 101 &&
               "setup: the pre-existing direct connection must be the first allocated innovation");

        const NodeSplitInnovation split = tracker.getNodeSplitInnovation(999, 1, 2);
        assert(split.newNodeId == 10 &&
               "the split's new node ID must be the next available one, matching the pre-existing connection's target");
        assert(split.incomingInnovation == preExisting &&
               "a connection innovation already registered directly must be reused by a later split needing the same pair");
        assert(tracker.getNextAvailableInnovation() == 102 &&
               "only the split's new (outgoing) connection should allocate a fresh innovation -- the incoming one was reused");
    }

    // 22 & 23: no node ID or innovation is ever reused for a different
    // structure, across a batch of distinct splits.
    {
        InnovationTracker tracker(1000, 0);
        std::vector<NodeId> nodeIds;
        std::vector<InnovationNumber> innovations;

        for (int i = 0; i < 5; ++i)
        {
            const NodeSplitInnovation split = tracker.getNodeSplitInnovation(i, i, i + 100);
            nodeIds.push_back(split.newNodeId);
            innovations.push_back(split.incomingInnovation);
            innovations.push_back(split.outgoingInnovation);
        }

        for (std::size_t i = 0; i < nodeIds.size(); ++i)
        {
            for (std::size_t j = i + 1; j < nodeIds.size(); ++j)
            {
                assert(nodeIds[i] != nodeIds[j] && "no node ID may ever be reused for a different split");
            }
        }
        for (std::size_t i = 0; i < innovations.size(); ++i)
        {
            for (std::size_t j = i + 1; j < innovations.size(); ++j)
            {
                assert(innovations[i] != innovations[j] &&
                       "no innovation number may ever be reused for a different directed pair");
            }
        }
    }

    // 24: a fixed request sequence is deterministic across two identically constructed trackers.
    {
        InnovationTracker trackerA(100, 0);
        InnovationTracker trackerB(100, 0);

        auto runSequence = [](InnovationTracker& t)
        {
            std::vector<InnovationNumber> results;
            results.push_back(t.getConnectionInnovation(0, 1));
            results.push_back(t.getConnectionInnovation(1, 2));
            const NodeSplitInnovation split = t.getNodeSplitInnovation(0, 0, 1);
            results.push_back(split.newNodeId);
            results.push_back(split.incomingInnovation);
            results.push_back(split.outgoingInnovation);
            results.push_back(t.getConnectionInnovation(2, 3));
            return results;
        };

        const std::vector<InnovationNumber> resultsA = runSequence(trackerA);
        const std::vector<InnovationNumber> resultsB = runSequence(trackerB);
        assert(resultsA == resultsB &&
               "an identical request sequence on two identically constructed trackers must be fully deterministic");
    }

    // 25: a different (still valid) request order may produce different
    // numbering for the same pair, while each tracker remains internally
    // consistent.
    {
        InnovationTracker trackerD(0, 50);
        InnovationTracker trackerE(0, 50);

        const InnovationNumber d_first = trackerD.getConnectionInnovation(1, 2);
        const InnovationNumber d_second = trackerD.getConnectionInnovation(3, 4);

        const InnovationNumber e_first = trackerE.getConnectionInnovation(3, 4);
        const InnovationNumber e_second = trackerE.getConnectionInnovation(1, 2);

        assert(d_first == 50 && d_second == 51 && "requesting (1,2) before (3,4) must give (1,2) the earlier innovation");
        assert(e_first == 50 && e_second == 51 && "requesting (3,4) before (1,2) must give (3,4) the earlier innovation");
        assert(d_first != e_second &&
               "different valid request orders may assign different innovation numbers to the same pair");

        assert(trackerD.getConnectionInnovation(1, 2) == d_first && trackerE.getConnectionInnovation(1, 2) == e_second &&
               "each tracker must remain internally self-consistent regardless of request order");
    }

    // 26: node-counter overflow is rejected.
    {
        InnovationTracker exhaustedNodes(std::numeric_limits<NodeId>::max(), 0);
        assert(throwsOverflowError([&]() { exhaustedNodes.getNodeSplitInnovation(0, 1, 2); }) &&
               "a node split when the node ID counter is exhausted must throw std::overflow_error");
    }

    // 27: innovation-counter overflow is rejected.
    {
        InnovationTracker exhaustedInnovations(0, std::numeric_limits<InnovationNumber>::max());
        assert(throwsOverflowError([&]() { exhaustedInnovations.getConnectionInnovation(1, 2); }) &&
               "a new connection when the innovation counter is exhausted must throw std::overflow_error");
    }

    // 28 & 29: no Genome parameter exists anywhere in InnovationTracker's API.
    // 30: all previous verification suites still pass (main() calls every verify*() function).

    TraceLog(LOG_INFO, "Innovation tracker verification: all deterministic checks passed");
}
} // namespace verification
