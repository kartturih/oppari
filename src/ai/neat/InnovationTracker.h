#pragma once

#include <map>
#include <utility>

#include "ai/neat/ConnectionGene.h"

namespace ai::neat
{

// Historical marking for one add-node ("split") mutation: splits an
// existing connection source -> target into source -> newNodeId -> target.
struct NodeSplitInnovation
{
    NodeId newNodeId;
    InnovationNumber incomingInnovation; // source -> newNodeId
    InnovationNumber outgoingInnovation; // newNodeId -> target
};

// Owns global historical-marking state for one evolutionary run: assigns
// connection innovation numbers and node-split records so the same
// structural mutation always gets the same marking wherever it recurs --
// the standard NEAT mechanism letting crossover/compatibility-distance line
// up genes by ancestry rather than topology.
//
// Deterministic: no RNG or hidden state. Which (sourceId, targetId) pairs
// and splits are requested, and the ORDER they are first requested in,
// determines which innovation numbers/node IDs they receive.
class InnovationTracker
{
public:
    // Throws std::invalid_argument if either argument is negative.
    InnovationTracker(NodeId firstAvailableNodeId, InnovationNumber firstAvailableInnovation);

    // Innovation number for directed connection sourceId -> targetId:
    // assigns a new one on first request, returns the stored one after.
    // The reverse pair is a distinct connection with its own number.
    // Throws std::invalid_argument on invalid/equal IDs, std::overflow_error
    // if the counter is exhausted.
    InnovationNumber getConnectionInnovation(NodeId sourceId, NodeId targetId);

    // Split record for splitConnectionInnovation (source -> newNode ->
    // target). First request allocates a node ID and the two connection
    // innovations (via getConnectionInnovation()); later requests for the
    // same split return the stored record unchanged.
    // Throws std::invalid_argument on invalid IDs or a mismatched re-request
    // for an already-recorded split, std::overflow_error on exhaustion.
    NodeSplitInnovation getNodeSplitInnovation(InnovationNumber splitConnectionInnovation, NodeId sourceId, NodeId targetId);

    // Read-only lookup -- never allocates, unlike getNodeSplitInnovation().
    // Returns nullptr if not recorded yet. Same argument validation as above.
    const NodeSplitInnovation* findNodeSplitInnovation(InnovationNumber splitConnectionInnovation, NodeId sourceId,
                                                         NodeId targetId) const;

    // Next ID/innovation number that would be allocated. Allocation throws
    // std::overflow_error rather than wrapping once the counter's type max
    // is reached.
    NodeId getNextAvailableNodeId() const { return m_nextNodeId; }
    InnovationNumber getNextAvailableInnovation() const { return m_nextInnovation; }

private:
    NodeId allocateNodeId();
    InnovationNumber allocateInnovation();

    // sourceId/targetId kept only for the getNodeSplitInnovation() consistency check.
    struct SplitRecord
    {
        NodeId sourceId;
        NodeId targetId;
        NodeSplitInnovation innovation;
    };

    NodeId m_nextNodeId;
    InnovationNumber m_nextInnovation;

    std::map<std::pair<NodeId, NodeId>, InnovationNumber> m_connectionInnovations;
    std::map<InnovationNumber, SplitRecord> m_splits;
};

} // namespace ai::neat
