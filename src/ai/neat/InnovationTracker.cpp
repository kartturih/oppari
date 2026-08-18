#include "ai/neat/InnovationTracker.h"

#include <limits>
#include <stdexcept>

namespace ai::neat
{

InnovationTracker::InnovationTracker(NodeId firstAvailableNodeId, InnovationNumber firstAvailableInnovation)
    : m_nextNodeId(firstAvailableNodeId)
    , m_nextInnovation(firstAvailableInnovation)
{
    if (firstAvailableNodeId < 0)
    {
        throw std::invalid_argument("InnovationTracker: firstAvailableNodeId must be non-negative");
    }
    if (firstAvailableInnovation < 0)
    {
        throw std::invalid_argument("InnovationTracker: firstAvailableInnovation must be non-negative");
    }
}

NodeId InnovationTracker::allocateNodeId()
{
    if (m_nextNodeId == std::numeric_limits<NodeId>::max())
    {
        throw std::overflow_error("InnovationTracker: node ID space exhausted");
    }
    const NodeId id = m_nextNodeId;
    ++m_nextNodeId;
    return id;
}

InnovationNumber InnovationTracker::allocateInnovation()
{
    if (m_nextInnovation == std::numeric_limits<InnovationNumber>::max())
    {
        throw std::overflow_error("InnovationTracker: innovation number space exhausted");
    }
    const InnovationNumber innovation = m_nextInnovation;
    ++m_nextInnovation;
    return innovation;
}

InnovationNumber InnovationTracker::getConnectionInnovation(NodeId sourceId, NodeId targetId)
{
    if (sourceId < 0 || targetId < 0)
    {
        throw std::invalid_argument("InnovationTracker: node IDs must be non-negative");
    }
    if (sourceId == targetId)
    {
        throw std::invalid_argument("InnovationTracker: source and target must differ");
    }

    const std::pair<NodeId, NodeId> key(sourceId, targetId);
    const auto existing = m_connectionInnovations.find(key);
    if (existing != m_connectionInnovations.end())
    {
        return existing->second;
    }

    const InnovationNumber innovation = allocateInnovation();
    m_connectionInnovations.emplace(key, innovation);
    return innovation;
}

NodeSplitInnovation InnovationTracker::getNodeSplitInnovation(InnovationNumber splitConnectionInnovation, NodeId sourceId,
                                                                NodeId targetId)
{
    if (splitConnectionInnovation < 0)
    {
        throw std::invalid_argument("InnovationTracker: splitConnectionInnovation must be non-negative");
    }
    if (sourceId < 0 || targetId < 0)
    {
        throw std::invalid_argument("InnovationTracker: node IDs must be non-negative");
    }
    if (sourceId == targetId)
    {
        throw std::invalid_argument("InnovationTracker: source and target must differ");
    }

    const auto existing = m_splits.find(splitConnectionInnovation);
    if (existing != m_splits.end())
    {
        if (existing->second.sourceId != sourceId || existing->second.targetId != targetId)
        {
            throw std::invalid_argument(
                "InnovationTracker: splitConnectionInnovation already recorded with a different source/target");
        }
        return existing->second.innovation;
    }

    const NodeId newNodeId = allocateNodeId();
    const InnovationNumber incoming = getConnectionInnovation(sourceId, newNodeId);
    const InnovationNumber outgoing = getConnectionInnovation(newNodeId, targetId);

    const NodeSplitInnovation innovation{newNodeId, incoming, outgoing};
    m_splits.emplace(splitConnectionInnovation, SplitRecord{sourceId, targetId, innovation});
    return innovation;
}

const NodeSplitInnovation* InnovationTracker::findNodeSplitInnovation(InnovationNumber splitConnectionInnovation,
                                                                        NodeId sourceId, NodeId targetId) const
{
    if (splitConnectionInnovation < 0)
    {
        throw std::invalid_argument("InnovationTracker: splitConnectionInnovation must be non-negative");
    }
    if (sourceId < 0 || targetId < 0)
    {
        throw std::invalid_argument("InnovationTracker: node IDs must be non-negative");
    }
    if (sourceId == targetId)
    {
        throw std::invalid_argument("InnovationTracker: source and target must differ");
    }

    const auto existing = m_splits.find(splitConnectionInnovation);
    if (existing == m_splits.end())
    {
        return nullptr;
    }
    if (existing->second.sourceId != sourceId || existing->second.targetId != targetId)
    {
        throw std::invalid_argument(
            "InnovationTracker: splitConnectionInnovation already recorded with a different source/target");
    }
    return &existing->second.innovation;
}

} // namespace ai::neat
