#include "ai/neat/ConnectionGene.h"

#include <stdexcept>

namespace ai::neat
{

ConnectionGene::ConnectionGene(NodeId sourceId, NodeId targetId, float weight, bool enabled,
                                InnovationNumber innovationNumber)
    : m_sourceId(sourceId)
    , m_targetId(targetId)
    , m_weight(weight)
    , m_enabled(enabled)
    , m_innovationNumber(innovationNumber)
{
    if (m_sourceId < 0 || m_targetId < 0)
    {
        throw std::invalid_argument("ConnectionGene: node IDs must be non-negative");
    }
    if (m_sourceId == m_targetId)
    {
        throw std::invalid_argument("ConnectionGene: source and target must differ");
    }
    if (m_innovationNumber < 0)
    {
        throw std::invalid_argument("ConnectionGene: innovation number must be non-negative");
    }
}

} // namespace ai::neat
