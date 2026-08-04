#include "ai/neat/NodeGene.h"

#include <stdexcept>

namespace ai::neat
{

NodeGene::NodeGene(NodeId id, NodeType type)
    : m_id(id)
    , m_type(type)
{
    if (m_id < 0)
    {
        throw std::invalid_argument("NodeGene: id must be non-negative");
    }
}

} // namespace ai::neat
