#pragma once

namespace ai::neat
{

// Stable identifier for a node gene; always non-negative.
using NodeId = int;

enum class NodeType
{
    Input,
    Bias,
    Hidden,
    Output
};

// Pure genetic description of a node: identity and role only, no runtime state.
class NodeGene
{
public:
    // Throws std::invalid_argument if id is negative.
    NodeGene(NodeId id, NodeType type);

    NodeId getId() const { return m_id; }
    NodeType getType() const { return m_type; }

    friend bool operator==(const NodeGene& lhs, const NodeGene& rhs)
    {
        return lhs.m_id == rhs.m_id && lhs.m_type == rhs.m_type;
    }
    friend bool operator!=(const NodeGene& lhs, const NodeGene& rhs) { return !(lhs == rhs); }

private:
    NodeId m_id;
    NodeType m_type;
};

} // namespace ai::neat
