#include "ai/neat/Genome.h"

#include <cstddef>
#include <stdexcept>

namespace ai::neat
{

Genome::Genome(std::vector<NodeGene> nodes, std::vector<ConnectionGene> connections)
    : m_nodes(std::move(nodes))
    , m_connections(std::move(connections))
{
}

void Genome::addNode(const NodeGene& node)
{
    if (hasNode(node.getId()))
    {
        throw std::invalid_argument("Genome: duplicate node ID");
    }
    m_nodes.push_back(node);
}

bool Genome::hasNode(NodeId id) const
{
    return findNode(id) != nullptr;
}

const NodeGene* Genome::findNode(NodeId id) const
{
    for (const NodeGene& node : m_nodes)
    {
        if (node.getId() == id)
        {
            return &node;
        }
    }
    return nullptr;
}

void Genome::addConnection(const ConnectionGene& connection)
{
    if (!hasNode(connection.getSourceId()))
    {
        throw std::invalid_argument("Genome: connection source node does not exist");
    }
    if (!hasNode(connection.getTargetId()))
    {
        throw std::invalid_argument("Genome: connection target node does not exist");
    }
    if (hasConnection(connection.getSourceId(), connection.getTargetId()))
    {
        throw std::invalid_argument("Genome: duplicate connection between the same source and target");
    }
    m_connections.push_back(connection);
}

bool Genome::hasConnection(NodeId sourceId, NodeId targetId) const
{
    return findConnection(sourceId, targetId) != nullptr;
}

const ConnectionGene* Genome::findConnection(NodeId sourceId, NodeId targetId) const
{
    for (const ConnectionGene& connection : m_connections)
    {
        if (connection.getSourceId() == sourceId && connection.getTargetId() == targetId)
        {
            return &connection;
        }
    }
    return nullptr;
}

void Genome::validate() const
{
    for (std::size_t i = 0; i < m_nodes.size(); ++i)
    {
        for (std::size_t j = i + 1; j < m_nodes.size(); ++j)
        {
            if (m_nodes[i].getId() == m_nodes[j].getId())
            {
                throw std::invalid_argument("Genome: duplicate node ID");
            }
        }
    }

    for (std::size_t i = 0; i < m_connections.size(); ++i)
    {
        const ConnectionGene& connection = m_connections[i];

        if (!hasNode(connection.getSourceId()))
        {
            throw std::invalid_argument("Genome: connection source node does not exist");
        }
        if (!hasNode(connection.getTargetId()))
        {
            throw std::invalid_argument("Genome: connection target node does not exist");
        }

        for (std::size_t j = i + 1; j < m_connections.size(); ++j)
        {
            if (connection.getSourceId() == m_connections[j].getSourceId() &&
                connection.getTargetId() == m_connections[j].getTargetId())
            {
                throw std::invalid_argument("Genome: duplicate connection between the same source and target");
            }
        }
    }
}

} // namespace ai::neat
