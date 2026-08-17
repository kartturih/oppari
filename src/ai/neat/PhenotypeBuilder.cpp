#include "ai/neat/PhenotypeBuilder.h"

#include <algorithm>
#include <array>
#include <utility>
#include <vector>

namespace ai::neat
{

namespace
{

ai::NodeType toRuntimeNodeType(NodeType type)
{
    switch (type)
    {
        case NodeType::Input:
            return ai::NodeType::Input;
        case NodeType::Bias:
            return ai::NodeType::Bias;
        case NodeType::Hidden:
            return ai::NodeType::Hidden;
        case NodeType::Output:
            return ai::NodeType::Output;
    }
    return ai::NodeType::Hidden; // unreachable -- NodeType is a closed enum class
}

bool byAscendingId(const NodeGene& lhs, const NodeGene& rhs)
{
    return lhs.getId() < rhs.getId();
}

} // namespace

ai::NeuralNetwork buildPhenotype(const Genome& genome)
{
    genome.validate();

    std::vector<NodeGene> inputs;
    std::vector<NodeGene> bias;
    std::vector<NodeGene> hidden;
    std::vector<NodeGene> outputs;

    for (const NodeGene& node : genome.nodes())
    {
        switch (node.getType())
        {
            case NodeType::Input:
                inputs.push_back(node);
                break;
            case NodeType::Bias:
                bias.push_back(node);
                break;
            case NodeType::Hidden:
                hidden.push_back(node);
                break;
            case NodeType::Output:
                outputs.push_back(node);
                break;
        }
    }

    std::sort(inputs.begin(), inputs.end(), byAscendingId);
    std::sort(bias.begin(), bias.end(), byAscendingId);
    std::sort(hidden.begin(), hidden.end(), byAscendingId);
    std::sort(outputs.begin(), outputs.end(), byAscendingId);

    // Fixed category order, each ascending by node ID -- see the header's
    // ordering contract.
    std::vector<ai::Node> nodes;
    nodes.reserve(genome.nodes().size());
    for (const std::vector<NodeGene>* group : {&inputs, &bias, &hidden, &outputs})
    {
        for (const NodeGene& node : *group)
        {
            nodes.push_back(ai::Node{node.getId(), toRuntimeNodeType(node.getType())});
        }
    }

    // Deterministic order: source ID, then target ID, then innovation number.
    std::vector<ConnectionGene> sortedConnections(genome.connections());
    std::sort(sortedConnections.begin(), sortedConnections.end(),
               [](const ConnectionGene& lhs, const ConnectionGene& rhs)
               {
                   if (lhs.getSourceId() != rhs.getSourceId())
                   {
                       return lhs.getSourceId() < rhs.getSourceId();
                   }
                   if (lhs.getTargetId() != rhs.getTargetId())
                   {
                       return lhs.getTargetId() < rhs.getTargetId();
                   }
                   return lhs.getInnovationNumber() < rhs.getInnovationNumber();
               });

    std::vector<ai::Connection> connections;
    connections.reserve(sortedConnections.size());
    for (const ConnectionGene& connection : sortedConnections)
    {
        connections.push_back(ai::Connection{connection.getSourceId(), connection.getTargetId(),
                                              connection.getWeight(), connection.isEnabled()});
    }

    return ai::NeuralNetwork(std::move(nodes), std::move(connections));
}

} // namespace ai::neat
