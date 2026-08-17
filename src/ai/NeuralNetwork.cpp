#include "ai/NeuralNetwork.h"

#include <cmath>
#include <set>
#include <stdexcept>
#include <unordered_map>

namespace ai
{

NeuralNetwork::NeuralNetwork(std::vector<Node> nodes, std::vector<Connection> connections)
{
    const int nodeCount = static_cast<int>(nodes.size());

    std::unordered_map<NodeId, int> idToIndex;
    idToIndex.reserve(nodes.size());
    for (int i = 0; i < nodeCount; ++i)
    {
        if (!idToIndex.emplace(nodes[i].id, i).second)
        {
            throw std::invalid_argument("NeuralNetwork: duplicate node ID");
        }
    }

    std::vector<int> inputIndices;
    std::vector<int> outputIndices;
    int biasCount = 0;

    for (int i = 0; i < nodeCount; ++i)
    {
        switch (nodes[i].type)
        {
            case NodeType::Input:
                inputIndices.push_back(i);
                break;
            case NodeType::Bias:
                m_biasIndex = i;
                ++biasCount;
                break;
            case NodeType::Output:
                outputIndices.push_back(i);
                break;
            case NodeType::Hidden:
                break;
        }
    }

    if (static_cast<int>(inputIndices.size()) != kInputCount)
    {
        throw std::invalid_argument("NeuralNetwork: must have exactly kInputCount Input nodes");
    }
    if (biasCount != 1)
    {
        throw std::invalid_argument("NeuralNetwork: must have exactly one Bias node");
    }
    if (static_cast<int>(outputIndices.size()) != kOutputCount)
    {
        throw std::invalid_argument("NeuralNetwork: must have exactly kOutputCount Output nodes");
    }

    for (int i = 0; i < kInputCount; ++i)
    {
        m_inputIndices[i] = inputIndices[i];
    }
    for (int i = 0; i < kOutputCount; ++i)
    {
        m_outputIndices[i] = outputIndices[i];
    }

    std::vector<std::vector<int>> adjacency(nodeCount);
    std::vector<int> indegree(nodeCount, 0);
    std::vector<std::vector<std::pair<int, float>>> incoming(nodeCount);
    std::set<std::pair<NodeId, NodeId>> seenConnections;

    for (const Connection& c : connections)
    {
        const auto srcIt = idToIndex.find(c.sourceId);
        const auto dstIt = idToIndex.find(c.targetId);
        if (srcIt == idToIndex.end() || dstIt == idToIndex.end())
        {
            throw std::invalid_argument("NeuralNetwork: connection references an unknown node ID");
        }

        const int srcIndex = srcIt->second;
        const int dstIndex = dstIt->second;

        if (nodes[dstIndex].type == NodeType::Input || nodes[dstIndex].type == NodeType::Bias)
        {
            throw std::invalid_argument("NeuralNetwork: connection cannot target an Input or Bias node");
        }

        if (!seenConnections.insert({c.sourceId, c.targetId}).second)
        {
            throw std::invalid_argument("NeuralNetwork: duplicate connection between the same source and target");
        }

        if (!c.enabled)
        {
            continue;
        }

        incoming[dstIndex].emplace_back(srcIndex, c.weight);
        adjacency[srcIndex].push_back(dstIndex);
        ++indegree[dstIndex];
    }

    // Kahn's algorithm over enabled edges only.
    std::vector<int> ready;
    ready.reserve(nodeCount);
    for (int i = 0; i < nodeCount; ++i)
    {
        if (indegree[i] == 0)
        {
            ready.push_back(i);
        }
    }

    std::vector<int> order;
    order.reserve(nodeCount);
    for (std::size_t head = 0; head < ready.size(); ++head)
    {
        const int n = ready[head];
        order.push_back(n);
        for (int next : adjacency[n])
        {
            if (--indegree[next] == 0)
            {
                ready.push_back(next);
            }
        }
    }

    if (static_cast<int>(order.size()) != nodeCount)
    {
        throw std::invalid_argument("NeuralNetwork: connections must form an acyclic graph");
    }

    m_nodeTypes.resize(nodeCount);
    for (int i = 0; i < nodeCount; ++i)
    {
        m_nodeTypes[i] = nodes[i].type;
    }

    // Input/Bias are seeded directly in evaluate(), not computed.
    m_evalOrder.reserve(nodeCount);
    for (int idx : order)
    {
        if (m_nodeTypes[idx] == NodeType::Hidden || m_nodeTypes[idx] == NodeType::Output)
        {
            m_evalOrder.push_back(idx);
        }
    }

    m_incoming = std::move(incoming);
    m_values.assign(nodeCount, 0.0f);
}

std::array<float, NeuralNetwork::kOutputCount> NeuralNetwork::evaluate(const Observation& observation) const
{
    for (int i = 0; i < kInputCount; ++i)
    {
        m_values[m_inputIndices[i]] = observation.values[i];
    }
    m_values[m_biasIndex] = 1.0f;

    for (int idx : m_evalOrder)
    {
        float sum = 0.0f;
        for (const auto& sourceAndWeight : m_incoming[idx])
        {
            sum += m_values[sourceAndWeight.first] * sourceAndWeight.second;
        }
        m_values[idx] = std::tanh(sum);
    }

    return {m_values[m_outputIndices[0]], m_values[m_outputIndices[1]]};
}

} // namespace ai
