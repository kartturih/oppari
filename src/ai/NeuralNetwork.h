#pragma once

#include <array>
#include <utility>
#include <vector>

#include "ai/Observation.h"

namespace ai
{

// Stable node identifier; not assumed to be a contiguous array index.
using NodeId = int;

enum class NodeType
{
    Input,
    Bias,
    Hidden,
    Output
};

struct Node
{
    NodeId id = 0;
    NodeType type = NodeType::Hidden;
};

// enabled lets a NEAT genome carry disabled genes through unchanged; a
// disabled connection is still validated but contributes nothing.
struct Connection
{
    NodeId sourceId = 0;
    NodeId targetId = 0;
    float weight = 0.0f;
    bool enabled = true;
};

// Executable feed-forward phenotype: nodes + weighted connections,
// evaluated in a topological order computed once at construction. Runtime
// only -- no mutation/crossover/fitness/species logic.
//
// Input node order in the constructor's `nodes` determines which
// Observation slot each receives; Output node order determines which is
// steering (first) vs. throttle (second). evaluate() never relies on node
// ID ordering, only on actual connection topology.
class NeuralNetwork
{
public:
    static constexpr int kInputCount = kObservationSize;
    static constexpr int kOutputCount = 2; // 0 = steering, 1 = throttle

    // Throws std::invalid_argument if structurally invalid: wrong
    // Input/Bias/Output counts, duplicate node IDs, an unknown connection
    // endpoint, a connection targeting Input/Bias, a duplicate (source,
    // target) pair, or a cycle.
    NeuralNetwork(std::vector<Node> nodes, std::vector<Connection> connections);

    // Seeds inputs from observation and bias with 1.0, evaluates
    // Hidden/Output nodes in topological order (tanh of weighted sum),
    // returns {steering, throttle}. No allocation.
    std::array<float, kOutputCount> evaluate(const Observation& observation) const;

private:
    std::vector<NodeType> m_nodeTypes;                          // internal index -> node type
    std::vector<std::vector<std::pair<int, float>>> m_incoming; // internal index -> (source index, weight)
    std::vector<int> m_evalOrder;                               // Hidden/Output indices, topologically sorted
    std::array<int, kInputCount> m_inputIndices{};              // Observation slot -> internal index
    std::array<int, kOutputCount> m_outputIndices{};            // output slot -> internal index
    int m_biasIndex = -1;

    mutable std::vector<float> m_values; // scratch activation buffer, reused per evaluate()
};

} // namespace ai
