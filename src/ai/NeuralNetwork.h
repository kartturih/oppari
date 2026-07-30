#pragma once

#include <array>
#include <utility>
#include <vector>

#include "ai/Observation.h"

namespace ai
{

// Stable identifier for a node. Never assumed to be a contiguous array
// index -- NeuralNetwork resolves IDs to internal storage once, at
// construction time, so a future NEAT Genome is free to assign IDs however
// it likes (e.g. NEAT innovation numbers).
using NodeId = int;

enum class NodeType
{
    Input,
    Bias,
    Hidden,
    Output
};

// One node definition, supplied at construction time.
struct Node
{
    NodeId id = 0;
    NodeType type = NodeType::Hidden;
};

// One directed, weighted connection definition, supplied at construction
// time. `enabled` exists only so a future NEAT genome can carry disabled
// genes through unchanged; a disabled connection is still validated (legal
// endpoints, no duplicates) but contributes nothing to evaluation.
struct Connection
{
    NodeId sourceId = 0;
    NodeId targetId = 0;
    float weight = 0.0f;
    bool enabled = true;
};

// Executable feed-forward phenotype: runtime nodes and weighted connections,
// evaluated in a topological order computed once at construction time. This
// class is runtime-only -- it has no notion of mutation, crossover,
// innovation numbers, fitness, species, or evolution. It is deliberately
// shaped so a future NEAT Genome can build one via the Node/Connection
// constructor below without any change to the evaluation logic here.
//
// Input/Output identification: the order Input nodes appear in the `nodes`
// argument determines which Observation value (0..kInputCount-1) each one
// receives; the order Output nodes appear determines which is steering
// (first) and which is throttle (second). This ordering is independent of
// node ID value, and evaluate() never relies on ID ordering either -- only
// on the topological order derived from actual connections.
class NeuralNetwork
{
public:
    static constexpr int kInputCount = kObservationSize; // one input node per Observation value
    static constexpr int kOutputCount = 2;                // index 0 = steering, index 1 = throttle

    // Validates and builds the runtime graph from explicit definitions.
    // Throws std::invalid_argument if the definition is structurally
    // invalid: wrong Input/Bias/Output counts, duplicate node IDs, a
    // connection referencing an unknown node ID, a connection targeting an
    // Input or Bias node, a duplicate (source, target) connection pair, or
    // a cyclic graph.
    NeuralNetwork(std::vector<Node> nodes, std::vector<Connection> connections);

    // Seeds the kInputCount input nodes from `observation` and the bias node
    // with 1.0, evaluates every Hidden/Output node in topological order
    // (tanh of the sum of its incoming weighted values), and returns
    // {steering, throttle}. No allocation occurs during evaluation.
    std::array<float, kOutputCount> evaluate(const Observation& observation) const;

private:
    std::vector<NodeType> m_nodeTypes;                          // internal index -> node type
    std::vector<std::vector<std::pair<int, float>>> m_incoming; // internal index -> (source internal index, weight)
    std::vector<int> m_evalOrder;                               // Hidden/Output internal indices, topologically sorted
    std::array<int, kInputCount> m_inputIndices{};              // Observation slot -> internal index
    std::array<int, kOutputCount> m_outputIndices{};            // output slot -> internal index
    int m_biasIndex = -1;

    // Scratch per-node activation buffer, sized once at construction and
    // overwritten (never reallocated) on every evaluate() call.
    mutable std::vector<float> m_values;
};

} // namespace ai
