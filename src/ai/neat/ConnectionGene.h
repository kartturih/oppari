#pragma once

#include "ai/neat/NodeGene.h"

namespace ai::neat
{

// Stable identifier assigned when a connection gene is first created by
// mutation, shared by every genome that inherits the same structural
// mutation. This header only validates that it is non-negative -- assigning
// innovation numbers is the job of whatever future code creates genes.
using InnovationNumber = int;

// Pure genetic description of one connection between two node genes. Carries
// no evaluation logic: it does not know how to be summed, activated, or
// executed. A future NEAT Genome/NeuralNetwork builder is responsible for
// turning a collection of these into a runtime graph.
//
// `enabled` and `weight` are the only fields this class allows to change
// after construction (via enable()/disable() and setWeight()). Source,
// target and innovation number are fixed for the gene's lifetime -- they
// are its identity. setWeight() performs no randomness and no validation of
// its own (e.g. it does not reject non-finite values); it exists purely as
// the mechanical write used by GenomeMutator, which is responsible for
// deciding what value to write.
class ConnectionGene
{
public:
    // Throws std::invalid_argument if sourceId or targetId is negative, if
    // sourceId == targetId, or if innovationNumber is negative.
    ConnectionGene(NodeId sourceId, NodeId targetId, float weight, bool enabled, InnovationNumber innovationNumber);

    NodeId getSourceId() const { return m_sourceId; }
    NodeId getTargetId() const { return m_targetId; }
    float getWeight() const { return m_weight; }
    bool isEnabled() const { return m_enabled; }
    InnovationNumber getInnovationNumber() const { return m_innovationNumber; }

    void enable() { m_enabled = true; }
    void disable() { m_enabled = false; }

    // Overwrites the weight in place. Does not touch source/target ID,
    // innovation number, or the enabled flag.
    void setWeight(float weight) { m_weight = weight; }

private:
    NodeId m_sourceId;
    NodeId m_targetId;
    float m_weight;
    bool m_enabled;
    InnovationNumber m_innovationNumber;
};

} // namespace ai::neat
