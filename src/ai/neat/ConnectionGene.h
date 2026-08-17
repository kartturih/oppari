#pragma once

#include "ai/neat/NodeGene.h"

namespace ai::neat
{

// Stable identifier assigned when a connection gene is first created by
// mutation; shared by every genome that inherits that structural mutation.
using InnovationNumber = int;

// Pure genetic description of one connection between two node genes, with
// no evaluation logic. Only weight/enabled may change after construction --
// source, target, and innovation number are fixed identity.
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
    void setWeight(float weight) { m_weight = weight; }

private:
    NodeId m_sourceId;
    NodeId m_targetId;
    float m_weight;
    bool m_enabled;
    InnovationNumber m_innovationNumber;
};

} // namespace ai::neat
