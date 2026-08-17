#pragma once

#include <vector>

#include "ai/neat/ConnectionGene.h"
#include "ai/neat/NodeGene.h"

namespace ai::neat
{

// Pure genetic container for one NEAT individual: node genes + connection
// genes. Enforces structural invariants (unique node IDs, connections only
// between existing nodes, no duplicate directed connections); performs no
// execution, mutation, or crossover itself.
class Genome
{
public:
    Genome() = default;

    // Takes already-built gene collections as-is, without validating them --
    // call validate() explicitly afterward if needed.
    Genome(std::vector<NodeGene> nodes, std::vector<ConnectionGene> connections);

    // Throws std::invalid_argument if a node with the same ID already exists.
    void addNode(const NodeGene& node);

    bool hasNode(NodeId id) const;
    const NodeGene* findNode(NodeId id) const; // nullptr if not found

    // Throws std::invalid_argument if either endpoint doesn't exist, or a
    // connection between the same (source, target) already exists.
    void addConnection(const ConnectionGene& connection);

    bool hasConnection(NodeId sourceId, NodeId targetId) const;
    const ConnectionGene* findConnection(NodeId sourceId, NodeId targetId) const; // nullptr if not found

    const std::vector<NodeGene>& nodes() const { return m_nodes; }
    const std::vector<ConnectionGene>& connections() const { return m_connections; }

    // Mutable access for GenomeMutator only: change genes via
    // ConnectionGene's own setWeight()/enable()/disable(), never resize,
    // reorder, or replace elements of the vector.
    std::vector<ConnectionGene>& mutableConnections() { return m_connections; }

    // Re-checks every invariant addNode/addConnection enforce incrementally.
    // Throws std::invalid_argument on the first violation found.
    void validate() const;

private:
    std::vector<NodeGene> m_nodes;
    std::vector<ConnectionGene> m_connections;
};

} // namespace ai::neat
