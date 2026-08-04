#pragma once

#include <vector>

#include "ai/neat/ConnectionGene.h"
#include "ai/neat/NodeGene.h"

namespace ai::neat
{

// Pure genetic container for one NEAT individual: an unordered collection of
// node genes and connection genes. Genome does not execute, mutate, cross
// over, or evaluate anything -- it only owns genes and enforces the
// structural invariants every future NEAT operation depends on (unique node
// IDs, connections only between existing nodes, no duplicate directed
// connections between the same source and target).
class Genome
{
public:
    // Starts as an empty genome (no nodes, no connections).
    Genome() = default;

    // Takes already-built gene collections as-is, without validating them.
    // Intended for a genome assembled in bulk by future code (e.g. a
    // crossover operation) where re-validating gene-by-gene via
    // addNode/addConnection would be redundant -- call validate()
    // explicitly once construction is complete.
    Genome(std::vector<NodeGene> nodes, std::vector<ConnectionGene> connections);

    // Throws std::invalid_argument if a node with the same ID already exists.
    void addNode(const NodeGene& node);

    bool hasNode(NodeId id) const;

    // Returns nullptr if no node with that ID exists.
    const NodeGene* findNode(NodeId id) const;

    // Throws std::invalid_argument if the source or target node does not
    // exist, or if a connection with the same (source, target) pair already
    // exists. Innovation number is not considered when checking for
    // duplicates.
    void addConnection(const ConnectionGene& connection);

    bool hasConnection(NodeId sourceId, NodeId targetId) const;

    // Returns nullptr if no connection with that (source, target) pair exists.
    const ConnectionGene* findConnection(NodeId sourceId, NodeId targetId) const;

    const std::vector<NodeGene>& nodes() const { return m_nodes; }
    const std::vector<ConnectionGene>& connections() const { return m_connections; }

    // Checks every structural invariant addNode/addConnection enforce
    // incrementally: unique node IDs, connection endpoints that exist, and
    // no duplicate directed connections. Throws std::invalid_argument on the
    // first violation found; does nothing otherwise.
    void validate() const;

private:
    std::vector<NodeGene> m_nodes;
    std::vector<ConnectionGene> m_connections;
};

} // namespace ai::neat
