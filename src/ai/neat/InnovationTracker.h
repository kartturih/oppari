#pragma once

#include <map>
#include <utility>

#include "ai/neat/ConnectionGene.h"

namespace ai::neat
{

// One historical-marking record for a single add-node ("split") mutation:
// splitting an existing connection source -> target into
// source -> newNodeId and newNodeId -> target.
struct NodeSplitInnovation
{
    NodeId newNodeId;
    InnovationNumber incomingInnovation; // source -> newNodeId
    InnovationNumber outgoingInnovation; // newNodeId -> target
};

// Owns the global historical-marking state for one evolutionary run: it
// assigns connection innovation numbers and node-split records so that the
// same structural mutation, wherever/whenever it recurs, is always given
// the same historical marking -- the standard NEAT mechanism that lets
// later stages (crossover, compatibility distance) line up genes that
// share ancestry without comparing topology directly.
//
// InnovationTracker is purely bookkeeping: no method anywhere in its API
// takes a Genome, so it cannot mutate one even in principle. It also knows
// nothing about fitness, species, population, Car, NeuralNetwork, or
// AIController. This stage (9B) implements only the tracker itself -- no
// structural mutation exists yet to call it, and GenomeMutator (Stage 9A)
// is not modified.
//
// Determinism: no randomness is used anywhere -- no std::random_device,
// rand(), time-based seed, or hidden static/global state. For a fixed pair
// of constructor values, an identical sequence of requests always produces
// identical results. The *set* of (sourceId, targetId) pairs and splits
// requested, together with the *order* they are first requested in,
// determines which of them receive which innovation numbers/node IDs -- so
// two different valid request orders may number the same eventual set of
// structures differently, while each individually remains fully internally
// consistent (every repeat of the same request still returns exactly what
// that request returned before).
class InnovationTracker
{
public:
    // Throws std::invalid_argument if firstAvailableNodeId < 0 or
    // firstAvailableInnovation < 0. These seed the counters so a caller can
    // reserve IDs/innovations already used by an initial genome (its
    // Input/Bias/Output nodes and their connections) before any new
    // structural mutation is tracked -- this constructor does not scan a
    // Genome to work that out itself.
    InnovationTracker(NodeId firstAvailableNodeId, InnovationNumber firstAvailableInnovation);

    // Returns the innovation number for the directed connection
    // sourceId -> targetId: assigns a new one (the next available) the
    // first time this exact directed pair is requested, and returns the
    // same stored number on every later request for it. The reverse pair
    // (targetId -> sourceId) is a structurally distinct connection with its
    // own, independent innovation number. Weight and enabled state are
    // never considered -- this is purely a
    // (sourceId, targetId) -> innovation mapping.
    //
    // Throws std::invalid_argument if sourceId or targetId is negative, or
    // sourceId == targetId. Throws std::overflow_error if the innovation
    // counter is exhausted (see the "Overflow" note on
    // getNextAvailableInnovation()).
    InnovationNumber getConnectionInnovation(NodeId sourceId, NodeId targetId);

    // Returns the split record for splitting the connection identified by
    // splitConnectionInnovation (understood to run sourceId -> targetId)
    // into sourceId -> newNode -> targetId. The identity of a split is
    // splitConnectionInnovation alone -- sourceId/targetId are only used to
    // detect an inconsistent re-request (see below), not as part of the
    // lookup key, since by definition an original connection's innovation
    // number already uniquely identifies its (source, target) pair.
    //
    // The *first* time a given splitConnectionInnovation is requested: one
    // new node ID is allocated, then the incoming/outgoing innovations are
    // obtained via getConnectionInnovation(sourceId, newNodeId) and
    // getConnectionInnovation(newNodeId, targetId) -- the exact same
    // connection-innovation history getConnectionInnovation() itself uses.
    // This means a split's two new connections can equally be looked up
    // directly afterward with getConnectionInnovation(), and if either of
    // those two exact directed pairs was already registered *before* this
    // split (e.g. because the same newNodeId was reused by coincidence in a
    // handcrafted scenario, or requested directly in advance), the split
    // reuses that existing innovation rather than allocating a new one.
    //
    // Every later request for the same splitConnectionInnovation returns
    // the stored record unchanged, allocating nothing new.
    //
    // Throws std::invalid_argument if splitConnectionInnovation, sourceId
    // or targetId is negative, if sourceId == targetId, or if
    // splitConnectionInnovation has already been recorded with a different
    // sourceId/targetId. Throws std::overflow_error if allocating the new
    // node ID or either connection innovation would exhaust its counter --
    // in that case no split record is stored (though a node ID allocated
    // just before an innovation-counter overflow is not recovered; it is
    // simply never assigned to anything, which is harmless).
    NodeSplitInnovation getNodeSplitInnovation(InnovationNumber splitConnectionInnovation, NodeId sourceId, NodeId targetId);

    // The node ID / innovation number that the next new allocation would
    // receive, given everything requested so far. Overflow: allocation
    // throws std::overflow_error, instead of wrapping, exactly when the
    // relevant counter has reached the maximum value representable by its
    // type -- that value is never itself handed out as a new ID once
    // reached, since there would be no valid "next" value left to advance
    // to afterward.
    NodeId getNextAvailableNodeId() const { return m_nextNodeId; }
    InnovationNumber getNextAvailableInnovation() const { return m_nextInnovation; }

private:
    NodeId allocateNodeId();
    InnovationNumber allocateInnovation();

    // Adds sourceId/targetId to NodeSplitInnovation purely for the
    // consistency check described on getNodeSplitInnovation() above; they
    // are not part of the public record.
    struct SplitRecord
    {
        NodeId sourceId;
        NodeId targetId;
        NodeSplitInnovation innovation;
    };

    NodeId m_nextNodeId;
    InnovationNumber m_nextInnovation;

    // std::map keys on std::pair's lexicographic operator< -- deterministic
    // and requires no custom hash function. Lookup/iteration order has no
    // bearing on which value gets assigned to which key: only the order
    // requests are first made in does.
    std::map<std::pair<NodeId, NodeId>, InnovationNumber> m_connectionInnovations;
    std::map<InnovationNumber, SplitRecord> m_splits;
};

} // namespace ai::neat
