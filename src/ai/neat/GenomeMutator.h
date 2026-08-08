#pragma once

#include <cstdint>
#include <random>

#include "ai/neat/Genome.h"
#include "ai/neat/InnovationTracker.h"
#include "ai/neat/MutationConfig.h"

namespace ai::neat
{

// Deterministically mutates a Genome. Owns its own std::mt19937 RNG, seeded
// exactly once at construction -- never std::random_device, never a
// global/static generator, never rand(), never time-based. Every mutation
// method continues drawing from that same generator, so the same seed
// driven through the same sequence of calls on the same starting genome(s)
// (and, for structural mutation, the same starting InnovationTracker state)
// always reproduces the same results; a different seed may produce
// different results.
//
// mutateWeights() (Stage 9A) only ever changes ConnectionGene weights, via
// ConnectionGene::setWeight(). It never adds or removes nodes or
// connections, and never changes node IDs/types, connection source/target
// IDs, innovation numbers, or enabled state -- both enabled and disabled
// connections are equally eligible for weight mutation, since a disabled
// gene may later be re-enabled by future structural mutation and should not
// be left stale.
//
// mutateAddConnection() (Stage 9C) adds at most one new, enabled
// ConnectionGene per call -- see its own doc comment below for the full
// algorithm. It never touches an existing node or connection gene, and
// never adds or removes a node.
//
// mutateAddNode() (Stage 9D) splits at most one existing enabled
// ConnectionGene per call into two new connections through one new Hidden
// node -- see its own doc comment below for the full algorithm. It disables
// the split connection but never removes it, and never touches any other
// existing node or connection gene.
//
// This stage (9D) implements add-node structural mutation only. No
// crossover, species, or population logic exists here or anywhere else in
// the codebase yet.
class GenomeMutator
{
public:
    explicit GenomeMutator(std::uint32_t seed);

    // Throws std::invalid_argument if config's weight-mutation fields are
    // invalid (probabilities outside [0,1], a negative perturbStrength, an
    // inverted replacement range, or any non-finite field). Otherwise, for
    // each connection gene in genome, in storage order:
    //   - with probability config.weightMutationProbability, the
    //     connection is selected;
    //   - if selected, with probability config.weightPerturbProbability its
    //     weight is perturbed (oldWeight + a value drawn uniformly from
    //     [-perturbStrength, +perturbStrength], not clamped), otherwise it
    //     is replaced outright with a value drawn uniformly from
    //     [replacementWeightMin, replacementWeightMax].
    // A genome with no connections is accepted and left unchanged.
    void mutateWeights(Genome& genome, const MutationConfig& config);

    // Attempts to add exactly one new directed connection to genome.
    //
    // Throws std::invalid_argument if config's add-connection fields are
    // invalid (addConnectionProbability outside [0,1] or non-finite,
    // non-finite new-connection weight bounds, newConnectionWeightMin >
    // newConnectionWeightMax, or addConnectionMaxAttempts <= 0). This check
    // runs first, before any RNG draw or access to genome/innovationTracker
    // -- even if the mutation ends up not being selected.
    //
    // Otherwise: draws once against config.addConnectionProbability; if not
    // selected, returns false, leaving genome and innovationTracker
    // completely unchanged (no further draws, no allocation).
    //
    // If selected, searches for one valid (source, target) pair. A source
    // may be Input, Bias, or Hidden (never Output); a target may be Hidden
    // or Output (never Input or Bias). A pair is valid when: source !=
    // target; no connection (enabled *or* disabled) already exists for
    // that exact directed pair; and adding it would not create a cycle
    // among the genome's currently *enabled* connections (equivalently:
    // target must not already have an enabled path to source -- disabled
    // connections never participate in this check, and node IDs/insertion
    // order are never used as a substitute for it). Up to
    // config.addConnectionMaxAttempts random (source, target) pairs, drawn
    // from the genome's actual valid-typed nodes, are tried first; if none
    // validate, a deterministic exhaustive scan follows as a fallback --
    // source nodes ascending by ID, then target nodes ascending by ID --
    // so a valid connection is never missed just because random attempts
    // kept landing on invalid pairs. This search only reads genome;
    // nothing is added until a candidate is found. Returns false, leaving
    // genome and innovationTracker unchanged, if no valid pair exists
    // (including an empty genome, one with no valid source or target
    // nodes, or one whose valid feed-forward connections are all already
    // present or would all create a cycle).
    //
    // Once a candidate is found: its innovation number is requested from
    // innovationTracker.getConnectionInnovation() -- the only point at
    // which the tracker's state changes -- an initial weight is drawn
    // uniformly from [newConnectionWeightMin, newConnectionWeightMax], and
    // the new, enabled ConnectionGene is added to genome via
    // Genome::addConnection(). Returns true. No existing node or
    // connection gene is touched, and no node is ever added.
    bool mutateAddConnection(Genome& genome, InnovationTracker& innovationTracker, const MutationConfig& config);

    // Attempts to split exactly one existing enabled connection of genome
    // into two, through one new Hidden node.
    //
    // Throws std::invalid_argument if config.addNodeProbability is
    // non-finite or outside [0,1]. This check runs first, before any RNG
    // draw or access to genome/innovationTracker -- even if the mutation
    // ends up not being selected.
    //
    // Otherwise: draws once against config.addNodeProbability; if not
    // selected, returns false, leaving genome and innovationTracker
    // completely unchanged (no further draws, no allocation).
    //
    // If selected: collects every currently *enabled* connection gene
    // (disabled connections are never eligible). If none exist, returns
    // false, leaving genome and innovationTracker unchanged. Otherwise
    // draws one random starting index into that eligible set from the
    // owned generator, then *inspects* candidates in that order, wrapping
    // around at most once, without modifying genome and -- critically --
    // without allocating anything from innovationTracker: only
    // innovationTracker.findNodeSplitInnovation() (a read-only lookup) and
    // innovationTracker.getNextAvailableNodeId() (a pure query) are used
    // during inspection, so a candidate that ends up rejected or skipped
    // never consumes a node ID or innovation number. For the candidate
    // source -> target (innovation oldInnovation):
    //   - If innovationTracker.findNodeSplitInnovation(oldInnovation,
    //     source, target) returns a previously recorded split (this exact
    //     structural event already happened, possibly in another genome):
    //       - If genome has no node with that split's newNodeId, this is a
    //         clean, reusable candidate -- select it immediately.
    //       - If genome already has a node there: throws
    //         std::invalid_argument if it is not Hidden, or if either new
    //         connection's slot is already occupied by a gene with a
    //         conflicting innovation number (a structural inconsistency,
    //         never silently repaired). Otherwise this exact split already
    //         exists in genome; unsuitable, move on to the next candidate.
    //   - If no split has been recorded yet, predicts the node ID a fresh
    //     allocation would receive (innovationTracker.getNextAvailableNodeId(),
    //     unread, not allocated) and checks it against genome the same
    //     way: an existing non-Hidden node there throws, an existing
    //     connection already occupying either of the split's directed
    //     pairs throws (no historical marking could legitimately justify
    //     it yet), an existing unrelated Hidden node with no conflicting
    //     connection is skipped as unsuitable, and no existing node at all
    //     means a clean, brand-new candidate -- select it immediately.
    // Once a candidate is selected (not merely inspected), exactly one call
    // to innovationTracker.getNodeSplitInnovation(oldInnovation, source,
    // target) is made -- the *only* point at which the tracker's state can
    // change, reused idempotently if this exact split was already
    // recorded. The original connection is then disabled (never removed),
    // a new Hidden NodeGene is added with the returned newNodeId, and
    // source -> newNodeId (weight exactly 1.0, enabled, innovation
    // incomingInnovation) and newNodeId -> target (weight exactly
    // oldWeight, enabled, innovation outgoingInnovation) are added via
    // Genome::addConnection(). Returns true.
    // If every eligible candidate turns out unsuitable (already split in
    // this exact genome), returns false without modifying genome or
    // innovationTracker. No existing node or connection gene's identity
    // (source/target ID, innovation number) is ever changed, and no gene
    // besides the disabled original connection is altered.
    bool mutateAddNode(Genome& genome, InnovationTracker& innovationTracker, const MutationConfig& config);

private:
    std::mt19937 m_rng;
};

} // namespace ai::neat
