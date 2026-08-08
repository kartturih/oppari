#pragma once

#include <cstdint>
#include <random>

#include "ai/neat/CrossoverConfig.h"
#include "ai/neat/Genome.h"

namespace ai::neat
{

// Deterministically crosses over two parent Genomes into one child Genome,
// per standard NEAT crossover: connection genes are aligned strictly by
// innovation number (never by vector index or endpoint pair). Matching
// genes (same innovation present in both parents) are inherited from a
// randomly chosen parent's copy, per gene. Non-matching genes (innovation
// present in only one parent) are inherited entirely from the fitter
// parent, or -- when fitness is exactly equal -- independently with 50%
// probability from either parent, skipping any candidate that would make
// the child structurally invalid (a duplicate directed connection or an
// enabled cycle) rather than ever inventing a replacement gene.
//
// Owns its own std::mt19937 RNG, seeded exactly once at construction --
// never std::random_device, never a global/static generator, never rand(),
// never time-based. Every crossover() call continues drawing from that same
// generator, so the same seed driven through the same sequence of calls on
// the same parent genomes/fitnesses/config always reproduces the same
// child; a different seed may produce a different child wherever this
// stage's randomness applies (matching-gene selection, the disabled-gene
// decision, and -- for equal-fitness parents only -- non-matching-gene
// inclusion).
//
// This stage (10) implements crossover only: it knows nothing about
// compatibility distance, species, population, selection, reproduction,
// generations, or fitness beyond the two float values passed in, and never
// calls InnovationTracker or mutates a parent Genome.
class GenomeCrossover
{
public:
    explicit GenomeCrossover(std::uint32_t seed);

    // Produces one child Genome from parentA/parentB and their fitness
    // values.
    //
    // Throws std::invalid_argument, in this order, before any RNG draw or
    // any modification of the child under construction -- so an invalid
    // call never partially advances this object's RNG state -- if:
    //   1. either of config's two probability fields is non-finite or
    //      outside [0,1];
    //   2. fitnessA or fitnessB is non-finite (NaN or +/-infinity);
    //   3. either parent fails its own Genome::validate();
    //   4. either parent contains two different ConnectionGenes with the
    //      same innovation number (Genome::validate() alone does not check
    //      this);
    //   5. the same NodeId exists in both parents with a different
    //      NodeType;
    //   6. a matching innovation (present in both parents) has a different
    //      source or target ID between the two parents.
    // It also throws std::invalid_argument if the resulting child fails
    // Genome::validate() or ai::neat::buildPhenotype() -- e.g. a
    // deterministic fitter-parent inheritance that unexpectedly produces a
    // cycle despite both parents individually being valid. This is never
    // silently repaired.
    //
    // Otherwise: for each matching innovation, in ascending order, draws
    // once against config.matchingGeneChooseParentAProbability to choose
    // which parent's source/target/weight the child inherits for that
    // gene; if both parents' copies are enabled the child's copy is
    // enabled with no further draw, otherwise draws once more against
    // config.disabledGeneRemainDisabledProbability to decide the child
    // copy's enabled state (independent of which parent's weight was
    // chosen). For each non-matching innovation (present in only one
    // parent): if one parent has strictly greater fitness, every
    // non-matching gene from that fitter parent is inherited as-is
    // (unchanged weight/enabled/source/target/innovation) and every
    // non-matching gene from the other parent is dropped entirely, with no
    // RNG draw; if fitness is exactly equal, every non-matching gene from
    // either parent, in ascending innovation order, is independently drawn
    // for inclusion with 50% probability, and an included candidate is
    // only actually added if doing so would not create a duplicate
    // directed connection or a cycle among the child's enabled connections
    // decided so far -- otherwise it is skipped, never repaired or
    // substituted.
    //
    // The child's node set is the union of parentA's and parentB's
    // Input/Bias/Output nodes (always included, by NodeId, regardless of
    // which connections were inherited) plus every Hidden node referenced
    // by an inherited connection (looked up by NodeId from whichever
    // parent defines it -- an unreferenced Hidden node from either parent
    // is never included). No node ID or innovation number is ever
    // invented, renumbered, or reassigned, and InnovationTracker is never
    // consulted. Child nodes are stored ascending by node ID; child
    // connections are stored ascending by innovation number -- independent
    // of either parent's own storage order.
    //
    // Neither parentA nor parentB is ever modified.
    Genome crossover(const Genome& parentA, float fitnessA, const Genome& parentB, float fitnessB,
                      const CrossoverConfig& config);

private:
    std::mt19937 m_rng;
};

} // namespace ai::neat
