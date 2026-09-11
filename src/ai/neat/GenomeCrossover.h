#pragma once

#include <cstdint>
#include <random>

#include "ai/neat/CrossoverConfig.h"
#include "ai/neat/Genome.h"

namespace ai::neat
{

// Standard NEAT crossover of two parent Genomes into one child. Connection
// genes align strictly by innovation number (never index or endpoint pair).
// Matching genes (same innovation in both parents) inherit one parent's
// copy at random, per gene. Non-matching genes inherit entirely from the
// fitter parent; on exactly equal fitness, each is independently included
// with 50% probability.
//
// Cycle safety: each parent's own enabled-connection set is independently
// acyclic, but MIXING genes from both parents can still combine into a
// cycle neither parent actually had enabled at once -- the standard NEAT
// crossover pitfall. Every connection considered for the child (matching,
// non-matching, either fitness case) is checked against the child's edges
// assembled so far, in a fixed deterministic order (ascending innovation
// number): a non-matching candidate that would create a duplicate directed
// connection or an enabled cycle is skipped entirely (never repaired); a
// matching gene is never dropped (it has shared ancestry in both parents)
// but is forced disabled instead if enabling it would close a cycle. No
// RNG is used to resolve a cycle -- only to make the ordinary matching-gene/
// re-enable choices every crossover already made, whose result cycle
// safety may then deterministically override.
//
// Owns one std::mt19937, seeded once at construction (never random_device/
// rand()/time-based) -- same seed + same call sequence always reproduces
// the same child.
class GenomeCrossover
{
public:
    explicit GenomeCrossover(std::uint32_t seed);

    // Throws std::invalid_argument before any RNG draw if: config's
    // probabilities are invalid; fitnessA/fitnessB are non-finite; either
    // parent fails validate() or has duplicate innovation numbers; a NodeId
    // has conflicting NodeType between parents; or a matching innovation
    // has conflicting endpoints. Also throws if the resulting child somehow
    // still fails validate() or buildPhenotype() -- defense-in-depth only;
    // the cycle-safety handling above (see class comment) is what actually
    // guarantees this never happens in practice.
    //
    // Matching genes: one draw (matchingGeneChooseParentAProbability)
    // chooses the parent copy; if either copy is disabled, one more draw
    // (disabledGeneRemainDisabledProbability) decides the child's enabled
    // state -- then cycle safety may deterministically force it back to
    // disabled (see class comment). Non-matching genes: see class comment.
    // Child nodes are the union of both parents' Input/Bias/Output nodes
    // plus every Hidden node an inherited (possibly cycle-disabled)
    // connection references. Neither parent is modified.
    Genome crossover(const Genome& parentA, float fitnessA, const Genome& parentB, float fitnessB,
                      const CrossoverConfig& config);

private:
    std::mt19937 m_rng;
};

} // namespace ai::neat
