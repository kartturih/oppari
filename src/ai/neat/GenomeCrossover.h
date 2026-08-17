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
// with 50% probability, skipping any that would make the child invalid (a
// duplicate directed connection or an enabled cycle) rather than repairing it.
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
    // has conflicting endpoints. Also throws if the resulting child fails
    // validate() or buildPhenotype() (e.g. an unexpected cycle).
    //
    // Matching genes: one draw (matchingGeneChooseParentAProbability)
    // chooses the parent copy; if either copy is disabled, one more draw
    // (disabledGeneRemainDisabledProbability) decides the child's enabled
    // state. Non-matching genes: see class comment. Child nodes are the
    // union of both parents' Input/Bias/Output nodes plus every Hidden node
    // an inherited connection references. Neither parent is modified.
    Genome crossover(const Genome& parentA, float fitnessA, const Genome& parentB, float fitnessB,
                      const CrossoverConfig& config);

private:
    std::mt19937 m_rng;
};

} // namespace ai::neat
