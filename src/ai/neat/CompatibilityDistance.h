#pragma once

#include <cstddef>

#include "ai/neat/CompatibilityConfig.h"
#include "ai/neat/Genome.h"

namespace ai::neat
{

// The full result of one compatibilityDistance() calculation, useful for
// debugging/visualization without needing to recompute the intermediate
// counts separately. Pure data, produced fresh by every call -- never
// mutated in place, never cached.
struct CompatibilityBreakdown
{
    std::size_t matching = 0;
    std::size_t disjoint = 0;
    std::size_t excess = 0;
    float averageWeightDifference = 0.0f;
    float normalization = 1.0f;
    float distance = 0.0f;
};

// Computes the standard NEAT-style compatibility distance between two
// Genomes, using connection genes only -- node genes never directly enter
// the calculation. A pure, stateless, deterministic function: it never
// mutates genomeA or genomeB, never draws randomness, never consults
// InnovationTracker, never builds a phenotype, and depends on nothing but
// its three arguments (no fitness, no Car/AIController/FitnessEvaluator).
//
//   delta = c1 * E / N + c2 * D / N + c3 * W
//
// Throws std::invalid_argument if:
//   - any of config's three coefficients is non-finite or negative;
//   - config.smallGenomeNormalizationThreshold is zero;
//   - either genome contains two different ConnectionGenes with the same
//     innovation number (Genome::validate() does not check this);
//   - the same innovation number exists in both genomes but with a
//     different source or target ID.
//
// Connection genes are aligned strictly by innovation number -- never by
// vector index, source/target pair, or insertion order. A gene is
// "matching" when its innovation number exists in both genomes (its
// enabled/disabled state never affects matching, disjoint/excess
// classification, or the distance -- there is no separate enabled-state
// term). For each matching pair, weightDifference = abs(weightA -
// weightB); W is their mean (0 if there are no matching genes at all).
//
// A gene whose innovation number exists in only one genome is disjoint or
// excess depending on where it falls relative to the *other* genome's
// highest innovation number (maxInnovationOther, treated as "below every
// non-negative innovation number" when the other genome has no connections
// at all, so every gene of a non-empty genome is excess relative to an
// empty one): innovation > maxInnovationOther -> excess (E), otherwise ->
// disjoint (D). This is a pure innovation-number range test, independent
// of vector position in either genome.
//
// N is the normalization factor: letting largerGenomeSize = the greater of
// the two genomes' connection counts, N = 1 when largerGenomeSize is below
// config.smallGenomeNormalizationThreshold, otherwise N = largerGenomeSize.
// This keeps small genomes (the common case early on, before genomes have
// accumulated many genes) from being over-penalized by division, while
// still normalizing by genome size once genomes grow past the threshold.
//
// distance(A, B) == distance(B, A), and the breakdown's matching/disjoint/
// excess counts are likewise symmetric -- classifying "only in A" first and
// "only in B" second internally does not depend on argument order.
float compatibilityDistance(const Genome& genomeA, const Genome& genomeB, const CompatibilityConfig& config);

// Same calculation as compatibilityDistance(), returning every intermediate
// count/value alongside the final distance (breakdown.distance ==
// compatibilityDistance(genomeA, genomeB, config)). Throws under the exact
// same conditions.
CompatibilityBreakdown compatibilityBreakdown(const Genome& genomeA, const Genome& genomeB,
                                               const CompatibilityConfig& config);

} // namespace ai::neat
