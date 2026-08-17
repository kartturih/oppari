#pragma once

#include <cstddef>

#include "ai/neat/CompatibilityConfig.h"
#include "ai/neat/Genome.h"

namespace ai::neat
{

// Full result of one compatibilityDistance() calculation, for
// debugging/visualization. Fresh data per call, never cached.
struct CompatibilityBreakdown
{
    std::size_t matching = 0;
    std::size_t disjoint = 0;
    std::size_t excess = 0;
    float averageWeightDifference = 0.0f;
    float normalization = 1.0f;
    float distance = 0.0f;
};

// Standard NEAT compatibility distance, connection genes only, aligned
// strictly by innovation number:
//
//   delta = c1 * E / N + c2 * D / N + c3 * W
//
// A gene matches if its innovation exists in both genomes; W is the mean
// abs weight difference over matches (0 if none). A gene present in only
// one genome is excess if its innovation exceeds the other genome's max
// innovation, otherwise disjoint. N = 1 if the larger genome's connection
// count is below smallGenomeNormalizationThreshold, else that count --
// avoids over-penalizing small early genomes.
//
// Pure and stateless (no mutation, RNG, or phenotype build); symmetric:
// distance(A, B) == distance(B, A).
// Throws std::invalid_argument if config is invalid, either genome reuses
// an innovation number, or a matching innovation has conflicting endpoints.
float compatibilityDistance(const Genome& genomeA, const Genome& genomeB, const CompatibilityConfig& config);

// Same calculation, returning every intermediate value (breakdown.distance
// == compatibilityDistance(...)). Same throw conditions.
CompatibilityBreakdown compatibilityBreakdown(const Genome& genomeA, const Genome& genomeB,
                                               const CompatibilityConfig& config);

} // namespace ai::neat
