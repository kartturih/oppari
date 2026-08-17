#pragma once

#include <cstddef>

namespace ai::neat
{

// Tunable coefficients/threshold for compatibilityDistance(), validated by
// that function before use.
struct CompatibilityConfig
{
    float excessCoefficient = 1.0f;         // c1, weight on excess genes / N
    float disjointCoefficient = 1.0f;       // c2, weight on disjoint genes / N
    float weightDifferenceCoefficient = 0.4f; // c3, weight on mean matching-gene weight diff
    std::size_t smallGenomeNormalizationThreshold = 20; // see compatibilityDistance()'s N rule
};

} // namespace ai::neat
