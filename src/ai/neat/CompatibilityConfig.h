#pragma once

#include <cstddef>

namespace ai::neat
{

// Tunable coefficients/threshold for compatibilityDistance(). Pure data --
// it performs no validation itself; compatibilityDistance() validates every
// field (throwing std::invalid_argument on an invalid one) before using
// them.
//
// excessCoefficient (c1) / disjointCoefficient (c2) -- weight applied to
// the excess/disjoint connection-gene counts, each divided by the
// normalization factor N.
//
// weightDifferenceCoefficient (c3) -- weight applied to the mean absolute
// weight difference of matching connection genes (not divided by N).
//
// smallGenomeNormalizationThreshold -- see compatibilityDistance()'s own
// doc comment for the exact normalization rule this drives.
struct CompatibilityConfig
{
    float excessCoefficient = 1.0f;
    float disjointCoefficient = 1.0f;
    float weightDifferenceCoefficient = 0.4f;
    std::size_t smallGenomeNormalizationThreshold = 20;
};

} // namespace ai::neat
