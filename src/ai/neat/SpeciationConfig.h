#pragma once

#include <cstddef>

namespace ai::neat
{

// Tunable threshold for Speciator::speciate(), validated by Speciator
// before use. Separate from CompatibilityConfig: that shapes the distance
// calculation, this decides how close is "close enough" to join a species.
//
// compatibilityThreshold is only the INITIAL value: Population maintains its
// own runtime-adaptive threshold, starting from this value and adjusted by
// +/- compatibilityThresholdAdjustment once per completed generation (never
// mid-speciation-pass) to nudge that generation's species count toward
// [targetSpeciesMin, targetSpeciesMax] -- decrease the threshold if there
// were too few species, increase it if there were too many, hold it
// otherwise. Deliberately simple (fixed-step, no proportional/PID control),
// and clamped to [minimumCompatibilityThreshold, maximumCompatibilityThreshold]
// after every adjustment. This struct itself is never mutated by that
// process -- see Population::m_currentCompatibilityThreshold.
struct SpeciationConfig
{
    float compatibilityThreshold = 3.0f;

    std::size_t targetSpeciesMin = 5;
    std::size_t targetSpeciesMax = 10;

    float compatibilityThresholdAdjustment = 0.1f;

    float minimumCompatibilityThreshold = 0.5f;
    float maximumCompatibilityThreshold = 10.0f;
};

} // namespace ai::neat
