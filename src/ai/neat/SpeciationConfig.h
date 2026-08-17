#pragma once

namespace ai::neat
{

// Tunable threshold for Speciator::speciate(), validated by Speciator
// before use. Separate from CompatibilityConfig: that shapes the distance
// calculation, this decides how close is "close enough" to join a species.
struct SpeciationConfig
{
    float compatibilityThreshold = 3.0f;
};

} // namespace ai::neat
