#pragma once

namespace ai::neat
{

// Tunable threshold for Speciator::speciate(). Pure data -- it performs no
// validation itself; Speciator validates it (throwing std::invalid_argument
// on an invalid value) before using it. Deliberately does not duplicate any
// CompatibilityConfig field -- the two configs are passed to speciate()
// separately and serve different purposes (CompatibilityConfig shapes the
// distance calculation itself; compatibilityThreshold only decides how
// close is "close enough" to join a species).
struct SpeciationConfig
{
    float compatibilityThreshold = 3.0f;
};

} // namespace ai::neat
