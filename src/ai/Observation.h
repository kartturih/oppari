#pragma once

#include <array>

#include "simulation/Car.h"

namespace ai
{

// Fixed-size, normalized snapshot of a Car's local perception and local
// dynamics, meant to become a future NeuralNetwork's external input vector.
// Deliberately contains no world-space position, absolute heading, or
// track-progress information -- only what the car itself could "feel".
inline constexpr int kObservationSize = 9;

// Conceptual value order (see buildObservation for the exact mapping):
//   0..4 = sensor normalizedDistance, at -60/-30/0/+30/+60 degrees
//   5    = total speed, normalized
//   6    = forward velocity, normalized
//   7    = lateral velocity, normalized
//   8    = slip angle, normalized
struct Observation
{
    std::array<float, kObservationSize> values{};
};

// Builds a normalized Observation from a Car's current perception/state.
// This is AI-layer code: Car itself has no knowledge of Observation.
Observation buildObservation(const simulation::Car& car);

} // namespace ai
