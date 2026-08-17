#pragma once

#include <array>

#include "simulation/Car.h"

namespace ai
{

// Fixed-size, normalized snapshot of a Car's local perception/dynamics,
// used as NeuralNetwork's input vector. No world position, absolute
// heading, or track-progress info -- only what the car itself could "feel".
inline constexpr int kObservationSize = 9;

// Slots: 0..4 = sensor normalizedDistance at -60/-30/0/+30/+60 deg,
// 5 = speed, 6 = forward velocity, 7 = lateral velocity, 8 = slip angle
// (all normalized).
struct Observation
{
    std::array<float, kObservationSize> values{};
};

Observation buildObservation(const simulation::Car& car);

} // namespace ai
