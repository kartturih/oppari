#include "ai/Observation.h"

#include <algorithm>
#include <cmath>

#include "raylib.h"

namespace ai
{

namespace
{

static_assert(simulation::Car::kSensorCount + 4 == kObservationSize,
              "Observation expects one slot per sensor plus four vehicle-state slots");

} // namespace

Observation buildObservation(const simulation::Car& car)
{
    Observation observation;

    const auto& sensors = car.getSensors();
    for (int i = 0; i < simulation::Car::kSensorCount; ++i)
    {
        observation.values[i] = std::clamp(sensors[i].normalizedDistance, 0.0f, 1.0f);
    }

    // Car is the single source of truth for its own configured max speed;
    // Observation only ever reads it, never duplicates the literal.
    const float maxSpeed = car.getMaxSpeed();

    observation.values[5] = std::clamp(car.getSpeed() / maxSpeed, 0.0f, 1.0f);
    observation.values[6] = std::clamp(car.getForwardVelocity() / maxSpeed, -1.0f, 1.0f);
    observation.values[7] = std::clamp(car.getLateralVelocity() / maxSpeed, -1.0f, 1.0f);
    observation.values[8] = std::clamp(car.getSlipAngle() / static_cast<float>(PI), -1.0f, 1.0f);

    return observation;
}

} // namespace ai
