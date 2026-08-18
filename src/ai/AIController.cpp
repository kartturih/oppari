#include "ai/AIController.h"

#include <algorithm>
#include <utility>

namespace ai
{

namespace
{

// Output 0 = steering, output 1 = throttle, output 2 = brake (matches
// PhenotypeBuilder's ordering).
static_assert(NeuralNetwork::kOutputCount == 3,
              "AIController assumes exactly three network outputs: steering, throttle, brake");

constexpr int kSteeringOutputIndex = 0;
constexpr int kThrottleOutputIndex = 1;
constexpr int kBrakeOutputIndex = 2;

// Clamp is defensive (tanh outputs are already in [-1,1]).
float mapSteering(float rawSteering)
{
    return std::clamp(rawSteering, -1.0f, 1.0f);
}

float mapThrottle(float rawThrottle)
{
    return std::clamp((rawThrottle + 1.0f) * 0.5f, 0.0f, 1.0f);
}

// Same [-1,1] -> [0,1] mapping as throttle -- brake is a fully independent
// output (never mutually exclusive with throttle; see Car.cpp's friction
// circle for why simultaneous throttle+brake is self-defeating rather than
// forbidden).
float mapBrake(float rawBrake)
{
    return std::clamp((rawBrake + 1.0f) * 0.5f, 0.0f, 1.0f);
}

} // namespace

AIController::AIController(NeuralNetwork network)
    : m_network(std::move(network))
{
}

simulation::CarInput AIController::update(const simulation::Car& car)
{
    if (!car.isAlive())
    {
        return simulation::CarInput{};
    }

    m_lastObservation = buildObservation(car);

    const std::array<float, NeuralNetwork::kOutputCount> outputs = m_network.evaluate(m_lastObservation);
    m_rawSteering = outputs[kSteeringOutputIndex];
    m_rawThrottle = outputs[kThrottleOutputIndex];
    m_rawBrake = outputs[kBrakeOutputIndex];

    simulation::CarInput input;
    input.steering = mapSteering(m_rawSteering);
    input.throttle = mapThrottle(m_rawThrottle);
    input.brake = mapBrake(m_rawBrake);
    return input;
}

} // namespace ai
