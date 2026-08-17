#include "ai/AIController.h"

#include <algorithm>
#include <utility>

namespace ai
{

namespace
{

// Output 0 = steering, output 1 = throttle (matches PhenotypeBuilder's ordering).
static_assert(NeuralNetwork::kOutputCount == 2,
              "AIController assumes exactly two network outputs: steering, throttle");

constexpr int kSteeringOutputIndex = 0;
constexpr int kThrottleOutputIndex = 1;

// Clamp is defensive (tanh outputs are already in [-1,1]). No braking/reverse.
float mapSteering(float rawSteering)
{
    return std::clamp(rawSteering, -1.0f, 1.0f);
}

float mapThrottle(float rawThrottle)
{
    return std::clamp((rawThrottle + 1.0f) * 0.5f, 0.0f, 1.0f);
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

    simulation::CarInput input;
    input.steering = mapSteering(m_rawSteering);
    input.throttle = mapThrottle(m_rawThrottle);
    return input;
}

} // namespace ai
