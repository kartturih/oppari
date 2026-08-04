#include "ai/AIController.h"

#include <algorithm>
#include <utility>

namespace ai
{

namespace
{

// Output ordering contract, matching NeuralNetwork.h and the PhenotypeBuilder
// ordering guarantee: evaluate() returns exactly two values, output 0 is
// steering and output 1 is throttle. kOutputCount is a compile-time
// constant, so "exactly two outputs" is enforced by the type system itself
// -- std::array<float, 2> cannot hold any other number of elements.
static_assert(NeuralNetwork::kOutputCount == 2,
              "AIController assumes exactly two network outputs: steering, throttle");

constexpr int kSteeringOutputIndex = 0;
constexpr int kThrottleOutputIndex = 1;

// Network outputs come from tanh and are already within [-1, 1]; the clamp
// is defensive so this mapping is correct even if that assumption ever
// changes. No braking or reverse: throttle is clamped to [0, 1], never
// negative.
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
