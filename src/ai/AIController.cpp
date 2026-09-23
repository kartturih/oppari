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

} // namespace

float AIController::mapThrottle(float rawThrottle)
{
    return std::clamp((rawThrottle + 1.0f) * 0.5f, 0.0f, 1.0f);
}

// Unlike throttle, neutral (or negative) raw output means NO brake request --
// a network that hasn't learned to brake yet must not coast at an effective
// 50% brake.
float AIController::mapBrake(float rawBrake)
{
    return std::clamp(rawBrake, 0.0f, 1.0f);
}

// Car.cpp still allows throttle and brake together (its friction circle just
// makes that self-defeating); this combiner is what keeps the controller from
// ever asking for both. The final clamps are defensive -- requests are already
// in [0, 1], so the difference is already in [-1, 1].
AIController::LongitudinalCommand AIController::combineLongitudinal(float throttleRequest, float brakeRequest)
{
    const float longitudinal = throttleRequest - brakeRequest;
    LongitudinalCommand command;
    command.throttle = std::clamp(std::max(0.0f, longitudinal), 0.0f, 1.0f);
    command.brake = std::clamp(std::max(0.0f, -longitudinal), 0.0f, 1.0f);
    return command;
}

AIController::AIController(NeuralNetwork network)
    : m_network(std::move(network))
{
}

simulation::CarInput AIController::update(const simulation::Car& car, const simulation::TrackProgress& progress)
{
    if (!car.isAlive())
    {
        return simulation::CarInput{};
    }

    m_lastObservation = buildObservation(car, progress);

    const std::array<float, NeuralNetwork::kOutputCount> outputs = m_network.evaluate(m_lastObservation);
    m_rawSteering = outputs[kSteeringOutputIndex];
    m_rawThrottle = outputs[kThrottleOutputIndex];
    m_rawBrake = outputs[kBrakeOutputIndex];

    m_throttleRequest = mapThrottle(m_rawThrottle);
    m_brakeRequest = mapBrake(m_rawBrake);
    const LongitudinalCommand longitudinal = combineLongitudinal(m_throttleRequest, m_brakeRequest);

    simulation::CarInput input;
    input.steering = mapSteering(m_rawSteering);
    input.throttle = longitudinal.throttle;
    input.brake = longitudinal.brake;
    return input;
}

} // namespace ai
