#pragma once

#include "ai/NeuralNetwork.h"
#include "ai/Observation.h"
#include "simulation/Car.h"
#include "simulation/TrackProgress.h"

namespace ai
{

// Closes the control loop: Car -> Observation -> NeuralNetwork -> CarInput.
// Owns its NeuralNetwork; knows nothing about NEAT genetics. Pure runtime
// adapter -- no mutation/crossover/training.
//
// No command-level steering smoothing lives here (an earlier experiment
// added a downstream slew-rate limiter on CarInput.steering; removed after
// it measurably hurt every driving metric in a full training comparison --
// see the control-policy investigation this was built for). Adding uniform
// latency to every steering change, needed or not, forced evolution to slow
// the whole car down to tolerate its own now-sluggish steering, rather than
// making the network itself learn smoother commands. Steering-command
// smoothness is instead shaped through FitnessEvaluator's steering-change
// penalty (see FitnessEvaluator.h) -- an evolutionary-pressure fix, not a
// physical constraint, so a genuinely necessary single fast reaction is
// still free, and only sustained oscillation actually costs fitness.
class AIController
{
public:
    explicit AIController(NeuralNetwork network);

    // Builds an Observation from car/progress, evaluates the network, maps
    // outputs to a CarInput. progress supplies only the local track
    // tangent/direction (for the heading-error observation, see
    // Observation.h) -- never a position/waypoint/target the controller
    // could follow. If the car is dead, returns a neutral CarInput without
    // evaluating; getRawThrottleOutput()/getRawBrakeOutput()/
    // getLastObservation() keep reporting the last live evaluation.
    simulation::CarInput update(const simulation::Car& car, const simulation::TrackProgress& progress);

    // Raw network outputs from the most recent live update(), before mapping.
    float getRawSteeringOutput() const { return m_rawSteering; }
    float getRawThrottleOutput() const { return m_rawThrottle; }
    float getRawBrakeOutput() const { return m_rawBrake; }

    // Observation from the most recent live update() (default if none yet).
    const Observation& getLastObservation() const { return m_lastObservation; }

    // The network has separate throttle and brake outputs, each mapped to a
    // [0, 1] REQUEST, and the two requests are then combined into one net
    // longitudinal command (see combineLongitudinal()) -- so the CarInput can
    // never carry throttle and brake at the same time.
    //
    // Request mappings, exposed for direct verification. Throttle:
    // (raw + 1) / 2, so a neutral raw output (0) requests 50% throttle.
    // Brake: neutral/negative raw output requests no brake (0 is not "50%
    // brake"), positive raw output ramps linearly up to full brake at +1.0.
    static float mapThrottle(float rawThrottle);
    static float mapBrake(float rawBrake);

    // Throttle/brake pair actually sent to the car.
    struct LongitudinalCommand
    {
        float throttle = 0.0f;
        float brake = 0.0f;
    };

    // longitudinal = throttleRequest - brakeRequest, then
    // throttle = max(0, longitudinal) and brake = max(0, -longitudinal), both
    // clamped to [0, 1]. Equal requests cancel to (0, 0); no dead zone,
    // smoothing or hysteresis. min(throttle, brake) is always 0. A brake
    // request below the throttle request therefore acts as a throttle trim.
    static LongitudinalCommand combineLongitudinal(float throttleRequest, float brakeRequest);

    // The [0, 1] requests (post mapThrottle()/mapBrake(), pre-combination)
    // from the most recent live update() -- diagnostics only.
    float getThrottleRequest() const { return m_throttleRequest; }
    float getBrakeRequest() const { return m_brakeRequest; }

private:
    NeuralNetwork m_network;
    Observation m_lastObservation;
    float m_rawSteering = 0.0f;
    float m_rawThrottle = 0.0f;
    float m_rawBrake = 0.0f;
    float m_throttleRequest = 0.0f;
    float m_brakeRequest = 0.0f;
};

} // namespace ai
