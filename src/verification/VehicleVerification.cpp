#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "raylib.h"

#include "ai/AIController.h"
#include "ai/FitnessEvaluator.h"
#include "ai/NeuralNetwork.h"
#include "ai/Observation.h"
#include "ai/neat/CompatibilityConfig.h"
#include "ai/neat/CompatibilityDistance.h"
#include "ai/neat/ConnectionGene.h"
#include "ai/neat/CrossoverConfig.h"
#include "ai/neat/Genome.h"
#include "ai/neat/GenomeCrossover.h"
#include "ai/neat/GenomeMutator.h"
#include "ai/neat/Individual.h"
#include "ai/neat/InnovationTracker.h"
#include "ai/neat/MutationConfig.h"
#include "ai/neat/NodeGene.h"
#include "ai/neat/PhenotypeBuilder.h"
#include "ai/neat/Population.h"
#include "ai/neat/PopulationConfig.h"
#include "ai/neat/SpeciationConfig.h"
#include "ai/neat/Speciator.h"
#include "ai/neat/Species.h"
#include "simulation/Car.h"
#include "simulation/Track.h"
#include "simulation/TrackProgress.h"
#include "simulation/TrackVisual.h"
#include "training/GenerationMetrics.h"
#include "training/TrainingLogger.h"

#include "AppConfig.h"
#include "verification/Verifications.h"

namespace verification
{

using app::createDemonstrationGenome;
using app::kSimulationDt;
using app::kSpawnHeading;
using app::kSpawnPosition;
using app::makeCarParams;


// One-shot, deterministic sanity check of Car's dynamics and collision,
// independent of any keyboard/render timing. Runs once at startup.
void verifyCar(const simulation::Track& track)
{
    simulation::Car car(makeCarParams(), track);
    car.reset(kSpawnPosition, kSpawnHeading);

    assert(car.isAlive() && "car must be spawned alive");
    assert(car.getPosition().x == kSpawnPosition.x && car.getPosition().y == kSpawnPosition.y &&
           "reset must place the car at the spawn position");

    // Steering alone, while stationary, must not rotate the car.
    simulation::CarInput steerOnly;
    steerOnly.throttle = 0.0f;
    steerOnly.steering = 1.0f;
    for (int i = 0; i < 30; ++i)
    {
        car.update(steerOnly, kSimulationDt);
    }
    // Not bit-for-bit equal: Box2D stores rotation as (cos, sin) and
    // getHeading() reconstructs via atan2 every update(), so even an
    // unchanging heading accumulates tiny round-trip drift.
    assert(std::fabs(car.getHeading() - kSpawnHeading) < 1e-3f &&
           "steering must have no effect while stationary");

    // Sustained throttle must build up speed from rest.
    simulation::CarInput throttleOnly;
    throttleOnly.throttle = 1.0f;
    throttleOnly.steering = 0.0f;
    for (int i = 0; i < 30; ++i)
    {
        car.update(throttleOnly, kSimulationDt);
    }
    const Vector2 v = car.getVelocity();
    const float speed = std::sqrt(v.x * v.x + v.y * v.y);
    assert(speed > 1.0f && "sustained throttle must build up visible speed");
    assert(speed <= car.getParams().maxSpeed + 0.01f && "speed must never exceed maxSpeed");

    // Driving straight off the track must kill the car and zero its velocity.
    car.reset(kSpawnPosition, kSpawnHeading);
    simulation::CarInput driveOffTrack;
    driveOffTrack.throttle = 1.0f;
    driveOffTrack.steering = 0.0f;
    for (int i = 0; i < 300 && car.isAlive(); ++i)
    {
        car.update(driveOffTrack, kSimulationDt);
    }
    assert(!car.isAlive() && "driving straight for 5s must leave the road band and kill the car");
    const Vector2 deadVelocity = car.getVelocity();
    assert(deadVelocity.x == 0.0f && deadVelocity.y == 0.0f && "a dead car must have zero velocity");

    // A dead car must ignore further input.
    const Vector2 deadPosition = car.getPosition();
    car.update(throttleOnly, kSimulationDt);
    assert(car.getPosition().x == deadPosition.x && car.getPosition().y == deadPosition.y &&
           "a dead car must not respond to further input");

    // Reset must revive the car.
    car.reset(kSpawnPosition, kSpawnHeading);
    assert(car.isAlive() && "reset must revive the car");
    assert(car.getVelocity().x == 0.0f && car.getVelocity().y == 0.0f && "reset must zero velocity");

    TraceLog(LOG_INFO, "Car verification: all deterministic checks passed");
}

// One-shot, deterministic sanity check of Car's five sensors, independent of
// keyboard/render timing. Runs once at startup.
void verifySensors(const simulation::Track& track)
{
    static_assert(simulation::Car::kSensorCount == 5, "Stage 3 requires exactly five sensors");

    simulation::Car car(makeCarParams(), track);
    car.reset(kSpawnPosition, kSpawnHeading);

    // 1: exactly five sensor readings exist, in fixed-size storage.
    const auto& sensors = car.getSensors();
    assert(sensors.size() == 5 && "exactly five sensor readings must exist");

    // 2 & 3: normalized readings stay within [0,1] and are valid immediately after reset.
    for (const simulation::SensorReading& s : sensors)
    {
        assert(s.normalizedDistance >= 0.0f && s.normalizedDistance <= 1.0f &&
               "normalized sensor distance must stay within [0,1]");
        assert(s.distance >= 0.0f && s.distance <= simulation::Car::kMaxSensorDistance &&
               "raw sensor distance must stay within [0, kMaxSensorDistance]");
    }

    // 4: the front (0 degree) sensor points in the same world direction as the car heading.
    {
        const Vector2 origin = car.getSensorOrigin();
        const simulation::SensorReading& front = sensors[2];
        const float rayAngle = std::atan2(front.endPoint.y - origin.y, front.endPoint.x - origin.x);
        assert(std::fabs(rayAngle - kSpawnHeading) < 0.01f &&
               "front sensor must point along the car heading");
    }

    // 5: resetting to a known heading rotates the whole sensor layout with it.
    {
        const float knownHeading = static_cast<float>(PI) * 0.5f; // pointing +y ("down")
        car.reset(kSpawnPosition, knownHeading);
        const Vector2 origin = car.getSensorOrigin();
        const auto& rotatedSensors = car.getSensors();

        for (int i = 0; i < simulation::Car::kSensorCount; ++i)
        {
            const float expectedAngle = knownHeading + simulation::Car::kSensorAngleDegrees[i] * DEG2RAD;
            const Vector2 expectedDir = {std::cos(expectedAngle), std::sin(expectedAngle)};
            const Vector2 rayDir = {rotatedSensors[i].endPoint.x - origin.x, rotatedSensors[i].endPoint.y - origin.y};
            const float rayLen = std::sqrt(rayDir.x * rayDir.x + rayDir.y * rayDir.y);
            assert(rayLen > 0.01f && "sensor ray must have nonzero length");
            const Vector2 rayDirNorm = {rayDir.x / rayLen, rayDir.y / rayLen};
            const float dot = rayDirNorm.x * expectedDir.x + rayDirNorm.y * expectedDir.y;
            assert(dot > 0.999f && "rotated sensor direction must match heading + relative angle");
        }

        car.reset(kSpawnPosition, kSpawnHeading);
    }

    // 6: a sensor aimed across the road (90 deg off spawn heading) must hit
    // the boundary well within the 200px range, for any track's spawn geometry.
    {
        car.reset(kSpawnPosition, kSpawnHeading + static_cast<float>(PI) * 0.5f);
        const simulation::SensorReading& front = car.getSensors()[2];
        assert(front.distance < simulation::Car::kMaxSensorDistance &&
               front.normalizedDistance < 1.0f &&
               "sensor aimed at a nearby boundary must report less than maximum distance");
        car.reset(kSpawnPosition, kSpawnHeading);
    }

    // 7: a ray with no obstacle within range reports exactly maximum distance / normalized 1.0.
    // At spawn heading, the front sensor points along the spawn straight, clear for 200px.
    {
        const simulation::SensorReading& front = car.getSensors()[2];
        assert(front.distance == simulation::Car::kMaxSensorDistance &&
               "unobstructed sensor must report exactly the maximum distance");
        assert(front.normalizedDistance == 1.0f &&
               "unobstructed sensor must report exactly normalized 1.0");
    }

    // 8: sensor endpoints correspond to their recorded distance and direction.
    {
        const Vector2 origin = car.getSensorOrigin();
        for (int i = 0; i < simulation::Car::kSensorCount; ++i)
        {
            const float angle = kSpawnHeading + simulation::Car::kSensorAngleDegrees[i] * DEG2RAD;
            const Vector2 direction = {std::cos(angle), std::sin(angle)};
            const simulation::SensorReading& s = sensors[i];
            const Vector2 expectedEnd = {origin.x + direction.x * s.distance, origin.y + direction.y * s.distance};
            assert(std::fabs(expectedEnd.x - s.endPoint.x) < 0.01f &&
                   std::fabs(expectedEnd.y - s.endPoint.y) < 0.01f &&
                   "sensor endpoint must match its recorded distance and direction");
        }
    }

    // 9: a dead car retains its final sensor readings and does not recast while stationary.
    {
        car.reset(kSpawnPosition, kSpawnHeading);
        simulation::CarInput driveOffTrack;
        driveOffTrack.throttle = 1.0f;
        driveOffTrack.steering = 0.0f;
        for (int i = 0; i < 300 && car.isAlive(); ++i)
        {
            car.update(driveOffTrack, kSimulationDt);
        }
        assert(!car.isAlive() && "driving straight for 5s must leave the road band and kill the car");

        const auto finalSensors = car.getSensors();
        car.update(driveOffTrack, kSimulationDt);
        const auto& sensorsAfterDeadUpdate = car.getSensors();
        for (int i = 0; i < simulation::Car::kSensorCount; ++i)
        {
            assert(finalSensors[i].distance == sensorsAfterDeadUpdate[i].distance &&
                   finalSensors[i].normalizedDistance == sensorsAfterDeadUpdate[i].normalizedDistance &&
                   "a dead car must retain its final sensor readings, not recast them");
        }
    }

    car.reset(kSpawnPosition, kSpawnHeading);

    TraceLog(LOG_INFO, "Sensor verification: all deterministic checks passed");
}

// One-shot, deterministic sanity check of ai::buildObservation, independent
// of keyboard/render timing. Runs once at startup.
void verifyObservation(const simulation::Track& track)
{
    static_assert(ai::kObservationSize == 9, "Stage 4 requires exactly nine observation values");

    simulation::Car car(makeCarParams(), track);
    car.reset(kSpawnPosition, kSpawnHeading);

    // 1: exactly nine values, fixed-size storage.
    ai::Observation obs = ai::buildObservation(car);
    assert(obs.values.size() == 9 && "observation must contain exactly nine values");

    // 2 & 3: sensor values occupy indices 0..4, in documented order, within [0,1].
    const auto& sensors = car.getSensors();
    for (int i = 0; i < simulation::Car::kSensorCount; ++i)
    {
        assert(std::fabs(obs.values[i] - sensors[i].normalizedDistance) < 1e-6f &&
               "observation sensor slot must match Car's own normalized sensor reading");
        assert(obs.values[i] >= 0.0f && obs.values[i] <= 1.0f && "sensor observation values must stay within [0,1]");
    }

    // 4: speed normalization (speed / maxSpeed, clamped to [0,1]).
    {
        const float expected = std::clamp(car.getSpeed() / car.getMaxSpeed(), 0.0f, 1.0f);
        assert(std::fabs(obs.values[5] - expected) < 1e-6f && "speed normalization mismatch");
    }

    // 5: forward velocity normalization (forwardVelocity / maxSpeed, clamped to [-1,1]).
    {
        const float expected = std::clamp(car.getForwardVelocity() / car.getMaxSpeed(), -1.0f, 1.0f);
        assert(std::fabs(obs.values[6] - expected) < 1e-6f && "forward velocity normalization mismatch");
    }

    // 6: lateral velocity normalization (lateralVelocity / maxSpeed, clamped to [-1,1]).
    {
        const float expected = std::clamp(car.getLateralVelocity() / car.getMaxSpeed(), -1.0f, 1.0f);
        assert(std::fabs(obs.values[7] - expected) < 1e-6f && "lateral velocity normalization mismatch");
    }

    // 7: slip angle normalization (slipAngle / pi, clamped to [-1,1]).
    {
        const float expected = std::clamp(car.getSlipAngle() / static_cast<float>(PI), -1.0f, 1.0f);
        assert(std::fabs(obs.values[8] - expected) < 1e-6f && "slip angle normalization mismatch");
    }

    // 8: all normalized values respect their documented ranges even under active driving/sliding.
    simulation::CarInput throttleAndSteer;
    throttleAndSteer.throttle = 1.0f;
    throttleAndSteer.steering = 1.0f;
    for (int i = 0; i < 30 && car.isAlive(); ++i)
    {
        car.update(throttleAndSteer, kSimulationDt);
    }
    const ai::Observation movingObs = ai::buildObservation(car);
    for (int i = 0; i < simulation::Car::kSensorCount; ++i)
    {
        assert(movingObs.values[i] >= 0.0f && movingObs.values[i] <= 1.0f && "sensor values must stay within [0,1]");
    }
    assert(movingObs.values[5] >= 0.0f && movingObs.values[5] <= 1.0f && "speed must stay within [0,1]");
    assert(movingObs.values[6] >= -1.0f && movingObs.values[6] <= 1.0f && "forward velocity must stay within [-1,1]");
    assert(movingObs.values[7] >= -1.0f && movingObs.values[7] <= 1.0f && "lateral velocity must stay within [-1,1]");
    assert(movingObs.values[8] >= -1.0f && movingObs.values[8] <= 1.0f && "slip angle must stay within [-1,1]");

    // 9: reset/spawn produces deterministic valid observation values.
    car.reset(kSpawnPosition, kSpawnHeading);
    const ai::Observation resetObsA = ai::buildObservation(car);
    car.reset(kSpawnPosition, kSpawnHeading);
    const ai::Observation resetObsB = ai::buildObservation(car);
    for (int i = 0; i < ai::kObservationSize; ++i)
    {
        assert(resetObsA.values[i] == resetObsB.values[i] && "reset must produce deterministic observation values");
    }

    TraceLog(LOG_INFO, "Observation verification: all deterministic checks passed");
}

// Deterministic behavioral check of the front/rear bicycle-model tire
// model: standing-start full throttle/steering curves without sliding
// sideways, ordinary cornering keeps slip angles small, turn radius widens
// with speed, grip loss under hard turns is progressive, steering release
// lets slip settle, straight-line acceleration is stable, and nothing ever
// produces a NaN or unbounded spin.
void verifyVehiclePhysics(const simulation::Track& track)
{
    simulation::Car car(makeCarParams(), track);

    // getHeading() wraps into (-pi, pi] (atan2), so a continuous rotation
    // can jump by ~2*pi crossing that branch cut -- always use this
    // shortest-path signed delta, never a naive subtraction.
    constexpr float kTwoPi = 2.0f * static_cast<float>(PI);
    auto headingDelta = [](float from, float to) -> float
    {
        float delta = to - from;
        while (delta > PI)
        {
            delta -= kTwoPi;
        }
        while (delta < -PI)
        {
            delta += kTwoPi;
        }
        return delta;
    };

    auto allFinite = [](const simulation::Car& c) -> bool
    {
        return std::isfinite(c.getPosition().x) && std::isfinite(c.getPosition().y) && std::isfinite(c.getHeading()) &&
               std::isfinite(c.getVelocity().x) && std::isfinite(c.getVelocity().y);
    };

    // 1: sustained throttle must increase forward speed over time.
    {
        car.reset(kSpawnPosition, kSpawnHeading);
        simulation::CarInput throttleOnly;
        throttleOnly.throttle = 1.0f;
        throttleOnly.steering = 0.0f;

        const float speedAtStart = car.getSpeed();
        for (int i = 0; i < 60; ++i)
        {
            car.update(throttleOnly, kSimulationDt);
        }
        const float speedAfter1s = car.getSpeed();
        assert(speedAtStart < 1.0f && "car must start from rest");
        assert(speedAfter1s > speedAtStart + 20.0f &&
               "sustained throttle must visibly increase forward speed within 1s");
        assert(car.getForwardVelocity() > 0.0f && "sustained throttle must produce positive forward velocity");
    }

    // 2: releasing throttle (lift-off) must let resistance reduce speed --
    // there is no separate brake input.
    {
        const float speedBeforeLiftOff = car.getSpeed();
        assert(speedBeforeLiftOff > 50.0f && "must still be moving meaningfully before lift-off");

        simulation::CarInput coasting; // throttle = 0, steering = 0
        for (int i = 0; i < 90; ++i)
        {
            car.update(coasting, kSimulationDt);
        }
        const float speedAfterLiftOff = car.getSpeed();
        assert(speedAfterLiftOff < speedBeforeLiftOff - 10.0f &&
               "releasing throttle must visibly reduce speed via rolling resistance/drag");
        TraceLog(LOG_INFO, "Vehicle physics: lift-off over 1.5s: %.1f -> %.1f px/s (-%.1f px/s)",
                 static_cast<double>(speedBeforeLiftOff), static_cast<double>(speedAfterLiftOff),
                 static_cast<double>(speedBeforeLiftOff - speedAfterLiftOff));
    }

    // 3: straight-line acceleration stays stable -- no NaNs, heading
    // frozen frame-to-frame and overall, under full throttle/zero steering.
    {
        car.reset(kSpawnPosition, kSpawnHeading);
        simulation::CarInput throttleOnly;
        throttleOnly.throttle = 1.0f;
        float previousHeading = car.getHeading();
        for (int i = 0; i < 150; ++i) // ~2.5s
        {
            car.update(throttleOnly, kSimulationDt);
            assert(allFinite(car) && "straight-line acceleration must never produce a NaN/Inf state");
            assert(std::fabs(headingDelta(previousHeading, car.getHeading())) < 0.01f &&
                   "heading must not snap frame to frame under zero steering");
            previousHeading = car.getHeading();
        }
        assert(car.isAlive() && "test setup must keep the car on the road");
        assert(std::fabs(headingDelta(kSpawnHeading, car.getHeading())) < 0.02f &&
               "zero steering must track straight overall, not just frame to frame");
        assert(std::fabs(car.getLateralVelocity()) < car.getSpeed() * 0.05f &&
               "zero steering must leave lateral velocity negligible relative to forward speed");
    }

    // 4: standing-start full throttle + full steering must curve the car
    // through a stable trajectory -- never instantly rotating the body out
    // from under the velocity vector or sliding near-perpendicular ("on ice").
    {
        car.reset(kSpawnPosition, kSpawnHeading);
        simulation::CarInput fullLock;
        fullLock.throttle = 1.0f;
        fullLock.steering = 1.0f;

        // 0.5s: this maneuver is an accelerating spiral that leaves the
        // 130px corridor around 0.7-0.8s regardless of physics -- 0.5s
        // stays comfortably inside that geometric margin.
        float maxAbsBodySlipAngle = 0.0f;
        for (int i = 0; i < 30 && car.isAlive(); ++i)
        {
            car.update(fullLock, kSimulationDt);
            assert(allFinite(car) && "standing-start full throttle+steering must never produce a NaN/Inf state");
            maxAbsBodySlipAngle = std::max(maxAbsBodySlipAngle, std::fabs(car.getSlipAngle()));
        }
        assert(car.isAlive() && "a standing-start full-lock turn on the spawn straight must not leave the road");
        assert(car.getSpeed() > 30.0f && "the car must still have accelerated meaningfully while turning");
        assert(std::fabs(headingDelta(kSpawnHeading, car.getHeading())) > 0.3f &&
               "the car must have genuinely turned, not just spun in place");
        // 60 deg -- short of a perpendicular slide, above rest-noise level.
        assert(maxAbsBodySlipAngle < 1.047f &&
               "standing-start full-lock steering must not make the car slide almost sideways");
        TraceLog(LOG_INFO,
                 "Vehicle physics: standing-start full throttle+steering over 0.5s -- final speed %.1fpx/s, heading "
                 "turned %.1fdeg, max body slip angle %.1fdeg",
                 static_cast<double>(car.getSpeed()),
                 static_cast<double>(headingDelta(kSpawnHeading, car.getHeading()) * RAD2DEG),
                 static_cast<double>(maxAbsBodySlipAngle * RAD2DEG));
    }

    // 5: moderate-speed cornering keeps slip angles small (planted, not "on
    // ice"), and axle forces + yaw rate are genuinely nonzero -- yaw comes
    // from the tire model's moment arms, not from nowhere.
    {
        car.reset(kSpawnPosition, kSpawnHeading);
        simulation::CarInput moderateTurn;
        moderateTurn.throttle = 1.0f;
        moderateTurn.steering = 0.5f;
        // 0.75s: stays inside the corridor (same geometric limit as test 4).
        for (int i = 0; i < 45 && car.isAlive(); ++i)
        {
            car.update(moderateTurn, kSimulationDt);
        }
        assert(car.isAlive() && "test setup must keep the car on the road");

        const simulation::TireDebugInfo& debug = car.getTireDebugInfo();
        // 20 deg -- rarely exceeded by real road cars even under enthusiastic cornering.
        assert(std::fabs(debug.frontSlipAngle) < 0.349f && "moderate-speed cornering must keep the front slip angle small");
        assert(std::fabs(debug.rearSlipAngle) < 0.349f && "moderate-speed cornering must keep the rear slip angle small");
        assert(std::fabs(car.getSlipAngle()) < 0.349f && "moderate-speed cornering must keep the whole-body slip angle small");

        assert(std::fabs(debug.frontForceY) > 10.0f && "cornering must produce a genuinely nonzero front tire force");
        assert(std::fabs(debug.yawRate) > 0.05f &&
               "a genuinely nonzero front tire force must be producing genuine rotation");
        TraceLog(LOG_INFO,
                 "Vehicle physics: moderate cornering -- speed %.1fpx/s, body slip %.1fdeg, front slip %.1fdeg "
                 "(Fy=%.0f), rear slip %.1fdeg (Fx=%.0f Fy=%.0f, grip=%.0f%%), yaw rate %.2frad/s",
                 static_cast<double>(car.getSpeed()), static_cast<double>(car.getSlipAngle() * RAD2DEG),
                 static_cast<double>(debug.frontSlipAngle * RAD2DEG), static_cast<double>(debug.frontForceY),
                 static_cast<double>(debug.rearSlipAngle * RAD2DEG), static_cast<double>(debug.rearForceX),
                 static_cast<double>(debug.rearForceY), static_cast<double>(debug.rearGripUtilization * 100.0f),
                 static_cast<double>(debug.yawRate));
    }

    // 6: releasing steering after a hard turn must let slip angle shrink,
    // not persist or grow.
    {
        car.reset(kSpawnPosition, kSpawnHeading);
        simulation::CarInput throttleOnly;
        throttleOnly.throttle = 1.0f;
        for (int i = 0; i < 30; ++i) // build up a moderate speed first (~0.5s)
        {
            car.update(throttleOnly, kSimulationDt);
        }
        simulation::CarInput hardTurn;
        hardTurn.throttle = 1.0f;
        hardTurn.steering = 1.0f;
        for (int i = 0; i < 15 && car.isAlive(); ++i) // induce real slip
        {
            car.update(hardTurn, kSimulationDt);
        }
        assert(car.isAlive() && "test setup must keep the car on the road");
        const float slipAfterTurn = std::fabs(car.getSlipAngle());
        assert(slipAfterTurn > 0.03f && "test setup must actually induce measurable slip before testing recovery");

        // Coast while straightening -- less distance per frame gives more
        // room to observe settling before the track boundary matters.
        simulation::CarInput straighten;
        straighten.throttle = 0.0f;
        straighten.steering = 0.0f;
        float previousSlip = slipAfterTurn;
        float worstGrowth = 0.0f; // largest single-step increase seen, to catch oscillation/growth
        for (int i = 0; i < 20 && car.isAlive(); ++i)
        {
            car.update(straighten, kSimulationDt);
            const float slip = std::fabs(car.getSlipAngle());
            worstGrowth = std::max(worstGrowth, slip - previousSlip);
            previousSlip = slip;
        }
        assert(car.isAlive() && "test setup must keep the car on the road while it settles");
        assert(worstGrowth < 0.02f && "slip angle must not oscillate or grow once steering is released");
        assert(previousSlip < slipAfterTurn * 0.5f &&
               "releasing steering must let slip angle settle back down substantially");
    }

    // 7: turn radius genuinely widens with speed, and grip loss under a
    // hard turn is progressive (front/rear slip angles growing smoothly
    // from low to high speed) rather than an instant jump to an extreme.
    // Turn radius is measured as speed / yaw-rate (R = v / omega) over a
    // short window once the tire forces have settled into the turn.
    {
        simulation::CarInput fullLeftLock;
        fullLeftLock.throttle = 1.0f;
        fullLeftLock.steering = -1.0f;

        // Reaches a speed tier via a throttleBurstFrames-long full-throttle
        // burst from spawn (steering neutral), then holds full lock for
        // kSettleFrames so the tire forces/yaw rate settle into the turn
        // (this is a real physical settling time now -- there is no
        // separate "controller" converging toward a commanded target),
        // then measures yaw rate over a further short window. Returns
        // {speed, radius, frontSlipAngle}.
        auto measureTurn = [&](int throttleBurstFrames) -> std::tuple<float, float, float>
        {
            // Deliberately short: at the highest speed tier, a full-lock
            // turn only has roughly 20-25 frames of room before leaving the
            // spawn straight's 130px corridor (by inspection -- same
            // geometric constraint as tests 4-6 above), so
            // kSettleFrames+kMeasureFrames must stay comfortably under
            // that. The tire model's own response is fast (a handful of
            // frames -- see kForceSubsteps in Car.cpp), so this is still
            // enough for yaw rate to settle into a representative value
            // for the turn, not just the initial transient.
            constexpr int kSettleFrames = 10;
            constexpr int kMeasureFrames = 6;
            constexpr float kMeasureWindowSeconds = static_cast<float>(kMeasureFrames) * kSimulationDt;

            car.reset(kSpawnPosition, kSpawnHeading);
            simulation::CarInput throttleOnly;
            throttleOnly.throttle = 1.0f;
            for (int i = 0; i < throttleBurstFrames; ++i)
            {
                car.update(throttleOnly, kSimulationDt);
            }

            for (int i = 0; i < kSettleFrames && car.isAlive(); ++i)
            {
                car.update(fullLeftLock, kSimulationDt);
            }
            assert(car.isAlive() && "turn-radius measurement must not leave the road while settling into the turn");
            const float speed = car.getSpeed();
            const float headingBefore = car.getHeading();

            for (int i = 0; i < kMeasureFrames && car.isAlive(); ++i)
            {
                car.update(fullLeftLock, kSimulationDt);
            }
            assert(car.isAlive() && "turn-radius measurement must not leave the road during the measurement window");
            const float yawRate = std::fabs(headingDelta(headingBefore, car.getHeading())) / kMeasureWindowSeconds;
            const float radius = speed / std::max(yawRate, 1e-4f);
            return {speed, radius, std::fabs(car.getTireDebugInfo().frontSlipAngle)};
        };

        const auto [lowSpeed, lowRadius, lowSlip] = measureTurn(15); // short burst -> a low but non-trivial speed
        const auto [mediumSpeed, mediumRadius, mediumSlip] = measureTurn(60); // ~1s
        const auto [highSpeed, highRadius, highSlip] = measureTurn(180); // ~3s, well up toward maxSpeed

        assert(lowSpeed > 5.0f && "test setup must reach a low, non-trivial speed");
        assert(mediumSpeed > lowSpeed * 1.3f && "test setup must reach a meaningfully higher medium speed");
        // medium (~1s of throttle) is already well up the resistance
        // curve's asymptote toward the analytic top speed (see
        // CarParams::rollingResistance's comment), so high (~3s) is closer
        // to medium than low is to medium -- 1.2x is still a clearly
        // higher speed, just not as dramatic a jump as low -> medium.
        assert(highSpeed > mediumSpeed * 1.2f && "test setup must reach a meaningfully higher high speed");

        assert(mediumRadius >= lowRadius * 0.9f && "turn radius must not shrink as speed increases");
        assert(highRadius > mediumRadius && "turn radius must keep widening from medium to high speed");
        assert(highRadius > lowRadius * 1.3f &&
               "the same full steering lock must produce a meaningfully wider turn radius at high speed than at low "
               "speed (understeer)");

        // Progressive grip loss: front slip angle should grow monotonically
        // (not jump) from low to medium to high speed, and even at high
        // speed must stay well short of the wheel being sideways to its
        // own velocity (90 degrees).
        assert(mediumSlip >= lowSlip - 0.01f && "front slip angle must not shrink as speed increases");
        assert(highSlip > mediumSlip && "front slip angle must keep growing from medium to high speed (progressive grip loss)");
        assert(highSlip < 1.396f && "even a hard high-speed turn must stay well short of a 90-degree wheel slip");

        TraceLog(LOG_INFO,
                 "Vehicle physics: full-lock turn -- low %.1fpx/s -> R=%.1fpx (frontSlip=%.1fdeg), medium %.1fpx/s -> "
                 "R=%.1fpx (frontSlip=%.1fdeg), high %.1fpx/s -> R=%.1fpx (frontSlip=%.1fdeg)",
                 static_cast<double>(lowSpeed), static_cast<double>(lowRadius), static_cast<double>(lowSlip * RAD2DEG),
                 static_cast<double>(mediumSpeed), static_cast<double>(mediumRadius),
                 static_cast<double>(mediumSlip * RAD2DEG), static_cast<double>(highSpeed), static_cast<double>(highRadius),
                 static_cast<double>(highSlip * RAD2DEG));
    }

    // 8: robustness -- a sustained, aggressive full-throttle/full-steering
    // stress run must never produce a NaN/Inf, and yaw rate must stay
    // within a generously sane bound throughout (catching any runaway
    // spin), even though the car is expected to eventually leave the road
    // (this is a deliberately extreme, sustained input).
    {
        car.reset(kSpawnPosition, kSpawnHeading);
        simulation::CarInput stress;
        stress.throttle = 1.0f;
        stress.steering = 1.0f;
        float maxAbsYawRate = 0.0f;
        for (int i = 0; i < 300 && car.isAlive(); ++i) // up to ~5s
        {
            car.update(stress, kSimulationDt);
            assert(allFinite(car) && "sustained aggressive input must never produce a NaN/Inf state");
            maxAbsYawRate = std::max(maxAbsYawRate, std::fabs(car.getTireDebugInfo().yawRate));
        }
        // 15 rad/s is far beyond anything this model's forces can
        // physically justify (its own kinematic-turn numbers above never
        // exceed a couple of rad/s) -- this bound exists purely to catch a
        // genuine runaway/instability, not to constrain normal behavior.
        assert(maxAbsYawRate < 15.0f && "yaw rate must never run away/spin uncontrollably under any input");
    }

    // 9: sensor transforms follow the Box2D-derived position/heading during
    // real motion, not just immediately after reset() (verifySensors()
    // already covers the reset() case). Drive forward while turning for a
    // while so both position and heading move away from spawn, then check
    // the sensor origin and every ray's direction against the car's own
    // reported pose.
    {
        car.reset(kSpawnPosition, kSpawnHeading);
        simulation::CarInput driveAndTurn;
        driveAndTurn.throttle = 1.0f;
        driveAndTurn.steering = 0.15f; // gentle: this only needs to move+rotate the car a little, not stress-test cornering
        for (int i = 0; i < 15 && car.isAlive(); ++i)
        {
            car.update(driveAndTurn, kSimulationDt);
        }
        assert(car.isAlive() && "test setup must keep the car on the road for the sensor-transform check");
        assert(std::fabs(headingDelta(kSpawnHeading, car.getHeading())) > 0.01f &&
               "test setup must actually have turned the car away from its spawn heading");

        const Vector2 forward = {std::cos(car.getHeading()), std::sin(car.getHeading())};
        const Vector2 expectedOrigin = {car.getPosition().x + forward.x * car.getParams().length * 0.5f,
                                         car.getPosition().y + forward.y * car.getParams().length * 0.5f};
        const Vector2 origin = car.getSensorOrigin();
        assert(std::fabs(origin.x - expectedOrigin.x) < 0.01f && std::fabs(origin.y - expectedOrigin.y) < 0.01f &&
               "sensor origin must track the car's current Box2D-derived position/heading, not spawn");

        const auto& sensors = car.getSensors();
        for (int i = 0; i < simulation::Car::kSensorCount; ++i)
        {
            const float angle = car.getHeading() + simulation::Car::kSensorAngleDegrees[i] * DEG2RAD;
            const Vector2 direction = {std::cos(angle), std::sin(angle)};
            const Vector2 expectedEnd = {origin.x + direction.x * sensors[i].distance,
                                          origin.y + direction.y * sensors[i].distance};
            assert(std::fabs(expectedEnd.x - sensors[i].endPoint.x) < 0.01f &&
                   std::fabs(expectedEnd.y - sensors[i].endPoint.y) < 0.01f &&
                   "sensor ray directions must track the car's current heading during real motion");
        }
    }

    // 10: reset() restores a clean, fully deterministic state regardless of
    // how chaotic the preceding motion was -- exact equality is correct
    // here (not just within an epsilon): reset() assigns the mirrored
    // position/heading/velocity directly from its spawnPosition/
    // spawnHeading parameters rather than reading them back from Box2D
    // (see Car::reset()), so no floating-point round-trip is involved.
    {
        simulation::CarInput chaos;
        chaos.throttle = 1.0f;
        chaos.steering = 1.0f;
        for (int i = 0; i < 40 && car.isAlive(); ++i)
        {
            car.update(chaos, kSimulationDt);
        }

        car.reset(kSpawnPosition, kSpawnHeading);
        assert(car.isAlive() && "reset must revive the car regardless of prior state");
        assert(car.getPosition().x == kSpawnPosition.x && car.getPosition().y == kSpawnPosition.y &&
               "reset must restore the exact spawn position after chaotic prior motion");
        assert(car.getHeading() == kSpawnHeading && "reset must restore the exact spawn heading after chaotic prior motion");
        assert(car.getVelocity().x == 0.0f && car.getVelocity().y == 0.0f &&
               "reset must zero velocity after chaotic prior motion");
        assert(car.getForwardVelocity() == 0.0f && car.getLateralVelocity() == 0.0f && car.getSlipAngle() == 0.0f &&
               "reset must leave all derived vehicle-state values at their zero/rest values");

        // Determinism: resetting from two DIFFERENT chaotic prior states
        // must land on identical sensor readings -- nothing about the
        // pre-reset state leaks through.
        const auto sensorsAfterFirstReset = car.getSensors();

        simulation::CarInput differentChaos;
        differentChaos.throttle = 1.0f;
        differentChaos.steering = -1.0f;
        for (int i = 0; i < 55 && car.isAlive(); ++i)
        {
            car.update(differentChaos, kSimulationDt);
        }
        car.reset(kSpawnPosition, kSpawnHeading);
        const auto sensorsAfterSecondReset = car.getSensors();

        for (int i = 0; i < simulation::Car::kSensorCount; ++i)
        {
            assert(sensorsAfterFirstReset[i].distance == sensorsAfterSecondReset[i].distance &&
                   "reset must produce deterministic sensor readings regardless of prior motion");
        }
    }

    // ---- RWD + per-axle friction circle ----
    // Exercises throttle and rear cornering force sharing one grip budget
    // at the rear axle (front stays lateral-only).

    // 11 & 12 & 20: neither axle's combined tire force ever exceeds its
    // configured grip budget, and nothing here produces a NaN/Inf state or
    // an unbounded spin, under a sustained, varying, aggressive input
    // sequence (alternating hard throttle and hard steering in both
    // directions -- deliberately harder to keep these invariants holding
    // than any single fixed input would be).
    {
        car.reset(kSpawnPosition, kSpawnHeading);
        float maxAbsYawRate = 0.0f;
        for (int i = 0; i < 200 && car.isAlive(); ++i)
        {
            simulation::CarInput stress;
            stress.throttle = 1.0f;
            stress.steering = ((i / 15) % 2 == 0) ? 1.0f : -1.0f; // flips every 15 frames (0.25s)
            car.update(stress, kSimulationDt);
            assert(allFinite(car) && "aggressive alternating input must never produce a NaN/Inf state");

            const simulation::TireDebugInfo& debug = car.getTireDebugInfo();
            constexpr float kCircleTolerance = 1.02f; // 2% slack for floating-point/tanh-saturation rounding
            const float frontMag = std::fabs(debug.frontForceY); // frontForceX is always 0 (RWD)
            assert(frontMag <= car.getParams().frontMaxTireForce * kCircleTolerance &&
                   "front axle combined tire force must never exceed its grip limit");
            const float rearMag = std::sqrt(debug.rearForceX * debug.rearForceX + debug.rearForceY * debug.rearForceY);
            assert(rearMag <= car.getParams().rearMaxTireForce * kCircleTolerance &&
                   "rear axle combined tire force must never exceed its grip limit (friction circle)");

            maxAbsYawRate = std::max(maxAbsYawRate, std::fabs(debug.yawRate));
        }
        // Same generous bound as the earlier robustness check -- exists
        // purely to catch a genuine runaway, not to constrain normal
        // behavior.
        assert(maxAbsYawRate < 15.0f && "yaw rate must never run away/spin uncontrollably under any input");
    }

    // 13: straight full-throttle acceleration remains stable, AND does not
    // consume nearly all rear grip -- rearMaxTireForce is sized so a
    // straight line leaves real cornering budget available (see
    // CarParams::rearMaxTireForce's comment).
    {
        car.reset(kSpawnPosition, kSpawnHeading);
        simulation::CarInput straight;
        straight.throttle = 1.0f;
        float maxRearGripStraight = 0.0f;
        for (int i = 0; i < 60; ++i) // ~1s
        {
            car.update(straight, kSimulationDt);
            assert(allFinite(car) && "straight full-throttle acceleration must never produce a NaN/Inf state");
            maxRearGripStraight = std::max(maxRearGripStraight, car.getTireDebugInfo().rearGripUtilization);
        }
        assert(car.isAlive() && "straight full-throttle acceleration must keep the car on the road");
        assert(maxRearGripStraight < 0.85f && "full throttle in a straight line must not consume nearly all rear grip");
        TraceLog(LOG_INFO, "Vehicle physics: straight full-throttle rear grip utilization peaked at %.0f%%",
                 static_cast<double>(maxRearGripStraight * 100.0f));
    }

    // 14: moderate cornering at PARTIAL throttle also stays planted (partial-throttle version of check 5).
    {
        car.reset(kSpawnPosition, kSpawnHeading);
        simulation::CarInput partialThrottleTurn;
        partialThrottleTurn.throttle = 0.5f;
        partialThrottleTurn.steering = 0.5f;
        for (int i = 0; i < 45 && car.isAlive(); ++i)
        {
            car.update(partialThrottleTurn, kSimulationDt);
        }
        assert(car.isAlive() && "test setup must keep the car on the road");
        assert(std::fabs(car.getSlipAngle()) < 0.349f &&
               "moderate cornering at partial throttle must keep the whole-body slip angle small");
    }

    // 15, 16 & 17: full throttle mid-corner must reduce available rear
    // lateral force and increase rear slip vs. the SAME corner at zero
    // throttle, and a sufficiently aggressive combination must saturate the
    // rear friction circle and produce genuine throttle-induced oversteer
    // (rear slip exceeding front slip).
    //
    // Two independent multi-frame runs would confound the comparison (full
    // throttle's trajectory diverges for reasons unrelated to the friction
    // circle). Instead both throttle values are applied from an IDENTICAL
    // checkpoint state (two deterministic replays of the same input
    // sequence -- no RNG, so bit-identical both times), each for one final
    // update() -- so any difference is attributable to the friction circle alone.
    {
        auto driveToCheckpoint = [&]()
        {
            car.reset(kSpawnPosition, kSpawnHeading);
            simulation::CarInput burst;
            burst.throttle = 1.0f;
            for (int i = 0; i < 150; ++i) // ~2.5s, well up in speed
            {
                car.update(burst, kSimulationDt);
            }

            // Moderate throttle approaching the corner, so both forks share
            // identical slip history right up to the checkpoint.
            simulation::CarInput approach;
            approach.throttle = 0.3f;
            approach.steering = 0.6f;
            for (int i = 0; i < 20 && car.isAlive(); ++i)
            {
                car.update(approach, kSimulationDt);
            }
        };

        driveToCheckpoint();
        assert(car.isAlive() && "checkpoint approach must keep the car on the road");
        simulation::CarInput zeroFinal;
        zeroFinal.throttle = 0.0f;
        zeroFinal.steering = 0.6f;
        car.update(zeroFinal, kSimulationDt);
        const simulation::TireDebugInfo zeroThrottle = car.getTireDebugInfo();
        assert(car.isAlive() && "the zero-throttle comparison frame must keep the car on the road");

        driveToCheckpoint(); // deterministic replay -> bit-identical checkpoint state
        simulation::CarInput fullFinal;
        fullFinal.throttle = 1.0f;
        fullFinal.steering = 0.6f;
        car.update(fullFinal, kSimulationDt);
        const simulation::TireDebugInfo fullThrottle = car.getTireDebugInfo();
        assert(car.isAlive() && "the full-throttle comparison frame must keep the car on the road");

        // 15: full throttle must leave less rear LATERAL force available
        // than zero throttle does, for the identical corner state.
        assert(std::fabs(fullThrottle.rearForceY) < std::fabs(zeroThrottle.rearForceY) &&
               "full throttle mid-corner must reduce the rear axle's available lateral force versus zero throttle");

        // 16 & 17: continue each fork at its own throttle for more frames --
        // oversteer is a compounding effect, not necessarily visible in one frame.
        constexpr int kFollowFrames = 14;
        for (int i = 0; i < kFollowFrames && car.isAlive(); ++i)
        {
            car.update(zeroFinal, kSimulationDt); // continues from the zero-throttle fork, still at zero throttle
        }
        const float zeroSlipAfter = std::fabs(car.getTireDebugInfo().rearSlipAngle);

        driveToCheckpoint();
        car.update(fullFinal, kSimulationDt);
        for (int i = 0; i < kFollowFrames && car.isAlive(); ++i)
        {
            car.update(fullFinal, kSimulationDt); // continues from the full-throttle fork, still at full throttle
        }
        const simulation::TireDebugInfo fullFollowed = car.getTireDebugInfo();
        const float fullSlipAfter = std::fabs(fullFollowed.rearSlipAngle);

        // 16: full throttle must grow rear slip angle faster than zero
        // throttle, from the identical checkpoint state.
        assert(fullSlipAfter > zeroSlipAfter &&
               "full throttle mid-corner must grow rear slip angle faster than zero throttle from the same state");

        // 17: throttle-induced oversteer -- rear circle saturated (~100%
        // grip) and rear sliding more than front (this same car understeers
        // -- checks 5/7 -- once throttle is out of the picture).
        assert(fullFollowed.rearGripUtilization > 0.9f &&
               "sustained full throttle through a hard corner must saturate the rear friction circle");
        assert(fullSlipAfter > std::fabs(fullFollowed.frontSlipAngle) &&
               "throttle-induced oversteer must show the rear sliding more than the front");

        TraceLog(LOG_INFO,
                 "Vehicle physics: throttle-vs-corner (same checkpoint) -- zero throttle rearFy=%.0f, full throttle "
                 "rearFx=%.0f rearFy=%.0f, after %d more frames: rearSlip=%.1fdeg frontSlip=%.1fdeg rearGrip=%.0f%%",
                 static_cast<double>(zeroThrottle.rearForceY), static_cast<double>(fullThrottle.rearForceX),
                 static_cast<double>(fullThrottle.rearForceY), kFollowFrames, static_cast<double>(fullSlipAfter * RAD2DEG),
                 static_cast<double>(fullFollowed.frontSlipAngle * RAD2DEG),
                 static_cast<double>(fullFollowed.rearGripUtilization * 100.0f));
    }

    // 18: releasing throttle during a throttle-induced slide lets the rear
    // regain lateral grip progressively -- rear slip angle must move back
    // toward normal, not stay pinned or grow further.
    {
        car.reset(kSpawnPosition, kSpawnHeading);
        simulation::CarInput burst;
        burst.throttle = 1.0f;
        for (int i = 0; i < 90; ++i) // build speed
        {
            car.update(burst, kSimulationDt);
        }
        simulation::CarInput hardCornerFullThrottle;
        hardCornerFullThrottle.throttle = 1.0f;
        hardCornerFullThrottle.steering = 1.0f;
        for (int i = 0; i < 15 && car.isAlive(); ++i) // induce a throttle-saturated slide
        {
            car.update(hardCornerFullThrottle, kSimulationDt);
        }
        assert(car.isAlive() && "test setup must keep the car on the road");
        const float rearSlipDuringSlide = std::fabs(car.getTireDebugInfo().rearSlipAngle);
        assert(rearSlipDuringSlide > 0.03f && "test setup must actually induce measurable rear slip before testing recovery");

        simulation::CarInput coastStraighten; // throttle = 0, steering = 0
        float previousSlip = rearSlipDuringSlide;
        float worstGrowth = 0.0f;
        for (int i = 0; i < 12 && car.isAlive(); ++i)
        {
            car.update(coastStraighten, kSimulationDt);
            const float slip = std::fabs(car.getTireDebugInfo().rearSlipAngle);
            worstGrowth = std::max(worstGrowth, slip - previousSlip);
            previousSlip = slip;
        }
        assert(car.isAlive() && "test setup must keep the car on the road while it recovers");
        assert(worstGrowth < 0.02f && "rear slip angle must not oscillate or grow once throttle is released");
        assert(previousSlip < rearSlipDuringSlide * 0.5f &&
               "releasing throttle during a slide must let rear slip angle recover substantially");
    }

    // 19: excessive high-speed corner entry produces understeer (front grip
    // utilization climbs toward saturation) -- an explicit grip-utilization
    // check on top of check 7's slip-angle evidence.
    {
        car.reset(kSpawnPosition, kSpawnHeading);
        simulation::CarInput burst;
        burst.throttle = 1.0f;
        for (int i = 0; i < 180; ++i) // ~3s, well up toward maxSpeed
        {
            car.update(burst, kSimulationDt);
        }
        simulation::CarInput fullLeftLock;
        fullLeftLock.steering = -1.0f; // throttle = 0: isolate front-axle behavior from the rear circle
        for (int i = 0; i < 20 && car.isAlive(); ++i)
        {
            car.update(fullLeftLock, kSimulationDt);
        }
        assert(car.isAlive() && "test setup must keep the car on the road");
        assert(car.getTireDebugInfo().frontGripUtilization > 0.5f &&
               "excessive high-speed corner entry must load the front axle's grip meaningfully toward its limit");
    }

    car.reset(kSpawnPosition, kSpawnHeading);

    TraceLog(LOG_INFO, "Vehicle physics verification: all deterministic checks passed");
}

// Deterministic check of the full Car -> Observation -> NeuralNetwork ->
// AIController -> CarInput loop, using small hand-built genomes (not
// createDemonstrationGenome(), so this stays independent of its weights).
void verifyAIController(const simulation::Track& track)
{
    using ai::AIController;
    using ai::neat::ConnectionGene;
    using ai::neat::Genome;
    using ai::neat::NodeGene;
    using ai::neat::NodeType;
    constexpr float kEps = 1e-4f;

    // 9 Input + 1 Bias + 2 Output, no connections -- every output is
    // deterministically 0 unless a test adds its own connections.
    auto makeDisconnectedGenome = []()
    {
        Genome genome;
        for (int i = 0; i < ai::NeuralNetwork::kInputCount; ++i)
        {
            genome.addNode(NodeGene{i, NodeType::Input});
        }
        genome.addNode(NodeGene{9, NodeType::Bias});
        genome.addNode(NodeGene{100, NodeType::Output});
        genome.addNode(NodeGene{101, NodeType::Output});
        return genome;
    };

    // 1 & 12: the controller stores and uses a valid two-output network --
    // construction and one update() succeed without throwing.
    {
        AIController controller(ai::neat::buildPhenotype(makeDisconnectedGenome()));
        simulation::Car car(makeCarParams(), track);
        car.reset(kSpawnPosition, kSpawnHeading);
        const simulation::CarInput input = controller.update(car);
        assert(input.steering == 0.0f && input.throttle == 0.5f &&
               "a disconnected network must map to zero steering and neutral (0.5) throttle");
    }

    // 2 & 3: output 0 drives steering, output 1 drives throttle. Bias
    // (always 1.0) connects only to steering, so a positive weight there
    // must move steering but leave throttle at its neutral 0.5.
    {
        Genome genome = makeDisconnectedGenome();
        genome.addConnection(ConnectionGene{9, 100, 1.0f, true, 0});
        AIController controller(ai::neat::buildPhenotype(genome));

        simulation::Car car(makeCarParams(), track);
        car.reset(kSpawnPosition, kSpawnHeading);
        const simulation::CarInput input = controller.update(car);

        assert(input.steering > 0.5f && "output index 0 must map to steering");
        assert(std::fabs(input.throttle - 0.5f) < kEps && "output index 1 (throttle) must be unaffected");
    }

    // 4, 5 & 6: raw throttle 0 maps to 0.5; negative maps below 0.5; positive maps above 0.5.
    {
        Genome zeroGenome = makeDisconnectedGenome(); // no Bias->throttle connection: raw throttle stays 0
        AIController zeroController(ai::neat::buildPhenotype(zeroGenome));
        simulation::Car car(makeCarParams(), track);
        car.reset(kSpawnPosition, kSpawnHeading);
        const simulation::CarInput zeroInput = zeroController.update(car);
        assert(zeroController.getRawThrottleOutput() == 0.0f && "raw throttle must be exactly 0 with no contribution");
        assert(std::fabs(zeroInput.throttle - 0.5f) < kEps && "raw throttle 0 must map to mapped throttle 0.5");

        Genome negativeGenome = makeDisconnectedGenome();
        negativeGenome.addConnection(ConnectionGene{9, 101, -1.0f, true, 0});
        AIController negativeController(ai::neat::buildPhenotype(negativeGenome));
        car.reset(kSpawnPosition, kSpawnHeading);
        const simulation::CarInput negativeInput = negativeController.update(car);
        assert(negativeController.getRawThrottleOutput() < 0.0f && "negative Bias->throttle weight must yield negative raw throttle");
        assert(negativeInput.throttle < 0.5f - kEps && "negative raw throttle must map below 0.5");

        Genome positiveGenome = makeDisconnectedGenome();
        positiveGenome.addConnection(ConnectionGene{9, 101, 1.0f, true, 0});
        AIController positiveController(ai::neat::buildPhenotype(positiveGenome));
        car.reset(kSpawnPosition, kSpawnHeading);
        const simulation::CarInput positiveInput = positiveController.update(car);
        assert(positiveController.getRawThrottleOutput() > 0.0f && "positive Bias->throttle weight must yield positive raw throttle");
        assert(positiveInput.throttle > 0.5f + kEps && "positive raw throttle must map above 0.5");
    }

    // 7 & 8: mapped steering stays within [-1,1], throttle within [0,1],
    // even under saturating weights and an actively driving car.
    {
        Genome genome = makeDisconnectedGenome();
        genome.addConnection(ConnectionGene{0, 100, 10.0f, true, 0});
        genome.addConnection(ConnectionGene{9, 101, 10.0f, true, 1});
        AIController controller(ai::neat::buildPhenotype(genome));

        simulation::Car car(makeCarParams(), track);
        car.reset(kSpawnPosition, kSpawnHeading);
        simulation::CarInput driveInput;
        driveInput.throttle = 1.0f;
        driveInput.steering = 1.0f;
        for (int i = 0; i < 30 && car.isAlive(); ++i)
        {
            car.update(driveInput, kSimulationDt);
            const simulation::CarInput aiInput = controller.update(car);
            assert(aiInput.steering >= -1.0f && aiInput.steering <= 1.0f && "mapped steering must stay within [-1, 1]");
            assert(aiInput.throttle >= 0.0f && aiInput.throttle <= 1.0f && "mapped throttle must stay within [0, 1]");
        }
        car.reset(kSpawnPosition, kSpawnHeading);
    }

    // 9: the Observation is actually built from the provided Car -- a
    // network wired from sensor 2 (center) to steering must react to it.
    {
        Genome genome = makeDisconnectedGenome();
        genome.addConnection(ConnectionGene{2, 100, 1.0f, true, 0});
        AIController controller(ai::neat::buildPhenotype(genome));

        simulation::Car car(makeCarParams(), track);
        car.reset(kSpawnPosition, kSpawnHeading);
        controller.update(car);

        const float expectedCenterSensor = car.getSensors()[2].normalizedDistance;
        assert(std::fabs(controller.getLastObservation().values[2] - expectedCenterSensor) < kEps &&
               "AIController's Observation must be built from the provided Car's own sensor readings");
    }

    // 10: repeated calls with unchanged Car state are deterministic.
    {
        Genome genome = makeDisconnectedGenome();
        genome.addConnection(ConnectionGene{0, 100, 0.7f, true, 0});
        genome.addConnection(ConnectionGene{9, 101, -0.3f, true, 1});
        AIController controller(ai::neat::buildPhenotype(genome));

        simulation::Car car(makeCarParams(), track);
        car.reset(kSpawnPosition, kSpawnHeading);

        const simulation::CarInput first = controller.update(car);
        const simulation::CarInput second = controller.update(car);
        assert(first.steering == second.steering && first.throttle == second.throttle &&
               "repeated updates against an unchanged Car must produce identical CarInput");
    }

    // 11: the controller does not mutate the Car it reads from.
    {
        Genome genome = makeDisconnectedGenome();
        genome.addConnection(ConnectionGene{0, 100, 0.5f, true, 0});
        AIController controller(ai::neat::buildPhenotype(genome));

        simulation::Car car(makeCarParams(), track);
        car.reset(kSpawnPosition, kSpawnHeading);
        const Vector2 positionBefore = car.getPosition();
        const Vector2 velocityBefore = car.getVelocity();
        const float headingBefore = car.getHeading();

        controller.update(car);

        assert(car.getPosition().x == positionBefore.x && car.getPosition().y == positionBefore.y &&
               "AIController::update must not move the Car");
        assert(car.getVelocity().x == velocityBefore.x && car.getVelocity().y == velocityBefore.y &&
               "AIController::update must not change the Car's velocity");
        assert(car.getHeading() == headingBefore && "AIController::update must not change the Car's heading");
    }

    // 13: a disabled connection stays inactive after phenotype construction.
    {
        Genome genome = makeDisconnectedGenome();
        genome.addConnection(ConnectionGene{0, 100, 1.0f, false, 0});
        AIController controller(ai::neat::buildPhenotype(genome));

        simulation::Car car(makeCarParams(), track);
        car.reset(kSpawnPosition, kSpawnHeading);
        const simulation::CarInput input = controller.update(car);
        assert(input.steering == 0.0f && "a disabled connection must not affect evaluation after phenotype construction");
    }

    // Dead car: the controller must not keep evaluating the network, and
    // must apply neutral CarInput instead.
    {
        Genome genome = makeDisconnectedGenome();
        genome.addConnection(ConnectionGene{9, 101, 1.0f, true, 0}); // would otherwise raise throttle above 0.5
        AIController controller(ai::neat::buildPhenotype(genome));

        simulation::Car car(makeCarParams(), track);
        car.reset(kSpawnPosition, kSpawnHeading);
        simulation::CarInput driveOffTrack;
        driveOffTrack.throttle = 1.0f;
        driveOffTrack.steering = 0.0f;
        for (int i = 0; i < 300 && car.isAlive(); ++i)
        {
            car.update(driveOffTrack, kSimulationDt);
        }
        assert(!car.isAlive() && "driving straight for 5s must leave the road band and kill the car");

        const simulation::CarInput deadInput = controller.update(car);
        assert(deadInput.steering == 0.0f && deadInput.throttle == 0.0f &&
               "a dead car must receive neutral CarInput from AIController, not a network-derived one");
        car.reset(kSpawnPosition, kSpawnHeading);
    }

    TraceLog(LOG_INFO, "AI controller verification: all deterministic checks passed");
}
} // namespace verification
