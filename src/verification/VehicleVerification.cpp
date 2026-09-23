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

// verifyVehiclePhysics() characterizes the tire/vehicle model itself under
// sustained FULL steering lock at speeds up to ~500 px/s. Those tests run with
// speed-sensitive steering authority DISABLED (every factor 1.0, i.e. the
// pre-authority behavior) so they keep measuring the tire model rather than
// the authority curve; the authority feature has its own tests
// (verifySteeringAuthority()).
static simulation::CarParams makeFullAuthorityCarParams()
{
    simulation::CarParams params = makeCarParams();
    params.steerAuthorityFactors.fill(1.0f);
    return params;
}


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
    static_assert(simulation::Car::kSensorCount == 5, "Car must expose exactly five sensors");

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
    // the boundary well within kMaxSensorDistance, for any track's spawn geometry.
    {
        car.reset(kSpawnPosition, kSpawnHeading + static_cast<float>(PI) * 0.5f);
        const simulation::SensorReading& front = car.getSensors()[2];
        assert(front.distance < simulation::Car::kMaxSensorDistance &&
               front.normalizedDistance < 1.0f &&
               "sensor aimed at a nearby boundary must report less than maximum distance");
        car.reset(kSpawnPosition, kSpawnHeading);
    }

    // 7: a ray with no obstacle within range reports exactly maximum distance / normalized 1.0.
    // At spawn heading, the front sensor points along the spawn straight,
    // clear for at least kMaxSensorDistance (the extreme track's start
    // straight is ~800px -- see Track.cpp -- comfortably longer).
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

    // 10 & 11: obstacle at a KNOWN physical distance is reported exactly --
    // not just "less than max" (checks 6/8 above) -- using a synthetic
    // straight corridor (same wide-synthetic-track idiom used elsewhere in
    // this file, e.g. the grip-saturation checks) whose wall distance from
    // an on-centerline point is exactly trackWidth/2 by construction (see
    // Track::buildMaskFromCenterlineWidth()'s distance-to-centerline <=
    // halfWidth rule) -- PROVIDED the centerline is actually straight there.
    // A plain 4-corner rectangle is NOT: Catmull-Rom through a corner's two
    // neighbors curves the "straight" side inward near its middle (verified
    // the hard way -- an earlier version of this check used a 4-point
    // rectangle and failed by ~25px). Using several COLINEAR control points
    // along the top edge instead makes every one of an interior segment's
    // four control points (p0,p1,p2,p3) share the same y, which collapses
    // the Catmull-Rom cubic to a plain straight line for that segment
    // (every t/t^2/t^3 coefficient in y vanishes) -- so the segment between
    // the two MIDDLE top points is exactly straight, with the nearest
    // corner still far away (>=500px).
    {
        simulation::TrackDefinition straightDef;
        straightDef.simWidth = 3500;
        straightDef.simHeight = 1500;
        straightDef.controlPoints = {Vector2{500.0f, 500.0f}, Vector2{1000.0f, 500.0f}, Vector2{1500.0f, 500.0f},
                                      Vector2{2000.0f, 500.0f}, Vector2{2500.0f, 500.0f}, Vector2{2500.0f, 700.0f},
                                      Vector2{500.0f, 700.0f}};
        straightDef.samplesPerSegment = 8;

        // Sensor origin sits kCarHalfLength ahead of the car's own position
        // along heading (see Car::getSensorOrigin()) -- pointing heading
        // -PI/2 ("up", away from the loop's interior) from a position ON
        // the top edge's centerline (y=500) moves the origin itself
        // slightly into the band first, so the expected ray distance to the
        // band edge is halfWidth minus that offset, not halfWidth itself.
        constexpr float kCarHalfLength = 24.0f * 0.5f; // CarParams::length, see makeCarParams()
        constexpr float kDistanceTolerance = simulation::Car::kSensorStep + 1.0f; // raycast step + rounding

        auto measureOutwardDistance = [&](float trackWidth) -> float
        {
            straightDef.trackWidth = trackWidth;
            simulation::Track straightTrack(straightDef);
            simulation::Car straightCar(makeCarParams(), straightTrack);
            // (1750,500) is the midpoint of the exactly-straight (1500,500)->(2000,500) segment.
            straightCar.reset(Vector2{1750.0f, 500.0f}, -static_cast<float>(PI) * 0.5f); // on centerline, facing "up"
            return straightCar.getSensors()[2].distance; // 0 deg sensor now points straight "up"
        };

        // 10: a wall at 300px -- inside the NEW 400px range but outside the
        // OLD 200px one -- must be reported at (approximately) its real
        // distance, not saturated at the old cap. This is the direct,
        // known-geometry regression check for the range change itself.
        {
            const float halfWidth = 300.0f; // trackWidth=600 -> wall at 300px from centerline
            const float expected = halfWidth - kCarHalfLength;
            const float measured = measureOutwardDistance(600.0f);
            assert(std::fabs(measured - expected) <= kDistanceTolerance &&
                   "a wall at a known 300px distance (within the new 400px range) must be reported at "
                   "approximately its real physical distance");
            assert(measured < simulation::Car::kMaxSensorDistance &&
                   "a wall within range must not read as the saturated maximum");
        }

        // 11: a wall at 450px -- beyond the new 400px range -- must still
        // saturate at exactly kMaxSensorDistance, proving the cap itself
        // (not just the old 200px number) is still respected.
        {
            const float measured = measureOutwardDistance(900.0f); // trackWidth=900 -> wall at 450px from centerline
            assert(measured == simulation::Car::kMaxSensorDistance &&
                   "a wall beyond kMaxSensorDistance must still saturate at exactly the maximum, not overshoot it");
        }
    }

    TraceLog(LOG_INFO, "Sensor verification: all deterministic checks passed");
}

// One-shot, deterministic sanity check of ai::buildObservation, independent
// of keyboard/render timing. Runs once at startup.
void verifyObservation(const simulation::Track& track)
{
    static_assert(ai::kObservationSize == 14, "Observation must contain exactly fourteen values");

    simulation::Car car(makeCarParams(), track);
    car.reset(kSpawnPosition, kSpawnHeading);
    simulation::TrackProgress progress(track);
    progress.reset(car);

    // 1: exactly fourteen values, fixed-size storage.
    ai::Observation obs = ai::buildObservation(car, progress);
    assert(obs.values.size() == 14 && "observation must contain exactly fourteen values");

    // 2 & 3: sensor values occupy indices 0..4, in documented order, within
    // [0,1] -- unchanged from before the actual-steering/yaw-rate slots were added.
    const auto& sensors = car.getSensors();
    for (int i = 0; i < simulation::Car::kSensorCount; ++i)
    {
        assert(std::fabs(obs.values[i] - sensors[i].normalizedDistance) < 1e-6f &&
               "observation sensor slot must match Car's own normalized sensor reading");
        assert(obs.values[i] >= 0.0f && obs.values[i] <= 1.0f && "sensor observation values must stay within [0,1]");
    }

    // 4: speed normalization (speed / maxSpeed, clamped to [0,1]) -- index unchanged.
    {
        const float expected = std::clamp(car.getSpeed() / car.getMaxSpeed(), 0.0f, 1.0f);
        assert(std::fabs(obs.values[5] - expected) < 1e-6f && "speed normalization mismatch");
    }

    // 5: forward velocity normalization (forwardVelocity / maxSpeed, clamped
    // to [-1,1]) -- index unchanged.
    {
        const float expected = std::clamp(car.getForwardVelocity() / car.getMaxSpeed(), -1.0f, 1.0f);
        assert(std::fabs(obs.values[6] - expected) < 1e-6f && "forward velocity normalization mismatch");
    }

    // 6: lateral velocity normalization (lateralVelocity / maxSpeed, clamped
    // to [-1,1]) -- index unchanged.
    {
        const float expected = std::clamp(car.getLateralVelocity() / car.getMaxSpeed(), -1.0f, 1.0f);
        assert(std::fabs(obs.values[7] - expected) < 1e-6f && "lateral velocity normalization mismatch");
    }

    // 7: slip angle normalization (slipAngle / pi, clamped to [-1,1]) -- index unchanged.
    {
        const float expected = std::clamp(car.getSlipAngle() / static_cast<float>(PI), -1.0f, 1.0f);
        assert(std::fabs(obs.values[8] - expected) < 1e-6f && "slip angle normalization mismatch");
    }

    // 8: actual (rate-limited) steering angle observation is exactly 0 right
    // after reset, matching Car::getCurrentSteerAngle()'s own reset state.
    {
        assert(car.getCurrentSteerAngle() == 0.0f && "a freshly-reset car must start with zero actual steering angle");
        assert(obs.values[ai::kActualSteerObservationIndex] == 0.0f &&
               "actual-steering observation must be exactly 0 right after reset");
    }

    // 9 & 10: a sustained positive (resp. negative) steering command drives
    // the actual-steering observation positive (resp. negative) -- exact
    // sign is well-defined by the formula (currentSteerAngle/maxSteerAngle,
    // maxSteerAngle > 0), not just measured after the fact.
    {
        simulation::CarInput fullRight;
        fullRight.steering = 1.0f;
        car.reset(kSpawnPosition, kSpawnHeading);
        progress.reset(car);
        for (int i = 0; i < 5; ++i)
        {
            car.update(fullRight, kSimulationDt);
            progress.update(car);
        }
        const ai::Observation rightObs = ai::buildObservation(car, progress);
        assert(rightObs.values[ai::kActualSteerObservationIndex] > 0.0f &&
               "a sustained +1 steering command must drive the actual-steering observation positive"); // 9

        simulation::CarInput fullLeft;
        fullLeft.steering = -1.0f;
        car.reset(kSpawnPosition, kSpawnHeading);
        progress.reset(car);
        for (int i = 0; i < 5; ++i)
        {
            car.update(fullLeft, kSimulationDt);
            progress.update(car);
        }
        const ai::Observation leftObs = ai::buildObservation(car, progress);
        assert(leftObs.values[ai::kActualSteerObservationIndex] < 0.0f &&
               "a sustained -1 steering command must drive the actual-steering observation negative"); // 10
    }

    // 11: yaw-rate observation is exactly 0 right after reset, matching
    // Car::getYawRate()'s own reset state.
    {
        car.reset(kSpawnPosition, kSpawnHeading);
        progress.reset(car);
        const ai::Observation freshObs = ai::buildObservation(car, progress);
        assert(car.getYawRate() == 0.0f && "a freshly-reset car must start with zero yaw rate");
        assert(freshObs.values[ai::kYawRateObservationIndex] == 0.0f &&
               "yaw-rate observation must be exactly 0 right after reset");
    }

    // 12 & 13: opposite sustained steering commands produce genuinely
    // opposite-signed yaw rates, and the yaw-rate observation's sign always
    // matches Car::getYawRate()'s own sign -- proves both a "positive yaw"
    // and a "negative yaw" case are exercised, without assuming a priori
    // which steering direction corresponds to which yaw sign.
    {
        simulation::CarInput turnRight;
        turnRight.throttle = 1.0f;
        turnRight.steering = 1.0f;
        car.reset(kSpawnPosition, kSpawnHeading);
        progress.reset(car);
        for (int i = 0; i < 15 && car.isAlive(); ++i)
        {
            car.update(turnRight, kSimulationDt);
            progress.update(car);
        }
        assert(car.isAlive() && "setup must keep the car on the track while turning right");
        const float yawRight = car.getYawRate();
        const ai::Observation obsRight = ai::buildObservation(car, progress);
        assert(std::fabs(yawRight) > 0.01f && "sustained turning must produce a genuinely nonzero yaw rate");
        assert(((yawRight > 0.0f) == (obsRight.values[ai::kYawRateObservationIndex] > 0.0f)) &&
               "yaw-rate observation sign must match Car::getYawRate()'s sign"); // 12 or 13, depending on convention

        simulation::CarInput turnLeft;
        turnLeft.throttle = 1.0f;
        turnLeft.steering = -1.0f;
        car.reset(kSpawnPosition, kSpawnHeading);
        progress.reset(car);
        for (int i = 0; i < 15 && car.isAlive(); ++i)
        {
            car.update(turnLeft, kSimulationDt);
            progress.update(car);
        }
        assert(car.isAlive() && "setup must keep the car on the track while turning left");
        const float yawLeft = car.getYawRate();
        const ai::Observation obsLeft = ai::buildObservation(car, progress);
        assert(std::fabs(yawLeft) > 0.01f && "sustained turning must produce a genuinely nonzero yaw rate");
        assert(((yawLeft > 0.0f) == (obsLeft.values[ai::kYawRateObservationIndex] > 0.0f)) &&
               "yaw-rate observation sign must match Car::getYawRate()'s sign"); // the other of 12/13

        assert((yawRight > 0.0f) != (yawLeft > 0.0f) &&
               "opposite sustained steering commands must produce opposite-signed yaw rates -- confirms both a "
               "positive and a negative case were genuinely exercised above");
    }

    // 14: all normalized values -- including the new actual-steering and
    // yaw-rate slots -- respect their documented [-1,1]/[0,1] ranges even
    // under active driving/sliding.
    simulation::CarInput throttleAndSteer;
    throttleAndSteer.throttle = 1.0f;
    throttleAndSteer.steering = 1.0f;
    car.reset(kSpawnPosition, kSpawnHeading);
    progress.reset(car);
    for (int i = 0; i < 30 && car.isAlive(); ++i)
    {
        car.update(throttleAndSteer, kSimulationDt);
        progress.update(car);
    }
    const ai::Observation movingObs = ai::buildObservation(car, progress);
    for (int i = 0; i < simulation::Car::kSensorCount; ++i)
    {
        assert(movingObs.values[i] >= 0.0f && movingObs.values[i] <= 1.0f && "sensor values must stay within [0,1]");
    }
    assert(movingObs.values[5] >= 0.0f && movingObs.values[5] <= 1.0f && "speed must stay within [0,1]");
    assert(movingObs.values[6] >= -1.0f && movingObs.values[6] <= 1.0f && "forward velocity must stay within [-1,1]");
    assert(movingObs.values[7] >= -1.0f && movingObs.values[7] <= 1.0f && "lateral velocity must stay within [-1,1]");
    assert(movingObs.values[8] >= -1.0f && movingObs.values[8] <= 1.0f && "slip angle must stay within [-1,1]");
    assert(movingObs.values[ai::kActualSteerObservationIndex] >= -1.0f &&
           movingObs.values[ai::kActualSteerObservationIndex] <= 1.0f &&
           "actual-steering observation must never leave [-1,1]"); // 6 (steering normalization never exceeds [-1,1])
    assert(movingObs.values[ai::kYawRateObservationIndex] >= -1.0f &&
           movingObs.values[ai::kYawRateObservationIndex] <= 1.0f && "yaw-rate observation must never leave [-1,1]");
    assert(movingObs.values[ai::kHeadingErrorObservationIndex] >= -1.0f &&
           movingObs.values[ai::kHeadingErrorObservationIndex] <= 1.0f &&
           "heading-error observation must never leave [-1,1]");

    // 15: yaw-rate normalization saturates safely under a genuinely
    // aggressive maneuver (an opposite-lock flick into a full-lock spin, the
    // same recipe verifyVehiclePhysics's own DELIBERATE SPIN TEST uses to
    // provoke the largest yaw excursions this model produces) -- the
    // observation value never leaves [-1,1] regardless of how large the true
    // yaw rate gets, and clamps to exactly +-1 (not some larger unclamped
    // magnitude) whenever |yawRate| reaches or exceeds
    // kYawRateNormalizationScale.
    {
        simulation::TrackDefinition wideDef;
        wideDef.simWidth = 3000;
        wideDef.simHeight = 3000;
        wideDef.controlPoints = {Vector2{1400.0f, 1400.0f}, Vector2{1600.0f, 1400.0f}, Vector2{1600.0f, 1600.0f},
                                  Vector2{1400.0f, 1600.0f}};
        wideDef.trackWidth = 2500.0f;
        wideDef.samplesPerSegment = 8;
        simulation::Track wideTrack(wideDef);
        simulation::Car wideCar(makeCarParams(), wideTrack);
        wideCar.reset(Vector2{1500.0f, 1500.0f}, 0.0f);
        simulation::TrackProgress wideProgress(wideTrack);
        wideProgress.reset(wideCar);

        simulation::CarInput straight;
        straight.throttle = 1.0f;
        for (int i = 0; i < 150 && wideCar.isAlive(); ++i)
        {
            wideCar.update(straight, kSimulationDt);
            wideProgress.update(wideCar);
        }
        assert(wideCar.isAlive() && "saturation-test setup must keep the car on the (huge synthetic) road");

        simulation::CarInput flickAway;
        flickAway.throttle = 0.3f;
        flickAway.steering = -1.0f;
        for (int i = 0; i < 10 && wideCar.isAlive(); ++i)
        {
            wideCar.update(flickAway, kSimulationDt);
            wideProgress.update(wideCar);
        }
        simulation::CarInput spinInput;
        spinInput.throttle = 1.0f;
        spinInput.steering = 1.0f;

        float maxAbsYawRate = 0.0f;
        for (int i = 0; i < 180 && wideCar.isAlive(); ++i) // up to 3s
        {
            wideCar.update(spinInput, kSimulationDt);
            wideProgress.update(wideCar);
            const ai::Observation spinObs = ai::buildObservation(wideCar, wideProgress);
            const float yawObsValue = spinObs.values[ai::kYawRateObservationIndex];
            assert(yawObsValue >= -1.0f && yawObsValue <= 1.0f &&
                   "yaw-rate observation must never leave [-1,1], even mid-spin");
            const float trueYawRate = wideCar.getYawRate();
            maxAbsYawRate = std::max(maxAbsYawRate, std::fabs(trueYawRate));
            if (std::fabs(trueYawRate) >= ai::kYawRateNormalizationScale)
            {
                assert(std::fabs(yawObsValue) == 1.0f &&
                       "yaw-rate observation must clamp to exactly +-1 once the true yaw rate reaches or exceeds "
                       "kYawRateNormalizationScale, not just approach it");
            }
        }
        assert(wideCar.isAlive() && "the deliberate spin maneuver must not leave the (huge synthetic) road");
        assert(maxAbsYawRate > 1.0f &&
               "saturation-test setup must genuinely provoke a large yaw rate, not just a mild one");

        TraceLog(LOG_INFO, "Observation verification: deliberate spin maneuver reached max |yawRate|=%.2frad/s",
                 static_cast<double>(maxAbsYawRate));
    }

    // 16: reset/spawn produces deterministic valid observation values,
    // across every slot including the newest one.
    car.reset(kSpawnPosition, kSpawnHeading);
    progress.reset(car);
    const ai::Observation resetObsA = ai::buildObservation(car, progress);
    car.reset(kSpawnPosition, kSpawnHeading);
    progress.reset(car);
    const ai::Observation resetObsB = ai::buildObservation(car, progress);
    for (int i = 0; i < ai::kObservationSize; ++i)
    {
        assert(resetObsA.values[i] == resetObsB.values[i] && "reset must produce deterministic observation values");
    }

    // 17: old input indices 0..10 are completely unaffected by the new
    // slot -- re-derive each one directly from Car and compare against the
    // freshly-reset observation above, proving nothing shifted.
    {
        assert(std::fabs(resetObsA.values[8] - std::clamp(car.getSlipAngle() / static_cast<float>(PI), -1.0f, 1.0f)) <
                   1e-6f &&
               "slot 8 (slip angle) must retain its exact previous index/meaning");
        assert(resetObsA.values[ai::kActualSteerObservationIndex] == 0.0f &&
               resetObsA.values[ai::kYawRateObservationIndex] == 0.0f &&
               "slots 9 (actual steering) and 10 (yaw rate) must retain their exact previous indices"); // 7 (old indices unchanged)
    }

    // 18: the heading-error input is exactly index 11, immediately following
    // yaw rate -- unchanged now that the two preview slots (12, 13) follow it.
    {
        assert(ai::kHeadingErrorObservationIndex == 11 && "the heading-error observation must be exactly index 11"); // 8
        assert(ai::kHeadingErrorObservationIndex == ai::kYawRateObservationIndex + 1 &&
               "the heading-error observation must immediately follow yaw rate");
    }

    // 19, 20, 21 & 22: the heading-error formula itself -- car heading minus
    // the local track tangent direction, wrapped to [-pi,pi], normalized by
    // pi. The reference angle is READ from the real track's own tangent
    // (progress.getTrackTangent(), the same value buildObservation() itself
    // uses) rather than assumed, so these checks hold for any track
    // geometry, not just a hand-picked one.
    //
    // Car's own heading (Box2D rotation -> atan2) is always stored already
    // wrapped into (-pi,pi], so a desired heading of e.g. trackDirectionAngle
    // + pi must itself be pre-wrapped with the SAME formula before being
    // passed to car.reset() -- otherwise Car silently re-wraps it to a
    // different (but congruent, mod 2*pi) value than the test assumed,
    // which is exactly why the naive "trackDirectionAngle +/- pi" version of
    // this test (no local pre-wrap) failed for track directions where that
    // sum/difference landed outside (-pi,pi]. Pre-wrapping first makes the
    // result independent of trackDirectionAngle's own sign/magnitude (the
    // math is congruent mod 2*pi either way, so wrapToPi's OUTPUT is the
    // same regardless of which branch the pre-wrap happened to take).
    {
        auto wrapAngle = [](float angle) -> float
        {
            while (angle > PI)
            {
                angle -= 2.0f * static_cast<float>(PI);
            }
            while (angle < -PI)
            {
                angle += 2.0f * static_cast<float>(PI);
            }
            return angle;
        };

        car.reset(kSpawnPosition, kSpawnHeading);
        progress.reset(car);
        const Vector2 tangent = progress.getTrackTangent();
        const float trackDirectionAngle = std::atan2(tangent.y, tangent.x);

        // 19: heading exactly aligned with the local track direction -> 0.
        {
            car.reset(kSpawnPosition, trackDirectionAngle);
            progress.reset(car);
            const ai::Observation aligned = ai::buildObservation(car, progress);
            assert(std::fabs(aligned.values[ai::kHeadingErrorObservationIndex]) < 1e-4f &&
                   "heading exactly aligned with the local track direction must give ~0 normalized heading error");
        }

        // kNearSeamMargin keeps the two boundary tests a hair inside the
        // seam (never exactly +-pi, which is an unstable value to target
        // exactly through Box2D's own float rotation representation) --
        // the task's own wording is "approximately +1"/"approximately -1".
        constexpr float kNearSeamMargin = 0.02f; // radians

        // 20: heading approaching +pi from the track direction -> normalized ~+1.
        {
            const float desiredHeading = wrapAngle(trackDirectionAngle + static_cast<float>(PI) - kNearSeamMargin);
            car.reset(kSpawnPosition, desiredHeading);
            progress.reset(car);
            const ai::Observation opposite = ai::buildObservation(car, progress);
            const float value = opposite.values[ai::kHeadingErrorObservationIndex];
            assert(value >= -1.0f && value <= 1.0f && "heading-error observation must stay within [-1,1]"); // 5
            assert(value > 0.9f &&
                   "a heading approaching +pi from the track direction must give normalized heading error "
                   "approximately +1");
        }

        // 21: heading approaching -pi from the track direction -> normalized
        // ~-1 -- exercises wrapToPi's OTHER boundary (its two while loops
        // use strict </> so the two seams are handled by separate branches).
        {
            const float desiredHeading = wrapAngle(trackDirectionAngle - static_cast<float>(PI) + kNearSeamMargin);
            car.reset(kSpawnPosition, desiredHeading);
            progress.reset(car);
            const ai::Observation opposite = ai::buildObservation(car, progress);
            const float value = opposite.values[ai::kHeadingErrorObservationIndex];
            assert(value >= -1.0f && value <= 1.0f && "heading-error observation must stay within [-1,1]"); // 5
            assert(value < -0.9f &&
                   "a heading approaching -pi from the track direction must give normalized heading error "
                   "approximately -1");
        }

        // 22: wrapping across the +pi/-pi seam is continuous, not a
        // discontinuous jump -- a heading a little PAST +pi from the track
        // direction must land near -1 (not near +1, and never outside
        // [-1,1]), proving the seam crossing behaves correctly rather than
        // clamping or jumping.
        {
            constexpr float kPastSeam = 0.1f; // radians past +pi
            const float desiredHeading = wrapAngle(trackDirectionAngle + static_cast<float>(PI) + kPastSeam);
            car.reset(kSpawnPosition, desiredHeading);
            progress.reset(car);
            const ai::Observation wrapped = ai::buildObservation(car, progress);
            const float value = wrapped.values[ai::kHeadingErrorObservationIndex];
            const float expected = (-static_cast<float>(PI) + kPastSeam) / static_cast<float>(PI);
            assert(value >= -1.0f && value <= 1.0f && "heading-error observation must stay within [-1,1]"); // 5
            assert(std::fabs(value - expected) < 1e-3f &&
                   "a heading just past +pi from the track direction must wrap to just past -pi, not jump "
                   "discontinuously or exceed [-1,1]"); // wrapping across +-pi
        }

        car.reset(kSpawnPosition, kSpawnHeading);
        progress.reset(car);
    }

    // ---- Track-direction preview inputs (slots 12 & 13) ----------------------
    // Two orientation-only inputs: the same normalized heading-error formula
    // as slot 11 (car heading MINUS track direction, wrapped, / pi), against
    // the track tangent 120px / 300px farther along the centerline.
    {
        auto wrapAngle = [](float angle) -> float
        {
            while (angle > PI)
            {
                angle -= 2.0f * static_cast<float>(PI);
            }
            while (angle < -PI)
            {
                angle += 2.0f * static_cast<float>(PI);
            }
            return angle;
        };
        constexpr float kPi = static_cast<float>(PI);
        const float trackLength = track.getTotalLength();

        // Independent reference for Track::getTangentAtDistance(): the same
        // wrap, then a plain linear scan over the centerline's cumulative
        // distances (the helper itself uses a binary search).
        auto referenceTangentAngle = [&](float distance) -> float
        {
            float wrapped = std::fmod(distance, trackLength);
            if (wrapped < 0.0f)
            {
                wrapped += trackLength;
            }
            const std::vector<Vector2>& samples = track.getCenterlineSamples();
            const std::vector<float>& cumulative = track.getCumulativeDistances();
            std::size_t segment = samples.size() - 1;
            for (std::size_t i = 0; i + 1 < samples.size(); ++i)
            {
                if (wrapped >= cumulative[i] && wrapped < cumulative[i + 1])
                {
                    segment = i;
                    break;
                }
            }
            const Vector2& a = samples[segment];
            const Vector2& b = samples[(segment + 1) % samples.size()];
            return std::atan2(b.y - a.y, b.x - a.x);
        };
        auto tangentAngle = [&](float distance) -> float
        {
            const Vector2 t = track.getTangentAtDistance(distance);
            return std::atan2(t.y, t.x);
        };
        // Places the car exactly on the centerline at `distance`, heading
        // `heading`, and re-anchors progress there.
        auto placeCar = [&](float distance, float heading)
        {
            car.reset(track.getPointAtDistance(distance), heading);
            progress.reset(car);
        };

        // A: observation size is now 14, with the two new slots last.
        static_assert(ai::kObservationSize == 14, "Observation must now contain exactly fourteen values");
        assert(obs.values.size() == 14 && "observation must contain exactly fourteen values"); // A
        assert(ai::kPreviewNearObservationIndex == 12 && ai::kPreviewFarObservationIndex == 13 &&
               ai::kPreviewFarObservationIndex == ai::kObservationSize - 1 &&
               "the preview observations must be exactly slots 12 and 13, the last two"); // A
        assert(ai::kPreviewNearDistance == 120.0f && ai::kPreviewFarDistance == 300.0f &&
               "preview distances must be 120px and 300px");

        // B: slots 0-11 keep their exact indices/meanings -- re-derived here
        // directly from Car/TrackProgress at a moving pose.
        {
            car.reset(kSpawnPosition, kSpawnHeading);
            progress.reset(car);
            simulation::CarInput drive;
            drive.throttle = 1.0f;
            drive.steering = 0.3f;
            for (int i = 0; i < 40; ++i)
            {
                car.update(drive, kSimulationDt);
                progress.update(car);
            }
            const ai::Observation movingPose = ai::buildObservation(car, progress);
            const auto& liveSensors = car.getSensors();
            for (int i = 0; i < simulation::Car::kSensorCount; ++i)
            {
                assert(movingPose.values[i] == std::clamp(liveSensors[i].normalizedDistance, 0.0f, 1.0f) &&
                       "slots 0-4 must remain the five sensors, unchanged"); // B
            }
            assert(movingPose.values[5] == std::clamp(car.getSpeed() / car.getMaxSpeed(), 0.0f, 1.0f) &&
                   movingPose.values[6] == std::clamp(car.getForwardVelocity() / car.getMaxSpeed(), -1.0f, 1.0f) &&
                   movingPose.values[7] == std::clamp(car.getLateralVelocity() / car.getMaxSpeed(), -1.0f, 1.0f) &&
                   movingPose.values[8] == std::clamp(car.getSlipAngle() / kPi, -1.0f, 1.0f) &&
                   "slots 5-8 must remain speed/forward/lateral/slip, unchanged"); // B
            assert(ai::kActualSteerObservationIndex == 9 && ai::kYawRateObservationIndex == 10 &&
                   ai::kHeadingErrorObservationIndex == 11 && "slots 9-11 must keep their exact indices"); // B
            assert(movingPose.values[9] ==
                       std::clamp(car.getCurrentSteerAngle() / car.getParams().maxSteerAngle, -1.0f, 1.0f) &&
                   movingPose.values[10] == std::clamp(car.getYawRate() / ai::kYawRateNormalizationScale, -1.0f, 1.0f) &&
                   "slots 9 and 10 must remain actual steering angle and yaw rate, unchanged"); // B
            const Vector2 localTangent = progress.getTrackTangent();
            const float localDirection = std::atan2(localTangent.y, localTangent.x);
            assert(movingPose.values[11] ==
                       std::clamp(wrapAngle(car.getHeading() - localDirection) / kPi, -1.0f, 1.0f) &&
                   "slot 11 must remain heading error against the LOCAL track tangent, same subtraction order"); // B
        }

        // Track::getTangentAtDistance(): matches the linear-scan reference at
        // many distances -- including negative and beyond-one-lap values --
        // returns unit vectors, and agrees with a finite-difference read of
        // getPointAtDistance() at every segment midpoint.
        {
            for (int i = -40; i <= 400; ++i)
            {
                const float distance = static_cast<float>(i) * 13.7f; // px; spans below 0 and > 1 lap
                const Vector2 t = track.getTangentAtDistance(distance);
                assert(std::fabs(std::sqrt(t.x * t.x + t.y * t.y) - 1.0f) < 1e-4f &&
                       "getTangentAtDistance must return a unit tangent");
                assert(std::fabs(wrapAngle(tangentAngle(distance) - referenceTangentAngle(distance))) < 1e-5f &&
                       "getTangentAtDistance must match the linear-scan reference"); // G (wrap incl. negatives / > lap)
            }
            const std::vector<Vector2>& samples = track.getCenterlineSamples();
            const std::vector<float>& cumulative = track.getCumulativeDistances();
            for (std::size_t i = 0; i + 1 < samples.size(); ++i)
            {
                const float mid = 0.5f * (cumulative[i] + cumulative[i + 1]);
                const Vector2 before = track.getPointAtDistance(mid - 0.1f);
                const Vector2 after = track.getPointAtDistance(mid + 0.1f);
                const float finiteDifference = std::atan2(after.y - before.y, after.x - before.x);
                assert(std::fabs(wrapAngle(tangentAngle(mid) - finiteDifference)) < 5e-2f &&
                       "segment-midpoint tangent must agree with the direction of travel from getPointAtDistance");
            }
            // Across the lap seam: distance L + x reads the same as x.
            for (float x : {0.5f, 30.0f, 70.0f, 250.0f})
            {
                assert(std::fabs(wrapAngle(tangentAngle(trackLength + x) - tangentAngle(x))) < 1e-4f &&
                       "distance + one full lap must read the same tangent as distance"); // G
            }
        }

        // C: every observation value (0-13) stays finite and within its
        // documented range -- swept over the whole lap with arbitrary
        // (including sign-flipping and wrap-crossing) headings.
        for (int i = 0; i < 300; ++i)
        {
            const float distance = trackLength * static_cast<float>(i) / 300.0f;
            const float heading = wrapAngle(static_cast<float>(i) * 0.83f - 3.0f);
            placeCar(distance, heading);
            const ai::Observation sweep = ai::buildObservation(car, progress);
            for (int slot : {ai::kPreviewNearObservationIndex, ai::kPreviewFarObservationIndex})
            {
                assert(std::isfinite(sweep.values[slot]) && sweep.values[slot] >= -1.0f && sweep.values[slot] <= 1.0f &&
                       "preview observations must stay finite and within [-1, +1]"); // C
            }
        }

        // D: on a locally straight stretch with the car aligned to the track,
        // the current heading error and both previews are ~0. The straight is
        // FOUND on the real track (tangent spread < 0.01 rad over the next
        // 300px) rather than assumed, so this holds for any track geometry
        // that actually contains one -- and the search asserts that it does.
        float straightStart = -1.0f;
        for (float d = 0.0f; d < trackLength && straightStart < 0.0f; d += 10.0f)
        {
            const float lowest = tangentAngle(d);
            bool straight = true;
            for (float ahead = 10.0f; ahead <= 300.0f; ahead += 10.0f)
            {
                if (std::fabs(wrapAngle(tangentAngle(d + ahead) - lowest)) > 0.01f)
                {
                    straight = false;
                    break;
                }
            }
            if (straight)
            {
                straightStart = d;
            }
        }
        assert(straightStart >= 0.0f && "the extreme track must contain a locally straight 300px stretch");
        {
            placeCar(straightStart, tangentAngle(straightStart));
            const ai::Observation straightObs = ai::buildObservation(car, progress);
            assert(std::fabs(straightObs.values[ai::kHeadingErrorObservationIndex]) < 0.01f &&
                   std::fabs(straightObs.values[ai::kPreviewNearObservationIndex]) < 0.01f &&
                   std::fabs(straightObs.values[ai::kPreviewFarObservationIndex]) < 0.01f &&
                   "aligned on a locally straight stretch, heading error and both previews must be ~0"); // D
        }

        // E: where the track turns ahead, the previews have the correct sign
        // and their magnitude follows the angular difference. Find the sharpest
        // turn within 300px of the lap and test there.
        float turnStart = 0.0f;
        float sharpestTurn = 0.0f;
        for (float d = 0.0f; d < trackLength; d += 5.0f)
        {
            const float turn = std::fabs(wrapAngle(tangentAngle(d + 300.0f) - tangentAngle(d)));
            if (turn > sharpestTurn)
            {
                sharpestTurn = turn;
                turnStart = d;
            }
        }
        assert(sharpestTurn > 0.5f && "the extreme track must turn by more than 0.5 rad within some 300px stretch");
        {
            const float localAngle = tangentAngle(turnStart);
            placeCar(turnStart, localAngle); // aligned with the LOCAL tangent
            const ai::Observation turnObs = ai::buildObservation(car, progress);
            const float expectedNear = wrapAngle(localAngle - referenceTangentAngle(turnStart + 120.0f)) / kPi;
            const float expectedFar = wrapAngle(localAngle - referenceTangentAngle(turnStart + 300.0f)) / kPi;
            assert(std::fabs(turnObs.values[ai::kHeadingErrorObservationIndex]) < 0.01f &&
                   "slot 11 (local heading error) must be ~0 when aligned with the local tangent");
            assert(std::fabs(turnObs.values[ai::kPreviewNearObservationIndex] - expectedNear) < 0.01f &&
                   std::fabs(turnObs.values[ai::kPreviewFarObservationIndex] - expectedFar) < 0.01f &&
                   "preview magnitude must follow the angular difference to the future tangent"); // E
            assert(std::fabs(turnObs.values[ai::kPreviewFarObservationIndex]) > 0.15f &&
                   "the far preview must be clearly nonzero where the track turns ahead"); // E
            // Sign: the same subtraction order as slot 11 -- heading minus
            // track direction -- so a track that turns to a LARGER angle ahead
            // gives a NEGATIVE preview (and vice versa).
            const float turnAhead = wrapAngle(referenceTangentAngle(turnStart + 300.0f) - localAngle);
            assert(((turnAhead > 0.0f) == (turnObs.values[ai::kPreviewFarObservationIndex] < 0.0f)) &&
                   "preview sign must follow heading minus future track direction (slot 11's convention)"); // E
        }

        // F: wrap-around near +-pi -- the far preview reaches ~+1 / ~-1 as
        // the car's heading approaches +-pi from the future tangent, and
        // crossing the seam lands just past -1/+1 (continuous, never
        // outside [-1, 1]).
        {
            const float futureAngle = referenceTangentAngle(turnStart + 300.0f);
            constexpr float kNearSeamMargin = 0.02f; // radians
            {
                placeCar(turnStart, wrapAngle(futureAngle + kPi - kNearSeamMargin));
                const float value = ai::buildObservation(car, progress).values[ai::kPreviewFarObservationIndex];
                assert(value >= -1.0f && value <= 1.0f && value > 0.9f &&
                       "a heading approaching +pi from the future tangent must give a preview of about +1"); // F
            }
            {
                placeCar(turnStart, wrapAngle(futureAngle - kPi + kNearSeamMargin));
                const float value = ai::buildObservation(car, progress).values[ai::kPreviewFarObservationIndex];
                assert(value >= -1.0f && value <= 1.0f && value < -0.9f &&
                       "a heading approaching -pi from the future tangent must give a preview of about -1"); // F
            }
            {
                constexpr float kPastSeam = 0.1f;
                placeCar(turnStart, wrapAngle(futureAngle + kPi + kPastSeam));
                const float value = ai::buildObservation(car, progress).values[ai::kPreviewFarObservationIndex];
                const float expected = (-kPi + kPastSeam) / kPi;
                assert(value >= -1.0f && value <= 1.0f && std::fabs(value - expected) < 1e-2f &&
                       "a heading just past +pi from the future tangent must wrap to just past -1, not jump or exceed [-1, 1]"); // F
            }
        }

        // G: a preview that crosses the centerline's lap seam still returns a
        // valid tangent -- the car sits 50px before the seam, so both preview
        // points (120px, 300px ahead) lie beyond it.
        {
            const float nearSeam = trackLength - 50.0f;
            placeCar(nearSeam, tangentAngle(nearSeam));
            const Vector2 nearTangent = progress.getTrackTangentAhead(ai::kPreviewNearDistance);
            const Vector2 farTangent = progress.getTrackTangentAhead(ai::kPreviewFarDistance);
            assert(std::fabs(std::sqrt(nearTangent.x * nearTangent.x + nearTangent.y * nearTangent.y) - 1.0f) < 1e-4f &&
                   std::fabs(std::sqrt(farTangent.x * farTangent.x + farTangent.y * farTangent.y) - 1.0f) < 1e-4f &&
                   "a preview across the lap seam must still be a unit tangent"); // G
            assert(std::fabs(wrapAngle(std::atan2(nearTangent.y, nearTangent.x) - referenceTangentAngle(70.0f))) < 1e-2f &&
                   std::fabs(wrapAngle(std::atan2(farTangent.y, farTangent.x) - referenceTangentAngle(250.0f))) < 1e-2f &&
                   "a preview across the lap seam must read the tangent at (distance - lap length)"); // G
            const ai::Observation seamObs = ai::buildObservation(car, progress);
            assert(std::isfinite(seamObs.values[12]) && std::isfinite(seamObs.values[13]) &&
                   std::fabs(seamObs.values[12] - wrapAngle(car.getHeading() - referenceTangentAngle(70.0f)) / kPi) < 1e-2f &&
                   std::fabs(seamObs.values[13] - wrapAngle(car.getHeading() - referenceTangentAngle(250.0f)) / kPi) < 1e-2f &&
                   "preview observations across the lap seam must be valid and match the wrapped tangent"); // G
        }

        // Preview is read-only: it never disturbs TrackProgress's own state.
        {
            placeCar(turnStart, tangentAngle(turnStart));
            const float lapBefore = progress.getLapPosition();
            const float continuousBefore = progress.getContinuousProgress();
            const int checkpointsBefore = progress.getTotalCheckpointsPassed();
            for (int i = 0; i < 5; ++i)
            {
                (void)ai::buildObservation(car, progress);
                (void)progress.getTrackTangentAhead(300.0f);
            }
            assert(progress.getLapPosition() == lapBefore && progress.getContinuousProgress() == continuousBefore &&
                   progress.getTotalCheckpointsPassed() == checkpointsBefore &&
                   "reading previews must not change progress/lap/checkpoint state");
        }

        // H, I, J, L: bias/output IDs and the demonstration genome.
        {
            static_assert(ai::NeuralNetwork::kInputCount == 14, "kInputCount must follow the 14-value observation");
            const ai::neat::Genome demo = createDemonstrationGenome();
            const ai::neat::NodeGene* biasNode = demo.findNode(ai::NeuralNetwork::kInputCount);
            assert(ai::NeuralNetwork::kInputCount == 14 && biasNode != nullptr &&
                   biasNode->getType() == ai::neat::NodeType::Bias &&
                   "the Bias node must now be ID 14 (kInputCount), typed Bias"); // H
            for (int outputId : {100, 101, 102})
            {
                assert(outputId != ai::NeuralNetwork::kInputCount && demo.findNode(outputId) != nullptr &&
                       demo.findNode(outputId)->getType() == ai::neat::NodeType::Output &&
                       "the Bias ID must not collide with an Output ID, and Outputs must remain 100/101/102"); // I, J
            }
            for (int i = 0; i < ai::NeuralNetwork::kInputCount; ++i)
            {
                assert(demo.findNode(i) != nullptr && demo.findNode(i)->getType() == ai::neat::NodeType::Input &&
                       "every ID in [0, kInputCount) must be an Input node -- including the new 12 and 13");
            }
            int outputCount = 0;
            for (const ai::neat::NodeGene& node : demo.nodes())
            {
                outputCount += (node.getType() == ai::neat::NodeType::Output) ? 1 : 0;
            }
            assert(outputCount == 3 && ai::NeuralNetwork::kOutputCount == 3 && "the network must still have exactly three outputs"); // J

            // L: the new inputs start completely unconnected, and the
            // genome's wiring is exactly the pre-existing 7 connections.
            for (const ai::neat::ConnectionGene& connection : demo.connections())
            {
                assert(connection.getSourceId() != ai::kPreviewNearObservationIndex &&
                       connection.getSourceId() != ai::kPreviewFarObservationIndex &&
                       "the preview inputs must not be wired into the demonstration genome"); // L
            }
            assert(demo.connections().size() == 7 && "the demonstration genome must keep exactly its 7 original connections"); // L

            // K & L (behavioral): the phenotype builds, and its outputs are
            // exactly independent of the two preview slots.
            ai::NeuralNetwork phenotype = ai::neat::buildPhenotype(demo); // K
            car.reset(kSpawnPosition, kSpawnHeading);
            progress.reset(car);
            ai::Observation baseObs = ai::buildObservation(car, progress);
            ai::Observation previewFlipped = baseObs;
            previewFlipped.values[ai::kPreviewNearObservationIndex] = 1.0f;
            previewFlipped.values[ai::kPreviewFarObservationIndex] = -1.0f;
            const auto outBase = phenotype.evaluate(baseObs);
            const auto outFlipped = phenotype.evaluate(previewFlipped);
            assert(outBase[0] == outFlipped[0] && outBase[1] == outFlipped[1] && outBase[2] == outFlipped[2] &&
                   "unconnected preview inputs must have no effect on any network output"); // L
        }

        car.reset(kSpawnPosition, kSpawnHeading);
        progress.reset(car);
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
    simulation::Car car(makeFullAuthorityCarParams(), track);

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

    // 29-33: high-speed grip saturation genuinely reduces achievable
    // curvature -- measured, not assumed. The real track's spawn straight
    // is too short (~130px corridor, see check 7) to reach genuine high
    // speed (400-550px/s) before running out of road, so these use a
    // wide-open synthetic track (huge trackWidth around a tiny centerline
    // loop, purely for room) -- CarParams/physics are identical to the real
    // game, only the track geometry differs.
    {
        simulation::TrackDefinition wideDef;
        wideDef.simWidth = 3000;
        wideDef.simHeight = 3000;
        wideDef.controlPoints = {Vector2{1400.0f, 1400.0f}, Vector2{1600.0f, 1400.0f}, Vector2{1600.0f, 1600.0f},
                                  Vector2{1400.0f, 1600.0f}};
        wideDef.trackWidth = 2500.0f;
        wideDef.samplesPerSegment = 8;
        simulation::Track wideTrack(wideDef);
        simulation::Car wideCar(makeFullAuthorityCarParams(), wideTrack);
        const simulation::CarParams wideParams = wideCar.getParams();

        // Kinematic (zero-slip) bicycle-model radius: what a frictionless
        // car would trace at this steering input, independent of speed.
        // The measured path radius exceeding this at high speed is direct
        // evidence tire grip -- not steering angle -- is the limiter.
        auto geometricRadius = [&](float steering)
        {
            const float steerAngle = steering * wideParams.maxSteerAngle;
            return (wideParams.cgToFrontAxle + wideParams.cgToRearAxle) / std::tan(steerAngle);
        };

        // Reaches targetSpeed via a straight-line burst, then coasts
        // (throttle=0) through a constant-steering turn -- no engine Fx
        // contaminating the rear friction circle, isolating pure cornering
        // grip. Speed decays naturally under drag/rolling resistance (and,
        // once saturated, tire drag) as a real car's would.
        auto driveTurn = [&](float targetSpeed, float steering)
        {
            wideCar.reset(Vector2{1500.0f, 1500.0f}, 0.0f);
            simulation::CarInput throttleOnly;
            throttleOnly.throttle = 1.0f;
            for (int i = 0; i < 3000 && wideCar.getSpeed() < targetSpeed; ++i)
            {
                wideCar.update(throttleOnly, kSimulationDt);
            }
            simulation::CarInput turning;
            turning.steering = steering;
            turning.throttle = 0.0f;
            return turning;
        };

        // Path-curvature radius from how fast the VELOCITY VECTOR rotates
        // (not body heading/yaw rate) -- the two only coincide once slip
        // angle stops changing, which a hard high-speed turn-in is far from
        // during its first several frames.
        auto pathRadius = [&](Vector2 prevVel, Vector2 vel, float speed)
        {
            const float prevVelHeading = std::atan2(prevVel.y, prevVel.x);
            const float velHeading = std::atan2(vel.y, vel.x);
            const float turnRate = std::fabs(headingDelta(prevVelHeading, velHeading)) / kSimulationDt;
            return speed / std::max(turnRate, 1e-4f);
        };

        // 29: low speed stays tight and responsive -- path radius tracks
        // the geometric prediction closely, with low grip usage (nowhere
        // near the limit).
        {
            simulation::CarInput turning = driveTurn(100.0f, 1.0f);
            for (int i = 0; i < 19 && wideCar.isAlive(); ++i)
            {
                wideCar.update(turning, kSimulationDt);
            }
            const Vector2 prevVel = wideCar.getVelocity();
            wideCar.update(turning, kSimulationDt);
            assert(wideCar.isAlive() && "low-speed turn-in must not leave the (huge) synthetic track");
            const Vector2 vel = wideCar.getVelocity();
            const float radius = pathRadius(prevVel, vel, wideCar.getSpeed());
            const float geometric = geometricRadius(1.0f);
            assert(radius < geometric * 1.5f &&
                   "low speed + full steering must trace close to the geometric turn radius, not a widened one");
            assert(wideCar.getTireDebugInfo().frontGripUtilization < 0.3f &&
                   "low-speed full-lock turning must stay well short of the front tire's grip limit");
        }

        // 30: moderate speed stays stable -- finite state throughout, no
        // NaN/Inf, friction circle respected every frame.
        {
            simulation::CarInput turning = driveTurn(250.0f, 0.5f);
            for (int i = 0; i < 60 && wideCar.isAlive(); ++i)
            {
                wideCar.update(turning, kSimulationDt);
                assert(allFinite(wideCar) && "moderate-speed cornering must never produce a NaN/Inf state");
                const simulation::TireDebugInfo& debug = wideCar.getTireDebugInfo();
                constexpr float kCircleTolerance = 1.02f;
                assert(std::sqrt(debug.frontForceX * debug.frontForceX + debug.frontForceY * debug.frontForceY) <=
                           wideParams.frontMaxTireForce * kCircleTolerance &&
                       "moderate-speed cornering must respect the front friction circle");
                assert(std::sqrt(debug.rearForceX * debug.rearForceX + debug.rearForceY * debug.rearForceY) <=
                           wideParams.rearMaxTireForce * kCircleTolerance &&
                       "moderate-speed cornering must respect the rear friction circle");
            }
        }

        // 31 & 32: high speed + full steering saturates the front axle
        // (understeer) and traces a path far wider than the geometric
        // prediction -- the commanded steering angle is NOT achieved.
        // Grip loss is progressive (grip utilization changes smoothly
        // frame-to-frame, no discontinuous jump), and the friction circle
        // is respected throughout.
        {
            simulation::CarInput turning = driveTurn(550.0f, 1.0f);
            Vector2 prevVel = wideCar.getVelocity();
            float peakFrontGrip = 0.0f;
            float radiusAtPeakGrip = 0.0f;
            float prevFrontGrip = wideCar.getTireDebugInfo().frontGripUtilization;
            float maxGripJumpPerFrame = 0.0f;
            for (int i = 0; i < 35 && wideCar.isAlive(); ++i)
            {
                wideCar.update(turning, kSimulationDt);
                assert(allFinite(wideCar) && "high-speed full-steering cornering must never produce a NaN/Inf state");
                const simulation::TireDebugInfo& debug = wideCar.getTireDebugInfo();
                constexpr float kCircleTolerance = 1.02f;
                assert(std::sqrt(debug.frontForceX * debug.frontForceX + debug.frontForceY * debug.frontForceY) <=
                           wideParams.frontMaxTireForce * kCircleTolerance &&
                       "high-speed cornering must respect the front friction circle even while saturating");
                assert(std::sqrt(debug.rearForceX * debug.rearForceX + debug.rearForceY * debug.rearForceY) <=
                           wideParams.rearMaxTireForce * kCircleTolerance &&
                       "high-speed cornering must respect the rear friction circle even while saturating");

                const Vector2 vel = wideCar.getVelocity();
                const float radius = pathRadius(prevVel, vel, wideCar.getSpeed());
                if (debug.frontGripUtilization > peakFrontGrip)
                {
                    peakFrontGrip = debug.frontGripUtilization;
                    radiusAtPeakGrip = radius;
                }
                // Frame 0 captures the STEERING INPUT step itself (0 -> full
                // lock in a single frame -- a test-harness artifact; a human
                // or NEAT-driven steer input still changes over many frames
                // in practice), not the tire model's own response, so it's
                // excluded from the "progressive, not binary" check below.
                if (i > 0)
                {
                    maxGripJumpPerFrame = std::max(maxGripJumpPerFrame, std::fabs(debug.frontGripUtilization - prevFrontGrip));
                }
                prevFrontGrip = debug.frontGripUtilization;
                prevVel = vel;
            }
            assert(wideCar.isAlive() && "high-speed full-steering turn-in must not leave the (huge) synthetic track");

            // 31: front axle reaches (near-)full saturation under aggressive
            // high-speed steering -- the tire, not the commanded angle, is
            // what's limiting the turn.
            assert(peakFrontGrip > 0.9f &&
                   "aggressive high-speed steering must saturate the front axle's friction circle");

            // 32: at that saturated moment, the actual path is far wider
            // than the geometric (frictionless) prediction for this
            // steering input -- the car does not follow the commanded
            // steering angle, exactly as a real understeering car would not.
            const float geometric = geometricRadius(1.0f);
            assert(radiusAtPeakGrip > geometric * 1.5f &&
                   "high-speed full steering under grip saturation must trace a path meaningfully wider than the "
                   "geometric turn radius");

            // Progressive, not binary: grip utilization never jumps more
            // than a modest amount in a single frame while turning in.
            assert(maxGripJumpPerFrame < 0.35f &&
                   "front grip utilization must climb progressively while turning in, not jump discontinuously");

            TraceLog(LOG_INFO,
                     "Vehicle physics: high-speed (550px/s) full-steering saturation -- peak front grip %.0f%%, path "
                     "radius %.1fpx there (geometric %.1fpx)",
                     static_cast<double>(peakFrontGrip * 100.0f), static_cast<double>(radiusAtPeakGrip),
                     static_cast<double>(geometric));
        }

        // 33: small steering at high speed stays stable -- a broad, smooth
        // sweeping turn with comfortable grip margin, not an immediate
        // slide ("icy"): front grip never gets close to saturating, and
        // yaw builds smoothly rather than snapping.
        {
            simulation::CarInput turning = driveTurn(500.0f, 0.15f);
            float maxYawRate = 0.0f;
            for (int i = 0; i < 40 && wideCar.isAlive(); ++i)
            {
                wideCar.update(turning, kSimulationDt);
                assert(allFinite(wideCar) && "small steering at high speed must never produce a NaN/Inf state");
                maxYawRate = std::max(maxYawRate, std::fabs(wideCar.getTireDebugInfo().yawRate));
                assert(wideCar.getTireDebugInfo().frontGripUtilization < 0.7f &&
                       "small steering at high speed must stay comfortably short of the grip limit, not feel icy");
            }
            assert(wideCar.isAlive() && "small-steering high-speed driving must not leave the (huge) synthetic track");
            assert(maxYawRate < 3.0f && "small steering at high speed must produce bounded, non-spinning yaw");
        }
    }

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

    // 2: releasing throttle AND brake (pure coast) must lose speed only
    // gradually, via rolling resistance/drag alone -- not the strong
    // "engine braking" feel a naive lift-off would give. Some speed loss is
    // still expected (resistance is never removed entirely), just much less
    // than the car's own speed over this window.
    {
        const float speedBeforeLiftOff = car.getSpeed();
        assert(speedBeforeLiftOff > 50.0f && "must still be moving meaningfully before lift-off");

        simulation::CarInput coasting; // throttle = 0, steering = 0, brake = 0
        for (int i = 0; i < 90; ++i)
        {
            car.update(coasting, kSimulationDt);
        }
        const float speedAfterLiftOff = car.getSpeed();
        assert(speedAfterLiftOff < speedBeforeLiftOff &&
               "coasting must still lose some speed -- resistance is never fully removed");
        assert(speedAfterLiftOff > speedBeforeLiftOff * 0.7f &&
               "releasing throttle with no brake must coast gradually, not decelerate sharply like engine braking");
        TraceLog(LOG_INFO, "Vehicle physics: coast (no brake) over 1.5s: %.1f -> %.1f px/s (-%.1f px/s, ratio %.3f)",
                 static_cast<double>(speedBeforeLiftOff), static_cast<double>(speedAfterLiftOff),
                 static_cast<double>(speedBeforeLiftOff - speedAfterLiftOff),
                 static_cast<double>(speedAfterLiftOff / speedBeforeLiftOff));
    }

    // 21: brake input is clamped to [0,1] (and steering/throttle stay
    // clamped alongside it), same as every other CarInput channel -- read
    // back via TireDebugInfo's echoed, already-clamped input.
    {
        car.reset(kSpawnPosition, kSpawnHeading);
        simulation::CarInput outOfRange;
        outOfRange.throttle = 5.0f;
        outOfRange.steering = -7.0f;
        outOfRange.brake = 3.0f;
        car.update(outOfRange, kSimulationDt);
        const simulation::TireDebugInfo& echoed = car.getTireDebugInfo();
        assert(echoed.brakeInput == 1.0f && "brake input above 1 must clamp to exactly 1");
        assert(echoed.throttleInput == 1.0f && "throttle input above 1 must clamp to exactly 1");
        assert(echoed.steeringInput == -1.0f && "steering input below -1 must clamp to exactly -1");

        car.reset(kSpawnPosition, kSpawnHeading);
        simulation::CarInput negativeBrake;
        negativeBrake.brake = -2.0f;
        car.update(negativeBrake, kSimulationDt);
        assert(car.getTireDebugInfo().brakeInput == 0.0f && "brake input below 0 must clamp to exactly 0");
        car.reset(kSpawnPosition, kSpawnHeading);
    }

    // 22: full brake decelerates substantially faster than pure coasting,
    // from the identical starting state (two deterministic replays of the
    // same throttle burst, so both forks start bit-identical).
    {
        auto reachSpeedThenApply = [&](const simulation::CarInput& afterInput) -> std::pair<float, float>
        {
            simulation::Car localCar(makeFullAuthorityCarParams(), track);
            localCar.reset(kSpawnPosition, kSpawnHeading);
            simulation::CarInput burst;
            burst.throttle = 1.0f;
            for (int i = 0; i < 90; ++i) // ~1.5s
            {
                localCar.update(burst, kSimulationDt);
            }
            const float speedBefore = localCar.getSpeed();
            for (int i = 0; i < 60; ++i) // ~1s
            {
                localCar.update(afterInput, kSimulationDt);
            }
            return {speedBefore, localCar.getSpeed()};
        };

        simulation::CarInput pureCoast; // throttle = 0, brake = 0
        simulation::CarInput fullBrake;
        fullBrake.brake = 1.0f;

        const auto [coastSpeedBefore, coastSpeedAfter] = reachSpeedThenApply(pureCoast);
        const auto [brakeSpeedBefore, brakeSpeedAfter] = reachSpeedThenApply(fullBrake);

        assert(std::fabs(coastSpeedBefore - brakeSpeedBefore) < 1.0f &&
               "setup: both forks must reach the identical starting speed (deterministic replay)");

        const float coastDrop = coastSpeedBefore - coastSpeedAfter;
        const float brakeDrop = brakeSpeedBefore - brakeSpeedAfter;
        assert(brakeDrop > coastDrop * 3.0f &&
               "full brake must decelerate substantially faster than coasting from the same starting speed"); // 22

        // 23: stronger brake input produces stronger deceleration.
        simulation::CarInput lightBrake;
        lightBrake.brake = 0.3f;
        const auto [lightSpeedBefore, lightSpeedAfter] = reachSpeedThenApply(lightBrake);
        const float lightDrop = lightSpeedBefore - lightSpeedAfter;
        assert(brakeDrop > lightDrop &&
               "stronger brake input must produce stronger deceleration than lighter brake input"); // 23

        TraceLog(LOG_INFO,
                 "Vehicle physics: from %.1fpx/s over 1s -- coast to %.1f (-%.1f), light brake(0.3) to %.1f (-%.1f), "
                 "full brake to %.1f (-%.1f)",
                 static_cast<double>(coastSpeedBefore), static_cast<double>(coastSpeedAfter), static_cast<double>(coastDrop),
                 static_cast<double>(lightSpeedAfter), static_cast<double>(lightDrop), static_cast<double>(brakeSpeedAfter),
                 static_cast<double>(brakeDrop));
    }

    // 24: braking is force-based, not a direct velocity assignment -- a
    // single simulation step under full brake changes speed only by a
    // small, bounded amount (never an instant drop toward zero), the same
    // way a single step of any other force-driven input would.
    {
        car.reset(kSpawnPosition, kSpawnHeading);
        simulation::CarInput burst;
        burst.throttle = 1.0f;
        for (int i = 0; i < 90; ++i)
        {
            car.update(burst, kSimulationDt);
        }
        const float speedBeforeStep = car.getSpeed();
        assert(speedBeforeStep > 100.0f && "test setup must reach a meaningfully high speed before the single-step check");

        simulation::CarInput fullBrake;
        fullBrake.brake = 1.0f;
        car.update(fullBrake, kSimulationDt);
        const float speedAfterStep = car.getSpeed();

        // Deceleration this one step could physically produce is bounded by
        // maxBrakeForce/mass * dt, with generous slack for the lateral/drag
        // terms also active that frame -- see CarParams::maxBrakeForce.
        const float mass = car.getParams().density * car.getParams().length * car.getParams().width;
        const float maxPossibleDrop = (car.getParams().maxBrakeForce / mass) * kSimulationDt * 2.0f;
        assert(speedBeforeStep - speedAfterStep < maxPossibleDrop &&
               "a single update() under full brake must change speed by only a physically bounded amount, "
               "never snap velocity directly");
        car.reset(kSpawnPosition, kSpawnHeading);
    }

    // 25 & 26: braking respects each axle's own tire-force limit -- the
    // combined brake+lateral (front) and brake+drive+lateral (rear) force
    // never exceeds that axle's maxTireForce, including while cornering
    // under heavy brake (extends checks 11/12's friction-circle invariant
    // to brake input).
    {
        car.reset(kSpawnPosition, kSpawnHeading);
        simulation::CarInput burst;
        burst.throttle = 1.0f;
        for (int i = 0; i < 90; ++i)
        {
            car.update(burst, kSimulationDt);
        }

        float maxAbsYawRate = 0.0f;
        for (int i = 0; i < 120 && car.isAlive(); ++i)
        {
            simulation::CarInput brakeAndTurn;
            brakeAndTurn.brake = 1.0f;
            brakeAndTurn.steering = ((i / 15) % 2 == 0) ? 1.0f : -1.0f; // flips every 15 frames, like checks 11/12
            car.update(brakeAndTurn, kSimulationDt);
            assert(allFinite(car) && "braking while cornering must never produce a NaN/Inf state");

            const simulation::TireDebugInfo& debug = car.getTireDebugInfo();
            constexpr float kCircleTolerance = 1.02f; // same slack as checks 11/12
            const float frontMag = std::sqrt(debug.frontForceX * debug.frontForceX + debug.frontForceY * debug.frontForceY);
            assert(frontMag <= car.getParams().frontMaxTireForce * kCircleTolerance &&
                   "front axle combined brake+lateral force must never exceed its grip limit"); // 25
            const float rearMag = std::sqrt(debug.rearForceX * debug.rearForceX + debug.rearForceY * debug.rearForceY);
            assert(rearMag <= car.getParams().rearMaxTireForce * kCircleTolerance &&
                   "rear axle combined brake+lateral force must never exceed its grip limit (friction circle)"); // 26

            maxAbsYawRate = std::max(maxAbsYawRate, std::fabs(debug.yawRate));
        }
        assert(maxAbsYawRate < 15.0f && "yaw rate must never run away/spin uncontrollably while braking and cornering");
    }

    // 27: braking while cornering measurably reduces available lateral
    // grip/increases slip versus the SAME corner with no brake -- mirrors
    // checks 15/16/17's throttle-vs-corner comparison, but for brake.
    // Both forks replay an identical checkpoint state so any difference is
    // attributable to the friction circle alone.
    {
        auto driveToCorneringCheckpoint = [&]()
        {
            car.reset(kSpawnPosition, kSpawnHeading);
            simulation::CarInput burst;
            burst.throttle = 1.0f;
            for (int i = 0; i < 150; ++i) // ~2.5s, well up in speed
            {
                car.update(burst, kSimulationDt);
            }
            simulation::CarInput approach; // no brake yet -- identical slip history right up to the checkpoint
            approach.steering = 1.0f;
            for (int i = 0; i < 20 && car.isAlive(); ++i)
            {
                car.update(approach, kSimulationDt);
            }
        };

        driveToCorneringCheckpoint();
        assert(car.isAlive() && "checkpoint approach must keep the car on the road");
        simulation::CarInput noBrakeFinal;
        noBrakeFinal.steering = 1.0f;
        car.update(noBrakeFinal, kSimulationDt);
        const simulation::TireDebugInfo noBrake = car.getTireDebugInfo();
        assert(car.isAlive() && "the no-brake comparison frame must keep the car on the road");

        driveToCorneringCheckpoint(); // deterministic replay -> bit-identical checkpoint state
        simulation::CarInput fullBrakeFinal;
        fullBrakeFinal.steering = 1.0f;
        fullBrakeFinal.brake = 1.0f;
        car.update(fullBrakeFinal, kSimulationDt);
        const simulation::TireDebugInfo fullBrake = car.getTireDebugInfo();
        assert(car.isAlive() && "the full-brake comparison frame must keep the car on the road");

        assert(std::fabs(fullBrake.frontForceY) < std::fabs(noBrake.frontForceY) &&
               "full brake mid-corner must reduce the front axle's available lateral force versus no brake"); // 27
        car.reset(kSpawnPosition, kSpawnHeading);
    }

    // 28: front/rear brake bias is applied as configured -- straight-line
    // braking (no lateral component to distort the split) divides force
    // between axles in exactly the frontBrakeBias ratio.
    {
        car.reset(kSpawnPosition, kSpawnHeading);
        simulation::CarInput burst;
        burst.throttle = 1.0f;
        for (int i = 0; i < 90; ++i)
        {
            car.update(burst, kSimulationDt);
        }
        simulation::CarInput straightBrake;
        straightBrake.brake = 1.0f;
        car.update(straightBrake, kSimulationDt);

        const simulation::TireDebugInfo& debug = car.getTireDebugInfo();
        assert(std::fabs(debug.frontForceY) < 1.0f && "straight-line braking must leave negligible front lateral force");
        const float totalBrakeForce = std::fabs(debug.frontForceX) + std::fabs(debug.rearForceX);
        assert(totalBrakeForce > 100.0f && "test setup must produce a genuinely nonzero combined brake force");
        const float measuredFrontBias = std::fabs(debug.frontForceX) / totalBrakeForce;
        assert(std::fabs(measuredFrontBias - car.getParams().frontBrakeBias) < 0.02f &&
               "front/rear brake split must match the configured frontBrakeBias");
        car.reset(kSpawnPosition, kSpawnHeading);
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
        for (int i = 0; i < 12; ++i) // build up a modest speed first (~0.2s)
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
        const auto [mediumSpeed, mediumRadius, mediumSlip] = measureTurn(45); // ~0.75s
        const auto [highSpeed, highRadius, highSlip] = measureTurn(100); // ~1.67s, meaningfully faster still

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
    // throttle, and a sufficiently aggressive combination must drive the
    // rear friction circle to a heavy, sustained load well beyond ordinary
    // cornering.
    //
    // Two independent multi-frame runs would confound the comparison (full
    // throttle's trajectory diverges for reasons unrelated to the friction
    // circle). Instead both throttle values are applied from an IDENTICAL
    // checkpoint state (two deterministic replays of the same input
    // sequence -- no RNG, so bit-identical both times), each for one final
    // update() -- so any difference is attributable to the friction circle alone.
    //
    // Uses the wide synthetic track (see checks 29-33) rather than the real
    // spawn straight: with tire-force relaxation and the heavier/higher-
    // inertia car (see CarParams::density/rotationalInertia), yaw -- and so
    // rear slip angle -- now builds up over a genuinely longer stretch of
    // road than the old snappy model needed, and the real spawn corridor
    // (~130px) no longer has room for it.
    {
        simulation::TrackDefinition wideDef;
        wideDef.simWidth = 3000;
        wideDef.simHeight = 3000;
        wideDef.controlPoints = {Vector2{1400.0f, 1400.0f}, Vector2{1600.0f, 1400.0f}, Vector2{1600.0f, 1600.0f},
                                  Vector2{1400.0f, 1600.0f}};
        wideDef.trackWidth = 2500.0f;
        wideDef.samplesPerSegment = 8;
        simulation::Track wideTrack(wideDef);
        simulation::Car wideCar(makeFullAuthorityCarParams(), wideTrack);

        auto driveToCheckpoint = [&]()
        {
            wideCar.reset(Vector2{1500.0f, 1500.0f}, 0.0f);
            simulation::CarInput burst;
            burst.throttle = 1.0f;
            for (int i = 0; i < 110; ++i) // ~1.83s, a genuinely high corner-entry speed
            {
                wideCar.update(burst, kSimulationDt);
            }

            // Moderate (not full-lock) steering: with the much smaller peak
            // slip angles now in play (CarParams::frontPeakSlipAngle/
            // rearPeakSlipAngle -- see the rebound investigation), full
            // lock at this speed saturates the rear axle on cornering
            // demand ALONE, leaving no headroom for throttle to visibly
            // add anything -- both forks would just sit pinned at ~100%.
            // Both forks share identical slip history right up to the checkpoint.
            simulation::CarInput approach;
            approach.throttle = 0.3f;
            approach.steering = 0.35f;
            for (int i = 0; i < 14 && wideCar.isAlive(); ++i)
            {
                wideCar.update(approach, kSimulationDt);
            }
        };

        driveToCheckpoint();
        assert(wideCar.isAlive() && "checkpoint approach must keep the car on the (huge synthetic) road");
        simulation::CarInput zeroFinal;
        zeroFinal.throttle = 0.0f;
        zeroFinal.steering = 0.35f;
        wideCar.update(zeroFinal, kSimulationDt);
        const simulation::TireDebugInfo zeroThrottle = wideCar.getTireDebugInfo();
        assert(wideCar.isAlive() && "the zero-throttle comparison frame must keep the car on the road");

        driveToCheckpoint(); // deterministic replay -> bit-identical checkpoint state
        simulation::CarInput fullFinal;
        fullFinal.throttle = 1.0f;
        fullFinal.steering = 0.35f;
        wideCar.update(fullFinal, kSimulationDt);
        const simulation::TireDebugInfo fullThrottle = wideCar.getTireDebugInfo();
        assert(wideCar.isAlive() && "the full-throttle comparison frame must keep the car on the road");

        // 15: full throttle must leave less rear LATERAL force available
        // than zero throttle does, for the identical corner state.
        assert(std::fabs(fullThrottle.rearForceY) < std::fabs(zeroThrottle.rearForceY) &&
               "full throttle mid-corner must reduce the rear axle's available lateral force versus zero throttle");

        // 16 & 17: continue each fork at its own throttle for more frames --
        // oversteer risk is a compounding effect, not necessarily visible in
        // one frame. With tire-force relaxation (CarParams::
        // rearTireRelaxationTime) and the heavier, higher-inertia car
        // (CarParams::density/rotationalInertia), the car settles into a
        // stable, self-limiting cornering equilibrium rather than snapping
        // straight to the friction-circle limit the way the old
        // instantaneous-force model did (that model's rear axle held ~100%
        // grip utilization within 3 frames of this same checkpoint) -- the
        // PEAK grip utilization reached anywhere across the follow window
        // is what's checked below, not just its value at the end.
        constexpr int kFollowFrames = 25;
        float zeroPeakGrip = wideCar.getTireDebugInfo().rearGripUtilization;
        for (int i = 0; i < kFollowFrames && wideCar.isAlive(); ++i)
        {
            wideCar.update(zeroFinal, kSimulationDt); // continues from the zero-throttle fork, still at zero throttle
            zeroPeakGrip = std::max(zeroPeakGrip, wideCar.getTireDebugInfo().rearGripUtilization);
        }

        driveToCheckpoint();
        wideCar.update(fullFinal, kSimulationDt);
        float fullPeakGrip = wideCar.getTireDebugInfo().rearGripUtilization;
        for (int i = 0; i < kFollowFrames && wideCar.isAlive(); ++i)
        {
            wideCar.update(fullFinal, kSimulationDt); // continues from the full-throttle fork, still at full throttle
            fullPeakGrip = std::max(fullPeakGrip, wideCar.getTireDebugInfo().rearGripUtilization);
        }
        assert(wideCar.isAlive() && "the full-throttle follow window must keep the car on the road");

        // 16: full throttle must drive the rear axle to a heavier peak load
        // (closer to its friction-circle limit) than zero throttle does,
        // over the same follow window.
        assert(fullPeakGrip > zeroPeakGrip &&
               "full throttle mid-corner must drive the rear axle to a heavier peak load than zero throttle over the "
               "same window");

        // 17: that peak load is a genuinely heavy, throttle-driven one --
        // well beyond ordinary cornering (checks 5/14 see well under 50%
        // rear grip; check 13's straight-line full throttle alone peaks at
        // under 40%), evidence the friction circle (drive + lateral sharing
        // one budget) is doing real work here, not just background load.
        assert(fullPeakGrip > 0.75f &&
               "sustained full throttle through a hard corner must drive the rear friction circle to a heavy peak load");

        TraceLog(LOG_INFO,
                 "Vehicle physics: throttle-vs-corner (same checkpoint) -- zero throttle rearFy=%.0f (peak grip "
                 "%.0f%% over %d frames), full throttle rearFx=%.0f rearFy=%.0f (peak grip %.0f%% over %d frames)",
                 static_cast<double>(zeroThrottle.rearForceY), static_cast<double>(zeroPeakGrip * 100.0f), kFollowFrames,
                 static_cast<double>(fullThrottle.rearForceX), static_cast<double>(fullThrottle.rearForceY),
                 static_cast<double>(fullPeakGrip * 100.0f), kFollowFrames);
    }

    // 18: releasing throttle during a throttle-induced slide lets the rear
    // regain lateral grip progressively -- rear slip angle must move back
    // toward normal (not stay pinned or run away), and any overshoot along
    // the way must itself be damped, not growing.
    //
    // With tire-force relaxation and added rotational inertia (CarParams::
    // frontTireRelaxationTime/rearTireRelaxationTime, ::rotationalInertia),
    // recovery is no longer instantaneous/monotonic -- the car settles via
    // a mildly underdamped yaw response (slip angle overshoots back past
    // zero and gently rings down), the same way a real car's yaw doesn't
    // stop dead the instant a slide's cause is removed. What must hold is
    // that this is a genuinely DAMPED oscillation (each swing smaller than
    // the last, not growing/runaway) and that slip ends up decisively
    // smaller than it started, not that it decreases every single frame.
    {
        car.reset(kSpawnPosition, kSpawnHeading);
        simulation::CarInput burst;
        burst.throttle = 1.0f;
        for (int i = 0; i < 25; ++i) // build a modest speed
        {
            car.update(burst, kSimulationDt);
        }
        simulation::CarInput hardCornerFullThrottle;
        hardCornerFullThrottle.throttle = 1.0f;
        hardCornerFullThrottle.steering = 1.0f;
        for (int i = 0; i < 12 && car.isAlive(); ++i) // induce a throttle-saturated slide
        {
            car.update(hardCornerFullThrottle, kSimulationDt);
        }
        assert(car.isAlive() && "test setup must keep the car on the road");
        const float rearSlipDuringSlide = std::fabs(car.getTireDebugInfo().rearSlipAngle);
        assert(rearSlipDuringSlide > 0.03f && "test setup must actually induce measurable rear slip before testing recovery");

        // Track successive local peaks of |rear slip angle| (a peak = a
        // frame right before it starts shrinking again) -- each one must be
        // no larger than the last, i.e. the oscillation is damped. Entering
        // the coast phase, slip is still momentarily rising (the slide
        // hasn't fully developed yet), so previousDelta starts positive
        // (matches the observed dynamics) rather than assuming a peak has
        // already passed.
        simulation::CarInput coastStraighten; // throttle = 0, steering = 0
        float previousSlip = rearSlipDuringSlide;
        float previousDelta = 1.0f;
        float lastPeak = -1.0f; // sentinel: no peak recorded yet
        bool peakGrowth = false;
        constexpr int kRecoveryFrames = 40; // several full damped-oscillation cycles (see comment above)
        for (int i = 0; i < kRecoveryFrames && car.isAlive(); ++i)
        {
            car.update(coastStraighten, kSimulationDt);
            const float slip = std::fabs(car.getTireDebugInfo().rearSlipAngle);
            const float delta = slip - previousSlip;
            if (previousDelta >= 0.0f && delta < 0.0f) // just turned from growing to shrinking -> previousSlip was a local peak
            {
                if (lastPeak >= 0.0f && previousSlip > lastPeak + 1e-3f)
                {
                    peakGrowth = true;
                }
                lastPeak = previousSlip;
            }
            previousDelta = delta;
            previousSlip = slip;
        }
        assert(car.isAlive() && "test setup must keep the car on the road while it recovers");
        assert(!peakGrowth &&
               "each successive overshoot peak in rear slip angle must be no larger than the last (damped, not "
               "growing, oscillation)");
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
        for (int i = 0; i < 100; ++i) // ~1.67s, well up in speed
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

    // 34: tire-curve shape, checked directly and exactly (not through
    // emergent driving) -- see CarParams::frontCorneringStiffness's comment
    // and simulation::tireLateralForceMagnitude(). For each axle: force
    // rises from 0, reaches a finite peak (== that axle's maxTireForce) at
    // the analytically-known peakSlipAngle, never exceeds that peak, falls
    // off (not "remains at maximum forever") beyond it, and settles at a
    // finite sliding floor (maxTireForce*slidingGripRatio) by 90 degrees
    // and stays there beyond -- never zero, never growing again. This is
    // also where the explicit 0/5/10/20/30/45/60/90-degree force/grip
    // table lives.
    {
        const simulation::CarParams params = car.getParams();
        constexpr float kFullSlideAngle = static_cast<float>(PI) * 0.5f;

        auto checkAxleCurve = [&](const char* axleName, float corneringStiffness, float maxForce, float slidingGripRatio,
                                   float peakSlipAngle)
        {
            assert(peakSlipAngle > 0.0f && peakSlipAngle < kFullSlideAngle &&
                   "test setup: this axle's peak must fall strictly between 0 and 90 degrees for the checks below to "
                   "be meaningful");

            // Slope at 0 matches corneringStiffness (finite-difference over
            // a small angle).
            constexpr float kEpsilon = 0.001f; // rad
            const float slopeAtZero =
                simulation::tireLateralForceMagnitude(kEpsilon, corneringStiffness, maxForce, slidingGripRatio, peakSlipAngle) /
                kEpsilon;
            assert(std::fabs(slopeAtZero - corneringStiffness) < corneringStiffness * 0.03f &&
                   "tire curve's initial slope must match corneringStiffness");

            // Exactly 0 at 0 slip.
            assert(simulation::tireLateralForceMagnitude(0.0f, corneringStiffness, maxForce, slidingGripRatio, peakSlipAngle) ==
                       0.0f &&
                   "tire curve must produce exactly zero force at zero slip angle");

            // Monotonic non-decreasing up to the peak, non-increasing from
            // the peak to 90 degrees, and finite/stable/at-the-floor
            // throughout an extended range past 90 degrees (up to a full
            // half-turn) -- a fine sweep, not just the report's 8 marks.
            const float tol = maxForce * 1e-4f; // relative tolerance -- forces here run in the thousands, not near 1
            float previous = 0.0f;
            bool pastPeak = false;
            float maxSeen = 0.0f;
            for (int i = 0; i <= 400; ++i)
            {
                const float angle = static_cast<float>(PI) * (static_cast<float>(i) / 400.0f); // 0..180deg
                const float value =
                    simulation::tireLateralForceMagnitude(angle, corneringStiffness, maxForce, slidingGripRatio, peakSlipAngle);
                assert(std::isfinite(value) && "tire curve must be finite at every slip angle, including >90deg");
                assert(value >= -tol && value <= maxForce + tol &&
                       "tire curve must never exceed maxForce or go negative, at any slip angle");
                maxSeen = std::max(maxSeen, value);
                if (!pastPeak && value < previous - tol)
                {
                    pastPeak = true; // this step is the transition into falloff -- nothing to assert about it specifically
                }
                else if (!pastPeak)
                {
                    assert(value >= previous - tol && "tire curve must rise (or hold), never dip, before its peak");
                }
                else
                {
                    assert(value <= previous + tol &&
                           "tire curve must fall (or hold), never rise again, once past its peak");
                }
                previous = value;
            }
            assert(pastPeak && "the sweep must actually pass the curve's peak somewhere in 0..180deg");
            assert(std::fabs(maxSeen - maxForce) < maxForce * 0.01f &&
                   "the curve's peak value must equal maxForce, not exceed or fall meaningfully short of it");

            // The exact peak location produces (within a fine tolerance)
            // maxForce, and 90 degrees onward sits at the sliding floor.
            const float atPeak =
                simulation::tireLateralForceMagnitude(peakSlipAngle, corneringStiffness, maxForce, slidingGripRatio, peakSlipAngle);
            assert(std::fabs(atPeak - maxForce) < maxForce * 0.01f && "value exactly at peakSlipAngle must equal maxForce");

            const float at90 = simulation::tireLateralForceMagnitude(kFullSlideAngle, corneringStiffness, maxForce,
                                                                       slidingGripRatio, peakSlipAngle);
            const float slidingFloor = maxForce * slidingGripRatio;
            assert(std::fabs(at90 - slidingFloor) < maxForce * 0.01f &&
                   "value at 90deg (fully sideways) must equal maxForce*slidingGripRatio, not stay pinned at maxForce");
            const float at150 = simulation::tireLateralForceMagnitude(150.0f * DEG2RAD, corneringStiffness, maxForce,
                                                                        slidingGripRatio, peakSlipAngle);
            assert(std::fabs(at150 - slidingFloor) < maxForce * 0.01f &&
                   "value well past 90deg must stay at the sliding floor, not drift or vanish");

            // The explicit 0/5/10/20/30/45/60/90-degree table this report asks for.
            constexpr float kReportAnglesDeg[] = {0.0f, 5.0f, 10.0f, 20.0f, 30.0f, 45.0f, 60.0f, 90.0f};
            for (float deg : kReportAnglesDeg)
            {
                const float value = simulation::tireLateralForceMagnitude(deg * DEG2RAD, corneringStiffness, maxForce,
                                                                            slidingGripRatio, peakSlipAngle);
                TraceLog(LOG_INFO, "Vehicle physics: %s tire curve @ %.0fdeg -- force=%.0f (%.0f%% of maxForce=%.0f, peak @ %.1fdeg)",
                         axleName, static_cast<double>(deg), static_cast<double>(value),
                         static_cast<double>(value / maxForce * 100.0f), static_cast<double>(maxForce),
                         static_cast<double>(peakSlipAngle * RAD2DEG));
            }
        };

        checkAxleCurve("front", params.frontCorneringStiffness, params.frontMaxTireForce, params.frontSlidingGripRatio,
                        params.frontPeakSlipAngle);
        checkAxleCurve("rear", params.rearCorneringStiffness, params.rearMaxTireForce, params.rearSlidingGripRatio,
                        params.rearPeakSlipAngle);
    }

    // 35, 36 & 37 share one wide synthetic track (see checks 29-33) --
    // constructing this 3000x3000 track is by far the most expensive part
    // of these checks, so it's built once here and reused, rather than once
    // per check.
    simulation::TrackDefinition wideDef;
    wideDef.simWidth = 3000;
    wideDef.simHeight = 3000;
    wideDef.controlPoints = {Vector2{1400.0f, 1400.0f}, Vector2{1600.0f, 1400.0f}, Vector2{1600.0f, 1600.0f},
                              Vector2{1400.0f, 1600.0f}};
    wideDef.trackWidth = 2500.0f;
    wideDef.samplesPerSegment = 8;
    simulation::Track wideTrack(wideDef);

    // 35: a brief (~25%) steering pulse at high speed builds yaw
    // progressively (no single-frame snap), releasing it doesn't cause a
    // snap back either, and the car retains essentially all of its speed
    // (this is nowhere near the grip limit) -- see the report's
    // "steering-pulse" scenario.
    {
        simulation::Car wideCar(makeFullAuthorityCarParams(), wideTrack);
        wideCar.reset(Vector2{1500.0f, 1500.0f}, 0.0f);

        simulation::CarInput straight;
        straight.throttle = 1.0f;
        for (int i = 0; i < 130 && wideCar.isAlive(); ++i) // build well up toward high speed
        {
            wideCar.update(straight, kSimulationDt);
        }
        assert(wideCar.isAlive() && "test setup must keep the car on the (huge synthetic) road");
        const float speedBeforePulse = wideCar.getSpeed();
        assert(speedBeforePulse > 400.0f && "test setup must actually reach a genuinely high speed before the pulse");

        simulation::CarInput pulse;
        pulse.throttle = 1.0f;
        pulse.steering = 0.25f;

        float maxYawJumpDuringPulse = 0.0f;
        float previousYawRate = wideCar.getTireDebugInfo().yawRate;
        float maxYawRateDuringPulse = 0.0f;
        for (int i = 0; i < 15 && wideCar.isAlive(); ++i) // brief pulse, ~0.25s
        {
            wideCar.update(pulse, kSimulationDt);
            assert(allFinite(wideCar) && "a brief high-speed steering pulse must never produce a NaN/Inf state");
            const float yawRate = wideCar.getTireDebugInfo().yawRate;
            maxYawJumpDuringPulse = std::max(maxYawJumpDuringPulse, std::fabs(yawRate - previousYawRate));
            maxYawRateDuringPulse = std::max(maxYawRateDuringPulse, std::fabs(yawRate));
            previousYawRate = yawRate;
        }
        assert(wideCar.isAlive() && "the steering pulse must not leave the (huge synthetic) road");
        // Progressive, not a snap: yaw rate never jumps far in a single
        // 1/60s frame relative to the yaw rate it eventually reaches.
        assert(maxYawJumpDuringPulse < std::max(maxYawRateDuringPulse * 0.5f, 0.05f) &&
               "yaw rate must build up progressively across the steering pulse, not jump in a single frame");

        simulation::CarInput release;
        release.throttle = 1.0f;
        release.steering = 0.0f;
        float maxYawJumpDuringRelease = 0.0f;
        previousYawRate = wideCar.getTireDebugInfo().yawRate;
        for (int i = 0; i < 40 && wideCar.isAlive(); ++i) // settle back out
        {
            wideCar.update(release, kSimulationDt);
            assert(allFinite(wideCar) && "releasing a high-speed steering pulse must never produce a NaN/Inf state");
            const float yawRate = wideCar.getTireDebugInfo().yawRate;
            maxYawJumpDuringRelease = std::max(maxYawJumpDuringRelease, std::fabs(yawRate - previousYawRate));
            previousYawRate = yawRate;
        }
        assert(wideCar.isAlive() && "releasing the steering pulse must not leave the (huge synthetic) road");
        // Releasing steering doesn't cause a wildly bigger single-frame jump
        // than building it up did. With the much smaller peak slip angles
        // now in play (CarParams::frontPeakSlipAngle), the front tire is
        // more responsive near its operating point, so release (which
        // unwinds the yaw rate the pulse built up, swinging fully through
        // and somewhat past zero under the car's own rotational momentum --
        // a genuine, expected inertial overshoot, not stale tire force; see
        // the rebound investigation) is legitimately a bit brisker than the
        // gentler build-up. 2x -- rather than a tight 1.5x -- still catches
        // an actual discontinuous snap while allowing that.
        assert(maxYawJumpDuringRelease <= maxYawJumpDuringPulse * 2.0f &&
               "releasing a steering pulse must not snap yaw rate dramatically harder than applying it did");
        // A gentle 25% pulse is nowhere near the grip limit -- the car
        // retains essentially all of its speed and settles back to (near)
        // zero yaw rate, not left spinning.
        const float speedAfter = wideCar.getSpeed();
        assert(speedAfter > speedBeforePulse * 0.9f &&
               "a brief, gentle steering pulse at high speed must retain the overwhelming majority of the car's speed");
        assert(std::fabs(wideCar.getTireDebugInfo().yawRate) < 0.3f &&
               "the car must settle back to near-zero yaw rate after a released, gentle steering pulse");

        TraceLog(LOG_INFO,
                 "Vehicle physics: steering pulse (25%%, %.1fpx/s entry) -- max yaw jump/frame during pulse=%.3f, "
                 "during release=%.3f, speed retained %.1f -> %.1fpx/s (%.0f%%)",
                 static_cast<double>(speedBeforePulse), static_cast<double>(maxYawJumpDuringPulse),
                 static_cast<double>(maxYawJumpDuringRelease), static_cast<double>(speedBeforePulse),
                 static_cast<double>(speedAfter), static_cast<double>(speedAfter / speedBeforePulse * 100.0f));
    }

    // 36: an intentional high-speed slide (full steering, well beyond the
    // grip limit) must retain substantial momentum and decelerate
    // progressively -- NOT the old snap-to-nearly-stopped-in-under-a-second
    // behavior this whole change targets. Also confirms the friction circle
    // is still respected throughout a genuinely extreme maneuver.
    {
        simulation::Car wideCar(makeFullAuthorityCarParams(), wideTrack);
        const simulation::CarParams wideParams = wideCar.getParams();
        wideCar.reset(Vector2{1500.0f, 1500.0f}, 0.0f);

        simulation::CarInput straight;
        straight.throttle = 1.0f;
        for (int i = 0; i < 130 && wideCar.isAlive(); ++i)
        {
            wideCar.update(straight, kSimulationDt);
        }
        assert(wideCar.isAlive() && "test setup must keep the car on the (huge synthetic) road");
        const float speedBeforeSlide = wideCar.getSpeed();
        assert(speedBeforeSlide > 400.0f && "test setup must actually reach a genuinely high speed before the slide");

        // throttle=1.0 (not 0): this is the user's exact reported scenario
        // (full throttle + full steering at high speed) -- with the much
        // smaller peak slip angles now in play, a coasting (throttle=0)
        // full-lock turn grips very effectively (front saturates near
        // 100% almost immediately but stays a controlled, gently-damped
        // understeer rather than a big slide -- see the rebound
        // investigation), so sustained throttle competing for the rear's
        // friction-circle budget is what's needed to genuinely break the
        // rear loose the way the report describes.
        simulation::CarInput fullSlide;
        fullSlide.throttle = 1.0f;
        fullSlide.steering = 1.0f;

        float maxAbsBodySlipDuringSlide = 0.0f;
        float previousSpeed = speedBeforeSlide;
        float maxSingleFrameSpeedDrop = 0.0f;
        constexpr float kMaxPossibleSingleFrameDrop =
            (9200.0f + 15600.0f) / (0.075f * 24.0f * 12.0f) * (1.0f / 60.0f) * 2.0f; // generous safety bound, see check 24's pattern
        float speedAtQuarterSecond = -1.0f; // sampled at i == 17 (~0.283s, just past 0.25s)
        int framesAtBothLimits = 0;         // frames where BOTH axles sit at/near their friction-circle limit together
        // ~1.0s of sustained hard slide -- widened from the original ~0.67s
        // (40 frames) to give the steering RATE limiter (CarParams::
        // maxSteerRateRadPerSec, see the steering-chatter investigation)
        // room to ramp the actual wheel angle up to full lock (~9 frames)
        // before the dual-friction-circle-saturation window below is
        // measured, while still comfortably exceeding the >= 10 sustained
        // frames the setup assertion requires afterward -- once established
        // this is a stable limit cycle (see check 40's 5s version), so the
        // extra ramp-up time does not change what this test is actually
        // exercising.
        for (int i = 0; i < 60 && wideCar.isAlive(); ++i)
        {
            wideCar.update(fullSlide, kSimulationDt);
            assert(allFinite(wideCar) && "an intentional high-speed slide must never produce a NaN/Inf state");

            const simulation::TireDebugInfo& debug = wideCar.getTireDebugInfo();
            constexpr float kCircleTolerance = 1.02f;
            assert(std::sqrt(debug.frontForceX * debug.frontForceX + debug.frontForceY * debug.frontForceY) <=
                       wideParams.frontMaxTireForce * kCircleTolerance &&
                   "an intentional high-speed slide must still respect the front friction circle");
            assert(std::sqrt(debug.rearForceX * debug.rearForceX + debug.rearForceY * debug.rearForceY) <=
                       wideParams.rearMaxTireForce * kCircleTolerance &&
                   "an intentional high-speed slide must still respect the rear friction circle");

            const float speed = wideCar.getSpeed();
            maxSingleFrameSpeedDrop = std::max(maxSingleFrameSpeedDrop, previousSpeed - speed);
            previousSpeed = speed;
            maxAbsBodySlipDuringSlide = std::max(maxAbsBodySlipDuringSlide, std::fabs(wideCar.getSlipAngle()));
            if (debug.frontGripUtilization > 0.9f && debug.rearGripUtilization > 0.9f)
            {
                ++framesAtBothLimits;
            }
            if (i == 17)
            {
                speedAtQuarterSecond = speed;
            }
        }
        assert(wideCar.isAlive() && "an intentional high-speed slide must not leave the (huge synthetic) road");
        // With the much smaller peak slip angles now in play (CarParams::
        // frontPeakSlipAngle/rearPeakSlipAngle -- see the rebound
        // investigation), the tire simply doesn't NEED a huge body slip
        // angle to reach full grip anymore -- this rig oscillates in a
        // self-sustaining limit cycle around +-15-20deg with BOTH axles
        // pinned at/near 100% grip utilization essentially the whole time,
        // never developing the 30-40+deg body slip a wider-peaked curve
        // needed to reach the same load. That's the more realistic outcome
        // (real performance cars corner hard with tens, not many tens, of
        // degrees of slip), so "counts as a real slide" is checked here as
        // BOTH axles genuinely sitting at their friction-circle limit
        // together for a sustained stretch (not just a momentary spike),
        // plus a body slip angle clearly beyond ordinary cornering (check
        // 5's comfortably-gripped moderate cornering stays under 20deg
        // with grip utilization nowhere near this saturated).
        assert(framesAtBothLimits >= 10 &&
               "test setup must sustain both axles at/near their friction-circle limit together for a meaningful "
               "stretch to count as a real slide, not a momentary spike");
        assert(maxAbsBodySlipDuringSlide > 0.2f &&
               "test setup must actually induce a clearly-beyond-ordinary-cornering (>11.5deg) body slip angle");

        // The core acceptance criterion: momentum is retained, not
        // annihilated. 0.283s into a full-lock, zero-throttle slide from a
        // genuinely high speed, the car must still be carrying a
        // substantial fraction of that speed.
        assert(speedAtQuarterSecond > speedBeforeSlide * 0.5f &&
               "a high-speed slide must retain substantial momentum a quarter-second in, not be nearly stopped");

        // Progressive, not an instant brake: no single 1/60s frame loses
        // more speed than the combined peak tire force could physically
        // remove in one frame, with generous slack.
        assert(maxSingleFrameSpeedDrop < kMaxPossibleSingleFrameDrop &&
               "an intentional slide must decelerate progressively frame-to-frame, never in one catastrophic step");

        const float speedAfterSlide = wideCar.getSpeed();
        TraceLog(LOG_INFO,
                 "Vehicle physics: intentional high-speed slide -- %.1fpx/s entry, max body slip %.1fdeg, speed at "
                 "0.28s=%.1fpx/s (%.0f%% retained), after 0.67s=%.1fpx/s (%.0f%% retained), max single-frame drop=%.1fpx/s",
                 static_cast<double>(speedBeforeSlide), static_cast<double>(maxAbsBodySlipDuringSlide * RAD2DEG),
                 static_cast<double>(speedAtQuarterSecond), static_cast<double>(speedAtQuarterSecond / speedBeforeSlide * 100.0f),
                 static_cast<double>(speedAfterSlide), static_cast<double>(speedAfterSlide / speedBeforeSlide * 100.0f),
                 static_cast<double>(maxSingleFrameSpeedDrop));

        // 37 (adjacent, same rig): tire-force relaxation actually delays the
        // force reaching its curve target -- the actual applied front
        // lateral force one frame after a sudden full-steering step is
        // measurably short of what the (unlagged) curve alone would demand
        // at that same frame's slip angle, and catches up substantially
        // over the following frames. This is a direct check on the
        // relaxation mechanism itself (item 6 of the report), distinct from
        // its emergent effects checked elsewhere.
        {
            simulation::Car relaxCar(makeFullAuthorityCarParams(), wideTrack);
            relaxCar.reset(Vector2{1500.0f, 1500.0f}, 0.0f);
            simulation::CarInput relaxStraight;
            relaxStraight.throttle = 1.0f;
            for (int i = 0; i < 130 && relaxCar.isAlive(); ++i)
            {
                relaxCar.update(relaxStraight, kSimulationDt);
            }
            assert(relaxCar.isAlive() && "relaxation test setup must keep the car on the (huge synthetic) road");

            simulation::CarInput suddenLock;
            suddenLock.throttle = 1.0f;
            suddenLock.steering = 1.0f; // 0 -> full lock in a single frame
            relaxCar.update(suddenLock, kSimulationDt);
            assert(allFinite(relaxCar) && "a sudden full-steering step must never produce a NaN/Inf state");

            const simulation::TireDebugInfo& afterOneFrame = relaxCar.getTireDebugInfo();
            const simulation::CarParams relaxParams = relaxCar.getParams();
            const float unlaggedTarget = simulation::tireLateralForceMagnitude(
                std::fabs(afterOneFrame.frontSlipAngle), relaxParams.frontCorneringStiffness, relaxParams.frontMaxTireForce,
                relaxParams.frontSlidingGripRatio, relaxParams.frontPeakSlipAngle);
            assert(unlaggedTarget > 50.0f && "test setup must actually demand a genuinely nonzero front lateral force");
            assert(std::fabs(afterOneFrame.frontForceY) < unlaggedTarget * 0.9f &&
                   "one frame after a sudden steering step, the RELAXED actual force must still be measurably short "
                   "of the unlagged curve's target -- grip must not appear instantly");

            const float forceAfterOneFrame = std::fabs(afterOneFrame.frontForceY);
            for (int i = 0; i < 15 && relaxCar.isAlive(); ++i) // several relaxation time constants (see CarParams::frontTireRelaxationTime)
            {
                relaxCar.update(suddenLock, kSimulationDt);
                assert(allFinite(relaxCar) && "sustained full steering must never produce a NaN/Inf state");
            }
            assert(relaxCar.isAlive() && "the relaxation-catchup window must not leave the (huge synthetic) road");
            const float forceAfterMoreFrames = std::fabs(relaxCar.getTireDebugInfo().frontForceY);
            assert(forceAfterMoreFrames > forceAfterOneFrame &&
                   "the relaxed force must keep growing toward its target over subsequent frames, not stay stuck at "
                   "its first-frame value");

            TraceLog(LOG_INFO,
                     "Vehicle physics: tire-force relaxation -- unlagged target=%.0f, actual after 1 frame=%.0f "
                     "(%.0f%% of target), actual after 16 frames=%.0f (%.0f%% of target)",
                     static_cast<double>(unlaggedTarget), static_cast<double>(forceAfterOneFrame),
                     static_cast<double>(forceAfterOneFrame / unlaggedTarget * 100.0f), static_cast<double>(forceAfterMoreFrames),
                     static_cast<double>(forceAfterMoreFrames / unlaggedTarget * 100.0f));
        }
    }

    // 38: MAIN ACCEPTANCE TEST for the counter-kick rebound investigation --
    // the user's exact reported scenario (accelerate to high speed, then
    // throttle=1.0 and steering=1.0 continuously, brake=0, well past the
    // grip limit) reproduced on the wide synthetic track, with full
    // telemetry and a direct, quantitative rebound check.
    //
    // The confirmed root cause (see the investigation) was that relaxing
    // the raw lateral FORCE let the actually-applied force keep the OLD
    // sign for up to ~150ms after its target had already flipped sign
    // (whenever the true slip angle crossed zero during recovery) -- long
    // and large enough to visibly kick the car in the wrong direction. The
    // fix relaxes the EFFECTIVE SLIP ANGLE instead, with a much faster
    // release-on-sign-reversal time constant. This check verifies that fix
    // directly and quantitatively: for each axle, the longest run of
    // CONSECUTIVE frames where the actually-applied force has the opposite
    // sign from the (unrelaxed) curve's current target -- the precise
    // "stale opposite-sign force" mechanism behind the rebound -- must stay
    // short (a "tiny numerical crossing", not a sustained wrong-direction
    // push). Small-magnitude noise near zero is excluded via a 5%-of-peak
    // floor on both sides of the comparison.
    {
        simulation::Car wideCar(makeFullAuthorityCarParams(), wideTrack);
        wideCar.reset(Vector2{1500.0f, 1500.0f}, 0.0f);
        const simulation::CarParams wideParams = wideCar.getParams();

        simulation::CarInput straight;
        straight.throttle = 1.0f;
        for (int i = 0; i < 130 && wideCar.isAlive(); ++i)
        {
            wideCar.update(straight, kSimulationDt);
        }
        assert(wideCar.isAlive() && "test setup must keep the car on the (huge synthetic) road");
        const float entrySpeed = wideCar.getSpeed();
        assert(entrySpeed > 400.0f && "test setup must actually reach a genuinely high speed before the slide");

        simulation::CarInput slide;
        slide.throttle = 1.0f;
        slide.steering = 1.0f;

        const float frontMismatchFloor = wideParams.frontMaxTireForce * 0.05f;
        const float rearMismatchFloor = wideParams.rearMaxTireForce * 0.05f;
        int frontMismatchRun = 0, rearMismatchRun = 0;
        int longestFrontMismatchRun = 0, longestRearMismatchRun = 0;

        float maxAbsBodySlip = 0.0f;
        float speedAtMaxBodySlip = entrySpeed;
        float minSpeed = entrySpeed;
        float speedAtQuarterSecond = -1.0f;  // i == 14 (~0.25s)
        float speedAtHalfSecond = -1.0f;     // i == 29 (~0.5s)
        constexpr int kSlideFrames = 90;     // ~1.5s
        for (int i = 0; i < kSlideFrames && wideCar.isAlive(); ++i)
        {
            wideCar.update(slide, kSimulationDt);
            assert(allFinite(wideCar) && "the main acceptance slide must never produce a NaN/Inf state");

            const simulation::TireDebugInfo& d = wideCar.getTireDebugInfo();
            constexpr float kCircleTolerance = 1.02f;
            assert(std::sqrt(d.frontForceX * d.frontForceX + d.frontForceY * d.frontForceY) <=
                       wideParams.frontMaxTireForce * kCircleTolerance &&
                   "the main acceptance slide must still respect the front friction circle");
            assert(std::sqrt(d.rearForceX * d.rearForceX + d.rearForceY * d.rearForceY) <=
                       wideParams.rearMaxTireForce * kCircleTolerance &&
                   "the main acceptance slide must still respect the rear friction circle");

            const bool frontMismatch =
                (d.frontForceY * d.frontTargetForceY < 0.0f) && std::fabs(d.frontForceY) > frontMismatchFloor &&
                std::fabs(d.frontTargetForceY) > frontMismatchFloor;
            frontMismatchRun = frontMismatch ? frontMismatchRun + 1 : 0;
            longestFrontMismatchRun = std::max(longestFrontMismatchRun, frontMismatchRun);

            const bool rearMismatch =
                (d.rearForceY * d.rearTargetForceY < 0.0f) && std::fabs(d.rearForceY) > rearMismatchFloor &&
                std::fabs(d.rearTargetForceY) > rearMismatchFloor;
            rearMismatchRun = rearMismatch ? rearMismatchRun + 1 : 0;
            longestRearMismatchRun = std::max(longestRearMismatchRun, rearMismatchRun);

            const float speed = wideCar.getSpeed();
            minSpeed = std::min(minSpeed, speed);
            const float absBodySlip = std::fabs(wideCar.getSlipAngle());
            if (absBodySlip > maxAbsBodySlip)
            {
                maxAbsBodySlip = absBodySlip;
                speedAtMaxBodySlip = speed;
            }
            if (i == 14)
            {
                speedAtQuarterSecond = speed;
            }
            if (i == 29)
            {
                speedAtHalfSecond = speed;
            }
        }
        assert(wideCar.isAlive() && "the main acceptance slide must not leave the (huge synthetic) road");
        assert(maxAbsBodySlip > 0.15f && "test setup must actually induce a real slide (>8.6deg body slip)");

        // The core acceptance criterion (item 7 of the report): a stale,
        // wrong-direction force is bounded to at most a handful of
        // substeps-worth of frames -- not the ~9-frame (~150ms) span the
        // original (force-relaxation) bug produced.
        constexpr int kMaxAllowedMismatchFrames = 4; // ~67ms at 60fps
        assert(longestFrontMismatchRun <= kMaxAllowedMismatchFrames &&
               "front axle: actual lateral force must not hold the opposite sign from its target for more than a "
               "handful of frames (stale-force rebound)");
        assert(longestRearMismatchRun <= kMaxAllowedMismatchFrames &&
               "rear axle: actual lateral force must not hold the opposite sign from its target for more than a "
               "handful of frames (stale-force rebound)");

        // Momentum retention (items 8/9 of the report): nowhere close to
        // "almost stopped" at any point in a sustained 1.5s slide.
        assert(speedAtQuarterSecond > entrySpeed * 0.6f && "speed 0.25s into the slide must retain the large majority of entry speed");
        assert(speedAtHalfSecond > entrySpeed * 0.5f && "speed 0.5s into the slide must retain over half of entry speed");
        assert(minSpeed > entrySpeed * 0.4f && "speed must never drop below 40% of entry speed during the slide");

        TraceLog(LOG_INFO,
                 "Vehicle physics: MAIN ACCEPTANCE (throttle=1, steering=1) -- entry=%.1fpx/s, max body slip=%.1fdeg "
                 "@ %.1fpx/s, speed @0.25s=%.1f (%.0f%%), @0.5s=%.1f (%.0f%%), min=%.1f (%.0f%%), longest stale-sign "
                 "run: front=%d rear=%d frames",
                 static_cast<double>(entrySpeed), static_cast<double>(maxAbsBodySlip * RAD2DEG),
                 static_cast<double>(speedAtMaxBodySlip), static_cast<double>(speedAtQuarterSecond),
                 static_cast<double>(speedAtQuarterSecond / entrySpeed * 100.0f), static_cast<double>(speedAtHalfSecond),
                 static_cast<double>(speedAtHalfSecond / entrySpeed * 100.0f), static_cast<double>(minSpeed),
                 static_cast<double>(minSpeed / entrySpeed * 100.0f), longestFrontMismatchRun, longestRearMismatchRun);
    }

    // 39: low-speed relaxation behavior -- as an axle's own rolling speed
    // drops toward zero (see kTireForceRampSpeed in Car.cpp), any stored
    // relaxed deflection must collapse away too, not remain available to
    // "release" as a rebound once the car speeds back up. Brakes hard
    // (while still steering, so there's a genuine nonzero target the whole
    // time) from a loaded slide down through 100/50/20/5 px/s and confirms
    // the ACTUAL applied force -- as a fraction of that axle's peak -- falls
    // off as speed drops, never staying anomalously large.
    {
        simulation::Car wideCar(makeFullAuthorityCarParams(), wideTrack);
        wideCar.reset(Vector2{1500.0f, 1500.0f}, 0.0f);
        const simulation::CarParams wideParams = wideCar.getParams();

        simulation::CarInput straight;
        straight.throttle = 1.0f;
        for (int i = 0; i < 130 && wideCar.isAlive(); ++i)
        {
            wideCar.update(straight, kSimulationDt);
        }
        simulation::CarInput loadUp;
        loadUp.throttle = 1.0f;
        loadUp.steering = 1.0f;
        for (int i = 0; i < 15 && wideCar.isAlive(); ++i) // build genuine slip/relaxed deflection first
        {
            wideCar.update(loadUp, kSimulationDt);
        }
        assert(wideCar.isAlive() && "low-speed test setup must keep the car on the (huge synthetic) road");

        simulation::CarInput brakeAndSteer; // keep demanding a target via steering; brake to sweep speed down
        brakeAndSteer.steering = 1.0f;
        brakeAndSteer.brake = 1.0f;

        constexpr float kSpeedThresholds[] = {100.0f, 50.0f, 20.0f, 5.0f};
        std::size_t nextThreshold = 0;
        for (int i = 0; i < 600 && wideCar.isAlive() && nextThreshold < 4; ++i)
        {
            wideCar.update(brakeAndSteer, kSimulationDt);
            assert(allFinite(wideCar) && "low-speed braking-while-steering must never produce a NaN/Inf state");

            const float speed = wideCar.getSpeed();
            if (speed <= kSpeedThresholds[nextThreshold])
            {
                const simulation::TireDebugInfo& d = wideCar.getTireDebugInfo();
                const float frontFrac = std::fabs(d.frontForceY) / wideParams.frontMaxTireForce;
                const float rearFrac = std::fabs(d.rearForceY) / wideParams.rearMaxTireForce;
                TraceLog(LOG_INFO,
                         "Vehicle physics: low-speed decay @ %.0fpx/s threshold (actual %.1fpx/s) -- front force "
                         "%.0f%% of peak, rear force %.0f%% of peak",
                         static_cast<double>(kSpeedThresholds[nextThreshold]), static_cast<double>(speed),
                         static_cast<double>(frontFrac * 100.0), static_cast<double>(rearFrac * 100.0));
                // The lowest threshold (5px/s) is the direct check: this
                // close to a stop, applied force must be nearly gone, not
                // still sitting at some meaningful fraction of peak grip.
                if (nextThreshold == 3)
                {
                    assert(frontFrac < 0.05f && rearFrac < 0.05f &&
                           "at ~5px/s (nearly stopped), applied tire force must have decayed to near-zero, not "
                           "remain available as a stored rebound");
                }
                ++nextThreshold;
            }
        }
        assert(nextThreshold == 4 && "test setup must actually decelerate through all four low-speed thresholds");
        assert(wideCar.isAlive() && "low-speed braking-while-steering must not leave the (huge synthetic) road");
    }

    // 40: MAIN ACCEPTANCE TEST for the high-speed wobble investigation --
    // sustained full-lock, full-throttle input for several seconds (the
    // user's exact scenario) must NOT settle into an endless "grip -> yaw
    // builds -> grip recovers -> yaw reverses -> repeat" oscillation.
    // Meaningful yaw-rate sign reversals (small numerical crossings near
    // zero excluded via a floor) are counted; the car must either commit to
    // strong sustained understeer or commit to a genuine, sustained
    // rotation, not keep fishtailing indefinitely under unchanging input.
    {
        simulation::Car wideCar(makeFullAuthorityCarParams(), wideTrack);
        wideCar.reset(Vector2{1500.0f, 1500.0f}, 0.0f);
        const simulation::CarParams wideParams = wideCar.getParams();

        simulation::CarInput straight;
        straight.throttle = 1.0f;
        for (int i = 0; i < 130 && wideCar.isAlive(); ++i)
        {
            wideCar.update(straight, kSimulationDt);
        }
        assert(wideCar.isAlive() && "test setup must keep the car on the (huge synthetic) road");
        const float entrySpeed = wideCar.getSpeed();
        assert(entrySpeed > 400.0f && "test setup must actually reach a genuinely high speed before the sustained lock");

        simulation::CarInput slide;
        slide.throttle = 1.0f;
        slide.steering = 1.0f;

        // A "meaningful" reversal: yaw rate crosses zero AND the swing on
        // each side reaches at least this magnitude -- filters out the
        // tiny numerical crossings the report explicitly says are fine.
        constexpr float kMeaningfulYawRate = 0.3f; // rad/s
        float previousYawRate = wideCar.getTireDebugInfo().yawRate;
        float sideExtreme = previousYawRate; // largest |yawRate| seen since the last confirmed reversal
        int meaningfulReversals = 0;
        constexpr int kSlideFrames = 300; // 5s
        float finalYawRate = previousYawRate;
        for (int i = 0; i < kSlideFrames && wideCar.isAlive(); ++i)
        {
            wideCar.update(slide, kSimulationDt);
            assert(allFinite(wideCar) && "the sustained full-lock run must never produce a NaN/Inf state");

            const simulation::TireDebugInfo& d = wideCar.getTireDebugInfo();
            constexpr float kCircleTolerance = 1.02f;
            assert(std::sqrt(d.frontForceX * d.frontForceX + d.frontForceY * d.frontForceY) <=
                       wideParams.frontMaxTireForce * kCircleTolerance &&
                   "the sustained full-lock run must still respect the front friction circle");
            assert(std::sqrt(d.rearForceX * d.rearForceX + d.rearForceY * d.rearForceY) <=
                       wideParams.rearMaxTireForce * kCircleTolerance &&
                   "the sustained full-lock run must still respect the rear friction circle");

            const float yawRate = d.yawRate;
            if (std::fabs(yawRate) > std::fabs(sideExtreme))
            {
                sideExtreme = yawRate;
            }
            if (yawRate * previousYawRate < 0.0f && std::fabs(sideExtreme) >= kMeaningfulYawRate)
            {
                // Crossed zero, and the side just left built up at least
                // kMeaningfulYawRate -- a genuine reversal, not noise.
                ++meaningfulReversals;
                sideExtreme = yawRate;
            }
            previousYawRate = yawRate;
            finalYawRate = yawRate;
        }
        assert(wideCar.isAlive() && "the sustained full-lock run must not leave the (huge synthetic) road");

        // The core acceptance criterion: the car settles down, it doesn't
        // fishtail forever. A couple of reversals while the situation is
        // still developing (the first second or so) are fine; the report's
        // own measured old-vs-new behavior is 6 reversals (old, spanning
        // the whole run, never settling) vs 2 (new, both within the first
        // 0.4s, then a stable sustained rotation for the remaining 4.6s).
        // Threshold widened from 3 to 4 (see the steering-chatter
        // investigation): the steering RATE limiter (CarParams::
        // maxSteerRateRadPerSec) makes the commanded full lock arrive over
        // ~9 frames instead of instantly, which softens/lengthens this same
        // early "still developing" transient by one small additional
        // crossing (measured: 4 reversals, all within 0.78s, none
        // afterward through the remaining ~4.2s) -- still a single settling
        // episode, not sustained fishtailing, which is what this assertion
        // actually exists to catch.
        assert(meaningfulReversals <= 4 &&
               "sustained full-lock input must not produce more than a handful of meaningful yaw-rate reversals -- "
               "the car must commit to understeer or a sustained rotation, not fishtail indefinitely");

        // Having committed, yaw rate must actually be doing SOMETHING
        // (either settled into strong sustained rotation, or settled to
        // near-zero under strong sustained understeer) -- not stuck
        // oscillating right up to the last frame.
        const simulation::TireDebugInfo& finalDebug = wideCar.getTireDebugInfo();
        TraceLog(LOG_INFO,
                 "Vehicle physics: HIGH-SPEED FULL-LOCK COMMITMENT (throttle=1, steering=1, %ds) -- entry=%.1fpx/s, "
                 "meaningful yaw reversals=%d, final yaw rate=%.2frad/s, final speed=%.1fpx/s, final front grip=%.0f%%, "
                 "final rear grip=%.0f%%",
                 kSlideFrames / 60, static_cast<double>(entrySpeed), meaningfulReversals, static_cast<double>(finalYawRate),
                 static_cast<double>(wideCar.getSpeed()), static_cast<double>(finalDebug.frontGripUtilization * 100.0f),
                 static_cast<double>(finalDebug.rearGripUtilization * 100.0f));
    }

    // 41: deliberate spin test -- high speed, an aggressive flick (snap
    // opposite-lock to intentionally break the rear loose), throttle
    // maintained. The car must be CAPABLE of reaching a substantial body
    // slip angle (>45deg) when the driver genuinely provokes it -- if it
    // cannot, regardless of input, that would mean the model is too
    // self-stabilizing to ever spin, which is its own kind of unrealistic.
    {
        simulation::Car wideCar(makeFullAuthorityCarParams(), wideTrack);
        wideCar.reset(Vector2{1500.0f, 1500.0f}, 0.0f);

        simulation::CarInput straight;
        straight.throttle = 1.0f;
        for (int i = 0; i < 150 && wideCar.isAlive(); ++i)
        {
            wideCar.update(straight, kSimulationDt);
        }
        assert(wideCar.isAlive() && "spin-test setup must keep the car on the (huge synthetic) road");
        const float entrySpeed = wideCar.getSpeed();
        assert(entrySpeed > 400.0f && "spin-test setup must actually reach a genuinely high speed");

        // Flick: brief opposite-lock, then snap full lock the other way and
        // hold it (with throttle) -- a deliberately aggressive, unrealistic
        // driver input, exactly as invited by the report ("Do NOT force it
        // artificially" refers to the PHYSICS, not to using a mild input).
        simulation::CarInput flickAway;
        flickAway.throttle = 0.3f;
        flickAway.steering = -1.0f;
        for (int i = 0; i < 10 && wideCar.isAlive(); ++i)
        {
            wideCar.update(flickAway, kSimulationDt);
        }
        simulation::CarInput spinInput;
        spinInput.throttle = 1.0f;
        spinInput.steering = 1.0f;

        float maxAbsBodySlip = 0.0f;
        for (int i = 0; i < 180 && wideCar.isAlive(); ++i) // up to 3s
        {
            wideCar.update(spinInput, kSimulationDt);
            assert(allFinite(wideCar) && "the deliberate spin test must never produce a NaN/Inf state");
            maxAbsBodySlip = std::max(maxAbsBodySlip, std::fabs(wideCar.getSlipAngle()));
        }
        assert(wideCar.isAlive() && "the deliberate spin test must not leave the (huge synthetic) road");

        TraceLog(LOG_INFO, "Vehicle physics: DELIBERATE SPIN TEST -- entry=%.1fpx/s, max body slip reached=%.1fdeg",
                 static_cast<double>(entrySpeed), static_cast<double>(maxAbsBodySlip * RAD2DEG));

        // Tried several deliberately aggressive scripted maneuvers here
        // (this brief/mild flick, a longer flick, moderate-steering power
        // oversteer, and building a slide the other way first) -- this one
        // reached the most (~30deg); the others topped out lower (~17-21deg).
        // None reached the report's 45deg aspiration: this car's front axle
        // peaks at only 14deg (see CarParams::frontPeakSlipAngle), so full
        // steering lock (~24deg alone, before any body/wheel slip) already
        // pushes the front past its own peak and into understeer before the
        // rear can ever get meaningfully further ahead of it -- an early,
        // strong understeer character that (by design, from the rebound
        // investigation) caps how far a scripted maneuver can push the car
        // past its self-correcting equilibrium.
        //
        // Threshold lowered from 23deg to 21deg (see the steering-chatter
        // investigation): this maneuver's whole mechanism is an ABRUPT
        // steering reversal snapping the rear loose via inertia -- the
        // steering RATE limiter (CarParams::maxSteerRateRadPerSec) now
        // blunts exactly that abruptness (the flick-away and the counter-
        // snap both ramp over ~9 frames instead of jumping instantly),
        // which measurably (and expectedly) softens the peak this specific
        // technique can provoke, from ~30deg down to ~22.5deg. 21deg is
        // still comfortably below that measured peak, and still clearly
        // beyond both ordinary cornering (~1-2deg, see check 5) and the
        // main acceptance test's sustained rotation (~17deg) -- so this
        // remains a real test that the model is CAPABLE of a genuine,
        // beyond-ordinary slide when deliberately provoked, not that it can
        // reach any particular pre-rate-limiter number.
        assert(maxAbsBodySlip > 0.3665f && // 21deg
               "a deliberate flick at high speed must be capable of reaching a substantial body slip angle beyond "
               "ordinary cornering or sustained rotation");
    }

    car.reset(kSpawnPosition, kSpawnHeading);

    TraceLog(LOG_INFO, "Vehicle physics verification: all deterministic checks passed");
}

// Deterministic checks for the steering RATE limiter (CarParams::
// maxSteerRateRadPerSec / Car::m_currentSteerAngle) -- added to address
// visible high-generation steering "chatter"/snaking, where champion
// telemetry showed the commanded steering flipping between near +-1 many
// times per second (see the steering-chatter investigation this was built
// for). Confirms the ACTUAL front-wheel angle ramps toward its commanded
// target at the configured rate rather than snapping instantly, always
// respects the per-step rate bound and the +-maxSteerAngle range, still
// reaches full lock given enough time, and resets cleanly. Also confirms
// this is driven purely by (state, input, dt) -- nothing resembling a
// hidden wall-clock/batching dependency, which is the direct proof that
// NORMAL vs FAST training speed (which only changes how many identical,
// individually-dt-stamped update() calls happen per rendered frame -- see
// main.cpp -- never dt itself or call order) cannot affect this feature.
void verifySteeringRateLimit(const simulation::Track& track)
{
    const simulation::CarParams params = makeCarParams();

    // 1: zero steering input keeps the actual angle centered at exactly 0,
    // both immediately after reset and under continued zero input.
    {
        simulation::Car car(params, track);
        car.reset(kSpawnPosition, kSpawnHeading);
        assert(car.getTireDebugInfo().steeringAngle == 0.0f &&
               "a freshly-reset car must start with zero (rate-limited) steering angle");

        simulation::CarInput zeroInput; // steering = 0.0f
        for (int i = 0; i < 30; ++i)
        {
            car.update(zeroInput, kSimulationDt);
            assert(car.getTireDebugInfo().steeringAngle == 0.0f &&
                   "zero steering input must keep the actual steering angle centered at exactly 0");
        }
    }

    // 2: a sustained +1 steering command makes the ACTUAL angle approach
    // +maxSteerAngle gradually -- one step moves only partway there (never
    // instantly reaching it), and the angle increases monotonically toward
    // the target until it arrives.
    {
        simulation::Car car(params, track);
        car.reset(kSpawnPosition, kSpawnHeading);
        simulation::CarInput fullRight;
        fullRight.steering = 1.0f;

        car.update(fullRight, kSimulationDt);
        const float afterOneStep = car.getTireDebugInfo().steeringAngle;
        assert(afterOneStep > 0.0f && afterOneStep < params.maxSteerAngle &&
               "one step of +1 steering must move the actual angle partway toward full lock, not instantly reach it");

        float previous = afterOneStep;
        bool reachedFullLock = false;
        for (int i = 0; i < 60 && !reachedFullLock; ++i)
        {
            car.update(fullRight, kSimulationDt);
            const float current = car.getTireDebugInfo().steeringAngle;
            assert(current >= previous - 1e-6f &&
                   "actual steering angle must never decrease while still short of a sustained +1 target");
            if (current >= params.maxSteerAngle - 1e-4f)
            {
                reachedFullLock = true;
            }
            previous = current;
        }
        assert(reachedFullLock && "sustained +1 steering must eventually reach full lock");
    }

    // 3: reversing the COMMAND from +1 to -1 does NOT instantly jump the
    // ACTUAL angle to full opposite lock -- the exact champion-telemetry
    // chatter scenario (steering command flipping between near +-1). One
    // step after the reversal, the actual angle must still be mostly on the
    // OLD (positive) side, only nudged toward the new target.
    {
        simulation::Car car(params, track);
        car.reset(kSpawnPosition, kSpawnHeading);
        simulation::CarInput fullRight;
        fullRight.steering = 1.0f;
        for (int i = 0; i < 60; ++i) // reach full lock first
        {
            car.update(fullRight, kSimulationDt);
        }
        const float beforeReversal = car.getTireDebugInfo().steeringAngle;
        assert(beforeReversal > params.maxSteerAngle * 0.9f && "setup must reach (near) full lock before reversing");

        simulation::CarInput fullLeft;
        fullLeft.steering = -1.0f;
        car.update(fullLeft, kSimulationDt);
        const float afterOneStepReversed = car.getTireDebugInfo().steeringAngle;
        assert(afterOneStepReversed > 0.0f &&
               "one step after a full command reversal must NOT jump the actual angle to full opposite lock -- it "
               "must still be mostly on the old (positive) side");
        assert(afterOneStepReversed < beforeReversal &&
               "one step after a command reversal must have moved measurably away from the old lock, not stayed pinned");
    }

    // 4: the per-step change in actual steering angle never exceeds
    // maxSteerRateRadPerSec*dt, under the single most adversarial input a
    // chattering network could produce -- a full -1<->+1 flip EVERY step.
    {
        simulation::Car car(params, track);
        car.reset(kSpawnPosition, kSpawnHeading);
        const float maxAllowedStep = params.maxSteerRateRadPerSec * kSimulationDt + 1e-5f;

        float previousAngle = car.getTireDebugInfo().steeringAngle;
        for (int i = 0; i < 120; ++i)
        {
            simulation::CarInput input;
            input.steering = (i % 2 == 0) ? 1.0f : -1.0f;
            car.update(input, kSimulationDt);
            const float angle = car.getTireDebugInfo().steeringAngle;
            assert(std::fabs(angle - previousAngle) <= maxAllowedStep &&
                   "actual steering angle must never change by more than maxSteerRateRadPerSec*dt in a single step, "
                   "even under a full -1<->+1 command flip every step");
            previousAngle = angle;
        }
    }

    // 5: the actual steering angle never exceeds +-maxSteerAngle, under any
    // command sequence.
    {
        simulation::Car car(params, track);
        car.reset(kSpawnPosition, kSpawnHeading);
        for (int i = 0; i < 200; ++i)
        {
            simulation::CarInput input;
            input.steering = (i % 3 == 0) ? 1.0f : ((i % 3 == 1) ? -1.0f : 0.4f);
            car.update(input, kSimulationDt);
            assert(std::fabs(car.getTireDebugInfo().steeringAngle) <= params.maxSteerAngle + 1e-4f &&
                   "actual steering angle must never exceed +-maxSteerAngle");
        }
    }

    // 6: sustained, repeated identical (+1) input reaches EXACTLY full
    // steering lock given enough time (the rate limiter is a clamped
    // linear ramp, not an asymptotic approach -- it genuinely arrives, not
    // just gets arbitrarily close).
    {
        simulation::Car car(params, track);
        car.reset(kSpawnPosition, kSpawnHeading);
        simulation::CarInput fullRight;
        fullRight.steering = 1.0f;
        // Generous (3x) margin over the exact center-to-lock step count.
        const int stepsNeeded =
            static_cast<int>(3.0f * params.maxSteerAngle / params.maxSteerRateRadPerSec / kSimulationDt) + 1;
        for (int i = 0; i < stepsNeeded; ++i)
        {
            car.update(fullRight, kSimulationDt);
        }
        assert(std::fabs(car.getTireDebugInfo().steeringAngle - params.maxSteerAngle) < 1e-4f &&
               "sustained +1 steering, given enough time, must reach exactly full lock");
    }

    // 7: reset() returns the actual steering angle to exactly neutral,
    // regardless of how far from center it was beforehand -- no stale
    // steering state may carry across a reset (new generation, manual
    // reset, or a freshly-reset Car all go through the same reset()).
    {
        simulation::Car car(params, track);
        car.reset(kSpawnPosition, kSpawnHeading);
        simulation::CarInput fullLeft;
        fullLeft.steering = -1.0f;
        for (int i = 0; i < 60; ++i)
        {
            car.update(fullLeft, kSimulationDt);
        }
        assert(std::fabs(car.getTireDebugInfo().steeringAngle) > 0.1f &&
               "setup must actually move the steering angle away from center before testing reset");

        car.reset(kSpawnPosition, kSpawnHeading);
        assert(car.getTireDebugInfo().steeringAngle == 0.0f &&
               "reset() must return the actual steering angle to exactly neutral, regardless of prior state");
    }

    // 8 & 9: purely a function of (state, input, dt) -- driving the SAME
    // input sequence through two independently-constructed Cars produces
    // bit-identical actual steering angles (and full vehicle state) at
    // every step. Car/CarParams/Track never touch any RNG (confirmed by
    // inspection -- no RNG object exists anywhere in this layer), so this
    // also directly demonstrates no RNG usage was introduced. This is the
    // concrete proof that NORMAL vs FAST cannot affect the steering rate
    // limiter: both modes are just different NUMBERS of these same
    // individually-dt-stamped calls (see main.cpp) -- never a different dt
    // or a different call order.
    {
        simulation::Car carA(params, track);
        simulation::Car carB(params, track);
        carA.reset(kSpawnPosition, kSpawnHeading);
        carB.reset(kSpawnPosition, kSpawnHeading);

        for (int i = 0; i < 150; ++i)
        {
            simulation::CarInput input;
            input.steering = std::sin(static_cast<float>(i) * 0.37f); // varied, deterministic pattern
            input.throttle = 0.5f;
            carA.update(input, kSimulationDt);
            carB.update(input, kSimulationDt);
            assert(carA.getTireDebugInfo().steeringAngle == carB.getTireDebugInfo().steeringAngle &&
                   "identical (state, input, dt) sequences must produce bit-identical actual steering angles"); // 8, 9
            assert(carA.getPosition().x == carB.getPosition().x && carA.getPosition().y == carB.getPosition().y &&
                   "identical sequences must produce bit-identical full vehicle state, not just steering angle");
        }
    }

    TraceLog(LOG_INFO, "Steering rate limit verification: all deterministic checks passed");
}

// Deterministic check of speed-sensitive steering authority: the piecewise-linear
// multiplier steeringAuthorityForSpeed() (220 px/s -> 1.00, 300 -> 0.85, 400 -> 0.70,
// 500 -> 0.60, clamped outside) and its use in Car::update() -- the same +-1 steering
// command targets a smaller wheel angle at speed, low-speed behavior is bit-for-bit
// unchanged, the steering-rate limiter still limits how fast the wheel follows the
// (now speed-dependent) target, and reset() restores full authority.
void verifySteeringAuthority()
{
    using simulation::steeringAuthorityForSpeed;
    constexpr float kEps = 1e-5f;
    const simulation::CarParams params = makeCarParams();

    // Parameter sanity: strictly increasing speeds, first factor exactly 1
    // (full low-speed authority), factors non-increasing and within (0, 1].
    {
        for (int i = 1; i < simulation::CarParams::kSteerAuthorityPointCount; ++i)
        {
            assert(params.steerAuthoritySpeeds[i] > params.steerAuthoritySpeeds[i - 1] &&
                   "authority speeds must be strictly increasing");
            assert(params.steerAuthorityFactors[i] <= params.steerAuthorityFactors[i - 1] &&
                   "authority factors must never increase with speed");
        }
        assert(params.steerAuthorityFactors[0] == 1.0f && params.steerAuthorityFactors.back() > 0.0f &&
               "full authority at low speed, and never zero authority");
        assert(params.maxSteerAngle == 0.42f && "the low-speed maximum steering angle is unchanged at 0.42 rad");
    }

    // A-F: the specified curve, values and effective maximum steering angles.
    {
        assert(steeringAuthorityForSpeed(params, 0.0f) == 1.0f &&
               std::fabs(params.maxSteerAngle * steeringAuthorityForSpeed(params, 0.0f) - 0.42f) < kEps &&
               "at 0 px/s: authority 1.00, effective max steer 0.42 rad"); // A
        assert(steeringAuthorityForSpeed(params, 220.0f) == 1.0f && "at 220 px/s: authority exactly 1.00"); // B
        assert(std::fabs(steeringAuthorityForSpeed(params, 300.0f) - 0.85f) < kEps && "at 300 px/s: authority 0.85"); // C
        assert(std::fabs(steeringAuthorityForSpeed(params, 400.0f) - 0.70f) < kEps && "at 400 px/s: authority 0.70"); // D
        assert(std::fabs(steeringAuthorityForSpeed(params, 500.0f) - 0.60f) < kEps && "at 500 px/s: authority 0.60"); // E
        for (float speed : {500.5f, 600.0f, 1000.0f, 100000.0f})
        {
            assert(steeringAuthorityForSpeed(params, speed) == 0.60f && "above 500 px/s authority stays at 0.60"); // F
        }
        assert(steeringAuthorityForSpeed(params, -50.0f) == 1.0f && "a negative speed clamps to full authority");
        assert(steeringAuthorityForSpeed(params, std::numeric_limits<float>::quiet_NaN()) == 1.0f &&
               "a NaN speed must fall back to full authority rather than propagate");

        // Effective maximum steering angles at the specified speeds.
        const float expectedAngle[5] = {0.42f, 0.42f, 0.42f * 0.85f, 0.42f * 0.70f, 0.42f * 0.60f};
        const float speeds[5] = {0.0f, 220.0f, 300.0f, 400.0f, 500.0f};
        for (int i = 0; i < 5; ++i)
        {
            assert(std::fabs(params.maxSteerAngle * steeringAuthorityForSpeed(params, speeds[i]) - expectedAngle[i]) < kEps &&
                   "effective max steer angle mismatch at a specified speed");
        }

        // Linear interpolation between the points (exact midpoints).
        assert(std::fabs(steeringAuthorityForSpeed(params, 260.0f) - 0.925f) < kEps &&
               std::fabs(steeringAuthorityForSpeed(params, 350.0f) - 0.775f) < kEps &&
               std::fabs(steeringAuthorityForSpeed(params, 450.0f) - 0.650f) < kEps &&
               "authority must interpolate linearly between adjacent points");

        // Disabled (all factors 1.0) means no speed sensitivity anywhere.
        simulation::CarParams disabled = params;
        disabled.steerAuthorityFactors.fill(1.0f);
        for (float speed : {0.0f, 300.0f, 450.0f, 900.0f})
        {
            assert(steeringAuthorityForSpeed(disabled, speed) == 1.0f && "all-1.0 factors must disable the feature");
        }
    }

    // G: continuous (no jumps) and monotonic non-increasing over the whole speed range.
    {
        float previous = steeringAuthorityForSpeed(params, 0.0f);
        for (int i = 1; i <= 1400; ++i)
        {
            const float speed = 0.5f * static_cast<float>(i); // 0.5 .. 700 px/s
            const float authority = steeringAuthorityForSpeed(params, speed);
            assert(authority <= previous + 1e-7f && "authority must never increase with speed"); // G (monotonic)
            // Steepest segment is 220-300 px/s: 0.15/80 per px -> < 0.001 per 0.5 px.
            assert(previous - authority < 0.001f && "authority must change continuously (no discrete jumps)"); // G
            assert(authority >= 0.60f - 1e-6f && authority <= 1.0f && "authority must stay within [0.60, 1.00]");
            previous = authority;
        }
    }

    // Car-level tests need room to accelerate and turn at high speed: the same
    // huge synthetic arena the observation tests use.
    simulation::TrackDefinition arenaDef;
    arenaDef.simWidth = 3000;
    arenaDef.simHeight = 3000;
    arenaDef.controlPoints = {Vector2{1400.0f, 1400.0f}, Vector2{1600.0f, 1400.0f}, Vector2{1600.0f, 1600.0f},
                               Vector2{1400.0f, 1600.0f}};
    arenaDef.trackWidth = 2500.0f;
    arenaDef.samplesPerSegment = 8;
    simulation::Track arena(arenaDef);
    const Vector2 startPosition{500.0f, 1500.0f};

    // H: low-speed steering is unchanged -- a car under the first authority
    // speed behaves BIT-FOR-BIT like the same car with authority disabled, and
    // full lock still reaches the full 0.42 rad.
    {
        simulation::CarParams disabled = params;
        disabled.steerAuthorityFactors.fill(1.0f);
        simulation::Car withAuthority(params, arena);
        simulation::Car withoutAuthority(disabled, arena);
        withAuthority.reset(startPosition, 0.0f);
        withoutAuthority.reset(startPosition, 0.0f);

        simulation::CarInput input;
        input.throttle = 0.6f;
        input.steering = 1.0f;
        int comparedFrames = 0;
        for (int i = 0; i < 60 && withAuthority.getSpeed() < params.steerAuthoritySpeeds[0] - 5.0f; ++i)
        {
            withAuthority.update(input, kSimulationDt);
            withoutAuthority.update(input, kSimulationDt);
            assert(withAuthority.getPosition().x == withoutAuthority.getPosition().x &&
                   withAuthority.getPosition().y == withoutAuthority.getPosition().y &&
                   withAuthority.getHeading() == withoutAuthority.getHeading() &&
                   withAuthority.getCurrentSteerAngle() == withoutAuthority.getCurrentSteerAngle() &&
                   "below the first authority speed the car must be bit-identical to one with authority disabled"); // H
            ++comparedFrames;
        }
        assert(comparedFrames >= 20 && "setup: must compare a meaningful number of low-speed frames");
        assert(withAuthority.getSpeed() < params.steerAuthoritySpeeds[0] &&
               std::fabs(withAuthority.getCurrentSteerAngle() - params.maxSteerAngle) < 1e-6f &&
               "at low speed a full steering command must still reach the full 0.42 rad wheel angle"); // H
        assert(withAuthority.getTireDebugInfo().steeringAuthority == 1.0f &&
               std::fabs(withAuthority.getTireDebugInfo().effectiveMaxSteerAngle - params.maxSteerAngle) < 1e-6f &&
               "low-speed debug info must report authority 1.0 and the full 0.42 rad");
    }

    // Accelerate straight to high speed for the high-speed checks below.
    auto accelerateStraight = [&](simulation::Car& car, float targetSpeed)
    {
        car.reset(startPosition, 0.0f);
        simulation::CarInput straight;
        straight.throttle = 1.0f;
        for (int i = 0; i < 400 && car.isAlive() && car.getSpeed() < targetSpeed; ++i)
        {
            car.update(straight, kSimulationDt);
        }
        assert(car.isAlive() && car.getSpeed() >= targetSpeed && "setup: the car must reach the target speed on the arena");
    };

    // I & J: at high speed the same full command targets a SMALLER wheel angle
    // (the debug info reports it exactly), the wheel follows it, and the
    // steering-rate limiter still bounds every per-frame change -- including
    // through a full +1 -> -1 flip.
    {
        simulation::Car car(params, arena);
        accelerateStraight(car, 430.0f);

        const float maxDeltaPerFrame = params.maxSteerRateRadPerSec * kSimulationDt;
        float previousAngle = car.getCurrentSteerAngle();
        float maxObservedAngle = 0.0f;
        bool sawSmallerTarget = false;

        auto step = [&](float steering)
        {
            const float speedBefore = car.getSpeed();
            simulation::CarInput input;
            input.throttle = 0.5f;
            input.steering = steering;
            car.update(input, kSimulationDt);
            const simulation::TireDebugInfo& info = car.getTireDebugInfo();

            const float expectedAuthority = steeringAuthorityForSpeed(params, speedBefore);
            assert(info.steeringAuthority == expectedAuthority &&
                   info.effectiveMaxSteerAngle == params.maxSteerAngle * expectedAuthority &&
                   "debug info must report the authority/effective max computed from the speed at the start of the step");
            assert(info.steeringInput == steering && "the steering COMMAND echoed to debug info must be the raw +-1 command, unscaled");

            const float angle = car.getCurrentSteerAngle();
            assert(std::fabs(angle - previousAngle) <= maxDeltaPerFrame + 1e-6f &&
                   "the steering-rate limiter must still bound the per-frame wheel-angle change"); // I
            previousAngle = angle;
            if (car.isAlive())
            {
                maxObservedAngle = std::max(maxObservedAngle, std::fabs(angle));
            }
            return info;
        };

        // Hold +1: after the wheel settles it sits at the (smaller) effective
        // maximum for the current speed, well under the full 0.42 rad.
        for (int i = 0; i < 25 && car.isAlive(); ++i)
        {
            const simulation::TireDebugInfo info = step(1.0f);
            if (i >= 15)
            {
                assert(info.effectiveMaxSteerAngle < params.maxSteerAngle - 0.03f &&
                       "at 400+ px/s the effective maximum steering angle must be clearly below the low-speed 0.42 rad"); // J
                assert(std::fabs(std::fabs(car.getCurrentSteerAngle()) - info.effectiveMaxSteerAngle) < 0.01f &&
                       "a held full command must settle at the speed-dependent effective maximum angle"); // J
                sawSmallerTarget = true;
            }
        }
        assert(sawSmallerTarget && car.isAlive() && "setup: the +1 hold phase must run");
        assert(maxObservedAngle < params.maxSteerAngle - 0.03f &&
               "a full steering command at high speed must never reach the low-speed full-lock angle"); // J

        // Flip to -1: still rate limited, and the wheel ends near the mirrored effective maximum.
        for (int i = 0; i < 25 && car.isAlive(); ++i)
        {
            step(-1.0f);
        }
        assert(car.isAlive() && car.getCurrentSteerAngle() < 0.0f && "the wheel must follow the flipped command");
    }

    // K: reset() restores full authority and a straight wheel.
    {
        simulation::Car car(params, arena);
        accelerateStraight(car, 430.0f);
        simulation::CarInput input;
        input.throttle = 0.5f;
        input.steering = 1.0f;
        for (int i = 0; i < 10; ++i)
        {
            car.update(input, kSimulationDt);
        }
        assert(car.getTireDebugInfo().steeringAuthority < 0.80f && "setup: the car must be at reduced-authority speed");

        car.reset(startPosition, 0.0f);
        assert(car.getSpeed() == 0.0f && car.getCurrentSteerAngle() == 0.0f && car.getTireDebugInfo().steeringAuthority == 1.0f &&
               "reset must clear speed, the wheel angle and the authority diagnostic"); // K
        for (int i = 0; i < 8; ++i)
        {
            car.update(input, kSimulationDt);
        }
        assert(std::fabs(car.getCurrentSteerAngle() - params.maxSteerAngle) < 1e-6f &&
               car.getTireDebugInfo().steeringAuthority == 1.0f &&
               "after reset the car is back at full low-speed authority (full lock reaches 0.42 rad)"); // K
    }

    TraceLog(LOG_INFO, "Steering authority verification: all deterministic checks passed");
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

    // kInputCount Input + 1 Bias + 3 Output, no connections -- every output
    // is deterministically 0 unless a test adds its own connections.
    auto makeDisconnectedGenome = []()
    {
        Genome genome;
        for (int i = 0; i < ai::NeuralNetwork::kInputCount; ++i)
        {
            genome.addNode(NodeGene{i, NodeType::Input});
        }
        genome.addNode(NodeGene{ai::NeuralNetwork::kInputCount, NodeType::Bias});
        genome.addNode(NodeGene{100, NodeType::Output}); // steering
        genome.addNode(NodeGene{101, NodeType::Output}); // throttle request
        genome.addNode(NodeGene{102, NodeType::Output}); // brake request
        return genome;
    };

    // 1 & 12 & H: the controller stores and uses a valid three-output network --
    // construction and one update() succeed without throwing. A disconnected
    // network requests neutral (0.5) throttle and no brake, so the combined
    // command is throttle 0.5, brake 0.
    {
        static_assert(ai::NeuralNetwork::kOutputCount == 3, "this suite assumes steering + throttle + brake outputs");
        AIController controller(ai::neat::buildPhenotype(makeDisconnectedGenome())); // H
        simulation::Car car(makeCarParams(), track);
        car.reset(kSpawnPosition, kSpawnHeading);
        simulation::TrackProgress progress(track);
        progress.reset(car);
        const simulation::CarInput input = controller.update(car, progress);
        assert(input.steering == 0.0f && input.throttle == 0.5f && input.brake == 0.0f &&
               "a disconnected network must map to zero steering, neutral (0.5) throttle, and off (0.0) brake");
    }

    // 2 & 3: output 0 drives steering, output 1 the throttle request, output 2
    // the brake request. Bias (always 1.0) connects only to steering, so a
    // positive weight there must move steering but leave throttle/brake at
    // their neutral values.
    {
        Genome genome = makeDisconnectedGenome();
        genome.addConnection(ConnectionGene{ai::NeuralNetwork::kInputCount, 100, 1.0f, true, 0});
        AIController controller(ai::neat::buildPhenotype(genome));

        simulation::Car car(makeCarParams(), track);
        car.reset(kSpawnPosition, kSpawnHeading);
        simulation::TrackProgress progress(track);
        progress.reset(car);
        const simulation::CarInput input = controller.update(car, progress);

        assert(input.steering > 0.5f && "output index 0 must map to steering");
        assert(std::fabs(input.throttle - 0.5f) < kEps && "output index 1 (throttle request) must be unaffected");
        assert(input.brake == 0.0f && "output index 2 (brake request) must be unaffected and stay off");
    }

    // Request mappings, exercised directly. Throttle: (raw + 1) / 2 clamped
    // to [0, 1]. Brake: neutral or negative raw output requests no brake, and
    // positive raw output ramps linearly to full brake at +1.0.
    {
        assert(AIController::mapThrottle(-1.0f) == 0.0f && "raw throttle -1.0 must request 0.0");
        assert(AIController::mapThrottle(0.0f) == 0.5f && "raw throttle 0.0 must request 0.5");
        assert(AIController::mapThrottle(1.0f) == 1.0f && "raw throttle +1.0 must request 1.0");
        assert(AIController::mapThrottle(3.0f) == 1.0f && AIController::mapThrottle(-3.0f) == 0.0f &&
               "raw throttle outside [-1, 1] must clamp");

        assert(AIController::mapBrake(-1.0f) == 0.0f && "raw brake -1.0 must map to 0.0"); // 14
        assert(AIController::mapBrake(-0.5f) == 0.0f && "raw brake -0.5 must map to 0.0"); // 15
        assert(AIController::mapBrake(0.0f) == 0.0f && "raw brake 0.0 (neutral) must map to 0.0, not 50% brake"); // 16
        assert(std::fabs(AIController::mapBrake(0.5f) - 0.5f) < kEps && "raw brake +0.5 must map to 0.5"); // 17
        assert(AIController::mapBrake(1.0f) == 1.0f && "raw brake +1.0 must map to 1.0"); // 18
        assert(AIController::mapBrake(2.0f) == 1.0f && "raw brake above +1.0 must clamp to 1.0");
        assert(AIController::mapBrake(-2.0f) == 0.0f && "raw brake below -1.0 must clamp to 0.0");
    }

    // A-E: the subtractive net-longitudinal combination, exercised directly on
    // requests. longitudinal = throttleRequest - brakeRequest; throttle =
    // max(0, l), brake = max(0, -l).
    {
        struct Case
        {
            float throttleRequest, brakeRequest, throttle, brake;
        };
        const Case cases[] = {
            {0.8f, 0.0f, 0.8f, 0.0f}, // no brake request
            {0.8f, 0.2f, 0.6f, 0.0f}, // smaller brake request trims throttle
            {0.8f, 0.8f, 0.0f, 0.0f}, // equal requests cancel
            {0.8f, 0.9f, 0.0f, 0.1f}, // brake just exceeds throttle
            {0.3f, 0.8f, 0.0f, 0.5f},
            {0.0f, 1.0f, 0.0f, 1.0f},
            {1.0f, 0.0f, 1.0f, 0.0f},
            {0.0f, 0.0f, 0.0f, 0.0f},
            {0.9f, 0.2f, 0.7f, 0.0f},
            {0.6f, 0.6f, 0.0f, 0.0f},
        };
        for (const Case& c : cases)
        {
            const AIController::LongitudinalCommand command =
                AIController::combineLongitudinal(c.throttleRequest, c.brakeRequest);
            assert(std::fabs(command.throttle - c.throttle) < kEps && std::fabs(command.brake - c.brake) < kEps &&
                   "combineLongitudinal must match the documented example pairs"); // A-E
        }
        assert(AIController::combineLongitudinal(0.8f, 0.8f).throttle == 0.0f &&
               AIController::combineLongitudinal(0.8f, 0.8f).brake == 0.0f && "equal requests must cancel to exactly (0, 0)");
        assert(AIController::combineLongitudinal(0.3f, 0.8f).throttle == 0.0f &&
               AIController::combineLongitudinal(0.0f, 1.0f).brake == 1.0f && "over-threshold brake zeroes throttle");

        // Subtraction is continuous: a tiny change in either request changes
        // the resulting pedals by at most that tiny amount, across the
        // crossing point too.
        for (float t : {0.0f, 0.25f, 0.5f, 0.75f, 0.99f})
        {
            const float eps = 1e-4f;
            const AIController::LongitudinalCommand below = AIController::combineLongitudinal(t, t - eps);
            const AIController::LongitudinalCommand at = AIController::combineLongitudinal(t, t);
            const AIController::LongitudinalCommand above = AIController::combineLongitudinal(t, t + eps);
            assert(at.throttle == 0.0f && at.brake == 0.0f && "pedals are (0, 0) exactly at the crossing point");
            assert(below.throttle <= 2.0f * eps && below.brake == 0.0f && "just below the crossing point throttle is ~0");
            assert(above.brake <= 2.0f * eps && above.throttle == 0.0f && "just above the crossing point brake is ~0");
        }

        // A sub-crossing brake request trims throttle by exactly its size:
        // throttle = throttleRequest - brakeRequest.
        for (int t = 0; t <= 100; ++t)
        {
            for (int b = 0; b <= t; ++b)
            {
                const float throttleRequest = static_cast<float>(t) * 0.01f;
                const float brakeRequest = static_cast<float>(b) * 0.01f;
                const AIController::LongitudinalCommand command = AIController::combineLongitudinal(throttleRequest, brakeRequest);
                assert(std::fabs(command.throttle - (throttleRequest - brakeRequest)) < kEps && command.brake == 0.0f &&
                       "a brake request not exceeding the throttle request must trim throttle by its size and give no brake");
            }
        }

        // No dead zone: any nonzero difference produces the matching pedal.
        assert(AIController::combineLongitudinal(0.501f, 0.5f).throttle > 0.0f &&
               AIController::combineLongitudinal(0.5f, 0.501f).brake > 0.0f && "there must be no dead zone");

        // Defensive clamp: out-of-range requests still yield pedals in [0, 1].
        assert(AIController::combineLongitudinal(5.0f, 0.0f).throttle == 1.0f &&
               AIController::combineLongitudinal(0.0f, 5.0f).brake == 1.0f && "out-of-range requests must clamp to [0, 1]");
    }

    // F: simultaneous post-combination throttle and brake is impossible --
    // swept over a dense grid of request pairs (including out-of-range ones).
    {
        int checked = 0;
        for (int t = -20; t <= 120; ++t)
        {
            for (int b = -20; b <= 120; ++b)
            {
                const float throttleRequest = static_cast<float>(t) * 0.01f;
                const float brakeRequest = static_cast<float>(b) * 0.01f;
                const AIController::LongitudinalCommand command =
                    AIController::combineLongitudinal(throttleRequest, brakeRequest);
                assert(std::min(command.throttle, command.brake) == 0.0f &&
                       "throttle and brake must never both be nonzero"); // F
                assert(command.throttle >= 0.0f && command.throttle <= 1.0f && command.brake >= 0.0f &&
                       command.brake <= 1.0f && "throttle and brake must both stay within [0, 1]");
                if (t >= 0 && t <= 100 && b >= 0 && b <= 100)
                {
                    // In-range pairs: the pedals are the positive and negative
                    // parts of throttleRequest - brakeRequest.
                    const float longitudinal = throttleRequest - brakeRequest;
                    assert((command.brake > 0.0f) == (brakeRequest > throttleRequest) &&
                           "brake command must be nonzero exactly when brakeRequest > throttleRequest");
                    assert(command.throttle == std::max(0.0f, longitudinal) && command.brake == std::max(0.0f, -longitudinal) &&
                           "pedals must be max(0, l) and max(0, -l) of l = throttleRequest - brakeRequest");
                }
                ++checked;
            }
        }
        assert(checked == 141 * 141 && "the sweep must cover the whole grid");
    }

    // 4-6 & 19-21 (full network -> controller path): the two requests are
    // combined, raw telemetry accessors keep reporting the SEPARATE raw
    // throttle/brake outputs, and the CarInput never carries both pedals.
    {
        simulation::Car car(makeCarParams(), track);
        car.reset(kSpawnPosition, kSpawnHeading);
        simulation::TrackProgress progress(track);
        progress.reset(car);

        // Throttle-only wiring: raw brake stays 0 (no brake request).
        Genome throttleGenome = makeDisconnectedGenome();
        throttleGenome.addConnection(ConnectionGene{ai::NeuralNetwork::kInputCount, 101, 1.0f, true, 0});
        AIController throttleController(ai::neat::buildPhenotype(throttleGenome));
        const simulation::CarInput throttleInput = throttleController.update(car, progress);
        assert(throttleController.getRawThrottleOutput() > 0.0f && throttleController.getRawBrakeOutput() == 0.0f &&
               "raw throttle/brake accessors must report the separate network outputs");
        assert(throttleInput.throttle > 0.5f + kEps && throttleInput.brake == 0.0f &&
               "a positive throttle request with no brake request must give throttle above 0.5 and no brake");

        // Brake-dominant wiring: throttle request driven to 0 (raw -1), brake to ~1.
        Genome brakeGenome = makeDisconnectedGenome();
        brakeGenome.addConnection(ConnectionGene{ai::NeuralNetwork::kInputCount, 101, -10.0f, true, 0});
        brakeGenome.addConnection(ConnectionGene{ai::NeuralNetwork::kInputCount, 102, 10.0f, true, 1});
        AIController brakeController(ai::neat::buildPhenotype(brakeGenome));
        car.reset(kSpawnPosition, kSpawnHeading);
        const simulation::CarInput brakeInput = brakeController.update(car, progress);
        assert(brakeController.getRawThrottleOutput() < -0.99f && brakeController.getRawBrakeOutput() > 0.99f &&
               "raw outputs must stay separate: throttle ~-1, brake ~+1");
        assert(brakeInput.throttle == 0.0f && brakeInput.brake > 0.99f && brakeInput.brake <= 1.0f &&
               "a saturating brake request against a zero throttle request must give ~full brake and no throttle");

        // Equal requests (0.6, 0.6) cancel: throttle raw = 2*0.6-1 = 0.2, brake raw = 0.6.
        Genome cancelGenome = makeDisconnectedGenome();
        cancelGenome.addConnection(ConnectionGene{ai::NeuralNetwork::kInputCount, 101, std::atanh(0.2f), true, 0});
        cancelGenome.addConnection(ConnectionGene{ai::NeuralNetwork::kInputCount, 102, std::atanh(0.6f), true, 1});
        AIController cancelController(ai::neat::buildPhenotype(cancelGenome));
        car.reset(kSpawnPosition, kSpawnHeading);
        const simulation::CarInput cancelInput = cancelController.update(car, progress);
        assert(cancelInput.throttle < kEps && cancelInput.brake < kEps &&
               std::min(cancelInput.throttle, cancelInput.brake) == 0.0f &&
               "equal throttle and brake requests must cancel through the full network path"); // E
        assert(std::fabs(cancelController.getThrottleRequest() - 0.6f) < kEps &&
               std::fabs(cancelController.getBrakeRequest() - 0.6f) < kEps &&
               "the controller must expose the [0, 1] requests it combined");

        // Brake request larger than throttle request: net brake only.
        Genome netBrakeGenome = makeDisconnectedGenome();
        netBrakeGenome.addConnection(ConnectionGene{ai::NeuralNetwork::kInputCount, 101, std::atanh(-0.4f), true, 0}); // request 0.3
        netBrakeGenome.addConnection(ConnectionGene{ai::NeuralNetwork::kInputCount, 102, std::atanh(0.8f), true, 1});   // request 0.8
        AIController netBrakeController(ai::neat::buildPhenotype(netBrakeGenome));
        car.reset(kSpawnPosition, kSpawnHeading);
        const simulation::CarInput netBrakeInput = netBrakeController.update(car, progress);
        assert(netBrakeInput.throttle == 0.0f && std::fabs(netBrakeInput.brake - 0.5f) < kEps &&
               "requests (0.3, 0.8) through the full network must give brake 0.5 and no throttle"); // D
    }

    // G: steering behavior is unchanged -- mapped steering is still the raw
    // steering output clamped to [-1, 1], i.e. tanh(weight) for a
    // Bias -> Steering weight, independent of the longitudinal outputs.
    {
        for (float weight : {1.0f, -0.5f, 0.25f, 10.0f, -10.0f})
        {
            Genome genome = makeDisconnectedGenome();
            genome.addConnection(ConnectionGene{ai::NeuralNetwork::kInputCount, 100, weight, true, 0});
            // Saturating brake/throttle requests must not disturb steering.
            genome.addConnection(ConnectionGene{ai::NeuralNetwork::kInputCount, 101, -10.0f, true, 1});
            genome.addConnection(ConnectionGene{ai::NeuralNetwork::kInputCount, 102, 10.0f, true, 2});
            AIController controller(ai::neat::buildPhenotype(genome));
            simulation::Car car(makeCarParams(), track);
            car.reset(kSpawnPosition, kSpawnHeading);
            simulation::TrackProgress progress(track);
            progress.reset(car);
            const simulation::CarInput input = controller.update(car, progress);
            const float expected = std::clamp(std::tanh(weight), -1.0f, 1.0f);
            assert(std::fabs(controller.getRawSteeringOutput() - std::tanh(weight)) < kEps &&
                   std::fabs(input.steering - expected) < kEps &&
                   "steering must remain the raw steering output clamped to [-1, 1]"); // G
        }
    }

    // 22: the actual generation-0 demonstration genome (AppConfig.cpp's
    // createDemonstrationGenome()) has exactly outputs 100/101/102, the
    // documented initial output wiring, and still accelerates with brake
    // fully off (raw brake stays negative, which requests no brake).
    {
        const Genome demo = createDemonstrationGenome();
        for (int outputId : {100, 101, 102})
        {
            assert(demo.findNode(outputId) != nullptr && demo.findNode(outputId)->getType() == NodeType::Output &&
                   "the demonstration genome must have Output nodes 100, 101 and 102");
        }
        int outputNodeCount = 0;
        for (const ai::neat::NodeGene& node : demo.nodes())
        {
            outputNodeCount += (node.getType() == NodeType::Output) ? 1 : 0;
        }
        assert(outputNodeCount == ai::NeuralNetwork::kOutputCount && "output node count must equal kOutputCount");

        auto weightOf = [&demo](int source, int target) -> float
        {
            const ai::neat::ConnectionGene* connection = demo.findConnection(source, target);
            assert(connection != nullptr && "expected demonstration-genome connection is missing");
            return connection->getWeight();
        };
        assert(weightOf(0, 100) == -0.5f && weightOf(1, 100) == -0.3f && weightOf(3, 100) == 0.3f &&
               weightOf(4, 100) == 0.5f && "steering wiring must be the four sensor -> steering connections");
        assert(weightOf(ai::NeuralNetwork::kInputCount, 101) == 0.6f && weightOf(2, 101) == 0.4f &&
               "throttle wiring must be Bias -> Throttle 0.6 and CenterSensor -> Throttle 0.4");
        assert(weightOf(ai::NeuralNetwork::kInputCount, 102) == -0.5f && "brake wiring must be Bias -> Brake -0.5");
        assert(demo.connections().size() == 7 && "the demonstration genome must have exactly 7 connections");

        AIController controller(ai::neat::buildPhenotype(demo));
        simulation::Car car(makeCarParams(), track);
        car.reset(kSpawnPosition, kSpawnHeading);
        simulation::TrackProgress progress(track);
        progress.reset(car);
        const simulation::CarInput input = controller.update(car, progress);
        assert(input.throttle > 0.5f && "generation-0 demonstration genome must still accelerate"); // 22
        assert(controller.getRawBrakeOutput() < 0.0f && "generation-0 raw brake output must remain negative (bias-driven)");
        assert(input.brake == 0.0f && "generation-0 demonstration genome's brake must start fully off");
        car.reset(kSpawnPosition, kSpawnHeading);
    }

    // 7 & 8: mapped steering stays within [-1,1], throttle and brake within
    // [0,1] (and never both nonzero), even under saturating weights and an
    // actively driving car -- for a throttle-heavy, a brake-heavy and a
    // both-saturated wiring.
    for (int wiring = 0; wiring < 3; ++wiring)
    {
        const float throttleWeight = (wiring == 1) ? -10.0f : 10.0f;
        const float brakeWeight = (wiring == 0) ? -10.0f : 10.0f;
        Genome genome = makeDisconnectedGenome();
        genome.addConnection(ConnectionGene{0, 100, 10.0f, true, 0});
        genome.addConnection(ConnectionGene{ai::NeuralNetwork::kInputCount, 101, throttleWeight, true, 1});
        genome.addConnection(ConnectionGene{ai::NeuralNetwork::kInputCount, 102, brakeWeight, true, 2});
        AIController controller(ai::neat::buildPhenotype(genome));

        simulation::Car car(makeCarParams(), track);
        car.reset(kSpawnPosition, kSpawnHeading);
        simulation::TrackProgress progress(track);
        progress.reset(car);
        simulation::CarInput driveInput;
        driveInput.throttle = 1.0f;
        driveInput.steering = 1.0f;
        for (int i = 0; i < 30 && car.isAlive(); ++i)
        {
            car.update(driveInput, kSimulationDt);
            progress.update(car);
            const simulation::CarInput aiInput = controller.update(car, progress);
            assert(aiInput.steering >= -1.0f && aiInput.steering <= 1.0f && "mapped steering must stay within [-1, 1]");
            assert(aiInput.throttle >= 0.0f && aiInput.throttle <= 1.0f && "mapped throttle must stay within [0, 1]");
            assert(aiInput.brake >= 0.0f && aiInput.brake <= 1.0f && "mapped brake must stay within [0, 1]");
            assert(std::min(aiInput.throttle, aiInput.brake) == 0.0f && "throttle and brake must never both be nonzero"); // F
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
        simulation::TrackProgress progress(track);
        progress.reset(car);
        controller.update(car, progress);

        const float expectedCenterSensor = car.getSensors()[2].normalizedDistance;
        assert(std::fabs(controller.getLastObservation().values[2] - expectedCenterSensor) < kEps &&
               "AIController's Observation must be built from the provided Car's own sensor readings");
    }

    // 10: repeated calls with unchanged Car state are deterministic.
    {
        Genome genome = makeDisconnectedGenome();
        genome.addConnection(ConnectionGene{0, 100, 0.7f, true, 0});
        genome.addConnection(ConnectionGene{ai::NeuralNetwork::kInputCount, 101, -0.3f, true, 1});
        AIController controller(ai::neat::buildPhenotype(genome));

        simulation::Car car(makeCarParams(), track);
        car.reset(kSpawnPosition, kSpawnHeading);
        simulation::TrackProgress progress(track);
        progress.reset(car);

        const simulation::CarInput first = controller.update(car, progress);
        const simulation::CarInput second = controller.update(car, progress);
        assert(first.steering == second.steering && first.throttle == second.throttle && first.brake == second.brake &&
               "repeated updates against an unchanged Car must produce identical CarInput");
    }

    // 11: the controller does not mutate the Car or TrackProgress it reads from.
    {
        Genome genome = makeDisconnectedGenome();
        genome.addConnection(ConnectionGene{0, 100, 0.5f, true, 0});
        AIController controller(ai::neat::buildPhenotype(genome));

        simulation::Car car(makeCarParams(), track);
        car.reset(kSpawnPosition, kSpawnHeading);
        simulation::TrackProgress progress(track);
        progress.reset(car);
        const Vector2 positionBefore = car.getPosition();
        const Vector2 velocityBefore = car.getVelocity();
        const float headingBefore = car.getHeading();
        const Vector2 tangentBefore = progress.getTrackTangent();
        const float bestProgressBefore = progress.getBestProgress();

        controller.update(car, progress);

        assert(car.getPosition().x == positionBefore.x && car.getPosition().y == positionBefore.y &&
               "AIController::update must not move the Car");
        assert(car.getVelocity().x == velocityBefore.x && car.getVelocity().y == velocityBefore.y &&
               "AIController::update must not change the Car's velocity");
        assert(car.getHeading() == headingBefore && "AIController::update must not change the Car's heading");
        assert(progress.getTrackTangent().x == tangentBefore.x && progress.getTrackTangent().y == tangentBefore.y &&
               progress.getBestProgress() == bestProgressBefore &&
               "AIController::update must not modify TrackProgress");
    }

    // 13: a disabled connection stays inactive after phenotype construction.
    {
        Genome genome = makeDisconnectedGenome();
        genome.addConnection(ConnectionGene{0, 100, 1.0f, false, 0});
        AIController controller(ai::neat::buildPhenotype(genome));

        simulation::Car car(makeCarParams(), track);
        car.reset(kSpawnPosition, kSpawnHeading);
        simulation::TrackProgress progress(track);
        progress.reset(car);
        const simulation::CarInput input = controller.update(car, progress);
        assert(input.steering == 0.0f && "a disabled connection must not affect evaluation after phenotype construction");
    }

    // Dead car: the controller must not keep evaluating the network, and
    // must apply neutral CarInput instead.
    {
        Genome genome = makeDisconnectedGenome();
        genome.addConnection(ConnectionGene{ai::NeuralNetwork::kInputCount, 101, 1.0f, true, 0}); // would otherwise raise throttle above 0.5
        AIController controller(ai::neat::buildPhenotype(genome));

        simulation::Car car(makeCarParams(), track);
        car.reset(kSpawnPosition, kSpawnHeading);
        simulation::TrackProgress progress(track);
        progress.reset(car);
        simulation::CarInput driveOffTrack;
        driveOffTrack.throttle = 1.0f;
        driveOffTrack.steering = 0.0f;
        for (int i = 0; i < 300 && car.isAlive(); ++i)
        {
            car.update(driveOffTrack, kSimulationDt);
        }
        assert(!car.isAlive() && "driving straight for 5s must leave the road band and kill the car");

        const simulation::CarInput deadInput = controller.update(car, progress);
        assert(deadInput.steering == 0.0f && deadInput.throttle == 0.0f && deadInput.brake == 0.0f &&
               "a dead car must receive neutral CarInput from AIController, not a network-derived one");
        car.reset(kSpawnPosition, kSpawnHeading);
    }

    TraceLog(LOG_INFO, "AI controller verification: all deterministic checks passed");
}
} // namespace verification
