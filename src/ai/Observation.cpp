#include "ai/Observation.h"

#include <algorithm>
#include <cmath>

#include "raylib.h"

namespace ai
{

namespace
{

static_assert(simulation::Car::kSensorCount + 9 == kObservationSize,
              "Observation expects one slot per sensor, seven vehicle-state/heading slots, and two preview slots");

// Wraps a radian angle into [-pi, pi]. Same formula as
// telemetry::HairpinTelemetry's/telemetry::ChampionTelemetry's own local
// wrapToPi() helpers -- kept as an independent copy here (small, TU-local
// utility, same precedent as those two) rather than a shared header, but
// deliberately identical in definition so the raw headingError this
// observation normalizes matches what telemetry already logs.
float wrapToPi(float angle)
{
    while (angle > PI)
    {
        angle -= 2.0f * PI;
    }
    while (angle < -PI)
    {
        angle += 2.0f * PI;
    }
    return angle;
}

} // namespace

Observation buildObservation(const simulation::Car& car, const simulation::TrackProgress& progress)
{
    Observation observation;

    const auto& sensors = car.getSensors();
    for (int i = 0; i < simulation::Car::kSensorCount; ++i)
    {
        observation.values[i] = std::clamp(sensors[i].normalizedDistance, 0.0f, 1.0f);
    }

    const float maxSpeed = car.getMaxSpeed();

    observation.values[5] = std::clamp(car.getSpeed() / maxSpeed, 0.0f, 1.0f);
    observation.values[6] = std::clamp(car.getForwardVelocity() / maxSpeed, -1.0f, 1.0f);
    observation.values[7] = std::clamp(car.getLateralVelocity() / maxSpeed, -1.0f, 1.0f);
    observation.values[8] = std::clamp(car.getSlipAngle() / static_cast<float>(PI), -1.0f, 1.0f);

    observation.values[kActualSteerObservationIndex] =
        std::clamp(car.getCurrentSteerAngle() / car.getParams().maxSteerAngle, -1.0f, 1.0f);
    observation.values[kYawRateObservationIndex] =
        std::clamp(car.getYawRate() / kYawRateNormalizationScale, -1.0f, 1.0f);

    // Heading error: car heading minus the local track tangent direction,
    // wrapped to [-pi,pi] then normalized by pi -- same formula (and same
    // zero-tangent fallback, only possible before the very first
    // reset()/update() has ever run on progress) as
    // telemetry::HairpinTelemetry's/telemetry::ChampionTelemetry's own
    // diagnostic headingError field, so this input and that telemetry column
    // agree exactly. Orientation only -- never lateral offset, never a
    // target position (see Observation.h's class comment).
    const Vector2 trackTangent = progress.getTrackTangent();
    const bool haveTrackTangent = (trackTangent.x != 0.0f || trackTangent.y != 0.0f);
    const float trackDirectionAngle = haveTrackTangent ? std::atan2(trackTangent.y, trackTangent.x) : car.getHeading();
    const float headingError = wrapToPi(car.getHeading() - trackDirectionAngle);
    observation.values[kHeadingErrorObservationIndex] = std::clamp(headingError / static_cast<float>(PI), -1.0f, 1.0f);

    // Preview heading errors (slots 12 & 13): identical formula and sign
    // convention to slot 11 (car heading MINUS track direction), against the
    // track tangent farther along the centerline instead of the local one.
    // Same zero-tangent fallback (only possible for a degenerate segment)
    // as slot 11 -- treated as "aligned", i.e. 0. Orientation only.
    auto previewHeadingError = [&](float distanceAhead)
    {
        const Vector2 futureTangent = progress.getTrackTangentAhead(distanceAhead);
        const bool haveFutureTangent = (futureTangent.x != 0.0f || futureTangent.y != 0.0f);
        const float futureDirectionAngle =
            haveFutureTangent ? std::atan2(futureTangent.y, futureTangent.x) : car.getHeading();
        return std::clamp(wrapToPi(car.getHeading() - futureDirectionAngle) / static_cast<float>(PI), -1.0f, 1.0f);
    };
    observation.values[kPreviewNearObservationIndex] = previewHeadingError(kPreviewNearDistance);
    observation.values[kPreviewFarObservationIndex] = previewHeadingError(kPreviewFarDistance);

    return observation;
}

} // namespace ai
