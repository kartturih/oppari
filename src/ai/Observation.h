#pragma once

#include <array>

#include "simulation/Car.h"
#include "simulation/TrackProgress.h"

namespace ai
{

// Fixed-size, normalized snapshot of a Car's local perception/dynamics (plus
// three track-relative ORIENTATION cues -- the current track direction, slot
// 11, and the track direction 120px and 300px farther along, slots 12-13),
// used as NeuralNetwork's input vector. Deliberately NOT a waypoint/racing-line
// follower: no world position, no lateral offset from the centerline, no
// future point coordinates or bearing to one, no distance-to-corner, no
// target speed -- the network must still discover steering magnitude,
// timing, throttle, braking, and its own line through the track's width
// entirely itself. The only preview it gets is which way the track will be
// pointing ahead (orientation), never where anything is.
inline constexpr int kObservationSize = 14;

// Slots: 0..4 = sensor normalizedDistance at -60/-30/0/+30/+60 deg,
// 5 = speed, 6 = forward velocity, 7 = lateral velocity, 8 = slip angle,
// 9 = actual (rate-limited) steering angle, 10 = yaw rate, 11 = heading
// error relative to the local track direction, 12 = heading error relative
// to the track direction kPreviewNearDistance px ahead, 13 = same for
// kPreviewFarDistance px ahead (all normalized).
//
// 9 and 10 close the loop the steering-rate limiter (see
// CarParams::maxSteerRateRadPerSec) otherwise leaves open: without them the
// network commands steering purely open-loop, with no way to sense that the
// actual wheel angle lags its own past commands, or how fast the car is
// currently rotating -- both plausible drivers of the post-rate-limit
// snaking behavior this pair of inputs was added to address.
inline constexpr int kActualSteerObservationIndex = 9;
inline constexpr int kYawRateObservationIndex = 10;

// 11: normalized heading error -- how far the car's own heading has drifted
// from the local track direction (the centerline's tangent at the car's
// currently tracked/projected point, TrackProgress::getTrackTangent()), NOT
// how far the car is laterally offset from that centerline. This is
// deliberately an ORIENTATION cue only: it tells the network whether it is
// currently pointed roughly along the track or badly askew, without ever
// saying where across the track's width it should be -- the same
// distinction HairpinTelemetry/ChampionTelemetry's own diagnostic
// headingError field already draws (see their kWrongWayHeadingErrorThresholdRad
// comment), and the reason this input uses the identical formula (car
// heading minus track tangent angle, wrapped to [-pi,pi]) rather than a
// newly-invented one.
inline constexpr int kHeadingErrorObservationIndex = 11;

// 12 & 13: track-direction PREVIEW -- the same heading-error formula as slot
// 11 (car heading minus track tangent direction, wrapped to [-pi,pi],
// normalized by pi, clamped to [-1,1], same sign convention) but against the
// tangent at a point kPreviewNearDistance / kPreviewFarDistance px farther
// along the centerline than the car's currently projected position
// (TrackProgress::getTrackTangentAhead(), wrapping across the lap seam). At
// ~230px/s these look ~0.5s and ~1.3s ahead; the 300px far preview stays
// inside the 400px sensor range. Still orientation only: unlike a bearing to
// a future waypoint, this value carries no information about where the car
// is laterally, nor how far away a corner is -- only what direction the
// track will be pointing there, relative to the car's current heading.
inline constexpr int kPreviewNearObservationIndex = 12;
inline constexpr int kPreviewFarObservationIndex = 13;
inline constexpr float kPreviewNearDistance = 120.0f; // px along the centerline
inline constexpr float kPreviewFarDistance = 300.0f;  // px along the centerline

// Fixed normalization scale (rad/s) for slot 10 -- chosen from real observed
// yaw-rate magnitudes rather than an arbitrary round number. Hairpin-failure
// telemetry across gen-0 individuals (results/telemetry/hairpin_*.csv,
// yaw_rate column) put the median at ~0.12 rad/s, the 90th percentile at
// ~1.57 rad/s, the 99th at ~3.35 rad/s, and the observed max at ~4.05 rad/s;
// VehicleVerification's own sustained-full-lock acceptance test settles to a
// steady committed rotation of ~2.6 rad/s. 4.0 sits just above that
// steady-state committed spin and right at the observed failure-data
// ceiling, so ordinary-to-hard cornering (the 0-2 rad/s range that matters
// most for control) keeps good resolution while genuinely extreme,
// spinning-out yaw rates saturate to +-1 instead of being compressed into a
// sliver of the range. Well below the 15 rad/s "runaway/NaN" safety bound
// VehicleVerification asserts against elsewhere (see
// verifyVehiclePhysics/verifySteeringRateLimit) -- that bound flags a
// broken simulation state, not a value this normalization should try to
// resolve.
inline constexpr float kYawRateNormalizationScale = 4.0f; // rad/s

struct Observation
{
    std::array<float, kObservationSize> values{};
};

Observation buildObservation(const simulation::Car& car, const simulation::TrackProgress& progress);

} // namespace ai
