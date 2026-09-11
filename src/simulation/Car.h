#pragma once

#include <array>

#include "box2d/box2d.h"
#include "raylib.h"

#include "simulation/Track.h"

namespace simulation
{

// The only way to drive a Car -- it never reads input devices itself.
struct CarInput
{
    float steering = 0.0f; // -1.0 (full left) to 1.0 (full right)
    float throttle = 0.0f; // 0.0 to 1.0
    float brake = 0.0f;    // 0.0 to 1.0 -- independent of throttle; both may be nonzero at once
};

// Tuning for the Box2D-backed top-down front/rear bicycle-model car. Shapes
// the simulation a genome is evaluated against -- NEAT never reads this.
struct CarParams
{
    // Body dimensions, px (also Box2D units -- no separate scale factor).
    float length = 24.0f;
    float width = 12.0f;

    // mass = density * length * width. Raised from the original 0.05
    // (mass 14.4) to 0.075 (mass 21.6, +50%) -- the original mass was too
    // light relative to the tire/engine force scale: peak single-axle tire
    // force (frontMaxTireForce/rearMaxTireForce below) divided by the
    // original mass gave implausibly large accelerations for a "real car"
    // feel (see rotationalInertia's comment for the more severe yaw-side
    // version of the same problem). engineForce, maxBrakeForce,
    // rollingResistance, dragCoefficient and rearMaxTireForce/
    // rearCorneringStiffness below are all scaled by this same 1.5x factor
    // so every straight-line acceleration/braking/coasting number (force /
    // mass) and the rear axle's own force-vs-slip-angle curve come out
    // numerically IDENTICAL to before -- only lateral/rotational response is
    // meant to change here, not straight-line feel. frontMaxTireForce/
    // frontCorneringStiffness are deliberately left UNSCALED (see their own
    // comment) so front-axle lateral grip softens relative to the heavier
    // car -- that axle (not the rear) dominates the turn-in "snap" this
    // change targets.
    float density = 0.075f;

    // Explicit override of Box2D's shape-derived rotational inertia (see
    // Car.cpp's createCarBody, which calls b2Body_SetMassData after body
    // creation) -- decoupled from mass/density above so yaw resistance can
    // be tuned independently of translational mass. At the ORIGINAL density
    // (0.05, mass 14.4), Box2D's automatic uniform-box inertia was 864;
    // dividing peak single-axle tire torque by that (e.g. frontMaxTireForce
    // * cgToFrontAxle = 9200*12 = 110400) gave ~128 rad/s^2 of angular
    // acceleration -- enough to add ~2 rad/s (~115deg/s) of yaw rate in a
    // SINGLE 1/60s frame from one axle alone, which is what produced the
    // "snap sideways" turn-in feel: heading rotating out from under the
    // velocity vector within a handful of frames instead of building up
    // over a real fraction of a second. At the new density alone, automatic
    // inertia would be 1296 (864 scaled by the same 1.5x as mass -- pure
    // scale-invariance, see density's comment, so it wouldn't by itself
    // change this ratio at all). This field pushes inertia further, to
    // 2400 (2.78x the original 864), cutting front-axle-driven angular
    // acceleration to ~46 rad/s^2 -- yaw now builds up over several frames
    // rather than snapping in one, while staying well short of making
    // steering feel laggy (see frontTireRelaxationTime's comment for the
    // complementary force-side smoothing).
    float rotationalInertia = 2400.0f;

    // Numerical-stability backstop only -- not the primary resistance/
    // steering mechanism (that's engineForce/resistance below and the tire
    // forces). angularDamping is small since tire slip angles already
    // provide the real yaw damping.
    float linearDamping = 0.05f;
    float angularDamping = 0.3f;

    // Rear axle's desired longitudinal force at throttle=1 (RWD -- front
    // never drives), combined with rear lateral force under the rear
    // friction circle (see rearMaxTireForce). Braking is separate -- see
    // maxBrakeForce/frontBrakeBias below. Scaled by the same 1.5x as
    // density (see its comment) to keep straight-line acceleration
    // (force/mass) numerically unchanged.
    float engineForce = 5850.0f;

    // rollingResistance opposes forward-rolling motion only; dragCoefficient
    // (quadratic) opposes the full velocity vector. Both are deliberately
    // small relative to engineForce/maxBrakeForce -- releasing the throttle
    // should coast, not "engine brake"; only actual brake input should
    // decelerate quickly (see maxBrakeForce). Together with engineForce/
    // density, sets the straight-line steady-state speed (~677px/s).
    // maxSpeed is a reference for Observation normalization, not an
    // enforced clamp (Car.cpp still applies a generous safety clamp). Both
    // scaled by the same 1.5x as density (see its comment) so the
    // steady-state speed and coast-down curve are numerically unchanged.
    float rollingResistance = 0.525f;
    float dragCoefficient = 0.012f;
    float maxSpeed = 590.0f; // px/s, approximate (below the ~677px/s analytic steady state --
                              // see rollingResistance's comment -- same as the safety-clamp margin below)

    // Braking: a longitudinal force opposing each axle's own rolling
    // motion, split front/rear by frontBrakeBias and folded into that
    // axle's friction circle alongside its lateral (and, at the rear,
    // drive) force -- so braking hard while cornering genuinely competes
    // with the grip that corner needs, exactly like engineForce does for
    // rear traction. maxBrakeForce is the combined (front+rear) budget at
    // brake=1; frontBrakeBias*maxBrakeForce and
    // (1-frontBrakeBias)*maxBrakeForce are each kept at or below that
    // axle's own maxTireForce, so straight-line braking is never silently
    // clamped away by the friction circle -- only cornering-while-braking
    // reduces it. Front-biased (60/40) like a real car, since braking
    // shifts load onto the front axle. Scaled by the same 1.5x as density
    // (see its comment) to keep braking deceleration (force/mass)
    // numerically unchanged.
    float maxBrakeForce = 9000.0f;
    float frontBrakeBias = 0.6f; // fraction of maxBrakeForce sent to the front axle

    // Bicycle-model axle distances from CG (assumed at geometric center).
    float cgToFrontAxle = 12.0f;
    float cgToRearAxle = 12.0f;

    float maxSteerAngle = 0.42f; // radians at steering = +-1 (~28.6 deg)

    // Per-axle lateral tire force curve, replacing the old plain
    // Fy = -axleMaxForce * tanh((corneringStiffness/axleMaxForce)*slipAngle)
    // (which only ever approached axleMaxForce asymptotically and NEVER
    // came back down -- a fully sideways (~90deg) tire produced the same
    // force as one right at its grip peak, which is what let a slide act
    // like slamming on the brakes: full-magnitude tire force, pointed
    // almost squarely against the velocity vector, sustained for the whole
    // slide). The new curve (see tireLateralForceMagnitude() in Car.cpp)
    // keeps that same near-zero-slip slope (corneringStiffness), rises to
    // the same peak (axleMaxForce) at the axle's own explicit
    // frontPeakSlipAngle/rearPeakSlipAngle, then FALLS OFF smoothly beyond
    // it toward axleMaxForce*slidingGripRatio as slip approaches 90 degrees
    // (a tire fully sliding sideways), staying at that floor beyond 90deg.
    // This models kinetic (sliding) friction being somewhat below the
    // static/peak grip a tire achieves right at its optimal slip angle --
    // the standard reason a slide, once fully broken away, doesn't keep
    // scrubbing off speed at the same rate as the initial grip peak.
    // Front slip angle includes steering; rear doesn't.
    float frontCorneringStiffness = 26000.0f; // force per radian
    float rearCorneringStiffness = 30000.0f;  // scaled 1.5x with density -- see its comment

    // Slip angle (radians) at which each axle's tire curve reaches its peak
    // (axleMaxForce) -- see tireLateralForceMagnitude()'s comment for how
    // this is independent of corneringStiffness/axleMaxForce (a peak this
    // early relative to that pair's implied "2*Fmax/C" would need a plain
    // quadratic rise to start impossibly steep; the cubic Hermite rise
    // decouples the two). Far smaller than a naive real-tire number
    // (~8-15deg) would suggest, because at this pixel/force/mass scale a
    // literal real-world peak angle made cornering grip vanish at absurdly
    // small steering inputs; chosen empirically instead by how THIS car's
    // full slide/recovery telemetry responds (see the rebound
    // investigation) -- 14deg front / 18deg rear keeps small-to-moderate
    // cornering (the previously-verified understeer/oversteer behavior)
    // essentially unchanged while making a genuine slide reach peak grip
    // -- and start falling off -- much sooner than the old 40.5/59.6deg,
    // which is what let the car keep building lateral force through an
    // extreme, unrealistic slip range. Front lower than rear preserves the
    // same understeer-biased margin as before (front saturates first).
    float frontPeakSlipAngle = 0.2443f; // 14 degrees
    float rearPeakSlipAngle = 0.3142f;  // 18 degrees

    // Fraction of axleMaxForce the tire curve settles to once fully
    // sliding (slip angle >= 90deg) -- see the tire-curve comment above.
    // Compared 0.7/0.5/0.4/0.3 (see the rebound investigation): with these
    // much smaller peak angles, every scripted maneuver tried -- including
    // a deliberate "flick" (snap the opposite way to break the rear loose)
    // -- kept the car's own self-stabilizing behavior from developing slip
    // much past ~28deg, where 0.7 vs 0.4 were statistically
    // indistinguishable (both ~32% min speed retained) since the falloff
    // curve has barely descended from its peak by then; slidingGripRatio's
    // real effect is specifically in the 60-90deg range no scripted input
    // reliably reached. Chosen by the curve's OWN 90deg value instead
    // (verified exactly by check 34): 0.7 leaves 70% of peak grip while
    // fully sideways -- not far below the ~90%+ available at a normal
    // cornering angle, i.e. close to the old behavior this whole
    // investigation is about fixing. 0.4 is the chosen middle ground (not
    // the lowest, 0.3, per the instruction not to default there) -- a
    // fully broken-away slide still generates 40% of peak grip (a tire
    // skidding sideways is still a tire, not frictionless), but clearly,
    // meaningfully below normal cornering grip, so an actual 90deg slide
    // (reachable by a human player combining inputs a scripted test
    // couldn't replicate) reads as a genuine, costly loss of grip without
    // acting like four anchors.
    float frontSlidingGripRatio = 0.4f;
    float rearSlidingGripRatio = 0.4f;

    // Per-axle tire relaxation: instead of the lateral force jumping
    // straight to tireLateralForceMagnitude()'s output every substep, an
    // internal EFFECTIVE SLIP ANGLE (Car::m_frontSlipAngleRelaxed/
    // m_rearSlipAngleRelaxed -- not the force itself) exponentially
    // approaches the axle's true (instantaneous) slip angle, and the tire
    // curve is evaluated from that relaxed angle each substep to get the
    // actually-applied force. This is what a real tire's finite relaxation
    // length gives for free (the contact patch takes a short, finite
    // time/distance to build up new deflection) and what this model had
    // none of before -- every substep recomputed force from the
    // INSTANTANEOUS slip angle with zero lag, so grip could appear/vanish
    // within a single ~1ms substep, producing the "snap" transitions the
    // previous rework targeted.
    //
    // Relaxing the ANGLE rather than the raw force is deliberate: an early
    // version relaxed force directly and, when the true slip angle crossed
    // zero (the car's own rotation catching up with/overshooting its
    // velocity direction during a slide's recovery), the stored force kept
    // its OLD sign for up to ~150ms after the target had already flipped --
    // long enough, and large enough, to visibly kick the car in the
    // opposite direction from the commanded turn before it caught up
    // (verified via full-slide telemetry). Relaxing angle instead means the
    // stored state is always run back through the (memoryless) curve, so
    // near a sign crossing the relaxed angle -- and so the force -- passes
    // through the curve's own near-zero region, not through an
    // independently-decaying large value.
    //
    // frontTireRelaxationTime/rearTireRelaxationTime (unchanged from
    // before, 0.08s) is used while the axle is LOADING -- true slip angle
    // magnitude growing beyond what's currently stored, i.e. ordinary
    // build-up toward a new (possibly same-sign) demand. frontTireReleaseTime/
    // rearTireReleaseTime is used whenever the axle is UNLOADING -- true
    // slip angle magnitude smaller than what's stored, whether or not the
    // sign has actually crossed zero yet. A real tire's stored lateral
    // deflection releases markedly faster than it builds up in a new
    // direction (a standard hysteresis asymmetry); gating release on
    // magnitude rather than sign alone matters because true slip angle
    // collapsing back toward zero WITHOUT crossing it is the common case
    // during recovery, not the exception -- gating on sign alone (an
    // earlier version) left the relaxed angle, and so the applied force,
    // pinned near its old peak for several frames after the real geometry
    // had already relieved, which is what drove a sustained high-speed
    // full-lock yaw oscillation (see the wobble investigation) on top of
    // the sign-reversal rebound the rebound investigation already fixed.
    // Both further shrink toward kTireRelaxationLowSpeedFloorTime
    // (Car.cpp) as axle speed drops below kTireForceRampSpeed, so a
    // nearly-stopped axle can't hold onto a meaningful stored deflection
    // at all.
    float frontTireRelaxationTime = 0.08f; // seconds, build-up (unchanged)
    float rearTireRelaxationTime = 0.08f;  // seconds, build-up (unchanged)
    float frontTireReleaseTime = 0.02f;    // seconds, release-on-sign-reversal

    // rearTireReleaseTime is deliberately a little higher than
    // frontTireReleaseTime (0.027s vs 0.02s) -- a small, rear-only handling
    // tuning pass (see the recovery-feel investigation) for manual driving
    // that, while no longer wobbling, still "organized itself" back to
    // grip a bit too neatly the instant the rear genuinely broke loose.
    // Measured (not assumed) which lever actually mattered: sweeping
    // rearSlidingGripRatio had NO measurable effect on any of five
    // scripted manual-style maneuvers (none pushed the rear far enough
    // past its 18deg peak for the sliding floor to matter); nudging
    // rearPeakSlipAngle by even 2deg regressed the small-steering-stability
    // and intentional-slide checks (it shifts the whole front/rear
    // cornering balance, not just recovery); rearTireReleaseTime was the
    // one lever that changed recovery specifically, and it did so exactly
    // where wanted -- a deliberate high-speed flick's recovery lengthened
    // from ~35 frames (~0.58s) to ~56 frames (~0.93s) and reached a
    // slightly larger peak slip (30->33deg), while the milder scenarios
    // (small steering, sustained full lock, throttle oversteer, release-
    // to-recover from a modest slide) barely moved at all -- because a
    // deeper excursion accumulates more of this per-substep lag before it
    // unwinds, a longer release time affects genuine breakaway far more
    // than everyday driving. 0.027s is the highest value that still passes
    // the 5-second sustained-full-lock commitment check (0.028s tips its
    // yaw-reversal count from 3 to 4, i.e. the wobble this same mechanism
    // fixed starts creeping back) -- frontTireReleaseTime is untouched, on
    // the report's own instruction to leave front alone unless necessary.
    float rearTireReleaseTime = 0.027f; // seconds, release-on-sign-reversal

    // Per-axle grip budget. Front spends its budget on lateral force alone;
    // rear SHARES its budget between drive force and lateral force (a
    // friction circle: sqrt(FxRear^2+FyRear^2) <= rearMaxTireForce) -- this
    // is what makes heavy throttle mid-corner reduce rear grip (power
    // oversteer). rearMaxTireForce is set above engineForce so full
    // straight-line throttle alone never saturates it.
    //
    // Required cornering force scales with speed SQUARED (F = m*v^2/R), so
    // raising the coasting/top speed (see rollingResistance's comment)
    // without raising these left ordinary highway-speed curves demanding
    // far more force than these budgets could supply -- tires saturated
    // (slid) at much gentler steering than before. Doubled here so a
    // moderate-radius sweeper at high speed again sits mid-curve on the
    // tire force curve instead of pinned at its limit; cornering stiffness
    // is deliberately left unchanged (it sets the curve's near-zero-slip
    // slope, i.e. on-center responsiveness) -- raising only the ceiling
    // widens the smooth, near-linear part of the curve before saturation
    // without changing how the car responds to small inputs at any speed.
    //
    // rearMaxTireForce (and rearCorneringStiffness above) is additionally
    // scaled by the same 1.5x as density (see its comment), keeping the
    // rear axle's own force-vs-slip-angle curve and its force/mass ratio
    // numerically identical to before the mass change -- only the front
    // axle's grip is deliberately softened relative to the new, heavier
    // mass (frontMaxTireForce stays at its original, UNSCALED value), since
    // it's the front axle -- not the rear -- whose sudden lateral force
    // dominates the turn-in "snap" this whole change targets.
    float frontMaxTireForce = 9200.0f;
    float rearMaxTireForce = 15600.0f;
};

// Magnitude (always >= 0) of one axle's lateral tire force at a given
// (unsigned) slip angle, in isolation from vehicle dynamics -- exposed
// (rather than kept private to Car.cpp) so its shape (rise to a peak,
// then fall off toward a sliding floor) can be verified directly and
// exactly at chosen slip angles, independent of how closely any driven
// maneuver manages to hold the body at that angle. See
// CarParams::frontCorneringStiffness's comment for the full shape
// rationale/formula.
float tireLateralForceMagnitude(float absSlipAngle, float corneringStiffness, float maxForce, float slidingGripRatio,
                                 float peakSlipAngle);

// One forward sensor's result, cast against the track's drivable mask.
struct SensorReading
{
    float distance = 0.0f;           // px to first non-drivable sample
    float normalizedDistance = 1.0f; // distance / kMaxSensorDistance, [0,1]
    Vector2 endPoint = {0.0f, 0.0f};
};

// Debug snapshot of the tire model's most recent update() (zeroed after
// reset()). Only for the manual-mode HUD and verification -- no production
// control/fitness path reads this.
struct TireDebugInfo
{
    // Clamped CarInput echoed back from the most recent update() -- for
    // HUD display only (Car itself never stores raw CarInput otherwise).
    float steeringInput = 0.0f; // [-1, 1]
    float throttleInput = 0.0f; // [0, 1]
    float brakeInput = 0.0f;    // [0, 1]

    float steeringAngle = 0.0f;  // radians, front wheel angle
    float yawRate = 0.0f;        // rad/s
    float frontSlipAngle = 0.0f; // radians, wheel frame (includes steering) -- TRUE (instantaneous), not relaxed
    float rearSlipAngle = 0.0f;  // radians, body frame -- TRUE (instantaneous), not relaxed

    // EFFECTIVE (relaxed) slip angle each axle's tire curve is actually
    // evaluated at this substep -- see CarParams::frontTireRelaxationTime's
    // comment. Lags frontSlipAngle/rearSlipAngle above by up to their
    // relaxation/release time constant.
    float frontSlipAngleRelaxed = 0.0f;
    float rearSlipAngleRelaxed = 0.0f;

    // Per-axle force in that axle's own frame (X = rolling dir, Y =
    // lateral). frontForceX is 0 unless braking (front never drives, but
    // does brake -- see CarParams::frontBrakeBias). frontForceY/rearForceY
    // are the ACTUAL (relaxed, friction-circle-clamped) lateral force
    // actually applied; front/rearTargetForceY are the tire curve's raw,
    // UNRELAXED target for that same substep -- comparing the two shows how
    // far behind its target the relaxed force currently is (see
    // CarParams::frontTireRelaxationTime).
    float frontForceX = 0.0f;
    float frontForceY = 0.0f;
    float rearForceX = 0.0f;
    float rearForceY = 0.0f;
    float frontTargetForceY = 0.0f;
    float rearTargetForceY = 0.0f;

    // Each axle's own velocity in its own rolling frame (X = rolling dir, Y
    // = lateral) -- front is in the STEERED wheel frame, rear in the body
    // frame (it never steers). Debug/HUD only, same as the rest of this struct.
    float frontAxleVx = 0.0f;
    float frontAxleVy = 0.0f;
    float rearAxleVx = 0.0f;
    float rearAxleVy = 0.0f;

    // Friction-circle utilization, sqrt(Fx^2+Fy^2)/axleMaxTireForce, [0,1].
    float frontGripUtilization = 0.0f;
    float rearGripUtilization = 0.0f;
};

// Box2D-backed top-down car with front/rear bicycle-model tire forces,
// driven purely through CarInput. Collision checks the four rotated-body
// corners against the Track's CPU mask; Box2D handles only rigid-body
// integration, never collision geometry.
//
// Yaw is never commanded directly -- update() computes front/rear tire
// lateral forces and applies each at its own axle position via Box2D's
// point-force API, so rotation is a genuine consequence of those forces'
// moment arms. The car is RWD; engine force is the rear axle's desired
// longitudinal force. Braking is a separate longitudinal force on both
// axles, front-biased (CarParams::frontBrakeBias). Every axle's
// longitudinal force (drive and/or brake) combines with its own lateral
// force under that axle's friction circle before being applied.
//
// Each Car owns a private single-body Box2D world (cars never collide with
// each other). Move-constructible only, not copyable/assignable.
class Car
{
public:
    // Five forward sensors at fixed angles, cast in fixed steps to a fixed range.
    static constexpr int kSensorCount = 5;
    static constexpr float kSensorAngleDegrees[kSensorCount] = {-60.0f, -30.0f, 0.0f, 30.0f, 60.0f};
    // Perception-range experiment: raised from the original 200px -- see
    // the hairpin-telemetry investigation, which measured real cars
    // reaching ~480-495px/s on the start straight while useful forward
    // warning only appeared around 110-136px, well under the ~276-292px
    // stopping distance that speed needs. 400px was chosen as the smallest
    // change that could plausibly close that gap for the speeds actually
    // observed (not an attempt to cover the full ~590px/s theoretical top
    // speed). Sensor angles/origin/step and every consumer (Observation's
    // normalization, HairpinTelemetry's raw-px fields) all derive from this
    // one constant, so nothing else needed to change alongside it.
    static constexpr float kMaxSensorDistance = 400.0f; // px

    static constexpr float kSensorStep = 2.0f;           // px

    // Box2D v3 soft-step sub-step count per update(); outer timestep is
    // always the caller's fixed dt, never measured frame time.
    static constexpr int kPhysicsSubStepCount = 4;

    Car(const CarParams& params, const Track& track);
    ~Car();

    Car(const Car&) = delete;
    Car& operator=(const Car&) = delete;
    Car(Car&& other) noexcept;
    Car& operator=(Car&&) = delete; // m_track is a reference

    // Advances by dt: computes resistance + front/rear tire forces from the
    // current Box2D state, applies each axle force at its world position,
    // steps the private world, re-derives position/velocity/heading, checks
    // collision, and recasts sensors. No-op if not alive.
    void update(const CarInput& input, float dt);

    // Zeroes velocity, sets alive = true, resets tire debug info, and
    // recasts sensors for the new pose.
    void reset(Vector2 spawnPosition, float spawnHeading);

    Vector2 getPosition() const { return m_position; }
    Vector2 getVelocity() const { return m_velocity; }
    float getHeading() const { return m_heading; }
    bool isAlive() const { return m_alive; }

    const CarParams& getParams() const { return m_params; }
    float getMaxSpeed() const { return m_params.maxSpeed; }

    // Rotated body corners, front-right/front-left/rear-left/rear-right.
    std::array<Vector2, 4> getCorners() const;

    // World-space origin all sensors are cast from (center of front edge).
    Vector2 getSensorOrigin() const;

    const std::array<SensorReading, kSensorCount>& getSensors() const { return m_sensors; }

    // Derived on demand from current velocity/heading -- the whole body's
    // motion, distinct from getTireDebugInfo()'s per-tire slip angles.
    float getSpeed() const;
    float getForwardVelocity() const;
    float getLateralVelocity() const; // positive = rightward
    float getSlipAngle() const;       // heading vs. velocity direction, radians

    const TireDebugInfo& getTireDebugInfo() const { return m_tireDebug; }

private:
    // Kills the car (and zeroes velocity) if any corner leaves the mask.
    void applyCollision();

    void updateSensors();

    CarParams m_params;
    const Track& m_track;

    b2WorldId m_worldId = b2_nullWorldId;
    b2BodyId m_bodyId = b2_nullBodyId;

    // Mirrors of the Box2D body's pose/velocity, refreshed each update()/reset().
    Vector2 m_position = {0.0f, 0.0f};
    Vector2 m_velocity = {0.0f, 0.0f};
    float m_heading = 0.0f;
    bool m_alive = true;

    std::array<SensorReading, kSensorCount> m_sensors{};
    TireDebugInfo m_tireDebug{};

    // Persistent per-axle EFFECTIVE (relaxed) slip angle, radians, relaxing
    // toward the axle's true instantaneous slip angle every substep -- see
    // CarParams::frontTireRelaxationTime's comment. The actually-applied
    // lateral force is the tire curve evaluated at this angle, recomputed
    // fresh every substep (never itself stored/relaxed). Zeroed on reset()
    // so a fresh spawn never inherits leftover deflection from a prior run.
    float m_frontSlipAngleRelaxed = 0.0f;
    float m_rearSlipAngleRelaxed = 0.0f;
};

} // namespace simulation
