#include "simulation/Car.h"

#include <algorithm>
#include <cmath>

namespace simulation
{

namespace
{

// 1 Box2D unit == 1px, no separate scale factor (Box2D v3 has no baked-in
// "meter-sized" assumption). Box2D's OWN maximumLinearSpeed default is
// 400 px/s here -- well BELOW maxSpeed/the safety clamp below -- so
// b2WorldDef::maximumLinearSpeed is explicitly raised past both in the Car
// constructor; this comment previously (incorrectly) assumed the default
// was already high enough, which silently capped every car at 400px/s
// regardless of maxSpeed/engineForce/tire tuning.
b2Vec2 toB2(Vector2 v)
{
    return b2Vec2{v.x, v.y};
}

Vector2 toRaylib(b2Vec2 v)
{
    return Vector2{v.x, v.y};
}

// Defensive-only ceiling above the analytic steady-state top speed
// (maxSpeed), guarding against velocity growing unbounded in a
// pathological state (e.g. a very large dt); ordinary driving never nears it.
constexpr float kSafetySpeedMultiplier = 1.3f;

// Below this axle speed, slip angle is forced to 0 instead of atan2()'d --
// near zero speed the angle is dominated by float noise, which a large
// cornering stiffness would turn into a real wrong-direction force spike.
// Also gives "steering has no effect while stationary" for free.
constexpr float kMinAxleSpeedForSlip = 0.5f; // px/s

// Below this axle forward speed, tire force ramps linearly to 0: a linear
// slip-angle model is only valid at reasonable rolling speed -- near zero,
// even tiny lateral velocity drives the vy/vx ratio (and so slip angle)
// toward +-90 deg, saturating force from physically insignificant motion.
// This (not kMinAxleSpeedForSlip, which only guards the angle itself) is
// what prevents heading-snapping off a standing start.
constexpr float kTireForceRampSpeed = 15.0f; // px/s

// Floor that per-axle tire relaxation time constants (CarParams::front/
// rearTireRelaxationTime and front/rearTireReleaseTime) shrink toward as
// axle speed drops below kTireForceRampSpeed (linearly, gated by the same
// ramp fraction) -- a nearly-stopped axle's slip angle is poorly
// conditioned anyway (see kMinAxleSpeedForSlip), so any stored relaxed
// deflection is made to collapse quickly rather than linger at a stale
// value into (or through) a near-stop, which is what let recovering from a
// slide "unfreeze" a stale sideways kick right as the car nearly stopped.
constexpr float kTireRelaxationLowSpeedFloorTime = 0.01f; // seconds

// For small perturbations the tire model acts like a rotational spring on
// yaw rate with stiffness kappa = (frontCorneringStiffness*cgToFrontAxle^2
// + rearCorneringStiffness*cgToRearAxle^2) / forwardSpeed (the tanh curve
// has this same slope at slipAngle=0). Integrating that spring explicitly
// over a full timestep is only stable while kappa*timestep/inertia stays
// below ~2 -- with this model's stiffness/geometry, normal driving speeds
// cross that bound, causing heading to snap/oscillate. Rather than
// softening the tires, forces are recomputed and reapplied kForceSubsteps
// times per update(), each over dt/kForceSubsteps, shrinking the effective
// timestep the bound depends on. 16 keeps the ratio comfortably under 1
// across the model's whole range -- independent of Box2D's own
// kPhysicsSubStepCount (constraint solving, not force recomputation).
constexpr int kForceSubsteps = 16;

b2BodyId createCarBody(b2WorldId worldId, const CarParams& params)
{
    b2BodyDef bodyDef = b2DefaultBodyDef();
    bodyDef.type = b2_dynamicBody;
    bodyDef.position = b2Vec2{0.0f, 0.0f};
    bodyDef.rotation = b2MakeRot(0.0f);
    bodyDef.linearDamping = params.linearDamping;
    bodyDef.angularDamping = params.angularDamping;
    bodyDef.gravityScale = 0.0f; // top-down
    bodyDef.enableSleep = true;

    const b2BodyId bodyId = b2CreateBody(worldId, &bodyDef);

    // Forward is local +x, so half-extent x = halfLength, y = halfWidth.
    const b2Polygon box = b2MakeBox(params.length * 0.5f, params.width * 0.5f);

    b2ShapeDef shapeDef = b2DefaultShapeDef();
    shapeDef.density = params.density;
    // Friction/restitution are inert -- this world never has a second body,
    // so density (mass/inertia) is the only field that matters.
    b2CreatePolygonShape(bodyId, &shapeDef, &box);

    // Override the shape-derived rotational inertia with
    // CarParams::rotationalInertia, keeping the shape/density-derived mass
    // and center of mass as-is -- see that field's comment for why yaw
    // inertia is deliberately decoupled from mass/density here.
    b2MassData massData = b2Body_GetMassData(bodyId);
    massData.rotationalInertia = params.rotationalInertia;
    b2Body_SetMassData(bodyId, massData);

    return bodyId;
}

} // namespace

// See the declaration in Car.h for the shape rationale: a rise from 0 with
// slope corneringStiffness, reaching maxForce at the given (now explicit,
// independently-tunable) peakSlipAngle, then a smooth (C1-continuous, zero
// slope at the peak) falloff toward maxForce*slidingGripRatio by 90 degrees
// of slip, staying at that floor beyond. Caller applies the sign (opposing
// the slip direction) and any low-speed ramp.
//
// The rise is a cubic Hermite in normalized slip x = absSlipAngle/
// peakSlipAngle, matching g(0)=0, g(1)=1, g'(0)=s0, g'(1)=0, where
// s0 = corneringStiffness*peakSlipAngle/maxForce is the initial slope
// EXPRESSED in these normalized units. Requiring a specific peakSlipAngle
// independent of corneringStiffness/maxForce (rather than the earlier
// peakSlipAngle = 2*maxForce/corneringStiffness, which ties peak location
// and on-center slope together) needs a curve whose slope isn't simply
// decreasing from corneringStiffness the whole way to the peak -- when
// peakSlipAngle is small relative to maxForce/corneringStiffness (i.e.
// s0 < 2), the curve must accelerate ABOVE its initial slope partway
// through before easing into the peak, which a plain quadratic (s0 fixed
// at 2) can't express. s0=2 reproduces that exact old quadratic as a
// special case. s0 is clamped well inside the range that keeps this cubic
// monotonic on [0,1] (no overshoot past maxForce before the peak).
float tireLateralForceMagnitude(float absSlipAngle, float corneringStiffness, float maxForce, float slidingGripRatio,
                                 float peakSlipAngle)
{
    if (maxForce <= 0.0f || peakSlipAngle <= 0.0f)
    {
        return 0.0f;
    }

    if (absSlipAngle <= peakSlipAngle)
    {
        const float x = std::clamp(absSlipAngle / peakSlipAngle, 0.0f, 1.0f);
        const float s0 = std::clamp(corneringStiffness * peakSlipAngle / maxForce, 0.0f, 2.9f);
        const float g = s0 * x + (3.0f - 2.0f * s0) * x * x + (s0 - 2.0f) * x * x * x;
        return maxForce * g;
    }

    // Post-peak: falls from maxForce toward maxForce*slidingGripRatio as
    // slip goes from peakSlipAngle to 90 degrees (a fully sideways tire),
    // via a smoothstep so the transition is C1-continuous with the ramp
    // above (both have zero slope exactly at the peak -- a genuine smooth
    // maximum, not a corner).
    constexpr float kFullSlideAngle = 1.5707963f; // 90 degrees, in radians
    const float span = std::max(kFullSlideAngle - peakSlipAngle, 1e-4f);
    const float u = std::clamp((absSlipAngle - peakSlipAngle) / span, 0.0f, 1.0f);
    const float smooth = u * u * (3.0f - 2.0f * u);
    return maxForce * (1.0f - (1.0f - slidingGripRatio) * smooth);
}

Car::Car(const CarParams& params, const Track& track)
    : m_params(params)
    , m_track(track)
{
    b2WorldDef worldDef = b2DefaultWorldDef();
    worldDef.gravity = b2Vec2{0.0f, 0.0f}; // top-down: no "down"
    // Box2D's own default maximumLinearSpeed is 400 px/s (see toB2's
    // comment) -- below this car's safety clamp (kSafetySpeedMultiplier *
    // maxSpeed), so it would silently override that clamp as the real
    // ceiling. Raised well past it so the explicit, documented safety clamp
    // in update() -- not an undocumented engine default -- is what actually
    // governs top speed.
    worldDef.maximumLinearSpeed = params.maxSpeed * kSafetySpeedMultiplier * 2.0f;
    m_worldId = b2CreateWorld(&worldDef);
    m_bodyId = createCarBody(m_worldId, m_params);
}

Car::~Car()
{
    // Destroying the world destroys every body in it; no-op for a moved-from Car.
    if (B2_IS_NON_NULL(m_worldId))
    {
        b2DestroyWorld(m_worldId);
    }
}

Car::Car(Car&& other) noexcept
    : m_params(other.m_params)
    , m_track(other.m_track)
    , m_worldId(other.m_worldId)
    , m_bodyId(other.m_bodyId)
    , m_position(other.m_position)
    , m_velocity(other.m_velocity)
    , m_heading(other.m_heading)
    , m_angularVelocity(other.m_angularVelocity)
    , m_alive(other.m_alive)
    , m_sensors(other.m_sensors)
    , m_tireDebug(other.m_tireDebug)
{
    other.m_worldId = b2_nullWorldId;
    other.m_bodyId = b2_nullBodyId;
}

float steeringAuthorityForSpeed(const CarParams& params, float speed)
{
    const auto& speeds = params.steerAuthoritySpeeds;
    const auto& factors = params.steerAuthorityFactors;
    constexpr int kLast = CarParams::kSteerAuthorityPointCount - 1;

    // Written as !(speed > first) so a NaN speed also lands on the first
    // (full-authority) factor rather than propagating.
    if (!(speed > speeds[0]))
    {
        return factors[0];
    }
    for (int i = 1; i <= kLast; ++i)
    {
        if (speed < speeds[i])
        {
            const float t = (speed - speeds[i - 1]) / (speeds[i] - speeds[i - 1]);
            return factors[i - 1] + (factors[i] - factors[i - 1]) * t;
        }
    }
    return factors[kLast];
}

void Car::reset(Vector2 spawnPosition, float spawnHeading)
{
    b2Body_SetTransform(m_bodyId, toB2(spawnPosition), b2MakeRot(spawnHeading));
    b2Body_SetLinearVelocity(m_bodyId, b2Vec2{0.0f, 0.0f});
    b2Body_SetAngularVelocity(m_bodyId, 0.0f);

    m_position = spawnPosition;
    m_velocity = {0.0f, 0.0f};
    m_heading = spawnHeading;
    m_angularVelocity = 0.0f;
    m_alive = true;
    m_tireDebug = TireDebugInfo{};
    m_frontSlipAngleRelaxed = 0.0f;
    m_rearSlipAngleRelaxed = 0.0f;
    m_currentSteerAngle = 0.0f;
    updateSensors();
}

void Car::update(const CarInput& input, float dt)
{
    if (!m_alive)
    {
        return;
    }

    const float throttle = std::clamp(input.throttle, 0.0f, 1.0f);
    const float steering = std::clamp(input.steering, -1.0f, 1.0f);
    const float brake = std::clamp(input.brake, 0.0f, 1.0f);

    // Steering RATE limit: the ACTUAL front-wheel angle moves toward the
    // commanded target at up to maxSteerRateRadPerSec, modeling a finite-
    // response steering system rather than a wheel that can snap between
    // any two angles within a single simulation step (see
    // CarParams::maxSteerRateRadPerSec's comment for why/how this value was
    // chosen). The steering COMMAND itself (`steering` above, and
    // everything upstream of it -- CarInput, AIController, the network
    // output) is completely untouched by this; only m_currentSteerAngle,
    // the physical wheel's own persistent state, is rate-limited. Computed
    // once per update() (not per force-substep below) since it's advancing
    // over the full dt, not a per-substep quantity -- same granularity the
    // old direct assignment already had (steerAngle was likewise computed
    // once and reused across every substep).
    //
    // The TARGET angle scales with speed-sensitive steering authority (see
    // CarParams::steerAuthoritySpeeds): the same +-1 command targets a smaller
    // wheel angle the faster the car is going. Speed is the car's speed at the
    // start of this update() (the previous step's result), a deterministic,
    // already-stored value. The rate limiter below is unchanged and simply
    // moves the wheel toward this target.
    const float steeringAuthority = steeringAuthorityForSpeed(m_params, getSpeed());
    const float effectiveMaxSteerAngle = m_params.maxSteerAngle * steeringAuthority;
    m_tireDebug.steeringAuthority = steeringAuthority;
    m_tireDebug.effectiveMaxSteerAngle = effectiveMaxSteerAngle;
    const float targetSteerAngle = steering * effectiveMaxSteerAngle;
    const float maxSteerDelta = m_params.maxSteerRateRadPerSec * dt;
    const float steerDelta = std::clamp(targetSteerAngle - m_currentSteerAngle, -maxSteerDelta, maxSteerDelta);
    m_currentSteerAngle += steerDelta;
    const float steerAngle = m_currentSteerAngle;

    const float cosSteer = std::cos(steerAngle);
    const float sinSteer = std::sin(steerAngle);

    // Forces recomputed/reapplied kForceSubsteps times per update() for
    // numerical stability (see kForceSubsteps).
    const float subDt = dt / static_cast<float>(kForceSubsteps);

    for (int sub = 0; sub < kForceSubsteps; ++sub)
    {
        const b2Rot rotation = b2Body_GetRotation(m_bodyId);
        const Vector2 forward = {rotation.c, rotation.s};
        const Vector2 right = {-forward.y, forward.x};

        const b2Vec2 comPosition = b2Body_GetPosition(m_bodyId);
        const b2Vec2 comVelocity = b2Body_GetLinearVelocity(m_bodyId);
        const float yawRate = b2Body_GetAngularVelocity(m_bodyId);

        const float speed = std::sqrt(comVelocity.x * comVelocity.x + comVelocity.y * comVelocity.y);
        // Body-frame velocity of the center of mass.
        const float vxBody = comVelocity.x * forward.x + comVelocity.y * forward.y;
        const float vyBody = comVelocity.x * right.x + comVelocity.y * right.y;

        // Longitudinal resistance, applied at the center of mass (drive
        // force is applied at the rear axle below).
        Vector2 centerForce = {0.0f, 0.0f};

        // Rolling resistance opposes forward-rolling motion only; lateral
        // resistance is the tire model's job below.
        const float rollingResistanceForce = m_params.rollingResistance * vxBody;
        centerForce.x -= forward.x * rollingResistanceForce;
        centerForce.y -= forward.y * rollingResistanceForce;

        // Aerodynamic-like drag, quadratic in speed, opposes full velocity.
        const float dragMagnitude = m_params.dragCoefficient * speed;
        centerForce.x -= comVelocity.x * dragMagnitude;
        centerForce.y -= comVelocity.y * dragMagnitude;

        b2Body_ApplyForceToCenter(m_bodyId, toB2(centerForce), true);

        // Lateral: single-track (bicycle model) front/rear tire forces.
        // Axle velocities via v_axle = v_com + yawRate x r_axle, in the
        // body frame (r_axle is along the forward axis for both axles):
        //   front: vx = vxBody,  vy = vyBody + cgToFrontAxle * yawRate
        //   rear:  vx = vxBody,  vy = vyBody - cgToRearAxle  * yawRate
        const float vyFrontBody = vyBody + m_params.cgToFrontAxle * yawRate;
        const float vyRearBody = vyBody - m_params.cgToRearAxle * yawRate;

        // Front wheel is rotated by steerAngle relative to the body, so its
        // slip angle needs velocity in the WHEEL frame: rotate by -steerAngle.
        const float vxFrontWheel = vxBody * cosSteer + vyFrontBody * sinSteer;
        const float vyFrontWheel = -vxBody * sinSteer + vyFrontBody * cosSteer;

        // Slip angle = atan2(lateral, forward) in the wheel's own frame,
        // gated on axle speed (see kMinAxleSpeedForSlip).
        const float frontAxleSpeed = std::sqrt(vxFrontWheel * vxFrontWheel + vyFrontWheel * vyFrontWheel);
        const float frontSlipAngle =
            (frontAxleSpeed > kMinAxleSpeedForSlip) ? std::atan2(vyFrontWheel, vxFrontWheel) : 0.0f;

        const float rearAxleSpeed = std::sqrt(vxBody * vxBody + vyRearBody * vyRearBody);
        const float rearSlipAngle = (rearAxleSpeed > kMinAxleSpeedForSlip) ? std::atan2(vyRearBody, vxBody) : 0.0f;

        const float frontSpeedRamp = std::clamp(std::fabs(vxFrontWheel) / kTireForceRampSpeed, 0.0f, 1.0f);
        const float rearSpeedRamp = std::clamp(std::fabs(vxBody) / kTireForceRampSpeed, 0.0f, 1.0f);

        // Relax the EFFECTIVE slip angle toward the true (instantaneous)
        // one -- see CarParams::frontTireRelaxationTime's comment for why
        // this replaced relaxing the force directly. Build-up uses the
        // normal (slower) time constant; UNLOADING uses the (faster)
        // release constant instead, so a stale deflection can't linger
        // through a slide's recovery.
        //
        // "Unloading" is |trueSlipAngle| < |storedAngle| -- the tire's
        // current demand is smaller than what's still stored, whether or
        // not the true angle has actually crossed zero yet. An earlier
        // version gated release on a SIGN reversal alone, which missed the
        // far more common case of true slip angle collapsing back toward
        // zero WITHOUT crossing it (e.g. true -14deg -> -2deg over a few
        // frames while, at the old build rate, stored stayed near -13 to
        // -17deg the whole time): the relaxed angle -- and so the applied
        // force -- stayed pinned near its old peak for several frames after
        // the real geometry had already relieved, sustaining a large, by
        // then stale restoring force right as the car was recovering. That
        // stale-but-same-sign force is what was driving the sustained
        // high-speed full-lock yaw oscillation (see the wobble
        // investigation) -- this is the same "stored force must not outlive
        // the demand that created it" principle as the sign-reversal fix,
        // just applied to magnitude as well as sign. Both directions
        // further shrink toward kTireRelaxationLowSpeedFloorTime as axle
        // speed drops (same ramp fraction as the force ramp below).
        auto relaxSlipAngle = [&](float trueSlipAngle, float storedAngle, float speedRamp, float buildTime, float releaseTime) -> float
        {
            const bool unloading = std::fabs(trueSlipAngle) < std::fabs(storedAngle);
            const float baseTime = unloading ? releaseTime : buildTime;
            const float effectiveTime = std::max(kTireRelaxationLowSpeedFloorTime + (baseTime - kTireRelaxationLowSpeedFloorTime) * speedRamp,
                                                  kTireRelaxationLowSpeedFloorTime);
            const float rate = std::clamp(subDt / std::max(effectiveTime, 1e-4f), 0.0f, 1.0f);
            return storedAngle + (trueSlipAngle - storedAngle) * rate;
        };

        m_frontSlipAngleRelaxed = relaxSlipAngle(frontSlipAngle, m_frontSlipAngleRelaxed, frontSpeedRamp,
                                                  m_params.frontTireRelaxationTime, m_params.frontTireReleaseTime);
        m_rearSlipAngleRelaxed = relaxSlipAngle(rearSlipAngle, m_rearSlipAngleRelaxed, rearSpeedRamp,
                                                 m_params.rearTireRelaxationTime, m_params.rearTireReleaseTime);

        // The actually-applied lateral force is the (memoryless) tire curve
        // evaluated at the relaxed angle, re-gated by the CURRENT low-speed
        // ramp every substep (never itself stored/relaxed) -- so even if
        // the relaxed angle still holds a meaningful value, the output
        // force still collapses toward zero as axle speed drops, rather
        // than staying available as stored force would.
        const float frontLateralForceDesired =
            -std::copysign(tireLateralForceMagnitude(std::fabs(m_frontSlipAngleRelaxed), m_params.frontCorneringStiffness,
                                                       m_params.frontMaxTireForce, m_params.frontSlidingGripRatio,
                                                       m_params.frontPeakSlipAngle),
                            m_frontSlipAngleRelaxed) *
            frontSpeedRamp;
        const float rearLateralForceDesired =
            -std::copysign(tireLateralForceMagnitude(std::fabs(m_rearSlipAngleRelaxed), m_params.rearCorneringStiffness,
                                                       m_params.rearMaxTireForce, m_params.rearSlidingGripRatio,
                                                       m_params.rearPeakSlipAngle),
                            m_rearSlipAngleRelaxed) *
            rearSpeedRamp;

        // Unrelaxed curve target, for debug/telemetry only (TireDebugInfo::
        // front/rearTargetForceY) -- never used to drive the applied force.
        const float frontLateralForceTarget =
            -std::copysign(tireLateralForceMagnitude(std::fabs(frontSlipAngle), m_params.frontCorneringStiffness,
                                                       m_params.frontMaxTireForce, m_params.frontSlidingGripRatio,
                                                       m_params.frontPeakSlipAngle),
                            frontSlipAngle) *
            frontSpeedRamp;
        const float rearLateralForceTarget =
            -std::copysign(tireLateralForceMagnitude(std::fabs(rearSlipAngle), m_params.rearCorneringStiffness,
                                                       m_params.rearMaxTireForce, m_params.rearSlidingGripRatio,
                                                       m_params.rearPeakSlipAngle),
                            rearSlipAngle) *
            rearSpeedRamp;

        // Braking: a longitudinal force opposing each axle's OWN current
        // rolling direction (never a fixed world/body direction), split
        // front/rear by frontBrakeBias. Reuses each axle's own speed ramp
        // (frontSpeedRamp/rearSpeedRamp) -- at rest there is nothing to
        // brake against, so the force (and any sign ambiguity from
        // copysign at exactly zero speed) vanishes with it, same as the
        // lateral tire curve above.
        const float rearBrakeForceDesired =
            -std::copysign(m_params.maxBrakeForce * (1.0f - m_params.frontBrakeBias) * brake * rearSpeedRamp, vxBody);
        const float frontBrakeForceDesired =
            -std::copysign(m_params.maxBrakeForce * m_params.frontBrakeBias * brake * frontSpeedRamp, vxFrontWheel);

        // Rear-wheel drive + brake + friction circle: desired drive force,
        // desired brake force, and desired lateral force combine into one
        // vector constrained to sqrt(FxRear^2+FyRear^2) <= rearMaxTireForce
        // -- exceeding the budget scales every component down together, so
        // heavy throttle or heavy braking mid-corner both reduce available
        // lateral grip (power oversteer / trail-braking oversteer alike).
        const float rearDriveForceDesired = m_params.engineForce * throttle;
        const float rearForceXDesired = rearDriveForceDesired + rearBrakeForceDesired;
        const float rearCombinedMagnitude =
            std::sqrt(rearForceXDesired * rearForceXDesired + rearLateralForceDesired * rearLateralForceDesired);

        float rearForceX = rearForceXDesired;
        float rearForceY = rearLateralForceDesired;
        if (rearCombinedMagnitude > m_params.rearMaxTireForce)
        {
            const float scale = m_params.rearMaxTireForce / rearCombinedMagnitude;
            rearForceX *= scale;
            rearForceY *= scale;
        }

        // Front brake + friction circle: the front axle never drives, but
        // now can brake, so its desired brake force and desired lateral
        // force combine under the same sqrt(FxFront^2+FyFront^2) <=
        // frontMaxTireForce constraint -- braking hard into a corner
        // genuinely competes with the front's cornering grip.
        const float frontCombinedMagnitude =
            std::sqrt(frontBrakeForceDesired * frontBrakeForceDesired + frontLateralForceDesired * frontLateralForceDesired);
        float frontForceX = frontBrakeForceDesired;
        float frontForceY = frontLateralForceDesired;
        if (frontCombinedMagnitude > m_params.frontMaxTireForce)
        {
            const float scale = m_params.frontMaxTireForce / frontCombinedMagnitude;
            frontForceX *= scale;
            frontForceY *= scale;
        }

        // Apply each force at its own axle's world position -- Box2D
        // derives torque from the offset from center of mass, so yaw is a
        // genuine consequence of these moment arms, never commanded
        // directly. Front force is in the steered wheel frame (X = wheel
        // rolling direction, Y = wheel lateral); rear in the body frame (it
        // never steers).
        const Vector2 wheelForward = {forward.x * cosSteer + right.x * sinSteer, forward.y * cosSteer + right.y * sinSteer};
        const Vector2 wheelRight = {-forward.x * sinSteer + right.x * cosSteer, -forward.y * sinSteer + right.y * cosSteer};
        const Vector2 frontForce = {wheelForward.x * frontForceX + wheelRight.x * frontForceY,
                                     wheelForward.y * frontForceX + wheelRight.y * frontForceY};
        const Vector2 rearForce = {forward.x * rearForceX + right.x * rearForceY, forward.y * rearForceX + right.y * rearForceY};

        const b2Vec2 frontAxlePosition = {comPosition.x + forward.x * m_params.cgToFrontAxle,
                                           comPosition.y + forward.y * m_params.cgToFrontAxle};
        const b2Vec2 rearAxlePosition = {comPosition.x - forward.x * m_params.cgToRearAxle,
                                          comPosition.y - forward.y * m_params.cgToRearAxle};

        b2Body_ApplyForce(m_bodyId, toB2(frontForce), frontAxlePosition, true);
        b2Body_ApplyForce(m_bodyId, toB2(rearForce), rearAxlePosition, true);

        m_tireDebug.steeringInput = steering;
        m_tireDebug.throttleInput = throttle;
        m_tireDebug.brakeInput = brake;
        m_tireDebug.steeringAngle = steerAngle;
        m_tireDebug.yawRate = yawRate;
        m_tireDebug.frontSlipAngle = frontSlipAngle;
        m_tireDebug.rearSlipAngle = rearSlipAngle;
        m_tireDebug.frontSlipAngleRelaxed = m_frontSlipAngleRelaxed;
        m_tireDebug.rearSlipAngleRelaxed = m_rearSlipAngleRelaxed;
        m_tireDebug.frontForceX = frontForceX;
        m_tireDebug.frontForceY = frontForceY;
        m_tireDebug.rearForceX = rearForceX;
        m_tireDebug.rearForceY = rearForceY;
        m_tireDebug.frontTargetForceY = frontLateralForceTarget;
        m_tireDebug.rearTargetForceY = rearLateralForceTarget;
        m_tireDebug.frontAxleVx = vxFrontWheel;
        m_tireDebug.frontAxleVy = vyFrontWheel;
        m_tireDebug.rearAxleVx = vxBody;
        m_tireDebug.rearAxleVy = vyRearBody;
        m_tireDebug.frontGripUtilization =
            std::sqrt(frontForceX * frontForceX + frontForceY * frontForceY) / std::max(m_params.frontMaxTireForce, 1e-4f);
        m_tireDebug.rearGripUtilization =
            std::sqrt(rearForceX * rearForceX + rearForceY * rearForceY) / std::max(m_params.rearMaxTireForce, 1e-4f);

        b2World_Step(m_worldId, subDt, kPhysicsSubStepCount);

        // Defensive safety clamp only (see kSafetySpeedMultiplier).
        const b2Vec2 postStepVelocity = b2Body_GetLinearVelocity(m_bodyId);
        const float postStepSpeed =
            std::sqrt(postStepVelocity.x * postStepVelocity.x + postStepVelocity.y * postStepVelocity.y);
        const float safetyLimit = m_params.maxSpeed * kSafetySpeedMultiplier;
        if (postStepSpeed > safetyLimit)
        {
            const float scale = safetyLimit / postStepSpeed;
            b2Body_SetLinearVelocity(m_bodyId, b2Vec2{postStepVelocity.x * scale, postStepVelocity.y * scale});
        }

        // Re-derive mirror state and check collision every substep, not
        // just at the end, so leaving the road mid-update() is caught immediately.
        m_position = toRaylib(b2Body_GetPosition(m_bodyId));
        m_heading = b2Rot_GetAngle(b2Body_GetRotation(m_bodyId));
        m_velocity = toRaylib(b2Body_GetLinearVelocity(m_bodyId));
        m_angularVelocity = b2Body_GetAngularVelocity(m_bodyId);

        applyCollision();
        if (!m_alive)
        {
            break;
        }
    }

    // Cast once per update() regardless of how the loop ended, including a
    // final cast from the death pose if this call just killed the car.
    updateSensors();
}

std::array<Vector2, 4> Car::getCorners() const
{
    const Vector2 forward = {std::cos(m_heading), std::sin(m_heading)};
    const Vector2 right = {-std::sin(m_heading), std::cos(m_heading)};

    const float halfLength = m_params.length * 0.5f;
    const float halfWidth = m_params.width * 0.5f;

    const Vector2 forwardOffset = {forward.x * halfLength, forward.y * halfLength};
    const Vector2 rightOffset = {right.x * halfWidth, right.y * halfWidth};

    return {
        Vector2{m_position.x + forwardOffset.x + rightOffset.x, m_position.y + forwardOffset.y + rightOffset.y},
        Vector2{m_position.x + forwardOffset.x - rightOffset.x, m_position.y + forwardOffset.y - rightOffset.y},
        Vector2{m_position.x - forwardOffset.x - rightOffset.x, m_position.y - forwardOffset.y - rightOffset.y},
        Vector2{m_position.x - forwardOffset.x + rightOffset.x, m_position.y - forwardOffset.y + rightOffset.y},
    };
}

void Car::applyCollision()
{
    for (const Vector2& corner : getCorners())
    {
        const int x = static_cast<int>(std::lround(corner.x));
        const int y = static_cast<int>(std::lround(corner.y));

        if (!m_track.isDrivable(x, y))
        {
            m_alive = false;
            m_velocity = {0.0f, 0.0f};
            m_angularVelocity = 0.0f;
            b2Body_SetLinearVelocity(m_bodyId, b2Vec2{0.0f, 0.0f});
            b2Body_SetAngularVelocity(m_bodyId, 0.0f);
            return;
        }
    }
}

Vector2 Car::getSensorOrigin() const
{
    const Vector2 forward = {std::cos(m_heading), std::sin(m_heading)};
    const float halfLength = m_params.length * 0.5f;
    return {m_position.x + forward.x * halfLength, m_position.y + forward.y * halfLength};
}

void Car::updateSensors()
{
    const Vector2 origin = getSensorOrigin();

    for (int i = 0; i < kSensorCount; ++i)
    {
        const float angle = m_heading + kSensorAngleDegrees[i] * DEG2RAD;
        const Vector2 direction = {std::cos(angle), std::sin(angle)};

        // Default to "nothing found": full range, in the ray's direction.
        float distance = kMaxSensorDistance;
        Vector2 endPoint = {origin.x + direction.x * kMaxSensorDistance, origin.y + direction.y * kMaxSensorDistance};

        for (float d = 0.0f; d <= kMaxSensorDistance; d += kSensorStep)
        {
            const Vector2 sample = {origin.x + direction.x * d, origin.y + direction.y * d};
            const int mx = static_cast<int>(std::lround(sample.x));
            const int my = static_cast<int>(std::lround(sample.y));

            if (!m_track.isDrivable(mx, my))
            {
                distance = d;
                endPoint = sample;
                break;
            }
        }

        m_sensors[i].distance = distance;
        m_sensors[i].normalizedDistance = distance / kMaxSensorDistance;
        m_sensors[i].endPoint = endPoint;
    }
}

float Car::getSpeed() const
{
    return std::sqrt(m_velocity.x * m_velocity.x + m_velocity.y * m_velocity.y);
}

float Car::getForwardVelocity() const
{
    const Vector2 forward = {std::cos(m_heading), std::sin(m_heading)};
    return m_velocity.x * forward.x + m_velocity.y * forward.y;
}

float Car::getLateralVelocity() const
{
    const Vector2 right = {-std::sin(m_heading), std::cos(m_heading)};
    return m_velocity.x * right.x + m_velocity.y * right.y;
}

float Car::getSlipAngle() const
{
    // Below this speed, velocity direction is dominated by float noise.
    constexpr float kMinSpeedForSlipAngle = 1.0f; // px/s

    if (getSpeed() < kMinSpeedForSlipAngle)
    {
        return 0.0f;
    }

    return std::atan2(getLateralVelocity(), getForwardVelocity());
}

} // namespace simulation
