#include "simulation/Car.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace simulation
{

namespace
{

// Unit convention: Box2D units are treated as identical to the
// simulation's existing pixel units -- 1 Box2D unit == 1px, no separate
// px-to-meters scale factor. Box2D v3 (unlike v2.x) has no baked-in
// assumption that bodies are "meter-sized"; the only real constraint is
// b2WorldDef::maximumLinearSpeed (default 400 units/s), comfortably above
// CarParams::maxSpeed (260px/s) with headroom for the one defensive safety
// clamp below, so the default is left untouched.
b2Vec2 toB2(Vector2 v)
{
    return b2Vec2{v.x, v.y};
}

Vector2 toRaylib(b2Vec2 v)
{
    return Vector2{v.x, v.y};
}

// Defensive-only velocity ceiling: CarParams::maxSpeed is an approximate
// analytic steady-state top speed reference (engine force == total
// resistance force under sustained full throttle -- see
// CarParams::rollingResistance's comment), never enforced by clamping
// every update(). This multiplier exists solely so a pathological state
// (e.g. a very large dt, or some future change that lets forces stack
// unexpectedly) cannot let velocity grow without bound; ordinary driving
// never approaches it.
constexpr float kSafetySpeedMultiplier = 1.3f;

// Below this axle speed (in the axle/wheel's own local frame), a tire
// slip angle is forced to exactly 0 instead of computed via atan2 --
// atan2's angle is dominated by float noise direction when both of its
// arguments are near zero (a near-stationary wheel has no meaningful
// "slip direction" at all), and a large cornering stiffness would turn
// that noise into a real, wrong-direction force spike. This also gives
// "steering has no effect while stationary" for free: with both axle
// speeds near zero, both tire forces are exactly 0, so no yaw torque is
// ever generated at rest.
constexpr float kMinAxleSpeedForSlip = 0.5f; // px/s

// A linear tire model's slip angle is only a valid small-signal
// approximation for a wheel that is actually rolling at a reasonable
// speed. atan2() itself never diverges, but as a wheel's own forward
// (rolling) speed approaches zero, even a physically tiny lateral
// velocity produces a slip angle approaching +-90 degrees (since the
// angle depends only on the vy/vx *ratio*, not on how small both are) --
// saturating the tire force to its maximum, applied through a real moment
// arm, from what is actually a physically insignificant motion. Below
// kTireForceRampSpeed, each axle's tire force is scaled down linearly
// toward 0 as that axle's own forward speed approaches 0, so a
// near-stationary (or just-starting-to-move) wheel cannot produce a large
// force/torque impulse from float-noise-scale lateral motion -- this is
// what actually fixes "steering has no effect while stationary" and
// prevents heading-snapping right off a standing start; kMinAxleSpeedForSlip
// above only guards the angle computation itself against a literal
// zero-speed division, it does not bound the force this ramp bounds.
constexpr float kTireForceRampSpeed = 15.0f; // px/s

// The tire model acts, for small perturbations, exactly like a rotational
// spring pulling yaw rate back toward equilibrium, with effective "spring
// constant"
// kappa = (frontCorneringStiffness*cgToFrontAxle^2 +
//          rearCorneringStiffness*cgToRearAxle^2) / forwardSpeed
// (see the derivation in Car::update()'s comment) -- Stage 20.3's
// tanh-based curve (see below) has this exact same slope at slipAngle = 0
// by construction, so this analysis and the substep count it justifies are
// unchanged by that curve replacement. Applying that spring
// force EXPLICITLY -- computed once from the current velocity, then held
// fixed while Box2D integrates over a full timestep, exactly like the
// engine/resistance forces above -- is only numerically stable while
// kappa * timestep / rotationalInertia stays below ~2 (the standard
// stability bound for explicit integration of a linear restoring force).
// With this model's actual stiffness/geometry/inertia, that bound is
// crossed well within the speed range normal driving needs (multiple
// hundred px/s), which shows up as the tire force overshooting further
// each frame instead of converging -- a genuine numerical instability
// (heading "snapping"/oscillating even at a constant, moderate speed and
// zero steering), not a low-speed-only edge case. kMinAxleSpeedForSlip and
// kTireForceRampSpeed above handle the SEPARATE near-zero-speed
// degeneracy; this is what keeps the model stable everywhere else. Rather
// than weakening the tire model's stiffness (which would make it feel
// soft/floaty -- exactly what this stage's physics fix is trying to get
// away from), Car::update() recomputes and reapplies the tire forces
// kForceSubsteps times per update() call, each over dt/kForceSubsteps --
// shrinking the effective timestep this stability bound depends on by the
// same factor. 16 keeps kappa*timestep/rotationalInertia comfortably
// under 1 across this model's whole parameter range (see the Stage 20.2
// report for the worked numbers), independent of Box2D's own
// kPhysicsSubStepCount (which governs Box2D's internal constraint solving,
// not how often external forces here are recomputed).
constexpr int kForceSubsteps = 16;

b2BodyId createCarBody(b2WorldId worldId, const CarParams& params)
{
    b2BodyDef bodyDef = b2DefaultBodyDef();
    bodyDef.type = b2_dynamicBody;
    bodyDef.position = b2Vec2{0.0f, 0.0f};
    bodyDef.rotation = b2MakeRot(0.0f);
    bodyDef.linearDamping = params.linearDamping;
    bodyDef.angularDamping = params.angularDamping;
    bodyDef.gravityScale = 0.0f; // top-down: the world itself already has zero gravity (see Car::Car)
    bodyDef.enableSleep = true;

    const b2BodyId bodyId = b2CreateBody(worldId, &bodyDef);

    // Forward is local +x (matches Car's existing forward = (cos, sin)
    // convention at heading 0), so the box's half-extent along x is
    // halfLength (front/back) and along y is halfWidth (left/right).
    const b2Polygon box = b2MakeBox(params.length * 0.5f, params.width * 0.5f);

    b2ShapeDef shapeDef = b2DefaultShapeDef();
    shapeDef.density = params.density;
    // Friction/restitution are inert here: this per-car b2World never
    // contains a second body (see the Car class comment), so this shape
    // never actually generates a contact -- density is the only field that
    // matters, since it drives mass/rotational inertia.
    b2CreatePolygonShape(bodyId, &shapeDef, &box);

    return bodyId;
}

} // namespace

Car::Car(const CarParams& params, const Track& track)
    : m_params(params)
    , m_track(track)
{
    b2WorldDef worldDef = b2DefaultWorldDef();
    worldDef.gravity = b2Vec2{0.0f, 0.0f}; // top-down: no "down"
    m_worldId = b2CreateWorld(&worldDef);
    m_bodyId = createCarBody(m_worldId, m_params);
}

Car::~Car()
{
    // b2DestroyBody is implied by b2DestroyWorld (destroying a world
    // destroys every body in it), so only the world needs an explicit
    // check/destroy here. B2_IS_NULL is false for a moved-from Car (see the
    // move constructor), so this is a no-op in that case, and also a no-op
    // if construction never actually created a world.
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
    , m_alive(other.m_alive)
    , m_sensors(other.m_sensors)
    , m_tireDebug(other.m_tireDebug)
{
    // Leave `other` holding null ids so its destructor becomes a no-op --
    // ownership of the Box2D world/body has moved entirely to *this.
    other.m_worldId = b2_nullWorldId;
    other.m_bodyId = b2_nullBodyId;
}

void Car::reset(Vector2 spawnPosition, float spawnHeading)
{
    b2Body_SetTransform(m_bodyId, toB2(spawnPosition), b2MakeRot(spawnHeading));
    b2Body_SetLinearVelocity(m_bodyId, b2Vec2{0.0f, 0.0f});
    b2Body_SetAngularVelocity(m_bodyId, 0.0f);

    m_position = spawnPosition;
    m_velocity = {0.0f, 0.0f};
    m_heading = spawnHeading;
    m_alive = true;
    m_tireDebug = TireDebugInfo{};
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
    const float steerAngle = steering * m_params.maxSteerAngle;
    const float cosSteer = std::cos(steerAngle);
    const float sinSteer = std::sin(steerAngle);

    // See kForceSubsteps' comment: the tire model's forces are recomputed
    // and reapplied kForceSubsteps times, each over dt/kForceSubsteps, for
    // numerical stability -- everything from reading Box2D's current state
    // through stepping the world lives inside this loop. This is an
    // internal detail of this one update() call only: dt itself is always
    // whatever fixed timestep the caller passed in (see main.cpp's
    // kSimulationDt), never altered or made FPS-dependent.
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

        // ---- Longitudinal resistance only (engine/drive force moved to
        // the rear axle in Stage 20.3 -- see below) -- applied at the
        // center of mass, same as Stage 20.1/20.2 ----
        Vector2 centerForce = {0.0f, 0.0f};

        // Rolling resistance opposes only the car's own forward-rolling
        // motion; lateral resistance is now entirely the tire model's job
        // below (see CarParams::rollingResistance's comment).
        const float rollingResistanceForce = m_params.rollingResistance * vxBody;
        centerForce.x -= forward.x * rollingResistanceForce;
        centerForce.y -= forward.y * rollingResistanceForce;

        // Aerodynamic-like drag opposes the full velocity vector, quadratic in speed.
        const float dragMagnitude = m_params.dragCoefficient * speed;
        centerForce.x -= comVelocity.x * dragMagnitude;
        centerForce.y -= comVelocity.y * dragMagnitude;

        b2Body_ApplyForceToCenter(m_bodyId, toB2(centerForce), true);

        // ---- Lateral: single-track (bicycle model) front/rear tire forces ----
        //
        // Axle velocities via the standard rigid-body point-velocity
        // relation v_axle = v_com + yawRate x r_axle, expressed directly in
        // the body frame (r_axle is purely along the body's forward axis
        // for both axles, so the cross product's only effect is a lateral
        // term):
        //   front: vx = vxBody,  vy = vyBody + cgToFrontAxle * yawRate
        //   rear:  vx = vxBody,  vy = vyBody - cgToRearAxle  * yawRate
        //
        // Stability note (see kForceSubsteps): linearizing Fy=-C*alpha
        // around a small yawRate perturbation (steering=0, vyBody=0) gives
        // a restoring torque of -((frontCorneringStiffness*cgToFrontAxle^2
        // + rearCorneringStiffness*cgToRearAxle^2) / vxBody) * yawRate --
        // i.e. a rotational spring whose "stiffness" grows without bound as
        // vxBody shrinks. Explicitly integrating a spring force this stiff
        // over too large a timestep overshoots and diverges (the model
        // "snapping"/oscillating) rather than settling -- exactly the
        // failure this stage's substep restructuring fixes.
        const float vyFrontBody = vyBody + m_params.cgToFrontAxle * yawRate;
        const float vyRearBody = vyBody - m_params.cgToRearAxle * yawRate;

        // The front wheel is rotated by steerAngle relative to the body, so
        // its slip angle needs the front axle's velocity expressed in the
        // WHEEL's own frame, not the body frame -- rotate
        // (vxBody, vyFrontBody) by -steerAngle.
        const float vxFrontWheel = vxBody * cosSteer + vyFrontBody * sinSteer;
        const float vyFrontWheel = -vxBody * sinSteer + vyFrontBody * cosSteer;

        // Slip angle = angle between a wheel's pointing direction and its
        // actual velocity, in that wheel's own frame -- exactly
        // atan2(lateral, forward) of the wheel-frame velocity. See
        // kMinAxleSpeedForSlip for why this is gated on axle speed rather
        // than computed unconditionally.
        const float frontAxleSpeed = std::sqrt(vxFrontWheel * vxFrontWheel + vyFrontWheel * vyFrontWheel);
        const float frontSlipAngle =
            (frontAxleSpeed > kMinAxleSpeedForSlip) ? std::atan2(vyFrontWheel, vxFrontWheel) : 0.0f;

        const float rearAxleSpeed = std::sqrt(vxBody * vxBody + vyRearBody * vyRearBody);
        const float rearSlipAngle = (rearAxleSpeed > kMinAxleSpeedForSlip) ? std::atan2(vyRearBody, vxBody) : 0.0f;

        // Smooth saturating tire curve (Stage 20.3, replaces Stage 20.2's
        // hard clamp() -- see CarParams::frontCorneringStiffness's
        // comment): Fy = -Fmax * tanh((C/Fmax) * slipAngle). This has slope
        // exactly C at slipAngle = 0 (so corneringStiffness keeps the same
        // physical meaning/tuning feel it always had) and approaches
        // +-Fmax smoothly as slip grows, with no discontinuity at the cap.
        // Each is additionally scaled by that axle's own low-speed ramp
        // factor -- see kTireForceRampSpeed. This is each axle's LATERAL
        // force alone, before the rear axle's friction-circle combination
        // with drive force below.
        const float frontSpeedRamp = std::clamp(std::fabs(vxFrontWheel) / kTireForceRampSpeed, 0.0f, 1.0f);
        const float rearSpeedRamp = std::clamp(std::fabs(vxBody) / kTireForceRampSpeed, 0.0f, 1.0f);

        const float frontLateralForce = -m_params.frontMaxTireForce *
                                         std::tanh((m_params.frontCorneringStiffness / m_params.frontMaxTireForce) * frontSlipAngle) *
                                         frontSpeedRamp;
        const float rearLateralForceDesired = -m_params.rearMaxTireForce *
                                               std::tanh((m_params.rearCorneringStiffness / m_params.rearMaxTireForce) * rearSlipAngle) *
                                               rearSpeedRamp;

        // ---- Rear-wheel drive + rear friction circle (Stage 20.3) ----
        //
        // The rear axle is the only driven axle: its desired longitudinal
        // (drive) force and its desired lateral (cornering) force are
        // computed independently above/below, then combined as one vector
        // and constrained to a single circular grip budget --
        // sqrt(FxRear^2 + FyRear^2) <= rearMaxTireForce -- exactly like a
        // real tire's contact patch, which cannot supply more total force
        // in any direction than its friction limit allows, regardless of
        // how that force is split between accelerating and cornering. When
        // the desired combination exceeds the budget, BOTH components are
        // scaled down by the same factor, so heavy throttle mid-corner
        // genuinely reduces the lateral force actually available -- this
        // is the entire mechanism behind power oversteer here; nothing
        // else in this function reduces rear lateral force based on
        // throttle. The front axle never receives drive force (FxFront =
        // 0), so it never needs this combination: its lateral force from
        // the tanh curve above already can't exceed frontMaxTireForce by
        // construction.
        const float rearDriveForceDesired = m_params.engineForce * throttle;
        const float rearCombinedMagnitude =
            std::sqrt(rearDriveForceDesired * rearDriveForceDesired + rearLateralForceDesired * rearLateralForceDesired);

        float rearForceX = rearDriveForceDesired;
        float rearForceY = rearLateralForceDesired;
        if (rearCombinedMagnitude > m_params.rearMaxTireForce)
        {
            const float scale = m_params.rearMaxTireForce / rearCombinedMagnitude;
            rearForceX *= scale;
            rearForceY *= scale;
        }

        // Apply each force at its own axle's WORLD position via Box2D's
        // point-force API -- Box2D derives the resulting torque from each
        // force's offset from the center of mass automatically, so yaw is
        // a genuine physical consequence of these forces' moment arms.
        // This is the crucial difference from the pre-Stage-20.2 model:
        // there is no separate, independently-commanded torque anywhere in
        // this function. The front force is expressed in the STEERED wheel
        // frame (wheelRight); the rear axle never steers, so its force is
        // expressed directly in the body frame (forward/right).
        const Vector2 wheelRight = {-forward.x * sinSteer + right.x * cosSteer, -forward.y * sinSteer + right.y * cosSteer};
        const Vector2 frontForce = {wheelRight.x * frontLateralForce, wheelRight.y * frontLateralForce};
        const Vector2 rearForce = {forward.x * rearForceX + right.x * rearForceY, forward.y * rearForceX + right.y * rearForceY};

        const b2Vec2 frontAxlePosition = {comPosition.x + forward.x * m_params.cgToFrontAxle,
                                           comPosition.y + forward.y * m_params.cgToFrontAxle};
        const b2Vec2 rearAxlePosition = {comPosition.x - forward.x * m_params.cgToRearAxle,
                                          comPosition.y - forward.y * m_params.cgToRearAxle};

        b2Body_ApplyForce(m_bodyId, toB2(frontForce), frontAxlePosition, true);
        b2Body_ApplyForce(m_bodyId, toB2(rearForce), rearAxlePosition, true);

        m_tireDebug.steeringAngle = steerAngle;
        m_tireDebug.yawRate = yawRate;
        m_tireDebug.frontSlipAngle = frontSlipAngle;
        m_tireDebug.rearSlipAngle = rearSlipAngle;
        m_tireDebug.frontForceX = 0.0f;
        m_tireDebug.frontForceY = frontLateralForce;
        m_tireDebug.rearForceX = rearForceX;
        m_tireDebug.rearForceY = rearForceY;
        m_tireDebug.frontGripUtilization =
            std::fabs(frontLateralForce) / std::max(m_params.frontMaxTireForce, 1e-4f);
        m_tireDebug.rearGripUtilization =
            std::sqrt(rearForceX * rearForceX + rearForceY * rearForceY) / std::max(m_params.rearMaxTireForce, 1e-4f);

        b2World_Step(m_worldId, subDt, kPhysicsSubStepCount);

        // Defensive numerical-safety clamp only -- see kSafetySpeedMultiplier.
        const b2Vec2 postStepVelocity = b2Body_GetLinearVelocity(m_bodyId);
        const float postStepSpeed =
            std::sqrt(postStepVelocity.x * postStepVelocity.x + postStepVelocity.y * postStepVelocity.y);
        const float safetyLimit = m_params.maxSpeed * kSafetySpeedMultiplier;
        if (postStepSpeed > safetyLimit)
        {
            const float scale = safetyLimit / postStepSpeed;
            b2Body_SetLinearVelocity(m_bodyId, b2Vec2{postStepVelocity.x * scale, postStepVelocity.y * scale});
        }

        // Re-derive the plain mirror state every caller reads from, and
        // check collision, after every substep -- not just once at the end
        // -- so a car that leaves the road mid-update() is caught as soon
        // as it happens (and, via the loop condition below, stops
        // consuming further substeps once dead, exactly like a fresh
        // update() call would with m_alive already false).
        m_position = toRaylib(b2Body_GetPosition(m_bodyId));
        m_heading = b2Rot_GetAngle(b2Body_GetRotation(m_bodyId));
        m_velocity = toRaylib(b2Body_GetLinearVelocity(m_bodyId));

        applyCollision();
        if (!m_alive)
        {
            break;
        }
    }

    // Sensors are only meaningful once per update() call (they feed the
    // next decision, not mid-update() bookkeeping), so cast them once here
    // regardless of how the loop above ended -- including the case where
    // this update() call just killed the car: applyCollision() already
    // moved the car to its final resting pose/state, so this is the "one
    // final, correct sensor cast from the death pose" verifySensors()
    // expects.
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

        // Default to "nothing found": the full range, in the ray's direction.
        float distance = kMaxSensorDistance;
        Vector2 endPoint = {origin.x + direction.x * kMaxSensorDistance, origin.y + direction.y * kMaxSensorDistance};

        // Step outward in fixed increments, querying only the CPU mask via
        // Track::isDrivable. Float sample coordinates are rounded to the
        // nearest integer pixel (consistent with applyCollision's corner
        // checks above); isDrivable is responsible for bounds safety.
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
    // Below this speed, forward/lateral velocity are dominated by float
    // noise rather than actual direction, so report a stable 0 instead of
    // an undefined/noisy atan2 result.
    constexpr float kMinSpeedForSlipAngle = 1.0f; // px/s

    if (getSpeed() < kMinSpeedForSlipAngle)
    {
        return 0.0f;
    }

    return std::atan2(getLateralVelocity(), getForwardVelocity());
}

} // namespace simulation
