#include "simulation/Car.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace simulation
{

namespace
{

// 1 Box2D unit == 1px, no separate scale factor (Box2D v3 has no baked-in
// "meter-sized" assumption; its maximumLinearSpeed default comfortably
// exceeds maxSpeed + the safety clamp below).
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
    , m_alive(other.m_alive)
    , m_sensors(other.m_sensors)
    , m_tireDebug(other.m_tireDebug)
{
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

        // Smooth saturating tire curve: Fy = -Fmax * tanh((C/Fmax) *
        // slipAngle) -- slope C at slipAngle=0, approaches +-Fmax smoothly.
        // Scaled by each axle's low-speed ramp (kTireForceRampSpeed). This
        // is lateral force alone, before the rear friction-circle combine below.
        const float frontSpeedRamp = std::clamp(std::fabs(vxFrontWheel) / kTireForceRampSpeed, 0.0f, 1.0f);
        const float rearSpeedRamp = std::clamp(std::fabs(vxBody) / kTireForceRampSpeed, 0.0f, 1.0f);

        const float frontLateralForce = -m_params.frontMaxTireForce *
                                         std::tanh((m_params.frontCorneringStiffness / m_params.frontMaxTireForce) * frontSlipAngle) *
                                         frontSpeedRamp;
        const float rearLateralForceDesired = -m_params.rearMaxTireForce *
                                               std::tanh((m_params.rearCorneringStiffness / m_params.rearMaxTireForce) * rearSlipAngle) *
                                               rearSpeedRamp;

        // Rear-wheel drive + friction circle: desired drive force and
        // desired lateral force combine into one vector constrained to
        // sqrt(FxRear^2+FyRear^2) <= rearMaxTireForce -- exceeding the
        // budget scales BOTH components down together, so heavy throttle
        // mid-corner reduces available lateral grip (power oversteer). The
        // front axle never drives, so its lateral force never needs this.
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

        // Apply each force at its own axle's world position -- Box2D
        // derives torque from the offset from center of mass, so yaw is a
        // genuine consequence of these moment arms, never commanded
        // directly. Front force is in the steered wheel frame; rear in the
        // body frame (it never steers).
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
