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
    float throttle = 0.0f; // 0.0 to 1.0
    float steering = 0.0f; // -1.0 (full left) to 1.0 (full right)
};

// Tuning for the Box2D-backed top-down front/rear bicycle-model car. Shapes
// the simulation a genome is evaluated against -- NEAT never reads this.
struct CarParams
{
    // Body dimensions, px (also Box2D units -- no separate scale factor).
    float length = 24.0f;
    float width = 12.0f;

    float density = 0.05f; // mass = density * length * width

    // Numerical-stability backstop only -- not the primary resistance/
    // steering mechanism (that's engineForce/resistance below and the tire
    // forces). angularDamping is small since tire slip angles already
    // provide the real yaw damping.
    float linearDamping = 0.05f;
    float angularDamping = 0.3f;

    // Rear axle's desired longitudinal force at throttle=1 (RWD -- front
    // never drives), combined with rear lateral force under the rear
    // friction circle (see rearMaxTireForce). No separate reverse/brake.
    float engineForce = 3900.0f;

    // rollingResistance opposes forward-rolling motion only; dragCoefficient
    // (quadratic) opposes the full velocity vector. Together with
    // engineForce/density, sets the straight-line steady-state speed
    // (~300px/s). maxSpeed is a reference for Observation normalization,
    // not an enforced clamp (Car.cpp still applies a generous safety clamp).
    float rollingResistance = 1.6f;
    float dragCoefficient = 0.038f;
    float maxSpeed = 260.0f; // px/s, approximate

    // Bicycle-model axle distances from CG (assumed at geometric center).
    float cgToFrontAxle = 12.0f;
    float cgToRearAxle = 12.0f;

    float maxSteerAngle = 0.42f; // radians at steering = +-1 (~28.6 deg)

    // Per-axle lateral tire force: Fy = -axleMaxForce *
    // tanh((corneringStiffness / axleMaxForce) * slipAngle) -- linear near
    // zero slip (slope = corneringStiffness), saturating smoothly toward
    // +-axleMaxForce. Front slip angle includes steering; rear doesn't.
    float frontCorneringStiffness = 26000.0f; // force per radian
    float rearCorneringStiffness = 20000.0f;

    // Per-axle grip budget. Front spends its budget on lateral force alone;
    // rear SHARES its budget between drive force and lateral force (a
    // friction circle: sqrt(FxRear^2+FyRear^2) <= rearMaxTireForce) -- this
    // is what makes heavy throttle mid-corner reduce rear grip (power
    // oversteer). rearMaxTireForce is set above engineForce so full
    // straight-line throttle alone never saturates it.
    float frontMaxTireForce = 4600.0f;
    float rearMaxTireForce = 5200.0f;
};

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
    float steeringAngle = 0.0f;  // radians, front wheel angle
    float yawRate = 0.0f;        // rad/s
    float frontSlipAngle = 0.0f; // radians, wheel frame (includes steering)
    float rearSlipAngle = 0.0f;  // radians, body frame

    // Per-axle force in that axle's own frame (X = rolling dir, Y = lateral).
    // frontForceX is always 0 (RWD).
    float frontForceX = 0.0f;
    float frontForceY = 0.0f;
    float rearForceX = 0.0f;
    float rearForceY = 0.0f;

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
// longitudinal force, combined with its lateral force under the rear
// friction circle before being applied.
//
// Each Car owns a private single-body Box2D world (cars never collide with
// each other). Move-constructible only, not copyable/assignable.
class Car
{
public:
    // Five forward sensors at fixed angles, cast in fixed steps to a fixed range.
    static constexpr int kSensorCount = 5;
    static constexpr float kSensorAngleDegrees[kSensorCount] = {-60.0f, -30.0f, 0.0f, 30.0f, 60.0f};
    static constexpr float kMaxSensorDistance = 200.0f; // px
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
};

} // namespace simulation
