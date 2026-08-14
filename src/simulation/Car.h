#pragma once

#include <array>

#include "box2d/box2d.h"
#include "raylib.h"

#include "simulation/Track.h"

namespace simulation
{

// External control signal for a car. This is the only way to drive a Car:
// it never reads input devices itself. In Stage 2 this is filled from the
// keyboard in main.cpp; Stage 8+ fills it from a neural network (see
// AIController). Stage 20/20.1/20.2 all keep this interface exactly as-is
// -- throttle/steering are still the only two control channels a caller
// (manual or AI) ever needs to supply; see Car.cpp for how they are turned
// into Box2D forces (steering maps to a front-wheel angle, never directly
// to a commanded body rotation -- see the class comment).
struct CarInput
{
    float throttle = 0.0f; // 0.0 (no throttle) to 1.0 (full throttle)
    float steering = 0.0f; // -1.0 (full left) to 1.0 (full right)
};

// Centralized tuning for the Box2D-backed top-down vehicle model. Stage
// 20.2 replaced the previous "shared grip budget + direct yaw-rate
// controller" model with a front/rear single-track (bicycle model) tire
// model -- see Car.cpp for the full derivation. Every field here has a
// concrete physical role in that model; nothing is kept around from the
// old model without one (see Car.cpp's Stage 20.2 comment for exactly what
// was removed and why). Nothing in NEAT (Genome/GenomeMutator/
// PhenotypeBuilder/...) reads or depends on any field here -- these shape
// the *simulation* a genome is evaluated against, not the genome itself.
struct CarParams
{
    // Body dimensions, in px -- also Box2D simulation units (see Car.cpp's
    // "Unit convention" comment for why no separate px-to-meters scale
    // factor is used).
    float length = 24.0f;
    float width = 12.0f;

    // Shape density (mass per unit area): mass = density * length * width.
    // Box2D derives rotational inertia from density + shape automatically
    // (Car.cpp reads it fresh every update() via
    // b2Body_GetRotationalInertia() rather than hardcoding a value, so it
    // always matches density/length/width exactly).
    float density = 0.05f; // mass = 0.05 * 24 * 12 = 14.4

    // Small Box2D-native damping, purely a numerical-stability backstop --
    // NOT the primary resistance/steering mechanism. Longitudinal
    // resistance is engineForce/rollingResistance/dragCoefficient below;
    // lateral/yaw behavior now comes entirely from the front/rear tire
    // forces (frontCorneringStiffness/rearCorneringStiffness/
    // tireMaxLateralForce), applied at the axle positions via Box2D's
    // point-force API so yaw is a genuine physical consequence of those
    // forces' moment arms -- never a directly commanded angular velocity
    // (see Car.cpp). angularDamping is deliberately small: a real car's
    // yaw naturally damps because tire slip angles (and therefore
    // restoring forces) grow as unwanted rotation develops -- that IS the
    // yaw damping, and an artificial damping term large enough to matter
    // would just mask or fight it.
    float linearDamping = 0.05f;
    float angularDamping = 0.3f;

    // Engine: throttle = 1 requests exactly this much drive force, every
    // update(), independent of current speed. Stage 20.3: the car is
    // rear-wheel drive -- this is no longer applied at the center of mass.
    // It is the REAR axle's desired longitudinal force, combined with the
    // rear axle's lateral tire force under the rear friction circle (see
    // rearMaxTireForce below and Car.cpp) before being applied at the rear
    // axle's world position -- so heavy throttle genuinely competes with
    // the rear tires' cornering grip, exactly like a real RWD car. The
    // front axle never receives drive force (FxFront = 0 always). There is
    // no separate reverse/brake force -- CarInput has none; releasing
    // throttle simply stops requesting drive force and lets
    // rollingResistance/dragCoefficient below slow the car down.
    float engineForce = 3900.0f;

    // Longitudinal resistance, split by what it physically represents
    // (Stage 20.2 change -- see Car.cpp): rollingResistance opposes only
    // the car's own forward-rolling motion (a tire rolling resistance
    // effect), while dragCoefficient (quadratic, aerodynamic-like) opposes
    // the full velocity vector. Before Stage 20.2, rollingResistance also
    // opposed lateral motion as a crude stand-in for tire grip -- now that
    // real tire forces (below) do that job, applying rollingResistance
    // laterally too would double-count it. Together with engineForce/
    // density above, these still set the analytic steady state
    // (engineForce == total resistance force) under sustained full
    // throttle on a straight, at ~300px/s -- unchanged from the Stage 20.1
    // tuning pass, since this split doesn't change straight-line behavior
    // (lateral velocity is 0 on a straight, so rollingResistance's scope
    // change has no effect there). maxSpeed is an approximate top-speed
    // reference (used only for Observation normalization), not a
    // separately enforced clamp (Car.cpp still applies one defensive,
    // generously-above-maxSpeed velocity clamp purely as numerical safety
    // -- see kSafetySpeedMultiplier).
    float rollingResistance = 1.6f;
    float dragCoefficient = 0.038f;
    float maxSpeed = 260.0f; // px/s -- see the note above; true steady state is somewhat higher

    // Single-track (bicycle model) steering geometry -- Stage 20.2. Every
    // lateral/yaw effect in this model is derived from these two axle
    // distances plus the car's current motion; there is no separate
    // "wheelBase" field (it is simply cgToFrontAxle + cgToRearAxle -- see
    // Car.cpp) and no kinematic "desired turn radius" formula anymore: the
    // turn radius is now an emergent result of the tire forces below, not
    // commanded directly. The car's CG is assumed to sit at its geometric
    // center (matching the symmetric rendered body -- see getCorners()),
    // so both distances equal half of length.
    float cgToFrontAxle = 12.0f;
    float cgToRearAxle = 12.0f;

    // Front wheel angle at steering = +-1, in radians. CarInput::steering
    // maps directly to this -- see Car.cpp.
    float maxSteerAngle = 0.42f; // radians, ~28.6 degrees

    // Tire model (Stage 20.2, curve replaced in Stage 20.3): each axle's
    // lateral force is
    //   Fy = -axleMaxForce * tanh((corneringStiffness / axleMaxForce) * slipAngle)
    // -- linear in slip angle for small slip, with slope exactly
    // corneringStiffness at slipAngle = 0 (matching real tires' behavior
    // near their grip limit), saturating SMOOTHLY toward +-axleMaxForce as
    // slip grows large (no hard corner/discontinuity, unlike Stage 20.2's
    // clamp() -- see Car.cpp). corneringStiffness is in force per radian of
    // slip. See Car.cpp for the full slip-angle derivation (front slip
    // angle additionally accounts for the steering angle; rear does not).
    float frontCorneringStiffness = 26000.0f; // force per radian
    float rearCorneringStiffness = 20000.0f;  // force per radian

    // Stage 20.3: explicit, independently-tunable per-axle grip budgets --
    // replaces the old single tireMaxLateralForce split by static load.
    // Explicit values (rather than a load-distribution split) are used
    // because the two axles are no longer symmetric in what they do with
    // their budget: the front axle spends its entire budget on lateral
    // (cornering) force alone, while the rear axle's budget is SHARED
    // between the drive force (engineForce * throttle) and lateral force
    // -- see the friction-circle comment in Car.cpp. rearMaxTireForce is
    // deliberately well above engineForce (3900) so that full throttle in
    // a straight line (zero lateral demand) only ever uses a fraction of
    // the rear budget, leaving real cornering grip available at normal
    // throttle -- it is specifically COMBINING heavy throttle with hard
    // cornering that can push the rear axle's combined force
    // sqrt(FxRear^2 + FyRear^2) up to rearMaxTireForce and beyond, which is
    // when rear lateral force gets scaled down (power oversteer). Both
    // values are also each axle's hard ceiling on lateral-only force (at
    // FxRear = 0, this reduces to the old per-axle cap).
    float frontMaxTireForce = 4600.0f;
    float rearMaxTireForce = 5200.0f;
};

// One forward-facing distance sensor's result for the current frame. Cast
// against Track's CPU drivable mask only; carries enough information for
// debug rendering now and NEAT observation construction later.
struct SensorReading
{
    float distance = 0.0f;           // px from the sensor origin to the first non-drivable sample
    float normalizedDistance = 1.0f; // distance / Car::kMaxSensorDistance, in [0, 1]
    Vector2 endPoint = {0.0f, 0.0f}; // world-space point where the ray stopped
};

// Debug snapshot of the front/rear tire model's most recent update() call
// (or a zeroed/rest state immediately after reset()) -- Stage 20.2. Exists
// purely to support the manual-mode HUD (see main.cpp's drawManualPanel())
// and deterministic verification of the tire model itself
// (verifyVehiclePhysics() in main.cpp); no production control or fitness
// path (AIController, Observation, FitnessEvaluator, Population) reads
// this. Every field reflects the SAME instant: the car's state at the
// start of the update() call that computed them (i.e. the causal inputs to
// that update()'s applied forces), so they are internally consistent with
// each other even though the body's actual pose/velocity has since been
// integrated one step further.
struct TireDebugInfo
{
    float steeringAngle = 0.0f;  // radians, front wheel angle (steering * maxSteerAngle)
    float yawRate = 0.0f;        // rad/s, body angular velocity used to compute this update()'s axle velocities
    float frontSlipAngle = 0.0f; // radians, front tire slip angle (wheel frame, includes steering)
    float rearSlipAngle = 0.0f;  // radians, rear tire slip angle (body frame, no steering)

    // Stage 20.3: per-axle force components, in that axle's own frame (X =
    // that wheel's forward/rolling direction, Y = its lateral direction --
    // for the front wheel this is the STEERED wheel frame, matching
    // frontSlipAngle; the rear axle never steers, so its frame is just the
    // body frame). frontForceX is always 0 -- the car is rear-wheel drive
    // (see CarParams::engineForce) -- kept here rather than omitted so the
    // manual-mode HUD and verification code have one uniform
    // front/rear-symmetric struct to read.
    float frontForceX = 0.0f;
    float frontForceY = 0.0f;
    float rearForceX = 0.0f;
    float rearForceY = 0.0f;

    // Stage 20.3: each axle's combined-force friction-circle utilization,
    // in [0, 1] -- sqrt(forceX^2 + forceY^2) / axleMaxTireForce. 1.0 means
    // that axle is fully saturated (any more demand gets scaled down --
    // see Car.cpp's friction-circle comment). Exists so the manual-mode
    // HUD can show e.g. "Rear grip: 94%" directly, and so
    // verifyVehiclePhysics() can assert the circle is never exceeded
    // without recomputing it independently.
    float frontGripUtilization = 0.0f;
    float rearGripUtilization = 0.0f;
};

// A single car with Box2D-backed top-down vehicle dynamics, driven purely
// through CarInput passed to update(). Car holds no reference to any input
// device. Collision is resolved against a Track's CPU drivable mask by
// checking the four world-space corners of the car's rotated rectangular
// body -- unchanged since Stage 20; Box2D is used only for the car's own
// rigid-body integration (forces -> velocity -> position/rotation), never
// for track collision geometry.
//
// Stage 20.2: yaw is never commanded directly (no target-angular-velocity
// controller, no direct b2Body_ApplyTorque call anywhere in this class).
// Instead, Car::update() computes a front and a rear tire lateral force
// from a single-track (bicycle model) tire model, and applies each at its
// own axle's world position via Box2D's point-force API
// (b2Body_ApplyForce) -- Box2D itself derives the resulting torque from
// each force's offset from the center of mass, so rotation is a genuine
// physical consequence of the tire forces' moment arms, exactly like a
// real car turning because its front tires push its front sideways.
//
// Stage 20.3: the car is rear-wheel drive and each axle's total tire force
// obeys a friction circle. Engine force is no longer a separate
// center-of-mass force -- it is the rear axle's desired longitudinal
// force, combined with the rear axle's lateral force under
// sqrt(Fx^2+Fy^2) <= rearMaxTireForce before being applied (see
// CarParams::rearMaxTireForce and Car.cpp). This is what makes throttle
// genuinely compete with rear cornering grip -- heavy throttle mid-corner
// can saturate the rear circle and reduce rear lateral force below what
// the same corner at part throttle would achieve, which is the real
// mechanism behind power oversteer, not a separate scripted effect. See
// Car.cpp for the full model.
//
// Each Car owns a private, single-body Box2D world (see Car.cpp): cars
// never collide with each other (they never did, even before Stage 20 --
// only against the track mask), so there is no need for one shared world,
// and a private world keeps every Individual's Car a fully independent
// instance exactly as the rest of this codebase already assumes. Car is
// move-constructible (transfers ownership of the Box2D world/body) but not
// copyable or assignable -- see Car.cpp.
class Car
{
public:
    // Sensor layout: five forward-facing rays at fixed angles relative to
    // heading, cast in fixed 2px steps out to a fixed maximum range. Kept as
    // explicit named constants (not a generic config framework) so they stay
    // easy to find and change.
    static constexpr int kSensorCount = 5;
    static constexpr float kSensorAngleDegrees[kSensorCount] = {-60.0f, -30.0f, 0.0f, 30.0f, 60.0f};
    static constexpr float kMaxSensorDistance = 200.0f; // px
    static constexpr float kSensorStep = 2.0f;           // px

    // Fixed Box2D sub-step count used by every update()'s b2World_Step()
    // call. Box2D v3 replaced the old v2.x separate velocity/position
    // solver iteration counts with a single soft-step sub-step count; 4
    // matches Box2D's own recommended default and is what its samples use
    // for ordinary rigid-body scenes. The outer timestep itself is always
    // the caller's dt -- in practice always main.cpp's fixed kSimulationDt
    // (1/60s), never a measured frame time (see main.cpp) -- so training
    // results stay independent of render FPS.
    static constexpr int kPhysicsSubStepCount = 4;

    Car(const CarParams& params, const Track& track);
    ~Car();

    Car(const Car&) = delete;
    Car& operator=(const Car&) = delete;
    Car(Car&& other) noexcept;
    Car& operator=(Car&&) = delete; // m_track is a reference: assignment can never rebind it

    // Advances the car by exactly dt seconds: computes resistance forces
    // (applied at the center of mass) and the front/rear tire forces --
    // front lateral-only, rear combining drive force with lateral force
    // under its friction circle -- from the car's current Box2D state (see
    // the class comment), applies each axle's force at its own world
    // position, steps the private Box2D world by dt (see
    // kPhysicsSubStepCount), then re-derives m_position/m_velocity/m_heading
    // from the resulting body pose, checks track collision, and recasts
    // the sensors. If the car is not alive, does nothing (including no
    // physics stepping and no sensor recast).
    void update(const CarInput& input, float dt);

    // Resets the car to a known transform: zero velocity (linear and
    // angular) on the Box2D body, alive = true, tire debug info back to its
    // zeroed/rest state, and recasts the sensors so they are valid for the
    // spawn pose immediately.
    void reset(Vector2 spawnPosition, float spawnHeading);

    Vector2 getPosition() const { return m_position; }
    Vector2 getVelocity() const { return m_velocity; }
    float getHeading() const { return m_heading; }
    bool isAlive() const { return m_alive; }

    const CarParams& getParams() const { return m_params; }

    // Configured maximum speed (px/s) -- see CarParams::maxSpeed's comment
    // for why this is only an approximate top-speed reference. Exposed on
    // its own so callers that only need this one number (e.g. AI-layer
    // normalization) don't need the full CarParams.
    float getMaxSpeed() const { return m_params.maxSpeed; }

    // Four world-space corners of the car's rotated rectangular body, in
    // order around the perimeter (front-right, front-left, rear-left,
    // rear-right). Used for both collision checks and rendering.
    std::array<Vector2, 4> getCorners() const;

    // World-space origin all five sensors are cast from: the center of the
    // car's front edge. Moves and rotates with the car.
    Vector2 getSensorOrigin() const;

    // The five sensor readings from the most recent update()/reset(), in
    // kSensorAngleDegrees order. Fixed-size storage, no per-frame allocation.
    const std::array<SensorReading, kSensorCount>& getSensors() const { return m_sensors; }

    // Local vehicle-state values, derived on demand from current velocity
    // and heading (no separate cached state to keep in sync). Note these
    // describe the whole CAR body's motion (its own sideslip), distinct
    // from the per-tire slip angles in getTireDebugInfo() below.
    float getSpeed() const;           // magnitude of world-space velocity, px/s
    float getForwardVelocity() const; // velocity projected onto heading's forward vector, px/s
    float getLateralVelocity() const; // velocity projected onto heading's right vector, px/s (positive = rightward)
    float getSlipAngle() const;       // signed angle between heading and velocity direction, radians; 0 when nearly stationary

    // The front/rear tire model's most recent update() snapshot -- see
    // TireDebugInfo's comment. Returns a zeroed struct if update() has
    // never run since construction/reset().
    const TireDebugInfo& getTireDebugInfo() const { return m_tireDebug; }

private:
    // Checks all four body corners against the track mask; on any
    // non-drivable corner, kills the car and zeroes its Box2D body's
    // velocity (linear and angular) as well as the mirrored m_velocity.
    void applyCollision();

    // Casts all five sensors from the current pose against the track mask.
    void updateSensors();

    CarParams m_params;
    const Track& m_track;

    // The private Box2D world/body backing this car's dynamics -- see the
    // class comment for why each Car owns its own world rather than
    // sharing one. Never null after construction until this Car is
    // moved-from or destroyed; see Car.cpp.
    b2WorldId m_worldId = b2_nullWorldId;
    b2BodyId m_bodyId = b2_nullBodyId;

    // Mirrors of the Box2D body's current pose/velocity, refreshed at the
    // end of every update()/reset() -- kept so the public interface stays
    // plain Vector2/float (no Box2D types leak past this class) and so
    // getPosition()/getVelocity()/getHeading() stay cheap, allocation-free
    // reads matching their Stage 2 signatures exactly.
    Vector2 m_position = {0.0f, 0.0f};
    Vector2 m_velocity = {0.0f, 0.0f};
    float m_heading = 0.0f;
    bool m_alive = true;

    std::array<SensorReading, kSensorCount> m_sensors{};
    TireDebugInfo m_tireDebug{};
};

} // namespace simulation
