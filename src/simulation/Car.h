#pragma once

#include <array>

#include "raylib.h"

#include "simulation/Track.h"

namespace simulation
{

// External control signal for a car. This is the only way to drive a Car:
// it never reads input devices itself. In Stage 2 this is filled from the
// keyboard in main.cpp; later stages may fill it from a neural network.
struct CarInput
{
    float throttle = 0.0f; // 0.0 (no throttle) to 1.0 (full throttle)
    float steering = 0.0f; // -1.0 (full left) to 1.0 (full right)
};

// Tunable constants for the lightweight 2D vehicle model. Values are chosen
// for the current 1200x700 oval track, whose road band is roughly
// 130-150px wide.
struct CarParams
{
    float length = 24.0f;
    float width = 12.0f;

    float acceleration = 260.0f; // px/s^2 applied along the forward vector at full throttle
    float maxSpeed = 260.0f;     // px/s, clamps total velocity magnitude
    float drag = 0.6f;           // 1/s exponential decay rate applied to velocity every update
    float lateralGrip = 6.0f;    // 1/s exponential decay rate applied to the lateral velocity component
    float turnRate = 3.2f;       // rad/s heading change at full steering and at/above maxSpeed
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

// A single car with lightweight custom 2D dynamics, driven purely through
// CarInput passed to update(). Car holds no reference to any input device.
// Collision is resolved against a Track's CPU drivable mask by checking the
// four world-space corners of the car's rotated rectangular body.
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

    Car(const CarParams& params, const Track& track);

    // Advances the car by exactly dt seconds: applies engine acceleration,
    // drag, lateral grip, steering and position integration, then checks
    // track collision and recasts the sensors from the resulting pose. If
    // the car is not alive, does nothing (including no sensor recast).
    void update(const CarInput& input, float dt);

    // Resets the car to a known transform: zero velocity, alive = true, and
    // recasts the sensors so they are valid for the spawn pose immediately.
    void reset(Vector2 spawnPosition, float spawnHeading);

    Vector2 getPosition() const { return m_position; }
    Vector2 getVelocity() const { return m_velocity; }
    float getHeading() const { return m_heading; }
    bool isAlive() const { return m_alive; }

    const CarParams& getParams() const { return m_params; }

    // Configured maximum speed (px/s), the same value used internally to
    // clamp velocity. Exposed on its own so callers that only need this one
    // number (e.g. AI-layer normalization) don't need the full CarParams.
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
    // and heading (no separate cached state to keep in sync).
    float getSpeed() const;           // magnitude of world-space velocity, px/s
    float getForwardVelocity() const; // velocity projected onto heading's forward vector, px/s
    float getLateralVelocity() const; // velocity projected onto heading's right vector, px/s (positive = rightward)
    float getSlipAngle() const;       // signed angle between heading and velocity direction, radians; 0 when nearly stationary

private:
    // Checks all four body corners against the track mask; on any
    // non-drivable corner, kills the car and zeroes its velocity.
    void applyCollision();

    // Casts all five sensors from the current pose against the track mask.
    void updateSensors();

    CarParams m_params;
    const Track& m_track;

    Vector2 m_position = {0.0f, 0.0f};
    Vector2 m_velocity = {0.0f, 0.0f};
    float m_heading = 0.0f;
    bool m_alive = true;

    std::array<SensorReading, kSensorCount> m_sensors{};
};

} // namespace simulation
