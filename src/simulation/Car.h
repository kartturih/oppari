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

// A single car with lightweight custom 2D dynamics, driven purely through
// CarInput passed to update(). Car holds no reference to any input device.
// Collision is resolved against a Track's CPU drivable mask by checking the
// four world-space corners of the car's rotated rectangular body.
class Car
{
public:
    Car(const CarParams& params, const Track& track);

    // Advances the car by exactly dt seconds: applies engine acceleration,
    // drag, lateral grip, steering and position integration, then checks
    // track collision. If the car is not alive, does nothing.
    void update(const CarInput& input, float dt);

    // Resets the car to a known transform: zero velocity, alive = true.
    void reset(Vector2 spawnPosition, float spawnHeading);

    Vector2 getPosition() const { return m_position; }
    Vector2 getVelocity() const { return m_velocity; }
    float getHeading() const { return m_heading; }
    bool isAlive() const { return m_alive; }

    const CarParams& getParams() const { return m_params; }

    // Four world-space corners of the car's rotated rectangular body, in
    // order around the perimeter (front-right, front-left, rear-left,
    // rear-right). Used for both collision checks and rendering.
    std::array<Vector2, 4> getCorners() const;

private:
    // Checks all four body corners against the track mask; on any
    // non-drivable corner, kills the car and zeroes its velocity.
    void applyCollision();

    CarParams m_params;
    const Track& m_track;

    Vector2 m_position = {0.0f, 0.0f};
    Vector2 m_velocity = {0.0f, 0.0f};
    float m_heading = 0.0f;
    bool m_alive = true;
};

} // namespace simulation
