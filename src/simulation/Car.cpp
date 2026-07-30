#include "simulation/Car.h"

#include <algorithm>
#include <cmath>

namespace simulation
{

Car::Car(const CarParams& params, const Track& track)
    : m_params(params)
    , m_track(track)
{
}

void Car::reset(Vector2 spawnPosition, float spawnHeading)
{
    m_position = spawnPosition;
    m_velocity = {0.0f, 0.0f};
    m_heading = spawnHeading;
    m_alive = true;
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

    const Vector2 forward = {std::cos(m_heading), std::sin(m_heading)};
    const Vector2 right = {-std::sin(m_heading), std::cos(m_heading)};

    // 1-2: engine acceleration along the forward vector.
    m_velocity.x += forward.x * m_params.acceleration * throttle * dt;
    m_velocity.y += forward.y * m_params.acceleration * throttle * dt;

    // 3: drag as exponential decay. This is the exact solution of
    // dv/dt = -drag * v, so it is stable at any dt and can only shrink
    // velocity toward zero, never overshoot past it and reverse sign.
    const float dragFactor = std::exp(-m_params.drag * dt);
    m_velocity.x *= dragFactor;
    m_velocity.y *= dragFactor;

    // 4: decompose velocity into forward/lateral components relative to heading.
    const float forwardSpeed = m_velocity.x * forward.x + m_velocity.y * forward.y;
    const float lateralSpeed = m_velocity.x * right.x + m_velocity.y * right.y;

    // 5: lateral grip bleeds off the lateral component exponentially rather
    // than zeroing it outright, allowing brief, controlled sliding.
    const float gripFactor = std::exp(-m_params.lateralGrip * dt);
    const float slidLateralSpeed = lateralSpeed * gripFactor;

    m_velocity.x = forward.x * forwardSpeed + right.x * slidLateralSpeed;
    m_velocity.y = forward.y * forwardSpeed + right.y * slidLateralSpeed;

    // 6: clamp total speed.
    const float speed = std::sqrt(m_velocity.x * m_velocity.x + m_velocity.y * m_velocity.y);
    if (speed > m_params.maxSpeed)
    {
        const float scale = m_params.maxSpeed / speed;
        m_velocity.x *= scale;
        m_velocity.y *= scale;
    }

    // 7: steering effectiveness scales with speed, so the car cannot
    // meaningfully rotate while stationary, and saturates at maxSpeed.
    const float speedFactor = std::clamp(speed / m_params.maxSpeed, 0.0f, 1.0f);
    m_heading += steering * m_params.turnRate * speedFactor * dt;

    // 8: position integration.
    m_position.x += m_velocity.x * dt;
    m_position.y += m_velocity.y * dt;

    applyCollision();
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
