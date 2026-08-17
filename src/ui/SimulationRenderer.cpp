#include "ui/SimulationRenderer.h"

#include <algorithm>
#include <array>

namespace ui
{

Color progressRankColor(float normalizedRank)
{
    const float clamped = std::clamp(normalizedRank, 0.0f, 1.0f);
    if (clamped < 0.5f)
    {
        const float t = clamped / 0.5f; // green -> yellow
        return Color{static_cast<unsigned char>(t * 255.0f), 255, 0, 255};
    }
    const float t = (clamped - 0.5f) / 0.5f; // yellow -> red
    return Color{255, static_cast<unsigned char>((1.0f - t) * 255.0f), 0, 255};
}

void drawIndividualCar(const simulation::Car& car, Color color, bool highlighted)
{
    const std::array<Vector2, 4> corners = car.getCorners();
    DrawTriangleFan(corners.data(), static_cast<int>(corners.size()), color);

    if (highlighted && car.isAlive())
    {
        const Vector2 position = car.getPosition();
        const Vector2 velocity = car.getVelocity();
        const Vector2 tip = {position.x + velocity.x * 0.25f, position.y + velocity.y * 0.25f};
        DrawLineEx(position, tip, 2.0f, YELLOW);

        const Vector2 origin = car.getSensorOrigin();
        for (const simulation::SensorReading& sensor : car.getSensors())
        {
            DrawLineEx(origin, sensor.endPoint, 1.5f, Color{80, 255, 120, 255});
            DrawCircleV(sensor.endPoint, 3.0f, Color{255, 90, 40, 255});
        }
    }
}

} // namespace ui
