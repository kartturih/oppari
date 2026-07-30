#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>

#include "raylib.h"

#include "simulation/Car.h"
#include "simulation/Track.h"

namespace
{

constexpr int kSimWidth = 1200;
constexpr int kSimHeight = 700;
constexpr int kPanelWidth = 400;
constexpr int kScreenWidth = kSimWidth + kPanelWidth;
constexpr int kScreenHeight = kSimHeight;

// Stage 2 executes exactly one fixed-size simulation step per rendered
// frame. This is intentionally not GetFrameTime(): Car must always see the
// same dt regardless of measured render duration.
constexpr float kSimulationDt = 1.0f / 60.0f;

// Bottom straight of the oval, where the road band is wide and its tangent
// is horizontal, so heading = 0 (pointing along +x) is track-aligned.
constexpr Vector2 kSpawnPosition = {600.0f, 565.0f};
constexpr float kSpawnHeading = 0.0f;

simulation::TrackDefinition makeTrackDefinition()
{
    simulation::TrackDefinition def;
    def.simWidth = kSimWidth;
    def.simHeight = kSimHeight;
    def.center = {600.0f, 350.0f};
    def.outerRadiusX = 500.0f;
    def.outerRadiusY = 280.0f;
    def.innerRadiusX = 350.0f;
    def.innerRadiusY = 150.0f;
    return def;
}

simulation::CarParams makeCarParams()
{
    return simulation::CarParams{};
}

// One-shot, deterministic sanity check of the CPU mask against the known
// track geometry. Runs once at startup, never inside the render loop.
void verifyTrack(const simulation::Track& track)
{
    const simulation::TrackDefinition& def = track.getDefinition();
    const int cx = static_cast<int>(def.center.x);
    const int cy = static_cast<int>(def.center.y);

    assert(!track.isDrivable(cx, cy) && "center of inner ellipse must be non-drivable");
    assert(track.isDrivable(cx, cy - 200) && "point in road band must be drivable");
    assert(!track.isDrivable(cx, cy - 300) && "point outside outer ellipse must be non-drivable");
    assert(!track.isDrivable(-5, -5) && "negative coordinates must be non-drivable");
    assert(!track.isDrivable(kSimWidth, cy) && "x at/beyond width must be non-drivable");
    assert(!track.isDrivable(cx, kSimHeight) && "y at/beyond height must be non-drivable");

    TraceLog(LOG_INFO, "Track verification: all CPU mask checks passed");
}

// One-shot, deterministic sanity check of Car's dynamics and collision,
// independent of any keyboard/render timing. Runs once at startup.
void verifyCar(const simulation::Track& track)
{
    simulation::Car car(makeCarParams(), track);
    car.reset(kSpawnPosition, kSpawnHeading);

    assert(car.isAlive() && "car must be spawned alive");
    assert(car.getPosition().x == kSpawnPosition.x && car.getPosition().y == kSpawnPosition.y &&
           "reset must place the car at the spawn position");

    // Steering alone, while stationary, must not rotate the car.
    simulation::CarInput steerOnly;
    steerOnly.throttle = 0.0f;
    steerOnly.steering = 1.0f;
    for (int i = 0; i < 30; ++i)
    {
        car.update(steerOnly, kSimulationDt);
    }
    assert(car.getHeading() == kSpawnHeading && "steering must have no effect while stationary");

    // Sustained throttle must build up speed from rest.
    simulation::CarInput throttleOnly;
    throttleOnly.throttle = 1.0f;
    throttleOnly.steering = 0.0f;
    for (int i = 0; i < 30; ++i)
    {
        car.update(throttleOnly, kSimulationDt);
    }
    const Vector2 v = car.getVelocity();
    const float speed = std::sqrt(v.x * v.x + v.y * v.y);
    assert(speed > 1.0f && "sustained throttle must build up visible speed");
    assert(speed <= car.getParams().maxSpeed + 0.01f && "speed must never exceed maxSpeed");

    // Driving straight off the track must kill the car and zero its velocity.
    car.reset(kSpawnPosition, kSpawnHeading);
    simulation::CarInput driveOffTrack;
    driveOffTrack.throttle = 1.0f;
    driveOffTrack.steering = 0.0f;
    for (int i = 0; i < 300 && car.isAlive(); ++i)
    {
        car.update(driveOffTrack, kSimulationDt);
    }
    assert(!car.isAlive() && "driving straight for 5s must leave the road band and kill the car");
    const Vector2 deadVelocity = car.getVelocity();
    assert(deadVelocity.x == 0.0f && deadVelocity.y == 0.0f && "a dead car must have zero velocity");

    // A dead car must ignore further input.
    const Vector2 deadPosition = car.getPosition();
    car.update(throttleOnly, kSimulationDt);
    assert(car.getPosition().x == deadPosition.x && car.getPosition().y == deadPosition.y &&
           "a dead car must not respond to further input");

    // Reset must revive the car.
    car.reset(kSpawnPosition, kSpawnHeading);
    assert(car.isAlive() && "reset must revive the car");
    assert(car.getVelocity().x == 0.0f && car.getVelocity().y == 0.0f && "reset must zero velocity");

    TraceLog(LOG_INFO, "Car verification: all deterministic checks passed");
}

// One-shot, deterministic sanity check of Car's five sensors, independent of
// keyboard/render timing. Runs once at startup.
void verifySensors(const simulation::Track& track)
{
    static_assert(simulation::Car::kSensorCount == 5, "Stage 3 requires exactly five sensors");

    simulation::Car car(makeCarParams(), track);
    car.reset(kSpawnPosition, kSpawnHeading);

    // 1: exactly five sensor readings exist, in fixed-size storage.
    const auto& sensors = car.getSensors();
    assert(sensors.size() == 5 && "exactly five sensor readings must exist");

    // 2 & 3: normalized readings stay within [0,1] and are valid immediately after reset.
    for (const simulation::SensorReading& s : sensors)
    {
        assert(s.normalizedDistance >= 0.0f && s.normalizedDistance <= 1.0f &&
               "normalized sensor distance must stay within [0,1]");
        assert(s.distance >= 0.0f && s.distance <= simulation::Car::kMaxSensorDistance &&
               "raw sensor distance must stay within [0, kMaxSensorDistance]");
    }

    // 4: the front (0 degree) sensor points in the same world direction as the car heading.
    {
        const Vector2 origin = car.getSensorOrigin();
        const simulation::SensorReading& front = sensors[2];
        const float rayAngle = std::atan2(front.endPoint.y - origin.y, front.endPoint.x - origin.x);
        assert(std::fabs(rayAngle - kSpawnHeading) < 0.01f &&
               "front sensor must point along the car heading");
    }

    // 5: resetting to a known heading rotates the whole sensor layout with it.
    {
        const float knownHeading = static_cast<float>(PI) * 0.5f; // pointing +y ("down")
        car.reset(kSpawnPosition, knownHeading);
        const Vector2 origin = car.getSensorOrigin();
        const auto& rotatedSensors = car.getSensors();

        for (int i = 0; i < simulation::Car::kSensorCount; ++i)
        {
            const float expectedAngle = knownHeading + simulation::Car::kSensorAngleDegrees[i] * DEG2RAD;
            const Vector2 expectedDir = {std::cos(expectedAngle), std::sin(expectedAngle)};
            const Vector2 rayDir = {rotatedSensors[i].endPoint.x - origin.x, rotatedSensors[i].endPoint.y - origin.y};
            const float rayLen = std::sqrt(rayDir.x * rayDir.x + rayDir.y * rayDir.y);
            assert(rayLen > 0.01f && "sensor ray must have nonzero length");
            const Vector2 rayDirNorm = {rayDir.x / rayLen, rayDir.y / rayLen};
            const float dot = rayDirNorm.x * expectedDir.x + rayDirNorm.y * expectedDir.y;
            assert(dot > 0.999f && "rotated sensor direction must match heading + relative angle");
        }

        car.reset(kSpawnPosition, kSpawnHeading);
    }

    // 6: a sensor directed at a nearby track boundary reports less than maximum distance.
    // At the spawn x, the inner ellipse boundary is ~65px above the car; heading -90deg
    // points the front sensor straight at it, well within the 200px range.
    {
        car.reset(kSpawnPosition, -static_cast<float>(PI) * 0.5f);
        const simulation::SensorReading& front = car.getSensors()[2];
        assert(front.distance < simulation::Car::kMaxSensorDistance &&
               front.normalizedDistance < 1.0f &&
               "sensor aimed at a nearby boundary must report less than maximum distance");
        car.reset(kSpawnPosition, kSpawnHeading);
    }

    // 7: a ray with no obstacle within range reports exactly maximum distance / normalized 1.0.
    // At spawn heading 0, the front sensor points along the long bottom straight, clear for 200px.
    {
        const simulation::SensorReading& front = car.getSensors()[2];
        assert(front.distance == simulation::Car::kMaxSensorDistance &&
               "unobstructed sensor must report exactly the maximum distance");
        assert(front.normalizedDistance == 1.0f &&
               "unobstructed sensor must report exactly normalized 1.0");
    }

    // 8: sensor endpoints correspond to their recorded distance and direction.
    {
        const Vector2 origin = car.getSensorOrigin();
        for (int i = 0; i < simulation::Car::kSensorCount; ++i)
        {
            const float angle = kSpawnHeading + simulation::Car::kSensorAngleDegrees[i] * DEG2RAD;
            const Vector2 direction = {std::cos(angle), std::sin(angle)};
            const simulation::SensorReading& s = sensors[i];
            const Vector2 expectedEnd = {origin.x + direction.x * s.distance, origin.y + direction.y * s.distance};
            assert(std::fabs(expectedEnd.x - s.endPoint.x) < 0.01f &&
                   std::fabs(expectedEnd.y - s.endPoint.y) < 0.01f &&
                   "sensor endpoint must match its recorded distance and direction");
        }
    }

    // 9: a dead car retains its final sensor readings and does not recast while stationary.
    {
        car.reset(kSpawnPosition, kSpawnHeading);
        simulation::CarInput driveOffTrack;
        driveOffTrack.throttle = 1.0f;
        driveOffTrack.steering = 0.0f;
        for (int i = 0; i < 300 && car.isAlive(); ++i)
        {
            car.update(driveOffTrack, kSimulationDt);
        }
        assert(!car.isAlive() && "driving straight for 5s must leave the road band and kill the car");

        const auto finalSensors = car.getSensors();
        car.update(driveOffTrack, kSimulationDt);
        const auto& sensorsAfterDeadUpdate = car.getSensors();
        for (int i = 0; i < simulation::Car::kSensorCount; ++i)
        {
            assert(finalSensors[i].distance == sensorsAfterDeadUpdate[i].distance &&
                   finalSensors[i].normalizedDistance == sensorsAfterDeadUpdate[i].normalizedDistance &&
                   "a dead car must retain its final sensor readings, not recast them");
        }
    }

    car.reset(kSpawnPosition, kSpawnHeading);

    TraceLog(LOG_INFO, "Sensor verification: all deterministic checks passed");
}

simulation::CarInput readInput()
{
    simulation::CarInput input;

    if (IsKeyDown(KEY_W) || IsKeyDown(KEY_UP))
    {
        input.throttle = 1.0f;
    }

    if (IsKeyDown(KEY_A) || IsKeyDown(KEY_LEFT))
    {
        input.steering -= 1.0f;
    }

    if (IsKeyDown(KEY_D) || IsKeyDown(KEY_RIGHT))
    {
        input.steering += 1.0f;
    }

    return input;
}

void drawCar(const simulation::Car& car)
{
    const std::array<Vector2, 4> corners = car.getCorners();
    const Color color = car.isAlive() ? Color{40, 180, 255, 255} : Color{70, 70, 70, 255};

    DrawTriangleFan(corners.data(), static_cast<int>(corners.size()), color);

    // Velocity vector debug overlay, scaled down so it stays on-screen.
    const Vector2 position = car.getPosition();
    const Vector2 velocity = car.getVelocity();
    const Vector2 tip = {position.x + velocity.x * 0.25f, position.y + velocity.y * 0.25f};
    DrawLineEx(position, tip, 2.0f, YELLOW);

    // Sensor rays: only meaningful while the car is alive and still moving.
    if (car.isAlive())
    {
        const Vector2 origin = car.getSensorOrigin();
        for (const simulation::SensorReading& sensor : car.getSensors())
        {
            DrawLineEx(origin, sensor.endPoint, 1.5f, Color{80, 255, 120, 255});
            DrawCircleV(sensor.endPoint, 3.0f, Color{255, 90, 40, 255});
        }
    }
}

void drawPanel(const simulation::Car& car, const simulation::CarInput& input)
{
    DrawRectangle(kSimWidth, 0, kPanelWidth, kScreenHeight, Color{30, 30, 30, 255});

    const int x = kSimWidth + 20;
    int y = 20;
    const int lineHeight = 22;

    DrawText("STAGE 3 - SENSORS", x, y, 20, RAYWHITE);
    y += lineHeight * 2;

    char line[128];

    std::snprintf(line, sizeof(line), "Speed: %.1f px/s", static_cast<double>(car.getSpeed()));
    DrawText(line, x, y, 18, RAYWHITE);
    y += lineHeight;

    std::snprintf(line, sizeof(line), "Forward velocity: %.1f px/s", static_cast<double>(car.getForwardVelocity()));
    DrawText(line, x, y, 18, RAYWHITE);
    y += lineHeight;

    std::snprintf(line, sizeof(line), "Lateral velocity: %.1f px/s", static_cast<double>(car.getLateralVelocity()));
    DrawText(line, x, y, 18, RAYWHITE);
    y += lineHeight;

    std::snprintf(line, sizeof(line), "Slip angle: %.1f deg", static_cast<double>(car.getSlipAngle() * RAD2DEG));
    DrawText(line, x, y, 18, RAYWHITE);
    y += lineHeight * 2;

    std::snprintf(line, sizeof(line), "Throttle: %.2f", static_cast<double>(input.throttle));
    DrawText(line, x, y, 18, RAYWHITE);
    y += lineHeight;

    std::snprintf(line, sizeof(line), "Steering: %.2f", static_cast<double>(input.steering));
    DrawText(line, x, y, 18, RAYWHITE);
    y += lineHeight * 2;

    DrawText("Sensor values (normalized / raw px):", x, y, 18, RAYWHITE);
    y += lineHeight;
    for (int i = 0; i < simulation::Car::kSensorCount; ++i)
    {
        const simulation::SensorReading& s = car.getSensors()[i];
        std::snprintf(line, sizeof(line), "%+4.0f deg: %.3f (%.1f px)",
                      static_cast<double>(simulation::Car::kSensorAngleDegrees[i]),
                      static_cast<double>(s.normalizedDistance), static_cast<double>(s.distance));
        DrawText(line, x, y, 16, LIGHTGRAY);
        y += lineHeight;
    }
    y += lineHeight;

    DrawText(car.isAlive() ? "ALIVE" : "CRASHED", x, y, 20, car.isAlive() ? GREEN : RED);
    y += lineHeight * 2;

    DrawText("Controls:", x, y, 18, RAYWHITE);
    y += lineHeight;
    DrawText("W / Up    - throttle", x, y, 16, LIGHTGRAY);
    y += lineHeight;
    DrawText("A / Left  - steer left", x, y, 16, LIGHTGRAY);
    y += lineHeight;
    DrawText("D / Right - steer right", x, y, 16, LIGHTGRAY);
    y += lineHeight;
    DrawText("R         - reset", x, y, 16, LIGHTGRAY);
}

} // namespace

int main()
{
    InitWindow(kScreenWidth, kScreenHeight, "NEAT Car Simulation");

    SetTargetFPS(60);

    simulation::Track track(makeTrackDefinition());
    verifyTrack(track);
    verifyCar(track);
    verifySensors(track);

    const simulation::TrackDefinition& def = track.getDefinition();

    simulation::Car car(makeCarParams(), track);
    car.reset(kSpawnPosition, kSpawnHeading);

    while (!WindowShouldClose())
    {
        if (IsKeyPressed(KEY_R))
        {
            car.reset(kSpawnPosition, kSpawnHeading);
        }

        const simulation::CarInput input = readInput();
        car.update(input, kSimulationDt);

        BeginDrawing();
        ClearBackground(BLACK);

        DrawRectangle(0, 0, kSimWidth, kSimHeight, BLACK);

        DrawEllipse(static_cast<int>(def.center.x), static_cast<int>(def.center.y),
                    def.outerRadiusX, def.outerRadiusY, GRAY);
        DrawEllipse(static_cast<int>(def.center.x), static_cast<int>(def.center.y),
                    def.innerRadiusX, def.innerRadiusY, BLACK);

        drawCar(car);
        drawPanel(car, input);

        EndDrawing();
    }

    CloseWindow();

    return 0;
}
