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
}

void drawPanel(const simulation::Car& car, const simulation::CarInput& input)
{
    DrawRectangle(kSimWidth, 0, kPanelWidth, kScreenHeight, Color{30, 30, 30, 255});

    const int x = kSimWidth + 20;
    int y = 20;
    const int lineHeight = 22;

    DrawText("STAGE 2 - MANUAL DRIVING", x, y, 20, RAYWHITE);
    y += lineHeight * 2;

    const Vector2 velocity = car.getVelocity();
    const float speed = std::sqrt(velocity.x * velocity.x + velocity.y * velocity.y);

    char line[128];

    std::snprintf(line, sizeof(line), "Speed: %.1f px/s", static_cast<double>(speed));
    DrawText(line, x, y, 18, RAYWHITE);
    y += lineHeight;

    std::snprintf(line, sizeof(line), "Throttle: %.2f", static_cast<double>(input.throttle));
    DrawText(line, x, y, 18, RAYWHITE);
    y += lineHeight;

    std::snprintf(line, sizeof(line), "Steering: %.2f", static_cast<double>(input.steering));
    DrawText(line, x, y, 18, RAYWHITE);
    y += lineHeight * 2;

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
