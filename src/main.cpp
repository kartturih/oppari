#include <cassert>

#include "raylib.h"

#include "simulation/Track.h"

namespace
{

constexpr int kSimWidth = 1200;
constexpr int kSimHeight = 700;
constexpr int kPanelWidth = 400;
constexpr int kScreenWidth = kSimWidth + kPanelWidth;
constexpr int kScreenHeight = kSimHeight;

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

} // namespace

int main()
{
    InitWindow(kScreenWidth, kScreenHeight, "NEAT Car Simulation");

    SetTargetFPS(60);

    simulation::Track track(makeTrackDefinition());
    verifyTrack(track);

    const simulation::TrackDefinition& def = track.getDefinition();

    while (!WindowShouldClose())
    {
        BeginDrawing();
        ClearBackground(BLACK);

        DrawRectangle(0, 0, kSimWidth, kSimHeight, BLACK);

        DrawEllipse(static_cast<int>(def.center.x), static_cast<int>(def.center.y),
                    def.outerRadiusX, def.outerRadiusY, GRAY);
        DrawEllipse(static_cast<int>(def.center.x), static_cast<int>(def.center.y),
                    def.innerRadiusX, def.innerRadiusY, BLACK);

        DrawRectangle(kSimWidth, 0, kPanelWidth, kScreenHeight, Color{30, 30, 30, 255});

        EndDrawing();
    }

    CloseWindow();

    return 0;
}
