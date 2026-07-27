#include "raylib.h"

int main()
{
    const int screenWidth = 1600;
    const int screenHeight = 700;

    InitWindow(screenWidth, screenHeight, "NEAT Car Simulation");

    SetTargetFPS(60);

    while (!WindowShouldClose())
    {
        BeginDrawing();
        ClearBackground(RAYWHITE);

        DrawText("NEAT Car Simulation", 20, 20, 30, DARKGRAY);

        EndDrawing();
    }

    CloseWindow();

    return 0;
}
