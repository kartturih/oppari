#include "input/ManualInput.h"

#include "raylib.h"

namespace input
{

simulation::CarInput readManualCarInput()
{
    simulation::CarInput input;
    input.throttle = (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W)) ? 1.0f : 0.0f;
    float steering = 0.0f;
    if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A))
    {
        steering -= 1.0f;
    }
    if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D))
    {
        steering += 1.0f;
    }
    input.steering = steering;
    return input;
}

} // namespace input
