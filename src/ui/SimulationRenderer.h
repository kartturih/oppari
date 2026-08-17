#pragma once

#include "raylib.h"

#include "simulation/Car.h"

// Ordinary per-frame simulation/car rendering -- no HUD text, no debug
// overlays.
namespace ui
{

// Maps a normalized progress rank (0 = best/leading, 1 = worst/trailing)
// to a color running green -> yellow -> red, for population-wide car
// rendering. Purely a rendering helper -- reads no evolution state and
// writes nothing.
Color progressRankColor(float normalizedRank);

// Draws one population member's car body in `color`. Sensor rays and the
// velocity-vector overlay are drawn only for the single highlighted individual.
void drawIndividualCar(const simulation::Car& car, Color color, bool highlighted);

} // namespace ui
