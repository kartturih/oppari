#pragma once

#include "simulation/Car.h"

namespace input
{

// Reads a CarInput straight from the keyboard for manual physics-
// verification mode (see main()'s manualMode branch). Arrow keys and
// WASD both work. No braking/reverse channel: the same two-channel
// CarInput (throttle in [0,1], steering in [-1,1]) the AI path uses --
// releasing throttle is the only way to slow down, by design (see
// CarParams::engineForce's comment).
simulation::CarInput readManualCarInput();

} // namespace input
