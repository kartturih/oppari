#pragma once

#include "simulation/Car.h"

namespace input
{

// Reads a CarInput straight from the keyboard for manual physics-
// verification mode (see main()'s manualMode branch). Arrow keys and WASD
// both work: UP/W throttle, DOWN/S brake, LEFT/A and RIGHT/D steer. Throttle
// and brake are independent (both may be held at once), matching the AI
// path's CarInput exactly -- see Car.cpp's friction circle for why that's
// self-defeating rather than forbidden.
simulation::CarInput readManualCarInput();

} // namespace input
