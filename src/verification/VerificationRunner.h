#pragma once

#include "simulation/Track.h"

namespace verification
{

// Runs every deterministic startup verification check, in the same fixed
// order main() has always called them in. Aborts via assert() on the first
// failed check, exactly like the individual verify*() functions always have.
void runAll(const simulation::Track& track);

} // namespace verification
