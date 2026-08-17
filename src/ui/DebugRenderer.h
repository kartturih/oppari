#pragma once

#include <cstddef>

#include "raylib.h"

#include "simulation/Car.h"
#include "simulation/TrackProgress.h"

// Projection-debug visualization/diagnostics for the single highlighted
// individual -- never read by any production control/fitness path.
namespace ui
{

// Diagnostic overlay for the highlighted car's TrackProgress projection
// state -- never read by any production control/fitness path. Draws a
// marker at the projected centerline point, a line to the car's actual
// position, and an arrow for the centerline's forward tangent there.
// Colored magenta to stay distinct from the sensor rays drawn above.
void drawProjectionDebug(const simulation::Car& car, const simulation::TrackProgress& progress);

// Flags and logs a "suspicious" per-frame projection change for the
// highlighted individual -- a diagnostic aid for whether TrackProgress's
// local tracking snaps to a physically-nearby-but-topologically-distant
// centerline section (see TrackProgress.h's "Local projection tracking"
// class comment). Purely reads this frame's already-computed
// ProjectionDebugInfo; never affects fitness, progress, or control.
void reportSuspiciousProjectionJump(std::size_t highlightedIndex, const simulation::TrackProgress& progress,
                                     const simulation::Car& car);

} // namespace ui
