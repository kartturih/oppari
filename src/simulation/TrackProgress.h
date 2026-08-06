#pragma once

#include "raylib.h"

#include "simulation/Car.h"
#include "simulation/Track.h"

namespace simulation
{

// Tracks one car's ordered angular progress around an existing oval Track,
// derived purely from the car's world position and the Track's own
// TrackDefinition geometry. TrackProgress never reads Car input, dynamics,
// or collision handling beyond querying its current position -- it does not
// control the car, and it knows nothing about Genome, NeuralNetwork, or
// AIController.
//
// Progress representation (angle-to-progress mapping)
// -----------------------------------------------------
// The car's position is converted into an angle around the oval's center,
// normalized by the track's mid-band radii (the average of the inner and
// outer radii, per axis -- this only corrects for the ellipse's aspect
// ratio, it does not need to match the car's actual radial offset within
// the road band) so an elliptical track still yields a proper monotonic
// angular parameter:
//
//   dx = carX - center.x,  dy = carY - center.y
//   nx = dx / referenceRadiusX,  ny = dy / referenceRadiusY
//   rawAngle = atan2(ny, nx)                     in (-pi, pi]
//   lapPosition = -rawAngle / (2*pi), wrapped into [0, 1)
//
// The car spawns near the bottom of the oval (below center in screen-space
// y, i.e. dy > 0) facing +x ("right"). At that spawn point rawAngle = +90
// degrees. Driving forward from there increases x first (toward the east
// side of the oval), which decreases rawAngle toward 0 -- and it is exactly
// the negation of rawAngle in the formula above that turns that decrease
// into an *increase* in lapPosition. This mapping was verified numerically
// against the actual spawn pose before being written down here.
//
// Continuous progress and best progress are both anchored to zero at the
// most recent reset() -- they measure forward progress made *since spawn*,
// in laps, not the raw absolute angle (which starts at 0.75 for the current
// spawn point and has no inherent meaning on its own).
//
// Ordered checkpoint validation
// ------------------------------
// getBestProgress() is a pure angle measurement: it cannot tell a real lap
// of driving apart from a single implausible jump that happens to land back
// near the start (only the plausibility gate in update() defends against
// that). Checkpoints exist as a second, independent line of defense: they
// are validated purely from the sequence of raw lap-position deltas the car
// actually crossed, one boundary at a time, in strict order -- never
// derived from getBestProgress()'s value. kCheckpointCount checkpoints sit
// at fixed positions i / kCheckpointCount around the lap. On every update()
// whose delta survives the plausibility gate, TrackProgress walks forward
// from the currently expected checkpoint and awards each boundary actually
// crossed, one at a time and strictly in order (so several can be awarded
// in one update, but never out of sequence or ahead of where the car has
// actually been). Because the car can spawn/reset anywhere around the lap,
// the checkpoint the car starts on or just after is treated as the
// starting reference rather than being awarded; a lap is only counted once
// kCheckpointCount checkpoints have been awarded since that reference (or
// since the last completed lap) -- i.e. once the car has gone all the way
// around back to the same ring position it started from, in strict order --
// not when the angle-based best progress happens to cross an integer number
// of laps.
class TrackProgress
{
public:
    // Number of evenly spaced ordered checkpoints per lap, at fixed
    // positions i / kCheckpointCount for i = 0 .. kCheckpointCount - 1.
    static constexpr int kCheckpointCount = 16;

    // Throws std::invalid_argument if the definition is not a valid oval
    // (mirrors Track's own validation rules). TrackProgress performs this
    // check independently rather than trusting that the caller already
    // validated the same definition via Track.
    explicit TrackProgress(const TrackDefinition& definition);

    // Re-anchors all progress state to the car's current position: lap
    // position is recomputed from it, continuous progress and best progress
    // both reset to 0, and lap count / checkpoint counters reset to 0.
    void reset(const Car& car);

    // Advances progress using the car's current position. Computes the
    // seam-aware forward/backward delta since the previous update() (or
    // reset()), rejects any single-step delta larger than a physically
    // implausible amount (guarding against teleporting across the track
    // interior), and updates continuous progress and best progress from the
    // result. When the delta is accepted and net forward, also walks the
    // ordered checkpoint state forward by that same delta (see the class
    // comment) -- backward or rejected deltas leave checkpoint/lap state
    // untouched. Does not read CarInput and does not care whether the car
    // is alive.
    void update(const Car& car);

    // Current raw angular position within the lap, in [0, 1). This is the
    // absolute angle-derived position, not anchored to spawn -- see the
    // class comment for the exact mapping.
    float getLapPosition() const { return m_lapPosition; }

    // Forward progress made since the last reset, in laps (anchored to 0 at
    // reset). Can decrease if the car drives backward -- unlike
    // getBestProgress(), this is not an anti-exploit value on its own.
    float getContinuousProgress() const { return m_continuousProgress; }

    // The highest continuous progress ever reached since the last reset.
    // Monotonically non-decreasing: driving backward, or an implausible
    // single-frame jump, can never reduce it. This is the value fitness
    // scoring should treat as "progress".
    float getBestProgress() const { return m_bestProgress; }

    // Number of laps completed via ordered checkpoint validation (see the
    // class comment) -- incremented only once kCheckpointCount checkpoints
    // have been awarded in order since the last lap (or since reset).
    // Independent of getBestProgress(); monotonically non-decreasing.
    int getLapCount() const { return m_lapCount; }

    // Index (0 .. kCheckpointCount - 1) of the next checkpoint expected to
    // be passed, in order.
    int getExpectedCheckpoint() const { return m_expectedCheckpoint; }

    // Total ordered checkpoints passed since the last reset, counting every
    // completed lap's worth of checkpoints cumulatively rather than
    // resetting each lap. Awarded strictly in order from raw position
    // deltas -- never derived from getBestProgress(). Monotonically
    // non-decreasing.
    int getTotalCheckpointsPassed() const { return m_totalCheckpointsPassed; }

private:
    float computeLapPosition(Vector2 position) const;

    // Walks the ordered checkpoint state forward by forwardDelta (> 0, and
    // already known to have passed the plausibility gate): awards every
    // checkpoint boundary actually crossed, one at a time and strictly in
    // order starting from m_expectedCheckpoint, counting a completed lap
    // once kCheckpointCount checkpoints have been awarded since the last
    // lap (or since reset).
    void advanceCheckpoints(float forwardDelta);

    TrackDefinition m_definition;
    float m_referenceRadiusX = 0.0f; // mid-band radius used to normalize dx before atan2
    float m_referenceRadiusY = 0.0f; // mid-band radius used to normalize dy before atan2

    float m_lapPosition = 0.0f;
    float m_previousLapPosition = 0.0f;

    float m_continuousProgress = 0.0f;
    float m_bestProgress = 0.0f;

    int m_lapCount = 0;
    int m_expectedCheckpoint = 0;
    int m_totalCheckpointsPassed = 0;
    int m_checkpointsPassedSinceLap = 0; // resets to 0 each time it reaches kCheckpointCount (one lap)
};

} // namespace simulation
