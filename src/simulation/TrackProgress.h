#pragma once

#include "raylib.h"

#include "simulation/Car.h"
#include "simulation/Track.h"

namespace simulation
{

// Tracks one car's ordered progress around an existing closed Track, derived
// purely from the car's world position and the Track's own sampled
// centerline/arc-length data (via Track::projectOntoCenterline()).
// TrackProgress never reads Car input, dynamics, or collision handling
// beyond querying its current position -- it does not control the car, and
// it knows nothing about Genome, NeuralNetwork, or AIController. It also
// never builds or maintains any track geometry of its own: the Track passed
// to it is the single source of truth for the centerline, and TrackProgress
// only ever calls Track::projectOntoCenterline() and Track::getTotalLength().
//
// Progress representation (arc-length-based lap position)
// ---------------------------------------------------------
// The car's position is projected onto the Track's sampled centerline, and
// that projection's arc-length distance is normalized by the track's total
// length:
//
//   projection = track.projectOntoCenterline(carPosition)
//   lapPosition = projection.distanceAlongTrack / track.getTotalLength()   in [0, 1)
//
// The forward direction is entirely defined by the Track's own centerline
// sample order (increasing sample index / arc length) -- see Track.h and
// createEasyTrackDefinition() for how Stage 14A's control points are chosen
// so that direction matches the old ellipse track's forward direction, and
// so that Track::getSpawnPosition()/getSpawnHeading() land at (and face
// along) that same direction.
//
// Continuous progress and best progress are both anchored to zero at the
// most recent reset() -- they measure forward progress made *since spawn*,
// in laps (i.e. in units of one full trip around the total track length),
// not the raw absolute lap position (which has no inherent meaning of its
// own beyond where the car happens to be on the loop).
//
// Ordered checkpoint validation
// ------------------------------
// getBestProgress() is a pure arc-length measurement: it cannot tell a real
// lap of driving apart from a single implausible jump that happens to land
// back near the start (only the plausibility gate in update() defends
// against that). Checkpoints exist as a second, independent line of
// defense: they are validated purely from the sequence of raw lap-position
// deltas the car actually crossed, one boundary at a time, in strict order
// -- never derived from getBestProgress()'s value. kCheckpointCount
// checkpoints sit at fixed positions i / kCheckpointCount around the lap
// (i.e. at fixed arc-length fractions of the total track length -- not
// placed by angle). On every update() whose delta survives the plausibility
// gate, TrackProgress walks forward from the currently expected checkpoint
// and awards each boundary actually crossed, one at a time and strictly in
// order (so several can be awarded in one update, but never out of sequence
// or ahead of where the car has actually been). Because the car can
// spawn/reset anywhere around the lap, the checkpoint the car starts on or
// just after is treated as the starting reference rather than being
// awarded; a lap is only counted once kCheckpointCount checkpoints have
// been awarded since that reference (or since the last completed lap) --
// i.e. once the car has gone all the way around back to the same ring
// position it started from, in strict order -- not when the arc-length-
// based best progress happens to cross an integer number of laps.
//
// Local projection ambiguity (documented limitation)
// -----------------------------------------------------
// projectOntoCenterline() finds the closest centerline sample by straight-
// line distance, with no notion of "the segment the car was near last
// frame". On a future track where two non-adjacent stretches of centerline
// pass close together (e.g. a self-intersecting or tightly folded track),
// the projection could jump from one stretch to the other between frames
// even though the car moved only a short physical distance -- and because
// the two stretches can sit at very different arc-length distances, that
// jump could look like a large forward or backward lap-position delta. This
// class does not attempt to resolve that ambiguity (no persistent nearest-
// segment tracking, no self-intersection solver): the existing plausibility
// gate in update() (see kMaxPlausibleLapDeltaPerFrame) is the only defense,
// and it defends by magnitude of the delta, not by understanding *why* a
// large delta occurred. Stage 14A and 14B's tracks are specified not to
// self-intersect, so this limitation is not exercised yet; a future
// self-intersecting track would need a real fix here (e.g. biasing
// projection toward the previous frame's segment index) before this class
// could be trusted on it.
class TrackProgress
{
public:
    // Number of evenly spaced ordered checkpoints per lap, at fixed
    // arc-length positions i / kCheckpointCount for i = 0 .. kCheckpointCount - 1.
    static constexpr int kCheckpointCount = 16;

    // Stores a reference to track (never copies its geometry) -- every
    // lap-position computation goes through track.projectOntoCenterline().
    // Track's own constructor is responsible for validating the
    // TrackDefinition it was built from; TrackProgress performs no
    // redundant validation of its own.
    explicit TrackProgress(const Track& track);

    // Re-anchors all progress state to the car's current position: lap
    // position is recomputed from it, continuous progress and best progress
    // both reset to 0, and lap count / checkpoint counters reset to 0.
    void reset(const Car& car);

    // Advances progress using the car's current position. Computes the
    // seam-aware forward/backward delta since the previous update() (or
    // reset()), rejects any single-step delta larger than a physically
    // implausible amount (guarding against teleporting across the track
    // interior, or -- on a future self-intersecting track -- a local
    // projection ambiguity jump; see the class comment), and updates
    // continuous progress and best progress from the result. When the delta
    // is accepted and net forward, also walks the ordered checkpoint state
    // forward by that same delta (see the class comment) -- backward or
    // rejected deltas leave checkpoint/lap state untouched. Does not read
    // CarInput and does not care whether the car is alive.
    void update(const Car& car);

    // Current raw arc-length-normalized position within the lap, in [0, 1).
    // This is the absolute projected position, not anchored to spawn -- see
    // the class comment for the exact mapping.
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

    const Track& m_track;

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
