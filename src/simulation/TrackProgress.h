#pragma once

#include <cstddef>

#include "raylib.h"

#include "simulation/Car.h"
#include "simulation/Track.h"

namespace simulation
{

// Tracks one car's ordered progress around a closed Track, from the car's
// world position and the Track's sampled centerline/arc-length data. Reads
// only the car's position -- no input, dynamics, collision, Genome, or
// NeuralNetwork knowledge. Track is the sole source of centerline geometry.
//
// Progress: position projects onto the centerline; lapPosition =
// distanceAlongTrack / totalLength, in [0,1). continuousProgress/bestProgress
// are anchored to 0 at reset() and measure forward progress since spawn, in
// laps -- bestProgress is monotonic (anti-exploit) and is what fitness
// should treat as "progress".
//
// Checkpoints: a second, independent defense against a single implausible
// jump masquerading as a lap. kCheckpointCount checkpoints sit at fixed
// arc-length fractions; each update() walks forward from the expected one,
// awarding boundaries actually crossed in strict order (never derived from
// bestProgress). A lap counts once all checkpoints are awarded since the
// last lap/reset -- not when bestProgress crosses an integer.
//
// Local projection tracking: a full-centerline projectOntoCenterline() scan
// could jump between two geometrically close but topologically unrelated
// stretches of a dense track. update() instead tracks the previous segment
// and searches only a small local window (projectOntoCenterlineLocal()),
// weighted to prefer continuing near that segment (fixes tight-corner
// snapping -- see kLocalContinuityWeight). If the local match is still too
// far away (kLocalProjectionRecoveryDistance), a one-off global scan
// re-anchors it. Either way the result still passes update()'s plausibility
// gate before affecting progress/checkpoints.
class TrackProgress
{
public:
    static constexpr int kCheckpointCount = 16;

    // Local search window radius (segments, each side of the tracked
    // segment). Sized so legitimate per-frame/synthetic-test jumps land as
    // an exact local match (not recovery) while staying under half a lap,
    // so genuinely distant sections stay excluded -- see Track.cpp for the
    // extreme track's measured segment spacing this was tuned against.
    static constexpr std::size_t kLocalSearchRadius = 160;

    // Distance (px) beyond which a local match is "clearly failed" and
    // global recovery runs instead. Set above the widest drivable band on
    // any active track and above Car::kMaxSensorDistance, so ordinary
    // driving never triggers it but a stale tracked segment reliably does.
    static constexpr float kLocalProjectionRecoveryDistance = 250.0f;

    // Continuity penalty weight/free-zone for projectOntoCenterlineLocal()
    // (distance + kLocalContinuityWeight * sqrt(max(0, indexGap -
    // kLocalContinuityFreeZone))): fixes local tracking snapping to a
    // nearby-in-distance-but-far-in-index point on a tight hairpin, while
    // the free zone keeps ordinary forward movement from being penalized
    // against standing still. Values were tuned empirically against a
    // hairpin regression case and a broad synthetic driving sweep -- see
    // Track.cpp's considerSegmentLocal() for the scoring itself.
    static constexpr float kLocalContinuityWeight = 9.0f;
    static constexpr std::size_t kLocalContinuityFreeZone = 5;

    explicit TrackProgress(const Track& track);

    // Re-anchors everything to the car's current position via a full global
    // projection; continuous/best progress, lap count, and checkpoints all
    // reset to 0.
    void reset(const Car& car);

    // Advances from the car's current position: local-tracked projection
    // (recovery if needed), seam-aware delta since the last update/reset,
    // rejects an implausibly large single-step delta, then updates
    // continuous/best progress and (if net forward) checkpoints. Does not
    // read CarInput or care whether the car is alive.
    void update(const Car& car);

    // Centerline segment currently tracked -- exposed only for verification.
    std::size_t getTrackedSegmentIndex() const { return m_previousSegmentIndex; }

    // Debug snapshot of the most recent update()'s projection outcome, for
    // optional runtime visualization -- not read by any production path.
    struct ProjectionDebugInfo
    {
        Vector2 point = {0.0f, 0.0f};
        Vector2 tangent = {0.0f, 0.0f};
        Vector2 previousTangent = {0.0f, 0.0f};
        std::size_t previousIndex = 0;
        std::size_t currentIndex = 0;
        int indexDelta = 0; // signed, shortest-direction, wrap-aware
        float distance = 0.0f;
        bool usedRecovery = false;
    };
    const ProjectionDebugInfo& getLastProjectionDebugInfo() const { return m_lastProjectionDebugInfo; }

    // Local track tangent (unit vector, forward direction of travel) at the
    // most recently tracked/projected point -- the SAME underlying value as
    // ProjectionDebugInfo::tangent, but exposed as a genuine, intentional
    // production-facing accessor (see ai::buildObservation(), which reads
    // this for the heading-error observation) rather than through the
    // debug-only struct above. {0,0} only before the first reset()/update()
    // call has ever run (never a live production state).
    Vector2 getTrackTangent() const { return m_trackTangent; }

    // Raw arc-length-normalized lap position, [0,1) -- not anchored to spawn.
    float getLapPosition() const { return m_lapPosition; }

    // Forward progress since reset, in laps; can decrease if driving backward.
    float getContinuousProgress() const { return m_continuousProgress; }

    // Highest continuousProgress reached since reset; monotonic (the value
    // fitness scoring should treat as "progress").
    float getBestProgress() const { return m_bestProgress; }

    // Laps completed via ordered checkpoints; independent of bestProgress, monotonic.
    int getLapCount() const { return m_lapCount; }

    int getExpectedCheckpoint() const { return m_expectedCheckpoint; }

    // Total checkpoints passed since reset (cumulative across laps); monotonic.
    int getTotalCheckpointsPassed() const { return m_totalCheckpointsPassed; }

private:
    // Local-first, global-recovery projection; updates m_previousSegmentIndex.
    TrackProjection projectWithLocalTracking(Vector2 position);

    // Forward tangent of the segment starting at segmentIndex. Debug-only.
    Vector2 tangentAtSegment(std::size_t segmentIndex) const;

    // Walks checkpoint state forward by forwardDelta (> 0, already gated).
    void advanceCheckpoints(float forwardDelta);

    const Track& m_track;

    std::size_t m_previousSegmentIndex = 0;
    bool m_lastProjectionUsedRecovery = false;
    ProjectionDebugInfo m_lastProjectionDebugInfo;

    // Mirrors m_lastProjectionDebugInfo.tangent -- see getTrackTangent()'s
    // comment for why this is kept as its own, separately-exposed member
    // rather than reading the debug struct from production code.
    Vector2 m_trackTangent = {0.0f, 0.0f};

    float m_lapPosition = 0.0f;
    float m_previousLapPosition = 0.0f;

    float m_continuousProgress = 0.0f;
    float m_bestProgress = 0.0f;

    int m_lapCount = 0;
    int m_expectedCheckpoint = 0;
    int m_totalCheckpointsPassed = 0;
    int m_checkpointsPassedSinceLap = 0; // resets to 0 at kCheckpointCount
};

} // namespace simulation
