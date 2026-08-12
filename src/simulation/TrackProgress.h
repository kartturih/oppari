#pragma once

#include <cstddef>

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
// Local projection tracking (Stage 18)
// -----------------------------------------------------
// projectOntoCenterline() finds the closest centerline sample by straight-
// line distance over the *entire* closed loop, with no notion of "the
// segment the car was near last frame". On a dense/complex track where two
// non-adjacent stretches of centerline pass close together, that full scan
// could jump between them frame to frame even though the car moved only a
// short physical distance -- and because the two stretches can sit at very
// different arc-length distances, that jump could look like a large
// forward or backward lap-position delta.
//
// TrackProgress defends against this in two layers:
//
//   1. Local tracking (the normal case): every update() remembers the
//      previously projected segment index (m_previousSegmentIndex) and
//      queries Track::projectOntoCenterlineLocal() with it, which only
//      searches a small window of kLocalSearchRadius segments on either
//      side (wrapping correctly at the closed loop's seam). A physically
//      nearby but logically distant section of centerline, several hundred
//      segments away in index space, is never even considered -- the
//      ambiguity is structurally excluded, not just detected after the
//      fact.
//
//      Stage 19.1: excluding far-away segments by window alone is not
//      enough on a track whose *tight corners* fold two much-closer-in-
//      index (tens, not hundreds, of samples apart) points of the same
//      curve near each other -- both still legitimately inside the window.
//      Pure nearest-distance selection has no reason to prefer the one
//      that's a small, continuous step from where tracking already was, so
//      it can snap forward/backward within the window whenever the "wrong"
//      candidate is even marginally closer. projectOntoCenterlineLocal()'s
//      continuityWeight parameter (see kLocalContinuityWeight below) fixes
//      this by scoring window candidates on distance *plus* a penalty
//      proportional to their index gap from m_previousSegmentIndex, so a
//      close continuation reliably beats a marginally-closer-but-far-in-
//      index candidate without weakening the window exclusion itself.
//   2. Global recovery (the exception): if the local window's best match is
//      still farther than kLocalProjectionRecoveryDistance away, the local
//      search has clearly failed (the car cannot really be that far from
//      every segment near where it was last tracked) -- this can happen
//      right after reset()/generation transition, or from an unusual
//      collision/teleport state. In that case a single full
//      projectOntoCenterline() scan re-anchors m_previousSegmentIndex.
//      This is recovery only, not a routine per-frame path.
//
// Both paths feed the exact same lapPosition into update()'s existing
// plausibility gate (kMaxPlausibleLapDeltaPerFrame) before any progress or
// checkpoint credit is awarded -- recovery re-anchors *which segment* is
// tracked, it never bypasses that gate, so an implausible jump is still
// rejected for progress purposes even if recovery had to run to explain it.
class TrackProgress
{
public:
    // Number of evenly spaced ordered checkpoints per lap, at fixed
    // arc-length positions i / kCheckpointCount for i = 0 .. kCheckpointCount - 1.
    static constexpr int kCheckpointCount = 16;

    // Local search window radius (in centerline segments, each side of
    // m_previousSegmentIndex) used by every normal per-frame update() -- see
    // the class comment. Real per-frame movement needs only a tiny radius
    // (at maxSpeed 260px/s and the fixed 1/60s step, a car moves ~4.3px per
    // frame, far less than one segment on any Stage 18/19 track).
    //
    // Stage 19 retuning: the Stage 18 test track sampled 288 centerline
    // segments (12 control points x 24 samples/segment); the Stage 19
    // extreme track (see createExtremeTrackDefinition()) samples 780 (39 x
    // 20) to capture its much more complex route. main.cpp's own
    // verifyTrackProgress()/verifyFitnessEvaluator() suites exercise
    // update() with large synthetic position jumps (via Car::reset() +
    // Track::getPointAtDistance()) expressed as fractions of a lap, up to
    // ~0.19 of a lap -- a track-agnostic fraction, but the equivalent
    // *segment* count scales with total sample count: ~55 segments at 288
    // samples (which the old radius of 60 covered), ~148 segments at 780
    // samples (which it no longer would). Old value: 60. New value: 160 --
    // chosen to keep those same legitimate fractional-lap jumps an exact,
    // non-recovery local match with a comfortable margin (~8% above the
    // ~148 segments actually needed), while staying well under half of a
    // full lap's segment count (390) so a genuinely distant, logically
    // unrelated section is still excluded from the window. This was
    // verified against the extreme track's actual geometry (see Track.cpp):
    // within a ~100-segment index gap, samples can come within ~62px of
    // each other, but that is ordinary tight-corner curvature (both samples
    // sit on the entry/exit of the SAME turn -- not a different, logically
    // unrelated part of the route, so a local window spanning both is
    // harmless). The closest that two segments MORE than ~100 apart --
    // genuinely distant, unrelated stretches of the route -- ever come
    // together is ~94px, and that only happens at a ~316-segment index gap
    // -- radius 160 leaves a wide (156-segment) buffer below that gap, so
    // the window can never reach it. Deliberately, the plausibility-gate
    // test case in verifyTrackProgress() uses a 0.35-lap/~273-segment jump
    // specifically to land outside even this wider window and exercise
    // recovery.
    static constexpr std::size_t kLocalSearchRadius = 160;

    // Distance (px) beyond which a local projection's result is treated as
    // "clearly failed" and a global recovery scan runs instead -- see the
    // class comment. Set comfortably larger than the widest drivable band
    // any Stage 18/19 track configures (190px on the Stage 18 test track's
    // broadest sweep; ~202px on the Stage 19 extreme track's widest
    // corridor, re-measured directly against its mask -- see Track.cpp) and
    // than Car::kMaxSensorDistance (200px), so ordinary on-mask driving --
    // which keeps the car within the local drivable band -- never triggers
    // it, while a stale/uninitialized m_previousSegmentIndex reliably does.
    // Unchanged from Stage 18: still comfortably above both tracks' widest
    // band with the same margin logic.
    static constexpr float kLocalProjectionRecoveryDistance = 250.0f;

    // Stage 19.1 bugfix -- root cause: on the extreme track's tight
    // top-left corner (a near-hairpin -- see createExtremeTrackDefinition()'s
    // P2-P6), cars observed via debug instrumentation would have their
    // tracked segment snap forward by 9-65 samples in a single frame
    // (physically impossible -- real per-frame movement is under 1 sample),
    // with a large tangent discontinuity, then resume normal small-delta
    // tracking from the wrong spot. Investigation (see
    // reportSuspiciousProjectionJump() in main.cpp) found the cause: pure
    // nearest-distance selection had no preference for continuing near the
    // previously tracked segment, so whenever a point further into the
    // hairpin (a different arc-length stretch that curves back close to the
    // entry -- geometrically near, topologically far, just at a much
    // smaller index gap than the cross-corridor case Stage 19 originally
    // tuned kLocalSearchRadius/kLocalProjectionRecoveryDistance for) became
    // even marginally closer than the true continuation, local search
    // snapped to it.
    //
    // Track::projectOntoCenterlineLocal()'s continuityWeight/
    // continuityFreeZone parameters fix this: within the search window,
    // candidates are scored by distance + kLocalContinuityWeight *
    // sqrt(max(0, indexGap - kLocalContinuityFreeZone)) instead of raw
    // distance alone (see considerSegmentLocal()). Reaching these two
    // values (and the square-root, not linear, shape of the penalty) took
    // three rounds of empirical measurement, because three different
    // failure modes had to be satisfied simultaneously:
    //   1. Too small a penalty leaves the hairpin ambiguity unresolved.
    //   2. Too large a *linear* penalty breaks legitimate large single-frame
    //      local jumps this class's own verification suite depends on (e.g.
    //      verifyFitnessEvaluator() requirement 28/29 needs one local
    //      update() to jump ~98 samples at true distance 0) -- a linear
    //      per-sample rate large enough to win the hairpin's small-gap ties
    //      compounds, over 98+ samples, into a penalty that starts beating
    //      the true (zero-distance) far target outright. Switching to
    //      sqrt(indexGap) fixes this: steep enough at small gaps, far
    //      gentler at large ones (sqrt(160) ~= 12.6 vs. linear 160).
    //   3. Without a free zone, even a small *sqrt* penalty makes ORDINARY
    //      forward progress refuse to advance: once anchored on an
    //      already-close-fitting candidate (small distance), even one
    //      sample of real forward movement costs more than the tiny real
    //      distance improvement it provides, so tracking never advances at
    //      all. Confirmed directly: a synthetic car driven smoothly through
    //      this same hairpin (no synthetic teleporting, small continuous
    //      per-frame steps) stayed pinned to its entry segment for 40+
    //      consecutive frames, reported distance climbing past 90px, with
    //      no free zone. kLocalContinuityFreeZone samples are exempted from
    //      the penalty entirely before the square-root cost applies beyond
    //      it, restoring smooth forward tracking (the same synthetic drive
    //      test lags the true position by at most ~8 samples at any point,
    //      never gets stuck for more than a handful of consecutive frames,
    //      and lands within 4 samples of the true final position).
    //
    // With free zone 5, every legitimate jump in this suite (the
    // determinism check above, the lap-completion sequence, the
    // 16-checkpoint traversal, and a broad synthetic sweep of ~200 forward/
    // backward jumps of varying size from many points around the loop)
    // remains correct for weight up to at least 15, while every collected
    // hairpin case is fully resolved (tracked index within the free zone of
    // the true match) from weight 9 upward. 9 was chosen at the safe side
    // of that window rather than its extreme edge. See the Stage 19.1
    // report for the full derivation (including earlier, smaller
    // weights/formulas that each fixed every bug instance known at the
    // time but were then found insufficient -- either against a wider
    // sample of live population runs, or against the smooth-driving check
    // above), and verifyImageBasedTrackSystem()'s Stage 19.1 regression
    // case for a constructed close-but-topologically-distant scenario
    // exercising this directly. Both constants apply only to
    // TrackProgress's own per-frame local tracking (passed explicitly to
    // projectOntoCenterlineLocal()); Track's global scan (reset()/recovery)
    // and any other direct caller of projectOntoCenterlineLocal() that
    // doesn't pass them are unaffected (both parameters default to 0).
    static constexpr float kLocalContinuityWeight = 9.0f;

    // Index-gap samples exempted from the continuity penalty entirely --
    // see kLocalContinuityWeight's comment for why this is necessary at
    // all (failure mode 3) and how 5 was derived (comfortably above real
    // per-frame movement, which needs under 1 sample; small enough that it
    // never meaningfully widens which candidates are competitive at the
    // gaps -- tens of samples -- where the hairpin ambiguity actually
    // occurs).
    static constexpr std::size_t kLocalContinuityFreeZone = 5;

    // Stores a reference to track (never copies its geometry) -- every
    // lap-position computation goes through Track::projectOntoCenterline()
    // or Track::projectOntoCenterlineLocal(). Track's own constructor is
    // responsible for validating the TrackDefinition it was built from;
    // TrackProgress performs no redundant validation of its own.
    explicit TrackProgress(const Track& track);

    // Re-anchors all progress state to the car's current position: lap
    // position is recomputed from it via a full global projection (there is
    // no meaningful "previous segment" yet, so local tracking does not
    // apply here), m_previousSegmentIndex is re-anchored to that
    // projection's segment, and continuous progress/best progress/lap
    // count/checkpoint counters all reset to 0.
    void reset(const Car& car);

    // Advances progress using the car's current position. Projects the
    // position via local tracking (falling back to global recovery only if
    // that clearly fails -- see the class comment), computes the seam-aware
    // forward/backward delta since the previous update() (or reset()),
    // rejects any single-step delta larger than a physically implausible
    // amount (guarding against teleporting across the track interior), and
    // updates continuous progress and best progress from the result. When
    // the delta is accepted and net forward, also walks the ordered
    // checkpoint state forward by that same delta (see the class comment)
    // -- backward or rejected deltas leave checkpoint/lap state untouched.
    // Does not read CarInput and does not care whether the car is alive.
    void update(const Car& car);

    // The centerline segment index this TrackProgress is currently tracking
    // (i.e. the center of the next update()'s local search window).
    // Exposed only for deterministic verification of the local-tracking/
    // recovery behavior itself -- no production code path needs to read it.
    std::size_t getTrackedSegmentIndex() const { return m_previousSegmentIndex; }

    // Debug-only snapshot of the most recent update()'s projection outcome
    // -- exists purely to support optional runtime visualization/diagnosis
    // of local-tracking/recovery behavior (see main.cpp's highlighted-car
    // projection debug rendering, added for Stage 19.1). Nothing in the
    // production control/fitness path (Car, AIController, FitnessEvaluator,
    // Population) reads this; it is a pure read-only diagnostic mirror of
    // state update() already computed.
    struct ProjectionDebugInfo
    {
        Vector2 point = {0.0f, 0.0f};           // projected point on the centerline
        Vector2 tangent = {0.0f, 0.0f};         // forward tangent (unit vector) at `point`
        Vector2 previousTangent = {0.0f, 0.0f}; // forward tangent at the previously tracked point
        std::size_t previousIndex = 0;          // tracked segment index before this update()
        std::size_t currentIndex = 0;           // tracked segment index after this update()
        int indexDelta = 0;                     // signed, shortest-direction, wrap-aware: currentIndex - previousIndex
        float distance = 0.0f;                  // distanceFromCenterline of the projection used
        bool usedRecovery = false;               // true if global recovery ran this update(), false if local tracking succeeded
    };
    const ProjectionDebugInfo& getLastProjectionDebugInfo() const { return m_lastProjectionDebugInfo; }

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
    // Local-first, global-recovery projection used by every update() -- see
    // the class comment. Also updates m_previousSegmentIndex to whichever
    // segment the returned projection landed on, and sets
    // m_lastProjectionUsedRecovery to record which path was taken (read by
    // update() when it fills in m_lastProjectionDebugInfo).
    TrackProjection projectWithLocalTracking(Vector2 position);

    // Forward tangent (unit vector) of the centerline segment starting at
    // `segmentIndex`, i.e. the direction from that sample toward the next
    // one. Debug-only (feeds ProjectionDebugInfo); never used by progress/
    // checkpoint math itself.
    Vector2 tangentAtSegment(std::size_t segmentIndex) const;

    // Walks the ordered checkpoint state forward by forwardDelta (> 0, and
    // already known to have passed the plausibility gate): awards every
    // checkpoint boundary actually crossed, one at a time and strictly in
    // order starting from m_expectedCheckpoint, counting a completed lap
    // once kCheckpointCount checkpoints have been awarded since the last
    // lap (or since reset).
    void advanceCheckpoints(float forwardDelta);

    const Track& m_track;

    std::size_t m_previousSegmentIndex = 0;
    bool m_lastProjectionUsedRecovery = false;
    ProjectionDebugInfo m_lastProjectionDebugInfo;

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
