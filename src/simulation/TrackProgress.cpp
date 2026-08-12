#include "simulation/TrackProgress.h"

#include <cmath>

namespace simulation
{

namespace
{

// Magnitude a lapPosition delta must exceed, relative to the previous
// update, before it is treated as a 0/1 lap-boundary (seam) crossing rather
// than ordinary local movement. Dimensionless (a fraction of one lap), so
// this does not need to change with track shape or scale.
constexpr float kSeamWrapThreshold = 0.5f;

// Maximum lapPosition delta (after seam correction) accepted as legitimate
// forward/backward movement in a single update(). Also dimensionless: since
// lapPosition is already normalized by the track's own total length, this
// threshold's meaning does not depend on track scale or shape. It is set
// well above real per-frame movement (at maxSpeed 260px/s and the fixed
// 1/60s simulation step, the car travels at most ~4.3px per frame, a small
// fraction of any reasonably sized closed loop) while staying generous
// enough that a single update() spanning several checkpoints
// (kCheckpointCount = 16, so one checkpoint gap is 1/16 = 0.0625 laps) is
// still accepted deterministically -- and far below both real driving and
// the kSeamWrapThreshold ambiguity boundary, so it reliably rejects an
// implausible jump such as teleporting straight across the track interior.
constexpr float kMaxPlausibleLapDeltaPerFrame = 0.2f;

} // namespace

TrackProgress::TrackProgress(const Track& track) : m_track(track)
{
}

TrackProjection TrackProgress::projectWithLocalTracking(Vector2 position)
{
    const TrackProjection local = m_track.projectOntoCenterlineLocal(
        position, m_previousSegmentIndex, kLocalSearchRadius, kLocalContinuityWeight, kLocalContinuityFreeZone);

    if (local.distanceFromCenterline <= kLocalProjectionRecoveryDistance)
    {
        // Normal case: local tracking succeeded (see the class comment for
        // why kLocalSearchRadius is generous enough that this is the common
        // path on every ordinary frame of driving).
        m_previousSegmentIndex = local.segmentIndex;
        m_lastProjectionUsedRecovery = false;
        return local;
    }

    // Recovery: the local window is clearly wrong -- the car cannot really
    // be more than kLocalProjectionRecoveryDistance from every segment near
    // where it was last tracked. Re-anchor via one full-centerline scan.
    // This only changes *which* segment/lapPosition are reported; the
    // caller (update()) still runs its own plausibility delta gate on
    // whatever lapPosition this produces, so recovery cannot award
    // implausible progress on its own.
    const TrackProjection global = m_track.projectOntoCenterline(position);
    m_previousSegmentIndex = global.segmentIndex;
    m_lastProjectionUsedRecovery = true;
    return global;
}

Vector2 TrackProgress::tangentAtSegment(std::size_t segmentIndex) const
{
    const std::vector<Vector2>& centerline = m_track.getCenterlineSamples();
    const std::size_t n = centerline.size();
    const Vector2& a = centerline[segmentIndex % n];
    const Vector2& b = centerline[(segmentIndex + 1) % n];
    const float dx = b.x - a.x;
    const float dy = b.y - a.y;
    const float len = std::sqrt(dx * dx + dy * dy);
    if (len <= 0.0f)
    {
        return Vector2{0.0f, 0.0f};
    }
    return Vector2{dx / len, dy / len};
}

void TrackProgress::reset(const Car& car)
{
    // reset() always uses a full global projection: there is no meaningful
    // "previous segment" to search locally around yet.
    const TrackProjection projection = m_track.projectOntoCenterline(car.getPosition());
    m_previousSegmentIndex = projection.segmentIndex;
    m_lastProjectionUsedRecovery = false;

    // Debug info: previous == current and delta == 0 -- reset() re-anchors
    // from scratch, so there is no meaningful "jump" to report here.
    const Vector2 tangent = tangentAtSegment(projection.segmentIndex);
    m_lastProjectionDebugInfo.point = projection.point;
    m_lastProjectionDebugInfo.tangent = tangent;
    m_lastProjectionDebugInfo.previousTangent = tangent;
    m_lastProjectionDebugInfo.previousIndex = projection.segmentIndex;
    m_lastProjectionDebugInfo.currentIndex = projection.segmentIndex;
    m_lastProjectionDebugInfo.indexDelta = 0;
    m_lastProjectionDebugInfo.distance = projection.distanceFromCenterline;
    m_lastProjectionDebugInfo.usedRecovery = false;

    m_lapPosition = projection.distanceAlongTrack / m_track.getTotalLength();
    m_previousLapPosition = m_lapPosition;

    m_continuousProgress = 0.0f;
    m_bestProgress = 0.0f;

    m_lapCount = 0;
    m_totalCheckpointsPassed = 0;
    m_checkpointsPassedSinceLap = 0;

    // The next checkpoint expected is the first one strictly ahead of the
    // car's current position, in order -- not index 0 unconditionally,
    // since the car may spawn anywhere around the lap (checkpoint indices
    // are fixed at i / kCheckpointCount regardless of spawn position). If
    // the car sits exactly on a checkpoint's position, that checkpoint is
    // treated as the starting reference (not awarded), so the very next one
    // is expected.
    const int currentIndex = static_cast<int>(std::floor(m_lapPosition * static_cast<float>(kCheckpointCount)));
    m_expectedCheckpoint = (currentIndex + 1) % kCheckpointCount;
}

void TrackProgress::advanceCheckpoints(float forwardDelta)
{
    float remaining = forwardDelta;

    float nextCheckpointPosition = static_cast<float>(m_expectedCheckpoint) / static_cast<float>(kCheckpointCount);
    float distanceToNext = nextCheckpointPosition - m_previousLapPosition;
    if (distanceToNext < 0.0f)
    {
        distanceToNext += 1.0f;
    }

    // Bounded by construction: forwardDelta never exceeds
    // kMaxPlausibleLapDeltaPerFrame (well under one full lap), and every
    // iteration after the first consumes a full checkpoint spacing, so this
    // can only run a handful of times per update().
    while (remaining >= distanceToNext)
    {
        remaining -= distanceToNext;

        ++m_totalCheckpointsPassed;
        ++m_checkpointsPassedSinceLap;
        if (m_checkpointsPassedSinceLap == kCheckpointCount)
        {
            // kCheckpointCount checkpoints awarded in order since the last
            // lap (or since reset) means the car has gone all the way
            // around back to its own starting ring position -- one
            // complete ordered lap, regardless of which absolute
            // checkpoint index it started/reset on.
            ++m_lapCount;
            m_checkpointsPassedSinceLap = 0;
        }
        m_expectedCheckpoint = (m_expectedCheckpoint + 1) % kCheckpointCount;

        distanceToNext = 1.0f / static_cast<float>(kCheckpointCount); // every checkpoint is evenly spaced
    }
}

namespace
{

// Signed, shortest-direction, wrap-aware index delta: how far (and in
// which direction) `currentIndex` sits from `previousIndex` around the
// closed loop, taking whichever of the two possible directions is shorter.
// Debug-only (feeds ProjectionDebugInfo); progress/checkpoint math never
// uses this, only the unsigned/seam-aware lapPosition delta in update().
int wrappedIndexDelta(std::size_t previousIndex, std::size_t currentIndex, std::size_t sampleCount)
{
    const long long n = static_cast<long long>(sampleCount);
    long long raw = static_cast<long long>(currentIndex) - static_cast<long long>(previousIndex);
    raw = ((raw % n) + n) % n; // normalize into [0, n)
    if (raw > n / 2)
    {
        raw -= n;
    }
    return static_cast<int>(raw);
}

} // namespace

void TrackProgress::update(const Car& car)
{
    const std::size_t previousIndex = m_previousSegmentIndex;
    const Vector2 previousTangent = tangentAtSegment(previousIndex);

    const TrackProjection projection = projectWithLocalTracking(car.getPosition());
    const float newLapPosition = projection.distanceAlongTrack / m_track.getTotalLength();

    m_lastProjectionDebugInfo.point = projection.point;
    m_lastProjectionDebugInfo.tangent = tangentAtSegment(projection.segmentIndex);
    m_lastProjectionDebugInfo.previousTangent = previousTangent;
    m_lastProjectionDebugInfo.previousIndex = previousIndex;
    m_lastProjectionDebugInfo.currentIndex = m_previousSegmentIndex;
    m_lastProjectionDebugInfo.indexDelta =
        wrappedIndexDelta(previousIndex, m_previousSegmentIndex, m_track.getCenterlineSamples().size());
    m_lastProjectionDebugInfo.distance = projection.distanceFromCenterline;
    m_lastProjectionDebugInfo.usedRecovery = m_lastProjectionUsedRecovery;

    float delta = newLapPosition - m_previousLapPosition;
    if (delta > kSeamWrapThreshold)
    {
        delta -= 1.0f; // backward seam crossing (e.g. 0.02 -> 0.98)
    }
    else if (delta < -kSeamWrapThreshold)
    {
        delta += 1.0f; // forward seam crossing (e.g. 0.98 -> 0.02, becomes +0.04)
    }

    if (std::fabs(delta) <= kMaxPlausibleLapDeltaPerFrame)
    {
        // Ordered checkpoint validation runs on the exact same accepted,
        // seam-corrected delta as continuous progress -- computed from raw
        // position deltas, never from m_bestProgress -- so it independently
        // confirms the car actually traversed the boundaries it's credited
        // for. Only a net-forward delta can award checkpoints/laps.
        if (delta > 0.0f)
        {
            advanceCheckpoints(delta);
        }

        m_continuousProgress += delta;
        if (m_continuousProgress > m_bestProgress)
        {
            m_bestProgress = m_continuousProgress;
        }
    }
    // else: an implausible single-update jump (e.g. teleporting straight
    // across the track interior, or a global recovery re-anchor landing far
    // from where local tracking last had the car; see the class comment) --
    // ignored for both progress and checkpoints. The new position is still
    // accepted as the baseline for future deltas below, so the rejection
    // does not cascade into subsequent updates.

    m_previousLapPosition = newLapPosition;
    m_lapPosition = newLapPosition;
}

} // namespace simulation
