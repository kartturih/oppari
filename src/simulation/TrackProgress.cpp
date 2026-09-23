#include "simulation/TrackProgress.h"

#include <cmath>

namespace simulation
{

namespace
{

// Delta magnitude (fraction of a lap) above which it's treated as a 0/1
// seam crossing rather than ordinary movement.
constexpr float kSeamWrapThreshold = 0.5f;

// Max plausible lapPosition delta (post seam-correction) per update() --
// well above real per-frame movement (~9.8px at maxSpeed 590px/s / 60fps)
// but under the seam-wrap boundary, so it reliably rejects teleport-style
// jumps.
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
        m_previousSegmentIndex = local.segmentIndex;
        m_lastProjectionUsedRecovery = false;
        return local;
    }

    // Recovery: local window is clearly wrong -- re-anchor via a full scan.
    // update()'s own plausibility gate still applies to the result.
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

Vector2 TrackProgress::getTrackTangentAhead(float distanceAhead) const
{
    return m_track.getTangentAtDistance(m_lapPosition * m_track.getTotalLength() + distanceAhead);
}

void TrackProgress::reset(const Car& car)
{
    // Full global projection -- no previous segment to search around yet.
    const TrackProjection projection = m_track.projectOntoCenterline(car.getPosition());
    m_previousSegmentIndex = projection.segmentIndex;
    m_lastProjectionUsedRecovery = false;

    const Vector2 tangent = tangentAtSegment(projection.segmentIndex);
    m_trackTangent = tangent;
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

    // Expected checkpoint is the first one strictly ahead of spawn position
    // (spawn's own checkpoint, if exactly on one, is the reference, not awarded).
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

    // Bounded: forwardDelta never exceeds kMaxPlausibleLapDeltaPerFrame, so
    // this only ever runs a handful of iterations.
    while (remaining >= distanceToNext)
    {
        remaining -= distanceToNext;

        ++m_totalCheckpointsPassed;
        ++m_checkpointsPassedSinceLap;
        if (m_checkpointsPassedSinceLap == kCheckpointCount)
        {
            ++m_lapCount;
            m_checkpointsPassedSinceLap = 0;
        }
        m_expectedCheckpoint = (m_expectedCheckpoint + 1) % kCheckpointCount;

        distanceToNext = 1.0f / static_cast<float>(kCheckpointCount); // every checkpoint is evenly spaced
    }
}

namespace
{

// Signed, shortest-direction, wrap-aware index delta. Debug-only.
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

    m_trackTangent = tangentAtSegment(projection.segmentIndex);
    m_lastProjectionDebugInfo.point = projection.point;
    m_lastProjectionDebugInfo.tangent = m_trackTangent;
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
        // Only a net-forward delta awards checkpoints/laps.
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
    // else: implausible jump, ignored for progress/checkpoints -- but the
    // new position still becomes the baseline for future deltas.

    m_previousLapPosition = newLapPosition;
    m_lapPosition = newLapPosition;
}

} // namespace simulation
