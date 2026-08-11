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

float TrackProgress::computeLapPosition(Vector2 position) const
{
    const TrackProjection projection = m_track.projectOntoCenterline(position);
    return projection.distanceAlongTrack / m_track.getTotalLength();
}

void TrackProgress::reset(const Car& car)
{
    m_lapPosition = computeLapPosition(car.getPosition());
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

void TrackProgress::update(const Car& car)
{
    const float newLapPosition = computeLapPosition(car.getPosition());

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
    // across the track interior, or -- on a future self-intersecting track
    // -- a local projection ambiguity jump; see the class comment) --
    // ignored for both progress and checkpoints. The new position is still
    // accepted as the baseline for future deltas below, so the rejection
    // does not cascade into subsequent updates.

    m_previousLapPosition = newLapPosition;
    m_lapPosition = newLapPosition;
}

} // namespace simulation
