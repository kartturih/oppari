#include "simulation/TrackProgress.h"

#include <cmath>
#include <stdexcept>

namespace simulation
{

namespace
{

// Magnitude a lapPosition delta must exceed, relative to the previous
// update, before it is treated as a 0/1 lap-boundary (seam) crossing rather
// than ordinary local movement.
constexpr float kSeamWrapThreshold = 0.5f;

// Maximum lapPosition delta (after seam correction) accepted as legitimate
// forward/backward movement in a single update(). At the car's maxSpeed
// (260 px/s) and the fixed simulation step (1/60 s), the car can travel at
// most ~4.3px per frame; even at this track's tightest mid-band reference
// radius (~215px, the Y axis), that bounds real per-frame angular movement
// to roughly 4.3 / 215 / (2*pi) =~ 0.003 laps. This threshold is set well
// above that -- generous enough that a single update() spanning several
// checkpoints (kCheckpointCount = 16, so one checkpoint gap is 1/16 =
// 0.0625 laps) is still accepted deterministically -- while staying far
// below both real driving and the kSeamWrapThreshold ambiguity boundary, so
// it reliably rejects an implausible jump such as teleporting straight
// across the track interior.
constexpr float kMaxPlausibleLapDeltaPerFrame = 0.2f;

bool isValidOvalDefinition(const TrackDefinition& def)
{
    return def.simWidth > 0 && def.simHeight > 0 && def.outerRadiusX > 0.0f && def.outerRadiusY > 0.0f &&
           def.innerRadiusX > 0.0f && def.innerRadiusY > 0.0f && def.innerRadiusX < def.outerRadiusX &&
           def.innerRadiusY < def.outerRadiusY;
}

} // namespace

TrackProgress::TrackProgress(const TrackDefinition& definition) : m_definition(definition)
{
    if (!isValidOvalDefinition(m_definition))
    {
        throw std::invalid_argument("TrackProgress: invalid oval track definition");
    }

    // Mid-band radii: the average of the inner and outer radii per axis.
    // Only used to correct for the ellipse's aspect ratio before taking an
    // angle -- see the class comment in TrackProgress.h for the full
    // angle-to-progress mapping.
    m_referenceRadiusX = (m_definition.outerRadiusX + m_definition.innerRadiusX) * 0.5f;
    m_referenceRadiusY = (m_definition.outerRadiusY + m_definition.innerRadiusY) * 0.5f;
}

float TrackProgress::computeLapPosition(Vector2 position) const
{
    const float dx = position.x - m_definition.center.x;
    const float dy = position.y - m_definition.center.y;
    const float nx = dx / m_referenceRadiusX;
    const float ny = dy / m_referenceRadiusY;

    const float rawAngle = std::atan2(ny, nx); // (-pi, pi]
    float lapPosition = -rawAngle / (2.0f * static_cast<float>(PI));
    if (lapPosition < 0.0f)
    {
        lapPosition += 1.0f;
    }
    return lapPosition;
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
    // across the track interior) -- ignored for both progress and
    // checkpoints. The new position is still accepted as the baseline for
    // future deltas below, so the rejection does not cascade into
    // subsequent updates.

    m_previousLapPosition = newLapPosition;
    m_lapPosition = newLapPosition;
}

} // namespace simulation
