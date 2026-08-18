#include "ai/FitnessEvaluator.h"

#include <algorithm>

namespace ai
{

namespace
{

constexpr float kProgressPointsPerLap = 1000.0f; // dominant fitness term
constexpr float kCheckpointReward = 10.0f;       // per ordered checkpoint (<=160/lap)
constexpr float kCompletedLapReward = 150.0f;

// Scales progressRate into progressRateReward; kept small relative to
// kProgressPointsPerLap so a high rate can't outscore substantially more
// actual progress.
constexpr float kProgressRateScale = 100.0f;

constexpr float kLapTimeReferenceSeconds = 15.0f; // "par" time for the lap-speed bonus scale
constexpr float kLapSpeedBonusScale = 200.0f;

// Bounds the lap-speed factor so bonus stays in [0, 400] -- comfortably
// under a lap's progress reward (up to ~1310).
constexpr float kMaxLapSpeedFactor = 2.0f;

// Floor under elapsedTime/lapTime so neither division blows up near zero.
constexpr float kSmallTimeEpsilon = 0.1f;

constexpr float kMaxEvaluationTime = 60.0f; // seconds, absolute cap

// Seconds without meaningful forward progress before early termination
// (catches circling/parked cars). Short enough to matter, long enough that
// ordinary hairpin cornering never trips it.
constexpr float kNoProgressTimeout = 3.0f;

// Minimum bestProgress increase (laps) counting as "meaningful" for
// kNoProgressTimeout. ~4.9px on the extreme track -- just above one frame's
// worth of near-top-speed movement (~4.3px), so stationary jitter never
// crosses it but real forward motion always does within a few frames.
constexpr float kProgressImprovementEpsilon = 0.001f;

// One-shot deadline: bestProgress must reach kMinimumInitialProgress by
// this many seconds, or the evaluation ends -- catches a car crawling just
// fast enough to dodge kNoProgressTimeout while still stuck near spawn.
constexpr float kInitialProgressDeadline = 5.0f; // seconds

// ~195px on the extreme track's ~800px starting straight (well before the
// first corner) -- unreachable only by a car that never got going.
constexpr float kMinimumInitialProgress = 0.04f; // laps

} // namespace

void FitnessEvaluator::reset()
{
    m_fitness = 0.0f;
    m_elapsedTime = 0.0f;

    m_lastMeaningfulBestProgress = 0.0f;
    m_timeSinceProgress = 0.0f;

    m_previousLapCount = 0;
    m_lapStartTime = 0.0f;
    m_lastLapTime = 0.0f;
    m_bestLapTime = 0.0f;
    m_hasCompletedLap = false;

    m_baseProgressFitness = 0.0f;
    m_progressRate = 0.0f;
    m_progressRateReward = 0.0f;
    m_lapSpeedBonus = 0.0f;

    m_finishReason = EvaluationFinishReason::None;
}

void FitnessEvaluator::computeFitnessComponents(const simulation::TrackProgress& progress)
{
    const float bestProgress = progress.getBestProgress();

    m_baseProgressFitness = bestProgress * kProgressPointsPerLap +
                            static_cast<float>(progress.getTotalCheckpointsPassed()) * kCheckpointReward +
                            static_cast<float>(progress.getLapCount()) * kCompletedLapReward;

    m_progressRate = bestProgress / std::max(m_elapsedTime, kSmallTimeEpsilon);
    m_progressRateReward = m_progressRate * kProgressRateScale;

    if (m_hasCompletedLap)
    {
        const float lapSpeedFactor =
            std::clamp(kLapTimeReferenceSeconds / std::max(m_bestLapTime, kSmallTimeEpsilon), 0.0f, kMaxLapSpeedFactor);
        m_lapSpeedBonus = lapSpeedFactor * kLapSpeedBonusScale;
    }
    else
    {
        m_lapSpeedBonus = 0.0f;
    }

    m_fitness = m_baseProgressFitness + m_progressRateReward + m_lapSpeedBonus;
}

void FitnessEvaluator::update(const simulation::Car& car, const simulation::TrackProgress& progress, float deltaTime)
{
    if (isEvaluationFinished())
    {
        return;
    }

    m_elapsedTime += deltaTime;

    const float bestProgress = progress.getBestProgress();
    if (bestProgress > m_lastMeaningfulBestProgress + kProgressImprovementEpsilon)
    {
        m_lastMeaningfulBestProgress = bestProgress;
        m_timeSinceProgress = 0.0f;
    }
    else
    {
        m_timeSinceProgress += deltaTime;
    }

    const int currentLapCount = progress.getLapCount();
    if (currentLapCount > m_previousLapCount)
    {
        const float lapTime = m_elapsedTime - m_lapStartTime;
        m_lastLapTime = lapTime;
        if (!m_hasCompletedLap || lapTime < m_bestLapTime)
        {
            m_bestLapTime = lapTime;
        }
        m_hasCompletedLap = true;
        m_lapStartTime = m_elapsedTime;
        m_previousLapCount = currentLapCount;
    }

    computeFitnessComponents(progress);

    if (!car.isAlive())
    {
        m_finishReason = EvaluationFinishReason::Collision;
    }
    else if (m_elapsedTime >= kMaxEvaluationTime)
    {
        m_finishReason = EvaluationFinishReason::TimeLimit;
    }
    else if (m_elapsedTime >= kInitialProgressDeadline && bestProgress < kMinimumInitialProgress)
    {
        // Checked before kNoProgressTimeout: more specific label, and since
        // bestProgress is monotonic this only ever fires on the one frame
        // the deadline is first reached.
        m_finishReason = EvaluationFinishReason::InsufficientInitialProgress;
    }
    else if (m_timeSinceProgress >= kNoProgressTimeout)
    {
        m_finishReason = EvaluationFinishReason::NoProgress;
    }
}

} // namespace ai
