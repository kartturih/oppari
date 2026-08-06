#include "ai/FitnessEvaluator.h"

namespace ai
{

namespace
{

// Progress is the dominant fitness term: one full lap of best (anti-exploit,
// forward-only) continuous progress is worth this many points.
constexpr float kProgressPointsPerLap = 1000.0f;

// Small supporting reward per ordered checkpoint validated. A full lap is
// simulation::TrackProgress::kCheckpointCount checkpoints, so this
// contributes at most 16 * 10 = 160 points per lap -- clearly secondary to
// the 1000 points progress alone grants for the same lap.
constexpr float kCheckpointReward = 10.0f;

// Meaningful per-lap bonus, but still smaller than the 1000 points a lap's
// worth of progress already grants on its own.
constexpr float kLapCompletionBonus = 150.0f;

// Very small per-second reward for staying alive, so it can never dominate
// over actual track progress even across the full kMaxEvaluationTime.
constexpr float kSurvivalRewardPerSecond = 0.5f;

// Evaluation termination constants.
constexpr float kMaxEvaluationTime = 60.0f; // seconds
constexpr float kNoProgressTimeout = 5.0f;  // seconds without meaningful forward progress

// Minimum increase in TrackProgress::getBestProgress() (in laps) that counts
// as "meaningful" forward progress for the no-progress timeout. Small enough
// to register genuine slow crawling forward, but above float noise from a
// car whose best progress is not actually changing at all.
constexpr float kMinMeaningfulProgressDelta = 0.001f;

} // namespace

void FitnessEvaluator::reset()
{
    m_fitness = 0.0f;
    m_elapsedTime = 0.0f;
    m_lastMeaningfulBestProgress = 0.0f;
    m_timeSinceProgress = 0.0f;
    m_finishReason = EvaluationFinishReason::None;
}

float FitnessEvaluator::computeFitness(const simulation::TrackProgress& progress) const
{
    const float progressReward = progress.getBestProgress() * kProgressPointsPerLap;
    const float checkpointReward = static_cast<float>(progress.getTotalCheckpointsPassed()) * kCheckpointReward;
    const float lapReward = static_cast<float>(progress.getLapCount()) * kLapCompletionBonus;
    const float survivalReward = m_elapsedTime * kSurvivalRewardPerSecond;

    return progressReward + checkpointReward + lapReward + survivalReward;
}

void FitnessEvaluator::update(const simulation::Car& car, const simulation::TrackProgress& progress, float deltaTime)
{
    if (isEvaluationFinished())
    {
        return;
    }

    m_elapsedTime += deltaTime;

    const float bestProgress = progress.getBestProgress();
    if (bestProgress > m_lastMeaningfulBestProgress + kMinMeaningfulProgressDelta)
    {
        m_lastMeaningfulBestProgress = bestProgress;
        m_timeSinceProgress = 0.0f;
    }
    else
    {
        m_timeSinceProgress += deltaTime;
    }

    m_fitness = computeFitness(progress);

    if (!car.isAlive())
    {
        m_finishReason = EvaluationFinishReason::Collision;
    }
    else if (m_elapsedTime >= kMaxEvaluationTime)
    {
        m_finishReason = EvaluationFinishReason::TimeLimit;
    }
    else if (m_timeSinceProgress >= kNoProgressTimeout)
    {
        m_finishReason = EvaluationFinishReason::NoProgress;
    }
}

} // namespace ai
