#include "ai/FitnessEvaluator.h"

#include <algorithm>

namespace ai
{

namespace
{

// Progress is the dominant fitness term: one full lap of best (anti-exploit,
// forward-only) continuous progress is worth this many points. Unchanged
// from Fitness v1.
constexpr float kProgressPointsPerLap = 1000.0f;

// Small supporting reward per ordered checkpoint validated. A full lap is
// simulation::TrackProgress::kCheckpointCount checkpoints, so this
// contributes at most 16 * 10 = 160 points per lap -- clearly secondary to
// the 1000 points progress alone grants for the same lap. Unchanged from
// Fitness v1.
constexpr float kCheckpointReward = 10.0f;

// Meaningful per-lap bonus, but still smaller than the 1000 points a lap's
// worth of progress already grants on its own. Unchanged from Fitness v1
// (there it was named kLapCompletionBonus).
constexpr float kCompletedLapReward = 150.0f;

// Scales progressRate (bestProgress / elapsedTime, in laps/second) into
// progressRateReward. Chosen small relative to kProgressPointsPerLap so
// that even a very high progress rate can never let a car with
// substantially less actual progress outscore one with substantially more
// -- see the worked example in FitnessEvaluator.h's class comment. This is
// the term that replaces Fitness v1's flat, always-positive survival
// reward: unlike that term, it can only ever reward reaching a *given*
// amount of progress sooner, never reward time passing on its own.
constexpr float kProgressRateScale = 100.0f;

// "Par" lap time, in seconds, that a completed lap is compared against to
// compute the lap-speed bonus (see kMaxLapSpeedFactor below). Not a hard
// limit on lap time -- purely a scale for the bonus formula.
constexpr float kLapTimeReferenceSeconds = 15.0f;

// Scales the (bounded) lap-speed factor into the final lap-speed bonus.
constexpr float kLapSpeedBonusScale = 200.0f;

// Upper bound on the lap-speed factor (kLapTimeReferenceSeconds /
// bestLapTime), so an extremely fast (or degenerate near-zero-time) lap
// cannot make the lap-speed bonus explode without limit. With this clamp,
// the lap-speed bonus is always in [0, kMaxLapSpeedFactor * kLapSpeedBonusScale]
// = [0, 400], still comfortably smaller than a single lap's worth of
// progress reward (1000 + up to 160 checkpoint + 150 lap = up to ~1310).
constexpr float kMaxLapSpeedFactor = 2.0f;

// Floor under elapsedTime (for progressRate) and under lap time (for the
// lap-speed factor) so neither division can blow up near zero -- both
// quantities stay finite and well-defined from the very first update().
constexpr float kSmallTimeEpsilon = 0.1f;

// Evaluation termination constants. Unchanged from Fitness v1 -- Stage 15A
// changes the objective score, not evaluation termination.
constexpr float kMaxEvaluationTime = 60.0f; // seconds
constexpr float kNoProgressTimeout = 5.0f;  // seconds without meaningful forward progress

// Minimum increase in TrackProgress::getBestProgress() (in laps) that counts
// as "meaningful" forward progress for the no-progress timeout. Small enough
// to register genuine slow crawling forward, but above float noise from a
// car whose best progress is not actually changing at all. Unchanged from
// Fitness v1.
constexpr float kMinMeaningfulProgressDelta = 0.001f;

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

    // Never uses Car's instantaneous speed -- only TrackProgress's own
    // best-progress measurement and this evaluator's own elapsed-time
    // accumulation, so it cannot be gamed by momentarily spiking velocity.
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
    if (bestProgress > m_lastMeaningfulBestProgress + kMinMeaningfulProgressDelta)
    {
        m_lastMeaningfulBestProgress = bestProgress;
        m_timeSinceProgress = 0.0f;
    }
    else
    {
        m_timeSinceProgress += deltaTime;
    }

    // Lap timing: detected purely by watching TrackProgress::getLapCount()
    // increase -- never derived from bestProgress or from any timer inside
    // TrackProgress itself (it exposes none). See the class comment for how
    // a same-update multi-lap jump (not expected from normal physics) is
    // handled.
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
    else if (m_timeSinceProgress >= kNoProgressTimeout)
    {
        m_finishReason = EvaluationFinishReason::NoProgress;
    }
}

} // namespace ai
