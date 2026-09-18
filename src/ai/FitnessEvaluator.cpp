#include "ai/FitnessEvaluator.h"

#include <algorithm>
#include <cmath>

namespace ai
{

namespace
{

constexpr float kProgressPointsPerLap = 1000.0f; // dominant fitness term
constexpr float kCheckpointReward = 10.0f;       // per ordered checkpoint (<=160/lap)
constexpr float kCompletedLapReward = 150.0f;

// Scales progressRate into rawProgressRateReward, rewarding a car for
// covering forward track progress faster -- this is what lets fitness
// distinguish two cars that reach similar progress but at different
// speeds, which raw progress alone cannot. Raised from the original 100
// (see the fitness-speed investigation this was built for): at 100, the
// term contributed only a few points against a baseProgressFitness in the
// thousands, giving speed essentially no selection pressure. 300 gives a
// genuinely meaningful reward while kMaxProgressRateFraction below is what
// actually guarantees it can never outweigh real progress -- unlike the
// old formula (bestProgress/elapsedTime), timeAtBestProgress can legitimately
// be very small (a car reaching real progress within the first second), so
// that explicit cap -- not just this scale relative to some assumed time
// floor -- is what carries the safety invariant now.
constexpr float kProgressRateScale = 300.0f;

// Hard ceiling on progressRateReward, as a fraction of baseProgressFitness
// computed in the SAME call to computeFitnessComponents() -- see
// FitnessEvaluator.h's class comment. Guarantees
// progressRateReward <= baseProgressFitness * kMaxProgressRateFraction
// unconditionally, regardless of how small timeAtBestProgress turns out to
// be (e.g. a car reaching a meaningful amount of progress within the first
// second, before any corner discipline is needed) -- speed can meaningfully
// improve fitness but can never dominate actual track progress. 0.35 was
// chosen so a car with a strong (near-ceiling) speed reward still earns
// clearly less from it than from the progress it takes to unlock that
// reward in the first place.
constexpr float kMaxProgressRateFraction = 0.35f;

// Floor under timeAtBestProgress so progressRate's denominator never blows
// up near zero -- deliberately larger than kSmallTimeEpsilon below (which
// exists for a different division, see its own comment): the very first
// checkpoint is 1/kCheckpointCount of a lap away, physically unreachable in
// a tiny fraction of a second even at this car's max speed, so a floor this
// size never suppresses genuinely-fast-but-real early progress, only the
// most degenerate near-zero-time edge case.
constexpr float kMinTimeToProgress = 0.5f;

constexpr float kLapTimeReferenceSeconds = 20.0f; // "par" time for the lap-speed bonus scale --
                                                    // raised from 15.0 (unreachable in practice; the
                                                    // fastest recorded lap on this track is ~21s) so the
                                                    // bonus stays meaningfully non-saturated across the
                                                    // lap times actually observed in training.
constexpr float kLapSpeedBonusScale = 350.0f;      // raised from 200: lapSpeedBonus can never reward a
                                                    // crash (a lap must be completed to earn it at all),
                                                    // so it is the safest lever for making speed matter
                                                    // once a controller can reliably finish laps.

// Bounds the lap-speed factor so bonus stays in [0, 700] -- comfortably
// under a lap's progress reward (up to ~1310).
constexpr float kMaxLapSpeedFactor = 2.0f;

// Floor under lapTime so its division never blows up near zero. Distinct
// from kMinTimeToProgress above -- this one guards bestLapTime (a
// completed-lap duration), not timeAtBestProgress.
constexpr float kSmallTimeEpsilon = 0.1f;

// Scales averageAbsSteeringDelta into rawSteeringSmoothnessPenalty (see
// FitnessEvaluator.h's class comment) -- 400 puts it in the same order of
// magnitude as kProgressRateScale above, a reasonable starting point for a
// term whose actual safety guarantee comes from kMaxSteeringPenaltyFraction
// below, not from this scale being precisely tuned (same philosophy as
// kProgressRateScale/kMaxProgressRateFraction).
constexpr float kSteeringSmoothnessPenaltyScale = 400.0f;

// Hard ceiling on steeringSmoothnessPenalty, as a fraction of
// baseProgressFitness computed in the SAME call -- guarantees
// steeringSmoothnessPenalty <= baseProgressFitness * kMaxSteeringPenaltyFraction
// unconditionally, so fitness can never be driven negative by this term
// (m_fitness >= baseProgressFitness * (1 - kMaxSteeringPenaltyFraction) >= 0),
// and a car that has made real progress despite imperfect steering still
// clearly beats one that has made none. Slightly smaller than
// kMaxProgressRateFraction (0.35) since this is a PENALTY acting on
// still-random early-generation genomes (which are typically far jerkier
// than anything evolution has had a chance to select against yet) -- a
// gentler cap avoids over-suppressing early exploration before smoother
// steering has had any chance to be discovered and rewarded.
constexpr float kMaxSteeringPenaltyFraction = 0.25f;

constexpr float kMaxEvaluationTime = 60.0f; // seconds, absolute cap

// Seconds without meaningful forward progress before early termination
// (catches circling/parked cars). Short enough to matter, long enough that
// ordinary hairpin cornering never trips it.
constexpr float kNoProgressTimeout = 3.0f;

// Minimum bestProgress increase (laps) counting as "meaningful" for
// kNoProgressTimeout. ~4.9px on the extreme track -- comfortably under one
// frame's worth of near-top-speed movement (~9.8px at maxSpeed 590px/s), so
// stationary jitter never crosses it but real forward motion always does,
// typically within a single frame.
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
    m_timeAtBestProgress = 0.0f;

    m_previousSteeringCommand = 0.0f;
    m_steeringDeltaSum = 0.0f;
    m_steeringSampleCount = 0;

    m_previousLapCount = 0;
    m_lapStartTime = 0.0f;
    m_lastLapTime = 0.0f;
    m_bestLapTime = 0.0f;
    m_hasCompletedLap = false;

    m_baseProgressFitness = 0.0f;
    m_progressRate = 0.0f;
    m_progressRateReward = 0.0f;
    m_lapSpeedBonus = 0.0f;
    m_averageAbsSteeringDelta = 0.0f;
    m_steeringSmoothnessPenalty = 0.0f;

    m_finishReason = EvaluationFinishReason::None;
}

void FitnessEvaluator::computeFitnessComponents(const simulation::TrackProgress& progress)
{
    const float bestProgress = progress.getBestProgress();

    m_baseProgressFitness = bestProgress * kProgressPointsPerLap +
                            static_cast<float>(progress.getTotalCheckpointsPassed()) * kCheckpointReward +
                            static_cast<float>(progress.getLapCount()) * kCompletedLapReward;

    m_progressRate = bestProgress / std::max(m_timeAtBestProgress, kMinTimeToProgress);
    const float rawProgressRateReward = m_progressRate * kProgressRateScale;
    m_progressRateReward = std::min(rawProgressRateReward, m_baseProgressFitness * kMaxProgressRateFraction);

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

    m_averageAbsSteeringDelta =
        m_steeringSampleCount > 0 ? m_steeringDeltaSum / static_cast<float>(m_steeringSampleCount) : 0.0f;
    const float rawSteeringSmoothnessPenalty = m_averageAbsSteeringDelta * kSteeringSmoothnessPenaltyScale;
    m_steeringSmoothnessPenalty =
        std::min(rawSteeringSmoothnessPenalty, m_baseProgressFitness * kMaxSteeringPenaltyFraction);

    m_fitness = m_baseProgressFitness + m_progressRateReward + m_lapSpeedBonus - m_steeringSmoothnessPenalty;
}

void FitnessEvaluator::update(const simulation::Car& car, const simulation::TrackProgress& progress,
                               float steeringCommand, float deltaTime)
{
    if (isEvaluationFinished())
    {
        return;
    }

    m_elapsedTime += deltaTime;

    const float steeringDelta = std::fabs(steeringCommand - m_previousSteeringCommand);
    m_steeringDeltaSum += steeringDelta;
    ++m_steeringSampleCount;
    m_previousSteeringCommand = steeringCommand;

    const float bestProgress = progress.getBestProgress();
    if (bestProgress > m_lastMeaningfulBestProgress + kProgressImprovementEpsilon)
    {
        m_lastMeaningfulBestProgress = bestProgress;
        m_timeSinceProgress = 0.0f;
        m_timeAtBestProgress = m_elapsedTime;
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
