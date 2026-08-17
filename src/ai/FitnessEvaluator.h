#pragma once

#include "simulation/Car.h"
#include "simulation/TrackProgress.h"

namespace ai
{

// Why an evaluation ended (None while still running). Completing a lap
// never itself ends an evaluation -- the car keeps driving.
enum class EvaluationFinishReason
{
    None,
    Collision,
    TimeLimit,
    NoProgress,
    InsufficientInitialProgress
};

// Computes one scalar fitness value from a Car's aliveness and its
// TrackProgress -- never reads Genome/NeuralNetwork; manual and AI control
// both feed it through the same update().
//
//   fitness = baseProgressFitness + progressRateReward + lapSpeedBonus
//
//   baseProgressFitness = bestProgress * kProgressPointsPerLap
//                        + checkpointsPassed * kCheckpointReward
//                        + lapCount * kCompletedLapReward
//   progressRate        = bestProgress / max(elapsedTime, kSmallTimeEpsilon)
//   progressRateReward   = progressRate * kProgressRateScale
//   lapSpeedBonus        = hasCompletedLap()
//                            ? clamp(kLapTimeReferenceSeconds / bestLapTime, 0, kMaxLapSpeedFactor) * kLapSpeedBonusScale
//                            : 0
//
// Rewards reaching progress QUICKLY rather than raw survival time (an
// earlier version rewarding elapsed time let agents learn to idle safely
// instead of driving) -- progressRateReward can only shrink as elapsedTime
// grows with progress held fixed, so waiting never helps. Nothing here
// reads instantaneous Car speed directly.
//
// Lap timing: detected purely by TrackProgress::getLapCount() increasing;
// lap time = elapsedTime since the last completed lap (or spawn).
//
// Early termination (independent of the formula above -- a terminated
// evaluation's fitness is just whatever it had already accumulated, as if
// kMaxEvaluationTime had hit early). Both are driven only by
// TrackProgress::getBestProgress(), never speed/coordinates:
//   1. No-progress timeout: ends the run if bestProgress hasn't grown by
//      more than kProgressImprovementEpsilon for kNoProgressTimeout seconds
//      (catches circling/parking; backward or revisited progress never
//      resets the timer, since bestProgress is monotonic).
//   2. Initial-progress deadline: a one-shot check at
//      kInitialProgressDeadline seconds requiring bestProgress >=
//      kMinimumInitialProgress (catches a car crawling just enough to dodge
//      rule 1 while still stuck near spawn).
// A car making genuine progress can still run to kMaxEvaluationTime.
class FitnessEvaluator
{
public:
    // Clears fitness, elapsed time, finish reason, and lap-timing state.
    void reset();

    // Advances by deltaTime from the car/progress's current state. No-op
    // once isEvaluationFinished().
    void update(const simulation::Car& car, const simulation::TrackProgress& progress, float deltaTime);

    float getFitness() const { return m_fitness; }
    float getElapsedTime() const { return m_elapsedTime; }
    bool isEvaluationFinished() const { return m_finishReason != EvaluationFinishReason::None; }
    EvaluationFinishReason getFinishReason() const { return m_finishReason; }

    // Fitness component breakdown from the most recent update(); sums to getFitness().
    float getBaseProgressFitness() const { return m_baseProgressFitness; }
    float getProgressRate() const { return m_progressRate; }
    float getProgressRateReward() const { return m_progressRateReward; }
    float getLapSpeedBonus() const { return m_lapSpeedBonus; }

    // getBestLapTime()/getLastLapTime() are 0.0f (sentinel) until hasCompletedLap().
    bool hasCompletedLap() const { return m_hasCompletedLap; }
    float getBestLapTime() const { return m_bestLapTime; }
    float getLastLapTime() const { return m_lastLapTime; }

private:
    void computeFitnessComponents(const simulation::TrackProgress& progress);

    float m_fitness = 0.0f;
    float m_elapsedTime = 0.0f;

    float m_lastMeaningfulBestProgress = 0.0f;
    float m_timeSinceProgress = 0.0f;

    int m_previousLapCount = 0;
    float m_lapStartTime = 0.0f;
    float m_lastLapTime = 0.0f;
    float m_bestLapTime = 0.0f;
    bool m_hasCompletedLap = false;

    float m_baseProgressFitness = 0.0f;
    float m_progressRate = 0.0f;
    float m_progressRateReward = 0.0f;
    float m_lapSpeedBonus = 0.0f;

    EvaluationFinishReason m_finishReason = EvaluationFinishReason::None;
};

} // namespace ai
