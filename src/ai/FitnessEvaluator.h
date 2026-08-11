#pragma once

#include "simulation/Car.h"
#include "simulation/TrackProgress.h"

namespace ai
{

// Why an evaluation ended. None while it is still running.
enum class EvaluationFinishReason
{
    None,
    Collision,
    TimeLimit,
    NoProgress
};

// Computes one scalar fitness value for one car from its TrackProgress and
// elapsed evaluation time. FitnessEvaluator depends only on
// simulation::Car (for isAlive()) and simulation::TrackProgress (for
// progress/lap/checkpoint state) -- it never reads Genome or NeuralNetwork,
// and manual and AI control modes both feed it through the exact same
// update() call.
//
// Fitness v2 (Stage 15A) -- see FitnessEvaluator.cpp for the exact constants
// -----------------------------------------------------------------------
// Fitness v1 (Stage 8-14) rewarded raw survival time (elapsedTime *
// kSurvivalRewardPerSecond). That is a real optimization target: standing
// still (or crawling as slowly as possible while still avoiding collision)
// reduces collision risk and directly increases fitness, with nothing in
// the objective pushing back against it. By generation 1, many agents had
// already discovered exactly that. Fitness v2 removes every term that
// rewards time merely passing, and instead rewards reaching progress
// *quickly*:
//
//   fitness = baseProgressFitness + progressRateReward + lapSpeedBonus
//
//   baseProgressFitness =
//       TrackProgress::getBestProgress()            * kProgressPointsPerLap
//     + TrackProgress::getTotalCheckpointsPassed()  * kCheckpointReward
//     + TrackProgress::getLapCount()                * kCompletedLapReward
//
//   progressRate       = bestProgress / max(elapsedTime, kSmallTimeEpsilon)
//   progressRateReward = progressRate * kProgressRateScale
//
//   lapSpeedBonus = hasCompletedLap()
//                       ? clamp(kLapTimeReferenceSeconds / max(bestLapTime, kSmallTimeEpsilon), 0, kMaxLapSpeedFactor)
//                         * kLapSpeedBonusScale
//                       : 0
//
// baseProgressFitness is unchanged from v1 and remains the dominant term
// (see the constants in the .cpp for the exact scale comparison).
// progressRateReward is the key new term: for a *fixed* amount of progress,
// it is strictly larger the less time that progress took, so two cars that
// reach the same point on the track no longer score equally -- the faster
// one wins. It can never reward waiting: with bestProgress held fixed,
// progressRate (and therefore progressRateReward) can only shrink or stay
// flat as elapsedTime grows, never grow. lapSpeedBonus is a separate,
// bounded bonus specifically for *completed* laps (see "Lap timing" below)
// -- faster completed laps score higher, a slower completed lap can never
// reduce the recorded best lap time, and there is no lap-speed bonus at all
// before any lap has been completed. Nothing in this formula rewards
// instantaneous Car speed directly (only Car::isAlive() is ever read) or
// survival time on its own.
//
// Lap timing
// ----------
// FitnessEvaluator has no notion of a clock beyond deltaTime accumulation;
// it detects a completed lap purely by watching
// TrackProgress::getLapCount() increase between update() calls (never by
// reading any other lap/timer state from TrackProgress, which exposes
// none). On an increase, the lap time is elapsedTime since the start of
// that lap attempt (m_lapStartTime, itself reset to the current elapsedTime
// right after every completed lap, and implicitly 0 at construction/reset
// -- i.e. the first lap's start is spawn). If getLapCount() somehow jumps by
// more than 1 in a single update() (not expected from normal physics --
// TrackProgress's own plausibility gate bounds how far a single update can
// advance), the whole jump is treated as one lap-timing event spanning the
// entire elapsed interval since the last completed lap: deterministic, and
// the only sensible behavior with no finer-grained timing information to
// split across the jump.
class FitnessEvaluator
{
public:
    // Clears all evaluation state: fitness and every fitness component back
    // to 0, elapsed time and the no-progress timer to 0, the finish reason
    // to None, and all lap-timing state (previous lap count, lap start
    // time, last/best lap time, hasCompletedLap()) back to "no laps
    // completed yet".
    void reset();

    // Advances the evaluation by deltaTime seconds using the car's current
    // alive state and the progress tracker's current values. Does nothing
    // once the evaluation has already finished (isEvaluationFinished() ==
    // true) -- fitness (and every component below), elapsed time, all lap
    // timing state, and the finish reason all stay frozen at their final
    // values.
    void update(const simulation::Car& car, const simulation::TrackProgress& progress, float deltaTime);

    float getFitness() const { return m_fitness; }
    float getElapsedTime() const { return m_elapsedTime; }
    bool isEvaluationFinished() const { return m_finishReason != EvaluationFinishReason::None; }
    EvaluationFinishReason getFinishReason() const { return m_finishReason; }

    // Fitness component breakdown from the most recent update() (or reset()
    // to 0) -- read-only, purely for HUD/debug visibility into why the
    // objective favors a given behavior. getFitness() always equals their
    // sum (within float rounding).
    float getBaseProgressFitness() const { return m_baseProgressFitness; }
    float getProgressRate() const { return m_progressRate; }
    float getProgressRateReward() const { return m_progressRateReward; }
    float getLapSpeedBonus() const { return m_lapSpeedBonus; }

    // Lap timing, read-only. getBestLapTime()/getLastLapTime() are only
    // meaningful once hasCompletedLap() is true -- before that, both read
    // as 0.0f (the documented "no completed lap yet" sentinel), which is
    // never itself a valid lap time (a lap always takes some positive
    // amount of simulated time).
    bool hasCompletedLap() const { return m_hasCompletedLap; }
    float getBestLapTime() const { return m_bestLapTime; }
    float getLastLapTime() const { return m_lastLapTime; }

private:
    void computeFitnessComponents(const simulation::TrackProgress& progress);

    float m_fitness = 0.0f;
    float m_elapsedTime = 0.0f;

    float m_lastMeaningfulBestProgress = 0.0f; // best progress last time meaningful forward progress was seen
    float m_timeSinceProgress = 0.0f;          // seconds since best progress last advanced meaningfully

    // Lap timing state (see the class comment's "Lap timing" section).
    int m_previousLapCount = 0;     // TrackProgress::getLapCount() as of the last update()
    float m_lapStartTime = 0.0f;    // m_elapsedTime at the start of the lap currently in progress
    float m_lastLapTime = 0.0f;     // most recently completed lap's time; 0 (sentinel) if none yet
    float m_bestLapTime = 0.0f;     // fastest completed lap's time; 0 (sentinel) if none yet
    bool m_hasCompletedLap = false; // true once at least one lap has been completed

    // Fitness component breakdown, recomputed every update() (see the class
    // comment's fitness formula).
    float m_baseProgressFitness = 0.0f;
    float m_progressRate = 0.0f;
    float m_progressRateReward = 0.0f;
    float m_lapSpeedBonus = 0.0f;

    EvaluationFinishReason m_finishReason = EvaluationFinishReason::None;
};

} // namespace ai
