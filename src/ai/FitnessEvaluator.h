#pragma once

#include "simulation/Car.h"
#include "simulation/TrackProgress.h"

namespace ai
{

// An evaluation's normal, successful end: the car has completed this many
// laps. Fitness is frozen the moment the lap count reaches it.
inline constexpr int kTargetLapCount = 3;

// Absolute simulated-time failsafe (seconds) so an evaluation can never run
// forever. NOT the intended way for a successful run to end -- that is
// completing kTargetLapCount laps, which a car of any useful speed does
// long before this (the current champion needs ~55 s).
inline constexpr float kSafetyTimeoutSeconds = 180.0f;

// Why an evaluation ended (None while still running).
//   Collision                   -- the car died
//   CompletedLaps               -- the car completed kTargetLapCount laps (success)
//   SafetyTimeout               -- kSafetyTimeoutSeconds elapsed first (failsafe only)
//   NoProgress                  -- no meaningful forward progress for kNoProgressTimeout
//   InsufficientInitialProgress -- the one-shot slow-start deadline was missed
enum class EvaluationFinishReason
{
    None,
    Collision,
    CompletedLaps,
    SafetyTimeout,
    NoProgress,
    InsufficientInitialProgress
};

// Computes one scalar fitness value from a Car's aliveness and its
// TrackProgress -- never reads Genome/NeuralNetwork; manual and AI control
// both feed it through the same update().
//
//   fitness = baseProgressFitness + progressRateReward + lapSpeedBonus
//             - steeringSmoothnessPenalty
//
//   baseProgressFitness = bestProgress * kProgressPointsPerLap
//                        + checkpointsPassed * kCheckpointReward
//                        + lapCount * kCompletedLapReward
//   progressRate         = bestProgress / max(timeAtBestProgress, kMinTimeToProgress)
//   rawProgressRateReward = progressRate * kProgressRateScale
//   progressRateReward   = min(rawProgressRateReward, baseProgressFitness * kMaxProgressRateFraction)
//   lapSpeedBonus        = hasCompletedLap()
//                            ? clamp(kLapTimeReferenceSeconds / bestLapTime, 0, kMaxLapSpeedFactor) * kLapSpeedBonusScale
//                            : 0
//   averageAbsSteeringDelta = (sum of |steeringCommand[t] - steeringCommand[t-1]|) / sampleCount
//   rawSteeringSmoothnessPenalty = averageAbsSteeringDelta * kSteeringSmoothnessPenaltyScale
//   steeringSmoothnessPenalty    = min(rawSteeringSmoothnessPenalty, baseProgressFitness * kMaxSteeringPenaltyFraction)
//
// steeringSmoothnessPenalty targets steering-command CHANGE, deliberately
// never steering MAGNITUDE: it is the average of |this frame's steering
// command - last frame's|, not of the command's absolute value, so a car
// committed to a single sustained hard lock through a long corner (delta ~0
// every frame after the initial turn-in) is barely charged at all, while a
// car whose command keeps flipping sign every few frames (bang-bang/
// oscillating steering) accumulates a real, ongoing cost -- exactly the
// distinction between legitimate cornering and chattering commands. It is
// an AVERAGE (sum/sampleCount), not a running total, specifically so a
// longer-lived (more successful) individual is never penalized more than a
// short-lived one purely for having accumulated more frames -- only the
// per-frame RATE of steering change matters, independent of episode length.
// Capped at a fraction of baseProgressFitness for the same reason
// progressRateReward is (see its own comment): the exact scale needed to
// meaningfully discourage oscillation is hard to predict a priori, so the
// explicit cap -- not a carefully-tuned raw scale -- is what actually
// guarantees this can never outweigh, or drive negative, real track
// progress. steeringCommand is CarInput.steering as actually sent to Car
// (post-mapping) -- see update()'s steeringCommand parameter.
//
// timeAtBestProgress is the elapsedTime at which bestProgress last made
// MEANINGFUL progress (same kProgressImprovementEpsilon-gated detection
// kNoProgressTimeout already uses -- see update()), not the evaluation's
// current/final elapsedTime. This is what lets progressRate measure "how
// quickly was the progress actually reached", independent of however much
// longer (or shorter) the evaluation happens to run afterward -- two cars
// that reach identical bestProgress at different real speeds now score
// differently even if both are later scored/terminated at the same final
// elapsedTime, which a plain bestProgress/elapsedTime ratio cannot
// distinguish (it dilutes toward zero the longer a car keeps running after
// its last meaningful gain, even while just continuing to drive normally).
//
// kMaxProgressRateFraction caps progressRateReward at a fixed fraction of
// baseProgressFitness EXPLICITLY, rather than relying only on the
// kProgressRateScale/kMinTimeToProgress ratio to stay small -- timeAtBestProgress
// can legitimately be very small (a car reaching real progress within the
// first second, before any corner discipline is needed), so an implicit-only
// ceiling would no longer reliably hold once kProgressRateScale is large
// enough to matter. This cap is what actually guarantees speed can never
// outweigh real progress, independent of how progress/time happen to combine
// in any single case -- see FitnessEvaluator.cpp's constant comments.
//
// Rewards reaching progress QUICKLY rather than raw survival time (an
// earlier version rewarding elapsed time let agents learn to idle safely
// instead of driving) -- progressRateReward can only shrink or stay flat as
// time passes with progress held fixed (timeAtBestProgress freezes, it never
// grows on its own), so waiting never helps. Nothing here reads
// instantaneous Car speed directly.
//
// Lap timing: detected purely by TrackProgress::getLapCount() increasing;
// lap time = elapsedTime since the last completed lap (or spawn). This is
// the one speed signal immune to "reach good progress fast, then crash":
// completing a lap requires surviving it, so lapSpeedBonus can never reward
// a crash the way an unbounded progressRateReward theoretically could.
//
// Termination (a terminated evaluation's fitness is just whatever it had
// already accumulated). Checked in this order each update(), after the
// progress/lap/fitness bookkeeping for that same step (so the final step's
// progress, lap time and fitness are always included):
//   1. Collision: the car is no longer alive.
//   2. Completed laps: TrackProgress::getLapCount() >= kTargetLapCount --
//      the normal successful end. There is no time limit on a successful run
//      other than the failsafe below.
//   3. Safety timeout: kSafetyTimeoutSeconds of simulated time (failsafe only).
//   4. Initial-progress deadline: a one-shot check at kInitialProgressDeadline
//      seconds requiring bestProgress >= kMinimumInitialProgress (catches a
//      car crawling just enough to dodge rule 5 while still stuck near spawn).
//   5. No-progress timeout: ends the run if bestProgress hasn't grown by more
//      than kProgressImprovementEpsilon for kNoProgressTimeout seconds
//      (catches circling/parking; backward or revisited progress never resets
//      the timer, since bestProgress is monotonic).
// Rules 4-5 are driven only by TrackProgress::getBestProgress(), never
// speed/coordinates.
class FitnessEvaluator
{
public:
    // Clears fitness, elapsed time, finish reason, lap-timing state, and
    // steering-smoothness bookkeeping.
    void reset();

    // Advances by deltaTime from the car/progress's current state.
    // steeringCommand is CarInput.steering as actually sent to Car this
    // frame (post-mapping, i.e. AIController::update()'s returned
    // CarInput.steering) -- used only for the steering-smoothness penalty's
    // frame-to-frame delta (see the class comment); never fed into
    // progress/lap tracking. No-op once isEvaluationFinished().
    void update(const simulation::Car& car, const simulation::TrackProgress& progress, float steeringCommand,
                float deltaTime);

    float getFitness() const { return m_fitness; }
    float getElapsedTime() const { return m_elapsedTime; }
    bool isEvaluationFinished() const { return m_finishReason != EvaluationFinishReason::None; }
    EvaluationFinishReason getFinishReason() const { return m_finishReason; }

    // Fitness component breakdown from the most recent update(); sums to getFitness().
    float getBaseProgressFitness() const { return m_baseProgressFitness; }
    float getProgressRate() const { return m_progressRate; }
    float getProgressRateReward() const { return m_progressRateReward; }
    float getLapSpeedBonus() const { return m_lapSpeedBonus; }
    float getSteeringSmoothnessPenalty() const { return m_steeringSmoothnessPenalty; }

    // Pre-scale, pre-cap average of |steeringCommand[t] - steeringCommand[t-1]|
    // (see the class comment) -- same role as getProgressRate() plays for
    // progressRateReward: exposed so verification/telemetry can inspect the
    // raw measurement independent of kSteeringSmoothnessPenaltyScale/
    // kMaxSteeringPenaltyFraction.
    float getAverageAbsSteeringDelta() const { return m_averageAbsSteeringDelta; }

    // elapsedTime at which bestProgress last made MEANINGFUL progress (see
    // the class comment) -- 0.0f (sentinel) until the first such increase.
    // Exposed for telemetry/verification; progressRate's own denominator
    // flooring (kMinTimeToProgress) is applied internally, not here.
    float getTimeAtBestProgress() const { return m_timeAtBestProgress; }

    // getBestLapTime()/getLastLapTime() are 0.0f (sentinel) until hasCompletedLap().
    bool hasCompletedLap() const { return m_hasCompletedLap; }
    float getBestLapTime() const { return m_bestLapTime; }
    float getLastLapTime() const { return m_lastLapTime; }

    // Elapsed time since the current lap began (or since spawn, if no lap
    // has completed yet) -- a read-only convenience over existing state
    // (m_elapsedTime, m_lapStartTime), never itself used by the fitness
    // formula. Exposed for telemetry only (see telemetry::ChampionTelemetry).
    float getCurrentLapElapsedTime() const { return m_elapsedTime - m_lapStartTime; }

private:
    void computeFitnessComponents(const simulation::TrackProgress& progress);

    float m_fitness = 0.0f;
    float m_elapsedTime = 0.0f;

    float m_lastMeaningfulBestProgress = 0.0f;
    float m_timeSinceProgress = 0.0f;
    float m_timeAtBestProgress = 0.0f;

    // Steering-smoothness bookkeeping (see the class comment): running
    // sum-of-|delta| and sample count, from which computeFitnessComponents()
    // derives the average each call.
    float m_previousSteeringCommand = 0.0f;
    float m_steeringDeltaSum = 0.0f;
    int m_steeringSampleCount = 0;

    int m_previousLapCount = 0;
    float m_lapStartTime = 0.0f;
    float m_lastLapTime = 0.0f;
    float m_bestLapTime = 0.0f;
    bool m_hasCompletedLap = false;

    float m_baseProgressFitness = 0.0f;
    float m_progressRate = 0.0f;
    float m_progressRateReward = 0.0f;
    float m_lapSpeedBonus = 0.0f;
    float m_averageAbsSteeringDelta = 0.0f;
    float m_steeringSmoothnessPenalty = 0.0f;

    EvaluationFinishReason m_finishReason = EvaluationFinishReason::None;
};

} // namespace ai
