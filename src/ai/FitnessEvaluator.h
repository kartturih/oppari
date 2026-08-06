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
// survival time. FitnessEvaluator depends only on simulation::Car (for
// isAlive()) and simulation::TrackProgress (for progress/lap/checkpoint
// state) -- it never reads Genome or NeuralNetwork, and manual and AI
// control modes both feed it through the exact same update() call.
//
// Fitness formula (see FitnessEvaluator.cpp for the exact constants):
//
//   fitness = progressReward + checkpointReward + lapReward + survivalReward
//
//   progressReward   = TrackProgress::getBestProgress() * kProgressPointsPerLap
//   checkpointReward = TrackProgress::getTotalCheckpointsPassed() * kCheckpointReward
//   lapReward        = TrackProgress::getLapCount() * kLapCompletionBonus
//   survivalReward   = elapsedTime * kSurvivalRewardPerSecond
//
// Progress is the dominant term by construction: a full lap's worth of
// checkpoint reward or of survival reward is small next to one lap of
// progress reward, and the flat per-lap bonus is meaningful but still
// smaller than the progress a full lap itself already grants.
class FitnessEvaluator
{
public:
    // Clears all evaluation state: fitness back to 0, elapsed time to 0, the
    // no-progress timer to 0, and the finish reason to None.
    void reset();

    // Advances the evaluation by deltaTime seconds using the car's current
    // alive state and the progress tracker's current values. Does nothing
    // once the evaluation has already finished (isEvaluationFinished() ==
    // true) -- fitness, elapsed time and finish reason all stay frozen at
    // their final values.
    void update(const simulation::Car& car, const simulation::TrackProgress& progress, float deltaTime);

    float getFitness() const { return m_fitness; }
    float getElapsedTime() const { return m_elapsedTime; }
    bool isEvaluationFinished() const { return m_finishReason != EvaluationFinishReason::None; }
    EvaluationFinishReason getFinishReason() const { return m_finishReason; }

private:
    float computeFitness(const simulation::TrackProgress& progress) const;

    float m_fitness = 0.0f;
    float m_elapsedTime = 0.0f;

    float m_lastMeaningfulBestProgress = 0.0f; // best progress last time meaningful forward progress was seen
    float m_timeSinceProgress = 0.0f;          // seconds since best progress last advanced meaningfully

    EvaluationFinishReason m_finishReason = EvaluationFinishReason::None;
};

} // namespace ai
