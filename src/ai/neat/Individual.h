#pragma once

#include "raylib.h"

#include "ai/AIController.h"
#include "ai/DrivingDiagnostics.h"
#include "ai/FitnessEvaluator.h"
#include "ai/neat/Genome.h"
#include "simulation/Car.h"
#include "simulation/Track.h"
#include "simulation/TrackProgress.h"

namespace ai::neat
{

// One evaluated Population member: a Genome plus its own Car, AIController
// (built from the Genome's phenotype), TrackProgress, and FitnessEvaluator.
// Every Individual is fully independent -- one crashing/finishing never
// affects another. Performs no reproduction/mutation/crossover itself.
class Individual
{
public:
    // Builds the phenotype from genome, constructs Car/TrackProgress
    // against track, and resets to spawnPosition/spawnHeading.
    Individual(Genome genome, const simulation::Track& track, const simulation::CarParams& carParams,
               Vector2 spawnPosition, float spawnHeading);

    // Advances by deltaTime: controller -> Car -> TrackProgress -> fitness.
    // No-op once isFinished().
    void update(float deltaTime);

    // Resets Car/TrackProgress/FitnessEvaluator to spawn; Genome/AIController
    // untouched.
    void reset();

    bool isFinished() const { return m_fitness.isEvaluationFinished(); }
    float getFitness() const { return m_fitness.getFitness(); }

    const Genome& getGenome() const { return m_genome; }
    const simulation::Car& getCar() const { return m_car; }
    const simulation::TrackProgress& getProgress() const { return m_progress; }
    const ai::FitnessEvaluator& getFitnessEvaluator() const { return m_fitness; }

    // Driving-quality diagnostics for this evaluation (steering reversals,
    // saturation, lateral acceleration, ...), measured at the real 60 Hz
    // control-step rate. Read-only reporting: nothing in fitness, control or
    // selection ever reads it. averageAbsSteeringDelta is FitnessEvaluator's
    // own value, merged in here.
    ai::DrivingDiagnosticsSummary getDrivingSummary() const
    {
        ai::DrivingDiagnosticsSummary summary = m_diagnostics.summary();
        summary.averageAbsSteeringDelta = m_fitness.getAverageAbsSteeringDelta();
        return summary;
    }

    // Read-only access to the controller's most recent raw network outputs
    // / observation (AIController::getRawSteeringOutput() etc.) -- debug/
    // telemetry only, never used to drive control from outside Individual.
    const ai::AIController& getController() const { return m_controller; }

private:
    Genome m_genome;
    Vector2 m_spawnPosition;
    float m_spawnHeading;
    simulation::Car m_car;
    ai::AIController m_controller;
    simulation::TrackProgress m_progress;
    ai::FitnessEvaluator m_fitness;
    ai::DrivingDiagnostics m_diagnostics; // observation only -- see getDrivingSummary()
};

} // namespace ai::neat
