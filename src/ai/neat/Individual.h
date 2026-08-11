#pragma once

#include "raylib.h"

#include "ai/AIController.h"
#include "ai/FitnessEvaluator.h"
#include "ai/neat/Genome.h"
#include "simulation/Car.h"
#include "simulation/Track.h"
#include "simulation/TrackProgress.h"

namespace ai::neat
{

// One evaluated member of a Population: a Genome plus everything needed to
// run it in simulation and score it -- its own Car, AIController (built
// once from the Genome's phenotype), TrackProgress, and FitnessEvaluator.
// Every Individual's Car is a fully independent instance (its own
// position, velocity, alive state, sensors); one Individual crashing,
// finishing, or otherwise changing state never affects any other.
// Individual performs no reproduction, mutation, or crossover of its own
// -- that is entirely Population's responsibility, operating on copies of
// the Genome this class exposes read-only.
//
// Conceptual construction: Genome -> buildPhenotype() -> AIController ->
// Individual. The constructor performs this chain internally, so callers
// only ever need to supply a Genome plus the shared simulation
// environment/spawn pose; no separate NeuralNetwork is stored anywhere --
// AIController already owns the one built from this Individual's Genome.
class Individual
{
public:
    // Builds the AIController's phenotype from genome (propagates whatever
    // ai::neat::buildPhenotype() throws for an invalid/incompatible
    // genome), constructs an independent Car and TrackProgress against
    // track, and resets car/progress/fitness to spawnPosition/spawnHeading
    // -- so a freshly constructed Individual is immediately ready to
    // update().
    Individual(Genome genome, const simulation::Track& track, const simulation::CarParams& carParams,
               Vector2 spawnPosition, float spawnHeading);

    // Advances this individual by deltaTime seconds: the AIController reads
    // the Car's current state and produces a CarInput, the Car updates,
    // then TrackProgress and FitnessEvaluator update from the result -- in
    // that fixed order, matching the existing single-car control loop.
    // Does nothing once isFinished() is true: a finished individual never
    // updates again, including never re-evaluating its controller.
    void update(float deltaTime);

    // Resets this individual back to its spawn pose: Car, TrackProgress,
    // and FitnessEvaluator all reset, exactly as at construction. The
    // Genome and AIController (and therefore its phenotype) are untouched
    // -- this restarts the same individual's evaluation, it does not
    // create a new one.
    void reset();

    bool isFinished() const { return m_fitness.isEvaluationFinished(); }
    float getFitness() const { return m_fitness.getFitness(); }

    const Genome& getGenome() const { return m_genome; }
    const simulation::Car& getCar() const { return m_car; }
    const ai::AIController& getController() const { return m_controller; }
    const simulation::TrackProgress& getProgress() const { return m_progress; }
    const ai::FitnessEvaluator& getFitnessEvaluator() const { return m_fitness; }

private:
    Genome m_genome;
    Vector2 m_spawnPosition;
    float m_spawnHeading;
    simulation::Car m_car;
    ai::AIController m_controller;
    simulation::TrackProgress m_progress;
    ai::FitnessEvaluator m_fitness;
};

} // namespace ai::neat
