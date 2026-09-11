#pragma once

// Internal declarations shared among the verification .cpp files and
// VerificationRunner.cpp. Not part of the public interface -- application
// code outside src/verification/ should only ever include
// VerificationRunner.h and call verification::runAll().

#include <cstdint>

#include "raylib.h"

#include "ai/neat/Genome.h"
#include "ai/neat/PopulationConfig.h"
#include "simulation/Track.h"

namespace verification
{

void verifyTrack(const simulation::Track& track);
void verifyCar(const simulation::Track& track);
void verifySensors(const simulation::Track& track);
void verifyObservation(const simulation::Track& track);
void verifyVehiclePhysics(const simulation::Track& track);
void verifyAIController(const simulation::Track& track);

void verifyNeuralNetwork();
void verifyPhenotypeBuilder();

void verifyNeatGenes();
void verifyGenome();
void verifyInnovationTracker();

void verifyGenomeMutator();
void verifyAddConnectionMutation();
void verifyAddNodeMutation();

void verifyGenomeCrossover();
void verifyGenomeCrossoverCycleSafety();

void verifyCompatibilityDistance();
void verifySpeciation();

void verifyTrackProgress(const simulation::Track& track);
void verifyImageBasedTrackSystem(const simulation::Track& track);

void verifyFitnessEvaluator(const simulation::Track& track);
void verifyEarlyTermination(const simulation::Track& track);

void verifyPopulation(const simulation::Track& track);
void verifySpeciesAwareReproduction(const simulation::Track& track);
void verifyPersistentSpeciesAndStagnation(const simulation::Track& track);
void verifyAdaptiveCompatibilityThreshold(const simulation::Track& track);

void verifyTrainingMetrics();
void verifyTrainingLogger();
void verifyGenerationMetricsPopulationIntegration(const simulation::Track& track);
void verifyTrainingSpeedDeterminism(const simulation::Track& track);

void verifyHairpinTelemetry(const simulation::Track& track);

// Cross-file test helpers: each is defined alongside its owning verify*()
// function (see the .cpp comment where it's defined) but also consumed by
// a verify*() function living in a different translation unit.
namespace track_progress_verify
{
// Defined in TrackVerification.cpp (alongside verifyTrackProgress); also
// used by FitnessVerification.cpp.
Vector2 positionAtLapPosition(const simulation::Track& track, float lapPos);
} // namespace track_progress_verify

namespace population_verify
{
// Defined in PopulationVerification.cpp (alongside verifyPopulation); also
// used by TrainingVerification.cpp.
ai::neat::PopulationConfig makeTestPopulationConfig(std::size_t populationSize, std::uint32_t seed);
ai::neat::Genome makeCrashGenome();
} // namespace population_verify

namespace speciation_verify
{
// Defined in NeatSpeciationVerification.cpp (alongside verifySpeciation);
// also used by PopulationVerification.cpp.
ai::neat::Genome makeSimpleGenome(float weight);
} // namespace speciation_verify

} // namespace verification
