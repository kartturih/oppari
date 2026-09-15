#include "verification/VerificationRunner.h"

#include "verification/Verifications.h"

namespace verification
{

void runAll(const simulation::Track& track)
{
    verifyTrack(track);
    verifyCar(track);
    verifySensors(track);
    verifyObservation(track);
    verifyVehiclePhysics(track);
    verifySteeringRateLimit(track);
    verifyNeuralNetwork();
    verifyNeatGenes();
    verifyGenome();
    verifyPhenotypeBuilder();
    verifyGenomeMutator();
    verifyInnovationTracker();
    verifyAddConnectionMutation();
    verifyAddNodeMutation();
    verifyGenomeCrossover();
    verifyGenomeCrossoverCycleSafety();
    verifyCompatibilityDistance();
    verifySpeciation();
    verifyAIController(track);
    verifyTrackProgress(track);
    verifyImageBasedTrackSystem(track);
    verifyFitnessEvaluator(track);
    verifyEarlyTermination(track);
    verifyPopulation(track);
    verifySpeciesAwareReproduction(track);
    verifyPersistentSpeciesAndStagnation(track);
    verifyAdaptiveCompatibilityThreshold(track);
    verifyTrainingMetrics();
    verifyTrainingLogger();
    verifyGenerationMetricsPopulationIntegration(track);
    verifyTrainingSpeedDeterminism(track);
    verifyHairpinTelemetry(track);
    verifyChampionTelemetry(track);
}

} // namespace verification
