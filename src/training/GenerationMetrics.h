#pragma once

#include <cstddef>
#include <vector>

#include "ai/FitnessEvaluator.h"
#include "ai/neat/Genome.h"

namespace training
{

// Pure, file-I/O-free statistics summarizing one completed NEAT generation.
// Free functions/plain data only -- no Population/Individual/Species access,
// no randomness or genome mutation. Population::reproduce() extracts raw
// per-individual/per-species values and passes them to
// buildGenerationMetrics(), keeping the arithmetic unit-testable in isolation.

// Throws std::invalid_argument if values is empty.
float computeMean(const std::vector<float>& values);

// Standard median (mean of the two middle elements for an even count).
// Throws std::invalid_argument if values is empty.
float computeMedian(std::vector<float> values);

float computeMin(const std::vector<float>& values);
float computeMax(const std::vector<float>& values);

// One genome's structural size at the moment it's measured.
struct GenomeComplexity
{
    std::size_t nodeCount = 0;
    std::size_t connectionGeneCount = 0;  // enabled + disabled
    std::size_t enabledConnectionCount = 0;
};

GenomeComplexity computeGenomeComplexity(const ai::neat::Genome& genome);

// One completed generation's metrics row (becomes one CSV line -- see
// TrainingLogger.h). Fields:
//   generation                 -- generation number that just finished (0-based)
//   best/avg/median/worstFitness -- over RAW evaluation fitness, never species-adjusted
//   avgAdjustedFitness         -- mean of species-adjusted fitness (0 if not supplied)
//   speciesCount/largest/smallestSpeciesSize -- this generation's species grouping
//   compatibilityThresholdUsed -- the (runtime-adaptive) compatibility threshold actually used to
//                                  produce this generation's species grouping (Population::reproduce())
//   bestSpeciesHistoricalFitness -- highest all-time species best (Species::getHistoricalBestFitness())
//   stagnantSpeciesExcluded    -- species excluded from offspring allocation this generation
//                                  (not removed -- that only happens once a species hits 0 members)
//   best/avgProgress           -- TrackProgress::getBestProgress() for the best individual / population mean
//   lapsCompletedCount/completionRate -- individuals with hasCompletedLap(), and their fraction
//   bestGenome*/avgGenome*     -- GenomeComplexity of the best individual, and population averages
//   generationDurationSeconds  -- longest FitnessEvaluator::getElapsedTime() this generation (can exceed
//                                  kMaxEvaluationTime by up to one simulation step, since the timeout is
//                                  checked post-step)
//   terminatedCollisionCount/terminatedMaxTimeCount/terminatedNoProgressCount/terminatedSlowStartCount
//                              -- per-EvaluationFinishReason counts (always sum to population size)
struct GenerationMetrics
{
    std::size_t generation = 0;

    float bestFitness = 0.0f;
    float avgFitness = 0.0f;
    float medianFitness = 0.0f;
    float worstFitness = 0.0f;
    float avgAdjustedFitness = 0.0f;

    std::size_t speciesCount = 0;
    std::size_t largestSpeciesSize = 0;
    std::size_t smallestSpeciesSize = 0;
    float compatibilityThresholdUsed = 0.0f;
    float bestSpeciesHistoricalFitness = 0.0f;
    std::size_t stagnantSpeciesExcluded = 0;

    float bestProgress = 0.0f;
    float avgProgress = 0.0f;
    std::size_t lapsCompletedCount = 0;
    float completionRate = 0.0f;

    std::size_t bestGenomeNodeCount = 0;
    std::size_t bestGenomeConnectionGeneCount = 0;
    std::size_t bestGenomeEnabledConnectionCount = 0;
    float avgGenomeNodeCount = 0.0f;
    float avgGenomeConnectionGeneCount = 0.0f;

    float generationDurationSeconds = 0.0f;

    std::size_t terminatedCollisionCount = 0;
    std::size_t terminatedMaxTimeCount = 0;
    std::size_t terminatedNoProgressCount = 0;
    std::size_t terminatedSlowStartCount = 0;
};

// Raw per-individual/per-species inputs buildGenerationMetrics() reduces.
// Every per-individual vector must be the same length as rawFitness (except
// adjustedFitness, which may be empty -- avgAdjustedFitness then reports 0).
struct GenerationMetricsInput
{
    std::size_t generation = 0;

    std::vector<float> rawFitness;
    std::vector<float> adjustedFitness; // may be empty
    std::vector<float> bestProgressValues;
    std::vector<bool> completedLap;
    std::vector<GenomeComplexity> genomeComplexities;

    // Every individual's finish reason; must never contain None (every
    // individual has finished by the time a generation transition happens).
    std::vector<ai::EvaluationFinishReason> finishReasons;

    // Index of the best individual (highest raw fitness, ties -> lowest index).
    std::size_t bestIndividualIndex = 0;

    std::size_t speciesCount = 0;
    std::size_t largestSpeciesSize = 0;
    std::size_t smallestSpeciesSize = 0;
    float compatibilityThresholdUsed = 0.0f;
    float bestSpeciesHistoricalFitness = 0.0f;
    std::size_t stagnantSpeciesExcluded = 0;

    float generationDurationSeconds = 0.0f;
};

// Throws std::invalid_argument if rawFitness is empty, any per-individual
// vector has a mismatched size, bestIndividualIndex is out of range, or
// finishReasons contains EvaluationFinishReason::None.
GenerationMetrics buildGenerationMetrics(const GenerationMetricsInput& input);

} // namespace training
