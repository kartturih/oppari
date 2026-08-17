#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "raylib.h"

#include "ai/neat/CompatibilityConfig.h"
#include "ai/neat/CrossoverConfig.h"
#include "ai/neat/Genome.h"
#include "ai/neat/GenomeCrossover.h"
#include "ai/neat/GenomeMutator.h"
#include "ai/neat/Individual.h"
#include "ai/neat/InnovationTracker.h"
#include "ai/neat/MutationConfig.h"
#include "ai/neat/PopulationConfig.h"
#include "ai/neat/SpeciationConfig.h"
#include "ai/neat/Speciator.h"
#include "simulation/Car.h"
#include "simulation/Track.h"
#include "training/GenerationMetrics.h"

namespace ai::neat
{

// Tournament tie-break: true if candidate should replace best -- strictly
// higher fitness, or equal fitness with a strictly lower index.
bool isBetterTournamentCandidate(float candidateFitness, std::size_t candidateIndex, float bestFitness,
                                  std::size_t bestIndex);

// Proportionally allocates remainingSlots across species by effective
// adjusted-fitness share (floor + largest-fractional-remainder, ties by
// lower SpeciesId). Falls back to an even split (by count) if every
// species' effective fitness sum is 0. Result always sums to remainingSlots.
std::vector<std::size_t> allocateSpeciesOffspring(const std::vector<SpeciesId>& speciesIds,
                                                    const std::vector<float>& effectiveFitnessSums,
                                                    std::size_t remainingSlots);

// A member's contribution to its species' effective adjusted-fitness sum
// (used only for offspring allocation): adjustedFitness clamped to >= 0, so
// negative raw fitness can't shrink another species' share.
float effectiveFitnessContribution(float adjustedFitness);

// Orchestrates one NEAT run: an initial population from one base Genome,
// advanced generation by generation via tournament selection, elitism,
// crossover, and mutation. Every Individual is fully independent.
//
// Owns, for the run's lifetime: one InnovationTracker shared by every
// individual/generation; one GenomeMutator and one GenomeCrossover, each
// seeded once and reused throughout; one orchestration RNG for tournament
// sampling only. All three seed deterministically from
// PopulationConfig::randomSeed -- no random_device/rand()/time-based
// seeding -- so a given seed + config + update() sequence always reproduces
// the same evolutionary run.
//
// Species (via Speciator) are persistent across generations and drive
// reproduction: fitness sharing, per-species offspring allocation, and
// species-local parent selection all use the same Species vector computed
// at the start of reproduce(). A species stagnant for
// speciesStagnationLimit generations is excluded from normal offspring
// allocation (with a global-elite exception and an all-stagnant fallback --
// see reproduce()). No adaptive compatibility threshold or interspecies mating.
class Population
{
public:
    // One species' reproduction outcome from the most recent reproduce()
    // call, for HUD/debug/testing. Empty until the first generation
    // transition. adjustedFitnessSum may be negative; allocatedOffspring
    // comes from effective (clamped) fitness among eligible species only.
    // age/historicalBestFitness/generationsSinceImprovement/stagnant mirror
    // the underlying Species. reproductionEligible is false only for a
    // stagnant species excluded this generation (at most one exception per
    // generation via the all-stagnant fallback).
    struct SpeciesReproductionStats
    {
        SpeciesId speciesId = 0;
        std::size_t memberCount = 0;
        float adjustedFitnessSum = 0.0f;
        std::size_t allocatedOffspring = 0;

        std::size_t age = 0;
        float historicalBestFitness = 0.0f;
        std::size_t generationsSinceImprovement = 0;
        bool stagnant = false;
        bool reproductionEligible = true;
    };

    // Builds generation 0: individual 0 is an unmutated copy of baseGenome;
    // the rest get mutateWeights() only (same topology). InnovationTracker
    // is seeded one past baseGenome's highest node ID/innovation number.
    // Throws std::invalid_argument on an invalid populationConfig;
    // propagates buildPhenotype()'s exceptions for an invalid baseGenome.
    Population(const Genome& baseGenome, const simulation::Track& track, const simulation::CarParams& carParams,
               Vector2 spawnPosition, float spawnHeading, const PopulationConfig& populationConfig,
               const MutationConfig& mutationConfig, const CrossoverConfig& crossoverConfig,
               const CompatibilityConfig& compatibilityConfig, const SpeciationConfig& speciationConfig);

    // Advances every unfinished individual by deltaTime. If every
    // individual finishes this call, immediately reproduces the next
    // generation and resets it to spawn.
    void update(float deltaTime);

    // Resets every individual in the CURRENT generation to spawn (same
    // Genomes/generation). Never reproduces.
    void restartGeneration();

    std::size_t getGeneration() const { return m_generation; }
    std::size_t size() const { return m_individuals.size(); }

    const Individual& getIndividual(std::size_t index) const { return m_individuals.at(index); }

    // Highest-fitness individual's index (ties -> lower index).
    std::size_t getBestIndividualIndex() const;

    std::size_t getRunningCount() const;
    std::size_t getFinishedCount() const;

    // Species count for the current/just-finished generation.
    std::size_t getSpeciesCount() const { return m_speciator.getSpecies().size(); }

    // The persistent Species vector backing getSpeciesCount(), ascending by
    // SpeciesId -- for HUD/debug lookups.
    const std::vector<Species>& getCurrentSpecies() const { return m_speciator.getSpecies(); }

    // Per-species reproduction stats from the most recent reproduce() call.
    const std::vector<SpeciesReproductionStats>& getReproductionStats() const { return m_reproductionStats; }

    // Highest fitness of the previous generation (0 before generation 0 finishes).
    float getLastGenerationBestFitness() const { return m_lastGenerationBestFitness; }

    // Full metrics row for the most recently completed generation, captured
    // inside reproduce() before individuals are replaced. Default (all-zero)
    // before the first transition.
    const training::GenerationMetrics& getLastGenerationMetrics() const { return m_lastGenerationMetrics; }

    const PopulationConfig& getPopulationConfig() const { return m_populationConfig; }
    const InnovationTracker& getInnovationTracker() const { return m_innovationTracker; }

    // Samples tournamentSize indices (with replacement) from fitnessValues,
    // picks the winner via isBetterTournamentCandidate(). Draws from the
    // owned orchestration RNG, so a direct call advances it like an
    // internal one would.
    std::size_t tournamentSelect(const std::vector<float>& fitnessValues);

private:
    bool isGenerationFinished() const;

    // Speciates m_individuals' current Genomes and returns m_speciator's
    // updated collection (valid as long as m_speciator's own does).
    const std::vector<Species>& computeCurrentSpecies();

    // Builds the next generation from the current one's Genomes/fitness,
    // replaces m_individuals, and increments m_generation.
    void reproduce();

    // Tournament-selects one parent from a species' own members only, by
    // RAW fitness (never adjusted). Valid for a species of any size.
    std::size_t selectParentFromSpecies(const Species& species, const std::vector<float>& fitnessValues);

    const simulation::Track& m_track;
    simulation::CarParams m_carParams;
    Vector2 m_spawnPosition;
    float m_spawnHeading;

    PopulationConfig m_populationConfig;
    MutationConfig m_mutationConfig;
    CrossoverConfig m_crossoverConfig;
    CompatibilityConfig m_compatibilityConfig;
    SpeciationConfig m_speciationConfig;

    InnovationTracker m_innovationTracker;
    GenomeMutator m_mutator;
    GenomeCrossover m_crossover;
    std::mt19937 m_orchestrationRng;
    Speciator m_speciator;

    std::size_t m_generation;
    float m_lastGenerationBestFitness;
    std::vector<SpeciesReproductionStats> m_reproductionStats;
    training::GenerationMetrics m_lastGenerationMetrics;

    std::vector<Individual> m_individuals;
};

} // namespace ai::neat
