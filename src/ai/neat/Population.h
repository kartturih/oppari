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

// The pure decision rule tournament selection uses to pick a winner among
// sampled candidates: true when the candidate should replace the current
// best -- either strictly higher fitness, or exactly equal fitness with a
// strictly lower population index (the documented tie-break). A free
// function, independent of any Population instance, RNG, or sampling --
// exposed so the exact selection/tie-break rule can be exercised directly
// and deterministically, with zero dependency on which indices a random
// sample happens to contain.
bool isBetterTournamentCandidate(float candidateFitness, std::size_t candidateIndex, float bestFitness,
                                  std::size_t bestIndex);

// Allocates remainingSlots non-elite offspring across species, from each
// species' EFFECTIVE (already clamped to >= 0 per-member) adjusted-fitness
// sum -- speciesIds and effectiveFitnessSums must be the same length, index
// i of both describing the same species; input order does not matter, and
// need not be sorted. Deterministic proportional allocation: each species'
// exact share is remainingSlots * itsSum / totalOfAllSums, floored, with any
// leftover slots (remainingSlots minus the sum of floors) distributed one at
// a time to the species with the largest fractional remainder, ties broken
// by lower SpeciesId. If totalOfAllSums is exactly 0 (e.g. every species'
// members have non-positive raw fitness), falls back to splitting
// remainingSlots as evenly as possible across every species (by count, not
// fitness), with species ascending by SpeciesId receiving the leftover
// remainder first.
//
// A free function, independent of any Population instance/RNG/Speciator --
// exposed so the allocation algorithm can be exercised directly and
// deterministically against arbitrary synthetic species-fitness data.
// Returns a vector the same length as speciesIds that always sums to
// exactly remainingSlots (an empty speciesIds returns an empty vector
// regardless of remainingSlots -- callers must never call this with a
// nonzero remainingSlots and no species).
std::vector<std::size_t> allocateSpeciesOffspring(const std::vector<SpeciesId>& speciesIds,
                                                    const std::vector<float>& effectiveFitnessSums,
                                                    std::size_t remainingSlots);

// The per-member contribution to a species' EFFECTIVE adjusted-fitness sum
// (used only for offspring allocation, via allocateSpeciesOffspring()
// above): adjustedFitness clamped to a minimum of 0, so a negative raw (and
// therefore negative adjusted) fitness can never reduce -- let alone make
// negative -- a species' allocation share. Never affects the stored raw
// fitness, and never affects the unclamped adjustedFitnessSum exposed via
// Population::SpeciesReproductionStats for HUD/debug display. A free
// function purely so this negative-fitness safety rule can be exercised
// directly and deterministically -- the exact same one-line clamp
// Population::reproduce() itself applies.
float effectiveFitnessContribution(float adjustedFitness);

// Orchestrates one complete NEAT evolutionary run: an initial population
// built from one base Genome, advanced generation by generation through
// tournament selection, elitism, crossover, and the existing structural/
// weight mutation operators. Population is the first class in this
// codebase to run more than one Car/AIController/TrackProgress/
// FitnessEvaluator simultaneously -- every Individual it owns is a fully
// independent instance (see Individual.h); nothing about one individual's
// simulation state (collision, progress, fitness) can affect another's.
//
// Owns, for the lifetime of the whole run:
//   - one InnovationTracker, shared by every structural mutation across
//     every individual and every generation (never one per Genome/
//     individual -- see the constructor's doc comment for how its starting
//     counters are derived from the base Genome);
//   - one GenomeMutator and one GenomeCrossover, each seeded once at
//     construction and reused for the whole run, so their RNG state keeps
//     advancing generation after generation exactly like any other stage's
//     mutator/crossover usage;
//   - one orchestration std::mt19937, used only for tournament-selection
//     sampling (mutation/crossover randomness always goes through their
//     own owned generators, never this one).
// All three of the above are seeded deterministically from
// PopulationConfig::randomSeed (see the .cpp for the exact derivation) --
// no std::random_device, no rand(), no time-based seeding, anywhere. The
// same seed, base Genome, configs, and sequence of update() calls/fitness
// outcomes always reproduces the same evolutionary sequence.
//
// Species membership (via the existing Speciator) is computed once per
// generation and, starting with Stage 16, actively drives reproduction:
// fitness sharing (adjusted fitness), per-species offspring allocation, and
// species-local parent selection all read the same single Species vector
// computed at the start of that generation's reproduce() call -- see the
// .cpp for the full algorithm. As of Stage 17, species identity is
// persistent across generations: m_speciator itself now owns the species
// collection for this Population's entire run (never rebuilt from scratch),
// so a species whose representative later genomes remain compatible with
// keeps its SpeciesId, and keeps accumulating age/historical-best-fitness/
// stagnation state, for as long as it keeps matching -- see
// Speciator::speciate()'s own doc comment for the exact per-call lifecycle.
// A species that stops improving for speciesStagnationLimit consecutive
// completed generations is excluded from normal offspring allocation (with
// a global-elite exception and an all-species-stagnant safety fallback --
// see reproduce() in the .cpp). Adaptive compatibility thresholds and
// interspecies mating remain out of scope.
class Population
{
public:
    // Read-only reproduction statistics for one Species from the most
    // recently completed reproduce() call -- purely for HUD/debug display
    // and testing; Population never mutates this after reproduce()
    // finishes, and it is empty until the first generation transition
    // occurs. adjustedFitnessSum is the unclamped sum of its members'
    // adjustedFitness (rawFitness / memberCount) -- it may be negative;
    // allocatedOffspring is the number of non-elite offspring slots this
    // species received, computed from the effective (clamped-at-zero)
    // adjusted fitness only among reproduction-eligible species (Stage 17).
    // age/historicalBestFitness/generationsSinceImprovement/stagnant mirror
    // the underlying persistent Species's own fields (see Species.h) at the
    // moment reproduce() ran. reproductionEligible is true unless this
    // species was excluded from this generation's offspring allocation for
    // being stagnant -- always true for a non-stagnant species, and true for
    // at most one stagnant species per generation (the all-stagnant safety
    // fallback -- see reproduce() in the .cpp). See Population.cpp for the
    // exact algorithm.
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
    // individuals 1..populationConfig.populationSize-1 are copies of
    // baseGenome with GenomeMutator::mutateWeights() applied (no structural
    // mutation at this stage, so every initial individual shares
    // baseGenome's exact topology). Every individual shares the same
    // track/carParams/spawn pose.
    //
    // The InnovationTracker is seeded so firstAvailableNodeId is one past
    // the highest NodeId in baseGenome, and firstAvailableInnovation is one
    // past the highest connection innovation number in baseGenome -- both
    // computed by scanning baseGenome itself, never hardcoded, so future
    // structural mutation can never collide with an ID/innovation
    // baseGenome already uses.
    //
    // Throws std::invalid_argument if populationConfig is invalid:
    // populationSize <= 1, eliteCount == 0, eliteCount >= populationSize,
    // tournamentSize == 0, or tournamentSize > populationSize. Propagates
    // whatever ai::neat::buildPhenotype() throws if baseGenome cannot be
    // turned into a phenotype.
    Population(const Genome& baseGenome, const simulation::Track& track, const simulation::CarParams& carParams,
               Vector2 spawnPosition, float spawnHeading, const PopulationConfig& populationConfig,
               const MutationConfig& mutationConfig, const CrossoverConfig& crossoverConfig,
               const CompatibilityConfig& compatibilityConfig, const SpeciationConfig& speciationConfig);

    // Advances every not-yet-finished individual by deltaTime seconds. If
    // every individual is finished after this call, immediately reproduces
    // the next generation (elitism + tournament-selected crossover +
    // mutation, all built from the just-finished generation's Genomes and
    // fitness values), replaces the population as one coherent operation,
    // increments the generation counter, and resets every new individual
    // to spawn -- see the .cpp for the full algorithm. A caller only ever
    // needs to call update() every frame; it never needs to separately
    // detect and drive a generation transition.
    void update(float deltaTime);

    // Resets every individual in the CURRENT generation back to spawn (same
    // Genomes, same generation number, same InnovationTracker/mutator/
    // crossover state) -- the population-wide equivalent of the old
    // single-car "R" restart. Never reproduces and never changes
    // getGeneration().
    void restartGeneration();

    std::size_t getGeneration() const { return m_generation; }
    std::size_t size() const { return m_individuals.size(); }

    const Individual& getIndividual(std::size_t index) const { return m_individuals.at(index); }

    // Index of the individual with the highest current fitness among all
    // individuals (ties broken by lower index). Valid for any Population
    // (populationSize > 1 is enforced by the constructor).
    std::size_t getBestIndividualIndex() const;

    std::size_t getRunningCount() const;
    std::size_t getFinishedCount() const;

    // Number of species m_speciator currently groups this population's
    // Genomes into, computed once per generation (at construction for
    // generation 0, and again each time a generation finishes, from the
    // Genomes of the generation that just finished) -- see the class
    // comment for how this same persistent Species vector now drives
    // reproduction as of Stage 16/17.
    std::size_t getSpeciesCount() const { return m_speciator.getSpecies().size(); }

    // The full, persistent Species vector backing getSpeciesCount() --
    // m_speciator's own owned collection (see Speciator::getSpecies()), in
    // ascending SpeciesId order -- read-only, for HUD/debug lookups such as
    // "which species is individual i in" or "how big/old/stagnant is that
    // species". The same vector reproduce() itself used to drive fitness
    // sharing/offspring allocation/parent selection for the generation that
    // just finished (or, before the first transition, the freshly
    // constructed generation 0).
    const std::vector<Species>& getCurrentSpecies() const { return m_speciator.getSpecies(); }

    // Per-species reproduction statistics from the most recently completed
    // reproduce() call -- see SpeciesReproductionStats above. Empty before
    // the first generation transition.
    const std::vector<SpeciesReproductionStats>& getReproductionStats() const { return m_reproductionStats; }

    // The highest fitness reached by the previous generation, frozen at
    // the moment it finished (0 before generation 0 has finished).
    float getLastGenerationBestFitness() const { return m_lastGenerationBestFitness; }

    // Stage 21: the full training::GenerationMetrics row for the most
    // recently completed generation, computed inside reproduce() (see the
    // .cpp) from that generation's individuals/species BEFORE m_individuals
    // is replaced -- the only point in this class where that raw
    // per-individual data (fitness, progress, hasCompletedLap, genome
    // complexity) is still available; once reproduce() returns, only this
    // captured snapshot survives. A default-constructed (all-zero)
    // GenerationMetrics before the first generation transition, exactly
    // like getReproductionStats() being empty until then.
    const training::GenerationMetrics& getLastGenerationMetrics() const { return m_lastGenerationMetrics; }

    const PopulationConfig& getPopulationConfig() const { return m_populationConfig; }
    const InnovationTracker& getInnovationTracker() const { return m_innovationTracker; }

    // Samples populationConfig.tournamentSize indices independently and
    // uniformly at random from fitnessValues (sampling with replacement --
    // the same index may be drawn more than once), reducing them via
    // isBetterTournamentCandidate() above, and returns the winning index.
    // Draws from this Population's own owned orchestration RNG, so calling
    // it directly (e.g. for testing) advances that same state exactly as
    // an internal call during reproduce() would. Exposed publicly mainly
    // so the selection algorithm can be exercised directly against an
    // explicit, caller-controlled fitness vector.
    std::size_t tournamentSelect(const std::vector<float>& fitnessValues);

private:
    bool isGenerationFinished() const;

    // Speciates the current m_individuals' Genomes with m_speciator, using
    // m_compatibilityConfig/m_speciationConfig, and returns a reference to
    // m_speciator's own (now updated) persistent Species collection -- the
    // single source of species membership for both HUD display and (from
    // within reproduce()) fitness sharing/offspring allocation/parent
    // selection. Never returns a Population-owned copy: the returned
    // reference stays valid exactly as long as m_speciator's own
    // getSpecies() does (see Speciator.h).
    const std::vector<Species>& computeCurrentSpecies();

    // Builds the entire next generation's Genomes from the current
    // (about-to-be-replaced) generation's Genomes/fitness values, then
    // replaces m_individuals as one coherent operation and increments
    // m_generation. See the .cpp for the full species-aware elitism/
    // fitness-sharing/offspring-allocation/parent-selection/crossover/
    // mutation algorithm (Stage 16), extended in Stage 17 to update each
    // species' persistent fitness history and exclude stagnant species from
    // normal offspring allocation.
    void reproduce();

    // Tournament-selects one parent from species' member indices only:
    // samples m_populationConfig.tournamentSize member indices independently
    // and uniformly at random, with replacement, from species and returns
    // the winner per isBetterTournamentCandidate() (highest RAW fitness,
    // ties broken by lower original population index) -- never adjusted
    // fitness. Valid for a species of any size, including exactly one (every
    // sample trivially resolves to that one member). Draws from this
    // Population's own owned orchestration RNG.
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
