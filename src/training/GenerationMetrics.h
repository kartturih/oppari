#pragma once

#include <cstddef>
#include <vector>

#include "ai/neat/Genome.h"

namespace training
{

// ---------------------------------------------------------------------------
// Stage 21: pure, file-I/O-free statistics used to summarize one completed
// NEAT generation. Every function/struct here is a free function or plain
// data -- nothing in this header touches ai::neat::Population, Individual,
// Species, or Speciator directly, and nothing performs any randomness or
// mutation of genetic state. The one caller (ai::neat::Population::
// reproduce(), see Population.cpp) is responsible for extracting the raw
// per-individual/per-species values from its own about-to-be-replaced
// generation and handing them to buildGenerationMetrics() below -- this
// keeps the arithmetic itself (mean/median/min/max, genome-complexity
// counting) exercisable in isolation, with synthetic data, from main.cpp's
// verify*() suite.
// ---------------------------------------------------------------------------

// Arithmetic mean of values. Throws std::invalid_argument if values is
// empty (there is no meaningful mean of zero numbers, and every real call
// site already guarantees a non-empty population).
float computeMean(const std::vector<float>& values);

// Median of values: sorts an internal copy ascending, then returns the
// single middle element for an odd count, or the arithmetic mean of the two
// middle elements for an even count (the standard definition -- e.g. for
// [1, 2, 3, 4] the median is (2 + 3) / 2 = 2.5). Throws std::invalid_argument
// if values is empty.
float computeMedian(std::vector<float> values);

// Minimum/maximum of values. Throw std::invalid_argument if values is empty.
float computeMin(const std::vector<float>& values);
float computeMax(const std::vector<float>& values);

// One genome's structural size, as of the moment it is measured -- never
// mutates or re-derives anything about the genome itself.
struct GenomeComplexity
{
    // Total node genes (Input + Bias + Hidden + Output).
    std::size_t nodeCount = 0;

    // Total connection GENES, counting both enabled and disabled ones (a
    // disabled connection gene is never removed by mutation/crossover --
    // see GenomeMutator/GenomeCrossover -- so it still occupies genome
    // space and still matters for compatibility distance).
    std::size_t connectionGeneCount = 0;

    // Connection genes with isEnabled() == true only -- the subset that
    // actually contributes to the phenotype's runtime graph (see
    // PhenotypeBuilder).
    std::size_t enabledConnectionCount = 0;
};

// Counts genome.nodes().size(), genome.connections().size(), and the subset
// of connections with isEnabled() == true. Pure read-only inspection --
// never mutates genome.
GenomeComplexity computeGenomeComplexity(const ai::neat::Genome& genome);

// One completed generation's full metrics row -- see buildGenerationMetrics()
// below for how every field is derived, and TrainingLogger.h for how a row
// of these becomes one CSV line. Field-by-field definitions (also mirrored
// in the Stage 21 report):
//
//   generation                    -- the generation number that just
//                                     finished (0-based, matching
//                                     ai::neat::Population::getGeneration()
//                                     BEFORE the transition that follows it).
//
//   bestFitness/avgFitness/medianFitness/worstFitness
//                                     -- computed over every individual's RAW
//                                     evaluation fitness (ai::FitnessEvaluator
//                                     ::getFitness() at the moment its
//                                     evaluation finished) -- never the
//                                     species-adjusted (fitness-shared) value
//                                     used internally for reproduction.
//   avgAdjustedFitness               -- mean of the species-ADJUSTED fitness
//                                     (rawFitness / speciesSize, the exact
//                                     value ai::neat::Population::reproduce()
//                                     itself uses for offspring allocation --
//                                     see Population.cpp) across every
//                                     individual. Logged purely as a separate,
//                                     clearly-named field alongside the raw
//                                     values above -- never substituted for
//                                     them. 0 if adjusted fitness was not
//                                     supplied to buildGenerationMetrics().
//
//   speciesCount                     -- number of species this generation
//                                     was grouped into (ai::neat::Speciator::
//                                     getSpecies().size() at the moment this
//                                     generation finished).
//   largestSpeciesSize/smallestSpeciesSize
//                                     -- member-count extremes across that
//                                     same species collection.
//   bestSpeciesHistoricalFitness      -- the highest ai::neat::Species::
//                                     getHistoricalBestFitness() (an
//                                     all-time-high per species, tracked
//                                     across every generation that species
//                                     has existed for -- see Species.h) among
//                                     this generation's species. Distinct
//                                     from bestFitness: it can be higher (an
//                                     earlier generation's peak, if this
//                                     generation regressed) but never lower
//                                     for the species containing this
//                                     generation's own best individual.
//   stagnantSpeciesExcluded           -- number of species EXCLUDED from
//                                     normal offspring allocation this
//                                     generation for being stagnant (Stage
//                                     17's speciesStagnationLimit rule -- see
//                                     Population::reproduce()). This is a
//                                     reproduction-time allocation count, NOT
//                                     a species-removal count: an excluded
//                                     species is not deleted here -- Speciator
//                                     only ever erases a species once it
//                                     actually ends up with zero members
//                                     after a LATER speciate() call.
//
//   bestProgress/avgProgress          -- simulation::TrackProgress::
//                                     getBestProgress() (forward progress
//                                     around the track since spawn, in laps,
//                                     anti-exploit/monotonic -- see
//                                     TrackProgress.h) for the best individual
//                                     (by raw fitness) and averaged across the
//                                     whole population, respectively.
//   lapsCompletedCount/completionRate -- lapsCompletedCount is how many
//                                     individuals had ai::FitnessEvaluator::
//                                     hasCompletedLap() == true (at least one
//                                     full ordered-checkpoint lap) by the time
//                                     their evaluation finished; completionRate
//                                     = lapsCompletedCount / population size.
//                                     This is the only "reached a completion
//                                     condition" signal the existing
//                                     architecture provides -- there is no
//                                     separate goal/finish-line concept.
//
//   bestGenomeNodeCount/bestGenomeConnectionGeneCount/
//   bestGenomeEnabledConnectionCount  -- GenomeComplexity of the best
//                                     individual's (by raw fitness) Genome.
//   avgGenomeNodeCount/avgGenomeConnectionGeneCount
//                                     -- population-average node/connection-
//                                     gene counts (connection genes counted
//                                     the same way as bestGenomeConnectionGeneCount
//                                     -- enabled and disabled both).
//
//   generationDurationSeconds         -- the longest ai::FitnessEvaluator::
//                                     getElapsedTime() reached by any
//                                     individual this generation, i.e. how
//                                     much SIMULATED time (not wall-clock
//                                     time, and not frame count) this
//                                     generation actually ran for before
//                                     every individual had finished. Bounded
//                                     above by FitnessEvaluator's fixed
//                                     30-second evaluation timeout, PLUS at
//                                     most one simulation step (kSimulationDt,
//                                     1/60s): the timeout is only detected
//                                     AFTER elapsedTime is advanced past 30s,
//                                     never pre-empted mid-step -- so a value
//                                     up to ~30.017s is normal, not a bug.
//                                     Can be considerably shorter if every
//                                     individual finished earlier (collision
//                                     or the no-progress timeout).
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
};

// Raw, per-individual/per-species inputs buildGenerationMetrics() reduces
// into one GenerationMetrics row. Every per-individual vector
// (rawFitness/adjustedFitness/bestProgressValues/completedLap/
// genomeComplexities) must be either empty (adjustedFitness only -- see
// below) or exactly the same length, index-aligned to the same individual --
// buildGenerationMetrics() throws std::invalid_argument otherwise.
// adjustedFitness may be passed empty (avgAdjustedFitness is then reported
// as 0) so this struct/function stay usable even from a context that has no
// notion of species-adjusted fitness at all.
struct GenerationMetricsInput
{
    std::size_t generation = 0;

    std::vector<float> rawFitness;
    std::vector<float> adjustedFitness; // may be empty -- see comment above
    std::vector<float> bestProgressValues;
    std::vector<bool> completedLap;
    std::vector<GenomeComplexity> genomeComplexities;

    // Index into every vector above identifying the best individual (highest
    // raw fitness, ties broken by lowest index) -- the caller already knows
    // this from its own ranking, so it is never recomputed here.
    std::size_t bestIndividualIndex = 0;

    std::size_t speciesCount = 0;
    std::size_t largestSpeciesSize = 0;
    std::size_t smallestSpeciesSize = 0;
    float bestSpeciesHistoricalFitness = 0.0f;
    std::size_t stagnantSpeciesExcluded = 0;

    float generationDurationSeconds = 0.0f;
};

// Reduces input into one GenerationMetrics row -- see GenerationMetrics'
// own field-by-field doc comment above for exactly what each output field
// means and how it is derived. Throws std::invalid_argument if
// input.rawFitness is empty, if bestProgressValues/completedLap/
// genomeComplexities do not each have the same size as rawFitness, if
// adjustedFitness is non-empty but a different size than rawFitness, or if
// bestIndividualIndex is out of range.
GenerationMetrics buildGenerationMetrics(const GenerationMetricsInput& input);

} // namespace training
