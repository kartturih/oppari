#include "ai/neat/Population.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace ai::neat
{

namespace
{

void validatePopulationConfig(const PopulationConfig& config)
{
    if (config.populationSize <= 1)
    {
        throw std::invalid_argument("PopulationConfig: populationSize must be greater than 1");
    }
    if (config.eliteCount == 0)
    {
        throw std::invalid_argument("PopulationConfig: eliteCount must be greater than 0");
    }
    if (config.eliteCount >= config.populationSize)
    {
        throw std::invalid_argument("PopulationConfig: eliteCount must be less than populationSize");
    }
    if (config.tournamentSize == 0)
    {
        throw std::invalid_argument("PopulationConfig: tournamentSize must be greater than 0");
    }
    if (config.tournamentSize > config.populationSize)
    {
        throw std::invalid_argument("PopulationConfig: tournamentSize must not exceed populationSize");
    }
    if (config.speciesStagnationLimit == 0)
    {
        throw std::invalid_argument("PopulationConfig: speciesStagnationLimit must be greater than 0");
    }
}

// One past the highest NodeId used anywhere in genome (0 if genome has no
// nodes) -- computed by scanning genome itself, never hardcoded, so a
// fresh InnovationTracker seeded from this can never allocate a node ID
// the base genome already uses.
NodeId computeFirstAvailableNodeId(const Genome& genome)
{
    NodeId maxId = -1;
    for (const NodeGene& node : genome.nodes())
    {
        maxId = std::max(maxId, node.getId());
    }
    return maxId + 1;
}

// One past the highest connection innovation number used anywhere in
// genome (0 if genome has no connections); same reasoning as above.
InnovationNumber computeFirstAvailableInnovation(const Genome& genome)
{
    InnovationNumber maxInnovation = -1;
    for (const ConnectionGene& connection : genome.connections())
    {
        maxInnovation = std::max(maxInnovation, connection.getInnovationNumber());
    }
    return maxInnovation + 1;
}

} // namespace

bool isBetterTournamentCandidate(float candidateFitness, std::size_t candidateIndex, float bestFitness,
                                  std::size_t bestIndex)
{
    if (candidateFitness > bestFitness)
    {
        return true;
    }
    if (candidateFitness == bestFitness && candidateIndex < bestIndex)
    {
        return true;
    }
    return false;
}

std::vector<std::size_t> allocateSpeciesOffspring(const std::vector<SpeciesId>& speciesIds,
                                                    const std::vector<float>& effectiveFitnessSums,
                                                    std::size_t remainingSlots)
{
    const std::size_t speciesCount = speciesIds.size();
    std::vector<std::size_t> allocation(speciesCount, 0);
    if (speciesCount == 0)
    {
        return allocation;
    }

    float totalEffectiveFitness = 0.0f;
    for (float sum : effectiveFitnessSums)
    {
        totalEffectiveFitness += sum;
    }

    if (totalEffectiveFitness > 0.0f)
    {
        std::vector<double> exactShares(speciesCount, 0.0);
        std::size_t allocatedSoFar = 0;
        for (std::size_t s = 0; s < speciesCount; ++s)
        {
            exactShares[s] = static_cast<double>(remainingSlots) * static_cast<double>(effectiveFitnessSums[s]) /
                              static_cast<double>(totalEffectiveFitness);
            const std::size_t floorShare = static_cast<std::size_t>(std::floor(exactShares[s]));
            allocation[s] = floorShare;
            allocatedSoFar += floorShare;
        }

        const std::size_t leftover = (allocatedSoFar < remainingSlots) ? (remainingSlots - allocatedSoFar) : 0;

        std::vector<std::size_t> remainderOrder(speciesCount);
        for (std::size_t s = 0; s < speciesCount; ++s)
        {
            remainderOrder[s] = s;
        }
        std::sort(remainderOrder.begin(), remainderOrder.end(),
                  [&exactShares, &allocation, &speciesIds](std::size_t a, std::size_t b)
                  {
                      const double fracA = exactShares[a] - static_cast<double>(allocation[a]);
                      const double fracB = exactShares[b] - static_cast<double>(allocation[b]);
                      if (fracA != fracB)
                      {
                          return fracA > fracB;
                      }
                      return speciesIds[a] < speciesIds[b];
                  });

        for (std::size_t j = 0; j < leftover && j < speciesCount; ++j)
        {
            allocation[remainderOrder[j]] += 1;
        }
    }
    else
    {
        std::vector<std::size_t> order(speciesCount);
        for (std::size_t s = 0; s < speciesCount; ++s)
        {
            order[s] = s;
        }
        std::sort(order.begin(), order.end(),
                  [&speciesIds](std::size_t a, std::size_t b) { return speciesIds[a] < speciesIds[b]; });

        const std::size_t quotient = remainingSlots / speciesCount;
        const std::size_t remainder = remainingSlots % speciesCount;
        for (std::size_t rank = 0; rank < speciesCount; ++rank)
        {
            allocation[order[rank]] = quotient + (rank < remainder ? 1u : 0u);
        }
    }

    return allocation;
}

float effectiveFitnessContribution(float adjustedFitness)
{
    return std::max(adjustedFitness, 0.0f);
}

Population::Population(const Genome& baseGenome, const simulation::Track& track, const simulation::CarParams& carParams,
                        Vector2 spawnPosition, float spawnHeading, const PopulationConfig& populationConfig,
                        const MutationConfig& mutationConfig, const CrossoverConfig& crossoverConfig,
                        const CompatibilityConfig& compatibilityConfig, const SpeciationConfig& speciationConfig)
    : m_track(track)
    , m_carParams(carParams)
    , m_spawnPosition(spawnPosition)
    , m_spawnHeading(spawnHeading)
    , m_populationConfig(populationConfig)
    , m_mutationConfig(mutationConfig)
    , m_crossoverConfig(crossoverConfig)
    , m_compatibilityConfig(compatibilityConfig)
    , m_speciationConfig(speciationConfig)
    , m_innovationTracker(computeFirstAvailableNodeId(baseGenome), computeFirstAvailableInnovation(baseGenome))
    // Deterministic, distinct seed streams derived from one
    // PopulationConfig::randomSeed: the orchestration RNG uses the seed
    // itself, GenomeMutator uses seed+1, GenomeCrossover uses seed+2 --
    // arbitrary but fixed offsets (std::uint32_t wraparound on overflow is
    // well-defined, not UB, so this is safe for any input seed). No
    // std::random_device, rand(), or time-based seeding anywhere.
    , m_mutator(populationConfig.randomSeed + 1u)
    , m_crossover(populationConfig.randomSeed + 2u)
    , m_orchestrationRng(populationConfig.randomSeed)
    , m_speciator()
    , m_generation(0)
    , m_lastGenerationBestFitness(0.0f)
{
    validatePopulationConfig(m_populationConfig);

    m_individuals.reserve(m_populationConfig.populationSize);
    for (std::size_t i = 0; i < m_populationConfig.populationSize; ++i)
    {
        Genome genome = baseGenome;
        if (i != 0)
        {
            m_mutator.mutateWeights(genome, m_mutationConfig);
        }
        m_individuals.emplace_back(std::move(genome), m_track, m_carParams, m_spawnPosition, m_spawnHeading);
    }

    // Establishes m_speciator's persistent species for generation 0 (for
    // HUD display before this generation has even started evaluating) --
    // no fitness history is recorded here; that only ever happens once a
    // generation has actually finished (see reproduce()).
    computeCurrentSpecies();
}

void Population::update(float deltaTime)
{
    for (Individual& individual : m_individuals)
    {
        individual.update(deltaTime);
    }

    if (isGenerationFinished())
    {
        reproduce();
    }
}

void Population::restartGeneration()
{
    for (Individual& individual : m_individuals)
    {
        individual.reset();
    }
}

bool Population::isGenerationFinished() const
{
    for (const Individual& individual : m_individuals)
    {
        if (!individual.isFinished())
        {
            return false;
        }
    }
    return true;
}

std::size_t Population::getBestIndividualIndex() const
{
    std::size_t bestIndex = 0;
    float bestFitness = m_individuals[0].getFitness();
    for (std::size_t i = 1; i < m_individuals.size(); ++i)
    {
        const float fitness = m_individuals[i].getFitness();
        if (fitness > bestFitness)
        {
            bestIndex = i;
            bestFitness = fitness;
        }
    }
    return bestIndex;
}

std::size_t Population::getRunningCount() const
{
    std::size_t count = 0;
    for (const Individual& individual : m_individuals)
    {
        if (!individual.isFinished())
        {
            ++count;
        }
    }
    return count;
}

std::size_t Population::getFinishedCount() const
{
    return m_individuals.size() - getRunningCount();
}

const std::vector<Species>& Population::computeCurrentSpecies()
{
    std::vector<Genome> genomes;
    genomes.reserve(m_individuals.size());
    for (const Individual& individual : m_individuals)
    {
        genomes.push_back(individual.getGenome());
    }
    return m_speciator.speciate(genomes, m_compatibilityConfig, m_speciationConfig);
}

std::size_t Population::selectParentFromSpecies(const Species& species, const std::vector<float>& fitnessValues)
{
    const std::vector<std::size_t>& members = species.getMemberIndices();
    std::uniform_int_distribution<std::size_t> pickMember(0, members.size() - 1);

    std::size_t bestIndex = members[pickMember(m_orchestrationRng)];
    float bestFitness = fitnessValues[bestIndex];

    for (std::size_t sample = 1; sample < m_populationConfig.tournamentSize; ++sample)
    {
        const std::size_t candidateIndex = members[pickMember(m_orchestrationRng)];
        const float candidateFitness = fitnessValues[candidateIndex];
        if (isBetterTournamentCandidate(candidateFitness, candidateIndex, bestFitness, bestIndex))
        {
            bestIndex = candidateIndex;
            bestFitness = candidateFitness;
        }
    }

    return bestIndex;
}

std::size_t Population::tournamentSelect(const std::vector<float>& fitnessValues)
{
    std::uniform_int_distribution<std::size_t> pickIndex(0, fitnessValues.size() - 1);

    std::size_t bestIndex = pickIndex(m_orchestrationRng);
    float bestFitness = fitnessValues[bestIndex];

    for (std::size_t sample = 1; sample < m_populationConfig.tournamentSize; ++sample)
    {
        const std::size_t candidateIndex = pickIndex(m_orchestrationRng);
        const float candidateFitness = fitnessValues[candidateIndex];
        if (isBetterTournamentCandidate(candidateFitness, candidateIndex, bestFitness, bestIndex))
        {
            bestIndex = candidateIndex;
            bestFitness = candidateFitness;
        }
    }

    return bestIndex;
}

void Population::reproduce()
{
    const std::size_t populationSize = m_individuals.size();

    // 1. Preserve final RAW fitness values before anything else changes --
    // never overwritten; adjusted fitness (below) is always a separate
    // vector.
    std::vector<float> fitnessValues;
    fitnessValues.reserve(populationSize);
    for (const Individual& individual : m_individuals)
    {
        fitnessValues.push_back(individual.getFitness());
    }

    float bestFitness = fitnessValues[0];
    for (float fitness : fitnessValues)
    {
        bestFitness = std::max(bestFitness, fitness);
    }
    m_lastGenerationBestFitness = bestFitness;

    // 2. Speciate the generation that just finished ONCE -- this single,
    // persistent Species vector (ascending SpeciesId order, per Speciator's
    // contract; a species compatible with its OLD representative keeps its
    // SpeciesId and its accumulated age/history from earlier generations --
    // see Speciator::speciate()) is the sole source of species membership
    // for every remaining step: fitness sharing, species fitness totals,
    // offspring allocation, and species-local parent pools.
    const std::vector<Species>& currentSpecies = computeCurrentSpecies();
    const std::size_t speciesCount = currentSpecies.size();

    // 2b. Update each species' persistent fitness history from this
    // generation's RAW fitness values, now that they are finally known --
    // Speciator itself only manages membership; this is the one point where
    // Population feeds it the fitness data needed to advance
    // age/historicalBestFitness/generationsSinceImprovement/stagnant (see
    // Species::recordGeneration()). currentSpecies (a reference into
    // m_speciator's own storage) reflects the updated values immediately
    // afterward, since recordGeneration() mutates each Species in place --
    // no reallocation of the species vector happens here.
    m_speciator.updateFitnessHistory(fitnessValues, m_populationConfig.speciesStagnationLimit);

    // 3. Fitness sharing: adjustedFitness[i] = rawFitness[i] / (size of
    // i's species). Exists only for reproduction below -- fitnessValues
    // (raw) is never modified.
    std::vector<float> adjustedFitness(populationSize, 0.0f);
    for (const Species& species : currentSpecies)
    {
        const float speciesSize = static_cast<float>(species.size());
        for (std::size_t memberIndex : species.getMemberIndices())
        {
            adjustedFitness[memberIndex] = fitnessValues[memberIndex] / speciesSize;
        }
    }

    // 4. Per-species adjusted-fitness sum (unclamped -- exposed via
    // SpeciesReproductionStats for HUD/debug) and effective adjusted-fitness
    // sum (each member's contribution clamped to >= 0, per the negative-
    // fitness safety rule -- used only for offspring allocation below).
    std::vector<float> speciesAdjustedFitnessSum(speciesCount, 0.0f);
    std::vector<float> speciesEffectiveFitnessSum(speciesCount, 0.0f);
    for (std::size_t s = 0; s < speciesCount; ++s)
    {
        for (std::size_t memberIndex : currentSpecies[s].getMemberIndices())
        {
            speciesAdjustedFitnessSum[s] += adjustedFitness[memberIndex];
            speciesEffectiveFitnessSum[s] += effectiveFitnessContribution(adjustedFitness[memberIndex]);
        }
    }

    // 4b. Reproduction eligibility (Stage 17): a stagnant species is
    // excluded from normal offspring allocation entirely. EXCEPTION: if
    // every species this generation is stagnant, exactly one -- the one
    // with the highest historicalBestFitness, ties broken by lower
    // SpeciesId -- is temporarily treated as eligible for THIS generation's
    // allocation only, to prevent total population collapse. Processing
    // currentSpecies in its already-ascending-SpeciesId order and only ever
    // replacing the fallback candidate on a STRICTLY greater
    // historicalBestFitness gives the lower-SpeciesId tie-break for free.
    // This override never touches the underlying Species's own stagnant/
    // generationsSinceImprovement state -- it is a reproduction-time
    // allocation decision only, recomputed fresh every generation.
    std::vector<bool> reproductionEligible(speciesCount, true);
    bool anyEligible = false;
    for (std::size_t s = 0; s < speciesCount; ++s)
    {
        reproductionEligible[s] = !currentSpecies[s].isStagnant();
        anyEligible = anyEligible || reproductionEligible[s];
    }
    if (!anyEligible && speciesCount > 0)
    {
        std::size_t fallbackIndex = 0;
        for (std::size_t s = 1; s < speciesCount; ++s)
        {
            if (currentSpecies[s].getHistoricalBestFitness() > currentSpecies[fallbackIndex].getHistoricalBestFitness())
            {
                fallbackIndex = s;
            }
        }
        reproductionEligible[fallbackIndex] = true;
    }

    // 5. Global elitism ranking: indices sorted by (higher RAW fitness
    // first, lower original index as a deterministic tie-break). Elites are
    // selected purely by raw fitness, regardless of their species'
    // stagnation/eligibility -- an exceptional individual from an otherwise
    // stagnant species can still survive as a global elite.
    std::vector<std::size_t> rankedIndices(populationSize);
    for (std::size_t i = 0; i < populationSize; ++i)
    {
        rankedIndices[i] = i;
    }
    std::sort(rankedIndices.begin(), rankedIndices.end(), [&fitnessValues](std::size_t a, std::size_t b)
              {
                  if (fitnessValues[a] != fitnessValues[b])
                  {
                      return fitnessValues[a] > fitnessValues[b];
                  }
                  return a < b;
              });

    // 6. Offspring allocation across species for the remaining (non-elite)
    // slots -- restricted to reproduction-eligible species only (Stage 17):
    // an ineligible (stagnant, non-fallback) species is simply left out of
    // the input to allocateSpeciesOffspring() entirely, so it contributes
    // nothing to the effective-fitness total and the zero-total-fitness
    // fallback (if it triggers) only ever splits slots among the eligible
    // subset. Within that eligible subset, allocation is unchanged from
    // Stage 16: deterministic proportional allocation (floor + largest
    // fractional remainder, ties broken by lower SpeciesId), or an
    // even-as-possible split if every eligible species has zero effective
    // fitness. Global elites are copied separately above/below and never
    // subtracted from any species' fitness total here -- this allocation
    // applies only to remainingSlots.
    const std::size_t remainingSlots = populationSize - m_populationConfig.eliteCount;
    std::vector<SpeciesId> eligibleSpeciesIds;
    std::vector<float> eligibleEffectiveFitnessSums;
    std::vector<std::size_t> eligibleToFullIndex;
    for (std::size_t s = 0; s < speciesCount; ++s)
    {
        if (reproductionEligible[s])
        {
            eligibleSpeciesIds.push_back(currentSpecies[s].getId());
            eligibleEffectiveFitnessSums.push_back(speciesEffectiveFitnessSum[s]);
            eligibleToFullIndex.push_back(s);
        }
    }
    const std::vector<std::size_t> eligibleAllocation =
        allocateSpeciesOffspring(eligibleSpeciesIds, eligibleEffectiveFitnessSums, remainingSlots);

    std::vector<std::size_t> offspringAllocation(speciesCount, 0);
    for (std::size_t e = 0; e < eligibleToFullIndex.size(); ++e)
    {
        offspringAllocation[eligibleToFullIndex[e]] = eligibleAllocation[e];
    }

    std::size_t totalAllocated = 0;
    for (std::size_t count : offspringAllocation)
    {
        totalAllocated += count;
    }
    if (totalAllocated != remainingSlots)
    {
        throw std::logic_error("Population::reproduce: species offspring allocation did not sum to the remaining slot count");
    }

    // 7. Record this generation's reproduction stats for read-only
    // HUD/debug/testing consumption -- derived entirely from the data
    // computed above, never mutated afterward.
    m_reproductionStats.clear();
    m_reproductionStats.reserve(speciesCount);
    for (std::size_t s = 0; s < speciesCount; ++s)
    {
        SpeciesReproductionStats stats;
        stats.speciesId = currentSpecies[s].getId();
        stats.memberCount = currentSpecies[s].size();
        stats.adjustedFitnessSum = speciesAdjustedFitnessSum[s];
        stats.allocatedOffspring = offspringAllocation[s];
        stats.age = currentSpecies[s].getAge();
        stats.historicalBestFitness = currentSpecies[s].getHistoricalBestFitness();
        stats.generationsSinceImprovement = currentSpecies[s].getGenerationsSinceImprovement();
        stats.stagnant = currentSpecies[s].isStagnant();
        stats.reproductionEligible = reproductionEligible[s];
        m_reproductionStats.push_back(stats);
    }

    // 8. Construct every new Genome -- entirely from the current (soon to
    // be replaced) generation's data. m_individuals is not touched at any
    // point during this construction, so parent selection always reads the
    // complete, unmodified old generation. Global elites are copied first
    // (no crossover, no mutation), then each species' allocated offspring
    // is built using parents drawn only from that same species -- species
    // are processed in ascending SpeciesId order, matching currentSpecies.
    // A species with zero allocated offspring (stagnant and not this
    // generation's fallback) simply contributes no non-elite children --
    // the inner loop below never executes for it.
    std::vector<Genome> newGenomes;
    newGenomes.reserve(populationSize);

    for (std::size_t i = 0; i < m_populationConfig.eliteCount; ++i)
    {
        newGenomes.push_back(m_individuals[rankedIndices[i]].getGenome()); // copy, unmutated
    }

    for (std::size_t s = 0; s < speciesCount; ++s)
    {
        const Species& species = currentSpecies[s];
        for (std::size_t offspring = 0; offspring < offspringAllocation[s]; ++offspring)
        {
            // Both parents come from this species' member indices only --
            // no cross-species mating. A singleton species (size 1) simply
            // resolves both selections to its one member; crossover(parent,
            // parent) with equal fitness is well-defined (GenomeCrossover
            // never throws on identical parents) and deterministically
            // reproduces that member's own genes, so no special-casing is
            // needed for singleton species.
            const std::size_t parentAIndex = selectParentFromSpecies(species, fitnessValues);
            const std::size_t parentBIndex = selectParentFromSpecies(species, fitnessValues);
            const Genome& parentA = m_individuals[parentAIndex].getGenome();
            const Genome& parentB = m_individuals[parentBIndex].getGenome();

            // RAW fitness is passed into crossover (never adjusted fitness)
            // -- preserves the existing fitter-parent gene inheritance rule.
            Genome child = m_crossover.crossover(parentA, fitnessValues[parentAIndex], parentB, fitnessValues[parentBIndex],
                                                  m_crossoverConfig);

            m_mutator.mutateWeights(child, m_mutationConfig);
            m_mutator.mutateAddConnection(child, m_innovationTracker, m_mutationConfig);
            m_mutator.mutateAddNode(child, m_innovationTracker, m_mutationConfig);

            newGenomes.push_back(std::move(child));
        }
    }

    if (newGenomes.size() != populationSize)
    {
        throw std::logic_error("Population::reproduce: constructed next generation does not match populationSize");
    }

    // 9. Rebuild every Individual from the new Genomes, then replace the
    // population as one coherent operation.
    std::vector<Individual> newIndividuals;
    newIndividuals.reserve(populationSize);
    for (Genome& genome : newGenomes)
    {
        newIndividuals.emplace_back(std::move(genome), m_track, m_carParams, m_spawnPosition, m_spawnHeading);
    }

    m_individuals = std::move(newIndividuals);
    ++m_generation;
}

} // namespace ai::neat
