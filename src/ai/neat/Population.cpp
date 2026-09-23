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

// Only the adaptive-threshold tuning fields -- compatibilityThreshold itself
// is still validated (finite, non-negative) by Speciator on every speciate()
// call, unchanged.
void validateSpeciationTuning(const SpeciationConfig& config)
{
    if (config.targetSpeciesMin == 0)
    {
        throw std::invalid_argument("SpeciationConfig: targetSpeciesMin must be greater than 0");
    }
    if (config.targetSpeciesMin > config.targetSpeciesMax)
    {
        throw std::invalid_argument("SpeciationConfig: targetSpeciesMin must not exceed targetSpeciesMax");
    }
    if (!std::isfinite(config.compatibilityThresholdAdjustment) || config.compatibilityThresholdAdjustment <= 0.0f)
    {
        throw std::invalid_argument("SpeciationConfig: compatibilityThresholdAdjustment must be finite and positive");
    }
    if (!std::isfinite(config.minimumCompatibilityThreshold) || !std::isfinite(config.maximumCompatibilityThreshold) ||
        config.minimumCompatibilityThreshold < 0.0f ||
        config.minimumCompatibilityThreshold > config.maximumCompatibilityThreshold)
    {
        throw std::invalid_argument("SpeciationConfig: minimumCompatibilityThreshold must be finite, non-negative, "
                                     "and not exceed maximumCompatibilityThreshold");
    }
}

// One past the highest NodeId in genome (0 if none) -- so a fresh
// InnovationTracker can never collide with an ID baseGenome already uses.
NodeId computeFirstAvailableNodeId(const Genome& genome)
{
    NodeId maxId = -1;
    for (const NodeGene& node : genome.nodes())
    {
        maxId = std::max(maxId, node.getId());
    }
    return maxId + 1;
}

// One past the highest connection innovation number in genome.
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
    // Distinct deterministic seed streams from one randomSeed (no
    // random_device/rand()/time-based seeding anywhere).
    , m_mutator(populationConfig.randomSeed + 1u)
    , m_crossover(populationConfig.randomSeed + 2u)
    , m_orchestrationRng(populationConfig.randomSeed)
    , m_speciator()
    , m_generation(0)
    , m_lastGenerationBestFitness(0.0f)
    , m_currentCompatibilityThreshold(speciationConfig.compatibilityThreshold)
{
    validatePopulationConfig(m_populationConfig);
    validateSpeciationTuning(m_speciationConfig);

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

    // Establishes species for generation 0 (for HUD display); no fitness
    // history recorded yet -- that happens once a generation finishes.
    computeCurrentSpecies();
}

void Population::update(float deltaTime)
{
    for (Individual& individual : m_individuals)
    {
        individual.update(deltaTime);
    }

    // Computed once and reused below -- identical to the previous single
    // inline call, just hoisted so an attached observer can see the same
    // answer reproduce() is about to act on.
    const bool generationFinished = isGenerationFinished();
    if (m_perStepObserver)
    {
        m_perStepObserver(*this, generationFinished);
    }

    if (generationFinished)
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

    // Speciator only ever sees "the threshold to use right now" -- the
    // runtime-adaptive value, not the (fixed, initial-only) one stored in
    // m_speciationConfig. m_speciationConfig itself is never mutated.
    SpeciationConfig effectiveSpeciationConfig = m_speciationConfig;
    effectiveSpeciationConfig.compatibilityThreshold = m_currentCompatibilityThreshold;
    return m_speciator.speciate(genomes, m_compatibilityConfig, effectiveSpeciationConfig);
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

    // 1. Preserve final raw fitness (adjusted fitness below is separate).
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

    // 2. Speciate the finished generation once; this Species vector drives
    // every remaining step (fitness sharing, allocation, parent selection).
    const std::vector<Species>& currentSpecies = computeCurrentSpecies();
    const std::size_t speciesCount = currentSpecies.size();

    // Threshold actually used to produce currentSpecies above -- this is
    // what gets reported (step 7b) as "this generation's" threshold. The
    // adaptive step-10 adjustment below only prepares the value for the
    // NEXT generation's speciate() call, so this stays the single, stable
    // threshold this whole reproduce() call (and the generation it
    // finishes) used.
    const float compatibilityThresholdUsed = m_currentCompatibilityThreshold;

    // 2b. Feed raw fitness into each species' persistent history now that
    // it's known (age/historicalBestFitness/stagnation).
    m_speciator.updateFitnessHistory(fitnessValues, m_populationConfig.speciesStagnationLimit);

    // 3. Fitness sharing: adjustedFitness[i] = rawFitness[i] / speciesSize.
    std::vector<float> adjustedFitness(populationSize, 0.0f);
    for (const Species& species : currentSpecies)
    {
        const float speciesSize = static_cast<float>(species.size());
        for (std::size_t memberIndex : species.getMemberIndices())
        {
            adjustedFitness[memberIndex] = fitnessValues[memberIndex] / speciesSize;
        }
    }

    // 4. Per-species adjusted-fitness sum (unclamped, for HUD) and
    // effective sum (clamped >= 0 per member, for offspring allocation).
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

    // 4b. A stagnant species is excluded from offspring allocation, unless
    // every species is stagnant -- then the one with the highest
    // historicalBestFitness (lowest SpeciesId breaks ties) is temporarily
    // allowed, to prevent total collapse. Never touches Species's own state.
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

    // 5. Global elitism ranking: higher raw fitness first, lower index breaks
    // ties. An exceptional individual in a stagnant species can still be an elite.
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

    // 6. Offspring allocation for the non-elite slots, restricted to
    // eligible species (an ineligible one gets nothing and contributes
    // nothing to the totals allocateSpeciesOffspring() sees).
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

    // 7. Record this generation's reproduction stats for HUD/debug/testing.
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

    // 7b. Snapshot this generation's metrics -- last point where
    // m_individuals still holds the just-finished generation.
    {
        training::GenerationMetricsInput metricsInput;
        metricsInput.generation = m_generation;
        metricsInput.rawFitness = fitnessValues;
        metricsInput.adjustedFitness = adjustedFitness;
        metricsInput.bestIndividualIndex = rankedIndices[0];
        metricsInput.compatibilityThresholdUsed = compatibilityThresholdUsed;

        metricsInput.bestProgressValues.reserve(populationSize);
        metricsInput.completedLap.reserve(populationSize);
        metricsInput.genomeComplexities.reserve(populationSize);
        metricsInput.finishReasons.reserve(populationSize);
        float generationDurationSeconds = 0.0f;
        for (const Individual& individual : m_individuals)
        {
            metricsInput.bestProgressValues.push_back(individual.getProgress().getBestProgress());
            metricsInput.completedLap.push_back(individual.getFitnessEvaluator().hasCompletedLap());
            metricsInput.genomeComplexities.push_back(training::computeGenomeComplexity(individual.getGenome()));
            metricsInput.finishReasons.push_back(individual.getFitnessEvaluator().getFinishReason());
            generationDurationSeconds =
                std::max(generationDurationSeconds, individual.getFitnessEvaluator().getElapsedTime());
        }
        metricsInput.generationDurationSeconds = generationDurationSeconds;
        metricsInput.bestDriving = m_individuals[rankedIndices[0]].getDrivingSummary();

        metricsInput.speciesCount = speciesCount;
        if (speciesCount > 0)
        {
            std::size_t largest = currentSpecies[0].size();
            std::size_t smallest = currentSpecies[0].size();
            float bestHistorical = currentSpecies[0].getHistoricalBestFitness();
            for (std::size_t s = 1; s < speciesCount; ++s)
            {
                largest = std::max(largest, currentSpecies[s].size());
                smallest = std::min(smallest, currentSpecies[s].size());
                bestHistorical = std::max(bestHistorical, currentSpecies[s].getHistoricalBestFitness());
            }
            metricsInput.largestSpeciesSize = largest;
            metricsInput.smallestSpeciesSize = smallest;
            metricsInput.bestSpeciesHistoricalFitness = bestHistorical;
        }
        std::size_t stagnantExcluded = 0;
        for (std::size_t s = 0; s < speciesCount; ++s)
        {
            if (currentSpecies[s].isStagnant() && !reproductionEligible[s])
            {
                ++stagnantExcluded;
            }
        }
        metricsInput.stagnantSpeciesExcluded = stagnantExcluded;

        m_lastGenerationMetrics = training::buildGenerationMetrics(metricsInput);
    }

    // 8. Build every new Genome from the current (soon-replaced) generation.
    // Elites copied first, then each species' allocated offspring from
    // parents drawn only within that species.
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
            // No cross-species mating; a singleton species mates with itself.
            const std::size_t parentAIndex = selectParentFromSpecies(species, fitnessValues);
            const std::size_t parentBIndex = selectParentFromSpecies(species, fitnessValues);
            const Genome& parentA = m_individuals[parentAIndex].getGenome();
            const Genome& parentB = m_individuals[parentBIndex].getGenome();

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

    // 9. Release the just-finished generation before building the next one.
    // newGenomes above is already complete, independent data (copied/moved
    // out of the old Individuals' Genomes), so nothing past this point needs
    // m_individuals -- clearing it now destroys every old Individual (and
    // therefore its Car's private Box2D b2World) before any new Individual
    // exists. Building the new generation while the old one was still alive
    // meant up to 2x populationSize concurrent Box2D worlds, which at
    // populationSize == 100 exceeds Box2D's B2_MAX_WORLDS (128) cap. Clearing
    // first instead of move-assigning a separately-built vector keeps the
    // peak at populationSize (old OR new, never both).
    m_individuals.clear();

    m_individuals.reserve(populationSize);
    for (Genome& genome : newGenomes)
    {
        m_individuals.emplace_back(std::move(genome), m_track, m_carParams, m_spawnPosition, m_spawnHeading);
    }

    // 10. Adapt the compatibility threshold for the NEXT generation's
    // speciation pass, based on speciesCount (step 2) -- the species count
    // THIS generation's speciate() call, using compatibilityThresholdUsed,
    // actually produced. Never touches m_speciationConfig (see its own
    // comment) or anything already computed above -- this generation's
    // reproduction used one stable threshold throughout.
    m_currentCompatibilityThreshold =
        adjustCompatibilityThreshold(m_currentCompatibilityThreshold, speciesCount, m_speciationConfig);

    ++m_generation;
}

float Population::adjustCompatibilityThreshold(float currentThreshold, std::size_t speciesCount,
                                                 const SpeciationConfig& config)
{
    float adjusted = currentThreshold;
    if (speciesCount < config.targetSpeciesMin)
    {
        adjusted -= config.compatibilityThresholdAdjustment;
    }
    else if (speciesCount > config.targetSpeciesMax)
    {
        adjusted += config.compatibilityThresholdAdjustment;
    }
    return std::clamp(adjusted, config.minimumCompatibilityThreshold, config.maximumCompatibilityThreshold);
}

} // namespace ai::neat
