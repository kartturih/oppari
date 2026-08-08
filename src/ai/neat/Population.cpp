#include "ai/neat/Population.h"

#include <algorithm>
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

Population::Population(const Genome& baseGenome, const simulation::Track& track, const simulation::CarParams& carParams,
                        const simulation::TrackDefinition& trackDefinition, Vector2 spawnPosition, float spawnHeading,
                        const PopulationConfig& populationConfig, const MutationConfig& mutationConfig,
                        const CrossoverConfig& crossoverConfig, const CompatibilityConfig& compatibilityConfig,
                        const SpeciationConfig& speciationConfig)
    : m_track(track)
    , m_carParams(carParams)
    , m_trackDefinition(trackDefinition)
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
    , m_currentSpeciesCount(0)
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
        m_individuals.emplace_back(std::move(genome), m_track, m_carParams, m_trackDefinition, m_spawnPosition,
                                    m_spawnHeading);
    }

    m_currentSpeciesCount = computeSpeciesCount();
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

std::size_t Population::computeSpeciesCount()
{
    std::vector<Genome> genomes;
    genomes.reserve(m_individuals.size());
    for (const Individual& individual : m_individuals)
    {
        genomes.push_back(individual.getGenome());
    }
    return m_speciator.speciate(genomes, m_compatibilityConfig, m_speciationConfig).size();
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

    // 1. Preserve final fitness values before anything else changes.
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

    // Species membership of the generation that just finished, purely for
    // statistics -- computed before anything about it is replaced. Never
    // consulted below: elitism/tournament/crossover/mutation operate
    // entirely on fitnessValues and Genomes, regardless of species.
    m_currentSpeciesCount = computeSpeciesCount();

    // 2. Elitism ranking: indices sorted by (higher fitness first, lower
    // original index as a deterministic tie-break).
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

    // 3. Construct every new Genome first -- entirely from the current
    // (soon to be replaced) generation's data. m_individuals is not
    // touched at any point during this construction, so parent selection
    // always reads the complete, unmodified old generation.
    std::vector<Genome> newGenomes;
    newGenomes.reserve(populationSize);

    for (std::size_t i = 0; i < m_populationConfig.eliteCount; ++i)
    {
        newGenomes.push_back(m_individuals[rankedIndices[i]].getGenome()); // copy, unmutated
    }

    while (newGenomes.size() < populationSize)
    {
        const std::size_t parentAIndex = tournamentSelect(fitnessValues);
        const std::size_t parentBIndex = tournamentSelect(fitnessValues);
        const Genome& parentA = m_individuals[parentAIndex].getGenome();
        const Genome& parentB = m_individuals[parentBIndex].getGenome();

        Genome child = m_crossover.crossover(parentA, fitnessValues[parentAIndex], parentB, fitnessValues[parentBIndex],
                                              m_crossoverConfig);

        m_mutator.mutateWeights(child, m_mutationConfig);
        m_mutator.mutateAddConnection(child, m_innovationTracker, m_mutationConfig);
        m_mutator.mutateAddNode(child, m_innovationTracker, m_mutationConfig);

        newGenomes.push_back(std::move(child));
    }

    // 4. Rebuild every Individual from the new Genomes, then replace the
    // population as one coherent operation.
    std::vector<Individual> newIndividuals;
    newIndividuals.reserve(populationSize);
    for (Genome& genome : newGenomes)
    {
        newIndividuals.emplace_back(std::move(genome), m_track, m_carParams, m_trackDefinition, m_spawnPosition,
                                     m_spawnHeading);
    }

    m_individuals = std::move(newIndividuals);
    ++m_generation;
}

} // namespace ai::neat
