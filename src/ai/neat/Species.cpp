#include "ai/neat/Species.h"

#include <utility>

namespace ai::neat
{

Species::Species(SpeciesId id, Genome representative) : m_id(id), m_representative(std::move(representative))
{
}

void Species::addMember(std::size_t genomeIndex)
{
    m_memberIndices.push_back(genomeIndex);
}

void Species::clearMembers()
{
    m_memberIndices.clear();
}

void Species::setRepresentative(Genome representative)
{
    m_representative = std::move(representative);
}

void Species::recordGeneration(float currentSpeciesBest, std::size_t speciesStagnationLimit)
{
    if (!m_hasFitnessHistory)
    {
        m_historicalBestFitness = currentSpeciesBest;
        m_generationsSinceImprovement = 0;
        m_hasFitnessHistory = true;
    }
    else if (currentSpeciesBest > m_historicalBestFitness)
    {
        m_historicalBestFitness = currentSpeciesBest;
        m_generationsSinceImprovement = 0;
    }
    else
    {
        ++m_generationsSinceImprovement;
    }

    ++m_age;
    m_stagnant = m_generationsSinceImprovement >= speciesStagnationLimit;
}

} // namespace ai::neat
