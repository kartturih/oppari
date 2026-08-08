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

} // namespace ai::neat
