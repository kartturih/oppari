#include "ai/neat/Speciator.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>

#include "ai/neat/CompatibilityDistance.h"

namespace ai::neat
{

namespace
{

void validateSpeciationConfig(const SpeciationConfig& config)
{
    if (!std::isfinite(config.compatibilityThreshold) || config.compatibilityThreshold < 0.0f)
    {
        throw std::invalid_argument("SpeciationConfig: compatibilityThreshold must be finite and non-negative");
    }
}

} // namespace

Speciator::Speciator() : m_nextSpeciesId(0)
{
}

const std::vector<Species>& Speciator::speciate(const std::vector<Genome>& genomes,
                                                  const CompatibilityConfig& compatibilityConfig,
                                                  const SpeciationConfig& speciationConfig)
{
    validateSpeciationConfig(speciationConfig);

    // 1. Clear this pass's membership from every persistent species. Its
    // SpeciesId, representative (still the OLD one, inherited from the
    // previous call), and age/history/stagnation state are untouched.
    for (Species& species : m_species)
    {
        species.clearMembers();
    }

    // 2 & 3. Assign every genome, strictly in vector-index order, to the
    // first existing persistent species (checked in ascending SpeciesId
    // order -- m_species is always maintained in that order) whose OLD
    // representative it is compatible with; otherwise found a new species.
    for (std::size_t genomeIndex = 0; genomeIndex < genomes.size(); ++genomeIndex)
    {
        const Genome& genome = genomes[genomeIndex];

        // Validate this genome before it can join or found any species --
        // reuses Genome::validate() for structural correctness, plus a
        // self-comparison through compatibilityDistance() purely to reuse
        // its own innovation-uniqueness check (it throws on two different
        // ConnectionGenes sharing an innovation number) rather than
        // duplicating that logic here. Runs unconditionally, before any
        // comparison against an existing representative and before any
        // SpeciesId is allocated, so a genome that fails validation can
        // never end up partially added to a species or consume a
        // SpeciesId -- the exception simply propagates out of speciate().
        genome.validate();
        (void)compatibilityDistance(genome, genome, compatibilityConfig);

        bool joined = false;
        for (Species& candidate : m_species)
        {
            const float distance = compatibilityDistance(genome, candidate.getRepresentative(), compatibilityConfig);
            if (distance <= speciationConfig.compatibilityThreshold)
            {
                candidate.addMember(genomeIndex);
                joined = true;
                break;
            }
        }

        if (!joined)
        {
            // m_species stores creation order via push_back, which is
            // exactly ascending-SpeciesId order since m_nextSpeciesId only
            // ever increments -- appending here always keeps the whole
            // vector sorted ascending by SpeciesId, even across calls.
            Species newSpecies(m_nextSpeciesId, genome);
            ++m_nextSpeciesId;
            newSpecies.addMember(genomeIndex);
            m_species.push_back(std::move(newSpecies));
        }
    }

    // 4. Remove extinct species (zero members after this pass). Erasing
    // preserves the relative order of survivors, so ascending-SpeciesId
    // order is preserved. The removed SpeciesId is never reused --
    // m_nextSpeciesId is never rolled back.
    m_species.erase(std::remove_if(m_species.begin(), m_species.end(), [](const Species& species) { return species.empty(); }),
                     m_species.end());

    // 5. Reselect every surviving species' representative from THIS pass's
    // own members (lowest genome index, no RNG) -- strictly after every
    // membership decision above, so this pass's own assignments always
    // compared against the OLD representative. The new representative only
    // takes effect starting with the next speciate() call. Member indices
    // are always ascending (Species::addMember()'s documented invariant),
    // so front() is exactly the lowest index.
    for (Species& species : m_species)
    {
        species.setRepresentative(genomes[species.getMemberIndices().front()]);
    }

    return m_species;
}

void Speciator::updateFitnessHistory(const std::vector<float>& rawFitnessByGenomeIndex, std::size_t speciesStagnationLimit)
{
    for (Species& species : m_species)
    {
        const std::vector<std::size_t>& members = species.getMemberIndices();

        float currentSpeciesBest = rawFitnessByGenomeIndex[members.front()];
        for (std::size_t memberIndex : members)
        {
            currentSpeciesBest = std::max(currentSpeciesBest, rawFitnessByGenomeIndex[memberIndex]);
        }

        species.recordGeneration(currentSpeciesBest, speciesStagnationLimit);
    }
}

} // namespace ai::neat
