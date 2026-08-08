#include "ai/neat/Speciator.h"

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

std::vector<Species> Speciator::speciate(const std::vector<Genome>& genomes, const CompatibilityConfig& compatibilityConfig,
                                          const SpeciationConfig& speciationConfig)
{
    validateSpeciationConfig(speciationConfig);

    std::vector<Species> species;

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
        for (Species& candidate : species)
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
            // species stores creation order via push_back, which is
            // exactly ascending-SpeciesId order since m_nextSpeciesId only
            // ever increments -- no separate sort is needed.
            Species newSpecies(m_nextSpeciesId, genome);
            ++m_nextSpeciesId;
            newSpecies.addMember(genomeIndex);
            species.push_back(std::move(newSpecies));
        }
    }

    return species;
}

} // namespace ai::neat
