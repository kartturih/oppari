#pragma once

#include <cstddef>
#include <vector>

#include "ai/neat/CompatibilityConfig.h"
#include "ai/neat/Genome.h"
#include "ai/neat/SpeciationConfig.h"
#include "ai/neat/Species.h"

namespace ai::neat
{

// Deterministically groups Genomes into persistent Species using
// compatibilityDistance() as the sole similarity measure. Does not
// reproduce, select parents, or compute fitness. A SpeciesId, once
// allocated, is never reused even after that species goes extinct. No RNG
// anywhere -- representative reselection is a deterministic, index-based rule.
class Speciator
{
public:
    Speciator();

    // One speciation pass; returns a reference to the owned, now-updated
    // species list (valid until the next mutating call).
    //
    // Per call: every species' current membership is cleared (representative
    // and history untouched). Each genome, in vector-index order, is
    // validated then compared (in ascending SpeciesId order) against each
    // existing species' OLD representative, joining the FIRST one within
    // compatibilityThreshold -- never the nearest. If none qualifies, a new
    // species is founded. Afterward, any species left with zero members is
    // removed (extinct). Finally, every surviving species' representative is
    // replaced with its own lowest-index member from THIS pass -- so this
    // pass's own joins always compared against the OLD representative.
    //
    // Throws std::invalid_argument if compatibilityThreshold is invalid. An
    // empty genomes vector empties m_species (all treated as extinct).
    // Never modifies genomes/compatibilityConfig/speciationConfig.
    const std::vector<Species>& speciate(const std::vector<Genome>& genomes, const CompatibilityConfig& compatibilityConfig,
                                          const SpeciationConfig& speciationConfig);

    // Species from the most recent speciate() call (empty if never called).
    const std::vector<Species>& getSpecies() const { return m_species; }

    // For each species, records this generation's best raw fitness among
    // rawFitnessByGenomeIndex's members (see Species::recordGeneration()).
    // rawFitnessByGenomeIndex must align with the genomes passed to the most
    // recent speciate() call; Population calls this once per completed
    // generation, before the next speciate() call.
    void updateFitnessHistory(const std::vector<float>& rawFitnessByGenomeIndex, std::size_t speciesStagnationLimit);

private:
    SpeciesId m_nextSpeciesId;
    std::vector<Species> m_species;
};

} // namespace ai::neat
