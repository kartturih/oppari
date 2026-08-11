#pragma once

#include <cstddef>
#include <vector>

#include "ai/neat/CompatibilityConfig.h"
#include "ai/neat/Genome.h"
#include "ai/neat/SpeciationConfig.h"
#include "ai/neat/Species.h"

namespace ai::neat
{

// Deterministically groups a vector of Genomes into Species, using the
// existing ai::neat::compatibilityDistance() (Stage 11) as its sole notion
// of similarity. Speciator manages species membership, representatives, and
// (as of Stage 17) persistent per-species identity/history -- it still does
// not reproduce, select parents, or compute fitness itself.
//
// Owns, for its entire lifetime:
//   - the next-SpeciesId counter, starting at 0, monotonically increasing --
//     an ID allocated once is never reused, even after the Species that held
//     it goes extinct;
//   - the persistent Species collection itself (m_species), carried forward
//     across every speciate() call. Unlike before Stage 17, a Species
//     object is no longer discarded and rebuilt from scratch on every call:
//     a species whose representative a new generation's genomes remain
//     compatible with keeps its SpeciesId, and keeps accumulating age/
//     fitness-history/stagnation state, across as many generations as it
//     keeps matching. A species that goes extinct (zero members after an
//     assignment pass) is erased from m_species -- see speciate()'s own doc
//     comment for the full per-call lifecycle.
//
// No RNG is used anywhere in this class -- representative reselection
// (see speciate()) is a deterministic, purely index-based rule.
class Speciator
{
public:
    Speciator();

    // Runs one persistent speciation pass and returns a const reference to
    // Speciator's own, now-updated m_species (owned by this Speciator, and
    // only valid until the next mutating call on it) -- callers that need a
    // snapshot independent of future calls must copy it explicitly.
    //
    // Exact per-call lifecycle:
    //   1. Every existing persistent species' CURRENT member list is
    //      cleared (Species::clearMembers()) -- its SpeciesId, its
    //      representative (inherited from the previous call, or from its
    //      own founding genome if it has none yet), and its age/history/
    //      stagnation state are all left untouched.
    //   2. Each genome in `genomes`, strictly in vector-index order, is
    //      first validated exactly as before Stage 17 (genome.validate(),
    //      then a self-comparison through compatibilityDistance() purely to
    //      reuse its innovation-uniqueness check) -- an exception here
    //      propagates immediately, before this genome joins/founds anything
    //      and before any SpeciesId is allocated for it.
    //   3. The genome is then compared, in ascending SpeciesId order (which
    //      is also m_species's storage order -- see below), against each
    //      existing persistent species' representative (still the OLD one
    //      from before this call -- see point 5). It joins the FIRST
    //      species within speciationConfig.compatibilityThreshold (distance
    //      <= threshold) -- never the nearest one. If none qualifies, a new
    //      Species is created: representative = a copy of this genome,
    //      SpeciesId = the next monotonically increasing one, and this
    //      genome's index becomes its first member. This first-match rule
    //      is unchanged from before Stage 17.
    //   4. Once every genome has been assigned, any species left with zero
    //      members (extinct -- nothing in `genomes` this pass was
    //      compatible with its old representative) is removed from
    //      m_species. Its SpeciesId is never reused: m_nextSpeciesId only
    //      ever increases, regardless of removals.
    //   5. Finally, EVERY surviving species' representative is replaced
    //      (Species::setRepresentative()) with a copy of its own lowest-
    //      index current member (getMemberIndices().front(), which is
    //      exactly the lowest index since members are always appended in
    //      ascending genome-index order) -- deterministic, no RNG, and
    //      never based on fitness (Speciator has no fitness to consult).
    //      This happens strictly AFTER every membership decision in this
    //      call, so this call's own assignments (step 3) always compared
    //      against the OLD representative; the new representative only
    //      takes effect starting with the NEXT speciate() call. For a
    //      species founded during THIS call (step 3), its founding genome
    //      is necessarily already its own lowest-index member (every other
    //      member, if any, was processed later in index order), so this
    //      step is a no-op for it.
    //
    // m_species remains sorted ascending by SpeciesId across every call:
    // surviving species keep their relative order (step 4 preserves order),
    // and newly founded species (step 3) always receive IDs higher than
    // every existing one, so they are correctly appended at the end.
    //
    // Throws std::invalid_argument if speciationConfig.compatibilityThreshold
    // is non-finite or negative -- checked before any genome is processed,
    // any species mutated, or any SpeciesId allocated. An empty genomes
    // vector clears every existing species' members, then removes all of
    // them as extinct (m_species becomes empty), without allocating any new
    // SpeciesId.
    //
    // Neither genomes, compatibilityConfig, nor speciationConfig is ever
    // modified.
    const std::vector<Species>& speciate(const std::vector<Genome>& genomes, const CompatibilityConfig& compatibilityConfig,
                                          const SpeciationConfig& speciationConfig);

    // The species produced by the most recent speciate() call (or empty, if
    // speciate() has never been called) -- the same reference speciate()
    // itself returns, exposed so callers can re-read it without re-running
    // speciation.
    const std::vector<Species>& getSpecies() const { return m_species; }

    // Updates every currently-tracked (post most-recent-speciate()) species'
    // persistent fitness history from this generation's raw fitness values:
    // for each species, currentSpeciesBest = the maximum of
    // rawFitnessByGenomeIndex[i] over every member index i the most recent
    // speciate() call assigned to that species, then
    // Species::recordGeneration(currentSpeciesBest, speciesStagnationLimit)
    // -- see its own doc comment for the exact history/stagnation rule.
    //
    // rawFitnessByGenomeIndex must be indexed exactly like the genomes
    // vector passed to the most recent speciate() call (same size, same
    // order) -- the caller (Population) is responsible for calling this
    // exactly once per fully evaluated generation, after that generation's
    // final raw fitness values are known, and before the next speciate()
    // call. Speciator itself never computes or stores raw fitness beyond
    // this one pass -- only each species' resulting historical summary.
    void updateFitnessHistory(const std::vector<float>& rawFitnessByGenomeIndex, std::size_t speciesStagnationLimit);

private:
    SpeciesId m_nextSpeciesId;
    std::vector<Species> m_species;
};

} // namespace ai::neat
