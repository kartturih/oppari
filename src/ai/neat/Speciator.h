#pragma once

#include <vector>

#include "ai/neat/CompatibilityConfig.h"
#include "ai/neat/Genome.h"
#include "ai/neat/SpeciationConfig.h"
#include "ai/neat/Species.h"

namespace ai::neat
{

// Deterministically groups a vector of Genomes into Species, using the
// existing ai::neat::compatibilityDistance() (Stage 11) as its sole notion
// of similarity. Speciator only manages membership assignment and
// representatives for one speciate() call -- it does not reproduce, select
// parents, track fitness, or persist species identity across calls beyond
// the monotonic SpeciesId counter described below.
//
// Owns the next-SpeciesId counter, starting at 0 -- the only state
// Speciator carries between speciate() calls. IDs allocated by one call are
// never reused by a later call on the same instance, but the Species
// objects themselves are not retained across calls: each speciate() call
// starts with zero species and builds a brand-new std::vector<Species> from
// scratch, even though the ID numbers keep counting up from wherever the
// previous call left off. Persistent cross-generation species lineage
// (reusing a Species object across calls, carrying its existing
// representative and history forward, choosing a new representative when a
// species survives) is out of scope for this stage.
//
// No RNG is used anywhere in this class.
class Speciator
{
public:
    Speciator();

    // Groups genomes into species by comparing each genome, strictly in
    // vector-index order, against the representatives of species already
    // assigned so far (species are evaluated in ascending SpeciesId order,
    // which is also the order they appear in the returned vector): the
    // genome joins the FIRST species whose representative is within
    // speciationConfig.compatibilityThreshold of it (distance <=
    // threshold, so a distance exactly equal to the threshold qualifies)
    // -- never the nearest one, even if a later-evaluated species would be
    // a strictly closer match. This first-match rule is intentional: it
    // keeps assignment a simple, deterministic function of input order, at
    // the cost of not always producing the tightest possible grouping. If
    // no existing species qualifies, a new Species is created: its
    // representative is a copy of this genome, it receives the next
    // monotonically increasing SpeciesId, and this genome's index becomes
    // its first (and, so far, only) member.
    //
    // Before a genome can either join an existing species or found a new
    // one, it is validated unconditionally -- regardless of whether any
    // species exist yet to compare it against: first genome.validate()
    // (structural correctness -- unique node IDs, connection endpoints
    // that exist, no duplicate directed connections), then a
    // self-comparison compatibilityDistance(genome, genome,
    // compatibilityConfig) purely to reuse that function's own
    // innovation-uniqueness check (it throws if genome contains two
    // different ConnectionGenes with the same innovation number) without
    // duplicating that logic here. This runs before this genome is
    // compared against any existing representative and before any
    // SpeciesId is allocated for it, so a genome that fails validation is
    // never partially added as a member and never causes a SpeciesId to be
    // consumed -- the exception propagates straight out of speciate(),
    // leaving only whatever species earlier genomes had already validly
    // produced. Comparisons against existing representatives also go
    // through ai::neat::compatibilityDistance(genome, representative,
    // compatibilityConfig), so any exception it throws there (invalid
    // compatibilityConfig, conflicting matching endpoints between two
    // genomes, ...) likewise propagates unchanged, and no genome is
    // silently skipped.
    //
    // Throws std::invalid_argument if speciationConfig.compatibilityThreshold
    // is non-finite or negative. This check runs before any genome is
    // processed or any SpeciesId is allocated.
    //
    // An empty genomes vector returns an empty species vector without
    // allocating any SpeciesId.
    //
    // Neither genomes, compatibilityConfig, nor speciationConfig is ever
    // modified; every Species's representative is an independent copy that
    // stays fixed for the lifetime of this call. The returned vector is
    // sorted ascending by SpeciesId (equivalently, species-creation order),
    // and each Species's member indices are stored in the ascending
    // genome-index order they were assigned in.
    //
    // Because assignment is first-match and strictly index-ordered, the
    // order of genomes in the input vector can change the resulting
    // grouping (which genome ends up founding a species and fixing its
    // representative, and which later species a borderline genome ends up
    // joining) even for the exact same underlying set of genomes -- this is
    // intentional, documented behavior, not nondeterminism: the same input
    // order, against the same Speciator state and configs, always produces
    // the same result.
    std::vector<Species> speciate(const std::vector<Genome>& genomes, const CompatibilityConfig& compatibilityConfig,
                                   const SpeciationConfig& speciationConfig);

private:
    SpeciesId m_nextSpeciesId;
};

} // namespace ai::neat
