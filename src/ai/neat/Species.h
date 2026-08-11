#pragma once

#include <cstddef>
#include <vector>

#include "ai/neat/Genome.h"

namespace ai::neat
{

// Stable identifier for one Species, assigned monotonically by Speciator
// and never reused during a Speciator's lifetime. Always non-negative.
using SpeciesId = int;

class Speciator;

// One species, persistent across Speciator::speciate() calls (Stage 17):
// an ID, a representative Genome used to test compatibility for the NEXT
// speciate() call, the indices (into the Genome vector passed to the MOST
// RECENT speciate() call) of every Genome currently assigned to it, and a
// small amount of historical bookkeeping (age/best-fitness/stagnation) that
// only Speciator -- which owns every Species and is the sole class able to
// mutate any of it -- ever updates. Species itself is still pure
// bookkeeping: it holds no fitness of its own (only a historical summary
// supplied by whoever evaluates its members), performs no reproduction or
// crossover, and never owns member Genomes -- only indices into a Genome
// vector someone else owns.
//
// Every mutating operation (membership, representative, fitness history) is
// private and reachable only from Speciator (declared a friend) -- nothing
// else can silently corrupt a Species's membership, representative, or
// history. Everything else about a Species is read-only from the outside.
class Species
{
public:
    // Creates a species with the given id and representative, and no
    // members yet. representative is stored by value (moved in), so later
    // changes to the Genome that produced it never affect this copy.
    //
    // A freshly created species' history starts in a documented "never
    // evaluated" state: age() == 0, historicalBestFitness() == 0,
    // generationsSinceImprovement() == 0, isStagnant() == false, and
    // hasFitnessHistory() == false (the flag that disambiguates this from a
    // species whose one real recorded best genuinely happened to be 0.0).
    Species(SpeciesId id, Genome representative);

    SpeciesId getId() const { return m_id; }
    const Genome& getRepresentative() const { return m_representative; }
    const std::vector<std::size_t>& getMemberIndices() const { return m_memberIndices; }
    std::size_t size() const { return m_memberIndices.size(); }
    bool empty() const { return m_memberIndices.empty(); }

    // Number of completed (evaluated, history-recorded) generations this
    // species has existed for -- incremented exactly once per
    // recordGeneration() call, never by speciate() membership assignment
    // alone. 0 for a species that has not yet completed a single evaluated
    // generation (including one still being evaluated for the very first
    // time).
    std::size_t getAge() const { return m_age; }

    // The highest RAW member fitness ever observed for this species across
    // every recordGeneration() call, undefined/meaningless (always exactly
    // 0) until hasFitnessHistory() is true.
    float getHistoricalBestFitness() const { return m_historicalBestFitness; }

    // Consecutive completed generations, up to and including the most
    // recent recordGeneration() call, without a new historical best.
    std::size_t getGenerationsSinceImprovement() const { return m_generationsSinceImprovement; }

    // True exactly when getGenerationsSinceImprovement() >= the
    // speciesStagnationLimit most recently passed to recordGeneration().
    bool isStagnant() const { return m_stagnant; }

    // False until recordGeneration() has been called at least once --
    // disambiguates "never evaluated" from "evaluated, and its one real
    // historical best genuinely is 0.0".
    bool hasFitnessHistory() const { return m_hasFitnessHistory; }

private:
    friend class Speciator;

    // Appends genomeIndex to the member list, in the order it is called.
    // Speciator always calls this in ascending genome-index order (it
    // processes genomes strictly in vector-index order), so member indices
    // stay input-order sorted -- in particular, getMemberIndices().front()
    // is always this pass's lowest member index.
    void addMember(std::size_t genomeIndex);

    // Clears this pass's member list only -- id, representative, and every
    // history field are left untouched. Called by Speciator once per
    // speciate() call, before any genome is (re-)assigned, so a persistent
    // species starts each pass with no members but keeps everything else
    // that makes it "the same species" across generations.
    void clearMembers();

    // Replaces the representative used for future compatibility checks.
    // Speciator only ever calls this once per surviving species per
    // speciate() call, strictly after every genome in that call has already
    // been assigned using the OLD representative -- see Speciator.cpp for
    // the exact ordering guarantee.
    void setRepresentative(Genome representative);

    // Records the outcome of one fully evaluated generation for this
    // species, from currentSpeciesBest (the highest RAW fitness among this
    // pass's members) and speciesStagnationLimit:
    //   - if this is the first ever call (hasFitnessHistory() was false):
    //     historicalBestFitness = currentSpeciesBest, generationsSinceImprovement = 0.
    //   - else if currentSpeciesBest is STRICTLY greater than the existing
    //     historicalBestFitness (exact >, no epsilon): historicalBestFitness
    //     = currentSpeciesBest, generationsSinceImprovement = 0.
    //   - else (currentSpeciesBest <= historicalBestFitness, including
    //     exactly equal): generationsSinceImprovement += 1.
    // Then, unconditionally: age += 1, hasFitnessHistory = true, and
    // stagnant = (generationsSinceImprovement >= speciesStagnationLimit).
    // Called by Speciator exactly once per species per completed generation
    // (via Speciator::updateFitnessHistory()), never by anything else.
    void recordGeneration(float currentSpeciesBest, std::size_t speciesStagnationLimit);

    SpeciesId m_id;
    Genome m_representative;
    std::vector<std::size_t> m_memberIndices;

    std::size_t m_age = 0;
    float m_historicalBestFitness = 0.0f;
    std::size_t m_generationsSinceImprovement = 0;
    bool m_stagnant = false;
    bool m_hasFitnessHistory = false;
};

} // namespace ai::neat
