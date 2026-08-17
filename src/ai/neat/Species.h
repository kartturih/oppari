#pragma once

#include <cstddef>
#include <vector>

#include "ai/neat/Genome.h"

namespace ai::neat
{

// Stable identifier for one Species, assigned monotonically by Speciator
// and never reused.
using SpeciesId = int;

class Speciator;

// One species, persistent across Speciator::speciate() calls: an ID, a
// representative Genome (used for the NEXT speciate() call), member
// indices (into the Genome vector from the MOST RECENT speciate() call),
// and historical bookkeeping (age/best-fitness/stagnation). Holds no
// fitness of its own and owns no Genomes, only indices. Every mutating
// operation is private to Speciator (declared a friend).
class Species
{
public:
    // Fresh history: age/generationsSinceImprovement == 0,
    // historicalBestFitness == 0, hasFitnessHistory() == false.
    Species(SpeciesId id, Genome representative);

    SpeciesId getId() const { return m_id; }
    const Genome& getRepresentative() const { return m_representative; }
    const std::vector<std::size_t>& getMemberIndices() const { return m_memberIndices; }
    std::size_t size() const { return m_memberIndices.size(); }
    bool empty() const { return m_memberIndices.empty(); }

    // Completed (recordGeneration()'d) generations this species has existed for.
    std::size_t getAge() const { return m_age; }

    // Highest raw member fitness ever seen; meaningless until hasFitnessHistory().
    float getHistoricalBestFitness() const { return m_historicalBestFitness; }

    // Consecutive completed generations without a new historical best.
    std::size_t getGenerationsSinceImprovement() const { return m_generationsSinceImprovement; }

    // True once getGenerationsSinceImprovement() reaches the stagnation limit.
    bool isStagnant() const { return m_stagnant; }

    bool hasFitnessHistory() const { return m_hasFitnessHistory; }

private:
    friend class Speciator;

    // Appends in ascending genome-index order (Speciator's calling contract).
    void addMember(std::size_t genomeIndex);

    // Clears this pass's members only; ID/representative/history untouched.
    void clearMembers();

    void setRepresentative(Genome representative);

    // If currentSpeciesBest beats the historical best (or none recorded
    // yet), resets the improvement counter and updates the best; otherwise
    // increments it. Then age += 1 and stagnant is recomputed.
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
