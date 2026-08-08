#pragma once

#include <cstddef>
#include <vector>

#include "ai/neat/Genome.h"

namespace ai::neat
{

// Stable identifier for one Species, assigned monotonically by Speciator
// and never reused during a Speciator's lifetime. Always non-negative.
using SpeciesId = int;

// One grouping of Genomes for a single Speciator::speciate() pass: an ID, a
// fixed representative Genome (owned by value -- a copy of whichever
// Genome founded this species), and the indices (into the Genome vector
// passed to speciate()) of every Genome assigned to it. Species is pure
// membership bookkeeping: it holds no fitness, performs no reproduction or
// crossover, and -- in this stage -- never changes its representative once
// created.
//
// Member indices can only be appended through addMember(), which is
// private and reachable only from Speciator (declared a friend) -- the
// "narrow internal API" the membership list is deliberately restricted to,
// so nothing else can silently corrupt a Species's membership after
// Speciator has produced it. Everything else about a Species is read-only
// from the outside.
class Species
{
public:
    // Creates a species with the given id and representative, and no
    // members yet -- Speciator immediately adds the founding Genome's own
    // index as the first member via addMember(). representative is stored
    // by value (moved in), so later changes to the Genome that produced it
    // never affect this copy.
    Species(SpeciesId id, Genome representative);

    SpeciesId getId() const { return m_id; }
    const Genome& getRepresentative() const { return m_representative; }
    const std::vector<std::size_t>& getMemberIndices() const { return m_memberIndices; }
    std::size_t size() const { return m_memberIndices.size(); }
    bool empty() const { return m_memberIndices.empty(); }

private:
    friend class Speciator;

    // Appends genomeIndex to the member list, in the order it is called.
    // Speciator always calls this in ascending genome-index order (it
    // processes genomes strictly in vector-index order), so member
    // indices stay input-order sorted.
    void addMember(std::size_t genomeIndex);

    SpeciesId m_id;
    Genome m_representative;
    std::vector<std::size_t> m_memberIndices;
};

} // namespace ai::neat
