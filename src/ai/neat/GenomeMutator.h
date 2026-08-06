#pragma once

#include <cstdint>
#include <random>

#include "ai/neat/Genome.h"
#include "ai/neat/MutationConfig.h"

namespace ai::neat
{

// Deterministically mutates the connection weights of a Genome. Owns its
// own std::mt19937 RNG, seeded exactly once at construction -- never
// std::random_device, never a global/static generator, never rand(), never
// time-based. Repeated calls to mutateWeights() continue drawing from that
// same generator, so the same seed driven through the same sequence of
// calls on the same starting genome(s) always reproduces the same results;
// a different seed may produce different results.
//
// mutateWeights() only ever changes ConnectionGene weights, via
// ConnectionGene::setWeight(). It never adds or removes nodes or
// connections, and never changes node IDs/types, connection source/target
// IDs, innovation numbers, or enabled state -- both enabled and disabled
// connections are equally eligible for weight mutation, since a disabled
// gene may later be re-enabled by future structural mutation and should not
// be left stale.
//
// This stage (9A) implements weight mutation only. No structural mutation,
// InnovationTracker, crossover, species, or population logic exists here or
// anywhere else in the codebase yet.
class GenomeMutator
{
public:
    explicit GenomeMutator(std::uint32_t seed);

    // Throws std::invalid_argument if config is invalid (see
    // MutationConfig.h: probabilities outside [0,1], a negative
    // perturbStrength, an inverted replacement range, or any non-finite
    // field). Otherwise, for each connection gene in genome, in storage
    // order:
    //   - with probability config.weightMutationProbability, the
    //     connection is selected;
    //   - if selected, with probability config.weightPerturbProbability its
    //     weight is perturbed (oldWeight + a value drawn uniformly from
    //     [-perturbStrength, +perturbStrength], not clamped), otherwise it
    //     is replaced outright with a value drawn uniformly from
    //     [replacementWeightMin, replacementWeightMax].
    // A genome with no connections is accepted and left unchanged.
    void mutateWeights(Genome& genome, const MutationConfig& config);

private:
    std::mt19937 m_rng;
};

} // namespace ai::neat
