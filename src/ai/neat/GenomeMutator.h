#pragma once

#include <cstdint>
#include <random>

#include "ai/neat/Genome.h"
#include "ai/neat/InnovationTracker.h"
#include "ai/neat/MutationConfig.h"

namespace ai::neat
{

// Deterministically mutates a Genome via one owned std::mt19937, seeded once
// at construction (never random_device/rand()/time-based). Same seed + same
// call sequence on the same starting state always reproduces the same result.
class GenomeMutator
{
public:
    explicit GenomeMutator(std::uint32_t seed);

    // For each connection gene, in storage order: selected with probability
    // weightMutationProbability, then either perturbed by a uniform delta in
    // [-perturbStrength, perturbStrength] (not clamped) or replaced outright
    // from [replacementWeightMin, replacementWeightMax], chosen by
    // weightPerturbProbability. Never adds/removes genes.
    // Throws std::invalid_argument on an invalid config.
    void mutateWeights(Genome& genome, const MutationConfig& config);

    // Attempts to add one new directed connection (source: Input/Bias/Hidden,
    // target: Hidden/Output, no self-loops, no duplicate directed pair, no
    // cycle among enabled connections). Draws once against
    // addConnectionProbability; if selected, tries up to
    // addConnectionMaxAttempts random candidates, falling back to a
    // deterministic exhaustive scan (source then target, ascending ID) if
    // none validate. Returns false (genome/tracker untouched) if not
    // selected or no valid pair exists.
    // Throws std::invalid_argument on an invalid config.
    bool mutateAddConnection(Genome& genome, InnovationTracker& innovationTracker, const MutationConfig& config);

    // Attempts to split one existing enabled connection into
    // source -> newHiddenNode -> target, reusing InnovationTracker's
    // historical marking for that split if it already exists (so the same
    // structural mutation gets the same node ID/innovation numbers across
    // genomes). Draws once against addNodeProbability; if selected, scans
    // eligible connections from a random starting point for one not already
    // split in this genome. Disables the original connection (never removes
    // it) and adds the two new ones. Returns false (genome/tracker
    // untouched) if not selected or every candidate is already split here.
    // Throws std::invalid_argument on an invalid config or a conflicting
    // (inconsistent) historical marking.
    bool mutateAddNode(Genome& genome, InnovationTracker& innovationTracker, const MutationConfig& config);

private:
    std::mt19937 m_rng;
};

} // namespace ai::neat
