#pragma once

namespace ai::neat
{

// Tunable probabilities/ranges for Genome mutation, validated by
// GenomeMutator before use.
//
// Weight mutation: each connection gene is independently selected with
// probability weightMutationProbability, then either perturbed (delta in
// [-perturbStrength, perturbStrength]) or replaced outright, with
// probability weightPerturbProbability choosing which.
//
// Add-connection: with probability addConnectionProbability, tries up to
// addConnectionMaxAttempts random candidates before an exhaustive fallback
// scan; new weight drawn from [newConnectionWeightMin, newConnectionWeightMax].
//
// Add-node: with probability addNodeProbability, splits one existing
// enabled connection via InnovationTracker (see GenomeMutator::mutateAddNode()).
struct MutationConfig
{
    float weightMutationProbability = 0.8f; // chance a connection is selected for mutation, [0,1]
    float weightPerturbProbability = 0.9f;  // given selection, chance it's a perturb vs. replace, [0,1]
    float perturbStrength = 0.5f;           // max perturbation delta
    float replacementWeightMin = -1.0f;
    float replacementWeightMax = 1.0f;

    float addConnectionProbability = 0.05f;
    float newConnectionWeightMin = -1.0f;
    float newConnectionWeightMax = 1.0f;
    int addConnectionMaxAttempts = 20;

    float addNodeProbability = 0.03f;
};

} // namespace ai::neat
