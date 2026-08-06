#pragma once

namespace ai::neat
{

// Tunable probabilities/ranges for connection-weight mutation. Pure data --
// it performs no validation and no randomness itself; GenomeMutator
// validates a MutationConfig (throwing std::invalid_argument on an invalid
// one) before using it.
//
// For each connection gene, in order:
//   1. With probability weightMutationProbability, the connection is
//      selected for mutation; otherwise it is left untouched.
//   2. If selected, with probability weightPerturbProbability the weight is
//      perturbed: newWeight = oldWeight + randomUniform(-perturbStrength,
//      +perturbStrength). Otherwise it is replaced outright: newWeight =
//      randomUniform(replacementWeightMin, replacementWeightMax).
//
// Perturbed weights are not clamped to any range.
struct MutationConfig
{
    float weightMutationProbability = 0.8f; // chance each connection is selected for mutation at all, in [0,1]
    float weightPerturbProbability = 0.9f;  // given selection, chance it's a perturbation rather than a replacement, in [0,1]
    float perturbStrength = 0.5f;           // perturbation delta is drawn uniformly from [-perturbStrength, +perturbStrength]
    float replacementWeightMin = -1.0f;     // inclusive lower bound for a replacement weight
    float replacementWeightMax = 1.0f;      // inclusive upper bound for a replacement weight
};

} // namespace ai::neat
