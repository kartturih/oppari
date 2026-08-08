#pragma once

namespace ai::neat
{

// Tunable probabilities/ranges for Genome mutation. Pure data -- it
// performs no validation and no randomness itself; GenomeMutator validates
// the fields relevant to whichever mutation it is about to perform
// (throwing std::invalid_argument on an invalid one) before using them.
//
// Weight mutation (Stage 9A) -- for each connection gene, in order:
//   1. With probability weightMutationProbability, the connection is
//      selected for mutation; otherwise it is left untouched.
//   2. If selected, with probability weightPerturbProbability the weight is
//      perturbed: newWeight = oldWeight + randomUniform(-perturbStrength,
//      +perturbStrength). Otherwise it is replaced outright: newWeight =
//      randomUniform(replacementWeightMin, replacementWeightMax).
// Perturbed weights are not clamped to any range.
//
// Add-connection structural mutation (Stage 9C) -- see
// GenomeMutator::mutateAddConnection() for the full algorithm:
//   1. With probability addConnectionProbability, a new connection is
//      attempted at all; otherwise nothing happens.
//   2. If attempted, up to addConnectionMaxAttempts random (source, target)
//      candidates are tried, falling back to a deterministic exhaustive
//      scan if none of them validate.
//   3. A newly added connection's initial weight is drawn uniformly from
//      [newConnectionWeightMin, newConnectionWeightMax].
//
// Add-node structural mutation (Stage 9D) -- see
// GenomeMutator::mutateAddNode() for the full algorithm:
//   1. With probability addNodeProbability, an existing enabled connection
//      is split into two; otherwise nothing happens.
//   2. If attempted, one enabled connection is selected (via a randomized
//      starting point over the eligible set) and split into
//      source -> newNode -> target, with newNode's identity and the two new
//      connections' innovation numbers coming from InnovationTracker.
struct MutationConfig
{
    float weightMutationProbability = 0.8f; // chance each connection is selected for mutation at all, in [0,1]
    float weightPerturbProbability = 0.9f;  // given selection, chance it's a perturbation rather than a replacement, in [0,1]
    float perturbStrength = 0.5f;           // perturbation delta is drawn uniformly from [-perturbStrength, +perturbStrength]
    float replacementWeightMin = -1.0f;     // inclusive lower bound for a replacement weight
    float replacementWeightMax = 1.0f;      // inclusive upper bound for a replacement weight

    float addConnectionProbability = 0.05f; // chance a new-connection mutation is attempted at all, in [0,1]
    float newConnectionWeightMin = -1.0f;   // inclusive lower bound for a newly added connection's initial weight
    float newConnectionWeightMax = 1.0f;    // inclusive upper bound for a newly added connection's initial weight
    int addConnectionMaxAttempts = 20;      // random (source, target) candidates tried before the deterministic fallback scan

    float addNodeProbability = 0.03f; // chance an add-node (connection-split) mutation is attempted at all, in [0,1]
};

} // namespace ai::neat
