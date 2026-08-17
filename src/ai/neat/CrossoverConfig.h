#pragma once

namespace ai::neat
{

// Tunable probabilities for GenomeCrossover, validated by GenomeCrossover
// before use.
struct CrossoverConfig
{
    // Chance a matching gene (same innovation number in both parents)
    // inherits parent A's copy rather than parent B's.
    float matchingGeneChooseParentAProbability = 0.5f;

    // If either parent's copy of a matching gene is disabled, chance the
    // child's copy stays disabled too. Ignored if both copies are enabled.
    float disabledGeneRemainDisabledProbability = 0.75f;
};

} // namespace ai::neat
