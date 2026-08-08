#pragma once

namespace ai::neat
{

// Tunable probabilities for GenomeCrossover. Pure data -- it performs no
// validation and no randomness itself; GenomeCrossover validates both
// fields (throwing std::invalid_argument on an invalid one) before using
// them.
//
// matchingGeneChooseParentAProbability -- for each matching connection gene
// (same innovation number present in both parents), the probability that
// the child inherits parent A's structural/weight copy of that gene rather
// than parent B's copy. Drawn independently for every matching gene.
//
// disabledGeneRemainDisabledProbability -- for a matching connection gene
// where at least one parent's copy is disabled, the probability that the
// child's copy is also disabled. This decision is independent of which
// parent's structural/weight copy was chosen via the probability above; if
// both parents' copies are enabled, the child's copy is always enabled and
// this field is not consulted at all for that gene.
struct CrossoverConfig
{
    float matchingGeneChooseParentAProbability = 0.5f;
    float disabledGeneRemainDisabledProbability = 0.75f;
};

} // namespace ai::neat
