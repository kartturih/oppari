#include "ai/neat/GenomeMutator.h"

#include <cmath>
#include <stdexcept>

namespace ai::neat
{

namespace
{

bool isProbability(float value)
{
    return std::isfinite(value) && value >= 0.0f && value <= 1.0f;
}

void validateConfig(const MutationConfig& config)
{
    if (!isProbability(config.weightMutationProbability))
    {
        throw std::invalid_argument("MutationConfig: weightMutationProbability must be finite and within [0,1]");
    }
    if (!isProbability(config.weightPerturbProbability))
    {
        throw std::invalid_argument("MutationConfig: weightPerturbProbability must be finite and within [0,1]");
    }
    if (!std::isfinite(config.perturbStrength) || config.perturbStrength < 0.0f)
    {
        throw std::invalid_argument("MutationConfig: perturbStrength must be finite and non-negative");
    }
    if (!std::isfinite(config.replacementWeightMin) || !std::isfinite(config.replacementWeightMax))
    {
        throw std::invalid_argument("MutationConfig: replacement weight bounds must be finite");
    }
    if (config.replacementWeightMin > config.replacementWeightMax)
    {
        throw std::invalid_argument("MutationConfig: replacementWeightMin must not exceed replacementWeightMax");
    }
}

} // namespace

GenomeMutator::GenomeMutator(std::uint32_t seed) : m_rng(seed)
{
}

void GenomeMutator::mutateWeights(Genome& genome, const MutationConfig& config)
{
    validateConfig(config);

    // Four independent draws per candidate connection (selection,
    // perturb-vs-replace choice, and one of two magnitude draws), all from
    // the same owned generator, in a fixed order -- this is what makes the
    // result a pure, repeatable function of (seed, genome, config).
    std::uniform_real_distribution<float> selectForMutation(0.0f, 1.0f);
    std::uniform_real_distribution<float> selectPerturbOverReplace(0.0f, 1.0f);
    std::uniform_real_distribution<float> perturbDelta(-config.perturbStrength, config.perturbStrength);
    std::uniform_real_distribution<float> replacementWeight(config.replacementWeightMin, config.replacementWeightMax);

    for (ConnectionGene& connection : genome.mutableConnections())
    {
        if (selectForMutation(m_rng) >= config.weightMutationProbability)
        {
            continue;
        }

        if (selectPerturbOverReplace(m_rng) < config.weightPerturbProbability)
        {
            connection.setWeight(connection.getWeight() + perturbDelta(m_rng));
        }
        else
        {
            connection.setWeight(replacementWeight(m_rng));
        }
    }
}

} // namespace ai::neat
