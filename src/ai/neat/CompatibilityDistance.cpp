#include "ai/neat/CompatibilityDistance.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

namespace ai::neat
{

namespace
{

void validateCompatibilityConfig(const CompatibilityConfig& config)
{
    if (!std::isfinite(config.excessCoefficient) || config.excessCoefficient < 0.0f)
    {
        throw std::invalid_argument("CompatibilityConfig: excessCoefficient must be finite and non-negative");
    }
    if (!std::isfinite(config.disjointCoefficient) || config.disjointCoefficient < 0.0f)
    {
        throw std::invalid_argument("CompatibilityConfig: disjointCoefficient must be finite and non-negative");
    }
    if (!std::isfinite(config.weightDifferenceCoefficient) || config.weightDifferenceCoefficient < 0.0f)
    {
        throw std::invalid_argument("CompatibilityConfig: weightDifferenceCoefficient must be finite and non-negative");
    }
    if (config.smallGenomeNormalizationThreshold == 0)
    {
        throw std::invalid_argument("CompatibilityConfig: smallGenomeNormalizationThreshold must be positive");
    }
}

// Maps each connection gene's innovation number to itself, throwing if the
// same genome uses the same innovation number twice -- Genome::validate()
// checks structural endpoints and duplicate (source, target) pairs, but not
// innovation-number uniqueness, so this must be checked explicitly before
// innovation numbers can be trusted as an alignment key.
std::map<InnovationNumber, const ConnectionGene*> buildInnovationMap(const Genome& genome, bool isGenomeA)
{
    std::map<InnovationNumber, const ConnectionGene*> result;
    for (const ConnectionGene& connection : genome.connections())
    {
        const auto insertion = result.emplace(connection.getInnovationNumber(), &connection);
        if (!insertion.second)
        {
            throw std::invalid_argument(
                isGenomeA ? "compatibilityDistance: genome A contains duplicate connection innovation numbers"
                          : "compatibilityDistance: genome B contains duplicate connection innovation numbers");
        }
    }
    return result;
}

// The largest innovation number present, or -1 if the map is empty --
// std::map iterates ascending by key, so the last element (via rbegin())
// holds the maximum. -1 is a safe "no genes at all" sentinel since
// ConnectionGene rejects negative innovation numbers, so every real
// innovation number compares greater than it.
InnovationNumber maxInnovation(const std::map<InnovationNumber, const ConnectionGene*>& innovationMap)
{
    return innovationMap.empty() ? InnovationNumber{-1} : innovationMap.rbegin()->first;
}

} // namespace

CompatibilityBreakdown compatibilityBreakdown(const Genome& genomeA, const Genome& genomeB,
                                               const CompatibilityConfig& config)
{
    validateCompatibilityConfig(config);

    const std::map<InnovationNumber, const ConnectionGene*> mapA = buildInnovationMap(genomeA, /*isGenomeA=*/true);
    const std::map<InnovationNumber, const ConnectionGene*> mapB = buildInnovationMap(genomeB, /*isGenomeA=*/false);

    const InnovationNumber maxA = maxInnovation(mapA);
    const InnovationNumber maxB = maxInnovation(mapB);

    std::size_t matching = 0;
    std::size_t disjoint = 0;
    std::size_t excess = 0;
    float weightDifferenceSum = 0.0f;

    // Genes present in A: matching (also validated for endpoint
    // consistency) or disjoint/excess relative to B's highest innovation.
    for (const auto& [innovation, connA] : mapA)
    {
        const auto itB = mapB.find(innovation);
        if (itB == mapB.end())
        {
            if (innovation > maxB)
            {
                ++excess;
            }
            else
            {
                ++disjoint;
            }
            continue;
        }

        const ConnectionGene* connB = itB->second;
        if (connA->getSourceId() != connB->getSourceId() || connA->getTargetId() != connB->getTargetId())
        {
            throw std::invalid_argument(
                "compatibilityDistance: matching innovation has conflicting endpoints between genomes");
        }

        ++matching;
        weightDifferenceSum += std::fabs(connA->getWeight() - connB->getWeight());
    }

    // Genes present only in B (already-matched innovations were counted
    // above): disjoint/excess relative to A's highest innovation.
    for (const auto& [innovation, connB] : mapB)
    {
        (void)connB;
        if (mapA.find(innovation) != mapA.end())
        {
            continue;
        }
        if (innovation > maxA)
        {
            ++excess;
        }
        else
        {
            ++disjoint;
        }
    }

    const float averageWeightDifference = matching > 0 ? weightDifferenceSum / static_cast<float>(matching) : 0.0f;

    const std::size_t largerGenomeSize = std::max(genomeA.connections().size(), genomeB.connections().size());
    const float normalization = largerGenomeSize < config.smallGenomeNormalizationThreshold
                                     ? 1.0f
                                     : static_cast<float>(largerGenomeSize);

    const float distance = config.excessCoefficient * static_cast<float>(excess) / normalization +
                            config.disjointCoefficient * static_cast<float>(disjoint) / normalization +
                            config.weightDifferenceCoefficient * averageWeightDifference;

    CompatibilityBreakdown breakdown;
    breakdown.matching = matching;
    breakdown.disjoint = disjoint;
    breakdown.excess = excess;
    breakdown.averageWeightDifference = averageWeightDifference;
    breakdown.normalization = normalization;
    breakdown.distance = distance;
    return breakdown;
}

float compatibilityDistance(const Genome& genomeA, const Genome& genomeB, const CompatibilityConfig& config)
{
    return compatibilityBreakdown(genomeA, genomeB, config).distance;
}

} // namespace ai::neat
