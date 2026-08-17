#include "training/GenerationMetrics.h"

#include <algorithm>
#include <stdexcept>

namespace training
{

float computeMean(const std::vector<float>& values)
{
    if (values.empty())
    {
        throw std::invalid_argument("computeMean: values must not be empty");
    }
    float sum = 0.0f;
    for (float value : values)
    {
        sum += value;
    }
    return sum / static_cast<float>(values.size());
}

float computeMedian(std::vector<float> values)
{
    if (values.empty())
    {
        throw std::invalid_argument("computeMedian: values must not be empty");
    }
    std::sort(values.begin(), values.end());

    const std::size_t count = values.size();
    const std::size_t mid = count / 2;
    if (count % 2 == 1)
    {
        return values[mid];
    }
    // Even count: the standard definition -- the mean of the two middle
    // (already sorted) elements, e.g. [1, 2, 3, 4] -> (2 + 3) / 2 = 2.5.
    return (values[mid - 1] + values[mid]) / 2.0f;
}

float computeMin(const std::vector<float>& values)
{
    if (values.empty())
    {
        throw std::invalid_argument("computeMin: values must not be empty");
    }
    return *std::min_element(values.begin(), values.end());
}

float computeMax(const std::vector<float>& values)
{
    if (values.empty())
    {
        throw std::invalid_argument("computeMax: values must not be empty");
    }
    return *std::max_element(values.begin(), values.end());
}

GenomeComplexity computeGenomeComplexity(const ai::neat::Genome& genome)
{
    GenomeComplexity complexity;
    complexity.nodeCount = genome.nodes().size();
    complexity.connectionGeneCount = genome.connections().size();
    complexity.enabledConnectionCount = 0;
    for (const ai::neat::ConnectionGene& connection : genome.connections())
    {
        if (connection.isEnabled())
        {
            ++complexity.enabledConnectionCount;
        }
    }
    return complexity;
}

GenerationMetrics buildGenerationMetrics(const GenerationMetricsInput& input)
{
    if (input.rawFitness.empty())
    {
        throw std::invalid_argument("buildGenerationMetrics: rawFitness must not be empty");
    }
    const std::size_t count = input.rawFitness.size();
    if (input.bestProgressValues.size() != count || input.completedLap.size() != count ||
        input.genomeComplexities.size() != count)
    {
        throw std::invalid_argument(
            "buildGenerationMetrics: bestProgressValues/completedLap/genomeComplexities must match rawFitness in size");
    }
    if (!input.adjustedFitness.empty() && input.adjustedFitness.size() != count)
    {
        throw std::invalid_argument("buildGenerationMetrics: adjustedFitness must be empty or match rawFitness in size");
    }
    if (input.bestIndividualIndex >= count)
    {
        throw std::invalid_argument("buildGenerationMetrics: bestIndividualIndex out of range");
    }

    GenerationMetrics metrics;
    metrics.generation = input.generation;

    metrics.bestFitness = computeMax(input.rawFitness);
    metrics.avgFitness = computeMean(input.rawFitness);
    metrics.medianFitness = computeMedian(input.rawFitness);
    metrics.worstFitness = computeMin(input.rawFitness);
    metrics.avgAdjustedFitness = input.adjustedFitness.empty() ? 0.0f : computeMean(input.adjustedFitness);

    metrics.speciesCount = input.speciesCount;
    metrics.largestSpeciesSize = input.largestSpeciesSize;
    metrics.smallestSpeciesSize = input.smallestSpeciesSize;
    metrics.bestSpeciesHistoricalFitness = input.bestSpeciesHistoricalFitness;
    metrics.stagnantSpeciesExcluded = input.stagnantSpeciesExcluded;

    metrics.bestProgress = input.bestProgressValues[input.bestIndividualIndex];
    metrics.avgProgress = computeMean(input.bestProgressValues);

    std::size_t lapsCompleted = 0;
    for (bool completed : input.completedLap)
    {
        if (completed)
        {
            ++lapsCompleted;
        }
    }
    metrics.lapsCompletedCount = lapsCompleted;
    metrics.completionRate = static_cast<float>(lapsCompleted) / static_cast<float>(count);

    const GenomeComplexity& bestComplexity = input.genomeComplexities[input.bestIndividualIndex];
    metrics.bestGenomeNodeCount = bestComplexity.nodeCount;
    metrics.bestGenomeConnectionGeneCount = bestComplexity.connectionGeneCount;
    metrics.bestGenomeEnabledConnectionCount = bestComplexity.enabledConnectionCount;

    std::size_t totalNodes = 0;
    std::size_t totalConnections = 0;
    for (const GenomeComplexity& complexity : input.genomeComplexities)
    {
        totalNodes += complexity.nodeCount;
        totalConnections += complexity.connectionGeneCount;
    }
    metrics.avgGenomeNodeCount = static_cast<float>(totalNodes) / static_cast<float>(count);
    metrics.avgGenomeConnectionGeneCount = static_cast<float>(totalConnections) / static_cast<float>(count);

    metrics.generationDurationSeconds = input.generationDurationSeconds;

    return metrics;
}

} // namespace training
