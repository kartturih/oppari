#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "raylib.h"

#include "ai/AIController.h"
#include "ai/FitnessEvaluator.h"
#include "ai/NeuralNetwork.h"
#include "ai/Observation.h"
#include "ai/neat/CompatibilityConfig.h"
#include "ai/neat/CompatibilityDistance.h"
#include "ai/neat/ConnectionGene.h"
#include "ai/neat/CrossoverConfig.h"
#include "ai/neat/Genome.h"
#include "ai/neat/GenomeCrossover.h"
#include "ai/neat/GenomeMutator.h"
#include "ai/neat/Individual.h"
#include "ai/neat/InnovationTracker.h"
#include "ai/neat/MutationConfig.h"
#include "ai/neat/NodeGene.h"
#include "ai/neat/PhenotypeBuilder.h"
#include "ai/neat/Population.h"
#include "ai/neat/PopulationConfig.h"
#include "ai/neat/SpeciationConfig.h"
#include "ai/neat/Speciator.h"
#include "ai/neat/Species.h"
#include "simulation/Car.h"
#include "simulation/Track.h"
#include "simulation/TrackProgress.h"
#include "simulation/TrackVisual.h"
#include "training/GenerationMetrics.h"
#include "training/TrainingLogger.h"

#include "AppConfig.h"
#include "verification/Verifications.h"

namespace verification
{

using app::kSimulationDt;
using app::kSpawnHeading;
using app::kSpawnPosition;
using app::makeCarParams;


namespace training_metrics_verify
{

using ai::neat::ConnectionGene;
using ai::neat::Genome;
using ai::neat::NodeGene;
using ai::neat::NodeId;
using ai::neat::NodeType;

template <typename Callable>
bool throwsInvalidArgument(Callable&& callable)
{
    try
    {
        callable();
    }
    catch (const std::invalid_argument&)
    {
        return true;
    }
    return false;
}

// A tiny genome with a known, hand-countable shape: 2 nodes, 3 connection
// genes, exactly 1 of them disabled -- so computeGenomeComplexity()'s three
// counts (nodes/all connections/enabled-only connections) are each a
// distinct, easily verified number.
Genome makeComplexityTestGenome()
{
    Genome genome;
    genome.addNode(NodeGene{0, NodeType::Input});
    genome.addNode(NodeGene{1, NodeType::Output});
    genome.addConnection(ConnectionGene{0, 1, 0.1f, true, 0});
    // A second, structurally distinct connection between the same two nodes
    // is not allowed by Genome (see addConnection()'s doc comment), so the
    // other two connection genes route through a third node instead.
    genome.addNode(NodeGene{2, NodeType::Hidden});
    genome.addConnection(ConnectionGene{0, 2, 0.2f, true, 1});
    genome.addConnection(ConnectionGene{2, 1, 0.3f, false, 2}); // disabled
    return genome;
}

} // namespace training_metrics_verify

// Deterministic check of training::GenerationMetrics' pure, file-I/O-free
// statistics: computeMean/computeMedian/computeMin/computeMax,
// computeGenomeComplexity, and buildGenerationMetrics itself. Every check
// uses hand-picked synthetic data -- no Population/Individual/Track
// involved (see verifyGenerationMetricsPopulationIntegration() below for
// the end-to-end check).
void verifyTrainingMetrics()
{
    using namespace training_metrics_verify;
    using training::GenerationMetrics;
    using training::GenerationMetricsInput;
    using training::GenomeComplexity;

    // 1: computeMean.
    assert(training::computeMean({1.0f, 2.0f, 3.0f, 4.0f}) == 2.5f && "mean of [1,2,3,4] must be 2.5");
    assert(throwsInvalidArgument([]() { training::computeMean({}); }) && "computeMean must reject an empty vector");

    // 2: computeMedian, odd count -- single middle element.
    assert(training::computeMedian({5.0f, 1.0f, 3.0f}) == 3.0f && "median of [5,1,3] (sorted [1,3,5]) must be 3");
    // 3: computeMedian, even count -- mean of the two middle elements.
    assert(training::computeMedian({1.0f, 2.0f, 3.0f, 4.0f}) == 2.5f && "median of [1,2,3,4] must be (2+3)/2 = 2.5");
    assert(throwsInvalidArgument([]() { training::computeMedian({}); }) && "computeMedian must reject an empty vector");

    // 4 & 5: computeMin/computeMax.
    assert(training::computeMin({5.0f, 1.0f, 3.0f}) == 1.0f && "min of [5,1,3] must be 1");
    assert(training::computeMax({5.0f, 1.0f, 3.0f}) == 5.0f && "max of [5,1,3] must be 5");
    assert(throwsInvalidArgument([]() { training::computeMin({}); }) && "computeMin must reject an empty vector");
    assert(throwsInvalidArgument([]() { training::computeMax({}); }) && "computeMax must reject an empty vector");

    // 6: computeGenomeComplexity counts nodes, all connection genes
    // (enabled and disabled), and enabled-only connections separately.
    {
        const GenomeComplexity complexity = training::computeGenomeComplexity(makeComplexityTestGenome());
        assert(complexity.nodeCount == 3 && "complexity test genome must report 3 nodes");
        assert(complexity.connectionGeneCount == 3 && "complexity test genome must report 3 connection genes total");
        assert(complexity.enabledConnectionCount == 2 &&
               "complexity test genome must report exactly 2 ENABLED connection genes");
    }

    // 7-16: buildGenerationMetrics reduces a synthetic 4-individual input
    // into every documented field correctly.
    {
        GenerationMetricsInput input;
        input.generation = 7;
        input.rawFitness = {10.0f, 20.0f, 30.0f, 40.0f};
        input.adjustedFitness = {5.0f, 10.0f, 15.0f, 20.0f};
        input.bestProgressValues = {0.1f, 0.2f, 0.9f, 0.3f};
        input.completedLap = {false, false, true, false};
        const GenomeComplexity complexityA{5, 8, 6};
        const GenomeComplexity complexityB{7, 12, 9};
        const GenomeComplexity complexityC{3, 4, 4};
        const GenomeComplexity complexityD{9, 14, 10}; // belongs to the best (highest-fitness) individual
        input.genomeComplexities = {complexityA, complexityB, complexityC, complexityD};
        input.finishReasons = {ai::EvaluationFinishReason::Collision, ai::EvaluationFinishReason::SafetyTimeout,
                                ai::EvaluationFinishReason::CompletedLaps, ai::EvaluationFinishReason::NoProgress};
        input.bestIndividualIndex = 3; // rawFitness[3] == 40, the maximum
        input.speciesCount = 2;
        input.largestSpeciesSize = 3;
        input.smallestSpeciesSize = 1;
        input.bestSpeciesHistoricalFitness = 999.0f;
        input.stagnantSpeciesExcluded = 1;
        input.generationDurationSeconds = 12.5f;
        input.bestDriving.averageAbsSteeringDelta = 0.07f;
        input.bestDriving.steeringReversalsPerSecond = 3.0f;
        input.bestDriving.physicalBrakeUsageFraction = 0.2f;

        const GenerationMetrics metrics = training::buildGenerationMetrics(input);
        assert(metrics.bestDriving.averageAbsSteeringDelta == 0.07f && metrics.bestDriving.steeringReversalsPerSecond == 3.0f &&
               metrics.bestDriving.physicalBrakeUsageFraction == 0.2f &&
               "buildGenerationMetrics must pass the best individual's driving diagnostics through unchanged");

        assert(metrics.generation == 7 && "generation must pass through unchanged"); // 7
        assert(metrics.bestFitness == 40.0f && metrics.avgFitness == 25.0f && metrics.medianFitness == 25.0f &&
               metrics.worstFitness == 10.0f && "raw fitness reduction (best/avg/median/worst) must match hand computation"); // 8
        assert(metrics.avgAdjustedFitness == 12.5f && "avgAdjustedFitness must be the mean of the adjusted vector"); // 9
        assert(metrics.speciesCount == 2 && metrics.largestSpeciesSize == 3 && metrics.smallestSpeciesSize == 1 &&
               metrics.bestSpeciesHistoricalFitness == 999.0f && metrics.stagnantSpeciesExcluded == 1 &&
               "species-level fields must pass through unchanged"); // 10
        assert(metrics.bestProgress == 0.3f && "bestProgress must be bestProgressValues[bestIndividualIndex]"); // 11
        assert(std::fabs(metrics.avgProgress - 0.375f) < 1e-5f &&
               "avgProgress must be the mean of bestProgressValues (0.1+0.2+0.9+0.3)/4 = 0.375"); // 12
        assert(metrics.lapsCompletedCount == 1 && metrics.completionRate == 0.25f &&
               "exactly one of four individuals completed a lap -> completionRate 0.25"); // 13
        assert(metrics.bestGenomeNodeCount == complexityD.nodeCount &&
               metrics.bestGenomeConnectionGeneCount == complexityD.connectionGeneCount &&
               metrics.bestGenomeEnabledConnectionCount == complexityD.enabledConnectionCount &&
               "best genome complexity must be genomeComplexities[bestIndividualIndex]"); // 14
        const float expectedAvgNodes = (5.0f + 7.0f + 3.0f + 9.0f) / 4.0f;
        const float expectedAvgConnections = (8.0f + 12.0f + 4.0f + 14.0f) / 4.0f;
        assert(std::fabs(metrics.avgGenomeNodeCount - expectedAvgNodes) < 1e-5f &&
               std::fabs(metrics.avgGenomeConnectionGeneCount - expectedAvgConnections) < 1e-5f &&
               "avg genome node/connection counts must be the population mean"); // 15
        assert(metrics.generationDurationSeconds == 12.5f && "generationDurationSeconds must pass through unchanged"); // 16
        assert(metrics.terminatedCollisionCount == 1 && metrics.terminatedSafetyTimeoutCount == 1 &&
               metrics.terminatedCompletedLapsCount == 1 && metrics.terminatedNoProgressCount == 1 &&
               metrics.terminatedSlowStartCount == 0 &&
               "finishReasons must be reduced into the five terminated*Count fields by simple counting"); // 21
    }

    // 22-27: buildGenerationMetrics input validation. Every negative case
    // below fills in every OTHER field with a valid, correctly-sized value
    // (finishReasons included) so each one exercises exactly the single
    // validation branch it names.
    {
        GenerationMetricsInput empty;
        assert(throwsInvalidArgument([&]() { training::buildGenerationMetrics(empty); }) &&
               "empty rawFitness must be rejected"); // 22

        GenerationMetricsInput mismatched;
        mismatched.rawFitness = {1.0f, 2.0f};
        mismatched.bestProgressValues = {0.1f}; // wrong size
        mismatched.completedLap = {false, false};
        mismatched.genomeComplexities = {GenomeComplexity{}, GenomeComplexity{}};
        mismatched.finishReasons = {ai::EvaluationFinishReason::Collision, ai::EvaluationFinishReason::Collision};
        assert(throwsInvalidArgument([&]() { training::buildGenerationMetrics(mismatched); }) &&
               "a per-individual vector with the wrong size must be rejected"); // 23

        GenerationMetricsInput badAdjusted;
        badAdjusted.rawFitness = {1.0f, 2.0f};
        badAdjusted.adjustedFitness = {1.0f}; // wrong size (and not empty)
        badAdjusted.bestProgressValues = {0.1f, 0.2f};
        badAdjusted.completedLap = {false, false};
        badAdjusted.genomeComplexities = {GenomeComplexity{}, GenomeComplexity{}};
        badAdjusted.finishReasons = {ai::EvaluationFinishReason::Collision, ai::EvaluationFinishReason::Collision};
        assert(throwsInvalidArgument([&]() { training::buildGenerationMetrics(badAdjusted); }) &&
               "a non-empty adjustedFitness with the wrong size must be rejected"); // 24

        GenerationMetricsInput badIndex;
        badIndex.rawFitness = {1.0f, 2.0f};
        badIndex.bestProgressValues = {0.1f, 0.2f};
        badIndex.completedLap = {false, false};
        badIndex.genomeComplexities = {GenomeComplexity{}, GenomeComplexity{}};
        badIndex.finishReasons = {ai::EvaluationFinishReason::Collision, ai::EvaluationFinishReason::Collision};
        badIndex.bestIndividualIndex = 2; // out of range for size 2
        assert(throwsInvalidArgument([&]() { training::buildGenerationMetrics(badIndex); }) &&
               "an out-of-range bestIndividualIndex must be rejected"); // 25

        GenerationMetricsInput badFinishReasonsSize;
        badFinishReasonsSize.rawFitness = {1.0f, 2.0f};
        badFinishReasonsSize.bestProgressValues = {0.1f, 0.2f};
        badFinishReasonsSize.completedLap = {false, false};
        badFinishReasonsSize.genomeComplexities = {GenomeComplexity{}, GenomeComplexity{}};
        badFinishReasonsSize.finishReasons = {ai::EvaluationFinishReason::Collision}; // wrong size
        assert(throwsInvalidArgument([&]() { training::buildGenerationMetrics(badFinishReasonsSize); }) &&
               "a wrong-sized finishReasons vector must be rejected"); // 26

        GenerationMetricsInput unfinishedIndividual;
        unfinishedIndividual.rawFitness = {1.0f, 2.0f};
        unfinishedIndividual.bestProgressValues = {0.1f, 0.2f};
        unfinishedIndividual.completedLap = {false, false};
        unfinishedIndividual.genomeComplexities = {GenomeComplexity{}, GenomeComplexity{}};
        unfinishedIndividual.finishReasons = {ai::EvaluationFinishReason::Collision, ai::EvaluationFinishReason::None};
        assert(throwsInvalidArgument([&]() { training::buildGenerationMetrics(unfinishedIndividual); }) &&
               "finishReasons containing EvaluationFinishReason::None (a still-running individual) must be rejected"); // 27
    }

    TraceLog(LOG_INFO, "Training metrics verification: all deterministic checks passed");
}

namespace training_logger_verify
{

// Reads every line of path (assumed to exist) into a vector, in order,
// without trailing newlines.
std::vector<std::string> readLines(const std::string& path)
{
    std::ifstream in(path);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line))
    {
        lines.push_back(line);
    }
    return lines;
}

std::size_t countFields(const std::string& csvLine)
{
    if (csvLine.empty())
    {
        return 0;
    }
    return static_cast<std::size_t>(std::count(csvLine.begin(), csvLine.end(), ',')) + 1;
}

training::RunMetadata makeTestMetadata()
{
    training::RunMetadata metadata;
    metadata.buildVersion = "test-build";
    metadata.trackName = "test-track";
    metadata.targetLapCount = ai::kTargetLapCount;
    metadata.safetyTimeoutSeconds = ai::kSafetyTimeoutSeconds;
    return metadata;
}

// A fresh, empty scratch directory under the OS temp directory -- never
// under the project's own results/ directory, so running this verification
// (which happens on every program startup) can never pollute or collide
// with real experiment output. Removed entirely both before (in case a
// previous crashed run left it behind) and after this suite runs.
std::filesystem::path scratchDir()
{
    return std::filesystem::temp_directory_path() / "oppari_training_logger_verify";
}

} // namespace training_logger_verify

// Deterministic check of training::TrainingLogger and its free
// CSV-formatting functions: csvHeaderLine()/generationMetricsToCsvRow()/
// formatFloat() as pure string-building (no file I/O), then TrainingLogger's
// actual file creation/append/no-overwrite behavior against a scratch
// directory under the OS temp directory (removed before and after).
void verifyTrainingLogger()
{
    using namespace training_logger_verify;

    // 1: formatFloat is locale-independent ('.' separator) and produces the
    // shortest round-trippable representation -- no fixed decimal count, no
    // trailing zeros.
    assert(training::formatFloat(812.4f) == "812.4" && "formatFloat must format 812.4f as \"812.4\"");
    assert(training::formatFloat(-3.5f) == "-3.5" && "formatFloat must handle negative values with '.' separator");
    assert(training::formatFloat(100.0f) == "100" && "formatFloat must not pad an exact integer value with trailing zeros");
    assert(training::formatFloat(0.0f) == "0" && "formatFloat must format zero as \"0\"");
    // 2: non-finite inputs never throw/crash -- they format as fixed
    // sentinel strings instead.
    assert(training::formatFloat(std::numeric_limits<float>::quiet_NaN()) == "nan" && "formatFloat must format NaN as \"nan\"");
    assert(training::formatFloat(std::numeric_limits<float>::infinity()) == "inf" && "formatFloat must format +Inf as \"inf\"");
    assert(training::formatFloat(-std::numeric_limits<float>::infinity()) == "-inf" &&
           "formatFloat must format -Inf as \"-inf\"");

    // 3: csvHeaderLine() column count matches GenerationMetrics' own field
    // count (37) exactly -- compatibility_threshold was appended after the
    // original columns, then the eight best-individual driving diagnostics, then
    // two brake-request columns, then terminated_completed_laps_count, so every
    // pre-existing column keeps its index.
    const std::string header = training::csvHeaderLine();
    assert(countFields(header) == 37 && "CSV header must have exactly 37 columns, one per GenerationMetrics field");
    assert(header.substr(0, 10) == "generation" && "CSV header's first column must be \"generation\"");
    const std::string tailColumns =
        "best_physical_brake_usage_fraction,best_brake_request_dominant_fraction,best_brake_onset_speed,"
        "terminated_completed_laps_count";
    assert(header.size() > tailColumns.size() &&
           header.compare(header.size() - tailColumns.size(), tailColumns.size(), tailColumns) == 0 &&
           "CSV header must end with the physical-brake-usage, brake-request-dominant, brake-onset-speed and "
           "completed-laps columns");
    assert(header.find(",terminated_safety_timeout_count,") != std::string::npos &&
           header.find("max_time") == std::string::npos &&
           "the old generic max_time termination column must be gone, replaced by terminated_safety_timeout_count");
    assert(header.find(",compatibility_threshold,best_avg_abs_steering_delta,") != std::string::npos &&
           "the driving-diagnostic columns must directly follow compatibility_threshold");

    // 4: generationMetricsToCsvRow() produces the same column COUNT as the
    // header, in the documented order, with every field's value recoverable
    // by parsing the row back.
    {
        training::GenerationMetrics metrics;
        metrics.generation = 42;
        metrics.bestFitness = 812.4f;
        metrics.avgFitness = 436.1f;
        metrics.medianFitness = 421.8f;
        metrics.worstFitness = 12.3f;
        metrics.avgAdjustedFitness = 200.5f;
        metrics.speciesCount = 5;
        metrics.largestSpeciesSize = 20;
        metrics.smallestSpeciesSize = 1;
        metrics.bestSpeciesHistoricalFitness = 900.0f;
        metrics.stagnantSpeciesExcluded = 2;
        metrics.bestProgress = 1.5f;
        metrics.avgProgress = 0.7f;
        metrics.lapsCompletedCount = 3;
        metrics.completionRate = 0.06f;
        metrics.bestGenomeNodeCount = 14;
        metrics.bestGenomeConnectionGeneCount = 31;
        metrics.bestGenomeEnabledConnectionCount = 29;
        metrics.avgGenomeNodeCount = 11.2f;
        metrics.avgGenomeConnectionGeneCount = 20.4f;
        metrics.generationDurationSeconds = 30.0f;
        metrics.terminatedCollisionCount = 7;
        metrics.terminatedSafetyTimeoutCount = 8;
        metrics.terminatedNoProgressCount = 9;
        metrics.terminatedSlowStartCount = 10;
        metrics.compatibilityThresholdUsed = 2.7f;
        metrics.bestDriving.averageAbsSteeringDelta = 0.125f;
        metrics.bestDriving.steeringReversalsPerSecond = 2.5f;
        metrics.bestDriving.steeringSaturationFraction = 0.75f;
        metrics.bestDriving.meanAbsSteering = 0.5f;
        metrics.bestDriving.meanLateralAcceleration = 640.0f;
        metrics.bestDriving.frontSlipBeyondPeakFraction = 0.25f;
        metrics.bestDriving.lap2PlusAverageSpeed = 262.5f;
        metrics.bestDriving.physicalBrakeUsageFraction = 0.0625f;
        metrics.bestDriving.brakeRequestDominantFraction = 0.125f;
        metrics.bestDriving.brakeOnsetSpeed = 300.0f;
        metrics.terminatedCompletedLapsCount = 11;

        const std::string row = training::generationMetricsToCsvRow(metrics);
        assert(countFields(row) == countFields(header) &&
               "a CSV data row must have exactly as many columns as the header"); // 4

        std::istringstream rowStream(row);
        std::string field;
        std::vector<std::string> fields;
        while (std::getline(rowStream, field, ','))
        {
            fields.push_back(field);
        }
        assert(fields.size() == 37 && "split CSV row must yield exactly 37 fields"); // 5
        assert(fields[0] == "42" && "column 0 (generation) must be \"42\""); // 6
        assert(fields[1] == "812.4" && "column 1 (best_fitness) must be locale-independent \"812.4\""); // 7
        assert(fields[15] == "14" && "column 15 (best_genome_nodes) must be \"14\""); // 8
        assert(fields[16] == "31" && "column 16 (best_genome_connections) must be \"31\""); // 9
        assert(fields[20] == "30" && "column 20 (generation_duration_seconds) must be \"30\""); // 10
        assert(fields[21] == "7" && fields[22] == "8" && fields[23] == "9" && fields[24] == "10" &&
               "columns 21-24 (terminated_collision/safety_timeout/no_progress/slow_start_count) must be appended, "
               "in that order, after every pre-existing column"); // 10 (continued)
        assert(fields[25] == "2.7" && "column 25 (compatibility_threshold) must keep its original index"); // 10 (continued)
        assert(fields[26] == "0.125" && fields[27] == "2.5" && fields[28] == "0.75" && fields[29] == "0.5" &&
               fields[30] == "640" && fields[31] == "0.25" && fields[32] == "262.5" && fields[33] == "0.0625" &&
               "columns 26-33 must be the eight best-individual driving diagnostics, in documented order"); // 10 (continued)
        assert(fields[34] == "0.125" && fields[35] == "300" &&
               "columns 34-35 must be best_brake_request_dominant_fraction and best_brake_onset_speed");
        assert(fields[36] == "11" && "column 36 must be terminated_completed_laps_count, appended last");
    }

    // 11-16: TrainingLogger's real file behavior, against a scratch temp
    // directory.
    {
        const std::filesystem::path dir = scratchDir();
        std::error_code ec;
        std::filesystem::remove_all(dir, ec); // clean slate, ignoring "did not exist"

        training::TrainingLogger logger(dir.string(), makeTestMetadata());
        assert(std::filesystem::exists(logger.getCsvPath()) && "TrainingLogger must create the CSV file"); // 11
        assert(std::filesystem::exists(logger.getMetadataPath()) && "TrainingLogger must create the metadata file"); // 12

        // 12b: the metadata file's [speciation] section reports the full
        // adaptive-threshold configuration -- initial threshold, target
        // range, adjustment step, and both bounds -- alongside the
        // pre-existing compatibility_threshold line.
        {
            const std::vector<std::string> metadataLines = readLines(logger.getMetadataPath());
            auto hasLine = [&metadataLines](const std::string& expected)
            {
                return std::find(metadataLines.begin(), metadataLines.end(), expected) != metadataLines.end();
            };
            assert(hasLine("target_lap_count = 3") && hasLine("safety_timeout_seconds = 180") &&
                   "metadata must report the 3-lap target and the 180 s safety timeout");
            assert(hasLine("compatibility_threshold = 3") &&
                   "metadata must report the configured initial compatibility_threshold"); // 12b
            assert(hasLine("target_species_min = 5") && "metadata must report target_species_min"); // 12b
            assert(hasLine("target_species_max = 10") && "metadata must report target_species_max"); // 12b
            assert(hasLine("compatibility_threshold_adjustment = 0.1") &&
                   "metadata must report compatibility_threshold_adjustment"); // 12b
            assert(hasLine("minimum_compatibility_threshold = 0.5") &&
                   "metadata must report minimum_compatibility_threshold"); // 12b
            assert(hasLine("maximum_compatibility_threshold = 10") &&
                   "metadata must report maximum_compatibility_threshold"); // 12b
        }

        const std::vector<std::string> headerOnly = readLines(logger.getCsvPath());
        assert(headerOnly.size() == 1 && headerOnly[0] == training::csvHeaderLine() &&
               "a freshly constructed TrainingLogger's CSV file must contain exactly the header line"); // 13

        training::GenerationMetrics m0;
        m0.generation = 0;
        m0.bestFitness = 100.0f;
        training::GenerationMetrics m1;
        m1.generation = 1;
        m1.bestFitness = 150.0f;
        logger.logGeneration(m0);
        logger.logGeneration(m1);

        const std::vector<std::string> afterTwoRows = readLines(logger.getCsvPath());
        assert(afterTwoRows.size() == 3 &&
               "header + 2 logGeneration() calls must produce exactly 3 lines, no duplicates/missing rows"); // 14
        assert(afterTwoRows[1] == training::generationMetricsToCsvRow(m0) &&
               afterTwoRows[2] == training::generationMetricsToCsvRow(m1) &&
               "logged rows must appear in call order with no row lost or duplicated"); // 15

        // 15b (L): the logger has no generation limit -- rows for generations
        // beyond 300 (including 1000 and past it) append exactly like any other.
        for (std::size_t generation : {std::size_t{299}, std::size_t{300}, std::size_t{301}, std::size_t{1000},
                                       std::size_t{1001}, std::size_t{5000}})
        {
            training::GenerationMetrics late;
            late.generation = generation;
            late.bestFitness = 200.0f;
            logger.logGeneration(late);
        }
        const std::vector<std::string> afterLateRows = readLines(logger.getCsvPath());
        assert(afterLateRows.size() == 3 + 6 && "generations past 300 must keep being logged, one row each");
        assert(afterLateRows[6].rfind("1000,", 0) == 0 && afterLateRows[8].rfind("5000,", 0) == 0 &&
               "rows for generation 1000 and beyond must carry their real generation number"); // L

        // 16: a second TrainingLogger constructed immediately afterward,
        // pointed at the SAME resultsDir (almost certainly within the same
        // second), must never overwrite the first run's files -- it gets a
        // distinct run id (numeric suffix) and its own fresh CSV/metadata
        // pair.
        training::TrainingLogger secondLogger(dir.string(), makeTestMetadata());
        assert(secondLogger.getRunId() != logger.getRunId() &&
               secondLogger.getCsvPath() != logger.getCsvPath() &&
               "two TrainingLoggers constructed against the same resultsDir must never collide on run id/CSV path"); // 16
        assert(std::filesystem::exists(logger.getCsvPath()) &&
               readLines(logger.getCsvPath()).size() == 9 &&
               "constructing a second TrainingLogger must not touch the first run's existing CSV file"); // 16 (continued)

        std::filesystem::remove_all(dir, ec); // leave no trace in the OS temp directory
    }

    TraceLog(LOG_INFO, "Training logger verification: all deterministic checks passed");
}

// End-to-end integration check: runs a small, fast-finishing crash-genome
// Population through its first generation transition, re-snapshotting
// every individual's raw fitness/progress/hasCompletedLap/genome complexity
// immediately before every update() call (same technique
// verifySpeciesAwareReproduction() uses), then confirms
// Population::getLastGenerationMetrics() matches those independently-
// recomputed values, and that its species-level fields agree with
// getCurrentSpecies()/getReproductionStats() read immediately after.
void verifyGenerationMetricsPopulationIntegration(const simulation::Track& track)
{
    using ai::neat::CompatibilityConfig;
    using ai::neat::CrossoverConfig;
    using ai::neat::Genome;
    using ai::neat::MutationConfig;
    using ai::neat::Population;
    using ai::neat::PopulationConfig;
    using ai::neat::SpeciationConfig;

    const Genome crashGenome = population_verify::makeCrashGenome();
    const PopulationConfig popConfig = population_verify::makeTestPopulationConfig(10, 77u);
    const MutationConfig mutationConfig;
    const CrossoverConfig crossoverConfig;
    const CompatibilityConfig compatibilityConfig;
    const SpeciationConfig speciationConfig;

    Population population(crashGenome, track, makeCarParams(), kSpawnPosition, kSpawnHeading, popConfig, mutationConfig,
                           crossoverConfig, compatibilityConfig, speciationConfig);

    // Re-snapshot every individual's raw state immediately before every
    // update() call, exactly like verifySpeciesAwareReproduction() above
    // (generation-finish detection and reproduction happen together inside
    // one update() call, so there's no other way to observe the exact
    // just-finished state). finishedBeforeTransitionCall records how many
    // individuals were already finished going into that call: only when it
    // equals the full population size is the snapshot provably exact --
    // otherwise a still-running individual could gain a final burst of
    // fitness/progress, and the strict comparisons below are skipped in
    // favor of invariants that hold unconditionally either way.
    std::vector<float> lastRawFitness;
    std::vector<float> lastBestProgress;
    std::vector<bool> lastCompletedLap;
    std::size_t finishedBeforeTransitionCall = 0;
    for (int step = 0; step < 4000 && population.getGeneration() == 0; ++step)
    {
        lastRawFitness.clear();
        lastBestProgress.clear();
        lastCompletedLap.clear();
        for (std::size_t i = 0; i < population.size(); ++i)
        {
            const ai::neat::Individual& individual = population.getIndividual(i);
            lastRawFitness.push_back(individual.getFitness());
            lastBestProgress.push_back(individual.getProgress().getBestProgress());
            lastCompletedLap.push_back(individual.getFitnessEvaluator().hasCompletedLap());
        }
        finishedBeforeTransitionCall = population.getFinishedCount();
        population.update(kSimulationDt);
    }
    assert(population.getGeneration() == 1 && "setup: population must complete generation 0 within the step budget");

    const training::GenerationMetrics& metrics = population.getLastGenerationMetrics();

    assert(metrics.generation == 0 && "logged metrics must be for generation 0, the generation that just finished"); // 1
    assert(metrics.bestFitness == population.getLastGenerationBestFitness() &&
           "GenerationMetrics::bestFitness must agree with Population::getLastGenerationBestFitness()"); // 2

    // 2b: generation 0's speciation pass used the configured INITIAL
    // threshold, unadjusted -- the adaptive step only prepares a value for
    // generation 1's speciate() call, which hasn't happened yet.
    assert(metrics.compatibilityThresholdUsed == speciationConfig.compatibilityThreshold &&
           "GenerationMetrics::compatibilityThresholdUsed for generation 0 must equal the configured initial "
           "threshold, unadjusted"); // 2b

    const bool snapshotIsExact = (finishedBeforeTransitionCall == population.size());
    if (snapshotIsExact)
    {
        assert(metrics.bestFitness == training::computeMax(lastRawFitness) &&
               metrics.avgFitness == training::computeMean(lastRawFitness) &&
               metrics.medianFitness == training::computeMedian(lastRawFitness) &&
               metrics.worstFitness == training::computeMin(lastRawFitness) &&
               "GenerationMetrics fitness reduction must match an independent recomputation from the same snapshot"); // 3

        float expectedAvgProgress = 0.0f;
        std::size_t expectedLaps = 0;
        for (std::size_t i = 0; i < lastBestProgress.size(); ++i)
        {
            expectedAvgProgress += lastBestProgress[i];
            if (lastCompletedLap[i])
            {
                ++expectedLaps;
            }
        }
        expectedAvgProgress /= static_cast<float>(lastBestProgress.size());
        assert(std::fabs(metrics.avgProgress - expectedAvgProgress) < 1e-4f &&
               "GenerationMetrics::avgProgress must match an independent recomputation"); // 4
        assert(metrics.lapsCompletedCount == expectedLaps &&
               std::fabs(metrics.completionRate -
                         static_cast<float>(expectedLaps) / static_cast<float>(lastCompletedLap.size())) < 1e-6f &&
               "GenerationMetrics laps-completed/completion-rate must match an independent recomputation"); // 5
    }
    else
    {
        // The snapshot's fitness/progress values are stale for whichever
        // individual(s) finished during the transition-triggering call
        // itself -- but metrics is still guaranteed self-consistent: its own
        // best/avg/median/worst must obey the same ordering relationship any
        // valid statistics of the same underlying (unobserved) data would.
        assert(metrics.worstFitness <= metrics.medianFitness && metrics.medianFitness <= metrics.bestFitness &&
               metrics.worstFitness <= metrics.avgFitness && metrics.avgFitness <= metrics.bestFitness &&
               "GenerationMetrics fitness statistics must stay internally ordered (worst <= median/avg <= best)");
        assert(metrics.lapsCompletedCount <= population.size() && metrics.completionRate >= 0.0f &&
               metrics.completionRate <= 1.0f && "lapsCompletedCount/completionRate must stay within valid bounds");
    }

    assert(metrics.speciesCount == population.getCurrentSpecies().size() &&
           "GenerationMetrics::speciesCount must match Population::getCurrentSpecies().size() read right after the "
           "transition"); // 6

    std::size_t expectedStagnantExcluded = 0;
    for (const Population::SpeciesReproductionStats& stats : population.getReproductionStats())
    {
        if (stats.stagnant && !stats.reproductionEligible)
        {
            ++expectedStagnantExcluded;
        }
    }
    assert(metrics.stagnantSpeciesExcluded == expectedStagnantExcluded &&
           "GenerationMetrics::stagnantSpeciesExcluded must match Population::getReproductionStats()"); // 7

    assert(metrics.generationDurationSeconds > 0.0f && metrics.generationDurationSeconds <= ai::kSafetyTimeoutSeconds + 1e-4f &&
           "generation duration must be positive and bounded by the safety timeout"); // 8
    assert(!std::isnan(metrics.bestFitness) && !std::isnan(metrics.avgFitness) && !std::isnan(metrics.avgProgress) &&
           "no metric field may be NaN"); // 9

    // 10: every individual's evaluation has finished with exactly one of
    // the five EvaluationFinishReason values by the time a generation
    // transition happens -- the five terminated*Count fields must always
    // sum to exactly the population size.
    const std::size_t terminatedTotal = metrics.terminatedCollisionCount + metrics.terminatedSafetyTimeoutCount +
                                         metrics.terminatedNoProgressCount + metrics.terminatedSlowStartCount +
                                         metrics.terminatedCompletedLapsCount;
    assert(terminatedTotal == population.size() &&
           "terminatedCollisionCount + terminatedSafetyTimeoutCount + terminatedNoProgressCount + "
           "terminatedSlowStartCount + terminatedCompletedLapsCount must sum to exactly the population size"); // 10

    TraceLog(LOG_INFO, "Generation metrics / Population integration verification: all deterministic checks passed");
}

// Proves the exact invariant main.cpp's NORMAL/FAST training-speed toggle
// relies on: Population::update() is a pure, deterministic state advance
// driven only by the fixed dt passed to each call -- Population/Individual/
// Car never read wall-clock time, frame count, or anything about how the
// caller's own loop happens to group those calls. So the SAME total number
// of update(kSimulationDt) calls must produce an IDENTICAL result whether
// grouped one-per-outer-iteration (NORMAL) or into large, irregular batches
// (FAST, and deliberately more irregular than FAST's own fixed
// kFastModeStepsPerFrame, to rule out the invariant only happening to hold
// for that one specific batch size).
void verifyTrainingSpeedDeterminism(const simulation::Track& track)
{
    using ai::neat::CompatibilityConfig;
    using ai::neat::CrossoverConfig;
    using ai::neat::Genome;
    using ai::neat::MutationConfig;
    using ai::neat::Population;
    using ai::neat::PopulationConfig;
    using ai::neat::SpeciationConfig;

    // 1: the fixed simulation step main.cpp passes to every update() call,
    // in both modes, is exactly 1/60s -- the one invariant FAST mode must
    // never touch (it only changes how many such calls happen per rendered
    // frame, never their size).
    assert(std::fabs(kSimulationDt - (1.0f / 60.0f)) < 1e-6f &&
           "kSimulationDt must be exactly 1/60s -- FAST mode must never change the fixed simulation step"); // 1

    const Genome crashGenome = population_verify::makeCrashGenome();
    const PopulationConfig popConfig = population_verify::makeTestPopulationConfig(12, 999u);
    const MutationConfig mutationConfig;
    const CrossoverConfig crossoverConfig;
    const CompatibilityConfig compatibilityConfig;
    const SpeciationConfig speciationConfig;

    auto buildPopulation = [&]() -> Population
    {
        return Population(crashGenome, track, makeCarParams(), kSpawnPosition, kSpawnHeading, popConfig,
                           mutationConfig, crossoverConfig, compatibilityConfig, speciationConfig);
    };

    Population singleStepPopulation = buildPopulation(); // "NORMAL": one update() call per outer iteration
    Population batchedPopulation = buildPopulation();    // "FAST": irregular multi-step batches

    // Comfortably enough steps (at ~60 steps/sim-second) for several
    // generation transitions with the fast-crashing genome (see
    // makeCrashGenome()'s own comment: crashes within a few hundred steps).
    constexpr int kTotalSteps = 6000;
    // Deliberately irregular and including main.cpp's real
    // kFastModeStepsPerFrame (100) among sizes that don't evenly divide
    // kTotalSteps, so no coincidental alignment could hide a bug.
    constexpr int kBatchSizes[] = {100, 7, 13, 1, 100, 41};
    constexpr std::size_t kBatchSizeCount = sizeof(kBatchSizes) / sizeof(kBatchSizes[0]);

    for (int i = 0; i < kTotalSteps; ++i)
    {
        singleStepPopulation.update(kSimulationDt);
    }

    {
        int stepsDone = 0;
        std::size_t batchIndex = 0;
        while (stepsDone < kTotalSteps)
        {
            const int batchSize = std::min(kBatchSizes[batchIndex % kBatchSizeCount], kTotalSteps - stepsDone);
            for (int s = 0; s < batchSize; ++s)
            {
                batchedPopulation.update(kSimulationDt);
            }
            stepsDone += batchSize;
            ++batchIndex;
        }
        assert(stepsDone == kTotalSteps && "setup: batched driver must reach exactly kTotalSteps, same as the single-step driver");
    }

    // 2: several generation transitions actually happened, so this exercises
    // real reproduction (crossover/mutation/speciation RNG draws), not just
    // per-frame physics -- the property under test only matters if RNG-
    // consuming code ran during the batches.
    assert(singleStepPopulation.getGeneration() >= 3 && batchedPopulation.getGeneration() >= 3 &&
           "setup: kTotalSteps must be enough for several generation transitions in both drivers"); // 2

    // 3: identical total update() calls, however grouped, must reach the
    // exact same generation number.
    assert(singleStepPopulation.getGeneration() == batchedPopulation.getGeneration() &&
           "the same total number of update() calls must reach the same generation regardless of batch grouping"); // 3

    // 4: every individual's raw fitness, best progress, and completed-lap
    // state must match exactly -- not just the aggregate generation count.
    assert(singleStepPopulation.size() == batchedPopulation.size() && "setup: population sizes must match");
    for (std::size_t i = 0; i < singleStepPopulation.size(); ++i)
    {
        const ai::neat::Individual& a = singleStepPopulation.getIndividual(i);
        const ai::neat::Individual& b = batchedPopulation.getIndividual(i);
        assert(a.getFitness() == b.getFitness() &&
               "per-individual fitness must be bit-identical regardless of update() batch grouping"); // 4
        assert(a.getProgress().getBestProgress() == b.getProgress().getBestProgress() &&
               "per-individual bestProgress must be bit-identical regardless of update() batch grouping");
        assert(a.getFitnessEvaluator().hasCompletedLap() == b.getFitnessEvaluator().hasCompletedLap() &&
               "per-individual lap-completion state must match regardless of update() batch grouping");
    }

    // 5: species count and the last completed generation's full metrics
    // (genome topology included) must also match exactly.
    assert(singleStepPopulation.getSpeciesCount() == batchedPopulation.getSpeciesCount() &&
           "species count must be identical regardless of update() batch grouping"); // 5

    const training::GenerationMetrics& metricsA = singleStepPopulation.getLastGenerationMetrics();
    const training::GenerationMetrics& metricsB = batchedPopulation.getLastGenerationMetrics();
    assert(metricsA.bestFitness == metricsB.bestFitness && metricsA.avgFitness == metricsB.avgFitness &&
           metricsA.bestProgress == metricsB.bestProgress && metricsA.avgProgress == metricsB.avgProgress &&
           "last completed generation's fitness/progress metrics must be bit-identical regardless of batch grouping"); // 6
    assert(metricsA.bestGenomeNodeCount == metricsB.bestGenomeNodeCount &&
           metricsA.bestGenomeConnectionGeneCount == metricsB.bestGenomeConnectionGeneCount &&
           metricsA.bestGenomeEnabledConnectionCount == metricsB.bestGenomeEnabledConnectionCount &&
           "best genome's topology (nodes/connections) must be identical regardless of batch grouping -- FAST mode "
           "cannot change which structural mutations occurred or in what order"); // 7
    assert(metricsA.compatibilityThresholdUsed == metricsB.compatibilityThresholdUsed &&
           "the adaptive compatibility threshold must evolve identically regardless of batch grouping");

    TraceLog(LOG_INFO, "Training speed determinism verification: all deterministic checks passed");
}

} // namespace verification
