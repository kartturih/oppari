#include "training/TrainingLogger.h"

#include <array>
#include <charconv>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <sstream>
#include <stdexcept>

namespace training
{

namespace
{

// "YYYYMMDD_HHMMSS" from the current local time -- second resolution, which
// is why TrainingLogger still needs a numeric-suffix fallback for two runs
// started within the same second (see makeUniqueRunId() below).
std::string formatTimestampForFilename(std::tm localTime)
{
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%04d%02d%02d_%02d%02d%02d", localTime.tm_year + 1900, localTime.tm_mon + 1,
                  localTime.tm_mday, localTime.tm_hour, localTime.tm_min, localTime.tm_sec);
    return std::string(buffer);
}

// "YYYY-MM-DD HH:MM:SS" -- human-readable form stored in the metadata file
// (RunMetadata::startedAtLocal), independent of the filename-safe format
// above.
std::string formatTimestampForDisplay(std::tm localTime)
{
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%04d-%02d-%02d %02d:%02d:%02d", localTime.tm_year + 1900, localTime.tm_mon + 1,
                  localTime.tm_mday, localTime.tm_hour, localTime.tm_min, localTime.tm_sec);
    return std::string(buffer);
}

std::tm currentLocalTime()
{
    const std::time_t now = std::time(nullptr);
    std::tm localTime{};
#if defined(_WIN32)
    localtime_s(&localTime, &now);
#else
    localtime_r(&now, &localTime);
#endif
    return localTime;
}

// Finds the first unused "training_run_<timestamp>[-N]" stem under
// resultsDir (which must already exist) by checking for either a matching
// .csv or .meta.txt file -- so a CSV/metadata pair from an earlier run can
// never be partially overwritten by a later one, even if only one of the
// two files from that earlier run still exists for some reason.
std::string makeUniqueRunId(const std::filesystem::path& resultsDir, std::tm localTime)
{
    const std::string baseStem = "training_run_" + formatTimestampForFilename(localTime);

    std::string candidate = baseStem;
    int suffix = 2;
    while (std::filesystem::exists(resultsDir / (candidate + ".csv")) ||
           std::filesystem::exists(resultsDir / (candidate + ".meta.txt")))
    {
        candidate = baseStem + "-" + std::to_string(suffix);
        ++suffix;
    }
    return candidate;
}

void writeMetadataFile(const std::string& path, const RunMetadata& metadata, const std::string& csvFileName)
{
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out.is_open())
    {
        throw std::runtime_error("TrainingLogger: failed to open metadata file for writing: " + path);
    }

    // Plain "key = value" lines, one per line -- deliberately not JSON/INI/
    // any other structured format: this is a human-readable reproducibility
    // note, not something other tooling is expected to parse. Grouped to
    // mirror the section headers in the Stage 21 report.
    out << "# Training run metadata (Stage 21)\n";
    out << "run_id = " << metadata.runId << "\n";
    out << "started_at_local = " << metadata.startedAtLocal << "\n";
    out << "build_version = " << metadata.buildVersion << "\n";
    out << "csv_file = " << csvFileName << "\n";
    out << "\n[track]\n";
    out << "track_name = " << metadata.trackName << "\n";
    out << "\n[population]\n";
    out << "population_size = " << metadata.populationConfig.populationSize << "\n";
    out << "elite_count = " << metadata.populationConfig.eliteCount << "\n";
    out << "tournament_size = " << metadata.populationConfig.tournamentSize << "\n";
    out << "species_stagnation_limit = " << metadata.populationConfig.speciesStagnationLimit << "\n";
    out << "random_seed = " << metadata.populationConfig.randomSeed << "\n";
    out << "max_evaluation_time_seconds = " << formatFloat(metadata.maxEvaluationTimeSeconds) << "\n";
    out << "\n[mutation]\n";
    out << "weight_mutation_probability = " << formatFloat(metadata.mutationConfig.weightMutationProbability) << "\n";
    out << "weight_perturb_probability = " << formatFloat(metadata.mutationConfig.weightPerturbProbability) << "\n";
    out << "perturb_strength = " << formatFloat(metadata.mutationConfig.perturbStrength) << "\n";
    out << "replacement_weight_min = " << formatFloat(metadata.mutationConfig.replacementWeightMin) << "\n";
    out << "replacement_weight_max = " << formatFloat(metadata.mutationConfig.replacementWeightMax) << "\n";
    out << "add_connection_probability = " << formatFloat(metadata.mutationConfig.addConnectionProbability) << "\n";
    out << "new_connection_weight_min = " << formatFloat(metadata.mutationConfig.newConnectionWeightMin) << "\n";
    out << "new_connection_weight_max = " << formatFloat(metadata.mutationConfig.newConnectionWeightMax) << "\n";
    out << "add_connection_max_attempts = " << metadata.mutationConfig.addConnectionMaxAttempts << "\n";
    out << "add_node_probability = " << formatFloat(metadata.mutationConfig.addNodeProbability) << "\n";
    out << "\n[crossover]\n";
    out << "matching_gene_choose_parent_a_probability = "
        << formatFloat(metadata.crossoverConfig.matchingGeneChooseParentAProbability) << "\n";
    out << "disabled_gene_remain_disabled_probability = "
        << formatFloat(metadata.crossoverConfig.disabledGeneRemainDisabledProbability) << "\n";
    out << "\n[compatibility]\n";
    out << "excess_coefficient = " << formatFloat(metadata.compatibilityConfig.excessCoefficient) << "\n";
    out << "disjoint_coefficient = " << formatFloat(metadata.compatibilityConfig.disjointCoefficient) << "\n";
    out << "weight_difference_coefficient = " << formatFloat(metadata.compatibilityConfig.weightDifferenceCoefficient) << "\n";
    out << "small_genome_normalization_threshold = " << metadata.compatibilityConfig.smallGenomeNormalizationThreshold
        << "\n";
    out << "\n[speciation]\n";
    out << "compatibility_threshold = " << formatFloat(metadata.speciationConfig.compatibilityThreshold) << "\n";
    out << "\n[vehicle_physics]\n";
    out << "length = " << formatFloat(metadata.carParams.length) << "\n";
    out << "width = " << formatFloat(metadata.carParams.width) << "\n";
    out << "density = " << formatFloat(metadata.carParams.density) << "\n";
    out << "linear_damping = " << formatFloat(metadata.carParams.linearDamping) << "\n";
    out << "angular_damping = " << formatFloat(metadata.carParams.angularDamping) << "\n";
    out << "engine_force = " << formatFloat(metadata.carParams.engineForce) << "\n";
    out << "rolling_resistance = " << formatFloat(metadata.carParams.rollingResistance) << "\n";
    out << "drag_coefficient = " << formatFloat(metadata.carParams.dragCoefficient) << "\n";
    out << "max_speed = " << formatFloat(metadata.carParams.maxSpeed) << "\n";
    out << "cg_to_front_axle = " << formatFloat(metadata.carParams.cgToFrontAxle) << "\n";
    out << "cg_to_rear_axle = " << formatFloat(metadata.carParams.cgToRearAxle) << "\n";
    out << "max_steer_angle = " << formatFloat(metadata.carParams.maxSteerAngle) << "\n";
    out << "front_cornering_stiffness = " << formatFloat(metadata.carParams.frontCorneringStiffness) << "\n";
    out << "rear_cornering_stiffness = " << formatFloat(metadata.carParams.rearCorneringStiffness) << "\n";
    out << "front_max_tire_force = " << formatFloat(metadata.carParams.frontMaxTireForce) << "\n";
    out << "rear_max_tire_force = " << formatFloat(metadata.carParams.rearMaxTireForce) << "\n";

    if (!out.good())
    {
        throw std::runtime_error("TrainingLogger: failed while writing metadata file: " + path);
    }
}

} // namespace

std::string formatFloat(float value)
{
    if (!std::isfinite(value))
    {
        if (std::isnan(value))
        {
            return "nan";
        }
        return value > 0.0f ? "inf" : "-inf";
    }

    // std::to_chars for floating point (C++17) is defined to never consult
    // locale and to produce the shortest decimal representation that reads
    // back to exactly `value` -- both properties matter here: the former is
    // the whole point (a '.' decimal separator regardless of the process's
    // global locale), and the latter keeps CSV output compact (e.g. "812.4"
    // rather than a long fixed-precision expansion).
    std::array<char, 64> buffer{};
    const auto result = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
    return std::string(buffer.data(), result.ptr);
}

std::string csvHeaderLine()
{
    return "generation,best_fitness,avg_fitness,median_fitness,worst_fitness,avg_adjusted_fitness,"
           "species_count,largest_species_size,smallest_species_size,best_species_historical_fitness,"
           "stagnant_species_excluded,best_progress,avg_progress,laps_completed_count,completion_rate,"
           "best_genome_nodes,best_genome_connections,best_genome_enabled_connections,"
           "avg_genome_nodes,avg_genome_connections,generation_duration_seconds";
}

std::string generationMetricsToCsvRow(const GenerationMetrics& metrics)
{
    std::ostringstream row;
    row << metrics.generation << ',' << formatFloat(metrics.bestFitness) << ',' << formatFloat(metrics.avgFitness) << ','
        << formatFloat(metrics.medianFitness) << ',' << formatFloat(metrics.worstFitness) << ','
        << formatFloat(metrics.avgAdjustedFitness) << ',' << metrics.speciesCount << ',' << metrics.largestSpeciesSize
        << ',' << metrics.smallestSpeciesSize << ',' << formatFloat(metrics.bestSpeciesHistoricalFitness) << ','
        << metrics.stagnantSpeciesExcluded << ',' << formatFloat(metrics.bestProgress) << ','
        << formatFloat(metrics.avgProgress) << ',' << metrics.lapsCompletedCount << ',' << formatFloat(metrics.completionRate)
        << ',' << metrics.bestGenomeNodeCount << ',' << metrics.bestGenomeConnectionGeneCount << ','
        << metrics.bestGenomeEnabledConnectionCount << ',' << formatFloat(metrics.avgGenomeNodeCount) << ','
        << formatFloat(metrics.avgGenomeConnectionGeneCount) << ',' << formatFloat(metrics.generationDurationSeconds);
    return row.str();
}

TrainingLogger::TrainingLogger(const std::string& resultsDir, const RunMetadata& metadata)
{
    const std::filesystem::path dir(resultsDir);
    std::filesystem::create_directories(dir);

    const std::tm localTime = currentLocalTime();
    m_runId = makeUniqueRunId(dir, localTime);

    const std::filesystem::path csvPath = dir / (m_runId + ".csv");
    m_csv.open(csvPath, std::ios::out | std::ios::trunc);
    if (!m_csv.is_open())
    {
        throw std::runtime_error("TrainingLogger: failed to open CSV file for writing: " + csvPath.string());
    }
    m_csvPath = csvPath.string();

    m_csv << csvHeaderLine() << "\n";
    m_csv.flush();
    if (!m_csv.good())
    {
        throw std::runtime_error("TrainingLogger: failed while writing CSV header: " + m_csvPath);
    }

    RunMetadata resolvedMetadata = metadata;
    resolvedMetadata.runId = m_runId;
    resolvedMetadata.startedAtLocal = formatTimestampForDisplay(localTime);

    const std::filesystem::path metadataPath = dir / (m_runId + ".meta.txt");
    writeMetadataFile(metadataPath.string(), resolvedMetadata, csvPath.filename().string());
    m_metadataPath = metadataPath.string();
}

void TrainingLogger::logGeneration(const GenerationMetrics& metrics)
{
    m_csv << generationMetricsToCsvRow(metrics) << "\n";
    // Flushed after every single row -- see the class comment: a long
    // training run that is closed or crashes mid-run must never lose more
    // than the generation currently in progress.
    m_csv.flush();
    if (!m_csv.good())
    {
        throw std::runtime_error("TrainingLogger: failed while writing CSV row to: " + m_csvPath);
    }
}

} // namespace training
