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

// "YYYYMMDD_HHMMSS", second resolution (see makeUniqueRunId()'s suffix fallback).
std::string formatTimestampForFilename(std::tm localTime)
{
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%04d%02d%02d_%02d%02d%02d", localTime.tm_year + 1900, localTime.tm_mon + 1,
                  localTime.tm_mday, localTime.tm_hour, localTime.tm_min, localTime.tm_sec);
    return std::string(buffer);
}

// "YYYY-MM-DD HH:MM:SS", for RunMetadata::startedAtLocal.
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

// First unused "training_run_<timestamp>[-N]" stem, checking both .csv and
// .meta.txt so an earlier run's pair is never partially overwritten.
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

    // Plain "key = value" lines -- a human-readable reproducibility note,
    // not meant for other tooling to parse.
    out << "# Training run metadata\n";
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
    out << "target_lap_count = " << metadata.targetLapCount << "\n";
    out << "safety_timeout_seconds = " << formatFloat(metadata.safetyTimeoutSeconds) << "\n";
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
    out << "target_species_min = " << metadata.speciationConfig.targetSpeciesMin << "\n";
    out << "target_species_max = " << metadata.speciationConfig.targetSpeciesMax << "\n";
    out << "compatibility_threshold_adjustment = "
        << formatFloat(metadata.speciationConfig.compatibilityThresholdAdjustment) << "\n";
    out << "minimum_compatibility_threshold = " << formatFloat(metadata.speciationConfig.minimumCompatibilityThreshold)
        << "\n";
    out << "maximum_compatibility_threshold = " << formatFloat(metadata.speciationConfig.maximumCompatibilityThreshold)
        << "\n";
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
    out << "steer_authority_speeds = ";
    for (std::size_t i = 0; i < metadata.carParams.steerAuthoritySpeeds.size(); ++i)
    {
        out << (i ? "," : "") << formatFloat(metadata.carParams.steerAuthoritySpeeds[i]);
    }
    out << "\n";
    out << "steer_authority_factors = ";
    for (std::size_t i = 0; i < metadata.carParams.steerAuthorityFactors.size(); ++i)
    {
        out << (i ? "," : "") << formatFloat(metadata.carParams.steerAuthorityFactors[i]);
    }
    out << "\n";
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

    // to_chars never consults locale and gives the shortest round-tripping
    // representation (e.g. "812.4", not "812.400000").
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
           "avg_genome_nodes,avg_genome_connections,generation_duration_seconds,"
           "terminated_collision_count,terminated_safety_timeout_count,terminated_no_progress_count,"
           "terminated_slow_start_count,compatibility_threshold,"
           "best_avg_abs_steering_delta,best_steering_reversals_per_second,best_steering_saturation_fraction,"
           "best_mean_abs_steering,best_mean_lateral_accel,best_front_slip_beyond_peak_fraction,"
           "best_lap2plus_average_speed,best_physical_brake_usage_fraction,"
           "best_brake_request_dominant_fraction,best_brake_onset_speed,"
           "terminated_completed_laps_count";
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
        << formatFloat(metrics.avgGenomeConnectionGeneCount) << ',' << formatFloat(metrics.generationDurationSeconds)
        << ',' << metrics.terminatedCollisionCount << ',' << metrics.terminatedSafetyTimeoutCount << ','
        << metrics.terminatedNoProgressCount << ',' << metrics.terminatedSlowStartCount << ','
        << formatFloat(metrics.compatibilityThresholdUsed) << ','
        << formatFloat(metrics.bestDriving.averageAbsSteeringDelta) << ','
        << formatFloat(metrics.bestDriving.steeringReversalsPerSecond) << ','
        << formatFloat(metrics.bestDriving.steeringSaturationFraction) << ','
        << formatFloat(metrics.bestDriving.meanAbsSteering) << ','
        << formatFloat(metrics.bestDriving.meanLateralAcceleration) << ','
        << formatFloat(metrics.bestDriving.frontSlipBeyondPeakFraction) << ','
        << formatFloat(metrics.bestDriving.lap2PlusAverageSpeed) << ','
        << formatFloat(metrics.bestDriving.physicalBrakeUsageFraction) << ','
        << formatFloat(metrics.bestDriving.brakeRequestDominantFraction) << ','
        << formatFloat(metrics.bestDriving.brakeOnsetSpeed) << ',' << metrics.terminatedCompletedLapsCount;
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
    m_csv.flush(); // a crash mid-run must never lose more than the in-progress row
    if (!m_csv.good())
    {
        throw std::runtime_error("TrainingLogger: failed while writing CSV row to: " + m_csvPath);
    }
}

} // namespace training
