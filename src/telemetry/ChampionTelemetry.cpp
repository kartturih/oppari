#include "telemetry/ChampionTelemetry.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "raylib.h"

#include "ai/AIController.h"
#include "ai/Observation.h"
#include "ai/neat/Individual.h"
#include "ai/neat/Population.h"
#include "simulation/Car.h"
#include "simulation/TrackProgress.h"
#include "telemetry/HairpinTelemetry.h" // reuses kWrongWayHeadingErrorThresholdRad -- see its comment
#include "training/TrainingLogger.h"    // reuses training::formatFloat -- see its comment

namespace telemetry
{

namespace
{

const char* finishReasonToString(ai::EvaluationFinishReason reason)
{
    switch (reason)
    {
        case ai::EvaluationFinishReason::None:
            return "none";
        case ai::EvaluationFinishReason::Collision:
            return "collision";
        case ai::EvaluationFinishReason::CompletedLaps:
            return "completed_laps";
        case ai::EvaluationFinishReason::SafetyTimeout:
            return "safety_timeout";
        case ai::EvaluationFinishReason::NoProgress:
            return "noprogress";
        case ai::EvaluationFinishReason::InsufficientInitialProgress:
            return "insufficient_initial_progress";
    }
    return "unknown";
}

// Wraps a radian angle into [-pi, pi]. Same formula as HairpinTelemetry.cpp's
// own (file-local) helper -- kept duplicated rather than shared across
// telemetry files, matching this codebase's existing per-file-helper pattern.
float wrapToPi(float angle)
{
    while (angle > PI)
    {
        angle -= 2.0f * PI;
    }
    while (angle < -PI)
    {
        angle += 2.0f * PI;
    }
    return angle;
}

std::string zeroPad(std::size_t value, int width)
{
    std::ostringstream out;
    out.width(width);
    out.fill('0');
    out << value;
    return out.str();
}

float mean(const std::vector<float>& values)
{
    if (values.empty())
    {
        return 0.0f;
    }
    double sum = 0.0;
    for (float v : values)
    {
        sum += v;
    }
    return static_cast<float>(sum / static_cast<double>(values.size()));
}

float median(std::vector<float> values) // takes by value -- sorts its own copy
{
    if (values.empty())
    {
        return 0.0f;
    }
    std::sort(values.begin(), values.end());
    const std::size_t n = values.size();
    if (n % 2 == 1)
    {
        return values[n / 2];
    }
    return 0.5f * (values[n / 2 - 1] + values[n / 2]);
}

} // namespace

std::string championCsvHeaderLine()
{
    return "generation,individual_index,sim_time,sample_index,"
           "pos_x,pos_y,heading,continuous_progress,best_progress,lap_count,checkpoint_index,total_checkpoints_passed,"
           "alive,finished,finish_reason,"
           "speed,normalized_speed,forward_velocity,lateral_velocity,slip_angle,yaw_rate,"
           "steering_cmd,throttle_cmd,brake_cmd,"
           "raw_steering,raw_throttle,raw_brake,"
           "obs_sensor_m60,obs_sensor_m30,obs_sensor_0,obs_sensor_p30,obs_sensor_p60,"
           "obs_speed_norm,obs_forward_vel_norm,obs_lateral_vel_norm,obs_slip_norm,"
           "sensor_m60_px,sensor_m30_px,sensor_0_px,sensor_p30_px,sensor_p60_px,"
           "track_direction_angle,heading_error,wrong_way,"
           "current_lap_elapsed_time,last_lap_time,best_lap_time,"
           "actual_steer_angle,"
           "obs_actual_steer_norm,obs_yaw_rate_norm,obs_heading_error_norm,"
           "obs_preview_heading_error_120_norm,obs_preview_heading_error_300_norm,"
           "steering_authority_factor,effective_max_steer_angle";
}

std::string championSampleToCsvRow(const ChampionTelemetrySample& s)
{
    using training::formatFloat;
    std::ostringstream row;
    row << s.generation << ',' << s.individualIndex << ',' << formatFloat(s.simTime) << ',' << s.sampleIndex << ','
        << formatFloat(s.posX) << ',' << formatFloat(s.posY) << ',' << formatFloat(s.heading) << ','
        << formatFloat(s.continuousProgress) << ',' << formatFloat(s.bestProgress) << ',' << s.lapCount << ','
        << s.checkpointIndex << ',' << s.totalCheckpointsPassed << ',' << (s.alive ? 1 : 0) << ',' << (s.finished ? 1 : 0)
        << ',' << finishReasonToString(s.finishReason) << ',' << formatFloat(s.speed) << ',' << formatFloat(s.normalizedSpeed)
        << ',' << formatFloat(s.forwardVelocity) << ',' << formatFloat(s.lateralVelocity) << ',' << formatFloat(s.slipAngle)
        << ',' << formatFloat(s.yawRate) << ',' << formatFloat(s.steeringCmd) << ',' << formatFloat(s.throttleCmd) << ','
        << formatFloat(s.brakeCmd) << ',' << formatFloat(s.rawSteering) << ',' << formatFloat(s.rawThrottle) << ','
        << formatFloat(s.rawBrake) << ',' << formatFloat(s.obsSensorM60) << ',' << formatFloat(s.obsSensorM30) << ','
        << formatFloat(s.obsSensor0) << ',' << formatFloat(s.obsSensorP30) << ',' << formatFloat(s.obsSensorP60) << ','
        << formatFloat(s.obsSpeedNorm) << ',' << formatFloat(s.obsForwardVelNorm) << ',' << formatFloat(s.obsLateralVelNorm)
        << ',' << formatFloat(s.obsSlipNorm) << ',' << formatFloat(s.sensorM60Px) << ',' << formatFloat(s.sensorM30Px) << ','
        << formatFloat(s.sensor0Px) << ',' << formatFloat(s.sensorP30Px) << ',' << formatFloat(s.sensorP60Px) << ','
        << formatFloat(s.trackDirectionAngle) << ',' << formatFloat(s.headingError) << ',' << (s.wrongWay ? 1 : 0) << ','
        << formatFloat(s.currentLapElapsedTime) << ',' << formatFloat(s.lastLapTime) << ',' << formatFloat(s.bestLapTime)
        << ',' << formatFloat(s.actualSteerAngle) << ',' << formatFloat(s.obsActualSteerNorm) << ','
        << formatFloat(s.obsYawRateNorm) << ',' << formatFloat(s.obsHeadingErrorNorm) << ','
        << formatFloat(s.obsPreviewHeadingError120Norm) << ',' << formatFloat(s.obsPreviewHeadingError300Norm) << ','
        << formatFloat(s.steeringAuthorityFactor) << ',' << formatFloat(s.effectiveMaxSteerAngle);
    return row.str();
}

std::string championSummaryCsvHeaderLine()
{
    return "generation,individual_index,final_fitness,best_progress,lap_count,total_evaluation_time,finish_reason,"
           "average_speed,max_speed,minimum_moving_speed,median_speed,average_forward_speed,"
           "percent_time_below_50,percent_time_below_100,percent_time_below_200,"
           "longest_continuous_low_speed_duration_seconds,"
           "average_throttle,percent_time_throttle_above_half,average_brake,percent_time_brake_above_005,"
           "percent_time_brake_above_025,percent_time_throttle_and_brake_overlap,average_abs_steering,"
           "completed_laps,best_lap_time,average_completed_lap_time,progress_per_second,"
           "evaluator_avg_abs_steering_delta,steering_reversals_per_second,steering_saturation_fraction,"
           "mean_abs_steering_60hz,mean_lateral_accel,front_slip_beyond_peak_fraction,"
           "lap2plus_average_speed,physical_brake_usage_fraction,"
           "brake_request_dominant_fraction,throttle_request_min,throttle_request_max,"
           "brake_request_min,brake_request_max,brake_onset_speed,"
           "brake_onset_preview_120,brake_onset_preview_300";
}

std::string championSummaryRowToCsvRow(const ChampionSummaryRow& r)
{
    using training::formatFloat;
    std::ostringstream row;
    row << r.generation << ',' << r.individualIndex << ',' << formatFloat(r.finalFitness) << ','
        << formatFloat(r.bestProgress) << ',' << r.lapCount << ',' << formatFloat(r.totalEvaluationTime) << ','
        << finishReasonToString(r.finishReason) << ',' << formatFloat(r.averageSpeed) << ',' << formatFloat(r.maxSpeed)
        << ',' << formatFloat(r.minimumMovingSpeed) << ',' << formatFloat(r.medianSpeed) << ','
        << formatFloat(r.averageForwardSpeed) << ',' << formatFloat(r.percentTimeBelow50) << ','
        << formatFloat(r.percentTimeBelow100) << ',' << formatFloat(r.percentTimeBelow200) << ','
        << formatFloat(r.longestContinuousLowSpeedDurationSeconds) << ',' << formatFloat(r.averageThrottle) << ','
        << formatFloat(r.percentTimeThrottleAboveHalf) << ',' << formatFloat(r.averageBrake) << ','
        << formatFloat(r.percentTimeBrakeAbove005) << ',' << formatFloat(r.percentTimeBrakeAbove025) << ','
        << formatFloat(r.percentTimeThrottleAndBrakeOverlap) << ',' << formatFloat(r.averageAbsSteering) << ','
        << r.completedLaps << ',' << formatFloat(r.bestLapTime) << ',' << formatFloat(r.averageCompletedLapTime) << ','
        << formatFloat(r.progressPerSecond) << ',' << formatFloat(r.driving.averageAbsSteeringDelta) << ','
        << formatFloat(r.driving.steeringReversalsPerSecond) << ',' << formatFloat(r.driving.steeringSaturationFraction)
        << ',' << formatFloat(r.driving.meanAbsSteering) << ',' << formatFloat(r.driving.meanLateralAcceleration) << ','
        << formatFloat(r.driving.frontSlipBeyondPeakFraction) << ',' << formatFloat(r.driving.lap2PlusAverageSpeed)
        << ',' << formatFloat(r.driving.physicalBrakeUsageFraction) << ','
        << formatFloat(r.driving.brakeRequestDominantFraction) << ',' << formatFloat(r.driving.throttleRequestMin) << ','
        << formatFloat(r.driving.throttleRequestMax) << ',' << formatFloat(r.driving.brakeRequestMin) << ','
        << formatFloat(r.driving.brakeRequestMax) << ',' << formatFloat(r.driving.brakeOnsetSpeed) << ','
        << formatFloat(r.driving.brakeOnsetPreview120) << ',' << formatFloat(r.driving.brakeOnsetPreview300);
    return row.str();
}

ChampionSummaryRow computeChampionSummary(std::size_t generation, std::size_t individualIndex,
                                           const std::vector<ChampionTelemetrySample>& orderedSamples,
                                           float finalFitness, ai::EvaluationFinishReason finishReason,
                                           const ai::DrivingDiagnosticsSummary* driving)
{
    ChampionSummaryRow row;
    row.generation = generation;
    row.individualIndex = individualIndex;
    row.finalFitness = finalFitness;
    row.finishReason = finishReason;
    if (driving != nullptr)
    {
        row.driving = *driving;
    }

    if (orderedSamples.empty())
    {
        return row;
    }

    row.bestProgress = orderedSamples.back().bestProgress; // monotonic -- last sample holds the max
    row.lapCount = orderedSamples.back().lapCount;
    row.totalEvaluationTime = orderedSamples.back().simTime;

    std::vector<float> speeds;
    std::vector<float> movingSpeeds;
    speeds.reserve(orderedSamples.size());
    std::vector<float> forwardSpeeds;
    forwardSpeeds.reserve(orderedSamples.size());
    std::vector<float> throttles;
    throttles.reserve(orderedSamples.size());
    std::vector<float> brakes;
    brakes.reserve(orderedSamples.size());
    std::vector<float> absSteerings;
    absSteerings.reserve(orderedSamples.size());

    float maxSpeed = orderedSamples.front().speed;
    for (const ChampionTelemetrySample& s : orderedSamples)
    {
        speeds.push_back(s.speed);
        forwardSpeeds.push_back(s.forwardVelocity);
        throttles.push_back(s.throttleCmd);
        brakes.push_back(s.brakeCmd);
        absSteerings.push_back(std::fabs(s.steeringCmd));
        maxSpeed = std::max(maxSpeed, s.speed);
        if (s.speed > kMovingSpeedThresholdPxPerSec)
        {
            movingSpeeds.push_back(s.speed);
        }
    }

    row.averageSpeed = mean(speeds);
    row.maxSpeed = maxSpeed;
    row.minimumMovingSpeed = movingSpeeds.empty() ? 0.0f : *std::min_element(movingSpeeds.begin(), movingSpeeds.end());
    row.medianSpeed = median(speeds);
    row.averageForwardSpeed = mean(forwardSpeeds);
    row.averageThrottle = mean(throttles);
    row.averageBrake = mean(brakes);
    row.averageAbsSteering = mean(absSteerings);

    // Duration-weighted pass: each sample i's state is treated as constant
    // over [simTime[i], simTime[i+1]) -- the interval it actually held
    // during the capture. The final sample contributes no interval (nothing
    // recorded after it), matching totalEvaluationTime = last sample's simTime.
    float timeBelow50 = 0.0f;
    float timeBelow100 = 0.0f;
    float timeBelow200 = 0.0f;
    float timeThrottleAboveHalf = 0.0f;
    float timeBrakeAbove005 = 0.0f;
    float timeBrakeAbove025 = 0.0f;
    float timeThrottleAndBrakeOverlap = 0.0f;
    float longestLowSpeedRun = 0.0f;
    float currentLowSpeedRun = 0.0f;
    for (std::size_t i = 0; i + 1 < orderedSamples.size(); ++i)
    {
        const ChampionTelemetrySample& s = orderedSamples[i];
        const float dt = std::max(0.0f, orderedSamples[i + 1].simTime - s.simTime);

        if (s.speed < 50.0f)
        {
            timeBelow50 += dt;
        }
        if (s.speed < 100.0f)
        {
            timeBelow100 += dt;
        }
        if (s.speed < 200.0f)
        {
            timeBelow200 += dt;
        }
        if (s.speed < 100.0f)
        {
            currentLowSpeedRun += dt;
            longestLowSpeedRun = std::max(longestLowSpeedRun, currentLowSpeedRun);
        }
        else
        {
            currentLowSpeedRun = 0.0f;
        }

        if (s.throttleCmd > 0.5f)
        {
            timeThrottleAboveHalf += dt;
        }
        if (s.brakeCmd > 0.05f)
        {
            timeBrakeAbove005 += dt;
        }
        if (s.brakeCmd > 0.25f)
        {
            timeBrakeAbove025 += dt;
        }
        if (s.throttleCmd > 0.5f && s.brakeCmd > 0.05f)
        {
            timeThrottleAndBrakeOverlap += dt;
        }
    }

    const float totalDuration = row.totalEvaluationTime;
    if (totalDuration > 0.0f)
    {
        row.percentTimeBelow50 = 100.0f * timeBelow50 / totalDuration;
        row.percentTimeBelow100 = 100.0f * timeBelow100 / totalDuration;
        row.percentTimeBelow200 = 100.0f * timeBelow200 / totalDuration;
        row.percentTimeThrottleAboveHalf = 100.0f * timeThrottleAboveHalf / totalDuration;
        row.percentTimeBrakeAbove005 = 100.0f * timeBrakeAbove005 / totalDuration;
        row.percentTimeBrakeAbove025 = 100.0f * timeBrakeAbove025 / totalDuration;
        row.percentTimeThrottleAndBrakeOverlap = 100.0f * timeThrottleAndBrakeOverlap / totalDuration;
    }
    row.longestContinuousLowSpeedDurationSeconds = longestLowSpeedRun;

    // Lap-time aggregation: a lapCount increase between consecutive samples
    // marks one completed lap, whose time is that sample's lastLapTime
    // (FitnessEvaluator overwrites lastLapTime exactly when a lap
    // completes -- see FitnessEvaluator.h). bestLapTime is read directly
    // from the final sample (monotonically improving, so it's always
    // already the true best by the end of the buffer).
    std::vector<float> completedLapTimes;
    int previousLapCount = 0;
    for (const ChampionTelemetrySample& s : orderedSamples)
    {
        if (s.lapCount > previousLapCount)
        {
            completedLapTimes.push_back(s.lastLapTime);
            previousLapCount = s.lapCount;
        }
    }
    row.completedLaps = previousLapCount;
    row.bestLapTime = orderedSamples.back().bestLapTime;
    row.averageCompletedLapTime = completedLapTimes.empty() ? 0.0f : mean(completedLapTimes);

    row.progressPerSecond = (totalDuration > 0.0f) ? (row.bestProgress / totalDuration) : 0.0f;

    return row;
}

std::string championGenomeToText(const ai::neat::Genome& genome)
{
    std::ostringstream out;
    for (const ai::neat::NodeGene& node : genome.nodes())
    {
        out << "N " << node.getId() << ' ' << static_cast<int>(node.getType()) << '\n';
    }
    for (const ai::neat::ConnectionGene& connection : genome.connections())
    {
        out << "C " << connection.getSourceId() << ' ' << connection.getTargetId() << ' '
            << training::formatFloat(connection.getWeight()) << ' ' << (connection.isEnabled() ? 1 : 0) << ' '
            << connection.getInnovationNumber() << '\n';
    }
    return out.str();
}

std::string championCsvFileName(std::size_t generation)
{
    return "champion_gen" + zeroPad(generation, 4) + ".csv";
}

std::string championGenomeFileName(std::size_t generation)
{
    return "champion_gen" + zeroPad(generation, 4) + "_genome.txt";
}

ChampionTelemetryRecorder::ChampionTelemetryRecorder(std::string resultsDir, float simulationDt, bool enabled,
                                                       std::vector<std::size_t> targetGenerations)
    : m_resultsDir(std::move(resultsDir))
    , m_enabled(enabled)
    , m_targetGenerations(std::move(targetGenerations))
    , m_simulationDt(std::max(simulationDt, 1e-6f))
{
}

bool ChampionTelemetryRecorder::isTargetGeneration(std::size_t generation) const
{
    return std::find(m_targetGenerations.begin(), m_targetGenerations.end(), generation) != m_targetGenerations.end();
}

void ChampionTelemetryRecorder::beginCapture(std::size_t generation, std::size_t individualCount)
{
    m_isCapturing = true;
    m_capturingGeneration = generation;
    m_stepIndexInGeneration = 0;

    // kChampionTelemetryReserveSeconds worth of samples at the configured
    // interval, plus slack for the extra per-individual finish sample -- a
    // reservation only (never a hard cap; push_back still grows it up to the
    // kSafetyTimeoutSeconds failsafe if an evaluation runs long).
    const std::size_t reserveHint = static_cast<std::size_t>(
        kChampionTelemetryReserveSeconds / (m_simulationDt * static_cast<float>(kChampionTelemetrySampleInterval))) + 16;

    m_buffers.assign(individualCount, {});
    for (std::vector<ChampionTelemetrySample>& buffer : m_buffers)
    {
        buffer.reserve(reserveHint);
    }
    m_recordedFinish.assign(individualCount, false);
}

void ChampionTelemetryRecorder::sampleIndividual(std::size_t individualIndex, const ai::neat::Individual& individual)
{
    const simulation::Car& car = individual.getCar();
    const simulation::TrackProgress& progress = individual.getProgress();
    const ai::FitnessEvaluator& fitness = individual.getFitnessEvaluator();
    const ai::AIController& controller = individual.getController();
    const simulation::TireDebugInfo& tire = car.getTireDebugInfo();
    const auto& sensors = car.getSensors();
    const ai::Observation& observation = controller.getLastObservation();

    ChampionTelemetrySample sample;
    sample.generation = m_capturingGeneration;
    sample.individualIndex = individualIndex;
    sample.simTime = fitness.getElapsedTime();
    sample.sampleIndex = m_buffers[individualIndex].size();

    sample.posX = car.getPosition().x;
    sample.posY = car.getPosition().y;
    sample.heading = car.getHeading();
    sample.continuousProgress = progress.getContinuousProgress();
    sample.bestProgress = progress.getBestProgress();
    sample.lapCount = progress.getLapCount();
    sample.checkpointIndex = progress.getExpectedCheckpoint();
    sample.totalCheckpointsPassed = progress.getTotalCheckpointsPassed();
    sample.alive = car.isAlive();
    sample.finished = individual.isFinished();
    sample.finishReason = fitness.getFinishReason();

    sample.speed = car.getSpeed();
    const float maxSpeed = car.getMaxSpeed();
    sample.normalizedSpeed = (maxSpeed > 0.0f) ? (sample.speed / maxSpeed) : 0.0f;
    sample.forwardVelocity = car.getForwardVelocity();
    sample.lateralVelocity = car.getLateralVelocity();
    sample.slipAngle = car.getSlipAngle();
    sample.yawRate = tire.yawRate;

    sample.steeringCmd = tire.steeringInput;
    sample.throttleCmd = tire.throttleInput;
    sample.brakeCmd = tire.brakeInput;

    sample.rawSteering = controller.getRawSteeringOutput();
    sample.rawThrottle = controller.getRawThrottleOutput();
    sample.rawBrake = controller.getRawBrakeOutput();

    sample.obsSensorM60 = observation.values[0];
    sample.obsSensorM30 = observation.values[1];
    sample.obsSensor0 = observation.values[2];
    sample.obsSensorP30 = observation.values[3];
    sample.obsSensorP60 = observation.values[4];
    sample.obsSpeedNorm = observation.values[5];
    sample.obsForwardVelNorm = observation.values[6];
    sample.obsLateralVelNorm = observation.values[7];
    sample.obsSlipNorm = observation.values[8];
    sample.obsActualSteerNorm = observation.values[ai::kActualSteerObservationIndex];
    sample.obsYawRateNorm = observation.values[ai::kYawRateObservationIndex];
    sample.obsHeadingErrorNorm = observation.values[ai::kHeadingErrorObservationIndex];
    sample.obsPreviewHeadingError120Norm = observation.values[ai::kPreviewNearObservationIndex];
    sample.obsPreviewHeadingError300Norm = observation.values[ai::kPreviewFarObservationIndex];
    sample.steeringAuthorityFactor = tire.steeringAuthority;
    sample.effectiveMaxSteerAngle = tire.effectiveMaxSteerAngle;

    sample.sensorM60Px = sensors[0].distance;
    sample.sensorM30Px = sensors[1].distance;
    sample.sensor0Px = sensors[2].distance;
    sample.sensorP30Px = sensors[3].distance;
    sample.sensorP60Px = sensors[4].distance;

    const Vector2 tangent = progress.getLastProjectionDebugInfo().tangent;
    const bool haveTangent = (tangent.x != 0.0f || tangent.y != 0.0f);
    const float trackDirectionAngle = haveTangent ? std::atan2(tangent.y, tangent.x) : sample.heading;
    sample.trackDirectionAngle = trackDirectionAngle;
    sample.headingError = wrapToPi(sample.heading - trackDirectionAngle);
    sample.wrongWay = std::fabs(sample.headingError) > kWrongWayHeadingErrorThresholdRad;

    sample.currentLapElapsedTime = fitness.getCurrentLapElapsedTime();
    sample.lastLapTime = fitness.getLastLapTime();
    sample.bestLapTime = fitness.getBestLapTime();

    sample.actualSteerAngle = tire.steeringAngle;

    m_buffers[individualIndex].push_back(sample);
}

std::string ChampionTelemetryRecorder::buildCsvPath(std::size_t generation) const
{
    const std::filesystem::path dir = std::filesystem::path(m_resultsDir) / "telemetry" / "champions";
    std::filesystem::create_directories(dir);
    return (dir / championCsvFileName(generation)).string();
}

std::string ChampionTelemetryRecorder::buildGenomePath(std::size_t generation) const
{
    const std::filesystem::path dir = std::filesystem::path(m_resultsDir) / "telemetry" / "champions";
    std::filesystem::create_directories(dir);
    return (dir / championGenomeFileName(generation)).string();
}

void ChampionTelemetryRecorder::writeSummaryCsv() const
{
    const std::filesystem::path dir = std::filesystem::path(m_resultsDir) / "telemetry" / "champions";
    std::filesystem::create_directories(dir);
    const std::filesystem::path path = dir / "champion_summary.csv";

    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out.is_open())
    {
        TraceLog(LOG_WARNING, "ChampionTelemetry: failed to open %s for writing", path.string().c_str());
        return;
    }
    out << championSummaryCsvHeaderLine() << "\n";
    for (const ChampionSummaryRow& row : m_summaryRows)
    {
        out << championSummaryRowToCsvRow(row) << "\n";
    }
    out.flush();
    if (!out.good())
    {
        TraceLog(LOG_WARNING, "ChampionTelemetry: error while writing %s", path.string().c_str());
    }
}

void ChampionTelemetryRecorder::finishCapture(const ai::neat::Population& population)
{
    const std::size_t bestIndex = population.getBestIndividualIndex();
    const ai::neat::Individual& bestIndividual = population.getIndividual(bestIndex);
    const std::vector<ChampionTelemetrySample>& samples = m_buffers[bestIndex];

    const std::string csvPath = buildCsvPath(m_capturingGeneration);
    std::ofstream csv(csvPath, std::ios::out | std::ios::trunc);
    if (!csv.is_open())
    {
        TraceLog(LOG_WARNING, "ChampionTelemetry: failed to open %s for writing", csvPath.c_str());
    }
    else
    {
        csv << championCsvHeaderLine() << "\n";
        for (const ChampionTelemetrySample& s : samples)
        {
            csv << championSampleToCsvRow(s) << "\n";
        }
        csv.flush();
        if (!csv.good())
        {
            TraceLog(LOG_WARNING, "ChampionTelemetry: error while writing %s", csvPath.c_str());
        }
        else
        {
            TraceLog(LOG_INFO, "ChampionTelemetry: captured champion gen %d (individual %d, %d samples) -> %s",
                     static_cast<int>(m_capturingGeneration), static_cast<int>(bestIndex),
                     static_cast<int>(samples.size()), csvPath.c_str());
        }
    }

    // The champion's genome, as plain text next to its CSV, so it can be
    // analyzed/replayed later without re-running training. Write-only: a
    // failure here is only logged, and nothing in training reads it back.
    {
        const std::string genomePath = buildGenomePath(m_capturingGeneration);
        std::ofstream genomeFile(genomePath, std::ios::out | std::ios::trunc);
        if (!genomeFile.is_open())
        {
            TraceLog(LOG_WARNING, "ChampionTelemetry: failed to open %s for writing", genomePath.c_str());
        }
        else
        {
            genomeFile << "# Champion genome -- generation " << m_capturingGeneration << ", individual " << bestIndex
                       << ", fitness " << training::formatFloat(bestIndividual.getFitness()) << "\n"
                       << "# N <nodeId> <nodeType: 0=Input 1=Bias 2=Hidden 3=Output>\n"
                       << "# C <sourceId> <targetId> <weight> <enabled> <innovation>\n"
                       << championGenomeToText(bestIndividual.getGenome());
            genomeFile.flush();
            if (!genomeFile.good())
            {
                TraceLog(LOG_WARNING, "ChampionTelemetry: error while writing %s", genomePath.c_str());
            }
        }
    }

    const ai::DrivingDiagnosticsSummary drivingSummary = bestIndividual.getDrivingSummary();
    m_summaryRows.push_back(computeChampionSummary(m_capturingGeneration, bestIndex, samples, bestIndividual.getFitness(),
                                                     bestIndividual.getFitnessEvaluator().getFinishReason(),
                                                     &drivingSummary));
    writeSummaryCsv();

    ++m_capturedGenerations;
    m_buffers.clear();
    m_recordedFinish.clear();
}

void ChampionTelemetryRecorder::onPopulationStep(const ai::neat::Population& population, bool generationJustFinished)
{
    if (!m_enabled)
    {
        return;
    }

    const std::size_t generation = population.getGeneration();

    if (!m_isCapturing)
    {
        if (!isTargetGeneration(generation))
        {
            return; // not a generation we care about -- cheapest possible path
        }
        beginCapture(generation, population.size());
    }

    const bool onSampleInterval = (m_stepIndexInGeneration % kChampionTelemetrySampleInterval) == 0;
    for (std::size_t i = 0; i < population.size(); ++i)
    {
        if (m_recordedFinish[i])
        {
            continue; // already have this individual's final sample -- stop
        }
        const ai::neat::Individual& individual = population.getIndividual(i);
        const bool justFinished = individual.isFinished();
        if (onSampleInterval || justFinished)
        {
            sampleIndividual(i, individual);
        }
        if (justFinished)
        {
            m_recordedFinish[i] = true;
        }
    }
    ++m_stepIndexInGeneration;

    if (generationJustFinished)
    {
        finishCapture(population);
        m_isCapturing = false;
    }
}

} // namespace telemetry
