#include "telemetry/HairpinTelemetry.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "raylib.h"

#include "ai/AIController.h"
#include "ai/Observation.h"
#include "ai/neat/Individual.h"
#include "ai/neat/Population.h"
#include "simulation/Car.h"
#include "simulation/TrackProgress.h"
#include "training/TrainingLogger.h" // reuses training::formatFloat -- see its comment

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
        case ai::EvaluationFinishReason::TimeLimit:
            return "timelimit";
        case ai::EvaluationFinishReason::NoProgress:
            return "noprogress";
        case ai::EvaluationFinishReason::InsufficientInitialProgress:
            return "insufficient_initial_progress";
    }
    return "unknown";
}

// Wraps a radian angle into [-pi, pi].
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

std::string twoDigitPad(std::size_t value, int width)
{
    std::ostringstream out;
    out.width(width);
    out.fill('0');
    out << value;
    return out.str();
}

} // namespace

HairpinRegion computeHairpinRegion(const simulation::Track& track)
{
    const simulation::TrackDefinition& def = track.getDefinition();
    if (def.controlPoints.size() <= kHairpinRegionEndControlPoint)
    {
        throw std::invalid_argument(
            "computeHairpinRegion: track has too few control points for the hairpin region indices "
            "this was built against (see kHairpinRegionStartControlPoint/kHairpinRegionEndControlPoint)");
    }

    const std::vector<float>& cumulative = track.getCumulativeDistances();
    const std::size_t centerlineSize = track.getCenterlineSamples().size();
    const std::size_t samplesPerSegment = static_cast<std::size_t>(def.samplesPerSegment);

    const std::size_t startSampleIndex = (kHairpinRegionStartControlPoint * samplesPerSegment) % centerlineSize;
    const std::size_t endSampleIndex = (kHairpinRegionEndControlPoint * samplesPerSegment) % centerlineSize;
    const float totalLength = track.getTotalLength();

    HairpinRegion region;
    region.startLapFraction = cumulative.at(startSampleIndex) / totalLength;
    region.endLapFraction = cumulative.at(endSampleIndex) / totalLength;
    return region;
}

std::string hairpinCsvHeaderLine()
{
    return "generation,individual_index,sim_time,sample_index,"
           "pos_x,pos_y,heading,continuous_progress,best_progress,checkpoint_index,lap_count,alive,finish_reason,"
           "speed,forward_velocity,lateral_velocity,slip_angle,yaw_rate,"
           "steering_cmd,throttle_cmd,brake_cmd,"
           "raw_steering,raw_throttle,raw_brake,"
           "obs_sensor_m60,obs_sensor_m30,obs_sensor_0,obs_sensor_p30,obs_sensor_p60,"
           "obs_speed_norm,obs_forward_vel_norm,obs_lateral_vel_norm,obs_slip_norm,"
           "sensor_m60_px,sensor_m30_px,sensor_0_px,sensor_p30_px,sensor_p60_px,min_forward_wall_distance_px,"
           "front_slip_angle_true,rear_slip_angle_true,front_slip_angle_relaxed,rear_slip_angle_relaxed,"
           "front_grip_utilization,rear_grip_utilization,"
           "track_direction_angle,heading_error,wrong_way";
}

std::string hairpinSampleToCsvRow(const HairpinTelemetrySample& s)
{
    using training::formatFloat;
    std::ostringstream row;
    row << s.generation << ',' << s.individualIndex << ',' << formatFloat(s.simTime) << ',' << s.sampleIndex << ','
        << formatFloat(s.posX) << ',' << formatFloat(s.posY) << ',' << formatFloat(s.heading) << ','
        << formatFloat(s.continuousProgress) << ',' << formatFloat(s.bestProgress) << ',' << s.checkpointIndex << ','
        << s.lapCount << ',' << (s.alive ? 1 : 0) << ',' << finishReasonToString(s.finishReason) << ','
        << formatFloat(s.speed) << ',' << formatFloat(s.forwardVelocity) << ',' << formatFloat(s.lateralVelocity) << ','
        << formatFloat(s.slipAngle) << ',' << formatFloat(s.yawRate) << ',' << formatFloat(s.steeringCmd) << ','
        << formatFloat(s.throttleCmd) << ',' << formatFloat(s.brakeCmd) << ',' << formatFloat(s.rawSteering) << ','
        << formatFloat(s.rawThrottle) << ',' << formatFloat(s.rawBrake) << ',' << formatFloat(s.obsSensorM60) << ','
        << formatFloat(s.obsSensorM30) << ',' << formatFloat(s.obsSensor0) << ',' << formatFloat(s.obsSensorP30) << ','
        << formatFloat(s.obsSensorP60) << ',' << formatFloat(s.obsSpeedNorm) << ',' << formatFloat(s.obsForwardVelNorm)
        << ',' << formatFloat(s.obsLateralVelNorm) << ',' << formatFloat(s.obsSlipNorm) << ',' << formatFloat(s.sensorM60Px)
        << ',' << formatFloat(s.sensorM30Px) << ',' << formatFloat(s.sensor0Px) << ',' << formatFloat(s.sensorP30Px) << ','
        << formatFloat(s.sensorP60Px) << ',' << formatFloat(s.minForwardWallDistancePx) << ','
        << formatFloat(s.frontSlipAngleTrue) << ',' << formatFloat(s.rearSlipAngleTrue) << ','
        << formatFloat(s.frontSlipAngleRelaxed) << ',' << formatFloat(s.rearSlipAngleRelaxed) << ','
        << formatFloat(s.frontGripUtilization) << ',' << formatFloat(s.rearGripUtilization) << ','
        << formatFloat(s.trackDirectionAngle) << ',' << formatFloat(s.headingError) << ',' << (s.wrongWay ? 1 : 0);
    return row.str();
}

HairpinFailureSummary summarizeHairpinFailure(const std::vector<HairpinTelemetrySample>& orderedSamples)
{
    HairpinFailureSummary summary;
    for (const HairpinTelemetrySample& s : orderedSamples)
    {
        if (!summary.hasBrakeOnset && s.brakeCmd > kBrakeOnsetThreshold)
        {
            summary.hasBrakeOnset = true;
            summary.brakeOnsetSpeed = s.speed;
            summary.brakeOnsetSensor0Px = s.sensor0Px;
            summary.brakeOnsetMinForwardWallDistancePx = s.minForwardWallDistancePx;
            summary.brakeOnsetBestProgress = s.bestProgress;
            summary.brakeOnsetThrottle = s.throttleCmd;
            summary.brakeOnsetSteering = s.steeringCmd;
        }
        if (!summary.hasSpinOnset && s.wrongWay)
        {
            summary.hasSpinOnset = true;
            summary.spinOnsetSimTime = s.simTime;
            summary.spinOnsetBodySlipAngle = s.slipAngle;
            summary.spinOnsetYawRate = s.yawRate;
            summary.spinOnsetFrontGripUtilization = s.frontGripUtilization;
            summary.spinOnsetRearGripUtilization = s.rearGripUtilization;
        }
        if (summary.hasBrakeOnset && summary.hasSpinOnset)
        {
            break;
        }
    }
    return summary;
}

HairpinTelemetryRecorder::HairpinTelemetryRecorder(const simulation::Track& track, std::string resultsDir,
                                                     float simulationDt, bool enabled)
    : m_track(track)
    , m_resultsDir(std::move(resultsDir))
    , m_enabled(enabled)
    , m_region(computeHairpinRegion(track))
{
    const float safeDt = std::max(simulationDt, 1e-6f);
    const std::size_t capacity =
        std::max<std::size_t>(1, static_cast<std::size_t>(kRingBufferCapacitySeconds / safeDt));
    m_buffer.resize(capacity);
}

void HairpinTelemetryRecorder::resetForNewGeneration(std::size_t generation)
{
    m_hasSeenGeneration = true;
    m_lastSeenGeneration = generation;
    m_dumpsThisGeneration = 0;
    // Force a fresh Phase-2 pick under the new generation's individuals --
    // whatever was tracked belonged to the generation that just ended.
    m_hasTrackedIndividual = false;
}

bool HairpinTelemetryRecorder::isInHairpinRegion(float lapFractionProgress) const
{
    float frac = std::fmod(lapFractionProgress, 1.0f);
    if (frac < 0.0f)
    {
        frac += 1.0f;
    }

    if (m_region.startLapFraction <= m_region.endLapFraction)
    {
        return frac >= m_region.startLapFraction && frac <= m_region.endLapFraction;
    }
    // Wrap-around region (start > end, i.e. the window straddles the lap
    // seam) -- not the extreme track's case today, but handled for
    // robustness if the control-point indices above ever move near the seam.
    return frac >= m_region.startLapFraction || frac <= m_region.endLapFraction;
}

void HairpinTelemetryRecorder::pushSample(const ai::neat::Individual& individual)
{
    const simulation::Car& car = individual.getCar();
    const simulation::TrackProgress& progress = individual.getProgress();
    const ai::FitnessEvaluator& fitness = individual.getFitnessEvaluator();
    const ai::AIController& controller = individual.getController();
    const simulation::TireDebugInfo& tire = car.getTireDebugInfo();
    const auto& sensors = car.getSensors();
    const ai::Observation& observation = controller.getLastObservation();

    HairpinTelemetrySample sample;
    sample.generation = m_trackedGeneration;
    sample.individualIndex = m_trackedIndividualIndex;
    sample.simTime = fitness.getElapsedTime();
    sample.sampleIndex = m_nextSampleIndex++;

    sample.posX = car.getPosition().x;
    sample.posY = car.getPosition().y;
    sample.heading = car.getHeading();
    sample.continuousProgress = progress.getContinuousProgress();
    sample.bestProgress = progress.getBestProgress();
    sample.checkpointIndex = progress.getExpectedCheckpoint();
    sample.lapCount = progress.getLapCount();
    sample.alive = car.isAlive();
    sample.finishReason = fitness.getFinishReason();

    sample.speed = car.getSpeed();
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

    sample.sensorM60Px = sensors[0].distance;
    sample.sensorM30Px = sensors[1].distance;
    sample.sensor0Px = sensors[2].distance;
    sample.sensorP30Px = sensors[3].distance;
    sample.sensorP60Px = sensors[4].distance;
    sample.minForwardWallDistancePx = std::min({sensors[1].distance, sensors[2].distance, sensors[3].distance});

    sample.frontSlipAngleTrue = tire.frontSlipAngle;
    sample.rearSlipAngleTrue = tire.rearSlipAngle;
    sample.frontSlipAngleRelaxed = tire.frontSlipAngleRelaxed;
    sample.rearSlipAngleRelaxed = tire.rearSlipAngleRelaxed;
    sample.frontGripUtilization = tire.frontGripUtilization;
    sample.rearGripUtilization = tire.rearGripUtilization;

    const Vector2 tangent = progress.getLastProjectionDebugInfo().tangent;
    const bool haveTangent = (tangent.x != 0.0f || tangent.y != 0.0f);
    const float trackDirectionAngle = haveTangent ? std::atan2(tangent.y, tangent.x) : sample.heading;
    sample.trackDirectionAngle = trackDirectionAngle;
    sample.headingError = wrapToPi(sample.heading - trackDirectionAngle);
    sample.wrongWay = std::fabs(sample.headingError) > kWrongWayHeadingErrorThresholdRad;

    const std::size_t capacity = m_buffer.size();
    m_buffer[m_writePos] = sample;
    m_writePos = (m_writePos + 1) % capacity;
    m_filledCount = std::min(m_filledCount + 1, capacity);
}

std::vector<HairpinTelemetrySample> HairpinTelemetryRecorder::getOrderedSamples() const
{
    std::vector<HairpinTelemetrySample> result;
    result.reserve(m_filledCount);
    const std::size_t capacity = m_buffer.size();
    const std::size_t startIndex = (m_filledCount < capacity) ? 0 : m_writePos;
    for (std::size_t i = 0; i < m_filledCount; ++i)
    {
        result.push_back(m_buffer[(startIndex + i) % capacity]);
    }
    return result;
}

void HairpinTelemetryRecorder::startTracking(std::size_t generation, std::size_t individualIndex,
                                              const ai::neat::Individual& individual)
{
    m_hasTrackedIndividual = true;
    m_trackedGeneration = generation;
    m_trackedIndividualIndex = individualIndex;
    m_writePos = 0;
    m_filledCount = 0;
    m_nextSampleIndex = 0;
    pushSample(individual);
}

std::string HairpinTelemetryRecorder::buildCsvPath(std::size_t generation, std::size_t individualIndex,
                                                     ai::EvaluationFinishReason finishReason) const
{
    const std::filesystem::path dir = std::filesystem::path(m_resultsDir) / "telemetry";
    std::filesystem::create_directories(dir);

    const std::string filename = "hairpin_gen" + twoDigitPad(generation, 4) + "_ind" + twoDigitPad(individualIndex, 4) +
                                  "_" + finishReasonToString(finishReason) + ".csv";
    return (dir / filename).string();
}

void HairpinTelemetryRecorder::evaluateAndMaybeDump(std::size_t generation, std::size_t individualIndex,
                                                      ai::EvaluationFinishReason finishReason)
{
    if (finishReason != ai::EvaluationFinishReason::Collision && finishReason != ai::EvaluationFinishReason::NoProgress)
    {
        return; // TimeLimit/InsufficientInitialProgress are not the hairpin failure this instrument targets
    }
    if (m_filledCount == 0)
    {
        return;
    }
    if (m_dumpsThisGeneration >= kMaxDumpsPerGeneration)
    {
        return;
    }

    const std::size_t lastIndex = (m_writePos + m_buffer.size() - 1) % m_buffer.size();
    const HairpinTelemetrySample& lastSample = m_buffer[lastIndex];
    if (!isInHairpinRegion(lastSample.bestProgress))
    {
        return; // this failure happened somewhere else on the track -- not the corner this instrument targets
    }

    const std::vector<HairpinTelemetrySample> ordered = getOrderedSamples();
    const std::string csvPath = buildCsvPath(generation, individualIndex, finishReason);

    std::ofstream csv(csvPath, std::ios::out | std::ios::trunc);
    if (!csv.is_open())
    {
        TraceLog(LOG_WARNING, "HairpinTelemetry: failed to open %s for writing", csvPath.c_str());
        return;
    }
    csv << hairpinCsvHeaderLine() << "\n";
    for (const HairpinTelemetrySample& s : ordered)
    {
        csv << hairpinSampleToCsvRow(s) << "\n";
    }
    csv.flush();
    if (!csv.good())
    {
        TraceLog(LOG_WARNING, "HairpinTelemetry: error while writing %s", csvPath.c_str());
        return;
    }

    const HairpinFailureSummary summary = summarizeHairpinFailure(ordered);

    const std::string summaryPath = csvPath.substr(0, csvPath.size() - 4) + ".summary.txt"; // strip ".csv"
    std::ofstream summaryFile(summaryPath, std::ios::out | std::ios::trunc);
    if (summaryFile.is_open())
    {
        using training::formatFloat;
        summaryFile << "generation = " << generation << "\n";
        summaryFile << "individual_index = " << individualIndex << "\n";
        summaryFile << "finish_reason = " << finishReasonToString(finishReason) << "\n";
        summaryFile << "sample_count = " << ordered.size() << "\n";
        summaryFile << "hairpin_region_lap_fraction = [" << formatFloat(m_region.startLapFraction) << ", "
                    << formatFloat(m_region.endLapFraction) << "]\n";

        summaryFile << "\n[brake_onset]\n";
        if (summary.hasBrakeOnset)
        {
            summaryFile << "speed = " << formatFloat(summary.brakeOnsetSpeed) << "\n";
            summaryFile << "sensor_0_px = " << formatFloat(summary.brakeOnsetSensor0Px) << "\n";
            summaryFile << "min_forward_wall_distance_px = " << formatFloat(summary.brakeOnsetMinForwardWallDistancePx)
                        << "\n";
            summaryFile << "best_progress = " << formatFloat(summary.brakeOnsetBestProgress) << "\n";
            summaryFile << "throttle = " << formatFloat(summary.brakeOnsetThrottle) << "\n";
            summaryFile << "steering = " << formatFloat(summary.brakeOnsetSteering) << "\n";
        }
        else
        {
            summaryFile << "none -- brake never exceeded " << formatFloat(kBrakeOnsetThreshold)
                        << " anywhere in this capture\n";
        }

        summaryFile << "\n[spin_onset]\n";
        if (summary.hasSpinOnset)
        {
            summaryFile << "sim_time = " << formatFloat(summary.spinOnsetSimTime) << "\n";
            summaryFile << "body_slip_angle = " << formatFloat(summary.spinOnsetBodySlipAngle) << "\n";
            summaryFile << "yaw_rate = " << formatFloat(summary.spinOnsetYawRate) << "\n";
            summaryFile << "front_grip_utilization = " << formatFloat(summary.spinOnsetFrontGripUtilization) << "\n";
            summaryFile << "rear_grip_utilization = " << formatFloat(summary.spinOnsetRearGripUtilization) << "\n";
        }
        else
        {
            summaryFile << "none -- car never crossed the wrong-way heading-error threshold in this capture\n";
        }
    }

    TraceLog(LOG_INFO, "HairpinTelemetry: dumped %d samples (gen %d, ind %d, %s) to %s",
             static_cast<int>(ordered.size()), static_cast<int>(generation), static_cast<int>(individualIndex),
             finishReasonToString(finishReason), csvPath.c_str());

    ++m_dumpsThisGeneration;
    ++m_totalDumpCount;
}

void HairpinTelemetryRecorder::update(const ai::neat::Population& population)
{
    if (!m_enabled)
    {
        return;
    }

    const std::size_t generation = population.getGeneration();
    if (!m_hasSeenGeneration || generation != m_lastSeenGeneration)
    {
        resetForNewGeneration(generation);
    }

    // Phase 1: advance (or finish) whatever individual is currently tracked.
    if (m_hasTrackedIndividual)
    {
        if (m_trackedIndividualIndex >= population.size())
        {
            // Defensive only -- population size is fixed for a generation's
            // lifetime and never actually changes mid-generation.
            m_hasTrackedIndividual = false;
        }
        else
        {
            const ai::neat::Individual& individual = population.getIndividual(m_trackedIndividualIndex);
            // Push every live frame AND the exact frame it finishes -- by
            // the time this runs (after population.update()), Individual's
            // own update() has already run this frame, so the finishing
            // frame's Car/TrackProgress/FitnessEvaluator state is already
            // valid and current (see Individual::update()'s ordering).
            pushSample(individual);

            const ai::EvaluationFinishReason reason = individual.getFitnessEvaluator().getFinishReason();
            if (reason != ai::EvaluationFinishReason::None)
            {
                evaluateAndMaybeDump(m_trackedGeneration, m_trackedIndividualIndex, reason);
                m_hasTrackedIndividual = false; // Phase 2 below picks a fresh leader
            }
        }
    }

    // Phase 2: pick the highest-bestProgress ALIVE individual to track, if
    // nobody is currently tracked (either nothing picked yet this
    // generation, or the previous pick just finished above). Ties go to the
    // lowest index -- deterministic.
    if (!m_hasTrackedIndividual)
    {
        bool found = false;
        std::size_t bestIndex = 0;
        float bestProgress = 0.0f;
        for (std::size_t i = 0; i < population.size(); ++i)
        {
            const ai::neat::Individual& candidate = population.getIndividual(i);
            if (candidate.isFinished())
            {
                continue;
            }
            const float candidateProgress = candidate.getProgress().getBestProgress();
            if (!found || candidateProgress > bestProgress)
            {
                found = true;
                bestIndex = i;
                bestProgress = candidateProgress;
            }
        }

        if (found)
        {
            startTracking(generation, bestIndex, population.getIndividual(bestIndex));
        }
    }
}

} // namespace telemetry
