#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "ai/FitnessEvaluator.h"
#include "simulation/Track.h"

// One-off debugging instrument for the recurring hairpin-spin failure (see
// the investigation this was built for): captures dense per-frame telemetry
// for a single individual at a time -- never the whole population -- and
// dumps it to CSV only when that individual fails (Collision/NoProgress)
// with bestProgress inside the track's first hairpin. Pure observer: reads
// Population/Individual/Car/TrackProgress/FitnessEvaluator/AIController
// through their existing const getters only, never writes into any of them,
// never advances RNG, never feeds anything back into control or fitness.
// Entirely separate from training::TrainingLogger/GenerationMetrics (which
// keep logging the same per-generation CSV they always have, unmodified).
namespace ai::neat
{
class Population;
class Individual;
} // namespace ai::neat

namespace telemetry
{

// Single point of control for this whole feature -- flip to false to fully
// disable it in main.cpp (HairpinTelemetryRecorder's default constructor
// argument reads this) without touching anything else. Also directly
// constructor-overridable (see HairpinTelemetryRecorder's `enabled` param)
// so verification can exercise the disabled path without recompiling.
inline constexpr bool kHairpinTelemetryEnabled = true;

// [start, end) lap-fraction window identifying the track's first hairpin,
// derived from the extreme track's own control points/centerline rather
// than a hardcoded world-space rectangle -- see computeHairpinRegion().
struct HairpinRegion
{
    float startLapFraction = 0.0f;
    float endLapFraction = 0.0f;
};

// Control points P2 (index 2) and P6 (index 6) bracket the extreme track's
// first hairpin -- see Track.cpp's createExtremeTrackDefinition() comments
// ("P4 -- top-left corner apex"). Track::buildCenterline() lays samples
// down segment_i = [i*samplesPerSegment, (i+1)*samplesPerSegment), so
// control point i always lands exactly on centerline sample
// i*samplesPerSegment (t=0 of that segment) regardless of samplesPerSegment
// -- padding one control point on each side (P1..P7) so a car braking just
// before P2 or spinning/crashing just past P6 still counts as "in the
// hairpin". Only meaningful for a track built the same way (Catmull-Rom
// over controlPoints, closed loop); see computeHairpinRegion()'s exception.
constexpr std::size_t kHairpinRegionStartControlPoint = 1;
constexpr std::size_t kHairpinRegionEndControlPoint = 7;

// Computes the [start,end) lap-fraction window from the track's actual
// centerline/arc-length tables (Track::getCumulativeDistances()/
// getTotalLength()) -- robust to any future re-tracing of the extreme
// track's control points, since it never hardcodes pixel coordinates or a
// fraction literal. Throws std::invalid_argument if the track has fewer
// than kHairpinRegionEndControlPoint+1 control points (this feature is only
// meaningful for the extreme track it was built against).
HairpinRegion computeHairpinRegion(const simulation::Track& track);

// One frame's worth of captured state for the tracked individual. Field
// order here is the authoritative CSV column order (see hairpinCsvHeaderLine()).
struct HairpinTelemetrySample
{
    std::size_t generation = 0;
    std::size_t individualIndex = 0;
    float simTime = 0.0f;
    std::size_t sampleIndex = 0;

    float posX = 0.0f;
    float posY = 0.0f;
    float heading = 0.0f;
    float continuousProgress = 0.0f;
    float bestProgress = 0.0f;
    int checkpointIndex = 0;
    int lapCount = 0;
    bool alive = true;
    ai::EvaluationFinishReason finishReason = ai::EvaluationFinishReason::None;

    float speed = 0.0f;
    float forwardVelocity = 0.0f;
    float lateralVelocity = 0.0f;
    float slipAngle = 0.0f;
    float yawRate = 0.0f;

    float steeringCmd = 0.0f;
    float throttleCmd = 0.0f;
    float brakeCmd = 0.0f;

    float rawSteering = 0.0f;
    float rawThrottle = 0.0f;
    float rawBrake = 0.0f;

    float obsSensorM60 = 0.0f;
    float obsSensorM30 = 0.0f;
    float obsSensor0 = 0.0f;
    float obsSensorP30 = 0.0f;
    float obsSensorP60 = 0.0f;
    float obsSpeedNorm = 0.0f;
    float obsForwardVelNorm = 0.0f;
    float obsLateralVelNorm = 0.0f;
    float obsSlipNorm = 0.0f;

    // Raw physical distances (px), not normalized -- see Observation.cpp for
    // the [0,1] versions above. minForwardWallDistancePx is
    // min(sensor[-30], sensor[0], sensor[+30]) in px, the "how much room is
    // actually left" number requested for comparing against braking distance.
    float sensorM60Px = 0.0f;
    float sensorM30Px = 0.0f;
    float sensor0Px = 0.0f;
    float sensorP30Px = 0.0f;
    float sensorP60Px = 0.0f;
    float minForwardWallDistancePx = 0.0f;

    float frontSlipAngleTrue = 0.0f;
    float rearSlipAngleTrue = 0.0f;
    float frontSlipAngleRelaxed = 0.0f;
    float rearSlipAngleRelaxed = 0.0f;
    float frontGripUtilization = 0.0f;
    float rearGripUtilization = 0.0f;

    // Track-relative heading, computed independently of (but with the same
    // formula as) Observation's own normalized heading-error input -- see
    // obsHeadingErrorNorm below, which reads the network's actual input
    // value for direct comparison against this raw radian pair. wrongWay
    // itself is still diagnostic-only (never used for control/fitness/
    // termination) -- see the class comment.
    float trackDirectionAngle = 0.0f; // radians, world frame
    float headingError = 0.0f;        // radians, wrapped to [-pi, pi], heading - trackDirectionAngle
    bool wrongWay = false;            // |headingError| > kWrongWayHeadingErrorThresholdRad

    // Actual (rate-limited) front-wheel steering angle, radians -- may
    // differ from steeringCmd*CarParams::maxSteerAngle while the physical
    // wheel is still catching up to a just-changed command (see
    // CarParams::maxSteerRateRadPerSec). Appended at the end so existing
    // column indices/order are unaffected.
    float actualSteerAngle = 0.0f;

    // Normalized versions of the two Observation slots added alongside
    // actualSteerAngle above (ai::kActualSteerObservationIndex/
    // kYawRateObservationIndex) -- appended at the very end, same rationale
    // as actualSteerAngle.
    float obsActualSteerNorm = 0.0f;
    float obsYawRateNorm = 0.0f;

    // Observation's own normalized heading-error input (ai::kHeadingErrorObservationIndex)
    // -- the exact value the network receives, for direct comparison against
    // the raw headingError/trackDirectionAngle pair above. Appended at the
    // end, same rationale as obsActualSteerNorm/obsYawRateNorm.
    float obsHeadingErrorNorm = 0.0f;
};

// abs(headingError) beyond this is flagged wrongWay -- diagnostic only (see
// section 8 of the investigation this was built for): never used to
// terminate the car, alter fitness, or feed the network.
inline constexpr float kWrongWayHeadingErrorThresholdRad = 1.5707963f; // 90 degrees

// brakeCmd threshold for the "first meaningful braking moment" event
// reported at dump time (see HairpinFailureSummary).
inline constexpr float kBrakeOnsetThreshold = 0.05f;

// Deterministic column order, no trailing newline.
std::string hairpinCsvHeaderLine();

// One CSV row (no trailing newline), same column order as
// hairpinCsvHeaderLine(). Floats go through training::formatFloat(), which
// never throws even for a non-finite value (formats "nan"/"inf"/"-inf").
std::string hairpinSampleToCsvRow(const HairpinTelemetrySample& sample);

// Human-readable summary of one captured failure, derived purely from the
// already-buffered samples (never recomputed from live state) -- printed via
// TraceLog at dump time and mirrored into the CSV's companion .summary.txt.
struct HairpinFailureSummary
{
    bool hasBrakeOnset = false;
    float brakeOnsetSpeed = 0.0f;
    float brakeOnsetSensor0Px = 0.0f;
    float brakeOnsetMinForwardWallDistancePx = 0.0f;
    float brakeOnsetBestProgress = 0.0f;
    float brakeOnsetThrottle = 0.0f;
    float brakeOnsetSteering = 0.0f;

    bool hasSpinOnset = false;
    float spinOnsetSimTime = 0.0f;
    float spinOnsetBodySlipAngle = 0.0f;
    float spinOnsetYawRate = 0.0f;
    float spinOnsetFrontGripUtilization = 0.0f;
    float spinOnsetRearGripUtilization = 0.0f;
};

// Scans a captured buffer (oldest-first) for the first brakeCmd >
// kBrakeOnsetThreshold sample and the first wrongWay sample -- pure
// analysis, no side effects.
HairpinFailureSummary summarizeHairpinFailure(const std::vector<HairpinTelemetrySample>& orderedSamples);

// Captures dense telemetry for one individual at a time (never the whole
// population) via a fixed-capacity ring buffer covering roughly the last
// kRingBufferCapacitySeconds, and dumps it to a CSV under
// resultsDir/telemetry/ the moment that individual finishes with
// Collision/NoProgress while its bestProgress sits inside the hairpin
// region. At most kMaxDumpsPerGeneration dumps per generation.
class HairpinTelemetryRecorder
{
public:
    // Roughly 4.8s at the caller's dt -- "last few seconds before failure"
    // per the investigation's request.
    static constexpr float kRingBufferCapacitySeconds = 4.8f;
    static constexpr std::size_t kMaxDumpsPerGeneration = 2;

    // `enabled` defaults to the single kHairpinTelemetryEnabled constant
    // above; overridable so verification can exercise the disabled path
    // without recompiling with a different constant.
    HairpinTelemetryRecorder(const simulation::Track& track, std::string resultsDir, float simulationDt,
                              bool enabled = kHairpinTelemetryEnabled);

    // Call once per simulation frame, after population.update(). No-op when
    // disabled. Never mutates population/individuals -- only reads through
    // their public const getters; never touches any RNG.
    void update(const ai::neat::Population& population);

    bool isEnabled() const { return m_enabled; }

    // Test-only introspection.
    std::size_t getTotalDumpCount() const { return m_totalDumpCount; }
    std::size_t getRingBufferCapacity() const { return m_buffer.size(); }
    const HairpinRegion& getRegion() const { return m_region; }

    // Currently-buffered samples, oldest-first -- the same sequence a dump
    // would write to CSV. Test-only (production code never reads this
    // mid-capture; a real dump happens internally at the moment of failure).
    std::vector<HairpinTelemetrySample> getOrderedSamples() const;

private:
    void resetForNewGeneration(std::size_t generation);
    void startTracking(std::size_t generation, std::size_t individualIndex, const ai::neat::Individual& individual);
    void pushSample(const ai::neat::Individual& individual);
    bool isInHairpinRegion(float lapFractionProgress) const;
    void evaluateAndMaybeDump(std::size_t generation, std::size_t individualIndex,
                               ai::EvaluationFinishReason finishReason);
    std::string buildCsvPath(std::size_t generation, std::size_t individualIndex,
                              ai::EvaluationFinishReason finishReason) const;

    const simulation::Track& m_track;
    std::string m_resultsDir;
    bool m_enabled;
    HairpinRegion m_region;

    bool m_hasTrackedIndividual = false;
    std::size_t m_trackedGeneration = 0;
    std::size_t m_trackedIndividualIndex = 0;
    std::size_t m_nextSampleIndex = 0;

    // Fixed-capacity ring buffer: m_buffer.size() is the (constant)
    // capacity; m_writePos is the next slot to overwrite, m_filledCount
    // tracks how many slots currently hold real samples (< capacity until
    // the buffer has filled once).
    std::vector<HairpinTelemetrySample> m_buffer;
    std::size_t m_writePos = 0;
    std::size_t m_filledCount = 0;

    bool m_hasSeenGeneration = false;
    std::size_t m_lastSeenGeneration = 0;
    std::size_t m_dumpsThisGeneration = 0;
    std::size_t m_totalDumpCount = 0;
};

} // namespace telemetry
