#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "ai/DrivingDiagnostics.h"
#include "ai/FitnessEvaluator.h"
#include "ai/neat/Genome.h"

// Whole-run champion telemetry: captures dense per-frame trajectory data for
// the population's best individual, across a fixed, evenly spaced set of
// generations (see kChampionTelemetryGenerations), covering that individual's ENTIRE evaluation (spawn to
// finish) -- not just a local region of the track like HairpinTelemetry.
//
// Pure observer, same contract as telemetry::HairpinTelemetryRecorder (see
// HairpinTelemetry.h): reads Population/Individual/Car/TrackProgress/
// FitnessEvaluator/AIController/Observation only through their existing
// const getters, never writes into any of them, never advances RNG, never
// feeds anything back into control/fitness/evolution. Entirely separate
// from HairpinTelemetryRecorder (kept unmodified) and from
// training::TrainingLogger/GenerationMetrics (also unmodified).
//
// Wiring: registered as ai::neat::Population's optional per-step observer
// (Population::setPerStepObserver()) -- the only point outside Population
// itself where a just-finished generation's final individuals are both
// fully evaluated and not yet replaced by reproduce(). "The champion" is
// whichever individual Population::getBestIndividualIndex() (highest raw
// fitness, ties -> lower index) reports at that exact moment -- the same
// rule reproduce() itself uses for global elitism ranking (rankedIndices[0]),
// so this always matches "the individual that ends the generation with
// highest fitness".
//
// Approach: candidate telemetry is buffered live, for every individual, only
// during a targeted generation's evaluation; the moment that generation
// finishes, every buffer except the champion's is discarded. A true
// replay-from-saved-genome approach was considered but rejected: nothing in
// this codebase saves/loads genomes today, and a full deterministic replay
// would mean re-running the whole Population from generation 0 up to the
// generation of interest (there is no per-generation checkpoint) -- far more
// code and risk for the same guaranteed-correct result this simpler,
// in-pass buffering approach already gives (see Population::getGeneration()
// being read from inside the observer -- always the CURRENT, in-progress
// generation, so no replay is ever needed).
namespace ai::neat
{
class Population;
class Individual;
} // namespace ai::neat

namespace telemetry
{

// Single point of control for this whole feature -- flip to false to fully
// disable it (ChampionTelemetryRecorder's default constructor argument
// reads this) without touching anything else. Also directly
// constructor-overridable so verification can exercise a small, fast
// synthetic generation list without waiting for real generation numbers.
inline constexpr bool kChampionTelemetryEnabled = true;

// Every Nth simulation step is sampled, plus -- unconditionally -- each
// individual's own final step. Counted in simulation steps (not wall-clock
// time), so NORMAL and FAST training-speed modes, which only change render
// cadence and never dt or call order (see main.cpp), produce byte-identical
// samples for identical trajectories.
inline constexpr int kChampionTelemetrySampleInterval = 5;

// Which generations to capture full champion telemetry (per-frame CSV, summary
// row, genome dump) for: every kChampionTelemetryGenerationInterval-th
// generation from kChampionTelemetryFirstGeneration up to and including
// kChampionTelemetryLastGeneration -- 100, 200, ..., 1000. This schedule is
// SEPARATE from how long training runs: training itself has no generation
// limit, generations after the last one simply produce no snapshot. Generation
// numbers are the same 0-based numbers Population::getGeneration() and the
// training CSV's `generation` column use, so champion_gen1000.csv is the
// champion of the row with generation == 1000.
inline constexpr std::size_t kChampionTelemetryFirstGeneration = 100;
inline constexpr std::size_t kChampionTelemetryGenerationInterval = 100;
inline constexpr std::size_t kChampionTelemetryLastGeneration = 1000;

// The full target list built from the three constants above -- the ONE source
// of truth for which generations are captured (ChampionTelemetryRecorder's
// default constructor argument reads kChampionTelemetryGenerations).
inline std::vector<std::size_t> makeChampionTelemetryGenerations()
{
    std::vector<std::size_t> generations;
    for (std::size_t g = kChampionTelemetryFirstGeneration; g <= kChampionTelemetryLastGeneration;
         g += kChampionTelemetryGenerationInterval)
    {
        generations.push_back(g);
    }
    return generations;
}

inline const std::vector<std::size_t> kChampionTelemetryGenerations = makeChampionTelemetryGenerations();

// Simulated seconds of samples reserved per individual buffer when a capture
// begins -- roughly one normal 3-lap evaluation. Only a memory pre-sizing hint.
inline constexpr float kChampionTelemetryReserveSeconds = 60.0f;

// Speed (px/s) below which a sample doesn't count toward "minimum MOVING
// speed" -- distinguishes genuinely near-stationary samples (spawn, a dead
// stop) from crawling-but-moving ones. Diagnostic only.
inline constexpr float kMovingSpeedThresholdPxPerSec = 20.0f;

// One frame's worth of captured state for a champion candidate. Field order
// here is the authoritative raw-CSV column order (see
// championCsvHeaderLine()). Deliberately mirrors HairpinTelemetrySample's
// field set/naming where the same quantity applies, plus the extra
// lap-timing/progress fields this whole-run instrument also needs.
struct ChampionTelemetrySample
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
    int lapCount = 0;
    int checkpointIndex = 0;
    int totalCheckpointsPassed = 0;
    bool alive = true;
    bool finished = false;
    ai::EvaluationFinishReason finishReason = ai::EvaluationFinishReason::None;

    float speed = 0.0f;
    float normalizedSpeed = 0.0f; // speed / car.getMaxSpeed() -- unclamped, unlike Observation's obsSpeedNorm
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

    // Raw physical sensor distances (px) -- see Observation.cpp for the
    // [0,1]-normalized versions above.
    float sensorM60Px = 0.0f;
    float sensorM30Px = 0.0f;
    float sensor0Px = 0.0f;
    float sensorP30Px = 0.0f;
    float sensorP60Px = 0.0f;

    // Track-relative heading, computed independently of (but with the same
    // formula as) Observation's own normalized heading-error input -- see
    // obsHeadingErrorNorm below, and HairpinTelemetry
    // (kWrongWayHeadingErrorThresholdRad there). wrongWay itself is still
    // diagnostic-only (never used for control/fitness/termination).
    float trackDirectionAngle = 0.0f; // radians, world frame
    float headingError = 0.0f;        // radians, wrapped to [-pi, pi], heading - trackDirectionAngle
    bool wrongWay = false;

    float currentLapElapsedTime = 0.0f;
    float lastLapTime = 0.0f; // 0.0f (sentinel) until the first lap completes
    float bestLapTime = 0.0f; // 0.0f (sentinel) until the first lap completes

    // Actual (rate-limited) front-wheel steering angle, radians -- may
    // differ from steeringCmd*CarParams::maxSteerAngle while the physical
    // wheel is still catching up to a just-changed command (see
    // CarParams::maxSteerRateRadPerSec). Appended at the very end so
    // existing column indices/order are unaffected.
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

    // Observation's two track-direction PREVIEW inputs
    // (ai::kPreviewNearObservationIndex/kPreviewFarObservationIndex): the
    // same normalized heading-error formula as obsHeadingErrorNorm, against
    // the track tangent ai::kPreviewNearDistance (120px) / kPreviewFarDistance
    // (300px) farther along the centerline. The exact values the network
    // receives. Appended at the end, same rationale as obsHeadingErrorNorm.
    float obsPreviewHeadingError120Norm = 0.0f;
    float obsPreviewHeadingError300Norm = 0.0f;

    // Speed-sensitive steering authority in effect this frame (multiplier on
    // CarParams::maxSteerAngle, see Car.h) and the wheel angle (radians) that
    // steering_cmd = +-1 targeted at that speed. steering_cmd itself is still
    // the raw network command in [-1, 1]; actual_steer_angle is the physical
    // result after this and the steering-rate limit. Appended at the end.
    float steeringAuthorityFactor = 1.0f;
    float effectiveMaxSteerAngle = 0.0f;
};

// Deterministic column order, no trailing newline.
std::string championCsvHeaderLine();

// One CSV row (no trailing newline), same column order as
// championCsvHeaderLine(). Floats go through training::formatFloat(), which
// never throws even for a non-finite value.
std::string championSampleToCsvRow(const ChampionTelemetrySample& sample);

// One compact summary row for one captured champion -- see
// computeChampionSummary(). Field order here is the authoritative summary
// CSV column order (see championSummaryCsvHeaderLine()). Every value here is
// diagnostic only; none of it feeds back into fitness/evolution.
struct ChampionSummaryRow
{
    std::size_t generation = 0;
    std::size_t individualIndex = 0;
    float finalFitness = 0.0f;
    float bestProgress = 0.0f;
    int lapCount = 0;
    float totalEvaluationTime = 0.0f;
    ai::EvaluationFinishReason finishReason = ai::EvaluationFinishReason::None;

    float averageSpeed = 0.0f;
    float maxSpeed = 0.0f;
    float minimumMovingSpeed = 0.0f; // lowest speed among samples above kMovingSpeedThresholdPxPerSec (0 if none)
    float medianSpeed = 0.0f;
    float averageForwardSpeed = 0.0f;

    // Duration-weighted (simulated seconds), not sample-count-weighted --
    // each sample's speed/control state is treated as constant over
    // [thisSample.simTime, nextSample.simTime).
    float percentTimeBelow50 = 0.0f;
    float percentTimeBelow100 = 0.0f;
    float percentTimeBelow200 = 0.0f;
    float longestContinuousLowSpeedDurationSeconds = 0.0f; // longest unbroken stretch with speed < 100 px/s

    float averageThrottle = 0.0f;
    float percentTimeThrottleAboveHalf = 0.0f;
    float averageBrake = 0.0f;
    float percentTimeBrakeAbove005 = 0.0f;
    float percentTimeBrakeAbove025 = 0.0f;
    float percentTimeThrottleAndBrakeOverlap = 0.0f; // throttle > 0.5 AND brake > 0.05, simultaneously
    float averageAbsSteering = 0.0f;

    int completedLaps = 0;
    float bestLapTime = 0.0f;
    float averageCompletedLapTime = 0.0f; // 0 if fewer than 1 completed lap was observed in the buffer

    float progressPerSecond = 0.0f; // bestProgress / totalEvaluationTime

    // Exact 60 Hz driving-quality diagnostics of this champion's evaluation
    // (steering delta/reversals/saturation, lateral acceleration, front slip
    // past peak, lap-2+ speed, physical brake usage, throttle/brake request
    // ranges, brake-request-dominant fraction, brake onset speed/previews) -- see
    // ai::DrivingDiagnosticsSummary for each definition. Unlike the statistics
    // above these are NOT estimated from the sparse sample buffer: they come
    // straight from the live Individual. All zero if not supplied. Appended
    // at the end of the summary CSV.
    ai::DrivingDiagnosticsSummary driving;
};

std::string championSummaryCsvHeaderLine();
std::string championSummaryRowToCsvRow(const ChampionSummaryRow& row);

// Pure computation from an ordered (oldest-first, by sampleIndex), non-empty
// sample buffer plus the champion's final scalar state -- no I/O, no
// simulation/Population state, so this is directly unit-testable (see
// verifyChampionTelemetry()). finalFitness/finishReason are passed in
// separately (read from the live Individual at capture time) rather than
// re-derived from the possibly-sparse sample buffer, so the summary's
// headline numbers are always exact even though the detailed
// speed/control/lap statistics are necessarily an estimate at the
// configured sampling interval. Returns a default (all-zero) row if
// orderedSamples is empty.
// `driving`, if non-null, is copied into the row verbatim (see ChampionSummaryRow::driving).
ChampionSummaryRow computeChampionSummary(std::size_t generation, std::size_t individualIndex,
                                           const std::vector<ChampionTelemetrySample>& orderedSamples,
                                           float finalFitness, ai::EvaluationFinishReason finishReason,
                                           const ai::DrivingDiagnosticsSummary* driving = nullptr);

// Plain-text dump of a genome, one node/connection per line, so a champion can
// be reloaded/analyzed later without re-running training:
//   N <nodeId> <nodeType>                              nodeType: 0=Input 1=Bias 2=Hidden 3=Output
//   C <sourceId> <targetId> <weight> <enabled> <innovation>
// Nodes first (genome storage order), then connections (storage order).
// Weights use training::formatFloat (shortest round-trippable). Pure function.
std::string championGenomeToText(const ai::neat::Genome& genome);

// File names of one captured generation's per-frame CSV and genome dump, e.g.
// champion_gen0100.csv / champion_gen1000_genome.txt (generation zero-padded to
// at least 4 digits, never truncated).
std::string championCsvFileName(std::size_t generation);
std::string championGenomeFileName(std::size_t generation);

// Captures every individual's telemetry live during a targeted generation's
// evaluation, then keeps only the champion's (see the file comment) --
// dumping it to `resultsDir/telemetry/champions/champion_gen<NNNN>.csv` and
// appending one row to `resultsDir/telemetry/champions/champion_summary.csv`
// (rewritten in full each time, so it's always consistent; cheap since at
// most a handful of generations are ever targeted). Generations not in
// targetGenerations cost one cheap membership check per simulation step and
// nothing else -- no buffers allocated, no individual state read.
class ChampionTelemetryRecorder
{
public:
    // `enabled` defaults to kChampionTelemetryEnabled;
    // `targetGenerations` defaults to kChampionTelemetryGenerations. Both
    // are directly overridable (like HairpinTelemetryRecorder's `enabled`)
    // so verification can exercise a small, fast synthetic generation list
    // without recompiling or waiting for real generation numbers.
    ChampionTelemetryRecorder(std::string resultsDir, float simulationDt, bool enabled = kChampionTelemetryEnabled,
                               std::vector<std::size_t> targetGenerations = kChampionTelemetryGenerations);

    // Intended to be registered via ai::neat::Population::setPerStepObserver()
    // (see the file comment for why that specific hook point matters).
    // No-op when disabled, or when neither an in-progress capture nor
    // population.getGeneration() is a targeted generation. Never mutates
    // population; never touches any RNG.
    void onPopulationStep(const ai::neat::Population& population, bool generationJustFinished);

    bool isEnabled() const { return m_enabled; }

    // True iff `generation` (0-based, as Population::getGeneration()) is in the target list.
    bool isTargetGeneration(std::size_t generation) const;

    // Test-only introspection.
    std::size_t getCapturedGenerationCount() const { return m_capturedGenerations; }
    const std::vector<ChampionSummaryRow>& getSummaryRows() const { return m_summaryRows; }
    bool isCapturing() const { return m_isCapturing; }

private:
    void beginCapture(std::size_t generation, std::size_t individualCount);
    void sampleIndividual(std::size_t individualIndex, const ai::neat::Individual& individual);
    void finishCapture(const ai::neat::Population& population);
    std::string buildCsvPath(std::size_t generation) const;
    std::string buildGenomePath(std::size_t generation) const;
    void writeSummaryCsv() const;

    std::string m_resultsDir;
    bool m_enabled;
    std::vector<std::size_t> m_targetGenerations;
    float m_simulationDt;

    bool m_isCapturing = false;
    std::size_t m_capturingGeneration = 0;
    std::size_t m_stepIndexInGeneration = 0;
    std::vector<std::vector<ChampionTelemetrySample>> m_buffers; // one per individual, this generation only
    std::vector<bool> m_recordedFinish;

    std::size_t m_capturedGenerations = 0;
    std::vector<ChampionSummaryRow> m_summaryRows;
};

} // namespace telemetry
