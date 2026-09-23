#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "raylib.h"

#include "ai/FitnessEvaluator.h"
#include "ai/neat/CompatibilityConfig.h"
#include "ai/neat/CrossoverConfig.h"
#include "ai/neat/Genome.h"
#include "ai/neat/MutationConfig.h"
#include "ai/neat/Population.h"
#include "ai/neat/PopulationConfig.h"
#include "ai/neat/SpeciationConfig.h"
#include "simulation/Track.h"
#include "telemetry/ChampionTelemetry.h"
#include "training/GenerationMetrics.h"

#include "AppConfig.h"
#include "verification/Verifications.h"

namespace verification
{

using app::createDemonstrationGenome;
using app::kSimulationDt;
using app::kSpawnHeading;
using app::kSpawnPosition;
using app::makeCarParams;

using ai::neat::CompatibilityConfig;
using ai::neat::CrossoverConfig;
using ai::neat::MutationConfig;
using ai::neat::Population;
using ai::neat::SpeciationConfig;

using population_verify::makeCrashGenome;
using population_verify::makeTestPopulationConfig;

using telemetry::championCsvHeaderLine;
using telemetry::championSampleToCsvRow;
using telemetry::championGenomeToText;
using telemetry::championSummaryCsvHeaderLine;
using telemetry::championSummaryRowToCsvRow;
using telemetry::ChampionSummaryRow;
using telemetry::ChampionTelemetryRecorder;
using telemetry::ChampionTelemetrySample;
using telemetry::computeChampionSummary;

namespace
{

std::size_t countCommas(const std::string& s)
{
    return static_cast<std::size_t>(std::count(s.begin(), s.end(), ','));
}

// Builds a minimal, valid sample: only simTime/speed/lapCount/lastLapTime/
// bestLapTime/bestProgress/throttleCmd/brakeCmd/steeringCmd are meaningful
// to computeChampionSummary() -- every other field defaults harmlessly.
ChampionTelemetrySample makeSample(float simTime, float speed, float bestProgress = 0.0f, int lapCount = 0,
                                    float lastLapTime = 0.0f, float bestLapTime = 0.0f, float throttle = 0.0f,
                                    float brake = 0.0f, float steering = 0.0f)
{
    ChampionTelemetrySample s;
    s.simTime = simTime;
    s.speed = speed;
    s.bestProgress = bestProgress;
    s.lapCount = lapCount;
    s.lastLapTime = lastLapTime;
    s.bestLapTime = bestLapTime;
    s.throttleCmd = throttle;
    s.brakeCmd = brake;
    s.steeringCmd = steering;
    return s;
}

} // namespace

// Verifies the whole-run champion telemetry instrument (src/telemetry/
// ChampionTelemetry.h/.cpp) end to end: its CSV shapes are correct and
// deterministic, every derived summary statistic matches hand-computed
// expected values on synthetic inputs, generation selection/filtering works,
// the captured "champion" genuinely matches Population's own authoritative
// best-fitness bookkeeping, and -- most importantly -- attaching it via
// Population::setPerStepObserver() (including the ACTIVE capture/CSV-write
// path, not just an idle observer) never changes training behavior, RNG
// usage, or evaluation results in any way.
void verifyChampionTelemetry(const simulation::Track& track)
{
    const MutationConfig mutationConfig;
    const CrossoverConfig crossoverConfig;
    const CompatibilityConfig compatibilityConfig;
    const SpeciationConfig speciationConfig;

    // 1: raw-sample CSV header/row have exactly matching column counts, and
    // the header names the documented fields (not a silently reordered/
    // renamed set).
    {
        const std::string header = championCsvHeaderLine();
        assert(header == championCsvHeaderLine() && "championCsvHeaderLine() must be a pure, deterministic function");
        assert(header.find("brake_cmd") != std::string::npos && header.find("best_lap_time") != std::string::npos &&
               header.find("wrong_way") != std::string::npos && header.find("generation") != std::string::npos &&
               "header must contain the documented column names");

        const ChampionTelemetrySample sample;
        const std::string row = championSampleToCsvRow(sample);
        assert(countCommas(header) == countCommas(row) &&
               "champion CSV header and row must have exactly the same number of columns");
    }

    // 1b: the summary CSV carries the eight exact 60 Hz driving diagnostics as
    // its LAST columns (appended, so every pre-existing column keeps its index),
    // copied verbatim from the supplied summary; without one they are zeros.
    {
        ai::DrivingDiagnosticsSummary driving;
        driving.averageAbsSteeringDelta = 0.125f;
        driving.steeringReversalsPerSecond = 2.5f;
        driving.steeringSaturationFraction = 0.75f;
        driving.meanAbsSteering = 0.5f;
        driving.meanLateralAcceleration = 640.0f;
        driving.frontSlipBeyondPeakFraction = 0.25f;
        driving.lap2PlusAverageSpeed = 262.5f;
        driving.physicalBrakeUsageFraction = 0.0625f;
        driving.brakeRequestDominantFraction = 0.125f;
        driving.throttleRequestMin = 0.25f;
        driving.throttleRequestMax = 0.75f;
        driving.brakeRequestMin = 0.0f;
        driving.brakeRequestMax = 0.5f;
        driving.brakeOnsetSpeed = 300.0f;
        driving.brakeOnsetPreview120 = -0.5f;
        driving.brakeOnsetPreview300 = 0.375f;

        const std::string header = championSummaryCsvHeaderLine();
        const std::string tailHeader =
            "evaluator_avg_abs_steering_delta,steering_reversals_per_second,steering_saturation_fraction,"
            "mean_abs_steering_60hz,mean_lateral_accel,front_slip_beyond_peak_fraction,"
            "lap2plus_average_speed,physical_brake_usage_fraction,"
            "brake_request_dominant_fraction,throttle_request_min,throttle_request_max,"
            "brake_request_min,brake_request_max,brake_onset_speed,"
            "brake_onset_preview_120,brake_onset_preview_300";
        assert(header.size() > tailHeader.size() &&
               header.compare(header.size() - tailHeader.size(), tailHeader.size(), tailHeader) == 0 &&
               "the driving-diagnostic and longitudinal-request columns must be the summary header's last sixteen, in documented order");
        assert(header.find("progress_per_second,evaluator_avg_abs_steering_delta") != std::string::npos &&
               "they must directly follow the pre-existing last column (progress_per_second)");

        const ChampionSummaryRow row = computeChampionSummary(3, 5, {}, 1234.5f, ai::EvaluationFinishReason::CompletedLaps, &driving);
        assert(row.driving.averageAbsSteeringDelta == 0.125f && row.driving.physicalBrakeUsageFraction == 0.0625f &&
               "the supplied driving diagnostics must be copied into the row verbatim");
        const std::string csvRow = championSummaryRowToCsvRow(row);
        const std::string tailRow = "0.125,2.5,0.75,0.5,640,0.25,262.5,0.0625,0.125,0.25,0.75,0,0.5,300,-0.5,0.375";
        assert(csvRow.size() > tailRow.size() && csvRow.compare(csvRow.size() - tailRow.size(), tailRow.size(), tailRow) == 0 &&
               "the CSV row must end with the sixteen diagnostics, formatted like every other float");
        assert(countCommas(header) == countCommas(csvRow) && "header and row column counts must still match");

        const ChampionSummaryRow noDriving = computeChampionSummary(3, 5, {}, 1234.5f, ai::EvaluationFinishReason::CompletedLaps);
        assert(noDriving.driving.meanLateralAcceleration == 0.0f && noDriving.driving.steeringReversalsPerSecond == 0.0f &&
               "without a supplied summary the diagnostics must default to zero");
    }

    // 1c: championGenomeToText -- exact, deterministic plain-text genome dump
    // (nodes first, then connections, in storage order; weights shortest-round-trip).
    {
        ai::neat::Genome genome;
        genome.addNode(ai::neat::NodeGene{0, ai::neat::NodeType::Input});
        genome.addNode(ai::neat::NodeGene{1, ai::neat::NodeType::Bias});
        genome.addNode(ai::neat::NodeGene{7, ai::neat::NodeType::Hidden});
        genome.addNode(ai::neat::NodeGene{100, ai::neat::NodeType::Output});
        genome.addConnection(ai::neat::ConnectionGene{0, 7, -0.5f, true, 3});
        genome.addConnection(ai::neat::ConnectionGene{7, 100, 1.25f, false, 9});
        const std::string text = championGenomeToText(genome);
        assert(text == "N 0 0\nN 1 1\nN 7 2\nN 100 3\nC 0 7 -0.5 1 3\nC 7 100 1.25 0 9\n" &&
               "championGenomeToText must list every node then every connection, one per line, in the documented format");
        assert(championGenomeToText(genome) == text && "championGenomeToText must be a pure, deterministic function");
        assert(championGenomeToText(ai::neat::Genome{}).empty() && "an empty genome must dump to an empty string");
    }

    // 2: summary CSV header/row also have exactly matching column counts,
    // including on the empty-buffer default row computeChampionSummary()
    // returns.
    {
        const std::string header = championSummaryCsvHeaderLine();
        const ChampionSummaryRow emptyRow = computeChampionSummary(0, 0, {}, 0.0f, ai::EvaluationFinishReason::None);
        const std::string row = championSummaryRowToCsvRow(emptyRow);
        assert(countCommas(header) == countCommas(row) &&
               "champion summary CSV header and row must have exactly the same number of columns");
        assert(emptyRow.bestProgress == 0.0f && emptyRow.totalEvaluationTime == 0.0f &&
               "an empty sample buffer must produce an all-zero (not garbage/NaN) summary row");
    }

    // 3: average speed / max speed / median speed / average forward speed --
    // hand-computed expected values on a small synthetic buffer.
    {
        std::vector<ChampionTelemetrySample> samples = {
            makeSample(0.0f, 100.0f), makeSample(1.0f, 200.0f), makeSample(2.0f, 300.0f)};
        samples[0].forwardVelocity = 90.0f;
        samples[1].forwardVelocity = 180.0f;
        samples[2].forwardVelocity = 270.0f;

        const ChampionSummaryRow row = computeChampionSummary(1, 0, samples, 42.0f, ai::EvaluationFinishReason::CompletedLaps);
        assert(std::fabs(row.averageSpeed - 200.0f) < 1e-3f && "average speed must be the plain mean of every sample's speed");
        assert(row.maxSpeed == 300.0f && "max speed must be the maximum sample speed");
        assert(row.medianSpeed == 200.0f && "median of {100,200,300} must be 200");
        assert(std::fabs(row.averageForwardSpeed - 180.0f) < 1e-3f && "average forward speed must be the mean forwardVelocity");
        assert(row.finalFitness == 42.0f && row.finishReason == ai::EvaluationFinishReason::CompletedLaps &&
               "finalFitness/finishReason must be passed through verbatim, not recomputed from the buffer");
    }

    // 4: minimum MOVING speed ignores near-stationary samples (below
    // kMovingSpeedThresholdPxPerSec) but keeps the genuine minimum among
    // moving ones.
    {
        const std::vector<ChampionTelemetrySample> samples = {makeSample(0.0f, 0.0f), makeSample(1.0f, 5.0f),
                                                                makeSample(2.0f, 150.0f), makeSample(3.0f, 80.0f)};
        const ChampionSummaryRow row = computeChampionSummary(1, 0, samples, 0.0f, ai::EvaluationFinishReason::None);
        assert(row.minimumMovingSpeed == 80.0f &&
               "minimum moving speed must exclude near-stationary samples (0, 5 px/s) and report the true minimum "
               "among moving ones (80, not 150)");
    }

    // 5: slow-driving percentages are duration-weighted (by simTime, not by
    // sample count), attributing each sample's state to the interval BEFORE
    // the next sample.
    {
        const std::vector<ChampionTelemetrySample> samples = {
            makeSample(0.0f, 30.0f),  // [0,1): below 50, below 100, below 200
            makeSample(1.0f, 70.0f),  // [1,2): below 100, below 200 only
            makeSample(2.0f, 150.0f), // [2,3): below 200 only
            makeSample(3.0f, 999.0f)  // last sample -- contributes no interval
        };
        const ChampionSummaryRow row = computeChampionSummary(1, 0, samples, 0.0f, ai::EvaluationFinishReason::None);
        assert(row.totalEvaluationTime == 3.0f && "total evaluation time must be the last sample's simTime");
        assert(std::fabs(row.percentTimeBelow50 - (100.0f / 3.0f)) < 1e-2f && "1 of 3 seconds spent below 50px/s");
        assert(std::fabs(row.percentTimeBelow100 - (200.0f / 3.0f)) < 1e-2f && "2 of 3 seconds spent below 100px/s");
        assert(std::fabs(row.percentTimeBelow200 - 100.0f) < 1e-2f && "all 3 seconds spent below 200px/s");
    }

    // 6: longest continuous low-speed (< 100px/s) interval correctly finds
    // the LONGER of two separated low-speed runs, not the first or the sum.
    {
        std::vector<ChampionTelemetrySample> samples;
        const std::vector<float> speeds = {50.0f, 60.0f, 150.0f, 40.0f, 45.0f, 42.0f, 999.0f};
        for (std::size_t i = 0; i < speeds.size(); ++i)
        {
            samples.push_back(makeSample(static_cast<float>(i), speeds[i]));
        }
        // Intervals: [50,60) -> 2s low run, break at 150, then [40,45,42) -> 3s low run.
        const ChampionSummaryRow row = computeChampionSummary(1, 0, samples, 0.0f, ai::EvaluationFinishReason::None);
        assert(std::fabs(row.longestContinuousLowSpeedDurationSeconds - 3.0f) < 1e-3f &&
               "must report the LONGER of the two low-speed runs (3s), not the first (2s) or their sum (5s)");
    }

    // 7: throttle/brake usage and their simultaneous-overlap percentage.
    {
        std::vector<ChampionTelemetrySample> samples = {
            makeSample(0.0f, 0.0f, 0.0f, 0, 0.0f, 0.0f, 0.6f, 0.10f), // [0,1): throttle>0.5 AND brake>0.05 -> overlap
            makeSample(1.0f, 0.0f, 0.0f, 0, 0.0f, 0.0f, 0.3f, 0.20f), // [1,2): brake>0.05 only
            makeSample(2.0f, 0.0f, 0.0f, 0, 0.0f, 0.0f, 0.9f, 0.02f), // [2,3): throttle>0.5 only
            makeSample(3.0f, 0.0f)                                    // last -- no interval
        };
        const ChampionSummaryRow row = computeChampionSummary(1, 0, samples, 0.0f, ai::EvaluationFinishReason::None);
        assert(std::fabs(row.percentTimeThrottleAndBrakeOverlap - (100.0f / 3.0f)) < 1e-2f &&
               "exactly 1 of 3 seconds had throttle>0.5 AND brake>0.05 simultaneously");
        assert(std::fabs(row.percentTimeThrottleAboveHalf - (200.0f / 3.0f)) < 1e-2f &&
               "2 of 3 seconds had throttle>0.5 (samples 0 and 2)");
        assert(std::fabs(row.percentTimeBrakeAbove005 - (200.0f / 3.0f)) < 1e-2f &&
               "2 of 3 seconds had brake>0.05 (samples 0 and 1)");
        assert(row.percentTimeBrakeAbove025 == 0.0f && "no sample ever exceeded brake>0.25");
    }

    // 8: lap-time aggregation -- detects each lapCount increase as one
    // completed lap, attributes lastLapTime at that transition, and reports
    // the final bestLapTime/averageCompletedLapTime correctly.
    {
        const std::vector<ChampionTelemetrySample> samples = {
            makeSample(0.0f, 100.0f, 0.0f, 0, 0.0f, 0.0f),
            makeSample(5.0f, 100.0f, 1.0f, 1, 12.0f, 12.0f),  // lap 1 completed in 12.0s
            makeSample(6.0f, 100.0f, 1.0f, 1, 12.0f, 12.0f),  // same lap, no new completion
            makeSample(15.0f, 100.0f, 2.0f, 2, 10.0f, 10.0f), // lap 2 completed in 10.0s (new best)
            makeSample(16.0f, 100.0f, 2.0f, 2, 10.0f, 10.0f)  // final sample, no new completion
        };
        const ChampionSummaryRow row = computeChampionSummary(1, 0, samples, 0.0f, ai::EvaluationFinishReason::CompletedLaps);
        assert(row.completedLaps == 2 && "must count exactly 2 lapCount increases");
        assert(std::fabs(row.averageCompletedLapTime - 11.0f) < 1e-3f && "average of the two completed lap times (12, 10) must be 11");
        assert(row.bestLapTime == 10.0f && "bestLapTime must be read from the final sample");
    }

    // 8b: zero completed laps must report 0, not a divide-by-zero artifact.
    {
        const std::vector<ChampionTelemetrySample> samples = {makeSample(0.0f, 100.0f), makeSample(5.0f, 100.0f)};
        const ChampionSummaryRow row = computeChampionSummary(1, 0, samples, 0.0f, ai::EvaluationFinishReason::Collision);
        assert(row.completedLaps == 0 && row.averageCompletedLapTime == 0.0f &&
               "zero completed laps must report completedLaps=0 and averageCompletedLapTime=0, not NaN/garbage");
    }

    // 9: progress-per-second is bestProgress divided by total evaluation time.
    {
        const std::vector<ChampionTelemetrySample> samples = {makeSample(0.0f, 100.0f, 0.0f),
                                                                makeSample(50.0f, 100.0f, 2.5f)};
        const ChampionSummaryRow row = computeChampionSummary(1, 0, samples, 0.0f, ai::EvaluationFinishReason::CompletedLaps);
        assert(std::fabs(row.progressPerSecond - 0.05f) < 1e-4f && "2.5 laps over 50s must be 0.05 laps/second");
    }

    // 10: selected-generation filtering -- only generations present in the
    // (constructor-overridden) target list are ever captured; everything
    // else costs nothing and produces nothing. Uses a tiny synthetic target
    // list ({1, 3}) rather than the real kChampionTelemetryGenerations
    // (133+) so this stays fast.
    {
        const std::string scratchDir = "verification_scratch/champion_filtering_test";
        std::filesystem::remove_all(scratchDir);

        Population population(makeCrashGenome(), track, makeCarParams(), kSpawnPosition, kSpawnHeading,
                               makeTestPopulationConfig(8, 55u), mutationConfig, crossoverConfig, compatibilityConfig,
                               speciationConfig);
        auto recorder = std::make_unique<ChampionTelemetryRecorder>(scratchDir, kSimulationDt, true,
                                                                      std::vector<std::size_t>{1, 3});
        population.setPerStepObserver(
            [&recorder](const Population& pop, bool finished) { recorder->onPopulationStep(pop, finished); });

        for (int step = 0; step < 6000 && population.getGeneration() <= 3; ++step)
        {
            population.update(kSimulationDt);
        }
        assert(population.getGeneration() > 3 &&
               "setup: the crash genome must reach generation 4 (i.e. finish generation 3) well within 6000 steps");

        assert(recorder->getCapturedGenerationCount() == 2 &&
               "exactly generations 1 and 3 must have been captured, never 0, 2, or any other");
        const std::vector<ChampionSummaryRow>& rows = recorder->getSummaryRows();
        assert(rows.size() == 2 && rows[0].generation == 1 && rows[1].generation == 3 &&
               "summary rows must be exactly generation 1 then generation 3, in completion order");

        const std::filesystem::path dir = std::filesystem::path(scratchDir) / "telemetry" / "champions";
        assert(std::filesystem::exists(dir / "champion_gen0001.csv") && "generation 1 must have produced a raw CSV");
        assert(std::filesystem::exists(dir / "champion_gen0003.csv") && "generation 3 must have produced a raw CSV");
        assert(!std::filesystem::exists(dir / "champion_gen0000.csv") && "generation 0 was not targeted -- must not exist");
        assert(!std::filesystem::exists(dir / "champion_gen0002.csv") && "generation 2 was not targeted -- must not exist");
        assert(std::filesystem::exists(dir / "champion_summary.csv") && "the summary CSV must exist once any generation is captured");

        std::filesystem::remove_all(scratchDir);
    }

    // 10b (J, K): the default target-generation schedule is exactly
    // 100, 200, ..., 1000 -- one central list -- and only those generations are
    // targeted; file names carry the real (0-based) generation number.
    {
        const std::vector<std::size_t> expected = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000};
        assert(telemetry::kChampionTelemetryGenerations == expected &&
               "the champion telemetry target generations must be exactly 100,200,...,1000"); // J
        assert(telemetry::makeChampionTelemetryGenerations() == expected && "the schedule must be built deterministically"); // J

        const ChampionTelemetryRecorder recorder("verification_scratch/champion_schedule_test", kSimulationDt);
        for (std::size_t generation : expected)
        {
            assert(recorder.isTargetGeneration(generation) && "every scheduled generation must be a target"); // J
        }
        for (std::size_t generation : {std::size_t{0}, std::size_t{1}, std::size_t{99}, std::size_t{101}, std::size_t{133},
                                       std::size_t{176}, std::size_t{234}, std::size_t{251}, std::size_t{299},
                                       std::size_t{999}, std::size_t{1001}, std::size_t{1100}, std::size_t{5000}})
        {
            assert(!recorder.isTargetGeneration(generation) &&
                   "generations outside the schedule (including the old 133/176/234/251 and anything past 1000) "
                   "must not be targets"); // K
        }
        assert(!std::filesystem::exists("verification_scratch/champion_schedule_test") &&
               "constructing a recorder must not create any output");

        assert(telemetry::championCsvFileName(100) == "champion_gen0100.csv" &&
               telemetry::championCsvFileName(1000) == "champion_gen1000.csv" &&
               telemetry::championGenomeFileName(1000) == "champion_gen1000_genome.txt" &&
               telemetry::championCsvFileName(0) == "champion_gen0000.csv" &&
               "file names must carry the real generation number, zero-padded to 4 digits without truncation");
    }

    // 11: a disabled recorder never captures anything and never creates its
    // output directory, even across several generations of a real population.
    {
        const std::string scratchDir = "verification_scratch/champion_disabled_test";
        std::filesystem::remove_all(scratchDir);

        Population population(makeCrashGenome(), track, makeCarParams(), kSpawnPosition, kSpawnHeading,
                               makeTestPopulationConfig(6, 77u), mutationConfig, crossoverConfig, compatibilityConfig,
                               speciationConfig);
        auto recorder = std::make_unique<ChampionTelemetryRecorder>(scratchDir, kSimulationDt, false,
                                                                      std::vector<std::size_t>{0, 1, 2});
        assert(!recorder->isEnabled() && "recorder constructed with enabled=false must report itself as disabled");
        population.setPerStepObserver(
            [&recorder](const Population& pop, bool finished) { recorder->onPopulationStep(pop, finished); });

        for (int step = 0; step < 2000 && population.getGeneration() <= 2; ++step)
        {
            population.update(kSimulationDt);
        }
        assert(recorder->getCapturedGenerationCount() == 0 && "a disabled recorder must never capture anything");
        assert(!std::filesystem::exists(scratchDir) &&
               "a disabled recorder must never create its results directory, let alone any file in it");
    }

    // 12: the captured champion's finalFitness genuinely matches
    // Population's OWN authoritative bestFitness for that generation
    // (GenerationMetrics, computed by an entirely separate code path inside
    // reproduce()) -- proof that Population::getBestIndividualIndex(), read
    // from inside the pre-reproduce observer hook, really does identify the
    // same individual reproduce() itself ranks first (rankedIndices[0]).
    // Also: every raw sample stays finite across a real, actively-captured
    // generation.
    {
        const std::string scratchDir = "verification_scratch/champion_correctness_test";
        std::filesystem::remove_all(scratchDir);

        Population population(createDemonstrationGenome(), track, makeCarParams(), kSpawnPosition, kSpawnHeading,
                               makeTestPopulationConfig(10, 909u), mutationConfig, crossoverConfig, compatibilityConfig,
                               speciationConfig);
        auto recorder = std::make_unique<ChampionTelemetryRecorder>(scratchDir, kSimulationDt, true,
                                                                      std::vector<std::size_t>{0});
        population.setPerStepObserver(
            [&recorder](const Population& pop, bool finished) { recorder->onPopulationStep(pop, finished); });

        for (int step = 0; step < 20000 && population.getGeneration() == 0; ++step)
        {
            population.update(kSimulationDt);
        }
        assert(population.getGeneration() == 1 && "setup: generation 0 must finish within 20000 steps");
        assert(recorder->getCapturedGenerationCount() == 1 && "generation 0 was targeted -- exactly one capture expected");

        const ChampionSummaryRow& captured = recorder->getSummaryRows().front();
        const training::GenerationMetrics& metrics = population.getLastGenerationMetrics();
        assert(captured.generation == 0 && metrics.generation == 0 && "both must refer to the just-finished generation 0");
        assert(captured.finalFitness == metrics.bestFitness &&
               "the champion telemetry's captured individual must have exactly the population's own best fitness "
               "for that generation -- proof the pre-reproduce best-index read matches reproduce()'s own ranking");

        const std::filesystem::path csvPath =
            std::filesystem::path(scratchDir) / "telemetry" / "champions" / "champion_gen0000.csv";
        assert(std::filesystem::exists(csvPath) && "the captured generation must have produced a raw CSV file");

        // The captured champion's exact diagnostics are the population's own best
        // individual's (same individual, same evaluation) -- field for field.
        assert(captured.driving.averageAbsSteeringDelta == metrics.bestDriving.averageAbsSteeringDelta &&
               captured.driving.steeringReversalsPerSecond == metrics.bestDriving.steeringReversalsPerSecond &&
               captured.driving.steeringSaturationFraction == metrics.bestDriving.steeringSaturationFraction &&
               captured.driving.meanAbsSteering == metrics.bestDriving.meanAbsSteering &&
               captured.driving.meanLateralAcceleration == metrics.bestDriving.meanLateralAcceleration &&
               captured.driving.frontSlipBeyondPeakFraction == metrics.bestDriving.frontSlipBeyondPeakFraction &&
               captured.driving.lap2PlusAverageSpeed == metrics.bestDriving.lap2PlusAverageSpeed &&
               captured.driving.physicalBrakeUsageFraction == metrics.bestDriving.physicalBrakeUsageFraction &&
               "the champion summary's driving diagnostics must equal the generation metrics' best-individual diagnostics");
        assert(captured.driving.meanAbsSteering >= 0.0f && captured.driving.meanAbsSteering <= 1.0f &&
               captured.driving.meanLateralAcceleration >= 0.0f && "champion diagnostics must be valid");

        // The champion genome is dumped next to its CSV, in the documented format.
        const std::filesystem::path genomePath =
            std::filesystem::path(scratchDir) / "telemetry" / "champions" / "champion_gen0000_genome.txt";
        assert(std::filesystem::exists(genomePath) && "the captured generation must have produced a champion genome file");
        {
            std::ifstream genomeFile(genomePath);
            std::string firstLine;
            std::getline(genomeFile, firstLine);
            assert(firstLine.rfind("# Champion genome -- generation 0", 0) == 0 &&
                   "the genome file must start with its descriptive header comment");
            std::size_t nodeLines = 0;
            std::size_t connectionLines = 0;
            std::string line;
            while (std::getline(genomeFile, line))
            {
                nodeLines += (line.rfind("N ", 0) == 0) ? 1 : 0;
                connectionLines += (line.rfind("C ", 0) == 0) ? 1 : 0;
            }
            const ai::neat::Genome baseline = createDemonstrationGenome();
            assert(nodeLines >= baseline.nodes().size() && connectionLines >= 1 &&
                   "the genome file must list the champion's nodes and connections");
        }

        std::filesystem::remove_all(scratchDir);
    }

    // 13, 14 & 15: attaching a ChampionTelemetryRecorder via
    // Population::setPerStepObserver() -- INCLUDING while it actively
    // captures and writes CSVs for a targeted generation (not merely an idle
    // observer) -- must not change anything about that Population's own
    // behavior. Proven the same way HairpinTelemetryVerification proves it
    // for its own instrument: comparing a full per-frame fitness trace
    // (every individual, every frame, plus the compatibility threshold)
    // between an observed run and an unobserved twin built from the exact
    // same base genome/config/seed, across a real generation transition (the
    // only place mutation/crossover/tournament selection touch the
    // orchestration RNG).
    {
        const std::string scratchDir = "verification_scratch/champion_determinism_test";

        auto driveAndCollectFitness = [&](bool attachRecorder) -> std::vector<float>
        {
            std::filesystem::remove_all(scratchDir);
            Population population(makeCrashGenome(), track, makeCarParams(), kSpawnPosition, kSpawnHeading,
                                   makeTestPopulationConfig(8, 6161u), mutationConfig, crossoverConfig,
                                   compatibilityConfig, speciationConfig);
            std::unique_ptr<ChampionTelemetryRecorder> recorder;
            if (attachRecorder)
            {
                // Target generations 0 AND 1 so the run captures across the
                // transition itself, not just one side of it -- the
                // strongest version of this proof.
                recorder = std::make_unique<ChampionTelemetryRecorder>(scratchDir, kSimulationDt, true,
                                                                         std::vector<std::size_t>{0, 1});
                population.setPerStepObserver(
                    [&recorder](const Population& pop, bool finished) { recorder->onPopulationStep(pop, finished); });
            }

            std::vector<float> trace;
            for (int i = 0; i < 900 && population.getGeneration() < 2; ++i) // up to 15s, spans >= 2 generations
            {
                population.update(kSimulationDt);
                trace.push_back(population.getLastGenerationBestFitness());
                trace.push_back(population.getCurrentCompatibilityThreshold());
                for (std::size_t idx = 0; idx < population.size(); ++idx)
                {
                    trace.push_back(population.getIndividual(idx).getFitness());
                }
            }
            return trace;
        };

        const std::vector<float> withTelemetry = driveAndCollectFitness(true);
        const std::vector<float> withoutTelemetry = driveAndCollectFitness(false);
        std::filesystem::remove_all(scratchDir);

        assert(withTelemetry.size() == withoutTelemetry.size() &&
               "identical scenarios with/without champion telemetry must run for the same number of steps"); // 15 (determinism)
        for (std::size_t i = 0; i < withTelemetry.size(); ++i)
        {
            assert(withTelemetry[i] == withoutTelemetry[i] &&
                   "champion telemetry capture (including active CSV writing) must not alter car control, fitness, "
                   "RNG-driven reproduction, or any other training behavior -- traces with and without an attached "
                   "recorder must be bit-identical"); // 13, 14
        }
    }

    std::filesystem::remove_all("verification_scratch");

    TraceLog(LOG_INFO, "Champion telemetry verification: all deterministic checks passed");
}

} // namespace verification
