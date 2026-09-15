#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <filesystem>
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

        const ChampionSummaryRow row = computeChampionSummary(1, 0, samples, 42.0f, ai::EvaluationFinishReason::TimeLimit);
        assert(std::fabs(row.averageSpeed - 200.0f) < 1e-3f && "average speed must be the plain mean of every sample's speed");
        assert(row.maxSpeed == 300.0f && "max speed must be the maximum sample speed");
        assert(row.medianSpeed == 200.0f && "median of {100,200,300} must be 200");
        assert(std::fabs(row.averageForwardSpeed - 180.0f) < 1e-3f && "average forward speed must be the mean forwardVelocity");
        assert(row.finalFitness == 42.0f && row.finishReason == ai::EvaluationFinishReason::TimeLimit &&
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
        const ChampionSummaryRow row = computeChampionSummary(1, 0, samples, 0.0f, ai::EvaluationFinishReason::TimeLimit);
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
        const ChampionSummaryRow row = computeChampionSummary(1, 0, samples, 0.0f, ai::EvaluationFinishReason::TimeLimit);
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
