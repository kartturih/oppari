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
#include "telemetry/HairpinTelemetry.h"

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

using telemetry::computeHairpinRegion;
using telemetry::HairpinFailureSummary;
using telemetry::HairpinRegion;
using telemetry::hairpinCsvHeaderLine;
using telemetry::hairpinSampleToCsvRow;
using telemetry::HairpinTelemetryRecorder;
using telemetry::HairpinTelemetrySample;
using telemetry::summarizeHairpinFailure;

// Verifies the hairpin telemetry debugging instrument (src/telemetry/) is a
// pure observer: it never changes Car/Population/RNG/fitness behavior, its
// region detection/CSV shape are correct and deterministic, it respects its
// own capacity/enable switch, and it never writes non-finite values.
void verifyHairpinTelemetry(const simulation::Track& track)
{
    const MutationConfig mutationConfig;
    const CrossoverConfig crossoverConfig;
    const CompatibilityConfig compatibilityConfig;
    const SpeciationConfig speciationConfig;

    // 1: computeHairpinRegion() derives a sane, non-degenerate window
    // entirely from the track's own centerline data -- both fractions valid,
    // start < end (this track's hairpin doesn't straddle the lap seam), and
    // it sits early on the lap (right after the long start straight),
    // matching the P1..P7 control points it's built from.
    {
        const HairpinRegion region = computeHairpinRegion(track);
        assert(region.startLapFraction >= 0.0f && region.startLapFraction < 1.0f && region.endLapFraction >= 0.0f &&
               region.endLapFraction < 1.0f && "hairpin region fractions must be valid lap fractions");
        assert(region.startLapFraction < region.endLapFraction &&
               "the extreme track's hairpin region must not straddle the lap seam");
        assert(region.endLapFraction < 0.35f && "the first hairpin must sit well within the first third of the lap");
    }

    // 2: CSV header is a pure, stable function, and every row has exactly
    // the same column count as the header (deterministic column order).
    {
        const std::string header = hairpinCsvHeaderLine();
        assert(header == hairpinCsvHeaderLine() && "hairpinCsvHeaderLine() must be a pure, deterministic function");
        assert(header.find("brake_cmd") != std::string::npos && header.find("wrong_way") != std::string::npos &&
               header.find("generation") != std::string::npos &&
               "header must contain the documented column names, not a silently reordered/renamed set");

        const HairpinTelemetrySample sample; // default-constructed is enough to exercise formatting
        const std::string row = hairpinSampleToCsvRow(sample);
        const auto countCommas = [](const std::string& s)
        { return static_cast<std::size_t>(std::count(s.begin(), s.end(), ',')); };
        assert(countCommas(header) == countCommas(row) &&
               "CSV header and row must have exactly the same number of columns");
    }

    // 3: ring buffer capacity matches kRingBufferCapacitySeconds/dt exactly,
    // landing in the requested ~4-5s / 240-300 sample range at 60Hz.
    {
        HairpinTelemetryRecorder recorder(track, "verification_scratch/hairpin_capacity_test", kSimulationDt, false);
        const std::size_t expected =
            static_cast<std::size_t>(HairpinTelemetryRecorder::kRingBufferCapacitySeconds / kSimulationDt);
        assert(recorder.getRingBufferCapacity() == expected &&
               "ring buffer capacity must match kRingBufferCapacitySeconds/dt exactly");
        assert(recorder.getRingBufferCapacity() >= 240 && recorder.getRingBufferCapacity() <= 300 &&
               "ring buffer capacity must land in the requested ~4-5 second / 240-300 sample range at 60Hz");
    }

    // 4: a disabled recorder never writes anything -- no dumps, no
    // directory, no files -- even when driven through a real, fast-crashing
    // population for several seconds (crash genome collides within a few
    // hundred steps, so this comfortably spans a generation transition).
    {
        const std::string scratchDir = "verification_scratch/hairpin_disabled_test";
        std::filesystem::remove_all(scratchDir);

        Population population(makeCrashGenome(), track, makeCarParams(), kSpawnPosition, kSpawnHeading,
                               makeTestPopulationConfig(6, 101u), mutationConfig, crossoverConfig, compatibilityConfig,
                               speciationConfig);
        HairpinTelemetryRecorder recorder(track, scratchDir, kSimulationDt, false);
        assert(!recorder.isEnabled() && "recorder constructed with enabled=false must report itself as disabled");

        for (int i = 0; i < 600; ++i) // 10s
        {
            population.update(kSimulationDt);
            recorder.update(population);
        }
        assert(recorder.getTotalDumpCount() == 0 && "a disabled recorder must never dump telemetry");
        assert(!std::filesystem::exists(scratchDir) &&
               "a disabled recorder must never create its results directory, let alone any file in it");
    }

    // 5: an enabled recorder driven through a real population only ever
    // accumulates finite values, with a strictly increasing sampleIndex, and
    // never reports more buffered samples than its own capacity.
    {
        Population population(createDemonstrationGenome(), track, makeCarParams(), kSpawnPosition, kSpawnHeading,
                               makeTestPopulationConfig(4, 202u), mutationConfig, crossoverConfig, compatibilityConfig,
                               speciationConfig);
        HairpinTelemetryRecorder recorder(track, "verification_scratch/hairpin_finite_test", kSimulationDt, true);

        for (int i = 0; i < 120; ++i) // 2s
        {
            population.update(kSimulationDt);
            recorder.update(population);
        }

        const std::vector<HairpinTelemetrySample> ordered = recorder.getOrderedSamples();
        assert(!ordered.empty() && "setup must have captured at least one sample");
        assert(ordered.size() <= recorder.getRingBufferCapacity() &&
               "ring buffer must never report more samples than its capacity");
        for (const HairpinTelemetrySample& s : ordered)
        {
            assert(std::isfinite(s.posX) && std::isfinite(s.posY) && std::isfinite(s.heading) &&
                   std::isfinite(s.bestProgress) && std::isfinite(s.speed) && std::isfinite(s.forwardVelocity) &&
                   std::isfinite(s.lateralVelocity) && std::isfinite(s.slipAngle) && std::isfinite(s.yawRate) &&
                   std::isfinite(s.steeringCmd) && std::isfinite(s.throttleCmd) && std::isfinite(s.brakeCmd) &&
                   std::isfinite(s.rawSteering) && std::isfinite(s.rawThrottle) && std::isfinite(s.rawBrake) &&
                   std::isfinite(s.obsSensorM60) && std::isfinite(s.obsSensor0) && std::isfinite(s.obsSensorP60) &&
                   std::isfinite(s.sensor0Px) && std::isfinite(s.minForwardWallDistancePx) &&
                   std::isfinite(s.frontSlipAngleTrue) && std::isfinite(s.rearSlipAngleTrue) &&
                   std::isfinite(s.frontGripUtilization) && std::isfinite(s.rearGripUtilization) &&
                   std::isfinite(s.trackDirectionAngle) && std::isfinite(s.headingError) &&
                   "every numeric telemetry field must stay finite for a normal simulation run");
        }
        for (std::size_t i = 1; i < ordered.size(); ++i)
        {
            assert(ordered[i].sampleIndex == ordered[i - 1].sampleIndex + 1 &&
                   "sampleIndex must increase by exactly 1 per captured frame, in buffer order");
        }
    }

    // 6: summarizeHairpinFailure() reports the FIRST brake-onset and FIRST
    // wrong-way sample in a synthetic sequence, never a later one.
    {
        std::vector<HairpinTelemetrySample> samples(5);
        for (std::size_t i = 0; i < samples.size(); ++i)
        {
            samples[i].sampleIndex = i;
        }
        samples[3].brakeCmd = 0.5f; // first brake-onset sample
        samples[3].speed = 123.0f;
        samples[4].brakeCmd = 0.9f; // later -- must not override index 3
        samples[4].speed = 999.0f;

        samples[2].wrongWay = true; // first spin-onset sample
        samples[2].simTime = 0.2f;
        samples[4].wrongWay = true; // later -- must not override index 2

        const HairpinFailureSummary summary = summarizeHairpinFailure(samples);
        assert(summary.hasBrakeOnset && summary.brakeOnsetSpeed == 123.0f &&
               "must report the FIRST brake-onset sample, not a later one");
        assert(summary.hasSpinOnset && std::fabs(summary.spinOnsetSimTime - 0.2f) < 1e-6f &&
               "must report the FIRST wrong-way sample, not a later one");
    }

    // 7: a sequence with no braking / no wrong-way orientation reports "no
    // onset" for both, rather than a false positive.
    {
        const std::vector<HairpinTelemetrySample> samples(3);
        const HairpinFailureSummary summary = summarizeHairpinFailure(samples);
        assert(!summary.hasBrakeOnset && !summary.hasSpinOnset &&
               "a sequence with no meaningful braking/wrong-way samples must report no onset for either");
    }

    // 8, 9, 10 & 11: attaching a HairpinTelemetryRecorder alongside a
    // Population and reading it every frame -- including across a real
    // generation transition (reproduce(), the only place mutation/crossover/
    // tournament selection touch the orchestration RNG) -- must not change
    // anything about that Population's own behavior. Proven by comparing a
    // full per-frame fitness trace (every individual, every frame) between a
    // telemetry-observed run and an unobserved twin built from the exact
    // same base genome/config/seed: this single bit-identical comparison is
    // the determinism/no-side-effect proof for control, fitness, RNG state,
    // and overall training determinism all at once.
    {
        auto driveAndCollectFitness = [&](bool attachRecorder) -> std::vector<float>
        {
            Population population(makeCrashGenome(), track, makeCarParams(), kSpawnPosition, kSpawnHeading,
                                   makeTestPopulationConfig(8, 4242u), mutationConfig, crossoverConfig,
                                   compatibilityConfig, speciationConfig);
            std::unique_ptr<HairpinTelemetryRecorder> recorder;
            if (attachRecorder)
            {
                recorder = std::make_unique<HairpinTelemetryRecorder>(
                    track, "verification_scratch/hairpin_determinism_test", kSimulationDt, true);
            }

            std::vector<float> trace;
            for (int i = 0; i < 900 && population.getGeneration() < 2; ++i) // up to 15s, spans >= 2 generations
            {
                population.update(kSimulationDt);
                if (recorder)
                {
                    recorder->update(population);
                }
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

        assert(withTelemetry.size() == withoutTelemetry.size() &&
               "identical scenarios with/without telemetry must run for the same number of steps"); // 10 (determinism)
        for (std::size_t i = 0; i < withTelemetry.size(); ++i)
        {
            assert(withTelemetry[i] == withoutTelemetry[i] &&
                   "telemetry capture must not alter car control, fitness, RNG-driven reproduction, or any other "
                   "training behavior -- traces with and without an attached recorder must be bit-identical"); // 8, 9, 11
        }
    }

    std::filesystem::remove_all("verification_scratch");

    TraceLog(LOG_INFO, "Hairpin telemetry verification: all deterministic checks passed");
}

} // namespace verification
