#include <algorithm>
#include <cstddef>
#include <vector>

#include "raylib.h"

#include "ai/neat/CompatibilityConfig.h"
#include "ai/neat/CrossoverConfig.h"
#include "ai/neat/Individual.h"
#include "ai/neat/MutationConfig.h"
#include "ai/neat/Population.h"
#include "ai/neat/PopulationConfig.h"
#include "ai/neat/SpeciationConfig.h"
#include "simulation/Car.h"
#include "simulation/Track.h"
#include "simulation/TrackVisual.h"
#include "training/GenerationMetrics.h"
#include "training/TrainingLogger.h"

#include "AppConfig.h"
#include "input/ManualInput.h"
#include "telemetry/ChampionTelemetry.h"
#include "telemetry/HairpinTelemetry.h"
#include "ui/DebugRenderer.h"
#include "ui/HudRenderer.h"
#include "ui/SimulationRenderer.h"
#include "verification/VerificationRunner.h"

namespace
{

using app::createDemonstrationGenome;
using app::kSimHeight;
using app::kSimWidth;
using app::kSimulationDt;
using app::kSpawnHeading;
using app::kSpawnPosition;
using app::makeCarParams;
using app::makeTrackDefinition;

using input::readManualCarInput;

using ui::drawIndividualCar;
using ui::drawManualPanel;
using ui::drawPopulationPanel;
using ui::drawProjectionDebug;
using ui::drawTrainingSpeedHud;
using ui::progressRankColor;
using ui::reportSuspiciousProjectionJump;
using ui::selectHighlightedIndividual;

constexpr int kPanelWidth = 400;
constexpr int kScreenWidth = kSimWidth + kPanelWidth;
// Taller than kSimHeight: the population panel's text (species stats, best-
// individual breakdown, last-generation summary, controls) runs longer than
// the simulation area is tall. The simulation area itself stays kSimHeight
// (see the DrawRectangle(0, 0, kSimWidth, kSimHeight, ...) sim-area clear
// below), only the window/panel grow to fit the panel's content.
constexpr int kScreenHeight = 950;

// FAST training-speed mode: how many fixed 1/60s simulation steps run per
// rendered frame (rendering happens once after the whole batch, never mid-
// batch -- see the main loop). Chosen empirically (see the training-speed
// investigation this was built for): large enough for a real, measurable
// wall-clock speedup, small enough that one batch's uninterrupted compute
// stays well under the few-hundred-ms range where Windows would start
// flagging the window as unresponsive (no window-message pump runs during
// a batch, only before/after it, via IsKeyPressed/WindowShouldClose and
// BeginDrawing/EndDrawing).
constexpr int kFastModeStepsPerFrame = 100;

// How often (wall-clock seconds) the achieved simulated-seconds-per-wall-
// second multiplier is (re)measured and displayed -- long enough to average
// out per-frame noise, short enough to feel live. Reset (and the multiplier
// marked invalid again) on every Normal<->Fast toggle so a stale reading
// from the other mode is never shown.
constexpr double kSimSpeedMeasurementWindowSeconds = 0.5;

} // namespace

int main()
{
    InitWindow(kScreenWidth, kScreenHeight, "NEAT Car Simulation");

    SetTargetFPS(60);

    simulation::Track track(makeTrackDefinition());

    // The visual layer's GPU texture is created here, separately from
    // Track's own (GPU-free) construction above and only now that a
    // window/GL context exists (see TrackVisual.h).
    simulation::TrackVisual trackVisual(track.getDefinition().visualImagePath, kSimWidth, kSimHeight);

    verification::runAll(track);

    // The whole training run starts from one hand-built, deterministic
    // demonstration Genome -- Population copies and mutates it to build
    // generation 0; the Genome itself is never touched again afterward.
    const ai::neat::PopulationConfig populationConfig;
    const ai::neat::MutationConfig mutationConfig;
    const ai::neat::CrossoverConfig crossoverConfig;
    const ai::neat::CompatibilityConfig compatibilityConfig;
    const ai::neat::SpeciationConfig speciationConfig;

    ai::neat::Population population(createDemonstrationGenome(), track, makeCarParams(), kSpawnPosition,
                                     kSpawnHeading, populationConfig, mutationConfig, crossoverConfig,
                                     compatibilityConfig, speciationConfig);

    // One CSV row + one companion metadata file per program run, under
    // results/ (created if missing, never overwritten -- see
    // TrainingLogger.h). Metadata captures everything needed to reproduce
    // this exact run later: the random seed the whole run is deterministic
    // from, the active track, and every NEAT/vehicle-physics config
    // constructed above/below.
    training::RunMetadata runMetadata;
    runMetadata.buildVersion = OPPARI_BUILD_VERSION;
    runMetadata.trackName = "extreme";
    runMetadata.targetLapCount = ai::kTargetLapCount;
    runMetadata.safetyTimeoutSeconds = ai::kSafetyTimeoutSeconds;
    runMetadata.populationConfig = populationConfig;
    runMetadata.mutationConfig = mutationConfig;
    runMetadata.crossoverConfig = crossoverConfig;
    runMetadata.compatibilityConfig = compatibilityConfig;
    runMetadata.speciationConfig = speciationConfig;
    runMetadata.carParams = makeCarParams();
    training::TrainingLogger trainingLogger("results", runMetadata);
    TraceLog(LOG_INFO, "Training metrics logging to %s (metadata: %s)", trainingLogger.getCsvPath().c_str(),
             trainingLogger.getMetadataPath().c_str());

    // Debugging-only instrument for the recurring first-hairpin spin failure
    // -- entirely separate from trainingLogger above (never touches its CSV
    // or GenerationMetrics), disabled in one place via
    // telemetry::kHairpinTelemetryEnabled. See HairpinTelemetry.h.
    telemetry::HairpinTelemetryRecorder hairpinTelemetry(track, "results", kSimulationDt);

    // Whole-run champion telemetry -- entirely separate from hairpinTelemetry
    // above (different instrument, different question: "how does the WHOLE
    // evaluation of the best car look", not "what happens right before one
    // car fails in the first hairpin"). Disabled in one place via
    // telemetry::kChampionTelemetryEnabled; which generations get captured
    // is controlled by telemetry::kChampionTelemetryGenerations. Registered
    // as Population's per-step observer rather than called after
    // population.update() (contrast hairpinTelemetry.update() below) because
    // it needs to see a just-finished generation's individuals BEFORE
    // reproduce() (called from inside that same population.update()) replaces
    // them -- see ChampionTelemetry.h's file comment and
    // Population::setPerStepObserver()'s comment for exactly why.
    telemetry::ChampionTelemetryRecorder championTelemetry("results", kSimulationDt);
    population.setPerStepObserver(
        [&championTelemetry](const ai::neat::Population& pop, bool generationJustFinished)
        { championTelemetry.onPopulationStep(pop, generationJustFinished); });

    // A standalone manual-control car, entirely independent of `population`
    // -- lets the vehicle handling be driven and felt directly (TAB to
    // toggle) without waiting on NEAT. Reset to spawn every time manual
    // mode is (re-)entered. Never touched by Population/FitnessEvaluator/
    // AIController.
    simulation::Car manualCar(makeCarParams(), track);
    manualCar.reset(kSpawnPosition, kSpawnHeading);
    bool manualMode = false;

    // Diagnostic only: logs every generation transition so training
    // progress is visible from console output too, not only the on-screen
    // panel. Reads Population's state only; never influences it.
    std::size_t lastLoggedGeneration = population.getGeneration();
    bool brakeFirstUseLogged = false;

    // Runtime-selectable NORMAL/FAST training-speed mode (F to toggle -- see
    // the main loop below). FAST removes SetTargetFPS's real-time pacing and
    // runs kFastModeStepsPerFrame fixed simulation steps per rendered frame
    // instead of one; the fixed step itself (kSimulationDt, always 1/60s) is
    // identical in both modes, and Population/Car never read wall-clock or
    // frame time, so which mode is active -- and how many render calls
    // happen -- cannot change simulation state, RNG call order, or results
    // (see verifyTrainingSpeedDeterminism() for the structural proof: this
    // is exactly why NORMAL and FAST are guaranteed to reproduce identical
    // evolutionary runs for the same seed/config).
    bool fastMode = false;

    // Simulated-seconds-per-wall-second multiplier, measured (never
    // fabricated) over rolling kSimSpeedMeasurementWindowSeconds windows;
    // invalid (not yet shown) until the first window completes after
    // startup or after a mode toggle.
    double simSpeedWindowStartWallTime = GetTime();
    float simSecondsInWindow = 0.0f;
    float simSpeedMultiplier = 0.0f;
    bool simSpeedMultiplierValid = false;

    while (!WindowShouldClose())
    {
        if (IsKeyPressed(KEY_TAB))
        {
            manualMode = !manualMode;
            if (manualMode)
            {
                manualCar.reset(kSpawnPosition, kSpawnHeading);
            }
        }

        if (IsKeyPressed(KEY_R))
        {
            if (manualMode)
            {
                manualCar.reset(kSpawnPosition, kSpawnHeading);
            }
            else
            {
                // Restarts the current generation's evaluation from its
                // existing Genomes -- same generation number, same genomes,
                // only Car/Progress/Fitness state resets.
                population.restartGeneration();
            }
        }

        if (IsKeyPressed(KEY_F))
        {
            fastMode = !fastMode;
            // 0 = uncapped in raylib: FAST renders (and polls input) as
            // fast as the CPU/GPU allow instead of pacing to 60Hz: In this
            // application the per-frame cost is now the training-step
            // batch's compute, not vsync, since rendering itself is cheap
            // and happens only once per batch either way.
            SetTargetFPS(fastMode ? 0 : 60);
            simSpeedWindowStartWallTime = GetTime();
            simSecondsInWindow = 0.0f;
            simSpeedMultiplierValid = false;
        }

        if (manualMode)
        {
            // Manual driving always advances one real-time step per
            // rendered frame regardless of fastMode -- batching human input
            // would just make control feel laggy/stale; FAST here only
            // still uncaps the render rate (harmless for a single car).
            manualCar.update(readManualCarInput(), kSimulationDt);
        }
        else
        {
            // NORMAL: exactly one step per rendered frame, identical to
            // this loop's behavior before FAST mode existed. FAST: up to
            // kFastModeStepsPerFrame steps, still each individually
            // dt=kSimulationDt=1/60s -- never a larger or variable dt.
            const int stepsThisFrame = fastMode ? kFastModeStepsPerFrame : 1;
            for (int step = 0; step < stepsThisFrame; ++step)
            {
                population.update(kSimulationDt);
                hairpinTelemetry.update(population);
                simSecondsInWindow += kSimulationDt;

                if (population.getGeneration() != lastLoggedGeneration)
                {
                    lastLoggedGeneration = population.getGeneration();

                    // The just-finished generation's full metrics row --
                    // captured inside Population::reproduce() before this
                    // transition replaced m_individuals -- is persisted to
                    // CSV and summarized on one console line.
                    // generationDurationSeconds here is (and always was)
                    // simulated elapsed time (FitnessEvaluator::
                    // getElapsedTime()), never wall-clock -- unaffected by
                    // which speed mode produced it.
                    const training::GenerationMetrics& metrics = population.getLastGenerationMetrics();
                    trainingLogger.logGeneration(metrics);

                    TraceLog(LOG_INFO, "Gen %d | best %.1f | avg %.1f | median %.1f | species %d | best nodes %d | best connections %d",
                             static_cast<int>(metrics.generation), static_cast<double>(metrics.bestFitness),
                             static_cast<double>(metrics.avgFitness), static_cast<double>(metrics.medianFitness),
                             static_cast<int>(metrics.speciesCount), static_cast<int>(metrics.bestGenomeNodeCount),
                             static_cast<int>(metrics.bestGenomeConnectionGeneCount));

                    // Observation only: announce (once) the first generation
                    // whose best individual actually used the physical brake.
                    if (!brakeFirstUseLogged &&
                        metrics.bestDriving.physicalBrakeUsageFraction > 0.0f)
                    {
                        brakeFirstUseLogged = true;
                        TraceLog(LOG_INFO, "First physical brake use by a champion: gen %d (%.2f%% of frames, onset %.0f px/s)",
                                 static_cast<int>(metrics.generation),
                                 static_cast<double>(metrics.bestDriving.physicalBrakeUsageFraction) * 100.0,
                                 static_cast<double>(metrics.bestDriving.brakeOnsetSpeed));
                    }
                }
            }
        }

        // Measured, not fabricated: only ever computed from actual
        // simulated-dt accumulated versus actual GetTime() elapsed.
        {
            const double wallNow = GetTime();
            const double wallElapsed = wallNow - simSpeedWindowStartWallTime;
            if (wallElapsed >= kSimSpeedMeasurementWindowSeconds)
            {
                simSpeedMultiplier = static_cast<float>(simSecondsInWindow / wallElapsed);
                simSpeedMultiplierValid = true;
                simSpeedWindowStartWallTime = wallNow;
                simSecondsInWindow = 0.0f;
            }
        }

        BeginDrawing();
        ClearBackground(BLACK);

        DrawRectangle(0, 0, kSimWidth, kSimHeight, BLACK);

        // The visual layer is a plain image draw -- track_visual.png,
        // loaded once into trackVisual above -- entirely independent of the
        // mask Car/sensors collide against (see Track::isDrivable()) and of
        // the centerline TrackProgress uses.
        trackVisual.draw();

        drawTrainingSpeedHud(fastMode, simSpeedMultiplier, simSpeedMultiplierValid);

        if (manualMode)
        {
            drawIndividualCar(manualCar, SKYBLUE, true);
            drawManualPanel(manualCar, kPanelWidth, kScreenHeight);
        }
        else
        {
            const std::size_t highlightedIndex = selectHighlightedIndividual(population);

            // Color every car by its progress ranking (leading = green,
            // trailing = red) so the population's spread is visible at a
            // glance; finished cars are drawn dimmed regardless of rank. This
            // only reads Population's state -- rendering never feeds back into
            // fitness or evolution.
            std::vector<std::size_t> rankOrder(population.size());
            for (std::size_t i = 0; i < population.size(); ++i)
            {
                rankOrder[i] = i;
            }
            std::sort(rankOrder.begin(), rankOrder.end(),
                      [&population](std::size_t a, std::size_t b)
                      {
                          return population.getIndividual(a).getProgress().getBestProgress() >
                                 population.getIndividual(b).getProgress().getBestProgress();
                      });
            std::vector<float> normalizedRank(population.size());
            for (std::size_t rank = 0; rank < rankOrder.size(); ++rank)
            {
                normalizedRank[rankOrder[rank]] = (population.size() > 1)
                                                       ? static_cast<float>(rank) / static_cast<float>(population.size() - 1)
                                                       : 0.0f;
            }

            for (std::size_t i = 0; i < population.size(); ++i)
            {
                const ai::neat::Individual& individual = population.getIndividual(i);
                const Color color =
                    individual.isFinished() ? Color{70, 70, 70, 140} : progressRankColor(normalizedRank[i]);
                drawIndividualCar(individual.getCar(), color, i == highlightedIndex);
            }

            // Projection debug overlay + suspicious-jump detection, for the
            // highlighted individual only (never every car).
            {
                const ai::neat::Individual& highlighted = population.getIndividual(highlightedIndex);
                drawProjectionDebug(highlighted.getCar(), highlighted.getProgress());
                reportSuspiciousProjectionJump(highlightedIndex, highlighted.getProgress(), highlighted.getCar());
            }

            drawPopulationPanel(population, highlightedIndex, kPanelWidth, kScreenHeight);
        }

        EndDrawing();
    }

    CloseWindow();

    return 0;
}
