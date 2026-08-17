#include <algorithm>
#include <cstddef>
#include <string>
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
using ui::progressRankColor;
using ui::reportSuspiciousProjectionJump;
using ui::selectHighlightedIndividual;

constexpr int kPanelWidth = 400;
constexpr int kScreenWidth = kSimWidth + kPanelWidth;
constexpr int kScreenHeight = kSimHeight;

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
    runMetadata.maxEvaluationTimeSeconds = 30.0f; // mirrors FitnessEvaluator.cpp's kMaxEvaluationTime -- see RunMetadata.h
    runMetadata.populationConfig = populationConfig;
    runMetadata.mutationConfig = mutationConfig;
    runMetadata.crossoverConfig = crossoverConfig;
    runMetadata.compatibilityConfig = compatibilityConfig;
    runMetadata.speciationConfig = speciationConfig;
    runMetadata.carParams = makeCarParams();
    training::TrainingLogger trainingLogger("results", runMetadata);
    TraceLog(LOG_INFO, "Training metrics logging to %s (metadata: %s)", trainingLogger.getCsvPath().c_str(),
             trainingLogger.getMetadataPath().c_str());

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

        if (manualMode)
        {
            manualCar.update(readManualCarInput(), kSimulationDt);
        }
        else
        {
            population.update(kSimulationDt);

            if (population.getGeneration() != lastLoggedGeneration)
            {
                lastLoggedGeneration = population.getGeneration();

                // The just-finished generation's full metrics row -- captured
                // inside Population::reproduce() before this transition
                // replaced m_individuals -- is persisted to CSV and
                // summarized on one console line.
                const training::GenerationMetrics& metrics = population.getLastGenerationMetrics();
                trainingLogger.logGeneration(metrics);

                TraceLog(LOG_INFO, "Gen %d | best %.1f | avg %.1f | median %.1f | species %d | best nodes %d | best connections %d",
                         static_cast<int>(metrics.generation), static_cast<double>(metrics.bestFitness),
                         static_cast<double>(metrics.avgFitness), static_cast<double>(metrics.medianFitness),
                         static_cast<int>(metrics.speciesCount), static_cast<int>(metrics.bestGenomeNodeCount),
                         static_cast<int>(metrics.bestGenomeConnectionGeneCount));
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
