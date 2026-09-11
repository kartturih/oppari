#include "ui/HudRenderer.h"

#include <algorithm>
#include <cstdio>
#include <vector>

#include "raylib.h"

#include "ai/FitnessEvaluator.h"
#include "ai/neat/Individual.h"
#include "ai/neat/Species.h"
#include "simulation/TrackProgress.h"

#include "AppConfig.h"

namespace ui
{

using app::kSimWidth;

namespace
{

const char* finishReasonLabel(ai::EvaluationFinishReason reason)
{
    switch (reason)
    {
        case ai::EvaluationFinishReason::None:
            return "-";
        case ai::EvaluationFinishReason::Collision:
            return "Collision";
        case ai::EvaluationFinishReason::TimeLimit:
            return "TimeLimit";
        case ai::EvaluationFinishReason::NoProgress:
            return "NoProgress";
        case ai::EvaluationFinishReason::InsufficientInitialProgress:
            return "SlowStart";
    }
    return "-";
}

} // namespace

std::size_t selectHighlightedIndividual(const ai::neat::Population& population)
{
    std::size_t bestActiveIndex = 0;
    float bestActiveProgress = -1.0f;
    bool foundActive = false;

    for (std::size_t i = 0; i < population.size(); ++i)
    {
        const ai::neat::Individual& individual = population.getIndividual(i);
        if (!individual.isFinished())
        {
            const float progress = individual.getProgress().getBestProgress();
            if (!foundActive || progress > bestActiveProgress)
            {
                foundActive = true;
                bestActiveProgress = progress;
                bestActiveIndex = i;
            }
        }
    }

    if (foundActive)
    {
        return bestActiveIndex;
    }
    return population.getBestIndividualIndex();
}

void drawPopulationPanel(const ai::neat::Population& population, std::size_t highlightedIndex, int panelWidth,
                          int screenHeight)
{
    DrawRectangle(kSimWidth, 0, panelWidth, screenHeight, Color{30, 30, 30, 255});

    const int x = kSimWidth + 20;
    int y = 20;
    const int lineHeight = 22;

    DrawText("NEAT TRAINING - EXTREME TRACK", x, y, 20, RAYWHITE);
    y += lineHeight * 2;

    char line[128];

    DrawText("TRAINING", x, y, 18, YELLOW);
    y += lineHeight;
    std::snprintf(line, sizeof(line), "Generation: %d", static_cast<int>(population.getGeneration()));
    DrawText(line, x, y, 16, LIGHTGRAY);
    y += lineHeight;
    std::snprintf(line, sizeof(line), "Population size: %d", static_cast<int>(population.size()));
    DrawText(line, x, y, 16, LIGHTGRAY);
    y += lineHeight;
    std::snprintf(line, sizeof(line), "Running: %d   Finished: %d", static_cast<int>(population.getRunningCount()),
                  static_cast<int>(population.getFinishedCount()));
    DrawText(line, x, y, 16, LIGHTGRAY);
    y += lineHeight;
    // Active vs. stagnant species counts, read from the persistent Species
    // collection (Population::getReproductionStats() is empty before the
    // first generation transition).
    {
        int activeCount = 0;
        int stagnantCount = 0;
        for (const ai::neat::Species& s : population.getCurrentSpecies())
        {
            if (s.isStagnant())
            {
                ++stagnantCount;
            }
            else
            {
                ++activeCount;
            }
        }
        std::snprintf(line, sizeof(line), "Species: %d (active %d, stagnant %d)", static_cast<int>(population.getSpeciesCount()),
                      activeCount, stagnantCount);
        DrawText(line, x, y, 16, LIGHTGRAY);
        y += lineHeight;
        std::snprintf(line, sizeof(line), "Compatibility threshold: %.2f",
                      static_cast<double>(population.getCurrentCompatibilityThreshold()));
        DrawText(line, x, y, 16, LIGHTGRAY);
        y += lineHeight;
    }

    DrawText("REPRODUCTION: SPECIES-AWARE", x, y, 16, SKYBLUE);
    y += lineHeight * 2;

    // Locate which of the current generation's species the highlighted
    // individual belongs to, and -- if a generation transition has already
    // happened at least once -- its matching reproduction stats. Both
    // reads are purely informational; neither mutates Population state.
    {
        const std::vector<ai::neat::Species>& species = population.getCurrentSpecies();
        const ai::neat::Species* highlightedSpecies = nullptr;
        for (const ai::neat::Species& s : species)
        {
            const std::vector<std::size_t>& members = s.getMemberIndices();
            if (std::find(members.begin(), members.end(), highlightedIndex) != members.end())
            {
                highlightedSpecies = &s;
                break;
            }
        }

        DrawText("HIGHLIGHTED SPECIES", x, y, 18, YELLOW);
        y += lineHeight;
        if (highlightedSpecies != nullptr)
        {
            std::snprintf(line, sizeof(line), "Species ID: %d   Size: %d", highlightedSpecies->getId(),
                          static_cast<int>(highlightedSpecies->size()));
            DrawText(line, x, y, 16, LIGHTGRAY);
            y += lineHeight;

            // Persistent species history, read straight from the Species
            // object (age/historical best start at their documented
            // "never evaluated" state for a brand-new species; see Species.h).
            std::snprintf(line, sizeof(line), "Age: %d   Historical best: %.1f", static_cast<int>(highlightedSpecies->getAge()),
                          static_cast<double>(highlightedSpecies->getHistoricalBestFitness()));
            DrawText(line, x, y, 16, LIGHTGRAY);
            y += lineHeight;
            std::snprintf(line, sizeof(line), "No-improve: %d gens   Stagnant: %s",
                          static_cast<int>(highlightedSpecies->getGenerationsSinceImprovement()),
                          highlightedSpecies->isStagnant() ? "YES" : "NO");
            DrawText(line, x, y, 16, highlightedSpecies->isStagnant() ? RED : LIGHTGRAY);
            y += lineHeight;

            const ai::neat::Population::SpeciesReproductionStats* stats = nullptr;
            for (const ai::neat::Population::SpeciesReproductionStats& candidate : population.getReproductionStats())
            {
                if (candidate.speciesId == highlightedSpecies->getId())
                {
                    stats = &candidate;
                    break;
                }
            }
            if (stats != nullptr)
            {
                std::snprintf(line, sizeof(line), "Adjusted fitness sum: %.2f", static_cast<double>(stats->adjustedFitnessSum));
                DrawText(line, x, y, 16, LIGHTGRAY);
                y += lineHeight;
                std::snprintf(line, sizeof(line), "Offspring allocated: %d", static_cast<int>(stats->allocatedOffspring));
                DrawText(line, x, y, 16, LIGHTGRAY);
                y += lineHeight;
            }
            else
            {
                DrawText("(no reproduction stats yet)", x, y, 16, GRAY);
                y += lineHeight;
            }
        }
        else
        {
            DrawText("(unavailable)", x, y, 16, GRAY);
            y += lineHeight;
        }
    }
    y += lineHeight;

    const ai::neat::Individual& best = population.getIndividual(highlightedIndex);
    const ai::FitnessEvaluator& bestFitness = best.getFitnessEvaluator();
    DrawText("BEST CURRENT (highlighted)", x, y, 18, YELLOW);
    y += lineHeight;
    std::snprintf(line, sizeof(line), "Index: %d", static_cast<int>(highlightedIndex));
    DrawText(line, x, y, 16, LIGHTGRAY);
    y += lineHeight;
    {
        const simulation::TireDebugInfo& input = best.getCar().getTireDebugInfo();
        std::snprintf(line, sizeof(line), "Steering: %+.2f  Throttle: %.0f%%  Brake: %.0f%%",
                      static_cast<double>(input.steeringInput), static_cast<double>(input.throttleInput * 100.0f),
                      static_cast<double>(input.brakeInput * 100.0f));
        DrawText(line, x, y, 16, LIGHTGRAY);
        y += lineHeight;
    }

    // Fitness breakdown -- see FitnessEvaluator.h for the exact formula
    // each of these terms comes from.
    std::snprintf(line, sizeof(line), "Fitness: %.1f", static_cast<double>(best.getFitness()));
    DrawText(line, x, y, 16, LIGHTGRAY);
    y += lineHeight;
    std::snprintf(line, sizeof(line), "  Base: %.1f  Rate: %.3f", static_cast<double>(bestFitness.getBaseProgressFitness()),
                  static_cast<double>(bestFitness.getProgressRate()));
    DrawText(line, x, y, 16, LIGHTGRAY);
    y += lineHeight;
    std::snprintf(line, sizeof(line), "  RateBonus: %.1f  LapBonus: %.1f",
                  static_cast<double>(bestFitness.getProgressRateReward()), static_cast<double>(bestFitness.getLapSpeedBonus()));
    DrawText(line, x, y, 16, LIGHTGRAY);
    y += lineHeight;

    char bestLapStr[24];
    if (bestFitness.hasCompletedLap())
    {
        std::snprintf(bestLapStr, sizeof(bestLapStr), "%.2fs", static_cast<double>(bestFitness.getBestLapTime()));
    }
    else
    {
        std::snprintf(bestLapStr, sizeof(bestLapStr), "--");
    }
    std::snprintf(line, sizeof(line), "Progress: %.3f  BestLap: %s", static_cast<double>(best.getProgress().getBestProgress()),
                  bestLapStr);
    DrawText(line, x, y, 16, LIGHTGRAY);
    y += lineHeight;
    std::snprintf(line, sizeof(line), "Laps: %d   Checkpoints: %d", best.getProgress().getLapCount(),
                  best.getProgress().getTotalCheckpointsPassed());
    DrawText(line, x, y, 16, LIGHTGRAY);
    y += lineHeight * 2;

    // Compact TrackProgress projection debug block for the highlighted
    // individual -- see drawProjectionDebug()/reportSuspiciousProjectionJump()
    // for the matching world-space overlay and console diagnostics.
    {
        const simulation::TrackProgress::ProjectionDebugInfo& info = best.getProgress().getLastProjectionDebugInfo();
        DrawText("PROJECTION DEBUG", x, y, 18, YELLOW);
        y += lineHeight;
        std::snprintf(line, sizeof(line), "Projection: %s", info.usedRecovery ? "RECOVERY" : "LOCAL");
        DrawText(line, x, y, 16, info.usedRecovery ? ORANGE : LIGHTGRAY);
        y += lineHeight;
        std::snprintf(line, sizeof(line), "Index: %d -> %d (%+d)", static_cast<int>(info.previousIndex),
                      static_cast<int>(info.currentIndex), info.indexDelta);
        DrawText(line, x, y, 16, LIGHTGRAY);
        y += lineHeight;
        std::snprintf(line, sizeof(line), "Distance: %.1f px", static_cast<double>(info.distance));
        DrawText(line, x, y, 16, LIGHTGRAY);
        y += lineHeight * 2;
    }

    DrawText("LAST GENERATION", x, y, 18, YELLOW);
    y += lineHeight;
    std::snprintf(line, sizeof(line), "Best fitness: %.1f",
                  static_cast<double>(population.getLastGenerationBestFitness()));
    DrawText(line, x, y, 16, LIGHTGRAY);
    y += lineHeight * 2;

    DrawText("CONFIG / STATUS", x, y, 18, YELLOW);
    y += lineHeight;
    std::snprintf(line, sizeof(line), "Seed: %u", population.getPopulationConfig().randomSeed);
    DrawText(line, x, y, 16, LIGHTGRAY);
    y += lineHeight;
    std::snprintf(line, sizeof(line), "State: %s (%s)", best.isFinished() ? "FINISHED" : "RUNNING",
                  finishReasonLabel(best.getFitnessEvaluator().getFinishReason()));
    DrawText(line, x, y, 16, best.isFinished() ? RED : GREEN);
    y += lineHeight * 2;

    DrawText("Controls:", x, y, 18, RAYWHITE);
    y += lineHeight;
    DrawText("R restart current generation", x, y, 16, LIGHTGRAY);
    y += lineHeight;
    DrawText("TAB manual control mode", x, y, 16, LIGHTGRAY);
    y += lineHeight;
    DrawText("F: Toggle training speed", x, y, 16, LIGHTGRAY);
}

void drawManualPanel(const simulation::Car& car, int panelWidth, int screenHeight)
{
    DrawRectangle(kSimWidth, 0, panelWidth, screenHeight, Color{30, 30, 30, 255});

    const int x = kSimWidth + 20;
    int y = 20;
    const int lineHeight = 22;

    DrawText("MANUAL CONTROL - VEHICLE PHYSICS", x, y, 20, RAYWHITE);
    y += lineHeight * 2;

    char line[128];

    DrawText("STATE", x, y, 18, YELLOW);
    y += lineHeight;
    DrawText(car.isAlive() ? "Alive" : "Crashed", x, y, 16, car.isAlive() ? GREEN : RED);
    y += lineHeight * 2;

    DrawText("TELEMETRY", x, y, 18, YELLOW);
    y += lineHeight;
    std::snprintf(line, sizeof(line), "Speed: %.1f / %.1f px/s", static_cast<double>(car.getSpeed()),
                  static_cast<double>(car.getMaxSpeed()));
    DrawText(line, x, y, 16, LIGHTGRAY);
    y += lineHeight;
    std::snprintf(line, sizeof(line), "Forward vel: %.1f px/s", static_cast<double>(car.getForwardVelocity()));
    DrawText(line, x, y, 16, LIGHTGRAY);
    y += lineHeight;
    std::snprintf(line, sizeof(line), "Lateral vel: %.1f px/s", static_cast<double>(car.getLateralVelocity()));
    DrawText(line, x, y, 16, LIGHTGRAY);
    y += lineHeight;
    std::snprintf(line, sizeof(line), "Body slip angle: %.1f deg", static_cast<double>(car.getSlipAngle() * RAD2DEG));
    DrawText(line, x, y, 16, LIGHTGRAY);
    y += lineHeight;
    std::snprintf(line, sizeof(line), "Heading: %.1f deg", static_cast<double>(car.getHeading() * RAD2DEG));
    DrawText(line, x, y, 16, LIGHTGRAY);
    y += lineHeight;
    {
        const simulation::TireDebugInfo& input = car.getTireDebugInfo();
        std::snprintf(line, sizeof(line), "Steering: %+.2f  Throttle: %.0f%%  Brake: %.0f%%",
                      static_cast<double>(input.steeringInput), static_cast<double>(input.throttleInput * 100.0f),
                      static_cast<double>(input.brakeInput * 100.0f));
        DrawText(line, x, y, 16, LIGHTGRAY);
        y += lineHeight;
    }
    y += lineHeight;

    // Front/rear tire model telemetry -- tire slip angles, the Fx/Fy forces
    // each axle applies, yaw rate, and friction-circle saturation. Rear
    // grip utilization is the number to watch for power oversteer: it
    // reaches 100% exactly when the rear axle's combined drive+lateral
    // force has hit rearMaxTireForce and further throttle can only come at
    // the expense of rear lateral force.
    {
        const simulation::TireDebugInfo& tire = car.getTireDebugInfo();
        DrawText("TIRE MODEL", x, y, 18, YELLOW);
        y += lineHeight;
        std::snprintf(line, sizeof(line), "Steering angle: %.1f deg", static_cast<double>(tire.steeringAngle * RAD2DEG));
        DrawText(line, x, y, 16, LIGHTGRAY);
        y += lineHeight;
        std::snprintf(line, sizeof(line), "Yaw rate: %.2f rad/s", static_cast<double>(tire.yawRate));
        DrawText(line, x, y, 16, LIGHTGRAY);
        y += lineHeight;
        std::snprintf(line, sizeof(line), "Front slip: %.1f deg  Fy=%.0f", static_cast<double>(tire.frontSlipAngle * RAD2DEG),
                      static_cast<double>(tire.frontForceY));
        DrawText(line, x, y, 16, LIGHTGRAY);
        y += lineHeight;
        std::snprintf(line, sizeof(line), "Rear slip: %.1f deg  Fx=%.0f Fy=%.0f", static_cast<double>(tire.rearSlipAngle * RAD2DEG),
                      static_cast<double>(tire.rearForceX), static_cast<double>(tire.rearForceY));
        DrawText(line, x, y, 16, LIGHTGRAY);
        y += lineHeight;
        std::snprintf(line, sizeof(line), "Front grip: %.0f%%", static_cast<double>(tire.frontGripUtilization * 100.0f));
        DrawText(line, x, y, 16, tire.frontGripUtilization > 0.95f ? RED : LIGHTGRAY);
        y += lineHeight;
        std::snprintf(line, sizeof(line), "Rear grip: %.0f%%", static_cast<double>(tire.rearGripUtilization * 100.0f));
        DrawText(line, x, y, 16, tire.rearGripUtilization > 0.95f ? RED : LIGHTGRAY);
        y += lineHeight * 2;
    }

    DrawText("Controls:", x, y, 18, RAYWHITE);
    y += lineHeight;
    DrawText("UP/W throttle", x, y, 16, LIGHTGRAY);
    y += lineHeight;
    DrawText("DOWN/S brake", x, y, 16, LIGHTGRAY);
    y += lineHeight;
    DrawText("LEFT/A, RIGHT/D steer", x, y, 16, LIGHTGRAY);
    y += lineHeight;
    DrawText("R reset car to spawn", x, y, 16, LIGHTGRAY);
    y += lineHeight;
    DrawText("TAB back to NEAT training", x, y, 16, LIGHTGRAY);
    y += lineHeight;
    DrawText("F: Toggle training speed", x, y, 16, LIGHTGRAY);
}

void drawTrainingSpeedHud(bool fastMode, float simSpeedMultiplier, bool simSpeedMultiplierValid)
{
    char line[64];
    const int x = 12;
    int y = 8;

    std::snprintf(line, sizeof(line), "Training speed: %s", fastMode ? "FAST" : "NORMAL");
    DrawText(line, x, y, 18, fastMode ? ORANGE : RAYWHITE);
    y += 20;

    if (fastMode && simSpeedMultiplierValid)
    {
        std::snprintf(line, sizeof(line), "Sim speed: %.1fx", static_cast<double>(simSpeedMultiplier));
        DrawText(line, x, y, 16, ORANGE);
    }
}

} // namespace ui
