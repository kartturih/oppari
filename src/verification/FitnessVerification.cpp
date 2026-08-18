#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "raylib.h"

#include "ai/AIController.h"
#include "ai/FitnessEvaluator.h"
#include "ai/NeuralNetwork.h"
#include "ai/Observation.h"
#include "ai/neat/CompatibilityConfig.h"
#include "ai/neat/CompatibilityDistance.h"
#include "ai/neat/ConnectionGene.h"
#include "ai/neat/CrossoverConfig.h"
#include "ai/neat/Genome.h"
#include "ai/neat/GenomeCrossover.h"
#include "ai/neat/GenomeMutator.h"
#include "ai/neat/Individual.h"
#include "ai/neat/InnovationTracker.h"
#include "ai/neat/MutationConfig.h"
#include "ai/neat/NodeGene.h"
#include "ai/neat/PhenotypeBuilder.h"
#include "ai/neat/Population.h"
#include "ai/neat/PopulationConfig.h"
#include "ai/neat/SpeciationConfig.h"
#include "ai/neat/Speciator.h"
#include "ai/neat/Species.h"
#include "simulation/Car.h"
#include "simulation/Track.h"
#include "simulation/TrackProgress.h"
#include "simulation/TrackVisual.h"
#include "training/GenerationMetrics.h"
#include "training/TrainingLogger.h"

#include "AppConfig.h"
#include "verification/Verifications.h"

namespace verification
{

using app::kSimulationDt;
using app::kSpawnHeading;
using app::kSpawnPosition;
using app::makeCarParams;


// Deterministic check of ai::FitnessEvaluator. Reads only Car::isAlive()
// and TrackProgress's getters -- no Genome/NeuralNetwork here. update() has
// no mode parameter, so manual and AI control share the same call.
void verifyFitnessEvaluator(const simulation::Track& track)
{
    using track_progress_verify::positionAtLapPosition;
    constexpr float kEps = 1e-3f;

    simulation::Car car(makeCarParams(), track);

    // 1: reset() produces exactly zero fitness (and every component)/elapsed
    // time and a fresh, unfinished evaluation.
    {
        ai::FitnessEvaluator evaluator;
        evaluator.reset();
        assert(evaluator.getFitness() == 0.0f && evaluator.getElapsedTime() == 0.0f &&
               !evaluator.isEvaluationFinished() && evaluator.getFinishReason() == ai::EvaluationFinishReason::None &&
               "reset must produce zero fitness/elapsed time and an unfinished evaluation");
        assert(evaluator.getBaseProgressFitness() == 0.0f && evaluator.getProgressRate() == 0.0f &&
               evaluator.getProgressRateReward() == 0.0f && evaluator.getLapSpeedBonus() == 0.0f &&
               "reset must clear every fitness component"); // 27
    }

    // 2, 3 & 40: no positive reward from elapsed survival time alone -- with
    // the car stationary at spawn, fitness must stay exactly zero the whole
    // time it sits there (well under the no-progress timeout).
    {
        car.reset(kSpawnPosition, kSpawnHeading);
        simulation::TrackProgress progress(track);
        progress.reset(car);
        ai::FitnessEvaluator evaluator;
        evaluator.reset();

        for (int i = 0; i < 150 && !evaluator.isEvaluationFinished(); ++i) // 2.5s of no movement, under the 3s timeout
        {
            progress.update(car);
            evaluator.update(car, progress, kSimulationDt);
            assert(evaluator.getFitness() == 0.0f &&
                   "standing still at zero progress must never earn positive fitness from elapsed time alone");
        }
        assert(!evaluator.isEvaluationFinished() && "2.5 seconds of no movement must stay under the no-progress timeout");
    }

    // 14: once some progress has been made, standing still afterward must
    // never increase progressRateReward or total fitness.
    {
        car.reset(positionAtLapPosition(track, 0.0f), kSpawnHeading);
        simulation::TrackProgress progress(track);
        progress.reset(car);
        ai::FitnessEvaluator evaluator;
        evaluator.reset();

        car.reset(positionAtLapPosition(track, 0.1f), kSpawnHeading);
        progress.update(car);
        evaluator.update(car, progress, kSimulationDt);
        float lastRateReward = evaluator.getProgressRateReward();
        float lastFitness = evaluator.getFitness();

        for (int i = 0; i < 120; ++i) // 2s of standing still at the same progress
        {
            progress.update(car); // car did not move; bestProgress unchanged
            evaluator.update(car, progress, kSimulationDt);
            assert(evaluator.getProgressRateReward() <= lastRateReward + kEps &&
                   "waiting must never increase progress-rate reward");
            assert(evaluator.getFitness() <= lastFitness + kEps && "waiting must never increase total fitness");
            lastRateReward = evaluator.getProgressRateReward();
            lastFitness = evaluator.getFitness();
        }
        car.reset(kSpawnPosition, kSpawnHeading);
    }

    // 4: forward progress increases base progress fitness (and therefore
    // total fitness).
    {
        car.reset(positionAtLapPosition(track, 0.0f), kSpawnHeading);
        simulation::TrackProgress progress(track);
        progress.reset(car);
        ai::FitnessEvaluator evaluator;
        evaluator.reset();

        evaluator.update(car, progress, kSimulationDt);
        const float baseBefore = evaluator.getBaseProgressFitness();
        const float fitnessBefore = evaluator.getFitness();

        car.reset(positionAtLapPosition(track, 0.10f), kSpawnHeading);
        progress.update(car);
        evaluator.update(car, progress, kSimulationDt);
        assert(evaluator.getBaseProgressFitness() > baseBefore && "forward progress must increase base progress fitness");
        assert(evaluator.getFitness() > fitnessBefore && "forward progress must increase total fitness");
    }

    // 13: backward movement must not increase base progress fitness.
    {
        car.reset(positionAtLapPosition(track, 0.0f), kSpawnHeading);
        simulation::TrackProgress progress(track);
        progress.reset(car);
        ai::FitnessEvaluator evaluator;
        evaluator.reset();

        car.reset(positionAtLapPosition(track, 0.15f), kSpawnHeading);
        progress.update(car);
        evaluator.update(car, progress, kSimulationDt);
        const float baseAfterForward = evaluator.getBaseProgressFitness();

        car.reset(positionAtLapPosition(track, 0.05f), kSpawnHeading);
        progress.update(car);
        evaluator.update(car, progress, kSimulationDt);
        assert(evaluator.getBaseProgressFitness() == baseAfterForward &&
               "backward movement must not increase base progress fitness");
    }

    // 5: passing a checkpoint increases base progress fitness.
    {
        car.reset(positionAtLapPosition(track, 0.0f), kSpawnHeading);
        simulation::TrackProgress progress(track);
        progress.reset(car);
        ai::FitnessEvaluator evaluator;
        evaluator.reset();

        evaluator.update(car, progress, kSimulationDt);
        const float baseAtStart = evaluator.getBaseProgressFitness();

        car.reset(positionAtLapPosition(track, 1.0f / static_cast<float>(simulation::TrackProgress::kCheckpointCount)),
                  kSpawnHeading);
        progress.update(car);
        evaluator.update(car, progress, kSimulationDt);
        assert(progress.getTotalCheckpointsPassed() >= 1 && "the setup must actually pass at least one checkpoint");
        assert(evaluator.getBaseProgressFitness() > baseAtStart && "passing a checkpoint must increase base progress fitness");
    }

    // 6: completing a lap increases base progress fitness with a distinct
    // lap bonus on top of the progress reward already earned.
    {
        car.reset(positionAtLapPosition(track, 0.0f), kSpawnHeading);
        simulation::TrackProgress progress(track);
        progress.reset(car);
        ai::FitnessEvaluator evaluator;
        evaluator.reset();

        const float toAlmostFull[] = {0.15f, 0.30f, 0.45f, 0.60f, 0.75f, 0.90f, 0.95f};
        for (float p : toAlmostFull)
        {
            car.reset(positionAtLapPosition(track, p), kSpawnHeading);
            progress.update(car);
            evaluator.update(car, progress, kSimulationDt);
        }
        const float baseBeforeLap = evaluator.getBaseProgressFitness();
        assert(progress.getLapCount() == 0 && "setup must not have completed a lap yet");

        car.reset(positionAtLapPosition(track, 0.05f), kSpawnHeading);
        progress.update(car);
        evaluator.update(car, progress, kSimulationDt);
        assert(progress.getLapCount() == 1 && "the final step must complete exactly one lap");
        assert(evaluator.getBaseProgressFitness() > baseBeforeLap && "completing a lap must increase base progress fitness");
        car.reset(kSpawnPosition, kSpawnHeading);
    }

    // 7 & 8: the same progress reached in less time gives strictly higher
    // fitness -- Car A (5s) must beat Car B (10s) at the same 0.5-lap point.
    {
        auto reachProgressInTime = [&](float targetProgress, float totalTime) -> float
        {
            simulation::Car localCar(makeCarParams(), track);
            localCar.reset(positionAtLapPosition(track, 0.0f), kSpawnHeading);
            simulation::TrackProgress localProgress(track);
            localProgress.reset(localCar);
            ai::FitnessEvaluator evaluator;
            evaluator.reset();

            // Small increments, each within TrackProgress's plausibility gate,
            // so bestProgress legitimately reaches targetProgress (a direct
            // jump of more than ~0.2 laps would be rejected as implausible).
            constexpr float kStep = 0.15f;
            float p = 0.0f;
            while (p + kStep < targetProgress)
            {
                p += kStep;
                localCar.reset(positionAtLapPosition(track, p), kSpawnHeading);
                localProgress.update(localCar);
            }
            localCar.reset(positionAtLapPosition(track, targetProgress), kSpawnHeading);
            localProgress.update(localCar);

            // FitnessEvaluator reads TrackProgress's current state, not an
            // integral over time, so one update() with totalTime suffices.
            evaluator.update(localCar, localProgress, totalTime);
            return evaluator.getFitness();
        };

        const float fitnessCarA = reachProgressInTime(0.5f, 5.0f);
        const float fitnessCarB = reachProgressInTime(0.5f, 10.0f);
        assert(fitnessCarA > fitnessCarB &&
               "the same progress reached in less time must give higher fitness (Car A/5s must beat Car B/10s)");
    }

    // 9: substantially greater progress beats a much faster but far less
    // advanced car -- Car C (0.8 laps in 10s) must beat Car D (0.2 laps in
    // 2s), since kProgressRateScale is deliberately small relative to
    // kProgressPointsPerLap.
    {
        auto reachProgressInTime = [&](float targetProgress, float totalTime) -> float
        {
            simulation::Car localCar(makeCarParams(), track);
            localCar.reset(positionAtLapPosition(track, 0.0f), kSpawnHeading);
            simulation::TrackProgress localProgress(track);
            localProgress.reset(localCar);
            ai::FitnessEvaluator evaluator;
            evaluator.reset();

            // Same stepping approach as the previous test block.
            constexpr float kStep = 0.15f;
            float p = 0.0f;
            while (p + kStep < targetProgress)
            {
                p += kStep;
                localCar.reset(positionAtLapPosition(track, p), kSpawnHeading);
                localProgress.update(localCar);
            }
            localCar.reset(positionAtLapPosition(track, targetProgress), kSpawnHeading);
            localProgress.update(localCar);
            evaluator.update(localCar, localProgress, totalTime);
            return evaluator.getFitness();
        };

        const float fitnessCarC = reachProgressInTime(0.8f, 10.0f);
        const float fitnessCarD = reachProgressInTime(0.2f, 2.0f);
        assert(fitnessCarC > fitnessCarD &&
               "substantially greater progress must still beat a much faster but far less advanced car");
    }

    // 10 & 11: progressRate matches bestProgress / max(elapsedTime,
    // smallTimeEpsilon) exactly, and stays finite even at elapsedTime == 0
    // (smallTimeEpsilon = 0.1f floor).
    {
        car.reset(positionAtLapPosition(track, 0.0f), kSpawnHeading);
        simulation::TrackProgress progress(track);
        progress.reset(car);
        ai::FitnessEvaluator evaluator;
        evaluator.reset();

        car.reset(positionAtLapPosition(track, 0.2f), kSpawnHeading);
        progress.update(car);
        evaluator.update(car, progress, 4.0f); // well above smallTimeEpsilon: max() is a no-op here
        const float expectedRate = progress.getBestProgress() / 4.0f;
        assert(std::fabs(evaluator.getProgressRate() - expectedRate) < kEps &&
               "progressRate must exactly match bestProgress / elapsedTime once elapsedTime is well above the epsilon floor"); // 10

        ai::FitnessEvaluator zeroTimeEvaluator;
        zeroTimeEvaluator.reset();
        simulation::Car zeroTimeCar(makeCarParams(), track);
        zeroTimeCar.reset(positionAtLapPosition(track, 0.0f), kSpawnHeading);
        simulation::TrackProgress zeroTimeProgress(track);
        zeroTimeProgress.reset(zeroTimeCar);
        zeroTimeCar.reset(positionAtLapPosition(track, 0.2f), kSpawnHeading); // 0.2 laps of progress since reset (within the plausibility gate)
        zeroTimeProgress.update(zeroTimeCar);
        zeroTimeEvaluator.update(zeroTimeCar, zeroTimeProgress, 0.0f); // zero deltaTime -- elapsedTime stays 0
        assert(std::isfinite(zeroTimeEvaluator.getProgressRate()) &&
               "progressRate must remain finite when elapsedTime is exactly zero"); // 11
        const float expectedZeroTimeRate = zeroTimeProgress.getBestProgress() / 0.1f; // documented smallTimeEpsilon
        assert(std::fabs(zeroTimeEvaluator.getProgressRate() - expectedZeroTimeRate) < kEps &&
               "progressRate at elapsedTime == 0 must use the documented smallTimeEpsilon floor");
    }

    // 12: progressRate never uses the Car's instantaneous speed -- driven
    // with real Car physics (nonzero, varying velocity) and cross-checked
    // against the exact bestProgress/elapsedTime formula.
    {
        car.reset(kSpawnPosition, kSpawnHeading);
        simulation::TrackProgress progress(track);
        progress.reset(car);
        ai::FitnessEvaluator evaluator;
        evaluator.reset();

        simulation::CarInput driveForward;
        driveForward.throttle = 1.0f;
        driveForward.steering = 0.0f;
        for (int i = 0; i < 60 && car.isAlive(); ++i) // 1s of real acceleration -- velocity is nonzero and changing
        {
            car.update(driveForward, kSimulationDt);
            progress.update(car);
            evaluator.update(car, progress, kSimulationDt);
        }
        assert(car.getSpeed() > 1.0f && "setup must actually be moving (nonzero instantaneous speed) for this check");
        const float expectedRate = progress.getBestProgress() / evaluator.getElapsedTime();
        assert(std::fabs(evaluator.getProgressRate() - expectedRate) < kEps &&
               "progressRate must match the pure bestProgress/elapsedTime formula regardless of the car's instantaneous speed");
        car.reset(kSpawnPosition, kSpawnHeading);
    }

    // 15, 16, 17 & 18: lap timing -- the first completed lap records a lap
    // time matching elapsedTime; a slower second lap updates last-lap time
    // without worsening best; a faster third lap becomes the new best.
    {
        car.reset(positionAtLapPosition(track, 0.0f), kSpawnHeading);
        simulation::TrackProgress progress(track);
        progress.reset(car);
        ai::FitnessEvaluator evaluator;
        evaluator.reset();

        assert(!evaluator.hasCompletedLap() && evaluator.getBestLapTime() == 0.0f && evaluator.getLastLapTime() == 0.0f &&
               "no completed lap yet must read as the documented zero sentinel"); // 20 (setup half)

        // Advances progress just over one lap in small increments (within
        // TrackProgress's plausibility gate), then spends lapTime seconds
        // via one update() call so the recorded lap time is exactly lapTime.
        // The 2% overshoot past the seam avoids landing the final sample
        // ambiguously at the exact wrap point (float rounding in
        // Track::getPointAtDistance() could miss the wrap by a hair) while
        // staying well short of the next checkpoint (6.25% of a lap apart).
        float cumulativeP = 0.0f;
        auto driveOneLap = [&](float lapTime)
        {
            constexpr float kStep = 0.15f;
            float remaining = 1.02f;
            while (remaining > kStep)
            {
                cumulativeP += kStep;
                remaining -= kStep;
                car.reset(positionAtLapPosition(track, cumulativeP), kSpawnHeading);
                progress.update(car);
            }
            cumulativeP += remaining;
            car.reset(positionAtLapPosition(track, cumulativeP), kSpawnHeading);
            progress.update(car);
            evaluator.update(car, progress, lapTime);
        };

        // First lap: 3s total (all three laps' times must sum to well under
        // FitnessEvaluator's kMaxEvaluationTime before TimeLimit triggers).
        driveOneLap(3.0f);
        assert(progress.getLapCount() == 1 && "the first full lap must complete exactly one lap");
        assert(evaluator.hasCompletedLap() && "completing a lap must set hasCompletedLap()"); // 15 (part 1)
        assert(std::fabs(evaluator.getLastLapTime() - evaluator.getElapsedTime()) < kEps &&
               "the first lap's time must equal elapsedTime, since the first lap starts at time 0"); // 15 (part 2)
        assert(std::fabs(evaluator.getBestLapTime() - evaluator.getLastLapTime()) < kEps &&
               "the only completed lap so far must also be the best lap");
        const float firstLapTime = evaluator.getLastLapTime(); // 3s

        // Second lap: slower (5s) -- new last-lap time, but best must stay at firstLapTime.
        driveOneLap(5.0f);
        assert(progress.getLapCount() == 2 && "the second full lap must complete a second lap");
        assert(std::fabs(evaluator.getLastLapTime() - 5.0f) < kEps &&
               "a second, slower lap must record a new last-lap time"); // 16
        assert(std::fabs(evaluator.getBestLapTime() - firstLapTime) < kEps &&
               "a slower later lap must not worsen (increase) the recorded best lap time"); // 17 & 18

        // Third lap: faster (1s) -- best must now update to this new fastest time.
        driveOneLap(1.0f);
        assert(progress.getLapCount() == 3 && "the third full lap must complete a third lap");
        assert(std::fabs(evaluator.getLastLapTime() - 1.0f) < kEps && "the third lap's time must be recorded as the new last-lap time");
        assert(std::fabs(evaluator.getBestLapTime() - 1.0f) < kEps &&
               "a faster later lap must become the new best lap time"); // 17 (fastest wins)
        car.reset(kSpawnPosition, kSpawnHeading);
    }

    // 19, 20, 21 & 22: faster completed laps give a larger lap-speed bonus;
    // there is no lap-speed bonus at all before any lap is completed; and
    // the bonus stays finite and bounded (<= kMaxLapSpeedFactor *
    // kLapSpeedBonusScale = 2 * 200 = 400) even for a near-instant lap.
    {
        auto completeOneLapIn = [&](float lapTime) -> float
        {
            simulation::Car localCar(makeCarParams(), track);
            localCar.reset(positionAtLapPosition(track, 0.0f), kSpawnHeading);
            simulation::TrackProgress localProgress(track);
            localProgress.reset(localCar);
            ai::FitnessEvaluator evaluator;
            evaluator.reset();

            assert(evaluator.getLapSpeedBonus() == 0.0f && "there must be no lap-speed bonus before any lap is completed"); // 20

            // Drive one full lap via small increments (see driveOneLap
            // above), spending lapTime seconds via one update() call.
            constexpr float kStep = 0.15f;
            float remaining = 1.02f; // slight overshoot past the seam -- see driveOneLap's comment above
            float p = 0.0f;
            while (remaining > kStep)
            {
                p += kStep;
                remaining -= kStep;
                localCar.reset(positionAtLapPosition(track, p), kSpawnHeading);
                localProgress.update(localCar);
            }
            p += remaining;
            localCar.reset(positionAtLapPosition(track, p), kSpawnHeading);
            localProgress.update(localCar);
            evaluator.update(localCar, localProgress, lapTime);
            assert(localProgress.getLapCount() == 1 && "setup must complete exactly one lap");
            return evaluator.getLapSpeedBonus();
        };

        const float bonusFast = completeOneLapIn(5.0f);   // fast lap
        const float bonusSlow = completeOneLapIn(30.0f);  // slow lap
        const float bonusInstant = completeOneLapIn(0.01f); // degenerate near-zero-time lap

        assert(bonusFast > bonusSlow && "a faster completed lap must give a larger lap-speed bonus"); // 19
        assert(std::isfinite(bonusFast) && std::isfinite(bonusSlow) && std::isfinite(bonusInstant) &&
               "lap-speed bonus must always remain finite"); // 21
        constexpr float kMaxPossibleLapSpeedBonus = 400.0f; // kMaxLapSpeedFactor(2) * kLapSpeedBonusScale(200)
        assert(bonusFast <= kMaxPossibleLapSpeedBonus + kEps && bonusInstant <= kMaxPossibleLapSpeedBonus + kEps &&
               "lap-speed bonus must stay bounded even for a near-instant lap"); // 22
    }

    // 23: a collided (dead) car ends the evaluation with Collision, and
    // fitness/elapsed time/finish reason freeze from that point on.
    {
        car.reset(kSpawnPosition, kSpawnHeading);
        simulation::TrackProgress progress(track);
        progress.reset(car);
        ai::FitnessEvaluator evaluator;
        evaluator.reset();

        simulation::CarInput driveOffTrack;
        driveOffTrack.throttle = 1.0f;
        driveOffTrack.steering = 0.0f;
        for (int i = 0; i < 300 && car.isAlive(); ++i)
        {
            car.update(driveOffTrack, kSimulationDt);
            progress.update(car);
            evaluator.update(car, progress, kSimulationDt);
        }
        assert(!car.isAlive() && "driving straight for 5s must leave the road band and kill the car");
        assert(evaluator.isEvaluationFinished() &&
               evaluator.getFinishReason() == ai::EvaluationFinishReason::Collision &&
               "a dead car must end the evaluation with Collision");

        const float fitnessAtFinish = evaluator.getFitness();
        const float elapsedAtFinish = evaluator.getElapsedTime();
        evaluator.update(car, progress, 10.0f); // must be a no-op: evaluation already finished
        assert(evaluator.getFitness() == fitnessAtFinish && evaluator.getElapsedTime() == elapsedAtFinish &&
               "Collision must freeze fitness and elapsed time");
        car.reset(kSpawnPosition, kSpawnHeading);
    }

    // 24: reaching the maximum evaluation time ends it with TimeLimit --
    // progress is nudged forward every simulated second so the no-progress
    // timeout cannot pre-empt it -- and fitness freezes from that point on.
    {
        car.reset(positionAtLapPosition(track, 0.0f), kSpawnHeading);
        simulation::TrackProgress progress(track);
        progress.reset(car);
        ai::FitnessEvaluator evaluator;
        evaluator.reset();

        // Loop bound and the elapsed-time threshold below match
        // FitnessEvaluator's kMaxEvaluationTime (60.0f).
        float p = 0.0f;
        for (int second = 0; second < 62 && !evaluator.isEvaluationFinished(); ++second)
        {
            p += 0.01f;
            car.reset(positionAtLapPosition(track, std::fmod(p, 1.0f)), kSpawnHeading);
            progress.update(car);
            evaluator.update(car, progress, 1.0f);
        }
        assert(evaluator.isEvaluationFinished() &&
               evaluator.getFinishReason() == ai::EvaluationFinishReason::TimeLimit &&
               "reaching the maximum evaluation time must end the evaluation with TimeLimit");
        assert(evaluator.getElapsedTime() >= 60.0f && "elapsed time at TimeLimit must reach the configured maximum"); // 32

        const float fitnessAtFinish = evaluator.getFitness();
        const float elapsedAtFinish = evaluator.getElapsedTime();
        car.reset(positionAtLapPosition(track, 0.5f), kSpawnHeading); // would otherwise be a big progress jump
        progress.update(car);
        evaluator.update(car, progress, 10.0f);
        assert(evaluator.getFitness() == fitnessAtFinish && evaluator.getElapsedTime() == elapsedAtFinish &&
               "TimeLimit must freeze fitness and elapsed time");
        car.reset(kSpawnPosition, kSpawnHeading);
    }

    // 25: standing still for the no-progress timeout ends the evaluation
    // with NoProgress, and fitness/finish reason freeze from that point on.
    {
        car.reset(kSpawnPosition, kSpawnHeading);
        simulation::TrackProgress progress(track);
        progress.reset(car);
        ai::FitnessEvaluator evaluator;
        evaluator.reset();

        for (int i = 0; i < 400 && !evaluator.isEvaluationFinished(); ++i)
        {
            progress.update(car);
            evaluator.update(car, progress, kSimulationDt);
        }
        assert(evaluator.isEvaluationFinished() &&
               evaluator.getFinishReason() == ai::EvaluationFinishReason::NoProgress &&
               "standing still past the no-progress timeout must end the evaluation with NoProgress");

        const ai::EvaluationFinishReason reasonAtFinish = evaluator.getFinishReason();
        const float fitnessAtFinish = evaluator.getFitness(); // 0.0f: no progress was ever made
        evaluator.update(car, progress, 10.0f);
        assert(evaluator.getFitness() == fitnessAtFinish && evaluator.getFinishReason() == reasonAtFinish &&
               "NoProgress must freeze fitness and finish reason");
        car.reset(kSpawnPosition, kSpawnHeading);
    }

    // Meaningful progress resets the no-progress timer. Both hold periods
    // below stay under kNoProgressTimeout (3.0s) and their total stays
    // under kInitialProgressDeadline (5.0s), so this isolates the timer-reset
    // mechanism from the separate initial-progress deadline tested in
    // verifyEarlyTermination() below.
    {
        car.reset(positionAtLapPosition(track, 0.0f), kSpawnHeading);
        simulation::TrackProgress progress(track);
        progress.reset(car);
        ai::FitnessEvaluator evaluator;
        evaluator.reset();

        // Stand still for 2 seconds (under the 3s timeout).
        for (int i = 0; i < 120; ++i)
        {
            progress.update(car);
            evaluator.update(car, progress, kSimulationDt);
        }
        assert(!evaluator.isEvaluationFinished() && "2 seconds of no movement must stay under the timeout");

        // A meaningful forward nudge must reset the no-progress timer.
        car.reset(positionAtLapPosition(track, 0.05f), kSpawnHeading);
        progress.update(car);
        evaluator.update(car, progress, kSimulationDt);

        // A further 2 seconds of standing still (< 3s since the nudge) must
        // still not finish the evaluation, proving the timer reset.
        for (int i = 0; i < 120; ++i)
        {
            progress.update(car);
            evaluator.update(car, progress, kSimulationDt);
        }
        assert(!evaluator.isEvaluationFinished() &&
               "meaningful progress must reset the no-progress timer, not merely delay the original deadline");
        car.reset(kSpawnPosition, kSpawnHeading);
    }

    // 26: reset() clears lap timing state back to the "no completed lap
    // yet" sentinel, even after laps were completed and fitness grew.
    {
        car.reset(positionAtLapPosition(track, 0.0f), kSpawnHeading);
        simulation::TrackProgress progress(track);
        progress.reset(car);
        ai::FitnessEvaluator evaluator;
        evaluator.reset();

        // Drive one full lap via small increments (see driveOneLap's
        // comment earlier in this function).
        {
            constexpr float kStep = 0.15f;
            float remaining = 1.02f; // slight overshoot past the seam -- see driveOneLap's comment earlier
            float p = 0.0f;
            while (remaining > kStep)
            {
                p += kStep;
                remaining -= kStep;
                car.reset(positionAtLapPosition(track, p), kSpawnHeading);
                progress.update(car);
            }
            p += remaining;
            car.reset(positionAtLapPosition(track, p), kSpawnHeading);
            progress.update(car);
        }
        evaluator.update(car, progress, 5.0f);
        assert(evaluator.hasCompletedLap() && evaluator.getFitness() > 0.0f &&
               "setup must have completed a lap with nonzero fitness");

        evaluator.reset();
        assert(!evaluator.hasCompletedLap() && evaluator.getBestLapTime() == 0.0f && evaluator.getLastLapTime() == 0.0f &&
               "reset must clear lap timing state back to the no-completed-lap sentinel");
        assert(evaluator.getFitness() == 0.0f && evaluator.getBaseProgressFitness() == 0.0f &&
               evaluator.getProgressRate() == 0.0f && evaluator.getProgressRateReward() == 0.0f &&
               evaluator.getLapSpeedBonus() == 0.0f && "reset must clear every fitness component"); // 27
        car.reset(kSpawnPosition, kSpawnHeading);
        progress.reset(car);
    }

    // 28 & 29: fitness is deterministic -- two independently constructed
    // evaluators driven through an identical sequence of states produce
    // identical fitness at every intermediate step, not merely at the end.
    {
        auto runScenario = [&](simulation::Car& localCar, std::vector<float>& fitnessTrace)
        {
            localCar.reset(positionAtLapPosition(track, 0.0f), kSpawnHeading);
            simulation::TrackProgress localProgress(track);
            localProgress.reset(localCar);
            ai::FitnessEvaluator localEvaluator;
            localEvaluator.reset();

            const float waypoints[] = {0.05f, 0.12f, 0.20f, 0.30f};
            for (float p : waypoints)
            {
                localCar.reset(positionAtLapPosition(track, p), kSpawnHeading);
                localProgress.update(localCar);
                localEvaluator.update(localCar, localProgress, kSimulationDt);
                fitnessTrace.push_back(localEvaluator.getFitness());
            }
        };

        simulation::Car carA(makeCarParams(), track);
        simulation::Car carB(makeCarParams(), track);
        std::vector<float> traceA;
        std::vector<float> traceB;
        runScenario(carA, traceA);
        runScenario(carB, traceB);

        assert(traceA.size() == traceB.size() && "identical scenarios must produce the same number of steps");
        for (std::size_t i = 0; i < traceA.size(); ++i)
        {
            assert(traceA[i] == traceB[i] && "identical state sequences must produce identical fitness at every step");
        }
    }

    // 30 & 31: FitnessEvaluator only reads Car/TrackProgress -- it never
    // modifies either.
    {
        car.reset(positionAtLapPosition(track, 0.2f), kSpawnHeading);
        simulation::TrackProgress progress(track);
        progress.reset(car);
        ai::FitnessEvaluator evaluator;
        evaluator.reset();

        const Vector2 positionBefore = car.getPosition();
        const Vector2 velocityBefore = car.getVelocity();
        const float headingBefore = car.getHeading();
        const float lapPositionBefore = progress.getLapPosition();
        const float bestProgressBefore = progress.getBestProgress();
        const int checkpointsBefore = progress.getTotalCheckpointsPassed();
        const int lapCountBefore = progress.getLapCount();

        evaluator.update(car, progress, kSimulationDt);

        assert(car.getPosition().x == positionBefore.x && car.getPosition().y == positionBefore.y &&
               car.getVelocity().x == velocityBefore.x && car.getVelocity().y == velocityBefore.y &&
               car.getHeading() == headingBefore && "FitnessEvaluator::update must not modify the Car"); // 30
        assert(progress.getLapPosition() == lapPositionBefore && progress.getBestProgress() == bestProgressBefore &&
               progress.getTotalCheckpointsPassed() == checkpointsBefore && progress.getLapCount() == lapCountBefore &&
               "FitnessEvaluator::update must not modify TrackProgress"); // 31
        car.reset(kSpawnPosition, kSpawnHeading);
    }

    // reset() permits a fresh evaluation after a finished one (the same
    // effect the R key has in main()).
    {
        car.reset(kSpawnPosition, kSpawnHeading);
        simulation::TrackProgress progress(track);
        progress.reset(car);
        ai::FitnessEvaluator evaluator;
        evaluator.reset();

        for (int i = 0; i < 400 && !evaluator.isEvaluationFinished(); ++i)
        {
            progress.update(car);
            evaluator.update(car, progress, kSimulationDt);
        }
        assert(evaluator.isEvaluationFinished() && "setup must have already finished the evaluation");

        car.reset(kSpawnPosition, kSpawnHeading);
        progress.reset(car);
        evaluator.reset();
        assert(!evaluator.isEvaluationFinished() && evaluator.getFitness() == 0.0f &&
               evaluator.getElapsedTime() == 0.0f && "reset must permit a fresh, unfinished evaluation");

        car.reset(positionAtLapPosition(track, 0.05f), kSpawnHeading);
        progress.update(car);
        evaluator.update(car, progress, kSimulationDt);
        assert(evaluator.getFitness() > 0.0f && "the fresh evaluation after reset must respond normally to new progress");
        car.reset(kSpawnPosition, kSpawnHeading);
    }

    TraceLog(LOG_INFO, "Fitness evaluator verification: all deterministic checks passed");
}

// Checks the two early-termination rules layered onto FitnessEvaluator (see
// its "Early termination" class-comment section) -- kNoProgressTimeout/
// kProgressImprovementEpsilon (rule A) and kInitialProgressDeadline/
// kMinimumInitialProgress (rule B). Does not re-test the fitness formula
// itself (covered by verifyFitnessEvaluator() above) -- only the early-exit
// conditions, driven via TrackProgress's positionAtLapPosition rather than
// raw world coordinates or Car velocity.
void verifyEarlyTermination(const simulation::Track& track)
{
    using track_progress_verify::positionAtLapPosition;

    // 1: real forward progress arriving every ~1s (well under
    // kNoProgressTimeout) keeps resetting the no-progress timer indefinitely,
    // proven over 10s straight.
    {
        simulation::Car car(makeCarParams(), track);
        car.reset(positionAtLapPosition(track, 0.0f), kSpawnHeading);
        simulation::TrackProgress progress(track);
        progress.reset(car);
        ai::FitnessEvaluator evaluator;
        evaluator.reset();

        float p = 0.0f;
        for (int second = 0; second < 10; ++second) // 10s total, comfortably under kMaxEvaluationTime (60s)
        {
            p += 0.02f; // >> kProgressImprovementEpsilon (0.001) every step
            car.reset(positionAtLapPosition(track, p), kSpawnHeading);
            progress.update(car);
            evaluator.update(car, progress, 1.0f);
            assert(!evaluator.isEvaluationFinished() &&
                   "a car making meaningful forward progress every second must never hit the no-progress timeout"); // 1
        }
    }

    // 2: driving without improving bestProgress terminates with NoProgress
    // at ~3.0s (kNoProgressTimeout). Starts from a nonzero bestProgress
    // (0.2 laps) to prove this tracks the rate of NEW progress, not just
    // whether any progress ever happened.
    {
        simulation::Car car(makeCarParams(), track);
        car.reset(positionAtLapPosition(track, 0.2f), kSpawnHeading);
        simulation::TrackProgress progress(track);
        progress.reset(car);
        ai::FitnessEvaluator evaluator;
        evaluator.reset();
        evaluator.update(car, progress, kSimulationDt); // registers the starting bestProgress (0.2)

        for (int i = 0; i < 400 && !evaluator.isEvaluationFinished(); ++i)
        {
            progress.update(car); // car does not move -- position held fixed
            evaluator.update(car, progress, kSimulationDt);
        }
        assert(evaluator.isEvaluationFinished() &&
               evaluator.getFinishReason() == ai::EvaluationFinishReason::NoProgress &&
               "holding still without new progress must terminate with NoProgress"); // 2
        assert(std::fabs(evaluator.getElapsedTime() - 3.0f) < 0.1f &&
               "the no-progress timeout must trip at ~3.0 seconds (kNoProgressTimeout), not earlier or much later"); // 2 (continued)
    }

    // 3: only new forward progress beyond the existing high-water mark may
    // reset the timer -- 2s of backward/revisit moves (never exceeding the
    // 0.3-lap high-water mark) must not delay the original 3s mark.
    {
        simulation::Car car(makeCarParams(), track);
        car.reset(positionAtLapPosition(track, 0.3f), kSpawnHeading);
        simulation::TrackProgress progress(track);
        progress.reset(car);
        ai::FitnessEvaluator evaluator;
        evaluator.reset();
        evaluator.update(car, progress, kSimulationDt); // registers the starting bestProgress (0.3)

        for (int i = 0; i < 120 && !evaluator.isEvaluationFinished(); ++i) // 2s of backward/revisited movement
        {
            const float p = (i % 2 == 0) ? 0.1f : 0.25f; // both < 0.3 -- backward, or already-covered ground
            car.reset(positionAtLapPosition(track, p), kSpawnHeading);
            progress.update(car);
            evaluator.update(car, progress, kSimulationDt);
        }
        assert(!evaluator.isEvaluationFinished() &&
               "2 seconds of backward/revisited movement must not have reset the no-progress timer"); // 3

        for (int i = 0; i < 400 && !evaluator.isEvaluationFinished(); ++i) // finish it out from here, no more movement
        {
            progress.update(car);
            evaluator.update(car, progress, kSimulationDt);
        }
        assert(evaluator.isEvaluationFinished() &&
               evaluator.getFinishReason() == ai::EvaluationFinishReason::NoProgress &&
               std::fabs(evaluator.getElapsedTime() - 3.0f) < 0.1f &&
               "the timer must still trip at the ORIGINAL ~3-second mark -- backward/revisited movement must not "
               "have delayed it even slightly"); // 3 (continued)
    }

    // 4: a car that crawls forward just fast enough to keep resetting rule
    // A's timer, but is still effectively stuck near spawn, must terminate
    // with InsufficientInitialProgress at exactly 5.0s (kInitialProgressDeadline).
    {
        simulation::Car car(makeCarParams(), track);
        car.reset(positionAtLapPosition(track, 0.0f), kSpawnHeading);
        simulation::TrackProgress progress(track);
        progress.reset(car);
        ai::FitnessEvaluator evaluator;
        evaluator.reset();

        float p = 0.0f;
        int secondsRun = 0;
        for (; secondsRun < 10 && !evaluator.isEvaluationFinished(); ++secondsRun)
        {
            p += 0.002f; // meaningful (> kProgressImprovementEpsilon) but far too slow to reach 0.04 laps by 5s
            car.reset(positionAtLapPosition(track, p), kSpawnHeading);
            progress.update(car);
            evaluator.update(car, progress, 1.0f);
        }
        assert(evaluator.isEvaluationFinished() &&
               evaluator.getFinishReason() == ai::EvaluationFinishReason::InsufficientInitialProgress &&
               "a car that never reaches the minimum initial progress must terminate with InsufficientInitialProgress"); // 4
        assert(secondsRun == 5 && "the initial-progress deadline must trip at exactly 5 seconds (kInitialProgressDeadline)"); // 4 (continued)
        assert(progress.getBestProgress() < 0.04f &&
               "the terminated car's bestProgress must genuinely be below kMinimumInitialProgress"); // 4 (continued)
    }

    // 5: a normally progressing car clears the 5-second check comfortably
    // and is NOT terminated by either rule.
    {
        simulation::Car car(makeCarParams(), track);
        car.reset(positionAtLapPosition(track, 0.0f), kSpawnHeading);
        simulation::TrackProgress progress(track);
        progress.reset(car);
        ai::FitnessEvaluator evaluator;
        evaluator.reset();

        float p = 0.0f;
        for (int second = 0; second < 6; ++second)
        {
            p += 0.03f; // reaches 0.09 laps by 3s -- comfortably above kMinimumInitialProgress (0.04) before 5s
            car.reset(positionAtLapPosition(track, p), kSpawnHeading);
            progress.update(car);
            evaluator.update(car, progress, 1.0f);
            assert(!evaluator.isEvaluationFinished() &&
                   "a normally progressing car must not be terminated by either early-termination rule"); // 5
        }
        assert(progress.getBestProgress() >= 0.04f && "setup: this scenario must actually clear kMinimumInitialProgress");
    }

    // 6: a car that keeps making steady progress the whole time can still
    // legitimately run all the way to the existing 60-second maximum --
    // neither new rule can cut it short as long as progress keeps coming.
    {
        simulation::Car car(makeCarParams(), track);
        car.reset(positionAtLapPosition(track, 0.0f), kSpawnHeading);
        simulation::TrackProgress progress(track);
        progress.reset(car);
        ai::FitnessEvaluator evaluator;
        evaluator.reset();

        float p = 0.0f;
        for (int second = 0; second < 62 && !evaluator.isEvaluationFinished(); ++second)
        {
            p += 0.01f;
            car.reset(positionAtLapPosition(track, std::fmod(p, 1.0f)), kSpawnHeading);
            progress.update(car);
            evaluator.update(car, progress, 1.0f);
        }
        assert(evaluator.isEvaluationFinished() &&
               evaluator.getFinishReason() == ai::EvaluationFinishReason::TimeLimit &&
               "a car making steady progress the whole time must still be able to reach the 60s TimeLimit, "
               "unaffected by either Stage 21.1 rule"); // 6
    }

    // 7: neither rule reads world coordinates or raw Car speed -- only
    // TrackProgress::getBestProgress(), which always starts at 0.0 right
    // after reset(). Two "hold still" runs at wildly different track
    // positions must therefore terminate identically.
    {
        auto holdStillFinishReason = [&](float lapPos) -> ai::EvaluationFinishReason
        {
            simulation::Car localCar(makeCarParams(), track);
            localCar.reset(positionAtLapPosition(track, lapPos), kSpawnHeading);
            simulation::TrackProgress localProgress(track);
            localProgress.reset(localCar);
            ai::FitnessEvaluator localEvaluator;
            localEvaluator.reset();
            for (int i = 0; i < 400 && !localEvaluator.isEvaluationFinished(); ++i)
            {
                localProgress.update(localCar);
                localEvaluator.update(localCar, localProgress, kSimulationDt);
            }
            return localEvaluator.getFinishReason();
        };

        const ai::EvaluationFinishReason reasonAtStart = holdStillFinishReason(0.0f);
        const ai::EvaluationFinishReason reasonMidTrack = holdStillFinishReason(0.55f);
        assert(reasonAtStart == ai::EvaluationFinishReason::NoProgress && reasonMidTrack == reasonAtStart &&
               "termination reason must depend only on track progress -- never on which world coordinates the car "
               "happens to be sitting at"); // 7
    }

    TraceLog(LOG_INFO, "Early termination verification: all deterministic checks passed");
}
} // namespace verification
