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

using app::assetPath;
using app::kSimulationDt;
using app::kSpawnHeading;
using app::kSpawnPosition;
using app::makeCarParams;

namespace track_verify
{

template <typename Callable>
bool throwsInvalidArgument(Callable&& callable)
{
    try
    {
        callable();
    }
    catch (const std::invalid_argument&)
    {
        return true;
    }
    return false;
}

// 2D cross product of (b - a) and (c - a); sign gives the orientation of
// the triplet a, b, c.
float cross2D(Vector2 a, Vector2 b, Vector2 c)
{
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

// True if segments (a1,a2) and (b1,b2) properly intersect or overlap, via
// the standard orientation + bounding-box test. Only used by verification
// (self-intersection check on the sampled centerline), never by any
// production/runtime code path.
bool segmentsIntersect(Vector2 a1, Vector2 a2, Vector2 b1, Vector2 b2)
{
    const float d1 = cross2D(b1, b2, a1);
    const float d2 = cross2D(b1, b2, a2);
    const float d3 = cross2D(a1, a2, b1);
    const float d4 = cross2D(a1, a2, b2);

    if (((d1 > 0.0f && d2 < 0.0f) || (d1 < 0.0f && d2 > 0.0f)) &&
        ((d3 > 0.0f && d4 < 0.0f) || (d3 < 0.0f && d4 > 0.0f)))
    {
        return true;
    }

    // Degenerate (collinear/touching) cases: treat any coincident bounding
    // box overlap with a near-zero cross product as an intersection too, so
    // a track segment that merely grazes another is still caught.
    constexpr float kEps = 1e-4f;
    auto onSegment = [](Vector2 p, Vector2 q, Vector2 r)
    {
        return std::min(p.x, r.x) - kEps <= q.x && q.x <= std::max(p.x, r.x) + kEps &&
               std::min(p.y, r.y) - kEps <= q.y && q.y <= std::max(p.y, r.y) + kEps;
    };
    if (std::fabs(d1) < kEps && onSegment(b1, a1, b2))
        return true;
    if (std::fabs(d2) < kEps && onSegment(b1, a2, b2))
        return true;
    if (std::fabs(d3) < kEps && onSegment(a1, b1, a2))
        return true;
    if (std::fabs(d4) < kEps && onSegment(a1, b2, a2))
        return true;

    return false;
}

// Shortest distance from point p to segment (a,b).
float pointToSegmentDistance(Vector2 p, Vector2 a, Vector2 b)
{
    const Vector2 ab = {b.x - a.x, b.y - a.y};
    const float lengthSq = ab.x * ab.x + ab.y * ab.y;
    float t = 0.0f;
    if (lengthSq > 0.0f)
    {
        t = std::clamp(((p.x - a.x) * ab.x + (p.y - a.y) * ab.y) / lengthSq, 0.0f, 1.0f);
    }
    const Vector2 closest = {a.x + ab.x * t, a.y + ab.y * t};
    return std::sqrt((p.x - closest.x) * (p.x - closest.x) + (p.y - closest.y) * (p.y - closest.y));
}

// Shortest distance between segments (a1,a2) and (b1,b2): the minimum of
// each endpoint's distance to the other segment. Exact except for the rare
// case of parallel, overlapping segments, which is more than sufficient for
// a conservative separation check between track sections.
float segmentToSegmentDistance(Vector2 a1, Vector2 a2, Vector2 b1, Vector2 b2)
{
    return std::min({pointToSegmentDistance(a1, b1, b2), pointToSegmentDistance(a2, b1, b2),
                      pointToSegmentDistance(b1, a1, a2), pointToSegmentDistance(b2, a1, a2)});
}

} // namespace track_verify

// One-shot, deterministic sanity check of the generalized centerline Track:
// TrackDefinition validation, Catmull-Rom centerline sampling, arc-length
// data, nearest-centerline projection, and the CPU drivable mask derived
// from all of the above. Runs once at startup, never inside the render
// loop. `track` is the real, already-constructed easy oval Track used by
// the rest of the program; additional throwaway Track instances are built
// here only to exercise validation and determinism.
void verifyTrack(const simulation::Track& track)
{
    using track_verify::throwsInvalidArgument;
    using track_verify::segmentsIntersect;
    using track_verify::segmentToSegmentDistance;

    const simulation::TrackDefinition& def = track.getDefinition();

    // 1-5: TrackDefinition validation.
    {
        simulation::TrackDefinition badWidth = def;
        badWidth.simWidth = 0;
        assert(throwsInvalidArgument([&]() { simulation::Track t(badWidth); }) &&
               "non-positive simWidth must be rejected");

        simulation::TrackDefinition badHeight = def;
        badHeight.simHeight = -10;
        assert(throwsInvalidArgument([&]() { simulation::Track t(badHeight); }) &&
               "non-positive simHeight must be rejected");

        simulation::TrackDefinition tooFewPoints = def;
        tooFewPoints.controlPoints.resize(3);
        assert(throwsInvalidArgument([&]() { simulation::Track t(tooFewPoints); }) &&
               "fewer than 4 control points must be rejected");

        simulation::TrackDefinition zeroWidth = def;
        zeroWidth.trackWidth = 0.0f;
        assert(throwsInvalidArgument([&]() { simulation::Track t(zeroWidth); }) &&
               "a non-positive track width must be rejected");

        simulation::TrackDefinition nanWidth = def;
        nanWidth.trackWidth = std::numeric_limits<float>::quiet_NaN();
        assert(throwsInvalidArgument([&]() { simulation::Track t(nanWidth); }) &&
               "a non-finite track width must be rejected");

        simulation::TrackDefinition infWidth = def;
        infWidth.trackWidth = std::numeric_limits<float>::infinity();
        assert(throwsInvalidArgument([&]() { simulation::Track t(infWidth); }) &&
               "an infinite track width must be rejected");

        simulation::TrackDefinition badSamples = def;
        badSamples.samplesPerSegment = 1;
        assert(throwsInvalidArgument([&]() { simulation::Track t(badSamples); }) &&
               "samplesPerSegment below 2 must be rejected");

        simulation::TrackDefinition nonFinitePoint = def;
        nonFinitePoint.controlPoints[0].x = std::numeric_limits<float>::quiet_NaN();
        assert(throwsInvalidArgument([&]() { simulation::Track t(nonFinitePoint); }) &&
               "a non-finite control point coordinate must be rejected");
    }

    // 6 & 43: centerline sampling (and therefore the whole derived Track)
    // is a deterministic function of TrackDefinition -- two independently
    // constructed Tracks from the exact same definition produce identical
    // centerline samples, cumulative distances, and spawn pose.
    {
        simulation::Track other(def);
        const std::vector<Vector2>& samplesA = track.getCenterlineSamples();
        const std::vector<Vector2>& samplesB = other.getCenterlineSamples();
        assert(samplesA.size() == samplesB.size() && "centerline sampling must be deterministic (sample count)");
        for (std::size_t i = 0; i < samplesA.size(); ++i)
        {
            assert(samplesA[i].x == samplesB[i].x && samplesA[i].y == samplesB[i].y &&
                   "centerline sampling must be deterministic (sample positions)");
        }
        assert(track.getTotalLength() == other.getTotalLength() &&
               "a fixed TrackDefinition must produce identical total length across runs");
        assert(track.getSpawnPosition().x == other.getSpawnPosition().x &&
               track.getSpawnPosition().y == other.getSpawnPosition().y &&
               track.getSpawnHeading() == other.getSpawnHeading() &&
               "a fixed TrackDefinition must produce an identical spawn pose across runs");
    }

    const std::vector<Vector2>& centerline = track.getCenterlineSamples();
    const std::vector<float>& cumulative = track.getCumulativeDistances();

    // 7: sampled centerline is non-empty.
    assert(!centerline.empty() && "sampled centerline must be non-empty");
    assert(centerline.size() == cumulative.size() && "cumulative distances must have one entry per sample");

    // 8 & 9: the loop is closed (wraparound segment is real, non-degenerate)
    // and every consecutive pair of samples is distinct.
    {
        const std::size_t n = centerline.size();
        float sumOfLengths = 0.0f;
        for (std::size_t i = 0; i < n; ++i)
        {
            const Vector2& a = centerline[i];
            const Vector2& b = centerline[(i + 1) % n];
            const float dx = b.x - a.x;
            const float dy = b.y - a.y;
            const float length = std::sqrt(dx * dx + dy * dy);
            assert(length > 0.0f && "no two consecutive centerline samples (including the closing seam) may coincide");
            sumOfLengths += length;
        }
        // 12: total length equals the independently recomputed sum of every
        // segment's length (including the closing one) within tolerance.
        assert(std::fabs(sumOfLengths - track.getTotalLength()) < 0.5f &&
               "total length must equal the sum of all segment lengths");
    }

    // 10: total length is positive.
    assert(track.getTotalLength() > 0.0f && "total track length must be positive");

    // 11: cumulative distances increase monotonically.
    for (std::size_t i = 1; i < cumulative.size(); ++i)
    {
        assert(cumulative[i] > cumulative[i - 1] && "cumulative distances must increase monotonically");
    }
    assert(cumulative.back() < track.getTotalLength() && "the last sample's cumulative distance must be below the total length");

    // 13, 14, 15, 17 & 18: projecting a centerline sample back onto the
    // centerline returns that same point at near-zero distance, with a
    // deterministic tie-break to the lower of the two segments meeting there.
    {
        const std::size_t sampleIndex = 5; // arbitrary interior sample, away from the wraparound seam
        const simulation::TrackProjection proj = track.projectOntoCenterline(centerline[sampleIndex]);

        assert(proj.distanceFromCenterline < 1e-3f &&
               "projecting a centerline sample onto the centerline must return near-zero distance");
        assert(proj.segmentT >= 0.0f && proj.segmentT <= 1.0f && "segmentT must stay within [0,1]");
        assert(proj.distanceAlongTrack >= 0.0f && proj.distanceAlongTrack < track.getTotalLength() &&
               "distanceAlongTrack must stay within [0, totalLength)");
        assert(proj.segmentIndex == sampleIndex - 1 && std::fabs(proj.segmentT - 1.0f) < 1e-3f &&
               "tie-break between two equally-close segments must deterministically choose the lower segment index");

        // Must be collinear with the segment's endpoints and within its span.
        const Vector2& a = centerline[proj.segmentIndex];
        const Vector2& b = centerline[(proj.segmentIndex + 1) % centerline.size()];
        const float cross = (b.x - a.x) * (proj.point.y - a.y) - (b.y - a.y) * (proj.point.x - a.x);
        assert(std::fabs(cross) < 1.0f && "the projected point must lie on its reported segment");
    }

    // 16: distanceFromCenterline is correct for a known simple geometry --
    // a point offset perpendicular to the spawn tangent by a known distance
    // must report that same distance.
    {
        const Vector2 spawn = track.getSpawnPosition();
        const float heading = track.getSpawnHeading();
        const Vector2 perpendicular = {-std::sin(heading), std::cos(heading)};
        constexpr float kOffset = 10.0f;
        const Vector2 offsetPoint = {spawn.x + perpendicular.x * kOffset, spawn.y + perpendicular.y * kOffset};

        const simulation::TrackProjection proj = track.projectOntoCenterline(offsetPoint);
        assert(std::fabs(proj.distanceFromCenterline - kOffset) < 1.0f &&
               "distanceFromCenterline must match a known perpendicular offset from the centerline");
    }

    // 19 & 21: the CPU mask agrees with the centerline at spawn, and
    // rejects out-of-bounds coordinates. (No generic "center must be
    // non-drivable" check: not true of every track shape -- see
    // verifyImageBasedTrackSystem() for a track-specific version.)
    {
        const Vector2 spawn = track.getSpawnPosition();
        assert(track.isDrivable(static_cast<int>(spawn.x), static_cast<int>(spawn.y)) &&
               "a point on the centerline must be drivable");

        assert(!track.isDrivable(-5, -5) && "negative coordinates must be non-drivable");
        assert(!track.isDrivable(def.simWidth, def.simHeight / 2) && "x at/beyond width must be non-drivable");
        assert(!track.isDrivable(def.simWidth / 2, def.simHeight) && "y at/beyond height must be non-drivable");
    }

    // 22: rendering and collision both derive from the exact same sampled
    // centerline/width -- every centerline sample itself must fall inside
    // the drivable mask built from that same centerline.
    {
        for (const Vector2& sample : centerline)
        {
            const int x = static_cast<int>(std::lround(sample.x));
            const int y = static_cast<int>(std::lround(sample.y));
            assert(track.isDrivable(x, y) && "every centerline sample must be drivable in the mask built from it");
        }
    }

    // 23 & 24: the spawn pose is on the drivable mask and its heading
    // follows the forward centerline tangent (sample 0 -> sample 1).
    {
        const Vector2 spawn = track.getSpawnPosition();
        assert(track.isDrivable(static_cast<int>(spawn.x), static_cast<int>(spawn.y)) &&
               "the spawn position must be drivable");

        const Vector2& next = centerline[1 % centerline.size()];
        const float expectedHeading = std::atan2(next.y - spawn.y, next.x - spawn.x);
        assert(std::fabs(track.getSpawnHeading() - expectedHeading) < 1e-4f &&
               "spawn heading must follow the forward centerline tangent");
    }

    // An asymmetric closed loop can fold two non-adjacent sections into a
    // crossing (unlike a convex oval). O(sampleCount^2) geometry check over
    // the real sampled centerline -- verification only.
    {
        const std::size_t n = centerline.size();

        // Segments within this many indices (either direction, wrapping)
        // are "adjacent" and excluded -- ordinary corner curvature can
        // bring same-turn entry/exit samples this close without it being a
        // real overlap; wide enough to cover a full corner on this track.
        constexpr std::size_t kAdjacencyWindow = 100;

        float worstSeparation = std::numeric_limits<float>::max();
        bool anyIntersection = false;

        for (std::size_t i = 0; i < n; ++i)
        {
            const Vector2& a1 = centerline[i];
            const Vector2& a2 = centerline[(i + 1) % n];

            for (std::size_t j = i + 1; j < n; ++j)
            {
                const std::size_t forwardGap = j - i;
                const std::size_t backwardGap = n - forwardGap;
                if (std::min(forwardGap, backwardGap) <= kAdjacencyWindow)
                {
                    continue; // adjacent segment -- not a candidate for overlap
                }

                const Vector2& b1 = centerline[j];
                const Vector2& b2 = centerline[(j + 1) % n];

                if (segmentsIntersect(a1, a2, b1, b2))
                {
                    anyIntersection = true;
                }

                worstSeparation = std::min(worstSeparation, segmentToSegmentDistance(a1, a2, b1, b2));
            }
        }

        // No two non-adjacent centerline segments may cross -- a literal
        // crossing would break arc-length monotonicity along the loop.
        assert(!anyIntersection && "the closed centerline must not self-intersect between non-adjacent sections");

        // No minimum-separation assertion: the extreme track deliberately
        // brings two non-adjacent corridors within ~94px (by design -- see
        // TrackProgress's local tracking/recovery). Logged as a diagnostic only.
        TraceLog(LOG_INFO, "Track verification: closest non-adjacent centerline separation is %.1fpx",
                 static_cast<double>(worstSeparation));
    }

    TraceLog(LOG_INFO, "Track verification: all centerline/arc-length/projection/mask checks passed");
}

namespace track_progress_verify
{

// Inverse of TrackProgress's arc-length-to-progress mapping (see
// TrackProgress.h for the forward mapping this undoes): returns the world
// position on the track's own sampled centerline at the given normalized
// lap position, via Track::getPointAtDistance(). Lets tests place the car
// at exact, hand-computed lap positions via Car::reset() instead of relying
// on real driving physics for anything but the one "real driving" sanity
// check below.
Vector2 positionAtLapPosition(const simulation::Track& track, float lapPos)
{
    return track.getPointAtDistance(lapPos * track.getTotalLength());
}

} // namespace track_progress_verify

// Deterministic check of simulation::TrackProgress. Mostly drives Car::reset()
// to precise hand-computed positions rather than real driving physics,
// plus one short real-driving check for direction sanity.
void verifyTrackProgress(const simulation::Track& track)
{
    using track_progress_verify::positionAtLapPosition;
    constexpr float kEps = 1e-3f;

    simulation::Car car(makeCarParams(), track);

    // 1 & 13 (setup half): reset() computes lap position from the car's
    // current position via Track::projectOntoCenterline(), and clears every
    // accumulated field to its baseline.
    {
        car.reset(kSpawnPosition, kSpawnHeading);
        simulation::TrackProgress progress(track);
        progress.reset(car);

        const simulation::TrackProjection expectedProjection = track.projectOntoCenterline(kSpawnPosition);
        const float expectedLapPosition = expectedProjection.distanceAlongTrack / track.getTotalLength();

        assert(std::fabs(progress.getLapPosition() - expectedLapPosition) < kEps &&
               "reset must compute lap position from the car's current spawn position");

        // Expected checkpoint is the first strictly ahead of the car, not
        // unconditionally 0 -- the car may spawn anywhere on the ring.
        const int expectedCheckpointIndex =
            (static_cast<int>(std::floor(expectedLapPosition * simulation::TrackProgress::kCheckpointCount)) + 1) %
            simulation::TrackProgress::kCheckpointCount;

        assert(progress.getContinuousProgress() == 0.0f && progress.getBestProgress() == 0.0f &&
               progress.getLapCount() == 0 && progress.getExpectedCheckpoint() == expectedCheckpointIndex &&
               progress.getTotalCheckpointsPassed() == 0 && "reset must clear all accumulated state");
    }

    // 2: normalized lap position stays within [0,1) at several distinct
    // positions around the oval.
    {
        simulation::TrackProgress progress(track);
        car.reset(positionAtLapPosition(track, 0.0f), kSpawnHeading);
        progress.reset(car);

        for (float p = 0.0f; p < 1.0f; p += 0.1f)
        {
            car.reset(positionAtLapPosition(track, p), kSpawnHeading);
            progress.update(car);
            assert(progress.getLapPosition() >= 0.0f && progress.getLapPosition() < 1.0f &&
                   "lap position must always stay within [0,1)");
        }
        car.reset(kSpawnPosition, kSpawnHeading);
    }

    // 3: progress increases forward -- driven with real Car physics
    // (throttle only) from spawn, not synthetic positions.
    {
        simulation::TrackProgress progress(track);
        car.reset(kSpawnPosition, kSpawnHeading);
        progress.reset(car);

        simulation::CarInput driveForward;
        driveForward.throttle = 1.0f;
        driveForward.steering = 0.0f;
        for (int i = 0; i < 30 && car.isAlive(); ++i)
        {
            car.update(driveForward, kSimulationDt);
            progress.update(car);
        }
        assert(car.isAlive() && "the car must still be on the spawn straight after 0.5s from spawn");
        assert(progress.getContinuousProgress() > 0.0f &&
               "driving forward from spawn must increase continuous progress");
        assert(progress.getBestProgress() == progress.getContinuousProgress() &&
               "purely forward driving must keep best progress equal to continuous progress");
        car.reset(kSpawnPosition, kSpawnHeading);
    }

    // 4 & 12: backward movement decreases continuous progress but never
    // reduces best progress.
    {
        simulation::TrackProgress progress(track);
        car.reset(positionAtLapPosition(track, 0.0f), kSpawnHeading);
        progress.reset(car);

        car.reset(positionAtLapPosition(track, 0.10f), kSpawnHeading);
        progress.update(car);
        const float bestAfterForward = progress.getBestProgress();
        assert(bestAfterForward > 0.09f && "forward synthetic movement must register as progress");

        car.reset(positionAtLapPosition(track, 0.07f), kSpawnHeading);
        progress.update(car);
        assert(progress.getContinuousProgress() < bestAfterForward - kEps &&
               "backward movement must decrease continuous progress");
        assert(progress.getBestProgress() == bestAfterForward && "backward movement must not change best progress");
        car.reset(kSpawnPosition, kSpawnHeading);
    }

    // 5 & 6: a forward seam (0/1) crossing increments continuous progress
    // and lap count correctly; a subsequent backward seam crossing does not
    // award an extra completed lap and cannot raise best progress.
    {
        simulation::TrackProgress progress(track);
        car.reset(positionAtLapPosition(track, 0.0f), kSpawnHeading);
        progress.reset(car);

        const float toSeam[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 0.98f};
        for (float p : toSeam)
        {
            car.reset(positionAtLapPosition(track, p), kSpawnHeading);
            progress.update(car);
        }
        assert(progress.getLapCount() == 0 && "lap must not be counted before crossing the seam");

        // Forward seam crossing: 0.98 -> 0.02, i.e. delta corrects to +0.04.
        car.reset(positionAtLapPosition(track, 0.02f), kSpawnHeading);
        progress.update(car);
        assert(progress.getContinuousProgress() > 1.0f &&
               "a forward seam crossing must push continuous progress past 1.0");
        assert(progress.getLapCount() == 1 && "a valid forward seam crossing must complete lap 1");
        const float bestAfterLap = progress.getBestProgress();

        // Backward seam crossing back across 0.02 -> 0.98 must not grant an
        // additional lap, and must not exceed the existing best.
        car.reset(positionAtLapPosition(track, 0.98f), kSpawnHeading);
        progress.update(car);
        assert(progress.getLapCount() <= 1 && "a backward seam crossing must never award a completed forward lap");
        assert(progress.getBestProgress() == bestAfterLap && "a backward seam crossing must not raise best progress");
        car.reset(kSpawnPosition, kSpawnHeading);
    }

    // 7 & 8: checkpoints are ordered-traversal state from raw position
    // deltas (never getBestProgress()) -- advance strictly in order,
    // several can be credited per update, and a lap needs the full
    // kCheckpointCount. Reset off a checkpoint boundary (0.03) so
    // completing a lap needs the full count, not one fewer.
    {
        simulation::TrackProgress progress(track);
        car.reset(positionAtLapPosition(track, 0.03f), kSpawnHeading);
        progress.reset(car);
        assert(progress.getExpectedCheckpoint() == 1 && progress.getTotalCheckpointsPassed() == 0 &&
               "reset at lap position 0.03 must expect checkpoint 1 next");

        // Six forward steps of ~0.15 laps each, each spanning one or more
        // checkpoint boundaries -- expected (total passed, next expected)
        // worked out from the fixed checkpoint positions i/16.
        struct Step
        {
            float targetLapPosition;
            int expectedTotalPassed;
            int expectedNextCheckpoint;
        };
        const Step steps[] = {
            {0.18f, 2, 3},   // awards checkpoints 1, 2
            {0.33f, 5, 6},   // awards checkpoints 3, 4, 5 -- several in one update (review req. 3)
            {0.48f, 7, 8},   // awards checkpoints 6, 7
            {0.63f, 10, 11}, // awards checkpoints 8, 9, 10
            {0.78f, 12, 13}, // awards checkpoints 11, 12
            {0.93f, 14, 15}, // awards checkpoints 13, 14
        };
        for (const Step& step : steps)
        {
            car.reset(positionAtLapPosition(track, step.targetLapPosition), kSpawnHeading);
            progress.update(car);
            assert(progress.getTotalCheckpointsPassed() == step.expectedTotalPassed &&
                   progress.getExpectedCheckpoint() == step.expectedNextCheckpoint &&
                   "checkpoints must advance strictly in order by exactly the boundaries actually crossed");
        }
        assert(progress.getLapCount() == 0 && "14 of 16 checkpoints passed must not yet complete a lap");

        // Forward seam crossing 0.93 -> 0.08 awards the final two
        // checkpoints and completes the lap in the same update -- seam
        // crossing alone is not what completes it.
        car.reset(positionAtLapPosition(track, 0.08f), kSpawnHeading);
        progress.update(car);
        assert(progress.getTotalCheckpointsPassed() == 17 && progress.getExpectedCheckpoint() == 2 &&
               progress.getLapCount() == 1 &&
               "a full ordered traversal of all checkpoints must complete the lap exactly once, at the seam crossing that finishes it");

        // Jumping to a later angular position without crossing the
        // intervening checkpoints in order (an implausible, teleport-sized
        // delta) must not award anything, even though the raw angle moves
        // forward. Jump target 0.65 is chosen (verified offline against this
        // track's geometry) so the local search window from 0.08 clearly
        // can't reach it (~442px, past kLocalProjectionRecoveryDistance's
        // 250px) and must fall back to global recovery.
        const int totalBeforeJump = progress.getTotalCheckpointsPassed();
        const int expectedBeforeJump = progress.getExpectedCheckpoint();
        const int lapsBeforeJump = progress.getLapCount();
        car.reset(positionAtLapPosition(track, 0.65f), kSpawnHeading); // 0.08 -> 0.65 is a 0.57 jump (seam-corrected to -0.43), > the plausibility threshold
        progress.update(car);
        assert(std::fabs(progress.getLapPosition() - 0.65f) < kEps &&
               "the raw lap position must still reflect the car's actual (teleported) position");
        assert(progress.getTotalCheckpointsPassed() == totalBeforeJump && progress.getExpectedCheckpoint() == expectedBeforeJump &&
               progress.getLapCount() == lapsBeforeJump &&
               "an implausible jump must not award checkpoints or laps even if it lands at a later angle");

        // Review requirement 4: backward movement (still within the
        // plausible-delta range) must not award checkpoints either.
        car.reset(positionAtLapPosition(track, 0.60f), kSpawnHeading); // 0.65 -> 0.60 is backward
        progress.update(car);
        assert(progress.getTotalCheckpointsPassed() == totalBeforeJump && progress.getExpectedCheckpoint() == expectedBeforeJump &&
               progress.getLapCount() == lapsBeforeJump && "backward movement must not award checkpoints or laps");

        car.reset(kSpawnPosition, kSpawnHeading);
    }

    // reset() restores expected checkpoint/counts correctly for whatever
    // position it's given, not just a fixed baseline.
    {
        simulation::TrackProgress progress(track);
        car.reset(positionAtLapPosition(track, 0.55f), kSpawnHeading);
        progress.reset(car);
        assert(progress.getExpectedCheckpoint() == 9 && progress.getTotalCheckpointsPassed() == 0 && progress.getLapCount() == 0 &&
               "reset at lap position 0.55 must expect checkpoint 9 next, with checkpoint/lap counts at zero");
        car.reset(kSpawnPosition, kSpawnHeading);
    }

    // 11: repeated updates at an unchanged position do not change progress.
    {
        simulation::TrackProgress progress(track);
        const Vector2 pos = positionAtLapPosition(track, 0.3f);
        car.reset(pos, kSpawnHeading);
        progress.reset(car);
        car.reset(pos, kSpawnHeading);
        progress.update(car);

        const float lapPos1 = progress.getLapPosition();
        const float cont1 = progress.getContinuousProgress();
        const float best1 = progress.getBestProgress();
        const int laps1 = progress.getLapCount();
        const int checkpoints1 = progress.getTotalCheckpointsPassed();

        for (int i = 0; i < 5; ++i)
        {
            car.reset(pos, kSpawnHeading);
            progress.update(car);
        }
        assert(progress.getLapPosition() == lapPos1 && progress.getContinuousProgress() == cont1 &&
               progress.getBestProgress() == best1 && progress.getLapCount() == laps1 &&
               progress.getTotalCheckpointsPassed() == checkpoints1 &&
               "repeated updates at an unchanged position must not change progress");
        car.reset(kSpawnPosition, kSpawnHeading);
    }

    // 12 (extended): best progress never decreases across a longer mixed
    // forward/backward sequence.
    {
        simulation::TrackProgress progress(track);
        car.reset(positionAtLapPosition(track, 0.0f), kSpawnHeading);
        progress.reset(car);

        float lastBest = progress.getBestProgress();
        const float waypoints[] = {0.05f, 0.15f, 0.08f, 0.20f, 0.10f, 0.25f, 0.15f, 0.30f};
        for (float p : waypoints)
        {
            car.reset(positionAtLapPosition(track, p), kSpawnHeading);
            progress.update(car);
            assert(progress.getBestProgress() >= lastBest - kEps && "best progress must never decrease");
            lastBest = std::max(lastBest, progress.getBestProgress());
        }
        car.reset(kSpawnPosition, kSpawnHeading);
    }

    // 13: reset clears accumulated state built up from nonzero progress.
    {
        simulation::TrackProgress progress(track);
        car.reset(positionAtLapPosition(track, 0.0f), kSpawnHeading);
        progress.reset(car);
        car.reset(positionAtLapPosition(track, 0.19f), kSpawnHeading);
        progress.update(car);
        assert(progress.getBestProgress() > 0.0f && progress.getTotalCheckpointsPassed() > 0 &&
               "setup for the reset-clears test must have accumulated nonzero state");

        car.reset(kSpawnPosition, kSpawnHeading);
        progress.reset(car);
        const int expectedCheckpointAtSpawn =
            (static_cast<int>(std::floor(progress.getLapPosition() * simulation::TrackProgress::kCheckpointCount)) + 1) %
            simulation::TrackProgress::kCheckpointCount;
        assert(progress.getContinuousProgress() == 0.0f && progress.getBestProgress() == 0.0f &&
               progress.getLapCount() == 0 && progress.getExpectedCheckpoint() == expectedCheckpointAtSpawn &&
               progress.getTotalCheckpointsPassed() == 0 && "reset must clear all previously accumulated state");
    }

    // 14: TrackProgress derives lap position from its own injected Track --
    // a differently shaped Track yields a different lap position for the same world point.
    {
        simulation::TrackDefinition otherDef = track.getDefinition();
        for (Vector2& point : otherDef.controlPoints)
        {
            point.x += 50.0f;
            point.y -= 30.0f;
        }
        simulation::Track otherTrack(otherDef);

        simulation::TrackProgress progressA(track);
        simulation::TrackProgress progressB(otherTrack);

        car.reset(kSpawnPosition, kSpawnHeading);
        progressA.reset(car);
        progressB.reset(car);

        assert(progressA.getLapPosition() != progressB.getLapPosition() &&
               "TrackProgress must derive lap position from its own injected Track, not a hardcoded shape");
    }

    // 15: invalid TrackDefinitions are rejected by Track's own constructor
    // (see verifyTrack) -- TrackProgress performs no redundant validation.

    TraceLog(LOG_INFO, "Track progress verification: all deterministic checks passed");
}

namespace image_track_verify
{

// A tiny (8x8) closed-loop centerline entirely inside fixtures/mask_a.png
// and mask_b.png's shared white region (each fixture's black square sits
// in an opposite corner). samplesPerSegment=2 keeps all 8 samples clear of both corners.
std::vector<Vector2> tinyLoopControlPoints()
{
    return {Vector2{3.0f, 3.0f}, Vector2{5.0f, 3.0f}, Vector2{5.0f, 5.0f}, Vector2{3.0f, 5.0f}};
}

// First control point (= spawn position, at the default
// spawnDistanceAlongTrack) sits at (1,1), inside mask_a.png's black corner.
std::vector<Vector2> spawnInNonDrivableCornerControlPoints()
{
    return {Vector2{1.0f, 1.0f}, Vector2{5.0f, 1.0f}, Vector2{5.0f, 5.0f}, Vector2{1.0f, 5.0f}};
}

simulation::TrackDefinition makeTinyFixtureDefinition(const std::string& visualImagePath,
                                                        const std::string& maskImagePath)
{
    simulation::TrackDefinition def;
    def.simWidth = 8;
    def.simHeight = 8;
    def.controlPoints = tinyLoopControlPoints();
    def.trackWidth = 1.0f; // unused for collision (maskImagePath is set); validate() only requires > 0
    def.samplesPerSegment = 2;
    def.visualImagePath = visualImagePath;
    def.maskImagePath = maskImagePath;
    return def;
}

// Steps outward from centerline sample sampleIndex, perpendicular, both
// directions, until leaving the mask -- sum of both distances is the
// drivable band's width there, measured purely via Track::isDrivable().
float measureRoadWidthAt(const simulation::Track& track, std::size_t sampleIndex)
{
    const std::vector<Vector2>& centerline = track.getCenterlineSamples();
    const std::size_t sampleCount = centerline.size();
    const Vector2& a = centerline[sampleIndex];
    const Vector2& b = centerline[(sampleIndex + 1) % sampleCount];

    const float dx = b.x - a.x;
    const float dy = b.y - a.y;
    const float len = std::sqrt(dx * dx + dy * dy);
    const Vector2 perp = {-dy / len, dx / len};

    constexpr float kMaxScan = 300.0f; // px, comfortably beyond any active track's widest band
    constexpr float kStep = 1.0f;

    float positiveExtent = 0.0f;
    for (float d = 0.0f; d <= kMaxScan; d += kStep)
    {
        const int x = static_cast<int>(std::lround(a.x + perp.x * d));
        const int y = static_cast<int>(std::lround(a.y + perp.y * d));
        if (!track.isDrivable(x, y))
        {
            break;
        }
        positiveExtent = d;
    }

    float negativeExtent = 0.0f;
    for (float d = 0.0f; d <= kMaxScan; d += kStep)
    {
        const int x = static_cast<int>(std::lround(a.x - perp.x * d));
        const int y = static_cast<int>(std::lround(a.y - perp.y * d));
        if (!track.isDrivable(x, y))
        {
            break;
        }
        negativeExtent = d;
    }

    return positiveExtent + negativeExtent;
}

} // namespace image_track_verify

// Image-based track system: asset loading/validation, mask thresholding,
// the three layers' mutual independence, and TrackProgress's local-
// tracking/global-recovery behavior. Geometry-probing checks are written
// generically against whatever `track` (the real active track) actually
// is; asset validation/layer independence use tiny throwaway Tracks from
// assets/tracks/test/fixtures/.
void verifyImageBasedTrackSystem(const simulation::Track& track)
{
    using track_verify::throwsInvalidArgument;
    using image_track_verify::makeTinyFixtureDefinition;
    using image_track_verify::measureRoadWidthAt;

    const std::string maskA = assetPath("tracks/test/fixtures/mask_a.png");
    const std::string maskB = assetPath("tracks/test/fixtures/mask_b.png");
    const std::string visualA = assetPath("tracks/test/fixtures/visual_a.png");
    const std::string visualB = assetPath("tracks/test/fixtures/visual_b.png");
    const std::string wrongSize = assetPath("tracks/test/fixtures/wrong_size.png");
    const std::string missingFile = assetPath("tracks/test/fixtures/does_not_exist.png");

    // 1 & 2: a visual/mask asset path that does not exist on disk is
    // rejected clearly, not silently ignored or defaulted.
    {
        simulation::TrackDefinition missingVisual = makeTinyFixtureDefinition(missingFile, maskA);
        assert(throwsInvalidArgument([&]() { simulation::Track t(missingVisual); }) &&
               "a missing visual image asset must be rejected");

        simulation::TrackDefinition missingMask = makeTinyFixtureDefinition(visualA, missingFile);
        assert(throwsInvalidArgument([&]() { simulation::Track t(missingMask); }) &&
               "a missing mask image asset must be rejected");
    }

    // 3 & 4: an asset that loads but whose dimensions do not match
    // (simWidth, simHeight) is rejected clearly, not silently stretched.
    {
        simulation::TrackDefinition badVisualDims = makeTinyFixtureDefinition(wrongSize, maskA);
        assert(throwsInvalidArgument([&]() { simulation::Track t(badVisualDims); }) &&
               "a visual image with mismatched dimensions must be rejected");

        simulation::TrackDefinition badMaskDims = makeTinyFixtureDefinition(visualA, wrongSize);
        assert(throwsInvalidArgument([&]() { simulation::Track t(badMaskDims); }) &&
               "a mask image with mismatched dimensions must be rejected");
    }

    // 5 & 6: mask pixel threshold produces correct drivable/non-drivable
    // results end to end (mask_a.png is white everywhere except a black
    // 2x2 corner); also confirms the CPU mask is sized to this definition's
    // 8x8, not the main track's 1200x700.
    {
        simulation::TrackDefinition def = makeTinyFixtureDefinition(visualA, maskA);
        simulation::Track fixtureTrack(def);

        assert(!fixtureTrack.isDrivable(1, 1) && "a black mask pixel must be non-drivable");
        assert(fixtureTrack.isDrivable(4, 4) && "a white mask pixel must be drivable");
        assert(!fixtureTrack.isDrivable(8, 8) && "coordinates at/beyond the fixture's own 8x8 bounds must be non-drivable");
    }

    // 7: isDrivable() is a direct m_mask[] index -- O(1), guaranteed by the
    // implementation itself.

    // 11 & 29: changing only the mask image (same centerline/visual)
    // changes collision, proving collision is read from the mask, not the centerline.
    {
        simulation::Track withMaskA(makeTinyFixtureDefinition(visualA, maskA));
        simulation::Track withMaskB(makeTinyFixtureDefinition(visualA, maskB));

        assert(withMaskA.isDrivable(1, 1) != withMaskB.isDrivable(1, 1) &&
               "changing only the mask image must change isDrivable() results");
        assert(withMaskA.getCenterlineSamples()[0].x == withMaskB.getCenterlineSamples()[0].x &&
               withMaskA.getCenterlineSamples()[0].y == withMaskB.getCenterlineSamples()[0].y &&
               "the two tracks' centerlines must be identical -- only the mask asset differs");
    }

    // 27, 28 & 30: changing only the visual image never changes
    // isDrivable(), and TrackProgress reports identical lap positions
    // regardless of the visual asset.
    {
        simulation::Track withVisualA(makeTinyFixtureDefinition(visualA, maskA));
        simulation::Track withVisualB(makeTinyFixtureDefinition(visualB, maskA));

        for (int y = 0; y < 8; ++y)
        {
            for (int x = 0; x < 8; ++x)
            {
                assert(withVisualA.isDrivable(x, y) == withVisualB.isDrivable(x, y) &&
                       "changing only the visual image must never change isDrivable()");
            }
        }

        simulation::TrackProgress progressA(withVisualA);
        simulation::TrackProgress progressB(withVisualB);
        simulation::Car probeCar(makeCarParams(), withVisualA);
        probeCar.reset(withVisualA.getSpawnPosition(), withVisualA.getSpawnHeading());
        progressA.reset(probeCar);
        progressB.reset(probeCar);
        assert(progressA.getLapPosition() == progressB.getLapPosition() &&
               "TrackProgress must not depend on which visual image its Track has");
    }

    // 14 (negative case): a computed spawn position landing on a
    // non-drivable mask pixel must be rejected.
    {
        simulation::TrackDefinition def = makeTinyFixtureDefinition(visualA, maskA);
        def.controlPoints = image_track_verify::spawnInNonDrivableCornerControlPoints();
        assert(throwsInvalidArgument([&]() { simulation::Track t(def); }) &&
               "a spawn position outside the drivable mask must be rejected");
    }

    // A known-non-drivable interior point for the active track: (948, 192)
    // is the deepest point (55px from the nearest drivable pixel) of the
    // mask's enclosed black region -- a robust interior point, not an edge case.
    {
        assert(!track.isDrivable(948, 192) &&
               "a known-deep-interior point of the extreme track's enclosed black region must be non-drivable");
    }

    // 12: the mask's road width genuinely varies along the route -- not one
    // constant. Both sample indices are measured on straight stretches
    // (measureRoadWidthAt()'s perpendicular scan inflates width on curves).
    {
        constexpr std::size_t wideSample = 20;   // mid top straight (P1->P2)
        constexpr std::size_t narrowSample = 571; // mid short S-connector straight (P28->P29)

        const float wideWidth = measureRoadWidthAt(track, wideSample);
        const float narrowWidth = measureRoadWidthAt(track, narrowSample);

        assert(wideWidth > 0.0f && narrowWidth > 0.0f && "both probed points must be on the drivable band");
        assert(wideWidth > narrowWidth + 15.0f &&
               "the mask must contain a measurably wider section than another -- no single constant width");
    }

    // 19 & 20: local tracking follows a normal, small forward step exactly
    // (matching a full global projection), including wrapping correctly
    // around the closed loop's seam.
    {
        const std::vector<Vector2>& centerline = track.getCenterlineSamples();
        const std::size_t sampleCount = centerline.size();

        // previousSegmentIndex=2, query a point near the seam from the far
        // side -- only reachable if the window wraps past index 0.
        const std::size_t nearSeamIndex = sampleCount - 8;
        const simulation::TrackProjection wrapped =
            track.projectOntoCenterlineLocal(centerline[nearSeamIndex], 2, simulation::TrackProgress::kLocalSearchRadius);
        assert(wrapped.distanceFromCenterline < 1.0f && wrapped.segmentIndex == nearSeamIndex - 1 &&
               "local search must wrap correctly across the closed loop's seam");

        // Symmetric case: previousSegmentIndex near the end, query a point
        // just past the seam at the start.
        const simulation::TrackProjection wrappedOther = track.projectOntoCenterlineLocal(
            centerline[5], sampleCount - 3, simulation::TrackProgress::kLocalSearchRadius);
        assert(wrappedOther.distanceFromCenterline < 1.0f && wrappedOther.segmentIndex == 4 &&
               "local search must wrap correctly in both directions across the seam");
    }

    // 21: a position exactly on a logically distant segment (outside the
    // local window) is not found by local search even at zero true
    // distance -- proving distant sections are structurally excluded, not
    // just deprioritized.
    {
        const std::vector<Vector2>& centerline = track.getCenterlineSamples();
        // Exactly opposite the loop -- the largest possible index gap, so
        // guaranteed outside the window regardless of kLocalSearchRadius.
        const std::size_t kFarIndex = centerline.size() / 2;
        constexpr std::size_t kPreviousIndex = 0;

        const simulation::TrackProjection localResult = track.projectOntoCenterlineLocal(
            centerline[kFarIndex], kPreviousIndex, simulation::TrackProgress::kLocalSearchRadius);

        assert(localResult.segmentIndex != kFarIndex &&
               "local search must not find a segment outside its window, even at zero true distance");
        assert(localResult.distanceFromCenterline > 10.0f &&
               "local search's in-window result must be measurably farther than the excluded true match");

        // Tie-break: segments kFarIndex-1 and kFarIndex are equally close;
        // ascending-order scan with strict '<' picks kFarIndex-1 (see verifyTrack()).
        const simulation::TrackProjection globalResult = track.projectOntoCenterline(centerline[kFarIndex]);
        assert(globalResult.segmentIndex == kFarIndex - 1 && globalResult.distanceFromCenterline < 1.0f &&
               "a full global scan, for contrast, must find the exact match local search deliberately excluded");
    }

    // 22, 23 & 24: global recovery re-anchors tracking when local
    // projection clearly fails, without awarding implausible progress --
    // and reset() re-anchors correctly afterward.
    {
        simulation::Car car(makeCarParams(), track);
        simulation::TrackProgress progress(track);

        car.reset(kSpawnPosition, kSpawnHeading);
        progress.reset(car);
        assert(progress.getTrackedSegmentIndex() == track.projectOntoCenterline(kSpawnPosition).segmentIndex &&
               "reset must anchor the tracked segment to a full global projection of the reset position");

        // Jump far away (beyond kLocalSearchRadius) to trigger recovery.
        // 0.6 of a lap is verified offline to land ~330px from spawn's
        // local window, safely past kLocalProjectionRecoveryDistance (250px).
        const Vector2 distantPoint = track_progress_verify::positionAtLapPosition(track, 0.6f);
        car.reset(distantPoint, kSpawnHeading);
        progress.update(car);

        const simulation::TrackProjection expectedRecovery = track.projectOntoCenterline(distantPoint);
        assert(progress.getTrackedSegmentIndex() == expectedRecovery.segmentIndex &&
               "recovery must re-anchor the tracked segment to the correct (globally projected) segment");
        assert(progress.getBestProgress() == 0.0f && progress.getTotalCheckpointsPassed() == 0 &&
               "recovery must not award progress or checkpoints for an implausible jump -- the plausibility "
               "gate still applies to whatever position recovery reports");

        // A small ordinary step now DOES register as progress -- recovery
        // re-anchors tracking, it doesn't leave it stuck.
        const Vector2 nearbyPoint = track_progress_verify::positionAtLapPosition(track, 0.605f);
        car.reset(nearbyPoint, kSpawnHeading);
        progress.update(car);
        assert(progress.getContinuousProgress() > 0.0f &&
               "a normal small step immediately after recovery must register as ordinary forward progress");

        // 24: reset() discards the stale (far-away) tracked segment and
        // re-anchors back to the car's actual (spawn) position.
        car.reset(kSpawnPosition, kSpawnHeading);
        progress.reset(car);
        assert(progress.getTrackedSegmentIndex() == track.projectOntoCenterline(kSpawnPosition).segmentIndex &&
               "reset must discard any previously tracked segment and re-anchor from scratch");
    }

    // Regression case for the extreme track's tight top-left hairpin
    // (createExtremeTrackDefinition()'s P2-P6). Section A (~index 42) is
    // the hairpin's entry; section B (~index 89) curves back close enough
    // to be geometrically nearer at some positions despite being a real
    // ~47-sample jump away. (153.5, 136.5), from a live debug log, is one
    // such position: under pure nearest-distance selection it was ~34.8px
    // from B vs. ~36-40px from A's continuation, snapping tracking forward
    // to B. With continuity-aware scoring, tracking must stay on A.
    {
        const std::vector<Vector2>& centerline = track.getCenterlineSamples();
        constexpr std::size_t kSectionA = 42;
        constexpr std::size_t kSectionB = 89;
        constexpr Vector2 kAmbiguousPosition = {153.5f, 136.5f};

        simulation::Car car(makeCarParams(), track);
        simulation::TrackProgress progress(track);

        // Anchor tracking on section A via a real reset() (exercises the
        // actual production path). Tie-break: kSectionA-1 and kSectionA are
        // equally close; the ascending-order scan picks kSectionA-1.
        car.reset(centerline[kSectionA], kSpawnHeading);
        progress.reset(car);
        assert(progress.getTrackedSegmentIndex() == kSectionA - 1 &&
               "setup must anchor tracking exactly on section A");

        // The ambiguous position: local tracking must stay close to
        // section A (within the documented free zone plus a small margin
        // for the update itself), never near section B.
        car.reset(kAmbiguousPosition, kSpawnHeading);
        progress.update(car);
        const std::size_t afterAmbiguous = progress.getTrackedSegmentIndex();
        const std::size_t gapFromA =
            std::min((afterAmbiguous + centerline.size() - kSectionA) % centerline.size(),
                     (kSectionA + centerline.size() - afterAmbiguous) % centerline.size());
        const std::size_t gapFromB =
            std::min((afterAmbiguous + centerline.size() - kSectionB) % centerline.size(),
                     (kSectionB + centerline.size() - afterAmbiguous) % centerline.size());
        assert(gapFromA <= 15 &&
               "local tracking must stay on (or very near) section A at the ambiguous position");
        assert(gapFromA < gapFromB &&
               "local tracking must end up clearly closer to section A than to section B");

        // Normal forward movement: a small step further along the actual
        // centerline (still well within the hairpin) must be tracked
        // forward normally, registering as ordinary positive progress.
        const float progressBeforeForward = progress.getContinuousProgress();
        car.reset(centerline[(afterAmbiguous + 3) % centerline.size()], kSpawnHeading);
        progress.update(car);
        assert(progress.getContinuousProgress() > progressBeforeForward &&
               "a small forward step must register as ordinary forward progress");

        // Small backward movement: stepping back a couple of samples must
        // still be tracked correctly (continuous progress decreases, best
        // progress does not).
        const float bestAfterForward = progress.getBestProgress();
        const std::size_t beforeBackward = progress.getTrackedSegmentIndex();
        car.reset(centerline[(beforeBackward + centerline.size() - 2) % centerline.size()], kSpawnHeading);
        progress.update(car);
        assert(progress.getContinuousProgress() < progressBeforeForward &&
               "a small backward step must decrease continuous progress");
        assert(progress.getBestProgress() == bestAfterForward && "backward movement must not raise best progress");

        // Loop wraparound: reset near the very end of the closed loop, then
        // step to a position just past the start -- must be tracked as a
        // small forward (seam-crossing) step, not a large jump or a
        // rejection, through TrackProgress's real update() path.
        const std::size_t nearEnd = centerline.size() - 3;
        car.reset(centerline[nearEnd], kSpawnHeading);
        progress.reset(car);
        car.reset(centerline[2], kSpawnHeading);
        progress.update(car);
        const std::size_t afterWrap = progress.getTrackedSegmentIndex();
        const std::size_t wrapGap = std::min((afterWrap + centerline.size() - 2) % centerline.size(),
                                              (2 + centerline.size() - afterWrap) % centerline.size());
        assert(wrapGap <= 5 && "wraparound tracking must land close to the true post-seam position");
        assert(progress.getContinuousProgress() > 0.0f &&
               "crossing the seam forward must register as ordinary forward progress, not be rejected");
    }

    TraceLog(LOG_INFO, "Image-based track system verification: all deterministic checks passed");
}
} // namespace verification
