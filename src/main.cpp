#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
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

namespace
{

constexpr int kSimWidth = 1200;
constexpr int kSimHeight = 700;
constexpr int kPanelWidth = 400;
constexpr int kScreenWidth = kSimWidth + kPanelWidth;
constexpr int kScreenHeight = kSimHeight;

// Stage 2 executes exactly one fixed-size simulation step per rendered
// frame. This is intentionally not GetFrameTime(): Car must always see the
// same dt regardless of measured render duration.
constexpr float kSimulationDt = 1.0f / 60.0f;

// Stage 14B: the track is the generalized, centerline-based hard training
// circuit (see simulation::createHardTrackDefinition()) -- this is the only
// place its shape is chosen.
simulation::TrackDefinition makeTrackDefinition()
{
    return simulation::createHardTrackDefinition(kSimWidth, kSimHeight);
}

// The spawn pose is derived entirely from the Track itself
// (Track::getSpawnPosition()/getSpawnHeading(), which in turn come from the
// sampled centerline -- see Track.h/.cpp), never hardcoded here. This is
// computed once, from a Track built solely for that purpose (identical
// geometry to the `track` constructed in main() below, since both come from
// the same deterministic makeTrackDefinition()), so every verify*()
// function and every Population/Individual construction in this file
// shares exactly the same deterministic spawn pose without duplicating any
// coordinates.
struct SpawnPose
{
    Vector2 position;
    float heading;
};

SpawnPose computeSpawnPose()
{
    const simulation::Track referenceTrack(makeTrackDefinition());
    return SpawnPose{referenceTrack.getSpawnPosition(), referenceTrack.getSpawnHeading()};
}

const SpawnPose kSpawnPose = computeSpawnPose();
const Vector2 kSpawnPosition = kSpawnPose.position;
const float kSpawnHeading = kSpawnPose.heading;

simulation::CarParams makeCarParams()
{
    return simulation::CarParams{};
}

// Builds one hand-built, deterministic demonstration Genome: 9 Input nodes
// (IDs 0..8, matching Observation slot order -- see ai::Observation), 1 Bias
// node, 2 Output nodes, and only direct Input/Bias -> Output connections (no
// hidden nodes). All weights below are fixed literals chosen by hand to
// produce visibly reactive steering/throttle -- this is NOT a trained or
// evolved network, and no random values are used anywhere in its
// construction.
//
// Observation input slots used here (see ai::Observation for the full list):
//   0 = sensor at -60 deg (left),   1 = sensor at -30 deg (left)
//   2 = sensor at   0 deg (center)
//   3 = sensor at +30 deg (right),  4 = sensor at +60 deg (right)
//
// Steering (output ID 100, the lower Output ID -> output slot 0):
//   left sensors  (0, 1) -> steering, weight -0.5 / -0.3  (more open space on
//                            the left pulls steering negative)
//   right sensors (3, 4) -> steering, weight +0.3 / +0.5  (more open space on
//                            the right pulls steering positive)
//   center sensor (2)    -> no connection (zero contribution, per the
//                            "may contribute zero" guidance)
//
// Throttle (output ID 101, the higher Output ID -> output slot 1):
//   Bias (9)       -> throttle, weight +0.6 (steady baseline forward drive)
//   center sensor  -> throttle, weight +0.4 (more open space ahead adds a
//                      little more throttle on top of the baseline)
ai::neat::Genome createDemonstrationGenome()
{
    using ai::neat::ConnectionGene;
    using ai::neat::Genome;
    using ai::neat::NodeGene;
    using ai::neat::NodeId;
    using ai::neat::NodeType;

    constexpr NodeId kSensorLeft60 = 0;
    constexpr NodeId kSensorLeft30 = 1;
    constexpr NodeId kSensorCenter = 2;
    constexpr NodeId kSensorRight30 = 3;
    constexpr NodeId kSensorRight60 = 4;
    constexpr NodeId kBiasId = 9;
    constexpr NodeId kSteeringOutputId = 100; // lower Output ID -> output slot 0
    constexpr NodeId kThrottleOutputId = 101; // higher Output ID -> output slot 1

    Genome genome;
    for (int i = 0; i < ai::NeuralNetwork::kInputCount; ++i)
    {
        genome.addNode(NodeGene{i, NodeType::Input});
    }
    genome.addNode(NodeGene{kBiasId, NodeType::Bias});
    genome.addNode(NodeGene{kSteeringOutputId, NodeType::Output});
    genome.addNode(NodeGene{kThrottleOutputId, NodeType::Output});

    int innovation = 0;
    genome.addConnection(ConnectionGene{kSensorLeft60, kSteeringOutputId, -0.5f, true, innovation++});
    genome.addConnection(ConnectionGene{kSensorLeft30, kSteeringOutputId, -0.3f, true, innovation++});
    genome.addConnection(ConnectionGene{kSensorRight30, kSteeringOutputId, 0.3f, true, innovation++});
    genome.addConnection(ConnectionGene{kSensorRight60, kSteeringOutputId, 0.5f, true, innovation++});
    genome.addConnection(ConnectionGene{kBiasId, kThrottleOutputId, 0.6f, true, innovation++});
    genome.addConnection(ConnectionGene{kSensorCenter, kThrottleOutputId, 0.4f, true, innovation++});

    return genome;
}

// One-shot, deterministic sanity check of the CPU mask against the known
// track geometry. Runs once at startup, never inside the render loop.
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
// (Stage 14B's self-intersection check on the sampled centerline), never by
// any production/runtime code path.
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

    // 8 & 9: the loop is closed (the wraparound segment from the last
    // sample back to the first is a real, non-degenerate segment, not a
    // duplicate zero-length seam sample) and every ordinary consecutive
    // pair of samples is distinct. Stage 14B's hard track deliberately
    // mixes a long straight with tight corners, so consecutive-sample
    // spacing varies a lot more than the old easy oval's did (Catmull-Rom
    // sampling is uniform in the parametric t, not in arc length, so a
    // straight section advances much further per sample than a tight
    // corner does) -- this check only requires every segment to be
    // non-degenerate, not similarly sized.
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

    // 13, 14, 15, 17 & 18: projecting a centerline sample itself back onto
    // the centerline returns (approximately) that same point, at
    // near-zero distance, with a deterministic tie-break to the lower of
    // the two segments meeting at that sample (the segment ending there,
    // not the one starting there, since segments are scanned in ascending
    // index order and only a strictly smaller distance updates the best
    // match).
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

        // The projected point must actually lie on the chosen segment: it
        // must be collinear with the segment's two endpoints and within
        // the segment's span (segmentT already checked above).
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

    // 19, 20 & 21: the CPU mask (built once from the centerline + track
    // width) agrees with that same geometry -- drivable near the
    // centerline, non-drivable well outside the road band, and
    // non-drivable outside the simulation bounds.
    {
        const Vector2 spawn = track.getSpawnPosition();
        assert(track.isDrivable(static_cast<int>(spawn.x), static_cast<int>(spawn.y)) &&
               "a point on the centerline must be drivable");

        const float centerX = static_cast<float>(def.simWidth) * 0.5f;
        const float centerY = static_cast<float>(def.simHeight) * 0.5f;
        assert(!track.isDrivable(static_cast<int>(centerX), static_cast<int>(centerY)) &&
               "the middle of a closed loop track, well beyond the road width, must be non-drivable");

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

    // Stage 14B: the hard track's asymmetric layout (a long straight, a
    // broad sweep, tighter corners and an S-chicane, all folded into a
    // single closed loop within a bounded simulation area) makes it
    // possible in principle for two *non-adjacent* sections to end up too
    // close together, or even cross -- unlike the old easy oval, whose
    // convex, single-curvature shape ruled that out by construction. These
    // two checks are deterministic, O(sampleCount^2) geometry checks over
    // the actual sampled centerline (the same one the mask/rendering/
    // progress all use) -- purely verification, not a runtime path, and
    // they do not change projectOntoCenterline() or add any spatial
    // acceleration structure to Track itself.
    {
        const std::size_t n = centerline.size();

        // Segments within this many indices of each other (in either
        // direction around the closed loop, including the wraparound) are
        // "adjacent" for these purposes -- ordinary consecutive curvature
        // (especially right around a sharp corner apex, where samples can
        // bunch up spatially) naturally brings them close/touching, so they
        // are excluded from both checks. samplesPerSegment is 24, so this
        // window covers a bit more than a full control-point segment on
        // either side of any shared vertex -- comfortably more than enough
        // for the sharpest corner this track uses, while still being a
        // small fraction of the hundreds of samples separating genuinely
        // distinct sections (e.g. the S-chicane from the bottom straight),
        // so it cannot hide a real accidental overlap between them.
        constexpr std::size_t kAdjacencyWindow = 40;

        // A conservative separation floor: comfortably greater than the
        // track's own road width, so two non-adjacent sections can never
        // have overlapping drivable bands (each extends trackWidth/2 from
        // its own centerline) with room to spare. 1.2x is a deliberately
        // moderate margin (not just barely over 1.0x, where bands would
        // only just avoid touching) chosen to fit Stage 14B's tighter,
        // more convoluted hard-track layout -- see createHardTrackDefinition().
        const float minSeparation = def.trackWidth * 1.2f;

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
                    continue; // adjacent (or the same/neighboring) segment -- not a candidate for overlap
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

        // 8: no two non-adjacent centerline segments may cross.
        assert(!anyIntersection && "the closed centerline must not self-intersect between non-adjacent sections");

        // 9: non-adjacent segments must stay comfortably farther apart than
        // the road is wide, so their drivable bands cannot overlap and a
        // car's nearest-centerline projection cannot ambiguously jump
        // between them.
        assert(worstSeparation >= minSeparation &&
               "non-adjacent centerline sections must maintain a reasonable separation");
    }

    TraceLog(LOG_INFO, "Track verification: all centerline/arc-length/projection/mask checks passed");
}

// One-shot, deterministic sanity check of Car's dynamics and collision,
// independent of any keyboard/render timing. Runs once at startup.
void verifyCar(const simulation::Track& track)
{
    simulation::Car car(makeCarParams(), track);
    car.reset(kSpawnPosition, kSpawnHeading);

    assert(car.isAlive() && "car must be spawned alive");
    assert(car.getPosition().x == kSpawnPosition.x && car.getPosition().y == kSpawnPosition.y &&
           "reset must place the car at the spawn position");

    // Steering alone, while stationary, must not rotate the car.
    simulation::CarInput steerOnly;
    steerOnly.throttle = 0.0f;
    steerOnly.steering = 1.0f;
    for (int i = 0; i < 30; ++i)
    {
        car.update(steerOnly, kSimulationDt);
    }
    assert(car.getHeading() == kSpawnHeading && "steering must have no effect while stationary");

    // Sustained throttle must build up speed from rest.
    simulation::CarInput throttleOnly;
    throttleOnly.throttle = 1.0f;
    throttleOnly.steering = 0.0f;
    for (int i = 0; i < 30; ++i)
    {
        car.update(throttleOnly, kSimulationDt);
    }
    const Vector2 v = car.getVelocity();
    const float speed = std::sqrt(v.x * v.x + v.y * v.y);
    assert(speed > 1.0f && "sustained throttle must build up visible speed");
    assert(speed <= car.getParams().maxSpeed + 0.01f && "speed must never exceed maxSpeed");

    // Driving straight off the track must kill the car and zero its velocity.
    car.reset(kSpawnPosition, kSpawnHeading);
    simulation::CarInput driveOffTrack;
    driveOffTrack.throttle = 1.0f;
    driveOffTrack.steering = 0.0f;
    for (int i = 0; i < 300 && car.isAlive(); ++i)
    {
        car.update(driveOffTrack, kSimulationDt);
    }
    assert(!car.isAlive() && "driving straight for 5s must leave the road band and kill the car");
    const Vector2 deadVelocity = car.getVelocity();
    assert(deadVelocity.x == 0.0f && deadVelocity.y == 0.0f && "a dead car must have zero velocity");

    // A dead car must ignore further input.
    const Vector2 deadPosition = car.getPosition();
    car.update(throttleOnly, kSimulationDt);
    assert(car.getPosition().x == deadPosition.x && car.getPosition().y == deadPosition.y &&
           "a dead car must not respond to further input");

    // Reset must revive the car.
    car.reset(kSpawnPosition, kSpawnHeading);
    assert(car.isAlive() && "reset must revive the car");
    assert(car.getVelocity().x == 0.0f && car.getVelocity().y == 0.0f && "reset must zero velocity");

    TraceLog(LOG_INFO, "Car verification: all deterministic checks passed");
}

// One-shot, deterministic sanity check of Car's five sensors, independent of
// keyboard/render timing. Runs once at startup.
void verifySensors(const simulation::Track& track)
{
    static_assert(simulation::Car::kSensorCount == 5, "Stage 3 requires exactly five sensors");

    simulation::Car car(makeCarParams(), track);
    car.reset(kSpawnPosition, kSpawnHeading);

    // 1: exactly five sensor readings exist, in fixed-size storage.
    const auto& sensors = car.getSensors();
    assert(sensors.size() == 5 && "exactly five sensor readings must exist");

    // 2 & 3: normalized readings stay within [0,1] and are valid immediately after reset.
    for (const simulation::SensorReading& s : sensors)
    {
        assert(s.normalizedDistance >= 0.0f && s.normalizedDistance <= 1.0f &&
               "normalized sensor distance must stay within [0,1]");
        assert(s.distance >= 0.0f && s.distance <= simulation::Car::kMaxSensorDistance &&
               "raw sensor distance must stay within [0, kMaxSensorDistance]");
    }

    // 4: the front (0 degree) sensor points in the same world direction as the car heading.
    {
        const Vector2 origin = car.getSensorOrigin();
        const simulation::SensorReading& front = sensors[2];
        const float rayAngle = std::atan2(front.endPoint.y - origin.y, front.endPoint.x - origin.x);
        assert(std::fabs(rayAngle - kSpawnHeading) < 0.01f &&
               "front sensor must point along the car heading");
    }

    // 5: resetting to a known heading rotates the whole sensor layout with it.
    {
        const float knownHeading = static_cast<float>(PI) * 0.5f; // pointing +y ("down")
        car.reset(kSpawnPosition, knownHeading);
        const Vector2 origin = car.getSensorOrigin();
        const auto& rotatedSensors = car.getSensors();

        for (int i = 0; i < simulation::Car::kSensorCount; ++i)
        {
            const float expectedAngle = knownHeading + simulation::Car::kSensorAngleDegrees[i] * DEG2RAD;
            const Vector2 expectedDir = {std::cos(expectedAngle), std::sin(expectedAngle)};
            const Vector2 rayDir = {rotatedSensors[i].endPoint.x - origin.x, rotatedSensors[i].endPoint.y - origin.y};
            const float rayLen = std::sqrt(rayDir.x * rayDir.x + rayDir.y * rayDir.y);
            assert(rayLen > 0.01f && "sensor ray must have nonzero length");
            const Vector2 rayDirNorm = {rayDir.x / rayLen, rayDir.y / rayLen};
            const float dot = rayDirNorm.x * expectedDir.x + rayDirNorm.y * expectedDir.y;
            assert(dot > 0.999f && "rotated sensor direction must match heading + relative angle");
        }

        car.reset(kSpawnPosition, kSpawnHeading);
    }

    // 6: a sensor directed at a nearby track boundary reports less than
    // maximum distance. Spawn sits on the centerline of a road band half
    // as wide as Track::getDefinition().trackWidth, so a sensor aimed 90
    // degrees off the spawn heading (i.e. across the road, not along it)
    // must hit that edge well within the 200px sensor range, regardless of
    // the spawn tangent's exact absolute direction -- unlike a hardcoded
    // absolute heading, this stays correct for any track's spawn geometry.
    {
        car.reset(kSpawnPosition, kSpawnHeading + static_cast<float>(PI) * 0.5f);
        const simulation::SensorReading& front = car.getSensors()[2];
        assert(front.distance < simulation::Car::kMaxSensorDistance &&
               front.normalizedDistance < 1.0f &&
               "sensor aimed at a nearby boundary must report less than maximum distance");
        car.reset(kSpawnPosition, kSpawnHeading);
    }

    // 7: a ray with no obstacle within range reports exactly maximum distance / normalized 1.0.
    // At spawn heading 0, the front sensor points along the long bottom straight, clear for 200px.
    {
        const simulation::SensorReading& front = car.getSensors()[2];
        assert(front.distance == simulation::Car::kMaxSensorDistance &&
               "unobstructed sensor must report exactly the maximum distance");
        assert(front.normalizedDistance == 1.0f &&
               "unobstructed sensor must report exactly normalized 1.0");
    }

    // 8: sensor endpoints correspond to their recorded distance and direction.
    {
        const Vector2 origin = car.getSensorOrigin();
        for (int i = 0; i < simulation::Car::kSensorCount; ++i)
        {
            const float angle = kSpawnHeading + simulation::Car::kSensorAngleDegrees[i] * DEG2RAD;
            const Vector2 direction = {std::cos(angle), std::sin(angle)};
            const simulation::SensorReading& s = sensors[i];
            const Vector2 expectedEnd = {origin.x + direction.x * s.distance, origin.y + direction.y * s.distance};
            assert(std::fabs(expectedEnd.x - s.endPoint.x) < 0.01f &&
                   std::fabs(expectedEnd.y - s.endPoint.y) < 0.01f &&
                   "sensor endpoint must match its recorded distance and direction");
        }
    }

    // 9: a dead car retains its final sensor readings and does not recast while stationary.
    {
        car.reset(kSpawnPosition, kSpawnHeading);
        simulation::CarInput driveOffTrack;
        driveOffTrack.throttle = 1.0f;
        driveOffTrack.steering = 0.0f;
        for (int i = 0; i < 300 && car.isAlive(); ++i)
        {
            car.update(driveOffTrack, kSimulationDt);
        }
        assert(!car.isAlive() && "driving straight for 5s must leave the road band and kill the car");

        const auto finalSensors = car.getSensors();
        car.update(driveOffTrack, kSimulationDt);
        const auto& sensorsAfterDeadUpdate = car.getSensors();
        for (int i = 0; i < simulation::Car::kSensorCount; ++i)
        {
            assert(finalSensors[i].distance == sensorsAfterDeadUpdate[i].distance &&
                   finalSensors[i].normalizedDistance == sensorsAfterDeadUpdate[i].normalizedDistance &&
                   "a dead car must retain its final sensor readings, not recast them");
        }
    }

    car.reset(kSpawnPosition, kSpawnHeading);

    TraceLog(LOG_INFO, "Sensor verification: all deterministic checks passed");
}

// One-shot, deterministic sanity check of ai::buildObservation, independent
// of keyboard/render timing. Runs once at startup.
void verifyObservation(const simulation::Track& track)
{
    static_assert(ai::kObservationSize == 9, "Stage 4 requires exactly nine observation values");

    simulation::Car car(makeCarParams(), track);
    car.reset(kSpawnPosition, kSpawnHeading);

    // 1: exactly nine values, fixed-size storage.
    ai::Observation obs = ai::buildObservation(car);
    assert(obs.values.size() == 9 && "observation must contain exactly nine values");

    // 2 & 3: sensor values occupy indices 0..4, in documented order, within [0,1].
    const auto& sensors = car.getSensors();
    for (int i = 0; i < simulation::Car::kSensorCount; ++i)
    {
        assert(std::fabs(obs.values[i] - sensors[i].normalizedDistance) < 1e-6f &&
               "observation sensor slot must match Car's own normalized sensor reading");
        assert(obs.values[i] >= 0.0f && obs.values[i] <= 1.0f && "sensor observation values must stay within [0,1]");
    }

    // 4: speed normalization (speed / maxSpeed, clamped to [0,1]).
    {
        const float expected = std::clamp(car.getSpeed() / car.getMaxSpeed(), 0.0f, 1.0f);
        assert(std::fabs(obs.values[5] - expected) < 1e-6f && "speed normalization mismatch");
    }

    // 5: forward velocity normalization (forwardVelocity / maxSpeed, clamped to [-1,1]).
    {
        const float expected = std::clamp(car.getForwardVelocity() / car.getMaxSpeed(), -1.0f, 1.0f);
        assert(std::fabs(obs.values[6] - expected) < 1e-6f && "forward velocity normalization mismatch");
    }

    // 6: lateral velocity normalization (lateralVelocity / maxSpeed, clamped to [-1,1]).
    {
        const float expected = std::clamp(car.getLateralVelocity() / car.getMaxSpeed(), -1.0f, 1.0f);
        assert(std::fabs(obs.values[7] - expected) < 1e-6f && "lateral velocity normalization mismatch");
    }

    // 7: slip angle normalization (slipAngle / pi, clamped to [-1,1]).
    {
        const float expected = std::clamp(car.getSlipAngle() / static_cast<float>(PI), -1.0f, 1.0f);
        assert(std::fabs(obs.values[8] - expected) < 1e-6f && "slip angle normalization mismatch");
    }

    // 8: all normalized values respect their documented ranges even under active driving/sliding.
    simulation::CarInput throttleAndSteer;
    throttleAndSteer.throttle = 1.0f;
    throttleAndSteer.steering = 1.0f;
    for (int i = 0; i < 30 && car.isAlive(); ++i)
    {
        car.update(throttleAndSteer, kSimulationDt);
    }
    const ai::Observation movingObs = ai::buildObservation(car);
    for (int i = 0; i < simulation::Car::kSensorCount; ++i)
    {
        assert(movingObs.values[i] >= 0.0f && movingObs.values[i] <= 1.0f && "sensor values must stay within [0,1]");
    }
    assert(movingObs.values[5] >= 0.0f && movingObs.values[5] <= 1.0f && "speed must stay within [0,1]");
    assert(movingObs.values[6] >= -1.0f && movingObs.values[6] <= 1.0f && "forward velocity must stay within [-1,1]");
    assert(movingObs.values[7] >= -1.0f && movingObs.values[7] <= 1.0f && "lateral velocity must stay within [-1,1]");
    assert(movingObs.values[8] >= -1.0f && movingObs.values[8] <= 1.0f && "slip angle must stay within [-1,1]");

    // 9: reset/spawn produces deterministic valid observation values.
    car.reset(kSpawnPosition, kSpawnHeading);
    const ai::Observation resetObsA = ai::buildObservation(car);
    car.reset(kSpawnPosition, kSpawnHeading);
    const ai::Observation resetObsB = ai::buildObservation(car);
    for (int i = 0; i < ai::kObservationSize; ++i)
    {
        assert(resetObsA.values[i] == resetObsB.values[i] && "reset must produce deterministic observation values");
    }

    TraceLog(LOG_INFO, "Observation verification: all deterministic checks passed");
}

namespace nn_verify
{

// Builds the 9 Input + 1 Bias + 2 Output nodes every test network needs.
// Input node IDs are 0..8 (Observation slot order), bias is 9, steering
// output is 100 (first Output -> output index 0), throttle output is 101
// (second Output -> output index 1). Deliberately not contiguous/sorted
// with any hidden node IDs used below, so tests can prove evaluation
// doesn't depend on ID ordering.
std::vector<ai::Node> makeBaseNodes()
{
    std::vector<ai::Node> nodes;
    for (int i = 0; i < ai::NeuralNetwork::kInputCount; ++i)
    {
        nodes.push_back(ai::Node{i, ai::NodeType::Input});
    }
    nodes.push_back(ai::Node{9, ai::NodeType::Bias});
    nodes.push_back(ai::Node{100, ai::NodeType::Output}); // steering
    nodes.push_back(ai::Node{101, ai::NodeType::Output}); // throttle
    return nodes;
}

ai::Observation makeObservation(int index, float value)
{
    ai::Observation obs;
    obs.values.fill(0.0f);
    if (index >= 0)
    {
        obs.values[index] = value;
    }
    return obs;
}

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

} // namespace nn_verify

// One-shot, deterministic sanity check of ai::NeuralNetwork's construction,
// validation and evaluation, independent of Car/Track/keyboard/render
// timing. Runs once at startup.
void verifyNeuralNetwork()
{
    using namespace nn_verify;
    constexpr float kEps = 1e-4f;

    // 1 & 9: exactly 9 inputs + 1 bias + 2 outputs can be constructed; a
    // fully disconnected Output produces 0 (tanh of an empty sum).
    {
        ai::NeuralNetwork net(makeBaseNodes(), {});
        const auto out = net.evaluate(makeObservation(-1, 0.0f));
        assert(out[0] == 0.0f && out[1] == 0.0f && "disconnected outputs must evaluate to exactly 0");
    }

    // 2: direct Input -> Output connection produces the expected tanh result.
    {
        ai::NeuralNetwork net(makeBaseNodes(), {ai::Connection{0, 100, 0.5f, true}});
        const auto out = net.evaluate(makeObservation(0, 1.0f));
        assert(std::fabs(out[0] - std::tanh(0.5f)) < kEps && "Input->Output must equal tanh(input * weight)");
        assert(out[1] == 0.0f && "unrelated disconnected output must stay 0");
    }

    // 3: Bias -> Output affects output correctly (bias is always 1.0).
    {
        ai::NeuralNetwork net(makeBaseNodes(), {ai::Connection{9, 100, 0.7f, true}});
        const auto out = net.evaluate(makeObservation(-1, 0.0f));
        assert(std::fabs(out[0] - std::tanh(0.7f)) < kEps && "Bias->Output must equal tanh(1.0 * weight)");
    }

    // 4: multiple incoming connections are summed before activation.
    {
        std::vector<ai::Connection> conns = {
            ai::Connection{0, 100, 0.3f, true},
            ai::Connection{1, 100, 0.4f, true},
        };
        ai::NeuralNetwork net(makeBaseNodes(), conns);
        ai::Observation obs;
        obs.values.fill(0.0f);
        obs.values[0] = 1.0f;
        obs.values[1] = 1.0f;
        const auto out = net.evaluate(obs);
        assert(std::fabs(out[0] - std::tanh(0.3f + 0.4f)) < kEps &&
               "multiple incoming connections must be summed before tanh");
    }

    // 5: Input -> Hidden -> Output produces the mathematically expected result.
    {
        std::vector<ai::Node> nodes = makeBaseNodes();
        nodes.push_back(ai::Node{50, ai::NodeType::Hidden});
        std::vector<ai::Connection> conns = {
            ai::Connection{0, 50, 0.5f, true},
            ai::Connection{50, 100, 2.0f, true},
        };
        ai::NeuralNetwork net(nodes, conns);
        const auto out = net.evaluate(makeObservation(0, 1.0f));
        const float hidden = std::tanh(1.0f * 0.5f);
        const float expected = std::tanh(hidden * 2.0f);
        assert(std::fabs(out[0] - expected) < kEps && "Input->Hidden->Output result mismatch");
    }

    // 6: a deeper feed-forward path Input -> HiddenA -> HiddenB -> Output evaluates correctly.
    {
        std::vector<ai::Node> nodes = makeBaseNodes();
        nodes.push_back(ai::Node{50, ai::NodeType::Hidden});
        nodes.push_back(ai::Node{51, ai::NodeType::Hidden});
        std::vector<ai::Connection> conns = {
            ai::Connection{0, 50, 1.0f, true},
            ai::Connection{50, 51, 1.0f, true},
            ai::Connection{51, 100, 1.0f, true},
        };
        ai::NeuralNetwork net(nodes, conns);
        const auto out = net.evaluate(makeObservation(0, 0.5f));
        const float a = std::tanh(0.5f);
        const float b = std::tanh(a);
        const float expected = std::tanh(b);
        assert(std::fabs(out[0] - expected) < kEps && "deep Input->HiddenA->HiddenB->Output result mismatch");
    }

    // 7: a connection that skips hidden nodes evaluates correctly alongside a hidden path.
    {
        std::vector<ai::Node> nodes = makeBaseNodes();
        nodes.push_back(ai::Node{50, ai::NodeType::Hidden});
        std::vector<ai::Connection> conns = {
            ai::Connection{0, 50, 1.0f, true},  // input -> hidden
            ai::Connection{50, 100, 1.0f, true}, // hidden -> output
            ai::Connection{0, 100, 1.0f, true},  // input -> output, skipping the hidden node
        };
        ai::NeuralNetwork net(nodes, conns);
        const auto out = net.evaluate(makeObservation(0, 0.5f));
        const float hidden = std::tanh(0.5f);
        const float expected = std::tanh(hidden * 1.0f + 0.5f * 1.0f);
        assert(std::fabs(out[0] - expected) < kEps && "skip connection combined with hidden path result mismatch");
    }

    // 8: evaluation does not depend on node ID numerical order. Hidden node
    // ID (999) is numerically larger than the Output node ID (100) it feeds,
    // and larger than the Bias ID (9); topological order must still put the
    // hidden node before the output regardless.
    {
        std::vector<ai::Node> nodes = makeBaseNodes();
        nodes.push_back(ai::Node{999, ai::NodeType::Hidden});
        std::vector<ai::Connection> conns = {
            ai::Connection{0, 999, 1.0f, true},
            ai::Connection{999, 100, 1.0f, true},
        };
        ai::NeuralNetwork net(nodes, conns);
        const auto out = net.evaluate(makeObservation(0, 1.0f));
        const float expected = std::tanh(std::tanh(1.0f));
        assert(std::fabs(out[0] - expected) < kEps &&
               "evaluation must follow actual dependencies, not node ID order");
    }

    // 10: invalid source/destination node IDs are rejected.
    {
        assert(throwsInvalidArgument([]() {
            ai::NeuralNetwork net(makeBaseNodes(), {ai::Connection{0, 12345, 0.1f, true}});
        }) && "connection to an unknown node ID must be rejected");
        assert(throwsInvalidArgument([]() {
            ai::NeuralNetwork net(makeBaseNodes(), {ai::Connection{12345, 100, 0.1f, true}});
        }) && "connection from an unknown node ID must be rejected");
    }

    // 11: duplicate node IDs are rejected.
    {
        assert(throwsInvalidArgument([]() {
            std::vector<ai::Node> nodes = makeBaseNodes();
            nodes.push_back(ai::Node{0, ai::NodeType::Hidden}); // reuses input 0's ID
            ai::NeuralNetwork net(nodes, {});
        }) && "duplicate node IDs must be rejected");
    }

    // 12: duplicate directed connections are rejected.
    {
        assert(throwsInvalidArgument([]() {
            std::vector<ai::Connection> conns = {
                ai::Connection{0, 100, 0.1f, true},
                ai::Connection{0, 100, 0.2f, true},
            };
            ai::NeuralNetwork net(makeBaseNodes(), conns);
        }) && "duplicate (source, target) connections must be rejected");
    }

    // 13: a connection targeting Input is rejected.
    {
        assert(throwsInvalidArgument([]() {
            ai::NeuralNetwork net(makeBaseNodes(), {ai::Connection{100, 0, 0.1f, true}});
        }) && "a connection targeting an Input node must be rejected");
    }

    // 14: a connection targeting Bias is rejected.
    {
        assert(throwsInvalidArgument([]() {
            ai::NeuralNetwork net(makeBaseNodes(), {ai::Connection{0, 9, 0.1f, true}});
        }) && "a connection targeting the Bias node must be rejected");
    }

    // 15: a cyclic graph is rejected.
    {
        assert(throwsInvalidArgument([]() {
            std::vector<ai::Node> nodes = makeBaseNodes();
            nodes.push_back(ai::Node{50, ai::NodeType::Hidden});
            nodes.push_back(ai::Node{51, ai::NodeType::Hidden});
            std::vector<ai::Connection> conns = {
                ai::Connection{50, 51, 1.0f, true},
                ai::Connection{51, 50, 1.0f, true},
            };
            ai::NeuralNetwork net(nodes, conns);
        }) && "a cyclic graph must be rejected");
    }

    // 16: incorrect Input/Bias/Output counts are rejected.
    {
        assert(throwsInvalidArgument([]() {
            std::vector<ai::Node> nodes = makeBaseNodes();
            nodes.erase(nodes.begin()); // drops one Input, leaving 8
            ai::NeuralNetwork net(nodes, {});
        }) && "fewer than kInputCount Input nodes must be rejected");

        assert(throwsInvalidArgument([]() {
            std::vector<ai::Node> nodes = makeBaseNodes();
            nodes.erase(nodes.begin() + ai::NeuralNetwork::kInputCount); // drops the Bias node
            ai::NeuralNetwork net(nodes, {});
        }) && "a missing Bias node must be rejected");

        assert(throwsInvalidArgument([]() {
            std::vector<ai::Node> nodes = makeBaseNodes();
            nodes.pop_back(); // drops the throttle Output, leaving 1
            ai::NeuralNetwork net(nodes, {});
        }) && "fewer than kOutputCount Output nodes must be rejected");
    }

    // 17: output values are returned in deterministic steering/throttle order.
    {
        std::vector<ai::Connection> conns = {
            ai::Connection{0, 100, 1.0f, true},
            ai::Connection{0, 101, 2.0f, true},
        };
        ai::NeuralNetwork net(makeBaseNodes(), conns);
        const auto out = net.evaluate(makeObservation(0, 1.0f));
        assert(std::fabs(out[0] - std::tanh(1.0f)) < kEps && "output index 0 must be the steering (first Output) node");
        assert(std::fabs(out[1] - std::tanh(2.0f)) < kEps && "output index 1 must be the throttle (second Output) node");
    }

    TraceLog(LOG_INFO, "Neural network verification: all deterministic checks passed");
}

// One-shot, deterministic sanity check of the NEAT gene definitions
// (NodeGene, ConnectionGene), independent of Car/Track/keyboard/render
// timing. Runs once at startup. These are pure data structures -- no
// runtime network is built or evaluated here.
void verifyNeatGenes()
{
    using ai::neat::ConnectionGene;
    using ai::neat::NodeGene;
    using ai::neat::NodeType;

    auto throwsInvalidArgument = [](auto&& callable) -> bool
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
    };

    // 1: valid construction for every node type, with getters reporting back
    // exactly what was passed in.
    {
        const NodeGene input(0, NodeType::Input);
        const NodeGene bias(1, NodeType::Bias);
        const NodeGene hidden(2, NodeType::Hidden);
        const NodeGene output(3, NodeType::Output);
        assert(input.getId() == 0 && input.getType() == NodeType::Input && "Input node gene must round-trip");
        assert(bias.getId() == 1 && bias.getType() == NodeType::Bias && "Bias node gene must round-trip");
        assert(hidden.getId() == 2 && hidden.getType() == NodeType::Hidden && "Hidden node gene must round-trip");
        assert(output.getId() == 3 && output.getType() == NodeType::Output && "Output node gene must round-trip");
    }

    // 2: a negative node ID is rejected.
    {
        assert(throwsInvalidArgument([]() { NodeGene bad(-1, NodeType::Hidden); }) &&
               "a negative node ID must be rejected");
    }

    // 3: equality compares both ID and type; either differing breaks equality.
    {
        const NodeGene a(5, NodeType::Hidden);
        const NodeGene b(5, NodeType::Hidden);
        const NodeGene differentId(6, NodeType::Hidden);
        const NodeGene differentType(5, NodeType::Output);
        assert(a == b && "identical node genes must compare equal");
        assert(a != differentId && "node genes with different IDs must not compare equal");
        assert(a != differentType && "node genes with different types must not compare equal");
    }

    // 4: valid construction preserves source, target, weight, enabled flag,
    // and innovation number exactly.
    {
        const ConnectionGene c(0, 1, 0.75f, true, 3);
        assert(c.getSourceId() == 0 && c.getTargetId() == 1 && "source/target must round-trip");
        assert(c.getWeight() == 0.75f && "weight must be preserved exactly, not clamped or rounded");
        assert(c.isEnabled() && "enabled flag must round-trip");
        assert(c.getInnovationNumber() == 3 && "innovation number must round-trip");
    }

    // 5: a negative weight is preserved exactly (weights are never clamped).
    {
        const ConnectionGene c(0, 1, -2.5f, false, 0);
        assert(c.getWeight() == -2.5f && "negative weight must be preserved exactly");
        assert(!c.isEnabled() && "enabled=false at construction must be preserved");
    }

    // 6: enable()/disable() toggle the flag and nothing else.
    {
        ConnectionGene c(0, 1, 1.0f, false, 0);
        assert(!c.isEnabled() && "must start disabled");
        c.enable();
        assert(c.isEnabled() && "enable() must set the flag");
        c.disable();
        assert(!c.isEnabled() && "disable() must clear the flag");
        assert(c.getSourceId() == 0 && c.getTargetId() == 1 && c.getWeight() == 1.0f &&
               c.getInnovationNumber() == 0 && "enable()/disable() must not affect other fields");
    }

    // 7: negative source/target node IDs are rejected.
    {
        assert(throwsInvalidArgument([]() { ConnectionGene c(-1, 1, 0.0f, true, 0); }) &&
               "a negative source node ID must be rejected");
        assert(throwsInvalidArgument([]() { ConnectionGene c(0, -1, 0.0f, true, 0); }) &&
               "a negative target node ID must be rejected");
    }

    // 8: a negative innovation number is rejected.
    {
        assert(throwsInvalidArgument([]() { ConnectionGene c(0, 1, 0.0f, true, -1); }) &&
               "a negative innovation number must be rejected");
    }

    // 9: a self-connection (source == target) is rejected.
    {
        assert(throwsInvalidArgument([]() { ConnectionGene c(4, 4, 0.0f, true, 0); }) &&
               "a self-connection must be rejected");
    }

    TraceLog(LOG_INFO, "NEAT gene verification: all deterministic checks passed");
}

// One-shot, deterministic sanity check of ai::neat::Genome, independent of
// Car/Track/keyboard/render timing. Runs once at startup. Genome is a pure
// genetic container -- no evaluation, mutation, or crossover is exercised
// here.
void verifyGenome()
{
    using ai::neat::ConnectionGene;
    using ai::neat::Genome;
    using ai::neat::NodeGene;
    using ai::neat::NodeType;

    auto throwsInvalidArgument = [](auto&& callable) -> bool
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
    };

    // 1: a default-constructed genome is empty and already valid.
    {
        Genome genome;
        assert(genome.nodes().empty() && "a fresh genome must have no nodes");
        assert(genome.connections().empty() && "a fresh genome must have no connections");
        genome.validate(); // must not throw
    }

    // 2: addNode appends nodes and they show up in nodes().
    {
        Genome genome;
        genome.addNode(NodeGene{0, NodeType::Input});
        genome.addNode(NodeGene{1, NodeType::Bias});
        genome.addNode(NodeGene{2, NodeType::Output});
        assert(genome.nodes().size() == 3 && "addNode must append to nodes()");
        assert(genome.nodes()[0].getId() == 0 && genome.nodes()[1].getId() == 1 &&
               genome.nodes()[2].getId() == 2 && "nodes() must preserve insertion order");
    }

    // 3: adding a node with a duplicate ID is rejected, and does not modify the genome.
    {
        Genome genome;
        genome.addNode(NodeGene{0, NodeType::Input});
        assert(throwsInvalidArgument([&]() { genome.addNode(NodeGene{0, NodeType::Hidden}); }) &&
               "a duplicate node ID must be rejected");
        assert(genome.nodes().size() == 1 && "a rejected addNode must not modify the genome");
    }

    // 4: hasNode/findNode report existence and identity correctly, including for missing IDs.
    {
        Genome genome;
        genome.addNode(NodeGene{7, NodeType::Hidden});
        assert(genome.hasNode(7) && "hasNode must find an existing node ID");
        assert(!genome.hasNode(8) && "hasNode must not find a missing node ID");
        const NodeGene* found = genome.findNode(7);
        assert(found != nullptr && found->getId() == 7 && found->getType() == NodeType::Hidden &&
               "findNode must return the matching node");
        assert(genome.findNode(8) == nullptr && "findNode must return nullptr for a missing node ID");
    }

    // 5: a valid connection between two existing nodes is accepted.
    {
        Genome genome;
        genome.addNode(NodeGene{0, NodeType::Input});
        genome.addNode(NodeGene{1, NodeType::Output});
        genome.addConnection(ConnectionGene{0, 1, 0.5f, true, 0});
        assert(genome.connections().size() == 1 && "addConnection must append to connections()");
        assert(genome.connections()[0].getSourceId() == 0 && genome.connections()[0].getTargetId() == 1 &&
               "the stored connection must match what was added");
    }

    // 6: a connection with a missing source node is rejected.
    {
        Genome genome;
        genome.addNode(NodeGene{1, NodeType::Output});
        assert(throwsInvalidArgument([&]() { genome.addConnection(ConnectionGene{0, 1, 0.1f, true, 0}); }) &&
               "a connection with an unknown source node must be rejected");
        assert(genome.connections().empty() && "a rejected addConnection must not modify the genome");
    }

    // 7: a connection with a missing target node is rejected.
    {
        Genome genome;
        genome.addNode(NodeGene{0, NodeType::Input});
        assert(throwsInvalidArgument([&]() { genome.addConnection(ConnectionGene{0, 1, 0.1f, true, 0}); }) &&
               "a connection with an unknown target node must be rejected");
        assert(genome.connections().empty() && "a rejected addConnection must not modify the genome");
    }

    // 8: a duplicate directed connection is rejected, even with a different
    // weight/innovation number -- only (source, target) identity matters.
    {
        Genome genome;
        genome.addNode(NodeGene{0, NodeType::Input});
        genome.addNode(NodeGene{1, NodeType::Output});
        genome.addConnection(ConnectionGene{0, 1, 0.1f, true, 0});
        assert(throwsInvalidArgument([&]() { genome.addConnection(ConnectionGene{0, 1, 0.9f, false, 99}); }) &&
               "a duplicate (source, target) connection must be rejected regardless of weight/innovation");
        assert(genome.connections().size() == 1 && "a rejected duplicate connection must not modify the genome");
    }

    // 9: hasConnection/findConnection report existence and identity correctly,
    // including for missing (source, target) pairs, and the reverse direction
    // is treated as a distinct, absent connection.
    {
        Genome genome;
        genome.addNode(NodeGene{0, NodeType::Input});
        genome.addNode(NodeGene{1, NodeType::Output});
        genome.addConnection(ConnectionGene{0, 1, 0.42f, true, 5});

        assert(genome.hasConnection(0, 1) && "hasConnection must find an existing (source, target) pair");
        assert(!genome.hasConnection(1, 0) && "hasConnection must not treat the reverse direction as existing");

        const ConnectionGene* found = genome.findConnection(0, 1);
        assert(found != nullptr && found->getWeight() == 0.42f && found->getInnovationNumber() == 5 &&
               "findConnection must return the matching connection");
        assert(genome.findConnection(1, 0) == nullptr &&
               "findConnection must return nullptr for a missing (source, target) pair");
    }

    // 10: validate() succeeds (does not throw) on a genome built entirely
    // through the validated addNode/addConnection API.
    {
        Genome genome;
        genome.addNode(NodeGene{0, NodeType::Input});
        genome.addNode(NodeGene{1, NodeType::Bias});
        genome.addNode(NodeGene{2, NodeType::Output});
        genome.addConnection(ConnectionGene{0, 2, 1.0f, true, 0});
        genome.addConnection(ConnectionGene{1, 2, 1.0f, true, 1});
        genome.validate(); // must not throw
    }

    // 11: validate() rejects a genome with duplicate node IDs. Such a genome
    // cannot be built via addNode, so it is assembled through the raw bulk
    // constructor, which intentionally skips validation at construction time.
    {
        std::vector<NodeGene> nodes = {NodeGene{0, NodeType::Input}, NodeGene{0, NodeType::Hidden}};
        Genome genome(nodes, {});
        assert(throwsInvalidArgument([&]() { genome.validate(); }) &&
               "validate() must reject duplicate node IDs");
    }

    // 12: validate() rejects a genome with a connection referencing a
    // nonexistent node.
    {
        std::vector<NodeGene> nodes = {NodeGene{0, NodeType::Input}};
        std::vector<ConnectionGene> connections = {ConnectionGene{0, 99, 0.1f, true, 0}};
        Genome genome(nodes, connections);
        assert(throwsInvalidArgument([&]() { genome.validate(); }) &&
               "validate() must reject a connection with a nonexistent endpoint");
    }

    // 13: validate() rejects a genome with duplicate directed connections.
    {
        std::vector<NodeGene> nodes = {NodeGene{0, NodeType::Input}, NodeGene{1, NodeType::Output}};
        std::vector<ConnectionGene> connections = {
            ConnectionGene{0, 1, 0.1f, true, 0},
            ConnectionGene{0, 1, 0.2f, true, 1},
        };
        Genome genome(nodes, connections);
        assert(throwsInvalidArgument([&]() { genome.validate(); }) &&
               "validate() must reject duplicate directed connections");
    }

    TraceLog(LOG_INFO, "Genome verification: all deterministic checks passed");
}

namespace phenotype_verify
{

// Builds a Genome with exactly ai::NeuralNetwork::kInputCount Input nodes
// (IDs 0..8, one per Observation slot), one Bias node (ID 9), and
// ai::NeuralNetwork::kOutputCount Output nodes (ID 100 = steering, the
// lower ID; ID 101 = throttle, the higher ID). Nodes are deliberately added
// out of ID order -- Bias first, then the higher-ID Output before the
// lower-ID one, then Inputs in descending ID order -- so any test built on
// top of this proves buildPhenotype()'s slot ordering depends on node ID,
// never on Genome insertion order.
ai::neat::Genome makeBaseGenome()
{
    ai::neat::Genome genome;
    genome.addNode(ai::neat::NodeGene{9, ai::neat::NodeType::Bias});
    genome.addNode(ai::neat::NodeGene{101, ai::neat::NodeType::Output});
    genome.addNode(ai::neat::NodeGene{100, ai::neat::NodeType::Output});
    for (int i = ai::NeuralNetwork::kInputCount - 1; i >= 0; --i)
    {
        genome.addNode(ai::neat::NodeGene{i, ai::neat::NodeType::Input});
    }
    return genome;
}

ai::Observation makeObservation(int index, float value)
{
    ai::Observation obs;
    obs.values.fill(0.0f);
    if (index >= 0)
    {
        obs.values[index] = value;
    }
    return obs;
}

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

} // namespace phenotype_verify

// One-shot, deterministic sanity check of ai::neat::buildPhenotype, covering
// Genome -> NeuralNetwork conversion end to end. Independent of
// Car/Track/keyboard/render timing. Runs once at startup. No mutation,
// crossover, or evolutionary behavior is exercised here -- only phenotype
// construction.
void verifyPhenotypeBuilder()
{
    using namespace phenotype_verify;
    using ai::neat::buildPhenotype;
    using ai::neat::ConnectionGene;
    using ai::neat::Genome;
    using ai::neat::NodeGene;
    using ai::neat::NodeType;
    constexpr float kEps = 1e-4f;

    // 1 & 3: a minimal valid Genome (9 Input + 1 Bias + 2 Output NodeGenes)
    // builds successfully. This is only possible if Input/Bias/Output
    // NodeGene types were mapped to the matching runtime NodeType -- a
    // mismapping would make NeuralNetwork's own Input/Bias/Output count
    // checks fail.
    {
        ai::NeuralNetwork net = buildPhenotype(makeBaseGenome());
        const auto out = net.evaluate(makeObservation(-1, 0.0f));
        assert(out[0] == 0.0f && out[1] == 0.0f && "a fully disconnected phenotype must evaluate to exactly 0");
    }

    // 2 & 8: node IDs (and the source/target IDs connections reference) are
    // preserved exactly, including when they are large and non-contiguous.
    // If the builder silently renumbered nodes without updating connection
    // endpoints to match, this construction would fail with unknown-node
    // errors; if it evaluated the wrong node, the arithmetic below would not
    // match.
    {
        Genome genome;
        for (int i = 0; i < ai::NeuralNetwork::kInputCount; ++i)
        {
            genome.addNode(NodeGene{1000 + i, NodeType::Input});
        }
        genome.addNode(NodeGene{2000, NodeType::Bias});
        genome.addNode(NodeGene{3000, NodeType::Output});
        genome.addNode(NodeGene{3001, NodeType::Output});
        genome.addConnection(ConnectionGene{1000, 3000, 0.5f, true, 0});

        ai::NeuralNetwork net = buildPhenotype(genome);
        const auto out = net.evaluate(makeObservation(0, 1.0f));
        assert(std::fabs(out[0] - std::tanh(0.5f)) < kEps &&
               "non-contiguous node IDs must be preserved through phenotype construction");
    }

    // 4: input slot ordering follows ascending node ID, independent of
    // Genome insertion order (makeBaseGenome adds Inputs in descending ID
    // order).
    {
        Genome genome = makeBaseGenome();
        genome.addConnection(ConnectionGene{0, 100, 1.0f, true, 0});
        genome.addConnection(ConnectionGene{ai::NeuralNetwork::kInputCount - 1, 101, 1.0f, true, 1});
        ai::NeuralNetwork net = buildPhenotype(genome);

        const auto outLow = net.evaluate(makeObservation(0, 1.0f));
        assert(std::fabs(outLow[0] - std::tanh(1.0f)) < kEps &&
               "Observation slot 0 must drive the lowest-ID Input node");
        assert(outLow[1] == 0.0f && "Observation slot 0 must not affect the highest-ID Input node");

        const auto outHigh = net.evaluate(makeObservation(ai::NeuralNetwork::kInputCount - 1, 1.0f));
        assert(outHigh[0] == 0.0f && "the last Observation slot must not affect the lowest-ID Input node");
        assert(std::fabs(outHigh[1] - std::tanh(1.0f)) < kEps &&
               "the last Observation slot must drive the highest-ID Input node");
    }

    // 5: output slot ordering follows ascending node ID, independent of
    // Genome insertion order (makeBaseGenome adds Output 101 before 100).
    {
        Genome genome = makeBaseGenome();
        genome.addConnection(ConnectionGene{0, 100, 1.0f, true, 0});
        genome.addConnection(ConnectionGene{0, 101, 2.0f, true, 1});
        ai::NeuralNetwork net = buildPhenotype(genome);

        const auto out = net.evaluate(makeObservation(0, 1.0f));
        assert(std::fabs(out[0] - std::tanh(1.0f)) < kEps && "output slot 0 must be the lower-ID Output node (100)");
        assert(std::fabs(out[1] - std::tanh(2.0f)) < kEps && "output slot 1 must be the higher-ID Output node (101)");
    }

    // 6: Bias maps correctly (always contributes 1.0) and stays internal --
    // it is not one of the kInputCount external Observation slots.
    {
        Genome genome = makeBaseGenome();
        genome.addConnection(ConnectionGene{9, 100, 0.7f, true, 0});
        ai::NeuralNetwork net = buildPhenotype(genome);
        const auto out = net.evaluate(makeObservation(-1, 0.0f));
        assert(std::fabs(out[0] - std::tanh(0.7f)) < kEps &&
               "Bias->Output must equal tanh(1.0 * weight) even with an all-zero Observation");
    }

    // 7 & 14: Hidden nodes map correctly; Input -> Hidden -> Output
    // evaluates to the mathematically expected result.
    {
        Genome genome = makeBaseGenome();
        genome.addNode(NodeGene{50, NodeType::Hidden});
        genome.addConnection(ConnectionGene{0, 50, 0.5f, true, 0});
        genome.addConnection(ConnectionGene{50, 100, 2.0f, true, 1});
        ai::NeuralNetwork net = buildPhenotype(genome);

        const auto out = net.evaluate(makeObservation(0, 1.0f));
        const float hidden = std::tanh(1.0f * 0.5f);
        const float expected = std::tanh(hidden * 2.0f);
        assert(std::fabs(out[0] - expected) < kEps && "Input->Hidden->Output result mismatch");
    }

    // 9, 10 & 11: weights and the enabled flag are preserved exactly, and a
    // disabled connection contributes nothing to evaluation.
    {
        Genome genome = makeBaseGenome();
        genome.addNode(NodeGene{60, NodeType::Hidden});
        genome.addConnection(ConnectionGene{0, 100, 0.37f, true, 0});   // enabled, contributes
        genome.addConnection(ConnectionGene{0, 60, -1.25f, false, 1}); // disabled, must not contribute
        genome.addConnection(ConnectionGene{60, 100, 4.0f, true, 2});  // would matter if 0->60 were active
        ai::NeuralNetwork net = buildPhenotype(genome);

        const auto out = net.evaluate(makeObservation(0, 1.0f));
        assert(std::fabs(out[0] - std::tanh(0.37f)) < kEps &&
               "a disabled connection must not contribute to evaluation, and enabled weight must be exact");
    }

    // 12: direct Input -> Output phenotype evaluates correctly.
    {
        Genome genome = makeBaseGenome();
        genome.addConnection(ConnectionGene{0, 100, 0.5f, true, 0});
        ai::NeuralNetwork net = buildPhenotype(genome);
        const auto out = net.evaluate(makeObservation(0, 1.0f));
        assert(std::fabs(out[0] - std::tanh(0.5f)) < kEps && "Input->Output must equal tanh(input * weight)");
    }

    // 13: Bias -> Output phenotype evaluates correctly (duplicate of #6's
    // arithmetic, kept as its own case per the required verification list).
    {
        Genome genome = makeBaseGenome();
        genome.addConnection(ConnectionGene{9, 100, 1.1f, true, 0});
        ai::NeuralNetwork net = buildPhenotype(genome);
        const auto out = net.evaluate(makeObservation(-1, 0.0f));
        assert(std::fabs(out[0] - std::tanh(1.1f)) < kEps && "Bias->Output must equal tanh(1.0 * weight)");
    }

    // 15: a deeper feed-forward DAG (Input -> HiddenA -> HiddenB -> Output)
    // evaluates correctly.
    {
        Genome genome = makeBaseGenome();
        genome.addNode(NodeGene{50, NodeType::Hidden});
        genome.addNode(NodeGene{51, NodeType::Hidden});
        genome.addConnection(ConnectionGene{0, 50, 1.0f, true, 0});
        genome.addConnection(ConnectionGene{50, 51, 1.0f, true, 1});
        genome.addConnection(ConnectionGene{51, 100, 1.0f, true, 2});
        ai::NeuralNetwork net = buildPhenotype(genome);

        const auto out = net.evaluate(makeObservation(0, 0.5f));
        const float a = std::tanh(0.5f);
        const float b = std::tanh(a);
        const float expected = std::tanh(b);
        assert(std::fabs(out[0] - expected) < kEps && "deep Input->HiddenA->HiddenB->Output result mismatch");
    }

    // 16: a skip connection alongside a hidden path evaluates correctly.
    {
        Genome genome = makeBaseGenome();
        genome.addNode(NodeGene{50, NodeType::Hidden});
        genome.addConnection(ConnectionGene{0, 50, 1.0f, true, 0});   // input -> hidden
        genome.addConnection(ConnectionGene{50, 100, 1.0f, true, 1}); // hidden -> output
        genome.addConnection(ConnectionGene{0, 100, 1.0f, true, 2});  // input -> output, skipping hidden
        ai::NeuralNetwork net = buildPhenotype(genome);

        const auto out = net.evaluate(makeObservation(0, 0.5f));
        const float hidden = std::tanh(0.5f);
        const float expected = std::tanh(hidden * 1.0f + 0.5f * 1.0f);
        assert(std::fabs(out[0] - expected) < kEps && "skip connection combined with hidden path result mismatch");
    }

    // 17: an invalid Genome (duplicate node IDs, only reachable via the raw
    // bulk constructor) fails because buildPhenotype calls genome.validate()
    // before ever touching NeuralNetwork.
    {
        std::vector<NodeGene> nodes = {NodeGene{0, NodeType::Input}, NodeGene{0, NodeType::Hidden}};
        Genome invalidGenome(nodes, {});
        assert(throwsInvalidArgument([&]() { buildPhenotype(invalidGenome); }) &&
               "a Genome that fails validate() must be rejected by buildPhenotype");
    }

    // 18: a Genome whose enabled connections form a cycle is structurally
    // valid at the Genome level (validate() does not check for cycles) but
    // must fail phenotype construction because NeuralNetwork rejects it.
    {
        Genome genome = makeBaseGenome();
        genome.addNode(NodeGene{50, NodeType::Hidden});
        genome.addNode(NodeGene{51, NodeType::Hidden});
        genome.addConnection(ConnectionGene{50, 51, 1.0f, true, 0});
        genome.addConnection(ConnectionGene{51, 50, 1.0f, true, 1});
        genome.validate(); // must not throw -- Genome has no cycle check
        assert(throwsInvalidArgument([&]() { buildPhenotype(genome); }) &&
               "a cyclic enabled Genome must be rejected by the feed-forward-only NeuralNetwork");
    }

    // 19: buildPhenotype does not alter the source Genome.
    {
        Genome genome = makeBaseGenome();
        genome.addNode(NodeGene{50, NodeType::Hidden});
        genome.addConnection(ConnectionGene{0, 50, 0.3f, true, 0});
        genome.addConnection(ConnectionGene{50, 100, 0.4f, true, 1});

        const std::vector<NodeGene> nodesBefore = genome.nodes();
        const std::size_t connectionCountBefore = genome.connections().size();

        ai::NeuralNetwork net = buildPhenotype(genome);
        (void)net;

        assert(genome.nodes().size() == nodesBefore.size() && "buildPhenotype must not add or remove genome nodes");
        for (std::size_t i = 0; i < nodesBefore.size(); ++i)
        {
            assert(genome.nodes()[i] == nodesBefore[i] && "buildPhenotype must not modify existing genome nodes");
        }
        assert(genome.connections().size() == connectionCountBefore &&
               "buildPhenotype must not add or remove genome connections");
        assert(genome.hasConnection(0, 50) && genome.hasConnection(50, 100) &&
               "buildPhenotype must leave genome connections intact");
    }

    // 20: phenotype evaluation is deterministic across repeated builds from
    // the same Genome.
    {
        Genome genome = makeBaseGenome();
        genome.addNode(NodeGene{50, NodeType::Hidden});
        genome.addConnection(ConnectionGene{0, 50, 0.6f, true, 0});
        genome.addConnection(ConnectionGene{50, 100, -0.9f, true, 1});
        genome.addConnection(ConnectionGene{9, 101, 0.2f, true, 2});

        ai::NeuralNetwork netA = buildPhenotype(genome);
        ai::NeuralNetwork netB = buildPhenotype(genome);

        const auto obs = makeObservation(0, 0.8f);
        const auto outA = netA.evaluate(obs);
        const auto outB = netB.evaluate(obs);
        assert(outA[0] == outB[0] && outA[1] == outB[1] &&
               "repeated builds from the same Genome must evaluate identically");
    }

    TraceLog(LOG_INFO, "Phenotype builder verification: all deterministic checks passed");
}

namespace genome_mutator_verify
{

// A small, fixed test genome: 9 Input + 1 Bias + 2 Output nodes (matching
// ai::NeuralNetwork::kInputCount/kOutputCount, so it can also be used to
// build a phenotype), plus four connections with known weights -- one of
// them disabled, specifically so mutation of disabled genes can be checked.
ai::neat::Genome makeTestGenome()
{
    using ai::neat::ConnectionGene;
    using ai::neat::Genome;
    using ai::neat::NodeGene;
    using ai::neat::NodeType;

    Genome genome;
    for (int i = 0; i < ai::NeuralNetwork::kInputCount; ++i)
    {
        genome.addNode(NodeGene{i, NodeType::Input});
    }
    genome.addNode(NodeGene{9, NodeType::Bias});
    genome.addNode(NodeGene{100, NodeType::Output});
    genome.addNode(NodeGene{101, NodeType::Output});

    genome.addConnection(ConnectionGene{0, 100, 0.5f, true, 0});
    genome.addConnection(ConnectionGene{1, 100, -0.3f, true, 1});
    genome.addConnection(ConnectionGene{9, 101, 0.2f, false, 2}); // disabled on purpose
    genome.addConnection(ConnectionGene{2, 101, 0.9f, true, 3});
    return genome;
}

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

// Genome with only two Output nodes: Output is never a valid add-connection
// source, so no source candidate can ever exist.
ai::neat::Genome makeOnlyOutputsGenome()
{
    using ai::neat::Genome;
    using ai::neat::NodeGene;
    using ai::neat::NodeType;

    Genome genome;
    genome.addNode(NodeGene{0, NodeType::Output});
    genome.addNode(NodeGene{1, NodeType::Output});
    return genome;
}

// Genome with only two Input nodes: Input is never a valid add-connection
// target, so no target candidate can ever exist.
ai::neat::Genome makeOnlyInputsGenome()
{
    using ai::neat::Genome;
    using ai::neat::NodeGene;
    using ai::neat::NodeType;

    Genome genome;
    genome.addNode(NodeGene{0, NodeType::Input});
    genome.addNode(NodeGene{1, NodeType::Input});
    return genome;
}

// Genome with only two Bias nodes: like makeOnlyInputsGenome(), Bias is
// never a valid target, so no target candidate can ever exist.
ai::neat::Genome makeOnlyBiasGenome()
{
    using ai::neat::Genome;
    using ai::neat::NodeGene;
    using ai::neat::NodeType;

    Genome genome;
    genome.addNode(NodeGene{0, NodeType::Bias});
    genome.addNode(NodeGene{1, NodeType::Bias});
    return genome;
}

// The only structurally possible pair is Input(0) -> Output(1): proves
// Input can be a source and Output can be a target.
ai::neat::Genome makeInputToOutputGenome()
{
    using ai::neat::Genome;
    using ai::neat::NodeGene;
    using ai::neat::NodeType;

    Genome genome;
    genome.addNode(NodeGene{0, NodeType::Input});
    genome.addNode(NodeGene{1, NodeType::Output});
    return genome;
}

// The only structurally possible pair is Bias(0) -> Output(1): proves Bias
// can be a source.
ai::neat::Genome makeBiasToOutputGenome()
{
    using ai::neat::Genome;
    using ai::neat::NodeGene;
    using ai::neat::NodeType;

    Genome genome;
    genome.addNode(NodeGene{0, NodeType::Bias});
    genome.addNode(NodeGene{1, NodeType::Output});
    return genome;
}

// The only structurally possible pair is Hidden(0) -> Hidden(1): proves
// Hidden can be both a source and a target.
ai::neat::Genome makeHiddenToHiddenGenome()
{
    using ai::neat::Genome;
    using ai::neat::NodeGene;
    using ai::neat::NodeType;

    Genome genome;
    genome.addNode(NodeGene{0, NodeType::Hidden});
    genome.addNode(NodeGene{1, NodeType::Hidden});
    return genome;
}

// A single Hidden node: the only "candidate" pair is a self-connection,
// which must never be added.
ai::neat::Genome makeSingleHiddenGenome()
{
    using ai::neat::Genome;
    using ai::neat::NodeGene;
    using ai::neat::NodeType;

    Genome genome;
    genome.addNode(NodeGene{0, NodeType::Hidden});
    return genome;
}

// Genome with exactly one structurally possible pair, already connected:
// proves a duplicate directed connection (enabled or disabled) is never
// re-added.
ai::neat::Genome makeAlreadyConnectedGenome(bool enabled)
{
    using ai::neat::ConnectionGene;
    using ai::neat::Genome;
    using ai::neat::NodeGene;
    using ai::neat::NodeType;

    Genome genome;
    genome.addNode(NodeGene{0, NodeType::Input});
    genome.addNode(NodeGene{1, NodeType::Output});
    genome.addConnection(ConnectionGene{0, 1, 0.1f, enabled, 0});
    return genome;
}

// Two Hidden nodes with a deliberately "descending" ID edge -- 10 -> 3, from
// the numerically larger ID to the smaller one -- so a check that assumed
// ascending IDs meant "topologically later" would get this wrong. With only
// two nodes, exactly two directed pairs are structurally possible: the
// existing 10 -> 3, and its reverse, 3 -> 10.
//
// When existingEdgeEnabled is true, 10 -> 3 blocks 3 -> 10 twice over: as an
// existing (source, target) pair in the reverse direction it is unrelated
// to duplication, but 3 -> 10 would close a direct 2-cycle (10 already has
// an enabled path to 3), so it must be rejected -- leaving no valid pair at
// all, since 10 -> 3 itself is already a duplicate.
//
// When existingEdgeEnabled is false, 10 -> 3 no longer contributes to the
// *enabled* graph, so 3 -> 10 is no longer a cycle and must be accepted --
// proving disabled edges do not participate in cycle detection.
ai::neat::Genome makeCycleGenome(bool existingEdgeEnabled)
{
    using ai::neat::ConnectionGene;
    using ai::neat::Genome;
    using ai::neat::NodeGene;
    using ai::neat::NodeType;

    Genome genome;
    genome.addNode(NodeGene{10, NodeType::Hidden});
    genome.addNode(NodeGene{3, NodeType::Hidden});
    genome.addConnection(ConnectionGene{10, 3, 0.1f, existingEdgeEnabled, 0});
    return genome;
}

// Nodes 0:Input, 1:Bias, 2:Hidden, 3:Output with every valid pair already
// connected except Bias(1) -> Output(3), out of the 6 (source, target)
// combinations reachable by candidate selection (3 sources x 2 targets).
// Used to exercise the deterministic exhaustive fallback: with very few
// random attempts, it is likely that none of them land on the single
// remaining valid pair, so the fallback scan has to find it.
ai::neat::Genome makeSparseValidPairGenome()
{
    using ai::neat::ConnectionGene;
    using ai::neat::Genome;
    using ai::neat::NodeGene;
    using ai::neat::NodeType;

    Genome genome;
    genome.addNode(NodeGene{0, NodeType::Input});
    genome.addNode(NodeGene{1, NodeType::Bias});
    genome.addNode(NodeGene{2, NodeType::Hidden});
    genome.addNode(NodeGene{3, NodeType::Output});
    genome.addConnection(ConnectionGene{0, 2, 0.1f, true, 0});
    genome.addConnection(ConnectionGene{0, 3, 0.1f, true, 1});
    genome.addConnection(ConnectionGene{1, 2, 0.1f, true, 2});
    genome.addConnection(ConnectionGene{2, 3, 0.1f, true, 3});
    return genome;
}

// Same node set as makeSparseValidPairGenome(), but with the fifth (and
// last remaining) valid pair, Bias(1) -> Output(3), also already connected
// -- every valid feed-forward pair now exists, so the genome is fully
// saturated.
ai::neat::Genome makeSaturatedGenome()
{
    using ai::neat::ConnectionGene;
    using ai::neat::Genome;
    using ai::neat::NodeGene;
    using ai::neat::NodeType;

    Genome genome;
    genome.addNode(NodeGene{0, NodeType::Input});
    genome.addNode(NodeGene{1, NodeType::Bias});
    genome.addNode(NodeGene{2, NodeType::Hidden});
    genome.addNode(NodeGene{3, NodeType::Output});
    genome.addConnection(ConnectionGene{0, 2, 0.1f, true, 0});
    genome.addConnection(ConnectionGene{0, 3, 0.1f, true, 1});
    genome.addConnection(ConnectionGene{1, 2, 0.1f, true, 2});
    genome.addConnection(ConnectionGene{1, 3, 0.1f, true, 3});
    genome.addConnection(ConnectionGene{2, 3, 0.1f, true, 4});
    return genome;
}

} // namespace genome_mutator_verify

// One-shot, deterministic sanity check of ai::neat::GenomeMutator, covering
// connection-weight mutation end to end. Independent of Car/Track/AI/
// keyboard/render timing. Runs once at startup. No structural mutation,
// InnovationTracker, crossover, species, or population logic exists to
// verify here -- only weight mutation, per Stage 9A's scope.
void verifyGenomeMutator()
{
    using namespace genome_mutator_verify;
    using ai::neat::ConnectionGene;
    using ai::neat::Genome;
    using ai::neat::GenomeMutator;
    using ai::neat::MutationConfig;
    using ai::neat::NodeGene;
    constexpr float kEps = 1e-5f;

    // 1: zero mutation probability changes no weights.
    {
        Genome genome = makeTestGenome();
        const std::vector<ConnectionGene> before = genome.connections();
        MutationConfig config;
        config.weightMutationProbability = 0.0f;
        GenomeMutator mutator(12345u);
        mutator.mutateWeights(genome, config);
        for (std::size_t i = 0; i < before.size(); ++i)
        {
            assert(genome.connections()[i].getWeight() == before[i].getWeight() &&
                   "zero mutation probability must leave every weight unchanged");
        }
    }

    // 2: mutation probability 1 selects every connection -- every weight
    // must change (a perturb/replace mix drawing continuous random values
    // colliding exactly with the original weight is astronomically
    // unlikely, so strict inequality is a safe deterministic check here).
    {
        Genome genome = makeTestGenome();
        const std::vector<ConnectionGene> before = genome.connections();
        MutationConfig config;
        config.weightMutationProbability = 1.0f;
        GenomeMutator mutator(1u);
        mutator.mutateWeights(genome, config);
        for (std::size_t i = 0; i < before.size(); ++i)
        {
            assert(genome.connections()[i].getWeight() != before[i].getWeight() &&
                   "mutation probability 1 must select and change every connection's weight");
        }
    }

    // 3: perturb probability 1 performs perturbation only -- every changed
    // weight stays within oldWeight +/- perturbStrength.
    {
        Genome genome = makeTestGenome();
        const std::vector<ConnectionGene> before = genome.connections();
        MutationConfig config;
        config.weightMutationProbability = 1.0f;
        config.weightPerturbProbability = 1.0f;
        config.perturbStrength = 0.3f;
        GenomeMutator mutator(7u);
        mutator.mutateWeights(genome, config);
        for (std::size_t i = 0; i < before.size(); ++i)
        {
            const float delta = genome.connections()[i].getWeight() - before[i].getWeight();
            assert(std::fabs(delta) <= config.perturbStrength + kEps &&
                   "perturb probability 1 must only ever perturb, never replace outright");
        }
    }

    // 4: perturb probability 0 performs replacement only -- every changed
    // weight lands within [replacementWeightMin, replacementWeightMax].
    {
        Genome genome = makeTestGenome();
        MutationConfig config;
        config.weightMutationProbability = 1.0f;
        config.weightPerturbProbability = 0.0f;
        config.replacementWeightMin = -2.0f;
        config.replacementWeightMax = 2.0f;
        GenomeMutator mutator(9u);
        mutator.mutateWeights(genome, config);
        for (const ConnectionGene& c : genome.connections())
        {
            assert(c.getWeight() >= config.replacementWeightMin && c.getWeight() <= config.replacementWeightMax &&
                   "perturb probability 0 must always replace within the configured range");
        }
    }

    // 5: zero perturb strength preserves the selected weight exactly (a
    // perturbation by +/-0 is a no-op).
    {
        Genome genome = makeTestGenome();
        const std::vector<ConnectionGene> before = genome.connections();
        MutationConfig config;
        config.weightMutationProbability = 1.0f;
        config.weightPerturbProbability = 1.0f;
        config.perturbStrength = 0.0f;
        GenomeMutator mutator(3u);
        mutator.mutateWeights(genome, config);
        for (std::size_t i = 0; i < before.size(); ++i)
        {
            assert(genome.connections()[i].getWeight() == before[i].getWeight() &&
                   "zero perturb strength must preserve the selected weight exactly");
        }
    }

    // 6: replacement weights remain within the configured range, checked
    // across many seeds for confidence.
    {
        MutationConfig config;
        config.weightMutationProbability = 1.0f;
        config.weightPerturbProbability = 0.0f;
        config.replacementWeightMin = -0.75f;
        config.replacementWeightMax = 1.25f;
        for (std::uint32_t seed = 0; seed < 20; ++seed)
        {
            Genome genome = makeTestGenome();
            GenomeMutator mutator(seed);
            mutator.mutateWeights(genome, config);
            for (const ConnectionGene& c : genome.connections())
            {
                assert(c.getWeight() >= config.replacementWeightMin && c.getWeight() <= config.replacementWeightMax &&
                       "replacement weights must always stay within the configured range");
            }
        }
    }

    // 7: a disabled connection's weight is also eligible for mutation.
    {
        Genome genome = makeTestGenome();
        const ConnectionGene* disabledBefore = genome.findConnection(9, 101);
        assert(disabledBefore != nullptr && !disabledBefore->isEnabled() &&
               "test genome must contain a disabled connection");
        const float disabledWeightBefore = disabledBefore->getWeight();

        MutationConfig config;
        config.weightMutationProbability = 1.0f;
        GenomeMutator mutator(11u);
        mutator.mutateWeights(genome, config);

        const ConnectionGene* disabledAfter = genome.findConnection(9, 101);
        assert(disabledAfter->getWeight() != disabledWeightBefore &&
               "a disabled connection's weight must still be eligible for mutation");
        assert(!disabledAfter->isEnabled() && "mutating weights must not change the enabled state");
    }

    // 8, 9, 10, 11 & 12: enabled states, node genes, connection endpoints,
    // innovation numbers, and node/connection counts all remain unchanged.
    {
        Genome genome = makeTestGenome();
        const std::vector<NodeGene> nodesBefore = genome.nodes();
        const std::vector<ConnectionGene> connectionsBefore = genome.connections();

        MutationConfig config;
        config.weightMutationProbability = 1.0f;
        GenomeMutator mutator(21u);
        mutator.mutateWeights(genome, config);

        assert(genome.nodes().size() == nodesBefore.size() && "mutation must not change node count");
        assert(genome.connections().size() == connectionsBefore.size() && "mutation must not change connection count");

        for (std::size_t i = 0; i < nodesBefore.size(); ++i)
        {
            assert(genome.nodes()[i] == nodesBefore[i] && "mutation must not change any node gene");
        }
        for (std::size_t i = 0; i < connectionsBefore.size(); ++i)
        {
            const ConnectionGene& before = connectionsBefore[i];
            const ConnectionGene& after = genome.connections()[i];
            assert(after.getSourceId() == before.getSourceId() && after.getTargetId() == before.getTargetId() &&
                   "mutation must not change connection endpoints");
            assert(after.getInnovationNumber() == before.getInnovationNumber() &&
                   "mutation must not change innovation numbers");
            assert(after.isEnabled() == before.isEnabled() && "mutation must not change enabled state");
        }
    }

    // 13: Genome::validate() succeeds after mutation.
    {
        Genome genome = makeTestGenome();
        MutationConfig config;
        config.weightMutationProbability = 1.0f;
        GenomeMutator mutator(33u);
        mutator.mutateWeights(genome, config);
        genome.validate(); // must not throw
    }

    // 14: identical seed + identical genome produces identical results.
    {
        Genome genomeA = makeTestGenome();
        Genome genomeB = makeTestGenome();
        MutationConfig config;
        GenomeMutator mutatorA(555u);
        GenomeMutator mutatorB(555u);
        mutatorA.mutateWeights(genomeA, config);
        mutatorB.mutateWeights(genomeB, config);
        for (std::size_t i = 0; i < genomeA.connections().size(); ++i)
        {
            assert(genomeA.connections()[i].getWeight() == genomeB.connections()[i].getWeight() &&
                   "identical seed and genome must produce identical mutation results");
        }
    }

    // 15: different seeds can produce different results.
    {
        Genome genomeA = makeTestGenome();
        Genome genomeB = makeTestGenome();
        MutationConfig config;
        GenomeMutator mutatorA(1u);
        GenomeMutator mutatorB(2u);
        mutatorA.mutateWeights(genomeA, config);
        mutatorB.mutateWeights(genomeB, config);

        bool anyDifferent = false;
        for (std::size_t i = 0; i < genomeA.connections().size(); ++i)
        {
            if (genomeA.connections()[i].getWeight() != genomeB.connections()[i].getWeight())
            {
                anyDifferent = true;
                break;
            }
        }
        assert(anyDifferent && "different seeds must be capable of producing different results");
    }

    // 16: repeated mutations advance the owned RNG state -- a second
    // mutation from the same mutator must continue from where the first
    // left off, not repeat the same draws.
    {
        Genome genomeA = makeTestGenome();
        Genome genomeB = makeTestGenome();
        MutationConfig config;
        GenomeMutator mutator(77u);
        mutator.mutateWeights(genomeA, config);
        mutator.mutateWeights(genomeB, config);

        bool anyDifferent = false;
        for (std::size_t i = 0; i < genomeA.connections().size(); ++i)
        {
            if (genomeA.connections()[i].getWeight() != genomeB.connections()[i].getWeight())
            {
                anyDifferent = true;
                break;
            }
        }
        assert(anyDifferent &&
               "repeated mutations from the same mutator must advance its RNG state, not repeat the same draws");
    }

    // 17: invalid probabilities are rejected.
    {
        Genome genome = makeTestGenome();
        GenomeMutator mutator(1u);

        MutationConfig tooHigh;
        tooHigh.weightMutationProbability = 1.5f;
        assert(throwsInvalidArgument([&]() { mutator.mutateWeights(genome, tooHigh); }) &&
               "a weightMutationProbability outside [0,1] must be rejected");

        MutationConfig tooLow;
        tooLow.weightPerturbProbability = -0.1f;
        assert(throwsInvalidArgument([&]() { mutator.mutateWeights(genome, tooLow); }) &&
               "a weightPerturbProbability outside [0,1] must be rejected");
    }

    // 18: negative perturb strength is rejected.
    {
        Genome genome = makeTestGenome();
        GenomeMutator mutator(1u);
        MutationConfig config;
        config.perturbStrength = -0.01f;
        assert(throwsInvalidArgument([&]() { mutator.mutateWeights(genome, config); }) &&
               "a negative perturbStrength must be rejected");
    }

    // 19: inverted replacement range is rejected.
    {
        Genome genome = makeTestGenome();
        GenomeMutator mutator(1u);
        MutationConfig config;
        config.replacementWeightMin = 1.0f;
        config.replacementWeightMax = -1.0f;
        assert(throwsInvalidArgument([&]() { mutator.mutateWeights(genome, config); }) &&
               "an inverted replacement range must be rejected");
    }

    // 20: non-finite values are rejected.
    {
        Genome genome = makeTestGenome();
        GenomeMutator mutator(1u);

        MutationConfig nanConfig;
        nanConfig.weightMutationProbability = std::numeric_limits<float>::quiet_NaN();
        assert(throwsInvalidArgument([&]() { mutator.mutateWeights(genome, nanConfig); }) &&
               "a NaN probability must be rejected");

        MutationConfig infConfig;
        infConfig.perturbStrength = std::numeric_limits<float>::infinity();
        assert(throwsInvalidArgument([&]() { mutator.mutateWeights(genome, infConfig); }) &&
               "an infinite perturbStrength must be rejected");
    }

    // 21: empty Genome mutation succeeds without error.
    {
        Genome empty;
        MutationConfig config;
        GenomeMutator mutator(1u);
        mutator.mutateWeights(empty, config); // must not throw
        assert(empty.connections().empty() && empty.nodes().empty() && "an empty genome must remain empty after mutation");
    }

    // 22 & 23: phenotype can still be built after mutation, and changed
    // weights produce a deterministic changed phenotype output.
    {
        Genome genome = makeTestGenome();
        ai::NeuralNetwork before = ai::neat::buildPhenotype(genome);

        MutationConfig config;
        config.weightMutationProbability = 1.0f;
        GenomeMutator mutator(999u);
        mutator.mutateWeights(genome, config);

        ai::NeuralNetwork after = ai::neat::buildPhenotype(genome); // must not throw

        ai::Observation obs;
        obs.values.fill(0.5f);
        const auto outBefore = before.evaluate(obs);
        const auto outAfter = after.evaluate(obs);
        assert((outBefore[0] != outAfter[0] || outBefore[1] != outAfter[1]) &&
               "mutated weights must produce a changed phenotype output");

        // Determinism: repeating the exact same mutation from the same seed
        // on a fresh identical genome must reproduce the same phenotype
        // output.
        Genome genomeRepeat = makeTestGenome();
        GenomeMutator mutatorRepeat(999u);
        mutatorRepeat.mutateWeights(genomeRepeat, config);
        ai::NeuralNetwork afterRepeat = ai::neat::buildPhenotype(genomeRepeat);
        const auto outAfterRepeat = afterRepeat.evaluate(obs);
        assert(outAfter[0] == outAfterRepeat[0] && outAfter[1] == outAfterRepeat[1] &&
               "identical seed and genome must produce a deterministic phenotype output after mutation");
    }

    // 24: no structural genes are added or removed, re-confirmed across a
    // batch of different seeds.
    {
        for (std::uint32_t seed = 100; seed < 110; ++seed)
        {
            Genome genome = makeTestGenome();
            const std::size_t nodeCountBefore = genome.nodes().size();
            const std::size_t connectionCountBefore = genome.connections().size();
            MutationConfig config;
            config.weightMutationProbability = 1.0f;
            GenomeMutator mutator(seed);
            mutator.mutateWeights(genome, config);
            assert(genome.nodes().size() == nodeCountBefore && genome.connections().size() == connectionCountBefore &&
                   "no structural genes may be added or removed by weight mutation");
        }
    }

    // 25: all previous verification suites still pass -- enforced by main()
    // continuing to call every earlier verify*() function unchanged.

    TraceLog(LOG_INFO, "Genome mutator verification: all deterministic checks passed");
}

namespace innovation_tracker_verify
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

template <typename Callable>
bool throwsOverflowError(Callable&& callable)
{
    try
    {
        callable();
    }
    catch (const std::overflow_error&)
    {
        return true;
    }
    return false;
}

} // namespace innovation_tracker_verify

// One-shot, deterministic sanity check of ai::neat::InnovationTracker,
// independent of Car/Track/AI/Genome/keyboard/render timing. Runs once at
// startup. InnovationTracker's API takes no Genome parameter anywhere, so
// every check below -- construction and every operation on it -- needed no
// Genome and could not have mutated one even in principle; that guarantee
// is enforced by the type signatures themselves, not by a runtime check.
// No structural mutation is implemented or exercised here -- only the
// historical-marking bookkeeping itself, per Stage 9B's scope.
void verifyInnovationTracker()
{
    using namespace innovation_tracker_verify;
    using ai::neat::InnovationNumber;
    using ai::neat::InnovationTracker;
    using ai::neat::NodeId;
    using ai::neat::NodeSplitInnovation;

    // 1: constructor preserves initial counters.
    {
        InnovationTracker tracker(5, 20);
        assert(tracker.getNextAvailableNodeId() == 5 && tracker.getNextAvailableInnovation() == 20 &&
               "constructor must preserve the initial node ID and innovation counters");
    }

    // 2: invalid initial node ID rejected.
    {
        assert(throwsInvalidArgument([]() { InnovationTracker tracker(-1, 0); }) &&
               "a negative firstAvailableNodeId must be rejected");
    }

    // 3: invalid initial innovation rejected.
    {
        assert(throwsInvalidArgument([]() { InnovationTracker tracker(0, -1); }) &&
               "a negative firstAvailableInnovation must be rejected");
    }

    // 4-10: connection innovation behavior.
    {
        InnovationTracker tracker(5, 20);

        // 4: first new connection receives the initial innovation value.
        const InnovationNumber first = tracker.getConnectionInnovation(0, 1);
        assert(first == 20 && "the first new connection must receive the initial innovation value");

        // 5: the same directed connection reuses its innovation.
        const InnovationNumber firstAgain = tracker.getConnectionInnovation(0, 1);
        assert(firstAgain == first && tracker.getNextAvailableInnovation() == 21 &&
               "requesting the same directed connection again must reuse its innovation, not allocate a new one");

        // 6: reversed connection receives a different innovation.
        const InnovationNumber reversed = tracker.getConnectionInnovation(1, 0);
        assert(reversed != first && reversed == 21 && "the reversed connection must be a distinct structural connection");

        // 7: different connections receive unique, monotonically increasing innovations.
        const InnovationNumber c1 = tracker.getConnectionInnovation(2, 3);
        const InnovationNumber c2 = tracker.getConnectionInnovation(3, 4);
        const InnovationNumber c3 = tracker.getConnectionInnovation(4, 5);
        assert(c1 == 22 && c2 == 23 && c3 == 24 &&
               "distinct new connections must receive unique, monotonically increasing innovations");

        // 8: negative source rejected.
        assert(throwsInvalidArgument([&]() { tracker.getConnectionInnovation(-1, 0); }) &&
               "a negative source ID must be rejected");

        // 9: negative target rejected.
        assert(throwsInvalidArgument([&]() { tracker.getConnectionInnovation(0, -1); }) &&
               "a negative target ID must be rejected");

        // 10: self-connection rejected.
        assert(throwsInvalidArgument([&]() { tracker.getConnectionInnovation(2, 2); }) &&
               "a self-connection must be rejected");
    }

    // 11-18: node-split innovation behavior.
    {
        InnovationTracker tracker(100, 200);

        // 11: first node split receives the initial available node ID.
        const NodeSplitInnovation split1 = tracker.getNodeSplitInnovation(50, 1, 2);
        assert(split1.newNodeId == 100 && "the first node split must receive the initial available node ID");

        // 12: split allocates two distinct connection innovations.
        assert(split1.incomingInnovation == 200 && split1.outgoingInnovation == 201 &&
               split1.incomingInnovation != split1.outgoingInnovation &&
               "a split must allocate two distinct connection innovations");
        assert(tracker.getNextAvailableNodeId() == 101 && tracker.getNextAvailableInnovation() == 202 &&
               "a fresh split must advance both counters");

        // 13: repeated identical split reuses the complete record.
        const NodeSplitInnovation split1Again = tracker.getNodeSplitInnovation(50, 1, 2);
        assert(split1Again.newNodeId == split1.newNodeId && split1Again.incomingInnovation == split1.incomingInnovation &&
               split1Again.outgoingInnovation == split1.outgoingInnovation &&
               "requesting the same split again must reuse the complete stored record");

        // 14: repeated split does not advance counters.
        assert(tracker.getNextAvailableNodeId() == 101 && tracker.getNextAvailableInnovation() == 202 &&
               "repeating a split must not allocate anything new");

        // 15 & 16: a different split receives a different node ID and
        // distinct connection innovations.
        const NodeSplitInnovation split2 = tracker.getNodeSplitInnovation(60, 3, 4);
        assert(split2.newNodeId == 101 && split2.newNodeId != split1.newNodeId &&
               "a different split must receive a different node ID");
        assert(split2.incomingInnovation == 202 && split2.outgoingInnovation == 203 &&
               split2.incomingInnovation != split1.incomingInnovation && split2.outgoingInnovation != split1.outgoingInnovation &&
               "a different split must receive distinct connection innovations");

        // 17: inconsistent source/target for an existing split innovation is rejected.
        assert(throwsInvalidArgument([&]() { tracker.getNodeSplitInnovation(50, 1, 3); }) &&
               "requesting an already-recorded split innovation with a different target must be rejected");
        assert(throwsInvalidArgument([&]() { tracker.getNodeSplitInnovation(50, 9, 2); }) &&
               "requesting an already-recorded split innovation with a different source must be rejected");

        // 18: negative split innovation rejected.
        assert(throwsInvalidArgument([&]() { tracker.getNodeSplitInnovation(-1, 1, 2); }) &&
               "a negative splitConnectionInnovation must be rejected");
    }

    // 19 & 20: split-created connection innovations are registered in the
    // same global connection history -- later direct requests for
    // source->newNode and newNode->target reuse them exactly.
    {
        InnovationTracker tracker(100, 200);
        const NodeSplitInnovation split = tracker.getNodeSplitInnovation(50, 1, 2);

        const InnovationNumber directIncoming = tracker.getConnectionInnovation(1, split.newNodeId);
        assert(directIncoming == split.incomingInnovation && tracker.getNextAvailableInnovation() == 202 &&
               "a split's incoming connection must be reusable via a direct getConnectionInnovation call, "
               "allocating nothing new");

        const InnovationNumber directOutgoing = tracker.getConnectionInnovation(split.newNodeId, 2);
        assert(directOutgoing == split.outgoingInnovation && tracker.getNextAvailableInnovation() == 202 &&
               "a split's outgoing connection must be reusable via a direct getConnectionInnovation call, "
               "allocating nothing new");
    }

    // 21: a direct connection innovation created before a split is reused
    // by a later split that needs the exact same (source, newNode) pair.
    {
        InnovationTracker tracker(10, 100);

        const InnovationNumber preExisting = tracker.getConnectionInnovation(1, 10); // 10 is what the split below will allocate
        assert(preExisting == 100 && tracker.getNextAvailableInnovation() == 101 &&
               "setup: the pre-existing direct connection must be the first allocated innovation");

        const NodeSplitInnovation split = tracker.getNodeSplitInnovation(999, 1, 2);
        assert(split.newNodeId == 10 &&
               "the split's new node ID must be the next available one, matching the pre-existing connection's target");
        assert(split.incomingInnovation == preExisting &&
               "a connection innovation already registered directly must be reused by a later split needing the same pair");
        assert(tracker.getNextAvailableInnovation() == 102 &&
               "only the split's new (outgoing) connection should allocate a fresh innovation -- the incoming one was reused");
    }

    // 22 & 23: no node ID or connection innovation is ever reused for a
    // different structure, across a batch of distinct splits. Node IDs are
    // allocated starting well above the source/target range used below, so
    // a freshly allocated new node ID can never coincide with one of them.
    {
        InnovationTracker tracker(1000, 0);
        std::vector<NodeId> nodeIds;
        std::vector<InnovationNumber> innovations;

        for (int i = 0; i < 5; ++i)
        {
            const NodeSplitInnovation split = tracker.getNodeSplitInnovation(i, i, i + 100);
            nodeIds.push_back(split.newNodeId);
            innovations.push_back(split.incomingInnovation);
            innovations.push_back(split.outgoingInnovation);
        }

        for (std::size_t i = 0; i < nodeIds.size(); ++i)
        {
            for (std::size_t j = i + 1; j < nodeIds.size(); ++j)
            {
                assert(nodeIds[i] != nodeIds[j] && "no node ID may ever be reused for a different split");
            }
        }
        for (std::size_t i = 0; i < innovations.size(); ++i)
        {
            for (std::size_t j = i + 1; j < innovations.size(); ++j)
            {
                assert(innovations[i] != innovations[j] &&
                       "no innovation number may ever be reused for a different directed pair");
            }
        }
    }

    // 24: a fixed request sequence is deterministic across two identically
    // constructed trackers. Node IDs start at 100, well above the 0..3
    // source/target range the sequence below uses, so the split's freshly
    // allocated new node ID can never collide with one of them.
    {
        InnovationTracker trackerA(100, 0);
        InnovationTracker trackerB(100, 0);

        auto runSequence = [](InnovationTracker& t)
        {
            std::vector<InnovationNumber> results;
            results.push_back(t.getConnectionInnovation(0, 1));
            results.push_back(t.getConnectionInnovation(1, 2));
            const NodeSplitInnovation split = t.getNodeSplitInnovation(0, 0, 1);
            results.push_back(split.newNodeId);
            results.push_back(split.incomingInnovation);
            results.push_back(split.outgoingInnovation);
            results.push_back(t.getConnectionInnovation(2, 3));
            return results;
        };

        const std::vector<InnovationNumber> resultsA = runSequence(trackerA);
        const std::vector<InnovationNumber> resultsB = runSequence(trackerB);
        assert(resultsA == resultsB &&
               "an identical request sequence on two identically constructed trackers must be fully deterministic");
    }

    // 25: a different (still valid) request order may produce different
    // numbering for the same pair, while each tracker remains internally
    // consistent.
    {
        InnovationTracker trackerD(0, 50);
        InnovationTracker trackerE(0, 50);

        const InnovationNumber d_first = trackerD.getConnectionInnovation(1, 2);
        const InnovationNumber d_second = trackerD.getConnectionInnovation(3, 4);

        const InnovationNumber e_first = trackerE.getConnectionInnovation(3, 4);
        const InnovationNumber e_second = trackerE.getConnectionInnovation(1, 2);

        assert(d_first == 50 && d_second == 51 && "requesting (1,2) before (3,4) must give (1,2) the earlier innovation");
        assert(e_first == 50 && e_second == 51 && "requesting (3,4) before (1,2) must give (3,4) the earlier innovation");
        assert(d_first != e_second &&
               "different valid request orders may assign different innovation numbers to the same pair");

        assert(trackerD.getConnectionInnovation(1, 2) == d_first && trackerE.getConnectionInnovation(1, 2) == e_second &&
               "each tracker must remain internally self-consistent regardless of request order");
    }

    // 26: node-counter overflow is rejected.
    {
        InnovationTracker exhaustedNodes(std::numeric_limits<NodeId>::max(), 0);
        assert(throwsOverflowError([&]() { exhaustedNodes.getNodeSplitInnovation(0, 1, 2); }) &&
               "a node split when the node ID counter is exhausted must throw std::overflow_error");
    }

    // 27: innovation-counter overflow is rejected.
    {
        InnovationTracker exhaustedInnovations(0, std::numeric_limits<InnovationNumber>::max());
        assert(throwsOverflowError([&]() { exhaustedInnovations.getConnectionInnovation(1, 2); }) &&
               "a new connection when the innovation counter is exhausted must throw std::overflow_error");
    }

    // 28 & 29: an InnovationTracker requires no Genome to construct or use
    // -- see the function's doc comment; enforced by the absence of any
    // Genome parameter anywhere in its API, not by a runtime check.

    // 30: all previous verification suites still pass -- enforced by
    // main() continuing to call every earlier verify*() function unchanged.

    TraceLog(LOG_INFO, "Innovation tracker verification: all deterministic checks passed");
}

// One-shot, deterministic sanity check of
// ai::neat::GenomeMutator::mutateAddConnection() (Stage 9C), independent of
// Car/Track/AI/keyboard/render timing. Runs once at startup. No add-node
// mutation, crossover, species, or population logic exists to verify here
// -- only add-connection structural mutation, per Stage 9C's scope. This
// mutation is never invoked from the normal driving loop; it is exercised
// only by this verification.
void verifyAddConnectionMutation()
{
    using namespace genome_mutator_verify;
    using ai::neat::ConnectionGene;
    using ai::neat::Genome;
    using ai::neat::GenomeMutator;
    using ai::neat::InnovationNumber;
    using ai::neat::InnovationTracker;
    using ai::neat::MutationConfig;
    using ai::neat::NodeGene;

    // 1: probability 0 returns false and changes nothing.
    {
        Genome genome = makeTestGenome();
        const std::vector<NodeGene> nodesBefore = genome.nodes();
        const std::vector<ConnectionGene> connectionsBefore = genome.connections();
        InnovationTracker tracker(200, 300);
        MutationConfig config;
        config.addConnectionProbability = 0.0f;
        GenomeMutator mutator(1u);

        const bool added = mutator.mutateAddConnection(genome, tracker, config);
        assert(!added && "probability 0 must never add a connection");
        assert(genome.nodes().size() == nodesBefore.size() && genome.connections().size() == connectionsBefore.size() &&
               "probability 0 must change nothing structurally");
        for (std::size_t i = 0; i < connectionsBefore.size(); ++i)
        {
            assert(genome.connections()[i].getWeight() == connectionsBefore[i].getWeight() &&
                   "probability 0 must not touch any existing connection");
        }
        assert(tracker.getNextAvailableNodeId() == 200 && tracker.getNextAvailableInnovation() == 300 &&
               "probability 0 must leave the tracker unchanged");
    }

    // 2 & 6-13: probability 1 attempts (and, given an available valid
    // pair, succeeds at) adding exactly one new, well-formed connection,
    // without touching anything that already existed.
    {
        Genome genome = makeTestGenome();
        const std::size_t nodeCountBefore = genome.nodes().size();
        const std::vector<NodeGene> nodesBefore = genome.nodes();
        const std::vector<ConnectionGene> connectionsBefore = genome.connections();

        InnovationTracker tracker(200, 300);
        MutationConfig config;
        config.addConnectionProbability = 1.0f;
        GenomeMutator mutator(42u);

        const bool added = mutator.mutateAddConnection(genome, tracker, config);
        assert(added && "probability 1 with an available valid pair must add a connection"); // 2 & 6 (part 1)

        assert(genome.nodes().size() == nodeCountBefore && "node count must remain unchanged"); // 7
        for (std::size_t i = 0; i < nodesBefore.size(); ++i)
        {
            assert(genome.nodes()[i] == nodesBefore[i] && "existing node genes must remain unchanged"); // 8 (nodes)
        }
        for (std::size_t i = 0; i < connectionsBefore.size(); ++i)
        {
            const ConnectionGene& before = connectionsBefore[i];
            const ConnectionGene& after = genome.connections()[i];
            assert(after.getSourceId() == before.getSourceId() && after.getTargetId() == before.getTargetId() &&
                   after.getWeight() == before.getWeight() && after.isEnabled() == before.isEnabled() &&
                   after.getInnovationNumber() == before.getInnovationNumber() &&
                   "existing connection genes must remain unchanged"); // 8 (connections)
        }
        assert(genome.connections().size() == connectionsBefore.size() + 1 &&
               "connection count must increase by exactly one"); // 6 (part 2)

        const ConnectionGene& newConnection = genome.connections().back();
        assert(genome.hasNode(newConnection.getSourceId()) && genome.hasNode(newConnection.getTargetId()) &&
               "new connection endpoints must reference existing nodes"); // 9
        assert(newConnection.isEnabled() && "a newly added connection must be enabled"); // 10
        assert(newConnection.getWeight() >= config.newConnectionWeightMin &&
               newConnection.getWeight() <= config.newConnectionWeightMax &&
               "new weight must lie within the configured bounds"); // 11

        // 12: innovation comes from InnovationTracker -- re-requesting the
        // same pair from the same tracker must reuse (not change) it.
        const InnovationNumber beforeReuseCheck = tracker.getNextAvailableInnovation();
        const InnovationNumber sameInnovation =
            tracker.getConnectionInnovation(newConnection.getSourceId(), newConnection.getTargetId());
        assert(sameInnovation == newConnection.getInnovationNumber() && tracker.getNextAvailableInnovation() == beforeReuseCheck &&
               "the new connection's innovation number must come from, and already be recorded in, InnovationTracker");

        // 13: tracker advances only after successful mutation -- node
        // counter untouched (add-connection never allocates a node ID),
        // innovation counter advanced by exactly one.
        assert(tracker.getNextAvailableNodeId() == 200 && "add-connection mutation must never allocate a node ID");
        assert(tracker.getNextAvailableInnovation() == 301 &&
               "a successful mutation must advance the innovation counter by exactly one");
    }

    // 14: a failed mutation (selected, but no valid candidate exists) does
    // not advance the tracker.
    {
        Genome genome = makeAlreadyConnectedGenome(/*enabled=*/true);
        InnovationTracker tracker(50, 60);
        MutationConfig config;
        config.addConnectionProbability = 1.0f;
        GenomeMutator mutator(1u);

        const bool added = mutator.mutateAddConnection(genome, tracker, config);
        assert(!added && "a genome whose only structurally possible pair already exists must fail to mutate");
        assert(tracker.getNextAvailableNodeId() == 50 && tracker.getNextAvailableInnovation() == 60 &&
               "a failed mutation must leave the tracker completely unchanged");
    }

    // 3: empty Genome returns false.
    {
        Genome empty;
        InnovationTracker tracker(0, 0);
        MutationConfig config;
        config.addConnectionProbability = 1.0f;
        GenomeMutator mutator(1u);
        assert(!mutator.mutateAddConnection(empty, tracker, config) && "an empty genome must return false");
        assert(tracker.getNextAvailableNodeId() == 0 && tracker.getNextAvailableInnovation() == 0 &&
               "an empty genome must leave the tracker unchanged");
    }

    // 4 & 18: a genome with no valid source node type (only Output nodes)
    // returns false -- Output can never be a source.
    {
        Genome genome = makeOnlyOutputsGenome();
        InnovationTracker tracker(0, 0);
        MutationConfig config;
        config.addConnectionProbability = 1.0f;
        GenomeMutator mutator(1u);
        assert(!mutator.mutateAddConnection(genome, tracker, config) &&
               "a genome with only Output nodes must have no valid source and must return false");
    }

    // 5 & 19: a genome with no valid target node type (only Input nodes)
    // returns false -- Input can never be a target.
    {
        Genome genome = makeOnlyInputsGenome();
        InnovationTracker tracker(0, 0);
        MutationConfig config;
        config.addConnectionProbability = 1.0f;
        GenomeMutator mutator(1u);
        assert(!mutator.mutateAddConnection(genome, tracker, config) &&
               "a genome with only Input nodes must have no valid target and must return false");
    }

    // 20: Bias can never be a target either.
    {
        Genome genome = makeOnlyBiasGenome();
        InnovationTracker tracker(0, 0);
        MutationConfig config;
        config.addConnectionProbability = 1.0f;
        GenomeMutator mutator(1u);
        assert(!mutator.mutateAddConnection(genome, tracker, config) &&
               "a genome with only Bias nodes must have no valid target and must return false");
    }

    // 21 & 25: Input can be a source, Output can be a target -- the only
    // structurally possible pair must be found and added.
    {
        Genome genome = makeInputToOutputGenome();
        InnovationTracker tracker(0, 0);
        MutationConfig config;
        config.addConnectionProbability = 1.0f;
        GenomeMutator mutator(7u);
        assert(mutator.mutateAddConnection(genome, tracker, config) && "Input -> Output must be found as a valid candidate");
        assert(genome.connections().size() == 1 && genome.connections()[0].getSourceId() == 0 &&
               genome.connections()[0].getTargetId() == 1 && "the added connection must be exactly Input(0) -> Output(1)");
    }

    // 22: Bias can be a source.
    {
        Genome genome = makeBiasToOutputGenome();
        InnovationTracker tracker(0, 0);
        MutationConfig config;
        config.addConnectionProbability = 1.0f;
        GenomeMutator mutator(7u);
        assert(mutator.mutateAddConnection(genome, tracker, config) && "Bias -> Output must be found as a valid candidate");
        assert(genome.connections().size() == 1 && genome.connections()[0].getSourceId() == 0 &&
               "Bias must be usable as a source");
    }

    // 23 & 24: Hidden can be both a source and a target.
    {
        Genome genome = makeHiddenToHiddenGenome();
        InnovationTracker tracker(0, 0);
        MutationConfig config;
        config.addConnectionProbability = 1.0f;
        GenomeMutator mutator(7u);
        assert(mutator.mutateAddConnection(genome, tracker, config) && "Hidden -> Hidden must be found as a valid candidate");
        assert(genome.connections().size() == 1 && genome.connections()[0].getSourceId() == 0 &&
               genome.connections()[0].getTargetId() == 1 && "Hidden must be usable as both source and target");
    }

    // 17: self-connection is never added, even when it is the only
    // structurally "available" pair.
    {
        Genome genome = makeSingleHiddenGenome();
        InnovationTracker tracker(0, 0);
        MutationConfig config;
        config.addConnectionProbability = 1.0f;
        GenomeMutator mutator(1u);
        assert(!mutator.mutateAddConnection(genome, tracker, config) && "a single node can never validly connect to itself");
        assert(genome.connections().empty() && "no self-connection may ever be added");
    }

    // 15: duplicate directed connection is never added.
    {
        Genome genome = makeAlreadyConnectedGenome(/*enabled=*/true);
        InnovationTracker tracker(0, 0);
        MutationConfig config;
        config.addConnectionProbability = 1.0f;
        GenomeMutator mutator(1u);
        assert(!mutator.mutateAddConnection(genome, tracker, config) &&
               "the only structurally possible pair already exists, so nothing may be added");
        assert(genome.connections().size() == 1 && "an already-existing connection must never be duplicated");
    }

    // 16: a disabled duplicate still blocks addition.
    {
        Genome genome = makeAlreadyConnectedGenome(/*enabled=*/false);
        InnovationTracker tracker(0, 0);
        MutationConfig config;
        config.addConnectionProbability = 1.0f;
        GenomeMutator mutator(1u);
        assert(!mutator.mutateAddConnection(genome, tracker, config) &&
               "a disabled connection between the only possible pair must still block a duplicate");
        assert(genome.connections().size() == 1 && !genome.connections()[0].isEnabled() &&
               "the existing disabled connection must not be re-enabled or duplicated");
    }

    // 26 & 28: a connection that would create an enabled cycle is
    // rejected, even with a deliberately "descending" node ID edge (10 ->
    // 3): the only other structurally possible pair, 3 -> 10, would close
    // a direct 2-cycle and must be rejected, leaving no valid pair at all.
    {
        Genome genome = makeCycleGenome(/*existingEdgeEnabled=*/true);
        InnovationTracker tracker(0, 0);
        MutationConfig config;
        config.addConnectionProbability = 1.0f;
        GenomeMutator mutator(1u);
        assert(!mutator.mutateAddConnection(genome, tracker, config) &&
               "3 -> 10 must be rejected as a cycle (10 already has an enabled path to 3), regardless of node ID order");
        assert(genome.connections().size() == 1 && "a rejected cycle candidate must not be added");
    }

    // 27: disabled edges do not participate in cycle detection -- with the
    // existing 10 -> 3 edge disabled, 3 -> 10 is no longer a cycle and must
    // be accepted.
    {
        Genome genome = makeCycleGenome(/*existingEdgeEnabled=*/false);
        InnovationTracker tracker(0, 0);
        MutationConfig config;
        config.addConnectionProbability = 1.0f;
        GenomeMutator mutator(1u);
        const bool added = mutator.mutateAddConnection(genome, tracker, config);
        assert(added && "with 10 -> 3 disabled, it is not an enabled path, so 3 -> 10 is not a cycle");
        assert(genome.connections().size() == 2 && genome.connections().back().getSourceId() == 3 &&
               genome.connections().back().getTargetId() == 10 &&
               "the new connection must be exactly 3 -> 10, proving disabled edges do not create false cycle detection");
    }

    // 29: the deterministic exhaustive fallback finds the sole valid pair
    // even when limited random attempts are likely to miss it. Looping
    // over many seeds with maxAttempts = 1 makes it overwhelmingly likely
    // that at least some of them exhaust their one random attempt without
    // landing on the only valid pair (Bias(1) -> Output(3), 1 of 6
    // reachable combinations), relying on the fallback scan to still find
    // it -- every seed must still succeed and land on that same pair.
    {
        for (std::uint32_t seed = 1; seed <= 20; ++seed)
        {
            Genome genome = makeSparseValidPairGenome();
            InnovationTracker tracker(0, 0);
            MutationConfig config;
            config.addConnectionProbability = 1.0f;
            config.addConnectionMaxAttempts = 1;
            GenomeMutator mutator(seed);

            const bool added = mutator.mutateAddConnection(genome, tracker, config);
            assert(added && genome.connections().back().getSourceId() == 1 && genome.connections().back().getTargetId() == 3 &&
                   "the single valid pair must always be found, whether by the one random attempt or the fallback scan");
        }
    }

    // 30: a saturated valid DAG (every valid feed-forward pair already
    // exists) returns false.
    {
        Genome genome = makeSaturatedGenome();
        InnovationTracker tracker(0, 0);
        MutationConfig config;
        config.addConnectionProbability = 1.0f;
        GenomeMutator mutator(1u);
        assert(!mutator.mutateAddConnection(genome, tracker, config) &&
               "a genome where every valid pair already exists must return false");
        assert(genome.connections().size() == 5 && "a saturated genome must not gain a new connection");
    }

    // 31: identical seed + identical genome + identical tracker state
    // produces the same added pair, weight, and innovation.
    {
        MutationConfig config;
        config.addConnectionProbability = 1.0f;

        Genome genomeA = makeTestGenome();
        InnovationTracker trackerA(200, 300);
        GenomeMutator mutatorA(777u);
        mutatorA.mutateAddConnection(genomeA, trackerA, config);
        const ConnectionGene& resultA = genomeA.connections().back();

        Genome genomeB = makeTestGenome();
        InnovationTracker trackerB(200, 300);
        GenomeMutator mutatorB(777u);
        mutatorB.mutateAddConnection(genomeB, trackerB, config);
        const ConnectionGene& resultB = genomeB.connections().back();

        assert(resultA.getSourceId() == resultB.getSourceId() && resultA.getTargetId() == resultB.getTargetId() &&
               resultA.getWeight() == resultB.getWeight() && resultA.getInnovationNumber() == resultB.getInnovationNumber() &&
               "identical seed, genome and tracker state must produce identical results");
    }

    // 32: different seeds can produce different valid results.
    {
        MutationConfig config;
        config.addConnectionProbability = 1.0f;

        Genome genomeA = makeTestGenome();
        InnovationTracker trackerA(200, 300);
        GenomeMutator mutatorA(1u);
        mutatorA.mutateAddConnection(genomeA, trackerA, config);
        const ConnectionGene& resultA = genomeA.connections().back();

        Genome genomeB = makeTestGenome();
        InnovationTracker trackerB(200, 300);
        GenomeMutator mutatorB(2u);
        mutatorB.mutateAddConnection(genomeB, trackerB, config);
        const ConnectionGene& resultB = genomeB.connections().back();

        assert((resultA.getSourceId() != resultB.getSourceId() || resultA.getTargetId() != resultB.getTargetId() ||
                resultA.getWeight() != resultB.getWeight()) &&
               "different seeds must be capable of producing different results");
    }

    // 33: repeated calls advance the mutator's owned RNG state.
    {
        MutationConfig config;
        config.addConnectionProbability = 1.0f;

        Genome genomeA = makeTestGenome();
        InnovationTracker trackerA(200, 300);
        Genome genomeB = makeTestGenome();
        InnovationTracker trackerB(200, 300);

        GenomeMutator mutator(55u);
        mutator.mutateAddConnection(genomeA, trackerA, config);
        mutator.mutateAddConnection(genomeB, trackerB, config);

        const ConnectionGene& resultA = genomeA.connections().back();
        const ConnectionGene& resultB = genomeB.connections().back();
        assert((resultA.getSourceId() != resultB.getSourceId() || resultA.getTargetId() != resultB.getTargetId() ||
                resultA.getWeight() != resultB.getWeight()) &&
               "repeated calls from the same mutator must advance its RNG state, not repeat the same draws");
    }

    // 34, 35, 36 & 37: invalid configuration is rejected, and never
    // touches the tracker.
    {
        Genome genome = makeTestGenome();
        InnovationTracker tracker(200, 300);
        GenomeMutator mutator(1u);

        MutationConfig badProbability;
        badProbability.addConnectionProbability = 1.5f;
        assert(throwsInvalidArgument([&]() { mutator.mutateAddConnection(genome, tracker, badProbability); }) &&
               "an addConnectionProbability outside [0,1] must be rejected"); // 34

        MutationConfig invertedRange;
        invertedRange.newConnectionWeightMin = 1.0f;
        invertedRange.newConnectionWeightMax = -1.0f;
        assert(throwsInvalidArgument([&]() { mutator.mutateAddConnection(genome, tracker, invertedRange); }) &&
               "an inverted new-connection weight range must be rejected"); // 35

        MutationConfig infiniteBound;
        infiniteBound.newConnectionWeightMin = std::numeric_limits<float>::infinity();
        assert(throwsInvalidArgument([&]() { mutator.mutateAddConnection(genome, tracker, infiniteBound); }) &&
               "a non-finite weight bound must be rejected"); // 36

        MutationConfig nanProbability;
        nanProbability.addConnectionProbability = std::numeric_limits<float>::quiet_NaN();
        assert(throwsInvalidArgument([&]() { mutator.mutateAddConnection(genome, tracker, nanProbability); }) &&
               "a NaN probability must be rejected"); // 36 (continued)

        MutationConfig zeroAttempts;
        zeroAttempts.addConnectionMaxAttempts = 0;
        assert(throwsInvalidArgument([&]() { mutator.mutateAddConnection(genome, tracker, zeroAttempts); }) &&
               "a zero addConnectionMaxAttempts must be rejected"); // 37

        MutationConfig negativeAttempts;
        negativeAttempts.addConnectionMaxAttempts = -5;
        assert(throwsInvalidArgument([&]() { mutator.mutateAddConnection(genome, tracker, negativeAttempts); }) &&
               "a negative addConnectionMaxAttempts must be rejected"); // 37 (continued)

        assert(tracker.getNextAvailableNodeId() == 200 && tracker.getNextAvailableInnovation() == 300 &&
               "rejected configuration must never touch the tracker");
    }

    // 38, 39 & 40: Genome::validate() and buildPhenotype() succeed after
    // mutation, and the resulting phenotype is deterministic.
    {
        Genome genome = makeTestGenome();
        InnovationTracker tracker(200, 300);
        MutationConfig config;
        config.addConnectionProbability = 1.0f;
        GenomeMutator mutator(123u);

        assert(mutator.mutateAddConnection(genome, tracker, config) && "setup for phenotype checks must succeed");

        genome.validate(); // 38: must not throw

        ai::NeuralNetwork phenotype = ai::neat::buildPhenotype(genome);      // 39: must not throw
        ai::NeuralNetwork phenotypeAgain = ai::neat::buildPhenotype(genome); // 40: rebuilding must match

        ai::Observation obs;
        obs.values.fill(0.3f);
        const auto out1 = phenotype.evaluate(obs);
        const auto out2 = phenotypeAgain.evaluate(obs);
        assert(out1[0] == out2[0] && out1[1] == out2[1] && "the phenotype built from a mutated genome must be deterministic");
    }

    // 41: weight mutation behavior from Stage 9A remains unchanged (spot
    // regression check -- zero probability still changes no weights).
    {
        Genome genome = makeTestGenome();
        const std::vector<ConnectionGene> before = genome.connections();
        MutationConfig config;
        config.weightMutationProbability = 0.0f;
        GenomeMutator mutator(1u);
        mutator.mutateWeights(genome, config);
        for (std::size_t i = 0; i < before.size(); ++i)
        {
            assert(genome.connections()[i].getWeight() == before[i].getWeight() &&
                   "Stage 9A's mutateWeights() behavior must remain unchanged: zero probability still changes nothing");
        }
    }

    // 42: no add-node behavior occurs -- across a batch of seeds, the
    // tracker's node-ID counter never advances (13 already checks this
    // once; re-confirmed here for extra confidence).
    {
        for (std::uint32_t seed = 1; seed <= 5; ++seed)
        {
            Genome genome = makeTestGenome();
            InnovationTracker tracker(200, 300);
            MutationConfig config;
            config.addConnectionProbability = 1.0f;
            GenomeMutator mutator(seed);
            mutator.mutateAddConnection(genome, tracker, config);
            assert(tracker.getNextAvailableNodeId() == 200 &&
                   "add-connection mutation must never allocate a node ID (no add-node behavior)");
        }
    }

    // 43: all previous verification suites still pass -- enforced by
    // main() continuing to call every earlier verify*() function unchanged.

    TraceLog(LOG_INFO, "Add-connection mutation verification: all deterministic checks passed");
}

namespace add_node_verify
{

// Minimal genome: one Input, one Output, one enabled connection. Used for
// the plain structural-invariant checks that don't need a full
// Observation-shaped (9 Input + 1 Bias + 2 Output) genome.
ai::neat::Genome makeSimpleSplitGenome()
{
    using ai::neat::ConnectionGene;
    using ai::neat::Genome;
    using ai::neat::NodeGene;
    using ai::neat::NodeType;

    Genome genome;
    genome.addNode(NodeGene{0, NodeType::Input});
    genome.addNode(NodeGene{1, NodeType::Output});
    genome.addConnection(ConnectionGene{0, 1, 0.75f, true, 0});
    return genome;
}

// A genome whose sole connection is already disabled: no eligible candidate
// exists to split.
ai::neat::Genome makeOnlyDisabledConnectionGenome()
{
    using ai::neat::ConnectionGene;
    using ai::neat::Genome;
    using ai::neat::NodeGene;
    using ai::neat::NodeType;

    Genome genome;
    genome.addNode(NodeGene{0, NodeType::Input});
    genome.addNode(NodeGene{1, NodeType::Output});
    genome.addConnection(ConnectionGene{0, 1, 0.5f, false, 0});
    return genome;
}

} // namespace add_node_verify

// One-shot, deterministic sanity check of
// ai::neat::GenomeMutator::mutateAddNode, covering add-node structural
// mutation end to end. Independent of Car/Track/AI/keyboard/render timing.
// Runs once at startup. No crossover, species, or population logic exists
// to verify here -- only add-node structural mutation, per Stage 9D's
// scope.
void verifyAddNodeMutation()
{
    using namespace add_node_verify;
    using namespace genome_mutator_verify;
    using ai::neat::ConnectionGene;
    using ai::neat::Genome;
    using ai::neat::GenomeMutator;
    using ai::neat::InnovationNumber;
    using ai::neat::InnovationTracker;
    using ai::neat::MutationConfig;
    using ai::neat::NodeGene;
    using ai::neat::NodeId;
    using ai::neat::NodeSplitInnovation;
    using ai::neat::NodeType;

    // 1 & 25: probability 0 returns false and changes nothing; tracker
    // unchanged.
    {
        Genome genome = makeSimpleSplitGenome();
        const std::vector<NodeGene> nodesBefore = genome.nodes();
        const std::vector<ConnectionGene> connectionsBefore = genome.connections();
        InnovationTracker tracker(50, 60);
        MutationConfig config;
        config.addNodeProbability = 0.0f;
        GenomeMutator mutator(1u);

        const bool result = mutator.mutateAddNode(genome, tracker, config);
        assert(!result && "probability 0 must never split a connection");
        assert(genome.nodes().size() == nodesBefore.size() && genome.connections().size() == connectionsBefore.size() &&
               "probability 0 must change nothing structurally");
        assert(tracker.getNextAvailableNodeId() == 50 && tracker.getNextAvailableInnovation() == 60 &&
               "probability 0 must leave the tracker unchanged");
    }

    // 3 & 26: empty genome returns false, tracker unchanged.
    {
        Genome empty;
        InnovationTracker tracker(0, 0);
        MutationConfig config;
        config.addNodeProbability = 1.0f;
        GenomeMutator mutator(1u);
        assert(!mutator.mutateAddNode(empty, tracker, config) && "an empty genome must return false");
        assert(tracker.getNextAvailableNodeId() == 0 && tracker.getNextAvailableInnovation() == 0 &&
               "an empty genome must leave the tracker unchanged");
    }

    // 4, 26 & 33: a genome with only disabled connections returns false,
    // tracker unchanged, and the disabled connection remains untouched.
    {
        Genome genome = makeOnlyDisabledConnectionGenome();
        InnovationTracker tracker(10, 20);
        MutationConfig config;
        config.addNodeProbability = 1.0f;
        GenomeMutator mutator(1u);
        assert(!mutator.mutateAddNode(genome, tracker, config) &&
               "a genome with only disabled connections must have no eligible candidate");
        assert(tracker.getNextAvailableNodeId() == 10 && tracker.getNextAvailableInnovation() == 20 &&
               "no eligible candidate must leave the tracker unchanged");
        assert(genome.connections().size() == 1 && !genome.connections()[0].isEnabled() &&
               "the sole disabled connection must remain untouched");
    }

    // 2, 5-21: probability 1 splits the sole enabled connection of a
    // minimal genome, producing exactly the structure NEAT's add-node
    // mutation specifies.
    {
        Genome genome = makeSimpleSplitGenome();
        InnovationTracker tracker(50, 60);
        MutationConfig config;
        config.addNodeProbability = 1.0f;
        GenomeMutator mutator(1u);

        const bool result = mutator.mutateAddNode(genome, tracker, config);
        assert(result && "probability 1 with an eligible connection must split it"); // 2

        // NEW 5: a successful, brand-new split advances the tracker by
        // exactly one node ID and exactly two innovation numbers.
        assert(tracker.getNextAvailableNodeId() == 51 && "a successful split must allocate exactly one new node ID");
        assert(tracker.getNextAvailableInnovation() == 62 &&
               "a successful split must allocate exactly two new innovation numbers");

        assert(genome.nodes().size() == 3 && "node count must increase by exactly one"); // 5
        assert(genome.connections().size() == 3 && "connection count must increase by exactly two"); // 6

        const ConnectionGene* original = genome.findConnection(0, 1);
        assert(original != nullptr && "the original connection must remain present"); // 7
        assert(!original->isEnabled() && "the original connection must become disabled"); // 8
        assert(original->getWeight() == 0.75f && "the original connection's weight must remain unchanged"); // 9 (weight)
        assert(original->getSourceId() == 0 && original->getTargetId() == 1 &&
               "the original connection's source/target must remain unchanged"); // 9 (source/target)
        assert(original->getInnovationNumber() == 0 && "the original connection's innovation must remain unchanged"); // 9 (innovation)

        const NodeGene& newNode = genome.nodes().back();
        assert(newNode.getType() == NodeType::Hidden && "the new node must be Hidden"); // 10

        // Re-requesting the same split from the tracker is idempotent and
        // must return exactly what mutateAddNode used -- proving the new
        // node ID and both new innovations came from InnovationTracker.
        const NodeSplitInnovation split = tracker.getNodeSplitInnovation(0, 0, 1);
        assert(newNode.getId() == split.newNodeId && "the new node's ID must come from InnovationTracker"); // 11

        const ConnectionGene* incoming = genome.findConnection(0, split.newNodeId);
        assert(incoming != nullptr && "source -> newNode connection must exist");
        assert(incoming->getWeight() == 1.0f && "source -> newNode weight must be exactly 1.0"); // 12
        assert(incoming->isEnabled() && "source -> newNode connection must be enabled"); // 14 (part 1)
        assert(incoming->getInnovationNumber() == split.incomingInnovation &&
               "source -> newNode innovation must come from InnovationTracker"); // 15 (part 1)

        const ConnectionGene* outgoing = genome.findConnection(split.newNodeId, 1);
        assert(outgoing != nullptr && "newNode -> target connection must exist");
        assert(outgoing->getWeight() == 0.75f && "newNode -> target weight must equal the old connection's weight"); // 13
        assert(outgoing->isEnabled() && "newNode -> target connection must be enabled"); // 14 (part 2)
        assert(outgoing->getInnovationNumber() == split.outgoingInnovation &&
               "newNode -> target innovation must come from InnovationTracker"); // 15 (part 2)

        // 16 & 17: no unrelated node/connection exists -- exactly the three
        // nodes and three connections accounted for above.
        assert(genome.hasNode(0) && genome.hasNode(1) && genome.hasNode(split.newNodeId) &&
               "no unrelated node may appear");
        assert(genome.connections().size() == 3 && "no unrelated connection may appear");

        genome.validate(); // 18: must not throw
    }

    // 18, 19, 20 & 21: on an Observation-shaped genome, Genome::validate()
    // and buildPhenotype() succeed after mutation, the phenotype is
    // acyclic (buildPhenotype would otherwise throw), and the disabled
    // original connection never affects evaluation.
    {
        Genome genome = makeTestGenome();
        InnovationTracker tracker(200, 300);
        MutationConfig config;
        config.addNodeProbability = 1.0f;
        GenomeMutator mutator(123u);

        assert(mutator.mutateAddNode(genome, tracker, config) && "setup for phenotype checks must succeed");

        genome.validate(); // 18: must not throw

        ai::NeuralNetwork phenotype = ai::neat::buildPhenotype(genome); // 19 & 20: must not throw (acyclic)
        ai::Observation obs;
        obs.values.fill(0.3f);
        const auto outBefore = phenotype.evaluate(obs);

        // 21: perturbing every disabled connection's weight (including the
        // split original) must not change the phenotype's evaluation.
        for (ConnectionGene& connection : genome.mutableConnections())
        {
            if (!connection.isEnabled())
            {
                connection.setWeight(connection.getWeight() + 999.0f);
            }
        }
        ai::NeuralNetwork phenotypeAfter = ai::neat::buildPhenotype(genome);
        const auto outAfter = phenotypeAfter.evaluate(obs);
        assert(outBefore[0] == outAfter[0] && outBefore[1] == outAfter[1] &&
               "disabled connections, including the split original, must never affect evaluation");
    }

    // 22: identical seed + identical genome + identical tracker state
    // produces the same split.
    {
        MutationConfig config;
        config.addNodeProbability = 1.0f;

        Genome genomeA = makeTestGenome();
        InnovationTracker trackerA(200, 300);
        GenomeMutator mutatorA(777u);
        mutatorA.mutateAddNode(genomeA, trackerA, config);

        Genome genomeB = makeTestGenome();
        InnovationTracker trackerB(200, 300);
        GenomeMutator mutatorB(777u);
        mutatorB.mutateAddNode(genomeB, trackerB, config);

        assert(genomeA.nodes().size() == genomeB.nodes().size() &&
               genomeA.connections().size() == genomeB.connections().size() &&
               "identical seed, genome and tracker state must produce a structurally identical result");
        const NodeGene& newNodeA = genomeA.nodes().back();
        const NodeGene& newNodeB = genomeB.nodes().back();
        assert(newNodeA.getId() == newNodeB.getId() && "identical runs must pick the same new node ID");
        const ConnectionGene& lastA = genomeA.connections().back();
        const ConnectionGene& lastB = genomeB.connections().back();
        assert(lastA.getSourceId() == lastB.getSourceId() && lastA.getTargetId() == lastB.getTargetId() &&
               lastA.getInnovationNumber() == lastB.getInnovationNumber() &&
               "identical runs must split the same original connection");
    }

    // 23 & 33: different seeds can select different enabled connections to
    // split (looped over a batch of seeds, at least two distinct choices
    // must occur since makeTestGenome() has three eligible connections),
    // and the pre-existing disabled connection is never selected or
    // altered.
    {
        MutationConfig config;
        config.addNodeProbability = 1.0f;
        std::vector<std::pair<NodeId, NodeId>> splitOriginals;

        for (std::uint32_t seed = 1; seed <= 10; ++seed)
        {
            Genome genome = makeTestGenome();
            InnovationTracker tracker(200, 300);
            GenomeMutator mutator(seed);
            const bool result = mutator.mutateAddNode(genome, tracker, config);
            assert(result && "an eligible connection must always be found in makeTestGenome()");

            const ConnectionGene* preExistingDisabled = genome.findConnection(9, 101);
            assert(preExistingDisabled != nullptr && !preExistingDisabled->isEnabled() &&
                   preExistingDisabled->getWeight() == 0.2f &&
                   "a connection that was already disabled must never be selected or altered");

            for (const ConnectionGene& connection : genome.connections())
            {
                if (!connection.isEnabled() && connection.getInnovationNumber() != 2)
                {
                    splitOriginals.emplace_back(connection.getSourceId(), connection.getTargetId());
                }
            }
        }

        const bool sawMoreThanOneChoice =
            std::any_of(splitOriginals.begin(), splitOriginals.end(),
                        [&](const std::pair<NodeId, NodeId>& p) { return p != splitOriginals.front(); });
        assert(sawMoreThanOneChoice && "different seeds must be capable of selecting different eligible connections");
    }

    // 24: repeated calls on the same mutator advance its RNG state. A
    // single pair of consecutive calls could coincidentally land on the
    // same one of makeTestGenome()'s three eligible connections even with
    // an advancing RNG, so this drives ten consecutive calls from one
    // mutator and requires at least one of them to differ from the first.
    {
        MutationConfig config;
        config.addNodeProbability = 1.0f;
        GenomeMutator mutator(55u);

        std::vector<std::pair<NodeId, NodeId>> choices;
        for (int i = 0; i < 10; ++i)
        {
            Genome genome = makeTestGenome();
            InnovationTracker tracker(200, 300);
            mutator.mutateAddNode(genome, tracker, config);
            const ConnectionGene& last = genome.connections().back();
            choices.emplace_back(last.getSourceId(), last.getTargetId());
        }

        const bool sawVariation = std::any_of(choices.begin(), choices.end(),
                                               [&](const std::pair<NodeId, NodeId>& p) { return p != choices.front(); });
        assert(sawVariation && "repeated calls from the same mutator must advance its RNG state, not repeat the same draws");
    }

    // 27, 28 & 29: the same historical split, requested for two
    // structurally equivalent (but distinct) genomes sharing one
    // InnovationTracker, reuses exactly the same new node ID and the same
    // two connection innovations -- never renumbered.
    {
        InnovationTracker sharedTracker(50, 60);
        MutationConfig config;
        config.addNodeProbability = 1.0f;

        Genome genomeA = makeSimpleSplitGenome();
        GenomeMutator mutatorA(1u);
        assert(mutatorA.mutateAddNode(genomeA, sharedTracker, config) && "first genome's split must succeed");

        // NEW 6: genomeB's split reuses the historical record genomeA's
        // split already committed, so it must not advance the tracker any
        // further.
        const NodeId nodeCounterBeforeReuse = sharedTracker.getNextAvailableNodeId();
        const InnovationNumber innovationCounterBeforeReuse = sharedTracker.getNextAvailableInnovation();

        Genome genomeB = makeSimpleSplitGenome();
        GenomeMutator mutatorB(2u); // different seed, same single eligible connection either way
        assert(mutatorB.mutateAddNode(genomeB, sharedTracker, config) && "second genome's split must succeed");

        assert(sharedTracker.getNextAvailableNodeId() == nodeCounterBeforeReuse &&
               "reusing an already-recorded historical split must not advance the node ID counter"); // NEW 6
        assert(sharedTracker.getNextAvailableInnovation() == innovationCounterBeforeReuse &&
               "reusing an already-recorded historical split must not advance the innovation counter"); // NEW 6

        const NodeGene& newNodeA = genomeA.nodes().back();
        const NodeGene& newNodeB = genomeB.nodes().back();
        assert(newNodeA.getId() == newNodeB.getId() &&
               "the same historical split across two genomes must reuse the same new node ID"); // 27

        const ConnectionGene* incomingA = genomeA.findConnection(0, newNodeA.getId());
        const ConnectionGene* incomingB = genomeB.findConnection(0, newNodeB.getId());
        const ConnectionGene* outgoingA = genomeA.findConnection(newNodeA.getId(), 1);
        const ConnectionGene* outgoingB = genomeB.findConnection(newNodeB.getId(), 1);
        assert(incomingA != nullptr && incomingB != nullptr && outgoingA != nullptr && outgoingB != nullptr &&
               incomingA->getInnovationNumber() == incomingB->getInnovationNumber() &&
               outgoingA->getInnovationNumber() == outgoingB->getInnovationNumber() &&
               "the same historical split across two genomes must reuse the same two connection innovations"); // 28

        const NodeSplitInnovation split = sharedTracker.getNodeSplitInnovation(0, 0, 1);
        assert(newNodeA.getId() == split.newNodeId && newNodeB.getId() == split.newNodeId &&
               "the historical node ID must never be renumbered"); // 29
    }

    // 30: a returned new node ID that already exists in the genome as a
    // non-Hidden node is a structural inconsistency and must be rejected
    // clearly, not silently repaired or renumbered.
    {
        InnovationTracker tracker(5, 10); // will allocate node ID 5 for this split
        Genome genome;
        genome.addNode(NodeGene{0, NodeType::Input});
        genome.addNode(NodeGene{1, NodeType::Output});
        genome.addNode(NodeGene{5, NodeType::Input}); // occupies the ID the split would allocate
        genome.addConnection(ConnectionGene{0, 1, 0.5f, true, 0});

        MutationConfig config;
        config.addNodeProbability = 1.0f;
        GenomeMutator mutator(1u);

        assert(throwsInvalidArgument([&]() { mutator.mutateAddNode(genome, tracker, config); }) &&
               "a conflicting existing non-Hidden node at the split's new node ID must be rejected");
        // NEW 4: the throw is detected purely by inspection, before any
        // allocation, so it must leave the tracker completely unchanged.
        assert(tracker.getNextAvailableNodeId() == 5 && tracker.getNextAvailableInnovation() == 10 &&
               "a structural conflict must never alter the tracker's state");
    }

    // 31: an existing connection occupying one of the split's two directed
    // pairs, but with a different innovation number than InnovationTracker
    // just returned, is a structural inconsistency and must be rejected.
    {
        InnovationTracker tracker(5, 10);
        Genome genome;
        genome.addNode(NodeGene{0, NodeType::Input});
        genome.addNode(NodeGene{1, NodeType::Output});
        genome.addNode(NodeGene{5, NodeType::Hidden}); // correct type -- not a conflict by itself
        genome.addConnection(ConnectionGene{0, 1, 0.5f, true, 0});
        genome.addConnection(ConnectionGene{0, 5, 1.0f, true, 999}); // wrong innovation for source -> newNode

        MutationConfig config;
        config.addNodeProbability = 1.0f;
        GenomeMutator mutator(1u);

        assert(throwsInvalidArgument([&]() { mutator.mutateAddNode(genome, tracker, config); }) &&
               "an existing connection with a conflicting innovation number at the split's endpoints must be rejected");
        // NEW 4: same as above -- a structural conflict must never alter
        // the tracker's state, whether detected via a recorded split or
        // (as here) via the not-yet-recorded-split prediction path.
        assert(tracker.getNextAvailableNodeId() == 5 && tracker.getNextAvailableInnovation() == 10 &&
               "a structural conflict must never alter the tracker's state");
    }

    // 32: when the selected candidate's split already fully exists in the
    // genome, the mutation does not duplicate it -- it moves on to another
    // eligible candidate instead.
    {
        InnovationTracker tracker(50, 60);

        Genome genome;
        genome.addNode(NodeGene{0, NodeType::Input});
        genome.addNode(NodeGene{1, NodeType::Output});
        genome.addNode(NodeGene{2, NodeType::Output});
        genome.addConnection(ConnectionGene{0, 1, 0.5f, true, 0}); // connA: already "split" below
        genome.addConnection(ConnectionGene{0, 2, 0.9f, true, 1}); // connB: fresh, another eligible candidate

        // Pre-populate the genome with connA's split genes (as if this
        // exact structural event already happened), while deliberately
        // leaving connA itself enabled, so it remains an
        // eligible-but-unsuitable candidate. Note this also makes the two
        // new split connections themselves fresh, otherwise-splittable
        // enabled connections -- so besides connB, either of them is an
        // equally valid alternative candidate; the test only requires that
        // *some* valid alternative gets split, and that connA and its
        // existing split are left completely untouched.
        const NodeSplitInnovation splitA = tracker.getNodeSplitInnovation(0, 0, 1);
        genome.addNode(NodeGene{splitA.newNodeId, NodeType::Hidden});
        genome.addConnection(ConnectionGene{0, splitA.newNodeId, 1.0f, true, splitA.incomingInnovation});
        genome.addConnection(ConnectionGene{splitA.newNodeId, 1, 0.5f, true, splitA.outgoingInnovation});

        const std::size_t nodeCountBefore = genome.nodes().size();
        const std::size_t connCountBefore = genome.connections().size();

        MutationConfig config;
        config.addNodeProbability = 1.0f;

        // Try every starting seed so the search is exercised regardless of
        // which candidate is inspected first.
        for (std::uint32_t seed = 1; seed <= 10; ++seed)
        {
            Genome trial = genome;
            InnovationTracker trialTracker = tracker;
            GenomeMutator mutator(seed);

            const NodeId nodeCounterBeforeCall = trialTracker.getNextAvailableNodeId();
            const InnovationNumber innovationCounterBeforeCall = trialTracker.getNextAvailableInnovation();

            const bool result = mutator.mutateAddNode(trial, trialTracker, config);
            assert(result && "some other eligible connection must always remain available even if connA is inspected first and rejected");

            // NEW 3: regardless of whether connA (already split, unsuitable)
            // was inspected and skipped first, the tracker must advance by
            // exactly one node ID and exactly two innovations -- the cost
            // of the one candidate actually selected, never more.
            assert(trialTracker.getNextAvailableNodeId() == nodeCounterBeforeCall + 1 &&
                   "an inspected-but-skipped candidate must never itself allocate a node ID");
            assert(trialTracker.getNextAvailableInnovation() == innovationCounterBeforeCall + 2 &&
                   "an inspected-but-skipped candidate must never itself allocate an innovation number");

            assert(trial.nodes().size() == nodeCountBefore + 1 &&
                   "exactly one new node must be added, never a duplicate of connA's split");
            assert(trial.connections().size() == connCountBefore + 2 &&
                   "exactly two new connections must be added, never duplicates of connA's split");

            const ConnectionGene* connAAfter = trial.findConnection(0, 1);
            assert(connAAfter != nullptr && connAAfter->isEnabled() && connAAfter->getWeight() == 0.5f &&
                   connAAfter->getInnovationNumber() == 0 && "connA, already fully split, must be left completely untouched");

            const ConnectionGene* incomingAAfter = trial.findConnection(0, splitA.newNodeId);
            const ConnectionGene* outgoingAAfter = trial.findConnection(splitA.newNodeId, 1);
            assert(incomingAAfter != nullptr && incomingAAfter->getInnovationNumber() == splitA.incomingInnovation &&
                   outgoingAAfter != nullptr && outgoingAAfter->getInnovationNumber() == splitA.outgoingInnovation &&
                   "connA's existing split genes must not be duplicated or altered");
        }
    }

    // NEW 1 & NEW 2: a failed add-node mutation (every eligible candidate
    // turns out unsuitable) must never advance either tracker counter.
    {
        InnovationTracker tracker(50, 60);
        Genome genome;
        genome.addNode(NodeGene{0, NodeType::Input});
        genome.addNode(NodeGene{1, NodeType::Output});
        genome.addConnection(ConnectionGene{0, 1, 0.5f, true, 0});

        // The sole enabled connection's split already fully exists in the
        // genome, so it is unsuitable and no candidate can ever be
        // selected. The two pre-populated split connections are themselves
        // added disabled -- purely so they don't become additional
        // eligible (enabled) candidates in their own right, which would
        // let the mutation succeed via one of them instead of exercising
        // the "nothing eligible works" path this test targets.
        const NodeSplitInnovation split = tracker.getNodeSplitInnovation(0, 0, 1);
        genome.addNode(NodeGene{split.newNodeId, NodeType::Hidden});
        genome.addConnection(ConnectionGene{0, split.newNodeId, 1.0f, false, split.incomingInnovation});
        genome.addConnection(ConnectionGene{split.newNodeId, 1, 0.5f, false, split.outgoingInnovation});

        const NodeId nodeCounterBeforeAttempt = tracker.getNextAvailableNodeId();
        const InnovationNumber innovationCounterBeforeAttempt = tracker.getNextAvailableInnovation();

        MutationConfig config;
        config.addNodeProbability = 1.0f;
        GenomeMutator mutator(1u);

        const bool result = mutator.mutateAddNode(genome, tracker, config);
        assert(!result && "the only eligible candidate is already fully split; nothing else to try");
        assert(tracker.getNextAvailableNodeId() == nodeCounterBeforeAttempt &&
               "a failed add-node mutation must never advance the node ID counter"); // NEW 1
        assert(tracker.getNextAvailableInnovation() == innovationCounterBeforeAttempt &&
               "a failed add-node mutation must never advance the innovation counter"); // NEW 2
    }

    // 34 & 35: invalid addNodeProbability configuration is rejected, and
    // never touches the tracker.
    {
        Genome genome = makeSimpleSplitGenome();
        InnovationTracker tracker(50, 60);
        GenomeMutator mutator(1u);

        MutationConfig outOfRange;
        outOfRange.addNodeProbability = 1.5f;
        assert(throwsInvalidArgument([&]() { mutator.mutateAddNode(genome, tracker, outOfRange); }) &&
               "an addNodeProbability outside [0,1] must be rejected"); // 34

        MutationConfig negative;
        negative.addNodeProbability = -0.1f;
        assert(throwsInvalidArgument([&]() { mutator.mutateAddNode(genome, tracker, negative); }) &&
               "a negative addNodeProbability must be rejected"); // 34 (continued)

        MutationConfig nanProbability;
        nanProbability.addNodeProbability = std::numeric_limits<float>::quiet_NaN();
        assert(throwsInvalidArgument([&]() { mutator.mutateAddNode(genome, tracker, nanProbability); }) &&
               "a NaN addNodeProbability must be rejected"); // 35

        MutationConfig infiniteProbability;
        infiniteProbability.addNodeProbability = std::numeric_limits<float>::infinity();
        assert(throwsInvalidArgument([&]() { mutator.mutateAddNode(genome, tracker, infiniteProbability); }) &&
               "a non-finite addNodeProbability must be rejected"); // 35 (continued)

        assert(tracker.getNextAvailableNodeId() == 50 && tracker.getNextAvailableInnovation() == 60 &&
               "rejected configuration must never touch the tracker");
    }

    // 36: weight mutation behavior from Stage 9A remains unchanged.
    {
        Genome genome = makeTestGenome();
        const std::vector<ConnectionGene> before = genome.connections();
        MutationConfig config;
        config.weightMutationProbability = 0.0f;
        GenomeMutator mutator(1u);
        mutator.mutateWeights(genome, config);
        for (std::size_t i = 0; i < before.size(); ++i)
        {
            assert(genome.connections()[i].getWeight() == before[i].getWeight() &&
                   "Stage 9A's mutateWeights() behavior must remain unchanged");
        }
    }

    // 37: add-connection mutation behavior from Stage 9C remains unchanged.
    {
        Genome genome = makeTestGenome();
        InnovationTracker tracker(200, 300);
        MutationConfig config;
        config.addConnectionProbability = 0.0f;
        GenomeMutator mutator(1u);
        const bool added = mutator.mutateAddConnection(genome, tracker, config);
        assert(!added &&
               "Stage 9C's mutateAddConnection() behavior must remain unchanged: probability 0 still adds nothing");
    }

    // 38 & 39: no crossover or population behavior exists to exercise -- no
    // such API is called anywhere above or anywhere else in the codebase.

    // 40: all previous verification suites still pass -- enforced by
    // main() continuing to call every earlier verify*() function
    // unchanged.

    TraceLog(LOG_INFO, "Add-node mutation verification: all deterministic checks passed");
}

namespace genome_crossover_verify
{

// 9 Input (0..8) + 1 Bias (9) + 2 Output (100 steering, 101 throttle), no
// connections -- the minimal interface every phenotype-buildable genome in
// this suite needs, matching ai::NeuralNetwork's fixed Input/Bias/Output
// counts. Every test that expects crossover() to *succeed* builds on a copy
// of this (both parents keep the same interface, so the child automatically
// inherits it regardless of which connections are inherited); tests that
// expect crossover() to throw don't need it, since the throw happens before
// buildPhenotype() is ever reached.
ai::neat::Genome makeInterfaceGenome()
{
    using ai::neat::Genome;
    using ai::neat::NodeGene;
    using ai::neat::NodeType;

    Genome genome;
    for (int i = 0; i < ai::NeuralNetwork::kInputCount; ++i)
    {
        genome.addNode(NodeGene{i, NodeType::Input});
    }
    genome.addNode(NodeGene{9, NodeType::Bias});
    genome.addNode(NodeGene{100, NodeType::Output});
    genome.addNode(NodeGene{101, NodeType::Output});
    return genome;
}

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

ai::Observation makeObservation(int index, float value)
{
    ai::Observation obs;
    obs.values.fill(0.0f);
    if (index >= 0)
    {
        obs.values[index] = value;
    }
    return obs;
}

} // namespace genome_crossover_verify

// One-shot, deterministic sanity check of ai::neat::GenomeCrossover, covering
// deterministic NEAT crossover end to end. Independent of
// Car/Track/AI/keyboard/render timing. Runs once at startup. No
// compatibility distance, species, population, selection, reproduction,
// generation, or training logic is exercised here -- only crossover, per
// Stage 10's scope.
void verifyGenomeCrossover()
{
    using namespace genome_crossover_verify;
    using ai::neat::ConnectionGene;
    using ai::neat::CrossoverConfig;
    using ai::neat::Genome;
    using ai::neat::GenomeCrossover;
    using ai::neat::NodeGene;
    using ai::neat::NodeType;

    // 1 & 2: invalid/non-finite config probabilities are rejected.
    {
        Genome a = makeInterfaceGenome();
        Genome b = makeInterfaceGenome();
        GenomeCrossover crossover(1u);

        CrossoverConfig outOfRange;
        outOfRange.matchingGeneChooseParentAProbability = 1.5f;
        assert(throwsInvalidArgument([&]() { crossover.crossover(a, 1.0f, b, 1.0f, outOfRange); }) &&
               "an out-of-range matchingGeneChooseParentAProbability must be rejected"); // 1

        CrossoverConfig negative;
        negative.disabledGeneRemainDisabledProbability = -0.1f;
        assert(throwsInvalidArgument([&]() { crossover.crossover(a, 1.0f, b, 1.0f, negative); }) &&
               "a negative disabledGeneRemainDisabledProbability must be rejected"); // 1 (continued)

        CrossoverConfig nanConfig;
        nanConfig.matchingGeneChooseParentAProbability = std::numeric_limits<float>::quiet_NaN();
        assert(throwsInvalidArgument([&]() { crossover.crossover(a, 1.0f, b, 1.0f, nanConfig); }) &&
               "a NaN config probability must be rejected"); // 2

        CrossoverConfig infConfig;
        infConfig.disabledGeneRemainDisabledProbability = std::numeric_limits<float>::infinity();
        assert(throwsInvalidArgument([&]() { crossover.crossover(a, 1.0f, b, 1.0f, infConfig); }) &&
               "a non-finite config probability must be rejected"); // 2 (continued)
    }

    // 3: non-finite parent fitness is rejected.
    {
        Genome a = makeInterfaceGenome();
        Genome b = makeInterfaceGenome();
        GenomeCrossover crossover(1u);
        CrossoverConfig config;

        assert(throwsInvalidArgument([&]() { crossover.crossover(a, std::numeric_limits<float>::quiet_NaN(), b, 1.0f, config); }) &&
               "a NaN fitnessA must be rejected");
        assert(throwsInvalidArgument([&]() { crossover.crossover(a, 1.0f, b, std::numeric_limits<float>::infinity(), config); }) &&
               "a non-finite fitnessB must be rejected");
    }

    // 4: empty (interface-only) compatible parents produce a valid,
    // interface-appropriate child -- all mandatory nodes, no connections,
    // still buildPhenotype()-able (a fully disconnected network is valid).
    {
        Genome a = makeInterfaceGenome();
        Genome b = makeInterfaceGenome();
        GenomeCrossover crossover(1u);
        CrossoverConfig config;

        const Genome child = crossover.crossover(a, 1.0f, b, 1.0f, config);
        assert(child.connections().empty() && "no connections in either parent must yield no connections in the child");
        assert(child.nodes().size() == 12 && "the child must contain exactly the 9 Input + 1 Bias + 2 Output nodes");

        child.validate();
        ai::NeuralNetwork net = ai::neat::buildPhenotype(child);
        const auto out = net.evaluate(makeObservation(-1, 0.0f));
        assert(out[0] == 0.0f && out[1] == 0.0f && "a fully disconnected child phenotype must evaluate to exactly 0");
    }

    // 5: matching genes align by innovation number, not vector index --
    // parent A stores innovation 5 before innovation 2, parent B stores
    // innovation 2 before innovation 5.
    {
        Genome parentA = makeInterfaceGenome();
        parentA.addConnection(ConnectionGene{1, 100, 0.33f, true, 5});
        parentA.addConnection(ConnectionGene{0, 100, 0.11f, true, 2});

        Genome parentB = makeInterfaceGenome();
        parentB.addConnection(ConnectionGene{0, 100, 0.22f, true, 2});
        parentB.addConnection(ConnectionGene{1, 100, 0.44f, true, 5});

        GenomeCrossover crossover(1u);
        CrossoverConfig config;
        config.matchingGeneChooseParentAProbability = 1.0f; // always A, so the result is fully deterministic here

        const Genome child = crossover.crossover(parentA, 1.0f, parentB, 1.0f, config);
        const ConnectionGene* innov2 = child.findConnection(0, 100);
        const ConnectionGene* innov5 = child.findConnection(1, 100);
        assert(innov2 != nullptr && innov2->getInnovationNumber() == 2 && innov2->getWeight() == 0.11f &&
               "innovation 2 must be matched to innovation 2, regardless of vector position");
        assert(innov5 != nullptr && innov5->getInnovationNumber() == 5 && innov5->getWeight() == 0.33f &&
               "innovation 5 must be matched to innovation 5, regardless of vector position");
    }

    // 6: matching gene can inherit parent A's weight.
    {
        Genome parentA = makeInterfaceGenome();
        parentA.addConnection(ConnectionGene{0, 100, 0.11f, true, 0});
        Genome parentB = makeInterfaceGenome();
        parentB.addConnection(ConnectionGene{0, 100, 0.99f, true, 0});

        GenomeCrossover crossover(1u);
        CrossoverConfig config;
        config.matchingGeneChooseParentAProbability = 1.0f;
        const Genome child = crossover.crossover(parentA, 1.0f, parentB, 1.0f, config);
        assert(child.findConnection(0, 100)->getWeight() == 0.11f &&
               "probability 1.0 must always inherit parent A's weight for a matching gene");
    }

    // 7: matching gene can inherit parent B's weight.
    {
        Genome parentA = makeInterfaceGenome();
        parentA.addConnection(ConnectionGene{0, 100, 0.11f, true, 0});
        Genome parentB = makeInterfaceGenome();
        parentB.addConnection(ConnectionGene{0, 100, 0.99f, true, 0});

        GenomeCrossover crossover(1u);
        CrossoverConfig config;
        config.matchingGeneChooseParentAProbability = 0.0f;
        const Genome child = crossover.crossover(parentA, 1.0f, parentB, 1.0f, config);
        assert(child.findConnection(0, 100)->getWeight() == 0.99f &&
               "probability 0.0 must always inherit parent B's weight for a matching gene");
    }

    // 8: matching innovation with conflicting endpoints between parents is
    // rejected.
    {
        Genome parentA = makeInterfaceGenome();
        parentA.addConnection(ConnectionGene{0, 100, 0.5f, true, 7});
        Genome parentB = makeInterfaceGenome();
        parentB.addConnection(ConnectionGene{1, 100, 0.5f, true, 7}); // same innovation, different source

        GenomeCrossover crossover(1u);
        CrossoverConfig config;
        assert(throwsInvalidArgument([&]() { crossover.crossover(parentA, 1.0f, parentB, 1.0f, config); }) &&
               "a matching innovation with conflicting endpoints between parents must be rejected");
    }

    // 9: a matching gene enabled in both parents remains enabled --
    // deliberately using probability 1.0 (which would force *disabled* if
    // the both-enabled shortcut were broken and it fell through to the
    // probability branch).
    {
        Genome parentA = makeInterfaceGenome();
        parentA.addConnection(ConnectionGene{0, 100, 0.5f, true, 0});
        Genome parentB = makeInterfaceGenome();
        parentB.addConnection(ConnectionGene{0, 100, 0.6f, true, 0});

        GenomeCrossover crossover(1u);
        CrossoverConfig config;
        config.disabledGeneRemainDisabledProbability = 1.0f;
        const Genome child = crossover.crossover(parentA, 1.0f, parentB, 1.0f, config);
        assert(child.findConnection(0, 100)->isEnabled() && "a matching gene enabled in both parents must remain enabled");
    }

    // 10: a matching gene disabled in one parent can remain disabled.
    {
        Genome parentA = makeInterfaceGenome();
        parentA.addConnection(ConnectionGene{0, 100, 0.5f, false, 0});
        Genome parentB = makeInterfaceGenome();
        parentB.addConnection(ConnectionGene{0, 100, 0.6f, true, 0});

        GenomeCrossover crossover(1u);
        CrossoverConfig config;
        config.disabledGeneRemainDisabledProbability = 1.0f;
        const Genome child = crossover.crossover(parentA, 1.0f, parentB, 1.0f, config);
        assert(!child.findConnection(0, 100)->isEnabled() &&
               "probability 1.0 must keep a partially-disabled matching gene disabled");
    }

    // 11: a matching gene disabled in one parent can become enabled.
    {
        Genome parentA = makeInterfaceGenome();
        parentA.addConnection(ConnectionGene{0, 100, 0.5f, false, 0});
        Genome parentB = makeInterfaceGenome();
        parentB.addConnection(ConnectionGene{0, 100, 0.6f, true, 0});

        GenomeCrossover crossover(1u);
        CrossoverConfig config;
        config.disabledGeneRemainDisabledProbability = 0.0f;
        const Genome child = crossover.crossover(parentA, 1.0f, parentB, 1.0f, config);
        assert(child.findConnection(0, 100)->isEnabled() &&
               "probability 0.0 must always enable a partially-disabled matching gene");
    }

    // 12: a matching gene disabled in both parents follows the same
    // disabled-gene probability rule.
    {
        Genome parentA = makeInterfaceGenome();
        parentA.addConnection(ConnectionGene{0, 100, 0.5f, false, 0});
        Genome parentB = makeInterfaceGenome();
        parentB.addConnection(ConnectionGene{0, 100, 0.6f, false, 0});

        CrossoverConfig config;

        GenomeCrossover crossoverStaysDisabled(1u);
        config.disabledGeneRemainDisabledProbability = 1.0f;
        const Genome childStaysDisabled = crossoverStaysDisabled.crossover(parentA, 1.0f, parentB, 1.0f, config);
        assert(!childStaysDisabled.findConnection(0, 100)->isEnabled() &&
               "probability 1.0 must keep a both-disabled matching gene disabled");

        GenomeCrossover crossoverBecomesEnabled(1u);
        config.disabledGeneRemainDisabledProbability = 0.0f;
        const Genome childBecomesEnabled = crossoverBecomesEnabled.crossover(parentA, 1.0f, parentB, 1.0f, config);
        assert(childBecomesEnabled.findConnection(0, 100)->isEnabled() &&
               "probability 0.0 must enable a both-disabled matching gene");
    }

    // 13 & 14: the fitter parent A's non-matching genes are inherited; the
    // less-fit parent B's non-matching genes are excluded.
    {
        Genome parentA = makeInterfaceGenome();
        parentA.addConnection(ConnectionGene{2, 100, 0.5f, true, 10});
        Genome parentB = makeInterfaceGenome();
        parentB.addConnection(ConnectionGene{3, 101, 0.6f, true, 11});

        GenomeCrossover crossover(1u);
        CrossoverConfig config;
        const Genome child = crossover.crossover(parentA, 10.0f, parentB, 5.0f, config); // A fitter

        assert(child.hasConnection(2, 100) && child.findConnection(2, 100)->getInnovationNumber() == 10 &&
               "the fitter parent A's non-matching gene must be inherited"); // 13
        assert(!child.hasConnection(3, 101) && "the less-fit parent B's non-matching gene must be excluded"); // 14
    }

    // 15 & 16: symmetric case -- fitter parent B's non-matching genes are
    // inherited; less-fit parent A's non-matching genes are excluded.
    {
        Genome parentA = makeInterfaceGenome();
        parentA.addConnection(ConnectionGene{2, 100, 0.5f, true, 10});
        Genome parentB = makeInterfaceGenome();
        parentB.addConnection(ConnectionGene{3, 101, 0.6f, true, 11});

        GenomeCrossover crossover(1u);
        CrossoverConfig config;
        const Genome child = crossover.crossover(parentA, 5.0f, parentB, 10.0f, config); // B fitter

        assert(child.hasConnection(3, 101) && child.findConnection(3, 101)->getInnovationNumber() == 11 &&
               "the fitter parent B's non-matching gene must be inherited"); // 15
        assert(!child.hasConnection(2, 100) && "the less-fit parent A's non-matching gene must be excluded"); // 16
    }

    // 17: with equal fitness, non-matching genes from *both* parents may
    // appear in the child -- looped over enough seeds that both must be
    // observed at least once.
    {
        Genome parentA = makeInterfaceGenome();
        parentA.addConnection(ConnectionGene{2, 100, 0.5f, true, 20});
        Genome parentB = makeInterfaceGenome();
        parentB.addConnection(ConnectionGene{3, 101, 0.6f, true, 21});

        CrossoverConfig config;
        bool sawA = false;
        bool sawB = false;
        for (std::uint32_t seed = 1; seed <= 20; ++seed)
        {
            GenomeCrossover crossover(seed);
            const Genome child = crossover.crossover(parentA, 10.0f, parentB, 10.0f, config); // equal fitness
            if (child.hasConnection(2, 100))
            {
                sawA = true;
            }
            if (child.hasConnection(3, 101))
            {
                sawB = true;
            }
        }
        assert(sawA && sawB && "equal-fitness non-matching genes from both parents must each be inheritable");
    }

    // 18: an equal-fitness candidate that would create a duplicate directed
    // connection is skipped rather than causing an invalid child -- two
    // non-matching genes from different parents target the exact same
    // (source, target) pair.
    {
        Genome parentA = makeInterfaceGenome();
        parentA.addNode(NodeGene{50, NodeType::Hidden});
        parentA.addConnection(ConnectionGene{0, 50, 0.5f, true, 20});
        Genome parentB = makeInterfaceGenome();
        parentB.addNode(NodeGene{50, NodeType::Hidden});
        parentB.addConnection(ConnectionGene{0, 50, 0.6f, true, 21}); // same (source, target), different innovation

        CrossoverConfig config;
        for (std::uint32_t seed = 1; seed <= 30; ++seed)
        {
            GenomeCrossover crossover(seed);
            const Genome child = crossover.crossover(parentA, 10.0f, parentB, 10.0f, config); // must never throw
            const ConnectionGene* found = child.findConnection(0, 50);
            assert((found == nullptr || found->getInnovationNumber() == 20 || found->getInnovationNumber() == 21) &&
                   "at most one of the two conflicting candidates may end up in the child");
        }
    }

    // 19: an equal-fitness candidate that would create an enabled cycle is
    // skipped -- a matching 50->51 edge is always present, and a
    // non-matching 51->50 candidate would close a 2-cycle if included.
    {
        Genome parentA = makeInterfaceGenome();
        parentA.addNode(NodeGene{50, NodeType::Hidden});
        parentA.addNode(NodeGene{51, NodeType::Hidden});
        parentA.addConnection(ConnectionGene{50, 51, 0.5f, true, 30}); // matching
        parentA.addConnection(ConnectionGene{51, 50, 0.7f, true, 31}); // A-only, would close a cycle

        Genome parentB = makeInterfaceGenome();
        parentB.addNode(NodeGene{50, NodeType::Hidden});
        parentB.addNode(NodeGene{51, NodeType::Hidden});
        parentB.addConnection(ConnectionGene{50, 51, 0.5f, true, 30}); // matching

        CrossoverConfig config;
        for (std::uint32_t seed = 1; seed <= 30; ++seed)
        {
            GenomeCrossover crossover(seed);
            const Genome child = crossover.crossover(parentA, 10.0f, parentB, 10.0f, config); // equal fitness, must never throw
            assert(!child.hasConnection(51, 50) && "a candidate that would close an enabled cycle must always be skipped");
        }
    }

    // 20, 21 & 22: the required Input/Bias/Output nodes are preserved
    // regardless of connections.
    {
        Genome parentA = makeInterfaceGenome();
        Genome parentB = makeInterfaceGenome();
        GenomeCrossover crossover(1u);
        CrossoverConfig config;
        const Genome child = crossover.crossover(parentA, 1.0f, parentB, 1.0f, config);

        for (int i = 0; i < ai::NeuralNetwork::kInputCount; ++i)
        {
            assert(child.hasNode(i) && child.findNode(i)->getType() == NodeType::Input &&
                   "every Input node must be preserved"); // 20
        }
        assert(child.hasNode(9) && child.findNode(9)->getType() == NodeType::Bias &&
               "the Bias node must be preserved"); // 21
        assert(child.hasNode(100) && child.findNode(100)->getType() == NodeType::Output && child.hasNode(101) &&
               child.findNode(101)->getType() == NodeType::Output && "every Output node must be preserved"); // 22
    }

    // 23: a Hidden node referenced by an inherited connection is preserved.
    {
        Genome parentA = makeInterfaceGenome();
        parentA.addNode(NodeGene{50, NodeType::Hidden});
        parentA.addConnection(ConnectionGene{0, 50, 0.5f, true, 10}); // A-only, references Hidden 50
        Genome parentB = makeInterfaceGenome();

        GenomeCrossover crossover(1u);
        CrossoverConfig config;
        const Genome child = crossover.crossover(parentA, 10.0f, parentB, 5.0f, config); // A fitter
        assert(child.hasNode(50) && child.findNode(50)->getType() == NodeType::Hidden &&
               "a Hidden node referenced by an inherited connection must be preserved");
    }

    // 24: an unreferenced Hidden node need not be preserved.
    {
        Genome parentA = makeInterfaceGenome();
        parentA.addNode(NodeGene{60, NodeType::Hidden}); // never referenced by any connection
        Genome parentB = makeInterfaceGenome();

        GenomeCrossover crossover(1u);
        CrossoverConfig config;
        const Genome child = crossover.crossover(parentA, 10.0f, parentB, 5.0f, config);
        assert(!child.hasNode(60) && "an unreferenced Hidden node need not be preserved");
    }

    // 25: the same NodeId with a conflicting NodeType between parents is
    // rejected.
    {
        Genome parentA = makeInterfaceGenome();
        parentA.addNode(NodeGene{50, NodeType::Hidden});
        Genome parentB = makeInterfaceGenome();
        parentB.addNode(NodeGene{50, NodeType::Output}); // conflicting type for the same ID

        GenomeCrossover crossover(1u);
        CrossoverConfig config;
        assert(throwsInvalidArgument([&]() { crossover.crossover(parentA, 1.0f, parentB, 1.0f, config); }) &&
               "the same NodeId with a conflicting NodeType between parents must be rejected");
    }

    // 26: node IDs are never renumbered, even when large/non-contiguous.
    {
        Genome parentA;
        for (int i = 0; i < ai::NeuralNetwork::kInputCount; ++i)
        {
            parentA.addNode(NodeGene{1000 + i, NodeType::Input});
        }
        parentA.addNode(NodeGene{2000, NodeType::Bias});
        parentA.addNode(NodeGene{3000, NodeType::Output});
        parentA.addNode(NodeGene{3001, NodeType::Output});
        parentA.addConnection(ConnectionGene{1000, 3000, 0.5f, true, 0});

        Genome parentB;
        for (int i = 0; i < ai::NeuralNetwork::kInputCount; ++i)
        {
            parentB.addNode(NodeGene{1000 + i, NodeType::Input});
        }
        parentB.addNode(NodeGene{2000, NodeType::Bias});
        parentB.addNode(NodeGene{3000, NodeType::Output});
        parentB.addNode(NodeGene{3001, NodeType::Output});
        parentB.addConnection(ConnectionGene{1000, 3000, 0.9f, true, 0}); // matching, different weight

        GenomeCrossover crossover(1u);
        CrossoverConfig config;
        const Genome child = crossover.crossover(parentA, 1.0f, parentB, 1.0f, config);
        assert(child.hasNode(1000) && child.hasNode(1008) && child.hasNode(2000) && child.hasNode(3000) &&
               child.hasNode(3001) && "node IDs must be preserved exactly, never renumbered");
        assert(child.findConnection(1000, 3000) != nullptr && child.findConnection(1000, 3000)->getInnovationNumber() == 0 &&
               "connection endpoints must reference the original, non-renumbered node IDs");
    }

    // 27: connection innovation numbers are preserved exactly.
    {
        Genome parentA = makeInterfaceGenome();
        parentA.addConnection(ConnectionGene{0, 100, 0.5f, true, 42});
        Genome parentB = makeInterfaceGenome();
        parentB.addConnection(ConnectionGene{0, 100, 0.6f, true, 42});

        GenomeCrossover crossover(1u);
        CrossoverConfig config;
        const Genome child = crossover.crossover(parentA, 1.0f, parentB, 1.0f, config);
        assert(child.findConnection(0, 100) != nullptr && child.findConnection(0, 100)->getInnovationNumber() == 42 &&
               "connection innovation numbers must be preserved exactly");
    }

    // 28 & 29: child nodes are stored ascending by node ID, and child
    // connections ascending by innovation number, regardless of either
    // parent's own storage order.
    {
        Genome parentA;
        parentA.addNode(NodeGene{9, NodeType::Bias});
        parentA.addNode(NodeGene{101, NodeType::Output});
        parentA.addNode(NodeGene{100, NodeType::Output});
        for (int i = ai::NeuralNetwork::kInputCount - 1; i >= 0; --i)
        {
            parentA.addNode(NodeGene{i, NodeType::Input});
        }
        parentA.addConnection(ConnectionGene{0, 101, 0.1f, true, 9});
        parentA.addConnection(ConnectionGene{1, 100, 0.2f, true, 3});

        Genome parentB = parentA; // identical structure -- every gene matches itself

        GenomeCrossover crossover(1u);
        CrossoverConfig config;
        const Genome child = crossover.crossover(parentA, 1.0f, parentB, 1.0f, config);

        for (std::size_t i = 1; i < child.nodes().size(); ++i)
        {
            assert(child.nodes()[i - 1].getId() < child.nodes()[i].getId() &&
                   "child nodes must be stored ascending by node ID"); // 28
        }
        for (std::size_t i = 1; i < child.connections().size(); ++i)
        {
            assert(child.connections()[i - 1].getInnovationNumber() < child.connections()[i].getInnovationNumber() &&
                   "child connections must be stored ascending by innovation number"); // 29
        }
    }

    // 30: duplicate connection innovation numbers inside parent A are
    // rejected (Genome::validate() alone would not catch this, since the
    // two connections use different (source, target) pairs).
    {
        std::vector<NodeGene> nodes = {NodeGene{0, NodeType::Input}, NodeGene{1, NodeType::Input},
                                        NodeGene{100, NodeType::Output}, NodeGene{101, NodeType::Output}};
        std::vector<ConnectionGene> connections = {ConnectionGene{0, 100, 0.1f, true, 5},
                                                    ConnectionGene{1, 101, 0.2f, true, 5}};
        Genome parentAWithDuplicateInnovation(nodes, connections);
        parentAWithDuplicateInnovation.validate(); // sanity: Genome::validate() itself does not reject this
        Genome parentB;

        GenomeCrossover crossover(1u);
        CrossoverConfig config;
        assert(throwsInvalidArgument(
                   [&]() { crossover.crossover(parentAWithDuplicateInnovation, 1.0f, parentB, 1.0f, config); }) &&
               "parent A with duplicate connection innovation numbers must be rejected");
    }

    // 31: symmetric case for parent B.
    {
        std::vector<NodeGene> nodes = {NodeGene{0, NodeType::Input}, NodeGene{1, NodeType::Input},
                                        NodeGene{100, NodeType::Output}, NodeGene{101, NodeType::Output}};
        std::vector<ConnectionGene> connections = {ConnectionGene{0, 100, 0.1f, true, 5},
                                                    ConnectionGene{1, 101, 0.2f, true, 5}};
        Genome parentBWithDuplicateInnovation(nodes, connections);
        Genome parentA;

        GenomeCrossover crossover(1u);
        CrossoverConfig config;
        assert(throwsInvalidArgument(
                   [&]() { crossover.crossover(parentA, 1.0f, parentBWithDuplicateInnovation, 1.0f, config); }) &&
               "parent B with duplicate connection innovation numbers must be rejected");
    }

    // 32 & 33: child.validate() and buildPhenotype() succeed for a
    // representative mixed scenario (matching genes plus fitter-only
    // genes together, including a Hidden node).
    {
        Genome parentA = makeInterfaceGenome();
        parentA.addNode(NodeGene{50, NodeType::Hidden});
        parentA.addConnection(ConnectionGene{0, 100, 0.5f, true, 0});   // matching
        parentA.addConnection(ConnectionGene{1, 50, 0.3f, true, 10});   // A-only
        parentA.addConnection(ConnectionGene{50, 101, 0.4f, true, 11}); // A-only

        Genome parentB = makeInterfaceGenome();
        parentB.addNode(NodeGene{50, NodeType::Hidden});
        parentB.addConnection(ConnectionGene{0, 100, 0.9f, true, 0}); // matching

        GenomeCrossover crossover(1u);
        CrossoverConfig config;
        const Genome child = crossover.crossover(parentA, 10.0f, parentB, 5.0f, config); // A fitter

        child.validate();                                      // 32: must not throw
        ai::NeuralNetwork net = ai::neat::buildPhenotype(child); // 33: must not throw
        (void)net;
    }

    // 34: the crossover child's phenotype evaluates deterministically.
    {
        Genome parentA = makeInterfaceGenome();
        parentA.addConnection(ConnectionGene{0, 100, 0.5f, true, 0});
        Genome parentB = makeInterfaceGenome();
        parentB.addConnection(ConnectionGene{0, 100, 0.5f, true, 0});

        GenomeCrossover crossover(1u);
        CrossoverConfig config;
        const Genome child = crossover.crossover(parentA, 1.0f, parentB, 1.0f, config);

        ai::NeuralNetwork netA = ai::neat::buildPhenotype(child);
        ai::NeuralNetwork netB = ai::neat::buildPhenotype(child);
        const auto obs = makeObservation(0, 1.0f);
        const auto outA = netA.evaluate(obs);
        const auto outB = netB.evaluate(obs);
        assert(outA[0] == outB[0] && outA[1] == outB[1] &&
               "the phenotype built from a crossover child must evaluate deterministically");
    }

    // 35: neither parent is ever modified by crossover(), even across the
    // fullest path (matching + equal-fitness non-matching genes).
    {
        Genome parentA = makeInterfaceGenome();
        parentA.addNode(NodeGene{50, NodeType::Hidden});
        parentA.addConnection(ConnectionGene{0, 100, 0.5f, true, 0}); // matching
        parentA.addConnection(ConnectionGene{1, 50, 0.3f, true, 10}); // A-only

        Genome parentB = makeInterfaceGenome();
        parentB.addNode(NodeGene{50, NodeType::Hidden});
        parentB.addConnection(ConnectionGene{0, 100, 0.9f, true, 0}); // matching, different weight
        parentB.addConnection(ConnectionGene{2, 50, 0.4f, true, 11}); // B-only

        const std::vector<NodeGene> nodesABefore = parentA.nodes();
        const std::vector<ConnectionGene> connectionsABefore = parentA.connections();
        const std::vector<NodeGene> nodesBBefore = parentB.nodes();
        const std::vector<ConnectionGene> connectionsBBefore = parentB.connections();

        GenomeCrossover crossover(1u);
        CrossoverConfig config;
        crossover.crossover(parentA, 10.0f, parentB, 10.0f, config); // equal fitness

        assert(parentA.nodes().size() == nodesABefore.size() && parentA.connections().size() == connectionsABefore.size() &&
               "crossover must not add or remove parent A's genes");
        for (std::size_t i = 0; i < nodesABefore.size(); ++i)
        {
            assert(parentA.nodes()[i] == nodesABefore[i] && "crossover must not modify parent A's nodes");
        }
        for (std::size_t i = 0; i < connectionsABefore.size(); ++i)
        {
            const ConnectionGene& before = connectionsABefore[i];
            const ConnectionGene& after = parentA.connections()[i];
            assert(after.getSourceId() == before.getSourceId() && after.getTargetId() == before.getTargetId() &&
                   after.getWeight() == before.getWeight() && after.isEnabled() == before.isEnabled() &&
                   after.getInnovationNumber() == before.getInnovationNumber() &&
                   "crossover must not modify parent A's connections");
        }

        assert(parentB.nodes().size() == nodesBBefore.size() && parentB.connections().size() == connectionsBBefore.size() &&
               "crossover must not add or remove parent B's genes");
        for (std::size_t i = 0; i < nodesBBefore.size(); ++i)
        {
            assert(parentB.nodes()[i] == nodesBBefore[i] && "crossover must not modify parent B's nodes");
        }
        for (std::size_t i = 0; i < connectionsBBefore.size(); ++i)
        {
            const ConnectionGene& before = connectionsBBefore[i];
            const ConnectionGene& after = parentB.connections()[i];
            assert(after.getSourceId() == before.getSourceId() && after.getTargetId() == before.getTargetId() &&
                   after.getWeight() == before.getWeight() && after.isEnabled() == before.isEnabled() &&
                   after.getInnovationNumber() == before.getInnovationNumber() &&
                   "crossover must not modify parent B's connections");
        }
    }

    // 36: the same seed, parents, fitnesses, and config produce an
    // identical child.
    {
        Genome parentA = makeInterfaceGenome();
        parentA.addConnection(ConnectionGene{0, 100, 0.5f, true, 0});
        parentA.addConnection(ConnectionGene{1, 100, 0.3f, true, 5});
        Genome parentB = makeInterfaceGenome();
        parentB.addConnection(ConnectionGene{0, 100, 0.9f, true, 0}); // matching, different weight
        parentB.addConnection(ConnectionGene{2, 101, 0.4f, true, 6}); // B-only

        CrossoverConfig config;

        GenomeCrossover crossoverX(777u);
        const Genome childX = crossoverX.crossover(parentA, 10.0f, parentB, 10.0f, config); // equal fitness

        GenomeCrossover crossoverY(777u);
        const Genome childY = crossoverY.crossover(parentA, 10.0f, parentB, 10.0f, config);

        assert(childX.nodes().size() == childY.nodes().size() &&
               childX.connections().size() == childY.connections().size() &&
               "identical seed/parents/fitness/config must produce a structurally identical child");
        for (std::size_t i = 0; i < childX.nodes().size(); ++i)
        {
            assert(childX.nodes()[i] == childY.nodes()[i] && "identical runs must produce identical child nodes");
        }
        for (std::size_t i = 0; i < childX.connections().size(); ++i)
        {
            const ConnectionGene& x = childX.connections()[i];
            const ConnectionGene& y = childY.connections()[i];
            assert(x.getSourceId() == y.getSourceId() && x.getTargetId() == y.getTargetId() &&
                   x.getWeight() == y.getWeight() && x.isEnabled() == y.isEnabled() &&
                   x.getInnovationNumber() == y.getInnovationNumber() &&
                   "identical runs must produce identical child connections");
        }
    }

    // 37: different seeds can select a different parent's copy for a
    // matching gene.
    {
        Genome parentA = makeInterfaceGenome();
        parentA.addConnection(ConnectionGene{0, 100, 0.11f, true, 0});
        Genome parentB = makeInterfaceGenome();
        parentB.addConnection(ConnectionGene{0, 100, 0.99f, true, 0});

        CrossoverConfig config; // default matchingGeneChooseParentAProbability = 0.5

        std::vector<float> weightsSeen;
        for (std::uint32_t seed = 1; seed <= 10; ++seed)
        {
            GenomeCrossover crossover(seed);
            const Genome child = crossover.crossover(parentA, 1.0f, parentB, 1.0f, config);
            weightsSeen.push_back(child.findConnection(0, 100)->getWeight());
        }
        const bool sawVariation =
            std::any_of(weightsSeen.begin(), weightsSeen.end(), [&](float w) { return w != weightsSeen.front(); });
        assert(sawVariation && "different seeds must be capable of selecting a different parent's copy for a matching gene");
    }

    // 38: repeated calls on the same GenomeCrossover instance advance its
    // RNG state.
    {
        Genome parentA = makeInterfaceGenome();
        parentA.addConnection(ConnectionGene{0, 100, 0.11f, true, 0});
        Genome parentB = makeInterfaceGenome();
        parentB.addConnection(ConnectionGene{0, 100, 0.99f, true, 0});

        CrossoverConfig config;
        GenomeCrossover crossover(55u);

        std::vector<float> weightsSeen;
        for (int i = 0; i < 10; ++i)
        {
            const Genome child = crossover.crossover(parentA, 1.0f, parentB, 1.0f, config);
            weightsSeen.push_back(child.findConnection(0, 100)->getWeight());
        }
        const bool sawVariation =
            std::any_of(weightsSeen.begin(), weightsSeen.end(), [&](float w) { return w != weightsSeen.front(); });
        assert(sawVariation && "repeated calls from the same GenomeCrossover instance must advance its RNG state");
    }

    // 39: no InnovationTracker state is involved -- GenomeCrossover.h/.cpp
    // never include or reference InnovationTracker anywhere, and
    // crossover()'s signature takes no tracker parameter, so no node ID or
    // innovation number handed to it can ever be consulted or advanced by
    // this stage.

    // 40: no mutation is performed during crossover -- reinforced by 35
    // above (both parents provably unchanged); GenomeMutator is never
    // referenced by GenomeCrossover.h/.cpp.

    // 41: no population/species logic exists -- crossover() takes exactly
    // two parent Genomes and two fitness floats, nothing resembling a
    // population, species, or selection concept exists anywhere in this
    // stage's files.

    // 42: all previous verification suites still pass -- enforced by
    // main() continuing to call every earlier verify*() function
    // unchanged.

    TraceLog(LOG_INFO, "Genome crossover verification: all deterministic checks passed");
}

namespace compatibility_verify
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

// A genome with nodeCount nodes (IDs 0..nodeCount-1, alternating
// Input/Output -- NodeType is irrelevant to compatibility distance, this
// just keeps every fixture trivially constructible) and no connections.
// Callers add whatever connections they need.
ai::neat::Genome makeNodeGenome(int nodeCount)
{
    using ai::neat::Genome;
    using ai::neat::NodeGene;
    using ai::neat::NodeType;

    Genome genome;
    for (int i = 0; i < nodeCount; ++i)
    {
        genome.addNode(NodeGene{i, (i % 2 == 0) ? NodeType::Input : NodeType::Output});
    }
    return genome;
}

// A genome with connectionCount connections 0->1, 1->2, ..., innovation
// numbers 0..connectionCount-1 in order, weight 0, all enabled. Two chain
// genomes built with the same node-index scheme necessarily agree on
// source/target/weight for every innovation number they both contain
// (both start counting from node 0), which is what makes them useful for
// the normalization tests below -- their overlapping prefix is always a
// clean set of matching genes with zero weight difference.
ai::neat::Genome makeChainGenome(int connectionCount)
{
    using ai::neat::ConnectionGene;
    using ai::neat::Genome;
    using ai::neat::NodeGene;
    using ai::neat::NodeType;

    Genome genome;
    for (int i = 0; i <= connectionCount; ++i)
    {
        genome.addNode(NodeGene{i, (i % 2 == 0) ? NodeType::Input : NodeType::Output});
    }
    for (int i = 0; i < connectionCount; ++i)
    {
        genome.addConnection(ConnectionGene{i, i + 1, 0.0f, true, i});
    }
    return genome;
}

} // namespace compatibility_verify

// One-shot, deterministic sanity check of ai::neat::compatibilityDistance()
// / compatibilityBreakdown(), covering the NEAT compatibility-distance
// calculation end to end. Independent of Car/Track/AI/keyboard/render
// timing. Runs once at startup. No Species, speciation, population,
// selection, reproduction, generation, or training logic is exercised here
// -- only distance calculation, per Stage 11's scope.
void verifyCompatibilityDistance()
{
    using namespace compatibility_verify;
    using ai::neat::CompatibilityBreakdown;
    using ai::neat::CompatibilityConfig;
    using ai::neat::compatibilityBreakdown;
    using ai::neat::compatibilityDistance;
    using ai::neat::ConnectionGene;
    using ai::neat::Genome;
    using ai::neat::NodeGene;
    using ai::neat::NodeType;
    constexpr float kEps = 1e-5f;

    // 1, 2 & 3: negative coefficients are rejected.
    {
        Genome a;
        Genome b;

        CompatibilityConfig negExcess;
        negExcess.excessCoefficient = -1.0f;
        assert(throwsInvalidArgument([&]() { compatibilityDistance(a, b, negExcess); }) &&
               "a negative excessCoefficient must be rejected"); // 1

        CompatibilityConfig negDisjoint;
        negDisjoint.disjointCoefficient = -1.0f;
        assert(throwsInvalidArgument([&]() { compatibilityDistance(a, b, negDisjoint); }) &&
               "a negative disjointCoefficient must be rejected"); // 2

        CompatibilityConfig negWeight;
        negWeight.weightDifferenceCoefficient = -1.0f;
        assert(throwsInvalidArgument([&]() { compatibilityDistance(a, b, negWeight); }) &&
               "a negative weightDifferenceCoefficient must be rejected"); // 3
    }

    // 4: non-finite coefficients are rejected.
    {
        Genome a;
        Genome b;

        CompatibilityConfig nanConfig;
        nanConfig.excessCoefficient = std::numeric_limits<float>::quiet_NaN();
        assert(throwsInvalidArgument([&]() { compatibilityDistance(a, b, nanConfig); }) &&
               "a NaN coefficient must be rejected");

        CompatibilityConfig infConfig;
        infConfig.disjointCoefficient = std::numeric_limits<float>::infinity();
        assert(throwsInvalidArgument([&]() { compatibilityDistance(a, b, infConfig); }) &&
               "an infinite coefficient must be rejected");
    }

    // 5: a zero normalization threshold is rejected.
    {
        Genome a;
        Genome b;
        CompatibilityConfig zeroThreshold;
        zeroThreshold.smallGenomeNormalizationThreshold = 0;
        assert(throwsInvalidArgument([&]() { compatibilityDistance(a, b, zeroThreshold); }) &&
               "a zero smallGenomeNormalizationThreshold must be rejected");
    }

    // 6: two empty connection sets give distance 0.
    {
        Genome a;
        Genome b;
        CompatibilityConfig config;
        assert(compatibilityDistance(a, b, config) == 0.0f && "two genomes with no connections must have distance 0");
    }

    // 7: identical genomes give distance 0.
    {
        Genome a = makeNodeGenome(4);
        a.addConnection(ConnectionGene{0, 1, 0.5f, true, 0});
        a.addConnection(ConnectionGene{2, 3, 0.7f, true, 1});
        Genome b = a; // structurally and weight-identical copy

        CompatibilityConfig config;
        assert(std::fabs(compatibilityDistance(a, b, config)) < kEps && "identical genomes must have distance 0");
    }

    // 8: an enabled-state difference alone gives distance 0 -- matching by
    // innovation, endpoints and weight all agree; only enabled differs.
    {
        Genome a = makeNodeGenome(2);
        a.addConnection(ConnectionGene{0, 1, 0.5f, true, 0});
        Genome b = makeNodeGenome(2);
        b.addConnection(ConnectionGene{0, 1, 0.5f, false, 0});

        CompatibilityConfig config;
        assert(std::fabs(compatibilityDistance(a, b, config)) < kEps &&
               "an enabled-state difference alone must not affect distance");
    }

    // 9: one matching gene with equal weight gives W=0.
    {
        Genome a = makeNodeGenome(2);
        a.addConnection(ConnectionGene{0, 1, 0.5f, true, 0});
        Genome b = makeNodeGenome(2);
        b.addConnection(ConnectionGene{0, 1, 0.5f, true, 0});

        CompatibilityConfig config;
        const CompatibilityBreakdown result = compatibilityBreakdown(a, b, config);
        assert(result.matching == 1 && result.averageWeightDifference == 0.0f &&
               "a matching gene with equal weight must contribute zero to W");
    }

    // 10: one matching gene with different weight gives the correct W.
    {
        Genome a = makeNodeGenome(2);
        a.addConnection(ConnectionGene{0, 1, 0.3f, true, 0});
        Genome b = makeNodeGenome(2);
        b.addConnection(ConnectionGene{0, 1, 0.8f, true, 0});

        CompatibilityConfig config;
        const CompatibilityBreakdown result = compatibilityBreakdown(a, b, config);
        assert(result.matching == 1 && "exactly one matching gene must be found");
        assert(std::fabs(result.averageWeightDifference - 0.5f) < kEps && "W must equal the single weight difference");
    }

    // 11: multiple matching genes calculate the mean absolute weight
    // difference correctly.
    {
        Genome a = makeNodeGenome(6);
        a.addConnection(ConnectionGene{0, 1, 0.0f, true, 0});
        a.addConnection(ConnectionGene{2, 3, 0.5f, true, 1});
        a.addConnection(ConnectionGene{4, 5, 1.0f, true, 2});

        Genome b = makeNodeGenome(6);
        b.addConnection(ConnectionGene{0, 1, 0.3f, true, 0}); // diff 0.3
        b.addConnection(ConnectionGene{2, 3, 0.1f, true, 1}); // diff 0.4
        b.addConnection(ConnectionGene{4, 5, 1.2f, true, 2}); // diff 0.2

        CompatibilityConfig config;
        const CompatibilityBreakdown result = compatibilityBreakdown(a, b, config);
        assert(result.matching == 3 && "three matching genes must be found");
        const float expected = (0.3f + 0.4f + 0.2f) / 3.0f;
        assert(std::fabs(result.averageWeightDifference - expected) < kEps &&
               "W must equal the mean of the individual absolute weight differences");
    }

    // 12: matching genes align by innovation number, independent of vector
    // order -- and independent of which parent's weight happens to be
    // larger, unlike a (broken) index-based pairing would produce.
    {
        Genome a = makeNodeGenome(4);
        a.addConnection(ConnectionGene{2, 3, 0.9f, true, 5}); // innovation 5 stored first
        a.addConnection(ConnectionGene{0, 1, 0.1f, true, 2}); // innovation 2 stored second

        Genome b = makeNodeGenome(4);
        b.addConnection(ConnectionGene{0, 1, 0.15f, true, 2}); // innovation 2 stored first
        b.addConnection(ConnectionGene{2, 3, 0.5f, true, 5});  // innovation 5 stored second

        CompatibilityConfig config;
        const CompatibilityBreakdown result = compatibilityBreakdown(a, b, config);
        assert(result.matching == 2 && "both connections must be matched, one per innovation number");
        // Correct (innovation-aligned) pairing: |0.1-0.15|=0.05, |0.9-0.5|=0.4 -> mean 0.225.
        // A broken index-aligned pairing would instead pair A[0](innov5,0.9)
        // with B[0](innov2,0.15) and A[1](innov2,0.1) with B[1](innov5,0.5),
        // giving |0.9-0.15|=0.75 and |0.1-0.5|=0.4 -> mean 0.575 -- a
        // clearly different, wrong result this assertion would catch.
        assert(std::fabs(result.averageWeightDifference - 0.225f) < kEps &&
               "matching must align by innovation number, not vector position");
    }

    // 13: the same innovation number with conflicting endpoints between
    // genomes is rejected.
    {
        Genome a = makeNodeGenome(4);
        a.addConnection(ConnectionGene{0, 1, 0.5f, true, 7});
        Genome b = makeNodeGenome(4);
        b.addConnection(ConnectionGene{2, 3, 0.5f, true, 7}); // same innovation, different endpoints

        CompatibilityConfig config;
        assert(throwsInvalidArgument([&]() { compatibilityDistance(a, b, config); }) &&
               "a matching innovation with conflicting endpoints must be rejected");
    }

    // 14: a duplicate connection innovation number inside genome A is
    // rejected (Genome::validate() alone would not catch this, since the
    // two connections use different (source, target) pairs).
    {
        std::vector<NodeGene> nodes = {NodeGene{0, NodeType::Input}, NodeGene{1, NodeType::Input},
                                        NodeGene{2, NodeType::Output}, NodeGene{3, NodeType::Output}};
        std::vector<ConnectionGene> connections = {ConnectionGene{0, 2, 0.1f, true, 5},
                                                    ConnectionGene{1, 3, 0.2f, true, 5}};
        Genome aWithDuplicateInnovation(nodes, connections);
        Genome b;

        CompatibilityConfig config;
        assert(throwsInvalidArgument([&]() { compatibilityDistance(aWithDuplicateInnovation, b, config); }) &&
               "genome A with duplicate connection innovation numbers must be rejected");
    }

    // 15: symmetric case for genome B.
    {
        std::vector<NodeGene> nodes = {NodeGene{0, NodeType::Input}, NodeGene{1, NodeType::Input},
                                        NodeGene{2, NodeType::Output}, NodeGene{3, NodeType::Output}};
        std::vector<ConnectionGene> connections = {ConnectionGene{0, 2, 0.1f, true, 5},
                                                    ConnectionGene{1, 3, 0.2f, true, 5}};
        Genome bWithDuplicateInnovation(nodes, connections);
        Genome a;

        CompatibilityConfig config;
        assert(throwsInvalidArgument([&]() { compatibilityDistance(a, bWithDuplicateInnovation, config); }) &&
               "genome B with duplicate connection innovation numbers must be rejected");
    }

    // 16 & 17: an A-only innovation at/below B's max is disjoint; an A-only
    // innovation above B's max is excess. Genome B's own single non-matching
    // gene (innovation 15) is, as a side effect, also disjoint relative to
    // A's max (20) -- accounted for in the expected counts below.
    {
        Genome a = makeNodeGenome(12);
        a.addConnection(ConnectionGene{0, 1, 0.1f, true, 1});  // matching
        a.addConnection(ConnectionGene{2, 3, 0.2f, true, 3});  // A-only, expect disjoint (3 <= maxB=15)
        a.addConnection(ConnectionGene{4, 5, 0.3f, true, 20}); // A-only, expect excess (20 > maxB=15)

        Genome b = makeNodeGenome(12);
        b.addConnection(ConnectionGene{0, 1, 0.1f, true, 1});   // matching
        b.addConnection(ConnectionGene{6, 7, 0.4f, true, 15}); // B-only, expect disjoint (15 <= maxA=20)

        CompatibilityConfig config;
        const CompatibilityBreakdown result = compatibilityBreakdown(a, b, config);
        assert(result.matching == 1 && "exactly one matching gene");
        assert(result.disjoint == 2 && "A's innovation 3 and B's innovation 15 must both be disjoint"); // 16
        assert(result.excess == 1 && "A's innovation 20 must be excess"); // 17
    }

    // 18 & 19: symmetric case -- a B-only innovation at/below A's max is
    // disjoint; a B-only innovation above A's max is excess.
    {
        Genome a = makeNodeGenome(12);
        a.addConnection(ConnectionGene{0, 1, 0.1f, true, 1});   // matching
        a.addConnection(ConnectionGene{2, 3, 0.2f, true, 15}); // A-only, expect disjoint (15 <= maxB=20)

        Genome b = makeNodeGenome(12);
        b.addConnection(ConnectionGene{0, 1, 0.1f, true, 1});  // matching
        b.addConnection(ConnectionGene{4, 5, 0.3f, true, 3});  // B-only, expect disjoint (3 <= maxA=15)
        b.addConnection(ConnectionGene{6, 7, 0.4f, true, 20}); // B-only, expect excess (20 > maxA=15)

        CompatibilityConfig config;
        const CompatibilityBreakdown result = compatibilityBreakdown(a, b, config);
        assert(result.matching == 1 && "exactly one matching gene");
        assert(result.disjoint == 2 && "A's innovation 15 and B's innovation 3 must both be disjoint"); // 18
        assert(result.excess == 1 && "B's innovation 20 must be excess"); // 19
    }

    // 20: a mixed matching/disjoint/excess case is classified correctly in
    // a single comparison.
    {
        Genome a = makeNodeGenome(12);
        a.addConnection(ConnectionGene{0, 1, 0.1f, true, 1});  // matching
        a.addConnection(ConnectionGene{2, 3, 0.2f, true, 2});  // matching
        a.addConnection(ConnectionGene{4, 5, 0.3f, true, 4});  // A-only, disjoint (4 <= maxB=6)
        a.addConnection(ConnectionGene{6, 7, 0.4f, true, 50}); // A-only, excess (50 > maxB=6)

        Genome b = makeNodeGenome(12);
        b.addConnection(ConnectionGene{0, 1, 0.5f, true, 1}); // matching
        b.addConnection(ConnectionGene{2, 3, 0.6f, true, 2}); // matching
        b.addConnection(ConnectionGene{8, 9, 0.7f, true, 6}); // B-only, disjoint (6 <= maxA=50)

        CompatibilityConfig config;
        const CompatibilityBreakdown result = compatibilityBreakdown(a, b, config);
        assert(result.matching == 2 && result.disjoint == 2 && result.excess == 1 &&
               "a mixed comparison must classify every gene into the correct bucket");
    }

    // 21 & 22: a genome pair whose larger connection count is below the
    // (default) normalization threshold uses N=1.
    {
        // 21: larger size 5.
        {
            Genome a = makeChainGenome(5);
            Genome b = makeChainGenome(3);
            CompatibilityConfig config;
            const CompatibilityBreakdown result = compatibilityBreakdown(a, b, config);
            assert(result.normalization == 1.0f && "a small genome pair (larger size 5) must use N=1");
            // matching={0,1,2} (W=0); A-only={3,4}, maxB=2, both excess.
            assert(result.matching == 3 && result.excess == 2 && result.disjoint == 0);
            assert(std::fabs(result.distance - 2.0f) < kEps && "distance must use N=1, not the genome size, here");
        }
        // 22: larger size 19 -- exactly one below the default threshold (20).
        {
            Genome a = makeChainGenome(19);
            Genome b = makeChainGenome(15);
            CompatibilityConfig config;
            const CompatibilityBreakdown result = compatibilityBreakdown(a, b, config);
            assert(result.normalization == 1.0f && "a genome pair one below the threshold must still use N=1");
            // matching={0..14} (15 genes, W=0); A-only={15,16,17,18}, maxB=14, all excess.
            assert(result.matching == 15 && result.excess == 4 && result.disjoint == 0);
            assert(std::fabs(result.distance - 4.0f) < kEps && "distance must use N=1 here, not N=19");
        }
    }

    // 23: a genome pair whose larger connection count is exactly at the
    // threshold uses N = larger genome size (not 1).
    {
        Genome a = makeChainGenome(20);
        Genome b = makeChainGenome(15);
        CompatibilityConfig config; // default threshold = 20
        const CompatibilityBreakdown result = compatibilityBreakdown(a, b, config);
        assert(result.normalization == 20.0f && "a genome pair exactly at the threshold must use N = larger size");
        // matching={0..14} (15 genes, W=0); A-only={15..19} (5), maxB=14, all excess.
        assert(result.matching == 15 && result.excess == 5 && result.disjoint == 0);
        assert(std::fabs(result.distance - 0.25f) < kEps && "distance must divide by N=20 here"); // 5/20 = 0.25
    }

    // 24: a genome pair well above the threshold normalizes correctly.
    {
        Genome a = makeChainGenome(25);
        Genome b = makeChainGenome(10);
        CompatibilityConfig config;
        const CompatibilityBreakdown result = compatibilityBreakdown(a, b, config);
        assert(result.normalization == 25.0f && "normalization must equal the larger genome's connection count");
        // matching={0..9} (10 genes, W=0); A-only={10..24} (15), maxB=9, all excess.
        assert(result.matching == 10 && result.excess == 15 && result.disjoint == 0);
        assert(std::fabs(result.distance - 0.6f) < kEps && "distance must divide by N=25 here"); // 15/25 = 0.6
    }

    // 25, 26, 27 & 28: custom coefficients affect the result correctly, and
    // setting any one coefficient to zero removes exactly its own
    // contribution -- using one shared fixture with known E=1, D=2, W=0.25
    // and a small genome pair (N=1), so every term's contribution is
    // directly visible in the total.
    {
        Genome a = makeNodeGenome(12);
        a.addConnection(ConnectionGene{0, 1, 0.2f, true, 0}); // matching, weight diff 0.5
        a.addConnection(ConnectionGene{2, 3, 0.1f, true, 1}); // matching, weight diff 0.0
        a.addConnection(ConnectionGene{4, 5, 0.9f, true, 2}); // A-only, disjoint (2 <= maxB=4)
        a.addConnection(ConnectionGene{6, 7, 0.9f, true, 10}); // A-only, excess (10 > maxB=4)

        Genome b = makeNodeGenome(12);
        b.addConnection(ConnectionGene{0, 1, 0.7f, true, 0}); // matching, weight diff 0.5
        b.addConnection(ConnectionGene{2, 3, 0.1f, true, 1}); // matching, weight diff 0.0
        b.addConnection(ConnectionGene{8, 9, 0.9f, true, 4}); // B-only, disjoint (4 <= maxA=10)

        // matching=2 (W = (0.5+0.0)/2 = 0.25), disjoint=2 (A's 2 + B's 4),
        // excess=1 (A's 10), larger size = 4 (< default threshold 20) -> N=1.

        CompatibilityConfig defaultConfig; // c1=1.0, c2=1.0, c3=0.4
        const float defaultDistance = compatibilityDistance(a, b, defaultConfig);
        assert(std::fabs(defaultDistance - 3.1f) < kEps && "baseline distance must equal 1*1 + 1*2 + 0.4*0.25 = 3.1");

        CompatibilityConfig doubledExcess = defaultConfig;
        doubledExcess.excessCoefficient = 2.0f;
        assert(std::fabs(compatibilityDistance(a, b, doubledExcess) - 4.1f) < kEps &&
               "doubling excessCoefficient must add exactly one extra excess contribution"); // 25

        CompatibilityConfig zeroExcess = defaultConfig;
        zeroExcess.excessCoefficient = 0.0f;
        assert(std::fabs(compatibilityDistance(a, b, zeroExcess) - 2.1f) < kEps &&
               "a zero excessCoefficient must remove exactly the excess contribution"); // 26

        CompatibilityConfig zeroDisjoint = defaultConfig;
        zeroDisjoint.disjointCoefficient = 0.0f;
        assert(std::fabs(compatibilityDistance(a, b, zeroDisjoint) - 1.1f) < kEps &&
               "a zero disjointCoefficient must remove exactly the disjoint contribution"); // 27

        CompatibilityConfig zeroWeight = defaultConfig;
        zeroWeight.weightDifferenceCoefficient = 0.0f;
        assert(std::fabs(compatibilityDistance(a, b, zeroWeight) - 3.0f) < kEps &&
               "a zero weightDifferenceCoefficient must remove exactly the weight contribution"); // 28
    }

    // 29: no matching genes at all gives W=0.
    {
        Genome a = makeNodeGenome(4);
        a.addConnection(ConnectionGene{0, 1, 0.5f, true, 0});
        a.addConnection(ConnectionGene{2, 3, 0.9f, true, 1});
        Genome b = makeNodeGenome(4);
        b.addConnection(ConnectionGene{0, 1, 0.5f, true, 5});
        b.addConnection(ConnectionGene{2, 3, 0.9f, true, 6});

        CompatibilityConfig config;
        const CompatibilityBreakdown result = compatibilityBreakdown(a, b, config);
        assert(result.matching == 0 && result.averageWeightDifference == 0.0f &&
               "with no matching genes at all, W must be exactly 0");
    }

    // 30: one empty genome vs. a non-empty genome classifies every gene of
    // the non-empty genome as excess (the empty genome's "max innovation"
    // is below every non-negative innovation number).
    {
        Genome a;
        Genome b = makeNodeGenome(4);
        b.addConnection(ConnectionGene{0, 1, 0.5f, true, 0});
        b.addConnection(ConnectionGene{2, 3, 0.9f, true, 1});

        CompatibilityConfig config;
        const CompatibilityBreakdown result = compatibilityBreakdown(a, b, config);
        assert(result.matching == 0 && result.disjoint == 0 && result.excess == 2 &&
               "every gene of a non-empty genome compared against an empty one must be excess");
    }

    // 31 & 32: distance and breakdown counts are symmetric.
    {
        Genome a = makeNodeGenome(12);
        a.addConnection(ConnectionGene{0, 1, 0.2f, true, 0});
        a.addConnection(ConnectionGene{2, 3, 0.9f, true, 2});
        a.addConnection(ConnectionGene{4, 5, 0.9f, true, 10});

        Genome b = makeNodeGenome(12);
        b.addConnection(ConnectionGene{0, 1, 0.7f, true, 0});
        b.addConnection(ConnectionGene{8, 9, 0.9f, true, 4});

        CompatibilityConfig config;
        const float distanceAB = compatibilityDistance(a, b, config);
        const float distanceBA = compatibilityDistance(b, a, config);
        assert(std::fabs(distanceAB - distanceBA) < kEps && "distance(A,B) must equal distance(B,A)"); // 31

        const CompatibilityBreakdown breakdownAB = compatibilityBreakdown(a, b, config);
        const CompatibilityBreakdown breakdownBA = compatibilityBreakdown(b, a, config);
        assert(breakdownAB.matching == breakdownBA.matching && breakdownAB.disjoint == breakdownBA.disjoint &&
               breakdownAB.excess == breakdownBA.excess &&
               std::fabs(breakdownAB.averageWeightDifference - breakdownBA.averageWeightDifference) < kEps &&
               breakdownAB.normalization == breakdownBA.normalization &&
               "breakdown counts, W and normalization must all be symmetric"); // 32
    }

    // 33: the order genes/nodes were inserted into either parent does not
    // affect the result.
    {
        Genome inOrderA = makeNodeGenome(4);
        inOrderA.addConnection(ConnectionGene{0, 1, 0.3f, true, 0});
        inOrderA.addConnection(ConnectionGene{2, 3, 0.6f, true, 1});
        Genome inOrderB = makeNodeGenome(4);
        inOrderB.addConnection(ConnectionGene{0, 1, 0.8f, true, 0});
        inOrderB.addConnection(ConnectionGene{2, 3, 0.6f, true, 1});

        CompatibilityConfig config;
        const float canonical = compatibilityDistance(inOrderA, inOrderB, config);

        // Same final gene set, built in a different order (nodes added
        // high-to-low, connections added in reverse innovation order).
        using ai::neat::Genome;
        using ai::neat::NodeGene;
        using ai::neat::NodeType;
        Genome shuffledA;
        shuffledA.addNode(NodeGene{3, NodeType::Output});
        shuffledA.addNode(NodeGene{1, NodeType::Output});
        shuffledA.addNode(NodeGene{2, NodeType::Input});
        shuffledA.addNode(NodeGene{0, NodeType::Input});
        shuffledA.addConnection(ConnectionGene{2, 3, 0.6f, true, 1});
        shuffledA.addConnection(ConnectionGene{0, 1, 0.3f, true, 0});

        Genome shuffledB;
        shuffledB.addNode(NodeGene{2, NodeType::Input});
        shuffledB.addNode(NodeGene{0, NodeType::Input});
        shuffledB.addNode(NodeGene{3, NodeType::Output});
        shuffledB.addNode(NodeGene{1, NodeType::Output});
        shuffledB.addConnection(ConnectionGene{2, 3, 0.6f, true, 1});
        shuffledB.addConnection(ConnectionGene{0, 1, 0.8f, true, 0});

        const float shuffled = compatibilityDistance(shuffledA, shuffledB, config);
        assert(std::fabs(canonical - shuffled) < kEps && "insertion order must not affect the computed distance");
    }

    // 34: node-only differences do not affect the connection compatibility
    // distance -- extra, unreferenced Hidden nodes differ between the two
    // comparisons below, but the connections are identical, so the
    // distance must be identical too.
    {
        Genome baselineA = makeNodeGenome(2);
        baselineA.addConnection(ConnectionGene{0, 1, 0.4f, true, 0});
        Genome baselineB = makeNodeGenome(2);
        baselineB.addConnection(ConnectionGene{0, 1, 0.9f, true, 0});
        CompatibilityConfig config;
        const float baselineDistance = compatibilityDistance(baselineA, baselineB, config);

        Genome withExtraNodesA = makeNodeGenome(2);
        withExtraNodesA.addNode(NodeGene{50, NodeType::Hidden});
        withExtraNodesA.addConnection(ConnectionGene{0, 1, 0.4f, true, 0});
        Genome withExtraNodesB = makeNodeGenome(2);
        withExtraNodesB.addNode(NodeGene{60, NodeType::Hidden}); // different extra node ID entirely
        withExtraNodesB.addConnection(ConnectionGene{0, 1, 0.9f, true, 0});

        const float extraNodesDistance = compatibilityDistance(withExtraNodesA, withExtraNodesB, config);
        assert(std::fabs(baselineDistance - extraNodesDistance) < kEps &&
               "node-only differences must not affect the connection-based compatibility distance");
    }

    // 35 & 36: neither genome's node or connection genes are ever modified.
    {
        Genome a = makeNodeGenome(4);
        a.addNode(NodeGene{50, NodeType::Hidden});
        a.addConnection(ConnectionGene{0, 1, 0.3f, true, 0});
        a.addConnection(ConnectionGene{2, 50, 0.6f, false, 5});

        Genome b = makeNodeGenome(4);
        b.addConnection(ConnectionGene{0, 1, 0.8f, true, 0});
        b.addConnection(ConnectionGene{2, 3, 0.6f, true, 9});

        const std::vector<NodeGene> nodesABefore = a.nodes();
        const std::vector<ConnectionGene> connectionsABefore = a.connections();
        const std::vector<NodeGene> nodesBBefore = b.nodes();
        const std::vector<ConnectionGene> connectionsBBefore = b.connections();

        CompatibilityConfig config;
        compatibilityDistance(a, b, config);

        assert(a.nodes().size() == nodesABefore.size() && "compatibilityDistance must not add/remove genome A's nodes");
        for (std::size_t i = 0; i < nodesABefore.size(); ++i)
        {
            assert(a.nodes()[i] == nodesABefore[i] && "compatibilityDistance must not modify genome A's nodes"); // 35
        }
        assert(a.connections().size() == connectionsABefore.size() &&
               "compatibilityDistance must not add/remove genome A's connections");
        for (std::size_t i = 0; i < connectionsABefore.size(); ++i)
        {
            const ConnectionGene& before = connectionsABefore[i];
            const ConnectionGene& after = a.connections()[i];
            assert(after.getSourceId() == before.getSourceId() && after.getTargetId() == before.getTargetId() &&
                   after.getWeight() == before.getWeight() && after.isEnabled() == before.isEnabled() &&
                   after.getInnovationNumber() == before.getInnovationNumber() &&
                   "compatibilityDistance must not modify genome A's connections"); // 36
        }

        assert(b.nodes().size() == nodesBBefore.size() && "compatibilityDistance must not add/remove genome B's nodes");
        for (std::size_t i = 0; i < nodesBBefore.size(); ++i)
        {
            assert(b.nodes()[i] == nodesBBefore[i] && "compatibilityDistance must not modify genome B's nodes");
        }
        assert(b.connections().size() == connectionsBBefore.size() &&
               "compatibilityDistance must not add/remove genome B's connections");
        for (std::size_t i = 0; i < connectionsBBefore.size(); ++i)
        {
            const ConnectionGene& before = connectionsBBefore[i];
            const ConnectionGene& after = b.connections()[i];
            assert(after.getSourceId() == before.getSourceId() && after.getTargetId() == before.getTargetId() &&
                   after.getWeight() == before.getWeight() && after.isEnabled() == before.isEnabled() &&
                   after.getInnovationNumber() == before.getInnovationNumber() &&
                   "compatibilityDistance must not modify genome B's connections");
        }
    }

    // 37: no InnovationTracker state is involved -- CompatibilityDistance.h/
    // .cpp never include or reference InnovationTracker anywhere, and
    // compatibilityDistance()'s signature takes no tracker parameter.

    // 38: no RNG state is involved -- CompatibilityDistance.cpp contains no
    // std::mt19937 or any other random generator; the calculation is a
    // pure function of its three arguments.

    // 39: repeated calls with identical inputs produce an identical result.
    {
        Genome a = makeNodeGenome(4);
        a.addConnection(ConnectionGene{0, 1, 0.3f, true, 0});
        a.addConnection(ConnectionGene{2, 3, 0.6f, true, 5});
        Genome b = makeNodeGenome(4);
        b.addConnection(ConnectionGene{0, 1, 0.8f, true, 0});

        CompatibilityConfig config;
        const float first = compatibilityDistance(a, b, config);
        const float second = compatibilityDistance(a, b, config);
        const float third = compatibilityDistance(a, b, config);
        assert(first == second && second == third && "repeated calls with identical inputs must produce identical results");
    }

    // 40: all previous verification suites still pass -- enforced by
    // main() continuing to call every earlier verify*() function
    // unchanged.

    TraceLog(LOG_INFO, "Compatibility distance verification: all deterministic checks passed");
}

namespace speciation_verify
{

// A 2-node genome with a single connection 0->1, innovation 0, and the
// given weight. Two genomes built this way always share innovation 0 as a
// matching gene with no disjoint/excess genes at all, so with the default
// CompatibilityConfig (c3 = 0.4, small-genome N = 1) their compatibility
// distance is exactly 0.4 * |weightA - weightB| -- simple, predictable
// arithmetic for driving Speciator's threshold behavior in these tests.
ai::neat::Genome makeSimpleGenome(float weight)
{
    using namespace compatibility_verify;
    using ai::neat::ConnectionGene;
    using ai::neat::Genome;

    Genome genome = makeNodeGenome(2);
    genome.addConnection(ConnectionGene{0, 1, weight, true, 0});
    return genome;
}

} // namespace speciation_verify

// One-shot, deterministic sanity check of ai::neat::Speciator /
// ai::neat::Species, covering deterministic species-membership assignment
// end to end. Independent of Car/Track/AI/keyboard/render timing. Runs
// once at startup. No Population, Individual/Agent wrapper, fitness
// storage, adjusted fitness, fitness sharing, champion tracking, parent
// selection, reproduction, elitism, mutation orchestration, crossover
// orchestration, persistent cross-generation species, representative
// reselection, stagnation, extinction, or generation-loop logic is
// exercised here -- only species assignment, per Stage 12's scope.
void verifySpeciation()
{
    using namespace compatibility_verify;
    using namespace speciation_verify;
    using ai::neat::CompatibilityConfig;
    using ai::neat::ConnectionGene;
    using ai::neat::Genome;
    using ai::neat::NodeGene;
    using ai::neat::NodeType;
    using ai::neat::SpeciationConfig;
    using ai::neat::Speciator;
    using ai::neat::Species;

    // 1 & 2: an invalid (negative or non-finite) compatibilityThreshold is
    // rejected, and never advances the SpeciesId counter.
    {
        std::vector<Genome> genomes = {makeSimpleGenome(0.0f)};
        Speciator speciator;
        CompatibilityConfig compatConfig;

        SpeciationConfig negativeThreshold;
        negativeThreshold.compatibilityThreshold = -1.0f;
        assert(throwsInvalidArgument([&]() { speciator.speciate(genomes, compatConfig, negativeThreshold); }) &&
               "a negative compatibilityThreshold must be rejected"); // 1

        SpeciationConfig nanThreshold;
        nanThreshold.compatibilityThreshold = std::numeric_limits<float>::quiet_NaN();
        assert(throwsInvalidArgument([&]() { speciator.speciate(genomes, compatConfig, nanThreshold); }) &&
               "a NaN compatibilityThreshold must be rejected"); // 2

        SpeciationConfig infThreshold;
        infThreshold.compatibilityThreshold = std::numeric_limits<float>::infinity();
        assert(throwsInvalidArgument([&]() { speciator.speciate(genomes, compatConfig, infThreshold); }) &&
               "a non-finite compatibilityThreshold must be rejected"); // 2 (continued)

        SpeciationConfig validConfig;
        const std::vector<Species> species = speciator.speciate(genomes, compatConfig, validConfig);
        assert(species.size() == 1 && species[0].getId() == 0 &&
               "earlier rejected calls must never have advanced the SpeciesId counter");
    }

    // 3 & 4: empty input returns no species and consumes no SpeciesId.
    {
        std::vector<Genome> empty;
        Speciator speciator;
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig;

        const std::vector<Species> species = speciator.speciate(empty, compatConfig, speciationConfig);
        assert(species.empty() && "an empty genome vector must produce no species"); // 3

        std::vector<Genome> nonEmpty = {makeSimpleGenome(0.0f)};
        const std::vector<Species> species2 = speciator.speciate(nonEmpty, compatConfig, speciationConfig);
        assert(species2.size() == 1 && species2[0].getId() == 0 &&
               "an empty call must not consume a SpeciesId"); // 4
    }

    // 5, 6 & 7: one genome creates exactly one species, with the first
    // available SpeciesId, whose representative equals the founding genome.
    {
        Genome g = makeSimpleGenome(0.5f);
        std::vector<Genome> genomes = {g};
        Speciator speciator;
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig;

        const std::vector<Species> species = speciator.speciate(genomes, compatConfig, speciationConfig);
        assert(species.size() == 1 && "one genome must create exactly one species"); // 5
        assert(species[0].getId() == 0 && "the first species must get SpeciesId 0"); // 6

        const ConnectionGene* repConnection = species[0].getRepresentative().findConnection(0, 1);
        assert(repConnection != nullptr && repConnection->getWeight() == 0.5f &&
               "the representative must equal the genome that founded the species"); // 7
    }

    // 8: the representative is a copy, not a reference -- mutating the
    // original genome afterward must not affect it.
    {
        std::vector<Genome> genomes = {makeSimpleGenome(0.5f)};
        Speciator speciator;
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig;

        const std::vector<Species> species = speciator.speciate(genomes, compatConfig, speciationConfig);
        genomes[0].mutableConnections()[0].setWeight(999.0f);
        assert(species[0].getRepresentative().findConnection(0, 1)->getWeight() == 0.5f &&
               "the representative must be an independent copy, unaffected by later changes to the source genome");
    }

    // 9: identical genomes join the same species.
    {
        std::vector<Genome> genomes = {makeSimpleGenome(0.5f), makeSimpleGenome(0.5f)};
        Speciator speciator;
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig;

        const std::vector<Species> species = speciator.speciate(genomes, compatConfig, speciationConfig);
        assert(species.size() == 1 && species[0].size() == 2 && "identical genomes must join the same species");
    }

    // 10: a distance exactly equal to the threshold still qualifies.
    {
        std::vector<Genome> genomes = {makeSimpleGenome(0.0f), makeSimpleGenome(2.5f)}; // distance = 0.4*2.5 = 1.0
        Speciator speciator;
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig;
        speciationConfig.compatibilityThreshold = 1.0f;

        const std::vector<Species> species = speciator.speciate(genomes, compatConfig, speciationConfig);
        assert(species.size() == 1 && species[0].size() == 2 &&
               "a distance exactly equal to the threshold must still qualify for membership");
    }

    // 11: a distance above the threshold creates a new species.
    {
        std::vector<Genome> genomes = {makeSimpleGenome(0.0f), makeSimpleGenome(3.0f)}; // distance = 0.4*3.0 = 1.2
        Speciator speciator;
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig;
        speciationConfig.compatibilityThreshold = 1.0f;

        const std::vector<Species> species = speciator.speciate(genomes, compatConfig, speciationConfig);
        assert(species.size() == 2 && species[0].getId() == 0 && species[1].getId() == 1 &&
               "a distance above the threshold must create a new species");
    }

    // 12: sufficiently different genomes can create more than two species.
    {
        std::vector<Genome> genomes = {makeSimpleGenome(0.0f), makeSimpleGenome(10.0f), makeSimpleGenome(20.0f)};
        Speciator speciator;
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig; // default threshold 3.0

        const std::vector<Species> species = speciator.speciate(genomes, compatConfig, speciationConfig);
        assert(species.size() == 3 && "sufficiently different genomes must be able to create multiple species");
    }

    // 13 & 14: member indices are exactly the assigned genomes' original
    // indices, stored in ascending (input) order.
    {
        std::vector<Genome> genomes = {makeSimpleGenome(0.0f), makeSimpleGenome(0.1f), makeSimpleGenome(10.0f),
                                        makeSimpleGenome(0.05f)};
        Speciator speciator;
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig; // default threshold 3.0

        const std::vector<Species> species = speciator.speciate(genomes, compatConfig, speciationConfig);
        assert(species.size() == 2 && "this fixture must produce exactly two species");
        const std::vector<std::size_t> expectedSpeciesZeroMembers = {0, 1, 3};
        const std::vector<std::size_t> expectedSpeciesOneMembers = {2};
        assert(species[0].getMemberIndices() == expectedSpeciesZeroMembers &&
               "member indices must exactly match the genomes assigned to this species"); // 13
        assert(species[1].getMemberIndices() == expectedSpeciesOneMembers &&
               "member indices must exactly match the genomes assigned to this species"); // 13 (continued)
        // 14: {0,1,3} is already strictly ascending -- input-order sorted.
        for (std::size_t i = 1; i < species[0].getMemberIndices().size(); ++i)
        {
            assert(species[0].getMemberIndices()[i - 1] < species[0].getMemberIndices()[i] &&
                   "member indices must remain in ascending input-genome-index order");
        }
    }

    // 15: the returned species vector is sorted ascending by SpeciesId.
    {
        std::vector<Genome> genomes = {makeSimpleGenome(0.0f), makeSimpleGenome(10.0f), makeSimpleGenome(20.0f)};
        Speciator speciator;
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig;

        const std::vector<Species> species = speciator.speciate(genomes, compatConfig, speciationConfig);
        for (std::size_t i = 1; i < species.size(); ++i)
        {
            assert(species[i - 1].getId() < species[i].getId() &&
                   "the returned species vector must be sorted ascending by SpeciesId");
        }
    }

    // 16 & 17: first-match rule -- a genome compatible with more than one
    // existing species joins the FIRST (lowest-SpeciesId) one, even when a
    // later species is a strictly closer match; nearest-species search is
    // never used.
    {
        // rep0 = 0.0, rep1 = 10.0 (dist to rep0 = 4.0 > 3.0 -> founds species1).
        // g2 = 7.0: dist to rep0 = 2.8 <= 3.0 (qualifies for species0);
        //           dist to rep1 = 1.2 <= 3.0 (qualifies for species1, and
        //           is the objectively closer match). First-match must
        //           still place it in species0.
        std::vector<Genome> genomes = {makeSimpleGenome(0.0f), makeSimpleGenome(10.0f), makeSimpleGenome(7.0f)};
        Speciator speciator;
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig; // default threshold 3.0

        const std::vector<Species> species = speciator.speciate(genomes, compatConfig, speciationConfig);
        assert(species.size() == 2 && "this fixture must produce exactly two species");
        const std::vector<std::size_t> expectedSpeciesZeroMembers = {0, 2};
        assert(species[0].getMemberIndices() == expectedSpeciesZeroMembers &&
               "genome 2 must join species 0 (first match), even though species 1's representative is a strictly "
               "closer match -- nearest-species search is not implemented"); // 16 & 17
        assert(species[1].getMemberIndices().size() == 1);
    }

    // 18: input genome order can change the resulting grouping under
    // first-match semantics -- the same three genomes, reordered, produce
    // a different membership split because a different genome ends up
    // founding (and fixing the representative of) the first species.
    {
        // A=0.0, B=3.0 (dist(A,B)=1.2), C=5.5 (dist(A,C)=2.2, dist(B,C)=1.0), threshold=1.5.
        const float weightA = 0.0f;
        const float weightB = 3.0f;
        const float weightC = 5.5f;
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig;
        speciationConfig.compatibilityThreshold = 1.5f;

        // Order [A, B, C]: species0 founded by A. B joins species0
        // (dist(A,B)=1.2). C compared only against species0's rep, still A
        // (dist(A,C)=2.2 > 1.5) -> founds species1.
        {
            std::vector<Genome> genomes = {makeSimpleGenome(weightA), makeSimpleGenome(weightB), makeSimpleGenome(weightC)};
            Speciator speciator;
            const std::vector<Species> species = speciator.speciate(genomes, compatConfig, speciationConfig);
            assert(species.size() == 2 && species[0].size() == 2 && species[1].size() == 1 &&
                   "order [A,B,C] must split C into its own species");
        }

        // Order [B, A, C]: species0 founded by B. A joins species0
        // (dist(B,A)=1.2, symmetric). C compared against species0's rep,
        // now B (dist(B,C)=1.0 <= 1.5) -> joins species0 too.
        {
            std::vector<Genome> genomes = {makeSimpleGenome(weightB), makeSimpleGenome(weightA), makeSimpleGenome(weightC)};
            Speciator speciator;
            const std::vector<Species> species = speciator.speciate(genomes, compatConfig, speciationConfig);
            assert(species.size() == 1 && species[0].size() == 3 &&
                   "order [B,A,C] must group all three genomes into a single species -- a different result from "
                   "[A,B,C] for the exact same underlying genomes, purely because a different genome founded the "
                   "first species and fixed its representative");
        }
    }

    // 19: a fresh Speciator run with identical input produces an identical
    // result to another fresh Speciator run with the same input.
    {
        std::vector<Genome> genomes = {makeSimpleGenome(0.0f), makeSimpleGenome(1.0f), makeSimpleGenome(10.0f)};
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig;

        Speciator speciatorX;
        const std::vector<Species> speciesX = speciatorX.speciate(genomes, compatConfig, speciationConfig);
        Speciator speciatorY;
        const std::vector<Species> speciesY = speciatorY.speciate(genomes, compatConfig, speciationConfig);

        assert(speciesX.size() == speciesY.size() && "repeated identical runs must produce the same species count");
        for (std::size_t i = 0; i < speciesX.size(); ++i)
        {
            assert(speciesX[i].getId() == speciesY[i].getId() && speciesX[i].getMemberIndices() == speciesY[i].getMemberIndices() &&
                   "repeated identical runs must produce identical species IDs and membership");
        }
    }

    // 20: no RNG state exists anywhere in Speciator or Species -- neither
    // class declares a std::mt19937 (or any other generator) member, and
    // Speciator's constructor takes no seed. Determinism above (19) is a
    // direct consequence, not a coincidence.

    // 21: input genomes are never modified by speciate().
    {
        Genome g1 = makeSimpleGenome(0.5f);
        Genome g2 = makeSimpleGenome(3.5f);
        std::vector<Genome> genomes = {g1, g2};
        const std::vector<NodeGene> nodes0Before = genomes[0].nodes();
        const std::vector<ConnectionGene> connections0Before = genomes[0].connections();
        const std::vector<NodeGene> nodes1Before = genomes[1].nodes();
        const std::vector<ConnectionGene> connections1Before = genomes[1].connections();

        Speciator speciator;
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig;
        speciator.speciate(genomes, compatConfig, speciationConfig);

        assert(genomes[0].nodes() == nodes0Before && genomes[1].nodes() == nodes1Before &&
               "speciate() must not modify any input genome's nodes");
        auto connectionsMatch = [](const std::vector<ConnectionGene>& after, const std::vector<ConnectionGene>& before)
        {
            if (after.size() != before.size())
            {
                return false;
            }
            for (std::size_t i = 0; i < before.size(); ++i)
            {
                if (after[i].getSourceId() != before[i].getSourceId() || after[i].getTargetId() != before[i].getTargetId() ||
                    after[i].getWeight() != before[i].getWeight() || after[i].isEnabled() != before[i].isEnabled() ||
                    after[i].getInnovationNumber() != before[i].getInnovationNumber())
                {
                    return false;
                }
            }
            return true;
        };
        assert(connectionsMatch(genomes[0].connections(), connections0Before) &&
               connectionsMatch(genomes[1].connections(), connections1Before) &&
               "speciate() must not modify any input genome's connections");
    }

    // 22 & 23: the representative stays fixed at the founding genome's copy
    // -- a later, compatible member joining the species never overwrites
    // or blends into it.
    {
        std::vector<Genome> genomes = {makeSimpleGenome(0.0f), makeSimpleGenome(0.1f)};
        Speciator speciator;
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig;

        const std::vector<Species> species = speciator.speciate(genomes, compatConfig, speciationConfig);
        assert(species.size() == 1 && species[0].size() == 2 && "both genomes must join the same species");
        assert(species[0].getRepresentative().findConnection(0, 1)->getWeight() == 0.0f &&
               "the representative must remain the founding genome's own weight, unaffected by the later member "
               "joining"); // 22 & 23
    }

    // 24 & 25: CompatibilityConfig is honored -- custom coefficients
    // change whether two genomes are considered compatible.
    {
        std::vector<Genome> genomes = {makeSimpleGenome(0.0f), makeSimpleGenome(2.0f)}; // weight diff = 2.0
        SpeciationConfig speciationConfig;
        speciationConfig.compatibilityThreshold = 1.0f;

        CompatibilityConfig lowWeightCoefficient;
        lowWeightCoefficient.weightDifferenceCoefficient = 0.1f; // distance = 0.1*2.0 = 0.2 <= 1.0
        Speciator speciatorLow;
        const std::vector<Species> speciesLow = speciatorLow.speciate(genomes, lowWeightCoefficient, speciationConfig);
        assert(speciesLow.size() == 1 &&
               "with a low weightDifferenceCoefficient, the genomes must be compatible"); // 24

        CompatibilityConfig highWeightCoefficient;
        highWeightCoefficient.weightDifferenceCoefficient = 1.0f; // distance = 1.0*2.0 = 2.0 > 1.0
        Speciator speciatorHigh;
        const std::vector<Species> speciesHigh = speciatorHigh.speciate(genomes, highWeightCoefficient, speciationConfig);
        assert(speciesHigh.size() == 2 &&
               "with a high weightDifferenceCoefficient, the same genomes must become incompatible"); // 25
    }

    // 26: a zero threshold groups only exactly-zero-distance genomes
    // together.
    {
        std::vector<Genome> genomes = {makeSimpleGenome(0.5f), makeSimpleGenome(0.5f), makeSimpleGenome(0.6f)};
        Speciator speciator;
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig;
        speciationConfig.compatibilityThreshold = 0.0f;

        const std::vector<Species> species = speciator.speciate(genomes, compatConfig, speciationConfig);
        assert(species.size() == 2 && species[0].size() == 2 && species[1].size() == 1 &&
               "a zero threshold must only group exactly-zero-distance genomes together");
    }

    // 27: a very large threshold groups all compatible genomes into one
    // species.
    {
        std::vector<Genome> genomes = {makeSimpleGenome(0.0f), makeSimpleGenome(100.0f), makeSimpleGenome(-50.0f)};
        Speciator speciator;
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig;
        speciationConfig.compatibilityThreshold = 1000.0f;

        const std::vector<Species> species = speciator.speciate(genomes, compatConfig, speciationConfig);
        assert(species.size() == 1 && species[0].size() == 3 &&
               "a very large threshold must group all genomes into a single species");
    }

    // 28 & 29: SpeciesIds keep increasing monotonically across separate
    // speciate() calls on the same Speciator, never reusing earlier IDs.
    {
        Speciator speciator;
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig;

        std::vector<Genome> firstBatch = {makeSimpleGenome(0.0f), makeSimpleGenome(10.0f), makeSimpleGenome(20.0f)};
        const std::vector<Species> firstSpecies = speciator.speciate(firstBatch, compatConfig, speciationConfig);
        assert(firstSpecies.size() == 3 && firstSpecies[0].getId() == 0 && firstSpecies[1].getId() == 1 &&
               firstSpecies[2].getId() == 2 && "the first call must allocate SpeciesIds 0, 1, 2");

        std::vector<Genome> secondBatch = {makeSimpleGenome(0.0f), makeSimpleGenome(30.0f)};
        const std::vector<Species> secondSpecies = speciator.speciate(secondBatch, compatConfig, speciationConfig);
        assert(secondSpecies.size() == 2 && secondSpecies[0].getId() == 3 && secondSpecies[1].getId() == 4 &&
               "a second call on the same Speciator must continue the SpeciesId counter monotonically, never "
               "reusing IDs 0-2"); // 28 & 29
    }

    // 30: an empty call sandwiched between non-empty calls does not
    // advance the SpeciesId counter.
    {
        Speciator speciator;
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig;

        std::vector<Genome> firstBatch = {makeSimpleGenome(0.0f)};
        const std::vector<Species> firstSpecies = speciator.speciate(firstBatch, compatConfig, speciationConfig);
        assert(firstSpecies.size() == 1 && firstSpecies[0].getId() == 0);

        std::vector<Genome> emptyBatch;
        const std::vector<Species> emptySpecies = speciator.speciate(emptyBatch, compatConfig, speciationConfig);
        assert(emptySpecies.empty());

        std::vector<Genome> secondBatch = {makeSimpleGenome(50.0f)};
        const std::vector<Species> secondSpecies = speciator.speciate(secondBatch, compatConfig, speciationConfig);
        assert(secondSpecies.size() == 1 && secondSpecies[0].getId() == 1 &&
               "an empty call between non-empty calls must not consume a SpeciesId");
    }

    // 31 & 32: size()/empty() report member counts correctly.
    {
        std::vector<Genome> genomes = {makeSimpleGenome(0.0f), makeSimpleGenome(0.1f), makeSimpleGenome(10.0f)};
        Speciator speciator;
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig;
        const std::vector<Species> species = speciator.speciate(genomes, compatConfig, speciationConfig);

        assert(species.size() == 2 && "this fixture must produce exactly two species");
        assert(species[0].size() == 2 && !species[0].empty() && "size()/empty() must reflect two members"); // 31 & 32
        assert(species[1].size() == 1 && !species[1].empty() && "size()/empty() must reflect one member");

        const Species freshlyConstructed(0, makeSimpleGenome(0.0f));
        assert(freshlyConstructed.size() == 0 && freshlyConstructed.empty() &&
               "a species with no members added yet must report empty()"); // 32 (continued)
    }

    // 33 & 35: a structural conflict compatibilityDistance() would reject
    // (conflicting matching endpoints) propagates unchanged out of
    // speciate() rather than being silently skipped.
    {
        Genome g1 = makeNodeGenome(4);
        g1.addConnection(ConnectionGene{0, 1, 0.5f, true, 7});
        Genome g2 = makeNodeGenome(4);
        g2.addConnection(ConnectionGene{2, 3, 0.5f, true, 7}); // same innovation, different endpoints
        std::vector<Genome> genomes = {g1, g2};

        Speciator speciator;
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig;
        assert(throwsInvalidArgument([&]() { speciator.speciate(genomes, compatConfig, speciationConfig); }) &&
               "a conflicting-endpoints structural error from compatibilityDistance() must propagate out of "
               "speciate(), not be silently skipped"); // 33 & 35
    }

    // 34: duplicate connection innovation numbers inside a genome
    // propagate as a rejection -- every genome is now validated
    // unconditionally (see the correction block below), so this is caught
    // whether or not the genome is ever compared against an existing
    // representative.
    {
        std::vector<NodeGene> nodes = {NodeGene{0, NodeType::Input}, NodeGene{1, NodeType::Input},
                                        NodeGene{2, NodeType::Output}, NodeGene{3, NodeType::Output}};
        std::vector<ConnectionGene> connections = {ConnectionGene{0, 2, 0.1f, true, 5},
                                                    ConnectionGene{1, 3, 0.2f, true, 5}};
        Genome genomeWithDuplicateInnovation(nodes, connections);
        std::vector<Genome> genomes = {makeSimpleGenome(0.0f), genomeWithDuplicateInnovation};

        Speciator speciator;
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig;
        assert(throwsInvalidArgument([&]() { speciator.speciate(genomes, compatConfig, speciationConfig); }) &&
               "a genome with duplicate connection innovation numbers must cause speciate() to throw once compared");
    }

    // Stage 12 correction: every genome is now validated -- via
    // genome.validate() plus a self compatibilityDistance() check that
    // reuses the existing innovation-uniqueness logic -- before it can
    // either join an existing species or found a new one. Previously, a
    // genome that never got compared against an existing representative
    // (specifically: the very first genome processed, when no species yet
    // exist to compare it against) could slip through completely
    // unvalidated. The following checks prove the fix.

    // Correction 1: a single (lone) structurally invalid genome is
    // rejected, even though -- before the fix -- it would simply have
    // founded species 0 with no comparison, and therefore no validation,
    // ever taking place.
    {
        std::vector<NodeGene> nodes = {NodeGene{0, NodeType::Input}};
        std::vector<ConnectionGene> connections = {ConnectionGene{0, 99, 0.1f, true, 0}}; // target node 99 does not exist
        Genome invalidGenome(nodes, connections);
        std::vector<Genome> genomes = {invalidGenome};

        Speciator speciator;
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig;
        assert(throwsInvalidArgument([&]() { speciator.speciate(genomes, compatConfig, speciationConfig); }) &&
               "a single structurally invalid genome (Genome::validate() failure) must be rejected even as the "
               "lone founding genome"); // correction 1
    }

    // Correction 2: a single (lone) genome with duplicate connection
    // innovation numbers is rejected the same way.
    {
        std::vector<NodeGene> nodes = {NodeGene{0, NodeType::Input}, NodeGene{1, NodeType::Input},
                                        NodeGene{2, NodeType::Output}, NodeGene{3, NodeType::Output}};
        std::vector<ConnectionGene> connections = {ConnectionGene{0, 2, 0.1f, true, 5},
                                                    ConnectionGene{1, 3, 0.2f, true, 5}}; // duplicate innovation 5
        Genome genomeWithDuplicateInnovation(nodes, connections);
        std::vector<Genome> genomes = {genomeWithDuplicateInnovation};

        Speciator speciator;
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig;
        assert(throwsInvalidArgument([&]() { speciator.speciate(genomes, compatConfig, speciationConfig); }) &&
               "a single genome with duplicate connection innovation numbers must be rejected even as the lone "
               "founding genome"); // correction 2
    }

    // Correction 3: a rejected invalid first genome must not consume a
    // SpeciesId -- a subsequent, valid call on the same Speciator must
    // still start at SpeciesId 0.
    {
        std::vector<NodeGene> nodes = {NodeGene{0, NodeType::Input}};
        std::vector<ConnectionGene> connections = {ConnectionGene{0, 99, 0.1f, true, 0}};
        Genome invalidGenome(nodes, connections);

        Speciator speciator;
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig;

        assert(throwsInvalidArgument([&]()
                                      {
                                          std::vector<Genome> invalidBatch = {invalidGenome};
                                          speciator.speciate(invalidBatch, compatConfig, speciationConfig);
                                      }) &&
               "setup: the invalid genome must be rejected");

        std::vector<Genome> validBatch = {makeSimpleGenome(0.0f)};
        const std::vector<Species> species = speciator.speciate(validBatch, compatConfig, speciationConfig);
        assert(species.size() == 1 && species[0].getId() == 0 &&
               "a rejected invalid first genome must not have consumed a SpeciesId"); // correction 3
    }

    // Correction 4: an invalid *later* genome (one that would be compared
    // against an already-existing species) still causes speciate() to
    // throw -- the exception aborts the whole call before it returns, so
    // there is no partially-built species vector for a caller to ever
    // observe; nothing is left dangling or half-updated.
    {
        Genome valid = makeSimpleGenome(0.0f);
        std::vector<NodeGene> nodes = {NodeGene{0, NodeType::Input}};
        std::vector<ConnectionGene> connections = {ConnectionGene{0, 99, 0.1f, true, 0}};
        Genome invalidGenome(nodes, connections);
        std::vector<Genome> genomes = {valid, invalidGenome};

        Speciator speciator;
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig;
        assert(throwsInvalidArgument([&]() { speciator.speciate(genomes, compatConfig, speciationConfig); }) &&
               "an invalid later genome must still cause speciate() to throw, with no partial result returned"); // correction 4
    }

    // Correction 5: a valid lone genome still creates exactly one species,
    // normally, after the fix.
    {
        std::vector<Genome> genomes = {makeSimpleGenome(0.5f)};
        Speciator speciator;
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig;
        const std::vector<Species> species = speciator.speciate(genomes, compatConfig, speciationConfig);
        assert(species.size() == 1 && species[0].getId() == 0 && species[0].size() == 1 &&
               "a valid lone genome must still create exactly one species"); // correction 5
    }

    // Correction 6: all previous Stage 12 verification (items 1-42 above)
    // still passes -- every earlier assertion in this same function ran
    // unmodified before reaching this point.

    // 36: Species holds no fitness data of any kind -- its only fields are
    // id, representative, and memberIndices (see Species.h); there is no
    // fitness, adjustedFitness, championIndex, or stagnation counter to
    // verify, by construction.

    // 37: no genome is ever mutated by speciate() -- reinforced by 21
    // (input genomes unchanged) and 8/22/23 (representatives are
    // independent, stable copies) above.

    // 38: no crossover occurs -- GenomeCrossover is never included or
    // referenced anywhere in Species.h/.cpp or Speciator.h/.cpp.

    // 39: no mutation occurs -- GenomeMutator is never included or
    // referenced anywhere in Species.h/.cpp or Speciator.h/.cpp.

    // 40: no population/reproduction behavior occurs -- no Population,
    // Individual/Agent, selection, or reproduction concept exists anywhere
    // in this stage's files; Speciator only assigns membership.

    // 41: the normal driving loop is unchanged -- Speciator/Species are
    // exercised only from this verify function, never from main()'s
    // simulation loop or createDemonstrationGenome().

    // 42: all previous verification suites still pass -- enforced by
    // main() continuing to call every earlier verify*() function
    // unchanged.

    TraceLog(LOG_INFO, "Speciation verification: all deterministic checks passed");
}

// One-shot, deterministic sanity check of ai::AIController, independent of
// keyboard/render timing. Runs once at startup. Exercises the full
// Car -> Observation -> NeuralNetwork -> AIController -> CarInput loop using
// small hand-built genomes/networks, not the demonstration genome (so this
// verification stays independent of createDemonstrationGenome()'s specific
// weights).
void verifyAIController(const simulation::Track& track)
{
    using ai::AIController;
    using ai::neat::ConnectionGene;
    using ai::neat::Genome;
    using ai::neat::NodeGene;
    using ai::neat::NodeType;
    constexpr float kEps = 1e-4f;

    // Genome with 9 Input + 1 Bias + 2 Output nodes (IDs matching
    // createDemonstrationGenome()'s layout) and no connections, so every
    // network output is deterministically 0 (tanh of an empty sum) unless a
    // test adds its own connections on top.
    auto makeDisconnectedGenome = []()
    {
        Genome genome;
        for (int i = 0; i < ai::NeuralNetwork::kInputCount; ++i)
        {
            genome.addNode(NodeGene{i, NodeType::Input});
        }
        genome.addNode(NodeGene{9, NodeType::Bias});
        genome.addNode(NodeGene{100, NodeType::Output});
        genome.addNode(NodeGene{101, NodeType::Output});
        return genome;
    };

    // 1 & 12: the controller stores and uses a valid two-output network --
    // construction and one update() succeed without throwing.
    {
        AIController controller(ai::neat::buildPhenotype(makeDisconnectedGenome()));
        simulation::Car car(makeCarParams(), track);
        car.reset(kSpawnPosition, kSpawnHeading);
        const simulation::CarInput input = controller.update(car);
        assert(input.steering == 0.0f && input.throttle == 0.5f &&
               "a disconnected network must map to zero steering and neutral (0.5) throttle");
    }

    // 2 & 3: output 0 drives steering, output 1 drives throttle. Bias (always
    // exactly 1.0, independent of any sensor/car geometry) connects only to
    // the steering output, so this test's outcome does not depend on the
    // car's spawn-time sensor readings: a positive weight there must move
    // steering but leave throttle at its neutral 0.5.
    {
        Genome genome = makeDisconnectedGenome();
        genome.addConnection(ConnectionGene{9, 100, 1.0f, true, 0});
        AIController controller(ai::neat::buildPhenotype(genome));

        simulation::Car car(makeCarParams(), track);
        car.reset(kSpawnPosition, kSpawnHeading);
        const simulation::CarInput input = controller.update(car);

        assert(input.steering > 0.5f && "output index 0 must map to steering");
        assert(std::fabs(input.throttle - 0.5f) < kEps && "output index 1 (throttle) must be unaffected");
    }

    // 4, 5 & 6: raw throttle 0 maps to 0.5; negative raw throttle maps below
    // 0.5; positive raw throttle maps above 0.5. Bias -> throttle with a
    // known weight makes the raw throttle output a known, non-zero value.
    {
        Genome zeroGenome = makeDisconnectedGenome(); // no Bias->throttle connection: raw throttle stays 0
        AIController zeroController(ai::neat::buildPhenotype(zeroGenome));
        simulation::Car car(makeCarParams(), track);
        car.reset(kSpawnPosition, kSpawnHeading);
        const simulation::CarInput zeroInput = zeroController.update(car);
        assert(zeroController.getRawThrottleOutput() == 0.0f && "raw throttle must be exactly 0 with no contribution");
        assert(std::fabs(zeroInput.throttle - 0.5f) < kEps && "raw throttle 0 must map to mapped throttle 0.5");

        Genome negativeGenome = makeDisconnectedGenome();
        negativeGenome.addConnection(ConnectionGene{9, 101, -1.0f, true, 0});
        AIController negativeController(ai::neat::buildPhenotype(negativeGenome));
        car.reset(kSpawnPosition, kSpawnHeading);
        const simulation::CarInput negativeInput = negativeController.update(car);
        assert(negativeController.getRawThrottleOutput() < 0.0f && "negative Bias->throttle weight must yield negative raw throttle");
        assert(negativeInput.throttle < 0.5f - kEps && "negative raw throttle must map below 0.5");

        Genome positiveGenome = makeDisconnectedGenome();
        positiveGenome.addConnection(ConnectionGene{9, 101, 1.0f, true, 0});
        AIController positiveController(ai::neat::buildPhenotype(positiveGenome));
        car.reset(kSpawnPosition, kSpawnHeading);
        const simulation::CarInput positiveInput = positiveController.update(car);
        assert(positiveController.getRawThrottleOutput() > 0.0f && "positive Bias->throttle weight must yield positive raw throttle");
        assert(positiveInput.throttle > 0.5f + kEps && "positive raw throttle must map above 0.5");
    }

    // 7 & 8: mapped steering always stays within [-1, 1] and mapped throttle
    // always stays within [0, 1], even when driven by saturating weights and
    // an actively steering/accelerating car (varied Observation values).
    {
        Genome genome = makeDisconnectedGenome();
        genome.addConnection(ConnectionGene{0, 100, 10.0f, true, 0});
        genome.addConnection(ConnectionGene{9, 101, 10.0f, true, 1});
        AIController controller(ai::neat::buildPhenotype(genome));

        simulation::Car car(makeCarParams(), track);
        car.reset(kSpawnPosition, kSpawnHeading);
        simulation::CarInput driveInput;
        driveInput.throttle = 1.0f;
        driveInput.steering = 1.0f;
        for (int i = 0; i < 30 && car.isAlive(); ++i)
        {
            car.update(driveInput, kSimulationDt);
            const simulation::CarInput aiInput = controller.update(car);
            assert(aiInput.steering >= -1.0f && aiInput.steering <= 1.0f && "mapped steering must stay within [-1, 1]");
            assert(aiInput.throttle >= 0.0f && aiInput.throttle <= 1.0f && "mapped throttle must stay within [0, 1]");
        }
        car.reset(kSpawnPosition, kSpawnHeading);
    }

    // 9: the Observation the controller evaluates against is actually built
    // from the provided Car -- a network wired straight from sensor 2
    // (center, ID 2) to steering must react to that specific Car's own
    // center-sensor reading.
    {
        Genome genome = makeDisconnectedGenome();
        genome.addConnection(ConnectionGene{2, 100, 1.0f, true, 0});
        AIController controller(ai::neat::buildPhenotype(genome));

        simulation::Car car(makeCarParams(), track);
        car.reset(kSpawnPosition, kSpawnHeading);
        controller.update(car);

        const float expectedCenterSensor = car.getSensors()[2].normalizedDistance;
        assert(std::fabs(controller.getLastObservation().values[2] - expectedCenterSensor) < kEps &&
               "AIController's Observation must be built from the provided Car's own sensor readings");
    }

    // 10: repeated calls with unchanged Car state are deterministic.
    {
        Genome genome = makeDisconnectedGenome();
        genome.addConnection(ConnectionGene{0, 100, 0.7f, true, 0});
        genome.addConnection(ConnectionGene{9, 101, -0.3f, true, 1});
        AIController controller(ai::neat::buildPhenotype(genome));

        simulation::Car car(makeCarParams(), track);
        car.reset(kSpawnPosition, kSpawnHeading);

        const simulation::CarInput first = controller.update(car);
        const simulation::CarInput second = controller.update(car);
        assert(first.steering == second.steering && first.throttle == second.throttle &&
               "repeated updates against an unchanged Car must produce identical CarInput");
    }

    // 11: the controller does not mutate the Car it reads from.
    {
        Genome genome = makeDisconnectedGenome();
        genome.addConnection(ConnectionGene{0, 100, 0.5f, true, 0});
        AIController controller(ai::neat::buildPhenotype(genome));

        simulation::Car car(makeCarParams(), track);
        car.reset(kSpawnPosition, kSpawnHeading);
        const Vector2 positionBefore = car.getPosition();
        const Vector2 velocityBefore = car.getVelocity();
        const float headingBefore = car.getHeading();

        controller.update(car);

        assert(car.getPosition().x == positionBefore.x && car.getPosition().y == positionBefore.y &&
               "AIController::update must not move the Car");
        assert(car.getVelocity().x == velocityBefore.x && car.getVelocity().y == velocityBefore.y &&
               "AIController::update must not change the Car's velocity");
        assert(car.getHeading() == headingBefore && "AIController::update must not change the Car's heading");
    }

    // 13: a disabled Genome connection remains behaviorally inactive after
    // phenotype construction -- disabling the same steering connection used
    // in test 2 must leave steering at its neutral 0.
    {
        Genome genome = makeDisconnectedGenome();
        genome.addConnection(ConnectionGene{0, 100, 1.0f, false, 0});
        AIController controller(ai::neat::buildPhenotype(genome));

        simulation::Car car(makeCarParams(), track);
        car.reset(kSpawnPosition, kSpawnHeading);
        const simulation::CarInput input = controller.update(car);
        assert(input.steering == 0.0f && "a disabled connection must not affect evaluation after phenotype construction");
    }

    // Dead car: the controller must not keep evaluating the network, and
    // must apply neutral CarInput instead.
    {
        Genome genome = makeDisconnectedGenome();
        genome.addConnection(ConnectionGene{9, 101, 1.0f, true, 0}); // would otherwise raise throttle above 0.5
        AIController controller(ai::neat::buildPhenotype(genome));

        simulation::Car car(makeCarParams(), track);
        car.reset(kSpawnPosition, kSpawnHeading);
        simulation::CarInput driveOffTrack;
        driveOffTrack.throttle = 1.0f;
        driveOffTrack.steering = 0.0f;
        for (int i = 0; i < 300 && car.isAlive(); ++i)
        {
            car.update(driveOffTrack, kSimulationDt);
        }
        assert(!car.isAlive() && "driving straight for 5s must leave the road band and kill the car");

        const simulation::CarInput deadInput = controller.update(car);
        assert(deadInput.steering == 0.0f && deadInput.throttle == 0.0f &&
               "a dead car must receive neutral CarInput from AIController, not a network-derived one");
        car.reset(kSpawnPosition, kSpawnHeading);
    }

    TraceLog(LOG_INFO, "AI controller verification: all deterministic checks passed");
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

// One-shot, deterministic sanity check of simulation::TrackProgress,
// independent of keyboard/render timing. Runs once at startup. TrackProgress
// never controls the Car -- only Car::reset() (to place the car at precise,
// hand-computed positions) and the Track's own sampled centerline/arc-length
// data are used here, plus one short real-driving check for direction
// sanity.
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

        // Expected checkpoint after reset is the first checkpoint strictly
        // ahead of the car, in order -- not unconditionally 0, since the
        // car may spawn anywhere around the fixed checkpoint ring.
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

    // 3: progress increases in the intended forward direction -- driven
    // with real Car physics (throttle only, no steering) from the actual
    // spawn pose, not with synthetic positions.
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
        assert(car.isAlive() && "the car must still be on the bottom straight after 0.5s from spawn");
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

    // 7, 8 & review-requirement 2/3/6: checkpoints are ordered-traversal
    // state computed independently from raw position deltas (never from
    // getBestProgress()) -- they advance strictly in order, several can be
    // credited within one valid forward update, and a lap is counted
    // exactly once a full ordered cycle of kCheckpointCount checkpoints has
    // been awarded since reset. Reset at a position that does NOT coincide
    // with a checkpoint boundary (0.03), so completing one lap requires the
    // full kCheckpointCount checkpoints, not one fewer.
    {
        simulation::TrackProgress progress(track);
        car.reset(positionAtLapPosition(track, 0.03f), kSpawnHeading);
        progress.reset(car);
        assert(progress.getExpectedCheckpoint() == 1 && progress.getTotalCheckpointsPassed() == 0 &&
               "reset at lap position 0.03 must expect checkpoint 1 next");

        // Six forward steps of 0.15 laps each starting from 0.03, each
        // spanning one or more checkpoint boundaries -- exact expected
        // (total passed, next expected) after each step, worked out from
        // the fixed checkpoint positions i/16.
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
        assert(progress.getLapCount() == 0 &&
               "14 of 16 checkpoints passed must not yet complete a lap (review req. 6, negative case)");

        // Forward seam crossing 0.93 -> 0.08 (delta corrects to +0.15) is
        // exactly the update that awards the final two checkpoints (15 and
        // the wrap to 0) -- the lap must complete here, in the same update
        // that both validates the last checkpoints AND crosses the seam
        // (review req. 5: seam crossing alone is not what completes it).
        car.reset(positionAtLapPosition(track, 0.08f), kSpawnHeading);
        progress.update(car);
        assert(progress.getTotalCheckpointsPassed() == 17 && progress.getExpectedCheckpoint() == 2 &&
               progress.getLapCount() == 1 &&
               "a full ordered traversal of all checkpoints must complete the lap exactly once, at the seam crossing that finishes it");

        // Review requirement 1: jumping to a later angular position without
        // crossing the intervening checkpoints in order (an implausible,
        // teleport-sized delta) must not award anything, even though the
        // raw angle itself does move forward.
        const int totalBeforeJump = progress.getTotalCheckpointsPassed();
        const int expectedBeforeJump = progress.getExpectedCheckpoint();
        const int lapsBeforeJump = progress.getLapCount();
        car.reset(positionAtLapPosition(track, 0.43f), kSpawnHeading); // 0.08 -> 0.43 is a 0.35 jump, > the plausibility threshold
        progress.update(car);
        assert(std::fabs(progress.getLapPosition() - 0.43f) < kEps &&
               "the raw lap position must still reflect the car's actual (teleported) position");
        assert(progress.getTotalCheckpointsPassed() == totalBeforeJump && progress.getExpectedCheckpoint() == expectedBeforeJump &&
               progress.getLapCount() == lapsBeforeJump &&
               "an implausible jump must not award checkpoints or laps even if it lands at a later angle");

        // Review requirement 4: backward movement (still within the
        // plausible-delta range) must not award checkpoints either.
        car.reset(positionAtLapPosition(track, 0.35f), kSpawnHeading); // 0.43 -> 0.35 is backward
        progress.update(car);
        assert(progress.getTotalCheckpointsPassed() == totalBeforeJump && progress.getExpectedCheckpoint() == expectedBeforeJump &&
               progress.getLapCount() == lapsBeforeJump && "backward movement must not award checkpoints or laps");

        car.reset(kSpawnPosition, kSpawnHeading);
    }

    // Review requirement 7: reset restores expected checkpoint, checkpoint
    // count and lap count correctly for whatever position it is given, not
    // just back to a fixed baseline.
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

    // 14: TrackProgress derives lap position from its own injected Track,
    // not a second hardcoded track shape -- a differently shaped Track
    // (control points shifted) yields a different lap position for the
    // exact same world position.
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
    // (see verifyTrack) before a TrackProgress could ever be built from
    // them -- TrackProgress itself performs no redundant validation, since
    // it only ever receives an already-validated Track by reference.

    TraceLog(LOG_INFO, "Track progress verification: all deterministic checks passed");
}

// One-shot, deterministic sanity check of ai::FitnessEvaluator (Fitness v2,
// Stage 15A), independent of keyboard/render timing. Runs once at startup.
// FitnessEvaluator reads only simulation::Car::isAlive() and
// simulation::TrackProgress's getters -- no Genome or NeuralNetwork is
// touched here. Its update() signature has no mode parameter at all, so
// manual and AI control paths need no separate fitness logic -- both simply
// call the same update() with whatever Car state resulted from that frame.
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

    // 2, 3 & 40: there is no positive reward merely from elapsed survival
    // time, and no-progress waiting does not increase fitness -- with the
    // car stationary at spawn (zero progress throughout), fitness must stay
    // exactly zero for as long as the car sits there (well under the
    // no-progress timeout), never creeping upward the way Fitness v1's
    // elapsedTime * kSurvivalRewardPerSecond term did.
    {
        car.reset(kSpawnPosition, kSpawnHeading);
        simulation::TrackProgress progress(track);
        progress.reset(car);
        ai::FitnessEvaluator evaluator;
        evaluator.reset();

        for (int i = 0; i < 240 && !evaluator.isEvaluationFinished(); ++i) // 4s of no movement, under the 5s timeout
        {
            progress.update(car);
            evaluator.update(car, progress, kSimulationDt);
            assert(evaluator.getFitness() == 0.0f &&
                   "standing still at zero progress must never earn positive fitness from elapsed time alone");
        }
        assert(!evaluator.isEvaluationFinished() && "4 seconds of no movement must stay under the no-progress timeout");
    }

    // 14: waiting cannot improve progress-rate reward -- once some progress
    // has been made, standing still afterward must never increase
    // progressRateReward (it can only shrink as elapsedTime grows with
    // bestProgress held fixed), and therefore never increase total fitness
    // either.
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

    // 13: backward movement must not increase the progress-derived part of
    // fitness (best progress, checkpoints, laps) -- unaffected by Fitness
    // v2, TrackProgress's own anti-exploit best-progress tracking already
    // guarantees this; re-verified here against the base fitness term.
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
    // total fitness than the same progress reached in more time -- Car A
    // (5s) must beat Car B (10s) at the identical 0.5-lap progress point.
    {
        auto reachProgressInTime = [&](float targetProgress, float totalTime) -> float
        {
            simulation::Car localCar(makeCarParams(), track);
            localCar.reset(positionAtLapPosition(track, 0.0f), kSpawnHeading);
            simulation::TrackProgress localProgress(track);
            localProgress.reset(localCar);
            ai::FitnessEvaluator evaluator;
            evaluator.reset();

            // Advance in small increments -- each well within
            // TrackProgress's plausibility gate -- so bestProgress
            // legitimately reaches targetProgress (a direct one-shot jump
            // of more than ~0.2 laps would be rejected as implausible and
            // never register). No evaluator time is spent on these interim
            // steps.
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

            // Only now spend the desired total elapsed time reaching this
            // point -- a single update() call is enough since
            // FitnessEvaluator reads TrackProgress's *current* state, not
            // an integral over time.
            evaluator.update(localCar, localProgress, totalTime);
            return evaluator.getFitness();
        };

        const float fitnessCarA = reachProgressInTime(0.5f, 5.0f);
        const float fitnessCarB = reachProgressInTime(0.5f, 10.0f);
        assert(fitnessCarA > fitnessCarB &&
               "the same progress reached in less time must give higher fitness (Car A/5s must beat Car B/10s)");
    }

    // 9: substantially greater progress still beats a much faster car that
    // only reached a small fraction of the track -- Car C (0.8 laps in 10s)
    // must beat Car D (0.2 laps in a mere 2s), because kProgressRateScale is
    // deliberately small relative to kProgressPointsPerLap.
    {
        auto reachProgressInTime = [&](float targetProgress, float totalTime) -> float
        {
            simulation::Car localCar(makeCarParams(), track);
            localCar.reset(positionAtLapPosition(track, 0.0f), kSpawnHeading);
            simulation::TrackProgress localProgress(track);
            localProgress.reset(localCar);
            ai::FitnessEvaluator evaluator;
            evaluator.reset();

            // See the identical stepping approach and rationale in the
            // previous test block.
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
    // smallTimeEpsilon) exactly, and stays finite (and correctly computed)
    // even at elapsedTime == 0 (using the documented smallTimeEpsilon =
    // 0.1f floor).
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
    // here with real Car physics (nonzero, varying velocity throughout,
    // unlike the synthetic teleports above) and cross-checked against the
    // exact bestProgress/elapsedTime formula; if velocity fed into the
    // formula anywhere, this exact match would not hold.
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
    // time (matching elapsedTime, since the first lap starts at time 0); a
    // second, slower lap records a new last-lap time but does not worsen
    // the recorded best lap time; a third, faster lap then does become the
    // new best.
    {
        car.reset(positionAtLapPosition(track, 0.0f), kSpawnHeading);
        simulation::TrackProgress progress(track);
        progress.reset(car);
        ai::FitnessEvaluator evaluator;
        evaluator.reset();

        assert(!evaluator.hasCompletedLap() && evaluator.getBestLapTime() == 0.0f && evaluator.getLastLapTime() == 0.0f &&
               "no completed lap yet must read as the documented zero sentinel"); // 20 (setup half)

        // Advances progress forward by slightly more than one full lap's
        // worth of arc length, in small increments each well within
        // TrackProgress's plausibility gate (a direct one-shot jump of a
        // full lap would be rejected as implausible), then spends exactly
        // lapTime seconds of evaluator time on the whole attempt via a
        // single update() call at the end -- so the recorded lap time comes
        // out to exactly lapTime. The slight (2%) overshoot past exactly
        // 1.0 lap deliberately avoids landing the final sample ambiguously
        // right at the seam itself, where float rounding inside
        // Track::getPointAtDistance() could put it on either side and miss
        // wrapping past the final checkpoint by a hair -- 2% of a lap is
        // still far short of the next checkpoint (1/16 = 6.25% of a lap
        // apart), so it cannot spuriously cross an extra one.
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

        // First lap: 8s total.
        driveOneLap(8.0f);
        assert(progress.getLapCount() == 1 && "the first full lap must complete exactly one lap");
        assert(evaluator.hasCompletedLap() && "completing a lap must set hasCompletedLap()"); // 15 (part 1)
        assert(std::fabs(evaluator.getLastLapTime() - evaluator.getElapsedTime()) < kEps &&
               "the first lap's time must equal elapsedTime, since the first lap starts at time 0"); // 15 (part 2)
        assert(std::fabs(evaluator.getBestLapTime() - evaluator.getLastLapTime()) < kEps &&
               "the only completed lap so far must also be the best lap");
        const float firstLapTime = evaluator.getLastLapTime(); // 8s

        // Second lap: slower (12s) -- new last-lap time, but best must stay at firstLapTime.
        driveOneLap(12.0f);
        assert(progress.getLapCount() == 2 && "the second full lap must complete a second lap");
        assert(std::fabs(evaluator.getLastLapTime() - 12.0f) < kEps &&
               "a second, slower lap must record a new last-lap time"); // 16
        assert(std::fabs(evaluator.getBestLapTime() - firstLapTime) < kEps &&
               "a slower later lap must not worsen (increase) the recorded best lap time"); // 17 & 18

        // Third lap: faster (4s) -- best must now update to this new fastest time.
        driveOneLap(4.0f);
        assert(progress.getLapCount() == 3 && "the third full lap must complete a third lap");
        assert(std::fabs(evaluator.getLastLapTime() - 4.0f) < kEps && "the third lap's time must be recorded as the new last-lap time");
        assert(std::fabs(evaluator.getBestLapTime() - 4.0f) < kEps &&
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
            // above for why: a direct one-shot jump of a full lap would be
            // rejected by TrackProgress's plausibility gate), spending
            // lapTime seconds of evaluator time on the whole attempt via a
            // single update() call at the end.
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

        float p = 0.0f;
        for (int second = 0; second < 61 && !evaluator.isEvaluationFinished(); ++second)
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

    // Meaningful progress resets the no-progress timer (unchanged from
    // Fitness v1 -- this is TrackProgress-adjacent timer bookkeeping inside
    // FitnessEvaluator, not part of the v2 scoring formula itself).
    {
        car.reset(positionAtLapPosition(track, 0.0f), kSpawnHeading);
        simulation::TrackProgress progress(track);
        progress.reset(car);
        ai::FitnessEvaluator evaluator;
        evaluator.reset();

        // Stand still for 3 seconds (under the 5s timeout).
        for (int i = 0; i < 180; ++i)
        {
            progress.update(car);
            evaluator.update(car, progress, kSimulationDt);
        }
        assert(!evaluator.isEvaluationFinished() && "3 seconds of no movement must stay under the timeout");

        // A meaningful forward nudge must reset the no-progress timer.
        car.reset(positionAtLapPosition(track, 0.05f), kSpawnHeading);
        progress.update(car);
        evaluator.update(car, progress, kSimulationDt);

        // A further 4 seconds of standing still (< 5s since the nudge)
        // must still not finish the evaluation, proving the timer reset.
        for (int i = 0; i < 240; ++i)
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

        // Drive one full lap via small increments (a direct one-shot jump
        // of a full lap would be rejected by TrackProgress's plausibility
        // gate -- see driveOneLap's comment earlier in this function).
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
    }
    return "-";
}

// Maps a normalized progress rank (0 = best/leading, 1 = worst/trailing)
// to a color running green -> yellow -> red, for population-wide car
// rendering. Purely a rendering helper -- reads no evolution state and
// writes nothing.
Color progressRankColor(float normalizedRank)
{
    const float clamped = std::clamp(normalizedRank, 0.0f, 1.0f);
    if (clamped < 0.5f)
    {
        const float t = clamped / 0.5f; // green -> yellow
        return Color{static_cast<unsigned char>(t * 255.0f), 255, 0, 255};
    }
    const float t = (clamped - 0.5f) / 0.5f; // yellow -> red
    return Color{255, static_cast<unsigned char>((1.0f - t) * 255.0f), 0, 255};
}

// Draws one population member's car body in `color`. Sensor rays and the
// velocity-vector overlay are drawn only for the single highlighted
// individual (Stage 13 explicitly avoids rendering sensors for every car).
void drawIndividualCar(const simulation::Car& car, Color color, bool highlighted)
{
    const std::array<Vector2, 4> corners = car.getCorners();
    DrawTriangleFan(corners.data(), static_cast<int>(corners.size()), color);

    if (highlighted && car.isAlive())
    {
        const Vector2 position = car.getPosition();
        const Vector2 velocity = car.getVelocity();
        const Vector2 tip = {position.x + velocity.x * 0.25f, position.y + velocity.y * 0.25f};
        DrawLineEx(position, tip, 2.0f, YELLOW);

        const Vector2 origin = car.getSensorOrigin();
        for (const simulation::SensorReading& sensor : car.getSensors())
        {
            DrawLineEx(origin, sensor.endPoint, 1.5f, Color{80, 255, 120, 255});
            DrawCircleV(sensor.endPoint, 3.0f, Color{255, 90, 40, 255});
        }
    }
}

// Chooses which single individual gets its sensors/velocity vector
// rendered this frame: the currently-running individual with the highest
// best-progress, or -- once every individual has finished -- the
// highest-fitness individual. Both selection rules break ties by lower
// index (via strict '>' comparisons only), so the result is a
// deterministic function of the population's current state alone. A pure
// query: reads population, mutates nothing.
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

void drawPopulationPanel(const ai::neat::Population& population, std::size_t highlightedIndex)
{
    DrawRectangle(kSimWidth, 0, kPanelWidth, kScreenHeight, Color{30, 30, 30, 255});

    const int x = kSimWidth + 20;
    int y = 20;
    const int lineHeight = 22;

    DrawText("STAGE 14B - HARD TRAINING TRACK", x, y, 20, RAYWHITE);
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
    std::snprintf(line, sizeof(line), "Species: %d", static_cast<int>(population.getSpeciesCount()));
    DrawText(line, x, y, 16, LIGHTGRAY);
    y += lineHeight;
    DrawText("REPRODUCTION: SPECIES-AWARE", x, y, 16, SKYBLUE);
    y += lineHeight * 2;

    // Stage 16: locate which of the current generation's species the
    // highlighted individual belongs to (by membership, not by rebuilding
    // any separate lookup), and -- if a generation transition has already
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

    // FITNESS V2 breakdown (Stage 15A) -- see FitnessEvaluator.h for the
    // exact formula each of these terms comes from.
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
}

namespace population_verify
{

using ai::neat::CompatibilityConfig;
using ai::neat::ConnectionGene;
using ai::neat::CrossoverConfig;
using ai::neat::Genome;
using ai::neat::GenomeCrossover;
using ai::neat::GenomeMutator;
using ai::neat::Individual;
using ai::neat::InnovationNumber;
using ai::neat::InnovationTracker;
using ai::neat::isBetterTournamentCandidate;
using ai::neat::MutationConfig;
using ai::neat::NodeGene;
using ai::neat::NodeId;
using ai::neat::NodeType;
using ai::neat::Population;
using ai::neat::PopulationConfig;
using ai::neat::SpeciationConfig;
using ai::neat::Speciator;

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

PopulationConfig makeTestPopulationConfig(std::size_t populationSize, std::uint32_t seed)
{
    PopulationConfig config;
    config.populationSize = populationSize;
    config.eliteCount = 1;
    config.tournamentSize = std::min<std::size_t>(3, populationSize);
    config.randomSeed = seed;
    return config;
}

// A genome that drives at near-maximum throttle with zero steering
// (Bias -> Throttle only, weight large enough that tanh saturates close to
// 1.0) -- from the fixed spawn pose this reliably leaves the road band and
// collides within a few hundred simulation steps, exactly like the
// deliberate "drive off track" scenario verifyCar() itself exercises. Used
// only to keep generation-transition tests fast and their step-count bound
// tight; it carries no meaning beyond that.
Genome makeCrashGenome()
{
    Genome genome;
    for (int i = 0; i < ai::NeuralNetwork::kInputCount; ++i)
    {
        genome.addNode(NodeGene{i, NodeType::Input});
    }
    genome.addNode(NodeGene{9, NodeType::Bias});
    genome.addNode(NodeGene{100, NodeType::Output});
    genome.addNode(NodeGene{101, NodeType::Output});
    genome.addConnection(ConnectionGene{9, 101, 5.0f, true, 0});
    return genome;
}

bool connectionsMatch(const std::vector<ConnectionGene>& a, const std::vector<ConnectionGene>& b)
{
    if (a.size() != b.size())
    {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        if (a[i].getSourceId() != b[i].getSourceId() || a[i].getTargetId() != b[i].getTargetId() ||
            a[i].getWeight() != b[i].getWeight() || a[i].isEnabled() != b[i].isEnabled() ||
            a[i].getInnovationNumber() != b[i].getInnovationNumber())
        {
            return false;
        }
    }
    return true;
}

} // namespace population_verify

// One-shot, deterministic sanity check of ai::neat::Population/Individual,
// covering the first complete generation loop end to end. Independent of
// rendering/keyboard timing (uses fixed-size kSimulationDt steps, exactly
// like every other verify*() function). Runs once at startup. This is
// Stage 13/15's original suite, left intentionally unchanged in scope --
// species-restricted mating, adjusted fitness, fitness sharing, and
// offspring allocation (Stage 16) are exercised separately, by
// verifySpeciesAwareReproduction() below. Persistent species lineage,
// stagnation, extinction, hall of fame, training history, and save/load
// logic remain untested anywhere (still out of scope for the whole
// codebase).
void verifyPopulation(const simulation::Track& track)
{
    using namespace population_verify;

    // 1, 2 & 3: invalid PopulationConfig fields are rejected.
    {
        const Genome base = createDemonstrationGenome();
        const MutationConfig mutationConfig;
        const CrossoverConfig crossoverConfig;
        const CompatibilityConfig compatibilityConfig;
        const SpeciationConfig speciationConfig;

        auto build = [&](const PopulationConfig& config)
        {
            Population population(base, track, makeCarParams(), kSpawnPosition, kSpawnHeading,
                                   config, mutationConfig, crossoverConfig, compatibilityConfig, speciationConfig);
            (void)population;
        };

        PopulationConfig zeroSize = makeTestPopulationConfig(6, 1u);
        zeroSize.populationSize = 0;
        assert(throwsInvalidArgument([&]() { build(zeroSize); }) && "populationSize of 0 must be rejected"); // 1
        PopulationConfig oneSize = makeTestPopulationConfig(6, 1u);
        oneSize.populationSize = 1;
        assert(throwsInvalidArgument([&]() { build(oneSize); }) && "populationSize of 1 must be rejected"); // 1 (continued)

        PopulationConfig zeroElite = makeTestPopulationConfig(6, 1u);
        zeroElite.eliteCount = 0;
        assert(throwsInvalidArgument([&]() { build(zeroElite); }) && "eliteCount of 0 must be rejected"); // 2
        PopulationConfig tooManyElite = makeTestPopulationConfig(6, 1u);
        tooManyElite.eliteCount = tooManyElite.populationSize;
        assert(throwsInvalidArgument([&]() { build(tooManyElite); }) &&
               "eliteCount == populationSize must be rejected"); // 2 (continued)

        PopulationConfig zeroTournament = makeTestPopulationConfig(6, 1u);
        zeroTournament.tournamentSize = 0;
        assert(throwsInvalidArgument([&]() { build(zeroTournament); }) && "tournamentSize of 0 must be rejected"); // 3
        PopulationConfig tooBigTournament = makeTestPopulationConfig(6, 1u);
        tooBigTournament.tournamentSize = tooBigTournament.populationSize + 1;
        assert(throwsInvalidArgument([&]() { build(tooBigTournament); }) &&
               "tournamentSize > populationSize must be rejected"); // 3 (continued)
    }

    // 4, 5, 6, 7 & 8: initial population size/generation/structure.
    {
        const Genome base = createDemonstrationGenome();
        const PopulationConfig popConfig = makeTestPopulationConfig(6, 42u);
        const MutationConfig mutationConfig;
        const CrossoverConfig crossoverConfig;
        const CompatibilityConfig compatibilityConfig;
        const SpeciationConfig speciationConfig;

        Population population(base, track, makeCarParams(), kSpawnPosition, kSpawnHeading,
                               popConfig, mutationConfig, crossoverConfig, compatibilityConfig, speciationConfig);

        assert(population.size() == 6 && "initial population must have the exact configured size"); // 4
        assert(population.getGeneration() == 0 && "initial generation must be 0"); // 5

        const Genome& genome0 = population.getIndividual(0).getGenome();
        assert(genome0.nodes().size() == base.nodes().size() &&
               connectionsMatch(genome0.connections(), base.connections()) &&
               "individual 0 must preserve the base Genome exactly"); // 6

        bool sawWeightChange = false;
        for (std::size_t i = 1; i < population.size(); ++i)
        {
            const Genome& g = population.getIndividual(i).getGenome();
            assert(g.nodes().size() == base.nodes().size() && g.connections().size() == base.connections().size() &&
                   "other individuals must preserve the base structure (node/connection counts)"); // 7
            for (std::size_t j = 0; j < base.connections().size(); ++j)
            {
                const ConnectionGene& b = base.connections()[j];
                const ConnectionGene& c = g.connections()[j];
                assert(c.getSourceId() == b.getSourceId() && c.getTargetId() == b.getTargetId() &&
                       c.getInnovationNumber() == b.getInnovationNumber() &&
                       "other individuals must preserve base structure/innovation numbers exactly"); // 7 (continued)
                if (c.getWeight() != b.getWeight())
                {
                    sawWeightChange = true;
                }
            }
        }
        assert(sawWeightChange && "initial weight mutation must be able to alter at least one weight"); // 8
    }

    // 9, 10, 45 & 46: each Individual owns fully independent Car and
    // evaluation state -- one finishing (colliding) never affects another,
    // proven directly at the object level rather than relying on emergent
    // trajectory divergence.
    {
        const Genome crashGenome = makeCrashGenome();
        Individual individualA(crashGenome, track, makeCarParams(), kSpawnPosition, kSpawnHeading);
        Individual individualB(crashGenome, track, makeCarParams(), kSpawnPosition, kSpawnHeading);

        for (int step = 0; step < 400 && !individualA.isFinished(); ++step)
        {
            individualA.update(kSimulationDt);
        }
        assert(individualA.isFinished() && "setup: the crash genome must finish (collide) within 400 steps");

        assert(!individualB.isFinished() &&
               "an unrelated Individual must not be affected by another Individual finishing"); // 9 & 45 (part 1)
        assert(individualB.getFitness() == 0.0f && individualB.getFitnessEvaluator().getElapsedTime() == 0.0f &&
               "an unrelated, never-updated Individual must keep its own independent evaluation state"); // 10 & 46
        assert(individualB.getCar().isAlive() &&
               individualB.getCar().getPosition().x == kSpawnPosition.x &&
               individualB.getCar().getPosition().y == kSpawnPosition.y &&
               "an unrelated Individual's Car must remain independently alive at its own spawn position"); // 9 (part 2)
    }

    // 11 & 12: the global InnovationTracker starts strictly above every
    // base NodeId/innovation number.
    {
        const Genome base = createDemonstrationGenome();
        NodeId maxNodeId = -1;
        for (const NodeGene& node : base.nodes())
        {
            maxNodeId = std::max(maxNodeId, node.getId());
        }
        InnovationNumber maxInnovation = -1;
        for (const ConnectionGene& connection : base.connections())
        {
            maxInnovation = std::max(maxInnovation, connection.getInnovationNumber());
        }

        const PopulationConfig popConfig = makeTestPopulationConfig(6, 2u);
        const MutationConfig mutationConfig;
        const CrossoverConfig crossoverConfig;
        const CompatibilityConfig compatibilityConfig;
        const SpeciationConfig speciationConfig;
        Population population(base, track, makeCarParams(), kSpawnPosition, kSpawnHeading,
                               popConfig, mutationConfig, crossoverConfig, compatibilityConfig, speciationConfig);

        assert(population.getInnovationTracker().getNextAvailableNodeId() > maxNodeId &&
               "the global InnovationTracker must start above every base NodeId"); // 11
        assert(population.getInnovationTracker().getNextAvailableInnovation() > maxInnovation &&
               "the global InnovationTracker must start above every base connection innovation"); // 12

        assert(population.getLastGenerationBestFitness() == 0.0f &&
               "a freshly constructed Population's last-generation best fitness must start at 0"); // 39 (initial value)
    }

    // 13: update() advances an unfinished individual's evaluation.
    {
        const Genome base = createDemonstrationGenome();
        const PopulationConfig popConfig = makeTestPopulationConfig(4, 3u);
        const MutationConfig mutationConfig;
        const CrossoverConfig crossoverConfig;
        const CompatibilityConfig compatibilityConfig;
        const SpeciationConfig speciationConfig;
        Population population(base, track, makeCarParams(), kSpawnPosition, kSpawnHeading,
                               popConfig, mutationConfig, crossoverConfig, compatibilityConfig, speciationConfig);

        const float elapsedBefore = population.getIndividual(0).getFitnessEvaluator().getElapsedTime();
        population.update(kSimulationDt);
        assert(population.getIndividual(0).getFitnessEvaluator().getElapsedTime() > elapsedBefore &&
               "update() must advance an unfinished individual's evaluation"); // 13
    }

    // 14: a finished individual no longer updates.
    {
        const Genome crashGenome = makeCrashGenome();
        Individual individual(crashGenome, track, makeCarParams(), kSpawnPosition, kSpawnHeading);
        for (int step = 0; step < 400 && !individual.isFinished(); ++step)
        {
            individual.update(kSimulationDt);
        }
        assert(individual.isFinished() && "setup: the crash genome must finish within 400 steps");

        const float frozenFitness = individual.getFitness();
        const float frozenElapsed = individual.getFitnessEvaluator().getElapsedTime();
        const Vector2 frozenPosition = individual.getCar().getPosition();
        for (int step = 0; step < 20; ++step)
        {
            individual.update(kSimulationDt);
        }
        assert(individual.getFitness() == frozenFitness && individual.getFitnessEvaluator().getElapsedTime() == frozenElapsed &&
               individual.getCar().getPosition().x == frozenPosition.x &&
               individual.getCar().getPosition().y == frozenPosition.y &&
               "a finished individual must not update further"); // 14
    }

    // 15, 16, 17, 18, 19, 20 & 39: a full generation transition -- one
    // finished individual alone does not finish the generation; once
    // every individual has finished, reproduction runs exactly once
    // (generation 0 -> 1); population size stays constant; elites are
    // copied byte-for-byte unchanged; last-generation best fitness updates.
    std::vector<Genome> gen0GenomesSnapshot;
    std::vector<float> gen0FitnessSnapshot;
    PopulationConfig transitionPopConfig = makeTestPopulationConfig(8, 11u);
    {
        const Genome crashGenome = makeCrashGenome();
        const MutationConfig mutationConfig; // default: individuals 1..N-1 get varied weights
        const CrossoverConfig crossoverConfig;
        const CompatibilityConfig compatibilityConfig;
        const SpeciationConfig speciationConfig;
        Population population(crashGenome, track, makeCarParams(), kSpawnPosition, kSpawnHeading,
                               transitionPopConfig, mutationConfig, crossoverConfig, compatibilityConfig, speciationConfig);

        // Wait for the first individual to finish -- whichever one it is;
        // weight-mutated individuals need not all crash at the same step,
        // so with 8 individuals it is very likely at least one is still
        // running when the first finishes.
        for (int step = 0; step < 400 && population.getFinishedCount() == 0; ++step)
        {
            population.update(kSimulationDt);
        }
        assert(population.getFinishedCount() > 0 && "setup: at least one individual must finish within 400 steps");
        if (population.getRunningCount() > 0)
        {
            assert(population.getGeneration() == 0 &&
                   "one (or more, but not all) finished individual(s) must not trigger a generation transition"); // 15
        }

        // Re-snapshot every iteration, immediately before each update()
        // call: whichever iteration turns out to be the one where
        // update() internally detects full completion and reproduces, the
        // snapshot taken just before that exact call captures precisely
        // the final generation-0 genomes/fitness values Population itself
        // used -- there is no other way to observe them from outside,
        // since generation-finish detection and reproduction happen
        // together inside a single update() call. Also record how many
        // individuals were *already* finished going into that exact call:
        // if it is every individual, the snapshot is provably exact (a
        // finished individual's Genome/fitness cannot change); if it is
        // fewer, the one or more still-running individuals could gain a
        // last, possibly checkpoint-sized burst of fitness during that very
        // call (see the ranking check below for how this is handled).
        std::size_t finishedBeforeTransitionCall = 0;
        for (int step = 0; step < 4000 && population.getGeneration() == 0; ++step)
        {
            gen0GenomesSnapshot.clear();
            gen0FitnessSnapshot.clear();
            for (std::size_t i = 0; i < population.size(); ++i)
            {
                gen0GenomesSnapshot.push_back(population.getIndividual(i).getGenome());
                gen0FitnessSnapshot.push_back(population.getIndividual(i).getFitness());
            }
            finishedBeforeTransitionCall = population.getFinishedCount();
            population.update(kSimulationDt);
        }
        assert(population.getGeneration() == 1 &&
               "once every individual has finished, a generation transition must occur exactly once"); // 16 & 17
        assert(population.size() == transitionPopConfig.populationSize &&
               "population size must remain constant across a generation transition"); // 18

        // 39: last-generation best fitness must have updated away from its
        // initial value (0, per a fresh Population -- verified separately
        // below) to a positive value reflecting the just-finished
        // generation's real evaluation data. An exact numeric match against
        // gen0FitnessSnapshot is not attempted: that snapshot is captured
        // immediately *before* each update() call as a best-effort
        // approximation (there is no way to observe Population's internal
        // fitness values at the exact instant reproduce() reads them,
        // since detection and reproduction happen inside one update() call)
        // and so can under-count by at most one deltaTime step's worth of
        // accumulation relative to the true final value.
        const float staleObservedBest = *std::max_element(gen0FitnessSnapshot.begin(), gen0FitnessSnapshot.end());
        assert(population.getLastGenerationBestFitness() > 0.0f &&
               population.getLastGenerationBestFitness() >= staleObservedBest - 1.0f &&
               "last-generation best fitness must update to reflect the just-finished generation's real fitness"); // 39

        std::vector<std::size_t> ranked(gen0GenomesSnapshot.size());
        for (std::size_t i = 0; i < ranked.size(); ++i)
        {
            ranked[i] = i;
        }
        std::sort(ranked.begin(), ranked.end(),
                  [&gen0FitnessSnapshot](std::size_t a, std::size_t b)
                  {
                      if (gen0FitnessSnapshot[a] != gen0FitnessSnapshot[b])
                      {
                          return gen0FitnessSnapshot[a] > gen0FitnessSnapshot[b];
                      }
                      return a < b;
                  });

        // Whether the very last snapshot (taken immediately before the
        // transition-triggering update() call) is provably exact: true only
        // if every individual was already finished going into that call, in
        // which case nothing about their Genome/fitness could still change.
        // If one or more individuals were still running, that call could
        // have awarded any of them a final burst of fitness (including a
        // whole checkpoint's worth, if their forward delta happened to
        // cross a checkpoint boundary in that exact frame) large enough to
        // change the ranking -- an inherent limit of what is externally
        // observable (see the comment above), not a Population/NEAT bug.
        const bool rankingSnapshotIsExact = (finishedBeforeTransitionCall == gen0GenomesSnapshot.size());

        for (std::size_t e = 0; e < transitionPopConfig.eliteCount; ++e)
        {
            const Genome& actualEliteSlot = population.getIndividual(e).getGenome();

            if (rankingSnapshotIsExact)
            {
                // Strong check: elite slot e must be an unchanged copy of
                // specifically the (e+1)-th ranked generation-0 genome.
                const Genome& expectedElite = gen0GenomesSnapshot[ranked[e]];
                assert(connectionsMatch(actualEliteSlot.connections(), expectedElite.connections()) &&
                       "the elite genome must be copied into the next generation completely unchanged"); // 19 & 20
            }
            else
            {
                // Weaker (but still meaningful) check for the acknowledged
                // staleness edge case: elite slot e must still be an
                // unchanged, verbatim copy of *some* generation-0
                // individual's genome -- ruling out corruption or accidental
                // mutation of the elite -- even though which specific
                // individual that was cannot be pinned down from outside.
                const bool matchesSomeGenome =
                    std::any_of(gen0GenomesSnapshot.begin(), gen0GenomesSnapshot.end(), [&](const Genome& candidate)
                                { return connectionsMatch(actualEliteSlot.connections(), candidate.connections()); });
                assert(matchesSomeGenome &&
                       "the elite genome must be copied into the next generation completely unchanged"); // 19 & 20
            }
        }

        // 30 & 31: every next-generation Genome validates and builds a
        // phenotype.
        for (std::size_t i = 0; i < population.size(); ++i)
        {
            population.getIndividual(i).getGenome().validate();       // 30
            ai::neat::buildPhenotype(population.getIndividual(i).getGenome()); // 31
        }

        // 32 & 33: all new individuals reset at spawn with fresh
        // FitnessEvaluators.
        for (std::size_t i = 0; i < population.size(); ++i)
        {
            const Individual& ind = population.getIndividual(i);
            assert(ind.getCar().getPosition().x == kSpawnPosition.x && ind.getCar().getPosition().y == kSpawnPosition.y &&
                   "every new individual must start at spawn"); // 32
            assert(ind.getFitnessEvaluator().getFitness() == 0.0f && ind.getFitnessEvaluator().getElapsedTime() == 0.0f &&
                   !ind.isFinished() && "every new individual's FitnessEvaluator must start fresh"); // 33
        }

        // 37 & 38: best index valid, running+finished == size.
        assert(population.getBestIndividualIndex() < population.size() && "the best individual index must be valid"); // 37
        assert(population.getRunningCount() + population.getFinishedCount() == population.size() &&
               "running + finished must equal population size"); // 38
    }

    // 21 & 22: tournament selection chooses the highest fitness among
    // sampled candidates, with ties resolved by lower index. The pure
    // decision rule is tested directly (zero probabilistic risk, since
    // sampling-with-replacement cannot guarantee any specific index is
    // ever drawn); tournamentSelect() itself is spot-checked afterward.
    {
        assert(isBetterTournamentCandidate(5.0f, 2, 3.0f, 1) == true &&
               "strictly higher fitness must win"); // 21
        assert(isBetterTournamentCandidate(3.0f, 2, 5.0f, 1) == false &&
               "strictly lower fitness must lose"); // 21 (continued)
        assert(isBetterTournamentCandidate(5.0f, 0, 5.0f, 3) == true &&
               "a tie must resolve to the lower index"); // 22
        assert(isBetterTournamentCandidate(5.0f, 3, 5.0f, 0) == false &&
               "a tie must not resolve to the higher index"); // 22 (continued)

        const Genome base = createDemonstrationGenome();
        PopulationConfig popConfig = makeTestPopulationConfig(2, 17u);
        popConfig.tournamentSize = 2;
        const MutationConfig mutationConfig;
        const CrossoverConfig crossoverConfig;
        const CompatibilityConfig compatibilityConfig;
        const SpeciationConfig speciationConfig;
        Population population(base, track, makeCarParams(), kSpawnPosition, kSpawnHeading,
                               popConfig, mutationConfig, crossoverConfig, compatibilityConfig, speciationConfig);

        // Index 1's fitness vastly exceeds index 0's: across many
        // independent tournamentSelect() calls (each advancing the
        // orchestration RNG), index 1 must win at least once whenever it
        // is actually sampled -- and with 10 independent trials of 2
        // draws each, the odds of index 1 never once being sampled are
        // astronomically small (0.25^10), making this a reliable,
        // effectively-deterministic check for this fixed seed.
        const std::vector<float> fitnessValues = {1.0f, 1000.0f};
        bool sawIndex1Win = false;
        for (int trial = 0; trial < 10; ++trial)
        {
            if (population.tournamentSelect(fitnessValues) == 1)
            {
                sawIndex1Win = true;
                break;
            }
        }
        assert(sawIndex1Win &&
               "tournament selection must be able to choose the highest-fitness sampled candidate"); // 21 (continued)
    }

    // 23: parents are never mutated -- a compile-time/structural
    // guarantee, not merely a runtime observation: Individual::getGenome()
    // returns const Genome&, GenomeCrossover::crossover() takes its
    // parents by const Genome&, and Population::reproduce() (see
    // Population.cpp) only ever passes m_individuals[...].getGenome() to
    // those const-reference parameters -- every mutating call
    // (mutateWeights/mutateAddConnection/mutateAddNode) is made only on
    // the freshly-copied `child` local variable, never on a parent
    // reference. It is not possible to mutate a parent genome through the
    // reference Population reads it via.

    // 24, 25, 26, 27 & 28: the crossover + mutation offspring pipeline
    // produces a valid child, each mutation kind can affect it, and the
    // structural mutations share one InnovationTracker.
    {
        const Genome parentA = createDemonstrationGenome();
        Genome parentB = createDemonstrationGenome();
        GenomeMutator setupMutator(321u);
        MutationConfig setupConfig;
        setupMutator.mutateWeights(parentB, setupConfig);

        NodeId maxNodeId = -1;
        for (const NodeGene& node : parentA.nodes())
        {
            maxNodeId = std::max(maxNodeId, node.getId());
        }
        InnovationNumber maxInnovation = -1;
        for (const ConnectionGene& connection : parentA.connections())
        {
            maxInnovation = std::max(maxInnovation, connection.getInnovationNumber());
        }
        InnovationTracker sharedTracker(maxNodeId + 1, maxInnovation + 1);

        GenomeCrossover crossover(444u);
        CrossoverConfig crossoverConfig;
        Genome child = crossover.crossover(parentA, 10.0f, parentB, 5.0f, crossoverConfig);
        child.validate();
        ai::neat::buildPhenotype(child); // 24: crossover must produce a valid, phenotype-buildable child

        GenomeMutator mutator(654u);
        MutationConfig mutationConfig;

        const Genome beforeWeights = child;
        mutationConfig.weightMutationProbability = 1.0f;
        mutator.mutateWeights(child, mutationConfig);
        bool weightChanged = false;
        for (std::size_t i = 0; i < child.connections().size(); ++i)
        {
            if (child.connections()[i].getWeight() != beforeWeights.connections()[i].getWeight())
            {
                weightChanged = true;
            }
        }
        assert(weightChanged && "weight mutation must be able to affect the offspring"); // 25

        const std::size_t connCountBeforeAddConnection = child.connections().size();
        mutationConfig.addConnectionProbability = 1.0f;
        const bool addedConnection = mutator.mutateAddConnection(child, sharedTracker, mutationConfig);
        assert(addedConnection && child.connections().size() == connCountBeforeAddConnection + 1 &&
               "add-connection mutation must be able to affect the offspring"); // 26

        const std::size_t nodeCountBeforeAddNode = child.nodes().size();
        const std::size_t connCountBeforeAddNode = child.connections().size();
        mutationConfig.addNodeProbability = 1.0f;
        const bool addedNode = mutator.mutateAddNode(child, sharedTracker, mutationConfig);
        assert(addedNode && child.nodes().size() == nodeCountBeforeAddNode + 1 &&
               child.connections().size() == connCountBeforeAddNode + 2 &&
               "add-node mutation must be able to affect the offspring"); // 27

        assert(sharedTracker.getNextAvailableNodeId() > maxNodeId + 1 &&
               "add-connection and add-node mutation must share the same InnovationTracker"); // 28

        child.validate();
        ai::neat::buildPhenotype(child);
    }

    // 29: the next generation is constructed before the old population is
    // replaced -- a structural guarantee, confirmed indirectly by 19/20
    // above: Population::reproduce() (see Population.cpp) builds
    // newGenomes/newIndividuals entirely from m_individuals as it existed
    // when reproduce() began, and only assigns
    // `m_individuals = std::move(newIndividuals)` as its final step. If
    // parent selection instead read a partially-overwritten population,
    // the elite-exactness check above could not reliably pass.

    // 34, 35 & 36: restartGeneration() resets evaluation state without
    // changing the generation number or Genome contents.
    {
        const Genome base = createDemonstrationGenome();
        const PopulationConfig popConfig = makeTestPopulationConfig(4, 5u);
        const MutationConfig mutationConfig;
        const CrossoverConfig crossoverConfig;
        const CompatibilityConfig compatibilityConfig;
        const SpeciationConfig speciationConfig;
        Population population(base, track, makeCarParams(), kSpawnPosition, kSpawnHeading,
                               popConfig, mutationConfig, crossoverConfig, compatibilityConfig, speciationConfig);

        for (int step = 0; step < 30; ++step)
        {
            population.update(kSimulationDt);
        }

        const std::size_t generationBefore = population.getGeneration();
        std::vector<Genome> genomesBefore;
        for (std::size_t i = 0; i < population.size(); ++i)
        {
            genomesBefore.push_back(population.getIndividual(i).getGenome());
        }

        population.restartGeneration();

        assert(population.getGeneration() == generationBefore &&
               "restartGeneration() must not change the generation number"); // 34
        for (std::size_t i = 0; i < population.size(); ++i)
        {
            const Genome& g = population.getIndividual(i).getGenome();
            assert(connectionsMatch(g.connections(), genomesBefore[i].connections()) &&
                   "restartGeneration() must not alter Genome contents"); // 35
            const Individual& ind = population.getIndividual(i);
            assert(ind.getFitnessEvaluator().getFitness() == 0.0f && !ind.isFinished() &&
                   ind.getCar().getPosition().x == kSpawnPosition.x && ind.getCar().getPosition().y == kSpawnPosition.y &&
                   "restartGeneration() must reset evaluation state"); // 36
        }
    }

    // 40: species count is computed from the current generation's actual
    // Genomes -- cross-checked against an independent Speciator run over
    // the same Genomes.
    {
        const Genome base = createDemonstrationGenome();
        const PopulationConfig popConfig = makeTestPopulationConfig(5, 8u);
        const MutationConfig mutationConfig;
        const CrossoverConfig crossoverConfig;
        const CompatibilityConfig compatibilityConfig;
        const SpeciationConfig speciationConfig;
        Population population(base, track, makeCarParams(), kSpawnPosition, kSpawnHeading,
                               popConfig, mutationConfig, crossoverConfig, compatibilityConfig, speciationConfig);

        std::vector<Genome> genomes;
        for (std::size_t i = 0; i < population.size(); ++i)
        {
            genomes.push_back(population.getIndividual(i).getGenome());
        }
        Speciator independentSpeciator;
        const std::size_t independentCount =
            independentSpeciator.speciate(genomes, compatibilityConfig, speciationConfig).size();
        assert(population.getSpeciesCount() == independentCount &&
               "species count must be computed from the current generation's actual Genomes"); // 40
    }

    // 41: as of Stage 16, species membership DOES drive reproduction --
    // Population::reproduce() (see Population.cpp) speciates the just-
    // finished generation once and uses that same Species vector for
    // fitness sharing, offspring allocation, and species-local parent
    // selection. See verifySpeciesAwareReproduction() below for the
    // dedicated Stage 16 verification suite; this function (Stage 13/15's
    // suite) is otherwise left intact and continues to pass unchanged.

    // 42 & 43: fixed seed + configs + same evaluation fitnesses produce
    // deterministic offspring; different seeds can produce different
    // offspring.
    {
        const Genome crashGenome = makeCrashGenome();
        const PopulationConfig popConfig = makeTestPopulationConfig(4, 2024u);
        const MutationConfig mutationConfig;
        const CrossoverConfig crossoverConfig;
        const CompatibilityConfig compatibilityConfig;
        const SpeciationConfig speciationConfig;

        Population populationX(crashGenome, track, makeCarParams(), kSpawnPosition, kSpawnHeading,
                                popConfig, mutationConfig, crossoverConfig, compatibilityConfig, speciationConfig);
        Population populationY(crashGenome, track, makeCarParams(), kSpawnPosition, kSpawnHeading,
                                popConfig, mutationConfig, crossoverConfig, compatibilityConfig, speciationConfig);

        for (int step = 0; step < 4000 && populationX.getGeneration() == 0; ++step)
        {
            populationX.update(kSimulationDt);
        }
        for (int step = 0; step < 4000 && populationY.getGeneration() == 0; ++step)
        {
            populationY.update(kSimulationDt);
        }
        assert(populationX.getGeneration() == 1 && populationY.getGeneration() == 1 &&
               "setup: both populations must reach generation 1 within the step budget");

        for (std::size_t i = 0; i < populationX.size(); ++i)
        {
            assert(connectionsMatch(populationX.getIndividual(i).getGenome().connections(),
                                     populationY.getIndividual(i).getGenome().connections()) &&
                   "identical seed/configs/base genome must produce deterministic offspring"); // 42
        }

        PopulationConfig popConfigDifferentSeed = popConfig;
        popConfigDifferentSeed.randomSeed = 999u;
        Population populationZ(crashGenome, track, makeCarParams(), kSpawnPosition, kSpawnHeading,
                                popConfigDifferentSeed, mutationConfig, crossoverConfig, compatibilityConfig,
                                speciationConfig);
        for (int step = 0; step < 4000 && populationZ.getGeneration() == 0; ++step)
        {
            populationZ.update(kSimulationDt);
        }
        assert(populationZ.getGeneration() == 1 && "setup: populationZ must reach generation 1 within the step budget");

        bool sawDifference = false;
        for (std::size_t i = 0; i < populationZ.size() && !sawDifference; ++i)
        {
            if (!connectionsMatch(populationX.getIndividual(i).getGenome().connections(),
                                   populationZ.getIndividual(i).getGenome().connections()))
            {
                sawDifference = true;
            }
        }
        assert(sawDifference && "different seeds must be capable of producing different offspring"); // 43
    }

    // 44: no random_device/global RNG/time seeding exists anywhere in
    // Population.cpp or Individual.cpp -- every RNG (the orchestration
    // std::mt19937, GenomeMutator, GenomeCrossover) is constructed with a
    // value derived purely from PopulationConfig::randomSeed.

    // 47: population rendering information can be queried without
    // mutating evolution state.
    {
        const Genome base = createDemonstrationGenome();
        const PopulationConfig popConfig = makeTestPopulationConfig(4, 9u);
        const MutationConfig mutationConfig;
        const CrossoverConfig crossoverConfig;
        const CompatibilityConfig compatibilityConfig;
        const SpeciationConfig speciationConfig;
        Population population(base, track, makeCarParams(), kSpawnPosition, kSpawnHeading,
                               popConfig, mutationConfig, crossoverConfig, compatibilityConfig, speciationConfig);

        const std::size_t generationBefore = population.getGeneration();
        const Genome genomeSnapshot = population.getIndividual(0).getGenome();

        for (int i = 0; i < 5; ++i)
        {
            (void)population.getGeneration();
            (void)population.getBestIndividualIndex();
            (void)population.getRunningCount();
            (void)population.getFinishedCount();
            (void)population.getSpeciesCount();
            (void)population.getLastGenerationBestFitness();
        }

        assert(population.getGeneration() == generationBefore &&
               "querying read-only accessors must not change the generation"); // 47 (part 1)
        assert(connectionsMatch(population.getIndividual(0).getGenome().connections(), genomeSnapshot.connections()) &&
               "querying read-only accessors must not mutate any individual's genome"); // 47 (part 2)
    }

    // 48: highlighted individual selection is deterministic.
    {
        const Genome base = createDemonstrationGenome();
        const PopulationConfig popConfig = makeTestPopulationConfig(5, 6u);
        const MutationConfig mutationConfig;
        const CrossoverConfig crossoverConfig;
        const CompatibilityConfig compatibilityConfig;
        const SpeciationConfig speciationConfig;
        Population population(base, track, makeCarParams(), kSpawnPosition, kSpawnHeading,
                               popConfig, mutationConfig, crossoverConfig, compatibilityConfig, speciationConfig);

        for (int step = 0; step < 100; ++step)
        {
            population.update(kSimulationDt);
        }

        const std::size_t highlightedFirst = selectHighlightedIndividual(population);
        const std::size_t highlightedSecond = selectHighlightedIndividual(population);
        assert(highlightedFirst == highlightedSecond && highlightedFirst < population.size() &&
               "highlighted individual selection must be deterministic for the same population state"); // 48
    }

    // 49: all previous verification suites still pass -- enforced by
    // main() continuing to call every earlier verify*() function
    // unchanged.

    // 50: an automatic generation transition without crashing is directly
    // exercised (and asserted) by the combined test above (items 15-20,
    // 39) and by 42/43 -- both drive Population through at least one full
    // generation transition purely via update() calls, with no exception
    // or assertion failure along the way. The real executable's own such
    // transition is additionally the subject of the separate manual
    // acceptance test.

    TraceLog(LOG_INFO, "Population verification: all deterministic checks passed");
}

// Stage 16 dedicated verification suite: species-aware reproduction.
// Exercises fitness sharing (adjusted fitness), per-species offspring
// allocation (both the general proportional case and the zero-total-
// fitness fallback), species-local parent selection, and the read-only
// getCurrentSpecies()/getReproductionStats() debug surface. Deliberately
// does not re-test what Stage 16 left untouched (global elitism's own
// ranking rule, the crossover/mutation pipeline, shared InnovationTracker
// usage, isBetterTournamentCandidate()'s tie-break) beyond a light
// confirmation that those code paths still behave the same way when driven
// through species-scoped reproduction -- verifyPopulation() above already
// covers them thoroughly.
void verifySpeciesAwareReproduction(const simulation::Track& track)
{
    using namespace population_verify;
    using ai::neat::allocateSpeciesOffspring;
    using ai::neat::effectiveFitnessContribution;
    using ai::neat::Species;
    using ai::neat::SpeciesId;

    // 7, 8, 9, 10 & 11: allocateSpeciesOffspring(), the pure offspring-
    // allocation algorithm, tested directly against synthetic species-
    // fitness data -- exact remaining-slot count, floor + largest-
    // fractional-remainder, fractional-remainder ties broken by lower
    // SpeciesId, the zero-total-fitness fallback, and its own tie-break.
    {
        // Exact proportional split (no remainder).
        const std::vector<SpeciesId> ids2 = {0, 1};
        const std::vector<float> sums2 = {10.0f, 30.0f};
        const std::vector<std::size_t> alloc2 = allocateSpeciesOffspring(ids2, sums2, 4);
        assert(alloc2.size() == 2 && alloc2[0] + alloc2[1] == 4 &&
               "allocation must always sum to exactly remainingSlots"); // 7
        assert(alloc2[0] == 1 && alloc2[1] == 3 &&
               "an exact proportional split must match the fitness ratio precisely"); // 7 (continued)
    }
    {
        // Floor + largest-fractional-remainder, with a three-way tie
        // broken by lower SpeciesId rather than input position.
        const std::vector<SpeciesId> ids3 = {5, 2, 8};
        const std::vector<float> sums3 = {10.0f, 10.0f, 10.0f};
        const std::vector<std::size_t> alloc3 = allocateSpeciesOffspring(ids3, sums3, 10);
        const std::size_t total3 = alloc3[0] + alloc3[1] + alloc3[2];
        assert(total3 == 10 && "floor + remainder allocation must still sum to exactly remainingSlots"); // 8
        assert(alloc3[1] == 4 && alloc3[0] == 3 && alloc3[2] == 3 &&
               "an equal fractional remainder must be broken by lower SpeciesId (id 2), not input position"); // 9
    }
    {
        // A species with zero effective fitness gets zero offspring even
        // while the total across all species is positive.
        const std::vector<SpeciesId> idsZero = {0, 1};
        const std::vector<float> sumsZero = {0.0f, 5.0f};
        const std::vector<std::size_t> allocZero = allocateSpeciesOffspring(idsZero, sumsZero, 6);
        assert(allocZero[0] == 0 && allocZero[1] == 6 &&
               "a species contributing zero effective fitness must receive zero offspring"); // 6 (allocation side)
    }
    {
        // Zero-total-fitness fallback: as-even-as-possible split, extra
        // remainder slots to the lowest SpeciesId first.
        const std::vector<SpeciesId> idsFallback = {3, 1, 2};
        const std::vector<float> sumsFallback = {0.0f, 0.0f, 0.0f};
        const std::vector<std::size_t> allocFallback = allocateSpeciesOffspring(idsFallback, sumsFallback, 7);
        const std::size_t totalFallback = allocFallback[0] + allocFallback[1] + allocFallback[2];
        assert(totalFallback == 7 && "the zero-total fallback must still sum to exactly remainingSlots"); // 10
        assert(allocFallback[1] == 3 && allocFallback[2] == 2 && allocFallback[0] == 2 &&
               "the zero-total fallback must give the extra remainder slot to the lowest SpeciesId first"); // 11
    }
    {
        // A single species receives every remaining slot; an empty species
        // list allocates nothing.
        const std::vector<SpeciesId> idsSingle = {42};
        const std::vector<float> sumsSingle = {5.0f};
        assert((allocateSpeciesOffspring(idsSingle, sumsSingle, 9) == std::vector<std::size_t>{9}) &&
               "a single species must receive every remaining slot");
        assert(allocateSpeciesOffspring({}, {}, 0).empty() && "no species means no allocation");
    }

    // 6: negative raw (and therefore negative adjusted) fitness contributes
    // exactly zero to a species' effective fitness -- the exact clamp
    // Population::reproduce() itself applies before calling
    // allocateSpeciesOffspring().
    {
        assert(effectiveFitnessContribution(-5.0f) == 0.0f &&
               "negative adjusted fitness must contribute exactly zero to a species' effective fitness"); // 6
        assert(effectiveFitnessContribution(0.0f) == 0.0f && "zero adjusted fitness must remain zero");
        assert(effectiveFitnessContribution(3.5f) == 3.5f &&
               "non-negative adjusted fitness must be passed through unchanged"); // 6 (continued)
    }

    // 1-5, 12-15, 17, 18, 32-36, 38-40 & 45: a real Population driven
    // through one full generation transition, cross-checked against an
    // independent recomputation of the documented formulas from an exact
    // pre-transition snapshot -- captured with the same re-snapshot-every-
    // iteration technique verifyPopulation's own items 15-20 use above,
    // since Population::update() can only ever be observed either before
    // every individual has finished or after reproduce() has already run
    // in that same call; there is no external way to observe "every
    // individual finished, but reproduce() has not run yet".
    {
        const Genome crashGenome = makeCrashGenome();
        const PopulationConfig popConfig = makeTestPopulationConfig(12, 777u);
        const MutationConfig mutationConfig;
        const CrossoverConfig crossoverConfig;
        const CompatibilityConfig compatibilityConfig;
        // A very tight threshold: only genomes at exactly 0 compatibility
        // distance from each other share a species. Since the crash genome
        // has a single connection and mutateWeights() only selects it for
        // mutation with 80% probability per individual (MutationConfig's
        // default weightMutationProbability), this deterministically (for
        // this fixed seed) tends to produce a mix of species -- exercising
        // both multi-member and singleton species in one run. None of the
        // checks below hardcode the resulting partition; they hold for
        // whatever partition this seed deterministically produces.
        SpeciationConfig speciationConfig;
        speciationConfig.compatibilityThreshold = 0.0f;

        Population population(crashGenome, track, makeCarParams(), kSpawnPosition, kSpawnHeading,
                               popConfig, mutationConfig, crossoverConfig, compatibilityConfig, speciationConfig);

        std::vector<Genome> genomeSnapshot;
        std::vector<float> fitnessSnapshot;
        std::size_t finishedBeforeTransitionCall = 0;
        for (int step = 0; step < 4000 && population.getGeneration() == 0; ++step)
        {
            genomeSnapshot.clear();
            fitnessSnapshot.clear();
            for (std::size_t i = 0; i < population.size(); ++i)
            {
                genomeSnapshot.push_back(population.getIndividual(i).getGenome());
                fitnessSnapshot.push_back(population.getIndividual(i).getFitness());
            }
            finishedBeforeTransitionCall = population.getFinishedCount();
            population.update(kSimulationDt);
        }
        assert(population.getGeneration() == 1 &&
               "setup: the crash genome must reach a generation transition within the step budget"); // 38 & 45

        const bool snapshotIsExact = (finishedBeforeTransitionCall == genomeSnapshot.size());

        const std::vector<Species>& species = population.getCurrentSpecies();
        const std::size_t speciesCount = species.size();
        const std::vector<Population::SpeciesReproductionStats>& stats = population.getReproductionStats();

        // 32 & 33: reproduction stats correspond 1:1 with getCurrentSpecies(),
        // in the exact same (ascending SpeciesId) order.
        assert(stats.size() == speciesCount &&
               "reproduction stats must cover exactly the species vector reproduce() itself used"); // 32
        SpeciesId previousId = -1;
        std::size_t totalMembersAcrossSpecies = 0;
        for (std::size_t s = 0; s < speciesCount; ++s)
        {
            assert(stats[s].speciesId == species[s].getId() && stats[s].memberCount == species[s].size() &&
                   "reproduction stats must describe the exact same Species objects, in the exact same order"); // 32
            assert(species[s].getId() > previousId &&
                   "species must be processed/reported in strictly ascending SpeciesId order"); // 33
            previousId = species[s].getId();
            totalMembersAcrossSpecies += species[s].size();
        }
        assert(totalMembersAcrossSpecies == population.size() &&
               "every individual from the just-finished generation must belong to exactly one species");

        // 17, 18 & 34: offspring allocation sums to exactly the non-elite
        // remaining slots, and elites + allocated offspring reconstruct the
        // configured population size.
        std::size_t totalAllocatedOffspring = 0;
        for (const Population::SpeciesReproductionStats& s : stats)
        {
            totalAllocatedOffspring += s.allocatedOffspring;
        }
        assert(totalAllocatedOffspring == popConfig.populationSize - popConfig.eliteCount &&
               "allocated offspring across every species must sum to exactly populationSize - eliteCount"); // 17 & 18
        assert(popConfig.eliteCount + totalAllocatedOffspring == popConfig.populationSize &&
               "elites + allocated offspring must reconstruct the full configured population size"); // 34

        // 35 & 36: the next generation is exactly populationSize, and every
        // one of its Genomes validates and builds a phenotype.
        assert(population.size() == popConfig.populationSize &&
               "next generation size must be exactly populationSize"); // 35
        for (std::size_t i = 0; i < population.size(); ++i)
        {
            population.getIndividual(i).getGenome().validate();
            ai::neat::buildPhenotype(population.getIndividual(i).getGenome()); // 36
        }

        // 39: every new individual reset cleanly (fresh spawn, zero
        // fitness, not finished).
        for (std::size_t i = 0; i < population.size(); ++i)
        {
            const Individual& ind = population.getIndividual(i);
            assert(!ind.isFinished() && ind.getFitnessEvaluator().getFitness() == 0.0f &&
                   ind.getCar().getPosition().x == kSpawnPosition.x && ind.getCar().getPosition().y == kSpawnPosition.y &&
                   "every new individual must reset to a fresh spawn state"); // 39
        }

        if (snapshotIsExact)
        {
            // 40: the Species partition is a pure function of the exact
            // snapshotted genomes (in index order) -- an independently
            // constructed Speciator must produce the identical sequence of
            // member-index sets (SpeciesId numbering may legitimately
            // differ, since this Population's own Speciator had already
            // allocated IDs for generation 0 before this call).
            Speciator independentSpeciator;
            const std::vector<Species> independentSpecies =
                independentSpeciator.speciate(genomeSnapshot, compatibilityConfig, speciationConfig);
            assert(independentSpecies.size() == speciesCount &&
                   "species count must be a pure function of the snapshotted genomes"); // 40
            for (std::size_t s = 0; s < speciesCount; ++s)
            {
                assert(independentSpecies[s].getMemberIndices() == species[s].getMemberIndices() &&
                       "species membership must be a pure function of the snapshotted genomes"); // 40 (continued)
            }

            // 1, 2, 3, 4 & 5: fitness sharing -- adjustedFitness ==
            // rawFitness/speciesSize for every member; raw fitness is
            // untouched (fitnessSnapshot, read straight from
            // Individual::getFitness(), is never mutated by this
            // recomputation); each species' reported adjustedFitnessSum
            // matches an independent recomputation from the exact snapshot.
            for (std::size_t s = 0; s < speciesCount; ++s)
            {
                const std::vector<std::size_t>& members = species[s].getMemberIndices();
                float expectedAdjustedSum = 0.0f;
                for (std::size_t memberIndex : members)
                {
                    const float raw = fitnessSnapshot[memberIndex];
                    const float adjusted = raw / static_cast<float>(members.size());
                    expectedAdjustedSum += adjusted;
                    if (members.size() == 1)
                    {
                        assert(adjusted == raw &&
                               "a singleton species' one member's adjusted fitness must equal its raw fitness"); // 3
                    }
                    if (members.size() > 1 && raw > 0.0f)
                    {
                        assert(adjusted < raw &&
                               "sharing fitness across more than one member must strictly reduce a positive raw "
                               "fitness"); // 4
                    }
                }
                assert(std::fabs(stats[s].adjustedFitnessSum - expectedAdjustedSum) < 1e-3f &&
                       "each species' reported adjustedFitnessSum must equal the sum of rawFitness/size over its "
                       "own members"); // 1, 2 & 5
            }

            // 12, 13, 14 & 15: global elitism is untouched by species-aware
            // reproduction -- elite slot 0 must be an unchanged copy of
            // whichever snapshot individual had the strictly highest raw
            // fitness (ties broken by lowest index).
            std::size_t topIndex = 0;
            for (std::size_t i = 1; i < fitnessSnapshot.size(); ++i)
            {
                if (fitnessSnapshot[i] > fitnessSnapshot[topIndex])
                {
                    topIndex = i;
                }
            }
            assert(connectionsMatch(population.getIndividual(0).getGenome().connections(),
                                     genomeSnapshot[topIndex].connections()) &&
                   "elite slot 0 must be an unchanged, byte-for-byte copy of the top raw-fitness genome"); // 12-15
        }

        TraceLog(LOG_INFO, "Species-aware reproduction: this run produced %d species (exact snapshot: %s)",
                 static_cast<int>(speciesCount), snapshotIsExact ? "yes" : "no");
    }

    // 19, 20, 21, 22, 23, 24, 25 & 42 (structural guarantees -- see
    // Population::reproduce()/selectParentFromSpecies() in Population.cpp):
    //   - selectParentFromSpecies() only ever samples from
    //     species.getMemberIndices(), and reproduce() only ever calls it
    //     with the one Species currently being processed -- both parents of
    //     every offspring are therefore guaranteed to come from that exact
    //     species, and never from any other species (19 & 20). This is the
    //     one behavioral difference from the pre-Stage-16 code path, which
    //     called the (still-public, still directly exercised by
    //     verifyPopulation's items 21/22 above) tournamentSelect() over the
    //     *entire* population's fitnessValues regardless of species --
    //     species-aware reproduction's candidate pool for any species
    //     smaller than the whole population is therefore provably narrower
    //     than, and different from, that global candidate pool (42).
    //   - selectParentFromSpecies() reduces its samples via the exact same
    //     isBetterTournamentCandidate() free function verifyPopulation's
    //     items 21/22 already exercise directly -- highest raw fitness
    //     wins, ties broken by lower original population index (21 & 22).
    //   - a species of size 1 gives std::uniform_int_distribution<...>(0, 0),
    //     which always resolves to its one member -- no special-casing
    //     exists or is needed for singleton species (23); the resulting
    //     crossover(parent, parent) call (equal fitness, identical genomes)
    //     is accepted unconditionally by GenomeCrossover, never throwing,
    //     and is exercised by the live run above whenever this seed
    //     produces a singleton species.
    //   - reproduce() passes fitnessValues[parentAIndex]/[parentBIndex] --
    //     RAW fitness -- into GenomeCrossover::crossover(), and never
    //     constructs or passes any adjusted-fitness value to it (24 & 25).

    // 26, 27, 28, 29, 30 & 31: the crossover + mutation offspring pipeline
    // (mutateWeights/mutateAddConnection/mutateAddNode sharing one
    // InnovationTracker, parents never mutated) is byte-for-byte the same
    // code Population::reproduce() already ran before Stage 16 -- only
    // which two parent indices feed it changed (species-scoped instead of
    // population-wide). Already exercised end to end, deterministically, by
    // verifyPopulation's items 24-28 above and by the live run above (every
    // offspring genome validates and builds a phenotype, per the 35/36
    // checks).

    // 37, 43 & 44: fixed seed + identical configs/base genome produce
    // identical species partitions, adjusted-fitness sums, and offspring
    // allocation -- the entire Stage 16 reproduction pipeline is
    // deterministic -- and its read-only species/stats accessors never
    // mutate state when queried repeatedly.
    {
        const Genome crashGenome = makeCrashGenome();
        const PopulationConfig popConfig = makeTestPopulationConfig(10, 555u);
        const MutationConfig mutationConfig;
        const CrossoverConfig crossoverConfig;
        const CompatibilityConfig compatibilityConfig;
        SpeciationConfig speciationConfig;
        speciationConfig.compatibilityThreshold = 0.0f;

        auto runToNextGeneration = [](Population& population)
        {
            for (int step = 0; step < 4000 && population.getGeneration() == 0; ++step)
            {
                population.update(kSimulationDt);
            }
        };

        Population populationX(crashGenome, track, makeCarParams(), kSpawnPosition, kSpawnHeading,
                                popConfig, mutationConfig, crossoverConfig, compatibilityConfig, speciationConfig);
        Population populationY(crashGenome, track, makeCarParams(), kSpawnPosition, kSpawnHeading,
                                popConfig, mutationConfig, crossoverConfig, compatibilityConfig, speciationConfig);
        runToNextGeneration(populationX);
        runToNextGeneration(populationY);
        assert(populationX.getGeneration() == 1 && populationY.getGeneration() == 1 &&
               "setup: both populations must reach generation 1 within the step budget");

        // 43: read-only accessors queried repeatedly must not mutate state.
        const std::size_t generationBeforeQueries = populationX.getGeneration();
        for (int i = 0; i < 5; ++i)
        {
            (void)populationX.getCurrentSpecies();
            (void)populationX.getReproductionStats();
            (void)populationX.getSpeciesCount();
        }
        assert(populationX.getGeneration() == generationBeforeQueries &&
               "querying species/reproduction-stats accessors repeatedly must not mutate Population state"); // 43

        const std::vector<Species>& speciesX = populationX.getCurrentSpecies();
        const std::vector<Species>& speciesY = populationY.getCurrentSpecies();
        const std::vector<Population::SpeciesReproductionStats>& statsX = populationX.getReproductionStats();
        const std::vector<Population::SpeciesReproductionStats>& statsY = populationY.getReproductionStats();

        assert(speciesX.size() == speciesY.size() && statsX.size() == statsY.size() &&
               "identical seed/configs/base genome must produce an identical species count"); // 37 & 44
        for (std::size_t s = 0; s < speciesX.size(); ++s)
        {
            assert(speciesX[s].getMemberIndices() == speciesY[s].getMemberIndices() &&
                   "identical seed/configs/base genome must produce identical species membership"); // 37
            assert(statsX[s].speciesId == statsY[s].speciesId && statsX[s].memberCount == statsY[s].memberCount &&
                   statsX[s].allocatedOffspring == statsY[s].allocatedOffspring &&
                   std::fabs(statsX[s].adjustedFitnessSum - statsY[s].adjustedFitnessSum) < 1e-6f &&
                   "identical seed/configs/base genome must produce identical reproduction stats"); // 44
        }
        for (std::size_t i = 0; i < populationX.size(); ++i)
        {
            assert(connectionsMatch(populationX.getIndividual(i).getGenome().connections(),
                                     populationY.getIndividual(i).getGenome().connections()) &&
                   "identical seed/configs/base genome must produce deterministic offspring genomes"); // 44 (continued)
        }
    }

    // 45 & 46: an automatic species-aware generation transition without
    // crashing is directly exercised (and asserted) by every live run
    // above, and by verifyPopulation() continuing to run and pass
    // immediately before this function (see main()).

    TraceLog(LOG_INFO, "Species-aware reproduction verification: all deterministic checks passed");
}

} // namespace

int main()
{
    InitWindow(kScreenWidth, kScreenHeight, "NEAT Car Simulation");

    SetTargetFPS(60);

    simulation::Track track(makeTrackDefinition());
    verifyTrack(track);
    verifyCar(track);
    verifySensors(track);
    verifyObservation(track);
    verifyNeuralNetwork();
    verifyNeatGenes();
    verifyGenome();
    verifyPhenotypeBuilder();
    verifyGenomeMutator();
    verifyInnovationTracker();
    verifyAddConnectionMutation();
    verifyAddNodeMutation();
    verifyGenomeCrossover();
    verifyCompatibilityDistance();
    verifySpeciation();
    verifyAIController(track);
    verifyTrackProgress(track);
    verifyFitnessEvaluator(track);
    verifyPopulation(track);
    verifySpeciesAwareReproduction(track);

    // The whole training run starts from one hand-built, deterministic
    // demonstration Genome (see createDemonstrationGenome()) -- Population
    // copies and mutates it to build generation 0; the Genome itself is
    // never touched again afterward. The spawn pose (kSpawnPosition/
    // kSpawnHeading) comes entirely from the Track itself -- see
    // computeSpawnPose() above.
    const ai::neat::PopulationConfig populationConfig;
    const ai::neat::MutationConfig mutationConfig;
    const ai::neat::CrossoverConfig crossoverConfig;
    const ai::neat::CompatibilityConfig compatibilityConfig;
    const ai::neat::SpeciationConfig speciationConfig;

    ai::neat::Population population(createDemonstrationGenome(), track, makeCarParams(), kSpawnPosition,
                                     kSpawnHeading, populationConfig, mutationConfig, crossoverConfig,
                                     compatibilityConfig, speciationConfig);

    while (!WindowShouldClose())
    {
        if (IsKeyPressed(KEY_R))
        {
            // Restarts the CURRENT generation's evaluation from its
            // existing Genomes -- same generation number, same genomes,
            // only Car/Progress/Fitness state resets. Population::update()
            // below never needs a separate "is finished" guard: a
            // population update is always safe to call, whether
            // individuals are still running, all finished (in which case
            // update() itself triggers the generation transition), or
            // freshly restarted.
            population.restartGeneration();
        }

        population.update(kSimulationDt);

        const std::size_t highlightedIndex = selectHighlightedIndividual(population);

        BeginDrawing();
        ClearBackground(BLACK);

        DrawRectangle(0, 0, kSimWidth, kSimHeight, BLACK);

        // Stage 14A: render the exact same sampled centerline the CPU mask
        // and progress/checkpoint queries are built from -- thick gray line
        // segments (with a rounding circle at each joint so short segments
        // still join smoothly) whose width matches
        // TrackDefinition::trackWidth, on the black background already
        // cleared above. No ellipse-specific rendering remains.
        {
            const std::vector<Vector2>& centerlineSamples = track.getCenterlineSamples();
            const float roadWidth = track.getDefinition().trackWidth;
            const std::size_t sampleCount = centerlineSamples.size();
            for (std::size_t i = 0; i < sampleCount; ++i)
            {
                const Vector2& a = centerlineSamples[i];
                const Vector2& b = centerlineSamples[(i + 1) % sampleCount];
                DrawLineEx(a, b, roadWidth, GRAY);
                DrawCircleV(a, roadWidth * 0.5f, GRAY);
            }
        }

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

        drawPopulationPanel(population, highlightedIndex);

        EndDrawing();
    }

    CloseWindow();

    return 0;
}
