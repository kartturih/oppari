#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "raylib.h"

namespace simulation
{

// A track as three independent layers:
//   1. VISUAL -- visualImagePath, background image (read only by TrackVisual).
//   2. MASK -- maskImagePath, binary drivable-area image; sole source of the
//      CPU mask if set. If empty, falls back to deriving the mask from
//      controlPoints + trackWidth (procedural tracks only).
//   3. CENTERLINE -- controlPoints + samplesPerSegment, interpolated into a
//      closed sampled centerline for progress/checkpoints/spawn/direction --
//      never used for collision when maskImagePath is set.
// Nothing derives one layer from another.
struct TrackDefinition
{
    int simWidth = 0;
    int simHeight = 0;

    std::vector<Vector2> controlPoints;

    // Nominal road width: the mask's basis when maskImagePath is empty,
    // otherwise only a verification heuristic.
    float trackWidth = 0.0f;

    int samplesPerSegment = 0;

    // Both empty means no image assets: render the raw centerline, derive
    // the mask from controlPoints + trackWidth.
    std::string visualImagePath;
    std::string maskImagePath;

    // Arc-length distance (wrapped into [0, total length)) of the spawn
    // pose along the centerline. Default 0 = centerline sample 0.
    float spawnDistanceAlongTrack = 0.0f;
};

// Result of projecting a world position onto the closest centerline point.
struct TrackProjection
{
    Vector2 point;
    std::size_t segmentIndex = 0;
    float segmentT = 0.0f;               // clamped parametric position within the segment, [0,1]
    float distanceAlongTrack = 0.0f;     // cumulative arc length to `point`, [0, getTotalLength())
    float distanceFromCenterline = 0.0f; // perpendicular distance from the query position
};

// A closed, centerline-based track: definition, dense sampled centerline +
// arc-length table, and a precomputed CPU drivable mask. Collision/sensor
// queries only ever read the mask.
class Track
{
public:
    // Throws std::invalid_argument on an invalid definition (bad
    // dimensions, too few control points, non-finite coordinates, invalid
    // width/samplesPerSegment), a missing/mismatched visual or mask image,
    // or a non-drivable computed spawn position. CPU-only image loading --
    // no raylib window required (see TrackVisual for the GPU layer).
    explicit Track(const TrackDefinition& definition);

    // False for out-of-bounds coordinates, without touching the mask.
    bool isDrivable(int x, int y) const;

    const TrackDefinition& getDefinition() const { return m_definition; }

    // Closed sampled centerline, forward-travel order. Does not repeat
    // sample 0 at the end -- the closing segment is index size() - 1.
    const std::vector<Vector2>& getCenterlineSamples() const { return m_centerline; }

    // Cumulative arc length per sample; [0] is 0, monotonically increasing.
    const std::vector<float>& getCumulativeDistances() const { return m_cumulativeDistances; }

    float getTotalLength() const { return m_totalLength; }

    // Closest point on the centerline to position, scanning every segment
    // (ties broken by lower segment index). Not used per-frame/per-sensor
    // (those stay O(1) via the mask) -- see projectOntoCenterlineLocal().
    TrackProjection projectOntoCenterline(Vector2 position) const;

    // Local variant: scans only a window of 2*searchRadius+1 segments
    // around previousSegmentIndex (wrapping correctly at the seam), for
    // per-frame tracking (see TrackProgress) where a full scan could jump
    // between two geometrically close but unrelated stretches of centerline.
    // previousSegmentIndex is taken modulo the sample count.
    //
    // continuityWeight/continuityFreeZone bias window candidates toward
    // continuing near previousSegmentIndex: score = distance +
    // continuityWeight * sqrt(max(0, indexGap - continuityFreeZone)). The
    // free zone keeps ordinary forward movement from being penalized versus
    // standing still. Both default to 0 (pure nearest-distance). The
    // returned distanceFromCenterline is always the true geometric
    // distance of the winner, never the penalized score.
    TrackProjection projectOntoCenterlineLocal(Vector2 position, std::size_t previousSegmentIndex,
                                                std::size_t searchRadius, float continuityWeight = 0.0f,
                                                std::size_t continuityFreeZone = 0) const;

    // Point on the closed loop at arc-length distanceAlongTrack from sample
    // 0 (wrapped into [0, getTotalLength())). For spawn placement/testing.
    Vector2 getPointAtDistance(float distanceAlongTrack) const;

    // Deterministic spawn pose at spawnDistanceAlongTrack along the
    // centerline; heading is that point's forward tangent. Constructor
    // throws if this position isn't drivable.
    Vector2 getSpawnPosition() const { return m_spawnPosition; }
    float getSpawnHeading() const { return m_spawnHeading; }

private:
    void buildCenterline();
    void buildArcLength();

    // Dispatches to the image-based or centerline-width mask builder.
    void buildMask();
    void buildMaskFromCenterlineWidth();
    void buildMaskFromImage();

    // CPU-only; throws if visualImagePath is set but invalid/mismatched.
    // Stores nothing -- only validates the asset TrackVisual will load later.
    void validateVisualAsset() const;

    void computeSpawn();

    // Unbiased nearest-distance fold for the global scan (used by
    // projectOntoCenterline() and recovery, which need the true nearest
    // match, not a continuity-preferring one).
    void considerSegment(Vector2 position, std::size_t segmentIndex, TrackProjection& best, float& bestDistSq) const;

    // Continuity-weighted fold for the local scan; best.distanceFromCenterline
    // is always set to the true geometric distance, never the weighted score.
    void considerSegmentLocal(Vector2 position, std::size_t segmentIndex, std::size_t referenceIndex,
                               float continuityWeight, std::size_t continuityFreeZone, TrackProjection& best,
                               float& bestScore) const;

    TrackDefinition m_definition;

    std::vector<Vector2> m_centerline;
    std::vector<float> m_segmentLengths;      // length of segment i: centerline[i] -> centerline[(i+1) % N]
    std::vector<float> m_cumulativeDistances; // arc length from sample 0 to sample i
    float m_totalLength = 0.0f;

    std::vector<std::uint8_t> m_mask;

    Vector2 m_spawnPosition = {0.0f, 0.0f};
    float m_spawnHeading = 0.0f;
};

// User-authored extreme track: a hand-traced 39-control-point closed loop
// following assets/tracks/extreme/track_mask.png. Road width genuinely
// varies (collision from maskImagePath only), and it deliberately has
// non-adjacent corridors passing close together -- what TrackProgress's
// local tracking/recovery exist to handle.
TrackDefinition createExtremeTrackDefinition(int simWidth, int simHeight, const std::string& visualImagePath,
                                              const std::string& maskImagePath);

} // namespace simulation
