#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "raylib.h"

namespace simulation
{

// Describes a closed, centerline-based track: a simulation-area size, a
// closed loop of control points, and a uniform road width. The control
// points are interpolated into a dense, smooth, closed sampled centerline
// (see Track) -- TrackDefinition itself carries no derived geometry.
//
// Control points do not need to lie inside [0, simWidth) x [0, simHeight) --
// Track does not clamp or reject them on that basis. A control point (or an
// interpolated centerline sample near it) outside the simulation bounds
// simply never becomes drivable, since isDrivable() rejects out-of-bounds
// coordinates outright; it does not otherwise affect validity.
struct TrackDefinition
{
    int simWidth = 0;
    int simHeight = 0;

    std::vector<Vector2> controlPoints;

    float trackWidth = 0.0f;

    int samplesPerSegment = 0;
};

// Result of projecting a world-space position onto the closest point of a
// Track's sampled centerline.
struct TrackProjection
{
    Vector2 point;                  // the closest point on the centerline
    std::size_t segmentIndex = 0;   // index of the centerline segment the point lies on
    float segmentT = 0.0f;          // clamped parametric position within that segment, in [0,1]
    float distanceAlongTrack = 0.0f; // cumulative arc length to `point`, in [0, getTotalLength())
    float distanceFromCenterline = 0.0f; // perpendicular distance from the query position to `point`
};

// Holds a closed, centerline-based track's definition, its dense sampled
// centerline and arc-length table, and a precomputed CPU-side drivable mask
// derived from that centerline. Collision/sensor queries only ever read the
// mask; the centerline geometry is evaluated once, during construction, and
// is the single source of truth for the mask, for rendering, and for
// progress/checkpoint/spawn queries alike -- there is no separate ellipse
// (or other shape-specific) model anywhere in this class.
class Track
{
public:
    // Throws std::invalid_argument if the definition is invalid: non-positive
    // sim dimensions, fewer than 4 control points, a non-finite control
    // point coordinate, a non-positive/non-finite track width, or
    // samplesPerSegment < 2.
    explicit Track(const TrackDefinition& definition);

    // Returns false for coordinates outside the simulation area without
    // touching the mask.
    bool isDrivable(int x, int y) const;

    const TrackDefinition& getDefinition() const { return m_definition; }

    // The dense, closed sampled centerline, in forward-travel order. Does
    // NOT repeat the first sample at the end -- the closing segment from
    // the last sample back to the first is segment index size() - 1 and is
    // real, just not duplicated in this array.
    const std::vector<Vector2>& getCenterlineSamples() const { return m_centerline; }

    // Cumulative arc length at each centerline sample: cumulativeDistances()[0]
    // is always 0, and cumulativeDistances()[i] is the arc length from
    // sample 0 to sample i, in forward-travel order. Monotonically
    // increasing; same size as getCenterlineSamples().
    const std::vector<float>& getCumulativeDistances() const { return m_cumulativeDistances; }

    // Total length of the closed centerline loop (sum of every segment's
    // length, including the closing segment from the last sample back to
    // the first). Always > 0 for a validly constructed Track.
    float getTotalLength() const { return m_totalLength; }

    // Finds the closest point on the sampled centerline to `position`,
    // checking the clamped projection onto every one of the closed loop's
    // segments and keeping the closest. Ties (equal distance) are broken
    // deterministically by lower segment index. This is the single
    // reusable query all arc-length-based progress logic is built from; it
    // is NOT called per sensor sample or per collision check (those stay
    // O(1) via the mask).
    TrackProjection projectOntoCenterline(Vector2 position) const;

    // The point on the closed centerline loop at the given arc-length
    // distance from sample 0, in forward-travel order. `distanceAlongTrack`
    // is wrapped into [0, getTotalLength()) first, so any finite input is
    // accepted. Provided for spawn placement and for deterministic
    // verification; not used by any per-frame collision/sensor path.
    Vector2 getPointAtDistance(float distanceAlongTrack) const;

    // A deterministic spawn pose derived entirely from the centerline: the
    // position is the first centerline sample (index 0), and the heading
    // is that sample's forward tangent direction (toward sample 1),
    // i.e. exactly where forward travel along the centerline begins.
    Vector2 getSpawnPosition() const { return m_spawnPosition; }
    float getSpawnHeading() const { return m_spawnHeading; }

private:
    void buildCenterline();
    void buildArcLength();
    void buildMask();
    void computeSpawn();

    TrackDefinition m_definition;

    std::vector<Vector2> m_centerline;
    std::vector<float> m_segmentLengths;      // length of segment i: centerline[i] -> centerline[(i+1) % N]
    std::vector<float> m_cumulativeDistances; // arc length from sample 0 to sample i
    float m_totalLength = 0.0f;

    std::vector<std::uint8_t> m_mask;

    Vector2 m_spawnPosition = {0.0f, 0.0f};
    float m_spawnHeading = 0.0f;
};

// Builds the deterministic, hand-authored Stage 14B hard training track: an
// asymmetric closed circuit with a long straight, a broad sweeping corner,
// two noticeably tighter corners, and an S-shaped chicane, laid out with
// generous separation between non-adjacent sections so the closed loop does
// not self-intersect. See Track.cpp for the exact control point placement
// and the reasoning behind each section's shape.
TrackDefinition createHardTrackDefinition(int simWidth, int simHeight);

} // namespace simulation
