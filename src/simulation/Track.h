#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "raylib.h"

namespace simulation
{

// Describes a track as three independent layers (Stage 18):
//
//   1. VISUAL   -- visualImagePath, an image drawn as the track background.
//                  Read only by TrackVisual (GPU rendering); Track itself
//                  never inspects its pixels.
//   2. MASK     -- maskImagePath, a binary (thresholded) image defining
//                  exactly where cars may drive. If non-empty, this is the
//                  sole source of Track's CPU drivable mask -- no width
//                  constant, no centerline distance, is involved. If empty,
//                  Track falls back to the original Stage 14 behavior of
//                  deriving the mask from controlPoints + trackWidth (kept
//                  only for the existing procedurally-generated tracks --
//                  see createHardTrackDefinition() -- so they keep working
//                  unmodified; new image-based tracks should always set
//                  maskImagePath).
//   3. CENTERLINE -- controlPoints + samplesPerSegment, interpolated into a
//                  dense, smooth, closed sampled centerline exactly as
//                  before. Used only for progress, checkpoints, spawn and
//                  forward direction -- never for collision when
//                  maskImagePath is set.
//
// These three layers are independent by construction: nothing in Track ever
// derives the mask from the visual image, or the visual rendering from the
// mask, or collision geometry from the centerline once maskImagePath is set.
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

    // Legacy/verification-only nominal road width: the sole basis of the
    // CPU mask when maskImagePath is empty (Stage 14 procedural tracks),
    // and otherwise used only as a separation heuristic by main.cpp's
    // centerline self-intersection verification -- never by an image-based
    // track's actual collision geometry, which reads maskImagePath instead.
    float trackWidth = 0.0f;

    int samplesPerSegment = 0;

    // Stage 18 image-based layers. Both empty (the Stage 14 default) means
    // "no image assets": rendering falls back to drawing the sampled
    // centerline directly, and the mask is derived from controlPoints +
    // trackWidth. If maskImagePath is set, visualImagePath should be too
    // (Track validates each independently; see Track.cpp).
    std::string visualImagePath;
    std::string maskImagePath;

    // Arc-length distance (wrapped into [0, total length)) along the
    // sampled centerline where the spawn pose is placed. Defaults to 0.0f,
    // i.e. exactly centerline sample index 0 -- byte-identical to Stage 14's
    // original hardcoded "spawn = centerline[0]" behavior.
    float spawnDistanceAlongTrack = 0.0f;
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
    // samplesPerSegment < 2. Also throws if visualImagePath or maskImagePath
    // is set but cannot be loaded, or its dimensions do not match
    // (simWidth, simHeight) -- see validateVisualAsset()/buildMask(). Also
    // throws if the computed spawn position is not drivable in the
    // resulting mask (see computeSpawn()). All image loading here is
    // CPU-only (raylib's LoadImage(), never LoadTexture()), so constructing
    // a Track never requires an initialized raylib window -- GPU texture
    // creation for the visual layer is a separate, later step (see
    // TrackVisual).
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

    // Local, previous-segment-biased variant of projectOntoCenterline():
    // scans only a window of 2*searchRadius + 1 segments centered on
    // previousSegmentIndex (clamped/wrapped correctly for the closed loop,
    // including across the seam), rather than every segment. Intended for
    // per-frame progress tracking (see TrackProgress) on a dense track
    // where two non-adjacent stretches of centerline could otherwise pass
    // close enough together for a full projectOntoCenterline() scan to jump
    // between them -- restricting the search window prevents that ambiguity
    // by construction. If searchRadius is large enough that the window
    // would cover the whole closed loop, every segment is considered
    // exactly once, matching projectOntoCenterline(). previousSegmentIndex
    // is taken modulo the sample count, so a stale index from a
    // differently-sized centerline can never index out of bounds.
    //
    // continuityWeight/continuityFreeZone (Stage 19.1): within the search
    // window, candidates are ranked by
    // distance + continuityWeight * sqrt(max(0, indexGapFromPrevious - continuityFreeZone))
    // (indexGapFromPrevious is the shortest-direction, wrap-aware distance
    // in samples from previousSegmentIndex -- see considerSegmentLocal()),
    // not by raw distance alone. The window already excludes segments more
    // than searchRadius away, but within a wide window on a dense/twisty
    // track, two *different* stretches of centerline that pass close to
    // each other can both end up nearly equidistant from the query
    // position -- with continuityWeight 0.0f (the default), the
    // geometrically nearer one wins even when it is a much larger index
    // jump than the other, which is exactly the "geometrically nearby,
    // topologically distant" ambiguity local tracking exists to avoid. A
    // nonzero weight breaks such near-ties in favor of continuing near
    // previousSegmentIndex.
    //
    // continuityFreeZone matters because the penalty must never make
    // *ordinary* forward progress look worse than standing still: once a
    // candidate is already a close match (small distance), even a single
    // sample of real forward movement is a strictly better match story than
    // staying put, so a gap of a handful of samples must cost nothing.
    // continuityFreeZone samples are exempted from the penalty entirely
    // before the (still sub-linear, square-root) cost applies beyond it --
    // see considerSegmentLocal()'s comment and
    // TrackProgress::kLocalContinuityWeight/kLocalContinuityFreeZone for the
    // empirical derivation of both. The returned TrackProjection's
    // distanceFromCenterline is always the winning candidate's true
    // (unpenalized) geometric distance -- continuityWeight only affects
    // which candidate wins, never what distance is reported for it (callers
    // like TrackProgress's recovery-distance check depend on that being
    // real). Both default to 0 (pure nearest-distance, Stage 18 behavior)
    // so existing callers that don't pass these arguments are unaffected.
    TrackProjection projectOntoCenterlineLocal(Vector2 position, std::size_t previousSegmentIndex,
                                                std::size_t searchRadius, float continuityWeight = 0.0f,
                                                std::size_t continuityFreeZone = 0) const;

    // The point on the closed centerline loop at the given arc-length
    // distance from sample 0, in forward-travel order. `distanceAlongTrack`
    // is wrapped into [0, getTotalLength()) first, so any finite input is
    // accepted. Provided for spawn placement and for deterministic
    // verification; not used by any per-frame collision/sensor path.
    Vector2 getPointAtDistance(float distanceAlongTrack) const;

    // A deterministic spawn pose derived entirely from the centerline, at
    // arc-length distance m_definition.spawnDistanceAlongTrack from sample
    // 0 (default 0.0f, i.e. exactly centerline sample index 0 -- the
    // original Stage 14 behavior). Heading is that point's forward tangent
    // direction. The constructor verifies the resulting position is
    // drivable in the mask (see computeSpawn()); it does not clamp or
    // adjust the position if not, it throws.
    Vector2 getSpawnPosition() const { return m_spawnPosition; }
    float getSpawnHeading() const { return m_spawnHeading; }

private:
    void buildCenterline();
    void buildArcLength();

    // Dispatches to buildMaskFromImage() when m_definition.maskImagePath is
    // set, or buildMaskFromCenterlineWidth() (the original Stage 14
    // algorithm) otherwise.
    void buildMask();
    void buildMaskFromCenterlineWidth();
    void buildMaskFromImage();

    // CPU-only (LoadImage/UnloadImage, no GPU): throws if
    // m_definition.visualImagePath is set but cannot be loaded or its
    // dimensions do not match (simWidth, simHeight). Does not store any
    // image data -- Track never reads visual pixels, only validates that
    // the asset TrackVisual will later load actually matches this Track.
    void validateVisualAsset() const;

    void computeSpawn();

    // Shared by projectOntoCenterline(): folds segment `segmentIndex`'s
    // clamped projection of `position` into (best, bestDistSq) if it is
    // strictly closer than the current best. Pure nearest-distance, no
    // continuity bias -- used only by the unbiased global scan (recovery
    // and reset() need the true nearest match, not a continuity-preferring
    // one).
    void considerSegment(Vector2 position, std::size_t segmentIndex, TrackProjection& best, float& bestDistSq) const;

    // Used only by projectOntoCenterlineLocal(): folds segment
    // `segmentIndex`'s clamped projection of `position` into (best,
    // bestScore) if its continuity-weighted score -- true geometric
    // distance plus continuityWeight * sqrt(max(0, indexGap - continuityFreeZone)),
    // where indexGap is its wrap-aware index gap from referenceIndex -- is
    // strictly lower than the current best score. best.distanceFromCenterline
    // is set to the true geometric distance, never the score, regardless of
    // which candidate wins.
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

// Builds the deterministic, hand-authored Stage 14B hard training track: an
// asymmetric closed circuit with a long straight, a broad sweeping corner,
// two noticeably tighter corners, and an S-shaped chicane, laid out with
// generous separation between non-adjacent sections so the closed loop does
// not self-intersect. See Track.cpp for the exact control point placement
// and the reasoning behind each section's shape.
TrackDefinition createHardTrackDefinition(int simWidth, int simHeight);

// Stage 18: builds the deterministic image-based test track definition --
// reuses createHardTrackDefinition()'s exact centerline (already verified
// non-self-intersecting with generous section separation) purely for
// progress/checkpoints/spawn, and points visualImagePath/maskImagePath at
// the Stage 18 test assets (see assets/tracks/test/) for rendering and
// collision. This is the track main() actually runs Stage 18's population
// on -- see assets/tracks/test/generate_test_track_assets.py for exactly
// how track_visual.png/track_mask.png were produced, and why the mask's
// drivable band deliberately varies in width along the track while the
// visual road does not (demonstrating the two layers' independence).
TrackDefinition createStage18TestTrackDefinition(int simWidth, int simHeight, const std::string& visualImagePath,
                                                  const std::string& maskImagePath);

// Stage 19: builds the definition for the user-authored extreme track -- a
// hand-traced, 39-control-point closed loop following the actual road
// geometry of assets/tracks/extreme/track_mask.png (see Track.cpp for how
// each control point was chosen and the route's section-by-section
// layout). Unlike Stage 18's test track, this track's road width genuinely
// varies (collision comes entirely from maskImagePath, never from
// trackWidth), and it deliberately contains non-adjacent corridors that
// pass close to each other -- exactly the case TrackProgress's local
// projection tracking and global recovery exist for.
TrackDefinition createExtremeTrackDefinition(int simWidth, int simHeight, const std::string& visualImagePath,
                                              const std::string& maskImagePath);

} // namespace simulation
