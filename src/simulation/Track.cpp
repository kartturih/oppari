#include "simulation/Track.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace simulation
{

namespace
{

void validate(const TrackDefinition& def)
{
    if (def.simWidth <= 0 || def.simHeight <= 0)
    {
        throw std::invalid_argument("TrackDefinition: simWidth/simHeight must be positive");
    }

    if (def.controlPoints.size() < 4)
    {
        throw std::invalid_argument("TrackDefinition: at least 4 control points are required for a closed loop");
    }

    if (!(def.trackWidth > 0.0f) || !std::isfinite(def.trackWidth))
    {
        throw std::invalid_argument("TrackDefinition: trackWidth must be positive and finite");
    }

    if (def.samplesPerSegment < 2)
    {
        throw std::invalid_argument("TrackDefinition: samplesPerSegment must be >= 2");
    }

    for (const Vector2& point : def.controlPoints)
    {
        if (!std::isfinite(point.x) || !std::isfinite(point.y))
        {
            throw std::invalid_argument("TrackDefinition: control point coordinates must be finite");
        }
    }
}

// Uniform Catmull-Rom between p1 and p2 (p0/p3 shape the end tangents),
// t in [0,1]. t=0 -> p1, t=1 -> p2.
Vector2 catmullRom(const Vector2& p0, const Vector2& p1, const Vector2& p2, const Vector2& p3, float t)
{
    const float t2 = t * t;
    const float t3 = t2 * t;

    const float x = 0.5f * ((2.0f * p1.x) + (-p0.x + p2.x) * t + (2.0f * p0.x - 5.0f * p1.x + 4.0f * p2.x - p3.x) * t2 +
                             (-p0.x + 3.0f * p1.x - 3.0f * p2.x + p3.x) * t3);
    const float y = 0.5f * ((2.0f * p1.y) + (-p0.y + p2.y) * t + (2.0f * p0.y - 5.0f * p1.y + 4.0f * p2.y - p3.y) * t2 +
                             (-p0.y + 3.0f * p1.y - 3.0f * p2.y + p3.y) * t3);

    return Vector2{x, y};
}

struct SegmentProjection
{
    Vector2 point;
    float t;
};

// Clamped projection of p onto segment a->b; zero-length segments resolve
// to t = 0, point = a.
SegmentProjection closestPointOnSegment(Vector2 p, Vector2 a, Vector2 b)
{
    const Vector2 ab = {b.x - a.x, b.y - a.y};
    const float lengthSq = ab.x * ab.x + ab.y * ab.y;

    float t = 0.0f;
    if (lengthSq > 0.0f)
    {
        const Vector2 ap = {p.x - a.x, p.y - a.y};
        t = (ap.x * ab.x + ap.y * ab.y) / lengthSq;
        t = std::clamp(t, 0.0f, 1.0f);
    }

    return SegmentProjection{Vector2{a.x + ab.x * t, a.y + ab.y * t}, t};
}

} // namespace

Track::Track(const TrackDefinition& definition) : m_definition(definition)
{
    validate(m_definition);
    validateVisualAsset();
    buildCenterline();
    buildArcLength();
    buildMask();
    computeSpawn();
}

void Track::buildCenterline()
{
    const std::vector<Vector2>& controlPoints = m_definition.controlPoints;
    const std::size_t pointCount = controlPoints.size();
    const int samplesPerSegment = m_definition.samplesPerSegment;

    m_centerline.clear();
    m_centerline.reserve(pointCount * static_cast<std::size_t>(samplesPerSegment));

    for (std::size_t i = 0; i < pointCount; ++i)
    {
        const Vector2& p0 = controlPoints[(i + pointCount - 1) % pointCount];
        const Vector2& p1 = controlPoints[i];
        const Vector2& p2 = controlPoints[(i + 1) % pointCount];
        const Vector2& p3 = controlPoints[(i + 2) % pointCount];

        // t never reaches 1 -- that sample is p2, produced again as the next
        // segment's t=0, so sampling it here would duplicate the seam.
        for (int j = 0; j < samplesPerSegment; ++j)
        {
            const float t = static_cast<float>(j) / static_cast<float>(samplesPerSegment);
            m_centerline.push_back(catmullRom(p0, p1, p2, p3, t));
        }
    }
}

void Track::buildArcLength()
{
    const std::size_t sampleCount = m_centerline.size();
    m_segmentLengths.assign(sampleCount, 0.0f);
    m_cumulativeDistances.assign(sampleCount, 0.0f);

    float cumulative = 0.0f;
    for (std::size_t i = 0; i < sampleCount; ++i)
    {
        m_cumulativeDistances[i] = cumulative;

        const Vector2& a = m_centerline[i];
        const Vector2& b = m_centerline[(i + 1) % sampleCount];
        const float dx = b.x - a.x;
        const float dy = b.y - a.y;
        const float length = std::sqrt(dx * dx + dy * dy);

        m_segmentLengths[i] = length;
        cumulative += length;
    }

    m_totalLength = cumulative;
}

void Track::validateVisualAsset() const
{
    if (m_definition.visualImagePath.empty())
    {
        return;
    }

    Image image = LoadImage(m_definition.visualImagePath.c_str());
    if (image.data == nullptr)
    {
        throw std::invalid_argument("Track: failed to load visual image asset: " + m_definition.visualImagePath);
    }

    const bool dimensionsMatch = image.width == m_definition.simWidth && image.height == m_definition.simHeight;
    UnloadImage(image);

    if (!dimensionsMatch)
    {
        throw std::invalid_argument("Track: visual image dimensions do not match simulation dimensions: " +
                                     m_definition.visualImagePath);
    }
}

void Track::buildMask()
{
    if (!m_definition.maskImagePath.empty())
    {
        buildMaskFromImage();
    }
    else
    {
        buildMaskFromCenterlineWidth();
    }
}

void Track::buildMaskFromImage()
{
    const int width = m_definition.simWidth;
    const int height = m_definition.simHeight;

    Image image = LoadImage(m_definition.maskImagePath.c_str());
    if (image.data == nullptr)
    {
        throw std::invalid_argument("Track: failed to load mask image asset: " + m_definition.maskImagePath);
    }

    if (image.width != width || image.height != height)
    {
        UnloadImage(image);
        throw std::invalid_argument("Track: mask image dimensions do not match simulation dimensions: " +
                                     m_definition.maskImagePath);
    }

    // Normalize format so pixel access below is consistent.
    ImageFormat(&image, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8);
    Color* pixels = LoadImageColors(image);

    m_mask.assign(static_cast<std::size_t>(width) * static_cast<std::size_t>(height), 0);

    // Luminance >= 128 -> drivable; alpha ignored.
    constexpr float kDrivableLuminanceThreshold = 128.0f;
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const Color c = pixels[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                                    static_cast<std::size_t>(x)];
            const float luminance = 0.299f * static_cast<float>(c.r) + 0.587f * static_cast<float>(c.g) +
                                     0.114f * static_cast<float>(c.b);
            if (luminance >= kDrivableLuminanceThreshold)
            {
                m_mask[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)] =
                    1;
            }
        }
    }

    UnloadImageColors(pixels);
    UnloadImage(image);
}

void Track::buildMaskFromCenterlineWidth()
{
    const int width = m_definition.simWidth;
    const int height = m_definition.simHeight;

    m_mask.assign(static_cast<std::size_t>(width) * static_cast<std::size_t>(height), 0);

    // Only the centerline's bounding box (expanded by trackWidth/2) needs
    // scanning -- nothing outside it can be drivable.
    float minX = m_centerline[0].x;
    float maxX = m_centerline[0].x;
    float minY = m_centerline[0].y;
    float maxY = m_centerline[0].y;
    for (const Vector2& point : m_centerline)
    {
        minX = std::min(minX, point.x);
        maxX = std::max(maxX, point.x);
        minY = std::min(minY, point.y);
        maxY = std::max(maxY, point.y);
    }

    const float halfWidth = m_definition.trackWidth * 0.5f;
    const int startX = std::max(0, static_cast<int>(std::floor(minX - halfWidth)));
    const int endX = std::min(width - 1, static_cast<int>(std::ceil(maxX + halfWidth)));
    const int startY = std::max(0, static_cast<int>(std::floor(minY - halfWidth)));
    const int endY = std::min(height - 1, static_cast<int>(std::ceil(maxY + halfWidth)));

    const float halfWidthSq = halfWidth * halfWidth;
    const std::size_t segmentCount = m_centerline.size();

    for (int y = startY; y <= endY; ++y)
    {
        for (int x = startX; x <= endX; ++x)
        {
            const Vector2 pixel = {static_cast<float>(x), static_cast<float>(y)};

            float bestDistSq = std::numeric_limits<float>::max();
            for (std::size_t i = 0; i < segmentCount; ++i)
            {
                const Vector2& a = m_centerline[i];
                const Vector2& b = m_centerline[(i + 1) % segmentCount];
                const SegmentProjection proj = closestPointOnSegment(pixel, a, b);

                const float dx = pixel.x - proj.point.x;
                const float dy = pixel.y - proj.point.y;
                const float distSq = dx * dx + dy * dy;
                if (distSq < bestDistSq)
                {
                    bestDistSq = distSq;
                }
            }

            if (bestDistSq <= halfWidthSq)
            {
                m_mask[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)] = 1;
            }
        }
    }
}

void Track::computeSpawn()
{
    // Default (0.0f) case handled exactly (not via getPointAtDistance()'s
    // fmod()-based lookup) so spawn pose never shifts by even a float ULP.
    if (m_definition.spawnDistanceAlongTrack == 0.0f)
    {
        m_spawnPosition = m_centerline[0];
        const Vector2& next = m_centerline[1 % m_centerline.size()];
        m_spawnHeading = std::atan2(next.y - m_spawnPosition.y, next.x - m_spawnPosition.x);
    }
    else
    {
        m_spawnPosition = getPointAtDistance(m_definition.spawnDistanceAlongTrack);

        // Small forward probe to estimate the tangent at an arbitrary distance.
        constexpr float kHeadingProbeDistance = 1.0f; // px, well below any segment length
        const Vector2 ahead = getPointAtDistance(m_definition.spawnDistanceAlongTrack + kHeadingProbeDistance);
        m_spawnHeading = std::atan2(ahead.y - m_spawnPosition.y, ahead.x - m_spawnPosition.x);
    }

    const int spawnX = static_cast<int>(std::lround(m_spawnPosition.x));
    const int spawnY = static_cast<int>(std::lround(m_spawnPosition.y));
    if (!isDrivable(spawnX, spawnY))
    {
        throw std::invalid_argument("Track: computed spawn position is not drivable in the mask");
    }
}

bool Track::isDrivable(int x, int y) const
{
    if (x < 0 || y < 0 || x >= m_definition.simWidth || y >= m_definition.simHeight)
    {
        return false;
    }

    const std::size_t index = static_cast<std::size_t>(y) * static_cast<std::size_t>(m_definition.simWidth) +
                               static_cast<std::size_t>(x);

    return m_mask[index] != 0;
}

void Track::considerSegment(Vector2 position, std::size_t segmentIndex, TrackProjection& best, float& bestDistSq) const
{
    const std::size_t sampleCount = m_centerline.size();
    const Vector2& a = m_centerline[segmentIndex];
    const Vector2& b = m_centerline[(segmentIndex + 1) % sampleCount];
    const SegmentProjection proj = closestPointOnSegment(position, a, b);

    const float dx = position.x - proj.point.x;
    const float dy = position.y - proj.point.y;
    const float distSq = dx * dx + dy * dy;

    // Strict '<': lowest-index segment wins ties.
    if (distSq < bestDistSq)
    {
        bestDistSq = distSq;
        best.point = proj.point;
        best.segmentIndex = segmentIndex;
        best.segmentT = proj.t;

        float distanceAlongTrack = m_cumulativeDistances[segmentIndex] + proj.t * m_segmentLengths[segmentIndex];
        if (distanceAlongTrack >= m_totalLength)
        {
            distanceAlongTrack -= m_totalLength;
        }
        if (distanceAlongTrack < 0.0f)
        {
            distanceAlongTrack = 0.0f;
        }
        best.distanceAlongTrack = distanceAlongTrack;
    }
}

TrackProjection Track::projectOntoCenterline(Vector2 position) const
{
    const std::size_t sampleCount = m_centerline.size();

    TrackProjection best;
    float bestDistSq = std::numeric_limits<float>::max();

    for (std::size_t i = 0; i < sampleCount; ++i)
    {
        considerSegment(position, i, best, bestDistSq);
    }

    best.distanceFromCenterline = std::sqrt(bestDistSq);
    return best;
}

void Track::considerSegmentLocal(Vector2 position, std::size_t segmentIndex, std::size_t referenceIndex,
                                  float continuityWeight, std::size_t continuityFreeZone, TrackProjection& best,
                                  float& bestScore) const
{
    const std::size_t sampleCount = m_centerline.size();
    const Vector2& a = m_centerline[segmentIndex];
    const Vector2& b = m_centerline[(segmentIndex + 1) % sampleCount];
    const SegmentProjection proj = closestPointOnSegment(position, a, b);

    const float dx = position.x - proj.point.x;
    const float dy = position.y - proj.point.y;
    const float distance = std::sqrt(dx * dx + dy * dy);

    // Shortest-direction, wrap-aware index gap from referenceIndex.
    const std::size_t forwardGap = (segmentIndex + sampleCount - referenceIndex) % sampleCount;
    const std::size_t indexGap = std::min(forwardGap, sampleCount - forwardGap);

    // continuityFreeZone samples are penalty-free (so ordinary forward
    // movement is never penalized vs. standing still); beyond that the
    // penalty grows as sqrt(gap), not linearly -- strong enough to break
    // small-gap ties without letting a large legitimate single-frame jump
    // lose to a nearby-but-wrong candidate. See TrackProgress::
    // kLocalContinuityWeight/kLocalContinuityFreeZone for the tuned values.
    const std::size_t penalizedGap = indexGap > continuityFreeZone ? indexGap - continuityFreeZone : 0;
    const float score = distance + continuityWeight * std::sqrt(static_cast<float>(penalizedGap));

    // Strict '<': lowest scan-order segment wins ties (matches considerSegment()).
    if (score < bestScore)
    {
        bestScore = score;
        best.point = proj.point;
        best.segmentIndex = segmentIndex;
        best.segmentT = proj.t;
        best.distanceFromCenterline = distance; // true geometric distance, never the penalized score

        float distanceAlongTrack = m_cumulativeDistances[segmentIndex] + proj.t * m_segmentLengths[segmentIndex];
        if (distanceAlongTrack >= m_totalLength)
        {
            distanceAlongTrack -= m_totalLength;
        }
        if (distanceAlongTrack < 0.0f)
        {
            distanceAlongTrack = 0.0f;
        }
        best.distanceAlongTrack = distanceAlongTrack;
    }
}

TrackProjection Track::projectOntoCenterlineLocal(Vector2 position, std::size_t previousSegmentIndex,
                                                    std::size_t searchRadius, float continuityWeight,
                                                    std::size_t continuityFreeZone) const
{
    const std::size_t sampleCount = m_centerline.size();
    previousSegmentIndex %= sampleCount;

    // Clamp so a large radius covers the loop exactly once (degenerates to
    // a full projectOntoCenterline() scan) rather than revisiting segments.
    const std::size_t radius = std::min(searchRadius, sampleCount - 1);
    const std::size_t windowSize = std::min(sampleCount, 2 * radius + 1);
    const std::size_t start = (previousSegmentIndex + sampleCount - radius) % sampleCount;

    TrackProjection best;
    float bestScore = std::numeric_limits<float>::max();

    for (std::size_t step = 0; step < windowSize; ++step)
    {
        const std::size_t i = (start + step) % sampleCount;
        considerSegmentLocal(position, i, previousSegmentIndex, continuityWeight, continuityFreeZone, best,
                              bestScore);
    }

    return best;
}

Vector2 Track::getPointAtDistance(float distanceAlongTrack) const
{
    float distance = std::fmod(distanceAlongTrack, m_totalLength);
    if (distance < 0.0f)
    {
        distance += m_totalLength;
    }

    const std::size_t sampleCount = m_centerline.size();
    std::size_t segmentIndex = sampleCount - 1;
    for (std::size_t i = 0; i < sampleCount; ++i)
    {
        const std::size_t nextIndex = (i + 1) % sampleCount;
        const float startDistance = m_cumulativeDistances[i];
        const float endDistance = (nextIndex == 0) ? m_totalLength : m_cumulativeDistances[nextIndex];
        if (distance >= startDistance && distance < endDistance)
        {
            segmentIndex = i;
            break;
        }
    }

    const float t = (m_segmentLengths[segmentIndex] > 0.0f)
                         ? (distance - m_cumulativeDistances[segmentIndex]) / m_segmentLengths[segmentIndex]
                         : 0.0f;

    const Vector2& a = m_centerline[segmentIndex];
    const Vector2& b = m_centerline[(segmentIndex + 1) % sampleCount];
    return Vector2{a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t};
}

TrackDefinition createHardTrackDefinition(int simWidth, int simHeight)
{
    // Hand-authored asymmetric closed circuit (1200x700 area): a long
    // straight (spawn), a broad sweeping corner, an S-chicane, two tighter
    // corners, and a closing section -- see inline point comments below.
    // Every section keeps generous clearance from every other non-adjacent
    // one (verified by verifyTrack() in main.cpp).
    TrackDefinition def;
    def.simWidth = simWidth;
    def.simHeight = simHeight;
    def.trackWidth = 110.0f;
    def.samplesPerSegment = 24;

    def.controlPoints = {
        // Long straight (spawn sits on P0, heading toward P1).
        Vector2{250.0f, 580.0f}, // P0 -- spawn
        Vector2{500.0f, 580.0f}, // P1
        Vector2{850.0f, 580.0f}, // P2

        // Broad sweeping corner, bottom-right around to the right side.
        Vector2{1000.0f, 540.0f}, // P3
        Vector2{1080.0f, 400.0f}, // P4
        Vector2{1020.0f, 220.0f}, // P5

        // S-shaped chicane across the top: dip down, then rise back up.
        Vector2{880.0f, 110.0f}, // P6
        Vector2{620.0f, 170.0f}, // P7
        Vector2{400.0f, 100.0f}, // P8

        // Tighter corner #1: sharp apex turning the circuit downward.
        Vector2{270.0f, 160.0f}, // P9

        // Gentle descent down the left side.
        Vector2{150.0f, 320.0f}, // P10

        // Tighter corner #2, positioned so the spawn tangent (via P1-P11) comes out level.
        Vector2{50.0f, 550.0f}, // P11
    };

    return def;
}

TrackDefinition createStage18TestTrackDefinition(int simWidth, int simHeight, const std::string& visualImagePath,
                                                   const std::string& maskImagePath)
{
    // Reuses createHardTrackDefinition()'s control points/sample density.
    // IMPORTANT: must exactly match CONTROL_POINTS/SAMPLES_PER_SEGMENT in
    // assets/tracks/test/generate_test_track_assets.py (the rasterizer for
    // track_visual.png/track_mask.png) -- regenerate assets if either changes.
    TrackDefinition def = createHardTrackDefinition(simWidth, simHeight);

    def.visualImagePath = visualImagePath;
    def.maskImagePath = maskImagePath;

    // trackWidth is now just a separation-check floor (mask comes from
    // maskImagePath); spawnDistanceAlongTrack stays at its default (sample 0).

    return def;
}

TrackDefinition createExtremeTrackDefinition(int simWidth, int simHeight, const std::string& visualImagePath,
                                              const std::string& maskImagePath)
{
    // Stage 19: a hand-traced closed loop following the actual road drawn in
    // assets/tracks/extreme/track_mask.png (1200x700). The control points
    // below were chosen by inspecting the mask's connected drivable region
    // directly (medial axis, traced offline), denser at hairpins/reversals.
    // Verified offline: the sampled centerline lands 100% inside the
    // drivable region, clearance never below 36px (car is 12x24px).
    //
    // Route (P0 = spawn, top straight) traces the mask clockwise through 5
    // horizontal corridors connected by corners and two notch detours/hooks
    // -- see each point's inline label below. The third and a later
    // corridor pass within ~94px of each other without touching; kept
    // deliberately, since it's exactly the case TrackProgress's local
    // tracking/recovery exist to handle.
    TrackDefinition def;
    def.simWidth = simWidth;
    def.simHeight = simHeight;
    def.samplesPerSegment = 20;

    // Nominal only (collision uses maskImagePath, which varies ~72-202px);
    // set to the spawn corridor's measured width.
    def.trackWidth = 130.0f;

    def.visualImagePath = visualImagePath;
    def.maskImagePath = maskImagePath;

    def.controlPoints = {
        Vector2{984.0f, 72.0f},  // P0 -- spawn, top straight
        Vector2{584.0f, 72.0f},  // P1
        Vector2{184.0f, 72.0f},  // P2
        Vector2{132.0f, 88.0f},  // P3
        Vector2{116.0f, 121.0f}, // P4 -- top-left corner apex
        Vector2{126.0f, 156.0f}, // P5
        Vector2{146.0f, 188.0f}, // P6
        Vector2{222.0f, 195.0f}, // P7
        Vector2{462.0f, 195.0f}, // P8
        Vector2{702.0f, 195.0f}, // P9
        Vector2{774.0f, 210.0f}, // P10
        Vector2{802.0f, 279.0f}, // P11 -- notch detour apex
        Vector2{775.0f, 347.0f}, // P12
        Vector2{662.0f, 360.0f}, // P13
        Vector2{486.0f, 306.0f}, // P14
        Vector2{246.0f, 306.0f}, // P15
        Vector2{166.0f, 306.0f}, // P16
        Vector2{119.0f, 358.0f}, // P17 -- left corner apex
        Vector2{159.0f, 417.0f}, // P18
        Vector2{319.0f, 417.0f}, // P19
        Vector2{373.0f, 458.0f}, // P20 -- hook apex 1
        Vector2{348.0f, 523.0f}, // P21 -- hook apex 2
        Vector2{190.0f, 527.0f}, // P22
        Vector2{130.0f, 559.0f}, // P23
        Vector2{143.0f, 626.0f}, // P24 -- left corner apex
        Vector2{296.0f, 638.0f}, // P25
        Vector2{536.0f, 638.0f}, // P26
        Vector2{585.0f, 589.0f}, // P27
        Vector2{663.0f, 472.0f}, // P28 -- top of S-connector
        Vector2{743.0f, 472.0f}, // P29
        Vector2{814.0f, 597.0f}, // P30
        Vector2{869.0f, 638.0f}, // P31
        Vector2{1029.0f, 638.0f}, // P32
        Vector2{1080.0f, 593.0f}, // P33 -- right corner apex
        Vector2{1003.0f, 483.0f}, // P34
        Vector2{1008.0f, 334.0f}, // P35
        Vector2{1093.0f, 223.0f}, // P36
        Vector2{1092.0f, 143.0f}, // P37 -- top-right corner apex
        Vector2{1063.0f, 76.0f},  // P38
    };

    // spawnDistanceAlongTrack stays at its default: spawn = P0, heading
    // toward P1, with the whole top straight ahead.

    return def;
}

} // namespace simulation
