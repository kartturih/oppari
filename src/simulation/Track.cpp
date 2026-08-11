#include "simulation/Track.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

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

// Uniform Catmull-Rom interpolation between p1 and p2 (p0/p3 are the
// neighboring control points used to shape the tangents at each end),
// t in [0,1]. At t=0 this reduces exactly to p1; at t=1 it reduces exactly
// to p2 -- deterministic, no randomness anywhere.
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

// Clamped projection of p onto the segment a->b. Degenerate (zero-length)
// segments deterministically resolve to t = 0, point = a.
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

        // t deliberately never reaches 1: that sample is p2, which is
        // produced again as t=0 of the next segment -- sampling it here too
        // would duplicate the seam between segments.
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

void Track::buildMask()
{
    const int width = m_definition.simWidth;
    const int height = m_definition.simHeight;

    m_mask.assign(static_cast<std::size_t>(width) * static_cast<std::size_t>(height), 0);

    // A pixel can only be drivable if it lies within trackWidth/2 of some
    // centerline sample, so only the centerline's bounding box (expanded by
    // that margin) ever needs to be scanned -- pixels outside it are left
    // at their zero-initialized default without touching a single segment.
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
    m_spawnPosition = m_centerline[0];

    const Vector2& next = m_centerline[1 % m_centerline.size()];
    const float dx = next.x - m_spawnPosition.x;
    const float dy = next.y - m_spawnPosition.y;
    m_spawnHeading = std::atan2(dy, dx);
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

TrackProjection Track::projectOntoCenterline(Vector2 position) const
{
    const std::size_t sampleCount = m_centerline.size();

    TrackProjection best;
    float bestDistSq = std::numeric_limits<float>::max();

    for (std::size_t i = 0; i < sampleCount; ++i)
    {
        const Vector2& a = m_centerline[i];
        const Vector2& b = m_centerline[(i + 1) % sampleCount];
        const SegmentProjection proj = closestPointOnSegment(position, a, b);

        const float dx = position.x - proj.point.x;
        const float dy = position.y - proj.point.y;
        const float distSq = dx * dx + dy * dy;

        // Strict '<' only: the first (lowest-index) segment achieving the
        // minimum distance wins, giving a deterministic tie-break.
        if (distSq < bestDistSq)
        {
            bestDistSq = distSq;
            best.point = proj.point;
            best.segmentIndex = i;
            best.segmentT = proj.t;

            float distanceAlongTrack = m_cumulativeDistances[i] + proj.t * m_segmentLengths[i];
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

    best.distanceFromCenterline = std::sqrt(bestDistSq);
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
    // Stage 14B: a hand-authored, asymmetric closed circuit -- deliberately
    // not an oval -- laid out (for the current 1200x700 simulation area) as
    // a sequence of clearly distinct driving challenges, in forward-travel
    // (control-point index) order:
    //
    //   P0 -> P1 -> P2   : one long straight (~600px) -- sample 0 (spawn)
    //                      sits here, well before the first corner, so cars
    //                      have room to accelerate.
    //   P2 -> P3 -> P4 -> P5 : one broad, wide-radius sweeping corner
    //                      (bottom-right around to the right side).
    //   P5 -> P6 -> P7 -> P8 : an S-shaped chicane across the top of the
    //                      circuit -- P6 -> P7 dips down, then P7 -> P8
    //                      rises back up, a genuine direction reversal.
    //                      This chicane's two legs occupy disjoint x-ranges
    //                      (P6-P7 spans roughly x in [620,880], P7-P8 spans
    //                      roughly x in [400,620]) -- a real zigzag that
    //                      keeps making net leftward progress, rather than
    //                      a fold that doubles back over the same ground
    //                      (which is what earlier design iterations of this
    //                      track got wrong: two bends sharing an x-range
    //                      inevitably pass close to each other away from
    //                      their shared control point, not just at it).
    //   P8 -> P9         : tighter corner #1 -- a noticeably sharper apex
    //                      turning the circuit from heading left to heading
    //                      down.
    //   P9 -> P10 -> P11 : a gentle descent down the left side.
    //   P11              : tighter corner #2 -- a second sharp apex,
    //                      turning the circuit back toward the start.
    //   P11 -> P0        : a closing section back to the spawn straight.
    //
    // Every section keeps generous clearance from every other, non-adjacent
    // section (see verifyTrack()'s self-intersection/separation checks in
    // main.cpp). All coordinates are deliberately hand-placed (no formula/
    // symmetry), and stay within a comfortable margin of the 1200x700
    // simulation area so the road band and Catmull-Rom overshoot near
    // sharp corners never approach the edges.
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

        // Tighter corner #2: sharp apex turning back toward the spawn
        // straight. Positioned low (close to the straight's own y) and far
        // to the left of spawn, so P1 - P11 (the pair Catmull-Rom uses to
        // shape the spawn tangent) is already close to horizontal -- the
        // spawn tangent comes out level, matching the straight it sits on,
        // without needing a separate lead-in control point.
        Vector2{50.0f, 550.0f}, // P11
    };

    return def;
}

} // namespace simulation
