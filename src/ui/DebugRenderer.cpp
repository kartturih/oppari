#include "ui/DebugRenderer.h"

namespace ui
{

void drawProjectionDebug(const simulation::Car& car, const simulation::TrackProgress& progress)
{
    if (!car.isAlive())
    {
        return;
    }

    const simulation::TrackProgress::ProjectionDebugInfo& info = progress.getLastProjectionDebugInfo();
    constexpr Color kProjectionColor = Color{255, 0, 220, 255};

    DrawLineEx(car.getPosition(), info.point, 1.5f, Color{255, 0, 220, 140});
    DrawCircleV(info.point, 5.0f, kProjectionColor);

    constexpr float kTangentArrowLength = 22.0f;
    const Vector2 tangentTip = {info.point.x + info.tangent.x * kTangentArrowLength,
                                 info.point.y + info.tangent.y * kTangentArrowLength};
    DrawLineEx(info.point, tangentTip, 2.5f, kProjectionColor);
    DrawCircleV(tangentTip, 3.0f, kProjectionColor);
}

void reportSuspiciousProjectionJump(std::size_t highlightedIndex, const simulation::TrackProgress& progress,
                                     const simulation::Car& car)
{
    const simulation::TrackProgress::ProjectionDebugInfo& info = progress.getLastProjectionDebugInfo();

    // A real car moves at most ~9.8px per frame (maxSpeed 590px/s at the
    // fixed 1/60s step) -- on the extreme track's centerline (~6.4px average
    // sample spacing), that is roughly 1.5 samples per frame. 15 samples is
    // still a comfortable ~10x that margin, so any LOCAL-mode index jump
    // this large in a single update() cannot reflect real continuous
    // movement -- it can only be local search snapping to a different,
    // geometrically nearby candidate (exactly the failure mode under
    // investigation).
    constexpr int kSuspiciousIndexJumpThreshold = 15;

    // Below this cosine (~72.5 degrees), the new projection's forward
    // tangent points in a meaningfully different direction than where the
    // car was previously tracked. Real curvature changes gradually
    // sample-to-sample (even through a tight hairpin), so a big one-frame
    // swing -- combined with any nonzero index movement -- is corroborating
    // evidence of a jump to an unrelated section, not just a tight corner.
    constexpr float kSuspiciousTangentDotThreshold = 0.3f;

    const float tangentDot = info.previousTangent.x * info.tangent.x + info.previousTangent.y * info.tangent.y;
    const int absIndexDelta = info.indexDelta < 0 ? -info.indexDelta : info.indexDelta;

    const bool suspicious = info.usedRecovery || absIndexDelta > kSuspiciousIndexJumpThreshold ||
                             (info.indexDelta != 0 && tangentDot < kSuspiciousTangentDotThreshold);

    if (!suspicious)
    {
        return;
    }

    const Vector2 pos = car.getPosition();
    TraceLog(LOG_WARNING,
             "[TrackProgress] suspicious projection jump (individual %d): pos=(%.1f,%.1f) index %d -> %d "
             "(delta=%+d), distance=%.1fpx, prevTangent=(%.2f,%.2f), newTangent=(%.2f,%.2f), mode=%s",
             static_cast<int>(highlightedIndex), static_cast<double>(pos.x), static_cast<double>(pos.y),
             static_cast<int>(info.previousIndex), static_cast<int>(info.currentIndex), info.indexDelta,
             static_cast<double>(info.distance), static_cast<double>(info.previousTangent.x),
             static_cast<double>(info.previousTangent.y), static_cast<double>(info.tangent.x),
             static_cast<double>(info.tangent.y), info.usedRecovery ? "RECOVERY" : "LOCAL");
}

} // namespace ui
