#pragma once

#include <algorithm>
#include <cmath>

namespace ai
{

// One evaluation's driving-quality summary, all measured at the real 60 Hz
// control-step rate (one sample per Individual::update(), exactly like
// FitnessEvaluator's own steering-delta sampling). Pure diagnostics: nothing
// here is ever read by fitness, the controller, Population or any selection
// logic -- it exists only so a training run can report HOW a champion drives
// (see training::GenerationMetrics and telemetry::ChampionSummaryRow).
//
//   averageAbsSteeringDelta       -- FitnessEvaluator::getAverageAbsSteeringDelta() verbatim: mean of
//                                    |steeringCmd[t] - steeringCmd[t-1]| (the exact quantity the
//                                    steering-smoothness penalty is built from)
//   steeringReversalsPerSecond    -- sign reversals of the steering command per simulated second;
//                                    see DrivingDiagnostics::kReversalThreshold for the definition
//   steeringSaturationFraction    -- fraction of frames with |steeringCmd| > kSaturationThreshold
//   meanAbsSteering               -- mean |steeringCmd| over every frame (60 Hz, unlike the
//                                    12 Hz-sampled telemetry average_abs_steering)
//   meanLateralAcceleration       -- mean speed * |yawRate| (px/s^2): the centripetal acceleration
//                                    the car's actual turning implies
//   frontSlipBeyondPeakFraction   -- fraction of frames with |front slip angle| above
//                                    CarParams::frontPeakSlipAngle (front tire past its force peak)
//   lap2PlusAverageSpeed          -- mean speed over frames after the first completed lap
//                                    (lapCount >= 1); 0 if no lap was completed
//   physicalBrakeUsageFraction    -- fraction of frames with the physical brake actually sent to the
//                                    car (post throttle/brake combination) above kBrakeUsageThreshold
struct DrivingDiagnosticsSummary
{
    float averageAbsSteeringDelta = 0.0f;
    float steeringReversalsPerSecond = 0.0f;
    float steeringSaturationFraction = 0.0f;
    float meanAbsSteering = 0.0f;
    float meanLateralAcceleration = 0.0f;
    float frontSlipBeyondPeakFraction = 0.0f;
    float lap2PlusAverageSpeed = 0.0f;
    float physicalBrakeUsageFraction = 0.0f;

    // Longitudinal-request diagnostics (see DrivingDiagnostics::recordLongitudinal()):
    //   brakeRequestDominantFraction  -- fraction of frames with brakeRequest > throttleRequest
    //                                    (the frames where the subtractive combination yields a nonzero
    //                                    physical brake; contrast with frames where a smaller brake
    //                                    request merely trims throttle)
    //   throttleRequestMin/Max, brakeRequestMin/Max -- range of the [0,1] requests over the evaluation
    //   brakeOnsetSpeed               -- car speed (px/s) at the FIRST frame the physical brake exceeded
    //                                    kBrakeUsageThreshold; 0 if it never did
    //   brakeOnsetPreview120/300      -- normalized preview observations (slots 12/13) at that frame
    float brakeRequestDominantFraction = 0.0f;
    float throttleRequestMin = 0.0f;
    float throttleRequestMax = 0.0f;
    float brakeRequestMin = 0.0f;
    float brakeRequestMax = 0.0f;
    float brakeOnsetSpeed = 0.0f;
    float brakeOnsetPreview120 = 0.0f;
    float brakeOnsetPreview300 = 0.0f;
};

// Per-evaluation accumulator behind DrivingDiagnosticsSummary. Plain scalar
// inputs (not Car/TrackProgress) so it is trivially, deterministically
// unit-testable -- see verifyDrivingDiagnostics().
class DrivingDiagnostics
{
public:
    // |steeringCmd| above this counts as "saturated" (the command is
    // effectively at full lock).
    static constexpr float kSaturationThreshold = 0.99f;

    // A steering REVERSAL is counted when a frame whose |steeringCmd| exceeds
    // this has the opposite sign to the previous such frame -- i.e. a swing
    // from committed left to committed right (or back). Frames inside the
    // +-threshold band are ignored, so hovering around zero is not counted as
    // a flurry of reversals.
    static constexpr float kReversalThreshold = 0.3f;

    // Physical brake above this counts as "brake in use" (same 0.05 threshold
    // ChampionTelemetry's percent_time_brake_above_005 uses).
    static constexpr float kBrakeUsageThreshold = 0.05f;

    void reset() { *this = DrivingDiagnostics(); }

    // steeringCmd/brakeCmd: the CarInput actually sent to the car this frame.
    // speed (px/s), yawRate (rad/s), frontSlipAngle/frontPeakSlipAngle (rad):
    // the car's state after this frame's physics step. lapCount: completed
    // laps so far. deltaTime: the fixed step (s).
    void update(float steeringCmd, float brakeCmd, float speed, float yawRate, float frontSlipAngle,
                float frontPeakSlipAngle, int lapCount, float deltaTime)
    {
        ++m_frames;
        m_elapsedTime += deltaTime;

        const float absSteering = std::fabs(steeringCmd);
        m_absSteeringSum += absSteering;
        if (absSteering > kSaturationThreshold)
        {
            ++m_saturatedFrames;
        }

        if (absSteering > kReversalThreshold)
        {
            const int sign = (steeringCmd > 0.0f) ? 1 : -1;
            if (m_committedSteeringSign != 0 && sign != m_committedSteeringSign)
            {
                ++m_reversals;
            }
            m_committedSteeringSign = sign;
        }

        m_lateralAccelerationSum += speed * std::fabs(yawRate);
        if (std::fabs(frontSlipAngle) > frontPeakSlipAngle)
        {
            ++m_frontSlipBeyondPeakFrames;
        }
        if (brakeCmd > kBrakeUsageThreshold)
        {
            ++m_brakeFrames;
        }
        if (lapCount >= 1)
        {
            m_lap2PlusSpeedSum += speed;
            ++m_lap2PlusFrames;
        }
    }

    // Called once per frame right after update(), with the controller's
    // [0,1] requests, the physical brake command actually sent, the car speed
    // and the two preview observations (slots 12/13) that drove this frame.
    void recordLongitudinal(float throttleRequest, float brakeRequest, float brakeCmd, float speed,
                            float preview120, float preview300)
    {
        if (m_longitudinalFrames == 0)
        {
            m_throttleRequestMin = m_throttleRequestMax = throttleRequest;
            m_brakeRequestMin = m_brakeRequestMax = brakeRequest;
        }
        else
        {
            m_throttleRequestMin = std::min(m_throttleRequestMin, throttleRequest);
            m_throttleRequestMax = std::max(m_throttleRequestMax, throttleRequest);
            m_brakeRequestMin = std::min(m_brakeRequestMin, brakeRequest);
            m_brakeRequestMax = std::max(m_brakeRequestMax, brakeRequest);
        }
        ++m_longitudinalFrames;
        if (brakeRequest > throttleRequest)
        {
            ++m_brakeDominantFrames;
        }
        if (!m_hasBrakeOnset && brakeCmd > kBrakeUsageThreshold)
        {
            m_hasBrakeOnset = true;
            m_brakeOnsetSpeed = speed;
            m_brakeOnsetPreview120 = preview120;
            m_brakeOnsetPreview300 = preview300;
        }
    }

    // averageAbsSteeringDelta is left 0 here (it lives in FitnessEvaluator);
    // ai::neat::Individual::getDrivingSummary() fills it in.
    DrivingDiagnosticsSummary summary() const
    {
        DrivingDiagnosticsSummary s;
        if (m_frames > 0)
        {
            const float frames = static_cast<float>(m_frames);
            s.steeringSaturationFraction = static_cast<float>(m_saturatedFrames) / frames;
            s.meanAbsSteering = m_absSteeringSum / frames;
            s.meanLateralAcceleration = m_lateralAccelerationSum / frames;
            s.frontSlipBeyondPeakFraction = static_cast<float>(m_frontSlipBeyondPeakFrames) / frames;
            s.physicalBrakeUsageFraction = static_cast<float>(m_brakeFrames) / frames;
        }
        if (m_longitudinalFrames > 0)
        {
            s.brakeRequestDominantFraction = static_cast<float>(m_brakeDominantFrames) / static_cast<float>(m_longitudinalFrames);
            s.throttleRequestMin = m_throttleRequestMin;
            s.throttleRequestMax = m_throttleRequestMax;
            s.brakeRequestMin = m_brakeRequestMin;
            s.brakeRequestMax = m_brakeRequestMax;
        }
        if (m_hasBrakeOnset)
        {
            s.brakeOnsetSpeed = m_brakeOnsetSpeed;
            s.brakeOnsetPreview120 = m_brakeOnsetPreview120;
            s.brakeOnsetPreview300 = m_brakeOnsetPreview300;
        }
        if (m_elapsedTime > 0.0f)
        {
            s.steeringReversalsPerSecond = static_cast<float>(m_reversals) / m_elapsedTime;
        }
        if (m_lap2PlusFrames > 0)
        {
            s.lap2PlusAverageSpeed = m_lap2PlusSpeedSum / static_cast<float>(m_lap2PlusFrames);
        }
        return s;
    }

    int getFrameCount() const { return m_frames; }
    int getReversalCount() const { return m_reversals; }

private:
    int m_frames = 0;
    float m_elapsedTime = 0.0f;

    float m_absSteeringSum = 0.0f;
    int m_saturatedFrames = 0;
    int m_committedSteeringSign = 0; // 0 until the first frame beyond kReversalThreshold
    int m_reversals = 0;

    float m_lateralAccelerationSum = 0.0f;
    int m_frontSlipBeyondPeakFrames = 0;
    int m_brakeFrames = 0;

    float m_lap2PlusSpeedSum = 0.0f;
    int m_lap2PlusFrames = 0;

    int m_longitudinalFrames = 0;
    int m_brakeDominantFrames = 0;
    float m_throttleRequestMin = 0.0f;
    float m_throttleRequestMax = 0.0f;
    float m_brakeRequestMin = 0.0f;
    float m_brakeRequestMax = 0.0f;
    bool m_hasBrakeOnset = false;
    float m_brakeOnsetSpeed = 0.0f;
    float m_brakeOnsetPreview120 = 0.0f;
    float m_brakeOnsetPreview300 = 0.0f;
};

} // namespace ai
