#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "raylib.h"

#include "ai/AIController.h"
#include "ai/FitnessEvaluator.h"
#include "ai/NeuralNetwork.h"
#include "ai/Observation.h"
#include "ai/neat/ConnectionGene.h"
#include "ai/neat/Genome.h"
#include "ai/neat/NodeGene.h"
#include "ai/neat/PhenotypeBuilder.h"
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

// Bottom straight of the oval, where the road band is wide and its tangent
// is horizontal, so heading = 0 (pointing along +x) is track-aligned.
constexpr Vector2 kSpawnPosition = {600.0f, 565.0f};
constexpr float kSpawnHeading = 0.0f;

simulation::TrackDefinition makeTrackDefinition()
{
    simulation::TrackDefinition def;
    def.simWidth = kSimWidth;
    def.simHeight = kSimHeight;
    def.center = {600.0f, 350.0f};
    def.outerRadiusX = 500.0f;
    def.outerRadiusY = 280.0f;
    def.innerRadiusX = 350.0f;
    def.innerRadiusY = 150.0f;
    return def;
}

simulation::CarParams makeCarParams()
{
    return simulation::CarParams{};
}

// How the single car currently receives its CarInput. Manual is the default;
// TAB toggles between the two. Switching modes never resets or rebuilds the
// AI's network -- it only changes where CarInput comes from each frame.
enum class ControlMode
{
    Manual,
    AI
};

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
void verifyTrack(const simulation::Track& track)
{
    const simulation::TrackDefinition& def = track.getDefinition();
    const int cx = static_cast<int>(def.center.x);
    const int cy = static_cast<int>(def.center.y);

    assert(!track.isDrivable(cx, cy) && "center of inner ellipse must be non-drivable");
    assert(track.isDrivable(cx, cy - 200) && "point in road band must be drivable");
    assert(!track.isDrivable(cx, cy - 300) && "point outside outer ellipse must be non-drivable");
    assert(!track.isDrivable(-5, -5) && "negative coordinates must be non-drivable");
    assert(!track.isDrivable(kSimWidth, cy) && "x at/beyond width must be non-drivable");
    assert(!track.isDrivable(cx, kSimHeight) && "y at/beyond height must be non-drivable");

    TraceLog(LOG_INFO, "Track verification: all CPU mask checks passed");
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

    // 6: a sensor directed at a nearby track boundary reports less than maximum distance.
    // At the spawn x, the inner ellipse boundary is ~65px above the car; heading -90deg
    // points the front sensor straight at it, well within the 200px range.
    {
        car.reset(kSpawnPosition, -static_cast<float>(PI) * 0.5f);
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

// Inverse of TrackProgress's angle-to-progress mapping (see TrackProgress.h
// for the forward mapping this undoes): returns a world position on the
// track's mid-band ellipse at the given lap position. Lets tests place the
// car at exact, hand-computed lap positions via Car::reset() instead of
// relying on real driving physics for anything but the one "real driving"
// sanity check below.
Vector2 positionAtLapPosition(const simulation::TrackDefinition& def, float refX, float refY, float lapPos)
{
    const float rawAngle = -lapPos * 2.0f * static_cast<float>(PI);
    return Vector2{def.center.x + std::cos(rawAngle) * refX, def.center.y + std::sin(rawAngle) * refY};
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

} // namespace track_progress_verify

// One-shot, deterministic sanity check of simulation::TrackProgress,
// independent of keyboard/render timing. Runs once at startup. TrackProgress
// never controls the Car -- only Car::reset() (to place the car at precise,
// hand-computed positions) and the Track's own TrackDefinition are used
// here, plus one short real-driving check for direction sanity.
void verifyTrackProgress(const simulation::Track& track)
{
    using track_progress_verify::positionAtLapPosition;
    using track_progress_verify::throwsInvalidArgument;
    constexpr float kEps = 1e-3f;

    const simulation::TrackDefinition& def = track.getDefinition();
    const float refX = (def.outerRadiusX + def.innerRadiusX) * 0.5f;
    const float refY = (def.outerRadiusY + def.innerRadiusY) * 0.5f;

    simulation::Car car(makeCarParams(), track);

    // 1 & 13 (setup half): reset() computes lap position from the car's
    // current position using the exact documented formula, and clears every
    // accumulated field to its baseline.
    {
        car.reset(kSpawnPosition, kSpawnHeading);
        simulation::TrackProgress progress(def);
        progress.reset(car);

        const float dx = kSpawnPosition.x - def.center.x;
        const float dy = kSpawnPosition.y - def.center.y;
        const float rawAngle = std::atan2(dy / refY, dx / refX);
        float expectedLapPosition = -rawAngle / (2.0f * static_cast<float>(PI));
        if (expectedLapPosition < 0.0f)
        {
            expectedLapPosition += 1.0f;
        }

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
        simulation::TrackProgress progress(def);
        car.reset(positionAtLapPosition(def, refX, refY, 0.0f), kSpawnHeading);
        progress.reset(car);

        for (float p = 0.0f; p < 1.0f; p += 0.1f)
        {
            car.reset(positionAtLapPosition(def, refX, refY, p), kSpawnHeading);
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
        simulation::TrackProgress progress(def);
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
        simulation::TrackProgress progress(def);
        car.reset(positionAtLapPosition(def, refX, refY, 0.0f), kSpawnHeading);
        progress.reset(car);

        car.reset(positionAtLapPosition(def, refX, refY, 0.10f), kSpawnHeading);
        progress.update(car);
        const float bestAfterForward = progress.getBestProgress();
        assert(bestAfterForward > 0.09f && "forward synthetic movement must register as progress");

        car.reset(positionAtLapPosition(def, refX, refY, 0.07f), kSpawnHeading);
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
        simulation::TrackProgress progress(def);
        car.reset(positionAtLapPosition(def, refX, refY, 0.0f), kSpawnHeading);
        progress.reset(car);

        const float toSeam[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 0.98f};
        for (float p : toSeam)
        {
            car.reset(positionAtLapPosition(def, refX, refY, p), kSpawnHeading);
            progress.update(car);
        }
        assert(progress.getLapCount() == 0 && "lap must not be counted before crossing the seam");

        // Forward seam crossing: 0.98 -> 0.02, i.e. delta corrects to +0.04.
        car.reset(positionAtLapPosition(def, refX, refY, 0.02f), kSpawnHeading);
        progress.update(car);
        assert(progress.getContinuousProgress() > 1.0f &&
               "a forward seam crossing must push continuous progress past 1.0");
        assert(progress.getLapCount() == 1 && "a valid forward seam crossing must complete lap 1");
        const float bestAfterLap = progress.getBestProgress();

        // Backward seam crossing back across 0.02 -> 0.98 must not grant an
        // additional lap, and must not exceed the existing best.
        car.reset(positionAtLapPosition(def, refX, refY, 0.98f), kSpawnHeading);
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
        simulation::TrackProgress progress(def);
        car.reset(positionAtLapPosition(def, refX, refY, 0.03f), kSpawnHeading);
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
            car.reset(positionAtLapPosition(def, refX, refY, step.targetLapPosition), kSpawnHeading);
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
        car.reset(positionAtLapPosition(def, refX, refY, 0.08f), kSpawnHeading);
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
        car.reset(positionAtLapPosition(def, refX, refY, 0.43f), kSpawnHeading); // 0.08 -> 0.43 is a 0.35 jump, > the plausibility threshold
        progress.update(car);
        assert(std::fabs(progress.getLapPosition() - 0.43f) < kEps &&
               "the raw lap position must still reflect the car's actual (teleported) position");
        assert(progress.getTotalCheckpointsPassed() == totalBeforeJump && progress.getExpectedCheckpoint() == expectedBeforeJump &&
               progress.getLapCount() == lapsBeforeJump &&
               "an implausible jump must not award checkpoints or laps even if it lands at a later angle");

        // Review requirement 4: backward movement (still within the
        // plausible-delta range) must not award checkpoints either.
        car.reset(positionAtLapPosition(def, refX, refY, 0.35f), kSpawnHeading); // 0.43 -> 0.35 is backward
        progress.update(car);
        assert(progress.getTotalCheckpointsPassed() == totalBeforeJump && progress.getExpectedCheckpoint() == expectedBeforeJump &&
               progress.getLapCount() == lapsBeforeJump && "backward movement must not award checkpoints or laps");

        car.reset(kSpawnPosition, kSpawnHeading);
    }

    // Review requirement 7: reset restores expected checkpoint, checkpoint
    // count and lap count correctly for whatever position it is given, not
    // just back to a fixed baseline.
    {
        simulation::TrackProgress progress(def);
        car.reset(positionAtLapPosition(def, refX, refY, 0.55f), kSpawnHeading);
        progress.reset(car);
        assert(progress.getExpectedCheckpoint() == 9 && progress.getTotalCheckpointsPassed() == 0 && progress.getLapCount() == 0 &&
               "reset at lap position 0.55 must expect checkpoint 9 next, with checkpoint/lap counts at zero");
        car.reset(kSpawnPosition, kSpawnHeading);
    }

    // 11: repeated updates at an unchanged position do not change progress.
    {
        simulation::TrackProgress progress(def);
        const Vector2 pos = positionAtLapPosition(def, refX, refY, 0.3f);
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
        simulation::TrackProgress progress(def);
        car.reset(positionAtLapPosition(def, refX, refY, 0.0f), kSpawnHeading);
        progress.reset(car);

        float lastBest = progress.getBestProgress();
        const float waypoints[] = {0.05f, 0.15f, 0.08f, 0.20f, 0.10f, 0.25f, 0.15f, 0.30f};
        for (float p : waypoints)
        {
            car.reset(positionAtLapPosition(def, refX, refY, p), kSpawnHeading);
            progress.update(car);
            assert(progress.getBestProgress() >= lastBest - kEps && "best progress must never decrease");
            lastBest = std::max(lastBest, progress.getBestProgress());
        }
        car.reset(kSpawnPosition, kSpawnHeading);
    }

    // 13: reset clears accumulated state built up from nonzero progress.
    {
        simulation::TrackProgress progress(def);
        car.reset(positionAtLapPosition(def, refX, refY, 0.0f), kSpawnHeading);
        progress.reset(car);
        car.reset(positionAtLapPosition(def, refX, refY, 0.19f), kSpawnHeading);
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

    // 14: TrackProgress derives lap position from its own injected
    // TrackDefinition, not a second hardcoded track shape.
    {
        simulation::TrackDefinition otherDef = def;
        otherDef.center = {def.center.x + 50.0f, def.center.y - 30.0f};

        simulation::TrackProgress progressA(def);
        simulation::TrackProgress progressB(otherDef);

        car.reset(kSpawnPosition, kSpawnHeading);
        progressA.reset(car);
        progressB.reset(car);

        assert(progressA.getLapPosition() != progressB.getLapPosition() &&
               "TrackProgress must derive lap position from its own injected TrackDefinition, not a hardcoded shape");
    }

    // 15: invalid track dimensions/radii are rejected clearly.
    {
        simulation::TrackDefinition badOuter = def;
        badOuter.outerRadiusX = 0.0f;
        assert(throwsInvalidArgument([&]() { simulation::TrackProgress p(badOuter); }) &&
               "a non-positive outer radius must be rejected");

        simulation::TrackDefinition badInner = def;
        badInner.innerRadiusX = badInner.outerRadiusX + 1.0f;
        assert(throwsInvalidArgument([&]() { simulation::TrackProgress p(badInner); }) &&
               "an inner radius not smaller than the outer radius must be rejected");

        simulation::TrackDefinition badSize = def;
        badSize.simWidth = 0;
        assert(throwsInvalidArgument([&]() { simulation::TrackProgress p(badSize); }) &&
               "a non-positive simulation width must be rejected");
    }

    TraceLog(LOG_INFO, "Track progress verification: all deterministic checks passed");
}

// One-shot, deterministic sanity check of ai::FitnessEvaluator, independent
// of keyboard/render timing. Runs once at startup. FitnessEvaluator reads
// only simulation::Car::isAlive() and simulation::TrackProgress's getters --
// no Genome or NeuralNetwork is touched here. Its update() signature has no
// mode parameter at all, so manual and AI control paths need no separate
// fitness logic -- both simply call the same update() with whatever Car
// state resulted from that frame.
void verifyFitnessEvaluator(const simulation::Track& track)
{
    using track_progress_verify::positionAtLapPosition;

    const simulation::TrackDefinition& def = track.getDefinition();
    const float refX = (def.outerRadiusX + def.innerRadiusX) * 0.5f;
    const float refY = (def.outerRadiusY + def.innerRadiusY) * 0.5f;

    simulation::Car car(makeCarParams(), track);

    // 16: reset() produces exactly zero fitness/elapsed time and a fresh,
    // unfinished evaluation.
    {
        ai::FitnessEvaluator evaluator;
        evaluator.reset();
        assert(evaluator.getFitness() == 0.0f && evaluator.getElapsedTime() == 0.0f &&
               !evaluator.isEvaluationFinished() && evaluator.getFinishReason() == ai::EvaluationFinishReason::None &&
               "reset must produce zero fitness/elapsed time and an unfinished evaluation");
    }

    // 17: forward progress increases fitness.
    {
        car.reset(positionAtLapPosition(def, refX, refY, 0.0f), kSpawnHeading);
        simulation::TrackProgress progress(def);
        progress.reset(car);
        ai::FitnessEvaluator evaluator;
        evaluator.reset();

        evaluator.update(car, progress, kSimulationDt);
        const float fitnessBefore = evaluator.getFitness();

        car.reset(positionAtLapPosition(def, refX, refY, 0.10f), kSpawnHeading);
        progress.update(car);
        evaluator.update(car, progress, kSimulationDt);
        assert(evaluator.getFitness() > fitnessBefore && "forward progress must increase fitness");
    }

    // 18: backward movement must not increase the progress-derived part of
    // fitness (best progress, checkpoints, laps). Survival time still ticks
    // up regardless of movement direction, so this checks the
    // progress-derived TrackProgress state directly rather than raw
    // getFitness(), which also includes that small survival term.
    {
        car.reset(positionAtLapPosition(def, refX, refY, 0.0f), kSpawnHeading);
        simulation::TrackProgress progress(def);
        progress.reset(car);
        ai::FitnessEvaluator evaluator;
        evaluator.reset();

        car.reset(positionAtLapPosition(def, refX, refY, 0.15f), kSpawnHeading);
        progress.update(car);
        evaluator.update(car, progress, kSimulationDt);
        const float bestProgressAfterForward = progress.getBestProgress();
        const int checkpointsAfterForward = progress.getTotalCheckpointsPassed();

        car.reset(positionAtLapPosition(def, refX, refY, 0.05f), kSpawnHeading);
        progress.update(car);
        evaluator.update(car, progress, kSimulationDt);
        assert(progress.getBestProgress() == bestProgressAfterForward &&
               progress.getTotalCheckpointsPassed() == checkpointsAfterForward &&
               "backward movement must not increase the progress-derived part of fitness");
    }

    // 19: passing a checkpoint increases fitness.
    {
        car.reset(positionAtLapPosition(def, refX, refY, 0.0f), kSpawnHeading);
        simulation::TrackProgress progress(def);
        progress.reset(car);
        ai::FitnessEvaluator evaluator;
        evaluator.reset();

        evaluator.update(car, progress, kSimulationDt);
        const float fitnessAtStart = evaluator.getFitness();

        car.reset(positionAtLapPosition(def, refX, refY, 1.0f / static_cast<float>(simulation::TrackProgress::kCheckpointCount)),
                  kSpawnHeading);
        progress.update(car);
        evaluator.update(car, progress, kSimulationDt);
        assert(progress.getTotalCheckpointsPassed() >= 1 && "the setup must actually pass at least one checkpoint");
        assert(evaluator.getFitness() > fitnessAtStart && "passing a checkpoint must increase fitness");
    }

    // 20: completing a lap increases fitness with a distinct lap bonus on
    // top of the progress reward already earned.
    {
        car.reset(positionAtLapPosition(def, refX, refY, 0.0f), kSpawnHeading);
        simulation::TrackProgress progress(def);
        progress.reset(car);
        ai::FitnessEvaluator evaluator;
        evaluator.reset();

        const float toAlmostFull[] = {0.15f, 0.30f, 0.45f, 0.60f, 0.75f, 0.90f, 0.95f};
        for (float p : toAlmostFull)
        {
            car.reset(positionAtLapPosition(def, refX, refY, p), kSpawnHeading);
            progress.update(car);
            evaluator.update(car, progress, kSimulationDt);
        }
        const float fitnessBeforeLap = evaluator.getFitness();
        assert(progress.getLapCount() == 0 && "setup must not have completed a lap yet");

        car.reset(positionAtLapPosition(def, refX, refY, 0.05f), kSpawnHeading);
        progress.update(car);
        evaluator.update(car, progress, kSimulationDt);
        assert(progress.getLapCount() == 1 && "the final step must complete exactly one lap");
        assert(evaluator.getFitness() > fitnessBeforeLap && "completing a lap must increase fitness");
        car.reset(kSpawnPosition, kSpawnHeading);
    }

    // 21: survival reward alone stays small relative to progress's scale
    // (1000 points/lap) even after several seconds with no movement.
    {
        car.reset(kSpawnPosition, kSpawnHeading);
        simulation::TrackProgress progress(def);
        progress.reset(car);
        ai::FitnessEvaluator evaluator;
        evaluator.reset();

        for (int i = 0; i < 240 && !evaluator.isEvaluationFinished(); ++i) // 4s of no movement, under the 5s timeout
        {
            progress.update(car);
            evaluator.update(car, progress, kSimulationDt);
        }
        assert(!evaluator.isEvaluationFinished() && "4 seconds of no movement must stay under the no-progress timeout");
        assert(evaluator.getFitness() < 10.0f &&
               "survival-only fitness must stay small relative to the progress scale (1000 points/lap)");
    }

    // 22: a collided (dead) car ends the evaluation with Collision.
    {
        car.reset(kSpawnPosition, kSpawnHeading);
        simulation::TrackProgress progress(def);
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
        car.reset(kSpawnPosition, kSpawnHeading);
    }

    // 23: reaching the maximum evaluation time ends it with TimeLimit --
    // progress is nudged forward every simulated second so the no-progress
    // timeout cannot pre-empt it.
    {
        car.reset(positionAtLapPosition(def, refX, refY, 0.0f), kSpawnHeading);
        simulation::TrackProgress progress(def);
        progress.reset(car);
        ai::FitnessEvaluator evaluator;
        evaluator.reset();

        float p = 0.0f;
        for (int second = 0; second < 61 && !evaluator.isEvaluationFinished(); ++second)
        {
            p += 0.01f;
            car.reset(positionAtLapPosition(def, refX, refY, std::fmod(p, 1.0f)), kSpawnHeading);
            progress.update(car);
            evaluator.update(car, progress, 1.0f);
        }
        assert(evaluator.isEvaluationFinished() &&
               evaluator.getFinishReason() == ai::EvaluationFinishReason::TimeLimit &&
               "reaching the maximum evaluation time must end the evaluation with TimeLimit");
        assert(evaluator.getElapsedTime() >= 60.0f && "elapsed time at TimeLimit must reach the configured maximum");
        car.reset(kSpawnPosition, kSpawnHeading);
    }

    // 24: standing still for the no-progress timeout ends the evaluation
    // with NoProgress.
    {
        car.reset(kSpawnPosition, kSpawnHeading);
        simulation::TrackProgress progress(def);
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
        car.reset(kSpawnPosition, kSpawnHeading);
    }

    // 25: meaningful progress resets the no-progress timer.
    {
        car.reset(positionAtLapPosition(def, refX, refY, 0.0f), kSpawnHeading);
        simulation::TrackProgress progress(def);
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
        car.reset(positionAtLapPosition(def, refX, refY, 0.05f), kSpawnHeading);
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

    // 26: a finished evaluation does not continue changing fitness, elapsed
    // time, or finish reason on further update() calls.
    {
        car.reset(kSpawnPosition, kSpawnHeading);
        simulation::TrackProgress progress(def);
        progress.reset(car);
        ai::FitnessEvaluator evaluator;
        evaluator.reset();

        for (int i = 0; i < 400 && !evaluator.isEvaluationFinished(); ++i)
        {
            progress.update(car);
            evaluator.update(car, progress, kSimulationDt);
        }
        assert(evaluator.isEvaluationFinished() && "setup must have already finished the evaluation");

        const float fitnessAtFinish = evaluator.getFitness();
        const float elapsedAtFinish = evaluator.getElapsedTime();
        const ai::EvaluationFinishReason reasonAtFinish = evaluator.getFinishReason();

        car.reset(positionAtLapPosition(def, refX, refY, 0.5f), kSpawnHeading); // would otherwise be a big progress jump
        progress.update(car);
        evaluator.update(car, progress, 10.0f); // would otherwise add a large survival reward and elapsed time

        assert(evaluator.getFitness() == fitnessAtFinish && evaluator.getElapsedTime() == elapsedAtFinish &&
               evaluator.getFinishReason() == reasonAtFinish &&
               "a finished evaluation must not change fitness, elapsed time, or finish reason on further updates");
        car.reset(kSpawnPosition, kSpawnHeading);
    }

    // 27: fitness is deterministic -- two independently constructed
    // evaluators driven through an identical sequence of states produce
    // identical fitness.
    {
        auto runScenario = [&](simulation::Car& localCar) -> float
        {
            localCar.reset(positionAtLapPosition(def, refX, refY, 0.0f), kSpawnHeading);
            simulation::TrackProgress localProgress(def);
            localProgress.reset(localCar);
            ai::FitnessEvaluator localEvaluator;
            localEvaluator.reset();

            const float waypoints[] = {0.05f, 0.12f, 0.20f, 0.30f};
            for (float p : waypoints)
            {
                localCar.reset(positionAtLapPosition(def, refX, refY, p), kSpawnHeading);
                localProgress.update(localCar);
                localEvaluator.update(localCar, localProgress, kSimulationDt);
            }
            return localEvaluator.getFitness();
        };

        simulation::Car carA(makeCarParams(), track);
        simulation::Car carB(makeCarParams(), track);
        const float fitnessA = runScenario(carA);
        const float fitnessB = runScenario(carB);
        assert(fitnessA == fitnessB && "identical state sequences must produce identical fitness");
    }

    // 28: reset() permits a fresh evaluation after a finished one (the same
    // effect the R key has in main()).
    {
        car.reset(kSpawnPosition, kSpawnHeading);
        simulation::TrackProgress progress(def);
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

        car.reset(positionAtLapPosition(def, refX, refY, 0.05f), kSpawnHeading);
        progress.update(car);
        evaluator.update(car, progress, kSimulationDt);
        assert(evaluator.getFitness() > 0.0f && "the fresh evaluation after reset must respond normally to new progress");
        car.reset(kSpawnPosition, kSpawnHeading);
    }

    TraceLog(LOG_INFO, "Fitness evaluator verification: all deterministic checks passed");
}

simulation::CarInput readInput()
{
    simulation::CarInput input;

    if (IsKeyDown(KEY_W) || IsKeyDown(KEY_UP))
    {
        input.throttle = 1.0f;
    }

    if (IsKeyDown(KEY_A) || IsKeyDown(KEY_LEFT))
    {
        input.steering -= 1.0f;
    }

    if (IsKeyDown(KEY_D) || IsKeyDown(KEY_RIGHT))
    {
        input.steering += 1.0f;
    }

    return input;
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

void drawCar(const simulation::Car& car)
{
    const std::array<Vector2, 4> corners = car.getCorners();
    const Color color = car.isAlive() ? Color{40, 180, 255, 255} : Color{70, 70, 70, 255};

    DrawTriangleFan(corners.data(), static_cast<int>(corners.size()), color);

    // Velocity vector debug overlay, scaled down so it stays on-screen.
    const Vector2 position = car.getPosition();
    const Vector2 velocity = car.getVelocity();
    const Vector2 tip = {position.x + velocity.x * 0.25f, position.y + velocity.y * 0.25f};
    DrawLineEx(position, tip, 2.0f, YELLOW);

    // Sensor rays: only meaningful while the car is alive and still moving.
    if (car.isAlive())
    {
        const Vector2 origin = car.getSensorOrigin();
        for (const simulation::SensorReading& sensor : car.getSensors())
        {
            DrawLineEx(origin, sensor.endPoint, 1.5f, Color{80, 255, 120, 255});
            DrawCircleV(sensor.endPoint, 3.0f, Color{255, 90, 40, 255});
        }
    }
}

void drawPanel(const simulation::Car& car, const simulation::CarInput& input, ControlMode mode,
               const ai::AIController& controller, const simulation::TrackProgress& progress,
               const ai::FitnessEvaluator& evaluator)
{
    DrawRectangle(kSimWidth, 0, kPanelWidth, kScreenHeight, Color{30, 30, 30, 255});

    const int x = kSimWidth + 20;
    int y = 20;
    const int lineHeight = 22;

    DrawText("STAGE 8 - PROGRESS & FITNESS", x, y, 20, RAYWHITE);
    y += lineHeight;

    char line[128];

    const bool aiMode = (mode == ControlMode::AI);
    std::snprintf(line, sizeof(line), "CONTROL MODE: %s", aiMode ? "AI" : "MANUAL");
    DrawText(line, x, y, 18, aiMode ? SKYBLUE : RAYWHITE);
    y += lineHeight * 2;

    DrawText("PROGRESS", x, y, 18, YELLOW);
    y += lineHeight;
    std::snprintf(line, sizeof(line), "Lap pos: %.3f  Cont: %.3f  Best: %.3f",
                  static_cast<double>(progress.getLapPosition()), static_cast<double>(progress.getContinuousProgress()),
                  static_cast<double>(progress.getBestProgress()));
    DrawText(line, x, y, 16, LIGHTGRAY);
    y += lineHeight;
    std::snprintf(line, sizeof(line), "Laps: %d   Checkpoints: %d (next: %d/%d)", progress.getLapCount(),
                  progress.getTotalCheckpointsPassed(), progress.getExpectedCheckpoint(),
                  simulation::TrackProgress::kCheckpointCount);
    DrawText(line, x, y, 16, LIGHTGRAY);
    y += lineHeight * 2;

    DrawText("FITNESS", x, y, 18, YELLOW);
    y += lineHeight;
    std::snprintf(line, sizeof(line), "Fitness: %.1f   Time: %.1fs", static_cast<double>(evaluator.getFitness()),
                  static_cast<double>(evaluator.getElapsedTime()));
    DrawText(line, x, y, 16, LIGHTGRAY);
    y += lineHeight;
    std::snprintf(line, sizeof(line), "State: %s (%s)", evaluator.isEvaluationFinished() ? "FINISHED" : "RUNNING",
                  finishReasonLabel(evaluator.getFinishReason()));
    DrawText(line, x, y, 16, evaluator.isEvaluationFinished() ? RED : GREEN);
    y += lineHeight * 2;

    std::snprintf(line, sizeof(line), "Speed: %.0f  Fwd: %.0f  Lat: %.0f px/s", static_cast<double>(car.getSpeed()),
                  static_cast<double>(car.getForwardVelocity()), static_cast<double>(car.getLateralVelocity()));
    DrawText(line, x, y, 16, RAYWHITE);
    y += lineHeight;

    std::snprintf(line, sizeof(line), "Slip: %.1f deg", static_cast<double>(car.getSlipAngle() * RAD2DEG));
    DrawText(line, x, y, 16, RAYWHITE);
    y += lineHeight;

    std::snprintf(line, sizeof(line), "Throttle: %.2f  Steering: %.2f", static_cast<double>(input.throttle),
                  static_cast<double>(input.steering));
    DrawText(line, x, y, 16, RAYWHITE);
    y += lineHeight * 2;

    if (aiMode)
    {
        DrawText("AI outputs:", x, y, 16, SKYBLUE);
        y += lineHeight;

        std::snprintf(line, sizeof(line), "raw    steer %+.2f throttle %+.2f",
                      static_cast<double>(controller.getRawSteeringOutput()),
                      static_cast<double>(controller.getRawThrottleOutput()));
        DrawText(line, x, y, 16, LIGHTGRAY);
        y += lineHeight;

        std::snprintf(line, sizeof(line), "mapped steer %+.2f throttle %+.2f", static_cast<double>(input.steering),
                      static_cast<double>(input.throttle));
        DrawText(line, x, y, 16, LIGHTGRAY);
        y += lineHeight * 2;
    }

    std::snprintf(line, sizeof(line), "Sensors (%d, obs=%d):", simulation::Car::kSensorCount, ai::kObservationSize);
    DrawText(line, x, y, 16, RAYWHITE);
    y += lineHeight;
    for (int i = 0; i < simulation::Car::kSensorCount; ++i)
    {
        const simulation::SensorReading& s = car.getSensors()[i];
        std::snprintf(line, sizeof(line), "%+4.0f deg: %.3f (%.1f px)",
                      static_cast<double>(simulation::Car::kSensorAngleDegrees[i]),
                      static_cast<double>(s.normalizedDistance), static_cast<double>(s.distance));
        DrawText(line, x, y, 16, LIGHTGRAY);
        y += lineHeight;
    }
    y += lineHeight;

    DrawText(car.isAlive() ? "ALIVE" : "CRASHED", x, y, 20, car.isAlive() ? GREEN : RED);
    y += lineHeight * 2;

    DrawText("Controls:", x, y, 18, RAYWHITE);
    y += lineHeight;
    DrawText("W/Up throttle   A/D or Left/Right steer", x, y, 16, LIGHTGRAY);
    y += lineHeight;
    DrawText("TAB toggle manual/AI   R reset", x, y, 16, LIGHTGRAY);
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
    verifyAIController(track);
    verifyTrackProgress(track);
    verifyFitnessEvaluator(track);

    const simulation::TrackDefinition& def = track.getDefinition();

    simulation::Car car(makeCarParams(), track);
    car.reset(kSpawnPosition, kSpawnHeading);

    simulation::TrackProgress trackProgress(track.getDefinition());
    trackProgress.reset(car);

    ai::FitnessEvaluator fitnessEvaluator;
    fitnessEvaluator.reset();

    // One demonstration AIController, built once from one hand-built,
    // deterministic Genome. Toggling control modes never rebuilds or resets
    // this network -- see createDemonstrationGenome() for its fixed weights.
    ai::AIController aiController(ai::neat::buildPhenotype(createDemonstrationGenome()));
    ControlMode controlMode = ControlMode::Manual;

    // Reflects the CarInput actually applied on the most recent simulation
    // step. Declared outside the loop so it keeps showing that final input
    // once the evaluation finishes and the sim step below stops running.
    simulation::CarInput input;

    while (!WindowShouldClose())
    {
        if (IsKeyPressed(KEY_TAB))
        {
            controlMode = (controlMode == ControlMode::Manual) ? ControlMode::AI : ControlMode::Manual;
        }

        if (IsKeyPressed(KEY_R))
        {
            car.reset(kSpawnPosition, kSpawnHeading);
            trackProgress.reset(car);
            fitnessEvaluator.reset();
        }

        // Once the evaluation has finished, the simulation step is skipped
        // entirely: no further input is applied, the car/progress/fitness
        // state stays exactly as it was at the moment of finishing, and only
        // R (above) starts a fresh evaluation.
        if (!fitnessEvaluator.isEvaluationFinished())
        {
            // Manual mode reads the keyboard directly; AI mode obtains
            // CarInput only from AIController, which itself reads the Car's
            // own already-refreshed sensors/state -- neither mode
            // duplicates sensor casting, and keyboard input never reaches
            // the car while in AI mode. Both modes feed the exact same
            // TrackProgress/FitnessEvaluator update calls below.
            input = (controlMode == ControlMode::Manual) ? readInput() : aiController.update(car);
            car.update(input, kSimulationDt);
            trackProgress.update(car);
            fitnessEvaluator.update(car, trackProgress, kSimulationDt);
        }

        BeginDrawing();
        ClearBackground(BLACK);

        DrawRectangle(0, 0, kSimWidth, kSimHeight, BLACK);

        DrawEllipse(static_cast<int>(def.center.x), static_cast<int>(def.center.y),
                    def.outerRadiusX, def.outerRadiusY, GRAY);
        DrawEllipse(static_cast<int>(def.center.x), static_cast<int>(def.center.y),
                    def.innerRadiusX, def.innerRadiusY, BLACK);

        drawCar(car);
        drawPanel(car, input, controlMode, aiController, trackProgress, fitnessEvaluator);

        EndDrawing();
    }

    CloseWindow();

    return 0;
}
