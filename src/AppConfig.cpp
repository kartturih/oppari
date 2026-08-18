#include "AppConfig.h"

#include "ai/NeuralNetwork.h"
#include "ai/neat/ConnectionGene.h"
#include "ai/neat/NodeGene.h"

namespace app
{

std::string assetPath(const std::string& relativePath)
{
    return std::string(OPPARI_ASSETS_DIR) + "/" + relativePath;
}

simulation::TrackDefinition makeTrackDefinition()
{
    return simulation::createExtremeTrackDefinition(kSimWidth, kSimHeight,
                                                      assetPath("tracks/extreme/track_visual.png"),
                                                      assetPath("tracks/extreme/track_mask.png"));
}

namespace
{

// Spawn pose derived from the Track itself, computed once from a
// throwaway Track so every verify*() function and every Population shares
// the same deterministic pose.
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

} // namespace

const Vector2 kSpawnPosition = kSpawnPose.position;
const float kSpawnHeading = kSpawnPose.heading;

simulation::CarParams makeCarParams()
{
    return simulation::CarParams{};
}

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
    constexpr NodeId kThrottleOutputId = 101; // middle Output ID -> output slot 1
    constexpr NodeId kBrakeOutputId = 102;    // higher Output ID -> output slot 2

    Genome genome;
    for (int i = 0; i < ai::NeuralNetwork::kInputCount; ++i)
    {
        genome.addNode(NodeGene{i, NodeType::Input});
    }
    genome.addNode(NodeGene{kBiasId, NodeType::Bias});
    genome.addNode(NodeGene{kSteeringOutputId, NodeType::Output});
    genome.addNode(NodeGene{kThrottleOutputId, NodeType::Output});
    genome.addNode(NodeGene{kBrakeOutputId, NodeType::Output});

    int innovation = 0;
    genome.addConnection(ConnectionGene{kSensorLeft60, kSteeringOutputId, -0.5f, true, innovation++});
    genome.addConnection(ConnectionGene{kSensorLeft30, kSteeringOutputId, -0.3f, true, innovation++});
    genome.addConnection(ConnectionGene{kSensorRight30, kSteeringOutputId, 0.3f, true, innovation++});
    genome.addConnection(ConnectionGene{kSensorRight60, kSteeringOutputId, 0.5f, true, innovation++});
    genome.addConnection(ConnectionGene{kBiasId, kThrottleOutputId, 0.6f, true, innovation++});
    genome.addConnection(ConnectionGene{kSensorCenter, kThrottleOutputId, 0.4f, true, innovation++});
    // Bias -> Brake, strongly negative: with no other wiring, raw brake
    // output is tanh(-5.0) =~ -1.0, mapping to =~0 -- brakes start off, so
    // generation 0 can actually drive; NEAT discovers real braking points
    // via mutation from here, the same way it discovers everything else.
    genome.addConnection(ConnectionGene{kBiasId, kBrakeOutputId, -5.0f, true, innovation++});

    return genome;
}

} // namespace app
