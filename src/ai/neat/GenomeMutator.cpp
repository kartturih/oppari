#include "ai/neat/GenomeMutator.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace ai::neat
{

namespace
{

bool isProbability(float value)
{
    return std::isfinite(value) && value >= 0.0f && value <= 1.0f;
}

void validateWeightMutationConfig(const MutationConfig& config)
{
    if (!isProbability(config.weightMutationProbability))
    {
        throw std::invalid_argument("MutationConfig: weightMutationProbability must be finite and within [0,1]");
    }
    if (!isProbability(config.weightPerturbProbability))
    {
        throw std::invalid_argument("MutationConfig: weightPerturbProbability must be finite and within [0,1]");
    }
    if (!std::isfinite(config.perturbStrength) || config.perturbStrength < 0.0f)
    {
        throw std::invalid_argument("MutationConfig: perturbStrength must be finite and non-negative");
    }
    if (!std::isfinite(config.replacementWeightMin) || !std::isfinite(config.replacementWeightMax))
    {
        throw std::invalid_argument("MutationConfig: replacement weight bounds must be finite");
    }
    if (config.replacementWeightMin > config.replacementWeightMax)
    {
        throw std::invalid_argument("MutationConfig: replacementWeightMin must not exceed replacementWeightMax");
    }
}

// Deliberately independent of validateWeightMutationConfig(): mutateWeights()
// must keep behaving exactly as it did in Stage 9A, so it must not start
// rejecting configs over add-connection fields it never reads, and vice
// versa.
void validateAddConnectionConfig(const MutationConfig& config)
{
    if (!isProbability(config.addConnectionProbability))
    {
        throw std::invalid_argument("MutationConfig: addConnectionProbability must be finite and within [0,1]");
    }
    if (!std::isfinite(config.newConnectionWeightMin) || !std::isfinite(config.newConnectionWeightMax))
    {
        throw std::invalid_argument("MutationConfig: new connection weight bounds must be finite");
    }
    if (config.newConnectionWeightMin > config.newConnectionWeightMax)
    {
        throw std::invalid_argument("MutationConfig: newConnectionWeightMin must not exceed newConnectionWeightMax");
    }
    if (config.addConnectionMaxAttempts <= 0)
    {
        throw std::invalid_argument("MutationConfig: addConnectionMaxAttempts must be positive");
    }
}

bool isValidSourceType(NodeType type)
{
    return type == NodeType::Input || type == NodeType::Bias || type == NodeType::Hidden;
}

bool isValidTargetType(NodeType type)
{
    return type == NodeType::Hidden || type == NodeType::Output;
}

// True if 'to' is reachable from 'from' by following only *enabled*
// connections. Plain graph-traversal (BFS/DFS, order irrelevant) over
// genome's actual edges -- deliberately never consults node IDs or the
// order nodes/connections happen to be stored in, since hidden-node IDs do
// not necessarily represent topological order.
bool hasEnabledPath(const Genome& genome, NodeId from, NodeId to)
{
    std::vector<NodeId> frontier{from};
    std::unordered_set<NodeId> visited{from};

    while (!frontier.empty())
    {
        const NodeId current = frontier.back();
        frontier.pop_back();

        if (current == to)
        {
            return true;
        }

        for (const ConnectionGene& connection : genome.connections())
        {
            if (connection.isEnabled() && connection.getSourceId() == current)
            {
                const NodeId next = connection.getTargetId();
                if (visited.insert(next).second)
                {
                    frontier.push_back(next);
                }
            }
        }
    }
    return false;
}

// A (source, target) pair -- already known to have valid source/target node
// types -- is a valid new-connection candidate when it is not a
// self-connection, no connection (enabled *or* disabled) already occupies
// that exact directed pair, and adding it would not create a cycle among
// enabled connections.
bool isValidCandidate(const Genome& genome, NodeId source, NodeId target)
{
    if (source == target)
    {
        return false;
    }
    if (genome.hasConnection(source, target))
    {
        return false;
    }
    // Adding source -> target closes a cycle exactly when target can
    // already reach source through enabled connections.
    return !hasEnabledPath(genome, target, source);
}

} // namespace

GenomeMutator::GenomeMutator(std::uint32_t seed) : m_rng(seed)
{
}

void GenomeMutator::mutateWeights(Genome& genome, const MutationConfig& config)
{
    validateWeightMutationConfig(config);

    // Four independent draws per candidate connection (selection,
    // perturb-vs-replace choice, and one of two magnitude draws), all from
    // the same owned generator, in a fixed order -- this is what makes the
    // result a pure, repeatable function of (seed, genome, config).
    std::uniform_real_distribution<float> selectForMutation(0.0f, 1.0f);
    std::uniform_real_distribution<float> selectPerturbOverReplace(0.0f, 1.0f);
    std::uniform_real_distribution<float> perturbDelta(-config.perturbStrength, config.perturbStrength);
    std::uniform_real_distribution<float> replacementWeight(config.replacementWeightMin, config.replacementWeightMax);

    for (ConnectionGene& connection : genome.mutableConnections())
    {
        if (selectForMutation(m_rng) >= config.weightMutationProbability)
        {
            continue;
        }

        if (selectPerturbOverReplace(m_rng) < config.weightPerturbProbability)
        {
            connection.setWeight(connection.getWeight() + perturbDelta(m_rng));
        }
        else
        {
            connection.setWeight(replacementWeight(m_rng));
        }
    }
}

bool GenomeMutator::mutateAddConnection(Genome& genome, InnovationTracker& innovationTracker, const MutationConfig& config)
{
    validateAddConnectionConfig(config);

    std::uniform_real_distribution<float> selectForMutation(0.0f, 1.0f);
    if (selectForMutation(m_rng) >= config.addConnectionProbability)
    {
        return false;
    }

    std::vector<NodeId> sourceCandidates;
    std::vector<NodeId> targetCandidates;
    for (const NodeGene& node : genome.nodes())
    {
        if (isValidSourceType(node.getType()))
        {
            sourceCandidates.push_back(node.getId());
        }
        if (isValidTargetType(node.getType()))
        {
            targetCandidates.push_back(node.getId());
        }
    }

    if (sourceCandidates.empty() || targetCandidates.empty())
    {
        return false;
    }

    std::uniform_int_distribution<std::size_t> pickSource(0, sourceCandidates.size() - 1);
    std::uniform_int_distribution<std::size_t> pickTarget(0, targetCandidates.size() - 1);

    NodeId chosenSource = 0;
    NodeId chosenTarget = 0;
    bool found = false;

    for (int attempt = 0; attempt < config.addConnectionMaxAttempts; ++attempt)
    {
        const NodeId candidateSource = sourceCandidates[pickSource(m_rng)];
        const NodeId candidateTarget = targetCandidates[pickTarget(m_rng)];
        if (isValidCandidate(genome, candidateSource, candidateTarget))
        {
            chosenSource = candidateSource;
            chosenTarget = candidateTarget;
            found = true;
            break;
        }
    }

    if (!found)
    {
        // Deterministic exhaustive fallback: source ascending by ID, then
        // target ascending by ID. Touches no RNG state, so it never affects
        // determinism -- it only ever runs after the fixed number of random
        // draws above, regardless of whether it finds anything.
        std::vector<NodeId> sortedSources = sourceCandidates;
        std::vector<NodeId> sortedTargets = targetCandidates;
        std::sort(sortedSources.begin(), sortedSources.end());
        std::sort(sortedTargets.begin(), sortedTargets.end());

        for (NodeId source : sortedSources)
        {
            for (NodeId target : sortedTargets)
            {
                if (isValidCandidate(genome, source, target))
                {
                    chosenSource = source;
                    chosenTarget = target;
                    found = true;
                    break;
                }
            }
            if (found)
            {
                break;
            }
        }
    }

    if (!found)
    {
        return false;
    }

    // Tracker state only changes from this point on -- every path above
    // that returns false leaves innovationTracker untouched.
    const InnovationNumber innovation = innovationTracker.getConnectionInnovation(chosenSource, chosenTarget);

    std::uniform_real_distribution<float> newWeight(config.newConnectionWeightMin, config.newConnectionWeightMax);
    const float weight = newWeight(m_rng);

    genome.addConnection(ConnectionGene(chosenSource, chosenTarget, weight, true, innovation));
    return true;
}

} // namespace ai::neat
