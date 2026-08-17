#include "ai/neat/GenomeCrossover.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "ai/neat/PhenotypeBuilder.h"

namespace ai::neat
{

namespace
{

bool isProbability(float value)
{
    return std::isfinite(value) && value >= 0.0f && value <= 1.0f;
}

void validateCrossoverConfig(const CrossoverConfig& config)
{
    if (!isProbability(config.matchingGeneChooseParentAProbability))
    {
        throw std::invalid_argument(
            "CrossoverConfig: matchingGeneChooseParentAProbability must be finite and within [0,1]");
    }
    if (!isProbability(config.disabledGeneRemainDisabledProbability))
    {
        throw std::invalid_argument(
            "CrossoverConfig: disabledGeneRemainDisabledProbability must be finite and within [0,1]");
    }
}

void validateFitness(float fitnessA, float fitnessB)
{
    if (!std::isfinite(fitnessA) || !std::isfinite(fitnessB))
    {
        throw std::invalid_argument("GenomeCrossover: fitness values must be finite");
    }
}

// Genome::validate() doesn't check innovation-number uniqueness, so this
// must, before innovation numbers can be trusted as an alignment key.
std::map<InnovationNumber, const ConnectionGene*> buildInnovationMap(const Genome& genome, bool isParentA)
{
    std::map<InnovationNumber, const ConnectionGene*> result;
    for (const ConnectionGene& connection : genome.connections())
    {
        const auto insertion = result.emplace(connection.getInnovationNumber(), &connection);
        if (!insertion.second)
        {
            throw std::invalid_argument(isParentA
                                             ? "GenomeCrossover: parent A contains duplicate connection innovation numbers"
                                             : "GenomeCrossover: parent B contains duplicate connection innovation numbers");
        }
    }
    return result;
}

// Throws if the same NodeId has a different NodeType in each genome.
void validateNodeTypeConsistency(const Genome& a, const Genome& b)
{
    for (const NodeGene& nodeInA : a.nodes())
    {
        const NodeGene* nodeInB = b.findNode(nodeInA.getId());
        if (nodeInB != nullptr && nodeInB->getType() != nodeInA.getType())
        {
            throw std::invalid_argument("GenomeCrossover: same NodeId has conflicting NodeType between parents");
        }
    }
}

const NodeGene* findNodeInEitherParent(NodeId id, const Genome& parentA, const Genome& parentB)
{
    const NodeGene* found = parentA.findNode(id);
    return found != nullptr ? found : parentB.findNode(id);
}

// Same as GenomeMutator's hasEnabledPath(), but over a staged vector (the
// child isn't a Genome yet during equal-fitness candidate inspection).
bool hasEnabledPathAmong(const std::vector<ConnectionGene>& connections, NodeId from, NodeId to)
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

        for (const ConnectionGene& connection : connections)
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

bool wouldCreateDuplicateDirectedConnection(const std::vector<ConnectionGene>& existing, const ConnectionGene& candidate)
{
    for (const ConnectionGene& connection : existing)
    {
        if (connection.getSourceId() == candidate.getSourceId() && connection.getTargetId() == candidate.getTargetId())
        {
            return true;
        }
    }
    return false;
}

// True if candidate is enabled and its target can already reach its source.
bool wouldCreateEnabledCycle(const std::vector<ConnectionGene>& existing, const ConnectionGene& candidate)
{
    if (!candidate.isEnabled())
    {
        return false;
    }
    return hasEnabledPathAmong(existing, candidate.getTargetId(), candidate.getSourceId());
}

// A connection gene present in both parents with the same innovation number.
struct MatchingPair
{
    InnovationNumber innovation;
    const ConnectionGene* a;
    const ConnectionGene* b;
};

} // namespace

GenomeCrossover::GenomeCrossover(std::uint32_t seed) : m_rng(seed)
{
}

Genome GenomeCrossover::crossover(const Genome& parentA, float fitnessA, const Genome& parentB, float fitnessB,
                                   const CrossoverConfig& config)
{
    validateCrossoverConfig(config);
    validateFitness(fitnessA, fitnessB);

    parentA.validate();
    parentB.validate();

    const std::map<InnovationNumber, const ConnectionGene*> mapA = buildInnovationMap(parentA, /*isParentA=*/true);
    const std::map<InnovationNumber, const ConnectionGene*> mapB = buildInnovationMap(parentB, /*isParentA=*/false);

    validateNodeTypeConsistency(parentA, parentB);

    // Align by innovation number, classifying matching/non-matching and
    // checking endpoint agreement -- all before any RNG draw.
    std::vector<MatchingPair> matchingPairs;
    std::vector<const ConnectionGene*> aOnly;
    std::vector<const ConnectionGene*> bOnly;

    for (const auto& [innovation, connA] : mapA)
    {
        const auto itB = mapB.find(innovation);
        if (itB == mapB.end())
        {
            aOnly.push_back(connA);
            continue;
        }

        const ConnectionGene* connB = itB->second;
        if (connA->getSourceId() != connB->getSourceId() || connA->getTargetId() != connB->getTargetId())
        {
            throw std::invalid_argument("GenomeCrossover: matching innovation has conflicting endpoints between parents");
        }
        matchingPairs.push_back(MatchingPair{innovation, connA, connB});
    }
    for (const auto& [innovation, connB] : mapB)
    {
        if (mapA.find(innovation) == mapA.end())
        {
            bOnly.push_back(connB);
        }
    }

    std::vector<ConnectionGene> childConnections;
    childConnections.reserve(matchingPairs.size() + aOnly.size() + bOnly.size());

    std::uniform_real_distribution<float> draw01(0.0f, 1.0f);

    for (const MatchingPair& pair : matchingPairs)
    {
        const bool chooseA = draw01(m_rng) < config.matchingGeneChooseParentAProbability;
        const ConnectionGene& chosen = chooseA ? *pair.a : *pair.b;

        bool childEnabled;
        if (pair.a->isEnabled() && pair.b->isEnabled())
        {
            childEnabled = true;
        }
        else
        {
            childEnabled = draw01(m_rng) >= config.disabledGeneRemainDisabledProbability;
        }

        childConnections.emplace_back(chosen.getSourceId(), chosen.getTargetId(), chosen.getWeight(), childEnabled,
                                       pair.innovation);
    }

    const bool fitterIsA = fitnessA > fitnessB;
    const bool fitterIsB = fitnessB > fitnessA;

    if (fitterIsA)
    {
        for (const ConnectionGene* connection : aOnly)
        {
            childConnections.push_back(*connection);
        }
    }
    else if (fitterIsB)
    {
        for (const ConnectionGene* connection : bOnly)
        {
            childConnections.push_back(*connection);
        }
    }
    else
    {
        // Equal fitness: merge both parents' non-matching candidates,
        // ascending by innovation, so RNG draw order is deterministic.
        std::vector<const ConnectionGene*> candidates;
        candidates.reserve(aOnly.size() + bOnly.size());
        candidates.insert(candidates.end(), aOnly.begin(), aOnly.end());
        candidates.insert(candidates.end(), bOnly.begin(), bOnly.end());
        std::sort(candidates.begin(), candidates.end(), [](const ConnectionGene* x, const ConnectionGene* y)
                  { return x->getInnovationNumber() < y->getInnovationNumber(); });

        for (const ConnectionGene* candidate : candidates)
        {
            const bool include = draw01(m_rng) < 0.5f;
            if (!include)
            {
                continue;
            }
            if (wouldCreateDuplicateDirectedConnection(childConnections, *candidate) ||
                wouldCreateEnabledCycle(childConnections, *candidate))
            {
                continue; // would make the child invalid -- skip, don't repair
            }
            childConnections.push_back(*candidate);
        }
    }

    // Interface nodes (Input/Bias/Output) always included; Hidden nodes only
    // if an inherited connection references them.
    std::map<NodeId, NodeGene> requiredNodes;
    for (const NodeGene& node : parentA.nodes())
    {
        if (node.getType() != NodeType::Hidden)
        {
            requiredNodes.emplace(node.getId(), node);
        }
    }
    for (const NodeGene& node : parentB.nodes())
    {
        if (node.getType() != NodeType::Hidden)
        {
            requiredNodes.emplace(node.getId(), node);
        }
    }
    for (const ConnectionGene& connection : childConnections)
    {
        for (NodeId endpoint : {connection.getSourceId(), connection.getTargetId()})
        {
            if (requiredNodes.find(endpoint) == requiredNodes.end())
            {
                const NodeGene* found = findNodeInEitherParent(endpoint, parentA, parentB);
                if (found == nullptr)
                {
                    // Defensive guard; shouldn't happen given the invariants above.
                    throw std::invalid_argument("GenomeCrossover: inherited connection references an unknown node");
                }
                requiredNodes.emplace(endpoint, *found);
            }
        }
    }

    std::vector<NodeGene> childNodes;
    childNodes.reserve(requiredNodes.size());
    for (const auto& [id, node] : requiredNodes)
    {
        childNodes.push_back(node);
    }

    std::sort(childConnections.begin(), childConnections.end(), [](const ConnectionGene& x, const ConnectionGene& y)
              { return x.getInnovationNumber() < y.getInnovationNumber(); });

    Genome child(childNodes, childConnections);
    child.validate();
    (void)buildPhenotype(child); // validity/acyclicity check only; result discarded

    return child;
}

} // namespace ai::neat
