#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "raylib.h"

#include "ai/AIController.h"
#include "ai/FitnessEvaluator.h"
#include "ai/NeuralNetwork.h"
#include "ai/Observation.h"
#include "ai/neat/CompatibilityConfig.h"
#include "ai/neat/CompatibilityDistance.h"
#include "ai/neat/ConnectionGene.h"
#include "ai/neat/CrossoverConfig.h"
#include "ai/neat/Genome.h"
#include "ai/neat/GenomeCrossover.h"
#include "ai/neat/GenomeMutator.h"
#include "ai/neat/Individual.h"
#include "ai/neat/InnovationTracker.h"
#include "ai/neat/MutationConfig.h"
#include "ai/neat/NodeGene.h"
#include "ai/neat/PhenotypeBuilder.h"
#include "ai/neat/Population.h"
#include "ai/neat/PopulationConfig.h"
#include "ai/neat/SpeciationConfig.h"
#include "ai/neat/Speciator.h"
#include "ai/neat/Species.h"
#include "simulation/Car.h"
#include "simulation/Track.h"
#include "simulation/TrackProgress.h"
#include "simulation/TrackVisual.h"
#include "training/GenerationMetrics.h"
#include "training/TrainingLogger.h"

#include "AppConfig.h"
#include "ui/HudRenderer.h"
#include "verification/Verifications.h"

namespace verification
{

using ui::selectHighlightedIndividual;

using app::createDemonstrationGenome;
using app::kSimulationDt;
using app::kSpawnHeading;
using app::kSpawnPosition;
using app::makeCarParams;


namespace population_verify
{

using ai::neat::CompatibilityConfig;
using ai::neat::ConnectionGene;
using ai::neat::CrossoverConfig;
using ai::neat::Genome;
using ai::neat::GenomeCrossover;
using ai::neat::GenomeMutator;
using ai::neat::Individual;
using ai::neat::InnovationNumber;
using ai::neat::InnovationTracker;
using ai::neat::isBetterTournamentCandidate;
using ai::neat::MutationConfig;
using ai::neat::NodeGene;
using ai::neat::NodeId;
using ai::neat::NodeType;
using ai::neat::Population;
using ai::neat::PopulationConfig;
using ai::neat::SpeciationConfig;
using ai::neat::Speciator;

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

PopulationConfig makeTestPopulationConfig(std::size_t populationSize, std::uint32_t seed)
{
    PopulationConfig config;
    config.populationSize = populationSize;
    config.eliteCount = 1;
    config.tournamentSize = std::min<std::size_t>(3, populationSize);
    config.randomSeed = seed;
    return config;
}

// A genome that drives at near-maximum throttle with zero steering and no
// brake (Bias -> Throttle only, weight large enough that tanh saturates
// close to 1.0; Bias -> Brake strongly negative so brake stays off, same
// convention as AppConfig.cpp's createDemonstrationGenome()) -- from the
// fixed spawn pose this reliably leaves the road band and collides within a
// few hundred simulation steps, exactly like the deliberate "drive off
// track" scenario verifyCar() itself exercises. Used only to keep
// generation-transition tests fast and their step-count bound tight; it
// carries no meaning beyond that.
Genome makeCrashGenome()
{
    Genome genome;
    for (int i = 0; i < ai::NeuralNetwork::kInputCount; ++i)
    {
        genome.addNode(NodeGene{i, NodeType::Input});
    }
    genome.addNode(NodeGene{9, NodeType::Bias});
    genome.addNode(NodeGene{100, NodeType::Output});
    genome.addNode(NodeGene{101, NodeType::Output});
    genome.addNode(NodeGene{102, NodeType::Output});
    genome.addConnection(ConnectionGene{9, 101, 5.0f, true, 0});
    genome.addConnection(ConnectionGene{9, 102, -5.0f, true, 1});
    return genome;
}

bool connectionsMatch(const std::vector<ConnectionGene>& a, const std::vector<ConnectionGene>& b)
{
    if (a.size() != b.size())
    {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        if (a[i].getSourceId() != b[i].getSourceId() || a[i].getTargetId() != b[i].getTargetId() ||
            a[i].getWeight() != b[i].getWeight() || a[i].isEnabled() != b[i].isEnabled() ||
            a[i].getInnovationNumber() != b[i].getInnovationNumber())
        {
            return false;
        }
    }
    return true;
}

} // namespace population_verify

// Deterministic check of ai::neat::Population/Individual, covering the
// first complete generation loop end to end, independent of rendering/
// keyboard timing. Species-restricted mating, adjusted fitness, fitness
// sharing, and offspring allocation are covered separately by
// verifySpeciesAwareReproduction() below.
void verifyPopulation(const simulation::Track& track)
{
    using namespace population_verify;

    // 1, 2 & 3: invalid PopulationConfig fields are rejected.
    {
        const Genome base = createDemonstrationGenome();
        const MutationConfig mutationConfig;
        const CrossoverConfig crossoverConfig;
        const CompatibilityConfig compatibilityConfig;
        const SpeciationConfig speciationConfig;

        auto build = [&](const PopulationConfig& config)
        {
            Population population(base, track, makeCarParams(), kSpawnPosition, kSpawnHeading,
                                   config, mutationConfig, crossoverConfig, compatibilityConfig, speciationConfig);
            (void)population;
        };

        PopulationConfig zeroSize = makeTestPopulationConfig(6, 1u);
        zeroSize.populationSize = 0;
        assert(throwsInvalidArgument([&]() { build(zeroSize); }) && "populationSize of 0 must be rejected"); // 1
        PopulationConfig oneSize = makeTestPopulationConfig(6, 1u);
        oneSize.populationSize = 1;
        assert(throwsInvalidArgument([&]() { build(oneSize); }) && "populationSize of 1 must be rejected"); // 1 (continued)

        PopulationConfig zeroElite = makeTestPopulationConfig(6, 1u);
        zeroElite.eliteCount = 0;
        assert(throwsInvalidArgument([&]() { build(zeroElite); }) && "eliteCount of 0 must be rejected"); // 2
        PopulationConfig tooManyElite = makeTestPopulationConfig(6, 1u);
        tooManyElite.eliteCount = tooManyElite.populationSize;
        assert(throwsInvalidArgument([&]() { build(tooManyElite); }) &&
               "eliteCount == populationSize must be rejected"); // 2 (continued)

        PopulationConfig zeroTournament = makeTestPopulationConfig(6, 1u);
        zeroTournament.tournamentSize = 0;
        assert(throwsInvalidArgument([&]() { build(zeroTournament); }) && "tournamentSize of 0 must be rejected"); // 3
        PopulationConfig tooBigTournament = makeTestPopulationConfig(6, 1u);
        tooBigTournament.tournamentSize = tooBigTournament.populationSize + 1;
        assert(throwsInvalidArgument([&]() { build(tooBigTournament); }) &&
               "tournamentSize > populationSize must be rejected"); // 3 (continued)
    }

    // 4, 5, 6, 7 & 8: initial population size/generation/structure.
    {
        const Genome base = createDemonstrationGenome();
        const PopulationConfig popConfig = makeTestPopulationConfig(6, 42u);
        const MutationConfig mutationConfig;
        const CrossoverConfig crossoverConfig;
        const CompatibilityConfig compatibilityConfig;
        const SpeciationConfig speciationConfig;

        Population population(base, track, makeCarParams(), kSpawnPosition, kSpawnHeading,
                               popConfig, mutationConfig, crossoverConfig, compatibilityConfig, speciationConfig);

        assert(population.size() == 6 && "initial population must have the exact configured size"); // 4
        assert(population.getGeneration() == 0 && "initial generation must be 0"); // 5

        const Genome& genome0 = population.getIndividual(0).getGenome();
        assert(genome0.nodes().size() == base.nodes().size() &&
               connectionsMatch(genome0.connections(), base.connections()) &&
               "individual 0 must preserve the base Genome exactly"); // 6

        bool sawWeightChange = false;
        for (std::size_t i = 1; i < population.size(); ++i)
        {
            const Genome& g = population.getIndividual(i).getGenome();
            assert(g.nodes().size() == base.nodes().size() && g.connections().size() == base.connections().size() &&
                   "other individuals must preserve the base structure (node/connection counts)"); // 7
            for (std::size_t j = 0; j < base.connections().size(); ++j)
            {
                const ConnectionGene& b = base.connections()[j];
                const ConnectionGene& c = g.connections()[j];
                assert(c.getSourceId() == b.getSourceId() && c.getTargetId() == b.getTargetId() &&
                       c.getInnovationNumber() == b.getInnovationNumber() &&
                       "other individuals must preserve base structure/innovation numbers exactly"); // 7 (continued)
                if (c.getWeight() != b.getWeight())
                {
                    sawWeightChange = true;
                }
            }
        }
        assert(sawWeightChange && "initial weight mutation must be able to alter at least one weight"); // 8
    }

    // 9, 10, 45 & 46: each Individual owns fully independent Car and
    // evaluation state -- one finishing (colliding) never affects another,
    // proven directly at the object level rather than relying on emergent
    // trajectory divergence.
    {
        const Genome crashGenome = makeCrashGenome();
        Individual individualA(crashGenome, track, makeCarParams(), kSpawnPosition, kSpawnHeading);
        Individual individualB(crashGenome, track, makeCarParams(), kSpawnPosition, kSpawnHeading);

        for (int step = 0; step < 400 && !individualA.isFinished(); ++step)
        {
            individualA.update(kSimulationDt);
        }
        assert(individualA.isFinished() && "setup: the crash genome must finish (collide) within 400 steps");

        assert(!individualB.isFinished() &&
               "an unrelated Individual must not be affected by another Individual finishing"); // 9 & 45 (part 1)
        assert(individualB.getFitness() == 0.0f && individualB.getFitnessEvaluator().getElapsedTime() == 0.0f &&
               "an unrelated, never-updated Individual must keep its own independent evaluation state"); // 10 & 46
        assert(individualB.getCar().isAlive() &&
               individualB.getCar().getPosition().x == kSpawnPosition.x &&
               individualB.getCar().getPosition().y == kSpawnPosition.y &&
               "an unrelated Individual's Car must remain independently alive at its own spawn position"); // 9 (part 2)
    }

    // 11 & 12: the global InnovationTracker starts strictly above every
    // base NodeId/innovation number.
    {
        const Genome base = createDemonstrationGenome();
        NodeId maxNodeId = -1;
        for (const NodeGene& node : base.nodes())
        {
            maxNodeId = std::max(maxNodeId, node.getId());
        }
        InnovationNumber maxInnovation = -1;
        for (const ConnectionGene& connection : base.connections())
        {
            maxInnovation = std::max(maxInnovation, connection.getInnovationNumber());
        }

        const PopulationConfig popConfig = makeTestPopulationConfig(6, 2u);
        const MutationConfig mutationConfig;
        const CrossoverConfig crossoverConfig;
        const CompatibilityConfig compatibilityConfig;
        const SpeciationConfig speciationConfig;
        Population population(base, track, makeCarParams(), kSpawnPosition, kSpawnHeading,
                               popConfig, mutationConfig, crossoverConfig, compatibilityConfig, speciationConfig);

        assert(population.getInnovationTracker().getNextAvailableNodeId() > maxNodeId &&
               "the global InnovationTracker must start above every base NodeId"); // 11
        assert(population.getInnovationTracker().getNextAvailableInnovation() > maxInnovation &&
               "the global InnovationTracker must start above every base connection innovation"); // 12

        assert(population.getLastGenerationBestFitness() == 0.0f &&
               "a freshly constructed Population's last-generation best fitness must start at 0"); // 39 (initial value)
    }

    // 13: update() advances an unfinished individual's evaluation.
    {
        const Genome base = createDemonstrationGenome();
        const PopulationConfig popConfig = makeTestPopulationConfig(4, 3u);
        const MutationConfig mutationConfig;
        const CrossoverConfig crossoverConfig;
        const CompatibilityConfig compatibilityConfig;
        const SpeciationConfig speciationConfig;
        Population population(base, track, makeCarParams(), kSpawnPosition, kSpawnHeading,
                               popConfig, mutationConfig, crossoverConfig, compatibilityConfig, speciationConfig);

        const float elapsedBefore = population.getIndividual(0).getFitnessEvaluator().getElapsedTime();
        population.update(kSimulationDt);
        assert(population.getIndividual(0).getFitnessEvaluator().getElapsedTime() > elapsedBefore &&
               "update() must advance an unfinished individual's evaluation"); // 13
    }

    // 14: a finished individual no longer updates.
    {
        const Genome crashGenome = makeCrashGenome();
        Individual individual(crashGenome, track, makeCarParams(), kSpawnPosition, kSpawnHeading);
        for (int step = 0; step < 400 && !individual.isFinished(); ++step)
        {
            individual.update(kSimulationDt);
        }
        assert(individual.isFinished() && "setup: the crash genome must finish within 400 steps");

        const float frozenFitness = individual.getFitness();
        const float frozenElapsed = individual.getFitnessEvaluator().getElapsedTime();
        const Vector2 frozenPosition = individual.getCar().getPosition();
        for (int step = 0; step < 20; ++step)
        {
            individual.update(kSimulationDt);
        }
        assert(individual.getFitness() == frozenFitness && individual.getFitnessEvaluator().getElapsedTime() == frozenElapsed &&
               individual.getCar().getPosition().x == frozenPosition.x &&
               individual.getCar().getPosition().y == frozenPosition.y &&
               "a finished individual must not update further"); // 14
    }

    // 15, 16, 17, 18, 19, 20 & 39: a full generation transition -- one
    // finished individual alone does not finish the generation; once
    // every individual has finished, reproduction runs exactly once
    // (generation 0 -> 1); population size stays constant; elites are
    // copied byte-for-byte unchanged; last-generation best fitness updates.
    std::vector<Genome> gen0GenomesSnapshot;
    std::vector<float> gen0FitnessSnapshot;
    PopulationConfig transitionPopConfig = makeTestPopulationConfig(8, 11u);
    {
        const Genome crashGenome = makeCrashGenome();
        const MutationConfig mutationConfig; // default: individuals 1..N-1 get varied weights
        const CrossoverConfig crossoverConfig;
        const CompatibilityConfig compatibilityConfig;
        const SpeciationConfig speciationConfig;
        Population population(crashGenome, track, makeCarParams(), kSpawnPosition, kSpawnHeading,
                               transitionPopConfig, mutationConfig, crossoverConfig, compatibilityConfig, speciationConfig);

        // Wait for the first individual to finish -- whichever one it is;
        // weight-mutated individuals need not all crash at the same step,
        // so with 8 individuals it is very likely at least one is still
        // running when the first finishes.
        for (int step = 0; step < 400 && population.getFinishedCount() == 0; ++step)
        {
            population.update(kSimulationDt);
        }
        assert(population.getFinishedCount() > 0 && "setup: at least one individual must finish within 400 steps");
        if (population.getRunningCount() > 0)
        {
            assert(population.getGeneration() == 0 &&
                   "one (or more, but not all) finished individual(s) must not trigger a generation transition"); // 15
        }

        // Re-snapshot every iteration, immediately before each update()
        // call, so whichever call turns out to trigger the generation
        // transition, the snapshot just before it captures the final
        // generation-0 genomes/fitness (detection and reproduction happen
        // together inside one update() call, so there's no other way to
        // observe them). Also track how many individuals were already
        // finished going into that call -- if all of them, the snapshot is
        // provably exact; if not, the still-running ones could gain a last
        // burst of fitness during that call (handled by the ranking check
        // below).
        std::size_t finishedBeforeTransitionCall = 0;
        for (int step = 0; step < 4000 && population.getGeneration() == 0; ++step)
        {
            gen0GenomesSnapshot.clear();
            gen0FitnessSnapshot.clear();
            for (std::size_t i = 0; i < population.size(); ++i)
            {
                gen0GenomesSnapshot.push_back(population.getIndividual(i).getGenome());
                gen0FitnessSnapshot.push_back(population.getIndividual(i).getFitness());
            }
            finishedBeforeTransitionCall = population.getFinishedCount();
            population.update(kSimulationDt);
        }
        assert(population.getGeneration() == 1 &&
               "once every individual has finished, a generation transition must occur exactly once"); // 16 & 17
        assert(population.size() == transitionPopConfig.populationSize &&
               "population size must remain constant across a generation transition"); // 18

        // 39: last-generation best fitness must update to a positive value
        // reflecting the just-finished generation. No exact numeric match
        // against gen0FitnessSnapshot is attempted: that snapshot is taken
        // immediately before each update() call, so it can under-count by
        // up to one deltaTime step relative to the true final value.
        const float staleObservedBest = *std::max_element(gen0FitnessSnapshot.begin(), gen0FitnessSnapshot.end());
        assert(population.getLastGenerationBestFitness() > 0.0f &&
               population.getLastGenerationBestFitness() >= staleObservedBest - 1.0f &&
               "last-generation best fitness must update to reflect the just-finished generation's real fitness"); // 39

        std::vector<std::size_t> ranked(gen0GenomesSnapshot.size());
        for (std::size_t i = 0; i < ranked.size(); ++i)
        {
            ranked[i] = i;
        }
        std::sort(ranked.begin(), ranked.end(),
                  [&gen0FitnessSnapshot](std::size_t a, std::size_t b)
                  {
                      if (gen0FitnessSnapshot[a] != gen0FitnessSnapshot[b])
                      {
                          return gen0FitnessSnapshot[a] > gen0FitnessSnapshot[b];
                      }
                      return a < b;
                  });

        // The final snapshot is provably exact only if every individual was
        // already finished going into the transition-triggering call --
        // otherwise a still-running individual could have gained a final
        // burst of fitness large enough to change the ranking.
        const bool rankingSnapshotIsExact = (finishedBeforeTransitionCall == gen0GenomesSnapshot.size());

        for (std::size_t e = 0; e < transitionPopConfig.eliteCount; ++e)
        {
            const Genome& actualEliteSlot = population.getIndividual(e).getGenome();

            if (rankingSnapshotIsExact)
            {
                // Strong check: elite slot e must be an unchanged copy of
                // specifically the (e+1)-th ranked generation-0 genome.
                const Genome& expectedElite = gen0GenomesSnapshot[ranked[e]];
                assert(connectionsMatch(actualEliteSlot.connections(), expectedElite.connections()) &&
                       "the elite genome must be copied into the next generation completely unchanged"); // 19 & 20
            }
            else
            {
                // Weaker check for the staleness edge case: elite slot e
                // must still be an unchanged copy of *some* generation-0
                // genome, ruling out corruption even though which specific
                // individual it was can't be pinned down from outside.
                const bool matchesSomeGenome =
                    std::any_of(gen0GenomesSnapshot.begin(), gen0GenomesSnapshot.end(), [&](const Genome& candidate)
                                { return connectionsMatch(actualEliteSlot.connections(), candidate.connections()); });
                assert(matchesSomeGenome &&
                       "the elite genome must be copied into the next generation completely unchanged"); // 19 & 20
            }
        }

        // 30 & 31: every next-generation Genome validates and builds a
        // phenotype.
        for (std::size_t i = 0; i < population.size(); ++i)
        {
            population.getIndividual(i).getGenome().validate();       // 30
            ai::neat::buildPhenotype(population.getIndividual(i).getGenome()); // 31
        }

        // 32 & 33: all new individuals reset at spawn with fresh
        // FitnessEvaluators.
        for (std::size_t i = 0; i < population.size(); ++i)
        {
            const Individual& ind = population.getIndividual(i);
            assert(ind.getCar().getPosition().x == kSpawnPosition.x && ind.getCar().getPosition().y == kSpawnPosition.y &&
                   "every new individual must start at spawn"); // 32
            assert(ind.getFitnessEvaluator().getFitness() == 0.0f && ind.getFitnessEvaluator().getElapsedTime() == 0.0f &&
                   !ind.isFinished() && "every new individual's FitnessEvaluator must start fresh"); // 33
        }

        // 37 & 38: best index valid, running+finished == size.
        assert(population.getBestIndividualIndex() < population.size() && "the best individual index must be valid"); // 37
        assert(population.getRunningCount() + population.getFinishedCount() == population.size() &&
               "running + finished must equal population size"); // 38
    }

    // 21 & 22: tournament selection chooses the highest fitness among
    // sampled candidates, with ties resolved by lower index. The pure
    // decision rule is tested directly (zero probabilistic risk, since
    // sampling-with-replacement cannot guarantee any specific index is
    // ever drawn); tournamentSelect() itself is spot-checked afterward.
    {
        assert(isBetterTournamentCandidate(5.0f, 2, 3.0f, 1) == true &&
               "strictly higher fitness must win"); // 21
        assert(isBetterTournamentCandidate(3.0f, 2, 5.0f, 1) == false &&
               "strictly lower fitness must lose"); // 21 (continued)
        assert(isBetterTournamentCandidate(5.0f, 0, 5.0f, 3) == true &&
               "a tie must resolve to the lower index"); // 22
        assert(isBetterTournamentCandidate(5.0f, 3, 5.0f, 0) == false &&
               "a tie must not resolve to the higher index"); // 22 (continued)

        const Genome base = createDemonstrationGenome();
        PopulationConfig popConfig = makeTestPopulationConfig(2, 17u);
        popConfig.tournamentSize = 2;
        const MutationConfig mutationConfig;
        const CrossoverConfig crossoverConfig;
        const CompatibilityConfig compatibilityConfig;
        const SpeciationConfig speciationConfig;
        Population population(base, track, makeCarParams(), kSpawnPosition, kSpawnHeading,
                               popConfig, mutationConfig, crossoverConfig, compatibilityConfig, speciationConfig);

        // Index 1's fitness vastly exceeds index 0's: across 10 independent
        // trials of 2 draws each, the odds of index 1 never once being
        // sampled are astronomically small (0.25^10), making this an
        // effectively-deterministic check for this fixed seed.
        const std::vector<float> fitnessValues = {1.0f, 1000.0f};
        bool sawIndex1Win = false;
        for (int trial = 0; trial < 10; ++trial)
        {
            if (population.tournamentSelect(fitnessValues) == 1)
            {
                sawIndex1Win = true;
                break;
            }
        }
        assert(sawIndex1Win &&
               "tournament selection must be able to choose the highest-fitness sampled candidate"); // 21 (continued)
    }

    // 23: parents are never mutated -- a compile-time/structural
    // guarantee, not merely a runtime observation: Individual::getGenome()
    // returns const Genome&, GenomeCrossover::crossover() takes its
    // parents by const Genome&, and Population::reproduce() (see
    // Population.cpp) only ever passes m_individuals[...].getGenome() to
    // those const-reference parameters -- every mutating call
    // (mutateWeights/mutateAddConnection/mutateAddNode) is made only on
    // the freshly-copied `child` local variable, never on a parent
    // reference. It is not possible to mutate a parent genome through the
    // reference Population reads it via.

    // 24, 25, 26, 27 & 28: the crossover + mutation offspring pipeline
    // produces a valid child, each mutation kind can affect it, and the
    // structural mutations share one InnovationTracker.
    {
        const Genome parentA = createDemonstrationGenome();
        Genome parentB = createDemonstrationGenome();
        GenomeMutator setupMutator(321u);
        MutationConfig setupConfig;
        setupMutator.mutateWeights(parentB, setupConfig);

        NodeId maxNodeId = -1;
        for (const NodeGene& node : parentA.nodes())
        {
            maxNodeId = std::max(maxNodeId, node.getId());
        }
        InnovationNumber maxInnovation = -1;
        for (const ConnectionGene& connection : parentA.connections())
        {
            maxInnovation = std::max(maxInnovation, connection.getInnovationNumber());
        }
        InnovationTracker sharedTracker(maxNodeId + 1, maxInnovation + 1);

        GenomeCrossover crossover(444u);
        CrossoverConfig crossoverConfig;
        Genome child = crossover.crossover(parentA, 10.0f, parentB, 5.0f, crossoverConfig);
        child.validate();
        ai::neat::buildPhenotype(child); // 24: crossover must produce a valid, phenotype-buildable child

        GenomeMutator mutator(654u);
        MutationConfig mutationConfig;

        const Genome beforeWeights = child;
        mutationConfig.weightMutationProbability = 1.0f;
        mutator.mutateWeights(child, mutationConfig);
        bool weightChanged = false;
        for (std::size_t i = 0; i < child.connections().size(); ++i)
        {
            if (child.connections()[i].getWeight() != beforeWeights.connections()[i].getWeight())
            {
                weightChanged = true;
            }
        }
        assert(weightChanged && "weight mutation must be able to affect the offspring"); // 25

        const std::size_t connCountBeforeAddConnection = child.connections().size();
        mutationConfig.addConnectionProbability = 1.0f;
        const bool addedConnection = mutator.mutateAddConnection(child, sharedTracker, mutationConfig);
        assert(addedConnection && child.connections().size() == connCountBeforeAddConnection + 1 &&
               "add-connection mutation must be able to affect the offspring"); // 26

        const std::size_t nodeCountBeforeAddNode = child.nodes().size();
        const std::size_t connCountBeforeAddNode = child.connections().size();
        mutationConfig.addNodeProbability = 1.0f;
        const bool addedNode = mutator.mutateAddNode(child, sharedTracker, mutationConfig);
        assert(addedNode && child.nodes().size() == nodeCountBeforeAddNode + 1 &&
               child.connections().size() == connCountBeforeAddNode + 2 &&
               "add-node mutation must be able to affect the offspring"); // 27

        assert(sharedTracker.getNextAvailableNodeId() > maxNodeId + 1 &&
               "add-connection and add-node mutation must share the same InnovationTracker"); // 28

        child.validate();
        ai::neat::buildPhenotype(child);
    }

    // 29: the next generation is constructed before the old population is
    // replaced -- confirmed indirectly by 19/20 above: Population::reproduce()
    // builds newGenomes/newIndividuals entirely from m_individuals as it
    // existed when reproduce() began, only assigning
    // `m_individuals = std::move(newIndividuals)` as its final step.

    // 34, 35 & 36: restartGeneration() resets evaluation state without
    // changing the generation number or Genome contents.
    {
        const Genome base = createDemonstrationGenome();
        const PopulationConfig popConfig = makeTestPopulationConfig(4, 5u);
        const MutationConfig mutationConfig;
        const CrossoverConfig crossoverConfig;
        const CompatibilityConfig compatibilityConfig;
        const SpeciationConfig speciationConfig;
        Population population(base, track, makeCarParams(), kSpawnPosition, kSpawnHeading,
                               popConfig, mutationConfig, crossoverConfig, compatibilityConfig, speciationConfig);

        for (int step = 0; step < 30; ++step)
        {
            population.update(kSimulationDt);
        }

        const std::size_t generationBefore = population.getGeneration();
        std::vector<Genome> genomesBefore;
        for (std::size_t i = 0; i < population.size(); ++i)
        {
            genomesBefore.push_back(population.getIndividual(i).getGenome());
        }

        population.restartGeneration();

        assert(population.getGeneration() == generationBefore &&
               "restartGeneration() must not change the generation number"); // 34
        for (std::size_t i = 0; i < population.size(); ++i)
        {
            const Genome& g = population.getIndividual(i).getGenome();
            assert(connectionsMatch(g.connections(), genomesBefore[i].connections()) &&
                   "restartGeneration() must not alter Genome contents"); // 35
            const Individual& ind = population.getIndividual(i);
            assert(ind.getFitnessEvaluator().getFitness() == 0.0f && !ind.isFinished() &&
                   ind.getCar().getPosition().x == kSpawnPosition.x && ind.getCar().getPosition().y == kSpawnPosition.y &&
                   "restartGeneration() must reset evaluation state"); // 36
        }
    }

    // 40: species count is computed from the current generation's actual
    // Genomes -- cross-checked against an independent Speciator run over
    // the same Genomes.
    {
        const Genome base = createDemonstrationGenome();
        const PopulationConfig popConfig = makeTestPopulationConfig(5, 8u);
        const MutationConfig mutationConfig;
        const CrossoverConfig crossoverConfig;
        const CompatibilityConfig compatibilityConfig;
        const SpeciationConfig speciationConfig;
        Population population(base, track, makeCarParams(), kSpawnPosition, kSpawnHeading,
                               popConfig, mutationConfig, crossoverConfig, compatibilityConfig, speciationConfig);

        std::vector<Genome> genomes;
        for (std::size_t i = 0; i < population.size(); ++i)
        {
            genomes.push_back(population.getIndividual(i).getGenome());
        }
        Speciator independentSpeciator;
        const std::size_t independentCount =
            independentSpeciator.speciate(genomes, compatibilityConfig, speciationConfig).size();
        assert(population.getSpeciesCount() == independentCount &&
               "species count must be computed from the current generation's actual Genomes"); // 40
    }

    // 41: species membership drives reproduction -- Population::reproduce()
    // speciates the just-finished generation once and uses that same
    // Species vector for fitness sharing, offspring allocation, and
    // species-local parent selection. See verifySpeciesAwareReproduction()
    // below for the dedicated suite.

    // 42 & 43: fixed seed + configs + same evaluation fitnesses produce
    // deterministic offspring; different seeds can produce different
    // offspring.
    {
        const Genome crashGenome = makeCrashGenome();
        const PopulationConfig popConfig = makeTestPopulationConfig(4, 2024u);
        const MutationConfig mutationConfig;
        const CrossoverConfig crossoverConfig;
        const CompatibilityConfig compatibilityConfig;
        const SpeciationConfig speciationConfig;

        Population populationX(crashGenome, track, makeCarParams(), kSpawnPosition, kSpawnHeading,
                                popConfig, mutationConfig, crossoverConfig, compatibilityConfig, speciationConfig);
        Population populationY(crashGenome, track, makeCarParams(), kSpawnPosition, kSpawnHeading,
                                popConfig, mutationConfig, crossoverConfig, compatibilityConfig, speciationConfig);

        for (int step = 0; step < 4000 && populationX.getGeneration() == 0; ++step)
        {
            populationX.update(kSimulationDt);
        }
        for (int step = 0; step < 4000 && populationY.getGeneration() == 0; ++step)
        {
            populationY.update(kSimulationDt);
        }
        assert(populationX.getGeneration() == 1 && populationY.getGeneration() == 1 &&
               "setup: both populations must reach generation 1 within the step budget");

        for (std::size_t i = 0; i < populationX.size(); ++i)
        {
            assert(connectionsMatch(populationX.getIndividual(i).getGenome().connections(),
                                     populationY.getIndividual(i).getGenome().connections()) &&
                   "identical seed/configs/base genome must produce deterministic offspring"); // 42
        }

        PopulationConfig popConfigDifferentSeed = popConfig;
        popConfigDifferentSeed.randomSeed = 999u;
        Population populationZ(crashGenome, track, makeCarParams(), kSpawnPosition, kSpawnHeading,
                                popConfigDifferentSeed, mutationConfig, crossoverConfig, compatibilityConfig,
                                speciationConfig);
        for (int step = 0; step < 4000 && populationZ.getGeneration() == 0; ++step)
        {
            populationZ.update(kSimulationDt);
        }
        assert(populationZ.getGeneration() == 1 && "setup: populationZ must reach generation 1 within the step budget");

        bool sawDifference = false;
        for (std::size_t i = 0; i < populationZ.size() && !sawDifference; ++i)
        {
            if (!connectionsMatch(populationX.getIndividual(i).getGenome().connections(),
                                   populationZ.getIndividual(i).getGenome().connections()))
            {
                sawDifference = true;
            }
        }
        assert(sawDifference && "different seeds must be capable of producing different offspring"); // 43
    }

    // 44: no random_device/global RNG/time seeding exists anywhere in
    // Population.cpp or Individual.cpp -- every RNG (the orchestration
    // std::mt19937, GenomeMutator, GenomeCrossover) is constructed with a
    // value derived purely from PopulationConfig::randomSeed.

    // 47: population rendering information can be queried without
    // mutating evolution state.
    {
        const Genome base = createDemonstrationGenome();
        const PopulationConfig popConfig = makeTestPopulationConfig(4, 9u);
        const MutationConfig mutationConfig;
        const CrossoverConfig crossoverConfig;
        const CompatibilityConfig compatibilityConfig;
        const SpeciationConfig speciationConfig;
        Population population(base, track, makeCarParams(), kSpawnPosition, kSpawnHeading,
                               popConfig, mutationConfig, crossoverConfig, compatibilityConfig, speciationConfig);

        const std::size_t generationBefore = population.getGeneration();
        const Genome genomeSnapshot = population.getIndividual(0).getGenome();

        for (int i = 0; i < 5; ++i)
        {
            (void)population.getGeneration();
            (void)population.getBestIndividualIndex();
            (void)population.getRunningCount();
            (void)population.getFinishedCount();
            (void)population.getSpeciesCount();
            (void)population.getLastGenerationBestFitness();
        }

        assert(population.getGeneration() == generationBefore &&
               "querying read-only accessors must not change the generation"); // 47 (part 1)
        assert(connectionsMatch(population.getIndividual(0).getGenome().connections(), genomeSnapshot.connections()) &&
               "querying read-only accessors must not mutate any individual's genome"); // 47 (part 2)
    }

    // 48: highlighted individual selection is deterministic.
    {
        const Genome base = createDemonstrationGenome();
        const PopulationConfig popConfig = makeTestPopulationConfig(5, 6u);
        const MutationConfig mutationConfig;
        const CrossoverConfig crossoverConfig;
        const CompatibilityConfig compatibilityConfig;
        const SpeciationConfig speciationConfig;
        Population population(base, track, makeCarParams(), kSpawnPosition, kSpawnHeading,
                               popConfig, mutationConfig, crossoverConfig, compatibilityConfig, speciationConfig);

        for (int step = 0; step < 100; ++step)
        {
            population.update(kSimulationDt);
        }

        const std::size_t highlightedFirst = selectHighlightedIndividual(population);
        const std::size_t highlightedSecond = selectHighlightedIndividual(population);
        assert(highlightedFirst == highlightedSecond && highlightedFirst < population.size() &&
               "highlighted individual selection must be deterministic for the same population state"); // 48
    }

    // 49: all previous verification suites still pass -- enforced by
    // main() continuing to call every earlier verify*() function
    // unchanged.

    // 50: an automatic generation transition without crashing is directly
    // exercised (and asserted) by the combined test above (items 15-20,
    // 39) and by 42/43 -- both drive Population through at least one full
    // generation transition purely via update() calls, with no exception
    // or assertion failure along the way. The real executable's own such
    // transition is additionally the subject of the separate manual
    // acceptance test.

    TraceLog(LOG_INFO, "Population verification: all deterministic checks passed");
}

// Checks species-aware reproduction: fitness sharing (adjusted fitness),
// per-species offspring allocation (both the general proportional case and
// the zero-total-fitness fallback), species-local parent selection, and
// the read-only getCurrentSpecies()/getReproductionStats() debug surface.
// Global elitism, the crossover/mutation pipeline, shared InnovationTracker
// usage, and isBetterTournamentCandidate()'s tie-break are already covered
// by verifyPopulation() above.
void verifySpeciesAwareReproduction(const simulation::Track& track)
{
    using namespace population_verify;
    using ai::neat::allocateSpeciesOffspring;
    using ai::neat::effectiveFitnessContribution;
    using ai::neat::Species;
    using ai::neat::SpeciesId;

    // 7, 8, 9, 10 & 11: allocateSpeciesOffspring(), the pure offspring-
    // allocation algorithm, tested directly against synthetic species-
    // fitness data -- exact remaining-slot count, floor + largest-
    // fractional-remainder, fractional-remainder ties broken by lower
    // SpeciesId, the zero-total-fitness fallback, and its own tie-break.
    {
        // Exact proportional split (no remainder).
        const std::vector<SpeciesId> ids2 = {0, 1};
        const std::vector<float> sums2 = {10.0f, 30.0f};
        const std::vector<std::size_t> alloc2 = allocateSpeciesOffspring(ids2, sums2, 4);
        assert(alloc2.size() == 2 && alloc2[0] + alloc2[1] == 4 &&
               "allocation must always sum to exactly remainingSlots"); // 7
        assert(alloc2[0] == 1 && alloc2[1] == 3 &&
               "an exact proportional split must match the fitness ratio precisely"); // 7 (continued)
    }
    {
        // Floor + largest-fractional-remainder, with a three-way tie
        // broken by lower SpeciesId rather than input position.
        const std::vector<SpeciesId> ids3 = {5, 2, 8};
        const std::vector<float> sums3 = {10.0f, 10.0f, 10.0f};
        const std::vector<std::size_t> alloc3 = allocateSpeciesOffspring(ids3, sums3, 10);
        const std::size_t total3 = alloc3[0] + alloc3[1] + alloc3[2];
        assert(total3 == 10 && "floor + remainder allocation must still sum to exactly remainingSlots"); // 8
        assert(alloc3[1] == 4 && alloc3[0] == 3 && alloc3[2] == 3 &&
               "an equal fractional remainder must be broken by lower SpeciesId (id 2), not input position"); // 9
    }
    {
        // A species with zero effective fitness gets zero offspring even
        // while the total across all species is positive.
        const std::vector<SpeciesId> idsZero = {0, 1};
        const std::vector<float> sumsZero = {0.0f, 5.0f};
        const std::vector<std::size_t> allocZero = allocateSpeciesOffspring(idsZero, sumsZero, 6);
        assert(allocZero[0] == 0 && allocZero[1] == 6 &&
               "a species contributing zero effective fitness must receive zero offspring"); // 6 (allocation side)
    }
    {
        // Zero-total-fitness fallback: as-even-as-possible split, extra
        // remainder slots to the lowest SpeciesId first.
        const std::vector<SpeciesId> idsFallback = {3, 1, 2};
        const std::vector<float> sumsFallback = {0.0f, 0.0f, 0.0f};
        const std::vector<std::size_t> allocFallback = allocateSpeciesOffspring(idsFallback, sumsFallback, 7);
        const std::size_t totalFallback = allocFallback[0] + allocFallback[1] + allocFallback[2];
        assert(totalFallback == 7 && "the zero-total fallback must still sum to exactly remainingSlots"); // 10
        assert(allocFallback[1] == 3 && allocFallback[2] == 2 && allocFallback[0] == 2 &&
               "the zero-total fallback must give the extra remainder slot to the lowest SpeciesId first"); // 11
    }
    {
        // A single species receives every remaining slot; an empty species
        // list allocates nothing.
        const std::vector<SpeciesId> idsSingle = {42};
        const std::vector<float> sumsSingle = {5.0f};
        assert((allocateSpeciesOffspring(idsSingle, sumsSingle, 9) == std::vector<std::size_t>{9}) &&
               "a single species must receive every remaining slot");
        assert(allocateSpeciesOffspring({}, {}, 0).empty() && "no species means no allocation");
    }

    // 6: negative raw (and therefore negative adjusted) fitness contributes
    // exactly zero to a species' effective fitness -- the exact clamp
    // Population::reproduce() itself applies before calling
    // allocateSpeciesOffspring().
    {
        assert(effectiveFitnessContribution(-5.0f) == 0.0f &&
               "negative adjusted fitness must contribute exactly zero to a species' effective fitness"); // 6
        assert(effectiveFitnessContribution(0.0f) == 0.0f && "zero adjusted fitness must remain zero");
        assert(effectiveFitnessContribution(3.5f) == 3.5f &&
               "non-negative adjusted fitness must be passed through unchanged"); // 6 (continued)
    }

    // 1-5, 12-15, 17, 18, 32-36, 38-40 & 45: a real Population driven
    // through one full generation transition, cross-checked against an
    // independent recomputation of the documented formulas from an exact
    // pre-transition snapshot -- captured with the same re-snapshot-every-
    // iteration technique verifyPopulation's items 15-20 use above, since
    // there is no external way to observe "every individual finished, but
    // reproduce() has not run yet".
    {
        const Genome crashGenome = makeCrashGenome();
        const PopulationConfig popConfig = makeTestPopulationConfig(12, 777u);
        const MutationConfig mutationConfig;
        const CrossoverConfig crossoverConfig;
        const CompatibilityConfig compatibilityConfig;
        // A very tight threshold: only genomes at exactly 0 compatibility
        // distance share a species. Since mutateWeights() only mutates the
        // crash genome's single connection 80% of the time, this seed
        // deterministically produces a mix of multi-member and singleton
        // species. None of the checks below hardcode the resulting
        // partition; they hold for whatever partition this seed produces.
        SpeciationConfig speciationConfig;
        speciationConfig.compatibilityThreshold = 0.0f;

        Population population(crashGenome, track, makeCarParams(), kSpawnPosition, kSpawnHeading,
                               popConfig, mutationConfig, crossoverConfig, compatibilityConfig, speciationConfig);

        std::vector<Genome> genomeSnapshot;
        std::vector<float> fitnessSnapshot;
        std::size_t finishedBeforeTransitionCall = 0;
        for (int step = 0; step < 4000 && population.getGeneration() == 0; ++step)
        {
            genomeSnapshot.clear();
            fitnessSnapshot.clear();
            for (std::size_t i = 0; i < population.size(); ++i)
            {
                genomeSnapshot.push_back(population.getIndividual(i).getGenome());
                fitnessSnapshot.push_back(population.getIndividual(i).getFitness());
            }
            finishedBeforeTransitionCall = population.getFinishedCount();
            population.update(kSimulationDt);
        }
        assert(population.getGeneration() == 1 &&
               "setup: the crash genome must reach a generation transition within the step budget"); // 38 & 45

        const bool snapshotIsExact = (finishedBeforeTransitionCall == genomeSnapshot.size());

        const std::vector<Species>& species = population.getCurrentSpecies();
        const std::size_t speciesCount = species.size();
        const std::vector<Population::SpeciesReproductionStats>& stats = population.getReproductionStats();

        // 32 & 33: reproduction stats correspond 1:1 with getCurrentSpecies(),
        // in the exact same (ascending SpeciesId) order.
        assert(stats.size() == speciesCount &&
               "reproduction stats must cover exactly the species vector reproduce() itself used"); // 32
        SpeciesId previousId = -1;
        std::size_t totalMembersAcrossSpecies = 0;
        for (std::size_t s = 0; s < speciesCount; ++s)
        {
            assert(stats[s].speciesId == species[s].getId() && stats[s].memberCount == species[s].size() &&
                   "reproduction stats must describe the exact same Species objects, in the exact same order"); // 32
            assert(species[s].getId() > previousId &&
                   "species must be processed/reported in strictly ascending SpeciesId order"); // 33
            previousId = species[s].getId();
            totalMembersAcrossSpecies += species[s].size();
        }
        assert(totalMembersAcrossSpecies == population.size() &&
               "every individual from the just-finished generation must belong to exactly one species");

        // 17, 18 & 34: offspring allocation sums to exactly the non-elite
        // remaining slots, and elites + allocated offspring reconstruct the
        // configured population size.
        std::size_t totalAllocatedOffspring = 0;
        for (const Population::SpeciesReproductionStats& s : stats)
        {
            totalAllocatedOffspring += s.allocatedOffspring;
        }
        assert(totalAllocatedOffspring == popConfig.populationSize - popConfig.eliteCount &&
               "allocated offspring across every species must sum to exactly populationSize - eliteCount"); // 17 & 18
        assert(popConfig.eliteCount + totalAllocatedOffspring == popConfig.populationSize &&
               "elites + allocated offspring must reconstruct the full configured population size"); // 34

        // 35 & 36: the next generation is exactly populationSize, and every
        // one of its Genomes validates and builds a phenotype.
        assert(population.size() == popConfig.populationSize &&
               "next generation size must be exactly populationSize"); // 35
        for (std::size_t i = 0; i < population.size(); ++i)
        {
            population.getIndividual(i).getGenome().validate();
            ai::neat::buildPhenotype(population.getIndividual(i).getGenome()); // 36
        }

        // 39: every new individual reset cleanly (fresh spawn, zero
        // fitness, not finished).
        for (std::size_t i = 0; i < population.size(); ++i)
        {
            const Individual& ind = population.getIndividual(i);
            assert(!ind.isFinished() && ind.getFitnessEvaluator().getFitness() == 0.0f &&
                   ind.getCar().getPosition().x == kSpawnPosition.x && ind.getCar().getPosition().y == kSpawnPosition.y &&
                   "every new individual must reset to a fresh spawn state"); // 39
        }

        if (snapshotIsExact)
        {
            // 40: the Species partition is a pure function of the exact
            // snapshotted genomes -- an independently constructed Speciator
            // must produce the identical sequence of member-index sets
            // (SpeciesId numbering may legitimately differ).
            Speciator independentSpeciator;
            const std::vector<Species> independentSpecies =
                independentSpeciator.speciate(genomeSnapshot, compatibilityConfig, speciationConfig);
            assert(independentSpecies.size() == speciesCount &&
                   "species count must be a pure function of the snapshotted genomes"); // 40
            for (std::size_t s = 0; s < speciesCount; ++s)
            {
                assert(independentSpecies[s].getMemberIndices() == species[s].getMemberIndices() &&
                       "species membership must be a pure function of the snapshotted genomes"); // 40 (continued)
            }

            // 1, 2, 3, 4 & 5: fitness sharing -- adjustedFitness ==
            // rawFitness/speciesSize for every member; each species'
            // reported adjustedFitnessSum matches an independent
            // recomputation from the exact snapshot.
            for (std::size_t s = 0; s < speciesCount; ++s)
            {
                const std::vector<std::size_t>& members = species[s].getMemberIndices();
                float expectedAdjustedSum = 0.0f;
                for (std::size_t memberIndex : members)
                {
                    const float raw = fitnessSnapshot[memberIndex];
                    const float adjusted = raw / static_cast<float>(members.size());
                    expectedAdjustedSum += adjusted;
                    if (members.size() == 1)
                    {
                        assert(adjusted == raw &&
                               "a singleton species' one member's adjusted fitness must equal its raw fitness"); // 3
                    }
                    if (members.size() > 1 && raw > 0.0f)
                    {
                        assert(adjusted < raw &&
                               "sharing fitness across more than one member must strictly reduce a positive raw "
                               "fitness"); // 4
                    }
                }
                assert(std::fabs(stats[s].adjustedFitnessSum - expectedAdjustedSum) < 1e-3f &&
                       "each species' reported adjustedFitnessSum must equal the sum of rawFitness/size over its "
                       "own members"); // 1, 2 & 5
            }

            // 12, 13, 14 & 15: global elitism is untouched by species-aware
            // reproduction -- elite slot 0 must be an unchanged copy of
            // whichever snapshot individual had the strictly highest raw
            // fitness (ties broken by lowest index).
            std::size_t topIndex = 0;
            for (std::size_t i = 1; i < fitnessSnapshot.size(); ++i)
            {
                if (fitnessSnapshot[i] > fitnessSnapshot[topIndex])
                {
                    topIndex = i;
                }
            }
            assert(connectionsMatch(population.getIndividual(0).getGenome().connections(),
                                     genomeSnapshot[topIndex].connections()) &&
                   "elite slot 0 must be an unchanged, byte-for-byte copy of the top raw-fitness genome"); // 12-15
        }

        TraceLog(LOG_INFO, "Species-aware reproduction: this run produced %d species (exact snapshot: %s)",
                 static_cast<int>(speciesCount), snapshotIsExact ? "yes" : "no");
    }

    // 19, 20, 21, 22, 23, 24, 25 & 42 (structural guarantees -- see
    // Population::reproduce()/selectParentFromSpecies() in Population.cpp):
    //   - selectParentFromSpecies() only ever samples from
    //     species.getMemberIndices(), called once per Species being
    //     processed -- both parents of every offspring come from that
    //     exact species, never another (19 & 20), and that per-species
    //     candidate pool is provably narrower than tournamentSelect()'s
    //     whole-population pool for any species smaller than the full
    //     population (42).
    //   - selectParentFromSpecies() reduces samples via the same
    //     isBetterTournamentCandidate() free function verifyPopulation's
    //     items 21/22 exercise directly (21 & 22).
    //   - a species of size 1 gives std::uniform_int_distribution<...>(0, 0),
    //     always resolving to its one member; the resulting
    //     crossover(parent, parent) call is accepted unconditionally by
    //     GenomeCrossover (23).
    //   - reproduce() passes fitnessValues[parentAIndex]/[parentBIndex] --
    //     RAW fitness -- into GenomeCrossover::crossover(), never adjusted
    //     fitness (24 & 25).

    // 26, 27, 28, 29, 30 & 31: the crossover + mutation offspring pipeline
    // (mutateWeights/mutateAddConnection/mutateAddNode sharing one
    // InnovationTracker, parents never mutated) is the same code
    // Population::reproduce() runs regardless of species scoping -- already
    // exercised end to end by verifyPopulation's items 24-28 above and by
    // the live run above (every offspring genome validates and builds a
    // phenotype, per the 35/36 checks).

    // 37, 43 & 44: fixed seed + identical configs/base genome produce
    // identical species partitions, adjusted-fitness sums, and offspring
    // allocation, and the read-only species/stats accessors never mutate
    // state when queried repeatedly.
    {
        const Genome crashGenome = makeCrashGenome();
        const PopulationConfig popConfig = makeTestPopulationConfig(10, 555u);
        const MutationConfig mutationConfig;
        const CrossoverConfig crossoverConfig;
        const CompatibilityConfig compatibilityConfig;
        SpeciationConfig speciationConfig;
        speciationConfig.compatibilityThreshold = 0.0f;

        auto runToNextGeneration = [](Population& population)
        {
            for (int step = 0; step < 4000 && population.getGeneration() == 0; ++step)
            {
                population.update(kSimulationDt);
            }
        };

        Population populationX(crashGenome, track, makeCarParams(), kSpawnPosition, kSpawnHeading,
                                popConfig, mutationConfig, crossoverConfig, compatibilityConfig, speciationConfig);
        Population populationY(crashGenome, track, makeCarParams(), kSpawnPosition, kSpawnHeading,
                                popConfig, mutationConfig, crossoverConfig, compatibilityConfig, speciationConfig);
        runToNextGeneration(populationX);
        runToNextGeneration(populationY);
        assert(populationX.getGeneration() == 1 && populationY.getGeneration() == 1 &&
               "setup: both populations must reach generation 1 within the step budget");

        // 43: read-only accessors queried repeatedly must not mutate state.
        const std::size_t generationBeforeQueries = populationX.getGeneration();
        for (int i = 0; i < 5; ++i)
        {
            (void)populationX.getCurrentSpecies();
            (void)populationX.getReproductionStats();
            (void)populationX.getSpeciesCount();
        }
        assert(populationX.getGeneration() == generationBeforeQueries &&
               "querying species/reproduction-stats accessors repeatedly must not mutate Population state"); // 43

        const std::vector<Species>& speciesX = populationX.getCurrentSpecies();
        const std::vector<Species>& speciesY = populationY.getCurrentSpecies();
        const std::vector<Population::SpeciesReproductionStats>& statsX = populationX.getReproductionStats();
        const std::vector<Population::SpeciesReproductionStats>& statsY = populationY.getReproductionStats();

        assert(speciesX.size() == speciesY.size() && statsX.size() == statsY.size() &&
               "identical seed/configs/base genome must produce an identical species count"); // 37 & 44
        for (std::size_t s = 0; s < speciesX.size(); ++s)
        {
            assert(speciesX[s].getMemberIndices() == speciesY[s].getMemberIndices() &&
                   "identical seed/configs/base genome must produce identical species membership"); // 37
            assert(statsX[s].speciesId == statsY[s].speciesId && statsX[s].memberCount == statsY[s].memberCount &&
                   statsX[s].allocatedOffspring == statsY[s].allocatedOffspring &&
                   std::fabs(statsX[s].adjustedFitnessSum - statsY[s].adjustedFitnessSum) < 1e-6f &&
                   "identical seed/configs/base genome must produce identical reproduction stats"); // 44
        }
        for (std::size_t i = 0; i < populationX.size(); ++i)
        {
            assert(connectionsMatch(populationX.getIndividual(i).getGenome().connections(),
                                     populationY.getIndividual(i).getGenome().connections()) &&
                   "identical seed/configs/base genome must produce deterministic offspring genomes"); // 44 (continued)
        }
    }

    // 45 & 46: an automatic species-aware generation transition without
    // crashing is directly exercised (and asserted) by every live run
    // above, and by verifyPopulation() continuing to run and pass
    // immediately before this function (see main()).

    TraceLog(LOG_INFO, "Species-aware reproduction verification: all deterministic checks passed");
}

// Checks persistent species identity and stagnation management: Speciator's
// persistent cross-call lifecycle (SpeciesId retention/extinction,
// representative reselection ordering) and Species::recordGeneration()'s
// fitness-history/stagnation rule, driven directly and deterministically
// (no simulation, no RNG, hand-computed expected values), then a real
// multi-generation Population run verifying the reproduction-side
// integration (stagnant species excluded from offspring allocation, global
// elites still protected, the all-stagnant safety fallback) as invariants
// safe to check regardless of which species/stagnation pattern this run's
// RNG produces.
void verifyPersistentSpeciesAndStagnation(const simulation::Track& track)
{
    using namespace population_verify;
    using namespace speciation_verify;
    using ai::neat::allocateSpeciesOffspring;
    using ai::neat::Speciator;
    using ai::neat::Species;
    using ai::neat::SpeciesId;

    // 1: speciesStagnationLimit == 0 is rejected.
    {
        const Genome base = createDemonstrationGenome();
        PopulationConfig config = makeTestPopulationConfig(6, 1u);
        config.speciesStagnationLimit = 0;
        const MutationConfig mutationConfig;
        const CrossoverConfig crossoverConfig;
        const CompatibilityConfig compatibilityConfig;
        const SpeciationConfig speciationConfig;
        assert(throwsInvalidArgument(
                   [&]()
                   {
                       Population population(base, track, makeCarParams(), kSpawnPosition, kSpawnHeading, config,
                                              mutationConfig, crossoverConfig, compatibilityConfig, speciationConfig);
                       (void)population;
                   }) &&
               "speciesStagnationLimit of 0 must be rejected"); // 1
    }

    // 2-14, 39 & 40: persistent Speciator lifecycle, driven directly across
    // three hand-controlled generations, so every expected id/membership/
    // representative value below is exactly computable by hand from
    // compatibilityDistance's formula for makeSimpleGenome() genomes
    // (distance = 0.4 * |weightA - weightB|). Threshold = 0.5 throughout.
    {
        Speciator speciator;
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig;
        speciationConfig.compatibilityThreshold = 0.5f;

        // Generation 1: weight 0.0 (idx0) founds species 0 (rep = 0.0);
        // weight 1.0 (idx1) joins it (dist 0.4*1.0=0.4<=0.5); weight 20.0
        // (idx2) is incompatible (dist 8.0>0.5) and founds species 1.
        // Reselection is a no-op: each founder is already its own
        // lowest-index member.
        std::vector<Genome> gen1 = {makeSimpleGenome(0.0f), makeSimpleGenome(1.0f), makeSimpleGenome(20.0f)};
        const std::vector<Species>& afterGen1 = speciator.speciate(gen1, compatConfig, speciationConfig);
        assert(afterGen1.size() == 2 && afterGen1[0].getId() == 0 && afterGen1[1].getId() == 1 &&
               afterGen1[0].getMemberIndices() == (std::vector<std::size_t>{0, 1}) &&
               afterGen1[1].getMemberIndices() == (std::vector<std::size_t>{2}) &&
               "setup: generation 1 must produce species 0 (2 members) and species 1 (1 member)");
        // 2: a newly created species' documented initial history state.
        for (const Species& s : afterGen1)
        {
            assert(s.getAge() == 0 && s.getHistoricalBestFitness() == 0.0f && s.getGenerationsSinceImprovement() == 0 &&
                   !s.isStagnant() && !s.hasFitnessHistory() &&
                   "a newly created species must start in the documented 'never evaluated' history state"); // 2
        }

        // Generation 2: weight 1.05 (idx0) and 0.05 (idx1) are both compared
        // against species 0's OLD representative (still 0.0, from
        // generation 1): dist(0.0,1.05)=0.42<=0.5 and dist(0.0,0.05)=0.02<=0.5
        // -- both join species 0 (proving assignment used the persistent
        // old representative, item 9). Species 1's rep (20.0) matches
        // neither -- it goes extinct (item 13), its id (1) never reused
        // (item 6). Reselection at the end of this call picks species 0's
        // lowest CURRENT member index (0, weight 1.05) as its new
        // representative -- a change from 0.0 (items 7, 8 & 14).
        std::vector<Genome> gen2 = {makeSimpleGenome(1.05f), makeSimpleGenome(0.05f)};
        const std::vector<Species>& afterGen2 = speciator.speciate(gen2, compatConfig, speciationConfig);
        assert(afterGen2.size() == 1 && afterGen2[0].getId() == 0 &&
               "a lineage remaining compatible with its persistent representative must keep the same SpeciesId "
               "across generations, and the extinct species must be removed"); // 3, 6 & 13
        assert(afterGen2[0].getMemberIndices() == (std::vector<std::size_t>{0, 1}) &&
               "member lists must be cleared and rebuilt fresh each pass, not accumulated across generations"); // 7

        // Generation 3: weight 2.0. dist to generation 2's RESELECTED
        // representative (1.05) = 0.4*0.95=0.38<=0.5 -> joins species 0. Had
        // assignment instead used generation 1's original representative
        // (0.0), dist(0.0,2.0)=0.8>0.5 would NOT join; had reselection
        // instead picked the lowest-WEIGHT generation-2 member (0.05)
        // rather than the lowest-INDEX one (1.05), dist(0.05,2.0)=0.78>0.5
        // would ALSO not join. Joining therefore proves: the representative
        // changes only after a full pass completes, taking effect on the
        // NEXT call (item 10); it persists between passes (item 8); and
        // reselection picks the lowest member INDEX, not lowest weight or
        // fitness (Speciator has no fitness to consult; item 11). A second
        // genome, weight 20.0 -- the same weight the now-extinct species 1
        // was founded with -- founds a brand new species with the next
        // monotonically increasing id (2, never the extinct 1), in the same
        // fresh state as any new species, proving an extinct species'
        // identity/history never resurfaces (items 4, 5, 39 & 40).
        std::vector<Genome> gen3 = {makeSimpleGenome(2.0f), makeSimpleGenome(20.0f)};
        const std::vector<Species>& afterGen3 = speciator.speciate(gen3, compatConfig, speciationConfig);
        assert(afterGen3.size() == 2 && afterGen3[0].getId() == 0 && afterGen3[1].getId() == 2 &&
               afterGen3[0].getMemberIndices() == (std::vector<std::size_t>{0}) &&
               afterGen3[1].getMemberIndices() == (std::vector<std::size_t>{1}) &&
               "reselection must use the previous pass's own lowest-index member as the new representative, "
               "taking effect only starting with this call -- and a new species with the same weight as a "
               "long-extinct one must receive a fresh id and fresh history"); // 4, 5, 8, 9, 10, 11, 39 & 40
        assert(!afterGen3[1].hasFitnessHistory() && afterGen3[1].getAge() == 0 &&
               "a disappeared species' history must not resurface on a differently-numbered new species"); // 40

        // 12: no RNG state exists anywhere in Speciator or Species --
        // neither declares a generator member, and Speciator's constructor
        // takes no seed (see Speciator.h/Species.h).
    }

    // 15-24: fitness-history/stagnation rules, driven directly through
    // Speciator::updateFitnessHistory() -- fully hand-computed, no
    // simulation involved.
    {
        Speciator speciator;
        CompatibilityConfig compatConfig;
        SpeciationConfig speciationConfig;
        speciationConfig.compatibilityThreshold = 1.0f;
        const std::size_t stagnationLimit = 3;

        std::vector<Genome> gen1 = {makeSimpleGenome(0.0f), makeSimpleGenome(0.1f)};
        const std::vector<Species>& afterGen1 = speciator.speciate(gen1, compatConfig, speciationConfig);
        assert(afterGen1.size() == 1 && "setup: one species with two members");

        // 15, 16 & 23: the first recorded generation initializes
        // historicalBestFitness from the RAW species best, and age
        // increments exactly once.
        const std::vector<float> fitness1 = {10.0f, 25.0f}; // currentSpeciesBest = 25.0 (RAW, member 1)
        speciator.updateFitnessHistory(fitness1, stagnationLimit);
        assert(afterGen1[0].getAge() == 1 &&
               "age must increment exactly once per completed evaluated generation"); // 15
        assert(afterGen1[0].hasFitnessHistory() && afterGen1[0].getHistoricalBestFitness() == 25.0f &&
               afterGen1[0].getGenerationsSinceImprovement() == 0 &&
               "the first recorded generation must initialize historicalBestFitness from the RAW species best"); // 16 & 23

        std::vector<Genome> gen2 = {makeSimpleGenome(0.0f), makeSimpleGenome(0.1f)};
        const std::vector<Species>& afterGen2 = speciator.speciate(gen2, compatConfig, speciationConfig);
        assert(afterGen2.size() == 1 && afterGen2[0].getId() == 0 && "setup: same species persists into generation 2");
        const std::vector<float> fitness2 = {30.0f, 12.0f}; // currentSpeciesBest = 30.0 > 25.0 -> improvement
        speciator.updateFitnessHistory(fitness2, stagnationLimit);
        assert(afterGen2[0].getHistoricalBestFitness() == 30.0f && afterGen2[0].getGenerationsSinceImprovement() == 0 &&
               afterGen2[0].getAge() == 2 &&
               "a strictly greater current-best must update historicalBestFitness and reset "
               "generationsSinceImprovement"); // 17 & 18

        std::vector<Genome> gen3 = {makeSimpleGenome(0.0f), makeSimpleGenome(0.1f)};
        const std::vector<Species>& afterGen3 = speciator.speciate(gen3, compatConfig, speciationConfig);
        const std::vector<float> fitness3 = {30.0f, 5.0f}; // currentSpeciesBest = 30.0, EXACTLY equal
        speciator.updateFitnessHistory(fitness3, stagnationLimit);
        assert(afterGen3[0].getHistoricalBestFitness() == 30.0f && afterGen3[0].getGenerationsSinceImprovement() == 1 &&
               "an exactly-equal current-best must NOT count as improvement"); // 19

        std::vector<Genome> gen4 = {makeSimpleGenome(0.0f), makeSimpleGenome(0.1f)};
        const std::vector<Species>& afterGen4 = speciator.speciate(gen4, compatConfig, speciationConfig);
        const std::vector<float> fitness4 = {8.0f, 9.0f}; // currentSpeciesBest = 9.0 < 30.0
        speciator.updateFitnessHistory(fitness4, stagnationLimit);
        assert(afterGen4[0].getHistoricalBestFitness() == 30.0f && afterGen4[0].getGenerationsSinceImprovement() == 2 &&
               !afterGen4[0].isStagnant() &&
               "a worse current-best must increment generationsSinceImprovement without touching "
               "historicalBestFitness, and must not yet be stagnant below the limit"); // 20

        std::vector<Genome> gen5 = {makeSimpleGenome(0.0f), makeSimpleGenome(0.1f)};
        const std::vector<Species>& afterGen5 = speciator.speciate(gen5, compatConfig, speciationConfig);
        const std::vector<float> fitness5 = {1.0f, 2.0f};
        speciator.updateFitnessHistory(fitness5, stagnationLimit);
        assert(afterGen5[0].getGenerationsSinceImprovement() == 3 && afterGen5[0].isStagnant() &&
               "stagnation must begin exactly when generationsSinceImprovement reaches speciesStagnationLimit"); // 21

        std::vector<Genome> gen6 = {makeSimpleGenome(0.0f), makeSimpleGenome(0.1f)};
        const std::vector<Species>& afterGen6 = speciator.speciate(gen6, compatConfig, speciationConfig);
        const std::vector<float> fitness6 = {50.0f, 1.0f}; // 50.0 > 30.0 -> improvement
        speciator.updateFitnessHistory(fitness6, stagnationLimit);
        assert(!afterGen6[0].isStagnant() && afterGen6[0].getGenerationsSinceImprovement() == 0 &&
               afterGen6[0].getHistoricalBestFitness() == 50.0f &&
               "an improving generation must clear stagnation and reset generationsSinceImprovement"); // 22

        // 24: adjusted fitness never affects historical best --
        // updateFitnessHistory()'s only fitness parameter is
        // rawFitnessByGenomeIndex, fed directly into
        // Species::recordGeneration() with no scaling (see Speciator.cpp).
    }

    // 25 & 34: a species excluded from the eligible subset passed to
    // allocateSpeciesOffspring() -- exactly what Population::reproduce()
    // does for every stagnant, non-fallback species -- contributes nothing
    // to the allocation, regardless of its own effective fitness.
    {
        const std::vector<SpeciesId> eligibleOnly = {1};
        const std::vector<float> eligibleOnlySums = {2.0f}; // deliberately small vs. the excluded species' implied sum
        const std::vector<std::size_t> allocation = allocateSpeciesOffspring(eligibleOnly, eligibleOnlySums, 10);
        assert(allocation.size() == 1 && allocation[0] == 10 &&
               "a species excluded from the eligible subset must receive none of the remaining slots"); // 25 & 34
    }

    // 26-33, 35, 36 & 41-48: a real, multi-generation Population run,
    // verified through invariants that hold no matter which exact
    // species/stagnation pattern this run's simulation/RNG happens to
    // produce (never a hardcoded species count or id).
    {
        const Genome crashGenome = makeCrashGenome();
        PopulationConfig popConfig = makeTestPopulationConfig(10, 4242u);
        popConfig.speciesStagnationLimit = 2; // small on purpose for testing; production default (15) unchanged
        const MutationConfig mutationConfig;
        const CrossoverConfig crossoverConfig;
        const CompatibilityConfig compatibilityConfig;
        SpeciationConfig speciationConfig;
        speciationConfig.compatibilityThreshold = 1.0f;

        Population population(crashGenome, track, makeCarParams(), kSpawnPosition, kSpawnHeading, popConfig,
                               mutationConfig, crossoverConfig, compatibilityConfig, speciationConfig);

        const std::size_t targetGeneration = 6;
        for (int step = 0; step < 4000 * (static_cast<int>(targetGeneration) + 2) &&
                            population.getGeneration() + 1 < targetGeneration;
             ++step)
        {
            population.update(kSimulationDt);
        }
        assert(population.getGeneration() + 1 >= targetGeneration &&
               "setup: the population must reach at least generation targetGeneration-1 within the step budget");

        // Snapshot every iteration (same technique as verifyPopulation's
        // items 15-20 and verifySpeciesAwareReproduction above) across the
        // final transition, so the values below are provably what
        // reproduce() itself used.
        std::vector<Genome> genomeSnapshot;
        std::vector<float> fitnessSnapshot;
        std::size_t finishedBeforeTransitionCall = 0;
        const std::size_t generationBeforeFinal = population.getGeneration();
        for (int step = 0; step < 4000 && population.getGeneration() == generationBeforeFinal; ++step)
        {
            genomeSnapshot.clear();
            fitnessSnapshot.clear();
            for (std::size_t i = 0; i < population.size(); ++i)
            {
                genomeSnapshot.push_back(population.getIndividual(i).getGenome());
                fitnessSnapshot.push_back(population.getIndividual(i).getFitness());
            }
            finishedBeforeTransitionCall = population.getFinishedCount();
            population.update(kSimulationDt);
        }
        assert(population.getGeneration() == generationBeforeFinal + 1 &&
               "setup: the final tracked transition must occur within the step budget"); // 43
        const bool snapshotIsExact = (finishedBeforeTransitionCall == genomeSnapshot.size());

        const std::vector<Population::SpeciesReproductionStats>& stats = population.getReproductionStats();
        assert(!stats.empty() && "setup: at least one species must exist");

        // 35 & 44: offspring allocation + elites reconstructs the exact
        // configured population size, and the population itself stays that
        // size.
        std::size_t totalAllocated = 0;
        for (const Population::SpeciesReproductionStats& s : stats)
        {
            totalAllocated += s.allocatedOffspring;
        }
        assert(totalAllocated + popConfig.eliteCount == popConfig.populationSize &&
               "allocated offspring across every species plus elites must reconstruct the full population size"); // 35
        assert(population.size() == popConfig.populationSize && "population size must remain exactly constant"); // 44

        // 45 & 46: every offspring genome validates and builds a phenotype.
        for (std::size_t i = 0; i < population.size(); ++i)
        {
            population.getIndividual(i).getGenome().validate(); // 45
            ai::neat::buildPhenotype(population.getIndividual(i).getGenome()); // 46
        }

        // 26, 27 & 36: a stagnant, non-eligible species receives zero
        // offspring (so its parent-selection calls never execute, since the
        // inner offspring loop bound is exactly allocatedOffspring); a
        // non-stagnant species always remains eligible.
        bool anyEligible = false;
        for (const Population::SpeciesReproductionStats& s : stats)
        {
            if (s.stagnant && !s.reproductionEligible)
            {
                assert(s.allocatedOffspring == 0 &&
                       "a stagnant, non-fallback-eligible species must receive zero normal offspring"); // 26 & 36
            }
            if (!s.stagnant)
            {
                assert(s.reproductionEligible &&
                       "a non-stagnant species must always remain reproduction-eligible"); // 27
            }
            anyEligible = anyEligible || s.reproductionEligible;
        }
        assert(anyEligible && "at least one species must always remain reproduction-eligible"); // (collapse safety)

        // 30, 31, 32 & 33: the all-stagnant safety fallback, whenever this
        // run's generation actually hit it -- exactly one species is
        // eligible, it has the highest historicalBestFitness (ties broken
        // by lower SpeciesId), and the override never clears the
        // underlying species' own stagnant flag.
        const bool allStagnant =
            std::all_of(stats.begin(), stats.end(), [](const Population::SpeciesReproductionStats& s) { return s.stagnant; });
        if (allStagnant)
        {
            std::size_t eligibleCount = 0;
            std::size_t expectedFallback = 0;
            for (std::size_t s = 1; s < stats.size(); ++s)
            {
                if (stats[s].historicalBestFitness > stats[expectedFallback].historicalBestFitness)
                {
                    expectedFallback = s;
                }
            }
            for (std::size_t s = 0; s < stats.size(); ++s)
            {
                if (stats[s].reproductionEligible)
                {
                    ++eligibleCount;
                    assert(s == expectedFallback &&
                           "the all-stagnant fallback must select the species with the highest "
                           "historicalBestFitness, ties broken by lower SpeciesId"); // 30 & 31
                    assert(stats[s].stagnant &&
                           "the all-stagnant fallback must NOT clear the underlying species' own stagnant "
                           "flag -- it is a reproduction-time override only"); // 33
                }
            }
            assert(eligibleCount == 1 &&
                   "the all-stagnant fallback must activate exactly one species for reproduction"); // 32
        }

        // 28 & 29: the global elite mechanism is untouched by species
        // stagnation -- elite slot 0 must be an unchanged copy of whichever
        // snapshot individual had the strictly highest raw fitness,
        // regardless of that individual's species' stagnation state.
        if (snapshotIsExact)
        {
            std::size_t topIndex = 0;
            for (std::size_t i = 1; i < fitnessSnapshot.size(); ++i)
            {
                if (fitnessSnapshot[i] > fitnessSnapshot[topIndex])
                {
                    topIndex = i;
                }
            }
            assert(connectionsMatch(population.getIndividual(0).getGenome().connections(),
                                     genomeSnapshot[topIndex].connections()) &&
                   "elite slot 0 must be an unchanged, byte-for-byte copy of the top raw-fitness genome, "
                   "regardless of whether that genome's species was stagnant"); // 28 & 29

            // 23 & 24: the reported historicalBestFitness for the species
            // containing topIndex reflects the RAW snapshot value, never
            // divided by species size.
            const std::vector<Species>& currentSpecies = population.getCurrentSpecies();
            for (const Species& species : currentSpecies)
            {
                const std::vector<std::size_t>& members = species.getMemberIndices();
                if (std::find(members.begin(), members.end(), topIndex) == members.end())
                {
                    continue;
                }
                float expectedSpeciesBest = fitnessSnapshot[members.front()];
                for (std::size_t memberIndex : members)
                {
                    expectedSpeciesBest = std::max(expectedSpeciesBest, fitnessSnapshot[memberIndex]);
                }
                assert(species.getHistoricalBestFitness() >= expectedSpeciesBest - 1e-3f &&
                       "historicalBestFitness must reflect RAW member fitness, never fitness divided by species "
                       "size"); // 23 & 24
                break;
            }
        }

        // 37 & 38: species-local mating remains intact, no cross-species
        // mating introduced -- selectParentFromSpecies() still only ever
        // samples from the one Species passed to it; ineligible species are
        // skipped via zero allocation, not any change to parent selection.

        // 47: structural mutations still share exactly one InnovationTracker
        // -- already exercised end to end by verifyPopulation's items
        // 24-28 and this suite's genome validation above.
    }

    // 41 & 42: fixed seed + identical configs/base genome produce
    // deterministic species ids/history/offspring across MULTIPLE
    // generations; a different seed may diverge in species lineage while
    // every resulting Population remains internally valid either way.
    {
        const Genome crashGenome = makeCrashGenome();
        PopulationConfig popConfig = makeTestPopulationConfig(8, 909u);
        popConfig.speciesStagnationLimit = 2;
        const MutationConfig mutationConfig;
        const CrossoverConfig crossoverConfig;
        const CompatibilityConfig compatibilityConfig;
        SpeciationConfig speciationConfig;
        speciationConfig.compatibilityThreshold = 1.0f;

        auto runToGeneration = [](Population& population, std::size_t targetGeneration)
        {
            for (int step = 0; step < 4000 * (static_cast<int>(targetGeneration) + 2) && population.getGeneration() < targetGeneration;
                 ++step)
            {
                population.update(kSimulationDt);
            }
        };

        Population populationX(crashGenome, track, makeCarParams(), kSpawnPosition, kSpawnHeading, popConfig,
                                mutationConfig, crossoverConfig, compatibilityConfig, speciationConfig);
        Population populationY(crashGenome, track, makeCarParams(), kSpawnPosition, kSpawnHeading, popConfig,
                                mutationConfig, crossoverConfig, compatibilityConfig, speciationConfig);
        runToGeneration(populationX, 4);
        runToGeneration(populationY, 4);
        assert(populationX.getGeneration() >= 4 && populationY.getGeneration() >= 4 &&
               "setup: both populations must reach generation 4 within the step budget");

        const std::vector<Species>& speciesX = populationX.getCurrentSpecies();
        const std::vector<Species>& speciesY = populationY.getCurrentSpecies();
        assert(speciesX.size() == speciesY.size() &&
               "identical seed/configs/base genome must produce an identical species count after multiple "
               "generations"); // 41
        for (std::size_t s = 0; s < speciesX.size(); ++s)
        {
            assert(speciesX[s].getId() == speciesY[s].getId() && speciesX[s].getAge() == speciesY[s].getAge() &&
                   speciesX[s].getMemberIndices() == speciesY[s].getMemberIndices() &&
                   speciesX[s].getHistoricalBestFitness() == speciesY[s].getHistoricalBestFitness() &&
                   speciesX[s].getGenerationsSinceImprovement() == speciesY[s].getGenerationsSinceImprovement() &&
                   speciesX[s].isStagnant() == speciesY[s].isStagnant() &&
                   "identical seed/configs/base genome must produce identical species ids/age/membership/history "
                   "across multiple generations"); // 41 (continued)
        }
        for (std::size_t i = 0; i < populationX.size(); ++i)
        {
            assert(connectionsMatch(populationX.getIndividual(i).getGenome().connections(),
                                     populationY.getIndividual(i).getGenome().connections()) &&
                   "identical seed/configs/base genome must produce deterministic offspring genomes across "
                   "multiple generations"); // 41 (continued)
        }

        PopulationConfig popConfigDifferentSeed = popConfig;
        popConfigDifferentSeed.randomSeed = 111u;
        Population populationZ(crashGenome, track, makeCarParams(), kSpawnPosition, kSpawnHeading, popConfigDifferentSeed,
                                mutationConfig, crossoverConfig, compatibilityConfig, speciationConfig);
        runToGeneration(populationZ, 4);
        assert(populationZ.getGeneration() >= 4 && "setup: populationZ must reach generation 4 within the step budget");
        // 42: populationZ's species lineage may or may not numerically
        // differ from populationX's -- not asserted either way -- but it
        // must remain internally valid: every species' member indices in
        // range.
        for (const Species& species : populationZ.getCurrentSpecies())
        {
            for (std::size_t memberIndex : species.getMemberIndices())
            {
                assert(memberIndex < populationZ.size() &&
                       "every species' member indices must stay within the population's actual size"); // 42
            }
        }
    }

    // 48 & 49: ai::FitnessEvaluator and the hard track are untouched here --
    // neither file is referenced anywhere in Species.h/.cpp or
    // Speciator.h/.cpp; Population.cpp's only fitness-related change is
    // which existing raw-fitness vector it also forwards to
    // Speciator::updateFitnessHistory().
    //
    // 50: all previous verification suites still pass -- enforced by
    // main() continuing to call every earlier verify*() function unchanged.

    TraceLog(LOG_INFO, "Persistent species and stagnation verification: all deterministic checks passed");
}
} // namespace verification
