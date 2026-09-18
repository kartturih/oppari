// Read-only diagnostic for the Bias->Brake evolvability experiment: builds
// the exact same deterministic Population main.cpp does (same seed/configs/
// base genome) and, at chosen generation boundaries, prints individual 0's
// and the best individual's full connection list -- with anything touching
// the Brake output (node 102) or a Hidden node called out -- so their real
// genome wiring can be inspected directly instead of inferred from
// behavior alone.
//
// Not part of the normal build or any CMake dependency chain (same
// EXCLUDE_FROM_ALL pattern as generate_test_fixtures.cpp) -- run this on
// demand only. No window/GPU needed: Track's own construction is CPU-only
// (see Track.h), and Population never touches raylib's window/render state.
//
// Usage (from the build directory):
//   cmake --build . --target inspect_brake_genomes
//   ./inspect_brake_genomes

#include <cstdio>
#include <string>
#include <vector>

#include "ai/NeuralNetwork.h"
#include "ai/Observation.h"
#include "ai/neat/CompatibilityConfig.h"
#include "ai/neat/CrossoverConfig.h"
#include "ai/neat/Genome.h"
#include "ai/neat/Individual.h"
#include "ai/neat/MutationConfig.h"
#include "ai/neat/Population.h"
#include "ai/neat/PopulationConfig.h"
#include "ai/neat/SpeciationConfig.h"
#include "simulation/Track.h"

#include "AppConfig.h"

namespace
{

// Node IDs 0..(kInputCount-1) are Observation's Input slots (see
// Observation.h), kInputCount itself is Bias, 100/101/102 are the fixed
// Steering/Throttle/Brake Output IDs (see AppConfig.cpp's
// createDemonstrationGenome()), and anything else is a Hidden node born from
// mutation.
std::string describeNode(int id)
{
    if (id >= 0 && id < ai::NeuralNetwork::kInputCount)
    {
        static const char* inputNames[ai::NeuralNetwork::kInputCount] = {
            "Sensor-60",     "Sensor-30",     "Sensor0",       "Sensor+30",   "Sensor+60",
            "SpeedNorm",     "ForwardVelNorm", "LateralVelNorm", "SlipNorm",
            "ActualSteerNorm", "YawRateNorm",  "HeadingErrorNorm"};
        return inputNames[id];
    }
    if (id == ai::NeuralNetwork::kInputCount) return "Bias";
    if (id == 100) return "Steering(out)";
    if (id == 101) return "Throttle(out)";
    if (id == 102) return "Brake(out)";
    return "Hidden" + std::to_string(id);
}

void printGenomeBrakeWiring(const char* label, const ai::neat::Genome& genome)
{
    std::printf("  [%s]\n", label);
    bool any = false;
    for (const ai::neat::ConnectionGene& c : genome.connections())
    {
        const bool touchesBrake = (c.getTargetId() == 102 || c.getSourceId() == 102);
        const bool touchesHidden = (c.getSourceId() >= 103 || c.getTargetId() >= 103);
        if (touchesBrake || touchesHidden)
        {
            std::printf("    %s -> %s : weight=%.4f enabled=%d%s%s\n", describeNode(c.getSourceId()).c_str(),
                        describeNode(c.getTargetId()).c_str(), c.getWeight(), c.isEnabled() ? 1 : 0,
                        touchesBrake ? " [BRAKE]" : "", touchesHidden ? " [HIDDEN]" : "");
            any = true;
        }
    }
    if (!any)
    {
        std::printf("    (no Brake- or Hidden-touching connections at all)\n");
    }

    std::printf("    full connection list:\n");
    for (const ai::neat::ConnectionGene& c : genome.connections())
    {
        std::printf("      %s -> %s : weight=%.4f enabled=%d\n", describeNode(c.getSourceId()).c_str(),
                    describeNode(c.getTargetId()).c_str(), c.getWeight(), c.isEnabled() ? 1 : 0);
    }
}

} // namespace

int main()
{
    simulation::Track track(app::makeTrackDefinition());

    const ai::neat::PopulationConfig populationConfig;
    const ai::neat::MutationConfig mutationConfig;
    const ai::neat::CrossoverConfig crossoverConfig;
    const ai::neat::CompatibilityConfig compatibilityConfig;
    const ai::neat::SpeciationConfig speciationConfig;

    ai::neat::Population population(app::createDemonstrationGenome(), track, app::makeCarParams(), app::kSpawnPosition,
                                     app::kSpawnHeading, populationConfig, mutationConfig, crossoverConfig,
                                     compatibilityConfig, speciationConfig);

    // Generations at which to dump wiring -- chosen from the telemetry pass
    // to bracket each observed brake-behavior regime change (0=base;
    // 8=first CONSTANT elite; 15=second, different CONSTANT elite; 19/20=
    // VARYING elite).
    const std::vector<std::size_t> targetGenerations = {0, 8, 15, 19, 20};
    std::size_t nextTargetIdx = 0;

    std::size_t lastGeneration = population.getGeneration();
    if (!targetGenerations.empty() && targetGenerations[0] == 0)
    {
        std::printf("=== Generation 0 ===\n");
        printGenomeBrakeWiring("individual 0", population.getIndividual(0).getGenome());
        const std::size_t bestIdx = population.getBestIndividualIndex();
        if (bestIdx != 0)
        {
            printGenomeBrakeWiring("best individual (differs from 0)", population.getIndividual(bestIdx).getGenome());
        }
        ++nextTargetIdx;
    }

    while (nextTargetIdx < targetGenerations.size())
    {
        population.update(app::kSimulationDt);
        if (population.getGeneration() != lastGeneration)
        {
            lastGeneration = population.getGeneration();
            if (lastGeneration == targetGenerations[nextTargetIdx])
            {
                std::printf("=== Generation %zu ===\n", lastGeneration);
                printGenomeBrakeWiring("individual 0", population.getIndividual(0).getGenome());
                const std::size_t bestIdx = population.getBestIndividualIndex();
                if (bestIdx != 0)
                {
                    printGenomeBrakeWiring("best individual (differs from 0)", population.getIndividual(bestIdx).getGenome());
                }
                ++nextTargetIdx;
            }
        }
    }

    return 0;
}
