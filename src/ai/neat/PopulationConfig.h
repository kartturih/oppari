#pragma once

#include <cstddef>
#include <cstdint>

namespace ai::neat
{

// Top-level tunables for Population; validated by Population itself.
struct PopulationConfig
{
    std::size_t populationSize = 50;
    std::size_t eliteCount = 2;
    std::size_t tournamentSize = 3;

    // Consecutive stagnant generations before a species is excluded from
    // reproduction (see Species::recordGeneration()).
    std::size_t speciesStagnationLimit = 15;

    std::uint32_t randomSeed = 12345;
};

} // namespace ai::neat
