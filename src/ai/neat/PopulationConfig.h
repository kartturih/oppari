#pragma once

#include <cstddef>
#include <cstdint>

namespace ai::neat
{

// Top-level tunables for Population. Pure data -- it performs no
// validation itself; Population validates every field (throwing
// std::invalid_argument on an invalid one) before using them. Deliberately
// small: everything mutation/crossover/compatibility/speciation-specific
// stays in its own existing config (MutationConfig, CrossoverConfig,
// CompatibilityConfig, SpeciationConfig), passed to Population separately
// rather than duplicated here.
struct PopulationConfig
{
    std::size_t populationSize = 50;
    std::size_t eliteCount = 2;
    std::size_t tournamentSize = 3;

    std::uint32_t randomSeed = 12345;
};

} // namespace ai::neat
