#include "ai/neat/Individual.h"

#include <utility>

#include "ai/neat/PhenotypeBuilder.h"

namespace ai::neat
{

Individual::Individual(Genome genome, const simulation::Track& track, const simulation::CarParams& carParams,
                        Vector2 spawnPosition, float spawnHeading)
    : m_genome(std::move(genome))
    , m_spawnPosition(spawnPosition)
    , m_spawnHeading(spawnHeading)
    , m_car(carParams, track)
    , m_controller(buildPhenotype(m_genome))
    , m_progress(track)
    , m_fitness()
{
    reset();
}

void Individual::update(float deltaTime)
{
    if (isFinished())
    {
        return;
    }

    const simulation::CarInput input = m_controller.update(m_car, m_progress);
    m_car.update(input, deltaTime);
    m_progress.update(m_car);
    m_fitness.update(m_car, m_progress, input.steering, deltaTime);
}

void Individual::reset()
{
    m_car.reset(m_spawnPosition, m_spawnHeading);
    m_progress.reset(m_car);
    m_fitness.reset();
}

} // namespace ai::neat
