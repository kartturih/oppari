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

    // Observation only (never read back by fitness/control): one 60 Hz sample
    // of how this step was driven.
    m_diagnostics.update(input.steering, input.brake, m_car.getSpeed(), m_car.getYawRate(),
                         m_car.getTireDebugInfo().frontSlipAngle, m_car.getParams().frontPeakSlipAngle,
                         m_progress.getLapCount(), deltaTime);
    const Observation& observation = m_controller.getLastObservation();
    m_diagnostics.recordLongitudinal(m_controller.getThrottleRequest(), m_controller.getBrakeRequest(), input.brake,
                                     m_car.getSpeed(), observation.values[kPreviewNearObservationIndex],
                                     observation.values[kPreviewFarObservationIndex]);
}

void Individual::reset()
{
    m_car.reset(m_spawnPosition, m_spawnHeading);
    m_progress.reset(m_car);
    m_fitness.reset();
    m_diagnostics.reset();
}

} // namespace ai::neat
