#pragma once

#include "ai/NeuralNetwork.h"
#include "ai/Observation.h"
#include "simulation/Car.h"

namespace ai
{

// Closes the control loop: Car -> Observation -> NeuralNetwork -> CarInput.
// Owns its NeuralNetwork; knows nothing about NEAT genetics. Pure runtime
// adapter -- no mutation/crossover/training.
class AIController
{
public:
    explicit AIController(NeuralNetwork network);

    // Builds an Observation from car, evaluates the network, maps outputs to
    // a CarInput. If the car is dead, returns a neutral CarInput without
    // evaluating; getRawThrottleOutput()/getRawBrakeOutput()/
    // getLastObservation() keep reporting the last live evaluation.
    simulation::CarInput update(const simulation::Car& car);

    // Raw network outputs from the most recent live update(), before mapping.
    float getRawThrottleOutput() const { return m_rawThrottle; }
    float getRawBrakeOutput() const { return m_rawBrake; }

    // Observation from the most recent live update() (default if none yet).
    const Observation& getLastObservation() const { return m_lastObservation; }

private:
    NeuralNetwork m_network;
    Observation m_lastObservation;
    float m_rawSteering = 0.0f;
    float m_rawThrottle = 0.0f;
    float m_rawBrake = 0.0f;
};

} // namespace ai
