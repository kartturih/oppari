#pragma once

#include "ai/NeuralNetwork.h"
#include "ai/Observation.h"
#include "simulation/Car.h"

namespace ai
{

// Closes the control loop for exactly one car:
//
//   Car -> Observation -> NeuralNetwork -> AIController -> CarInput
//
// AIController owns its NeuralNetwork by value and knows nothing about NEAT
// genetics (Genome/NodeGene/ConnectionGene) -- it only consumes an
// already-built phenotype, produced elsewhere (see PhenotypeBuilder). Car
// has no knowledge of this class: it is driven purely through CarInput,
// exactly like manual keyboard control. NeuralNetwork has no knowledge of
// Car: it only ever consumes an Observation.
//
// This class performs no mutation, crossover, training, or any other
// evolutionary behavior -- it is a pure runtime adapter.
class AIController
{
public:
    explicit AIController(NeuralNetwork network);

    // Builds an Observation from `car` (via ai::buildObservation, so sensor
    // casting and local-state derivation stay solely Car's responsibility),
    // evaluates the owned network, maps its two outputs to a CarInput (see
    // NeuralNetwork output-ordering / mapping documented in AIController.cpp),
    // and returns it. Does not mutate `car`.
    //
    // If the car is not alive, the network is not evaluated (nothing left to
    // usefully react to) and a neutral CarInput (zero throttle, zero
    // steering) is returned; getRawSteeringOutput()/getRawThrottleOutput()/
    // getLastObservation() keep reporting the values from the last live
    // evaluation.
    simulation::CarInput update(const simulation::Car& car);

    // Network outputs from the most recent evaluating update() call
    // (i.e. one where the car was alive), before output mapping.
    float getRawSteeringOutput() const { return m_rawSteering; }
    float getRawThrottleOutput() const { return m_rawThrottle; }

    // The Observation built during the most recent evaluating update() call.
    // Returns a default (all-zero) Observation if update() has never run
    // while the car was alive.
    const Observation& getLastObservation() const { return m_lastObservation; }

private:
    NeuralNetwork m_network;
    Observation m_lastObservation;
    float m_rawSteering = 0.0f;
    float m_rawThrottle = 0.0f;
};

} // namespace ai
