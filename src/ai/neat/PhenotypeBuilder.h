#pragma once

#include "ai/NeuralNetwork.h"
#include "ai/neat/Genome.h"

namespace ai::neat
{

// Converts a validated Genome into an executable ai::NeuralNetwork. Read-only
// on genome; no mutation, crossover, or fitness logic here.
//
// Node ordering: NeuralNetwork assigns Observation/output slots purely by
// constructor order, so nodes are always assembled as Input (ascending ID),
// Bias, Hidden (ascending ID), Output (ascending ID) -- independent of
// Genome insertion order. Observation slot 0 = lowest-ID Input, output slot
// 0 (steering) = lower-ID Output, slot 1 (throttle) = higher-ID Output.
//
// Throws std::invalid_argument if genome.validate() fails, or if
// NeuralNetwork's constructor rejects the result (bad node counts, unknown
// connection endpoint, or a cycle among enabled connections).
ai::NeuralNetwork buildPhenotype(const Genome& genome);

} // namespace ai::neat
