#pragma once

#include "ai/NeuralNetwork.h"
#include "ai/neat/Genome.h"

namespace ai::neat
{

// Converts one validated Genome into one executable ai::NeuralNetwork. This
// is phenotype construction only -- no mutation, crossover, innovation
// tracking, fitness, or any other evolutionary behavior happens here. The
// Genome is read-only input: buildPhenotype() never modifies it, and keeps
// no reference to it once it returns.
//
// Node ordering contract
// -----------------------
// ai::NeuralNetwork determines which Observation slot feeds which Input
// node, and which output slot is steering vs. throttle, purely from the
// order nodes are passed to its constructor (see NeuralNetwork.h) -- it has
// no notion of node ID ordering. To make that order a deterministic
// function of the Genome's node IDs, independent of the order genes happen
// to be stored in the Genome, buildPhenotype() always assembles the runtime
// node list in this fixed category order:
//
//   1. Input nodes,  ascending node ID
//   2. Bias node
//   3. Hidden nodes, ascending node ID
//   4. Output nodes, ascending node ID
//
// Consequences: Observation index 0 always maps to the lowest-ID Input
// node, index 8 to the highest-ID Input node; output slot 0 (steering)
// always maps to the lower-ID Output node, slot 1 (throttle) to the
// higher-ID Output node. This holds regardless of the order the
// corresponding NodeGenes were added to the Genome.
//
// Throws std::invalid_argument if genome.validate() rejects the genome, or
// if the resulting node/connection set is rejected by the NeuralNetwork
// constructor (wrong Input/Bias/Output counts, a connection referencing an
// unknown node, or a cycle among enabled connections -- NeuralNetwork
// supports feed-forward DAGs only).
ai::NeuralNetwork buildPhenotype(const Genome& genome);

} // namespace ai::neat
