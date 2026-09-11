#pragma once

#include <cstddef>

#include "ai/neat/Population.h"
#include "simulation/Car.h"

// The on-screen side panel (training HUD and manual-control HUD) plus the
// highlighted-individual selection logic that decides which car those
// panels/overlays describe.
namespace ui
{

// Chooses which individual gets its sensors/velocity vector rendered this
// frame: the running individual with the highest best-progress, or -- once
// all have finished -- the highest-fitness individual. Ties break to the
// lower index. Pure query: reads population, mutates nothing.
std::size_t selectHighlightedIndividual(const ai::neat::Population& population);

void drawPopulationPanel(const ai::neat::Population& population, std::size_t highlightedIndex, int panelWidth,
                          int screenHeight);

// HUD for manual single-car control mode -- shows the live vehicle-state
// values (speed/forward/lateral velocity/slip angle) that Observation also
// reads (see ai::buildObservation). Purely a rendering helper -- reads Car
// state only, never mutates it.
void drawManualPanel(const simulation::Car& car, int panelWidth, int screenHeight);

// Small always-on-top overlay (top-left of the simulation view, over the
// track) showing the current Normal/Fast training-speed mode and, once
// measured, the achieved simulated-seconds-per-wall-second multiplier.
// Purely a rendering helper -- reads nothing but the values passed in,
// never touches Population/Car state. simSpeedMultiplierValid is false
// until at least one full measurement window has completed since the last
// mode switch (see main.cpp) -- never fabricate a multiplier before then.
void drawTrainingSpeedHud(bool fastMode, float simSpeedMultiplier, bool simSpeedMultiplierValid);

} // namespace ui
