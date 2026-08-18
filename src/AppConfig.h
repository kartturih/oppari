#pragma once

#include <string>

#include "raylib.h"

#include "ai/neat/Genome.h"
#include "simulation/Car.h"
#include "simulation/Track.h"

// Application-wide constants/helpers shared between the real runtime setup
// in main.cpp and the entire verification suite (src/verification/), so
// both build the exact same Track/Car/demonstration Genome.
namespace app
{

constexpr int kSimWidth = 1200;
constexpr int kSimHeight = 700;

// One fixed simulation step per rendered frame -- never GetFrameTime(), so
// Car always sees the same dt regardless of render duration.
constexpr float kSimulationDt = 1.0f / 60.0f;

// Absolute path baked in at compile time (OPPARI_ASSETS_DIR), so asset
// loading is independent of the executable's working directory.
std::string assetPath(const std::string& relativePath);

// The active track: the user-authored extreme track. This is the only place
// the active track is chosen -- see createHardTrackDefinition() for the
// other available (unused by normal training) track.
simulation::TrackDefinition makeTrackDefinition();

// Spawn pose derived from the Track itself, shared by every verify*()
// function and every Population so they all use the same deterministic pose.
extern const Vector2 kSpawnPosition;
extern const float kSpawnHeading;

simulation::CarParams makeCarParams();

// Hand-built, deterministic demonstration Genome (not trained/evolved): 9
// Input nodes (0-8, matching Observation slot order), 1 Bias, 3 Output
// (100 = steering, 101 = throttle, 102 = brake), direct Input/Bias -> Output
// only. Left sensors steer negative, right sensors steer positive; bias +
// center sensor drive throttle; brake starts wired off (see the definition).
ai::neat::Genome createDemonstrationGenome();

} // namespace app
