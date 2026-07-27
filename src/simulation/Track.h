#pragma once

#include <cstdint>
#include <vector>

#include "raylib.h"

namespace simulation
{

// Describes an oval track: a simulation-area size plus the outer and inner
// ellipse radii that carve out the drivable road band.
struct TrackDefinition
{
    int simWidth = 0;
    int simHeight = 0;

    Vector2 center = {0.0f, 0.0f};

    float outerRadiusX = 0.0f;
    float outerRadiusY = 0.0f;

    float innerRadiusX = 0.0f;
    float innerRadiusY = 0.0f;
};

// Holds an oval track's definition and a precomputed CPU-side drivable mask
// derived from that definition. Collision queries only ever read the mask;
// the ellipse geometry is evaluated once, during construction.
class Track
{
public:
    // Throws std::invalid_argument if the definition is not a valid oval
    // (non-positive sizes/radii, or the inner ellipse not smaller than the
    // outer ellipse).
    explicit Track(const TrackDefinition& definition);

    // Returns false for coordinates outside the simulation area without
    // touching the mask.
    bool isDrivable(int x, int y) const;

    const TrackDefinition& getDefinition() const { return m_definition; }

private:
    void buildMask();

    TrackDefinition m_definition;
    std::vector<std::uint8_t> m_mask;
};

} // namespace simulation
