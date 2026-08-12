#pragma once

#include <string>

#include "raylib.h"

namespace simulation
{

// Owns the GPU texture for a track's visual background image -- the
// topmost of Stage 18's three independent layers (see Track.h for the
// collision-mask and centerline layers, neither of which know this class
// exists). Loading a GPU texture requires an active raylib window/GL
// context, unlike Track's own CPU-only image handling (LoadImage(), never
// LoadTexture()) -- so a TrackVisual is always constructed separately from,
// and after, the Track it renders for, once a window exists. It is never
// read by Car, sensors, or TrackProgress: this class exists purely for
// rendering, and nothing about collision or progress can depend on it.
class TrackVisual
{
public:
    // Loads and uploads the texture at imagePath. Throws std::invalid_argument
    // if the file cannot be loaded, or if its dimensions do not match
    // (expectedWidth, expectedHeight). Track already performs this same
    // validation (CPU-only, via LoadImage) on this same path while
    // constructing the Track this visual belongs to -- this is an
    // independent re-check, not a reuse of that result, so a TrackVisual
    // can never silently end up out of sync with its Track.
    TrackVisual(const std::string& imagePath, int expectedWidth, int expectedHeight);

    ~TrackVisual();

    // Copying would double-free the underlying GPU texture; only moving is
    // allowed, which transfers ownership and leaves the source empty.
    TrackVisual(const TrackVisual&) = delete;
    TrackVisual& operator=(const TrackVisual&) = delete;
    TrackVisual(TrackVisual&& other) noexcept;
    TrackVisual& operator=(TrackVisual&& other) noexcept;

    // Draws the texture at (0,0), unscaled -- callers are responsible for
    // making sure the simulation area they draw it into matches
    // (expectedWidth, expectedHeight) passed at construction.
    void draw() const;

private:
    void unload();

    Texture2D m_texture{};
    bool m_loaded = false;
};

} // namespace simulation
