#pragma once

#include <string>

#include "raylib.h"

namespace simulation
{

// Owns the GPU texture for a track's visual background image. Requires an
// active raylib window/GL context, so it's always constructed after the
// Track it renders for. Purely for rendering -- collision/progress never
// depend on it.
class TrackVisual
{
public:
    // Throws std::invalid_argument if the file can't load, or its
    // dimensions don't match (expectedWidth, expectedHeight).
    TrackVisual(const std::string& imagePath, int expectedWidth, int expectedHeight);

    ~TrackVisual();

    TrackVisual(const TrackVisual&) = delete;
    TrackVisual& operator=(const TrackVisual&) = delete;
    TrackVisual(TrackVisual&& other) noexcept;
    TrackVisual& operator=(TrackVisual&& other) noexcept;

    // Draws at (0,0), unscaled.
    void draw() const;

private:
    void unload();

    Texture2D m_texture{};
    bool m_loaded = false;
};

} // namespace simulation
