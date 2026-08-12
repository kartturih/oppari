#include "simulation/TrackVisual.h"

#include <stdexcept>
#include <utility>

namespace simulation
{

TrackVisual::TrackVisual(const std::string& imagePath, int expectedWidth, int expectedHeight)
{
    Image image = LoadImage(imagePath.c_str());
    if (image.data == nullptr)
    {
        throw std::invalid_argument("TrackVisual: failed to load visual image asset: " + imagePath);
    }

    if (image.width != expectedWidth || image.height != expectedHeight)
    {
        UnloadImage(image);
        throw std::invalid_argument("TrackVisual: visual image dimensions do not match simulation dimensions: " +
                                     imagePath);
    }

    m_texture = LoadTextureFromImage(image);
    UnloadImage(image);
    m_loaded = true;
}

TrackVisual::~TrackVisual()
{
    unload();
}

TrackVisual::TrackVisual(TrackVisual&& other) noexcept : m_texture(other.m_texture), m_loaded(other.m_loaded)
{
    other.m_loaded = false;
}

TrackVisual& TrackVisual::operator=(TrackVisual&& other) noexcept
{
    if (this != &other)
    {
        unload();
        m_texture = other.m_texture;
        m_loaded = other.m_loaded;
        other.m_loaded = false;
    }
    return *this;
}

void TrackVisual::unload()
{
    if (m_loaded)
    {
        UnloadTexture(m_texture);
        m_loaded = false;
    }
}

void TrackVisual::draw() const
{
    DrawTexture(m_texture, 0, 0, WHITE);
}

} // namespace simulation
