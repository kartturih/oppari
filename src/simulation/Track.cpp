#include "simulation/Track.h"

#include <stdexcept>

namespace simulation
{

namespace
{

void validate(const TrackDefinition& def)
{
    if (def.simWidth <= 0 || def.simHeight <= 0)
    {
        throw std::invalid_argument("TrackDefinition: simWidth/simHeight must be positive");
    }

    if (def.outerRadiusX <= 0.0f || def.outerRadiusY <= 0.0f ||
        def.innerRadiusX <= 0.0f || def.innerRadiusY <= 0.0f)
    {
        throw std::invalid_argument("TrackDefinition: ellipse radii must be positive");
    }

    if (def.innerRadiusX >= def.outerRadiusX || def.innerRadiusY >= def.outerRadiusY)
    {
        throw std::invalid_argument("TrackDefinition: inner ellipse must be smaller than outer ellipse");
    }
}

} // namespace

Track::Track(const TrackDefinition& definition)
    : m_definition(definition)
{
    validate(m_definition);
    buildMask();
}

void Track::buildMask()
{
    const int width = m_definition.simWidth;
    const int height = m_definition.simHeight;

    m_mask.assign(static_cast<std::size_t>(width) * static_cast<std::size_t>(height), 0);

    const float cx = m_definition.center.x;
    const float cy = m_definition.center.y;

    const float outerXSq = m_definition.outerRadiusX * m_definition.outerRadiusX;
    const float outerYSq = m_definition.outerRadiusY * m_definition.outerRadiusY;
    const float innerXSq = m_definition.innerRadiusX * m_definition.innerRadiusX;
    const float innerYSq = m_definition.innerRadiusY * m_definition.innerRadiusY;

    for (int y = 0; y < height; ++y)
    {
        const float dy = static_cast<float>(y) - cy;
        const float dySq = dy * dy;

        for (int x = 0; x < width; ++x)
        {
            const float dx = static_cast<float>(x) - cx;
            const float dxSq = dx * dx;

            const float outerValue = (dxSq / outerXSq) + (dySq / outerYSq);
            const float innerValue = (dxSq / innerXSq) + (dySq / innerYSq);

            const bool insideOuter = outerValue <= 1.0f;
            const bool outsideInner = innerValue >= 1.0f;

            m_mask[static_cast<std::size_t>(y) * width + x] = (insideOuter && outsideInner) ? 1 : 0;
        }
    }
}

bool Track::isDrivable(int x, int y) const
{
    if (x < 0 || y < 0 || x >= m_definition.simWidth || y >= m_definition.simHeight)
    {
        return false;
    }

    const std::size_t index = static_cast<std::size_t>(y) * static_cast<std::size_t>(m_definition.simWidth) +
                               static_cast<std::size_t>(x);

    return m_mask[index] != 0;
}

} // namespace simulation
