// Regenerates the tiny synthetic PNG fixtures under
// assets/tracks/test/fixtures/ used by verification::verifyImageBasedTrackSystem()
// (see src/verification/TrackVerification.cpp) to exercise Track's
// image-based mask/visual loading: dimension-mismatch rejection, mask-only
// drivability, and mask/visual independence.
//
// Not part of the normal build or any CMake dependency chain -- run this
// on demand only if a fixture needs to be regenerated or its intent
// changed. Determinism: no randomness anywhere in this file; re-running it
// reproduces the exact same pixel content every time.
//
// Usage (from the build directory):
//   cmake --build . --target generate_test_fixtures
//   ./generate_test_fixtures

#include <cstdio>
#include <string>

#include "raylib.h"

namespace
{

void exportRgb(Image& image, const std::string& path)
{
    // 24bpp RGB (no alpha channel), matching what every consumer of these
    // fixtures actually needs -- Track::buildMaskFromImage() normalizes to
    // RGBA8888 on load regardless, so the on-disk channel count is purely
    // cosmetic, not behavioral.
    ImageFormat(&image, PIXELFORMAT_UNCOMPRESSED_R8G8B8);
    if (!ExportImage(image, path.c_str()))
    {
        std::fprintf(stderr, "Failed to write %s\n", path.c_str());
    }
    UnloadImage(image);
}

} // namespace

int main()
{
    const std::string fixturesDir = std::string(OPPARI_ASSETS_DIR) + "/tracks/test/fixtures/";

    // mask_a: 8x8, nearly all-white, except a 2x2 black corner (top-left)
    // that never touches the tiny fixture centerline used by
    // verifyImageBasedTrackSystem().
    {
        Image image = GenImageColor(8, 8, WHITE);
        ImageDrawRectangle(&image, 0, 0, 2, 2, BLACK);
        exportRgb(image, fixturesDir + "mask_a.png");
    }

    // mask_b: same size, DIFFERENT drivable layout (black corner moved to
    // the opposite side) -- used to prove changing the mask alone changes
    // isDrivable() results while the centerline/visual stay fixed.
    {
        Image image = GenImageColor(8, 8, WHITE);
        ImageDrawRectangle(&image, 6, 6, 2, 2, BLACK);
        exportRgb(image, fixturesDir + "mask_b.png");
    }

    // visual_a / visual_b: same size as the fixtures above, arbitrary and
    // clearly different solid colors -- used to prove changing the visual
    // image alone never changes isDrivable().
    {
        Image image = GenImageColor(8, 8, Color{200, 30, 30, 255});
        exportRgb(image, fixturesDir + "visual_a.png");
    }
    {
        Image image = GenImageColor(8, 8, Color{30, 30, 200, 255});
        exportRgb(image, fixturesDir + "visual_b.png");
    }

    // wrong_size: deliberately mismatched dimensions, for dimension-validation tests.
    {
        Image image = GenImageColor(4, 4, Color{128, 128, 128, 255});
        exportRgb(image, fixturesDir + "wrong_size.png");
    }

    std::printf("Test-track fixture assets generated successfully.\n");
    return 0;
}
