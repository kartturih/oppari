#!/usr/bin/env python3
"""Stage 18 deterministic test-track asset generator.

Produces track_visual.png and track_mask.png (plus small fixtures/ images
used only by main.cpp's Stage 18 verification suite) for the image-based
track architecture. This script is NOT part of the CMake build -- the PNGs
it writes are committed as static assets, and this script exists purely so
that generation is reproducible and auditable.

IMPORTANT -- single source of truth coupling:
The centerline control points, CONTROL_POINTS, and SAMPLES_PER_SEGMENT below
MUST exactly match simulation::createStage18TestTrackDefinition() in
src/simulation/Track.cpp (same values, same order, same uniform Catmull-Rom
formula). Track.cpp's centerline is what Car/TrackProgress actually use at
runtime; this script only needs to match it so that track_mask.png's
drivable band actually contains that real centerline (Stage 18 requires the
centerline to remain entirely inside the drivable mask). If either side
changes the control points or sample density, update the other to match.

Determinism: no randomness anywhere in this file. Re-running it produces
byte-identical PNGs.
"""

import os
import numpy as np
from PIL import Image

SIM_WIDTH = 1200
SIM_HEIGHT = 700
SAMPLES_PER_SEGMENT = 24

# Must match simulation::createStage18TestTrackDefinition()'s control points
# exactly (same values reused from Stage 14B's hard track geometry, which is
# already verified non-self-intersecting with generous separation between
# non-adjacent sections -- see Track.cpp's verifyTrack()/createHardTrackDefinition()
# commentary. Reusing it here is deliberate: it lets Stage 18's test track
# inherit that already-proven centerline shape instead of re-deriving one).
CONTROL_POINTS = [
    (250.0, 580.0),  # P0 -- spawn
    (500.0, 580.0),  # P1
    (850.0, 580.0),  # P2
    (1000.0, 540.0), # P3
    (1080.0, 400.0), # P4
    (1020.0, 220.0), # P5
    (880.0, 110.0),  # P6
    (620.0, 170.0),  # P7
    (400.0, 100.0),  # P8
    (270.0, 160.0),  # P9
    (150.0, 320.0),  # P10
    (50.0, 550.0),   # P11
]

# Per-control-point-segment half-width (px) of the DRIVABLE MASK band. This
# is authored directly into the mask image, not derived from any single
# TrackDefinition.trackWidth constant -- Stage 18's whole point is that the
# mask may vary its width along the track. Segment i runs from control point
# i to control point (i+1) % 12; half-width is linearly interpolated between
# halfWidth[i] and halfWidth[(i+1) % 12] across that segment's samples, so
# the width transitions smoothly at every control point (matching the
# centerline's own smooth Catmull-Rom transition).
#
#   seg 0,1  (P0->P1->P2)   : 65   -- ordinary straight
#   seg 2    (P2->P3)       : 75   -- easing into the sweep
#   seg 3,4  (P3->P4->P5)   : 95   -- WIDE: broad sweeping corner
#   seg 5    (P5->P6)       : 70   -- easing into the chicane
#   seg 6,7  (P6->P7->P8)   : 45   -- NARROW: S-chicane
#   seg 8    (P8->P9)       : 55   -- tight corner 1
#   seg 9    (P9->P10)      : 65   -- descent
#   seg 10   (P10->P11)     : 60   -- lead-in to tight corner 2
#   seg 11   (P11->P0)      : 65   -- closing straight
HALF_WIDTHS = [65, 65, 75, 95, 95, 70, 45, 45, 55, 65, 60, 65]

# The VISUAL image intentionally uses a single constant width, different
# from every varying mask width above (except where interpolation happens
# to cross it) -- this is the concrete, visible proof that the visual layer
# is not derived from the mask layer (or vice versa): the painted road you
# see and the area a car can actually drive on are simply two independent
# images.
VISUAL_HALF_WIDTH = 55.0


def catmull_rom(p0, p1, p2, p3, t):
    """Uniform Catmull-Rom, identical formula to Track.cpp's catmullRom()."""
    t2 = t * t
    t3 = t2 * t
    x = 0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t +
               (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
               (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
    y = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t +
               (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
               (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
    return (x, y)


def build_centerline():
    n = len(CONTROL_POINTS)
    centerline = []
    half_widths = []
    for i in range(n):
        p0 = CONTROL_POINTS[(i - 1) % n]
        p1 = CONTROL_POINTS[i]
        p2 = CONTROL_POINTS[(i + 1) % n]
        p3 = CONTROL_POINTS[(i + 2) % n]
        hw0 = HALF_WIDTHS[i]
        hw1 = HALF_WIDTHS[(i + 1) % n]
        for j in range(SAMPLES_PER_SEGMENT):
            t = j / SAMPLES_PER_SEGMENT
            centerline.append(catmull_rom(p0, p1, p2, p3, t))
            half_widths.append(hw0 + (hw1 - hw0) * t)
    return centerline, half_widths


def rasterize_band(centerline, half_widths_per_sample):
    """Vectorized: for every pixel, drivable iff within the LOCAL half-width
    of the nearest point on the nearest centerline segment. half_widths_per_sample
    may be a single float (constant width) or a per-sample list (varying width)."""
    n = len(centerline)
    xs = np.arange(SIM_WIDTH, dtype=np.float32)
    ys = np.arange(SIM_HEIGHT, dtype=np.float32)
    px, py = np.meshgrid(xs, ys)  # shape (H, W)

    drivable = np.zeros((SIM_HEIGHT, SIM_WIDTH), dtype=bool)

    const_width = not isinstance(half_widths_per_sample, (list, np.ndarray))

    for i in range(n):
        ax, ay = centerline[i]
        bx, by = centerline[(i + 1) % n]
        abx, aby = bx - ax, by - ay
        length_sq = abx * abx + aby * aby
        if length_sq > 0.0:
            t = ((px - ax) * abx + (py - ay) * aby) / length_sq
            t = np.clip(t, 0.0, 1.0)
        else:
            t = np.zeros_like(px)
        projx = ax + abx * t
        projy = ay + aby * t
        dx = px - projx
        dy = py - projy
        dist = np.sqrt(dx * dx + dy * dy)

        if const_width:
            hw = half_widths_per_sample
        else:
            hw0 = half_widths_per_sample[i]
            hw1 = half_widths_per_sample[(i + 1) % n]
            hw = hw0 + (hw1 - hw0) * t

        drivable |= dist <= hw

    return drivable


def count_connected_components(mask_bool):
    """Simple flood-fill connected-component count (4-connectivity) over the
    foreground (True) pixels, used only as a generation-time sanity check
    (not part of the C++ build) -- catches accidental band self-overlap or
    disconnection before the asset is committed."""
    visited = np.zeros_like(mask_bool, dtype=bool)
    h, w = mask_bool.shape
    components = 0
    stack = []
    for y in range(h):
        for x in range(w):
            if mask_bool[y, x] and not visited[y, x]:
                components += 1
                stack.append((y, x))
                visited[y, x] = True
                while stack:
                    cy, cx = stack.pop()
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if 0 <= ny < h and 0 <= nx < w and mask_bool[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
    return components


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    fixtures_dir = os.path.join(out_dir, "fixtures")
    os.makedirs(fixtures_dir, exist_ok=True)

    centerline, half_widths = build_centerline()

    # ---- track_mask.png : varying-width drivable band -------------------
    mask_bool = rasterize_band(centerline, half_widths)

    components = count_connected_components(mask_bool)
    assert components == 1, f"expected exactly 1 connected drivable region, got {components}"

    # Every centerline sample itself must be drivable (Stage 18 requirement).
    for (cx, cy) in centerline:
        ix, iy = int(round(cx)), int(round(cy))
        assert mask_bool[iy, ix], f"centerline sample ({cx},{cy}) is NOT drivable in the generated mask"

    # The middle of the closed loop must be far enough from every section
    # to be non-drivable (mirrors main.cpp's verifyTrack() check #19-21).
    cx_mid, cy_mid = SIM_WIDTH // 2, SIM_HEIGHT // 2
    assert not mask_bool[cy_mid, cx_mid], "simulation center must be non-drivable"

    min_hw = min(half_widths)
    max_hw = max(half_widths)
    print(f"drivable band half-width range: {min_hw:.1f}px .. {max_hw:.1f}px "
          f"(full width {2*min_hw:.0f}px .. {2*max_hw:.0f}px) -- varying, not constant")

    mask_img = np.where(mask_bool[..., None], 255, 0).astype(np.uint8)
    mask_rgb = np.repeat(mask_img, 3, axis=2)
    Image.fromarray(mask_rgb, mode="RGB").save(os.path.join(out_dir, "track_mask.png"))

    # ---- track_visual.png : constant-width painted road, independent art -
    visual_bool = rasterize_band(centerline, VISUAL_HALF_WIDTH)

    # Background: flat green field.
    visual = np.zeros((SIM_HEIGHT, SIM_WIDTH, 3), dtype=np.uint8)
    visual[..., 0] = 34
    visual[..., 1] = 120
    visual[..., 2] = 34

    # Road surface: flat gray, painted independently of the mask band.
    visual[visual_bool] = (90, 90, 90)

    # Dashed white centerline stripe, purely decorative -- drawn along the
    # exact same centerline samples, every third sample, as a thin square.
    for idx in range(0, len(centerline), 3):
        cx, cy = centerline[idx]
        ix, iy = int(round(cx)), int(round(cy))
        for oy in range(-1, 2):
            for ox in range(-1, 2):
                yy, xx = iy + oy, ix + ox
                if 0 <= yy < SIM_HEIGHT and 0 <= xx < SIM_WIDTH:
                    visual[yy, xx] = (235, 235, 235)

    Image.fromarray(visual, mode="RGB").save(os.path.join(out_dir, "track_visual.png"))

    # ---- fixtures for main.cpp's Stage 18 verification suite ------------
    # 8x8: nearly all-white, except a 2x2 black corner that never touches
    # the tiny fixture centerline used by that suite (see main.cpp).
    fx = np.full((8, 8, 3), 255, dtype=np.uint8)
    fx[0:2, 0:2] = (0, 0, 0)
    Image.fromarray(fx, mode="RGB").save(os.path.join(fixtures_dir, "mask_a.png"))

    # mask_b: same size, DIFFERENT drivable layout (black corner moved to
    # the opposite side) -- used to prove changing the mask alone changes
    # isDrivable() results while the centerline/visual stay fixed.
    fx_b = np.full((8, 8, 3), 255, dtype=np.uint8)
    fx_b[6:8, 6:8] = (0, 0, 0)
    Image.fromarray(fx_b, mode="RGB").save(os.path.join(fixtures_dir, "mask_b.png"))

    # visual_a / visual_b: same size as the fixtures above, arbitrary and
    # clearly different solid colors -- used to prove changing the visual
    # image alone never changes isDrivable().
    Image.fromarray(np.full((8, 8, 3), (200, 30, 30), dtype=np.uint8), mode="RGB").save(
        os.path.join(fixtures_dir, "visual_a.png"))
    Image.fromarray(np.full((8, 8, 3), (30, 30, 200), dtype=np.uint8), mode="RGB").save(
        os.path.join(fixtures_dir, "visual_b.png"))

    # wrong_size: deliberately mismatched dimensions, for dimension-validation tests.
    Image.fromarray(np.full((4, 4, 3), (128, 128, 128), dtype=np.uint8), mode="RGB").save(
        os.path.join(fixtures_dir, "wrong_size.png"))

    print("Stage 18 test track assets generated successfully.")


if __name__ == "__main__":
    main()
