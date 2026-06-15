"""Tests for the tiled upscaler (pure functions, no GPU needed).

Run from the project root::

    .venv/bin/python -m pytest backend/test_upscale.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from upscale import feather_weights, tile_grid, tile_starts


# ── tile_starts ────────────────────────────────────────────────────

def test_tile_starts_single_when_smaller():
    """When dim <= tile, a single start at 0 is returned."""
    assert tile_starts(512, 1024, 128) == [0]
    assert tile_starts(1024, 1024, 128) == [0]


def test_tile_starts_covers_full_extent():
    """The last start is ``dim - tile`` so the whole canvas is covered."""
    starts = tile_starts(2048, 1024, 128)
    assert starts[0] == 0
    assert starts[-1] == 2048 - 1024


def test_tile_starts_uniform_spacing():
    """Consecutive-start deltas are within 1 px of uniform."""
    starts = tile_starts(2048, 1024, 128)
    deltas = [starts[i + 1] - starts[i] for i in range(len(starts) - 1)]
    expected = (2048 - 1024) / (len(starts) - 1) if len(starts) > 1 else 0
    for d in deltas:
        assert abs(d - expected) <= 1.0


def test_tile_starts_degenerate():
    """dim == tile + 1 yields two tiles: [0, 1]."""
    starts = tile_starts(1025, 1024, 128)
    assert len(starts) == 2
    assert starts[0] == 0
    assert starts[-1] == 1025 - 1024


# ── tile_grid ──────────────────────────────────────────────────────

def test_tile_grid_full_coverage():
    """Union of all tile boxes equals the full canvas area."""
    w, h, tile, overlap = 1536, 2048, 1024, 128
    boxes = tile_grid(w, h, tile, overlap)
    covered = np.zeros((h, w), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        covered[y1:y2, x1:x2] = True
    assert covered.all(), "Every pixel is covered by at least one tile"


def test_tile_grid_all_tiles_full_size():
    """Every tile is exactly ``tile × tile`` when canvas >= tile."""
    w, h, tile, overlap = 2048, 2048, 1024, 128
    boxes = tile_grid(w, h, tile, overlap)
    for x1, y1, x2, y2 in boxes:
        assert x2 - x1 == tile
        assert y2 - y1 == tile


def test_tile_grid_single_tile():
    """A single tile when canvas <= tile."""
    boxes = tile_grid(800, 600, 1024, 128)
    assert len(boxes) == 1
    x1, y1, x2, y2 = boxes[0]
    assert x1 == 0 and y1 == 0
    assert x2 == 800 and y2 == 600


def test_tile_grid_square_aspect():
    """tile_grid produces product = len(xs) * len(ys) boxes."""
    w, h, tile, overlap = 2048, 1024, 1024, 128
    boxes = tile_grid(w, h, tile, overlap)
    xs = tile_starts(w, tile, overlap)
    ys = tile_starts(h, tile, overlap)
    assert len(boxes) == len(xs) * len(ys)


# ── feather_weights ──────────────────────────────────────────────

def test_feather_weights_shape():
    w = feather_weights(64, 48, 8)
    assert w.shape == (48, 64)


def test_feather_weights_range():
    w = feather_weights(64, 48, 8)
    assert w.min() > 0.0
    assert w.max() <= 1.0


def test_feather_weights_no_overlap():
    w = feather_weights(64, 48, 0)
    assert np.allclose(w, 1.0)


def test_feather_weights_peaks_at_center():
    w = feather_weights(64, 64, 16)
    cy, cx = 32, 32
    assert w[cy, cx] == 1.0, "Center should be full weight"


def test_feather_weights_zero_at_edges():
    """Edge pixels get min weight = 1/(overlap+1), never zero."""
    w = feather_weights(64, 64, 16)
    assert w[0, 0] > 0.0
    assert w[-1, -1] > 0.0


def test_feather_weights_corner_ne_zero():
    """Corner pixel gets (1/(overlap+1))² from the outer product, not zero."""
    w = feather_weights(64, 64, 16)
    assert w[0, 0] > 0.0
    expected = 1.0 / (16 + 1) ** 2
    assert abs(w[0, 0] - expected) < 1e-4, (
        f"Corner weight should be {expected:.6f}, got {w[0, 0]:.6f}"
    )


def test_feather_spans_actual_overlap_no_hard_seam():
    """When the actual tile overlap exceeds the requested value (few tiles span
    the axis), feathering over the *actual* overlap avoids a wide flat 50/50
    band that would average divergent tile detail into blur.

    Reproduces the reported blur: a 2x of 1024 → 2048 packs 3 tiles/axis with a
    512px overlap; a 128px ramp leaves ~260px of hard 50/50 averaging per seam.
    """
    starts = tile_starts(2048, 1024, 128)
    ov = 1024 - (starts[1] - starts[0])
    assert ov == 512, "2x of 1024 packs 3 tiles → 512px actual overlap"

    def hard_band(overlap: int) -> int:
        row = feather_weights(1024, 1024, overlap)[512]   # 1-D weight along x
        w0, w1 = row[512:1024], row[0:512]                 # tile0 right / tile1 left
        return int(np.sum((w0 > 0.98) & (w1 > 0.98)))

    assert hard_band(128) > 200, "the old (requested-overlap) ramp blurs a wide band"
    assert hard_band(ov) == 0, "feathering the actual overlap removes the hard band"


# ── blend reconstruction ────────────────────────────────────────────

def test_blend_constant_reconstructed():
    """Two overlapping constant-colour tiles reconstruct the constant."""
    from upscale import tile_grid, feather_weights
    w, h, tile, overlap = 1200, 1200, 1024, 128
    boxes = tile_grid(w, h, tile, overlap)
    acc = np.zeros((h, w, 3), dtype=np.float64)
    wsum = np.zeros((h, w, 1), dtype=np.float64)
    colour = np.array([0.5, 0.6, 0.7], dtype=np.float64)
    for x1, y1, x2, y2 in boxes:
        tw, th = x2 - x1, y2 - y1
        fw = feather_weights(tw, th, overlap)[..., None]
        tile_arr = np.ones((th, tw, 3), dtype=np.float64) * colour
        acc[y1:y2, x1:x2] += tile_arr * fw
        wsum[y1:y2, x1:x2] += fw
    result = acc / np.clip(wsum, 1e-6, None)
    err = np.abs(result - colour).max()
    assert err < 1e-10, f"Max error for constant blend: {err}"


def test_blend_gradient_seamless():
    """A smooth vertical gradient split across tiles reconstructs without
    step artefacts at tile boundaries."""
    from upscale import tile_grid, feather_weights
    w, h, tile, overlap = 1200, 1200, 1024, 128
    boxes = tile_grid(w, h, tile, overlap)
    acc = np.zeros((h, w, 3), dtype=np.float64)
    wsum = np.zeros((h, w, 1), dtype=np.float64)
    # Build a reference gradient
    yy, xx = np.mgrid[:h, :w]
    gradient = np.stack([xx / w, yy / h, (xx + yy) / (w + h)], axis=-1)
    for x1, y1, x2, y2 in boxes:
        tw, th = x2 - x1, y2 - y1
        fw = feather_weights(tw, th, overlap)[..., None]
        tile_arr = gradient[y1:y2, x1:x2]
        acc[y1:y2, x1:x2] += tile_arr * fw
        wsum[y1:y2, x1:x2] += fw
    result = acc / np.clip(wsum, 1e-6, None)
    # Check neighbour deltas across the tile seam
    ys = tile_starts(h, tile, overlap)
    for y_seam in ys[1:]:
        if y_seam < h:
            delta = np.abs(result[y_seam] - result[y_seam - 1]).max()
            assert delta < 0.1, (
                f"Seam at y={y_seam}: max neighbour delta {delta:.4f}"
            )
    xs = tile_starts(w, tile, overlap)
    for x_seam in xs[1:]:
        if x_seam < w:
            delta = np.abs(result[:, x_seam] - result[:, x_seam - 1]).max()
            assert delta < 0.1, (
                f"Seam at x={x_seam}: max neighbour delta {delta:.4f}"
            )
