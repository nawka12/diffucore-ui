"""Tile geometry + feather-weight helpers for tiled upscaling.

Unlike the detailer (which uses YOLO boxes), the upscaler covers the whole
canvas with a deterministic grid of overlapping tiles and blends them back
with a weighted-accumulate composite — the MultiDiffusion/Ultimate SD Upscale
approach. Pure numpy — no model, no PyTorch.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np


def tile_starts(dim: int, tile: int, overlap: int) -> List[int]:
    """Evenly-spaced start offsets along one axis so overlap is uniform.

    ``n = ceil((dim - overlap) / (tile - overlap))``, then starts are
    ``round(i * (dim - tile) / (n - 1))``. Returns ``[0]`` when
    ``dim <= tile`` (single tile that may be smaller than ``tile``).
    """
    if dim <= tile:
        return [0]
    n = math.ceil((dim - overlap) / (tile - overlap))
    starts = [round(i * (dim - tile) / (n - 1)) for i in range(n)]
    return starts


def tile_grid(
    w: int, h: int, tile: int, overlap: int,
) -> List[Tuple[int, int, int, int]]:
    """Product of x/y tile starts → crop boxes ``(x1, y1, x2, y2)``.

    Every box is ``min(tile, w) × min(tile, h)``.  When the canvas is larger
    than ``tile``, the last start is ``dim - tile`` so edge tiles are also
    exactly ``tile²``.
    """
    xs = tile_starts(w, tile, overlap)
    ys = tile_starts(h, tile, overlap)
    boxes: list[tuple[int, int, int, int]] = []
    for y in ys:
        for x in xs:
            boxes.append((x, y, min(x + tile, w), min(y + tile, h)))
    return boxes


def feather_weights(
    tw: int, th: int, overlap_x: int, overlap_y: int | None = None,
) -> np.ndarray:
    """Feather-weight map of shape ``(th, tw)`` (float32, ``[0, 1]``).

    Outer product of 1-D ramps that rise from ``1/(overlap+1)`` to ``1``
    over ``overlap`` pixels at each edge, and stay at ``1`` in the center.
    When overlapping tiles are accumulated with these weights and divided by
    the per-pixel weight sum (MultiDiffusion normalisation), seams disappear
    — even at the true canvas edge where only one tile contributes.

    ``overlap_x``/``overlap_y`` should be the *actual* per-axis tile overlap
    (``tile - stride``), which can exceed the requested overlap when only a few
    tiles span an axis. Feathering over the full overlap avoids a wide flat
    50/50 band that would average divergent tile detail into blur. ``overlap_y``
    defaults to ``overlap_x``.
    """
    if overlap_y is None:
        overlap_y = overlap_x

    def _ramp(length: int, overlap: int) -> np.ndarray:
        r = np.ones(length, dtype=np.float32)
        o = min(overlap, length // 2)
        if o <= 0:
            return r
        vals = np.linspace(0, 1, o + 2)[1:-1]
        r[:o] = vals
        r[-o:] = vals[::-1]
        return r

    rx = _ramp(tw, overlap_x)
    ry = _ramp(th, overlap_y)
    return np.outer(ry, rx)
