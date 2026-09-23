"""Tile geometry and feather weights for tiled upscaling: a deterministic grid
of overlapping tiles blended by weighted accumulation (MultiDiffusion /
Ultimate SD Upscale). Pure numpy.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np


def tile_starts(dim: int, tile: int, overlap: int) -> List[int]:
    """Evenly spaced start offsets along one axis, so overlap is uniform.
    ``[0]`` when ``dim <= tile``. ``overlap`` is clamped to ``tile - 1``, since
    a zero or negative stride divides by zero or yields an empty grid.
    """
    if dim <= tile:
        return [0]
    overlap = min(overlap, tile - 1)
    n = math.ceil((dim - overlap) / (tile - overlap))
    starts = [round(i * (dim - tile) / (n - 1)) for i in range(n)]
    return starts


def tile_grid(
    w: int, h: int, tile: int, overlap: int,
) -> List[Tuple[int, int, int, int]]:
    """Crop boxes ``(x1, y1, x2, y2)`` for every x/y tile start. Each box is
    ``min(tile, w) × min(tile, h)``.
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
    """Feather-weight map of shape ``(th, tw)``: 1-D ramps from
    ``1/(overlap+1)`` to 1 at each edge, combined by outer product. Normalised
    by the per-pixel weight sum, they hide seams.

    Pass the actual per-axis overlap (``tile - stride``); feathering over only
    the requested overlap leaves a flat 50/50 band that blurs detail.
    ``overlap_y`` defaults to ``overlap_x``.
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
