"""Detection + region geometry for the detailer (ADetailer-style refinement).

ADetailer hooks deep into A1111's UNet processing, so it can't drive a DiT. This
module keeps only the model-agnostic half — YOLO detection and the crop/expand
math — as pure functions. ``Engine.detail`` pairs them with diffucore's ``Inpaint``
pipeline, which runs on UNet (SD/SDXL) *and* DiT (Anima, FLUX) backbones, so the
same detail pass works across all of them.

The crop math (``get_crop_region`` / ``expand_crop_region``) is ported from
AUTO1111's ``modules/masking.py``.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

BBox = List[float]

# Cache YOLO models by path — reloading the .pt from disk on every detailer pass
# (and per stacked pass) is wasteful.
_YOLO_CACHE: dict = {}


def detect_regions(
    detector_path: str, image: Image.Image, confidence: float = 0.3,
) -> List[Tuple[BBox, float]]:
    """Run a YOLO detector and return ``[(xyxy_bbox, confidence), …]`` sorted
    largest-area first. Lazy-imports ultralytics so the app loads without it."""
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError(
            "Detailer needs ultralytics — `pip install ultralytics`"
        ) from e

    model = _YOLO_CACHE.get(detector_path)
    if model is None:
        model = YOLO(detector_path)
        _YOLO_CACHE[detector_path] = model
    pred = model(image, conf=confidence, verbose=False)
    boxes = pred[0].boxes
    if boxes is None or boxes.xyxy.shape[0] == 0:
        return []
    bboxes = boxes.xyxy.cpu().numpy().tolist()
    confs = boxes.conf.cpu().numpy().tolist()
    order = sorted(range(len(bboxes)), key=lambda i: -_area(bboxes[i]))
    return [(bboxes[i], confs[i]) for i in order]


def _area(b: BBox) -> float:
    return (b[2] - b[0]) * (b[3] - b[1])


def bbox_to_mask(bbox: BBox, size: Tuple[int, int]) -> Image.Image:
    """White (255) rectangle over ``bbox`` on a black ``size`` (W, H) canvas."""
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rectangle(bbox, fill=255)
    return mask


def dilate_mask(mask: Image.Image, value: int) -> Image.Image:
    """Grow the white region by ``value`` px (no-op for value <= 0)."""
    if value <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (value, value))
    arr = cv2.dilate(np.array(mask), kernel, iterations=1)
    return Image.fromarray(arr)


def get_crop_region(mask: Image.Image, pad: int = 0):
    """Smallest box containing the mask's white pixels, padded by ``pad`` and
    clamped to the image. Returns ``(x1, y1, x2, y2)`` or ``None`` if all black.
    Ported from AUTO1111 ``masking.get_crop_region_v2``."""
    box = mask.getbbox()
    if box is None:
        return None
    if not pad:
        return box
    x1, y1, x2, y2 = box
    w, h = mask.size
    return (max(x1 - pad, 0), max(y1 - pad, 0), min(x2 + pad, w), min(y2 + pad, h))


def expand_crop_region(crop_region, processing_width, processing_height,
                       image_width, image_height):
    """Expand a crop region to match the processing aspect ratio (so the inpaint
    doesn't squish), keeping it inside the image. Ported verbatim from AUTO1111
    ``masking.expand_crop_region``."""
    x1, y1, x2, y2 = crop_region

    ratio_crop_region = (x2 - x1) / (y2 - y1)
    ratio_processing = processing_width / processing_height

    if ratio_crop_region > ratio_processing:
        desired_height = (x2 - x1) / ratio_processing
        desired_height_diff = int(desired_height - (y2 - y1))
        y1 -= desired_height_diff // 2
        y2 += desired_height_diff - desired_height_diff // 2
        if y2 >= image_height:
            diff = y2 - image_height
            y2 -= diff
            y1 -= diff
        if y1 < 0:
            y2 -= y1
            y1 -= y1
        if y2 >= image_height:
            y2 = image_height
    else:
        desired_width = (y2 - y1) * ratio_processing
        desired_width_diff = int(desired_width - (x2 - x1))
        x1 -= desired_width_diff // 2
        x2 += desired_width_diff - desired_width_diff // 2
        if x2 >= image_width:
            diff = x2 - image_width
            x2 -= diff
            x1 -= diff
        if x1 < 0:
            x2 -= x1
            x1 -= x1
        if x2 >= image_width:
            x2 = image_width

    return x1, y1, x2, y2
