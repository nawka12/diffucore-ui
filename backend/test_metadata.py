"""Tests for the PNG metadata round-trip (pure functions, no GPU needed).

Run from the project root::

    .venv/bin/python -m pytest backend/test_metadata.py -v
"""

from __future__ import annotations

import metadata as md


class _StubEngine:
    """Minimal stand-in for the fields ``format_metadata`` reads off the engine."""
    loaded_name = "anima-test"
    last_seed = 2150942283
    perf_flags_str = "default"


def _roundtrip(gen_kwargs: dict, **kw) -> dict:
    params = md.format_metadata(gen_kwargs, _StubEngine(), **kw)
    return md.workspace_fields(md.parse_metadata(params))


_BASE_GEN = {
    "prompt": "a fox", "negative_prompt": "blurry",
    "steps": 32, "sampler": "euler", "scheduler": "flow",
    "cfg_scale": 5.0, "width": 1024, "height": 1536,
}


def test_upscale_roundtrips_through_metadata():
    """An upscaled image's settings survive format -> parse -> workspace_fields."""
    upscale = {
        "scale": 4.0, "tile": 1024, "overlap": 128, "denoise": 0.25,
        "teacache": 0.1, "base": "4x-UltraSharp.pth", "prompt": "a fox, sharp",
    }
    up = _roundtrip(_BASE_GEN, upscale=upscale)["upscale"]
    assert up["enabled"] is True
    assert up["scale"] == 4.0
    assert up["tile"] == 1024
    assert up["overlap"] == 128
    assert up["denoise"] == 0.25
    assert up["teacache"] == 0.1
    assert up["base"] == "4x-UltraSharp.pth"
    assert up["prompt"] == "a fox, sharp"   # comma survives the quote/unquote path


def test_lanczos_base_maps_to_form_default():
    """Lanczos (no ESRGAN model) round-trips to the form's empty base."""
    up = md.extract_upscale({"upscale_scale": "2.0", "upscale_base": "Lanczos"})
    assert up["base"] == ""


def test_no_upscale_meta_leaves_fields_untouched():
    """An image generated without the upscaler yields no ``upscale`` key."""
    assert "upscale" not in _roundtrip(_BASE_GEN)


def test_teacache_calibrated_roundtrips():
    fields = _roundtrip({**_BASE_GEN, "teacache_thresh": 0.15, "teacache_use_coeffs": True})
    assert fields["teacacheOn"] is True
    assert fields["teacache"] == 0.15
    assert fields["teacacheCalibrated"] is True


def test_teacache_raw_roundtrips():
    """The "(raw)" suffix restores the calibrated toggle as off."""
    fields = _roundtrip({**_BASE_GEN, "teacache_thresh": 0.3, "teacache_use_coeffs": False})
    assert fields["teacacheOn"] is True
    assert fields["teacache"] == 0.3
    assert fields["teacacheCalibrated"] is False


def test_no_teacache_meta_leaves_toggle_untouched():
    """TeaCache off writes no line, so nothing is restored (additive)."""
    fields = _roundtrip(_BASE_GEN)
    assert "teacacheOn" not in fields
    assert "teacacheCalibrated" not in fields
