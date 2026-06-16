"""The VRAM -> offload auto-set (Engine.recommended_offload). Pure logic, no GPU.

Run from the project root::

    .venv/bin/python -m pytest backend/test_engine_offload.py -v
"""
from __future__ import annotations

import pytest
import torch

from engine import Engine


def _engine_with_vram(monkeypatch, gb):
    eng = Engine(device="cpu")           # don't require a real CUDA device
    eng.device = torch.device("cuda")    # exercise the cuda branch of the picker

    class _Props:
        total_memory = int(gb * 1024**3)

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *_a, **_k: _Props())
    return eng


@pytest.mark.parametrize("gb,expected", [
    (24, "none"), (23, "none"),
    (16, "encoders"), (12, "encoders"), (11, "encoders"),
    (10, "full"), (8, "full"), (6, "full"),
    (5, "stream"), (4, "stream"), (3, "stream"),   # ≤6 GB auto-picks stream
])
def test_recommended_offload_tiers(monkeypatch, gb, expected):
    assert _engine_with_vram(monkeypatch, gb).recommended_offload() == expected


def test_recommended_offload_cpu_is_full():
    eng = Engine(device="cpu")
    eng.device = torch.device("cpu")
    assert eng.recommended_offload() == "full"


# ── compile vs. backbone-moving offload coercion (the load-time guard) ──────────
# The guard runs before the file checks, so a missing-file load still exercises it:
# we catch the expected FileNotFoundError and inspect the staging the engine baked in.

@pytest.mark.parametrize("call", [
    lambda e: e.load_model("does-not-exist", offload="stream", compile=True, cuda_graphs=True),
    lambda e: e.load_anima("nope", "nope", "nope", offload="stream", compile=True, cuda_graphs=True),
    lambda e: e.load_flux("nope", "nope", "nope", offload="stream", compile=True, cuda_graphs=True),
])
def test_compile_with_stream_drops_compile(call):
    # stream can't downgrade to "encoders" (would OOM the small card it was picked
    # for), so compile is dropped instead — and cuda_graphs with it (it needs compile).
    eng = Engine(device="cpu")
    with pytest.raises(FileNotFoundError):
        call(eng)
    assert eng._offload == "stream"
    assert eng._compile is False
    assert eng._cuda_graphs is False


def test_compile_with_full_still_coerces_to_encoders():
    # Regression: the pre-existing True/"full" path keeps the backbone resident
    # (downgrade to "encoders") and keeps compile on.
    eng = Engine(device="cpu")
    with pytest.raises(FileNotFoundError):
        eng.load_model("does-not-exist", offload=True, compile=True)
    assert eng._offload == "encoders"
    assert eng._compile is True
