"""Tests for the base-image cache and the standalone detailer endpoint.

The cache lets a second Generate click skip re-sampling when only the post-gen
passes (upscaler / detailer) changed. These drive ``_run_generation`` against a
stub engine — no GPU, no model.

Run from the project root::

    .venv/bin/python -m pytest backend/test_base_cache.py -v
"""

from __future__ import annotations

import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from engine import Engine
import server


# ── stub engine ─────────────────────────────────────────────────────

class _StubEngine:
    """Minimal stand-in for ``engine.ENGINE``: counts base generations."""

    def __init__(self):
        self.loaded_name = "stub.safetensors"
        self.loaded_family = "anima"
        self.can_inpaint = True
        self.weights_epoch = 0
        self.last_seed = -1
        self.gen_calls = 0
        self.upscale_calls = 0
        self.detail_calls = 0
        self.applied = []

    # Real parser: the LoRA tags it strips are exactly what the fingerprint
    # has to key separately.
    parse_lora_prompt = staticmethod(Engine.parse_lora_prompt)

    def apply_temp_loras(self, loras):
        self.applied = list(loras)
        return "loras"

    def clear_temp_loras(self):
        self.applied = []

    def _make(self, tint):
        return Image.new("RGB", (64, 64), tint)

    def generate_t2i(self, **kwargs):
        self.gen_calls += 1
        seed = kwargs["seed"]
        self.last_seed = 4242 if seed == -1 else seed
        return self._make((self.gen_calls, 0, 0)), f"gen#{self.gen_calls}"

    generate_i2i = generate_inpaint = generate_t2i

    def upscale(self, image, **kwargs):
        self.upscale_calls += 1
        return image.resize((128, 128)), "upscaled"

    def detail(self, image, **kwargs):
        self.detail_calls += 1
        return image.copy(), "Detailer: 1 region"


@pytest.fixture
def stub(monkeypatch):
    eng = _StubEngine()
    monkeypatch.setattr(server, "ENGINE", eng)
    monkeypatch.setattr(server, "_save_output",
                        lambda *a, **k: server.OUTPUTS_DIR / "stub.png")
    server._BASE_CACHE.clear()
    yield eng
    server._BASE_CACHE.clear()


def _run(**overrides):
    p = server.GeneratePayload(**overrides)
    return server._run_generation(p, lambda *a: None, None)


# ── the cache ───────────────────────────────────────────────────────

def test_post_pass_toggle_reuses_the_base(stub):
    """The whole point: enabling the upscaler must not re-sample the base."""
    _run(prompt="a cat", seed=7)
    assert stub.gen_calls == 1

    _run(prompt="a cat", seed=7, upscale_enabled=True, upscale_scale=2.0)
    assert stub.gen_calls == 1, "the base was re-sampled"
    assert stub.upscale_calls == 1


def test_retuning_a_post_pass_keeps_reusing(stub):
    """The real saving is the tweak loop, not the single toggle."""
    _run(prompt="a cat", seed=7)
    for strength in (0.3, 0.5, 0.7):
        _run(prompt="a cat", seed=7, detail_enabled=True,
             detail_models=[{"model": "face.pt", "prompt": ""}],
             detail_strength=strength)
    assert stub.gen_calls == 1
    assert stub.detail_calls == 3


@pytest.mark.parametrize("field,value", [
    ("prompt", "a dog"),
    ("neg", "blurry"),
    ("steps", 30),
    ("cfg", 7.5),
    ("width", 512),
    ("height", 512),
    ("sampler", "euler"),
    ("scheduler", "beta"),
    ("shift", 4.0),
    ("teacache", 0.2),
    ("deepcache", 2),
    ("seed", 8),
])
def test_changing_the_base_re_samples(stub, field, value):
    base = dict(prompt="a cat", seed=7)
    _run(**base)
    _run(**{**base, field: value})
    assert stub.gen_calls == 2, f"{field} must invalidate the base"


def test_random_seed_never_reuses(stub):
    """seed == -1 is a request for a *new* image."""
    _run(prompt="a cat", seed=-1)
    _run(prompt="a cat", seed=-1)
    assert stub.gen_calls == 2


def test_random_seed_stores_under_the_seed_it_resolved(stub):
    """Recycling the seed of a random run then hits, so the '♻ then upscale'
    workflow costs one generation, not two."""
    r = _run(prompt="a cat", seed=-1)
    assert r["seed"] == 4242
    _run(prompt="a cat", seed=4242, upscale_enabled=True)
    assert stub.gen_calls == 1


def test_weights_change_invalidates(stub):
    """A model swap or a LoRA change bumps the epoch; the old base is dead."""
    _run(prompt="a cat", seed=7)
    stub.weights_epoch += 1
    _run(prompt="a cat", seed=7)
    assert stub.gen_calls == 2


def test_cache_holds_one_entry(stub):
    _run(prompt="a cat", seed=7)
    _run(prompt="a dog", seed=7)
    assert len(server._BASE_CACHE) == 1


def test_reuse_does_not_hand_out_the_cached_object(stub):
    """A ``post_generate`` extension may draw on the image it is given, so the
    cache must never expose the entry it holds."""
    _run(prompt="a cat", seed=7)
    held = next(iter(server._BASE_CACHE.values()))[0]
    _run(prompt="a cat", seed=7, upscale_enabled=True)
    still = next(iter(server._BASE_CACHE.values()))[0]
    assert still is held
    assert still.size == (64, 64), "the upscale leaked into the cached base"


def test_reused_run_reports_the_locked_seed(stub):
    """``last_seed`` is stale on a hit (no generation ran), so the result and
    its metadata must take the seed from the payload."""
    _run(prompt="a cat", seed=7)
    stub.last_seed = 999           # e.g. another device generated in between
    r = _run(prompt="a cat", seed=7, upscale_enabled=True)
    assert r["seed"] == 7


def test_i2i_source_is_part_of_the_key(stub):
    def b64(color):
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), color).save(buf, "PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    _run(mode="i2i", prompt="a cat", seed=7, input_image=b64((10, 10, 10)))
    _run(mode="i2i", prompt="a cat", seed=7, input_image=b64((10, 10, 10)))
    assert stub.gen_calls == 1, "the same source image should reuse"
    _run(mode="i2i", prompt="a cat", seed=7, input_image=b64((200, 10, 10)))
    assert stub.gen_calls == 2, "a different source image must re-sample"


def test_settings_panel_knob_invalidates(stub, monkeypatch):
    """Sampler knobs live in SETTINGS, not the payload, but they change the
    base — the key is built from the kwargs the engine actually receives, so
    they are covered without being listed anywhere."""
    monkeypatch.setitem(server.SETTINGS, "gate_reduce", "all")
    _run(prompt="a cat", seed=7, sampler="cogent")
    monkeypatch.setitem(server.SETTINGS, "gate_reduce", "per_channel")
    _run(prompt="a cat", seed=7, sampler="cogent")
    assert stub.gen_calls == 2


# ── fingerprint ─────────────────────────────────────────────────────

def test_fingerprint_ignores_the_callbacks(stub):
    a = server._base_fingerprint(
        {"prompt": "x", "seed": 1, "progress_callback": lambda *a: None}, "t2i", [])
    b = server._base_fingerprint(
        {"prompt": "x", "seed": 1, "progress_callback": print,
         "preview_callback": print}, "t2i", [])
    assert a == b


def test_fingerprint_separates_modes(stub):
    kw = {"prompt": "x", "seed": 1}
    assert (server._base_fingerprint(kw, "t2i", [])
            != server._base_fingerprint(kw, "i2i", []))


def test_fingerprint_keys_the_lora_set(stub):
    """The engine gets the *stripped* prompt, so the LoRA tags have to be keyed
    on their own or two different weight sets would share an entry."""
    kw = {"prompt": "x", "seed": 1}
    assert (server._base_fingerprint(kw, "t2i", [("a", 1.0)])
            != server._base_fingerprint(kw, "t2i", [("b", 1.0)]))
    assert (server._base_fingerprint(kw, "t2i", [("a", 1.0)])
            != server._base_fingerprint(kw, "t2i", [("a", 0.5)]))
    assert (server._base_fingerprint(kw, "t2i", [("a", 1.0), ("b", 0.5)])
            == server._base_fingerprint(kw, "t2i", [("b", 0.5), ("a", 1.0)]))


# ── LoRA prompts ────────────────────────────────────────────────────

def test_lora_prompt_still_reuses(stub):
    """Temp LoRAs are applied before and cleared after every run. If that churn
    bumped the weights epoch, a LoRA prompt would miss the cache every time."""
    prompt = "a cat <lora:style:0.8>"
    _run(prompt=prompt, seed=7)
    _run(prompt=prompt, seed=7, upscale_enabled=True)
    assert stub.gen_calls == 1


def test_changing_only_the_lora_re_samples(stub):
    _run(prompt="a cat <lora:style:0.8>", seed=7)
    _run(prompt="a cat <lora:other:0.8>", seed=7)
    assert stub.gen_calls == 2
    _run(prompt="a cat <lora:other:0.4>", seed=7)
    assert stub.gen_calls == 3


# ── the engine contract the cache key rests on ──────────────────────

class _FakeLoaded:
    """Enough of a ``LoadedModel`` for the cond-cache helpers and ``_unload``."""
    def __init__(self):
        self.model = type("Bundle", (), {})()
        self.applied_loras = []


def _engine_with_fake_model():
    eng = Engine()
    eng._loaded = _FakeLoaded()
    return eng


def test_temp_lora_cycle_does_not_move_the_epoch():
    """``apply_temp_loras``/``clear_temp_loras`` bracket every LoRA generation.
    If they moved the epoch, the server's base cache would miss on every run
    whose prompt carries a ``<lora:…>`` tag."""
    eng = _engine_with_fake_model()
    before = eng.weights_epoch
    eng._invalidate_cond_cache()
    eng._invalidate_cond_cache()
    assert eng.weights_epoch == before


def test_load_and_unload_move_the_epoch():
    eng = _engine_with_fake_model()
    before = eng.weights_epoch
    eng._attach_cond_cache()             # every load/restore ends in this
    assert eng.weights_epoch == before + 1
    eng._unload()
    assert eng.weights_epoch == before + 2


# ── standalone detailer ─────────────────────────────────────────────

def test_detail_route_rejects_an_empty_model_stack():
    """The stack is filtered for real detector names before the job is queued,
    so an empty pick fails fast instead of occupying the worker."""
    with TestClient(server.app) as c:
        r = c.post("/api/detail", json={"input_image": "", "models": []})
        assert r.status_code == 400
        r = c.post("/api/detail", json={"input_image": "",
                                        "models": [{"model": "(none)"}]})
        assert r.status_code == 400


def test_detail_payload_bounds():
    with pytest.raises(ValidationError):
        server.DetailPayload(strength=2.0)
    with pytest.raises(ValidationError):
        server.DetailPayload(steps=0)
    assert server.DetailPayload(models=[{"model": "face.pt"}]).confidence == 0.3
