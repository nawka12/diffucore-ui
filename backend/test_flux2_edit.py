"""FLUX.2 reference-image Edit mode: payload validation, dispatch, base-cache
fingerprinting, metadata and the engine wrapper. No model is loaded.

    .venv/bin/python -m pytest backend/test_flux2_edit.py -v
"""

from __future__ import annotations

import base64
import io
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import engine as engine_mod
import metadata as md
import server
from engine import Engine, LoadedModel, SCHEDULER_GROUPS, SCHEDULERS_FLUX


def _b64(color, size=(64, 64)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class _StubEngine:
    """Records the edit call; everything else the run reads is inert."""

    def __init__(self, family="flux2"):
        self.loaded_name = "flux-2-klein-4b.safetensors"
        self.loaded_family = family
        self.can_inpaint = True
        self.weights_epoch = 0
        self.last_seed = 5
        self.edit_calls = []

    parse_lora_prompt = staticmethod(Engine.parse_lora_prompt)

    def apply_temp_loras(self, loras):
        return "loras"

    def clear_temp_loras(self):
        pass

    def generate_edit(self, **kwargs):
        self.edit_calls.append(kwargs)
        self.last_seed = kwargs["seed"]
        return Image.new("RGB", (64, 64), (len(self.edit_calls), 0, 0)), "edit-info"

    def generate_t2i(self, **kwargs):
        self.edit_calls.append(kwargs)
        return Image.new("RGB", (64, 64)), "t2i-info"


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


# ── validation ──────────────────────────────────────────────────────

def test_payload_accepts_edit_and_defaults_to_no_references():
    p = server.GeneratePayload(mode="edit", prompt="x")
    assert p.mode == "edit" and p.reference_images == []


def test_edit_without_a_reference_is_rejected(stub):
    with pytest.raises(RuntimeError, match="Provide a reference image"):
        _run(mode="edit", prompt="make it red")
    assert stub.edit_calls == []


def test_edit_rejects_more_than_four_references(stub):
    with pytest.raises(RuntimeError, match="at most 4"):
        _run(mode="edit", prompt="x", reference_images=[_b64((i, 0, 0)) for i in range(5)])
    assert stub.edit_calls == []


def test_edit_takes_exactly_four(stub):
    _run(mode="edit", prompt="x", seed=1, reference_images=[_b64((i, 0, 0)) for i in range(4)])
    assert len(stub.edit_calls[0]["reference_images"]) == 4


@pytest.mark.parametrize("family", ["anima", "sdxl", "flux1", "krea2", "z_image"])
def test_edit_needs_a_flux2_model(stub, family):
    stub.loaded_family = family
    with pytest.raises(RuntimeError, match="FLUX.2"):
        _run(mode="edit", prompt="x", reference_images=[_b64((1, 2, 3))])
    assert stub.edit_calls == []


# ── dispatch ────────────────────────────────────────────────────────

def test_edit_branch_builds_the_edit_kwargs(stub):
    r = _run(mode="edit", prompt="make her shirt red", steps=4, cfg=1.0, seed=9,
             sampler="euler", scheduler="flux2", width=512, height=768,
             reference_images=[_b64((9, 9, 9), (80, 80)), _b64((1, 1, 1), (96, 64))])
    (kw,) = stub.edit_calls
    assert kw["prompt"] == "make her shirt red"
    assert [im.size for im in kw["reference_images"]] == [(80, 80), (96, 64)]
    assert all(isinstance(im, Image.Image) and im.mode == "RGB" for im in kw["reference_images"])
    assert (kw["width"], kw["height"]) == (512, 768)
    assert (kw["steps"], kw["cfg_scale"], kw["seed"]) == (4, 1.0, 9)
    assert (kw["sampler"], kw["scheduler"]) == ("euler", "flux2")
    assert callable(kw["progress_callback"])
    assert "negative_prompt" not in kw and "input_image" not in kw
    assert "edit-info" in r["info"] and r["seed"] == 9


def test_edit_strips_lora_tags_from_the_prompt(stub):
    _run(mode="edit", prompt="red shirt <lora:foo:0.5>", seed=1,
         reference_images=[_b64((5, 5, 5))])
    assert stub.edit_calls[0]["prompt"].strip() == "red shirt"


def test_t2i_does_not_reach_the_edit_engine(stub):
    _run(mode="t2i", prompt="a fox", seed=1, reference_images=[_b64((5, 5, 5))])
    assert "reference_images" not in stub.edit_calls[0]


def test_flux1_run_falls_back_from_the_flux2_scheduler(stub):
    stub.loaded_family = "flux1"
    _run(prompt="a fox", seed=1, scheduler="flux2")
    assert stub.edit_calls[0]["scheduler"] == "flux"


# ── base-image cache fingerprint ────────────────────────────────────

def test_fingerprint_hashes_a_list_of_images_by_pixels():
    a1, a2 = Image.new("RGB", (8, 8), (1, 1, 1)), Image.new("RGB", (8, 8), (1, 1, 1))
    b = Image.new("RGB", (8, 8), (2, 2, 2))
    fp = server._fingerprint_value
    assert fp([a1, b]) == fp([a2, b])
    assert fp([a1, b]) != fp([b, a1])
    assert fp([a1]) != fp([b])
    assert "object at 0x" not in json.dumps(fp([a1, b]), default=str)


def test_base_fingerprint_ignores_object_identity(stub):
    def fp(color):
        kw = dict(prompt="x", reference_images=[Image.new("RGB", (8, 8), color)])
        return server._base_fingerprint(kw, "edit", [])
    assert fp((1, 1, 1)) == fp((1, 1, 1))
    assert fp((1, 1, 1)) != fp((2, 2, 2))


def test_same_references_reuse_the_base_and_new_ones_re_sample(stub):
    refs = [_b64((10, 10, 10)), _b64((20, 20, 20))]
    _run(mode="edit", prompt="x", seed=7, reference_images=refs)
    _run(mode="edit", prompt="x", seed=7, reference_images=refs, upscale_enabled=False,
         detail_enabled=False)
    assert len(stub.edit_calls) == 1, "identical references should reuse the base"
    _run(mode="edit", prompt="x", seed=7, reference_images=[refs[1], refs[0]])
    assert len(stub.edit_calls) == 2, "reordered references must re-sample"
    _run(mode="edit", prompt="x", seed=7, reference_images=[refs[0]])
    assert len(stub.edit_calls) == 3, "a different reference set must re-sample"


# ── post passes after an edit ───────────────────────────────────────

def test_upscale_runs_after_an_edit(stub):
    stub.upscale = lambda image, **kw: (image.resize((128, 128)), "upscaled")
    _run(mode="edit", prompt="x", seed=7, reference_images=[_b64((10, 10, 10))],
         upscale_enabled=True, upscale_scale=2.0)
    assert len(stub.edit_calls) == 1


# ── metadata ────────────────────────────────────────────────────────

class _MetaEngine:
    loaded_name = "flux-2-klein-4b.safetensors"
    last_seed = 3
    perf_flags_str = "default"


_EDIT_GEN = {"prompt": "make her shirt red", "negative_prompt": "", "steps": 4,
             "sampler": "euler", "scheduler": "flux2", "cfg_scale": 1.0,
             "width": 1024, "height": 768,
             "reference_images": [Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8))]}


def test_a1111_metadata_records_the_reference_count():
    params = md.format_metadata(_EDIT_GEN, _MetaEngine())
    assert "Edit references: 2" in params
    assert "Scheduler: flux2" in params and "Size: 1024x768" in params
    assert md.parse_metadata(params)["edit_references"] == "2"


def test_swarmui_metadata_records_the_reference_count():
    blob = json.loads(md.format_swarmui_metadata(_EDIT_GEN, _MetaEngine()))
    assert blob["sui_extra_data"]["edit_references"] == 2
    assert md.parse_metadata(json.dumps(blob))["edit_references"] == 2


def test_metadata_embeds_no_image_data():
    params = md.format_metadata(_EDIT_GEN, _MetaEngine())
    assert "base64" not in params and "PNG" not in params


def test_plain_runs_record_no_edit_field():
    gen = {k: v for k, v in _EDIT_GEN.items() if k != "reference_images"}
    assert "Edit references" not in md.format_metadata(gen, _MetaEngine())
    assert "edit_references" not in json.loads(
        md.format_swarmui_metadata(gen, _MetaEngine()))["sui_extra_data"]


# ── engine wrapper and option lists ─────────────────────────────────

def _flux(family="flux2") -> LoadedModel:
    return LoadedModel(name="FLUX(klein)", family=family, model=object(), native_res=1024,
                       vae_name="v.safetensors", te_name="t.safetensors")


def test_engine_generate_edit_passes_references_to_the_pipeline(monkeypatch):
    seen = {}

    class FakePipeline:
        def __init__(self, model):
            pass

        def __call__(self, **kw):
            seen.update(kw)
            return Image.new("RGB", (32, 32)), SimpleNamespace(vae_decode_mode="full")

    monkeypatch.setattr(engine_mod, "TextToImage", FakePipeline)
    eng = Engine(device="cpu")
    eng._loaded = _flux()
    refs = [Image.new("RGB", (64, 64)), Image.new("RGB", (64, 64))]
    image, info = eng.generate_edit("make it red", refs, steps=4, scheduler="flux2", seed=11)
    assert seen["reference_images"] is refs
    assert seen["width"] is None and seen["height"] is None
    assert seen["scheduler"] == "flux2" and seen["seed"] == 11
    assert info == "Seed: 11 | 2 refs | 4 steps | VAE: full"


def test_engine_generate_edit_rejects_other_families_and_empty_refs():
    eng = Engine(device="cpu")
    with pytest.raises(RuntimeError, match="No model"):
        eng.generate_edit("x", [Image.new("RGB", (64, 64))])
    eng._loaded = _flux("flux1")
    with pytest.raises(RuntimeError, match="FLUX.2"):
        eng.generate_edit("x", [Image.new("RGB", (64, 64))])
    eng._loaded = _flux()
    with pytest.raises(RuntimeError, match="reference"):
        eng.generate_edit("x", [])


def test_flux_scheduler_list_offers_the_flux2_schedule():
    from diffucore.pipelines._flux import _FLUX_SCHEDULERS
    assert "flux2" in SCHEDULERS_FLUX
    assert set(SCHEDULERS_FLUX) <= set(_FLUX_SCHEDULERS)
    assert "flux2" in SCHEDULER_GROUPS["Model default"]


def test_state_reports_the_loaded_family(monkeypatch):
    eng = _StubEngine()
    eng.status_text = lambda: "ok"
    monkeypatch.setattr(server, "ENGINE", eng)
    assert server._state_payload()["loaded_family"] == "flux2"


def test_models_endpoint_lists_the_flux2_scheduler_and_loaded_family():
    with TestClient(server.app) as c:
        m = c.get("/api/models").json()
    assert m["schedulers_flux"] == SCHEDULERS_FLUX and "flux2" in m["schedulers_flux"]
    assert "loaded_family" in m
