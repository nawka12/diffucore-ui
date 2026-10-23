"""MOSS-Image wiring: model-name unwrapping, skip-block metadata, load
validation and dispatch, feature availability and caps. No model is loaded.

    .venv/bin/python -m pytest backend/test_moss.py -v
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

import engine as engine_mod
import metadata as md
import model_hash as mh
import server
from engine import (ENGINE, Engine, LoadedModel, MOSS_UNCOND_SKIP, SAMPLERS_ANIMA, SAMPLERS_MOSS,
                    SCHEDULERS_MOSS)

DIT = "moss_image_sft_int8_convrot.safetensors"


def _moss(dit=DIT, vae="v.safetensors", te="t.safetensors") -> LoadedModel:
    return LoadedModel(name=f"MOSS({dit})", family="moss_image", model=object(),
                       native_res=1024, vae_name=vae, te_name=te)


class _Engine:
    loaded_name = f"MOSS({DIT})"
    last_seed = 7
    perf_flags_str = "default"
    loaded_variant = None


_GEN = {"prompt": "a fox", "negative_prompt": "", "steps": 50, "sampler": "euler",
        "scheduler": "moss", "cfg_scale": 3.0, "width": 1024, "height": 1024}


@pytest.fixture
def moss_loaded(monkeypatch):
    monkeypatch.setattr(ENGINE, "_loaded", _moss())


# ── engine option lists stay in step with the pipeline ─────────────

def test_option_lists_match_the_pipeline():
    from diffucore.pipelines._moss import _MOSS_SAMPLERS, _MOSS_SCHEDULERS, MOSS_UNCOND_SKIP as ref_skip
    assert SCHEDULERS_MOSS == list(_MOSS_SCHEDULERS)
    assert SCHEDULERS_MOSS[0] == "moss"
    assert set(SAMPLERS_MOSS) == set(_MOSS_SAMPLERS) == set(SAMPLERS_ANIMA)
    assert MOSS_UNCOND_SKIP == ref_skip == (22,)


# ── model_hash / metadata ────────────────────────────────────────────

def test_model_name_strips_moss_wrapper():
    assert mh.clean_model_name("MOSS(a.safetensors)") == "a.safetensors"
    assert mh.clean_model_name("MOSS(a.safetensors)", strip_ext=True) == "a"
    assert mh._unwrap("MOSS(a.safetensors)") == ("a.safetensors", True)


@pytest.mark.parametrize("layers, text", [([22], "22"), ([22, 30], '"22, 30"'), ([], "none")])
def test_a1111_metadata_records_and_restores_skip_blocks(layers, text):
    out = md.format_metadata({**_GEN, "uncond_skip_layers": layers}, _Engine())
    assert f"Uncond skip layers: {text}" in out
    assert f"Model: {DIT}" in out and "Scheduler: moss" in out
    fields = md.workspace_fields(md.parse_metadata(out))
    assert fields["uncondSkip"] == ", ".join(str(i) for i in layers)


def test_swarmui_metadata_records_skip_blocks():
    parsed = md.parse_metadata(md.format_swarmui_metadata({**_GEN, "uncond_skip_layers": [22, 30]}, _Engine()))
    assert md.workspace_fields(parsed)["uncondSkip"] == "22, 30"


def test_other_families_write_no_skip_blocks():
    out = md.format_metadata(_GEN, _Engine())
    assert "Uncond skip" not in out
    assert "uncondSkip" not in md.workspace_fields(md.parse_metadata(out))


# ── engine ───────────────────────────────────────────────────────────

def test_engine_moss_surface():
    eng = Engine(device="cpu")
    eng._loaded = _moss()
    assert eng.available_schedulers == SCHEDULERS_MOSS
    assert eng.can_inpaint
    assert Engine._native_res("moss_image") == 1024


def test_moss_snaps_img2img_sizes_to_its_16px_grid():
    # The UI rounds loaded images to ÷8; MOSS-Image runs on 16 px tokens.
    eng = Engine(device="cpu")
    eng._loaded = _moss()
    assert eng._anima_gen_size(1000, 760) == (1008, 768, True)
    assert eng._anima_gen_size(1024, 768) == (1024, 768, False)
    assert eng._anima_gen_size(256, 200) == (256, 208, True)


def test_moss_never_parks_in_the_checkpoint_cache():
    eng = Engine(device="cpu")
    eng._loaded = _moss()
    eng._offload = False
    assert not eng._cacheable_for_stash()


def test_moss_live_preview_unpatchifies_to_the_vae_grid():
    eng = Engine(device="cpu")
    eng._loaded = _moss()
    lat = torch.zeros(1, 128, 8, 6)
    lat[:, 3 * 4 + 1 * 2 + 0] = 1.0     # unpatchified channel 3, sub-pixel (row 1, col 0)
    img = torch.from_numpy(np.array(eng._latent_to_preview(lat))).int()
    assert img.shape == (16, 12, 3)
    assert (img[1::2, 0::2] > img[0::2, 0::2]).all()
    assert torch.equal(img[0::2, 0::2], img[:, 1::2][0::2])


def test_engine_runs_lora_img2img_and_inpaint_for_moss(monkeypatch):
    seen = {}

    class FakePipeline:
        def __init__(self, model):
            pass

        def __call__(self, **kw):
            seen.update(kw)
            return "image", SimpleNamespace(vae_decode_mode="full")

    monkeypatch.setattr(engine_mod, "ImageToImage", FakePipeline)
    monkeypatch.setattr(engine_mod, "Inpaint", FakePipeline)
    eng = Engine(device="cpu")
    eng._loaded = _moss()
    eng.generate_i2i("p", None, uncond_skip_layers=[22, 30])
    assert seen["uncond_skip_layers"] == [22, 30]
    seen.clear()
    eng.generate_inpaint("p", None, None, uncond_skip_layers=[])
    assert seen["uncond_skip_layers"] == []
    # A family without the knob is never sent it.
    eng._loaded = LoadedModel(name="Krea2(k.safetensors)", family="krea2", model=object(), native_res=1024)
    seen.clear()
    eng.generate_i2i("p", None, uncond_skip_layers=[22])
    assert "uncond_skip_layers" not in seen


def test_moss_loras_go_through_the_hooked_apply(monkeypatch, tmp_path):
    (tmp_path / "style.safetensors").write_bytes(b"x")
    monkeypatch.setattr(engine_mod, "lora_path", lambda n: tmp_path / f"{n}.safetensors")
    applied = []
    monkeypatch.setattr(engine_mod, "apply_lora",
                        lambda model, path, multiplier=1.0: applied.append((path, multiplier))
                        or SimpleNamespace(applied=3, remapped=False))
    monkeypatch.setattr(engine_mod, "clear_bundle_loras", lambda model: None)
    eng = Engine(device="cpu")
    eng._loaded = _moss()
    assert eng.apply_temp_loras([("style", 0.8)]) == "style@0.8: 3 matched"
    assert applied == [(str(tmp_path / "style.safetensors"), 0.8)]
    eng.apply_lora("style")
    assert len(applied) == 2 and eng._loaded.applied_loras == ["style", "style"]


def test_generate_t2i_passes_skip_blocks_and_shift(monkeypatch):
    seen = {}

    class FakeT2I:
        def __init__(self, model):
            pass

        def __call__(self, **kw):
            seen.update(kw)
            return "image", SimpleNamespace(vae_decode_mode="full")

    monkeypatch.setattr(engine_mod, "TextToImage", FakeT2I)
    eng = Engine(device="cpu")
    eng._loaded = _moss()
    eng.generate_t2i("a fox", steps=50, cfg_scale=3.0, sampler="euler", scheduler="moss", shift=5.0,
                     uncond_skip_layers=[22, 30])
    assert seen["uncond_skip_layers"] == [22, 30] and seen["shift"] == 5.0
    eng.generate_t2i("a fox", uncond_skip_layers=None)
    assert seen["uncond_skip_layers"] is None


def _fake_files(monkeypatch, tmp_path):
    for name in ("diffusion_model_path", "vae_path", "te_path"):
        monkeypatch.setattr(engine_mod, name, lambda n, d=tmp_path: d / n)
    for n in (DIT, "v.safetensors", "t.safetensors"):
        (tmp_path / n).write_bytes(b"x")


def test_load_moss_builds_the_policy_and_labels_the_model(monkeypatch, tmp_path):
    _fake_files(monkeypatch, tmp_path)
    seen = {}

    def fake_load(dit, te, vae, policy):
        seen.update(dit=dit, te=te, vae=vae, policy=policy)
        return SimpleNamespace(spec=SimpleNamespace(architecture="moss_image"), backbone=None)

    monkeypatch.setattr(engine_mod, "load_moss_checkpoint", fake_load)
    eng = Engine(device="cpu")
    msg = eng.load_moss(DIT, "v.safetensors", "t.safetensors", offload="stream", int8=True,
                        attention="int8_turing", fp16_accumulation=True)
    assert msg.startswith("Loaded MOSS-Image")
    assert eng.loaded_name == f"MOSS({DIT})" and eng.loaded_family == "moss_image"
    assert eng.loaded_variant is None
    pol = seen["policy"]
    assert (pol.offload, pol.int8, pol.attention, pol.fp16_accumulation) == ("stream", True, "int8_turing", True)
    assert not pol.compile
    # No CUDA Graphs, channels_last, tf32 or fused glue for MOSS.
    assert not (eng._cuda_graphs or eng._channels_last or eng._tf32 or eng._fused_glue)
    again = eng.load_moss(DIT, "v.safetensors", "t.safetensors", offload="stream", int8=True,
                          attention="int8_turing", fp16_accumulation=True)
    assert again.startswith("Model already loaded")


def test_load_moss_reports_missing_files(monkeypatch, tmp_path):
    _fake_files(monkeypatch, tmp_path)
    (tmp_path / "t.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="MOSS-Image file not found"):
        Engine(device="cpu").load_moss(DIT, "v.safetensors", "t.safetensors")


def test_reload_model_keeps_the_moss_companions(monkeypatch):
    eng = Engine(device="cpu")
    eng._loaded = _moss()
    eng._offload, eng._vae_tile, eng._int8 = "stream", True, True
    seen = {}
    monkeypatch.setattr(eng, "load_moss", lambda *a, **kw: seen.update(args=a, kw=kw) or "ok")
    assert eng.reload_model("other.safetensors") == "ok"
    assert seen["args"] == ("other.safetensors", "v.safetensors", "t.safetensors")
    assert seen["kw"]["offload"] == "stream" and seen["kw"]["int8"] is True
    # Name-less reload (a TE-free no-op) recovers the DiT from the label.
    eng.reload_model()
    assert seen["args"][0] == DIT
    with pytest.raises(ValueError, match="Text Encoder axis"):
        eng.reload_model(te_name="t2.safetensors")


# ── server: load validation and dispatch ───────────────────────────

def test_validate_load_moss(tmp_path, monkeypatch):
    for attr, name in (("DIFFUSION_DIR", "dit"), ("VAE_DIR", "vae"), ("TE_DIR", "te")):
        d = tmp_path / name
        d.mkdir()
        (d / "f.safetensors").write_bytes(b"x")
        monkeypatch.setattr(server, attr, d)
    ok = dict(model_type="MOSS", dit="f.safetensors", vae="f.safetensors", te="f.safetensors")
    assert server._validate_load(server.LoadPayload(**ok)) is None
    assert server._validate_load(server.LoadPayload(**ok, int8=True, attention="int8_turing",
                                                    fp16_accumulation=True)) is None
    err = server._validate_load(server.LoadPayload(**{**ok, "te": "ghost.safetensors"}))
    assert err and "not found" in err.lower()
    assert "all three MOSS-Image" in server._validate_load(server.LoadPayload(**{**ok, "vae": "(none)"}))
    for bad in ({"cuda_graphs": True}, {"fused_glue": True}):
        assert "not supported for MOSS-Image" in server._validate_load(server.LoadPayload(**ok, **bad))


def test_load_dispatch_streams_by_default(monkeypatch):
    seen = {}
    monkeypatch.setattr(ENGINE, "load_moss", lambda *a, **kw: seen.update(args=a, kw=kw) or "Loaded x")
    assert server._do_load_impl(server.LoadPayload(
        model_type="MOSS", dit="d", vae="v", te="t", int8=True, attention="int8_turing")) == "Loaded x"
    assert seen["args"] == ("d", "v", "t")
    assert seen["kw"]["offload"] == "stream" and seen["kw"]["int8"] is True
    assert seen["kw"]["attention"] == "int8_turing"
    server._do_load_impl(server.LoadPayload(model_type="MOSS", dit="d", vae="v", te="t", offload="none"))
    assert seen["kw"]["offload"] is False
    assert "all three MOSS-Image" in server._do_load_impl(server.LoadPayload(model_type="MOSS", dit="d"))


# ── server: generation, gates ──────────────────────────────────────

def test_run_generation_sends_and_records_the_skip_blocks(moss_loaded, monkeypatch):
    seen = {}

    def fake_t2i(**kw):
        seen.update(kw)
        raise RuntimeError("stop")

    monkeypatch.setattr(ENGINE, "generate_t2i", fake_t2i, raising=False)
    for sent, want in ((None, [22]), ([], []), ([22, 30], [22, 30])):
        p = server.GeneratePayload(prompt="a fox", scheduler="karras", teacache=0.3, upscale_teacache=0.4,
                                   deepcache=3, uncond_skip_layers=sent)
        with pytest.raises(RuntimeError, match="stop"):
            server._run_generation(p, lambda *a: None)
        assert seen["uncond_skip_layers"] == want
        # An SD scheduler is coerced to the one the pipeline uses; cache knobs are dropped.
        assert p.scheduler == "moss"
        assert (p.teacache, p.deepcache, p.upscale_teacache) == (0.0, 1, 0.0)
    keep = server.GeneratePayload(prompt="a fox", scheduler="beta_mix")
    with pytest.raises(RuntimeError, match="stop"):
        server._run_generation(keep, lambda *a: None)
    assert keep.scheduler == "beta_mix"


def test_other_families_get_no_skip_blocks(monkeypatch):
    monkeypatch.setattr(ENGINE, "_loaded", LoadedModel(
        name="Krea2(k.safetensors)", family="krea2", model=object(), native_res=1024))
    seen = {}

    def fake_t2i(**kw):
        seen.update(kw)
        raise RuntimeError("stop")

    monkeypatch.setattr(ENGINE, "generate_t2i", fake_t2i, raising=False)
    with pytest.raises(RuntimeError, match="stop"):
        server._run_generation(server.GeneratePayload(prompt="a fox"), lambda *a: None)
    assert "uncond_skip_layers" not in seen


def test_skip_blocks_field_is_validated():
    assert server.GeneratePayload().uncond_skip_layers is None
    assert server.GeneratePayload(uncond_skip_layers=[]).uncond_skip_layers == []
    for bad in ([-1], ["a"], [1.5], list(range(65))):
        with pytest.raises(Exception):
            server.GeneratePayload(uncond_skip_layers=bad)


def test_moss_accepts_the_features_every_family_has(moss_loaded, monkeypatch):
    queued = []
    monkeypatch.setattr(server, "_enqueue", queued.append)
    with TestClient(server.app) as c:
        for body in ({"mode": "i2i"}, {"mode": "inpaint"}, {"detail_enabled": True},
                     {"upscale_enabled": True}, {"prompt": "a fox <lora:style:0.8>"},
                     {"neg": "<lora:style:0.8>"}, {"prompt": "a fox", "uncond_skip_layers": [22]}):
            assert c.post("/api/generate", json=body).status_code == 200, body
        assert c.post("/api/upscale", json={"input_image": "x"}).status_code == 200
        assert c.post("/api/xyz", json={"prompt": "a <lora:s:1>"}).status_code == 200
        assert len(queued) == 9


def test_run_generation_sends_the_skip_blocks_to_img2img_and_inpaint(moss_loaded, monkeypatch):
    seen = {}

    def fake(**kw):
        seen.update(kw)
        raise RuntimeError("stop")

    img = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGD4DwABBAEAwS2OUAAAAABJRU5ErkJggg=="
    monkeypatch.setattr(ENGINE, "generate_i2i", fake, raising=False)
    monkeypatch.setattr(ENGINE, "generate_inpaint", fake, raising=False)
    for mode in ("i2i", "inpaint"):
        for sent, want in ((None, [22]), ([], [])):
            seen.clear()
            p = server.GeneratePayload(prompt="a fox", mode=mode, input_image=img, mask_image=img,
                                       uncond_skip_layers=sent)
            with pytest.raises(RuntimeError, match="stop"):
                server._run_generation(p, lambda *a: None)
            assert seen["uncond_skip_layers"] == want


def test_other_families_are_not_gated(monkeypatch):
    monkeypatch.setattr(ENGINE, "_loaded", LoadedModel(
        name="Krea2(k.safetensors)", family="krea2", model=object(), native_res=1024))
    queued = []
    monkeypatch.setattr(server, "_enqueue", queued.append)
    with TestClient(server.app) as c:
        for body in ({"mode": "i2i"}, {"detail_enabled": True}, {"prompt": "a fox <lora:style:0.8>"}):
            assert c.post("/api/generate", json=body).status_code == 200
        assert c.post("/api/upscale", json={"input_image": "x"}).status_code == 200


def test_moss_settings_knobs_match_anima(monkeypatch):
    monkeypatch.setattr(ENGINE, "_loaded", _moss())
    knobs = server._settings_knobs("secant_anneal", "beta", 0.0)
    assert {"curvature", "eta_max", "beta_alpha", "beta_beta"} <= set(knobs)


def test_models_endpoint_lists_moss_options():
    with TestClient(server.app) as c:
        m = c.get("/api/models").json()
    assert m["schedulers_moss"] == SCHEDULERS_MOSS and m["schedulers_moss"][0] == "moss"
    assert m["samplers_moss"] == SAMPLERS_MOSS
    assert m["moss_uncond_skip"] == [22]


def test_enhance_endpoint_rejects_moss(moss_loaded):
    with TestClient(server.app) as c:
        assert c.post("/api/enhance_prompt", json={"prompt": "a fox"}).status_code == 400
