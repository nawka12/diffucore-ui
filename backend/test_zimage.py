"""Z-Image wiring: model-name unwrapping, variant metadata, load validation and
dispatch, feature availability and caps. No model is loaded.

    .venv/bin/python -m pytest backend/test_zimage.py -v
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

import engine as engine_mod
import metadata as md
import model_hash as mh
import server
from engine import ENGINE, Engine, LoadedModel, SAMPLERS_ANIMA, SAMPLERS_ZIMAGE, SCHEDULERS_ZIMAGE

DIT = "z_image_turbo_int8_convrot.safetensors"


def _zimage(dit=DIT, variant="turbo", vae="v.safetensors", te="t.safetensors") -> LoadedModel:
    return LoadedModel(name=f"ZImage({dit})", family="z_image", model=object(),
                       native_res=1024, vae_name=vae, te_name=te, variant=variant)


class _Engine:
    loaded_name = f"ZImage({DIT})"
    loaded_family = "z_image"
    last_seed = 7
    perf_flags_str = "default"
    loaded_variant = "turbo"


_GEN = {"prompt": "a fox", "negative_prompt": "", "steps": 8, "sampler": "euler",
        "scheduler": "zimage", "cfg_scale": 1.0, "width": 1024, "height": 1024}


@pytest.fixture
def zimage_loaded(monkeypatch):
    monkeypatch.setattr(ENGINE, "_loaded", _zimage())


# ── engine option lists stay in step with the pipeline ─────────────

def test_option_lists_match_the_pipeline():
    from diffucore.pipelines._zimage import _ZIMAGE_SAMPLERS, _ZIMAGE_SCHEDULERS
    assert SCHEDULERS_ZIMAGE == list(_ZIMAGE_SCHEDULERS)
    assert SCHEDULERS_ZIMAGE[0] == "zimage"
    assert set(SAMPLERS_ZIMAGE) | {"cogent3_pump_onset", "secant_onset"} == set(_ZIMAGE_SAMPLERS) == set(SAMPLERS_ANIMA)


# ── model_hash / metadata ────────────────────────────────────────────

def test_model_name_strips_zimage_wrapper():
    assert mh.clean_model_name("ZImage(a.safetensors)") == "a.safetensors"
    assert mh.clean_model_name("ZImage(a.safetensors)", strip_ext=True) == "a"
    assert mh._unwrap("ZImage(a.safetensors)") == ("a.safetensors", True)


def test_a1111_metadata_records_the_z_image_variant():
    text = md.format_metadata(_GEN, _Engine())
    assert "Z Image variant: turbo" in text and "Krea2" not in text
    assert "Scheduler: zimage" in text and "CFG scale: 1.0" in text
    assert f"Model: {DIT}" in text
    parsed = md.parse_metadata(text)
    assert parsed["z_image_variant"] == "turbo"
    assert parsed["model"] == DIT


def test_swarmui_metadata_records_the_z_image_variant():
    parsed = md.parse_metadata(md.format_swarmui_metadata(_GEN, _Engine()))
    assert parsed["z_image_variant"] == "turbo"
    assert "krea2_variant" not in parsed


def test_krea2_metadata_keeps_its_own_variant_key():
    class Krea(_Engine):
        loaded_family = "krea2"
    assert "Krea2 variant: turbo" in md.format_metadata(_GEN, Krea())
    assert md.parse_metadata(md.format_swarmui_metadata(_GEN, Krea()))["krea2_variant"] == "turbo"


# ── engine ───────────────────────────────────────────────────────────

def test_engine_zimage_surface():
    eng = Engine(device="cpu")
    eng._loaded = _zimage()
    assert eng.available_schedulers == SCHEDULERS_ZIMAGE
    assert eng.loaded_variant == "turbo"
    assert eng.can_inpaint
    assert Engine._native_res("z_image") == 1024


def test_zimage_snaps_img2img_sizes_to_its_16px_grid():
    # The UI rounds loaded images to ÷8; Z-Image runs on 16 px patches.
    eng = Engine(device="cpu")
    eng._loaded = _zimage()
    assert eng._anima_gen_size(1000, 760) == (1008, 768, True)
    assert eng._anima_gen_size(1024, 768) == (1024, 768, False)


def test_zimage_never_parks_in_the_checkpoint_cache():
    eng = Engine(device="cpu")
    eng._loaded = _zimage()
    eng._offload = False
    assert not eng._cacheable_for_stash()


def test_zimage_live_preview_uses_the_flux1_factors():
    eng = Engine(device="cpu")
    eng._loaded = _zimage()
    lat = torch.zeros(1, 16, 8, 6)
    lat[:, 4] = 5.0             # a channel with positive red/green/blue weight
    img = np.array(eng._latent_to_preview(lat))
    assert img.shape == (8, 6, 3)
    assert img.mean() > np.array(eng._latent_to_preview(torch.zeros(1, 16, 8, 6))).mean()
    # A 5-D (B, C, 1, H, W) latent previews too.
    assert eng._latent_to_preview(torch.zeros(1, 16, 1, 8, 6)).size == (6, 8)


def test_engine_runs_lora_img2img_and_inpaint_for_zimage(monkeypatch):
    seen = {}

    class FakePipeline:
        def __init__(self, model):
            pass

        def __call__(self, **kw):
            seen.update(kw)
            return Image.new("RGB", (kw["width"], kw["height"])), SimpleNamespace(vae_decode_mode="full")

    monkeypatch.setattr(engine_mod, "ImageToImage", FakePipeline)
    monkeypatch.setattr(engine_mod, "Inpaint", FakePipeline)
    monkeypatch.setattr(engine_mod, "TextToImage", FakePipeline)
    eng = Engine(device="cpu")
    eng._loaded = _zimage()
    image, _ = eng.generate_i2i("p", None, width=1000, height=760)
    assert (seen["width"], seen["height"]) == (1008, 768)
    assert image.size == (1000, 760)        # mapped back to the requested size
    eng.generate_inpaint("p", None, None, width=1024, height=1024)
    assert "uncond_skip_layers" not in seen and "slg_layers" not in seen
    eng.generate_t2i("a fox", steps=8, cfg_scale=1.0, scheduler="zimage", shift=3.0, width=1024, height=1024)
    assert seen["shift"] == 3.0 and seen["scheduler"] == "zimage"


def test_zimage_loras_go_through_the_hooked_apply(monkeypatch, tmp_path):
    (tmp_path / "style.safetensors").write_bytes(b"x")
    monkeypatch.setattr(engine_mod, "lora_path", lambda n: tmp_path / f"{n}.safetensors")
    applied = []
    monkeypatch.setattr(engine_mod, "apply_lora",
                        lambda model, path, multiplier=1.0: applied.append((path, multiplier))
                        or SimpleNamespace(applied=3, remapped=False))
    monkeypatch.setattr(engine_mod, "clear_bundle_loras", lambda model: None)
    eng = Engine(device="cpu")
    eng._loaded = _zimage()
    assert eng.apply_temp_loras([("style", 0.8)]) == "style@0.8: 3 matched"
    assert applied == [(str(tmp_path / "style.safetensors"), 0.8)]
    eng.apply_lora("style")
    assert len(applied) == 2 and eng._loaded.applied_loras == ["style", "style"]


def _fake_files(monkeypatch, tmp_path):
    for name in ("diffusion_model_path", "vae_path", "te_path"):
        monkeypatch.setattr(engine_mod, name, lambda n, d=tmp_path: d / n)
    for n in (DIT, "v.safetensors", "t.safetensors"):
        (tmp_path / n).write_bytes(b"x")


def test_load_zimage_builds_the_policy_and_labels_the_model(monkeypatch, tmp_path):
    _fake_files(monkeypatch, tmp_path)
    seen = {}

    def fake_load(dit, te, vae, variant=None, policy=None):
        seen.update(dit=dit, te=te, vae=vae, variant=variant, policy=policy)
        return SimpleNamespace(spec=SimpleNamespace(architecture="z_image", variant="turbo"), backbone=None)

    monkeypatch.setattr(engine_mod, "load_zimage_checkpoint", fake_load)
    eng = Engine(device="cpu")
    msg = eng.load_zimage(DIT, "v.safetensors", "t.safetensors", offload="stream", int8=True,
                          attention="int8_turing", fp16_accumulation=True)
    assert msg.startswith("Loaded Z-Image turbo")
    assert eng.loaded_name == f"ZImage({DIT})" and eng.loaded_family == "z_image"
    assert eng.loaded_variant == "turbo" and seen["variant"] is None
    pol = seen["policy"]
    assert (pol.offload, pol.int8, pol.attention, pol.fp16_accumulation) == ("stream", True, "int8_turing", True)
    assert not pol.compile
    assert not (eng._cuda_graphs or eng._channels_last or eng._tf32 or eng._fused_glue)
    again = eng.load_zimage(DIT, "v.safetensors", "t.safetensors", offload="stream", int8=True,
                            attention="int8_turing", fp16_accumulation=True)
    assert again.startswith("Model already loaded")
    # Same files, other variant: a real reload.
    reload = eng.load_zimage(DIT, "v.safetensors", "t.safetensors", variant="base", offload="stream",
                             int8=True, attention="int8_turing", fp16_accumulation=True)
    assert reload.startswith("Loaded Z-Image") and seen["variant"] == "base"


def test_load_zimage_reports_missing_files(monkeypatch, tmp_path):
    _fake_files(monkeypatch, tmp_path)
    (tmp_path / "t.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="Z-Image file not found"):
        Engine(device="cpu").load_zimage(DIT, "v.safetensors", "t.safetensors")


def test_reload_model_keeps_the_zimage_companions(monkeypatch):
    eng = Engine(device="cpu")
    eng._loaded = _zimage()
    eng._offload, eng._vae_tile, eng._int8 = "stream", True, True
    seen = {}
    monkeypatch.setattr(eng, "load_zimage", lambda *a, **kw: seen.update(args=a, kw=kw) or "ok")
    assert eng.reload_model("other.safetensors") == "ok"
    assert seen["args"] == ("other.safetensors", "v.safetensors", "t.safetensors")
    assert seen["kw"]["offload"] == "stream" and seen["kw"]["int8"] is True
    # The variant is re-read from the new file's name, never carried over.
    assert "variant" not in seen["kw"]
    eng.reload_model()
    assert seen["args"][0] == DIT
    with pytest.raises(ValueError, match="Text Encoder axis"):
        eng.reload_model(te_name="t2.safetensors")


# ── server: load validation and dispatch ───────────────────────────

def _fake_dirs(tmp_path, monkeypatch):
    for attr, name in (("DIFFUSION_DIR", "dit"), ("VAE_DIR", "vae"), ("TE_DIR", "te")):
        d = tmp_path / name
        d.mkdir()
        (d / "f.safetensors").write_bytes(b"x")
        monkeypatch.setattr(server, attr, d)


def test_validate_load_zimage(tmp_path, monkeypatch):
    _fake_dirs(tmp_path, monkeypatch)
    ok = dict(model_type="ZImage", dit="f.safetensors", vae="f.safetensors", te="f.safetensors")
    assert server._validate_load(server.LoadPayload(**ok)) is None
    for variant in ("turbo", "base"):
        assert server._validate_load(server.LoadPayload(**ok, variant=variant)) is None
    assert server._validate_load(server.LoadPayload(**ok, int8=True, attention="int8_turing",
                                                    fp16_accumulation=True)) is None
    err = server._validate_load(server.LoadPayload(**{**ok, "te": "ghost.safetensors"}))
    assert err and "not found" in err.lower()
    assert "all three Z-Image" in server._validate_load(server.LoadPayload(**{**ok, "vae": "(none)"}))
    for bad in ({"cuda_graphs": True}, {"fused_glue": True}):
        assert "not supported for Z-Image" in server._validate_load(server.LoadPayload(**ok, **bad))
    assert "turbo or base" in server._validate_load(server.LoadPayload(**ok, variant="raw"))


def test_each_family_rejects_the_other_s_variant(tmp_path, monkeypatch):
    _fake_dirs(tmp_path, monkeypatch)
    ok = dict(dit="f.safetensors", vae="f.safetensors", te="f.safetensors")
    assert "raw or turbo" in server._validate_load(server.LoadPayload(model_type="Krea2", variant="base", **ok))
    assert server._validate_load(server.LoadPayload(model_type="Krea2", variant="raw", **ok)) is None


def test_load_dispatch(monkeypatch):
    seen = {}
    monkeypatch.setattr(ENGINE, "load_zimage", lambda *a, **kw: seen.update(args=a, kw=kw) or "Loaded x")
    assert server._do_load_impl(server.LoadPayload(
        model_type="ZImage", dit="d", vae="v", te="t", variant="base", int8=True,
        attention="int8_turing")) == "Loaded x"
    assert seen["args"] == ("d", "v", "t")
    assert seen["kw"]["offload"] == "stream"        # the default; resident is no faster on a 2060
    assert seen["kw"]["variant"] == "base" and seen["kw"]["int8"] is True
    assert seen["kw"]["attention"] == "int8_turing"
    server._do_load_impl(server.LoadPayload(model_type="ZImage", dit="d", vae="v", te="t", offload="none"))
    assert seen["kw"]["offload"] is False
    assert "all three Z-Image" in server._do_load_impl(server.LoadPayload(model_type="ZImage", dit="d"))


# ── server: generation, gates ──────────────────────────────────────

def _capture(monkeypatch, name="generate_t2i"):
    seen = {}

    def fake(**kw):
        seen.clear()
        seen.update(kw)
        raise RuntimeError("stop")

    monkeypatch.setattr(ENGINE, name, fake, raising=False)
    return seen


def test_run_generation_coerces_the_scheduler_and_drops_cache_knobs(zimage_loaded, monkeypatch):
    seen = _capture(monkeypatch)
    p = server.GeneratePayload(prompt="a fox", scheduler="karras", teacache=0.3, upscale_teacache=0.4,
                               deepcache=3, steps=8, cfg=1.0)
    with pytest.raises(RuntimeError, match="stop"):
        server._run_generation(p, lambda *a: None)
    # An SD scheduler is coerced to the one the pipeline uses; cache knobs are dropped.
    assert seen["scheduler"] == p.scheduler == "zimage"
    assert (p.teacache, p.deepcache, p.upscale_teacache) == (0.0, 1, 0.0)
    assert seen["cfg_scale"] == 1.0 and seen["steps"] == 8 and seen["shift"] == p.shift
    keep = server.GeneratePayload(prompt="a fox", scheduler="beta_mix")
    with pytest.raises(RuntimeError, match="stop"):
        server._run_generation(keep, lambda *a: None)
    assert keep.scheduler == "beta_mix"
    # Another family's scheduler list is left alone.
    monkeypatch.setattr(ENGINE, "_loaded", LoadedModel(
        name="Krea2(k.safetensors)", family="krea2", model=object(), native_res=1024))
    other = server.GeneratePayload(prompt="a fox", scheduler="karras")
    with pytest.raises(RuntimeError, match="stop"):
        server._run_generation(other, lambda *a: None)
    assert other.scheduler == "karras"


def test_zimage_gets_no_skip_options(zimage_loaded, monkeypatch):
    seen = _capture(monkeypatch)
    with pytest.raises(RuntimeError, match="stop"):
        server._run_generation(server.GeneratePayload(
            prompt="a fox", uncond_skip_layers=[22], slg_layers=[8], slg_scale=3.0), lambda *a: None)
    assert not {"uncond_skip_layers", "slg_layers", "slg_scale"} & set(seen)


def test_zimage_accepts_the_features_every_family_has(zimage_loaded, monkeypatch):
    queued = []
    monkeypatch.setattr(server, "_enqueue", queued.append)
    with TestClient(server.app) as c:
        for body in ({"mode": "i2i"}, {"mode": "inpaint"}, {"detail_enabled": True},
                     {"upscale_enabled": True}, {"prompt": "a fox <lora:style:0.8>"},
                     {"neg": "<lora:style:0.8>"}):
            assert c.post("/api/generate", json=body).status_code == 200, body
        assert c.post("/api/upscale", json={"input_image": "x", "teacache": 0.3}).status_code == 200
        assert c.post("/api/xyz", json={"prompt": "a <lora:s:1>"}).status_code == 200
        assert len(queued) == 8


def test_standalone_upscale_and_detail_drop_teacache(zimage_loaded, monkeypatch):
    monkeypatch.setattr(server, "_enqueue", lambda job: None)
    seen = []
    monkeypatch.setattr(server, "_teacache_cuda_graphs_conflict", lambda tc: seen.append(tc))
    with TestClient(server.app) as c:
        c.post("/api/upscale", json={"input_image": "x", "teacache": 0.3})
        c.post("/api/detail", json={"input_image": "x", "teacache": 0.3,
                                    "models": [{"model": "face.pt"}]})
    assert seen == [0.0, 0.0]


def test_zimage_settings_knobs_match_anima(monkeypatch):
    monkeypatch.setattr(ENGINE, "_loaded", _zimage())
    knobs = server._settings_knobs("secant_anneal", "beta", 0.0)
    assert {"curvature", "eta_max", "beta_alpha", "beta_beta"} <= set(knobs)


def test_xyz_applies_the_settings_and_drops_teacache(zimage_loaded, monkeypatch):
    captured = {}

    def fake_grid(base_kwargs, *a, cell_knobs=None, **kw):
        captured.update(base=dict(base_kwargs), knobs=cell_knobs({"sampler": "secant_anneal", "scheduler": "beta"}))
        raise RuntimeError("stop")

    monkeypatch.setattr(server, "generate_xyz_grid", fake_grid)
    with pytest.raises(RuntimeError, match="stop"):
        server._run_xyz(server.XYZPayload(prompt="a fox", scheduler="zimage", teacache=0.3), lambda *a, **k: None)
    assert captured["base"]["teacache_thresh"] == 0.0
    assert {"curvature", "eta_max", "beta_alpha", "beta_beta"} <= set(captured["knobs"])


def test_models_endpoint_lists_zimage_options():
    with TestClient(server.app) as c:
        m = c.get("/api/models").json()
    assert m["schedulers_zimage"] == SCHEDULERS_ZIMAGE and m["schedulers_zimage"][0] == "zimage"
    assert m["samplers_zimage"] == SAMPLERS_ZIMAGE


def test_enhance_endpoint_rejects_zimage(zimage_loaded):
    with TestClient(server.app) as c:
        assert c.post("/api/enhance_prompt", json={"prompt": "a fox"}).status_code == 400
