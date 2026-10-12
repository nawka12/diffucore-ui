"""Krea 2 wiring: model-name unwrapping, variant metadata, load validation,
unsupported-feature gates and caps. No model is loaded.

    .venv/bin/python -m pytest backend/test_krea2.py -v
"""

from __future__ import annotations

import pytest
import torch
from fastapi.testclient import TestClient

import metadata as md
import model_hash as mh
import server
from engine import ENGINE, Engine, LoadedModel, SCHEDULERS_KREA2, SAMPLERS_ANIMA


def _krea2(dit="krea2_turbo_bf16.safetensors", variant="turbo",
           vae="v.safetensors", te="t.safetensors") -> LoadedModel:
    return LoadedModel(name=f"Krea2({dit})", family="krea2", model=object(),
                       native_res=1024, vae_name=vae, te_name=te, variant=variant)


class _Engine:
    loaded_name = "Krea2(krea2_turbo_bf16.safetensors)"
    last_seed = 7
    perf_flags_str = "default"
    loaded_variant = "turbo"


_GEN = {"prompt": "a fox", "negative_prompt": "", "steps": 8, "sampler": "euler",
        "scheduler": "krea2", "cfg_scale": 1.0, "width": 1024, "height": 1024}


# ── model_hash / metadata ────────────────────────────────────────────

def test_model_name_strips_krea2_wrapper():
    assert mh.clean_model_name("Krea2(a.safetensors)") == "a.safetensors"
    assert mh.clean_model_name("Krea2(a.safetensors)", strip_ext=True) == "a"
    assert mh._unwrap("Krea2(a.safetensors)") == ("a.safetensors", True)


def test_a1111_metadata_records_variant_and_standard_cfg():
    text = md.format_metadata(_GEN, _Engine())
    assert "Krea2 variant: turbo" in text
    assert "CFG scale: 1.0" in text
    assert "Model: krea2_turbo_bf16.safetensors" in text
    parsed = md.parse_metadata(text)
    assert parsed["krea2_variant"] == "turbo"
    assert parsed["model"] == "krea2_turbo_bf16.safetensors"


def test_swarmui_metadata_records_variant():
    parsed = md.parse_metadata(md.format_swarmui_metadata(_GEN, _Engine()))
    assert parsed["krea2_variant"] == "turbo"


def test_other_families_write_no_variant():
    class Plain(_Engine):
        loaded_variant = None
    assert "Krea2 variant" not in md.format_metadata(_GEN, Plain())


# ── engine ───────────────────────────────────────────────────────────

def test_engine_krea2_surface():
    eng = Engine(device="cpu")
    eng._loaded = _krea2()
    assert eng.available_schedulers == SCHEDULERS_KREA2
    assert eng.loaded_variant == "turbo"
    assert eng.can_inpaint
    assert Engine._native_res("krea2") == 1024


def test_preview_accepts_a_4d_latent():
    eng = Engine(device="cpu")
    eng._loaded = _krea2()
    img = eng._latent_to_preview(torch.zeros(1, 16, 8, 6))
    assert img is not None and img.size == (6, 8)


def test_krea2_never_parks_in_the_checkpoint_cache():
    eng = Engine(device="cpu")
    eng._loaded = _krea2()
    eng._offload = False
    assert not eng._cacheable_for_stash()


def test_load_krea2_skips_when_everything_matches():
    eng = Engine(device="cpu")
    eng._loaded = _krea2()
    eng._offload, eng._vae_tile = True, True
    msg = eng.load_krea2("krea2_turbo_bf16.safetensors", "v.safetensors", "t.safetensors",
                         variant="turbo", offload=True, vae_tile=True)
    assert msg.startswith("Model already loaded")


def test_load_krea2_reloads_on_variant_override():
    eng = Engine(device="cpu")
    eng._loaded = _krea2()
    eng._offload, eng._vae_tile = True, True
    # Same files, other variant: must reach the file check instead of skipping.
    with pytest.raises(FileNotFoundError):
        eng.load_krea2("krea2_turbo_bf16.safetensors", "v.safetensors", "t.safetensors",
                       variant="raw", offload=True, vae_tile=True)


# ── server: load validation, caps ─────────────────────────────

def test_validate_load_krea2(tmp_path, monkeypatch):
    for attr, name in (("DIFFUSION_DIR", "dit"), ("VAE_DIR", "vae"), ("TE_DIR", "te")):
        d = tmp_path / name
        d.mkdir()
        (d / "f.safetensors").write_bytes(b"x")
        monkeypatch.setattr(server, attr, d)
    ok = dict(model_type="Krea2", dit="f.safetensors", vae="f.safetensors", te="f.safetensors")
    assert server._validate_load(server.LoadPayload(**ok)) is None
    assert server._validate_load(server.LoadPayload(**ok, variant="turbo")) is None
    err = server._validate_load(server.LoadPayload(**{**ok, "te": "ghost.safetensors"}))
    assert err and "not found" in err.lower()
    assert "all three" in server._validate_load(server.LoadPayload(**{**ok, "vae": "(none)"}))
    for bad in ({"cuda_graphs": True}, {"fused_glue": True}):
        assert "not supported for Krea 2" in server._validate_load(server.LoadPayload(**ok, **bad))
    with pytest.raises(Exception):
        server.LoadPayload(**ok, variant="fast")


@pytest.fixture
def krea2_loaded(monkeypatch):
    monkeypatch.setattr(ENGINE, "_loaded", _krea2())


def test_krea2_takes_every_mode_and_drops_cache_knobs(krea2_loaded, monkeypatch):
    """img2img / inpaint / detailer / upscaler / LoRA tags reach the engine;
    TeaCache and DeepCache, which Krea 2 lacks, are zeroed first."""
    seen = {}
    def fake_t2i(**kw):
        seen.update(kw)
        raise RuntimeError("stop")
    monkeypatch.setattr(ENGINE, "generate_t2i", fake_t2i, raising=False)
    monkeypatch.setattr(ENGINE, "apply_temp_loras", lambda loras: "", raising=False)
    p = server.GeneratePayload(prompt="a fox <lora:style:0.8>", teacache=0.3, upscale_teacache=0.4,
                               deepcache=3, detail_enabled=True, upscale_enabled=True)
    with pytest.raises(RuntimeError, match="stop"):
        server._run_generation(p, lambda *a: None)
    assert (p.teacache, p.deepcache, p.upscale_teacache) == (0.0, 1, 0.0)
    with TestClient(server.app) as c:
        for body in ({"mode": "i2i"}, {"mode": "inpaint"}, {"detail_enabled": True},
                     {"upscale_enabled": True}, {"prompt": "a fox <lora:style:0.8>"}):
            assert c.post("/api/generate", json=body).status_code == 200


def test_krea2_settings_knobs_match_anima(monkeypatch):
    monkeypatch.setattr(ENGINE, "_loaded", _krea2())
    knobs = server._settings_knobs("secant_anneal", "beta", 0.0)
    assert {"curvature", "eta_max", "beta_alpha", "beta_beta"} <= set(knobs)


def test_models_endpoint_lists_krea2_options():
    with TestClient(server.app) as c:
        m = c.get("/api/models").json()
    assert m["schedulers_krea2"] == SCHEDULERS_KREA2
    assert m["schedulers_krea2"][0] == "krea2"
    assert m["samplers_krea2"] == SAMPLERS_ANIMA
