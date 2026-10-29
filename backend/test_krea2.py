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


class _FakeEnhancer:
    def __init__(self, seen):
        self.seen = seen
    def enhance(self, text, **kw):
        self.seen.update(mode="enhance", text=text, **kw)
        return "a detailed fox"
    def expand(self, text, **kw):
        self.seen.update(mode="expand", text=text, **kw)
        return text + ". It sleeps."


def _fake_bundle_loaded(monkeypatch, family, seen, te="t.safetensors"):
    from types import SimpleNamespace
    import engine as engine_mod
    bundle = SimpleNamespace(backbone="dit", vae="vae", text_encoder="te", text_encoder_2=None,
                             policy="policy")
    monkeypatch.setattr(ENGINE, "_loaded", LoadedModel(
        name="M(m.safetensors)", family=family, model=bundle, native_res=1024,
        vae_name="v.safetensors", te_name=te, variant="raw" if family == "krea2" else None))
    def load(path, policy, int8=False):
        seen.update(path=path, int8=int8)
        return _FakeEnhancer(seen)
    monkeypatch.setattr(engine_mod, "load_prompt_enhancer", load)


def test_enhance_prompt_sets_lora_tags_aside(monkeypatch):
    """Only the text reaches the LM; the tags come back at the end, and the
    loaded model's modules are offered as room to free."""
    seen = {}
    _fake_bundle_loaded(monkeypatch, "krea2", seen)
    out = ENGINE.enhance_prompt("a fox <lora:style:0.8> in  snow <lora:b:1>", int8=True)
    assert out == "a detailed fox <lora:style:0.8> <lora:b:1>"
    assert seen["mode"] == "enhance" and seen["text"] == "a fox in snow"
    assert seen["path"].endswith("t.safetensors") and seen["int8"] is True
    assert seen["make_room"] == ["dit", "vae", "te"]
    with pytest.raises(ValueError):
        ENGINE.enhance_prompt("<lora:style:0.8>")


def test_anima_enhance_runs_the_settings_model_as_expand(monkeypatch, tmp_path):
    import engine as engine_mod
    seen = {}
    _fake_bundle_loaded(monkeypatch, "anima", seen)
    (tmp_path / "vl.safetensors").write_bytes(b"")
    monkeypatch.setattr(engine_mod, "te_path", lambda name: tmp_path / name)
    out = ENGINE.enhance_prompt("1girl, @artist <lora:a:1>", enhancer_model="vl.safetensors")
    assert out == "1girl, @artist. It sleeps. <lora:a:1>"
    assert seen["mode"] == "expand" and seen["path"].endswith("vl.safetensors")
    with pytest.raises(FileNotFoundError):
        ENGINE.enhance_prompt("1girl", enhancer_model="missing.safetensors")
    # A CUDA-Graphs backbone keeps its addresses: never parked.
    monkeypatch.setattr(ENGINE, "_cuda_graphs", True)
    ENGINE.enhance_prompt("1girl", enhancer_model="vl.safetensors")
    assert seen["make_room"] == ["vae", "te"]


def test_enhance_endpoint_needs_krea2_and_a_prompt(monkeypatch):
    with TestClient(server.app) as c:
        monkeypatch.setattr(ENGINE, "_loaded", None)
        assert c.post("/api/enhance_prompt", json={"prompt": "a fox"}).status_code == 400
        monkeypatch.setattr(ENGINE, "_loaded", LoadedModel(
            name="FLUX(f.safetensors)", family="flux", model=object(), native_res=1024))
        assert c.post("/api/enhance_prompt", json={"prompt": "a fox"}).status_code == 400
        monkeypatch.setattr(ENGINE, "_loaded", _krea2())
        assert c.post("/api/enhance_prompt", json={"prompt": "  "}).status_code == 400


def test_enhance_job_keeps_tag_loras():
    assert "enhance" in server._KEEPS_TAG_LORAS
