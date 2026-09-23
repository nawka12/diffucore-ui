"""A same-DiT load that swaps the VAE, TE or CLIP must reload, not be skipped
as "already loaded". A reload shows up as a ``FileNotFoundError`` from the
companion-file check; a true skip returns before touching the filesystem.
"""
from __future__ import annotations

import pytest

from engine import Engine, LoadedModel

_MISSING = "____nonexistent____.safetensors"


def _engine_with(loaded: LoadedModel | None, offload: bool | str = True) -> Engine:
    eng = Engine(device="cpu")
    eng._loaded = loaded
    # The engine stores the bundle offload value (True/False/"encoders"/
    # "stream"), so only the component change is under test.
    eng._offload = offload
    eng._vae_tile = True
    return eng


def _anima(dit="d.safetensors", vae="v.safetensors", te="t.safetensors") -> LoadedModel:
    return LoadedModel(name=f"Anima({dit})", family="anima", model=object(),
                       native_res=1024, vae_name=vae, te_name=te)


# ── the component-equality helper ──────────────────────────────────────────

def test_components_match_anima():
    eng = _engine_with(_anima(vae="v", te="t"))
    assert eng._components_match("v", "t")
    assert not eng._components_match("v2", "t")     # VAE changed
    assert not eng._components_match("v", "t2")     # TE changed


def test_components_match_flux_includes_clip():
    eng = _engine_with(LoadedModel(name="FLUX(d)", family="flux1", model=object(),
                                   native_res=1024, vae_name="v", te_name="t", clip_name="c"))
    assert eng._components_match("v", "t", "c")
    assert not eng._components_match("v", "t", "c2")   # CLIP changed
    assert not eng._components_match("v", "t", None)   # CLIP cleared


# ── load_anima skip vs reload ──────────────────────────────────────────────

def test_load_anima_skips_when_everything_matches():
    eng = _engine_with(_anima())
    msg = eng.load_anima("d.safetensors", "v.safetensors", "t.safetensors",
                         offload=True, vae_tile=True)
    assert msg.startswith("Model already loaded")


@pytest.mark.parametrize("vae,te", [
    (_MISSING, "t.safetensors"),   # VAE swapped
    ("v.safetensors", _MISSING),   # TE swapped
])
def test_load_anima_reloads_when_component_changes(vae, te):
    eng = _engine_with(_anima())
    # Same DiT, changed component: must reach the companion-file check.
    with pytest.raises(FileNotFoundError):
        eng.load_anima("d.safetensors", vae, te, offload=True, vae_tile=True)


def test_load_anima_drops_stale_cache_entry_on_component_change():
    """A cached DiT with a different VAE/TE must be invalidated, not restored."""
    eng = _engine_with(None)
    label = f"Anima({_MISSING})"
    eng._ckpt_cache[label] = _anima(dit=_MISSING, vae="v_old.safetensors", te="t.safetensors")
    with pytest.raises(FileNotFoundError):
        eng.load_anima(_MISSING, "v_new.safetensors", "t.safetensors",
                       offload=True, vae_tile=True)
    assert label not in eng._ckpt_cache


# ── load_flux skip vs reload (CLIP is a component too) ─────────────────────

def test_load_flux_reloads_when_clip_changes():
    eng = _engine_with(LoadedModel(name=f"FLUX({_MISSING})", family="flux1", model=object(),
                                   native_res=1024, vae_name="v.safetensors",
                                   te_name="t.safetensors", clip_name="c.safetensors"))
    with pytest.raises(FileNotFoundError):
        eng.load_flux(_MISSING, "v.safetensors", "t.safetensors",
                      clip_name="c_new.safetensors", offload=True, vae_tile=True)
