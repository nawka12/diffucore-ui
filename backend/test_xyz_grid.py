"""Tests for X/Y/Z axis parsing and cell/plain-generation parity."""

from __future__ import annotations

import pytest

import xyz_grid
from xyz_grid import resolve_values


# ── typed axes ─────────────────────────────────────────────────────

def test_numeric_axes_parse():
    assert resolve_values("Steps", "10, 20,30", 25) == [10, 20, 30]
    assert resolve_values("Seed", "1,2", -1) == [1, 2]
    assert resolve_values("CFG Scale", "3, 4.5", 7.0) == [3.0, 4.5]
    assert resolve_values("SLG Scale", "0, 1.5", 0.0) == [0.0, 1.5]
    assert resolve_values("SLG Blocks", "10, 14+24, 8 9", []) == [[10], [14, 24], [8, 9]]
    assert xyz_grid._label([14, 24]) == "14+24"


def test_empty_or_none_falls_back_to_base():
    assert resolve_values("None", "1,2", 25) == [25]
    assert resolve_values("Steps", "   ", 25) == [25]


# ── a malformed numeric axis must not kill the grid ───────────────────

@pytest.mark.parametrize("param_type,values", [
    ("Steps", "10, twenty, 30"),
    ("Seed", "1, 2.5"),
    ("CFG Scale", "3, high"),
    ("SLG Scale", "0, strong"),
    ("SLG Blocks", "14, -3"),
    ("SLG Blocks", "14, 1.5"),
])
def test_malformed_numeric_axis_names_the_token(param_type, values):
    with pytest.raises(ValueError) as e:
        resolve_values(param_type, values, 1)
    assert param_type in str(e.value)
    assert "invalid literal" not in str(e.value)


@pytest.mark.parametrize("token", ["nan", "inf", "-inf", "Infinity"])
def test_cfg_axis_rejects_non_finite(token):
    """float() accepts these; they'd reach the sampler as CFG."""
    with pytest.raises(ValueError, match="finite"):
        resolve_values("CFG Scale", f"3, {token}", 7.0)


def test_slg_axis_rejects_negative():
    with pytest.raises(ValueError, match="SLG Scale"):
        resolve_values("SLG Scale", "0, -1", 0.0)


# ── Prompt S/R's first value is the search token ────────────────────

def test_prompt_sr_keeps_trailing_empty():
    """A trailing comma means "and a cell without the term"."""
    assert resolve_values("Prompt S/R", "sunny,", "") == ["sunny", ""]


def test_prompt_sr_rejects_empty_search_token():
    """An empty search token made str.replace("", val) splice the replacement
    between every character."""
    with pytest.raises(ValueError, match="cannot be empty"):
        resolve_values("Prompt S/R", ",foo", "")
    with pytest.raises(ValueError, match="cannot be empty"):
        resolve_values("Prompt S/R", " , foo", "")


# ── a cell samples what the Generate page would ─────────────────────
# X/Y/Z cells once skipped the settings-panel knobs and ran at engine defaults.

import inspect
from types import SimpleNamespace

from PIL import Image

import server
import xyz_grid
from engine import Engine

_T2I_SIG = inspect.signature(Engine.generate_t2i)


def _effective(kw: dict) -> dict:
    """The arguments ``generate_t2i`` actually sees, defaults filled in, so an
    omitted kwarg and an explicit default compare equal."""
    bound = _T2I_SIG.bind(None, **kw)
    bound.apply_defaults()
    args = dict(bound.arguments)
    for k in ("self", "progress_callback", "preview_callback"):
        args.pop(k)
    return args


class _FakeEngine:
    loaded_name = "fake.safetensors"
    loaded_family = "anima"
    weights_epoch = 0
    last_seed = 0

    def __init__(self):
        self.calls: list[dict] = []

    def parse_lora_prompt(self, text):
        return text, []

    def apply_temp_loras(self, loras):
        return ""

    def clear_temp_loras(self):
        pass

    def generate_t2i(self, **kw):
        self.calls.append(kw)
        self.last_seed = kw["seed"]
        return Image.new("RGB", (kw["width"], kw["height"])), "info"


def test_xyz_cells_match_plain_generation(monkeypatch):
    fake = _FakeEngine()
    monkeypatch.setattr(server, "ENGINE", fake)
    monkeypatch.setattr(xyz_grid, "ENGINE", fake)
    monkeypatch.setattr(server, "_save_output",
                        lambda image, kwargs, **kw: server.OUTPUTS_DIR / "fake.png")
    monkeypatch.setattr(server.EXTENSIONS, "run_hook",
                        lambda name, **kw: SimpleNamespace(image=kw.get("image")))
    monkeypatch.setattr(server, "_BASE_CACHE", {})
    # Every panel knob off its engine default, so a dropped one shows up.
    for k, v in dict(gate_reduce="per_channel", eta_max=0.5, curvature=0.3,
                     beta_alpha=0.7, beta_beta=0.8, lq_threshold=0.05,
                     cfg_interval_start=0.1, cfg_interval_end=0.75,
                     teacache_uncond_scale=1.5).items():
        monkeypatch.setitem(server.SETTINGS, k, v)

    samplers = ["cogent3_pump", "secant_anneal", "euler"]
    schedulers = ["beta", "linear_quadratic"]
    common = dict(prompt="a cat", neg="blurry", steps=8, cfg=4.5, seed=123,
                  width=64, height=64, shift=3.0, teacache=0.1, preview=False)

    server._run_xyz(server.XYZPayload(
        **common, sampler=samplers[0], scheduler=schedulers[0],
        x_type="Sampler", x_vals=", ".join(samplers),
        y_type="Scheduler", y_vals=", ".join(schedulers),
    ), on_progress=lambda *a: None)
    cells = [_effective(kw) for kw in fake.calls]
    assert len(cells) == len(samplers) * len(schedulers)

    for i, (sch, smp) in enumerate((s, m) for s in schedulers for m in samplers):
        fake.calls.clear()
        server._BASE_CACHE.clear()
        server._run_generation(server.GeneratePayload(
            **common, sampler=smp, scheduler=sch), on_progress=lambda *a: None)
        assert cells[i] == _effective(fake.calls[0]), (smp, sch)

    # Parity alone can't catch a knob both paths drop, so pin the ones that were.
    for cell in cells:
        assert (cell["cfg_interval_start"], cell["cfg_interval_end"]) == (0.1, 0.75)
        assert cell["teacache_uncond_scale"] == 1.5


def test_bad_checkpoint_draws_an_error_cell(monkeypatch):
    """A Checkpoint value that fails to load must not abort the whole sweep."""
    fake = _FakeEngine()

    def reload_model(name, te_name=None):
        if name == "bad.safetensors":
            raise FileNotFoundError("Model not found: bad.safetensors")
        return f"Loaded {name}"

    fake.reload_model = reload_model
    monkeypatch.setattr(xyz_grid, "ENGINE", fake)
    grids, _ = xyz_grid.generate_xyz_grid(
        dict(prompt="a cat", negative_prompt="", width=64, height=64, steps=1,
             cfg_scale=4.0, sampler="euler", scheduler="flow", seed=1, shift=3.0),
        "Checkpoint", "good.safetensors, bad.safetensors",
        "None", "", "None", "",
    )
    assert len(grids) == 1
    assert len(fake.calls) == 1  # only the good cell sampled


def test_checkpoint_and_te_axes_load_once_per_cell(monkeypatch):
    fake = _FakeEngine()
    loads = []
    fake.reload_model = lambda name, te_name=None: loads.append((name, te_name)) or "Loaded"
    monkeypatch.setattr(xyz_grid, "ENGINE", fake)
    xyz_grid.generate_xyz_grid(
        dict(prompt="a cat", negative_prompt="", width=64, height=64, steps=1,
             cfg_scale=4.0, sampler="euler", scheduler="flow", seed=1, shift=3.0),
        "Text Encoder", "a.safetensors, b.safetensors",
        "Checkpoint", "d1.safetensors, d2.safetensors",
        "None", "",
    )
    assert loads == [("d1.safetensors", "a.safetensors"), ("d1.safetensors", "b.safetensors"),
                     ("d2.safetensors", "a.safetensors"), ("d2.safetensors", "b.safetensors")]
    assert len(fake.calls) == 4


def test_te_axis_restores_the_users_model(monkeypatch):
    fake = _FakeEngine()
    fake.reload_model = lambda name, te_name=None: "Loaded"
    monkeypatch.setattr(server, "ENGINE", fake)
    monkeypatch.setattr(xyz_grid, "ENGINE", fake)
    monkeypatch.setattr(server, "_save_output",
                        lambda image, kwargs, **kw: server.OUTPUTS_DIR / "fake.png")
    restored = []
    monkeypatch.setattr(server, "_do_load", lambda p: restored.append(p.te) or "Loaded")
    monkeypatch.setattr(server, "_push", lambda ev: None)
    monkeypatch.setattr(server, "_state_payload", lambda: {})
    monkeypatch.setattr(server, "LAST_LOAD_FORM", dict(model_type="Anima", dit="d.safetensors",
                                                       vae="v.safetensors", te="t.safetensors"))
    server._run_xyz(server.XYZPayload(
        prompt="a cat", steps=1, width=64, height=64, seed=1, preview=False,
        x_type="Text Encoder", x_vals="a.safetensors, b.safetensors",
    ), on_progress=lambda *a: None)
    assert restored == ["t.safetensors"]


def test_te_axis_needs_a_split_file_model(monkeypatch):
    fake = _FakeEngine()
    fake.loaded_family = "sdxl"
    monkeypatch.setattr(server, "ENGINE", fake)
    with pytest.raises(RuntimeError, match="Text Encoder"):
        server._run_xyz(server.XYZPayload(
            prompt="a cat", x_type="Text Encoder", x_vals="a.safetensors",
        ), on_progress=lambda *a: None)


# ── SLG axes: every cell samples what Generate would ────────────────

@pytest.mark.parametrize("family, form", [
    ("anima", dict(slg_layers=[14], slg_start=0.1, slg_stop=0.8)),
    ("anima", dict(slg_layers=[14], uncond_skip_layers=[15, 16, 17], slg_start=0.35, slg_stop=0.7)),
    ("moss_image", dict(slg_layers=[24], uncond_skip_layers=[22, 30],
                        slg_start=0.1, slg_stop=0.8)),
])
def test_slg_axes_cells_match_plain_generation(monkeypatch, family, form):
    fake = _FakeEngine()
    fake.loaded_family = family
    monkeypatch.setattr(server, "ENGINE", fake)
    monkeypatch.setattr(xyz_grid, "ENGINE", fake)
    monkeypatch.setattr(server, "_save_output",
                        lambda image, kwargs, **kw: server.OUTPUTS_DIR / "fake.png")
    monkeypatch.setattr(server.EXTENSIONS, "run_hook",
                        lambda name, **kw: SimpleNamespace(image=kw.get("image")))
    monkeypatch.setattr(server, "_BASE_CACHE", {})
    common = dict(prompt="a cat", neg="blurry", steps=8, cfg=4.5, seed=123, width=64,
                  height=64, sampler="euler", scheduler="beta", preview=False)
    scales, blocks = [0.0, 1.5], [[10], [14, 24]]

    server._run_xyz(server.XYZPayload(
        **common, **form, slg_scale=3.0,
        x_type="SLG Scale", x_vals="0, 1.5", y_type="SLG Blocks", y_vals="10, 14+24",
    ), on_progress=lambda *a: None)
    cells = [_effective(kw) for kw in fake.calls]
    assert len(cells) == len(scales) * len(blocks)

    for cell, (layers, scale) in zip(cells, ((b, s) for b in blocks for s in scales)):
        fake.calls.clear()
        server._BASE_CACHE.clear()
        server._run_generation(server.GeneratePayload(
            **common, **{**form, "slg_layers": layers}, slg_scale=scale),
            on_progress=lambda *a: None)
        assert cell == _effective(fake.calls[0]), (layers, scale)
        if scale:
            assert (cell["slg_scale"], cell["slg_layers"]) == (scale, layers)
        else:
            assert (cell["slg_scale"], cell["slg_layers"]) == (0.0, None)


def test_xyz_cells_use_the_forms_slg(monkeypatch):
    """Without an SLG axis, cells run the form's SLG, as Generate does."""
    fake = _FakeEngine()
    monkeypatch.setattr(server, "ENGINE", fake)
    monkeypatch.setattr(xyz_grid, "ENGINE", fake)
    monkeypatch.setattr(server, "_save_output",
                        lambda image, kwargs, **kw: server.OUTPUTS_DIR / "fake.png")
    server._run_xyz(server.XYZPayload(
        prompt="a cat", steps=1, width=64, height=64, seed=1, preview=False,
        slg_layers=[14], slg_scale=1.5, x_type="Seed", x_vals="1, 2",
    ), on_progress=lambda *a: None)
    assert [(kw["slg_layers"], kw["slg_scale"]) for kw in fake.calls] == [([14], 1.5)] * 2


@pytest.mark.parametrize("family, form, axis, error", [
    ("sdxl", dict(slg_layers=[14]), ("SLG Scale", "0, 1.5"), "Anima or MOSS-Image"),
    ("sdxl", dict(slg_scale=1.5), ("SLG Blocks", "10, 14"), "Anima or MOSS-Image"),
    ("anima", dict(slg_layers=[]), ("SLG Scale", "0, 1.5"), "SLG blocks"),
    ("anima", dict(slg_scale=0.0), ("SLG Blocks", "10, 14"), "scale above 0"),
])
def test_slg_axes_need_an_slg_model_and_the_other_half(monkeypatch, family, form, axis, error):
    fake = _FakeEngine()
    fake.loaded_family = family
    monkeypatch.setattr(server, "ENGINE", fake)
    with pytest.raises(RuntimeError, match=error):
        server._run_xyz(server.XYZPayload(
            prompt="a cat", **form, x_type=axis[0], x_vals=axis[1],
        ), on_progress=lambda *a: None)


def test_xyz_slg_for_the_submit_guard():
    p = server.XYZPayload(slg_scale=0.0, slg_layers=[14], x_type="SLG Scale", x_vals="0, 1.5")
    assert server._xyz_slg(p) == (1.5, [14])
    # An axis replaces the form's value, so a form value no cell uses is ignored.
    p = server.XYZPayload(slg_scale=2.0, slg_layers=[14], y_type="SLG Scale", y_vals="0")
    assert server._xyz_slg(p) == (0.0, [14])
    p = server.XYZPayload(slg_scale=2.0, z_type="SLG Blocks", z_vals="18, 10+14")
    assert server._xyz_slg(p) == (2.0, [10, 14, 18])
    assert server._xyz_slg(server.XYZPayload(slg_scale=2.0)) == (2.0, [])
