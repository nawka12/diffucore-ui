"""Tests for X/Y/Z axis parsing and cell/plain-generation parity."""

from __future__ import annotations

import pytest

from xyz_grid import resolve_values


# ── typed axes ─────────────────────────────────────────────────────

def test_numeric_axes_parse():
    assert resolve_values("Steps", "10, 20,30", 25) == [10, 20, 30]
    assert resolve_values("Seed", "1,2", -1) == [1, 2]
    assert resolve_values("CFG Scale", "3, 4.5", 7.0) == [3.0, 4.5]


def test_empty_or_none_falls_back_to_base():
    assert resolve_values("None", "1,2", 25) == [25]
    assert resolve_values("Steps", "   ", 25) == [25]


# ── a malformed numeric axis must not kill the grid ───────────────────

@pytest.mark.parametrize("param_type,values", [
    ("Steps", "10, twenty, 30"),
    ("Seed", "1, 2.5"),
    ("CFG Scale", "3, high"),
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
