"""Tests for X/Y/Z axis value parsing (pure functions, no GPU needed).

Run from the project root::

    .venv/bin/python -m pytest backend/test_xyz_grid.py -v
"""

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


# ── BUG.md L14: a malformed numeric axis must not kill the grid ─────

@pytest.mark.parametrize("param_type,values", [
    ("Steps", "10, twenty, 30"),
    ("Seed", "1, 2.5"),
    ("CFG Scale", "3, high"),
])
def test_malformed_numeric_axis_names_the_token(param_type, values):
    with pytest.raises(ValueError) as e:
        resolve_values(param_type, values, 1)
    assert param_type in str(e.value)
    assert "invalid literal" not in str(e.value)   # not the bare int() message


@pytest.mark.parametrize("token", ["nan", "inf", "-inf", "Infinity"])
def test_cfg_axis_rejects_non_finite(token):
    """float() happily accepts these; they'd flow into the sampler as CFG."""
    with pytest.raises(ValueError, match="finite"):
        resolve_values("CFG Scale", f"3, {token}", 7.0)


# ── BUG.md L15: Prompt S/R's first value is the search token ────────

def test_prompt_sr_keeps_trailing_empty():
    """A trailing comma means "and a cell without the term" — still supported."""
    assert resolve_values("Prompt S/R", "sunny,", "") == ["sunny", ""]


def test_prompt_sr_rejects_empty_search_token():
    """A leading empty token made str.replace("", val) splice the replacement
    between every character of the prompt."""
    with pytest.raises(ValueError, match="cannot be empty"):
        resolve_values("Prompt S/R", ",foo", "")
    with pytest.raises(ValueError, match="cannot be empty"):
        resolve_values("Prompt S/R", " , foo", "")
