"""Output folder naming: ISO date dirs, 5-digit counters, and the startup
migration of legacy ``DD-MM-YYYY`` folders (v0.1.7 naming change).

Run from the project root::

    .venv/bin/python -m pytest backend/test_outputs_naming.py -v
"""

from __future__ import annotations

from datetime import date

import server
import utils


# ── new naming convention ─────────────────────────────────────────────

def test_next_output_path_iso_folder_and_5digit_counter(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "OUTPUTS_DIR", tmp_path / "outputs")
    p = utils.next_output_path(123)
    assert p.parent.name == date.today().isoformat()
    assert p.name == "00001-123.png"
    p.write_bytes(b"x")
    assert utils.next_output_path(7).name == "00002-7.png"


def test_counter_continues_after_legacy_2digit_files(tmp_path, monkeypatch):
    # A migrated folder can hold pre-ISO ``NN-seed.png`` files; the counter
    # must pick up after them, not collide.
    monkeypatch.setattr(utils, "OUTPUTS_DIR", tmp_path / "outputs")
    day = tmp_path / "outputs" / date.today().isoformat()
    day.mkdir(parents=True)
    (day / "07-99.png").write_bytes(b"x")
    assert utils.next_output_path(1).name == "00008-1.png"


def test_parse_date_dir_iso_and_fallback():
    assert utils._parse_date_dir("2026-07-04") == date(2026, 7, 4)
    assert utils._parse_date_dir("04-07-2026") == date.min  # legacy: sorts last
    assert utils._parse_date_dir("notes") == date.min


# ── startup migration ─────────────────────────────────────────────────

def test_migrate_renames_legacy_folders_and_thumbs(tmp_path, monkeypatch):
    out = tmp_path / "outputs"
    (out / "04-07-2026").mkdir(parents=True)
    (out / "04-07-2026" / "01-123.png").write_bytes(b"x")
    (out / "2026-07-03").mkdir()   # already ISO
    (out / "notes").mkdir()        # not date-shaped
    (out / "99-99-2026").mkdir()   # date-shaped but not a real date
    thumbs = tmp_path / "thumbs"
    (thumbs / "04-07-2026").mkdir(parents=True)
    (thumbs / "04-07-2026" / "01-123_1_1.webp").write_bytes(b"t")
    monkeypatch.setattr(server, "OUTPUTS_DIR", out)
    monkeypatch.setattr(server, "_THUMBS_DIR", thumbs)

    assert server._migrate_output_dirs() == 1
    assert not (out / "04-07-2026").exists()
    assert (out / "2026-07-04" / "01-123.png").is_file()
    assert (out / "2026-07-03").is_dir()    # untouched
    assert (out / "notes").is_dir()         # untouched
    assert (out / "99-99-2026").is_dir()    # untouched
    # Thumb-cache mirror moved with it (keys are stem+mtime+size → still valid).
    assert (thumbs / "2026-07-04" / "01-123_1_1.webp").is_file()


def test_migrate_merges_into_existing_iso_folder(tmp_path, monkeypatch):
    out = tmp_path / "outputs"
    (out / "04-07-2026").mkdir(parents=True)
    (out / "04-07-2026" / "01-1.png").write_bytes(b"a")
    (out / "2026-07-04").mkdir()
    (out / "2026-07-04" / "00002-2.png").write_bytes(b"b")
    monkeypatch.setattr(server, "OUTPUTS_DIR", out)
    monkeypatch.setattr(server, "_THUMBS_DIR", tmp_path / "thumbs")

    assert server._migrate_output_dirs() == 1
    assert not (out / "04-07-2026").exists()
    names = {f.name for f in (out / "2026-07-04").iterdir()}
    assert names == {"01-1.png", "00002-2.png"}


def test_migrate_noop_when_all_iso(tmp_path, monkeypatch):
    out = tmp_path / "outputs"
    (out / "2026-07-04").mkdir(parents=True)
    monkeypatch.setattr(server, "OUTPUTS_DIR", out)
    monkeypatch.setattr(server, "_THUMBS_DIR", tmp_path / "thumbs")
    assert server._migrate_output_dirs() == 0
