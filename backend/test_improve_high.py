"""Tests for extension install/update, share URL, gallery trash, shutdown
partial save, alpha decode, queue priority, logging, the ratings pipeline and
the X/Y/Z checkpoint cache.

    .venv/bin/python -m pytest backend/test_improve_high.py -v
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
from pathlib import Path

import pytest
from PIL import Image

import server


def test_cogent4_setting_defaults_global_and_rejects_unknown_modes():
    assert server.Settings().gate_reduce == "all"
    assert server.Settings(gate_reduce="per_channel").gate_reduce == "per_channel"
    with pytest.raises(ValueError):
        server.Settings(gate_reduce="bogus")



def test_blur_min_rating_defaults_to_r_and_rejects_unknown_tiers():
    assert server.Settings().blur_min_rating == "R"
    for tier in ("PG13", "R", "X"):
        assert server.Settings(blur_min_rating=tier).blur_min_rating == tier
    # XXX is not offered: the AI rater never outputs it, so it would blur nothing.
    for bad in ("XXX", "PG", "pg13"):
        with pytest.raises(ValueError):
            server.Settings(blur_min_rating=bad)

# ── opt-in pip + install routed through the job queue ─────────────────

def test_install_payload_pip_deps_defaults_off():
    from extensions import InstallPayload
    p = InstallPayload(url="https://github.com/foo/bar.git")
    assert p.install_pip_deps is False
    q = InstallPayload(url="https://x.zip", install_pip_deps=True)
    assert q.install_pip_deps is True


def test_install_endpoint_rejects_non_https_with_400(client):
    r = client.post("/api/extensions/install",
                    json={"url": "file:///etc/passwd"})
    assert r.status_code == 400


# ── uninstall must never target extensions/ itself ─────────────────────

@pytest.mark.parametrize("name", [".", "", "..", "sub/dir", "../elsewhere"])
def test_uninstall_rejects_non_child_names(tmp_path, monkeypatch, name):
    """``name="."`` once resolved to EXTENSIONS_DIR itself and wiped it."""
    import extensions as extmod

    ext_dir = tmp_path / "extensions"
    (ext_dir / "keeper").mkdir(parents=True)
    monkeypatch.setattr(extmod, "EXTENSIONS_DIR", ext_dir)
    monkeypatch.setattr(extmod, "STATE_PATH", ext_dir / "state.json")
    loader = extmod.ExtensionLoader(None, enqueue_job=lambda *a, **k: 0,
                                    broadcast=lambda ev: None)
    (ext_dir / "state.json").write_text("{}")

    with pytest.raises(ValueError):
        loader.uninstall(name)
    assert (ext_dir / "keeper").is_dir()
    assert (ext_dir / "state.json").is_file()


def test_uninstall_endpoint_returns_400_not_500(client):
    r = client.post("/api/extensions/uninstall", json={"name": "."})
    assert r.status_code == 400


def test_install_endpoint_enqueues_install_job_and_passes_pip_flag(client, monkeypatch):
    captured = {}

    class _FakeExt:
        def to_dict(self):
            return {"name": "fake", "loaded": True, "has_ui": False}

    def fake_install(url, *, install_pip_deps=False):
        captured["url"] = url
        captured["install_pip_deps"] = install_pip_deps
        return _FakeExt()

    monkeypatch.setattr(server.EXTENSIONS, "install", fake_install)
    monkeypatch.setattr(server.EXTENSIONS, "mount_into", lambda app: None)

    enqueued = {}
    def fake_enqueue(job):
        enqueued["job"] = job
    monkeypatch.setattr(server, "_enqueue", fake_enqueue)

    r = client.post("/api/extensions/install",
                    json={"url": "https://github.com/foo/bar.git",
                          "install_pip_deps": True})
    assert r.status_code == 200
    assert "job" in r.json()
    job = enqueued["job"]
    assert job.kind == "install"  # routed through the queue, not the request thread
    result = job.run(job)
    assert captured["install_pip_deps"] is True
    assert result["extension"]["name"] == "fake"


def test_install_endpoint_defaults_pip_off(client, monkeypatch):
    captured = {}
    class _FakeExt:
        def to_dict(self): return {"name": "fake", "loaded": True, "has_ui": False}
    def fake_install(url, *, install_pip_deps=False):
        captured["pip"] = install_pip_deps
        return _FakeExt()
    monkeypatch.setattr(server.EXTENSIONS, "install", fake_install)
    monkeypatch.setattr(server.EXTENSIONS, "mount_into", lambda app: None)
    held = {}
    monkeypatch.setattr(server, "_enqueue", lambda job: held.__setitem__("job", job))
    client.post("/api/extensions/install",
                json={"url": "https://github.com/foo/bar.git"})
    held["job"].run(held["job"])
    assert captured["pip"] is False


def test_update_endpoint_enqueues_update_job_and_passes_pip_flag(client, monkeypatch):
    server.EXTENSIONS.extensions["fake"] = object()
    captured = {}

    class _FakeExt:
        def to_dict(self):
            return {"name": "fake", "version": "1.2.0", "loaded": True, "has_ui": False}

    def fake_update(name, *, install_pip_deps=False):
        captured["name"] = name
        captured["install_pip_deps"] = install_pip_deps
        return _FakeExt()

    monkeypatch.setattr(server.EXTENSIONS, "update", fake_update)
    monkeypatch.setattr(server.EXTENSIONS, "mount_into", lambda app: None)
    held = {}
    monkeypatch.setattr(server, "_enqueue", lambda job: held.__setitem__("job", job))

    assert client.post("/api/extensions/update", json={"name": "nope"}).status_code == 404

    r = client.post("/api/extensions/update",
                    json={"name": "fake", "install_pip_deps": True})
    assert r.status_code == 200
    job = held["job"]
    assert job.kind == "update"  # serialized through the queue, not the request thread
    result = job.run(job)
    assert captured == {"name": "fake", "install_pip_deps": True}
    assert result["extension"]["version"] == "1.2.0"
    server.EXTENSIONS.extensions.pop("fake", None)


def test_update_rejects_non_git_checkout(monkeypatch, tmp_path):
    import extensions as extmod
    loader = extmod.ExtensionLoader.__new__(extmod.ExtensionLoader)
    loader.extensions = {"z": extmod.Extension(name="z", title="z", version="1",
                                               path=tmp_path / "z")}
    (tmp_path / "z").mkdir()
    monkeypatch.setattr(extmod, "EXTENSIONS_DIR", tmp_path)
    with pytest.raises(ValueError, match="git-only"):
        loader.update("z")


def test_install_skips_pip_by_default_and_notes_requirements(monkeypatch, tmp_path):
    # Pre-create the scratch dir (manifest + requirements.txt) instead of cloning.
    import extensions as extmod

    ext_dir = tmp_path / "exts"
    ext_dir.mkdir()
    monkeypatch.setattr(extmod, "EXTENSIONS_DIR", ext_dir)
    monkeypatch.setattr(extmod, "STATE_PATH", ext_dir / "state.json")

    loader = extmod.ExtensionLoader.__new__(extmod.ExtensionLoader)
    loader.extensions = {}
    loader._state = {"enabled": {}, "ext_settings": {}}
    loader._hooks = {}
    loader._routers = []
    loader._statics = []
    loader._mounted = set()
    loader.engine = None
    loader._enqueue_job_fn = lambda *a, **k: 0
    loader._broadcast_fn = lambda *a, **k: None

    def fake_git(url, target):
        target.mkdir(parents=True, exist_ok=True)
        (target / "extension.json").write_text(json.dumps({"name": "demo"}))
        (target / "extension.py").write_text("def setup(api): pass\n")
        (target / "requirements.txt").write_text("numpy\n")
    monkeypatch.setattr(extmod.ExtensionLoader, "_install_git", staticmethod(fake_git))
    monkeypatch.setattr(extmod.ExtensionLoader, "_install_zip", staticmethod(lambda *a: None))
    monkeypatch.setattr(extmod.ExtensionLoader, "_pip_install_requirements",
                        staticmethod(lambda *a, **k: pytest.fail("pip must not run by default")))

    ext = loader.install("https://github.com/foo/demo.git")
    assert ext.name == "demo"
    assert ext.load_error and "requirements.txt" in ext.load_error
    assert "opt-in" in ext.load_error


# ── share URL written to a chmod 600 file, not stdout ─────────────────

def test_share_url_file_is_chmod_600_and_contains_url(monkeypatch, tmp_path):
    import share
    monkeypatch.setattr(share, "_BIN_DIR", tmp_path)
    f = share._write_share_url_file("https://abc.trycloudflare.com?token=secret")
    assert f is not None and f.is_file()
    assert f.read_text().strip() == "https://abc.trycloudflare.com?token=secret"
    if os.name == "posix":
        assert (f.stat().st_mode & 0o777) == 0o600


def test_share_warning_omits_url_when_file_written(monkeypatch, tmp_path, capsys):
    import share
    monkeypatch.setattr(share, "_BIN_DIR", tmp_path)
    share._print_share_warning("https://abc.trycloudflare.com", "?token=secret")
    out = capsys.readouterr().out
    assert "share_url.txt" in out
    assert "secret" not in out
    assert "trycloudflare.com?token" not in out


def test_share_warning_falls_back_to_printing_url_on_file_failure(monkeypatch, tmp_path, capsys):
    import share
    monkeypatch.setattr(share, "_BIN_DIR", tmp_path)
    def boom(_url):
        return None
    monkeypatch.setattr(share, "_write_share_url_file", boom)
    share._print_share_warning("https://abc.trycloudflare.com", "?token=tok")
    out = capsys.readouterr().out
    assert "https://abc.trycloudflare.com?token=tok" in out  # fallback prints it
    assert "WARNING" in out


# ── gallery soft-delete, trash purge, scan_outputs skips dot-dirs ─────

def _setup_outputs(tmp_path, monkeypatch):
    out = tmp_path / "outputs"
    out.mkdir()
    day = out / "01-01-2026"
    day.mkdir()
    img = Image.new("RGB", (8, 8), (10, 20, 30))
    img.save(day / "01-123.png")
    monkeypatch.setattr(server, "OUTPUTS_DIR", out)
    monkeypatch.setattr(server, "_TRASH_DIR", out / ".trash")
    monkeypatch.setattr(server, "_THUMBS_DIR", tmp_path / "thumbs")
    return out, day / "01-123.png"


def test_gallery_delete_moves_to_trash(monkeypatch, tmp_path):
    _setup_outputs(tmp_path, monkeypatch)
    monkeypatch.setattr(server, "_invalidate_gallery_index", lambda: None)
    monkeypatch.setattr(server, "_purge_trash", lambda *a, **k: 0)
    target = server.OUTPUTS_DIR / "01-01-2026" / "01-123.png"
    assert target.is_file()
    from fastapi import HTTPException
    resp = server.api_gallery_delete(path="01-01-2026/01-123.png")
    assert not target.exists()  # gone from the gallery
    trash = server._TRASH_DIR
    assert trash.is_dir()
    trashed = list(trash.iterdir())
    assert len(trashed) == 1
    assert trashed[0].name.endswith("_01-123.png")
    assert resp["deleted"] == "01-01-2026/01-123.png"


# ── gallery nsfw flags (prompt-derived rating for the blur) ──────────

def _save_with_params(path, params: str) -> None:
    """Save a tiny PNG whose ``parameters`` chunk carries ``params``."""
    from PIL.PngImagePlugin import PngInfo
    meta = PngInfo()
    meta.add_text("parameters", params)
    Image.new("RGB", (8, 8), (10, 20, 30)).save(path, pnginfo=meta)


def test_gallery_entries_carry_prompt_rating(monkeypatch, tmp_path):
    import utils
    out, target = _setup_outputs(tmp_path, monkeypatch)
    monkeypatch.setattr(utils, "OUTPUTS_DIR", out)
    _save_with_params(
        target,
        "nude woman, portrait\nNegative prompt: landscape\n"
        "Steps: 20, Sampler: euler, Seed: 1",
    )
    server._invalidate_gallery_index()
    utils.invalidate_outputs_cache()

    images = server.api_gallery()["images"]
    assert len(images) == 1
    e = images[0]
    assert e["nsfw"] is True
    assert e["rating"] == "R"
    assert e["url"] == "/outputs/01-01-2026/01-123.png"

    # A clean prompt stays unflagged (and "ass" inside "badass" must not fire).
    _save_with_params(
        out / "01-01-2026" / "01-124.png",
        "a badass cat in a field\nSteps: 20, Sampler: euler, Seed: 2",
    )
    utils.invalidate_outputs_cache()
    images = server.api_gallery()["images"]
    assert len(images) == 2
    sfw = next(i for i in images if i["name"] == "01-124.png")
    assert sfw["nsfw"] is False
    assert sfw["rating"] == "PG"

    hit = server.api_gallery(q="nude")["images"]
    assert len(hit) == 1 and hit[0]["nsfw"] is True


# ── AI NSFW rating: cache, tag jobs, gallery scan ──────────────────

@pytest.fixture
def _isolated_ratings(monkeypatch, tmp_path):
    """Point the rating cache at a throwaway file and reset the in-memory copy."""
    monkeypatch.setattr(server, "_RATINGS_PATH", tmp_path / "ratings.json")
    monkeypatch.setattr(server, "_RATINGS", None)
    yield
    monkeypatch.setattr(server, "_RATINGS", None)


def _fake_tagger(rate):
    class _Fake:
        loaded = False
        def load(self): self.loaded = True
        def unload(self): self.loaded = False
        def rate_one(self, path): return rate(path)
    return _Fake()


def test_gallery_index_prefers_vision_rating(monkeypatch, tmp_path, _isolated_ratings):
    import utils
    out, target = _setup_outputs(tmp_path, monkeypatch)
    monkeypatch.setattr(utils, "OUTPUTS_DIR", out)
    # Prompt says safe, so without a vision rating the image stays unblurred...
    _save_with_params(target, "a cat\nSteps: 20, Sampler: euler, Seed: 1")
    server._invalidate_gallery_index()
    utils.invalidate_outputs_cache()
    assert server._gallery_index()[0]["nsfw"] is False
    # ...but a cached vision verdict overrides the prompt.
    server._store_ratings({"01-01-2026/01-123.png": {
        "rating": "X", "nsfw": True, "conf": 0.9,
        "key": server._image_key(target)}})
    server._invalidate_gallery_index()
    entry = server._gallery_index()[0]
    assert entry["rating"] == "X"
    assert entry["nsfw"] is True
    # A file changed since rating busts the entry.
    server._store_ratings({"01-01-2026/01-123.png": {
        "rating": "X", "nsfw": True, "conf": 0.9, "key": "stale-key"}})
    server._invalidate_gallery_index()
    assert server._gallery_index()[0]["nsfw"] is False


def test_cached_rating_invalidated_by_decision_version(monkeypatch, tmp_path,
                                                       _isolated_ratings):
    # A verdict from an older decision layer is stale even for an unchanged file.
    import utils
    out, target = _setup_outputs(tmp_path, monkeypatch)
    monkeypatch.setattr(utils, "OUTPUTS_DIR", out)
    server._store_ratings({"01-01-2026/01-123.png": {
        "rating": "X", "nsfw": True, "conf": 0.5, "key": server._image_key(target)}})
    assert server._cached_rating(target)["rating"] == "X"
    monkeypatch.setattr(server.tagger_mod, "DECISION_VERSION",
                        server.tagger_mod.DECISION_VERSION + 1)
    assert server._cached_rating(target) is None
    server._store_ratings({"01-01-2026/01-123.png": {
        "rating": "PG", "nsfw": False, "conf": 0.9, "key": server._image_key(target)}})
    assert server._cached_rating(target)["rating"] == "PG"


def test_maybe_auto_rescan_rerates_stale_only(_isolated_queue, monkeypatch,
                                              tmp_path, _isolated_ratings):
    import utils
    out, target = _setup_outputs(tmp_path, monkeypatch)
    monkeypatch.setattr(utils, "OUTPUTS_DIR", out)
    monkeypatch.setattr(server.tagger_mod, "timm_available", lambda: True)
    monkeypatch.setitem(server.SETTINGS, "nsfw_blur", True)
    server._maybe_auto_rescan()
    with server.QUEUE_LOCK:
        assert len(server.QUEUE) == 0
    server._store_ratings({"01-01-2026/01-123.png": {
        "rating": "PG", "nsfw": False, "conf": 0.9, "key": server._image_key(target)}})
    server._maybe_auto_rescan()
    with server.QUEUE_LOCK:
        assert len(server.QUEUE) == 0
    monkeypatch.setattr(server.tagger_mod, "DECISION_VERSION",
                        server.tagger_mod.DECISION_VERSION + 1)
    server._maybe_auto_rescan()
    with server.QUEUE_LOCK:
        job = server.QUEUE[0]
    assert job.kind == "tag"
    assert job.priority == -10


def test_enqueue_tag_job_gated_by_timm(_isolated_queue, monkeypatch,
                                       tmp_path, _isolated_ratings):
    out, target = _setup_outputs(tmp_path, monkeypatch)
    monkeypatch.setattr(server.tagger_mod, "timm_available", lambda: True)
    assert server._enqueue_tag_job([target], "Rate") is not None
    with server.QUEUE_LOCK:
        job = server.QUEUE[0]
    assert job.kind == "tag"
    assert job.priority == -10  # never delays a generation
    assert job.total == 1
    # timm missing: falls back to the prompt heuristic.
    monkeypatch.setattr(server.tagger_mod, "timm_available", lambda: False)
    assert server._enqueue_tag_job([target], "Rate") is None
    monkeypatch.setattr(server.tagger_mod, "timm_available", lambda: True)
    monkeypatch.setitem(server.SETTINGS, "nsfw_blur", False)
    assert server._enqueue_tag_job([target], "Rate") is not None


def test_auto_tag_wanted_by_either_blur_surface(_isolated_queue, monkeypatch,
                                                tmp_path, _isolated_ratings):
    # The gallery setting and the page's "Blur NSFW" toggle are independent;
    # either one is enough.
    out, target = _setup_outputs(tmp_path, monkeypatch)
    monkeypatch.setattr(server.tagger_mod, "timm_available", lambda: True)
    monkeypatch.setattr(server, "_PENDING_TAG", [])

    monkeypatch.setitem(server.SETTINGS, "nsfw_blur", False)
    server._maybe_auto_tag(target, blur_check=False)
    assert server._PENDING_TAG == []
    server._maybe_auto_tag(target, blur_check=True)
    assert server._PENDING_TAG == [target]
    server._PENDING_TAG.clear()
    monkeypatch.setitem(server.SETTINGS, "nsfw_blur", True)
    server._maybe_auto_tag(target, blur_check=False)
    assert server._PENDING_TAG == [target]
    server._PENDING_TAG.clear()
    monkeypatch.setattr(server.tagger_mod, "timm_available", lambda: False)
    server._maybe_auto_tag(target, blur_check=True)
    assert server._PENDING_TAG == []


def test_tag_job_run_rates_caches_and_invalidates(monkeypatch, tmp_path,
                                                  _isolated_ratings):
    import utils
    out, target = _setup_outputs(tmp_path, monkeypatch)
    monkeypatch.setattr(utils, "OUTPUTS_DIR", out)
    monkeypatch.setattr(
        server.tagger_mod, "TAGGER",
        _fake_tagger(lambda path: {"rating": "R", "nsfw": True, "confidence": 0.9}))
    monkeypatch.setattr(server, "_push", lambda ev: None)
    patched = []
    monkeypatch.setattr(server, "_gallery_index_patch_ratings",
                        lambda r: patched.append(r))
    job = server.Job("tag", "rate", lambda j: {})
    job.total = 1
    assert server._tag_job_run(job, [target]) == {"rated": 1}
    # Patched in place: a rebuild would re-open every PNG in outputs/.
    assert len(patched) == 1
    assert patched[0]["01-01-2026/01-123.png"]["rating"] == "R"
    entry = server._read_ratings()["01-01-2026/01-123.png"]
    assert entry["rating"] == "R" and entry["nsfw"] is True
    assert entry["key"] == server._image_key(target)


def test_gallery_index_add_splices_without_rebuilding(monkeypatch, tmp_path,
                                                      _isolated_ratings):
    # A save must not drop the index: rebuilding re-opens every PNG.
    import utils
    out, target = _setup_outputs(tmp_path, monkeypatch)
    monkeypatch.setattr(utils, "OUTPUTS_DIR", out)
    _save_with_params(target, "a cat\nSteps: 20, Sampler: euler, Seed: 1")
    server._invalidate_gallery_index()
    utils.invalidate_outputs_cache()
    assert len(server._gallery_index()) == 1

    # Reading a PNG now would mean a rebuild.
    monkeypatch.setattr(server.md, "read_png_metadata",
                        lambda p: pytest.fail("rebuilt the index instead of splicing"))
    newer = out / "01-01-2026" / "01-124.png"
    _save_with_params(newer, "nude woman\nSteps: 20, Sampler: euler, Seed: 2")
    server._gallery_index_add(newer, "nude woman\nSteps: 20, Sampler: euler, Seed: 2")

    index = server._gallery_index()
    assert len(index) == 2
    assert index[0]["path"] == "01-01-2026/01-124.png"
    assert index[0]["prompt"] == "nude woman"
    assert index[0]["nsfw"] is True and index[0]["rating"] == "R"
    assert index[1]["path"] == "01-01-2026/01-123.png"


def test_gallery_index_add_is_a_noop_while_cold(monkeypatch, tmp_path,
                                                _isolated_ratings):
    # Cold index: the next read builds it from disk, so don't half-populate it.
    import utils
    out, target = _setup_outputs(tmp_path, monkeypatch)
    monkeypatch.setattr(utils, "OUTPUTS_DIR", out)
    server._invalidate_gallery_index()
    server._gallery_index_add(target, "a cat\nSteps: 20")
    assert server._GALLERY_INDEX is None


def test_gallery_index_patch_ratings_updates_in_place(monkeypatch, tmp_path,
                                                      _isolated_ratings):
    import utils
    out, target = _setup_outputs(tmp_path, monkeypatch)
    monkeypatch.setattr(utils, "OUTPUTS_DIR", out)
    _save_with_params(target, "a cat\nSteps: 20, Sampler: euler, Seed: 1")
    server._invalidate_gallery_index()
    utils.invalidate_outputs_cache()
    assert server._gallery_index()[0]["nsfw"] is False

    server._gallery_index_patch_ratings(
        {"01-01-2026/01-123.png": {"rating": "X", "nsfw": True},
         "01-01-2026/not-indexed.png": {"rating": "X", "nsfw": True}})
    entry = server._gallery_index()[0]
    assert entry["rating"] == "X" and entry["nsfw"] is True
    assert entry["prompt"] == "a cat"      # nothing else touched
    assert len(server._gallery_index()) == 1   # unknown paths are skipped


def test_auto_tag_coalesces_into_one_job(_isolated_queue, monkeypatch, tmp_path,
                                         _isolated_ratings):
    # Saves accumulate and flush as one job, not one queue row per image.
    out, target = _setup_outputs(tmp_path, monkeypatch)
    monkeypatch.setattr(server.tagger_mod, "timm_available", lambda: True)
    monkeypatch.setitem(server.SETTINGS, "nsfw_blur", True)
    monkeypatch.setattr(server, "_PENDING_TAG", [])
    second = out / "01-01-2026" / "01-124.png"
    Image.new("RGB", (8, 8)).save(second)

    server._maybe_auto_tag(target)
    server._maybe_auto_tag(second)
    with server.QUEUE_LOCK:
        assert len(server.QUEUE) == 0      # nothing queued yet

    server._flush_pending_tags()
    with server.QUEUE_LOCK:
        assert len(server.QUEUE) == 1
        job = server.QUEUE[0]
    assert job.kind == "tag" and job.total == 2
    assert job.label == "Rate NSFW (2 images)"
    assert server._PENDING_TAG == []
    with server.QUEUE_LOCK:
        server.QUEUE.clear()
    server._flush_pending_tags()
    with server.QUEUE_LOCK:
        assert len(server.QUEUE) == 0


def test_pending_tags_wait_for_the_rest_of_the_queue(_isolated_queue, monkeypatch,
                                                     tmp_path, _isolated_ratings):
    # Mid-batch, flushing would still make one tag job per image.
    out, target = _setup_outputs(tmp_path, monkeypatch)
    monkeypatch.setattr(server.tagger_mod, "timm_available", lambda: True)
    monkeypatch.setitem(server.SETTINGS, "nsfw_blur", True)
    monkeypatch.setattr(server, "_PENDING_TAG", [])
    server._maybe_auto_tag(target)

    server._enqueue(server.Job("generate", "gen 2", lambda j: {}))
    server._flush_pending_tags()
    with server.QUEUE_LOCK:
        assert [j.kind for j in server.QUEUE] == ["generate"]
    assert server._PENDING_TAG == [target]   # still held

    with server.QUEUE_LOCK:
        server.QUEUE.clear()
    server._flush_pending_tags()
    with server.QUEUE_LOCK:
        assert [j.kind for j in server.QUEUE] == ["tag"]


def test_gallery_scan_requires_blur_and_timm(_isolated_queue, monkeypatch,
                                             tmp_path, _isolated_ratings):
    from fastapi import HTTPException
    _setup_outputs(tmp_path, monkeypatch)
    monkeypatch.setattr(server.tagger_mod, "timm_available", lambda: True)
    monkeypatch.setitem(server.SETTINGS, "nsfw_blur", False)
    with pytest.raises(HTTPException):
        server.api_gallery_scan()
    monkeypatch.setitem(server.SETTINGS, "nsfw_blur", True)
    monkeypatch.setattr(server.tagger_mod, "timm_available", lambda: False)
    with pytest.raises(HTTPException):
        server.api_gallery_scan()


def test_gallery_scan_enqueues_job_and_skips_rated(_isolated_queue, monkeypatch,
                                                   tmp_path, _isolated_ratings):
    import utils
    out, target = _setup_outputs(tmp_path, monkeypatch)
    monkeypatch.setattr(utils, "OUTPUTS_DIR", out)
    monkeypatch.setattr(server.tagger_mod, "timm_available", lambda: True)
    monkeypatch.setitem(server.SETTINGS, "nsfw_blur", True)
    r = server.api_gallery_scan()
    assert r["total"] == 1 and r["job"] is not None
    with server.QUEUE_LOCK:
        # Lowest priority, like the auto/rescan jobs.
        assert server.QUEUE[0].priority == -10
    server._store_ratings({"01-01-2026/01-123.png": {
        "rating": "PG", "nsfw": False, "conf": 0.9,
        "key": server._image_key(target)}})
    with server.QUEUE_LOCK:
        server.QUEUE.clear()
    r2 = server.api_gallery_scan()
    assert r2["job"] is None and r2["total"] == 0


def test_tagger_status_reports_counts(monkeypatch, tmp_path, _isolated_ratings):
    import utils
    out, target = _setup_outputs(tmp_path, monkeypatch)
    monkeypatch.setattr(utils, "OUTPUTS_DIR", out)
    monkeypatch.setattr(server.tagger_mod, "timm_available", lambda: True)
    st = server.api_tagger_status()
    assert st["available"] is True
    assert st["total"] == 1 and st["rated"] == 0
    server._store_ratings({"01-01-2026/01-123.png": {
        "rating": "PG", "nsfw": False, "conf": 0.9,
        "key": server._image_key(target)}})
    assert server.api_tagger_status()["rated"] == 1
    # Stale rows don't count: a scan would re-do them.
    monkeypatch.setattr(server.tagger_mod, "DECISION_VERSION",
                        server.tagger_mod.DECISION_VERSION + 1)
    assert server.api_tagger_status()["rated"] == 0
    # Nor do rows for deleted images (would read as "2400 / 2320 rated").
    monkeypatch.setattr(server.tagger_mod, "DECISION_VERSION",
                        server.tagger_mod.DECISION_VERSION - 1)
    server._store_ratings({"01-01-2026/gone.png": {
        "rating": "X", "nsfw": True, "conf": 0.9, "key": "whatever"}})
    st = server.api_tagger_status()
    assert st["rated"] == 1 and st["total"] == 1


def test_tag_job_cancel_keeps_what_it_already_rated(monkeypatch, tmp_path,
                                                    _isolated_ratings):
    # Cancelling must keep the verdicts already computed.
    import utils
    out, target = _setup_outputs(tmp_path, monkeypatch)
    monkeypatch.setattr(utils, "OUTPUTS_DIR", out)
    second = out / "01-01-2026" / "01-124.png"
    Image.new("RGB", (8, 8)).save(second)

    job = server.Job("tag", "rate", lambda j: {})
    job.total = 2

    def _rate(path):
        job.cancel.set()   # cancel arrives right after the first image
        return {"rating": "R", "nsfw": True, "confidence": 0.9}

    monkeypatch.setattr(server.tagger_mod, "TAGGER", _fake_tagger(_rate))
    monkeypatch.setattr(server, "_push", lambda ev: None)
    monkeypatch.setattr(server, "_invalidate_gallery_index", lambda: None)
    with pytest.raises(server._Cancelled):
        server._tag_job_run(job, [target, second])
    assert server._read_ratings()["01-01-2026/01-123.png"]["rating"] == "R"


def test_purge_trash_removes_aged_entries(monkeypatch, tmp_path):
    _setup_outputs(tmp_path, monkeypatch)
    trash = tmp_path / "outputs" / ".trash"
    trash.mkdir(exist_ok=True)
    old = trash / "100_old.png"
    old.write_bytes(b"x")
    new = trash / "9999999999_new.png"
    new.write_bytes(b"x")
    old_time = time.time() - (server.TRASH_RETENTION_DAYS + 1) * 86400
    os.utime(old, (old_time, old_time))
    purged = server._purge_trash()
    assert purged == 1
    assert not old.exists()
    assert new.exists()


def test_scan_outputs_skips_dot_trash(tmp_path, monkeypatch):
    import utils
    out = tmp_path / "outputs"
    out.mkdir()
    day = out / "01-01-2026"
    day.mkdir()
    (day / "01-1.png").write_bytes(b"x")
    trash = out / ".trash"
    trash.mkdir()
    (trash / "trashed.png").write_bytes(b"x")
    monkeypatch.setattr(utils, "OUTPUTS_DIR", out)
    monkeypatch.setattr(server, "OUTPUTS_DIR", out)
    files = utils.scan_outputs()
    names = [f.name for f in files]
    assert "01-1.png" in names
    assert "trashed.png" not in names


# ── scan_outputs in-memory cache ─────────────────────────────────────

def test_scan_outputs_cache_avoids_rewalk_and_invalidates(tmp_path, monkeypatch):
    import utils
    out = tmp_path / "outputs"
    out.mkdir()
    day = out / "01-01-2026"
    day.mkdir()
    (day / "01-1.png").write_bytes(b"x")
    monkeypatch.setattr(utils, "OUTPUTS_DIR", out)
    utils.invalidate_outputs_cache()  # clean slate: a prior test may have cached

    calls = {"n": 0}
    real = utils._scan_outputs_uncached

    def counting():
        calls["n"] += 1
        return real()

    monkeypatch.setattr(utils, "_scan_outputs_uncached", counting)

    first = utils.scan_outputs()
    assert calls["n"] == 1                       # cold start: walked once
    assert [f.name for f in first] == ["01-1.png"]

    second = utils.scan_outputs()
    assert calls["n"] == 1                       # cache hit: no re-walk
    assert [f.name for f in second] == ["01-1.png"]

    # A new file appears only after the invalidation hook fires.
    (day / "02-2.png").write_bytes(b"x")
    utils.invalidate_outputs_cache()
    third = utils.scan_outputs()
    assert calls["n"] == 2                       # rebuilt after invalidation
    names = [f.name for f in third]
    assert "01-1.png" in names and "02-2.png" in names


def test_scan_outputs_cache_misses_when_outputs_dir_repointed(tmp_path, monkeypatch):
    """A repointed OUTPUTS_DIR must always miss (the key includes the path)."""
    import utils
    utils.invalidate_outputs_cache()

    out_a = tmp_path / "a" / "outputs"
    out_a.mkdir(parents=True)
    (out_a / "01-01-2026").mkdir()
    (out_a / "01-01-2026" / "01-a.png").write_bytes(b"x")
    monkeypatch.setattr(utils, "OUTPUTS_DIR", out_a)
    assert [f.name for f in utils.scan_outputs()] == ["01-a.png"]

    out_b = tmp_path / "b" / "outputs"
    out_b.mkdir(parents=True)
    (out_b / "02-02-2026").mkdir()
    (out_b / "02-02-2026" / "02-b.png").write_bytes(b"x")
    monkeypatch.setattr(utils, "OUTPUTS_DIR", out_b)
    assert [f.name for f in utils.scan_outputs()] == ["02-b.png"]


# ── /api/thumb cache keyed by source mtime+size ──────────────────────

def test_thumb_cache_busts_on_source_overwrite(monkeypatch, tmp_path):
    _setup_outputs(tmp_path, monkeypatch)  # outputs/01-01-2026/01-123.png (8x8)
    target = server.OUTPUTS_DIR / "01-01-2026" / "01-123.png"
    assert target.is_file()

    server.api_thumb(path="01-01-2026/01-123.png")
    cache1 = server._thumb_cache_path(target)
    assert cache1.is_file()                       # thumbnail generated on first hit

    Image.new("RGB", (16, 16), (200, 100, 50)).save(target)
    now = time.time()
    os.utime(target, (now + 5, now + 5))           # guarantee an mtime advance

    cache2 = server._thumb_cache_path(target)
    assert cache2 != cache1                        # mtime+size key changed → fresh cache file
    server.api_thumb(path="01-01-2026/01-123.png")
    assert cache2.is_file()                        # new thumbnail generated
    assert not cache1.exists()                     # stale version purged


def test_thumb_cache_hit_serves_existing_without_regen(monkeypatch, tmp_path):
    _setup_outputs(tmp_path, monkeypatch)
    target = server.OUTPUTS_DIR / "01-01-2026" / "01-123.png"
    server.api_thumb(path="01-01-2026/01-123.png")
    cache = server._thumb_cache_path(target)
    assert cache.is_file()
    mtime_before = cache.stat().st_mtime_ns
    server.api_thumb(path="01-01-2026/01-123.png")
    assert cache.stat().st_mtime_ns == mtime_before


def test_gallery_delete_purges_thumb_cache(monkeypatch, tmp_path):
    _setup_outputs(tmp_path, monkeypatch)
    monkeypatch.setattr(server, "_invalidate_gallery_index", lambda: None)
    monkeypatch.setattr(server, "_purge_trash", lambda *a, **k: 0)
    target = server.OUTPUTS_DIR / "01-01-2026" / "01-123.png"
    server.api_thumb(path="01-01-2026/01-123.png")
    cache = server._thumb_cache_path(target)
    assert cache.is_file()
    server.api_gallery_delete(path="01-01-2026/01-123.png")
    assert not cache.exists()                      # thumbnail dropped with the image


# ── shutdown partial-preview save ────────────────────────────────────

def test_save_partial_preview_writes_file(monkeypatch, tmp_path):
    job = server.Job("generate", "t", lambda j: {})
    job.last_preview = Image.new("RGB", (64, 64), (40, 50, 60))
    out = tmp_path / "partial.png"
    monkeypatch.setattr(server, "next_output_path", lambda seed: out)
    monkeypatch.setattr(server, "_invalidate_gallery_index", lambda: None)
    path = server._save_partial_preview(job)
    assert path == out and out.is_file()
    with Image.open(out) as im:
        assert im.size == (64, 64)
    assert "PARTIAL" in server.md.read_png_metadata(str(out))


def test_save_partial_preview_noop_without_preview(monkeypatch, tmp_path):
    job = server.Job("generate", "t", lambda j: {})
    assert server._save_partial_preview(job) is None
    assert server._save_partial_preview(None) is None


# ── alpha-aware image + mask decode ──────────────────────────────────

def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def test_decode_image_composites_rgba_onto_white_not_black():
    # convert("RGB") alone would leave the transparent center black.
    rgba = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    import base64
    data = "data:image/png;base64," + base64.b64encode(_png_bytes(rgba)).decode()
    rgb = server._decode_image(data)
    assert rgb.mode == "RGB"
    px = rgb.getpixel((0, 0))
    assert px == (255, 255, 255)


def test_decode_mask_uses_alpha_as_mask():
    # Alpha is the mask: opaque (paint) on the left, transparent on the right.
    rgba = Image.new("RGBA", (4, 2), (0, 0, 0, 0))
    for x in range(2):
        rgba.putpixel((x, 0), (0, 0, 0, 255))   # opaque → paint
    for x in range(2, 4):
        rgba.putpixel((x, 0), (0, 0, 0, 0))     # transparent → don't paint
    import base64
    data = "data:image/png;base64," + base64.b64encode(_png_bytes(rgba)).decode()
    mask = server._decode_mask(data)
    assert mask.mode == "L"
    assert mask.getpixel((0, 0)) == 255
    assert mask.getpixel((3, 0)) == 0


def test_decode_mask_luminance_fallback_for_no_alpha():
    import base64
    rgb = Image.new("RGB", (2, 2), (200, 100, 50))
    data = "data:image/png;base64," + base64.b64encode(_png_bytes(rgb)).decode()
    mask = server._decode_mask(data)
    assert mask.mode == "L"
    # Luminance of (200,100,50) ≈ 132, neither 255 (alpha path) nor 0.
    assert 120 < mask.getpixel((0, 0)) < 145


# ── priority enqueue (load jobs jump the queue) ──────────────────────

class _NoOpWake:
    """QUEUE_WAKE stand-in that never wakes the lingering daemon worker, so it
    can't pop the job before the test inspects the queue."""
    def set(self): pass
    def clear(self): pass
    def wait(self, timeout=None): return True


@pytest.fixture
def _isolated_queue(monkeypatch):
    monkeypatch.setattr(server, "QUEUE_WAKE", _NoOpWake())
    monkeypatch.setattr(server, "_broadcast_queue", lambda: None)
    with server.QUEUE_LOCK:
        server.QUEUE.clear()
    yield
    with server.QUEUE_LOCK:
        server.QUEUE.clear()


def test_enqueue_priority_inserts_before_lower_priority(_isolated_queue):
    gen = server.Job("generate", "g1", lambda j: {})
    load = server.Job("load", "L", lambda j: {}, priority=10)
    server._enqueue(gen)
    server._enqueue(load)
    with server.QUEUE_LOCK:
        order = [j.kind for j in server.QUEUE]
    assert order == ["load", "generate"]  # load jumped ahead


def test_enqueue_preserves_fifo_within_equal_priority(_isolated_queue):
    a = server.Job("generate", "a", lambda j: {})
    b = server.Job("generate", "b", lambda j: {})
    server._enqueue(a)
    server._enqueue(b)
    with server.QUEUE_LOCK:
        order = [j.label for j in server.QUEUE]
    assert order == ["a", "b"]


def test_load_job_priority_jumps_queue(_isolated_queue):
    # Built like the /api/load handler builds it (priority=10).
    job = server.Job("load", "load SD/SDXL", lambda j: {}, priority=10)
    assert job.priority == 10
    gen = server.Job("generate", "g", lambda j: {})
    server._enqueue(gen)
    server._enqueue(job)
    with server.QUEUE_LOCK:
        assert server.QUEUE[0].kind == "load"


# ── structured logging (--log-file, run-id, chmod 600) ────────────────

def test_log_setup_writes_run_id_stamped_file(monkeypatch, tmp_path):
    import log_setup
    log_file = tmp_path / "diffucore.log"
    rid = log_setup.configure(log_file=str(log_file), level="INFO")
    assert rid and len(rid) >= 4
    logging.getLogger("diffucore.test").info("hello-from-test")
    text = log_file.read_text()
    assert rid in text               # run-id stamped on every line
    assert "hello-from-test" in text
    if os.name == "posix":
        assert (log_file.stat().st_mode & 0o777) == 0o600


def test_log_setup_run_id_stable_until_reconfigure():
    import log_setup
    log_setup.configure(log_file=None, level="WARNING")
    a = log_setup.run_id()
    assert log_setup.run_id() == a   # stable
    log_setup.configure(log_file=None, level="WARNING")
    assert log_setup.run_id() != a   # new run on reconfigure


def test_log_runtime_env_does_not_crash(caplog):
    with caplog.at_level(logging.INFO, logger="diffucore.server"):
        server._log_runtime_env()
    assert any("runtime:" in r.getMessage() for r in caplog.records)


# ── X/Y/Z Checkpoint LRU cache ────────────────────────────────────────

class _FakePart:
    def __init__(self):
        self.placements = []
    def to(self, target):
        self.placements.append(str(target))
        return self


class _FakeModel:
    def __init__(self, name):
        self.name = name
        self.backbone = _FakePart()


def _fake_loaded(name, family="sdxl"):
    from engine import LoadedModel
    return LoadedModel(name=name, family=family, model=_FakeModel(name),
                       native_res=1024)


def _cache_engine():
    from engine import Engine
    eng = Engine(device="cpu")
    eng._offload = False  # the UI's "none"; cacheable
    return eng


def test_ckpt_cache_stash_and_restore():
    eng = _cache_engine()
    eng._loaded = _fake_loaded("A")
    part = eng._loaded.model.backbone
    assert eng._stash_loaded() is True
    assert eng._loaded is None
    assert "A" in eng._ckpt_cache
    assert part.placements == ["cpu"]
    restored = eng._try_cache_restore("A", False, True, False, False, False, False, False)
    assert restored is not None and restored.name == "A"
    assert "A" not in eng._ckpt_cache  # popped on restore
    assert part.placements == ["cpu", str(eng.device)]


def test_ckpt_cache_engages_through_load_model(monkeypatch):
    """The server hands the UI's "none" to the engine as offload=False."""
    import engine as engine_mod
    eng = _cache_engine()
    eng._loaded = _fake_loaded("A")
    eng._offload = True
    assert eng._cacheable_for_stash() is False
    eng._offload = False
    monkeypatch.setattr(engine_mod, "checkpoint_path",
                        lambda n: Path("/nonexistent") / n)
    with pytest.raises(FileNotFoundError):
        eng.load_model("B", offload=False)
    assert "A" in eng._ckpt_cache
    msg = eng.load_model("A", offload=False)
    assert "(from cache)" in msg and eng.loaded_name == "A"


def test_ckpt_cache_restore_stashes_current_first():
    """Restoring A while B is loaded must not evict A to make room for B, and
    B leaves the device before A moves back."""
    eng = _cache_engine()
    for nm in ("A", "C"):
        eng._loaded = _fake_loaded(nm)
        eng._stash_loaded()
    eng._loaded = _fake_loaded("B")
    b_part = eng._loaded.model.backbone
    restored = eng._try_cache_restore("A", False, True, False, False, False, False, False)
    assert restored is not None and restored.name == "A"
    assert list(eng._ckpt_cache) == ["C", "B"]
    assert b_part.placements == ["cpu"]


def test_ckpt_cache_stash_drops_fused_loras(monkeypatch):
    """LoRA snapshots alias the old storage, so a stash unfuses first."""
    import engine as engine_mod
    cleared = []
    monkeypatch.setattr(engine_mod, "clear_bundle_loras", cleared.append)
    eng = _cache_engine()
    eng._loaded = _fake_loaded("A")
    eng._loaded.applied_loras.append("style")
    model = eng._loaded.model
    assert eng._stash_loaded() is True
    assert cleared == [model]
    assert eng._ckpt_cache["A"].applied_loras == []


def test_ckpt_cache_evicts_lru_on_overflow():
    eng = _cache_engine()
    eng.CKPT_CACHE_MAX = 2
    for nm in ("A", "B", "C"):
        eng._loaded = _fake_loaded(nm)
        eng._stash_loaded()
    assert list(eng._ckpt_cache) == ["B", "C"]  # A evicted (LRU)


def test_ckpt_cache_skips_flux_and_offloaded():
    from engine import MODEL_FAMILY_FLUX1
    eng = _cache_engine()
    eng._loaded = _fake_loaded("flux1", family=MODEL_FAMILY_FLUX1)
    assert eng._stash_loaded() is False
    assert eng._loaded is not None  # left for _unload
    eng._loaded = _fake_loaded("D")
    eng._offload = "stream"
    assert eng._stash_loaded() is False


def test_teacache_rejected_on_cuda_graphs_anima():
    """TeaCache + CUDA Graphs on Anima is refused up front."""
    eng = _cache_engine()
    eng._loaded = _fake_loaded("A", family="anima")
    eng._cuda_graphs = True
    with pytest.raises(RuntimeError, match="incompatible with CUDA Graphs"):
        eng.generate_t2i(prompt="x", teacache_thresh=0.4)
    with pytest.raises(RuntimeError, match="incompatible with CUDA Graphs"):
        eng.calibrate_teacache(prompt="x")
    # TeaCache off passes the guard (then fails later on the fake model).
    try:
        eng.generate_t2i(prompt="x", teacache_thresh=0.0)
    except Exception as e:  # noqa: BLE001
        assert "incompatible with CUDA Graphs" not in str(e)


def test_ckpt_cache_restore_rejects_settings_mismatch():
    eng = _cache_engine()
    eng._loaded = _fake_loaded("A")
    eng._stash_loaded()
    # Request a different offload than the cached "none" → miss, entry dropped.
    restored = eng._try_cache_restore("A", "stream", True, False, False, False, False, False)
    assert restored is None
    assert "A" not in eng._ckpt_cache


# ── compare against the entry's settings, not the live ones ───────────

def test_ckpt_cache_restore_uses_entry_settings_not_live_flags():
    """A stashed model must be compared against the settings it was staged
    under, not the live flags, which describe whichever model loaded after."""
    eng = _cache_engine()
    # A is stashed while the engine is on offload="none".
    eng._loaded = _fake_loaded("A")
    assert eng._stash_loaded() is True
    # B then loads with a different policy, leaving the live flags on B.
    eng._offload = "stream"
    # Asking for A under B's policy must miss: A's placement is "none".
    assert eng._try_cache_restore("A", "stream", True, False, False,
                                  False, False, False) is None
    assert "A" not in eng._ckpt_cache


# ── calibration caches must never feed NaN into generation ────────────

def test_cache_json_rejects_nan_and_corruption(tmp_path):
    """json round-trips NaN silently, so a degenerate fit must read as
    "not calibrated"."""
    from engine import _read_cache_json, _finite_series, _write_cache_json

    good = tmp_path / "good.json"
    _write_cache_json(good, [1.0, -2.5, 0.0])
    assert _read_cache_json(good) == [1.0, -2.5, 0.0]

    nan_file = tmp_path / "nan.json"
    nan_file.write_text("[1.0, NaN, 3.0]")
    assert json.loads(nan_file.read_text())[1] != json.loads(nan_file.read_text())[1]
    assert _read_cache_json(nan_file) is None    # NaN → "not calibrated"

    truncated = tmp_path / "partial.json"
    truncated.write_text("[1.0, 2.0")
    assert _read_cache_json(truncated) is None   # no raw JSONDecodeError

    assert _finite_series([]) is False
    assert _finite_series([float("inf")]) is False
    assert _finite_series([1, 2.0]) is True


def test_cache_json_write_is_atomic(tmp_path):
    """An interrupted write must leave the previous file intact, not a stub."""
    from engine import _write_cache_json

    p = tmp_path / "c.json"
    _write_cache_json(p, [1.0, 2.0])

    class _Boom:
        def __iter__(self): raise RuntimeError("interrupted")

    with pytest.raises((RuntimeError, TypeError)):
        _write_cache_json(p, _Boom())
    assert json.loads(p.read_text()) == [1.0, 2.0]
    assert not list(tmp_path.glob("*.tmp*")), "temp file cleaned up"


# ── "oss" is a t2i-only schedule ─────────────────────────────────────

def test_oss_scheduler_degrades_outside_t2i():
    """Picking oss in t2i, then switching to img2img, used to fail the job."""
    eng = _cache_engine()
    eng._loaded = _fake_loaded("A", family="anima")
    assert eng._degrade_oss("oss") == "flow"
    eng._loaded = _fake_loaded("A", family="sdxl")
    assert eng._degrade_oss("oss") == "karras"
    assert eng._degrade_oss("beta") == "beta"


def test_anima_snap_is_never_a_downscale_above_1536():
    """A 2048 px img2img used to be generated at 1536 and upscaled back."""
    eng = _cache_engine()
    eng._loaded = _fake_loaded("A", family="anima")
    assert eng._anima_gen_size(2048, 2000) == (2048, 2048, True)
    assert eng._anima_gen_size(1600, 1600) == (1600, 1600, False)
    assert eng._anima_gen_size(300, 1024) == (512, 1024, True)


def test_png_perf_flags_record_fp16_vae():
    eng = _cache_engine()
    eng._vae_fp16 = True
    assert "fp16_vae" in eng.perf_flags_str


# ── shared HTTP fixture (last so module-level client use above works) ──

@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    with TestClient(server.app) as c:
        yield c

