"""Tests for the Critical + High IMPROVE.md fixes shipped after the first round.

Covers:
- #1 (Critical) extension install: opt-in pip deps, install routed through the
  job queue (#8), URL scheme fast-fail stays a 400.
- #2 share URL written to a chmod 600 file, not stdout.
- #3 gallery delete is a soft-delete to outputs/.trash/ with age-based purge;
  scan_outputs skips dot-dirs.
- #4 shutdown saves a partial preview for the in-flight job.
- #5 _decode_image composites RGBA onto white (not black); _decode_mask uses
  the alpha channel as the mask.
- #6 priority enqueue: load jobs jump the queue.
- #9 structured logging: --log-file writes a run-id-stamped, chmod 600 file.
- #10 X/Y/Z Checkpoint LRU cache: stash / restore / eviction / FLUX-skip.

Run from the project root::

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


# ── #1 + #8: opt-in pip + install routed through the job queue ───────

def test_install_payload_pip_deps_defaults_off():
    from extensions import InstallPayload
    p = InstallPayload(url="https://github.com/foo/bar.git")
    assert p.install_pip_deps is False
    q = InstallPayload(url="https://x.zip", install_pip_deps=True)
    assert q.install_pip_deps is True


def test_install_endpoint_rejects_non_https_with_400(client):
    # Fast-fail scheme validation happens before enqueueing.
    r = client.post("/api/extensions/install",
                    json={"url": "file:///etc/passwd"})
    assert r.status_code == 400


def test_install_endpoint_enqueues_install_job_and_passes_pip_flag(client, monkeypatch):
    # Stub the real install so no network / git / pip runs.
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
        # Don't touch the real queue / wake — the worker must not run this.
    monkeypatch.setattr(server, "_enqueue", fake_enqueue)

    r = client.post("/api/extensions/install",
                    json={"url": "https://github.com/foo/bar.git",
                          "install_pip_deps": True})
    assert r.status_code == 200
    assert "job" in r.json()
    job = enqueued["job"]
    assert job.kind == "install"  # routed through the queue, not the request thread
    # Run the job body directly and confirm the opt-in flag propagates.
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
    # Run the held job body; the default payload must pass install_pip_deps=False.
    held["job"].run(held["job"])
    assert captured["pip"] is False


def test_update_endpoint_enqueues_update_job_and_passes_pip_flag(client, monkeypatch):
    # Update mirrors install: 404 for an unknown name, otherwise routed through
    # the job queue with the opt-in pip flag propagated.
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

    # Unknown extension → 404 before any job is enqueued.
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
    # A zip install (no .git dir) has no remote to update from.
    import extensions as extmod
    loader = extmod.ExtensionLoader.__new__(extmod.ExtensionLoader)
    loader.extensions = {"z": extmod.Extension(name="z", title="z", version="1",
                                               path=tmp_path / "z")}
    (tmp_path / "z").mkdir()
    monkeypatch.setattr(extmod, "EXTENSIONS_DIR", tmp_path)
    with pytest.raises(ValueError, match="git-only"):
        loader.update("z")


def test_install_skips_pip_by_default_and_notes_requirements(monkeypatch, tmp_path):
    # Drive ExtensionLoader.install() with a fake clone so we can observe the
    # pip-skip note without network. We bypass _install_git/_install_zip by
    # pre-creating the scratch dir with a manifest + requirements.txt.
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
    # No pip should run: assert _pip_install_requirements is never called.
    monkeypatch.setattr(extmod.ExtensionLoader, "_pip_install_requirements",
                        staticmethod(lambda *a, **k: pytest.fail("pip must not run by default")))

    ext = loader.install("https://github.com/foo/demo.git")
    assert ext.name == "demo"
    # A requirements.txt was present and pip skipped → note surfaced on the record.
    assert ext.load_error and "requirements.txt" in ext.load_error
    assert "opt-in" in ext.load_error


# ── #2: share URL written to a chmod 600 file, not stdout ─────────────

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
    # The path is named, but the full secret URL is NOT printed.
    assert "share_url.txt" in out
    assert "secret" not in out
    assert "trycloudflare.com?token" not in out


def test_share_warning_falls_back_to_printing_url_on_file_failure(monkeypatch, tmp_path, capsys):
    import share
    monkeypatch.setattr(share, "_BIN_DIR", tmp_path)
    # Force the file write to fail.
    def boom(_url):
        return None
    monkeypatch.setattr(share, "_write_share_url_file", boom)
    share._print_share_warning("https://abc.trycloudflare.com", "?token=tok")
    out = capsys.readouterr().out
    assert "https://abc.trycloudflare.com?token=tok" in out  # fallback prints it
    assert "WARNING" in out


# ── #3: gallery soft-delete + trash purge + scan_outputs skips dot-dirs ─

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
    # Call the handler directly with a relative path under outputs/.
    resp = server.api_gallery_delete(path="01-01-2026/01-123.png")
    assert not target.exists()  # gone from the gallery
    trash = server._TRASH_DIR
    assert trash.is_dir()
    trashed = list(trash.iterdir())
    assert len(trashed) == 1
    assert trashed[0].name.endswith("_01-123.png")
    assert resp["deleted"] == "01-01-2026/01-123.png"


def test_purge_trash_removes_aged_entries(monkeypatch, tmp_path):
    _setup_outputs(tmp_path, monkeypatch)
    trash = tmp_path / "outputs" / ".trash"
    trash.mkdir(exist_ok=True)
    old = trash / "100_old.png"
    old.write_bytes(b"x")
    new = trash / "9999999999_new.png"
    new.write_bytes(b"x")
    # Backdate the old entry below the retention cutoff.
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


# ── #11: scan_outputs in-memory cache ────────────────────────────────

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

    # A new file appears only after the save/delete invalidation hook fires.
    (day / "02-2.png").write_bytes(b"x")
    utils.invalidate_outputs_cache()
    third = utils.scan_outputs()
    assert calls["n"] == 2                       # rebuilt after invalidation
    names = [f.name for f in third]
    assert "01-1.png" in names and "02-2.png" in names


def test_scan_outputs_cache_misses_when_outputs_dir_repointed(tmp_path, monkeypatch):
    """A repointed OUTPUTS_DIR (tests, a future DIFFUCORE_DATA_DIR) must always
    miss — the cache key includes the dir path so a stale list from another dir
    is never served."""
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
    # No explicit invalidate — the dir change alone must force a fresh walk.
    assert [f.name for f in utils.scan_outputs()] == ["02-b.png"]


# ── #12: /api/thumb cache keyed by source mtime+size ─────────────────

def test_thumb_cache_busts_on_source_overwrite(monkeypatch, tmp_path):
    _setup_outputs(tmp_path, monkeypatch)  # outputs/01-01-2026/01-123.png (8x8)
    target = server.OUTPUTS_DIR / "01-01-2026" / "01-123.png"
    assert target.is_file()

    # First request generates and caches the thumbnail.
    server.api_thumb(path="01-01-2026/01-123.png")
    cache1 = server._thumb_cache_path(target)
    assert cache1.is_file()                       # thumbnail generated on first hit

    # Overwrite the source with different content + a clearly newer mtime.
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
    # A second hit must not regenerate the webp (the cache file is untouched).
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


# ── #4: shutdown partial-preview save ─────────────────────────────────

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
    # The PNG parameters flag it as a partial.
    assert "PARTIAL" in server.md.read_png_metadata(str(out))


def test_save_partial_preview_noop_without_preview(monkeypatch, tmp_path):
    job = server.Job("generate", "t", lambda j: {})
    assert server._save_partial_preview(job) is None
    assert server._save_partial_preview(None) is None


# ── #5: alpha-preserving image + mask decode ──────────────────────────

def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def test_decode_image_composites_rgba_onto_white_not_black():
    # Transparent center on an RGBA image: dropping alpha via convert("RGB")
    # would leave black; compositing onto white leaves white.
    rgba = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    import base64
    data = "data:image/png;base64," + base64.b64encode(_png_bytes(rgba)).decode()
    rgb = server._decode_image(data)
    assert rgb.mode == "RGB"
    # The previously-transparent pixel is now white, not black.
    px = rgb.getpixel((0, 0))
    assert px == (255, 255, 255)


def test_decode_mask_uses_alpha_as_mask():
    # An RGBA image where alpha is the mask: opaque on the left, transparent
    # on the right. The mask should be 255 (opaque) on the left and 0 on the right.
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
    # Luminance of (200,100,50) ≈ 132 — not 255 (alpha path) nor 0.
    assert 120 < mask.getpixel((0, 0)) < 145


# ── #6: priority enqueue (load jobs jump the queue) ───────────────────

class _NoOpWake:
    """A stand-in for QUEUE_WAKE that never wakes the (possibly lingering)
    daemon worker thread, so a priority-insert test isn't raced by the worker
    popping the job before we inspect — and doesn't broadcast on a closed loop
    left behind by a prior TestClient session."""
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
    # A load-shaped job built the same way the /api/load handler builds it
    # (priority=10) jumps ahead of a queued generate.
    job = server.Job("load", "load SD/SDXL", lambda j: {}, priority=10)
    assert job.priority == 10
    gen = server.Job("generate", "g", lambda j: {})
    server._enqueue(gen)
    server._enqueue(job)
    with server.QUEUE_LOCK:
        assert server.QUEUE[0].kind == "load"


# ── #9: structured logging (--log-file, run-id, chmod 600) ────────────

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


# ── #10: X/Y/Z Checkpoint LRU cache ───────────────────────────────────

class _FakeModel:
    def __init__(self, name):
        self.name = name
        self.placements = []
    def to(self, target):
        self.placements.append(str(target))
        return self


def _fake_loaded(name, family="sdxl"):
    from engine import LoadedModel
    return LoadedModel(name=name, family=family, model=_FakeModel(name),
                       native_res=1024)


def _cache_engine():
    from engine import Engine
    eng = Engine(device="cpu")
    eng._offload = "none"  # cacheable
    return eng


def test_ckpt_cache_stash_and_restore():
    eng = _cache_engine()
    eng._loaded = _fake_loaded("A")
    assert eng._stash_loaded() is True
    assert eng._loaded is None
    assert "A" in eng._ckpt_cache
    restored = eng._try_cache_restore("A", "none", True, False, False, False, False, False)
    assert restored is not None and restored.name == "A"
    assert "A" not in eng._ckpt_cache  # popped on restore


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
    """TeaCache keeps tensors from inside the compiled forward alive across
    steps; CUDA Graphs replays overwrite them — the engine must refuse the
    combination up front instead of crashing mid-generation."""
    eng = _cache_engine()
    eng._loaded = _fake_loaded("A", family="anima")
    eng._cuda_graphs = True
    with pytest.raises(RuntimeError, match="incompatible with CUDA Graphs"):
        eng.generate_t2i(prompt="x", teacache_thresh=0.4)
    with pytest.raises(RuntimeError, match="incompatible with CUDA Graphs"):
        eng.calibrate_teacache(prompt="x")
    # TeaCache off passes the guard — it then fails later on the fake model,
    # which is fine; just assert the guard isn't what trips.
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


# ── shared HTTP fixture (last so module-level client use above works) ──

@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    with TestClient(server.app) as c:
        yield c
