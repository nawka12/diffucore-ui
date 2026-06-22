"""Tests for the urgent hardening fixes from IMPROVE.md.

Covers: auth gate + CSRF/Origin (#1/#2), body-size cap (#5), pydantic param
bounds (#54), CUDA OOM hint (#57), atomic state writes (#58), SSE queue cap
(#17), gallery index lock/invalidation (#11), load fail-fast validation (#8),
and the extension-install URL scheme allowlist / SSRF guard (#4).

Run from the project root::

    .venv/bin/python -m pytest backend/test_server_security.py -v

Some cases drive the real ``server.app`` through TestClient, which starts the
worker thread and loads extensions (the normal startup path). The pure-function
cases avoid that cost.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import auth as authmod
import server


# ── #54: pydantic Field bounds on generation params ─────────────────

def test_generate_payload_rejects_runaway_steps():
    with pytest.raises(ValidationError):
        server.GeneratePayload(steps=100000)
    with pytest.raises(ValidationError):
        server.GeneratePayload(width=99999)
    with pytest.raises(ValidationError):
        server.GeneratePayload(cfg=-1.0)
    with pytest.raises(ValidationError):
        server.GeneratePayload(upscale_tile=0)
    # In-range values still parse.
    assert server.GeneratePayload(steps=200, width=8192, cfg=0.0).steps == 200


def test_xyz_and_upscale_payloads_bounded():
    with pytest.raises(ValidationError):
        server.XYZPayload(steps=201)
    with pytest.raises(ValidationError):
        server.UpscalePayload(scale=16.0)
    with pytest.raises(ValidationError):
        server.CalibratePayload(grid=0)


# ── #2: CSRF / Origin allowlist ─────────────────────────────────────

class _FakeReq:
    def __init__(self, method, host, origin=None, referer=None, token=None, bearer=None):
        self.method = method
        self.headers = {"host": host}
        self.cookies = {}
        self.query_params = {"token": token} if token else {}
        if origin is not None:
            self.headers["origin"] = origin
        if referer is not None:
            self.headers["referer"] = referer
        if bearer:
            self.headers["authorization"] = f"Bearer {bearer}"


def test_origin_ok_same_origin_post_passes():
    req = _FakeReq("POST", "192.168.1.5:7860", origin="http://192.168.1.5:7860")
    assert authmod.origin_ok(req) is True


def test_origin_ok_cross_origin_post_blocked():
    req = _FakeReq("POST", "192.168.1.5:7860", origin="https://evil.example")
    assert authmod.origin_ok(req) is False


def test_origin_ok_no_origin_passes_curl():
    # Non-browser clients (curl) send no Origin — not a CSRF vector.
    req = _FakeReq("POST", "192.168.1.5:7860")
    assert authmod.origin_ok(req) is True


def test_origin_ok_get_always_passes():
    req = _FakeReq("GET", "x", origin="https://evil.example")
    assert authmod.origin_ok(req) is True


def test_origin_ok_referer_fallback():
    req = _FakeReq("POST", "app.local", referer="http://app.local/api/x")
    assert authmod.origin_ok(req) is True
    req = _FakeReq("POST", "app.local", referer="http://evil.example/x")
    assert authmod.origin_ok(req) is False


# ── #1: AuthGate token / cookie ─────────────────────────────────────

TOKEN = "test-token-abc123"


def test_auth_gate_disabled_allows_everything():
    gate = authmod.AuthGate(token=TOKEN, enabled=False)
    assert gate.has_access(_FakeReq("GET", "x")) is True
    assert gate.has_access(_FakeReq("POST", "x")) is True


def test_auth_gate_enabled_rejects_without_credential():
    gate = authmod.AuthGate(token=TOKEN, enabled=True)
    assert gate.has_access(_FakeReq("POST", "x")) is False


def test_auth_gate_accepts_bearer_and_query_token():
    gate = authmod.AuthGate(token=TOKEN, enabled=True)
    assert gate.has_access(_FakeReq("POST", "x", bearer=TOKEN)) is True
    assert gate.has_access(_FakeReq("GET", "x", token=TOKEN)) is True
    assert gate.has_access(_FakeReq("POST", "x", bearer="wrong")) is False


def test_auth_gate_accept_sets_cookie_and_redirects():
    gate = authmod.AuthGate(token=TOKEN, enabled=True)
    resp = gate.accept(TOKEN)
    assert resp is not None
    assert resp.status_code == 303
    set_cookie = resp.headers.get("set-cookie", "")
    assert authmod.COOKIE_NAME in set_cookie
    assert "HttpOnly" in set_cookie
    assert "samesite=lax" in set_cookie.lower()
    assert gate.accept("wrong") is None


def test_load_or_create_token_persists_and_chmods(tmp_path: Path):
    path = tmp_path / ".auth_token"
    t1 = authmod.load_or_create_token(path)
    assert t1 and path.is_file()
    # Second call returns the same persisted token (stable across restarts).
    assert authmod.load_or_create_token(path) == t1
    mode = path.stat().st_mode & 0o777
    # On POSIX the file should be owner-only (0600). Windows ignores chmod.
    if os.name == "posix":
        assert mode == 0o600


# ── #4 + #53: extension install URL scheme allowlist / SSRF ─────────

def test_validate_install_url_https_passes():
    authmod  # noqa — ensure module import side effects are loaded
    from extensions import _validate_install_url
    _validate_install_url("https://github.com/foo/bar.git")
    _validate_install_url("https://example.com/ext.zip")


@pytest.mark.parametrize("url", [
    "file:///etc/passwd",
    "http://localhost/ext.zip",
    "http://169.254.169.254/latest/meta-data/",
    "ftp://example.com/x.zip",
    "gopher://example.com/",
    "git@github.com:foo/bar.git",
    "ssh://git@github.com/foo/bar.git",
    "git://github.com/foo/bar.git",
])
def test_validate_install_url_blocks_ssrf(url):
    from extensions import _validate_install_url
    with pytest.raises(ValueError):
        _validate_install_url(url)


# ── #8: load fail-fast validation ───────────────────────────────────

def test_validate_load_missing_checkpoint_returns_error(tmp_path: Path, monkeypatch):
    # Point the model dirs at an empty temp dir so nothing resolves.
    monkeypatch.setattr(server, "CHECKPOINTS_DIR", tmp_path / "ckpts")
    monkeypatch.setattr(server, "DIFFUSION_DIR", tmp_path / "dit")
    monkeypatch.setattr(server, "VAE_DIR", tmp_path / "vae")
    monkeypatch.setattr(server, "TE_DIR", tmp_path / "te")
    for d in ("ckpts", "dit", "vae", "te"):
        (tmp_path / d).mkdir()
    err = server._validate_load(server.LoadPayload(model_type="SD/SDXL",
                                                   checkpoint="ghost.safetensors"))
    assert err and "not found" in err.lower()
    err = server._validate_load(server.LoadPayload(model_type="FLUX",
                                                   dit="g.safetensors", vae="v.safetensors",
                                                   te="t.safetensors"))
    assert err and "not found" in err.lower()
    assert server._validate_load(
        server.LoadPayload(model_type="SD/SDXL", checkpoint="(choose)")) is not None


def test_validate_load_existing_file_passes(tmp_path: Path, monkeypatch):
    ck = tmp_path / "ckpts"
    ck.mkdir()
    (ck / "real.safetensors").write_bytes(b"x")
    monkeypatch.setattr(server, "CHECKPOINTS_DIR", ck)
    assert server._validate_load(
        server.LoadPayload(model_type="SD/SDXL", checkpoint="real.safetensors")) is None


# ── #58: atomic state writes ────────────────────────────────────────

def test_atomic_write_text_replaces_cleanly(tmp_path: Path):
    path = tmp_path / "state.json"
    path.write_text('{"old": true}', encoding="utf-8")
    server._atomic_write_text(path, json.dumps({"new": True}))
    assert json.loads(path.read_text()) == {"new": True}
    # The temp file is gone (no litter left in the dir).
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".state.json-")]
    assert leftovers == []


def test_atomic_write_does_not_truncate_on_oserror(tmp_path: Path, monkeypatch):
    path = tmp_path / "state.json"
    path.write_text('{"keep": true}', encoding="utf-8")
    # Force os.replace to fail so we can confirm the original survives.
    import server as srv
    monkeypatch.setattr(srv.os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))
    srv._atomic_write_text(path, json.dumps({"new": True}))
    # Original content is intact — no truncation, no data loss.
    assert json.loads(path.read_text()) == {"keep": True}


def test_extensions_state_write_is_atomic(tmp_path: Path, monkeypatch):
    import extensions
    state_path = tmp_path / "state.json"
    monkeypatch.setattr(extensions, "STATE_PATH", state_path)
    loader = extensions.ExtensionLoader.__new__(extensions.ExtensionLoader)
    loader._state = {"enabled": {"x": True}, "ext_settings": {}}
    loader._write_state()
    assert json.loads(state_path.read_text())["enabled"] == {"x": True}


# ── #17: SSE queue cap + drop-oldest ────────────────────────────────

def test_force_put_drops_oldest_on_overflow():
    q: asyncio.Queue = asyncio.Queue(maxsize=2)
    server._force_put(q, "a")
    server._force_put(q, "b")
    server._force_put(q, "c")  # overflows → drops "a"
    assert q.qsize() == 2
    assert q.get_nowait() == "b"
    assert q.get_nowait() == "c"


def test_force_put_shutdown_sentinel_lands_on_full_queue():
    q: asyncio.Queue = asyncio.Queue(maxsize=1)
    server._force_put(q, {"type": "progress"})
    server._force_put(q, None)  # must land even though full
    assert q.get_nowait() is None


# ── #57: CUDA OOM friendly error ────────────────────────────────────

def test_friendly_error_maps_cuda_oom():
    try:
        import torch
        oom = torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
    except Exception:  # noqa: BLE001 — torch missing in a minimal CI env
        pytest.skip("torch not available")
    msg = server._friendly_error(oom)
    assert "out of VRAM" in msg
    assert "offload=stream" in msg


def test_friendly_error_passthrough_for_other_errors():
    msg = server._friendly_error(RuntimeError("boom"))
    assert msg == "boom"


# ── #11: gallery index invalidation ─────────────────────────────────

def test_invalidate_gallery_index_clears_cache():
    server._GALLERY_INDEX = ["stale"]
    server._GALLERY_INDEX_KEY = 12345.0
    server._invalidate_gallery_index()
    assert server._GALLERY_INDEX is None
    assert server._GALLERY_INDEX_KEY == 0.0


# ── #55: X/Y/Z grid cell-count cap ──────────────────────────────────

def test_xyz_grid_rejects_over_cap():
    import xyz_grid
    # One axis with MAX_XYZ_CELLS+1 seeds, the other two None → over the cap.
    seeds = ",".join(str(i) for i in range(xyz_grid.MAX_XYZ_CELLS + 1))
    with pytest.raises(ValueError, match="cap"):
        xyz_grid.generate_xyz_grid(
            {"seed": 0, "steps": 1}, "Seed", seeds, "None", "", "None", "",
        )


# ── #51: extensions mounted at runtime (idempotent re-mount) ─────────

def test_mount_into_is_idempotent_and_picks_up_new_routes():
    from fastapi import APIRouter, FastAPI
    import extensions as extmod

    loader = extmod.ExtensionLoader.__new__(extmod.ExtensionLoader)
    loader._routers = []
    loader._statics = []
    loader._mounted = set()
    loader.extensions = {}

    app = FastAPI()
    r1 = APIRouter()
    r1.add_api_route("/ping", lambda: {"ok": True}, methods=["GET"])
    loader._routers.append(("a", r1, "/api/ext/a"))

    loader.mount_into(app)
    after_first = len(app.router.routes)
    assert ("router", "/api/ext/a") in loader._mounted
    assert any(getattr(rt, "path", "") == "/api/ext/a/ping" for rt in app.router.routes)

    # Re-calling must not double-mount the same router.
    loader.mount_into(app)
    assert len(app.router.routes) == after_first

    # A freshly-installed extension's router is attached on the next call.
    r2 = APIRouter()
    r2.add_api_route("/ping", lambda: {"ok": True}, methods=["GET"])
    loader._routers.append(("b", r2, "/api/ext/b"))
    loader.mount_into(app)
    assert ("router", "/api/ext/b") in loader._mounted
    assert any(getattr(rt, "path", "") == "/api/ext/b/ping" for rt in app.router.routes)
    assert len(app.router.routes) > after_first


# ── #13: LoRA weight validation ─────────────────────────────────────

def test_parse_lora_prompt_rejects_non_numeric_weight():
    import engine
    with pytest.raises(ValueError, match="not a number"):
        engine.Engine.parse_lora_prompt("a cat <lora:mychar:high> sitting")
    # A comma-decimal typo is caught too (would otherwise crash in apply_lora).
    with pytest.raises(ValueError, match="not a number"):
        engine.Engine.parse_lora_prompt("<lora:x:0,8>")


def test_parse_lora_prompt_accepts_valid_weight():
    import engine
    cleaned, loras = engine.Engine.parse_lora_prompt("a cat <lora:mychar:0.8>")
    assert loras == [("mychar", 0.8)]
    assert "<lora" not in cleaned


# ── #18: live-preview throttle + resolution/format cap ──────────────

def test_on_preview_throttles_and_caps(monkeypatch):
    from PIL import Image

    pushed = []
    monkeypatch.setattr(server, "_push", lambda ev: pushed.append(ev))
    clock = [1000.0]
    monkeypatch.setattr(server.time, "monotonic", lambda: clock[0])

    job = server.Job("generate", "t", lambda j: {})
    _on_progress, on_preview = server._make_callbacks(job)

    big = Image.new("RGB", (1024, 1024), (120, 60, 30))
    on_preview(big)                                   # first → emits
    clock[0] += server.PREVIEW_MIN_INTERVAL / 2
    on_preview(big)                                   # within interval → dropped
    clock[0] += server.PREVIEW_MIN_INTERVAL
    on_preview(big)                                   # interval elapsed → emits

    assert len(pushed) == 2                           # the mid frame was throttled
    ev = pushed[0]
    assert ev["type"] == "preview" and ev["job"] == job.id
    assert ev["image"].startswith("data:image/webp;base64,")

    raw = base64.b64decode(ev["image"].split(",", 1)[1])
    im = Image.open(io.BytesIO(raw))
    assert im.format == "WEBP"
    assert max(im.size) <= server.PREVIEW_MAX_SIDE     # downscaled to the cap


def test_preview_webp_payload_smaller_than_full_png():
    """The shipped path (≤512 WebP) is far smaller than the old full-res PNG for
    worst-case noisy content — what drove the per-client serialize cost."""
    import os
    from PIL import Image
    noisy = Image.frombytes("RGB", (1024, 1024), os.urandom(1024 * 1024 * 3))

    b = io.BytesIO(); noisy.save(b, "PNG"); png = len(b.getvalue())
    small = noisy.copy(); small.thumbnail((server.PREVIEW_MAX_SIDE, server.PREVIEW_MAX_SIDE))
    b = io.BytesIO(); small.save(b, "WEBP", quality=80); webp = len(b.getvalue())
    assert webp < png


# ── HTTP integration through the real app ───────────────────────────

@pytest.fixture
def client():
    with TestClient(server.app) as c:
        yield c


@pytest.fixture
def auth_enabled():
    server.AUTH.token = TOKEN
    server.AUTH.enabled = True
    server.AUTH.secure_cookie = False
    yield
    server.AUTH.enabled = False
    server.AUTH.token = ""


def test_auth_status_endpoint(client):
    r = client.get("/api/auth/status")
    assert r.status_code == 200
    assert r.json() == {"auth": False, "secure": False}


def test_auth_gate_blocks_api_without_cookie(client, auth_enabled):
    r = client.get("/api/models")
    assert r.status_code == 401


def test_auth_gate_serves_login_page_at_root(client, auth_enabled):
    r = client.get("/")
    assert r.status_code == 200
    assert "sign in" in r.text.lower()
    assert "Diffu<em>core</em>" not in r.text  # the app shell isn't exposed


def test_auth_gate_auto_login_with_token_query(client, auth_enabled):
    r = client.get("/", params={"token": TOKEN}, follow_redirects=False)
    assert r.status_code == 303
    assert authmod.COOKIE_NAME in r.headers.get("set-cookie", "")
    # A wrong token is rejected.
    r2 = client.get("/", params={"token": "nope"}, follow_redirects=False)
    assert r2.status_code == 401


def test_auth_gate_accepts_bearer_header(client, auth_enabled):
    r = client.get("/api/models", headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.status_code == 200


def test_auth_gate_accepts_cookie(client, auth_enabled):
    r = client.get("/", params={"token": TOKEN}, follow_redirects=False)
    cookie = r.headers["set-cookie"].split(";")[0]
    name, _, value = cookie.partition("=")
    r2 = client.get("/api/models", cookies={name: value})
    assert r2.status_code == 200


def test_csrf_blocks_cross_origin_post(client):
    # Auth is off here; the Origin check runs regardless.
    r = client.post("/api/cancel", json={"job": None},
                    headers={"Origin": "https://evil.example"})
    assert r.status_code == 403


def test_csrf_allows_same_origin_post(client):
    r = client.post("/api/cancel", json={"job": None},
                    headers={"Origin": "http://testserver"})
    # No job running → 200 with {cancelling: False}; the point is it isn't 403.
    assert r.status_code == 200


def test_body_size_cap_rejects_oversize(client):
    # Content-Length over the global cap → 413 before the handler runs.
    r = client.post("/api/generate", json={"steps": 999999999},
                    headers={"Content-Length": str(server.MAX_BODY_BYTES + 1)})
    assert r.status_code == 413


def test_load_returns_400_for_missing_file(client):
    r = client.post("/api/load", json={"model_type": "SD/SDXL",
                                       "checkpoint": "ghost.safetensors"})
    assert r.status_code == 400
    assert "not found" in r.json()["detail"].lower()


def test_generate_rejects_out_of_range_steps(client):
    r = client.post("/api/generate", json={"steps": 100000})
    assert r.status_code == 422  # pydantic validation error
