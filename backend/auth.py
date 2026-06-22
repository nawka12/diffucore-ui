"""Token + cookie authentication and CSRF/Origin guards for Diffucore UI.

The local single-user case (no ``--listen`` / ``--share``) needs no auth — the
loopback interface is already private. The moment the UI is exposed beyond
localhost (``--share`` publishes it to the public internet; ``--listen`` to the
LAN), an unauthenticated surface lets anyone drive the GPU, delete gallery
images, and install extensions (RCE). This module gates that surface with a
bearer token round-tripped through an HttpOnly ``SameSite=Lax`` cookie, plus an
Origin allowlist that blocks cross-site POSTs (CSRF) regardless of auth.

Flow:
  - ``--share`` (or explicit ``--auth-token``) enables the gate. A token is
    generated if one isn't supplied, printed once, and written to ``.auth_token``
    (chmod 600) so a restarted server keeps the same token.
  - ``GET /`` without a valid cookie serves a small login page. Submitting the
    token (or visiting ``/?token=<token>``) sets the cookie and redirects to ``/``.
  - Every other request (API, SSE, /outputs, /api/thumb) requires the cookie, a
    ``?token=`` query, or an ``Authorization: Bearer <token>`` header. Same-origin
    fetch / EventSource / <img> send the Lax cookie automatically, so the existing
    frontend needs no changes.
  - State-changing methods (POST/PUT/DELETE/PATCH) additionally require the
    ``Origin`` (or ``Referer``) host to match the request ``Host`` — blocking
    cross-site browser attacks. Requests with no Origin (curl, native clients)
    pass through, since the CSRF risk is browser-only.

Token comparison uses ``secrets.compare_digest`` to avoid timing oracles.
"""

from __future__ import annotations

import html
import secrets
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from fastapi import Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

COOKIE_NAME = "diffucore_auth"
_TOKEN_PATH = Path(__file__).resolve().parent.parent / ".auth_token"
_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
_STATE_CHANGE = {"POST", "PUT", "DELETE", "PATCH"}

# Paths reachable without a valid cookie when auth is enabled. ``GET /`` serves
# the login page (and the app once the cookie is set); the auth endpoints
# obviously can't require a cookie to log in. Everything else is gated.
_PUBLIC_PATHS = {"/", "/api/auth/login", "/api/auth/status"}


def generate_token() -> str:
    return secrets.token_urlsafe(24)


def load_or_create_token(path: Path = _TOKEN_PATH) -> str:
    """Return the persisted token, creating + chmod-600-ing one if absent.

    A stable token across restarts matters: a user who bookmarks the share URL
    with ``?token=…`` keeps working after a restart, and other devices that
    already hold the cookie aren't logged out.
    """
    try:
        if path.is_file():
            token = path.read_text(encoding="utf-8").strip()
            if token:
                return token
    except OSError:
        pass
    token = generate_token()
    try:
        path.write_text(token + "\n", encoding="utf-8")
        # 600: owner read/write only — the token grants full UI access.
        path.chmod(0o600)
    except OSError:
        # Read-only install / sandboxed FS: carry on with an in-memory token.
        pass
    return token


def _header_origin_host(request: Request) -> Optional[str]:
    """The host part of the request's ``Origin`` or ``Referer`` header, or None.

    Used for the CSRF check: a browser-driven cross-site POST carries an Origin
    whose host differs from the server's; a same-origin fetch carries a matching
    one; a non-browser client (curl) sends neither.
    """
    for header in ("origin", "referer"):
        val = request.headers.get(header)
        if not val:
            continue
        try:
            host = urlparse(val).hostname
        except ValueError:
            continue
        if host:
            return host.lower()
    return None


def origin_ok(request: Request) -> bool:
    """True if a state-changing request is same-origin (or has no Origin).

    Browsers always emit an ``Origin`` on cross-site POSTs and (in modern
    browsers) same-origin ones too; an absent Origin means a non-browser client,
    which isn't a CSRF vector. We only reject when an Origin *is* present and its
    host doesn't match the request's ``Host``.
    """
    if request.method not in _STATE_CHANGE:
        return True
    origin_host = _header_origin_host(request)
    if origin_host is None:
        return True  # curl / native client — not a browser CSRF vector
    request_host = (request.headers.get("host") or "").split("@")[-1].lower()
    # ``Host`` may carry the port; compare the hostname portion only.
    request_hostname = request_host.split(":")[0]
    return origin_host == request_hostname


class AuthGate:
    """Token + cookie auth state. ``enabled=False`` short-circuits every check."""

    def __init__(self, token: str, *, enabled: bool, secure_cookie: bool = False):
        self.enabled = enabled
        self.token = token
        self.secure_cookie = secure_cookie

    # ── credential extraction ──────────────────────────────────────

    def _presented_token(self, request: Request) -> Optional[str]:
        cookie = request.cookies.get(COOKIE_NAME)
        if cookie:
            return cookie
        qp = request.query_params.get("token")
        if qp:
            return qp
        authz = request.headers.get("authorization") or ""
        if authz.lower().startswith("bearer "):
            return authz[7:].strip()
        return None

    def _valid(self, request: Request) -> bool:
        presented = self._presented_token(request)
        if presented is None:
            return False
        return secrets.compare_digest(presented, self.token)

    def has_access(self, request: Request) -> bool:
        return not self.enabled or self._valid(request)

    # ── cookie + responses ──────────────────────────────────────────

    def _set_cookie(self, response: Response) -> None:
        opts = {
            "key": COOKIE_NAME,
            "value": self.token,
            "httponly": True,
            "samesite": "lax",
            "path": "/",
            "max_age": 30 * 24 * 3600,  # 30 days; a restart keeps the token
        }
        if self.secure_cookie:
            opts["secure"] = True
        response.set_cookie(**opts)

    def clear_cookie(self, response: Response) -> None:
        response.delete_cookie(COOKIE_NAME, path="/")

    def login_page(self) -> HTMLResponse:
        return HTMLResponse(_LOGIN_HTML, headers={"Cache-Control": "no-store"})

    def accept(self, token: Optional[str]) -> Optional[Response]:
        """Validate a presented token; return a redirect that sets the cookie,
        or None if the token is wrong (caller returns 401)."""
        if not token or not secrets.compare_digest(token, self.token):
            return None
        # Redirect to bare "/" so the token isn't left in the browser history.
        resp = RedirectResponse("/", status_code=303)
        self._set_cookie(resp)
        return resp

    def gate_response(self, request: Request) -> Optional[Response]:
        """Return a 401 response if the request lacks credentials, else None."""
        if self.has_access(request):
            return None
        if request.method == "GET" and request.url.path == "/api/events":
            # EventSource can't surface 401 to the user; return a short text
            # so the reconnect loop doesn't spin silently.
            return JSONResponse({"error": "unauthorized"}, status_code=401,
                                headers={"WWW-Authenticate": 'Bearer realm="diffucore"'})
        return JSONResponse({"error": "unauthorized"}, status_code=401,
                            headers={"WWW-Authenticate": 'Bearer realm="diffucore"'})


# Reading an arbitrary body inside the auth gate (form/json) without consuming it
# for the route handler is fiddly; for the login endpoint we accept either a
# query param (used by the redirect-after-?token= flow) or a small JSON/form body.
async def read_login_token(request: Request) -> Optional[str]:
    qp = request.query_params.get("token")
    if qp:
        return qp
    ct = (request.headers.get("content-type") or "").lower()
    if "application/json" in ct:
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return None
        if isinstance(body, dict):
            return body.get("token")
    if "application/x-www-form-urlencoded" in ct:
        try:
            body = await request.body()
        except Exception:  # noqa: BLE001
            return None
        from urllib.parse import parse_qs
        parsed = parse_qs(body.decode("utf-8", "replace"))
        vals = parsed.get("token")
        return vals[0] if vals else None
    return None


_LOGIN_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Diffucore — sign in</title>
<style>
  html,body{height:100%;margin:0;background:#0f1115;color:#e6e6e6;
    font-family:Inter,system-ui,Segoe UI,sans-serif}
  .card{position:absolute;inset:0;display:flex;align-items:center;
    justify-content:center}
  form{background:#161a21;padding:2rem 2.25rem;border-radius:14px;
    border:1px solid #232831;box-shadow:0 10px 40px rgba(0,0,0,.4);
    width:min(420px,92vw)}
  h1{font-size:1.15rem;margin:.1rem 0 1.1rem;font-weight:600}
  label{display:block;font-size:.8rem;color:#9aa4b2;margin-bottom:.4rem}
  input{width:100%;box-sizing:border-box;padding:.7rem .8rem;border-radius:8px;
    border:1px solid #2a313d;background:#0f1115;color:#e6e6e6;font-size:.95rem}
  input:focus{outline:none;border-color:#4a7dff}
  button{margin-top:1.1rem;width:100%;padding:.7rem;border:0;border-radius:8px;
    background:#4a7dff;color:#fff;font-weight:600;font-size:.95rem;cursor:pointer}
  button:hover{background:#3a6ae8}
  .hint{margin-top:1rem;font-size:.78rem;color:#6b7280;line-height:1.4}
  .err{color:#ff6b6b;margin-top:.6rem;font-size:.82rem;min-height:1em}
</style></head><body>
<div class="card"><form method="POST" action="/api/auth/login">
  <h1>Diffucore — sign in</h1>
  <label for="token">Access token</label>
  <input name="token" id="token" type="password" autofocus required
         placeholder="Paste the token printed by the server">
  <button type="submit">Unlock</button>
  <div class="err"></div>
  <p class="hint">The token was printed in the terminal when the server started
     (and saved to <code>.auth_token</code>). It is required because this UI is
     exposed beyond localhost.</p>
</form></div></body></html>"""
