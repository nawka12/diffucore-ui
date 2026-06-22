"""Extension platform for Diffucore UI.

AUTO1111 / ComfyUI style: each subdirectory of ``extensions/`` is one
extension, declared by an ``extension.json`` manifest. The loader scans the
directory at startup, imports each enabled extension's Python entry point, and
hands it an :class:`ExtensionAPI` it can use to register API routes, static
assets, generation hooks, custom job types, and SSE broadcasts. Extension JS
assets are served and injected into the index page so extensions can add their
own UI.

One broken extension never breaks the app: each load and each hook call is
wrapped, failures are recorded on the extension and surfaced in the Extensions
settings panel, and a disabled extension is simply never imported.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import random
import shutil
import string
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from fastapi import APIRouter
from pydantic import BaseModel

log = logging.getLogger("diffucore.extensions")

ROOT = Path(__file__).resolve().parent.parent
EXTENSIONS_DIR = ROOT / "extensions"
STATE_PATH = EXTENSIONS_DIR / "state.json"


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (temp file + os.replace) so a crash
    mid-write leaves the previous state intact instead of a truncated file."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix="." + path.name + "-", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(tmp, path)
        except Exception:
            try: os.unlink(tmp)
            except OSError: pass
            raise
    except OSError:
        pass


# Schemes permitted for ``/api/extensions/install``. Restricting to HTTPS closes
# the SSRF / local-file-read vector (``file://``, ``http://localhost``, the
# ``169.254.169.254`` metadata service, ``gopher://``, ``ftp://``) and the SSH
# agent exfil vector (``git@host:``, ``ssh://``). Users with private repos can
# clone manually into extensions/ or use an https URL with an embed token.
_BLOCKED_HOSTS = {"169.254.169.254", "metadata.google.internal"}


def _validate_install_url(url: str) -> None:
    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    if scheme != "https":
        raise ValueError(
            f"install URL must use https:// (got {scheme or 'no scheme'!r}); "
            "file://, http://, ssh, git@, ftp://, gopher:// are blocked")
    host = (parsed.hostname or "").lower()
    if host in _BLOCKED_HOSTS:
        raise ValueError(f"install URL host {host!r} is blocked (metadata service)")


def _random_suffix() -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=8))

# Events an extension can hook via ``api.on(event, handler)``. Handlers receive
# a :class:`HookContext` and may mutate it; the server reads the relevant
# fields back after running the hooks (e.g. ``post_generate`` can replace
# ``ctx.image``).
HOOK_EVENTS = {
    "startup", "shutdown",
    "pre_generate", "post_generate", "post_save",
    "pre_load", "post_load",
}

# The manifest fields we read out of extension.json, with their defaults. An
# unknown field is left in place but ignored, so a newer manifest never breaks
# an older loader. ``default_enabled`` controls whether a freshly-discovered
# extension (no state.json entry yet) loads on startup — an example or opt-in
# extension sets it to False so it shows up in the panel but doesn't run until
# the user turns it on.
_DEFAULTS = dict(
    title="",
    version="0.0.0",
    author="",
    description="",
    entry="extension.py",
    web="web",
    min_ui_version="0.1.0",
    default_enabled=True,
)


@dataclass
class HookContext:
    """Mutable bag passed through a hook chain. Fields are only set for the
    events that carry them; an extension should check before use.

    - ``pre_generate`` / ``post_generate`` / ``post_save``: ``payload`` is the
      :class:`GeneratePayload`, ``image`` is the PIL image (post-gen / post-save
      only), ``info`` is the human-readable info string, ``path`` is the saved
      file Path (post_save only).
    - ``pre_load`` / ``post_load``: ``payload`` is the :class:`LoadPayload`,
      ``status`` is the load result string (post_load only).
    - ``startup`` / ``shutdown``: nothing is set.
    """
    event: str
    payload: Any = None
    image: Any = None
    info: str = ""
    path: Any = None
    status: str = ""
    extra: dict = field(default_factory=dict)


@dataclass
class Extension:
    name: str
    title: str
    version: str
    author: str = ""
    description: str = ""
    entry: str = "extension.py"
    web: str = "web"
    min_ui_version: str = "0.1.0"
    default_enabled: bool = True
    path: Path = field(default_factory=lambda: Path())
    enabled: bool = True
    load_error: Optional[str] = None
    module: Any = None
    api: Any = None
    web_scripts: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "title": self.title or self.name,
            "version": self.version,
            "author": self.author,
            "description": self.description,
            "enabled": self.enabled,
            "default_enabled": self.default_enabled,
            "loaded": self.module is not None and self.load_error is None,
            "load_error": self.load_error,
            "web_scripts": self.web_scripts,
            "has_ui": bool(self.web_scripts),
        }


class ExtensionAPI:
    """The surface a single extension is given to register itself.

    All registration flows through this object so the loader can track what an
    extension installed (routes, hooks, job types) and unwind it cleanly if the
    extension is disabled or reloaded.
    """

    def __init__(self, ext: Extension, loader: "ExtensionLoader"):
        self._ext = ext
        self._loader = loader
        # Read-only access to the Engine singleton. Reloading a model outside
        # the shared job worker would race with generation, so extensions should
        # only inspect state here and do model work through enqueue_job.
        self.engine = loader.engine
        self.root_dir = ROOT
        self.ext_dir = ext.path

    def on(self, event: str, handler: Callable[[HookContext], None]) -> None:
        if event not in HOOK_EVENTS:
            raise ValueError(f"unknown hook event {event!r}; one of {sorted(HOOK_EVENTS)}")
        self._loader._hooks.setdefault(event, []).append((self._ext.name, handler))

    def add_api_router(self, router: APIRouter, *, prefix: str = "") -> None:
        """Mount a FastAPI router under ``/api/ext/<name><prefix>``."""
        full = f"/api/ext/{self._ext.name}{prefix}"
        self._loader._routers.append((self._ext.name, router, full))

    def serve_static(self, path: str, directory: Path) -> None:
        """Serve a directory at ``/ext-static/<name>/<path>``. The prefix is
        separate from the app's ``/static`` mount so the two never collide on
        path resolution."""
        self._loader._statics.append((self._ext.name, path, Path(directory)))

    def enqueue_job(self, label: str, run: Callable, *, kind: str = "ext") -> int:
        """Queue a callable on the shared background worker (one at a time with
        generation, so it shares the GPU safely). ``run`` receives the server's
        Job object. Returns the job id."""
        return self._loader._enqueue_job_fn(self._ext.name, label, run, kind=kind)

    def broadcast(self, event: dict) -> None:
        """Push an event dict to every connected SSE client."""
        self._loader._broadcast_fn(event)

    def add_web_scripts(self, files: List[str]) -> None:
        """Explicit JS files (relative to the extension's web dir) to inject
        into the index page. By default every ``.js`` file directly in the web
        dir is injected; this overrides that list."""
        self._ext.web_scripts = list(files)

    def get_setting(self, key: str, default: Any = None) -> Any:
        return self._loader._ext_settings(self._ext.name).get(key, default)

    def set_setting(self, key: str, value: Any) -> None:
        self._loader._set_ext_setting(self._ext.name, key, value)


class ExtensionLoader:
    """Owns the extension registry, the hook chains, and the install/toggle
    lifecycle. The server constructs one and wires it into FastAPI."""

    def __init__(
        self,
        engine,
        *,
        enqueue_job: Callable[[str, str, Callable, str], int],
        broadcast: Callable[[dict], None],
    ):
        self.engine = engine
        self._enqueue_job_fn = enqueue_job
        self._broadcast_fn = broadcast
        self.extensions: Dict[str, Extension] = {}
        self._hooks: Dict[str, List[Tuple[str, Callable]]] = {}
        self._routers: List[Tuple[str, APIRouter, str]] = []
        self._statics: List[Tuple[str, str, Path]] = []
        # Keys of what mount_into has already attached to the app, so it can be
        # re-called after a runtime install/enable/reload without double-mounting.
        self._mounted: set = set()
        self._state: Dict[str, Any] = self._read_state()
        EXTENSIONS_DIR.mkdir(parents=True, exist_ok=True)

    # ── persisted state ─────────────────────────────────────────────

    def _read_state(self) -> dict:
        try:
            return json.loads(STATE_PATH.read_text())
        except (OSError, ValueError):
            return {"enabled": {}, "ext_settings": {}}

    def _write_state(self) -> None:
        _atomic_write_text(STATE_PATH, json.dumps(self._state, indent=2))

    def _ext_settings(self, name: str) -> dict:
        return self._state.setdefault("ext_settings", {}).setdefault(name, {})

    def _set_ext_setting(self, name: str, key: str, value: Any) -> None:
        self._ext_settings(name)[key] = value
        self._write_state()

    def _is_enabled(self, ext: Extension) -> bool:
        # An explicit state.json entry (from the user toggling the extension)
        # always wins. Without one, fall back to the manifest's
        # ``default_enabled`` — True for a normal extension (so a freshly-dropped
        # folder works immediately), False for one that opts out (e.g. an
        # example extension that should show in the panel but not auto-load).
        return bool(self._state.get("enabled", {}).get(ext.name, ext.default_enabled))

    def _set_enabled(self, name: str, enabled: bool) -> None:
        self._state.setdefault("enabled", {})[name] = bool(enabled)
        self._write_state()

    # ── scan + load ─────────────────────────────────────────────────

    def scan(self) -> List[Extension]:
        """List every extension folder that has a manifest, without loading."""
        out: List[Extension] = []
        for child in sorted(EXTENSIONS_DIR.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            manifest = child / "extension.json"
            if not manifest.is_file():
                continue
            ext = self._parse_manifest(child)
            if ext is not None:
                out.append(ext)
        return out

    @staticmethod
    def _parse_manifest(path: Path) -> Optional[Extension]:
        try:
            data = json.loads((path / "extension.json").read_text())
        except (OSError, ValueError) as e:
            return Extension(
                name=path.name, title=path.name, version="",
                path=path, enabled=False, load_error=f"bad manifest: {e}",
            )
        if not isinstance(data, dict) or "name" not in data:
            return Extension(
                name=path.name, title=path.name, version="",
                path=path, enabled=False, load_error="manifest missing 'name'",
            )
        kw = {k: data.get(k, default) for k, default in _DEFAULTS.items()}
        return Extension(
            name=str(data["name"]),
            path=path,
            enabled=True,  # refined by _is_enabled in load()
            **kw,
        )

    def load_all(self) -> None:
        """Scan and load every enabled extension. Called once at startup."""
        for ext in self.scan():
            ext.enabled = self._is_enabled(ext)
            self.extensions[ext.name] = ext
            if ext.enabled and ext.load_error is None:
                self._load_one(ext)
        self.run_hook("startup")

    def _load_one(self, ext: Extension) -> None:
        """Import the entry module and call its ``setup(api)`` if present."""
        entry = ext.path / ext.entry
        if not entry.is_file():
            ext.load_error = f"entry file {ext.entry!r} not found"
            return
        # Unique module name so a reload after an edit doesn't hit a cached
        # sys.modules entry from the previous version.
        mod_name = f"_diffucore_ext_{ext.name}"
        spec = importlib.util.spec_from_file_location(mod_name, entry)
        if spec is None or spec.loader is None:
            ext.load_error = "could not import entry module"
            return
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        try:
            spec.loader.exec_module(module)
            api = ExtensionAPI(ext, self)
            ext.api = api
            ext.web_scripts = self._default_web_scripts(ext)
            setup = getattr(module, "setup", None)
            if callable(setup):
                setup(api)
            ext.module = module
            ext.load_error = None
        except Exception as e:  # noqa: BLE001 — isolate one extension's failure
            ext.load_error = f"{type(e).__name__}: {e}"
            log.exception("extension %s failed to load", ext.name)
            sys.modules.pop(mod_name, None)
            ext.module = None

    @staticmethod
    def _default_web_scripts(ext: Extension) -> List[str]:
        web = ext.path / ext.web
        if not web.is_dir():
            return []
        return sorted(f.name for f in web.iterdir() if f.is_file() and f.suffix == ".js")

    def reload_one(self, name: str) -> None:
        """Re-scan the manifest and re-import one extension (after an edit or a
        toggle). Drops its old hooks/routes/statics so they don't double up."""
        self._unload_one(name)
        path = EXTENSIONS_DIR / name
        if not path.is_dir():
            self.extensions.pop(name, None)
            return
        ext = self._parse_manifest(path)
        if ext is None:
            return
        ext.enabled = self._is_enabled(ext)
        self.extensions[name] = ext
        if ext.enabled and ext.load_error is None:
            self._load_one(ext)

    def _unload_one(self, name: str) -> None:
        ext = self.extensions.pop(name, None)
        if ext is None:
            return
        # Drop this extension's hooks / routes / statics so a reload is clean.
        for ev, handlers in self._hooks.items():
            self._hooks[ev] = [(n, h) for (n, h) in handlers if n != name]
        self._routers = [(n, r, p) for (n, r, p) in self._routers if n != name]
        self._statics = [(n, p, d) for (n, p, d) in self._statics if n != name]
        sys.modules.pop(f"_diffucore_ext_{name}", None)

    # ── hook dispatch ───────────────────────────────────────────────

    def run_hook(self, event: str, **fields) -> HookContext:
        """Run every handler registered for ``event`` in registration order.
        A handler that raises is logged and skipped — a buggy extension can't
        abort a generation or a load."""
        ctx = HookContext(event=event, **fields)
        for name, handler in list(self._hooks.get(event, [])):
            try:
                handler(ctx)
            except Exception as e:  # noqa: BLE001
                log.exception("extension %s hook %s failed", name, event)
                # Surface the failure on the extension so the settings panel
                # can show it, without disabling the extension outright.
                ext = self.extensions.get(name)
                if ext is not None and ext.load_error is None:
                    ext.load_error = f"{event} hook: {type(e).__name__}: {e}"
        return ctx

    # ── install / uninstall / toggle ────────────────────────────────

    def install(self, url: str) -> Extension:
        """Install from a git URL or a .zip archive URL. Returns the new
        extension's record (loaded, if it succeeded).

        The final directory name is the manifest's ``name`` field (not the URL
        basename), so uninstall/toggle keyed on the manifest name always find
        the right folder. Extraction happens into a scratch name first, then
        the folder is moved to its canonical slot once the manifest is known.
        """
        import re as _re
        EXTENSIONS_DIR.mkdir(parents=True, exist_ok=True)
        _validate_install_url(url)
        scratch = EXTENSIONS_DIR / ("__installing__" + _random_suffix())
        try:
            if url.lower().endswith(".zip"):
                self._install_zip(url, scratch)
            else:
                self._install_git(url, scratch)
            # The cloned/extracted folder may contain the extension at its root
            # or one level down (a common repo layout). Normalise to the root.
            root = self._find_extension_root(scratch)
            if root is None:
                raise ValueError("no extension.json found in the downloaded source")
            ext = self._parse_manifest(root)
            if ext is None:
                raise ValueError("invalid manifest in the downloaded source")
            # Sanitize the manifest name: only allow filename-safe chars and
            # reject path escapes, so a malicious manifest can't write outside
            # extensions/.
            safe = _re.sub(r"[^A-Za-z0-9._-]", "_", ext.name)
            if not safe or safe in (".", ".."):
                raise ValueError(f"invalid extension name {ext.name!r}")
            target = EXTENSIONS_DIR / safe
            if target.exists():
                raise ValueError(f"an extension named {safe!r} already exists")
            root.rename(target)
        except Exception:
            shutil.rmtree(scratch, ignore_errors=True)
            raise
        finally:
            # Clean up the scratch dir if a failed install left it behind and
            # it wasn't already moved into place.
            if scratch.exists() and scratch.name.startswith("__installing__"):
                shutil.rmtree(scratch, ignore_errors=True)
        ext = self._parse_manifest(target)
        if ext is None:
            raise ValueError("invalid manifest after install")
        # Use the sanitized name for the registry key, URL prefix
        # (/api/ext/<name>), and state — matching the on-disk directory — so a
        # manifest with odd characters can't escape via path or URL.
        ext.name = target.name
        ext.enabled = True
        self._set_enabled(ext.name, True)
        self.extensions[ext.name] = ext
        self._load_one(ext)
        # Install deps after the module load attempt so a missing dependency
        # (the common failure) shows up as the extension's own load_error, and a
        # pip failure that leaves an importable-but-broken extension is still
        # surfaced on the record instead of silently "succeeding".
        pip_err = self._pip_install_requirements(target)
        if pip_err and ext.load_error is None:
            ext.load_error = pip_err
        self._write_state()
        return ext

    @staticmethod
    def _derive_name(url: str) -> str:
        tail = url.rstrip("/").split("/")[-1]
        if tail.endswith(".git"):
            tail = tail[:-4]
        if tail.endswith(".zip"):
            tail = tail[:-4]
        return tail or "extension"

    @staticmethod
    def _install_git(url: str, target: Path) -> None:
        _validate_install_url(url)
        # Abort if the clone stalls for 30s of <1KB/s (e.g. a tarpit host), and
        # cap the whole clone at 5 minutes so a slow/malicious server can't hang
        # the install request forever.
        env = {
            **os.environ,
            "GIT_HTTP_LOW_SPEED_TIME": "30",
            "GIT_HTTP_LOW_SPEED_LIMIT": "1024",
        }
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", url, str(target)],
                check=True, capture_output=True, timeout=300, env=env,
            )
        except subprocess.TimeoutExpired as e:
            raise ValueError("git clone timed out (5 min cap reached)") from e

    @staticmethod
    def _install_zip(url: str, target: Path) -> None:
        _validate_install_url(url)
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            # 60s connect/read timeout: same stall protection as the git path,
            # since urlretrieve has no per-host timeout of its own.
            with urllib.request.urlopen(url, timeout=60) as resp, \
                    open(tmp_path, "wb") as out:
                shutil.copyfileobj(resp, out)
            with zipfile.ZipFile(tmp_path) as zf:
                zf.extractall(target)
        except urllib.error.URLError as e:
            raise ValueError(f"could not download zip: {e}") from e
        finally:
            tmp_path.unlink(missing_ok=True)

    @staticmethod
    def _find_extension_root(path: Path) -> Optional[Path]:
        if (path / "extension.json").is_file():
            return path
        for child in path.iterdir():
            if child.is_dir() and (child / "extension.json").is_file():
                return child
        return None

    @staticmethod
    def _pip_install_requirements(ext_path: Path) -> Optional[str]:
        """Run ``pip install -r requirements.txt`` if present.

        Returns ``None`` on success (or no requirements.txt); returns the pip
        stderr on failure so ``install()`` can surface it on the extension
        record instead of silently reporting a successful install of an
        extension that won't load. Does not raise — the extension is already on
        disk, so we register it and let the user retry a reload after fixing the
        dependency."""
        req = ext_path / "requirements.txt"
        if not req.is_file():
            return None
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(req)],
                capture_output=True, text=True, timeout=600,
            )
        except subprocess.TimeoutExpired:
            return "pip install timed out (10 min cap reached)"
        except Exception as e:  # noqa: BLE001
            log.warning("pip install for extension failed: %s", e)
            return f"pip install failed: {e}"
        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "").strip().splitlines()
            tail = "\n".join(tail[-8:]) if tail else "(no output)"
            log.warning("pip install for %s failed (rc=%d):\n%s",
                        ext_path.name, result.returncode, tail)
            return f"pip install failed (rc={result.returncode}):\n{tail}"
        return None

    def uninstall(self, name: str) -> None:
        self._unload_one(name)
        self._state.get("enabled", {}).pop(name, None)
        self._state.get("ext_settings", {}).pop(name, None)
        self._write_state()
        target = EXTENSIONS_DIR / name
        # Only delete a real subdirectory of extensions/ — never a parent or a
        # symlink that escapes it.
        try:
            resolved = target.resolve()
            if (EXTENSIONS_DIR.resolve() not in resolved.parents
                    and resolved != EXTENSIONS_DIR.resolve()):
                raise ValueError("refusing to delete path outside extensions/")
            if target.is_dir():
                shutil.rmtree(target)
        except OSError:
            pass

    def set_enabled(self, name: str, enabled: bool) -> Extension:
        self._set_enabled(name, enabled)
        # Reload picks up the new state: load if just enabled, unload if just
        # disabled (so its routes/hooks go away).
        self.reload_one(name)
        return self.extensions.get(name) or Extension(
            name=name, title=name, path=EXTENSIONS_DIR / name, enabled=False,
            load_error="not found",
        )

    # ── introspection (for the server + UI) ─────────────────────────

    def list_serializable(self) -> List[dict]:
        # Refresh manifest fields (title/description may have changed on disk)
        # without reloading modules, so the panel reflects edits after a
        # restart-free file change only if the user reloads.
        return [ext.to_dict() for ext in self.extensions.values()]

    def web_script_urls(self) -> List[dict]:
        """The <script src=...> entries the index page should inject, one per
        enabled, loaded extension JS file."""
        out = []
        for ext in self.extensions.values():
            if not ext.enabled or ext.load_error or ext.module is None:
                continue
            for f in ext.web_scripts:
                out.append({
                    "name": ext.name,
                    "src": f"/ext-static/{ext.name}/{f}",
                })
        return out

    def mount_into(self, app) -> None:
        """Attach loaded extensions' routers and static mounts to the FastAPI app.

        Safe to call repeatedly: each router/static is mounted at most once
        (tracked in ``self._mounted``), so re-calling it after a runtime install,
        enable, or reload attaches a newly-loaded extension's routes without
        doubling up the ones already there. Starlette has no unmount, so the
        routes/statics of a *disabled* or *uninstalled* extension keep serving
        until the server restarts (documented in docs/EXTENSIONS.md)."""
        from fastapi.staticfiles import StaticFiles
        for _name, router, prefix in self._routers:
            if ("router", prefix) in self._mounted:
                continue
            try:
                app.include_router(router, prefix=prefix)
                self._mounted.add(("router", prefix))
            except Exception as e:  # noqa: BLE001
                log.warning("mounting router %s failed: %s", prefix, e)
        for name, path, directory in self._statics:
            mount = f"/ext-static/{name}/{path}"
            if ("static", mount) in self._mounted:
                continue
            try:
                app.mount(mount, StaticFiles(directory=str(directory)))
                self._mounted.add(("static", mount))
            except Exception as e:  # noqa: BLE001
                log.warning("mounting static %s failed: %s", mount, e)
        # Always serve each extension's own web/ dir under the canonical URL so
        # the injected script tags resolve even if the extension didn't call
        # serve_static for it.
        for ext in self.extensions.values():
            if ("web", ext.name) in self._mounted:
                continue
            web = ext.path / ext.web
            if not web.is_dir():
                continue
            try:
                app.mount(
                    f"/ext-static/{ext.name}",
                    StaticFiles(directory=str(web)),
                    name=f"ext_web_{ext.name}",
                )
                self._mounted.add(("web", ext.name))
            except Exception as e:  # noqa: BLE001
                log.warning("mounting web dir for %s failed: %s", ext.name, e)


# ── request / response models for the management API ────────────────

class InstallPayload(BaseModel):
    url: str


class TogglePayload(BaseModel):
    name: str
    enabled: bool


class UninstallPayload(BaseModel):
    name: str
