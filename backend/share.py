"""Optional Cloudflare quick-tunnel support for ``--share``.

Exposes the local UI over a public ``trycloudflare.com`` URL with no Cloudflare
account or login. The official ``cloudflared`` binary is used from ``PATH`` if
present, otherwise downloaded once and cached in the repo. The tunnel runs in
the background and is torn down when the process exits.
"""

from __future__ import annotations

import atexit
import logging
import os
import platform
import re
import stat
import subprocess
import sys
import tarfile
import threading
import urllib.request
from pathlib import Path
from shutil import which

log = logging.getLogger("diffucore.share")

_BIN_DIR = Path(__file__).resolve().parent.parent / ".cloudflared"
_SHARE_URL_FILE = _BIN_DIR / "share_url.txt"
_RELEASE = "https://github.com/cloudflare/cloudflared/releases/latest/download"
_URL_RE = re.compile(r"https://[-\w.]+\.trycloudflare\.com")


def _asset() -> tuple[str, bool]:
    """Return (release asset filename, is_tgz) for the current platform."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    arch = {
        "x86_64": "amd64", "amd64": "amd64",
        "aarch64": "arm64", "arm64": "arm64",
        "i386": "386", "i686": "386", "x86": "386",
    }.get(machine, "arm" if machine.startswith("arm") else "amd64")

    if system == "windows":
        return f"cloudflared-windows-{arch}.exe", False
    if system == "darwin":
        return f"cloudflared-darwin-{arch}.tgz", True
    return f"cloudflared-linux-{arch}", False


def _binary() -> Path:
    """Locate cloudflared: PATH first, else the repo cache, else download it."""
    found = which("cloudflared")
    if found:
        return Path(found)

    is_windows = platform.system().lower() == "windows"
    target = _BIN_DIR / ("cloudflared.exe" if is_windows else "cloudflared")
    if target.exists():
        return target

    asset, is_tgz = _asset()
    url = f"{_RELEASE}/{asset}"
    _BIN_DIR.mkdir(parents=True, exist_ok=True)
    log.info("downloading cloudflared (%s)", url)

    if is_tgz:
        archive = _BIN_DIR / asset
        urllib.request.urlretrieve(url, archive)
        with tarfile.open(archive) as tf:
            member = next(m for m in tf.getmembers() if m.name.endswith("cloudflared"))
            member.name = target.name
            tf.extract(member, _BIN_DIR)
        archive.unlink()
    else:
        urllib.request.urlretrieve(url, target)

    target.chmod(target.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return target


def _write_share_url_file(url: str) -> Path | None:
    """Persist the full share URL (with ``?token=…``) to a ``chmod 600`` file so
    it isn't sitting in terminal scrollback / CI logs / screen-share captures.

    The URL + token grants full GPU access, so it's treated like the auth token
    (``.auth_token``): written to an owner-only file and referenced by path from
    the terminal warning, never printed to stdout or the structured log. Returns
    the path on success or ``None`` if the file couldn't be written (in which
    case we fall back to printing the URL with a loud warning)."""
    try:
        _BIN_DIR.mkdir(parents=True, exist_ok=True)
        _SHARE_URL_FILE.write_text(url + "\n", encoding="utf-8")
        try:
            os.chmod(_SHARE_URL_FILE, 0o600)
        except OSError:
            pass  # Windows ignores chmod
        return _SHARE_URL_FILE
    except OSError as e:
        log.warning("could not write share URL file: %s", e)
        return None


def _print_share_warning(url: str, suffix: str) -> None:
    """Print the public share URL with an explicit, hard-to-miss warning.

    The full link (with ``?token=…``) is written to a ``chmod 600`` file and the
    terminal only names the file + the risk — anyone with the link can reach the
    GPU, so don't paste it, clear scrollback if you've screen-shared, and Ctrl+C
    to stop. If the file write failed we fall back to printing the URL (still
    with the warning) so the user isn't locked out."""
    full = url + suffix
    path = _write_share_url_file(full)
    print(
        "\n=== Public share URL ===\n"
        + (f"  written to: {path} (chmod 600)\n"
           "  Open that file to copy the link. It is NOT printed here to keep it\n"
           "  out of terminal scrollback, screen-shares, and CI logs.\n"
           if path is not None
           else f"  {full}\n"
           "  WARNING: could not write the URL to a protected file, so it is\n"
           "  printed here. Treat this terminal like a secret.\n")
        + "  Anyone with this link can reach your UI and GPU. Ctrl+C to stop.\n"
        "  If you've already screen-shared, clear your scrollback now.\n",
        flush=True,
    )


def start(port: int, token: str | None = None) -> None:
    """Launch a quick tunnel to 127.0.0.1:<port> and print the public URL.

    Returns immediately; the URL is printed from a background thread once
    cloudflared registers with the Cloudflare edge. When ``token`` is given the
    persisted URL carries ``?token=…`` so the auth gate (enabled for --share) lets
    the first visit straight through to the app.
    """
    try:
        binary = _binary()
    except Exception as exc:  # noqa: BLE001 — download/extraction can fail many ways
        log.warning("could not obtain cloudflared: %s", exc)
        return

    proc = subprocess.Popen(
        [str(binary), "tunnel", "--no-autoupdate",
         "--url", f"http://127.0.0.1:{port}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    atexit.register(proc.terminate)

    def _watch() -> None:
        printed = False
        for line in proc.stderr:  # drains continuously so cloudflared never stalls
            if not printed:
                match = _URL_RE.search(line)
                if match:
                    url = match.group(0)
                    suffix = f"?token={token}" if token else ""
                    _print_share_warning(url, suffix)
                    printed = True

    threading.Thread(target=_watch, daemon=True).start()
