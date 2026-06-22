"""Optional Cloudflare quick-tunnel support for ``--share``.

Exposes the local UI over a public ``trycloudflare.com`` URL with no Cloudflare
account or login. The official ``cloudflared`` binary is used from ``PATH`` if
present, otherwise downloaded once and cached in the repo. The tunnel runs in
the background and is torn down when the process exits.
"""

from __future__ import annotations

import atexit
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

_BIN_DIR = Path(__file__).resolve().parent.parent / ".cloudflared"
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
    print(f"[share] downloading cloudflared ({url}) ...", flush=True)

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


def start(port: int, token: str | None = None) -> None:
    """Launch a quick tunnel to 127.0.0.1:<port> and print the public URL.

    Returns immediately; the URL is printed from a background thread once
    cloudflared registers with the Cloudflare edge. When ``token`` is given the
    printed URL carries ``?token=…`` so the auth gate (enabled for --share) lets
    the first visit straight through to the app.
    """
    try:
        binary = _binary()
    except Exception as exc:  # noqa: BLE001 — download/extraction can fail many ways
        print(f"[share] could not obtain cloudflared: {exc}", file=sys.stderr)
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
                    print(
                        f"\n=== Public share URL ===\n  {url}{suffix}\n"
                        "  (anyone with this link can reach your UI; Ctrl+C to "
                        "stop)\n",
                        flush=True,
                    )
                    printed = True

    threading.Thread(target=_watch, daemon=True).start()
