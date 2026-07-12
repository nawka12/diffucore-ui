"""SwarmUI-style tensor hashes for model files: compute, cache, and scan.

SwarmUI's model hash is a SHA256 over the safetensors *tensor-data* section —
the 8-byte header-length prefix and the JSON header are skipped — formatted as
``"0x"`` + hexdigest. (See SwarmUI ``T2IModel.GetOrGenerateTensorHashSha256``.)
We also compute the *full-file* SHA256 in the same pass: its first 10 hex chars
are Civitai/A1111's AutoV2 hash, which is what Civitai matches resources on and
what the A1111 ``Model hash:`` / ``Lora hashes:`` fields carry.
Hashing a multi-GB checkpoint is slow, so results are cached in a JSON file
keyed by the model's path + mtime + size (a re-download busts its entry). A
background scan at startup fills in any missing hashes; the metadata writer
only ever *reads* the cache, so saving an image never blocks on a hash.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import struct
import threading
import time
from pathlib import Path

from utils import (CHECKPOINTS_DIR, DIFFUSION_DIR, LORAS_DIR, MODELS_DIR,
                   checkpoint_path, diffusion_model_path, lora_path)

log = logging.getLogger("diffucore-ui.model_hash")

_CACHE_PATH = MODELS_DIR / ".model_hashes.json"
_LOCK = threading.Lock()
_CACHE: dict | None = None  # {rel_path: {"mtime": int_ns, "size": int, "hash": str, "sha256": str}}

_MODEL_EXTS = {".safetensors", ".ckpt", ".pt", ".pth"}

# Anima/FLUX load a split DiT and wrap its filename as ``Anima(dit.safetensors)``
# / ``FLUX(dit.safetensors)`` (see engine.load_anima/load_flux); everything else
# is a single-file checkpoint whose loaded name is the filename itself.
_WRAP_RE = re.compile(r"^(?:Anima|FLUX)\((.*)\)$")


# ── model-name / path identity ──────────────────────────────────────

def _unwrap(loaded_name: str) -> tuple[str, bool]:
    """``(filename, is_diffusion_model)`` from an engine ``loaded_name``."""
    m = _WRAP_RE.match(loaded_name)
    if m:
        return m.group(1), True
    return loaded_name, False


def clean_model_name(loaded_name: str, strip_ext: bool = False) -> str:
    """Human/tool-friendly model name: drop the ``Anima(...)``/``FLUX(...)``
    family wrapper. With ``strip_ext``, also drop a trailing model extension
    (SwarmUI's ``model`` param carries no extension; its ``sui_models`` name does)."""
    name = _unwrap(loaded_name)[0]
    if strip_ext:
        p = Path(name)
        if p.suffix.lower() in _MODEL_EXTS:
            name = p.stem
    return name


def resolve_model_file(loaded_name: str) -> Path | None:
    """Reverse an engine ``loaded_name`` to its on-disk weights file, or ``None``."""
    fname, is_dm = _unwrap(loaded_name)
    path = diffusion_model_path(fname) if is_dm else checkpoint_path(fname)
    return path if path.exists() else None


def resolve_lora_file(name: str) -> Path | None:
    """Resolve a ``<lora:name:…>`` name to its on-disk file, or ``None``.

    ``lora_path`` accepts the bare name (no extension), as LoRA prompt tags use."""
    p = lora_path(name)
    return p if p.exists() else None


# ── hash cache ──────────────────────────────────────────────────────

def _load_cache() -> dict:
    """Return the in-memory cache, loading it from disk on first use. Call under _LOCK."""
    global _CACHE
    if _CACHE is None:
        try:
            _CACHE = json.loads(_CACHE_PATH.read_text())
        except (OSError, json.JSONDecodeError):
            _CACHE = {}
    return _CACHE


def _save_cache() -> None:
    """Atomically persist the in-memory cache. Call under _LOCK."""
    try:
        tmp = _CACHE_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(_CACHE))
        os.replace(tmp, _CACHE_PATH)
    except OSError as e:
        log.warning("could not write model-hash cache: %s", e)


def _rel_key(path: Path) -> str:
    try:
        return path.resolve().relative_to(MODELS_DIR.resolve()).as_posix()
    except ValueError:
        return path.name


def _hash_file(path: Path) -> tuple[str | None, str | None]:
    """One-pass ``(tensor_hash, full_sha256)`` for a safetensors file, or ``(None, None)``.

    ``tensor_hash`` is SwarmUI's ``"0x"`` + SHA256 over just the tensor-data
    section (header skipped); ``full_sha256`` is the SHA256 over the *whole*
    file — Civitai/A1111's key, whose first 10 hex chars are AutoV2. Both are
    fed from a single read so hashing a multi-GB file only touches disk once."""
    try:
        with open(path, "rb") as f:
            head = f.read(8)
            header_len = struct.unpack("<Q", head)[0]
            full = hashlib.sha256()
            tensor = hashlib.sha256()
            full.update(head)
            full.update(f.read(header_len))          # 8-byte prefix + JSON header
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                full.update(chunk)                   # …then the tensor data:
                tensor.update(chunk)                 # full = whole file, tensor = data only
    except (OSError, struct.error) as e:
        log.warning("model-hash failed for %s: %s", path, e)
        return None, None
    return "0x" + tensor.hexdigest(), full.hexdigest()


def _tensor_hash(path: Path) -> str | None:
    """SwarmUI tensor-data SHA256 of a safetensors file (``"0x"`` + hex), or None."""
    return _hash_file(path)[0]


def _fresh_entry(cache: dict, key: str, st: os.stat_result) -> dict | None:
    """The cache entry for ``key`` if it still matches the file's mtime+size."""
    entry = cache.get(key)
    if entry and entry.get("mtime") == st.st_mtime_ns and entry.get("size") == st.st_size:
        return entry
    return None


def get_hash(path: Path | None) -> str | None:
    """Cached tensor hash for a model file, or ``None`` if not yet computed.

    Non-blocking: never hashes here (that's :func:`ensure_hash` / :func:`scan_all`),
    so the metadata write path stays fast. ``None`` for non-safetensors or a
    stale/missing cache entry."""
    if path is None or path.suffix.lower() != ".safetensors":
        return None
    try:
        st = path.stat()
    except OSError:
        return None
    with _LOCK:
        entry = _fresh_entry(_load_cache(), _rel_key(path), st)
        return entry.get("hash") if entry else None


def get_autov2(path: Path | None) -> str | None:
    """Cached AutoV2 (first 10 hex of the full-file SHA256) for Civitai/A1111, or None.

    Non-blocking, same contract as :func:`get_hash`: ``None`` for a non-safetensors
    file or a stale/missing/tensor-only (pre-upgrade) cache entry."""
    if path is None or path.suffix.lower() != ".safetensors":
        return None
    try:
        st = path.stat()
    except OSError:
        return None
    with _LOCK:
        entry = _fresh_entry(_load_cache(), _rel_key(path), st)
    full = entry.get("sha256") if entry else None
    return full[:10] if full else None


def ensure_hash(path: Path) -> str | None:
    """Return the tensor hash, computing + caching it if missing/stale. Blocking.

    Computes both the tensor hash and the full-file SHA256 in one pass; an older
    cache entry carrying only ``hash`` (no ``sha256``) is treated as stale so the
    AutoV2 value gets filled in on the next scan."""
    if path.suffix.lower() != ".safetensors":
        return None
    try:
        st = path.stat()
    except OSError:
        return None
    key = _rel_key(path)
    with _LOCK:
        entry = _fresh_entry(_load_cache(), key, st)
        if entry and entry.get("hash") and entry.get("sha256"):
            return entry.get("hash")
    tensor, full = _hash_file(path)  # slow — computed outside the lock
    if tensor is None:
        return None
    with _LOCK:
        _load_cache()[key] = {"mtime": st.st_mtime_ns, "size": st.st_size,
                              "hash": tensor, "sha256": full}
        _save_cache()
    return tensor


def _fmt_size(n: int) -> str:
    """Human-readable byte count, e.g. ``6442450944`` -> ``"6.0 GB"``."""
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024


def scan_all(background: bool = True) -> None:
    """Hash every checkpoint / diffusion model / LoRA missing a fresh cache entry.

    Runs once at startup. With ``background=False`` the caller blocks until every
    file is hashed (the server does this before serving, so hashing never contends
    with generation I/O); ``background=True`` runs it on a daemon thread instead.
    Either way, progress is logged per file so a big first batch shows advancement."""
    def _run() -> None:
        targets: list[Path] = []
        for d in (CHECKPOINTS_DIR, DIFFUSION_DIR, LORAS_DIR):
            if d.is_dir():
                targets += [p for p in sorted(d.iterdir())
                            if p.is_file() and p.suffix.lower() == ".safetensors"]
        missing = [p for p in targets if get_hash(p) is None or get_autov2(p) is None]
        if not missing:
            return
        total = len(missing)
        total_bytes = sum(p.stat().st_size for p in missing)
        log.info("[model-hash] hashing %d model(s) missing a hash (%s total)…",
                 total, _fmt_size(total_bytes))
        started = time.monotonic()
        for i, p in enumerate(missing, 1):
            log.info("[model-hash] [%d/%d] %s (%s)", i, total, p.name,
                     _fmt_size(p.stat().st_size))
            ensure_hash(p)
        log.info("[model-hash] scan complete: %d hashed in %.1fs",
                 total, time.monotonic() - started)

    if background:
        threading.Thread(target=_run, name="model-hash-scan", daemon=True).start()
    else:
        _run()
