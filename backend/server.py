"""FastAPI web layer over the ``ENGINE`` singleton.

Jobs run one at a time on a background worker thread. Every device subscribes
to one Server-Sent-Events stream (``/api/events``) carrying queue changes,
progress, live previews and model-load status.
"""

from __future__ import annotations

import asyncio
import atexit
import base64
import hashlib
import io
import itertools
import json
import logging
import os
import random
import re
import shutil
import tempfile
import threading
import time
from collections import deque
from datetime import date
from functools import partial
from pathlib import Path
from typing import Callable, List, Literal, Optional

import uvicorn
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from engine import ENGINE, SAMPLERS_SD, SAMPLERS_ANIMA, SAMPLERS_FLUX, SCHEDULERS_SD, SCHEDULERS_ANIMA, SCHEDULERS_FLUX
from utils import (
    OUTPUTS_DIR, MODELS_DIR, CHECKPOINTS_DIR, DIFFUSION_DIR, VAE_DIR, TE_DIR,
    detector_path, model_name_ok,
    scan_checkpoints, scan_loras, scan_diffusion_models,
    scan_vae, scan_text_encoders, scan_detectors, scan_upscalers, scan_outputs, next_output_path,
    invalidate_outputs_cache,
)
from xyz_grid import generate_xyz_grid, PARAM_TYPES as XYZ_PARAM_TYPES
import metadata as md
import tagger as tagger_mod
from auth import (
    AuthGate, COOKIE_NAME, load_or_create_token, origin_ok, read_login_token,
    _STATE_CHANGE as _STATE_CHANGE_METHODS,
)
from extensions import (
    ExtensionLoader, InstallPayload, TogglePayload, UninstallPayload, UpdatePayload,
)

log = logging.getLogger("diffucore.server")

MAX_UPLOAD_BYTES = 64 * 1024 * 1024  # 64 MB cap for metadata-parse uploads
MAX_BODY_BYTES = 128 * 1024 * 1024   # global cap: base64 of a 4K PNG fits, GBs don't

_ROOT = Path(__file__).resolve().parent.parent
_STATIC = _ROOT / "static"

# Cache-bust token: newest static-file mtime in hex. Updates restart the server,
# so a startup snapshot is enough; index.html itself is served no-cache.
def _asset_version() -> str:
    mtimes = [
        (_STATIC / name).stat().st_mtime
        for name in ("index.html", "app.js", "style.css", "alpine.min.js")
        if (_STATIC / name).exists()
    ]
    return format(int(max(mtimes, default=0)), "x")

ASSET_VERSION = _asset_version()
_INDEX_HTML = (_STATIC / "index.html").read_text(encoding="utf-8").replace(
    "__ASSETV__", ASSET_VERSION
)
# DIFFUCORE_DEV=1 re-reads index.html (and recomputes the asset version) on every
# request, so frontend edits show up on a plain refresh.
_DEV_MODE = os.environ.get("DIFFUCORE_DEV") in ("1", "true", "yes")


# ── auth + CSRF guard ───────────────────────────────────────────────
# Off by default. app.py enables it for --share / --auth-token, or set
# DIFFUCORE_AUTH_TOKEN. CSRF origin checks apply regardless.
AUTH = AuthGate(token="", enabled=False)


def configure_auth(*, token: str, enabled: bool, secure: bool = False) -> None:
    """Turn the auth gate on. The middleware reads ``AUTH`` per request, so
    calling this before uvicorn.run covers the first request."""
    AUTH.token = token
    AUTH.enabled = enabled
    AUTH.secure_cookie = secure


if os.environ.get("DIFFUCORE_AUTH_TOKEN"):
    configure_auth(
        token=os.environ["DIFFUCORE_AUTH_TOKEN"],
        enabled=True,
        secure=os.environ.get("DIFFUCORE_AUTH_SECURE", "") in ("1", "true", "yes"),
    )


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` via a temp file + ``os.replace``, so a crash
    mid-write leaves the previous file intact."""
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


def _ext_script_tags() -> str:
    # Built per render so a freshly installed extension shows up without a restart.
    return "".join(
        f'<script src="{s["src"]}?v={ASSET_VERSION}" defer></script>'
        for s in EXTENSIONS.web_script_urls()
    )


def _render_index() -> str:
    if not _DEV_MODE:
        return _INDEX_HTML.replace("__EXT_SCRIPTS__", _ext_script_tags())
    version = _asset_version()
    return (_STATIC / "index.html").read_text(encoding="utf-8").replace(
        "__ASSETV__", version
    ).replace("__EXT_SCRIPTS__", _ext_script_tags())
_THUMBS_DIR = _ROOT / ".cache" / "thumbs"
THUMB_MAX = 384  # long-edge px; the grid uses these instead of the full PNGs

# ── AI NSFW ratings (WD tagger, .cache/ratings.json) ─────────────────
# Keyed by gallery path and validated against the file's mtime+size, so an
# overwritten image re-rates. The prompt heuristic in metadata.py is the
# fallback for unrated images.
_RATINGS_PATH = _ROOT / ".cache" / "ratings.json"

_RATINGS: Optional[dict] = None     # path -> {"rating","nsfw","conf","v","key"}
_RATINGS_LOCK = threading.Lock()


def _image_key(path: Path) -> str:
    """mtime_ns+size fingerprint, same idea as the thumb cache."""
    try:
        st = path.stat()
        return f"{st.st_mtime_ns}_{st.st_size}"
    except OSError:
        return ""


def _read_ratings() -> dict:
    global _RATINGS
    if _RATINGS is None:
        try:
            _RATINGS = json.loads(_RATINGS_PATH.read_text())
        except (OSError, ValueError):
            _RATINGS = {}
    return _RATINGS


def _rating_current(entry: Optional[dict], path: Path) -> bool:
    """Whether a cached entry still describes ``path``: same mtime+size and the
    current decision-layer version."""
    return bool(entry and entry.get("key") == _image_key(path)
                and entry.get("v") == tagger_mod.DECISION_VERSION)


def _cached_rating(path: Path) -> Optional[dict]:
    """The vision rating for ``path`` if it's still current on disk."""
    rel = path.relative_to(OUTPUTS_DIR).as_posix()
    entry = _read_ratings().get(rel)
    if not _rating_current(entry, path):
        return None
    return {"rating": entry["rating"], "nsfw": entry["nsfw"],
            "conf": entry.get("conf", 0.0)}


def _store_ratings(entries: dict) -> None:
    """Merge rated entries into the cache and persist atomically."""
    with _RATINGS_LOCK:
        cache = _read_ratings()
        for rel, e in entries.items():
            e["v"] = tagger_mod.DECISION_VERSION
            cache[rel] = e
        _atomic_write_text(_RATINGS_PATH, json.dumps(cache))


# ── gallery soft-delete ───────────────────────────────────────────────
# DELETE /api/gallery moves files here so a mistaken delete can be recovered by
# hand. Entries older than TRASH_RETENTION_DAYS are purged at startup and on
# each delete.
_TRASH_DIR = OUTPUTS_DIR / ".trash"
TRASH_RETENTION_DAYS = 7


def _purge_trash(max_age_days: int = TRASH_RETENTION_DAYS) -> int:
    """Delete trash entries older than ``max_age_days``. Returns the count purged."""
    if not _TRASH_DIR.is_dir():
        return 0
    cutoff = time.time() - max_age_days * 86400
    purged = 0
    for f in _TRASH_DIR.iterdir():
        try:
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink()
                purged += 1
        except OSError:
            pass
    return purged


# ── output folder naming migration (v0.1.7) ──────────────────────────
# Date folders were DD-MM-YYYY through v0.1.6; ISO sorts chronologically.
_LEGACY_DATE_RE = re.compile(r"^(\d{2})-(\d{2})-(\d{4})$")


def _migrate_output_dirs() -> int:
    """Rename legacy ``DD-MM-YYYY`` date folders under outputs/ to ISO
    ``YYYY-MM-DD``, merging into an existing ISO twin. Returns the count.

    The mirrored thumb-cache folder is renamed too (its keys survive a rename).
    Date-shaped names that aren't real dates are left alone."""
    if not OUTPUTS_DIR.is_dir():
        return 0
    migrated = 0
    for d in sorted(OUTPUTS_DIR.iterdir()):
        m = _LEGACY_DATE_RE.match(d.name)
        if not m or not d.is_dir():
            continue
        dd, mm, yyyy = m.groups()
        try:
            iso = date(int(yyyy), int(mm), int(dd)).isoformat()
        except ValueError:
            continue
        target = OUTPUTS_DIR / iso
        try:
            if target.exists():
                for f in d.iterdir():
                    if not (target / f.name).exists():
                        f.rename(target / f.name)
                d.rmdir()  # raises if a name collision was left behind
            else:
                d.rename(target)
        except OSError as e:
            log.warning("[startup] could not migrate outputs/%s: %s", d.name, e)
            continue
        old_thumbs = _THUMBS_DIR / d.name
        if old_thumbs.is_dir() and not (_THUMBS_DIR / iso).exists():
            try:
                old_thumbs.rename(_THUMBS_DIR / iso)
            except OSError:
                pass
        migrated += 1
    if migrated:
        invalidate_outputs_cache()
    return migrated

class _Cancelled(BaseException):
    """Raised from the progress callback to unwind a running generation.

    A ``BaseException`` so it slips past the engine's ``except Exception``
    guards and only the worker catches it; ``finally`` blocks still run."""


# ── request models ─────────────────────────────────────────────────

class LoadPayload(BaseModel):
    model_type: str = "SD/SDXL"
    checkpoint: Optional[str] = None
    dit: Optional[str] = None
    vae: Optional[str] = None
    te: Optional[str] = None
    clip: Optional[str] = None          # FLUX.1 second text encoder (CLIP-L)
    offload: Optional[str] = None       # full | encoders | stream | none; None = per-family default
    compile: bool = False
    cuda_graphs: bool = False
    channels_last: bool = False
    tf32: bool = False                  # SD/SDXL only (fp32 VAE path; Ampere+)
    fp16_accumulation: bool = False     # fp16-accumulate matmuls; all families
    vae_fp16: bool = False              # non-finite output falls back to fp32
    attention: str = "sdpa"             # "sdpa" | "fa2_turing" (sm75-only; Anima/FLUX)


class DetailerModel(BaseModel):
    """One stacked detailer pass: a detection model + its own optional prompt."""
    model: str = ""
    prompt: str = ""


class GeneratePayload(BaseModel):
    mode: str = "t2i"                 # t2i | i2i | inpaint
    prompt: str = ""
    neg: str = ""
    sampler: str = "dpmpp_2m"
    scheduler: str = "karras"
    steps: int = Field(25, ge=1, le=200)
    cfg: float = Field(6.0, ge=0.0, le=50.0)
    seed: int = Field(-1, ge=-1, le=2**63 - 1)
    width: int = Field(1024, ge=64, le=8192)
    height: int = Field(1024, ge=64, le=8192)
    strength: float = Field(0.6, ge=0.0, le=1.0)
    shift: float = Field(3.0, ge=0.0, le=30.0)
    teacache: float = Field(0.0, ge=0.0, le=1.0)           # rel-L1 threshold (0 = off; Anima only)
    teacache_calibrated: bool = True     # fitted rescale polynomial vs the raw identity path
    teacache_forecast: str = "hermite"   # "hermite" (HiCache) | "taylor" (TaylorSeer)
    teacache_rule: Literal["drift", "easy"] = "drift"   # input drift | EasyCache output change
    deepcache: int = Field(1, ge=1, le=64)                # reuse interval (1 = off; SD/SDXL UNet only)
    input_image: Optional[str] = None   # base64 / data-URL
    mask_image: Optional[str] = None
    preview: bool = True
    blur_check: bool = False             # the page will blur this result: rate it even if gallery blur is off

    # ── detailer (ADetailer-style passes after the main image) ──
    detail_enabled: bool = False
    detail_models: List[DetailerModel] = []
    detail_neg: str = ""
    detail_confidence: float = Field(0.3, ge=0.0, le=1.0)
    detail_strength: float = Field(0.4, ge=0.0, le=1.0)
    detail_dilation: int = Field(4, ge=0, le=128)
    detail_padding: int = Field(32, ge=0, le=512)
    detail_blur: int = Field(4, ge=0, le=64)
    detail_max: int = Field(0, ge=0, le=1000)              # 0 = all detections
    detail_teacache: bool = False        # use the main TeaCache threshold on detailer passes (Anima)

    # ── upscaler (tiled, post-gen) ─────────────────────────────────
    upscale_enabled: bool = False
    upscale_scale: float = Field(2.0, ge=1.0, le=8.0)
    upscale_denoise: float = Field(0.35, ge=0.0, le=1.0)
    upscale_tile: int = Field(1024, ge=128, le=4096)
    upscale_overlap: int = Field(128, ge=0, le=2048)
    upscale_prompt: str = ""
    upscale_teacache: float = Field(0.0, ge=0.0, le=1.0)   # refine-pass TeaCache (0 = off)
    upscale_base: str = ""               # ESRGAN model in models/upscalers/ (blank = Lanczos)

    @model_validator(mode="after")
    def _upscale_overlap_fits(self):
        if self.upscale_enabled and self.upscale_overlap >= self.upscale_tile:
            raise ValueError("upscale_overlap must be < upscale_tile")
        return self


class DetailPayload(BaseModel):
    """Standalone detailer: refine an existing image without re-sampling it."""
    input_image: str = ""
    models: List[DetailerModel] = []
    prompt: str = ""                     # fallback for a pass with no prompt of its own
    neg: str = ""
    confidence: float = Field(0.3, ge=0.0, le=1.0)
    strength: float = Field(0.4, ge=0.0, le=1.0)
    dilation: int = Field(4, ge=0, le=128)
    padding: int = Field(32, ge=0, le=512)
    blur: int = Field(4, ge=0, le=64)
    max_det: int = Field(0, ge=0, le=1000)                 # 0 = all detections
    steps: int = Field(25, ge=1, le=200)
    cfg: float = Field(6.0, ge=0.0, le=50.0)
    sampler: str = "dpmpp_2m"
    scheduler: str = "karras"
    seed: int = Field(-1, ge=-1, le=2**63 - 1)
    teacache: float = Field(0.0, ge=0.0, le=1.0)
    teacache_calibrated: bool = True
    teacache_forecast: str = "hermite"
    teacache_rule: Literal["drift", "easy"] = "drift"
    preview: bool = True
    blur_check: bool = False             # see GeneratePayload.blur_check


class UpscalePayload(BaseModel):
    """Standalone tiled upscale of an existing image."""
    input_image: str = ""
    scale: float = Field(2.0, ge=1.0, le=8.0)
    tile: int = Field(1024, ge=128, le=4096)
    overlap: int = Field(128, ge=0, le=2048)
    denoise: float = Field(0.35, ge=0.0, le=1.0)
    base: str = ""                       # ESRGAN model in models/upscalers/ (blank = Lanczos)
    prompt: str = ""
    neg: str = ""
    steps: int = Field(25, ge=1, le=200)
    cfg: float = Field(6.0, ge=0.0, le=50.0)
    sampler: str = "dpmpp_2m"
    scheduler: str = "karras"
    seed: int = Field(-1, ge=-1, le=2**63 - 1)
    teacache: float = Field(0.0, ge=0.0, le=1.0)
    teacache_calibrated: bool = True
    teacache_forecast: str = "hermite"
    teacache_rule: Literal["drift", "easy"] = "drift"
    preview: bool = True
    blur_check: bool = False             # see GeneratePayload.blur_check

    @model_validator(mode="after")
    def _overlap_fits(self):
        # overlap == tile divides by zero in tile_starts; overlap > tile yields
        # an empty grid that blends to black.
        if self.overlap >= self.tile:
            raise ValueError("overlap must be < tile")
        return self


class CalibratePayload(BaseModel):
    prompt: str = ""
    neg: str = ""
    steps: int = Field(12, ge=1, le=200)
    cfg: float = Field(4.0, ge=0.0, le=50.0)
    seed: int = Field(0, ge=0, le=2**63 - 1)
    width: int = Field(1024, ge=64, le=8192)
    height: int = Field(1024, ge=64, le=8192)
    shift: float = Field(3.0, ge=0.0, le=30.0)
    grid: int = Field(80, ge=1, le=500)    # teacher-trajectory candidate count (K)


class GenDefaults(BaseModel):
    """The Generate form's reusable params, seeded on load."""
    sampler: str = "dpmpp_2m"
    scheduler: str = "karras"
    steps: int = 25
    cfg: float = 6.0
    width: int = 1024
    height: int = 1024
    shift: float = 3.0
    # None = leave the form's own value.
    prompt: Optional[str] = None
    neg: Optional[str] = None


class Settings(BaseModel):
    """Persisted global settings (the settings panel). Defaults mirror the
    submodule's; applied at generation time by ``_settings_knobs``."""
    gate_reduce: Literal["all", "per_channel"] = "all"
    # Anima-only sampler/scheduler knobs.
    curvature: float = 0.25       # secant / secant_anneal x0 extrapolation strength
    eta_max: float = 1.0          # ancestral noise of the anneal / cogent samplers
    beta_alpha: float = 0.6       # beta scheduler: low-t (σ→0) density
    beta_beta: float = 0.6        # beta scheduler: high-t (σ→1) density
    lq_threshold: float = 0.025   # linear_quadratic knee
    # CFG guidance interval (Kynkäänniemi et al., 2024): the uncond forward is
    # skipped outside this fraction of the run. (0, 1) = off. Not FLUX.
    cfg_interval_start: float = Field(0.0, ge=0.0, lt=1.0)
    cfg_interval_end: float = Field(1.0, gt=0.0, le=1.0)
    # TeaCache threshold multiplier for the uncond stream only (Anima). 1.0 = off.
    teacache_uncond_scale: float = Field(1.0, ge=1.0, le=4.0)

    @model_validator(mode="after")
    def _cfg_interval_ordered(self):
        if self.cfg_interval_start >= self.cfg_interval_end:
            raise ValueError("cfg_interval_start must be < cfg_interval_end")
        return self
    # "auto" tiles VAE decode only when it won't fit free VRAM (FLUX always tiles).
    vae_tiling: str = "auto"      # "auto" | "always"
    metadata_format: str = "a1111"   # "a1111" | "swarmui"
    gen_defaults: Optional[GenDefaults] = None
    # Display-only; a click reveals the image.
    nsfw_blur: bool = True
    # Lowest rating that gets blurred. No "XXX": the AI rater never outputs it.
    blur_min_rating: Literal["PG13", "R", "X"] = "R"


class XYZPayload(BaseModel):
    prompt: str = ""
    neg: str = ""
    width: int = Field(1024, ge=64, le=8192)
    height: int = Field(1024, ge=64, le=8192)
    steps: int = Field(25, ge=1, le=200)
    cfg: float = Field(6.0, ge=0.0, le=50.0)
    sampler: str = "dpmpp_2m"
    scheduler: str = "karras"
    seed: int = Field(-1, ge=-1, le=2**63 - 1)
    shift: float = Field(3.0, ge=0.0, le=30.0)
    teacache: float = Field(0.0, ge=0.0, le=1.0)
    teacache_calibrated: bool = True
    teacache_forecast: str = "hermite"
    teacache_rule: Literal["drift", "easy"] = "drift"
    x_type: str = "None"
    x_vals: str = ""
    y_type: str = "None"
    y_vals: str = ""
    z_type: str = "None"
    z_vals: str = ""
    preview: bool = True
    blur_check: bool = False             # see GeneratePayload.blur_check


class CancelPayload(BaseModel):
    job: Optional[int] = None        # None = cancel whatever is currently running


class ParseTextPayload(BaseModel):
    text: str = ""


# ── helpers ─────────────────────────────────────────────────────────

def _decode_image(data: str) -> Image.Image:
    """Decode a base64 / data-URL image to RGB. Alpha is composited onto white
    (``convert("RGB")`` alone would turn transparent regions black)."""
    if data.strip().startswith("data:") and "," in data:
        data = data.split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(data)))
    if img.mode in ("RGBA", "LA", "PA") or (
        "A" in img.getbands() and img.mode not in ("RGB", "L", "P")
    ):
        log.warning("input image had an alpha channel; composited onto white "
                    "(transparency is not preserved), mode=%s", img.mode)
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg.convert("RGB")
    return img.convert("RGB")


def _decode_mask(data: str) -> Image.Image:
    """Decode a base64 / data-URL mask to ``L``. An alpha channel, when present,
    is the mask (``convert("L")`` would read luminance and ignore it)."""
    if data.strip().startswith("data:") and "," in data:
        data = data.split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(data)))
    if img.mode in ("RGBA", "LA", "PA") or "A" in img.getbands():
        return img.split()[-1]
    return img.convert("L")


def _output_url(path: Path) -> str:
    return f"/outputs/{path.relative_to(OUTPUTS_DIR).as_posix()}"


def _save_output(image: Image.Image, gen_kwargs: dict,
                 detailer: Optional[dict] = None,
                 upscale: Optional[dict] = None,
                 seed: Optional[int] = None,
                 blur_check: bool = False) -> Path:
    """Save an image to outputs/ with generation metadata; return its path.
    ``blur_check`` carries the page's "Blur NSFW" toggle to the auto-rating."""
    seed = ENGINE.last_seed if seed is None else seed
    out = next_output_path(seed)
    meta = PngInfo()
    meta_kwargs = {k: v for k, v in gen_kwargs.items() if k != "progress_callback"}
    formatter = (md.format_swarmui_metadata if SETTINGS.get("metadata_format") == "swarmui"
                 else md.format_metadata)
    params = formatter(meta_kwargs, ENGINE, detailer=detailer,
                       upscale=upscale, seed=seed)
    meta.add_text("parameters", params)
    image.save(out, pnginfo=meta)
    # Index the new image for search now (same string a rebuild reads back), and
    # drop the listing cache so /api/gallery sees it.
    _gallery_index_add(out, params)
    invalidate_outputs_cache()
    _maybe_auto_tag(out, blur_check)
    return out


# ── base-image cache ────────────────────────────────────────────────
# The upscaler and detailer don't change the base image, so re-running with only
# those re-tuned reuses the last base. One slot, touched only by the worker
# thread, so no lock.
_BASE_CACHE: "dict[str, tuple[Image.Image, str]]" = {}


def _fingerprint_value(v):
    """JSON-safe stand-in for one generation kwarg; images hash by pixels."""
    if isinstance(v, Image.Image):
        return ["image", v.mode, v.size, hashlib.sha256(v.tobytes()).hexdigest()]
    return v


def _base_fingerprint(gen_kwargs: dict, mode: str,
                      loras: "list[tuple[str, float]]") -> str:
    """Identity of the base image ``gen_kwargs`` would sample, including the
    loaded weights and the ``<lora:…>`` tags (the engine only sees the stripped
    prompt)."""
    parts = {
        "mode": mode,
        "epoch": ENGINE.weights_epoch,
        "loras": sorted((str(n), float(m)) for n, m in loras),
        "kwargs": {k: _fingerprint_value(v) for k, v in sorted(gen_kwargs.items())
                   if k not in ("progress_callback", "preview_callback")},
    }
    return hashlib.sha256(
        json.dumps(parts, sort_keys=True, default=str).encode()
    ).hexdigest()


# ── generation ──────────────────────────────────────────────────────

def _settings_knobs(sampler: str, scheduler: str, teacache: float) -> dict:
    """Settings-panel engine kwargs for this sampler/scheduler. Shared by plain
    generations and every X/Y/Z cell so both sample the same thing."""
    knobs: dict = {}
    if sampler in ("cogent", "cogent3", "cogent3_pump", "cogent3_pump_rate"):
        knobs["gate_reduce"] = SETTINGS["gate_reduce"]

    # Anima only. Inject just the knobs the sampler/scheduler consumes, so the
    # metadata carries no unused keys.
    if ENGINE.loaded_family == "anima":
        if sampler in ("secant", "secant_anneal"):
            knobs["curvature"] = float(SETTINGS["curvature"])
        if sampler in ("secant_anneal", "euler_ancestral_anneal", "dpmpp_2m_anneal", "cogent", "cogent3", "cogent3_pump",
                       "cogent3_pump_rate"):
            knobs["eta_max"] = float(SETTINGS["eta_max"])
        # uni_pc_anneal keeps its own baked-in eta_max (0.2).
        if scheduler == "beta":
            knobs["beta_alpha"] = float(SETTINGS["beta_alpha"])
            knobs["beta_beta"] = float(SETTINGS["beta_beta"])
        if scheduler == "linear_quadratic":
            knobs["lq_threshold"] = float(SETTINGS["lq_threshold"])

    # FLUX is guidance-distilled and has no CFG pass. Non-default values only.
    if ENGINE.loaded_family not in ("flux1", "flux2"):
        ivl_start = float(SETTINGS["cfg_interval_start"])
        ivl_end = float(SETTINGS["cfg_interval_end"])
        if (ivl_start, ivl_end) != (0.0, 1.0) and ivl_start < ivl_end:
            knobs["cfg_interval_start"] = ivl_start
            knobs["cfg_interval_end"] = ivl_end

    uncond_scale = float(SETTINGS["teacache_uncond_scale"])
    if uncond_scale != 1.0 and teacache > 0:
        knobs["teacache_uncond_scale"] = uncond_scale

    return knobs


def _run_generation(p: GeneratePayload, on_progress: Callable[[int, int], None],
                    on_preview: Optional[Callable] = None) -> dict:
    if not ENGINE.loaded_name:
        raise RuntimeError("Load a model first")

    # Extensions may mutate the payload in place.
    EXTENSIONS.run_hook("pre_generate", payload=p)

    clean_prompt, prompt_loras = ENGINE.parse_lora_prompt(p.prompt)
    clean_neg, neg_loras = ENGINE.parse_lora_prompt(p.neg)
    loras = prompt_loras + neg_loras

    try:
        lora_info = ""
        if loras:
            lora_info = ENGINE.apply_temp_loras(loras) + "  |  "

        common = dict(
            negative_prompt=clean_neg, steps=int(p.steps), cfg_scale=float(p.cfg),
            sampler=p.sampler, scheduler=p.scheduler, seed=int(p.seed),
            teacache_thresh=float(p.teacache),
            teacache_use_coeffs=bool(p.teacache_calibrated),
            teacache_forecast=p.teacache_forecast,
            teacache_rule=p.teacache_rule,
            deepcache_interval=int(p.deepcache),
            progress_callback=on_progress,
            preview_callback=on_preview if p.preview else None,
        )
        common.update(_settings_knobs(p.sampler, p.scheduler, p.teacache))

        if p.mode == "i2i":
            if not p.input_image:
                raise RuntimeError("Provide an input image")
            gen_kwargs = dict(
                prompt=clean_prompt, input_image=_decode_image(p.input_image),
                width=int(p.width), height=int(p.height),
                strength=float(p.strength), **common,
            )
            gen_fn = ENGINE.generate_i2i
        elif p.mode == "inpaint":
            if not p.input_image or not p.mask_image:
                raise RuntimeError("Provide both an input image and a mask")
            gen_kwargs = dict(
                prompt=clean_prompt, input_image=_decode_image(p.input_image),
                mask_image=_decode_mask(p.mask_image),
                width=int(p.width), height=int(p.height),
                strength=float(p.strength), **common,
            )
            gen_fn = ENGINE.generate_inpaint
        else:  # t2i
            gen_kwargs = dict(
                prompt=clean_prompt, width=int(p.width), height=int(p.height),
                shift=float(p.shift), **common,
            )
            gen_fn = ENGINE.generate_t2i

        t0 = time.perf_counter()
        # Reuse the last base when only the post passes changed. Seed -1 asks for
        # a new image, so it never reads the cache but still writes one under the
        # resolved seed.
        fp = _base_fingerprint(gen_kwargs, p.mode, loras) if p.seed != -1 else None
        cached = _BASE_CACHE.get(fp) if fp else None
        # Copies in and out: a post_generate extension may draw on its image.
        if cached is not None:
            base, info = cached
            image, seed = base.copy(), int(p.seed)
        else:
            image, info = gen_fn(**gen_kwargs)
            seed = ENGINE.last_seed
            if fp is None and seed >= 0:
                fp = _base_fingerprint({**gen_kwargs, "seed": seed}, p.mode, loras)
            if fp is not None:
                _BASE_CACHE.clear()
                _BASE_CACHE[fp] = (image.copy(), info)

        # Upscale first so the detailer refines at the final resolution.
        upscale_info = ""
        upscaled = False
        if p.upscale_enabled and float(p.upscale_scale) > 1.0:
            try:
                image, unote = ENGINE.upscale(
                    image,
                    scale=float(p.upscale_scale), tile=int(p.upscale_tile),
                    overlap=int(p.upscale_overlap), denoise=float(p.upscale_denoise),
                    base_upscaler=p.upscale_base,
                    prompt=p.upscale_prompt.strip() or clean_prompt,
                    negative_prompt=clean_neg,
                    steps=int(p.steps), cfg_scale=float(p.cfg),
                    sampler=p.sampler, scheduler=p.scheduler,
                    gate_reduce=SETTINGS["gate_reduce"],
                    seed=int(p.seed),
                    teacache_thresh=float(p.upscale_teacache),
                    teacache_use_coeffs=bool(p.teacache_calibrated),
                    teacache_forecast=p.teacache_forecast,
                    teacache_rule=p.teacache_rule,
                    progress_callback=on_progress,
                    preview_callback=on_preview if p.preview else None,
                )
                upscale_info = "  |  " + unote
                upscaled = True
            except Exception as e:  # noqa: BLE001
                # Keep the base image, and leave out the upscale metadata so a
                # swallowed OOM can't pass for a successful upscale.
                upscale_info = f"  |  ⚠ UPSCALE FAILED, saved the un-upscaled base image ({e})"

        # Stacked detection models run in sequence, each on the previous result.
        detail_info = ""
        active = [dm for dm in p.detail_models
                  if dm.model and not dm.model.startswith("(")] if p.detail_enabled else []
        applied = []  # models that actually refined the image (drives metadata)
        if active and not ENGINE.can_inpaint:
            detail_info = "  |  detailer skipped (no inpaint for this model)"
        elif active:
            notes = []
            for dm in active:
                try:
                    image, dnote = ENGINE.detail(
                        image,
                        detector_path=str(detector_path(dm.model)),
                        prompt=dm.prompt.strip() or clean_prompt,
                        negative_prompt=p.detail_neg.strip() or clean_neg,
                        confidence=float(p.detail_confidence),
                        strength=float(p.detail_strength),
                        steps=int(p.steps), cfg_scale=float(p.cfg),
                        sampler=p.sampler, scheduler=p.scheduler,
                        gate_reduce=SETTINGS["gate_reduce"],
                        dilation=int(p.detail_dilation), padding=int(p.detail_padding),
                        blur=int(p.detail_blur), max_det=int(p.detail_max),
                        seed=int(p.seed),
                        teacache_thresh=float(p.teacache) if p.detail_teacache else 0.0,
                        teacache_use_coeffs=bool(p.teacache_calibrated),
                        teacache_forecast=p.teacache_forecast,
                        teacache_rule=p.teacache_rule,
                        progress_callback=on_progress,
                        preview_callback=on_preview if p.preview else None,
                    )
                    notes.append(f"{dm.model}: {dnote.replace('Detailer: ', '')}")
                    applied.append(dm)
                except Exception as e:  # noqa: BLE001
                    # Keep the image and leave this model out of the metadata.
                    notes.append(f"⚠ {dm.model} FAILED: {e}")
            detail_info = "  |  detailer [" + "; ".join(notes) + "]"

        # Everything but the disk save.
        elapsed = time.perf_counter() - t0

        # Save the raw prompt/neg so <lora:…> tags round-trip through metadata.
        gen_kwargs["prompt"], gen_kwargs["negative_prompt"] = p.prompt, p.neg
        detailer_meta = {
            "models": [{"model": dm.model, "prompt": dm.prompt} for dm in applied],
            "neg": p.detail_neg,
            "confidence": p.detail_confidence,
            "strength": p.detail_strength,
            "dilation": p.detail_dilation,
            "padding": p.detail_padding,
            "blur": p.detail_blur,
            "maxDet": p.detail_max,
        } if applied else None
        upscale_meta = {
            "scale": float(p.upscale_scale),
            "tile": int(p.upscale_tile),
            "overlap": int(p.upscale_overlap),
            "denoise": float(p.upscale_denoise),
            "teacache": float(p.upscale_teacache),
            "base": p.upscale_base or "Lanczos",
            "prompt": p.upscale_prompt.strip() or "",
        } if upscaled else None
        gctx = EXTENSIONS.run_hook(
            "post_generate", payload=p, image=image, info=info,
        )
        image = gctx.image

        out = _save_output(image, gen_kwargs, detailer=detailer_meta,
                           upscale=upscale_meta, seed=seed, blur_check=p.blur_check)
        rel = out.relative_to(OUTPUTS_DIR)
        EXTENSIONS.run_hook("post_save", payload=p, image=image, path=out)
        return {
            "image_url": _output_url(out),
            # Instant prompt verdict; the background AI rating arrives later as
            # a "rated" SSE event.
            "path": rel.as_posix(),
            "nsfw_prompt": md.prompt_is_nsfw(clean_prompt),
            "prompt_rating": md.prompt_rating(clean_prompt),
            "info": f"{lora_info}{info}  |  inference: {elapsed:.2f}s{upscale_info}{detail_info}  |  saved to {rel}",
            "seed": seed,
        }
    finally:
        if loras:
            ENGINE.clear_temp_loras()


def _run_xyz(p: XYZPayload, on_progress: Callable[..., None],
             on_preview: Optional[Callable] = None) -> dict:
    if not ENGINE.loaded_name:
        raise RuntimeError("Load a model first")
    base_kwargs = dict(
        prompt=p.prompt, negative_prompt=p.neg,
        width=int(p.width), height=int(p.height),
        steps=int(p.steps), cfg_scale=float(p.cfg),
        sampler=p.sampler, scheduler=p.scheduler,
        seed=int(p.seed), shift=float(p.shift),
        teacache_thresh=float(p.teacache),
        teacache_use_coeffs=bool(p.teacache_calibrated),
        teacache_forecast=p.teacache_forecast,
        teacache_rule=p.teacache_rule,
    )
    # A Checkpoint axis leaves the last swept model loaded; reload the user's.
    swaps_model = "Checkpoint" in (p.x_type, p.y_type, p.z_type)
    try:
        grids, info = generate_xyz_grid(
            base_kwargs,
            p.x_type, p.x_vals, p.y_type, p.y_vals, p.z_type, p.z_vals,
            progress_callback=on_progress,
            preview_callback=on_preview if p.preview else None,
            save_callback=partial(_save_output, blur_check=p.blur_check),
            cell_knobs=lambda sampler, scheduler: _settings_knobs(
                sampler, scheduler, p.teacache),
        )
        # generate_xyz_grid mutated base_kwargs in place (prompt cleaned, seed
        # resolved), so it now carries the grid's metadata params.
        grid_kwargs = {**base_kwargs,
                       **_settings_knobs(p.sampler, p.scheduler, p.teacache)}
        urls = []
        for grid in grids:
            out = _save_output(grid, grid_kwargs, blur_check=p.blur_check)
            urls.append(_output_url(out))
        return {
            "grids": urls, "info": info,
            "path": out.relative_to(OUTPUTS_DIR).as_posix(),
            "nsfw_prompt": md.prompt_is_nsfw(p.prompt),
            "prompt_rating": md.prompt_rating(p.prompt),
        }
    finally:
        if swaps_model and LAST_LOAD_FORM:
            try:
                _do_load(LoadPayload(**LAST_LOAD_FORM))
            except Exception:  # noqa: BLE001  keep the grid result; report real state
                pass
            _push({"type": "status", **_state_payload()})


def _run_calibrate(p: CalibratePayload, on_progress: Callable[[int, int], None]) -> dict:
    if not ENGINE.loaded_name:
        raise RuntimeError("Load a model first")
    info = ENGINE.calibrate_oss(
        prompt=p.prompt, negative_prompt=p.neg, steps=int(p.steps),
        width=int(p.width), height=int(p.height), shift=float(p.shift),
        cfg_scale=float(p.cfg), seed=int(p.seed), grid=int(p.grid),
        progress_callback=on_progress,
    )
    return {"info": info}


def _run_calibrate_teacache(p: CalibratePayload, on_progress: Callable[[int, int], None]) -> dict:
    if not ENGINE.loaded_name:
        raise RuntimeError("Load a model first")
    info = ENGINE.calibrate_teacache(
        prompt=p.prompt, negative_prompt=p.neg, steps=int(p.steps),
        width=int(p.width), height=int(p.height), shift=float(p.shift),
        cfg_scale=float(p.cfg), seed=int(p.seed),
        progress_callback=on_progress,
    )
    return {"info": info}


# ── job queue + SSE broadcast ───────────────────────────────────────
# One worker thread runs jobs one at a time, so the thread itself is the
# serialization. Every device shares one SSE stream.

_job_ids = itertools.count(1)


class Job:
    def __init__(self, kind: str, label: str, run: Callable[["Job"], dict],
                 *, priority: int = 0):
        self.id = next(_job_ids)
        self.kind = kind            # generate | xyz | calibrate | load | install | update | tag
        self.label = label
        self.run = run              # run(job) -> result dict; may raise _Cancelled
        self.status = "queued"      # queued | running | done | error | cancelled
        self.cancel = threading.Event()
        self.step = 0               # for snapshots on (re)connect
        self.total = 0
        self.priority = int(priority)  # higher runs sooner
        self.last_preview: Optional[Image.Image] = None  # saved as a partial on shutdown


QUEUE: "deque[Job]" = deque()
QUEUE_LOCK = threading.Lock()
QUEUE_WAKE = threading.Event()
CURRENT: Optional[Job] = None

# One capped asyncio.Queue per SSE client; on overflow the oldest event drops.
SSE_QUEUE_MAX = 256
SUBSCRIBERS: "set[asyncio.Queue]" = set()
APP_LOOP: Optional[asyncio.AbstractEventLoop] = None

# Previews are broadcast to every client, so cap their rate and size and send
# lossy WebP; the saved result is always full quality.
PREVIEW_MIN_INTERVAL = 0.2   # seconds between streamed previews
PREVIEW_MAX_SIDE = 512       # downscale to this long side before encoding

# The long-lived SSE streams would stall uvicorn's graceful shutdown (Ctrl+C
# hangs until a second one). uvicorn calls Server.handle_exit as soon as a
# signal arrives, so wrap it to wake every stream with a sentinel.
SHUTDOWN = asyncio.Event()


def _wake_for_shutdown() -> None:
    SHUTDOWN.set()
    for q in list(SUBSCRIBERS):
        _force_put(q, None)


def _force_put(q: "asyncio.Queue", ev) -> None:
    """Put on a capped queue, popping the oldest until it fits."""
    while True:
        try:
            q.put_nowait(ev)
            return
        except asyncio.QueueFull:
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                return


_uvicorn_handle_exit = uvicorn.Server.handle_exit


def _handle_exit(self, sig, frame):
    if APP_LOOP is not None:
        APP_LOOP.call_soon_threadsafe(_wake_for_shutdown)
    _uvicorn_handle_exit(self, sig, frame)


uvicorn.Server.handle_exit = _handle_exit

# The last successful /api/load payload, persisted so a new device or a restart
# can restore the load form. The model itself is not reloaded.
_LAST_LOAD_PATH = _ROOT / "last_load.json"


def _read_last_load() -> Optional[dict]:
    try:
        return json.loads(_LAST_LOAD_PATH.read_text())
    except (OSError, ValueError):
        return None


def _write_last_load(form: dict) -> None:
    _atomic_write_text(_LAST_LOAD_PATH, json.dumps(form))


LAST_LOAD_FORM: Optional[dict] = _read_last_load()

# Round-tripped through Settings so missing keys get defaults and unknown keys
# are dropped.
_SETTINGS_PATH = _ROOT / "settings.json"


def _read_settings() -> dict:
    try:
        return Settings(**json.loads(_SETTINGS_PATH.read_text())).model_dump()
    except (OSError, ValueError, TypeError):
        return Settings().model_dump()


def _write_settings(s: Settings) -> None:
    _atomic_write_text(_SETTINGS_PATH, json.dumps(s.model_dump()))


SETTINGS: dict = _read_settings()


def _push(ev: dict) -> None:
    """Fan one event out to every SSE client. Safe to call from any thread."""
    loop = APP_LOOP
    if loop is None:
        return
    def deliver():
        for q in list(SUBSCRIBERS):
            _force_put(q, ev)
    loop.call_soon_threadsafe(deliver)


def _state_payload() -> dict:
    """Model-load state shared on connect and after every load."""
    return {
        "status": ENGINE.status_text(),
        "loaded": bool(ENGINE.loaded_name),
        "load_form": LAST_LOAD_FORM,
        "last_seed": ENGINE.last_seed,
    }


def _queue_list() -> list:
    with QUEUE_LOCK:
        jobs = ([CURRENT] if CURRENT else []) + list(QUEUE)
        return [{"id": j.id, "kind": j.kind, "label": j.label, "status": j.status}
                for j in jobs]


def _broadcast_queue() -> None:
    _push({"type": "queue", "jobs": _queue_list(),
           "running": CURRENT.id if CURRENT else None})


def _make_callbacks(job: Job):
    def on_progress(step, total, cell=None, cells=None):
        if job.cancel.is_set():
            raise _Cancelled
        job.step, job.total = int(step), int(total)
        ev = {"type": "progress", "job": job.id, "step": int(step), "total": int(total)}
        if cells is not None:  # X/Y/Z: 1-based current cell
            ev["cell"], ev["cells"] = int(cell), int(cells)
        _push(ev)

    preview_last = [0.0]   # monotonic time of the last emitted preview

    def on_preview(image):
        now = time.monotonic()
        if now - preview_last[0] < PREVIEW_MIN_INTERVAL:
            return
        preview_last[0] = now
        if max(image.size) > PREVIEW_MAX_SIDE:
            thumb = image.copy()
            thumb.thumbnail((PREVIEW_MAX_SIDE, PREVIEW_MAX_SIDE))
        else:
            thumb = image
        # Kept so a shutdown mid-job can still save a partial result.
        job.last_preview = thumb
        buf = io.BytesIO()
        thumb.save(buf, format="WEBP", quality=80)
        data = "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()
        _push({"type": "preview", "job": job.id, "image": data})

    return on_progress, on_preview


def _enqueue(job: Job) -> None:
    # Insert before the first lower-priority job (FIFO within a priority), so a
    # load (priority 10) runs next. The running job still finishes.
    with QUEUE_LOCK:
        idx = len(QUEUE)
        for i, j in enumerate(QUEUE):
            if job.priority > j.priority:
                idx = i
                break
        QUEUE.insert(idx, job)
    QUEUE_WAKE.set()
    _broadcast_queue()


# ── AI NSFW rating jobs (kind "tag") ────────────────────────────────
# The WD tagger runs on the shared worker so it never races the GPU. Lowest
# priority: a rating always waits behind a generation.

def _tag_job_run(job: Job, paths: List[Path], notify: bool = False) -> dict:
    """Rate ``paths``, cache the results, and patch the gallery index.

    ``notify`` broadcasts each verdict as a ``rated`` event. Full-gallery scans
    leave it off; nobody is looking at those images.
    """
    tagger_mod.TAGGER.load()
    total = max(1, job.total or len(paths))
    rated = 0

    def _flush(entries: dict) -> None:
        _store_ratings(entries)
        _gallery_index_patch_ratings(entries)
        if notify:
            for rel, e in entries.items():
                _push({"type": "rated", "path": rel,
                       "rating": e["rating"], "nsfw": e["nsfw"]})

    entries: dict = {}
    for i, p in enumerate(paths):
        if job.cancel.is_set():
            # Keep what's already rated; a full scan takes minutes.
            if entries:
                _flush(entries)
            raise _Cancelled
        r = tagger_mod.TAGGER.rate_one(p)
        if r is not None:
            entries[p.relative_to(OUTPUTS_DIR).as_posix()] = {
                "rating": r["rating"], "nsfw": r["nsfw"],
                "conf": round(r["confidence"], 4), "key": _image_key(p),
            }
        job.step = i + 1
        if i % 8 == 0:  # throttle progress events
            _push({"type": "progress", "job": job.id, "step": job.step, "total": total})
        if len(entries) >= 64:  # checkpoint so a crash keeps progress
            _flush(entries)
            rated += len(entries)
            entries = {}
    rated += len(entries)
    if entries:
        _flush(entries)
    return {"rated": rated}


def _enqueue_tag_job(paths: List[Path], label: str,
                     notify: bool = False) -> Optional[int]:
    """Queue a background rating for ``paths``. None when there's nothing to do
    or the tagger isn't installed."""
    if not paths or not tagger_mod.timm_available():
        return None
    job = Job("tag", label, lambda j: _tag_job_run(j, paths, notify=notify),
              priority=-10)
    job.total = len(paths)
    _enqueue(job)
    return job.id


# Outputs saved since the last flush, rated as one job instead of one queue row
# per image.
_PENDING_TAG: List[Path] = []
_PENDING_TAG_LOCK = threading.Lock()


def _maybe_auto_tag(path: Path, blur_check: bool = False) -> None:
    """Queue a freshly saved image for background rating when the gallery blur
    or the page's own "Blur NSFW" toggle (``blur_check``) wants a verdict."""
    if not (SETTINGS.get("nsfw_blur") or blur_check) or not tagger_mod.timm_available():
        return
    with _PENDING_TAG_LOCK:
        _PENDING_TAG.append(path)


def _flush_pending_tags() -> None:
    """Enqueue one rating job for everything saved since the last flush, once no
    other work is queued (a batch is N generate jobs; this makes one tag job)."""
    with QUEUE_LOCK:
        if any(j.kind != "tag" for j in QUEUE):
            return
    with _PENDING_TAG_LOCK:
        paths = _PENDING_TAG[:]
        _PENDING_TAG.clear()
    if not paths:
        return
    label = (f"Rate NSFW: {paths[0].name}" if len(paths) == 1
             else f"Rate NSFW ({len(paths)} images)")
    _enqueue_tag_job(paths, label, notify=True)


# ── extension platform ──────────────────────────────────────────────
# The loader gets callables into the queue and SSE stream so extensions don't
# import server.py.

def _ext_enqueue_job(ext_name: str, label: str, run: Callable, kind: str = "ext") -> int:
    job = Job(f"{kind}:{ext_name}", label, run)
    _enqueue(job)
    return job.id


EXTENSIONS = ExtensionLoader(
    ENGINE,
    enqueue_job=_ext_enqueue_job,
    broadcast=_push,
)


def _worker() -> None:
    global CURRENT
    while True:
        with QUEUE_LOCK:
            CURRENT = QUEUE.popleft() if QUEUE else None
        job = CURRENT
        if job is None:
            QUEUE_WAKE.wait()
            QUEUE_WAKE.clear()
            continue
        if job.cancel.is_set():     # cancelled while queued
            job.status = "cancelled"
            with QUEUE_LOCK:
                CURRENT = None
            _push({"type": "cancelled", "job": job.id})
            _broadcast_queue()
            continue
        job.status = "running"
        _broadcast_queue()
        # The tagger stays resident across jobs, except when the loaded model
        # streams its backbone: there its ~600 MB would OOM the next generation.
        # Read the active mode, not recommended_offload(); FLUX streams on any card.
        if (job.kind != "tag" and tagger_mod.TAGGER.loaded
                and ENGINE.active_offload == "stream"):
            tagger_mod.TAGGER.unload()
        try:
            result = job.run(job)
            job.status = "done"
            _push({"type": "done", "job": job.id, **result})
        except _Cancelled:
            job.status = "cancelled"
            _push({"type": "cancelled", "job": job.id})
        except Exception as e:  # noqa: BLE001
            job.status = "error"
            _push({"type": "error", "job": job.id, "message": _friendly_error(e)})
        except BaseException as e:  # noqa: BLE001
            # The only worker thread: anything escaping would kill it and leave
            # every later job queued forever.
            job.status = "error"
            log.exception("worker job %s raised %s", job.id, type(e).__name__)
            _push({"type": "error", "job": job.id,
                   "message": f"job failed: {type(e).__name__}"})
        finally:
            with QUEUE_LOCK:
                CURRENT = None
            try:
                _flush_pending_tags()
            except Exception as e:  # noqa: BLE001
                log.warning("could not queue background rating: %s", e)
            _broadcast_queue()


def _friendly_error(e: Exception) -> str:
    """Map an engine exception to a user-actionable message. CUDA OOM frees the
    cache and gets a hint appended."""
    try:
        import torch  # heavy and optional for the tests
        if isinstance(e, torch.cuda.OutOfMemoryError):
            try: torch.cuda.empty_cache()
            except Exception: pass  # noqa: BLE001
            return (f"{e}  →  out of VRAM. Try offload=stream (Settings or the "
                    f"Load panel), a smaller width/height, fewer steps, or a "
                    f"smaller detailer/upscale tile.")
    except Exception:  # noqa: BLE001  torch not available
        pass
    return str(e)


# ── shutdown: save a partial result for the in-flight job ─────────────
# The worker is a daemon thread, so Ctrl+C mid-sampling loses the run. atexit
# runs while daemon threads are alive, so the last streamed preview is still
# readable and can be written out.

def _save_partial_preview(job: Optional[Job]) -> Optional[Path]:
    """Write ``job.last_preview`` to outputs/, flagged as a shutdown partial in
    its ``parameters`` line. Returns the path, or ``None`` if there's nothing."""
    if job is None or job.last_preview is None:
        return None
    try:
        out = next_output_path(ENGINE.last_seed)
        meta = PngInfo()
        info = md.format_metadata(
            {"prompt": "", "sampler": "", "scheduler": "", "steps": 0,
             "cfg_scale": 0.0, "seed": ENGINE.last_seed},
            ENGINE,
        )
        meta.add_text("parameters", f"PARTIAL: interrupted by shutdown. {info}")
        job.last_preview.save(out, pnginfo=meta)
        _invalidate_gallery_index()
        invalidate_outputs_cache()
        try:
            rel = str(out.relative_to(OUTPUTS_DIR))
        except ValueError:
            rel = str(out)
        log.warning("shutdown: saved partial preview for job %s to %s",
                    job.id, rel)
        return out
    except Exception as e:  # noqa: BLE001  never let the exit path raise
        log.warning("shutdown: could not save partial preview: %s", e)
        return None


def _on_shutdown_save_partial() -> None:
    _save_partial_preview(CURRENT)


atexit.register(_on_shutdown_save_partial)


# ── app ─────────────────────────────────────────────────────────────

app = FastAPI(title="Diffucore")
log.info("[startup] offload default '%s' (device: %s)",
         ENGINE.recommended_offload(), ENGINE.device)


# ── request guard: body-size cap, CSRF/Origin check, auth gate ──────
# Cheap header checks first (Content-Length, Origin), then the auth gate.
_PUBLIC_AUTH = {("GET", "/"), ("POST", "/api/auth/login"),
                ("GET", "/api/auth/status"), ("POST", "/api/auth/logout")}


@app.middleware("http")
async def _request_guard(request: Request, call_next):
    # 1. Body-size backstop. /api/metadata/parse checks its tighter cap itself.
    cl = request.headers.get("content-length")
    if cl:
        try:
            if int(cl) > MAX_BODY_BYTES:
                return JSONResponse({"error": "request body too large"},
                                    status_code=413)
        except ValueError:
            pass
    elif request.method in _STATE_CHANGE_METHODS:
        # No Content-Length means chunked transfer, which would skip the cap
        # while uvicorn still buffers the whole body. Nothing here streams
        # uploads, so require a declared length.
        if "chunked" in (request.headers.get("transfer-encoding") or "").lower():
            return JSONResponse(
                {"error": "chunked request bodies are not accepted; "
                          "send a Content-Length"},
                status_code=411)
    # 2. Block cross-origin state-changing requests (CSRF). curl sends no Origin.
    if not origin_ok(request):
        return JSONResponse({"error": "cross-origin request blocked"},
                            status_code=403)
    # 3. Auth gate. Public paths stay open so the login page is reachable.
    if AUTH.enabled and (request.method, request.url.path) not in _PUBLIC_AUTH:
        denied = AUTH.gate_response(request)
        if denied is not None:
            return denied
    return await call_next(request)


@app.on_event("startup")
async def _startup():
    global APP_LOOP
    APP_LOOP = asyncio.get_running_loop()
    threading.Thread(target=_worker, daemon=True).start()
    # A rating-decision upgrade invalidates every cached verdict; re-rate now.
    try:
        _maybe_auto_rescan()
    except Exception as e:  # noqa: BLE001
        log.warning("[startup] gallery re-rate not enqueued: %s", e)
    # At startup rather than import, so the app object exists.
    EXTENSIONS.load_all()
    EXTENSIONS.mount_into(app)
    n = sum(1 for e in EXTENSIONS.extensions.values() if e.module is not None)
    log.info("[startup] extensions: %d loaded, %d total", n,
             len(EXTENSIONS.extensions))
    try:
        n_migrated = _migrate_output_dirs()
        if n_migrated:
            log.info("[startup] renamed %d legacy output date folder(s) to ISO YYYY-MM-DD",
                     n_migrated)
    except Exception as e:  # noqa: BLE001
        log.warning("[startup] output folder migration failed: %s", e)
    try:
        purged = _purge_trash()
        if purged:
            log.info("[startup] purged %d trash entries older than %d days",
                     purged, TRASH_RETENTION_DAYS)
    except Exception as e:  # noqa: BLE001
        log.warning("[startup] trash purge failed: %s", e)
    # Hash models before serving so metadata carries real hashes and the disk-
    # heavy hashing never contends with generation I/O.
    try:
        await asyncio.to_thread(md.model_hash.scan_all, background=False)
    except Exception as e:  # noqa: BLE001
        log.warning("[startup] model-hash scan failed: %s", e)
    _log_runtime_env()


def _log_runtime_env() -> None:
    """Log torch / CUDA / GPU and engine versions for triage."""
    parts = [f"ui={md.UI_ID}", f"diff={md.DIFF_ID}"]
    try:
        import torch  # noqa: heavy and optional in test envs
        parts.append(f"torch={torch.__version__}")
        if torch.cuda.is_available():
            try:
                p = torch.cuda.get_device_properties(0)
                parts.append(f"cuda={torch.version.cuda}")
                parts.append(f"gpu={p.name} ({p.total_memory / 1024**3:.1f} GiB)")
            except Exception as e:  # noqa: BLE001
                parts.append(f"cuda=available (props failed: {e})")
        else:
            parts.append("cuda=unavailable (CPU)")
    except Exception as e:  # noqa: BLE001
        parts.append(f"torch=missing ({e})")
    log.info("[startup] runtime: %s", "  ".join(parts))


@app.get("/")
def index(request: Request):
    # With auth on: ?token= sets the cookie and redirects to bare "/" (keeps the
    # token out of history); no cookie gets the login page.
    if AUTH.enabled:
        qp_token = request.query_params.get("token")
        if qp_token is not None:
            resp = AUTH.accept(qp_token)
            if resp is None:
                return JSONResponse({"error": "invalid token"}, status_code=401)
            return resp
        if not AUTH.has_access(request):
            return AUTH.login_page()
    return HTMLResponse(_render_index(), headers={"Cache-Control": "no-cache"})


# ── auth endpoints ──────────────────────────────────────────────────

@app.get("/api/auth/status")
def api_auth_status():
    return {"auth": AUTH.enabled, "secure": AUTH.secure_cookie}


@app.post("/api/auth/login")
async def api_auth_login(request: Request):
    token = await read_login_token(request)
    resp = AUTH.accept(token)
    if resp is None:
        return JSONResponse({"error": "invalid token"}, status_code=401)
    return resp


@app.post("/api/auth/logout")
def api_auth_logout():
    resp = JSONResponse({"ok": True})
    AUTH.clear_cookie(resp)
    return resp


@app.get("/api/models")
def api_models():
    return {
        "checkpoints": scan_checkpoints(),
        "dits": scan_diffusion_models(),
        "vaes": scan_vae(),
        "tes": scan_text_encoders(),
        "loras": scan_loras(),
        "detailers": scan_detectors(),
        "upscalers": scan_upscalers(),
        "samplers_sd": SAMPLERS_SD,
        "samplers_anima": SAMPLERS_ANIMA,
        "samplers_flux": SAMPLERS_FLUX,
        "schedulers_sd": SCHEDULERS_SD,
        "schedulers_anima": SCHEDULERS_ANIMA,
        "schedulers_flux": SCHEDULERS_FLUX,
        "xyz_param_types": XYZ_PARAM_TYPES,
        "status": ENGINE.status_text(),
        "loaded": bool(ENGINE.loaded_name),
        "load_form": LAST_LOAD_FORM,
        "last_seed": ENGINE.last_seed,
        "recommended_offload": ENGINE.recommended_offload(),
        # Gates the UI's "fa2 attn" chip (package installed and an sm75 GPU).
        "fa2_available": ENGINE.fa2_attention_available(),
        "ui_id": md.UI_ID,
        "diff_id": md.DIFF_ID,
    }


@app.get("/api/status")
def api_status():
    return {"status": ENGINE.status_text(), "last_seed": ENGINE.last_seed}


def _do_load(p: LoadPayload) -> str:
    EXTENSIONS.run_hook("pre_load", payload=p)
    status = _do_load_impl(p)
    # status starts with "Loaded" or "Model already loaded" on success.
    EXTENSIONS.run_hook("post_load", payload=p, status=status)
    return status


def _validate_load(p: LoadPayload) -> Optional[str]:
    """Return an error string if a named file isn't on disk, else None, so
    /api/load can 400 at submit time instead of failing after the queue."""
    def _missing(label: str, name: str, d: Path) -> Optional[str]:
        # Plain filenames only; ".." or a separator would escape the models dir.
        if not model_name_ok(name) or not (d / name).is_file():
            return f"{label} not found: {name}"
        return None

    # The custom op graph-breaks in every block under compile.
    if p.attention == "fa2_turing" and p.compile:
        return "fa2 attention is incompatible with torch.compile; disable one"

    if p.model_type == "Anima":
        for label, name, d in (("DiT", p.dit, DIFFUSION_DIR),
                               ("VAE", p.vae, VAE_DIR),
                               ("Text encoder", p.te, TE_DIR)):
            if not name or name.startswith("("):
                return "Select all three Anima files"
            err = _missing(label, name, d)
            if err:
                return err
        return None
    if p.model_type == "FLUX":
        if p.checkpoint and not p.checkpoint.startswith("("):
            return _missing("Checkpoint", p.checkpoint, CHECKPOINTS_DIR)
        for label, name, d in (("DiT", p.dit, DIFFUSION_DIR),
                               ("VAE", p.vae, VAE_DIR),
                               ("Text encoder", p.te, TE_DIR)):
            if not name or name.startswith("("):
                return "Select an all-in-one checkpoint, or DiT + VAE + Text encoder"
            err = _missing(label, name, d)
            if err:
                return err
        if p.clip and not p.clip.startswith("("):
            err = _missing("CLIP", p.clip, TE_DIR)
            if err:
                return err
        return None
    # SD/SDXL
    if not p.checkpoint or p.checkpoint.startswith("("):
        return "Select a model"
    return _missing("Checkpoint", p.checkpoint, CHECKPOINTS_DIR)


def _do_load_impl(p: LoadPayload) -> str:
    # Default offload: stream for FLUX (its ~23 GB transformer OOMs under full)
    # and on low-VRAM cards, full otherwise.
    _to_bundle = {"none": False, "full": True,
                  "encoders": "encoders", "stream": "stream"}
    if p.offload is None:
        stream = p.model_type == "FLUX" or ENGINE.recommended_offload() == "stream"
        offload = "stream" if stream else True
    else:
        offload = _to_bundle.get(p.offload, True)

    # "always" forces tiled VAE decode; "auto" decides per decode from free VRAM.
    # FLUX always tiles.
    vae_tile_pref = SETTINGS.get("vae_tiling") == "always"

    if p.model_type == "Anima":
        for name in (p.dit, p.vae, p.te):
            if not name or name.startswith("("):
                return "Select all three Anima files"
        return ENGINE.load_anima(
            p.dit, p.vae, p.te,
            offload=offload, vae_tile=vae_tile_pref,
            compile=p.compile, cuda_graphs=p.cuda_graphs,
            fp16_accumulation=p.fp16_accumulation,
            attention=p.attention,
            vae_fp16=p.vae_fp16,
        )
    if p.model_type == "FLUX":
        # An all-in-one checkpoint takes precedence over split files.
        if p.checkpoint and not p.checkpoint.startswith("("):
            return ENGINE.load_model(
                p.checkpoint, offload=offload, vae_tile=True,
                compile=p.compile, cuda_graphs=p.cuda_graphs,
                fp16_accumulation=p.fp16_accumulation,
                attention=p.attention,
                vae_fp16=p.vae_fp16,
            )
        for name in (p.dit, p.vae, p.te):
            if not name or name.startswith("("):
                return "Select an all-in-one checkpoint, or DiT + VAE + Text encoder"
        return ENGINE.load_flux(
            p.dit, p.vae, p.te, clip_name=p.clip,
            offload=offload, vae_tile=True,
            compile=p.compile, cuda_graphs=p.cuda_graphs,
            fp16_accumulation=p.fp16_accumulation,
            attention=p.attention,
            vae_fp16=p.vae_fp16,
        )
    if not p.checkpoint or p.checkpoint.startswith("("):
        return "Select a model"
    return ENGINE.load_model(
        p.checkpoint,
        offload=offload, vae_tile=vae_tile_pref,
        compile=p.compile, cuda_graphs=p.cuda_graphs,
        channels_last=p.channels_last, tf32=p.tf32,
        fp16_accumulation=p.fp16_accumulation,
        vae_fp16=p.vae_fp16,
    )


@app.post("/api/load")
async def api_load(p: LoadPayload):
    """Queue a model load on the generation worker (it waits its turn) after
    checking the named files exist. Success is broadcast to every device."""
    err = _validate_load(p)
    if err:
        raise HTTPException(status_code=400, detail=err)
    def run(job: Job) -> dict:
        global LAST_LOAD_FORM
        status = _do_load(p)
        if status.startswith(("Loaded", "Model already loaded")):
            LAST_LOAD_FORM = p.dict()
            _write_last_load(LAST_LOAD_FORM)
        _push({"type": "status", **_state_payload()})
        return {"status": status, "loaded": bool(ENGINE.loaded_name)}

    job = Job("load", f"load {p.model_type}", run, priority=10)
    _enqueue(job)
    return {"job": job.id}


def _teacache_cuda_graphs_conflict(*thresholds: float) -> Optional[str]:
    """Submit-time guard: TeaCache can't run on a CUDA-Graphs Anima backbone.
    The engine re-checks at run time (the model can change while queued)."""
    if (ENGINE.cuda_graphs_enabled and ENGINE.loaded_family == "anima"
            and any(t > 0 for t in thresholds)):
        return ("TeaCache is incompatible with CUDA Graphs. Disable TeaCache "
                "or reload the model without the CUDA Graphs flag.")
    return None


@app.post("/api/generate")
async def api_generate(p: GeneratePayload):
    err = _teacache_cuda_graphs_conflict(
        p.teacache, p.upscale_teacache if p.upscale_enabled else 0.0)
    if err:
        raise HTTPException(status_code=400, detail=err)
    def run(job: Job) -> dict:
        on_progress, on_preview = _make_callbacks(job)
        return _run_generation(p, on_progress, on_preview)
    job = Job("generate", f"{p.mode} {p.width}×{p.height} · {p.steps} steps", run)
    _enqueue(job)
    return {"job": job.id}


@app.post("/api/upscale")
async def api_upscale(p: UpscalePayload):
    err = _teacache_cuda_graphs_conflict(p.teacache)
    if err:
        raise HTTPException(status_code=400, detail=err)
    def run(job: Job) -> dict:
        if not ENGINE.loaded_name:
            raise RuntimeError("Load a model first")
        on_progress, on_preview = _make_callbacks(job)
        input_image = _decode_image(p.input_image)
        image, unote = ENGINE.upscale(
            input_image,
            scale=float(p.scale), tile=int(p.tile),
            overlap=int(p.overlap), denoise=float(p.denoise),
            base_upscaler=p.base,
            prompt=p.prompt, negative_prompt=p.neg,
            steps=int(p.steps), cfg_scale=float(p.cfg),
            sampler=p.sampler, scheduler=p.scheduler,
            gate_reduce=SETTINGS["gate_reduce"],
            seed=int(p.seed),
            teacache_thresh=float(p.teacache),
            teacache_use_coeffs=bool(p.teacache_calibrated),
            teacache_forecast=p.teacache_forecast,
            teacache_rule=p.teacache_rule,
            progress_callback=on_progress,
            preview_callback=on_preview if p.preview else None,
        )
        upscale_meta = {
            "scale": float(p.scale), "tile": int(p.tile),
            "overlap": int(p.overlap), "denoise": float(p.denoise),
            "teacache": float(p.teacache),
            "base": p.base or "Lanczos",
            "prompt": p.prompt.strip() or "",
        }
        gen_kwargs = dict(
            prompt=p.prompt, negative_prompt=p.neg,
            steps=int(p.steps), cfg_scale=float(p.cfg),
            sampler=p.sampler, scheduler=p.scheduler,
            gate_reduce=SETTINGS["gate_reduce"],
        )
        # ENGINE.last_seed still holds whatever ran before; use the tile passes' seed.
        seed = ENGINE.last_upscale_seed
        out = _save_output(image, gen_kwargs, upscale=upscale_meta, seed=seed,
                           blur_check=p.blur_check)
        rel = out.relative_to(OUTPUTS_DIR)
        return {
            "image_url": _output_url(out),
            "path": rel.as_posix(),
            "nsfw_prompt": md.prompt_is_nsfw(p.prompt),
            "prompt_rating": md.prompt_rating(p.prompt),
            "info": f"{unote}  |  saved to {rel}",
            "seed": seed,
        }
    job = Job("upscale", f"upscale {p.scale}x", run)
    _enqueue(job)
    return {"job": job.id}


@app.post("/api/detail")
async def api_detail(p: DetailPayload):
    err = _teacache_cuda_graphs_conflict(p.teacache)
    if err:
        raise HTTPException(status_code=400, detail=err)
    active = [dm for dm in p.models if dm.model and not dm.model.startswith("(")]
    if not active:
        raise HTTPException(status_code=400, detail="Pick at least one detection model")

    def run(job: Job) -> dict:
        if not ENGINE.loaded_name:
            raise RuntimeError("Load a model first")
        if not ENGINE.can_inpaint:
            raise RuntimeError("Detailer needs inpaint, unavailable for this model")
        on_progress, on_preview = _make_callbacks(job)
        image = _decode_image(p.input_image)
        # The detailer doesn't record the seed it picked, so resolve it here and
        # share it across stacked passes (reproducible from the metadata).
        seed = int(p.seed) if p.seed >= 0 else random.randrange(2 ** 32 - 1)
        notes, applied = [], []
        for dm in active:
            # One failing detector must not lose the earlier passes' work.
            try:
                image, dnote = ENGINE.detail(
                    image,
                    detector_path=str(detector_path(dm.model)),
                    prompt=dm.prompt.strip() or p.prompt,
                    negative_prompt=p.neg,
                    confidence=float(p.confidence),
                    strength=float(p.strength),
                    steps=int(p.steps), cfg_scale=float(p.cfg),
                    sampler=p.sampler, scheduler=p.scheduler,
                    gate_reduce=SETTINGS["gate_reduce"],
                    dilation=int(p.dilation), padding=int(p.padding),
                    blur=int(p.blur), max_det=int(p.max_det),
                    seed=seed,
                    teacache_thresh=float(p.teacache),
                    teacache_use_coeffs=bool(p.teacache_calibrated),
                    teacache_forecast=p.teacache_forecast,
                    teacache_rule=p.teacache_rule,
                    progress_callback=on_progress,
                    preview_callback=on_preview if p.preview else None,
                )
                notes.append(f"{dm.model}: {dnote.replace('Detailer: ', '')}")
                applied.append(dm)
            except Exception as e:  # noqa: BLE001
                notes.append(f"⚠ {dm.model} FAILED: {e}")
        if not applied:
            raise RuntimeError("; ".join(notes))
        detailer_meta = {
            "models": [{"model": dm.model, "prompt": dm.prompt} for dm in applied],
            "neg": p.neg,
            "confidence": p.confidence,
            "strength": p.strength,
            "dilation": p.dilation,
            "padding": p.padding,
            "blur": p.blur,
            "maxDet": p.max_det,
        }
        gen_kwargs = dict(
            prompt=p.prompt, negative_prompt=p.neg,
            steps=int(p.steps), cfg_scale=float(p.cfg),
            sampler=p.sampler, scheduler=p.scheduler,
        )
        out = _save_output(image, gen_kwargs, detailer=detailer_meta, seed=seed,
                           blur_check=p.blur_check)
        rel = out.relative_to(OUTPUTS_DIR)
        return {
            "image_url": _output_url(out),
            "path": rel.as_posix(),
            "nsfw_prompt": md.prompt_is_nsfw(p.prompt),
            "prompt_rating": md.prompt_rating(p.prompt),
            "info": "detailer [" + "; ".join(notes) + f"]  |  saved to {rel}",
            "seed": seed,
        }

    job = Job("detail", f"detail {len(active)} pass(es)", run)
    _enqueue(job)
    return {"job": job.id}


@app.post("/api/xyz")
async def api_xyz(p: XYZPayload):
    def run(job: Job) -> dict:
        on_progress, on_preview = _make_callbacks(job)
        return _run_xyz(p, on_progress, on_preview)
    job = Job("xyz", "x/y/z grid", run)
    _enqueue(job)
    return {"job": job.id}


@app.post("/api/calibrate_oss")
async def api_calibrate_oss(p: CalibratePayload):
    def run(job: Job) -> dict:
        on_progress, _ = _make_callbacks(job)
        return _run_calibrate(p, on_progress)
    job = Job("calibrate", "OSS calibrate", run)
    _enqueue(job)
    return {"job": job.id}


@app.post("/api/calibrate_teacache")
async def api_calibrate_teacache(p: CalibratePayload):
    def run(job: Job) -> dict:
        on_progress, _ = _make_callbacks(job)
        return _run_calibrate_teacache(p, on_progress)
    job = Job("calibrate", "TeaCache calibrate", run)
    _enqueue(job)
    return {"job": job.id}


@app.get("/api/teacache_status")
def api_teacache_status():
    return ENGINE.teacache_status()


@app.get("/api/settings")
def api_settings():
    return SETTINGS


@app.post("/api/settings")
def api_save_settings(s: Settings):
    global SETTINGS
    _write_settings(s)
    SETTINGS = s.model_dump()
    # Apply VAE tiling to the loaded model now; later loads read it in _do_load.
    ENGINE.apply_vae_tiling(SETTINGS["vae_tiling"] == "always")
    # Blur off: free the tagger's VRAM. A blur_check generation reloads it.
    if not SETTINGS["nsfw_blur"]:
        tagger_mod.TAGGER.unload()
    return SETTINGS


# ── AI NSFW rating: status + full-gallery scan ─────────────────────

@app.get("/api/tagger_status")
def api_tagger_status():
    """Whether the WD tagger is usable, and how many existing outputs carry a
    current verdict (stale and deleted entries don't count)."""
    files = scan_outputs()
    rated = sum(1 for f in files if _cached_rating(f) is not None)
    return {"available": tagger_mod.timm_available(),
            "loaded": tagger_mod.TAGGER.loaded,
            "rated": rated, "total": len(files)}


@app.post("/api/gallery_scan")
def api_gallery_scan():
    """Rate every output lacking a current rating, as one low-priority job."""
    if not SETTINGS.get("nsfw_blur"):
        raise HTTPException(400, "Enable 'Blur R-rated and up' in Settings first")
    if not tagger_mod.timm_available():
        raise HTTPException(400, "The WD tagger needs the optional 'timm' package (pip install timm)")
    paths = []
    for f in scan_outputs():
        rel = f.relative_to(OUTPUTS_DIR).as_posix()
        if not _rating_current(_read_ratings().get(rel), f):
            paths.append(f)
    if not paths:
        return {"job": None, "total": 0}
    job = Job("tag", f"Rate NSFW gallery ({len(paths)} images)",
              lambda j: _tag_job_run(j, paths), priority=-10)
    job.total = len(paths)
    _enqueue(job)
    return {"job": job.id, "total": len(paths)}


def _maybe_auto_rescan() -> None:
    """Re-rate the gallery once, at the lowest priority, after a DECISION_VERSION
    bump made cached verdicts stale."""
    if not SETTINGS.get("nsfw_blur") or not tagger_mod.timm_available():
        return
    ratings = _read_ratings()
    if not ratings:
        return
    if all(e.get("v") == tagger_mod.DECISION_VERSION for e in ratings.values()):
        return
    paths = [f for f in scan_outputs()
             if not _rating_current(ratings.get(f.relative_to(OUTPUTS_DIR).as_posix()), f)]
    if not paths:
        return
    job = Job("tag", f"Re-rate NSFW gallery ({len(paths)} images)",
              lambda j: _tag_job_run(j, paths), priority=-10)
    job.total = len(paths)
    _enqueue(job)


# ── extension management ────────────────────────────────────────────

@app.get("/api/extensions")
def api_extensions():
    """List every discovered extension with its load state and web scripts."""
    return {"extensions": EXTENSIONS.list_serializable()}


@app.get("/api/extensions/web")
def api_extensions_web():
    """Script URLs of the enabled extensions' JS files."""
    return {"scripts": EXTENSIONS.web_script_urls()}


@app.post("/api/extensions/install")
def api_extensions_install(p: InstallPayload):
    """Install an extension from a git URL or a .zip archive URL, as a job on
    the shared worker (it imports modules and may pip install). The ``done``
    event carries the new extension's record."""
    # Validate the URL now so a bad one 400s instead of failing as a job.
    from extensions import _validate_install_url
    try:
        _validate_install_url(p.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    def run(job: Job) -> dict:
        ext = EXTENSIONS.install(p.url, install_pip_deps=p.install_pip_deps)
        EXTENSIONS.mount_into(app)
        return {"extension": ext.to_dict()}
    job = Job("install", f"install {p.url}", run)
    _enqueue(job)
    return {"job": job.id}


@app.post("/api/extensions/toggle")
def api_extensions_toggle(p: TogglePayload):
    """Enable or disable an extension. Backend hooks/routes apply at once; the
    frontend script tags refresh on the next page load."""
    ext = EXTENSIONS.set_enabled(p.name, p.enabled)
    if p.enabled:
        EXTENSIONS.mount_into(app)
    return {"extension": ext.to_dict()}


@app.post("/api/extensions/update")
def api_extensions_update(p: UpdatePayload):
    """Pull the latest version of a git-installed extension and reload it, as a
    job on the shared worker."""
    if p.name not in EXTENSIONS.extensions:
        raise HTTPException(status_code=404, detail="extension not found")
    def run(job: Job) -> dict:
        ext = EXTENSIONS.update(p.name, install_pip_deps=p.install_pip_deps)
        EXTENSIONS.mount_into(app)
        return {"extension": ext.to_dict()}
    job = Job("update", f"update {p.name}", run)
    _enqueue(job)
    return {"job": job.id}


@app.post("/api/extensions/reload")
def api_extensions_reload(name: str):
    """Re-import an extension's entry module, dropping its old hooks/routes first."""
    EXTENSIONS.reload_one(name)
    EXTENSIONS.mount_into(app)
    ext = EXTENSIONS.extensions.get(name)
    if ext is None:
        raise HTTPException(status_code=404, detail="extension not found")
    return {"extension": ext.to_dict()}


@app.post("/api/extensions/uninstall")
def api_extensions_uninstall(p: UninstallPayload):
    """Remove an extension's folder, hooks, routes and persisted state."""
    try:
        EXTENSIONS.uninstall(p.name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"uninstalled": p.name}


@app.post("/api/cancel")
def api_cancel(p: CancelPayload):
    """Cancel a job by id (``None`` = the running one). A running job aborts at
    its next step; a queued one is dropped."""
    with QUEUE_LOCK:
        target = CURRENT if p.job is None else None
        if p.job is not None:
            if CURRENT and CURRENT.id == p.job:
                target = CURRENT
            else:
                target = next((j for j in QUEUE if j.id == p.job), None)
        queued = target is not None and target in QUEUE
        if queued:
            QUEUE.remove(target)
    if target is None:
        return {"cancelling": False}
    target.cancel.set()
    if queued:  # never ran, so report it now
        target.status = "cancelled"
        _push({"type": "cancelled", "job": target.id})
        _broadcast_queue()
    return {"cancelling": True}


@app.get("/api/events")
async def api_events(request: Request):
    """Shared SSE stream of queue changes, progress, previews and model status."""
    q: asyncio.Queue = asyncio.Queue(maxsize=SSE_QUEUE_MAX)
    SUBSCRIBERS.add(q)
    snapshot = {"type": "snapshot", **_state_payload(),
                "jobs": _queue_list(), "running": CURRENT.id if CURRENT else None}
    if CURRENT:
        snapshot["progress"] = {"job": CURRENT.id, "step": CURRENT.step, "total": CURRENT.total}
    await q.put(snapshot)

    async def gen():
        try:
            while not SHUTDOWN.is_set():
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": ping\n\n"   # keep-alive; also surfaces disconnects
                    continue
                if ev is None:           # shutdown sentinel
                    break
                yield "data: " + json.dumps(ev) + "\n\n"
        finally:
            SUBSCRIBERS.discard(q)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/oss_status")
def api_oss_status(steps: int, width: int, height: int, shift: float):
    return {"calibrated": ENGINE.oss_calibrated(steps, width, height, shift)}


@app.get("/api/gallery")
def api_gallery(q: str = ""):
    """List gallery images, optionally filtered by a case-insensitive substring
    of the prompt, negative, model, sampler or scheduler. Each entry carries its
    ``rating`` and ``nsfw`` flag, read from the cached index."""
    def dto(entry):
        return {
            "url": f"/outputs/{entry['path']}",
            "name": entry["name"],
            "path": entry["path"],
            "date": entry["date"],
            "nsfw": entry["nsfw"],
            "rating": entry["rating"],
        }
    query = (q or "").strip().lower()
    if not query:
        return {"images": [dto(e) for e in _gallery_index()]}
    return {"images": [dto(e) for e in _gallery_search(query)]}


# ── gallery search index ────────────────────────────────────────────
# Parsing every PNG is too costly per keystroke, so the index is cached and
# rebuilt when the newest output folder's mtime advances. Saves and deletes
# also update it explicitly, since ext4 mtimes have 1 s resolution.
_GALLERY_INDEX: Optional[list] = None
_GALLERY_INDEX_KEY: float = 0.0
_GALLERY_INDEX_LOCK = threading.Lock()


def _outputs_dir_mtime() -> float:
    """Newest date-folder mtime under outputs/, the index's freshness key."""
    try:
        return max(
            (d.stat().st_mtime for d in OUTPUTS_DIR.iterdir() if d.is_dir()),
            default=0.0,
        )
    except OSError:
        return 0.0


def _index_entry(f: Path, fields: dict) -> dict:
    """One index row from a file and its parsed AUTO1111 fields."""
    # A current vision rating wins over the prompt heuristic.
    rating = md.prompt_rating(str(fields.get("prompt", "")))
    vis = _cached_rating(f)
    if vis:
        rating = vis["rating"]
    return {
        "path": f.relative_to(OUTPUTS_DIR).as_posix(),
        "name": f.name,
        "date": f.parent.name,
        "prompt": str(fields.get("prompt", "")),
        "neg": str(fields.get("negative_prompt", "")),
        "model": str(fields.get("model", "")),
        "sampler": str(fields.get("sampler", "")),
        "scheduler": str(fields.get("scheduler", "")),
        "rating": rating,
        "nsfw": rating in ("R", "X", "XXX"),
    }


def _invalidate_gallery_index() -> None:
    global _GALLERY_INDEX, _GALLERY_INDEX_KEY
    with _GALLERY_INDEX_LOCK:
        _GALLERY_INDEX = None
        _GALLERY_INDEX_KEY = 0.0


def _gallery_index_add(path: Path, params: str) -> None:
    """Splice a just-saved output into the cached index (newest first) instead
    of forcing a rebuild that re-opens every PNG. No-op while the index is cold.
    """
    global _GALLERY_INDEX_KEY
    with _GALLERY_INDEX_LOCK:
        if _GALLERY_INDEX is None:
            return
        _GALLERY_INDEX.insert(0, _index_entry(path, md.parse_metadata(params)))
        # Adopt our own save's folder mtime, or the next read would rebuild.
        _GALLERY_INDEX_KEY = _outputs_dir_mtime()


def _gallery_index_patch_ratings(ratings: dict) -> None:
    """Update the rating fields of already-indexed rows, keyed by relative path.
    Unknown rows are skipped; the next build picks them up.
    """
    with _GALLERY_INDEX_LOCK:
        if _GALLERY_INDEX is None:
            return
        by_path = {e["path"]: e for e in _GALLERY_INDEX}
        for rel, r in ratings.items():
            entry = by_path.get(rel)
            if entry is not None:
                entry["rating"] = r["rating"]
                entry["nsfw"] = bool(r["nsfw"])


def _gallery_index() -> list:
    global _GALLERY_INDEX, _GALLERY_INDEX_KEY
    newest = _outputs_dir_mtime()
    with _GALLERY_INDEX_LOCK:
        if _GALLERY_INDEX is not None and newest <= _GALLERY_INDEX_KEY:
            return _GALLERY_INDEX
        index: list = []
        for f in scan_outputs():
            raw = md.read_png_metadata(str(f))
            index.append(_index_entry(f, md.parse_metadata(raw) if raw else {}))
        _GALLERY_INDEX = index
        _GALLERY_INDEX_KEY = newest
        return index


def _gallery_search(query: str) -> list:
    """Filter the cached index by a lowercased substring across metadata fields."""
    out = []
    for entry in _gallery_index():
        haystack = " ".join(
            (entry["prompt"], entry["neg"], entry["model"],
             entry["sampler"], entry["scheduler"])
        ).lower()
        if query in haystack:
            out.append(entry)
    return out


def _thumb_cache_path(target: Path) -> Path:
    """Thumbnail cache path for ``target``, keyed by its mtime+size so an
    overwritten source gets a fresh thumbnail."""
    rel = target.relative_to(OUTPUTS_DIR.resolve())
    try:
        st = target.stat()
        key = f"{target.stem}_{st.st_mtime_ns}_{st.st_size}"
    except OSError:
        key = target.stem
    return _THUMBS_DIR / rel.parent / f"{key}.webp"


def _purge_thumb_cache(target: Path, keep: Optional[Path] = None) -> None:
    """Remove cached thumbnails for ``target`` (all mtime/size versions, except
    ``keep``). Best-effort."""
    try:
        rel = target.relative_to(OUTPUTS_DIR.resolve())
    except ValueError:
        return
    thumb_dir = _THUMBS_DIR / rel.parent
    if not thumb_dir.is_dir():
        return
    for f in thumb_dir.glob(f"{target.stem}_*.webp"):
        if f != keep:
            try: f.unlink()
            except OSError: pass


@app.get("/api/thumb")
def api_thumb(path: str):
    """Serve a cached thumbnail for a gallery image (path under outputs/),
    building it on first request under .cache/thumbs/."""
    target = (OUTPUTS_DIR / path).resolve()
    outputs_root = OUTPUTS_DIR.resolve()
    if outputs_root not in target.parents or not target.is_file():
        raise HTTPException(status_code=404)
    cache = _thumb_cache_path(target)
    if not cache.is_file():
        cache.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(target) as im:
            im = im.convert("RGB")
            im.thumbnail((THUMB_MAX, THUMB_MAX))
            im.save(cache, "WEBP", quality=80)
        _purge_thumb_cache(target, keep=cache)
    return FileResponse(cache, media_type="image/webp")


@app.delete("/api/gallery")
def api_gallery_delete(path: str):
    """Soft-delete a gallery image: move it to ``outputs/.trash/`` under a
    timestamped name, drop its thumbnails and invalidate the gallery caches."""
    target = (OUTPUTS_DIR / path).resolve()
    outputs_root = OUTPUTS_DIR.resolve()
    if outputs_root not in target.parents or not target.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    _TRASH_DIR.mkdir(parents=True, exist_ok=True)
    # Nanosecond stamp: same-named files deleted in the same second must not
    # overwrite each other.
    trashed = _TRASH_DIR / f"{time.time_ns()}_{target.name}"
    try:
        shutil.move(str(target), str(trashed))
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Could not delete: {e}")
    _purge_thumb_cache(target)
    _invalidate_gallery_index()
    invalidate_outputs_cache()
    try:
        _purge_trash()
    except Exception as e:  # noqa: BLE001  never let cleanup fail the delete
        log.warning("trash purge failed: %s", e)
    return {"deleted": path, "trashed": trashed.name}


@app.get("/api/metadata")
def api_metadata(path: str):
    """Raw and workspace-normalised metadata for a gallery image."""
    target = (OUTPUTS_DIR / path).resolve()
    if OUTPUTS_DIR.resolve() not in target.parents or not target.is_file():
        return {"raw": "", "fields": {}}
    raw = md.read_png_metadata(str(target))
    fields = md.workspace_fields(md.parse_metadata(raw))
    return {"raw": raw, "fields": fields}


@app.post("/api/metadata/parse")
async def api_metadata_parse(file: UploadFile = File(...)):
    """Dump every PNG chunk plus the parsed A1111/ComfyUI views of an upload."""
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image too large (max 64 MB)")
    try:
        with Image.open(io.BytesIO(data)) as img:
            info = dict(img.info)
    except Exception as e:  # noqa: BLE001
        return {"text": f"Could not read image: {e}", "fields": {}}

    if not info:
        return {"text": "No metadata found in this image.", "fields": {}}

    lines = ["═ ALL PNG METADATA KEYS ═"]
    for k, v in info.items():
        val = str(v)
        if len(val) > 600:
            val = val[:600] + "..."
        lines.append(f"  {k}: {val}")

    auto1111 = info.get("parameters", "")
    if auto1111:
        lines += ["", "═ AUTO1111 / FORGE PARSED ═"]
        lines += [f"  {k}: {v}" for k, v in md.parse_metadata(auto1111).items()]

    comfyui = info.get("prompt", "")
    if comfyui:
        lines += ["", "═ COMFYUI PARSED ═"]
        lines += [f"  {k}: {v}" for k, v in md.parse_comfyui_metadata(comfyui).items()]

    if auto1111:
        fields = md.workspace_fields(md.parse_metadata(auto1111))
    else:
        fields = md.workspace_fields(md.parse_comfyui_metadata(comfyui))

    return {"text": "\n".join(lines), "fields": fields}


@app.post("/api/metadata/parse_text")
def api_metadata_parse_text(p: ParseTextPayload):
    """Parse a pasted ``parameters`` string into workspace fields (SD WebUI's
    read-generation-parameters)."""
    return {"fields": md.workspace_fields(md.parse_metadata(p.text))}


# Static mounts (declared last so /api routes win).
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)  # StaticFiles errors if missing
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")
app.mount("/static", StaticFiles(directory=_STATIC), name="static")
