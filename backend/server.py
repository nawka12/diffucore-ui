"""FastAPI web layer for Diffucore.

Wraps the framework-agnostic ``ENGINE`` singleton. Generation is blocking torch
code, so jobs (generate / xyz / calibrate / load) are queued and run one at a
time on a single background worker thread. Every connected device subscribes to
one shared Server-Sent-Events stream (``/api/events``); queue changes, sampling
progress, live previews, and model-load status are broadcast to all of them, so
a second device — or a refresh — stays in sync without reloading the model.
"""

from __future__ import annotations

import asyncio
import atexit
import base64
import io
import itertools
import json
import logging
import os
import shutil
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, List, Optional

import uvicorn
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from engine import ENGINE, SAMPLERS_SD, SAMPLERS_ANIMA, SAMPLERS_FLUX, SCHEDULERS_SD, SCHEDULERS_ANIMA, SCHEDULERS_FLUX
from utils import (
    OUTPUTS_DIR, MODELS_DIR, CHECKPOINTS_DIR, DIFFUSION_DIR, VAE_DIR, TE_DIR,
    detector_path,
    scan_checkpoints, scan_loras, scan_diffusion_models,
    scan_vae, scan_text_encoders, scan_detectors, scan_upscalers, scan_outputs, next_output_path,
)
from xyz_grid import generate_xyz_grid, PARAM_TYPES as XYZ_PARAM_TYPES
import metadata as md
from auth import AuthGate, COOKIE_NAME, load_or_create_token, origin_ok, read_login_token
from extensions import (
    ExtensionLoader, InstallPayload, TogglePayload, UninstallPayload,
)

log = logging.getLogger("diffucore.server")

MAX_UPLOAD_BYTES = 64 * 1024 * 1024  # 64 MB cap for metadata-parse uploads
MAX_BODY_BYTES = 128 * 1024 * 1024   # global cap: base64 of a 4K PNG fits, GBs don't

_ROOT = Path(__file__).resolve().parent.parent
_STATIC = _ROOT / "static"

# Cache-bust token for static assets: the newest mtime among the bundled files,
# in hex. After an update (git pull rewrites the files) the token changes, so
# the versioned ?v= URLs in index.html miss the browser cache and refetch the
# new app.js/style.css. A startup snapshot is enough since updates restart the
# server. index.html itself is served no-cache so the fresh token always wins.
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
# Dev mode (DIFFUCORE_DEV=1): re-read index.html, app.js, style.css from disk on
# every request so frontend edits show up on a plain refresh without restarting
# the server. The asset version is recomputed too, so the ?v= URLs in index.html
# change and the browser refetches the new JS/CSS. Production stays on the
# startup snapshot — a single read, no per-request stat calls.
_DEV_MODE = os.environ.get("DIFFUCORE_DEV") in ("1", "true", "yes")


# ── auth + CSRF guard ───────────────────────────────────────────────
# Disabled by default (localhost is private). ``app.py`` enables it for
# ``--share`` (auto-generated token) and ``--auth-token`` (explicit), or it can
# be turned on by setting DIFFUCORE_AUTH_TOKEN in the environment. The middleware
# below gates every non-public path on a valid cookie / bearer token, and blocks
# cross-origin state-changing requests (CSRF) regardless of auth.
AUTH = AuthGate(token="", enabled=False)


def configure_auth(*, token: str, enabled: bool, secure: bool = False) -> None:
    """Turn the auth gate on (called from app.py once args are parsed, before
    uvicorn.run). The middleware reads ``AUTH`` at request time, so configuring
    here is in time for the first request."""
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
    """Write ``text`` to ``path`` atomically: write a temp file in the same
    directory, then ``os.replace`` it over the target. A crash or kill mid-write
    leaves the previous file intact instead of a truncated one — so settings,
    last-load, and extension state don't silently fall back to defaults."""
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
    # One <script> tag per enabled extension JS file (see extensions.py). Built
    # at render time so a freshly-installed extension's UI shows up after the
    # install endpoint returns, without a server restart.
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
_THUMBS_DIR = _ROOT / ".cache" / "thumbs"  # lazily-built gallery-grid thumbnails
THUMB_MAX = 384  # long-edge px; the grid uses these instead of the full PNGs

# ── gallery soft-delete (#3) ──────────────────────────────────────────
# DELETE /api/gallery moves files here instead of unlink()-ing them, so a
# fat-fingered double-click (the two-click confirm is racy on a slow link) can
# be recovered by hand from outputs/.trash/ until the purge runs. Files are
# purged after TRASH_RETENTION_DAYS; the purge runs once at startup and on each
# delete (cheap: one iterdir + mtime check).
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

class _Cancelled(BaseException):
    """Raised from the progress callback to unwind a running generation.

    Inherits ``BaseException`` so it slips past the engine's broad
    ``except Exception`` handlers (e.g. the per-pass detailer guard) and is only
    ever caught by the worker — while ``finally`` blocks still reclaim VRAM and
    clear temp LoRAs on the way out."""


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
    teacache: float = Field(0.0, ge=0.0, le=1.0)           # TeaCache rel-L1 threshold (0 = off; Anima only)
    teacache_calibrated: bool = True     # use the fitted rescale polynomial vs the raw identity path
    deepcache: int = Field(1, ge=1, le=64)                # DeepCache reuse interval (1 = off; SD/SDXL UNet only)
    input_image: Optional[str] = None   # base64 / data-URL
    mask_image: Optional[str] = None
    preview: bool = True                 # stream live latent previews while sampling

    # ── detailer (ADetailer-style passes run after the main image) ──
    # Each entry is one detection model + its own prompt; the rest is shared.
    detail_enabled: bool = False
    detail_models: List[DetailerModel] = []
    detail_neg: str = ""
    detail_confidence: float = Field(0.3, ge=0.0, le=1.0)
    detail_strength: float = Field(0.4, ge=0.0, le=1.0)
    detail_dilation: int = Field(4, ge=0, le=128)
    detail_padding: int = Field(32, ge=0, le=512)
    detail_blur: int = Field(4, ge=0, le=64)
    detail_max: int = Field(0, ge=0, le=1000)              # 0 = all detections
    detail_teacache: bool = False        # apply the main TeaCache threshold to detailer passes (Anima)

    # ── upscaler (tiled, post-gen) ─────────────────────────────────
    upscale_enabled: bool = False
    upscale_scale: float = Field(2.0, ge=1.0, le=8.0)
    upscale_denoise: float = Field(0.35, ge=0.0, le=1.0)
    upscale_tile: int = Field(1024, ge=128, le=4096)
    upscale_overlap: int = Field(128, ge=0, le=2048)
    upscale_prompt: str = ""
    upscale_teacache: float = Field(0.0, ge=0.0, le=1.0)   # TeaCache for the refine pass (0 = off); independent of main gen
    upscale_base: str = ""               # ESRGAN model in models/upscalers/ (blank = Lanczos)


class UpscalePayload(BaseModel):
    """Standalone upscale — input image + all params the tiled upscaler needs."""
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
    preview: bool = True


class CalibratePayload(BaseModel):
    prompt: str = ""
    neg: str = ""
    steps: int = Field(12, ge=1, le=200)
    cfg: float = Field(4.0, ge=0.0, le=50.0)
    seed: int = Field(0, ge=0, le=2**63 - 1)
    width: int = Field(1024, ge=64, le=8192)
    height: int = Field(1024, ge=64, le=8192)
    shift: float = Field(3.0, ge=0.0, le=30.0)
    grid: int = Field(80, ge=1, le=500)    # dense teacher-trajectory candidate count (K)


class GenDefaults(BaseModel):
    """A snapshot of the Generate form's reusable params, saved as the per-session
    defaults the UI seeds on load (the model-type-specific samplers still fall back
    if invalid for the loaded family)."""
    sampler: str = "dpmpp_2m"
    scheduler: str = "karras"
    steps: int = 25
    cfg: float = 6.0
    width: int = 1024
    height: int = 1024
    shift: float = 3.0
    # Prompt/negative are saved only when filled (None = leave the form's own value).
    prompt: Optional[str] = None
    neg: Optional[str] = None


class Settings(BaseModel):
    """Persisted global settings exposed through the settings panel — the knobs
    that aren't per-image. Defaults mirror the submodule's, so an unset settings
    file leaves generation behaviour unchanged. Applied at generation time when
    the active sampler/scheduler uses them (see ``_run_generation``)."""
    # Anima sampler/scheduler knobs.
    curvature: float = 0.25       # secant / secant_anneal x0 extrapolation strength
    eta_max: float = 1.0          # secant_anneal / euler_ancestral_anneal ancestral noise
    beta_alpha: float = 0.6       # beta scheduler Beta(α, β) — low-t (σ→0) density
    beta_beta: float = 0.6        # beta scheduler — high-t (σ→1) density
    lq_threshold: float = 0.025   # linear_quadratic threshold_noise (linear/quad knee)
    # VAE decode: "auto" tiles only when a full decode won't fit free VRAM;
    # "always" forces tiled decode. Applies to Anima + SD/SDXL (FLUX always tiles).
    vae_tiling: str = "auto"      # "auto" | "always"
    # Generate-form defaults seeded on load (None = use the app's built-in defaults).
    gen_defaults: Optional[GenDefaults] = None


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
    x_type: str = "None"
    x_vals: str = ""
    y_type: str = "None"
    y_vals: str = ""
    z_type: str = "None"
    z_vals: str = ""
    preview: bool = True                 # stream live latent previews per cell


class CancelPayload(BaseModel):
    job: Optional[int] = None        # None = cancel whatever is currently running


class ParseTextPayload(BaseModel):
    text: str = ""


# ── helpers ─────────────────────────────────────────────────────────

def _decode_image(data: str) -> Image.Image:
    """Decode a base64 / data-URL image to RGB.

    An RGBA / LA / PA source is composited onto **white** before dropping the
    alpha channel — ``convert("RGB")`` alone leaves the stored RGB values where
    pixels were transparent, which renders transparent regions as black (a
    silent "composite-black"). Compositing onto white makes the loss explicit
    and predictable, and a warning is logged so an inpaint input that lost its
    transparency is traceable in the log.
    """
    if data.strip().startswith("data:") and "," in data:
        data = data.split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(data)))
    if img.mode in ("RGBA", "LA", "PA") or (
        "A" in img.getbands() and img.mode not in ("RGB", "L", "P")
    ):
        log.warning("input image had an alpha channel; composited onto white "
                    "(transparency is not preserved) — mode=%s", img.mode)
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg.convert("RGB")
    return img.convert("RGB")


def _decode_mask(data: str) -> Image.Image:
    """Decode a base64 / data-URL mask to a single-channel ``L`` image.

    Masks often travel as the alpha channel of an RGBA PNG (transparent = don't
    paint, opaque = paint). ``convert("L")`` from RGBA takes the *luminance* of
    the RGB channels and silently ignores the alpha — so an alpha-mask would be
    misread as a blank or wrong mask. If an alpha channel is present, extract it
    via ``split()`` (``convert("A")`` isn't a supported PIL transform) and use
    that band as the mask; otherwise fall back to luminance. The engine's
    ``_fit_inpaint`` re-converts to ``L`` anyway, so returning ``L`` here is
    lossless.
    """
    if data.strip().startswith("data:") and "," in data:
        data = data.split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(data)))
    if img.mode in ("RGBA", "LA", "PA") or "A" in img.getbands():
        # Alpha is the mask intent: opaque = paint here. split() returns the
        # bands as single-channel "L" images; the last is alpha.
        return img.split()[-1]
    return img.convert("L")


def _output_url(path: Path) -> str:
    return f"/outputs/{path.relative_to(OUTPUTS_DIR).as_posix()}"


def _save_output(image: Image.Image, gen_kwargs: dict,
                 detailer: Optional[dict] = None,
                 upscale: Optional[dict] = None) -> Path:
    """Save an image to outputs/ with AUTO1111 metadata; return its path.

    Used for single generations and for each individual X/Y/Z cell. ``detailer``,
    when given, is appended to the ``parameters`` line as ADetailer-compatible
    keys so the post-gen detailer settings can be restored later.
    """
    out = next_output_path(ENGINE.last_seed)
    meta = PngInfo()
    meta_kwargs = {k: v for k, v in gen_kwargs.items() if k != "progress_callback"}
    meta.add_text("parameters", md.format_metadata(meta_kwargs, ENGINE, detailer=detailer, upscale=upscale))
    image.save(out, pnginfo=meta)
    # Invalidate the gallery search index so the new image is visible to the next
    # search without waiting for the day-folder mtime to advance (covers
    # same-second saves on 1s-mtime filesystems, and standalone upscale saves).
    _invalidate_gallery_index()
    return out


# ── generation (ported from the old _generate_with_loras) ───────────

def _run_generation(p: GeneratePayload, on_progress: Callable[[int, int], None],
                    on_preview: Optional[Callable] = None) -> dict:
    if not ENGINE.loaded_name:
        raise RuntimeError("Load a model first")

    # pre_generate: extensions can tweak the payload (prompt, seed, steps, …)
    # before the engine runs. Mutations land on the Pydantic model in place.
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
            deepcache_interval=int(p.deepcache),
            progress_callback=on_progress,
            preview_callback=on_preview if p.preview else None,
        )

        # Global sampler/scheduler knobs from the settings panel (Anima only).
        # Inject only the ones the active sampler/scheduler actually consumes, so
        # they round-trip into PNG metadata without polluting it with unused keys.
        if ENGINE.loaded_family == "anima":
            if p.sampler in ("secant", "secant_anneal"):
                common["curvature"] = float(SETTINGS["curvature"])
            if p.sampler in ("secant_anneal", "euler_ancestral_anneal", "dpmpp_2m_anneal"):
                common["eta_max"] = float(SETTINGS["eta_max"])
            # uni_pc_anneal omitted on purpose: it uses its own low baked-in
            # eta_max (0.2); the shared 1.0 panel default over-smooths it.
            if p.scheduler == "beta":
                common["beta_alpha"] = float(SETTINGS["beta_alpha"])
                common["beta_beta"] = float(SETTINGS["beta_beta"])
            if p.scheduler == "linear_quadratic":
                common["lq_threshold"] = float(SETTINGS["lq_threshold"])

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
        image, info = gen_fn(**gen_kwargs)

        # Upscaler (tiled, post-gen): run first, on the base image, so the
        # detailer below refines at the upscaled resolution and gets the final
        # say (its region inpaints aren't re-disturbed by the upscale pass).
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
                    seed=int(p.seed),
                    teacache_thresh=float(p.upscale_teacache),
                    teacache_use_coeffs=bool(p.teacache_calibrated),
                    progress_callback=on_progress,
                    preview_callback=on_preview if p.preview else None,
                )
                upscale_info = "  |  " + unote
                upscaled = True
            except Exception as e:  # noqa: BLE001 — keep the base image, but surface the
                # failure loudly and skip the upscale metadata below, so a swallowed
                # OOM can't masquerade as a successful upscale (saved base + metadata
                # that claims it was upscaled).
                upscale_info = f"  |  ⚠ UPSCALE FAILED — saved un-upscaled base image ({e})"

        # Detailer: run last, on the (possibly upscaled) image, so its region
        # inpaints get the final say. Each stacked detection model runs in
        # sequence, feeding the refined image into the next pass. Per-model
        # prompt; the rest is shared.
        detail_info = ""
        active = [dm for dm in p.detail_models
                  if dm.model and not dm.model.startswith("(")] if p.detail_enabled else []
        applied = []  # detection models that actually refined the image (drives metadata)
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
                        dilation=int(p.detail_dilation), padding=int(p.detail_padding),
                        blur=int(p.detail_blur), max_det=int(p.detail_max),
                        seed=int(p.seed),
                        teacache_thresh=float(p.teacache) if p.detail_teacache else 0.0,
                        teacache_use_coeffs=bool(p.teacache_calibrated),
                        progress_callback=on_progress,
                        preview_callback=on_preview if p.preview else None,
                    )
                    notes.append(f"{dm.model}: {dnote.replace('Detailer: ', '')}")
                    applied.append(dm)
                except Exception as e:  # noqa: BLE001 — keep the image, surface the failure
                    # loudly, and exclude this model from the metadata below so a
                    # swallowed pass can't masquerade as a successful detail.
                    notes.append(f"⚠ {dm.model} FAILED: {e}")
            detail_info = "  |  detailer [" + "; ".join(notes) + "]"

        # inference clock spans base generation plus the detailer and upscaler
        # passes — i.e. everything but the disk save below.
        elapsed = time.perf_counter() - t0

        # Save the *raw* prompt/neg (the <lora:…> tags survive parse_lora_prompt
        # stripping) so the LoRA selection round-trips through metadata restore.
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
        # post_generate: extensions can post-process the image (watermark,
        # filter, composite) before it's written. A handler may replace
        # ctx.image; info carries the base gen info for reference.
        gctx = EXTENSIONS.run_hook(
            "post_generate", payload=p, image=image, info=info,
        )
        image = gctx.image

        out = _save_output(image, gen_kwargs, detailer=detailer_meta, upscale=upscale_meta)
        rel = out.relative_to(OUTPUTS_DIR)
        # post_save: fire-and-forget notification that the PNG is on disk; an
        # extension might mirror it elsewhere, log it, etc.
        EXTENSIONS.run_hook("post_save", payload=p, image=image, path=out)
        return {
            "image_url": _output_url(out),
            "info": f"{lora_info}{info}  |  inference: {elapsed:.2f}s{upscale_info}{detail_info}  |  saved to {rel}",
            "seed": ENGINE.last_seed,
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
    )
    # A "Checkpoint" axis swaps the in-memory model per cell, leaving the last
    # swept checkpoint loaded. Restore the model the user actually had loaded so
    # the app returns to its prior state (and the next plain generation isn't run
    # on a surprise checkpoint).
    swaps_model = "Checkpoint" in (p.x_type, p.y_type, p.z_type)
    try:
        grids, info = generate_xyz_grid(
            base_kwargs,
            p.x_type, p.x_vals, p.y_type, p.y_vals, p.z_type, p.z_vals,
            progress_callback=on_progress,
            preview_callback=on_preview if p.preview else None,
            save_callback=_save_output,
        )
        # base_kwargs was mutated in-place by generate_xyz_grid (prompt cleaned,
        # base seed resolved), so it carries the right params for the grid metadata.
        urls = []
        for grid in grids:
            out = _save_output(grid, base_kwargs)
            urls.append(_output_url(out))
        return {"grids": urls, "info": info}
    finally:
        if swaps_model and LAST_LOAD_FORM:
            try:
                _do_load(LoadPayload(**LAST_LOAD_FORM))
            except Exception:  # noqa: BLE001 — keep the grid result; report real state
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
# A single background worker runs jobs one at a time (the GPU can only do one),
# so the worker thread itself is the serialization — no lock needed. Every
# device subscribes to one shared SSE stream and sees the same queue + progress.

_job_ids = itertools.count(1)


class Job:
    def __init__(self, kind: str, label: str, run: Callable[["Job"], dict],
                 *, priority: int = 0):
        self.id = next(_job_ids)
        self.kind = kind            # generate | xyz | calibrate | load | install
        self.label = label          # short human description for the queue list
        self.run = run              # run(job) -> result dict; may raise _Cancelled
        self.status = "queued"      # queued | running | done | error | cancelled
        self.cancel = threading.Event()
        self.step = 0               # live progress, for snapshots on (re)connect
        self.total = 0
        self.priority = int(priority)  # higher runs sooner; load jobs jump the queue
        self.last_preview: Optional[Image.Image] = None  # last streamed preview, for shutdown partial-save


QUEUE: "deque[Job]" = deque()
QUEUE_LOCK = threading.Lock()
QUEUE_WAKE = threading.Event()
CURRENT: Optional[Job] = None

# One asyncio.Queue per connected SSE client; the worker fans events out to all.
# Capped so a slow client (or a runaway SSE reconnector swarm on a flaky network)
# can't grow RSS unbounded: on overflow we drop the oldest event — the newest
# state (progress, preview, queue) is what the UI wants anyway.
SSE_QUEUE_MAX = 256
SUBSCRIBERS: "set[asyncio.Queue]" = set()
APP_LOOP: Optional[asyncio.AbstractEventLoop] = None

# Live-preview cost control. A preview is a rough latent→RGB approximation that
# is broadcast to every connected client and json-serialized per client on the
# event loop. Re-encoding a full-res lossless PNG on every sampling step and
# fanning a multi-MB data-URL out N times stalls the loop, so cap the rate
# (time-based, so it adapts to both step speed and step count) and the
# resolution, and use lossy WebP — the preview is noisy and transient while the
# saved result is always full quality.
PREVIEW_MIN_INTERVAL = 0.2   # seconds between streamed previews (≤5 fps)
PREVIEW_MAX_SIDE = 512       # downscale to this long side before encoding

# The /api/events SSE streams are long-lived, so uvicorn's graceful shutdown would
# wait on them forever — Ctrl+C appears to hang until a second, forced Ctrl+C.
# uvicorn calls Server.handle_exit the instant a signal arrives (before it starts
# waiting for connections to close), so wrap it to set SHUTDOWN and wake every
# stream with a sentinel; each gen() then returns and the server exits cleanly.
SHUTDOWN = asyncio.Event()


def _wake_for_shutdown() -> None:
    SHUTDOWN.set()
    for q in list(SUBSCRIBERS):
        _force_put(q, None)


def _force_put(q: "asyncio.Queue", ev) -> None:
    """Put on a capped queue, popping oldest until it fits. Used for the
    shutdown sentinel (must land) and for broadcasts (drop-oldest on overflow)."""
    while True:
        try:
            q.put_nowait(ev)
            return
        except asyncio.QueueFull:
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                return  # can't happen under QueueFull, but guard the race


_uvicorn_handle_exit = uvicorn.Server.handle_exit


def _handle_exit(self, sig, frame):
    if APP_LOOP is not None:
        APP_LOOP.call_soon_threadsafe(_wake_for_shutdown)
    _uvicorn_handle_exit(self, sig, frame)


uvicorn.Server.handle_exit = _handle_exit

# The last successful /api/load payload, so a fresh device — or a server restart
# — can restore the exact checkpoint/DiT/VAE/offload selections (not just "a model
# is loaded"). Persisted to disk so it survives a restart; the form is repopulated
# but the model itself is not reloaded (status stays "no model loaded").
_LAST_LOAD_PATH = _ROOT / "last_load.json"


def _read_last_load() -> Optional[dict]:
    try:
        return json.loads(_LAST_LOAD_PATH.read_text())
    except (OSError, ValueError):
        return None


def _write_last_load(form: dict) -> None:
    _atomic_write_text(_LAST_LOAD_PATH, json.dumps(form))


LAST_LOAD_FORM: Optional[dict] = _read_last_load()

# Persisted global settings (the settings panel). Round-tripped through the
# Settings model so an old file missing newer keys gets their defaults and
# unknown keys are dropped — forward/backward compatible across versions.
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
    """Fan one event out to every connected SSE client. Thread-safe: callable
    from the worker thread or a request handler. On a full subscriber queue we
    drop the oldest event (newest state wins) rather than block the worker."""
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
            raise _Cancelled  # unwinds the sampler; caught in the worker
        job.step, job.total = int(step), int(total)
        ev = {"type": "progress", "job": job.id, "step": int(step), "total": int(total)}
        if cells is not None:  # X/Y/Z: carry the 1-based current cell ("image N/total")
            ev["cell"], ev["cells"] = int(cell), int(cells)
        _push(ev)

    preview_last = [0.0]   # monotonic time of the last emitted preview (closure cell)

    def on_preview(image):
        # Throttle to PREVIEW_MIN_INTERVAL so a fast sampler doesn't flood the
        # event loop; the dropped frames are never the final result (the `done`
        # event always carries the full-quality saved image).
        now = time.monotonic()
        if now - preview_last[0] < PREVIEW_MIN_INTERVAL:
            return
        preview_last[0] = now
        # Cap resolution + encode lossy WebP on the worker thread so the data-URL
        # broadcast/serialized per client stays small.
        if max(image.size) > PREVIEW_MAX_SIDE:
            thumb = image.copy()
            thumb.thumbnail((PREVIEW_MAX_SIDE, PREVIEW_MAX_SIDE))
        else:
            thumb = image
        # Stash the last emitted preview so a shutdown mid-job can still save a
        # partial result (the daemon worker is killed on exit; without this the
        # running generation is lost with nothing on disk).
        job.last_preview = thumb
        buf = io.BytesIO()
        thumb.save(buf, format="WEBP", quality=80)
        data = "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()
        _push({"type": "preview", "job": job.id, "image": data})

    return on_progress, on_preview


def _enqueue(job: Job) -> None:
    # Priority insert: a higher-priority job runs before any lower-priority
    # queued job, while preserving FIFO order among equal priorities. Used so a
    # model ``load`` (priority 10) submitted behind a stack of generations runs
    # next instead of stalling the user who just switched models. It only jumps
    # the *queue* — the currently-running job still finishes (the GPU can't be
    # preempted mid-sampling); see IMPROVE.md #6.
    with QUEUE_LOCK:
        idx = len(QUEUE)
        for i, j in enumerate(QUEUE):
            if job.priority > j.priority:
                idx = i
                break
        QUEUE.insert(idx, job)
    QUEUE_WAKE.set()
    _broadcast_queue()


# ── extension platform ──────────────────────────────────────────────
# The loader is constructed with callables back into the server's queue and SSE
# stream so extensions can enqueue GPU-sharing jobs and broadcast events without
# importing server.py. EXTENSIONS is referenced by _render_index (script
# injection), the /api/ext/* routes, and the generation/load hooks below.

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
            QUEUE_WAKE.wait()       # sleep until something is enqueued
            QUEUE_WAKE.clear()
            continue
        if job.cancel.is_set():     # cancelled while still queued
            job.status = "cancelled"
            with QUEUE_LOCK:
                CURRENT = None
            _push({"type": "cancelled", "job": job.id})
            _broadcast_queue()
            continue
        job.status = "running"
        _broadcast_queue()
        try:
            result = job.run(job)
            job.status = "done"
            _push({"type": "done", "job": job.id, **result})
        except _Cancelled:
            job.status = "cancelled"
            _push({"type": "cancelled", "job": job.id})
        except Exception as e:  # noqa: BLE001 — surface any engine error to the UI
            job.status = "error"
            _push({"type": "error", "job": job.id, "message": _friendly_error(e)})
        finally:
            with QUEUE_LOCK:
                CURRENT = None
            _broadcast_queue()


def _friendly_error(e: Exception) -> str:
    """Map a raw engine exception to a user-actionable message.

    CUDA OOM is the common one and surfaces verbatim from torch otherwise —
    ``"CUDA out of memory. Tried to allocate …"`` with no guidance. Catch it
    specifically, free the cache so the next job isn't starved, and append a hint.
    """
    try:
        import torch  # local import — torch is heavy and optional for the tests
        if isinstance(e, torch.cuda.OutOfMemoryError):
            try: torch.cuda.empty_cache()
            except Exception: pass  # noqa: BLE001 — best-effort cleanup
            return (f"{e}  →  out of VRAM. Try offload=stream (Settings or the "
                    f"Load panel), a smaller width/height, fewer steps, or a "
                    f"smaller detailer/upscale tile.")
    except Exception:  # noqa: BLE001 — torch not available (e.g. CPU-only test env)
        pass
    return str(e)


# ── shutdown: save a partial result for the in-flight job (#4) ────────
# The worker is a daemon thread, so a Ctrl+C mid-sampling kills it abruptly and
# the running generation is lost — temp LoRAs un-applied, no file on disk. We
# can't preempt torch cleanly, but the last streamed latent→RGB preview is a PIL
# image in memory and safe to write from the exit path. atexit runs while daemon
# threads are still alive, so CURRENT + job.last_preview are readable. Best
# effort: a downscaled preview is far better than nothing for a 20-minute run
# that got Ctrl+C'd at step 28/30.

def _save_partial_preview(job: Optional[Job]) -> Optional[Path]:
    """Write ``job.last_preview`` to outputs/ tagged as a shutdown partial.

    Returns the saved path (so callers/tests can assert) or ``None`` if there's
    nothing to save. The file carries a ``parameters`` line that flags it as a
    partial so the gallery / metadata reader can distinguish it from a real
    generation (and so the user isn't confused by a low-res WebP-quality PNG
    that looks like a finished image)."""
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
        meta.add_text("parameters", f"PARTIAL — interrupted by shutdown. {info}")
        job.last_preview.save(out, pnginfo=meta)
        _invalidate_gallery_index()
        try:
            rel = str(out.relative_to(OUTPUTS_DIR))
        except ValueError:
            rel = str(out)
        log.warning("shutdown: saved partial preview for job %s to %s",
                    job.id, rel)
        return out
    except Exception as e:  # noqa: BLE001 — never let the exit path raise
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
# Runs before route handlers for every request. Order: cheap header checks
# first (Content-Length, Origin), then the auth gate which may read a cookie.
_PUBLIC_AUTH = {("GET", "/"), ("POST", "/api/auth/login"),
                ("GET", "/api/auth/status"), ("POST", "/api/auth/logout")}


@app.middleware("http")
async def _request_guard(request: Request, call_next):
    # 1. Global body-size cap (#5): /api/metadata/parse keeps its tighter
    #    MAX_UPLOAD_BYTES check inside the handler; this is the backstop for
    #    /api/generate, /api/upscale, etc. so a client can't stream GBs of
    #    base64 into the queue and OOM the process.
    cl = request.headers.get("content-length")
    if cl:
        try:
            if int(cl) > MAX_BODY_BYTES:
                return JSONResponse({"error": "request body too large"},
                                    status_code=413)
        except ValueError:
            pass
    # 2. CSRF / Origin allowlist (#2): block cross-origin state-changing
    #    requests. Same-origin fetches carry a matching Origin; curl sends none.
    if not origin_ok(request):
        return JSONResponse({"error": "cross-origin request blocked"},
                            status_code=403)
    # 3. Auth gate (#1): when enabled, every non-public path needs a valid
    #    cookie / bearer token / ?token=. Public paths are served so the user
    #    can reach the login page and submit the token.
    if AUTH.enabled and (request.method, request.url.path) not in _PUBLIC_AUTH:
        denied = AUTH.gate_response(request)
        if denied is not None:
            return denied
    return await call_next(request)


@app.on_event("startup")
async def _startup():
    # Capture the event loop so the worker thread can broadcast SSE events, then
    # start the single job worker.
    global APP_LOOP
    APP_LOOP = asyncio.get_running_loop()
    threading.Thread(target=_worker, daemon=True).start()
    # Load every enabled extension and mount its routes/statics into the app.
    # Done at startup (not import) so a manifest edit between imports and the
    # server actually starting is picked up, and so the app object exists.
    EXTENSIONS.load_all()
    EXTENSIONS.mount_into(app)
    n = sum(1 for e in EXTENSIONS.extensions.values() if e.module is not None)
    log.info("[startup] extensions: %d loaded, %d total", n,
             len(EXTENSIONS.extensions))
    # Purge aged gallery trash on boot (cheap; runs again on each delete).
    try:
        purged = _purge_trash()
        if purged:
            log.info("[startup] purged %d trash entries older than %d days",
                     purged, TRASH_RETENTION_DAYS)
    except Exception as e:  # noqa: BLE001
        log.warning("[startup] trash purge failed: %s", e)
    # Stamp the runtime environment into the log once so a "it broke" report
    # carries the engine/torch/CUDA context without the user having to dig.
    _log_runtime_env()


def _log_runtime_env() -> None:
    """Log torch / CUDA / GPU + engine version info for triage (IMPROVE.md #9)."""
    parts = [f"ui={md.UI_ID}", f"diff={md.DIFF_ID}"]
    try:
        import torch  # noqa: local import — heavy and optional in test envs
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
    except Exception as e:  # noqa: BLE001 — torch missing
        parts.append(f"torch=missing ({e})")
    log.info("[startup] runtime: %s", "  ".join(parts))


@app.get("/")
def index(request: Request):
    # Auth on + a ``?token=…`` query → validate, set the cookie, redirect to bare
    # "/" (so the token isn't left in browser history). Auth on + no cookie →
    # serve the login page (the app JS isn't exposed until the token is supplied).
    # Auth on + valid cookie → fall through to the app index. Auth off → index.
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


# ── auth endpoints (only meaningful when the gate is enabled) ───────

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
        "ui_id": md.UI_ID,
        "diff_id": md.DIFF_ID,
    }


@app.get("/api/status")
def api_status():
    return {"status": ENGINE.status_text(), "last_seed": ENGINE.last_seed}


def _do_load(p: LoadPayload) -> str:
    # pre_load: extensions can observe/adjust the load request before it runs.
    EXTENSIONS.run_hook("pre_load", payload=p)
    status = _do_load_impl(p)
    # post_load: notify extensions of the outcome (status starts with "Loaded"
    # on success, or "Model already loaded"; anything else is an error message).
    EXTENSIONS.run_hook("post_load", payload=p, status=status)
    return status


def _validate_load(p: LoadPayload) -> Optional[str]:
    """Fail-fast: return an error string if a named file isn't on disk, else None.

    A misspelled checkpoint would otherwise wait its turn in the queue (behind
    any running generation) before failing inside ``ENGINE.load_*``. Checking at
    submit time lets ``/api/load`` return a 400 immediately. Mirrors the
    "Select …" guards in ``_do_load_impl`` so the error message stays consistent.
    """
    def _missing(label: str, name: str, d: Path) -> Optional[str]:
        if not (d / name).is_file():
            return f"{label} not found: {name}"
        return None

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
    # Offload: explicit UI choice, else the per-family default. FLUX's ~23 GB
    # transformer OOMs under whole-module staging (full), so it always defaults to
    # "stream" block-streaming. Every family (SD/SDXL, FLUX, Anima) streams on a
    # very-low-VRAM card (the backend recommends "stream" ≤6 GB — fits the backbone
    # on ~4 GB); otherwise full. full/encoders/none/stream all work for all.
    _to_bundle = {"none": False, "full": True,
                  "encoders": "encoders", "stream": "stream"}
    if p.offload is None:
        stream = p.model_type == "FLUX" or ENGINE.recommended_offload() == "stream"
        offload = "stream" if stream else True
    else:
        offload = _to_bundle.get(p.offload, True)

    # VAE tiling preference (settings panel): "always" forces tiled decode; "auto"
    # lets the pipeline decide per decode from free VRAM. FLUX ignores this — it's
    # force-tiled below regardless.
    vae_tile_pref = SETTINGS.get("vae_tiling") == "always"

    if p.model_type == "Anima":
        for name in (p.dit, p.vae, p.te):
            if not name or name.startswith("("):
                return "Select all three Anima files"
        return ENGINE.load_anima(
            p.dit, p.vae, p.te,
            # vae_tile from the settings panel: "auto" lets the pipeline auto-decide
            # per decode via can_decode_untiled; "always" forces tiled (even at 1024²).
            offload=offload, vae_tile=vae_tile_pref,
            compile=p.compile, cuda_graphs=p.cuda_graphs,
        )
    if p.model_type == "FLUX":
        # All-in-one checkpoint takes precedence; otherwise load split files.
        if p.checkpoint and not p.checkpoint.startswith("("):
            return ENGINE.load_model(
                p.checkpoint, offload=offload, vae_tile=True,
                compile=p.compile, cuda_graphs=p.cuda_graphs,
            )
        for name in (p.dit, p.vae, p.te):
            if not name or name.startswith("("):
                return "Select an all-in-one checkpoint, or DiT + VAE + Text encoder"
        return ENGINE.load_flux(
            p.dit, p.vae, p.te, clip_name=p.clip,
            offload=offload, vae_tile=True,
            compile=p.compile, cuda_graphs=p.cuda_graphs,
        )
    if not p.checkpoint or p.checkpoint.startswith("("):
        return "Select a model"
    return ENGINE.load_model(
        p.checkpoint,
        # vae_tile from the settings panel; "auto" → SD/SDXL auto-decide per decode
        # via can_decode_untiled, "always" → force tiled.
        offload=offload, vae_tile=vae_tile_pref,
        compile=p.compile, cuda_graphs=p.cuda_graphs,
        channels_last=p.channels_last,
    )


@app.post("/api/load")
async def api_load(p: LoadPayload):
    """Queue a model load. Loading swaps the single in-memory model, so it runs
    on the same worker as generation — it simply waits its turn instead of being
    refused. On success the new load state is broadcast to every device.

    File existence is validated up front so a misspelled checkpoint returns a 400
    immediately instead of failing after waiting its turn in the queue."""
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


@app.post("/api/generate")
async def api_generate(p: GeneratePayload):
    def run(job: Job) -> dict:
        on_progress, on_preview = _make_callbacks(job)
        return _run_generation(p, on_progress, on_preview)
    job = Job("generate", f"{p.mode} {p.width}×{p.height} · {p.steps} steps", run)
    _enqueue(job)
    return {"job": job.id}


@app.post("/api/upscale")
async def api_upscale(p: UpscalePayload):
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
            seed=int(p.seed),
            teacache_thresh=float(p.teacache),
            teacache_use_coeffs=bool(p.teacache_calibrated),
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
        )
        out = _save_output(image, gen_kwargs, upscale=upscale_meta)
        rel = out.relative_to(OUTPUTS_DIR)
        return {
            "image_url": _output_url(out),
            "info": f"{unote}  |  saved to {rel}",
            "seed": ENGINE.last_seed,
        }
    job = Job("upscale", f"upscale {p.scale}x", run)
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
    # Apply the VAE-tiling choice to the already-loaded model so it takes effect
    # without a reload (future loads pick it up via _do_load).
    ENGINE.apply_vae_tiling(SETTINGS["vae_tiling"] == "always")
    return SETTINGS


# ── extension management ────────────────────────────────────────────

@app.get("/api/extensions")
def api_extensions():
    """List every discovered extension with its load state and web scripts."""
    return {"extensions": EXTENSIONS.list_serializable()}


@app.get("/api/extensions/web")
def api_extensions_web():
    """Script URLs to inject into the index page (one per enabled ext JS file).
    The frontend reads this on connect to know which extension scripts already
    loaded inline; the page itself is built server-side with the tags in place."""
    return {"scripts": EXTENSIONS.web_script_urls()}


@app.post("/api/extensions/install")
def api_extensions_install(p: InstallPayload):
    """Install an extension from a git URL or a .zip archive URL.

    Runs on the shared job worker (not the request threadpool) so a slow clone /
    pip doesn't tie up a request worker, the install is visible + cancellable in
    the queue panel, and it serializes with generation / loads (it imports
    Python modules and may pip install — racing the GPU worker is bad). Returns
    a job id; the terminal ``done`` event carries the new extension's record so
    the frontend can refresh the panel. See IMPROVE.md #8."""
    # Fast-fail scheme/SSRF validation up front so a malformed URL returns 400
    # immediately instead of enqueueing a job that errors a moment later.
    from extensions import _validate_install_url
    try:
        _validate_install_url(p.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    def run(job: Job) -> dict:
        ext = EXTENSIONS.install(p.url, install_pip_deps=p.install_pip_deps)
        EXTENSIONS.mount_into(app)  # attach the new extension's routes/statics
        return {"extension": ext.to_dict()}
    job = Job("install", f"install {p.url}", run)
    _enqueue(job)
    return {"job": job.id}


@app.post("/api/extensions/toggle")
def api_extensions_toggle(p: TogglePayload):
    """Enable or disable an extension. Disabling unloads its hooks/routes so they
    stop firing until re-enabled (no server restart needed for the backend; the
    frontend script tags refresh on the next page load)."""
    ext = EXTENSIONS.set_enabled(p.name, p.enabled)
    if p.enabled:
        EXTENSIONS.mount_into(app)  # a just-enabled extension's routes need attaching
    return {"extension": ext.to_dict()}


@app.post("/api/extensions/reload")
def api_extensions_reload(name: str):
    """Re-import an extension's entry module — handy while developing one. Drops
    its old hooks/routes/statics first so nothing doubles up."""
    EXTENSIONS.reload_one(name)
    EXTENSIONS.mount_into(app)  # attach any routes/statics the reload re-added
    ext = EXTENSIONS.extensions.get(name)
    if ext is None:
        raise HTTPException(status_code=404, detail="extension not found")
    return {"extension": ext.to_dict()}


@app.post("/api/extensions/uninstall")
def api_extensions_uninstall(p: UninstallPayload):
    """Remove an extension's folder and drop its hooks/routes. Its persisted
    enabled/state entries are cleared too."""
    EXTENSIONS.uninstall(p.name)
    return {"uninstalled": p.name}


@app.post("/api/cancel")
def api_cancel(p: CancelPayload):
    """Cancel a job by id. A running job aborts at its next sampling step; a
    still-queued job is dropped from the queue. ``job=None`` targets whatever is
    currently running."""
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
    if queued:  # never ran — report its cancellation now
        target.status = "cancelled"
        _push({"type": "cancelled", "job": target.id})
        _broadcast_queue()
    return {"cancelling": True}


@app.get("/api/events")
async def api_events(request: Request):
    """Shared SSE stream: queue changes, progress, previews, and model status
    are broadcast here to every connected device."""
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
                if ev is None:           # shutdown sentinel — let the server exit
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
    """List gallery images, optionally filtered by a metadata substring.

    ``q`` matches case-insensitively across the parsed prompt, negative prompt,
    model name, sampler and scheduler fields. When empty, every output is
    returned (the existing behaviour). Search uses a cached metadata index that
    rebuilds when the outputs directory's newest folder mtime advances — so a
    newly saved image shows up on the next search without a manual refresh."""
    query = (q or "").strip().lower()
    if not query:
        return {"images": [
            {
                "url": _output_url(f),
                "name": f.name,
                "path": f.relative_to(OUTPUTS_DIR).as_posix(),
                "date": f.parent.name,
            }
            for f in scan_outputs()
        ]}
    return {"images": _gallery_search(query)}


# ── gallery search index ────────────────────────────────────────────
# One pass opens every output PNG to parse its AUTO1111 metadata — too costly to
# repeat per keystroke, so the result is cached and only rebuilt when the outputs
# dir's newest folder mtime advances (a saved image bumps its date folder's
# mtime, invalidating the cache). The rebuild is guarded by a lock so two
# concurrent searches don't each re-open every PNG and race to assign the index
# (last writer wins, wasted work). Saves and deletes invalidate explicitly so a
# same-second save on a 1s-mtime filesystem (ext4 default) still shows up.
_GALLERY_INDEX: Optional[list] = None
_GALLERY_INDEX_KEY: float = 0.0
_GALLERY_INDEX_LOCK = threading.Lock()


def _invalidate_gallery_index() -> None:
    global _GALLERY_INDEX, _GALLERY_INDEX_KEY
    with _GALLERY_INDEX_LOCK:
        _GALLERY_INDEX = None
        _GALLERY_INDEX_KEY = 0.0


def _gallery_index() -> list:
    global _GALLERY_INDEX, _GALLERY_INDEX_KEY
    try:
        newest = max(
            (d.stat().st_mtime for d in OUTPUTS_DIR.iterdir() if d.is_dir()),
            default=0.0,
        )
    except OSError:
        newest = 0.0
    with _GALLERY_INDEX_LOCK:
        if _GALLERY_INDEX is not None and newest <= _GALLERY_INDEX_KEY:
            return _GALLERY_INDEX
        index: list = []
        for f in scan_outputs():
            raw = md.read_png_metadata(str(f))
            fields = md.parse_metadata(raw) if raw else {}
            index.append({
                "path": f.relative_to(OUTPUTS_DIR).as_posix(),
                "name": f.name,
                "date": f.parent.name,
                "prompt": str(fields.get("prompt", "")),
                "neg": str(fields.get("negative_prompt", "")),
                "model": str(fields.get("model", "")),
                "sampler": str(fields.get("sampler", "")),
                "scheduler": str(fields.get("scheduler", "")),
            })
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
            out.append({
                "url": f"/outputs/{entry['path']}",
                "name": entry["name"],
                "path": entry["path"],
                "date": entry["date"],
            })
    return out


@app.get("/api/thumb")
def api_thumb(path: str):
    """Serve a small cached thumbnail for a gallery image (path under outputs/).

    The grid loads hundreds of these instead of the full ~1 MB PNGs. Resized on
    the first request and cached under .cache/thumbs/ (outside outputs/, so
    ``scan_outputs`` never lists them); every later request is served from disk."""
    target = (OUTPUTS_DIR / path).resolve()
    if OUTPUTS_DIR.resolve() not in target.parents or not target.is_file():
        raise HTTPException(status_code=404)
    cache = (_THUMBS_DIR / target.relative_to(OUTPUTS_DIR.resolve())).with_suffix(".webp")
    if not cache.is_file():
        cache.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(target) as im:
            im = im.convert("RGB")
            im.thumbnail((THUMB_MAX, THUMB_MAX))
            im.save(cache, "WEBP", quality=80)
    return FileResponse(cache, media_type="image/webp")


@app.delete("/api/gallery")
def api_gallery_delete(path: str):
    """Soft-delete a gallery image: move it to ``outputs/.trash/`` (recoverable
    by hand) and drop its cached thumbnail.

    A hard ``unlink()`` is racy with the two-click confirm on a slow connection —
    a double-click costs the user real work. The trash is purged of entries
    older than ``TRASH_RETENTION_DAYS`` on each call (cheap). Path is scoped
    under ``outputs/`` with the same traversal guard as ``/api/thumb``; the
    trashed name is prefixed with a timestamp so repeated deletes of same-named
    files don't clobber each other. On success the cached search index is
    invalidated so the next search reflects the deletion."""
    target = (OUTPUTS_DIR / path).resolve()
    outputs_root = OUTPUTS_DIR.resolve()
    if outputs_root not in target.parents or not target.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    _TRASH_DIR.mkdir(parents=True, exist_ok=True)
    trashed = _TRASH_DIR / f"{int(time.time())}_{target.name}"
    try:
        shutil.move(str(target), str(trashed))
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Could not delete: {e}")
    # Drop the cached thumbnail (may not exist if never viewed).
    cache = (_THUMBS_DIR / target.relative_to(outputs_root)).with_suffix(".webp")
    if cache.is_file():
        try: cache.unlink()
        except OSError: pass
    # Invalidate the search index so the next /api/gallery?q= reflects the
    # deletion. Rebuilding is cheap (50 ms for ~1k images) and only happens
    # when search is actually used next.
    _invalidate_gallery_index()
    # Purge aged trash entries on the way out — one iterdir + mtime check.
    try:
        _purge_trash()
    except Exception as e:  # noqa: BLE001 — never let cleanup fail the delete
        log.warning("trash purge failed: %s", e)
    return {"deleted": path, "trashed": trashed.name}


@app.get("/api/metadata")
def api_metadata(path: str):
    """Raw + workspace-normalised metadata for a gallery image (path under outputs/)."""
    target = (OUTPUTS_DIR / path).resolve()
    if OUTPUTS_DIR.resolve() not in target.parents or not target.is_file():
        return {"raw": "", "fields": {}}
    raw = md.read_png_metadata(str(target))
    fields = md.workspace_fields(md.parse_metadata(raw))
    return {"raw": raw, "fields": fields}


@app.post("/api/metadata/parse")
async def api_metadata_parse(file: UploadFile = File(...)):
    """Dump every PNG chunk + parsed AUTO1111/ComfyUI views for an uploaded image."""
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
    """Parse a pasted AUTO1111-style ``parameters`` string into workspace
    fields — the same path as a gallery image, but for text dropped straight
    into the prompt box (SD WebUI's read-generation-parameters)."""
    return {"fields": md.workspace_fields(md.parse_metadata(p.text))}


# Static mounts (declared last so /api routes win).
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)  # StaticFiles errors if missing (fresh install)
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")
app.mount("/static", StaticFiles(directory=_STATIC), name="static")
