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
import base64
import io
import itertools
import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, List, Optional

import uvicorn
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from engine import ENGINE, SAMPLERS_SD, SAMPLERS_ANIMA, SAMPLERS_FLUX, SCHEDULERS_SD, SCHEDULERS_ANIMA, SCHEDULERS_FLUX
from utils import (
    OUTPUTS_DIR, detector_path,
    scan_checkpoints, scan_loras, scan_diffusion_models,
    scan_vae, scan_text_encoders, scan_detectors, scan_outputs, next_output_path,
)
from xyz_grid import generate_xyz_grid, PARAM_TYPES as XYZ_PARAM_TYPES
import metadata as md

_ROOT = Path(__file__).resolve().parent.parent
_STATIC = _ROOT / "static"

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
    steps: int = 25
    cfg: float = 6.0
    seed: int = -1
    width: int = 1024
    height: int = 1024
    strength: float = 0.6
    shift: float = 3.0
    input_image: Optional[str] = None   # base64 / data-URL
    mask_image: Optional[str] = None
    preview: bool = True                 # stream live latent previews while sampling

    # ── detailer (ADetailer-style passes run after the main image) ──
    # Each entry is one detection model + its own prompt; the rest is shared.
    detail_enabled: bool = False
    detail_models: List[DetailerModel] = []
    detail_neg: str = ""
    detail_confidence: float = 0.3
    detail_strength: float = 0.4
    detail_dilation: int = 4
    detail_padding: int = 32
    detail_blur: int = 4
    detail_max: int = 0                  # 0 = all detections


class CalibratePayload(BaseModel):
    prompt: str = ""
    neg: str = ""
    steps: int = 12
    cfg: float = 4.0
    seed: int = 0
    width: int = 1024
    height: int = 1024
    shift: float = 3.0
    grid: int = 80                    # dense teacher-trajectory candidate count (K)


class XYZPayload(BaseModel):
    prompt: str = ""
    neg: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 25
    cfg: float = 6.0
    sampler: str = "dpmpp_2m"
    scheduler: str = "karras"
    seed: int = -1
    shift: float = 3.0
    x_type: str = "None"
    x_vals: str = ""
    y_type: str = "None"
    y_vals: str = ""
    z_type: str = "None"
    z_vals: str = ""


class CancelPayload(BaseModel):
    job: Optional[int] = None        # None = cancel whatever is currently running


class ParseTextPayload(BaseModel):
    text: str = ""


# ── helpers ─────────────────────────────────────────────────────────

def _decode_image(data: str) -> Image.Image:
    if data.strip().startswith("data:") and "," in data:
        data = data.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")


def _output_url(path: Path) -> str:
    return f"/outputs/{path.relative_to(OUTPUTS_DIR).as_posix()}"


def _save_output(image: Image.Image, gen_kwargs: dict) -> Path:
    """Save an image to outputs/ with AUTO1111 metadata; return its path.

    Used for single generations and for each individual X/Y/Z cell.
    """
    out = next_output_path(ENGINE.last_seed)
    meta = PngInfo()
    meta_kwargs = {k: v for k, v in gen_kwargs.items() if k != "progress_callback"}
    meta.add_text("parameters", md.format_metadata(meta_kwargs, ENGINE))
    image.save(out, pnginfo=meta)
    return out


# ── generation (ported from the old _generate_with_loras) ───────────

def _run_generation(p: GeneratePayload, on_progress: Callable[[int, int], None],
                    on_preview: Optional[Callable] = None) -> dict:
    if not ENGINE.loaded_name:
        raise RuntimeError("Load a model first")

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
            progress_callback=on_progress,
            preview_callback=on_preview if p.preview else None,
        )

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
                mask_image=_decode_image(p.mask_image),
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
        elapsed = time.perf_counter() - t0

        # Detailer: run each stacked detection model in sequence, feeding the
        # refined image into the next pass. Per-model prompt; the rest is shared.
        detail_info = ""
        active = [dm for dm in p.detail_models
                  if dm.model and not dm.model.startswith("(")] if p.detail_enabled else []
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
                        seed=int(p.seed), progress_callback=on_progress,
                        preview_callback=on_preview if p.preview else None,
                    )
                    notes.append(f"{dm.model}: {dnote.replace('Detailer: ', '')}")
                except Exception as e:  # noqa: BLE001 — keep current image on a pass failure
                    notes.append(f"{dm.model} error: {e}")
            detail_info = "  |  detailer [" + "; ".join(notes) + "]"

        out = _save_output(image, gen_kwargs)
        rel = out.relative_to(OUTPUTS_DIR)
        return {
            "image_url": _output_url(out),
            "info": f"{lora_info}{info}  |  inference: {elapsed:.2f}s{detail_info}  |  saved to {rel}",
            "seed": ENGINE.last_seed,
        }
    finally:
        if loras:
            ENGINE.clear_temp_loras()


def _run_xyz(p: XYZPayload, on_progress: Callable[[int, int], None]) -> dict:
    if not ENGINE.loaded_name:
        raise RuntimeError("Load a model first")
    base_kwargs = dict(
        prompt=p.prompt, negative_prompt=p.neg,
        width=int(p.width), height=int(p.height),
        steps=int(p.steps), cfg_scale=float(p.cfg),
        sampler=p.sampler, scheduler=p.scheduler,
        seed=int(p.seed), shift=float(p.shift),
    )
    grids, info = generate_xyz_grid(
        base_kwargs,
        p.x_type, p.x_vals, p.y_type, p.y_vals, p.z_type, p.z_vals,
        progress_callback=on_progress,
        save_callback=_save_output,
    )
    # base_kwargs was mutated in-place by generate_xyz_grid (prompt cleaned,
    # base seed resolved), so it carries the right params for the grid metadata.
    urls = []
    for grid in grids:
        out = _save_output(grid, base_kwargs)
        urls.append(_output_url(out))
    return {"grids": urls, "info": info}


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


# ── job queue + SSE broadcast ───────────────────────────────────────
# A single background worker runs jobs one at a time (the GPU can only do one),
# so the worker thread itself is the serialization — no lock needed. Every
# device subscribes to one shared SSE stream and sees the same queue + progress.

_job_ids = itertools.count(1)


class Job:
    def __init__(self, kind: str, label: str, run: Callable[["Job"], dict]):
        self.id = next(_job_ids)
        self.kind = kind            # generate | xyz | calibrate | load
        self.label = label          # short human description for the queue list
        self.run = run              # run(job) -> result dict; may raise _Cancelled
        self.status = "queued"      # queued | running | done | error | cancelled
        self.cancel = threading.Event()
        self.step = 0               # live progress, for snapshots on (re)connect
        self.total = 0


QUEUE: "deque[Job]" = deque()
QUEUE_LOCK = threading.Lock()
QUEUE_WAKE = threading.Event()
CURRENT: Optional[Job] = None

# One asyncio.Queue per connected SSE client; the worker fans events out to all.
SUBSCRIBERS: "set[asyncio.Queue]" = set()
APP_LOOP: Optional[asyncio.AbstractEventLoop] = None

# The /api/events SSE streams are long-lived, so uvicorn's graceful shutdown would
# wait on them forever — Ctrl+C appears to hang until a second, forced Ctrl+C.
# uvicorn calls Server.handle_exit the instant a signal arrives (before it starts
# waiting for connections to close), so wrap it to set SHUTDOWN and wake every
# stream with a sentinel; each gen() then returns and the server exits cleanly.
SHUTDOWN = asyncio.Event()


def _wake_for_shutdown() -> None:
    SHUTDOWN.set()
    for q in list(SUBSCRIBERS):
        q.put_nowait(None)


_uvicorn_handle_exit = uvicorn.Server.handle_exit


def _handle_exit(self, sig, frame):
    if APP_LOOP is not None:
        APP_LOOP.call_soon_threadsafe(_wake_for_shutdown)
    _uvicorn_handle_exit(self, sig, frame)


uvicorn.Server.handle_exit = _handle_exit

# The last successful /api/load payload, so a fresh device can restore the exact
# checkpoint/DiT/VAE/offload selections (not just "a model is loaded").
LAST_LOAD_FORM: Optional[dict] = None


def _push(ev: dict) -> None:
    """Fan one event out to every connected SSE client. Thread-safe: callable
    from the worker thread or a request handler."""
    loop = APP_LOOP
    if loop is None:
        return
    def deliver():
        for q in list(SUBSCRIBERS):
            q.put_nowait(ev)
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
    def on_progress(step, total):
        if job.cancel.is_set():
            raise _Cancelled  # unwinds the sampler; caught in the worker
        job.step, job.total = int(step), int(total)
        _push({"type": "progress", "job": job.id, "step": int(step), "total": int(total)})

    def on_preview(image):
        # Encode the approx preview to a PNG data-URL on the worker thread, then
        # hand the small string to the broadcaster.
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        data = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        _push({"type": "preview", "job": job.id, "image": data})

    return on_progress, on_preview


def _enqueue(job: Job) -> None:
    with QUEUE_LOCK:
        QUEUE.append(job)
    QUEUE_WAKE.set()
    _broadcast_queue()


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
            _push({"type": "error", "job": job.id, "message": str(e)})
        finally:
            with QUEUE_LOCK:
                CURRENT = None
            _broadcast_queue()


# ── app ─────────────────────────────────────────────────────────────

app = FastAPI(title="Diffucore")
print(f"[startup] offload default '{ENGINE.recommended_offload()}' "
      f"(device: {ENGINE.device})")


@app.on_event("startup")
async def _startup():
    # Capture the event loop so the worker thread can broadcast SSE events, then
    # start the single job worker.
    global APP_LOOP
    APP_LOOP = asyncio.get_running_loop()
    threading.Thread(target=_worker, daemon=True).start()


@app.get("/")
def index():
    return FileResponse(_STATIC / "index.html")


@app.get("/api/models")
def api_models():
    return {
        "checkpoints": scan_checkpoints(),
        "dits": scan_diffusion_models(),
        "vaes": scan_vae(),
        "tes": scan_text_encoders(),
        "loras": scan_loras(),
        "detailers": scan_detectors(),
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
    # Offload: explicit UI choice, else the per-family default. FLUX's ~23 GB
    # transformer OOMs under whole-module staging (full), so it defaults to
    # "stream" block-streaming — the only mode that fits a 24 GB card. "stream"
    # is FLUX-only (it streams the DiT blocks); full/encoders/none work for all.
    if p.offload is None:
        offload = "stream" if p.model_type == "FLUX" else True
    else:
        offload = {"none": False, "full": True,
                   "encoders": "encoders", "stream": "stream"}.get(p.offload, True)

    if p.model_type == "Anima":
        for name in (p.dit, p.vae, p.te):
            if not name or name.startswith("("):
                return "Select all three Anima files"
        return ENGINE.load_anima(
            p.dit, p.vae, p.te,
            # vae_tile=False → let the pipeline auto-decide per decode via
            # can_decode_untiled (forcing True always-tiles, even at 1024²).
            offload=offload, vae_tile=False,
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
        # vae_tile=False → SD/SDXL auto-decide per decode via can_decode_untiled.
        offload=offload, vae_tile=False,
        compile=p.compile, cuda_graphs=p.cuda_graphs,
        channels_last=p.channels_last,
    )


@app.post("/api/load")
async def api_load(p: LoadPayload):
    """Queue a model load. Loading swaps the single in-memory model, so it runs
    on the same worker as generation — it simply waits its turn instead of being
    refused. On success the new load state is broadcast to every device."""
    def run(job: Job) -> dict:
        global LAST_LOAD_FORM
        status = _do_load(p)
        if status.startswith(("Loaded", "Model already loaded")):
            LAST_LOAD_FORM = p.dict()
        _push({"type": "status", **_state_payload()})
        return {"status": status, "loaded": bool(ENGINE.loaded_name)}

    job = Job("load", f"load {p.model_type}", run)
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


@app.post("/api/xyz")
async def api_xyz(p: XYZPayload):
    def run(job: Job) -> dict:
        on_progress, _ = _make_callbacks(job)
        return _run_xyz(p, on_progress)
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
    q: asyncio.Queue = asyncio.Queue()
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
def api_gallery():
    return {"images": [
        {
            "url": _output_url(f),
            "name": f.name,
            "path": f.relative_to(OUTPUTS_DIR).as_posix(),
            "date": f.parent.name,
        }
        for f in scan_outputs()
    ]}


@app.get("/api/metadata")
def api_metadata(path: str):
    """Raw + workspace-normalised metadata for a gallery image (path under outputs/)."""
    target = (OUTPUTS_DIR / path).resolve()
    if OUTPUTS_DIR.resolve() not in target.parents or not target.is_file():
        return {"raw": "", "fields": {}}
    raw = md.read_png_metadata(str(target))
    return {"raw": raw, "fields": md.workspace_fields(md.parse_metadata(raw))}


@app.post("/api/metadata/parse")
async def api_metadata_parse(file: UploadFile = File(...)):
    """Dump every PNG chunk + parsed AUTO1111/ComfyUI views for an uploaded image."""
    data = await file.read()
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
