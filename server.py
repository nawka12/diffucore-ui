"""FastAPI web layer for Diffucore.

Wraps the framework-agnostic ``ENGINE`` singleton. Generation is blocking torch
code, so it runs in a threadpool while sampling progress is streamed back to the
browser as newline-delimited JSON (one job at a time, guarded by ``GEN_LOCK``).
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from engine import ENGINE, SAMPLERS, SCHEDULERS_SD, SCHEDULERS_ANIMA, SCHEDULERS_FLUX
from utils import (
    OUTPUTS_DIR,
    scan_checkpoints, scan_loras, scan_diffusion_models,
    scan_vae, scan_text_encoders, scan_outputs, next_output_path,
)
from xyz_grid import generate_xyz_grid, PARAM_TYPES as XYZ_PARAM_TYPES
import metadata as md

_ROOT = Path(__file__).resolve().parent
_STATIC = _ROOT / "static"

# Only one generation may touch the GPU at a time.
GEN_LOCK = threading.Lock()


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

def _run_generation(p: GeneratePayload, on_progress: Callable[[int, int], None]) -> dict:
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
        )

        if p.mode == "i2i":
            if not p.input_image:
                raise RuntimeError("Provide an input image")
            gen_kwargs = dict(
                prompt=clean_prompt, input_image=_decode_image(p.input_image),
                strength=float(p.strength), **common,
            )
            gen_fn = ENGINE.generate_i2i
        elif p.mode == "inpaint":
            if not p.input_image or not p.mask_image:
                raise RuntimeError("Provide both an input image and a mask")
            gen_kwargs = dict(
                prompt=clean_prompt, input_image=_decode_image(p.input_image),
                mask_image=_decode_image(p.mask_image),
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

        out = _save_output(image, gen_kwargs)
        rel = out.relative_to(OUTPUTS_DIR)
        return {
            "image_url": _output_url(out),
            "info": f"{lora_info}{info}  |  inference: {elapsed:.2f}s  |  saved to {rel}",
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


# ── streaming wrapper ───────────────────────────────────────────────

async def _stream_job(work_fn: Callable[[Callable[[int, int], None]], dict]) -> StreamingResponse:
    """Run ``work_fn`` on a worker thread; stream progress + result as ndjson."""
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()

    def on_progress(step, total):
        loop.call_soon_threadsafe(
            q.put_nowait, {"type": "progress", "step": int(step), "total": int(total)},
        )

    def worker():
        if not GEN_LOCK.acquire(blocking=False):
            loop.call_soon_threadsafe(
                q.put_nowait, {"type": "error", "message": "Busy — another job is running"},
            )
            loop.call_soon_threadsafe(q.put_nowait, None)
            return
        try:
            result = work_fn(on_progress)
            loop.call_soon_threadsafe(q.put_nowait, {"type": "done", **result})
        except Exception as e:  # noqa: BLE001 — surface any engine error to the UI
            loop.call_soon_threadsafe(q.put_nowait, {"type": "error", "message": str(e)})
        finally:
            GEN_LOCK.release()
            loop.call_soon_threadsafe(q.put_nowait, None)

    loop.run_in_executor(None, worker)

    async def events():
        while True:
            ev = await q.get()
            if ev is None:
                break
            yield json.dumps(ev) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


# ── app ─────────────────────────────────────────────────────────────

app = FastAPI(title="Diffucore")


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
        "samplers": SAMPLERS,
        "schedulers_sd": SCHEDULERS_SD,
        "schedulers_anima": SCHEDULERS_ANIMA,
        "schedulers_flux": SCHEDULERS_FLUX,
        "xyz_param_types": XYZ_PARAM_TYPES,
        "status": ENGINE.status_text(),
        "last_seed": ENGINE.last_seed,
        "ui_id": md.UI_ID,
        "diff_id": md.DIFF_ID,
    }


@app.get("/api/status")
def api_status():
    return {"status": ENGINE.status_text(), "last_seed": ENGINE.last_seed}


@app.post("/api/load")
async def api_load(p: LoadPayload):
    def work():
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
                offload=offload, vae_tile=True,
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
            offload=offload, vae_tile=True,
            compile=p.compile, cuda_graphs=p.cuda_graphs,
            channels_last=p.channels_last,
        )

    def locked_work():
        # Loading swaps the single in-memory model; never do it while a generation
        # is using that model. Same lock the generators take — refuse if it's held.
        if not GEN_LOCK.acquire(blocking=False):
            return "Busy — finish the running generation first"
        try:
            return work()
        finally:
            GEN_LOCK.release()

    try:
        status = await asyncio.to_thread(locked_work)
    except Exception as e:  # noqa: BLE001
        status = f"Error: {e}"
    return {"status": status}


@app.post("/api/generate")
async def api_generate(p: GeneratePayload):
    return await _stream_job(lambda cb: _run_generation(p, cb))


@app.post("/api/xyz")
async def api_xyz(p: XYZPayload):
    return await _stream_job(lambda cb: _run_xyz(p, cb))


@app.get("/api/gallery")
def api_gallery():
    return {"images": [
        {
            "url": _output_url(f),
            "name": f.name,
            "path": f.relative_to(OUTPUTS_DIR).as_posix(),
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


# Static mounts (declared last so /api routes win).
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")
app.mount("/static", StaticFiles(directory=_STATIC), name="static")
