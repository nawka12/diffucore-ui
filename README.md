# Diffucore UI

**A web frontend for the Diffucore diffusion inference engine.**

Point it at your checkpoints, pick a prompt, and generate — a darkroom-themed
interface with a unified txt2img / img2img / inpaint workspace, an X/Y/Z
parameter-sweep mode, and a gallery that recycles past generations' metadata
back into the workspace.

![status](https://img.shields.io/badge/status-under%20development-orange)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-Apache--2.0-green)

> ⚠️ **Under active development.** Diffucore UI is pre-1.0 and moving quickly —
> HTTP endpoints, APIs, and UI layout may change between commits without notice,
> and rough edges are expected. Use it and report issues, but don't rely on
> stability yet.

```bash
python backend/app.py    # serve on http://127.0.0.1:7860
```

Then open `http://localhost:7860` in your browser. The backend is FastAPI +
Uvicorn; the frontend is plain HTML/CSS/JS with Alpine.js (no build step).

## Highlights

- **Four model families, one interface** — Stable Diffusion 1.5, SDXL,
  **Anima** (a 2 B DiT built on Cosmos-Predict2), and **FLUX** (FLUX.1 and
  FLUX.2 Klein). All four do txt2img, img2img, and inpaint — Anima and FLUX use
  soft, latent-mask inpaint (no dedicated inpaint model). Switch between them
  from the model bar.
- **Unified Generate workspace** — one shared control panel with a
  txt2img / img2img / inpaint mode toggle; switching modes keeps your prompt and
  settings. Full sampler / scheduler / steps / CFG / seed controls.
- **X/Y/Z parameter sweep** — toggle it on inside txt2img to render a comparison
  grid across samplers, schedulers, steps, CFG, or seed. The assembled grid and
  every individual cell are saved to `outputs/`, sharing one seed for a fair
  comparison.
- **Prompt-based LoRA loading** — embed `<lora:name:mult>` directly in your
  prompt to load adapters on the fly.
- **Detailer** — an ADetailer-style toggle that detects faces/hands with a YOLO
  model and inpaints each region at native resolution after generation. Works on
  UNet (SD/SDXL) and DiT (Anima, FLUX) backbones.
- **Live preview** — watch the image form during sampling. A fast latent→RGB
  approximation (no VAE decode) streams a rough preview each step; toggle it off
  in the Generate view. SD/SDXL and Anima.
- **11 samplers, multiple schedulers** — Euler, Heun, DPM++ family, ER-SDE,
  SECANT; Karras, exponential, sgm_uniform, flow, and more.
- **Gallery with metadata round-trip** — every generated image saves its full
  generation parameters as PNG metadata. Browse past outputs grouped by date
  (phone-gallery style) in a swipeable fullscreen carousel and load any
  generation's settings back into the Generate view.
- **Metadata reader** — drop in any PNG to inspect its AUTO1111 / Forge or
  ComfyUI parameters and send them straight to txt2img.
- **Anima auto-defaults** — switching to Anima mode sets sampler / steps / CFG
  to sensible values (er_sde, 30, 4.0) automatically.
- **Seed recycle & randomize** — reuse the last seed or roll a new one with one
  click.
- **Selectable CPU offload & tiled VAE** — the offload default is auto-picked
  from your GPU's VRAM on startup (24 GB → keep everything resident, 16 GB → park
  encoders, ≤12 GB → full offload), and you can override it per load
  (full / encoders / none, plus `stream` for FLUX) to fit the model on your GPU.
  Tiled VAE decode triggers automatically when a full-resolution decode
  wouldn't fit free VRAM, keeping large images within budget.
- **Live progress** — sampling step/total streams to a real progress bar as the
  image is generated.
- **Multi-device & job queue** — drive it from several devices at once. Jobs
  (generate, sweeps, calibrations, and model loads) run one at a time through a
  shared FIFO queue, and one live event stream keeps every device in sync —
  queue contents, progress, previews, and which model is loaded. A second device
  (or a refresh) picks up the already-loaded model without reloading weights, and
  any job can be cancelled from any device.
- **Custom darkroom theme** — warm amber safelight aesthetics on a hand-rolled
  dark UI, Fraunces serif + Inter + JetBrains Mono fonts.

## Install

```bash
# 1. Clone with submodules
git clone --recurse-submodules https://github.com/nawka12/diffucore-ui.git
cd diffucore-ui

# 2. Create a venv and install dependencies
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e ./diffucore

# 3. Install the CUDA build of torch for your GPU
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Or run `./setup.sh` to do all of the above automatically. On Windows, run
`setup.bat` instead (double-click it or run it from a terminal).

### Update

To pull the latest UI code, sync the `diffucore` submodule to its pinned
revision, and refresh dependencies:

```bash
./update.sh          # Linux / macOS
```

```bat
update.bat           REM Windows
```

Both reuse the existing `.venv` — run setup first if you don't have one yet.

## Usage

### Place your models

```
models/
├── checkpoints/             # SD/SDXL .safetensors or .ckpt (+ FLUX all-in-one)
├── diffusion-models/        # Anima / FLUX DiT .safetensors
├── vae/                     # Anima / FLUX VAE .safetensors
├── text-encoders/           # Anima / FLUX text encoders .safetensors
├── loras/                   # LoRA adapters (.safetensors)
└── detailers/               # YOLO detection models for the detailer (.pt)
```

The detailer needs `ultralytics` (installed via `requirements.txt`) and at least
one YOLO model in `detailers/` — e.g. ADetailer's `face_yolov8n.pt` / `hand_yolov8n.pt`.

### Start the UI

```bash
./launch.sh          # Linux / macOS
```

```bat
launch.bat           REM Windows
```

Or, manually:

```bash
source .venv/bin/activate
python backend/app.py
```

By default the UI binds to `127.0.0.1` (localhost only). Flags passed to
`launch.sh` are forwarded to `backend/app.py`:

```bash
./launch.sh --listen        # bind 0.0.0.0 — reachable from other machines on the network
./launch.sh --port 8000     # serve on a different port (default: 7860)
./launch.sh --listen --port 8000
./launch.sh --share         # public link via a Cloudflare quick tunnel
```

With `--listen`, several devices can use the UI at once. They share one job
queue and one live event stream, so any device sees the running queue and
progress, and a device that opens the page after a model is loaded starts
already loaded — no reload. Generations from different devices simply queue up
and run one at a time.

With `--share`, the UI is exposed over a public `trycloudflare.com` URL (printed
to the console) so you can reach it from anywhere — no Cloudflare account or
login needed. The `cloudflared` binary is used from your `PATH` if present,
otherwise downloaded once and cached in `.cloudflared/`. The tunnel closes when
you stop the server. Anyone with the link can reach your UI, so treat it as
public.

### Load a model

1. Select **SD/SDXL**, **Anima**, or **FLUX** from the top-bar radio.
2. Pick your checkpoint files from the dropdowns. FLUX takes either an all-in-one
   checkpoint or split DiT / VAE / text-encoder files (CLIP-L for FLUX.1 only).
3. Click **Load** — the status bar shows model info and VRAM usage.

### Generate

In the **Generate** view, pick a mode (**txt2img**, **img2img**, or
**inpaint**), enter a prompt, adjust your sampler / steps / CFG, and click
**Generate**. Progress streams to a live bar, a rough **live preview** updates in
the canvas as it samples (toggle it off beside the button), and the final result
lands in the panel on the right.

For **img2img** and **inpaint**, drag an image onto the input zone (or click to
browse); in **inpaint**, paint over the region to repaint — tune the brush size
or clear the mask to start over.

LoRAs can be activated inline: `a castle in autumn, <lora:autumn_style:0.8>`.

> **The first image is slower** — and so is the first image at each new
> resolution. The first generation pays a one-time GPU warmup (CUDA kernel
> loading + cuDNN autotune) that later images reuse, so subsequent images at the
> same size are noticeably faster. This is expected, not a stall.

### Working with Anima

Anima is an LLM-conditioned DiT, and its `shift = 3` flow schedule makes a few
settings behave differently from SD/SDXL. Three things worth knowing — none are
bugs, just how the model responds:

- **Write detailed prompts.** Anima conditions on a language model (Qwen3 +
  T5-XXL) trained on long, descriptive captions, so short tag-style prompts give
  weak guidance and drifty, low-quality results. Describe the whole scene in
  natural language, not just keywords. This matters most in img2img and inpaint,
  where the prompt has to carry more of the image.

- **img2img strength is more aggressive than the number suggests.** The
  `shift = 3` schedule front-loads noise, so a given strength injects far more
  than the same value on SD/SDXL — around `0.6` already noises away most of the
  input's structure. To restyle while keeping the composition, use a **low
  strength (~0.2–0.4)** and a detailed prompt; reserve higher values for
  near-full regeneration. If img2img seems to "lose" your input, lower the
  strength.

- **Inpaint wants high denoise.** Anima has no dedicated inpaint conditioning, so
  at partial denoise the original content bleeds into the masked region and
  ghosts through the fill. For a clean repaint use **denoise ~0.9–1.0**; drop to
  ~0.35–0.45 only for subtle, structure-preserving edits.

### Detailer (after generate)

Enable **Detailer** in the Generate view to run an ADetailer-style refinement
pass on each result. A YOLO model detects regions (faces, hands, …); each is
cropped, inpainted at the model's native resolution, and composited back — the
fix for soft, low-detail small faces. Unlike ADetailer it drives Diffucore's
own inpaint, so it works for **UNet (SD/SDXL)** and **DiT (Anima, FLUX)** alike.

**Stack multiple detection models** — add a pass per model (e.g. a face model
then a hand model); each runs in sequence, refining the previous result, and
carries its own optional prompt (blank reuses the main prompt). Confidence,
denoise strength, and the mask padding / blur / dilation are shared across passes.

**Denoise strength is model-aware** — the flow-matching DiTs (Anima, FLUX)
front-load high σ, so a given strength turns into far more effective noise than
SD/SDXL's EDM and regenerates much more of the face. Loading an Anima model
therefore defaults the detailer strength to **0.25** (a true refine); SD/SDXL and
FLUX keep **0.4**, so on FLUX you'll usually want to lower it by hand. Lower it to
preserve more of the original, raise it to regenerate more.

### Sweep parameters (X/Y/Z)

In txt2img mode, enable **X/Y/Z sweep** to compare a grid of settings. Each axis
picks a parameter (Sampler, Scheduler, Steps, CFG, Seed); Sampler and Scheduler
axes get a multi-select dropdown, numeric axes take a comma-separated list. The
assembled grid and every individual cell are saved to `outputs/`.

### Browse past outputs

The **Gallery** shows every image you've generated, grouped by date (newest
first) like a phone gallery. Click or tap a thumbnail to
open it in a fullscreen carousel — step through your outputs with the on-screen
arrows, the ←/→ keys, or a swipe on touch; toggle **Info** to read the image's
metadata, and hit **Load to Generate** to pull that generation's settings into
the Generate view, or **To img2img** / **To inpaint** to send the image itself in
as the input. The **Metadata** view reads parameters out of any PNG you drop in
(AUTO1111 / Forge or ComfyUI) and can send them to txt2img.

## Project structure

```
├── backend/            Python backend (FastAPI server + engine glue)
│   ├── app.py          Entry point — launches the FastAPI server (uvicorn)
│   ├── server.py       FastAPI app — REST, a job queue, and a shared SSE event stream over the engine
│   ├── engine.py       Engine singleton — model lifecycle, generation, LoRA, detailer
│   ├── detailer.py     YOLO detection + crop/expand geometry for the detailer
│   ├── metadata.py     PNG metadata — write params, read/parse AUTO1111 & ComfyUI
│   ├── utils.py        Directory scanning helpers (checkpoints, LoRAs, outputs)
│   ├── xyz_grid.py     X/Y/Z plot grid assembly
│   └── calibrate_oss.py  Headless CLI to calibrate an Anima OSS schedule
├── static/             Frontend — index.html, app.js (Alpine), style.css
├── requirements.txt    Python dependencies
├── setup.sh / setup.bat    One-shot setup (submodule init, venv, pip install) — Linux / Windows
├── launch.sh / launch.bat  Activate venv and run `python backend/app.py` — Linux / Windows
├── update.sh / update.bat  Pull latest, sync submodule, refresh deps — Linux / Windows
├── diffucore/          Git submodule — the Diffucore inference engine
├── models/             Model weight directories (user-provided)
└── outputs/            Generated images, organised by date
```

## Architecture

The project has two layers:

| Layer | Location | Responsibility |
|---|---|---|
| **Engine** | `diffucore/` (submodule) | Checkpoint loading, text conditioning, sampling loop, VAE decode, LoRA fusion |
| **UI** | Root project | FastAPI server, browser frontend, model management, prompt parsing, PNG metadata, gallery |

The [`Engine`](backend/engine.py) class is the bridge: it holds the loaded model,
exposes `generate_t2i`, `generate_i2i`, and `generate_inpaint` methods, and
handles LoRA lifecycle. [`server.py`](backend/server.py) wraps it in a FastAPI app —
jobs (generate, sweeps, calibrations, model loads) run one at a time on a single
background worker thread, and every connected device subscribes to one shared
Server-Sent-Events stream that broadcasts the queue, sampling progress, live
previews, and model-load status. The frontend in [`static/`](static/) is plain
HTML/CSS/JS with Alpine.js and no build step. No ML logic lives in the web layer.

## Status

Diffucore UI is **under active development** (pre-1.0). The interface is
functional end-to-end across its model families, with full metadata
round-trip, LoRA support, and X/Y/Z sweeps — but APIs, HTTP endpoints, and UI
layout may shift between commits, and rough edges are expected. The engine
itself is seed-reproducible and verified against reference implementations.

## License

Apache-2.0 — see [`LICENSE`](diffucore/LICENSE) and [`NOTICE`](diffucore/NOTICE).
Diffucore is an independent implementation; model architectures and sampling
algorithms are implemented from their original research publications.
