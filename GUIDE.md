# Diffucore UI — Guide

Everything beyond the [README](README.md) quick start: the full feature list,
model setup, all three generation modes, the detailer, X/Y/Z sweeps, the gallery,
network/share flags, architecture, and status.

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
- **Tiled upscaler** — an Ultimate-SD-Upscale-style toggle (and a standalone
  "Upscale" action on any result or gallery image) that enlarges 2×/4× by
  re-running low-denoise img2img over overlapping tiles and feather-blending them
  back, so large factors fit in modest VRAM. Optional **ESRGAN** base (via
  `spandrel`) for genuine detail; Lanczos otherwise. Works on every family.
- **Live preview** — watch the image form during sampling. A fast latent→RGB
  approximation (no VAE decode) streams a rough preview each step; toggle it off
  in the Generate view. SD/SDXL and Anima.
- **TeaCache** — opt-in sampling speedup for Anima: reuses the DiT's output on
  low-change steps, with a fidelity/speed threshold and optional calibration.
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
  encoders, 6–12 GB → full offload, ≤6 GB → `stream`), and you can override it
  per load (full / encoders / none / stream) to fit the model on your GPU.
  `stream` is the low-VRAM mode (ComfyUI `--lowvram` analog): it shuttles the
  backbone's blocks on/off the GPU one at a time, so SDXL's UNet or Anima's DiT
  fit a ~4 GB card where whole-backbone staging would OOM — at the cost of some
  speed. Works for SD/SDXL, FLUX, and Anima (FLUX always uses it). Because
  `stream` moves the backbone on and off the GPU per step, it can't be combined
  with `torch.compile` — enabling both auto-disables compile (with a one-line
  notice in the server log) instead of failing the load.
  Tiled VAE decode triggers automatically when a full-resolution decode
  wouldn't fit free VRAM, keeping large images within budget — or set the
  **VAE decode** mode in Settings to *Always tiled* to force it every time
  (Anima and SD/SDXL; FLUX always tiles).
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
- **Extensions** — an AUTO1111 / ComfyUI-style extension platform. Drop a
  folder under `extensions/` (or install from a git/zip URL in Settings →
  Extensions) to add API endpoints, hook into generation and model loads, queue
  jobs on the shared worker, and add tabs/panels to the UI. A reference
  `example-watermark` extension ships with the app; see
  [`docs/EXTENSIONS.md`](docs/EXTENSIONS.md) for the full API.

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
#    cu124 covers most cards; RTX 50-series (Blackwell) needs cu128 instead.
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

If you update with a plain `git pull` instead, the launch script has a safety
net: it re-syncs `requirements.txt` whenever it changes (hash-gated, so it's a
no-op otherwise), so a newly added dependency won't surface as a runtime error.

## Usage

### Place your models

```
models/
├── checkpoints/             # SD/SDXL .safetensors or .ckpt (+ FLUX all-in-one)
├── diffusion-models/        # Anima / FLUX DiT .safetensors
├── vae/                     # Anima / FLUX VAE .safetensors
├── text-encoders/           # Anima / FLUX text encoders .safetensors
├── loras/                   # LoRA adapters (.safetensors)
├── detailers/               # YOLO detection models for the detailer (.pt)
└── upscalers/               # ESRGAN-family models for the upscaler base (.pth)
```

The detailer needs `ultralytics` (installed via `requirements.txt`) and at least
one YOLO model in `detailers/` — e.g. ADetailer's `face_yolov8n.pt` / `hand_yolov8n.pt`.

The upscaler's ESRGAN base is optional: it needs `spandrel` (installed via
`requirements.txt`) and an ESRGAN-family model in `upscalers/` — e.g.
`4x-UltraSharp.pth`, or an anime model like `4x_IllustrationJaNai`. Without one
the upscaler falls back to a Lanczos base.

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

The launch scripts also pass `--autolaunch`, which opens the UI in your default
browser once the server is up; running `python backend/app.py` directly skips it.

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
3. Click **Load** — the status bar shows model info and VRAM usage. Loading a
   large model (e.g. Anima's multi-GB files) prints per-stage progress to the
   server terminal, so a slow load is distinguishable from a stuck one.

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
settings behave differently from SD/SDXL. A few things worth knowing — none are
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

- **For fast, low-step sampling, prefer a deterministic multistep sampler.**
  Anima's rectified-flow trajectory converges fine detail (faces, small text)
  faster under a deterministic 2nd-order solver than under the ancestral /
  annealed samplers (`er_sde`, `secant_anneal`) — the ancestral noise injection
  needs more steps to settle, so it can garble small details at low step counts.
  **`dpmpp_2m` on the `beta` schedule stays clean and coherent down to ~16–20
  steps**, where `er_sde` / `secant_anneal` want ~24–30 for the same result — at
  the same per-step cost. `res_multistep` and `gradient_estimation` are
  equivalent, and the scheduler barely matters for these (`beta`, `flow`,
  `sgm_uniform`, `simple` all work). Avoid `lcm`, `dpmpp_sde`, `lms`, `ipndm_v`,
  and `dpmpp_3m_sde` at low steps — they go muddy or break.

- **`exp_heun_2_x0`** is a deterministic 2nd-order option in the same family — a
  true single-step exponential Heun (two model evaluations per step, no multistep
  history) instead of `dpmpp_2m`'s one-eval history-reuse multistep. It costs one
  extra evaluation per step but needs no warm-up history, which can help at very
  low step counts; image-quality A/B versus `dpmpp_2m` is still pending.

- **`uni_pc` / `uni_pc_bh2`** (UniPC, a unified predictor-corrector multistep
  solver) are deterministic and, like `dpmpp_2m`, stay ~one model evaluation per
  step — the corrector's evaluation doubles as the next step's history. The
  corrector gives them an edge over `dpmpp_2m` at the same step count, so they're
  a strong low-step default. `uni_pc` uses the `bh1` solver variant and
  `uni_pc_bh2` the `bh2` variant (often a touch better at very low steps); both
  default to 3rd order and ramp the order down over the final steps.

- **`uni_pc_anneal`** is the *stochastic* sibling of `uni_pc`: the same UniPC
  predictor-corrector core plus a light, σ-annealed ancestral noise term (noise at
  high σ, vanishing as σ→0) for stochastic sample diversity and a shot at the
  merge-robustness that makes `er_sde` reliable — but on UniPC's higher-accuracy
  drift instead of a first-order one. It is a strict generalization of `uni_pc`
  (its `eta_max=0` limit is deterministic UniPC, exactly). Because the high-order
  core *amplifies* injected noise, it ships a deliberately small baked-in noise
  level (`eta_max=0.2`) and ignores the shared `eta_max` settings knob, which is
  tuned for the lower-order anneal samplers and over-smooths this one. Use it when
  you want UniPC quality with a touch of stochastic variation; for the crispest
  deterministic result, use plain `uni_pc`. See `docs/uni-pc-anneal.md`.

### TeaCache — faster Anima sampling

**TeaCache** (opt-in, Anima only) skips recomputing the 28-block DiT on steps
where its output barely changes, reusing the cached result instead — a large
speedup over the smooth middle of a trajectory. Enable it in the Generate panel.

- **Threshold is the speed/fidelity knob.** TeaCache accumulates how much the
  step input drifts and forces a real recompute once that crosses the threshold;
  higher = more skipping = faster but lower fidelity. There is no universal sweet
  spot — it depends on your sampler and step count. High step counts with
  single-step or secant-family samplers stay near-lossless up to ~0.3–0.5. Start
  low and raise it until quality dips.

- **Multistep solvers (`dpmpp_2m`, `res_multistep`, `ipndm`) have essentially no
  usable TeaCache window — get their speed from fewer steps instead.** These
  linearly combine the current and previous model evaluations, so reusing a stale
  one on a skipped step breaks the update. The drift between their steps is also
  tiny, so the threshold scale is far smaller than for other samplers: even
  `≤0.012` only skips ~1 step (~4%, negligible), and pushing higher corrupts the
  image in stages — color cast (~2 skips) → distorted anatomy (~5) → blur (~7+) —
  long before you get a real speedup. Since `dpmpp_2m` already stays coherent at
  16–20 steps (see *Working with Anima*), lowering the step count is the clean,
  controlled way to go faster with it; reserve TeaCache for the single-step /
  ancestral samplers that tolerate skipping.

- **Calibration (Settings → TeaCache).** Calibrating fits a per-architecture
  polynomial that remaps the raw per-step *input* drift into an estimate of the
  *output* change, so the threshold tracks what actually matters for fidelity. It
  runs once for the Anima family, is cached to `models/teacache_cache/anima.json`,
  and is then reused for every Anima checkpoint.

- **Use calibrated coefficients (toggle, on by default).** When on, generation
  applies that fitted polynomial. Turn it **off** to gate on the raw estimate
  instead — the threshold then *means* something different, so re-tune it.

- **When to turn calibration off.** Calibration is fit on a single *deterministic
  Euler* trajectory over the flow schedule, so it matches deterministic samplers
  best. Stochastic / second-order samplers — e.g. `secant_anneal` on the `beta`
  schedule — run a trajectory the fit never saw, where it can both recompute
  *more* (slower) and place those recomputes on the wrong steps (lower fidelity,
  i.e. "seed-breaking"). If a calibrated run is somehow **slower and worse** than
  uncalibrated, that's the mismatch: turn calibration off for that sampler and
  tune the raw threshold directly. Re-calibrating won't fix it — the calibration
  loop is deterministic and can't reproduce an ancestral sampler's dynamics.

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

### Upscaler (after generate)

Enable **Upscaler** in the Generate view, or use the standalone **Upscale ⬆**
button on a result or any gallery image, to enlarge with an
Ultimate-SD-Upscale-style pass: the image is pre-upscaled, then refined by
low-denoise img2img over overlapping tiles and feather-blended back. Because each
tile is only ~1024², 2× **and** 4× both fit in modest VRAM.

**Pick a base upscaler.** With **Lanczos** (the default) the base is soft, so the
refine needs high denoise to add detail — but high denoise makes each tile redraw
the whole prompt and duplicate the subject. Drop an **ESRGAN** model into
`models/upscalers/` and select it instead: it synthesises real per-pixel detail,
so the refine only needs a low denoise (~0.2) to clean it up — sharp, with no
duplication. ESRGAN is the recommended path; Lanczos is a fallback. (ESRGAN runs
through `spandrel`; see [Place your models](#place-your-models).)

**TeaCache is separate here.** The upscale pass has its own TeaCache control
(default **off**), independent of the main generation: caching is more
detail-costly on a low-denoise refine, so leave it off (or low) for the sharpest
result. Tile size, overlap, denoise, and an optional per-pass prompt (blank
reuses the main prompt) round out the controls; all upscale settings are written
into the output PNG's metadata.

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

### Extend the UI

**Settings → Extensions** lists every extension under `extensions/`, lets you
install new ones from a git URL or a `.zip` archive URL, and enable/disable,
reload (handy while developing), or uninstall each one. A broken extension is
shown with its error and never blocks the app.

An extension is just a folder with an `extension.json` manifest, a Python entry
point, and an optional `web/` directory of JS that gets injected into the UI.
Extensions can add API endpoints, hook into generation and model loading, queue
jobs on the shared worker, broadcast SSE events, store their own settings, and
add tabs and panels to the frontend. A reference `example-watermark`
extension ships with the app — read it alongside
[`docs/EXTENSIONS.md`](docs/EXTENSIONS.md) for the full API.

## Project structure

```
├── backend/            Python backend (FastAPI server + engine glue)
│   ├── app.py          Entry point — launches the FastAPI server (uvicorn)
│   ├── server.py       FastAPI app — REST, a job queue, and a shared SSE event stream over the engine
│   ├── engine.py       Engine singleton — model lifecycle, generation, LoRA, detailer, upscaler
│   ├── detailer.py     YOLO detection + crop/expand geometry for the detailer
│   ├── upscale.py      Tile geometry + feather-blend helpers for the tiled upscaler
│   ├── metadata.py     PNG metadata — write params, read/parse AUTO1111 & ComfyUI
│   ├── utils.py        Directory scanning helpers (checkpoints, LoRAs, outputs)
│   ├── xyz_grid.py     X/Y/Z plot grid assembly
│   └── calibrate_oss.py  Headless CLI to calibrate an Anima OSS schedule
├── static/             Frontend — index.html, app.js (Alpine), style.css
├── extensions/         Drop-in extensions (AUTO1111/ComfyUI-style); ships example-watermark
├── docs/               EXTENSIONS.md and feature design notes
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

Diffucore UI is at **v0.1.1** — its first tagged release was v0.1.0. The interface is
functional end-to-end across its model families, with full metadata
round-trip, LoRA support, and X/Y/Z sweeps, and the feature surface has been
verified on real hardware. It's an early release, so expect the occasional
rough edge and please report issues. The engine itself is seed-reproducible
and verified against reference implementations.
