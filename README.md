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
python app.py            # serve on http://127.0.0.1:7860
```

Then open `http://localhost:7860` in your browser. The backend is FastAPI +
Uvicorn; the frontend is plain HTML/CSS/JS with Alpine.js (no build step).

## Highlights

- **Three model families, one interface** — Stable Diffusion 1.5, SDXL, and
  **Anima** (a 2 B DiT built on Cosmos-Predict2). Switch between them from the
  model bar.
- **Unified Generate workspace** — one shared control panel with a
  txt2img / img2img / inpaint mode toggle; switching modes keeps your prompt and
  settings. Full sampler / scheduler / steps / CFG / seed controls.
- **X/Y/Z parameter sweep** — toggle it on inside txt2img to render a comparison
  grid across samplers, schedulers, steps, CFG, or seed. The assembled grid and
  every individual cell are saved to `outputs/`, sharing one seed for a fair
  comparison.
- **Prompt-based LoRA loading** — embed `<lora:name:mult>` directly in your
  prompt to load adapters on the fly.
- **11 samplers, multiple schedulers** — Euler, Heun, DPM++ family, ER-SDE,
  SECANT; Karras, exponential, sgm_uniform, flow, and more.
- **Gallery with metadata round-trip** — every generated image saves its full
  generation parameters as PNG metadata. The Gallery browses past outputs and
  loads any generation's settings back into the Generate view.
- **Metadata reader** — drop in any PNG to inspect its AUTO1111 / Forge or
  ComfyUI parameters and send them straight to txt2img.
- **Anima auto-defaults** — switching to Anima mode sets sampler / steps / CFG
  to sensible values (er_sde, 30, 4.0) automatically.
- **Seed recycle & randomize** — reuse the last seed or roll a new one with one
  click.
- **CPU offload & tiled VAE** — run SDXL on modest GPUs without running out of
  VRAM.
- **Live progress** — sampling step/total streams to a real progress bar as the
  image is generated.
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

Or run `./setup.sh` to do all of the above automatically.

## Usage

### Place your models

```
models/
├── checkpoints/             # SD/SDXL .safetensors or .ckpt
├── diffusion-models/        # Anima DiT .safetensors
├── vae/                     # Anima VAE .safetensors
├── text-encoders/           # Anima text encoder .safetensors
└── loras/                   # LoRA adapters (.safetensors)
```

### Start the UI

```bash
./launch.sh
```

Or, manually:

```bash
source .venv/bin/activate
python app.py
```

By default the UI binds to `127.0.0.1` (localhost only). Flags passed to
`launch.sh` are forwarded to `app.py`:

```bash
./launch.sh --listen        # bind 0.0.0.0 — reachable from other machines on the network
./launch.sh --port 8000     # serve on a different port (default: 7860)
./launch.sh --listen --port 8000
```

### Load a model

1. Select **SD/SDXL** or **Anima** from the top-bar radio.
2. Pick your checkpoint files from the dropdowns.
3. Click **Load** — the status bar shows model info and VRAM usage.

### Generate

In the **Generate** view, pick a mode (**txt2img**, **img2img**, or
**inpaint**), enter a prompt, adjust your sampler / steps / CFG, and click
**Generate**. Progress streams to a live bar and the result lands in the panel
on the right.

LoRAs can be activated inline: `a castle in autumn, <lora:autumn_style:0.8>`.

### Sweep parameters (X/Y/Z)

In txt2img mode, enable **X/Y/Z sweep** to compare a grid of settings. Each axis
picks a parameter (Sampler, Scheduler, Steps, CFG, Seed); Sampler and Scheduler
axes get a multi-select dropdown, numeric axes take a comma-separated list. The
assembled grid and every individual cell are saved to `outputs/`.

### Browse past outputs

The **Gallery** shows every image you've generated. Click one to inspect its
metadata, then click **Load to Generate** to load that generation's settings
into the Generate view. The **Metadata** view reads parameters out of any PNG
you drop in (AUTO1111 / Forge or ComfyUI) and can send them to txt2img.

## Project structure

```
├── app.py              Entry point — launches the FastAPI server (uvicorn)
├── server.py           FastAPI app — REST + streaming endpoints over the engine
├── metadata.py         PNG metadata — write params, read/parse AUTO1111 & ComfyUI
├── static/             Frontend — index.html, app.js (Alpine), style.css
├── engine.py           Engine singleton — model lifecycle, generation, LoRA
├── utils.py            Directory scanning helpers (checkpoints, LoRAs, outputs)
├── xyz_grid.py         X/Y/Z plot grid assembly
├── requirements.txt    Python dependencies
├── setup.sh            One-shot setup (submodule init, venv, pip install)
├── launch.sh           Activate venv and run `python app.py`
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

The [`Engine`](engine.py) class is the bridge: it holds the loaded model,
exposes `generate_t2i`, `generate_i2i`, and `generate_inpaint` methods, and
handles LoRA lifecycle. [`server.py`](server.py) wraps it in a FastAPI app —
blocking generation runs in a threadpool while sampling progress streams to the
browser as newline-delimited JSON. The frontend in [`static/`](static/) is plain
HTML/CSS/JS with Alpine.js and no build step. No ML logic lives in the web layer.

## Status

Diffucore UI is **under active development** (pre-1.0). The interface is
functional end-to-end across all three model families, with full metadata
round-trip, LoRA support, and X/Y/Z sweeps — but APIs, HTTP endpoints, and UI
layout may shift between commits, and rough edges are expected. The engine
itself is seed-reproducible and verified against reference implementations.

## License

Apache-2.0 — see [`LICENSE`](diffucore/LICENSE) and [`NOTICE`](diffucore/NOTICE).
Diffucore is an independent implementation; model architectures and sampling
algorithms are implemented from their original research publications.
