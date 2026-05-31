# Diffucore UI

**A Gradio web frontend for the Diffucore diffusion inference engine.**

Point it at your checkpoints, pick a prompt, and generate — with a darkroom-themed
interface for txt2img, img2img, inpainting, and a gallery that recycles past
generations' metadata back into the workspace.

![status](https://img.shields.io/badge/status-alpha-orange)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-Apache--2.0-green)

```python
from ui import build_ui

app = build_ui()
app.launch(server_name="0.0.0.0", server_port=7860)
```

Then open `http://localhost:7860` in your browser.

## Highlights

- **Three model families, one interface** — Stable Diffusion 1.5, SDXL, and
  **Anima** (a 2 B DiT built on Cosmos-Predict2). Switch between them from the
  top bar.
- **txt2img, img2img, and inpainting** tabs, each with full sampler / scheduler /
  CFG / seed controls.
- **Prompt-based LoRA loading** — embed `<lora:name:mult>` directly in your
  prompt to load adapters on the fly.
- **10 samplers, multiple schedulers** — Euler, Heun, DPM++ family, ER-SDE;
  Karras, exponential, sgm_uniform, flow, and more.
- **Gallery with metadata round-trip** — every generated image saves its full
  generation parameters as PNG metadata. The Gallery tab browses past outputs
  and can load any generation's settings back into the workspace.
- **Anima auto-defaults** — switching to Anima mode sets sampler / steps / CFG
  to sensible values (er_sde, 30, 4.0) automatically.
- **Seed recycle & randomize** — reuse the last seed or roll a new one with one
  click.
- **CPU offload & tiled VAE** — run SDXL on modest GPUs without running out of
  VRAM.
- **Custom darkroom theme** — warm amber safelight aesthetics, Fraunces serif +
  IBM Plex Mono fonts, film grain overlay, and a CSS-only aperture iris logo.

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
source .venv/bin/activate
python app.py
```

### Load a model

1. Select **SD/SDXL** or **Anima** from the top-bar radio.
2. Pick your checkpoint files from the dropdowns.
3. Click **Load** — the status bar shows model info and VRAM usage.

### Generate

Switch to the **txt2img**, **img2img**, or **inpaint** tab, enter a prompt,
adjust your sampler / steps / CFG, and click **Generate**.

LoRAs can be activated inline: `a castle in autumn, <lora:autumn_style:0.8>`.

### Browse past outputs

The **Gallery** tab shows every image you've generated. Click one to inspect its
metadata, then click **Load metadata to workspace** to re-populate all tabs with
that generation's settings.

## Project structure

```
├── app.py              Entry point — builds and launches the Gradio UI
├── ui.py               Gradio Blocks layout, callbacks, theme, and CSS
├── engine.py           Engine singleton — model lifecycle, generation, LoRA
├── utils.py            Directory scanning helpers (checkpoints, LoRAs, outputs)
├── requirements.txt    Python dependencies
├── setup.sh            One-shot setup (submodule init, venv, pip install)
├── diffucore/          Git submodule — the Diffucore inference engine
├── models/             Model weight directories (user-provided)
└── outputs/            Generated images, organised by date
```

## Architecture

The project has two layers:

| Layer | Location | Responsibility |
|---|---|---|
| **Engine** | `diffucore/` (submodule) | Checkpoint loading, text conditioning, sampling loop, VAE decode, LoRA fusion |
| **UI** | Root project | Gradio web interface, model management, prompt parsing, PNG metadata, gallery |

The [`Engine`](engine.py) class is the bridge: it holds the loaded model,
exposes `generate_t2i`, `generate_i2i`, and `generate_inpaint` methods, and
handles LoRA lifecycle. The UI code in [`ui.py`](ui.py) is pure layout and
callbacks — no ML logic lives there.

## Status

Diffucore UI is in **alpha**. The interface is functional and end-to-end working
across all three model families, with full metadata round-trip and LoRA support.
APIs and UI layout may still shift before 1.0. The engine itself is seed-
reproducible and verified against reference implementations.

## License

Apache-2.0 — see [`LICENSE`](diffucore/LICENSE) and [`NOTICE`](diffucore/NOTICE).
Diffucore is an independent implementation; model architectures and sampling
algorithms are implemented from their original research publications.
