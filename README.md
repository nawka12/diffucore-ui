# Diffucore UI

**A web frontend for the Diffucore diffusion inference engine.**

Point it at your checkpoints, pick a prompt, and generate — a darkroom-themed
interface with a unified txt2img / img2img / inpaint workspace, an X/Y/Z
parameter-sweep mode, and a gallery that recycles past generations' metadata
back into the workspace.

![version](https://img.shields.io/badge/version-0.1.5-blue)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-Apache--2.0-green)

> 🚀 **v0.1.0 — first release.** Diffucore UI is functional end-to-end across
> its model families and verified on real hardware. It's an early release, so
> expect the occasional rough edge — please report issues. Check out the
> `v0.1.0` tag for the tagged release, or track `main` for the latest.

## Quick start

```bash
# 1. Clone with submodules, then set up the venv + deps + CUDA torch
git clone --recurse-submodules https://github.com/nawka12/diffucore-ui.git
cd diffucore-ui
./setup.sh                  # Windows: setup.bat

# 2. Drop your model files under models/ (see the guide for the layout)

# 3. Launch — serves on http://127.0.0.1:7860
./launch.sh                 # Windows: launch.bat
```

Then open `http://localhost:7860`. The backend is FastAPI + Uvicorn; the
frontend is plain HTML/CSS/JS with Alpine.js (no build step).

## 📖 Full guide

**[GUIDE.md](GUIDE.md)** covers everything else — the full feature list, model
setup, all three modes (txt2img / img2img / inpaint), the detailer, the tiled
upscaler, X/Y/Z sweeps, the gallery, network/share flags (`--listen`, `--share`),
architecture, and status.

## License

Apache-2.0 — see [`LICENSE`](diffucore/LICENSE) and [`NOTICE`](diffucore/NOTICE).
Diffucore is an independent implementation; model architectures and sampling
algorithms are implemented from their original research publications.
