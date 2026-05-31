"""Gradio 6 UI for Diffucore — stable-diffusion-webui inspired layout."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import List

_ROOT = Path(__file__).resolve().parent
_LOCAL_DIFFUCORE_SRC = _ROOT / "diffucore" / "src"
if _LOCAL_DIFFUCORE_SRC.exists():
    sys.path.insert(0, str(_LOCAL_DIFFUCORE_SRC))

import gradio as gr
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from diffucore import __version__ as _DIFFUCORE_VERSION
from engine import ENGINE, SAMPLERS, SCHEDULERS_SD, SCHEDULERS_ANIMA
from utils import (
    OUTPUTS_DIR,
    scan_checkpoints, scan_loras, scan_diffusion_models,
    scan_vae, scan_text_encoders, scan_outputs, next_output_path,
)

_UI_COMMIT = ""
_DIFF_COMMIT = ""
try:
    _UI_COMMIT = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=_ROOT,
        stderr=subprocess.DEVNULL, text=True,
    ).strip()
except Exception:
    pass
try:
    _DIFF_COMMIT = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=_ROOT / "diffucore",
        stderr=subprocess.DEVNULL, text=True,
    ).strip()
except Exception:
    pass

_UI_ID = f"diffucore-ui+{_UI_COMMIT}" if _UI_COMMIT else "diffucore-ui"
_DIFF_ID = f"diffucore+{_DIFF_COMMIT}" if _DIFF_COMMIT else f"diffucore+{_DIFFUCORE_VERSION}"

# ── helpers ────────────────────────────────────────────────────────

def _checkpoint_list() -> List[str]:
    return scan_checkpoints() or ["(none in models/checkpoints/)"]

def _lora_list() -> List[str]:
    return scan_loras() or ["(none in models/loras/)"]

def _dit_list() -> List[str]:
    return scan_diffusion_models() or ["(none in models/diffusion-models/)"]

def _vae_list() -> List[str]:
    return scan_vae() or ["(none in models/vae/)"]

def _te_list() -> List[str]:
    return scan_text_encoders() or ["(none in models/text-encoders/)"]

def _sampler_list() -> List[str]:
    return SAMPLERS

def _scheduler_list() -> List[str]:
    return ENGINE.available_schedulers


# ── callbacks ──────────────────────────────────────────────────────

def cb_load_model(model_name, model_type,
                  anima_dit, anima_vae, anima_te,
                  compile_enabled, cuda_graphs_enabled, channels_last_enabled):
    try:
        if model_type == "Anima":
            for name in (anima_dit, anima_vae, anima_te):
                if not name or name.startswith("("):
                    return "Select all three Anima files"
            return ENGINE.load_anima(
                anima_dit, anima_vae, anima_te,
                offload=True, vae_tile=True,
                compile=compile_enabled, cuda_graphs=cuda_graphs_enabled,
            )
        else:
            if not model_name or model_name.startswith("("):
                return "Select a model"
            return ENGINE.load_model(
                model_name,
                offload=True, vae_tile=True,
                compile=compile_enabled, cuda_graphs=cuda_graphs_enabled,
                channels_last=channels_last_enabled,
            )
    except Exception as e:
        return f"Error: {e}"


def cb_status():
    return ENGINE.status_text()


def cb_toggle_model_type(model_type):
    is_anima = model_type == "Anima"
    return (
        gr.update(visible=not is_anima),
        gr.update(visible=is_anima),
    )


def cb_anima_defaults(model_type):
    if model_type == "Anima" and not ENGINE._anima_defaults_applied:
        ENGINE._anima_defaults_applied = True
        return [gr.update(value="er_sde")] * 3 + [gr.update(value=30)] * 3 + [gr.update(value=4.0)] * 3
    return [gr.update()] * 9


def cb_update_schedulers(model_type):
    if model_type == "Anima":
        return (
            gr.update(choices=SCHEDULERS_ANIMA, value=SCHEDULERS_ANIMA[0]),
            gr.update(choices=SCHEDULERS_ANIMA, value=SCHEDULERS_ANIMA[0]),
            gr.update(choices=SCHEDULERS_ANIMA, value=SCHEDULERS_ANIMA[0]),
        )
    return (
        gr.update(choices=SCHEDULERS_SD, value=SCHEDULERS_SD[0]),
        gr.update(choices=SCHEDULERS_SD, value=SCHEDULERS_SD[0]),
        gr.update(choices=SCHEDULERS_SD, value=SCHEDULERS_SD[0]),
    )


# ── metadata ───────────────────────────────────────────────────────

def _metadata_str(gen_kwargs) -> str:
    prompt = gen_kwargs.get("prompt", "")
    neg = gen_kwargs.get("negative_prompt", "")
    model = ENGINE.loaded_name or "unknown"
    fields = []
    if "steps" in gen_kwargs:
        fields.append(f"Steps: {gen_kwargs['steps']}")
    fields.append(f"Sampler: {gen_kwargs.get('sampler', 'euler')}")
    fields.append(f"Scheduler: {gen_kwargs.get('scheduler', 'karras')}")
    fields.append(f"CFG scale: {gen_kwargs.get('cfg_scale', 7.0)}")
    fields.append(f"Seed: {ENGINE.last_seed}")
    if "width" in gen_kwargs and "height" in gen_kwargs:
        fields.append(f"Size: {gen_kwargs['width']}x{gen_kwargs['height']}")
    fields.append(f"Model: {model}")
    if "strength" in gen_kwargs:
        fields.append(f"Denoising strength: {gen_kwargs['strength']}")
    if "shift" in gen_kwargs:
        fields.append(f"Shift: {gen_kwargs['shift']}")
    fields.append(f"diffucore-ui: {_UI_ID}")
    fields.append(_DIFF_ID)
    flags = ENGINE.perf_flags_str
    if flags != "default":
        fields.append(f"Perf flags: {flags}")
    return f"{prompt}\nNegative prompt: {neg}\n{', '.join(fields)}"


def _read_png_metadata(path: str) -> str:
    img = Image.open(path)
    return img.info.get("parameters", "")


def _parse_metadata(params_str: str) -> dict:
    if not params_str:
        return {}
    result = {}
    neg_marker = "\nNegative prompt: "
    if neg_marker in params_str:
        prompt, rest = params_str.split(neg_marker, 1)
        result["prompt"] = prompt.strip()
        if "\n" in rest:
            neg, fields_str = rest.split("\n", 1)
            result["negative_prompt"] = neg.strip()
        else:
            result["negative_prompt"] = rest.strip()
            fields_str = ""
    else:
        lines = params_str.split("\n", 1)
        result["prompt"] = lines[0].strip()
        fields_str = lines[1] if len(lines) > 1 else ""
        result["negative_prompt"] = ""
    for pair in fields_str.split(","):
        pair = pair.strip()
        if ":" in pair:
            key, val = pair.split(":", 1)
            result[key.strip().lower().replace(" ", "_")] = val.strip()
    return result


def _parse_comfyui_metadata(prompt_json: str) -> dict:
    import json
    try:
        data = json.loads(prompt_json)
    except json.JSONDecodeError:
        return {}
    result = {}
    for node_id, node in data.items():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type", "")
        inputs = node.get("inputs", {})
        if class_type == "KSampler":
            for key in ("seed", "steps", "cfg"):
                if key in inputs:
                    result[key] = inputs[key]
            if "sampler_name" in inputs:
                result["sampler"] = inputs["sampler_name"]
            if "scheduler" in inputs:
                result["scheduler"] = inputs["scheduler"]
            if "denoise" in inputs:
                result["denoising_strength"] = inputs["denoise"]
        if class_type == "CLIPTextEncode":
            text = inputs.get("text", "")
            if text:
                if "prompt" not in result:
                    result["prompt"] = text
                else:
                    result["negative_prompt"] = text
        if class_type == "EmptyLatentImage":
            if "width" in inputs and "height" in inputs:
                result["size"] = f"{inputs['width']}x{inputs['height']}"
    return result


# ── generation ─────────────────────────────────────────────────────

def _generate_with_loras(prompt, neg, gen_fn, gen_kwargs, progress: gr.Progress | None = None):
    """Parse LoRA tags from prompt, apply them temporarily, generate, clean up."""
    if not ENGINE.loaded_name:
        gr.Warning("Load a model first")
        return None, "No model loaded"

    clean_prompt, prompt_loras = ENGINE.parse_lora_prompt(prompt)
    clean_neg, neg_loras = ENGINE.parse_lora_prompt(neg)
    loras = prompt_loras + neg_loras

    try:
        lora_info = ""
        if loras:
            lora_info = ENGINE.apply_temp_loras(loras) + "  |  "

        def _on_sampling_step(step: int, total: int) -> None:
            if progress is not None and total > 0:
                progress((step, total), desc=f"Sampling {step}/{total}")

        gen_kwargs["prompt"] = clean_prompt
        gen_kwargs["negative_prompt"] = clean_neg
        gen_kwargs["progress_callback"] = _on_sampling_step
        t0 = time.perf_counter()
        image, info = gen_fn(**gen_kwargs)
        inference_time = time.perf_counter() - t0
        if progress is not None:
            progress(1, desc="Saving…")

        out = next_output_path(ENGINE.last_seed)
        meta = PngInfo()
        meta_kwargs = {k: v for k, v in gen_kwargs.items() if k != "progress_callback"}
        meta.add_text("parameters", _metadata_str(meta_kwargs))
        image.save(out, pnginfo=meta)
        return image, f"{lora_info}{info}  |  inference: {inference_time:.2f}s  |  saved to {out.relative_to(OUTPUTS_DIR)}"
    except Exception as e:
        return None, f"Error: {e}"
    finally:
        if loras:
            ENGINE.clear_temp_loras()
        if progress is not None:
            progress(1, desc="Done")


def cb_generate_t2i(prompt, neg, width, height, steps, cfg, sampler, scheduler, seed, shift,
                    progress: gr.Progress = gr.Progress()):
    progress(0, desc="Generating…")
    return _generate_with_loras(prompt, neg, ENGINE.generate_t2i, dict(
        width=int(width), height=int(height), steps=int(steps),
        cfg_scale=float(cfg), sampler=sampler, scheduler=scheduler,
        seed=int(seed), shift=float(shift),
    ), progress=progress)


def cb_generate_i2i(prompt, neg, input_image, strength, steps, cfg, sampler, scheduler, seed,
                    progress: gr.Progress = gr.Progress()):
    if input_image is None:
        gr.Warning("Provide an input image")
        return None, "No input image"
    progress(0, desc="Generating…")
    return _generate_with_loras(prompt, neg, ENGINE.generate_i2i, dict(
        input_image=input_image, strength=float(strength), steps=int(steps),
        cfg_scale=float(cfg), sampler=sampler, scheduler=scheduler, seed=int(seed),
    ), progress=progress)


def cb_generate_inpaint(prompt, neg, input_image, mask_image, strength, steps, cfg, sampler, scheduler, seed,
                        progress: gr.Progress = gr.Progress()):
    if input_image is None or mask_image is None:
        gr.Warning("Provide both input and mask images")
        return None, "Missing images"
    progress(0, desc="Generating…")
    return _generate_with_loras(prompt, neg, ENGINE.generate_inpaint, dict(
        input_image=input_image, mask_image=mask_image,
        strength=float(strength), steps=int(steps),
        cfg_scale=float(cfg), sampler=sampler, scheduler=scheduler, seed=int(seed),
    ), progress=progress)


# ── gallery ────────────────────────────────────────────────────────

def _recycle_from_gallery(path: str | None):
    if not path:
        return [gr.update()] * 26
    meta = _parse_metadata(_read_png_metadata(path))
    prompt = meta.get("prompt", "")
    neg = meta.get("negative_prompt", "")
    try:
        steps = int(meta.get("steps", 0)) or None
    except Exception:
        steps = None
    try:
        cfg = float(meta.get("cfg_scale", 0)) or None
    except Exception:
        cfg = None
    sampler = meta.get("sampler", None)
    scheduler = meta.get("scheduler", None)
    try:
        seed = int(meta.get("seed", -1))
    except Exception:
        seed = -1
    try:
        shift = float(meta.get("shift", 3.0))
    except Exception:
        shift = 3.0
    try:
        strength = float(meta.get("denoising_strength", 0.6))
    except Exception:
        strength = 0.6
    width = gr.update()
    height = gr.update()
    size_str = meta.get("size", "")
    if "x" in size_str:
        try:
            parts = size_str.split("x")
            w = int(parts[0])
            h = int(parts[1])
            if 256 <= w <= 2048 and 256 <= h <= 2048:
                width = gr.update(value=w)
                height = gr.update(value=h)
        except Exception:
            pass

    def _upd(v):
        return gr.update(value=v) if v is not None else gr.update()

    return [
        _upd(prompt), _upd(neg),          # t2i prompt, neg
        width, height,                      # t2i width, height
        _upd(steps), _upd(cfg), _upd(sampler), _upd(scheduler), _upd(seed),
        _upd(shift),                        # t2i shift
        _upd(prompt), _upd(neg),          # i2i prompt, neg
        _upd(steps), _upd(cfg), _upd(sampler), _upd(scheduler), _upd(seed),
        _upd(strength),                     # i2i strength
        _upd(prompt), _upd(neg),          # inpaint prompt, neg
        _upd(steps), _upd(cfg), _upd(sampler), _upd(scheduler), _upd(seed),
        _upd(strength),                     # inpaint strength
    ]


# ── CSS ────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

/* ════════════════════════════════════════════════════════════════════
   DIFFUCORE — "darkroom" theme
   warm near-black under amber safelight · developer-teal active states
   ════════════════════════════════════════════════════════════════════ */
:root {
  --ink:        #08070500;       /* deepest pit            */
  --bg:         #0d0b09;         /* page                   */
  --surface:    #161210;         /* panels                 */
  --surface-2:  #1d1813;         /* elevated / fields      */
  --surface-3:  #261f17;         /* hover / chips          */

  --ember:      #ff6a2b;         /* safelight — primary    */
  --ember-soft: #ff8c4d;
  --ember-deep: #c2410c;
  --halo:       rgba(255,106,43,0.18);
  --teal:       #57d6c2;         /* developed / active     */
  --teal-halo:  rgba(87,214,194,0.16);

  --line:        rgba(255,138,76,0.10);
  --line-strong: rgba(255,138,76,0.26);

  --txt:   #f4ead9;
  --txt-2: #b9aa95;
  --txt-3: #7d7060;

  --r-sm: 4px;
  --r-md: 8px;
  --r-lg: 14px;

  --serif: 'Fraunces', Georgia, 'Times New Roman', serif;
  --mono:  'IBM Plex Mono', ui-monospace, 'SF Mono', monospace;

  --shadow-card:     0 1px 0 rgba(255,200,150,0.03) inset, 0 8px 30px rgba(0,0,0,0.45);
  --shadow-elevated: 0 1px 0 rgba(255,200,150,0.05) inset, 0 14px 44px rgba(0,0,0,0.55);

  --t-fast: 150ms cubic-bezier(0.4,0,0.2,1);
  --t:      320ms cubic-bezier(0.16,1,0.3,1);
}

body {
  background:
    radial-gradient(115% 75% at 50% -12%, rgba(255,106,43,0.11) 0%, transparent 58%),
    radial-gradient(85% 60% at 100% 105%, rgba(87,214,194,0.045) 0%, transparent 55%),
    var(--bg) !important;
  color: var(--txt);
  scrollbar-width: thin;
  scrollbar-color: var(--ember-deep) transparent;
}

/* film grain overlay */
body::after {
  content: "";
  position: fixed; inset: 0;
  z-index: 9998;
  pointer-events: none;
  opacity: 0.045;
  mix-blend-mode: overlay;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='140' height='140'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
}

.gradio-container {
  max-width: 1480px !important;
  margin: 0 auto !important;
  padding: 0 22px 28px !important;
  background: transparent !important;
  font-family: var(--mono) !important;
}

/* ── header ───────────────────────────────────────────────────────── */
#app-header {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 26px 2px 16px;
  margin-bottom: 10px;
  border-bottom: 1px solid var(--line);
  position: relative;
}
#app-header::after {                       /* glowing safelight rule */
  content: ""; position: absolute; left: 0; bottom: -1px;
  width: 180px; height: 1px;
  background: linear-gradient(90deg, var(--ember), transparent);
  box-shadow: 0 0 12px var(--halo);
}
/* aperture iris logo (pure CSS) */
#app-header .logo-icon {
  width: 42px; height: 42px;
  border-radius: 50%;
  flex-shrink: 0;
  position: relative;
  background: conic-gradient(from 18deg,
    #ff8c4d, #c2410c, #ff8c4d, #c2410c, #ff8c4d, #c2410c, #ff8c4d);
  box-shadow: 0 0 0 1px var(--line-strong), 0 0 26px var(--halo);
}
#app-header .logo-icon::before {           /* aperture blade seams */
  content: ""; position: absolute; inset: 0; border-radius: 50%;
  background: repeating-conic-gradient(from 0deg,
    transparent 0 56deg, rgba(0,0,0,0.40) 56deg 60deg);
}
#app-header .logo-icon::after {            /* central pupil */
  content: ""; position: absolute; inset: 31%; border-radius: 50%;
  background: var(--bg);
  box-shadow: inset 0 0 8px rgba(0,0,0,0.9), 0 0 5px var(--halo);
}
#app-header .logo-text {
  font-family: var(--serif) !important;
  font-optical-sizing: auto;
  font-size: 30px; font-weight: 600;
  line-height: 1;
  color: var(--txt);
  letter-spacing: -0.5px;
}
#app-header .logo-text em {
  font-style: italic;
  font-weight: 400;
  color: var(--ember-soft);
}
#app-header .header-subtitle {
  font-family: var(--mono);
  font-size: 11px;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  color: var(--ember-soft);
  margin-left: auto;
  background: var(--surface);
  padding: 6px 13px;
  border-radius: 100px;
  border: 1px solid var(--line-strong);
  display: inline-flex; align-items: center; gap: 8px;
}
#app-header .header-subtitle::before {     /* blinking safelight dot */
  content: ""; width: 7px; height: 7px; border-radius: 50%;
  background: var(--ember);
  box-shadow: 0 0 8px var(--ember);
  animation: pulse 2.4s ease-in-out infinite;
}
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }

/* one-time staggered page reveal */
@keyframes rise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
#app-header { animation: rise 0.6s backwards; }
.gr-group { animation: rise 0.7s backwards 0.08s; }
.tabs { animation: rise 0.7s backwards 0.16s; }

/* ── panels ───────────────────────────────────────────────────────── */
.gr-group, .panel {
  background:
    linear-gradient(180deg, rgba(255,138,76,0.025), transparent 120px),
    var(--surface) !important;
  border-radius: var(--r-lg) !important;
  border: 1px solid var(--line) !important;
  box-shadow: var(--shadow-card) !important;
  padding: 18px !important;
  transition: box-shadow var(--t), border-color var(--t);
}
.gr-group:hover {
  border-color: var(--line-strong) !important;
  box-shadow: var(--shadow-elevated) !important;
}

/* ── tabs (labelled like a filmstrip) ─────────────────────────────── */
.tabs { background: transparent !important; border: none !important; }
.tabs > .tab-nav {
  background: transparent !important;
  border: none !important;
  border-bottom: 1px solid var(--line) !important;
  border-radius: 0 !important;
  padding: 0 !important;
  gap: 4px !important;
  margin-bottom: 16px !important;
}
.tabs > .tab-nav button {
  font-family: var(--mono) !important;
  border-radius: 0 !important;
  font-weight: 500 !important;
  font-size: 12px !important;
  letter-spacing: 1.5px !important;
  text-transform: uppercase !important;
  padding: 11px 20px !important;
  color: var(--txt-3) !important;
  background: transparent !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  margin-bottom: -1px !important;
  transition: all var(--t-fast) !important;
}
.tabs > .tab-nav button:hover {
  color: var(--txt) !important;
  background: linear-gradient(180deg, transparent, rgba(255,106,43,0.05)) !important;
}
.tabs > .tab-nav button.selected {
  color: var(--ember-soft) !important;
  border-bottom: 2px solid var(--ember) !important;
  text-shadow: 0 0 14px var(--halo) !important;
}

/* ── accordions ───────────────────────────────────────────────────── */
details.accordion {
  background: var(--surface-2) !important;
  border-radius: var(--r-md) !important;
  border: 1px solid var(--line) !important;
  margin-bottom: 10px !important;
  overflow: hidden !important;
  transition: border-color var(--t-fast);
}
details.accordion[open] { border-color: var(--line-strong) !important; }
details.accordion > summary {
  font-family: var(--mono) !important;
  padding: 11px 15px !important;
  font-weight: 500 !important;
  font-size: 11px !important;
  letter-spacing: 1.2px !important;
  text-transform: uppercase !important;
  color: var(--txt-2) !important;
  background: var(--surface-3) !important;
  cursor: pointer !important;
  transition: color var(--t-fast);
}
details.accordion > summary:hover { color: var(--ember-soft) !important; }
details.accordion > .accordion-body { padding: 14px 15px 16px !important; }

/* ── labels & fields ──────────────────────────────────────────────── */
label, .gr-label {
  font-family: var(--mono) !important;
  font-size: 10.5px !important;
  font-weight: 500 !important;
  color: var(--txt-3) !important;
  letter-spacing: 1.2px !important;
  text-transform: uppercase !important;
}

textarea, input[type="text"], input[type="number"], .gr-text-input, .gr-box {
  background: var(--surface-2) !important;
  border: 1px solid var(--line) !important;
  border-radius: var(--r-sm) !important;
  color: var(--txt) !important;
  font-family: var(--mono) !important;
  transition: border-color var(--t-fast), box-shadow var(--t-fast) !important;
}
textarea:focus, input:focus {
  border-color: var(--ember) !important;
  box-shadow: 0 0 0 3px var(--halo) !important;
  outline: none !important;
}
textarea::placeholder { color: var(--txt-3) !important; font-style: italic; }

/* ── sliders ──────────────────────────────────────────────────────── */
.gr-slider input[type="range"] { accent-color: var(--ember) !important; }
.gr-slider .slider-value {
  background: var(--surface-3) !important;
  border: 1px solid var(--line-strong) !important;
  border-radius: var(--r-sm) !important;
  padding: 1px 8px !important;
  font-family: var(--mono) !important;
  font-size: 12px !important;
  font-weight: 600 !important;
  color: var(--ember-soft) !important;
}

/* ── dropdowns / checks ───────────────────────────────────────────── */
select, .gr-dropdown {
  background: var(--surface-2) !important;
  border: 1px solid var(--line) !important;
  border-radius: var(--r-sm) !important;
  color: var(--txt) !important;
  font-family: var(--mono) !important;
}
select:focus, .gr-dropdown:focus {
  border-color: var(--ember) !important;
  box-shadow: 0 0 0 3px var(--halo) !important;
}
/* long checkpoint / file names: truncate with ellipsis, never hard-clip */
.gr-dropdown input, .gr-dropdown .secondary-wrap input, select, input[type="text"] {
  text-overflow: ellipsis !important;
  white-space: nowrap !important;
  overflow: hidden !important;
}

/* model bar: compact control strip */
#model-bar {
  padding: 14px 16px !important;
  overflow: visible !important;
}
#model-bar .form {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
  padding: 0 !important;
  overflow: visible !important;
}
#model-controls {
  align-items: flex-start !important;
  gap: 12px !important;
  overflow: visible !important;
}
#model-selectors {
  min-width: 0 !important;
  gap: 8px !important;
  overflow: visible !important;
}
#model-actions {
  flex: 0 0 auto !important;
  width: auto !important;
  min-width: 160px !important;
  gap: 3px !important;
  align-items: flex-end !important;
  padding-top: 15px !important;
  overflow: visible !important;
}
#model-actions button {
  height: 36px !important;
}
.gr-checkbox, .gr-radio { accent-color: var(--ember) !important; }
.gr-checkbox label, .gr-radio label { color: var(--txt) !important; }

/* compact segmented model-family switch */
#model-type-switcher {
  min-width: 172px !important;
  max-width: 184px !important;
  align-self: flex-start !important;
  padding: 0 !important;
  overflow: visible !important;
}
#model-type-switcher,
#model-type-switcher .wrap,
#model-type-switcher .container {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}
#model-type-switcher > label,
#model-type-switcher .gr-label {
  display: block !important;
  margin: 0 0 9px !important;
  white-space: nowrap !important;
}
#model-type-switcher .wrap,
#model-type-switcher .options,
#model-type-switcher .radio-group,
#model-type-switcher [role="radiogroup"] {
  display: grid !important;
  grid-template-columns: 1fr 1fr !important;
  gap: 4px !important;
  width: 100% !important;
  padding: 3px !important;
  border: 1px solid var(--line) !important;
  border-radius: var(--r-md) !important;
  background: rgba(255,138,76,0.045) !important;
  overflow: visible !important;
}
#model-type-switcher input[type="radio"] {
  position: absolute !important;
  opacity: 0 !important;
  pointer-events: none !important;
}
#model-type-switcher label:has(input[type="radio"]) {
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  min-width: 0 !important;
  min-height: 34px !important;
  margin: 0 !important;
  padding: 0 10px !important;
  border: 1px solid transparent !important;
  border-radius: 5px !important;
  color: var(--txt-3) !important;
  background: transparent !important;
  font-size: 11px !important;
  font-weight: 600 !important;
  letter-spacing: 1px !important;
  line-height: 1 !important;
  text-align: center !important;
  text-transform: uppercase !important;
  cursor: pointer !important;
  transition: color var(--t-fast), background var(--t-fast), border-color var(--t-fast), box-shadow var(--t-fast) !important;
}
#model-type-switcher label:has(input[type="radio"]:hover) {
  color: var(--txt) !important;
  background: rgba(255,106,43,0.06) !important;
}
#model-type-switcher label:has(input[type="radio"]:checked) {
  color: #fff3ec !important;
  border-color: rgba(255,138,76,0.34) !important;
  background: linear-gradient(180deg, rgba(255,138,76,0.22), rgba(255,106,43,0.12)) !important;
  box-shadow: 0 0 0 1px rgba(255,106,43,0.08) inset, 0 8px 22px rgba(255,106,43,0.10) !important;
}
#model-type-switcher label:has(input[type="radio"]:checked)::before {
  content: "";
  width: 6px;
  height: 6px;
  margin-right: 7px;
  border-radius: 50%;
  background: var(--ember);
  box-shadow: 0 0 8px var(--ember);
  flex: 0 0 auto;
}

/* perf flags as stable toggle chips, without Gradio's internal scroll boxes */
#perf-flags {
  display: flex !important;
  flex-wrap: wrap !important;
  gap: 8px !important;
  width: 100% !important;
  min-height: 34px !important;
  margin: 0 !important;
  overflow: visible !important;
}
#perf-flags,
#perf-flags .form,
#perf-flags .wrap,
#perf-flags .container,
#perf-flags .gr-checkbox {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
  padding: 0 !important;
  overflow: visible !important;
}
#perf-flags .gr-checkbox {
  flex: 0 1 auto !important;
  width: auto !important;
  min-width: 0 !important;
}
#perf-flags label:has(input[type="checkbox"]) {
  display: inline-flex !important;
  align-items: center !important;
  justify-content: flex-start !important;
  gap: 8px !important;
  min-width: 0 !important;
  min-height: 34px !important;
  margin: 0 !important;
  padding: 0 12px !important;
  border: 1px solid var(--line) !important;
  border-radius: var(--r-sm) !important;
  background: rgba(255,138,76,0.045) !important;
  color: var(--txt-2) !important;
  font-size: 11px !important;
  font-weight: 600 !important;
  letter-spacing: 1px !important;
  line-height: 1 !important;
  white-space: nowrap !important;
  text-transform: uppercase !important;
}
#perf-flags input[type="checkbox"] {
  width: 14px !important;
  height: 14px !important;
  margin: 0 !important;
  flex: 0 0 auto !important;
}
#perf-flags label:has(input[type="checkbox"]:checked) {
  border-color: var(--line-strong) !important;
  color: var(--txt) !important;
  background: rgba(255,106,43,0.10) !important;
}

/* ── buttons ──────────────────────────────────────────────────────── */
button, .gr-button {
  font-family: var(--mono) !important;
  border-radius: var(--r-sm) !important;
  font-weight: 600 !important;
  font-size: 12px !important;
  letter-spacing: 0.6px !important;
  transition: all var(--t-fast) !important;
  cursor: pointer !important;
}
button.gr-button-primary, .gr-button.variant-primary {
  background: linear-gradient(135deg, var(--ember-soft), var(--ember-deep)) !important;
  color: #fff5ee !important;
  border: 1px solid rgba(255,170,120,0.35) !important;
  box-shadow: 0 3px 16px var(--halo), 0 1px 0 rgba(255,220,190,0.25) inset !important;
  text-transform: uppercase !important;
}
button.gr-button-primary:hover, .gr-button.variant-primary:hover {
  box-shadow: 0 6px 26px rgba(255,106,43,0.45), 0 1px 0 rgba(255,220,190,0.3) inset !important;
  transform: translateY(-1px) !important;
}
button.gr-button-primary:active, .gr-button.variant-primary:active {
  transform: translateY(0) !important;
  box-shadow: 0 2px 10px var(--halo) !important;
}
button.gr-button-stop, .gr-button.variant-stop {
  background: rgba(239,68,68,0.10) !important;
  color: #f87171 !important;
  border: 1px solid rgba(239,68,68,0.30) !important;
}
button.gr-button-stop:hover, .gr-button.variant-stop:hover {
  background: rgba(239,68,68,0.20) !important;
  border-color: rgba(239,68,68,0.45) !important;
}
button:not(.gr-button-primary):not(.variant-primary):not(.gr-button-stop):not(.variant-stop) {
  background: var(--surface-3) !important;
  color: var(--txt-2) !important;
  border: 1px solid var(--line) !important;
}
button:not(.gr-button-primary):not(.variant-primary):not(.gr-button-stop):not(.variant-stop):hover {
  background: var(--surface-2) !important;
  color: var(--ember-soft) !important;
  border-color: var(--line-strong) !important;
}

/* ── status readout ───────────────────────────────────────────────── */
#status-bar {
  font-family: var(--mono) !important;
  font-size: 0.76em !important;
  color: var(--teal) !important;
  border: 1px solid var(--line) !important;
  border-left: 2px solid var(--teal) !important;
  box-shadow: 0 0 18px var(--teal-halo) !important;
  background: var(--surface) !important;
  border-radius: var(--r-sm) !important;
}
#status-bar > div > textarea {
  background: transparent !important;
  border: none !important;
  color: var(--teal) !important;
  font-family: var(--mono) !important;
  resize: none !important;
}

/* ── image / result frame ─────────────────────────────────────────── */
.gr-image {
  border-radius: var(--r-md) !important;
  border: 1px solid var(--line-strong) !important;
  background:
    repeating-conic-gradient(#161210 0% 25%, #1a1512 0% 50%) 0 0 / 22px 22px !important;
  box-shadow: 0 0 0 1px rgba(0,0,0,0.4) inset, 0 10px 34px rgba(0,0,0,0.5) !important;
}
.gr-image img { border-radius: 0 !important; }

/* ── generate (the developing tray) ───────────────────────────────── */
.generate-btn {
  min-width: 160px !important;
  min-height: 48px !important;
  font-size: 14px !important;
  letter-spacing: 2px !important;
  text-transform: uppercase !important;
}

.gr-upload, .gr-box.has-image {
  border: 1px dashed var(--line-strong) !important;
  border-radius: var(--r-md) !important;
  background: var(--surface-2) !important;
}
.gr-upload:hover {
  border-color: var(--ember) !important;
  background: rgba(255,106,43,0.04) !important;
}

.gr-row { gap: 10px !important; }

/* LoRA hint markdown */
.gr-group + div .md, .prose code, code {
  font-family: var(--mono) !important;
  color: var(--teal) !important;
  background: var(--surface-3) !important;
  border-radius: 3px;
  padding: 1px 5px;
}
"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.Color(            # ember / safelight
        c50="#fff3ec", c100="#ffe0cf", c200="#ffc1a0",
        c300="#ff9c6e", c400="#ff7a40", c500="#ff6a2b",
        c600="#e2530f", c700="#c2410c", c800="#9a330a",
        c900="#7c2a08", c950="#5a1d05",
    ),
    secondary_hue=gr.themes.Color(          # developer teal
        c50="#effbf8", c100="#d6f5ee", c200="#aeeade",
        c300="#7fded0", c400="#57d6c2", c500="#2bbfa9",
        c600="#1f9c89", c700="#1c7d6f", c800="#1a635a",
        c900="#17514a", c950="#0b302c",
    ),
    neutral_hue=gr.themes.Color(            # warm charcoal
        c50="#f4ead9", c100="#e6d8c4", c200="#b9aa95",
        c300="#9c8d78", c400="#7d7060", c500="#5e5446",
        c600="#463e33", c700="#2f2a22", c800="#1d1813",
        c900="#161210", c950="#0d0b09",
    ),
    font=gr.themes.GoogleFont("IBM Plex Mono"),
    font_mono=gr.themes.GoogleFont("IBM Plex Mono"),
).set(
    body_background_fill="#0d0b09",
    body_background_fill_dark="#0d0b09",
    block_background_fill="#161210",
    block_border_color="rgba(255,138,76,0.10)",
    body_text_color="#f4ead9",
)

# ── build ──────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Diffucore",
        fill_height=True,
        fill_width=True,
    ) as app:

        # ════════════════════════════════════════════════════════════
        # HEADER
        # ════════════════════════════════════════════════════════════
        with gr.Row(elem_id="app-header"):
            gr.HTML(
                '''<div class="logo-icon"></div>'''
            )
            gr.HTML(
                '''<span class="logo-text">Diffu<em>core</em></span>'''
            )
            gr.HTML(
                '''<span class="header-subtitle">safelight on · sdxl · anima</span>'''
            )

        # ════════════════════════════════════════════════════════════
        # TOP BAR — model selector + perf flags + LoRA + load
        # ════════════════════════════════════════════════════════════
        with gr.Group(elem_id="model-bar"):
            with gr.Row(equal_height=False, elem_id="model-controls"):
                model_type = gr.Radio(
                    ["SD/SDXL", "Anima"], value="SD/SDXL", label="Model type",
                    scale=0, min_width=172, elem_id="model-type-switcher",
                )
                with gr.Column(scale=4, min_width=0, elem_id="model-selectors"):
                    model_dd = gr.Dropdown(
                        choices=_checkpoint_list(), label="Checkpoint",
                        value=_checkpoint_list()[0], interactive=True,
                    )
                    anima_row = gr.Row(visible=False)
                    with anima_row:
                        anima_dit = gr.Dropdown(
                            choices=_dit_list(), label="DiT",
                            value=_dit_list()[0], interactive=True, scale=2,
                        )
                        anima_vae = gr.Dropdown(
                            choices=_vae_list(), label="VAE",
                            value=_vae_list()[0], interactive=True, scale=1,
                        )
                        anima_te = gr.Dropdown(
                            choices=_te_list(), label="TE",
                            value=_te_list()[0], interactive=True, scale=1,
                        )
                    with gr.Row(equal_height=False, elem_id="perf-flags"):
                        compile_cb = gr.Checkbox(label="torch.compile", value=False)
                        cuda_graphs_cb = gr.Checkbox(label="CUDA Graphs", value=False)
                        channels_last_cb = gr.Checkbox(label="channels_last", value=False)
                with gr.Row(equal_height=False, elem_id="model-actions"):
                    refresh_btn = gr.Button("↻", scale=0, min_width=50)
                    load_btn = gr.Button("Load", variant="primary", scale=0, min_width=110)

        # CUDA Graphs requires compile
        compile_cb.change(
            lambda c: gr.update(interactive=c), compile_cb, cuda_graphs_cb,
        )

        gr.Markdown(
            "**LoRA**: use `<lora:name:mult>` in your prompt — e.g. `<lora:my_lora:0.75>`"
        )

        status_bar = gr.Textbox(
            value=ENGINE.status_text(), interactive=False,
            show_label=False, elem_id="status-bar",
        )

        # toggle SD checkpoint vs Anima file selectors
        model_type.change(
            cb_toggle_model_type, [model_type],
            [model_dd, anima_row],
        )

        # refresh all dropdowns
        def _refresh_all():
            c = _checkpoint_list()
            d = _dit_list()
            v = _vae_list()
            t = _te_list()
            return (
                gr.Dropdown(choices=c, value=c[0]),
                gr.Dropdown(choices=d, value=d[0]),
                gr.Dropdown(choices=v, value=v[0]),
                gr.Dropdown(choices=t, value=t[0]),
            )

        refresh_btn.click(
            _refresh_all, [],
            [model_dd, anima_dit, anima_vae, anima_te],
        )

        load_btn.click(
            cb_load_model,
            [model_dd, model_type, anima_dit, anima_vae, anima_te,
             compile_cb, cuda_graphs_cb, channels_last_cb],
            [status_bar],
        )

        # seed button handlers (forward refs — wired before tabs, built after)
        def _recycle_seed():
            return ENGINE.last_seed if ENGINE.last_seed >= 0 else -1

        # ════════════════════════════════════════════════════════════
        # MAIN TABS — txt2img / img2img / inpaint
        # ════════════════════════════════════════════════════════════
        with gr.Tabs():
            # ── txt2img ────────────────────────────────────────────
            with gr.Tab("txt2img"):
                # Top row: prompts left, generate right (reForge classic style)
                with gr.Row():
                    with gr.Column(scale=4):
                        t2i_prompt = gr.Textbox(
                            label="Prompt", lines=3, max_lines=6,
                            placeholder="a watercolor fox in a misty forest",
                        )
                        t2i_neg = gr.Textbox(
                            label="Negative prompt", lines=2,
                            placeholder="blurry, low quality",
                        )
                    with gr.Column(scale=1, min_width=180):
                        t2i_gen = gr.Button("Generate", variant="primary", elem_classes="generate-btn")
                # Main area: settings left, result right (reForge split panel)
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Accordion("Sampling", open=True):
                            t2i_sampler = gr.Dropdown(choices=_sampler_list(), value="dpmpp_2m", label="Sampler")
                            t2i_scheduler = gr.Dropdown(choices=_scheduler_list(), value=_scheduler_list()[0], label="Scheduler")
                            t2i_steps = gr.Slider(1, 150, value=25, step=1, label="Steps")
                            t2i_cfg = gr.Slider(1, 30, value=6.0, step=0.5, label="CFG scale")
                        with gr.Accordion("Size & Seed", open=True):
                            t2i_width = gr.Slider(256, 2048, value=1024, step=64, label="Width")
                            t2i_height = gr.Slider(256, 2048, value=1024, step=64, label="Height")
                            with gr.Row():
                                t2i_seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0, scale=3)
                                t2i_recycle = gr.Button("♻️", scale=0, min_width=40)
                                t2i_randomize = gr.Button("🎲", scale=0, min_width=40)
                        with gr.Accordion("Anima", open=False):
                            t2i_shift = gr.Slider(0.1, 20, value=3.0, step=0.1, label="Shift")
                    with gr.Column(scale=2):
                        t2i_image = gr.Image(label="Result", type="pil", height=512)
                        t2i_info = gr.Textbox(label="Info", interactive=False, show_label=False)

                t2i_gen.click(
                    cb_generate_t2i,
                    [t2i_prompt, t2i_neg, t2i_width, t2i_height, t2i_steps, t2i_cfg,
                     t2i_sampler, t2i_scheduler, t2i_seed, t2i_shift],
                    [t2i_image, t2i_info],
                )
                t2i_recycle.click(_recycle_seed, None, t2i_seed)
                t2i_randomize.click(lambda: -1, None, t2i_seed)

            # ── img2img ────────────────────────────────────────────
            with gr.Tab("img2img"):
                # Top row: prompts left, generate + input image right
                with gr.Row():
                    with gr.Column(scale=3):
                        i2i_prompt = gr.Textbox(label="Prompt", lines=2)
                        i2i_neg = gr.Textbox(label="Negative prompt", lines=2)
                    with gr.Column(scale=2):
                        with gr.Row():
                            i2i_input = gr.Image(label="Input image", type="pil", scale=3)
                            with gr.Column(scale=1, min_width=120):
                                i2i_gen = gr.Button("Generate", variant="primary", elem_classes="generate-btn")
                # Main area: settings left, result right
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Accordion("Sampling", open=True):
                            i2i_sampler = gr.Dropdown(choices=_sampler_list(), value="dpmpp_2m", label="Sampler")
                            i2i_scheduler = gr.Dropdown(choices=_scheduler_list(), value=_scheduler_list()[0], label="Scheduler")
                            i2i_steps = gr.Slider(1, 150, value=25, step=1, label="Steps")
                            i2i_cfg = gr.Slider(1, 30, value=6.0, step=0.5, label="CFG scale")
                        with gr.Accordion("Denoising", open=True):
                            i2i_strength = gr.Slider(0.05, 1.0, value=0.6, step=0.05, label="Denoising strength")
                            with gr.Row():
                                i2i_seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0, scale=3)
                                i2i_recycle = gr.Button("♻️", scale=0, min_width=40)
                                i2i_randomize = gr.Button("🎲", scale=0, min_width=40)
                    with gr.Column(scale=2):
                        i2i_image = gr.Image(label="Result", type="pil", height=512)
                        i2i_info = gr.Textbox(label="Info", interactive=False, show_label=False)

                i2i_gen.click(
                    cb_generate_i2i,
                    [i2i_prompt, i2i_neg, i2i_input, i2i_strength, i2i_steps, i2i_cfg,
                     i2i_sampler, i2i_scheduler, i2i_seed],
                    [i2i_image, i2i_info],
                )
                i2i_recycle.click(_recycle_seed, None, i2i_seed)
                i2i_randomize.click(lambda: -1, None, i2i_seed)

            # ── inpaint ────────────────────────────────────────────
            with gr.Tab("inpaint"):
                # Top row: prompts left, generate + input images right
                with gr.Row():
                    with gr.Column(scale=3):
                        inp_prompt = gr.Textbox(label="Prompt", lines=2)
                        inp_neg = gr.Textbox(label="Negative prompt", lines=2)
                    with gr.Column(scale=2):
                        with gr.Row():
                            inp_input = gr.Image(label="Input image", type="pil", scale=1)
                            inp_mask = gr.Image(label="Mask (white = repaint)", type="pil", scale=1)
                            with gr.Column(scale=0, min_width=120):
                                inp_gen = gr.Button("Generate", variant="primary", elem_classes="generate-btn")
                # Main area: settings left, result right
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Accordion("Sampling", open=True):
                            inp_sampler = gr.Dropdown(choices=_sampler_list(), value="dpmpp_2m", label="Sampler")
                            inp_scheduler = gr.Dropdown(choices=_scheduler_list(), value=_scheduler_list()[0], label="Scheduler")
                            inp_steps = gr.Slider(1, 150, value=25, step=1, label="Steps")
                            inp_cfg = gr.Slider(1, 30, value=6.0, step=0.5, label="CFG scale")
                        with gr.Accordion("Denoising", open=True):
                            inp_strength = gr.Slider(0.05, 1.0, value=0.6, step=0.05, label="Denoising strength")
                            with gr.Row():
                                inp_seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0, scale=3)
                                inp_recycle = gr.Button("♻️", scale=0, min_width=40)
                                inp_randomize = gr.Button("🎲", scale=0, min_width=40)
                    with gr.Column(scale=2):
                        inp_image = gr.Image(label="Result", type="pil", height=512)
                        inp_info = gr.Textbox(label="Info", interactive=False, show_label=False)

                inp_gen.click(
                    cb_generate_inpaint,
                    [inp_prompt, inp_neg, inp_input, inp_mask, inp_strength, inp_steps,
                     inp_cfg, inp_sampler, inp_scheduler, inp_seed],
                    [inp_image, inp_info],
                )
                inp_recycle.click(_recycle_seed, None, inp_seed)
                inp_randomize.click(lambda: -1, None, inp_seed)

            # ── gallery ────────────────────────────────────────────
            with gr.Tab("Gallery"):
                gallery_refresh = gr.Button("↻ Refresh", scale=0)
                gallery = gr.Gallery(
                    value=[(str(f), f.name) for f in scan_outputs()],
                    label="Outputs",
                    columns=4, height=500, object_fit="contain",
                )
                selected_path = gr.State(None)
                with gr.Row():
                    gallery_load = gr.Button(
                        "Load metadata to workspace", variant="primary", scale=0,
                    )
                gallery_meta = gr.Textbox(
                    label="Raw metadata", lines=6, max_lines=10, interactive=False,
                )

                def _refresh_gallery():
                    files = scan_outputs()
                    return gr.Gallery(
                        value=[(str(f), f.name) for f in files],
                        columns=4, height=500, object_fit="contain",
                    )

                gallery_refresh.click(_refresh_gallery, None, gallery)

                def _on_select(evt: gr.SelectData):
                    files = scan_outputs()
                    idx = evt.index
                    if 0 <= idx < len(files):
                        path = str(files[idx])
                        meta = _read_png_metadata(path)
                        return path, meta
                    return None, ""

                gallery.select(
                    _on_select, None, [selected_path, gallery_meta],
                )

                gallery_load.click(
                    _recycle_from_gallery, [selected_path],
                    [
                        t2i_prompt, t2i_neg, t2i_width, t2i_height,
                        t2i_steps, t2i_cfg, t2i_sampler, t2i_scheduler, t2i_seed,
                        t2i_shift,
                        i2i_prompt, i2i_neg,
                        i2i_steps, i2i_cfg, i2i_sampler, i2i_scheduler, i2i_seed,
                        i2i_strength,
                        inp_prompt, inp_neg,
                        inp_steps, inp_cfg, inp_sampler, inp_scheduler, inp_seed,
                        inp_strength,
                    ],
                )

            # ── metadata reader ──────────────────────────────────────
            with gr.Tab("Metadata"):
                meta_input = gr.Image(label="Upload a PNG image", type="pil")
                meta_info = gr.Textbox(
                    label="Metadata", lines=15, max_lines=30, interactive=False,
                )
                with gr.Row():
                    meta_send = gr.Button("Send to txt2img", variant="primary", scale=0)

                def _format_metadata(img):
                    if img is None:
                        return "Upload a PNG image to view its metadata."

                    try:
                        with Image.open(img.filename) as raw:
                            info = dict(raw.info)
                    except Exception:
                        info = dict(img.info)

                    if not info:
                        return "No metadata found in this image."

                    lines = []

                    lines.append("═ ALL PNG METADATA KEYS ═")
                    for k, v in info.items():
                        val = str(v)
                        if len(val) > 600:
                            val = val[:600] + "..."
                        lines.append(f"  {k}: {val}")

                    auto1111 = info.get("parameters", "")
                    if auto1111:
                        lines.append("")
                        lines.append("═ AUTO1111 / FORGE PARSED ═")
                        for k, v in _parse_metadata(auto1111).items():
                            lines.append(f"  {k}: {v}")

                    comfyui = info.get("prompt", "")
                    if comfyui:
                        lines.append("")
                        lines.append("═ COMFYUI PARSED ═")
                        for k, v in _parse_comfyui_metadata(comfyui).items():
                            lines.append(f"  {k}: {v}")

                    return "\n".join(lines)

                def _meta_send_to_txt2img(img):
                    if img is None:
                        return [gr.update()] * 10

                    try:
                        with Image.open(img.filename) as raw:
                            info = dict(raw.info)
                    except Exception:
                        info = dict(img.info)

                    auto1111 = info.get("parameters", "")
                    if auto1111:
                        meta = _parse_metadata(auto1111)
                    else:
                        meta = _parse_comfyui_metadata(info.get("prompt", ""))

                    def _upd(v):
                        return gr.update(value=v) if v is not None and v != "" else gr.update()

                    prompt = meta.get("prompt", "")
                    neg = meta.get("negative_prompt", "")
                    try:
                        steps = int(meta.get("steps", 0)) or None
                    except Exception:
                        steps = None
                    try:
                        cfg = float(meta.get("cfg_scale", 0)) or None
                    except Exception:
                        cfg = None
                    sampler = meta.get("sampler", None)
                    scheduler = meta.get("scheduler", None)
                    try:
                        seed = int(meta.get("seed", -1))
                    except Exception:
                        seed = -1
                    try:
                        shift = float(meta.get("shift", 3.0))
                    except Exception:
                        shift = 3.0

                    width = gr.update()
                    height = gr.update()
                    size_str = meta.get("size", "")
                    if "x" in size_str:
                        try:
                            parts = size_str.split("x")
                            w = int(parts[0])
                            h = int(parts[1])
                            if 256 <= w <= 2048 and 256 <= h <= 2048:
                                width = gr.update(value=w)
                                height = gr.update(value=h)
                        except Exception:
                            pass

                    return [
                        _upd(prompt), _upd(neg), width, height,
                        _upd(steps), _upd(cfg), _upd(sampler), _upd(scheduler), _upd(seed),
                        _upd(shift),
                    ]

                meta_input.change(_format_metadata, meta_input, meta_info)
                meta_send.click(
                    _meta_send_to_txt2img, meta_input,
                    [
                        t2i_prompt, t2i_neg, t2i_width, t2i_height,
                        t2i_steps, t2i_cfg, t2i_sampler, t2i_scheduler, t2i_seed,
                        t2i_shift,
                    ],
                )

        # Anima defaults on first select since launch (wired after all tab vars exist)
        model_type.change(
            cb_anima_defaults, [model_type],
            [t2i_sampler, t2i_steps, t2i_cfg,
             i2i_sampler, i2i_steps, i2i_cfg,
             inp_sampler, inp_steps, inp_cfg],
        )
        model_type.change(
            cb_update_schedulers, [model_type],
            [t2i_scheduler, i2i_scheduler, inp_scheduler],
        )

    return app
