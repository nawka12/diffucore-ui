"""PNG metadata: write generation params, read them back, parse foreign formats.

Pure helpers (no web framework) so the server layer stays thin. Mirrors the
AUTO1111 / Forge ``parameters`` text format and also parses ComfyUI workflow
JSON found in the ``prompt`` chunk.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from PIL import Image

from diffucore import __version__ as _DIFFUCORE_VERSION

_ROOT = Path(__file__).resolve().parent.parent


def _git_short(cwd: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd, stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return ""


_UI_VERSION = "0.1.6"
_UI_COMMIT = _git_short(_ROOT)
_DIFF_COMMIT = _git_short(_ROOT / "diffucore")

UI_ID = f"diffucore-ui+{_UI_VERSION}" + (f"+{_UI_COMMIT}" if _UI_COMMIT else "")
DIFF_ID = f"diffucore+{_DIFF_COMMIT}" if _DIFF_COMMIT else f"diffucore+{_DIFFUCORE_VERSION}"


def _quote(text) -> str:
    """Wrap a value in JSON double-quotes only when it would otherwise break the
    flat comma/colon parser (free-text prompts). Mirrors AUTO1111's ``quote()``."""
    text = str(text)
    if any(c in text for c in (",", "\n", '"')):
        return json.dumps(text, ensure_ascii=False)
    return text


def _unquote(text: str):
    """Inverse of :func:`_quote`: decode a JSON-quoted value, else return as-is."""
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    return text


def _ordinal(n: int) -> str:
    """``2`` -> ``"2nd"``. Matches the per-unit suffix ADetailer writes."""
    suffix = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _detailer_fields(detail: dict) -> list[str]:
    """ADetailer-compatible ``Key: value`` pairs for the ``parameters`` line.

    This app shares one set of knobs across the stack, while ADetailer stores
    them per detection unit — so each model repeats the shared knobs under its
    own ``2nd``/``3rd`` suffix, which is what AUTO1111 + ADetailer expects.
    """
    neg = detail.get("neg", "")
    fields: list[str] = []
    for i, m in enumerate(detail.get("models") or []):
        s = "" if i == 0 else f" {_ordinal(i + 1)}"
        fields.append(f"ADetailer model{s}: {_quote(m.get('model', ''))}")
        if m.get("prompt"):
            fields.append(f"ADetailer prompt{s}: {_quote(m['prompt'])}")
        if neg:
            fields.append(f"ADetailer negative prompt{s}: {_quote(neg)}")
        fields.append(f"ADetailer confidence{s}: {detail.get('confidence')}")
        fields.append(f"ADetailer dilate erode{s}: {detail.get('dilation')}")
        fields.append(f"ADetailer mask blur{s}: {detail.get('blur')}")
        fields.append(f"ADetailer denoising strength{s}: {detail.get('strength')}")
        fields.append(f"ADetailer inpaint only masked{s}: True")
        fields.append(f"ADetailer inpaint padding{s}: {detail.get('padding')}")
        fields.append(f"ADetailer mask only top k largest{s}: {detail.get('maxDet')}")
    return fields


def _upscale_fields(upscale: dict) -> list[str]:
    """Upscale-compatible ``Key: value`` pairs for the ``parameters`` line."""
    fields = [
        f"Upscale scale: {upscale.get('scale')}",
        f"Upscale tile: {upscale.get('tile')}",
        f"Upscale overlap: {upscale.get('overlap')}",
        f"Upscale denoise: {upscale.get('denoise')}",
        f"Upscale TeaCache: {upscale.get('teacache')}",
        f"Upscale base: {upscale.get('base')}",
    ]
    if upscale.get("prompt"):
        fields.append(f"Upscale prompt: {_quote(upscale['prompt'])}")
    return fields


def format_metadata(gen_kwargs: dict, engine, detailer: dict | None = None,
                    upscale: dict | None = None) -> str:
    """Build the AUTO1111-style ``parameters`` string for a finished generation.

    Reads loaded-model name, resolved seed, and perf flags off ``engine``.
    """
    prompt = gen_kwargs.get("prompt", "")
    neg = gen_kwargs.get("negative_prompt", "")
    model = engine.loaded_name or "unknown"
    fields = []
    if "steps" in gen_kwargs:
        fields.append(f"Steps: {gen_kwargs['steps']}")
    fields.append(f"Sampler: {gen_kwargs.get('sampler', 'euler')}")
    fields.append(f"Scheduler: {gen_kwargs.get('scheduler', 'karras')}")
    fields.append(f"CFG scale: {gen_kwargs.get('cfg_scale', 7.0)}")
    # Guidance interval — only injected (and only written) when it actually
    # restricted CFG, so older metadata stays byte-identical.
    if "cfg_interval_start" in gen_kwargs:
        fields.append(f"CFG interval: {gen_kwargs['cfg_interval_start']}-"
                      f"{gen_kwargs.get('cfg_interval_end', 1.0)}")
    fields.append(f"Seed: {engine.last_seed}")
    if "width" in gen_kwargs and "height" in gen_kwargs:
        fields.append(f"Size: {gen_kwargs['width']}x{gen_kwargs['height']}")
    fields.append(f"Model: {model}")
    if "strength" in gen_kwargs:
        fields.append(f"Denoising strength: {gen_kwargs['strength']}")
    if "shift" in gen_kwargs:
        fields.append(f"Shift: {gen_kwargs['shift']}")
    if gen_kwargs.get("teacache_thresh", 0):
        calib = "" if gen_kwargs.get("teacache_use_coeffs", True) else " (raw)"
        fields.append(f"TeaCache: {gen_kwargs['teacache_thresh']}{calib}")
    if gen_kwargs.get("deepcache_interval", 1) > 1:
        fields.append(f"DeepCache: {gen_kwargs['deepcache_interval']}")
    if detailer:
        fields.extend(_detailer_fields(detailer))
    if upscale:
        fields.extend(_upscale_fields(upscale))
    fields.append(f"diffucore-ui: {UI_ID}")
    fields.append(DIFF_ID)
    flags = engine.perf_flags_str
    if flags != "default":
        fields.append(f"Perf flags: {flags}")
    return f"{prompt}\nNegative prompt: {neg}\n{', '.join(fields)}"


def read_png_metadata(path: str) -> str:
    """Return the AUTO1111 ``parameters`` chunk of a PNG, or ``""``.

    Uses a context manager so the file handle is released promptly — under heavy
    gallery use (lightbox paging, search indexing), the prior lazy-open form
    leaked descriptors until the GC closed the underlying PIL image."""
    with Image.open(path) as img:
        return img.info.get("parameters", "")


# key: value, where value is a JSON-quoted string (so commas/colons inside a
# free-text prompt survive) or a bare run up to the next comma. Mirrors AUTO1111.
_PARAM_RE = re.compile(r'\s*([\w \-/]+?):\s*("(?:\\.|[^"\\])*"|[^,]*)\s*(?:,|$)')


def parse_metadata(params_str: str) -> dict:
    """Parse an AUTO1111 / Forge ``parameters`` string into a flat dict.

    Keys are lowercased with spaces turned to underscores
    (``"CFG scale"`` -> ``"cfg_scale"``).
    """
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
    for m in _PARAM_RE.finditer(fields_str):
        key = m.group(1).strip().lower().replace(" ", "_")
        result[key] = _unquote(m.group(2).strip())
    return result


def parse_comfyui_metadata(prompt_json: str) -> dict:
    """Parse a ComfyUI workflow (the ``prompt`` PNG chunk) into a flat dict."""
    try:
        data = json.loads(prompt_json)
    except json.JSONDecodeError:
        return {}
    result = {}
    for node in data.values():
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


def extract_detailer(meta: dict) -> dict:
    """Reconstruct this app's detailer settings from the ADetailer-compatible
    keys on a parsed ``parameters`` line. ``{}`` when no detailer ran.

    Models come from the ``ADetailer model``/``... 2nd``/... stack; the shared
    knobs and negative are read from the first unit (this app shares them).
    """
    if "adetailer_model" not in meta:
        return {}
    models = []
    i = 0
    while True:
        suf = "" if i == 0 else f"_{_ordinal(i + 1)}"
        if f"adetailer_model{suf}" not in meta:
            break
        models.append({
            "model": meta[f"adetailer_model{suf}"],
            "prompt": meta.get(f"adetailer_prompt{suf}", ""),
        })
        i += 1
    det = {
        "enabled": True,
        "models": models,
        "neg": meta.get("adetailer_negative_prompt", ""),
    }
    for key, src, cast in (
        ("confidence", "adetailer_confidence", float),
        ("strength", "adetailer_denoising_strength", float),
        ("dilation", "adetailer_dilate_erode", int),
        ("padding", "adetailer_inpaint_padding", int),
        ("blur", "adetailer_mask_blur", int),
        ("maxDet", "adetailer_mask_only_top_k_largest", int),
    ):
        try:
            det[key] = cast(meta[src])
        except (KeyError, TypeError, ValueError):
            pass
    return det


def extract_upscale(meta: dict) -> dict:
    """Reconstruct this app's upscaler settings from the ``Upscale …`` keys on a
    parsed ``parameters`` line. ``{}`` when no upscaler ran.

    Mirrors :func:`extract_detailer`: ``base`` "Lanczos" maps back to the form's
    empty default (no ESRGAN model); the rest are cast to the form's types.
    """
    if "upscale_scale" not in meta:
        return {}
    up = {"enabled": True}
    for key, src, cast in (
        ("scale", "upscale_scale", float),
        ("denoise", "upscale_denoise", float),
        ("tile", "upscale_tile", int),
        ("overlap", "upscale_overlap", int),
        ("teacache", "upscale_teacache", float),
    ):
        try:
            up[key] = cast(meta[src])
        except (KeyError, TypeError, ValueError):
            pass
    base = meta.get("upscale_base", "")
    up["base"] = "" if base == "Lanczos" else base
    up["prompt"] = meta.get("upscale_prompt", "")
    return up


def workspace_fields(meta: dict) -> dict:
    """Normalise a parsed metadata dict into typed workspace form values.

    Returns only the keys that were present and parseable, so the caller can
    leave the rest of the form untouched.
    """
    out: dict = {}
    if meta.get("prompt"):
        out["prompt"] = meta["prompt"]
    if "negative_prompt" in meta:
        out["neg"] = meta["negative_prompt"]
    try:
        steps = int(meta.get("steps", 0))
        if steps:
            out["steps"] = steps
    except (TypeError, ValueError):
        pass
    try:
        cfg = float(meta.get("cfg_scale", 0))
        if cfg:
            out["cfg"] = cfg
    except (TypeError, ValueError):
        pass
    if meta.get("sampler"):
        out["sampler"] = meta["sampler"]
    if meta.get("scheduler"):
        out["scheduler"] = meta["scheduler"]
    try:
        out["seed"] = int(meta.get("seed", -1))
    except (TypeError, ValueError):
        pass
    try:
        out["shift"] = float(meta["shift"])
    except (TypeError, ValueError, KeyError):
        pass
    try:
        out["strength"] = float(meta["denoising_strength"])
    except (TypeError, ValueError, KeyError):
        pass
    # TeaCache is only written when it ran: "TeaCache: <thresh>" (calibrated) or
    # "TeaCache: <thresh> (raw)". Absent means it was off — additive, like the
    # detailer/upscale chunks, so we don't toggle it off on older metadata.
    tc = meta.get("teacache")
    if tc is not None:
        raw = str(tc).strip()
        try:
            thresh = float(raw.split()[0])
        except (ValueError, IndexError):
            thresh = 0.0
        if thresh > 0:
            out["teacacheOn"] = True
            out["teacache"] = thresh
            out["teacacheCalibrated"] = not raw.endswith("(raw)")
    # DeepCache is only written when it ran: "DeepCache: <interval>". Absent
    # means off — additive, like TeaCache above.
    dc = meta.get("deepcache")
    if dc is not None:
        try:
            interval = int(str(dc).strip())
        except (TypeError, ValueError):
            interval = 1
        if interval > 1:
            out["deepcacheOn"] = True
            out["deepcache"] = interval
    size_str = meta.get("size", "")
    if "x" in size_str:
        try:
            w_str, h_str = size_str.split("x")[:2]
            w, h = int(w_str), int(h_str)
            if 256 <= w <= 2048 and 256 <= h <= 2048:
                out["width"], out["height"] = w, h
        except ValueError:
            pass
    detailer = extract_detailer(meta)
    if detailer:
        out["detailer"] = detailer
    upscale = extract_upscale(meta)
    if upscale:
        out["upscale"] = upscale
    return out
