"""What each model file is, from its safetensors header alone, so the model rack
can pick the family and matching VAE / text encoder by itself."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Tuple

from diffucore.loading import detect_architecture

Shapes = Mapping[str, Tuple[int, ...]]

# A safetensors header is a few MB at most; anything bigger isn't one.
_MAX_HEADER = 64 * 1024 * 1024
_cache: Dict[Tuple[str, str, int, int], Optional[str]] = {}


def _shapes(path: Path) -> Optional[Dict[str, Tuple[int, ...]]]:
    if path.suffix.lower() != ".safetensors":
        return None
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if n > _MAX_HEADER:
                return None
            header = json.loads(f.read(n))
    except (OSError, ValueError, struct.error):
        return None
    if not isinstance(header, dict):
        return None
    header.pop("__metadata__", None)
    return {k: tuple(v.get("shape", ())) for k, v in header.items() if isinstance(v, dict)}


def model_arch(shapes: Shapes) -> Optional[str]:
    """"sd15" / "sdxl" / "anima" / "krea2" / "flux1" / "flux2", or None."""
    try:
        return detect_architecture(shapes).architecture
    except (ValueError, NotImplementedError):
        pass
    if "img_in.weight" in shapes:  # diffusers-named Krea 2
        from diffucore.models.krea2_dit import from_diffusers_name
        try:
            return detect_architecture({from_diffusers_name(k): v for k, v in shapes.items()}).architecture
        except (ValueError, NotImplementedError):
            pass
    return None


def vae_kind(shapes: Shapes) -> Optional[str]:
    """"qwen_image" (Anima / Krea 2), "qwen2d" (Anzhc's 2-D Qwen-Image VAE),
    "flux1", "flux2", or None."""
    if any(len(s) == 5 for s in shapes.values()):
        return "qwen_image"  # Wan-style 3-D causal convs
    conv_in = shapes.get("decoder.conv_in.weight")
    if conv_in is None or len(conv_in) != 4:
        return None
    if any(".to_qkv." in k for k in shapes) and conv_in[1] == 16:
        return "qwen2d"
    if "decoder.mid.attn_1.q.weight" in shapes:
        return {16: "flux1", 32: "flux2"}.get(conv_in[1])
    return None


def te_kind(shapes: Shapes) -> Optional[str]:
    """"qwen3" (Anima), "qwen35" (Anima's experimental swap), "qwen3vl" (Krea 2),
    "t5" / "clip_l" (FLUX.1), "flux2_lm" (FLUX.2's Mistral or Qwen3), or None."""
    keys = list(shapes)
    if any("linear_attn.A_log" in k for k in keys):
        return "qwen35"
    if any(k.endswith("encoder.block.0.layer.0.SelfAttention.q.weight") for k in keys):
        return "t5"
    if any(k.endswith("text_model.embeddings.token_embedding.weight") for k in keys):
        return "clip_l"
    if any(k.endswith("layers.0.self_attn.q_norm.weight") for k in keys):
        if any(k.startswith(("model.language_model.", "language_model.")) for k in keys):
            return "qwen3vl"
        emb = next((shapes[k] for k in keys if k.endswith("embed_tokens.weight")), None)
        return "qwen3" if emb is not None and emb[1] == 1024 else "flux2_lm"
    if any(k.endswith("layers.0.self_attn.q_proj.weight") for k in keys):
        return "flux2_lm"  # Mistral: no per-head q_norm
    return None


def _classify(path: Path, what: str, fn: Callable[[Shapes], Optional[str]]) -> Optional[str]:
    try:
        st = path.stat()
    except OSError:
        return None
    key = (str(path), what, st.st_size, st.st_mtime_ns)
    if key not in _cache:
        shapes = _shapes(path)
        _cache[key] = fn(shapes) if shapes else None
    return _cache[key]


def kinds(directory: Path, names: List[str], what: str) -> Dict[str, Optional[str]]:
    fn = {"model": model_arch, "vae": vae_kind, "te": te_kind}[what]
    return {n: _classify(directory / n, what, fn) for n in names}
