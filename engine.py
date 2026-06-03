"""Model manager — keeps the loaded model in memory across generations."""

from __future__ import annotations

import ctypes
import ctypes.util
import gc
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple


def _resolve_malloc_trim():
    """glibc ``malloc_trim`` if available, else None. Used after each generation to
    return free heap pages to the OS — offload round-trips churn many GB of CPU
    weights per call, and glibc's allocator otherwise grows RSS indefinitely."""
    try:
        libc_path = ctypes.util.find_library("c")
        if not libc_path:
            return None
        libc = ctypes.CDLL(libc_path)
        if not hasattr(libc, "malloc_trim"):
            return None
        libc.malloc_trim.argtypes = [ctypes.c_size_t]
        libc.malloc_trim.restype = ctypes.c_int
        return libc.malloc_trim
    except OSError:
        return None


_MALLOC_TRIM = _resolve_malloc_trim()

_ROOT = Path(__file__).resolve().parent
_LOCAL_DIFFUCORE_SRC = _ROOT / "diffucore" / "src"
if _LOCAL_DIFFUCORE_SRC.exists():
    sys.path.insert(0, str(_LOCAL_DIFFUCORE_SRC))

import torch
from PIL import Image

from diffucore import (
    apply_lora,
    clear_loras as clear_bundle_loras,
    load_anima_checkpoint,
    load_checkpoint,
    load_flux_checkpoint,
    ImageToImage,
    Inpaint,
    TextToImage,
)
from diffucore.runtime import DevicePolicy

from utils import checkpoint_path, lora_path, diffusion_model_path, vae_path, te_path


LORA_PROMPT_RE = re.compile(r"<lora:([^:]+):([^>]+)>")

SAMPLERS = [
    "euler",
    "heun",
    "euler_ancestral",
    "dpm_2",
    "dpm_2_ancestral",
    "dpmpp_2m",
    "dpmpp_sde",
    "dpmpp_2m_sde",
    "dpmpp_3m_sde",
    "er_sde",
    "secant",
]

SCHEDULERS_SD = ["karras", "exponential", "polyexponential", "sgm_uniform", "simple"]
SCHEDULERS_ANIMA = ["flow", "acas", "sgm_uniform", "simple"]
SCHEDULERS_FLUX = ["flux", "flow", "sgm_uniform", "simple"]

# (family, native_res) tuples — used for defaults
MODEL_FAMILY_SD15 = "sd15"
MODEL_FAMILY_SDXL = "sdxl"
MODEL_FAMILY_ANIMA = "anima"
MODEL_FAMILY_FLUX1 = "flux1"
MODEL_FAMILY_FLUX2 = "flux2"
_FLUX_FAMILIES = (MODEL_FAMILY_FLUX1, MODEL_FAMILY_FLUX2)


@dataclass
class LoadedModel:
    name: str
    family: str
    model: object
    native_res: int
    applied_loras: List[str] = field(default_factory=list)


class Engine:
    def __init__(self, device: str = "cuda", dtype_str: str = "float16"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(torch, dtype_str)
        self._loaded: Optional[LoadedModel] = None
        self._offload: bool | str = True
        self._vae_tile = True
        self._compile = False
        self._cuda_graphs = False
        self._channels_last = False
        self._tf32 = False
        self._last_seed: int = -1
        self._anima_defaults_applied: bool = False

    # ── state queries ──────────────────────────────────────────────

    @property
    def loaded_name(self) -> Optional[str]:
        return self._loaded.name if self._loaded else None

    @property
    def loaded_family(self) -> Optional[str]:
        return self._loaded.family if self._loaded else None

    @property
    def applied_loras(self) -> List[str]:
        return list(self._loaded.applied_loras) if self._loaded else []

    @property
    def last_seed(self) -> int:
        return self._last_seed

    @property
    def available_schedulers(self) -> List[str]:
        if self._loaded and self._loaded.family == MODEL_FAMILY_ANIMA:
            return SCHEDULERS_ANIMA
        if self._loaded and self._loaded.family in _FLUX_FAMILIES:
            return SCHEDULERS_FLUX
        return SCHEDULERS_SD

    def status_text(self) -> str:
        if not self._loaded:
            return "No model loaded"
        vram = ""
        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated() / 1024**3
            vram = f"  |  VRAM {used:.1f} GB"
        lora_str = ""
        if self._loaded.applied_loras:
            lora_str = f"  |  LoRAs: {', '.join(self._loaded.applied_loras)}"
        flags = self._perf_flag_summary().strip()
        return f"{self._loaded.name} ({self._loaded.family}, {self._loaded.native_res}){vram}{lora_str}{'  ' + flags if flags else ''}"

    # ── temporary LoRA (reForge-style via prompt) ────────────────

    @staticmethod
    def parse_lora_prompt(prompt: str) -> tuple[str, list[tuple[str, float]]]:
        loras = []
        def _extract(m):
            name = m.group(1)
            mult = float(m.group(2))
            loras.append((name, mult))
            return ""
        cleaned = LORA_PROMPT_RE.sub(_extract, prompt)
        return cleaned, loras

    def apply_temp_loras(self, loras: list[tuple[str, float]]) -> str:
        if not self._loaded:
            return "No model loaded"
        clear_bundle_loras(self._loaded.model)
        self._loaded.applied_loras.clear()
        msgs = []
        for name, mult in loras:
            path = lora_path(name)
            if not path.exists():
                msgs.append(f"LoRA '{name}' not found")
                continue
            report = apply_lora(self._loaded.model, str(path), multiplier=mult)
            self._loaded.applied_loras.append(name)
            msgs.append(f"{name}@{mult}: {report.applied} matched")
        return " | ".join(msgs) if msgs else "No LoRAs"

    def clear_temp_loras(self) -> None:
        if self._loaded:
            clear_bundle_loras(self._loaded.model)
            self._loaded.applied_loras.clear()

    # ── model loading ──────────────────────────────────────────────

    def load_model(
        self, model_name: str, offload: bool | str = True, vae_tile: bool = True,
        compile: bool = False, cuda_graphs: bool = False,
        channels_last: bool = False, tf32: bool = False,
    ) -> str:
        if self._loaded and self._loaded.name == model_name:
            return f"Model already loaded: {model_name}"

        if compile and offload is True:
            offload = "encoders"

        self._unload()
        self._offload = offload
        self._vae_tile = vae_tile
        self._compile = compile
        self._cuda_graphs = cuda_graphs
        self._channels_last = channels_last
        self._tf32 = tf32

        path = checkpoint_path(model_name)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        policy = DevicePolicy(
            device=self.device, compute_dtype=self.dtype,
            offload=offload, vae_tile=vae_tile,
            compile=compile, cuda_graphs=cuda_graphs,
            channels_last=channels_last, tf32=tf32,
        )

        t0 = time.time()
        model = load_checkpoint(str(path), policy=policy)
        elapsed = time.time() - t0

        # Family comes from the detected architecture (sd15/sdxl/flux1/flux2), so
        # an all-in-one FLUX checkpoint dropped in checkpoints/ is recognised too.
        family = model.spec.architecture
        self._loaded = LoadedModel(
            name=model_name,
            family=family,
            model=model,
            native_res=self._native_res(family),
        )
        flags = self._perf_flag_summary()
        return f"Loaded {model_name} ({family}) in {elapsed:.1f}s{flags}"

    def load_anima(
        self, dit_name: str, vae_name: str, te_name: str,
        offload: bool = True, vae_tile: bool = True,
        compile: bool = False, cuda_graphs: bool = False,
    ) -> str:
        label = f"Anima({dit_name})"
        if self._loaded and self._loaded.name == label:
            return f"Model already loaded: {label}"

        if compile and offload is True:
            offload = "encoders"

        self._unload()
        self._offload = offload
        self._vae_tile = vae_tile
        self._compile = compile
        self._cuda_graphs = cuda_graphs
        self._channels_last = False
        self._tf32 = False

        dit_path = diffusion_model_path(dit_name)
        vae_file = vae_path(vae_name)
        te_file = te_path(te_name)
        for p in (dit_path, vae_file, te_file):
            if not p.exists():
                raise FileNotFoundError(f"Anima file not found: {p}")

        policy = DevicePolicy(
            device=self.device, compute_dtype=self.dtype,
            offload=offload, vae_tile=vae_tile,
            compile=compile, cuda_graphs=cuda_graphs,
        )

        t0 = time.time()
        model = load_anima_checkpoint(
            str(dit_path), str(vae_file), str(te_file), policy=policy,
        )
        elapsed = time.time() - t0

        self._loaded = LoadedModel(
            name=label,
            family=MODEL_FAMILY_ANIMA,
            model=model,
            native_res=1024,
        )
        flags = self._perf_flag_summary()
        return f"Loaded Anima in {elapsed:.1f}s{flags}  (DiT: {dit_name}, VAE: {vae_name}, TE: {te_name})"

    def load_flux(
        self, dit_name: str, vae_name: str, te_name: str, clip_name: str | None = None,
        offload: bool | str = True, vae_tile: bool = True,
        compile: bool = False, cuda_graphs: bool = False,
    ) -> str:
        """Load a split-file FLUX model. ``te_name`` is the primary text encoder
        (T5-XXL for FLUX.1, Mistral-3 for FLUX.2); ``clip_name`` is the CLIP-L
        encoder (FLUX.1 only — ignored for FLUX.2). The detector picks which path
        applies from the transformer."""
        label = f"FLUX({dit_name})"
        if self._loaded and self._loaded.name == label:
            return f"Model already loaded: {label}"

        if compile and offload is True:
            offload = "encoders"

        self._unload()
        self._offload = offload
        self._vae_tile = vae_tile
        self._compile = compile
        self._cuda_graphs = cuda_graphs
        self._channels_last = False
        self._tf32 = False

        dit_path = diffusion_model_path(dit_name)
        vae_file = vae_path(vae_name)
        te_file = te_path(te_name)
        for p in (dit_path, vae_file, te_file):
            if not p.exists():
                raise FileNotFoundError(f"FLUX file not found: {p}")
        clip_file = None
        if clip_name and not clip_name.startswith("("):
            clip_file = te_path(clip_name)
            if not clip_file.exists():
                raise FileNotFoundError(f"FLUX CLIP file not found: {clip_file}")

        policy = DevicePolicy(
            device=self.device, compute_dtype=self.dtype,
            offload=offload, vae_tile=vae_tile,
            compile=compile, cuda_graphs=cuda_graphs,
        )

        t0 = time.time()
        # te_file is passed as both T5 and Mistral candidate; the loader uses
        # whichever the detected architecture needs.
        model = load_flux_checkpoint(
            transformer_path=str(dit_path), vae_path=str(vae_file),
            t5_path=str(te_file), mistral_path=str(te_file),
            clip_path=str(clip_file) if clip_file else None,
            policy=policy,
        )
        elapsed = time.time() - t0

        family = model.spec.architecture
        self._loaded = LoadedModel(
            name=label, family=family, model=model, native_res=1024,
        )
        flags = self._perf_flag_summary()
        return f"Loaded {family} in {elapsed:.1f}s{flags}  (DiT: {dit_name}, VAE: {vae_name}, TE: {te_name})"

    def _perf_flag_summary(self) -> str:
        flags = []
        if self._compile:
            flags.append("compile")
        if self._cuda_graphs:
            flags.append("cuda_graphs")
        if self._channels_last:
            flags.append("channels_last")
        if self._tf32:
            flags.append("tf32")
        if self._offload is True:
            flags.append("offload=full")
        elif self._offload == "encoders":
            flags.append("offload=encoders")
        elif self._offload == "stream":
            flags.append("offload=stream")
        elif not self._offload:
            flags.append("no-offload")
        return f"  [{', '.join(flags)}]" if flags else ""

    @property
    def perf_flags_str(self) -> str:
        parts = []
        if self._compile:
            parts.append("compile")
        if self._cuda_graphs:
            parts.append("cuda_graphs")
        if self._channels_last:
            parts.append("channels_last")
        if self._tf32:
            parts.append("tf32")
        return ", ".join(parts) if parts else "default"

    def _unload(self) -> None:
        if self._loaded is not None:
            del self._loaded.model
            self._loaded = None
        self._reclaim_memory()

    def _reclaim_memory(self) -> None:
        """Drop dead refs and hand free heap pages back to the OS. Called after
        each generation so glibc doesn't grow RSS indefinitely from the
        multi-GB CPU malloc/free churn the offload round-trips produce."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if _MALLOC_TRIM is not None:
            _MALLOC_TRIM(0)

    @staticmethod
    def _native_res(family: str) -> int:
        if family in (MODEL_FAMILY_SDXL, MODEL_FAMILY_ANIMA, *_FLUX_FAMILIES):
            return 1024
        return 512

    # ── LoRA ───────────────────────────────────────────────────────

    def apply_lora(self, lora_name: str, multiplier: float = 1.0) -> str:
        if not self._loaded:
            raise RuntimeError("No model loaded")
        path = lora_path(lora_name)
        if not path.exists():
            raise FileNotFoundError(f"LoRA not found: {path}")
        report = apply_lora(self._loaded.model, str(path), multiplier=multiplier)
        self._loaded.applied_loras.append(lora_name)
        return f"Applied {lora_name}: {report}"

    def clear_loras(self) -> str:
        if not self._loaded:
            return "No model loaded"
        self._unload()
        return "All LoRAs cleared (model reloaded without adapters)"

    # ── generation ─────────────────────────────────────────────────

    def _resolve_seed(self, seed: int) -> int:
        if seed == -1:
            seed = int(torch.randint(0, 2**32 - 1, (1,)).item())
        self._last_seed = seed
        return seed

    def generate_t2i(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        steps: int = 25,
        cfg_scale: float = 6.0,
        sampler: str = "dpmpp_2m",
        scheduler: str = "karras",
        seed: int = -1,
        shift: float = 1.0,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Tuple[Image.Image, str]:
        if not self._loaded:
            raise RuntimeError("No model loaded")
        seed = self._resolve_seed(seed)
        gen = TextToImage(self._loaded.model)
        kwargs: dict = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            cfg_scale=cfg_scale,
            width=width,
            height=height,
            sampler=sampler,
            scheduler=scheduler,
            seed=seed,
            progress_callback=progress_callback,
            return_info=True,
        )
        if self._loaded.family in (MODEL_FAMILY_ANIMA, *_FLUX_FAMILIES):
            kwargs["shift"] = shift
        try:
            image, pipeline_info = gen(**kwargs)
        finally:
            self._reclaim_memory()
        info = f"Seed: {seed} | {width}x{height} | {steps} steps | VAE: {pipeline_info.vae_decode_mode}"
        return image, info

    def generate_i2i(
        self,
        prompt: str,
        input_image: Image.Image,
        negative_prompt: str = "",
        strength: float = 0.6,
        steps: int = 25,
        cfg_scale: float = 6.0,
        sampler: str = "dpmpp_2m",
        scheduler: str = "karras",
        seed: int = -1,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Tuple[Image.Image, str]:
        if not self._loaded:
            raise RuntimeError("No model loaded")
        if self._loaded.family in _FLUX_FAMILIES:
            raise RuntimeError("FLUX supports text-to-image only in this build")
        seed = self._resolve_seed(seed)
        gen = ImageToImage(self._loaded.model)
        gen_kwargs = dict(
            prompt=prompt,
            init_image=input_image,
            negative_prompt=negative_prompt,
            strength=strength,
            steps=steps,
            cfg_scale=cfg_scale,
            sampler=sampler,
            scheduler=scheduler,
            seed=seed,
            progress_callback=progress_callback,
            return_info=True,
        )
        try:
            image, pipeline_info = gen(**gen_kwargs)
        finally:
            self._reclaim_memory()
        info = f"Seed: {seed} | strength={strength} | {steps} steps | VAE: {pipeline_info.vae_decode_mode}"
        return image, info

    def generate_inpaint(
        self,
        prompt: str,
        input_image: Image.Image,
        mask_image: Image.Image,
        negative_prompt: str = "",
        strength: float = 0.6,
        steps: int = 25,
        cfg_scale: float = 6.0,
        sampler: str = "dpmpp_2m",
        scheduler: str = "karras",
        seed: int = -1,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Tuple[Image.Image, str]:
        if not self._loaded:
            raise RuntimeError("No model loaded")
        if self._loaded.family in _FLUX_FAMILIES:
            raise RuntimeError("FLUX supports text-to-image only in this build")
        seed = self._resolve_seed(seed)
        gen = Inpaint(self._loaded.model)
        gen_kwargs = dict(
            prompt=prompt,
            init_image=input_image,
            mask_image=mask_image,
            negative_prompt=negative_prompt,
            strength=strength,
            steps=steps,
            cfg_scale=cfg_scale,
            sampler=sampler,
            scheduler=scheduler,
            seed=seed,
            progress_callback=progress_callback,
            return_info=True,
        )
        try:
            image, pipeline_info = gen(**gen_kwargs)
        finally:
            self._reclaim_memory()
        info = f"Seed: {seed} | inpainted | {steps} steps | VAE: {pipeline_info.vae_decode_mode}"
        return image, info


# ── singleton ──────────────────────────────────────────────────────
ENGINE = Engine()
