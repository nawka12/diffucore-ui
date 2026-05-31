"""Model manager — keeps the loaded model in memory across generations."""

from __future__ import annotations

import copy
import gc
import io
import re
import time
from contextlib import redirect_stderr
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from PIL import Image

from diffucore import (
    apply_lora,
    load_anima_checkpoint,
    load_checkpoint,
    ImageToImage,
    Inpaint,
    TextToImage,
)
from diffucore.runtime import DevicePolicy

from utils import checkpoint_path, lora_path, diffusion_model_path, vae_path, te_path

LORA_PROMPT_RE = re.compile(r"<lora:([^:]+):([^>]+)>")


def _append_tqdm_info(buf: io.StringIO, info: str) -> str:
    raw = buf.getvalue()
    if not raw.strip():
        return info
    lines = [l for l in raw.split("\r") if l.strip()]
    if lines:
        return f"{info}  |  {lines[-1].strip()}"
    return info

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
]

SCHEDULERS_SD = ["karras", "exponential", "polyexponential", "sgm_uniform", "simple"]
SCHEDULERS_ANIMA = ["flow", "sgm_uniform", "simple"]

# (family, native_res) tuples — used for defaults
MODEL_FAMILY_SD15 = "sd15"
MODEL_FAMILY_SDXL = "sdxl"
MODEL_FAMILY_ANIMA = "anima"


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
        self._clean_state: Optional[dict] = None
        self._offload = True
        self._vae_tile = True
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
        return f"{self._loaded.name} ({self._loaded.family}, {self._loaded.native_res}){vram}{lora_str}"

    # ── temporary LoRA (reForge-style via prompt) ────────────────

    def _save_clean_state(self) -> None:
        m = self._loaded.model
        sd = {"backbone": copy.deepcopy(m.backbone.state_dict())}
        if m.text_encoder is not None:
            sd["text_encoder"] = copy.deepcopy(m.text_encoder.state_dict())
        if m.text_encoder_2 is not None:
            sd["text_encoder_2"] = copy.deepcopy(m.text_encoder_2.state_dict())
        self._clean_state = sd

    def _restore_clean_state(self) -> None:
        if self._clean_state is None:
            return
        m = self._loaded.model
        m.backbone.load_state_dict(self._clean_state["backbone"])
        if m.text_encoder is not None and "text_encoder" in self._clean_state:
            m.text_encoder.load_state_dict(self._clean_state["text_encoder"])
        if m.text_encoder_2 is not None and "text_encoder_2" in self._clean_state:
            m.text_encoder_2.load_state_dict(self._clean_state["text_encoder_2"])
        self._loaded.applied_loras.clear()

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
        self._restore_clean_state()
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
        if self._loaded and self._clean_state is not None:
            self._restore_clean_state()

    # ── model loading ──────────────────────────────────────────────

    def load_model(self, model_name: str, offload: bool = True, vae_tile: bool = True) -> str:
        if self._loaded and self._loaded.name == model_name:
            return f"Model already loaded: {model_name}"

        self._unload()
        self._offload = offload
        self._vae_tile = vae_tile

        path = checkpoint_path(model_name)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        policy = DevicePolicy(device=self.device, compute_dtype=self.dtype, offload=offload, vae_tile=vae_tile)

        t0 = time.time()
        family = self._detect_family(path)
        model = load_checkpoint(str(path), policy=policy)
        elapsed = time.time() - t0

        self._loaded = LoadedModel(
            name=model_name,
            family=family,
            model=model,
            native_res=self._native_res(family),
        )
        self._clean_state = None
        self._save_clean_state()
        return f"Loaded {model_name} ({family}) in {elapsed:.1f}s"

    def load_anima(
        self, dit_name: str, vae_name: str, te_name: str,
        offload: bool = True, vae_tile: bool = True,
    ) -> str:
        label = f"Anima({dit_name})"
        if self._loaded and self._loaded.name == label:
            return f"Model already loaded: {label}"

        self._unload()
        self._offload = offload
        self._vae_tile = vae_tile

        dit_path = diffusion_model_path(dit_name)
        vae_file = vae_path(vae_name)
        te_file = te_path(te_name)
        for p in (dit_path, vae_file, te_file):
            if not p.exists():
                raise FileNotFoundError(f"Anima file not found: {p}")

        policy = DevicePolicy(device=self.device, compute_dtype=self.dtype, offload=offload, vae_tile=vae_tile)

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
        self._clean_state = None
        self._save_clean_state()
        return f"Loaded Anima in {elapsed:.1f}s  (DiT: {dit_name}, VAE: {vae_name}, TE: {te_name})"

    def _unload(self) -> None:
        if self._loaded is not None:
            del self._loaded.model
            self._loaded = None
        self._clean_state = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _detect_family(self, path: Path) -> str:
        name_lower = path.stem.lower()
        if "sdxl" in name_lower or "xl" in name_lower:
            return MODEL_FAMILY_SDXL
        return MODEL_FAMILY_SD15

    @staticmethod
    def _native_res(family: str) -> int:
        if family == MODEL_FAMILY_SDXL:
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
        progress_cb: Callable | None = None,
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
        )
        if self._loaded.family == MODEL_FAMILY_ANIMA:
            kwargs["shift"] = shift
        if progress_cb is not None:
            kwargs["callback"] = progress_cb
        buf = io.StringIO()
        with redirect_stderr(buf):
            image = gen(**kwargs)
        info = f"Seed: {seed} | {width}x{height} | {steps} steps"
        return image, _append_tqdm_info(buf, info)

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
        progress_cb: Callable | None = None,
    ) -> Tuple[Image.Image, str]:
        if not self._loaded:
            raise RuntimeError("No model loaded")
        seed = self._resolve_seed(seed)
        gen = ImageToImage(self._loaded.model)
        gen_kwargs = dict(
            prompt=prompt,
            input_image=input_image,
            negative_prompt=negative_prompt,
            strength=strength,
            steps=steps,
            cfg_scale=cfg_scale,
            sampler=sampler,
            scheduler=scheduler,
            seed=seed,
        )
        if progress_cb is not None:
            gen_kwargs["callback"] = progress_cb
        buf = io.StringIO()
        with redirect_stderr(buf):
            image = gen(**gen_kwargs)
        info = f"Seed: {seed} | strength={strength} | {steps} steps"
        return image, _append_tqdm_info(buf, info)

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
        progress_cb: Callable | None = None,
    ) -> Tuple[Image.Image, str]:
        if not self._loaded:
            raise RuntimeError("No model loaded")
        seed = self._resolve_seed(seed)
        gen = Inpaint(self._loaded.model)
        gen_kwargs = dict(
            prompt=prompt,
            input_image=input_image,
            mask_image=mask_image,
            negative_prompt=negative_prompt,
            strength=strength,
            steps=steps,
            cfg_scale=cfg_scale,
            sampler=sampler,
            scheduler=scheduler,
            seed=seed,
        )
        if progress_cb is not None:
            gen_kwargs["callback"] = progress_cb
        buf = io.StringIO()
        with redirect_stderr(buf):
            image = gen(**gen_kwargs)
        info = f"Seed: {seed} | inpainted | {steps} steps"
        return image, _append_tqdm_info(buf, info)


# ── singleton ──────────────────────────────────────────────────────
ENGINE = Engine()

