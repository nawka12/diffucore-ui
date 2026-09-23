"""Model manager: keeps the loaded model in memory across generations."""

from __future__ import annotations

import ctypes
import ctypes.util
import gc
import json
import logging
import math
import os
import re
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

log = logging.getLogger("diffucore.engine")


def _resolve_malloc_trim():
    """glibc ``malloc_trim``, or None. Offload round-trips churn GBs of CPU
    weights, and glibc otherwise grows RSS indefinitely."""
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

_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_DIFFUCORE_SRC = _ROOT / "diffucore" / "src"
if _LOCAL_DIFFUCORE_SRC.exists():
    sys.path.insert(0, str(_LOCAL_DIFFUCORE_SRC))

import numpy as np
import torch
from PIL import Image, ImageFilter

# Optional ESRGAN-family base upscalers; without spandrel only Lanczos is offered.
try:
    import spandrel as _spandrel
except Exception:
    _spandrel = None

from diffucore import (
    anima_calibrate_oss,
    anima_calibrate_teacache,
    apply_lora,
    clear_loras as clear_bundle_loras,
    fa2_turing_available,
    load_anima_checkpoint,
    load_checkpoint,
    load_flux_checkpoint,
    ImageToImage,
    Inpaint,
    TextToImage,
)
from diffucore.runtime import ConditioningCache, DevicePolicy

from utils import checkpoint_path, lora_path, diffusion_model_path, vae_path, te_path


LORA_PROMPT_RE = re.compile(r"<lora:([^:]+):([^>]+)>")

# Flow families (Anima, FLUX) drive everything except "ddpm" (VP/VE-only). Keep
# these lists in sync with the pipelines' _ANIMA_SAMPLERS / _FLUX_SAMPLERS.
# GUIDE.md describes each sampler and scheduler.
SAMPLERS_SD = [
    "euler",
    "euler_ancestral",
    "heun",
    "heunpp2",
    "dpm_2",
    "dpm_2_ancestral",
    "dpmpp_2s_ancestral",
    "dpmpp_2m",
    "dpmpp_2m_sde",
    "dpmpp_2m_sde_heun",
    "dpmpp_sde",
    "dpmpp_3m_sde",
    "ipndm",
    "ipndm_v",
    "res_multistep",
    "res_multistep_ancestral",
    "lumen",
    "gradient_estimation",
    "stork2",
    "infinity",
    "infinity_realism",
    "infinity_nano",
    "infinity_omega",
    "infinity_aether",
    "lms",
    "er_sde",
    "ddpm",
    "lcm",
    "sa_solver",
    "sa_solver_pece",
    "secant",
    "exp_heun_2_x0",
    "uni_pc",
    "uni_pc_bh2",
    "cogent",
    "cogent3",
    "cogent3_pump",
    "cogent3_pump_rate",
]
_SAMPLERS_SD_ONLY = set()
SAMPLERS_FLOW = [s for s in SAMPLERS_SD if s != "ddpm" and s not in _SAMPLERS_SD_ONLY]
# These need a [B, C, H, W] latent (2-D convolutions, per-channel spatial
# statistics). FLUX samples a patchified token sequence, so it doesn't get them.
_SAMPLERS_4D_ONLY = {"infinity_nano", "infinity_omega", "infinity_realism",
                     "infinity_aether", "cogent3_pump", "cogent3_pump_rate"}
# Anima-only additions.
SAMPLERS_ANIMA = SAMPLERS_FLOW + ["euler_ancestral_anneal", "secant_anneal",
                                  "dpmpp_2m_anneal", "uni_pc_anneal"]
SAMPLERS_FLUX = [s for s in SAMPLERS_FLOW if s not in _SAMPLERS_4D_ONLY]

SCHEDULERS_SD = ["karras", "exponential", "polyexponential", "kl_optimal",
                 "align_your_steps", "sgm_uniform", "simple", "normal",
                 "infinity", "infinity_htds", "ddim_uniform", "linear_quadratic"]
# align_your_steps is SD/SDXL only (its tables are VE-scale). "oss" needs a
# one-time calibration per (model, steps, resolution, shift); see calibrate_oss.
# Flow families omit "ddim_uniform": it starts below σ_max, and the flow
# pipelines init from pure noise (σ_max == 1).
SCHEDULERS_ANIMA = ["flow", "flow_dyn", "oss", "sgm_uniform", "simple",
                    "normal", "infinity", "infinity_htds", "kl_optimal",
                    "linear_quadratic", "smoothstep", "beta", "beta_mix",
                    "pump_dual", "pump_taper"]
SCHEDULERS_FLUX = ["flux", "flow", "sgm_uniform", "simple", "normal",
                   "infinity", "infinity_htds", "kl_optimal", "linear_quadratic"]

# Calibrated OSS schedules: one JSON of descending sigmas per
# (model, steps, resolution, shift), written by calibrate_oss.py.
_OSS_CACHE_DIR = _ROOT / "models" / "oss_cache"


def oss_cache_path(name: str, steps: int, width: int, height: int, shift: float) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    return _OSS_CACHE_DIR / f"{safe}__{steps}s_{width}x{height}_shift{shift:g}.json"

# TeaCache coefficients are per architecture: one JSON per family, with an
# optional per-checkpoint override.
_TEACACHE_CACHE_DIR = _ROOT / "models" / "teacache_cache"


def teacache_cache_path(family: str) -> Path:
    return _TEACACHE_CACHE_DIR / f"{family}.json"


def teacache_override_path(name: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    return _TEACACHE_CACHE_DIR / f"{safe}.json"


def _write_cache_json(path: Path, values: list) -> None:
    """Write a calibration cache atomically (temp file + os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    try:
        tmp.write_text(json.dumps(values))
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _read_cache_json(path: Path) -> Optional[list]:
    """Load a calibration cache, or None if it's absent, unreadable, or not a
    list of finite numbers."""
    try:
        with open(path) as f:
            values = json.load(f)
    except (OSError, ValueError):
        log.warning("ignoring unreadable calibration cache %s", path.name)
        return None
    if not _finite_series(values):
        log.warning("ignoring non-finite calibration cache %s", path.name)
        return None
    return values


def _finite_series(values) -> bool:
    """True if ``values`` is a non-empty list of finite real numbers."""
    return (isinstance(values, list) and len(values) > 0
            and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                    and math.isfinite(v) for v in values))

MODEL_FAMILY_SD15 = "sd15"
MODEL_FAMILY_SDXL = "sdxl"
MODEL_FAMILY_ANIMA = "anima"

# TeaCache keeps tensors from inside the compiled forward alive across steps.
# Under CUDA Graphs every replay overwrites them, and no call-site clone can fix
# state stored inside the graph.
_TEACACHE_CUDA_GRAPHS_ERROR = (
    "TeaCache is incompatible with CUDA Graphs (its cached tensors are "
    "overwritten by each graph replay). Disable TeaCache, or reload the "
    "model without the CUDA Graphs flag."
)
MODEL_FAMILY_FLUX1 = "flux1"
MODEL_FAMILY_FLUX2 = "flux2"
_FLUX_FAMILIES = (MODEL_FAMILY_FLUX1, MODEL_FAMILY_FLUX2)

# Latent→RGB preview factors from ComfyUI's comfy/latent_formats.py, as
# (factors[C][3], bias[3] | None). Anima uses the Wan2.1 latent format.
_PREVIEW_RGB = {
    MODEL_FAMILY_SD15: (
        [[0.3512, 0.2297, 0.3227], [0.3250, 0.4974, 0.2350],
         [-0.2829, 0.1762, 0.2721], [-0.2120, -0.2616, -0.7177]],
        None,
    ),
    MODEL_FAMILY_SDXL: (
        [[0.3651, 0.4232, 0.4341], [-0.2533, -0.0042, 0.1068],
         [0.1076, 0.1111, -0.0362], [-0.3165, -0.2492, -0.2188]],
        [0.1084, -0.0175, -0.0011],
    ),
    MODEL_FAMILY_ANIMA: (
        [[-0.1299, -0.1692, 0.2932], [0.0671, 0.0406, 0.0442],
         [0.3568, 0.2548, 0.1747], [0.0372, 0.2344, 0.1420],
         [0.0313, 0.0189, -0.0328], [0.0296, -0.0956, -0.0665],
         [-0.3477, -0.4059, -0.2925], [0.0166, 0.1902, 0.1975],
         [-0.0412, 0.0267, -0.1364], [-0.1293, 0.0740, 0.1636],
         [0.0680, 0.3019, 0.1128], [0.0032, 0.0581, 0.0639],
         [-0.1251, 0.0927, 0.1699], [0.0060, -0.0633, 0.0005],
         [0.3477, 0.2275, 0.2950], [0.1984, 0.0913, 0.1861]],
        [-0.1835, -0.0868, -0.3360],
    ),
}


@dataclass
class LoadedModel:
    name: str
    family: str
    model: object
    native_res: int
    applied_loras: List[str] = field(default_factory=list)
    # Split-file companions (Anima DiT + VAE + TE; FLUX adds CLIP-L), so swapping
    # one triggers a reload.
    vae_name: Optional[str] = None
    te_name: Optional[str] = None
    clip_name: Optional[str] = None
    # Staging settings it was loaded under (offload, vae_tile, …), so an LRU
    # restore can check the placement still matches.
    stage_settings: Optional[tuple] = None


class Engine:
    # X/Y/Z Checkpoint-axis LRU cache of swept models, kept off-GPU. Only fully
    # resident (offload="none") non-FLUX models qualify; any torch error falls
    # back to a disk reload.
    CKPT_CACHE_MAX = 2

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
        self._fp16_accumulation = False
        self._vae_fp16 = False
        self._attention = "sdpa"
        self._last_seed: int = -1
        self._last_upscale_seed: int = -1
        self._weights_epoch: int = 0
        self._anima_defaults_applied: bool = False
        self._ckpt_cache: "OrderedDict[str, LoadedModel]" = OrderedDict()

    # ── state queries ──────────────────────────────────────────────

    @property
    def loaded_name(self) -> Optional[str]:
        return self._loaded.name if self._loaded else None

    @property
    def loaded_family(self) -> Optional[str]:
        return self._loaded.family if self._loaded else None

    @property
    def cuda_graphs_enabled(self) -> bool:
        """Whether the current load baked CUDA Graphs into the backbone."""
        return self._cuda_graphs

    @property
    def active_offload(self) -> "bool | str":
        """The offload mode the current load actually runs with. Unlike
        :meth:`recommended_offload` it reflects the UI's choice and FLUX's
        forced ``"stream"``."""
        return self._offload

    @property
    def applied_loras(self) -> List[str]:
        return list(self._loaded.applied_loras) if self._loaded else []

    @property
    def weights_epoch(self) -> int:
        """Counter bumped whenever the loaded weights change identity (load,
        restore, unload, permanent LoRA fuse), for result caches keyed on it.

        Not bumped by temp LoRAs, which bracket every tagged generation and
        would churn it; callers key the requested LoRA set separately."""
        return self._weights_epoch

    @property
    def last_seed(self) -> int:
        return self._last_seed

    @property
    def last_upscale_seed(self) -> int:
        """Base seed of the last ``upscale()`` tile passes. ``upscale()`` leaves
        ``last_seed`` alone so a post-gen upscale keeps its generation's seed."""
        return self._last_upscale_seed

    @property
    def can_inpaint(self) -> bool:
        """Whether the detailer (per-region ``Inpaint``) can run; every family
        can."""
        return bool(self._loaded)

    @staticmethod
    def fa2_attention_available() -> bool:
        """Whether the locally built FA2-Turing kernel is installed and the GPU
        is sm75."""
        return fa2_turing_available()

    def recommended_offload(self) -> str:
        """Default offload mode from the GPU's VRAM: ``none`` > ``encoders`` >
        ``full`` > ``stream`` as VRAM shrinks. CPU-only gets ``full``."""
        if self.device.type != "cuda":
            return "full"
        vram_gb = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
        if vram_gb >= 23:      # 24 GB class: everything resident
            return "none"
        if vram_gb >= 11:      # 12/16 GB: backbone resident, park encoders + VAE
            return "encoders"
        if vram_gb >= 6:
            return "full"
        return "stream"        # full staging OOMs once 1024² activations land

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

    # ── temporary LoRA (via prompt tags) ─────────────────────────

    @staticmethod
    def parse_lora_prompt(prompt: str) -> tuple[str, list[tuple[str, float]]]:
        loras = []
        def _extract(m):
            name = m.group(1)
            raw = m.group(2)
            try:
                mult = float(raw)
            except ValueError:
                raise ValueError(
                    f"Invalid LoRA weight in <lora:{name}:{raw}>: "
                    f"{raw!r} is not a number (expected e.g. <lora:{name}:0.8>)")
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
            # An unusable name in a prompt tag is reported, not raised.
            try:
                path = lora_path(name)
            except ValueError:
                path = None
            if path is None or not path.exists():
                msgs.append(f"LoRA '{name}' not found")
                continue
            report = apply_lora(self._loaded.model, str(path), multiplier=mult)
            self._loaded.applied_loras.append(name)
            msgs.append(f"{name}@{mult}: {report.applied} matched")
        self._invalidate_cond_cache()  # LoRA patches the TE/adapter
        return " | ".join(msgs) if msgs else "No LoRAs"

    def clear_temp_loras(self) -> None:
        if self._loaded:
            clear_bundle_loras(self._loaded.model)
            self._loaded.applied_loras.clear()
            self._invalidate_cond_cache()

    # ── model loading ──────────────────────────────────────────────

    def _settings_match(
        self, offload: bool | str, vae_tile: bool, compile: bool,
        cuda_graphs: bool, channels_last: bool, tf32: bool,
        fp16_accumulation: bool, attention: str = "sdpa",
        vae_fp16: bool = False,
    ) -> bool:
        """Whether the requested staging settings equal the loaded model's.
        They're baked in at load, so a same-name reload must re-stage otherwise."""
        return (
            self._offload == offload
            and self._vae_tile == vae_tile
            and self._compile == compile
            and self._cuda_graphs == cuda_graphs
            and self._channels_last == channels_last
            and self._tf32 == tf32
            and self._fp16_accumulation == fp16_accumulation
            and self._attention == attention
            and self._vae_fp16 == vae_fp16
        )

    def _components_match(
        self, vae_name: str, te_name: str, clip_name: Optional[str] = None,
    ) -> bool:
        """Whether the loaded split-file model's VAE / TE / CLIP equal the
        requested ones. Assumes ``self._loaded``."""
        lm = self._loaded
        return (
            lm.vae_name == vae_name
            and lm.te_name == te_name
            and lm.clip_name == clip_name
        )

    def _vae_dtype(self, vae_fp16: bool) -> torch.dtype:
        """DevicePolicy ``vae_dtype``. fp16 VAE is CUDA-only; overflow is handled
        by the pipelines' non-finite fp32 fallback."""
        return torch.float16 if (vae_fp16 and self.device.type == "cuda") else torch.float32

    def load_model(
        self, model_name: str, offload: bool | str = True, vae_tile: bool = True,
        compile: bool = False, cuda_graphs: bool = False,
        channels_last: bool = False, tf32: bool = False,
        fp16_accumulation: bool = False, attention: str = "sdpa",
        vae_fp16: bool = False,
    ) -> str:
        if compile and offload is True:
            offload = "encoders"
        elif compile and offload == "stream":
            # stream is the only mode that fits a tiny card, so drop compile
            # (and cuda_graphs, which needs it) instead of the offload mode.
            compile = False
            cuda_graphs = False
            log.warning("compile disabled: incompatible with offload='stream' "
                         "(backbone is block-streamed to fit VRAM)")
        if (self._loaded and self._loaded.name == model_name
                and self._settings_match(offload, vae_tile, compile,
                                         cuda_graphs, channels_last, tf32,
                                         fp16_accumulation, attention, vae_fp16)):
            return f"Model already loaded: {model_name}"

        # Restore from the Checkpoint LRU cache first, then stash the previous
        # model, so an alternating A/B axis keeps both.
        restored = self._try_cache_restore(model_name, offload, vae_tile,
                                           compile, cuda_graphs, channels_last, tf32,
                                           fp16_accumulation, attention, vae_fp16)
        stashed = self._stash_loaded()
        if restored is not None:
            self._offload = offload
            self._vae_tile = vae_tile
            self._compile = compile
            self._cuda_graphs = cuda_graphs
            self._channels_last = channels_last
            self._tf32 = tf32
            self._fp16_accumulation = fp16_accumulation
            self._attention = attention
            self._vae_fp16 = vae_fp16
            self._loaded = restored
            self._attach_cond_cache()
            self._reclaim_memory()
            log.info("[load] %s (%s) restored from ckpt cache", model_name, restored.family)
            return f"Loaded {model_name} ({restored.family}) (from cache)"
        if not stashed:
            self._unload()
        self._offload = offload
        self._vae_tile = vae_tile
        self._compile = compile
        self._cuda_graphs = cuda_graphs
        self._channels_last = channels_last
        self._tf32 = tf32
        self._fp16_accumulation = fp16_accumulation
        self._attention = attention
        self._vae_fp16 = vae_fp16

        path = checkpoint_path(model_name)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        policy = DevicePolicy(
            device=self.device, compute_dtype=self.dtype,
            vae_dtype=self._vae_dtype(vae_fp16),
            offload=offload, vae_tile=vae_tile,
            compile=compile, cuda_graphs=cuda_graphs,
            channels_last=channels_last, tf32=tf32,
            fp16_accumulation=fp16_accumulation, attention=attention,
            # Overlap block-streaming copies with compute.
            stream_prefetch=(offload == "stream"),
        )

        t0 = time.time()
        model = load_checkpoint(str(path), policy=policy)
        elapsed = time.time() - t0

        # Detected architecture, so an all-in-one FLUX checkpoint works here too.
        family = model.spec.architecture
        self._loaded = LoadedModel(
            name=model_name,
            family=family,
            model=model,
            native_res=self._native_res(family),
        )
        self._attach_cond_cache()
        flags = self._perf_flag_summary()
        return f"Loaded {model_name} ({family}) in {elapsed:.1f}s{flags}"

    def reload_model(self, name: str) -> str:
        """Swap the model file for an X/Y/Z Checkpoint sweep, keeping the
        current staging settings and (for Anima) VAE + text encoder. The loaders
        no-op when ``name`` is already current."""
        lm = self._loaded
        if lm and lm.family == MODEL_FAMILY_ANIMA:
            return self.load_anima(
                name, lm.vae_name, lm.te_name,
                offload=self._offload, vae_tile=self._vae_tile,
                compile=self._compile, cuda_graphs=self._cuda_graphs,
                fp16_accumulation=self._fp16_accumulation,
                attention=self._attention,
                vae_fp16=self._vae_fp16,
            )
        return self.load_model(
            name,
            offload=self._offload, vae_tile=self._vae_tile,
            compile=self._compile, cuda_graphs=self._cuda_graphs,
            channels_last=self._channels_last, tf32=self._tf32,
            fp16_accumulation=self._fp16_accumulation,
            attention=self._attention,
            vae_fp16=self._vae_fp16,
        )

    def load_anima(
        self, dit_name: str, vae_name: str, te_name: str,
        offload: bool | str = True, vae_tile: bool = True,
        compile: bool = False, cuda_graphs: bool = False,
        fp16_accumulation: bool = False, attention: str = "sdpa",
        vae_fp16: bool = False,
    ) -> str:
        label = f"Anima({dit_name})"
        if compile and offload is True:
            offload = "encoders"
        elif compile and offload == "stream":
            # stream is the only mode that fits a tiny card, so drop compile
            # (and cuda_graphs, which needs it) instead of the offload mode.
            compile = False
            cuda_graphs = False
            log.warning("compile disabled: incompatible with offload='stream' "
                         "(backbone is block-streamed to fit VRAM)")
        if (self._loaded and self._loaded.name == label
                and self._components_match(vae_name, te_name)
                and self._settings_match(offload, vae_tile, compile,
                                         cuda_graphs, False, False,
                                         fp16_accumulation, attention, vae_fp16)):
            return f"Model already loaded: {label}"

        # A cached entry for this DiT with a different VAE/TE is stale.
        cached = self._ckpt_cache.get(label)
        if cached is not None and (cached.vae_name != vae_name or cached.te_name != te_name):
            self._ckpt_cache.pop(label, None)
            try: del cached.model
            except Exception: pass  # noqa: BLE001
        restored = self._try_cache_restore(label, offload, vae_tile,
                                           compile, cuda_graphs, False, False,
                                           fp16_accumulation, attention, vae_fp16)
        stashed = self._stash_loaded()
        if restored is not None:
            self._offload = offload
            self._vae_tile = vae_tile
            self._compile = compile
            self._cuda_graphs = cuda_graphs
            self._channels_last = False
            self._tf32 = False
            self._fp16_accumulation = fp16_accumulation
            self._attention = attention
            self._vae_fp16 = vae_fp16
            self._loaded = restored
            self._attach_cond_cache()
            self._reclaim_memory()
            log.info("[load] %s restored from ckpt cache", label)
            return f"Loaded Anima (from cache)  (DiT: {dit_name}, VAE: {vae_name}, TE: {te_name})"
        if not stashed:
            self._unload()
        self._offload = offload
        self._vae_tile = vae_tile
        self._compile = compile
        self._cuda_graphs = cuda_graphs
        self._channels_last = False
        self._tf32 = False
        self._fp16_accumulation = fp16_accumulation
        self._attention = attention
        self._vae_fp16 = vae_fp16

        dit_path = diffusion_model_path(dit_name)
        vae_file = vae_path(vae_name)
        te_file = te_path(te_name)
        for p in (dit_path, vae_file, te_file):
            if not p.exists():
                raise FileNotFoundError(f"Anima file not found: {p}")

        policy = DevicePolicy(
            device=self.device, compute_dtype=self.dtype,
            vae_dtype=self._vae_dtype(vae_fp16),
            offload=offload, vae_tile=vae_tile,
            compile=compile, cuda_graphs=cuda_graphs,
            fp16_accumulation=fp16_accumulation, attention=attention,
            # Overlap block-streaming copies with compute.
            stream_prefetch=(offload == "stream"),
        )

        log.info("[load] Anima: DiT=%s VAE=%s TE=%s offload=%s compile=%s",
                 dit_name, vae_name, te_name, offload, compile)
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
            vae_name=vae_name,
            te_name=te_name,
        )
        self._attach_cond_cache()
        flags = self._perf_flag_summary()
        return f"Loaded Anima in {elapsed:.1f}s{flags}  (DiT: {dit_name}, VAE: {vae_name}, TE: {te_name})"

    def load_flux(
        self, dit_name: str, vae_name: str, te_name: str, clip_name: str | None = None,
        offload: bool | str = True, vae_tile: bool = True,
        compile: bool = False, cuda_graphs: bool = False,
        fp16_accumulation: bool = False, attention: str = "sdpa",
        vae_fp16: bool = False,
    ) -> str:
        """Load a split-file FLUX model. ``te_name`` is T5-XXL (FLUX.1) or
        Mistral-3 (FLUX.2); ``clip_name`` is CLIP-L, FLUX.1 only."""
        label = f"FLUX({dit_name})"
        if compile and offload is True:
            offload = "encoders"
        elif compile and offload == "stream":
            # stream is the only mode that fits a tiny card, so drop compile
            # (and cuda_graphs, which needs it) instead of the offload mode.
            compile = False
            cuda_graphs = False
            log.warning("compile disabled: incompatible with offload='stream' "
                         "(backbone is block-streamed to fit VRAM)")
        if (self._loaded and self._loaded.name == label
                and self._components_match(vae_name, te_name, clip_name)
                and self._settings_match(offload, vae_tile, compile,
                                         cuda_graphs, False, False,
                                         fp16_accumulation, attention, vae_fp16)):
            return f"Model already loaded: {label}"

        self._unload()
        self._offload = offload
        self._vae_tile = vae_tile
        self._compile = compile
        self._cuda_graphs = cuda_graphs
        self._channels_last = False
        self._tf32 = False
        self._fp16_accumulation = fp16_accumulation
        self._attention = attention
        self._vae_fp16 = vae_fp16

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
            vae_dtype=self._vae_dtype(vae_fp16),
            offload=offload, vae_tile=vae_tile,
            compile=compile, cuda_graphs=cuda_graphs,
            fp16_accumulation=fp16_accumulation, attention=attention,
            # Overlap block-streaming copies with compute.
            stream_prefetch=(offload == "stream"),
        )

        t0 = time.time()
        # te_file serves as both T5 and Mistral candidate; the detected
        # architecture picks.
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
            vae_name=vae_name, te_name=te_name, clip_name=clip_name,
        )
        self._attach_cond_cache()
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
        if self._fp16_accumulation:
            flags.append("fp16_acc")
        if self._vae_fp16:
            flags.append("fp16_vae")
        if self._attention != "sdpa":
            flags.append("fa2_attn")
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
        if self._fp16_accumulation:
            parts.append("fp16_acc")
        if self._attention != "sdpa":
            parts.append("fa2_attn")
        return ", ".join(parts) if parts else "default"

    def _unload(self) -> None:
        if self._loaded is not None:
            del self._loaded.model
            self._loaded = None
            self._weights_epoch += 1
        self._reclaim_memory()

    # ── conditioning cache ─────────────────────────────────────────
    def _attach_cond_cache(self) -> None:
        """Give the (re)activated model an empty conditioning cache, so entries
        can't go stale across models."""
        if self._loaded is not None:
            self._loaded.model.cond_cache = ConditioningCache()
            self._weights_epoch += 1

    def _invalidate_cond_cache(self) -> None:
        """Drop cached conditioning after a LoRA change (LoRAs can patch the text
        encoders and the Anima LLM-Adapter)."""
        if self._loaded is not None:
            cache = getattr(self._loaded.model, "cond_cache", None)
            if cache is not None:
                cache.clear()

    # ── X/Y/Z Checkpoint LRU cache ─────────────────────────────────
    def _cacheable_for_stash(self) -> bool:
        """Only fully resident, non-FLUX models can be parked: offloaded models
        sit behind a staging proxy that ``.to()`` would break, and FLUX is too
        large for a second copy."""
        if self._loaded is None or self._offload != "none":
            return False
        if self._loaded.family in _FLUX_FAMILIES:
            return False
        return True

    def _current_stage_settings(self) -> tuple:
        """The staging settings the currently-loaded model was loaded under."""
        return (self._offload, self._vae_tile, self._compile, self._cuda_graphs,
                self._channels_last, self._tf32, self._fp16_accumulation,
                self._attention, self._vae_fp16)

    def _stash_loaded(self) -> bool:
        """Park ``self._loaded`` on CPU in the LRU cache, evicting the oldest on
        overflow. Returns whether it stashed; on success the model is detached
        (the caller must not ``_unload``)."""
        if not self._cacheable_for_stash():
            return False
        lm = self._loaded
        try:
            lm.model.to("cpu")
        except Exception as e:  # noqa: BLE001  proxy/wrapper can't be moved wholesale
            log.debug("ckpt cache: can't move %s to CPU (%s); dropping", lm.name, e)
            return False
        # Record now, while the engine's flags still describe this model.
        lm.stage_settings = self._current_stage_settings()
        cache = self._ckpt_cache
        cache.pop(lm.name, None)
        cache[lm.name] = lm
        while len(cache) > self.CKPT_CACHE_MAX:
            _key, evicted = cache.popitem(last=False)
            try: del evicted.model
            except Exception: pass  # noqa: BLE001
            log.debug("ckpt cache: evicted %s (max %d)", _key, self.CKPT_CACHE_MAX)
        self._loaded = None
        self._reclaim_memory()
        return True

    def _try_cache_restore(self, key: str,
                           offload, vae_tile, compile, cuda_graphs,
                           channels_last, tf32, fp16_accumulation,
                           attention="sdpa", vae_fp16=False) -> Optional[LoadedModel]:
        """Pop the cached model for ``key`` and move it back to the device, if
        the settings it was staged under match the request. Otherwise (or on a
        restore error) drop it and return ``None``."""
        lm = self._ckpt_cache.get(key)
        if lm is None:
            return None
        if lm.stage_settings != (offload, vae_tile, compile, cuda_graphs,
                                 channels_last, tf32, fp16_accumulation,
                                 attention, vae_fp16):
            self._ckpt_cache.pop(key, None)
            try: del lm.model
            except Exception: pass  # noqa: BLE001
            return None
        self._ckpt_cache.pop(key, None)
        try:
            lm.model.to(self.device)
        except Exception as e:  # noqa: BLE001
            log.debug("ckpt cache: can't restore %s (%s); reloading", key, e)
            try: del lm.model
            except Exception: pass  # noqa: BLE001
            return None
        return lm

    def _reclaim_memory(self) -> None:
        """Drop dead refs and hand free heap pages back to the OS."""
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
        self._invalidate_cond_cache()
        self._weights_epoch += 1       # fused in permanently
        return f"Applied {lora_name}: {report}"

    def clear_loras(self) -> str:
        if not self._loaded:
            return "No model loaded"
        # Fused LoRAs can only be dropped by discarding the model.
        self._unload()
        return "All LoRAs cleared. Load the model again to generate."

    # ── generation ─────────────────────────────────────────────────

    def _resolve_seed(self, seed: int) -> int:
        if seed == -1:
            seed = int(torch.randint(0, 2**32 - 1, (1,)).item())
        self._last_seed = seed
        return seed

    def _load_oss_sigmas(self, steps: int, width: int, height: int, shift: float):
        """Calibrated OSS sigmas for the current model/config, or None."""
        if not self._loaded:
            return None
        p = oss_cache_path(self._loaded.name, steps, width, height, shift)
        if not p.exists():
            return None
        return _read_cache_json(p)

    def _degrade_oss(self, scheduler: str) -> str:
        """Swap ``oss`` for a plain scheduler outside full-trajectory t2i
        (img2img/inpaint and the refine passes), where no matching calibration
        can exist."""
        if scheduler != "oss":
            return scheduler
        family = self._loaded.family if self._loaded else None
        return "flow" if family == MODEL_FAMILY_ANIMA else "karras"

    def oss_calibrated(self, steps: int, width: int, height: int, shift: float) -> bool:
        """Whether a calibrated OSS schedule already exists for this config."""
        if not self._loaded:
            return False
        return oss_cache_path(self._loaded.name, steps, width, height, shift).exists()

    def calibrate_oss(
        self, *, prompt: str, negative_prompt: str = "",
        steps: int, width: int, height: int, shift: float,
        cfg_scale: float = 4.0, seed: int = 0, grid: int = 80,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> str:
        """Calibrate and cache an OSS schedule for the current Anima model/config."""
        if not self._loaded or self._loaded.family != MODEL_FAMILY_ANIMA:
            raise RuntimeError("Load an Anima model first")
        try:
            sigmas = anima_calibrate_oss(
                self._loaded.model, prompt, negative_prompt,
                steps=steps, width=width, height=height, shift=shift,
                cfg_scale=cfg_scale, grid=grid, seed=seed,
                progress_callback=progress_callback,
            )
        finally:
            self._reclaim_memory()
        values = [round(float(s), 8) for s in sigmas]
        if not _finite_series(values):
            raise RuntimeError(
                "OSS calibration produced a non-finite schedule and was not "
                "cached (try different steps/resolution)")
        p = oss_cache_path(self._loaded.name, steps, width, height, shift)
        _write_cache_json(p, values)
        return f"Calibrated OSS: {steps} steps @ {width}x{height}, shift={shift:g} → {p.name}"

    def apply_vae_tiling(self, always: bool) -> None:
        """Set the loaded model's tiled-VAE preference live (``True`` = always
        tiled, ``False`` = auto). FLUX is always tiled and left alone."""
        # Called from a request thread: snapshot the reference against a
        # concurrent unload.
        lm = self._loaded
        if lm is None or lm.family in _FLUX_FAMILIES:
            return
        lm.model.policy.vae_tile = always
        self._vae_tile = always

    def _load_teacache_coeffs(self) -> "list[float] | None":
        """Per-checkpoint override, else the family fit, else None (identity)."""
        if not self._loaded:
            return None
        for p in (teacache_override_path(self._loaded.name),
                  teacache_cache_path(self._loaded.family)):
            if p.exists():
                coeffs = _read_cache_json(p)
                if coeffs is not None:
                    return coeffs
        return None

    def teacache_status(self) -> dict:
        """TeaCache calibration state of the loaded family for the settings panel."""
        family = self.loaded_family
        return {
            "loaded": bool(family),
            "family": family,
            "calibratable": family == MODEL_FAMILY_ANIMA,
            "coefficients": self._load_teacache_coeffs(),
        }

    def calibrate_teacache(
        self, *, prompt: str, negative_prompt: str = "",
        steps: int = 50, width: int = 1024, height: int = 1024, shift: float = 3.0,
        cfg_scale: float = 4.0, seed: int = 0,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> str:
        """Fit and cache TeaCache coefficients for the loaded Anima family
        (``teacache_cache/<family>.json``, shared by every Anima checkpoint)."""
        if not self._loaded or self._loaded.family != MODEL_FAMILY_ANIMA:
            raise RuntimeError("Load an Anima model first")
        if self._cuda_graphs:
            # The calibration probe is TeaCache's own recording stream.
            raise RuntimeError(_TEACACHE_CUDA_GRAPHS_ERROR)
        try:
            coeffs = anima_calibrate_teacache(
                self._loaded.model, prompt, negative_prompt,
                steps=steps, width=width, height=height, shift=shift,
                cfg_scale=cfg_scale, seed=seed,
                progress_callback=progress_callback,
            )
        finally:
            self._reclaim_memory()
        values = [round(float(c), 10) for c in coeffs]
        # A degenerate fit makes np.polyfit return NaN, which json round-trips
        # silently. Refuse to cache it.
        if not _finite_series(values):
            raise RuntimeError(
                "TeaCache calibration produced non-finite coefficients and was "
                "not cached (try more steps)")
        p = teacache_cache_path(self._loaded.family)
        _write_cache_json(p, values)
        return f"Calibrated TeaCache for {self._loaded.family}: {steps} steps → {p.name}"

    # ── live preview (latent→RGB approximation) ────────────────────

    def _latent_to_preview(self, latent) -> Optional[Image.Image]:
        """Render the sampler's x0 estimate as a rough RGB preview (no VAE decode),
        or None if the family has no factor table."""
        entry = _PREVIEW_RGB.get(self._loaded.family) if self._loaded else None
        if entry is None:
            return None
        factors, bias = entry
        x = latent
        if x.ndim == 5:                 # (B, C, 1, H, W) → (B, C, H, W)
            x = x[:, :, 0]
        x = x[0].float()                # [C, H, W]
        w = torch.tensor(factors, device=x.device, dtype=x.dtype).t()  # [3, C]
        b = torch.tensor(bias, device=x.device, dtype=x.dtype) if bias else None
        img = torch.nn.functional.linear(x.movedim(0, -1), w, b)       # [H, W, 3]
        img = ((img + 1.0) / 2.0).clamp(0, 1).mul(255).to(torch.uint8).cpu().numpy()
        return Image.fromarray(img)

    def _make_preview_cb(self, out_cb, min_interval: float = 0.12):
        """Wrap ``out_cb(PIL.Image)`` as the pipeline's ``preview_callback(latent)``,
        throttled to ``min_interval`` seconds. Preview errors are swallowed."""
        state = {"last": 0.0}

        def cb(latent):
            now = time.perf_counter()
            if now - state["last"] < min_interval:
                return
            try:
                img = self._latent_to_preview(latent)
            except Exception:  # noqa: BLE001  a preview must never fail the job
                img = None
            if img is not None:
                state["last"] = now
                out_cb(img)

        return cb

    # ── Anima resolution snapping ───────────────────────────────────
    # Anima was trained on the SDXL ÷64 grid; ÷16-only sizes (848×1200 → 53×75
    # tokens) misregister in img2img/inpaint. Generate on the ÷64 grid and map
    # the result back.
    def _anima_gen_size(self, width, height) -> Tuple[int | None, int | None, bool]:
        """Generation size to run at, plus whether it was snapped. Snaps up, so
        mapping back is a downscale."""
        if (self._loaded and self._loaded.family == MODEL_FAMILY_ANIMA
                and width is not None and height is not None):
            snap = lambda n: max(512, min(1536, ((n + 63) // 64) * 64))
            gen_w, gen_h = snap(width), snap(height)
            return gen_w, gen_h, (gen_w, gen_h) != (width, height)
        return width, height, False

    @staticmethod
    def _fit_inpaint(generated, init_image, mask_image, width, height) -> Image.Image:
        """Resize a snapped inpaint result to the requested size and re-paste the
        original pixels outside the mask (hard edge)."""
        resized = generated.convert("RGB").resize((width, height), Image.LANCZOS)
        original = init_image.convert("RGB").resize((width, height), Image.LANCZOS)
        mask = (mask_image.convert("L").resize((width, height), Image.NEAREST)
                .point(lambda v: 255 if v >= 128 else 0))
        return Image.composite(resized, original, mask)

    def generate_t2i(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        steps: int = 25,
        cfg_scale: float = 6.0,
        cfg_interval_start: float = 0.0,
        cfg_interval_end: float = 1.0,
        sampler: str = "dpmpp_2m",
        scheduler: str = "karras",
        seed: int = -1,
        shift: float = 1.0,
        curvature: float = 0.25,
        eta_max: float = 1.0,
        gate_reduce: str = "all",
        beta_alpha: float = 0.6,
        beta_beta: float = 0.6,
        lq_threshold: float = 0.025,
        bm_weight: float = 0.5,
        bm_alpha1: float = 0.8, bm_beta1: float = 2.0,
        bm_alpha2: float = 3.0, bm_beta2: float = 0.7,
        teacache_thresh: float = 0.0,
        teacache_use_coeffs: bool = True,
        teacache_forecast: str = "hermite",
        teacache_rule: str = "drift",
        teacache_uncond_scale: float = 1.0,
        deepcache_interval: int = 1,
        progress_callback: Callable[[int, int], None] | None = None,
        preview_callback: Callable[[Image.Image], None] | None = None,
    ) -> Tuple[Image.Image, str]:
        if not self._loaded:
            raise RuntimeError("No model loaded")
        if (teacache_thresh > 0 and self._cuda_graphs
                and self._loaded.family == MODEL_FAMILY_ANIMA):
            raise RuntimeError(_TEACACHE_CUDA_GRAPHS_ERROR)
        seed = self._resolve_seed(seed)
        gen = TextToImage(self._loaded.model)
        kwargs: dict = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            cfg_scale=cfg_scale,
            cfg_interval_start=cfg_interval_start,
            cfg_interval_end=cfg_interval_end,
            width=width,
            height=height,
            sampler=sampler,
            scheduler=scheduler,
            seed=seed,
            teacache_thresh=teacache_thresh,
            teacache_coefficients=(self._load_teacache_coeffs() if teacache_use_coeffs else None),
            teacache_forecast=teacache_forecast,
            teacache_rule=teacache_rule,
            teacache_uncond_scale=teacache_uncond_scale,
            deepcache_interval=deepcache_interval,
            curvature=curvature, eta_max=eta_max, gate_reduce=gate_reduce,
            beta_alpha=beta_alpha,
            beta_beta=beta_beta, lq_threshold=lq_threshold,
            bm_weight=bm_weight, bm_alpha1=bm_alpha1, bm_beta1=bm_beta1,
            bm_alpha2=bm_alpha2, bm_beta2=bm_beta2,
            progress_callback=progress_callback,
            preview_callback=self._make_preview_cb(preview_callback) if preview_callback else None,
            return_info=True,
        )
        if self._loaded.family in (MODEL_FAMILY_ANIMA, *_FLUX_FAMILIES):
            kwargs["shift"] = shift
        if self._loaded.family == MODEL_FAMILY_ANIMA and scheduler == "oss":
            kwargs["oss_sigmas"] = self._load_oss_sigmas(steps, width, height, shift)
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
        cfg_interval_start: float = 0.0,
        cfg_interval_end: float = 1.0,
        sampler: str = "dpmpp_2m",
        scheduler: str = "karras",
        seed: int = -1,
        width: int | None = None,
        height: int | None = None,
        curvature: float = 0.25,
        eta_max: float = 1.0,
        gate_reduce: str = "all",
        beta_alpha: float = 0.6,
        beta_beta: float = 0.6,
        lq_threshold: float = 0.025,
        bm_weight: float = 0.5,
        bm_alpha1: float = 0.8, bm_beta1: float = 2.0,
        bm_alpha2: float = 3.0, bm_beta2: float = 0.7,
        teacache_thresh: float = 0.0,
        teacache_use_coeffs: bool = True,
        teacache_forecast: str = "hermite",
        teacache_rule: str = "drift",
        teacache_uncond_scale: float = 1.0,
        deepcache_interval: int = 1,
        progress_callback: Callable[[int, int], None] | None = None,
        preview_callback: Callable[[Image.Image], None] | None = None,
    ) -> Tuple[Image.Image, str]:
        if not self._loaded:
            raise RuntimeError("No model loaded")
        if (teacache_thresh > 0 and self._cuda_graphs
                and self._loaded.family == MODEL_FAMILY_ANIMA):
            raise RuntimeError(_TEACACHE_CUDA_GRAPHS_ERROR)
        scheduler = self._degrade_oss(scheduler)
        seed = self._resolve_seed(seed)
        gen_w, gen_h, snapped = self._anima_gen_size(width, height)
        gen = ImageToImage(self._loaded.model)
        gen_kwargs = dict(
            prompt=prompt,
            init_image=input_image,
            negative_prompt=negative_prompt,
            strength=strength,
            steps=steps,
            cfg_scale=cfg_scale,
            cfg_interval_start=cfg_interval_start,
            cfg_interval_end=cfg_interval_end,
            sampler=sampler,
            scheduler=scheduler,
            seed=seed,
            width=gen_w,
            height=gen_h,
            teacache_thresh=teacache_thresh,
            teacache_coefficients=(self._load_teacache_coeffs() if teacache_use_coeffs else None),
            teacache_forecast=teacache_forecast,
            teacache_rule=teacache_rule,
            teacache_uncond_scale=teacache_uncond_scale,
            deepcache_interval=deepcache_interval,
            curvature=curvature, eta_max=eta_max, gate_reduce=gate_reduce,
            beta_alpha=beta_alpha,
            beta_beta=beta_beta, lq_threshold=lq_threshold,
            bm_weight=bm_weight, bm_alpha1=bm_alpha1, bm_beta1=bm_beta1,
            bm_alpha2=bm_alpha2, bm_beta2=bm_beta2,
            progress_callback=progress_callback,
            preview_callback=self._make_preview_cb(preview_callback) if preview_callback else None,
            return_info=True,
        )
        try:
            image, pipeline_info = gen(**gen_kwargs)
        finally:
            self._reclaim_memory()
        if snapped:
            image = image.resize((width, height), Image.LANCZOS)
        grid = f" | grid {gen_w}×{gen_h}" if snapped else ""
        info = f"Seed: {seed} | strength={strength} | {steps} steps{grid} | VAE: {pipeline_info.vae_decode_mode}"
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
        cfg_interval_start: float = 0.0,
        cfg_interval_end: float = 1.0,
        sampler: str = "dpmpp_2m",
        scheduler: str = "karras",
        seed: int = -1,
        width: int | None = None,
        height: int | None = None,
        curvature: float = 0.25,
        eta_max: float = 1.0,
        gate_reduce: str = "all",
        beta_alpha: float = 0.6,
        beta_beta: float = 0.6,
        lq_threshold: float = 0.025,
        bm_weight: float = 0.5,
        bm_alpha1: float = 0.8, bm_beta1: float = 2.0,
        bm_alpha2: float = 3.0, bm_beta2: float = 0.7,
        teacache_thresh: float = 0.0,
        teacache_use_coeffs: bool = True,
        teacache_forecast: str = "hermite",
        teacache_rule: str = "drift",
        teacache_uncond_scale: float = 1.0,
        deepcache_interval: int = 1,
        progress_callback: Callable[[int, int], None] | None = None,
        preview_callback: Callable[[Image.Image], None] | None = None,
    ) -> Tuple[Image.Image, str]:
        if not self._loaded:
            raise RuntimeError("No model loaded")
        if (teacache_thresh > 0 and self._cuda_graphs
                and self._loaded.family == MODEL_FAMILY_ANIMA):
            raise RuntimeError(_TEACACHE_CUDA_GRAPHS_ERROR)
        scheduler = self._degrade_oss(scheduler)
        seed = self._resolve_seed(seed)
        gen_w, gen_h, snapped = self._anima_gen_size(width, height)
        gen = Inpaint(self._loaded.model)
        gen_kwargs = dict(
            prompt=prompt,
            init_image=input_image,
            mask_image=mask_image,
            negative_prompt=negative_prompt,
            strength=strength,
            steps=steps,
            cfg_scale=cfg_scale,
            cfg_interval_start=cfg_interval_start,
            cfg_interval_end=cfg_interval_end,
            sampler=sampler,
            scheduler=scheduler,
            seed=seed,
            width=gen_w,
            height=gen_h,
            teacache_thresh=teacache_thresh,
            teacache_coefficients=(self._load_teacache_coeffs() if teacache_use_coeffs else None),
            teacache_forecast=teacache_forecast,
            teacache_rule=teacache_rule,
            teacache_uncond_scale=teacache_uncond_scale,
            deepcache_interval=deepcache_interval,
            curvature=curvature, eta_max=eta_max, gate_reduce=gate_reduce,
            beta_alpha=beta_alpha,
            beta_beta=beta_beta, lq_threshold=lq_threshold,
            bm_weight=bm_weight, bm_alpha1=bm_alpha1, bm_beta1=bm_beta1,
            bm_alpha2=bm_alpha2, bm_beta2=bm_beta2,
            progress_callback=progress_callback,
            preview_callback=self._make_preview_cb(preview_callback) if preview_callback else None,
            return_info=True,
        )
        try:
            image, pipeline_info = gen(**gen_kwargs)
        finally:
            self._reclaim_memory()
        if snapped:
            image = self._fit_inpaint(image, input_image, mask_image, width, height)
        grid = f" | grid {gen_w}×{gen_h}" if snapped else ""
        info = f"Seed: {seed} | inpainted | {steps} steps{grid} | VAE: {pipeline_info.vae_decode_mode}"
        return image, info

    # ── detailer (ADetailer-style region refinement) ───────────────

    def detail(
        self,
        image: Image.Image,
        *,
        detector_path: str,
        prompt: str = "",
        negative_prompt: str = "",
        confidence: float = 0.3,
        strength: float = 0.4,
        steps: int = 25,
        cfg_scale: float = 6.0,
        sampler: str = "dpmpp_2m",
        scheduler: str = "karras",
        gate_reduce: str = "all",
        dilation: int = 4,
        padding: int = 32,
        blur: int = 4,
        max_det: int = 0,
        seed: int = -1,
        teacache_thresh: float = 0.0,
        teacache_use_coeffs: bool = True,
        teacache_forecast: str = "hermite",
        teacache_rule: str = "drift",
        teacache_uncond_scale: float = 1.0,
        progress_callback: Callable[[int, int], None] | None = None,
        preview_callback: Callable[[Image.Image], None] | None = None,
    ) -> Tuple[Image.Image, str]:
        """ADetailer-style: detect regions with a YOLO model, inpaint each at
        native resolution and composite it back. Leaves ``last_seed`` alone."""
        if not self._loaded:
            raise RuntimeError("No model loaded")
        if not self.can_inpaint:
            raise RuntimeError("Detailer needs inpaint, unavailable for this model")
        if (teacache_thresh > 0 and self._cuda_graphs
                and self._loaded.family == MODEL_FAMILY_ANIMA):
            raise RuntimeError(_TEACACHE_CUDA_GRAPHS_ERROR)

        scheduler = self._degrade_oss(scheduler)

        from detailer import (
            bbox_to_mask, detect_regions, dilate_mask,
            expand_crop_region, get_crop_region,
        )

        dets = detect_regions(detector_path, image, confidence)
        if max_det and max_det > 0:
            dets = dets[:max_det]
        n = len(dets)
        if n == 0:
            return image, "Detailer: no detections"

        base_seed = seed if seed is not None and seed >= 0 else \
            int(torch.randint(0, 2**32 - 1, (1,)).item())

        result = image.convert("RGB")
        W, H = result.size
        gen = Inpaint(self._loaded.model)
        # thresh 0 = off (Anima-only at the pipeline level).
        tc_coeffs = self._load_teacache_coeffs() if teacache_use_coeffs else None
        # Previews show the region crop being refined.
        preview_cb = self._make_preview_cb(preview_callback) if preview_callback else None
        try:
            for i, (bbox, _conf) in enumerate(dets):
                mask = dilate_mask(bbox_to_mask(bbox, (W, H)), dilation)
                region = get_crop_region(mask, padding)
                if region is None:
                    continue
                # Square the region so the native-res inpaint doesn't distort it.
                region = expand_crop_region(region, 1, 1, W, H)
                crop = result.crop(region)
                crop_mask = mask.crop(region)

                def sub_cb(step, total, _i=i):
                    if progress_callback:
                        progress_callback(_i * total + step, n * total)

                out, _ = gen(
                    prompt=prompt, init_image=crop, mask_image=crop_mask,
                    negative_prompt=negative_prompt, strength=strength,
                    steps=steps, cfg_scale=cfg_scale, sampler=sampler,
                    scheduler=scheduler, seed=base_seed + i,
                    gate_reduce=gate_reduce,
                    teacache_thresh=teacache_thresh, teacache_coefficients=tc_coeffs,
                    teacache_forecast=teacache_forecast,
                    teacache_rule=teacache_rule,
                    teacache_uncond_scale=teacache_uncond_scale,
                    progress_callback=sub_cb, preview_callback=preview_cb,
                    return_info=True,
                )
                out = out.resize(crop.size, Image.LANCZOS)
                alpha = crop_mask.filter(ImageFilter.GaussianBlur(blur)) if blur else crop_mask
                result.paste(out, region, alpha)
                self._reclaim_memory()
        finally:
            self._reclaim_memory()
        return result, f"Detailer: refined {n} region(s)"

    # ── tiled upscaler (Ultimate SD Upscale style) ────────────────

    def upscale(
        self,
        image: Image.Image,
        *,
        scale: float = 2.0,
        tile: int = 1024,
        overlap: int = 128,
        denoise: float = 0.35,
        base_upscaler: str = "",
        prompt: str = "",
        negative_prompt: str = "",
        steps: int = 25,
        cfg_scale: float = 6.0,
        sampler: str = "dpmpp_2m",
        scheduler: str = "karras",
        gate_reduce: str = "all",
        seed: int = -1,
        teacache_thresh: float = 0.0,
        teacache_use_coeffs: bool = True,
        teacache_forecast: str = "hermite",
        teacache_rule: str = "drift",
        teacache_uncond_scale: float = 1.0,
        progress_callback: Callable[[int, int], None] | None = None,
        preview_callback: Callable[[Image.Image], None] | None = None,
    ) -> Tuple[Image.Image, str]:
        """Upscale (ESRGAN base or Lanczos), then refine overlapping tiles with a
        low-denoise img2img pass and feather-blend them. Leaves ``last_seed``
        alone."""
        if not self._loaded:
            raise RuntimeError("No model loaded")
        if (teacache_thresh > 0 and self._cuda_graphs
                and self._loaded.family == MODEL_FAMILY_ANIMA):
            raise RuntimeError(_TEACACHE_CUDA_GRAPHS_ERROR)

        scheduler = self._degrade_oss(scheduler)

        from upscale import feather_weights, tile_grid, tile_starts

        W, H = image.size
        target_w, target_h = round(W * scale), round(H * scale)
        rgb = image.convert("RGB")
        # An ESRGAN base lets the refine run at low denoise; a soft Lanczos base
        # needs high denoise, which duplicates subjects per tile.
        if base_upscaler:
            base = self._esrgan_upscale(base_upscaler, rgb)
            if base.size != (target_w, target_h):
                base = base.resize((target_w, target_h), Image.LANCZOS)
            base_note = base_upscaler
        else:
            base = rgb.resize((target_w, target_h), Image.LANCZOS)
            base_note = "Lanczos"

        base_seed = seed if seed is not None and seed >= 0 else \
            int(torch.randint(0, 2**32 - 1, (1,)).item())
        self._last_upscale_seed = base_seed

        boxes = tile_grid(target_w, target_h, tile, overlap)
        n = len(boxes)
        # Feather over the actual per-axis overlap (tile - stride), not the
        # requested one: 2x of 1024 packs 3 tiles/axis with 512px overlaps.
        xs = tile_starts(target_w, tile, overlap)
        ys = tile_starts(target_h, tile, overlap)
        ov_x = tile - (xs[1] - xs[0]) if len(xs) > 1 else 0
        ov_y = tile - (ys[1] - ys[0]) if len(ys) > 1 else 0

        acc = np.zeros((target_h, target_w, 3), dtype=np.float64)
        wsum = np.zeros((target_h, target_w, 1), dtype=np.float64)

        gen = ImageToImage(self._loaded.model)
        tc_coeffs = self._load_teacache_coeffs() if teacache_use_coeffs else None
        preview_cb = self._make_preview_cb(preview_callback) if preview_callback else None

        try:
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                crop = base.crop((x1, y1, x2, y2))
                tw, th = x2 - x1, y2 - y1
                # Anima ÷64 snap; only sub-size edge tiles are affected.
                gen_w, gen_h, snapped = self._anima_gen_size(tw, th)

                def sub_cb(step, total, _i=i):
                    if progress_callback:
                        progress_callback(_i * total + step, n * total)

                out, _ = gen(
                    prompt=prompt, init_image=crop,
                    negative_prompt=negative_prompt, strength=denoise,
                    steps=steps, cfg_scale=cfg_scale, sampler=sampler,
                    scheduler=scheduler, seed=base_seed + i,
                    gate_reduce=gate_reduce,
                    width=gen_w, height=gen_h,
                    teacache_thresh=teacache_thresh, teacache_coefficients=tc_coeffs,
                    teacache_forecast=teacache_forecast,
                    teacache_rule=teacache_rule,
                    teacache_uncond_scale=teacache_uncond_scale,
                    progress_callback=sub_cb, preview_callback=preview_cb,
                    return_info=True,
                )
                if snapped:
                    out = out.resize((tw, th), Image.LANCZOS)
                out_arr = np.array(out.convert("RGB"), dtype=np.float64) / 255.0
                w = feather_weights(tw, th, ov_x, ov_y)[..., None]
                acc[y1:y2, x1:x2] += out_arr * w
                wsum[y1:y2, x1:x2] += w
                self._reclaim_memory()
        finally:
            self._reclaim_memory()

        result = np.clip(acc / np.clip(wsum, 1e-6, None), 0, 1)
        result = (result * 255).astype(np.uint8)
        result_img = Image.fromarray(result)
        info = (
            f"Upscale: {W}×{H} → {target_w}×{target_h}, "
            f"{n} tiles @ denoise {denoise} (base {base_note})"
        )
        return result_img, info

    def _esrgan_upscale(
        self, model_name: str, image: Image.Image,
        in_tile: int = 512, in_overlap: int = 32,
    ) -> Image.Image:
        """Run an ESRGAN-family model (spandrel) over the image in feather-blended
        tiles, so it fits next to a resident diffusion model. Returns the
        model-scale upscale; the caller resizes to the target."""
        if _spandrel is None:
            raise RuntimeError(
                "spandrel is not installed. Run `pip install spandrel` to use an "
                "ESRGAN base, or pick Lanczos."
            )
        from upscale import feather_weights, tile_grid, tile_starts
        from utils import upscaler_path

        path = upscaler_path(model_name)
        if not path.is_file():
            raise RuntimeError(f"Upscaler model not found: {model_name}")

        desc = _spandrel.ModelLoader().load_from_file(str(path))
        desc.to(self.device).eval()
        sf = int(desc.scale)
        mdtype = next(desc.model.parameters()).dtype

        W, H = image.size
        out_w, out_h = W * sf, H * sf
        acc = np.zeros((out_h, out_w, 3), dtype=np.float64)
        wsum = np.zeros((out_h, out_w, 1), dtype=np.float64)

        xs = tile_starts(W, in_tile, in_overlap)
        ys = tile_starts(H, in_tile, in_overlap)
        ov_x = (in_tile - (xs[1] - xs[0])) * sf if len(xs) > 1 else 0
        ov_y = (in_tile - (ys[1] - ys[0])) * sf if len(ys) > 1 else 0

        try:
            with torch.inference_mode():
                for (x1, y1, x2, y2) in tile_grid(W, H, in_tile, in_overlap):
                    crop = image.crop((x1, y1, x2, y2))
                    t = torch.from_numpy(np.asarray(crop, dtype=np.float32) / 255.0)
                    t = t.permute(2, 0, 1).unsqueeze(0).to(self.device, mdtype)
                    out = desc(t).clamp(0, 1).squeeze(0).permute(1, 2, 0)
                    out = out.float().cpu().numpy()
                    oh, ow = out.shape[:2]
                    ox, oy = x1 * sf, y1 * sf
                    w = feather_weights(ow, oh, ov_x, ov_y)[..., None]
                    acc[oy:oy + oh, ox:ox + ow] += out * w
                    wsum[oy:oy + oh, ox:ox + ow] += w
                    self._reclaim_memory()
        finally:
            del desc
            self._reclaim_memory()

        result = np.clip(acc / np.clip(wsum, 1e-6, None), 0, 1)
        return Image.fromarray((result * 255).astype(np.uint8))


# ── singleton ──────────────────────────────────────────────────────
ENGINE = Engine()
