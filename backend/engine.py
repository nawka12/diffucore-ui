"""Model manager — keeps the loaded model in memory across generations."""

from __future__ import annotations

import ctypes
import ctypes.util
import gc
import json
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

_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_DIFFUCORE_SRC = _ROOT / "diffucore" / "src"
if _LOCAL_DIFFUCORE_SRC.exists():
    sys.path.insert(0, str(_LOCAL_DIFFUCORE_SRC))

import torch
from PIL import Image, ImageFilter

from diffucore import (
    anima_calibrate_oss,
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

# The flow families (Anima, FLUX) only drive the flow-friendly subset; their
# pipelines reject "heun" and "euler_ancestral" (see _ANIMA_SAMPLERS /
# _FLUX_SAMPLERS), so the UI must not offer those when a flow model is selected.
SAMPLERS_SD = [
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
SAMPLERS_FLOW = [s for s in SAMPLERS_SD if s not in ("heun", "euler_ancestral")]
SAMPLERS_ANIMA = SAMPLERS_FLOW
SAMPLERS_FLUX = SAMPLERS_FLOW

SCHEDULERS_SD = ["karras", "exponential", "polyexponential", "sgm_uniform", "simple"]
# "oss" is a calibrated optimal-stepsize schedule: it needs a one-time
# calibration for the exact (model, steps, resolution, shift) before it works.
# The UI's OSS panel runs that calibration (Engine.calibrate_oss) and writes the
# cache; selecting "oss" before calibrating errors with a clear message.
SCHEDULERS_ANIMA = ["flow", "flow_dyn", "oss", "sgm_uniform", "simple"]
SCHEDULERS_FLUX = ["flux", "flow", "sgm_uniform", "simple"]

# Calibrated OSS schedules are cached one JSON (list of descending sigmas) per
# (model, steps, resolution, shift); calibrate_oss.py writes them here.
_OSS_CACHE_DIR = _ROOT / "models" / "oss_cache"


def oss_cache_path(name: str, steps: int, width: int, height: int, shift: float) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    return _OSS_CACHE_DIR / f"{safe}__{steps}s_{width}x{height}_shift{shift:g}.json"

# (family, native_res) tuples — used for defaults
MODEL_FAMILY_SD15 = "sd15"
MODEL_FAMILY_SDXL = "sdxl"
MODEL_FAMILY_ANIMA = "anima"
MODEL_FAMILY_FLUX1 = "flux1"
MODEL_FAMILY_FLUX2 = "flux2"
_FLUX_FAMILIES = (MODEL_FAMILY_FLUX1, MODEL_FAMILY_FLUX2)

# Cheap latent→RGB approximation for live previews (factors from ComfyUI's
# comfy/latent_formats.py). Per family: (factors[C][3], bias[3] | None). Applied
# to the sampler's x0 estimate to render a rough preview without a VAE decode.
# Anima uses the Wan2.1 latent format (its VAE carries Wan2.1 latent stats).
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
    def can_inpaint(self) -> bool:
        """Whether the loaded family supports inpainting (the detailer's backbone).
        FLUX is text-to-image only in this build."""
        return bool(self._loaded) and self._loaded.family not in _FLUX_FAMILIES

    def recommended_offload(self) -> str:
        """A sensible default offload mode for the UI, picked from the GPU's VRAM.
        More VRAM → keep more resident (faster): ``none`` > ``encoders`` > ``full``.
        FLUX overrides this to ``stream`` in the UI regardless (it can't stage its
        ~23 GB DiT as one blob). CPU-only falls back to ``full``."""
        if self.device.type != "cuda":
            return "full"
        vram_gb = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
        if vram_gb >= 23:      # 24 GB-class (3090/4090): hold everything resident
            return "none"
        if vram_gb >= 11:      # 12/16 GB-class: keep the (small) backbone resident,
            return "encoders"  # only park encoders + VAE. SD/SDXL UNet (~5 GB) and the
                               # Anima DiT (~4 GB) fit alongside activations; the heavy
                               # VAE decode auto-tiles. Avoids shuffling the backbone
                               # on/off the GPU every image.
        return "full"          # ≤10 GB: shuttle everything (safest against OOM)

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

    def _settings_match(
        self, offload: bool | str, vae_tile: bool, compile: bool,
        cuda_graphs: bool, channels_last: bool, tf32: bool,
    ) -> bool:
        """Whether the requested load-time staging settings equal those of the
        currently loaded model. Offload and the perf flags are baked in at load
        via DevicePolicy, so a same-name reload only skips re-staging when these
        also match — otherwise the change would be silently dropped."""
        return (
            self._offload == offload
            and self._vae_tile == vae_tile
            and self._compile == compile
            and self._cuda_graphs == cuda_graphs
            and self._channels_last == channels_last
            and self._tf32 == tf32
        )

    def load_model(
        self, model_name: str, offload: bool | str = True, vae_tile: bool = True,
        compile: bool = False, cuda_graphs: bool = False,
        channels_last: bool = False, tf32: bool = False,
    ) -> str:
        if compile and offload is True:
            offload = "encoders"
        if (self._loaded and self._loaded.name == model_name
                and self._settings_match(offload, vae_tile, compile,
                                         cuda_graphs, channels_last, tf32)):
            return f"Model already loaded: {model_name}"

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
        if compile and offload is True:
            offload = "encoders"
        if (self._loaded and self._loaded.name == label
                and self._settings_match(offload, vae_tile, compile,
                                         cuda_graphs, False, False)):
            return f"Model already loaded: {label}"

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
        if compile and offload is True:
            offload = "encoders"
        if (self._loaded and self._loaded.name == label
                and self._settings_match(offload, vae_tile, compile,
                                         cuda_graphs, False, False)):
            return f"Model already loaded: {label}"

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

    def _load_oss_sigmas(self, steps: int, width: int, height: int, shift: float):
        """Calibrated OSS sigma list for the current model/config, or None if
        none has been calibrated yet."""
        if not self._loaded:
            return None
        p = oss_cache_path(self._loaded.name, steps, width, height, shift)
        if not p.exists():
            return None
        with open(p) as f:
            return json.load(f)

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
        p = oss_cache_path(self._loaded.name, steps, width, height, shift)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps([round(s, 8) for s in sigmas]))
        return f"Calibrated OSS: {steps} steps @ {width}x{height}, shift={shift:g} → {p.name}"

    # ── live preview (cheap latent→RGB approximation) ──────────────

    def _latent_to_preview(self, latent) -> Optional[Image.Image]:
        """Render the sampler's x0 estimate to a small RGB preview, or None if the
        loaded family has no factor table. Rough by design — no VAE decode."""
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
        """Wrap an ``out_cb(PIL.Image)`` consumer into the pipeline's
        ``preview_callback(latent)``: throttle to ``min_interval`` seconds and
        swallow any decode error so a preview never breaks a generation."""
        state = {"last": 0.0}

        def cb(latent):
            now = time.perf_counter()
            if now - state["last"] < min_interval:
                return
            try:
                img = self._latent_to_preview(latent)
            except Exception:  # noqa: BLE001 — a preview must never fail the job
                img = None
            if img is not None:
                state["last"] = now
                out_cb(img)

        return cb

    # ── Anima resolution snapping ───────────────────────────────────
    # Anima was trained on the SDXL ÷64 resolution grid; sizes that are ÷16
    # (the architectural minimum) but not ÷64 land on an odd latent-token grid
    # (e.g. 848×1200 → 53×75) that's out of distribution, and img2img/inpaint
    # expose it as misregistered content. Generate on the nearest in-range ÷64
    # grid, then map the result back to the requested size. No-op for any size
    # that's already ÷64 (all SDXL buckets, 1024×1536, …) and for non-Anima.
    def _anima_gen_size(self, width, height) -> Tuple[int | None, int | None, bool]:
        """Generation size to actually run at, plus whether it was snapped.

        Snaps **up** to the next ÷64 grid so the map-back to the requested size
        is a downscale (sharper) rather than an upscale.
        """
        if (self._loaded and self._loaded.family == MODEL_FAMILY_ANIMA
                and width is not None and height is not None):
            snap = lambda n: max(512, min(1536, ((n + 63) // 64) * 64))
            gen_w, gen_h = snap(width), snap(height)
            return gen_w, gen_h, (gen_w, gen_h) != (width, height)
        return width, height, False

    @staticmethod
    def _fit_inpaint(generated, init_image, mask_image, width, height) -> Image.Image:
        """Resize an inpaint result generated on the snapped grid back to the
        requested size, then re-paste the original pixels into the keep region
        (hard mask edge) so untouched areas stay exact — same as the pipeline's
        own composite, just at the requested resolution."""
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
        sampler: str = "dpmpp_2m",
        scheduler: str = "karras",
        seed: int = -1,
        shift: float = 1.0,
        progress_callback: Callable[[int, int], None] | None = None,
        preview_callback: Callable[[Image.Image], None] | None = None,
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
        sampler: str = "dpmpp_2m",
        scheduler: str = "karras",
        seed: int = -1,
        width: int | None = None,
        height: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        preview_callback: Callable[[Image.Image], None] | None = None,
    ) -> Tuple[Image.Image, str]:
        if not self._loaded:
            raise RuntimeError("No model loaded")
        if self._loaded.family in _FLUX_FAMILIES:
            raise RuntimeError("FLUX supports text-to-image only in this build")
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
            sampler=sampler,
            scheduler=scheduler,
            seed=seed,
            width=gen_w,
            height=gen_h,
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
        sampler: str = "dpmpp_2m",
        scheduler: str = "karras",
        seed: int = -1,
        width: int | None = None,
        height: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        preview_callback: Callable[[Image.Image], None] | None = None,
    ) -> Tuple[Image.Image, str]:
        if not self._loaded:
            raise RuntimeError("No model loaded")
        if self._loaded.family in _FLUX_FAMILIES:
            raise RuntimeError("FLUX supports text-to-image only in this build")
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
            sampler=sampler,
            scheduler=scheduler,
            seed=seed,
            width=gen_w,
            height=gen_h,
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
        dilation: int = 4,
        padding: int = 32,
        blur: int = 4,
        max_det: int = 0,
        seed: int = -1,
        progress_callback: Callable[[int, int], None] | None = None,
        preview_callback: Callable[[Image.Image], None] | None = None,
    ) -> Tuple[Image.Image, str]:
        """Detect regions with a YOLO model, then inpaint each one at the model's
        native resolution and composite it back — the same idea as ADetailer, but
        driven through diffucore's ``Inpaint`` so it works for UNet and DiT alike.

        Does not touch ``last_seed`` so the caller's generation seed is preserved
        for output naming/metadata."""
        if not self._loaded:
            raise RuntimeError("No model loaded")
        if not self.can_inpaint:
            raise RuntimeError("Detailer needs inpaint, unavailable for this model")

        # OSS is a full-trajectory t2i schedule (calibrated, not usable mid-denoise);
        # fall back to a plain scheduler for the masked inpaint passes.
        if scheduler == "oss":
            scheduler = "flow" if self._loaded.family == MODEL_FAMILY_ANIMA else "karras"

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
        # Show each region's crop being refined in the live preview (shared
        # throttle across regions). The crop denoises at native res, so the
        # preview shows just the region, not the full image.
        preview_cb = self._make_preview_cb(preview_callback) if preview_callback else None
        try:
            for i, (bbox, _conf) in enumerate(dets):
                mask = dilate_mask(bbox_to_mask(bbox, (W, H)), dilation)
                region = get_crop_region(mask, padding)
                if region is None:
                    continue
                # square the region so the native-res inpaint doesn't distort it
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


# ── singleton ──────────────────────────────────────────────────────
ENGINE = Engine()
