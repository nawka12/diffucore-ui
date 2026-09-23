"""Directory scanning helpers for models, LoRAs, and outputs."""

from __future__ import annotations

import threading
from datetime import date
from pathlib import Path
from typing import List, Optional, Set

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
DIFFUSION_DIR = MODELS_DIR / "diffusion-models"
VAE_DIR = MODELS_DIR / "vae"
TE_DIR = MODELS_DIR / "text-encoders"
LORAS_DIR = MODELS_DIR / "loras"
DETAILERS_DIR = MODELS_DIR / "detailers"
UPSCALERS_DIR = MODELS_DIR / "upscalers"
OUTPUTS_DIR = ROOT / "outputs"

_CHECKPOINT_EXTS = {".safetensors", ".ckpt", ".pt", ".pth"}
_LORA_EXTS = {".safetensors"}
_DETECTOR_EXTS = {".pt", ".pth"}
_UPSCALER_EXTS = {".pth", ".safetensors", ".pt"}

_ALL_DIRS = (CHECKPOINTS_DIR, DIFFUSION_DIR, VAE_DIR, TE_DIR, LORAS_DIR,
             DETAILERS_DIR, UPSCALERS_DIR, OUTPUTS_DIR)


def _ensure_dirs() -> None:
    for d in _ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def _scan(directory: Path, exts: Set[str]) -> List[str]:
    _ensure_dirs()
    return [
        p.name for p in sorted(directory.iterdir())
        if p.is_file() and p.suffix.lower() in exts
    ]


def scan_checkpoints() -> List[str]:
    return _scan(CHECKPOINTS_DIR, _CHECKPOINT_EXTS)


def scan_diffusion_models() -> List[str]:
    return _scan(DIFFUSION_DIR, _CHECKPOINT_EXTS)


def scan_vae() -> List[str]:
    return _scan(VAE_DIR, _CHECKPOINT_EXTS)


def scan_text_encoders() -> List[str]:
    return _scan(TE_DIR, _CHECKPOINT_EXTS)


def scan_loras() -> List[str]:
    return _scan(LORAS_DIR, _LORA_EXTS)


def scan_detectors() -> List[str]:
    return _scan(DETAILERS_DIR, _DETECTOR_EXTS)


def scan_upscalers() -> List[str]:
    return _scan(UPSCALERS_DIR, _UPSCALER_EXTS)


def model_name_ok(name: str) -> bool:
    """Whether ``name`` is a plain filename that stays inside its model dir.
    Names come from API payloads, and a loader like ``torch.load`` on an
    attacker-chosen ``.pt`` is code execution."""
    return bool(name) and name == Path(name).name and name not in (".", "..")


def _model_path(directory: Path, name: str) -> Path:
    if not model_name_ok(name):
        raise ValueError(f"invalid model name: {name!r}")
    return directory / name


def checkpoint_path(name: str) -> Path:
    return _model_path(CHECKPOINTS_DIR, name)


def diffusion_model_path(name: str) -> Path:
    return _model_path(DIFFUSION_DIR, name)


def vae_path(name: str) -> Path:
    return _model_path(VAE_DIR, name)


def te_path(name: str) -> Path:
    return _model_path(TE_DIR, name)


def lora_path(name: str) -> Path:
    """Resolve a LoRA name, with or without its extension (the tagcomplete
    extension inserts bare names), to its path."""
    p = _model_path(LORAS_DIR, name)
    if p.exists():
        return p
    for ext in _LORA_EXTS:
        candidate = LORAS_DIR / (name + ext)
        if candidate.exists():
            return candidate
    return p  # the original path, so the caller's "not found" error is clear


def detector_path(name: str) -> Path:
    return _model_path(DETAILERS_DIR, name)


def upscaler_path(name: str) -> Path:
    return _model_path(UPSCALERS_DIR, name)


def _parse_date_dir(name: str) -> date:
    # Non-ISO folder names sort last rather than erroring.
    try:
        return date.fromisoformat(name)
    except ValueError:
        return date.min


def _output_sort_key(f: Path) -> tuple:
    """Newest-first within a day folder, by the numeric index of
    ``{i:05d}-{seed}.png`` (a string sort misorders legacy 2-digit names).
    Unindexed names fall back to mtime, below indexed ones.
    """
    try:
        return (1, int(f.stem.split("-")[0]), 0.0)
    except (ValueError, IndexError):
        return (0, 0, f.stat().st_mtime)


def scan_outputs() -> List[Path]:
    """List output PNGs newest-first. Cached; rebuilt when the newest date
    folder's mtime advances or the outputs dir moves, and invalidated by the
    server on save/delete (ext4 mtimes have 1 s resolution).
    """
    global _OUTPUTS_CACHE, _OUTPUTS_CACHE_KEY
    newest = _outputs_newest_mtime()
    key = (str(OUTPUTS_DIR), newest)
    with _OUTPUTS_CACHE_LOCK:
        if _OUTPUTS_CACHE is not None and key == _OUTPUTS_CACHE_KEY:
            return list(_OUTPUTS_CACHE)
        _ensure_dirs()
        cache = _scan_outputs_uncached()
        _OUTPUTS_CACHE = cache
        _OUTPUTS_CACHE_KEY = key
        return list(cache)


_OUTPUTS_CACHE: Optional[List[Path]] = None
_OUTPUTS_CACHE_KEY: tuple = ()  # (str(OUTPUTS_DIR), newest date-folder mtime)
_OUTPUTS_CACHE_LOCK = threading.Lock()


def invalidate_outputs_cache() -> None:
    """Clear the outputs listing cache (after a save or delete)."""
    global _OUTPUTS_CACHE, _OUTPUTS_CACHE_KEY
    with _OUTPUTS_CACHE_LOCK:
        _OUTPUTS_CACHE = None
        _OUTPUTS_CACHE_KEY = ()


def _outputs_newest_mtime() -> float:
    """Newest mtime among the date folders in OUTPUTS_DIR (0.0 if none)."""
    try:
        return max(
            (d.stat().st_mtime for d in OUTPUTS_DIR.iterdir() if d.is_dir()),
            default=0.0,
        )
    except OSError:
        return 0.0


def _scan_outputs_uncached() -> List[Path]:
    _ensure_dirs()
    # Skip dot-dirs, notably the gallery's .trash/.
    dirs = [d for d in OUTPUTS_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")]
    dirs.sort(key=lambda d: _parse_date_dir(d.name), reverse=True)
    files: List[Path] = []
    for d in dirs:
        pngs = [f for f in d.iterdir() if f.suffix.lower() == ".png"]
        pngs.sort(key=_output_sort_key, reverse=True)
        files.extend(pngs)
    return files


def next_output_path(seed: int, ext: str = "png") -> Path:
    _ensure_dirs()
    date_str = date.today().isoformat()
    dir_path = OUTPUTS_DIR / date_str
    dir_path.mkdir(parents=True, exist_ok=True)
    max_i = 0
    for f in dir_path.iterdir():
        if f.suffix.lower() == f".{ext}":
            try:
                num = int(f.stem.split("-")[0])
                max_i = max(max_i, num)
            except (ValueError, IndexError):
                pass
    i = max_i + 1
    # 5-digit padding (A1111-style) keeps string sort correct past 99/day.
    name = f"{i:05d}-{seed}.{ext}"
    return dir_path / name
