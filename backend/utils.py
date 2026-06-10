"""Directory scanning helpers for models, LoRAs, and outputs."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import List, Set

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
DIFFUSION_DIR = MODELS_DIR / "diffusion-models"
VAE_DIR = MODELS_DIR / "vae"
TE_DIR = MODELS_DIR / "text-encoders"
LORAS_DIR = MODELS_DIR / "loras"
DETAILERS_DIR = MODELS_DIR / "detailers"
OUTPUTS_DIR = ROOT / "outputs"

_CHECKPOINT_EXTS = {".safetensors", ".ckpt", ".pt", ".pth"}
_LORA_EXTS = {".safetensors"}
_DETECTOR_EXTS = {".pt", ".pth"}

_ALL_DIRS = (CHECKPOINTS_DIR, DIFFUSION_DIR, VAE_DIR, TE_DIR, LORAS_DIR,
             DETAILERS_DIR, OUTPUTS_DIR)


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


def checkpoint_path(name: str) -> Path:
    return CHECKPOINTS_DIR / name


def diffusion_model_path(name: str) -> Path:
    return DIFFUSION_DIR / name


def vae_path(name: str) -> Path:
    return VAE_DIR / name


def te_path(name: str) -> Path:
    return TE_DIR / name


def lora_path(name: str) -> Path:
    return LORAS_DIR / name


def detector_path(name: str) -> Path:
    return DETAILERS_DIR / name


def _parse_date_dir(name: str) -> date:
    try:
        return date(int(name[6:10]), int(name[3:5]), int(name[0:2]))
    except (ValueError, IndexError):
        return date.min


def _output_sort_key(f: Path) -> tuple:
    """Newest-first ordering within a day folder. The leading index in
    ``{i:02d}-{seed}.png`` climbs with save time, so sort on it numerically — a
    string sort misorders once a day passes 99 images (``"100"`` < ``"99"``).
    Names without a parseable index fall back to mtime, ranked below indexed files.
    """
    try:
        return (1, int(f.stem.split("-")[0]), 0.0)
    except (ValueError, IndexError):
        return (0, 0, f.stat().st_mtime)


def scan_outputs() -> List[Path]:
    _ensure_dirs()
    files = []
    dirs = [d for d in OUTPUTS_DIR.iterdir() if d.is_dir()]
    dirs.sort(key=lambda d: _parse_date_dir(d.name), reverse=True)
    for d in dirs:
        pngs = [f for f in d.iterdir() if f.suffix.lower() == ".png"]
        pngs.sort(key=_output_sort_key, reverse=True)
        files.extend(pngs)
    return files


def next_output_path(seed: int, ext: str = "png") -> Path:
    _ensure_dirs()
    date_str = date.today().strftime("%d-%m-%Y")
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
    name = f"{i:02d}-{seed}.{ext}"
    return dir_path / name
