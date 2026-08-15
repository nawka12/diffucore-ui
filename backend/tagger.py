"""AI NSFW rating for gallery images — SmilingWolf/wd-eva02-large-tagger-v3.

A 315M-param ViT (EVA02-L, 448px) that classifies each output's content rating
as general / sensitive / questionable / explicit. Ratings feed the gallery
blur (questionable and up = this app's "R and up"), replacing the prompt
heuristic when the model is available — a machine that can run a diffusion
model has plenty of headroom for this one.

The model is loaded lazily on the first rating and held across tag jobs; the
server calls :func:`TAGGER.unload` before any GPU generation so the ~600 MB
(fp16) never competes with a checkpoint. ``timm`` is an optional dependency
(guarded import): without it the server silently falls back to the prompt
rating and everything else keeps working.
"""

from __future__ import annotations

import csv
import threading
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

REPO_ID = "SmilingWolf/wd-eva02-large-tagger-v3"
IMG_SIZE = 448
# Inference batch size for scan jobs — keeps the VRAM spike modest.
BATCH_SIZE = 4

# WD rating tag -> this app's rating tier (mirrors Civitai's names).
RATING_TO_TIER = {
    "general": "PG",
    "sensitive": "PG13",
    "questionable": "R",
    "explicit": "X",
}
NSFW_TIERS = ("R", "X")

# ── decision layer ────────────────────────────────────────────────────
# The 4 rating heads are *independent sigmoids* (multi-label BCE), not a
# softmax — so argmax is a brittle mutual-exclusion assumption that breaks on
# near-ties (an image can be sensitive=0.98 AND explicit=0.92). Ratings are
# therefore re-derived with a calibrated cascade that cross-checks the model's
# own content tags against its rating head:
#   * any hard explicit tag (anatomy/sex) ≥ HARD_TAG_THRESH  →  forced X
#   * explicit-headed but zero suggestive/nudity tags  →  de-escalated (the
#     stylized-VTuber false positive: bare shoulders, choker, no anatomy tags)
#   * otherwise the rating sigmoids are logit-normalised with a temperature
#     and cascaded down to R / PG13 / PG.
# DECISION_VERSION stamps every cached verdict; a bump invalidates the whole
# cache so a logic change forces a re-rate instead of serving stale verdicts.
DECISION_VERSION = 2
HARD_TAG_THRESH = 0.35   # ≥ this probability on any hard tag → explicit
SOFT_TAG_THRESH = 0.25   # corroborating evidence for R/PG13 escalation
SOFTMAX_TEMP = 0.7       # temperature on the logit-normalised rating head

# Verified present in selected_tags.csv (checked against the v3 tag list).
HARD_EXPLICIT_TAGS = frozenset((
    "nipples", "pussy", "penis", "fellatio", "cunnilingus", "sex", "vaginal",
    "anal", "uncensored", "clitoris", "erection", "ejaculation",
    "cross-section", "nude", "completely_nude", "spread_legs", "dildo",
    "vibrator", "masturbation", "paizuri", "facial", "cum", "tentacles",
    "incest", "gangbang", "handjob", "orgasm", "oral",
))
# Strong suggestive tags — enough to corroborate an explicit rating head.
# Genuinely sexualized context, not just an outfit.
STRONG_SUGGESTIVE_TAGS = frozenset((
    "lingerie", "panties", "underwear", "underboob", "sideboob", "thong",
    "bondage", "upskirt", "skirt_lift", "undressing", "cameltoe",
    "covered_nipples",
))
# Weak suggestive tags — fire on tons of SFW anime (busty idol costumes,
# swimsuits) and must never alone force an X; they only corroborate R/PG13.
SUGGESTIVE_TAGS = frozenset((
    "cleavage", "bra", "swimsuit", "bikini", "micro_bikini", "midriff",
    "short_shorts", "fishnets", "collar", "leash", "ass",
))


def _logit(p: np.ndarray) -> np.ndarray:
    eps = 1e-7
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def decide_rating(rating_scores: np.ndarray, tag_scores: np.ndarray,
                  hard_idx, strong_idx, soft_idx):
    """Turn the raw WD output into a final tier.

    ``rating_scores`` is the 4 sigmoid probabilities (general, sensitive,
    questionable, explicit); ``tag_scores`` the full 10861-dim sigmoid vector;
    ``hard_idx``/``strong_idx``/``soft_idx`` are the tag indices that
    corroborate explicit / strongly-suggestive / weakly-suggestive content.
    Returns ``(tier, conf, reason)`` where ``conf`` is the peak raw rating
    probability (informational — gating happens in the cascade, not on a bare
    threshold) and ``reason`` says which rule fired.
    """
    cal = _logit(rating_scores) / SOFTMAX_TEMP
    cal = np.exp(cal); cal = cal / cal.sum()
    r_gen, r_sens, r_ques, r_expl = (float(v) for v in rating_scores)
    s_sens, s_ques, s_expl = (float(v) for v in cal[1:])
    conf = max(r_gen, r_sens, r_ques, r_expl)
    n_hard = sum(1 for i in hard_idx if tag_scores[i] >= HARD_TAG_THRESH)
    n_strong = sum(1 for i in strong_idx if tag_scores[i] >= SOFT_TAG_THRESH)
    n_soft = sum(1 for i in soft_idx if tag_scores[i] >= SOFT_TAG_THRESH)
    # Hard anatomy/sex tags override a contradictory rating head.
    if n_hard:
        return "X", conf, f"hard_tag({n_hard})"
    # Explicit-headed. Only hard or *strong* suggestive evidence keeps it X —
    # cleavage/bikini fire on plenty of SFW anime and must not force a blur.
    if s_expl >= 0.40 or r_expl >= 0.70:
        if n_strong == 0:
            tier = "PG13" if (n_soft or s_sens > 0.30) else "PG"
            return tier, conf, "de_escalated"
        return "X", conf, "explicit"
    if s_ques >= 0.40 or r_ques >= 0.60 or (s_expl + s_ques >= 0.50 and (n_strong or n_soft)):
        return "R", conf, "questionable"
    if s_sens >= 0.35 or r_sens >= 0.50 or n_strong or n_soft:
        return "PG13", conf, "sensitive"
    return "PG", conf, "general"


def timm_available() -> bool:
    """Whether the optional ``timm`` dependency is installed."""
    try:
        import timm  # noqa: F401
        return True
    except ImportError:
        return False


def _preprocess(img: Image.Image) -> torch.Tensor:
    """Resize/crop to the model's input and normalise to [-1, 1] (mean 0.5)."""
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(x).permute(2, 0, 1)
    return (x - 0.5) / 0.5


class Tagger:
    """Lazily-loaded process-global tagger. Not thread-safe for concurrent
    inference — the server runs it on the single job worker thread, and the
    unload path takes the same lock that load/infer hold."""

    def __init__(self):
        self._lock = threading.Lock()
        self._model = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._rows: Optional[List[list]] = None   # selected_tags.csv rows (incl. header)
        self._rating_idx: Optional[List[int]] = None
        self._hard_idx: List[int] = []
        self._strong_idx: List[int] = []
        self._soft_idx: List[int] = []

    # ── model lifecycle ─────────────────────────────────────────────
    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self):
        """Load (and download, on first use) the model. Idempotent."""
        with self._lock:
            if self._model is not None:
                return self._model
            import timm  # guarded import — the caller checks timm_available()
            model = timm.create_model(f"hf_hub:{REPO_ID}", pretrained=True)
            model.eval()
            if self._device == "cuda":
                model = model.half().to(self._device)
            self._model = model
            return model

    def unload(self) -> None:
        """Free the model + its VRAM so a generation can use the card."""
        with self._lock:
            if self._model is None:
                return
            self._model = None
        if self._device == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001 — best-effort
                pass

    # ── tag table ───────────────────────────────────────────────────
    def _rating_spec(self):
        """The 4 rating rows (category 9) of selected_tags.csv — found by
        category, never by position, so tag order can't break the mapping.

        WARNING: the model's output index i corresponds to CSV data row i+1
        (row 0 is the header), so the category-9 rows live at output indices
        0..3 — enumerate with start=0, not 1. A one-off shift here puts the
        `1girl` tag (≈0.99 on any anime image) into the explicit slot and
        blurs the whole gallery.
        Also builds the decision layer's corroborating tag index sets."""
        if self._rating_idx is None:
            from huggingface_hub import hf_hub_download
            csv_path = hf_hub_download(REPO_ID, "selected_tags.csv")
            with open(csv_path, newline="") as fh:
                rows = list(csv.reader(fh))
            self._rows = rows
            self._rating_idx = [
                i for i, r in enumerate(rows[1:])
                if len(r) > 2 and r[2] == "9"
            ]
            by_name = {r[1]: i for i, r in enumerate(rows[1:])}
            # Missing tag names drop out silently — the decision layer only
            # ever *adds* corroboration, never invents it.
            self._hard_idx = [by_name[t] for t in HARD_EXPLICIT_TAGS if t in by_name]
            self._strong_idx = [by_name[t] for t in STRONG_SUGGESTIVE_TAGS if t in by_name]
            self._soft_idx = [by_name[t] for t in SUGGESTIVE_TAGS if t in by_name]
        return self._rating_idx

    # ── inference ───────────────────────────────────────────────────
    def rate(self, paths: List[Path]) -> List[Optional[dict]]:
        """Rate each image; returns one entry per input, ``None`` on failure.

        Entry: ``{"rating": tier, "nsfw": bool, "confidence": float,
        "reason": str}``. The verdict comes from the tag-corroborated decision
        layer, not the raw rating argmax. Batching amortises the fixed forward
        cost for gallery scans.
        """
        if not paths:
            return []
        self.load()
        self._rating_spec()
        idx = self._rating_idx
        # One slot per input, filled by position: a file that fails to open
        # leaves its own slot None instead of shifting every later verdict of
        # the batch onto the wrong image.
        out: List[Optional[dict]] = [None] * len(paths)
        with torch.no_grad():
            for start in range(0, len(paths), BATCH_SIZE):
                chunk = paths[start:start + BATCH_SIZE]
                tensors = []
                slots = []
                for j, p in enumerate(chunk):
                    try:
                        with Image.open(p) as im:
                            tensors.append(_preprocess(im))
                        slots.append(start + j)
                    except Exception:  # noqa: BLE001 — one bad file skips only itself
                        pass
                if not tensors:
                    continue
                x = torch.stack(tensors)
                if self._device == "cuda":
                    x = x.half().to(self._device)
                logits = self._model(x)
                probs = torch.sigmoid(logits).float().cpu().numpy()
                for slot, row in zip(slots, probs):
                    rating, conf, reason = decide_rating(
                        row[idx], row, self._hard_idx, self._strong_idx,
                        self._soft_idx)
                    out[slot] = {"rating": rating, "nsfw": rating in NSFW_TIERS,
                                 "confidence": conf, "reason": reason}
        return out

    def rate_one(self, path: Path) -> Optional[dict]:
        res = self.rate([path])
        return res[0] if res else None


TAGGER = Tagger()
