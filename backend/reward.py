"""WD-tagger reward for the ``cogent3_reward`` samplers.

The prompt's own danbooru tags are the target: an image scores the mean log
probability the WD tagger gives them. The tagger is the one ``tagger.py`` loads
(lazy, fp16, resident); only the tag table and a GPU preprocess live here.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F

import tagger as tagger_mod

# Fewer matched tags than this and the prompt gives the reward nothing to measure.
MIN_TAGS = 3
PROB_FLOOR = 0.01

_QUALITY = re.compile(
    r"^(masterpiece|best quality|high quality|good quality|normal quality|low quality|"
    r"worst quality|amazing quality|very aesthetic|aesthetic|absurdres|highres|newest|"
    r"oldest|recent|early|late|mid|year \d{4}|score_\d+(_up)?|safe|sensitive|nsfw|"
    r"explicit|questionable|general|rating:.*)$")
# 1girl, 2boys, 6+girls, 1other, solo, multiple_girls: true of almost any image, so they
# don't count as evidence that the prompt describes anything.
_COUNT = re.compile(r"^(\d+\+?(girl|boy|other)s?|solo|multiple_(girls|boys|others))$")


def prompt_tags(prompt: str, vocab: Dict[str, int]) -> List[str]:
    """The prompt's tags that are in ``vocab`` (the tagger's general/character names), in
    prompt order, without repeats. Quality, score, year and rating tokens, ``@artist``
    and weights/brackets are dropped; escaped parens (``bow \\(weapon\\)``) are kept."""
    out: List[str] = []
    for raw in re.split(r"[,\n]", prompt):
        t = raw.strip().lower()
        if not t or t.startswith("@"):
            continue
        t = re.sub(r":[\d.]+\)?$", "", t)
        t = re.sub(r"(?<!\\)[()\[\]{}]", "", t).replace("\\(", "(").replace("\\)", ")")
        t = t.strip().rstrip(".").strip()
        if not t or _QUALITY.match(t):
            continue
        t = t.replace(" ", "_")
        if t in vocab and t not in out:
            out.append(t)
    return out


def has_enough_tags(tags: List[str]) -> bool:
    """Whether ``tags`` holds at least :data:`MIN_TAGS` besides the count tags."""
    return sum(1 for t in tags if not _COUNT.match(t)) >= MIN_TAGS


def tag_probs(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Tag probabilities ``[B, n_tags]`` for ``images`` ``[B, 3, H, W]`` in [-1, 1]: padded to
    a white square, 448 bicubic, RGB to BGR, as the wdv3-timm reference does, on the GPU."""
    param = next(model.parameters())
    img = images.to(param.device).float().clamp(-1, 1)
    _, _, h, w = img.shape
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    img = F.pad(img, (left, side - w - left, top, side - h - top), value=1.0)
    img = F.interpolate(img, size=(tagger_mod.IMG_SIZE,) * 2, mode="bicubic",
                        antialias=True, align_corners=False)
    img = img.clamp(-1, 1).flip(1)
    with torch.no_grad():
        return torch.sigmoid(model(img.to(param.dtype))).float()


def _vocab() -> Dict[str, int]:
    """Tagger output index of every general (0) and character (4) tag."""
    tagger_mod.TAGGER._rating_spec()
    rows = tagger_mod.TAGGER._rows[1:]
    return {r[1]: i for i, r in enumerate(rows) if r[2] in ("0", "4")}


def reward_fn_for(prompt: str) -> Optional[Callable[[torch.Tensor], torch.Tensor]]:
    """Reward function for ``prompt`` (``images[B, 3, H, W]`` in [-1, 1] to ``Tensor[B]``,
    higher is better), with the matched tags on ``.tags``; ``None`` when the prompt has too
    few usable tags or ``timm`` is missing. The tagger loads only once there are enough."""
    if not tagger_mod.timm_available():
        return None
    vocab = _vocab()
    tags = prompt_tags(prompt, vocab)
    if not has_enough_tags(tags):
        return None
    model = tagger_mod.TAGGER.load()
    idx = torch.tensor([vocab[t] for t in tags], device=next(model.parameters()).device)

    def reward(images: torch.Tensor) -> torch.Tensor:
        p = tag_probs(model, images)[:, idx]
        return p.clamp(min=PROB_FLOOR, max=1.0).log().mean(dim=1)

    reward.tags = tags
    return reward
