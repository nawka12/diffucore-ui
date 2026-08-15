"""Tests for the WD tagger rating logic (pure functions, no GPU, no model).

Run from the project root::

    .venv/bin/python -m pytest backend/test_tagger.py -v
"""

from __future__ import annotations

import numpy as np

import tagger


def _decide(ratings, hard_probs=(), strong_probs=(), soft_probs=(), n_tags=32):
    """Run the decision layer over synthetic sigmoid outputs.

    ``ratings`` = [general, sensitive, questionable, explicit]; hard/strong/
    soft tag probabilities are placed at reserved indices (hard=0..,
    strong=8.., soft=16..).
    """
    tag_scores = np.zeros(n_tags)
    for i, p in enumerate(hard_probs):
        tag_scores[i] = p
    for i, p in enumerate(strong_probs):
        tag_scores[8 + i] = p
    for i, p in enumerate(soft_probs):
        tag_scores[16 + i] = p
    return tagger.decide_rating(
        np.array(ratings, dtype=np.float64), tag_scores,
        hard_idx=[0, 1, 2], strong_idx=[8, 9, 10], soft_idx=[16, 17, 18])


def test_nerissa_case_hard_tag_overrides_rating_head():
    # The real failure: sensitive=0.981 beats explicit=0.922 at argmax, yet the
    # model detected nipples (0.80). The hard-tag rule must force X.
    tier, conf, reason = _decide([0.002, 0.981, 0.120, 0.922], hard_probs=[0.80])
    assert tier == "X"
    assert "hard_tag" in reason
    assert conf > 0.9


def test_moona_case_explicit_head_without_corroboration_deescalates():
    # The real failure: a SFW VTuber portrait scored explicit=0.993 with only
    # weak cues (cleavage on a busty idol costume) and no anatomy/strong
    # suggestive tags. Must not blur.
    tier, _, reason = _decide([0.153, 0.002, 0.0009, 0.993], soft_probs=[0.8])
    assert tier in ("PG", "PG13")
    assert "de_escalated" in reason


def test_explicit_with_strong_corroboration_stays_explicit():
    # Explicit head + lingerie → blurred (genuinely NSFW).
    tier, _, reason = _decide([0.05, 0.10, 0.10, 0.90], strong_probs=[0.9])
    assert tier == "X"
    assert reason == "explicit"


def test_weak_suggestive_alone_never_forces_explicit():
    # Even a high-confidence explicit head with cleavage/bikini only (the
    # busty-idol false positive) must not blur — weak cues corroborate R at
    # most.
    tier, _, _ = _decide([0.10, 0.10, 0.20, 0.85], soft_probs=[0.95])
    assert tier in ("PG", "PG13")


def test_near_tie_without_tags_deescalates():
    # The sigmoid near-tie that argmax loses: explicit 0.55 vs sensitive 0.52,
    # no corroborating tags → downgraded, not a coin-flip X.
    tier, _, reason = _decide([0.10, 0.52, 0.30, 0.55])
    assert tier == "PG13"
    assert "de_escalated" in reason


def test_questionable_head_rates_r():
    tier, _, reason = _decide([0.10, 0.30, 0.60, 0.40])
    assert tier == "R"
    assert reason == "questionable"


def test_sensitive_head_rates_pg13():
    tier, _, reason = _decide([0.20, 0.60, 0.10, 0.10])
    assert tier == "PG13"
    assert reason == "sensitive"


def test_low_confidence_everywhere_rates_pg():
    # The old bare-threshold gate, folded into the cascade: uncertainty → safe.
    tier, _, reason = _decide([0.30, 0.28, 0.26, 0.24])
    assert tier == "PG"
    assert reason == "general"


def test_suggestive_tags_alone_keep_it_pg13():
    tier, _, _ = _decide([0.40, 0.30, 0.10, 0.10], soft_probs=[0.8])
    assert tier == "PG13"


def test_preprocess_normalises_to_unit_range():
    from PIL import Image
    img = Image.new("RGB", (900, 1200), (0, 0, 0))
    x = tagger._preprocess(img)
    assert tuple(x.shape) == (3, tagger.IMG_SIZE, tagger.IMG_SIZE)
    assert x.min() >= -1.0 and x.max() <= 1.0
    img = Image.new("RGB", (64, 64), (255, 255, 255))
    x = tagger._preprocess(img)
    assert abs(float(x.max()) - 1.0) < 1e-5


def test_rate_keeps_results_aligned_when_a_file_fails_to_open():
    # A file that can't be decoded must leave *its own* slot None. Appending the
    # None as it happens would shift every later verdict of the batch onto the
    # wrong image — i.e. blur the wrong output, or unblur an explicit one.
    import tempfile
    import torch
    from pathlib import Path
    from PIL import Image

    t = tagger.Tagger()
    t._device = "cpu"
    t.load = lambda: None
    t._rating_idx = [0, 1, 2, 3]
    t._rating_spec = lambda: t._rating_idx
    t._hard_idx = t._strong_idx = t._soft_idx = []
    # explicit head high → "de_escalated" for every decodable image.
    t._model = lambda x: torch.tensor([[-5.0, -5.0, -5.0, 5.0] + [-5.0] * 4]
                                      ).repeat(x.shape[0], 1)

    d = Path(tempfile.mkdtemp())
    good1, bad, good2 = d / "a.png", d / "b.png", d / "c.png"
    Image.new("RGB", (8, 8)).save(good1)
    bad.write_text("not an image")
    Image.new("RGB", (8, 8), (255, 0, 0)).save(good2)

    res = t.rate([good1, bad, good2])
    assert len(res) == 3
    assert res[0] is not None and res[1] is None and res[2] is not None


def test_rating_tier_mapping_has_all_wd_ratings():
    for wd in ("general", "sensitive", "questionable", "explicit"):
        assert wd in tagger.RATING_TO_TIER
    assert tagger.RATING_TO_TIER["questionable"] == "R"
    assert tagger.RATING_TO_TIER["explicit"] == "X"


def test_corroborating_tag_sets_only_contain_known_tags():
    # A tag that doesn't exist in the model's vocabulary must not silently
    # weaken the decision layer — build from a fake CSV and check filtering.
    import tempfile
    from pathlib import Path
    import tagger as tagger_mod

    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "tags.csv"
        csv_path.write_text(
            "tag_id,name,category,count\n"
            "9999999,general,9,1\n9999998,sensitive,9,1\n"
            "9999997,questionable,9,1\n9999996,explicit,9,1\n"
            "1,nipples,0,1\n2,lingerine,0,1\n"          # typo: not in any set
            "3,lingerie,0,1\n4,cleavage,0,1\n"
        )
        t = tagger_mod.Tagger()
        import huggingface_hub
        orig = huggingface_hub.hf_hub_download
        huggingface_hub.hf_hub_download = lambda *a, **k: str(csv_path)
        try:
            t._rating_spec()
        finally:
            huggingface_hub.hf_hub_download = orig
        assert t._rating_idx == [0, 1, 2, 3]
        # nipples (4) → hard; lingerie (6) → strong; cleavage (7) → weak; the
        # typo "lingerine" (5) is in none of them.
        assert 4 in t._hard_idx
        assert 6 in t._strong_idx
        assert 7 in t._soft_idx
        assert 5 not in t._hard_idx and 5 not in t._strong_idx and 5 not in t._soft_idx
