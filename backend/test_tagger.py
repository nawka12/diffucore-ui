"""Tests for the WD tagger rating logic (pure functions, no GPU, no model)."""

from __future__ import annotations

import numpy as np

import tagger


def _decide(ratings, hard_probs=(), strong_probs=(), soft_probs=(), n_tags=32):
    """Run the decision layer over synthetic sigmoids. ``ratings`` is
    [general, sensitive, questionable, explicit]; hard/strong/soft tag
    probabilities sit at indices 0.., 8.., 16..
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
    # Real case: sensitive=0.981 beats explicit=0.922 at argmax, but nipples
    # scored 0.80.
    tier, conf, reason = _decide([0.002, 0.981, 0.120, 0.922], hard_probs=[0.80])
    assert tier == "X"
    assert "hard_tag" in reason
    assert conf > 0.9


def test_moona_case_explicit_head_without_corroboration_deescalates():
    # Real case: a SFW VTuber portrait at explicit=0.993 with only weak cues.
    tier, _, reason = _decide([0.153, 0.002, 0.0009, 0.993], soft_probs=[0.8])
    assert tier in ("PG", "PG13")
    assert "de_escalated" in reason


def test_explicit_head_backed_by_questionable_deescalates_to_r_not_pg():
    # Real misses: explicit-headed, no strong tags, nipples just under
    # HARD_TAG_THRESH. A questionable head >= 0.5 holds them at R.
    for ratings, hard in (([0.0167, 0.0032, 0.5269, 0.8179], [0.298]),
                          ([0.0443, 0.0121, 0.6226, 0.5762], [0.063]),
                          ([0.0244, 0.0044, 0.6729, 0.6401], [0.345])):
        tier, _, reason = _decide(ratings, hard_probs=hard)
        assert tier == "R", ratings
        assert "de_escalated" in reason


def test_explicit_head_with_weak_questionable_still_deescalates_to_pg():
    # Real false alarm: explicit 0.49 / questionable 0.38 stays unblurred.
    tier, _, reason = _decide([0.1682, 0.0371, 0.3784, 0.4875], hard_probs=[0.07])
    assert tier == "PG"
    assert "de_escalated" in reason


def test_explicit_with_strong_corroboration_stays_explicit():
    tier, _, reason = _decide([0.05, 0.10, 0.10, 0.90], strong_probs=[0.9])
    assert tier == "X"
    assert reason == "explicit"


def test_weak_suggestive_alone_never_forces_explicit():
    # A high explicit head with only cleavage/bikini must not blur.
    tier, _, _ = _decide([0.10, 0.10, 0.20, 0.85], soft_probs=[0.95])
    assert tier in ("PG", "PG13")


def test_near_tie_without_tags_deescalates():
    # Near-tie (explicit 0.55 vs sensitive 0.52), no tags: downgraded.
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
    # A file that can't be decoded must leave its own slot None, not shift
    # later verdicts onto the wrong images.
    import tempfile
    import torch
    from pathlib import Path
    from PIL import Image

    t = tagger.Tagger()
    t._device = "cpu"
    t._rating_idx = [0, 1, 2, 3]
    t._rating_spec = lambda: t._rating_idx
    t._hard_idx = t._strong_idx = t._soft_idx = []
    t._model = lambda x: torch.tensor([[-5.0, -5.0, -5.0, 5.0] + [-5.0] * 4]
                                      ).repeat(x.shape[0], 1)
    t.load = lambda: t._model

    d = Path(tempfile.mkdtemp())
    good1, bad, good2 = d / "a.png", d / "b.png", d / "c.png"
    Image.new("RGB", (8, 8)).save(good1)
    bad.write_text("not an image")
    Image.new("RGB", (8, 8), (255, 0, 0)).save(good2)

    res = t.rate([good1, bad, good2])
    assert len(res) == 3
    assert res[0] is not None and res[1] is None and res[2] is not None


def test_rate_survives_unload_mid_scan(monkeypatch):
    # Turning the blur off calls unload() from a request thread while a scan
    # runs; the batch in flight must finish on the model it started with.
    import tempfile
    import torch
    from pathlib import Path
    from PIL import Image

    t = tagger.Tagger()
    t._device = "cpu"
    t._rating_idx = [0, 1, 2, 3]
    t._rating_spec = lambda: t._rating_idx
    t._hard_idx = t._strong_idx = t._soft_idx = []

    def model(x):
        t.unload()
        return torch.tensor([[-5.0, -5.0, -5.0, 5.0] + [-5.0] * 4]).repeat(x.shape[0], 1)

    t._model = model
    monkeypatch.setattr(tagger, "BATCH_SIZE", 1)
    d = Path(tempfile.mkdtemp())
    paths = [d / "a.png", d / "b.png"]
    for p in paths:
        Image.new("RGB", (8, 8)).save(p)

    res = t.rate(paths)
    assert all(r is not None for r in res)
    assert not t.loaded


def test_rating_tier_mapping_has_all_wd_ratings():
    for wd in ("general", "sensitive", "questionable", "explicit"):
        assert wd in tagger.RATING_TO_TIER
    assert tagger.RATING_TO_TIER["questionable"] == "R"
    assert tagger.RATING_TO_TIER["explicit"] == "X"


def test_corroborating_tag_sets_only_contain_known_tags():
    # Tags missing from the vocabulary are filtered out.
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
        # nipples (4) hard, lingerie (6) strong, cleavage (7) weak; the typo
        # "lingerine" (5) is in none.
        assert 4 in t._hard_idx
        assert 6 in t._strong_idx
        assert 7 in t._soft_idx
        assert 5 not in t._hard_idx and 5 not in t._strong_idx and 5 not in t._soft_idx
