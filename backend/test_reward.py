"""WD-tagger reward for the cogent3_reward samplers: prompt parsing, the few-tags fallback,
the tagger preprocess/reward maths, and sampler wiring. No model, no download.

    .venv/bin/python -m pytest backend/test_reward.py -v
"""

from __future__ import annotations

import pytest
import torch

import engine
import metadata as md
import reward as rw
import tagger

_VOCAB = {t: i for i, t in enumerate([
    "1girl", "solo", "long_hair", "blonde_hair", "blue_eyes", "white_shirt", "blue_skirt", "lying",
    "on_back", "on_bed", "holding_book", "bow_(weapon)", "holding", "smile", "2girls", "multiple_girls",
    "1boy", "6+girls", "indoors", "window", "red_dress", "hatsune_miku",
])}


def _tags(prompt):
    return rw.prompt_tags(prompt, _VOCAB)


def test_splits_on_commas_and_newlines_and_lowercases():
    assert _tags("Long Hair,\nBlonde HAIR , smile") == ["long_hair", "blonde_hair", "smile"]


def test_drops_artist_tags_and_unknown_words():
    assert _tags("@teruwo, @some artist, long hair, nonsense tag") == ["long_hair"]


@pytest.mark.parametrize("prompt, want", [
    ("(long hair:1.2), smile", ["long_hair", "smile"]),
    ("(blonde hair:0.8)", ["blonde_hair"]),
    ("[smile], {indoors}, ((window))", ["smile", "indoors", "window"]),
    ("bow \\(weapon\\), (holding:1.1)", ["bow_(weapon)", "holding"]),
    ("(bow \\(weapon\\):1.3)", ["bow_(weapon)"]),
])
def test_strips_weights_and_brackets_but_keeps_escaped_parens(prompt, want):
    assert _tags(prompt) == want


def test_strips_a_trailing_period_and_whitespace():
    assert _tags("smile., long hair. , window .") == ["smile", "long_hair", "window"]


@pytest.mark.parametrize("junk", [
    "masterpiece", "best quality", "worst quality", "score_9", "score_6_up", "newest", "year 2024",
    "safe", "general", "sensitive", "questionable", "explicit", "nsfw", "rating:safe", "highres",
])
def test_drops_quality_score_year_and_rating_tokens(junk):
    vocab = {**_VOCAB, junk.replace(" ", "_"): 99}      # even if the tagger knew the word
    assert rw.prompt_tags(f"{junk}, smile", vocab) == ["smile"]


def test_only_the_tagger_vocab_matches_and_repeats_collapse():
    assert _tags("smile, smile, Smile") == ["smile"]
    assert _tags("hatsune miku, red dress") == ["hatsune_miku", "red_dress"]


def test_count_tags_are_kept_for_the_reward_but_do_not_count_toward_the_minimum():
    only_counts = _tags("1girl, solo, 2girls, multiple girls, 1boy, 6+girls, smile, long hair")
    assert only_counts == ["1girl", "solo", "2girls", "multiple_girls", "1boy", "6+girls", "smile", "long_hair"]
    assert not rw.has_enough_tags(only_counts)          # just 2 real tags
    assert rw.has_enough_tags(only_counts + ["window"])
    assert not rw.has_enough_tags([])


def test_threshold_is_three_non_count_tags():
    assert not rw.has_enough_tags(["smile", "window"])
    assert rw.has_enough_tags(["smile", "window", "indoors"])
    assert rw.MIN_TAGS == 3


def test_the_bed_prompt_keeps_its_tags():
    prompt = ("masterpiece, best quality, safe, @teruwo, 1girl, solo, mature female, long hair, blonde hair, "
              "blue eyes, white shirt, blue skirt, lying, on back, on bed, holding book, pillow, "
              "bedroom, indoors, window, smile")
    assert _tags(prompt) == ["1girl", "solo", "long_hair", "blonde_hair", "blue_eyes", "white_shirt",
                             "blue_skirt", "lying", "on_back", "on_bed", "holding_book", "indoors",
                             "window", "smile"]


# ── preprocess and reward maths ────────────────────────────────────────

class _Probe(torch.nn.Module):
    """Logits that expose the preprocessed input: channel means in channels 0-2, size in 3."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.seen = None

    def forward(self, x):
        self.seen = x
        return torch.cat([x.mean(dim=(2, 3)), torch.full((x.shape[0], 1), float(x.shape[-1]))], dim=1)


def test_tag_probs_pads_to_a_white_square_resizes_to_448_and_flips_to_bgr():
    probe = _Probe()
    img = torch.zeros(1, 3, 100, 60)
    img[:, 0] = 1.0                                     # pure red, in [-1, 1] terms: R=1, G=B=0
    img[:, 1:] = -1.0
    probs = rw.tag_probs(probe, img)
    assert probe.seen.shape == (1, 3, 448, 448)
    assert probs.shape == (1, 4) and probs[0, 3] == pytest.approx(torch.sigmoid(torch.tensor(448.0)))
    # BGR: the red channel ends up last
    centre = probe.seen[0, :, 224, 224]
    assert centre[2] == pytest.approx(1.0, abs=1e-3) and centre[0] == pytest.approx(-1.0, abs=1e-3)
    # padding is white on the short sides and untouched in the middle
    assert probe.seen[0, :, 224, 2].min() > 0.99 and probe.seen[0, :, 224, 445].min() > 0.99


def test_reward_is_the_mean_log_of_the_floored_probabilities(monkeypatch):
    monkeypatch.setattr(rw.tagger_mod, "timm_available", lambda: True)
    monkeypatch.setattr(rw, "_vocab", lambda: _VOCAB)
    probs = torch.zeros(2, len(_VOCAB))
    probs[0, [_VOCAB["smile"], _VOCAB["window"], _VOCAB["indoors"]]] = torch.tensor([0.5, 0.25, 1.0])
    probs[1, _VOCAB["smile"]] = 0.0                     # floored to 0.01
    monkeypatch.setattr(rw, "tag_probs", lambda model, images: probs)
    monkeypatch.setattr(rw.tagger_mod.TAGGER, "load", lambda: _Probe())
    fn = rw.reward_fn_for("smile, window, indoors")
    assert fn.tags == ["smile", "window", "indoors"]
    got = fn(torch.zeros(2, 3, 8, 8))
    assert got[0].item() == pytest.approx((torch.log(torch.tensor(0.5)) + torch.log(torch.tensor(0.25))).item() / 3)
    assert got[1].item() == pytest.approx(torch.log(torch.tensor(0.01)).item())


def test_too_few_tags_returns_none_without_loading_the_tagger(monkeypatch):
    monkeypatch.setattr(rw.tagger_mod, "timm_available", lambda: True)
    monkeypatch.setattr(rw, "_vocab", lambda: _VOCAB)

    def boom():
        raise AssertionError("the tagger must not load")
    monkeypatch.setattr(rw.tagger_mod.TAGGER, "load", boom)
    assert rw.reward_fn_for("1girl, solo, smile, window") is None
    assert rw.reward_fn_for("masterpiece, @artist") is None
    monkeypatch.setattr(rw.tagger_mod, "timm_available", lambda: False)
    assert rw.reward_fn_for("smile, window, indoors, long hair") is None


def test_preprocess_leaves_the_taggers_own_alone():
    assert tagger._preprocess.__module__ == "tagger" and rw.tag_probs is not tagger._preprocess


# ── wiring ─────────────────────────────────────────────────────────────

NAMES = ["cogent3_reward"]


def test_names_are_anima_only_and_grouped_right_after_cogent3():
    for n in NAMES:
        assert n in engine.SAMPLERS_ANIMA
        for other in (engine.SAMPLERS_SD, engine.SAMPLERS_FLUX, engine.SAMPLERS_FLOW):
            assert n not in other
    group = engine.SAMPLER_GROUPS["Diffucore"]
    assert group[group.index("cogent3") + 1:group.index("cogent3") + 1 + len(NAMES)] == NAMES


def test_registry_and_pipeline_know_both_names():
    from diffucore.pipelines._anima import _ANIMA_SAMPLERS, _FLOW_AWARE_SAMPLERS
    from diffucore.sampling import get_sampler
    for n in NAMES:
        assert get_sampler(n).keywords["k"] == 6
        assert n in _ANIMA_SAMPLERS and n in _FLOW_AWARE_SAMPLERS


def test_gate_reduce_and_eta_max_knobs_reach_the_reward_samplers():
    import server
    for n in NAMES:
        knobs = server._settings_knobs(n, "flow", 0.0)
        assert "gate_reduce" in knobs
    from unittest import mock
    with mock.patch.object(server.ENGINE, "_loaded", mock.Mock(family="anima")):
        for n in NAMES:
            assert "eta_max" in server._settings_knobs(n, "flow", 0.0)


_BASE = {"prompt": "a fox", "negative_prompt": "", "steps": 32, "sampler": "cogent3_reward",
         "scheduler": "pump_taper", "cfg_scale": 7.0, "width": 1024, "height": 1536,
         "gate_reduce": "per_channel"}


class _Eng:
    loaded_name = "anima-test"
    last_seed = 5
    perf_flags_str = "default"


def test_metadata_gets_one_reward_line_only_when_the_pick_happened():
    assert "Reward" not in md.format_metadata(_BASE, _Eng())
    gen = {**_BASE, "reward": "k=6, pick=3, tags=smile, window, indoors"}
    text = md.format_metadata(gen, _Eng())
    assert 'Reward: "k=6, pick=3, tags=smile, window, indoors"' in text
    assert "Gate reduce: per_channel" in text
    parsed = md.parse_metadata(text)
    assert parsed["reward"] == "k=6, pick=3, tags=smile, window, indoors" and parsed["sampler"] == "cogent3_reward"
    assert "reward" in md.format_swarmui_metadata(gen, _Eng())


def test_calibration_line_counts_the_band_a_reward_pick_forced():
    gen = {**_BASE, "lowres_calibrated": True}
    assert "Half-res calibration" not in md.format_metadata(gen, _Eng())
    assert "Half-res calibration: on" in md.format_metadata({**gen, "reward": "k=6, pick=0, tags=a"}, _Eng())


def _anima_engine(monkeypatch, pick, parents=None):
    from types import SimpleNamespace
    from PIL import Image
    seen = {}

    class FakePipeline:
        def __init__(self, model):
            pass

        def __call__(self, **kw):
            seen.clear()
            seen.update(kw)
            reward = {"k": 6, "pick": pick, "scores": [0.0] * 6} if "reward_fn" in kw else None
            if reward and parents:
                reward.update(parents=parents, weights=[1 / parents] * parents)
            return Image.new("RGB", (8, 8)), SimpleNamespace(vae_decode_mode="full", reward=reward)

    monkeypatch.setattr(engine, "TextToImage", FakePipeline)
    monkeypatch.setattr(engine, "reward_fn_for", lambda prompt: _fn if len(prompt) > 5 else None)
    eng = engine.Engine(device="cpu")
    eng._loaded = engine.LoadedModel(name="Anima(x)", family="anima", native_res=1024,
                                     model=SimpleNamespace(backbone=SimpleNamespace(blocks=[0] * 28)))
    return eng, seen


def _fn(images):
    return torch.zeros(1)


_fn.tags = ["smile", "window", "indoors"]


def test_engine_passes_the_reward_fn_only_to_a_reward_sampler_with_enough_tags(monkeypatch):
    eng, seen = _anima_engine(monkeypatch, pick=2)
    _, info = eng.generate_t2i("a long prompt", sampler="cogent3_reward", scheduler="flow", steps=8)
    assert seen["reward_fn"] is _fn
    assert "Reward: k=6, pick=2, tags=smile, window, indoors" in info
    assert eng.last_reward == "k=6, pick=2, tags=smile, window, indoors"
    _, info = eng.generate_t2i("a long prompt", sampler="cogent3", scheduler="flow", steps=8)
    assert "reward_fn" not in seen and "Reward" not in info and eng.last_reward == ""
