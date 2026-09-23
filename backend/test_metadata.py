"""Tests for the PNG metadata round-trip (pure functions, no GPU)."""

from __future__ import annotations

import metadata as md


class _StubEngine:
    """Minimal stand-in for the fields ``format_metadata`` reads off the engine."""
    loaded_name = "anima-test"
    last_seed = 2150942283
    perf_flags_str = "default"


class _WrappedEngine(_StubEngine):
    """An Anima split-DiT load, whose name carries the ``Anima(...)`` wrapper."""
    loaded_name = "Anima(AnimaPulse-1.1.safetensors)"


def _roundtrip(gen_kwargs: dict, **kw) -> dict:
    params = md.format_metadata(gen_kwargs, _StubEngine(), **kw)
    return md.workspace_fields(md.parse_metadata(params))


def _roundtrip_swarm(gen_kwargs: dict, **kw) -> dict:
    """:func:`_roundtrip` through the SwarmUI formatter (parse_metadata
    auto-detects the JSON)."""
    params = md.format_swarmui_metadata(gen_kwargs, _StubEngine(), **kw)
    return md.workspace_fields(md.parse_metadata(params))


_BASE_GEN = {
    "prompt": "a fox", "negative_prompt": "blurry",
    "steps": 32, "sampler": "euler", "scheduler": "flow",
    "cfg_scale": 5.0, "width": 1024, "height": 1536,
}


def test_upscale_roundtrips_through_metadata():
    """An upscaled image's settings survive format -> parse -> workspace_fields."""
    upscale = {
        "scale": 4.0, "tile": 1024, "overlap": 128, "denoise": 0.25,
        "teacache": 0.1, "base": "4x-UltraSharp.pth", "prompt": "a fox, sharp",
    }
    up = _roundtrip(_BASE_GEN, upscale=upscale)["upscale"]
    assert up["enabled"] is True
    assert up["scale"] == 4.0
    assert up["tile"] == 1024
    assert up["overlap"] == 128
    assert up["denoise"] == 0.25
    assert up["teacache"] == 0.1
    assert up["base"] == "4x-UltraSharp.pth"
    assert up["prompt"] == "a fox, sharp"   # comma survives the quote/unquote path


def test_lanczos_base_maps_to_form_default():
    """Lanczos (no ESRGAN model) round-trips to the form's empty base."""
    up = md.extract_upscale({"upscale_scale": "2.0", "upscale_base": "Lanczos"})
    assert up["base"] == ""


def test_no_upscale_meta_leaves_fields_untouched():
    """An image generated without the upscaler yields no ``upscale`` key."""
    assert "upscale" not in _roundtrip(_BASE_GEN)


def test_teacache_calibrated_roundtrips():
    fields = _roundtrip({**_BASE_GEN, "teacache_thresh": 0.15, "teacache_use_coeffs": True})
    assert fields["teacacheOn"] is True
    assert fields["teacache"] == 0.15
    assert fields["teacacheCalibrated"] is True


def test_teacache_raw_roundtrips():
    """The "(raw)" suffix restores the calibrated toggle as off."""
    fields = _roundtrip({**_BASE_GEN, "teacache_thresh": 0.3, "teacache_use_coeffs": False})
    assert fields["teacacheOn"] is True
    assert fields["teacache"] == 0.3
    assert fields["teacacheCalibrated"] is False


def test_no_teacache_meta_leaves_toggle_untouched():
    """TeaCache off writes no line, so nothing is restored (additive)."""
    fields = _roundtrip(_BASE_GEN)
    assert "teacacheOn" not in fields
    assert "teacacheForecast" not in fields


def test_teacache_forecast_roundtrips():
    """Both forecast bases survive the A1111 and SwarmUI round-trips."""
    for forecast in ("hermite", "taylor"):
        gen = {**_BASE_GEN, "teacache_thresh": 0.15, "teacache_forecast": forecast}
        assert _roundtrip(gen)["teacacheForecast"] == forecast
        assert _roundtrip_swarm(gen)["teacacheForecast"] == forecast


def test_teacache_forecast_absent_restores_taylor():
    """Pre-HiCache images carry no forecast key but used taylor."""
    fields = md.workspace_fields({"teacache": "0.15"})
    assert fields["teacacheForecast"] == "taylor"


def test_teacache_rule_roundtrips():
    """Both skip-decision rules survive the A1111 and SwarmUI round-trips."""
    for rule in ("drift", "easy"):
        gen = {**_BASE_GEN, "teacache_thresh": 0.15, "teacache_rule": rule}
        assert _roundtrip(gen)["teacacheRule"] == rule
        assert _roundtrip_swarm(gen)["teacacheRule"] == rule


def test_teacache_rule_absent_restores_drift():
    """Pre-EasyCache images carry no rule key and all used the drift rule."""
    assert md.workspace_fields({"teacache": "0.15"})["teacacheRule"] == "drift"
    # Off entirely -> nothing restored, as for the forecast (additive).
    assert "teacacheRule" not in _roundtrip(_BASE_GEN)


def test_teacache_uncond_scale_written_only_when_set():
    """Written when raised, absent at 1.0, never restored onto the form."""
    gen = {**_BASE_GEN, "teacache_thresh": 0.15}
    assert "TeaCache uncond scale" not in md.format_metadata(gen, _StubEngine())
    gen2 = {**gen, "teacache_uncond_scale": 2.0}
    assert "TeaCache uncond scale: 2.0" in md.format_metadata(gen2, _StubEngine())
    assert md.format_swarmui_metadata(gen2, _StubEngine()).count("teacache_uncond_scale") == 1
    assert "teacacheUncondScale" not in _roundtrip(gen2)


def test_deepcache_roundtrips():
    fields = _roundtrip({**_BASE_GEN, "deepcache_interval": 3})
    assert fields["deepcacheOn"] is True
    assert fields["deepcache"] == 3


def test_deepcache_off_leaves_toggle_untouched():
    """interval 1 (off) writes no line, so nothing is restored (additive)."""
    fields = _roundtrip({**_BASE_GEN, "deepcache_interval": 1})
    assert "deepcacheOn" not in fields
    assert "deepcache" not in fields


def test_cogent4_gate_reduce_is_recorded_in_both_metadata_formats():
    """The gate changes seeded output, so both writers must preserve the arm."""
    import json

    gen = {**_BASE_GEN, "sampler": "cogent3_pump", "gate_reduce": "per_channel"}
    a1111 = md.parse_metadata(md.format_metadata(gen, _StubEngine()))
    swarm = json.loads(md.format_swarmui_metadata(gen, _StubEngine()))
    assert a1111["gate_reduce"] == "per_channel"
    assert swarm["sui_extra_data"]["gate_reduce"] == "per_channel"


def test_gate_reduce_metadata_is_scoped_to_cogent_samplers():
    """An internal default passed to another sampler must not pollute its metadata."""
    gen = {**_BASE_GEN, "sampler": "euler", "gate_reduce": "all"}
    assert "gate_reduce" not in md.parse_metadata(md.format_metadata(gen, _StubEngine()))


# ── SwarmUI format ──────────────────────────────────────────────────

def test_swarmui_blob_shape_and_ids():
    """The SwarmUI writer emits the three root keys with idiomatic param IDs."""
    import json
    blob = json.loads(md.format_swarmui_metadata(_BASE_GEN, _StubEngine()))
    p = blob["sui_image_params"]
    assert p["prompt"] == "a fox"
    assert p["negativeprompt"] == "blurry"
    assert p["cfgscale"] == 5.0
    assert p["seed"] == 2150942283
    assert p["width"] == 1024 and p["height"] == 1536
    assert p["aspectratio"] == "2:3"
    assert blob["sui_models"][0]["param"] == "model"


def test_swarmui_core_fields_roundtrip():
    """prompt/neg/steps/cfg/sampler/scheduler/seed/size survive the SwarmUI path."""
    fields = _roundtrip_swarm(_BASE_GEN)
    assert fields["prompt"] == "a fox"
    assert fields["neg"] == "blurry"
    assert fields["steps"] == 32
    assert fields["cfg"] == 5.0
    assert fields["sampler"] == "euler"
    assert fields["scheduler"] == "flow"
    assert fields["seed"] == 2150942283
    assert fields["width"] == 1024 and fields["height"] == 1536


def test_swarmui_teacache_and_extras_roundtrip():
    """App-specific extras (TeaCache raw, shift, denoise) round-trip via sui_extra_data."""
    fields = _roundtrip_swarm({
        **_BASE_GEN, "shift": 3.0, "strength": 0.6,
        "teacache_thresh": 0.3, "teacache_use_coeffs": False,
    })
    assert fields["shift"] == 3.0
    assert fields["strength"] == 0.6
    assert fields["teacacheOn"] is True
    assert fields["teacache"] == 0.3
    assert fields["teacacheCalibrated"] is False


def test_swarmui_cleans_wrapped_model_name():
    """The ``Anima(...)`` wrapper is stripped; SwarmUI's model/name/hash conventions."""
    import json
    blob = json.loads(md.format_swarmui_metadata(_BASE_GEN, _WrappedEngine()))
    assert blob["sui_image_params"]["model"] == "AnimaPulse-1.1"          # no wrapper, no ext
    assert blob["sui_models"][0]["name"] == "AnimaPulse-1.1.safetensors"  # no wrapper, keeps ext
    assert blob["sui_models"][0]["hash"] is None    # not on disk, so uncached (not faked)


def test_a1111_cleans_wrapped_model_name():
    """The A1111 ``Model:`` field drops the family wrapper too (keeps extension)."""
    params = md.format_metadata(_BASE_GEN, _WrappedEngine())
    assert "Model: AnimaPulse-1.1.safetensors" in params
    assert "Anima(" not in params


def test_a1111_emits_civitai_hashes(monkeypatch):
    """AutoV2 hashes land in ``Model hash:`` and a quoted ``Lora hashes:``
    without disturbing the round-trip."""
    from pathlib import Path
    hashes = {"AnimaPulse-1.1.safetensors": "aaaaaaaaaa",
              "add-detail.safetensors": "bbbbbbbbbb"}
    monkeypatch.setattr(md.model_hash, "resolve_model_file",
                        lambda name: Path("/x/AnimaPulse-1.1.safetensors"))
    monkeypatch.setattr(md.model_hash, "resolve_lora_file",
                        lambda name: Path(f"/x/{name}.safetensors"))
    monkeypatch.setattr(md.model_hash, "get_autov2",
                        lambda path: hashes.get(path.name) if path else None)
    gen = {**_BASE_GEN, "prompt": "a fox <lora:add-detail:0.8>"}
    params = md.format_metadata(gen, _WrappedEngine())
    assert "Model hash: aaaaaaaaaa" in params
    assert 'Lora hashes: "add-detail: bbbbbbbbbb"' in params   # Forge-style quoted map
    fields = md.workspace_fields(md.parse_metadata(params))
    assert fields["prompt"] == "a fox <lora:add-detail:0.8>"   # inline tag preserved
    assert fields["seed"] == 2150942283                        # other fields intact


def test_a1111_no_hashes_when_uncached(monkeypatch):
    """No cached hash, no ``Model hash:`` / ``Lora hashes:`` lines."""
    monkeypatch.setattr(md.model_hash, "get_autov2", lambda path: None)
    params = md.format_metadata({**_BASE_GEN, "prompt": "a fox <lora:x:1>"}, _WrappedEngine())
    assert "Model hash:" not in params
    assert "Lora hashes:" not in params


def test_swarmui_loras_structured_not_inline():
    """LoRA tags are pulled out of the prompt into loras/loraweights + sui_models."""
    import json
    gen = {**_BASE_GEN,
           "prompt": "a fox <lora:add-detail:0.8>",
           "negative_prompt": "blurry <lora:bad-hands:1>"}
    blob = json.loads(md.format_swarmui_metadata(gen, _StubEngine()))
    p = blob["sui_image_params"]
    assert p["prompt"] == "a fox"                    # tag stripped from the prompt
    assert p["negativeprompt"] == "blurry"           # and from the negative
    assert "<lora:" not in p["prompt"]
    assert p["loras"] == "add-detail,bad-hands"      # comma-joined, prompt then neg
    assert p["loraweights"] == "0.8,1"
    lora_models = [m for m in blob["sui_models"] if m["param"] == "loras"]
    assert [m["name"] for m in lora_models] == ["add-detail", "bad-hands"]  # not on disk: bare name
    assert all(m["hash"] is None for m in lora_models)


def test_swarmui_loras_roundtrip_into_prompt():
    """loras/loraweights rebuild the <lora:…> tags in the prompt on read (A1111 parity)."""
    gen = {**_BASE_GEN, "prompt": "a fox <lora:add-detail:0.8>"}
    params = md.format_swarmui_metadata(gen, _StubEngine())
    fields = md.workspace_fields(md.parse_metadata(params))
    assert fields["prompt"] == "a fox <lora:add-detail:0.8>"


def test_swarmui_no_loras_omits_keys():
    """A generation with no LoRAs writes neither loras nor loraweights."""
    import json
    p = json.loads(md.format_swarmui_metadata(_BASE_GEN, _StubEngine()))["sui_image_params"]
    assert "loras" not in p and "loraweights" not in p


def test_swarmui_detailer_and_upscale_roundtrip():
    """The detailer stack and upscale knobs survive the SwarmUI path too."""
    detailer = {
        "models": [{"model": "face_yolov8n.pt", "prompt": "a face, sharp"}],
        "neg": "blurry", "confidence": 0.3, "strength": 0.4,
        "dilation": 4, "padding": 32, "blur": 4, "maxDet": 0,
    }
    upscale = {
        "scale": 4.0, "tile": 1024, "overlap": 128, "denoise": 0.25,
        "teacache": 0.1, "base": "4x-UltraSharp.pth", "prompt": "a fox, sharp",
    }
    fields = _roundtrip_swarm(_BASE_GEN, detailer=detailer, upscale=upscale)
    det = fields["detailer"]
    assert det["models"][0]["model"] == "face_yolov8n.pt"
    assert det["models"][0]["prompt"] == "a face, sharp"
    assert det["strength"] == 0.4
    up = fields["upscale"]
    assert up["scale"] == 4.0
    assert up["base"] == "4x-UltraSharp.pth"
    assert up["prompt"] == "a fox, sharp"                   # comma survives quote/unquote


# ── workspace_fields returns only keys that were present ──────────────

def test_workspace_fields_omits_absent_seed():
    """No Seed once yielded ``seed: -1``, which overwrote the form as "random"."""
    assert "seed" not in md.workspace_fields({"prompt": "a fox"})
    assert "seed" not in md.workspace_fields({"seed": "not-a-number"})
    assert md.workspace_fields({"seed": "1234"})["seed"] == 1234


# ── an explicit seed overrides engine.last_seed ──────────────────────

def test_format_metadata_seed_override():
    """A standalone upscale must not inherit the previous generation's seed."""
    params = md.format_metadata(_BASE_GEN, _StubEngine(), seed=777)
    assert "Seed: 777" in params
    assert md.workspace_fields(md.parse_metadata(params))["seed"] == 777
    # Without the override the engine's seed is still used.
    assert f"Seed: {_StubEngine.last_seed}" in md.format_metadata(_BASE_GEN, _StubEngine())


def test_format_swarmui_metadata_seed_override():
    import json
    blob = json.loads(md.format_swarmui_metadata(_BASE_GEN, _StubEngine(), seed=777))
    assert blob["sui_image_params"]["seed"] == 777


# ── gallery NSFW rating (prompt-derived, drives the blur) ──────────

def test_prompt_rating_tiers_and_false_positive_guards():
    # Safe prompts stay below R, including substrings that trip naive matching
    # ("ass" in "badass", "anal" in "analog", "sex" in "sextant").
    for p in ("portrait of a woman", "woman in bikini on the beach",
              "a badass knight", "analog photography", "a sextant at sea",
              "18-year-old student in class"):
        assert md.prompt_rating(p) == "PG"
    # R and up are blur-worthy.
    for p in ("nude study", "topless portrait", "lingerie photoshoot",
              "sexy pinup"):
        assert md.prompt_is_nsfw(p)
    assert md.prompt_rating("lingerie photoshoot") == "R"
    assert md.prompt_rating("nsfw, explicit scene") == "X"
    assert md.prompt_rating("hentai ahegao") == "X"
    # XXX wins over X when both are present (most-restrictive tier).
    assert md.prompt_rating("porn with gore") == "XXX"
    # Underscore-glued tags still count.
    assert md.prompt_rating("<lora:nsfw_v2:1.0> woman in underwear") == "X"
    assert md.prompt_rating("nsfw_art, portrait") == "X"


def test_prompt_rating_stems_match_word_forms():
    # Stem entries ("masturbat-") must fire on their inflections.
    assert md.prompt_rating("masturbating") == "X"
    assert md.prompt_rating("ejaculation") == "X"
    assert md.prompt_rating("penetrating") == "X"
    assert md.prompt_rating("seductive pose") == "R"
    assert md.prompt_rating("necrophilia") == "XXX"
    assert md.prompt_rating("zoophilia") == "XXX"
    # "18+" ends in punctuation, so it takes no trailing \b.
    assert md.prompt_rating("a photo, 18+") == "X"
    # Stems still can't swallow unrelated words that merely start the same way.
    for p in ("assassin creed", "classic car", "cumulus clouds",
              "a scatter plot", "glorious sunset", "analog photography"):
        assert md.prompt_rating(p) == "PG", p


def test_prompt_rating_ignores_negative_prompt():
    # Someone banning "nude" in the negative isn't generating nudity.
    assert md.prompt_rating("a cat") == "PG"
    assert md.prompt_rating("a cat\nNegative prompt: nude, gore") == "PG"
