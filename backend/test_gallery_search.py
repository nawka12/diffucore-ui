"""Tests for the gallery search query language and filters (pure functions).

    .venv/bin/python -m pytest backend/test_gallery_search.py -v
"""

from __future__ import annotations

import gallery_search as gs


def _entry(name, **kw):
    e = {"name": name, "prompt": "", "neg": "", "model": "", "sampler": "",
         "scheduler": "", "seed": "", "steps": "", "cfg": "", "size": "",
         "loras": [], "rating": "PG", "date": "2026-09-01"}
    e.update(kw)
    return e


ENTRIES = [
    _entry("a", prompt="1girl, red hair, sword <lora:inkwash:0.8>", neg="bad hands",
           model="AnimaPulse-1.1.safetensors", sampler="euler_ancestral",
           scheduler="beta", seed="42", steps="30", cfg="4.5", size="1024x1024",
           loras=["inkwash"], rating="PG", date="2026-09-01"),
    _entry("b", prompt="1girl, redhead, cat ears", neg="worst quality",
           model="AnimaFranken-v1.2.safetensors", sampler="euler",
           scheduler="beta_mix", seed="7", steps="24", cfg="4", size="832x1216",
           rating="R", date="2026-09-10"),
    _entry("c", prompt="landscape, mountains, sword in stone", neg="bad hands, 1girl",
           model="AnimaPulse-1.1.safetensors", sampler="cogent3_pump",
           scheduler="pump_taper", seed="42", steps="30", cfg="4.50", size="1024x1024",
           rating="X", date="2026-09-20"),
]


def _names(query="", **filters):
    return [e["name"] for e in gs.search(ENTRIES, query, filters)]


def test_empty_query_returns_everything_in_order():
    assert _names() == ["a", "b", "c"]


def test_words_are_anded_across_commas_and_spaces():
    assert _names("1girl sword") == ["a"]
    assert _names("1girl, sword") == ["a"]


def test_bare_word_is_a_substring():
    assert _names("red") == ["a", "b"]


def test_quoted_phrase_matches_whole_words_only():
    assert _names('"red"') == ["a"]
    assert _names('"red hair"') == ["a"]
    assert _names('"sword in stone"') == ["c"]


def test_unclosed_quote_still_filters_while_typing():
    assert _names('"red ha') == []
    assert _names('"red hair') == ["a"]


def test_minus_excludes():
    assert _names("sword -landscape") == ["a"]
    assert _names('1girl -"cat ears"') == ["a"]


def test_default_search_skips_the_negative_prompt():
    assert _names("hands") == []
    assert _names("1girl") == ["a", "b"]
    assert _names("neg:hands") == ["a", "c"]
    assert _names("negative:1girl") == ["c"]


def test_scoped_text_fields():
    assert _names("model:franken") == ["b"]
    assert _names("sampler:euler") == ["a", "b"]
    assert _names('sampler:"euler"') == ["b"]
    assert _names("scheduler:pump") == ["c"]
    assert _names("prompt:sword") == ["a", "c"]


def test_lora_field_searches_lora_names():
    assert _names("lora:ink") == ["a"]
    assert _names("-lora:ink") == ["b", "c"]


def test_exact_fields():
    assert _names("seed:42") == ["a", "c"]
    assert _names("seed:4") == []
    assert _names("steps:24") == ["b"]
    assert _names("size:1024x1024") == ["a", "c"]
    assert _names("cfg:4.5") == ["a", "c"]
    assert _names("cfg:4") == ["b"]


def test_unknown_field_prefix_is_plain_text():
    terms = gs.parse_query("score:9 -foo:bar")
    assert [(t["field"], t["value"], t["exclude"]) for t in terms] == [
        (None, "score:9", False), (None, "foo:bar", True)]


def test_lora_tag_typed_in_the_box_is_plain_text():
    assert _names("<lora:inkwash") == ["a"]


def test_field_names_are_case_insensitive():
    assert _names("MODEL:Franken") == ["b"]


def test_panel_filters_are_exact():
    assert _names(sampler="euler") == ["b"]
    assert _names(model="AnimaPulse-1.1.safetensors") == ["a", "c"]
    assert _names(lora="inkwash") == ["a"]
    assert _names(size="832x1216") == ["b"]
    assert _names(seed="42") == ["a", "c"]


def test_rating_and_date_filters():
    assert _names(ratings={"R", "X"}) == ["b", "c"]
    assert _names(date_from="2026-09-10") == ["b", "c"]
    assert _names(date_to="2026-09-10") == ["a", "b"]
    assert _names(date_from="2026-09-05", date_to="2026-09-15") == ["b"]


def test_query_and_filters_combine():
    assert _names("sword", model="AnimaPulse-1.1.safetensors", ratings={"X"}) == ["c"]


def test_facets_count_and_order_by_use():
    f = gs.facets(ENTRIES)
    assert f["models"] == [["AnimaPulse-1.1.safetensors", 2],
                           ["AnimaFranken-v1.2.safetensors", 1]]
    assert f["loras"] == [["inkwash", 1]]
    assert f["sizes"][0] == ["1024x1024", 2]
