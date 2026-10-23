"""Header-only file classification for the model rack, on synthetic headers.

    .venv/bin/python -m pytest backend/test_model_arch.py -v
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

import engine
import model_arch as ma


def _write(path, shapes):
    header = {k: {"dtype": "BF16", "shape": list(s), "data_offsets": [0, 0]}
              for k, s in shapes.items()}
    header["__metadata__"] = {"format": "pt"}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)
    return path.name


SDXL = {"model.diffusion_model.input_blocks.0.0.weight": (320, 4, 3, 3),
        "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight": (640, 2048)}
ANIMA = {"net.llm_adapter.blocks.0.cross_attn.q_proj.weight": (1024, 1024)}
KREA2_DIFFUSERS = {"img_in.weight": (3072, 64), "text_fusion.projector.weight": (3072, 2560)}
MOSS_DIT = {"shared_block_modulation.weight": (18432, 3072), "video_head_modulation.weight": (6144, 3072),
            "video_embedder.weight": (3072, 128), "text_embedder.proj.weight": (3072, 4096)}
FLUX1 = {"double_blocks.0.img_attn.qkv.weight": (9216, 3072), "img_in.weight": (3072, 64),
         "txt_in.weight": (3072, 4096)}
FLUX2 = {**FLUX1, "double_stream_modulation_img.lin.weight": (18432, 3072),
         "img_in.weight": (3072, 128)}


@pytest.mark.parametrize("shapes, arch", [
    (SDXL, "sdxl"), (ANIMA, "anima"), (KREA2_DIFFUSERS, "krea2"), (MOSS_DIT, "moss_image"),
    (FLUX1, "flux1"), (FLUX2, "flux2"), ({"foo.weight": (1,)}, None),
])
def test_model_arch(tmp_path, shapes, arch):
    name = _write(tmp_path / "m.safetensors", shapes)
    assert ma.kinds(tmp_path, [name], "model") == {name: arch}


@pytest.mark.parametrize("shapes, kind", [
    ({"conv1.weight": (32, 32, 1, 1, 1), "decoder.conv1.weight": (384, 16, 3, 3, 3)}, "qwen_image"),
    ({"decoder.conv_in.weight": (384, 16, 3, 3),
      "decoder.mid_block.attentions.0.to_qkv.weight": (1152, 384, 1, 1)}, "qwen2d"),
    ({"decoder.conv_in.weight": (512, 16, 3, 3), "decoder.mid.attn_1.q.weight": (512, 512, 1, 1)}, "flux1"),
    ({"decoder.conv_in.weight": (512, 32, 3, 3), "decoder.mid.attn_1.q.weight": (512, 512, 1, 1)}, "flux2"),
    ({"decoder.conv_in.weight": (512, 4, 3, 3), "decoder.mid.attn_1.q.weight": (512, 512, 1, 1)}, None),
    ({"decoder.conv_in.weight": (384, 32, 3, 3),
      "decoder.mid_block.attentions.0.to_q.weight": (384, 384)}, "flux2_small"),
    ({"decoder.conv_in.weight": (512, 32, 3, 3),
      "decoder.mid_block.attentions.0.to_q.weight": (512, 512)}, "flux2"),
    ({"decoder.conv_in.weight": (384, 16, 3, 3),
      "decoder.mid_block.attentions.0.to_q.weight": (384, 384)}, None),
])
def test_vae_kind(tmp_path, shapes, kind):
    name = _write(tmp_path / "v.safetensors", shapes)
    assert ma.kinds(tmp_path, [name], "vae") == {name: kind}


@pytest.mark.parametrize("shapes, kind", [
    ({"model.embed_tokens.weight": (151936, 1024),
      "model.layers.0.self_attn.q_norm.weight": (128,)}, "qwen3"),
    ({"model.language_model.embed_tokens.weight": (248320, 1024),
      "model.language_model.layers.0.linear_attn.A_log": (16,)}, "qwen35"),
    ({"model.language_model.embed_tokens.weight": (248320, 2560),
      "model.language_model.layers.0.linear_attn.A_log": (32,)}, "qwen35"),
    ({"model.language_model.embed_tokens.weight": (248320, 4096),
      "model.language_model.layers.0.linear_attn.A_log": (32,)}, "qwen35_9b"),
    ({"model.language_model.embed_tokens.weight": (151936, 2560),
      "model.language_model.layers.0.self_attn.q_norm.weight": (128,)}, "qwen3vl"),
    ({"model.embed_tokens.weight": (151936, 2560),
      "model.layers.0.self_attn.q_norm.weight": (128,)}, "flux2_lm"),
    ({"model.layers.0.self_attn.q_proj.weight": (4096, 5120)}, "flux2_lm"),
    ({"encoder.block.0.layer.0.SelfAttention.q.weight": (4096, 4096)}, "t5"),
    ({"text_model.embeddings.token_embedding.weight": (49408, 768)}, "clip_l"),
])
def test_te_kind(tmp_path, shapes, kind):
    name = _write(tmp_path / "t.safetensors", shapes)
    assert ma.kinds(tmp_path, [name], "te") == {name: kind}


_MODELS = Path(__file__).resolve().parent.parent / "models"


@pytest.mark.parametrize("folder, name, what, kind", [
    ("diffusion-models", "moss_image_sft_int8_convrot.safetensors", "model", "moss_image"),
    ("diffusion-models", "moss_image_sft_bf16.safetensors", "model", "moss_image"),
    ("text-encoders", "qwen3.5_9b_int8_convrot.safetensors", "te", "qwen35_9b"),
    ("text-encoders", "qwen35_4b.safetensors", "te", "qwen35"),
    ("vae", "flux2-vae_small.safetensors", "vae", "flux2_small"),
    ("vae", "flux2-vae.safetensors", "vae", "flux2"),
])
def test_real_moss_files_are_recognised(folder, name, what, kind):
    """Header-only reads of the real MOSS-Image downloads, when present."""
    if not (_MODELS / folder / name).is_file():
        pytest.skip(f"{folder}/{name} not downloaded")
    assert ma.kinds(_MODELS / folder, [name], what) == {name: kind}


def test_unreadable_files_are_unknown(tmp_path):
    (tmp_path / "old.ckpt").write_bytes(b"\x80\x02")
    (tmp_path / "broken.safetensors").write_bytes(b"\xff" * 16)
    assert ma.kinds(tmp_path, ["old.ckpt", "broken.safetensors", "gone.safetensors"], "model") == {
        "old.ckpt": None, "broken.safetensors": None, "gone.safetensors": None}


def test_rewritten_file_is_reclassified(tmp_path):
    name = _write(tmp_path / "m.safetensors", ANIMA)
    assert ma.kinds(tmp_path, [name], "model")[name] == "anima"
    _write(tmp_path / "m.safetensors", {**FLUX1, "padding.weight": (1, 2, 3)})
    assert ma.kinds(tmp_path, [name], "model")[name] == "flux1"


@pytest.mark.parametrize("names, groups", [
    (engine.SAMPLERS_SD + engine.SAMPLERS_ANIMA + engine.SAMPLERS_FLUX + engine.SAMPLERS_KREA2
     + engine.SAMPLERS_MOSS, engine.SAMPLER_GROUPS),
    (engine.SCHEDULERS_SD + engine.SCHEDULERS_ANIMA + engine.SCHEDULERS_FLUX + engine.SCHEDULERS_KREA2
     + engine.SCHEDULERS_MOSS, engine.SCHEDULER_GROUPS),
])
def test_every_option_has_one_dropdown_section(names, groups):
    grouped = [n for members in groups.values() for n in members]
    assert len(grouped) == len(set(grouped))
    assert set(names) == set(grouped)
