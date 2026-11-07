# Z-Image (Turbo + Base): implementation plan

> Status: **Phase 1 and 3 (engine) done 2026-11-07, uncommitted; gate passed.**
> **Phase 2 (UI) done 2026-11-07, uncommitted:** see "Phase 2" below for what
> shipped and the offload measurements. MOSS-Image (`docs/moss-image-plan.md`) was the
> template: same phase shape, same low-RAM machinery, same conventions.
> Results of the real-weight gate are in "Gate results" below. Harness:
> `diffucore/tests/_zimage_parity.py`.

## Goal

One engine family, `z_image`, that loads the ComfyUI repackaging (bf16 and
int8_convrot) and the official diffusers folders for Z-Image-Turbo (8-step
distilled, no CFG) and Z-Image (Base, full CFG with a negative prompt) from
Tongyi-MAI (Apache-2.0), and runs on the dev PC (RTX 2060 12 GB, **23 GB system
RAM**) and natively on 24 GB+ cards. Text-to-image, img2img, inpaint, LoRA.

Success = (1) our DiT and schedule match the diffusers code exactly in fp32
(tiny config) and closely on real weights; (2) Turbo and Base render coherent
images end to end on the 2060 with the INT8 files; (3) img2img / inpaint / LoRA
work. Out of scope: Z-Image-Omni (SigLIP; rejected with a clear error), the
ControlNet pipelines, Z-Image-Edit.

## Verified facts

Read on 2026-11-07 from the diffusers 0.42.1 source (`transformer_z_image.py`,
`pipeline_z_image.py`, `_img2img.py`, `_inpaint.py`), the `Tongyi-MAI/Z-Image` and
`Z-Image-Turbo` repos (configs, scheduler configs, tokenizer, READMEs) and the
`Comfy-Org/Z-Image-Turbo` / `Comfy-Org/Z-Image` safetensors headers.

**Architecture** (`ZImageTransformer2DModel`, ~6.15B params). Config (identical
for Turbo and Base): dim 3840, 30 layers, 2 noise-refiner + 2 context-refiner
blocks, 30 heads of 128 (`n_kv_heads` = 30, no GQA), `norm_eps` 1e-5, q/k RMSNorm,
`cap_feat_dim` 2560, `in_channels` 16, patch 2 / f_patch 1, RoPE theta 256 with
axes (32, 48, 48) (sum = head_dim 128), `t_scale` 1000, FFN hidden
`int(dim / 3 * 8)` = 10240.
- Image latent `[16, 1, H/8, W/8]` -> 2x2 patches in the order `(h, w, ph, pw,
  C)` -> `x_embedder` Linear(64, 3840) with bias. Unpatchify is the inverse.
- Image tokens are padded to a multiple of 32 and caption tokens too (the
  caption is the *unpadded* hidden states of the attended tokens). Pad
  positions are replaced after embedding by the learned `x_pad_token` /
  `cap_pad_token` (`[1, 3840]`) and **take part in attention** (no mask in the
  single-prompt case).
- Timestep: `t_embedder(t * 1000)`, sinusoid `cos | sin` of dim 256 in fp32
  (`max_period` 1e4) -> Linear(256, 1024) SiLU Linear(1024, 256) = `adaln_input`
  (256 wide). The model's `t` is **`1 - sigma`**; its output is **minus** the
  velocity (the pipeline does `noise_pred = -out`).
- Block: `x += tanh(gate_msa) * RMSNorm2(attn(RMSNorm1(x) * (1 + scale_msa)))`,
  then the same with the SwiGLU FFN and `(scale_mlp, gate_mlp)`. The four
  vectors are `chunk(4)` of `adaLN_modulation = Linear(256, 4 * 3840)` of
  `adaln_input` (order `scale_msa, gate_msa, scale_mlp, gate_mlp`). Every block
  has four RMSNorms (`attention_norm1/2`, `ffn_norm1/2`, eps 1e-5, plain
  `weight`). Attention: fused or split q/k/v (bias-free), per-head RMSNorm
  (eps 1e-5) on q and k, RoPE, SDPA, `out`. Context-refiner blocks have no
  `adaLN_modulation` and no gating or scale.
- Flow: `x_embedder` -> pad -> 2 noise-refiner blocks (modulated) on the image
  tokens; `cap_embedder` (RMSNorm(2560) + Linear(2560, 3840)) -> pad -> 2
  context-refiner blocks on the caption; **joint sequence `[image, caption]`**,
  30 modulated blocks, `final_layer` = `LayerNorm(no affine, eps 1e-6) * (1 +
  Linear(SiLU(adaln)))` -> `Linear(3840, 64)` on the image tokens.
- **RoPE positions** (3 axes: seq, h, w): caption token i at `(1 + i, 0, 0)` for
  i over the *padded* caption length L (pad tokens included), image token
  `(r, c)` at `(L + 1, r, c)`, image pad tokens at `(0, 0, 0)`. The rotation
  pairs adjacent channels (complex multiply of `(x[2k], x[2k+1])`) in fp32.
  Angles are `pos * theta^(-2i/d)` formed in fp64 and rounded to fp32; image
  RoPE depends on the (padded) prompt length.
- The context refiner depends only on the caption, so it runs once per prompt.

**Checkpoint naming** (Comfy bf16 header, 453 keys, all bf16, 12.31 GB;
int8_convrot Turbo file 857 keys, 6.20 GB; the Base int8 header is
identical key for key):
- Original layout: `x_embedder.{weight,bias}`, `final_layer.linear`,
  `final_layer.adaLN_modulation.1`, `t_embedder.mlp.{0,2}`, `cap_embedder.{0,1}`,
  `cap_pad_token`, `x_pad_token`, `layers.N` / `noise_refiner.N` /
  `context_refiner.N` with **fused `attention.qkv`** `[11520, 3840]` (rows q, k, v),
  `attention.out`, `attention.q_norm/k_norm`, `feed_forward.w1/w2/w3`,
  `attention_norm1/2`, `ffn_norm1/2`, `adaLN_modulation.0`. Our module names
  are exactly these, so the Comfy files load strictly.
- diffusers layout (521 keys): `all_x_embedder.2-1`, `all_final_layer.2-1.*`,
  separate `attention.to_q/to_k/to_v`, `attention.to_out.0`,
  `attention.norm_q/norm_k`, otherwise the same. `from_diffusers` renames and
  fuses (checked against the vendored reference's own key set, which is
  diffusers' bit for bit).
- int8_convrot: weight `I8` + `weight_scale` `[out, 1]` fp32 + `comfy_quant` tag
  (`int8_tensorwise`, ConvRot group 256) on `attention.qkv/out`,
  `feed_forward.w1/w2/w3` of all 34 blocks **and the per-block
  `adaLN_modulation.0`** (30 + 2); everything else fp32.
  The existing `models/_int8.py` path reads it unchanged.

**Text encoder**: Qwen3-4B (`Qwen3Model`, 36 layers, hidden 2560, 32 heads / 8 KV
heads of 128, vocab 151936, rope 1e6, tied head), the *plain* Qwen3-4B, not
VL. The conditioning is **`hidden_states[-2]`** = the output of layer 35 of 36
(no final norm); the last layer never runs. Comfy's `qwen_3_4b.safetensors`
is the HF layout (`model.*`, bf16, no lm_head). Features are the unpadded
attended tokens; the pipeline pads to 512 on the right with a mask, which for a
causal model changes nothing. Chat template (`enable_thinking=True`, no system
turn, content verbatim, assistant turn left open):
`<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`, tokenized
whole (truncated to 512 from the right). The official `tokenizer.json` is byte
identical to our vendored Qwen3 one.

**VAE**: the FLUX.1 `AutoencoderKL` (`ae.safetensors`, 16 latent channels, 8x),
`scaling_factor` 0.3611, `shift_factor` 0.1159: decode is `z / 0.3611 +
0.1159`. No quant convs. Already supported (`bundle._build_flux_vae`).

**Scheduler and sampling** (`FlowMatchEulerDiscreteScheduler`,
`use_dynamic_shifting` false, so `mu` is unused):
- Sigmas: `s = linspace(1, 1/N, N)`, `sigma = shift * s / (1 + (shift - 1) * s)`,
  terminal 0 appended. This is exactly our `flow_matching_schedule(N, shift)`.
  **Shift is 3.0 for Turbo, 6.0 for Base** (each repo's `scheduler_config.json`).
  ComfyUI's templates use `ModelSamplingAuraFlow 3` for both.
- Euler: `x += (sigma_next - sigma) * (-model_out)`; N steps = N forwards
  (8 steps = 8 forwards; the Turbo model card's "9 steps = 8 forwards" is the
  older pipeline, not this one).
- Turbo: 8 steps, `guidance_scale` 0 (no CFG, no negative prompt). Base: 28 to
  50 steps, README example 50 steps, guidance 4 (range 3 to 5), `cfg_normalization`
  False, negative prompt optional (`""` by default).
- **Guidance is `pos + g * (pos - neg)`**, i.e. the usual CFG scale g + 1. Our
  `cfg_scale` is the usual scale, so the Base default is **5.0** (= the README's
  g = 4). ComfyUI's Base template: 25 steps, cfg 4, `res_multistep` + `simple`.
- Optional in the reference pipeline: `cfg_normalization` c > 0 caps
  `||pred||` at `c * ||pos||`; `cfg_truncation` T < 1 turns CFG off once
  `1 - sigma > T` (the same thing as our guidance interval).
- Latents are fp32, cond and uncond are batched in the reference (padding
  per item with an attention mask); we run two sequential unpadded forwards.
- img2img (`strength`) starts at `sigma0` of the truncated schedule from
  `(1 - sigma0) * z0 + sigma0 * eps`, z0 = the VAE mean (argmax) scaled
  `(z - shift) * scale`; the repo's convention (ComfyUI denoise) is kept here.

## Design decisions

1. **Names follow the original / Comfy layout**, one family `z_image`, module
   tree identical to the Comfy keys (fused qkv). diffusers folders load via
   `from_diffusers` (rename + fuse), exactly like MOSS.
2. **Variant = preset**, like Krea 2: `spec.variant` in {"turbo", "base"}, read
   off the DiT file name (`turbo` in the path => turbo, else base; a diffusers
   folder's whole path is read), overridable with `variant=`. It only picks
   defaults (steps, cfg, shift). The weights are the same network.
3. **`ModelSpec`**: `architecture="z_image"`, flow, 16 latent channels, context
   2560, `latent_scale 0.3611`, `latent_shift 0.1159` (the FLUX.1 VAE path:
   `_build_flux_vae` already reads those off the spec). Detection fingerprint:
   `cap_pad_token` + `x_pad_token`, any prefix.
4. **Sampling**: `zimage` scheduler = the variant's own shift; `flow` takes a
   shift; the other flow schedulers and all Anima samplers are available.
   `cfg_scale` is the usual scale (reference g + 1); `cfg_scale == 1` runs no
   uncond forward (Turbo default); `cfg_normalization` and the guidance
   interval mirror the reference's options.
5. **Prompt path**: TE once per prompt (inside a `staged` block), then the
   DiT's caption path (context refiner) once, both cached in `cond_cache`.
6. **Low-RAM / low-VRAM**: copy Krea 2's machinery: TE parked on the file's mmap
   with layers streamed (`masters="file"`), DiT blocks resident as far as VRAM
   allows and the rest streamed from the file. The INT8 DiT (6.2 GB) fits the
   2060 whole; the bf16 file does not and streams.
7. **LoRA** as forward hooks (the weights may sit on a read-only mmap): reuses
   the MOSS hook machinery; q/k/v LoRAs add into the row thirds of the fused
   `attention.qkv`. Real community LoRAs (checked: pixel-art, DeJPEG, realism,
   classic painting, coloring book) are `diffusion_model.layers.N.attention.
   to_{q,k,v}` / `to_out.0` / `feed_forward.w1-3` / `adaLN_modulation.0`,
   bf16, no alpha tensor (alpha = rank), 240 modules each, only the 30 main
   layers.
8. **fp16** (the 2060 has no fast bf16 SDPA): the bf16-trained weights overflow
   fp16 inside both branches (see "fp16 range"), fixed by an exact fold: the
   attention input and the feed-forward `w3` output are carried at 1/256 and
   1/4096 (`ZImageDiT.attn_scale` / `ffn_scale`, set by the loader for fp16
   only), and the RMSNorm after each branch uses `eps * scale^2`. The stream
   itself needs no scaling (peak ~7e3). Unlike MOSS / Krea 2 there is no
   residual-stream scale.
9. **Not ported**: batched CFG with per-item masks (two sequential forwards,
   as for MOSS), Omni / SigLIP, ControlNet, the reference's gradient
   checkpointing and attention-backend dispatch.

## Phase 1: engine, text-to-image (submodule `diffucore/`)

Files (all in `diffucore/`):
- `src/diffucore/models/z_image_dit.py`: `ZImageConfig.from_state_dict`,
  `ZImageDiT` (`encode_text`, `forward`), `int8_layer_names`,
  `from_diffusers`, `from_diffusers_name`.
- `src/diffucore/conditioning/z_image_tokenizer.py`: `ZImageTokenizer`.
- `src/diffucore/bundle.py`: `load_zimage_checkpoint`, `load_zimage_dit`,
  `_zimage_dit_state`, `_zimage_text_encoder`, `_zimage_variant`.
- `src/diffucore/loading/detect.py`: the `z_image` fingerprint.
- `src/diffucore/pipelines/_zimage.py` + dispatch in `text_to_image.py`.
- `NOTICE`.
- Tests: `tests/test_z_image.py`, `tests/_zimage_reference.py`,
  `tests/_zimage_parity.py` (GPU harness), fixtures `z_image_headers.json`,
  `z_image_schedule.json`, `z_image_tokenizer_cases.json`.

## Phase 2: UI + the 2060 path (`backend/`, `static/`): done 2026-11-07

Shipped (uncommitted):
- **Picker / arch**: `model_arch` tags the DiT `z_image` (engine fingerprint) and
  the TE `qwen3_4b` (plain Qwen3, hidden 2560). Krea 2's Qwen3-VL-4B has the same
  width but nests its LM under `language_model.`, so it stays `qwen3vl`; before this
  change the plain 4B was lumped into `flux2_lm`. The Z-Image companions are VAE
  `flux1` + TE `qwen3_4b`; FLUX.2 keeps working because its companion list is
  now `flux2_lm` + `qwen3_4b` (klein 4B uses the same file).
- **Loading**: model type `ZImage`, label `ZImage(<dit>)`, `Engine.load_zimage`
  (variant from the file name, overridable in the rack; compile / int8 / attention /
  fp16 acc / vae fp16 like Krea 2; no CUDA Graphs or fused glue). Default offload
  `stream`.
- **Defaults** (variant select next to the files): Turbo 8 steps, CFG 1, shift 3;
  Base 50 steps, CFG 5, shift 6; euler + `zimage` scheduler. The negative prompt
  box is hidden while CFG is 1 (Turbo's setting) and shows once CFG > 1. The shift
  slider is ignored by `zimage` / `flow_dyn`. Returning to a family restores the
  Generate knobs and VAE / TE it was left with (all families).
- **Generation**: txt2img, img2img, inpaint (img2img sizes snap to the 16 px grid and map
  back), detailer, tiled upscaler, `<lora:...>` tags, XYZ (applies the Settings
  knobs through `_settings_knobs` like Anima / Krea 2 / MOSS; the Checkpoint axis
  lists Z-Image DiTs), TeaCache / DeepCache knobs dropped from metadata, unknown
  schedulers coerced to `zimage`.
- **Metadata**: `Z Image variant: turbo|base` (A1111), `z_image_variant` (SwarmUI);
  model hash unwraps `ZImage(...)`.
- **Live preview**: ComfyUI's FLUX.1 latent RGB factors on the 16-ch sampling-space
  x0.
- Tests: `backend/test_zimage.py`, `backend/test_model_arch.py`.

**Offload measurement** (RTX 2060 12 GB, Turbo INT8 file, 1024^2, 8 steps,
fp16 acc + fp16 VAE, through the server, sampling 1.83 to 1.91 s/step in every mode):

| offload | DiT | per image, cached prompt | per image, new prompt | load |
|---|---|---|---|---|
| stream, pinned masters (RAM free) | all 30 blocks streamed with prefetch | 16.3 s | not measured | 84 s (cold TE) |
| stream, file masters (RAM tight) | 30 of 30 blocks resident, TE layers streamed with prefetch | n/a | 17.2 to 17.9 s | 34 s |
| encoders | resident, TE streamed from the mmap without prefetch | 16.2 s | 62 / 46 / 31 s (page cache warming) | 27 s |

Resident vs streamed makes no per-step difference (INT8 GEMMs at 4096 tokens are
compute bound, prefetch hides the copies), and `stream` already keeps every block
that fits resident under file masters, so `stream` stays the default for both files.
`encoders` loses 15 to 45 s on every new prompt because the file-backed TE has no
prefetch outside `stream`. The bf16 file (12.3 GB) streams: 13 of 30 blocks resident,
22.8 s per image warm (165 s for the first image, a cold 12 GB read).

## Phase 3: img2img, inpaint, LoRA, folders (engine, done)

- `zimage_img2img` (img2img + latent-mask inpaint, VAE-mean encode), dispatched
  from `ImageToImage` / `Inpaint`.
- LoRA: `lora.py::_zimage_targets` / `_zimage_attach`.
- diffusers `transformer/` folder (as the folder, the repo root or an index
  json) and `text_encoder/` folder load.

## Gate results (2026-11-07, RTX 2060 12 GB, real weights)

**TE parity** (Qwen3-4B, `hidden_states[-2]`): ours vs transformers 5.9
`Qwen3Model`. fp32 CPU, first 5 layers: rel L2 1.5e-7 to 2.4e-7 per layer.
All 35 layers, ours fp16 (streamed from the file) vs transformers bf16 on the
GPU: rel L2 0.0090, cos 1.0000 (dominated by the massive-activation token at
position 0, |x| 13760 in both; features finite in fp16).

**DiT parity** at 512^2 (latent 64^2, seed 0), same TE features: the vendored
reference (diffusers' code, bit-exact to it on a tiny config) in fp32 on the CPU,
blocks streamed from the bf16 file, vs ours on the GPU:

| run | fp16 (bf16 file) rel L2 / cos | INT8 file (Turbo) rel L2 / cos | bf16 file, int8 on load rel L2 / cos |
|---|---|---|---|
| sigma 1.0 | 0.0030 / 0.999994 | 0.0244 / 0.999706 | 0.0292 / 0.999573 |
| sigma 0.7 | 0.0025 / 0.999995 | 0.0231 / 0.999732 | 0.0223 / 0.999750 |
| sigma 0.3 | 0.0017 / 0.999997 | 0.0180 / 0.999837 | 0.0179 / 0.999838 |
| sigma 0.05 | 0.0017 / 0.999998 | 0.0174 / 0.999849 | 0.0174 / 0.999848 |
| empty prompt, 0.7 | 0.0024 / 0.999996 | 0.0225 / 0.999745 | 0.0235 / 0.999723 |

Targets (fp16 rel L2 < 3e-2, cos >= 0.998, finite) met everywhere; int8 is at
1.7 to 2.9 % with cos >= 0.9996. No inf / nan.

**fp16 range** (1024^2, Turbo, Euler 8 steps from noise; real latent
`x0 + noise` at sigma 1 / 0.7 / 0.3 / 0.05; bf16 and INT8 files). Without the
fold, fp16 produces NaN in `noise_refiner.0`'s attention output at every
sigma (and its feed-forward from sigma 0.3) and the very first forward is
NaN. True magnitudes (bf16 compute at 768^2): Turbo attention output 1.4e5,
feed-forward output 2.5e5 (1e6 later in the run); **Base** is far worse:
qkv 1.3e6, attention output 3e6, feed-forward hidden 7e6, output 6e7. The
stream peaks at only 7e3 (Base 1.4e4). With attn / ffn scale 1/256 and 1/4096
the largest value anywhere is the stream (7.4e3), the next 540; Base's 50-step
CFG 5 Euler loop stays finite at every step (largest non-stream value 1.6e4).
A first guess of a single 1/128 fold fixed Turbo but Base NaN'd, hence the
two scales.

**End to end** (Turbo INT8 file, Base INT8 file, FLUX.1 VAE, offload stream with
prefetch, 1024^2, seeds 11 to 14; pinned masters since the RAM allowed):

| | load | per image | VRAM alloc peak | RSS peak |
|---|---|---|---|---|
| Turbo, 8 steps, no CFG | 97 s cold (TE page cache) | 22 to 26 s (first image 39 s: kernel build) | 8.1 GB | 14.2 GB |
| Base, 50 steps, CFG 5 | 41 s | 281 to 300 s (3.8 s/step = 2 x 1.9 s) | 7.3 GB | 14.2 GB |
| img2img 0.6 / inpaint (Turbo) | | 23 s / 29 s | | |
| Turbo, bf16 file (fp16 compute), 13 of 30 blocks resident, rest from the mmap | 48 s | 92 s | 7.2 GB (smi 11.8 GB) | 14.5 GB |
| Turbo, bf16 file + int8 on load, 27 of 30 resident | 61 s | 65 s (incl. fused-GEMM build) | 8.7 GB | 16.6 GB |

Images (landscape, still life, cafe, text poster) are coherent for both;
the Turbo poster renders "VISIT THE COAST" correctly (plus a repeated "COAST").
The inpaint keep region is pixel-identical; the composite shows the usual
hard-mask seam. A real community LoRA (`tarn59/pixel_art_style_lora_z_image_turbo`,
r32): 240 modules hooked, 0 unmapped, applied in 0.4 s, visibly pixel-art at
1.0, `clear_loras` restores the no-LoRA image bit for bit.

## Deviations from the reference, open issues

- Two sequential unpadded forwards for CFG instead of the reference's padded
  batch of 2 with an attention mask; the DiT oracle test covers per-item padding
  and a batch of equal-length prompts, not unequal-length batches (not needed).
- `cfg_scale` is the usual scale (reference g + 1); Base default 5.0 = README g 4.
- img2img / inpaint use the repo's ComfyUI-style denoise (schedule of
  `steps / strength` steps, last `steps` run) and x0-pinned inpaint, not
  diffusers' A1111-style slice or per-step re-noised blend.
- fp16 uses the exact branch-scale fold above; bf16 / fp32 compute run
  unscaled. INT8 W8A8 error (1.7 to 2.9 % rel L2 per forward) is the existing
  ConvRot path's; images look right.
- Turbo default is 8 steps (this diffusers pipeline: N steps = N forwards); the
  model card's "9 steps = 8 forwards" refers to an older pipeline. Not A/B'd
  against 9.
- Not verified: Z-Image Omni (rejected), a diffusers folder on real files (only
  synthetic shards plus the header layout), a 4 GB card, Windows, `compile`,
  `int8_turing` / FA2 attention backends (the slots exist on every attention
  module; the fp16 SDPA path is what was run), TeaCache-style caching, Base
  with `cfg_normalization` on real files, a Turbo LoRA on Base.
- With int8 on load of the bf16 file the first image pays the fused-GEMM JIT
  build; the INT8 file path pays it once per process too.
- Offload: resolved in Phase 2 above (`stream` by default; resident is no faster).
