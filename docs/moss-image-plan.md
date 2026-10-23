# MOSS-Image (Base + SFT): implementation plan

> Status: **Phases 1 to 3 done 2026-10-23 (uncommitted).** Written for an
> executor agent. Krea 2 (`docs/krea2-plan.md`) is the template: same phase
> shape, same low-RAM machinery, same conventions.
>
> **Gate 1 results** (2060, real weights; harness `diffucore/tests/_moss_parity.py`):
> - Comfy `proj.qkv` rows = q, k, v, bit-exact vs the official shards.
> - DiT single forwards at 256² vs the reference in fp32 on CPU (streamed per
>   block): ours fp16 c=1/16 rel L2 0.21-0.44%, cos >= 0.99999; INT8 file
>   0.8-5.9% (worst at sigma 0.3, cos 0.9983).
> - fp16 headroom at 1024²: stream peak 61,024 at block 26 unscaled (fp16
>   max 65,504), 3,814 at c=1/16. Matches KBlueLeaf.
> - E2E INT8 DiT + INT8 TE + small VAE, defaults, 1024²: load 7.5 s, ~195
>   s/img (1.9 s/forward, sdpa), torch peak 8.4 GB, nvidia-smi 10.9 GB, RSS
>   9.9 GB. All 42 INT8 blocks resident. Chinese sticky-note text and "OPEN
>   LATE" render correctly. Full flux2-vae decode = 41.6 dB vs the small one.
> - Tokenizer: the vendored qwen35 json splits `[\p{L}\p{M}]+`, the official
>   9B one `\p{L}+` (vocab and merges identical); MossTokenizer swaps in the
>   official split, so Thai / Devanagari ids now match transformers.
> - TE vs HF transformers 5.9 `Qwen3_5TextModel`, fp32 CPU, first 4 layers
>   (3 Gated DeltaNet + 1 full attention, final norm) with the same weights
>   dequantized from the INT8 file: rel diff 5e-8 to 4e-7 per layer.
>
> **Phase 3 done 2026-10-23 (uncommitted)**: `moss_img2img` (img2img + soft
> token-grid latent-mask inpaint, VAE-mean encode = inverse of `_decode`),
> dispatched from `ImageToImage` / `Inpaint` (both take `uncond_skip_layers`);
> LoRA as forward-hook side branches (`lora.py::_moss_attach`: q/k/v LoRAs add
> into row thirds of `proj.qkv`, text-embedder targets, metadata / `.alpha`
> alpha, weight-shape-changing control LoRAs rejected, cond cache cleared on
> apply/remove/clear); the official `text_encoder/` folder / repo root / index
> json load; `load_moss_checkpoint` exported. UI gates for i2i, inpaint,
> detailer, upscaler and LoRA removed. img2img / inpaint / upscaler tiles snap
> up to ÷16 in `Engine._anima_gen_size` and map back (the UI rounds loaded
> images to ÷8). The WD tagger stays resident under MOSS (as Krea 2): 1024²
> peak 8.1 GiB with it (the VAE decode tiles) vs 10.5 GiB without, same speed.
>
> Gate 2/3 on the 2060 (INT8 DiT + INT8 TE + small VAE, server API):
> - 1024² defaults: ~164 s/img with sdpa, 137 s with `int8_turing` attention
>   (x1.20); metadata `Scheduler: moss`, `Uncond skip layers: 22`;
>   `uncond_skip_layers: []` changes the image and records `none`.
> - img2img 0.6 / 0.3, hard-mask inpaint (keep region pixel-identical),
>   detailer (face), tiled upscaler all run.
> - LoRAs: pixel-art (bare `layers.*`, metadata alpha) visibly applies; the
>   anime LoRA in `transformer.*` and `diffusion_model.*` formats gives
>   bit-identical images; 252 hooked, 0 unmapped each; `clear_loras` restores
>   the no-LoRA image exactly.
> - bf16 DiT file-backed: 31 of 42 blocks resident with on-load int8, 19
>   without; both run.
> - Not run on real files: kohya `lora_down/up/alpha` and `lora_unet_` forms,
>   the real 4-shard official `text_encoder/` folder (synthetic shards only).
>
> **Phase 2**: UI wired (model type "MOSS", picker group, companions,
> defaults, "Uncond skip blocks" field, metadata), 402 backend tests pass;
> i2i / inpaint / detailer / upscaler / LoRA were gated until Phase 3.
>
> **Few-step distill adapters are NOT plain LoRAs** (`lodestones/...-8-steps-dmd`,
> `darask0/MOSS-Image_4-step_adapter_lora`): they need the LoRA unmerged, a
> replacement fp32 time path (`time_embedding.*` + the two modulation biases,
> `time_embedding.safetensors`) and their own data-space sampler (`x = z +
> Q x0`, `x0 = z - v`, timestep = `step_codes[r]`, from `dataspace.safetensors`),
> no CFG. A plain LoRA + euler gives purple, soft images. Phase 4 item.

## Goal

One engine family, `moss_image`, that loads the official diffusers folders and
the Comfy-Org repackaging (bf16 and int8_convrot), for both MOSS-Image-Base and
MOSS-Image-SFT (same network, same sampling), and runs from the UI on the dev
PC (RTX 2060 12 GB, **23 GB system RAM**) and natively on 24 GB+ cards.

Success = (1) our DiT and scheduler match the reference code exactly in fp32
(tiny config) and closely on real weights; (2) t2i runs end to end from the UI
on the 2060 with the Comfy INT8 files; (3) img2img / inpaint / LoRA work.

## Verified facts

Read on 2026-10-23 from `OpenMOSS-Team/MOSS-Image-SFT` (configs,
`pipeline_moss_image.py`, README), `OpenMOSS/MOSS-Image` (comfyui plugin code,
README), `Comfy-Org/MOSS-Image` safetensors headers, and
`KBlueLeaf/MOSS-Image-FP16-fixed` README. Local copies of the configs, the
reference pipeline and the headers (JSON) are in the session scratchpad; the
executor should re-download what it needs (all public, Apache-2.0).

**License**: Apache-2.0 for weights and the reference pipeline code. Porting
the reference modules is fine; add a NOTICE entry. Do not copy ComfyUI code
(GPL); reading Comfy file *headers* / workflow JSON is data and fine.

**Base vs SFT**: same architecture, same pipeline, same defaults. Weights
cannot tell them apart and nothing needs to. No variant field.

**DiT** (`MOSSImageTransformer`, ~5.08B params, config: dim 3072, ffn 8192,
24 heads of 128, 42 layers, text_dim 4096, freq_dim 256, latent_channels 128,
rope_base 1e4). Patch size 1: the DiT takes the 4-D latent `[B, 128, H/16,
W/16]` directly, flattens to `H/16 * W/16` image tokens.
- `video_embedder`: Linear(128, 3072) with bias.
- `text_embedder`: `proj` Linear(4096, 3072) + 2 `_TextBlock`s (pre-norm,
  fused `qkv` Linear with bias, per-head q/k RMSNorm, `proj`, SITU FFN with
  `ffn_gate`/`ffn_up`/`ffn_down`, all with bias). Not modulated; uses the
  text part of the RoPE.
- `time_embedding`: sinusoid of `t` (t = sigma * 1000, the scheduler's
  timesteps) computed in **float64**: `freq = 10000^(-i/128)`, `cat(cos,
  sin)`, then fp32 into Linear(256,3072), SiLU, Linear(3072,3072).
- **One shared modulation for all blocks**: `shared_block_modulation`
  Linear(3072, 6*3072) -> `(shift_a, scale_a, gate_a, shift_f, scale_f,
  gate_f)`. Blocks own no modulation weights. Head: `video_head_modulation`
  Linear(3072, 2*3072) -> `(shift, scale)`.
- `_Block` (all Linear **bias-free**): `y = norm1(x)*(1+scale_a)+shift_a`;
  q/k/v projections, per-head RMSNorm on q and k, RoPE, SDPA (no mask), `o`;
  `x += o(...) * gate_a`; `y = norm2(x)*(1+scale_f)+shift_f`; `hidden, gate =
  ffn_up_gate(y).chunk(2)` (hidden first!); `x += ffn_down(situ(hidden,
  gate)) * gate_f`.
- `situ(h, g) = (4*tanh(g/4)*sigmoid(g)) * (25*tanh(h/25))`.
- RMSNorm: fp32 compute, eps 1e-6, plain `weight` (NOT `1+weight`), result
  cast back then multiplied by weight.
- Joint sequence order: **image tokens first, then text**. All 42 blocks see
  the whole joint sequence. Final: slice image tokens, `final_norm_video`,
  `*(1+scale)+shift`, `final_linear_video` (bias-free) -> 128 channels.
- **RoPE**: 4 axes (r, f, h, w), `num_bands = head_dim/8 = 16` per axis,
  `bands = 1/base^(arange(16)/16)` built in fp32 on CPU (the checkpoint's
  `rope.bands` key is ignored and re-derived). Image token coords: `(0, L, h -
  (H-1)/2, w - (W-1)/2)` where **L = the text length** (so image RoPE depends
  on the prompt length; cond and uncond differ). Text coords: `(0, i, 0, 0)`
  for i in 0..L-1. `phase = coords[..., None] * bands` -> `[S, 4, 16]`; the
  rope tensor is `cat(sin, cos)` over a new axis, flattened to 128 = `[sin(64)
  | cos(64)]`. `_rotate`: `sin, cos = rope.chunk(2)`; each duplicated
  `cat(s, s)` to 128; `x*cos + cat(-x2, x1)*sin` with `x1, x2 = x.chunk(2)`
  (rotate-half layout, the 64 frequencies ordered axis-major). Computed in
  fp32, cast to the activation dtype.
- Text masking: each sample runs at its **true text length, unpadded**; no
  attention mask anywhere.
- `skip_layers`: a set of block indices simply not executed.

**Checkpoint naming** (load-bearing):
- Official diffusers shards (`transformer/`, 3 files): `layers.N.proj.{q,k,v,o,
  ffn_up_gate,ffn_down}.weight`, plus `rope.bands`.
- **Comfy-Org files fuse q/k/v**: `layers.N.proj.qkv.weight` [9216, 3072]
  (rows q, k, v in that order; verify against the official shards). Same
  names otherwise. bf16 file 379 keys; int8_convrot file adds
  `weight_scale` [out,1] + `comfy_quant` (`{"format": "int8_tensorwise",
  "convrot": true, "convrot_groupsize": 256}`) on the 4 block linears, and
  `layers.N.comfy_attention.config` (`{"attention": "comfy_kitchen_int8"}`,
  an attention-backend hint: drop it).
- Text-embedder and non-block params stay bf16 in the int8 file.

**Text encoder**: Qwen3.5-9B (`Qwen3_5Model`, hybrid: 24 Gated DeltaNet + 8
gated full-attention layers at 3,7,...,31; hidden 4096, 16 q / 4 kv heads of
256, partial rotary 0.25, theta 1e7, ffn 12288, vocab 248320, every layer has
an MLP). The DiT reads `last_hidden_state` = **after the final norm**
(Qwen3.5 norms are `1 + weight`). Files:
- official `text_encoder/` 4 shards, keys `model.language_model.*`,
  `model.visual.*` (vision tower, drop), `mtp.*` (drop), `lm_head` (drop);
- Comfy `qwen3.5_9b_bf16` (19.3 GB, same keys incl. visual/mtp/lm_head),
  `qwen3.5_9b_int8_convrot` (9.7 GB: every big linear int8 convrot **and
  `embed_tokens` int8 convrot** with per-row scale), `w4a8_gptq` (skip).
- fp16 is safe (KBlueLeaf: fp16 embedding cosine 0.999996 vs fp32).

**Prompt template** (HF reference = `apply_chat_template([{role: user,
content: prompt}], add_generation_prompt=True, enable_thinking=True)`):
`<|im_start|>user\n{prompt.strip()}<|im_end|>\n<|im_start|>assistant\n<think>\n`.
No system prompt. The chat template trims the content (the Comfy port does
not; follow HF). Tokenized with truncation at **4096 tokens total** (the
suffix can be cut; warn). One prompt per forward, unpadded. Negative prompt
default `""` (the encoded empty template, never zeros).

**VAE**: FLUX.2 VAE (32 latent ch, f8), latent bridge identical to FLUX.2:
2x2 pixel-shuffle to 128 ch at /16, then `(z - bn.running_mean) /
sqrt(bn.running_var + 1e-4)`. Decode = inverse. Reference decodes in fp32;
KBlueLeaf measured fp16 decode at 77 dB PSNR vs fp32 (fine).
- `flux2-vae_small` (the default, shipped with MOSS): encoder like FLUX.2,
  **decoder widths [96, 192, 384, 384]** instead of [128, 256, 512, 512].
  Keys are **diffusers-style** (`decoder.up_blocks.N.resnets.M`,
  `mid_block.attentions.0.to_q`, Linear attention weights). The official
  `vae/diffusion_pytorch_model.safetensors` is the same file.
- `flux2-vae` (full decoder): LDM keys (`decoder.up.N.block.M`), already
  supported by diffucore's FLUX.2 loader. Both decode the same latent space.

**Scheduler** (`MOSSImageScheduler`, reference): `total=1000, shift=5,
training_shift=1`. `grid(count, s)`: `lower = s/(total+s-1)`, `v =
linspace(1, lower, count)`, `s*v/(1+(s-1)*v)`. `sigmas = grid(steps, 5)`
snapped to the **nearest** value of `training = grid(1000, 1).flip(0)` (ties
go to the lower index), then `unique_consecutive`, then diffusers appends a
terminal 0. Model timestep = `sigma * 1000`. Euler: `x += (sigma_next -
sigma) * v`. Note the last nonzero sigma is ~0.024, far below the plain
shift-5 flow schedule's ~0.093 at 50 steps, so a dedicated scheduler matters.
Flow convention: `x_t = (1-s) x0 + s eps`, `v = eps - x0`, `x0 = x - s v`.

**Guidance** (defaults: 50 steps, CFG 3, standard `uncond + s*(cond -
uncond)`, s == 1 skips uncond): the **uncond forward skips block 22**
(`uncond_skip_layers=(22,)`, zero-based) on every step. Optional SLG (a third
forward) exists in the reference but is not a quality-tuned default: skip it.
Initial noise is drawn in fp32; CFG combine and the Euler update in fp32.

**fp16** (KBlueLeaf measurements, 189 forwards): the residual stream peaks at
**~62,000 at block 26 of 42** at 1024² (fp16 max 65,504) and overflows from
~2560². Fix = carry the stream at c = 1/16: scale every writer into the joint
stream (`video_embedder` output, `text_embedder` output) and both gates
(`gate_a`, `gate_f`) by c; the text refiner's own stream likewise (`proj`
output and its two branch outputs) to be safe. Every reader is an RMSNorm
(scale-invariant up to eps; use `eps * c^2` for exactness). Branch outputs
before their gate peak at ~1.3k (o) / 0.7k (ffn_down): no pre-gate scaling
needed. Their LPIPS sweep is flat for c in 1/2..1/64.

**Ecosystem** (HF search 2026-10-23): Comfy-Org/MOSS-Image is the main
distribution (1.46M downloads). Community: 4-step / 8-step distill LoRAs
(`darask0/MOSS-Image_4-step_adapter_lora`, `lodestones/...-8-steps-dmd`
`lora.safetensors`, kohya `diffusion_model.<module>.lora_down/up/alpha` on
`proj.{q,k,v,o,ffn_up_gate,ffn_down}`), style LoRAs, fp16-fixed DiT files
(`KBlueLeaf/MOSS-Image-FP16-fixed`: same keys, stream pre-scaled by 1/16 in
the weights), GGUF, FP8 (`unsloth/*-FP8`). Comfy-Org's
`moss_image_color_depth_control` LoRA changes `video_embedder` to 384 input
channels (a control adapter): out of scope, reject with a clear message.

**Size arithmetic**: per block 3072 x (9216 + 3072 + 16384 + 8192) = 113M
params (113 MB INT8, 226 MB fp16); 42 blocks 4.76B. Non-block ~0.32B (the
text embedder ~0.2B). INT8: all 42 blocks fit in 12 GB VRAM next to a 5 GB
reserve. Per forward at 1024² (~4096 + L tokens): ~40 TFLOP GEMM + ~9 TFLOP
attention, so on the 2060 INT8 roughly 1.3 s/forward and ~2 min per 50-step
CFG image.

## Design decisions

**D1. Canonical module names = the Comfy-Org layout** (fused `proj.qkv`),
which is the main distribution and one GEMM instead of three.
- Comfy bf16 / int8 / fp16-fixed files load `strict=True` with no remap
  (after dropping `comfy_attention.config` and `rope.bands`).
- Official diffusers shards: concatenate `proj.q/k/v` into `proj.qkv` at load
  (a RAM copy of ~2.4 GB total for bf16; acceptable; file-backed masters
  apply to single files only).
- Keep names identical to the reference otherwise so LoRA keys map 1:1.

**D2. fp16 via a runtime residual scale**, like Krea 2's
`residual_scale`: `0.0625` when compute dtype is fp16, else 1.0. Applied at
runtime (scale embedder outputs and gates, norm eps `* c^2`), not by
rewriting weights, so LoRAs and int8 files are untouched.
- KBlueLeaf's fp16-fixed files already carry the stream at 1/16 in the
  weights (same 379 keys as Comfy bf16, fp16 dtype). Their safetensors
  metadata has `kohakumoss_fp16_fix = {"c": 0.0625, "via": "gate", "eps":
  "stock", ...}`. When that record exists, use runtime scale 1.0 for every
  dtype (a pre-scaled stream is harmless in bf16/fp32), never 1/16 on top
  (1/256 overall broke quality in their sweep).

**D3. No padding, one prompt per forward.** Cond and uncond have different
text lengths, so CFG runs two sequential forwards (as Krea 2). Maskless
`attention_bhld`/`attention_blhd` backends (sdpa / fa2 / int8) run unchanged.

**D4. Hoist prompt-only and step-only work (exact).**
- `text_embedder` (proj + 2 refiner blocks) depends only on the prompt (its
  RoPE only on L): run once per prompt per generation, cache (key includes
  LoRA state, since LoRAs hit the refiner).
- `time_embedding`, `shared_block_modulation`, `video_head_modulation`: once
  per step, shared by cond and uncond.
- RoPE cached per `(L, h, w)`.

**D5. Guidance knobs.** Standard CFG scale (MOSS "3" = diffucore 3, no
offset, unlike Krea). `uncond_skip_layers` default `(22,)`, exposed to the
UI as one Settings/advanced field ("Uncond skip blocks", comma list, empty
disables). The existing CFG-interval knobs apply as usual (outside the band
no uncond runs at all).

**D6. Scheduler `moss`** = the reference grid-snapped schedule above, the
family default. Other flow schedulers stay selectable with the shift slider
(default shift 5 for this family).

**D7. Sampler-level latent is 4-D `[B, 128, H/16, W/16]`**, normalized
(bn-applied) space, like FLUX.2's DiT space. Reuse the FLUX.2 bridge helpers
in `pipelines/_flux.py` (`_flux2_patchify_latents`, the
`flux2_latent_mean/std` buffers set in `bundle.load_flux_checkpoint`) rather
than re-deriving. H, W multiples of 16.

**D8. VAE**: support both decoders. Add (a) a diffusers -> LDM key converter
for the AutoencoderKL layout (note: diffusers `up_blocks.i` = LDM `up.(n-1-i)`;
attention `to_q/to_k/to_v/to_out.0` Linear `[C, C]` -> LDM `q/k/v/proj_out`
conv `[C, C, 1, 1]`; `group_norm` -> `norm`; `conv_shortcut` ->
`nin_shortcut`; check against what `models/vae.py` actually names), and (b)
separate decoder widths in `VAEConfig` / `AutoencoderKL`, inferred from the
decoder weights in `_infer_vae_config`. Default UI VAE = `flux2-vae_small`.

**D9. Text encoder**: reuse `Qwen35TextEncoder` (ported from HF, verified to
~1e-6/layer on 0.8B; the 9B uses the same code paths, `output_projection=
False` = final `1+w` RMSNorm). Strip `model.language_model.`, never
materialize `visual.*`, `mtp.*`, `lm_head` (filter before reading). INT8
convrot TE: int8 linears via the existing `int8_layers` / `adopt_int8_layers`
path; **`embed_tokens` int8** is new: either dequantize at load (2 GB fp16)
or an int8 embedding that gathers rows and dequantizes + un-rotates only the
looked-up rows (preferred: tiny, exact). Tokenizer: the vendored
`conditioning/qwen35_tokenizer.json`; verify ids against the official
`tokenizer/tokenizer.json`, in particular that `<|im_start|>`, `<|im_end|>`,
`<think>` encode as single special ids.

**D10. Memory on the 2060** (reuse Krea 2's machinery as is):
`_krea2_masters`-style "auto" -> file-backed masters when files + 4 GB exceed
available RAM; TE layer-streamed from the mmap; leading DiT blocks resident
(INT8: likely all 42), the rest streamed; on-load INT8 for bf16 DiT files
(existing `quantize_linears`) when the int8 chip is on. Generalize the
`_krea2_*` helpers' names only if trivial; otherwise call them as they are.

**D11. Don't refactor the scattered family string compares**; add
`moss_image` beside `krea2` (Krea 2 D10).

## Phase 1: engine, text-to-image (submodule `diffucore/`)

1.1 `models/moss_dit.py` (new; port of the Apache-2.0 reference modules, NOTICE
entry). `MossConfig.from_state_dict` (dim from `video_embedder.weight`, heads
from `layers.0.q_norm.weight`, depth = count of `layers.N`, ffn from
`ffn_up_gate` rows / 2, latent ch from `final_linear_video` rows, text_dim from
`text_embedder.proj.weight`). Methods: `encode_text(ctx_feats) -> text tokens`
(D4), `time_vectors(t) -> (mod, head)`, `forward(x, mod, head, text,
skip_layers=())`. `residual_scale` (D2). An `attn_backend` slot per attention
as Krea 2 / FLUX have. `int8_layer_names(depth)` = `layers.N.proj.{qkv, o,
ffn_up_gate, ffn_down}`. `from_diffusers(sd)` (fuse q/k/v).

1.2 `conditioning/moss_tokenizer.py` (new): template, strip, truncation at
4096 with a warning, no padding.

1.3 VAE (D8): diffusers key conversion + decoder widths.

1.4 TE (D9): int8 embedding, key filtering, width check (TE hidden must equal
the DiT's text_dim 4096; reject Qwen3.5-4B/0.8B with a clear message).

1.5 `bundle.load_moss_checkpoint(dit_path, te_path, vae_path, *, device,
dtype, policy)`, modeled on `load_krea2_checkpoint` (masters, resident
blocks, int8, attention backend, compile hook).

1.6 `loading/detect.py`: fingerprint `shared_block_modulation.weight` +
`video_head_modulation.weight` (bare or under the usual prefixes), checked
before the FLUX branch. `ModelSpec(architecture="moss_image",
prediction="flow", latent_channels=128, context_dim=4096, image_size=1024,
latent_scale=1.0)`.

1.7 `pipelines/_moss.py` + dispatch in `text_to_image.py`:
`moss_text_to_image` (defaults 50 / 3.0 / `moss` / euler), sequential CFG
with `uncond_skip_layers`, CFG-interval knobs, conditioning cache, fp32 noise
/ combine / update, decode via the FLUX.2 bridge. The diffucore sampler zoo
(flow samplers) works through the usual `velocity -> denoised` wrapper.

1.8 **Offline tests** (`tests/test_moss_image.py`, CPU):
- **Keys**: meta-built model loads strictly from fixtures made from the real
  headers (Comfy bf16, Comfy int8, official diffusers renamed, VAE small +
  full). Store compressed key/shape fixtures like `tests/krea2_headers.json`.
- **Oracle DiT**: the reference `MOSSImageTransformer` (vendor the reference
  module code into the test as a fixture or exec it with stubbed diffusers
  base classes; it needs only torch) vs ours, tiny random config (e.g. dim
  256, 2 heads of 128, 3 layers), fp32, two different text lengths, with and
  without `skip_layers`, `residual_scale` 1 and 1/16: max abs diff <= 1e-5.
- **Schedule**: `moss` sigmas == reference `MOSSImageScheduler.set_timesteps`
  for steps in {1, 8, 28, 50, 200} (port the math in the test; diffusers is
  not in `.venv`).
- **Tokenizer**: ids == HF `tokenizers` with the official `tokenizer.json`
  for ~8 prompts (short, long, CJK, emoji, whitespace, multi-line, JSON
  layout, `""`).
- **VAE**: small-decoder fixture builds and loads with no missing keys.
- **INT8 embedding**: dequantized rows == the reference dequant of a
  synthetic convrot int8 table.

1.9 **Gate 1** (GPU, kill `backend/app.py` first; fp16 on the 2060):
- Real-weight DiT parity, no Runpod: run the **reference module on CPU in
  fp32** from `moss_image_sft_bf16.safetensors` (split `proj.qkv` back into
  q/k/v for it) at 256² for single forwards at sigma {1.0, 0.7, 0.3, 0.05}
  with TE features from our encoder; ours fp16 (c=1/16) and INT8 on the GPU:
  rel L2 < 3e-2, cos >= 0.998, no inf/nan. Also check ours fp16 *without*
  the residual scale at 1024² stays finite but near the limit (sanity of D2).
- Optional E2E: 50-step loop at 256² reference-on-CPU vs ours, same noise and
  same TE features: same image, small RMSE.
- One 1024² image of the README's refrigerator prompt (seed 42) and a few
  SFW prompts look right (text rendering is MOSS's strength: check legible
  text). Record s/img and peak VRAM.

## Phase 2: UI + the 2060 path (`backend/`, `static/`)

Mirror the Krea 2 wiring (grep `krea2` / `Krea2` in `backend/` and `static/`;
~150 hits across engine.py, server.py, app.js, index.html, model_arch.py,
model_hash.py, metadata.py, test files).
- `engine.py`: `MODEL_FAMILY_MOSS`, `load_moss` wrapper (label
  `MOSS(<dit>)`), samplers = flow samplers (4-D ones included), schedulers =
  `["moss", "flow", ...]`, `_native_res` 1024, family string compares.
- `server.py`: model type validation + load dispatch + caps (int8, fa2/int8
  attention, fp16 acc; hide fused glue, TeaCache, cuda graphs), default
  offload by free VRAM vs weight size.
- `model_arch.py`: detect the MOSS DiT; TE kind `qwen35_9b` (distinguish
  from the 4B/0.8B Anima encoders by hidden size); VAE kind `flux2` and
  `flux2_small`.
- `static/app.js`: ARCHES / SOURCES / COMPANIONS entries (picking a MOSS DiT
  swaps TE to a `qwen3.5_9b*` file and VAE to `flux2-vae_small*`), defaults
  euler / 50 / CFG 3 / `moss` / shift 5 / empty negative applied once per
  family switch, the uncond-skip field (D5), CFG hint.
- Metadata: model label, the skip layers, scheduler actually used; hash regex.
- **Gate 2** (2060, through the server API): INT8 DiT + INT8 TE + small VAE,
  1024²: load time, s/img, peak VRAM, RSS. bf16 DiT + bf16 TE through
  file-backed masters: it must run (slow is fine).

## Phase 3: img2img, inpaint, LoRA, official folders

3.1 `moss_img2img` (img2img + soft inpaint), Anima / Krea 2 conventions
(ComfyUI full-steps denoise, memory `img2img-denoise-convention`), encode via
the FLUX.2 bridge inverse. Then the detailer and tiled upscaler on top.
3.2 LoRA as forward-hook side branches (Krea 2 approach). Formats: bare
`<module>.lora_A/B`, `transformer.<module>.lora_A/B`,
`diffusion_model.<module>.lora_down/up/alpha`, kohya `lora_unet_<mangled>`.
`proj.q/k/v` LoRAs land on row slices `[0:3072] / [3072:6144] /
[6144:9216]` of the fused `qkv` output. Unmapped count logged (0 expected on
the community LoRAs). Reject shape-changing control LoRAs. Invalidate the D4
text cache on LoRA change.
3.3 Official diffusers folders (`transformer/`, `text_encoder/` shards, `vae/`
diffusers file) via folder / repo root / `*.index.json`, as Krea 2.
**Gate 3**: img2img + inpaint keep region pixel-identical; a 4-step or
8-step distill LoRA at its recommended steps / CFG 1 gives a clean image.

## Phase 4: optional, each its own decision

Live preview latent->RGB factors for the 128-ch space (fit by least squares
on decoded images; FLUX.2 has none today); TeaCache calibration; fused Triton
glue for modulation + RMSNorm + gates; fp8 file loading (unsloth); GGUF (not
planned); MOSS layout-JSON prompt helper in the UI.

## Risks

1. **fp16 stream at 95% of the limit** (D2 fixes it; a LoRA or > 2k res
   would otherwise NaN). Double scaling with fp16-fixed files (D2: read the
   metadata record).
2. **RAM on the dev PC**: INT8 DiT 5.4 GB + INT8 TE 9.7 GB fits with
   file-backed masters; bf16 TE 19.3 GB must stay on the mmap.
3. **Scheduler tail**: using plain `flow` shift 5 silently changes the last
   steps (sigma 0.093 vs 0.024). Family default `moss` + metadata.
4. **RoPE layout**: axis-major rotate-half with the image `f` coordinate =
   L; the oracle test is the guard.
5. **TE int8 embedding**: a wrong un-rotation gives plausible-looking but
   wrong features; unit-test against a synthetic table and spot-check vs the
   bf16 TE file's rows.

## Executor notes

- "Commit" means UI `dev` + diffucore `main` only. No merge to UI main, no tag
  or release unless asked. No `Claude-Session` trailer. Do not commit at all
  unless the coordinator says so.
- No em-dashes anywhere; comments only for the non-obvious why; match the
  surrounding code's style (see `models/krea2_dit.py`, `pipelines/_krea2.py`).
- Kill `python backend/app.py` (by PID, never `pkill -f` with a pattern that
  appears in your own command line) before any GPU work. fp16, not bf16, on
  the 2060.
- Weights (downloading 2026-10-23 into `models/`): `diffusion-models/
  moss_image_sft_int8_convrot.safetensors`, `.../moss_image_sft_bf16.
  safetensors`, `text-encoders/qwen3.5_9b_int8_convrot.safetensors`,
  `vae/flux2-vae_small.safetensors`, `vae/flux2-vae.safetensors`.
- diffusers/transformers are not in `.venv`; for oracles vendor the reference
  module code (torch only) into the test, or use a throwaway venv in the
  scratchpad. Run diffucore tests with `.venv/bin/python -m pytest` from
  `diffucore/`.
