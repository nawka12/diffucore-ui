# Krea 2 (Raw + Turbo): implementation plan

> Status: **Phases 0, 1 and 2 done 2026-10-12 (uncommitted); Phase 3 next.**
> Written for an executor agent.
>
> **Phase 2 results.**
>
> D9 is realized as **file-backed masters**, which are new in the runtime:
> - `DevicePolicy.stream_masters` takes "auto" / "pinned" / "file";
> - `file_backed()`, plus `on_device()` re-pointing parked tensors back at the
>   file;
> - `stream_blocks(masters="file", dtype=)`, and `_GroupStreamer(pin=False)`.
>
> When the DiT + TE files plus 4 GB exceed available RAM, the DiT blocks and
> the parked TE stay on the checkpoint mmap (page cache) and are cast on
> upload. So bf16 no longer has to be refused, and nothing is copied into
> RAM. The loader also rejects a wrong TE with a clear message (layer count,
> width vs the DiT's txtdim) and warns when the file has no Qwen3-VL markers.
>
> UI (backend + static, done by a subagent, reviewed): `Krea2` model type, a
> DiT/VAE/TE/Variant grid, per-variant defaults, the `krea2` scheduler, the
> CFG hint, the Raw >1.3 MP warning, preview, label/hash/metadata
> (`Krea2 variant:`), XYZ, and 400 gates for i2i / inpaint / detailer /
> upscaler / LoRA. 341 backend tests pass.
>
> Gate 2 on the 2060, through the real server API, Turbo INT8:
> - **Load**: 3.1 s.
> - **Turbo**: 39 s for the first image, then about 29 s per image (prompt
>   cached) at both 1024² and 1216x832. Images are correct.
> - **Raw** INT8, 28 steps + CFG, through the real loader: 282 s at 1024².
> - **Memory**: peak GPU about 9.6 GB (nvidia-smi, all processes). Server RSS
>   is about 13 GB, but it is file-backed: system "available" stayed at
>   about 15 GB.
>
> Small open items:
> - Metadata records the `Shift` slider even under the `krea2` scheduler,
>   which ignores it.
> - Restoring a generation from image metadata doesn't set the variant (the
>   load-form restore does).
> - Unspecified details (e.g. neckerchief colour) flip with about 1%
>   conditioning noise (fp16 vs bf16 TE). This is chaotic sensitivity, not a
>   bug.
>
> **Phase 1 results** (diffucore, uncommitted at time of writing).
>
> New files:
> - `models/krea2_dit.py`
> - `conditioning/krea2_tokenizer.py`
> - `pipelines/_krea2.py`
> - `bundle.load_krea2_checkpoint` / `load_krea2_dit`
> - `detect` (`krea2`, plus `ModelSpec.variant`)
> - `tests/test_krea2.py` (8 tests)
> - `tests/_krea2_parity.py` (manual GPU parity)
>
> Gate 1, on the 2060 against the golden bundle:
> - **Tokenizer**: ids exact on all 9 prompts.
> - **Text encoder, fp16**: rel L2 0.9% / 0.9% / 1.9% (anime / photo /
>   empty). Our bf16 is *further* from the reference (1.4% / 3.0%), so this
>   is the bf16 noise floor.
> - **DiT single forwards**, against our own fp32 run:
>
>   | Run | rel L2 vs fp32 |
>   |---|---|
>   | diffusers reference (bf16) | 3.7–4.3% |
>   | ours, bf16 | 3.9–4.1% |
>   | ours, fp16 | 1.6–2.6% |
>
>   **fp16 is the most accurate.** Against the reference: fp16 1–6% (cos ≥
>   0.998); INT8 ConvRot files Turbo 1–5.6%, Raw 0.7–2.0%. The residual /4
>   fold changes fp16 by 0.2–1%, which is rounding.
> - **E2E from the golden noise**: image RMSE (/255) against the reference
>   PNG:
>
>   | Run | RMSE vs reference |
>   |---|---|
>   | Turbo, fp16 | 15.8 |
>   | Turbo, INT8 | 14.6 |
>   | Raw INT8, 28 steps + CFG | 10.6 |
>   | **Our bf16 (floor)** | **19.4** |
>
>   Every fp16/INT8 run is inside that floor, and visually identical.
> - **2060 speed** (INT8, disk-backed blocks): Turbo about 43 s, Raw about
>   244 s at 1024².
> - The **RAM-bound full-loader run** (TE + streamed DiT together) is left
>   to Phase 2 (D9). The loader wiring is covered by a tiny-checkpoint CPU
>   smoke test.
> - Note: `TextToImage` passes concrete defaults (`cfg_scale=7`, `steps=20`)
>   for every family, so the per-variant defaults only apply to direct
>   `krea2_text_to_image` calls. The UI must send them (2.3).
>
> **Phase 0 results.**
> - Pod: RTX PRO 5000 Blackwell 48 GB, community, about 9 min. Stack:
>   diffusers 0.41.0.dev0, transformers 5.19.0, torch 2.8.0+cu128.
> - Golden bundle: `/run/media/kayfa/a75ef841-da8c-45bc-9f54-ce6166f2c98a/krea2-golden-20261012/`
>   - `golden.safetensors`: 138 tensors (token ids, masks, positions; TE
>     features; DiT in/out at 3 steps per run; noise; sigmas; final packed
>     latents),
>   - 6 PNGs, `probe.json`, `meta.json`,
>   - the capture script `golden_capture.py`.
> - Reference speed on that card:
>   - Raw 28 steps + CFG at 1024²: 54 s,
>   - Turbo 8 steps at 1024²: 8 s,
>   - Turbo at 2048x1152: 21 s.
>
> **Gate 0 decision** (max |x| over all runs; fp16 max is 65504):
> - **DiT: only the residual stream is hot.** Block outputs peak at 22784
>   (0.35 of the fp16 max), rising toward the last block and the last steps.
>   Every GEMM input and output stays at or below 4352 (0.07).
>   - That leaves only 2.9x headroom, below the 2^14 bar, so use **rung (b)**,
>     the residual downscale fold, with **c = 4**. The peak becomes about 5.7k.
>     This is exact (power of two).
> - **TE: rung (a), plain fp16.** The residual peaks at 12800 (0.195) at
>   layer 16. This is the Qwen massive activation on the fixed system-prefix
>   tokens, so it doesn't depend on the prompt.
> - Text path (txtfusion) peak: 3456.
> Origin: model-variant wishlist; user picked Krea 2 and asked for Raw and
> Turbo in one pass, because other diffucore users run stronger GPUs than the
> dev PC's RTX 2060 12 GB.

## Goal

One engine family, `krea2`, that loads both official checkpoints and every
common repackaging, with Raw vs Turbo handled as a **sampling preset**, not a
second code path. Nothing built for Turbo has to be redone for Raw.

Success = (1) Turbo and Raw each match the diffusers reference image for the
same noise; (2) Turbo runs end to end from the UI on the 2060 (12 GB VRAM,
**23 GB system RAM**); (3) Raw runs on the same PC (slow is fine) and natively
on 24 to 32 GB cards.

## Verified facts

All read from source or checkpoint headers on 2026-10-12, not from blog posts.
Sources: `krea-ai/krea-2` (code Apache-2.0: `mmdit.py`, `encoder.py`,
`sampling.py`, `inference.py`), diffusers `pipelines/krea2/pipeline_krea2.py`,
`krea/Krea-2-{Raw,Turbo}` configs + safetensors headers, `Comfy-Org/Krea-2`
headers.

**Raw and Turbo are the same network.** Both official files have the same 430
keys with identical dtypes and shapes; the Comfy bf16 files are byte-for-byte
the same layout. Weights cannot tell the variants apart. The only differences
are in the sampling setup:

| | Raw (`is_distilled: false`) | Turbo (`is_distilled: true`, TDM) |
|---|---|---|
| shift | resolution-aware: mu linear in image tokens, (256, 0.5) to (6400, 1.15) | fixed mu = 1.15 (shift e^1.15 = 3.158) |
| steps / Krea cfg | 28 / 4.5 (code default); card example 52 / 3.5 | 8 / 0 (no uncond pass) |
| resolution | trained up to ~1k | 1k to 2k |

**DiT** (`SingleStreamDiT`, about 12.8B params): single stream; text and image
tokens are concatenated. Config: features 6144, 28 blocks, 48 query heads / 12
KV heads (GQA 4:1), head_dim 128, SwiGLU MLP 16384, patch 2, 16 latent
channels (`first`: 64 -> 6144), tdim 256, `bias=False` except `first`,
`last.linear`, `tmlp`, `tproj`, `txtmlp`.
- Attention: separate `wq wk wv`, per-head QK RMSNorm, then RoPE, then
  `wo(attn * sigmoid(gate(x)))`.
- RMSNorm weight is `scale + 1` (zero-init scale), computed in fp32.
- RoPE: 3 axes `[32, 48, 48]`, theta 1e3, computed in fp64 then fp32, applied
  in fp32. Text tokens sit at position (0,0,0), i.e. no rotation; image tokens
  at (0, h, w).
- Modulation: `t = tmlp(temb(t))`, then `tvec = tproj(t)` (GELU-tanh, then
  Linear to 6F), shared by all blocks. Each block adds its own `mod.lin` (6F)
  and splits it into pre/post scale, shift and gate. Gates are linear (no
  tanh).
- `temb`: dim 256, `t * 1000`, period 1e4, **cos then sin**.
- `last`: RMSNorm, then `(1 + scale) * x + shift` using **`t` (not `tvec`)**
  plus a learned (2, F) table, then Linear to 64. The output is sliced to image
  tokens.
- Text path: TE features `(B, L, 12, 2560)` go through `txtfusion`:
  - 2 blocks attending over the **layer axis** (12 "tokens" per text token, no
    mask),
  - `projector` Linear(12 -> 1),
  - 2 refiner blocks over tokens (masked),
  - then `txtmlp` (RMSNorm, 2560 -> 6144, GELU-tanh, 6144 -> 6144).

  None of this depends on the timestep.
- Padding: the sequence is padded to a multiple of 256 with a key mask, only
  "to stabilize compiled kernel shapes".

**Text encoder**: Qwen3-VL-4B-Instruct, text only.
- Text config: 36 layers, hidden 2560, 32 q / 8 kv heads, head_dim 128, ffn
  9728, rms eps 1e-6, **rope_theta 5,000,000**. Interleaved mRoPE [24,20,20]
  reduces exactly to plain 1-D RoPE when T=H=W, which always holds for text.
- Uses HF `hidden_states[i]` for i in (2,5,8,...,35), stacked on dim 2. Layer 36
  and the final norm are never used.
- Template: prefix `<|im_start|>system\nDescribe the image by detailing the
  color, shape, size, texture, quantity, text, spatial relationships of the
  objects and background:<|im_end|>\n<|im_start|>user\n` (34 tokens), then the
  prompt, right-padded to 541 tokens in total, then the suffix
  `<|im_end|>\n<|im_start|>assistant\n` (5 tokens). Features and mask are
  `[34:]`, so a prompt can be at most 507 tokens.
- Tokenizer `padding_side` is unset, which means right padding.
- Positions: diffusers builds `cumsum(mask) - 1`, so padding takes no position,
  and its comment says this matches training. **`krea-ai/krea-2/encoder.py`
  does not pass position_ids**, so there the suffix gets raw positions near
  541. Treat diffusers as the reference.

**VAE**: Qwen-Image VAE, f8, 16 channels. Krea's `vae/config.json`
mean/std are identical to diffucore's `_WAN21_LATENTS_MEAN/_STD`
(`models/qwen_image_vae.py:23`), and the Comfy file is the same
`qwen_image_vae.safetensors` Anima uses. Krea uses it 4-D (T = 1).

**Sampler (reference)**: flow matching, Euler, t from 1 to 0 over
`linspace(1, 0, N+1)` passed through `e^mu / (e^mu + (1/t - 1))` (the same as
diffucore's `flow_matching_schedule` with shift = e^mu). The model's `t` is
sigma. CFG is **`v = cond + g * (cond - uncond)`**, which is standard CFG with
**scale g + 1**. Krea "4.5" = diffucore 5.5; Krea "0" = 1.0 (off). The negative
prompt defaults to `""` (the encoded template, not zeros).

**Files in the wild** (Comfy-Org/Krea-2 has 6.2M downloads and is ungated;
it is the main distribution):

| file | size | format |
|---|---|---|
| `krea2_{raw,turbo}_bf16` | 26.28 GB | original keys, big linears bf16, small params (`tproj`, `tmlp`, `txtmlp`, norms, `mod.lin`) **F32** |
| `krea2_{raw,turbo}_int8_convrot` | 13.49 GB | block linears I8 + `weight_scale` [out,1] + `comfy_quant` tags; rest bf16 |
| `krea2_{raw,turbo}_fp8_scaled` | 13.14 GB | F8_E4M3 + scalar `weight_scale`, `_quantization_metadata` header |
| `krea2_turbo_{mxfp8,nvfp4}` | 13.5 / 7.7 GB | Blackwell-only formats |
| `qwen3vl_4b_bf16` / `_fp8_scaled` | 8.88 / 5.24 GB | keys `model.language_model.*` + `model.visual.*` (vision tower, drop it) |
| official `raw.safetensors` / `turbo.safetensors` | 26.28 GB | same as Comfy bf16 |
| official `transformer/` (3 shards) | 26.3 GB | diffusers names (table R) |

LoRAs already come in two formats, even within Comfy-Org's own repo:
- `diffusion_model.<original name>.lora_down/lora_up` (e.g.
  `krea2_turbo_lora_rank_64_bf16`, which turns Raw into Turbo),
- `transformer.<diffusers name>.lora_A/lora_B` (e.g. `krea2_retroanime`).

Both target every Linear, including `txtfusion`, `tproj`, `tmlp`, `first`
and `last`.

**Size arithmetic**:
- Per block: 434M params, so 434 MB INT8 and 868 MB fp16.
- All 28 blocks: 12.16B params.
- Non-block params: 0.66B, of which `tproj` alone is 226M.
- TE text-only: about 3.9B params with 35 layers, so 7.8 GB in bf16 and about
  4.3 GB INT8.

## Design decisions

**D1. One family, a variant preset.**
- `variant in {"raw", "turbo"}` is a load-time field. When the user leaves it
  unset, infer it from the filename (`turbo` substring means turbo, otherwise
  raw); they can override it.
- The variant drives:
  - default scheduler shift mode (dynamic vs mu 1.15),
  - steps, CFG and sampler defaults,
  - metadata.
- It never changes the network. Raw + `krea2_turbo_lora` is used with the
  Turbo preset; the user switches it.

**D2. Standard CFG in the UI.**
- Show diffucore's usual scale s, with `g = s - 1` internally.
- Defaults: Raw 5.5 (Krea 4.5), Turbo 1.0.
- `s == 1` must skip the uncond forward. This is the existing Anima rule at
  `_anima.py:283`; reuse it.
- Tooltip and GUIDE: "Krea's guidance 4.5 = 5.5 here".
- Before shipping, check which number ComfyUI's Krea 2 template workflow uses.
  The workflow JSON is data, so reading it is fine under the license policy.

**D3. Drop all padding (exact).**
- One prompt per forward, so all batch rows have the same text length.
- Tokenize `prefix + prompt` with no padding, append the 5 suffix ids, and use
  positions `0..n-1`. That is exactly diffusers' cumsum positions over the
  valid tokens.
- Padded keys are masked in every masked attention, and the layerwise
  txtfusion blocks are per-token. So removing the pads (and the multiple-of-256
  pad) changes nothing, and the DiT needs **no mask**. That lets the existing
  maskless `attention_bhld` backends (sdpa / fa2 / int8) run unchanged.
- Assert there is no mask in forward.

**D4. GQA by expanding K/V.**
- `repeat_interleave(4, dim=heads)` before `attention_bhld`, which matches SDPA
  `enable_gqa` semantics: query head h uses KV head h // 4.
- About 50 MB extra at 1024². GQA-native kernels are Phase 4.

**D5. Hoist the timestep-free and prompt-only work (exact).**
- `txtfusion + txtmlp` run once per prompt per generation (cond and uncond) and
  the result is cached.
- `tmlp/tproj` run once per step and are shared by cond and uncond.
- Under `offload="stream"`, those modules (about 0.66B params, 1.3 GB fp16)
  leave VRAM after use.
- The cache key must include LoRA state, because LoRAs hit these layers.

**D6. 4-D latent at the sampler level.**
- Keep `(B, 16, H/8, W/8)` and patchify 2x2 inside the backbone wrapper (Anima
  style, not FLUX's packed tokens).
- This gives the 4-D-only samplers (realism/aether/omega...) and live preview
  for free.
- Its latent space equals Anima's (same VAE, same mean/std), so reuse Anima's
  `_PREVIEW_RGB` row after checking 4-D vs 5-D shape.

**D7. Precision per device.**
- sm80+: bf16 compute, the training dtype, no overflow concern.
- sm75 (the 2060): fp16. Whether that is safe is **decided by the Phase 0
  probe**, not assumed. Mitigation ladder, cheapest first:
  - (a) Nothing, if every site stays below 2^14.
  - (b) **Residual downscale fold.** The residual stream is only read through
    RMSNorm (scale-invariant) in every block and in `last`. So store x/c, and
    fold 1/c into `first` (w, b), `txtmlp.3` (w, b), and the pregate/postgate
    rows of `tproj.1` (w, b) plus each `mod.lin`. With c a power of 2 the fold
    is exact up to RMSNorm eps.
  - (c) fp32 for the specific hot ops (MLP hidden `silu(g)*u`, `attn*sigmoid`).
  - (d) fp32 residual adds with fp16 GEMMs.
- The TE gets its own probe; Qwen-family residual streams have known massive
  activations. Fallback: fp32 compute with fp16/INT8 storage, streamed layer
  by layer. It runs once per prompt, so speed barely matters.

**D8. Module names = original checkpoint keys.**
- Comfy bf16, Comfy INT8 and the official files `load_state_dict(strict=True)`
  with no remap.
- `diffusion_model.*` LoRAs map 1:1.
- diffusers keys go through table R.
- Fold the RMSNorm `+1` at load and keep fp32 for norm params.

**D9. Memory on the 2060.**
- bf16 DiT (26.3 GB) is larger than the 23 GB of RAM: refuse with a clear
  message pointing to `int8_convrot`. Check file size against total RAM
  **before** loading.
- INT8 DiT (13.5 GB) plus a bf16 TE (7.8 GB) is about 21.3 GB, too much. So on
  low RAM the TE is INT8-quantized on load (about 4.3 GB, existing
  `quantize_linears`) or unloaded after encode. Pick by measurement.
- VRAM: stream all 28 INT8 blocks with prefetch. One block is about 36 ms of
  PCIe 3.0 x16 transfer against about 110 ms of compute, so it should be
  hidden. Measure it. Keep text-path modules out of VRAM during sampling
  (D5).

**D10. Don't refactor the ~30 scattered family string compares** (survey
item k). Add `krea2` alongside them; a cleanup is a separate decision.

## Rename table R (diffusers -> original)

Verified on real tensors: the diffusers values are **identical** to the
original ones (the norm `+1` is not folded). `scale_shift_table` is only a
reshape.

| diffusers | original |
|---|---|
| `transformer_blocks.N.attn.to_{q,k,v}` | `blocks.N.attn.w{q,k,v}` |
| `transformer_blocks.N.attn.to_gate` / `to_out.0` | `blocks.N.attn.gate` / `wo` |
| `transformer_blocks.N.attn.norm_{q,k}.weight` | `blocks.N.attn.qknorm.{q,k}norm.scale` |
| `transformer_blocks.N.ff.{gate,up,down}` | `blocks.N.mlp.{gate,up,down}` |
| `transformer_blocks.N.norm{1,2}.weight` | `blocks.N.{pre,post}norm.scale` |
| `transformer_blocks.N.scale_shift_table` [6,F] | `blocks.N.mod.lin` [6F] (reshape) |
| `img_in` | `first` |
| `final_layer.{linear, norm.weight, scale_shift_table}` | `last.{linear, norm.scale, modulation.lin}` |
| `time_embed.linear_{1,2}` | `tmlp.{0,2}` |
| `time_mod_proj` | `tproj.1` |
| `txt_in.norm.weight` / `txt_in.linear_{1,2}` | `txtmlp.0.scale` / `txtmlp.{1,3}` |
| `text_fusion.{layerwise,refiner}_blocks.N.*` | `txtfusion.{same}.N.*` with the block renames above (`norm{1,2}` -> `{pre,post}norm`) |
| `text_fusion.projector` | `txtfusion.projector` |

## Phase 0: ground truth + fp16 probe (Runpod, before engine code)

You need any bf16-capable GPU that holds the 26.3 GB bf16 DiT. Cheapest
options:
- a 48 GB card (A40 / RTX A6000 / L40S): DiT + TE stay resident;
- a 32 GB RTX 5090 with `enable_model_cpu_offload()`, which parks the TE in
  RAM while the DiT runs.

A100/H100 is overkill. This is about an hour of work: a few dozen forwards and
one probe pass. **Quote the live hourly price and get the user's go before
creating the pod.** See memory `runpod-pod-access` for SSH quirks. Install
diffusers from source (Krea2Pipeline is 0.39.0.dev0).

0.1 **Golden bundle** (safetensors + PNG, keep it under about 50 MB; store it
outside the repo and give its path in the test env var):
- Token ids for about 8 prompts: short, long (> 507 tokens, to check the
  truncation boundary), CJK, emoji, leading/trailing whitespace, multi-line,
  and `""`.
- TE features `(L, 12, 2560)` for 3 of them.
- One DiT forward for Raw and Turbo: fixed seeded latent, t in {1.0, 0.75,
  0.5, 0.25, 0.05}, one prompt.
- Full runs with the initial noise saved:
  - Turbo 8 steps, cfg 0, 1024², seeds 0 and 1,
  - Raw 28 steps, Krea cfg 4.5, 1024², seeds 0 and 1,
  - Raw at 1280x768 (exercises the dynamic shift),
  - Turbo at 2048x1152.

0.2 **Activation probe**: forward hooks record max |x| per site across those
trajectories. Sites:
- DiT residual after each block,
- `attn*sigmoid(gate)` before `wo`,
- MLP hidden,
- MLP and attention branch outputs before the gate,
- modulation vectors,
- txtfusion and txtmlp outputs,
- the TE residual after each layer.

Output a table of max / 65504 per site.

**Gate 0**: choose a rung of the D7 ladder for the DiT and for the TE, and
write the table and the choice into this doc's status block.

## Phase 1: engine, text-to-image, both variants (submodule `diffucore/`)

1.1 `models/krea2_dit.py` (new; port of Apache-2.0 code, add a NOTICE entry).
- Config derived from the state dict:
  - depth = count of `blocks.N`,
  - F from `first.weight`,
  - head_dim from `qnorm.scale`,
  - kv heads from `wk` rows,
  - mlp from `mlp.gate` rows,
  - txt dims from `txtfusion`,
  - text layers from `projector` [1, 12],
  - tdim from `tmlp.0`.
- Methods:
  - `encode_text(feats) -> ctx (B, L, F)`: txtfusion + txtmlp.
  - `time_vectors(t) -> (t_emb, tvec)`.
  - `forward(x_tokens, ctx, t_emb, tvec, freqs)`.
  - RoPE freqs cached per (L_txt, h, w).
- An `attn_backend` slot on `Attention`, as FLUX has.
- int8 layer list: 8 linears x depth (`wq wk wv gate wo`, `mlp.{gate,up,down}`).

1.2 **TE**: reuse `Qwen3TextEncoder`.
- Strip `model.language_model.` or `language_model.` down to the `model.*`
  layout that `_qwen3_config_from_sd` (`bundle.py`) expects.
- Never materialize `visual.*` tensors (filter keys before reading; the Comfy
  file is 8.88 GB with vision).
- Pass **rope_theta = 5e6** explicitly. It can't be inferred from weights;
  Klein's Qwen3-4B is 1e6, and the wrong theta is a silent quality loss.
- Build only 35 layers.
- `hidden_layers=[2,5,...,35]`. diffucore's index convention already equals HF
  `hidden_states[i]` (`qwen3_text.py:179-184`).

1.3 `conditioning/krea2_tokenizer.py` (new): vendored
`qwen3_tokenizer.json`, the template from Facts, truncation to 541 total, the
suffix appended, slice `[34:]`, no padding (D3).

1.4 `bundle.load_krea2_checkpoint(dit_path, te_path, vae_path, *, variant,
device, dtype, policy, int8)`.
- DiT sources:
  - original keys (bare, `diffusion_model.`, `model.diffusion_model.`),
  - Comfy `int8_convrot` via `int8_layers` / `adopt_int8_layers`,
  - diffusers shards (directory or `*.index.json`, via R).
- VAE: the same `QwenImageVAE` path as Anima.
- On-load INT8 for bf16 files. Stream wiring `("blocks",)` (D9).
- Attention backend stamping. Stamp the variant on the bundle.

1.5 `loading/detect.py`.
- Fingerprint: `txtfusion.projector.weight` or `text_fusion.projector.weight`,
  under the usual prefixes, checked **before** the FLUX branch.
- `ModelSpec(architecture="krea2", prediction="flow", latent_channels=16,
  context_dim=2560, image_size=1024, latent_scale=1.0)`.

1.6 `pipelines/_krea2.py` + dispatch in `text_to_image.py`.
- Flow CONST: denoised = x - sigma*v. H and W must be multiples of 16.
- Sequential CFG with g = s - 1, plus the CFG interval knobs as in Anima.
- Scheduler `krea2` (family default):
  - Raw: `flow_matching_dynamic_shift(seq, base_seq_len=256,
    max_seq_len=6400, base_shift=0.5, max_shift=1.15)` with seq = (H/16)(W/16).
  - Turbo: shift = e^1.15.
- The other flow schedulers stay selectable with a manual shift.
- Conditioning cache (D5).
- Warn when a prompt is truncated at 507 tokens.

1.7 **Offline tests** (CPU):
- **Keys**: for each fixture (Comfy bf16, Comfy int8, official raw,
  diffusers-renamed), a meta-built model loads strictly. Fixtures are
  compressed key/shape lists from the headers.
- **Oracle**: skipped unless `KREA2_REF` points at a `krea-ai/krea-2`
  checkout. Same pattern as the ComfyUI oracle tests; run with
  `TORCHDYNAMO_DISABLE=1`, because the reference compiles RMSNorm/rope/last.
  - Tiny random config: F 256, 2 heads, 1 kv head, head_dim 128 (satisfies the
    axes rule).
  - The reference runs padded + masked; ours runs unpadded.
  - fp32 max abs diff <= 1e-5. This proves D3 and D4.
- **Schedule**: equals `sampling.timesteps` for Raw at 4 resolutions and
  Turbo mu 1.15.
- **CFG**: `s = g + 1` equivalence.
- **Tokenizer**: ids equal the Phase 0 fixture.

1.8 **Gate 1** (GPU, needs Phase 0):
- TE features: rel L2 < 1e-2 per layer.
- DiT forward: bf16 on sm80 rel L2 < 2e-2; fp16 on the 2060 with the chosen
  D7 rung rel L2 < 3e-2, no inf/nan.
- E2E with the golden noise injected:
  - Turbo: visually identical, RMSE <= 3/255.
  - Raw 28 steps: same image, small RMSE. This compares runs at the same step
    count, so memory `no-converged-reference-at-cfg45` does not apply.

## Phase 2: UI + the 2060 path (`backend/`, `static/`)

2.1 `backend/engine.py`.
- `MODEL_FAMILY_KREA2`, plus a `load_krea2` wrapper with label
  `Krea2(<dit>)`.
- `SAMPLERS_KREA2` = flow samplers, 4-D ones included (D6).
- `SCHEDULERS_KREA2` = `["krea2", "flow", ...]`.
- `_PREVIEW_RGB` row (D6), `_native_res` 1024.
- The family string compares listed in the survey (`grep -n "MODEL_FAMILY_ANIMA\|_FLUX_FAMILIES"`).
- Add Krea2 to the `model_hash.py` regex.

2.2 `backend/server.py`.
- Model type `"Krea 2"`: validation, load dispatch, a `variant` field.
- Caps:
  - samplers and schedulers,
  - chips: int8, fa2 / int8 attn, fp16 acc,
  - hidden: fused glue, TeaCache, cuda graphs.
- Default offload chosen by free VRAM against weight size.
- The RAM pre-check (D9).

2.3 `static/` (`index.html`, `app.js`).
- A radio and a file grid: DiT, TE, VAE, with the VAE defaulting to
  `qwen_image_vae.safetensors`.
- A Variant select: auto from filename, the user can override.
- Per-variant defaults, applied once like `fluxApplied`:
  - Raw: euler / 28 / cfg 5.5 / `krea2`,
  - Turbo: euler / 8 / cfg 1.0 / `krea2`.
- CFG tooltip (D2).
- A Raw warning above about 1.3 MP.

2.4 **Metadata**: model label, variant, standard CFG, the shift actually used.
Loading a past job must restore the variant (multi-device queue form restore).

2.5 **Gate 2** (on the 2060; kill the running diffucore-ui server first):
- Turbo 1024² from the UI. Record s/img, peak VRAM, peak RSS (target RSS
  <= 18 GB) and RMSE against the Gate 1 image.
- Raw 28-step runs. Record s/img.
- Estimate before measuring (arithmetic only): about 114 TFLOP per forward at
  1024², which gives about 3 s per forward with INT8. That is roughly 30 to 40
  s per Turbo image and about 3 min per Raw image (56 forwards).

## Phase 3: img2img, inpaint, LoRA, diffusers folders

3.1 **img2img / inpaint**: `krea2_img2img` modeled on `anima_img2img`.
- Same denoise convention as Anima (ComfyUI full-steps; memory
  `img2img-denoise-convention`).
- Soft latent-mask inpaint.
- Then verify that the detailer and the tiled upscaler work on top of it.

3.2 **LoRA**: a `krea2` branch in `lora.py`.
- Formats:
  - `diffusion_model.<orig>.lora_down/up/alpha`,
  - `lora_unet_<mangled orig>`,
  - `transformer.<diffusers>.lora_A/B` via R.
- Log an unmapped count, which must be 0 on both Comfy-Org LoRA formats.
- Invalidate the D5 caches when a LoRA changes. INT8 requant uses the existing
  path.
- **Gate 3**: Raw + `krea2_turbo_lora_rank_64` with the Turbo preset gives a
  Turbo-like image, and a style LoRA visibly applies.

3.3 **Official folders**: diffusers `transformer/` shards via R, and
`text_encoder/model.safetensors` (check its prefix against the header first).

## Phase 4: optional, each its own decision

- **fp8_scaled loading** (DiT 13.1 GB, TE 5.2 GB). diffucore has no fp8 at all
  today. Store fp8 and upcast per layer. This is generic, so all families gain;
  native fp8 GEMM on sm89+ comes later.
- **TeaCache calibration** for Krea. Raw at 28 to 52 steps benefits; Turbo at
  8 steps leaves little to skip.
- **Fused Triton glue** for modulation + RMSNorm + gates (the Anima precedent:
  x1.25).
- **GQA-native fa2/int8 attention** (drops the D4 expansion).
- **ControlNets**: community pose and depth models exist
  (`thedeoxen/Krea-2-pose-controlnet`, `Patil/Krea-2-depth-controlnet`).
- Not planned: mxfp8 / nvfp4 (Blackwell only), GGUF, Krea's prompt-expansion
  LLM (`docs/expansion.txt`).

## Risks

1. **fp16 overflow on Turing** (DiT and TE). Phase 0 decides; D7 has the
   ladder. Not a concern on sm80+.
2. **RAM on the dev PC** (23 GB). bf16 is impossible here, and INT8 with a
   bf16 TE is too tight (D9).
3. **Position-id divergence.** The `krea-ai/krea-2` encoder and diffusers
   differ. Diffusers is the target; the golden bundle comes from diffusers.
4. **CFG convention.** Users pasting Krea's documented "4.5" get weaker
   guidance (s = 4.5, i.e. Krea 3.5). Mitigated by D2's defaults and tooltip.
5. **Wrong variant preset** degrades the image silently (Turbo with the
   dynamic shift, or Raw with mu 1.15). Mitigated by the filename default, a
   visible field, and recording it in metadata.
6. **License.** The weights are under the Krea 2 Community License: commercial
   use only under $1M ARR, and §4.2 obliges deployers to use content filters.
   diffucore ships no weights, so §3 does not bind us. GUIDE should link the
   license and note that the UI's NSFW filter exists for users who need §4.2.
   The ported code is Apache-2.0: keep the notice in NOTICE.

## Executor notes

- "Commit" means UI `dev` + diffucore `main` only. No merge to UI main, no tag
  or release unless asked. No `Claude-Session` trailer.
- No em-dashes anywhere; comments only for the non-obvious why.
- Kill `python backend/app.py` before any GPU work. Use fp16, not bf16, on the
  2060.
- Phase order is a dependency chain: 0, then 1, then 2. Phase 3 items are
  independent of each other.
