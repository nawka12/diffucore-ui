# Tiled (Ultimate-SD-Upscale style) upscaler — implementation plan

> Status: **implemented & shipped**, including the ESRGAN/spandrel base (the one
> item this plan originally deferred). Kept as the design record; the "Deferred"
> section below tracks what's still outstanding.

## Context

The UI can generate at ~1024² but has no way to produce higher-resolution
images. A full-image hires-fix at 2x+ won't fit the dev GPU (RTX 2060 12 GB),
and a pure pixel/latent upscale either invents no real detail or decodes to mush
on Anima's Wan2.1 latent. The chosen approach is **tiled refinement** (Ultimate
SD Upscale): pre-upscale the whole image, then re-run a low-denoise img2img pass
over overlapping tiles and blend them back. Because every tile is only ~1024²,
2x **and** 4x both fit in 12 GB — tiling is exactly what makes large factors
feasible here.

It's structurally the detailer (`backend/detailer.py` + `Engine.detail`,
engine.py:931) with three swaps: a deterministic tile grid instead of YOLO
boxes, img2img (`ImageToImage`) instead of `Inpaint`, and the canvas is the
upscaled image. The detailer's feathered-mask composite (engine.py:1020-1021)
is the same seam-hiding idea, generalized to a weighted-accumulate blend.

**Decisions made:** base upscale = **Lanczos** now (ESRGAN/spandrel a later
drop-in swap); placement = **both** a standalone "Upscale this image" action and
an auto post-gen toggle.

## Pipeline

```
image → Lanczos resize to target (round(W*scale), round(H*scale))
      → split into overlapping `tile`² tiles (uniform overlap)
      → img2img each tile @ low denoise (default 0.35)   ← refines real detail
      → weighted-feather accumulate → target image → save to gallery
```

Defaults: scale 2.0, tile 1024, overlap 128, denoise 0.35; steps/cfg/sampler/
scheduler reuse the caller's; per-tile seed = base_seed + i.

## New file: `backend/upscale.py` (pure functions, mirrors `detailer.py`)

- `tile_starts(dim, tile, overlap) -> list[int]` — evenly-spaced start offsets so
  overlap is uniform: `n = ceil((dim-overlap)/(tile-overlap))`, starts =
  `round(i*(dim-tile)/(n-1))`; single `[0]` when `dim <= tile`.
- `tile_grid(w, h, tile, overlap) -> list[tuple[int,int,int,int]]` — product of
  x/y starts → boxes, each `min(tile,w) × min(tile,h)`.
- `feather_weights(tw, th, overlap) -> np.ndarray` — outer product of 1-D ramps
  that rise over `overlap` px using `linspace(0,1,overlap+2)[1:-1]` (min weight
  `1/(overlap+1)`, never 0). Uniform feather is correct even at the true canvas
  edge: those pixels are covered by one tile only, so dividing by the weight sum
  cancels the low edge weight (the MultiDiffusion normalization trick).

## `Engine.upscale(...)` in `backend/engine.py` (mirrors `Engine.detail`)

Signature mirrors `detail()`: `image, *, scale, tile, overlap, denoise, prompt,
negative_prompt, steps, cfg_scale, sampler, scheduler, seed, teacache_thresh,
teacache_use_coeffs, progress_callback, preview_callback`.

- Guard `self._loaded`. (i2i works for every family — no `can_inpaint` guard.)
- `scheduler == "oss"` → fall back to `flow`/`karras` (same reason/line as
  detailer, engine.py:967-968: OSS is a full-trajectory schedule).
- `base = image.convert("RGB").resize((round(W*scale), round(H*scale)), LANCZOS)`.
- Resolve `tc_coeffs` once; build `preview_cb` via `_make_preview_cb`.
- numpy float accumulators `acc[H,W,3]`, `wsum[H,W,1]`. For each tile box:
  crop → run `ImageToImage(self._loaded.model)(prompt, crop, strength=denoise,
  steps, cfg_scale, sampler, scheduler, seed=base_seed+i, width=tw, height=th,
  teacache_thresh, teacache_coefficients=tc_coeffs, progress_callback=sub_cb,
  preview_callback=preview_cb, return_info=True)` → accumulate `out*w`, `wsum+=w`
  → `self._reclaim_memory()` per tile (VRAM headroom). Call the pipeline
  directly like `detail()` calls `Inpaint`; it dispatches Anima/FLUX/SD
  internally (image_to_image.py:56-81). Does **not** touch `last_seed`.
- `result = (acc / np.clip(wsum, 1e-6, None)).astype(uint8)` → PIL.
- `sub_cb` maps tile i: `progress(i*steps + step, n_tiles*steps)` (detailer
  pattern, engine.py:1006-1008). Return `(result, "Upscale: W×H → tw×th, n tiles
  @ denoise d")`.

Tiles are 1024 (÷64) so Anima's grid stays happy with no snapping; edge tiles
stay exactly `tile²` (start clamps to `dim-tile`). Low denoise keeps tiles
mutually consistent and curbs per-tile hallucination + Anima's prompt-smearing.

## `backend/server.py`

**Auto post-gen toggle** — extend `GeneratePayload` (after the `detail_*`
block, ~line 108) with `upscale_enabled/scale/denoise/tile/overlap/prompt`. In
`_run_generation`, **after** the detailer loop and before `_save_output`
(~line 311): if enabled and `scale > 1`, `image, unote = ENGINE.upscale(image,
…, prompt=p.upscale_prompt.strip() or clean_prompt, negative_prompt=clean_neg,
steps=p.steps, cfg_scale=p.cfg, sampler/scheduler=p.…, seed=p.seed,
teacache_thresh=p.teacache if p.detail_teacache else 0.0,
teacache_use_coeffs=p.teacache_calibrated, callbacks=…)`; append `unote` to info.

**Standalone endpoint** `/api/upscale` — `UpscalePayload` (input_image base64 +
the same upscale params + prompt/neg/sampler/scheduler/steps/cfg/seed/teacache/
preview). Queue a `Job("upscale", f"upscale {scale}x", run)`; `run` resolves
callbacks via `_make_callbacks`, guards a loaded model, calls `ENGINE.upscale`,
saves via `_save_output`, returns `{image_url, info}`. The generic worker
(server.py:563-591) and SSE need **no** changes — the `done` event already
carries `**result` to the frontend, and the queue list shows `job.label`.

**Metadata** — record upscale params in the PNG: add an `upscale=` kwarg to
`format_metadata` + a `_upscale_fields(...)` helper mirroring `_detailer_fields`
(metadata.py:64, 90-116), pass it from both `_save_output` call sites. Full
restore (`extract_upscale` + frontend form repopulation) is deferred — write
side only, so outputs stay self-documenting without the extra surface.

No model-dir scanner needed (Lanczos base ships no model files).

## `static/index.html` + `static/app.js` (+ minor `style.css`)

- **Auto toggle:** an "Upscale (after generate)" `<fieldset>` under the Detailer
  one (index.html:288), mirroring its Alpine markup. State `upscale:{enabled,
  scale,denoise,tile,overlap,prompt}`; add the `upscale_*` keys to the
  `/api/generate` payload object (app.js:727-736, next to `detail_*`).
- **Standalone action:** an "Upscale ⬆" button on the result view and the
  gallery lightbox that opens a small popover (scale/denoise/tile/overlap, +
  optional prompt) and POSTs `/api/upscale`. Reuse the existing gallery
  image→input-image loading path (app.js:~1091, "send to img2img") to get the
  source image as a data-URL. Progress + the saved result arrive over the
  existing SSE stream — no new client plumbing.

## Verification

**Offline (no GPU), the loop-until-green criterion** — add `backend/test_upscale.py`
(first UI-repo test; run `.venv/bin/python -m pytest backend/test_upscale.py`):
1. `tile_grid` covers the whole canvas (box union == full area), every tile is
   `tile²` for a ÷64 `tile`, overlap uniform within ±1 px, single tile when
   `canvas <= tile`.
2. Blend is seam-free: two overlapping **constant-color** tiles reconstruct the
   constant (max abs err ≈ 0); a smooth gradient split across tiles reconstructs
   with bounded neighbour delta (no step at seams). Pure numpy — no model.
3. Import smoke: `from upscale import tile_grid, feather_weights` and app boots
   (`/api/models` 200).

**GPU-deferred** (matches the project's verify-on-Runpod pattern for the
detailer etc.): real Anima 1024→2048 via the standalone button + the auto
toggle — eyeball seams, confirm gallery save + PNG metadata. Note as
not-yet-GPU-tested.

## Deferred (not in this change)

- ~~ESRGAN/spandrel base~~ — **done.** `Engine._esrgan_upscale` (guarded
  `import spandrel`, tiled + feather-blended), `models/upscalers/` scan, and a
  "Base upscaler" dropdown; Lanczos stays the default fallback. ESRGAN base + low
  denoise (~0.2) is the recommended path — a Lanczos base has no denoise sweet
  spot (low = no detail added, high = per-tile subject duplication).
- Metadata `extract_upscale` + frontend restore of upscale settings (the write
  side ships: `Upscale scale/tile/overlap/denoise/teacache/base` in the PNG).
