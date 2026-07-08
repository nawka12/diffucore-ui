# Conditioning cache — implementation plan

> Status: **implemented 2026-07-08.** `runtime/cond_cache.py` (`ConditioningCache`,
> LRU 16, CPU storage) + `ModelBundle.cond_cache` + the three pipeline touch points
> (Anima t2i/i2i share one entry; FLUX keyed on prompt; SD/SDXL split into a cached
> encode + a resolution-dependent `y` assembly) + engine create-per-load /
> clear-on-LoRA wiring. Verified: offline unit tests (LRU + SD/SDXL hit/miss/
> y-reassembly, `test_cond_cache.py`); Anima real-weights (hit skips encode+adapter,
> new negative misses, warm==cold bit-identical); SDXL real-weights (default path
> unchanged, warm==cold bit-identical, cross-resolution context reuse). GPU e2e
> timing (verification #4) still worth a look but the correctness bar is met.
>
> Written 2026-07-08 from the perf review that also removed the per-step
> LLM-Adapter re-run and the per-forward RoPE H2D copy (`_anima.py` /
> `anima_dit.py`).

## Context

Every generation re-tokenizes and re-encodes the prompt *and* the negative
prompt, even when both are unchanged — the dominant workflow (seed hunting,
X/Y/Z sweeps, batch runs) repeats the exact same conditioning dozens of times.
With any offload mode the cost is not just the encoder forward: `staged()`
shuttles the text encoder(s) over PCIe both ways per image
(`_anima.py:170`, `_base.py:157`, and the FLUX equivalent). Per image that is:

- **Anima** (`offload=encoders`, the 12 GB default): ~1.2 GB Qwen3-0.6B H2D +
  encode + D2H + `empty_cache` — a few tenths of a second.
- **FLUX**: T5-XXL staging — seconds per image. The biggest beneficiary.
- **SD/SDXL**: CLIP staging + encode — small but free to skip.

A small LRU keyed on the prompt pair makes repeat-prompt generations skip the
whole conditioning stage, including the encoder staging itself.

## Design

**Mechanism in the submodule, policy in the backend.** diffucore is a library;
it gets an optional cache object and consults it. The engine (which owns model
lifecycle and LoRA state) creates it and invalidates it.

### New: `diffucore/src/diffucore/runtime/cond_cache.py`

```python
class ConditioningCache:
    """LRU of prompt → conditioning tensors. Entries are stored on CPU
    (a few MB each) and moved to the compute device on hit."""
    def __init__(self, max_entries: int = 16): ...
    def get(self, key: tuple) -> dict | None      # returns tensors .to(device) by caller
    def put(self, key: tuple, value: dict) -> None
    def clear(self) -> None
```

- `ModelBundle` gains `cond_cache: ConditioningCache | None = None`.
  `None` (default) = exactly today's behavior, so all existing tests and
  direct-library users are untouched.
- Stored on CPU: VRAM-neutral; the move-on-hit is ~1 MB (Anima) to ~40 MB
  (FLUX T5 context), negligible next to the staging it replaces.

### What each family caches

| Family | Key (beyond prompt/negative) | Value |
|---|---|---|
| Anima | — | **post-adapter** `cond_ctx`/`uncond_ctx` (1024-d, L=512, fp16) |
| SD1.5 | clip_skip | `context` cond/uncond |
| SDXL | — | `context` + `pooled` cond/uncond (NOT `y` — time_ids depend on width/height; keep the cheap `_sdxl_y` assembly outside the cache) |
| FLUX | — | T5 + CLIP embeds |

Anima caching the *post-adapter* context matters: it also skips the
once-per-generation `preprocess_text_embeds` call, and it means a hit needs
neither the TE **nor** an early backbone touch. Wrinkle: the adapter weights
live inside the DiT, so the Anima value is invalid across DiT swaps — handled
by cache lifetime (below), not by key.

### Cache consultation (pipeline side)

In `_anima.py` (t2i + i2i; leave the two calibrate paths uncached — they run
once), `_base._encode_prompts`, and the FLUX pipeline:

```python
key = (prompt, negative_prompt)           # + family-specific fields per table
cached = cache.get(key) if cache else None
if cached is None:
    with staged([...text encoders...], device, policy.offload_idle):
        ... encode as today ...
    if cache: cache.put(key, value_on_cpu)
```

The check must sit **before** `staged()` so a hit never stages the encoder.
For Anima the adapter half of the value is produced inside the backbone
`staged()` block (where it is computed today); a partial-hit design is not
worth it — cache the whole thing or recompute the whole thing.

### Lifetime & invalidation (engine side, `backend/engine.py`)

- Create one `ConditioningCache` per successful load and hang it on the bundle;
  a (re)load or checkpoint-cache restore replaces it → stale-across-models is
  structurally impossible. TE/DiT swap on the Anima three-file load is just a
  load → same story.
- **LoRA**: `clear()` on every apply/remove/strength change. LoRAs can patch
  text encoders (SD) and the LLM-Adapter inside the Anima DiT, and diffing
  which modules a given LoRA touched is not worth the fragility — clear
  unconditionally.
- Concurrency: the job worker is a single FIFO consumer; no locking needed.

### Out of scope

- Persisting to disk (embeds are model-coupled; cheap to recompute once).
- Caching across LoRA states (key explosion for a rare workflow).
- The XYZ prompt axis benefits automatically (each distinct prompt is one entry;
  LRU 16 covers typical grids — bump if someone sweeps more prompts than that).

## Verification

1. Unit (tiny stub bundle, counting encoder — same monkeypatch pattern as
   `test_adapter_runs_once_per_branch`): second generation with the same
   prompts runs **zero** TE forwards and zero adapter calls (Anima); different
   negative → miss; LRU evicts at capacity.
2. Bit-identity: fixed-seed image hash equal with cache cold vs warm (the
   cached tensors are the same values — fp16 CPU round-trip is lossless).
3. Engine: LoRA apply → next gen re-encodes (counting test); model reload →
   fresh cache.
4. GPU: time 4 consecutive same-prompt 1024² Anima gens with `offload=encoders`,
   expect gens 2–4 to drop the `[load]`-style TE staging pause; FLUX same at
   whatever fits.

## Estimated effort

Submodule: cache class + 3 pipeline touch points + tests. Backend: create/clear
wiring (~20 lines). ~half a day including the GPU check.
