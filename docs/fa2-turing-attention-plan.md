# FlashAttention-2-for-Turing attention swap — spike plan

> Status: **shipped (2026-07-08).** All gates passed:
> - Phase 0: builds on torch 2.12+cu130 after two `setup.py` tweaks for GCC 15+
>   hosts (`-std=c++20` — torch headers need C++20 optional-typename in the
>   nvcc host pass — and `-Xcompiler -fpermissive`). Pinned commit `52a67d7`.
> - Phase 1: max-abs error vs fp32 reference within 0.86–1.13× of SDPA's own
>   error on every shape, incl. cross-attn (Lq≠Lkv works) and odd L.
> - Phase 2: ×1.44–1.64 self-attn D=128, ×1.3–1.4 cross-attn, ×1.27–1.36 D=64.
> - Phase 4 (2060, Anima 1024², 20 steps, CFG, seed 42): ×1.083 default config,
>   **×1.103 with fp16_accumulation on** (the realistic fast config; the saved
>   attention time is constant, so the relative win grows as GEMMs shrink) —
>   images visually identical. Kernel delivered exactly its microbench win
>   (177 ms/step measured vs 173 ms predicted).
>
> Integration: `models/_attention.py` dispatch + `DevicePolicy.attention`
> (`"sdpa"` default bit-exact / `"auto"` / `"fa2_turing"`), stamped at load;
> "fa2 attn" LoadPayload chip gated on `fa2_available`. Never engages off
> sm75 — sm80+ keeps native flash SDPA (not Turing-locked). fa2+compile
> rejected at submit. Wheel installed in the app venv; not in requirements.

## Context

PyTorch's flash SDPA backend requires sm80+; on the RTX 2060 (sm75) every DiT
attention call falls back to the mem-efficient (CUTLASS) kernel. At 1024²
Anima runs self-attention over L=4096 tokens, head_dim=128 — roughly a third
of per-block FLOPs — so a faster attention kernel is the single largest
remaining raw-speed lever (~10–20% e2e if the kernel delivers).

Candidate: **[ssiu/flash-attention-turing](https://github.com/ssiu/flash-attention-turing)**
— a community FA2 port for sm75. fp16 only, head_dim ∈ {64, 128}, forward
claims up to ~2.2× over PyTorch attention on Turing. CUDA/CUTLASS source, so
the Triton int8-lowering bug that killed SageAttention does **not** apply.
Caveats: no releases, tested against torch 2.5.1/2.8.0 + CUDA 12.4 (venv has
torch 2.12) — extension-ABI drift is the main build risk.

Applicability by family (dispatch by head_dim, per call site):

| Backbone | head_dim | eligible |
|---|---|---|
| Anima DiT (`anima_dit.py:293`) | 128 | yes (self + cross attn) |
| FLUX DiT (`flux_dit.py:208`) | 128 | yes |
| SD1.5 UNet | 40/80/160 | no — SDPA fallback |
| SDXL UNet | 64 | maybe (phase 5, only if Anima pans out) |

## Phases and gates

**Phase 0 — build.** Kill the diffucore-ui server (holds VRAM). Clone the repo,
build the wheel with the venv's torch (`.venv/bin/pip install --no-build-isolation .`),
matching `torch.version.cuda`. Keep the build in scratchpad; nothing lands in
requirements. *Gate:* builds and `flash_attn_func` imports on torch 2.12.
If it needs an older torch: **stop** (same dead end as SageAttention; do not
downgrade torch).

**Phase 1 — correctness harness** (scratchpad script). Compare FA2-Turing and
SDPA-mem-efficient each against an fp32 math reference (not against each
other) on real shapes: fp16, B∈{1,2}, H=16, D=128; self-attn L∈{1024, 4096,
9216} (512²/1024²/1536²); cross-attn Lq∈{1024,4096} × Lkv=512 — confirm the
fork supports Lq≠Lkv at all. *Gate:* max abs error vs fp32 reference within
~2× of SDPA's own error. Also probe empty/odd L (the ÷64 snap makes L regular,
but the detailer crops vary).

**Phase 2 — microbench.** CUDA-event timing, 100 iters post-warmup, same
shapes, vs SDPA. *Gate:* ≥1.3× on the L=4096 self-attn shape, else the e2e
win can't clear ~10% and the integration isn't worth the dependency.

**Phase 3 — integration.**
- New `diffucore/src/diffucore/models/_attention.py`: a `dispatch_sdpa(q, k, v)`
  used by the two DiT call sites. Import-guarded — the package is optional and
  its absence must be silent (SDPA path unchanged, bit-exact default).
- FA2 takes `(B, L, H, D)`; the DiTs already hold that layout *before* the
  `.transpose(1, 2)` they do for SDPA — the FA2 branch skips both transposes
  (small bonus win; keep them on the SDPA branch).
- `DevicePolicy.attention: "auto" | "sdpa" | "fa2_turing"` (default `"sdpa"`
  = today, bit-exact; `"auto"` picks fa2 iff importable ∧ CUDA ∧ fp16 ∧
  head_dim∈{64,128}). Surface as a LoadPayload perf chip exactly like
  `fp16_accumulation` (opt-in, not bit-exact).
- **compile/cuda_graphs interplay:** an unregistered custom op graph-breaks in
  every one of the 28 blocks, which defeats compile. First cut: reject
  `attention="fa2_turing"` + `compile=True` at load with a clear error
  (mirror the TeaCache+graphs guard). Registering it as a proper
  `torch.library` custom op (with a meta/fake impl) is a follow-up only if
  the eager win is confirmed.

**Phase 4 — e2e A/B on the 2060.** Fixed seed, Anima 1024², 20 steps, CFG on:
wall-time + side-by-side images (outputs will NOT be bit-identical — different
accumulation order — so judge visually, as with fp16_accumulation). *Gate:*
≥10% e2e faster and no visible quality change, else revert to a documented
dead end (memory note, like SageAttention).

**Phase 5 (optional).** Extend dispatch to SDXL's head_dim-64 attention;
re-run phase 1/2 at SDXL shapes first.

## Distribution

Local-build optional extra only. Setup scripts unchanged — the default install
must never require nvcc. Document in GUIDE.md as an advanced knob if shipped.

## Risks

- Unmaintained fork, no releases — pin a commit hash; vendor the wheel to
  scratchpad only until proven.
- fp16-only: fine (Anima/FLUX policy is fp16 on this card; bf16 is slow on
  Turing anyway).
- Backward pass not needed (inference only) — ignore its limitations.
- If Lq≠Lkv is unsupported, self-attn-only dispatch still covers ~⅔ of
  attention FLOPs; re-run the phase-2 math before deciding.
