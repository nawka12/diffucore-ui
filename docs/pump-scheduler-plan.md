# `pump_dual`: a two-band scheduler for `cogent3_pump`

*Status: design + prototype implemented (`pump_dual_schedule` in
`diffucore/src/diffucore/sampling/schedules.py`, unit-tested, wired into the
Anima dropdown). Not yet A/B'd on images. **Revision 2 (2026-08-15):** the
first version ("coarsen", coarse pumped band) lost prompt coherency in the
user's A/B; §1.1 and §3 record what the failure taught and the corrected
design. This document states the research — offline measurements from this
repo, online findings — and the design that falls out of it, including the
things that did not work.*

> **Revision 3 (2026-08-16) supersedes §1.3, §3.1 and §3.2 below.** The v2
> refinement band was measured and is wrong: running to the σ table floor
> costs the exponential core 2.6× on this repo's own benchmark, and the
> "finer last λ-step ⇒ detail" rationale in §3.1 is *backwards*. Shipped
> defaults are now `pump_share = 0.85` and a terminus at `flow`'s
> `σ(t = 1/steps)`. See §7; the rest of the document is kept as the record of
> how the design got here.

## Summary

`cogent3_pump` is `cogent3` (a measured-gate 3rd-order exponential integrator)
plus a high-σ structure-coherence pump: noise scaled by `1 − C` injected after
each step in the band `sigma_frac ∈ (0.45, 1]`, forcing the next CFG-guided
model call to re-answer "what belongs here" — one **re-deciding round** for
the prompt per injected step. The measured constraint is a step *count*:

| scheduler (32 steps) | pump injections | per-injection λ-step (median) |
|---|---|---|
| `flow` | 26 | 0.15 |
| `smoothstep` | 23 | 0.22 |
| `beta_mix` (coherency A/B winner) | 21 | 0.21 |
| `pump_dual` v1 ("coarsen") — **lost coherency** | 13 | 0.38 |
| **`pump_dual` v2 (this design)** | **21** | **0.24** |

Every scheduler the real-image A/B ranked well spends ⅔–⅘ of the run **above**
the pump cutoff with moderate λ-steps; v1 moved steps *out* of the pumped band
into the refinement band (below 0.45 the pump is off), halving the number of
re-deciding rounds and losing coherency.

`pump_dual` v2 is a single uniform-in-`u` grid run through a **piecewise-linear
warp of the half-logSNR coordinate** `λ = −logit(σ)`, with `pump_share` (default
0.65) of the steps in the pumped band (`σ ∈ [pump_end, top_sigma]`, uniform in
λ at ≈ 0.24 λ-steps — `beta_mix`'s measured-winning count *and* step size), a
fixed burn-in first step (`σ_max → top_sigma ≈ 0.99`), and the remaining steps
in the refinement band, also uniform in λ. The tail lands at ≈ 0.52 λ last-step
at 32 steps — between `flow`'s 0.73 and far below `beta_mix`'s 1.90 — so the
gate floor stays low where detail resolves.

## 1. The offline half: what the repo already measures

### 1.1 The mechanism (`GUIDE.md`; `sample_cogent3`'s docstring)

The pump injects `nu = pump_strength · σ_next · ramp(sigma_frac)` with
`ramp = clamp((sigma_frac − 0.45)/0.25, 0, 1)` (full strength above 0.70, off
below 0.45), gated spatially by `1 − C` (structure-tensor coherence of the
*denoised* prediction), on top of a completed exponential-integrator step.
Because it perturbs `x`, the next model call sees a latent noisier than the σ
it is handed and must explain the excess as signal — under CFG, that is a
re-deciding of prompt-adherence in still-ambiguous regions. The real-image A/B
found prompt coherency is the standout win, not the character-stature it was
built to chase.

Three measured constraints on any scheduler:

1. **The pump's value is CFG re-deciding** — the uncond pass must be active in
   the pumped band (don't set a CFG interval that ends inside it).
2. **Coherency scales with the number of injected steps.** `beta_mix` wins
   prompt coherency, `smoothstep` wins fine detail, at 28–32 steps; the
   GUIDE's surviving explanation for the beta_mix-vs-smoothstep gap ("fewer,
   bigger steps, more denoising per injection") is **wrong as a design rule**:
   it was read off a 2-injection difference and extrapolated. v1 of this
   schedule took it literally, halved the injection count to 13, and lost
   coherency on real images. What every well-ranked scheduler actually shares
   is a pumped band carrying ⅔–⅘ of the run with moderate λ-steps (0.13–0.9,
   median ≈ 0.2). The re-deciding rounds, not their size, are the engine.
3. **The pump must not run at low σ** (aether's failure mode: skin and
   gradients to mush). The hard shutoff at 0.45 is what makes the scheduler's
   job two-band instead of one.

### 1.2 What the offline toy says — and does not say

`scripts/ab_cogent3.py --schedulers` (the GMM-flow toy at Anima's shift=3.0)
ran **plain `cogent3`**, not `cogent3_pump`: the toy latent is `[B, 512]`, and
the pump raises `ValueError` on non-4-D latents (structure tensor is a 2-D
convolution). So the toy's verdict — "`flow`/`simple`/`sgm_uniform` are the
safe defaults; `beta`/`beta_mix` are markedly worse, 0.51/0.64 vs 0.115 rough
energy at 24 steps" — is evidence about the **core**, and the image A/B's
`beta_mix` win is evidence about the **pump**. They do not contradict; they
describe the two halves.

The toy also identifies the core's failure mode precisely (cogent.md §6):
`beta`/`beta_mix`/`kl_optimal`/`normal`/`infinity` "leave a **coarser minimum
λ-step** than `flow` (0.24–0.25 vs 0.167 at 24 steps)... it pins the
`1 − e^(−h)` floor high enough that the coherence term can never damp — so on
those schedules `cogent` degenerates toward `dpmpp_2m_anneal`". Measured at 32
steps here, the final λ-step is 1.90 (`beta_mix`) and 1.81 (`beta`) against
0.73 (`flow`).

### 1.3 The λ geometry at 32 steps, shift=3.0 (measured in this repo)

Half-log-SNR `λ = ln((1−σ)/σ)`; the pumped band is σ ≥ 0.45 (λ from the
offset σ_max ≈ 0.99997 at −10.3 to +0.20 — a span of 10.5, matching the
GUIDE's 10.5–10.7), the refinement band is σ ∈ [σ_min ≈ 0.003, 0.45]
(span 5.6).

| scheduler | pumped steps | pumped λ-step | refined steps | refined λ-step | last λ-step |
|---|---|---|---|---|---|
| `flow` | 24 | 0.13–0.73 | 6 | 0.19–0.73 | 0.73 |
| `beta_mix` | 19 | 0.19–0.90 | 11 | 0.21–1.90 | 1.90 |
| `smoothstep` | 22 | 0.19–0.88 | 9 | 0.23–1.37 | 1.37 |
| `pump_dual` v1 (r=1.25) | 11 | 0.38 | 19 | 0.31–0.35 | 0.31 |
| **`pump_dual` v2 (share=0.65)** | **21** | **0.24** | **11** | **0.48–0.52** | **0.52** |

(The v1 row is kept so the failure is on record: its 11 pumped steps — half
the coherency range — were the A/B's reported loss.)

Two extra observations that shaped the design:

- **The winning schedulers pack the pumped band; the loser starves it.** At 32
  steps every well-ranked scheduler fires 21–26 injections (pumped steps);
  v1 fired 13 and lost coherency. Injection count, not per-injection size, is
  the coherency axis.
- **`flow` puts its pumped steps where the model is asleep.** `flow`/`beta`
  spend 24–25 steps at σ ≥ 0.45, but a λ-uniform grid run all the way to
  σ_max is worse: 9 of 32 σ land at ≥ 0.995, where the denoiser is
  σ-invariant (its output at σ=0.9999 and 0.9998 is the same tensor to
  working precision) — the first ~6 model calls would do almost nothing. The
  whole family sidesteps this with a big first step (σ_max → 0.989–0.995);
  `pump_dual` fixes `top_sigma = 0.99` and does the same.
- **The band join is a free lunch.** A uniform-in-λ grid within each band
  means `h` is constant inside the bands — the exact DPM-Solver++ coefficient
  case (`r0 = r1 = 1`) — with one blended step across the join, in the ramp
  region where the pump is already fading.

## 2. The online half: what the literature says

- **U-shaped timestep distributions are right for rectified flow** (Lee et
  al., "Improving the Training of Rectified Flows", NeurIPS 2024,
  arXiv:2405.20320): a U-shaped training-time distribution is a large part of
  why one-round rectified flow competes with distillation at 1–2 NFE. The
  beta-quantile schedules (arXiv:2407.12173, WACV 2025 — "Beta Sampling is All
  You Need") are the *sampling*-side mirror: low-frequency structure changes
  early, high-frequency detail changes late, so steps should cluster at both
  ends. This repo's `beta`/`beta_mix` are exactly that idea — and the toy
  shows the U-shape's *detail end* must not get so dense it leaves a coarse
  minimum λ-step.
- **The shift map is a σ(t) warp, not a sampling strategy** (Esser et al.,
  "Scaling Rectified Flow Transformers", arXiv:2403.03206): Anima's shift=3.0
  concentrates t near σ=1 to match training compute. Schedulers that place
  steps in t (and let the shift map re-place them) inherit that bias in σ
  space *nonlinearly* — which is exactly why `beta`'s endpoint density in t
  becomes a coarse λ-tail after the map (measured in §1.2).
- **Noise scheduling is a first-class lever, and it interacts with the
  solver** (Ting Chen, arXiv:2301.10972: the optimal schedule is
  task/resolution-dependent; shifting log-SNR is the transferable recipe;
  DPM-Solver++ arXiv:2211.01095: high-order multistep cores need fine smooth
  steps where the trajectory curves). The λ coordinate is the exponential
  core's native grid; scheduling in λ rather than t removes the shift map's
  nonlinearity from the solver's view.
- **Model-derived schedules are the asymptote** (Pu et al., "Optimal Stepsize
  for Diffusion Sampling", arXiv:2503.21774): a DP over measured single-step
  errors on a teacher trajectory provably minimizes discretization error.
  This repo already ships `calibrate_oss_schedule` — but its teacher is a
  *plain Euler* trajectory, so the pumped trajectory's error structure (where
  the pump's injections demand immediate correction) is invisible to it.

## 3. The design

### 3.1 `pump_dual_schedule` (implemented, revision 2)

```
λ(u) = λ(top_sigma) + (S_hi/pump_share)·u        for u ≤ pump_share  (pumped band)
λ(u) = λ(pump_end)  + (S_lo/(1−pump_share))·(u − pump_share)          (refinement)
σ_i = sigmoid(−λ(u_i)) on u = linspace(0, 1, steps);  σ_0 = σ_max;  + trailing 0
S_hi = λ(pump_end) − λ(top_sigma)      S_lo = λ(sigma_min) − λ(pump_end)
```

Defaults: `pump_end = 0.45` (the sampler's cutoff), `pump_share = 0.65`,
`top_sigma = 0.99`. Registered as the `pump_dual` flow-table scheduler; in the
Anima dropdown (the UI renders scheduler lists dynamically; no frontend
change was needed). At `pump_share = S_hi/(S_hi + S_lo) ≈ 0.46` the schedule is
bit-for-bit one uniform-in-λ grid.

Properties (all unit-tested in `test_schedules.py`):

- Strictly descending, `σ_0 = 1.0` (pure-noise init), last nonzero σ = the
  table floor, trailing 0 — same contract as every flow table scheduler.
- No σ ≥ 0.995 waste: first step is the family-standard burn-in (≈ 5.7 λ),
  the second σ ≈ 0.985.
- **Injection count in the coherency range**: 18/19/21 pumped steps at
  28/30/32 steps — within ±1 of `beta_mix` (19/20/21) — pinned by test
  `test_pump_dual_injection_count_matches_beta_mix`.
- `pump_share` trades injections for tail fineness monotonically (tested at
  24/32/50 steps): raising it adds re-deciding rounds and coarsens the last
  λ-step; lowering it is the reverse.
- `pump_end` moves the knee (tested at 0.3/0.6).

### 3.2 What the parameters mean, and the knob to A/B first

`pump_share` is the load-bearing parameter: it decides how many of the run's
steps are re-deciding rounds (pumped band) vs detail refinement (tail). The
default 0.65 reproduces `beta_mix`'s injection count and step size at 28–32
steps while keeping a strictly finer tail — the hypothesis is that it inherits
`beta_mix`'s coherency without its detail loss. The image A/B should sweep
`pump_share ∈ {0.6, 0.65, 0.7, 0.75}` at 28–32 steps.

`pump_end` and `top_sigma` are alignment knobs: `pump_end` should track the
sampler's pump cutoff if the user ever changes it; `top_sigma` only matters at
the top, where the model is asleep.

### 3.3 The asymptote: pump-aware OSS

The "maximum potential" endpoint is not a parametric family but a calibrated
one. Extend `calibrate_oss_schedule` to teacher-run the **pumped** trajectory:
at each candidate level, take one `cogent3_pump` step (exponential-integrator
update + pump injection) and measure the deviation from a fine unpumped
reference. The DP then places steps exactly where the pump's perturbations
need immediate correction and where the core's multistep error concentrates —
no hand-tuned share. Cost: one calibration per (model, resolution, shift),
the same deal `oss` already sells. This is the validation-protocol step that
would also *measure* whether the optimal density is uniform across the pumped
band (the parametric design assumes it is).

## 4. Validation protocol

The toy cannot test the pump (4-D requirement), so the protocol has two
halves, and the doc states which evidence each half can give.

**Offline, extended toy.** Reshape the ab_cogent3 GMM toy's 512-d latent to
`[B, 1, 16, 32]` so `cogent3_pump` runs (the structure tensor averages over
~512 elements either way). Add `pump_dual` and `cogent3_pump` to the
`--schedulers` sweep: `flow`, `beta_mix`, `pump_dual(pump_share ∈ {0.6,
0.65, 0.7})` × det RMSE + rough energy distance. Honest expectation: the toy
will say *little* about coherency (it has no prompt dimension) but should
confirm the core half — `pump_dual` ≥ `flow` at 16–32 steps, and never the
`beta_mix`-class tail degradation — and confirm the pump's hard cutoff holds
on the new schedule (σ below 0.45 must see no pump energy).

**Real images (the decisive half).** Reuse the `ab_franken_sampler_sched`
harness at the user's production settings (CFG 4.5, shift 3.0, CFG interval
(0, 0.75), 28–32 steps): `cogent3_pump × {beta_mix, smoothstep, pump_dual
share=0.6, share=0.65, share=0.7, share=0.75}`. Existing composite (sharpness
/ edge / color / high-freq) plus SSIM-vs-32-step-ref for step-count coherence
— and, because the pump's headline win is prompt adherence, a prompt-coherency
read (the repo has no objective proxy for it; the GUIDE's own A/B was human).

## 5. Risks and honest caveats

- **The injection-count rule is inferred from the A/B's *winners*, not
  measured directly.** The v1 failure (13 injections → lost coherency) is the
  user's A/B; the v2 default targets `beta_mix`'s exact count and step size.
  The claim "count, not size, drives coherency" is the best reading of the
  evidence but is not itself A/B-proven — the `pump_share` sweep is the proof
  step.
- **The offline evidence is for `cogent3`, not `cogent3_pump`** (the toy
  cannot run the pump). The tail-fineness fix is grounded in the core's
  measured failure mode; the pumped-band step sizing is grounded in the image
  A/B. Both halves are load-bearing and neither covers the other's regime.
- **`pump_share` interacts with `eta_max`**: step count per band changes the
  ancestral-noise churn per step. The A/B should fix `eta_max` (1.0,
  production default) and sweep `pump_share` before touching both.
- **CFG interval coupling**: `pump_dual` assumes the uncond pass covers the
  pumped band. With the production interval (0, 0.75) — a *step* fraction —
  CFG applies for σ above σ_24, which is 0.025 on `pump_dual` at 32 steps
  (0.29 on `beta_mix`): the entire pumped band sits inside the CFG band
  (verified from the σ runs). A user who shortens the CFG interval into the
  pumped band removes the pump's engine; that is a sampler/CFG interaction,
  not a scheduler bug.

## 6. What is in the tree

- `diffucore/src/diffucore/sampling/schedules.py` — `pump_dual_schedule`,
  registered in `_FLOW_TABLE_SCHEDULERS` and `__all__`, knobs plumbed through
  `flow_table_schedule`.
- `diffucore/tests/test_schedules.py` — 9 tests: endpoints/descent/floor,
  uniform-λ share point, share monotonicity (injections ↑, tail coarsens),
  **injection count ≈ `beta_mix` at 28/30/32 steps** (the coherency fix),
  tail-finer-than-beta_mix, knee placement, knee tracking, no-waste-at-top +
  `top_sigma`, arg validation.
- `diffucore/src/diffucore/pipelines/_anima.py` — `pump_dual` in
  `_ANIMA_SCHEDULERS`.
- `backend/engine.py` — `pump_dual` in `SCHEDULERS_ANIMA` (the UI dropdown
  renders from this list; no frontend change).

## 7. Revision 3 (2026-08-16): the refinement band was the problem

v2 shipped a refinement band that ran to the σ table floor (0.003) on the
argument that a fine final λ-step (0.52 vs `beta_mix`'s 1.90) keeps the
`psi_1` gate's `1 − e^(−h)` floor low so detail can resolve. Measured, that
argument is backwards, and the band cost more than the pump's band gained.

**The controlled experiment.** `scripts/ab_cogent3.py`'s toy, `cogent3` core,
`eta_max=1.0` under the rough model, 5 seeds, 32 steps, moving *only* the
terminus (`pump_share` fixed at 0.65, so the pumped σ are bit-identical —
`torch.equal` on the first 21 returns True):

| terminus | last λ-step | rough energy ↓ |
|---|---|---|
| 0.0882 (`flow`'s) | 0.197 | **0.145** |
| 0.03 | 0.238 | 0.210 |
| 0.01 | 0.238 | 0.287 |
| 0.003 (v2) | 0.517 | 0.365 |
| `flow` baseline | 0.726 | 0.141 |

Monotone in depth, holds across three rough-error fields (τ 0.2–0.35, freq
6–12) and 8–40 steps; at 8 steps v2 is 16× `flow` (3.04 vs 0.18).

**It is the depth, not the step size.** Holding the terminus at 0.003 and
making the *final* step finer makes it worse (0.517 λ → 0.370, 0.241 λ →
0.420, 0.213 λ → 0.438), while `flow`'s far coarser 0.726 λ final step at
σ_end 0.088 is fine. Across the eleven schedulers of `cogent.md` §6, Spearman
ρ against the published rough-32 ranking is **+0.91** for terminal depth and
**+0.94** for final λ-step, but only **+0.46** for the *minimum* λ-step that
section names as the mechanism — and **−0.19** for pumped-step count. §1.2's
reading of the `beta`/`beta_mix` failure mode was the wrong half of the
geometry.

**What shipped.**

- Terminus is now `σ(t = 1/steps)` — `flow`'s own, taken through the view's
  `t_to_sigma`, so it tracks `shift` and the step budget. No step ever lands
  below where `flow` stops.
- `pump_share` default 0.65 → **0.85**. With the flow terminus that beats
  `flow` on both toy metrics at 24–40 steps (rough 0.127 vs 0.141;
  deterministic RMSE 0.0121 vs 0.0156 at 32) while firing **27** pump
  injections against `flow`'s 26 and `beta_mix`'s 21. The uniform-λ point
  drifts with the budget (≈0.77 at 16 steps, 0.69 at 32), so 0.85 is
  pump-dense at every budget. 0.95 collapses — the 2-step tail hits 1.4 λ.
- New degenerate-case branch: when `σ(t = 1/steps)` sits at or above
  `pump_end` (few steps, or a high `shift` — exactly `shift=9, steps=12`),
  there is no refinement band and the run is one uniform-λ grid, pumped end
  to end. The old code produced duplicate sigmas there.

**Caveats that did not change.** The toy's frozen error field (τ=0.35) is far
larger at low σ than a real DiT's, so the *magnitude* above is very likely
exaggerated — on real images `beta_mix` terminates at 0.003 and is the user's
coherency winner. The direction is supported by both the controlled
experiment and the repo's published ranking; the size is not. And §3.2's
`pump_share` sweep is still the decisive experiment — the knob is still not
exposed in the UI, so it needs a code edit to run.

**One premise still unproven, and now cheap to test.** The whole design rests
on injection count driving coherency. `flow` fires 26 injections at 32 steps
and was *never in the coherency A/B*; `smoothstep` fires 23 against
`beta_mix`'s 21 and lost. So the rule is not monotone on the evidence that
exists. A `cogent3_pump × flow` vs `× beta_mix` pair at 32 steps tests it for
the price of two images and no code. (Independently: the Flux community
reports plain `beta` as *the* prompt-adherence scheduler with no pump
involved at all, which would put the coherency win in the σ distribution
rather than in the injections.)
