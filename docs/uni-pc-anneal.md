# `uni_pc_anneal`: a σ-annealed-stochastic UniPC sampler for rectified-flow models

*Status: implemented in diffucore (`sample_uni_pc_anneal`), offline-green, with a
local characterization A/B on Anima (RTX 2060). This document states only what the
measurements support.*

## Summary

`uni_pc_anneal` is the **stochastic sibling of UniPC** for rectified-flow (CONST)
diffusion models such as **Anima**. It runs the UniPC unified predictor-corrector
multistep solver and adds a **σ-annealed ancestral noise** term
(`η_i = η_max·σ_i`): noise confined to the high-σ part of the trajectory, decaying
to zero as σ→0. At `η_max = 0` it reduces **bit-for-bit** to deterministic UniPC;
for small `η_max` it injects a controlled amount of stochasticity on top of
UniPC's high-accuracy drift — a principled middle ground between fully
deterministic `uni_pc` and the fully stochastic `er_sde` default.

It is implemented as the natural generalization of diffucore's `dpmpp_2m_anneal`
(swap the DPM++(2M) core for UniPC's higher-order predictor-corrector) and folds
the stochasticity in exactly as `dpmpp_2m_sde` does.

## 1. What it is

A rectified-flow / CONST model has σ ∈ (0, 1], data-prediction scaling
α_t = 1 − σ_t, half-logSNR λ_t = log(α_t / σ_t); the denoiser returns the x0
estimate. UniPC (Zhao et al., 2023) integrates the probability-flow ODE in λ space
as a predictor-corrector multistep (see `_uni_pc_bh_update` / `sample_uni_pc` in
`diffucore/src/diffucore/sampling/samplers.py`).

`uni_pc_anneal` keeps every model evaluation on the schedule σ's (so the multistep
history stays consistent) and folds the annealed ancestral noise into the
exponential weights, **exactly as `dpmpp_2m_sde` folds `eta` into the 2M
multistep**. Per step σ → σ_next, with the annealed fraction η = η_max · σ and
h = λ_t − λ_s, define hh = −h·(1 + η). Then:

- the predictor/corrector **phi / B(h) weights use `hh`** instead of `−h`
  (the step-size ratios `r_k` stay on `h` — pure geometry);
- the first-order **carry is contracted** by `e^{−h·η}`;
- after the corrector, **ancestral noise** of std `σ_t·sqrt(−expm1(−2·h·η))·s_noise`
  is re-injected (for σ_next > 0).

### Exact degradations (unit-tested)

- **`η_max = 0` ⇒ deterministic UniPC, bit-for-bit** — η = 0 ⇒ hh = −h, carry
  factor `e^0 = 1`, noise std `sqrt(−expm1(0)) = 0`
  (`test_uni_pc_anneal_eta_max_zero_equals_uni_pc_bh2`).
- **Constant-x0 ⇒ lands on the clean target** (`..._constant_x0_ends_clean`).
- **Rectified-flow only** — raises for `model_type="ve"` (`..._flow_only`).
- **Seed-reproducible and genuinely stochastic** for `η_max > 0`
  (`..._seed_reproducible_and_stochastic`).

## 2. Strengths

1. **A strict generalization of UniPC.** `η_max = 0` recovers deterministic UniPC
   exactly (proven, test-pinned), so `uni_pc_anneal ⊇ uni_pc`: nothing is lost
   relative to the deterministic solver, and a stochastic dial is gained.
2. **Stochastic robustness on the UniPC core.** Re-injecting ancestral noise is
   the mechanism that makes `er_sde` the robust default on merged velocity fields
   (it lets per-step inconsistencies average out). `uni_pc_anneal` brings that
   mechanism to UniPC's higher-accuracy predictor-corrector drift, rather than to
   a first-order Euler core — a principled middle ground between deterministic
   `uni_pc` and fully-stochastic `er_sde`.
3. **σ-confined noise.** Because η = η_max·σ, the stochasticity lives only at high
   σ and vanishes as σ→0, so low-σ detail is preserved — unlike a constant-η
   ancestral step, which keeps injecting noise into the detail-forming steps.
4. **Clean, auditable construction.** The η-fold reuses the exact, already-tested
   exponential-integrator machinery (`_uni_pc_bh_update`, the `dpmpp_2m_sde`
   noise terms); the only new state is the per-step η.

## 3. The order-ramp, and why `eta_max` is small here

UniPC's higher-order predictor builds its extrapolation from divided differences of
the x0 history, which *amplify* the injected ancestral noise — so the high-order
core is markedly more noise-sensitive than the Euler / DPM++(2M) cores. Two design
responses follow:

1. **σ-tied order-ramp-up.** While η > 0, the predictor order is held near 1 (a
   noise-robust first-order exponential step, no divided-difference residual) at
   high σ, rising toward `order` as σ→0 where the noise has annealed away and the
   x0 history is clean. The ramp is gated on η > 0, so the `eta_max = 0`
   degradation to deterministic UniPC is preserved bit-for-bit. Because
   η/η_max = σ, the ramp tracks position on the trajectory, independent of the
   noise scale. The §5 sweep confirms it works as intended: *without* it the image
   collapses to a washed-out ghost at `eta_max = 1.0`; *with* it, `eta_max = 1.0`
   stays a coherent (if soft) image — the amplification mechanism is real and the
   ramp neutralizes its catastrophic failure.

2. **A low default.** Even with the ramp, quality is highest near `eta_max = 0`
   (deterministic UniPC is the cleanest), and the ramp's order-capping costs a
   little drift accuracy at low η. So `uni_pc_anneal` **defaults to
   `eta_max = 0.2`** and is **not** wired to the shared `eta_max` settings-panel
   knob (whose 1.0 default over-softens this sampler). Treat `eta_max` as a small
   stochastic-diversity dial with the order-ramp as a safety net against
   over-cranking — not a high-σ burn-in to crank for quality. Pair with `beta_mix`
   / `beta` / `flow`, like its siblings.

## 4. Relation to prior work

- **UniPC** (Zhao et al., NeurIPS 2023, arXiv:2302.04867) — the deterministic
  predictor-corrector core. `uni_pc_anneal` is UniPC at `η_max = 0`.
- **`dpmpp_2m_anneal`** (this engine) — same σ-annealed mechanism, DPM++(2M) core;
  `uni_pc_anneal` swaps in the higher-order UniPC predictor-corrector.
- **DPM-Solver++(2M) SDE** (Lu et al., arXiv:2211.01095) — the η-fold mechanism
  (`hh = −h(1+η)`, carry `e^{−hη}`, noise `sqrt(−expm1(−2hη))`) is taken from here.
- **ER-SDE-Solver** (Cui et al., arXiv:2309.06169) — the engine's stochastic
  default; `uni_pc_anneal` differs by a predictor-corrector drift and an explicitly
  σ-annealed-to-zero noise level.
- **SA-Solver** (Xue et al., arXiv:2309.05019) — a stochastic predictor-corrector
  (stochastic Adams) with constant/scheduled noise; it does not anneal noise to
  zero at low σ (and underperformed in this engine, so it was removed).
- **Restart Sampling** (Xu et al., arXiv:2306.14878) — alternates ODE segments with
  discrete noise restarts; `uni_pc_anneal` realizes the same SDE-error-contraction
  idea as a continuous per-step annealed re-noise, at no extra NFE.

**Novelty of the combination.** UniPC is deterministic; SA-Solver uses non-annealed
noise; the engine's `*_anneal` family applies σ-annealed noise to lower-order
cores. A UniPC-order predictor-corrector drift with a σ-annealed-to-zero ancestral
noise schedule, in the rectified-flow half-logSNR parameterization, is new to this
engine and does not correspond to a published method we could find. It is offered
as a **principled, tunable stochastic UniPC**, not on the basis of a benchmarked
quality win.

## 5. Local characterization (Anima, RTX 2060, fp16, 1024², CFG 4, seed 1234)

Two danbooru-tag prompts; `er_sde + simple + 30` as the reference point;
deterministic `uni_pc + beta_mix + 16` as the deterministic-core reference;
`uni_pc_anneal + beta_mix` swept over `eta_max` and steps. Metric: wall-clock + a
cv2 Laplacian-variance sharpness proxy (no LPIPS/CLIP available locally). Images and
montages in `outputs/ab_uni_pc_anneal/`.

| config | steps | eta_max | wall (s) | sharpness | note |
|---|---|---|---|---|---|
| er_sde / simple (reference) | 30 | — | 71–85 | 1442 / 2067 | coherent baseline |
| uni_pc / beta_mix (det. core) | 16 | 0.0 | ~41 | 795–1246 | coherent |
| uni_pc_anneal / beta_mix | 16 | 0.00 | ~41–55 | 2331 | == det. UniPC, sharpest |
| uni_pc_anneal / beta_mix | 16 | 0.15 | ~41 | 1887 | coherent |
| uni_pc_anneal / beta_mix | 16 | 0.30 | ~41 | 1932 | coherent |
| uni_pc_anneal / beta_mix | 16 | 0.50 | ~40 | 1423 | softening |
| uni_pc_anneal / beta_mix | 16 | 1.00 | ~41 | 251–445 | **washed out** |
| uni_pc_anneal / beta_mix | 24 | 0.30 | ~58 | 1781 | coherent |

### 5.1 `eta_max` sweep, before vs after the order-ramp

`uni_pc_anneal + beta_mix + 16`, kirakishou prompt, seed 1234, sharpness proxy:

| eta_max | no ramp | with ramp | note |
|---|---|---|---|
| 0.00 | 2331 | 2331 | identical — ramp inactive at η = 0 (degradation intact) |
| 0.30 | 1932 | 1514 | both coherent; ramp slightly softer (order-cap costs accuracy) |
| 0.50 | 1423 | 1126 | coherent |
| 0.75 | — | 1069 | coherent |
| 1.00 | **251 (washed-out ghost)** | **681 (coherent, soft)** | ramp neutralizes the collapse |

**Findings, stated honestly.**

- **The order-ramp works and validates the mechanism.** Without it, `eta_max = 1.0`
  collapses to a ghost (sharpness 251); with it, the same setting produces a
  coherent image (681). The high-order-amplifies-noise hypothesis is confirmed, and
  the ramp removes the catastrophic failure mode — the safe `eta_max` range widens
  from `≲ 0.3` to the whole `[0, 1]` interval (no collapse anywhere).
- **It is not a quality win, though.** Quality (by the sharpness proxy and visually)
  is still monotone-decreasing in `eta_max`; deterministic `uni_pc` (η = 0) remains
  the sharpest and cleanest, and the ramp even softens low-η slightly. The ancestral
  stochasticity widens robustness, it does not improve fidelity at these step counts.
- Hence the low baked-in default (`eta_max = 0.2`) and the decoupling from the shared
  panel knob. **No SOTA claim is made.**

*Limitations / future work.* The intended benefit of ancestral stochasticity
(averaging out *merge* artifacts) would show up as fewer *structural* failures on
*hard* prompts/seeds — which this small grid (well-behaved prompts, one seed, a
sharpness proxy) does not probe. A fair test needs adversarial prompts/seeds where
deterministic UniPC visibly fails, a perceptual metric (LPIPS/CLIP), and seed sweeps
on heavier GPUs. The order-ramp is now in place as the enabling mechanism for that
study (it makes higher `eta_max` usable without collapse).

## References

1. Zhao et al. "UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of
   Diffusion Models." NeurIPS 2023. arXiv:2302.04867.
2. Lu et al. "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion
   Probabilistic Models." arXiv:2211.01095.
3. Cui et al. "Elucidating the solution space of extended reverse-time SDE for
   diffusion models" (ER-SDE-Solver). arXiv:2309.06169.
4. Xue et al. "SA-Solver: Stochastic Adams Solver for Fast Sampling of Diffusion
   Models." NeurIPS 2023. arXiv:2309.05019.
5. Xu et al. "Restart Sampling for Improving Generative Processes." NeurIPS 2023.
   arXiv:2306.14878.
