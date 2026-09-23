# `reprise`: a Restart-contracted UniPC sampler

> **2026-09-23: `reprise` was removed.** On real images at matched model calls
> it never beat plain `cogent3_pump`. `cogent3_pump` + `pump_dual` at 50 steps
> beat `reprise` at 32 steps (also 50 calls) with `eta_max` 1.0, and they tied at
> 0.2. The cause is structural. All the extra calls land in draft passes that
> the next jump throws away (it keeps 0.25·x), so the final image comes from one
> pass at 32-step density. At `eta_max` 1.0 a single pass already replaces ~92 %
> of the band's noise variance, so the restarts mostly repeat what the
> ancestral noise already does. Restart sampling was designed for an ODE inner
> loop. This document is kept as the record.

> **2026-09-22: the shipped `reprise` no longer uses a UniPC core.** The UniPC
> version this document describes rendered visibly over-saturated at CFG 4.5
> ("burnt" hair colour on AnimaFranken-v1.2), and no restart knob closed the gap
> (K8 at 50 NFE still landed above plain `cogent3`). So `reprise` now runs the
> same restart schedule over the `cogent3_pump` core, and the UniPC sampler was
> removed. The band snapping, drafting and forward jump (§2) are unchanged.
> Everything that depends on the core describes the removed version and has not
> been re-run on the new one: the "no per-step noise" rationale (cogent3's
> `eta_max` noise and the pump's grain now stack with the jumps), the NFE table
> (the cogent3 core costs `n + K·⌈band/s⌉`, **K fewer** calls: `beta_mix` 20 → 29,
> 28 → 43), and every offline measurement in §5. `restarts=0` is now bit-for-bit
> `cogent3_pump`, not `uni_pc_bh2`.

*Status: implemented in diffucore (`sample_reprise`), offline-green (unit tests),
benchmarked on the analytically-solvable GMM-flow toy (`scripts/ab_reprise.py`),
**not yet A/B'd on real images** — §6 is the protocol that decides it. Designed
in `reprise-20260913.md`; this document is the shipped version, and where the
implementation departs from that design note it says so. Following the family's
rule (docs/cogent.md, docs/cogent3.md): only what the measurements support,
including what did not work.*

## Summary

`reprise` is a deterministic UniPC (bh2, order 3) sampler that, on reaching the
lower edge of a *structure band* (`sigma_frac` 0.45), re-noises the latent
**exactly along the forward process** back up to the band's upper edge (0.85)
and re-integrates the band deterministically — three times (`restarts=3`). Only
the final pass through the band runs at the schedule's full density; the first
pass and the intermediate re-integrations are cheap *drafts* (every second grid
point). Below 0.45 it is plain UniPC to the end. No per-step ancestral noise
anywhere.

```
σ_max ──drafts──► σ_lo ══jump══► σ_hi ──draft──► σ_lo ══jump══► σ_hi ──draft──► σ_lo ══jump══► σ_hi ──full──► σ_lo ──full──► 0
   [UniPC, fresh history]            [UniPC]                       [UniPC]                     [UniPC, continues to 0]
```

It is a rectified-flow adaptation of Restart sampling (Xu, Liu, Tong, Vahdat,
Kautz, Jaakkola — "Restart Sampling for Improving Generative Processes",
NeurIPS 2023, arXiv:2306.14878), clean-room from the paper, with four
repo-specific changes: the exact rectified-flow forward jump; a UniPC core
(fresh history per segment) instead of Heun; a band snapped to the *user's*
schedule grid so the scheduler keeps control of σ placement; and the draft /
full-pass density split, which is what makes three restarts affordable inside a
normal step budget.

**Why it exists.** It resolves a tension this repo has measured three times: a
high-order deterministic core wants *no* per-step noise (`uni_pc_anneal` washed
out to a ghost at η=1 — docs/uni-pc-anneal.md §5; the cogent gates turned out to
be step-size-floor-driven on 35–100 % of a real schedule —
`COGENT-IMPROVE-IMPLEMENTED.md` §3), but prompt coherency wants many CFG
re-deciding rounds at high σ (`cogent3_pump` + `beta_mix` was the real-image
coherency win, and halving the injection count lost it —
docs/pump-scheduler-plan.md §1.1). Restarts give both: the noise is added
*between* complete ODE segments, so no divided difference ever straddles an
injection, and each restart is a full re-decision of the layout under CFG.
Restart theory says the same thing quantitatively — a forward jump is exact
(zero discretization error) where every SDE step pays one, so the contraction of
accumulated error, including the off-manifold drift CFG causes, is strongest
with a few large jumps between deterministic segments.

## 1. The algorithm

Inputs: the x0 denoiser closure `model(x, σ) → x0`, the initial latent, the
user's σ schedule (descending, trailing 0), `generator`.

| parameter | default | meaning |
|---|---|---|
| `restart_lo` | 0.45 | band lower edge, in the family-invariant `sigma_frac` coordinate (σ on flow, σ/(1+σ) on VE) — the coherence pump's cutoff |
| `restart_hi` | 0.85 | band upper edge |
| `restarts` | 3 | number of re-noise + re-integrate rounds (K) |
| `draft_stride` | 2 | grid stride of the first pass through the band and of the intermediate restart passes; the final pass is always full density |
| `order`, `variant` | 3, `"bh2"` | UniPC core (same as `uni_pc_bh2` / `uni_pc_anneal`) |
| `model_type`, `shift` | `"flow"`, schedule shift | flow half-logSNR map + first-σ offset, as every flow-aware sampler here |

1. **Snap the band to the grid.** `i_hi` = first index with
   `sigma_frac(sigmas[i]) ≤ restart_hi`; `i_lo` = first index with
   `sigma_frac ≤ restart_lo`. If either is missing, `i_hi ≥ i_lo`, or `i_lo` is
   the terminal 0 → **no restart is possible**: run `sample_uni_pc` on the whole
   schedule and return (bit-identical to `uni_pc_bh2`). This is the
   img2img/inpaint path whenever the sliced schedule starts at or below
   `restart_lo`.
2. **Band and segments.** `band = sigmas[i_hi : i_lo+1]`; `strided(band, s)`
   keeps indices `0, s, 2s, …` plus the last point (both edges always present).
   - `seg_0 = cat(sigmas[:i_hi], strided(band, draft_stride))` — σ_max → σ_lo,
     drafting the band;
   - `seg_k = strided(band, draft_stride)` for `k = 1 … restarts−1`;
   - `seg_last = sigmas[i_hi:]` — the final full-density pass through the band,
     **continuing without a history reset all the way to 0**.
3. **Integrate `seg_0`** with `sample_uni_pc(…, lower_order_final=False)` (the
   segment does not end at 0, so no order ramp-down).
4. **For each restart segment**: jump, then integrate.
   - **Jump** σ_lo → σ_hi, the exact forward-process transition of the
     rectified-flow path:
     ```
     a = (1 − σ_hi) / (1 − σ_lo)                      (flow;  a = 1 on VE)
     b = sqrt(σ_hi² − a²·σ_lo²)
     x ← a·x + b·ε,   ε ~ N(0, I)
     ```
     the result has the marginal at σ_hi given the same x0. After the jump only
     `(1 − σ_hi)·x0 ≈ 0.15·x0` of the draft survives as signal — the layout is
     *seeded* by the draft, not kept (that is the point).
   - **Integrate** with a *fresh* `sample_uni_pc` call (history reset — the
     pre-jump x0 history is stale by construction). Intermediate segments use
     `lower_order_final=False`; the last uses the caller's value (default True)
     and lands on the x0 estimate at σ=0.
5. **Callback / NFE.** The inner UniPC calls emit the callback once per model
   evaluation; the wrapper re-indexes them onto one running counter, so a
   progress bar advances once per model call across the whole run.

## 2. Why each choice

- **Deterministic UniPC core, no η.** UniPC bh2 is the cleanest deterministic
  solver this repo has measured on real Anima images (the eta-sweep verdict) and
  the toy's best at 24–32 NFE; keeping every segment fully deterministic means
  the corrector/predictor never sees an injection inside its history window.
  `restarts=0` *is* `uni_pc_bh2`.
- **Exact forward jump, not an ancestral split.** Restart's contraction argument
  needs the re-noised latent to sit on the forward marginal at σ_hi. Written as
  above there is no η bookkeeping to get wrong: `a²σ_lo² + b² = σ_hi²` exactly
  (unit-pinned).
- **Band 0.45–0.85.** Lower edge = the pump's measured cutoff; the toy confirms
  0.35 and 0.55 are no better. Upper edge: bands capped at 0.70 do nothing for
  CFG error (too much of the draft's layout survives the re-noise); 0.90/0.95
  waste draft steps where the model is σ-invariant. In flow-time 0.45–0.85 is
  t ∈ [0.21, 0.65]: the middle of the trajectory, where the structure is visible
  enough to be judged and still cheap to change.
- **Grid-snapped band.** The restart re-integrates the *user's* grid points, so
  `beta_mix` / `smoothstep` / `flow` keep control of σ placement, the NFE is an
  integer function of the schedule, and no new scheduler is needed.
- **Drafts (stride 2) + one full final pass.** The departure from Xu et al., who
  run every segment at the main step size. At matched NFE the first pass and the
  intermediate passes are re-noised anyway, so their only job is a good-enough
  seed; spending full density there is the wrong trade. One-step "drafts" are
  too crude — the seed must be a real integration, just a coarse one.
- **Three restarts.** K is the coherency-vs-softness dial: ED to the CFG target
  keeps improving with K, but the sample spread starts to *under*-shoot the
  truth from K4 on — the same over-contraction the per-step-noise samplers show,
  i.e. the wash-out direction — and exact-model accuracy leaves the floor. K3 is
  the knee on all three metrics simultaneously.
- **Final pass continues to 0 without reset.** The last restart's descent through
  the band and the low-σ detail phase are one continuous ODE, so UniPC keeps its
  3rd-order history across σ_lo; only the K jumps reset it.
- **No per-step noise knob, no coherence gating of the jump.** The pump's
  `(1 − C)` spatial weighting was considered and rejected for a *large* jump: a
  spatially varying re-noise level puts the latent off the forward marginal the
  model is told it is at, which the pump's small amplitudes get away with and a
  0.8-std jump would not. Isotropic keeps the theory intact.

## 3. Degradation invariants (pinned by tests)

| pin | result | test |
|---|---|---|
| `restarts=0` | bit-for-bit `sample_uni_pc(variant="bh2", order=3)`, no noise drawn | `test_reprise_restarts_zero_is_uni_pc_bh2` |
| no grid point in the band (img2img start ≤ `restart_lo`, or a band the grid steps over) | same, and seed-independent | `test_reprise_band_absent_is_deterministic`, `..._band_off_grid_is_uni_pc_bh2` |
| constant-x0 denoiser | lands exactly on the target | `test_reprise_constant_x0_ends_clean` |
| same `generator` seed | bit-identical; different seed ⇒ different output | `test_reprise_seed_reproducible_and_stochastic` |
| the jump | `a²σ_lo² + b² = σ_hi²` (1e-12, float64); `a = (1−σ_hi)/(1−σ_lo)` on flow, `a = 1` on VE; post-jump variance check on a real forward-marginal latent | `test_reprise_jump_preserves_forward_marginal` |
| NFE | model calls == callbacks == `reprise_nfe(sigmas, …)`, callbacks a single contiguous run | `test_reprise_nfe_matches_model_calls_and_callbacks` |
| segment geometry | `seg_0` starts at σ_max and ends at `sigmas[i_lo]`; every restart segment starts at `sigmas[i_hi]`; the last is `sigmas[i_hi:]` verbatim; `i_hi`/`i_lo` are the *first* indices at or below the thresholds | `test_reprise_segments_geometry` |
| solve structure | exactly `restarts + 1` inner `sample_uni_pc` calls, `lower_order_final` False on all but the last | `test_reprise_final_pass_is_one_continuous_solve` |
| VE | finite, lands clean on a karras schedule (the core is family-agnostic) | `test_reprise_ve_finite_and_lands_clean` |

## 4. Cost (NFE) — and a correction to the design note

```
NFE = n + K·(⌈band/s⌉ + 1)        n = len(sigmas) − 1,  band = i_lo − i_hi
                                  s = draft_stride,      K = restarts
```

**The `+ 1` per restart is a correction to `reprise-20260913.md` §2.4**, which
gave `Σ (len(seg) − 1)` and so undercounted by exactly K. Each restart segment
begins by evaluating x0 at the freshly re-noised latent — the jump reset the
history, so there is nothing to reuse — and that is a real model call.
`sample_uni_pc` makes one call per σ in a run, minus one when the run ends on
the terminal σ=0 (which it lands on from the history), so a segment that does
*not* end at 0 costs `len(seg)`, not `len(seg) − 1`. Verified against the real
call and callback counts (`test_reprise_nfe_matches_model_calls_and_callbacks`).
The segment *geometry* in the design note is exactly right; only the arithmetic
over it was off.

True NFE at the defaults on the production schedulers (shift 3.0):

| nominal steps | 8 | 12 | 16 | 20 | 21 | 22 | 24 | 28 | 32 |
|---|---|---|---|---|---|---|---|---|---|
| `beta_mix` | 14 | 21 | 28 | **32** | 33 | 37 | 39 | 46 | 50 |
| `flow` | 17 | 24 | 31 | 38 | 39 | 40 | 42 | 49 | 56 |
| `smoothstep` | 14 | 21 | 28 | 35 | 33 | 37 | 39 | 43 | 50 |
| `beta` | 17 | 21 | 28 | **32** | 36 | 37 | 39 | 46 | 50 |

So the parity point for the user's 32-NFE production run is **`reprise ×
beta_mix × 20` = 32 NFE exactly** (the design note said 21 steps / 30 NFE; that
same run really costs 33). On `flow` the parity point is 17 steps, on
`smoothstep` 19 (31 NFE).

Worked example, `beta_mix` N=21 — `i_hi=8` (σ 0.812), `i_lo=14` (σ 0.408),
band 6 intervals:

```
seg0 (12 calls): 1.000 0.991 0.978 0.962 0.942 0.918 0.889 0.854 0.812 | 0.703 0.565 0.408   (draft)
jump: x ← 0.318·x + 0.801·ε
seg1 ( 4 calls): 0.812 0.703 0.565 0.408                                                    (draft)
jump
seg2 ( 4 calls): 0.812 0.703 0.565 0.408                                                    (draft)
jump
seg3 (13 calls): 0.812 0.762 0.703 0.637 0.565 0.488 0.408 | 0.326 0.245 0.168 0.096 0.037 0.003 0.000   (full)
```

The pipelines size the progress bar with `reprise_nfe`, not `len(sigmas) − 1`,
at both Anima dispatch sites; without that the bar overshoots 100 % and the SSE
progress fraction reports > 1.

## 5. Offline evidence

Toy: `scripts/ab_reprise.py`, built on `scripts/ab_cogent3.py`'s GMM-flow (512-d,
6 modes, power-law covariance, analytic optimal denoiser, rectified flow at
shift 3.0), 384 samples, 3 seeds, energy distance (ED) to 2048 true samples.
Three regimes:

- **A** exact model — does the sampler distort the data law? A no-harm check:
  the finite-sample ED floor (384 true samples against the 2048) is **0.110**
  and a 4000-step Euler reference scores 0.102, so *everything* in the 0.09–0.11
  band is at the floor and indistinguishable. Only a clear excursion above it
  counts.
- **B** rough model error (`rough_error`, τ=0.35, freq ∈ {3, 6, 12}) — the
  family's proxy for a merged / imperfect velocity field.
- **C** the CFG arm — cond law = modes {0,1}, uncond = all six,
  `x0 = x0_u + w·(x0_c − x0_u)` (CFG on x0 ≡ CFG on v for CONST). Two metrics:
  ED to the *conditional* law, and mean nearest-mode distance against the
  truth's 16.03 — above = overshoot past the modes (CFG over-saturation), below
  = over-contraction toward centres (the wash-out direction). Adherence is 1.000
  for every sampler at w ≥ 2, so it is not reported.

**All comparisons are at matched NFE using the true cost formula of §4.** At a
32-NFE budget on `flow` that gives reprise K3 17 nominal steps against the
baselines' 32; on `beta_mix`, 20. The design note's own harness used the
undercounted formula, so its reprise arms silently ran ~3 model calls *over*
budget; the tables below do not, and the ordering survives.

### 5.1 Head-to-head (GMM seed 1234, 3 schedulers, 32 NFE)

| metric | sched | uni_pc_bh2 | cogent3 η1 | cogent3_pump | secant_anneal | uni_pc_anneal | reprise K2 | **reprise K3** | reprise K4 s3 |
|---|---|---|---|---|---|---|---|---|---|
| A exact (floor 0.110) | flow | 0.1023 | 0.1177 | 0.1087 | 0.1044 | 0.1070 | 0.1033 | **0.0966** | 0.1194 |
| | beta_mix | 0.1023 | 0.1203 | 0.1158 | 0.1021 | 0.1089 | 0.1020 | **0.0920** | 0.1105 |
| | smoothstep | 0.1023 | 0.1172 | 0.1169 | 0.0991 | 0.1100 | 0.1019 | **0.0933** | 0.1110 |
| B freq 3 | flow | 0.3684 | 0.3128 | 0.3104 | 0.3909 | 0.3204 | 0.2548 | 0.2030 | **0.1965** |
| B freq 6 | flow | 0.2202 | 0.1396 | 0.1441 | 0.2096 | 0.1723 | 0.1670 | **0.1313** | 0.1362 |
| B freq 12 | flow | 0.1447 | 0.1078 | 0.1035 | 0.1387 | 0.1129 | 0.1210 | **0.1009** | 0.1126 |
| B freq 6 | beta_mix | 0.4855 | 0.4569 | 0.4612 | 0.4221 | 0.3887 | 0.4303 | 0.3873 | **0.3760** |
| B freq 12 | beta_mix | 0.2603 | 0.3136 | 0.3162 | **0.2021** | 0.1995 | 0.2537 | 0.2270 | 0.2215 |
| C w=2 ED | flow | 0.2537 | 0.1881 | 0.2040 | 0.1702 | 0.2216 | 0.1430 | **0.1321** | 0.1658 |
| C w=4.5 ED | flow | 1.3820 | 0.7078 | 0.7619 | 0.7123 | 1.1645 | 0.5700 | **0.5069** | 0.5928 |
| | beta_mix | 1.3806 | 0.7296 | 0.7576 | 0.7460 | 1.1740 | 0.6398 | **0.5365** | 0.6577 |
| | smoothstep | 1.3805 | 0.7263 | 0.7638 | 0.7503 | 1.1717 | 0.6176 | **0.5543** | 0.6758 |
| C w=7 ED | flow | 2.9474 | 1.3107 | 1.3926 | 1.3585 | 2.3969 | 1.1925 | **1.0361** | 1.1420 |
| C w=4.5 mode-d (truth 16.03) | flow | 16.70 | 15.51 | 15.75 | 16.23 | 16.14 | 16.17 | **15.93** | 15.65 |
| C w=7 mode-d (truth 16.03) | flow | 17.97 | 15.93 | 16.19 | 16.71 | 17.14 | 16.61 | **16.28** | 16.00 |

**The CFG rows are the result.** reprise K3 wins the ED-to-conditional-law
metric in **every** cell measured — 3 guidance scales × 3 schedulers × 3 budgets,
27 of 27 — by 24–28 % against the family's best stochastic samplers at w=4.5 and
by ~61 % against deterministic UniPC. The margin grows with the guidance scale
(w=7: −21 % against cogent3, −65 % against uni_pc_bh2), which is what a
mechanism aimed at CFG's off-manifold drift should do.

The mode-distance column says *how* the samplers are wrong, and it splits the
field cleanly in two: every deterministic solver overshoots past the modes
(uni_pc_bh2 16.70, stork2 16.69 at w=4.5, rising to ~18.0 at w=7 — CFG
over-saturation), and every per-step-noise sampler contracts inside them
(cogent3 15.51, cogent3_pump 15.75 — the wash-out direction). reprise sits
between, closest to the truth at w=7 where the two failure modes separate
most (16.28 vs 15.93/16.71 either side). At w=4.5 the claim has to be weaker
than the design note made it: `secant_anneal` (16.23) and `uni_pc_anneal`
(16.14) are comparably near the truth there — but they reach it with 29–56 %
worse ED, i.e. right spread, wrong distribution.

### 5.2 Where it does *not* win

- **B on `beta_mix` / `smoothstep` with a high-frequency error field.** At freq
  12 on `beta_mix`, `secant_anneal` (0.202) and `uni_pc_anneal` (0.200) beat
  reprise K3 (0.227) at 32 NFE, and at freq 6 / 24 NFE `secant_anneal` (0.292)
  beats it (0.328). Per-step churn is the purpose-built answer to a
  high-frequency error field, and restarts are not it — they re-decide coarse
  layout. reprise still beats `uni_pc_bh2` and both cogent variants in those
  cells, and it leads everywhere on the *coarse* (freq 3) error field, which is
  the one that looks like a merged checkpoint's structural disagreement.
- **A at 24 NFE on `flow`** (0.1139 vs uni_pc_bh2's 0.1029) — the one no-harm
  excursion in the table. At that budget the true cost formula leaves K3 only 12
  nominal steps, below its own "16+ nominal steps" floor. At 28 and 32 NFE it is
  back at the sampling floor. On `beta_mix` and `smoothstep`, where the band is
  narrower in grid terms, it is at the floor at every budget.
- **K4 (stride 3)** buys a little more CFG contraction at high w and a little
  more rough-model robustness, and pays for it in A (0.111–0.147, clearly off
  the floor) and in spread (15.65 at w=4.5, under-shooting like the
  per-step-noise samplers). This is the softness direction the K knob trades in.

### 5.3 Replication on a second toy instance (GMM seed 777, `flow`, 32 NFE)

| metric | uni_pc_bh2 | cogent3 η1 | secant_anneal | reprise K2 | **reprise K3** | reprise K4 s3 |
|---|---|---|---|---|---|---|
| A (floor 0.121, ref 0.117) | 0.1162 | 0.1196 | 0.1049 | 0.1085 | 0.1037 | 0.1151 |
| B freq 6 | 0.2572 | 0.1481 | 0.2222 | 0.1821 | 0.1477 | **0.1394** |
| C w=4.5 ED | 0.6610 | 0.2400 | 0.2509 | 0.2338 | 0.2059 | **0.1781** |
| C w=4.5 mode-d (truth 15.85) | 16.60 | 15.26 | 16.00 | 16.02 | **15.77** | 15.46 |

Same ordering, same knee. (This instance's CFG problem is much easier — every
sampler's ED is 3–5× lower — and K4 edges K3 on the CFG rows there, so K3 vs K4
is instance-dependent at the margin; K3 keeps the accuracy and spread.)

### 5.4 The knobs (GMM seed 1234, `flow`, matched NFE)

Restart count and draft density (`Ks N` = K restarts at stride N; A's floor is
0.110, C's truth spread is 16.03):

| config | A @24/28/32 | B6 @24/28/32 | C w=4.5 ED @24/28/32 | C mode-d @32 |
|---|---|---|---|---|
| K1 s1 (Xu et al. form) | 0.1045 / 0.1023 / 0.1020 | 0.1411 / 0.1610 / 0.1739 | 0.8056 / 0.7923 / 0.8036 | 16.44 |
| K2 s1 (full density) | 0.1146 / 0.1085 / 0.1053 | 0.1128 / 0.1279 / 0.1420 | 0.6735 / 0.5790 / 0.5836 | 16.20 |
| K2 s2 | 0.1072 / 0.1034 / 0.1033 | 0.1325 / 0.1538 / 0.1670 | 0.6673 / 0.6181 / 0.5700 | 16.17 |
| **K3 s2 (default)** | 0.1139 / **0.1005 / 0.0966** | **0.1020** / 0.1238 / 0.1313 | **0.6406 / 0.5430 / 0.5069** | **15.93** |
| K3 s3 | **0.1062** / 0.1012 / 0.0939 | 0.1169 / 0.1228 / 0.1464 | 0.6422 / 0.6020 / 0.5173 | 15.93 |
| K4 s2 | 0.1529 / 0.1302 / 0.1157 | 0.1101 / 0.1146 / 0.1361 | 0.7067 / 0.6205 / 0.5338 | 15.85 |
| K4 s3 | 0.1467 / 0.1234 / 0.1194 | 0.1120 / 0.1289 / 0.1362 | 0.7577 / 0.6228 / 0.5928 | 15.65 |
| K5 s2 | 0.1864 / 0.1597 / 0.1348 | 0.1198 / 0.1149 / 0.1195 | 0.7218 / 0.5787 / 0.4984 | 15.55 |
| K6 s2 | 0.4506 / 0.1588 / 0.1588 | 0.2860 / **0.1094 / 0.1094** | 0.8952 / **0.4960 / 0.4960** | 15.18 |
| K8 s4 | 0.3232 / 0.2230 / 0.2230 | 0.1821 / 0.1261 / 0.1261 | 0.7263 / 0.6319 / 0.6319 | 14.46 |

The trend is monotone and legible: every extra restart buys CFG contraction and
rough-model robustness, and costs exact-model accuracy and sample spread. Past
K3 the spread drops below the truth — the toy's version of "softer" — and A
leaves the floor decisively (K5 0.135, K6 0.159). **K3 s2 is the last point where
all three hold**; K3 s3 is a near-tie (a hair better on A, a hair worse on C and
on the rough field) and would be a defensible alternative default.

Two structural readings, both load-bearing for the design:

- **K1 at full density — the paper's own form — is the worst reprise arm on CFG
  error** (0.804 vs 0.507 at 32 NFE, barely better than plain `secant_anneal`).
  One re-decision is not enough; the injection *count* is the engine, exactly as
  `pump_dual` v1 found when halving it lost the coherency win.
- **Drafts pay for themselves indirectly.** At matched K the draft/full-density
  choice is nearly a wash (K2 s2 0.570 vs K2 s1 0.584 at 32 NFE). What drafts buy
  is the *budget* for a third restart, and that is where the win is: K2 → K3
  moves CFG ED 0.570 → 0.507 at the same 32 NFE. Beyond stride 3 the seed gets
  too crude to be worth integrating (K8 s4 is worse than K2 on every metric).

Band placement (K3 s2, same runs):

| band | A @32 | C w=4.5 ED @24/28/32 | note |
|---|---|---|---|
| **0.45–0.85** | 0.0966 | **0.6406 / 0.5430 / 0.5069** | default |
| 0.55–0.85 | 0.0963 | 0.6466 / 0.5210 / 0.5148 | statistical tie |
| 0.35–0.85 | 0.0979 | 0.6474 / 0.5927 / 0.5341 | no gain from reaching higher σ |
| 0.45–0.90 | 0.0989 | 0.6224 / 0.5625 / 0.5598 | better at 24, worse at 32 |
| 0.45–0.80 | 0.0974 | 0.6942 / 0.5684 / 0.6267 | less CFG contraction |
| 0.30–0.70 | 0.1002 | 0.8795 / 0.7772 / 0.8744 | **capped at 0.70 does little for CFG** |
| 0.45–0.95 | 0.1018 | 0.8048 / 0.8257 / 0.8334 | wastes draft steps at σ≈1 |

The two extremes the design rejected are confirmed rejected, and for the stated
reasons: a band that stops at 0.70 keeps too much of the draft's layout through
the re-noise (73 % worse CFG error than the default), and one that reaches 0.95
spends draft steps where the model is nearly σ-invariant (64 % worse). Between
0.45–0.80 and 0.55–0.85 the answer is flat, so the lower edge is set by the
pump's measured real-image cutoff rather than by this toy.


### 5.5 What the toy cannot say

- Nothing about *images*: no prompt, no VAE, no texture. The CFG arm models
  guidance's off-manifold push on a mixture, which is the mechanism Restart
  targets, but "mode distance" is not "detail".
- The rough-error field is a fixed random high-frequency function of x. A real
  merged checkpoint's error is not this field, and §5.2 shows the ranking there
  depends on its frequency.
- The pump's real win (prompt coherency) was called by eye; the toy never tested
  it and cannot. §6 is the only judge of that.
- At 16 NFE reprise K3 is not better than a deterministic solver on the exact
  model — like the rest of the family, it wants ≥ 24 NFE.

## 6. Real-image validation protocol (GPU, the deciding half)

Kill the UI server first (it holds VRAM). Harness: extend
`scripts/ab_franken_sampler_sched.py` (or a sibling) at the user's production
settings: AnimaFranken-v1.2, CFG 4.5, shift 3.0, CFG interval (0, 0.75),
TeaCache **off**, fp16, 1024×1024, `kirakishou` + `pekora` + one deliberately
short prompt (the `anima-needs-detailed-prompts` failure mode), seeds
{1234, 5678, 9012}.

**Arms at matched NFE ≈ 32** (recomputed with the true cost formula):

| arm | steps | NFE |
|---|---|---|
| `secant_anneal × beta_mix` (production) | 32 | 32 |
| `cogent3_pump × beta_mix` (coherency pick) | 28 | 28 |
| `uni_pc_bh2 × beta_mix` | 32 | 32 |
| **`reprise × beta_mix`** (K3 default) | 20 | 32 |
| `reprise × smoothstep` | 19 | 31 |
| `reprise × beta_mix`, K=2 | 22 | 32 |
| `reprise × beta_mix`, K=4 (stride 3) | 20 | 32 |

For the K-sweep arms, rebuild the registry entry in-process
(`SAMPLERS["reprise"] = partial(sample_reprise, restarts=K, draft_stride=s)`),
the way `scripts/ab_cogent4_per_channel.py`'s `set_arm()` does, so no production
code is touched before the verdict.

**Judging.** No "distance to a high-step reference" — that metric is void at
CFG 4.5 (`no-converged-reference-at-cfg45`). Per seed, a paired sheet
`secant_anneal | cogent3_pump | reprise` at full resolution; the human verdict
on (a) prompt coherency — every tag present, nothing extra, especially on the
short prompt — and (b) fine detail — faces, hands, fabric edges, no wash-out;
plus the harness's objective proxies (sharpness / edge density / colorfulness)
as *paired per-seed deltas* with a sign test, used only to flag, never to
decide.

**What would confirm the design:** reprise ≥ `cogent3_pump` on coherency on ≥ 2
of 3 seeds per prompt, with detail ≥ `secant_anneal` (crisper is expected: the
core is deterministic UniPC).

**What would falsify it:** (i) wash-out / softness at K3 → the toy's mode-d knee
did not transfer; retest at K2, and if K2 is also soft, the restart noise level
is the culprit (try `restart_hi` 0.80). (ii) Coherency no better than
`uni_pc_bh2` → the re-decisions are not reaching the layout; try `restart_hi`
0.90 or K4 before concluding. (iii) Seams / duplicated subjects after a restart
→ the 15 % draft seed is too weak at 1024² — a genuine negative to record, not a
knob to tune. Record whichever outcome here and in the memory note.

## 7. Interactions, risks, caveats

- **CFG guidance interval.** The interval is applied in σ (the closure tests
  `sigma`), so restart passes get CFG exactly where the main pass would. At the
  production (0, 0.75) on `beta_mix` N=20 CFG is active for σ > 0.245 — the
  whole band. An interval whose σ bound rises into the band (end < ~0.6 at these
  step counts) leaves the restarts nothing to re-decide — the same caveat the
  pump documents.
- **NFE vs nominal steps.** ~1.5× on the production schedulers (§4). The XYZ
  grid's per-cell time will look 1.5× a plain 20-step cell — that is the honest
  cost.
- **TeaCache.** Untested with restarts. The drift rule sees a huge input change
  at each jump and recomputes, so nothing *breaks*, but the Hermite forecast's
  residual history straddles a jump; run the A/B with TeaCache off and treat
  "reprise + TeaCache" as a separate, later measurement.
- **Live preview** visibly rewinds at each restart (drafts, then the final
  pass). Cosmetic.
- **img2img at low strength** silently degrades to `uni_pc_bh2` (no band in the
  sliced schedule) — by design.
- **Inpaint** (`MaskedDenoiser` pins the keep region every step): the jump
  re-noises the keep region too, consistently with the forward process, and the
  mask re-pins it on the next call. No change needed; verify once on GPU.
- **cuda_graphs / compile**: sampler-level only, no interaction.
- **At ≤ 16 NFE** reprise is not a win; use 24+ NFE (16+ nominal steps).

## 8. Not in scope (recorded so it is not rediscovered)

- **Phase 2, the `restarts` panel knob.** Deliberately not plumbed: no settings
  field, no metadata key, no XYZ axis. `Sampler: reprise` in PNG metadata implies
  the baked defaults. Plumb it exactly like `gate_reduce`
  (`COGENT-IMPROVE-IMPLEMENTED.md`, "Cogent4 release and UI plumbing") *only* if
  §6 finds K is worth exposing.
- **Z-Sampling-style low-CFG deterministic inversion as the re-noise** (Bai et
  al., ICLR 2025): same "re-decide at high σ" idea, but it needs a *second*
  denoiser closure at a lower guidance scale — a protocol change in every
  pipeline. The natural follow-up once the stochastic version has an image
  verdict: the jump would become `x ← inv_{cfg=1}(x)` + a smaller noise.
- **CFG++ / guidance-aware re-noising** — needs the uncond x0 exposed; already
  deferred in this repo for that reason.
- **Coherence-gated (pump-style `1 − C`) restart noise** — off-marginal for a
  large jump; not sound without a different model contract.
- **Listing on SD/SDXL/FLUX** — the core is family-agnostic and the VE jump is
  one line (and unit-tested), but the evidence and the progress-bar plumbing are
  Anima's.
- **A `core=` switch** (restart over `cogent3` / `dpmpp_2m`) — speculative
  flexibility; the toy has UniPC bh2 ≥ 3M at 24–32 and the pipelines already
  special-case UniPC's kwargs.
- **An NFE-neutral variant** (the sampler carving the restart budget out of the
  given schedule) — the sampler only sees `sigmas`, not the scheduler, so
  subsampling the grid would silently change σ placement. The honest contract is
  "restarts cost model calls, here is the formula", like `heun`'s 2×.
- **Adaptive K per run from the cogent gate's `psi` stream** — the closed-loop
  idea, already deferred on record.
- Already killed elsewhere, not revisited here: lag-2 gate, `v_model`
  separation, EasyCache rule, sa_solver, ACAS / flow_karras / logsnr_front
  schedulers, self-correcting variable-NFE schedulers.
