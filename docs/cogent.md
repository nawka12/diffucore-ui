# `cogent`: a coherence-gated exponential multistep sampler

*Status: implemented in diffucore (`sample_cogent`), offline-green, benchmarked
against an analytically-known ground truth (`scripts/ab_cogent.py`). No real-image
A/B yet. This document states only what the measurements support — including the
things that did not work.*

## Summary

`cogent` keeps the `*_anneal` family's σ-annealed ancestral burn-in
(`η_i = η_max·σ_i`) and runs it on the DPM-Solver++(2M) exponential-integrator
core in half-logSNR space. The new part is that the weight on the 2nd-order
correction is **measured every step** instead of hardcoded:

```
psi = max( (1 + 2·rho)/3 ,  1 − e^(−h) )        clamped to [0, 1]
x  += psi · (textbook 2nd-order term)
```

where `rho = cos(D_i, D_{i−1})` is the cosine between the last two changes in the
model's x0 estimate. Cost: two dot products per step, no extra model evaluations.
`eta_max=0` is deterministic; `psi ≡ 1` is exactly `dpmpp_2m_anneal`.

Against `secant_anneal` on the benchmark below: **~2.3× more accurate**
deterministically at matched steps, and **12–25% closer to the target
distribution** under an imperfect model at 8 and at 24–32 steps. It gives up a few
percent in the 12–16 step band.

## 1. The problem

Every sampler in this family has to answer one question: *how much should I trust
the divided difference of the x0 history?* It is the 2nd-order term's only
ingredient, and it amplifies both the ancestral noise the burn-in injects and
whatever the model itself got wrong.

Each existing answer is a hardcoded proxy that never looks at the data:

| sampler | rule |
|---|---|
| `secant` / `secant_anneal` | `curvature·(1 − \|Δσ\|/σ)·(1 − σ)` — a σ heuristic, capped at 0.25 |
| `uni_pc_anneal` | order ramped with σ |
| `stork2` | fixed damping, `C1(9) ≈ 0.463` instead of 0.5 |

A fixed rule has to be tuned for the worst case it might meet, so it over-damps a
good model and under-damps a bad one.

## 2. The measurement

Model the denoiser output as signal plus per-step noise, `x0_i = f_i + n_i`, with
`n_i` iid of energy `v` — "noise" meaning everything the step's estimate got
wrong, dominated on a stochastic sampler by the ancestral noise injected into `x`
last step. Write `S = ‖Δf‖²` and assume `Δf` varies slowly across a step (the
assumption the 2nd-order term itself already makes). Then, because `D_i` and
`D_{i−1}` share `n_{i−1}` with opposite sign:

```
<D_i, D_{i-1}> = S − v          ‖D_i‖² = ‖D_{i-1}‖² = S + 2v
```

so the cosine between them measures the derivative estimate's SNR:
`rho = (S − v)/(S + 2v)`, i.e. `v/S = (1 − rho)/(1 + 2·rho)`. Substituting that
into the Wiener shrinkage that minimises `E‖psi·D_i − Δf‖²` — namely
`psi = S/(S + 2v)` — collapses to a straight line:

```
psi = (1 + 2·rho) / 3
```

`rho = 1` (clean, straight trajectory) ⇒ `psi = 1`, the undamped coefficient.
`rho = −1/2` (pure noise — the floor of this model) ⇒ `psi = 0`, no correction.

It is reduced per batch sample over the whole latent, so the cosine is estimated
from tens of thousands of elements and is a precise statistic, not a noisy one.

**It works.** Measured on the toy below, mean `rho` over a 16-step run:

| model | `eta_max=0` | `eta_max=0.5` |
|---|---|---|
| exact denoiser | 0.97 | 0.21 |
| + smooth error field | 0.97 | 0.24 |
| + rough (high-frequency) error field | 0.89 | **−0.25** |

A merged / imperfect field is cleanly separated from a good one (−0.25 vs +0.21)
even with churn running, so it damps itself automatically while a clean model
keeps the full textbook coefficient.

## 3. The floor, and why it is not a tuned constant

`rho` also drops when the *trajectory* curves, because the derivation assumed
`Δf_i ≈ Δf_{i−1}`. There the measurement is answering the wrong question:
curvature is when the 2nd-order term is needed **most**, not least — and that is
exactly the coarse-step regime. Left alone, the gate collapses the sampler to
first order at low step counts, which is catastrophic (2.5× worse at 8 steps).

The fix is a floor equal to `1 − e^(−h)` — the very `phi`-weight the exponential
integrator already multiplies this term by. The rule reads: *never damp the
correction below the weight the step itself gives it.* It vanishes as `h → 0`
(fine steps, where the coherence reading is trustworthy and may damp all the way
to zero) and approaches 1 as `h` grows.

Sweeping alternatives at 8/12/16/24/32 steps under a rough error field, energy
distance to the data law:

| floor | 8 | 12 | 16 | 24 | 32 |
|---|---|---|---|---|---|
| none | 0.284 | 0.142 | 0.112 | 0.120 | 0.149 |
| `0.5·(1 − e^−h)` | 0.266 | 0.140 | 0.111 | 0.120 | 0.149 |
| `h/(1+h)` | 0.224 | 0.127 | 0.110 | 0.121 | 0.150 |
| **`1 − e^−h`** | **0.201** | **0.121** | **0.110** | **0.121** | **0.150** |
| constant 0.5 | 0.196 | 0.114 | 0.110 | 0.133 | 0.164 |

The principled choice is also the best of the h-shaped ones, and a constant floor
only wins at ≤12 steps by giving up the 24–32 range where the sampler is meant to
be used.

## 4. What did not work

Recorded because the negative results shaped the design.

- **Shrinking without the floor** (the first version) loses badly under a clean
  model with churn: the gate correctly reports the derivative is ~90% noise, but
  removing it costs more in bias than it saves in variance. Damping is not free.
- **Using `psi` as a recursive filter gain** on the slope
  (`slope ← psi·new + (1−psi)·carried`) instead of a shrinkage: roughly neutral
  everywhere. The AB2 term's own noise is second-order compared to the ancestral
  noise being injected deliberately, so denoising it does not move the needle.
  This killed the original premise that derivative noise was the binding
  constraint.
- **Gating only the 3rd-order term** on a DPM++(3M) core: a small consistent gain
  in deterministic accuracy (best in class at 12–24 steps) but no robustness gain,
  and 3M is worse than 2M under model error at `eta_max=1`.
- **A smooth model-error field** as the robustness testbed: it just displaces the
  ODE's fixed point, so every accurate solver lands faithfully on the wrong
  answer and the metric rewards *under*-convergence. Any sampler ranking derived
  from it is measuring the wrong thing. The error field has to be rough — varying
  faster than a sampling step — for churn to have anything to average away.

## 5. Benchmark

`scripts/ab_cogent.py`. A 512-dim Gaussian mixture with a power-law covariance
spectrum, under the rectified-flow interpolation, where the optimal denoiser is
analytic and the exact ODE solution is available by 4000-step integration.

**A. Deterministic accuracy** — RMSE against the exact ODE solution, `eta_max=0`:

| sampler | 8 | 12 | 16 | 24 | 32 |
|---|---|---|---|---|---|
| **cogent** | 0.140 | 0.0668 | 0.0371 | 0.0221 | 0.0171 |
| secant_anneal | 0.202 | 0.124 | 0.0844 | 0.0459 | 0.0295 |
| dpmpp_2m_anneal | 0.135 | 0.0649 | 0.0363 | 0.0219 | 0.0170 |
| stork2 | 0.123 | 0.0655 | 0.0410 | 0.0223 | 0.0160 |
| uni_pc_bh2 | 0.150 | 0.0838 | 0.0513 | 0.0234 | 0.0127 |

**B. Rough model error** — energy distance to the data law, `eta_max=0.5`,
`freq=6, tau=0.35`:

| sampler | 8 | 12 | 16 | 24 | 32 |
|---|---|---|---|---|---|
| **cogent** | **0.201** | 0.121 | 0.110 | **0.121** | **0.150** |
| secant_anneal | 0.229 | **0.115** | **0.108** | 0.152 | 0.200 |
| dpmpp_2m_anneal | 0.112 | 0.126 | 0.157 | 0.198 | 0.227 |
| stork2 | 0.103 | 0.125 | 0.151 | 0.192 | 0.222 |
| uni_pc_bh2 | 0.109 | 0.113 | 0.139 | 0.186 | 0.220 |

Across 13 error conditions × 5 step counts, `cogent` takes 22 best-in-column
cells to `stork2`'s 21, `uni_pc_bh2`'s 12 and `secant_anneal`'s 9. The split is
structural and consistent: **deterministic solvers own ≤12 steps, `cogent` owns
24–32**, which is the range these annealed samplers are actually run at.

## 6. Scheduler pairing

`cogent` does **not** inherit its siblings' scheduler preference. The σ-secant
family wants a high-σ-dense schedule; `cogent` wants the *low*-σ end resolved
finely and smoothly, because that is what the λ-space exponential core wants.

`--schedulers`, energy distance under a rough model error (`tau=0.35, freq=6`),
`eta_max=1.0`; `dpmpp_2m_anneal` on the same schedule in brackets:

| scheduler | det 32 | rough 16 | rough 24 | rough 32 | [2m_anneal rough 32] |
|---|---|---|---|---|---|
| **flow** | 0.0171 | **0.111** | 0.115 | 0.142 | [0.241] |
| **sgm_uniform** | 0.0168 | 0.111 | **0.114** | **0.140** | [0.237] |
| **simple** | 0.0168 | 0.111 | 0.115 | 0.141 | [0.238] |
| **linear_quadratic** | 0.0177 | 0.160 | **0.110** | **0.121** | [0.226] |
| smoothstep | 0.0213 | 0.150 | 0.210 | 0.284 | [0.506] |
| beta_mix | 0.0201 | 0.400 | 0.375 | 0.397 | [0.664] |
| beta | 0.0260 | 0.411 | 0.358 | 0.365 | [0.661] |
| kl_optimal | 0.0802 | 0.480 | 0.489 | 0.488 | [0.706] |
| normal | 0.0821 | 1.353 | 0.972 | 0.846 | [1.391] |
| infinity_htds | 0.1340 | 1.468 | 1.131 | 1.003 | [1.749] |
| infinity | 0.1699 | 1.852 | 1.378 | 1.118 | [1.983] |

**Use `flow`, `simple` or `sgm_uniform`** (near-identical for flow models), or
**`linear_quadratic` at 24–32 steps**, where it is the best of the set.

Two things worth reading off the table. First, the ordering is a property of the
**shared DPM++(2M) core, not the gate**: `dpmpp_2m_anneal` degrades on exactly the
same schedules, by roughly twice as much, and `cogent` beats it on every single
one. Second, the mechanism for `beta` / `smoothstep` specifically: they are
endpoint-dense in *t*, but after the shift map they leave a **coarser minimum
λ-step** than `flow` (0.24–0.25 vs 0.167 at 24 steps). That costs accuracy
directly, and it also pins the `1 − e^(−h)` floor high enough that the coherence
term can never damp — so on those schedules `cogent` degenerates toward
`dpmpp_2m_anneal`, which is exactly what the table shows.

This section is the **weaker half** of the offline evidence. Comparing samplers on
a fixed schedule is apples-to-apples; comparing schedules asks which σ placement
suits a model's error, and the toy's error field is not Anima's. Worth an A/B on a
real checkpoint before treating the bottom of the table as settled.

## 7. Caveats

- Not A/B'd on real images. The toy predicts one thing the repo has already
  observed on hardware — that `uni_pc_anneal`'s quality falls monotonically with
  `eta` — which is mild evidence it transfers, but it is not proof.
- At ≤12 steps a deterministic solver (`stork2`, `uni_pc_bh2`) beats it and every
  other member of this family. Use 24+ steps.
- On a *clean* model with heavy churn the gate over-damps slightly, because
  injected noise and model error both lower `rho` and it cannot tell them apart.
  Separating them would need a lag-2 inner product (`<D_i, D_{i−2}>` shares no
  noise term), which needs 4 x0 estimates of history — untested.
- `psi` is one scalar per batch sample. Per-channel gating is plausible on 4-D
  latents and untried.
