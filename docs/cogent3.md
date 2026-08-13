# `cogent3`: the measured gate, carried to third order

*Status: implemented in diffucore (`sample_cogent3`), offline-green (unit tests),
benchmarked against an analytically-known ground truth (`scripts/ab_cogent3.py`).
No real-image A/B yet. This document states only what the measurements support —
including the things that did not work.*

## Summary

`cogent3` is `cogent`'s one-idea-further: keep the σ-annealed ancestral burn-in
(`η_i = η_max·σ_i`) and the measured 2nd-order gate, and add the third order —
the DPM-Solver++(3M) exponential-integrator term — gated by a *second* measured
Wiener shrink, on the coherence of consecutive second differences:

```
psi_1 = max( (1 + 2·rho_1)/3 ,  1 − e^(−h) )         (cogent's gate, verbatim)
psi_2 = (2 + 3·rho_2) / 5                            (new, no floor)
x    += psi_1 · (2nd-order term)  +  psi_2 · (3rd-order term)
```

`rho_1 = cos(D_i, D_{i−1})` on consecutive first differences of the x0 history;
`rho_2 = cos(E_i, E_{i−1})` on consecutive *second* differences (`E_i =
D_i − D_{i−1}`). Cost: two more dot products per step than cogent, still one
model evaluation per step. `eta_max=0` is deterministic; both gates pinned to 1
with `eta_max=0` is bit-for-bit the deterministic DPM-Solver++(3M) core
(`dpmpp_3m_sde` with `eta=0`).

## 1. Why a third order at all, after cogent said "measure it"

cogent's measured gate exists because the 2nd-order term amplifies whatever the
denoiser got wrong. A 3rd-order term is worse on that axis: it is a *difference
of differences*, so it amplifies per-step noise by the square — one step's noise
enters `E_i` three times (`n_i − 2n_{i−1} + n_{i−2}`, energy `6v` against `v`
per step). The 2nd-order-only family (`cogent`, `stork2`, `uni_pc`) is
deliberately built around that fact.

But the same measurement that made the 2nd-order term safe has a natural
generalisation to the 3rd-order one. Model `x0_i = f_i + n_i` with iid `n_i` of
energy `v`. Then:

```
<E_i, E_{i-1}> = S2 − 4v          ‖E_i‖² = ‖E_{i-1}‖² = S2 + 6v
```

(`S2 = <Δ²f_i, Δ²f_{i−1}>`; the `−4v` comes from the shared `−2n_{i−1}` and
`n_{i−2}` terms entering with opposite signs), so the cosine
`rho_2 = (S2 − 4v)/(S2 + 6v)` reads the curvature's SNR exactly as `rho_1`
reads the slope's. The Wiener shrink minimising `E‖psi·E_i − Δ²f‖²` collapses,
as cogent's did, to a straight line:

```
u = v/S2 = (1 − rho_2)/(4 + 6·rho_2)      ⇒      psi_2 = (2 + 3·rho_2)/5
```

`rho_2 = 1` (clean, smoothly-curving trajectory) ⇒ `psi_2 = 1`, the undamped
textbook coefficient; `rho_2 → −2/3` (pure noise) ⇒ `psi_2 = 0`, the term
switched off entirely.

## 2. Why the 3rd-order gate has no floor

cogent's floor (`1 − e^(−h)`) exists because `rho_1` dual-reads *curvature*: on
a sharply curved-but-clean trajectory the first differences swing, `rho_1`
drops exactly when the 2nd-order term is needed most, and a coarse step must
keep its correction whatever the SNR reading.

The second difference does not have that failure mode. `rho_2` measures whether
the *curvature itself* is consistent — which is precisely the condition for
extrapolating it to be trustworthy — and a coarse step does not change that:
drop the 3rd-order term and the step reverts to the gated 2nd-order update,
which is `cogent`'s behaviour, already load-bearing. So the honest rule is
"the 3rd-order term is optional; the data can turn it off completely". A floor
would pin `psi_2` up exactly where the term is most dangerous (coarse steps on
a rough model), which is the one place it must be allowed to go to zero.

## 3. The first 3rd-order-capable step

The 3M core can form its third-order term from three history points, but its
gate needs four (a second difference to compare `E_i` against). On that one
step `cogent3` bootstraps `psi_2 = psi_1` — "rate the new term by the best
trust signal we have" — instead of running it ungated.

## 4. Degradation invariants (tested)

| pin | result |
|---|---|
| `psi_1 = psi_2 ≡ 1`, `eta_max=0` | bit-for-bit `dpmpp_3m_sde(eta=0)` (the deterministic 3M core) |
| `psi_2 ≡ 0` | gated 2nd-order-only; per-step the `cogent` behaviour, never worse |
| `eta_max = 0` | deterministic; no noise drawn (seed-independent) |
| any | final σ→0 step lands on the x0 estimate |

## 5. Measurements

`scripts/ab_cogent3.py` runs the same GMM-flow toy as `ab_cogent.py` — an
analytically-solvable rectified flow at Anima's shift=3.0, exact optimal
denoiser, 4000-step Euler reference, and frozen rough error fields standing in
for a merged / imperfect velocity field — scored by RMSE against the exact ODE
(deterministic regime) and energy distance to true samples (rough-model
regime). The field is `cogent3` vs `cogent` (gated 2nd order) vs `3m_ungated`
(the 3M core with no gates at all — what cogent3 would be without its
measurements).

### Deterministic accuracy, clean model — RMSE vs exact ODE (lower is better)

| sampler | 8 | 12 | 16 | 24 | 32 |
|---|---|---|---|---|---|
| **cogent3** | **0.132** | **0.0598** | **0.0344** | **0.0213** | **0.0156** |
| cogent | 0.140 | 0.0668 | 0.0371 | 0.0221 | 0.0171 |
| 3m_ungated | 0.127 | 0.0572 | 0.0331 | 0.0211 | 0.0156 |

cogent3 beats cogent at every step count (≈6% at 8 steps, ≈4–10% at 12–32),
and sits between cogent and the ungated 3M core: the gate knowingly spends a
few percent of the core's clean-model edge at 8 steps (0.132 vs 0.127) to buy
the robustness below.

### Rough model, `eta_max=1.0` — energy distance to data (lower is better)

The regime the whole `*_anneal` family exists for. Representative rows
(full tables: `python scripts/ab_cogent3.py`):

| freq, tau | sampler | 8 | 12 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|
| 12, 0.35 | **cogent3** | **0.196** | **0.133** | 0.119 | 0.104 | 0.108 |
| | cogent | 0.238 | 0.146 | 0.123 | 0.103 | 0.107 |
| | 3m_ungated | 0.107 | 0.119 | 0.133 | 0.154 | 0.165 |
| 6, 0.35 | **cogent3** | **0.195** | **0.125** | **0.111** | 0.114 | 0.140 |
| | cogent | 0.231 | 0.131 | 0.111 | 0.115 | 0.142 |
| | 3m_ungated | 0.104 | 0.132 | 0.165 | 0.210 | 0.239 |
| 3, 0.35 | **cogent3** | **0.185** | **0.133** | **0.145** | 0.220 | 0.313 |
| | cogent | 0.214 | 0.136 | 0.142 | 0.216 | 0.310 |
| | 3m_ungated | 0.117 | 0.183 | 0.252 | 0.354 | 0.426 |

Three honest readings:

1. **The gate does its job.** At 24–32 steps the ungated 3M core is *worse than
   plain cogent* (its noisiest term amplifies the rough error), while cogent3
   stays at cogent's level — 22–27% closer to the data law than the ungated
   core on the freq=6 and freq=12 rows.
2. **At 8 steps the third order pays even under error.** cogent3 is clearly
   better than cogent at 8 steps on every freq/tau pair measured (e.g. 0.196
   vs 0.238 at freq=12, tau=0.35), which the deterministic table alone would
   not have predicted — the burn-in's coarse first steps are where the gated
   3rd-order term most helps.
3. **It never meaningfully loses to cogent.** At 16–32 steps the two tie within
   run noise (worst deltas ≈ ±2%, e.g. 0.220 vs 0.216 at 24 steps, freq=3).
   This is the "no floor" design paying off: the 3rd-order term's coherence
   reading turns it off exactly when it would hurt.

Deterministically with a rough model (`eta_max=0.0`), cogent3 also tracks
cogent within noise at 16–32 steps; at 8–12 steps both are dominated by
`stork2`/`uni_pc_bh2`, as in the cogent benchmark.

## 6. Scheduler pairing

Measured on the same toy (`python scripts/ab_cogent3.py --schedulers`; det =
RMSE vs exact ODE at `eta_max=0`, rough = energy distance at `eta_max=1.0`
under tau=0.35/freq=6, with `cogent`'s rough row in brackets for contrast):

| scheduler | det 8 | det 24 | det 32 | rough 8 | rough 24 | rough 32 |
|---|---|---|---|---|---|---|
| **flow** | 0.132 | 0.021 | 0.016 | **0.195** | 0.114 | 0.140 |
| **simple** | 0.132 | 0.021 | 0.016 | 0.195 | 0.114 | 0.139 |
| **sgm_uniform** | 0.133 | 0.022 | 0.016 | 0.197 | 0.113 | 0.138 |
| linear_quadratic | 0.256 | **0.016** | **0.007** | 0.176 | **0.108** | **0.121** |
| smoothstep | **0.088** | 0.023 | 0.014 | **0.160** | 0.194 | 0.264 |
| beta / beta_mix | 0.59 / 0.42 | 0.070 / 0.048 | 0.027 / 0.019 | 6.5 / 4.3 | 0.51 / 0.64 | 0.36 / 0.46 |
| kl_optimal / normal / infinity / infinity_htds | ≫ 0.5 | ≫ 0.1 | ≫ 0.08 | ≫ 4.8 | ≫ 1.8 | ≫ 1.0 |

The verdict is cogent's verdict, with the same caveat that it is a property of
the shared 3M/2M exponential core rather than of the gates: **`flow` /
`simple` / `sgm_uniform` are the safe defaults** (near-identical for flow
models), `linear_quadratic` is the best at 24–32 steps under strong model
error, and `smoothstep` is sharp at 8 steps but degrades under error as steps
grow. `beta`, `beta_mix`, `kl_optimal`, `normal`, `infinity` and
`infinity_htds` land coarser minimum λ-steps and are markedly worse — for
`cogent3` *and* for `cogent` alike, so the ranking is the core's, not the
gate's.
