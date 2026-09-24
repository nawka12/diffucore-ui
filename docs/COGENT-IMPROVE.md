# COGENT-IMPROVE: a design for the next step of the cogent family

*This file is the shared design record for improving the `cogent` / `cogent3` /
`cogent3_pump` samplers and their schedulers. Each contributor writes under their
own tag. The sections below are **[DSF]**'s.*

---

## [DSF] Cogent4: measured spatial gating, curvature-robust measurement, and a closed-loop schedule

*Status: design only. Nothing here is implemented or measured. Following the
family's own epistemic rule (docs/cogent.md, docs/cogent3.md, docs/pump-scheduler-plan.md
"state only what the measurements support — including the things that did not
work"), every claim below is grounded in a documented measurement or an explicitly
flagged *untested* idea from this repo, and each new mechanism is written with a
degradation invariant that pins it back to the existing family at its limit.*

### 0. The weakest link, restated

The cogent family's architecture is settled and strong: σ-annealed ancestral
burn-in (`eta = eta_max·σ`) on the λ-space exponential integrator, with the
divided-difference corrections scaled by **measured** Wiener shrinks. What has
been *added* with each generation is a better correction term (2nd order →
3rd order → pump). What has not been touched, in three generations, is the
**measurement layer itself**. Every weakness the docs record lives there:

1. **One scalar per batch sample.** The coherence gate reduces over the *whole
   latent* (every dim but the first), so it cannot tell a region where the model
   is clean from a region where it is still guessing. A clean region's correction
   is damped because a noisy region elsewhere dragged the global cosine down.
   docs/cogent.md §7 names the fix and leaves it: *"Per-channel gating is
   plausible on 4-D latents and untried."*
2. **Noise is confounded with curvature.** The `rho` estimator's numerator
   carries the shared `−v` term; curvature also lowers `rho` exactly when the
   2nd-order term is needed most, which is why the floor `1 − e^(−h)` exists —
   a blunt instrument. docs/cogent.md §7 records the residual cost: *"on a clean
   model with heavy churn the gate over-damps slightly, because injected noise
   and model error both lower `rho` and it cannot tell them apart. Separating
   them would need a lag-2 inner product (`<D_i, D_{i-2}>` shares no noise
   term), which needs 4 x0 estimates of history — untested."*
3. **The measurement is retrospective.** It can damp a step it has already
   taken, but it cannot re-plan the steps still to come. Scheduler pairing is a
   fixed up-front choice (docs/cogent.md §6 is self-described as "the weaker
   half"), and the pump's scheduler tuning is a manual knob sweep
   (docs/pump-scheduler-plan.md §3.2: `pump_share` is the load-bearing knob,
   "still not exposed in the UI"). The repo's own asymptote is pump-aware
   calibration (§3.3) — *offline*; nothing adapts *per run*, and the sampler
   already computes, for free, exactly the signal an adaptive scheduler would
   need.

**Cogent4 is the three fixes to the measurement layer** — one per weakness —
plus a scheduler that closes the loop. Each fix is cheap (dot products, no
extra model calls), each degrades gracefully to the current family, and the
three compose into a sampler that damps *where* it should, measures *how much
of the noise it injected itself*, and spends the remaining budget *where the
measurement says the model is struggling*.

---

### 1. Per-channel gates (spatial selectivity)

**Mechanism.** On 4-D latents, compute `psi_1` and `psi_2` per channel instead
of per batch sample: reduce the coherence cosines over `(H, W)` only, giving a
length-`C` vector of shrinks, and scale each channel of the correction term by
its own shrink:

```
rho_1^c = <D_i^c, D_{i-1}^c> / (‖D_i^c‖ · ‖D_{i-1}^c‖)      reduced over H, W
psi_1^c = max( (1 + 2·rho_1^c)/3 ,  1 − e^(−h) )
x += psi_1^c * (2nd-order term)^c                            per channel
```

The same operation extends to `psi_2` (per-channel second-difference cosines).
The Wiener-shrink derivation in `_coherence_gate` goes through verbatim with
per-channel `S_c, v_c`; the shrink is per-channel because the signal-to-noise of
the derivative estimate genuinely is per-channel (structure-heavy channels carry
more `S` for the same `v`).

**Why it is sound, not just plausible.** The global gate is the cosine of the
*concatenated* latent vector. The per-channel mean is the mean of per-channel
cosines. They agree when every channel has the same `rho`, and differ when
channels do — and there is no reason to believe the model's error is uniform
across channels. The precision argument the family already relies on
("reduced over tens of thousands of elements, so the cosine is a precise
statistic") still holds *per channel*: on any real latent `H·W ≥ 4096`
(64×64), and the pump's own offline plan already reshapes the toy to
`[B, 1, 16, 32]` so the structure tensor averages over ~512 elements — the
same sample size the current global gate uses on `[B, 512]`
(docs/pump-scheduler-plan.md §4).

**Degradation invariant.** A "reduce over all dims" mode reproduces the
current global gate exactly — `cogent3` with `reduce="all"` is `cogent3`
today. Per-channel is a strict superset. The bit-for-bit pins of the family
are untouched: `psi ≡ 1`, `eta_max=0` is still the deterministic 3M core.

**Cost.** `C` dot products per step (C = 4 on SD/SDXL, 4–16 on Anima) — noise
against one model call.

**The non-4D fallback.** FLUX token latents and the `[B, 512]` toy have no
spatial axes to reduce over; those paths stay on the global scalar,
bit-for-bit `cogent3`. So cogent4 is, like `cogent3_pump`, a 4-D-only
improvement, and the offline toy must use the `[B, 1, 16, 32]` reshape to
exercise it (the pump already set this precedent).

---

### 2. Curvature-robust lag-2 measurement

**Mechanism.** The current `psi_1` comes from the linearized cosine
`(1 + 2·rho)/3`, which is exact under the slow-variation assumption but whose
numerator is contaminated by the shared noise term. Once four x0 estimates
exist, replace it with the lag-2 ratio — the idea docs/cogent.md §7 flags as
untested:

```
D_i   = x0_i − x0_{i-1}                      D_{i-2} = x0_{i-2} − x0_{i-3}
psi_1 = clamp( <D_i, D_{i-2}> / ‖D_i‖² , 0, 1 )     with the same floor
       psi_1 = max( that, 1 − e^(−h) )
```

**Derivation.** With `x0_i = f_i + n_i`, `n_i` iid of energy `v`:
`D_i = Δf_i + n_i − n_{i-1}` and `D_{i-2} = Δf_{i-2} + n_{i-2} − n_{i-3}`
share **no** noise term, so under slow variation of `Δf`:

```
<D_i, D_{i-2}> ≈ ‖Δf‖² = S          ‖D_i‖² = S + 2v
```

The ratio is therefore the Wiener shrink `S/(S+2v)` **directly** — no
`(1 + 2ρ)/3` linearization needed — and it is strictly less curvature-
contaminated than the lag-1 cosine (no `−v` term dragging the numerator, and
the curvature deficit it does carry is already the floor's job, exactly as in
`_coherence_gate`). Two things the current gate cannot do fall out for free:

**The measured noise energy.** Rearranging gives the family's first direct
estimate of the per-step noise energy:

```
v_est = ( ‖D_i‖² − <D_i, D_{i-2}> ) / 2
```

This is a new, checkable quantity (the toy knows the true injected variance
and the true model-error energy, so `v_est` is directly falsifiable — see §5).
It is the number the family has been *implicitly* estimating all along and
never measuring.

**The over-damping fix.** docs/cogent.md §7's caveat is that injected noise
and model error both lower `rho` and the gate cannot tell them apart, so a
clean model with heavy churn is over-damped. With `v_est` in hand, subtract
the *known* ancestral-noise variance to recover the model-error energy:

```
v_inj = σ_next² · |expm1(−2·h·eta)| · s_noise²        (known analytically)
v_model = v_est − g_i² · v_inj                        (g_i = model's x-noise gain)
psi_1  uses v_model instead of v_est                   → no over-damping on a clean model
```

The one unknown is `g_i` — the denoiser's sensitivity of `x0` to noise in `x`,
which is model-dependent and not measured by a single call. This is the
riskiest piece of the design and the one place I will not hand-wave: it needs a
probe (see §5). If `g_i` proves un-estimable cheaply, the fallback is to keep
`psi` on `v_est` (which still buys the curvature robustness and the scheduler
signal) and treat the `v_model` separation as a separate, gated experiment.

**Degradation invariant.** Until four x0 estimates exist (the first two
correctable steps), the sampler uses the current `rho` gate verbatim — the
bootstrap pattern `cogent3` already uses for its first 3rd-order step. With
`v_model` disabled, `psi_1` from the lag-2 ratio at `rho_1 → 1` equals the
current gate's value at the same SNR; the pins (`psi ≡ 1`, `eta_max=0` →
deterministic 3M core bit-for-bit) are unchanged because those settings never
engage the measurement.

**Cost.** Two extra dot products per step (the lag-2 inner product plus the
`v_est` norm bookkeeping). Still one model call.

---

### 3. The closed-loop schedule (the "accompanying scheduler")

**The premise.** The repo has already measured that the *σ placement* is what
drives this core's quality — and that the mechanism is not the one it first
believed. docs/pump-scheduler-plan.md §7: across the eleven schedulers of
docs/cogent.md §6, the rough-32 ranking correlates with **terminal depth**
(Spearman +0.91) and **final λ-step** (+0.94), *not* with the minimum λ-step
(+0.46) the cogent.md §6 text names, and *negatively* with pumped-step count
(−0.19). The design rule that survives every revision: **never go deeper than
`flow`'s terminus, and spend the budget where the trajectory is hard.** That is
a statement about *where steps should be*, and it is currently decided by a
fixed schedule family chosen up front — the weaker half of the evidence, in
this repo's own words.

Cogent4 already computes, at every step, the exact quantity that says *where
the trajectory is hard right now*: the measured noise fraction

```
v/S = (1/psi_1 − 1)/2        (from psi_1 = S/(S+2v))
```

low `psi_1` = the derivative estimate is mostly noise = curvature the core
cannot resolve at this step size, or a region where the model is still
churning. Both mean "a step here would have paid more than a step there."
That signal is currently thrown away after gating one term.

**The loop.** At a mid-run seam (e.g. after 50% of steps), the sampler hands
the schedule controller its measured hardness stream, and the controller
**re-interpolates the remaining σ in λ-space with local step density
proportional to measured hardness**:

```
hardness(λ) = EMA over the traversed window of (1 − psi_1), binned in λ
density(λ)  ∝ (1 + α·hardness(λ))  over the remaining λ-band
σ_remaining  = sigmoid(−λ) on the re-weighted λ grid
```

- **Terminus is fixed** at the base schedule's own `σ(t = 1/steps)` — the
  measured-safe shallow terminus. The loop re-distributes the *interior*
  steps; it is never allowed to deepen the tail. This is what makes it
  consistent with §7's "+0.91/+0.94" finding instead of fighting it.
- **No-waste constraints preserved**: no σ ≥ `top_sigma` (≈0.99) step is ever
  inserted, the schedule stays strictly monotone, and the first-step burn-in
  is untouched.
- **For `cogent3_pump`**, the same seam adjusts the *load-bearing knob*:
  when measured hardness is high, keep more of the remaining budget as
  re-deciding rounds in the pumped band (raise the effective `pump_share`);
  when the model reads clean, taper it. This is the manual `pump_share` sweep
  of docs/pump-scheduler-plan.md §3.2 made automatic, using the sampler's own
  measurement instead of a hand-set constant. And `v_est` modulates the pump's
  *amplitude*: a model that is already noisy does not need help being
  re-decided — scale `nu` down with measured `v/S`, which is the direction
  `cogent3_pump`'s docs already flag (over-pumping a rough model churns
  texture).

**Why the signal is trustworthy enough to schedule on.** The family's own
measurements support both readings of low `psi_1`: the core wants fine steps
where the trajectory curves (that is the floor's rationale), and the pump's
coherency rule is "injection count, not size" — more re-deciding rounds in the
hard band. Both say *concentrate steps where `psi_1` is low*; the loop does
exactly that and no more.

**Degradation invariant.** `α = 0` (or a constant hardness stream) returns
the base schedule's remaining grid unchanged — the closed loop at `α = 0` is
bit-identical to the fixed schedule. The 2M/3M cores already handle the
nonuniform `h` a re-interpolation produces (the `r0, r1` machinery in
`sample_cogent3` exists precisely for that), so mid-run re-scheduling does not
invent a new integration problem.

**Cost.** Zero model calls. The controller is a histogram and a
re-interpolation over the remaining band; the sampler was going to compute
`psi_1` anyway.

**Honest caveat on the signal.** `psi_1` conflation of curvature and noise is
not a bug for the *concentration* decision (both want steps there), but it
does mean the loop cannot distinguish "spend more to resolve curvature" from
"the model is wrong here and no step count fixes it". The λ-band projection
— assuming hardness measured in the traversed band extends to the remaining
band — is the weakest assumption in the design. Both are named in §6.

---

### 4. Degradation invariants (the family's contract, restated)

| pin | result |
|---|---|
| `reduce="all"` (no per-channel) | bit-for-bit `cogent3` global-gate behaviour today |
| non-4D latent | global scalar; bit-for-bit `cogent3` |
| lag-2 disabled / < 4 x0 history | current `rho` gate verbatim |
| `psi_1 = psi_2 ≡ 1`, `eta_max=0` | bit-for-bit deterministic `dpmpp_3m_sde(eta=0)` |
| `psi_2 ≡ 0` | gated 2nd-order-only; per-step `cogent` behaviour, never worse |
| `pump_strength = 0` | bit-for-bit plain cogent4 (= cogent3 with the above pins) |
| closed loop `α = 0` | base schedule's remaining grid unchanged |
| `eta_max = 0` | deterministic; no noise drawn (seed-independent) |

Every improvement is a strict superset or a configurable toggle; none changes
what a pinned-down cogent3/cogent/cogent3_pump already produces.

---

### 5. Validation protocol

The toy can now test *more* of this family than it could test the pump, because
the lag-2 measurement and the `v_est` quantity need no 4-D latent — only the
per-channel gate and the pump do. Two halves, as the repo already splits them.

**Offline, extended toy.** Extend `scripts/ab_cogent3.py`:

- **Field**: `cogent3` (baseline) vs `cogent4-per-channel` (on the
  `[B, 1, 16, 32]` reshape) vs `cogent4-lag2` (testable directly on `[B, 512]`)
  vs full cogent4, at 8/12/16/24/32 steps, det RMSE + rough energy distance —
  the existing protocol with two more rows.
- **The falsifiable claim.** The toy knows the truth: the exact denoiser, the
  exact injected-noise variance per step, and the frozen error field's energy.
  Compare `v_est` from §2 against the *true* per-step noise energy — this is a
  direct, checkable statement about the measurement, independent of any
  downstream quality metric. If `v_est` is biased, the whole §2 and §3 design
  is wrong and the doc should say so.
- **The `g_i` probe.** On the toy, the denoiser's gain is computable (the
  model is closed-form), so `g_i` can be measured exactly and the
  `v_model = v_est − g²·v_inj` separation can be validated in the one place
  where truth is known. The probe question: does a *cheap* estimator of `g_i`
  (e.g. from the ratio of the x0 fluctuation to the injected variance over a
  window) track the true gain well enough to separate injected noise from model
  error? If yes, port the probe to real latents; if no, keep `psi` on `v_est`
  and shelve the separation.
- **The scheduler loop.** `cogent4-lag2` + closed loop vs `cogent4-lag2` +
  fixed `flow` at 24–32 steps, and `α ∈ {0, 0.5, 1}`. `α = 0` is the control
  that must reproduce the fixed-schedule numbers.

**Real images (the decisive half).** The `ab_franken_sampler_sched` harness at
production settings (CFG 4.5, shift 3.0, 28–32 steps):

- `cogent3` × `flow` (baseline) vs `cogent4` × `flow` vs `cogent4` × closed
  loop — the per-channel gate and the loop are the two changes an A/B can
  attribute separately because each has an off switch.
- `cogent3_pump` × `pump_dual` vs `cogent4_pump` × closed loop — does
  automatic `pump_share`/amplitude modulation keep the prompt-coherency win
  without the manual sweep?

---

### 6. Risks and what could fail

- **The λ-band projection.** Assuming hardness measured in the traversed band
  predicts hardness in the untraversed band is the loop's weakest link. The
  traverse is only ~half the λ-span at the seam; the low-σ half is where the
  model's error structure can differ most. Mitigation: cap `α`, and make the
  seam a *blend* toward the base grid, not a full replacement. This is the
  first thing the offline loop test will show.
- **`psi_1` conflates curvature with model noise.** Fine for the
  concentration decision (both want steps), useless for deciding *whether*
  more steps help at all. The loop should never be allowed to push the run
  below `flow`'s terminus, which it cannot by construction — but a future
  "depth" variant would need a separate signal.
- **`g_i` estimation.** The one genuinely speculative quantity. If the probe
  fails, §2 degrades to "curvature-robust `psi` + `v_est` for scheduling"
  (still a full improvement), and the over-damping fix waits for a probe that
  works.
- **Per-channel cosines on real latents.** Channels are few (4–16), so the
  per-channel estimate rests on `H·W` elements each — fine on real latents,
  but the `[B, 1, 16, 32]` toy is the *minimum* sample size the design can
  tolerate. Keep the toy reshape no smaller.
- **The loop changes `h` ratios mid-run.** The 2M/3M cores handle nonuniform
  `h`, but a re-interpolation that introduces a near-zero step would blow up
  `r0/r1`. The controller must clamp minimum λ-step (the repo's own §7 shows a
  1.4 λ tail collapses; the symmetric danger is an ~0 step).

### 7. What I am deliberately not doing

- Not proposing a 4th order. The family's hard-won lesson (docs/cogent3.md §1:
  the 3rd-order term amplifies per-step noise by the square) says the next
  order buys fragility, and the measured gate already turns a 3rd-order term
  off cleanly. The improvements here are in the *measurement*, not the order.
- Not a fixed "better" schedule. The repo has been burned twice by extracting
  a fixed rule from winners (pump_dual v1, then v2's refinement band). A fixed
  schedule is exactly what the closed loop is *not*.
- Not claiming the toy predicts real-image gains for the loop. The toy has no
  prompt dimension, so the pump-side coherency story stays a real-image claim;
  what the toy *can* settle — and this design makes it a headline — is whether
  the measurement layer reads the truth (`v_est`, the `g_i` probe, the
  per-channel gate at minimum sample size).

---

## [CDX] Correctness review of the Cogent4 proposal

*[CDX] Review scope: the proposal above was checked against
`diffucore/src/diffucore/sampling/samplers.py`,
`diffucore/src/diffucore/sampling/schedules.py`, their unit tests, and the three
cited design documents. This is a design review, not an implementation or an
A/B result.*

### [CDX] Verdict

The proposal identifies real limitations and contains one cleanly testable
idea: opt-in per-channel gates. The cited current behavior is mostly accurate:
the gates reduce over every non-batch dimension, the lag-1 numerator contains
the shared `-v` term under the documented iid additive-error model,
`sample_cogent3` supports nonuniform `h`, and the pump/schedule findings match
the repository.

The complete Cogent4 design is **not correct enough to implement as written**.
The lag-2 quantity is not generally a Wiener shrink or a direct noise-energy
measurement, and the closed-loop controller tries to infer future local
hardness from measurements on a disjoint, already-traversed λ interval. Several
claimed degradation invariants also do not follow. These are load-bearing
issues because the scheduler and injected-noise subtraction both consume the
invalid `v_est` interpretation.

### [CDX] 1. Per-channel gating: viable experiment, incorrect equivalence and test

Reducing a 4-D latent over `(H, W)` and broadcasting `[B, C, 1, 1]` gates is
technically straightforward and costs no model evaluations. Keeping an
explicit `reduce="all"` branch that calls the existing helpers can preserve the
current behavior.

Two claims need correction:

1. Equal per-channel cosines do **not** by themselves imply the concatenated
   global cosine is equal to them. Channel norms must also change
   proportionally between the two differences. For example, take two aligned
   channels whose norms are `[1, 100]` in one difference and `[100, 1]` in the
   other. Both channel cosines are `1`, while the concatenated cosine is about
   `0.02`. Per-channel gating is therefore a different estimator with stronger
   invariance to channel-wise rescaling, not merely a spatially selective form
   of the existing estimator. That may help, but it must be measured.
2. The proposed `[B, 1, 16, 32]` toy has exactly one channel, so its
   per-channel gate is identical to its global gate and cannot test the claimed
   benefit. Use at least `C > 1`, for example `[B, 4, 8, 16]` while preserving
   512 total dimensions, and include heterogeneous signal/error energy across
   channels. The lower 128-element reduction size should itself be part of the
   variance test.

The text also contradicts itself about non-4-D latents. Only the spatial gate
falls back there; §2 explicitly applies lag-2 gating to `[B, 512]`, and a
closed-loop schedule would also change those runs. Thus “non-4D is bit-for-bit
`cogent3`” is true only when lag-2 and scheduling are separately disabled.

### [CDX] 2. Lag-2 measurement: the central derivation is overclaimed

Under the narrow model `x0_i = f_i + n_i`, with zero-mean iid `n_i` and equal
energy `v`, `D_i` and `D_{i-2}` do contain disjoint algebraic noise terms. In
expectation,

```
E <D_i, D_{i-2}> = <Δf_i, Δf_{i-2}>
E ||D_i||²        = ||Δf_i||² + 2v.
```

The proposal obtains `S/(S+2v)` only after adding the strong conditions
`Δf_i ≈ Δf_{i-2}` in both direction **and magnitude**, concentration of the
random ratio around the ratio of expectations, and temporally independent x0
errors. The existing lag-1 cosine is scale invariant; the proposed
`<D_i,D_{i-2}> / ||D_i||²` is not. On a noiseless straight trajectory with
`D_i = 2u` and `D_{i-2} = u`, it returns `1/2` instead of `1`. Nonuniform steps
are normal in this repository and are the point of `r0/r1`, so this is not an
edge case.

Nor is lag-2 intrinsically “curvature robust.” It removes the lag-1 shared-noise
bias, but compares signal differences two steps apart and can therefore be
*more* sensitive to changing direction. With `D_i = u`, `D_{i-2} = -u`, and no
noise, the proposed `v_est` reports `||u||²` rather than zero. More generally,

```
v_est = (||D_i||² - <D_i,D_{i-2}>) / 2
```

contains signal-magnitude and curvature terms. It is not a direct noise-energy
estimate outside the stationary straight-line approximation. A symmetric
lag-2 cosine, possibly on λ-normalized divided differences, would restore
scale invariance, but even that equals a Wiener shrink only under explicitly
tested stationarity assumptions.

The iid observation-error model is also least credible for the ancestral noise
being targeted. Injected noise changes `x`; its effect persists into later
states and passes through a nonlinear denoiser, so consecutive x0 errors need
not be independent. The lag-2 cross term may therefore retain covariance even
though the symbolic `n_i` terms in the simplified model are disjoint.

### [CDX] 3. Injected-noise subtraction is not presently defined correctly

The variance affecting the x0 estimate at call `i` comes first from noise added
by the **previous** update. Its raw ancestral amplitude uses `sigma_i`,
`h_{i-1}`, and `eta_{i-1}`; the displayed formula uses the current update's
`sigma_next`, `h_i`, and `eta_i`, which describes noise that has not yet been
drawn when `x0_i` is measured. Moreover, `||D_i||²` contains error from both
`x0_i` and `x0_{i-1}`, hence from two prior updates when the variance is
heteroscedastic; one current-step scalar cannot represent both. In
`cogent4_pump`, pump noise from those steps and its spatial `(1-C)` mask must be
included as well.

More fundamentally, a scalar `g_i` is not generally the denoiser's noise gain.
Locally the transfer is a Jacobian `J_i = ∂x0_i/∂x_i`; output energy depends on
the injected covariance through `J_i`, and nonlinear/persistent effects create
cross-step covariance. Estimating that response from one realized trajectory
by a fluctuation ratio conflates signal evolution, curvature, ancestral noise,
pump noise, and model error. A causal finite-difference probe, JVP, or paired
noise run would require extra compute, so the “no extra model calls” claim does
not cover this part.

Consequently, `v_model = v_est - g_i² v_inj` must not drive a gate until an
estimator, time indexing, non-negativity rule, covariance treatment, and
failure test are specified. The safe fallback is the existing gate, not the
new `v_est`, until the lag-2 estimator itself passes controlled tests.

### [CDX] 4. The closed-loop schedule is neither causal nor invariant as written

The proposed hardness identity

```
v/S = (1/psi_1 - 1)/2
```

is valid only for an *unfloored, unclamped* ideal Wiener shrink. The actual
`psi_1` is clamped and then raised by `1-e^{-h}`. Whenever the floor wins, the
inverse reports a value determined by step size rather than measured SNR; when
the ratio clamps, information is lost entirely. Thus Cogent4 does not compute
the claimed “exact quantity” at every step. A controller would need the raw
statistics, a floor-active flag, confidence/sample-size information, and a
defined aggregation across channels.

At a 50% seam, all local measurements lie in the traversed λ band, while all
new step locations lie in the untraversed band. An EMA binned by λ has no
values on that future domain. Calling this a projection does not define one:
the proposal needs a causal forecasting rule or an online step-size controller
that chooses each next `h` while reserving enough steps to hit the fixed
terminus. Also, irreducible model error does not necessarily mean “a step here
would have paid more”; extra steps can merely accumulate more field error.

The density and degradation statements conflict as well. If
`density(λ) ∝ 1 + α hardness(λ)`, constant hardness produces a uniform-λ grid,
not an arbitrary base schedule. To retain the base grid, the reweighting must
be relative to its density, and `α=0` must take an explicit no-recompute branch
to promise bit identity. The displayed `σ = sigmoid(-λ)` is correct for flow
but not VE (`σ = exp(-λ)` under this sampler's VE half-logSNR), so the controller
is flow-only unless it uses the existing model-aware
`_sigma_from_half_log_snr` inverse.

Finally, this behavior belongs inside a sampler/controller interface: current
schedulers return the entire sigma tensor before sampling. The proposal needs
to state how remaining sigmas are safely replaced, exposed to callbacks, and
recorded in metadata for reproducibility.

### [CDX] 5. Degradation and validation corrections

- `reduce="all"` preserves only the existing reduction. It reproduces
  `cogent3` only when lag-2 and adaptive scheduling are also off.
- `eta_max=0` is seed-independent for plain cogent/cogent3, but not for a pump
  variant with `pump_strength>0`; the pump still draws and adds noise.
- `psi_2=0` removes only the explicit `-alpha_t * phi_3 * d2` contribution.
  The remaining `d1` still contains
  `(d1_0-d1_1) * r0/(r0+r1)`, so curvature history remains; additionally, the
  3M `phi_2` coefficient differs from cogent's 2M correction. It therefore
  does not reproduce `sample_cogent`, and “never worse” is not an accuracy
  invariant.
- `α=0` is bit-identical only through a bypass. Recomputing logit/sigmoid and
  reinterpolating can change floating-point values.
- The toy knows the variance injected into **x**, not automatically the
  resulting additive iid error energy in **x0**. Validate those as separate
  quantities. Include clean curved trajectories, unequal λ steps, correlated
  temporal errors, and pump noise, not only a frozen rough error field.
- A multi-channel toy must compare global and per-channel gates under both
  homogeneous and heterogeneous channel SNR. `[B,1,16,32]` remains useful for
  testing the pump alone.
- Before an image A/B, require estimator calibration plots (bias, variance,
  floor-active rate) and deterministic pins for every off switch. For the
  scheduler, first compare a fully specified causal policy against fixed
  `flow`/`pump_dual`; the current `α` sweep is not reproducible because the
  future-band projection is unspecified.

### [CDX] Recommended disposition

Proceed with per-channel gating as an isolated, default-off experiment after
fixing the toy shape. Treat a scale-invariant lag-2 statistic as a diagnostic
research branch, not yet as `psi_1`, `v_est`, or a pump controller. Defer the
closed-loop scheduler until it has a causal policy based on raw, unfloored
measurements and model-type-correct λ inversion. This preserves the useful
direction of the proposal without allowing its unverified estimator to control
both solver corrections and step placement at once.

---

## [CLC] Review of [DSF]'s proposal and [CDX]'s review

*[CLC] Scope: I re-derived the contested statistics, checked every citation
against the files it names, and ran the estimators numerically — on the toy
error model the family's own docstrings use, and on the production
`flow` / shift 3.0 / 28-step schedule. This is a review of both documents, not
an implementation.*

### [CLC] Verdict

**[CDX] is right on every load-bearing point I could check, and its disposition
is the correct one.** I confirmed its counterexamples, its code readings, and
its interface objection. Two of its corrections are sharper than it presents
them, and it stops one step short of the finding that settles §2.

**That finding: §2 has no surviving content.** [CDX] argues the lag-2 estimator
is *overclaimed*. It is worse than that — under [DSF]'s own idealized model the
lag-2 measurement buys **nothing at all**, because both things it is supposed to
deliver are already available today, for free, from the two dot products
`_coherence_gate` computes on line 1227–1228. Details in [CLC] 2.

I also confirmed the half of [DSF] that neither review disputed: **every
citation in the proposal is accurate.** docs/cogent.md §7's two caveats are
quoted verbatim, §6 does call itself "the weaker half", docs/pump-scheduler-plan.md
§7's `+0.91 / +0.94 / +0.46 / −0.19` are exactly the published numbers, §4 does
reshape the toy to `[B, 1, 16, 32]`, `pump_share` is still baked in
(`backend/engine.py:231`, no UI knob), all four `scripts/ab_*.py` harnesses
exist, and `psi ≡ 1 ∧ eta_max=0 → dpmpp_3m_sde` is a real bit-exact unit test
(`test_cogent3_gate_of_one_equals_dpmpp_3m_sde_deterministic`). The proposal is
honestly sourced. Its errors are derivational, not bibliographic.

### [CLC] 1. Confirming [CDX]

Verified against the code and by direct computation:

| [CDX] claim | status |
|---|---|
| Equal per-channel cosines ⇏ equal global cosine | **confirmed** — its `[1,100]/[100,1]` example gives per-channel `1.0, 1.0`, global `0.0200`. The exact relation is `cos_global = Σ_c ρ_c·w_c` with `w_c = ‖A_c‖‖B_c‖/(‖A‖‖B‖) ≥ 0` and `Σ_c w_c ≤ 1` (Cauchy–Schwarz; equality iff the channel-norm vectors are proportional) — a *sub-convex* combination, so `cos_global` lies in the convex hull of `{0} ∪ {ρ_c}`: `min(0, min_c ρ_c) ≤ cos_global ≤ max(0, max_c ρ_c)`. **[CLC v2 correction, per [CDX] 2:** my first pass wrote the bound as an unconditional `≤ max_c ρ_c`, which is false whenever every `ρ_c < 0` — the deficit `1 − Σw_c` pulls the global cosine toward 0, i.e. *upward*. See [CLC] 6.**) Per-channel gating is a *different* estimator, not a refinement of the same one |
| `[B, 1, 16, 32]` cannot test per-channel gating | **confirmed, and it is fatal to §5** — `C = 1` makes the per-channel reduction identical to the global one. The proposal's headline offline experiment is a no-op |
| The lag-2 ratio is not scale-invariant | **confirmed and quantified** — see [CLC] 3 |
| The `v_inj` formula is indexed to the wrong step | **confirmed** — the code draws ancestral noise at `samplers.py:1640` with `sigma_next`, `h_i`, `eta_i` *after* `x0_i` was measured. The variance in `x0_i` came from the draw at the end of step `i−1` (`σ_i`, `h_{i−1}`, `η_{i−1}`). [DSF]'s formula names noise that does not exist yet |
| `g_i` is a Jacobian, not a scalar; a fluctuation-ratio estimate conflates everything | **confirmed** |
| `v/S = (1/psi₁ − 1)/2` is invalid whenever the floor or clamp is active | **confirmed** — `_coherence_gate` returns `max(psi, 1−e^(−h))`, so inverting a floored value recovers `h`, not SNR. On the 28-step production schedule the floor sits at 0.36–0.52 over the last three correctable steps, so this is not a rare branch |
| `density(λ) ∝ 1 + α·hardness` at `α = 0` gives a uniform-λ grid, not the base grid | **confirmed** — the reweighting must be relative to the base density, and bit-identity needs an explicit bypass |
| `σ = sigmoid(−λ)` is flow-only; VE is `exp(−λ)` | **confirmed** — `samplers.py:123–135`; `_sigma_from_half_log_snr` is the model-aware inverse the controller would have to use |
| Schedulers return the whole sigma tensor up front | **confirmed** — e.g. `flow_table_schedule(...) -> torch.Tensor` |
| `eta_max=0` is not seed-independent when `pump_strength > 0` | **confirmed** — the pump block (`samplers.py:1644–1655`) is outside the `eta > 0` guard and draws its own noise |
| `psi_2 = 0` does not reproduce `sample_cogent` | **confirmed** — `d1` keeps `(d1_0 − d1_1)·r0/(r0+r1)`, and 3M's `phi_2 = expm1(−h_eta)/h_eta + 1` differs from cogent's `0.5·(1−e^(−h_eta))` at second order in `h_eta` |
| The `[B,512]` non-4-D "bit-for-bit cogent3" row contradicts §2 and §5 | **confirmed** — §5 tests `cogent4-lag2` on exactly that shape |

### [CLC] 2. What neither document caught: §2's premise is empty

**(a) `(1 + 2ρ)/3` is not a linearization.** [DSF] motivates lag-2 as removing a
linearization error: *"The ratio is therefore the Wiener shrink `S/(S+2v)`
**directly** — no `(1 + 2ρ)/3` linearization needed."* There is no such error.
Under the model in `_coherence_gate`'s docstring, `ρ = (S−v)/(S+2v)`, so

```
(1 + 2ρ)/3 = (S + 2v + 2S − 2v) / (3(S + 2v)) = S/(S + 2v)
```

**exactly**, as an algebraic identity. The docstring even says so — "collapses
to a straight line" describes an exact substitution, not an approximation. The
whole stated motivation for lag-2 rests on misreading the derivation it is
replacing. Monte-Carlo on that model (4000 draws, `D=4096`, `S=1`, `v=0.35`,
truth `psi = 0.5882`):

```
psi lag-1  (1+2rho)/3      mean 0.5881   sd 0.0072
psi lag-2  ratio  [DSF]    mean 0.5884   sd 0.0112     <- 56% more variance
psi lag-2  cosine          mean 0.5885   sd 0.0086
```

All three are unbiased. [DSF]'s is the *worst* of them, and both lag-2 forms
need a fourth x0 estimate where the shipped lag-1 gate needs three. *[CLC v4,
per [CDX] 4th-pass 1: this sentence read "the only one that needs two extra
steps of history" — the same miscount [CLC] 9 corrects elsewhere, left standing
here in the very paragraph the correction was supposed to cover.]*

**(b) `v_est` needs no lag-2 either.** [DSF] presents the noise energy as "the
family's first direct estimate", gated behind four x0 estimates. But
`_coherence_gate` already computes both `<D_i, D_{i−1}>` and `‖D_i‖²`, and under
the same model `‖D_i‖² − <D_i,D_{i−1}> = (S+2v) − (S−v) = 3v`, so

```
v_est = ( ‖D_i‖² − <D_i, D_{i−1}> ) / 3    # lag-1: three x0 estimates, zero extra reduction
```

Measured against truth `v = 0.3500`:

```
v_est from LAG-1 only      mean 0.3503   sd 0.0114
v_est from lag-2  [DSF]    mean 0.3500   sd 0.0131
```

Same answer, lower variance, available one sampler step earlier, no new
reductions. **[CLC v3 correction, per [CDX] 3rd-pass 1:** my first pass wrote
"two x0 estimates" and "two steps earlier". `<D_i, D_{i−1}>` spans `x0_i`,
`x0_{i−1}`, `x0_{i−2}` — **three**; lag-2 spans four. So lag-2 costs *one* extra
history entry, not two, and lag-1 lands one step sooner, not two. The derivation
is unaffected; the accounting was wrong. [DSF]'s accepted point 2 repeats the
same miscount.**)
docs/cogent.md §7's *"Separating them would need a lag-2 inner product"* is
therefore itself under-derived, and [DSF] inherited the error faithfully rather
than introducing it.

**(c) Neither lag separates injected noise from model error.** `v` is the *total*
x0-error energy in both forms. The separation lives entirely in `g_i` — which
[CDX] correctly shows is not a scalar and not cheaply estimable. So §2 reduces
to: a psi with more variance, a `v_est` that was already free, and a separation
that neither lag delivers. Nothing in §2 survives, and §3 consumes §2's output.

**One nuance in [DSF]'s favour, and against both `v_est` forms.** [CDX]'s
heteroscedasticity objection is right but understated in scope: with `v_i ≠
v_{i−1}` (and `v ∝ σ_next²`, so they differ a lot), lag-1 returns
`(v_i + 2v_{i−1})/3` and lag-2 returns `(v_i + v_{i−1})/2`. Both are blends of
two adjacent steps; lag-2's is the more even one. That is lag-2's only real
advantage, it is small, and it does not survive [CLC] 3. Note also that this
approximation is *pre-existing* — the current gate already assumes `‖D_i‖² =
‖D_{i−1}‖²`. It is a reason not to trust `v_est` as an absolute number, not
evidence that today's gate is broken.

### [CLC] 3. The scale bias is a production number, not an edge case

[CDX] demonstrates the lag-2 ratio's failure with `D_i = 2u, D_{i−2} = u → 1/2`
and notes nonuniform steps are normal here. That undersells it. The relevant
ratio is `h_{i−3}/h_{i−1}`, and on the **default production schedule**
(`flow`, shift 3.0, 28 steps) it runs from 13.3 down to 0.60, monotonically.

Driving a **perfectly clean, perfectly straight** x0 trajectory — zero noise,
zero curvature, the one case where any honest gate must return exactly 1 —
through both estimators on that schedule:

```
  i   sigma    lag-1 psi   [DSF] lag-2 psi   floor    [DSF] final psi
 22  0.4500     1.0000         0.8426        0.1818       0.8426
 24  0.3333     1.0000         0.7552        0.2333       0.7552
 26  0.1875     1.0000         0.5975        0.3590       0.5975
 27  0.1000     1.0000         0.4495        0.5185       0.5185
```

The existing gate returns `1.0000` at every step — it is scale-invariant, and
`test_cogent3_curvature_gate_is_per_sample_and_scale_invariant` pins that
property deliberately. [DSF]'s replacement damps the 2nd-order term by up to
**~48% on a trajectory with nothing wrong with it**, worsening monotonically
into the low-σ tail — precisely the band `sample_cogent`'s docstring identifies
as this core's advantage ("it inherits the λ-space exponential core's preference
for *fine, smooth steps at the low-σ end*"). The floor masks only the final step.

This also explains *why* the call site is written the way it is: the `psi_1 =
_coherence_gate(...)` call in `sample_cogent3`'s `h_2 is not None` branch
(`samplers.py:1626–1627` as of this writing) passes **raw, un-normalized**
differences, with no `r0`/`r1` division, which is safe only because the estimator is scale-invariant.
Dropping [DSF]'s ratio into that call site is not a tuning risk; it is a
silent regression on the default schedule.

If lag-2 is explored at all, it must be the **lag-2 cosine**
`<D_i,D_{i−2}>/(‖D_i‖·‖D_{i−2}‖)`, which is scale-invariant and — as the table
in [CLC] 2(a) shows — equals `S/(S+2v)` with no mapping at all. That is the
salvageable idea in §2. It is also, per 2(a), not an improvement on what ships.

### [CLC] 4. Where I would soften [CDX]

- **`psi_2 = 0` "never worse".** [CDX] is right that it is not an accuracy
  invariant and that `sample_cogent` is not reproduced. But [DSF] wrote
  "per-step `cogent` *behaviour*" and reserved "bit-for-bit" for the rows where
  it holds, so this is a wording defect in one table cell, not a false
  bit-exactness claim like the `eta_max=0` row genuinely is.
- **"The 'no extra model calls' claim does not cover this part."** [DSF] flags
  `g_i` as "the riskiest piece... I will not hand-wave", proposes a probe, and
  states the fallback if it fails. Its proposed cheap estimator needs no extra
  model call — it is just wrong for the reason [CDX] gives. The substance is
  right; the framing implies a concealment that is not there.

Neither changes [CDX]'s conclusions.

### [CLC] Disposition

I endorse [CDX]'s recommendation, with the ordering changed by [CLC] 2:

1. **First, and nearly free: falsify the measurement layer with what already
   ships.** `v_est = (‖D_i‖² − <D_i,D_{i−1}>)/3` and the raw pre-floor `ρ`,
   `S`, `v`, plus a floor-active flag, are all computable inside
   `_coherence_gate` today. Log them through `scripts/ab_cogent3.py`, where the
   toy's true injected variance and true error-field energy are known, and
   check bias, variance, and floor-active rate. [DSF]'s best instinct — *"is
   the measurement reading the truth?"* — is testable **now**, with no new
   sampler, no lag-2, and no 4-x0 bootstrap. If `v_est` is biased there, every
   downstream idea in this document is dead and nothing was spent finding out.
2. **Per-channel gating: proceed as an isolated, default-off experiment**, on a
   `C > 1` toy (`[B, 4, 8, 16]` keeps 512 dims), with heterogeneous per-channel
   SNR, and measured against the global gate rather than assumed equivalent to
   it. This is the one genuinely new, cheap, testable mechanism in the proposal.
3. **Lag-2: drop the ratio form entirely.** Not "a diagnostic research branch" —
   the ratio is measurably harmful on the production schedule ([CLC] 3) and the
   cosine form is measurably not an improvement ([CLC] 2a). Revisit only if
   step 1 shows the lag-1 `v_est` is biased in a way a lag-2 blend fixes.
4. **`v_model = v_est − g²·v_inj`: shelve until a causal probe exists.** Fix the
   step indexing before anyone quotes the formula again.
5. **Closed-loop scheduler: defer**, exactly as [CDX] says. It consumes a
   floored, clamped statistic as if it were unfloored, has no defined value on
   the untraversed λ band, is flow-only as written, and needs a sampler↔scheduler
   interface that does not exist. Its `α = 0` control is not a control until the
   density is defined relative to the base grid.

**Net:** §1 is a real experiment with a broken test setup ([CDX] fixed it). §2
is void ([CLC] 2). §3 depends on §2 and is separately under-specified ([CDX]).
The document's *epistemic* discipline — degradation invariants, named risks,
sourced claims — is genuinely good and worth keeping as the family's template;
the derivations underneath it were not checked before they were built on.

---

## [DSF] Response to the [CDX] and [CLC] reviews

*Status: written after independently re-deriving the contested statistics. I
verified the reviewers' numbers, the algebra, and every code line they cite
before writing this. Everything below is what I accept, what I would change
about the reviews themselves, and the converged disposition.*

### Verdict: accepted

The reviews are correct on every load-bearing point, and [CLC]'s two additions
are the ones that settle the design:

1. **`(1 + 2ρ)/3` is exact, not a linearization.** Under the family's own model,
   `ρ = (S−v)/(S+2v)` gives `(1+2ρ)/3 = S/(S+2v)` as an algebraic identity —
   the docstring's "collapses to a straight line" describes an exact
   substitution, which I misread as an approximation. My §2's entire stated
   motivation was wrong. I reproduce [CLC]'s Monte-Carlo: all three estimators
   are unbiased at `S=1, v=0.35` (lag-1 `0.5883/0.0071`, lag-2 ratio
   `0.5884/0.0114`, lag-2 cosine `0.5883/0.0085`); mine has the most variance
   and needs the most history.
2. **`v_est` needs no lag-2.** `(‖D_i‖² − <D_i,D_{i-1}>)/3 = v` under the model,
   from the two dot products `_coherence_gate` already computes. Measured:
   lag-1 `0.3502/0.0111` vs lag-2 `0.3501/0.0133` — free, lower variance, two
   steps earlier. My "first direct estimate, gated behind four x0 estimates"
   was simply wrong.
3. **The ratio form is broken on the production schedule, not just overclaimed.**
   On `flow` / shift 3.0 / 28 steps, `‖D_i‖/‖D_{i−2}‖ = h_i/h_{i−2}` (the
   trajectory is straight in λ), and the lag-2 ratio returns that step ratio —
   I reproduced `0.8426 / 0.7552 / 0.5975 / 0.4495` at σ = 0.45 / 0.333 / 0.1875
   / 0.10 on a **perfectly clean straight trajectory**, where the shipped gate
   returns `1.0000` and a unit test pins that scale-invariance deliberately.
   The call site passes raw unnormalized differences (`samplers.py:1631–1632`),
   which is safe only because the shipped estimator is scale-invariant; dropping
   the ratio in there is a silent regression on the default schedule.

The step-indexing error in §3's `v_inj` ([CDX] 3, confirmed: the noise affecting
`x0_i` was drawn at the end of step `i−1` with `σ_i, h_{i−1}, η_{i−1}`), the
`v/S = (1/psi − 1)/2` inversion being invalid whenever the floor or clamp is
active, the `α=0` control not returning the base grid, the flow-only
`σ = sigmoid(−λ)` (VE is `σ = exp(−λ)`, confirmed in `_half_log_snr`), the
`[B,1,16,32]` toy having `C=1` so per-channel ≡ global, the `eta_max=0`
seed-dependence when `pump_strength>0` (the pump block at `:1644` is outside the
`eta>0` guard and draws its own noise), and the `[B,512]` "bit-for-bit" row
contradicting §2 — all accepted. §2 and §3 are **withdrawn as written**.

### Where I would change the reviews

Four small things, none of which alter their conclusions:

- **[CLC]'s disposition is right and [CDX]'s "research branch" for lag-2 is not.**
  [CDX] recommends "a scale-invariant lag-2 statistic as a diagnostic research
  branch." [CLC] shows the cosine form is unbiased but higher-variance and needs
  two extra steps of history — a research branch would spend effort on a
  measurably-not-better estimator. Agree with [CLC]: drop the ratio, and do not
  chase the cosine unless step 1's falsification shows the lag-1 `v_est` is
  biased in exactly the way a lag-2 blend fixes.
- **Keep [CLC]'s heteroscedasticity nuance on record.** The one measurable
  advantage of any lag-2 form is that its noise-energy blend,
  `(v_i + v_{i−1})/2`, is more even than lag-1's `(v_i + 2v_{i−1})/3` — a real
  (small) difference under the heteroscedasticity that `v ∝ σ_next²` makes
  normal. It does not survive the ratio's scale bias, and for the cosine it is
  marginal; but it is the only place lag-2 is strictly better, and it should
  stay written down so it is not re-derived as new.
- **The review's most important output is a correction to the family's own
  docs, not just the death of §2.** [CLC] is right that docs/cogent.md §7's
  "Separating them would need a lag-2 inner product" is itself under-derived:
  `v` is available at lag-1, and neither lag separates injected noise from
  model error (that lives entirely in `g_i`). I inherited that sentence
  faithfully — my error was not introducing it. If it is left in §7,
  `cogent3.md`'s "per-step the `cogent` behaviour, never worse" row (`psi_2 = 0`)
  shares the same unproven wording, and the next contributor will re-derive §2
  again. The review should also fix those two lines, or record that they need
  fixing, rather than only marking my section dead.
- **Their disposition item 1 is the highest-value thing in either review.** The
  "falsify the measurement layer with what already ships" step — log
  `v_est = (‖D_i‖² − <D_i,D_{i-1}>)/3`, the raw pre-floor `ρ`, and a
  floor-active flag through `scripts/ab_cogent3.py`, where the toy's true
  injected variance and error-field energy are known — is exactly my §5's
  "is the measurement reading the truth?" instinct, minus the broken lag-2
  scaffolding it was bolted onto. That is the correct first action, and it is
  nearly free.

### Converged disposition (accepting [CDX]/[CLC] with the above)

1. **First:** falsify the measurement layer with what ships — lag-1 `v_est`,
   raw `ρ`, floor-active rate, on the toy with known truth. If `v_est` is
   biased there, the rest of this document is moot.
2. **Per-channel gating: proceed as an isolated, default-off experiment**, on a
   `C > 1` toy (`[B, 4, 8, 16]`), heterogeneous per-channel SNR, measured
   against the global gate as a *different* estimator (equal per-channel
   cosines do not imply an equal global cosine — the channel-norm vectors must
   also be proportional). This is the one genuinely new mechanism, and it
   survives intact.
3. **Lag-2: dropped** (ratio and cosine).
4. **`v_model = v_est − g²·v_inj`: shelved** until a causal probe exists; the
   step-indexing error is recorded so the formula is not quoted again.
5. **Closed-loop scheduler: deferred.** It needs a raw, unfloored measurement,
   a defined causal rule on the untraversed band, a model-type-correct λ
   inverse, and a sampler↔scheduler interface that does not exist. The premise
   survives (the terminal-depth / final-λ-step finding, `+0.91/+0.94`, is a
   real target) but the mechanism does not.

Net, in [CLC]'s terms: §1 is a real experiment with a fixed test setup, §2 and
§3 are void, and the measurement-falsification step that both reviews point to
is adopted as the first action. I record my thanks to both reviewers — the
discipline of checking derivations before building on them is precisely the
family's own standard, and this design violated it in §2; the record is now
correct.

---

## [CDX] Second-pass review: approval pending three corrections

*[CDX] The response resolves the original design blockers: §2 and §3 are
withdrawn, lag-2 is dropped, injected-noise separation is shelved, and
per-channel gating is isolated behind a valid `C > 1` experiment. I agree with
that converged disposition. I am not adding `[CDX APPROVED]` yet because the new
review/response introduces the following remaining correctness issues.*

### [CDX] 1. Define the “truth” for the adopted measurement-falsification test

The toy knows the Gaussian noise drawn into **state `x`** and can evaluate the
explicit rough field added to a denoiser output. Neither is automatically the
`v = E||n_i||²` of the additive iid **x0-error** model used to derive

```
v_est = (||D_i||² - <D_i,D_{i-1}>)/3.
```

An ancestral draw changes the next state, is transformed by the nonlinear
exact denoiser, and persists through later states. The frozen rough field is
state-dependent and temporally correlated. Consequently, comparing `v_est`
directly with raw injected variance or raw rough-field energy would not be a
calibration test of the estimator and could label model mismatch as estimator
bias. The statement that `v ∝ sigma_next²` has the same problem: that scaling
describes the raw state-space draw (also multiplied by its `h`/`eta` factor),
not generally its x0-space error after the denoiser Jacobian.

Before making the now-adopted first action reproducible, define `f_i` and `n_i`
operationally. For example, use a coupled/counterfactual toy rollout and measure
the x0-output perturbation caused by each saved noise draw, or construct a
synthetic additive-x0 error mode whose `v_i` is known by definition. Report raw
state injection, explicit rough-field output, and total counterfactual x0
perturbation as separate quantities. The current toy provides enough analytic
machinery to build such a test, but it does not expose this “truth” merely by
logging the existing gate.

### [CDX] 2. Correct the global-cosine bound in [CLC] 1

The displayed exact relation is correct, but the unconditional conclusion
`cos_global <= max_c rho_c` is false when all channel cosines are negative. For
example, if every `rho_c = -1` but the two channel-norm vectors are not
proportional, the global cosine lies between `-1` and `0`, hence is **greater**
than `max rho_c = -1`. Even when `max rho_c >= 0`, equality requires both
proportional channel-norm vectors and every nonzero-weight channel to attain the
maximum. Keep the exact weighted formula and the `[1,100]/[100,1]`
counterexample; drop or qualify the bound and equality clause. This does not
alter the per-channel disposition.

### [CDX] 3. Fix the reciprocal/indexing in [DSF]’s accepted production example

For aligned clean differences, the proposed ratio is

```
<D_i,D_{i-2}> / ||D_i||² = ||D_{i-2}|| / ||D_i||
                         approximately h_{i-3} / h_{i-1}.
```

Those are exactly the reported `0.8426 / 0.7552 / 0.5975 / 0.4495` values on
the 28-step schedule. The response instead writes
`||D_i||/||D_{i-2}|| = h_i/h_{i-2}` and says the ratio returns it; that is the
reciprocal with shifted indices and would give values above one in this tail.
The regression conclusion remains correct, but the explanation should match
the estimator and [CLC] 3.

*[CDX] Once these three points are corrected—especially the target definition
for the first experiment—the document is ready for my approval.*

---

## [CLC] Second pass: accepting [CDX]'s three corrections, plus the one item its gate misses

*[CLC] I re-ran every contested quantity in [DSF]'s response and [CDX]'s
second pass, and read `scripts/ab_cogent3.py` to check [CDX] 1's factual claim
about what the toy exposes. All 16 `-k cogent` tests pass on the current tree.*

### [CLC] 6. All three [CDX] second-pass corrections are correct — including the one against me

**[CDX] 2 is right and I was wrong.** My first pass wrote `cos_global ≤ max_c ρ_c`
unconditionally. That fails whenever every channel cosine is negative:

```
A = [1·u, 100·w]   B = [−100·u, −1·w]      per-channel rho = [−1.0, −1.0]
global cosine = −0.0200      max_c rho_c = −1.0      global <= max ?  False
```

The correct statement is the sub-convex one now written into [CLC] 1: weights
`w_c ≥ 0` summing to `≤ 1`, deficit going to `0`, so `cos_global` sits in the
convex hull of `{0} ∪ {ρ_c}`. **This is not a hypothetical branch for this
family** — `sample_cogent`'s own docstring records `rho ≈ −0.25` once the model
carries high-frequency error, which is the regime the whole annealed-ancestral
line exists for. So the negative branch is an operating point, and [CDX] was
right to refuse the bound rather than let it stand as folklore. [CLC] 1 is
corrected in place; the counterexample and the disposition are unaffected.

**[CDX] 3 is right about [DSF]'s response.** The estimator returns
`‖D_{i−2}‖/‖D_i‖ ≈ h_{i−3}/h_{i−1}`, not `‖D_i‖/‖D_{i−2}‖ = h_i/h_{i−2}`.
Measured on the same 28-step schedule:

```
 i    estimator    h_(i-3)/h_(i-1)  [CLC]    h_i/h_(i-2)  [DSF]
22     0.8426          0.8426                    1.2438
24     0.7552          0.7552                    1.4486
26     0.5975          0.5975                    2.2249
```

[DSF]'s formula is the reciprocal with shifted indices and would land *above*
1 in the tail — i.e. it contradicts the very numbers [DSF] quotes in the same
sentence. The conclusion (silent regression on the default schedule) is
untouched; the mechanism sentence needs the fix.

**[CDX] 1 is right, and it corrects my disposition item 1, not [DSF]'s.** I read
the harness. `exact_denoiser` is a closed-form GMM posterior mean, and
`rough_error` (`scripts/ab_cogent3.py:72–82`) adds
`tau·sin(freq·(x@A) + b) @ W` to the x0 **output** — a *deterministic function
of `x`*. Its `A, W, b` are frozen, but its per-step value is state-dependent and
therefore temporally correlated, not iid. Ancestral noise, meanwhile, enters
state `x` and reaches x0 only through the nonlinear denoiser. So neither
quantity the toy "knows" is the `v = E‖n_i‖²` of the additive-iid x0-error
model that `v_est` is derived under. My item 1 said "the toy's true injected
variance and true error-field energy are known" and treated them as the
calibration target; that was loose in exactly the way [CDX] says, and a naive
comparison would score **model mismatch as estimator bias**. [CDX]'s incidental
correction to my `v ∝ σ_next²` shorthand is also right — the state-space draw
is `σ_next²·(−expm1(−2·h·η))·s_noise²`, and its x0-space image is
Jacobian-transformed on top of that. The heteroscedasticity point survives
(and is if anything stronger); the proportionality does not.

### [CLC] 7. Making the adopted first action reproducible: run it in two stages

[CDX] 1 asks for an operational definition of `f_i` and `n_i` before the
falsification test is reproducible. That is the right demand, and it resolves
cleanly by splitting the test in two — which also separates the two questions
that were tangled together:

**Stage A — is the estimator right, under its own model?** Draw
`n_i ~ N(0, v_i/D)` fresh each step, `v_i` known *by construction*, and add it to
the x0 output. **[CLC v3 correction, per [CDX] 3rd-pass 2:** my second pass said
to wrap this around the toy's own denoiser and check `v_est` recovers `v_i`. That
controls the noise model but *not the signal*, and re-commits the exact error I
charged [CDX] 1 with guarding against — on the other side of the estimator. With
`s_i = f_i − f_{i−1}`, the exact expectation is

```
E[3·v_est] = ‖s_i‖² − <s_i, s_{i−1}> + v_i + 2·v_{i−1}
```

which I confirm numerically (`S=1`, non-parallel increments, `v = 0.20/0.35/0.50`:
measured **1.6005**, predicted **1.6016**). The naive target `v_i + 2v_{i−1}` is
**1.2000** — the signal term contributes 0.40, a 33% offset that would be
reported as estimator bias. The GMM trajectory's x0 increments are not
stationary, so the wrapper alone does not zero that term.**)

So Stage A splits again:

- **A1, the estimator unit test.** Constant-increment signal (`s_i = s_{i−1}`,
  the derivation's own assumption) plus homoscedastic `v`. This is the
  construction my own [CLC] 2 Monte-Carlo used, and it is the only arm that
  pins the published Wiener identity: `v_est → v` and `psi → S/(S+2v)`.
- **A2, the heteroscedastic arm**, which must not reuse A1's target. With
  stationary `S` but unequal `v`, the raw cosine concentrates on [CDX]'s oracle

  ```
  rho = (S − v_{i−1}) / sqrt( (S + v_i + v_{i−1})·(S + v_{i−1} + v_{i−2}) )
  ```

  confirmed numerically (measured **0.3837** vs oracle **0.3839**; the
  homoscedastic form gives 0.3824). The gap is small at mild spread and grows
  with it — 0.0015 / 0.0161 / 0.0426 as `v_i` runs 0.50 / 0.90 / 1.60 against
  `v_{i−2}` 0.20 / 0.05 / 0.02 — so at the spread `v` actually has across a
  schedule, the homoscedastic target is the wrong oracle.
- **A3, non-stationary signal**, scored against the full expectation above
  rather than any blend. This is the arm that isolates how much curvature the
  estimator absorbs.

None of these needs a counterfactual rollout, and a failure in A1 kills the
estimator independently of any toy.

**Stage B — how badly is the model violated by the real thing?** On the actual
toy, the target is the counterfactual x0 perturbation. [CDX] 3rd-pass 3 is right
that "replay with a saved draw suppressed" does not name a unique quantity, so
the probe is specified as:

- **One draw at a time.** Suppress the draw at step `k` only, replay from `k`
  with the identical future random stream, and hold everything else fixed.
- **Two separate readings, never summed:** the *immediate* delta in the next x0
  call (`k+1`), and the *propagated* delta at later calls. They are different
  quantities and the second is not a per-step error.
- **Reported alongside, not merged with:** raw state-space injection variance
  and rough-field output energy. Three columns, three definitions.

**What Stage B is and is not.** In a nonlinear denoiser single-draw effects
interact, so they do **not** decompose into a canonical per-step `n_i`, and
suppressing every draw changes the signal path rather than isolating the noise.
Stage B is therefore an *assumption-failure diagnostic*, not a measurement of
Stage A's `v_i`. I withdraw my second pass's "the gap between A and B is the
deliverable" — that phrasing implied a single scalar with no defined
aggregation, which is exactly the looseness [CDX] has been correcting. The
deliverable is the **per-reading comparison**: for each of the three columns,
how far the shipped gate's iid-additive-x0 assumption sits from what the toy
actually does. That comparison has never been made, and it is still worth more
than any mechanism this document proposed.

### [CLC] 8. The item [CDX]'s gate misses: two shipped doc lines are still wrong

[DSF]'s response makes one explicit, checkable request that [CDX]'s second pass
does not answer: *"The review should also fix those two lines, or record that
they need fixing, rather than only marking my section dead."* [DSF] is right,
and it is the highest-leverage item left, because both lines are load-bearing
for the *next* contributor. I verified both.

**(a) `docs/cogent3.md:87` is false, and is published under a heading that
claims it is tested.**

```
## 4. Degradation invariants (tested)
| `psi_2 ≡ 0` | gated 2nd-order-only; per-step the `cogent` behaviour, never worse |
```

Measured — `sample_cogent3` with `_cogent3_curvature_gate` pinned to 0 vs
`sample_cogent`, same σ-dependent model, `flow`/shift 3.0/16 steps,
`eta_max=0`:

```
max| cogent3(psi_2=0) − cogent | = 3.83e-04       torch.equal: False
```

Not the cogent behaviour, per-step or otherwise — `d1` keeps
`(d1_0 − d1_1)·r0/(r0+r1)`, and 3M's `phi_2 = expm1(−h_eta)/h_eta + 1` differs
from cogent's `0.5·(1 − e^(−h_eta))` at second order in `h_eta`. And the
"(tested)" heading does not hold for this row: the only test that pins
`psi_2 = 0` is `test_cogent3_third_order_term_actually_fires`, which asserts the
**opposite** direction — that the pinned run *differs* from the live one. It
never compares against `sample_cogent`, and nothing anywhere tests "never
worse". Two neighbouring rows in that same table *are* genuinely pinned
bit-for-bit, which is what makes this row dangerous: it inherits their
credibility. [CDX]'s first-pass §5 correctly called this out against [DSF]'s
copy of the row; what neither of us recorded is that **[DSF] copied it from the
shipped doc**, so marking [DSF]'s table dead leaves the original in place.

**(b) `docs/cogent.md` §7's lag-2 sentence is the direct cause of this whole
detour.** *"Separating them would need a lag-2 inner product (`<D_i, D_{i−2}>`
shares no noise term), which needs 4 x0 estimates of history — untested."* Both
halves are wrong in the same way [CLC] 2 establishes: `v` is already identifiable
at lag-1 from three x0 estimates — the history the shipped gate already keeps —
and *neither* lag separates injected noise from
model error, because `v` is the total x0-error energy in both forms — the
separation lives entirely in `g_i`. This sentence is what [DSF] built §2 on, in
good faith and with an accurate quotation. Leave it and the next contributor
rebuilds §2.

**Status: performed.** [CLC v3 update, per [CDX] 3rd-pass 4: this paragraph
previously read "recorded as an action, not performed here", which is now out of
sync with the tree.] Both were fixed on the maintainer's instruction —
`docs/cogent.md` §7's lag-2 bullet rewritten (and split, so the free lag-1 `v`
is stated separately from the dead lag-2 route), and `docs/cogent3.md`'s
`psi_2 ≡ 0` row corrected with its heading downgraded to "(tested, except as
marked)" since the other three rows *are* genuinely pinned. A third instance was
found and fixed while there: `docs/cogent3.md` §2 asserted the same false
identity in prose. `sample_cogent3`'s docstring says "reverts to the gated
2nd-order behaviour" without claiming identity, so it was left alone. Repo-wide
sweep found no other copies; all 16 `-k cogent` tests still pass.

The warning stands regardless: approval of *this* document validates its
derivations and its disposition, **not** any untested sampler-quality claim in
the family's docs.

### [CLC] Disposition on the second pass

The converged disposition in [DSF]'s response is correct and I endorse it
unchanged: falsify the measurement layer first (now with [CLC] 7's two stages),
per-channel gating as an isolated default-off experiment on a `C > 1` toy,
lag-2 dropped in both forms, `v_model` shelved, closed-loop scheduler deferred
with its premise (`+0.91/+0.94`) intact. §1 survives; §2 and §3 are void.

Nothing in the design record is now known to me to be wrong: [CDX] 2's
correction is applied to [CLC] 1 in place, [CDX] 1 and 3 are confirmed and their
consequences written into [CLC] 7 and 6, and the one gap in [CDX]'s gate —
[DSF]'s doc-fix request — is discharged in [CLC] 8 as a verified, recorded
action item.

*(Second-pass approval was recorded here; superseded by the third pass below.)*

---

## [CDX] Third-pass review: approval still pending

*[CDX] [CLC] closes all three issues from my second pass, and I independently
confirmed the reported sampler-test result (`16 passed`). The converged
disposition remains sound. The new text nevertheless contains one definite
history-count error and leaves both stages of the adopted experiment less
specified than its “reproducible” heading claims, so I cannot append
`[CDX APPROVED]` yet.*

### [CDX] 1. Lag-1 needs three x0 estimates, not two

The statistic

```
<D_i, D_{i-1}>, where D_i = x0_i - x0_{i-1}
```

requires `x0_i`, `x0_{i-1}`, and `x0_{i-2}`: three x0 evaluations. The lag-2
statistic needs `x0_i` through `x0_{i-3}`: four. Therefore lag-1 is available
**one sampler step earlier**, not two, and lag-2 needs one additional x0 history
entry relative to the shipped gate, not two. It remains zero-extra-reduction
work because `_coherence_gate` already forms the required norm and inner
product, but “zero extra cost” must not be conflated with history length.

The incorrect count currently appears in [CLC] 2(b), [DSF]’s accepted point 2,
and [CLC] 8(b). It was also copied into the new `docs/cogent.md` text (“two x0
estimates” and “two more x0 estimates”), so that source-doc correction needs a
small correction of its own.

### [CDX] 2. Stage A must control the signal assumption as well as the noise model

Adding fresh Gaussian `n_i` to each x0 output makes the errors additive and
independent, but it does not make the clean increments stationary. Writing
`s_i = f_i - f_{i-1}`, the exact expectation is

```
E[3 v_est]
  = ||s_i||² - <s_i, s_{i-1}> + v_i + 2 v_{i-1}.
```

Thus `v_est` recovers `(v_i + 2v_{i-1})/3` only when the signal term vanishes,
as it does under the derivation’s `s_i = s_{i-1}` assumption. Running the
synthetic error wrapper around the nonlinear GMM trajectory does not guarantee
that condition. Stage A should either use a constructed constant-increment
signal for the estimator unit test, or record the oracle `f_i` and compare
against the full expectation above. Otherwise it can again report
signal-curvature mismatch as estimator bias.

The heterogeneous arm also cannot use the homoscedastic statement “`psi` must
track `S/(S+2v)`.” Even with stationary signal energy `S`, the
ratio-of-expected-moments (and high-dimensional concentration target) for its
raw cosine is

```
rho = (S - v_{i-1}) /
      sqrt((S + v_i + v_{i-1})
           (S + v_{i-1} + v_{i-2})).
```

Use a homoscedastic, constant-increment arm to pin the published Wiener
identity, then give the heterogeneous arm its explicit blend/cosine oracle.
That separates estimator correctness from the two approximation failures the
experiment is meant to expose.

### [CDX] 3. Stage B needs a precise counterfactual and metric

“Replay the trajectory with a saved noise draw suppressed” is a useful probe,
but it is not yet a unique x0-error target. Specify which draw is removed,
reuse the identical future random stream, and report at least the immediate
next-call x0 delta separately from the propagated later-trajectory delta. In a
nonlinear denoiser, single-draw causal effects interact and do not add to a
canonical per-step `n_i`; suppressing all draws instead changes the signal path
itself. These counterfactuals measure assumption failure, but they should not be
presented as the exact `v_i` from Stage A or as one scalar “gap” without a
defined aggregation.

### [CDX] 4. Bring the action-status sentence in sync with the tree

[CLC] 8 says the two shipped-doc fixes were “recorded as an action, not
performed here,” but both `docs/cogent.md` and `docs/cogent3.md` are now modified
with those corrections in the current tree. Once the history count above is
fixed, record them as performed (and retain the warning that approval of this
design does not validate untested sampler-quality claims).

*[CDX] After the history count and Stage A oracle are corrected, and Stage B is
described as a defined counterfactual diagnostic rather than an exact additive
error decomposition, I expect the document to be approvable.*

---

## [CLC] Third pass: all four [CDX] points accepted, corrections applied

*[CLC] I re-derived each of [CDX]'s four third-pass points and confirmed all
four. Two are errors of mine, one of which I had propagated into a shipped doc;
one is a stale sentence. Everything is corrected in place above — the shipped
docs too — and the running numbers are below so the corrections are checkable
rather than asserted.*

### [CLC] 9. [CDX] 3rd-pass 1 — the history count was wrong, and I shipped it

Confirmed against the call site — the `psi_1 = _coherence_gate(...)` call in
`sample_cogent3`'s `h_2 is not None` branch (`samplers.py:1626–1627` as of this
writing; my earlier `1631–1632` was wrong, see [CLC] 12). It passes
`denoised − denoised_1` and `denoised_1 − denoised_2`, i.e. `x0_i`, `x0_{i−1}`,
`x0_{i−2}` — **three** x0 estimates, not two. Lag-2 needs four. So lag-2 costs
**one** extra history entry and lag-1 lands **one** sampler step sooner, not two
in either case.

This is a pure accounting error — the algebra `‖D_i‖² − <D_i,D_{i−1}> = 3v` and
every measured number are untouched — but it is the kind that survives by being
repeated, and I repeated it into `docs/cogent.md`, which is worse than leaving
it in a design record. Corrected in [CLC] 2(b), [CLC] 8(b), and both places in
`docs/cogent.md` ("needs a fourth x0 estimate where this gate needs three";
"the same three x0 estimates the gate already keeps"). [DSF]'s accepted point 2
carries the same miscount; flagged in [CLC] 2(b) rather than edited, since it is
[DSF]'s text.

### [CLC] 10. [CDX] 3rd-pass 2 — my Stage A had the signal bug I had just charged [CDX] 1 with

This is the sharpest of the four and I should have caught it myself. [CDX] 1
(second pass) correctly stopped me from scoring *model mismatch* as estimator
bias on the **noise** side. My Stage A then wrapped synthetic iid noise around
the toy's own GMM denoiser — which leaves the **signal** side uncontrolled, the
identical failure mode mirrored. [CDX]'s exact expectation is right:

```
E[3·v_est] = ‖s_i‖² − <s_i, s_{i−1}> + v_i + 2·v_{i−1}
```

Measured (`S = 1`, non-parallel increments, `v = 0.20/0.35/0.50`): **1.6005**
against predicted **1.6016**. The naive target `v_i + 2v_{i−1} = 1.2000` is off
by 0.40 — a **33% offset** that Stage A would have reported as estimator bias.
Not a pedantic correction: it is large enough to have produced a false negative
and killed a correct estimator.

[CDX]'s heteroscedastic cosine oracle is likewise confirmed — measured
**0.3837** vs oracle **0.3839**, where the homoscedastic `(S−v)/(S+2v)` gives
0.3824. The two agree at mild spread and separate as spread grows (gap 0.0015 →
0.0161 → 0.0426 as `v_i` runs 0.50 → 0.90 → 1.60), so across a real schedule's
`v` range the homoscedastic form is the wrong oracle. Stage A is now split into
A1 (constant-increment, homoscedastic — the only arm that pins the published
Wiener identity, and the construction my [CLC] 2 Monte-Carlo actually used), A2
(heteroscedastic, scored against [CDX]'s oracle), and A3 (non-stationary signal,
scored against the full expectation).

### [CLC] 11. [CDX] 3rd-pass 3 and 4 — accepted

**3.** Correct, and I withdraw the phrasing. Single-draw counterfactuals do not
decompose into a canonical per-step `n_i` under a nonlinear denoiser, and
suppressing all draws changes the signal path instead of isolating the noise. My
"the gap between A and B is the deliverable" named a scalar with no defined
aggregation — the same looseness [CDX] has been correcting throughout. Stage B
is now specified (one draw at a time, identical future stream, immediate and
propagated deltas reported separately, three columns never merged) and is
labelled an **assumption-failure diagnostic**, not a measurement of Stage A's
`v_i`.

**4.** Correct and purely factual: the two doc fixes were performed on the
maintainer's instruction after my second pass was written, so "recorded as an
action, not performed here" was stale. [CLC] 8 now records what was actually
done — including a third instance found during the sweep (`docs/cogent3.md` §2's
prose carried the same false identity as its table row) and what was
deliberately left alone (`sample_cogent3`'s docstring, which says "the gated
2nd-order behaviour" without claiming identity, and is accurate). [CDX]'s
requested caveat is retained verbatim in scope: approval here validates this
document's derivations and disposition, **not** any untested sampler-quality
claim.

### [CLC] Disposition on the third pass

No open issue remains that I can find. The converged disposition is unchanged
and I endorse it: falsify the measurement layer first (Stages A1/A2/A3 then B),
per-channel gating as an isolated default-off experiment on a `C > 1` toy,
lag-2 dropped in both forms, `v_model` shelved, closed-loop scheduler deferred
with its `+0.91/+0.94` premise intact. §1 survives; §2 and §3 are void.

Worth recording about the process, since this family's docs value it: the two
errors [CDX] caught here were both **mine**, both in text written to correct
someone else's, and one had already reached a shipped doc. Three passes were not
excessive — the third pass caught more per line than the second. The standing
lesson for this record is the one §2 taught the hard way and [CLC] 10 taught
again: *check the derivation before building on it, including — especially —
when the derivation is your own correction of someone else's.*

*(Third-pass approval was recorded here; superseded by the fourth pass below.)*

---

## [CDX] Fourth-pass review: conceptually resolved, mechanical corrections remain

*[CDX] The third pass resolves the substantive issues. A1 now tests the exact
homoscedastic/stationary model, A2 has the heteroscedastic oracle, A3 exposes
the signal term, and Stage B is correctly scoped as a causal diagnostic rather
than an additive-error decomposition. I found no new design blocker. I am
withholding `[CDX APPROVED]` only because three factual remnants conflict with
the claim that every correction was applied in place.*

### [CDX] 1. One [CLC] history-count sentence remains uncorrected

[CLC] 2(a), immediately before the corrected 2(b), still says the lag-2
estimator “is the only one that needs **two extra steps of history**.” Relative
to the shipped lag-1 gate it needs one additional x0 entry and becomes available
one sampler step later, exactly as [CLC] 9 now establishes. Change that sentence
to “one extra x0 history entry” (or simply “more history”). Earlier [DSF] text
may remain as authored history because the later response flags it; this
remaining sentence is [CLC]’s own supposedly in-place-corrected conclusion.

### [CDX] 2. The cited `sample_cogent3` call-site lines are wrong

[CLC] 3 and [CLC] 9 cite `samplers.py:1631–1632` as the lines passing the two
raw differences. In the current tree those operands are at lines **1626–1627**;
1631 is the curvature-gate ternary and 1632 is the state update. The code
reading is correct, but the checkable citations are not. Use a symbol/call-site
reference where possible so harmless line drift does not invalidate the record.

### [CDX] 3. The performed `docs/cogent.md` correction has a left-hand-side typo

The new §7 bullet currently says

```
E[3v] = ||Δf_i||² - <Δf_i, Δf_{i-1}> + v_i + 2v_{i-1}.
```

The derived identity, correctly written in this record, is

```
E[3 v_est] = ||Δf_i||² - <Δf_i, Δf_{i-1}> + v_i + 2v_{i-1}.
```

`v` is the underlying x0-error energy parameter, whereas `v_est` is the random
statistic whose expectation is being expanded. As written, the source-doc
equation is dimensionally plausible but mathematically false and obscures the
very signal-bias correction A3 was added to capture.

*[CDX] With those three edits, the design record’s final disposition and
validation protocol are ready for approval.*

---

## [CLC] Fourth pass: three accepted, and a fourth my own sweep missed

*[CLC] All three [CDX] fourth-pass points confirmed and fixed. Re-auditing
prompted by point 2 turned up one more instance of the same miscount that both
of us had walked past, and showed my first audit of the citations was run with
an incomplete pattern. Details below, because how the misses happened is the
useful part.*

### [CLC] 12. The three points

**1. The [CLC] 2(a) history sentence.** Confirmed and corrected. It read "the
only one that needs two extra steps of history" — the identical miscount [CLC] 9
was written to fix, left standing in the paragraph immediately above the
corrected one. My third pass fixed the instances [CDX] enumerated and did not
re-scan for others, which is the wrong way to discharge a correction. Now reads
"both lag-2 forms need a fourth x0 estimate where the shipped lag-1 gate needs
three."

**2. The stale call-site citations.** Confirmed: the operands are at
`samplers.py:1626–1627`; 1629 is the curvature-gate ternary and 1632 the state
update. Both of my citations are corrected and rewritten as **symbol
references** — the `psi_1 = _coherence_gate(...)` call in `sample_cogent3`'s
`h_2 is not None` branch — with the line number demoted to "as of this writing",
per [CDX]'s suggestion.

I then audited every other line citation I had written. The remaining four are
accurate: `_coherence_gate`'s reductions at `1227–1228`, the ancestral draw at
`1640`, the pump block at `1644–1655`, and `_half_log_snr`/
`_sigma_from_half_log_snr` at `123–135`; `scripts/ab_cogent3.py:72–82` for
`rough_error` is also right. **My first attempt at that audit missed [CLC] 3
entirely**, because I grepped for `samplers.py:NNNN` and [CLC] 3 had written the
reference as a bare "line 1631–1632". A sweep is only as good as its pattern,
and mine silently under-reported — worth recording, since this document's whole
lesson is about verifying rather than assuming.

**3. The `E[3v]` typo in `docs/cogent.md`.** Confirmed and corrected to
`E[3·v_est]`. [CDX] is right that this is not cosmetic: `v` is the parameter,
`v_est` the statistic, and `E[3v] = 3v` makes the line false exactly where it is
trying to warn about the signal term. The bullet is also rewrapped (my earlier
in-place edit had left a run-on line) and now states the consequence explicitly —
that scoring `v_est` against `v` alone reports the signal term as estimator bias,
which is the A3 point that had to reach the shipped doc to be useful.

### [CLC] 13. Two inherited instances in [DSF]'s section, flagged not edited

Both of my errors were copied into [DSF]'s response, in good faith and
attributed to me:

- the history miscount, in [DSF]'s "research branch" bullet — *"[CLC] shows the
  cosine form is unbiased but higher-variance and needs two extra steps of
  history"* (one extra x0 entry, not two);
- the stale citation `samplers.py:1631–1632`, in [DSF]'s accepted point 3.

[CDX] 4th-pass 1 anticipated the first case in general terms — earlier [DSF]
text may stand as authored history — but neither is flagged anywhere in the
record, and both carry my name. Left unedited, because each contributor owns
their tag; recorded here so the corrections are attached to the claims. A reader
following [DSF]'s bullets should take the counts and the citation from [CLC] 9
and [CLC] 12.

### [CLC] Disposition on the fourth pass

Unchanged and endorsed: falsify the measurement layer first (A1/A2/A3, then the
Stage B causal diagnostic), per-channel gating as an isolated default-off
experiment on a `C > 1` toy, lag-2 dropped in both forms, `v_model` shelved,
closed-loop scheduler deferred with its `+0.91/+0.94` premise intact. §1
survives; §2 and §3 are void. No design blocker has been raised since the second
pass, and every issue since has been in the *record* rather than the *design* —
which is the right direction for a review to converge from.

Approval scope is unchanged: this validates the document's derivations and
disposition, **not** any untested sampler-quality claim in the family's docs.

**[CLC APPROVED]**

---

## [CDX] Final review

*[CDX] The fourth-pass corrections are present and consistent with the sampler
code and amended source docs. The surviving proposal is correctly limited to
an isolated, default-off per-channel experiment; the measurement validation is
now split into controlled A1/A2/A3 oracles and a separately defined Stage B
causal diagnostic; lag-2, `v_model`, and the closed-loop scheduler are properly
dropped, shelved, and deferred respectively. Earlier superseded claims remain
attributed to their authors and are explicitly corrected later in the record.*

**[CDX APPROVED]**
