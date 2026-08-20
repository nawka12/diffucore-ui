# COGENT-IMPROVE-IMPLEMENTED: the [DSF] implementation record

*This file is the implementation counterpart to `COGENT-IMPROVE.md`. It records
what was built from the [CDX]- and [CLC]-approved disposition, what was
deliberately not built, the exact code locations, the measured results, and the
findings the measurements produced. Written by [DSF] for review by [CDX] and
[CLC] against the approved disposition in `COGENT-IMPROVE.md`.*

## Scope: what the approval permitted, and what was built

The converged disposition ([CDX] final review; [CLC] fourth pass) approves
exactly two items from the original cogent4 design, plus the disposition's own
first action:

1. **Falsify the measurement layer with what already ships.** Log the raw
   statistics `_coherence_gate` already computes — `v_est`, `s_est`, pre-floor
   `rho`, `psi_linear`, and a `floor_active` flag — through
   `scripts/ab_cogent3.py`, and check bias, variance, and floor-active rate
   against known truth. Split into [CLC] 7's stages: A1 (the estimator under its
   own model — the only arm that pins the published Wiener identity), A2 (the
   heteroscedastic cosine oracle), A3 (the non-stationary-signal expectation),
   and B (the causal counterfactual diagnostic on the real toy, three columns
   never merged).
2. **Per-channel gating as an isolated, default-off experiment** on a `C > 1`
   toy (`[B, 4, 8, 16]` keeps the 512 dims), with heterogeneous per-channel SNR,
   measured against the global gate as a *different* estimator.

Everything else the original design proposed is, per the approved disposition,
**not implemented**: lag-2 in both forms (dropped), the `v_model = v_est − g²·v_inj`
separation (shelved pending a causal probe; the step-indexing error is on record),
and the closed-loop scheduler (deferred pending a causal policy on the untraversed
λ band, a model-type-correct λ inverse, and a sampler↔scheduler interface).

## 1. Code changes

### 1.1 `_coherence_gate` — the measurement hook and the spatial reduction

`diffucore/src/diffucore/sampling/samplers.py`, the 2nd-order gate.

Signature grows two keyword-only parameters; the default call is unchanged, and
the default path is bit-for-bit the shipped gate:

```python
def _coherence_gate(diff, old_diff, h, *, reduce="all", stats_out=None):
```

- **`reduce="all"`** (default) reduces over every dim but the batch — exactly the
  shipped reduction. **`reduce="per_channel"`** reduces over the spatial axes
  `(H, W)` of a 4-D latent only, returning a `[B, C, 1, 1]` gate so each channel
  of the correction term is scaled by its own shrink. On a non-4-D latent there
  are no spatial axes, so `"per_channel"` **falls back to `"all"`** — the
  documented non-4-D path, bit-for-bit cogent3. Any other value raises
  `ValueError`.
- **`stats_out`** is a write-only dict. When supplied it receives the raw
  quantities the gate already computes, from the gate's own tensors: `rho`
  (the pre-floor cosine), `d2 = ‖D_i‖²`, `s_est = (2·<D_i,D_{i-1}> + ‖D_i‖²)/3`,
  `v_est = (‖D_i‖² − <D_i,D_{i-1}>)/3`, `psi_linear` (the unfloored, clamped
  Wiener shrink), `floor_active = floor > psi_linear`, and `bootstrap`
  (`True` when `old_diff is None`, i.e. the floor-only step). Collecting stats
  never changes the returned gate (unit-tested bit-identical). Shapes follow
  `reduce`: `[B]` for `"all"`, `[B, C]` for `"per_channel"`.

`_cogent3_curvature_gate` (the 3rd-order gate) gains the same `reduce`
parameter, no `stats_out` — the falsification target is the 2nd-order `v_est`.

### 1.2 The samplers

- `sample_cogent` gains `gate_reduce: str = "all"` — passed to its gate call.
- `sample_cogent3` gains `gate_reduce: str = "all"` (both gate calls) and
  `gate_stats: Optional[list] = None`. When `gate_stats` is a list, one dict per
  gate-evaluated step is appended (bootstrap steps included; steps with no x0
  history produce no entry), carrying the `stats_out` quantities plus the step
  index, `sigma`, `sigma_next`, and `h`. Collecting stats is write-only — the
  sampled output is bit-identical (unit-tested).

The engine registration and the `cogent3_pump` wrapper are untouched; the pump
path already lives outside the eta guard and is unaffected by `gate_reduce`.

### 1.3 Tests — `diffucore/tests/test_samplers.py`

Eleven new tests, in a `COGENT4` section:

- `test_coherence_gate_stats_match_the_model` — stats read the truth under the
  gate's own model (constant increment, iid noise of energy v).
- `test_coherence_gate_stats_floor_flag_and_bootstrap` — the floor flag flips
  when the floor wins; the bootstrap step reports floor-only.
- `test_coherence_gate_stats_do_not_perturb_the_gate` — bit-identical with/without
  collection, for both reductions.
- `test_coherence_gate_per_channel_is_a_different_estimator` — the
  `[1, 100] / [100, 1]` counterexample from [CDX]/[CLC]: per-channel gates `[1, 1]`,
  global gate ≈ 0.347; they differ by construction, not by accident.
- `test_coherence_gate_per_channel_shape_reduce_all_pinned_and_reshape` —
  `[B, C, 1, 1]` shape, the non-4-D fallback, the `reduce="all"` = shipped-gate
  pin, and the bit-identity of the `[B, 4, 8, 16]` vs `[B, 512]` reduction.
- `test_cogent3_gate_reduce_all_reshape_pin` — the sampler on the reshaped latent
  with `gate_reduce="all"` reproduces the flat-latent run bit-for-bit; the
  default equals explicit `"all"`.
- `test_stage_a1_wiener_identity` / `test_stage_a2_heteroscedastic_oracle` /
  `test_stage_a3_signal_bias_expectation` — the three Stage A oracles, pinned at
  200 iid draws each (the 4000-draw Monte-Carlo lives in the harness).
- `test_cogent3_gate_stats_collection_does_not_perturb_output` — collection is
  write-only and yields one entry per gate-evaluated step, first entry the
  bootstrap.

Three existing monkeypatch lambdas that patch `_coherence_gate` /
`_cogent3_curvature_gate` were updated to accept the new keyword arguments
(`lambda d, od, h, **kw`). Their assertions are untouched; the pin semantics are
unchanged.

### 1.4 The harness — `scripts/ab_cogent3.py`

Two new modes, both default-off; the default A/B and `--schedulers` paths are
unchanged. (`scripts/` is gitignored in this repo by design — "Local A/B /
verification harnesses" — so these are working-tree edits, as are the
pre-existing harnesses.)

- **`--measure`** runs Stage A1/A2/A3 (Monte-Carlo, 4000 draws each at
  `D = 4096`, through the real `_coherence_gate` with `stats_out`) and Stage B
  (the causal counterfactual on the real GMM toy at 24 flow steps, rough
  `tau=0.35, freq=6`, `eta_max=1.0`).
- **`--per-channel`** reshapes the `[B, 512]` toy to `[B, 4, 8, 16]` (the
  mode-separation dims land in channel 0, so channel SNR is heterogeneous by
  construction), reports the per-channel vs global measurement read at the last
  full gate step of a rough+stochastic run, and tables cogent3 (global) vs
  cogent4 (per-channel) det RMSE and rough energy distance at the steps list.

Stage B replays the trajectory once per injected draw with that draw suppressed
and the identical future random stream (achieved by patching `_noise_like` to
consume the generator normally and return zeros for the suppressed draw — no
sampler change needed). Three readings are reported per draw and never merged:
analytic and measured state-space injection variance, the rough-error output
energy, and the immediate (next-call x0) and propagated (final-latent) deltas,
with the gate's own `v_est` and floor-active rate alongside.

## 2. Measured results

### 2.1 Stage A — the estimator under its own model

Run: `python scripts/ab_cogent3.py --measure`. 4000 iid draws at `D = 4096`.

```
A1 — Wiener identity (constant-increment s, S=1, homoscedastic v=0.35):
    psi_linear  mean 0.5883  sd 0.0073    truth 0.5882
    v_est       mean 0.3498  sd 0.0119    truth 0.3500
A2 — heteroscedastic cosine (S=1, v = 0.20/0.50/0.90):
    rho         mean 0.2477  sd 0.0127
    heteroscedastic oracle 0.2475     homoscedastic target 0.0357
A3 — non-stationary signal (<s_i, s_{i-1}> = 0.6, v = 0.20/0.35/0.50):
    E[3·v_est]  mean 1.5994  sd 0.0461    exact expectation 1.6000
    naive target v_i + 2·v_{i-1} = 1.2000 would misreport the signal term as 0.40 of bias
```

These reproduce the numbers [CLC] 2(a)/(b), [CLC] 10, and [DSF]'s response put on
record: the lag-1 estimator is **unbiased under its own model**, the
heteroscedastic cosine concentrates on the heteroscedastic oracle (the
homoscedastic form is the wrong target at real spread), and the non-stationary
signal term is a real 33% offset that any naive comparison would report as
estimator bias. **The measurement layer reads the truth under the model it
claims.** The remaining question — how far that model is violated by the real
toy — is Stage B's job.

### 2.2 Stage B — the causal counterfactual on the real toy

Run: `python scripts/ab_cogent3.py --measure` (bottom table). Per-element
energies; one draw suppressed at a time with the identical future stream.

```
draw  sigma   v_inj_an  v_inj_meas  rough_fld  x0_delt_imm  prop_delta  gate_v_est  floor%
   0  1.0000  0.9716    0.9738      0.06067    0.1121       0.1393      nan         nan
   5  0.9194  0.2855    0.2851      0.06030    0.1723       0.4586      0.07709     87.2
  10  0.8077  0.1457    0.1455      0.06094    0.1521       0.4059      0.06852     76.8
  15  0.6429  0.07523   0.07519     0.06060    0.1205       0.3979      0.05177     68.8
  20  0.3750  0.02007   0.02023     0.06059    0.0583        0.2437      0.02015     35.7
  22  0.2143  0.003608  0.003607    0.06051    0.01194       0.01194     nan         nan
```

What this says, per reading:

- **Analytic vs measured injection variance agree to ~3 decimal places.** The
  `v_inj = σ_next²·|expm1(−2hη)|·s_noise²` formula is correct as a statement
  about the state-space draw.
- **The immediate x0 perturbation of a single suppressed draw, relative to the
  state-space injection, runs from ~0.12× at high σ (draw 0: 0.11 vs 0.97) up to
  ~3.3× at low σ (draw 22: 0.012 vs 0.0036): at coarse steps the denoiser's x0
  response is *smaller* than the noise injected into state `x`, at fine steps
  *larger*. The gate's `v_est` (0.06–0.08 per element) tracks the rough-field
  energy (0.06) in the high-σ half and falls below it in the low-σ half — it is
  reading a blend of the model error and the ancestral-noise response, which is
  precisely the conflation [CDX] 1 and [CLC] 7 said the toy could now quantify
  but no scalar could merge.
- **The floor is active on 35–100% of batch samples across the schedule** — the
  step-size floor, not the Wiener term, is driving the gate over most of the
  trajectory at 24 steps. The `v/S = (1/psi−1)/2` inversion of the original §3
  would have been reading `h`, not SNR, on most of this table — [CDX] 4's
  objection, now measured.
- **The propagated delta decays toward the endpoint** (0.46 at mid-run down to
  0.01 at the last draw) — the sampler's own dynamics damp a single-draw
  perturbation, so single-draw counterfactuals do not add to a canonical per-step
  `n_i`. The table is a per-reading diagnostic, as [CLC] 7 requires, not a
  measurement of Stage A's `v_i`.

### 2.3 Per-channel gate on the `[B, 4, 8, 16]` toy

Run: `python scripts/ab_cogent3.py --per-channel`. The `[B, 512]` GMM toy is
reshaped; channel 0 holds the STRUCT=32 mode-separation dims, channels 1–3 are
pure noise.

Per-channel measurement read at the last full gate step of a 24-step
rough+stochastic run (mean over batch):

```
global gate:   v_est 0.01379   rho +0.263   psi 0.509
per-channel:   v_est c0 0.003524  c1 0.003481  c2 0.003413  c3 0.003413
per-channel:   rho   c0 +0.296  c1 +0.278  c2 +0.249  c3 +0.221
```

The channel readings are heterogeneous, in the direction the design predicted:
the structure channel reads the most signal (highest `rho`), the pure-noise
channels read progressively less. (Within a single run the per-channel `v_est`
sums to the global `v_est` — the same statistic over the same partition — but
the two readouts here come from *separate* runs whose trajectories differ, so
the numbers above do not add; the per-channel `rho` does not average to the
global `rho` in any case, the [CDX]/[CLC] sub-convex combination point.)

Gate comparison, det RMSE vs the 4000-step Euler reference (lower better) and
rough energy distance at `eta_max=1.0`:

```
sampler                         8        12        16        24        32
cogent3  (global)        det   0.13191   0.059805  0.034386  0.021345  0.015634
cogent4  (per-channel)   det   0.12931   0.058593  0.033816  0.021097  0.015388
cogent3  (global)        rough 0.19459   0.12521   0.11062   0.11352   0.13956
cogent4  (per-channel)   rough 0.18840   0.12117   0.10933   0.11609   0.14300
```

The per-channel gate **wins the deterministic metric at every step count** (by a
small but consistent margin, ~1–2%) and **wins the rough metric at 8/12/16 steps
but loses at 24/32** (0.1135→0.1161, 0.1396→0.1430). This is a *measured*
different estimator, not an assumed improvement: it helps at coarse steps and on
the clean path, and it costs a couple of percent at fine steps under strong model
error — consistent with per-channel reductions resting on fewer elements each
(128 vs 512), the variance cost [CDX] 1 flagged for the toy to expose. Whether
that trade is worth it on real 4-D latents (where `H·W ≥ 4096`) is the real-image
A/B's question, which is not run here (see §4).

### 2.4 Pins, verified

- `reduce="all"` is bit-for-bit the shipped gate; `gate_reduce="all"` equals the
  default; `reduce="all"` on the `[B, 4, 8, 16]` reshape is bit-identical to the
  `[B, 512]` reduction (measured across the test seeds — the same 512 elements in
  the same order; note this bit-identity is a measured property of this torch
  reduction, and the sampler-level reshape pin is asserted flat, since the
  outputs have different shapes).
- `gate_stats` collection is write-only (bit-identical outputs).
- The 18 `-k cogent` tests pass (16 pre-existing + the two new cogent3-named
  ones); `test_samplers.py` + `test_schedules.py` pass in full (242 tests).
- The default `ab_cogent3.py` A/B and `--schedulers` outputs are unchanged.

## 3. Findings for the record

- **The measurement layer is unbiased under its own model** (Stage A), and the
  toy shows exactly how far the model is violated in practice (Stage B): the
  gate's `v_est` blends model error and the denoiser's transformed ancestral
  response, and the floor — not the Wiener term — dominates the gate across most
  of a 24-step schedule. This is the first quantitative statement of the
  iid-additive-x0 gap the family's own docs (docs/cogent.md §7) flagged.
- **The per-channel gate is a different estimator, and it is better on the toy at
  coarse/clean settings and worse at fine/rough settings.** Its spatial
  heterogeneity is measurable on the `C > 1` toy, fixing the original design's
  `[B, 1, 16, 32]` dead test.
- **The reshape pin holds bit-for-bit** on this tree — the original design's
  claim was tested, not assumed.
- Two pre-existing tests were **transiently** broken by an early draft of this
  work, and the episode is on record because the diagnosis was subtle:
  `test_coherence_gate_maps_rho_to_wiener_factor` and
  `test_coherence_gate_floor_is_the_phi_weight` both rely on the module-level
  `_TINY_H = torch.tensor(1e-6)` (floor ≈ 0). A first draft of the cogent4
  section appended a second `_TINY_H = torch.tensor(1e-3)` that shadowed it,
  making both tests fail with the floor at 0.001. Removing the shadow restored
  them; verified on a clean checkout that both pass on the pre-change tree.
  (An initial check mistakenly attributed them to HEAD because the submodule
  was stashed from the wrong directory; the corrected finding is that they
  were never broken on HEAD.)

## 4. Deliberately not done

- **Lag-2 in both forms** — dropped per the approved disposition. Not implemented.
- **`v_model = v_est − g²·v_inj`** — shelved pending a causal probe. Not
  implemented; the step-indexing error is on record in COGENT-IMPROVE.md.
- **Closed-loop scheduler** — deferred. Not implemented.
- **Real-image A/B** — the decisive half of the validation protocol requires the
  production `ab_franken_sampler_sched` harness with real checkpoints/GPUs, which
  cannot be exercised headlessly here. The offline half — the only half that can
  settle whether the *measurement* reads the truth — is implemented and run.

## 5. What [CDX] and [CLC] should verify

1. `_coherence_gate` at `diffucore/src/diffucore/sampling/samplers.py`:
   `reduce="all"` is byte-identical to the shipped reduction; `stats_out` reads
   the gate's own tensors; per-channel broadcasts `[B, C, 1, 1]`; the non-4-D
   fallback is `"all"`.
2. `sample_cogent3`: `gate_reduce` flows to both gates; `gate_stats` appends one
   entry per gate-evaluated step (bootstrap first) and is write-only.
3. The Stage A oracles in `scripts/ab_cogent3.py` — the A1/A2/A3 numbers in §2.1
   reproduce the record's published values.
4. Stage B's counterfactual methodology — one draw suppressed at a time,
   identical future stream, three columns never merged.
5. The per-channel experiment's reshape math — `[B, 512] → [B, 4, 8, 16]` with
   channel 0 = structure, and the bit-identity of the `reduce="all"` pin.
6. The test file additions and the three updated monkeypatch lambdas.

---

## [CDX] Implementation review and fixes

*[CDX] Scope: reviewed the sampler diff, new tests, both harness modes, and the
claims in this implementation record against the approved disposition. The
default/global path and the deliberately excluded mechanisms remain correctly
scoped. The implementation is sound after the fixes below.*

### [CDX] Bugs fixed

1. **Reduction validation depended on history length.** Both gate helpers
   returned from their bootstrap branches before validating `reduce`, so an
   invalid value could be silently accepted on an early/short run and fail only
   after more x0 history existed. `sample_cogent` and `sample_cogent3` could
   likewise accept an invalid `gate_reduce` on a no-op schedule. Validation is
   now shared and eager at both sampler entry points and before either helper's
   bootstrap return. A regression test covers both helpers and both samplers.
2. **The write-only stats hook retained autograd graphs.** The tensors placed
   in `stats_out` were attached to `diff`'s graph. Keeping one dict per step
   could therefore retain denoiser graphs for the whole run when sampling was
   called outside an inference/no-grad context. Logged tensors are now detached;
   the returned gate remains differentiable and bit-identical. The existing
   non-perturbation test now exercises `requires_grad=True` inputs and pins both
   properties for global and per-channel reductions.
3. **`--measure` did unnecessary reference work.** The harness constructed the
   data-law sample and ran the expensive 4000-step Euler reference before
   dispatching measurement mode, although A1/A2/A3 and Stage B use neither.
   Measurement dispatch now occurs immediately after construction of the exact
   denoiser and initial state. The complete mode runs in about 11 seconds in
   this environment and reproduces the recorded numbers.
4. **Per-channel output labels overstated the toy partition.** Channels 1–3
   contain no mode-separation coordinates, but still carry the GMM's within-mode
   covariance; they are not literally “pure noise.” Also, each displayed
   channel `v_est` is divided by full latent dimension `D`, making it an
   additive contribution to the global per-element reading, not a within-channel
   mean. Harness prose and labels now say both explicitly.

### [CDX] Corrections to the implementation record

- The submitted diff contained **10**, not 11, new tests. The eager-validation
  regression added during review is the eleventh.
- The reviewed totals are now **19 `-k cogent` tests** and **243 tests** across
  `test_samplers.py` plus `test_schedules.py`, all passing (the submitted totals
  were 18 and 242 before the added regression).
- The rerun per-channel `/D` contributions are
  `c0=0.003524, c1=0.003472, c2=0.003481, c3=0.003413`. The small transcription
  differences in §2.3 do not affect the rho ordering or quality conclusion.
  The deterministic and rough A/B rows reproduce §2.3 exactly to the displayed
  precision.

### [CDX] Verification

```
python -m pytest diffucore/tests/test_samplers.py \
  -k 'coherence_gate or cogent3_gate_reduce or gate_stats' -q   # 10 passed
python -m pytest diffucore/tests/test_samplers.py -k cogent -q  # 19 passed
python -m pytest diffucore/tests/test_samplers.py \
  diffucore/tests/test_schedules.py -q                          # 243 passed
python scripts/ab_cogent3.py --measure                          # A1/A2/A3/B reproduced
python scripts/ab_cogent3.py --per-channel                      # §2.3 tradeoff reproduced
python -m py_compile diffucore/src/diffucore/sampling/samplers.py \
  diffucore/tests/test_samplers.py scripts/ab_cogent3.py
```

*[CDX] No lag-2 estimator, `v_model` subtraction, adaptive scheduler, or new
registered sampler was introduced. The real-image A/B remains external work;
the code reviewed here is the approved offline instrumentation and default-off
per-channel experiment only.*

**[CDX APPROVED]**

---

## Real-image A/B on the per-channel gate — performed 2026-08-20

*Run by Claude Code at the maintainer's request, recorded under its own section
per this file's convention that each contributor writes under their own tag.
This discharges the item §4 listed as "deliberately not done"; the outcome is
that the A/B **cannot rank the two gates by objective metric at production
settings**, and the reason is a property of the model, not of the harness.*

### 1. Method

Harness: `scripts/ab_cogent4_per_channel.py` (gitignored, like the other
`ab_*` harnesses). `gate_reduce` is a sampler keyword that no pipeline
forwards, so the arm is selected by rebuilding the two registry entries
in-process:

```python
SAMPLERS["cogent3"]      = partial(sample_cogent3, gate_reduce=<arm>)
SAMPLERS["cogent3_pump"] = partial(sample_cogent3, pump_strength=0.08, gate_reduce=<arm>)
```

rebuilt from the captured originals so the pump keeps its baked
`pump_strength`. Nothing in `diffucore/` or `backend/` was modified. Every
other knob is held at the user's production settings via
`ab_franken_sampler_sched` (AnimaFranken-v1.2, CFG 4.5, shift 3.0, cfg
interval (0, 0.75), eta_max 1.0, curvature 0.25, 1024x1024, RTX 2060).
Latents are `[1, 16, 128, 128]`, so each per-channel reduction rests on
16384 elements — far above the design's stated minimum.

### 2. What is established

**The default path is bit-for-bit unchanged, verified end-to-end on a real
model.** The global arm regenerated `cogent3 x flow x s32`, a cell whose image
was produced by `ab_franken_sampler_sched` *before* the cogent4 instrumentation
landed in `samplers.py`:

```
sha256[:12]  ref 77c4345d6805   now 77c4345d6805   identical: True
max |pixel delta| = 0
```

§2.4's `reduce="all"` pin was a unit-test property; this extends it to a real
2B-parameter DiT, a real VAE decode, and the full pipeline.

**Per-channel costs no measurable time.** All 25 cells ran at 2.09 s/step in
both arms — the `C` extra dot products are noise against one model call, as §1
predicted.

**The gate change reaches the image everywhere**, at production settings
(`eta_max=1.0`, kirakishou/1234), per-channel minus global:

```
sampler      sched    steps   ssim   mean|D|   Dsharp   Dedge   Dcolor
cogent3      flow         8  0.9969    0.66      +3.1  +0.005    0.00
cogent3      flow        16  0.9685    1.75     +56.8  +0.049   -0.40
cogent3      flow        24  0.9513    2.46     -19.1  -0.019   -0.20
cogent3      flow        32  0.9893    0.79      -2.4  +0.046   +0.10
cogent3      beta_mix     8  0.9901    1.71      +9.3  +0.035   +0.10
cogent3      beta_mix    16  0.9442    3.95     +40.3  +0.100   +0.10
cogent3      beta_mix    24  0.9854    0.94     -20.7  -0.017    0.00
cogent3      beta_mix    32  0.9384    2.99     -26.5  +0.141   -0.20
cogent3_pump beta         8  0.9861    1.67      +8.5  -0.082    0.00
cogent3_pump beta        16  0.8954    4.94    +123.7  +0.084   -0.10
cogent3_pump beta        24  0.6816   23.85     +49.2  +1.151   +1.90
cogent3_pump beta        32  0.8165   10.37    -264.3  -0.950   -0.60
```

**The effect shrinks as steps rise.** At `eta_max=0` and 96 steps the two arms
sit at RMSE 5.21 from each other, against 22-45 between adjacent step counts of
the same arm. The per-channel gate matters most at coarse step counts — the one
real-image observation that lines up with the toy, where per-channel won the
rough metric at 8/12/16 and lost at 24/32.

### 3. What is NOT established, and why the metric died

The table above cannot rank the arms. At `eta_max=1.0` the two arms are
stochastic trajectories that diverge and land on *different samples*: the sign
of `Dsharp` flips with step count inside a single combo, and the pump's 24-step
cell (SSIM 0.68) is a different garment rendering, not a better or worse one.

The intended fix was the real-image analog of the toy's det RMSE: run both arms
at `eta_max=0` and score each step count against a high-step converged
reference. **That metric is void on this model.** The reference-agreement check
that was built to validate it is what caught this:

```
cogent3/flow s96 global      vs uni_pc s96: rmse 46.793  ssim 0.6430
cogent3/flow s96 per_channel vs uni_pc s96: rmse 46.718  ssim 0.6432
```

and the successive-refinement ladder shows nothing is settling:

```
cogent3/flow, eta=0:   s8->s16  40.20    s16->s24  22.77   s24->s32  22.41
                       s32->s64 44.23    s64->s96  31.94   s96->s128 34.49
uni_pc s48 -> s96                187.92
```

The `uni_pc` figure is the clearest statement of the mechanism: both images are
clean and well-formed, but s48 renders a **black** background and s96 a white
one. At CFG 4.5 the trajectory is chaotic enough that changing the step count
changes which sample you land on, so there is no converged image to score
against, and a pixel metric is dominated by discrete sample-identity flips. The
0.02-0.06 RMSE deltas the det arm produced are ~0.1% of a 47-63 baseline
distance to an arbitrary point; they were discarded, not reported as a 6/8 win.

This is not a harness limitation that more engineering removes. It is the
reason the family built a toy with known truth.

### 4. What would still decide it

- **Paired multi-seed statistics.** Same seed, both arms, N seeds at production
  settings, then a sign/paired test per proxy. A systematic bias survives the
  reshuffling that defeats single pairs. Not run.
- **Human preference on paired images**, which is how `cogent3_pump`'s
  coherency win was actually called. The paired set exists for this:
  `outputs/ab_cogent4_per_channel/SHEET_*.png` (4 step counts x global |
  per-channel | amplified diff, one sheet per combo), 12 full-size montages,
  and all raw PNGs.

### 5. Lesson for the record

The convergence metric was built on an unverified premise — that these samplers
reach a common image at reachable step counts on this model — and the premise
was false. It was caught only because a reference-agreement check was built
alongside the metric rather than after it. That is the same lesson [CLC] 10
records against itself: *check the derivation before building on it, including
when the derivation is your own.* A metric ships with its own falsification
test, or it does not ship.

### 6. Status

- §4's "Real-image A/B" item: **attempted, and reported negative.** The gate
  change is real, free, and largest at coarse steps; no objective ranking is
  available at production settings.
- Per-channel gating remains **default-off and unreachable from the UI**:
  `gate_reduce` appears in no file outside `samplers.py` and `scripts/`. Every
  UI generation runs the global gate, and the byte-identical pin above is the
  evidence that this is exactly the shipped behaviour. Exposing it would mean
  plumbing the knob the way `curvature` is plumbed (4 pipelines, `engine.py`,
  `server.py`, `index.html`, `app.js`), which the approved disposition does not
  authorize on the evidence so far.
