# EasyCache decision rule + cheaper uncond pass — implementation plan

> Status: **planned 2026-09-02, not started.** Written for an executor agent.
> Origin: the Anima speed research of the same date (items 2 and 3 of the
> ledger; report at https://claude.ai/code/artifact/44858fdc-097d-45b5-833b-d2f07f132b02,
> memory `anima-speed-research-2026-09`). Everything below is grounded in the
> code as of ui `9647437` / diffucore `0042fcf`; line numbers are from that
> state and are a guide, not a contract — re-grep before editing.

## Goals

1. **Part A — a second skip *decision* rule for the existing Anima TeaCache**,
   EasyCache's runtime-adaptive criterion (Zhou et al., 2025,
   arXiv:2507.02860). No calibration, no fitted polynomial, and the threshold
   is meant to transfer across samplers and step counts — the property the
   current rule lacks (see "Why" below). Selected per generation, default stays
   the current rule, existing images reproduce unchanged.
2. **Part B — make the uncond (negative-prompt) pass cheaper** on top of the
   CFG guidance interval: (B1) measure raising the interval *start* above 0,
   which the paper says is the quality-improving direction and which has never
   been A/B'd here; (B2) a separate, looser skip threshold for the uncond cache
   stream; (B3, conditional) let the uncond stream follow the cond stream's
   skip decisions.

## Non-goals

- No change to the skip *output* (HiCache/TaylorSeer forecast stays as is).
- No change to the drift rule, its calibration, or the calibrated/raw toggle.
- No CUDA-graphs work (TeaCache + cuda_graphs stays hard-rejected).
- No new default flipped without the GPU A/B in A5 / B1 passing its gate.
- No FLUX/SD: TeaCache is Anima-only by earlier scope choice; keep it so.

## Why (the problem the current rule has)

`TeaCache.should_compute` (`diffucore/src/diffucore/models/anima_dit.py`,
class at ~L497) accumulates the relative-L1 drift of the **block-0 modulated
input**, optionally remapped through a degree-4 polynomial fitted by
`anima_calibrate_teacache` on a *deterministic euler* trajectory. Two known
consequences (memory `teacache-feature-status`):

- Calibrated mode misallocates recompute budget for ancestral / secant
  samplers (fit domain mismatch) — slower *and* "seed-breaking".
- Raw mode works but the threshold does not transfer: `secant_anneal/beta/32`
  is near-lossless at 0.3–0.5, `dpmpp_2m/flow/25` is destroyed above 0.005,
  `euler_ancestral` raw needs ≥0.6 to skip at all.

EasyCache gates on a running estimate of the model's own **output**
sensitivity instead of an input proxy:

```
k_t   = ‖v_t − v_{t−1}‖ / ‖x_t − x_{t−1}‖     "transformation rate", refreshed only on computed steps
ε_t   = k · ‖x_t − x_{t−1}‖ / ‖v_{t−1}‖        predicted relative change of this step's output
E_t   = Σ ε_n  since the last computed step
compute if  E_t ≥ τ   (τ ≈ 0.02–0.10, 0.05 default)   or during the first R warm-up calls
```

Because it reads actual velocity change, an ancestral noise injection shows up
as a large ‖x_t − x_{t−1}‖ and forces a recompute exactly where structure is
being decided — what the raw threshold achieved by accident. Paper numbers on
FLUX.1-dev, 50 steps: ×4.64 vs TeaCache ×3.27, FID 25.8→23.2. Treat those as
motivation only; the gate for us is A5.

---

## Part A — EasyCache rule inside `TeaCache`

### A1. Core: `models/anima_dit.py`

**`TeaCache.__init__`** gains two keyword args, validated like `basis`:

- `rule: str = "drift"` — `"drift"` (today's behavior, bit-for-bit) or `"easy"`.
- `warmup: int = 3` — number of *calls* of this stream that always compute
  under `"easy"` (paper: R = 5–10 of 50 steps; 3 ≈ 10 % of a 28–32 step run).
  Ignored under `"drift"` (which forces only the first call — keep that).

New state (all `None`/0 until used): `prev_x`, `prev_out`, `k`
(`Optional[float]`), `pending_dx` (`float`), `last_computed: bool`.

**`should_compute(modulated)` — drift path: unchanged.** Only add
`self.last_computed = <decision>` before each `return` (needed by B3 and by
tests; zero behavioral effect).

**New `should_compute_easy(x)`** — `x` is the DiT's `(B, C, T, H, W)` latent
input (post `_pad_to_patch_size`, see A1-forward). Pseudo-code:

```python
def should_compute_easy(self, x):
    self.calls += 1
    if self.prev_x is None:                     # first call: no history
        self.prev_x = x.detach().clone()
        self.pending_dx = 0.0
        self.accumulated = 0.0
        self.last_computed = True
        return True
    dx = (x - self.prev_x).abs().mean().item()  # same mean-abs convention as the drift rule
    self.prev_x = x.detach().clone()
    self.pending_dx = dx
    if self.calls <= self.warmup or self.k is None or self.prev_out is None:
        decision = True                         # warm-up / rate not yet measurable
    else:
        v_norm = self.prev_out.abs().mean().item()
        self.accumulated += self.k * dx / max(v_norm, 1e-8)
        decision = self.accumulated >= self.rel_l1_thresh
    if decision:
        self.accumulated = 0.0
    else:
        self.skips += 1
    self.last_computed = decision
    return decision
```

**New `record_output(out)`** — called by the forward on *every* call (computed
or skipped), with the unpatchified output (same shape as `x`):

```python
def record_output(self, out):
    if self.last_computed and self.prev_out is not None and self.pending_dx > 0:
        self.k = (out - self.prev_out).abs().mean().item() / self.pending_dx
    self.prev_out = out.detach().clone()
```

Notes for the executor:
- `rel_l1_thresh` is reused as τ under `"easy"` (a fraction: 0.05 = 5 %).
  Don't add a second threshold field; the UI slider's 0.005–1.0 range already
  covers τ. `coefficients` are **ignored** under `"easy"` (no rescale exists in
  this rule) — document that in the docstring.
- `k` is refreshed only after a *computed* step (paper); on a skipped step
  `prev_out` becomes the forecast output, as in the paper's cached-output reuse.
- `record` (calibration) mode is drift-only: raise `ValueError` on
  `record=True, rule="easy"`.
- Check the reference implementation's norm choice
  (https://github.com/H-EmbodVis/EasyCache — the FLUX/Wan wrappers) and mirror
  it if it differs from mean-abs; only ratios matter, but stay faithful.
- Memory cost: two latent-sized clones per stream (≈0.5–1 MB at 1024²).
- `.item()` syncs: same count as the drift rule already pays. Fine.

**`CosmosDiT.forward`** (~L702–752). Today:

```python
compute = teacache.should_compute(
    self.blocks[0].modulated_self_attn_input(x_B_T_H_W_D, emb_B_T_D, adaln_lora_B_T_3D)
) if teacache is not None else True
```

Becomes a rule dispatch — do **not** compute the modulated probe under
`"easy"` (it is a LayerNorm + adaLN we no longer need):

```python
if teacache is None:
    compute = True
elif teacache.rule == "easy":
    compute = teacache.should_compute_easy(x)          # x = padded (B,C,T,H,W) latent
else:
    compute = teacache.should_compute(self.blocks[0].modulated_self_attn_input(...))
```

and after `out = self._unpatchify(out)[...]`:

```python
if teacache is not None and teacache.rule == "easy":
    teacache.record_output(out)
return out
```

The skip branch (`forecast()`) and the computed branch (`update(residual)`) are
untouched — the forecast machinery is shared by both rules.

### A2. Pipeline: `pipelines/_anima.py`

`_make_teacache(thresh, coeffs, cfg_scale, forecast="hermite")` (L96–115)
gains `rule: str = "drift"`, `warmup: int = 3`, and (for B2) `uncond_scale:
float = 1.0`. Under `rule == "easy"` drop `coefficients` from the kwargs (the
engine still passes them; the pipeline is where the "ignored" contract lives).
Both streams get the same rule/warmup. Validate `rule` here too, mirroring the
`forecast` ValueError, so the existing
`test_unknown_basis_and_forecast_mode_are_rejected` pattern extends naturally.

`anima_text_to_image` (signature ~L117–144) and `anima_img2img` (~L348–378)
gain `teacache_rule: str = "drift"` next to `teacache_forecast` and pass it to
`_make_teacache` at the two call sites (~L249, ~L481). Update the docstring
paragraph that documents `teacache_forecast` (~L167–173).

Extend the two `print(f"[teacache] cached {tc_cond.skips}/{tc_cond.calls}
steps")` lines (~L333, ~L536) to also print the uncond stream's `skips/calls`
when `tc_uncond` exists, and — for B3's go/no-go — a count of steps where the
cond stream skipped but the uncond computed (see B3; add `follow_misses` or
compute it in the pipeline from per-call `last_computed` pairs).

`anima_calibrate_teacache` (~L633, builds `TeaCache(0.0, record=True)`) is
drift-only and needs no change.

### A3. Plumbing (mirror `teacache_forecast` exactly)

Every place `teacache_forecast` flows, `teacache_rule` flows beside it:

| Layer | File / anchor | Change |
|---|---|---|
| dispatchers | `pipelines/text_to_image.py` L58/L82, `image_to_image.py` L53/L79, `inpaint.py` L65/L94 | new kwarg, forwarded to the Anima branch only |
| engine | `backend/engine.py` `generate_t2i` ~L1347–1375, `generate_i2i` ~L1421–1453, `generate_inpaint` ~L1499–1532, `detail` ~L1574–1645, `upscale` ~L1676–1758 | `teacache_rule: str = "drift"` param, threaded into the pipeline kwargs beside `teacache_forecast` |
| server payloads | `backend/server.py` `GeneratePayload` L348–350, `DetailPayload` L409–411, `UpscalePayload` L431–433, `XYZPayload` L525–527 | `teacache_rule: str = "drift"`; validate `in ("drift", "easy")` (a `field_validator`, like nothing else here does — a 400 beats a 500 from the pipeline) |
| server handoff | `common` dict L702–710; upscale site L813–815; detailer site L854–856; XYZ `base_kwargs` L935–937; standalone detail L1875–1877; standalone upscale L1953–1954 | pass `teacache_rule=p.teacache_rule` |
| metadata writer | `backend/metadata.py` L237–242 (A1111 fields) and L341–344 (`extra` dict) | add `TeaCache rule: <rule>` / `extra["teacache_rule"]` **only when TeaCache ran**, same guard as the forecast line; write it always (absent ⇒ pre-feature image ⇒ drift) |
| metadata parser | `backend/metadata.py` L608–625 | `out["teacacheRule"] = meta.get("teacache_rule", "drift")`, validated against the two values, inside the `thresh > 0` block |
| UI form | `static/app.js` L95 form defaults; payloads L1112–1114 (generate), L1274–1276, L1636–1637 (upscale), L1753–1755 (detail); `applyFields` key list L2065 | `teacacheRule: 'drift'` default; add to every payload that sends `teacache_forecast`; add `'teacacheRule'` to the restore keys |
| UI markup | `static/index.html` TeaCache block L321–346 | a **Decision rule** `<select>` above the forecast select: `drift` = "Input drift (TeaCache, calibratable)", `easy` = "Output change (EasyCache, no calibration)". Hide the *Use calibrated coefficients* chip and its `<p class="sub">` when `form.teacacheRule === 'easy'` (`x-show`). Swap the threshold hint (`<details>` ~L330–334) by rule: for easy, "τ is the predicted relative output change allowed before a recompute; 0.05 is the paper default, 0.02 conservative, 0.10 aggressive." |

The `settings.json` / `/api/settings` layer is **not** involved in Part A (the
rule is a per-generation form field, like the forecast basis).

### A4. Tests (offline, must stay CPU-runnable)

`diffucore/tests/test_anima_teacache.py` — reuse `_tiny()`:

1. `test_easy_rule_warmup_always_computes` — `TeaCache(0.05, rule="easy",
   warmup=3)`: three forwards with different inputs all compute
   (`skips == 0`), outputs bit-equal to the plain forward.
2. `test_easy_rule_identical_input_skips_after_warmup` — `warmup=1`, threshold
   huge: call twice with *different* inputs (so `k` exists), then the same input
   again → `dx == 0` → skip; output ≈ previous (allclose, as
   `test_skip_reuses_exact_residual`).
3. `test_easy_rate_updates_only_on_computed_steps` — drive `should_compute_easy`
   + `record_output` directly with 1-D tensors; assert `k` unchanged across a
   skipped call and refreshed after a computed one; `accumulated` resets on
   compute and carries across skips (mirror `test_threshold_forces_recompute_and_resets`).
4. `test_easy_zero_threshold_never_skips` (τ = 0 ⇒ `accumulated >= 0` always).
5. `test_easy_rejects_record_mode` and `rule="chebyshev"` → `ValueError`; extend
   `test_unknown_basis_and_forecast_mode_are_rejected` for `_make_teacache(...,
   rule="bogus")`.
6. `test_make_teacache_rule_and_coeffs` — `rule="easy"` with coefficients
   passed ⇒ both streams have identity `coefficients` and `rule == "easy"`;
   `rule="drift"` keeps them.
7. `test_drift_rule_unchanged` — a guard: run the existing skip/forecast tests'
   scenario with an explicit `rule="drift"` and compare `skips`, `accumulated`,
   and outputs against a `TeaCache` built without the kwarg (`torch.equal`).

Backend (find the existing suite with `grep -rn teacache_forecast backend/tests
diffucore/tests`): metadata writer/parser round-trip for `TeaCache rule:` incl.
the absent-key ⇒ `drift` case; payload default + rejection of an unknown rule.

Gate: full offline collection green, incl. the CUDA-gated ones being skipped
cleanly. **Do not run the CUDA suite with the server up** (memory
`kill-diffucore-ui-before-gpu`).

### A5. GPU A/B (the acceptance gate for the feature and for any default flip)

Kill the server first (`pgrep -af backend/app.py` → `kill`). Write a small
harness under `scripts/` (e.g. `ab_teacache_rule.py`) that loads the user's
current Anima checkpoint through `backend/engine.py` (or
`load_anima_checkpoint` as `diffucore/tests/_perf_validation.py` does) and,
for a fixed prompt/negative/seed, renders:

- **Reference:** TeaCache off.
- **Drift raw** at the thresholds already characterised (0.3, 0.6).
- **Easy** at τ ∈ {0.03, 0.05, 0.08, 0.12} × warmup ∈ {2, 3, 5}.

Configs (the user's real regime, 768², CFG 4.5, fp16, fp16_acc + fa2 on):
`secant_anneal/beta_mix/32`, `cogent3_pump/beta_mix/28`,
`euler_ancestral/flow/28`, plus the known-fragile `dpmpp_2m/flow/25` as a
canary. Three seeds each. Record per run: wall time, `skips/calls` for both
streams, RMSE vs the reference image (0–255 scale), and save the PNGs for the
user's eyes. The uncached same-config image **is** a valid reference here
(same sampler trajectory, only caching differs) — the "no converged reference"
caveat is about comparing *different* configs, not this.

Pass criteria (all three needed to flip the UI default to `easy`):

1. At matched skip counts, easy RMSE ≤ drift-raw RMSE on ≥ 2/3 seeds per config.
2. One τ (expected 0.05–0.08) is "imperceptible" (RMSE at or below the
   drift-raw-0.3 level on secant_anneal) on **all three** real configs without
   retuning — the transfer property.
3. The dpmpp_2m canary degrades no *faster* than drift does at matched skips.

If only 1 passes: ship the option, keep `drift` default, document τ per sampler.
If none: keep the code behind the dropdown, mark it experimental in GUIDE.md, and
record the numbers in memory so nobody re-runs the study.

Also smoke once: `easy` + `compile=True` (no cuda_graphs) at 512², 6 steps —
the added `.item()`s graph-break like the drift rule's do; just confirm no crash.

### A6. Docs

- `GUIDE.md` TeaCache section (L631–694): a **Decision rule** bullet mirroring
  the *Forecast basis* one — what it measures, that calibration does not apply,
  τ guidance from A5, and which rule metadata restores.
- `diffucore/docs/RUNTIME_SPEC.md` if it lists TeaCache parameters (grep).
- Memory: update `teacache-feature-status` with the A5 numbers and the chosen
  default; link `[[anima-speed-research-2026-09]]`.

---

## Part B — cheaper uncond pass

### B1. Guidance-interval **start > 0** — measurement only, no code

The settings knobs already exist (`Settings → Sampler & scheduler defaults`,
`cfg_interval_start/end`, `static/index.html` L941–946; injected into
`common` only when non-default, `backend/server.py` L736–744; mapped to sigma
bounds by `guidance_interval_bounds`, `sampling/denoiser.py` L20–36). The
paper's headline (Kynkäänniemi et al. 2024) is that skipping guidance at the
*highest* noise improves quality; the repo only ever measured `(0, 0.75)`.

Protocol (server down, same harness style as A5, TeaCache **off** so the two
studies don't confound): start ∈ {0, 0.1, 0.2} × end ∈ {0.75, 1.0}, three
seeds, two prompts (one short, one detailed — Anima behaves differently,
memory `anima-needs-detailed-prompts`), the user's samplers. Report wall time
and save PNGs side by side. Quality here is **visual** — RMSE vs the `(0,1)`
image is not a quality metric because the start knob changes the image on
purpose. Deliverable: a recommendation for the shipped default written into
the GUIDE bullet at L707–708 (which currently says "worth an A/B"). Do not flip
the default yourself; propose it with the images.

Expected: start 0.1–0.2 saves a further 10–20 % of uncond forwards with equal
or better composition. If a prompt-adherence loss is visible on the short
prompt, say so — that is the likely failure mode.

### B2. Looser threshold on the uncond cache stream

**Read this first.** With `v = v_uncond + s·(v_cond − v_uncond)` at s = 4.5,
an error in `v_uncond` enters the guided velocity with weight |1 − s| = 3.5
and an error in `v_cond` with weight 4.5. The uncond stream is *not* less
important; it is only empirically *smoother* (fewer prompt-specific features),
so at the same threshold it already skips more. A looser uncond threshold is
therefore a hypothesis, not a given — B2 exposes the knob and measures.

Code (small):

- `_make_teacache(..., uncond_scale=1.0)`: `uncond = TeaCache(thresh *
  uncond_scale, ...)`. Works under both rules.
- Pipeline param `teacache_uncond_scale: float = 1.0` on `anima_text_to_image`
  / `anima_img2img`, dispatchers, and the engine's 3 main generate methods (also
  `detail`/`upscale` for consistency — they take every other TeaCache knob).
- **Settings-level knob, not a form field** — same handling as the CFG interval:
  `Settings` model in `backend/server.py` (~L474; add `teacache_uncond_scale:
  float = Field(1.0, ge=1.0, le=4.0)`), persisted in `settings.json`, a field in
  the Settings → TeaCache tab (`static/index.html` ~L880+, `static/app.js` L193
  defaults + the save path ~L1395–1404), injected into `common` only when ≠ 1.0
  (beside the CFG-interval injection, L736–744). XYZ deliberately does not
  inject settings knobs (consistent with curvature/eta_max/CFG interval) — keep
  that.
- Metadata: `TeaCache uncond scale: <x>` line only when ≠ 1.0, inside the
  TeaCache-ran guard; **not** restored on load (settings-level, like the CFG
  interval line at `metadata.py` L218).
- Tests: `_make_teacache` scale mapping; server injection only when non-default;
  metadata line presence/absence.

A/B (after A5, using its winning rule and τ): scale ∈ {1.0, 1.5, 2.0, 3.0} on
the three real configs, three seeds; report both streams' skips, wall time,
RMSE vs uncached. Pass: a scale > 1 that cuts total forwards by ≥ 10 % with RMSE
within noise of scale 1.0 on all three configs. If none, leave the default at
1.0 and note the amplification arithmetic above as the explanation.

### B3. Uncond follows cond — conditional on instrumentation

Rationale: a step where the cond stream skipped but the uncond computed is a
forward spent on a step the sampler already deemed "smooth". **Instrument
before building:** the A2 print extension reports `follow_misses` (cond
skipped ∧ uncond computed) per generation. If, across the A5/B2 runs, that is
< 10 % of steps, **skip B3 entirely** — the uncond stream is already the more
aggressive skipper and there is nothing to save. Record the count in memory
either way.

If it is ≥ 10 %:

- `TeaCache` gains `leader: Optional["TeaCache"] = None`. In both
  `should_compute*` paths, after the stream's own decision: `if self.leader is
  not None and not self.leader.last_computed: decision = False` — and when this
  overrides a would-be compute, **do not reset `accumulated`** (the debt carries
  to the next step where the leader computes). Count the override as a skip.
- Pipeline: `tc_uncond.leader = tc_cond` when `teacache_uncond_follow` (a
  Settings boolean beside `teacache_uncond_scale`, default off, same injection
  rule). The CFG interval already prevents uncond calls out of band; the leader
  link only ever acts in band, so no interaction.
- Tests: leader-skip forces a skip without resetting the follower's
  accumulator; leader-compute leaves the follower's own decision untouched;
  `leader=None` is bit-identical to today.
- A/B: on/off at the B2 winner; pass = forwards down ≥ 5 % at RMSE within noise.

---

## Order of work and verification checklist

1. A1 + A2 + A4 (core + pipeline + offline tests) → `pytest diffucore/tests
   -q` green; existing TeaCache tests untouched and green; `rule="drift"`
   bit-identical guard test green.
2. A3 (plumbing end-to-end) → backend tests green; start the server, generate
   once with each rule via the UI, confirm the `[teacache] cached a/b` line
   prints both streams, the PNG carries `TeaCache rule:`, and loading that PNG
   back restores the dropdown. Hard-reload the browser after editing
   `static/` (memory `static-asset-stale-cache`).
3. A5 GPU A/B (server down) → decide default → A6 docs + memory.
4. B1 measurement (server down) → recommendation + GUIDE bullet.
5. B2 code + tests → B2 A/B → default decision.
6. B3 only if the instrumentation says so.

Commit granularity: one commit per numbered step, `feat(teacache): …` /
`docs(teacache): …` style as in the log; the plan doc's status line gets
updated at the end with what shipped and the measured numbers.

## Don'ts

- Don't change `should_compute`'s drift math, `update`, `forecast`, or
  `_basis_weight` — the HiCache/TaylorSeer tests pin them.
- Don't couple the rule to `num_steps` (the cache is sampler-agnostic by
  design; a forced last-step compute was deliberately dropped). If the A5 tail
  shows artifacts, prefer a smaller τ over re-adding it, and report.
- Don't add a UI knob for `warmup`; pick it in A5 and hard-code in
  `_make_teacache`.
- Don't run GPU work with the server up, and don't run the full CUDA suite
  back-to-back with a full-res generation (machine freeze, see memory).
- Don't judge B1 with RMSE; don't judge A5/B2 without it.
