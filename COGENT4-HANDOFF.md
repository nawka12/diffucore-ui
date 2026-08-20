# COGENT4 handoff — what is left before this is release-able

*Written 2026-08-20 for whoever picks this up next. Read this first, then
`COGENT-IMPROVE.md` (the design record and its five review passes) and
`COGENT-IMPROVE-IMPLEMENTED.md` (what was built and every measurement).
Everything below is either measured or explicitly flagged as unmeasured.*

## 1. Where things stand in one paragraph

The cogent4 design was cut down by review to exactly two items: a measurement
hook on the coherence gate, and a per-channel gate reduction as a default-off
experiment. Both are implemented, tested, and merged. The measurement hook has
already done its job and produced findings that are recorded in the docs. The
per-channel gate works, costs nothing, and **has no evidence that it is
better** — the offline toy says it wins at coarse steps and loses at fine ones,
and the real-image A/B could not rank it at all. Nothing is reachable from the
UI, and nothing about a user's generations has changed: the default path is
byte-identical, verified against a pre-instrumentation image on a real model.

**So cogent4 is not "almost shipped". It is one open question away from either
shipping or being reverted, and that question is empirical.**

## 2. The one question that decides everything

> Is the per-channel gate better than the global gate on real images?

Evidence so far, all of it in `COGENT-IMPROVE-IMPLEMENTED.md`:

- **Toy (offline):** per-channel wins det RMSE at every step count (~1-2%),
  wins rough energy distance at 8/12/16 steps, **loses at 24/32**.
- **Real images (production settings):** the change reaches every image
  (SSIM 0.94-0.99 on `cogent3`, down to 0.68 on `cogent3_pump`) and costs no
  time, but no objective ranking is possible — see §4.
- **Convergence:** the two arms agree more with each other as steps rise
  (RMSE 5.21 at 96 steps), so whatever per-channel does, it does it at
  **coarse** step counts. That is consistent with the toy.

## 3. The plan, in order

### Step 1 — the multi-seed paired test (the only objective route left)

Single-seed comparisons cannot work (§4). A paired test over many seeds can,
because a systematic bias survives the sampling chaos that defeats individual
pairs.

```
.venv/bin/python scripts/ab_cogent4_per_channel.py confirm --confirm-steps 32
```

The `confirm` stage is wired but **has never been run**. Before running it:

- Widen it to N >= 12 seeds (it currently uses `base.SEEDS`, which is 2).
  Prompts x seeds x arms x combos; at ~64 s/image budget accordingly
  (12 seeds x 2 prompts x 2 arms x 1 combo ~ 51 min on a 2060).
- Score with a **paired** test per proxy (Wilcoxon signed-rank or a plain sign
  test over the per-seed deltas), not a mean over unpaired images. The proxies
  are already computed per cell in `pair_stats`.
- Run `cogent3 x beta_mix` first. If you can afford it, `cogent3_pump x beta`
  second — it is where the gate change is largest, so it is both the most
  likely win and the most likely regression.

Interpret honestly: proxies are not quality. A significant sharpness shift is a
reason to *look*, not a verdict.

### Step 2 — the human verdict

This family's real wins were called by eye, not by metric (the pump's
prompt-coherency win is the precedent). The paired image set is ready:

```
outputs/ab_cogent4_per_channel/SHEET_<sampler>__<scheduler>.png
```

Three sheets, each 4 rows (steps 8/16/24/32) x 3 columns
(GLOBAL | PER-CHANNEL | amplified |diff| x8). Trust the low-step rows most —
that is where the gate difference is largest and where the toy predicts a win.

### Step 3a — if it wins: plumb it

`gate_reduce` currently appears in **no file** outside
`diffucore/src/diffucore/sampling/samplers.py` and `scripts/`. Expose it the
way `curvature` is exposed — that is the reference implementation to copy:

```
diffucore/src/diffucore/pipelines/{text_to_image,image_to_image,inpaint,_anima}.py
backend/engine.py   backend/server.py   static/index.html   static/app.js
```

Plus two things `curvature` will not teach you:

- **Record it in generation metadata.** Two images from the same seed and
  settings differ depending on this flag; without it in metadata they are not
  reproducible.
- **Decide the pump's behaviour explicitly.** `cogent3_pump` is
  `partial(sample_cogent3, pump_strength=0.08)`, so a naive plumb changes the
  pump too. That is the highest-variance cell measured (SSIM 0.68 vs global).
  Either gate them separately or state in the docs that the knob moves both.

### Step 3b — if it loses or stays inconclusive: revert the arm, keep the hook

Drop the `reduce` / `per_channel` branch, keep `stats_out` / `gate_stats`, and
record the negative in `COGENT-IMPROVE-IMPLEMENTED.md`. The hook is worth
keeping on its own merits: it is exactly the *raw, unfloored statistics plus
floor-active flag* interface that [CDX] 4 named as a precondition for ever
building the deferred closed-loop scheduler.

## 4. Landmines — read before designing any experiment

These cost real GPU hours to discover. Do not rediscover them.

1. **There is no converged reference image at CFG 4.5.** The successive-
   refinement ladder does not shrink (`s24->s32` 22.4, `s32->s64` 44.2,
   `s96->s128` 34.5), and `uni_pc` at 48 vs 96 steps differs by RMSE 187.9
   because one renders a black background and the other white. Both images are
   clean. Changing step count changes *which sample you land on*. **Any metric
   of the form "distance to a converged/high-step reference" is void here.** A
   metric ships with its own falsification test or it does not ship — the
   reference-agreement check is the only reason this was caught.
2. **Single-seed deltas at `eta_max=1.0` are trajectory reshuffling.** The sign
   of the sharpness delta flips with step count inside one combo. Do not read
   them as quality.
3. **`gate_reduce` is not forwarded by any pipeline.** To A/B it without
   touching production code, rebuild the registry entries in-process — see
   `set_arm()` in `scripts/ab_cogent4_per_channel.py`. Rebuild from the
   captured originals so `cogent3_pump` keeps its baked `pump_strength`.
4. **`| tee` masks a Python traceback behind exit code 0.** Use
   `set -o pipefail` for background harness runs.
5. **Kill the UI server (`python backend/app.py`) before any GPU work** — it
   holds VRAM and causes contention on the 12 GB 2060.
6. **The default-path pin is a real safety net.** `pin` stage regenerates a
   cell whose image predates the instrumentation and compares sha256. If it
   ever stops matching, the "bit-for-bit unchanged" claim has broken.

## 5. Do not rebuild these

The design record killed three mechanisms after five review passes. Each has a
derivation on file explaining exactly why. If you find yourself designing one,
read `COGENT-IMPROVE.md` first:

- **Lag-2 measurement (both ratio and cosine forms) — dropped.** `(1+2rho)/3`
  is an exact identity, not a linearization; `v_est` is already free at lag-1;
  and the ratio form damps a clean straight trajectory by up to 48% on the
  default schedule.
- **`v_model = v_est - g^2 * v_inj` — shelved.** `g_i` is a Jacobian, not a
  scalar, and the published `v_inj` formula was indexed to the wrong step.
- **Closed-loop scheduler — deferred.** It consumed a floored, clamped
  statistic as if it were unfloored, had no defined value on the untraversed
  lambda band, and needs a sampler<->scheduler interface that does not exist.
  Its premise (terminal depth / final lambda-step correlate at +0.91/+0.94)
  is still real.

## 6. Definition of release-able

cogent4 can ship when **all** of these hold:

1. A paired multi-seed test or a recorded human verdict says per-channel is
   better, on a named combo, at named step counts — with the negative cells
   recorded too.
2. `gate_reduce` is reachable from the UI and written into generation metadata.
3. The pump's coupling is decided and documented.
4. The default (`"all"`) path is still byte-identical — re-run the `pin` stage.
5. `COGENT-IMPROVE-IMPLEMENTED.md` carries the deciding measurement, including
   whatever did not work.

If step 1 fails, the release-able outcome is §3b: revert the arm, keep the
hook, and the work still ends net-positive because the measurement findings
(the floor drives the gate on 35-100% of a 24-step schedule; `v_est` blends
model error with the denoiser's ancestral response) are already banked in
`docs/cogent.md`.
