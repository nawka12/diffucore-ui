# Model ports: todo

One model at a time, each finished end to end (engine, review, merge to
diffucore `main`, UI on `dev`) before the next starts.

Done: Z-Image (Turbo + Base), diffucore `5743599`, UI `71fe774` (pushed).

## Dropped (2026-11-07)

FLUX.3 Klein, Qwen-Image 2.1 and Qwen-Image: not popular with the anime
community, so only Z-Image ships. Their worktrees and branches were deleted.

## Later

- Z-Image Omni (editing).

## Z-Image follow-ups

- Settings modal: the one-line CFG notes for Krea 2 / MOSS / Z-Image conflict
  with "UI should explain itself"; redesign all three together.
- img2img on Base with a non-`zimage` scheduler uses shift 3 instead of 6
  (`ImageToImage` has no `shift` argument).
- Unverified: `int8_turing` / `fa2` attention, `compile`, a Base LoRA, the
  XYZ Checkpoint axis, a 4 GB card.
- Decide whether to keep the per-family picker memory (`rememberFamily` /
  `recallFamily` in `static/app.js`).

## Environment notes

- The diffusers reference env lived in a session scratchpad; the recipe to
  recreate it is in Claude's memory (`four-model-ports-2026-11`).
- KyoNaniTabeta (`/dev/sda2`) is mounted at `/mnt` by `~/mount-kyo.sh` and
  does not survive a reboot; nothing live is stored there yet.
