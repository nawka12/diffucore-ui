"""Headless CLI to fit TeaCache rescaling coefficients for an Anima model.

A thin wrapper over ``Engine.calibrate_teacache``. TeaCache coefficients are an
architecture-level property (arXiv:2411.19108): one fit transfers across a
family's checkpoints and across step counts / resolutions, so this writes a
single ``models/teacache_cache/<family>.json`` that every Anima checkpoint then
picks up automatically. Run it once; generations with the TeaCache toggle on
will use the fitted polynomial instead of the identity fallback.

    python calibrate_teacache.py \
        --dit anima-base-v1.0.safetensors \
        --vae qwen_image_vae.safetensors \
        --te  qwen_3_06b_base.safetensors \
        --steps 50
"""

from __future__ import annotations

import argparse

from engine import Engine


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dit", required=True, help="Anima DiT name (under models/diffusion-models)")
    ap.add_argument("--vae", required=True, help="Qwen-Image VAE name (under models/vae)")
    ap.add_argument("--te", required=True, help="Qwen3 text-encoder name (under models/text-encoders)")
    ap.add_argument("--steps", type=int, default=50,
                    help="trajectory length for the fit; higher = denser σ sampling")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--shift", type=float, default=3.0)
    ap.add_argument("--cfg", type=float, default=4.0, help="CFG scale used during calibration")
    ap.add_argument("--prompt", default="a detailed photograph of a fox in a forest")
    ap.add_argument("--negative", default="blurry, low quality")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    engine = Engine()
    print(engine.load_anima(args.dit, args.vae, args.te))

    def on_progress(done: int, total: int) -> None:
        print(f"\r  trajectory {done}/{total}", end="", flush=True)

    info = engine.calibrate_teacache(
        prompt=args.prompt, negative_prompt=args.negative,
        steps=args.steps, width=args.width, height=args.height, shift=args.shift,
        cfg_scale=args.cfg, seed=args.seed,
        progress_callback=on_progress,
    )
    print()
    print(info)


if __name__ == "__main__":
    main()
