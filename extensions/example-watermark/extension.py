"""Example Diffucore UI extension — a seed watermark.

Demonstrates the full extension surface:

* a ``post_generate`` hook that post-processes the image (stamps the seed into
  the bottom-right corner),
* a custom API router (``GET /api/ext/example-watermark/status`` and
  ``POST /api/ext/example-watermark/settings``),
* extension-specific settings stored via ``api.get_setting`` / ``api.set_setting``,
* a ``post_save`` hook that just logs the saved path.

The matching ``web/example.js`` registers a tab and a settings panel through the
``window.DiffucoreExt`` bridge so the UI side is covered too.

Read this file alongside docs/EXTENSIONS.md.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

# PIL ships with the app (Pillow is a core dependency).
from PIL import Image, ImageDraw, ImageFont


router = APIRouter()


class WatermarkSettings(BaseModel):
    enabled: bool = True
    color: str = "#e8a065"
    prefix: str = "seed:"


def _draw_watermark(image: Image.Image, text: str, color: str) -> Image.Image:
    """Stamp ``text`` into the bottom-right corner of ``image``.

    Returns a new image (the hook replaces ctx.image with it). Uses a default
    PIL font so the extension works without shipping any assets; the text is
    drawn on a copy so the caller's image is never mutated in place.
    """
    img = image.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.load_default()
    except OSError:
        font = None
    # size the text, then place it with a small margin from the corner
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    margin = 8
    x = img.size[0] - tw - margin
    y = img.size[1] - th - margin
    # a subtle dark plate behind the text so it reads on any image
    draw.rectangle([x - 4, y - 2, x + tw + 4, y + th + 2], fill=(0, 0, 0, 140))
    draw.text((x, y), text, font=font, fill=color)
    return Image.alpha_composite(img, overlay).convert("RGB")


def setup(api):
    """Entry point called by the loader with an ExtensionAPI instance.

    Everything the extension registers (hooks, routes, settings) flows through
    ``api`` so the loader can unwind it on disable/reload.
    """
    # ── settings: persisted in extensions/state.json under this ext's name ──
    # Defaults are read on first access; set_setting writes through to disk.
    def is_on() -> bool:
        return bool(api.get_setting("enabled", True))

    def get_color() -> str:
        return str(api.get_setting("color", "#e8a065"))

    def get_prefix() -> str:
        return str(api.get_setting("prefix", "seed:"))

    # ── hooks ──────────────────────────────────────────────────────
    # post_generate fires after generation (and the detailer/upscaler) but
    # before the image is saved, so the stamped image is what hits disk. The
    # hook receives a HookContext; replacing ctx.image swaps the saved output.
    def on_post_generate(ctx):
        if not is_on():
            return
        seed = api.engine.last_seed if api.engine is not None else "?"
        text = f"{get_prefix()}{seed}"
        ctx.image = _draw_watermark(ctx.image, text, get_color())

    # post_save is fire-and-forget; here it just demonstrates the event.
    def on_post_save(ctx):
        Path(ctx.path).name  # touch the path so the example is self-contained
        api.broadcast({
            "type": "ext:example-watermark",
            "path": str(ctx.path),
        })

    api.on("post_generate", on_post_generate)
    api.on("post_save", on_post_save)

    # ── custom API ─────────────────────────────────────────────────
    # The router is mounted at /api/ext/example-watermark (the loader adds the
    # prefix), so a route defined as "" resolves to that exact path.
    @router.get("")
    def status():
        return {
            "enabled": is_on(),
            "color": get_color(),
            "prefix": get_prefix(),
            "loaded_model": (api.engine.loaded_name if api.engine is not None else None),
        }

    @router.post("/settings")
    def update_settings(s: WatermarkSettings):
        api.set_setting("enabled", s.enabled)
        api.set_setting("color", s.color)
        api.set_setting("prefix", s.prefix)
        return {"ok": True, "settings": s.model_dump()}

    api.add_api_router(router)
