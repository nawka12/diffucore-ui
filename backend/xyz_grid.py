"""X/Y/Z plot grid generation — compare parameter combinations side by side."""

from __future__ import annotations

import random
import time
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageFont

from engine import ENGINE

# ── public constants ──────────────────────────────────────────────

PARAM_TYPES = ["None", "Seed", "Sampler", "Scheduler", "Steps", "CFG Scale", "Prompt S/R"]

_PARAM_MAP: dict[str, str] = {
    "Seed": "seed",
    "Sampler": "sampler",
    "Scheduler": "scheduler",
    "Steps": "steps",
    "CFG Scale": "cfg_scale",
}

# ── font helpers ──────────────────────────────────────────────────

_FONT_CACHE: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}


def _get_font(size: int = 14):
    if size not in _FONT_CACHE:
        for path in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/noto/NotoSans-Regular.ttf",
            "DejaVuSans.ttf",
        ):
            try:
                _FONT_CACHE[size] = ImageFont.truetype(path, size)
                break
            except (IOError, OSError):
                continue
        else:
            # Pillow >= 10.1 returns a *scalable* default when given a size;
            # the no-arg form is a fixed ~10px bitmap that ignores `size`.
            try:
                _FONT_CACHE[size] = ImageFont.load_default(size)
            except TypeError:
                _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


# ── value parsing ─────────────────────────────────────────────────

def resolve_values(param_type: str, values_str: str, base_value: Any) -> list:
    """Parse a comma-separated values string into a typed list.

    Falls back to ``[base_value]`` when the string is empty or param is None.
    """
    if param_type == "None" or not values_str.strip():
        return [base_value]
    # Prompt S/R keeps empties: a trailing comma (``promptA,``) asks for a cell
    # with the search term replaced by nothing — i.e. an image *without* it — so
    # an empty replacement is meaningful and must survive the filter below.
    if param_type == "Prompt S/R":
        return [v.strip() for v in values_str.split(",")]
    raw = [v.strip() for v in values_str.split(",") if v.strip()]
    if not raw:
        return [base_value]
    if param_type in ("Seed", "Steps"):
        return [int(v) for v in raw]
    if param_type == "CFG Scale":
        return [float(v) for v in raw]
    return raw


# ── grid assembly ─────────────────────────────────────────────────

CELL_BORDER = 2


def _make_grid(
    images: list[list[Image.Image]],
    x_labels: list[str],
    y_labels: list[str],
) -> Image.Image:
    """Assemble a 2-D grid from a list-of-lists of PIL Images.

    Label/header sizes scale with the cell resolution so the text stays
    readable on large grids.
    """
    n_rows = len(images)
    n_cols = len(images[0]) if n_rows else 0
    if n_rows == 0 or n_cols == 0:
        raise ValueError("Empty image grid")

    cell_w, cell_h = images[0][0].size

    font_size = max(20, min(cell_w, cell_h) // 22)
    font = _get_font(font_size)
    pad = max(10, font_size // 2)

    def _text_w(s: str) -> int:
        box = font.getbbox(str(s))
        return box[2] - box[0]

    header_h = font_size + 2 * pad
    max_label = max((_text_w(s) for s in y_labels), default=0)
    label_w = max_label + 2 * pad if max_label else CELL_BORDER

    grid_w = label_w + n_cols * cell_w + (n_cols + 1) * CELL_BORDER
    grid_h = header_h + n_rows * cell_h + (n_rows + 1) * CELL_BORDER

    canvas = Image.new("RGB", (grid_w, grid_h), (17, 17, 19))
    draw = ImageDraw.Draw(canvas)

    # Column headers (X labels) — teal
    for xi in range(n_cols):
        cx = label_w + CELL_BORDER + xi * (cell_w + CELL_BORDER) + cell_w // 2
        draw.text(
            (cx, header_h // 2),
            str(x_labels[xi]),
            font=font,
            fill=(93, 214, 192),
            anchor="mm",
        )

    # Row labels (Y labels) — accent orange
    for yi in range(n_rows):
        cy = header_h + CELL_BORDER + yi * (cell_h + CELL_BORDER) + cell_h // 2
        draw.text(
            (label_w // 2, cy),
            str(y_labels[yi]),
            font=font,
            fill=(232, 162, 101),
            anchor="mm",
        )

    # Cells
    for yi in range(n_rows):
        for xi in range(n_cols):
            x0 = label_w + CELL_BORDER + xi * (cell_w + CELL_BORDER)
            y0 = header_h + CELL_BORDER + yi * (cell_h + CELL_BORDER)
            img = images[yi][xi]
            if img.size != (cell_w, cell_h):
                img = img.resize((cell_w, cell_h), Image.LANCZOS)
            canvas.paste(img, (x0, y0))
            draw.rectangle(
                [x0 - 1, y0 - 1, x0 + cell_w + 1, y0 + cell_h + 1],
                outline=(44, 44, 50),
                width=1,
            )

    return canvas


# ── main generation ───────────────────────────────────────────────

def generate_xyz_grid(
    base_kwargs: dict,
    x_type: str,
    x_values_str: str,
    y_type: str,
    y_values_str: str,
    z_type: str,
    z_values_str: str,
    progress_callback: Callable[[int, int], None] | None = None,
    save_callback: Callable[[Image.Image, dict], None] | None = None,
) -> tuple[list[Image.Image], str]:
    """Generate XYZ plot grid(s).

    Parameters
    ----------
    base_kwargs : dict
        Base generation kwargs — must include *prompt*, *negative_prompt*,
        *width*, *height*, *steps*, *cfg_scale*, *sampler*, *scheduler*,
        *seed*, *shift*.
    x_type, y_type, z_type : str
        One of ``PARAM_TYPES``.
    x_values_str, y_values_str, z_values_str : str
        Comma-separated raw values for each axis.
    progress_callback : callable or None
        Called with ``(current, total)`` for progress reporting.
    save_callback : callable or None
        Called as ``(image, kwargs)`` for each successfully generated cell so
        the caller can persist individual images.

    Returns
    -------
    (grid_images, info_text)
        ``grid_images`` is a list of PIL Images — one per Z value.
    """
    # Parse axis values
    x_vals = resolve_values(
        x_type, x_values_str,
        base_kwargs.get(_PARAM_MAP.get(x_type, ""), ""),
    )
    y_vals = resolve_values(
        y_type, y_values_str,
        base_kwargs.get(_PARAM_MAP.get(y_type, ""), ""),
    )
    z_vals = resolve_values(
        z_type, z_values_str,
        base_kwargs.get(_PARAM_MAP.get(z_type, ""), ""),
    )

    total_cells = len(z_vals) * len(y_vals) * len(x_vals)
    done = 0

    # A "Prompt S/R" axis and the prompt's own <lora:…> tags both vary the prompt
    # per cell, so the prompt is re-derived (and its LoRAs re-fused) inside the
    # loop. The raw prompt — tags intact — is the search/replace target; the
    # base-parsed clean strings are only the grid's representative metadata.
    raw_prompt = base_kwargs["prompt"]
    raw_neg = base_kwargs.get("negative_prompt", "")
    clean_prompt, base_p = ENGINE.parse_lora_prompt(raw_prompt)
    clean_neg, base_n = ENGINE.parse_lora_prompt(raw_neg)

    # Search tokens for any Prompt S/R axis = that axis's first value.
    x_search = str(x_vals[0]) if x_type == "Prompt S/R" else ""
    y_search = str(y_vals[0]) if y_type == "Prompt S/R" else ""
    z_search = str(z_vals[0]) if z_type == "Prompt S/R" else ""

    info_parts = []
    t_start = time.perf_counter()
    last_loras: list | None = None   # LoRA set currently fused (None = untouched)

    try:
        if base_p or base_n:
            info_parts.append(f"LoRAs: {len(base_p) + len(base_n)}")

        base_kwargs["prompt"] = clean_prompt
        base_kwargs["negative_prompt"] = clean_neg
        base_kwargs.pop("progress_callback", None)

        # Resolve a random base seed once so every cell shares it — a fair
        # comparison grid. (When Seed is itself an axis, each cell's seed comes
        # from the axis values, so the base seed is left alone.)
        seed_is_axis = "Seed" in (x_type, y_type, z_type)
        if not seed_is_axis and base_kwargs.get("seed", -1) == -1:
            base_kwargs["seed"] = random.randint(0, 2**32 - 1)

        grid_images: list[Image.Image] = []

        for zi, z_val in enumerate(z_vals):
            rows: list[list[Image.Image]] = []

            for yi, y_val in enumerate(y_vals):
                cols: list[Image.Image] = []

                for xi, x_val in enumerate(x_vals):
                    if progress_callback is not None:
                        progress_callback(done, total_cells)

                    kwargs = dict(base_kwargs)
                    cell_prompt, cell_neg = raw_prompt, raw_neg
                    for a_type, a_val, a_search in (
                        (x_type, x_val, x_search),
                        (y_type, y_val, y_search),
                        (z_type, z_val, z_search),
                    ):
                        if a_type == "None":
                            continue
                        if a_type == "Prompt S/R":
                            cell_prompt = cell_prompt.replace(a_search, str(a_val))
                            cell_neg = cell_neg.replace(a_search, str(a_val))
                        else:
                            kwargs[_PARAM_MAP[a_type]] = a_val

                    # Re-derive this cell's prompt LoRAs; only re-fuse when the
                    # set actually changed (a cheap no-op for non-S/R sweeps).
                    cp, cp_loras = ENGINE.parse_lora_prompt(cell_prompt)
                    cn, cn_loras = ENGINE.parse_lora_prompt(cell_neg)
                    kwargs["prompt"], kwargs["negative_prompt"] = cp, cn
                    cell_loras = cp_loras + cn_loras
                    if cell_loras != last_loras:
                        ENGINE.apply_temp_loras(cell_loras)
                        last_loras = cell_loras

                    try:
                        img, _ = ENGINE.generate_t2i(**kwargs)
                        if save_callback is not None:
                            save_callback(img, kwargs)
                    except Exception as e:
                        w = kwargs.get("width", 512)
                        h = kwargs.get("height", 512)
                        img = Image.new("RGB", (w, h), (50, 20, 20))
                        ed = ImageDraw.Draw(img)
                        ef = _get_font(max(18, min(w, h) // 22))
                        ed.text(
                            (w // 2, h // 2), f"Error\n{e}",
                            fill=(255, 100, 100), font=ef, anchor="mm",
                        )

                    cols.append(img)
                    done += 1

                rows.append(cols)

            # Build labels
            x_labels = [str(v) for v in x_vals]
            y_labels = [
                f"{y_type}: {v}" if y_type != "None" else ""
                for v in y_vals
            ]

            grid = _make_grid(rows, x_labels, y_labels)

            # If Z is active, prepend a Z header strip
            if z_type != "None":
                zf_size = max(22, grid.width // 45)
                strip_h = zf_size + 20
                final = Image.new(
                    "RGB", (grid.width, grid.height + strip_h), (17, 17, 19),
                )
                final.paste(grid, (0, strip_h))
                zd = ImageDraw.Draw(final)
                zf = _get_font(zf_size)
                zd.text(
                    (final.width // 2, strip_h // 2),
                    f"{z_type}: {z_val}",
                    font=zf, fill=(232, 230, 227), anchor="mm",
                )
                grid = final

            grid_images.append(grid)

        if not seed_is_axis:
            info_parts.append(f"seed {base_kwargs['seed']}")
        elapsed = time.perf_counter() - t_start
        n_xy = len(x_vals) * len(y_vals)
        info_parts.append(
            f"XYZ grid: {len(x_vals)}×{len(y_vals)}×{len(z_vals)} "
            f"= {total_cells} images "
            f"({elapsed:.1f}s, ~{elapsed / total_cells:.1f}s/img)"
        )
        return grid_images, " | ".join(info_parts)

    finally:
        if last_loras is not None:
            ENGINE.clear_temp_loras()
