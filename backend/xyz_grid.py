"""X/Y/Z plot grids: compare parameter combinations side by side."""

from __future__ import annotations

import math
import random
import time
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageFont

from engine import ENGINE

# ── public constants ──────────────────────────────────────────────

PARAM_TYPES = ["None", "Seed", "Sampler", "Scheduler", "Steps", "CFG Scale", "Prompt S/R", "Checkpoint"]

# Cap on |x|·|y|·|z|: "50,50,50" would be 125k generations.
MAX_XYZ_CELLS = 1000

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
            # Pillow >= 10.1 scales the default font; older ignores the size.
            try:
                _FONT_CACHE[size] = ImageFont.load_default(size)
            except TypeError:
                _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


def _wrap_lines(text: str, font, max_w: int) -> list[str]:
    """Break *text* into lines no wider than *max_w* pixels, preferring breaks
    after " -_+." and hard-breaking only runs wider than the line."""
    text = str(text)

    def w(s: str) -> int:
        b = font.getbbox(s)
        return b[2] - b[0]

    if max_w <= 0 or w(text) <= max_w:
        return [text]

    tokens, buf = [], ""
    for ch in text:
        buf += ch
        if ch in " -_+.":
            tokens.append(buf)
            buf = ""
    if buf:
        tokens.append(buf)

    lines, cur = [], ""
    for tok in tokens:
        # Hard-break a token wider than the line.
        while w(tok) > max_w and len(tok) > 1:
            cut = len(tok)
            while cut > 1 and w(tok[:cut]) > max_w:
                cut -= 1
            if cur:
                lines.append(cur.rstrip())
                cur = ""
            lines.append(tok[:cut])
            tok = tok[cut:]
        if cur and w(cur + tok) > max_w:
            lines.append(cur.rstrip())
            cur = tok
        else:
            cur += tok
    if cur:
        lines.append(cur.rstrip())
    return lines or [text]


# ── value parsing ─────────────────────────────────────────────────

def resolve_values(param_type: str, values_str: str, base_value: Any) -> list:
    """Parse a comma-separated values string into a typed list, or
    ``[base_value]`` when empty. Malformed numbers raise ``ValueError`` naming
    the token."""
    if param_type == "None" or not values_str.strip():
        return [base_value]
    # Prompt S/R keeps empties: "promptA," means a cell with the term removed.
    if param_type == "Prompt S/R":
        vals = [v.strip() for v in values_str.split(",")]
        # An empty search token would splice the replacement between every char.
        if not vals[0]:
            raise ValueError(
                "Prompt S/R: the first value is the text to search for and "
                "cannot be empty")
        return vals
    raw = [v.strip() for v in values_str.split(",") if v.strip()]
    if not raw:
        return [base_value]
    if param_type in ("Seed", "Steps"):
        return [_parse_num(param_type, v, int) for v in raw]
    if param_type == "CFG Scale":
        return [_parse_num(param_type, v, float) for v in raw]
    return raw


def _parse_num(param_type: str, token: str, cast):
    """``cast(token)`` with an axis-labelled error, rejecting NaN/Inf."""
    try:
        value = cast(token)
    except ValueError:
        raise ValueError(
            f"{param_type} axis: {token!r} is not a valid "
            f"{'integer' if cast is int else 'number'}") from None
    if cast is float and not math.isfinite(value):
        raise ValueError(f"{param_type} axis: {token!r} is not a finite number")
    return value


# ── grid assembly ─────────────────────────────────────────────────

CELL_BORDER = 2


def _make_grid(
    images: list[list[Image.Image]],
    x_labels: list[str],
    y_labels: list[str],
) -> Image.Image:
    """Assemble a 2-D grid of PIL Images with labels scaled to the cell size."""
    n_rows = len(images)
    n_cols = len(images[0]) if n_rows else 0
    if n_rows == 0 or n_cols == 0:
        raise ValueError("Empty image grid")

    cell_w, cell_h = images[0][0].size

    font_size = max(20, min(cell_w, cell_h) // 22)
    font = _get_font(font_size)
    pad = max(10, font_size // 2)
    line_h = font_size + max(2, pad // 4)

    def _text_w(s: str) -> int:
        box = font.getbbox(str(s))
        return box[2] - box[0]

    # Wrap long labels (e.g. checkpoint filenames) to the available width.
    x_wrapped = [_wrap_lines(s, font, cell_w - 2 * pad) for s in x_labels]
    y_wrapped = [_wrap_lines(s, font, cell_w - 2 * pad) for s in y_labels]

    x_lines = max((len(w) for w in x_wrapped), default=1)
    header_h = x_lines * line_h + 2 * pad
    max_label = max((_text_w(ln) for w in y_wrapped for ln in w), default=0)
    label_w = max_label + 2 * pad if max_label else CELL_BORDER

    grid_w = label_w + n_cols * cell_w + (n_cols + 1) * CELL_BORDER
    grid_h = header_h + n_rows * cell_h + (n_rows + 1) * CELL_BORDER

    canvas = Image.new("RGB", (grid_w, grid_h), (17, 17, 19))
    draw = ImageDraw.Draw(canvas)

    def _draw_stack(cx: int, cy: int, lines: list[str], fill) -> None:
        top = cy - (len(lines) * line_h) // 2 + line_h // 2
        for i, ln in enumerate(lines):
            draw.text((cx, top + i * line_h), ln, font=font, fill=fill, anchor="mm")

    # Column headers (X): teal
    for xi in range(n_cols):
        cx = label_w + CELL_BORDER + xi * (cell_w + CELL_BORDER) + cell_w // 2
        _draw_stack(cx, header_h // 2, x_wrapped[xi], (93, 214, 192))

    # Row labels (Y): accent orange
    for yi in range(n_rows):
        cy = header_h + CELL_BORDER + yi * (cell_h + CELL_BORDER) + cell_h // 2
        _draw_stack(label_w // 2, cy, y_wrapped[yi], (232, 162, 101))

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
    progress_callback: Callable[..., None] | None = None,
    preview_callback: Callable[[Image.Image], None] | None = None,
    save_callback: Callable[[Image.Image, dict], None] | None = None,
    cell_knobs: Callable[[str, str], dict] | None = None,
) -> tuple[list[Image.Image], str]:
    """Generate one X/Y/Z grid image per Z value; returns ``(grids, info)``.

    ``base_kwargs`` needs prompt, negative_prompt, width, height, steps,
    cfg_scale, sampler, scheduler, seed and shift. ``progress_callback`` gets
    ``(step, total_steps, cell, total_cells)`` cumulative over the grid;
    ``save_callback(image, kwargs)`` runs per successful cell; ``cell_knobs
    (sampler, scheduler)`` returns settings-panel kwargs merged into each cell.
    """
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
    if total_cells > MAX_XYZ_CELLS:
        raise ValueError(
            f"X/Y/Z grid is {total_cells} cells "
            f"({len(x_vals)}×{len(y_vals)}×{len(z_vals)}), over the "
            f"{MAX_XYZ_CELLS}-cell cap. Narrow an axis and try again.")
    done = 0

    # One progress bar for the whole grid. A Steps axis overrides the base
    # count (x→y→z, last wins, like the generation loop).
    base_steps = int(base_kwargs.get("steps", 0))

    def _cell_steps(xv, yv, zv) -> int:
        s = base_steps
        for a_type, a_val in ((x_type, xv), (y_type, yv), (z_type, zv)):
            if a_type == "Steps":
                s = int(a_val)
        return s

    total_steps = sum(
        _cell_steps(xv, yv, zv)
        for zv in z_vals for yv in y_vals for xv in x_vals
    )
    steps_done = 0

    # Prompt S/R and <lora:…> tags vary per cell, so the raw prompt (tags intact)
    # is the S/R target and LoRAs are re-fused per cell.
    raw_prompt = base_kwargs["prompt"]
    raw_neg = base_kwargs.get("negative_prompt", "")
    clean_prompt, base_p = ENGINE.parse_lora_prompt(raw_prompt)
    clean_neg, base_n = ENGINE.parse_lora_prompt(raw_neg)

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

        # One random base seed shared by every cell, unless Seed is an axis.
        seed_is_axis = "Seed" in (x_type, y_type, z_type)
        if not seed_is_axis and base_kwargs.get("seed", -1) == -1:
            base_kwargs["seed"] = random.randint(0, 2**32 - 1)

        grid_images: list[Image.Image] = []

        for zi, z_val in enumerate(z_vals):
            rows: list[list[Image.Image]] = []

            for yi, y_val in enumerate(y_vals):
                cols: list[Image.Image] = []

                for xi, x_val in enumerate(x_vals):
                    kwargs = dict(base_kwargs)
                    cell_prompt, cell_neg = raw_prompt, raw_neg
                    load_error: Exception | None = None
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
                        elif a_type == "Checkpoint":
                            # reload_model no-ops on the current name; a real
                            # reload drops the fused LoRAs, so re-fuse. A failed
                            # load becomes this cell's error, not the grid's.
                            try:
                                msg = ENGINE.reload_model(str(a_val))
                            except Exception as e:
                                load_error, msg = e, ""
                            if not msg.startswith("Model already loaded"):
                                last_loras = None
                        else:
                            kwargs[_PARAM_MAP[a_type]] = a_val

                    if cell_knobs is not None:
                        kwargs.update(cell_knobs(kwargs["sampler"], kwargs["scheduler"]))

                    # Re-fuse only when the cell's LoRA set changed.
                    cp, cp_loras = ENGINE.parse_lora_prompt(cell_prompt)
                    cn, cn_loras = ENGINE.parse_lora_prompt(cell_neg)
                    kwargs["prompt"], kwargs["negative_prompt"] = cp, cn
                    cell_loras = cp_loras + cn_loras
                    if cell_loras != last_loras:
                        ENGINE.apply_temp_loras(cell_loras)
                        last_loras = cell_loras

                    cur_steps = int(kwargs.get("steps", base_steps))
                    if progress_callback is not None:
                        progress_callback(steps_done, total_steps, done + 1, total_cells)
                        kwargs["progress_callback"] = (
                            lambda step, _t, base=steps_done, cell=done:
                                progress_callback(base + step, total_steps,
                                                  cell + 1, total_cells)
                        )
                    if preview_callback is not None:
                        kwargs["preview_callback"] = preview_callback

                    try:
                        if load_error is not None:
                            raise load_error
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
                    steps_done += cur_steps
                    done += 1

                rows.append(cols)

            x_labels = [str(v) for v in x_vals]
            y_labels = [
                f"{y_type}: {v}" if y_type != "None" else ""
                for v in y_vals
            ]

            grid = _make_grid(rows, x_labels, y_labels)

            if z_type != "None":
                zf_size = max(22, grid.width // 45)
                zf = _get_font(zf_size)
                z_line_h = zf_size + 8
                z_lines = _wrap_lines(
                    f"{z_type}: {z_val}", zf, grid.width - 2 * zf_size,
                )
                strip_h = len(z_lines) * z_line_h + 12
                final = Image.new(
                    "RGB", (grid.width, grid.height + strip_h), (17, 17, 19),
                )
                final.paste(grid, (0, strip_h))
                zd = ImageDraw.Draw(final)
                top = strip_h // 2 - (len(z_lines) * z_line_h) // 2 + z_line_h // 2
                for i, ln in enumerate(z_lines):
                    zd.text(
                        (final.width // 2, top + i * z_line_h),
                        ln, font=zf, fill=(232, 230, 227), anchor="mm",
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
