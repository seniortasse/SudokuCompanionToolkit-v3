from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


DEFAULT_CONFIG = Path(
    "datasets/sudoku_books/classic9/marketing_specs/"
    "BK-CL9-DW-B01.marketing.phase_c2_banner_01_overlay.json"
)


# -----------------------------------------------------------------------------
# Basic IO
# -----------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _sha12(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Color / fonts
# -----------------------------------------------------------------------------


def _hex_to_rgba(value: str, alpha: int = 255) -> Tuple[int, int, int, int]:
    value = str(value or "#FFFFFF").strip().lstrip("#")
    if len(value) != 6:
        return (255, 255, 255, alpha)
    return (
        int(value[0:2], 16),
        int(value[2:4], 16),
        int(value[4:6], 16),
        alpha,
    )


def _require_pillow() -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Pillow is required. Install with: python -m pip install pillow"
        ) from exc


def _font_from_candidates(candidates: Sequence[str], size: int):
    from PIL import ImageFont

    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)

    return ImageFont.load_default()


def _font(
    config: Dict[str, Any],
    size: int,
    *,
    bold: bool,
    role: Optional[str] = None,
):
    font_candidates = config.get("font_candidates") or {}

    candidates: List[str] = []

    if role:
        role_candidates = font_candidates.get(role) or []
        candidates.extend(role_candidates)

    key = "bold" if bold else "regular"
    candidates.extend(font_candidates.get(key) or [])

    fallback = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    return _font_from_candidates(list(candidates) + fallback, size)


def _fit_font_to_width(
    draw: Any,
    text: str,
    *,
    config: Dict[str, Any],
    start_size: int,
    max_width: int,
    bold: bool,
    min_size: int = 12,
    role: Optional[str] = None,
) -> Any:
    """
    Returns the largest font <= start_size whose widest line fits max_width.
    Handles explicit newline-separated text.
    """
    lines = str(text).split("\n")
    for size in range(int(start_size), int(min_size) - 1, -1):
        font = _font(config, size, bold=bold, role=role)
        ok = True
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            if (bbox[2] - bbox[0]) > max_width:
                ok = False
                break
        if ok:
            return font

    return _font(config, min_size, bold=bold, role=role)


# -----------------------------------------------------------------------------
# Image helpers
# -----------------------------------------------------------------------------


def _load_rgba(path: Path):
    from PIL import Image

    if not path.exists():
        raise FileNotFoundError(str(path))
    return Image.open(path).convert("RGBA")


def _fit_cover(src: Any, size: Tuple[int, int]):
    from PIL import Image

    target_w, target_h = size
    src_w, src_h = src.size
    scale = max(target_w / src_w, target_h / src_h)
    resized = src.resize((int(src_w * scale), int(src_h * scale)), Image.LANCZOS)

    x = (resized.width - target_w) // 2
    y = (resized.height - target_h) // 2
    return resized.crop((x, y, x + target_w, y + target_h))


def _fit_contain(
    src: Any,
    size: Tuple[int, int],
    bg: Tuple[int, int, int, int] = (255, 255, 255, 0),
):
    from PIL import Image

    target_w, target_h = size
    copy = src.copy()
    copy.thumbnail((target_w, target_h), Image.LANCZOS)

    canvas = Image.new("RGBA", (target_w, target_h), bg)
    x = (target_w - copy.width) // 2
    y = (target_h - copy.height) // 2
    canvas.alpha_composite(copy, (x, y))
    return canvas


def _round_mask(size: Tuple[int, int], radius: int):
    from PIL import Image, ImageDraw

    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=radius, fill=255)
    return mask


def _rounded_image(im: Any, radius: int):
    from PIL import Image

    im = im.convert("RGBA")
    mask = _round_mask(im.size, radius)
    out = Image.new("RGBA", im.size, (0, 0, 0, 0))
    out.alpha_composite(im)
    out.putalpha(mask)
    return out


def _trim_transparent_or_light_border(im: Any) -> Any:
    """
    Trims transparent or very-light border around icon assets.
    """
    from PIL import Image, ImageChops

    rgba = im.convert("RGBA")

    alpha = rgba.getchannel("A")
    bbox = alpha.getbbox()
    if bbox:
        cropped = rgba.crop(bbox)
    else:
        cropped = rgba

    bg = Image.new("RGBA", cropped.size, (255, 255, 255, 255))
    diff = ImageChops.difference(cropped, bg)
    bbox2 = diff.getbbox()
    if bbox2:
        cropped = cropped.crop(bbox2)

    return cropped


# -----------------------------------------------------------------------------
# Text helpers
# -----------------------------------------------------------------------------


def _cfg_first(
    cfg: Dict[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:
    for key in keys:
        if key in cfg and cfg[key] is not None:
            return cfg[key]
    return default


def _resolve_text_controls(
    cfg: Dict[str, Any],
    *,
    prefix: Optional[str] = None,
    default_align: str = "center",
    default_valign: str = "center",
) -> Dict[str, Any]:
    if prefix:
        align = _cfg_first(cfg, f"{prefix}_align", "align", default=default_align)
        valign = _cfg_first(cfg, f"{prefix}_valign", "valign", default=default_valign)

        padding_left = _cfg_first(
            cfg, f"{prefix}_padding_left", "padding_left", default=0
        )
        padding_right = _cfg_first(
            cfg, f"{prefix}_padding_right", "padding_right", default=0
        )
        padding_top = _cfg_first(
            cfg, f"{prefix}_padding_top", "padding_top", default=0
        )
        padding_bottom = _cfg_first(
            cfg, f"{prefix}_padding_bottom", "padding_bottom", default=0
        )
    else:
        align = _cfg_first(cfg, "align", default=default_align)
        valign = _cfg_first(cfg, "valign", default=default_valign)
        padding_left = _cfg_first(cfg, "padding_left", default=0)
        padding_right = _cfg_first(cfg, "padding_right", default=0)
        padding_top = _cfg_first(cfg, "padding_top", default=0)
        padding_bottom = _cfg_first(cfg, "padding_bottom", default=0)

    return {
        "align": str(align or default_align).strip().lower(),
        "valign": str(valign or default_valign).strip().lower(),
        "padding_left": int(padding_left or 0),
        "padding_right": int(padding_right or 0),
        "padding_top": int(padding_top or 0),
        "padding_bottom": int(padding_bottom or 0),
    }


def _resolve_box(
    cfg: Dict[str, Any],
    *,
    fallback_h: Optional[int] = None,
) -> List[int]:
    """
    Supports either:
      - "box": [x, y, w, h]
    or legacy:
      - x, y, w, optional h
    """
    if "box" in cfg and cfg["box"] is not None:
        box = cfg["box"]
        return [int(box[0]), int(box[1]), int(box[2]), int(box[3])]

    x = int(cfg.get("x", 0))
    y = int(cfg.get("y", 0))
    w = int(cfg.get("w", 0))
    h = int(cfg.get("h", fallback_h if fallback_h is not None else 24))
    return [x, y, w, h]


def _draw_text_with_shadow(
    draw: Any,
    xy: Tuple[int, int],
    text: str,
    *,
    font: Any,
    fill: Tuple[int, int, int, int],
    shadow: bool = False,
    shadow_fill: Tuple[int, int, int, int] = (0, 18, 45, 135),
    shadow_offset: Tuple[int, int] = (2, 2),
) -> None:
    x, y = xy
    if shadow:
        draw.text(
            (x + shadow_offset[0], y + shadow_offset[1]),
            text,
            font=font,
            fill=shadow_fill,
        )
    draw.text((x, y), text, font=font, fill=fill)


def _wrap_text(draw: Any, text: str, font: Any, max_width: int) -> List[str]:
    words = str(text).split()
    lines: List[str] = []
    current = ""

    for word in words:
        trial = word if not current else current + " " + word
        bbox = draw.textbbox((0, 0), trial, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word

    if current:
        lines.append(current)

    return lines


def _line_metrics(
    draw: Any,
    lines: Sequence[str],
    font: Any,
    line_gap: int,
) -> Tuple[int, List[Tuple[str, int, int]]]:
    metrics: List[Tuple[str, int, int]] = []
    total_h = 0

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        metrics.append((line, w, h))
        total_h += h

    total_h += line_gap * max(0, len(lines) - 1)
    return total_h, metrics


def _draw_text_in_box(
    draw: Any,
    box: Sequence[int],
    text: str,
    *,
    font: Any,
    fill: Tuple[int, int, int, int],
    line_gap: int = 2,
    shadow: bool = False,
    align: str = "center",
    valign: str = "center",
    padding_left: int = 0,
    padding_right: int = 0,
    padding_top: int = 0,
    padding_bottom: int = 0,
) -> None:
    """
    Draw text inside a [x, y, w, h] box.

    Horizontal align:
      - left
      - center
      - right

    Vertical align:
      - top
      - center
      - bottom
    """
    x, y, w, h = [int(v) for v in box]
    lines = str(text).split("\n")

    align = str(align or "center").strip().lower()
    valign = str(valign or "center").strip().lower()

    content_x = x + int(padding_left)
    content_y = y + int(padding_top)
    content_w = max(1, w - int(padding_left) - int(padding_right))
    content_h = max(1, h - int(padding_top) - int(padding_bottom))

    total_h, metrics = _line_metrics(draw, lines, font, line_gap)

    if valign == "top":
        yy = content_y
    elif valign == "bottom":
        yy = content_y + (content_h - total_h)
    else:
        yy = content_y + ((content_h - total_h) // 2)

    for line, tw, th in metrics:
        if align == "left":
            xx = content_x
        elif align == "right":
            xx = content_x + (content_w - tw)
        else:
            xx = content_x + ((content_w - tw) // 2)

        _draw_text_with_shadow(
            draw,
            (xx, yy),
            line,
            font=font,
            fill=fill,
            shadow=shadow,
        )
        yy += th + line_gap


def _draw_fitted_text_block(
    draw: Any,
    config: Dict[str, Any],
    *,
    text: str,
    box: Sequence[int],
    font_size: int,
    min_size: int,
    bold: bool,
    fill: Tuple[int, int, int, int],
    shadow: bool,
    line_gap: int = 2,
    role: Optional[str] = None,
    align: str = "center",
    valign: str = "center",
    padding_left: int = 0,
    padding_right: int = 0,
    padding_top: int = 0,
    padding_bottom: int = 0,
) -> Any:
    inner_w = max(1, int(box[2]) - int(padding_left) - int(padding_right))

    font = _fit_font_to_width(
        draw,
        text,
        config=config,
        start_size=int(font_size),
        max_width=inner_w,
        bold=bold,
        min_size=int(min_size),
        role=role,
    )

    _draw_text_in_box(
        draw,
        box,
        text,
        font=font,
        fill=fill,
        line_gap=line_gap,
        shadow=shadow,
        align=align,
        valign=valign,
        padding_left=padding_left,
        padding_right=padding_right,
        padding_top=padding_top,
        padding_bottom=padding_bottom,
    )

    return font


def _draw_multiline_headline(
    draw: Any,
    config: Dict[str, Any],
    *,
    box_cfg: Dict[str, Any],
    lines: Sequence[str],
) -> None:
    x = int(box_cfg["x"])
    y = int(box_cfg["y"])
    max_w = int(box_cfg.get("w") or 520)
    start_size = int(box_cfg["font_size"])
    min_size = int(box_cfg.get("min_font_size", 28))
    line_gap = int(box_cfg.get("line_gap", 4))
    fill = _hex_to_rgba(box_cfg["color"], 255)
    bold = bool(box_cfg.get("bold", True))
    shadow = bool(box_cfg.get("shadow", True))
    align = str(box_cfg.get("align", "left")).strip().lower()

    padding_left = int(box_cfg.get("padding_left", 0))
    padding_right = int(box_cfg.get("padding_right", 0))

    content_x = x + padding_left
    content_w = max(1, max_w - padding_left - padding_right)

    all_text = "\n".join(lines)
    font = _fit_font_to_width(
        draw,
        all_text,
        config=config,
        start_size=start_size,
        max_width=content_w,
        bold=bold,
        min_size=min_size,
        role=str(box_cfg.get("font_role") or "headline"),
    )

    yy = y
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_w = bbox[2] - bbox[0]
        line_h = bbox[3] - bbox[1]

        if align == "center":
            xx = content_x + ((content_w - line_w) // 2)
        elif align == "right":
            xx = content_x + (content_w - line_w)
        else:
            xx = content_x

        _draw_text_with_shadow(
            draw,
            (xx, yy),
            line,
            font=font,
            fill=fill,
            shadow=shadow,
        )
        yy += line_h + line_gap


# -----------------------------------------------------------------------------
# Icon helpers
# -----------------------------------------------------------------------------


def _remove_light_background_and_recolor(
    src: Any,
    *,
    color: Tuple[int, int, int, int],
) -> Any:
    """
    Converts generated icon PNGs into recolorable RGBA icons.
    """
    from PIL import Image

    src = src.convert("RGBA")
    out = Image.new("RGBA", src.size, (0, 0, 0, 0))

    src_px = src.load()
    out_px = out.load()

    for yy in range(src.height):
        for xx in range(src.width):
            r, g, b, a = src_px[xx, yy]
            if a < 8:
                continue

            brightness = (r + g + b) / 3.0
            saturation_blue = b - max(r, g)

            keep = brightness < 205 and (brightness < 175 or saturation_blue > 8)

            if keep:
                edge_alpha = int(max(40, min(255, (220 - brightness) / 185 * 255)))
                final_alpha = int(edge_alpha * (a / 255.0))
                out_px[xx, yy] = (color[0], color[1], color[2], final_alpha)

    return out


def _make_icon_rgba(
    icon_path: Path,
    *,
    color: Tuple[int, int, int, int],
    recolor: bool = True,
):
    from PIL import Image

    if not icon_path.exists():
        raise FileNotFoundError(str(icon_path))

    src = Image.open(icon_path).convert("RGBA")

    if recolor:
        icon = _remove_light_background_and_recolor(src, color=color)
    else:
        icon = src

    icon = _trim_transparent_or_light_border(icon)
    return icon


def _paste_icon(
    base: Any,
    icon_path: Path,
    box: Sequence[int],
    *,
    color: Tuple[int, int, int, int],
    recolor: bool = True,
) -> None:
    x, y, w, h = [int(v) for v in box]
    icon = _make_icon_rgba(icon_path, color=color, recolor=recolor)
    icon = _fit_contain(icon, (w, h), bg=(255, 255, 255, 0))
    base.alpha_composite(icon, (x, y))


# -----------------------------------------------------------------------------
# Card-image helpers
# -----------------------------------------------------------------------------


def _paste_card_image(
    base: Any,
    image_path: Path,
    box: Sequence[int],
    *,
    mode: str = "cover",
    radius: int = 4,
) -> None:
    x, y, w, h = [int(v) for v in box]
    src = _load_rgba(image_path)

    if mode == "contain":
        fitted = _fit_contain(src, (w, h), bg=(255, 255, 255, 255))
    else:
        fitted = _fit_cover(src, (w, h))

    fitted = _rounded_image(fitted, radius)
    base.alpha_composite(fitted, (x, y))


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------


def _resolve_required_files(config: Dict[str, Any]) -> List[Path]:
    assets_dir = Path(config["assets_dir"])
    icons_dir = assets_dir / "icons"

    required = [
        assets_dir / config["background"],
        assets_dir / "hero_puzzle_page.png",
        assets_dir / "hero_pattern_page.png",
        assets_dir / "hero_features_page.png",
    ]

    for item in config["copy"]["feature_pills"]:
        required.append(icons_dir / item["icon"])

    for item in config["copy"]["content_cards"]:
        required.append(icons_dir / item["icon"])
        required.append(assets_dir / item["image"])

    seen = set()
    out: List[Path] = []
    for path in required:
        key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)

    return out


def render_banner(config: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    _require_pillow()

    from PIL import ImageDraw

    assets_dir = Path(config["assets_dir"])
    icons_dir = assets_dir / "icons"
    bg_path = assets_dir / config["background"]

    export = config["exports"]["main"]
    width = int(export["width_px"])
    height = int(export["height_px"])

    background = _load_rgba(bg_path)
    background = _fit_cover(background, (width, height))

    img = background.copy()
    draw = ImageDraw.Draw(img)

    colors = config["colors"]
    layout = config["layout"]
    copy = config["copy"]

    # -------------------------------------------------------------------------
    # 1) Top header pill text.
    # -------------------------------------------------------------------------
    header_cfg = layout["header_pill_text"]
    header_box = _resolve_box(
        header_cfg,
        fallback_h=int(header_cfg.get("font_size", 18)) + 8,
    )
    header_controls = _resolve_text_controls(
        header_cfg,
        default_align="center",
        default_valign="center",
    )

    _draw_fitted_text_block(
        draw,
        config,
        text=copy["header_pill"],
        box=header_box,
        font_size=int(header_cfg["font_size"]),
        min_size=int(header_cfg.get("min_font_size", 10)),
        bold=bool(header_cfg.get("bold", True)),
        fill=_hex_to_rgba(header_cfg["color"], 255),
        shadow=bool(header_cfg.get("shadow", False)),
        line_gap=int(header_cfg.get("line_gap", 1)),
        role=str(header_cfg.get("font_role") or "header_pill"),
        align=header_controls["align"],
        valign=header_controls["valign"],
        padding_left=header_controls["padding_left"],
        padding_right=header_controls["padding_right"],
        padding_top=header_controls["padding_top"],
        padding_bottom=header_controls["padding_bottom"],
    )

    # -------------------------------------------------------------------------
    # 2) Main headline.
    # -------------------------------------------------------------------------
    _draw_multiline_headline(
        draw,
        config,
        box_cfg=layout["headline"],
        lines=copy["headline_lines"],
    )

    # -------------------------------------------------------------------------
    # 3) Subtitle.
    # -------------------------------------------------------------------------
    subtitle_cfg = layout["subtitle"]
    subtitle_box = _resolve_box(
        subtitle_cfg,
        fallback_h=int(subtitle_cfg.get("font_size", 18)) + 8,
    )
    subtitle_controls = _resolve_text_controls(
        subtitle_cfg,
        default_align="left",
        default_valign="center",
    )

    _draw_fitted_text_block(
        draw,
        config,
        text=copy["subtitle"],
        box=subtitle_box,
        font_size=int(subtitle_cfg["font_size"]),
        min_size=int(subtitle_cfg.get("min_font_size", 12)),
        bold=bool(subtitle_cfg.get("bold", True)),
        fill=_hex_to_rgba(subtitle_cfg["color"], 255),
        shadow=bool(subtitle_cfg.get("shadow", True)),
        line_gap=int(subtitle_cfg.get("line_gap", 1)),
        role=str(subtitle_cfg.get("font_role") or "subtitle"),
        align=subtitle_controls["align"],
        valign=subtitle_controls["valign"],
        padding_left=subtitle_controls["padding_left"],
        padding_right=subtitle_controls["padding_right"],
        padding_top=subtitle_controls["padding_top"],
        padding_bottom=subtitle_controls["padding_bottom"],
    )

    # -------------------------------------------------------------------------
    # 4) Support line.
    # -------------------------------------------------------------------------
    support_cfg = layout["support_line"]
    support_box = _resolve_box(
        support_cfg,
        fallback_h=int(support_cfg.get("font_size", 16)) + 8,
    )
    support_controls = _resolve_text_controls(
        support_cfg,
        default_align="left",
        default_valign="center",
    )

    _draw_fitted_text_block(
        draw,
        config,
        text=copy["support_line"],
        box=support_box,
        font_size=int(support_cfg["font_size"]),
        min_size=int(support_cfg.get("min_font_size", 11)),
        bold=bool(support_cfg.get("bold", True)),
        fill=_hex_to_rgba(support_cfg["color"], 255),
        shadow=bool(support_cfg.get("shadow", True)),
        line_gap=int(support_cfg.get("line_gap", 1)),
        role=str(support_cfg.get("font_role") or "support_line"),
        align=support_controls["align"],
        valign=support_controls["valign"],
        padding_left=support_controls["padding_left"],
        padding_right=support_controls["padding_right"],
        padding_top=support_controls["padding_top"],
        padding_bottom=support_controls["padding_bottom"],
    )

    # -------------------------------------------------------------------------
    # 5) Four feature pills: icons + labels.
    # -------------------------------------------------------------------------
    for item, pill_cfg in zip(copy["feature_pills"], layout["feature_pills"]):
        _paste_icon(
            img,
            icons_dir / item["icon"],
            pill_cfg["icon_box"],
            color=_hex_to_rgba(item.get("icon_color") or colors["navy"], 255),
            recolor=bool(item.get("recolor", True)),
        )

        text_box = pill_cfg["text_box"]
        feature_font_role = str(pill_cfg.get("font_role") or "feature_pill")
        feature_line_gap = int(pill_cfg.get("line_gap", 1))
        feature_bold = bool(pill_cfg.get("bold", True))
        feature_controls = _resolve_text_controls(
            pill_cfg,
            prefix="text",
            default_align="center",
            default_valign="center",
        )

        _draw_fitted_text_block(
            draw,
            config,
            text=item["label"],
            box=text_box,
            font_size=int(pill_cfg["font_size"]),
            min_size=int(pill_cfg.get("min_font_size", 10)),
            bold=feature_bold,
            fill=_hex_to_rgba(pill_cfg.get("color") or colors["navy"], 255),
            shadow=bool(pill_cfg.get("shadow", False)),
            line_gap=feature_line_gap,
            role=feature_font_role,
            align=feature_controls["align"],
            valign=feature_controls["valign"],
            padding_left=feature_controls["padding_left"],
            padding_right=feature_controls["padding_right"],
            padding_top=feature_controls["padding_top"],
            padding_bottom=feature_controls["padding_bottom"],
        )

    # -------------------------------------------------------------------------
    # 6) Three content card images.
    # -------------------------------------------------------------------------
    for item, card_cfg in zip(copy["content_cards"], layout["content_cards"]):
        mode = item.get("image_mode")
        if not mode:
            mode = "contain" if "features" in item["image"] else "cover"

        _paste_card_image(
            img,
            assets_dir / item["image"],
            card_cfg["image_box"],
            mode=mode,
            radius=int(card_cfg.get("image_radius", 4)),
        )

    # -------------------------------------------------------------------------
    # 7) Footer bars: icons + labels.
    # -------------------------------------------------------------------------
    for item, card_cfg in zip(copy["content_cards"], layout["content_cards"]):
        _paste_icon(
            img,
            icons_dir / item["icon"],
            card_cfg["footer_icon_box"],
            color=_hex_to_rgba(colors["white"], 255),
            recolor=bool(item.get("recolor_icon", True)),
        )

        text_box = card_cfg["footer_text_box"]
        footer_font_role = str(card_cfg.get("footer_font_role") or "footer_label")
        footer_line_gap = int(card_cfg.get("footer_line_gap", 1))
        footer_bold = bool(card_cfg.get("footer_bold", True))
        footer_controls = _resolve_text_controls(
            card_cfg,
            prefix="footer_text",
            default_align="center",
            default_valign="center",
        )

        _draw_fitted_text_block(
            draw,
            config,
            text=item["label"],
            box=text_box,
            font_size=int(card_cfg["footer_font_size"]),
            min_size=int(card_cfg.get("footer_min_font_size", 10)),
            bold=footer_bold,
            fill=_hex_to_rgba(card_cfg.get("footer_color") or colors["white"], 255),
            shadow=bool(card_cfg.get("footer_shadow", False)),
            line_gap=footer_line_gap,
            role=footer_font_role,
            align=footer_controls["align"],
            valign=footer_controls["valign"],
            padding_left=footer_controls["padding_left"],
            padding_right=footer_controls["padding_right"],
            padding_top=footer_controls["padding_top"],
            padding_bottom=footer_controls["padding_bottom"],
        )

    report = {
        "width_px": width,
        "height_px": height,
        "background": str(bg_path),
        "assets_dir": str(assets_dir),
        "icons_dir": str(icons_dir),
        "content_images": [item["image"] for item in copy["content_cards"]],
        "feature_icons": [item["icon"] for item in copy["feature_pills"]],
        "footer_icons": [item["icon"] for item in copy["content_cards"]],
    }

    return img.convert("RGB"), report


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def run(args: argparse.Namespace) -> Dict[str, Any]:
    config_path = Path(args.config)
    config = _read_json(config_path)

    out_dir = Path(args.out_dir or config["output_dir"])
    reports_dir = out_dir / "_reports"

    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)

    _ensure_dir(out_dir)
    _ensure_dir(reports_dir)

    if args.dry_run:
        assets_dir = Path(config["assets_dir"])
        icons_dir = assets_dir / "icons"
        bg_path = assets_dir / config["background"]

        required_files = _resolve_required_files(config)
        missing = [str(p) for p in required_files if not p.exists()]

        report = {
            "ok": len(missing) == 0,
            "dry_run": True,
            "phase": config.get("phase", "C2"),
            "banner_id": config["banner_id"],
            "config_path": str(config_path),
            "assets_dir": str(assets_dir),
            "icons_dir": str(icons_dir),
            "background": str(bg_path),
            "output_dir": str(out_dir),
            "expected_output": str(out_dir / config["exports"]["main"]["filename"]),
            "required_files_checked": [str(p) for p in required_files],
            "missing_files": missing,
        }

        _write_json(
            reports_dir / "phase_c2_banner_01_overlay_dry_run_report.json",
            report,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report

    img, render_report = render_banner(config)

    out_path = out_dir / config["exports"]["main"]["filename"]
    img.save(out_path)

    shutil.copy2(config_path, reports_dir / config_path.name)

    report = {
        "ok": True,
        "phase": config.get("phase", "C2"),
        "banner_id": config["banner_id"],
        "banner_type": config["banner_type"],
        "book_id": config["book_id"],
        "campaign_id": config["campaign_id"],
        "config_path": str(config_path),
        "output_dir": str(out_dir),
        "outputs": {
            "main": {
                "path": str(out_path),
                "width_px": img.width,
                "height_px": img.height,
                "sha12": _sha12(out_path),
            }
        },
        "render": render_report,
    }

    _write_json(reports_dir / "phase_c2_banner_01_overlay_report.json", report)

    print(json.dumps(report["outputs"], ensure_ascii=False, indent=2))
    print(f"[OK] Phase C2 Banner 1 overlay written to: {out_path}")

    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render Banner 1 by overlaying text, icons, and card images "
            "onto a locked background."
        )
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Phase C2 Banner 1 overlay config JSON.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Optional output directory. Defaults to config output_dir.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing output directory before rendering.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve paths without rendering.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        run(args)
        return 0
    except Exception as exc:
        print(f"[ERROR] Banner 1 overlay render failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())