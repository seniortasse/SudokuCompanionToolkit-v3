from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


DEFAULT_CONFIG = Path(
    "datasets/sudoku_books/classic9/marketing_specs/"
    "BK-CL9-DW-B01.marketing.phase_c_banner_01_hero.json"
)


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


def _hex_to_rgb(value: str) -> Tuple[int, int, int]:
    value = str(value or "#FFFFFF").strip().lstrip("#")
    if len(value) != 6:
        return (255, 255, 255)
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


def _require_pillow() -> None:
    try:
        from PIL import Image, ImageDraw, ImageFilter, ImageFont  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Pillow is required for Phase C. Install with: python -m pip install pillow"
        ) from exc


def _font(size: int, bold: bool = False):
    from PIL import ImageFont

    candidates = [
        "C:/Windows/Fonts/montserrat/Montserrat-ExtraBold.ttf" if bold else "C:/Windows/Fonts/montserrat/Montserrat-Regular.ttf",
        "C:/Windows/Fonts/Montserrat-ExtraBold.ttf" if bold else "C:/Windows/Fonts/Montserrat-Regular.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)

    return ImageFont.load_default()


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


def _draw_wrapped_text(
    draw: Any,
    xy: Tuple[int, int],
    text: str,
    *,
    font: Any,
    fill: Tuple[int, int, int],
    max_width: int,
    line_gap: int = 6,
) -> int:
    x, y = xy
    for line in _wrap_text(draw, text, font, max_width):
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((0, 0), line, font=font)
        y += bbox[3] - bbox[1] + line_gap
    return y


def _fit_crop(src: Any, size: Tuple[int, int]):
    from PIL import Image

    target_w, target_h = size
    src_w, src_h = src.size
    scale = max(target_w / src_w, target_h / src_h)
    resized = src.resize((int(src_w * scale), int(src_h * scale)), Image.LANCZOS)

    x = (resized.width - target_w) // 2
    y = (resized.height - target_h) // 2
    return resized.crop((x, y, x + target_w, y + target_h))


def _fit_contain(src: Any, size: Tuple[int, int], bg: Tuple[int, int, int] = (255, 255, 255)):
    from PIL import Image

    target_w, target_h = size
    copy = src.copy()
    copy.thumbnail((target_w, target_h), Image.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), bg)
    x = (target_w - copy.width) // 2
    y = (target_h - copy.height) // 2
    canvas.paste(copy, (x, y))
    return canvas


def _mask_rounded(im: Any, radius: int):
    from PIL import Image, ImageDraw

    mask = Image.new("L", im.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, im.width, im.height), radius=radius, fill=255)
    out = Image.new("RGBA", im.size, (0, 0, 0, 0))
    out.paste(im.convert("RGBA"), (0, 0), mask)
    return out


def _load_image(path: Path, fallback_size: Tuple[int, int]):
    from PIL import Image, ImageDraw

    if path.exists():
        return Image.open(path).convert("RGB")

    img = Image.new("RGB", fallback_size, (245, 248, 252))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, fallback_size[0] - 1, fallback_size[1] - 1], outline=(180, 190, 205), width=2)
    draw.text((20, 20), f"Missing: {path.name}", font=_font(16, bold=True), fill=(90, 100, 115))
    return img


def _draw_shadowed_rounded_panel(
    base: Any,
    box: Tuple[int, int, int, int],
    *,
    radius: int,
    fill: Tuple[int, int, int],
    outline: Optional[Tuple[int, int, int]] = None,
    shadow_alpha: int = 52,
    shadow_offset: int = 8,
    shadow_blur: int = 18,
) -> None:
    from PIL import Image, ImageDraw, ImageFilter

    x0, y0, x1, y1 = box

    shadow = Image.new("RGBA", base.size, (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shadow)
    sdraw.rounded_rectangle(
        (x0 + shadow_offset, y0 + shadow_offset, x1 + shadow_offset, y1 + shadow_offset),
        radius=radius,
        fill=(0, 15, 35, shadow_alpha),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(shadow_blur))
    base.alpha_composite(shadow)

    draw = ImageDraw.Draw(base)
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline)


def _linear_gradient(
    width: int,
    height: int,
    top: Tuple[int, int, int],
    bottom: Tuple[int, int, int],
):
    from PIL import Image

    img = Image.new("RGB", (width, height), top)
    pixels = img.load()
    for y in range(height):
        t = y / max(1, height - 1)
        rgb = tuple(int(top[i] * (1 - t) + bottom[i] * t) for i in range(3))
        for x in range(width):
            pixels[x, y] = rgb
    return img


def _draw_technical_grid(draw: Any, width: int, height: int, color: Tuple[int, int, int]) -> None:
    # Subtle technical grid echoing the cover background.
    step = 34
    for x in range(-step, width + step, step):
        draw.line((x, 0, x + int(height * 0.25), height), fill=color, width=1)
    for y in range(0, height, step):
        draw.line((0, y, width, y), fill=color, width=1)


def _draw_curved_separator_echo(draw: Any, width: int, height: int, color: Tuple[int, int, int]) -> None:
    # Wide arcs inspired by the book cover separators.
    for offset, alpha_width in [(0, 3), (34, 2)]:
        points = []
        for x in range(-20, width + 21, 8):
            y = int(452 + offset + 28 * math.sin((x / width) * math.pi))
            points.append((x, y))
        if len(points) >= 2:
            draw.line(points, fill=color, width=alpha_width)


def _draw_header(config: Dict[str, Any], tokens: Dict[str, Any]) -> Any:
    _require_pillow()
    from PIL import Image, ImageDraw

    export = config["exports"]["header"]
    width = int(export["width_px"])
    height = int(export["height_px"])
    colors = {k: _hex_to_rgb(v) for k, v in tokens["colors"].items()}

    img = Image.new("RGBA", (width, height), colors["paper"] + (255,))
    draw = ImageDraw.Draw(img)

    # Left navy block.
    draw.rounded_rectangle((-18, 10, width + 18, height + 26), radius=28, fill=colors["navy_950"])

    # Gold rule.
    draw.line((42, height - 10, width - 42, height - 10), fill=colors["gold"], width=2)

    title = config["copy"]["header"]
    font = _font(34, bold=True)
    bbox = draw.textbbox((0, 0), title, font=font)
    x = (width - (bbox[2] - bbox[0])) // 2
    y = 19
    draw.text((x, y), title, font=font, fill=colors["paper"])

    return img.convert("RGB")


def _draw_badge(
    draw: Any,
    x: int,
    y: int,
    text: str,
    *,
    colors: Dict[str, Tuple[int, int, int]],
    fill: Tuple[int, int, int],
    text_fill: Tuple[int, int, int],
    outline: Optional[Tuple[int, int, int]] = None,
) -> int:
    font = _font(15, bold=True)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    w = text_w + 30
    h = 38
    draw.rounded_rectangle((x, y, x + w, y + h), radius=19, fill=fill, outline=outline, width=2 if outline else 1)
    draw.text((x + 15, y + 10), text, font=font, fill=text_fill)
    return w


def _draw_pattern_tile_accents(
    base: Any,
    pattern_dir: Path,
    *,
    colors: Dict[str, Tuple[int, int, int]],
) -> None:
    from PIL import ImageEnhance

    tiles = sorted(pattern_dir.glob("*.png"))[:6]
    positions = [
        (28, 32, 58),
        (345, 34, 46),
        (895, 58, 44),
        (865, 510, 54),
        (318, 528, 42),
        (42, 526, 48),
    ]

    for idx, path in enumerate(tiles):
        x, y, size = positions[idx]
        tile = _load_image(path, (size, size))
        tile = _fit_crop(tile, (size, size))
        tile = ImageEnhance.Contrast(tile).enhance(0.8)
        tile = ImageEnhance.Brightness(tile).enhance(1.15)
        tile_rgba = _mask_rounded(tile, 10)

        # Make it subtle.
        alpha = tile_rgba.getchannel("A").point(lambda p: int(p * 0.28))
        tile_rgba.putalpha(alpha)
        base.alpha_composite(tile_rgba, (x, y))


def _draw_cover_mockup(
    base: Any,
    cover_path: Path,
    *,
    box: Dict[str, Any],
    colors: Dict[str, Tuple[int, int, int]],
) -> None:
    from PIL import Image, ImageDraw, ImageFilter

    x = int(box["x"])
    y = int(box["y"])
    w = int(box["w"])
    h = int(box["h"])
    rot = float(box.get("rotation_degrees", -4))

    cover = _load_image(cover_path, (w, h))
    cover = _fit_contain(cover, (w, h), bg=colors["navy_950"])
    cover_rgba = _mask_rounded(cover, 18)

    # Add a thin white edge before rotation.
    framed = Image.new("RGBA", (w + 18, h + 18), (0, 0, 0, 0))
    fdraw = ImageDraw.Draw(framed)
    fdraw.rounded_rectangle((0, 0, w + 17, h + 17), radius=22, fill=(255, 255, 255, 255))
    framed.alpha_composite(cover_rgba, (9, 9))

    rotated = framed.rotate(rot, expand=True, resample=Image.Resampling.BICUBIC)

    shadow = Image.new("RGBA", rotated.size, (0, 0, 0, 0))
    shadow_alpha = rotated.getchannel("A").point(lambda p: int(p * 0.40))
    shadow.putalpha(shadow_alpha)
    shadow = shadow.filter(ImageFilter.GaussianBlur(16))

    base.alpha_composite(shadow, (x + 20, y + 24))
    base.alpha_composite(rotated, (x, y))


def _draw_page_card(
    base: Any,
    src_path: Path,
    *,
    x: int,
    y: int,
    w: int,
    h: int,
    label: str,
    colors: Dict[str, Tuple[int, int, int]],
    crop_mode: str = "crop",
) -> None:
    from PIL import ImageDraw

    _draw_shadowed_rounded_panel(
        base,
        (x, y, x + w, y + h),
        radius=18,
        fill=colors["paper"],
        outline=colors["line_soft"],
        shadow_alpha=44,
        shadow_offset=7,
        shadow_blur=13,
    )

    img = _load_image(src_path, (w - 20, h - 52))
    if crop_mode == "contain":
        fitted = _fit_contain(img, (w - 20, h - 58), bg=(255, 255, 255))
    else:
        fitted = _fit_crop(img, (w - 20, h - 58))

    fitted = _mask_rounded(fitted, 12)
    base.alpha_composite(fitted, (x + 10, y + 10))

    draw = ImageDraw.Draw(base)
    draw.rounded_rectangle((x + 12, y + h - 38, x + w - 12, y + h - 12), radius=13, fill=colors["navy_900"])
    draw.text((x + 24, y + h - 32), label, font=_font(13, bold=True), fill=colors["paper"])


def _draw_main(config: Dict[str, Any], tokens: Dict[str, Any]) -> Any:
    _require_pillow()
    from PIL import Image, ImageDraw, ImageFilter

    export = config["exports"]["main"]
    width = int(export["width_px"])
    height = int(export["height_px"])

    colors = {k: _hex_to_rgb(v) for k, v in tokens["colors"].items()}
    curated_dir = Path(config["curated_assets_dir"])

    bg = _linear_gradient(width, height, colors["navy_950"], colors["navy_800"]).convert("RGBA")
    draw = ImageDraw.Draw(bg, "RGBA")

    _draw_technical_grid(draw, width, height, (121, 201, 234, 32))
    _draw_curved_separator_echo(draw, width, height, (255, 255, 255, 70))

    if config["style"].get("use_pattern_tile_accents", True):
        _draw_pattern_tile_accents(
            bg,
            curated_dir / config["assets"]["pattern_tiles_dir"],
            colors=colors,
        )

    # Soft spotlight behind content.
    spotlight = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(spotlight, "RGBA")
    sdraw.ellipse((250, -150, 1100, 590), fill=(121, 201, 234, 42))
    spotlight = spotlight.filter(ImageFilter.GaussianBlur(55))
    bg.alpha_composite(spotlight)

    # Cover.
    _draw_cover_mockup(
        bg,
        curated_dir / config["assets"]["cover"],
        box=config["layout"]["cover_box"],
        colors=colors,
    )

    # Text.
    draw = ImageDraw.Draw(bg, "RGBA")
    text_box = config["layout"]["text_box"]
    tx = int(text_box["x"])
    ty = int(text_box["y"])
    tw = int(text_box["w"])

    # Kicker pill.
    kicker = config["copy"]["kicker"]
    kicker_font = _font(15, bold=True)
    kb = draw.textbbox((0, 0), kicker, font=kicker_font)
    kw = kb[2] - kb[0] + 30
    draw.rounded_rectangle((tx, ty, tx + kw, ty + 34), radius=17, fill=colors["cream"], outline=colors["gold"], width=2)
    draw.text((tx + 15, ty + 9), kicker, font=kicker_font, fill=colors["navy_950"])
    ty += 48

    headline = config["copy"]["headline"]
    ty = _draw_wrapped_text(
        draw,
        (tx, ty),
        headline,
        font=_font(35, bold=True),
        fill=colors["paper"],
        max_width=tw,
        line_gap=4,
    )

    ty += 8
    ty = _draw_wrapped_text(
        draw,
        (tx, ty),
        config["copy"]["subheadline"],
        font=_font(17, bold=False),
        fill=colors["blue_100"],
        max_width=tw,
        line_gap=4,
    )

    ty += 12
    draw.text((tx, ty), config["copy"]["support_line"], font=_font(15, bold=False), fill=colors["cream"])

    # Badges.
    badge_cfg = config["layout"]["badge_row"]
    bx = int(badge_cfg["x"])
    by = int(badge_cfg["y"])
    gap = int(badge_cfg["gap"])

    for idx, badge in enumerate(config["badges"]):
        if idx % 2 == 0:
            fill = colors["paper"]
            text_fill = colors["navy_950"]
            outline = colors["gold"]
        else:
            fill = colors["blue_100"]
            text_fill = colors["navy_950"]
            outline = colors["blue_300"]

        bw = _draw_badge(
            draw,
            bx,
            by,
            badge,
            colors=colors,
            fill=fill,
            text_fill=text_fill,
            outline=outline,
        )
        bx += bw + gap

    # Collage.
    collage = config["layout"]["collage"]
    cx = int(collage["x"])
    cy = int(collage["y"])
    card_w = int(collage["card_w"])
    card_h = int(collage["card_h"])
    card_gap = int(collage["gap"])

    _draw_page_card(
        bg,
        curated_dir / config["assets"]["puzzle_page"],
        x=cx,
        y=cy,
        w=card_w,
        h=card_h,
        label="Clean 6-up pages",
        colors=colors,
        crop_mode="crop",
    )
    _draw_page_card(
        bg,
        curated_dir / config["assets"]["pattern_page"],
        x=cx + card_w + card_gap,
        y=cy - 18,
        w=card_w,
        h=card_h,
        label="Pattern identity",
        colors=colors,
        crop_mode="crop",
    )
    _draw_page_card(
        bg,
        curated_dir / config["assets"]["features_page"],
        x=cx + 2 * (card_w + card_gap),
        y=cy,
        w=card_w,
        h=card_h,
        label="Help when stuck",
        colors=colors,
        crop_mode="crop",
    )

    # CTA.
    cta = config["copy"]["cta"]
    cta_x = 48
    cta_y = 548
    cta_font = _font(15, bold=True)
    cta_bbox = draw.textbbox((0, 0), cta, font=cta_font)
    cta_w = cta_bbox[2] - cta_bbox[0] + 36
    draw.rounded_rectangle((cta_x, cta_y, cta_x + cta_w, cta_y + 36), radius=18, fill=colors["gold"])
    draw.text((cta_x + 18, cta_y + 10), cta, font=cta_font, fill=colors["navy_950"])

    return bg.convert("RGB")


def _make_combined(header: Any, main: Any, config: Dict[str, Any]) -> Any:
    from PIL import Image

    width = int(config["exports"]["combined"]["width_px"])
    height = int(config["exports"]["combined"]["height_px"])

    combined = Image.new("RGB", (width, height), "white")
    combined.paste(header, (0, 0))
    combined.paste(main, (0, header.height))
    return combined


def run(args: argparse.Namespace) -> Dict[str, Any]:
    config_path = Path(args.config)
    config = _read_json(config_path)

    tokens_path = Path(args.design_tokens or config["design_tokens"])
    tokens = _read_json(tokens_path)

    out_dir = Path(args.out_dir or config["output_dir"])
    reports_dir = out_dir / "_reports"

    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)

    _ensure_dir(out_dir)
    _ensure_dir(reports_dir)

    if args.dry_run:
        report = {
            "ok": True,
            "dry_run": True,
            "phase": "C",
            "banner_id": config["banner_id"],
            "config_path": str(config_path),
            "design_tokens": str(tokens_path),
            "curated_assets_dir": config["curated_assets_dir"],
            "output_dir": str(out_dir),
            "expected_outputs": [
                str(out_dir / config["exports"]["header"]["filename"]),
                str(out_dir / config["exports"]["main"]["filename"]),
                str(out_dir / config["exports"]["combined"]["filename"])
            ]
        }
        _write_json(reports_dir / "phase_c_banner_01_dry_run_report.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report

    header = _draw_header(config, tokens)
    main = _draw_main(config, tokens)
    combined = _make_combined(header, main, config)

    header_path = out_dir / config["exports"]["header"]["filename"]
    main_path = out_dir / config["exports"]["main"]["filename"]
    combined_path = out_dir / config["exports"]["combined"]["filename"]

    header.save(header_path)
    main.save(main_path)
    combined.save(combined_path)

    shutil.copy2(config_path, reports_dir / config_path.name)

    report = {
        "ok": True,
        "phase": "C",
        "banner_id": config["banner_id"],
        "banner_type": config["banner_type"],
        "book_id": config["book_id"],
        "campaign_id": config["campaign_id"],
        "config_path": str(config_path),
        "design_tokens": str(tokens_path),
        "output_dir": str(out_dir),
        "outputs": {
            "header": {
                "path": str(header_path),
                "width_px": header.width,
                "height_px": header.height,
                "sha12": _sha12(header_path)
            },
            "main": {
                "path": str(main_path),
                "width_px": main.width,
                "height_px": main.height,
                "sha12": _sha12(main_path)
            },
            "combined": {
                "path": str(combined_path),
                "width_px": combined.width,
                "height_px": combined.height,
                "sha12": _sha12(combined_path)
            }
        }
    }

    _write_json(reports_dir / "phase_c_banner_01_hero_report.json", report)

    print(json.dumps(report["outputs"], ensure_ascii=False, indent=2))
    print(f"[OK] Phase C Banner 1 written to: {out_dir}")

    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render Phase C Banner 1: TYPE 3 Hero Product Promise."
    )

    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Phase C Banner 1 render config JSON."
    )
    parser.add_argument(
        "--design-tokens",
        default=None,
        help="Optional design tokens JSON path. Defaults to config design_tokens."
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Optional output directory. Defaults to config output_dir."
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing Banner 1 output directory before rendering."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve inputs and outputs without rendering."
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        run(args)
        return 0
    except Exception as exc:
        print(f"[ERROR] Phase C Banner 1 render failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())