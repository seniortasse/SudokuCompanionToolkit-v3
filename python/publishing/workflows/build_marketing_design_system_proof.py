from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_CONFIG = Path(
    "datasets/sudoku_books/classic9/marketing_specs/"
    "BK-CL9-DW-B01.marketing.phase_b_design_system.json"
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
            "Pillow is required for Phase B. Install with: python -m pip install pillow"
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


def _draw_text(
    draw: Any,
    xy: Tuple[int, int],
    text: str,
    *,
    font: Any,
    fill: Tuple[int, int, int],
) -> None:
    draw.text(xy, text, font=font, fill=fill)


def _wrap_text(draw: Any, text: str, font: Any, max_width: int) -> List[str]:
    words = str(text).split()
    lines: List[str] = []
    current = ""

    for word in words:
        trial = word if not current else f"{current} {word}"
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
    line_gap: int = 5,
) -> int:
    x, y = xy
    for line in _wrap_text(draw, text, font, max_width):
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((0, 0), line, font=font)
        y += bbox[3] - bbox[1] + line_gap
    return y


def _rounded_rect_with_shadow(
    img: Any,
    box: Tuple[int, int, int, int],
    *,
    radius: int,
    fill: Tuple[int, int, int],
    outline: Optional[Tuple[int, int, int]] = None,
    shadow: bool = True,
    shadow_offset: int = 8,
    shadow_blur: int = 14,
) -> None:
    from PIL import Image, ImageDraw, ImageFilter

    if not shadow:
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline)
        return

    x0, y0, x1, y1 = box
    shadow_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shadow_layer)
    sdraw.rounded_rectangle(
        (x0 + shadow_offset, y0 + shadow_offset, x1 + shadow_offset, y1 + shadow_offset),
        radius=radius,
        fill=(0, 20, 40, 45),
    )
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(shadow_blur))
    img.alpha_composite(shadow_layer)

    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline)


def _load_image(path: Path, fallback_size: Tuple[int, int]):
    from PIL import Image, ImageDraw

    if path.exists():
        return Image.open(path).convert("RGB")

    img = Image.new("RGB", fallback_size, (240, 244, 250))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, fallback_size[0] - 1, fallback_size[1] - 1], outline=(190, 200, 215), width=2)
    draw.text((20, 20), "Missing asset", font=_font(18, bold=True), fill=(90, 100, 115))
    return img


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


def _find_first(root: Path, patterns: Sequence[str]) -> Optional[Path]:
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[0]
    return None


def _find_many(root: Path, patterns: Sequence[str], limit: int) -> List[Path]:
    out: List[Path] = []
    seen = set()
    for pattern in patterns:
        for p in sorted(root.glob(pattern)):
            if p.is_file() and p not in seen:
                seen.add(p)
                out.append(p)
                if len(out) >= limit:
                    return out
    return out


def _draw_section_title(
    draw: Any,
    y: int,
    title: str,
    colors: Dict[str, Tuple[int, int, int]],
    width: int,
) -> int:
    x = 70
    font = _font(28, bold=True)
    draw.text((x, y), title, font=font, fill=colors["navy_950"])
    y += 42
    draw.line((x, y, width - x, y), fill=colors["line_soft"], width=2)
    return y + 28


def _draw_brand_header(
    img: Any,
    draw: Any,
    *,
    y: int,
    config: Dict[str, Any],
    colors: Dict[str, Tuple[int, int, int]],
) -> int:
    width = img.width

    _rounded_rect_with_shadow(
        img,
        (48, y, width - 48, y + 210),
        radius=28,
        fill=colors["navy_950"],
        shadow=True,
    )

    x = 82
    yy = y + 34

    hero_font = _font(42, bold=True)
    sub_font = _font(21, bold=False)
    micro_font = _font(17, bold=False)

    brand = config["brand"]

    draw.text((x, yy), brand["product_title"], font=hero_font, fill=colors["paper"])
    yy += 56

    draw.text((x, yy), brand["product_subtitle"], font=sub_font, fill=colors["blue_100"])
    yy += 38

    draw.text((x, yy), brand["support_line"], font=micro_font, fill=colors["cream"])
    yy += 34

    draw.text(
        (x, yy),
        brand["campaign_title"],
        font=_font(18, bold=True),
        fill=colors["gold"],
    )

    # right-side campaign chips
    chips = ["2027", "1008 puzzles", "Medium → Hard"]
    cx = width - 365
    cy = y + 48
    for chip in chips:
        w = 260
        h = 38
        draw.rounded_rectangle((cx, cy, cx + w, cy + h), radius=19, fill=colors["paper"])
        draw.text((cx + 22, cy + 9), chip, font=_font(16, bold=True), fill=colors["navy_950"])
        cy += 52

    return y + 245


def _draw_palette(
    img: Any,
    draw: Any,
    *,
    y: int,
    config: Dict[str, Any],
    colors: Dict[str, Tuple[int, int, int]],
) -> int:
    y = _draw_section_title(draw, y, "1. Campaign palette", colors, img.width)

    x = 70
    chip_w = 170
    chip_h = 92
    gap = 18

    for idx, (name, hex_value) in enumerate(config["colors"].items()):
        row = idx // 6
        col = idx % 6
        xx = x + col * (chip_w + gap)
        yy = y + row * (chip_h + 38)

        rgb = _hex_to_rgb(hex_value)
        draw.rounded_rectangle((xx, yy, xx + chip_w, yy + chip_h), radius=18, fill=rgb, outline=colors["line_soft"])

        label_fill = colors["paper"] if sum(rgb) < 410 else colors["ink"]
        draw.text((xx + 12, yy + 14), name, font=_font(14, bold=True), fill=label_fill)
        draw.text((xx + 12, yy + 42), hex_value, font=_font(13, bold=False), fill=label_fill)

    rows = math.ceil(len(config["colors"]) / 6)
    return y + rows * (chip_h + 38) + 8


def _draw_typography(
    img: Any,
    draw: Any,
    *,
    y: int,
    config: Dict[str, Any],
    colors: Dict[str, Tuple[int, int, int]],
) -> int:
    y = _draw_section_title(draw, y, "2. Typography hierarchy", colors, img.width)

    x = 70
    examples = [
        ("Hero", "1000+ Sudoku Puzzles for Adults", 42, True, colors["navy_950"]),
        ("Section heading", "Built for more than filling squares", 30, True, colors["navy_900"]),
        ("Card title", "Beautiful on the page", 22, True, colors["ink"]),
        ("Body", "Clean 6-up layouts designed for calm, focused solving.", 17, False, colors["muted"]),
        ("Badge", "Medium → Hard", 16, True, colors["paper"]),
    ]

    for label, sample, size, bold, fill in examples:
        draw.text((x, y + 4), label, font=_font(14, bold=True), fill=colors["gold"])
        if label == "Badge":
            draw.rounded_rectangle((x + 180, y, x + 380, y + 40), radius=20, fill=colors["navy_900"])
            draw.text((x + 204, y + 10), sample, font=_font(size, bold=bold), fill=fill)
        else:
            draw.text((x + 180, y), sample, font=_font(size, bold=bold), fill=fill)
        y += 58

    return y + 10


def _draw_badges(
    img: Any,
    draw: Any,
    *,
    y: int,
    config: Dict[str, Any],
    colors: Dict[str, Tuple[int, int, int]],
) -> int:
    y = _draw_section_title(draw, y, "3. Badge system", colors, img.width)

    badges = config["sample_copy"]["badges"]
    x = 70
    yy = y

    styles = [
        (colors["navy_900"], colors["paper"], colors["gold"]),
        (colors["paper"], colors["navy_950"], colors["navy_900"]),
        (colors["blue_100"], colors["navy_950"], colors["blue_300"]),
        (colors["cream"], colors["navy_950"], colors["gold"]),
    ]

    for idx, badge in enumerate(badges):
        fill, text_fill, outline = styles[idx % len(styles)]
        xx = x + idx * 245
        draw.rounded_rectangle((xx, yy, xx + 210, yy + 48), radius=24, fill=fill, outline=outline, width=2)
        draw.text((xx + 24, yy + 14), badge, font=_font(16, bold=True), fill=text_fill)

    return y + 85


def _draw_sample_card(
    img: Any,
    draw: Any,
    *,
    x: int,
    y: int,
    image_path: Optional[Path],
    image_size: int,
    text_h: int,
    title: str,
    body: str,
    microcopy: Optional[str],
    colors: Dict[str, Tuple[int, int, int]],
    radius: int,
) -> None:
    card_w = image_size
    card_h = image_size + text_h

    _rounded_rect_with_shadow(
        img,
        (x, y, x + card_w, y + card_h),
        radius=radius,
        fill=colors["paper"],
        outline=colors["line_soft"],
        shadow=True,
    )

    src = _load_image(image_path or Path("__missing__"), (image_size, image_size))
    crop = _fit_crop(src, (image_size - 18, image_size - 18))
    crop = _mask_rounded(crop, radius=max(8, radius - 4))
    img.alpha_composite(crop, (x + 9, y + 9))

    tx = x + 14
    ty = y + image_size + 12
    draw.text((tx, ty), title, font=_font(17 if image_size == 220 else 20, bold=True), fill=colors["ink"])
    ty += 27 if image_size == 220 else 32
    ty = _draw_wrapped_text(
        draw,
        (tx, ty),
        body,
        font=_font(12 if image_size == 220 else 14, bold=False),
        fill=colors["muted"],
        max_width=card_w - 28,
        line_gap=3,
    )

    if microcopy:
        ty += 3
        draw.text((tx, ty), microcopy, font=_font(11, bold=True), fill=colors["navy_900"])


def _draw_cards(
    img: Any,
    draw: Any,
    *,
    y: int,
    config: Dict[str, Any],
    curated_root: Path,
    colors: Dict[str, Tuple[int, int, int]],
) -> int:
    y = _draw_section_title(draw, y, "4. Card system: TYPE 1 and TYPE 2", colors, img.width)

    # TYPE 1 row, 4 cards.
    draw.text((70, y), "TYPE 1 card rhythm — 4 × 220 image cards with text below", font=_font(18, bold=True), fill=colors["navy_950"])
    y += 35

    type1_assets = [
        curated_root / "banner_02_four_pillars" / "card_01_beautiful_on_the_page.png",
        curated_root / "banner_02_four_pillars" / "card_02_real_climb.png",
        curated_root / "banner_02_four_pillars" / "card_03_patterns_matter.png",
        curated_root / "banner_02_four_pillars" / "card_04_help_not_answer_dumping.png",
    ]

    type1_cards = config["sample_copy"]["type1_cards"]
    x = 70
    for i, card in enumerate(type1_cards):
        _draw_sample_card(
            img,
            draw,
            x=x + i * 250,
            y=y,
            image_path=type1_assets[i],
            image_size=220,
            text_h=90,
            title=card["title"],
            body=card["body"],
            microcopy=card.get("microcopy"),
            colors=colors,
            radius=18,
        )

    y += 350

    # TYPE 2 row, 3 cards.
    draw.text((70, y), "TYPE 2 card rhythm — 3 × 300 image cards with deeper benefit copy", font=_font(18, bold=True), fill=colors["navy_950"])
    y += 35

    type2_assets = [
        curated_root / "banner_03_three_benefits" / "card_01_relax_into_the_grid.png",
        curated_root / "banner_03_three_benefits" / "card_02_sharpen_pattern_recognition.png",
        curated_root / "banner_03_three_benefits" / "card_03_keep_going_when_stuck.png",
    ]

    type2_cards = config["sample_copy"]["type2_cards"]
    x = 70
    for i, card in enumerate(type2_cards):
        _draw_sample_card(
            img,
            draw,
            x=x + i * 335,
            y=y,
            image_path=type2_assets[i],
            image_size=300,
            text_h=105,
            title=card["title"],
            body=card["body"],
            microcopy=None,
            colors=colors,
            radius=22,
        )

    return y + 455


def _draw_image_treatments(
    img: Any,
    draw: Any,
    *,
    y: int,
    config: Dict[str, Any],
    curated_root: Path,
    colors: Dict[str, Tuple[int, int, int]],
) -> int:
    y = _draw_section_title(draw, y, "5. Image treatments", colors, img.width)

    x = 70
    w = 250
    h = 210
    gap = 28

    samples = [
        (
            "Cover anchor",
            curated_root / "_shared" / "cover_front.png",
            "Product identity stays visible in opener and closer."
        ),
        (
            "Interior proof",
            curated_root / "_shared" / "page_puzzle_6up_1.png",
            "Real page crops prove the printed layout."
        ),
        (
            "Pattern identity",
            curated_root / "_shared" / "page_pattern_sneak_peek_1.png",
            "Pattern previews make the book distinctive."
        ),
        (
            "Grid symbol",
            curated_root / "banner_05_brand_closer" / "closer_hero_grid.png",
            "Clean Sudoku grid becomes the campaign icon."
        )
    ]

    for idx, (label, path, note) in enumerate(samples):
        xx = x + idx * (w + gap)
        _rounded_rect_with_shadow(
            img,
            (xx, y, xx + w, y + h),
            radius=20,
            fill=colors["paper"],
            outline=colors["line_soft"],
            shadow=True,
        )

        src = _load_image(path, (w, 140))
        treatment = _fit_crop(src, (w - 24, 130))
        treatment = _mask_rounded(treatment, 14)
        img.alpha_composite(treatment, (xx + 12, y + 12))

        draw.text((xx + 14, y + 154), label, font=_font(16, bold=True), fill=colors["ink"])
        _draw_wrapped_text(
            draw,
            (xx + 14, y + 178),
            note,
            font=_font(11, bold=False),
            fill=colors["muted"],
            max_width=w - 28,
            line_gap=2,
        )

    return y + h + 35


def _draw_banner_rhythm(
    img: Any,
    draw: Any,
    *,
    y: int,
    config: Dict[str, Any],
    colors: Dict[str, Tuple[int, int, int]],
) -> int:
    y = _draw_section_title(draw, y, "6. Full 5-banner rhythm", colors, img.width)

    x = 70
    row_h = 70
    labels = [
        ("01", "TYPE 3", "Hero Product Promise", "What it is"),
        ("02", "TYPE 1", "Four Pillars", "Why it stands apart"),
        ("03", "TYPE 2", "Three Benefits", "Why the buyer wants it"),
        ("04", "TYPE 1", "Logic Journey", "How the book teaches the eye"),
        ("05", "TYPE 3", "Brand Closer", "What the brand means"),
    ]

    for idx, (num, kind, name, purpose) in enumerate(labels):
        yy = y + idx * (row_h + 12)
        fill = colors["navy_950"] if idx in (0, 4) else colors["paper"]
        text_fill = colors["paper"] if idx in (0, 4) else colors["ink"]
        outline = colors["gold"] if idx in (0, 4) else colors["line_soft"]

        _rounded_rect_with_shadow(
            img,
            (x, yy, img.width - 70, yy + row_h),
            radius=20,
            fill=fill,
            outline=outline,
            shadow=True,
            shadow_offset=5,
            shadow_blur=9,
        )

        draw.text((x + 24, yy + 18), num, font=_font(24, bold=True), fill=colors["gold"] if idx in (0, 4) else colors["navy_900"])
        draw.text((x + 95, yy + 17), kind, font=_font(16, bold=True), fill=text_fill)
        draw.text((x + 230, yy + 15), name, font=_font(21, bold=True), fill=text_fill)
        draw.text((x + 610, yy + 20), purpose, font=_font(16, bold=False), fill=colors["blue_100"] if idx in (0, 4) else colors["muted"])

    return y + len(labels) * (row_h + 12) + 20


def _estimate_height(config: Dict[str, Any]) -> int:
    return 2600


def _draw_design_system_proof(config: Dict[str, Any], out_path: Path) -> None:
    _require_pillow()
    from PIL import Image, ImageDraw

    colors = {k: _hex_to_rgb(v) for k, v in config["colors"].items()}
    curated_root = Path(config["curated_assets_root"])

    width = int(config["proof_sheet"]["width_px"])
    height = _estimate_height(config)
    bg = _hex_to_rgb(config["proof_sheet"].get("background", "#F4F7FB"))

    img = Image.new("RGBA", (width, height), bg + (255,))
    draw = ImageDraw.Draw(img)

    y = 42
    y = _draw_brand_header(img, draw, y=y, config=config, colors=colors)
    y = _draw_palette(img, draw, y=y, config=config, colors=colors)
    y = _draw_typography(img, draw, y=y, config=config, colors=colors)
    y = _draw_badges(img, draw, y=y, config=config, colors=colors)
    y = _draw_cards(img, draw, y=y, config=config, curated_root=curated_root, colors=colors)
    y = _draw_image_treatments(img, draw, y=y, config=config, curated_root=curated_root, colors=colors)
    y = _draw_banner_rhythm(img, draw, y=y, config=config, colors=colors)

    # Crop to actual content height.
    final_h = min(height, y + 40)
    img = img.crop((0, 0, width, final_h)).convert("RGB")

    _ensure_dir(out_path.parent)
    img.save(out_path)


def _build_tokens(config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": "marketing_design_tokens.v1",
        "book_id": config["book_id"],
        "campaign_id": config["campaign_id"],
        "brand": config["brand"],
        "colors": config["colors"],
        "typography": config["typography"],
        "layout": config["layout"],
        "components": config["components"],
        "usage_notes": [
            "Use navy_950/navy_900 as the dominant campaign background.",
            "Use paper and paper_soft for cards and page surfaces.",
            "Use gold sparingly for premium accent, badge outlines, and campaign memory points.",
            "Use real curated assets from Phase A2 rather than generic Sudoku illustrations.",
            "Keep card copy short and let grids/patterns carry the visual identity."
        ]
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    config_path = Path(args.config)
    config = _read_json(config_path)

    out_root = Path(args.out_root or config["output_root"])
    reports_dir = out_root / "_reports"

    if args.clean and out_root.exists():
        shutil.rmtree(out_root)

    _ensure_dir(out_root)
    _ensure_dir(reports_dir)

    if args.dry_run:
        report = {
            "ok": True,
            "dry_run": True,
            "phase": "B",
            "book_id": config["book_id"],
            "config_path": str(config_path),
            "curated_assets_root": str(Path(config["curated_assets_root"])),
            "output_root": str(out_root),
            "expected_outputs": [
                str(out_root / "B01_marketing_design_tokens.json"),
                str(out_root / "B01_marketing_design_system_proof.png"),
                str(reports_dir / "phase_b_design_system_report.json")
            ]
        }
        _write_json(reports_dir / "phase_b_dry_run_report.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report

    tokens = _build_tokens(config)
    tokens_path = out_root / "B01_marketing_design_tokens.json"
    proof_path = out_root / "B01_marketing_design_system_proof.png"

    _write_json(tokens_path, tokens)
    _draw_design_system_proof(config, proof_path)

    # Keep a copy of the spec for traceability.
    shutil.copy2(config_path, reports_dir / config_path.name)

    report = {
        "ok": True,
        "phase": "B",
        "book_id": config["book_id"],
        "campaign_id": config["campaign_id"],
        "config_path": str(config_path),
        "output_root": str(out_root),
        "outputs": {
            "tokens": str(tokens_path),
            "proof_sheet": str(proof_path),
            "proof_sha12": _sha12(proof_path),
            "tokens_sha12": _sha12(tokens_path)
        }
    }

    _write_json(reports_dir / "phase_b_design_system_report.json", report)

    print(json.dumps(report["outputs"], ensure_ascii=False, indent=2))
    print(f"[OK] Phase B design-system proof written to: {proof_path}")
    print(f"[OK] Phase B tokens written to: {tokens_path}")

    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Phase B marketing design-system proof for the Sudoku banner campaign."
    )

    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Phase B design-system config JSON."
    )
    parser.add_argument(
        "--out-root",
        default=None,
        help="Optional output root. Defaults to config output_root."
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing Phase B output folder before building."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve inputs and outputs without generating the proof sheet."
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        run(args)
        return 0
    except Exception as exc:
        print(f"[ERROR] Phase B design-system proof failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())