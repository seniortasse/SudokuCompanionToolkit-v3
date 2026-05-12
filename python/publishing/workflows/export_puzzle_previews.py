from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

try:
    from python.publishing.techniques.technique_catalog import (
        TECHNIQUE_CATALOG,
        normalize_technique_id,
    )
except Exception:  # pragma: no cover
    TECHNIQUE_CATALOG = {}

    def normalize_technique_id(value: str) -> str:
        return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


RGB = Tuple[int, int, int]


DIGIT_SIZE_PRESETS = {
    "tiny": 0.34,
    "small": 0.42,
    "medium": 0.50,
    "large": 0.58,
    "very_large": 0.66,
    "huge": 0.72,
    "super_huge": 0.78,
}


LINE_THICKNESS_PRESETS = {
    # thin_ratio, divider_ratio, outer_ratio as a fraction of cell size
    "very_thin": (0.010, 0.026, 0.034),
    "thin": (0.014, 0.034, 0.044),
    "normal": (0.018, 0.044, 0.056),
    "thick": (0.021, 0.052, 0.066),
    "very_thick": (0.025, 0.062, 0.078),
    "super_thick": (0.030, 0.076, 0.094),
}


STYLE_PRESETS = {
    "book_like": {
        "digit_size": "huge",
        "line_thickness": "thick",
        "digit_y_offset": 0.000,
        "margin": 78,
        "font_family": "arial",
    },
    "book_heavy": {
        "digit_size": "huge",
        "line_thickness": "very_thick",
        "digit_y_offset": 0.000,
        "margin": 78,
        "font_family": "arial",
    },
    "preview_clean": {
        "digit_size": "very_large",
        "line_thickness": "thick",
        "digit_y_offset": 0.000,
        "margin": 70,
        "font_family": "arial",
    },
    "minimal": {
        "digit_size": "large",
        "line_thickness": "normal",
        "digit_y_offset": 0.000,
        "margin": 80,
        "font_family": "arial",
    },
}


def _log(message: str = "") -> None:
    print(message, flush=True)


def _safe_str(value: Any) -> str:
    return str(value or "").strip()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _norm(value: Any) -> str:
    return normalize_technique_id(str(value or ""))


def _safe_filename(value: str) -> str:
    raw = str(value or "").strip() or "puzzle"
    raw = re.sub(r"[^A-Za-z0-9._-]+", "_", raw)
    return raw.strip("._") or "puzzle"


def _unique(values: Iterable[str]) -> list:
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _parse_rgb(value: str, default: RGB) -> RGB:
    raw = _safe_str(value)
    if not raw:
        return default

    named = {
        "black": (0, 0, 0),
        "dark_gray": (60, 60, 60),
        "gray": (90, 90, 90),
        "light_gray": (180, 180, 180),
        "white": (255, 255, 255),
        "navy": (10, 23, 45),
        "blue": (0, 72, 150),
        "red": (180, 0, 0),
    }

    key = raw.lower().replace("-", "_").replace(" ", "_")
    if key in named:
        return named[key]

    if raw.startswith("#"):
        raw = raw[1:]

    if re.fullmatch(r"[0-9A-Fa-f]{6}", raw):
        return (int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16))

    parts = [p.strip() for p in raw.split(",")]
    if len(parts) == 3:
        try:
            r, g, b = [max(0, min(255, int(p))) for p in parts]
            return (r, g, b)
        except Exception:
            return default

    return default


def _load_json_records(records_dir: Path) -> list:
    if not records_dir.exists():
        raise FileNotFoundError(f"Puzzle records directory does not exist: {records_dir}")

    records = []

    for path in sorted(records_dir.rglob("*.json")):
        if path.name.startswith("_"):
            continue

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            _log(f"WARNING | could not read JSON: {path} | {exc}")
            continue

        if not isinstance(data, dict):
            continue

        if "givens81" not in data and "solution81" not in data:
            continue

        data.setdefault("_source_path", str(path))
        records.append(data)

    return records


def _load_records(*, records_dir: str = "", book_dir: str = "") -> tuple:
    if book_dir:
        root = Path(book_dir)
        puzzles_dir = root / "puzzles"
        return _load_json_records(puzzles_dir), puzzles_dir, "built_book"

    root = Path(records_dir or "datasets/sudoku_books/classic9/puzzle_records")
    return _load_json_records(root), root, "records_dir"


def _record_id_variants(value: str) -> set:
    raw = _safe_str(value)
    variants = {raw} if raw else set()

    match = re.fullmatch(r"(REC-CL9-)(\d+)", raw)
    if match:
        prefix, digits = match.groups()
        number = int(digits)
        variants.add(f"{prefix}{number:06d}")
        variants.add(f"{prefix}{number:08d}")
        variants.add(str(number))

    if raw.isdigit():
        number = int(raw)
        variants.add(f"REC-CL9-{number:06d}")
        variants.add(f"REC-CL9-{number:08d}")

    return {v for v in variants if v}


def _record_id_candidates(record: dict) -> set:
    fields = [
        "record_id",
        "puzzle_uid",
        "local_puzzle_code",
        "friendly_puzzle_id",
        "title",
    ]

    candidates = set()
    for field in fields:
        value = _safe_str(record.get(field))
        if not value:
            continue
        candidates.add(value)
        candidates.update(_record_id_variants(value))

    return candidates


def _display_id(record: dict) -> str:
    return (
        _safe_str(record.get("record_id"))
        or _safe_str(record.get("puzzle_uid"))
        or _safe_str(record.get("local_puzzle_code"))
        or "puzzle"
    )


def _catalog_lookup_by_public_name(raw: str) -> Optional[str]:
    wanted = _norm(raw)
    if not wanted:
        return None

    for key, entry in TECHNIQUE_CATALOG.items():
        candidates = {
            _norm(key),
            _norm(getattr(entry, "engine_id", "")),
            _norm(getattr(entry, "canonical_id", "")),
            _norm(getattr(entry, "public_name", "")),
            _norm(getattr(entry, "public_name_plural", "")),
        }
        if wanted in candidates:
            return _norm(getattr(entry, "engine_id", key))

    return None


def _resolve_technique_query(raw: str) -> str:
    normalized = _norm(raw)
    if not normalized:
        return ""

    entry = TECHNIQUE_CATALOG.get(normalized)
    if entry:
        return _norm(getattr(entry, "engine_id", normalized))

    by_public = _catalog_lookup_by_public_name(raw)
    if by_public:
        return by_public

    return normalized


def _parse_repeated_techniques(values: Sequence[str]) -> list:
    resolved = []
    for raw in values:
        for part in str(raw or "").replace(",", "+").split("+"):
            token = _resolve_technique_query(part)
            if token:
                resolved.append(token)
    return _unique(resolved)


def _record_techniques(record: dict) -> set:
    techniques = set()

    for technique in list(record.get("techniques_used") or []):
        normalized = _resolve_technique_query(str(technique))
        if normalized:
            techniques.add(normalized)

    histogram = dict(record.get("technique_histogram") or {})
    for technique in histogram.keys():
        normalized = _resolve_technique_query(str(technique))
        if normalized:
            techniques.add(normalized)

    return techniques


def _matches(
    record: dict,
    *,
    record_ids: Sequence[str],
    pattern_ids: Sequence[str],
    pattern_family_ids: Sequence[str],
    section: str,
    difficulty: str,
    min_weight: Optional[int],
    max_weight: Optional[int],
    min_clues: Optional[int],
    max_clues: Optional[int],
    min_technique_count: Optional[int],
    max_technique_count: Optional[int],
    required_techniques: Sequence[str],
    any_techniques: Sequence[str],
    excluded_techniques: Sequence[str],
) -> bool:
    if record_ids:
        wanted = set()
        for value in record_ids:
            wanted.update(_record_id_variants(value))
            wanted.add(_safe_str(value))

        if not (_record_id_candidates(record) & wanted):
            return False

    if pattern_ids:
        wanted = {_safe_str(x) for x in pattern_ids if _safe_str(x)}
        if _safe_str(record.get("pattern_id")) not in wanted:
            return False

    if pattern_family_ids:
        wanted = {_safe_str(x) for x in pattern_family_ids if _safe_str(x)}
        if _safe_str(record.get("pattern_family_id")) not in wanted:
            return False

    if section and _safe_str(record.get("section_code")).lower() != section.lower():
        return False

    if difficulty:
        values = {
            _safe_str(record.get("puzzle_difficulty")).lower(),
            _safe_str(record.get("difficulty_label")).lower(),
            _safe_str(record.get("difficulty_band_code")).lower(),
        }
        if difficulty.lower() not in values:
            return False

    weight = _safe_int(record.get("weight"))
    clue_count = _safe_int(record.get("clue_count"))
    technique_count = _safe_int(record.get("technique_count"))

    if min_weight is not None and weight < min_weight:
        return False
    if max_weight is not None and weight > max_weight:
        return False
    if min_clues is not None and clue_count < min_clues:
        return False
    if max_clues is not None and clue_count > max_clues:
        return False
    if min_technique_count is not None and technique_count < min_technique_count:
        return False
    if max_technique_count is not None and technique_count > max_technique_count:
        return False

    used = _record_techniques(record)

    if required_techniques and not all(t in used for t in required_techniques):
        return False

    if any_techniques and not any(t in used for t in any_techniques):
        return False

    if excluded_techniques and any(t in used for t in excluded_techniques):
        return False

    return True


def _sort_records(records: Sequence[dict], sort_key: str) -> list:
    if sort_key == "weight_desc":
        return sorted(records, key=lambda r: (-_safe_int(r.get("weight")), _safe_str(r.get("record_id"))))

    if sort_key == "weight_asc":
        return sorted(records, key=lambda r: (_safe_int(r.get("weight")), _safe_str(r.get("record_id"))))

    if sort_key == "technique_count_desc":
        return sorted(
            records,
            key=lambda r: (
                -_safe_int(r.get("technique_count")),
                -_safe_int(r.get("weight")),
                _safe_str(r.get("record_id")),
            ),
        )

    if sort_key == "clue_count_asc":
        return sorted(records, key=lambda r: (_safe_int(r.get("clue_count")), _safe_str(r.get("record_id"))))

    return sorted(
        records,
        key=lambda r: (
            _safe_int(r.get("position_in_book"), default=10**9),
            _safe_int(r.get("position_in_section"), default=10**9),
            _safe_str(r.get("record_id")),
        ),
    )


def _load_font(
    *,
    size: int,
    font_family: str,
    font_path: str = "",
    bold: bool = False,
) -> ImageFont.ImageFont:
    if font_path:
        try:
            return ImageFont.truetype(font_path, size=size)
        except Exception as exc:
            _log(f"WARNING | could not load font path {font_path}: {exc}")

    family_key = _safe_str(font_family).lower().replace("-", "_").replace(" ", "_")

    if family_key in {"arial", "arial_regular"}:
        regular_candidates = [
            "arial.ttf",
            "Arial.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
            "/usr/share/fonts/truetype/msttcorefonts/arial.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        bold_candidates = [
            "arialbd.ttf",
            "Arial Bold.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
    elif family_key in {"liberation_sans", "liberation"}:
        regular_candidates = [
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "LiberationSans-Regular.ttf",
            "arial.ttf",
            "DejaVuSans.ttf",
        ]
        bold_candidates = [
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "LiberationSans-Bold.ttf",
            "arialbd.ttf",
            "DejaVuSans-Bold.ttf",
        ]
    elif family_key in {"dejavu_sans", "dejavu"}:
        regular_candidates = [
            "DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "arial.ttf",
        ]
        bold_candidates = [
            "DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "arialbd.ttf",
        ]
    else:
        regular_candidates = [
            font_family,
            f"{font_family}.ttf",
            "arial.ttf",
            "Arial.ttf",
            "DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        bold_candidates = [
            f"{font_family}-Bold.ttf",
            "arialbd.ttf",
            "DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]

    candidates = bold_candidates + regular_candidates if bold else regular_candidates

    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            pass

    return ImageFont.load_default()


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple,
    text: str,
    font: ImageFont.ImageFont,
    fill: RGB,
) -> None:
    x, y = xy
    value = str(text or "")

    try:
        draw.text((x, y), value, font=font, fill=fill, anchor="mm")
        return
    except Exception:
        pass

    bbox = draw.textbbox((0, 0), value, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    draw.text((x - width / 2 - bbox[0], y - height / 2 - bbox[1]), value, font=font, fill=fill)


def _draw_digit_centered(
    draw: ImageDraw.ImageDraw,
    *,
    center_x: float,
    center_y: float,
    value: str,
    font: ImageFont.ImageFont,
    fill: RGB,
    y_offset_ratio: float,
    x_offset_ratio: float,
    cell_size: float,
) -> None:
    x = center_x + (cell_size * x_offset_ratio)
    y = center_y + (cell_size * y_offset_ratio)

    # Use actual glyph bounding box for the strongest cross-version centering.
    # This works more consistently than relying on baseline placement.
    bbox = draw.textbbox((0, 0), value, font=font)
    glyph_w = bbox[2] - bbox[0]
    glyph_h = bbox[3] - bbox[1]

    left = x - (glyph_w / 2.0) - bbox[0]
    top = y - (glyph_h / 2.0) - bbox[1]

    draw.text((left, top), value, font=font, fill=fill)


def _stroke_widths_from_preset(
    *,
    preset: str,
    cell_size: float,
    thin_px: Optional[int],
    divider_px: Optional[int],
    outer_px: Optional[int],
) -> tuple:
    ratios = LINE_THICKNESS_PRESETS.get(preset, LINE_THICKNESS_PRESETS["thick"])
    thin_ratio, divider_ratio, outer_ratio = ratios

    thin = thin_px if thin_px is not None else round(cell_size * thin_ratio)
    divider = divider_px if divider_px is not None else round(cell_size * divider_ratio)
    outer = outer_px if outer_px is not None else round(cell_size * outer_ratio)

    thin = max(1, int(thin))
    divider = max(thin + 1, int(divider))
    outer = max(divider, int(outer))

    return thin, divider, outer


def _draw_puzzle_preview_png(
    *,
    record: dict,
    output_png: Path,
    show_solution: bool,
    image_size_px: int,
    margin_px: int,
    include_header: bool,
    digit_size_preset: str,
    digit_font_size: Optional[int],
    font_family: str,
    font_path: str,
    line_thickness_preset: str,
    thin_line_px: Optional[int],
    divider_line_px: Optional[int],
    outer_line_px: Optional[int],
    digit_color: RGB,
    grid_color: RGB,
    background_color: RGB,
    header_color: RGB,
    header_subtitle_color: RGB,
    digit_y_offset_ratio: float,
    digit_x_offset_ratio: float,
) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)

    digits = _safe_str(record.get("solution81") if show_solution else record.get("givens81"))
    if len(digits) != 81:
        raise ValueError(f"{_display_id(record)} has invalid digits length: {len(digits)}")

    image = Image.new("RGB", (image_size_px, image_size_px), background_color)
    draw = ImageDraw.Draw(image)

    title_font = _load_font(
        size=max(16, image_size_px // 45),
        font_family=font_family,
        font_path=font_path,
        bold=False,
    )
    subtitle_font = _load_font(
        size=max(10, image_size_px // 82),
        font_family=font_family,
        font_path=font_path,
        bold=False,
    )

    header_height = 74 if include_header else 0

    if include_header:
        title = _display_id(record)
        subtitle_parts = [
            _safe_str(record.get("puzzle_difficulty") or record.get("difficulty_label")),
            f"weight={_safe_int(record.get('weight'))}",
            f"clues={_safe_int(record.get('clue_count'))}",
            _safe_str(record.get("pattern_id")),
        ]
        subtitle = " | ".join([x for x in subtitle_parts if x])

        _draw_centered_text(draw, (image_size_px / 2, 24), title, title_font, header_color)
        _draw_centered_text(draw, (image_size_px / 2, 51), subtitle, subtitle_font, header_subtitle_color)

    available_h = image_size_px - header_height - margin_px
    available_w = image_size_px - (2 * margin_px)

    grid_size = min(available_w, available_h)
    grid_size = int(grid_size - (grid_size % 9))

    grid_x = int(round((image_size_px - grid_size) / 2.0))
    grid_y = int(round(header_height + ((image_size_px - header_height - grid_size) / 2.0)))

    cell = grid_size / 9.0

    thin, divider, outer = _stroke_widths_from_preset(
        preset=line_thickness_preset,
        cell_size=cell,
        thin_px=thin_line_px,
        divider_px=divider_line_px,
        outer_px=outer_line_px,
    )

    # Critical fix:
    # Internal lines are inset by half the outer border width so they terminate inside
    # the outside frame instead of poking through or beyond it.
    inset = max(1, int(round(outer / 2.0)))
    inner_left = grid_x + inset
    inner_top = grid_y + inset
    inner_right = grid_x + grid_size - inset
    inner_bottom = grid_y + grid_size - inset

    # Draw thin internal cell lines first.
    for i in range(1, 9):
        if i % 3 == 0:
            continue

        x = int(round(grid_x + i * cell))
        draw.line(
            (x, inner_top, x, inner_bottom),
            fill=grid_color,
            width=thin,
        )

        y = int(round(grid_y + i * cell))
        draw.line(
            (inner_left, y, inner_right, y),
            fill=grid_color,
            width=thin,
        )

    # Draw 3x3 dividers second, also clipped/inset to the inner side of the outer border.
    for i in (3, 6):
        x = int(round(grid_x + i * cell))
        draw.line(
            (x, inner_top, x, inner_bottom),
            fill=grid_color,
            width=divider,
        )

        y = int(round(grid_y + i * cell))
        draw.line(
            (inner_left, y, inner_right, y),
            fill=grid_color,
            width=divider,
        )

    # Draw outer border last, so it cleanly caps all internal line ends.
    # Coordinates are inset by half the stroke width to keep the stroke inside the image.
    half_outer = outer / 2.0
    border_box = (
        int(round(grid_x + half_outer)),
        int(round(grid_y + half_outer)),
        int(round(grid_x + grid_size - half_outer)),
        int(round(grid_y + grid_size - half_outer)),
    )
    draw.rectangle(border_box, outline=grid_color, width=outer)

    digit_ratio = DIGIT_SIZE_PRESETS.get(digit_size_preset, DIGIT_SIZE_PRESETS["huge"])
    resolved_digit_font_size = int(digit_font_size) if digit_font_size else int(cell * digit_ratio)
    resolved_digit_font_size = max(8, resolved_digit_font_size)

    digit_font = _load_font(
        size=resolved_digit_font_size,
        font_family=font_family,
        font_path=font_path,
        bold=False,
    )

    for row in range(9):
        for col in range(9):
            value = digits[row * 9 + col]
            if value in {"0", ".", "-", " "}:
                continue

            cx = grid_x + (col + 0.5) * cell
            cy = grid_y + (row + 0.5) * cell

            _draw_digit_centered(
                draw,
                center_x=cx,
                center_y=cy,
                value=value,
                font=digit_font,
                fill=digit_color,
                y_offset_ratio=digit_y_offset_ratio,
                x_offset_ratio=digit_x_offset_ratio,
                cell_size=cell,
            )

    image.save(output_png)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export PNG previews for Sudoku puzzle records."
    )

    source = parser.add_mutually_exclusive_group()
    source.add_argument("--records-dir", default="")
    source.add_argument("--book-dir", default="")

    parser.add_argument("--out-dir", default="datasets/sudoku_books/classic9/previews")

    parser.add_argument("--record-id", action="append", default=[])
    parser.add_argument("--record-id-file", default="")

    parser.add_argument("--pattern-id", action="append", default=[])
    parser.add_argument("--pattern-family-id", action="append", default=[])
    parser.add_argument("--section", default="")
    parser.add_argument("--difficulty", default="")

    parser.add_argument("--min-weight", type=int, default=None)
    parser.add_argument("--max-weight", type=int, default=None)
    parser.add_argument("--min-clues", type=int, default=None)
    parser.add_argument("--max-clues", type=int, default=None)
    parser.add_argument("--min-technique-count", type=int, default=None)
    parser.add_argument("--max-technique-count", type=int, default=None)

    parser.add_argument("--technique", action="append", default=[])
    parser.add_argument("--any-technique", action="append", default=[])
    parser.add_argument("--exclude-technique", action="append", default=[])

    parser.add_argument(
        "--sort",
        default="book_order",
        choices=["book_order", "weight_asc", "weight_desc", "technique_count_desc", "clue_count_asc"],
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=900)
    parser.add_argument("--margin", type=int, default=None)
    parser.add_argument("--solutions", action="store_true")
    parser.add_argument("--no-header", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--report-json", default="")

    parser.add_argument(
        "--style-preset",
        default="book_like",
        choices=sorted(STYLE_PRESETS.keys()),
        help="High-level visual style preset.",
    )
    parser.add_argument(
        "--digit-size",
        default=None,
        choices=sorted(DIGIT_SIZE_PRESETS.keys()),
        help="Preset digit size relative to cell size. Overrides style preset.",
    )
    parser.add_argument(
        "--digit-font-size",
        type=int,
        default=None,
        help="Exact digit font size in pixels. Overrides --digit-size when provided.",
    )
    parser.add_argument(
        "--font-family",
        default=None,
        help="Preferred font family. Default comes from style preset.",
    )
    parser.add_argument(
        "--font-path",
        default="",
        help="Optional direct path to a .ttf font file. Overrides family lookup.",
    )
    parser.add_argument(
        "--line-thickness",
        default=None,
        choices=sorted(LINE_THICKNESS_PRESETS.keys()),
        help="Grid line thickness preset. Overrides style preset.",
    )
    parser.add_argument(
        "--thin-line-px",
        type=int,
        default=None,
        help="Exact normal cell line thickness in pixels. Overrides preset thin line.",
    )
    parser.add_argument(
        "--divider-line-px",
        type=int,
        default=None,
        help="Exact 3x3 box divider thickness in pixels. Overrides preset divider line.",
    )
    parser.add_argument(
        "--outer-line-px",
        type=int,
        default=None,
        help="Exact outer border thickness in pixels. Overrides preset outer border.",
    )
    parser.add_argument(
        "--digit-color",
        default="black",
        help="Digit color. Supports names, #RRGGBB, or R,G,B.",
    )
    parser.add_argument(
        "--grid-color",
        default="black",
        help="Grid color. Supports names, #RRGGBB, or R,G,B.",
    )
    parser.add_argument(
        "--background-color",
        default="white",
        help="Background color. Supports names, #RRGGBB, or R,G,B.",
    )
    parser.add_argument(
        "--header-color",
        default="black",
        help="Header title color. Supports names, #RRGGBB, or R,G,B.",
    )
    parser.add_argument(
        "--header-subtitle-color",
        default="gray",
        help="Header subtitle color. Supports names, #RRGGBB, or R,G,B.",
    )
    parser.add_argument(
        "--digit-y-offset",
        type=float,
        default=None,
        help="Optical vertical offset as a ratio of cell size. Positive moves digits down.",
    )
    parser.add_argument(
        "--digit-x-offset",
        type=float,
        default=0.0,
        help="Optical horizontal offset as a ratio of cell size. Positive moves digits right.",
    )

    return parser.parse_args()


def _resolve_style_args(args: argparse.Namespace) -> dict:
    preset = STYLE_PRESETS.get(args.style_preset, STYLE_PRESETS["book_like"])

    return {
        "margin": int(args.margin if args.margin is not None else preset["margin"]),
        "digit_size": str(args.digit_size or preset["digit_size"]),
        "line_thickness": str(args.line_thickness or preset["line_thickness"]),
        "digit_y_offset": float(
            args.digit_y_offset if args.digit_y_offset is not None else preset["digit_y_offset"]
        ),
        "font_family": str(args.font_family or preset["font_family"]),
    }


def main() -> int:
    args = _parse_args()
    style = _resolve_style_args(args)

    records, source_path, source_kind = _load_records(
        records_dir=str(args.records_dir or "").strip(),
        book_dir=str(args.book_dir or "").strip(),
    )

    record_ids = list(args.record_id or [])
    if args.record_id_file:
        id_path = Path(args.record_id_file)
        record_ids.extend(
            line.strip()
            for line in id_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )

    required_techniques = _parse_repeated_techniques(args.technique)
    any_techniques = _parse_repeated_techniques(args.any_technique)
    excluded_techniques = _parse_repeated_techniques(args.exclude_technique)

    matches = [
        record
        for record in records
        if _matches(
            record,
            record_ids=record_ids,
            pattern_ids=list(args.pattern_id or []),
            pattern_family_ids=list(args.pattern_family_id or []),
            section=str(args.section or "").strip(),
            difficulty=str(args.difficulty or "").strip(),
            min_weight=args.min_weight,
            max_weight=args.max_weight,
            min_clues=args.min_clues,
            max_clues=args.max_clues,
            min_technique_count=args.min_technique_count,
            max_technique_count=args.max_technique_count,
            required_techniques=required_techniques,
            any_techniques=any_techniques,
            excluded_techniques=excluded_techniques,
        )
    ]

    matches = _sort_records(matches, args.sort)
    selected = matches if int(args.limit) == 0 else matches[: int(args.limit)]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exported = []
    skipped = []
    failed = []

    digit_color = _parse_rgb(args.digit_color, (0, 0, 0))
    grid_color = _parse_rgb(args.grid_color, (0, 0, 0))
    background_color = _parse_rgb(args.background_color, (255, 255, 255))
    header_color = _parse_rgb(args.header_color, (0, 0, 0))
    header_subtitle_color = _parse_rgb(args.header_subtitle_color, (90, 90, 90))

    for record in selected:
        display_id = _display_id(record)
        output_png = out_dir / f"{_safe_filename(display_id)}.png"

        if output_png.exists() and not args.force:
            skipped.append({"record_id": display_id, "path": str(output_png), "reason": "exists"})
            continue

        try:
            _draw_puzzle_preview_png(
                record=record,
                output_png=output_png,
                show_solution=bool(args.solutions),
                image_size_px=int(args.image_size),
                margin_px=int(style["margin"]),
                include_header=not bool(args.no_header),
                digit_size_preset=str(style["digit_size"]),
                digit_font_size=args.digit_font_size,
                font_family=str(style["font_family"]),
                font_path=str(args.font_path or ""),
                line_thickness_preset=str(style["line_thickness"]),
                thin_line_px=args.thin_line_px,
                divider_line_px=args.divider_line_px,
                outer_line_px=args.outer_line_px,
                digit_color=digit_color,
                grid_color=grid_color,
                background_color=background_color,
                header_color=header_color,
                header_subtitle_color=header_subtitle_color,
                digit_y_offset_ratio=float(style["digit_y_offset"]),
                digit_x_offset_ratio=float(args.digit_x_offset),
            )
            exported.append(
                {
                    "record_id": _safe_str(record.get("record_id")),
                    "pattern_id": _safe_str(record.get("pattern_id")),
                    "weight": _safe_int(record.get("weight")),
                    "clue_count": _safe_int(record.get("clue_count")),
                    "path": str(output_png),
                }
            )
        except Exception as exc:
            failed.append({"record_id": display_id, "error": str(exc)})

    example_ids = [_safe_str(r.get("record_id")) for r in records[:10] if _safe_str(r.get("record_id"))]

    report = {
        "source_kind": source_kind,
        "source_path": str(source_path),
        "output_dir": str(out_dir),
        "total_loaded": len(records),
        "total_matched": len(matches),
        "total_selected": len(selected),
        "exported_count": len(exported),
        "skipped_count": len(skipped),
        "failed_count": len(failed),
        "example_record_ids": example_ids,
        "style": {
            "style_preset": args.style_preset,
            "image_size": args.image_size,
            "margin": style["margin"],
            "digit_size": style["digit_size"],
            "digit_font_size": args.digit_font_size,
            "font_family": style["font_family"],
            "font_path": args.font_path,
            "line_thickness": style["line_thickness"],
            "thin_line_px": args.thin_line_px,
            "divider_line_px": args.divider_line_px,
            "outer_line_px": args.outer_line_px,
            "digit_color": args.digit_color,
            "grid_color": args.grid_color,
            "background_color": args.background_color,
            "digit_y_offset": style["digit_y_offset"],
            "digit_x_offset": args.digit_x_offset,
        },
        "filters": {
            "record_id": record_ids,
            "pattern_id": list(args.pattern_id or []),
            "pattern_family_id": list(args.pattern_family_id or []),
            "section": str(args.section or "").strip(),
            "difficulty": str(args.difficulty or "").strip(),
            "min_weight": args.min_weight,
            "max_weight": args.max_weight,
            "min_clues": args.min_clues,
            "max_clues": args.max_clues,
            "required_techniques": required_techniques,
            "any_techniques": any_techniques,
            "excluded_techniques": excluded_techniques,
            "sort": args.sort,
            "limit": args.limit,
            "solutions": bool(args.solutions),
        },
        "exported": exported,
        "skipped": skipped,
        "failed": failed,
    }

    report_path = Path(args.report_json) if args.report_json else out_dir / "_last_puzzle_preview_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    _log("=" * 72)
    _log("export_puzzle_previews.py")
    _log("=" * 72)
    _log(f"Source:     {source_kind}")
    _log(f"Path:       {source_path}")
    _log(f"Output dir: {out_dir}")
    _log(f"Loaded:     {len(records)}")
    _log(f"Matched:    {len(matches)}")
    _log(f"Selected:   {len(selected)}")
    _log(f"Exported:   {len(exported)}")
    _log(f"Skipped:    {len(skipped)}")
    _log(f"Failed:     {len(failed)}")
    _log(f"Report:     {report_path}")

    if len(matches) == 0 and record_ids:
        _log("")
        _log("No records matched the requested IDs.")
        _log("Example available record IDs:")
        for rid in example_ids[:10]:
            _log(f"  {rid}")

    _log("=" * 72)

    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())