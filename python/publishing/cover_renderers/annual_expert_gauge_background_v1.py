from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from python.publishing.cover_designs.models import ResolvedCoverDesignContext
from python.publishing.cover_renderers.base_renderer import (
    BaseCoverRenderer,
    CoverRenderResult,
)

_CANONICAL_BG_WIDTH = 1584
_CANONICAL_BG_HEIGHT = 2048

_SAMPLE_MAIN_GIVENS81 = (
    "200000690"
    "360004700"
    "078013000"
    "002530000"
    "000907000"
    "000021400"
    "000470180"
    "007100049"
    "024000007"
)

_SAMPLE_LEFT_GIVENS81 = (
    "900000100"
    "600000000"
    "300000700"
    "700000000"
    "800000900"
    "400000000"
    "500000600"
    "000000000"
    "107400000"
)

_SAMPLE_RIGHT_GIVENS81 = (
    "000000000"
    "000000000"
    "000000000"
    "300000800"
    "000000300"
    "500000000"
    "600400000"
    "200000000"
    "000000700"
)


_FONT_FACE_CANDIDATES: dict[str, list[str]] = {
    # Best title-style presets.
    #
    # Priority for the annual title preset:
    # 1. Montserrat Bold first: your chosen best fit.
    # 2. Montserrat SemiBold / Medium next.
    # 3. Other Montserrat weights as fallback.
    # 4. Other display fonts / Windows-safe fallbacks.
    # 5. Anton last.
    #
    # Put these files under assets/fonts/ for deterministic rendering.
    "annual_expert_gauge_title": [
        "assets/fonts/Montserrat-Bold.ttf",
        "assets/fonts/Montserrat-SemiBold.ttf",
        "assets/fonts/Montserrat-Medium.ttf",
        "assets/fonts/Montserrat-ExtraBold.ttf",
        "assets/fonts/Montserrat-Black.ttf",
        "assets/fonts/Montserrat-Regular.ttf",
        "assets/fonts/LeagueSpartan-Black.ttf",
        "assets/fonts/BebasNeue-Regular.ttf",
        "C:/Windows/Fonts/ariblk.ttf",
        "C:/Windows/Fonts/impact.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/montserrat/Montserrat-Bold.ttf",
        "/usr/share/fonts/truetype/montserrat/Montserrat-SemiBold.ttf",
        "/usr/share/fonts/truetype/montserrat/Montserrat-Medium.ttf",
        "/usr/share/fonts/truetype/montserrat/Montserrat-Regular.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "assets/fonts/Anton-Regular.ttf",
    ],

    "montserrat_regular": [
        "assets/fonts/Montserrat-Regular.ttf",
        "assets/fonts/Montserrat-Medium.ttf",
        "assets/fonts/Montserrat-SemiBold.ttf",
        "assets/fonts/Montserrat-Bold.ttf",
        "/usr/share/fonts/truetype/montserrat/Montserrat-Regular.ttf",
        "/usr/share/fonts/truetype/montserrat/Montserrat-Medium.ttf",
        "/usr/share/fonts/truetype/montserrat/Montserrat-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ],

    "montserrat_medium": [
        "assets/fonts/Montserrat-Medium.ttf",
        "assets/fonts/Montserrat-Regular.ttf",
        "assets/fonts/Montserrat-SemiBold.ttf",
        "assets/fonts/Montserrat-Bold.ttf",
        "/usr/share/fonts/truetype/montserrat/Montserrat-Medium.ttf",
        "/usr/share/fonts/truetype/montserrat/Montserrat-Regular.ttf",
        "/usr/share/fonts/truetype/montserrat/Montserrat-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ],

    "montserrat_semibold": [
        "assets/fonts/Montserrat-SemiBold.ttf",
        "assets/fonts/Montserrat-Bold.ttf",
        "assets/fonts/Montserrat-Medium.ttf",
        "assets/fonts/Montserrat-Regular.ttf",
        "/usr/share/fonts/truetype/montserrat/Montserrat-SemiBold.ttf",
        "/usr/share/fonts/truetype/montserrat/Montserrat-Bold.ttf",
        "/usr/share/fonts/truetype/montserrat/Montserrat-Medium.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ],

    "montserrat_bold": [
        "assets/fonts/Montserrat-Bold.ttf",
        "assets/fonts/Montserrat-SemiBold.ttf",
        "assets/fonts/Montserrat-Medium.ttf",
        "assets/fonts/Montserrat-ExtraBold.ttf",
        "assets/fonts/Montserrat-Black.ttf",
        "/usr/share/fonts/truetype/montserrat/Montserrat-Bold.ttf",
        "/usr/share/fonts/truetype/montserrat/Montserrat-SemiBold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/ariblk.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ],

    "montserrat_extrabold": [
        "assets/fonts/Montserrat-ExtraBold.ttf",
        "assets/fonts/Montserrat-Bold.ttf",
        "assets/fonts/Montserrat-SemiBold.ttf",
        "assets/fonts/Montserrat-Black.ttf",
        "/usr/share/fonts/truetype/montserrat/Montserrat-Bold.ttf",
        "C:/Windows/Fonts/ariblk.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ],

    "montserrat_black": [
        "assets/fonts/Montserrat-Black.ttf",
        "assets/fonts/Montserrat-ExtraBold.ttf",
        "assets/fonts/Montserrat-Bold.ttf",
        "assets/fonts/Montserrat-SemiBold.ttf",
        "/usr/share/fonts/truetype/montserrat/Montserrat-Black.ttf",
        "/usr/share/fonts/truetype/montserrat/Montserrat-Bold.ttf",
        "C:/Windows/Fonts/ariblk.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ],

    # Common misspelling aliases, so "monserrat_*" in JSON still works.
    "monserrat_regular": [
        "assets/fonts/Montserrat-Regular.ttf",
        "assets/fonts/Montserrat-Medium.ttf",
        "assets/fonts/Montserrat-SemiBold.ttf",
        "assets/fonts/Montserrat-Bold.ttf",
        "/usr/share/fonts/truetype/montserrat/Montserrat-Regular.ttf",
        "/usr/share/fonts/truetype/montserrat/Montserrat-Medium.ttf",
        "/usr/share/fonts/truetype/montserrat/Montserrat-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ],

    "monserrat_medium": [
        "assets/fonts/Montserrat-Medium.ttf",
        "assets/fonts/Montserrat-Regular.ttf",
        "assets/fonts/Montserrat-SemiBold.ttf",
        "assets/fonts/Montserrat-Bold.ttf",
        "/usr/share/fonts/truetype/montserrat/Montserrat-Medium.ttf",
        "/usr/share/fonts/truetype/montserrat/Montserrat-Regular.ttf",
        "/usr/share/fonts/truetype/montserrat/Montserrat-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ],

    "monserrat_bold": [
        "assets/fonts/Montserrat-Bold.ttf",
        "assets/fonts/Montserrat-SemiBold.ttf",
        "assets/fonts/Montserrat-Medium.ttf",
        "assets/fonts/Montserrat-ExtraBold.ttf",
        "assets/fonts/Montserrat-Black.ttf",
        "/usr/share/fonts/truetype/montserrat/Montserrat-Bold.ttf",
        "/usr/share/fonts/truetype/montserrat/Montserrat-SemiBold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/ariblk.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ],

    "monserrat_black": [
        "assets/fonts/Montserrat-Black.ttf",
        "assets/fonts/Montserrat-ExtraBold.ttf",
        "assets/fonts/Montserrat-Bold.ttf",
        "assets/fonts/Montserrat-SemiBold.ttf",
        "/usr/share/fonts/truetype/montserrat/Montserrat-Black.ttf",
        "/usr/share/fonts/truetype/montserrat/Montserrat-Bold.ttf",
        "C:/Windows/Fonts/ariblk.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ],

    "bebas_neue": [
        "assets/fonts/BebasNeue-Regular.ttf",
        "C:/Windows/Fonts/impact.ttf",
        "C:/Windows/Fonts/ariblk.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "assets/fonts/Anton-Regular.ttf",
    ],

    "league_spartan_black": [
        "assets/fonts/LeagueSpartan-Black.ttf",
        "C:/Windows/Fonts/ariblk.ttf",
        "C:/Windows/Fonts/impact.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "assets/fonts/Anton-Regular.ttf",
    ],

    "arial_black": [
        "C:/Windows/Fonts/ariblk.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "assets/fonts/Anton-Regular.ttf",
    ],

    "impact": [
        "C:/Windows/Fonts/impact.ttf",
        "C:/Windows/Fonts/ariblk.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "assets/fonts/Anton-Regular.ttf",
    ],

    "anton": [
        "assets/fonts/Anton-Regular.ttf",
        "C:/Windows/Fonts/impact.ttf",
        "C:/Windows/Fonts/ariblk.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ],

    "bold_sans": [
        "assets/fonts/Montserrat-Bold.ttf",
        "assets/fonts/Montserrat-SemiBold.ttf",
        "assets/fonts/Montserrat-Medium.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/calibrib.ttf",
        "/usr/share/fonts/truetype/montserrat/Montserrat-Bold.ttf",
        "/usr/share/fonts/truetype/montserrat/Montserrat-SemiBold.ttf",
        "/usr/share/fonts/truetype/montserrat/Montserrat-Medium.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ],

    "regular_sans": [
        "assets/fonts/Montserrat-Regular.ttf",
        "assets/fonts/Montserrat-Medium.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "/usr/share/fonts/truetype/montserrat/Montserrat-Regular.ttf",
        "/usr/share/fonts/truetype/montserrat/Montserrat-Medium.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ],

    # Matches the grid-digit font priority used by COV-CL9-ANNUAL-ARENA-BLUE-001
    # in annual_arena_blue_multigrid_v1.py when drawing Sudoku digits with bold=False.
    # On Windows this resolves to Arial Regular first.
    "arena_blue_grid_regular": [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ],
}


def _font_candidates_for_face(face: str | None, *, bold: bool) -> list[str]:
    normalized = str(face or "").strip().lower().replace(" ", "_").replace("-", "_")

    candidates: list[str] = []
    if normalized:
        candidates.extend(_FONT_FACE_CANDIDATES.get(normalized, []))

        # Allow a direct font file path in JSON, for local/private production use.
        if normalized.endswith((".ttf", ".otf")):
            candidates.append(str(face))

    if bold:
        candidates.extend(_FONT_FACE_CANDIDATES["bold_sans"])
    else:
        candidates.extend(_FONT_FACE_CANDIDATES["regular_sans"])

    return candidates


def _font(
    size: int,
    bold: bool = True,
    face: str | None = None,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in _font_candidates_for_face(face, bold=bold):
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)

    return ImageFont.load_default()


def _text_bbox(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int, int, int]:
    return draw.textbbox((0, 0), text, font=font)


def _measure_text_with_tracking(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    font: ImageFont.ImageFont,
    tracking_px: int = 0,
) -> tuple[int, int, int, int]:
    if not text:
        return (0, 0, 0, 0)

    if tracking_px == 0 or len(text) <= 1:
        return draw.textbbox((0, 0), text, font=font)

    total_width = 0
    min_top = 0
    max_bottom = 0

    for index, char in enumerate(text):
        bbox = draw.textbbox((0, 0), char, font=font)
        char_width = bbox[2] - bbox[0]
        total_width += char_width

        if index < len(text) - 1:
            total_width += tracking_px

        min_top = min(min_top, bbox[1])
        max_bottom = max(max_bottom, bbox[3])

    return (0, min_top, total_width, max_bottom)


def _draw_text_with_tracking(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
    tracking_px: int = 0,
) -> None:
    if not text:
        return

    x, y = xy

    if tracking_px == 0 or len(text) <= 1:
        draw.text((x, y), text, font=font, fill=fill)
        return

    cursor_x = x
    for char in text:
        draw.text((cursor_x, y), char, font=font, fill=fill)
        bbox = draw.textbbox((0, 0), char, font=font)
        cursor_x += (bbox[2] - bbox[0]) + tracking_px


def _fit_font(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    max_width: int,
    max_height: int,
    start_size: int,
    min_size: int = 12,
    bold: bool = True,
    face: str | None = None,
    tracking_px: int = 0,
) -> ImageFont.ImageFont:
    size = max(start_size, min_size)

    while size >= min_size:
        font = _font(size, bold=bold, face=face)
        bbox = _measure_text_with_tracking(
            draw,
            text,
            font=font,
            tracking_px=tracking_px,
        )
        if bbox[2] - bbox[0] <= max_width and bbox[3] - bbox[1] <= max_height:
            return font
        size -= 2

    return _font(min_size, bold=bold, face=face)





def _draw_centered_text_with_shadow(
    img: Image.Image,
    *,
    text: str,
    box: tuple[int, int, int, int],
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
    shadow_fill: tuple[int, int, int, int] = (0, 0, 0, 175),
    shadow_offset: tuple[int, int] = (14, 16),
    shadow_blur: int = 8,
) -> None:
    if not text:
        return

    x0, y0, x1, y1 = box
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    bbox = sd.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = int(round(x0 + (x1 - x0 - tw) / 2 - bbox[0]))
    ty = int(round(y0 + (y1 - y0 - th) / 2 - bbox[1]))
    sd.text((tx + shadow_offset[0], ty + shadow_offset[1]), text, font=font, fill=shadow_fill)
    shadow = shadow.filter(ImageFilter.GaussianBlur(shadow_blur))
    img.alpha_composite(shadow)

    draw = ImageDraw.Draw(img)
    draw.text((tx, ty), text, font=font, fill=fill)



def _draw_centered_text_with_tracking_and_shadow(
    img: Image.Image,
    *,
    text: str,
    box: tuple[int, int, int, int],
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
    tracking_px: int = 0,
    shadow_fill: tuple[int, int, int, int] = (0, 0, 0, 175),
    shadow_offset: tuple[int, int] = (14, 16),
    shadow_blur: int = 8,
) -> None:
    if not text:
        return

    x0, y0, x1, y1 = box
    bbox = _measure_text_with_tracking(
        ImageDraw.Draw(Image.new("RGBA", (8, 8), (0, 0, 0, 0))),
        text,
        font=font,
        tracking_px=tracking_px,
    )
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = int(round(x0 + (x1 - x0 - tw) / 2 - bbox[0]))
    ty = int(round(y0 + (y1 - y0 - th) / 2 - bbox[1]))

    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    _draw_text_with_tracking(
        sd,
        (tx + shadow_offset[0], ty + shadow_offset[1]),
        text,
        font=font,
        fill=shadow_fill,
        tracking_px=tracking_px,
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(shadow_blur))
    img.alpha_composite(shadow)

    draw = ImageDraw.Draw(img)
    _draw_text_with_tracking(
        draw,
        (tx, ty),
        text,
        font=font,
        fill=fill,
        tracking_px=tracking_px,
    )

def _pair_ints(value: Any, fallback: tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return fallback
    return fallback


def _curve_number(curve: dict[str, Any], key: str, fallback: float) -> float:
    try:
        return float(curve.get(key, fallback))
    except (TypeError, ValueError):
        return fallback


def _curve_int(curve: dict[str, Any], key: str, fallback: int) -> int:
    try:
        return int(curve.get(key, fallback))
    except (TypeError, ValueError):
        return fallback


def _scale_native_x(value: float, width_px: int) -> int:
    return int(round(value * width_px / _CANONICAL_BG_WIDTH))


def _scale_native_y(value: float, height_px: int) -> int:
    return int(round(value * height_px / _CANONICAL_BG_HEIGHT))


def _warp_layer_to_title_curve(
    layer: Image.Image,
    *,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    amplitude_px: int,
    direction: str,
) -> Image.Image:
    """
    Bend a rendered title strip vertically so the text follows the curved
    title-band centerline. The curve is a smooth parabola: edges stay close
    to the original baseline, while the center receives the full displacement.
    """
    x0 = max(0, min(layer.width - 1, x0))
    x1 = max(x0 + 1, min(layer.width, x1))
    y0 = max(0, min(layer.height - 1, y0))
    y1 = max(y0 + 1, min(layer.height, y1))

    source = layer.crop((x0, y0, x1, y1))
    warped = Image.new("RGBA", layer.size, (0, 0, 0, 0))

    sign = 1 if str(direction).lower().strip() != "up" else -1
    width = source.width

    for ix in range(width):
        if width <= 1:
            u = 0.0
        else:
            u = (ix / float(width - 1)) * 2.0 - 1.0

        # Parabolic centerline: 0 at the edges, 1 at the center.
        curve_t = 1.0 - (u * u)
        dy = int(round(sign * amplitude_px * curve_t))

        column = source.crop((ix, 0, ix + 1, source.height))
        warped.alpha_composite(column, (x0 + ix, y0 + dy))

    return warped


def _draw_curved_centered_text_with_shadow(
    img: Image.Image,
    *,
    text: str,
    curve: dict[str, Any],
    fill: tuple[int, int, int, int],
    width_px: int,
    height_px: int,
    fallback_box: tuple[int, int, int, int],
    fallback_font_ratio: float,
) -> bool:
    """
    Draw a large display title along the Expert Gauge title-band curve.

    This is intentionally title-specific rather than a generic text primitive:
    the title band is part of the background artwork, so the renderer needs
    background-aware calibration values: x-limits, vertical band limits,
    centerline, arc amplitude, fitting width, and shadow behavior.

    Returns True when the curved title was drawn. Returns False so callers can
    fall back to the old rectangular renderer.
    """
    if not text:
        return False
    if not isinstance(curve, dict):
        return False
    if not bool(curve.get("enabled", False)):
        return False

    x0 = _scale_native_x(_curve_number(curve, "x0", 105), width_px)
    x1 = _scale_native_x(_curve_number(curve, "x1", 1480), width_px)
    band_y0 = _scale_native_y(_curve_number(curve, "band_y0", 485), height_px)
    band_y1 = _scale_native_y(_curve_number(curve, "band_y1", 690), height_px)
    center_y = _scale_native_y(_curve_number(curve, "center_y", 585), height_px)
    baseline_offset_y = _scale_native_y(_curve_number(curve, "baseline_offset_y", 0), height_px)
    amplitude_px = _scale_native_y(_curve_number(curve, "arc_amplitude", 30), height_px)

    if x1 <= x0 or band_y1 <= band_y0:
        return False

    max_text_width_ratio = _curve_number(curve, "max_text_width_ratio", 0.985)
    max_text_height_ratio = _curve_number(curve, "max_text_height_ratio", 0.120)
    font_height_ratio = _curve_number(curve, "font_height_ratio", fallback_font_ratio)
    font_face = str(curve.get("font_face") or "annual_expert_gauge_title").strip()

    tracking_native_px = _curve_number(curve, "tracking_px", 0)
    tracking_px = _scale_native_x(tracking_native_px, width_px)

    target_width = max(1, int(round((x1 - x0) * max_text_width_ratio)))
    target_height = max(1, int(round(height_px * max_text_height_ratio)))
    start_size = max(
        _curve_int(curve, "min_font_size", 72),
        int(round(height_px * font_height_ratio)),
    )
    min_size = _curve_int(curve, "min_font_size", 72)

    scratch = Image.new("RGBA", (max(32, target_width), max(32, target_height)), (0, 0, 0, 0))
    scratch_draw = ImageDraw.Draw(scratch)
    font = _fit_font(
        scratch_draw,
        text,
        max_width=target_width,
        max_height=target_height,
        start_size=start_size,
        min_size=min_size,
        bold=True,
        face=font_face,
        tracking_px=tracking_px,
    )

    bbox = _measure_text_with_tracking(
        scratch_draw,
        text,
        font=font,
        tracking_px=tracking_px,
    )
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    tx = int(round(x0 + (x1 - x0 - tw) / 2 - bbox[0]))
    ty = int(round(center_y + baseline_offset_y - th / 2 - bbox[1]))

    shadow_offset_native = _pair_ints(curve.get("shadow_offset"), (14, 16))
    shadow_offset = (
        _scale_native_x(shadow_offset_native[0], width_px),
        _scale_native_y(shadow_offset_native[1], height_px),
    )
    shadow_alpha = max(0, min(255, _curve_int(curve, "shadow_alpha", 175)))
    shadow_blur = max(0, _curve_int(curve, "shadow_blur", 8))
    shadow_fill = (0, 0, 0, shadow_alpha)

    flat = Image.new("RGBA", img.size, (0, 0, 0, 0))

    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    _draw_text_with_tracking(
        sd,
        (tx + shadow_offset[0], ty + shadow_offset[1]),
        text,
        font=font,
        fill=shadow_fill,
        tracking_px=tracking_px,
    )
    if shadow_blur:
        shadow = shadow.filter(ImageFilter.GaussianBlur(shadow_blur))
    flat.alpha_composite(shadow)

    fd = ImageDraw.Draw(flat)
    _draw_text_with_tracking(
        fd,
        (tx, ty),
        text,
        font=font,
        fill=fill,
        tracking_px=tracking_px,
    )

    # Expand vertical crop so the warped title and shadow do not clip.
    pad_y = max(abs(amplitude_px) + shadow_blur * 3 + abs(shadow_offset[1]), 12)
    warp_y0 = max(0, band_y0 - pad_y)
    warp_y1 = min(height_px, band_y1 + pad_y)

    warped = _warp_layer_to_title_curve(
        flat,
        x0=x0,
        x1=x1,
        y0=warp_y0,
        y1=warp_y1,
        amplitude_px=amplitude_px,
        direction=str(curve.get("arc_direction", "down")),
    )
    img.alpha_composite(warped)
    return True


def _hex_to_rgba(value: str, fallback: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    if not isinstance(value, str):
        return fallback

    raw = value.strip().lstrip("#")
    if len(raw) != 6:
        return fallback

    try:
        return (
            int(raw[0:2], 16),
            int(raw[2:4], 16),
            int(raw[4:6], 16),
            255,
        )
    except ValueError:
        return fallback


def _get_text_variables(context: ResolvedCoverDesignContext) -> dict[str, Any]:
    return dict(context.variables.get("text", {}))


def _get_background_variables(context: ResolvedCoverDesignContext) -> dict[str, Any]:
    return dict(context.variables.get("background", {}))


def _get_palette_variables(context: ResolvedCoverDesignContext) -> dict[str, Any]:
    return dict(context.variables.get("palette", {}))


def _get_typography_variables(context: ResolvedCoverDesignContext) -> dict[str, Any]:
    return dict(context.variables.get("typography", {}))


def _get_feature_variables(context: ResolvedCoverDesignContext) -> dict[str, Any]:
    return dict(context.variables.get("features", {}))


def _get_layout_calibration_variables(context: ResolvedCoverDesignContext) -> dict[str, Any]:
    return dict(context.variables.get("layout_calibration", {}))


def _feature_enabled(features: dict[str, Any], key: str, default: bool = True) -> bool:
    return bool(features.get(key, default))


def _get_resolved_puzzle_art_variables(context: ResolvedCoverDesignContext) -> dict[str, Any]:
    return dict(context.variables.get("resolved_puzzle_art", {}))


def _resolve_asset_path(context: ResolvedCoverDesignContext, asset_path: str) -> Path:
    raw = Path(asset_path)
    if raw.is_absolute() and raw.exists():
        return raw

    candidates: list[Path] = []
    candidates.append(Path.cwd() / raw)
    candidates.append(context.design_dir / raw)

    for parent in context.design_dir.parents:
        candidates.append(parent / raw)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    checked = "\n".join(f"  - {candidate}" for candidate in candidates[:12])
    raise FileNotFoundError(
        f"Could not find Expert Gauge background asset {asset_path!r}. Checked:\n{checked}"
    )


def _scale_box(
    box: tuple[int, int, int, int],
    width_px: int,
    height_px: int,
) -> tuple[int, int, int, int]:
    sx = width_px / _CANONICAL_BG_WIDTH
    sy = height_px / _CANONICAL_BG_HEIGHT
    x0, y0, x1, y1 = box
    return (
        int(round(x0 * sx)),
        int(round(y0 * sy)),
        int(round(x1 * sx)),
        int(round(y1 * sy)),
    )


def _scale_quad(
    quad: list[tuple[int, int]],
    width_px: int,
    height_px: int,
) -> list[tuple[int, int]]:
    sx = width_px / _CANONICAL_BG_WIDTH
    sy = height_px / _CANONICAL_BG_HEIGHT
    return [(int(round(x * sx)), int(round(y * sy))) for x, y in quad]


def _as_native_box(value: Any, fallback: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            return (
                int(value[0]),
                int(value[1]),
                int(value[2]),
                int(value[3]),
            )
        except (TypeError, ValueError):
            return fallback

    return fallback


def _as_native_quad(
    value: Any,
    fallback: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    if isinstance(value, (list, tuple)) and len(value) == 4:
        points: list[tuple[int, int]] = []
        for point in value:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                return fallback
            try:
                points.append((int(point[0]), int(point[1])))
            except (TypeError, ValueError):
                return fallback
        return points

    return fallback


def _layout_number(front_calibration: dict[str, Any], key: str, fallback: float) -> float:
    try:
        return float(front_calibration.get(key, fallback))
    except (TypeError, ValueError):
        return fallback


def _resolve_page_count(
    context: ResolvedCoverDesignContext,
    geometry: Any,
) -> int:
    """
    Resolve publication/interior page count from whichever source is populated
    in the export pipeline.
    """
    candidates = [
        getattr(geometry, "page_count", None),
        getattr(geometry, "interior_page_count", None),
        getattr(geometry, "num_pages", None),
        getattr(geometry, "total_pages", None),
        getattr(geometry, "page_count_total", None),
        dict(context.variables.get("publication", {})).get("page_count"),
        dict(context.variables.get("book", {})).get("page_count"),
        context.variables.get("page_count"),
    ]

    for candidate in candidates:
        try:
            value = int(candidate)
        except (TypeError, ValueError):
            continue

        if value > 0:
            return value

    return 0

def _layout_string(front_calibration: dict[str, Any], key: str, fallback: str) -> str:
    value = front_calibration.get(key, fallback)
    if value is None:
        return fallback

    text = str(value).strip()
    return text if text else fallback

def _layout_int_list(
    front_calibration: dict[str, Any],
    key: str,
    fallback: list[int],
    *,
    min_value: int,
    max_value: int,
) -> list[int]:
    raw = front_calibration.get(key, fallback)

    if not isinstance(raw, (list, tuple)):
        return list(fallback)

    values: list[int] = []
    for item in raw:
        try:
            value = int(item)
        except (TypeError, ValueError):
            return list(fallback)

        if value < min_value or value > max_value:
            return list(fallback)

        values.append(value)

    if not values:
        return list(fallback)

    # Preserve order, remove duplicates.
    deduped: list[int] = []
    seen: set[int] = set()
    for value in values:
        if value in seen:
            continue
        deduped.append(value)
        seen.add(value)

    return deduped


def _layout_box(
    front_calibration: dict[str, Any],
    key: str,
    fallback: tuple[int, int, int, int],
    width_px: int,
    height_px: int,
) -> tuple[int, int, int, int]:
    return _scale_box(
        _as_native_box(front_calibration.get(key), fallback),
        width_px,
        height_px,
    )


def _layout_quad(
    front_calibration: dict[str, Any],
    key: str,
    fallback: list[tuple[int, int]],
    width_px: int,
    height_px: int,
) -> list[tuple[int, int]]:
    return _scale_quad(
        _as_native_quad(front_calibration.get(key), fallback),
        width_px,
        height_px,
    )


def _front_layout_calibration(context: ResolvedCoverDesignContext) -> dict[str, Any]:
    calibration = _get_layout_calibration_variables(context)
    front = calibration.get("front", {})
    return dict(front) if isinstance(front, dict) else {}


def _back_layout_calibration(context: ResolvedCoverDesignContext) -> dict[str, Any]:
    calibration = _get_layout_calibration_variables(context)
    back = calibration.get("back", {})
    return dict(back) if isinstance(back, dict) else {}


def _spine_layout_calibration(context: ResolvedCoverDesignContext) -> dict[str, Any]:
    calibration = _get_layout_calibration_variables(context)
    spine = calibration.get("spine", {})
    return dict(spine) if isinstance(spine, dict) else {}

def _resolved_front_text_curve(
    front_calibration: dict[str, Any],
    typography: dict[str, Any],
    *,
    curve_key: str,
    font_face_key: str,
    font_role_key: str,
    tracking_key: str,
    default_face: str,
    legacy_font_face_key: str | None = None,
    legacy_font_role_key: str | None = None,
    legacy_tracking_key: str | None = None,
) -> dict[str, Any]:
    raw = front_calibration.get(curve_key, {})
    curve = dict(raw) if isinstance(raw, dict) else {}

    if isinstance(typography, dict):
        font_face_value = (
            typography.get(font_face_key)
            or (typography.get(legacy_font_face_key) if legacy_font_face_key else None)
            or typography.get(font_role_key)
            or (typography.get(legacy_font_role_key) if legacy_font_role_key else None)
            or default_face
        )
        curve.setdefault("font_face", font_face_value)

        tracking_value = typography.get(tracking_key, None)
        if tracking_value is None and legacy_tracking_key:
            tracking_value = typography.get(legacy_tracking_key, None)
        if tracking_value is None:
            tracking_value = 0
        curve.setdefault("tracking_px", tracking_value)

    return curve


def _draw_front_display_text(
    img: Image.Image,
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    box: tuple[int, int, int, int],
    curve: dict[str, Any],
    fill: tuple[int, int, int, int],
    width_px: int,
    height_px: int,
    fallback_font_ratio: float,
    fallback_min_size: int,
    fallback_shadow_offset: tuple[int, int],
    fallback_shadow_blur: int,
) -> None:
    if not text:
        return

    curved_drawn = _draw_curved_centered_text_with_shadow(
        img,
        text=text,
        curve=curve,
        fill=fill,
        width_px=width_px,
        height_px=height_px,
        fallback_box=box,
        fallback_font_ratio=fallback_font_ratio,
    )
    if curved_drawn:
        return

    font_face = str(curve.get("font_face") or "montserrat_bold").strip()

    tracking_px = _scale_native_x(
        _curve_number(curve, "tracking_px", 0),
        width_px,
    )

    font_height_ratio = _curve_number(curve, "font_height_ratio", fallback_font_ratio)
    min_size = _curve_int(curve, "min_font_size", fallback_min_size)
    start_size = max(min_size, int(round(height_px * font_height_ratio)))

    font = _fit_font(
        draw,
        text,
        max_width=max(1, box[2] - box[0]),
        max_height=max(1, box[3] - box[1]),
        start_size=start_size,
        min_size=min_size,
        bold=True,
        face=font_face,
        tracking_px=tracking_px,
    )

    shadow_offset_native = _pair_ints(curve.get("shadow_offset"), fallback_shadow_offset)
    shadow_offset = (
        _scale_native_x(shadow_offset_native[0], width_px),
        _scale_native_y(shadow_offset_native[1], height_px),
    )
    shadow_blur = max(0, _curve_int(curve, "shadow_blur", fallback_shadow_blur))
    shadow_alpha = max(0, min(255, _curve_int(curve, "shadow_alpha", 175)))

    _draw_centered_text_with_tracking_and_shadow(
        img,
        text=text,
        box=box,
        font=font,
        fill=fill,
        tracking_px=tracking_px,
        shadow_fill=(0, 0, 0, shadow_alpha),
        shadow_offset=shadow_offset,
        shadow_blur=shadow_blur,
    )

def _as_ratio_box(
    value: Any,
    fallback: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            return (
                float(value[0]),
                float(value[1]),
                float(value[2]),
                float(value[3]),
            )
        except (TypeError, ValueError):
            return fallback

    return fallback


def _ratio_box(
    calibration: dict[str, Any],
    key: str,
    fallback: tuple[float, float, float, float],
    width_px: int,
    height_px: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = _as_ratio_box(calibration.get(key), fallback)
    return (
        int(round(width_px * x0)),
        int(round(height_px * y0)),
        int(round(width_px * x1)),
        int(round(height_px * y1)),
    )


def _normalize_givens(givens81: str, fallback: str) -> str:
    givens = str(givens81 or "").strip()
    if len(givens) == 81:
        return givens
    return fallback


def _draw_givens_in_grid_box(
    img: Image.Image,
    *,
    givens81: str,
    box: tuple[int, int, int, int],
    digit_fill: tuple[int, int, int, int],
    digit_scale: float,
    bold: bool = False,
    font_face: str | None = None,
) -> None:
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = box
    cell_w = (x1 - x0) / 9.0
    cell_h = (y1 - y0) / 9.0
    font_size = max(18, int(min(cell_w, cell_h) * digit_scale))
    font = _font(font_size, bold=bold, face=font_face)

    for idx, ch in enumerate(givens81):
        if ch in ("0", ".", "-", " "):
            continue

        row = idx // 9
        col = idx % 9
        cx = x0 + (col + 0.5) * cell_w
        cy = y0 + (row + 0.5) * cell_h

        bbox = draw.textbbox((0, 0), ch, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = int(round(cx - tw / 2 - bbox[0]))
        ty = int(round(cy - th / 2 - bbox[1]))

        draw.text((tx, ty), ch, font=font, fill=digit_fill)


def _find_perspective_coeffs(
    src: list[tuple[float, float]],
    dst: list[tuple[float, float]],
) -> list[float]:
    import numpy as np

    matrix = []
    vector = []

    for (x_src, y_src), (x_dst, y_dst) in zip(src, dst):
        matrix.append([x_dst, y_dst, 1, 0, 0, 0, -x_src * x_dst, -x_src * y_dst])
        matrix.append([0, 0, 0, x_dst, y_dst, 1, -y_src * x_dst, -y_src * y_dst])
        vector.append(x_src)
        vector.append(y_src)

    coeffs = np.linalg.solve(
        np.array(matrix, dtype=float),
        np.array(vector, dtype=float),
    )
    return coeffs.tolist()


def _perspective_warp_quad(
    source: Image.Image,
    dst_quad: list[tuple[int, int]],
    canvas_size: tuple[int, int],
) -> Image.Image:
    """
    Warp a transparent source layer into a destination quadrilateral.

    Important:
    This function must preserve the source alpha channel.

    The previous implementation did this:

        out.alpha_composite(warped)
        out.putalpha(mask)

    That made every pixel inside the destination polygon fully opaque,
    including pixels that were transparent in the source layer. Since those
    transparent pixels had black RGB values, the side Sudoku overlays became
    solid black trapezoids.

    Correct behavior:
    - warp the source layer
    - keep only the warped source pixels that were actually non-transparent
    - use the polygon mask only as a clip
    """
    source_rgba = source.convert("RGBA")
    src_w, src_h = source_rgba.size
    src_quad = [
        (0.0, 0.0),
        (float(src_w), 0.0),
        (float(src_w), float(src_h)),
        (0.0, float(src_h)),
    ]

    coeffs = _find_perspective_coeffs(
        src_quad,
        [(float(x), float(y)) for x, y in dst_quad],
    )

    warped = source_rgba.transform(
        canvas_size,
        Image.Transform.PERSPECTIVE,
        coeffs,
        Image.Resampling.BICUBIC,
    ).convert("RGBA")

    clip_mask = Image.new("L", canvas_size, 0)
    md = ImageDraw.Draw(clip_mask)
    md.polygon(dst_quad, fill=255)

    # Preserve the warped source alpha, then clip it to the destination quad.
    # This prevents transparent source pixels from becoming opaque black pixels.
    source_alpha = warped.getchannel("A")
    clipped_alpha = ImageChops.multiply(source_alpha, clip_mask)

    out = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    out.alpha_composite(warped)
    out.putalpha(clipped_alpha)
    return out


def _make_side_visible_digit_layer(
    *,
    source_height: int,
    givens81: str,
    visible_cols: list[int],
    digit_fill: tuple[int, int, int, int],
    digit_scale: float = 0.50,
    font_face: str | None = None,
) -> Image.Image:
    """
    Build a transparent side-grid digit layer for only the visible columns.

    Expert Gauge is a background-composite cover. The purple side grids are
    already painted into the background artwork. The renderer should therefore
    draw only the digits that belong to the visible slice of each side grid.

    Example:
    - left side:  visible_cols = [0, 1, 2]
    - right side: visible_cols = [6, 7, 8]

    This prevents the old behavior where a full 9x9 digit field was squeezed
    into a narrow 3-column visible side panel.
    """
    cols = max(1, len(visible_cols))
    rows = 9

    # Keep source cells roughly square before perspective warping.
    cell_size = max(1, int(round(source_height / rows)))
    source_width = max(1, cell_size * cols)
    layer = Image.new("RGBA", (source_width, source_height), (0, 0, 0, 0))

    draw = ImageDraw.Draw(layer)
    cell_w = source_width / float(cols)
    cell_h = source_height / float(rows)
    font_size = max(12, int(min(cell_w, cell_h) * digit_scale))
    font = _font(font_size, bold=False, face=font_face)

    for row in range(rows):
        for visible_col_index, source_col in enumerate(visible_cols):
            idx = row * 9 + source_col
            if idx < 0 or idx >= len(givens81):
                continue

            ch = givens81[idx]
            if ch in ("0", ".", "-", " "):
                continue

            cx = (visible_col_index + 0.5) * cell_w
            cy = (row + 0.5) * cell_h

            bbox = draw.textbbox((0, 0), ch, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            tx = int(round(cx - tw / 2 - bbox[0]))
            ty = int(round(cy - th / 2 - bbox[1]))

            draw.text((tx, ty), ch, font=font, fill=digit_fill)

    # Soft halo improves legibility on the purple side panels without drawing grid lines.
    alpha = layer.getchannel("A")
    halo = Image.new("RGBA", layer.size, (255, 255, 255, 0))
    halo.putalpha(alpha.filter(ImageFilter.GaussianBlur(2)))

    return Image.alpha_composite(halo, layer)




def _quad_interp(
    quad: list[tuple[int, int]],
    *,
    u: float,
    v: float,
) -> tuple[int, int]:
    """
    Bilinear interpolation inside a quadrilateral.

    quad order is expected to be:
    top-left, top-right, bottom-right, bottom-left.
    """
    tl, tr, br, bl = quad

    top_x = tl[0] + (tr[0] - tl[0]) * u
    top_y = tl[1] + (tr[1] - tl[1]) * u
    bottom_x = bl[0] + (br[0] - bl[0]) * u
    bottom_y = bl[1] + (br[1] - bl[1]) * u

    x = top_x + (bottom_x - top_x) * v
    y = top_y + (bottom_y - top_y) * v
    return int(round(x)), int(round(y))


def _draw_debug_dot(
    draw: ImageDraw.ImageDraw,
    *,
    center: tuple[int, int],
    radius: int,
    fill: tuple[int, int, int, int],
    outline: tuple[int, int, int, int] | None = None,
) -> None:
    x, y = center
    box = [x - radius, y - radius, x + radius, y + radius]
    draw.ellipse(box, fill=fill, outline=outline)


def _draw_debug_label(
    draw: ImageDraw.ImageDraw,
    *,
    xy: tuple[int, int],
    text: str,
    fill: tuple[int, int, int, int],
) -> None:
    font = _font(18, bold=True)
    x, y = xy
    draw.text((x + 4, y + 4), text, font=font, fill=(0, 0, 0, 190))
    draw.text((x, y), text, font=font, fill=fill)


def _draw_rect_grid_debug_overlay(
    img: Image.Image,
    *,
    box: tuple[int, int, int, int],
    rows: int,
    cols: int,
    label: str,
    fill: tuple[int, int, int, int],
) -> None:
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = box
    line_w = max(2, int(round(min(img.size) * 0.002)))

    draw.rectangle([x0, y0, x1, y1], outline=fill, width=line_w)
    _draw_debug_label(draw, xy=(x0, max(0, y0 - 28)), text=label, fill=fill)

    cell_w = (x1 - x0) / float(cols)
    cell_h = (y1 - y0) / float(rows)
    dot_radius = max(3, int(round(min(cell_w, cell_h) * 0.045)))

    for row in range(rows):
        for col in range(cols):
            cx = int(round(x0 + (col + 0.5) * cell_w))
            cy = int(round(y0 + (row + 0.5) * cell_h))
            _draw_debug_dot(
                draw,
                center=(cx, cy),
                radius=dot_radius,
                fill=fill,
                outline=(0, 0, 0, 210),
            )


def _draw_quad_grid_debug_overlay(
    img: Image.Image,
    *,
    quad: list[tuple[int, int]],
    rows: int,
    cols: int,
    label: str,
    fill: tuple[int, int, int, int],
) -> None:
    draw = ImageDraw.Draw(img)
    line_w = max(2, int(round(min(img.size) * 0.002)))

    polygon = [quad[0], quad[1], quad[2], quad[3], quad[0]]
    draw.line(polygon, fill=fill, width=line_w)

    label_anchor = quad[0]
    _draw_debug_label(
        draw,
        xy=(label_anchor[0], max(0, label_anchor[1] - 28)),
        text=label,
        fill=fill,
    )

    # Draw temporary centers according to the current Phase 1 full-9x9 mapping.
    # Phase 2 will replace this with a visible-3-column mapping.
    dot_radius = max(2, int(round(min(img.size) * 0.0035)))

    for row in range(rows):
        for col in range(cols):
            u = (col + 0.5) / float(cols)
            v = (row + 0.5) / float(rows)
            cx, cy = _quad_interp(quad, u=u, v=v)
            _draw_debug_dot(
                draw,
                center=(cx, cy),
                radius=dot_radius,
                fill=fill,
                outline=(0, 0, 0, 210),
            )

def _draw_quad_outline_debug_overlay(
    img: Image.Image,
    *,
    quad: list[tuple[int, int]],
    label: str,
    fill: tuple[int, int, int, int],
) -> None:
    draw = ImageDraw.Draw(img)
    line_w = max(2, int(round(min(img.size) * 0.002)))

    polygon = [quad[0], quad[1], quad[2], quad[3], quad[0]]
    draw.line(polygon, fill=fill, width=line_w)

    label_anchor = quad[0]
    _draw_debug_label(
        draw,
        xy=(label_anchor[0], max(0, label_anchor[1] - 28)),
        text=label,
        fill=fill,
    )

def _draw_front_grid_debug_overlay(
    img: Image.Image,
    *,
    hero_grid_box: tuple[int, int, int, int],
    left_visible_quad: list[tuple[int, int]],
    right_visible_quad: list[tuple[int, int]],
    left_digit_quad: list[tuple[int, int]],
    right_digit_quad: list[tuple[int, int]],
    left_visible_cols: list[int],
    right_visible_cols: list[int],
) -> None:
    """
    Optional calibration overlay for the Expert Gauge front cover.

    This is intentionally off by default. Enable it temporarily with either:

        variables.features.front_debug_overlay_enabled = true

    or:

        variables.layout_calibration.front.debug_overlay_enabled = true

    Do not enable it for production exports.
    """
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))

    _draw_rect_grid_debug_overlay(
        overlay,
        box=hero_grid_box,
        rows=9,
        cols=9,
        label="DEBUG HERO 9x9",
        fill=(255, 40, 40, 210),
    )

    # Show the visible panel boundary as a reference.
    _draw_quad_outline_debug_overlay(
        overlay,
        quad=left_visible_quad,
        label="DEBUG LEFT VISIBLE PANEL",
        fill=(255, 220, 40, 180),
    )
    _draw_quad_outline_debug_overlay(
        overlay,
        quad=right_visible_quad,
        label="DEBUG RIGHT VISIBLE PANEL",
        fill=(255, 220, 40, 180),
    )

    # Show the actual digit-placement lattice.
    _draw_quad_grid_debug_overlay(
        overlay,
        quad=left_digit_quad,
        rows=9,
        cols=max(1, len(left_visible_cols)),
        label=f"DEBUG LEFT DIGIT QUAD COLS {left_visible_cols}",
        fill=(40, 255, 120, 210),
    )
    _draw_quad_grid_debug_overlay(
        overlay,
        quad=right_digit_quad,
        rows=9,
        cols=max(1, len(right_visible_cols)),
        label=f"DEBUG RIGHT DIGIT QUAD COLS {right_visible_cols}",
        fill=(40, 160, 255, 210),
    )

    img.alpha_composite(overlay)

def _lerp(a: int, b: int, t: float) -> int:
    return int(round(a + (b - a) * t))


def _draw_vertical_gradient(
    img: Image.Image,
    *,
    top: tuple[int, int, int, int],
    bottom: tuple[int, int, int, int],
) -> None:
    w, h = img.size
    grad = Image.new("RGBA", (1, h), (0, 0, 0, 0))
    gd = ImageDraw.Draw(grad)

    for y in range(h):
        t = y / max(1, h - 1)
        gd.point(
            (0, y),
            fill=(
                _lerp(top[0], bottom[0], t),
                _lerp(top[1], bottom[1], t),
                _lerp(top[2], bottom[2], t),
                _lerp(top[3], bottom[3], t),
            ),
        )

    img.alpha_composite(grad.resize((w, h), Image.Resampling.BICUBIC))


def _draw_expert_network_texture(
    img: Image.Image,
    *,
    gold: tuple[int, int, int, int],
    purple: tuple[int, int, int, int],
    enabled: bool = True,
) -> None:
    if not enabled:
        return

    w, h = img.size
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    points = [
        (int(w * 0.08), int(h * 0.18)),
        (int(w * 0.18), int(h * 0.12)),
        (int(w * 0.31), int(h * 0.22)),
        (int(w * 0.44), int(h * 0.15)),
        (int(w * 0.59), int(h * 0.25)),
        (int(w * 0.74), int(h * 0.16)),
        (int(w * 0.89), int(h * 0.22)),
        (int(w * 0.12), int(h * 0.42)),
        (int(w * 0.28), int(h * 0.36)),
        (int(w * 0.49), int(h * 0.46)),
        (int(w * 0.69), int(h * 0.39)),
        (int(w * 0.86), int(h * 0.48)),
        (int(w * 0.16), int(h * 0.70)),
        (int(w * 0.35), int(h * 0.62)),
        (int(w * 0.55), int(h * 0.73)),
        (int(w * 0.78), int(h * 0.66)),
        (int(w * 0.92), int(h * 0.78)),
    ]

    line_fill = (purple[0], purple[1], purple[2], 90)
    gold_fill = (gold[0], gold[1], gold[2], 150)
    glow_fill = (gold[0], gold[1], gold[2], 60)

    links = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
        (0, 7), (2, 8), (4, 9), (6, 11),
        (7, 8), (8, 9), (9, 10), (10, 11),
        (7, 12), (8, 13), (9, 14), (10, 15), (11, 16),
        (12, 13), (13, 14), (14, 15), (15, 16),
    ]

    for a, b in links:
        d.line([points[a], points[b]], fill=line_fill, width=max(1, int(w * 0.003)))

    for x, y in points:
        r = max(3, int(w * 0.006))
        d.ellipse([x - r * 3, y - r * 3, x + r * 3, y + r * 3], fill=glow_fill)
        d.ellipse([x - r, y - r, x + r, y + r], fill=gold_fill)

    layer = layer.filter(ImageFilter.GaussianBlur(0.4))
    img.alpha_composite(layer)


def _draw_gold_frame(
    img: Image.Image,
    *,
    gold: tuple[int, int, int, int],
    gold_dark: tuple[int, int, int, int],
    margin: int,
    width: int,
) -> None:
    d = ImageDraw.Draw(img)

    for i in range(width):
        t = i / max(1, width - 1)
        fill = (
            _lerp(gold[0], gold_dark[0], t),
            _lerp(gold[1], gold_dark[1], t),
            _lerp(gold[2], gold_dark[2], t),
            255,
        )
        d.rounded_rectangle(
            [
                margin + i,
                margin + i,
                img.size[0] - margin - i,
                img.size[1] - margin - i,
            ],
            radius=max(12, int(img.size[0] * 0.025)),
            outline=fill,
            width=1,
        )


def _wrap_text_to_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    words = str(text or "").split()
    if not words:
        return []

    lines: list[str] = []
    current = words[0]

    for word in words[1:]:
        candidate = f"{current} {word}"
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if (bbox[2] - bbox[0]) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word

    lines.append(current)
    return lines


def _draw_wrapped_text(
    img: Image.Image,
    *,
    text: str,
    box: tuple[int, int, int, int],
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
    line_spacing: float = 1.15,
    align: str = "center",
) -> None:
    if not text:
        return

    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = box
    max_width = max(1, x1 - x0)
    max_height = max(1, y1 - y0)
    lines = _wrap_text_to_width(d, text, font, max_width)

    if not lines:
        return

    line_heights: list[int] = []
    line_widths: list[int] = []

    for line in lines:
        bbox = d.textbbox((0, 0), line, font=font)
        line_widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])

    step = int(max(line_heights) * line_spacing)
    total_h = step * (len(lines) - 1) + max(line_heights)

    # If text overflows vertically, clip by only drawing lines that fit.
    start_y = int(round(y0 + max(0, (max_height - total_h) / 2)))

    y = start_y
    for line, lw in zip(lines, line_widths):
        if y > y1:
            break

        if align == "left":
            x = x0
        elif align == "right":
            x = x1 - lw
        else:
            x = int(round(x0 + (max_width - lw) / 2))

        if y + max(line_heights) <= y1:
            d.text((x, y), line, font=font, fill=fill)

        y += step


def _draw_feature_card(
    img: Image.Image,
    *,
    box: tuple[int, int, int, int],
    label: str,
    body: str,
    gold: tuple[int, int, int, int],
    white: tuple[int, int, int, int],
    purple: tuple[int, int, int, int],
) -> None:
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = box

    card_fill = (8, 18, 55, 178)
    border_fill = (gold[0], gold[1], gold[2], 210)
    d.rounded_rectangle([x0, y0, x1, y1], radius=22, fill=card_fill, outline=border_fill, width=3)

    label_font = _fit_font(
        d,
        label,
        max_width=max(1, x1 - x0 - 42),
        max_height=max(1, int((y1 - y0) * 0.25)),
        start_size=max(28, int((y1 - y0) * 0.18)),
        min_size=20,
        bold=True,
    )
    body_font = _font(max(22, int((y1 - y0) * 0.105)), bold=False)

    d.rounded_rectangle(
        [x0 + 22, y0 + 18, x1 - 22, y0 + 18 + int((y1 - y0) * 0.25)],
        radius=14,
        fill=(purple[0], purple[1], purple[2], 120),
        outline=(gold[0], gold[1], gold[2], 135),
        width=1,
    )

    _draw_centered_text_with_shadow(
        img,
        text=label,
        box=(x0 + 26, y0 + 18, x1 - 26, y0 + 18 + int((y1 - y0) * 0.25)),
        font=label_font,
        fill=gold,
        shadow_offset=(2, 3),
        shadow_blur=2,
    )

    _draw_wrapped_text(
        img,
        text=body,
        box=(x0 + 28, y0 + int((y1 - y0) * 0.36), x1 - 28, y1 - 24),
        font=body_font,
        fill=white,
        line_spacing=1.20,
        align="center",
    )


def _draw_mini_sudoku_tile(
    img: Image.Image,
    *,
    box: tuple[int, int, int, int],
    givens81: str,
    gold: tuple[int, int, int, int],
) -> None:
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = box
    size = min(x1 - x0, y1 - y0)
    x0 = int(round(x0 + ((x1 - x0) - size) / 2))
    y0 = int(round(y0 + ((y1 - y0) - size) / 2))
    x1 = x0 + size
    y1 = y0 + size

    d.rounded_rectangle(
        [x0 - 10, y0 - 10, x1 + 10, y1 + 10],
        radius=18,
        fill=(255, 255, 255, 245),
        outline=gold,
        width=4,
    )

    cell = size / 9.0
    for i in range(10):
        width = 4 if i % 3 == 0 else 1
        x = int(round(x0 + i * cell))
        y = int(round(y0 + i * cell))
        d.line([(x, y0), (x, y1)], fill=(0, 0, 0, 210), width=width)
        d.line([(x0, y), (x1, y)], fill=(0, 0, 0, 210), width=width)

    _draw_givens_in_grid_box(
        img,
        givens81=givens81,
        box=(x0, y0, x1, y1),
        digit_fill=(0, 0, 0, 235),
        digit_scale=0.48,
        bold=False,
    )


def _draw_rotated_text_centered(
    img: Image.Image,
    *,
    text: str,
    center: tuple[int, int],
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
    angle_degrees: float,
) -> None:
    if not text:
        return

    temp = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(temp)
    bbox = d.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = int(round(center[0] - tw / 2 - bbox[0]))
    ty = int(round(center[1] - th / 2 - bbox[1]))

    d.text((tx + 5, ty + 6), text, font=font, fill=(0, 0, 0, 150))
    d.text((tx, ty), text, font=font, fill=fill)

    rotated = temp.rotate(angle_degrees, resample=Image.Resampling.BICUBIC, center=center)
    img.alpha_composite(rotated)


class AnnualExpertGaugeBackgroundV1Renderer(BaseCoverRenderer):
    renderer_key = "annual_expert_gauge_background_v1"

    def render_back_cover(
        self,
        context: ResolvedCoverDesignContext,
        out_dir: str | Path,
        geometry: Any,
        width_px: int = 2550,
        height_px: int = 3300,
    ) -> Path | None:
        features = _get_feature_variables(context)
        if not _feature_enabled(features, "custom_back_enabled", True):
            return None

        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        palette = _get_palette_variables(context)
        back = dict(context.variables.get("back", {}))
        barcode = dict(context.variables.get("barcode", {}))
        resolved_puzzle_art = _get_resolved_puzzle_art_variables(context)

        navy = _hex_to_rgba(str(palette.get("navy", "#061B49")), (6, 27, 73, 255))
        navy_dark = _hex_to_rgba(str(palette.get("navy_dark", "#020B22")), (2, 11, 34, 255))
        purple = _hex_to_rgba(str(palette.get("purple", "#2A075F")), (42, 7, 95, 255))
        purple_dark = _hex_to_rgba(str(palette.get("purple_dark", "#14022F")), (20, 2, 47, 255))
        gold = _hex_to_rgba(str(palette.get("gold", "#F4C451")), (244, 196, 81, 255))
        gold_dark = _hex_to_rgba(str(palette.get("gold_dark", "#A66A05")), (166, 106, 5, 255))
        white = _hex_to_rgba(str(palette.get("white", "#FFFFFF")), (255, 255, 255, 255))

        img = Image.new("RGBA", (width_px, height_px), navy_dark)
        _draw_vertical_gradient(img, top=navy_dark, bottom=purple_dark)

        back_calibration = _back_layout_calibration(context)

        # Soft central glow.
        glow_box = _ratio_box(
            back_calibration,
            "glow_box",
            (-0.10, 0.12, 1.10, 0.82),
            width_px,
            height_px,
        )
        glow_alpha = int(_layout_number(back_calibration, "glow_alpha", 100))
        glow_blur = int(
            round(width_px * _layout_number(back_calibration, "glow_blur_width_ratio", 0.08))
        )

        glow = Image.new("RGBA", img.size, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        gd.ellipse(
            list(glow_box),
            fill=(purple[0], purple[1], purple[2], glow_alpha),
        )
        glow = glow.filter(ImageFilter.GaussianBlur(glow_blur))
        img.alpha_composite(glow)

        _draw_expert_network_texture(
            img,
            gold=gold,
            purple=purple,
            enabled=_feature_enabled(features, "network_texture_enabled", True),
        )

        if _feature_enabled(features, "gold_frame_enabled", True):
            _draw_gold_frame(
                img,
                gold=gold,
                gold_dark=gold_dark,
                margin=int(
                    round(width_px * _layout_number(back_calibration, "gold_frame_margin_ratio", 0.045))
                ),
                width=max(
                    5,
                    int(
                        round(
                            width_px
                            * _layout_number(back_calibration, "gold_frame_width_ratio", 0.006)
                        )
                    ),
                ),
            )

        d = ImageDraw.Draw(img)

        heading = str(back.get("heading", "Challenge your Sudoku skills from Medium to Expert.") or "")
        subheading = str(back.get("subheading", "A progressive 2027 collection for focused puzzle solvers.") or "")
        copy = str(
            back.get(
                "copy",
                "Work through a carefully organized Sudoku collection designed to move from confident medium-level solving into harder and expert-level logic.",
            )
            or ""
        )
        section_1_label = str(back.get("section_1_label", "MEDIUM") or "")
        section_1_copy = str(back.get("section_1_copy", "Warm up with clean logic and steady progress.") or "")
        section_2_label = str(back.get("section_2_label", "HARD") or "")
        section_2_copy = str(back.get("section_2_copy", "Push deeper with stronger deduction and tighter grids.") or "")
        section_3_label = str(back.get("section_3_label", "EXPERT") or "")
        section_3_copy = str(back.get("section_3_copy", "Finish with demanding puzzles built for serious solvers.") or "")
        footer_note = str(back.get("footer_note", "Classic 9x9 Sudoku • Progressive difficulty • Print-ready puzzle book") or "")

        main_givens81 = _normalize_givens(
            str(resolved_puzzle_art.get("main_givens81") or ""),
            _SAMPLE_MAIN_GIVENS81,
        )

        safe_x0 = int(
            round(width_px * _layout_number(back_calibration, "safe_x0_ratio", 0.105))
        )
        safe_x1 = int(
            round(width_px * _layout_number(back_calibration, "safe_x1_ratio", 0.895))
        )

        heading_box = _ratio_box(
            back_calibration,
            "heading_box",
            (0.105, 0.095, 0.895, 0.225),
            width_px,
            height_px,
        )
        heading_font = _fit_font(
            d,
            heading,
            max_width=heading_box[2] - heading_box[0],
            max_height=heading_box[3] - heading_box[1],
            start_size=int(
                round(height_px * _layout_number(back_calibration, "heading_font_height_ratio", 0.042))
            ),
            min_size=int(_layout_number(back_calibration, "heading_min_size", 44)),
            bold=True,
        )
        _draw_wrapped_text(
            img,
            text=heading,
            box=heading_box,
            font=heading_font,
            fill=white,
            line_spacing=1.05,
            align="center",
        )

        subheading_box = _ratio_box(
            back_calibration,
            "subheading_box",
            (0.105, 0.235, 0.895, 0.305),
            width_px,
            height_px,
        )
        subheading_font = _font(
            int(
                round(
                    height_px
                    * _layout_number(back_calibration, "subheading_font_height_ratio", 0.025)
                )
            ),
            bold=True,
        )
        _draw_wrapped_text(
            img,
            text=subheading,
            box=subheading_box,
            font=subheading_font,
            fill=gold,
            line_spacing=1.10,
            align="center",
        )

        copy_panel = _ratio_box(
            back_calibration,
            "copy_panel",
            (0.115, 0.335, 0.885, 0.505),
            width_px,
            height_px,
        )
        d.rounded_rectangle(
            copy_panel,
            radius=24,
            fill=(3, 14, 45, 165),
            outline=(gold[0], gold[1], gold[2], 155),
            width=3,
        )
        copy_font = _font(
            int(
                round(
                    height_px
                    * _layout_number(back_calibration, "copy_font_height_ratio", 0.022)
                )
            ),
            bold=False,
        )
        copy_inset_x = int(
            round(width_px * _layout_number(back_calibration, "copy_inset_x_ratio", 0.035))
        )
        copy_inset_y = int(
            round(height_px * _layout_number(back_calibration, "copy_inset_y_ratio", 0.025))
        )
        _draw_wrapped_text(
            img,
            text=copy,
            box=(
                copy_panel[0] + copy_inset_x,
                copy_panel[1] + copy_inset_y,
                copy_panel[2] - copy_inset_x,
                copy_panel[3] - copy_inset_y,
            ),
            font=copy_font,
            fill=white,
            line_spacing=_layout_number(back_calibration, "copy_line_spacing", 1.22),
            align="center",
        )

        card_top = int(
            round(height_px * _layout_number(back_calibration, "card_top_ratio", 0.545))
        )
        card_bottom = int(
            round(height_px * _layout_number(back_calibration, "card_bottom_ratio", 0.720))
        )
        card_gap = int(
            round(width_px * _layout_number(back_calibration, "card_gap_x_ratio", 0.025))
        )
        card_w = int((safe_x1 - safe_x0 - 2 * card_gap) / 3)
        cards = [
            (
                safe_x0,
                card_top,
                safe_x0 + card_w,
                card_bottom,
                section_1_label,
                section_1_copy,
            ),
            (
                safe_x0 + card_w + card_gap,
                card_top,
                safe_x0 + 2 * card_w + card_gap,
                card_bottom,
                section_2_label,
                section_2_copy,
            ),
            (
                safe_x0 + 2 * (card_w + card_gap),
                card_top,
                safe_x0 + 3 * card_w + 2 * card_gap,
                card_bottom,
                section_3_label,
                section_3_copy,
            ),
        ]

        for x0, y0, x1, y1, label, body in cards:
            _draw_feature_card(
                img,
                box=(x0, y0, x1, y1),
                label=label,
                body=body,
                gold=gold,
                white=white,
                purple=purple,
            )

        if _feature_enabled(features, "back_mini_sudoku_enabled", False):
            # Optional decorative Sudoku tile.
            _draw_mini_sudoku_tile(
                img,
                box=_ratio_box(
                    back_calibration,
                    "mini_sudoku_box",
                    (0.145, 0.765, 0.390, 0.935),
                    width_px,
                    height_px,
                ),
                givens81=main_givens81,
                gold=gold,
            )

        if _feature_enabled(features, "back_footer_note_enabled", False):
            footer_font = _font(
                int(
                    round(
                        height_px
                        * _layout_number(back_calibration, "footer_font_height_ratio", 0.018)
                    )
                ),
                bold=True,
            )
            _draw_wrapped_text(
                img,
                text=footer_note,
                box=_ratio_box(
                    back_calibration,
                    "footer_box",
                    (0.415, 0.790, 0.780, 0.875),
                    width_px,
                    height_px,
                ),
                font=footer_font,
                fill=(255, 255, 255, 230),
                line_spacing=_layout_number(back_calibration, "footer_line_spacing", 1.20),
                align="left",
            )

        barcode_enabled = bool(barcode.get("enabled", False)) and _feature_enabled(
            features,
            "barcode_box_enabled",
            False,
        )
        if barcode_enabled:
            width_in = float(barcode.get("width_in", 2.0))
            height_in = float(barcode.get("height_in", 1.2))
            margin_right_in = float(barcode.get("margin_right_in", 0.45))
            margin_bottom_in = float(barcode.get("margin_bottom_in", 0.45))

            px_per_in_x = width_px / max(0.1, float(getattr(geometry, "trim_width_in", 8.5)))
            px_per_in_y = height_px / max(0.1, float(getattr(geometry, "trim_height_in", 11.0)))

            bw = int(round(width_in * px_per_in_x))
            bh = int(round(height_in * px_per_in_y))
            bx1 = width_px - int(round(margin_right_in * px_per_in_x))
            by1 = height_px - int(round(margin_bottom_in * px_per_in_y))
            bx0 = bx1 - bw
            by0 = by1 - bh

            d.rounded_rectangle(
                [bx0, by0, bx1, by1],
                radius=8,
                fill=(255, 255, 255, 255),
                outline=(230, 230, 230, 255),
                width=2,
            )

        output_file = out_path / "back_cover.png"
        img.convert("RGB").save(output_file, quality=95)
        return output_file

    def render_spine_cover(
        self,
        context: ResolvedCoverDesignContext,
        out_dir: str | Path,
        geometry: Any,
        width_px: int = 120,
        height_px: int = 3300,
    ) -> Path | None:
        features = _get_feature_variables(context)
        if not _feature_enabled(features, "custom_spine_enabled", True):
            return None

        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        palette = _get_palette_variables(context)
        spine = dict(context.variables.get("spine", {}))
        text = _get_text_variables(context)

        navy = _hex_to_rgba(str(palette.get("navy", "#061B49")), (6, 27, 73, 255))
        navy_dark = _hex_to_rgba(str(palette.get("navy_dark", "#020B22")), (2, 11, 34, 255))
        purple = _hex_to_rgba(str(palette.get("purple", "#2A075F")), (42, 7, 95, 255))
        purple_dark = _hex_to_rgba(str(palette.get("purple_dark", "#14022F")), (20, 2, 47, 255))
        gold = _hex_to_rgba(str(palette.get("gold", "#F4C451")), (244, 196, 81, 255))
        gold_dark = _hex_to_rgba(str(palette.get("gold_dark", "#A66A05")), (166, 106, 5, 255))
        white = _hex_to_rgba(str(palette.get("white", "#FFFFFF")), (255, 255, 255, 255))

        spine_calibration = _spine_layout_calibration(context)

        img = Image.new("RGBA", (max(1, width_px), height_px), navy_dark)
        _draw_vertical_gradient(img, top=navy_dark, bottom=purple_dark)

        _draw_expert_network_texture(
            img,
            gold=gold,
            purple=purple,
            enabled=_feature_enabled(features, "network_texture_enabled", True),
        )

        d = ImageDraw.Draw(img)
        w, h = img.size

        trim_w = max(
            1,
            int(round(w * _layout_number(spine_calibration, "gold_trim_width_ratio", 0.12))),
        )
        d.rectangle([0, 0, trim_w, h], fill=gold)
        d.rectangle([w - trim_w, 0, w, h], fill=gold_dark)

        inner_w = max(
            1,
            int(round(w * _layout_number(spine_calibration, "inner_trim_width_ratio", 0.035))),
        )
        d.rectangle([trim_w, 0, trim_w + inner_w, h], fill=(255, 244, 185, 210))
        d.rectangle([w - trim_w - inner_w, 0, w - trim_w, h], fill=(255, 244, 185, 180))

        spine_width_in = float(getattr(geometry, "spine_width_in", 0.0))
        page_count = _resolve_page_count(context, geometry)

        min_page_count_for_spine_text = int(
            _layout_number(spine_calibration, "min_page_count_for_spine_text", 150)
        )

        kdp_width_safe = (
            spine_width_in >= _layout_number(spine_calibration, "text_safe_spine_width_in", 0.33)
            and width_px >= int(_layout_number(spine_calibration, "text_safe_width_px", 90))
        )

        show_spine_text = page_count >= min_page_count_for_spine_text and kdp_width_safe

        if show_spine_text:
            spine_year = str(spine.get("year") or text.get("year") or "")
            spine_title = str(spine.get("title") or "").strip()
            if not spine_title:
                puzzle_count_label = str(text.get("puzzle_count_label", "1000+") or "")
                title_word = str(text.get("title_word", "SUDOKU") or "")
                title_joiner = str(text.get("title_joiner", " "))
                spine_title = f"{puzzle_count_label}{title_joiner}{title_word}".strip()

            spine_subtitle = str(spine.get("subtitle") or text.get("difficulty_label") or "")

            title_font = _fit_font(
                d,
                spine_title,
                max_width=int(h * _layout_number(spine_calibration, "title_max_width_height_ratio", 0.55)),
                max_height=int(w * _layout_number(spine_calibration, "title_max_height_width_ratio", 0.46)),
                start_size=max(
                    int(_layout_number(spine_calibration, "title_min_start_size", 30)),
                    int(w * _layout_number(spine_calibration, "title_start_width_ratio", 0.34)),
                ),
                min_size=int(_layout_number(spine_calibration, "title_min_size", 14)),
                bold=True,
            )
            subtitle_font = _fit_font(
                d,
                spine_subtitle,
                max_width=int(h * _layout_number(spine_calibration, "subtitle_max_width_height_ratio", 0.33)),
                max_height=int(w * _layout_number(spine_calibration, "subtitle_max_height_width_ratio", 0.32)),
                start_size=max(
                    int(_layout_number(spine_calibration, "subtitle_min_start_size", 22)),
                    int(w * _layout_number(spine_calibration, "subtitle_start_width_ratio", 0.22)),
                ),
                min_size=int(_layout_number(spine_calibration, "subtitle_min_size", 10)),
                bold=True,
            )
            year_font = _fit_font(
                d,
                spine_year,
                max_width=int(h * _layout_number(spine_calibration, "year_max_width_height_ratio", 0.16)),
                max_height=int(w * _layout_number(spine_calibration, "year_max_height_width_ratio", 0.34)),
                start_size=max(
                    int(_layout_number(spine_calibration, "year_min_start_size", 22)),
                    int(w * _layout_number(spine_calibration, "year_start_width_ratio", 0.24)),
                ),
                min_size=int(_layout_number(spine_calibration, "year_min_size", 10)),
                bold=True,
            )

            cx = int(round(w * _layout_number(spine_calibration, "text_center_x_ratio", 0.50)))
            rotation_degrees = _layout_number(spine_calibration, "text_rotation_degrees", 90)

            _draw_rotated_text_centered(
                img,
                text=spine_year,
                center=(
                    cx,
                    int(round(h * _layout_number(spine_calibration, "year_center_y_ratio", 0.125))),
                ),
                font=year_font,
                fill=gold,
                angle_degrees=rotation_degrees,
            )
            _draw_rotated_text_centered(
                img,
                text=spine_title,
                center=(
                    cx,
                    int(round(h * _layout_number(spine_calibration, "title_center_y_ratio", 0.455))),
                ),
                font=title_font,
                fill=white,
                angle_degrees=rotation_degrees,
            )
            _draw_rotated_text_centered(
                img,
                text=spine_subtitle,
                center=(
                    cx,
                    int(round(h * _layout_number(spine_calibration, "subtitle_center_y_ratio", 0.710))),
                ),
                font=subtitle_font,
                fill=gold,
                angle_degrees=rotation_degrees,
            )

        output_file = out_path / "spine_cover.png"
        img.convert("RGB").save(output_file, quality=95)
        return output_file

    def render_front_cover(
        self,
        context: ResolvedCoverDesignContext,
        out_dir: str | Path,
        width_px: int = 2550,
        height_px: int = 3300,
    ) -> CoverRenderResult:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        text = _get_text_variables(context)
        background = _get_background_variables(context)
        features = _get_feature_variables(context)
        palette = _get_palette_variables(context)
        typography = _get_typography_variables(context)
        resolved_puzzle_art = _get_resolved_puzzle_art_variables(context)

        asset_path = str(
            background.get(
                "asset_path",
                "assets/backgrounds/annual_expert_gauge_bg_1.png",
            )
        )
        background_path = _resolve_asset_path(context, asset_path)

        img = Image.open(background_path).convert("RGBA")

        # The source artwork already has the intended 8.5x11 proportions.
        # Direct resize avoids accidental cropping of gold edge details.
        img = img.resize((width_px, height_px), Image.Resampling.LANCZOS)

        white = _hex_to_rgba(str(palette.get("white", "#FFFFFF")), (255, 255, 255, 255))
        black = _hex_to_rgba(str(palette.get("black", "#000000")), (0, 0, 0, 255))
        gold = _hex_to_rgba(str(palette.get("gold", "#F4C451")), (244, 196, 81, 255))

        year = str(text.get("year", "2027") or "")
        puzzle_count_label = str(text.get("puzzle_count_label", "1000+") or "")
        title_word = str(text.get("title_word", "SUDOKU") or "")
        title_joiner = str(text.get("title_joiner", " "))
        difficulty_label = str(text.get("difficulty_label", "MEDIUM TO EXPERT") or "")
        gauge_label = str(text.get("gauge_label", "DIFFICULTY") or "")
        gauge_value_label = str(text.get("gauge_value_label", "EXPERT LEVEL") or "")

        title = f"{puzzle_count_label}{title_joiner}{title_word}".strip()

        main_givens81 = _normalize_givens(
            str(resolved_puzzle_art.get("main_givens81") or ""),
            _SAMPLE_MAIN_GIVENS81,
        )
        left_side_givens81 = _normalize_givens(
            str(resolved_puzzle_art.get("left_side_givens81") or ""),
            _SAMPLE_LEFT_GIVENS81,
        )
        right_side_givens81 = _normalize_givens(
            str(resolved_puzzle_art.get("right_side_givens81") or ""),
            _SAMPLE_RIGHT_GIVENS81,
        )

        draw = ImageDraw.Draw(img)

        # Calibrated against assets/backgrounds/annual_expert_gauge_bg_1.png.
        #
        # Coordinates are expressed in the native background coordinate system:
        # 1584 x 2048. They are scaled to the requested output size.
        #
        # Defaults live here for safety, but production tuning can override them
        # through cover_design.variables.layout_calibration.front.
        front_calibration = _front_layout_calibration(context)

        year_box = _layout_box(
            front_calibration,
            "year_box",
            (330, 130, 1254, 455),
            width_px,
            height_px,
        )
        title_box = _layout_box(
            front_calibration,
            "title_box",
            (105, 515, 1480, 675),
            width_px,
            height_px,
        )
        hero_grid_box = _layout_box(
            front_calibration,
            "hero_grid_box",
            (307, 720, 1271, 1682),
            width_px,
            height_px,
        )
        gauge_label_box = _layout_box(
            front_calibration,
            "gauge_label_box",
            (610, 1742, 974, 1782),
            width_px,
            height_px,
        )
        gauge_value_box = _layout_box(
            front_calibration,
            "gauge_value_box",
            (560, 1778, 1024, 1824),
            width_px,
            height_px,
        )
        difficulty_box = _layout_box(
            front_calibration,
            "difficulty_box",
            (70, 1795, 1514, 1965),
            width_px,
            height_px,
        )

        year_font_ratio = _layout_number(front_calibration, "year_font_height_ratio", 0.125)
        title_font_ratio = _layout_number(front_calibration, "title_font_height_ratio", 0.064)
        gauge_label_font_ratio = _layout_number(front_calibration, "gauge_label_font_height_ratio", 0.019)
        gauge_value_font_ratio = _layout_number(front_calibration, "gauge_value_font_height_ratio", 0.022)
        difficulty_font_ratio = _layout_number(front_calibration, "difficulty_font_height_ratio", 0.080)
        hero_digit_scale = _layout_number(front_calibration, "hero_digit_scale", 0.78)
        side_digit_scale = _layout_number(front_calibration, "side_digit_scale", 0.78)
        hero_digit_font_face = _layout_string(
            front_calibration,
            "hero_digit_font_face",
            "arena_blue_grid_regular",
        )
        side_digit_font_face = _layout_string(
            front_calibration,
            "side_digit_font_face",
            "arena_blue_grid_regular",
        )

        # Visible side panel boundary from the background artwork.
        left_visible_quad = _scale_quad(
            _as_native_quad(
                front_calibration.get(
                    "left_side_visible_quad",
                    front_calibration.get("left_side_quad"),
                ),
                [(122, 940), (300, 890), (300, 1575), (122, 1534)],
            ),
            width_px,
            height_px,
        )
        right_visible_quad = _scale_quad(
            _as_native_quad(
                front_calibration.get(
                    "right_side_visible_quad",
                    front_calibration.get("right_side_quad"),
                ),
                [(1284, 890), (1462, 940), (1462, 1534), (1284, 1575)],
            ),
            width_px,
            height_px,
        )

        # Digit-placement lattice quad.
        # This is intentionally allowed to be slightly inset relative to the
        # visible panel boundary so digits align with the true cell centers.
        left_digit_quad = _scale_quad(
            _as_native_quad(
                front_calibration.get("left_side_digit_quad"),
                [(141, 963), (284, 923), (284, 1548), (138, 1516)],
            ),
            width_px,
            height_px,
        )
        right_digit_quad = _scale_quad(
            _as_native_quad(
                front_calibration.get("right_side_digit_quad"),
                [(1302, 923), (1446, 964), (1446, 1516), (1304, 1548)],
            ),
            width_px,
            height_px,
        )

        left_visible_cols = _layout_int_list(
            front_calibration,
            "left_side_visible_cols",
            [0, 1, 2],
            min_value=0,
            max_value=8,
        )
        right_visible_cols = _layout_int_list(
            front_calibration,
            "right_side_visible_cols",
            [6, 7, 8],
            min_value=0,
            max_value=8,
        )

        if _feature_enabled(features, "side_digits_enabled", True):
            side_digit_fill = (255, 255, 255, 235)
            side_source_size = int(_layout_number(front_calibration, "side_source_size", 900))

            left_layer = _make_side_visible_digit_layer(
                source_height=side_source_size,
                givens81=left_side_givens81,
                visible_cols=left_visible_cols,
                digit_fill=side_digit_fill,
                digit_scale=side_digit_scale,
                font_face=side_digit_font_face,
            )
            right_layer = _make_side_visible_digit_layer(
                source_height=side_source_size,
                givens81=right_side_givens81,
                visible_cols=right_visible_cols,
                digit_fill=side_digit_fill,
                digit_scale=side_digit_scale,
                font_face=side_digit_font_face,
            )

            img.alpha_composite(_perspective_warp_quad(left_layer, left_digit_quad, img.size))
            img.alpha_composite(_perspective_warp_quad(right_layer, right_digit_quad, img.size))

        if _feature_enabled(features, "hero_digits_enabled", True):
            _draw_givens_in_grid_box(
                img,
                givens81=main_givens81,
                box=hero_grid_box,
                digit_fill=black,
                digit_scale=hero_digit_scale,
                bold=False,
                font_face=hero_digit_font_face,
            )

        debug_overlay_enabled = (
            _feature_enabled(features, "front_debug_overlay_enabled", False)
            or bool(front_calibration.get("debug_overlay_enabled", False))
        )
        if debug_overlay_enabled:
            _draw_front_grid_debug_overlay(
                img,
                hero_grid_box=hero_grid_box,
                left_visible_quad=left_visible_quad,
                right_visible_quad=right_visible_quad,
                left_digit_quad=left_digit_quad,
                right_digit_quad=right_digit_quad,
                left_visible_cols=left_visible_cols,
                right_visible_cols=right_visible_cols,
            )

        if _feature_enabled(features, "front_text_enabled", True):
            year_curve = _resolved_front_text_curve(
                front_calibration,
                typography,
                curve_key="year_curve",
                font_face_key="year_font_face",
                font_role_key="year_font_role",
                tracking_key="year_tracking_px",
                default_face="montserrat_bold",
            )
            _draw_front_display_text(
                img,
                draw,
                text=year,
                box=year_box,
                curve=year_curve,
                fill=white,
                width_px=width_px,
                height_px=height_px,
                fallback_font_ratio=year_font_ratio,
                fallback_min_size=96,
                fallback_shadow_offset=(18, 22),
                fallback_shadow_blur=9,
            )

            title_curve = _resolved_front_text_curve(
                front_calibration,
                typography,
                curve_key="title_curve",
                font_face_key="title_font_face",
                font_role_key="title_font_role",
                tracking_key="title_tracking_px",
                default_face="annual_expert_gauge_title",
            )
            _draw_front_display_text(
                img,
                draw,
                text=title,
                box=title_box,
                curve=title_curve,
                fill=white,
                width_px=width_px,
                height_px=height_px,
                fallback_font_ratio=title_font_ratio,
                fallback_min_size=72,
                fallback_shadow_offset=(14, 16),
                fallback_shadow_blur=8,
            )

            if _feature_enabled(features, "gauge_text_enabled", True):
                gauge_label_curve = _resolved_front_text_curve(
                    front_calibration,
                    typography,
                    curve_key="gauge_label_curve",
                    font_face_key="gauge_label_font_face",
                    font_role_key="gauge_label_font_role",
                    tracking_key="gauge_label_tracking_px",
                    default_face="montserrat_bold",
                    legacy_font_face_key="gauge_font_face",
                    legacy_font_role_key="gauge_font_role",
                    legacy_tracking_key="gauge_tracking_px",
                )
                _draw_front_display_text(
                    img,
                    draw,
                    text=gauge_label,
                    box=gauge_label_box,
                    curve=gauge_label_curve,
                    fill=white,
                    width_px=width_px,
                    height_px=height_px,
                    fallback_font_ratio=gauge_label_font_ratio,
                    fallback_min_size=22,
                    fallback_shadow_offset=(3, 4),
                    fallback_shadow_blur=3,
                )

                gauge_value_curve = _resolved_front_text_curve(
                    front_calibration,
                    typography,
                    curve_key="gauge_value_curve",
                    font_face_key="gauge_value_font_face",
                    font_role_key="gauge_value_font_role",
                    tracking_key="gauge_value_tracking_px",
                    default_face="montserrat_bold",
                    legacy_font_face_key="gauge_font_face",
                    legacy_font_role_key="gauge_font_role",
                    legacy_tracking_key="gauge_tracking_px",
                )
                _draw_front_display_text(
                    img,
                    draw,
                    text=gauge_value_label,
                    box=gauge_value_box,
                    curve=gauge_value_curve,
                    fill=gold,
                    width_px=width_px,
                    height_px=height_px,
                    fallback_font_ratio=gauge_value_font_ratio,
                    fallback_min_size=24,
                    fallback_shadow_offset=(3, 4),
                    fallback_shadow_blur=3,
                )

            difficulty_curve = _resolved_front_text_curve(
                front_calibration,
                typography,
                curve_key="difficulty_curve",
                font_face_key="difficulty_font_face",
                font_role_key="difficulty_font_role",
                tracking_key="difficulty_tracking_px",
                default_face="montserrat_bold",
            )
            _draw_front_display_text(
                img,
                draw,
                text=difficulty_label,
                box=difficulty_box,
                curve=difficulty_curve,
                fill=white,
                width_px=width_px,
                height_px=height_px,
                fallback_font_ratio=difficulty_font_ratio,
                fallback_min_size=72,
                fallback_shadow_offset=(16, 18),
                fallback_shadow_blur=8,
            )

        output_file = out_path / "front_cover.png"
        img.convert("RGB").save(output_file, quality=95)

        return CoverRenderResult(
            front_cover_png=output_file,
            width_px=width_px,
            height_px=height_px,
            renderer_key=self.renderer_key,
        )