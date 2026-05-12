from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from python.publishing.cover_designs.models import ResolvedCoverDesignContext
from python.publishing.cover_renderers.base_renderer import (
    BaseCoverRenderer,
    CoverRenderResult,
)


def _font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)

    return ImageFont.load_default()


def _text_bbox(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int, int, int]:
    return draw.textbbox((0, 0), text, font=font)


def _fit_font(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    start_size: int,
    min_size: int = 40,
    bold: bool = True,
) -> ImageFont.ImageFont:
    size = start_size
    while size >= min_size:
        f = _font(size, bold=bold)
        bbox = _text_bbox(draw, text, f)
        if bbox[2] - bbox[0] <= max_width:
            return f
        size -= 4
    return _font(min_size, bold=bold)


def _center_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    y: int,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    canvas_width: int,
) -> None:
    bbox = _text_bbox(draw, text, font)
    text_width = bbox[2] - bbox[0]
    x = (canvas_width - text_width) // 2
    draw.text((x, y), text, font=font, fill=fill)


def _center_text_shadow(
    img: Image.Image,
    text: str,
    y: int,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    shadow_fill: tuple[int, int, int, int],
    canvas_width: int,
    shadow_offset: tuple[int, int] = (18, 22),
    blur_radius: int = 10,
) -> None:
    shadow_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow_layer)
    bbox = sd.textbbox((0, 0), text, font=font)
    x = (canvas_width - (bbox[2] - bbox[0])) // 2
    sd.text((x + shadow_offset[0], y + shadow_offset[1]), text, font=font, fill=shadow_fill)
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(blur_radius))
    img.alpha_composite(shadow_layer)

    draw = ImageDraw.Draw(img)
    draw.text((x, y), text, font=font, fill=fill)




def _text_width_by_glyphs(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    tracking: int = 0,
) -> int:
    total = 0
    for index, ch in enumerate(text):
        bbox = draw.textbbox((0, 0), ch, font=font)
        total += bbox[2] - bbox[0]
        if index < len(text) - 1:
            total += tracking
    return total


def _draw_centered_text_on_curve(
    img: Image.Image,
    text: str,
    y: int,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
    canvas_width: int,
    curve_amplitude: int,
    tracking: int = 0,
    x_offset: int = 0,
    shadow_fill: tuple[int, int, int, int] | None = None,
    shadow_offset: tuple[int, int] = (0, 0),
    shadow_blur: int = 0,
) -> None:
    """
    Draw upright letters along a gentle curved baseline.

    curve_amplitude < 0:
        center letters are higher than edge letters, visually ∪.

    curve_amplitude > 0:
        center letters are lower than edge letters, visually ∩.

    This does not rotate or distort glyphs.
    """
    measure = ImageDraw.Draw(img)
    text_width = _text_width_by_glyphs(measure, text, font, tracking=tracking)
    start_x = (canvas_width - text_width) // 2 + x_offset

    def draw_layer(
        layer: Image.Image,
        dx: int,
        dy: int,
        color: tuple[int, int, int, int],
    ) -> None:
        d = ImageDraw.Draw(layer, "RGBA")
        cursor_x = start_x
        for index, ch in enumerate(text):
            bbox = d.textbbox((0, 0), ch, font=font)
            ch_w = bbox[2] - bbox[0]

            glyph_center_x = cursor_x + ch_w / 2
            t = (glyph_center_x - start_x) / max(1, text_width)
            curve_y = int(round(math.sin(math.pi * t) * curve_amplitude))

            d.text((int(cursor_x + dx), int(y + curve_y + dy)), ch, font=font, fill=color)
            cursor_x += ch_w + tracking

    if shadow_fill is not None:
        shadow_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw_layer(
            shadow_layer,
            dx=shadow_offset[0],
            dy=shadow_offset[1],
            color=shadow_fill,
        )
        if shadow_blur > 0:
            shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(shadow_blur))
        img.alpha_composite(shadow_layer)

    text_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw_layer(text_layer, dx=0, dy=0, color=fill)
    img.alpha_composite(text_layer)





def _draw_centered_text_on_band_centerline(
    img: Image.Image,
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
    canvas_width: int,
    band_x0: int,
    band_x1: int,
    centerline_edge_y: int,
    centerline_amplitude: int,
    tracking: int = 0,
    x_offset: int = 0,
    y_offset: int = 0,
    rotate_to_curve: bool = True,
    rotation_strength: float = 1.0,
    max_rotation_degrees: float = 7.0,
    shadow_fill: tuple[int, int, int, int] | None = None,
    shadow_offset: tuple[int, int] = (0, 0),
    shadow_blur: int = 0,
) -> None:
    """
    Draw text along a band's curved centerline.

    rotate_to_curve=True rotates each glyph slightly to follow the curve tangent.
    This removes the visual "stair-step / ladder" effect from upright letters.
    """
    measure = ImageDraw.Draw(img)
    text_width = _text_width_by_glyphs(measure, text, font, tracking=tracking)
    start_x = (canvas_width - text_width) // 2 + x_offset

    full_bbox = measure.textbbox((0, 0), text, font=font)
    text_h = full_bbox[3] - full_bbox[1]

    def centerline_y(abs_x: float) -> float:
        t = (abs_x - band_x0) / max(1, band_x1 - band_x0)
        t = max(0.0, min(1.0, t))
        return centerline_edge_y + math.sin(math.pi * t) * centerline_amplitude + y_offset

    def tangent_angle_degrees(abs_x: float) -> float:
        t = (abs_x - band_x0) / max(1, band_x1 - band_x0)
        t = max(0.0, min(1.0, t))

        # Derivative of y = edge + sin(pi*t) * amplitude.
        slope = (
            centerline_amplitude
            * math.pi
            * math.cos(math.pi * t)
            / max(1, band_x1 - band_x0)
        )

        # PIL image coordinates have y increasing downward.
        # The mathematical tangent angle must therefore be inverted
        # so glyphs tilt in the same visual direction as the curve.
        angle = -math.degrees(math.atan(slope)) * rotation_strength
        return max(-max_rotation_degrees, min(max_rotation_degrees, angle))

    def paste_glyph(
        layer: Image.Image,
        ch: str,
        x: float,
        baseline_y: float,
        dx: int,
        dy: int,
        color: tuple[int, int, int, int],
        angle: float,
    ) -> None:
        bbox = measure.textbbox((0, 0), ch, font=font)
        ch_w = bbox[2] - bbox[0]
        ch_h = bbox[3] - bbox[1]

        pad = max(24, int(font.size * 0.18))
        glyph = Image.new("RGBA", (ch_w + pad * 2, ch_h + pad * 2), (0, 0, 0, 0))
        gd = ImageDraw.Draw(glyph, "RGBA")

        gd.text(
            (pad - bbox[0], pad - bbox[1]),
            ch,
            font=font,
            fill=color,
        )

        if rotate_to_curve and abs(angle) > 0.05:
            glyph = glyph.rotate(
                angle,
                resample=Image.Resampling.BICUBIC,
                expand=True,
            )

        paste_x = int(round(x + ch_w / 2 - glyph.size[0] / 2 + dx))
        paste_y = int(round(baseline_y - glyph.size[1] / 2 + dy))

        layer.alpha_composite(glyph, (paste_x, paste_y))

    def draw_layer(
        layer: Image.Image,
        dx: int,
        dy: int,
        color: tuple[int, int, int, int],
    ) -> None:
        cursor_x = start_x

        for ch in text:
            bbox = measure.textbbox((0, 0), ch, font=font)
            ch_w = bbox[2] - bbox[0]

            glyph_center_x = cursor_x + ch_w / 2
            cy = centerline_y(glyph_center_x)
            angle = tangent_angle_degrees(glyph_center_x)

            glyph_center_y = cy - text_h / 2
            paste_glyph(
                layer=layer,
                ch=ch,
                x=cursor_x,
                baseline_y=glyph_center_y + text_h / 2,
                dx=dx,
                dy=dy,
                color=color,
                angle=angle,
            )

            cursor_x += ch_w + tracking

    if shadow_fill is not None:
        shadow_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw_layer(
            shadow_layer,
            dx=shadow_offset[0],
            dy=shadow_offset[1],
            color=shadow_fill,
        )
        if shadow_blur > 0:
            shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(shadow_blur))
        img.alpha_composite(shadow_layer)

    text_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw_layer(text_layer, dx=0, dy=0, color=fill)
    img.alpha_composite(text_layer)

def _draw_soft_rect_shadow(
    img: Image.Image,
    box: tuple[int, int, int, int],
    offset: tuple[int, int],
    blur: int,
    fill: tuple[int, int, int, int],
) -> None:
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    x0, y0, x1, y1 = box
    ox, oy = offset
    d.rectangle([x0 + ox, y0 + oy, x1 + ox, y1 + oy], fill=fill)
    layer = layer.filter(ImageFilter.GaussianBlur(blur))
    img.alpha_composite(layer)


def _draw_technical_texture(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    color: tuple[int, int, int, int],
) -> None:
    # Sparse blueprint-like linework instead of uniform graph paper.
    step = 190
    short = 78

    for y in range(170, height - 120, step):
        for x in range(40, width - 40, step):
            if (x // step + y // step) % 2 == 0:
                draw.line([x, y, x + short, y], fill=color, width=3)
                draw.line([x, y, x, y + short], fill=color, width=3)
            else:
                draw.line([x + 42, y + 42, x + short + 42, y + 42], fill=color, width=2)
                draw.line([x + short + 42, y + 42, x + short + 42, y + short], fill=color, width=2)


def _draw_curved_ribbon(
    img: Image.Image,
    y_center: int,
    thickness: int,
    amplitude: int,
    fill: tuple[int, int, int, int],
    phase: float = 0.0,
) -> None:
    width, _ = img.size
    points_top: list[tuple[int, int]] = []
    points_bottom: list[tuple[int, int]] = []

    for x in range(-80, width + 81, 24):
        t = (x / width) * math.pi
        y = y_center + int(math.sin(t + phase) * amplitude)
        points_top.append((x, y - thickness // 2))
        points_bottom.append((x, y + thickness // 2))

    ribbon = points_top + list(reversed(points_bottom))
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.polygon(ribbon, fill=fill)
    img.alpha_composite(layer)


def _draw_curved_ribbon_shadow(
    img: Image.Image,
    y_center: int,
    thickness: int,
    amplitude: int,
    offset_y: int = 18,
    alpha: int = 95,
    blur_radius: int = 14,
    phase: float = 0.0,
) -> None:
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    _draw_curved_ribbon(
        shadow,
        y_center=y_center + offset_y,
        thickness=thickness,
        amplitude=amplitude,
        fill=(0, 0, 0, alpha),
        phase=phase,
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(blur_radius))
    img.alpha_composite(shadow)


def _curve_y(
    x: int,
    x0: int,
    x1: int,
    edge_y: int,
    amplitude: int,
) -> int:
    """
    amplitude > 0 = concave downward / center lower / ∩ visual boundary
    amplitude < 0 = concave upward / center higher / ∪ visual boundary
    """
    t = (x - x0) / max(1, x1 - x0)
    return int(round(edge_y + math.sin(math.pi * t) * amplitude))


def _curve_points(
    x0: int,
    x1: int,
    edge_y: int,
    amplitude: int,
    step: int = 18,
) -> list[tuple[int, int]]:
    return [
        (x, _curve_y(x, x0, x1, edge_y, amplitude))
        for x in range(x0, x1 + 1, step)
    ] + [(x1, _curve_y(x1, x0, x1, edge_y, amplitude))]


def _draw_band_between_curves(
    img: Image.Image,
    x0: int,
    x1: int,
    top_y: int,
    top_amp: int,
    bottom_y: int,
    bottom_amp: int,
    fill: tuple[int, int, int, int],
) -> None:
    top = _curve_points(x0, x1, top_y, top_amp)
    bottom = _curve_points(x0, x1, bottom_y, bottom_amp)

    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer, "RGBA")
    d.polygon(top + list(reversed(bottom)), fill=fill)
    img.alpha_composite(layer)


def _draw_curved_white_band(
    img: Image.Image,
    x0: int,
    x1: int,
    center_y: int,
    amplitude: int,
    thickness: int,
    fill: tuple[int, int, int, int],
    shadow_offset_y: int,
    shadow_alpha: int,
    shadow_blur: int,
) -> None:
    # Directional shadow first.
    _draw_band_between_curves(
        img,
        x0=x0,
        x1=x1,
        top_y=center_y - thickness // 2 + shadow_offset_y,
        top_amp=amplitude,
        bottom_y=center_y + thickness // 2 + shadow_offset_y,
        bottom_amp=amplitude,
        fill=(0, 0, 0, shadow_alpha),
    )
    shadow = img.copy()
    # The shadow has already been composited, so this helper keeps blur simple
    # by drawing a dedicated blurred overlay instead.
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    _draw_band_between_curves(
        layer,
        x0=x0,
        x1=x1,
        top_y=center_y - thickness // 2 + shadow_offset_y,
        top_amp=amplitude,
        bottom_y=center_y + thickness // 2 + shadow_offset_y,
        bottom_amp=amplitude,
        fill=(0, 0, 0, shadow_alpha),
    )
    layer = layer.filter(ImageFilter.GaussianBlur(shadow_blur))
    img.alpha_composite(layer)

    # Actual white band.
    _draw_band_between_curves(
        img,
        x0=x0,
        x1=x1,
        top_y=center_y - thickness // 2,
        top_amp=amplitude,
        bottom_y=center_y + thickness // 2,
        bottom_amp=amplitude,
        fill=fill,
    )


def _draw_elliptic_radial_shadow(
    img: Image.Image,
    center: tuple[int, int],
    radius_x: int,
    radius_y: int,
    color: tuple[int, int, int] = (0, 18, 42),
    max_alpha: int = 62,
    power: float = 2.2,
    blur: int = 10,
) -> None:
    """
    Draw an elliptical radial shadow:
    strongest at center, fading toward ellipse edge.
    """
    width, height = img.size
    cx, cy = center

    alpha = Image.new("L", img.size, 0)
    px = alpha.load()

    x0 = max(0, cx - radius_x)
    x1 = min(width, cx + radius_x)
    y0 = max(0, cy - radius_y)
    y1 = min(height, cy + radius_y)

    for y in range(y0, y1):
        dy = (y - cy) / max(1, radius_y)
        for x in range(x0, x1):
            dx = (x - cx) / max(1, radius_x)
            d = math.sqrt(dx * dx + dy * dy)

            if d <= 1.0:
                strength = (1.0 - d) ** power
                px[x, y] = int(max_alpha * strength)

    if blur > 0:
        alpha = alpha.filter(ImageFilter.GaussianBlur(blur))

    shadow = Image.new("RGBA", img.size, (*color, 0))
    shadow.putalpha(alpha)
    img.alpha_composite(shadow)


def _draw_light_blue_puzzle_arena(
    img: Image.Image,
    panel_x0: int,
    panel_x1: int,
    top_y: int,
    bottom_y: int,
    top_amplitude: int,
    bottom_amplitude: int,
    fill: tuple[int, int, int, int],
    pattern_enabled: bool = True,
) -> None:
    """
    Draw central light-blue puzzle arena with same base color as the outer
    edge bands, plus intentional technical linework:
    - complete squares
    - partial squares
    - 90-degree corners
    - complete and partial crosses

    All strokes are horizontal/vertical and vary in thickness/opacity.
    """
    import random

    rng = random.Random(42)

    points_top: list[tuple[int, int]] = []
    points_bottom: list[tuple[int, int]] = []

    for x in range(panel_x0, panel_x1 + 1, 24):
        t = ((x - panel_x0) / max(1, panel_x1 - panel_x0)) * math.pi
        y_top = top_y + int(math.sin(t) * top_amplitude)
        y_bottom = bottom_y + int(math.sin(t) * bottom_amplitude)
        points_top.append((x, y_top))
        points_bottom.append((x, y_bottom))

    polygon = points_top + list(reversed(points_bottom))

    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer, "RGBA")
    d.polygon(polygon, fill=fill)

    # Texture toggle:
    # clean_flat or arena_pattern_enabled=false keeps the light-blue arena flat.
    if not pattern_enabled:
        img.alpha_composite(layer)
        return

    mask = Image.new("L", img.size, 0)
    md = ImageDraw.Draw(mask)
    md.polygon(polygon, fill=255)

    pattern = Image.new("RGBA", img.size, (0, 0, 0, 0))
    pd = ImageDraw.Draw(pattern, "RGBA")

    palette = [
        ((23, 96, 136, 145), 5),
        ((30, 112, 154, 120), 4),
        ((42, 132, 174, 92), 3),
        ((65, 158, 196, 68), 2),
        ((82, 175, 210, 48), 1),
    ]

    def pick() -> tuple[tuple[int, int, int, int], int]:
        return rng.choice(palette)

    def h(x0: int, y: int, x1: int, color: tuple[int, int, int, int], width: int) -> None:
        pd.line([x0, y, x1, y], fill=color, width=width)

    def v(x: int, y0: int, y1: int, color: tuple[int, int, int, int], width: int) -> None:
        pd.line([x, y0, x, y1], fill=color, width=width)

    def draw_square_sides(x: int, y: int, w: int, hgt: int, sides: str) -> None:
        """
        sides contains any of:
        T = top, B = bottom, L = left, R = right
        """
        for side in sides:
            color, width = pick()
            if side == "T":
                h(x, y, x + w, color, width)
            elif side == "B":
                h(x, y + hgt, x + w, color, width)
            elif side == "L":
                v(x, y, y + hgt, color, width)
            elif side == "R":
                v(x + w, y, y + hgt, color, width)

    def draw_corner(x: int, y: int, arm_h: int, arm_v: int, orientation: str) -> None:
        """
        orientation:
        tl, tr, bl, br
        """
        color1, width1 = pick()
        color2, width2 = pick()

        if orientation == "tl":
            h(x, y, x + arm_h, color1, width1)
            v(x, y, y + arm_v, color2, width2)
        elif orientation == "tr":
            h(x - arm_h, y, x, color1, width1)
            v(x, y, y + arm_v, color2, width2)
        elif orientation == "bl":
            h(x, y, x + arm_h, color1, width1)
            v(x, y - arm_v, y, color2, width2)
        elif orientation == "br":
            h(x - arm_h, y, x, color1, width1)
            v(x, y - arm_v, y, color2, width2)

    def draw_cross(x: int, y: int, left: int, right: int, up: int, down: int, missing: str = "") -> None:
        """
        Cross made of four possible arms:
        L, R, U, D. Missing can contain any of those.
        """
        if "L" not in missing:
            color, width = pick()
            h(x - left, y, x, color, width)
        if "R" not in missing:
            color, width = pick()
            h(x, y, x + right, color, width)
        if "U" not in missing:
            color, width = pick()
            v(x, y - up, y, color, width)
        if "D" not in missing:
            color, width = pick()
            v(x, y, y + down, color, width)

    shapes = [
        # left upper
        ("square", panel_x0 + 75, top_y + 80, 155, 125, "TLRB"),
        ("square", panel_x0 + 165, top_y + 210, 135, 105, "TR"),
        ("corner", panel_x0 + 90, top_y + 380, 130, 110, "tl"),
        ("cross", panel_x0 + 245, top_y + 450, 115, 150, 90, 130, "R"),

        # left middle
        ("square", panel_x0 + 150, top_y + 570, 210, 150, "TLR"),
        ("square", panel_x0 + 70, top_y + 805, 150, 125, "TLB"),
        ("cross", panel_x0 + 180, top_y + 1035, 120, 155, 105, 150, ""),
        ("corner", panel_x0 + 340, top_y + 1135, 100, 135, "br"),

        # left lower
        ("square", panel_x0 + 260, bottom_y - 260, 175, 120, "TBR"),
        ("cross", panel_x0 + 120, bottom_y - 385, 95, 145, 120, 175, "U"),
        ("corner", panel_x0 + 455, bottom_y - 335, 130, 160, "tl"),

        # center sparse
        ("corner", (panel_x0 + panel_x1) // 2 - 250, top_y + 165, 175, 145, "tl"),
        ("cross", (panel_x0 + panel_x1) // 2 + 35, top_y + 285, 160, 180, 90, 135, "D"),
        ("square", (panel_x0 + panel_x1) // 2 - 110, bottom_y - 315, 190, 140, "TB"),
        ("cross", (panel_x0 + panel_x1) // 2 + 160, bottom_y - 185, 140, 170, 115, 130, "L"),

        # right upper
        ("square", panel_x1 - 255, top_y + 85, 160, 125, "TLRB"),
        ("square", panel_x1 - 390, top_y + 230, 155, 120, "TL"),
        ("cross", panel_x1 - 180, top_y + 430, 115, 145, 95, 140, "D"),
        ("corner", panel_x1 - 315, top_y + 560, 155, 125, "tl"),

        # right middle
        ("square", panel_x1 - 250, top_y + 805, 150, 125, "TRB"),
        ("cross", panel_x1 - 180, top_y + 1045, 145, 125, 110, 145, "R"),
        ("corner", panel_x1 - 95, top_y + 1170, 95, 125, "br"),

        # right lower
        ("square", panel_x1 - 455, bottom_y - 255, 180, 120, "TLR"),
        ("cross", panel_x1 - 225, bottom_y - 360, 130, 155, 125, 160, "U"),
        ("corner", panel_x1 - 120, bottom_y - 170, 100, 135, "br"),
    ]

    for shape in shapes:
        kind = shape[0]

        if kind == "square":
            _, x, y, w, hgt, sides = shape
            draw_square_sides(x, y, w, hgt, sides)

        elif kind == "corner":
            _, x, y, arm_h, arm_v, orientation = shape
            draw_corner(x, y, arm_h, arm_v, orientation)

        elif kind == "cross":
            _, x, y, left, right, up, down, missing = shape
            draw_cross(x, y, left, right, up, down, missing)

    # Add a few isolated sides, still belonging to square/cross language.
    for x, y, length, vertical in [
        (panel_x0 + 70, top_y + 705, 180, False),
        (panel_x0 + 235, top_y + 690, 135, True),
        (panel_x0 + 95, bottom_y - 485, 145, False),
        (panel_x0 + 540, top_y + 195, 165, False),
        (panel_x1 - 545, top_y + 365, 150, False),
        (panel_x1 - 305, bottom_y - 505, 160, False),
        (panel_x1 - 115, top_y + 710, 120, True),
        (panel_x1 - 170, bottom_y - 90, 120, False),
    ]:
        color, width = pick()
        if vertical:
            v(x, y, y + length, color, width)
        else:
            h(x, y, x + length, color, width)

    pattern_alpha = pattern.getchannel("A")
    clipped_alpha = Image.composite(pattern_alpha, Image.new("L", img.size, 0), mask)
    pattern.putalpha(clipped_alpha)

    layer.alpha_composite(pattern)
    img.alpha_composite(layer)

def _draw_sudoku_grid(
    img: Image.Image,
    x: int,
    y: int,
    size: int,
    givens81: str | None = None,
    fill: tuple[int, int, int, int] = (255, 255, 255, 255),
    line: tuple[int, int, int, int] = (0, 0, 0, 255),
    digit: tuple[int, int, int, int] = (0, 0, 0, 255),
    outer_width: int = 8,
    box_width: int = 7,
    cell_width: int = 2,
    digit_scale: float = 0.58,
) -> None:
    draw = ImageDraw.Draw(img)
    draw.rectangle([x, y, x + size, y + size], fill=fill, outline=line, width=outer_width)

    cell = size / 9.0
    for i in range(10):
        w = box_width if i % 3 == 0 else cell_width
        xx = int(round(x + i * cell))
        yy = int(round(y + i * cell))
        draw.line([xx, y, xx, y + size], fill=line, width=w)
        draw.line([x, yy, x + size, yy], fill=line, width=w)

    sample = givens81 if givens81 and len(givens81) == 81 else (
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

    f = _font(max(28, int(cell * digit_scale)), bold=False)
    for idx, ch in enumerate(sample):
        if ch in ("0", ".", "-"):
            continue

        r = idx // 9
        c = idx % 9
        cx = x + c * cell + cell / 2
        cy = y + r * cell + cell / 2

        bbox = draw.textbbox((0, 0), ch, font=f)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        # True optical centering inside the cell.
        # bbox[0]/bbox[1] are not always zero, especially for larger fonts.
        tx = int(round(cx - tw / 2 - bbox[0]))
        ty = int(round(cy - th / 2 - bbox[1]))

        draw.text((tx, ty), ch, font=f, fill=digit)


def _find_perspective_coeffs(
    src: list[tuple[float, float]],
    dst: list[tuple[float, float]],
) -> list[float]:
    """
    Return perspective coefficients mapping dst -> src for PIL transform.

    src and dst are four points:
    top-left, top-right, bottom-right, bottom-left.
    """
    import numpy as np

    matrix = []
    vector = []

    for (x_src, y_src), (x_dst, y_dst) in zip(src, dst):
        matrix.append([x_dst, y_dst, 1, 0, 0, 0, -x_src * x_dst, -x_src * y_dst])
        matrix.append([0, 0, 0, x_dst, y_dst, 1, -y_src * x_dst, -y_src * y_dst])
        vector.append(x_src)
        vector.append(y_src)

    coeffs = np.linalg.solve(np.array(matrix, dtype=float), np.array(vector, dtype=float))
    return coeffs.tolist()


def _perspective_warp_quad(
    source: Image.Image,
    dst_quad: list[tuple[int, int]],
    canvas_size: tuple[int, int],
) -> Image.Image:
    """
    Warp source image into a quadrilateral on a transparent canvas.
    """
    src_w, src_h = source.size
    src_quad = [
        (0.0, 0.0),
        (float(src_w), 0.0),
        (float(src_w), float(src_h)),
        (0.0, float(src_h)),
    ]

    coeffs = _find_perspective_coeffs(src_quad, [(float(x), float(y)) for x, y in dst_quad])

    warped = source.transform(
        canvas_size,
        Image.Transform.PERSPECTIVE,
        coeffs,
        Image.Resampling.BICUBIC,
    )

    mask = Image.new("L", canvas_size, 0)
    md = ImageDraw.Draw(mask)
    md.polygon(dst_quad, fill=255)

    out = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    out.alpha_composite(warped)
    out.putalpha(mask)
    return out


def _make_side_grid(
    size: int,
    givens81: str,
) -> Image.Image:
    """
    Build a side Sudoku grid from the exact same drawing language
    as the hero grid, then it will be perspective-warped later.
    """
    board = Image.new("RGBA", (size, size), (255, 255, 255, 255))

    _draw_sudoku_grid(
        board,
        0,
        0,
        size - 1,
        givens81=givens81,
        fill=(255, 255, 255, 255),
        line=(0, 0, 0, 255),
        digit=(0, 0, 0, 255),
        outer_width=8,
        box_width=10,
        cell_width=2,
        digit_scale=0.78,
    )

    return board


def _get_text_variables(context: ResolvedCoverDesignContext) -> dict[str, Any]:
    return dict(context.variables.get("text", {}))


def _get_feature_variables(context: ResolvedCoverDesignContext) -> dict[str, Any]:
    return dict(context.variables.get("features", {}))


def _get_palette_variables(context: ResolvedCoverDesignContext) -> dict[str, Any]:
    palette = dict(context.variables.get("palette", {}))
    palette_id = str(context.variables.get("palette_id", "") or "")

    built_in_palettes = {
        "classic_annual_blue": {
            "navy": "#04326D",
            "navy_dark": "#042346",
            "sky": "#84CFF6",
            "white": "#FFFFFF",
        },
        "emerald_logic": {
            "navy": "#064F46",
            "navy_dark": "#03322D",
            "sky": "#8EE6D5",
            "white": "#FFFFFF",
        },
        "royal_purple": {
            "navy": "#3D2475",
            "navy_dark": "#25154A",
            "sky": "#C7B5FF",
            "white": "#FFFFFF",
        },
        "sunset_orange": {
            "navy": "#783A12",
            "navy_dark": "#4D2308",
            "sky": "#FFC48B",
            "white": "#FFFFFF",
        },
    }

    resolved = dict(built_in_palettes.get(palette_id, built_in_palettes["classic_annual_blue"]))
    resolved.update(palette)
    return resolved

def _get_resolved_puzzle_art_variables(context: ResolvedCoverDesignContext) -> dict[str, Any]:
    return dict(context.variables.get("resolved_puzzle_art", {}))

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


def _feature_enabled(
    features: dict[str, Any],
    key: str,
    default: bool = True,
) -> bool:
    value = features.get(key, default)
    return bool(value)


class AnnualArenaBlueMultiGridV1Renderer(BaseCoverRenderer):
    renderer_key = "annual_arena_blue_multigrid_v1"

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
        features = _get_feature_variables(context)
        palette = _get_palette_variables(context)
        resolved_puzzle_art = _get_resolved_puzzle_art_variables(context)
        texture_id = str(context.variables.get("texture_id", "technical_arena_strokes") or "technical_arena_strokes")

        year = str(text.get("year", "2027"))
        puzzle_count_label = str(text.get("puzzle_count_label", "1000+"))
        title_word = str(text.get("title_word", "SUDOKU"))
        title_joiner = str(text.get("title_joiner", ""))
        difficulty_label = str(text.get("difficulty_label", "MEDIUM TO HARD"))

        main_givens81 = str(resolved_puzzle_art.get("main_givens81") or "")
        left_side_givens81 = str(resolved_puzzle_art.get("left_side_givens81") or "")
        right_side_givens81 = str(resolved_puzzle_art.get("right_side_givens81") or "")

        # Model-matched default palette.
        # These can now be overridden by cover_design.variables.palette.
        navy = _hex_to_rgba(str(palette.get("navy", "#04326D")), (4, 50, 109, 255))
        navy_dark = _hex_to_rgba(str(palette.get("navy_dark", "#042346")), (4, 35, 70, 255))
        sky = _hex_to_rgba(str(palette.get("sky", "#84CFF6")), (132, 207, 246, 255))
        white = _hex_to_rgba(str(palette.get("white", "#FFFFFF")), (255, 255, 255, 255))

        img = Image.new("RGBA", (width_px, height_px), sky)
        draw = ImageDraw.Draw(img, "RGBA")

        # Background.
        draw.rectangle([0, 0, width_px, height_px], fill=sky)

        # No global texture: the outer light-blue edge bands must stay flat.

        # ============================================================
        # FINAL GLOBAL LAYOUT MODEL
        # ============================================================
        # Framed Column System:
        #   Bands 1, 3, 4, 5 live between the inner edges of the white pillars.
        # Full-Bleed Overlay System:
        #   Band 2 / Year Ribbon spans the full canvas width.

        pillar_w = int(width_px * 0.020)

        left_pillar_x0 = int(width_px * 0.038)
        left_pillar_x1 = left_pillar_x0 + pillar_w

        right_pillar_x1 = int(width_px * 0.962)
        right_pillar_x0 = right_pillar_x1 - pillar_w

        # Inner framed column bounds.
        panel_x0 = left_pillar_x1
        panel_x1 = right_pillar_x0

        # White vertical pillars first. These are behind the full-width year ribbon.
        # No side-cast shadows: only the year-ribbon white bands keep shadows.
        draw.rectangle([left_pillar_x0, 0, left_pillar_x1, height_px], fill=white)
        draw.rectangle([right_pillar_x0, 0, right_pillar_x1, height_px], fill=white)

        # ============================================================
        # BAND 1 — TOP CAP
        # Inside pillars only.
        # Top edge: almost straight, tiny downward sag.
        # Bottom edge: concave downward / center lower.
        # ============================================================
        _draw_band_between_curves(
            img,
            x0=panel_x0,
            x1=panel_x1,
            top_y=0,
            top_amp=8,
            bottom_y=760,
            bottom_amp=48,
            fill=navy,
        )

        # Subtle top darkening inside top cap.
        shade = Image.new("RGBA", img.size, (0, 0, 0, 0))
        sd = ImageDraw.Draw(shade, "RGBA")
        for i in range(260):
            alpha = int(76 * (1 - i / 260))
            sd.rectangle([panel_x0, i, panel_x1, i + 1], fill=(0, 0, 0, alpha))
        img.alpha_composite(shade)

        # ============================================================
        # LIGHT-BLUE PUZZLE ARENA
        # Under Band 3 and above Band 4.
        # ============================================================
        _draw_light_blue_puzzle_arena(
            img,
            panel_x0=panel_x0,
            panel_x1=panel_x1,
            top_y=1190,
            bottom_y=2660,
            top_amplitude=-48,
            bottom_amplitude=44,
            fill=sky,
            pattern_enabled=(
                _feature_enabled(features, "arena_pattern_enabled", True)
                and texture_id != "clean_flat"
            ),
        )

        # ============================================================
        # BAND 3 — TITLE RIBBON
        # Inside pillars only.
        # Top edge: concave upward / center higher.
        # Bottom edge: concave upward / nearly parallel.
        # Uniform strip, not inflated.
        # ============================================================
        _draw_band_between_curves(
            img,
            x0=panel_x0,
            x1=panel_x1,
            top_y=735,
            top_amp=-38,
            bottom_y=1215,
            bottom_amp=-58,
            fill=navy,
        )

        # ============================================================
        # BAND 4 — DIFFICULTY RIBBON
        # Inside pillars only.
        # Top edge: concave downward / center lower.
        # Bottom edge: concave downward / nearly parallel.
        # ============================================================
        _draw_band_between_curves(
            img,
            x0=panel_x0,
            x1=panel_x1,
            top_y=2655,
            top_amp=54,
            bottom_y=3070,
            bottom_amp=54,
            fill=navy,
        )

        # ============================================================
        # WHITE SEPARATOR BETWEEN BAND 4 AND BAND 5
        # ============================================================
        # Band 4 bottom edge:
        #   bottom_y=3070, bottom_amp=54
        #
        # The separator is a true parallel curved strip:
        #   top edge    = Band 4 bottom edge
        #   bottom edge = same curve shifted down by 52 px
        #
        # 52 px matches the main white ribbon thickness used elsewhere.
        _draw_band_between_curves(
            img,
            x0=panel_x0,
            x1=panel_x1,
            top_y=3070,
            top_amp=54,
            bottom_y=3122,
            bottom_amp=54,
            fill=white,
        )

        # ============================================================
        # BAND 5 — BOTTOM CAP
        # Inside pillars only.
        # Top edge exactly matches the white separator bottom edge,
        # making it perfectly parallel to Band 4's bottom edge.
        # Bottom edge is cropped by canvas.
        # ============================================================
        _draw_band_between_curves(
            img,
            x0=panel_x0,
            x1=panel_x1,
            top_y=3122,
            top_amp=54,
            bottom_y=height_px + 80,
            bottom_amp=0,
            fill=navy,
        )

        # ============================================================
        # BAND 2 — YEAR RIBBON SYSTEM
        # Full-bleed overlay. This is the only band crossing the pillars.
        # Structure:
        #   top white band ∩
        #   navy inflated ribbon body
        #   bottom white band ∪
        # ============================================================

        # Adjustable year-ribbon widening.
        # Increase this value to push the year ribbon bottom edge downward.
        # Decrease it to bring the bottom edge back up.
        year_ribbon_bottom_push_px = 40

        # Year navy body: full width and inflated because its top/bottom
        # curves move away from each other at center.
        _draw_band_between_curves(
            img,
            x0=0,
            x1=width_px,
            top_y=135,
            top_amp=42,
            bottom_y=735 + year_ribbon_bottom_push_px,
            bottom_amp=-52,
            fill=navy,
        )

        # Top white band: full width, concave downward, shadow cast upward.
        _draw_curved_white_band(
            img,
            x0=0,
            x1=width_px,
            center_y=150,
            amplitude=62,
            thickness=50,
            fill=white,
            shadow_offset_y=-18,
            shadow_alpha=42 if _feature_enabled(features, "year_ribbon_shadows_enabled", True) else 0,
            shadow_blur=28,
        )

        # Bottom white band: full width, concave upward, shadow cast downward.
        _draw_curved_white_band(
            img,
            x0=0,
            x1=width_px,
            center_y=742 + year_ribbon_bottom_push_px,
            amplitude=-62,
            thickness=50,
            fill=white,
            shadow_offset_y=26,
            shadow_alpha=58 if _feature_enabled(features, "year_ribbon_shadows_enabled", True) else 0,
            shadow_blur=34,
        )

        # Year text knobs.
        year_font_size = 480        # increase = larger digits
        year_tracking = 60          # increase = more space between digits
        year_x_offset = 0           # negative = left, positive = right

        # Year ribbon body:
        #   top_y=135, top_amp=42
        #   bottom_y=735 + year_ribbon_bottom_push_px, bottom_amp=-52
        #
        # Centerline:
        #   edge_y = average top/bottom edge y
        #   amp    = average top/bottom curve amplitude
        year_centerline_y_delta = 5       # negative = up, positive = down
        year_centerline_amp_delta = 0       # negative = center higher, positive = center lower

        year_centerline_edge_y = int(round((135 + (735 + year_ribbon_bottom_push_px)) / 2)) + year_centerline_y_delta
        year_centerline_amplitude = int(round((42 + -52) / 2)) + year_centerline_amp_delta

        # Year shadow knobs.
        year_shadow_dx = 24          # positive = shadow moves right
        year_shadow_dy = 24          # positive = shadow moves down
        year_shadow_blur = 8         # larger = softer shadow
        year_shadow_alpha = 190      # larger = darker shadow
        year_shadow_color = (0, 13, 30)

        year_font = _font(year_font_size, bold=True)

        _draw_centered_text_on_band_centerline(
            img,
            text=year,
            font=year_font,
            fill=(255, 255, 255, 255),
            canvas_width=width_px,
            band_x0=0,
            band_x1=width_px,
            centerline_edge_y=year_centerline_edge_y,
            centerline_amplitude=year_centerline_amplitude,
            tracking=year_tracking,
            x_offset=year_x_offset,
            y_offset=0,
            rotate_to_curve=True,
            rotation_strength=0.95,
            max_rotation_degrees=5.0,
            shadow_fill=(
                (*year_shadow_color, year_shadow_alpha)
                if _feature_enabled(features, "year_text_shadow_enabled", True)
                else None
            ),
            shadow_offset=(year_shadow_dx, year_shadow_dy),
            shadow_blur=year_shadow_blur,
        )

        # Main title.
        # Vertically centered on Band 3's own curved centerline.
        #
        # Band 3:
        #   top_y=735,  top_amp=-38
        #   bottom_y=1215, bottom_amp=-58
        #
        # Centerline:
        #   edge_y = (735 + 1215) / 2 = 975
        #   amp    = (-38 + -58) / 2 = -48
        title = f"{puzzle_count_label}{title_joiner}{title_word}"

        # Band 3 title knobs.
        title_font_size = 282       # increase = larger title
        title_tracking = 6          # increase = wider spread
        title_x_offset = 0          # negative = left, positive = right
        title_centerline_y = 970    # smaller = up, larger = down

        title_font = _font(title_font_size, bold=True)

        _draw_centered_text_on_band_centerline(
            img,
            text=title,
            font=title_font,
            fill=(255, 255, 255, 255),
            canvas_width=width_px,
            band_x0=panel_x0,
            band_x1=panel_x1,
            centerline_edge_y=title_centerline_y,
            centerline_amplitude=-48,
            tracking=title_tracking,
            x_offset=title_x_offset,
            y_offset=0,
            rotate_to_curve=True,
            rotation_strength=0.95,
            max_rotation_degrees=5.5,
            shadow_fill=(
                (0, 13, 30, 85)
                if _feature_enabled(features, "title_text_shadow_enabled", True)
                else None
            ),
            shadow_offset=(5, 7),
            shadow_blur=4,
        )

        # Side perspective panels behind central board.
        side_sample_left = (
            "901000000"
            "600000000"
            "300700000"
            "700000000"
            "800900000"
            "400000000"
            "500060000"
            "000000000"
            "174000000"
        )
        side_sample_right = (
            "000000000"
            "000000000"
            "000000038"
            "000000030"
            "000000050"
            "000000064"
            "000000020"
            "000000000"
            "000000700"
        )

        # Side perspective grids.
        # Built from the same exact grid design as the hero grid,
        # then warped so only the outer Sudoku columns remain visible
        # after the hero grid is drawn on top.
        if _feature_enabled(features, "side_panels_enabled", True):
            side_grid_source_size = 1080

            # ========================================================
            # SIDE GRID TUNING KNOBS
            # ========================================================
            side_grid_alpha = 205

            # Move both side grids up/down.
            # Negative = up, positive = down.
            side_grid_y_offset = 40

            # Controls visible side-grid height.
            side_grid_height = 1130

            # Controls top/bottom slopes.
            side_top_slope_px = 100
            side_bottom_slope_px = 160

            # Horizontal push controls.
            # LEFT: positive = push right / hide more behind hero grid.
            #       negative = pull left / expose more.
            left_side_grid_x_push = 6

            # RIGHT: positive = push left / hide more behind hero grid.
            #        negative = pull right / expose more.
            right_side_grid_x_push = 6

            left_visible_outer_x = int(width_px * 0.070) + left_side_grid_x_push
            left_hidden_inner_x = int(width_px * 0.435) + left_side_grid_x_push

            right_hidden_inner_x = int(width_px * 0.565) - right_side_grid_x_push
            right_visible_outer_x = int(width_px * 0.930) - right_side_grid_x_push

            side_top_outer_y = 1325 + side_grid_y_offset
            side_bottom_outer_y = side_top_outer_y + side_grid_height

            left_top_inner_y = side_top_outer_y - side_top_slope_px
            right_top_inner_y = side_top_outer_y - side_top_slope_px

            left_bottom_inner_y = side_bottom_outer_y + side_bottom_slope_px
            right_bottom_inner_y = side_bottom_outer_y + side_bottom_slope_px

            left_source = _make_side_grid(
                side_grid_source_size,
                left_side_givens81 if len(left_side_givens81) == 81 else side_sample_left,
            )
            right_source = _make_side_grid(
                side_grid_source_size,
                right_side_givens81 if len(right_side_givens81) == 81 else side_sample_right,
            )

            left_source.putalpha(side_grid_alpha)
            right_source.putalpha(side_grid_alpha)

            left_quad = [
                (left_visible_outer_x, side_top_outer_y),
                (left_hidden_inner_x, left_top_inner_y),
                (left_hidden_inner_x, left_bottom_inner_y),
                (left_visible_outer_x, side_bottom_outer_y),
            ]

            right_quad = [
                (right_hidden_inner_x, right_top_inner_y),
                (right_visible_outer_x, side_top_outer_y),
                (right_visible_outer_x, side_bottom_outer_y),
                (right_hidden_inner_x, right_bottom_inner_y),
            ]

            # --------------------------------------------------------
            # Soft lower-edge shadows for side grids.
            # Layer order:
            #   arena
            #   lower-edge shadows
            #   side grids
            #   arena radial shadow
            #   hero grid
            # --------------------------------------------------------
            side_lower_shadow_enabled = True

            if side_lower_shadow_enabled:
                # Tuning knobs.
                side_lower_shadow_alpha = 70       # stronger/lighter shadow
                side_lower_shadow_blur = 18        # softer/harder shadow
                side_lower_shadow_drop_px = 34     # how far shadow falls below the grid
                side_lower_shadow_thickness_px = 46  # vertical spread below lower border
                side_lower_shadow_color = (0, 12, 30)

                shadow_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
                sd = ImageDraw.Draw(shadow_layer, "RGBA")

                def shifted(p: tuple[int, int], dx: int = 0, dy: int = 0) -> tuple[int, int]:
                    return (p[0] + dx, p[1] + dy)

                # Left lower border shadow: follows the left grid bottom edge.
                left_shadow_poly = [
                    shifted(left_quad[3], dy=side_lower_shadow_drop_px),
                    shifted(left_quad[2], dy=side_lower_shadow_drop_px),
                    shifted(left_quad[2], dy=side_lower_shadow_drop_px + side_lower_shadow_thickness_px),
                    shifted(left_quad[3], dy=side_lower_shadow_drop_px + side_lower_shadow_thickness_px),
                ]

                # Right lower border shadow: follows the right grid bottom edge.
                right_shadow_poly = [
                    shifted(right_quad[3], dy=side_lower_shadow_drop_px),
                    shifted(right_quad[2], dy=side_lower_shadow_drop_px),
                    shifted(right_quad[2], dy=side_lower_shadow_drop_px + side_lower_shadow_thickness_px),
                    shifted(right_quad[3], dy=side_lower_shadow_drop_px + side_lower_shadow_thickness_px),
                ]

                sd.polygon(
                    left_shadow_poly,
                    fill=(*side_lower_shadow_color, side_lower_shadow_alpha),
                )
                sd.polygon(
                    right_shadow_poly,
                    fill=(*side_lower_shadow_color, side_lower_shadow_alpha),
                )

                shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(side_lower_shadow_blur))
                img.alpha_composite(shadow_layer)

            left_panel = _perspective_warp_quad(left_source, left_quad, img.size)
            right_panel = _perspective_warp_quad(right_source, right_quad, img.size)

            img.alpha_composite(left_panel)
            img.alpha_composite(right_panel)

            edge_draw = ImageDraw.Draw(img, "RGBA")

            edge_draw.line(
                [left_quad[0], left_quad[3]],
                fill=(20, 55, 80, 145),
                width=5,
            )
            edge_draw.line(
                [right_quad[1], right_quad[2]],
                fill=(20, 55, 80, 145),
                width=5,
            )


            edge_draw.line(
                [right_quad[1], right_quad[2]],
                fill=(20, 55, 80, 145),
                width=5,
            )

        # Main hero Sudoku grid.
        # Tuned to match the model: large centered white 9x9 board,
        # black outer border, heavier 3x3 separators, thin inner cell lines,
        # large regular-weight digits, and a soft lower-right cast shadow.
        
        hero_grid_size = int(width_px * 0.640)
        hero_grid_y = 1120

        # Fine-tuning knobs:
        hero_grid_size_delta = 0
        hero_grid_y_delta = 0

        hero_grid_size += hero_grid_size_delta
        hero_grid_x = (width_px - hero_grid_size) // 2
        hero_grid_y += hero_grid_y_delta

        # Elliptic arena shadow.
        # Layer order:
        #   1) arena + side grids already drawn
        #   2) this broad radial shadow darkens the side grids
        #   3) hero grid is drawn after and hides the strongest center
        arena_shadow_enabled = _feature_enabled(features, "arena_shadow_enabled", True)

        if arena_shadow_enabled:
            # Centered on the puzzle arena, not on the full page.
            arena_shadow_center_x = (panel_x0 + panel_x1) // 2
            arena_shadow_center_y = (1190 + 2660) // 2

            # Tuning knobs.
            # Wider radius + lower power makes the shadow still visible
            # on the exposed left/right side grids instead of disappearing
            # under the hero grid only.
            arena_shadow_radius_x = int(width_px * 0.620)
            arena_shadow_radius_y = 820
            arena_shadow_alpha = 70
            arena_shadow_power = 1.05
            arena_shadow_blur = 34
            arena_shadow_color = (60, 40, 120)

            _draw_elliptic_radial_shadow(
                img,
                center=(arena_shadow_center_x, arena_shadow_center_y),
                radius_x=arena_shadow_radius_x,
                radius_y=arena_shadow_radius_y,
                color=arena_shadow_color,
                max_alpha=arena_shadow_alpha,
                power=arena_shadow_power,
                blur=arena_shadow_blur,
            )

        if _feature_enabled(features, "hero_grid_enabled", True):
            _draw_sudoku_grid(
                img,
                hero_grid_x,
                hero_grid_y,
                hero_grid_size,
                givens81=main_givens81 if len(main_givens81) == 81 else None,
                fill=(255, 255, 255, 255),
                line=(0, 0, 0, 255),
                digit=(0, 0, 0, 255),
                outer_width=7,
                box_width=10,
                cell_width=2,
                digit_scale=0.78,
            )
        

        # Bottom difficulty text.
        # Vertically centered on Band 4's own curved centerline.
        #
        # Band 4:
        #   top_y=2655,  top_amp=54
        #   bottom_y=3070, bottom_amp=54
        #
        # Centerline:
        #   edge_y = (2655 + 3070) / 2 = 2862.5 ≈ 2863
        #   amp    = 54
        # Keep the English design size, but protect longer localized labels.
        #
        # The original max width used 90% of the whole canvas:
        #     int(width_px * 0.900)
        #
        # That is too generous for translated labels because Band 4 does not span
        # the whole canvas. It lives inside the framed column:
        #     panel_x0 .. panel_x1
        #
        # We therefore keep the old behavior for labels up to the English length,
        # and switch to a safer band-inside width for longer market labels such as:
        #     MITTEL BIS SCHWER
        #     MOYEN À DIFFICILE
        #     MEDIO A DIFÍCIL
        difficulty_text = difficulty_label.strip()
        english_threshold_len = len("MEDIUM TO HARD")

        difficulty_base_max_width = int(width_px * 0.980)
        difficulty_safe_max_width = max(1, (panel_x1 - panel_x0) - 180)

        difficulty_fit_max_width = (
            difficulty_base_max_width
            if len(difficulty_text) <= english_threshold_len
            else min(difficulty_base_max_width, difficulty_safe_max_width)
        )

        difficulty_font = _fit_font(
            draw,
            difficulty_label,
            difficulty_fit_max_width,
            start_size=235,
            min_size=120,
            bold=True,
        )
        _draw_centered_text_on_band_centerline(
            img,
            text=difficulty_label,
            font=difficulty_font,
            fill=(255, 255, 255, 255),
            canvas_width=width_px,
            band_x0=panel_x0,
            band_x1=panel_x1,
            # Slightly below centerline, using Band 4 bottom-edge curvature.
            # Band 4 bottom edge is: edge_y=3070, amplitude=54.
            # This guide line sits 90 px above that bottom edge.
            centerline_edge_y=2900,
            centerline_amplitude=54,
            tracking=0,
            y_offset=0,
            rotate_to_curve=True,
            rotation_strength=0.95,
            max_rotation_degrees=6.0,
            shadow_fill=(
                (0, 13, 30, 125)
                if _feature_enabled(features, "difficulty_text_shadow_enabled", True)
                else None
            ),
            shadow_offset=(7, 9),
            shadow_blur=4,
        )

        output_file = out_path / "front_cover.png"
        img.convert("RGB").save(output_file, quality=95)

        return CoverRenderResult(
            front_cover_png=output_file,
            width_px=width_px,
            height_px=height_px,
            renderer_key=self.renderer_key,
        )