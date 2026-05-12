from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from python.publishing.cover_designs.models import ResolvedCoverDesignContext
from python.publishing.cover_renderers.base_renderer import BaseCoverRenderer, CoverRenderResult


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


def _hex_to_rgba(value: str, fallback: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    raw = str(value or "").strip().lstrip("#")
    if len(raw) != 6:
        return fallback
    try:
        return (int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16), 255)
    except ValueError:
        return fallback


def _fit_font(draw: ImageDraw.ImageDraw, text: str, max_width: int, start_size: int, min_size: int, bold: bool = True) -> ImageFont.ImageFont:
    size = start_size
    while size >= min_size:
        f = _font(size, bold=bold)
        bbox = draw.textbbox((0, 0), text, font=f)
        if bbox[2] - bbox[0] <= max_width:
            return f
        size -= 4
    return _font(min_size, bold=bold)


def _center_text(draw: ImageDraw.ImageDraw, text: str, y: int, font: ImageFont.ImageFont, fill: tuple[int, int, int, int], width: int) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    x = (width - (bbox[2] - bbox[0])) // 2
    draw.text((x, y), text, font=font, fill=fill)


def _draw_sudoku_grid(
    img: Image.Image,
    x: int,
    y: int,
    size: int,
    givens81: str | None,
    *,
    fill: tuple[int, int, int, int],
    line: tuple[int, int, int, int],
    digit: tuple[int, int, int, int],
) -> None:
    draw = ImageDraw.Draw(img, "RGBA")
    draw.rectangle([x, y, x + size, y + size], fill=fill, outline=line, width=8)

    cell = size / 9.0
    for i in range(10):
        w = 7 if i % 3 == 0 else 2
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

    f = _font(max(28, int(cell * 0.72)), bold=False)
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
        draw.text((int(cx - tw / 2 - bbox[0]), int(cy - th / 2 - bbox[1])), ch, font=f, fill=digit)


def _variables(context: ResolvedCoverDesignContext, key: str) -> dict[str, Any]:
    return dict(context.variables.get(key, {}) or {})


class MinimalWhiteGridV1Renderer(BaseCoverRenderer):
    renderer_key = "minimal_white_grid_v1"

    def render_front_cover(
        self,
        context: ResolvedCoverDesignContext,
        out_dir: str | Path,
        width_px: int = 2550,
        height_px: int = 3300,
    ) -> CoverRenderResult:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        text = _variables(context, "text")
        palette = _variables(context, "palette")
        resolved_puzzle_art = _variables(context, "resolved_puzzle_art")

        year = str(text.get("year", "2027"))
        puzzle_count_label = str(text.get("puzzle_count_label", "1000+"))
        title_word = str(text.get("title_word", "SUDOKU"))
        difficulty_label = str(text.get("difficulty_label", "MEDIUM TO HARD"))

        background = _hex_to_rgba(str(palette.get("background", "#F8FAFC")), (248, 250, 252, 255))
        ink = _hex_to_rgba(str(palette.get("ink", "#111827")), (17, 24, 39, 255))
        accent = _hex_to_rgba(str(palette.get("accent", "#0B4F9C")), (11, 79, 156, 255))
        muted = _hex_to_rgba(str(palette.get("muted", "#E5E7EB")), (229, 231, 235, 255))
        white = (255, 255, 255, 255)

        main_givens81 = str(resolved_puzzle_art.get("main_givens81") or "")

        img = Image.new("RGBA", (width_px, height_px), background)
        draw = ImageDraw.Draw(img, "RGBA")

        # Thin collection frame.
        margin = 110
        draw.rectangle([margin, margin, width_px - margin, height_px - margin], outline=accent, width=12)

        # Year.
        year_font = _fit_font(draw, year, int(width_px * 0.55), 250, 150, bold=True)
        _center_text(draw, year, 260, year_font, accent, width_px)

        # Title.
        title = f"{puzzle_count_label} {title_word}".strip()
        title_font = _fit_font(draw, title, int(width_px * 0.80), 170, 90, bold=True)
        _center_text(draw, title, 650, title_font, ink, width_px)

        # Divider.
        draw.line([int(width_px * 0.18), 890, int(width_px * 0.82), 890], fill=muted, width=8)

        # Hero grid.
        grid_size = int(width_px * 0.66)
        grid_x = (width_px - grid_size) // 2
        grid_y = 1040

        shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
        sd = ImageDraw.Draw(shadow, "RGBA")
        sd.rectangle([grid_x + 28, grid_y + 32, grid_x + grid_size + 28, grid_y + grid_size + 32], fill=(0, 0, 0, 45))
        shadow = shadow.filter(ImageFilter.GaussianBlur(18))
        img.alpha_composite(shadow)

        _draw_sudoku_grid(
            img,
            grid_x,
            grid_y,
            grid_size,
            main_givens81 if len(main_givens81) == 81 else None,
            fill=white,
            line=ink,
            digit=ink,
        )

        # Difficulty footer.
        footer_y = 2760
        draw.rounded_rectangle(
            [int(width_px * 0.13), footer_y, int(width_px * 0.87), footer_y + 250],
            radius=36,
            fill=accent,
        )
        difficulty_font = _fit_font(draw, difficulty_label, int(width_px * 0.68), 145, 70, bold=True)
        _center_text(draw, difficulty_label, footer_y + 62, difficulty_font, white, width_px)

        output_file = out_path / "front_cover.png"
        img.convert("RGB").save(output_file, quality=95)

        return CoverRenderResult(
            front_cover_png=output_file,
            width_px=width_px,
            height_px=height_px,
            renderer_key=self.renderer_key,
        )