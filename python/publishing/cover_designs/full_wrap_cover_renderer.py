from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from reportlab.lib import utils as reportlab_utils
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfdoc
from reportlab.pdfgen import canvas

from python.publishing.pdf_renderer.typography import resolve_font_pack

# Compatibility fix for some Python/OpenSSL/ReportLab combinations.
# Some ReportLab versions call md5(data, usedforsecurity=False), while
# some hashlib/OpenSSL builds accept only md5(data).
import hashlib


def _md5_compat(data: Any = b"", *args: Any, **kwargs: Any) -> Any:
    kwargs.pop("usedforsecurity", None)
    return hashlib.md5(data)


pdfdoc.md5 = _md5_compat
reportlab_utils.md5 = _md5_compat


@dataclass(frozen=True)
class FullWrapGeometry:
    trim_width_in: float
    trim_height_in: float
    bleed_in: float
    spine_width_in: float
    full_width_in: float
    full_height_in: float

    back_x_in: float
    spine_x_in: float
    front_x_in: float
    trim_y_in: float


def _paper_thickness_in(paper_type: str) -> float:
    paper_type = (paper_type or "").lower()

    if paper_type in {"cream", "cream_bw"}:
        return 0.0025

    if paper_type in {"premium_color", "standard_color", "color"}:
        return 0.002347

    # Amazon/KDP white B/W paperback approximation.
    return 0.002252


def build_full_wrap_geometry(
    *,
    page_count: int,
    paper_type: str,
    trim_width_in: float = 8.5,
    trim_height_in: float = 11.0,
    bleed_in: float = 0.125,
) -> FullWrapGeometry:
    spine_width_in = max(0.0, page_count * _paper_thickness_in(paper_type))

    full_width_in = bleed_in + trim_width_in + spine_width_in + trim_width_in + bleed_in
    full_height_in = bleed_in + trim_height_in + bleed_in

    back_x_in = bleed_in
    spine_x_in = back_x_in + trim_width_in
    front_x_in = spine_x_in + spine_width_in
    trim_y_in = bleed_in

    return FullWrapGeometry(
        trim_width_in=trim_width_in,
        trim_height_in=trim_height_in,
        bleed_in=bleed_in,
        spine_width_in=spine_width_in,
        full_width_in=full_width_in,
        full_height_in=full_height_in,
        back_x_in=back_x_in,
        spine_x_in=spine_x_in,
        front_x_in=front_x_in,
        trim_y_in=trim_y_in,
    )


def _inch(value: float) -> float:
    return value * 72.0


def _read_image_average_rgb(path: Path) -> tuple[int, int, int]:
    img = Image.open(path).convert("RGB").resize((1, 1))
    return img.getpixel((0, 0))


def _draw_panel_image(
    c: canvas.Canvas,
    image_path: Path,
    x_in: float,
    y_in: float,
    width_in: float,
    height_in: float,
) -> None:
    c.drawImage(
        ImageReader(str(image_path)),
        _inch(x_in),
        _inch(y_in),
        width=_inch(width_in),
        height=_inch(height_in),
        preserveAspectRatio=True,
        anchor="c",
        mask="auto",
    )


def render_full_wrap_cover_pdf(
    *,
    front_cover_png: Path,
    output_pdf: Path,
    page_count: int,
    paper_type: str,
    title: str = "",
    spine_text: str = "",
    back_copy: str = "",
    trim_width_in: float = 8.5,
    trim_height_in: float = 11.0,
    bleed_in: float = 0.125,
    show_barcode_placeholder: bool = False,
    back_cover_png: Path | None = None,
    spine_cover_png: Path | None = None,
) -> dict[str, Any]:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    geometry = build_full_wrap_geometry(
        page_count=page_count,
        paper_type=paper_type,
        trim_width_in=trim_width_in,
        trim_height_in=trim_height_in,
        bleed_in=bleed_in,
    )

    page_w_pt = _inch(geometry.full_width_in)
    page_h_pt = _inch(geometry.full_height_in)

    c = canvas.Canvas(str(output_pdf), pagesize=(page_w_pt, page_h_pt))

    # KDP-safe cover text fonts.
    #
    # Do not use ReportLab's built-in PDF fonts here ("Helvetica",
    # "Helvetica-Bold", etc.). Those standard PDF fonts may appear as
    # unembedded in KDP Previewer.
    #
    # resolve_font_pack("arial") registers Arial TrueType files from
    # C:/Windows/Fonts when available, which lets ReportLab embed the font
    # into the generated cover PDF.
    cover_fonts = resolve_font_pack("arial")

    # Use the front image's corner color as a safe bleed/background color.
    bg_r, bg_g, bg_b = _read_image_average_rgb(front_cover_png)
    c.setFillColorRGB(bg_r / 255.0, bg_g / 255.0, bg_b / 255.0)
    c.rect(0, 0, page_w_pt, page_h_pt, stroke=0, fill=1)

    # Back panel.
    #
    # If a renderer provides a custom back_cover_png, place it directly on the
    # back panel. Otherwise keep the legacy generic navy placeholder behavior.
    if back_cover_png is not None:
        _draw_panel_image(
            c,
            Path(back_cover_png),
            geometry.back_x_in,
            geometry.trim_y_in,
            geometry.trim_width_in,
            geometry.trim_height_in,
        )
    else:
        c.setFillColorRGB(4 / 255.0, 50 / 255.0, 109 / 255.0)
        c.rect(
            _inch(geometry.back_x_in),
            _inch(geometry.trim_y_in),
            _inch(geometry.trim_width_in),
            _inch(geometry.trim_height_in),
            stroke=0,
            fill=1,
        )

    # Spine panel.
    #
    # If a renderer provides a custom spine_cover_png, place it directly on the
    # spine panel. Otherwise keep the legacy generic dark navy placeholder.
    if spine_cover_png is not None and geometry.spine_width_in > 0:
        _draw_panel_image(
            c,
            Path(spine_cover_png),
            geometry.spine_x_in,
            geometry.trim_y_in,
            geometry.spine_width_in,
            geometry.trim_height_in,
        )
    else:
        c.setFillColorRGB(4 / 255.0, 35 / 255.0, 70 / 255.0)
        c.rect(
            _inch(geometry.spine_x_in),
            _inch(geometry.trim_y_in),
            _inch(geometry.spine_width_in),
            _inch(geometry.trim_height_in),
            stroke=0,
            fill=1,
        )

    # Front panel: place generated front cover artwork.
    _draw_panel_image(
        c,
        Path(front_cover_png),
        geometry.front_x_in,
        geometry.trim_y_in,
        geometry.trim_width_in,
        geometry.trim_height_in,
    )

    # Back copy placeholder, intentionally simple for Series F.
    # Do not draw generic back copy over a custom renderer-provided back panel.
    if back_copy and back_cover_png is None:
        c.setFillColorRGB(1, 1, 1)
        c.setFont(cover_fonts.regular, 16)
        text = c.beginText()
        text.setTextOrigin(_inch(geometry.back_x_in + 0.55), _inch(geometry.trim_y_in + 9.6))
        text.setLeading(22)

        for raw_line in back_copy.splitlines():
            line = raw_line.strip()
            if not line:
                text.textLine("")
                continue
            while len(line) > 62:
                cut = line.rfind(" ", 0, 62)
                if cut <= 0:
                    cut = 62
                text.textLine(line[:cut].strip())
                line = line[cut:].strip()
            text.textLine(line)

        c.drawText(text)

    # Spine text.
    #
    # Platform rule:
    # KDP-style paperback spines should receive printed spine text only when
    # the interior has enough pages to make the spine safely readable.
    # For this publishing platform, the threshold is 150 pages.
    effective_spine_text = str(spine_text or "").strip()
    if page_count < 150:
        effective_spine_text = ""

    if spine_cover_png is None and effective_spine_text and geometry.spine_width_in >= 0.12:
        c.saveState()
        c.translate(
            _inch(geometry.spine_x_in + geometry.spine_width_in / 2),
            _inch(geometry.trim_y_in + geometry.trim_height_in / 2),
        )
        c.rotate(90)
        c.setFillColorRGB(1, 1, 1)
        c.setFont(cover_fonts.bold, 11)
        c.drawCentredString(0, -4, effective_spine_text[:80])
        c.restoreState()

    # Optional barcode placeholder.
    # Default is False because KDP usually wants the barcode area controlled
    # by the final upload/export rules, not always drawn into every cover.
    if show_barcode_placeholder and back_cover_png is None:
        barcode_w = 2.0
        barcode_h = 1.2
        barcode_x = geometry.back_x_in + geometry.trim_width_in - barcode_w - 0.45
        barcode_y = geometry.trim_y_in + 0.45

        c.setFillColorRGB(1, 1, 1)
        c.rect(_inch(barcode_x), _inch(barcode_y), _inch(barcode_w), _inch(barcode_h), stroke=0, fill=1)
        c.setStrokeColorRGB(0.65, 0.65, 0.65)
        c.rect(_inch(barcode_x), _inch(barcode_y), _inch(barcode_w), _inch(barcode_h), stroke=1, fill=0)
        c.setFillColorRGB(0.25, 0.25, 0.25)
        c.setFont(cover_fonts.regular, 8)
        c.drawCentredString(
            _inch(barcode_x + barcode_w / 2),
            _inch(barcode_y + barcode_h / 2),
            "BARCODE",
        )

    c.showPage()
    c.save()

    metadata = {
        "output_pdf": str(output_pdf),
        "front_cover_png": str(front_cover_png),
        "back_cover_png": str(back_cover_png) if back_cover_png is not None else None,
        "spine_cover_png": str(spine_cover_png) if spine_cover_png is not None else None,
        "page_count": page_count,
        "paper_type": paper_type,
        "geometry": asdict(geometry),
    }

    metadata_path = output_pdf.with_suffix(".geometry.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    return metadata