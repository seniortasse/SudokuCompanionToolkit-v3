from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.utils import ImageReader

from python.publishing.cover_builder.cover_asset_loader import resolve_cover_asset_path


def export_book_cover_pdf(
    *,
    publication_dir: Path,
    output_pdf_path: Path,
) -> Path:
    cover_manifest_path = publication_dir / "cover_manifest.json"
    if not cover_manifest_path.exists():
        raise FileNotFoundError(f"Missing cover_manifest.json in {publication_dir}")

    cover_manifest = json.loads(cover_manifest_path.read_text(encoding="utf-8"))
    geometry = cover_manifest["geometry"]

    page_width_pts = float(geometry["total_width_in"]) * 72.0
    page_height_pts = float(geometry["total_height_in"]) * 72.0

    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = Canvas(str(output_pdf_path), pagesize=(page_width_pts, page_height_pts))
    canvas.setTitle(f"{cover_manifest.get('publication_id', 'Sudoku Cover')} - Cover")

    _render_cover(canvas, cover_manifest, publication_dir=publication_dir)

    canvas.save()
    return output_pdf_path


def _render_cover(canvas: Canvas, cover_manifest: dict, *, publication_dir: Path) -> None:
    geometry = cover_manifest["geometry"]

    back = _panel_to_pts(geometry["back_panel"])
    spine = _panel_to_pts(geometry["spine_panel"])
    front = _panel_to_pts(geometry["front_panel"])
    barcode_box = _box_to_pts(geometry["barcode_box"])

    page_width, page_height = canvas._pagesize
    template_id = str(cover_manifest.get("cover_template", "basic_full_wrap")).strip().lower()

    if template_id == "basic_full_wrap":
        _render_basic_full_wrap(canvas, cover_manifest, publication_dir, back, spine, front, barcode_box)
    elif template_id == "bold_year_banner":
        _render_bold_year_banner(canvas, cover_manifest, publication_dir, back, spine, front, barcode_box)
    else:
        _render_basic_full_wrap(canvas, cover_manifest, publication_dir, back, spine, front, barcode_box)

    _draw_trim_guides(canvas, geometry)
    _draw_safe_boxes(canvas, geometry)


def _render_basic_full_wrap(
    canvas: Canvas,
    cover_manifest: dict,
    publication_dir: Path,
    back: dict,
    spine: dict,
    front: dict,
    barcode_box: dict,
) -> None:
    page_width, page_height = canvas._pagesize

    canvas.setFillColor(colors.HexColor("#eaf0fb"))
    canvas.rect(0, 0, page_width, page_height, stroke=0, fill=1)

    canvas.setFillColor(colors.HexColor("#d8e4fa"))
    canvas.rect(back["x"], back["y"], back["w"], back["h"], stroke=0, fill=1)

    canvas.setFillColor(colors.HexColor("#3159a6"))
    canvas.rect(spine["x"], spine["y"], spine["w"], spine["h"], stroke=0, fill=1)

    canvas.setFillColor(colors.HexColor("#1f3c88"))
    canvas.rect(front["x"], front["y"], front["w"], front["h"], stroke=0, fill=1)

    _draw_front_asset_if_present(canvas, cover_manifest, publication_dir, front)
    _draw_back_asset_if_present(canvas, cover_manifest, publication_dir, back)

    _render_front_panel(canvas, cover_manifest, front)
    _render_back_panel(canvas, cover_manifest, back, barcode_box)
    _render_spine(canvas, cover_manifest, spine)


def _render_bold_year_banner(
    canvas: Canvas,
    cover_manifest: dict,
    publication_dir: Path,
    back: dict,
    spine: dict,
    front: dict,
    barcode_box: dict,
) -> None:
    page_width, page_height = canvas._pagesize

    canvas.setFillColor(colors.HexColor("#f4f7fd"))
    canvas.rect(0, 0, page_width, page_height, stroke=0, fill=1)

    canvas.setFillColor(colors.HexColor("#edf3ff"))
    canvas.rect(back["x"], back["y"], back["w"], back["h"], stroke=0, fill=1)

    canvas.setFillColor(colors.HexColor("#173872"))
    canvas.rect(spine["x"], spine["y"], spine["w"], spine["h"], stroke=0, fill=1)

    canvas.setFillColor(colors.HexColor("#214a95"))
    canvas.rect(front["x"], front["y"], front["w"], front["h"], stroke=0, fill=1)

    banner_h = 86
    canvas.setFillColor(colors.HexColor("#f7d34a"))
    canvas.rect(front["x"], front["y"] + front["h"] - banner_h, front["w"], banner_h, stroke=0, fill=1)

    _draw_front_asset_if_present(canvas, cover_manifest, publication_dir, front)
    _draw_back_asset_if_present(canvas, cover_manifest, publication_dir, back)

    _render_front_panel(canvas, cover_manifest, front, banner_mode=True)
    _render_back_panel(canvas, cover_manifest, back, barcode_box)
    _render_spine(canvas, cover_manifest, spine)


def _draw_front_asset_if_present(canvas: Canvas, cover_manifest: dict, publication_dir: Path, front: dict) -> None:
    asset = resolve_cover_asset_path(publication_dir, cover_manifest.get("front_design_asset"))
    if asset is None:
        return
    _draw_image_cover(canvas, asset, front["x"], front["y"], front["w"], front["h"])


def _draw_back_asset_if_present(canvas: Canvas, cover_manifest: dict, publication_dir: Path, back: dict) -> None:
    asset = resolve_cover_asset_path(publication_dir, cover_manifest.get("back_design_asset"))
    if asset is None:
        return
    _draw_image_cover(canvas, asset, back["x"], back["y"], back["w"], back["h"])


def _draw_image_cover(canvas: Canvas, image_path: Path, x: float, y: float, w: float, h: float) -> None:
    img = ImageReader(str(image_path))
    iw, ih = img.getSize()
    if iw <= 0 or ih <= 0:
        return

    scale = max(w / float(iw), h / float(ih))
    draw_w = iw * scale
    draw_h = ih * scale
    draw_x = x + (w - draw_w) / 2.0
    draw_y = y + (h - draw_h) / 2.0

    canvas.saveState()
    path = canvas.beginPath()
    path.rect(x, y, w, h)
    canvas.clipPath(path, stroke=0, fill=0)
    canvas.drawImage(img, draw_x, draw_y, width=draw_w, height=draw_h, mask="auto")
    canvas.restoreState()


def _render_front_panel(canvas: Canvas, cover_manifest: dict, front: dict, banner_mode: bool = False) -> None:
    safe = _panel_to_pts(cover_manifest["geometry"]["front_panel"]["safe_box"])

    title = str(cover_manifest.get("spine_text", "")).strip() or "Sudoku Companion"
    subtitle = str(cover_manifest.get("cover_template", "")).replace("_", " ").title()

    canvas.setFillColor(colors.white if not banner_mode else colors.HexColor("#173872"))
    canvas.setFont("Helvetica-Bold", 24)
    y = safe["y"] + safe["h"] - (74 if banner_mode else 40)
    for line in _wrap_text(title, width=20):
        canvas.drawString(safe["x"], y, line)
        y -= 28

    if subtitle:
        y -= 8
        canvas.setFont("Helvetica", 12)
        for line in _wrap_text(subtitle, width=28):
            canvas.drawString(safe["x"], y, line)
            y -= 16

    canvas.setFillColor(colors.HexColor("#dce7ff"))
    hero_y = safe["y"] + 70
    hero_h = max(100, safe["h"] * 0.42)

    if not str(cover_manifest.get("front_design_asset", "")).strip():
        canvas.rect(safe["x"], hero_y, safe["w"], hero_h, stroke=0, fill=1)
        canvas.setFillColor(colors.HexColor("#1f3c88"))
        canvas.setFont("Helvetica-Bold", 16)
        canvas.drawCentredString(
            safe["x"] + (safe["w"] / 2.0),
            hero_y + (hero_h / 2.0),
            "FRONT COVER ART AREA",
        )


def _render_back_panel(canvas: Canvas, cover_manifest: dict, back: dict, barcode_box: dict) -> None:
    safe = _panel_to_pts(cover_manifest["geometry"]["back_panel"]["safe_box"])
    body = str(cover_manifest.get("back_copy", "")).strip()
    imprint = str(cover_manifest.get("author_imprint", "")).strip()
    isbn = str(cover_manifest.get("isbn", "") or "").strip()

    canvas.setFillColor(colors.HexColor("#1f3c88"))
    canvas.setFont("Helvetica-Bold", 18)
    canvas.drawString(safe["x"], safe["y"] + safe["h"] - 28, "Back Cover")

    canvas.setFillColor(colors.black)
    canvas.setFont("Helvetica", 10)

    text = canvas.beginText(safe["x"], safe["y"] + safe["h"] - 54)
    text.setLeading(14)
    for line in _wrap_text(
        body or "Back-cover copy goes here. This area is reserved for the publication description and selling points.",
        width=44,
    ):
        text.textLine(line)
    canvas.drawText(text)

    if imprint:
        canvas.setFont("Helvetica-Bold", 10)
        canvas.drawString(safe["x"], safe["y"] + 18, imprint)

    canvas.setFillColor(colors.white)
    canvas.rect(barcode_box["x"], barcode_box["y"], barcode_box["w"], barcode_box["h"], stroke=1, fill=1)
    canvas.setFillColor(colors.black)
    canvas.setFont("Helvetica", 8)
    canvas.drawCentredString(
        barcode_box["x"] + (barcode_box["w"] / 2.0),
        barcode_box["y"] + (barcode_box["h"] / 2.0),
        isbn or "ISBN / BARCODE",
    )


def _render_spine(canvas: Canvas, cover_manifest: dict, spine: dict) -> None:
    if spine["w"] <= 10:
        return

    page_count = int(cover_manifest.get("page_count") or 0)
    if page_count < 150:
        return

    title = str(cover_manifest.get("spine_text", "")).strip()
    if not title:
        return

    safe = _panel_to_pts(cover_manifest["geometry"]["spine_panel"]["safe_box"])

    canvas.saveState()
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 10)
    canvas.translate(safe["x"] + (safe["w"] / 2.0), safe["y"] + 16)
    canvas.rotate(90)
    canvas.drawCentredString(0, 0, title)
    canvas.restoreState()


def _draw_trim_guides(canvas: Canvas, geometry: dict) -> None:
    bleed = float(geometry["bleed_in"]) * 72.0
    trim_w = float(geometry["trim_width_in"]) * 72.0
    trim_h = float(geometry["trim_height_in"]) * 72.0
    spine_w = float(geometry["spine_width_in"]) * 72.0

    canvas.setStrokeColor(colors.HexColor("#7a7a7a"))
    canvas.setLineWidth(0.4)

    canvas.rect(bleed, bleed, (trim_w * 2.0) + spine_w, trim_h, stroke=1, fill=0)

    x1 = bleed + trim_w
    x2 = bleed + trim_w + spine_w
    canvas.line(x1, bleed, x1, bleed + trim_h)
    canvas.line(x2, bleed, x2, bleed + trim_h)


def _draw_safe_boxes(canvas: Canvas, geometry: dict) -> None:
    canvas.setStrokeColor(colors.HexColor("#c0392b"))
    canvas.setLineWidth(0.4)

    for key in ("back_panel", "spine_panel", "front_panel"):
        safe = _panel_to_pts(geometry[key]["safe_box"])
        if safe["w"] > 0 and safe["h"] > 0:
            canvas.rect(safe["x"], safe["y"], safe["w"], safe["h"], stroke=1, fill=0)


def _panel_to_pts(panel: dict) -> dict:
    return {
        "x": float(panel["x_in"]) * 72.0,
        "y": float(panel["y_in"]) * 72.0,
        "w": float(panel["width_in"]) * 72.0,
        "h": float(panel["height_in"]) * 72.0,
    }


def _box_to_pts(box: dict) -> dict:
    return {
        "x": float(box["x_in"]) * 72.0,
        "y": float(box["y_in"]) * 72.0,
        "w": float(box["width_in"]) * 72.0,
        "h": float(box["height_in"]) * 72.0,
    }


def _wrap_text(value: str, width: int) -> list[str]:
    words = value.split()
    if not words:
        return []

    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines