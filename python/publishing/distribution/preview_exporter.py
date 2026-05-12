from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any


from reportlab.lib.colors import HexColor
from reportlab.graphics.shapes import Drawing, Rect, String

try:
    from reportlab.graphics import renderPM
except Exception as exc:  # pragma: no cover
    renderPM = None
    _RENDERPM_IMPORT_ERROR = exc
else:
    _RENDERPM_IMPORT_ERROR = None



def export_publication_previews(
    *,
    publication_dir: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    if renderPM is None:
        raise RuntimeError(
            "ReportLab renderPM preview backend is unavailable. "
            f"Original import error: {_RENDERPM_IMPORT_ERROR}"
    )

    publication_manifest_path = publication_dir / "publication_manifest.json"
    cover_manifest_path = publication_dir / "cover_manifest.json"
    interior_plan_path = publication_dir / "interior_plan.json"

    if not publication_manifest_path.exists():
        raise FileNotFoundError(f"Missing publication_manifest.json in {publication_dir}")
    if not interior_plan_path.exists():
        raise FileNotFoundError(f"Missing interior_plan.json in {publication_dir}")

    publication_manifest = json.loads(publication_manifest_path.read_text(encoding="utf-8"))
    interior_plan = json.loads(interior_plan_path.read_text(encoding="utf-8"))
    cover_manifest = None
    if cover_manifest_path.exists():
        cover_manifest = json.loads(cover_manifest_path.read_text(encoding="utf-8"))

    previews = {}

    cover_preview = output_dir / "preview_cover.png"
    if cover_manifest is not None:
        _render_cover_preview_png(cover_manifest, cover_preview)
        previews["cover_preview"] = str(cover_preview)

    interior_preview = output_dir / "preview_interior_plan.png"
    _render_interior_plan_preview_png(
        publication_manifest=publication_manifest,
        interior_plan=interior_plan,
        output_path=interior_preview,
    )
    previews["interior_plan_preview"] = str(interior_preview)

    return previews


def _render_cover_preview_png(cover_manifest: dict, output_path: Path) -> None:
    geometry = cover_manifest.get("geometry", {})
    total_w = float(geometry.get("total_width_in", 17.5))
    total_h = float(geometry.get("total_height_in", 11.25))

    scale = 40.0
    width = int(total_w * scale)
    height = int(total_h * scale)

    back = geometry.get("back_panel", {})
    spine = geometry.get("spine_panel", {})
    front = geometry.get("front_panel", {})

    d = Drawing(width, height)
    d.add(Rect(0, 0, width, height, fillColor=HexColor("#f4f7fd"), strokeColor=HexColor("#aab7cf")))

    def _panel_rect(panel: dict, fill: str):
        x = float(panel.get("x_in", 0.0)) * scale
        y = float(panel.get("y_in", 0.0)) * scale
        w = float(panel.get("width_in", 0.0)) * scale
        h = float(panel.get("height_in", 0.0)) * scale
        d.add(Rect(x, y, w, h, fillColor=HexColor(fill), strokeColor=HexColor("#6f7f9c")))

    _panel_rect(back, "#d8e4fa")
    _panel_rect(spine, "#3159a6")
    _panel_rect(front, "#1f3c88")

    title = str(cover_manifest.get("spine_text", "") or "Sudoku Companion")
    d.add(String(20, height - 30, "Cover Preview", fontName="Helvetica-Bold", fontSize=16, fillColor=HexColor("#173872")))
    d.add(String(20, height - 52, title[:48], fontName="Helvetica", fontSize=11, fillColor=HexColor("#173872")))

    renderPM.drawToFile(d, str(output_path), fmt="PNG")


def _render_interior_plan_preview_png(
    *,
    publication_manifest: dict,
    interior_plan: dict,
    output_path: Path,
) -> None:
    blocks = interior_plan.get("page_blocks", []) or []
    page_count = len(blocks)

    width = 1000
    height = max(300, 110 + (page_count * 18))

    d = Drawing(width, height)
    d.add(Rect(0, 0, width, height, fillColor=HexColor("#ffffff"), strokeColor=HexColor("#d8dde8")))

    title = str(publication_manifest.get("book_title", "") or "Publication")
    d.add(String(20, height - 28, "Interior Plan Preview", fontName="Helvetica-Bold", fontSize=16, fillColor=HexColor("#173872")))
    d.add(String(20, height - 48, title[:72], fontName="Helvetica", fontSize=11, fillColor=HexColor("#173872")))

    y = height - 78
    for block in blocks:
        page_index = block.get("page_index", "")
        page_type = str(block.get("page_type", ""))
        template_id = str(block.get("template_id", ""))
        printed = block.get("payload", {}).get("printed_page_number", "")
        show_number = block.get("show_page_number", False)

        line = f"{page_index:>3} | {page_type:<22} | {template_id:<24}"
        if show_number:
            line += f" | printed={printed}"

        d.add(String(24, y, line[:140], fontName="Courier", fontSize=9, fillColor=HexColor("#333333")))
        y -= 16
        if y < 18:
            break

    renderPM.drawToFile(d, str(output_path), fmt="PNG")