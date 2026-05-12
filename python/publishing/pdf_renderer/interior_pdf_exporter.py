from __future__ import annotations

from pathlib import Path
from typing import Literal

from reportlab.pdfgen.canvas import Canvas

from python.publishing.print_specs import get_print_format_spec

from .page_block_renderer import render_page_block
from .page_number_renderer import render_page_number
from .render_models import load_publication_render_context


InteriorBleedMode = Literal["no_bleed", "bleed"]


def _resolve_trim_size_in(context) -> tuple[float, float]:
    trim = context.publication_manifest.get("trim_size", {})
    page_width = trim.get("width_in")
    page_height = trim.get("height_in")

    if not page_width or not page_height:
        raise ValueError(
            f"Publication manifest in {context.publication_dir} is missing trim_size.width_in/height_in"
        )

    return float(page_width), float(page_height)


def _resolve_bleed_in(context) -> float:
    explicit = context.publication_manifest.get("interior_bleed_in")
    if explicit is not None:
        return max(0.0, float(explicit))

    format_id = str(context.publication_manifest.get("format_id") or "").strip()
    if not format_id:
        return 0.0

    try:
        return max(0.0, float(get_print_format_spec(format_id).bleed_in))
    except Exception:
        return 0.0


def _resolve_pagesize_pt(
    *,
    trim_width_in: float,
    trim_height_in: float,
    bleed_in: float,
    bleed_mode: InteriorBleedMode,
) -> tuple[float, float]:
    if bleed_mode == "bleed" and bleed_in > 0:
        return (
            (trim_width_in + bleed_in) * 72.0,
            (trim_height_in + (2.0 * bleed_in)) * 72.0,
        )

    return (trim_width_in * 72.0, trim_height_in * 72.0)


def export_book_interior_pdf(
    *,
    publication_dir: Path,
    output_pdf_path: Path,
    bleed_mode: InteriorBleedMode = "no_bleed",
) -> Path:
    if bleed_mode not in {"no_bleed", "bleed"}:
        raise ValueError(f"Unsupported bleed_mode={bleed_mode!r}. Use 'no_bleed' or 'bleed'.")

    context = load_publication_render_context(publication_dir)

    trim_width_in, trim_height_in = _resolve_trim_size_in(context)
    bleed_in = _resolve_bleed_in(context)

    pagesize = _resolve_pagesize_pt(
        trim_width_in=trim_width_in,
        trim_height_in=trim_height_in,
        bleed_in=bleed_in,
        bleed_mode=bleed_mode,
    )

    # Make the render mode visible to lower-level frame/layout code.
    context.publication_manifest["_render_bleed_mode"] = bleed_mode
    context.publication_manifest["_render_bleed_pt"] = bleed_in * 72.0 if bleed_mode == "bleed" else 0.0
    context.publication_manifest["_render_trim_size_pt"] = (
        trim_width_in * 72.0,
        trim_height_in * 72.0,
    )

    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = Canvas(str(output_pdf_path), pagesize=pagesize)
    canvas.setTitle(str(context.publication_manifest.get("book_title") or context.render_model.book_manifest.title))

    for block in context.interior_plan.page_blocks:
        render_page_block(canvas, block=block, context=context)
        render_page_number(canvas, block=block, context=context)
        canvas.showPage()

    canvas.save()
    return output_pdf_path