from __future__ import annotations

from typing import Any, Dict

from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen.canvas import Canvas


def wrap_text_to_width_by_font(
    value: str,
    *,
    font_name: str,
    font_size: float,
    max_width: float,
) -> list[str]:
    words = str(value or "").split()
    if not words:
        return []

    lines: list[str] = []
    current = words[0]

    for word in words[1:]:
        candidate = f"{current} {word}"
        if pdfmetrics.stringWidth(candidate, font_name, font_size) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word

    lines.append(current)
    return lines


def plan_smart_headline(
    *,
    text: str,
    font_name: str,
    preferred_font_size: float,
    min_font_size: float,
    max_width: float,
    max_lines: int = 2,
    shrink_step: float = 0.5,
    leading_multiplier: float = 1.10,
) -> Dict[str, Any]:
    cleaned = str(text or "").strip()
    if not cleaned:
        return {
            "lines": [],
            "font_size": float(preferred_font_size),
            "line_count": 0,
            "leading": float(preferred_font_size) * float(leading_multiplier),
            "height": 0.0,
        }

    font_size = float(preferred_font_size)
    min_size = float(min_font_size)
    best_lines = [cleaned]

    while font_size >= min_size - 1e-6:
        lines = wrap_text_to_width_by_font(
            cleaned,
            font_name=font_name,
            font_size=font_size,
            max_width=max_width,
        )
        best_lines = lines
        if len(lines) <= max_lines:
            leading = font_size * float(leading_multiplier)
            return {
                "lines": lines,
                "font_size": font_size,
                "line_count": len(lines),
                "leading": leading,
                "height": max(1, len(lines)) * leading,
            }
        font_size = round(font_size - float(shrink_step), 3)

    font_size = min_size
    leading = font_size * float(leading_multiplier)
    return {
        "lines": best_lines,
        "font_size": font_size,
        "line_count": len(best_lines),
        "leading": leading,
        "height": max(1, len(best_lines)) * leading,
    }


def draw_smart_headline(
    canvas: Canvas,
    *,
    text: str,
    font_name: str,
    preferred_font_size: float,
    min_font_size: float,
    max_width: float,
    x: float,
    first_baseline_y: float,
    align: str = "center",
    max_lines: int = 2,
    shrink_step: float = 0.5,
    leading_multiplier: float = 1.10,
    fill_color=colors.black,
) -> Dict[str, Any]:
    plan = plan_smart_headline(
        text=text,
        font_name=font_name,
        preferred_font_size=preferred_font_size,
        min_font_size=min_font_size,
        max_width=max_width,
        max_lines=max_lines,
        shrink_step=shrink_step,
        leading_multiplier=leading_multiplier,
    )

    canvas.setFillColor(fill_color)
    canvas.setFont(font_name, plan["font_size"])

    if not plan["lines"]:
        plan["first_baseline_y"] = first_baseline_y
        plan["bottom_y"] = first_baseline_y
        return plan

    normalized_align = str(align or "center").strip().lower()

    for idx, line in enumerate(plan["lines"]):
        yy = first_baseline_y - (idx * plan["leading"])

        if normalized_align == "left":
            canvas.drawString(x, yy, line)
        elif normalized_align == "right":
            canvas.drawRightString(x, yy, line)
        else:
            canvas.drawCentredString(x, yy, line)

    plan["first_baseline_y"] = first_baseline_y
    plan["bottom_y"] = first_baseline_y - ((len(plan["lines"]) - 1) * plan["leading"])
    return plan