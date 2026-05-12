from __future__ import annotations

from typing import Any, Dict

from python.publishing.schemas.models import CoverSpec, PrintFormatSpec


def build_cover_geometry(
    *,
    format_spec: PrintFormatSpec,
    cover_spec: CoverSpec,
) -> Dict[str, Any]:
    trim_w = float(format_spec.trim_width_in)
    trim_h = float(format_spec.trim_height_in)
    bleed = float(format_spec.bleed_in)
    safe = float(format_spec.safe_margin_in)
    spine = float(cover_spec.spine_width_in)

    total_width = (trim_w * 2.0) + spine + (bleed * 2.0)
    total_height = trim_h + (bleed * 2.0)

    back_x = bleed
    spine_x = bleed + trim_w
    front_x = bleed + trim_w + spine

    return {
        "units": "in",
        "total_width_in": round(total_width, 6),
        "total_height_in": round(total_height, 6),
        "trim_width_in": trim_w,
        "trim_height_in": trim_h,
        "bleed_in": bleed,
        "safe_margin_in": safe,
        "spine_width_in": spine,
        "back_panel": {
            "x_in": round(back_x, 6),
            "y_in": round(bleed, 6),
            "width_in": trim_w,
            "height_in": trim_h,
            "safe_box": {
                "x_in": round(back_x + safe, 6),
                "y_in": round(bleed + safe, 6),
                "width_in": round(max(0.0, trim_w - (safe * 2.0)), 6),
                "height_in": round(max(0.0, trim_h - (safe * 2.0)), 6),
            },
        },
        "spine_panel": {
            "x_in": round(spine_x, 6),
            "y_in": round(bleed, 6),
            "width_in": spine,
            "height_in": trim_h,
            "safe_box": {
                "x_in": round(spine_x + min(safe * 0.5, spine / 6.0 if spine > 0 else 0.0), 6),
                "y_in": round(bleed + safe, 6),
                "width_in": round(max(0.0, spine - (min(safe * 0.5, spine / 6.0 if spine > 0 else 0.0) * 2.0)), 6),
                "height_in": round(max(0.0, trim_h - (safe * 2.0)), 6),
            },
        },
        "front_panel": {
            "x_in": round(front_x, 6),
            "y_in": round(bleed, 6),
            "width_in": trim_w,
            "height_in": trim_h,
            "safe_box": {
                "x_in": round(front_x + safe, 6),
                "y_in": round(bleed + safe, 6),
                "width_in": round(max(0.0, trim_w - (safe * 2.0)), 6),
                "height_in": round(max(0.0, trim_h - (safe * 2.0)), 6),
            },
        },
        "barcode_box": {
            "x_in": round(back_x + trim_w - 2.25 - safe, 6),
            "y_in": round(bleed + safe, 6),
            "width_in": 2.0,
            "height_in": 1.2,
        },
    }