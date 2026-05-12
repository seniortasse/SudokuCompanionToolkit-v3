from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


_ALLOWED_FONT_FAMILIES = {"helvetica", "times", "courier", "arial"}

_DIGIT_PRESET_TO_SCALE = {
    "small": 0.40,
    "medium": 0.48,
    "large": 0.56,
    "very_large": 0.62,
}


@dataclass(frozen=True)
class FontPack:
    family: str
    regular: str
    bold: str
    italic: str
    bold_italic: str


def _register_font_if_needed(font_name: str, font_path: Path) -> bool:
    if font_name in pdfmetrics.getRegisteredFontNames():
        return True
    if not font_path.exists():
        return False
    pdfmetrics.registerFont(TTFont(font_name, str(font_path)))
    return True


def _try_register_arial_family() -> bool:
    windows_fonts = Path("C:/Windows/Fonts")

    regular_ok = _register_font_if_needed("Arial", windows_fonts / "arial.ttf")
    bold_ok = _register_font_if_needed("Arial-Bold", windows_fonts / "arialbd.ttf")
    italic_ok = _register_font_if_needed("Arial-Italic", windows_fonts / "ariali.ttf")
    bold_italic_ok = _register_font_if_needed("Arial-BoldItalic", windows_fonts / "arialbi.ttf")

    return regular_ok and bold_ok and italic_ok and bold_italic_ok


def resolve_font_pack(font_family: str | None) -> FontPack:
    key = str(font_family or "helvetica").strip().lower()
    if key not in _ALLOWED_FONT_FAMILIES:
        key = "helvetica"

    if key == "arial":
        if _try_register_arial_family():
            return FontPack(
                family="arial",
                regular="Arial",
                bold="Arial-Bold",
                italic="Arial-Italic",
                bold_italic="Arial-BoldItalic",
            )
        key = "helvetica"

    if key == "times":
        return FontPack(
            family="times",
            regular="Times-Roman",
            bold="Times-Bold",
            italic="Times-Italic",
            bold_italic="Times-BoldItalic",
        )

    if key == "courier":
        return FontPack(
            family="courier",
            regular="Courier",
            bold="Courier-Bold",
            italic="Courier-Oblique",
            bold_italic="Courier-BoldOblique",
        )

    return FontPack(
        family="helvetica",
        regular="Helvetica",
        bold="Helvetica-Bold",
        italic="Helvetica-Oblique",
        bold_italic="Helvetica-BoldOblique",
    )


def resolve_digit_scale(
    *,
    explicit_scale: float | None,
    preset: str | None,
    fallback: float,
) -> float:
    if explicit_scale is not None:
        return float(explicit_scale)

    key = str(preset or "").strip().lower()
    if key in _DIGIT_PRESET_TO_SCALE:
        return _DIGIT_PRESET_TO_SCALE[key]

    return float(fallback)


def is_supported_font_family(value: str | None) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in _ALLOWED_FONT_FAMILIES


def is_supported_digit_preset(value: str | None) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in _DIGIT_PRESET_TO_SCALE