from __future__ import annotations

from typing import Dict, List

from python.publishing.book_builder.preset_templates import (
    DW_PRESET_RANGES,
    build_dw_book_spec,
    build_dw_sections,
)

DEFAULT_DW_AISLE_SHORT = "DW"
DEFAULT_DW_AISLE_ID_SHORT_FOR_MANIFEST = "DWEIGHT"


def list_dw_presets() -> List[str]:
    return sorted(DW_PRESET_RANGES.keys())


def build_dw_preset_book_spec(**kwargs):
    return build_dw_book_spec(**kwargs)