from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class PublicationVariantRequest:
    base_spec_path: Path
    locale: Optional[str] = None
    language: Optional[str] = None
    locale_pack_path: Optional[Path] = None
    layout_preset_id: Optional[str] = None
    puzzles_per_page: Optional[int] = None
    rows: Optional[int] = None
    cols: Optional[int] = None
    font_family: Optional[str] = None
    publication_id: Optional[str] = None