from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class PublicationManifest:
    publication_id: str
    book_id: str
    channel_id: str
    format_id: str
    trim_width_in: float
    trim_height_in: float
    paper_type: str
    include_cover: bool
    include_solutions: bool
    mirror_margins: bool
    page_numbering_policy: str
    blank_page_policy: str
    puzzle_page_template: str
    solution_page_template: str
    cover_template: str
    estimated_page_count: int

    # Defaults must come AFTER all required fields for Python 3.8 dataclasses.
    interior_bleed_mode: str = "both"
    interior_bleed_in: float = 0.0

    cover_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    layout_config: Dict[str, Any] = field(default_factory=dict)
    solution_section_config: Dict[str, Any] = field(default_factory=dict)
    variant_identity: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "publication_id": self.publication_id,
            "book_id": self.book_id,
            "channel_id": self.channel_id,
            "format_id": self.format_id,
            "trim_width_in": self.trim_width_in,
            "trim_height_in": self.trim_height_in,
            "trim_size": {
                "width_in": self.trim_width_in,
                "height_in": self.trim_height_in,
            },
            "paper_type": self.paper_type,
            "interior_bleed_mode": self.interior_bleed_mode,
            "interior_bleed_in": self.interior_bleed_in,
            "include_cover": self.include_cover,
            "include_solutions": self.include_solutions,
            "mirror_margins": bool(self.mirror_margins),
            "page_numbering_policy": self.page_numbering_policy,
            "blank_page_policy": self.blank_page_policy,
            "puzzle_page_template": self.puzzle_page_template,
            "solution_page_template": self.solution_page_template,
            "cover_template": self.cover_template,
            "estimated_page_count": self.estimated_page_count,
            "cover_id": self.cover_id,
            "metadata": dict(self.metadata or {}),
            "layout_config": dict(self.layout_config or {}),
            "solution_section_config": dict(self.solution_section_config or {}),
            "variant_identity": dict(self.variant_identity or {}),
        }