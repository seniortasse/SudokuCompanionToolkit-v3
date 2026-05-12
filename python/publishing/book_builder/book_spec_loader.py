from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


@dataclass
class BookSectionSpec:
    section_code: str
    title: str
    subtitle: str
    puzzle_count: int
    criteria: Dict[str, Any] = field(default_factory=dict)
    difficulty_label_hint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "section_code": self.section_code,
            "title": self.title,
            "subtitle": self.subtitle,
            "puzzle_count": self.puzzle_count,
            "criteria": dict(self.criteria),
            "difficulty_label_hint": self.difficulty_label_hint,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BookSectionSpec":
        return cls(
            section_code=str(data["section_code"]),
            title=str(data["title"]),
            subtitle=str(data.get("subtitle", "")),
            puzzle_count=int(data["puzzle_count"]),
            criteria=dict(data.get("criteria", {})),
            difficulty_label_hint=data.get("difficulty_label_hint"),
        )


@dataclass
class BookSpec:
    book_id: str
    library_id: str
    aisle_id: str
    title: str
    subtitle: str
    series_name: str
    volume_number: Optional[int]
    isbn: Optional[str]
    description: str
    target_audience: str
    trim_size: str
    puzzles_per_page: int
    page_layout_profile: str
    solution_section_policy: str
    cover_theme: str
    layout_type: str
    grid_size: int
    search_tags: List[str] = field(default_factory=list)
    publication_status: str = "draft"
    global_filters: Dict[str, Any] = field(default_factory=dict)
    ordering_policy: Dict[str, Any] = field(default_factory=dict)
    reuse_policy: str = "book_exclusive"
    sections: List[BookSectionSpec] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "book_id": self.book_id,
            "library_id": self.library_id,
            "aisle_id": self.aisle_id,
            "title": self.title,
            "subtitle": self.subtitle,
            "series_name": self.series_name,
            "volume_number": self.volume_number,
            "isbn": self.isbn,
            "description": self.description,
            "target_audience": self.target_audience,
            "trim_size": self.trim_size,
            "puzzles_per_page": self.puzzles_per_page,
            "page_layout_profile": self.page_layout_profile,
            "solution_section_policy": self.solution_section_policy,
            "cover_theme": self.cover_theme,
            "layout_type": self.layout_type,
            "grid_size": self.grid_size,
            "search_tags": list(self.search_tags),
            "publication_status": self.publication_status,
            "global_filters": dict(self.global_filters),
            "ordering_policy": dict(self.ordering_policy),
            "reuse_policy": self.reuse_policy,
            "sections": [section.to_dict() for section in self.sections],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BookSpec":
        return cls(
            book_id=str(data["book_id"]),
            library_id=str(data["library_id"]),
            aisle_id=str(data["aisle_id"]),
            title=str(data["title"]),
            subtitle=str(data.get("subtitle", "")),
            series_name=str(data.get("series_name", "")),
            volume_number=data.get("volume_number"),
            isbn=data.get("isbn"),
            description=str(data.get("description", "")),
            target_audience=str(data.get("target_audience", "general")),
            trim_size=str(data["trim_size"]),
            puzzles_per_page=int(data["puzzles_per_page"]),
            page_layout_profile=str(data["page_layout_profile"]),
            solution_section_policy=str(data.get("solution_section_policy", "appendix")),
            cover_theme=str(data.get("cover_theme", "")),
            layout_type=str(data["layout_type"]),
            grid_size=int(data["grid_size"]),
            search_tags=list(data.get("search_tags", [])),
            publication_status=str(data.get("publication_status", "draft")),
            global_filters=dict(data.get("global_filters", {})),
            ordering_policy=dict(data.get("ordering_policy", {})),
            reuse_policy=str(data.get("reuse_policy", "book_exclusive")),
            sections=[BookSectionSpec.from_dict(item) for item in data.get("sections", [])],
        )


def load_book_spec(path: Path) -> BookSpec:
    if not path.exists():
        raise FileNotFoundError(f"Book spec not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    return BookSpec.from_dict(data)