from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from python.publishing.schemas.models import (
    BookManifest,
    CoverSpec,
    InteriorPlan,
    PublicationPackage,
    PuzzleRecord,
    SectionManifest,
)
from python.publishing.schemas.publication_io import (
    load_cover_spec,
    load_interior_plan,
    load_publication_package,
)


@dataclass
class RenderSection:
    section_manifest: SectionManifest
    puzzles: List[PuzzleRecord] = field(default_factory=list)


@dataclass
class BuiltBookRenderModel:
    book_manifest: BookManifest
    sections: List[RenderSection] = field(default_factory=list)


@dataclass
class PublicationRenderContext:
    publication_dir: Path
    publication_package: PublicationPackage
    publication_manifest: Dict[str, Any]
    render_model: BuiltBookRenderModel
    interior_plan: InteriorPlan
    cover_spec: Optional[CoverSpec] = None


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_built_book_render_model(book_dir: Path) -> BuiltBookRenderModel:
    if not book_dir.exists():
        raise FileNotFoundError(f"Built book directory not found: {book_dir}")

    book_manifest_path = book_dir / "book_manifest.json"
    sections_dir = book_dir / "sections"
    puzzles_dir = book_dir / "puzzles"

    if not book_manifest_path.exists():
        raise FileNotFoundError(f"Missing book_manifest.json in {book_dir}")
    if not sections_dir.exists():
        raise FileNotFoundError(f"Missing sections directory in {book_dir}")
    if not puzzles_dir.exists():
        raise FileNotFoundError(f"Missing puzzles directory in {book_dir}")

    book_manifest = BookManifest.from_dict(_load_json(book_manifest_path))

    puzzle_by_id: Dict[str, PuzzleRecord] = {}
    for puzzle_path in sorted(puzzles_dir.glob("*.json")):
        puzzle = PuzzleRecord.from_dict(_load_json(puzzle_path))
        puzzle_by_id[puzzle.puzzle_uid] = puzzle

    sections: List[RenderSection] = []
    for section_path in sorted(sections_dir.glob("*.json")):
        section_manifest = SectionManifest.from_dict(_load_json(section_path))
        puzzles = [
            puzzle_by_id[puzzle_id]
            for puzzle_id in section_manifest.puzzle_ids
            if puzzle_id in puzzle_by_id
        ]
        sections.append(RenderSection(section_manifest=section_manifest, puzzles=puzzles))

    return BuiltBookRenderModel(
        book_manifest=book_manifest,
        sections=sections,
    )


def load_publication_render_context(publication_dir: Path) -> PublicationRenderContext:
    if not publication_dir.exists():
        raise FileNotFoundError(f"Publication directory not found: {publication_dir}")

    publication_package_path = publication_dir / "publication_package.json"
    publication_manifest_path = publication_dir / "publication_manifest.json"

    if not publication_package_path.exists():
        raise FileNotFoundError(f"Missing publication_package.json in {publication_dir}")
    if not publication_manifest_path.exists():
        raise FileNotFoundError(f"Missing publication_manifest.json in {publication_dir}")

    publication_package = load_publication_package(publication_package_path)
    publication_manifest = _load_json(publication_manifest_path)
    interior_plan = load_interior_plan(Path(publication_package.interior_plan_path))
    render_model = load_built_book_render_model(Path(publication_package.book_dir))

    cover_spec = None
    if publication_package.cover_spec_path:
        cover_spec_path = Path(publication_package.cover_spec_path)
        if cover_spec_path.exists():
            cover_spec = load_cover_spec(cover_spec_path)

    return PublicationRenderContext(
        publication_dir=publication_dir,
        publication_package=publication_package,
        publication_manifest=publication_manifest,
        render_model=render_model,
        interior_plan=interior_plan,
        cover_spec=cover_spec,
    )