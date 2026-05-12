from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from python.publishing.schemas.models import BookManifest, PuzzleRecord, SectionManifest


def load_built_book_package(book_dir: Path) -> Tuple[BookManifest, List[SectionManifest], List[PuzzleRecord]]:
    manifest_path = book_dir / "book_manifest.json"
    sections_dir = book_dir / "sections"
    puzzles_dir = book_dir / "puzzles"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Book manifest not found: {manifest_path}")
    if not sections_dir.exists():
        raise FileNotFoundError(f"Sections directory not found: {sections_dir}")
    if not puzzles_dir.exists():
        raise FileNotFoundError(f"Puzzles directory not found: {puzzles_dir}")

    book_manifest = BookManifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))

    section_manifests: List[SectionManifest] = []
    for path in sorted(sections_dir.glob("*.json")):
        section_manifests.append(SectionManifest.from_dict(json.loads(path.read_text(encoding="utf-8"))))

    assigned_puzzles: List[PuzzleRecord] = []
    for path in sorted(puzzles_dir.glob("*.json")):
        assigned_puzzles.append(PuzzleRecord.from_dict(json.loads(path.read_text(encoding="utf-8"))))

    return book_manifest, section_manifests, assigned_puzzles


def save_built_book_package(
    *,
    book_dir: Path,
    book_manifest: BookManifest,
    section_manifests: List[SectionManifest],
    assigned_puzzles: List[PuzzleRecord],
) -> None:
    book_dir.mkdir(parents=True, exist_ok=True)
    sections_dir = book_dir / "sections"
    puzzles_dir = book_dir / "puzzles"
    sections_dir.mkdir(parents=True, exist_ok=True)
    puzzles_dir.mkdir(parents=True, exist_ok=True)

    (book_dir / "book_manifest.json").write_text(
        json.dumps(book_manifest.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for old in sections_dir.glob("*.json"):
        old.unlink()
    for old in puzzles_dir.glob("*.json"):
        old.unlink()

    for section in section_manifests:
        (sections_dir / f"{section.section_id}.json").write_text(
            json.dumps(section.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    for record in assigned_puzzles:
        file_stem = record.puzzle_uid or record.record_id
        (puzzles_dir / f"{file_stem}.json").write_text(
            json.dumps(record.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )