from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from python.publishing.schemas.models import PuzzleRecord


def _is_puzzle_record_file(path: Path) -> bool:
    name = path.name
    if path.suffix.lower() != ".json":
        return False
    if name.startswith("_"):
        return False
    if not name.startswith("REC-"):
        return False
    return True


def _record_path(directory: Path, record_id: str) -> Path:
    return Path(directory) / f"{str(record_id).strip()}.json"


def load_puzzle_record(path: Path) -> PuzzleRecord:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))

    if "record_id" not in data or "library_id" not in data:
        raise ValueError(f"Not a canonical puzzle record file: {path}")

    return PuzzleRecord.from_dict(data)


def load_puzzle_records_from_dir(directory: Path) -> List[PuzzleRecord]:
    directory = Path(directory)
    if not directory.exists():
        return []

    records: List[PuzzleRecord] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        if not _is_puzzle_record_file(path):
            continue
        records.append(load_puzzle_record(path))
    return records


def save_puzzle_record(record: PuzzleRecord, directory: Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    path = _record_path(directory, record.record_id)
    path.write_text(
        json.dumps(record.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def save_puzzle_records_batch(records: Iterable[PuzzleRecord], directory: Path) -> List[Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    written_paths: List[Path] = []
    for record in records:
        written_paths.append(save_puzzle_record(record, directory))
    return written_paths