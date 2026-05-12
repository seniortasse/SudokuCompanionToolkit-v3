from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from python.publishing.schemas.models import PuzzleRecord

INDEX_FILENAME = "_catalog_index.json"


def _default_index() -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "next_record_ordinal": 1,
        "records_by_id": {},
        "solution_signature_to_record_id": {},
    }


def _index_path(output_dir: Path) -> Path:
    return output_dir / INDEX_FILENAME


def load_catalog_index(output_dir: Path) -> Dict[str, Any]:
    path = _index_path(output_dir)
    if not path.exists():
        return _default_index()

    data = json.loads(path.read_text(encoding="utf-8"))
    index = _default_index()
    index.update(data)

    index["records_by_id"] = dict(index.get("records_by_id", {}))
    index["solution_signature_to_record_id"] = dict(index.get("solution_signature_to_record_id", {}))
    index["next_record_ordinal"] = int(index.get("next_record_ordinal", 1))
    return index


def save_catalog_index(index: Dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = _index_path(output_dir)
    path.write_text(
        json.dumps(index, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def reserve_next_record_ordinal(index: Dict[str, Any]) -> int:
    ordinal = int(index.get("next_record_ordinal", 1))
    index["next_record_ordinal"] = ordinal + 1
    return ordinal


def find_record_id_by_solution_signature(
    index: Dict[str, Any],
    solution_signature: str,
) -> Optional[str]:
    return index.get("solution_signature_to_record_id", {}).get(solution_signature)


def register_record(
    index: Dict[str, Any],
    record: PuzzleRecord,
    *,
    relative_path: str,
) -> None:
    records_by_id = index.setdefault("records_by_id", {})
    signature_map = index.setdefault("solution_signature_to_record_id", {})

    if record.record_id in records_by_id:
        raise ValueError(f"record_id already exists in catalog index: {record.record_id}")

    existing = signature_map.get(record.solution_signature)
    if existing is not None:
        raise ValueError(
            f"solution_signature already exists in catalog index: "
            f"{record.solution_signature} -> {existing}"
        )

    records_by_id[record.record_id] = {
        "relative_path": relative_path,
        "candidate_status": record.candidate_status,
        "solution_signature": record.solution_signature,
        "weight": record.weight,
        "difficulty_label": record.difficulty_label,
        "puzzle_difficulty": record.puzzle_difficulty,
        "difficulty_version": record.difficulty_version,
        "pattern_id": record.pattern_id,
        "pattern_name": record.pattern_name,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }
    signature_map[record.solution_signature] = record.record_id


def update_record_status(
    index: Dict[str, Any],
    *,
    record_id: str,
    candidate_status: str,
) -> None:
    records_by_id = index.setdefault("records_by_id", {})
    entry = records_by_id.get(record_id)
    if entry is None:
        return
    entry["candidate_status"] = candidate_status


def remove_record_from_index(
    index: Dict[str, Any],
    *,
    record_id: str,
) -> bool:
    records_by_id = index.setdefault("records_by_id", {})
    signature_map = index.setdefault("solution_signature_to_record_id", {})

    entry = records_by_id.get(record_id)
    if entry is None:
        return False

    solution_signature = entry.get("solution_signature")
    del records_by_id[record_id]

    if solution_signature and signature_map.get(solution_signature) == record_id:
        del signature_map[solution_signature]

    return True