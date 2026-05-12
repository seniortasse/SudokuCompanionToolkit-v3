from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

from python.publishing.puzzle_catalog.catalog_index import (
    load_catalog_index,
    remove_record_from_index,
    save_catalog_index,
    update_record_status,
)
from python.publishing.puzzle_catalog.catalog_store import load_puzzle_record


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def archive_candidate_record(
    *,
    records_dir: Path,
    record_id: str,
) -> Tuple[bool, str]:
    record_path = records_dir / f"{record_id}.json"
    if not record_path.exists():
        return False, f"Record file not found: {record_path}"

    record = load_puzzle_record(record_path)
    record.candidate_status = "archived"
    record.updated_at = _utc_now_iso()

    record_path.write_text(
        json.dumps(record.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    index = load_catalog_index(records_dir)
    update_record_status(
        index,
        record_id=record_id,
        candidate_status="archived",
    )
    save_catalog_index(index, records_dir)

    return True, f"Archived record {record_id}"


def delete_candidate_record(
    *,
    records_dir: Path,
    record_id: str,
) -> Tuple[bool, str]:
    record_path = records_dir / f"{record_id}.json"
    if not record_path.exists():
        return False, f"Record file not found: {record_path}"

    index = load_catalog_index(records_dir)
    removed = remove_record_from_index(index, record_id=record_id)
    if not removed:
        return False, f"record_id {record_id} not found in catalog index"

    record_path.unlink()
    save_catalog_index(index, records_dir)

    return True, f"Deleted record {record_id}"