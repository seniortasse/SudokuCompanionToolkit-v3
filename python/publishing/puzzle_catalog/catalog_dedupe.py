from __future__ import annotations

from typing import Any, Dict, Optional

from python.publishing.puzzle_catalog.catalog_index import find_record_id_by_solution_signature


def find_duplicate_record_id(
    *,
    index: Dict[str, Any],
    pending_signature_to_record_id: Dict[str, str],
    solution_signature: str,
) -> Optional[str]:
    existing = find_record_id_by_solution_signature(index, solution_signature)
    if existing is not None:
        return existing
    return pending_signature_to_record_id.get(solution_signature)