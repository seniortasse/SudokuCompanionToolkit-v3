from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_int_set(values: Optional[Sequence[int]]) -> set[int]:
    return {int(v) for v in (values or [])}


def _normalize_str_set(values: Optional[Sequence[str]]) -> set[str]:
    return {_normalize_text(v) for v in (values or []) if _normalize_text(v)}


def load_candidate_jsonl_lines(jsonl_path: Path) -> List[Dict[str, Any]]:
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Candidates JSONL file not found: {jsonl_path}")

    results: List[Dict[str, Any]] = []
    raw_lines = jsonl_path.read_text(encoding="utf-8").splitlines()

    for line_number, raw_line in enumerate(raw_lines, start=1):
        stripped = raw_line.strip()
        if not stripped:
            results.append(
                {
                    "_line_number": line_number,
                    "_raw_line": raw_line,
                    "_is_blank": True,
                    "_is_valid_json": False,
                    "_payload": None,
                }
            )
            continue

        try:
            payload = json.loads(stripped)
            results.append(
                {
                    "_line_number": line_number,
                    "_raw_line": raw_line,
                    "_is_blank": False,
                    "_is_valid_json": True,
                    "_payload": payload,
                }
            )
        except Exception:
            results.append(
                {
                    "_line_number": line_number,
                    "_raw_line": raw_line,
                    "_is_blank": False,
                    "_is_valid_json": False,
                    "_payload": None,
                }
            )

    return results


def select_candidate_jsonl_lines(
    entries: Sequence[Dict[str, Any]],
    *,
    line_numbers: Optional[Sequence[int]] = None,
    generation_seeds: Optional[Sequence[int]] = None,
    pattern_ids: Optional[Sequence[str]] = None,
    givens81_values: Optional[Sequence[str]] = None,
    solution81_values: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []

    normalized_line_numbers = _normalize_int_set(line_numbers)
    normalized_generation_seeds = _normalize_int_set(generation_seeds)
    normalized_pattern_ids = _normalize_str_set(pattern_ids)
    normalized_givens = _normalize_str_set(givens81_values)
    normalized_solutions = _normalize_str_set(solution81_values)

    any_selector = any([
        normalized_line_numbers,
        normalized_generation_seeds,
        normalized_pattern_ids,
        normalized_givens,
        normalized_solutions,
    ])

    for entry in entries:
        if not any_selector:
            continue

        line_number = int(entry["_line_number"])
        payload = entry.get("_payload") or {}

        match = False
        if normalized_line_numbers and line_number in normalized_line_numbers:
            match = True
        if normalized_generation_seeds and int(payload.get("generation_seed", -1)) in normalized_generation_seeds:
            match = True
        if normalized_pattern_ids and _normalize_text(payload.get("pattern_id")) in normalized_pattern_ids:
            match = True
        if normalized_givens and _normalize_text(payload.get("givens81")) in normalized_givens:
            match = True
        if normalized_solutions and _normalize_text(payload.get("solution81")) in normalized_solutions:
            match = True

        if match:
            selected.append(entry)

    return selected


def summarize_selected_candidate_lines(
    selected_entries: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    valid_json_count = 0
    invalid_json_count = 0
    blank_count = 0
    line_numbers: List[int] = []
    generation_seeds: List[int] = []
    pattern_ids: List[str] = []

    for entry in selected_entries:
        line_numbers.append(int(entry["_line_number"]))

        if entry.get("_is_blank"):
            blank_count += 1
            continue

        if not entry.get("_is_valid_json"):
            invalid_json_count += 1
            continue

        valid_json_count += 1
        payload = entry.get("_payload") or {}
        if "generation_seed" in payload:
            try:
                generation_seeds.append(int(payload["generation_seed"]))
            except Exception:
                pass
        pattern_id = _normalize_text(payload.get("pattern_id"))
        if pattern_id:
            pattern_ids.append(pattern_id)

    return {
        "selected_count": len(selected_entries),
        "valid_json_count": valid_json_count,
        "invalid_json_count": invalid_json_count,
        "blank_count": blank_count,
        "line_numbers": sorted(line_numbers),
        "generation_seeds": sorted(set(generation_seeds)),
        "pattern_ids": sorted(set(pattern_ids)),
    }


def archive_candidate_jsonl_lines(
    *,
    selected_entries: Sequence[Dict[str, Any]],
    archive_path: Path,
) -> Optional[Path]:
    if not selected_entries:
        return None

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with archive_path.open("a", encoding="utf-8", newline="\n") as handle:
        for entry in selected_entries:
            handle.write(str(entry["_raw_line"]).rstrip("\n"))
            handle.write("\n")
    return archive_path


def rewrite_candidate_jsonl_without_selected(
    *,
    jsonl_path: Path,
    selected_entries: Sequence[Dict[str, Any]],
) -> Tuple[int, int]:
    selected_line_numbers = {int(entry["_line_number"]) for entry in selected_entries}
    raw_lines = jsonl_path.read_text(encoding="utf-8").splitlines()

    kept_lines: List[str] = []
    removed_count = 0

    for line_number, raw_line in enumerate(raw_lines, start=1):
        if line_number in selected_line_numbers:
            removed_count += 1
            continue
        kept_lines.append(raw_line)

    final_text = "\n".join(kept_lines)
    if kept_lines:
        final_text += "\n"

    jsonl_path.write_text(final_text, encoding="utf-8")
    return removed_count, len(kept_lines)