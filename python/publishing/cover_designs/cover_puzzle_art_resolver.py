from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SAMPLE_MAIN_GIVENS81 = (
    "200000690"
    "360004700"
    "078013000"
    "002530000"
    "000907000"
    "000021400"
    "000470180"
    "007100049"
    "024000007"
)

SAMPLE_LEFT_GIVENS81 = (
    "901000000"
    "600000000"
    "300700000"
    "700000000"
    "800900000"
    "400000000"
    "500060000"
    "000000000"
    "174000000"
)

SAMPLE_RIGHT_GIVENS81 = (
    "000000000"
    "000000000"
    "000000038"
    "000000030"
    "000000050"
    "000000064"
    "000000020"
    "000000000"
    "000000700"
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _valid_givens81(value: Any) -> str | None:
    if not isinstance(value, str):
        return None

    cleaned = value.strip()
    if len(cleaned) != 81:
        return None

    allowed = set("0123456789.-")
    if not set(cleaned).issubset(allowed):
        return None

    return cleaned.replace(".", "0").replace("-", "0")


def _find_givens81(payload: Any) -> str | None:
    if isinstance(payload, dict):
        for key in ("givens81", "given81", "puzzle81", "grid81"):
            found = _valid_givens81(payload.get(key))
            if found:
                return found

        for value in payload.values():
            found = _find_givens81(value)
            if found:
                return found

    if isinstance(payload, list):
        for item in payload:
            found = _find_givens81(item)
            if found:
                return found

    return None


def _find_first_string(payload: Any, keys: tuple[str, ...]) -> str | None:
    if isinstance(payload, dict):
        for key in keys:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        for value in payload.values():
            found = _find_first_string(value, keys)
            if found:
                return found

    if isinstance(payload, list):
        for item in payload:
            found = _find_first_string(item, keys)
            if found:
                return found

    return None


def _find_first_int(payload: Any, keys: tuple[str, ...]) -> int | None:
    if isinstance(payload, dict):
        for key in keys:
            value = payload.get(key)
            if isinstance(value, int):
                return value
            if isinstance(value, str) and value.strip().isdigit():
                return int(value.strip())

        for value in payload.values():
            found = _find_first_int(value, keys)
            if found is not None:
                return found

    if isinstance(payload, list):
        for item in payload:
            found = _find_first_int(item, keys)
            if found is not None:
                return found

    return None


def _record_id_from_payload(payload: dict[str, Any], path: Path | None = None) -> str:
    found = _find_first_string(payload, ("record_id", "puzzle_uid", "puzzle_id"))
    if found:
        return found

    if path is not None:
        return path.stem

    return ""


def _pattern_id_from_payload(payload: dict[str, Any]) -> str | None:
    return _find_first_string(
        payload,
        (
            "pattern_id",
            "patternId",
            "pattern_uid",
            "patternUid",
            "source_pattern_id",
        ),
    )


def _record_matches_pattern(payload: Any, pattern_id: str) -> bool:
    if not pattern_id:
        return False

    if isinstance(payload, dict):
        for key in (
            "pattern_id",
            "patternId",
            "pattern_uid",
            "patternUid",
            "source_pattern_id",
        ):
            if str(payload.get(key) or "") == pattern_id:
                return True

        for key in ("patterns", "pattern_ids", "patternIds", "pattern_uids"):
            value = payload.get(key)
            if isinstance(value, list) and pattern_id in {str(x) for x in value}:
                return True

        for value in payload.values():
            if _record_matches_pattern(value, pattern_id):
                return True

    if isinstance(payload, list):
        return any(_record_matches_pattern(item, pattern_id) for item in payload)

    return False


def _extract_record_ids(payload: Any) -> list[str]:
    ids: list[str] = []

    if isinstance(payload, dict):
        for key in ("record_id", "puzzle_uid", "puzzle_id"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                ids.append(value.strip())

        for key in ("puzzle_ids", "record_ids"):
            value = payload.get(key)
            if isinstance(value, list):
                ids.extend(str(x).strip() for x in value if str(x).strip())

        for value in payload.values():
            ids.extend(_extract_record_ids(value))

    elif isinstance(payload, list):
        for item in payload:
            ids.extend(_extract_record_ids(item))

    seen: set[str] = set()
    unique: list[str] = []
    for value in ids:
        if value not in seen:
            unique.append(value)
            seen.add(value)

    return unique


def _load_book_record_ids(book_dir: Path) -> list[str]:
    candidates: list[str] = []

    for filename in ("book_manifest.json", "publication_manifest.json"):
        path = book_dir / filename
        if path.exists():
            candidates.extend(_extract_record_ids(_read_json(path)))

    for path in sorted(book_dir.rglob("*.json")):
        if path.name in {"book_manifest.json", "publication_manifest.json"}:
            continue
        if path.name.startswith("_"):
            continue
        try:
            candidates.extend(_extract_record_ids(_read_json(path)))
        except Exception:
            continue

    seen: set[str] = set()
    unique: list[str] = []
    for value in candidates:
        if value not in seen:
            unique.append(value)
            seen.add(value)

    return unique


def _default_puzzle_records_dir(book_dir: Path) -> Path:
    return book_dir.parent.parent / "puzzle_records"


def _load_record_payload(record_id: str, puzzle_records_dir: Path) -> dict[str, Any] | None:
    direct = puzzle_records_dir / f"{record_id}.json"
    if direct.exists():
        return _read_json(direct)

    index_path = puzzle_records_dir / "_catalog_index.json"
    if index_path.exists():
        try:
            index = _read_json(index_path)
            items = index.get("records") if isinstance(index, dict) else None
            if isinstance(items, list):
                for item in items:
                    if str(item.get("record_id") or item.get("puzzle_uid") or "") == record_id:
                        rel = item.get("path") or item.get("file") or item.get("filename")
                        if rel:
                            candidate = puzzle_records_dir / str(rel)
                            if candidate.exists():
                                return _read_json(candidate)
        except Exception:
            pass

    return None


def _book_order_key(payload: dict[str, Any]) -> tuple[int, int, int, str]:
    position = _find_first_int(
        payload,
        (
            "position_in_book",
            "book_position",
            "book_index",
            "global_index",
            "sequence_index",
        ),
    )
    page = _find_first_int(payload, ("page", "page_number", "physical_page"))
    page_slot = _find_first_int(
        payload,
        (
            "puzzle_on_page",
            "slot_on_page",
            "position_on_page",
            "page_slot",
            "puzzle_index_on_page",
        ),
    )

    if position is None:
        position = 10**9
    if page is None:
        page = 10**9
    if page_slot is None:
        page_slot = 10**9

    return (
        position,
        page,
        page_slot,
        _record_id_from_payload(payload),
    )


def _load_book_records_from_book_dir(book_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    puzzles_dir = book_dir / "puzzles"

    if not puzzles_dir.exists():
        return records

    for path in sorted(puzzles_dir.glob("*.json")):
        if path.name.startswith("_"):
            continue
        try:
            payload = _read_json(path)
        except Exception:
            continue

        if not isinstance(payload, dict):
            continue

        payload.setdefault("record_id", path.stem)
        records.append(payload)

    records.sort(key=_book_order_key)
    return records


def _load_book_records_from_ids(book_dir: Path, puzzle_records_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for record_id in _load_book_record_ids(book_dir):
        payload = _load_record_payload(record_id, puzzle_records_dir)
        if payload is not None:
            payload.setdefault("record_id", record_id)
            records.append(payload)

    records.sort(key=_book_order_key)
    return records


def _load_book_records(book_dir: Path, puzzle_records_dir: Path) -> list[dict[str, Any]]:
    direct_records = _load_book_records_from_book_dir(book_dir)
    if direct_records:
        return direct_records

    return _load_book_records_from_ids(book_dir, puzzle_records_dir)


def _selection_metadata(
    *,
    role: str,
    source: dict[str, Any],
    givens81: str,
    payload: dict[str, Any] | None,
    fallback_used: bool,
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "role": role,
        "requested_source": source,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "givens81": givens81,
    }

    if payload:
        meta.update(
            {
                "record_id": _record_id_from_payload(payload),
                "pattern_id": _pattern_id_from_payload(payload),
                "position_in_book": _find_first_int(
                    payload,
                    (
                        "position_in_book",
                        "book_position",
                        "book_index",
                        "global_index",
                        "sequence_index",
                    ),
                ),
                "page": _find_first_int(payload, ("page", "page_number", "physical_page")),
                "puzzle_on_page": _find_first_int(
                    payload,
                    (
                        "puzzle_on_page",
                        "slot_on_page",
                        "position_on_page",
                        "page_slot",
                        "puzzle_index_on_page",
                    ),
                ),
                "local_puzzle_code": _find_first_string(
                    payload,
                    ("local_puzzle_code", "puzzle_code", "display_code"),
                ),
            }
        )

    return meta


def _fallback_for_role(role: str) -> str:
    if role == "left":
        return SAMPLE_LEFT_GIVENS81
    if role == "right":
        return SAMPLE_RIGHT_GIVENS81
    return SAMPLE_MAIN_GIVENS81


def _choose_decorative_index(role: str, count: int) -> int:
    if count <= 0:
        return 0
    if role == "left":
        return max(0, min(count - 1, count // 4))
    if role == "right":
        return max(0, min(count - 1, (count * 3) // 4))
    return 0


def _source_record_id(source: dict[str, Any]) -> str:
    return str(
        source.get("record_id")
        or source.get("puzzle_uid")
        or source.get("puzzle_id")
        or source.get("id")
        or ""
    ).strip()


def _source_book_index(source: dict[str, Any]) -> tuple[int, Any]:
    raw_index = source.get("index")
    if raw_index is None:
        raw_index = source.get("book_index")
    if raw_index is None:
        raw_index = source.get("position_in_book")
    if raw_index is None:
        raw_index = source.get("position")
    if raw_index is None:
        raw_index = source.get("sequence_index")

    try:
        return int(raw_index), raw_index
    except Exception:
        return -1, raw_index


def _resolve_source(
    source: dict[str, Any],
    *,
    role: str,
    book_records: list[dict[str, Any]],
    puzzle_records_dir: Path,
    fallback: str,
) -> dict[str, Any]:
    mode = str(source.get("mode") or "sample_fallback")

    if mode == "manual_givens81":
        value = _valid_givens81(source.get("givens81"))
        if value:
            return _selection_metadata(
                role=role,
                source=source,
                givens81=value,
                payload=None,
                fallback_used=False,
            )
        return _selection_metadata(
            role=role,
            source=source,
            givens81=fallback,
            payload=None,
            fallback_used=True,
            fallback_reason="manual_givens81 missing or invalid",
        )

    if mode in {"specific_record_id", "record_id", "puzzle_id", "specific_puzzle_id"}:
        record_id = _source_record_id(source)
        payload = None

        for candidate in book_records:
            candidate_ids = {
                str(_record_id_from_payload(candidate) or ""),
                str(_find_first_string(candidate, ("puzzle_uid", "puzzle_id")) or ""),
            }
            if record_id and record_id in candidate_ids:
                payload = candidate
                break

        if payload is None and record_id:
            payload = _load_record_payload(record_id, puzzle_records_dir)

        givens = _find_givens81(payload) if payload else None
        if givens:
            return _selection_metadata(
                role=role,
                source=source,
                givens81=givens,
                payload=payload,
                fallback_used=False,
            )

        return _selection_metadata(
            role=role,
            source=source,
            givens81=fallback,
            payload=None,
            fallback_used=True,
            fallback_reason=f"record_id/puzzle_id not found or invalid: {record_id}",
        )

    if mode == "first_book_puzzle":
        for payload in book_records:
            givens = _find_givens81(payload)
            if givens:
                return _selection_metadata(
                    role=role,
                    source=source,
                    givens81=givens,
                    payload=payload,
                    fallback_used=False,
                )

        return _selection_metadata(
            role=role,
            source=source,
            givens81=fallback,
            payload=None,
            fallback_used=True,
            fallback_reason="book has no valid puzzle records",
        )

    if mode in {"book_index", "index", "position"}:
        index, raw_index = _source_book_index(source)

        if 0 <= index < len(book_records):
            payload = book_records[index]
            givens = _find_givens81(payload)
            if givens:
                return _selection_metadata(
                    role=role,
                    source=source,
                    givens81=givens,
                    payload=payload,
                    fallback_used=False,
                )

        return _selection_metadata(
            role=role,
            source=source,
            givens81=fallback,
            payload=None,
            fallback_used=True,
            fallback_reason=f"book_index out of range or invalid: {raw_index}",
        )

    if mode == "pattern_id":
        pattern_id = str(source.get("pattern_id") or source.get("patternId") or "").strip()
        for payload in book_records:
            if _record_matches_pattern(payload, pattern_id):
                givens = _find_givens81(payload)
                if givens:
                    return _selection_metadata(
                        role=role,
                        source=source,
                        givens81=givens,
                        payload=payload,
                        fallback_used=False,
                    )

        return _selection_metadata(
            role=role,
            source=source,
            givens81=fallback,
            payload=None,
            fallback_used=True,
            fallback_reason=f"pattern_id absent from book or invalid: {pattern_id}",
        )

    if mode == "decorative_from_book":
        if book_records:
            index = _choose_decorative_index(role, len(book_records))
            payload = book_records[index]
            givens = _find_givens81(payload)
            if givens:
                enriched_source = dict(source)
                enriched_source.setdefault("resolved_strategy", "book_quarter_index")
                enriched_source.setdefault("resolved_index", index)
                return _selection_metadata(
                    role=role,
                    source=enriched_source,
                    givens81=givens,
                    payload=payload,
                    fallback_used=False,
                )

        return _selection_metadata(
            role=role,
            source=source,
            givens81=fallback,
            payload=None,
            fallback_used=True,
            fallback_reason="decorative_from_book requested but book has no valid records",
        )

    if mode == "sample_fallback":
        return _selection_metadata(
            role=role,
            source=source,
            givens81=fallback,
            payload=None,
            fallback_used=True,
            fallback_reason="sample_fallback requested",
        )

    return _selection_metadata(
        role=role,
        source=source,
        givens81=fallback,
        payload=None,
        fallback_used=True,
        fallback_reason=f"unsupported puzzle_art mode: {mode}",
    )


def resolve_cover_puzzle_art_variables(
    *,
    context_variables: dict[str, Any],
    book_dir: Path | None,
    puzzle_records_dir: Path | None = None,
) -> dict[str, Any]:
    puzzle_art = dict(context_variables.get("puzzle_art") or {})

    if book_dir is None:
        return {
            "main_givens81": SAMPLE_MAIN_GIVENS81,
            "left_side_givens81": SAMPLE_LEFT_GIVENS81,
            "right_side_givens81": SAMPLE_RIGHT_GIVENS81,
            "selections": {
                "main": _selection_metadata(
                    role="main",
                    source={"mode": "sample_fallback"},
                    givens81=SAMPLE_MAIN_GIVENS81,
                    payload=None,
                    fallback_used=True,
                    fallback_reason="No book_dir supplied",
                ),
                "left": _selection_metadata(
                    role="left",
                    source={"mode": "sample_fallback"},
                    givens81=SAMPLE_LEFT_GIVENS81,
                    payload=None,
                    fallback_used=True,
                    fallback_reason="No book_dir supplied",
                ),
                "right": _selection_metadata(
                    role="right",
                    source={"mode": "sample_fallback"},
                    givens81=SAMPLE_RIGHT_GIVENS81,
                    payload=None,
                    fallback_used=True,
                    fallback_reason="No book_dir supplied",
                ),
            },
            "book_record_count": 0,
            "resolution_notes": ["No book_dir supplied; using sample fallback puzzle art."],
        }

    book_dir = Path(book_dir)
    records_dir = Path(puzzle_records_dir) if puzzle_records_dir is not None else _default_puzzle_records_dir(book_dir)
    book_records = _load_book_records(book_dir, records_dir)

    main_selection = _resolve_source(
        dict(puzzle_art.get("main_grid_source") or {"mode": "sample_fallback"}),
        role="main",
        book_records=book_records,
        puzzle_records_dir=records_dir,
        fallback=SAMPLE_MAIN_GIVENS81,
    )
    left_selection = _resolve_source(
        dict(puzzle_art.get("left_side_grid_source") or {"mode": "sample_fallback"}),
        role="left",
        book_records=book_records,
        puzzle_records_dir=records_dir,
        fallback=SAMPLE_LEFT_GIVENS81,
    )
    right_selection = _resolve_source(
        dict(puzzle_art.get("right_side_grid_source") or {"mode": "sample_fallback"}),
        role="right",
        book_records=book_records,
        puzzle_records_dir=records_dir,
        fallback=SAMPLE_RIGHT_GIVENS81,
    )

    return {
        "main_givens81": main_selection["givens81"],
        "left_side_givens81": left_selection["givens81"],
        "right_side_givens81": right_selection["givens81"],
        "selections": {
            "main": main_selection,
            "left": left_selection,
            "right": right_selection,
        },
        "book_record_count": len(book_records),
        "puzzle_records_dir": str(records_dir),
        "book_dir": str(book_dir),
        "resolution_notes": [],
    }