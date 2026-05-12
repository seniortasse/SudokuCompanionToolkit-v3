from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, List, Sequence

from python.publishing.puzzle_catalog.catalog_store import load_puzzle_records_from_dir
from python.publishing.schemas.models import PuzzleRecord
from python.publishing.techniques.technique_catalog import (
    TECHNIQUE_CATALOG,
    collapse_to_public_names,
    get_public_technique_name,
    normalize_technique_id,
    public_combo_label,
)


def _log(message: str = "") -> None:
    print(message, flush=True)


def _norm(value: Any) -> str:
    return normalize_technique_id(str(value or ""))


def _get(record: PuzzleRecord, name: str, default: Any = None) -> Any:
    return getattr(record, name, default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _safe_str(value: Any) -> str:
    return str(value or "").strip()


def _load_records(*, book_dir: str = "", records_dir: str = "") -> tuple[list[PuzzleRecord], Path, str]:
    if book_dir:
        root = Path(book_dir)
        puzzles_dir = root / "puzzles"
        if not puzzles_dir.exists():
            raise FileNotFoundError(f"Built book puzzles directory does not exist: {puzzles_dir}")
        return load_puzzle_records_from_dir(puzzles_dir), puzzles_dir, "built_book"

    if records_dir:
        root = Path(records_dir)
        if not root.exists():
            raise FileNotFoundError(f"Puzzle records directory does not exist: {root}")
        return load_puzzle_records_from_dir(root), root, "records_dir"

    raise ValueError("Provide either --book-dir or --records-dir.")


def _catalog_lookup_by_public_name(raw: str) -> str | None:
    wanted = _norm(raw)
    if not wanted:
        return None

    for key, entry in TECHNIQUE_CATALOG.items():
        candidates = {
            _norm(key),
            _norm(entry.engine_id),
            _norm(entry.canonical_id),
            _norm(entry.public_name),
            _norm(entry.public_name_plural),
        }
        if wanted in candidates:
            return _norm(entry.engine_id)

    return None


def _resolve_technique_query(raw: str) -> str:
    normalized = _norm(raw)
    if not normalized:
        return ""

    entry = TECHNIQUE_CATALOG.get(normalized)
    if entry:
        return _norm(entry.engine_id)

    by_public = _catalog_lookup_by_public_name(raw)
    if by_public:
        return by_public

    return normalized


def _parse_repeated_techniques(values: Sequence[str]) -> list[str]:
    resolved: list[str] = []
    for raw in values:
        for part in str(raw or "").replace(",", "+").split("+"):
            token = _resolve_technique_query(part)
            if token:
                resolved.append(token)
    return _unique(resolved)


def _parse_combo(raw: str) -> tuple[str, ...]:
    parts = [
        _resolve_technique_query(part)
        for part in str(raw or "").replace(",", "+").split("+")
        if str(part or "").strip()
    ]
    return tuple(_unique([part for part in parts if part]))


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        key = str(value)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _record_techniques(record: PuzzleRecord) -> set[str]:
    techniques: set[str] = set()

    for technique in list(_get(record, "techniques_used", []) or []):
        normalized = _resolve_technique_query(str(technique))
        if normalized:
            techniques.add(normalized)

    histogram = dict(_get(record, "technique_histogram", {}) or {})
    for technique in histogram.keys():
        normalized = _resolve_technique_query(str(technique))
        if normalized:
            techniques.add(normalized)

    return techniques


def _record_matches_technique_filters(
    record: PuzzleRecord,
    *,
    required_techniques: Sequence[str],
    any_techniques: Sequence[str],
    excluded_techniques: Sequence[str],
    combos: Sequence[tuple[str, ...]],
) -> bool:
    used = _record_techniques(record)

    if required_techniques and not all(t in used for t in required_techniques):
        return False

    if any_techniques and not any(t in used for t in any_techniques):
        return False

    if excluded_techniques and any(t in used for t in excluded_techniques):
        return False

    for combo in combos:
        if combo and not all(t in used for t in combo):
            return False

    return True


def _record_matches_basic_filters(
    record: PuzzleRecord,
    *,
    section: str,
    difficulty: str,
    pattern_id: Sequence[str],
    pattern_family_id: Sequence[str],
    min_weight: int | None,
    max_weight: int | None,
    min_clues: int | None,
    max_clues: int | None,
    min_technique_count: int | None,
    max_technique_count: int | None,
) -> bool:
    if section and _safe_str(_get(record, "section_code")).lower() != section.lower():
        return False

    if difficulty and _safe_str(_get(record, "puzzle_difficulty")).lower() != difficulty.lower():
        return False

    if pattern_id:
        wanted = {str(x) for x in pattern_id}
        if _safe_str(_get(record, "pattern_id")) not in wanted:
            return False

    if pattern_family_id:
        wanted = {str(x) for x in pattern_family_id}
        if _safe_str(_get(record, "pattern_family_id")) not in wanted:
            return False

    weight = _safe_int(_get(record, "weight"), default=0)
    clue_count = _safe_int(_get(record, "clue_count"), default=0)
    technique_count = _safe_int(_get(record, "technique_count"), default=0)

    if min_weight is not None and weight < min_weight:
        return False
    if max_weight is not None and weight > max_weight:
        return False
    if min_clues is not None and clue_count < min_clues:
        return False
    if max_clues is not None and clue_count > max_clues:
        return False
    if min_technique_count is not None and technique_count < min_technique_count:
        return False
    if max_technique_count is not None and technique_count > max_technique_count:
        return False

    return True


def _sort_records(records: Sequence[PuzzleRecord], sort_key: str) -> list[PuzzleRecord]:
    if sort_key == "weight_desc":
        return sorted(records, key=lambda r: (-_safe_int(_get(r, "weight")), _safe_str(_get(r, "record_id"))))

    if sort_key == "weight_asc":
        return sorted(records, key=lambda r: (_safe_int(_get(r, "weight")), _safe_str(_get(r, "record_id"))))

    if sort_key == "technique_count_desc":
        return sorted(
            records,
            key=lambda r: (
                -_safe_int(_get(r, "technique_count")),
                -_safe_int(_get(r, "weight")),
                _safe_str(_get(r, "record_id")),
            ),
        )

    if sort_key == "clue_count_asc":
        return sorted(records, key=lambda r: (_safe_int(_get(r, "clue_count")), _safe_str(_get(r, "record_id"))))

    return sorted(
        records,
        key=lambda r: (
            _safe_int(_get(r, "position_in_book"), default=10**9),
            _safe_int(_get(r, "position_in_section"), default=10**9),
            _safe_str(_get(r, "record_id")),
        ),
    )


def _record_to_row(record: PuzzleRecord) -> dict[str, Any]:
    used = sorted(_record_techniques(record))
    return {
        "record_id": _safe_str(_get(record, "record_id")),
        "puzzle_uid": _safe_str(_get(record, "puzzle_uid")),
        "section_code": _safe_str(_get(record, "section_code")),
        "position_in_section": _get(record, "position_in_section"),
        "position_in_book": _get(record, "position_in_book"),
        "puzzle_difficulty": _safe_str(_get(record, "puzzle_difficulty")),
        "difficulty_label": _safe_str(_get(record, "difficulty_label")),
        "weight": _get(record, "weight"),
        "clue_count": _get(record, "clue_count"),
        "technique_count": _get(record, "technique_count"),
        "pattern_id": _safe_str(_get(record, "pattern_id")),
        "pattern_family_id": _safe_str(_get(record, "pattern_family_id")),
        "techniques_used": used,
        "public_techniques_used": collapse_to_public_names(used, plural=True),
    }


def _summarize_matches(records: Sequence[PuzzleRecord], *, top_n: int) -> dict[str, Any]:
    technique_presence = Counter()
    technique_occurrences = Counter()
    pattern_counts = Counter()
    family_counts = Counter()
    pair_counts = Counter()
    triple_counts = Counter()
    quad_counts = Counter()

    for record in records:
        used = sorted(_record_techniques(record))
        for technique in used:
            technique_presence[technique] += 1

        histogram = dict(_get(record, "technique_histogram", {}) or {})
        if histogram:
            for technique, count in histogram.items():
                technique_occurrences[_resolve_technique_query(str(technique))] += _safe_int(count, default=0)
        else:
            for technique in used:
                technique_occurrences[technique] += 1

        pattern_counts[_safe_str(_get(record, "pattern_id")) or "(missing)"] += 1
        family_counts[_safe_str(_get(record, "pattern_family_id")) or "(missing)"] += 1

        for combo in combinations(used, 2):
            pair_counts[" + ".join(combo)] += 1
        for combo in combinations(used, 3):
            triple_counts[" + ".join(combo)] += 1
        for combo in combinations(used, 4):
            quad_counts[" + ".join(combo)] += 1

    return {
        "match_count": len(records),
        "top_techniques_by_presence": _top_techniques(technique_presence, top_n),
        "top_techniques_by_occurrence": _top_techniques(technique_occurrences, top_n),
        "top_patterns": _top_plain(pattern_counts, top_n),
        "top_pattern_families": _top_plain(family_counts, top_n),
        "top_2_technique_combos": _top_combos(pair_counts, top_n),
        "top_3_technique_combos": _top_combos(triple_counts, top_n),
        "top_4_technique_combos": _top_combos(quad_counts, top_n),
    }


def _top_plain(counter: Counter, top_n: int) -> list[dict[str, Any]]:
    return [
        {"key": str(key), "count": int(count)}
        for key, count in counter.most_common(top_n)
    ]


def _top_techniques(counter: Counter, top_n: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, count in counter.most_common(top_n):
        technique = _resolve_technique_query(str(key))
        rows.append(
            {
                "key": technique,
                "public_name": get_public_technique_name(technique, plural=True),
                "count": int(count),
            }
        )
    return rows


def _top_combos(counter: Counter, top_n: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, count in counter.most_common(top_n):
        combo = str(key)
        rows.append(
            {
                "key": combo,
                "public_name": public_combo_label(combo, plural=True),
                "count": int(count),
            }
        )
    return rows


def _print_summary(summary: dict[str, Any]) -> None:
    _log(f"Matches: {summary['match_count']}")
    _log()

    def print_rows(title: str, rows: Sequence[dict[str, Any]]) -> None:
        _log(title)
        if not rows:
            _log("  (none)")
            _log()
            return

        for row in rows:
            key = str(row.get("key") or "")
            public = str(row.get("public_name") or "")
            count = row.get("count")
            if public and public != key:
                _log(f"  {public}: {count}  [{key}]")
            else:
                _log(f"  {key}: {count}")
        _log()

    print_rows("Top techniques by presence:", summary.get("top_techniques_by_presence", []))
    print_rows("Top patterns:", summary.get("top_patterns", []))
    print_rows("Top pattern families:", summary.get("top_pattern_families", []))
    print_rows("Top 2-technique combos:", summary.get("top_2_technique_combos", []))


def _print_records(records: Sequence[PuzzleRecord], *, limit: int) -> None:
    if limit <= 0:
        return

    _log("Matching puzzles:")
    if not records:
        _log("  (none)")
        _log()
        return

    for index, record in enumerate(records[:limit], start=1):
        row = _record_to_row(record)
        public_techniques = ", ".join(row["public_techniques_used"])
        engine_techniques = ", ".join(row["techniques_used"])
        _log(
            f"  {index:>3}. "
            f"{row['record_id']} | {row['puzzle_uid']} | "
            f"section={row['section_code']} pos={row['position_in_section']} "
            f"book_pos={row['position_in_book']} | "
            f"diff={row['puzzle_difficulty']} | "
            f"weight={row['weight']} clues={row['clue_count']} techs={row['technique_count']} | "
            f"pattern={row['pattern_id']} family={row['pattern_family_id']}"
        )
        _log(f"       public: {public_techniques}")
        _log(f"       engine: {engine_techniques}")

    if len(records) > limit:
        _log(f"  ... {len(records) - limit} more not shown")
    _log()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query built-book or catalog puzzle records by technique, combo, pattern, difficulty, and numeric filters."
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--book-dir",
        default="",
        help="Built book directory, e.g. datasets/sudoku_books/classic9/books/BK-CL9-DW-B02",
    )
    source.add_argument(
        "--records-dir",
        default="",
        help="Puzzle records directory, e.g. datasets/sudoku_books/classic9/puzzle_records",
    )

    parser.add_argument("--section", default="", help="Filter by section code, e.g. L1, L2, L3.")
    parser.add_argument("--difficulty", default="", help="Filter by puzzle_difficulty, e.g. medium, hard, expert.")

    parser.add_argument(
        "--technique",
        action="append",
        default=[],
        help="Required technique. Can be repeated. Accepts engine IDs or public names, e.g. x_wings_4 or Jellyfish.",
    )
    parser.add_argument(
        "--any-technique",
        action="append",
        default=[],
        help="At least one of these techniques must appear. Can be repeated.",
    )
    parser.add_argument(
        "--exclude-technique",
        action="append",
        default=[],
        help="Exclude puzzles containing this technique. Can be repeated.",
    )
    parser.add_argument(
        "--combo",
        action="append",
        default=[],
        help="Required technique combo, e.g. x_wings_4+x_wings_3 or Jellyfish+Swordfish.",
    )

    parser.add_argument(
        "--pattern-id",
        action="append",
        default=[],
        help="Filter by pattern id. Can be repeated.",
    )
    parser.add_argument(
        "--pattern-family-id",
        action="append",
        default=[],
        help="Filter by pattern family id. Can be repeated.",
    )

    parser.add_argument("--min-weight", type=int, default=None)
    parser.add_argument("--max-weight", type=int, default=None)
    parser.add_argument("--min-clues", type=int, default=None)
    parser.add_argument("--max-clues", type=int, default=None)
    parser.add_argument("--min-technique-count", type=int, default=None)
    parser.add_argument("--max-technique-count", type=int, default=None)

    parser.add_argument(
        "--sort",
        default="book_order",
        choices=[
            "book_order",
            "weight_asc",
            "weight_desc",
            "technique_count_desc",
            "clue_count_asc",
        ],
    )
    parser.add_argument("--show", type=int, default=30, help="How many matching puzzles to print.")
    parser.add_argument("--top-n", type=int, default=20, help="How many top stats to include.")
    parser.add_argument("--output-json", default="", help="Optional JSON report output path.")

    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    records, source_path, source_kind = _load_records(book_dir=args.book_dir, records_dir=args.records_dir)

    required_techniques = _parse_repeated_techniques(args.technique)
    any_techniques = _parse_repeated_techniques(args.any_technique)
    excluded_techniques = _parse_repeated_techniques(args.exclude_technique)
    combos = [_parse_combo(raw) for raw in list(args.combo or []) if str(raw or "").strip()]
    combos = [combo for combo in combos if combo]

    matches: list[PuzzleRecord] = []
    for record in records:
        if not _record_matches_basic_filters(
            record,
            section=str(args.section or "").strip(),
            difficulty=str(args.difficulty or "").strip(),
            pattern_id=list(args.pattern_id or []),
            pattern_family_id=list(args.pattern_family_id or []),
            min_weight=args.min_weight,
            max_weight=args.max_weight,
            min_clues=args.min_clues,
            max_clues=args.max_clues,
            min_technique_count=args.min_technique_count,
            max_technique_count=args.max_technique_count,
        ):
            continue

        if not _record_matches_technique_filters(
            record,
            required_techniques=required_techniques,
            any_techniques=any_techniques,
            excluded_techniques=excluded_techniques,
            combos=combos,
        ):
            continue

        matches.append(record)

    matches = _sort_records(matches, args.sort)
    summary = _summarize_matches(matches, top_n=int(args.top_n))

    query = {
        "source_kind": source_kind,
        "source_path": str(source_path),
        "total_records_loaded": len(records),
        "required_techniques": required_techniques,
        "required_techniques_public": collapse_to_public_names(required_techniques, plural=True),
        "any_techniques": any_techniques,
        "any_techniques_public": collapse_to_public_names(any_techniques, plural=True),
        "excluded_techniques": excluded_techniques,
        "excluded_techniques_public": collapse_to_public_names(excluded_techniques, plural=True),
        "combos": [" + ".join(combo) for combo in combos],
        "combos_public": [public_combo_label(" + ".join(combo), plural=True) for combo in combos],
        "section": str(args.section or "").strip(),
        "difficulty": str(args.difficulty or "").strip(),
        "pattern_id": list(args.pattern_id or []),
        "pattern_family_id": list(args.pattern_family_id or []),
        "min_weight": args.min_weight,
        "max_weight": args.max_weight,
        "min_clues": args.min_clues,
        "max_clues": args.max_clues,
        "min_technique_count": args.min_technique_count,
        "max_technique_count": args.max_technique_count,
        "sort": args.sort,
    }

    report = {
        "query": query,
        "summary": summary,
        "matches": [_record_to_row(record) for record in matches],
    }

    _log("=" * 72)
    _log("query_puzzles.py")
    _log("=" * 72)
    _log(f"Source:  {source_kind}")
    _log(f"Path:    {source_path}")
    _log(f"Loaded:  {len(records)} puzzle records")
    _log(f"Matched: {len(matches)} puzzle records")
    _log("=" * 72)
    _log()

    if required_techniques:
        _log(f"Required techniques: {', '.join(collapse_to_public_names(required_techniques, plural=True))}")
    if any_techniques:
        _log(f"Any techniques:      {', '.join(collapse_to_public_names(any_techniques, plural=True))}")
    if excluded_techniques:
        _log(f"Excluded techniques: {', '.join(collapse_to_public_names(excluded_techniques, plural=True))}")
    if combos:
        _log(f"Required combos:     {', '.join(public_combo_label(' + '.join(c), plural=True) for c in combos)}")
    if required_techniques or any_techniques or excluded_techniques or combos:
        _log()

    _print_summary(summary)
    _print_records(matches, limit=int(args.show))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        _log(f"JSON report written: {output_path}")

    _log("=" * 72)
    _log("query_puzzles.py completed successfully")
    _log("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())