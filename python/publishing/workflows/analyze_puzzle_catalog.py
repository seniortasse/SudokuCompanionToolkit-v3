from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Sequence

from python.publishing.book_builder.criteria_engine import (
    record_matches_global_filters,
    record_matches_section_criteria,
)
from python.publishing.puzzle_catalog.catalog_store import load_puzzle_records_from_dir
from python.publishing.schemas.models import PuzzleRecord


def _log(message: str = "") -> None:
    print(message, flush=True)


def _safe_mean(values: Sequence[int | float]) -> float | None:
    if not values:
        return None
    return round(float(mean(values)), 3)


def _safe_median(values: Sequence[int | float]) -> float | None:
    if not values:
        return None
    return round(float(median(values)), 3)


def _top_counter(counter: Counter, limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key, count in counter.most_common(limit):
        rows.append({"key": str(key), "count": int(count)})
    return rows


def _build_distribution(values: Iterable[str]) -> Dict[str, int]:
    counter = Counter()
    for value in values:
        token = str(value).strip() if value is not None else ""
        if not token:
            token = "(missing)"
        counter[token] += 1
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _summed_technique_histogram(records: Sequence[PuzzleRecord]) -> Dict[str, int]:
    total = Counter()
    for record in records:
        for name, count in dict(record.technique_histogram or {}).items():
            total[str(name)] += int(count)
    return dict(sorted(total.items(), key=lambda item: (-item[1], item[0])))


def _presence_technique_counter(records: Sequence[PuzzleRecord]) -> Counter:
    counter = Counter()
    for record in records:
        for name in list(record.techniques_used or []):
            counter[str(name)] += 1
    return counter


def _pattern_counter(records: Sequence[PuzzleRecord]) -> Counter:
    counter = Counter()
    for record in records:
        key = record.pattern_id or "(missing)"
        counter[str(key)] += 1
    return counter


def _pattern_family_counter(records: Sequence[PuzzleRecord]) -> Counter:
    counter = Counter()
    for record in records:
        key = record.pattern_family_id or "(missing)"
        counter[str(key)] += 1
    return counter


def _featured_technique_counter(records: Sequence[PuzzleRecord]) -> Counter:
    counter = Counter()
    for record in records:
        key = record.featured_technique or "(missing)"
        counter[str(key)] += 1
    return counter


def _numeric_summary(records: Sequence[PuzzleRecord], attr: str) -> Dict[str, Any]:
    values = [int(getattr(record, attr)) for record in records]
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
        }
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": _safe_mean(values),
        "median": _safe_median(values),
    }


def _field_coverage(records: Sequence[PuzzleRecord]) -> Dict[str, int]:
    def count_nonempty(fn) -> int:
        return sum(1 for record in records if fn(record))

    return {
        "pattern_id_present": count_nonempty(lambda r: bool(r.pattern_id)),
        "pattern_family_id_present": count_nonempty(lambda r: bool(r.pattern_family_id)),
        "featured_technique_present": count_nonempty(lambda r: bool(r.featured_technique)),
        "techniques_used_present": count_nonempty(lambda r: bool(r.techniques_used)),
        "difficulty_label_present": count_nonempty(lambda r: bool(r.difficulty_label)),
        "puzzle_difficulty_present": count_nonempty(lambda r: bool(r.puzzle_difficulty)),
        "clue_count_present": count_nonempty(lambda r: r.clue_count is not None),
        "weight_present": count_nonempty(lambda r: r.weight is not None),
    }


def _aisle_suitability(records: Sequence[PuzzleRecord]) -> Dict[str, Dict[str, Any]]:
    total = len(records)
    return {
        "AIS-DWEIGHT": {
            "title": "Puzzle Difficulty / weight_range",
            "eligible_count": sum(1 for r in records if r.weight is not None and r.difficulty_label),
            "notes": "Uses weight and difficulty_label fields.",
        },
        "AIS-TCOUNT": {
            "title": "Technique Count",
            "eligible_count": sum(1 for r in records if r.technique_count is not None),
            "notes": "Uses technique_count field.",
        },
        "AIS-CTECH": {
            "title": "Custom Technique",
            "eligible_count": sum(1 for r in records if bool(r.techniques_used)),
            "notes": "Uses techniques_used / featured_technique fields.",
        },
        "AIS-PATTERN": {
            "title": "Custom Pattern",
            "eligible_count": sum(1 for r in records if bool(r.pattern_id or r.pattern_family_id)),
            "notes": "Uses pattern_id / pattern_family_id fields.",
        },
        "AIS-MISC": {
            "title": "Miscellaneous",
            "eligible_count": total,
            "notes": "Custom/manual aisle; all records are potentially usable depending on the book spec.",
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze canonical puzzle records and report catalog composition and subset capacity."
    )
    parser.add_argument(
        "--puzzle-records-dir",
        default="datasets/sudoku_books/classic9/puzzle_records",
        help="Directory containing canonical puzzle record JSON files.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to write the analysis report as JSON.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=25,
        help="Number of top entries to print for pattern/technique distributions.",
    )

    # Optional subset filters for planning a future book.
    parser.add_argument("--weight-min", type=int, default=None)
    parser.add_argument("--weight-max", type=int, default=None)
    parser.add_argument("--clue-count-min", type=int, default=None)
    parser.add_argument("--clue-count-max", type=int, default=None)
    parser.add_argument("--technique-count-min", type=int, default=None)
    parser.add_argument("--technique-count-max", type=int, default=None)
    parser.add_argument(
        "--puzzle-difficulty",
        default="",
        help="Optional exact puzzle_difficulty filter (easy, medium, hard, expert, genius).",
    )
    parser.add_argument(
        "--puzzle-difficulty-in",
        action="append",
        default=[],
        help="Optional repeated allowed puzzle_difficulty values.",
    )
    parser.add_argument(
        "--required-technique",
        action="append",
        default=[],
        help="Optional repeated required technique.",
    )
    parser.add_argument(
        "--required-any-technique",
        action="append",
        default=[],
        help="Optional repeated technique; at least one must be present.",
    )
    parser.add_argument(
        "--excluded-technique",
        action="append",
        default=[],
        help="Optional repeated excluded technique.",
    )
    parser.add_argument(
        "--pattern-id",
        action="append",
        default=[],
        help="Optional repeated pattern_id filter.",
    )
    parser.add_argument(
        "--pattern-family-id",
        action="append",
        default=[],
        help="Optional repeated pattern_family_id filter.",
    )
    parser.add_argument(
        "--candidate-status-in",
        action="append",
        default=["available"],
        help="Global candidate status filter. Defaults to available.",
    )
    parser.add_argument(
        "--include-all-statuses",
        action="store_true",
        help="Ignore candidate_status filter and include all records.",
    )

    return parser.parse_args()


def _normalize_list(values: Sequence[str]) -> List[str]:
    return [str(v).strip() for v in values if str(v).strip()]


def _build_filters(args: argparse.Namespace) -> tuple[Dict[str, Any], Dict[str, Any]]:
    global_filters: Dict[str, Any] = {}
    if not args.include_all_statuses:
        statuses = _normalize_list(args.candidate_status_in)
        if statuses:
            global_filters["candidate_status_in"] = statuses

    section_criteria: Dict[str, Any] = {}
    if args.weight_min is not None:
        section_criteria["weight_min"] = int(args.weight_min)
    if args.weight_max is not None:
        section_criteria["weight_max"] = int(args.weight_max)
    if args.clue_count_min is not None:
        section_criteria["clue_count_min"] = int(args.clue_count_min)
    if args.clue_count_max is not None:
        section_criteria["clue_count_max"] = int(args.clue_count_max)
    if args.technique_count_min is not None:
        section_criteria["technique_count_min"] = int(args.technique_count_min)
    if args.technique_count_max is not None:
        section_criteria["technique_count_max"] = int(args.technique_count_max)

    puzzle_difficulty = str(args.puzzle_difficulty).strip()
    if puzzle_difficulty:
        section_criteria["puzzle_difficulty"] = puzzle_difficulty

    puzzle_difficulty_in = _normalize_list(args.puzzle_difficulty_in)
    if puzzle_difficulty_in:
        section_criteria["puzzle_difficulty_in"] = puzzle_difficulty_in

    required_techniques = _normalize_list(args.required_technique)
    if required_techniques:
        section_criteria["required_techniques"] = required_techniques

    required_any_techniques = _normalize_list(args.required_any_technique)
    if required_any_techniques:
        section_criteria["required_any_techniques"] = required_any_techniques

    excluded_techniques = _normalize_list(args.excluded_technique)
    if excluded_techniques:
        section_criteria["excluded_techniques"] = excluded_techniques

    pattern_ids = _normalize_list(args.pattern_id)
    if pattern_ids:
        section_criteria["pattern_ids"] = pattern_ids

    family_ids = _normalize_list(args.pattern_family_id)
    if family_ids:
        section_criteria["pattern_family_ids"] = family_ids

    return global_filters, section_criteria


def _filter_records(
    records: Sequence[PuzzleRecord],
    *,
    global_filters: Dict[str, Any],
    section_criteria: Dict[str, Any],
) -> List[PuzzleRecord]:
    filtered: List[PuzzleRecord] = []
    for record in records:
        if not record_matches_global_filters(record, global_filters):
            continue
        if not record_matches_section_criteria(record, section_criteria):
            continue
        filtered.append(record)
    return filtered


def _build_report(records: Sequence[PuzzleRecord], *, top_n: int) -> Dict[str, Any]:
    weights = _numeric_summary(records, "weight")
    clue_counts = _numeric_summary(records, "clue_count")
    technique_counts = _numeric_summary(records, "technique_count")

    difficulty_label_distribution = _build_distribution(record.difficulty_label for record in records)
    puzzle_difficulty_distribution = _build_distribution(record.puzzle_difficulty for record in records)
    difficulty_version_distribution = _build_distribution(record.difficulty_version for record in records)
    difficulty_band_distribution = _build_distribution(record.difficulty_band_code for record in records)
    candidate_status_distribution = _build_distribution(record.candidate_status for record in records)

    pattern_counts = _pattern_counter(records)
    family_counts = _pattern_family_counter(records)
    featured_counts = _featured_technique_counter(records)
    technique_presence = _presence_technique_counter(records)
    technique_histogram_total = _summed_technique_histogram(records)

    return {
        "total_records": len(records),
        "field_coverage": _field_coverage(records),
        "aisle_suitability": _aisle_suitability(records),
        "weight_summary": weights,
        "clue_count_summary": clue_counts,
        "technique_count_summary": technique_counts,
        "difficulty_label_distribution": difficulty_label_distribution,
        "difficulty_band_distribution": difficulty_band_distribution,
        "puzzle_difficulty_distribution": puzzle_difficulty_distribution,
        "difficulty_version_distribution": difficulty_version_distribution,
        "candidate_status_distribution": candidate_status_distribution,
        "unique_patterns": len(pattern_counts),
        "unique_pattern_families": len(family_counts),
        "top_patterns": _top_counter(pattern_counts, top_n),
        "top_pattern_families": _top_counter(family_counts, top_n),
        "top_featured_techniques": _top_counter(featured_counts, top_n),
        "top_techniques_by_record_presence": _top_counter(technique_presence, top_n),
        "overall_technique_histogram": technique_histogram_total,
        "top_techniques_by_total_occurrences": _top_counter(Counter(technique_histogram_total), top_n),
    }


def _print_distribution(title: str, data: Dict[str, int], *, limit: int | None = None) -> None:
    _log(title)
    items = list(data.items())
    if limit is not None:
        items = items[:limit]
    for key, value in items:
        _log(f"  {key}: {value}")
    if not items:
        _log("  (none)")
    _log()


def _print_top_rows(title: str, rows: List[Dict[str, Any]]) -> None:
    _log(title)
    for row in rows:
        _log(f"  {row['key']}: {row['count']}")
    if not rows:
        _log("  (none)")
    _log()


def main() -> int:
    args = _parse_args()

    puzzle_records_dir = Path(args.puzzle_records_dir)
    if not puzzle_records_dir.exists():
        _log(f"ERROR: puzzle records directory does not exist: {puzzle_records_dir}")
        return 1

    records = load_puzzle_records_from_dir(puzzle_records_dir)
    if not records:
        _log("ERROR: no puzzle records found.")
        return 1

    global_filters, section_criteria = _build_filters(args)
    filtered = _filter_records(
        records,
        global_filters=global_filters,
        section_criteria=section_criteria,
    )

    full_report = _build_report(records, top_n=args.top_n)
    subset_report = _build_report(filtered, top_n=args.top_n)

    _log("=" * 72)
    _log("analyze_puzzle_catalog.py")
    _log("=" * 72)
    _log(f"Puzzle records dir: {puzzle_records_dir.resolve()}")
    _log(f"Total catalog records: {len(records)}")
    _log(f"Filtered subset records: {len(filtered)}")
    _log(f"Global filters: {json.dumps(global_filters, ensure_ascii=False)}")
    _log(f"Section criteria: {json.dumps(section_criteria, ensure_ascii=False)}")
    _log("=" * 72)
    _log()

    for label, report in (("FULL CATALOG", full_report), ("FILTERED SUBSET", subset_report)):
        _log("-" * 72)
        _log(label)
        _log("-" * 72)
        _log(f"Total records: {report['total_records']}")
        _log(f"Unique patterns: {report['unique_patterns']}")
        _log(f"Unique pattern families: {report['unique_pattern_families']}")
        _log()

        _log(f"Weight summary: {report['weight_summary']}")
        _log(f"Clue count summary: {report['clue_count_summary']}")
        _log(f"Technique count summary: {report['technique_count_summary']}")
        _log()

        _print_distribution("Difficulty label distribution:", report["difficulty_label_distribution"])
        _print_distribution("Puzzle difficulty distribution:", report["puzzle_difficulty_distribution"])
        _print_distribution("Difficulty band distribution:", report["difficulty_band_distribution"])
        _print_distribution("Candidate status distribution:", report["candidate_status_distribution"])

        _print_top_rows("Top patterns:", report["top_patterns"])
        _print_top_rows("Top pattern families:", report["top_pattern_families"])
        _print_top_rows("Top featured techniques:", report["top_featured_techniques"])
        _print_top_rows("Top techniques by record presence:", report["top_techniques_by_record_presence"])
        _print_top_rows("Top techniques by total occurrences:", report["top_techniques_by_total_occurrences"])

        _log("Aisle suitability:")
        for aisle_id, payload in report["aisle_suitability"].items():
            _log(f"  {aisle_id}: eligible_count={payload['eligible_count']}  ({payload['title']})")
        _log()

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "puzzle_records_dir": str(puzzle_records_dir),
                    "global_filters": global_filters,
                    "section_criteria": section_criteria,
                    "full_catalog": full_report,
                    "filtered_subset": subset_report,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        _log(f"JSON report written: {output_path}")

    _log("=" * 72)
    _log("analyze_puzzle_catalog.py completed successfully")
    _log("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())