from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from python.publishing.puzzle_catalog.catalog_store import load_puzzle_records_from_dir
from python.publishing.schemas.models import PuzzleRecord
from python.publishing.techniques.technique_catalog import (
    get_public_technique_name,
    normalize_technique_id,
    public_combo_label,
)


EASY_TECHNIQUES = {
    "singles_1",
    "singles_2",
    "singles_3",
}

DEFAULT_DISLIKED_TECHNIQUES = [
    "y_wings",
    "quads_naked",
    "triplets_naked",
]

DEFAULT_PREFERRED_COMBOS = [
    ("singles_naked_2", "singles_naked_3"),
    ("singles_pointing", "singles_boxed"),
    ("doubles_naked", "singles_pointing"),
    ("doubles_naked", "singles_boxed"),
]


def _log(message: str = "") -> None:
    print(message, flush=True)


def _norm(value: Any) -> str:
    return normalize_technique_id(str(value))


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


def _top_technique_counter(counter: Counter, limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key, count in counter.most_common(limit):
        engine_id = _norm(key)
        rows.append(
            {
                "key": engine_id,
                "public_name": get_public_technique_name(engine_id, plural=True),
                "count": int(count),
            }
        )
    return rows


def _top_combo_counter(counter: Counter, limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key, count in counter.most_common(limit):
        engine_combo = str(key)
        rows.append(
            {
                "key": engine_combo,
                "public_name": public_combo_label(engine_combo, plural=True),
                "count": int(count),
            }
        )
    return rows


def _numeric_summary(values: Sequence[int | float]) -> Dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "first_10": [],
            "last_10": [],
        }
    ordered = list(values)
    return {
        "count": len(ordered),
        "min": min(ordered),
        "max": max(ordered),
        "mean": _safe_mean(ordered),
        "median": _safe_median(ordered),
        "first_10": ordered[:10],
        "last_10": ordered[-10:],
    }


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
            total[_norm(name)] += int(count)
    return dict(sorted(total.items(), key=lambda item: (-item[1], item[0])))


def _technique_presence_counter(records: Sequence[PuzzleRecord]) -> Counter:
    counter = Counter()
    for record in records:
        used = {_norm(t) for t in list(record.techniques_used or [])}
        for name in used:
            counter[name] += 1
    return counter


def _pattern_counter(records: Sequence[PuzzleRecord]) -> Counter:
    counter = Counter()
    for record in records:
        counter[str(record.pattern_id or "(missing)")] += 1
    return counter


def _family_counter(records: Sequence[PuzzleRecord]) -> Counter:
    counter = Counter()
    for record in records:
        counter[str(record.pattern_family_id or "(missing)")] += 1
    return counter


def _records_sorted_in_section(records: Sequence[PuzzleRecord]) -> List[PuzzleRecord]:
    return sorted(
        records,
        key=lambda r: (
            int(r.position_in_section) if r.position_in_section is not None else 10**9,
            int(r.position_in_book) if r.position_in_book is not None else 10**9,
            str(r.record_id),
        ),
    )


def _combo_counter(
    records: Sequence[PuzzleRecord],
    *,
    combo_size: int,
    ignore_easy: bool,
) -> Counter:
    counter = Counter()
    for record in records:
        used = {_norm(t) for t in list(record.techniques_used or [])}
        if ignore_easy:
            used = {t for t in used if t not in EASY_TECHNIQUES}
        if len(used) < combo_size:
            continue
        for combo in combinations(sorted(used), combo_size):
            counter[" + ".join(combo)] += 1
    return counter


def _preferred_combo_summary(
    records: Sequence[PuzzleRecord],
    preferred_combos: Sequence[Tuple[str, ...]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for combo in preferred_combos:
        normalized_combo = tuple(_norm(x) for x in combo)
        count = 0
        matching_ids: List[str] = []
        for record in records:
            used = {_norm(t) for t in list(record.techniques_used or [])}
            if all(token in used for token in normalized_combo):
                count += 1
                if len(matching_ids) < 10:
                    matching_ids.append(str(record.record_id))
        combo_label = " + ".join(normalized_combo)
        rows.append(
            {
                "combo": list(normalized_combo),
                "public_combo": public_combo_label(combo_label, plural=True),
                "count": count,
                "sample_record_ids": matching_ids,
            }
        )
    rows.sort(key=lambda row: (-row["count"], row["combo"]))
    return rows


def _disliked_summary(
    records: Sequence[PuzzleRecord],
    disliked_techniques: Sequence[str],
) -> List[Dict[str, Any]]:
    normalized_targets = [_norm(x) for x in disliked_techniques]
    histogram_total = _summed_technique_histogram(records)

    rows: List[Dict[str, Any]] = []
    for target in normalized_targets:
        presence = 0
        total_occurrences = int(histogram_total.get(target, 0))
        sample_record_ids: List[str] = []
        for record in records:
            used = {_norm(t) for t in list(record.techniques_used or [])}
            if target in used:
                presence += 1
                if len(sample_record_ids) < 10:
                    sample_record_ids.append(str(record.record_id))
        rows.append(
            {
                "technique": target,
                "public_technique": get_public_technique_name(target, plural=True),
                "record_presence": presence,
                "total_occurrences": total_occurrences,
                "sample_record_ids": sample_record_ids,
            }
        )
    rows.sort(key=lambda row: (-row["record_presence"], row["technique"]))
    return rows


def _top_density_puzzles(records: Sequence[PuzzleRecord], limit: int = 15) -> List[Dict[str, Any]]:
    ordered = sorted(
        records,
        key=lambda r: (
            -(int(r.technique_count) if r.technique_count is not None else -1),
            -(int(r.weight) if r.weight is not None else -1),
            str(r.record_id),
        ),
    )
    rows: List[Dict[str, Any]] = []
    for record in ordered[:limit]:
        rows.append(
            {
                "record_id": str(record.record_id),
                "section_code": str(record.section_code),
                "pattern_id": str(record.pattern_id or ""),
                "pattern_family_id": str(record.pattern_family_id or ""),
                "weight": int(record.weight),
                "clue_count": int(record.clue_count),
                "technique_count": int(record.technique_count),
                "puzzle_difficulty": str(record.puzzle_difficulty),
                "techniques_used": list(record.techniques_used or []),
                "public_techniques_used": [
                    get_public_technique_name(str(t), plural=True)
                    for t in list(record.techniques_used or [])
                ],
            }
        )
    return rows


def _top_weight_puzzles(records: Sequence[PuzzleRecord], limit: int = 15) -> List[Dict[str, Any]]:
    ordered = sorted(records, key=lambda r: (-int(r.weight), str(r.record_id)))
    rows: List[Dict[str, Any]] = []
    for record in ordered[:limit]:
        rows.append(
            {
                "record_id": str(record.record_id),
                "section_code": str(record.section_code),
                "pattern_id": str(record.pattern_id or ""),
                "pattern_family_id": str(record.pattern_family_id or ""),
                "weight": int(record.weight),
                "clue_count": int(record.clue_count),
                "technique_count": int(record.technique_count),
                "puzzle_difficulty": str(record.puzzle_difficulty),
                "techniques_used": list(record.techniques_used or []),
                "public_techniques_used": [
                    get_public_technique_name(str(t), plural=True)
                    for t in list(record.techniques_used or [])
                ],
            }
        )
    return rows


def _concentration_summary(counter: Counter, total: int, top_n: int = 10) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    cumulative = 0
    for key, count in counter.most_common(top_n):
        cumulative += int(count)
        pct = round((int(count) / total) * 100.0, 3) if total else 0.0
        rows.append(
            {
                "key": str(key),
                "count": int(count),
                "pct_of_section": pct,
            }
        )
    return {
        "top": rows,
        "top_n_total_pct": round((cumulative / total) * 100.0, 3) if total else 0.0,
    }


def _section_report(
    section_code: str,
    records: Sequence[PuzzleRecord],
    *,
    disliked_techniques: Sequence[str],
    preferred_combos: Sequence[Tuple[str, ...]],
    top_n: int,
) -> Dict[str, Any]:
    ordered = _records_sorted_in_section(records)

    weights = [int(r.weight) for r in ordered]
    clue_counts = [int(r.clue_count) for r in ordered]
    technique_counts = [int(r.technique_count) for r in ordered]

    pattern_counts = _pattern_counter(ordered)
    family_counts = _family_counter(ordered)
    technique_presence = _technique_presence_counter(ordered)
    technique_histogram = _summed_technique_histogram(ordered)

    pair_counter = _combo_counter(ordered, combo_size=2, ignore_easy=True)
    triple_counter = _combo_counter(ordered, combo_size=3, ignore_easy=True)
    quad_counter = _combo_counter(ordered, combo_size=4, ignore_easy=True)

    first_record = ordered[0] if ordered else None
    last_record = ordered[-1] if ordered else None

    return {
        "section_code": section_code,
        "puzzle_count": len(ordered),
        "first_record_id": str(first_record.record_id) if first_record else None,
        "last_record_id": str(last_record.record_id) if last_record else None,
        "weight_summary": _numeric_summary(weights),
        "clue_count_summary": _numeric_summary(clue_counts),
        "technique_count_summary": _numeric_summary(technique_counts),
        "puzzle_difficulty_distribution": _build_distribution(r.puzzle_difficulty for r in ordered),
        "difficulty_label_distribution": _build_distribution(r.difficulty_label for r in ordered),
        "unique_patterns": len(pattern_counts),
        "unique_pattern_families": len(family_counts),
        "top_patterns": _top_counter(pattern_counts, top_n),
        "top_pattern_families": _top_counter(family_counts, top_n),
        "pattern_concentration": _concentration_summary(pattern_counts, len(ordered), top_n=min(top_n, 10)),
        "family_concentration": _concentration_summary(family_counts, len(ordered), top_n=min(top_n, 10)),
        "top_techniques_by_record_presence": _top_technique_counter(technique_presence, top_n),
        "top_techniques_by_total_occurrences": _top_technique_counter(Counter(technique_histogram), top_n),
        "disliked_techniques": _disliked_summary(ordered, disliked_techniques),
        "preferred_combos": _preferred_combo_summary(ordered, preferred_combos),
        "top_2_technique_combos": _top_combo_counter(pair_counter, top_n),
        "top_3_technique_combos": _top_combo_counter(triple_counter, top_n),
        "top_4_technique_combos": _top_combo_counter(quad_counter, top_n),
        "top_weight_puzzles": _top_weight_puzzles(ordered, limit=15),
        "top_density_puzzles": _top_density_puzzles(ordered, limit=15),
    }


def _load_book_manifest(book_dir: Path) -> Dict[str, Any]:
    path = book_dir / "book_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_section_manifests(book_dir: Path) -> List[Dict[str, Any]]:
    sections_dir = book_dir / "sections"
    manifests: List[Dict[str, Any]] = []
    for path in sorted(sections_dir.glob("*.json")):
        manifests.append(json.loads(path.read_text(encoding="utf-8")))
    manifests.sort(key=lambda m: (int(m.get("section_order", 9999)), str(m.get("section_code", ""))))
    return manifests


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a built book section-by-section, including diversity, disliked techniques, and combo statistics."
    )
    parser.add_argument(
        "--book-dir",
        required=True,
        help="Path to a built book directory, e.g. datasets/sudoku_books/classic9/books/BK-CL9-DW-B01",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to write a JSON report.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="How many top rows to show for distributions and combo lists.",
    )
    parser.add_argument(
        "--disliked-technique",
        action="append",
        default=[],
        help="Repeated disliked technique name. Defaults to y_wings, quads_naked, triplets_naked.",
    )
    parser.add_argument(
        "--preferred-combo",
        action="append",
        default=[],
        help="Repeated preferred combo in the form techniqueA+techniqueB or techniqueA+techniqueB+techniqueC.",
    )
    return parser.parse_args()


def _parse_preferred_combos(raw_values: Sequence[str]) -> List[Tuple[str, ...]]:
    if not raw_values:
        return [tuple(x) for x in DEFAULT_PREFERRED_COMBOS]

    combos: List[Tuple[str, ...]] = []
    for raw in raw_values:
        parts = [_norm(x) for x in str(raw).split("+") if _norm(x)]
        if len(parts) >= 2:
            combos.append(tuple(parts))
    return combos


def _print_top_rows(title: str, rows: List[Dict[str, Any]]) -> None:
    _log(title)
    if not rows:
        _log("  (none)")
        _log()
        return

    for row in rows:
        public_name = str(row.get("public_name") or "").strip()
        engine_key = str(row.get("key") or "").strip()
        count = row.get("count")

        if public_name and public_name != engine_key:
            _log(f"  {public_name}: {count}  [{engine_key}]")
        else:
            _log(f"  {engine_key}: {count}")
    _log()


def _print_concentration(title: str, payload: Dict[str, Any]) -> None:
    _log(title)
    for row in payload.get("top", []):
        _log(f"  {row['key']}: {row['count']} ({row['pct_of_section']}%)")
    _log(f"  top_n_total_pct: {payload.get('top_n_total_pct')}")
    _log()


def _print_preferred_combos(rows: List[Dict[str, Any]]) -> None:
    _log("Preferred combo counts:")
    for row in rows:
        combo_label = str(row.get("public_combo") or " + ".join(row["combo"]))
        engine_combo = " + ".join(row["combo"])
        if combo_label != engine_combo:
            _log(f"  {combo_label}: {row['count']}  [{engine_combo}]")
        else:
            _log(f"  {combo_label}: {row['count']}")
    if not rows:
        _log("  (none)")
    _log()


def _print_disliked(rows: List[Dict[str, Any]]) -> None:
    _log("Disliked technique counts:")
    for row in rows:
        public_name = str(row.get("public_technique") or row["technique"])
        engine_id = str(row["technique"])
        label = f"{public_name} [{engine_id}]" if public_name != engine_id else engine_id
        _log(
            f"  {label}: presence={row['record_presence']} | total_occurrences={row['total_occurrences']}"
        )
    if not rows:
        _log("  (none)")
    _log()


def main() -> int:
    args = _parse_args()

    book_dir = Path(args.book_dir)
    if not book_dir.exists():
        _log(f"ERROR: built book directory does not exist: {book_dir}")
        return 1

    puzzles_dir = book_dir / "puzzles"
    if not puzzles_dir.exists():
        _log(f"ERROR: built book puzzles directory does not exist: {puzzles_dir}")
        return 1

    book_manifest = _load_book_manifest(book_dir)
    section_manifests = _load_section_manifests(book_dir)
    records = load_puzzle_records_from_dir(puzzles_dir)

    if not records:
        _log("ERROR: no puzzle records found in built book.")
        return 1

    disliked_techniques = args.disliked_technique or list(DEFAULT_DISLIKED_TECHNIQUES)
    preferred_combos = _parse_preferred_combos(args.preferred_combo)

    by_section: Dict[str, List[PuzzleRecord]] = {}
    for record in records:
        code = str(record.section_code or "")
        by_section.setdefault(code, []).append(record)

    overall_report = _section_report(
        "ALL",
        records,
        disliked_techniques=disliked_techniques,
        preferred_combos=preferred_combos,
        top_n=args.top_n,
    )

    section_reports: List[Dict[str, Any]] = []
    for manifest in section_manifests:
        code = str(manifest.get("section_code"))
        section_records = by_section.get(code, [])
        report = _section_report(
            code,
            section_records,
            disliked_techniques=disliked_techniques,
            preferred_combos=preferred_combos,
            top_n=args.top_n,
        )
        report["section_title"] = manifest.get("title")
        report["section_subtitle"] = manifest.get("subtitle")
        report["section_order"] = manifest.get("section_order")
        report["criteria"] = manifest.get("criteria", {})
        report["expected_puzzle_count"] = manifest.get("puzzle_count")
        section_reports.append(report)

    _log("=" * 72)
    _log("analyze_built_book.py")
    _log("=" * 72)
    _log(f"Book dir:   {book_dir.resolve()}")
    _log(f"Book id:    {book_manifest.get('book_id')}")
    _log(f"Title:      {book_manifest.get('title')}")
    _log(f"Subtitle:   {book_manifest.get('subtitle')}")
    _log(f"Total puzzles in built book: {len(records)}")
    _log(f"Sections:   {len(section_reports)}")
    _log("=" * 72)
    _log()

    _log("OVERALL BUILT BOOK")
    _log("-" * 72)
    _log(f"Weight summary: {overall_report['weight_summary']}")
    _log(f"Clue count summary: {overall_report['clue_count_summary']}")
    _log(f"Technique count summary: {overall_report['technique_count_summary']}")
    _log(f"Unique patterns: {overall_report['unique_patterns']}")
    _log(f"Unique pattern families: {overall_report['unique_pattern_families']}")
    _log()

    for report in section_reports:
        _log("-" * 72)
        _log(f"SECTION {report['section_code']} — {report.get('section_title')}")
        _log("-" * 72)
        _log(f"Expected puzzles: {report.get('expected_puzzle_count')}")
        _log(f"Actual puzzles:   {report['puzzle_count']}")
        _log(f"Criteria:         {json.dumps(report.get('criteria', {}), ensure_ascii=False)}")
        _log()
        _log(f"Weight summary: {report['weight_summary']}")
        _log(f"Clue count summary: {report['clue_count_summary']}")
        _log(f"Technique count summary: {report['technique_count_summary']}")
        _log(f"Unique patterns: {report['unique_patterns']}")
        _log(f"Unique families: {report['unique_pattern_families']}")
        _log()
        _print_concentration("Pattern concentration:", report["pattern_concentration"])
        _print_concentration("Family concentration:", report["family_concentration"])
        _print_disliked(report["disliked_techniques"])
        _print_preferred_combos(report["preferred_combos"])
        _print_top_rows("Top 2-technique combos:", report["top_2_technique_combos"])
        _print_top_rows("Top 3-technique combos:", report["top_3_technique_combos"])
        _print_top_rows("Top 4-technique combos:", report["top_4_technique_combos"])

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "book_manifest": book_manifest,
                    "overall_report": overall_report,
                    "section_reports": section_reports,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        _log(f"JSON report written: {output_path}")

    _log("=" * 72)
    _log("analyze_built_book.py completed successfully")
    _log("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())