from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

from python.publishing.ids.id_policy import build_aisle_id, build_library_id
from python.publishing.puzzle_catalog.catalog_dedupe import find_duplicate_record_id
from python.publishing.puzzle_catalog.catalog_index import (
    load_catalog_index,
    register_record,
    reserve_next_record_ordinal,
    save_catalog_index,
)
from python.publishing.puzzle_catalog.catalog_store import save_puzzle_records_batch
from python.publishing.puzzle_catalog.generator_bridge import iter_candidates_from_jsonl
from python.publishing.puzzle_catalog.pattern_linker import load_pattern_lookup
from python.publishing.puzzle_catalog.puzzle_record_builder import build_puzzle_record
from python.publishing.puzzle_catalog.solution_signature import build_solution_signature
from python.publishing.qc.validate_puzzle_record import validate_puzzle_record


DEFAULT_LIBRARY_SHORT = "CL9"
DEFAULT_LIBRARY_ID = build_library_id(DEFAULT_LIBRARY_SHORT)
DEFAULT_AISLE_SHORT = "DW"
DEFAULT_AISLE_ID = build_aisle_id("DWEIGHT")


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _log(message: str) -> None:
    print(message, flush=True)


def _write_ingest_report(
    *,
    output_dir: Path,
    timestamp: str,
    total_candidates: int,
    valid_records: int,
    duplicate_blocked: list[dict],
    validation_failures: list[dict],
) -> Path:
    report = {
        "schema_version": 1,
        "timestamp": timestamp,
        "total_candidates": total_candidates,
        "valid_records": valid_records,
        "duplicate_blocked_count": len(duplicate_blocked),
        "validation_failure_count": len(validation_failures),
        "duplicate_blocked": duplicate_blocked,
        "validation_failures": validation_failures,
    }
    path = output_dir / "_last_ingest_report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build canonical classic9 puzzle records from normalized generator JSONL input."
    )
    parser.add_argument(
        "--input-jsonl",
        required=True,
        help="Path to normalized generator JSONL input.",
    )
    parser.add_argument(
        "--patterns-dir",
        default="datasets/sudoku_books/classic9/patterns",
        help="Path to canonical pattern assets directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/sudoku_books/classic9/puzzle_records",
        help="Where puzzle record JSON files should be written.",
    )
    parser.add_argument(
        "--library-id",
        default=DEFAULT_LIBRARY_ID,
        help="Canonical library id.",
    )
    parser.add_argument(
        "--library-short",
        default=DEFAULT_LIBRARY_SHORT,
        help="Short library code used in record ids.",
    )
    parser.add_argument(
        "--aisle-id",
        default=DEFAULT_AISLE_ID,
        help="Legacy argument retained for backward compatibility. Ignored for raw catalog generation.",
    )
    parser.add_argument(
        "--aisle-short",
        default=DEFAULT_AISLE_SHORT,
        help="Legacy argument retained for backward compatibility. Ignored for raw catalog generation.",
    )
    parser.add_argument(
        "--book-number",
        type=int,
        default=1,
        help="Legacy argument retained for backward compatibility. Ignored for raw catalog generation.",
    )
    parser.add_argument(
        "--section-code",
        default="L1",
        help="Legacy argument retained for backward compatibility. Ignored for raw catalog generation.",
    )
    parser.add_argument(
        "--layout-type",
        default="classic9x9",
        help="Layout type.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=9,
        help="Grid size.",
    )
    parser.add_argument(
        "--charset",
        default="123456789",
        help="Character set for this library.",
    )
    parser.add_argument(
        "--verbose-candidates",
        action="store_true",
        help="Print one detailed line per candidate processed.",
    )
    parser.add_argument(
        "--allow-missing-pattern-registry",
        action="store_true",
        help="Continue even if patterns/registry.json does not exist. Pattern linkage will be skipped.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    input_jsonl = Path(args.input_jsonl)
    patterns_dir = Path(args.patterns_dir)
    output_dir = Path(args.output_dir)
    registry_path = patterns_dir / "registry.json"

    _log("=" * 72)
    _log("generate_classic9_puzzles.py starting")
    _log("=" * 72)
    _log(f"UTC timestamp:         {_now_iso()}")
    _log(f"Input JSONL:           {input_jsonl.resolve()}")
    _log(f"Patterns dir:          {patterns_dir.resolve()}")
    _log(f"Pattern registry:      {registry_path.resolve()}")
    _log(f"Output dir:            {output_dir.resolve()}")
    _log(f"Library id:            {args.library_id}")
    _log(f"Library short:         {args.library_short}")
    _log(f"Aisle id (legacy):     {args.aisle_id}")
    _log(f"Aisle short (legacy):  {args.aisle_short}")
    _log(f"Book number (legacy):  {args.book_number}")
    _log(f"Section code (legacy): {args.section_code}")
    _log(f"Layout type:           {args.layout_type}")
    _log(f"Grid size:             {args.grid_size}")
    _log(f"Charset:               {args.charset}")
    _log(f"Allow missing registry: {args.allow_missing_pattern_registry}")
    _log("=" * 72)

    if not input_jsonl.exists():
        _log(f"ERROR: input JSONL file does not exist: {input_jsonl}")
        return 1

    if not patterns_dir.exists():
        _log(f"ERROR: patterns directory does not exist: {patterns_dir}")
        return 1

    pattern_lookup = None

    if registry_path.exists():
        try:
            _log("Loading pattern registry...")
            pattern_lookup = load_pattern_lookup(patterns_dir)
            _log(
                f"Pattern registry loaded successfully: "
                f"{len(pattern_lookup.by_id)} patterns, {len(pattern_lookup.by_mask)} masks"
            )
        except Exception as exc:
            _log(f"ERROR while loading pattern registry: {exc}")
            _log(traceback.format_exc())
            return 1
    else:
        if args.allow_missing_pattern_registry:
            _log("WARNING: pattern registry does not exist.")
            _log("Pattern linkage will be skipped for this run.")
            _log("Expected file:")
            _log(f"  {registry_path}")
        else:
            _log(f"ERROR: pattern registry does not exist: {registry_path}")
            _log("You have two options:")
            _log("  1) Run the pattern ingestion workflow first to create pattern assets and registry.json")
            _log("  2) Re-run this command with --allow-missing-pattern-registry")
            return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = _now_iso()

    try:
        catalog_index = load_catalog_index(output_dir)
    except Exception as exc:
        _log(f"ERROR while loading catalog index: {exc}")
        _log(traceback.format_exc())
        return 1

    records = []
    validation_failures: list[dict] = []
    duplicate_blocked: list[dict] = []
    total_candidates = 0
    direct_pattern_id_count = 0

    pending_signature_to_record_id: dict[str, str] = {}

    try:
        _log("Beginning candidate scan...")
        for input_ordinal, candidate in enumerate(iter_candidates_from_jsonl(input_jsonl), start=1):
            total_candidates += 1

            if getattr(candidate, "pattern_id", None):
                direct_pattern_id_count += 1



            if not bool(candidate.is_unique):
                validation_failures.append(
                    {
                        "input_ordinal": input_ordinal,
                        "record_id": None,
                        "errors": ["Generator candidate has is_unique=False"],
                    }
                )
                _log(f"[candidate {input_ordinal}] SKIPPED non-unique candidate")
                continue

            if not bool(candidate.is_human_solvable):
                validation_failures.append(
                    {
                        "input_ordinal": input_ordinal,
                        "record_id": None,
                        "errors": ["Generator candidate has is_human_solvable=False"],
                    }
                )
                _log(f"[candidate {input_ordinal}] SKIPPED non-human-solvable candidate")
                continue






            if args.verbose_candidates:
                _log(
                    f"[candidate {input_ordinal}] "
                    f"weight={candidate.weight} "
                    f"givens_len={len(candidate.givens81)} "
                    f"solution_len={len(candidate.solution81)} "
                    f"pattern_id={getattr(candidate, 'pattern_id', None)} "
                    f"pattern_mask_present={'yes' if candidate.pattern_mask81 else 'no'}"
                )

            try:
                solution_signature = build_solution_signature(candidate.solution81)
            except Exception as exc:
                validation_failures.append(
                    {
                        "input_ordinal": input_ordinal,
                        "record_id": None,
                        "errors": [f"Failed to build solution_signature: {exc}"],
                    }
                )
                _log(f"[candidate {input_ordinal}] FAILED signature canonicalization")
                _log(f"    * Failed to build solution_signature: {exc}")
                continue

            duplicate_of = find_duplicate_record_id(
                index=catalog_index,
                pending_signature_to_record_id=pending_signature_to_record_id,
                solution_signature=solution_signature,
            )
            if duplicate_of is not None:
                duplicate_event = {
                    "input_ordinal": input_ordinal,
                    "candidate_status": "duplicate_blocked",
                    "duplicate_of_record_id": duplicate_of,
                    "solution_signature": solution_signature,
                    "title": candidate.title,
                    "generation_seed": candidate.generation_seed,
                    "weight": candidate.weight,
                    "pattern_id": getattr(candidate, "pattern_id", None),
                }
                duplicate_blocked.append(duplicate_event)
                _log(
                    f"[candidate {input_ordinal}] DUPLICATE BLOCKED -> existing {duplicate_of}"
                )
                continue

            record_ordinal = reserve_next_record_ordinal(catalog_index)

            record = build_puzzle_record(
                candidate=candidate,
                library_id=args.library_id,
                layout_short=args.library_short,
                ordinal=record_ordinal,
                layout_type=args.layout_type,
                grid_size=args.grid_size,
                charset=args.charset,
                pattern_lookup=pattern_lookup,
                created_at=timestamp,
                updated_at=timestamp,
            )

            errors = validate_puzzle_record(record)
            if errors:
                validation_failures.append(
                    {
                        "input_ordinal": input_ordinal,
                        "record_id": record.record_id,
                        "errors": errors,
                    }
                )
                _log(f"[candidate {input_ordinal}] FAILED validation -> {record.record_id}")
                for error in errors:
                    _log(f"    * {error}")
                continue

            pending_signature_to_record_id[record.solution_signature] = record.record_id
            records.append(record)
            _log(
                f"[candidate {input_ordinal}] OK -> {record.record_id} "
                f"(difficulty={record.difficulty_label}, "
                f"weight={record.weight}, "
                f"pattern_id={record.pattern_id})"
            )

    except Exception as exc:
        _log(f"ERROR while processing candidates: {exc}")
        _log(traceback.format_exc())
        return 1

    _log("-" * 72)
    _log(f"Total candidates read:       {total_candidates}")
    _log(f"Candidates with direct pattern_id: {direct_pattern_id_count}")
    _log(f"Valid puzzle records:        {len(records)}")
    _log(f"Duplicate blocked:           {len(duplicate_blocked)}")
    _log(f"Validation failures:         {len(validation_failures)}")
    _log("-" * 72)

    if total_candidates == 0:
        _log("WARNING: No candidates were found in the input JSONL.")
        _log("Check that the file exists, is non-empty, and has one valid JSON object per line.")
        return 1

    try:
        _log("Writing puzzle record files...")
        written = save_puzzle_records_batch(records, output_dir)
    except Exception as exc:
        _log(f"ERROR while saving puzzle records: {exc}")
        _log(traceback.format_exc())
        return 1

    try:
        for record, path in zip(records, written):
            register_record(
                catalog_index,
                record,
                relative_path=path.name,
            )
        index_path = save_catalog_index(catalog_index, output_dir)
    except Exception as exc:
        _log(f"ERROR while updating catalog index: {exc}")
        _log(traceback.format_exc())
        return 1

    try:
        report_path = _write_ingest_report(
            output_dir=output_dir,
            timestamp=timestamp,
            total_candidates=total_candidates,
            valid_records=len(records),
            duplicate_blocked=duplicate_blocked,
            validation_failures=validation_failures,
        )
    except Exception as exc:
        _log(f"ERROR while writing ingest report: {exc}")
        _log(traceback.format_exc())
        return 1

    _log(f"Written puzzle record files:  {len(written)}")
    for path in written[:10]:
        _log(f"  + {path}")
    if len(written) > 10:
        _log(f"  ... and {len(written) - 10} more")

    _log(f"Catalog index written:        {index_path}")
    _log(f"Ingest report written:        {report_path}")

    if duplicate_blocked:
        _log("-" * 72)
        _log("Duplicate-blocked candidates:")
        for item in duplicate_blocked[:10]:
            _log(
                f"  - input={item['input_ordinal']} "
                f"duplicate_of={item['duplicate_of_record_id']}"
            )
        if len(duplicate_blocked) > 10:
            _log(f"  ... and {len(duplicate_blocked) - 10} more")

    if validation_failures:
        _log("-" * 72)
        _log("Puzzle validation failures:")
        for item in validation_failures:
            _log(f"  - {item['record_id']}")
            for error in item["errors"]:
                _log(f"      * {error}")

    _log("=" * 72)
    _log("generate_classic9_puzzles.py completed successfully")
    _log("=" * 72)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FATAL ERROR: {exc}", flush=True)
        print(traceback.format_exc(), flush=True)
        raise