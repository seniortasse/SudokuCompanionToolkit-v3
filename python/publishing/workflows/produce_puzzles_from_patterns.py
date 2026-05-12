from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

from python.publishing.pattern_library.pattern_stats import record_production_outcomes
from python.publishing.pattern_library.pattern_store import (
    load_pattern_store,
    rebuild_compiled_pattern_artifacts,
)
from python.publishing.puzzle_catalog.pattern_production import (
    build_production_requests,
    parse_legacy_output_workbook,
    run_legacy_pattern_generator,
    select_patterns_from_catalog,
    write_candidates_jsonl,
    write_pattern_requests_workbook,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


_PATTERN_ID_RE = re.compile(r"^(PAT-[A-Z0-9]+-\d+)$", re.IGNORECASE)


def _dedupe_keep_order(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in values:
        value = str(raw).strip()
        if not value:
            continue
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _pattern_ids_from_folders(folder_paths: Iterable[str]) -> List[str]:
    discovered: List[str] = []

    for raw_folder in folder_paths:
        folder = Path(str(raw_folder)).expanduser()
        if not folder.exists():
            raise FileNotFoundError(f"Pattern folder does not exist: {folder}")
        if not folder.is_dir():
            raise NotADirectoryError(f"Pattern folder is not a directory: {folder}")

        for path in sorted(folder.rglob("*")):
            if not path.is_file():
                continue
            match = _PATTERN_ID_RE.match(path.stem.strip())
            if match:
                discovered.append(match.group(1).upper())

    return _dedupe_keep_order(discovered)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Produce generator candidates from selected catalog patterns using the legacy pattern generator."
    )
    parser.add_argument(
        "--patterns-dir",
        default="datasets/sudoku_books/classic9/patterns",
        help="Canonical pattern catalog directory.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="runs/publishing/classic9/puzzle_generation/candidates.jsonl",
        help="Destination JSONL for generated candidates.",
    )
    parser.add_argument(
        "--generator-root",
        default="python/puzzle_generator/pattern_sudoku_all_sizes",
        help="Root directory of the legacy pattern generator.",
    )
    parser.add_argument(
        "--count",
        type=int,
        required=True,
        help="Total number of production requests to issue.",
    )
    parser.add_argument(
        "--pattern-id",
        action="append",
        default=[],
        help="Explicit pattern id to include. Can be provided multiple times.",
    )

    parser.add_argument(
        "--pattern-folder",
        action="append",
        default=[],
        help=(
            "Folder containing files named with canonical pattern ids "
            "(for example PAT-CL9-0248.png). All matching pattern ids found "
            "in the folder tree will be included. Can be provided multiple times."
        ),
    )




    parser.add_argument(
        "--family-id",
        action="append",
        default=[],
        help="Pattern family id to include. Can be provided multiple times.",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        help="Include patterns matching any supplied tag. Can be provided multiple times.",
    )
    parser.add_argument(
        "--max-patterns",
        type=int,
        default=None,
        help="Optional cap on the number of selected seed patterns before cycling.",
    )
    parser.add_argument(
        "--min-clue-count",
        type=int,
        default=None,
        help="Optional minimum clue count for pattern selection.",
    )
    parser.add_argument(
        "--max-clue-count",
        type=int,
        default=None,
        help="Optional maximum clue count for pattern selection.",
    )
    parser.add_argument(
        "--min-aesthetic-score",
        type=float,
        default=None,
        help="Optional minimum aesthetic score for pattern selection.",
    )
    parser.add_argument(
        "--min-print-score",
        type=float,
        default=None,
        help="Optional minimum print score for pattern selection.",
    )
    parser.add_argument(
        "--min-legibility-score",
        type=float,
        default=None,
        help="Optional minimum legibility score for pattern selection.",
    )
    parser.add_argument(
        "--charset",
        default="123456789",
        help='Character set passed to the legacy generator. Default: "123456789"',
    )
    parser.add_argument(
        "--weight-min",
        type=int,
        default=None,
        help="Optional minimum target weight for the legacy generator.",
    )
    parser.add_argument(
        "--weight-max",
        type=int,
        default=None,
        help="Optional maximum target weight for the legacy generator.",
    )
    parser.add_argument(
        "--technique-count-min",
        type=int,
        default=None,
        help="Optional minimum number of distinct techniques.",
    )
    parser.add_argument(
        "--technique-count-max",
        type=int,
        default=None,
        help="Optional maximum number of distinct techniques.",
    )
    parser.add_argument(
        "--required-technique",
        action="append",
        default=[],
        help="Technique required by the legacy generator. Can be provided multiple times.",
    )
    parser.add_argument(
        "--excluded-technique",
        action="append",
        default=[],
        help="Technique excluded by the legacy generator. Can be provided multiple times.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to output JSONL instead of overwriting it.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    patterns_dir = Path(args.patterns_dir)
    output_jsonl = Path(args.output_jsonl)
    generator_root = Path(args.generator_root)

    registry = load_pattern_store(patterns_dir)

    folder_pattern_ids = _pattern_ids_from_folders(args.pattern_folder)
    requested_pattern_ids = _dedupe_keep_order(list(args.pattern_id) + list(folder_pattern_ids))

    selected_patterns = select_patterns_from_catalog(
        registry=registry,
        pattern_ids=requested_pattern_ids,
        family_ids=args.family_id,
        tags_any=args.tag,
        min_clue_count=args.min_clue_count,
        max_clue_count=args.max_clue_count,
        min_aesthetic_score=args.min_aesthetic_score,
        min_print_score=args.min_print_score,
        min_legibility_score=args.min_legibility_score,
        max_patterns=args.max_patterns,
    )

    if not selected_patterns:
        print("ERROR: no active patterns matched the requested filters.", flush=True)
        return 1
    

    print(f"Folder-discovered pattern ids: {len(folder_pattern_ids)}", flush=True)
    if folder_pattern_ids:
        print("From folders: " + ", ".join(folder_pattern_ids), flush=True)

    print(f"Selected patterns: {len(selected_patterns)}", flush=True)
    print(
        "Pattern ids: " + ", ".join(str(p.pattern_id) for p in selected_patterns),
        flush=True,
    )
    print(f"Planned requests: {args.count}", flush=True)
    print(f"Output JSONL: {output_jsonl}", flush=True)
    print(f"Generator root: {generator_root}", flush=True)
    print("", flush=True)

    requests = build_production_requests(
        patterns=selected_patterns,
        count=args.count,
    )

    run_dir = output_jsonl.parent
    run_dir.mkdir(parents=True, exist_ok=True)

    request_workbook = run_dir / "_last_pattern_generation_input.xlsx"
    write_pattern_requests_workbook(
        requests=requests,
        workbook_path=request_workbook,
    )

    request_lookup = {request.request_id: request.pattern for request in requests}

    output_workbook, stdout_text, stderr_text = run_legacy_pattern_generator(
        generator_root=generator_root,
        input_workbook=request_workbook,
        charset=args.charset,
        weight_min=args.weight_min,
        weight_max=args.weight_max,
        technique_count_min=args.technique_count_min,
        technique_count_max=args.technique_count_max,
        required_techniques=args.required_technique,
        excluded_techniques=args.excluded_technique,
    )

    candidates, rejected = parse_legacy_output_workbook(
        output_workbook=output_workbook,
        request_lookup=request_lookup,
    )

    write_candidates_jsonl(
        candidates=candidates,
        output_jsonl=output_jsonl,
        append=args.append,
    )

    stats_summary = record_production_outcomes(
        registry=registry,
        requests=requests,
        candidates=candidates,
        rejected=rejected,
        run_id="CLI-PATTERN-PRODUCTION",
        timestamp=_now_iso(),
    )
    artifact_paths = rebuild_compiled_pattern_artifacts(registry, patterns_dir)

    report = {
        "timestamp": _now_iso(),
        "patterns_dir": str(patterns_dir),
        "generator_root": str(generator_root),
        "output_jsonl": str(output_jsonl),
        "request_workbook": str(request_workbook),
        "output_workbook": str(output_workbook),
        "selected_pattern_count": len(selected_patterns),
        "request_count": len(requests),
        "candidate_count": len(candidates),
        "rejected_count": len(rejected),
        "selected_pattern_ids": [p.pattern_id for p in selected_patterns],
        "pattern_stats_updated": int(stats_summary["updated_patterns"]),
        "pattern_stats_touched_ids": list(stats_summary["touched_pattern_ids"]),
        "catalog_path": str(patterns_dir / "pattern_catalog.jsonl"),
        "registry_path": str(artifact_paths["registry"]),
    }
    report_path = run_dir / "_last_pattern_production_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    stdout_path = run_dir / "_last_pattern_production_stdout.txt"
    stderr_path = run_dir / "_last_pattern_production_stderr.txt"
    rejected_path = run_dir / "_last_pattern_production_rejected.json"

    stdout_path.write_text(stdout_text, encoding="utf-8")
    stderr_path.write_text(stderr_text, encoding="utf-8")
    rejected_path.write_text(json.dumps(rejected, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Selected patterns:   {len(selected_patterns)}", flush=True)
    print(f"Production requests: {len(requests)}", flush=True)
    print(f"Candidates written:  {len(candidates)}", flush=True)
    print(f"Rejected requests:   {len(rejected)}", flush=True)
    print(f"Pattern stats updated: {stats_summary['updated_patterns']}", flush=True)
    print(f"Candidates JSONL:    {output_jsonl}", flush=True)
    print(f"Input workbook:      {request_workbook}", flush=True)
    print(f"Output workbook:     {output_workbook}", flush=True)
    print(f"Catalog path:        {patterns_dir / 'pattern_catalog.jsonl'}", flush=True)
    print(f"Registry path:       {artifact_paths['registry']}", flush=True)
    print(f"Index by id:         {artifact_paths['by_id']}", flush=True)
    print(f"Index by mask:       {artifact_paths['by_mask']}", flush=True)
    print(f"Index by family:     {artifact_paths['by_family']}", flush=True)
    print(f"Run report:          {report_path}", flush=True)
    print(f"STDOUT log:          {stdout_path}", flush=True)
    print(f"STDERR log:          {stderr_path}", flush=True)
    print(f"Rejected log:        {rejected_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())