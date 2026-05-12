from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from python.publishing.pattern_library.pattern_preview import render_pattern_previews
from python.publishing.pattern_library.pattern_store import (
    load_pattern_store,
    rebuild_compiled_pattern_artifacts,
)


def _symmetry_ratio(mask81: str) -> float:
    h_matches = 0
    v_matches = 0
    total = 81

    rows = [mask81[i:i + 9] for i in range(0, 81, 9)]
    mirror_h = [row[::-1] for row in rows]
    mirror_v = rows[::-1]

    flat_h = "".join(mirror_h)
    flat_v = "".join(mirror_v)

    for a, b in zip(mask81, flat_h):
        if a == b:
            h_matches += 1
    for a, b in zip(mask81, flat_v):
        if a == b:
            v_matches += 1

    return max(h_matches / total, v_matches / total)


def _distribution_balance(mask81: str) -> float:
    rows = [mask81[i:i + 9] for i in range(0, 81, 9)]
    row_counts = [row.count("1") for row in rows]
    col_counts = [sum(1 for r in range(9) if rows[r][c] == "1") for c in range(9)]

    row_span = max(row_counts) - min(row_counts)
    col_span = max(col_counts) - min(col_counts)

    penalty = min(1.0, (row_span + col_span) / 18.0)
    return 1.0 - penalty


def _center_presence(mask81: str) -> float:
    center_indexes = [30, 31, 32, 39, 40, 41, 48, 49, 50]
    hits = sum(1 for idx in center_indexes if mask81[idx] == "1")
    return hits / 9.0


def _density_score(mask81: str) -> float:
    clue_count = mask81.count("1")
    target = 37
    distance = abs(clue_count - target)
    score = max(0.0, 1.0 - (distance / 25.0))
    return score


def _compute_scores(mask81: str) -> dict:
    symmetry = _symmetry_ratio(mask81)
    balance = _distribution_balance(mask81)
    center = _center_presence(mask81)
    density = _density_score(mask81)

    print_score = round((0.45 * density) + (0.35 * balance) + (0.20 * symmetry), 3)
    legibility_score = round((0.40 * balance) + (0.35 * density) + (0.25 * center), 3)
    aesthetic_score = round((0.35 * symmetry) + (0.30 * balance) + (0.20 * center) + (0.15 * density), 3)

    return {
        "print_score": print_score,
        "legibility_score": legibility_score,
        "aesthetic_score": aesthetic_score,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute pattern scores and render previews."
    )
    parser.add_argument(
        "--patterns-dir",
        default="datasets/sudoku_books/classic9/patterns",
        help="Pattern catalog directory.",
    )
    parser.add_argument(
        "--status",
        default=None,
        help="Optional status filter, e.g. active.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    patterns_dir = Path(args.patterns_dir)
    previews_dir = patterns_dir / "previews"

    registry = load_pattern_store(patterns_dir)

    target_patterns: List = []
    for pattern in registry.patterns:
        if args.status and str(pattern.status) != str(args.status):
            continue
        target_patterns.append(pattern)

    for pattern in target_patterns:
        scores = _compute_scores(pattern.mask81)
        pattern.print_score = scores["print_score"]
        pattern.legibility_score = scores["legibility_score"]
        pattern.aesthetic_score = scores["aesthetic_score"]

    written_previews = render_pattern_previews(target_patterns, previews_dir)
    paths = rebuild_compiled_pattern_artifacts(registry, patterns_dir)

    print(f"Scored patterns:   {len(target_patterns)}", flush=True)
    print(f"Rendered previews: {len(written_previews)}", flush=True)
    print(f"Catalog path:      {patterns_dir / 'pattern_catalog.jsonl'}", flush=True)
    print(f"Registry path:     {paths['registry']}", flush=True)
    print(f"Index by id:       {paths['by_id']}", flush=True)
    print(f"Index by mask:     {paths['by_mask']}", flush=True)
    print(f"Index by family:   {paths['by_family']}", flush=True)
    print(f"Previews dir:      {previews_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())