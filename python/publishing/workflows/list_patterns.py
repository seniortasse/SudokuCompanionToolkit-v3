from __future__ import annotations

import argparse
from pathlib import Path

from python.publishing.pattern_library.pattern_filters import filter_patterns
from python.publishing.pattern_library.pattern_store import load_pattern_store


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List patterns from the canonical pattern store.")
    parser.add_argument("--patterns-dir", default="datasets/sudoku_books/classic9/patterns")
    parser.add_argument("--status", default=None)
    parser.add_argument("--family", default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--min-clue-count", type=int, default=None)
    parser.add_argument("--max-clue-count", type=int, default=None)
    parser.add_argument("--min-aesthetic-score", type=float, default=None)
    parser.add_argument("--min-print-score", type=float, default=None)
    parser.add_argument("--min-legibility-score", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    patterns_dir = Path(args.patterns_dir)
    registry = load_pattern_store(patterns_dir)

    patterns = filter_patterns(
        registry.patterns,
        status=args.status,
        family_id=args.family,
        tag=args.tag,
        min_clue_count=args.min_clue_count,
        max_clue_count=args.max_clue_count,
        min_aesthetic_score=args.min_aesthetic_score,
        min_print_score=args.min_print_score,
        min_legibility_score=args.min_legibility_score,
    )

    print(f"Library: {registry.library_id}", flush=True)
    print(f"Pattern count: {len(patterns)}", flush=True)

    for pattern in patterns:
        stats = dict(pattern.production_stats or {})
        attempts = stats.get("generation_attempts")
        successes = stats.get("successful_candidates")
        success_rate = stats.get("success_rate")
        avg_weight = stats.get("avg_weight")

        print(
            " | ".join(
                [
                    str(pattern.pattern_id),
                    str(pattern.name),
                    f"family={pattern.family_id or '-'}",
                    f"status={pattern.status}",
                    f"clues={pattern.clue_count}",
                    f"aesthetic={pattern.aesthetic_score if pattern.aesthetic_score is not None else '-'}",
                    f"print={pattern.print_score if pattern.print_score is not None else '-'}",
                    f"legibility={pattern.legibility_score if pattern.legibility_score is not None else '-'}",
                    f"attempts={attempts if attempts is not None else '-'}",
                    f"successes={successes if successes is not None else '-'}",
                    f"success_rate={success_rate if success_rate is not None else '-'}",
                    f"avg_weight={avg_weight if avg_weight is not None else '-'}",
                    f"tags={','.join(pattern.tags or [])}",
                ]
            ),
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())