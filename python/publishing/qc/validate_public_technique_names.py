from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


# These are internal engine names / implementation labels that should not appear
# in customer-facing publication specs, compiled specs, manifests, metadata,
# cover specs, bundles, or rendered payloads.
#
# Keep this list focused on *leakage patterns*. It is okay for internal puzzle
# records and search indexes to contain engine IDs.
FORBIDDEN_PUBLIC_TECHNIQUE_PATTERNS: tuple[str, ...] = (
    r"\bsingles[-_]1\b",
    r"\bsingles[-_]2\b",
    r"\bsingles[-_]3\b",
    r"\bsingles[-_]naked[-_]2\b",
    r"\bsingles[-_]naked[-_]3\b",
    r"\bdoubles[-_]naked\b",
    r"\btriplets[-_]naked\b",
    r"\bquads[-_]naked\b",
    r"\bsingles[-_]pointing\b",
    r"\bsingles[-_]boxed\b",
    r"\bx[-_]wings[-_]3\b",
    r"\bx[-_]wings[-_]4\b",
    r"\bab[-_]rings\b",
    r"\bab[-_]chains\b",
    r"\bboxed[-_]doubles\b",
    r"\bboxed[-_]triplets\b",
    r"\bboxed[-_]quads\b",
    r"\bboxed[-_]wings\b",
    r"\bboxed[-_]rays\b",
    r"\bleftovers[-_][1-9]\b",
    r"\b3[- ]line x[- ]wings\b",
    r"\b4[- ]line x[- ]wings\b",
    r"\bY[- ]Wings\b",
)

DEFAULT_ALLOWED_FILENAMES: tuple[str, ...] = (
    "publication_manifest.json",
    "publication_package.json",
    "interior_plan.json",
    "cover_spec.json",
    "cover_manifest.json",
    "metadata.json",
    "kdp_profile.json",
    "bundle_summary.json",
    "asset_manifest.json",
)

DEFAULT_ALLOWED_SUFFIXES: tuple[str, ...] = (
    ".json",
    ".txt",
    ".md",
)

DEFAULT_EXCLUDED_DIR_NAMES: set[str] = {
    "__pycache__",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".venv",
    "venv",
    "puzzles",
    "puzzle_records",
    "search_indexes",
    "indexes",
}


@dataclass(frozen=True)
class PublicTechniqueNameViolation:
    path: str
    pattern: str
    line_number: int
    line: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "pattern": self.pattern,
            "line_number": self.line_number,
            "line": self.line,
        }


def _is_probably_public_file(path: Path) -> bool:
    if path.name in DEFAULT_ALLOWED_FILENAMES:
        return True

    if path.suffix.lower() not in DEFAULT_ALLOWED_SUFFIXES:
        return False

    # Locale/spec files are public-facing enough to validate.
    lower_name = path.name.lower()
    if lower_name.endswith(".json") and (
        "publication" in lower_name
        or "locale" in lower_name
        or "cover" in lower_name
        or "metadata" in lower_name
        or "bundle" in lower_name
        or "manifest" in lower_name
        or "spec" in lower_name
    ):
        return True

    return False


def _should_skip_path(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    return bool(parts.intersection(DEFAULT_EXCLUDED_DIR_NAMES))


def _iter_candidate_files(paths: Sequence[Path]) -> Iterable[Path]:
    for root in paths:
        root = Path(root)
        if not root.exists():
            continue

        if root.is_file():
            if not _should_skip_path(root) and _is_probably_public_file(root):
                yield root
            continue

        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if _should_skip_path(path):
                continue
            if not _is_probably_public_file(path):
                continue
            yield path


def _line_matches(line: str, compiled_patterns: Sequence[tuple[str, re.Pattern[str]]]) -> list[str]:
    matches: list[str] = []
    for raw_pattern, pattern in compiled_patterns:
        if pattern.search(line):
            matches.append(raw_pattern)
    return matches


def find_public_technique_name_violations(
    *,
    paths: Sequence[Path],
    forbidden_patterns: Sequence[str] = FORBIDDEN_PUBLIC_TECHNIQUE_PATTERNS,
) -> list[PublicTechniqueNameViolation]:
    compiled_patterns = [
        (raw, re.compile(raw, flags=re.IGNORECASE))
        for raw in forbidden_patterns
    ]

    violations: list[PublicTechniqueNameViolation] = []

    for path in _iter_candidate_files(paths):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        for line_number, line in enumerate(text.splitlines(), start=1):
            matched = _line_matches(line, compiled_patterns)
            if not matched:
                continue

            for pattern in matched:
                violations.append(
                    PublicTechniqueNameViolation(
                        path=str(path),
                        pattern=pattern,
                        line_number=line_number,
                        line=line.strip(),
                    )
                )

    return violations


def validate_public_technique_names(
    *,
    paths: Sequence[Path],
) -> dict[str, Any]:
    violations = find_public_technique_name_violations(paths=paths)
    return {
        "error_count": len(violations),
        "errors": [violation.to_dict() for violation in violations],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fail if internal Sudoku engine technique IDs leak into public-facing "
            "publication specs, manifests, metadata, or bundle files."
        )
    )
    parser.add_argument(
        "--path",
        action="append",
        required=True,
        help="File or directory to validate. Can be repeated.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write a JSON validation report.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    paths = [Path(value) for value in list(args.path or [])]
    report = validate_public_technique_names(paths=paths)

    output_json = str(args.output_json or "").strip()
    if output_json:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    errors = list(report.get("errors") or [])
    if errors:
        print("PUBLIC TECHNIQUE NAME VALIDATION FAILED")
        print("-" * 72)
        for error in errors[:80]:
            print(
                f"{error['path']}:{error['line_number']} "
                f"matched {error['pattern']}: {error['line']}"
            )
        if len(errors) > 80:
            print(f"... and {len(errors) - 80} more violation(s)")
        return 1

    print("PUBLIC TECHNIQUE NAME VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())