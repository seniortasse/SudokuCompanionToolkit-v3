from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from python.publishing.workflows.publication_package_reader import (
    load_publication_artifacts,
    normalize_page_type,
)


def _log(message: str) -> None:
    print(message, flush=True)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_page_type_token(token: str) -> str:
    token = str(token or "").strip()
    if not token:
        return ""
    if token.lower().endswith("_page"):
        return normalize_page_type(token)
    return normalize_page_type(f"{token}")


def _ordered_unique_section_ids_from_puzzle_pages(page_blocks: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    seen = set()
    for block in page_blocks:
        if normalize_page_type(block.get("page_type")) != "puzzle_page":
            continue
        section_id = str(block.get("section_id") or "").strip()
        if not section_id or section_id in seen:
            continue
        seen.add(section_id)
        out.append(section_id)
    return out


def _page_type_counts(page_blocks: List[Dict[str, Any]]) -> Counter:
    return Counter(normalize_page_type(block.get("page_type")) for block in page_blocks)


def _find_section_block_prefix(page_blocks: List[Dict[str, Any]], section_id: str) -> List[str]:
    out: List[str] = []
    in_section = False

    for block in page_blocks:
        block_section_id = str(block.get("section_id") or "").strip()
        page_type = normalize_page_type(block.get("page_type"))

        if block_section_id != section_id:
            if in_section:
                break
            continue

        in_section = True
        if page_type == "puzzle_page":
            break

        out.append(page_type)

    return out


def _require_editorial_examples(
    page_blocks: List[Dict[str, Any]],
    *,
    page_type: str,
    editorial_key: str,
    errors: List[str],
) -> None:
    expected_type = normalize_page_type(page_type)
    matching = [b for b in page_blocks if normalize_page_type(b.get("page_type")) == expected_type]

    if not matching:
        errors.append(f"Missing required page type: {page_type}")
        return

    for idx, block in enumerate(matching, start=1):
        payload = dict(block.get("payload") or {})
        editorial_copy = dict(payload.get("editorial_copy") or {})
        subtree = dict(editorial_copy.get(editorial_key) or {})
        examples = subtree.get("examples")

        if not subtree:
            errors.append(
                f"{page_type} #{idx} is missing payload.editorial_copy.{editorial_key}"
            )
            continue

        if examples in (None, {}, []):
            errors.append(
                f"{page_type} #{idx} is missing payload.editorial_copy.{editorial_key}.examples"
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate that a built publication still matches the approved gold base publication."
    )
    parser.add_argument(
        "--publication-dir",
        required=True,
        help="Path to the built publication directory to validate.",
    )
    parser.add_argument(
        "--gold-spec",
        required=True,
        help="Path to the approved gold publication spec JSON file.",
    )
    parser.add_argument(
        "--expect-estimated-page-count",
        type=int,
        default=None,
        help="Optional exact expected estimated page count for the gold build.",
    )
    parser.add_argument(
        "--expect-publication-id",
        default=None,
        help="Optional exact publication id. If omitted, the gold spec publication_id is used.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    publication_dir = Path(args.publication_dir)
    gold_spec_path = Path(args.gold_spec)

    _log("=" * 72)
    _log("check_base_publication_gold.py starting")
    _log("=" * 72)
    _log(f"Publication dir: {publication_dir.resolve()}")
    _log(f"Gold spec:       {gold_spec_path.resolve()}")
    _log("=" * 72)

    if not publication_dir.exists():
        _log(f"ERROR: publication directory not found: {publication_dir}")
        return 1

    if not gold_spec_path.exists():
        _log(f"ERROR: gold spec not found: {gold_spec_path}")
        return 1

    try:
        artifacts = load_publication_artifacts(publication_dir)
        manifest = dict(artifacts.get("manifest") or {})
        interior = dict(artifacts.get("interior") or {})
        gold = _load_json(gold_spec_path)
    except Exception as exc:
        _log(f"ERROR: Unable to load artifacts/spec: {exc}")
        return 1

    errors: List[str] = []

    page_blocks = list(interior.get("page_blocks") or [])
    if not page_blocks:
        errors.append("interior_plan.json has no page_blocks")

    actual_publication_id = str(manifest.get("publication_id") or "")
    expected_publication_id = str(args.expect_publication_id or gold.get("publication_id") or "")
    if expected_publication_id and actual_publication_id != expected_publication_id:
        errors.append(
            f"Expected publication_id={expected_publication_id}, got {actual_publication_id}"
        )

    actual_layout = dict(manifest.get("layout_config") or {})
    expected_layout = dict(gold.get("layout_config") or {})

    for key in ["puzzles_per_page", "rows", "cols", "language", "font_family"]:
        actual = actual_layout.get(key)
        expected = expected_layout.get(key)
        if actual != expected:
            errors.append(f"layout_config.{key}: expected {expected!r}, got {actual!r}")

    for key in ["puzzle_page_template", "solution_page_template", "include_solutions"]:
        actual = manifest.get(key)
        expected = gold.get(key)
        if actual != expected:
            errors.append(f"{key}: expected {expected!r}, got {actual!r}")

    actual_front = list(manifest.get("front_matter_sequence") or [])
    expected_front = list(gold.get("front_matter_sequence") or [])
    if actual_front != expected_front:
        errors.append(
            f"front_matter_sequence mismatch: expected {expected_front!r}, got {actual_front!r}"
        )

    actual_prelude = list(manifest.get("section_prelude_sequence") or [])
    expected_prelude = list(gold.get("section_prelude_sequence") or [])
    if actual_prelude != expected_prelude:
        errors.append(
            f"section_prelude_sequence mismatch: expected {expected_prelude!r}, got {actual_prelude!r}"
        )

    if args.expect_estimated_page_count is not None:
        actual_page_count = int(manifest.get("estimated_page_count") or 0)
        if actual_page_count != int(args.expect_estimated_page_count):
            errors.append(
                f"estimated_page_count: expected {args.expect_estimated_page_count}, got {actual_page_count}"
            )

    normalized_page_types = [normalize_page_type(block.get("page_type")) for block in page_blocks]
    expected_prefix = ["title_page"] + [_normalize_page_type_token(x) for x in expected_front]
    actual_prefix = normalized_page_types[: len(expected_prefix)]
    if actual_prefix != expected_prefix:
        errors.append(
            f"Opening page sequence mismatch: expected {expected_prefix!r}, got {actual_prefix!r}"
        )

    counts = _page_type_counts(page_blocks)
    if counts["title_page"] != 1:
        errors.append(f"Expected exactly 1 title_page, got {counts['title_page']}")
    if counts["welcome_page"] != 1:
        errors.append(f"Expected exactly 1 welcome_page, got {counts['welcome_page']}")
    if counts["features_page"] != 1:
        errors.append(f"Expected exactly 1 features_page, got {counts['features_page']}")
    if counts["toc_page"] != 1:
        errors.append(f"Expected exactly 1 toc_page, got {counts['toc_page']}")
    if counts["rules_page"] != expected_front.count("RULES_PAGE"):
        errors.append(
            f"Expected {expected_front.count('RULES_PAGE')} rules_page blocks, got {counts['rules_page']}"
        )
    if counts["tutorial_page"] != expected_front.count("TUTORIAL_PAGE"):
        errors.append(
            f"Expected {expected_front.count('TUTORIAL_PAGE')} tutorial_page blocks, got {counts['tutorial_page']}"
        )
    if counts["warmup_page"] != expected_front.count("WARMUP_PAGE"):
        errors.append(
            f"Expected {expected_front.count('WARMUP_PAGE')} warmup_page blocks, got {counts['warmup_page']}"
        )

    section_ids = _ordered_unique_section_ids_from_puzzle_pages(page_blocks)
    if not section_ids:
        errors.append("No section ids discovered from puzzle pages")

    if section_ids:
        if counts["section_opener_page"] != len(section_ids):
            errors.append(
                f"Expected {len(section_ids)} section_opener_page blocks, got {counts['section_opener_page']}"
            )
        if counts["section_highlights_page"] != len(section_ids):
            errors.append(
                f"Expected {len(section_ids)} section_highlights_page blocks, got {counts['section_highlights_page']}"
            )
        if counts["section_pattern_gallery_page"] < len(section_ids):
            errors.append(
                "Expected at least one section_pattern_gallery_page per section, "
                f"got {counts['section_pattern_gallery_page']} across {len(section_ids)} sections"
            )

        expected_section_prefix = [_normalize_page_type_token(x) for x in expected_prelude]
        for section_id in section_ids:
            actual_section_prefix = _find_section_block_prefix(page_blocks, section_id)
            actual_prefix_trimmed = actual_section_prefix[: len(expected_section_prefix)]
            if actual_prefix_trimmed != expected_section_prefix:
                errors.append(
                    f"Section {section_id} prelude mismatch: expected prefix "
                    f"{expected_section_prefix!r}, got {actual_prefix_trimmed!r}"
                )

    if counts["puzzle_page"] == 0:
        errors.append("Expected at least one puzzle_page, found none")

    if bool(gold.get("include_solutions", True)) and counts["solution_page"] == 0:
        errors.append("Gold spec includes solutions, but no solution_page blocks were found")

    if counts["promo_page"] < 1:
        errors.append("Expected at least one promo_page at the end of the publication")

    _require_editorial_examples(
        page_blocks,
        page_type="rules_page",
        editorial_key="rules",
        errors=errors,
    )
    _require_editorial_examples(
        page_blocks,
        page_type="tutorial_page",
        editorial_key="tutorial",
        errors=errors,
    )
    _require_editorial_examples(
        page_blocks,
        page_type="warmup_page",
        editorial_key="warmup",
        errors=errors,
    )

    if errors:
        _log("GOLD CHECK FAILED")
        _log("-" * 72)
        for err in errors:
            _log(f"- {err}")
        _log("-" * 72)
        return 1

    _log("GOLD CHECK PASSED")
    _log("=" * 72)
    _log("check_base_publication_gold.py completed successfully")
    _log("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())