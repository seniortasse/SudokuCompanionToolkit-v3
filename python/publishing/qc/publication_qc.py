from __future__ import annotations

from typing import Any, Dict, List


def run_publication_qc(artifacts: Dict[str, Any]) -> List[str]:
    errors: List[str] = []

    manifest = dict(artifacts.get("publication_manifest") or {})
    interior = dict(artifacts.get("interior_plan") or {})
    page_blocks = list(interior.get("page_blocks") or [])

    errors.extend(_check_page_numbering(page_blocks))
    errors.extend(_check_blank_page_number_visibility(page_blocks))
    errors.extend(_check_toc_entries(page_blocks))
    errors.extend(_check_manifest_layout_consistency(manifest, page_blocks))
    errors.extend(_check_solution_presence_consistency(manifest, page_blocks))

    return errors


def _check_page_numbering(page_blocks: List[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []

    for idx, block in enumerate(page_blocks, start=1):
        page_index = block.get("page_index")
        physical_page_number = block.get("physical_page_number")
        logical_page_number = block.get("logical_page_number")

        if page_index != idx:
            errors.append(
                f"Page block #{idx} has page_index={page_index}, expected {idx}"
            )

        if physical_page_number != idx:
            errors.append(
                f"Page block #{idx} has physical_page_number={physical_page_number}, expected {idx}"
            )

        if logical_page_number in (None, 0):
            errors.append(
                f"Page block #{idx} is missing logical_page_number"
            )

    return errors


def _check_blank_page_number_visibility(page_blocks: List[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []

    for idx, block in enumerate(page_blocks, start=1):
        page_type = str(block.get("page_type") or "").upper()
        show_page_number = bool(block.get("show_page_number", False))

        if page_type == "BLANK_PAGE" and show_page_number:
            errors.append(
                f"Blank page #{idx} incorrectly shows a page number"
            )

        if page_type != "BLANK_PAGE" and block.get("logical_page_number") not in (None, 0):
            if not show_page_number:
                errors.append(
                    f"Non-blank page #{idx} suppresses page number unexpectedly"
                )

    return errors


def _check_toc_entries(page_blocks: List[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []

    toc_blocks = [b for b in page_blocks if str(b.get("page_type") or "").upper() == "TOC_PAGE"]
    if not toc_blocks:
        return errors

    toc = toc_blocks[0]
    entries = list((toc.get("payload") or {}).get("entries") or [])
    existing_logical_pages = {
        int(b["logical_page_number"])
        for b in page_blocks
        if b.get("logical_page_number") not in (None, 0)
    }

    for entry in entries:
        title = str(entry.get("title") or "").strip()
        page_index = entry.get("page_index")
        if not title:
            errors.append("TOC contains an entry with blank title")
            continue
        if page_index in (None, 0):
            errors.append(f"TOC entry '{title}' is missing page_index")
            continue
        if int(page_index) not in existing_logical_pages:
            errors.append(
                f"TOC entry '{title}' points to missing logical page {page_index}"
            )

    return errors


def _check_manifest_layout_consistency(
    manifest: Dict[str, Any],
    page_blocks: List[Dict[str, Any]],
) -> List[str]:
    errors: List[str] = []

    layout = dict(manifest.get("layout_config") or {})
    puzzles_per_page = layout.get("puzzles_per_page")
    rows = layout.get("rows")
    cols = layout.get("cols")

    if puzzles_per_page is not None and rows is not None and cols is not None:
        if int(rows) * int(cols) != int(puzzles_per_page):
            errors.append(
                "publication_manifest.layout_config rows*cols != puzzles_per_page"
            )

    for block in page_blocks:
        page_type = str(block.get("page_type") or "").upper()
        if page_type not in {"PUZZLE_PAGE", "SOLUTION_PAGE"}:
            continue

        puzzle_ids = list((block.get("payload") or {}).get("puzzle_ids") or [])
        if puzzles_per_page is not None and len(puzzle_ids) > int(puzzles_per_page):
            errors.append(
                f"{page_type} exceeds puzzles_per_page: {len(puzzle_ids)} > {puzzles_per_page}"
            )

    return errors


def _check_solution_presence_consistency(
    manifest: Dict[str, Any],
    page_blocks: List[Dict[str, Any]],
) -> List[str]:
    errors: List[str] = []

    include_solutions = bool(manifest.get("include_solutions", True))
    has_solution_pages = any(
        str(b.get("page_type") or "").upper() == "SOLUTION_PAGE"
        for b in page_blocks
    )

    if include_solutions and not has_solution_pages:
        errors.append("Manifest requests solutions but interior plan has no solution pages")

    if not include_solutions and has_solution_pages:
        errors.append("Manifest disables solutions but interior plan still contains solution pages")

    return errors