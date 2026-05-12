from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Dict, List, Tuple

from python.publishing.cover_builder import build_cover_manifest
from python.publishing.interior_builder import (
    apply_page_numbering,
    build_toc_entries,
    insert_required_blank_pages,
)
from python.publishing.i18n.strings import normalize_language, tr, translate_difficulty_label
from python.publishing.interior_templates import (
    resolve_end_matter_profile,
    resolve_front_matter_page_spec,
    resolve_front_matter_profile,
    resolve_section_prelude_page_spec,
)
from python.publishing.pdf_renderer.render_models import load_built_book_render_model
from python.publishing.print_specs import (
    compute_spine_width_in,
    get_print_format_spec,
    validate_print_format_spec,
    validate_publication_spec,
)
from python.publishing.publication_builder.publication_manifest_builder import build_publication_manifest
from python.publishing.publication_builder.publication_paths import get_publication_dir
from python.publishing.publication_builder.publication_spec_loader import load_publication_spec
from python.publishing.publication_builder.section_preview_payloads import (
    build_section_highlights_payload,
    build_section_pattern_gallery_payloads,
)
from python.publishing.schemas.models import CoverSpec, InteriorPlan, PageBlock, PublicationPackage
from python.publishing.schemas.page_types import (
    FEATURES_PAGE,
    PROMO_PAGE,
    PUZZLE_PAGE,
    RULES_PAGE,
    SECTION_HIGHLIGHTS_PAGE,
    SECTION_OPENER_PAGE,
    SECTION_PATTERN_GALLERY_PAGE,
    SOLUTION_PAGE,
    TITLE_PAGE,
    TOC_PAGE,
    TUTORIAL_PAGE,
    WARMUP_PAGE,
    WELCOME_PAGE,
)
from python.publishing.schemas.publication_io import write_json


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()



def _resolve_front_matter_items(publication_spec) -> List[dict]:
    explicit_sequence = list(publication_spec.front_matter_sequence or [])
    if explicit_sequence:
        return [
            resolve_front_matter_page_spec(page_type)
            for page_type in explicit_sequence
        ]
    return resolve_front_matter_profile(publication_spec.front_matter_profile)


def _resolve_section_prelude_items(publication_spec) -> List[dict]:
    explicit_sequence = list(publication_spec.section_prelude_sequence or [])
    if explicit_sequence:
        return [
            resolve_section_prelude_page_spec(page_type)
            for page_type in explicit_sequence
        ]

    if publication_spec.section_separator_policy == "section_openers":
        return [
            resolve_section_prelude_page_spec(SECTION_OPENER_PAGE),
        ]

    return []


def _section_key_for_manifest(section_manifest) -> str:
    """
    Resolve the semantic section key used by editorial_copy.sections.

    Do not infer difficulty from section_code. L1/L2/L3 are ordering labels,
    not stable difficulty identities. For example:

        B02: L1=medium, L2=hard,   L3=expert
        B03: L1=easy,   L2=medium, L3=hard
    """
    hint = str(section_manifest.difficulty_label_hint or "").strip().lower()
    title = str(section_manifest.title or "").strip().lower()

    for value in (hint, title):
        normalized = value.replace("_", " ").replace("-", " ").strip()
        if normalized in {"easy", "medium", "hard", "expert"}:
            return normalized

    code = str(section_manifest.section_code or "").strip().lower()
    return hint or title or code or "section"

def _short_book_code_from_book_id(book_id: str) -> str:
    """
    Convert BK-CL9-DW-B01 -> B01.

    Falls back to the raw book id if the expected trailing Bxx token
    is not present.
    """
    raw = str(book_id or "").strip()
    for part in reversed([p for p in raw.split("-") if p]):
        upper = part.upper()
        if upper.startswith("B") and upper[1:].isdigit():
            return upper
    return raw


def _locale_code_for_publication(publication_spec) -> str:
    metadata = dict(publication_spec.metadata or {})
    layout = publication_spec.layout_config

    raw = str(
        metadata.get("locale")
        or layout.language
        or layout.language_code
        or metadata.get("language")
        or "en"
    ).strip().lower()

    # Convert "English" / "German" labels when needed.
    language_to_locale = {
        "english": "en",
        "german": "de",
        "deutsch": "de",
        "french": "fr",
        "français": "fr",
        "francais": "fr",
        "italian": "it",
        "italiano": "it",
        "spanish": "es",
        "español": "es",
        "espanol": "es",
    }

    raw = language_to_locale.get(raw, raw)
    raw = raw.split("-")[0].split("_")[0]
    return raw or "en"


def _publication_book_code(book_manifest, publication_spec) -> str:
    """
    Public reader-facing book code, for support emails.

    Example:
        BK-CL9-DW-B01 + en -> B01-en
        BK-CL9-DW-B05 + de -> B05-de
    """
    return f"{_short_book_code_from_book_id(book_manifest.book_id)}-{_locale_code_for_publication(publication_spec)}"

def _resolve_publication_book_labels(book_manifest, publication_spec) -> dict:
    metadata = dict(publication_spec.metadata or {})
    kdp_listing = dict(metadata.get("kdp_listing") or {})

    description = str(
        metadata.get("description")
        or metadata.get("marketplace_description")
        or metadata.get("back_copy")
        or book_manifest.description
    ).strip()

    return {
        "title": str(
            kdp_listing.get("title")
            or metadata.get("title")
            or book_manifest.title
        ),
        "subtitle": str(
            kdp_listing.get("subtitle")
            or metadata.get("subtitle")
            or book_manifest.subtitle
        ),
        "series_name": str(metadata.get("series_name") or book_manifest.series_name),
        "description": description,
        "kdp_listing": kdp_listing,
    }


def _resolve_localized_section_labels(section, publication_spec) -> dict:
    editorial = dict(publication_spec.editorial_copy or {})
    sections = dict(editorial.get("sections") or {})
    section_key = _section_key_for_manifest(section.section_manifest)
    section_copy = dict(sections.get(section_key) or {})

    language = normalize_language(
        str(
            publication_spec.layout_config.language
            or publication_spec.layout_config.language_code
            or publication_spec.metadata.get("locale")
            or publication_spec.metadata.get("language")
            or "en"
        )
    )

    difficulty_hint = str(section.section_manifest.difficulty_label_hint or section_key or "").strip()

    localized_title = str(section_copy.get("title") or "").strip()
    if not localized_title:
        localized_title = translate_difficulty_label(difficulty_hint, language)

    if not localized_title:
        localized_title = translate_difficulty_label(str(section.section_manifest.title or ""), language)

    if not localized_title:
        localized_title = str(section.section_manifest.title or "").strip()

    localized_subtitle = str(section_copy.get("subtitle") or "").strip()

    if not localized_subtitle and section_key in {"easy", "medium", "hard", "expert"}:
        subtitle_key = f"section_subtitle_{section_key}"
        translated_subtitle = str(tr(subtitle_key, language) or "").strip()

        # Keep this defensive: if tr() ever returns the key itself for missing
        # translations, do not print "section_subtitle_easy" in the book.
        if translated_subtitle and translated_subtitle != subtitle_key:
            localized_subtitle = translated_subtitle

    # Last resort only. The built-book manifest is often English, so it should
    # not beat locale strings or i18n strings.
    if not localized_subtitle:
        localized_subtitle = str(section.section_manifest.subtitle or "").strip()

    return {
        "title": localized_title,
        "subtitle": localized_subtitle,
    }

def _append_profile_blocks(
    page_blocks: List[PageBlock],
    profile_items: List[dict],
    *,
    book_manifest,
    publication_spec=None,
    puzzles_per_page: int | None = None,
) -> None:
    page_type_totals = {}
    for item in profile_items:
        page_type = str(item["page_type"])
        page_type_totals[page_type] = int(page_type_totals.get(page_type, 0)) + 1

    page_type_seen = {}
    book_labels = _resolve_publication_book_labels(book_manifest, publication_spec) if publication_spec is not None else {
        "title": book_manifest.title,
        "subtitle": book_manifest.subtitle,
        "series_name": book_manifest.series_name,
        "description": book_manifest.description,
    }

    resolved_language = "en"
    if publication_spec is not None:
        resolved_language = str(
            publication_spec.layout_config.language
            or publication_spec.layout_config.language_code
            or publication_spec.metadata.get("locale")
            or publication_spec.metadata.get("language")
            or "en"
        )

    for item in profile_items:
        payload = dict(item.get("payload", {}))
        page_type = str(item["page_type"])

        page_type_seen[page_type] = int(page_type_seen.get(page_type, 0)) + 1
        payload.setdefault("page_occurrence_index", int(page_type_seen[page_type]))
        payload.setdefault("page_occurrence_total", int(page_type_totals.get(page_type, 1)))
        payload.setdefault("language", resolved_language)

        if page_type == WELCOME_PAGE:
            payload.setdefault("title", book_labels["title"])
            payload.setdefault("subtitle", book_labels["subtitle"])
            payload.setdefault("description", book_labels["description"])
            payload.setdefault("series_name", book_labels["series_name"])
            payload.setdefault("library_id", book_manifest.library_id)
            payload.setdefault("aisle_id", book_manifest.aisle_id)
            payload.setdefault("book_title", book_labels["title"])
            payload.setdefault("book_subtitle", book_labels["subtitle"])
            if publication_spec is not None:
                payload.setdefault("editorial_copy", dict(publication_spec.editorial_copy or {}))
                payload.setdefault("ecosystem_config", dict(publication_spec.ecosystem_config or {}))
                payload.setdefault("publication_metadata", dict(publication_spec.metadata or {}))

        if page_type == FEATURES_PAGE and publication_spec is not None:
            payload.setdefault("title", book_labels["title"])
            payload.setdefault("subtitle", book_labels["subtitle"])
            payload.setdefault("series_name", book_labels["series_name"])
            payload.setdefault("book_id", book_manifest.book_id)
            payload.setdefault("features_page_config", dict(publication_spec.features_page_config or {}))
            payload.setdefault("editorial_copy", dict(publication_spec.editorial_copy or {}))
            payload.setdefault("ecosystem_config", dict(publication_spec.ecosystem_config or {}))
            payload.setdefault("publication_metadata", dict(publication_spec.metadata or {}))

        if page_type in {RULES_PAGE, TUTORIAL_PAGE, WARMUP_PAGE} and publication_spec is not None:
            payload.setdefault("editorial_copy", dict(publication_spec.editorial_copy or {}))
            payload.setdefault("publication_metadata", dict(publication_spec.metadata or {}))

        if page_type == PROMO_PAGE and publication_spec is not None:
            payload.setdefault("editorial_copy", dict(publication_spec.editorial_copy or {}))
            payload.setdefault("ecosystem_config", dict(publication_spec.ecosystem_config or {}))
            payload.setdefault("publication_metadata", dict(publication_spec.metadata or {}))
            payload.setdefault("book_title", book_labels["title"])
            payload.setdefault("book_id", book_manifest.book_id)
            payload.setdefault("book_code", _publication_book_code(book_manifest, publication_spec))
            if puzzles_per_page is not None:
                payload.setdefault("publication_puzzles_per_page", int(puzzles_per_page))

        if page_type == RULES_PAGE:
            payload.setdefault("rules_page_index", int(page_type_seen[page_type]))
            payload.setdefault("rules_page_total", int(page_type_totals.get(page_type, 1)))

        if page_type == TUTORIAL_PAGE:
            payload.setdefault("tutorial_page_index", int(page_type_seen[page_type]))
            payload.setdefault("tutorial_page_total", int(page_type_totals.get(page_type, 1)))

        if page_type == WARMUP_PAGE:
            payload.setdefault("warmup_page_index", int(page_type_seen[page_type]))
            payload.setdefault("warmup_page_total", int(page_type_totals.get(page_type, 1)))

        page_blocks.append(
            PageBlock(
                page_type=page_type,
                template_id=str(item["template_id"]),
                section_id=item.get("section_id"),
                show_page_number=bool(item.get("show_page_number", False)),
                page_number_style=item.get("page_number_style"),
                payload=payload,
            )
        )


def _page_refs_label(page_numbers: List[int]) -> str:
    nums = [int(x) for x in page_numbers if x is not None]
    if not nums:
        return "Pages: —"

    if len(nums) == 1:
        return f"Pages: {nums[0]}"

    if len(nums) == 2:
        return f"Pages: {nums[0]} and {nums[1]}"

    return "Pages: " + ", ".join(str(x) for x in nums[:-1]) + f", and {nums[-1]}"


def _format_reference_display_puzzle_id(value: str) -> str:
    raw = str(value or "").strip().upper()
    match = re.match(r"^L-?(\d+)-(\d+)$", raw, flags=re.IGNORECASE)
    if match:
        return f"L-{int(match.group(1))}-{int(match.group(2))}"
    return raw


def _normalize_reference_lookup_code(value: str) -> str:
    raw = str(value or "").strip().upper()
    match = re.match(r"^L-?(\d+)-(\d+)$", raw, flags=re.IGNORECASE)
    if match:
        return f"L{int(match.group(1))}-{int(match.group(2)):03d}"
    return raw


def _book_number_for_reference_sentence(book_manifest) -> int:
    volume_number = getattr(book_manifest, "volume_number", None)
    if volume_number not in (None, ""):
        return int(volume_number)

    match = re.search(r"-B(\d+)$", str(getattr(book_manifest, "book_id", "") or "").strip(), flags=re.IGNORECASE)
    if match:
        return int(match.group(1))

    return 0


def _printed_page_number_for_block(block: PageBlock) -> int | None:
    payload = dict(block.payload or {})
    for candidate in (
        payload.get("printed_page_number"),
        block.logical_page_number,
        block.physical_page_number,
        block.page_index,
    ):
        if candidate not in (None, 0, ""):
            return int(candidate)
    return None


def _build_puzzle_placement_index(plan: InteriorPlan) -> Dict[str, Dict[str, int]]:
    placement: Dict[str, Dict[str, int]] = {}

    for block in plan.page_blocks:
        if block.page_type != PUZZLE_PAGE:
            continue

        printed_page_number = _printed_page_number_for_block(block)
        if printed_page_number is None:
            continue

        for slot_index, puzzle_uid in enumerate(list((block.payload or {}).get("puzzle_ids") or []), start=1):
            placement[str(puzzle_uid)] = {
                "page_number": int(printed_page_number),
                "slot_index": int(slot_index),
            }

    return placement


def _build_local_code_index(render_model) -> Dict[str, Dict[str, str]]:
    lookup: Dict[str, Dict[str, str]] = {}

    for section in render_model.sections:
        section_key = _section_key_for_manifest(section.section_manifest)
        for puzzle in section.puzzles:
            local_code = str(getattr(puzzle, "local_puzzle_code", "") or "").strip().upper()
            if not local_code:
                continue
            lookup[local_code] = {
                "puzzle_uid": str(puzzle.puzzle_uid),
                "section_key": str(section_key),
            }

    return lookup


def _ordinal_word(index: int, language: str) -> str:
    return tr(f"ordinal_{int(index)}", language)


def _slot_label(index: int, language: str) -> str:
    template = tr("puzzle_slot_on_page_pattern", language)
    return template.format(ordinal=_ordinal_word(index, language))


def _section_reference_label(section_key: str, editorial_copy: Dict[str, Any], language: str) -> str:
    sections = dict(editorial_copy.get("sections") or {})
    section_copy = dict(sections.get(str(section_key or "").strip().lower()) or {})

    title = str(section_copy.get("title") or "").strip()
    if not title:
        title = translate_difficulty_label(str(section_key or "").strip(), language)

    pattern = tr("section_reference_pattern", language)
    return pattern.format(section=title)


def _render_frontmatter_puzzle_reference(
    ref: Dict[str, Any],
    *,
    editorial_copy: Dict[str, Any],
    language: str,
    local_code_index: Dict[str, Dict[str, str]],
    puzzle_placement_index: Dict[str, Dict[str, int]],
    book_manifest,
) -> str:
    ref = dict(ref or {})
    kind = str(ref.get("kind") or "source").strip().lower()
    puzzle_id = str(ref.get("puzzle_id") or "").strip()
    if not puzzle_id:
        return ""

    lookup_code = _normalize_reference_lookup_code(puzzle_id)
    display_puzzle_id = _format_reference_display_puzzle_id(puzzle_id)
    local_entry = dict(local_code_index.get(lookup_code) or {})
    puzzle_uid = str(local_entry.get("puzzle_uid") or "").strip()
    placement = dict(puzzle_placement_index.get(puzzle_uid) or {})

    section_key = str(ref.get("section_key") or local_entry.get("section_key") or "").strip().lower()
    if not section_key:
        section_key = "section"

    page_number = placement.get("page_number")
    slot_index = placement.get("slot_index")

    render_mode = str(ref.get("render_mode") or "source_note").strip().lower()
    if render_mode == "solver_source_sentence":
        template = tr("solver_source_sentence_pattern", language)
        return template.format(
            book_number=_book_number_for_reference_sentence(book_manifest),
            display_puzzle_id=display_puzzle_id,
            section_label=_section_reference_label(section_key, editorial_copy, language),
            page_number=page_number if page_number is not None else "—",
            slot_label=_slot_label(slot_index, language) if slot_index is not None else tr("puzzle_slot_unknown", language),
        )

    label_key = "practice_puzzle" if kind == "practice" else "source_puzzle"
    parts = [display_puzzle_id]

    if bool(ref.get("include_section", True)):
        parts.append(_section_reference_label(section_key, editorial_copy, language))
    if bool(ref.get("include_page", True)):
        parts.append(f"{tr('page_word', language)} {page_number if page_number is not None else '—'}")
    if bool(ref.get("include_slot", True)):
        parts.append(_slot_label(slot_index, language) if slot_index is not None else tr("puzzle_slot_unknown", language))

    return f"{tr(label_key, language)} {' • '.join(parts)}"


def _resolve_frontmatter_puzzle_references(plan: InteriorPlan, render_model) -> None:
    local_code_index = _build_local_code_index(render_model)
    puzzle_placement_index = _build_puzzle_placement_index(plan)
    book_manifest = render_model.book_manifest

    for block in plan.page_blocks:
        if block.page_type not in {RULES_PAGE, TUTORIAL_PAGE, WARMUP_PAGE}:
            continue

        editorial_copy = deepcopy(dict((block.payload or {}).get("editorial_copy") or {}))
        if not editorial_copy:
            continue

        language = normalize_language(
            str((block.payload or {}).get("language") or (block.payload or {}).get("publication_language") or "en")
        )

        if block.page_type == RULES_PAGE:
            rules = dict(editorial_copy.get("rules") or {})

            page2 = dict(rules.get("page2") or {})
            source_ref = dict(page2.get("source_ref") or {})
            if source_ref:
                page2["source_note"] = _render_frontmatter_puzzle_reference(
                    source_ref,
                    editorial_copy=editorial_copy,
                    language=language,
                    local_code_index=local_code_index,
                    puzzle_placement_index=puzzle_placement_index,
                    book_manifest=book_manifest,
                )

            examples = dict(rules.get("examples") or {})
            examples_source_ref = dict(examples.get("source_ref") or {})
            if examples_source_ref:
                examples["source"] = _render_frontmatter_puzzle_reference(
                    examples_source_ref,
                    editorial_copy=editorial_copy,
                    language=language,
                    local_code_index=local_code_index,
                    puzzle_placement_index=puzzle_placement_index,
                    book_manifest=book_manifest,
                )

            rules["page2"] = page2
            rules["examples"] = examples
            editorial_copy["rules"] = rules

        if block.page_type == TUTORIAL_PAGE:
            tutorial = dict(editorial_copy.get("tutorial") or {})
            page_key = f"page{int((block.payload or {}).get('tutorial_page_index') or 1)}"
            page_cfg = dict(tutorial.get(page_key) or {})
            source_ref = dict(page_cfg.get("source_ref") or {})
            if source_ref:
                page_cfg["source_note"] = _render_frontmatter_puzzle_reference(
                    source_ref,
                    editorial_copy=editorial_copy,
                    language=language,
                    local_code_index=local_code_index,
                    puzzle_placement_index=puzzle_placement_index,
                    book_manifest=book_manifest,
                )
            tutorial[page_key] = page_cfg
            editorial_copy["tutorial"] = tutorial

        if block.page_type == WARMUP_PAGE:
            warmup = dict(editorial_copy.get("warmup") or {})
            page_key = f"page{int((block.payload or {}).get('warmup_page_index') or 1)}"
            page_cfg = dict(warmup.get(page_key) or {})
            source_ref = dict(page_cfg.get("source_ref") or {})
            if source_ref:
                page_cfg["source_note"] = _render_frontmatter_puzzle_reference(
                    source_ref,
                    editorial_copy=editorial_copy,
                    language=language,
                    local_code_index=local_code_index,
                    puzzle_placement_index=puzzle_placement_index,
                    book_manifest=book_manifest,
                )
            warmup[page_key] = page_cfg
            editorial_copy["warmup"] = warmup

        block.payload["editorial_copy"] = editorial_copy

def _apply_absolute_gallery_page_refs(plan: InteriorPlan) -> None:
    section_puzzle_pages: dict[str, List[int]] = {}

    for block in plan.page_blocks:
        if block.page_type != PUZZLE_PAGE:
            continue

        section_id = str(block.section_id or "").strip()
        if not section_id:
            continue

        abs_page = block.physical_page_number
        if abs_page in (None, 0):
            abs_page = block.logical_page_number
        if abs_page in (None, 0):
            abs_page = block.page_index

        if abs_page in (None, 0):
            continue

        section_puzzle_pages.setdefault(section_id, []).append(int(abs_page))

    for block in plan.page_blocks:
        if block.page_type != SECTION_PATTERN_GALLERY_PAGE:
            continue

        section_id = str(block.section_id or "").strip()
        if not section_id:
            continue

        absolute_pages_for_section = section_puzzle_pages.get(section_id, [])
        if not absolute_pages_for_section:
            continue

        patterns = list(block.payload.get("patterns") or [])
        for item in patterns:
            relative_refs = [int(x) for x in list(item.get("page_refs") or []) if x is not None]
            absolute_refs: List[int] = []

            for rel_num in relative_refs:
                if 1 <= rel_num <= len(absolute_pages_for_section):
                    absolute_refs.append(int(absolute_pages_for_section[rel_num - 1]))

            item["page_refs"] = absolute_refs
            item["page_refs_label"] = _page_refs_label(absolute_refs)


def _inject_toc_entries(plan: InteriorPlan) -> None:
    toc_entries = build_toc_entries(plan)
    for block in plan.page_blocks:
        if block.page_type == TOC_PAGE:
            block.payload["entries"] = toc_entries



def _resolve_effective_puzzles_per_page(book_manifest, publication_spec) -> int:
    cfg = publication_spec.layout_config

    if cfg.puzzles_per_page is not None:
        return max(1, int(cfg.puzzles_per_page))

    if cfg.rows is not None and cfg.cols is not None:
        return max(1, int(cfg.rows) * int(cfg.cols))

    return max(1, int(book_manifest.puzzles_per_page))


def _resolve_effective_solution_puzzles_per_page(publication_spec, fallback: int) -> int:
    cfg = publication_spec.layout_config

    value = getattr(cfg, "solution_puzzles_per_page", None)
    if value is not None:
        return max(1, int(value))

    rows = getattr(cfg, "solution_rows", None)
    cols = getattr(cfg, "solution_cols", None)
    if rows is not None and cols is not None:
        return max(1, int(rows) * int(cols))

    template = str(publication_spec.solution_page_template or "").lower()
    if "12up" in template:
        return 12
    if "6up" in template:
        return 6
    if "4up" in template:
        return 4
    if "2up" in template:
        return 2
    if "1up" in template:
        return 1

    return max(1, int(fallback))


def _build_initial_interior_plan(
    render_model,
    publication_spec,
    *,
    book_dir: Path,
    skip_puzzle_sections: bool = False,
) -> InteriorPlan:
    book_manifest = render_model.book_manifest
    page_blocks: List[PageBlock] = []

    puzzles_per_page = _resolve_effective_puzzles_per_page(book_manifest, publication_spec)
    solution_puzzles_per_page = _resolve_effective_solution_puzzles_per_page(
        publication_spec,
        puzzles_per_page,
    )

    book_labels = _resolve_publication_book_labels(book_manifest, publication_spec)
    book_code = _publication_book_code(book_manifest, publication_spec)

    page_blocks.append(
        PageBlock(
            page_type=TITLE_PAGE,
            template_id="title_page_basic",
            show_page_number=False,
            page_number_style=None,
            payload={
                "book_id": book_manifest.book_id,
                "book_code": book_code,
                "title": book_labels["title"],
                "subtitle": book_labels["subtitle"],
                "series_name": book_labels["series_name"],
                "library_id": book_manifest.library_id,
                "aisle_id": book_manifest.aisle_id,
                "layout_type": book_manifest.layout_type,
                "trim_size": book_manifest.trim_size,
                "publication_trim_size_label": book_manifest.trim_size,
                "publication_puzzles_per_page": puzzles_per_page,
                "publication_language": str(
                    publication_spec.layout_config.language
                    or publication_spec.layout_config.language_code
                    or publication_spec.metadata.get("language")
                    or "en"
                ),
                "total_puzzles": book_manifest.puzzle_count,
                "description": book_labels["description"],
                "editorial_copy": dict(publication_spec.editorial_copy or {}),
                "publication_metadata": dict(publication_spec.metadata or {}),
                "kdp_listing": dict(book_labels.get("kdp_listing") or {}),
            },
        )
    )

    front_profile_items = _resolve_front_matter_items(publication_spec)
    _append_profile_blocks(
        page_blocks,
        front_profile_items,
        book_manifest=book_manifest,
        publication_spec=publication_spec,
        puzzles_per_page=puzzles_per_page,
    )

    
    all_puzzle_ids: List[str] = []

    for section in render_model.sections:
        # Always collect the puzzle ids, even when we are building a solution-only
        # booklet. The booklet skips puzzle problem pages, but it still needs the
        # full answer-key sequence.
        section_puzzle_ids = [p.puzzle_uid for p in section.puzzles]
        all_puzzle_ids.extend(section_puzzle_ids)

        if skip_puzzle_sections:
            continue

        section_prelude_items = _resolve_section_prelude_items(publication_spec)

        for item in section_prelude_items:
            page_type = str(item["page_type"])

            if page_type == SECTION_HIGHLIGHTS_PAGE:
                payload = build_section_highlights_payload(
                    section=section,
                    publication_spec=publication_spec,
                    book_dir=book_dir,
                )
                page_blocks.append(
                    PageBlock(
                        page_type=page_type,
                        template_id=str(item["template_id"]),
                        section_id=section.section_manifest.section_id,
                        show_page_number=bool(item.get("show_page_number", False)),
                        page_number_style=item.get("page_number_style"),
                        payload=payload,
                    )
                )
                continue

            if page_type == SECTION_PATTERN_GALLERY_PAGE:
                gallery_payloads = build_section_pattern_gallery_payloads(
                    section=section,
                    publication_spec=publication_spec,
                    book_dir=book_dir,
                )
                for payload in gallery_payloads:
                    page_blocks.append(
                        PageBlock(
                            page_type=page_type,
                            template_id=str(item["template_id"]),
                            section_id=section.section_manifest.section_id,
                            show_page_number=bool(item.get("show_page_number", False)),
                            page_number_style=item.get("page_number_style"),
                            payload=payload,
                        )
                    )
                continue

            payload = dict(item.get("payload", {}))
            section_labels = _resolve_localized_section_labels(section, publication_spec)
            payload.setdefault("section_id", section.section_manifest.section_id)
            payload.setdefault("section_code", section.section_manifest.section_code)
            payload.setdefault("title", section_labels["title"])
            payload.setdefault("subtitle", section_labels["subtitle"])
            payload.setdefault("difficulty_label_hint", section.section_manifest.difficulty_label_hint)
            payload.setdefault("puzzle_count", len(section.puzzles))
            payload.setdefault(
                "language",
                str(
                    publication_spec.layout_config.language
                    or publication_spec.layout_config.language_code
                    or publication_spec.metadata.get("locale")
                    or publication_spec.metadata.get("language")
                    or "en"
                ),
            )
            payload.setdefault("section_preview_config", dict(publication_spec.section_preview_config or {}))
            payload.setdefault("editorial_copy", dict(publication_spec.editorial_copy or {}))
            payload.setdefault("publication_metadata", dict(publication_spec.metadata or {}))

            page_blocks.append(
                PageBlock(
                    page_type=page_type,
                    template_id=str(item["template_id"]),
                    section_id=section.section_manifest.section_id,
                    show_page_number=bool(item.get("show_page_number", False)),
                    page_number_style=item.get("page_number_style"),
                    payload=payload,
                )
            )

        for i in range(0, len(section.puzzles), puzzles_per_page):
            chunk = section.puzzles[i : i + puzzles_per_page]
            page_blocks.append(
                PageBlock(
                    page_type=PUZZLE_PAGE,
                    template_id=publication_spec.puzzle_page_template,
                    section_id=section.section_manifest.section_id,
                    show_page_number=True,
                    page_number_style="arabic",
                    payload={
                        "section_id": section.section_manifest.section_id,
                        "section_code": section.section_manifest.section_code,
                        "section_title": _resolve_localized_section_labels(section, publication_spec)["title"],
                        "language": str(
                            publication_spec.layout_config.language
                            or publication_spec.layout_config.language_code
                            or publication_spec.metadata.get("locale")
                            or publication_spec.metadata.get("language")
                            or "en"
                        ),
                        "puzzle_ids": [p.puzzle_uid for p in chunk],
                    },
                )
            )

    if publication_spec.include_solutions and all_puzzle_ids:
        solution_cfg = dict(getattr(publication_spec, "solution_section_config", {}) or {})

        if bool(solution_cfg.get("enabled", False)):
            solution_opener_payload = dict(solution_cfg)

            solution_opener_payload.update(
                {
                    "headline": str(solution_cfg.get("headline") or "Solutions Included"),
                    "kicker": str(solution_cfg.get("kicker") or "Answer key"),
                    "body": str(
                        solution_cfg.get("body")
                        or "Use these pages to check your completed grids."
                    ),
                    "usage_tips": list(solution_cfg.get("usage_tips") or []),
                    "language": str(
                        publication_spec.layout_config.language
                        or publication_spec.metadata.get("locale")
                        or publication_spec.metadata.get("language")
                        or "en"
                    ),
                    "solution_puzzles_per_page": solution_puzzles_per_page,
                    "solution_page_template": publication_spec.solution_page_template,
                }
            )

            page_blocks.append(
                PageBlock(
                    page_type="SOLUTION_SECTION_OPENER_PAGE",
                    template_id=str(solution_cfg.get("template_id") or "solution_section_opener_basic"),
                    show_page_number=True,
                    page_number_style="arabic",
                    payload=solution_opener_payload,
                )
            )

        for i in range(0, len(all_puzzle_ids), solution_puzzles_per_page):
            page_blocks.append(
                PageBlock(
                    page_type=SOLUTION_PAGE,
                    template_id=publication_spec.solution_page_template,
                    show_page_number=True,
                    page_number_style="arabic",
                    payload={
                        "puzzle_ids": all_puzzle_ids[i : i + solution_puzzles_per_page],
                        "language": str(
                            publication_spec.layout_config.language
                            or publication_spec.metadata.get("locale")
                            or publication_spec.metadata.get("language")
                            or "en"
                        ),
                        "page_title": tr(
                            "solutions",
                            str(
                                publication_spec.layout_config.language
                                or publication_spec.metadata.get("locale")
                                or publication_spec.metadata.get("language")
                                or "en"
                            ),
                        ),
                        "solution_puzzles_per_page": solution_puzzles_per_page,
                    },
                )
            )

    end_profile_items = resolve_end_matter_profile(publication_spec.end_matter_profile)
    _append_profile_blocks(
        page_blocks,
        end_profile_items,
        book_manifest=book_manifest,
        publication_spec=publication_spec,
        puzzles_per_page=puzzles_per_page,
    )

    
    plan = InteriorPlan(
        page_blocks=page_blocks,
        estimated_page_count=len(page_blocks),
        requires_blank_page_adjustment=False,
        notes=[
            "Wave 5 publication package: cover manifest added alongside interior plan.",
            "Phase 1 pagination constitution: physical and logical page numbers are assigned book-wide.",
            "Phase 1B full mirror parity is active for odd/even page geometry.",
            "Phase 2 layout config is now spec-driven and overrides rows, cols, margins, header/footer reserves, and gutters when provided.",
            "Phase 3 puzzle tiles use a black-band commercial header and tighter grid framing.",
            "Phase 3B adds built-in 2-up, 4-up, 6-up, and 12-up template layout defaults.",
            "Phase 4 adds explicit spacing controls and geometry validation for slots and tile internals.",
            "Phase 5 adds typography controls and digit-size presets via publication layout_config.",
            "Phase 6 adds localization via language-aware system strings and translated difficulty/header labels.",
            "Phase 7 adds command-line publication spec overrides.",
            "Phase 8 adds publication QC, richer plan preview, and lightweight regression checks.",
            "Rendering supports physical page planning, printed page numbering, and cover geometry export.",
        ],
    )

    insert_required_blank_pages(
        plan,
        publication_spec.blank_page_policy,
        recto_start_policy=publication_spec.recto_start_policy,
    )
    apply_page_numbering(plan, publication_spec.page_numbering_policy)
    _apply_absolute_gallery_page_refs(plan)
    _resolve_frontmatter_puzzle_references(plan, render_model)
    _inject_toc_entries(plan)

    return plan





def _build_cover_spec(render_model, publication_spec, format_spec, interior_plan: InteriorPlan) -> CoverSpec | None:
    if not publication_spec.include_cover:
        return None

    book_manifest = render_model.book_manifest
    page_count = max(1, interior_plan.estimated_page_count)
    spine_width_in = compute_spine_width_in(
        page_count=page_count,
        paper_type=publication_spec.paper_type,
        channel_id=publication_spec.channel_id,
    )

    cover_metadata = publication_spec.metadata or {}

    # Platform rule:
    # Only books with 150 pages or more receive printed spine text.
    # Smaller books keep the spine visually clean, even if metadata.spine_text exists.
    spine_text = str(cover_metadata.get("spine_text") or book_manifest.title)
    if page_count < 150:
        spine_text = ""

    return CoverSpec(
        cover_id=f"COVER-{publication_spec.publication_id}",
        publication_id=publication_spec.publication_id,
        format_id=publication_spec.format_id,
        page_count=page_count,
        paper_type=publication_spec.paper_type,
        spine_width_in=spine_width_in,
        front_design_asset=cover_metadata.get("front_design_asset"),
        back_design_asset=cover_metadata.get("back_design_asset"),
        spine_text=spine_text,
        back_copy=str(cover_metadata.get("back_copy") or book_manifest.description),
        author_imprint=str(cover_metadata.get("imprint_name", "")),
        isbn=cover_metadata.get("isbn") or book_manifest.isbn,
    )


def build_publication_package(
    *,
    book_dir: Path,
    publication_spec_path: Path,
    output_publications_dir: Path,
    skip_puzzle_sections: bool = False,
) -> Tuple[Path, PublicationPackage]:
    render_model = load_built_book_render_model(book_dir)
    publication_spec = load_publication_spec(publication_spec_path)

    if publication_spec.book_id != render_model.book_manifest.book_id:
        raise ValueError(
            f"Publication spec book_id '{publication_spec.book_id}' does not match built book "
            f"'{render_model.book_manifest.book_id}'"
        )

    format_spec = get_print_format_spec(publication_spec.format_id)

    errors = []
    errors.extend(validate_print_format_spec(format_spec))
    errors.extend(validate_publication_spec(publication_spec, format_spec))
    if errors:
        raise ValueError("Publication validation failed:\n- " + "\n- ".join(errors))

    publication_dir = get_publication_dir(
        book_id=render_model.book_manifest.book_id,
        publication_id=publication_spec.publication_id,
        base_dir=output_publications_dir,
    )
    publication_dir.mkdir(parents=True, exist_ok=True)

    interior_plan = _build_initial_interior_plan(
        render_model,
        publication_spec,
        book_dir=book_dir,
        skip_puzzle_sections=skip_puzzle_sections,
    )
    cover_spec = _build_cover_spec(render_model, publication_spec, format_spec, interior_plan)

    publication_manifest = build_publication_manifest(
        book_manifest=render_model.book_manifest,
        publication_spec=publication_spec,
        format_spec=format_spec,
        interior_plan=interior_plan,
        cover_spec=cover_spec,
    )

    publication_manifest_path = publication_dir / "publication_manifest.json"
    interior_plan_path = publication_dir / "interior_plan.json"
    cover_spec_path = publication_dir / "cover_spec.json"
    cover_manifest_path = publication_dir / "cover_manifest.json"

    write_json(publication_manifest_path, publication_manifest.to_dict())
    write_json(interior_plan_path, interior_plan.to_dict())
    if cover_spec is not None:
        write_json(cover_spec_path, cover_spec.to_dict())
        cover_manifest = build_cover_manifest(
            publication_spec=publication_spec,
            format_spec=format_spec,
            cover_spec=cover_spec,
        )
        write_json(cover_manifest_path, cover_manifest)

    package = PublicationPackage(
        publication_id=publication_spec.publication_id,
        book_id=render_model.book_manifest.book_id,
        book_dir=str(book_dir),
        publication_dir=str(publication_dir),
        format_id=publication_spec.format_id,
        channel_id=publication_spec.channel_id,
        publication_manifest_path=str(publication_manifest_path),
        interior_plan_path=str(interior_plan_path),
        cover_spec_path=str(cover_spec_path) if cover_spec is not None else None,
        generated_at=_now_iso(),
        warnings=list(interior_plan.notes),
    )

    write_json(publication_dir / "publication_package.json", package.to_dict())
    return publication_dir, package