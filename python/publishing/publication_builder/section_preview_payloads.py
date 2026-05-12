from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from python.publishing.i18n.strings import tr, translate_difficulty_label
from python.publishing.pattern_library.pattern_store import load_pattern_store
from python.publishing.techniques.technique_catalog import (
    get_public_technique_name,
    normalize_technique_id,
)


_EASY_TECHNIQUES = {
    normalize_technique_id("singles_1"),
    normalize_technique_id("singles_2"),
    normalize_technique_id("singles_3"),
}
_DEFAULT_MAX_SNEAK_PEEK_PATTERNS = 16
_DEFAULT_GALLERY_PAGE_SIZE = 8


def _norm(value: Any) -> str:
    return normalize_technique_id(str(value))


def _publication_language(publication_spec) -> str:
    metadata = dict(getattr(publication_spec, "metadata", {}) or {})
    raw = str(metadata.get("locale") or metadata.get("language") or "").strip().lower()

    aliases = {
        "en": "en",
        "english": "en",
        "fr": "fr",
        "french": "fr",
        "de": "de",
        "german": "de",
        "it": "it",
        "italian": "it",
        "es": "es",
        "sp": "es",
        "spanish": "es",
    }
    return aliases.get(raw, "en")


def _unique_preserve_order(values) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []

    for value in values:
        cleaned = str(value or "").strip()
        if not cleaned:
            continue

        key = cleaned.lower()
        if key in seen:
            continue

        seen.add(key)
        result.append(cleaned)

    return result


def _public_technique_label(value: Any, language: str) -> str:
    key = _norm(value)

    localized = {
        "singles_1": tr("hidden_single", language),
        "singles_2": tr("hidden_single", language),
        "singles_3": tr("hidden_single", language),
        "singles_naked_2": tr("naked_single", language),
        "singles_naked_3": tr("naked_single", language),
        "doubles_naked": tr("naked_pair", language),
        "triplets_naked": tr("naked_triple", language),
        "singles_pointing": tr("pointing_pair", language),
        "singles_boxed": tr("claiming_pair", language),
    }

    label = str(localized.get(key) or "").strip()
    if label:
        return label

    return get_public_technique_name(key, plural=True)


def _public_combo_label(value: Any, language: str) -> str:
    parts = [part.strip() for part in str(value or "").split("+") if part.strip()]
    if not parts:
        return ""

    if len(parts) == 1:
        return _public_technique_label(parts[0], language)

    return " + ".join(
        _unique_preserve_order(
            _public_technique_label(part, language)
            for part in parts
        )
    )


def _infer_patterns_dir(book_dir: Path) -> Path:
    return Path(book_dir).resolve().parent.parent / "patterns"


def _load_pattern_lookup(patterns_dir: Path) -> Dict[str, Any]:
    if not patterns_dir.exists():
        return {}
    try:
        registry = load_pattern_store(patterns_dir)
    except Exception:
        return {}

    out: Dict[str, Any] = {}
    for pattern in registry.patterns:
        out[str(pattern.pattern_id)] = pattern
    return out


def _section_key(section) -> str:
    """
    Resolve the semantic section key used by editorial_copy.sections.

    Do not infer difficulty from section_code. L1/L2/L3 are ordering labels,
    not stable difficulty identities. For example:

        B02: L1=medium, L2=hard,   L3=expert
        B03: L1=easy,   L2=medium, L3=hard
    """
    hint = str(section.section_manifest.difficulty_label_hint or "").strip().lower()
    title = str(section.section_manifest.title or "").strip().lower()

    for value in (hint, title):
        normalized = value.replace("_", " ").replace("-", " ").strip()
        if normalized in {"easy", "medium", "hard", "expert"}:
            return normalized

    code = str(section.section_manifest.section_code or "").strip().lower()
    return hint or title or code or "section"


def _section_editorial(publication_spec, section) -> Dict[str, Any]:
    editorial = dict(publication_spec.editorial_copy or {})
    sections = dict(editorial.get("sections") or {})
    return dict(sections.get(_section_key(section)) or {})

def _section_identity(section, publication_spec) -> Tuple[str, str]:
    section_copy = _section_editorial(publication_spec, section)
    language = _publication_language(publication_spec)

    hint = str(section.section_manifest.difficulty_label_hint or "").strip()
    key = _section_key(section)

    title = str(section_copy.get("title") or "").strip()
    if not title:
        title = translate_difficulty_label(hint or key or section.section_manifest.title, language)

    subtitle = str(section_copy.get("subtitle") or "").strip()
    if not subtitle:
        subtitle = str(section.section_manifest.subtitle or "").strip()

    if not subtitle and key in {"easy", "medium", "hard", "expert"}:
        subtitle = tr(f"section_subtitle_{key}", language)

    return title, subtitle

def _section_pattern_stats(section) -> Tuple[Counter, Dict[str, int]]:
    counts = Counter()
    first_seen: Dict[str, int] = {}

    for idx, puzzle in enumerate(section.puzzles, start=1):
        pattern_id = str(puzzle.pattern_id or "").strip()
        if not pattern_id:
            continue
        counts[pattern_id] += 1
        if pattern_id not in first_seen:
            first_seen[pattern_id] = idx

    return counts, first_seen


def _section_pattern_page_map(section, *, puzzles_per_page: int) -> Dict[str, List[int]]:
    page_map: Dict[str, List[int]] = {}
    safe_ppp = max(1, int(puzzles_per_page))

    for idx, puzzle in enumerate(section.puzzles, start=1):
        pattern_id = str(puzzle.pattern_id or "").strip()
        if not pattern_id:
            continue

        page_num = ((idx - 1) // safe_ppp) + 1

        existing = page_map.setdefault(pattern_id, [])
        if page_num not in existing:
            existing.append(page_num)

    return page_map


def _top_techniques_by_presence(
    section,
    *,
    limit: int = 4,
    language: str = "en",
) -> List[str]:
    presence = Counter()
    for puzzle in section.puzzles:
        used = {_norm(x) for x in list(puzzle.techniques_used or [])}
        used = {x for x in used if x not in _EASY_TECHNIQUES}
        for name in used:
            presence[name] += 1

    if not presence:
        for puzzle in section.puzzles:
            used = {_norm(x) for x in list(puzzle.techniques_used or [])}
            for name in used:
                presence[name] += 1

    return _unique_preserve_order(
        _public_technique_label(name, language)
        for name, _count in presence.most_common(limit)
    )


def _top_two_technique_combos(
    section,
    *,
    limit: int = 3,
    language: str = "en",
) -> List[str]:
    combo_counts = Counter()

    for puzzle in section.puzzles:
        used = sorted(
            {
                _norm(x)
                for x in list(puzzle.techniques_used or [])
                if _norm(x) not in _EASY_TECHNIQUES
            }
        )
        if len(used) < 2:
            continue

        for i in range(len(used)):
            for j in range(i + 1, len(used)):
                combo_counts[(used[i], used[j])] += 1

    rows: List[str] = []
    for (a, b), _count in combo_counts.most_common(limit):
        rows.append(_public_combo_label(f"{a} + {b}", language))

    return _unique_preserve_order(rows)


def _section_story(section, publication_spec) -> str:
    section_copy = _section_editorial(publication_spec, section)
    explicit_story = str(section_copy.get("story") or "").strip()
    if explicit_story:
        return explicit_story

    hint = str(section.section_manifest.difficulty_label_hint or "").strip().lower()
    if hint == "medium":
        return (
            "This section establishes rhythm: approachable but rewarding grids, broad pattern variety, "
            "and a smooth climb into stronger logic."
        )
    if hint == "hard":
        return (
            "This section sharpens the challenge: denser logic, stronger interactions between techniques, "
            "and a more demanding solving tempo."
        )
    return (
        "This section offers a curated set of puzzles with visible structure, pattern variety, and a clear solving identity."
    )


def _effective_puzzles_per_page(publication_spec) -> int:
    cfg = publication_spec.layout_config
    if cfg.puzzles_per_page is not None:
        return max(1, int(cfg.puzzles_per_page))
    if cfg.rows is not None and cfg.cols is not None:
        return max(1, int(cfg.rows) * int(cfg.cols))
    return 6


def _gallery_limits(publication_spec) -> tuple[int, int]:
    config = dict(publication_spec.section_preview_config or {})
    max_patterns = int(config.get("max_patterns_total") or _DEFAULT_MAX_SNEAK_PEEK_PATTERNS)
    cards_per_page = int(config.get("cards_per_page") or _DEFAULT_GALLERY_PAGE_SIZE)
    return max(1, max_patterns), max(1, cards_per_page)


def _pick_sneak_peek_pattern_ids(ordered_pattern_ids: Sequence[str], max_items: int) -> List[str]:
    ordered = list(ordered_pattern_ids)
    n = len(ordered)

    if n <= max_items:
        return ordered

    chosen_indices = []
    for i in range(max_items):
        idx = round(i * (n - 1) / (max_items - 1))
        if idx not in chosen_indices:
            chosen_indices.append(idx)

    cursor = 0
    while len(chosen_indices) < max_items and cursor < n:
        if cursor not in chosen_indices:
            chosen_indices.append(cursor)
        cursor += 1

    chosen_indices = sorted(chosen_indices[:max_items])
    return [ordered[i] for i in chosen_indices]


def _page_refs_label(page_numbers: Sequence[int]) -> str:
    nums = [int(x) for x in page_numbers if x is not None]
    if not nums:
        return "Pages: —"

    if len(nums) == 1:
        return f"Pages: {nums[0]}"

    if len(nums) == 2:
        return f"Pages: {nums[0]} and {nums[1]}"

    return "Pages: " + ", ".join(str(x) for x in nums[:-1]) + f", and {nums[-1]}"


def build_section_highlights_payload(
    *,
    section,
    publication_spec,
    book_dir: Path,
) -> Dict[str, Any]:
    section_copy = _section_editorial(publication_spec, section)
    title, subtitle = _section_identity(section, publication_spec)
    language = _publication_language(publication_spec)

    weights = [int(p.weight) for p in section.puzzles if p.weight is not None]
    clues = [int(p.clue_count) for p in section.puzzles if p.clue_count is not None]
    pattern_counts, _first_seen = _section_pattern_stats(section)
    families = {
        str(p.pattern_family_name or p.pattern_family_id or "").strip()
        for p in section.puzzles
        if str(p.pattern_family_name or p.pattern_family_id or "").strip()
    }

    return {
        "section_id": section.section_manifest.section_id,
        "section_code": section.section_manifest.section_code,
        "title": title,
        "subtitle": subtitle,
        "difficulty_label_hint": section.section_manifest.difficulty_label_hint,
        "headline": str(section_copy.get("highlights_headline") or f"{title} Highlights"),
        "puzzle_count": len(section.puzzles),
        "story": _section_story(section, publication_spec),
        "weight_min": min(weights) if weights else None,
        "weight_max": max(weights) if weights else None,
        "clue_min": min(clues) if clues else None,
        "clue_max": max(clues) if clues else None,
        "unique_patterns": len(pattern_counts),
        "unique_pattern_families": len(families),
        "top_techniques": _top_techniques_by_presence(
            section,
            limit=4,
            language=language,
        ),
        "top_combos": _top_two_technique_combos(
            section,
            limit=3,
            language=language,
        ),
        "section_preview_config": dict(publication_spec.section_preview_config or {}),
        "editorial_copy": dict(publication_spec.editorial_copy or {}),
        "publication_metadata": dict(publication_spec.metadata or {}),
    }


def build_section_pattern_gallery_payloads(
    *,
    section,
    publication_spec,
    book_dir: Path,
) -> List[Dict[str, Any]]:
    section_copy = _section_editorial(publication_spec, section)
    title, subtitle = _section_identity(section, publication_spec)

    patterns_dir = _infer_patterns_dir(book_dir)
    pattern_lookup = _load_pattern_lookup(patterns_dir)

    pattern_counts, first_seen = _section_pattern_stats(section)
    puzzles_per_page = _effective_puzzles_per_page(publication_spec)
    pattern_page_map = _section_pattern_page_map(section, puzzles_per_page=puzzles_per_page)
    max_patterns, cards_per_page = _gallery_limits(publication_spec)

    ordered_pattern_ids = sorted(
        pattern_counts.keys(),
        key=lambda pattern_id: (first_seen.get(pattern_id, 10**9), pattern_id),
    )

    selected_pattern_ids = _pick_sneak_peek_pattern_ids(ordered_pattern_ids, max_items=max_patterns)

    items: List[Dict[str, Any]] = []
    for pattern_id in selected_pattern_ids:
        pattern = pattern_lookup.get(pattern_id)
        pattern_name = ""
        family_name = ""
        mask81 = ""

        if pattern is not None:
            pattern_name = str(pattern.name or pattern_id)
            family_name = str(pattern.family_name or pattern.visual_family or "")
            mask81 = str(pattern.mask81 or "")
        else:
            for puzzle in section.puzzles:
                if str(puzzle.pattern_id or "") == pattern_id:
                    pattern_name = str(puzzle.pattern_name or pattern_id)
                    family_name = str(puzzle.pattern_family_name or puzzle.pattern_family_id or "")
                    break

        page_refs = list(pattern_page_map.get(pattern_id, []))

        items.append(
            {
                "pattern_id": pattern_id,
                "pattern_name": pattern_name or pattern_id,
                "family_name": family_name,
                "occurrence_count": int(pattern_counts[pattern_id]),
                "page_refs": page_refs,
                "page_refs_label": _page_refs_label(page_refs),
                "mask81": mask81,
            }
        )

    pages: List[Dict[str, Any]] = []
    if not items:
        pages.append(
            {
                "section_id": section.section_manifest.section_id,
                "section_code": section.section_manifest.section_code,
                "title": title,
                "subtitle": subtitle,
                "headline": str(section_copy.get("sneak_peek_headline") or f"{title} Pattern Sneak Peek"),
                "intro": str(section_copy.get("sneak_peek_intro") or ""),
                "gallery_page_index": 1,
                "gallery_page_count": 1,
                "patterns": [],
                "section_preview_config": dict(publication_spec.section_preview_config or {}),
                "editorial_copy": dict(publication_spec.editorial_copy or {}),
                "publication_metadata": dict(publication_spec.metadata or {}),
            }
        )
        return pages

    chunks = [items[i : i + cards_per_page] for i in range(0, len(items), cards_per_page)]
    total_pages = len(chunks)

    for page_idx, chunk in enumerate(chunks, start=1):
        pages.append(
            {
                "section_id": section.section_manifest.section_id,
                "section_code": section.section_manifest.section_code,
                "title": title,
                "subtitle": subtitle,
                "headline": str(section_copy.get("sneak_peek_headline") or f"{title} Pattern Sneak Peek"),
                "intro": str(section_copy.get("sneak_peek_intro") or ""),
                "gallery_page_index": page_idx,
                "gallery_page_count": total_pages,
                "patterns": chunk,
                "section_preview_config": dict(publication_spec.section_preview_config or {}),
                "editorial_copy": dict(publication_spec.editorial_copy or {}),
                "publication_metadata": dict(publication_spec.metadata or {}),
            }
        )

    return pages