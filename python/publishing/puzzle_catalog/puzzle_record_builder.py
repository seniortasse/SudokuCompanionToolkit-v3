from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from python.publishing.ids.catalog_ids import build_record_id
from python.publishing.puzzle_catalog.catalog_identity import DEFAULT_CANDIDATE_STATUS
from python.publishing.puzzle_catalog.difficulty_enricher import enrich_candidate_difficulty
from python.publishing.puzzle_catalog.metadata_enricher import enrich_puzzle_metadata, build_print_header
from python.publishing.puzzle_catalog.pattern_linker import PatternLookup
from python.publishing.puzzle_catalog.solution_signature import build_solution_signature
from python.publishing.puzzle_catalog.technique_profile import build_technique_profile
from python.publishing.schemas.models import PatternRecord, PuzzleRecord


@dataclass
class GeneratorCandidate:
    givens81: str
    solution81: str
    weight: int
    techniques_used: List[str] = field(default_factory=list)
    technique_histogram: Dict[str, int] = field(default_factory=dict)

    pattern_id: Optional[str] = None
    pattern_name: Optional[str] = None
    pattern_family_id: Optional[str] = None
    pattern_family_name: Optional[str] = None
    pattern_mask81: Optional[str] = None

    generation_seed: Optional[int] = None
    generator_version: Optional[str] = None
    generation_method: Optional[str] = "pattern_fill"
    is_unique: Optional[bool] = True
    is_human_solvable: Optional[bool] = True
    title: Optional[str] = ""
    subtitle: Optional[str] = "Classic 9x9"

    request_id: Optional[str] = None
    hint_count: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GeneratorCandidate":
        return cls(
            givens81=str(data["givens81"]),
            solution81=str(data["solution81"]),
            weight=int(data.get("weight", 0)),
            techniques_used=[str(x) for x in list(data.get("techniques_used") or [])],
            technique_histogram={
                str(k): int(v)
                for k, v in dict(data.get("technique_histogram") or {}).items()
            },
            pattern_id=str(data["pattern_id"]) if data.get("pattern_id") is not None else None,
            pattern_name=str(data["pattern_name"]) if data.get("pattern_name") is not None else None,
            pattern_family_id=str(data["pattern_family_id"]) if data.get("pattern_family_id") is not None else None,
            pattern_family_name=str(data["pattern_family_name"]) if data.get("pattern_family_name") is not None else None,
            pattern_mask81=str(data["pattern_mask81"]) if data.get("pattern_mask81") is not None else None,
            generation_seed=int(data["generation_seed"]) if data.get("generation_seed") is not None else None,
            generator_version=str(data["generator_version"]) if data.get("generator_version") is not None else None,
            generation_method=str(data.get("generation_method", "pattern_fill")) if data.get("generation_method") is not None else "pattern_fill",
            is_unique=bool(data.get("is_unique", True)),
            is_human_solvable=bool(data.get("is_human_solvable", True)),
            title=str(data.get("title", "")) if data.get("title") is not None else "",
            subtitle=str(data.get("subtitle", "Classic 9x9")) if data.get("subtitle") is not None else "Classic 9x9",
            request_id=str(data["request_id"]) if data.get("request_id") is not None else None,
            hint_count=int(data["hint_count"]) if data.get("hint_count") is not None else None,
        )


def _count_clues(givens81: str) -> int:
    return sum(1 for ch in givens81 if ch != "0")


def _resolve_pattern_metadata(
    candidate: GeneratorCandidate,
    pattern_lookup: PatternLookup | None,
) -> Dict[str, Optional[str]]:
    resolved: Dict[str, Optional[str]] = {
        "pattern_id": candidate.pattern_id,
        "pattern_name": candidate.pattern_name,
        "pattern_family_id": candidate.pattern_family_id,
        "pattern_family_name": candidate.pattern_family_name,
        "pattern_mask81": candidate.pattern_mask81,
        "symmetry_type": None,
    }

    if pattern_lookup is None:
        return resolved

    # First enrich by explicit pattern_id if present.
    if resolved["pattern_id"]:
        pattern = pattern_lookup.find_by_id(str(resolved["pattern_id"]))
        if pattern is not None:
            resolved["pattern_id"] = pattern.pattern_id
            resolved["pattern_name"] = resolved["pattern_name"] or pattern.name
            resolved["pattern_family_id"] = resolved["pattern_family_id"] or pattern.family_id
            resolved["pattern_family_name"] = resolved["pattern_family_name"] or pattern.family_name
            resolved["pattern_mask81"] = resolved["pattern_mask81"] or pattern.mask81
            resolved["symmetry_type"] = pattern.symmetry_type
            return resolved

    # Fallback: reverse lookup by mask.
    if resolved["pattern_mask81"]:
        pattern = pattern_lookup.find_by_mask(str(resolved["pattern_mask81"]))
        if pattern is not None:
            resolved["pattern_id"] = resolved["pattern_id"] or pattern.pattern_id
            resolved["pattern_name"] = resolved["pattern_name"] or pattern.name
            resolved["pattern_family_id"] = resolved["pattern_family_id"] or pattern.family_id
            resolved["pattern_family_name"] = resolved["pattern_family_name"] or pattern.family_name
            resolved["pattern_mask81"] = resolved["pattern_mask81"] or pattern.mask81
            resolved["symmetry_type"] = pattern.symmetry_type

    return resolved


def _build_pattern_stub(
    resolved_pattern: Dict[str, Optional[str]],
) -> PatternRecord | None:
    pattern_id = resolved_pattern.get("pattern_id")
    pattern_name = resolved_pattern.get("pattern_name")
    pattern_mask81 = resolved_pattern.get("pattern_mask81")

    if not pattern_id and not pattern_name and not pattern_mask81:
        return None

    return PatternRecord(
        pattern_id=pattern_id or "UNRESOLVED-PATTERN",
        library_id="",
        name=pattern_name or "Unresolved Pattern",
        slug="",
        aliases=[],
        description="",
        grid_size=9,
        layout_type="classic9x9",
        mask81=pattern_mask81 or "",
        canonical_mask_signature="",
        clue_count=sum(1 for ch in (pattern_mask81 or "") if ch == "1"),
        symmetry_type=resolved_pattern.get("symmetry_type"),
        visual_family=resolved_pattern.get("pattern_family_id") or "uncategorized",
        family_id=resolved_pattern.get("pattern_family_id"),
        family_name=resolved_pattern.get("pattern_family_name"),
        variant_code="",
        tags=[],
        status="active",
        source_type="derived",
        source_ref=None,
        author=None,
        notes=None,
        is_verified=False,
        created_at=None,
        updated_at=None,
    )


def build_puzzle_record(
    *,
    candidate: GeneratorCandidate,
    library_id: str,
    layout_short: str,
    ordinal: int,
    layout_type: str,
    grid_size: int,
    charset: str,
    pattern_lookup: PatternLookup | None = None,
    created_at: str | None = None,
    updated_at: str | None = None,
) -> PuzzleRecord:
    record_id = build_record_id(
        layout_short=layout_short,
        ordinal=ordinal,
    )

    technique_profile = build_technique_profile(
        techniques_used=candidate.techniques_used,
        technique_histogram=candidate.technique_histogram,
    )

    resolved_pattern = _resolve_pattern_metadata(
        candidate=candidate,
        pattern_lookup=pattern_lookup,
    )

    resolved_pattern_id = resolved_pattern["pattern_id"]
    resolved_pattern_name = resolved_pattern["pattern_name"]
    resolved_pattern_family_id = resolved_pattern["pattern_family_id"]
    resolved_pattern_family_name = resolved_pattern["pattern_family_name"]
    resolved_symmetry_type = resolved_pattern["symmetry_type"]

    pattern_for_metadata = _build_pattern_stub(resolved_pattern)

    clue_count = _count_clues(candidate.givens81)
    solution_signature = build_solution_signature(candidate.solution81)
    difficulty_payload = enrich_candidate_difficulty(
        techniques_used=technique_profile.techniques_used,
    )

    enriched = enrich_puzzle_metadata(
        weight=candidate.weight,
        technique_count=technique_profile.technique_count,
        techniques_used=technique_profile.techniques_used,
        pattern=pattern_for_metadata,
        layout_type=layout_type,
        clue_count=clue_count,
        display_code=record_id,
    )

    print_header = build_print_header(
        display_code=record_id,
        difficulty_label=str(difficulty_payload["puzzle_difficulty"]),
        weight=int(candidate.weight),
    )

    title = (candidate.title or "").strip() or f"Candidate {record_id}"

    return PuzzleRecord(
        record_id=record_id,
        candidate_status=DEFAULT_CANDIDATE_STATUS,
        solution_signature=solution_signature,
        library_id=library_id,
        aisle_id=None,
        book_id=None,
        section_id=None,
        section_code=None,
        local_puzzle_code=None,
        friendly_puzzle_id=None,
        puzzle_uid=None,
        title=title,
        subtitle=candidate.subtitle or "Classic 9x9",
        layout_type=layout_type,
        grid_size=grid_size,
        charset=charset,
        givens81=candidate.givens81,
        solution81=candidate.solution81,

        pattern_id=resolved_pattern_id,
        pattern_name=resolved_pattern_name,
        pattern_family_id=resolved_pattern_family_id,
        pattern_family_name=resolved_pattern_family_name,
        clue_count=clue_count,

        symmetry_type=resolved_symmetry_type,
        is_unique=bool(candidate.is_unique),
        is_human_solvable=bool(candidate.is_human_solvable),
        generation_method=candidate.generation_method or "pattern_fill",
        generation_seed=int(candidate.generation_seed) if candidate.generation_seed is not None else None,
        generator_version=candidate.generator_version,
        weight=int(candidate.weight),
        difficulty_label=str(enriched["difficulty_label"]),
        difficulty_band_code=str(enriched["difficulty_band_code"]),
        techniques_difficulty=list(difficulty_payload["techniques_difficulty"]),
        puzzle_difficulty=str(difficulty_payload["puzzle_difficulty"]),
        difficulty_version=str(difficulty_payload["difficulty_version"]),
        technique_count=technique_profile.technique_count,
        techniques_used=technique_profile.techniques_used,
        technique_histogram=technique_profile.technique_histogram,
        featured_technique=technique_profile.featured_technique,
        technique_prominence_score=technique_profile.technique_prominence_score,
        app_search_tags=list(enriched["app_search_tags"]),
        book_page_number=None,
        position_in_section=None,
        position_in_book=None,
        print_header=print_header,
        publication_status="draft",
        reuse_policy="book_exclusive",
        created_at=created_at,
        updated_at=updated_at,
    )