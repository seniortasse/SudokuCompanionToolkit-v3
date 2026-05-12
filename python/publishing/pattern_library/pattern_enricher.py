from __future__ import annotations

from dataclasses import replace
from typing import List, Optional

from python.publishing.pattern_library.families import infer_visual_family, infer_symmetry_type
from python.publishing.pattern_library.pattern_identity import build_canonical_mask_signature


def normalize_pattern_slug(name: str) -> str:
    raw = str(name).strip().lower()
    out = []
    prev_dash = False
    for ch in raw:
        if ch.isalnum():
            out.append(ch)
            prev_dash = False
        else:
            if not prev_dash:
                out.append("-")
                prev_dash = True
    slug = "".join(out).strip("-")
    return slug or "pattern"


def _normalize_aliases(values: Optional[List[str]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values or []:
        alias = str(value).strip()
        if not alias:
            continue
        key = alias.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(alias)
    return out


def _normalize_family_id(value: Optional[str], fallback_family_name: str) -> str:
    raw = str(value or fallback_family_name).strip().lower()
    out = []
    prev_dash = False
    for ch in raw:
        if ch.isalnum():
            out.append(ch)
            prev_dash = False
        else:
            if not prev_dash:
                out.append("-")
                prev_dash = True
    family_id = "".join(out).strip("-")
    return family_id or "uncategorized"


def _score_print(mask81: str) -> float:
    clue_count = sum(1 for ch in mask81 if ch == "1")
    if 26 <= clue_count <= 38:
        return 0.92
    if 22 <= clue_count <= 45:
        return 0.84
    return 0.74


def _score_legibility(mask81: str) -> float:
    clue_count = sum(1 for ch in mask81 if ch == "1")
    if 28 <= clue_count <= 42:
        return 0.90
    if 20 <= clue_count <= 55:
        return 0.82
    return 0.70


def _score_aesthetic(mask81: str, tags: Optional[List[str]] = None) -> float:
    symmetry = infer_symmetry_type(mask81)
    bonus = 0.05 if symmetry != "none" else 0.0
    if tags and any(str(t).strip().lower() in {"heart", "diamond", "spiral"} for t in tags):
        bonus += 0.03
    return min(0.95, 0.80 + bonus)


def enrich_pattern_record(
    pattern,
    *,
    force_slug: bool = True,
    infer_family: bool = True,
    infer_symmetry: bool = True,
    infer_scores: bool = True,
):
    slug = normalize_pattern_slug(pattern.name) if force_slug else pattern.slug
    visual_family = (
        infer_visual_family(pattern.mask81, explicit_tags=pattern.tags)
        if infer_family
        else pattern.visual_family
    )
    symmetry_type = infer_symmetry_type(pattern.mask81) if infer_symmetry else pattern.symmetry_type
    canonical_mask_signature = build_canonical_mask_signature(pattern.mask81)
    family_name = pattern.family_name or visual_family
    family_id = _normalize_family_id(pattern.family_id, family_name)
    aliases = _normalize_aliases(pattern.aliases)

    print_score = _score_print(pattern.mask81) if infer_scores and pattern.print_score is None else pattern.print_score
    legibility_score = (
        _score_legibility(pattern.mask81)
        if infer_scores and pattern.legibility_score is None
        else pattern.legibility_score
    )
    aesthetic_score = (
        _score_aesthetic(pattern.mask81, pattern.tags)
        if infer_scores and pattern.aesthetic_score is None
        else pattern.aesthetic_score
    )

    return replace(
        pattern,
        slug=slug,
        aliases=aliases,
        visual_family=visual_family,
        family_id=family_id,
        family_name=family_name,
        canonical_mask_signature=canonical_mask_signature,
        symmetry_type=symmetry_type,
        status=str(pattern.status or "active").strip().lower() or "active",
        print_score=print_score,
        legibility_score=legibility_score,
        aesthetic_score=aesthetic_score,
    )