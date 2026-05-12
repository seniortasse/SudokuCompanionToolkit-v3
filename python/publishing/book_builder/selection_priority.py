from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from python.publishing.schemas.models import PuzzleRecord
from python.publishing.techniques.technique_catalog import normalize_technique_id


def _norm(value: Any) -> str:
    return normalize_technique_id(str(value))


def _parse_combo_spec(raw_values: Sequence[Any]) -> List[Tuple[str, ...]]:
    combos: List[Tuple[str, ...]] = []
    for raw in list(raw_values or []):
        if isinstance(raw, str):
            parts = [_norm(x) for x in raw.split("+") if _norm(x)]
        else:
            parts = [_norm(x) for x in list(raw or []) if _norm(x)]
        if len(parts) >= 2:
            combos.append(tuple(sorted(set(parts))))
    return combos


def _candidate_pool_stats(records: Sequence[PuzzleRecord]) -> Dict[str, Counter]:
    pattern_counts = Counter()
    family_counts = Counter()
    clue_counts = Counter()

    for record in records:
        pattern_key = str(record.pattern_id or "")
        family_key = str(getattr(record, "pattern_family_id", None) or "")
        clue_key = int(record.clue_count) if record.clue_count is not None else None

        if pattern_key:
            pattern_counts[pattern_key] += 1
        if family_key:
            family_counts[family_key] += 1
        if clue_key is not None:
            clue_counts[clue_key] += 1

    return {
        "pattern_counts": pattern_counts,
        "family_counts": family_counts,
        "clue_counts": clue_counts,
    }


def _record_combo_count(record: PuzzleRecord, combos: Sequence[Tuple[str, ...]]) -> int:
    if not combos:
        return 0

    used = {_norm(t) for t in list(record.techniques_used or [])}
    count = 0
    for combo in combos:
        if all(token in used for token in combo):
            count += 1
    return count


def _record_disliked_penalty(
    record: PuzzleRecord,
    disliked_techniques: Sequence[str],
    *,
    presence_penalty: float,
    occurrence_penalty: float,
) -> float:
    if not disliked_techniques:
        return 0.0

    histogram = {_norm(k): int(v) for k, v in dict(record.technique_histogram or {}).items()}
    used = {_norm(t) for t in list(record.techniques_used or [])}

    penalty = 0.0
    for technique in disliked_techniques:
        name = _norm(technique)
        if name in used:
            penalty += float(presence_penalty)
            penalty += float(histogram.get(name, 0)) * float(occurrence_penalty)
    return penalty


def _record_rarity_bonus(
    record: PuzzleRecord,
    *,
    pattern_counts: Counter,
    family_counts: Counter,
    clue_counts: Counter,
    pattern_weight: float,
    family_weight: float,
    clue_weight: float,
) -> float:
    bonus = 0.0

    pattern_key = str(record.pattern_id or "")
    if pattern_key and pattern_counts.get(pattern_key, 0) > 0:
        bonus += float(pattern_weight) / float(pattern_counts[pattern_key])

    family_key = str(getattr(record, "pattern_family_id", None) or "")
    if family_key and family_counts.get(family_key, 0) > 0:
        bonus += float(family_weight) / float(family_counts[family_key])

    clue_key = int(record.clue_count) if record.clue_count is not None else None
    if clue_key is not None and clue_counts.get(clue_key, 0) > 0:
        bonus += float(clue_weight) / float(clue_counts[clue_key])

    return bonus


def apply_selection_priority(
    *,
    records: Iterable[PuzzleRecord],
    section_criteria: Dict[str, Any] | None,
) -> List[PuzzleRecord]:
    ordered = list(records)
    criteria = dict(section_criteria or {})
    soft = dict(criteria.get("soft_preferences") or {})

    if not soft:
        return ordered

    pattern_diversity_weight = float(soft.get("pattern_diversity_weight", 0.0))
    family_diversity_weight = float(soft.get("family_diversity_weight", 0.0))
    clue_diversity_weight = float(soft.get("clue_diversity_weight", 0.0))

    disliked_techniques = [_norm(x) for x in list(soft.get("disliked_techniques", []))]
    disliked_presence_penalty = float(soft.get("disliked_technique_presence_penalty", 0.0))
    disliked_occurrence_penalty = float(soft.get("disliked_technique_occurrence_penalty", 0.0))

    preferred_combos = _parse_combo_spec(soft.get("preferred_technique_combos", []))
    preferred_combo_bonus = float(soft.get("preferred_combo_bonus", 0.0))

    stats = _candidate_pool_stats(ordered)

    def score(record: PuzzleRecord) -> float:
        total = 0.0

        total += _record_rarity_bonus(
            record,
            pattern_counts=stats["pattern_counts"],
            family_counts=stats["family_counts"],
            clue_counts=stats["clue_counts"],
            pattern_weight=pattern_diversity_weight,
            family_weight=family_diversity_weight,
            clue_weight=clue_diversity_weight,
        )

        combo_count = _record_combo_count(record, preferred_combos)
        total += combo_count * preferred_combo_bonus

        total -= _record_disliked_penalty(
            record,
            disliked_techniques,
            presence_penalty=disliked_presence_penalty,
            occurrence_penalty=disliked_occurrence_penalty,
        )

        return total

    # Stable sort: if records were seeded-shuffled before this call, ties retain that order.
    return sorted(
        ordered,
        key=lambda record: (
            -score(record),
            int(record.weight) if record.weight is not None else 10**9,
            int(record.technique_count) if record.technique_count is not None else 10**9,
            int(record.clue_count) if record.clue_count is not None else 10**9,
            str(record.record_id),
        ),
    )