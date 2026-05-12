from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Sequence

from python.publishing.techniques.technique_catalog import normalize_technique_id


@dataclass
class TechniqueProfile:
    technique_count: int
    techniques_used: list[str] = field(default_factory=list)
    technique_histogram: Dict[str, int] = field(default_factory=dict)
    featured_technique: str | None = None
    technique_prominence_score: float | None = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "technique_count": self.technique_count,
            "techniques_used": list(self.techniques_used),
            "technique_histogram": dict(self.technique_histogram),
            "featured_technique": self.featured_technique,
            "technique_prominence_score": self.technique_prominence_score,
        }


def _normalize_name(value: str) -> str:
    return normalize_technique_id(value)


def _compute_featured_technique(histogram: Mapping[str, int]) -> tuple[str | None, float | None]:
    if not histogram:
        return None, None

    total = sum(max(0, int(v)) for v in histogram.values())
    if total <= 0:
        return None, None

    ranked = sorted(
        ((name, int(count)) for name, count in histogram.items()),
        key=lambda item: (-item[1], item[0]),
    )
    featured, count = ranked[0]
    prominence = round(float(count) / float(total), 4)
    return featured, prominence


def build_technique_profile(
    *,
    techniques_used: Sequence[str] | None = None,
    technique_histogram: Mapping[str, int] | None = None,
) -> TechniqueProfile:
    normalized_histogram: Dict[str, int] = {}

    if technique_histogram:
        for raw_name, raw_count in technique_histogram.items():
            name = _normalize_name(raw_name)
            count = int(raw_count)
            if count > 0:
                normalized_histogram[name] = normalized_histogram.get(name, 0) + count

    if techniques_used:
        for raw_name in techniques_used:
            name = _normalize_name(raw_name)
            if name not in normalized_histogram:
                normalized_histogram[name] = normalized_histogram.get(name, 0) + 1

    normalized_used = sorted(normalized_histogram.keys())
    technique_count = len(normalized_used)
    featured_technique, prominence = _compute_featured_technique(normalized_histogram)

    return TechniqueProfile(
        technique_count=technique_count,
        techniques_used=normalized_used,
        technique_histogram=normalized_histogram,
        featured_technique=featured_technique,
        technique_prominence_score=prominence,
    )