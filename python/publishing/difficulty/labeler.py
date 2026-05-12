from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .weight_bands import DEFAULT_CLASSIC9_BANDS, WeightBand


@dataclass(frozen=True)
class DifficultyBand:
    code: str
    label: str
    min_weight: int
    max_weight: int | None


def classify_weight(
    weight: int,
    *,
    bands: Iterable[WeightBand] = DEFAULT_CLASSIC9_BANDS,
) -> DifficultyBand:
    for band in bands:
        if band.contains(weight):
            return DifficultyBand(
                code=band.code,
                label=band.label,
                min_weight=band.min_weight,
                max_weight=band.max_weight,
            )

    fallback = list(bands)[-1]
    return DifficultyBand(
        code=fallback.code,
        label=fallback.label,
        min_weight=fallback.min_weight,
        max_weight=fallback.max_weight,
    )


def make_effort_label(weight: int) -> str:
    return f"Effort {int(weight)}"