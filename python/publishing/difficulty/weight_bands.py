from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class WeightBand:
    code: str
    label: str
    min_weight: int
    max_weight: Optional[int]

    def contains(self, weight: int) -> bool:
        if weight < self.min_weight:
            return False
        if self.max_weight is not None and weight > self.max_weight:
            return False
        return True


DEFAULT_CLASSIC9_BANDS: List[WeightBand] = [
    WeightBand(code="D1", label="easy", min_weight=0, max_weight=120),
    WeightBand(code="D2", label="medium", min_weight=121, max_weight=240),
    WeightBand(code="D3", label="hard", min_weight=241, max_weight=420),
    WeightBand(code="D4", label="expert", min_weight=421, max_weight=None),
]