from __future__ import annotations

import random
from typing import Iterable, List

from python.publishing.schemas.models import PuzzleRecord


def apply_seeded_shuffle(
    records: Iterable[PuzzleRecord],
    *,
    random_seed: int | None,
) -> List[PuzzleRecord]:
    out = list(records)
    if random_seed is None:
        return out

    rng = random.Random(int(random_seed))
    rng.shuffle(out)
    return out