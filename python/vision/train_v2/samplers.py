from __future__ import annotations
import math, random
from typing import List, Sequence, Optional, Iterator
from torch.utils.data import Sampler

class Pools:
    """
    Build head-aware index pools from a manifest-like list of rows:
      row = (path, given_digit, solution_digit, candidates_list, source)
    """
    def __init__(self, rows: Sequence):
        self.idx_Gp: List[int] = []      # given != 0
        self.idx_Sp: List[int] = []      # solution != 0
        self.idx_Cp: List[int] = []      # len(candidates) > 0
        self.idx_Empty: List[int] = []   # none present
        self.idx_Mixed: List[int] = []   # >= 2 positives

        for i, row in enumerate(rows):
            _, given, sol, cand, _ = row
            gpos = (given != 0)
            spos = (sol != 0)
            cpos = (len(cand or []) > 0)
            positives = int(gpos) + int(spos) + int(cpos)

            if gpos: self.idx_Gp.append(i)
            if spos: self.idx_Sp.append(i)
            if cpos: self.idx_Cp.append(i)

            if positives == 0:
                self.idx_Empty.append(i)
            elif positives >= 2:
                self.idx_Mixed.append(i)

        # Fallbacks: keep everything robust even if some pools are tiny
        # (sampling uses replacement anyway).
        if not self.idx_Mixed:
            self.idx_Mixed = list(self.idx_Empty)  # harmless substitute

    def __repr__(self) -> str:
        return (f"Pools(G+={len(self.idx_Gp)}, S+={len(self.idx_Sp)}, "
                f"C+={len(self.idx_Cp)}, Empty={len(self.idx_Empty)}, Mixed={len(self.idx_Mixed)})")


def _choices(indexes: List[int], k: int) -> List[int]:
    if not indexes:
        return []
    # with replacement
    return [indexes[random.randrange(len(indexes))] for _ in range(k)]


class HeadAwareBatchSampler(Sampler[List[int]]):
    """
    Composes each batch with a fixed recipe so every head
    gets positives + negatives on every step.

    Example defaults for batch_size=128:
      k_g=32, k_s=32, k_c=32, k_e=24, k_m=8

    steps_per_epoch: by default ceil(N / batch_size).
    """
    def __init__(
        self,
        pools: Pools,
        dataset_size: int,
        batch_size: int = 128,
        k_g: int = 32,
        k_s: int = 32,
        k_c: int = 32,
        k_e: int = 24,
        k_m: int = 8,
        steps_per_epoch: Optional[int] = None,
        hard_idx: Optional[List[int]] = None,
        k_hard: int = 0,
    ):
        assert k_g + k_s + k_c + k_e + k_m + k_hard == batch_size, \
            "Sum of k_* must equal batch_size"
        self.pools = pools
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.k_g, self.k_s, self.k_c = k_g, k_s, k_c
        self.k_e, self.k_m, self.k_hard = k_e, k_m, k_hard
        self.hard_idx = hard_idx or []

        if steps_per_epoch is None:
            steps_per_epoch = math.ceil(max(1, dataset_size) / float(batch_size))
        self._steps = steps_per_epoch

    def __len__(self) -> int:
        return self._steps

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self._steps):
            batch = []
            batch += _choices(self.pools.idx_Gp, self.k_g)
            batch += _choices(self.pools.idx_Sp, self.k_s)
            batch += _choices(self.pools.idx_Cp, self.k_c)
            batch += _choices(self.pools.idx_Empty, self.k_e)
            batch += _choices(self.pools.idx_Mixed, self.k_m)
            if self.k_hard > 0:
                batch += _choices(self.hard_idx, self.k_hard)
            random.shuffle(batch)
            yield batch