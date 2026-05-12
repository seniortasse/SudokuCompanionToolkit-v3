
import itertools

from generator.model import DIMENSIONS


def get_idxs_in_dimensions(options, i1, i2):
    idxs = sorted(set(itertools.chain.from_iterable(
        (
            options.get_idxs_in_dim(dim, i1, i2)
            for dim in DIMENSIONS
        )
    )))
    idxs.remove((i1, i2))
    assert len(idxs) == options.size * 3 - sum((idx[0] == i1) + (idx[1] == i2) for idx in options.get_idxs_in_dim("box", i1, i2)) - 1
    return idxs
