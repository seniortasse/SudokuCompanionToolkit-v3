

def subword_overlaps_idx(subword, placement_i1, placement_i2, rotation, target_i1, target_i2):

    idxs = get_idxs_for_subword_placement(subword, rotation, (placement_i1, placement_i2))

    overlaps = (target_i1, target_i2) in idxs

    return overlaps


def get_idxs_for_subword_placement(subword, rotation, idx):

    i1, i2 = idx

    idxs = []
    for i in range(len(subword)):

        if rotation in ["hor-lr", "hor-rl"]:
            next_i1 = i1
        elif rotation in ["ver-ud", "diag-lrd"]:
            next_i1 = i1 + i
        elif rotation in ["ver-du", "diag-lru"]:
            next_i1 = i1 - i
        else:
            raise Exception(f"rotation {rotation} not implemented in subword_overlaps_idx")

        if rotation in ["ver-ud", "ver-du"]:
            next_i2 = i2
        elif rotation in ["hor-lr", "diag-lrd", "diag-lru"]:
            next_i2 = i2 + i
        elif rotation in ["hor-rl"]:
            next_i2 = i2 - i
        else:
            raise Exception(f"rotation {rotation} not implemented in subword_overlaps_idx")

        next_idx = (next_i1, next_i2)
        idxs.append(next_idx)

    return idxs
