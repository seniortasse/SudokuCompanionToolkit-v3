
from collections import defaultdict


def fits_diagonally(dims, subword, i1, i2, is_lrd):
    """
    Determine whether the subword fits at the given position diag-lrd,
     to be used in model to detemrine which vars to create
    """

    box_height, box_width, size = dims

    if is_lrd:
        fits = 0 <= i1 <= size - len(subword) and 0 <= i2 <= size - len(subword)
    else:
        fits = len(subword) - 1 <= i1 < size and 0 <= i2 <= size - len(subword)

    if not fits:
        return False

    # Extra check for duplicate chars
    has_duplicate_chars = len(subword) != len(set(subword))
    if has_duplicate_chars:
        # No duplicate chars in the same box
        chars_in_boxes = defaultdict(list)
        for i in range(len(subword)):
            _row_idx = i1 + i if is_lrd else i1 - i
            _col_idx = i2 + i
            box_idxs = (_row_idx // box_height, _col_idx // box_width)
            chars_in_boxes[box_idxs].append(subword[i])
        fits = all(len(chars_for_box) == len(set(chars_for_box)) for chars_for_box in chars_in_boxes.values())

    return fits


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


def subword_overlaps_idx(subword, placement_i1, placement_i2, rotation, target_i1, target_i2):

    idxs = get_idxs_for_subword_placement(subword, rotation, (placement_i1, placement_i2))

    overlaps = (target_i1, target_i2) in idxs

    return overlaps
