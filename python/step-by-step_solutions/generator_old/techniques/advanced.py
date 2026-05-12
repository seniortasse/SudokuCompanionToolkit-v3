

def get_idxs_in_dimensions(dims, i1, i2):
    box_height, box_width, size = dims
    idxs_row = [(i1, _i2) for _i2 in range(size) if i2 != _i2]
    idxs_col = [(_i1, i2) for _i1 in range(size) if i1 != _i1]
    b1, b2 = i1 // box_height, i2 // box_width
    idxs_box = [
        (_i1, _i2)
        for _i1 in range(b1 * box_height, (b1 + 1) * box_height)
        for _i2 in range(b2 * box_width, (b2 + 1) * box_width)
        if (i1, i2) != (_i1, _i2)
    ]
    idxs = idxs_row + idxs_col + idxs_box
    idxs = list(set(idxs))  # Make sure there are no duplicate idxs
    assert len(idxs) == 3 * size - (box_height + box_width) - 1
    return idxs
