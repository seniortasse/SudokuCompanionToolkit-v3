

def create_default_layout_boxes(box_width, box_height):

    size = box_width * box_height

    raw_data = [
        [
            1 + i1 // box_height * box_height + i2 // box_width
            for i2 in range(size)
        ]
        for i1 in range(size)
    ]

    layout_boxes = preprocess_layout_boxes(raw_data, size)

    return layout_boxes


def preprocess_layout_boxes(raw_data, size):

    layout_boxes = {i: [] for i in range(size)}

    for i1 in range(size):
        for i2 in range(size):
            box_id = raw_data[i1][i2]
            assert str(box_id).isdigit(), f"Layout of boxes not properly defined: box ID of cell ({i1 + 1}, {i2 + 1})"
            idx_box = int(box_id) - 1  # Convert from user 1-based to internal 0-based
            assert idx_box in range(size), f"Box ID of cell ({i1 + 1}, {i2 + 1}) invalid"
            layout_boxes[idx_box].append((i1, i2))

    # Check: All boxes contain 'size' cells
    for idx_box, idxs_for_box in layout_boxes.items():
        assert (_len := len(idxs_for_box)) == size, f"Number of cells for box ID '{idx_box + 1}' not correct: {_len}"

    def is_adjacent(idx_1, idx_2):
        assert idx_1 != idx_2
        assert all(len(idx) == 2 for idx in (idx_1, idx_2))
        assert all(isinstance(e, int) for idx in (idx_1, idx_2) for e in idx)
        return (
            idx_1[0] == idx_2[0] and abs(idx_1[1] - idx_2[1]) == 1 or
            idx_1[1] == idx_2[1] and abs(idx_1[0] - idx_2[0]) == 1
        )

    # Check: The cells of all boxes are adjacent
    for idx_box, idxs_for_box in layout_boxes.items():
        idxs_cells_adjacent, idxs_cells_to_check = idxs_for_box[:1], idxs_for_box[1:]
        while idxs_cells_to_check:
            idxs_cells_to_add = []
            for idx in idxs_cells_to_check:
                if any(is_adjacent(idx, _idx) for _idx in idxs_cells_adjacent):
                    idxs_cells_to_add.append(idx)
            assert idxs_cells_to_add, f"Not all cells of box ID '{idx_box + 1}' adjacent: {idxs_cells_to_check}"
            idxs_cells_adjacent.extend(idxs_cells_to_add)
            for idx in idxs_cells_to_add:
                idxs_cells_to_check.remove(idx)

    return layout_boxes
