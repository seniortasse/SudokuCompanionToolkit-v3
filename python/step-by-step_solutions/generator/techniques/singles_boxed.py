

# A technique related to singles-pointing
#  singles-pointing:
#   When a value can only occur in one row/col of a certain box, it removes all options from that same row/col in the
#   other boxes (so the dynamic is box -> row/col)
#  singles-boxed:
#   when a value can only occur in one box for a certain row/col, it removes all other options for that value from that
#   box (dynamic is row/col -> box, the reverse of singles-pointing, and therefore a natural complementing technique)


# TODO The updated implementation is very similar to singles-pointing, and we can probably merge the logic
def find_singles_boxed(options, chars, show_logs=False):

    # Steps:
    #  1 For each row/col, determine whether there is a character which can only occur in a one box
    #  2 Remove all other options in that box for that character

    details = []

    # For each row check whether some value can only occur in a single box
    for idx_row, idxs_for_row in options.idxs_for_dims["row"].items():

        for char in chars:

            idxs_possible = []
            for (i1, i2) in idxs_for_row:
                if char in options[i1][i2]:
                    idxs_possible.append((i1, i2))

            # Identify in which boxes the value can occur
            idxs_boxs_possible = set(options.get_idx_box(*idx) for idx in idxs_possible)
            if len(idxs_boxs_possible) == 1:
                idx_box = idxs_boxs_possible.pop()
                name_application = f"singles-boxed for char '{char}' in row {idx_row + 1} and box {idx_box + 1}"
                if show_logs:
                    print(f"Identified {name_application}, removing all other options from the box")

                # Remove all other occurrences in the box besides the current row
                removed_chars = []
                for (i1, i2) in options.idxs_for_dims["box"][idx_box]:
                    if i1 != idx_row:
                        if char in options[i1][i2]:
                            if show_logs:
                                print(f"Remove '{char}' from {(i1 + 1, i2 + 1)}")
                            removed_chars.append(((i1, i2), char))

                # Add details
                application = ("row", char, idx_row, idx_box, idxs_possible)
                details.append((name_application, application, removed_chars))

    # TODO We can merge the logic with the above for rows, but let's do this in a future update

    # For each col check whether some value can only occur in a single box
    for idx_col, idxs_for_col in options.idxs_for_dims["col"].items():

        for char in chars:

            idxs_possible = []
            for (i1, i2) in idxs_for_col:
                if char in options[i1][i2]:
                    idxs_possible.append((i1, i2))

            # Identify in which boxes the value can occur
            idxs_boxs_possible = set(options.get_idx_box(*idx) for idx in idxs_possible)
            if len(idxs_boxs_possible) == 1:
                idx_box = idxs_boxs_possible.pop()
                name_application = f"singles-boxed for char '{char}' in col {idx_col + 1} and box {idx_box + 1}"
                if show_logs:
                    print(f"Identified {name_application}, removing all other options from the box")

                # Remove all other occurrences in the box besides the current col
                removed_chars = []
                for (i1, i2) in options.idxs_for_dims["box"][idx_box]:
                    if i2 != idx_col:
                        if char in options[i1][i2]:
                            if show_logs:
                                print(f"Remove '{char}' from {(i1 + 1, i2 + 1)}")
                            removed_chars.append(((i1, i2), char))

                # Add details
                application = ("col", char, idx_col, idx_box, idxs_possible)
                details.append((name_application, application, removed_chars))

    return details
