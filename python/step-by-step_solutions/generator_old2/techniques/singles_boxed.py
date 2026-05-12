

# A technique related to singles-pointing
#  singles-pointing:
#   When a value can only occur in one row/col of a certain box, it removes all options from that same row/col in the
#   other boxes (so the dynamic is box -> row/col)
#  singles-boxed:
#   when a value can only occur in one box for a certain row/col, it removes all other options for that value from that
#   box (dynamic is row/col -> box, the reverse of singles-pointing, and therefore a natural complementing technique)


def find_singles_boxed(options, chars, show_logs=False):

    # Steps:
    #  1 For each row/col, determine whether there is a character which can only occur in a one box
    #  2 Remove all other options in that box for that character

    box_height, box_width, size = options.box_height, options.box_width, options.size

    details = []

    # Row-based
    for i1, row in options.get_rows():
        for char in chars:

            # 1 Identify in which boxes the value occurs
            # Implementation taking into account efficiency:
            #  - Determine the box idxs of the first occurrence
            #  - If all next occurrences have the same box idx, it is a match
            first_idx = None
            for i2, e in enumerate(row):
                if char in e:
                    b1, b2 = i1 // box_height, i2 // box_width
                    if first_idx is None:
                        first_idx = (b1, b2)
                    else:
                        # Options are in different boxes
                        if (b1, b2) != first_idx:
                            first_idx = None
                            break

            # An occurrence was found and no other occurrence was in a different box
            if first_idx is not None:
                (b1, b2) = first_idx

                name_application = f"singles-boxed for char '{char}' in row {i1 + 1} and box {(b1 + 1, b2 + 1)}"
                if show_logs:
                    print(f"Identified {name_application}, removing all other options from the box")

                # 2 Remove all other occurrences in the box besides the current row
                removed_chars = []
                for _i1 in range(box_height):
                    for _i2 in range(box_width):
                        i1_box = b1 * box_height + _i1
                        i2_box = b2 * box_width + _i2
                        if i1_box != i1:
                            if char in options[i1_box][i2_box]:
                                # options[i1_box][i2_box].remove(char)
                                if show_logs:
                                    print(f"Remove {char} from {(i1_box + 1, i2_box + 1)}")
                                removed_chars.append(((i1_box, i2_box), char))

                # Add details
                idx_row = i1
                idx_box = first_idx
                idxs_char = [(i1, i2) for i2 in range(size) if char in options[i1][i2]]  # TODO Merge with main logic
                application = ("row", char, idx_row, idx_box, idxs_char)
                details.append((name_application, application, removed_chars))

    # Col-based
    for i2, col in options.get_cols():
        for char in chars:

            # 1 Identify in which boxes the value occurs
            first_idx = None
            for i1, e in enumerate(col):
                if char in e:
                    b1, b2 = i1 // box_height, i2 // box_width
                    if first_idx is None:
                        first_idx = (b1, b2)
                    else:
                        # Options are in different boxes
                        if (b1, b2) != first_idx:
                            first_idx = None
                            break

            # An occurrence was found and no other occurrence was in a different box
            if first_idx is not None:
                (b1, b2) = first_idx

                name_application = f"singles-boxed for char '{char}' in col {i2 + 1} and box {(b1 + 1, b2 + 1)}"
                if show_logs:
                    print(f"Identified {name_application}, removing all other options from the box")

                # 2 Remove all other occurrences in the box besides the current row
                removed_chars = []
                for _i1 in range(box_height):
                    for _i2 in range(box_width):
                        i1_box = b1 * box_height + _i1
                        i2_box = b2 * box_width + _i2
                        if i2_box != i2:
                            if char in options[i1_box][i2_box]:
                                # options[i1_box][i2_box].remove(char)
                                if show_logs:
                                    print(f"Remove {char} from {(i1_box + 1, i2_box + 1)}")
                                removed_chars.append(((i1_box, i2_box), char))

                # Add details
                idx_col = i2
                idx_box = first_idx
                idxs_char = [(i1, i2) for i1 in range(size) if char in options[i1][i2]]  # TODO Merge with main logic
                application = ("col", char, idx_col, idx_box, idxs_char)
                details.append((name_application, application, removed_chars))

    return details
