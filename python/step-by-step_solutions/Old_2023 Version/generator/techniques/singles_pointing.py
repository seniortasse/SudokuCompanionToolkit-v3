
import operator


# Technique description
#  singles-pointing:
#   When a value can only occur in a single row/col in a certain box, this option can be removed from the empty cells
#   in the same row/col outside the box


def find_pointing_singles(options, chars, show_logs=False):

    # Adaptation of the previous version applied to options

    details = []

    # Apply technique based on original (not updated during the process!) options

    # For each box check whether a value can only occur in a single row or col
    for idx_box, idxs_for_box in options.idxs_for_dims["box"].items():

        for char in chars:

            idxs_possible = []
            for (i1, i2) in idxs_for_box:
                if char in options[i1][i2]:
                    idxs_possible.append((i1, i2))

            # Identify row rays
            idxs_rows_possible = set(map(operator.itemgetter(0), idxs_possible))
            if len(idxs_rows_possible) == 1:
                idx_row = idxs_rows_possible.pop()
                # TODO Use technique name in application name, to make sure it is unique
                name_application = f"hor ray for '{char}' in box {idx_box + 1} at row {idx_row + 1}"
                if show_logs:
                    print(f"Identified {name_application}")

                removed_chars = []
                for i2 in range(options.size):
                    # Only remove options for cols outside the box
                    if options.get_idx_box(idx_row, i2) != idx_box:
                        if char in options[idx_row][i2]:
                            if show_logs:
                                print(f"Remove '{char}' from {(idx_row + 1, i2 + 1)}")
                            removed_chars.append(((idx_row, i2), char))

                details.append((name_application, (char, idx_box, idxs_possible, "hor", idx_row), removed_chars))

            # Identify col rays
            idxs_cols_possible = set(map(operator.itemgetter(1), idxs_possible))
            if len(idxs_cols_possible) == 1:
                idx_col = idxs_cols_possible.pop()
                name_application = f"ver ray for '{char}' in box {idx_box + 1} at col {idx_col + 1}"
                if show_logs:
                    print(f"Identified {name_application}")

                removed_chars = []
                for i1 in range(options.size):
                    # Only remove options for rows outside the box
                    if options.get_idx_box(i1, idx_col) != idx_box:
                        if char in options[i1][idx_col]:
                            if show_logs:
                                print(f"Remove '{char}' from {(i1 + 1, idx_col + 1)}")
                            removed_chars.append(((i1, idx_col), char))

                details.append((name_application, (char, idx_box, idxs_possible, "ver", idx_col), removed_chars))

    return details
