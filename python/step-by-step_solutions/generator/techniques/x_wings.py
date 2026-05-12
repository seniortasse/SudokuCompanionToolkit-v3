
from collections import Counter


def find_x_wings(options, chars, show_logs=False):

    size = options.size

    details = []

    # We do not need to find pairs, just whether some value is only possible in a square pattern (row/col/box)
    #  -> This relies on doubles being called first, in order to accurately find all options

    # Find squares

    # Apply row-based
    for char in chars:
        options_per_row = []
        for i1 in range(size):
            options_for_row = ()
            for i2 in range(size):
                if char in options[i1][i2]:
                    options_for_row += (i2, )
            options_per_row.append(options_for_row)

        for col_idxs, count in Counter(options_per_row).items():
            if len(col_idxs) == 2 and count == 2:
                row_idxs = [idx for idx, option in enumerate(options_per_row) if option == col_idxs]
                name_application = f"row-based x-wing for '{char}' in rows {tuple(e + 1 for e in row_idxs)} and cols {tuple(e + 1 for e in col_idxs)}"
                if show_logs:
                    print(f"Found {name_application}")

                removed_chars = []
                for row_idx in row_idxs:
                    for i2 in range(size):
                        if i2 not in col_idxs:
                            # print(f"Try to remove {char} from {(row_idx, i2)}")
                            if char in options[row_idx][i2]:
                                # options[row_idx][i2].remove(char)
                                if show_logs:
                                    print(f"Remove {char} from {(row_idx + 1, i2 + 1)}")
                                removed_chars.append(((row_idx, i2), char))
                for col_idx in col_idxs:
                    for i1 in range(size):
                        if i1 not in row_idxs:
                            # print(f"Try to remove {char} from {(i1, col_idx)}")
                            if char in options[i1][col_idx]:
                                # options[i1][col_idx].remove(char)
                                if show_logs:
                                    print(f"Remove {char} from {(i1 + 1, col_idx + 1)}")
                                removed_chars.append(((i1, col_idx), char))

                # Add details
                # Note: One call of this function applies the technique multiple times, every found value is also
                #  found for the next application making the details incorrect; Therefore we store the application
                #  details together with the removed options as for all techniques
                # details = ("row", row_idxs, col_idxs, char)
                # new_values = [(*new_value[:2], details) for new_value in new_values]
                application = ("row", char, row_idxs, col_idxs)
                details.append((name_application, application, removed_chars))

    # Apply col-based
    for char in chars:
        options_per_col = []
        for i2 in range(size):
            options_for_col = ()
            for i1 in range(size):
                if char in options[i1][i2]:
                    options_for_col += (i1, )
            options_per_col.append(options_for_col)

        for row_idxs, count in Counter(options_per_col).items():
            if len(row_idxs) == 2 and count == 2:
                col_idxs = [idx for idx, option in enumerate(options_per_col) if option == row_idxs]
                name_application = f"col-based x-wing for '{char}' in cols {tuple(e + 1 for e in col_idxs)} and rows {tuple(e + 1 for e in row_idxs)}"
                if show_logs:
                    print(f"Found {name_application}")

                removed_chars = []
                for row_idx in row_idxs:
                    for i2 in range(size):
                        if i2 not in col_idxs:
                            # print(f"Try to remove {char} from {(row_idx, i2)}")
                            if char in options[row_idx][i2]:
                                # options[row_idx][i2].remove(char)
                                if show_logs:
                                    print(f"Remove {char} from {(row_idx + 1, i2 + 1)}")
                                removed_chars.append(((row_idx, i2), char))
                for col_idx in col_idxs:
                    for i1 in range(size):
                        if i1 not in row_idxs:
                            # print(f"Try to remove {char} from {(i1, col_idx)}")
                            if char in options[i1][col_idx]:
                                # options[i1][col_idx].remove(char)
                                if show_logs:
                                    print(f"Remove {char} from {(i1 + 1, col_idx + 1)}")
                                removed_chars.append(((i1, col_idx), char))

                # Add details
                # details = ("col", row_idxs, col_idxs, char)
                # new_values = [(*new_value[:2], details) for new_value in new_values]
                application = ("col", char, row_idxs, col_idxs)
                details.append((name_application, application, removed_chars))

    return details
