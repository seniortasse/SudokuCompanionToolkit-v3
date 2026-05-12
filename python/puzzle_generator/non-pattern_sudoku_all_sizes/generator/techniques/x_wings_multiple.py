
import itertools


# A generalisation of x-wings, which was applied on 2 rows + columns, x-wings-multiple is the same technique applied to
#  more than 2 rows/columns

# Algorithm steps (for multiple=3):
#  For each character (one-by-one):
#   - Determine the number of occurrences in each row (or columns)
#   - Find all combinations of 3 rows (or columns) where the number of occurrences is at most 3
#   - Select the combinations where the values occur in at most 3 columns (or rows)
#   - For all such combinations, the other occurences in other rows (or columns) and the same columns (or rows) can be
#     removed


def find_x_wings_multiple(options, chars, multiple, show_logs=False):

    size = options.size

    details = []

    # Row-based
    # TODO Sorting assumes we do not abort before all options are removed, as otherwise this would lead to a biased search
    for char in sorted(chars):
        if show_logs:
            print(f"Trying to find x-wings-{multiple} row-based for '{char}'")
        # idxs_rows = [
        #     idx
        #     for idx, row in options.get_rows()
        #     if sum(
        #         char in options_for_cell
        #         for options_for_cell in row
        #     ) <= multiple
        # ]
        # Create a mapping from row to the columns containing the values
        mapping = {
            idx_row: [
                idx_col
                for idx_col, options_for_cell in enumerate(row)
                if char in options_for_cell
            ]
            for idx_row, row in options.get_rows()
        }
        # Filter relevant idxs
        idxs_rows = [
            idx_row
            for idx_row, idxs_col in mapping.items()
            if 1 <= len(idxs_col) <= multiple
        ]
        if show_logs:
            print(f" Rows with at most {multiple} occurrences of '{char}':", idxs_rows)
        # Extra stop condition to prevent clogging the logs
        if len(idxs_rows) < multiple:
            continue
        combs_idxs_rows = list(itertools.combinations(idxs_rows, multiple))
        if show_logs:
            print(f" Checking whether the cols are lined up for {len(combs_idxs_rows)} combinations of {multiple} rows")
        for comb_idxs_row in combs_idxs_rows:
            all_col_idxs = {
                col_idx
                for idx_row in comb_idxs_row
                for col_idx in mapping[idx_row]
            }
            # TODO Why use <= here and not ==? If not applying x-wings-2 first, this will likely identify wrong-sized
            #  x-wings -> this is indeed what happens, so modified to ==
            if len(all_col_idxs) == multiple:
                name_application = f"x-wings-{multiple} for '{char}' in rows {tuple(sorted(comb_idxs_row))}, with cols {tuple(sorted(all_col_idxs))}"
                if show_logs:
                    print(f"  Found a valid {name_application}")

                removed_chars = []
                for _row_idx in range(size):
                    if _row_idx not in comb_idxs_row:
                        for _col_idx in all_col_idxs:
                            # options_for_cell = options[_row_idx][_col_idx]
                            if char in options[_row_idx][_col_idx]:
                                if show_logs:
                                    print(f"  Remove char '{char}' from options of cell {(_row_idx, _col_idx)}")
                                # options_for_cell.remove(char)
                                removed_chars.append(((_row_idx, _col_idx), char))

                # Add details
                col_idxs = {idx_row: mapping[idx_row] for idx_row in comb_idxs_row}
                application = ("row", char, comb_idxs_row, col_idxs)
                details.append((name_application, application, removed_chars))

    # Col-based (TODO Reuse code above on a rotated grid)
    for char in sorted(chars):
        if show_logs:
            print(f"Trying to find x-wings-{multiple} col-based for '{char}'")
        # Create a mapping from row to the columns containing the values
        mapping = {
            idx_col: [
                idx_row
                for idx_row, options_for_cell in enumerate(col)
                if char in options_for_cell
            ]
            for idx_col, col in options.get_cols()
        }
        # Filter relevant idxs
        idxs_cols = [
            idx_col
            for idx_col, idxs_row in mapping.items()
            if 1 <= len(idxs_row) <= multiple
        ]
        if show_logs:
            print(f" Cols with at most {multiple} occurrences of '{char}':", idxs_cols)
        # Extra stop condition to prevent clogging the logs
        if len(idxs_cols) < multiple:
            continue
        combs_idxs_cols = list(itertools.combinations(idxs_cols, multiple))
        if show_logs:
            print(f" Checking whether the rows are lined up for {len(combs_idxs_cols)} combinations of {multiple} cols")
        for comb_idxs_col in combs_idxs_cols:
            all_row_idxs = {
                row_idx
                for idx_col in comb_idxs_col
                for row_idx in mapping[idx_col]
            }
            if len(all_row_idxs) == multiple:
                name_application = f"x-wings-{multiple} for '{char}' in cols {tuple(sorted(comb_idxs_col))}, with rows {tuple(sorted(all_row_idxs))}"
                if show_logs:
                    print(f"  Found a valid {name_application}")

                removed_chars = []
                for _col_idx in range(size):
                    if _col_idx not in comb_idxs_col:
                        for _row_idx in all_row_idxs:
                            # options_for_cell = options[_row_idx][_col_idx]
                            if char in options[_row_idx][_col_idx]:
                                if show_logs:
                                    print(f"  Remove char '{char}' from options of cell {(_row_idx, _col_idx)}")
                                # options_for_cell.remove(char)
                                removed_chars.append(((_row_idx, _col_idx), char))

                # Add details
                row_idxs = {idx_col: mapping[idx_col] for idx_col in comb_idxs_col}
                application = ("col", char, comb_idxs_col, row_idxs)
                details.append((name_application, application, removed_chars))

    # TODONE Can count the number of removed options once at the end, by comparing the initial state with the final state;
    #  This removes some unnecessary complexity, as we do not have to keep track of an extra value

    # TODONE Do not need to return anything when code is cleaned -> Need application details

    return details


def find_x_wings_3(options, chars, show_logs=False):
    return find_x_wings_multiple(options, chars, multiple=3, show_logs=show_logs)


def find_x_wings_4(options, chars, show_logs=False):
    return find_x_wings_multiple(options, chars, multiple=4, show_logs=show_logs)
