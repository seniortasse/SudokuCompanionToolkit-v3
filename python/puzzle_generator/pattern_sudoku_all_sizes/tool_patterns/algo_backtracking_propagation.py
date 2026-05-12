
import itertools
import random

from generator.model import Instance


EMPTY_CHAR = ' '


class BadInitialisationException(Exception):
    pass


MAX_NUMBER_INVALID_PROPAGATIONS_BEFORE_RESTART = 50


# TODO Reuse from generator
def _identify_cells_with_single_options(pattern: Instance, chars, cell_options_copy, show_logs=False):

    size = pattern.size

    cells_with_single_option = []

    # Based on cell only
    for _idx, options in cell_options_copy.items():
        if len(options) == 1:
            if show_logs:
                print(f"Identified {options[0]} as being the only option in cell {_idx}")
            cells_with_single_option.append((_idx, options[0]))

    # Based on row
    for _row_idx in range(size):
        for _char in chars:
            single_idx = None
            for _col_idx in range(size):
                if _char in cell_options_copy[(_row_idx, _col_idx)]:
                    if single_idx is None:
                        single_idx = (_row_idx, _col_idx)
                    else:
                        single_idx = None
                        break
            if single_idx:
                if show_logs:
                    print(f"Identified {_char} as being the only option in row {_row_idx} at {single_idx}")
                cells_with_single_option.append((single_idx, _char))

    # Based on col
    for _col_idx in range(size):
        for _char in chars:
            single_idx = None
            for _row_idx in range(size):
                if _char in cell_options_copy[(_row_idx, _col_idx)]:
                    if single_idx is None:
                        single_idx = (_row_idx, _col_idx)
                    else:
                        single_idx = None
                        break
            if single_idx:
                if show_logs:
                    print(f"Identified {_char} as being the only option in col {_col_idx} at {single_idx}")
                cells_with_single_option.append((single_idx, _char))

    # Based on box
    for idx_box, idxs_for_box in pattern.idxs_for_dims["box"].items():
        for char in chars:
            single_idx = None
            for idx in idxs_for_box:
                if char in cell_options_copy[idx]:
                    if single_idx is None:
                        single_idx = idx
                    else:
                        single_idx = None
                        break
            if single_idx:
                if show_logs:
                    print(f"Identified {char} as being the only option in box {idx_box} at {single_idx}")
                cells_with_single_option.append((single_idx, char))

    # Make sure to only try to add each value once
    cells_with_single_option = sorted(set(cells_with_single_option))

    return cells_with_single_option


def generate_grids_using_propagation(chars, pattern: Instance, show_logs=False):

    size = pattern.size

    # Dimension checks
    assert len(chars) == size
    assert len(pattern) == size and all(len(row) == size for row in pattern)

    flat_pattern = list(itertools.chain(*[row for row in pattern]))

    counter_invalid_propagations = 0

    # TODO Make sure the logs handle the yield / yield from structure properly
    logs = {}
    logs["steps"] = []

    def recursion(flat_grid, data, idx):
        """
        Steps:
         - Fill in one of the optional values in the next cell of the pattern
         - Propagate constraints
         - Determine whether the result can still be feasible (see if this can be integrated with the propagation, as
           we might want to prematurely abort the propagation if we can easily check that there is an issue)
        """

        if show_logs:
            print(f"Recursion step {idx}")

        # Stop condition
        if idx == size ** 2:
            # Note: This is the only place where an actual value is returned, other yields are "yield from" and are only
            #  recursive function calls
            yield flat_grid, logs
        # Continuation condition: The current cell does not need to be filled
        elif not flat_pattern[idx]:
            yield from recursion(flat_grid, data, idx + 1)
        # Extra continuation condition: We might have already filled in the value with propagation
        elif flat_grid[idx] != EMPTY_CHAR:
            yield from recursion(flat_grid, data, idx + 1)
        else:

            (cell_options, rows_remaining_values, cols_remaining_values, boxs_remaining_values) = data

            # Pre-process idxs
            new_row_idx = idx // size
            new_col_idx = idx % size
            new_box_idx = pattern.get_idx_box(new_row_idx, new_col_idx)
            new_cell_idx = (new_row_idx, new_col_idx)

            # Note: These have already been pre-shuffled before starting the recursion, so they do not have to be
            #  shuffled again at every step
            possible_chars = cell_options[new_cell_idx]

            for new_char in possible_chars:

                if show_logs:
                    print(f"Try to fill in {new_char} at {new_cell_idx} for grid:")
                    for row in [flat_grid[i * size:(i + 1) * size] for i in range(size)]: print(row)
                    print()

                flat_grid_copy = flat_grid.copy()
                cell_options_copy = cell_options.copy()
                rows_remaining_values_copy = rows_remaining_values.copy()
                cols_remaining_values_copy = cols_remaining_values.copy()
                boxs_remaining_values_copy = boxs_remaining_values.copy()

                # Used for logs
                idxs_propagated = []

                # 2 Propagate

                # Initialise the propagation recursion by filling in the new value
                cell_options_copy[new_cell_idx] = [new_char]

                is_valid_propagation = True
                while is_valid_propagation:

                    cells_with_single_option = _identify_cells_with_single_options(pattern, chars, cell_options_copy, show_logs=show_logs)

                    if len(cells_with_single_option) == 0:
                        break

                    for (cell_idx, char) in cells_with_single_option:

                        (row_idx, col_idx) = cell_idx
                        box_idx = pattern.get_idx_box(row_idx, col_idx)

                        if show_logs:
                            if cell_idx == new_cell_idx:
                                print(f"Fill {char} at {cell_idx}")
                            else:
                                print(f"Propagate {char} at {cell_idx}")

                        # 1 Fill in value
                        flat_grid_copy[cell_idx[0] * size + cell_idx[1]] = char

                        # Add to logs
                        idxs_propagated.append(cell_idx)

                        # TODO This can all be reused from generator, for which no extra logic is needed

                        # Remove options from cell
                        cell_options_copy[cell_idx] = []

                        # Remove from cells in row
                        for _col_idx in range(size):
                            key = (row_idx, _col_idx)
                            # This is expected to be more efficient: Only need to copy if the value is present
                            if char in cell_options_copy[key]:
                                cell_options_copy[key] = cell_options_copy[key].copy()
                                cell_options_copy[key].remove(char)

                        # Remove from cells in col
                        for _row_idx in range(size):
                            key = (_row_idx, col_idx)
                            if char in cell_options_copy[key]:
                                cell_options_copy[key] = cell_options_copy[key].copy()
                                cell_options_copy[key].remove(char)

                        # Remove from cells in box
                        # TODO Create a shared function which generates all cell indices for a box index with dimensions
                        for _idx in pattern.idxs_for_dims["box"][box_idx]:
                            key = _idx
                            if char in cell_options_copy[key]:
                                cell_options_copy[key] = cell_options_copy[key].copy()
                                cell_options_copy[key].remove(char)

                        try:
                            rows_remaining_values_copy[row_idx] = rows_remaining_values_copy[row_idx].copy()
                            rows_remaining_values_copy[row_idx].remove(char)
                        except ValueError:
                            # Note: This only happens when 2 conflicting values are found in the same propagation step
                            if show_logs:
                                print(f"Value {char} could not be filled at {cell_idx} as it was already present in row {row_idx}")
                            is_valid_propagation = False
                            break

                        try:
                            cols_remaining_values_copy[col_idx] = cols_remaining_values_copy[col_idx].copy()
                            cols_remaining_values_copy[col_idx].remove(char)
                        except ValueError:
                            if show_logs:
                                print(f"Value {char} could not be filled at {cell_idx} as it was already present in col {col_idx}")
                            is_valid_propagation = False
                            break

                        try:
                            boxs_remaining_values_copy[box_idx] = boxs_remaining_values_copy[box_idx].copy()
                            boxs_remaining_values_copy[box_idx].remove(char)
                        except ValueError:
                            if show_logs:
                                print(f"Value {char} could not be filled at {cell_idx} as it was already present in box {box_idx}")
                            is_valid_propagation = False
                            break

                if show_logs:
                    print("After propagation")
                    print("valid?", is_valid_propagation)
                    for i1 in range(size):
                        print([len(cell_options_copy[(i1, i2)]) for i2 in range(size)])
                    for i1 in range(size):
                        print([flat_grid_copy[i1 * size + i2] for i2 in range(size)])
                    print(f"Remaining values row {new_row_idx}:", rows_remaining_values_copy[new_row_idx])
                    print(f"Remaining values col {new_col_idx}:", cols_remaining_values_copy[new_col_idx])
                    print(f"Remaining values box {new_box_idx}:", boxs_remaining_values_copy[new_box_idx])
                    print()

                # 3 Check whether there are any empty cells without remaining options
                if is_valid_propagation:
                    for i1, i2 in itertools.product(range(size), repeat=2):
                        # No value filled in and no options available
                        if flat_grid_copy[i1 * size + i2] == EMPTY_CHAR and not cell_options_copy[(i1, i2)]:
                            if show_logs:
                                print(f"Grid invalid: No option available for empty cell {(i1, i2)}")
                            is_valid_propagation = False
                            break

                # Update logs
                # Note: There's a lot of continue/break statements going on, and have to make sure to capture all the
                #  relevant steps -> both valid and invalid propagations reach this point, and are added to the logs
                filled_grid = [[flat_grid_copy[i1 * size + i2] for i2 in range(size)] for i1 in range(size)]
                assert isinstance(possible_chars, list)
                # is_last_option = possible_chars.index(new_char) == len(possible_chars) - 1
                logs_step = (
                    (new_row_idx, new_col_idx), idxs_propagated, filled_grid, is_valid_propagation,
                    (possible_chars.index(new_char) + 1, len(possible_chars)), cell_options_copy,
                )
                logs["steps"].append(logs_step)

                nonlocal counter_invalid_propagations
                if not is_valid_propagation:
                    counter_invalid_propagations += 1
                    # import time
                    # time.sleep(0.5)
                    print("Propagation not valid! Continuing..", idx, counter_invalid_propagations)
                    if counter_invalid_propagations == MAX_NUMBER_INVALID_PROPAGATIONS_BEFORE_RESTART:
                        raise BadInitialisationException()
                    continue

                # Prepare data for the next iteration
                data_copy = (cell_options_copy, rows_remaining_values_copy, cols_remaining_values_copy, boxs_remaining_values_copy)

                yield from recursion(flat_grid_copy, data_copy, idx + 1)

    # Initialise data structure
    # TODO This probably requires some experimentation to see what is most efficient
    # It would be easiest if propagation is done automatically when filling in a value
    # Note: We assume we start with an empty grid
    # Note: Although using a lot of different sets might seem slow as they have to be copied, we do not need to copy all
    #  sets all the time, but only if one is updated through propagation
    cell_options = {(i1, i2): list(chars) for i1 in range(size) for i2 in range(size)}
    rows_remaining_values = {i1: list(chars) for i1 in range(size)}
    cols_remaining_values = {i2: list(chars) for i2 in range(size)}
    boxs_remaining_values = {idx_box: list(chars) for idx_box in range(size)}

    # Make sure runs are easily reproduceable
    seed = random.randint(0, 100_000)
    random.seed(seed)
    print("Using random seed for generating solution:", seed)
    # Randomise the search
    for _, coll in cell_options.items():
        random.shuffle(coll)

    data = (cell_options, rows_remaining_values, cols_remaining_values, boxs_remaining_values)

    # The iteratively filled in grid
    flat_grid = [EMPTY_CHAR] * size ** 2

    # Initialise the steps taken, to be updated during each step of the recursion
    # steps = []

    # Note: Using yield functionality allows both for
    #  - investigating intermediate results without having to wait until all solutions are generated
    #  - having the option to continue the search after a certain number of solutions, or starting over from scratch,
    #    providing flexibility in case either option turns out to be more efficient (or even allowing for a more
    #    targeted search in case we learn in which situations we are close or far away from a good solution)
    # and also reduces memory and runtime issues in case the number of solutions is very large and/or takes a long time
    # to find
    try:
        yield from recursion(flat_grid, data, idx=0)
    except BadInitialisationException:
        # It could be that the initial values in the grid were unfortunately chosen, start from scratch to be able
        # to find an instance more quickly
        yield from generate_grids_using_propagation(chars, pattern, show_logs=show_logs)
