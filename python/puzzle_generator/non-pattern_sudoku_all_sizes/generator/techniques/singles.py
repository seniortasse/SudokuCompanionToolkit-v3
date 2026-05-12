
from collections import defaultdict
import itertools
import operator

from generator.model import EMPTY_CHAR, DIMENSIONS
from generator.techniques.options import determine_options


# The base techniques
#  singles-1: A value is the only missing value in a dimension
#  singles-2: Looking at a certain dimension, and using the values of one helper dimension, a value can only fit in one
#   cell in the original dimension
#  singles-3: Looking at a certain dimension, and using the values of both other dimensions, a value can only fit in one
#   cell in the original dimension
#  singles-naked-1: Looking at a certain cell, the values of a dimension make that there is only one value which can fit
#   in that cell (note that this is the same as singles-1, and therefore we have not implemented this technique)
#  singles-naked-2: Looking at a certain cell, the values of any combination of two dimensions make that there is only
#   one value which can fit in that cell (all values other than the to be filled value occur in the two dimensions)
#  singles-naked-3: Looking at a certain cell, the values of all three dimensions make that there is only one value
#   which can fit in that cell (all values other than the to be filled value occur in the three dimensions)


def fits(instance, char, i1, i2, check_row=True, check_col=True, check_box=True):
    if instance[i1][i2] != EMPTY_CHAR:
        return False
    if check_row:
        row = instance.get_values("row", i1)
        if char in row:
            return False
    if check_col:
        col = instance.get_values("col", i2)
        if char in col:
            return False
    if check_box:
        box = instance.get_values("box", instance.get_idx_box(i1, i2))
        if char in box:
            return False
    return True


def find_singles_1(instance, show_logs=False):
    singles = []
    for dim in DIMENSIONS:
        for idx_dim, idxs_for_dim in instance.idxs_for_dims[dim].items():
            vals_for_dim = instance.get_values(dim, idx_dim)
            if vals_for_dim.count(EMPTY_CHAR) == 1:
                idx_empty_char = idxs_for_dim[vals_for_dim.index(EMPTY_CHAR)]
                missing_char = instance.chars.difference(vals_for_dim).pop()
                if show_logs:
                    print(f"Looking at {dim} {idx_dim + 1}, '{missing_char}' is the only missing value at {tuple(map(lambda x: x + 1, idx_empty_char))}")
                singles.append((idx_empty_char, missing_char, dim))
    # Group result
    grouped_singles = _group_hits(singles)
    return grouped_singles


def find_singles_2(instance, show_logs=False):
    singles = []
    for dim in DIMENSIONS:
        dims_help = [_dim for _dim in DIMENSIONS if _dim != dim]
        for idx_dim, idxs_for_dim in instance.idxs_for_dims[dim].items():
            vals_for_dim = instance.get_values(dim, idx_dim)
            missing_chars = instance.chars.difference(vals_for_dim)
            for char in missing_chars:
                for dim_help in dims_help:
                    check_row = dim_help == "row"
                    check_col = dim_help == "col"
                    check_box = dim_help == "box"
                    idxs_possible = []
                    for (i1, i2) in idxs_for_dim:
                        if fits(instance, char, i1, i2, check_row=check_row, check_col=check_col, check_box=check_box):
                            idxs_possible.append((i1, i2))
                    if len(idxs_possible) == 1:
                        idx_fit = idxs_possible[0]
                        if show_logs:
                            print(f"Looking at {dim} {idx_dim + 1} and {dim} + {dim_help} dimensions, '{char}' only fits at {tuple(map(lambda x: x + 1, idx_fit))}")
                        singles.append((idx_fit, char, f"{dim} + {dim_help}"))
    # Group result
    grouped_singles = _group_hits(singles)
    return grouped_singles


# TODO As with singles-naked, the logic for singles-1/2/3 can likely be aggregated into one function
def find_singles_3(instance, show_logs=False):
    singles = []
    for dim in DIMENSIONS:
        check_row = dim != "row"
        check_col = dim != "col"
        check_box = dim != "box"
        for idx_dim, idxs_for_dim in instance.idxs_for_dims[dim].items():
            vals_for_dim = instance.get_values(dim, idx_dim)
            missing_chars = instance.chars.difference(vals_for_dim)
            for char in missing_chars:
                idxs_possible = []
                for (i1, i2) in idxs_for_dim:
                    if fits(instance, char, i1, i2, check_row=check_row, check_col=check_col, check_box=check_box):
                        idxs_possible.append((i1, i2))
                if len(idxs_possible) == 1:
                    idx_fit = idxs_possible[0]
                    if show_logs:
                        print(f"Looking at {dim} {idx_dim + 1} and {' + '.join([dim] + [_dim for _dim in DIMENSIONS if _dim != dim])} dimensions, '{char}' only fits at {tuple(map(lambda x: x + 1, idx_fit))}")
                    singles.append((idx_fit, char, dim))
    # Group result
    grouped_singles = _group_hits(singles)
    return grouped_singles


def _find_naked_singles(instance, number_dims, show_logs=False):
    singles = []
    combinations = itertools.combinations(DIMENSIONS, number_dims)
    for dims in combinations:
        check_row = "row" in dims
        check_col = "col" in dims
        check_box = "box" in dims
        for (i1, i2) in itertools.product(range(instance.size), repeat=2):
            if instance.is_empty(i1, i2):
                options = determine_options(instance, i1, i2, check_row=check_row, check_col=check_col, check_box=check_box)
                if len(options) == 1:
                    char = options.pop()
                    dims_str = ' + '.join(dims)
                    if show_logs:
                        print(f"Looking at {dims_str}, found naked single '{char}' at {(i1 + 1, i2 + 1)}")
                    singles.append(((i1, i2), char, dims_str))
    singles = _group_hits(singles)
    return singles


def find_naked_singles_1(instance, show_logs=False):
    return _find_naked_singles(instance, number_dims=1, show_logs=show_logs)


def find_naked_singles_2(instance, show_logs=False):
    return _find_naked_singles(instance, number_dims=2, show_logs=show_logs)


def find_naked_singles_3(instance, show_logs=False):
    return _find_naked_singles(instance, number_dims=3, show_logs=show_logs)


def _group_hits(hits):
    grouped_idxs = defaultdict(list)
    for hit in hits:
        grouped_idxs[hit[0]].append(hit)
    grouped_hits = [
        (idx, _singles[0][1], " & ".join(_single[-1] for _single in _singles))
        for idx, _singles in grouped_idxs.items()
    ]
    # TODO This grouping functionality does not guarantee the same ordering in different runs, and requires additional
    #  sorting (we want the same result for multiple runs mostly for testing purposes)
    #  -> This does mess up the order of displaying first rows then cols then boxes, but this can be fixed manually by
    #     custom sorting later
    grouped_hits = sorted(grouped_hits, key=operator.itemgetter(0))
    return grouped_hits
