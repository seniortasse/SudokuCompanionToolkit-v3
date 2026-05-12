
from collections import defaultdict
import itertools
import operator

from generator.model import EMPTY_CHAR, get_box, fits
from generator.techniques.options import determine_options


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


def find_singles_1(instance, show_logs=False):
    singles = []
    box_height, box_width = instance.box_height, instance.box_width
    dimension = "row"
    for i1, row in instance.get_rows():
        if row.count(EMPTY_CHAR) == 1:
            i2 = row.index(EMPTY_CHAR)
            char = instance.chars.difference(row).pop()
            if show_logs:
                print(f"Looking at row {i1 + 1}, {char} is the only missing value at {(i1 + 1, i2 + 1)}")
            singles.append(((i1, i2), char, dimension))
    dimension = "col"
    for i2, col in instance.get_cols():
        if col.count(EMPTY_CHAR) == 1:
            i1 = col.index(EMPTY_CHAR)
            char = instance.chars.difference(col).pop()
            if show_logs:
                print(f"Looking at col {i2 + 1}, {char} is the only missing value at {(i1 + 1, i2 + 1)}")
            singles.append(((i1, i2), char, dimension))
    dimension = "box"
    for (b1, b2), box in instance.get_boxs():
        if box.count(EMPTY_CHAR) == 1:
            i = box.index(EMPTY_CHAR)
            i1 = b1 * box_height + i // box_width
            i2 = b2 * box_width + i % box_width
            char = instance.chars.difference(box).pop()
            if show_logs:
                print(f"Looking at box {(b1 + 1, b2 + 1)}, {char} is the only missing value at {(i1 + 1, i2 + 1)}")
            singles.append(((i1, i2), char, dimension))
    # Group result
    grouped_singles = _group_hits(singles)
    return grouped_singles


def find_singles_2(instance, show_logs=False):
    singles = []
    box_height, box_width, size = instance.box_height, instance.box_width, instance.size
    dimension = "row"
    for i1, row in enumerate(instance):
        missing_chars = instance.chars.difference(row)
        for char in missing_chars:
            for dim in ["col", "box"]:
                check_col = dim == "col"
                check_box = dim == "box"
                possible_idxs = []
                for i2 in range(size):
                    if fits(instance, char, i1, i2, check_row=False, check_col=check_col, check_box=check_box):
                        possible_idxs.append((i1, i2))
                if len(possible_idxs) == 1:
                    if show_logs:
                        print(f"Looking at row {i1 + 1} and row + {dim} dimensions, {char} only fits at {tuple(e + 1 for e in possible_idxs[0])}")
                    singles.append((possible_idxs[0], char, f"{dimension} + {dim}"))
    dimension = "col"
    for i2 in range(size):
        col = [row[i2] for row in instance]
        missing_chars = instance.chars.difference(col)
        for char in missing_chars:
            for dim in ["row", "box"]:
                check_row = dim == "row"
                check_box = dim == "box"
                possible_idxs = []
                for i1 in range(size):
                    if fits(instance, char, i1, i2, check_row=check_row, check_col=False, check_box=check_box):
                        possible_idxs.append((i1, i2))
                if len(possible_idxs) == 1:
                    if show_logs:
                        print(f"Looking at col {i2 + 1} and col + {dim} dimensions, {char} only fits at {tuple(e + 1 for e in possible_idxs[0])}")
                    singles.append((possible_idxs[0], char, f"{dimension} + {dim}"))
    dimension = "box"
    for (b1, b2), box in instance.get_boxs():
        # if (b1, b2) == (2, 0):
        #     raise Exception()
        missing_chars = instance.chars.difference(box)
        for char in missing_chars:
            for dim in ["row", "col"]:
                check_row = dim == "row"
                check_col = dim == "col"
                possible_idxs = []
                for i1 in range(b1 * box_height, (b1 + 1) * box_height):
                    for i2 in range(b2 * box_width, (b2 + 1) * box_width):
                        if fits(instance, char, i1, i2, check_row=check_row, check_col=check_col, check_box=False):
                            possible_idxs.append((i1, i2))
                if len(possible_idxs) == 1:
                    if show_logs:
                        print(f"Looking at box {(b1 + 1, b2 + 1)} and box + {dim} dimensions, {char} only fits at {tuple(e + 1 for e in possible_idxs[0])}")
                    singles.append((possible_idxs[0], char, f"{dimension} + {dim}"))
    # Group result
    grouped_singles = _group_hits(singles)
    return grouped_singles


def find_singles(instance, rays=None, show_logs=False):
    singles = []
    box_height, box_width, size = instance.box_height, instance.box_width, instance.size
    dimension = "row"
    for i1, row in enumerate(instance):
        missing_chars = instance.chars.difference(row)
        for char in missing_chars:
            possible_idxs = []
            for i2 in range(size):
                if fits(instance, char, i1, i2, check_row=False):
                    if rays is None:
                        possible_idxs.append((i1, i2))
                    else:
                        rays_col = rays["cols"][char]
                        is_rayed = any(
                            i2 == _i2 and not _b1 * box_height <= i1 < (_b1 + 1) * box_height
                            for (_b1, _b2), _i2 in rays_col
                        )
                        if not is_rayed:
                            possible_idxs.append((i1, i2))
            if len(possible_idxs) == 1:
                if show_logs:
                    print(f"Looking at row {i1 + 1} and row + col + box dimensions, {char} only fits at {tuple(e + 1 for e in possible_idxs[0])}")
                singles.append((possible_idxs[0], char, dimension))
    dimension = "col"
    for i2 in range(size):
        col = [row[i2] for row in instance]
        missing_chars = instance.chars.difference(col)
        for char in missing_chars:
            possible_idxs = []
            for i1 in range(size):
                if fits(instance, char, i1, i2, check_col=False):
                    if rays is None:
                        possible_idxs.append((i1, i2))
                    else:
                        rays_row = rays["rows"][char]
                        is_rayed = any(
                            i1 == _i1 and not _b2 * box_width <= i2 < (_b2 + 1) * box_width
                            for (_b1, _b2), _i1 in rays_row
                        )
                        if not is_rayed:
                            possible_idxs.append((i1, i2))
            if len(possible_idxs) == 1:
                if show_logs:
                    print(f"Looking at col {i2 + 1} and col + row + box dimensions, {char} only fits at {tuple(e + 1 for e in possible_idxs[0])}")
                singles.append((possible_idxs[0], char, dimension))
    dimension = "box"
    for (b1, b2), box in instance.get_boxs():
        missing_chars = instance.chars.difference(box)
        for char in missing_chars:
            possible_idxs = []
            for i1 in range(b1 * box_height, (b1 + 1) * box_height):
                for i2 in range(b2 * box_width, (b2 + 1) * box_width):
                    if fits(instance, char, i1, i2, check_box=False):
                        if rays is None:
                            possible_idxs.append((i1, i2))
                        else:
                            rays_col = rays["cols"][char]
                            is_rayed_col = any(
                                i2 == _i2 and not _b1 * box_height <= i1 < (_b1 + 1) * box_height
                                for (_b1, _b2), _i2 in rays_col
                            )
                            rays_row = rays["rows"][char]
                            is_rayed_row = any(
                                i1 == _i1 and not _b2 * box_width <= i2 < (_b2 + 1) * box_width
                                for (_b1, _b2), _i1 in rays_row
                            )
                            is_rayed = is_rayed_col or is_rayed_row
                            if not is_rayed:
                                possible_idxs.append((i1, i2))
            if len(possible_idxs) == 1:
                if show_logs:
                    print(f"Looking at box {(b1 + 1, b2 + 1)} and box + row + col dimensions, {char} only fits at {tuple(e + 1 for e in possible_idxs[0])}")
                singles.append((possible_idxs[0], char, dimension))
    # Group result
    grouped_singles = _group_hits(singles)
    return grouped_singles


def _find_naked_singles(instance, number_dims, show_logs=False):
    singles = []
    size = instance.size
    combinations = itertools.combinations(("row", "col", "box"), number_dims)
    for dims in combinations:
        check_row = "row" in dims
        check_col = "col" in dims
        check_box = "box" in dims
        for i1 in range(size):
            for i2 in range(size):
                if instance.is_empty(i1, i2):
                    options = determine_options(instance, i1, i2, check_row=check_row, check_col=check_col, check_box=check_box)
                    if len(options) == 1:
                        dims_str = ' + '.join(dims)
                        if show_logs:
                            print(f"Looking at {dims_str}, found naked single {options} at {(i1 + 1, i2 + 1)}")
                        char = options.pop()
                        singles.append(((i1, i2), char, dims_str))
    singles = _group_hits(singles)
    return singles


def find_naked_singles_2(instance, show_logs=False):
    return _find_naked_singles(instance, number_dims=2, show_logs=show_logs)


def find_naked_singles_3(instance, show_logs=False):
    return _find_naked_singles(instance, number_dims=3, show_logs=show_logs)
