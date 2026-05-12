
from collections import defaultdict
import itertools


# TODO It would be an improvement to apply multiples iteratively, as removing options might lead to new multiples
#  arising


def _find_multiples_v2(options, chars, multiple_number, show_logs=False):
    """
    Implementation V2: There is a combination of {multiple} characters which occur only in {multiple} cells, but do not
     all have to be present in all {multiple} cells (which was the case in V1)
    """

    # Steps:
    #  1 For each character not yet present in the dimension, determine in which cells it can occur
    #  2 For all combinations of {multiple} characters, determine whether the collection of cells in which they can
    #    occur is {multiple}

    box_height, box_width, size = options.box_height, options.box_width, options.size

    pairs_rows = defaultdict(list)
    pairs_cols = defaultdict(list)
    pairs_boxs = defaultdict(list)

    for i1, row in options.get_rows():

        # 1 Determine the indices in which the char is an option
        locations = defaultdict(list)
        for char in chars:
            for i2, e in enumerate(row):
                if char in e:
                    locations[char].append(i2)

        # 2 Find combinations which occur in {multiple} cells
        chars_not_present = set(locations.keys())
        combs = itertools.combinations(chars_not_present, multiple_number)

        for comb in combs:
            idxs = set(itertools.chain(*[locations[char] for char in comb]))
            if len(idxs) == multiple_number:
                multiple = tuple(sorted(comb))
                pairs_rows[multiple].append((i1, tuple(sorted(idxs))))
                if show_logs:
                    print(f"Found multiple in row {i1}: {comb}")

    for i2, col in options.get_cols():

        # 1 Determine the indices in which the char is an option
        locations = defaultdict(list)
        for char in chars:
            for i1, e in enumerate(col):
                if char in e:
                    locations[char].append(i1)

        # 2 Find combinations which occur in {multiple} cells
        chars_not_present = set(locations.keys())
        combs = itertools.combinations(chars_not_present, multiple_number)

        for comb in combs:
            idxs = set(itertools.chain(*[locations[char] for char in comb]))
            if len(idxs) == multiple_number:
                multiple = tuple(sorted(comb))
                pairs_cols[multiple].append((i2, tuple(sorted(idxs))))
                if show_logs:
                    print(f"Found multiple in col {i2}: {comb}")

    for (b1, b2), box in options.get_boxs():

        # 1 Determine the indices in which the char is an option
        locations = defaultdict(list)
        for char in chars:
            for _i1 in range(box_height):
                for _i2 in range(box_width):
                    e = box[_i1 * box_width + _i2]
                    if char in e:
                        i1 = b1 * box_height + _i1
                        i2 = b2 * box_width + _i2
                        locations[char].append((i1, i2))

        # 2 Find combinations which occur in {multiple} cells
        chars_not_present = set(locations.keys())
        combs = itertools.combinations(chars_not_present, multiple_number)

        for comb in combs:
            idxs = set(itertools.chain(*[locations[char] for char in comb]))
            if len(idxs) == multiple_number:
                multiple = tuple(sorted(comb))
                pairs_boxs[multiple].append(tuple(sorted(idxs)))
                if show_logs:
                    print(f"Found multiple in box {(b1, b2)}: {multiple}")

    return pairs_rows, pairs_cols, pairs_boxs


def _find_multiples_v1(options, chars, multiple_number, show_logs=False):
    """
    Implementation V1: There is a combination of {multiple} characters which occur in exactly the same {multiple} cells
    """

    box_height, box_width, size = options.box_height, options.box_width, options.size

    pairs_rows = defaultdict(list)
    for i1, row in enumerate(options):
        locations = defaultdict(list)
        for i2, e in enumerate(row):
            for char in chars:
                if e is not None and char in e:
                    locations[char].append(i2)
        groups = defaultdict(list)
        for char, idxs in locations.items():
            if len(idxs) == multiple_number:
                groups[tuple(sorted(idxs))].append(char)
        for idxs, possible_chars in groups.items():
            if len(possible_chars) == multiple_number:
                pair = tuple(sorted(possible_chars))
                pairs_rows[pair].append((i1, idxs))
                if show_logs:
                    print(f"Found multiple in row {i1}: {pair}")

    pairs_cols = defaultdict(list)
    for i2 in range(size):
        col = [row[i2] for row in options]
        locations = defaultdict(list)
        for i1, e in enumerate(col):
            for char in chars:
                if e is not None and char in e:
                    locations[char].append(i1)
        groups = defaultdict(list)
        for char, idxs in locations.items():
            if len(idxs) == multiple_number:
                groups[tuple(sorted(idxs))].append(char)
        for idxs, possible_chars in groups.items():
            if len(possible_chars) == multiple_number:
                pair = tuple(sorted(possible_chars))
                pairs_cols[pair].append((i2, idxs))
                if show_logs:
                    print(f"Found multiple in col {i2}: {pair}")

    pairs_boxs = defaultdict(list)
    for (b1, b2), box in options.get_boxs():
        locations = defaultdict(list)
        for _i1 in range(box_height):
            for _i2 in range(box_width):
                e = box[_i1 * box_width + _i2]
                for char in chars:
                    if e is not None and char in e:
                        i1 = b1 * box_height + _i1
                        i2 = b2 * box_width + _i2
                        locations[char].append((i1, i2))
        groups = defaultdict(list)
        for char, idxs in locations.items():
            if len(idxs) == multiple_number:
                groups[tuple(sorted(idxs))].append(char)
        for idxs, possible_chars in groups.items():
            if len(possible_chars) == multiple_number:
                pair = tuple(sorted(possible_chars))
                pairs_boxs[pair].append(idxs)
                if show_logs:
                    print(f"Found multiple in box {(b1, b2)}: {pair}")

    return pairs_rows, pairs_cols, pairs_boxs


def find_multiples(options, chars, multiple_number, version, show_logs=False):

    # Approach: For each dimension, identify pairs, and remove all other options from those cells

    box_height, box_width = options.box_height, options.box_width

    # Keep details to backtrack in the logs which multiples had an effect on the found values
    details = []

    # TODO Really have to make a more consistent flow for this, modifying options during the process is very tricky!

    # TODO Make options object immutable

    if show_logs:
        print(f"Using version {version}")

    if version == 1:
        multiples_rows, multiples_cols, multiples_boxs = _find_multiples_v1(options, chars, multiple_number=multiple_number, show_logs=show_logs)
    elif version == 2:
        multiples_rows, multiples_cols, multiples_boxs = _find_multiples_v2(options, chars, multiple_number=multiple_number, show_logs=show_logs)
        # TODO Temporary check to verify that the new implementation finds the same pairs as the old one
        if multiple_number == 2:
            # if len(multiples_rows) + len(multiples_cols) + len(multiples_boxs) > 0:
            #     raise Exception("Check done")
            _multiples_rows, _multiples_cols, _multiples_boxs = _find_multiples_v1(options, chars, multiple_number=multiple_number, show_logs=show_logs)
            assert multiples_rows == _multiples_rows, f"{multiples_rows} | {_multiples_rows}"
            assert multiples_cols == _multiples_cols, f"{multiples_cols} | {_multiples_cols}"
            assert multiples_boxs == _multiples_boxs, f"{multiples_boxs} | {_multiples_boxs}"
    else:
        raise Exception(f"Version {version} not implemented in find_multiples()")

    for multiple, idxs in multiples_rows.items():
        for row_idx, col_idxs in idxs:
            name_application = f"multiple {tuple(sorted(multiple))} in row {row_idx + 1} and cols {tuple(e + 1 for e in col_idxs)}"
            if show_logs:
                print(f"Identified {name_application}, removing all other options")

            removed_chars = []
            for col_idx in col_idxs:
                remove_options = options[row_idx][col_idx].difference(multiple)
                for char in remove_options:
                    # options[row_idx][col_idx].remove(char)
                    if show_logs:
                        print(f"Remove {char} from {(row_idx + 1, col_idx + 1)}")
                    removed_chars.append(((row_idx, col_idx), char))

            idxs_multiple = [(row_idx, col_idx) for col_idx in col_idxs]
            # For multiples larger than 2 not all values need to be present in all cells
            # TODO Perhaps we want to add the entire options to details, perhaps even for all techniques modifying the
            #  structure; For now we don't do that as the updated options issue is still a thing and has to be
            #  restuctured first
            options_for_idxs = {
                idx: options[idx[0]][idx[1]].intersection(multiple)
                for idx in idxs_multiple
            }
            details.append((name_application, (multiple, idxs_multiple, options_for_idxs, "row"), removed_chars))

    for multiple, idxs in multiples_cols.items():
        for col_idx, row_idxs in idxs:
            name_application = f"multiple {tuple(sorted(multiple))} in col {col_idx + 1} and rows {tuple(e + 1 for e in row_idxs)}"
            if show_logs:
                print(f"Identified {name_application}, removing all other options")

            removed_chars = []
            for row_idx in row_idxs:
                remove_options = options[row_idx][col_idx].difference(multiple)
                for char in remove_options:
                    # options[row_idx][col_idx].remove(char)
                    if show_logs:
                        print(f"Remove {char} from {(row_idx + 1, col_idx + 1)}")
                    removed_chars.append(((row_idx, col_idx), char))

            idxs_multiple = [(row_idx, col_idx) for row_idx in row_idxs]
            options_for_idxs = {
                idx: options[idx[0]][idx[1]].intersection(multiple)
                for idx in idxs_multiple
            }
            details.append((name_application, (multiple, idxs_multiple, options_for_idxs, "col"), removed_chars))

    for multiple, idxs in multiples_boxs.items():
        for _idxs in idxs:
            idx_box = (_idxs[0][0] // box_height + 1, _idxs[0][1] // box_width + 1)
            name_application = f"multiple {tuple(sorted(multiple))} in box {idx_box} at {[(row_idx + 1, col_idx + 1) for (row_idx, col_idx) in _idxs]}"
            if show_logs:
                print(f"Identified {name_application}, removing all other options")

            removed_chars = []
            for row_idx, col_idx in _idxs:
                remove_options = options[row_idx][col_idx].difference(multiple)
                for char in remove_options:
                    # options[row_idx][col_idx].remove(char)
                    if show_logs:
                        print(f"Remove {char} from {(row_idx + 1, col_idx + 1)}")
                    removed_chars.append(((row_idx, col_idx), char))

            idxs_multiple = _idxs
            options_for_idxs = {
                idx: options[idx[0]][idx[1]].intersection(multiple)
                for idx in idxs_multiple
            }
            details.append((name_application, (multiple, idxs_multiple, options_for_idxs, "box"), removed_chars))

    return details


def find_doubles(options, chars, version=2, show_logs=False):
    return find_multiples(options, chars, multiple_number=2, version=version, show_logs=show_logs)


def find_triplets(options, chars, version=2, show_logs=False):
    return find_multiples(options, chars, multiple_number=3, version=version, show_logs=show_logs)


def find_quads(options, chars, version=2, show_logs=False):
    return find_multiples(options, chars, multiple_number=4, version=version, show_logs=show_logs)
