
import itertools


def _find_naked_multiples(options, chars, multiple_number, show_logs=False):

    # Similar to the two types of singles, for multiples there are also two types: regular (hidden) and naked;
    # So far we had only implemented the regular multiples, this function implements naked multiples;
    # While regular multiples are defined as 2/3/4 cells in a dimension which are the only cells containing a certain
    #  combination of 2/3/4 values (the multiple), removing all other options from those cells besides the multiple,
    #  naked multiples looks for a combination of 2/3/4 cells which contain only 2/3/4 options, removing those options
    #  from the other cells in that dimension;

    # Note that as for singles, naked multiples is much easier to implement than regular multiples

    # Steps:
    #  - For each dimension, determine whether there is a combination of 2/3/4 cells which contain a combination of
    #    2/3/4 options
    #  - Remove these options from the other cells in that dimension

    box_height, box_width, size = options.box_height, options.box_width, options.size

    details = []

    # Preprocess data structure
    # options_for_idxs = {
    #     (i1, i2): options[i1][i2]
    #     for (i1, i2) in itertools.product(range(size), repeat=2)
    # }

    for i1, row in options.get_rows():
        idxs_empty = [i2 for i2, e in enumerate(row) if len(e) > 0]
        combs_idxs = list(itertools.combinations(idxs_empty, multiple_number))
        if show_logs:
            print(f"Checking {len(combs_idxs)} possible combinations for {len(idxs_empty)} empty cols in row {i1 + 1}")
        for comb_idxs in combs_idxs:
            _combined_options = set(itertools.chain.from_iterable(options[i1][i2] for i2 in comb_idxs))
            # print(comb_idxs, _combined_options)
            if len(_combined_options) == multiple_number:
                multiple = tuple(sorted(_combined_options))
                name_application = f"multiple-naked {multiple} in row {i1 + 1} and cols {tuple(map(lambda x: x + 1, comb_idxs))}"
                if show_logs:
                    print(f"Found {name_application}, removing multiple from other cells in row {i1 + 1}")
                removed_chars = []
                for i2 in range(size):
                    if i2 not in comb_idxs:
                        remove_options = options[i1][i2].intersection(multiple)
                        for char in remove_options:
                            if show_logs:
                                print(f"Remove {char} from {(i1, i2)}")
                            removed_chars.append(((i1, i2), char))

                idxs_multiple = [(i1, i2) for i2 in comb_idxs]
                options_for_idxs = {idx: options[idx[0]][idx[1]] for idx in idxs_multiple}
                application = ("row", i1, idxs_multiple, options_for_idxs, multiple)
                details.append((name_application, application, removed_chars))

    for i2, col in options.get_cols():
        idxs_empty = [i1 for i1, e in enumerate(col) if len(e) > 0]
        combs_idxs = list(itertools.combinations(idxs_empty, multiple_number))
        if show_logs:
            print(f"Checking {len(combs_idxs)} possible combinations for {len(idxs_empty)} empty rows in col {i2 + 1}")
        for comb_idxs in combs_idxs:
            _combined_options = set(itertools.chain.from_iterable(options[i1][i2] for i1 in comb_idxs))
            # print(comb_idxs, _combined_options)
            if len(_combined_options) == multiple_number:
                multiple = tuple(sorted(_combined_options))
                name_application = f"multiple-naked {multiple} in col {i2 + 1} and rows {tuple(map(lambda x: x + 1, comb_idxs))}"
                if show_logs:
                    print(f"Found {name_application}, removing multiple from other cells in col {i2 + 1}")
                removed_chars = []
                for i1 in range(size):
                    if i1 not in comb_idxs:
                        remove_options = options[i1][i2].intersection(multiple)
                        for char in remove_options:
                            if show_logs:
                                print(f"Remove {char} from {(i1, i2)}")
                            removed_chars.append(((i1, i2), char))

                idxs_multiple = [(i1, i2) for i1 in comb_idxs]
                options_for_idxs = {idx: options[idx[0]][idx[1]] for idx in idxs_multiple}
                application = ("col", i2, idxs_multiple, options_for_idxs, multiple)
                details.append((name_application, application, removed_chars))

    for (b1, b2), box in options.get_boxs():
        assert len(box) == size
        idxs_empty = [
            (b1 * box_height + i // box_width, b2 * box_width + i % box_width)
            for i, e in enumerate(box) if len(e) > 0
        ]
        combs_idxs = list(itertools.combinations(idxs_empty, multiple_number))
        if show_logs:
            print(f"Checking {len(combs_idxs)} possible combinations for {len(idxs_empty)} empty cells in box {(b1 + 1, b2 + 1)}")
        for comb_idxs in combs_idxs:
            _combined_options = set(itertools.chain.from_iterable(options[idx[0]][idx[1]] for idx in comb_idxs))
            # print(comb_idxs, _combined_options)
            if len(_combined_options) == multiple_number:
                multiple = tuple(sorted(_combined_options))
                name_application = f"multiple-naked {multiple} in box {(b1 + 1, b2 + 1)} with cells {tuple(map(lambda idx: tuple(map(lambda x: x + 1, idx)), comb_idxs))}"
                if show_logs:
                    print(f"Found {name_application}, removing multiple from other cells in box {(b1 + 1, b2 + 1)}")
                removed_chars = []
                for _b1, _b2 in itertools.product(range(box_height), range(box_width)):
                    i1, i2 = b1 * box_height + _b1, b2 * box_width + _b2
                    if (i1, i2) not in comb_idxs:
                        remove_options = options[i1][i2].intersection(multiple)
                        for char in remove_options:
                            if show_logs:
                                print(f"Remove {char} from {(i1, i2)}")
                            removed_chars.append(((i1, i2), char))

                idxs_multiple = comb_idxs
                options_for_idxs = {idx: options[idx[0]][idx[1]] for idx in idxs_multiple}
                application = ("box", (b1, b2), idxs_multiple, options_for_idxs, multiple)
                details.append((name_application, application, removed_chars))

    return details


def find_naked_doubles(options, chars, show_logs=False):
    return _find_naked_multiples(options, chars, multiple_number=2, show_logs=show_logs)


def find_naked_triplets(options, chars, show_logs=False):
    return _find_naked_multiples(options, chars, multiple_number=3, show_logs=show_logs)


def find_naked_quads(options, chars, show_logs=False):
    return _find_naked_multiples(options, chars, multiple_number=4, show_logs=show_logs)
