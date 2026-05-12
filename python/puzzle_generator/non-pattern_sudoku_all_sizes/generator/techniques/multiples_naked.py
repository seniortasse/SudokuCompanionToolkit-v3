
import itertools

from generator.model import DIMENSIONS


# Technique description:
#  multiples-naked:
#   A combination of {multiple} cells in some dimension contains only {multiple} possible values
#   -> These values can be removed from the other cells in this dimension


# TODO After rewriting to handle custom boxes layout, the logic has become very similar to regular multiples - see if
#  the logic can be combined
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

    details = []

    # Preprocess data structure
    # options_for_idxs = {
    #     (i1, i2): options[i1][i2]
    #     for (i1, i2) in itertools.product(range(size), repeat=2)
    # }

    for dim in DIMENSIONS:
        for idx_dim, idxs_for_dim in options.idxs_for_dims[dim].items():

            # TODO Here we would like to use the instance
            idxs_empty = [(i1, i2) for (i1, i2) in idxs_for_dim if len(options[i1][i2]) > 0]

            idxs_combs = list(itertools.combinations(idxs_empty, multiple_number))
            if show_logs:
                print(f"Checking {len(idxs_combs)} combinations of {multiple_number} cells for {len(idxs_empty)} empty cells in {dim} {idx_dim + 1}")
            for idxs_comb in idxs_combs:
                chars_combined = tuple(sorted(set.union(*(options[i1][i2] for (i1, i2) in idxs_comb))))
                assert multiple_number <= len(chars_combined) <= len(idxs_empty)

                if len(chars_combined) == multiple_number:
                    name_application = f"multiple-naked {chars_combined} in {dim} {idx_dim + 1} with cells {tuple(map(lambda idx: tuple(map(lambda x: x + 1, idx)), idxs_comb))}"
                    if show_logs:
                        print(f"Found {name_application}, removing chars of multiple from other cells in {dim} {idx_dim + 1}")

                    # Remove chars of multiple from empty cells in the same dimension outside the cells containing the
                    #  multiple
                    removed_chars = []
                    for (i1, i2) in idxs_for_dim:
                        if (i1, i2) not in idxs_comb:
                            for char in chars_combined:
                                if char in options[i1][i2]:
                                    if show_logs:
                                        print(f"Remove '{char}' from {(i1 + 1, i2 + 1)}")
                                    removed_chars.append(((i1, i2), char))

                    options_for_idxs = {idx: options[idx[0]][idx[1]] for idx in idxs_comb}
                    application = (dim, idx_dim, idxs_comb, options_for_idxs, chars_combined)
                    details.append((name_application, application, removed_chars))

    return details


def find_naked_doubles(options, chars, show_logs=False):
    return _find_naked_multiples(options, chars, multiple_number=2, show_logs=show_logs)


def find_naked_triplets(options, chars, show_logs=False):
    return _find_naked_multiples(options, chars, multiple_number=3, show_logs=show_logs)


def find_naked_quads(options, chars, show_logs=False):
    return _find_naked_multiples(options, chars, multiple_number=4, show_logs=show_logs)
