
from collections import defaultdict
import itertools

from generator.model import DIMENSIONS


# Technique description:
#  multiples:
#   A combination of {multiple} values can only occur in {multiple} cells in a dimension
#   -> The other values can be removed from these cells
# Note: The version 2 implementation does not require all values to be present in all these cells


# TODO It would be an improvement to apply multiples iteratively, as removing options might lead to new multiples
#  arising -> this should be done in the logic in algo_human, when at least one option is removed using the technique,
#  apply it until no more options are removed (we might already do this by now)


def _find_multiples(options, chars, multiple_number, show_logs=False):
    """
    Implementation V2: There is a combination of {multiple} characters which occur only in {multiple} cells, but do not
     all have to be present in all {multiple} cells (which was the case in V1)
    """

    # Approach: For each dimension, identify multiples, and remove all other options from those cells

    # Steps:
    #  1 For each character not yet present in the dimension, determine in which cells it can occur
    #  2 For all combinations of {multiple} characters, determine whether the collection of cells in which they can
    #    occur is {multiple}

    # Keep details to backtrack in the logs which multiples had an effect on the found values
    details = []

    # TODO Really have to make a more consistent flow for this, modifying options during the process is very tricky!

    # TODO Make options object immutable

    for dim in DIMENSIONS:
        for idx_dim, idxs_for_dim in options.idxs_for_dims[dim].items():
            # TODO Here we would like to have the instance to easily retrieve the missing chars in the dimension
            #  -> We should probably give the instance by default to all techniques
            # vals_for_dim = instance.get_values(dim, idx_dim)

            # For each char determine in which cells it is an option
            idxs_possible_for_chars = defaultdict(list)
            for char in chars:
                for (i1, i2) in idxs_for_dim:
                    if char in options[i1][i2]:
                        idxs_possible_for_chars[char].append((i1, i2))

            # For each combination of {multiple} chars, check whether the combined options are limited to {multiple}
            #  cells
            chars_missing = sorted(set(idxs_possible_for_chars.keys()))
            assert len(chars_missing) == sum(len(options[i1][i2]) > 0 for (i1, i2) in idxs_for_dim)

            chars_combs = itertools.combinations(chars_missing, multiple_number)
            for chars_comb in chars_combs:
                idxs_combined = sorted(set(itertools.chain.from_iterable(idxs_possible_for_chars[char] for char in chars_comb)))
                # This might fail when not applying easier techniques first? No, we are looking at a combination of
                #  {multiple} chars, which always have to occur in at least {multiple} cells, unless the instance is
                #  invalid
                assert multiple_number <= len(idxs_combined) <= len(chars_missing)

                if len(idxs_combined) == multiple_number:
                    # multiples[dim].append((idx_dim, idxs_combined))
                    # if show_logs:
                    #     print(f"Found multiple in {dim} {idx_dim + 1}: {chars_comb}")
                    name_application = f"multiple {chars_comb} in {dim} {idx_dim + 1} in {tuple(map(lambda idx: tuple(map(lambda x: x + 1, idx)), idxs_combined))}"
                    if show_logs:
                        print(f"Identified {name_application}, removing all other options")

                    # Remove all other values from the cells containing the multiple
                    removed_chars = []
                    for (i1, i2) in idxs_combined:
                        # TODO Sort for testing purposes
                        for char in chars:
                            if char in options[i1][i2] and char not in chars_comb:
                                if show_logs:
                                    print(f"Remove '{char}' from {(i1 + 1, i2 + 1)}")
                                removed_chars.append(((i1, i2), char))

                    # For multiples larger than 2 not all values need to be present in all cells
                    # TODO Perhaps we want to add the entire options to details, perhaps even for all techniques modifying the
                    #  structure; For now we don't do that as the updated options issue is still a thing and has to be
                    #  restuctured first
                    options_for_idxs = {
                        (i1, i2): options[i1][i2].intersection(chars_comb)
                        for (i1, i2) in idxs_combined
                    }
                    details.append((name_application, (chars_comb, idxs_combined, options_for_idxs, dim), removed_chars))

    return details


def find_doubles(options, chars, show_logs=False):
    return _find_multiples(options, chars, multiple_number=2, show_logs=show_logs)


def find_triplets(options, chars, show_logs=False):
    return _find_multiples(options, chars, multiple_number=3, show_logs=show_logs)


def find_quads(options, chars, show_logs=False):
    return _find_multiples(options, chars, multiple_number=4, show_logs=show_logs)
