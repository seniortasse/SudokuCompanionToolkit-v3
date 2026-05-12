
import itertools

from generator.techniques.advanced import get_idxs_in_dimensions


def find_y_wings(options, chars, show_logs=False):

    # Similar to but slightly different from ab-chains and x-wings, y-wings require a "chain" of length 3, containing
    #  only pairs of values, where the chain ends ("wings") have only 1 value in common ("wing value"), while the chain
    #  middle contains the other values of the wings;
    # The result is that all cells in the dimensions of both wings can have the wing value removed

    # Steps:
    #  1 Identify empty cells with two only options
    #  2 For all combinations of such cells, check whether they can be valid wings (having only 1 option in common)
    #  3 Check whether there is a cell in a dimension of both potential wings, which contains the other options as
    #    a pair - if so, we have identified a valid y-wing
    #  4 For all other cells in a dimension of both of potential wings, the wing value can be removed

    details = []

    # Note: For this technique we introduced a new structure to be adopted by all other techniques:
    #  Only collect which options are removed during the application, but only actually remove them afterwards

    # 1
    idxs_with_pairs = [
        (i1, i2) for (i1, i2) in itertools.product(range(options.size), repeat=2)
        if len(options[i1][i2]) == 2
    ]

    # Note: The wings can have any number of dimensions in common, as well as the center with the wings; When all have
    #  one dimension in common, it can be seen as a special case of triplets-naked;

    # 2
    for idx_1, idx_2 in itertools.combinations(idxs_with_pairs, 2):

        # Check whether the potential wings have one option in common
        options_1 = options[idx_1[0]][idx_1[1]]
        options_2 = options[idx_2[0]][idx_2[1]]

        chars_in_common = set.intersection(options_1, options_2)
        if len(chars_in_common) == 1:
            char_wing = chars_in_common.pop()

            # Establish what must be the center chars
            chars_center = set.symmetric_difference(options_1, options_2)
            assert len(chars_center) == 2

            # Identify cells sharing a dimension with both wings, which are potential centers
            # Note: The idxs of the wings are not included, as they are not included in the functions determining the
            #  idxs in all dimensions
            idxs_shared = sorted(
                set.intersection(
                    set(get_idxs_in_dimensions(options, *idx_1)),
                    set(get_idxs_in_dimensions(options, *idx_2)),
                )
            )

            # 3
            # For all identified potential centers, check whether it contains the other options
            for idx_center in idxs_shared:

                if options[idx_center[0]][idx_center[1]] == chars_center:
                    name_application = f"y-wing with wings {tuple(map(lambda x: x + 1, idx_1))} and {tuple(map(lambda x: x + 1, idx_2))} and center {tuple(map(lambda x: x + 1, idx_center))}"
                    if show_logs:
                        print(f"Found a {name_application}")
                        print(f"Removing wing value '{char_wing}' from cells with shared dimension:", sorted(map(lambda idx: tuple(map(lambda x: x + 1, idx)), idxs_shared)))

                    # 4
                    # Remove the wing value from all cells sharing a dimension with both wings
                    removed_chars = []
                    for (i1, i2) in idxs_shared:
                        # Note: The wings themselves are not included in the shared cells, and the center does not
                        #  contain the wing value, so this works well
                        if char_wing in options[i1][i2]:
                            if show_logs:
                                print(f"Remove '{char_wing}' from {(i1 + 1, i2 + 1)}")
                            removed_chars.append(((i1, i2), char_wing))

                    # Careful! Even when modifying options afterwards, the original sets will be modified
                    application = ((idx_1, idx_2), (options_1.copy(), options_2.copy()), idx_center, chars_center, char_wing)
                    details.append((name_application, application, removed_chars))

    return details
