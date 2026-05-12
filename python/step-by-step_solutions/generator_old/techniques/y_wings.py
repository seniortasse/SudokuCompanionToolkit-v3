
import itertools


def find_y_wings(options, chars, show_logs=False):

    # Similar to but slightly different from ab-chains and x-wings, y-wings require a "chain" of length 3, containing
    #  only pairs of values, where the chain ends ("wings") have only 1 value in common ("wing value"), while the chain
    #  middle contains the other values of the wings;
    # The result is that all cells in the dimensions of both wings can have the wing value removed

    # Steps:
    #  1 Identify potential wings (pairs with only 1 value in common)
    #  2 Check whether there is a cell in a dimension of both of the potential wings, which contains the other values as
    #    a pair - if so, we have identified an y-wing
    #  3 For all other cells in a dimension of both of the potential wings, the wing value can be removed

    box_height, box_width, size = options.box_height, options.box_width, options.size

    details = []

    # Note: For this technique we introduced a new structure to be adopted by all other techniques:
    #  Only collect which options are removed during the application, but only actually remove them afterwards

    # 1
    idxs_with_pairs = [
        (i1, i2) for i1 in range(size) for i2 in range(size)
        if len(options[i1][i2]) == 2
    ]

    # 2
    idx_combs = list(itertools.combinations(idxs_with_pairs, 2))
    for idx_1, idx_2 in idx_combs:

        # Check whether the wings do not have any dimension in common -> This is not correct, in fact the wings can have
        #  two dimensions in common
        # TODO Very that this is correct
        idx_box_1 = (idx_1[0] // box_height, idx_1[1] // box_width)
        idx_box_2 = (idx_2[0] // box_height, idx_2[1] // box_width)
        # if idx_1[0] != idx_2[0] and idx_1[1] != idx_2[1] and idx_box_1 != idx_box_2:

        # Check whether they have exactly 1 value in common
        options_1 = options[idx_1[0]][idx_1[1]]
        options_2 = options[idx_2[0]][idx_2[1]]

        chars_in_common = options_1.intersection(options_2)
        if len(chars_in_common) == 1:
            char_wing = chars_in_common.pop()

            # Identify cells in a dimension of both wings
            # Note: This can never contain either of the wings, as before we have established that they have no
            #  dimension in common
            idxs_shared = []
            for i1 in range(size):
                for i2 in range(size):
                    is_in_dim_1 = i1 == idx_1[0] or i2 == idx_1[1] or (i1 // box_height, i2 // box_width) == idx_box_1
                    is_in_dim_2 = i1 == idx_2[0] or i2 == idx_2[1] or (i1 // box_height, i2 // box_width) == idx_box_2
                    # The target cell cannot be either of the wing cells
                    if is_in_dim_1 and is_in_dim_2 and (i1, i2) not in [idx_1, idx_2]:
                        idxs_shared.append((i1, i2))

            # Check whether there is a cell which contains the other options
            chars_center = options_1.symmetric_difference(options_2)
            assert len(chars_center) == 2
            for idx_center in idxs_shared:
                if options[idx_center[0]][idx_center[1]] == chars_center:
                    # idx_center = (i1, i2)
                    name_application = f"y-wing with wings {tuple(map(lambda x: x + 1, idx_1))} and {tuple(map(lambda x: x + 1, idx_2))} and center {tuple(map(lambda x: x + 1, idx_center))}"
                    if show_logs:
                        print(f"Found a {name_application}")
                        print(f"Removing wing value '{char_wing}' from cells with shared dimension:", sorted(map(lambda idx: tuple(map(lambda x: x + 1, idx)), idxs_shared)))

                    # 3
                    # Remove the wing value from all shared cells
                    removed_chars = []
                    for (i1, i2) in idxs_shared:
                        # Note: The wings are not included in the shared cells, and the center does not contain the wing
                        #  value, so this works well
                        if char_wing in options[i1][i2]:
                            if show_logs:
                                print(f"Remove '{char_wing}' from {(i1 + 1, i2 + 1)}")
                            removed_chars.append(((i1, i2), char_wing))

                    # Careful! Even when modifying options afterwards, the original sets will be modified
                    application = ((idx_1, idx_2), (options_1.copy(), options_2.copy()), idx_center, chars_center, char_wing)
                    details.append((name_application, application, removed_chars))

    return details
