
import itertools


def find_boxed_wings(options, chars, show_logs=False):

    # Related to the recently implemented y-wings, boxed-wings or box-and-wings, uses a box as the center, and some
    #  additional conditions are required:
    #  - the wings are outside of the box but both face the box
    #  - the cell in the box which sees both of the wings cannot contain the wing value
    #  x the cells in the box that don't see any of the wings cannot contain the wing value -> invalid condition
    #  - the wing value is a possible option in at least one of the empty cells of the box seen by the wing
    #  - the wing is the only cell in the row/col outside the box which contains the wing value as an option
    #  The number of options in the wings is not restricted, and there can be multiple wings values if they all
    #    satisfy above conditions
    #  One of the wings should face the box row-wise, the other one col-wise
    #  The result is that the wing value(s) can be removed from the cells that are seen by both wings (which are not in
    #   the box (assert this) as one of the conditions is that these cells cannot contain the wing value

    # Implementation:
    # It seems to make sense to start off from a box, and for all combinations of empty cells (potential wings) check
    #  whether they satisfy all conditions

    # Steps:
    #  - For each box, loop through the options or row/col combinations, and check for each character whether all
    #    conditions are satisfied
    #  - For all such combinations, check in the row/col whether there are wings containing this character
    #  - For all identified wings, remove the options from the shared cell outside the box

    # The boxed-x techniques are only implemented for and can only be applied to instances with default boxes layout
    assert not options.uses_custom_boxes_layout

    box_height, box_width, size = options.box_height, options.box_width, options.size

    details = []

    for (b1, b2) in itertools.product(range(size // box_height), range(size // box_width)):
        # if show_logs:
        #     print(f"Looking for boxed-wings in box {(b1 + 1, b2 + 1)}")

        # Use the fact that one wing should face the box row-wise, and one col-wise: Iterate through all rows/cols
        #  facing the box, and check whether the box contains at least one empty cell for these combinations

        for _b1, _b2 in itertools.product(range(box_height), range(box_width)):
            i1, i2 = b1 * box_height + _b1, b2 * box_width + _b2
            # if show_logs:
            #     print(f" Looking for empty cells in the box for row {i1 + 1} and col {i2 + 1}")

            options_row = set(itertools.chain.from_iterable(
                options[i1][b2 * box_width + __i2] for __i2 in range(box_width) if __i2 != _b2
            ))
            options_col = set(itertools.chain.from_iterable(
                options[b1 * box_height + __i1][i2] for __i1 in range(box_height) if __i1 != _b1
            ))
            shared_options = options_row.intersection(options_col)

            # The char cannot be present in the cell facing both wings
            options_facing_both_wings = options[i1][i2]

            # The char cannot be present in the cells not facing any wing -> invalid condition
            # options_not_facing_wings = set(itertools.chain.from_iterable(
            #     options[b1 * box_height + __b1][b2 * box_width + __b2]
            #     for (__b1, __b2) in itertools.product(range(box_height), range(box_width))
            #     if not (__b1 == _b1 or __b2 == _b2)
            # ))

            potential_wing_values = shared_options.difference(
                options_facing_both_wings  # .union(options_not_facing_wings)
            )
            # if show_logs:
            #     print("Possible wing values:", potential_wing_values)

            # Sort the potential wing chars to keep the order of applications consistent, especially for the user logs
            # TODO It would probably be more elegant to first collect all applications, and then sort based on some
            #  custom sort key to be defined
            for char in sorted(potential_wing_values):
                # if char not in options[i1][i2]:

                # Identify cells in the row/col containing the char, which will be a valid boxed-wing
                row_idxs_containing_char = [
                    _i1 for _i1 in range(size)
                    if _i1 // box_height != b1 and char in options[_i1][i2]
                ]
                col_idxs_containing_char = [
                    _i2 for _i2 in range(size)
                    if _i2 // box_width != b2 and char in options[i1][_i2]
                ]

                # The product defines all valid wings
                # for (_i1, _i2) in itertools.product(row_idxs_containing_char, col_idxs_containing_char):

                # Condition 4: The application is only valid when there is only one occurrence of the char in the row/col
                if len(row_idxs_containing_char) == len(col_idxs_containing_char) == 1:
                    _i1, _i2 = row_idxs_containing_char[0], col_idxs_containing_char[0]
                    shared_cell_outside_box = (_i1, _i2)

                    name_application = f"boxed-wing for '{char}' with box {(b1 + 1, b2 + 1)} and wings " \
                                       f"{(i1 + 1, _i2 + 1)} and {(_i1 + 1, i2 + 1)}"
                    if show_logs:
                        print(f"Found a {name_application}")
                        # print(f"Making it possible to remove wing value '{char}' from shared cell "
                        #       f"{tuple(map(lambda x: x + 1, shared_cell_outside_box))}")

                    removed_chars = []
                    if char in options[_i1][_i2]:
                        if show_logs:
                            print(f"Remove '{char}' from {tuple(map(lambda x: x + 1, shared_cell_outside_box))}")
                        removed_chars.append((shared_cell_outside_box, char))

                    idxs_wings = [(i1, _i2), (_i1, i2)]
                    application = ((b1, b2), idxs_wings, char)
                    details.append((name_application, application, removed_chars))

    # TODO Prioritise the applications which find a new value immediately? We already do this for chaining techniques,
    #  perhaps we can generalise this outside the function?

    return details
