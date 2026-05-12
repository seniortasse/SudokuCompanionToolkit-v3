
import itertools


def find_boxed_rays(options, chars, instance, show_logs=False):

    # Another "boxed" technique, boxed rays requires the cooperation of two boxes: one box containing a bent ray, ie
    #  only options for a value in one row & col combination, and some extra conditions have to hold to identify a
    #  target box:
    #  - The cells in the arm seeing the bent ray are empty
    #  - The cells in the arm not seeing the bent ray do not contain the ray value (neither filled in nor as an option)
    #  - The ray contains at least 2 empty cells with the ray value as an option in two directions
    #  This results in being able to remove the ray value from the target box outside of the arms

    # Implementation:
    #  Instead of starting from a target box, as for the other boxed techniques, here we start by identifying bent rays,
    #  and for all potential target boxes see if they quality by satisfying all conditions

    # Steps:
    #  - For each box, identify whether it contains a bent ray for each character
    #  - For each identified bent ray, check whether there is a target box which satisfies all conditions

    # The boxed-x techniques are only implemented for and can only be applied to instances with default boxes layout
    assert not options.uses_custom_boxes_layout

    box_height, box_width, size = options.box_height, options.box_width, options.size

    details = []

    # Identify boxed rays
    rays = []
    for (b1, b2) in itertools.product(range(size // box_height), range(size // box_width)):
        for char in sorted(chars):
            # Updated logic: check whether there is a combination of row & col which contains all options for the char,
            #  and there is at least one char in the row and one in the col outside where they meet
            # Note: When there is only one value in the row and one in the col, the ray can be detected in two ways, and
            #  both are valid rays
            idxs_box = [
                (b1 * box_height + _b1, b2 * box_width + _b2)
                for (_b1, _b2) in itertools.product(range(box_height), range(box_width))
            ]
            idxs_char = [(i1, i2) for (i1, i2) in idxs_box if char in options[i1][i2]]
            for (_b1, _b2) in itertools.product(range(box_height), range(box_width)):
                i1, i2 = b1 * box_height + _b1, b2 * box_width + _b2
                row_contains = any(
                    (_i1, _i2) in idxs_char
                    for (_i1, _i2) in idxs_box
                    if _i1 == i1 and _i2 != i2
                )
                col_contains = any(
                    (_i1, _i2) in idxs_char
                    for (_i1, _i2) in idxs_box
                    if _i2 == i2 and _i1 != i1
                )
                no_options_outside_ray = not any(
                    (_i1, _i2) in idxs_char
                    for (_i1, _i2) in idxs_box
                    if not (_i1 == i1 or _i2 == i2)
                )
                if row_contains and col_contains and no_options_outside_ray:
                    if show_logs:
                        print(f"Identified bent ray for '{char}' in box {(b1 + 1, b2 + 1)} in row {i1 + 1} and col {i2 + 1}")
                    idxs_ray = [(_i1, _i2) for (_i1, _i2) in idxs_box if (_i1, _i2) in idxs_char]
                    rays.append(((b1, b2), (i1, i2), idxs_ray, char))

    # Search for target boxes
    for ray in rays:
        (b1_ray, b2_ray), (i1_ray, i2_ray), idxs_ray, char_ray = ray
        for (b1, b2) in itertools.product(range(size // box_height), range(size // box_width)):
            if b1 != b1_ray and b2 != b2_ray:
                # if show_logs:
                #     print(f"Searching for valid target boxes for bent ray {ray}")
                for (_b1, _b2) in itertools.product(range(box_height), range(box_width)):
                    # These are the arms of the target box
                    i1, i2 = b1 * box_height + _b1, b2 * box_width + _b2

                    # Condition 1: Cells in the arms seen by the bent ray are empty
                    idxs_cells_seen = [(i1, i2_ray), (i1_ray, i2)]

                    # Condition 2: Cells in the arms not seen by the bent ray do not contain the ray char
                    idxs_cells_not_seen = [
                        (i1, _i2) for _i2 in range(size)
                        if _i2 // box_width != b2 and _i2 != i2_ray
                    ] + [
                        (_i1, i2) for _i1 in range(size)
                        if _i1 // box_height != b1 and _i1 != i1_ray
                    ]

                    is_valid = \
                        all(
                            len(options[_i1][_i2]) > 0
                            for (_i1, _i2) in idxs_cells_seen
                        ) and \
                        not any(
                            char_ray in options[_i1][_i2] or instance[_i1][_i2] == char_ray
                            for (_i1, _i2) in idxs_cells_not_seen
                        )

                    if is_valid:
                        name_application = f"boxed-ray for '{char_ray}' in box {(b1_ray + 1, b2_ray + 1)} with " \
                                           f"row {i1_ray + 1} and col {i2_ray + 1} and target box {(b1 + 1, b2 + 1)} and " \
                                           f"arms row {i1 + 1} and col {i2 + 1}"
                        if show_logs:
                            print(f"Found a {name_application}, removing the ray char from the other cells in the target box")

                        removed_chars = []
                        for (__b1, __b2) in itertools.product(range(box_height), range(box_width)):
                            if __b1 != _b1 and __b2 != _b2:
                                _i1, _i2 = b1 * box_height + __b1, b2 * box_width + __b2
                                if char_ray in options[_i1][_i2]:
                                    if show_logs:
                                        print(f"Remove '{char_ray}' from {(_i1 + 1, _i2 + 1)}")
                                    removed_chars.append(((_i1, _i2), char_ray))

                        # Used for formatting
                        idxs_target = sorted(set([
                            (_i1, _i2)
                            for __b1 in range(box_height)
                            if char_ray in options[(_i1 := b1 * box_height + __b1)][(_i2 := b2 * box_width + _b2)]
                        ] + [
                            (_i1, _i2)
                            for __b2 in range(box_width)
                            if char_ray in options[(_i1 := b1 * box_height + _b1)][(_i2 := b2 * box_width + __b2)]
                        ])) # idxs in the target box containing the ray char

                        application = ((b1, b2), (i1, i2), idxs_target, (b1_ray, b2_ray), (i1_ray, i2_ray), idxs_ray, char_ray)
                        details.append((name_application, application, removed_chars))

    return details
