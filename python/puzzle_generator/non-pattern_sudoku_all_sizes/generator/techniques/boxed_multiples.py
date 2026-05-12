
import itertools


def _find_boxed_multiples(options, chars, instance, multiple_number, show_logs=False):

    # Sharing some characteristics with boxed-wings and regular multiples, boxed-multiples infers a final pairing cell
    #  which can only contain the multiple chars;

    # Note: As the structure of the techniques is similar for doubles, triplets, etc. I aimed to generalise the logic
    #  just as for regular multiples

    # Implementation:
    #  Similar to boxed-wings, loop through all boxes, identify (partial) multiples outside of the box and check
    #  whether they satisfy the conditions to identify the final pairing cell

    # Steps:
    #  - For each box, identify combinations of (partial) multiples (eg for doubles: one cell with a pair, for triplets:
    #    two cells with a combined number of three options)
    #    -> added condition: all cells should contain pairs
    #  - For each empty cell in the box (potential target cells), determine whether the following conditions hold:
    #    - the cells in the row/col outside the box (arms) seen by the multiple cells are empty
    #    x the cells in the row/col outside the box (arms) not seen by the multiple cells do not contain any of the
    #      multiple chars as an option (either filled it or not an option)
    #    - modified condition: the multiple chars can only be present in the arms where they are already seen by a
    #      multiple cell
    #  - When those conditions are satisfied, all options but the multiple options can be removed from the target cell

    assert multiple_number in [2, 3, 4]

    # The boxed-x techniques are only implemented for and can only be applied to instances with default boxes layout
    assert not options.uses_custom_boxes_layout

    box_height, box_width, size = options.box_height, options.box_width, options.size

    details = []

    # Pre-processing: Identify cells with number of options of at most the multiple number
    relevant_idxs_with_options = {
        (i1, i2): _options
        for (i1, i2) in itertools.product(range(size), repeat=2)
        # if 2 <= len(_options := options[i1][i2]) <= multiple_number  # Don't include filled cells
        if len(_options := options[i1][i2]) == 2  # Added condition: all cells should contain pairs
    }

    # if show_logs:
    #     print("Relevant idxs with options:")
    #     for idx, _options in relevant_idxs_with_options.items():
    #         print(idx, _options)

    def is_empty(i1, i2):
        # Only empty cells have options
        return len(options[i1][i2]) > 0

    for (b1, b2) in itertools.product(range(size // box_height), range(size // box_width)):
        # print((b1, b2))

        relevant_idxs_outside_box_with_options = {
            (i1, i2): _options
            for (i1, i2), _options in relevant_idxs_with_options.items()
            if i1 // box_height != b1 and i2 // box_width != b2
        }

        # Make combinations
        combs_idxs = itertools.combinations(sorted(relevant_idxs_outside_box_with_options.keys()), multiple_number - 1)
        for comb_idxs in combs_idxs:

            # Combine options
            _options_combined = set(itertools.chain.from_iterable(
                relevant_idxs_with_options[idx] for idx in comb_idxs
            ))

            if len(_options_combined) == multiple_number:
                multiple = _options_combined
                # if show_logs:
                #     print(f"Checking whether box {(b1 + 1, b2 + 1)} has a valid target cell "
                #           f"for boxed-multiple {_options_combined} "
                #           f"in {tuple(map(lambda idx: tuple(map(lambda x: x + 1, idx)), comb_idxs))} ")

                # Identify potential target cells (empty cells in the box)
                idxs_potential_target = [
                    (i1, i2)
                    for (_b1, _b2) in itertools.product(range(box_height), range(box_width))
                    if is_empty((i1 := b1 * box_height + _b1), (i2 := b2 * box_width + _b2))
                ]

                for (i1, i2) in idxs_potential_target:
                    # if show_logs:
                    #     print(f"Checking potential target cell: {(i1, i2)}")

                    # Condition 1: Cells in arms seen by the multiple cells are empty
                    idxs_cells_seen = [(i1, _i2) for _i1, _i2 in comb_idxs] + [(_i1, i2) for _i1, _i2 in comb_idxs]
                    is_valid_condition_1 = all(is_empty(_i1, _i2) for (_i1, _i2) in idxs_cells_seen)

                    # print(" valid condition 1:", is_valid_condition_1)

                    # Modified condition 2: A check should be done for all chars of the multiple separately - the char
                    #  cannot be present in the arms where it is not seen by a multiple cell containing that char
                    is_valid_condition_2 = True
                    for char_multiple in multiple:
                        idxs_multiple_containing_char = [idx for idx in comb_idxs if char_multiple in relevant_idxs_with_options[idx]]

                        idxs_cells_not_seen = [
                            (_i1, i2) for _i1 in range(size)
                            if _i1 // box_height != b1 and _i1 not in [idx[0] for idx in idxs_multiple_containing_char]
                        ] + [
                            (i1, _i2) for _i2 in range(size)
                            if _i2 // box_width != b2 and _i2 not in [idx[1] for idx in idxs_multiple_containing_char]
                        ]

                        is_valid_condition_2_for_char = not any(
                            char_multiple in options[_i1][_i2] or instance[_i1][_i2] == char_multiple
                            for (_i1, _i2) in idxs_cells_not_seen
                        )

                        # print(f" idxs for char '{char_multiple}': {idxs_multiple_containing_char}")
                        # print(f" idxs cells not seen: {idxs_cells_not_seen}")
                        # print(f" valid condition 2 for '{char_multiple}':", is_valid_condition_2_for_char)

                        if not is_valid_condition_2_for_char:
                            is_valid_condition_2 = False
                            # break

                    is_valid = is_valid_condition_1 and is_valid_condition_2

                    if is_valid:
                        name_application = f"boxed-multiple {tuple(sorted(multiple))} for target cell {(i1 + 1, i2 + 1)} " \
                                           f"in box {(b1 + 1, b2 + 1)} and cells {[tuple(map(lambda x: x + 1, idx)) for idx in comb_idxs]}"
                        if show_logs:
                            print(f"Found a {name_application}")

                        removed_chars = []

                        # Update: There are 2 cases:
                        #  - The multiple cells occur all in different rows/cols (not boxs)
                        #    -> All options of the multiple could be contained in the target cell
                        #  - Some multiple cells occur in the same row/col (not box)
                        #    -> Only the options of the multiple not included in both multiple cells sharing the same
                        #       row/col could be contained in the target cell
                        options_target_cell = multiple.copy()
                        # For all combinations of multiple cells, check if they share a dimension (box/row/col), in
                        #  which case the shared value(s) (all multiple cells contain pairs) cannot occur in the target
                        #  cell
                        for comb_multiple_cells in itertools.combinations(comb_idxs, 2):
                            # print(f"Check whether {tuple(map(lambda idx: tuple(map(lambda x: x + 1, idx)), comb_multiple_cells))} share a dimension")
                            idx_multiple_cell_1, idx_multiple_cell_2 = comb_multiple_cells
                            have_shared_row = idx_multiple_cell_1[0] == idx_multiple_cell_2[0]
                            have_shared_col = idx_multiple_cell_1[1] == idx_multiple_cell_2[1]
                            # have_shared_box = \
                            #     (idx_multiple_cell_1[0] // box_height, idx_multiple_cell_1[1] // box_width) == \
                            #     (idx_multiple_cell_2[0] // box_height, idx_multiple_cell_2[1] // box_width)
                            have_shared_dimension = have_shared_row or have_shared_col  # or have_shared_box
                            if have_shared_dimension:
                                shared_chars = relevant_idxs_with_options[idx_multiple_cell_1].intersection(
                                    relevant_idxs_with_options[idx_multiple_cell_2]
                                )
                                if show_logs:
                                    print(f"Remove options {shared_chars} from the target cell as they are contained in "
                                          f"multiple cells {tuple(map(lambda idx: tuple(map(lambda x: x + 1, idx)), comb_multiple_cells))} "
                                          f"as they share one or more dimensions")
                                for char in shared_chars:
                                    try:
                                        options_target_cell.remove(char)
                                    except KeyError:
                                        pass

                        _chars_to_remove = options[i1][i2].difference(options_target_cell)
                        for char in _chars_to_remove:
                            if show_logs:
                                print(f"Remove '{char}' from {(i1 + 1, i2 + 1)}")
                            removed_chars.append(((i1, i2), char))

                        options_for_idxs = {idx: options[idx[0]][idx[1]] for idx in comb_idxs}
                        options_for_idxs.update({(i1, i2): options[i1][i2]})
                        application = ((b1, b2), (i1, i2), comb_idxs, options_for_idxs, multiple, options_target_cell)
                        details.append((name_application, application, removed_chars))

    return details


def find_boxed_doubles(options, chars, instance, show_logs=False):
    return _find_boxed_multiples(options, chars, instance, multiple_number=2, show_logs=show_logs)


def find_boxed_triplets(options, chars, instance, show_logs=False):
    return _find_boxed_multiples(options, chars, instance, multiple_number=3, show_logs=show_logs)


def find_boxed_quads(options, chars, instance, show_logs=False):
    return _find_boxed_multiples(options, chars, instance, multiple_number=4, show_logs=show_logs)
