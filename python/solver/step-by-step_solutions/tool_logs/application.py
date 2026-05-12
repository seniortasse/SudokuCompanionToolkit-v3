
import itertools

from generator.model import EMPTY_CHAR, DIMENSIONS


def identify_relevant_applications(step_instance_before, coords, char, dim_base, details):
    """
    Identify which applications in "details" helped find the final value, which are applications which
     - removed any option from the final cell if the final value was found using "cell" dimension
     - removed the final value as an option from any other cell in the dimension used to find the final value
    """
    assert dim_base in DIMENSIONS + ["cell"], f"Incorrect dim: {dim_base}"
    relevant_applications = []
    if dim_base == "cell":
        # Find all applications which removed a value from the value cell
        for names, application_details, removed_chars in details:
            # is_relevant = any(
            #     (idx_row, idx_col) == coords
            #     for (idx_row, idx_col), removed_char in removed_chars
            # )
            # if is_relevant:
            #     relevant_applications.append((application_details, removed_chars))
            relevant_removed_chars = [
                ((i1, i2), removed_char)
                for (i1, i2), removed_char in removed_chars
                if (i1, i2) == coords
            ]
            # Check: The final value cannot have been removed from the cell in which the final value was found
            assert not any(_char == char for _, _char in relevant_removed_chars)
            if len(relevant_removed_chars) > 0:
                relevant_applications.append((names, application_details, relevant_removed_chars))
    else:
        # Find all applications which removed the found value from one of the cells in the final dimension
        for names, application_details, removed_chars in details:
            # if is_relevant:
            #     relevant_applications.append((application_details, removed_chars))
            relevant_removed_chars = [
                ((i1, i2), removed_char)
                for (i1, i2), removed_char in removed_chars
                if removed_char == char and (
                    step_instance_before.get_idx_for_dim(dim_base, i1, i2) == step_instance_before.get_idx_for_dim(dim_base, *coords)
                )
            ]
            # Check: The final value cannot have been removed from the cell in which the final value was found
            assert not any(_coords == coords for _coords, _ in relevant_removed_chars)
            if len(relevant_removed_chars) > 0:
                relevant_applications.append((names, application_details, relevant_removed_chars))
    return relevant_applications


def identify_earlier_applications(application, step_instance_before, options_natural, all_applications):
    """
    Layer 2 is recursive - each application can have multiple applications that were required to enable it

    The logic consists of partly a custom piece of code for each technique, and once the unexplained removed options are
     identified the existing logic can be reused to identify relevant applications from the list of all details
    """

    # Extra steps for cleanup process:
    #  - Determine whether other techniques were needed to find the final value, and if so identify them
    #  - For all techniques needed to find the final value, trace back whether other techniques were needed to be
    #    able to apply them, and if so identify them

    # New logic: Iteratively determine whether another technique was needed to be able to apply the latest technique
    # This is custom for each technique

    (name_technique, _), application_details, removed_chars = application

    chars = step_instance_before.chars

    # Part 1 - Custom logic per technique
    # Identify which options are missing
    earlier_removed_chars = []

    # TODO There might be two ways of implementing:
    #  - Options removed in order for technique to become available
    #  - All options removed which had an impact on the structure of the technique;
    #    For example, a triplets where one of the cells contains only 2 options, where one of the options was removed
    #    earlier, should we consider this application or not? The technique would have been available without removing
    #    that option, but it did impact the details of the application

    # TODO This can probably be generalised, making the code much cleaner

    if name_technique in ["doubles", "triplets", "quads"]:
        multiple, idxs_multiple, _, dim_multiple = application_details

        # For multiples, identify whether there are any chars in the multiple present in the natural options for
        #  the (empty) cells not part of the multiple in the application dimension of the multiple, in other words,
        #  whether there are any empty cells not part of the multiple in the application dimension of the multiple which
        #  have a char from the multiple in their natural options
        # TODO For the alternative implementation, we should also identify chars included in the mutiple removed from
        #  a cell part of the multiple

        idxs = step_instance_before.get_idxs_in_dim(dim_multiple, *idxs_multiple[0])
        for (i1, i2) in idxs:
            # TODO The first check might be redundant as the options are only defined for empty cells
            if step_instance_before[i1][i2] == EMPTY_CHAR and (i1, i2) not in idxs_multiple:
                chars_present = options_natural[i1][i2].intersection(multiple)
                if len(chars_present) > 0:
                    earlier_removed_chars.extend((((i1, i2), char) for char in chars_present))

    elif name_technique == "singles-pointing":
        char_pointing, _, idxs_options, direction, idx_dim_pointing = application_details

        # TODO Try to include as much processing in the technique itself
        assert direction in ["hor", "ver"]
        dim_pointing = "row" if direction == "hor" else "col"

        idxs_box = step_instance_before.get_idxs_in_dim("box", *idxs_options[0])
        for (i1, i2) in idxs_box:
            if step_instance_before[i1][i2] == EMPTY_CHAR and (
                # Note: We have to check all cells in the dim, as options removed in the allowed cells (even if they are
                #  not present) do not contribute to enabling this application
                i1 != idx_dim_pointing if dim_pointing == "row" else i2 != idx_dim_pointing
            ):
                if char_pointing in options_natural[i1][i2]:
                    earlier_removed_chars.append(((i1, i2), char_pointing))

    elif name_technique == "singles-boxed":
        dim_boxed, char_boxed, idx_base, idx_box, _ = application_details

        assert dim_boxed in ["row", "col"]

        for (i1, i2) in step_instance_before.idxs_for_dims[dim_boxed][idx_base]:
            # Similar to singles-pointing, only options removed outside of the box contribute to enabling this
            #  application and are relevant
            # TODO The empty char check can probably be removed
            if step_instance_before[i1][i2] == EMPTY_CHAR and step_instance_before.get_idx_box(i1, i2) != idx_box:
                if char_boxed in options_natural[i1][i2]:
                    earlier_removed_chars.append(((i1, i2), char_boxed))

    elif name_technique in ["x-wings", "x-wings-3", "x-wings-4"]:
        # TODO Generalise
        if name_technique == "x-wings":
            dim_wing, char_wing, _idxs_rows, _idxs_cols = application_details
            idxs_wing, idxs_help = (_idxs_rows, _idxs_cols) if dim_wing == "row" else (_idxs_cols, _idxs_rows)
        else:
            dim_wing, char_wing, _idxs_wing_dim, _mapping_idxs_help_dim = application_details
            # Some extra processing for the larger x-wings
            idxs_wing = _idxs_wing_dim
            idxs_help = sorted(set(itertools.chain.from_iterable(_mapping_idxs_help_dim.values())))

        # For x-wings, all occurrences of the wing value in the dimension of the wing that were not there when applying
        #  the technique are relevant
        for idx in idxs_wing:
            for (i1, i2) in step_instance_before.get_idxs_in_dim(dim_wing, idx, idx):  # The hack
                if step_instance_before[i1][i2] == EMPTY_CHAR and (i2 if dim_wing == "row" else i1) not in idxs_help:
                    if char_wing in options_natural[i1][i2]:
                        earlier_removed_chars.append(((i1, i2), char_wing))

    elif name_technique == "ab-chains":
        chain_conflict, options_for_idxs, conflicting_value = application_details

        # Technique specific: The entire chain contains only pairs, as used during the application
        for idx in chain_conflict:
            options_nat = options_natural[idx[0]][idx[1]]
            options_app = options_for_idxs[idx]
            assert len(options_app) == 2
            assert len(options_nat) >= 2
            assert not options_app.difference(options_nat)
            options_removed = options_nat.difference(options_app)
            for char in options_removed:
                earlier_removed_chars.append((idx, char))

    elif name_technique == "remote-pairs":
        idx_start, valid_chain, options_for_idxs = application_details

        # Note: The required earlier removed chars depend on the specifics of the technique implementation; If it
        #  requires to start in a cell with a pair, we have to take this into account, and update this logic when the
        #  implementation changes; This motivates looking for a more general implementation, which can probably be
        #  found by:
        #  Trying to apply this technique using only the natural options; If it is not available, it required removing
        #  some options; Although it does not identify which options need to be removed..
        #  -> This would be way more elegant, as well as fault-tolerant, but seems too difficult to implement, as it
        #     would require identifying the min possible set of options to be removed in order for the technique to
        #     become available, which is an optimisation problem and requires a lot of computing

        # Technique specific: The starting point should contain a pair
        options_start_nat = options_natural[idx_start[0]][idx_start[1]]
        options_start_app = options_for_idxs[idx_start]
        assert len(options_start_app) == 2
        assert len(options_start_nat) >= 2
        assert not options_start_app.difference(options_start_nat)
        options_removed = options_start_nat.difference(options_start_app)
        for char in options_removed:
            earlier_removed_chars.append((idx_start, char))

        # Technique specific: The entire chain should contain a pair, with specific values (the ones present in the
        #  options used within the technique, as contained in the details)
        for idx in valid_chain:
            options_idx_nat = options_natural[idx[0]][idx[1]]
            options_idx_app = options_for_idxs[idx]
            assert len(options_idx_app) == 2
            assert len(options_idx_nat) >= 2
            assert not options_idx_app.difference(options_idx_nat)
            options_removed = options_idx_nat.difference(options_idx_app)
            for char in options_removed:
                earlier_removed_chars.append((idx, char))

    elif name_technique == "y-wings":
        idxs_wings, pairs_wings, idx_center, options_center, char_wing = application_details

        # Technique specific: All options in the wings/center not present when applying the technique
        for idx, pair in zip(idxs_wings + (idx_center, ), pairs_wings + (options_center, )):
            options_nat = options_natural[idx[0]][idx[1]]
            options_app = pair
            assert len(options_app) == 2
            assert len(options_nat) >= 2
            assert not options_app.difference(options_nat)
            options_removed = options_nat.difference(options_app)
            for char in options_removed:
                earlier_removed_chars.append((idx, char))

    elif name_technique in ["doubles-naked", "triplets-naked", "quads-naked"]:
        _, _, idxs_multiple, _, multiple = application_details

        # All options removed from the cells containing the multiple, besides the multiple chars themselves

        for (i1, i2) in idxs_multiple:
            chars_removed_earlier = options_natural[i1][i2].difference(multiple)
            for char in chars_removed_earlier:
                earlier_removed_chars.append(((i1, i2), char))

    elif name_technique in ["boxed-doubles", "boxed-triplets", "boxed-quads"]:
        (b1_target, b2_target), (i1_target, i2_target), comb_idxs, options_for_idxs, multiple, options_target_cell = application_details

        # All options removed from the cells containing the multiple, besides the multiple chars themselves,
        #  plus the options removed from the empty cells in the arms not seen by the multiple

        for (_i1, _i2) in comb_idxs:
            # The updated implementation requires all cells to contain multiples
            options_for_idx = options_for_idxs[(_i1, _i2)]
            chars_removed_earlier = options_natural[_i1][_i2].difference(options_for_idx)
            for char in chars_removed_earlier:
                earlier_removed_chars.append(((_i1, _i2), char))

        # The updated implementation requires that the cells in the arms cannot contain an option not already seen by
        #  a multiple cell
        for char in multiple:
            idxs_multiple_containing_char = [idx for idx in comb_idxs if char in options_for_idxs[idx]]

            idxs_rows = [idx[0] for idx in idxs_multiple_containing_char]
            idxs_cols = [idx[1] for idx in idxs_multiple_containing_char]

            size, box_height, box_width = options_natural.size, options_natural.box_height, options_natural.box_width

            # Col-based arm
            for _i1 in range(size):
                if _i1 not in idxs_rows and _i1 // box_height != b1_target:
                    # chars_removed_earlier = options_natural[_i1][i2_target].intersection(_options_combined)
                    # for char in chars_removed_earlier:
                    if char in options_natural[_i1][i2_target]:
                        earlier_removed_chars.append(((_i1, i2_target), char))

            # Row-based arm
            for _i2 in range(size):
                if _i2 not in idxs_cols and _i2 // box_width != b2_target:
                    # chars_removed_earlier = options_natural[i1_target][_i2].intersection(_options_combined)
                    # for char in chars_removed_earlier:
                    if char in options_natural[i1_target][_i2]:
                        earlier_removed_chars.append(((i1_target, _i2), char))

    elif name_technique == "boxed-wings":
        (b1_target, b2_target), idxs_wings, char_wing = application_details

        size, box_height, box_width = options_natural.size, options_natural.box_height, options_natural.box_width

        # All occurrences of the wing char in the arms outside the target box

        i1_wing_hor, i2_wing_hor = idxs_wings[0]
        i1_wing_ver, i2_wing_ver = idxs_wings[1]
        i1, i2 = i1_wing_hor, i2_wing_ver

        # Col-based arm
        for _i1 in range(size):
            if _i1 != i1_wing_ver and _i1 // box_height != b1_target:
                if char_wing in options_natural[_i1][i2]:
                    earlier_removed_chars.append(((_i1, i2), char_wing))

        # Row-based arm
        for _i2 in range(size):
            if _i2 != i2_wing_hor and _i2 // box_width != b2_target:
                if char_wing in options_natural[i1][_i2]:
                    earlier_removed_chars.append(((i1, _i2), char_wing))

    elif name_technique == "boxed-rays":
        (b1_target, b2_target), (i1_target, i2_target), idxs_target, (b1_ray, b2_ray), (i1_ray, i2_ray), idxs_ray, char_ray = application_details

        size, box_height, box_width = options_natural.size, options_natural.box_height, options_natural.box_width

        # All options of the ray char in the box and in the arms outside of the target box not seen by the ray

        for (_b1, _b2) in itertools.product(range(box_height), range(box_width)):
            _i1, _i2 = b1_ray * box_height + _b1, b2_ray * box_width + _b2

            if _i1 != i1_ray and _i2 != i2_ray:
                if char_ray in options_natural[_i1][_i2]:
                    earlier_removed_chars.append(((_i1, _i2), char_ray))

        # Col-based arm
        for _i1 in range(size):
            if _i1 != i1_ray and _i1 // box_height != b1_target:
                if char_ray in options_natural[_i1][i2_target]:
                    earlier_removed_chars.append(((_i1, i2_target), char_ray))

        # Row-based arm
        for _i2 in range(size):
            if _i2 != i2_ray and _i2 // box_width != b2_target:
                if char_ray in options_natural[i1_target][_i2]:
                    earlier_removed_chars.append(((i1_target, _i2), char_ray))

    elif name_technique == "ab-rings":
        idxs_ring, options_for_idxs, combined_options = application_details

        # All options removed from the ring corners, besides the pair present during application

        for (i1, i2) in idxs_ring:
            chars_removed_earlier = options_natural[i1][i2].difference(options_for_idxs[(i1, i2)])
            for char in chars_removed_earlier:
                earlier_removed_chars.append(((i1, i2), char))

    elif name_technique in [f"leftovers-{i}" for i in range(1, 9 + 1)]:
        dim, region, box_idxs_inside, box_idxs_outside, idxs_cells_inside, idxs_cells_outside, num_cells_inside, options_inside, options_outside, options = application_details

        # It seems that this technique does not require any previous applications to be enabled?
        # Wrong! Any options removed from the cells inside/outside which then enabled those options to be removed from
        #  the cells outside/inside were enabling

        # TODO Only the removed options which were actually used to remove options with this application are relevant,
        #  but it seems this is automatically postprocessed? -> no, it is not

        # Note: This technique requires some special processing, as while usually techniques can only be applied under
        #  specific conditions, this technique can always be applied, but the removed options are based on the options
        #  present in the cells inside/outside the region;
        # For this technique, we have to check which options were actually removed (which already only contains the
        #  relevant options as filtered by identify_relevant_applications() and subsequently recursively at the end of
        #  this function) - and identify whether this option first had to be removed from the other side
        #  (inside/outside) by another application;

        for idx_removed_char, removed_char in removed_chars:
            # Options are always only removed from the cells inside/outside the region
            assert (idx_removed_char in idxs_cells_inside) + (idx_removed_char in idxs_cells_outside) == 1

            is_removed_char_inside = idx_removed_char in idxs_cells_inside
            if is_removed_char_inside:
                # Check whether this char was removed by another application from the cells outside the region
                for (i1, i2) in idxs_cells_outside:
                    if removed_char in options_natural[i1][i2]:
                        earlier_removed_chars.append(((i1, i2), removed_char))
            else:
                # Check whether this char was removed by another application from the cells inside the region
                for (i1, i2) in idxs_cells_inside:
                    if removed_char in options_natural[i1][i2]:
                        earlier_removed_chars.append(((i1, i2), removed_char))

    else:
        raise Exception(f"Undefined logic for identifying earlier applications for technique: {name_technique}")

    if len(earlier_removed_chars) > 0:
        # Application {name_application}
        print(f" Could only be applied after removing {len(earlier_removed_chars)} options earlier:",
              [(tuple(map(lambda x: x + 1, idx)), char) for idx, char in earlier_removed_chars])
    else:
        print(f" Did not require any earlier applications")

    # Part 2 - Identify relevant applications
    earlier_applications = []
    for removed_char in earlier_removed_chars:
        # Trace back which application removed this option
        relevant_applications = [
            (_names, _application_details, _removed_chars)
            for (_names, _application_details, _removed_chars) in all_applications
            if removed_char in _removed_chars
        ]
        assert len(relevant_applications) == 1, \
            f"Relevant application not correctly identified for {removed_char}: {len(relevant_applications)}"
        earlier_application = relevant_applications[0]

        # Filter the removed chars relevant to this application, to be used in the message
        earlier_application = (*earlier_application[:-1], [removed_char])
        earlier_applications.append(earlier_application)

        print(f"  Option '{removed_char[1]}' at {tuple(map(lambda x: x + 1, removed_char[0]))} was removed by:", earlier_application[0][1])

    # TODO Filter duplicate application: Some application might remove multiple relevant options
    sort_key = lambda application: application[:-1]
    earlier_applications = sorted(earlier_applications, key=sort_key)
    grouper = itertools.groupby(earlier_applications, sort_key)

    grouped_applications = []
    for key, group in grouper:
        # Unpack
        group = list(group)
        if len(group) > 1:
            print(f"  Application {key[0][1]} removed multiple options: {len(group)}")
        grouped_removed_chars = list(itertools.chain.from_iterable(application[-1] for application in group))
        grouped_applications.append((*key, grouped_removed_chars))

    # TODO Which removed chars are relevant for an application might depend on the application it enabled, so we have
    #  to make sure the application initially contains all removed chars, which gets filtered for its specific
    #  application in this function

    return grouped_applications
