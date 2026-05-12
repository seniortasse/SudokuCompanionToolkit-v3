
from collections import defaultdict
import itertools
import operator

from generator.model import EMPTY_CHAR, DIMENSIONS


def process_formatting(application, step_instance_after, coords):

    # Unpack tuple
    (name_technique, _), application_details, removed_chars = application

    # Initialise collections
    idxs_bg_color_cells = []
    show_values = defaultdict(list)
    idxs_claiming_options = defaultdict(list)
    # Used for now only for new technique "leftovers", but can be expanded to further customise the formatting
    idxs_bg_color_cells_highlighted = []

    if name_technique in ["doubles", "triplets", "quads"]:
        multiple, idxs_multiple, options_for_idxs, dim_multiple = application_details

        # Show multiple
        for (i1, i2) in idxs_multiple:
            options_for_idx = options_for_idxs[(i1, i2)]
            show_values[(i1, i2)].append(options_for_idx)
            idxs_claiming_options[(i1, i2)].extend(options_for_idx)

        dims_help_multiple = [dim for dim in DIMENSIONS if dim not in dim_multiple]

        # Background coloring for doubles
        # TODO Reuse shared logic -- now it is not consistent with the latest improvements
        for (i1, i2) in step_instance_after.get_idxs_in_dim(dim_multiple, *idxs_multiple[0]):
            # Note: The value cell was initially also empty and possibly part of the technique
            # TODO Check whether this could also occur for other techniques
            if (step_instance_after[i1][i2] == EMPTY_CHAR or (i1, i2) == coords) and (i1, i2) not in idxs_multiple:
                # Only apply background coloring if there actually is a value present in one of the help dims
                should_color = False
                for dim_help_multiple in dims_help_multiple:
                    for (_i1, _i2) in step_instance_after.get_idxs_in_dim(dim_help_multiple, i1, i2):
                        if step_instance_after[_i1][_i2] in multiple:
                            should_color = True
                            idxs_bg_color_cells.append((_i1, _i2))
                if should_color:
                    idxs_bg_color_cells.append((i1, i2))

    elif name_technique == "singles-pointing":

        # TODO We should introduce a general structure of returning details, and a single function which determines
        #  which applications were relevant to find the new value based on how it was identified

        # Note: For formatting we only need the relevant application details
        char_pointing, _, idxs_options, direction, idx_dim_pointing = application_details

        # Show ray chars
        for (i1, i2) in idxs_options:
            show_values[(i1, i2)].append(f"-{char_pointing}")
            idxs_claiming_options[(i1, i2)].append(char_pointing)

        for (i1, i2) in step_instance_after.get_idxs_in_dim("box", *idxs_options[0]):
            if step_instance_after[i1][i2] == EMPTY_CHAR and (i1, i2) not in idxs_options:
                # TODO Reuse this from generalised structure
                should_color = False
                dims_help = ["row", "col"]
                for dim_help in dims_help:
                    for (_i1, _i2) in step_instance_after.get_idxs_in_dim(dim_help, i1, i2):
                        if step_instance_after[_i1][_i2] == char_pointing and (_i1, _i2) != coords:
                            should_color = True
                            idxs_bg_color_cells.append((_i1, _i2))
                if should_color:
                    idxs_bg_color_cells.append((i1, i2))

        # Final value (for this technique not formatted?)
        # TODO Reuse for all advanced techniques

    elif name_technique == "singles-boxed":
        dim_boxed, char_boxed, _, _, idxs_char = application_details

        # Show boxed char
        for (i1, i2) in idxs_char:
            show_values[(i1, i2)].append(f"-{char_boxed}")
            idxs_claiming_options[(i1, i2)].append(char_boxed)

        # Identify key cells
        for (i1, i2) in step_instance_after.get_idxs_in_dim(dim_boxed, *idxs_char[0]):
            if step_instance_after[i1][i2] == EMPTY_CHAR and (i1, i2) not in idxs_char:
                # TODO Reuse this from generalised structure
                should_color = False
                dims_help = [dim for dim in DIMENSIONS if dim != dim_boxed]
                for dim_help in dims_help:
                    for (_i1, _i2) in step_instance_after.get_idxs_in_dim(dim_help, i1, i2):
                        if step_instance_after[_i1][_i2] == char_boxed and (_i1, _i2) != coords:
                            should_color = True
                            idxs_bg_color_cells.append((_i1, _i2))
                if should_color:
                    idxs_bg_color_cells.append((i1, i2))

    elif name_technique == "x-wings":

        # Note: For formatting we only need the relevant application details

        dim_wing, char_wing, idxs_rows, idxs_cols = application_details

        assert dim_wing in ["row", "col"]
        idxs_base, idxs_help = (idxs_rows, idxs_cols) if dim_wing == "row" else (idxs_cols, idxs_rows)

        # Show wing chars
        for (i1, i2) in itertools.product(idxs_rows, idxs_cols):
            show_values[(i1, i2)].append(f"-{char_wing}")
            # cell.font = Font(name=cell.font.name, size=cell.font.size, color="95b3d7")
            idxs_claiming_options[(i1, i2)].append(char_wing)

        for idx in idxs_base:
            for (i1, i2) in step_instance_after.get_idxs_in_dim(dim_wing, idx, idx):  # A hack which returns the correct idxs regardless of which dim was the base
                idxs_bg_color_cells.append((i1, i2))

    # TODO Try to generalise with x-wings
    elif name_technique in ["x-wings-3", "x-wings-4"]:
        dim_wing, char_wing, idxs_wing_dim, mapping_idxs_help_dim = application_details

        assert dim_wing in ["row", "col"]
        if dim_wing == "row":
            idxs_wing_char = [
                (idx_wing, idx_help)
                for idx_wing in idxs_wing_dim
                for idx_help in mapping_idxs_help_dim[idx_wing]
            ]
        else:
            idxs_wing_char = [
                (idx_help, idx_wing)
                for idx_wing in idxs_wing_dim
                for idx_help in mapping_idxs_help_dim[idx_wing]
            ]

        # Show wing chars
        for (i1, i2) in idxs_wing_char:
            show_values[(i1, i2)].append(f"-{char_wing}")
            idxs_claiming_options[(i1, i2)].append(char_wing)

        for idx in idxs_wing_dim:
            for (i1, i2) in step_instance_after.get_idxs_in_dim(dim_wing, idx, idx):
                idxs_bg_color_cells.append((i1, i2))

    elif name_technique == "ab-chains":
        chain_conflict, options_for_idxs, conflicting_value = application_details

        # Show chain chars
        # Don't show the options for the cell for which the new value was found, which is the first cell of the
        #  chain for which the conflict occurred
        for (i1, i2) in chain_conflict[1:]:
            options_for_idx = options_for_idxs[(i1, i2)]
            show_values[(i1, i2)].append(options_for_idx)

        idxs_bg_color_cells.extend(chain_conflict[1:])

    # TODO See if we can merge this logic with ab-chains
    elif name_technique == "remote-pairs":
        idx_start, valid_chain, options_for_idxs = application_details

        # Show chain chars
        for (i1, i2) in valid_chain:
            options_for_idx = options_for_idxs[(i1, i2)]
            show_values[(i1, i2)].append(options_for_idx)

        idxs_bg_color_cells.extend(valid_chain)

    elif name_technique == "y-wings":
        idxs_wings, pairs_wings, idx_center, options_center, char_wing = application_details

        # Show wing chars
        for (i1, i2), options_for_idx in zip(idxs_wings, pairs_wings):
            show_values[(i1, i2)].append(options_for_idx)
            # TODO Claiming options
            idxs_claiming_options[(i1, i2)].extend(char_wing)

        # Show center chars
        show_values[idx_center].append(options_center)

        idxs_bg_color_cells.extend(idxs_wings)
        idxs_bg_color_cells.append(idx_center)

    elif name_technique in ["doubles-naked", "triplets-naked", "quads-naked"]:
        _, _, idxs_multiple, options_for_idxs, _ = application_details

        # Show multiple
        for idx in idxs_multiple:
            options_for_idx = options_for_idxs[idx]
            show_values[idx].append(options_for_idx)
            idxs_claiming_options[idx].extend(options_for_idx)

        # To avoid clogging the coloring, we do not apply any background coloring here

    elif name_technique in ["boxed-doubles", "boxed-triplets", "boxed-quads"]:
        (b1_target, b2_target), (i1_target, i2_target), comb_idxs, options_for_idxs, multiple, options_target_cell = application_details

        # Show multiple
        for idx in comb_idxs:
            options_for_idx = options_for_idxs[idx]
            show_values[idx].append(options_for_idx)
            idxs_claiming_options[idx].extend(options_for_idx)

        # Show mirroring multiple
        # Note: Only show the options present at the time of application
        options_for_target = options_for_idxs[(i1_target, i2_target)].intersection(options_target_cell)
        show_values[(i1_target, i2_target)].append(options_for_target)
        idxs_claiming_options[(i1_target, i2_target)].extend(options_for_target)

        # Background color
        for (i1_multiple, i2_multiple) in comb_idxs:
            i1_from, i1_to = sorted([i1_target, i1_multiple])
            i2_from, i2_to = sorted([i2_target, i2_multiple])
            idxs_square = \
                [(_i1, _i2) for _i1 in [i1_target, i1_multiple] for _i2 in range(i2_from, i2_to + 1)] + \
                [(_i1, _i2) for _i2 in [i2_target, i2_multiple] for _i1 in range(i1_from, i1_to + 1)]
            idxs_bg_color_cells.extend(idxs_square)

    elif name_technique == "boxed-wings":
        (b1_target, b2_target), idxs_wings, char_wing = application_details

        size, box_height, box_width = step_instance_after.size, step_instance_after.box_height, step_instance_after.box_width

        # Show wing chars
        for idx in idxs_wings:
            show_values[idx].append(f"-{char_wing}")
            idxs_claiming_options[idx].append(char_wing)

        # Background color
        for (_b1, _b2) in itertools.product(range(box_height), range(box_width)):
            i1, i2 = b1_target * box_height + _b1, b2_target * box_width + _b2
            idxs_bg_color_cells.append((i1, i2))

        idxs_bg_color_cells.extend(idxs_wings)

    elif name_technique == "boxed-rays":
        (b1_target, b2_target), (i1_target, i2_target), idxs_target, (b1_ray, b2_ray), (i1_ray, i2_ray), idxs_ray, char_ray = application_details

        # Show ray chars
        for idx in idxs_ray:
            show_values[idx].append(f"-{char_ray}")
            idxs_claiming_options[idx].append(char_ray)

        # Also show the ray chars in the target box
        for idx in idxs_target:
            show_values[idx].append(f"-{char_ray}")
            idxs_claiming_options[idx].append(char_ray)

        # Background color
        # TODO Reuse functionality of determining idxs in square from ab-rings
        i1_from, i1_to = sorted([i1_target, i1_ray])
        i2_from, i2_to = sorted([i2_target, i2_ray])
        idxs_square = \
            [(_i1, _i2) for _i1 in [i1_target, i1_ray] for _i2 in range(i2_from, i2_to + 1)] + \
            [(_i1, _i2) for _i2 in [i2_target, i2_ray] for _i1 in range(i1_from, i1_to + 1)]
        idxs_bg_color_cells.extend(idxs_square)

        # In case the rays pointing the "opposite" direction, also color these extending arms
        i1_ray_from, i1_ray_to = operator.itemgetter(0, -1)(sorted([idx[0] for idx in idxs_ray if idx[1] == i2_ray]))
        i2_ray_from, i2_ray_to = operator.itemgetter(0, -1)(sorted([idx[1] for idx in idxs_ray if idx[0] == i1_ray]))
        for i1 in range(i1_ray_from, i1_ray_to + 1):
            idxs_bg_color_cells.append((i1, i2_ray))
        for i2 in range(i2_ray_from, i2_ray_to + 1):
            idxs_bg_color_cells.append((i1_ray, i2))

        i1_target_from, i1_target_to = operator.itemgetter(0, -1)(sorted([idx[0] for idx in idxs_target if idx[1] == i2_target]))
        i2_target_from, i2_target_to = operator.itemgetter(0, -1)(sorted([idx[1] for idx in idxs_target if idx[0] == i1_target]))
        for i1 in range(i1_target_from, i1_target_to + 1):
            idxs_bg_color_cells.append((i1, i2_target))
        for i2 in range(i2_target_from, i2_target_to + 1):
            idxs_bg_color_cells.append((i1_target, i2))

    elif name_technique in ["ab-rings"]:
        idxs_ring, options_for_idxs, combined_options = application_details

        # Show ring chars
        for idx in idxs_ring:
            options_for_idx = options_for_idxs[idx]
            show_values[idx].append(options_for_idx)

        # Background color
        # TODO Reuse this logic from boxed-multiples
        for i in range(4):
            idx_from, idx_to = idxs_ring[i - 1], idxs_ring[i]
            i1_from, i1_to = sorted([idx_from[0], idx_to[0]])
            i2_from, i2_to = sorted([idx_from[1], idx_to[1]])
            idxs_edge = list(itertools.product(range(i1_from, i1_to + 1), range(i2_from, i2_to + 1)))
            idxs_bg_color_cells.extend(idxs_edge)

    elif name_technique in [f"leftovers-{i}" for i in range(1, 9 + 1)]:
        # TODO By default add the options to the application details, as they are not modified during the application
        #  process now, similar to the instance, which is also not modified throughout the process
        dim, region, box_idxs_inside, box_idxs_outside, idxs_cells_inside, idxs_cells_outside, num_cells_inside, options_inside, options_outside, options = application_details

        # TODO We might need more customisation to make a proper coloring for this technique

        # Color the region and cells outside the region
        for idx_dim in region:
            idxs_bg_color_cells.extend(step_instance_after.idxs_for_dims[dim][idx_dim])

        idxs_bg_color_cells.extend(idxs_cells_outside)

        # Highlight the ins/outs
        idxs_bg_color_cells_highlighted.extend(idxs_cells_inside)
        idxs_bg_color_cells_highlighted.extend(idxs_cells_outside)

        # TODO How do show_values and idxs_claiming_options work again?

        # Add the options for the ins/outs
        for (i1, i2) in idxs_cells_inside + idxs_cells_outside:
            # TODO The sorting and joining should be done when writing to file, this is just too much redundant code
            # Only show for empty cells!
            # TODO It would be better to use step_instance_before here, but some more logic has to be rewritten to be
            #  able to do this; Here both can be used as the final value cell is overwritten with the final value
            #  anyways;
            if step_instance_after[i1][i2] == EMPTY_CHAR:
                show_values[(i1, i2)].append(options[i1][i2])

    else:
        raise Exception(f"Undefined formatting for technique: {name_technique}")
        # pass

    return idxs_bg_color_cells, show_values, idxs_claiming_options, idxs_bg_color_cells_highlighted
