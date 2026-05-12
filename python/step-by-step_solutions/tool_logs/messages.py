
import itertools

from generator.algo_human import ADVANCED_TECHNIQUES, TECHNIQUES
from generator.model import DIMENSIONS

from tool_logs.template import LENGTH
from tool_logs.layers import identify_layers
import tool_logs.messages_templates as messages_templates
from tool_logs.messages_templates import templates_messages, templates_messages_advanced, MAP_DIMENSIONS, \
    MAP_DIRECTIONS, MAP_COUNTS, MAP_SIZE_TO_NAME, MAP_TECHNIQUE_NAMES, MAP_RATINGS, MAP_CONJUNCTIONS, MAP_PLURALS


# Note: Multiples can only lead to "only occurrence", AB-chains & remote pairs (chain techniques) only to "single value"


# TODO Centralise this function
def convert_to_user_readable_value(instance, coords, dimension):
    """
    Converts internally used 0-based coords to user-understandable output
    """
    assert dimension in DIMENSIONS + ["cell"]
    if dimension == "row":
        idx = coords[0] + 1
    elif dimension == "col":
        idx = coords[1] + 1
    elif dimension == "box":
        # Format 1
        # idx = (coords[0] // LENGTH + 1, coords[1] // LENGTH + 1)
        # Format 2
        # idx = (coords[0] // LENGTH) * LENGTH + (coords[1] // LENGTH) + 1
        # Format 3: Allow for custom boxes layout
        idx = instance.get_idx_box(*coords) + 1
        # print("Conversion:", coords, dimension, idx)
    else:
        # idx = (coords[0] + 1, coords[1] + 1)
        # TODO This can be pre-calculated
        template_format_cell_value = (
            MAP_DIMENSIONS["row"][0].split()[-1][0].capitalize() + "{}" +
            MAP_DIMENSIONS["col"][0].split()[-1][0].capitalize() + "{}"
        )
        idx = template_format_cell_value.format(coords[0] + 1, coords[1] + 1)
    return idx


def _convert_legacy_box_idx_to_user_readable_value(idx_box):
    """
    Function to convert the old format for default boxes layout, only used for the boxed-x techniques
    """
    assert isinstance(idx_box, tuple) and len(idx_box) == 2 and all(isinstance(e, int) for e in idx_box)
    # Format 1
    # idx = (idx_box[0] + 1, idx_box[1] + 1)
    # Format 2
    idx = idx_box[0] * LENGTH + idx_box[1] + 1
    # print("Conversion for box:", idx_box, idx)
    return idx


def map_dimension(dimension, plural=False):
    assert dimension in DIMENSIONS
    return MAP_DIMENSIONS[dimension][int(plural)]


def map_direction(direction):
    assert direction in ["hor", "ver"]
    return MAP_DIRECTIONS[direction]


def join_and(iterable, add_comma=False):
    base = ',' * add_comma + ' ' + MAP_CONJUNCTIONS["and"] + ' '
    return base.join(iterable)


def join_comma_and(iterable):
    return _join_comma_keyword(iterable, MAP_CONJUNCTIONS["and"])


def join_comma_or(iterable):
    return _join_comma_keyword(iterable, MAP_CONJUNCTIONS["or"])


def _join_comma_keyword(iterable, keyword):
    l = list(map(str, iterable))
    if len(l) == 1:
        return l[0]
    elif len(l) == 2:
        return f" {keyword} ".join(l)
    else:
        return ", ".join(l[:-1]) + f", {keyword} " + l[-1]


def num_to_word(num):
    assert isinstance(num, int)
    assert num in MAP_COUNTS, f"Converting {num} to word not implemented"
    return MAP_COUNTS[num]


def format_char(char):
    return f"'{char}'"


def generate_message(step_logs):
    step_instance_before, step_instance_after, step_new_values, step_technique_used, step_is_cleanup_issue, step_cleanup_steps = step_logs

    name, difficulty = MAP_TECHNIQUE_NAMES.get(step_technique_used, step_technique_used)

    # Note: Values might be found in multiple dimensions, if this is the case use this priority
    # dimension_priority = ["row", "col", "box"]

    # TODO The name of the technique might depend on how many applications were used to find the new value
    #  -> This should be picked up with the cleanup process

    # Note: For some techniques this can be overwritten
    title = name + ' ' + f"({MAP_RATINGS[difficulty]})"
    # message = ""
    messages = []

    # message = f"({step_technique_used}) " + message

    # TODO Remove get() functionality when all techniques have been implemented
    template_message = templates_messages.get(step_technique_used)

    if step_technique_used == "singles-1":
        message = ""
        for new_value in step_new_values:
            # dimension = sorted(map(str.strip, new_value[2].split('&')), key=lambda x: dimension_priority.index(x))[0]
            coords = new_value[0]
            char = new_value[1]
            dimension = list(map(str.strip, new_value[2].split('&')))[0]
            message += \
                template_message.format(
                    # TODO This can probably be combined into one function
                    dim="{} {}".format(
                        map_dimension(dimension),
                        convert_to_user_readable_value(step_instance_before, coords, dimension),
                    ),
                    char=format_char(char),
                    cell=convert_to_user_readable_value(step_instance_before, coords, "cell"),
            )
        messages.append(message)

    elif step_technique_used == "singles-2":
        # values = []
        message = ""
        for new_value in step_new_values:
            # Generate values to be shown
            coords = new_value[0]
            char = new_value[1]
            dim_base, dim_help = new_value[2].split(' & ')[0].split(' + ')
            # values.append((coords, char, dim_base, dim_help, idx))

            # Construct message
            message += \
                template_message.format(
                    char=format_char(char),
                    dim="{} {}".format(
                        map_dimension(dim_base),
                        convert_to_user_readable_value(step_instance_before, coords, dim_base),
                    ),
                    dims=map_dimension(dim_help, plural=True),
                )

        # # TODO We could read this value from the template
        # if len(step_new_values) <= 10:
        #     separator = " / "
        # else:
        #     message += \
        #         "The other cells in these dimensions cannot contain them because their rows, cols, or boxes already contain them."

        messages.append(message)

    elif step_technique_used == "singles-3":
        # values = []
        message = ""
        for new_value in step_new_values:
            # Generate values to be shown
            coords = new_value[0]
            char = new_value[1]
            dim_base = new_value[2].split(' & ')[0]
            dim_help = [dim for dim in DIMENSIONS if dim != dim_base]
            # values.append((coords, char, dim_base, dim_help, idx))

            # Construct message
            message += \
                template_message.format(
                    char=format_char(char),
                    dim="{} {}".format(
                        map_dimension(dim_base),
                        convert_to_user_readable_value(step_instance_before, coords, dim_base),
                    ),
                    dims=join_and(
                        map_dimension(dim, plural=True)
                        for dim in dim_help
                    ),
                )

        messages.append(message)

    elif step_technique_used in ("singles-naked-2", "singles-naked-3"):
        message = ""
        for new_value in step_new_values:
            coords = new_value[0]
            char = new_value[1]
            dimensions = new_value[2].split(' & ')[0].split(' + ')

            message += \
                template_message.format(
                    dims=join_comma_and(
                        "{} {}".format(
                            map_dimension(dimension), convert_to_user_readable_value(step_instance_before, coords, dimension)
                        )
                        for dimension in dimensions
                    ),
                    char=format_char(char),
                    cell=convert_to_user_readable_value(step_instance_before, coords, "cell"),
            )
        messages.append(message)

    # Advanced techniques

    else:
        assert step_technique_used in ADVANCED_TECHNIQUES

        # Reuse same functionality across advanced techniques
        assert len(step_new_values) == 1
        new_value = step_new_values[0]

        assert len(new_value) == 4  # Generalised structure for all techniques
        coords = new_value[0]
        char = new_value[1]
        dim_base = new_value[2].split(' & ')[0]
        details = new_value[3]

        # Identify relevant applications
        # TODO Pass this as argument instead of identifying again
        print(" ~~~~~~ Identify layers in messages ")
        applications_layer_1, applications_layer_2, unused_applications = identify_layers(
            step_instance_before, step_cleanup_steps, new_value
        )

        messages, title = _generate_concatenated_message_and_title(applications_layer_1, applications_layer_2, step_instance_before)
        print("Concatenated message:")
        for message in messages:
            for line in message.replace('.', ".\n").splitlines():
                print("", line.strip())
            print()

        # TODO We either want to remove this check, or make it more general, considering all layer 1 applications
        #  -> For now disabled, can rewrite to make it more general, but this is not very necessary
        # if step_technique_used in ["doubles", "triplets", "quads"]:
        #     # Note: Multiples always lead to a final value based on only occurrence in row/col/box, as they can never
        #     #  reduce the number of options for a cell to 1
        #     assert dim_base in DIMENSIONS
        # elif step_technique_used in ["ab-chains", "remote-pairs"]:
        #     assert dim_base == "cell"
        # else:
        #     assert dim_base in DIMENSIONS + ["cell"]

        # Add the final message for the found value
        # The last part of the message depends on how the final value was found
        # TODONE We can probably create a function for this and reuse for all advanced techniques
        #  -> Actually this message depends on the technique
        if dim_base == "cell":
            # Missing value
            final_message = \
                messages_templates.template_message_final_cell.format(
                    char=format_char(char),
                    cell=convert_to_user_readable_value(step_instance_before, coords, "cell"),
                )
        else:
            # Single position
            final_message = \
                messages_templates.template_message_final_rcb.format(
                    char=format_char(char),
                    dim="{} {}".format(
                        map_dimension(dim_base),
                        convert_to_user_readable_value(step_instance_before, coords, dim_base),
                    ),
                )

        messages.append(final_message)

    # message = f"{title} : {message}"
    message = '\n\n'.join([title] + messages)

    return message


def _generate_message_and_custom_title_for_application(application, step_instance_before):

    # The title of the application depends on the application specifics for some techniques

    # Unpack tuple
    (name_technique, _), application_details, removed_chars = application

    template_message = templates_messages_advanced.get(name_technique)

    # A custom title is generated for the singles-pointing and singles-boxed techniques, for which the ray size is used
    title = None

    if name_technique in ["doubles", "triplets", "quads"]:
        multiple, idxs_multiple, _, dim_multiple = application_details

        message = \
            template_message.format(
                cells=join_comma_and(
                    map(lambda idx: convert_to_user_readable_value(step_instance_before, idx, "cell"), idxs_multiple)
                ),
                chars_and=join_comma_and(map(format_char, multiple)),
                dim="{} {}".format(
                    map_dimension(dim_multiple),
                    convert_to_user_readable_value(step_instance_before, idxs_multiple[0], dim_multiple),
                ),
                chars_or=join_comma_or(map(format_char, multiple)),
            )

    elif name_technique == "singles-pointing":
        char_pointing, idx_box, idxs_options, direction, idx_dim_pointing = application_details

        assert direction in ["hor", "ver"]
        dim_pointing = "row" if direction == "hor" else "col"
        dim_not_pointing = "col" if direction == "hor" else "row"
        ray_size = len(idxs_options)

        message = \
            template_message.format(
                dim="{} {}".format(
                    map_dimension(dim_pointing),
                    convert_to_user_readable_value(step_instance_before, idxs_options[0], dim_pointing),
                ),
                char=format_char(char_pointing),
                size=num_to_word(ray_size),
                box=idx_box + 1,
                dir=map_direction(direction),
                # Note: Here we assume that only relevant removed chars are kept
                num=num_to_word(len(removed_chars)),
                _position_s_=MAP_PLURALS["position"][int(len(removed_chars) > 1)],
                cells=join_comma_and(
                    "{} in {} {}".format(
                        convert_to_user_readable_value(step_instance_before, (idx_row, idx_col), "cell"),
                        # TODONE This seems too hardcoded? It is not incorrect but it is probably better to indicate the
                        #  direction of the ray
                        # TODO Missed this conversion, introduce auto-conversion for placeholder keywords
                        map_dimension(dim_not_pointing),
                        convert_to_user_readable_value(step_instance_before, (idx_row, idx_col), dim_not_pointing),
                    )
                    for (idx_row, idx_col), removed_char in removed_chars
                )
            )

        # With custom shaped boxes, the ray size can be up to the instance size, in case the box overlaps an entire
        #  row/col
        # TODO This check can be made more specific, but validation the application should not be part of generating
        #  messages
        assert 1 <= ray_size <= (step_instance_before.size if step_instance_before.uses_custom_boxes_layout else LENGTH)
        # The name of the technique depends on how many values are in the box
        assert ray_size in MAP_SIZE_TO_NAME
        title = MAP_TECHNIQUE_NAMES[name_technique][0].format(PLACEHOLDER=MAP_SIZE_TO_NAME[ray_size])

    # TODO See if we can merge this with singles-pointing logic
    elif name_technique == "singles-boxed":
        dim_boxed, char_boxed, idx_base, idx_box, idxs_char = application_details

        assert dim_boxed in ["row", "col"]
        ray_size = len(idxs_char)

        message = \
            template_message.format(
                dim="{} {}".format(
                    map_dimension(dim_boxed),
                    idx_base + 1,
                ),
                char=format_char(char_boxed),
                size=num_to_word(ray_size),
                box=idx_box + 1,
            )

        # With custom shaped boxes, the ray size can be up to the instance size, in case the box overlaps an entire
        #  row/col
        assert 1 <= ray_size <= (step_instance_before.size if step_instance_before.uses_custom_boxes_layout else LENGTH)
        # The name of the technique depends on how many values are in the box
        assert ray_size in MAP_SIZE_TO_NAME
        title = MAP_TECHNIQUE_NAMES[name_technique][0].format(PLACEHOLDER=MAP_SIZE_TO_NAME[ray_size])

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

        dim_help = "col" if dim_wing == "row" else "row"

        if name_technique[-1].isdigit():
            wing_size = int(name_technique.split('-')[-1])
        else:
            wing_size = 2

        # Originally the idxs are 0-based
        idxs_wing = [idx + 1 for idx in idxs_wing]
        idxs_help = [idx + 1 for idx in idxs_help]

        message = \
            template_message.format(
                dims=map_dimension(dim_wing, plural=True),
                vals=join_comma_and(idxs_wing),
                char=format_char(char_wing),
                size=num_to_word(wing_size),
                dims_help=map_dimension(dim_help, plural=True),
                vals_help=join_comma_and(idxs_help),
            )

    elif name_technique == "ab-chains":
        chain_conflict, options_for_idxs, conflicting_value = application_details

        # The chain should contain the conflicting cell as well as the head/tail of the chain
        assert len(chain_conflict) >= 3
        chain_start = chain_conflict[0]
        chain_head, chain_tail = chain_conflict[1], chain_conflict[-1]
        assert conflicting_value in options_for_idxs[chain_head]
        assert conflicting_value in options_for_idxs[chain_tail]

        message = \
            template_message.format(
                cells=join_and(
                    map((MAP_PLURALS["cell"][0] + " {}").format,
                        (
                            convert_to_user_readable_value(step_instance_before, chain_head, "cell"),
                            convert_to_user_readable_value(step_instance_before, chain_tail, "cell"),
                        )
                    )
                ),
                char=format_char(conflicting_value),
                cell=convert_to_user_readable_value(step_instance_before, chain_start, "cell"),
            )

    # TODO See if we can merge this logic with ab-chains, as the logic for chaining techniques should be very
    #  similar
    elif name_technique == "remote-pairs":
        idx_start, valid_chain, options_for_idxs = application_details

        # The chain should contain the conflicting cell as well as the head/tail of the chain
        assert len(valid_chain) >= 2
        chain_head, chain_tail = valid_chain[0], valid_chain[-1]
        pair = sorted(options_for_idxs[chain_head])
        assert len(pair) == 2
        assert sorted(options_for_idxs[chain_tail]) == pair

        message = \
            template_message.format(
                cells=join_and(
                    map((MAP_PLURALS["cell"][0] + " {}").format,
                        (
                            convert_to_user_readable_value(step_instance_before, chain_head, "cell"),
                            convert_to_user_readable_value(step_instance_before, chain_tail, "cell"),
                        )
                    )
                ),
                char_1=format_char(pair[0]),
                char_2=format_char(pair[1]),
                cell=convert_to_user_readable_value(step_instance_before, idx_start, "cell"),
            )

    elif name_technique == "y-wings":
        idxs_wings, pairs_wings, idx_center, options_center, char_wing = application_details

        # Sort values
        idxs_wings = sorted(idxs_wings)
        options_center = sorted(options_center)

        assert all(char == char_wing for idx, char in removed_chars)

        message = \
            template_message.format(
                cells=join_comma_and((
                    convert_to_user_readable_value(step_instance_before, idxs_wings[0], "cell"),
                    convert_to_user_readable_value(step_instance_before, idx_center, "cell"),
                    convert_to_user_readable_value(step_instance_before, idxs_wings[1], "cell"),
                )),
                cell=convert_to_user_readable_value(step_instance_before, idx_center, "cell"),
                chars=join_comma_or(
                    map(format_char, options_center)
                ),
                cells_wings=join_comma_or((
                    convert_to_user_readable_value(step_instance_before, idxs_wings[0], "cell"),
                    convert_to_user_readable_value(step_instance_before, idxs_wings[1], "cell"),
                )),
                char=format_char(char_wing),
                _cell_s_=MAP_PLURALS["cell"][int(len(removed_chars) > 1)],
                cells_removed=join_comma_and(
                    convert_to_user_readable_value(step_instance_before, idx, "cell")
                    for idx, char in removed_chars
                )
            )

    elif name_technique in ["doubles-naked", "triplets-naked", "quads-naked"]:
        dim_multiple, _, idxs_multiple, _, multiple = application_details

        message = \
            template_message.format(
                cells=join_comma_and(
                    map(lambda idx: convert_to_user_readable_value(step_instance_before, idx, "cell"), idxs_multiple)
                ),
                # Note that although this value is not needed in the template for doubles-naked, it will just be ignored
                size=num_to_word(len(multiple)),
                chars=join_comma_and(map(format_char, multiple)),
                dim="{} {}".format(
                    map_dimension(dim_multiple),
                    convert_to_user_readable_value(step_instance_before, idxs_multiple[0], dim_multiple),
                ),
            )

    elif name_technique in ["boxed-doubles", "boxed-triplets", "boxed-quads"]:
        (b1_target, b2_target), (i1_target, i2_target), comb_idxs, options_for_idxs, multiple, options_target_cell = application_details

        multiple_number = len(multiple)
        multiple = sorted(multiple)
        options_target_cell = sorted(options_target_cell)

        # "" if multiple_number == 2 else f"the {num_to_word(multiple_number)} ", "s" * (multiple_number > 2),

        if name_technique == "boxed-doubles":
            assert len(comb_idxs) == 1
            idx = comb_idxs[0]
            assert len(multiple) == 2

            message = \
                template_message.format(
                    cell=convert_to_user_readable_value(step_instance_before, idx, "cell"),
                    box=_convert_legacy_box_idx_to_user_readable_value((b1_target, b2_target)),
                    char_1=format_char(multiple[0]),
                    cell_target=convert_to_user_readable_value(step_instance_before, (i1_target, i2_target), "cell"),
                    char_2=format_char(multiple[1]),
                    chars_target=join_comma_or(map(format_char, options_target_cell)),
                )

        else:

            message = \
                template_message.format(
                    num=num_to_word(multiple_number - 1),
                    cells=join_comma_and(
                        map(lambda idx: convert_to_user_readable_value(step_instance_before, idx, "cell"), comb_idxs)
                    ),
                    box=_convert_legacy_box_idx_to_user_readable_value((b1_target, b2_target)),
                    chars=join_comma_or(map(format_char, multiple)),
                    cell_target=convert_to_user_readable_value(step_instance_before, (i1_target, i2_target), "cell"),
                    chars_target=join_comma_or(map(format_char, options_target_cell)),
                )

    elif name_technique == "boxed-wings":
        (b1_target, b2_target), idxs_wings, char_wing = application_details

        idx_wing_row, idx_wing_col = idxs_wings
        idx_remove_option = (idx_wing_col[0], idx_wing_row[1])

        message = \
            template_message.format(
                cell_row=convert_to_user_readable_value(step_instance_before, idx_wing_row, "cell"),
                row=idx_wing_row[0] + 1,
                cell_col=convert_to_user_readable_value(step_instance_before, idx_wing_col, "cell"),
                col=idx_wing_col[1] + 1,
                box=_convert_legacy_box_idx_to_user_readable_value((b1_target, b2_target)),
                char=format_char(char_wing),
                cell=convert_to_user_readable_value(step_instance_before, idx_remove_option, "cell"),
            )

    elif name_technique == "boxed-rays":
        (b1_target, b2_target), (i1_target, i2_target), idxs_target, (b1_ray, b2_ray), (i1_ray, i2_ray), idxs_ray, char_ray = application_details

        idxs_hor = sorted([idx for idx in idxs_ray if idx[0] == i1_ray])
        idxs_ver = sorted([idx for idx in idxs_ray if idx[1] == i2_ray])

        # TODO Exclude filled values
        idxs_target_hor = [idx for idx in idxs_target if idx[0] == i1_target]
        idxs_target_ver = [idx for idx in idxs_target if idx[1] == i2_target]

        message = \
            template_message.format(
                char=format_char(char_ray),
                box_ray=_convert_legacy_box_idx_to_user_readable_value((b1_ray, b2_ray)),
                cells_hor=' - '.join(map(str, tuple(map(lambda idx: convert_to_user_readable_value(step_instance_before, idx, "cell"), idxs_hor)))),
                cells_ver=' - '.join(map(str, tuple(map(lambda idx: convert_to_user_readable_value(step_instance_before, idx, "cell"), idxs_ver)))),
                cells_ver_target=' - '.join(map(str, tuple(map(lambda idx: convert_to_user_readable_value(step_instance_before, idx, "cell"), idxs_target_ver)))),
                box_target=_convert_legacy_box_idx_to_user_readable_value((b1_target, b2_target)),
                cells_hor_target=' - '.join(map(str, tuple(map(lambda idx: convert_to_user_readable_value(step_instance_before, idx, "cell"), idxs_target_hor)))),
                num=len(idxs_target),
                _cell_s_=MAP_PLURALS["cell"][len(idxs_target) > 1],
                cells=join_comma_and(
                    tuple(map(lambda idx: convert_to_user_readable_value(step_instance_before, idx, "cell"), idxs_target))
                ),
                _cell_s_remove_=MAP_PLURALS["cell"][len(removed_chars) > 1],
                cells_remove=join_comma_and(
                    convert_to_user_readable_value(step_instance_before, idx, "cell") for idx, char in removed_chars
                )
            )

    elif name_technique == "ab-rings":
        idxs_ring, options_for_idxs, combined_options = application_details

        template_message_part_1, template_message_part_2, = template_message

        message = \
            template_message_part_1.format(
                cells=join_comma_and(
                    map(lambda idx: convert_to_user_readable_value(step_instance_before, idx, "cell"), idxs_ring)
                ),
                chars=join_comma_and(map(format_char, sorted(combined_options))),
            )

        for i in range(4):
            idx_from, idx_to = sorted([idxs_ring[i], idxs_ring[(i + 1) % 4]])

            dim = "row" if i % 2 == 0 else "col"

            shared_chars = set.intersection(options_for_idxs[idx_from], options_for_idxs[idx_to])
            assert len(shared_chars) == 1
            shared_char = shared_chars.pop()

            message += \
                template_message_part_2.format(
                    cells=join_comma_and(
                        map(lambda idx: convert_to_user_readable_value(step_instance_before, idx, "cell"), [idx_from, idx_to])
                    ),
                    dim="{} {}".format(
                        map_dimension(dim),
                        convert_to_user_readable_value(step_instance_before, idx_from, dim),
                    ),
                    char=format_char(shared_char),
                )

    elif name_technique in [f"leftovers-{i}" for i in range(1, 9 + 1)]:
        dim, region, box_idxs_inside, box_idxs_outside, idxs_cells_inside, idxs_cells_outside, num_cells_inside, options_inside, options_outside, options = application_details

        assert len(idxs_cells_inside) == len(idxs_cells_outside) == num_cells_inside
        assert num_cells_inside >= 1

        chars = step_instance_before.chars

        template_message_part_1, template_message_part_2, template_message_part_3, = template_message

        message = \
            template_message_part_1.format(
                dims=join_comma_and(
                    "{} {}".format(
                        map_dimension(dim),
                        # TODO Reuse centralised function to convert dimensions
                        idx_dim + 1
                    )
                    for idx_dim in region
                ),
                num=num_cells_inside,
                _in_s_=MAP_PLURALS["in"][num_cells_inside > 1],
                cells_in=", ".join((convert_to_user_readable_value(step_instance_before, idx, "cell") for idx in sorted(idxs_cells_inside))),
                _out_s_=MAP_PLURALS["out"][num_cells_inside > 1],
                cells_out=", ".join((convert_to_user_readable_value(step_instance_before, idx, "cell") for idx in sorted(idxs_cells_outside))),
                _contain_s_=MAP_PLURALS["contain"][num_cells_inside > 1],
                _candidate_s_in_=MAP_PLURALS["candidate"][int(len(options_inside) > 1)],
                chars_in=join_comma_and(map(format_char, sorted(options_inside))),
                _candidate_s_out_=MAP_PLURALS["candidate"][int(len(options_outside) > 1)],
                chars_out=join_comma_and(map(format_char, sorted(options_outside))),
            )

        # Reconstruct the actual options that could be removed by the application, as also done in the application
        #  itself
        chars_outside_but_not_inside = options_outside.difference(options_inside)
        chars_inside_but_not_outside = options_inside.difference(options_outside)

        # Pre-construct and only show if any options were removed
        _message_remove_options = join_and(
            (
                [
                    template_message_part_3.format(
                        chars=join_comma_and(map(format_char, sorted(chars_outside_but_not_inside))),
                        num=num_cells_inside,
                        _in_s_out_s_=MAP_PLURALS["out"][num_cells_inside > 1],
                    )
                ] if chars_outside_but_not_inside else []
            ) + (
                [
                    template_message_part_3.format(
                        chars=join_comma_and(map(format_char, sorted(chars_inside_but_not_outside))),
                        num=num_cells_inside,
                        _in_s_out_s_=MAP_PLURALS["in"][num_cells_inside > 1],
                    )
                ] if chars_inside_but_not_outside else []
            ),
            add_comma=True
        )

        message += \
            template_message_part_2.format(_message_remove_options)

        # Update name of technique based on number of cells inside
        assert num_cells_inside in range(1, 9 + 1)
        placeholder_value = f'({num_cells_inside} {MAP_PLURALS["in"][num_cells_inside > 1]}, {num_cells_inside} {MAP_PLURALS["out"][num_cells_inside > 1]})'
        title = MAP_TECHNIQUE_NAMES[name_technique][0].format(PLACEHOLDER=placeholder_value)

    else:
        # message = f"Technique used: {name_technique}"
        raise Exception(f"Undefined title/message template for technique: {name_technique}")

    return message, title


def _generate_concatenated_message_and_title(applications_layer_1, applications_layer_2, step_instance_before):

    message = ""
    # title = ""

    titles = []
    titles_order = {}
    difficulties = []

    # Note: This recursive structure is the same as in writer
    def generate_message_recursively(earlier_applications):
        nonlocal message
        for application in earlier_applications:
            name_technique, name_application = application[0]
            print(f"Generate message for layer 2 application: {name_application}")
            _message, _title_custom = _generate_message_and_custom_title_for_application(application, step_instance_before)
            message = _message + message
            _title, _difficulty = MAP_TECHNIQUE_NAMES.get(name_technique, name_technique)
            _title_order = TECHNIQUES.index(name_technique)
            # TODO De-duplicate this logic - a modification to one should be a modification to both!
            if _title_custom is not None:
                _title = _title_custom
                if name_technique not in [f"leftovers-{i}" for i in range(1, 9 + 1)]:
                    _key = _title_custom.split(' ')[-1]
                    # Note: We assume that all custom titles conform to the same naming format
                    assert _key in MAP_SIZE_TO_NAME.values()
                    # Note: We have to make sure that the value added with the custom part is smaller than 1, as it has
                    #  to remain its own priority and not mix with other techniques
                    assert max(MAP_SIZE_TO_NAME.keys()) <= len(MAP_SIZE_TO_NAME)
                    custom_title_order = {v: k for k, v in MAP_SIZE_TO_NAME.items()}[_key] / (len(MAP_SIZE_TO_NAME) + 1)
                    _title_order += custom_title_order
            # title = _title + ', ' + title
            titles.append(_title)
            titles_order[_title] = _title_order
            difficulties.append(_difficulty)
            generate_message_recursively(applications_layer_2[name_application])

    messages = []
    for application in applications_layer_1:
        name_technique, name_application = application[0]
        # message_application_layer_1 = ""
        # Generate messages recursively and prepend every time
        message = ""
        # title = ""

        print(f"Generate message for layer 1 application: {name_application}")
        _message, _title_custom = _generate_message_and_custom_title_for_application(application, step_instance_before)
        message = _message
        _title, _difficulty = MAP_TECHNIQUE_NAMES.get(name_technique, name_technique)
        _title_order = TECHNIQUES.index(name_technique)
        if _title_custom is not None:
            _title = _title_custom
            if name_technique not in [f"leftovers-{i}" for i in range(1, 9 + 1)]:
                _key = _title_custom.split(' ')[-1]
                # Note: We assume that all custom titles conform to the same naming format
                assert _key in MAP_SIZE_TO_NAME.values()
                # Note: We have to make sure that the value added with the custom part is smaller than 1, as it has
                #  to remain its own priority and not mix with other techniques
                assert max(MAP_SIZE_TO_NAME.keys()) <= len(MAP_SIZE_TO_NAME)
                custom_title_order = {v: k for k, v in MAP_SIZE_TO_NAME.items()}[_key] / (len(MAP_SIZE_TO_NAME) + 1)
                _title_order += custom_title_order
        # title = _title
        titles.append(_title)
        titles_order[_title] = _title_order
        difficulties.append(_difficulty)

        generate_message_recursively(applications_layer_2[name_application])
        messages.append(message)
        # titles.append(title)

    ORDER_DIFFICULTY = ["Easy", "Medium", "Hard"]
    assert not set(difficulties).difference(ORDER_DIFFICULTY)
    difficulty = sorted(difficulties, key=lambda item: ORDER_DIFFICULTY.index(item))[-1]

    # message = ''.join(messages)  # Don't merge messages yet, leave it to the outer function how to concatenate
    # title = ', '.join(titles)

    # Post-process title: Remove duplicates and sort by difficulty
    titles_without_duplicates = set(titles)
    titles_sorted = sorted(titles_without_duplicates, key=lambda key: titles_order[key])

    # print("All titles:", titles)
    # print("Titles without duplicates:", titles_without_duplicates)
    # print("Titles sorted:", titles_sorted)

    title = ', '.join(titles_sorted) + ' ' + f"({MAP_RATINGS[difficulty]})"

    return messages, title
