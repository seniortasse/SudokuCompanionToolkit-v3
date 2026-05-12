
import string

from generator.algo_human import TECHNIQUES, BASE_TECHNIQUES, ADVANCED_TECHNIQUES


template_header_main = "{instance_id} STEP-BY-STEP SOLUTION"

template_header_step = "STEP {step_no}"


template_message_singles_1 = (
    "Looking at {dim}, {char} is the only missing value at {cell}. "
)

template_message_singles_2_3 = (
    "There is a single position for {char} in {dim}. "
    "The other cells in {dim} cannot contain {char} because their {dims} already contain {char}. "
)

template_message_singles_naked_2_3 = (
    "Looking at {dims}, {char} is the only missing value at {cell}. "
)

# TODO To be expanded with all techniques
# TODO Aggregating base and advanced would be nice
templates_messages = {
    "singles-1": template_message_singles_1,
    "singles-2": template_message_singles_2_3,
    "singles-3": template_message_singles_2_3,
    "singles-naked-2": template_message_singles_naked_2_3,
    "singles-naked-3": template_message_singles_naked_2_3,
}
assert not set(BASE_TECHNIQUES).symmetric_difference(templates_messages.keys())


template_message_multiples = (
    "Cells {cells} are the only positions for the candidates {chars_and} in {dim} because the other positions cannot be {chars_or}. "
)

template_message_singles_pointing = (
    "Looking at {dim}, {char} is in one of {size} cells in box {box} and points {dir}, "
    "thus removing {num} {_position_s_} for {char} at {cells}. "
)

template_message_singles_boxed = (
    "Looking at {dim}, {char} is in one of {size} cells in box {box} because the other cells of {dim} cannot contain {char}. "
)

template_message_x_wings = (
    "Looking at {dims} {vals}, the value {char} is restricted exactly to {size} {dims_help} {vals_help}. "
    "Therefore each of those {size} {dims_help} contains a {char} that is restricted only to {dims} {vals}. "
    "Therefore {char} can be removed from the other cells of {dims_help} {vals_help}. "
)

template_message_ab_chains = (
    "Looking at {cells}, if the value in one of the two cells is not {char} then the value in the other has to be {char}. "
    "Therefore, cell {cell} cannot contain {char}. "
)

template_message_remote_pairs = (
    "Looking at {cells}, "
    "if the value in one of the two cells is not {char_1} then the value in the other cell has to be {char_1}, and "
    "if the value in one of the two cells is not {char_2} then the value in the other cell has to be {char_2}. "
    "Therefore, cell {cell} cannot contain {char_1} and cannot contain {char_2}. "
)

template_message_y_wings = (
    "Looking at cells {cells}, whether the value in cell {cell} is {chars}, "
    "the value in one of the two other cells {cells_wings} has to be {char}, "
    "thus {char} can be removed from {_cell_s_} {cells_removed}. "
)

template_message_multiples_naked_2 = (
    "Cells {cells} contain exactly the same two candidates {chars} in {dim}. "
)

template_message_multiples_naked_3_4 = (
    "Cells {cells} collectively contain exactly {size} candidates {chars} in {dim}. "
)

# TODO We can reduce duplicate parts, but the template will become harder to interpret - see the template for
#  "leftovers", where the message length is dynamic, based on whether options were actually removed by the applications
template_message_boxed_multiples_2 = (
    "Looking at the interaction between cell {cell} and box {box}, "
    "if the value in cell {cell} is {char_1}, then the value in cell {cell_target} has to be {char_1}, and "
    "if the value in cell {cell} is {char_2}, then the value in cell {cell_target} has to be {char_2}. "
    "Therefore cell {cell_target} can only contain the values {chars_target}. "
)

template_message_boxed_multiples_3_4 = (
    "Looking at the interaction between the {num} cells {cells} and box {box}, "
    "if cells {cells} are either {chars}, then cell {cell_target} has to be {chars_target}. "
    "Therefore cell {cell_target} can only contain the values {chars_target}. "
)

template_message_boxed_wings = (
    "Looking at cell {cell_row} in row {row} and cell {cell_col} in col {col}, and their interaction with box {box}, "
    "if the value in cell {cell_row} is not {char}, then the value in cell {cell_col} has to be {char}, and "
    "if the value in cell {cell_col} is not {char}, then the value in cell {cell_row} has to be {char}. "
    "Therefore {char} can be removed from cell {cell}. "
)

template_message_boxed_rays = (
    "The value {char} in box {box_ray} is contained either in the horizontal group of cells {cells_hor} or in the vertical group of cells {cells_ver}. "
    "If {char} is contained in the horizontal group of cells {cells_hor}, then {char} must also be contained in the vertical group of cells {cells_ver_target} in box {box_target}. "
    "But if {char} is instead contained in the vertical group of cells {cells_ver}, then {char} must also be contained in the horizontal group of cells {cells_hor_target} in box {box_target}. "
    "Either way, {char} can only be in 1 of {num} {_cell_s_} {cells} in box {box_target}. "
    "Therefore, {char} can be removed from {_cell_s_remove_} {cells_remove}. "
)

template_message_ab_rings = (
    (
        "Cells {cells} contain exactly two values each and collectively exactly four candidates {chars}. ",
        "If the value in one of the two cells {cells} in {dim} is not {char}, then the value in the other cell has to be {char}. "
        "Therefore {char} can be removed from all other cells in {dim}. "
    )
)

template_message_leftovers = (
    (
        "When we look at the houses consisting of {dims}, we can identify {num} {_in_s_} {cells_in} and {num} {_out_s_} {cells_out}. "
        "The {num} {_in_s_} {_contain_s_} {_candidate_s_in_} {chars_in}, and the {num} {_out_s_} {_contain_s_} {_candidate_s_out_} {chars_out}. ",
        "So {}. ",  # Keep this placeholder empty
        "{chars} can be removed from the {num} {_in_s_out_s_}"  # This line is repeated twice and filled into the part above, do not close with a dot (.)
    )
)

# TODO Somewhere group techniques and reuse here to add from keys
# Note: We first define this collection to be able to generate templates more easily
_templates_messages_advanced = {
    ("doubles", "triplets", "quads"): template_message_multiples,
    "singles-pointing": template_message_singles_pointing,
    "singles-boxed": template_message_singles_boxed,
    ("x-wings", "x-wings-3", "x-wings-4"): template_message_x_wings,
    "ab-chains": template_message_ab_chains,
    "remote-pairs": template_message_remote_pairs,
    "y-wings": template_message_y_wings,
    "doubles-naked": template_message_multiples_naked_2,
    ("triplets-naked", "quads-naked"): template_message_multiples_naked_3_4,
    "boxed-doubles": template_message_boxed_multiples_2,
    ("boxed-triplets", "boxed-quads"): template_message_boxed_multiples_3_4,
    "boxed-wings": template_message_boxed_wings,
    "boxed-rays": template_message_boxed_rays,
    "ab-rings": template_message_ab_rings,
    tuple(f"leftovers-{i}" for i in range(1, 9 + 1)): template_message_leftovers,
}

# TODO I'd like a multi-key dict, where keys can be multiple of keys, and getters find keys in tuples
templates_messages_advanced = {
    key: value
    for _key, value in _templates_messages_advanced.items()
    for key in ((_key, ) if isinstance(_key, str) else _key)
}
assert not set(ADVANCED_TECHNIQUES).symmetric_difference(templates_messages_advanced.keys())


template_message_final_cell = (
    "Therefore {char} remains the only candidate left at {cell}. "
)

template_message_final_rcb = (
    "Therefore, there is a single position left for {char} in {dim}. "
)


# placeholders to add here?


# Mapping from technical used in the code to user-understandable names shown in the message
MAP_TECHNIQUE_NAMES = {

    "singles-1": ("Open Single", "Easy"),
    "singles-2": ("Hidden Single, 2 directions", "Easy"),
    "singles-3": ("Hidden Single, 3 directions", "Medium"),
    "singles-naked-2": ("Naked Single, 2 directions", "Medium"),
    "singles-naked-3": ("Naked Single, 3 directions", "Medium"),

    "doubles": ("Hidden Pair", "Hard"),
    "triplets": ("Hidden Triple", "Hard"),
    "quads": ("Hidden Quad", "Hard"),

    "singles-pointing": ("Pointing {PLACEHOLDER}", "Hard"),  # Note: This title customised based on the specifics of the application
    "singles-boxed": ("Claiming {PLACEHOLDER}", "Hard"),  # Note: This title customised based on the specifics of the application

    "x-wings": ("X-Wing", "Hard"),
    "x-wings-3": ("Swordfish", "Hard"),
    "x-wings-4": ("Jellyfish", "Hard"),

    "ab-chains": ("XY-Chain", "Hard"),
    "remote-pairs": ("Remote Pairs", "Hard"),

    "y-wings": ("Y-Wing", "Hard"),

    "doubles-naked": ("Naked Pair", "Medium"),
    "triplets-naked": ("Naked Triple", "Medium"),
    "quads-naked": ("Naked Quad", "Medium"),

    "boxed-doubles": ("Mirroring Pairs", "Hard"),
    "boxed-triplets": ("Mirroring Triples", "Hard"),
    "boxed-quads": ("Mirroring Quads", "Hard"),

    "boxed-wings": ("Box Pivot and Pincers", "Hard"),
    "boxed-rays": ("Multidirectional Claiming", "Hard"),
    "ab-rings": ("XY-Ring", "Hard"),

    # Add dynamically
    **{
        f"leftovers-{i}": ("Leftovers {PLACEHOLDER}", "Hard")
        for i in range(1, 9 + 1)
    }
}
assert not set(TECHNIQUES).symmetric_difference(MAP_TECHNIQUE_NAMES.keys())


MAP_RATINGS = {
    rating: rating
    for rating in ["Easy", "Medium", "Hard"]
}


MAP_DIMENSIONS = {
    "row": tuple("row" + "s" * i for i in range(2)),
    "col": tuple("column" + "s" * i for i in range(2)),
    "box": tuple("box" + "es" * i for i in range(2)),
}

MAP_DIRECTIONS = {
    "hor": "horizontally",
    "ver": "vertically",
}

MAP_COUNTS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine",
}

# Used for singles-pointing and singles-boxed custom technique names
MAP_SIZE_TO_NAME = {
    1: "Single",
    2: "Pair",
    3: "Triple",
    4: "Quadruple",
    5: "Quintuple",
    6: "Sextuple",
    7: "Septuple",
    8: "Octuple",
    9: "Nonuple",
}

MAP_CONJUNCTIONS = {
    key: key
    for key in ("and", "or")
}

MAP_PLURALS = {
    "cell": tuple("cell" + "s" * i for i in range(2)),
    "position": tuple("position" + "s" * i for i in range(2)),
    "candidate": tuple("candidate" + "s" * i for i in range(2)),
    # Used for leftovers technique
    **{
        name: tuple(
            name + name[-1] * (name == "in") + ''.join(string.ascii_letters[i] for i in (8, 8 - string.ascii_letters.index('e')) + is_plural * (8 + 5 * 2, ))
            for is_plural in (False, True)
        )
        for name in ("in", "out")
    }
}
