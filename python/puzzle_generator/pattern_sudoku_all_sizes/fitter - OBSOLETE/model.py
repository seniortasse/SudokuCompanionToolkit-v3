
import pulp

from fitter.positioner import fits_diagonally, subword_overlaps_idx


def create_base_vars_and_constrs(dims):

    box_height, box_width, size = dims

    vars = {
        (n, i1, i2): pulp.LpVariable(f"x_{n}_{i1}_{i2}", cat="Binary")
        for i1 in range(size)
        for i2 in range(size)
        for n in range(size)
    }

    constrs_hor = {
        (n, i1): pulp.lpSum(
            vars.get((n, i1, i2))
            for i2 in range(size)
        ) == 1
        for n in range(size)
        for i1 in range(size)
    }

    constrs_ver = {
        (n, i2): pulp.lpSum(
            vars.get((n, i1, i2))
            for i1 in range(size)
        ) == 1
        for n in range(size)
        for i2 in range(size)
    }

    constrs_box = {
        (n, b1, b2): pulp.lpSum(
            vars.get((n, box_height * b1 + i1, box_width * b2 + i2))
            for i1 in range(box_height)
            for i2 in range(box_width)
        ) == 1
        for n in range(size)
        for b1 in range(size // box_height)
        for b2 in range(size // box_width)
    }

    constrs_cell = {
        (i1, i2): pulp.lpSum(
            vars.get((n, i1, i2))
            for n in range(size)
        ) == 1
        for i1 in range(size)
        for i2 in range(size)
    }

    return vars, (constrs_hor, constrs_ver, constrs_box, constrs_cell)


def create_vars_and_constrs_hor_subwords(dims, subwords, map_char_to_number, base_vars, is_lr):

    _, _, size = dims

    name = "lr" if is_lr else "rl"
    vars_subwords_hor = {
        (idx, i1, i2): pulp.LpVariable(f"sw-hor-{name}_{idx}_{i1}_{i2}", cat="Binary")
        for idx, subword in enumerate(subwords) if subword is not None
        for i1 in range(size)
        for i2 in (range(size - len(subword) + 1) if is_lr else range(len(subword) - 1, size))
        # Exclude subwords with duplicate characters
        if len(subword) == len(set(subword))
    }

    constrs_subwords_hor = {
        (idx, i1, i2): var_subword <= pulp.lpSum(
            base_vars[(map_char_to_number[subwords[idx][i]], i1, (i2 + i) if is_lr else (i2 - i))]
            for i in range(len(subwords[idx]))
        ) / len(subwords[idx])
        for (idx, i1, i2), var_subword in vars_subwords_hor.items()
    }

    return vars_subwords_hor, constrs_subwords_hor


def create_vars_and_constrs_ver_subwords(dims, subwords, map_char_to_number, base_vars, is_ud):

    _, _, size = dims

    name = "ud" if is_ud else "du"
    vars_subwords_ver = {
        (idx, i1, i2): pulp.LpVariable(f"sw-ver-{name}_{idx}_{i1}_{i2}", cat="Binary")
        for idx, subword in enumerate(subwords) if subword is not None
        for i1 in (range(size - len(subword) + 1) if is_ud else range(len(subword) - 1, size))
        for i2 in range(size)
        # Exclude subwords with duplicate characters
        if len(subword) == len(set(subword))
    }

    constrs_subwords_ver = {
        (idx, i1, i2): var_subword <= pulp.lpSum(
            base_vars[(map_char_to_number[subwords[idx][i]], (i1 + i) if is_ud else (i1 - i), i2)]
            for i in range(len(subwords[idx]))
        ) / len(subwords[idx])
        for (idx, i1, i2), var_subword in vars_subwords_ver.items()
    }

    return vars_subwords_ver, constrs_subwords_ver


def create_vars_and_constrs_diag_subwords(dims, subwords, map_char_to_number, base_vars, is_top_down):

    box_height, box_width, size = dims

    name = "lrd" if is_top_down else "lru"
    vars_subwords_diag = {
        (idx, i1, i2): pulp.LpVariable(f"sw-diag-{name}_{idx}_{i1}_{i2}", cat="Binary")
        for idx, subword in enumerate(subwords) if subword is not None
        for i1 in (range(size - len(subword) + 1) if is_top_down else range(len(subword) - 1, size))
        for i2 in range(size - len(subword) + 1)
        # Exclude subwords which do not satisfy the box constraint
        if fits_diagonally(dims, subword, i1, i2, is_lrd=is_top_down)
    }

    constrs_subwords_diag = {
        (idx, i1, i2): var_subword <= pulp.lpSum(
            base_vars[(map_char_to_number[subwords[idx][i]], (i1 + i) if is_top_down else (i1 - i), i2 + i)]
            for i in range(len(subwords[idx]))
        ) / len(subwords[idx])
        for (idx, i1, i2), var_subword in vars_subwords_diag.items()
    }

    return vars_subwords_diag, constrs_subwords_diag


def create_vars_and_constrs_subwords_present(subwords, vars_subwords):

    vars_subwords_present = {
        idx: pulp.LpVariable(f"sw_present_{idx}", cat="Binary")
        for idx in range(len(subwords))
    }

    constrs_subwords_present = {
        idx: (
            vars_subwords_present[idx] <=
            pulp.lpSum(
                var for t, var in vars_subwords.items()
                if t[0] == idx
            )
        )
        for idx in range(len(subwords))
    }

    return vars_subwords_present, constrs_subwords_present


def create_overlap_constraints(size, subwords, vars_subwords_for_orientations):

    # Category 1: No overlap allowed between subwords of the same orientation
    constrs_subwords_overlap_same_orientation = {
        (i1, i2, orientation): pulp.lpSum(
            var
            for (idx, placement_i1, placement_i2), var in vars_subwords.items()
            if subword_overlaps_idx(subwords[idx], placement_i1, placement_i2, orientation, i1, i2)
        ) <= 1
        for i1 in range(size)
        for i2 in range(size)
        for orientation, vars_subwords in vars_subwords_for_orientations.items()
    }

    # Category 2: Cost for overlap of subwords of different orientations
    vars_subwords_overlap = {
        (i1, i2): pulp.LpVariable(f"overlap-{i1}_{i2}", lowBound=0, cat="Integer")
        for i1 in range(size)
        for i2 in range(size)
    }

    constrs_subwords_overlap_diff_orientation = {
        (i1, i2): vars_subwords_overlap[(i1, i2)] >= pulp.lpSum(
            var
            for orientation, vars_subwords in vars_subwords_for_orientations.items()
            for (idx, placement_i1, placement_i2), var in vars_subwords.items()
            if subword_overlaps_idx(subwords[idx], placement_i1, placement_i2, orientation, i1, i2)
        ) - 1
        for i1 in range(size)
        for i2 in range(size)
    }

    return vars_subwords_overlap, constrs_subwords_overlap_same_orientation, constrs_subwords_overlap_diff_orientation
