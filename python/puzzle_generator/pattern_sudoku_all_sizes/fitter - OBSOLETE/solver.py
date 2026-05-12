
from functools import partial
import itertools
import pulp
import random

from fitter.model import create_base_vars_and_constrs, create_vars_and_constrs_subwords_present, \
    create_vars_and_constrs_hor_subwords, create_vars_and_constrs_ver_subwords, create_vars_and_constrs_diag_subwords, \
    create_overlap_constraints
from fitter.verification import verify_base_constraints
from fitter.formatter import determine_highlights, pretty_print


OPTION_FIX_DIAGONAL = "fix-mainword"


def solve(dims, mainword, subwords, options={}, pattern=None, max_number_minutes=None, include_orientations=None):

    chars = set(mainword)
    box_height, box_width, size = dims
    assert len(chars) == size

    map_number_to_char = {i: e for i, e in enumerate(chars)}
    map_char_to_number = {v: k for k, v in map_number_to_char.items()}

    print()
    print("Create solver..")
    problem = pulp.LpProblem("words", sense=pulp.LpMaximize)

    print("Add vars and constrs..")
    base_vars, (constrs_hor, constrs_ver, constrs_box, constrs_cell) = create_base_vars_and_constrs(dims)
    assert len(constrs_hor) == size ** 2
    assert len(constrs_ver) == size ** 2
    assert len(constrs_box) == size ** 2
    assert len(constrs_cell) == size ** 2

    # We do not filter redundant vars/constr when creating the model, which should not be necessary anyways when using
    #  a good solver package, as they should be quickly recognised as being redundant; Besides, now that we include the
    #  option to not fix the mainword, we need all vars, so this is also the easier and less error-prone implementation

    is_fixed_diagonal = options.get(OPTION_FIX_DIAGONAL, False)
    if is_fixed_diagonal:
        # Fix main word (only if specified by the input option)
        for i in range(size):
            var = base_vars[(map_char_to_number[mainword[i]], i, i)]
            var.setInitialValue(True)
            var.fixValue()

    all_constrs = list(itertools.chain(*map(lambda d: d.values(),  [
        constrs_hor,
        constrs_ver,
        constrs_box,
        constrs_cell
    ])))

    print("Number base vars:", len(base_vars))
    print("Number base constraints:", len(all_constrs))

    sorted_orientations = ["hor-lr", "hor-rl", "ver-ud", "ver-du", "diag-lrd", "diag-lru"]

    if include_orientations is not None:
        sorted_orientations = [orientation for orientation in sorted_orientations if orientation in include_orientations]

    print("Using orientations:", sorted_orientations)

    fncs = {
        "hor-lr": partial(create_vars_and_constrs_hor_subwords, is_lr=True),
        "hor-rl": partial(create_vars_and_constrs_hor_subwords, is_lr=False),
        "ver-ud": partial(create_vars_and_constrs_ver_subwords, is_ud=True),
        "ver-du": partial(create_vars_and_constrs_ver_subwords, is_ud=False),
        "diag-lrd": partial(create_vars_and_constrs_diag_subwords, is_top_down=True),
        "diag-lru": partial(create_vars_and_constrs_diag_subwords, is_top_down=False),
    }

    vars_subwords_for_orientations = {}
    for orientation in sorted_orientations:
        fnc = fncs[orientation]
        # Only include subwords with one character in the first orientation
        if sorted_orientations.index(orientation) == 0:
            subwords_for_orientation = subwords
        else:
            subwords_for_orientation = [subword if len(subword) > 1 else None for subword in subwords]
        print(f"Subwords included for {orientation} locations:", subwords_for_orientation)
        vars_subwords_orientation, constrs_subwords_orientation = fnc(dims, subwords_for_orientation, map_char_to_number, base_vars)
        vars_subwords_for_orientations[orientation] = vars_subwords_orientation
        all_constrs += list(constrs_subwords_orientation.values())
        # print(" -> %s vars, %s constrs" % (len(vars_subwords_orientation), len(constrs_subwords_orientation)))
        print(" Adding %s vars, %s constrs" % (len(vars_subwords_orientation), len(constrs_subwords_orientation)))

    # 2: Single endpoint
    # Aggregate all subword variables for each rotation
    #  -> We assume that all keys are of the form (idx, i1, i2)
    print(" Subwords (all) present")
    vars_subwords_all = {}
    for orientation, vars_subwords in vars_subwords_for_orientations.items():
        vars_subwords_all.update({
            (idx, i1, i2, orientation): var for (idx, i1, i2), var in vars_subwords.items()
        })
    vars_subwords_present, constrs_subwords_present = create_vars_and_constrs_subwords_present(subwords, vars_subwords_all)
    all_constrs += list(constrs_subwords_present.values())
    print(" -> %s vars, %s constrs" % (len(vars_subwords_present), len(constrs_subwords_present)))

    # Overlap constraints
    print(" Subword overlap constrs..")
    vars_subwords_overlap, constrs_subwords_overlap_same_orientation, constrs_subwords_overlap_diff_orientation = \
        create_overlap_constraints(size, subwords, vars_subwords_for_orientations)
    all_constrs += list(constrs_subwords_overlap_same_orientation.values())
    all_constrs += list(constrs_subwords_overlap_diff_orientation.values())

    if pattern is not None:
        # Extra constraint related to patterns:
        #  Make sure the cells to be kept (black squares) contains each character at least once;
        #  Rephrased, each character should be present in at least one of the black squares
        #  -> This works indeed, at only a very slight runtime cost, while it saves generating solutions for which 50%
        #     has to be discarded without this constraint
        constrs_pattern = {
            i: pulp.lpSum(
                base_vars[(i, i1, i2)]
                for i1 in range(size)
                for i2 in range(size)
                if pattern[i1][i2]
            ) >= 1
            for i in range(size)
        }
        all_constrs += list(constrs_pattern.values())

    for constr in all_constrs:
        problem.addConstraint(constr)

    print("Add obj..")
    obj = pulp.lpSum(len(subwords[idx]) * 10_000 * var for idx, var in vars_subwords_present.items())

    # Steer the subwords towards a certain (random) location
    #  -> Do not use positive values, as it will try to fit in as many occurrences as possible
    obj -= pulp.lpSum(random.random() * var for _, var in vars_subwords_all.items())

    # Penalise less desirable rotations
    map_values = {"hor-lr": 0, "hor-rl": 1, "ver-ud": 2, "ver-du": 3, "diag-lrd": 4, "diag-lru": 5}
    if len(vars_subwords_for_orientations) > 1:
        min_val = min([map_values[orientation] for orientation in vars_subwords_for_orientations.keys()])
        for orientation, vars_subwords in vars_subwords_for_orientations.items():
            val = 100 * (map_values[orientation] - min_val)
            obj -= pulp.lpSum(val * var for _, var in vars_subwords.items())

    # Penalise overlaps
    obj -= pulp.lpSum(10 * var for _, var in vars_subwords_overlap.items())

    problem += obj

    print("Number variables: %s" % problem.numVariables())
    print("Number constraints: %s" % problem.numConstraints())

    print("Solve problem..")
    if max_number_minutes is not None:
        solver = pulp.PULP_CBC_CMD(timeLimit=60 * max_number_minutes)
    else:
        solver = pulp.PULP_CBC_CMD()

    status = problem.solve(solver)

    print(pulp.LpStatus[status])
    print(pulp.value(obj))

    # Verify solution
    print()
    print("Verify solution..")
    try:
        assert all(var.value() in [0, 1] for var in base_vars.values())
        assert sum(var.value() for var in base_vars.values()) == size ** 2
    except AssertionError as e:
        print("Solution found is not valid, please investigate..")
        raise e

    # Reconstruct solution
    print()
    print("Reconstruct solution..")
    solution, subwords_placements = reconstruct_solution(size, subwords, map_number_to_char, base_vars, vars_subwords_all)

    # Verify solution
    verify_base_constraints(solution, dims, chars)

    # Identify which subwords could not be included
    subwords_included = [subword for (subword, _, _, _) in subwords_placements]
    subwords_not_included = [subword for subword in subwords if subword not in subwords_included]

    return solution, subwords_placements, subwords_not_included


def reconstruct_solution(size, subwords, map_number_to_char, base_vars, vars_subwords_all):

    solution = [[None] * size for _ in range(size)]

    for (n, i1, i2), var in base_vars.items():
        if var.value() == 1:
            solution[i1][i2] = map_number_to_char[n]

    subwords_placements = []

    for (idx, i1, i2, orientation), var in vars_subwords_all.items():
        if var.value() == 1:
            subwords_placements.append((subwords[idx], orientation, i1, i2))

    return solution, subwords_placements


# TODO To be moved to output/main file
def log_solution(solution, subwords_placements, subwords_not_included, is_fixed_diagonal):

    # Format solution
    highlights = determine_highlights(solution.size, subwords_placements)

    print()
    print("Solution:")
    pretty_print(solution, highlights, is_fixed_diagonal)

    for subword, orientation, i1, i2 in subwords_placements:
        print(f"Subword {subword} placed {orientation} at {(i1, i2)}")

    # Show warning for words that could not be included
    for subword in subwords_not_included:
        print(f"WARNING - Subword \"{subword}\" was not included in the solution!")
    print()
