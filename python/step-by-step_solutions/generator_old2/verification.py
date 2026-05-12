
import itertools
import logging
import pulp

from generator.model import EMPTY_CHAR, Instance, count_non_empty_cells, DUMMY_CHAR
from generator.solver import create_base_vars_and_constrs
from generator.algo_human import solve_using_human_techniques, TECHNIQUES


logging.basicConfig(level=logging.INFO)


def verify_has_unique_solution(instance, target_solution, preprocess=False):
    """
    Verify whether the given instance can only result in the target solution

    params:
      preprocess - whether to do an initial search with the human techniques to speed up the search (this can be toggled
       of for verification purposes)
    """

    # Speed up to verify unique solution: Search some values with human techniques
    if preprocess:
        prefilled_instance, _ = solve_using_human_techniques(instance, use_techniques=TECHNIQUES, include_magic_technique=False)
        if prefilled_instance == target_solution:
            return True
        else:
            # Check that the prefilling went correctly
            size = instance.size
            assert not any(prefilled_instance[i1][i2] != EMPTY_CHAR and prefilled_instance[i1][i2] != target_solution[i1][i2] for i1 in range(size) for i2 in range(size))
    else:
        prefilled_instance = instance

    instance = prefilled_instance
    target_solution = target_solution

    status = _solve(instance, exclude_solution=target_solution)

    solution_is_unique = not status == 1

    return solution_is_unique


def _solve(instance, exclude_solution):
    """
    Steps
     - Fill in known values
     - Solve with basic constraints
    """

    problem, base_vars, _ = _create_base_model(instance, exclude_solution)

    # # Fix instance values
    # for i1, row in enumerate(instance):
    #     for i2, e in enumerate(row):
    #         if e != EMPTY_CHAR:
    #             var = base_vars[(e, i1, i2)]
    #             var.setInitialValue(True)
    #             var.fixValue()
    #             logging.debug(f"Fix {(i1, i2)} to {e}")

    logging.debug("Solve problem..")
    solver = pulp.PULP_CBC_CMD(msg=False)
    status = problem.solve(solver)

    logging.debug(pulp.LpStatus[status])

    return status


# TODO It seems we do not have a function yet which simply solves an instance and returns the solution; Remove this one
#  later
def solve(instance):

    problem, base_vars, map_number_to_char = _create_base_model(instance)

    logging.debug("Solve problem..")
    solver = pulp.PULP_CBC_CMD(msg=False)
    status = problem.solve(solver)

    size = instance.size

    solution = [[None] * size for _ in range(size)]

    for (n, i1, i2), var in base_vars.items():
        if var.value() == 1:
            solution[i1][i2] = map_number_to_char[n]

    # In case the layout has to be specified, copy this from the instance
    solution = Instance(solution, layout=instance.layout)

    return status, solution


def _create_base_model(instance, exclude_solution=None):

    size = instance.size

    if exclude_solution is not None:
        assert len(exclude_solution) == size and all(len(row) == size for row in exclude_solution)

    chars = set(e for row in instance for e in row)
    chars.remove(EMPTY_CHAR)

    # Allow for one missing (dummy) character
    if len(chars) == size - 1:
        chars.update({DUMMY_CHAR})

    assert len(chars) == size

    map_char_to_number = {c: i for i, c in enumerate(chars)}
    map_number_to_char = {v: k for k, v in map_char_to_number.items()}

    logging.debug("Create solver..")
    problem = pulp.LpProblem("words", sense=pulp.LpMaximize)

    logging.debug("Add vars and constrs..")
    base_vars, (constrs_hor, constrs_ver, constrs_box, constrs_cell) = create_base_vars_and_constrs(instance, map_char_to_number)
    assert len(base_vars) == size ** 3 - (size - 1) * count_non_empty_cells(instance)
    assert len(constrs_hor) == size ** 2
    assert len(constrs_ver) == size ** 2
    assert len(constrs_box) == size ** 2
    assert len(constrs_cell) == size ** 2

    all_constrs = []
    all_constrs += list(itertools.chain(*[
        constrs.values()
        for constrs in [constrs_hor, constrs_ver, constrs_box, constrs_cell]
    ]))

    # Exclude solution
    if exclude_solution is not None:
        # TODO Only exclude last value
        constr = pulp.lpSum(
            base_vars[(map_char_to_number[exclude_solution[i1][i2]], i1, i2)]
            for i1 in range(size)
            for i2 in range(size)
        ) <= size ** 2 - 1
        # constr = base_vars[exclude_value] == 0
        constrs_prevent_solutions = [constr]
        all_constrs += constrs_prevent_solutions

    for constr in all_constrs:
        problem.addConstraint(constr)

    return problem, base_vars, map_number_to_char
