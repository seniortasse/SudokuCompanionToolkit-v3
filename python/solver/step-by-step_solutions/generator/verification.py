
from collections import Counter
import itertools
import logging
import pulp

from generator.model import EMPTY_CHAR, Instance, count_non_empty_cells, DUMMY_CHAR, DIMENSIONS
from generator.algo_human import solve_using_human_techniques, TECHNIQUES


logging.basicConfig(level=logging.INFO)


class NoSolutionException(Exception):
    pass


def check_has_solution(instance):
    try:
        solve(instance)
    except NoSolutionException:
        return False
    return True


def check_has_unique_solution(instance):
    """
    Check whether the instance has a unique solution is a two-step approach, and is used when the solution is not yet
    known/available:
      1: Find a solution for the instance
      2: Check whether there are more solutions besides the one found, by solving again excluding the first solution
    """

    try:
        solution = solve(instance)
    except NoSolutionException:
        return False

    is_unique_solution = check_is_unique_solution(instance, solution)

    return is_unique_solution


def check_is_unique_solution(instance, solution, preprocess=False):
    """
    Check whether the provided solution is the only possible solution for the given instance

    params:
      preprocess - whether to do an initial search with the human techniques to speed up the search (this can be toggled
       of for verification purposes)
    """

    # Although it might speed up the process, do not use human techniques to prefill as it is more sensitive to errors
    assert not preprocess

    # Speed up to verify unique solution: Search some values with human techniques
    if preprocess:
        prefilled_instance, _ = solve_using_human_techniques(instance, use_techniques=TECHNIQUES, include_magic_technique=False)
        if prefilled_instance == solution:
            return True
        else:
            # Check that the prefilling went correctly
            size = instance.size
            assert not any(prefilled_instance[i1][i2] != EMPTY_CHAR and prefilled_instance[i1][i2] != solution[i1][i2] for i1 in range(size) for i2 in range(size))
    else:
        prefilled_instance = instance

    instance = prefilled_instance

    try:
        solve(instance, exclude_solutions=[solution])
        solution_is_unique = False
    except NoSolutionException:
        solution_is_unique = True

    return solution_is_unique


def find_all_solutions(instance, max_number_solutions=None):
    """
    A natural extension of the helper functions above, is searching for all solutions for an instance
    """

    solutions = []

    while max_number_solutions is None or len(solutions) < max_number_solutions:
        try:
            solution = solve(instance, exclude_solutions=solutions)
            solutions.append(solution)
        except NoSolutionException:
            break

    return solutions


# TODO It seems we do not have a function yet which simply solves an instance and returns the solution; Remove this one
#  later
# TODO Make local and only use the above helper functions
def solve(instance, exclude_solutions=None):

    problem, base_vars, map_number_to_char = _create_base_model(instance, exclude_solutions)

    logging.debug("Solve problem..")
    solver = pulp.PULP_CBC_CMD(msg=False)
    status = problem.solve(solver)

    logging.debug(pulp.LpStatus[status])

    # TODO Return only the solution or fail if there is not solution
    # has_solution = status == 1
    if status != 1:
        raise NoSolutionException()

    size = instance.size
    solution = [[None] * size for _ in range(size)]
    for (n, i1, i2), var in base_vars.items():
        if var.value() == 1:
            solution[i1][i2] = map_number_to_char[n]

    # In case the layout has to be specified, copy this from the instance
    solution = Instance(solution, preprocessed_dims=instance.dims)

    # Check: All base constraints satisfied
    verify_base_constraints(solution)

    return solution


def _create_base_model(instance, exclude_solutions=None):

    size = instance.size

    if exclude_solutions is not None:
        for exclude_solution in exclude_solutions:
            assert len(exclude_solution) == size and all(len(row) == size for row in exclude_solution)

    chars = set(e for row in instance for e in row if e != EMPTY_CHAR)

    # Allow for one missing (dummy) character
    if len(chars) == size - 1:
        chars.update({DUMMY_CHAR})

    assert len(chars) == size

    # TODO We can remove this conversion entirely and remove the complexity of the code
    map_char_to_number = {c: i for i, c in enumerate(chars)}
    map_number_to_char = {v: k for k, v in map_char_to_number.items()}

    logging.debug("Create solver..")
    problem = pulp.LpProblem("words", sense=pulp.LpMaximize)

    logging.debug("Add vars and constrs..")
    base_vars, (constrs_hor, constrs_ver, constrs_box, constrs_cell) = _create_base_vars_and_constrs(instance, map_char_to_number)
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

    # TODO Experiment which implementation is faster: Only including one base_var for filled values, fixing the base_var
    #  for filled values, or both; We do use this functionality for fixing the main diagonal, so this is rather
    #  inconsistent;
    # # Fix instance values
    # for i1, row in enumerate(instance):
    #     for i2, e in enumerate(row):
    #         if e != EMPTY_CHAR:
    #             var = base_vars[(e, i1, i2)]
    #             var.setInitialValue(True)
    #             var.fixValue()
    #             logging.debug(f"Fix {(i1, i2)} to {e}")

    # Exclude solutions
    if exclude_solutions is not None:
        for exclude_solution in exclude_solutions:
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


def _create_base_vars_and_constrs(instance, map_char_to_number):

    size, layout_boxes = instance.size, instance.layout_boxes

    vars = {
        (n, i1, i2): pulp.LpVariable(f"x_{n}_{i1}_{i2}", cat="Binary")
        for i1 in range(size)
        for i2 in range(size)
        # TODO We might be able to speed up the solving by only creating vars for options which are present after
        #  applying singles
        for n in (range(size) if instance[i1][i2] == EMPTY_CHAR else [map_char_to_number[instance[i1][i2]]])
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
        (n, idx_box): pulp.lpSum(
            vars.get((n, i1, i2))
            for (i1, i2) in layout_boxes[idx_box]
        ) == 1
        for n in range(size)
        for idx_box in range(size)
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


def verify_base_constraints(solution, show_logs=False):

    size = solution.size
    chars = solution.chars
    assert isinstance(chars, set) and len(chars) == size

    # Correct characters used
    chars_solution = [e for row in solution for e in row]
    assert set(chars_solution) == chars, "Chars in solution do not match original chars"

    # Character count
    char_counts = Counter(chars_solution)
    assert all(e == size for e in char_counts.values()), "Not all chars included a correct amount"

    # All cells have a value
    for (i1, i2) in itertools.product(range(size), repeat=2):
        assert solution[i1][i2] in chars, f"No element present in cell {(i1, i2)}"
    if show_logs:
        print("Verified all cells")

    # All dims contain all values
    for dim in DIMENSIONS:
        for idx_dim in range(size):
            vals_for_dim = solution.get_values(dim, idx_dim)
            assert len(vals_for_dim) == size and set(vals_for_dim) == chars, f"Not all elements present in {dim} {idx_dim + 1}!"
        if show_logs:
            print(f"Verified all {dim}s")
