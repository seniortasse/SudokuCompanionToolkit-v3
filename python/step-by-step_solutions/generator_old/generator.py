
from collections import Counter
import itertools
import random

from generator.model import EMPTY_CHAR, count_empty_cells, count_non_empty_cells, copy_instance, Instance, is_only_char
from generator.algo_human import solve_using_human_techniques, TECHNIQUES
from generator.selection import select_next_idx
from generator.verification import verify_has_unique_solution


# found_instance = None


# TODO A speedup of is_solvable_with_human_techniques: Store cache of already solved partial solutions; Especially when
#  difficult techniques are required, which take the most time, probably the search always gets stuck on the exact same
#  instance -- So possibly only store partial solutions which require at least some difficult technique, to be tuned,
#  as a balance between memory usage and speed


def can_remove_idx(instance, idxs, solution, only_accept_human_solvable_instances, allowed_techniques):

    # Check whether the character is the only occurrence, in which case we cannot remove it (this check is added to this
    # function to be able to check validity of removing subwords more easily, and in general to centralise the validity
    # constraints)

    instance = copy_instance(instance)

    for idx in idxs:
        instance[idx[0]][idx[1]] = EMPTY_CHAR

    # Check that all characters will still be present after removing the values
    chars_present = {char for row in instance for char in row if char != EMPTY_CHAR}
    if chars_present != instance.chars:
        vals = tuple(f"'{solution[idx[0]][idx[1]]}'" for idx in idxs)
        print(f"Cannot remove {vals} at {tuple(idxs)} as it removes a single-occurrence character")
        result = "no:single_occurrence"
        return result

    # With some of the newest techniques it is possible to find a new value even if the solution is not unique, so we
    #  have to do this check first
    has_unique_solution = verify_has_unique_solution(instance, solution)
    if not has_unique_solution:
        result = "no:no_unique_solution"
        return result

    # Check first whether it is humanly solvable, as this is faster to check;
    # If the instance is humanly solvable, it also means it is unique -> why is this?
    if only_accept_human_solvable_instances:
        # At every step, check whether the instance remains solvable using human techniques
        solved_instance, _ = solve_using_human_techniques(
            instance, use_techniques=allowed_techniques,
            include_magic_technique=not only_accept_human_solvable_instances, magic_solution=solution
        )
        if solved_instance == solution:
            result = "yes"
        else:
            result = "no:not_solvable_with_techniques"
    else:
        result = "yes"

    # for idx in idxs:
    #     instance[idx[0]][idx[1]] = solution[idx[0]][idx[1]]

    return result


def generate_instance(solution, target_non_empty_cells=0, allowed_techniques=TECHNIQUES, max_uses_techniques={},
                      only_accept_human_solvable_instances=True, options={}, show_logs=False) -> (list, (dict, object)):
    """
    Recognised options:
     - empty-diagonal - bool: remove the entire diagonal
     - remove-idxs - list of tuples (row_idx, col_idx): try to remove as many of those idxs as possible
    """

    # if not isinstance(solution, Instance):
    #     solution = Instance(solution, length, size)
    assert isinstance(solution, Instance)

    size = solution.size

    logs = {}

    # Iterative removal
    instance = solution.copy()

    # Initialise the random seed for every run, to be able to reproduce individual runs
    seed = random.randint(0, 100_000)
    # TODO Optionally, be able to give this as an argument
    random.seed(seed)
    print("Using random seed:", seed)

    idxs_to_check = [(i1, i2) for i1 in range(size) for i2 in range(size)]
    random.shuffle(idxs_to_check)

    sym = options.get("-s")

    # First remove the diagonal (note: the will always be uniquely and easily solvable by only removing the diagonal);
    #  We assume that the target number of empty cells is always at least the grid size, so this does not violate any
    #  final result constraints
    if options.get("empty-diagonal"):
        print(f"Pre-removing diagonal..")
        for i in range(size):
            idx = (i, i)
            idxs_to_check.remove(idx)
            instance[i][i] = EMPTY_CHAR

    # Remove specific idxs
    # TODO These can be prepended to idxs_to_check, so that they are tried first, while the code is more generalised
    if (_remove_idxs := options.get("remove-idxs")) is not None:
        print(f"Pre-removing {len(_remove_idxs)} values:")
        # Shuffle this list so that if a certain order does not lead to a valid instance, the next time we try another
        #  order is tried
        random.shuffle(_remove_idxs)
        for idx in _remove_idxs:
            # Skip when the value was already removed from the diagonal
            if idx not in idxs_to_check:
                continue
            # Steering conditions
            #  -> We already have to check those here, as for small instances we might remove too many values when
            #     removing all subwords
            if count_non_empty_cells(instance) == target_non_empty_cells:
                print(f"Target number of empty cells reached while removing values for subwords: {target_non_empty_cells}")
                break
            char = instance[idx[0]][idx[1]]
            print("", char, "at", idx)
            # has_unique_solution = verify_has_unique_solution(instance, solution)
            # All constraints that would otherwise hold when removing a value should be checked here, as otherwise this
            #  step could lead to an invalid result: Uniquely solvable, solvable with allowed techniques, at least 1
            #  occurrence of each character present; Done this by moving the single occurrence check to
            #  can_remove_idx(), which is the main function checking whether removal of an idx is valid
            result = can_remove_idx(instance, [idx], solution, only_accept_human_solvable_instances, allowed_techniques)
            if result == "yes":
                instance[idx[0]][idx[1]] = EMPTY_CHAR
            else:
                print(f"WARNING: Could not remove '{char}' at {idx} as it would violate one of the constraints:", " ".join(result[3:].split('_')))
            idxs_to_check.remove(idx)

    print()
    print("Instance after removing diagonal and/or subwords:")
    print(instance)

    step = 0
    while len(idxs_to_check) > 0 and count_non_empty_cells(instance) > target_non_empty_cells:
        step += 1

        logs[step] = {}
        logs[step]["instance"] = copy_instance(instance)

        number_empty_cells = count_empty_cells(instance)
        print(f"Step {step}, {size ** 2 - number_empty_cells} non empty cells")
        print(instance)

        # Do not use selection for low ratings as we do not need advanced techniques, and it would steer the algorithm
        #  towards removing values in the same regions
        if size <= 9:
            use_selection = False
        else:
            use_selection = target_non_empty_cells < 30

        if not use_selection:
            print("Select next value randomly")
            # idx = random.choice(idxs_to_check)
            # Note: idxs have been shuffled already
            idx = idxs_to_check[0]
        else:
            print("Select next value based on which removal leads to the largest weight")
            idx = select_next_idx(
                instance, idxs_to_check, solution,
                allowed_techniques, max_uses_techniques, only_accept_human_solvable_instances, show_logs
            )

        idxs = [idx]

        # Add idxs based on symmetry
        if sym is not None:
            if sym == "center":
                idx_to_add = (size - 1 - idx[0], size - 1 - idx[1])
            elif sym == "horizontal":
                idx_to_add = (idx[0], size - 1 - idx[1])
            elif sym == "vertical":
                idx_to_add = (size - 1 - idx[0], idx[1])
            elif sym == "diagonal-1":
                idx_to_add = (idx[1], idx[0])
            elif sym == "diagonal-2":
                idx_to_add = (size - 1 - idx[1], size - 1 - idx[0])
            else:
                raise Exception(f"Symmetry option {sym} not implemented")
            # Some cells do not have a symmetric version, prevent adding them twice
            if idx_to_add != idx:
                # Note: To avoid crashing when subwords/diagonal have been removed first, check whether the value is
                #  still present
                if idx_to_add in idxs_to_check:
                    idxs.append(idx_to_add)

        vals = tuple(f"'{solution[idx[0]][idx[1]]}'" for idx in idxs)
        print(f"Try to remove {vals} at {tuple(idxs)} ({len(idxs_to_check)} more remaining)")

        for idx in idxs:
            idxs_to_check.remove(idx)

        result = can_remove_idx(instance, idxs, solution, only_accept_human_solvable_instances, allowed_techniques)
        if result == "yes":
            for idx in idxs:
                instance[idx[0]][idx[1]] = EMPTY_CHAR
            print(" -> successful")
        else:
            print(" -> cannot remove:", " ".join(result[3:].split('_')))

        # print("Empty cells:", number_empty_cells)
        # print("Result:", result)
        # print("Remaining idxs to try to remove:", len(idxs))

        print()

        # Steering conditions -> Now checked in the while-loop
        # number_non_empty_cells = count_non_empty_cells(instance)
        # if number_non_empty_cells <= min_non_empty_cells:
        #     break

    print("Accepted solution:")
    print(instance)

    count = sum(e == EMPTY_CHAR for row in instance for e in row)
    print("Number values removed:", count)
    print("Number values remaining:", size ** 2 - count)
    counts = Counter(e for row in instance for e in row if e != EMPTY_CHAR)
    print("Remaining element counts:", dict(sorted(counts.items(), key=lambda item: item[1], reverse=True)))

    # Final metric
    _solved_instance, (_techniques, _) = solve_using_human_techniques(instance, use_techniques=allowed_techniques)
    print("Techniques used:", _techniques)

    # Final verification of the instance:
    #  - the instance has a unique solution
    #  - the instance can be solved with the allowed techniques
    #  - each character occurs at least once
    assert verify_has_unique_solution(instance, solution, preprocess=False)
    assert not set(_techniques.keys()).difference(allowed_techniques), f"Allowed: {allowed_techniques}, Used: {_techniques}"
    assert not instance.chars.difference(*itertools.chain(instance))

    return instance, (_techniques, logs)
