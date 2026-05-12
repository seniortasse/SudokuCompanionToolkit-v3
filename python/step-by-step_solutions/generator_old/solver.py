
from collections import Counter
import pulp

from generator.model import EMPTY_CHAR


def create_base_vars_and_constrs(instance, map_char_to_number):

    box_height, box_width, size = instance.box_height, instance.box_width, instance.size

    vars = {
        (n, i1, i2): pulp.LpVariable(f"x_{n}_{i1}_{i2}", cat="Binary")
        for i1 in range(size)
        for i2 in range(size)
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


def verify_base_constraints(solution):

    box_height, box_width, size = solution.box_height, solution.box_width, solution.size

    chars = solution.chars

    # Correct characters used
    solution_chars = [e for row in solution for e in row]
    assert set(solution_chars) == chars, "Chars in solution do not match original chars"

    # Character count
    char_counts = Counter(solution_chars)
    assert all(e == size for e in char_counts.values()), "Not all chars included a correct amount"

    # All cells have a value
    for i1 in range(size):
        for i2 in range(size):
            assert solution[i1][i2] in chars, f"No element present in cell {(i1, i2)}"
    print("Verified all cells")

    # All rows contain all values
    for i1 in range(size):
        assert set(solution[i1]) == chars, f"Not all elements present in row {i1}!"
    print("Verified all rows")

    # All cols contain all values
    for i2 in range(size):
        assert set([solution[i1][i2] for i1 in range(size)]) == chars, f"Not all elements present in col {i2}!"
    print("Verified all cols")

    # All boxs contain all values
    for b1 in range(size // box_height):
        for b2 in range(size // box_width):
            chars_in_box = [
                solution[box_height * b1 + i1][box_width * b2 + i2]
                for i1 in range(box_height)
                for i2 in range(box_width)
            ]
            assert set(chars_in_box) == chars, f"Not all elements present in box {(b1, b2)}!"
    print("Verified all boxes")
