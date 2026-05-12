
from collections import Counter


def verify_base_constraints(solution, dims, chars: set):
    """
    Verify the base Sudoku rules, including a check that the correct characters were used in the solution
    Note that this does not include checks for subwords, as these are not part of the basic constraints
    """

    box_height, box_width, size = dims

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
