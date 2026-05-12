
from generator.model import copy_instance, EMPTY_CHAR


def apply_pattern(solution, pattern):
    """
    Generate an instance from a solution by removing the values as specified by the pattern.

    Data structure of pattern:
      True - Keep value
      False - Remove value
    """

    instance = copy_instance(solution)

    for i1, row in enumerate(pattern):
        for i2, e in enumerate(row):
            keep_value = e
            if not keep_value:
                instance[i1][i2] = EMPTY_CHAR

    return instance
