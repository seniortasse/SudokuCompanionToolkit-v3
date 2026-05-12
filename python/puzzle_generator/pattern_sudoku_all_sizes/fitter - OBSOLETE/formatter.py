
import colorama

from fitter.positioner import subword_overlaps_idx


colorama.init(autoreset=True)


def determine_highlights(size, subwords_placements):

    highlights = [[False for _ in range(size)] for _ in range(size)]

    for i1 in range(size):
        for i2 in range(size):
            # We can reuse the overlap functionality here
            any_subword_overlaps = any(
                subword_overlaps_idx(subword, placement_i1, placement_i2, orientation, i1, i2)
                for subword, orientation, placement_i1, placement_i2 in subwords_placements
            )
            if any_subword_overlaps:
                highlights[i1][i2] = True

    return highlights


def pretty_print(solution, highlights, is_fixed_diagonal):
    for i1, row in enumerate(solution):
        print('[', end='')
        for i2, e in enumerate(row):
            if highlights[i1][i2]:
                text = colorama.Fore.RED + e
            elif is_fixed_diagonal and i1 == i2:
                text = colorama.Fore.LIGHTYELLOW_EX + e
            else:
                text = e
            print('\'', end='')
            print(text, end='')
            print('\'', end='')
            if i2 < len(row) - 1:
                print(', ', end='')
        print(']')
