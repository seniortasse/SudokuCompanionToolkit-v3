"""Core Sudoku utilities used by higher-level techniques: index math, peers, house iterators, and grid mutation helpers."""

# solver_core.py (updated)
# Human-style Sudoku utilities:
# - candidate computation
# - naked singles & hidden singles (placements)
# - locked candidates (pointing & claiming) (eliminations)
# Grid is 9x9 list of lists of ints (0..9). 0 = blank.


Cell = tuple[int, int]  # (row, col) 1-based
Grid = list[list[int]]


def in_bounds(r: int, c: int) -> bool:
    return 1 <= r <= 9 and 1 <= c <= 9


def rc_to_key(r: int, c: int) -> str:
    return f"r{r}c{c}"


def key_to_rc(key: str) -> Cell:
    r = int(key.split("c")[0][1:])
    c = int(key.split("c")[1])
    return (r, c)


def clone_grid(grid: Grid) -> Grid:
    return [row[:] for row in grid]


def row_values(grid: Grid, r: int) -> set:
    return set(grid[r - 1]) - {0}


def col_values(grid: Grid, c: int) -> set:
    return {grid[i][c - 1] for i in range(9)} - {0}


def box_values(grid: Grid, r: int, c: int) -> set:
    r0 = 3 * ((r - 1) // 3)
    c0 = 3 * ((c - 1) // 3)
    vals = {grid[r0 + i][c0 + j] for i in range(3) for j in range(3)} - {0}
    return vals


def peers(grid: Grid, r: int, c: int):
    """Return the set of peer coordinates for a given cell (same row, column, and 3x3 box)."""
    ps = set()
    for j in range(1, 10):
        if j != c:
            ps.add((r, j))
    for i in range(1, 10):
        if i != r:
            ps.add((i, c))
    r0 = 3 * ((r - 1) // 3)
    c0 = 3 * ((c - 1) // 3)
    for i in range(3):
        for j in range(3):
            rr = r0 + i + 1
            cc = c0 + j + 1
            if (rr, cc) != (r, c):
                ps.add((rr, cc))
    return ps


def unit_cells_row(r: int):
    return [(r, c) for c in range(1, 10)]


def unit_cells_col(c: int):
    return [(r, c) for r in range(1, 10)]


def unit_cells_box(b: int):
    br = (b - 1) // 3
    bc = (b - 1) % 3
    r0 = 3 * br + 1
    c0 = 3 * bc + 1
    return [(r0 + i, c0 + j) for i in range(3) for j in range(3)]


def which_box(r: int, c: int) -> int:
    return 3 * ((r - 1) // 3) + ((c - 1) // 3) + 1


def compute_candidates(grid: Grid) -> dict[str, list]:
    cand = {}
    for r in range(1, 10):
        for c in range(1, 10):
            if grid[r - 1][c - 1] == 0:
                used = row_values(grid, r) | col_values(grid, c) | box_values(grid, r, c)
                opts = [d for d in range(1, 10) if d not in used]
                cand[rc_to_key(r, c)] = opts
    return cand


def find_naked_singles(grid: Grid, candidates: dict[str, list]):
    moves = []
    for key, opts in candidates.items():
        if len(opts) == 1:
            r, c = key_to_rc(key)
            moves.append(
                {
                    "technique": "naked_single",
                    "type": "placement",
                    "cell": key,
                    "digit": opts[0],
                    "explanation": {
                        "why": f"Only one candidate fits r{r}c{c}.",
                        "units": {"row": f"r{r}", "col": f"c{c}", "box": f"b{which_box(r,c)}"},
                    },
                    "highlights": {"cells": [key]},
                }
            )
    return moves


def find_hidden_singles(grid: Grid, candidates: dict[str, list]):
    moves = []
    # rows
    for r in range(1, 10):
        pos_for_digit = {d: [] for d in range(1, 10)}
        for c in range(1, 10):
            key = rc_to_key(r, c)
            if key in candidates:
                for d in candidates[key]:
                    pos_for_digit[d].append(key)
        for d, cells in pos_for_digit.items():
            if len(cells) == 1:
                key = cells[0]
                rr, cc = key_to_rc(key)
                moves.append(
                    {
                        "technique": "hidden_single",
                        "type": "placement",
                        "cell": key,
                        "digit": d,
                        "explanation": {
                            "why": f"Digit {d} appears in only one cell in row {r}.",
                            "units": {"row": f"r{r}", "box": f"b{which_box(rr,cc)}"},
                        },
                        "highlights": {"cells": [key], "row": f"r{r}"},
                    }
                )
    # cols
    for c in range(1, 10):
        pos_for_digit = {d: [] for d in range(1, 10)}
        for r in range(1, 10):
            key = rc_to_key(r, c)
            if key in candidates:
                for d in candidates[key]:
                    pos_for_digit[d].append(key)
        for d, cells in pos_for_digit.items():
            if len(cells) == 1:
                key = cells[0]
                rr, cc = key_to_rc(key)
                moves.append(
                    {
                        "technique": "hidden_single",
                        "type": "placement",
                        "cell": key,
                        "digit": d,
                        "explanation": {
                            "why": f"Digit {d} appears in only one cell in column {c}.",
                            "units": {"col": f"c{c}", "box": f"b{which_box(rr,cc)}"},
                        },
                        "highlights": {"cells": [key], "col": f"c{c}"},
                    }
                )
    # boxes
    for b in range(1, 10):
        pos_for_digit = {d: [] for d in range(1, 10)}
        for r, c in unit_cells_box(b):
            key = rc_to_key(r, c)
            if key in candidates:
                for d in candidates[key]:
                    pos_for_digit[d].append(key)
        for d, cells in pos_for_digit.items():
            if len(cells) == 1:
                key = cells[0]
                rr, cc = key_to_rc(key)
                moves.append(
                    {
                        "technique": "hidden_single",
                        "type": "placement",
                        "cell": key,
                        "digit": d,
                        "explanation": {
                            "why": f"Digit {d} appears in only one cell in box {b}.",
                            "units": {"box": f"b{b}"},
                        },
                        "highlights": {"cells": [key], "box": f"b{b}"},
                    }
                )
    return moves


def find_locked_candidates_pointing(grid: Grid, candidates: dict[str, list]):
    """If in a box, a digit's candidates lie in a single row (or column), eliminate that digit
    from the rest of that row (or column) outside the box.
    """
    moves = []
    for b in range(1, 10):
        cells = unit_cells_box(b)
        for d in range(1, 10):
            locs = [
                rc_to_key(r, c)
                for (r, c) in cells
                if rc_to_key(r, c) in candidates and d in candidates[rc_to_key(r, c)]
            ]
            if len(locs) < 2:
                continue
            rows = sorted(set([key_to_rc(k)[0] for k in locs]))
            cols = sorted(set([key_to_rc(k)[1] for k in locs]))
            # single row inside box
            if len(rows) == 1:
                r = rows[0]
                # eliminate d from row r outside the box
                elim = []
                for c in range(1, 10):
                    if (r, c) not in cells:
                        key = rc_to_key(r, c)
                        if key in candidates and d in candidates[key]:
                            elim.append(key)
                if elim:
                    moves.append(
                        {
                            "technique": "locked_candidates_pointing",
                            "type": "elimination",
                            "digit": d,
                            "box": f"b{b}",
                            "line": f"r{r}",
                            "eliminate": elim,
                            "in_box": locs,
                            "explanation": {
                                "why": f"In box {b}, digit {d}'s candidates lie only in row {r}. Eliminate {d} from row {r} outside this box."
                            },
                            "highlights": {
                                "box": f"b{b}",
                                "row": f"r{r}",
                                "cells": elim,
                                "in_box": locs,
                            },
                        }
                    )
            # single column inside box
            if len(cols) == 1:
                c = cols[0]
                elim = []
                for r in range(1, 10):
                    if (r, c) not in cells:
                        key = rc_to_key(r, c)
                        if key in candidates and d in candidates[key]:
                            elim.append(key)
                if elim:
                    moves.append(
                        {
                            "technique": "locked_candidates_pointing",
                            "type": "elimination",
                            "digit": d,
                            "box": f"b{b}",
                            "line": f"c{c}",
                            "eliminate": elim,
                            "in_box": locs,
                            "explanation": {
                                "why": f"In box {b}, digit {d}'s candidates lie only in column {c}. Eliminate {d} from column {c} outside this box."
                            },
                            "highlights": {
                                "box": f"b{b}",
                                "col": f"c{c}",
                                "cells": elim,
                                "in_box": locs,
                            },
                        }
                    )
    return moves


def find_locked_candidates_claiming(grid: Grid, candidates: dict[str, list]):
    """If in a row/column, a digit's candidates are confined to a single box, eliminate that digit
    from other cells in that box.
    """
    moves = []
    # rows
    for r in range(1, 10):
        for d in range(1, 10):
            locs = [
                rc_to_key(r, c)
                for c in range(1, 10)
                if rc_to_key(r, c) in candidates and d in candidates[rc_to_key(r, c)]
            ]
            if len(locs) < 2:
                continue
            boxes = sorted(set([which_box(*key_to_rc(k)) for k in locs]))
            if len(boxes) == 1:
                b = boxes[0]
                box_cells = unit_cells_box(b)
                elim = []
                for rr, cc in box_cells:
                    if rr != r:
                        key = rc_to_key(rr, cc)
                        if key in candidates and d in candidates[key]:
                            elim.append(key)
                if elim:
                    moves.append(
                        {
                            "technique": "locked_candidates_claiming",
                            "type": "elimination",
                            "digit": d,
                            "box": f"b{b}",
                            "line": f"r{r}",
                            "eliminate": elim,
                            "in_line": locs,
                            "explanation": {
                                "why": f"In row {r}, digit {d}'s candidates are confined to box {b}. Eliminate {d} from other cells in box {b}."
                            },
                            "highlights": {
                                "box": f"b{b}",
                                "row": f"r{r}",
                                "cells": elim,
                                "in_line": locs,
                            },
                        }
                    )
    # cols
    for c in range(1, 10):
        for d in range(1, 10):
            locs = [
                rc_to_key(r, c)
                for r in range(1, 10)
                if rc_to_key(r, c) in candidates and d in candidates[rc_to_key(r, c)]
            ]
            if len(locs) < 2:
                continue
            boxes = sorted(set([which_box(*key_to_rc(k)) for k in locs]))
            if len(boxes) == 1:
                b = boxes[0]
                box_cells = unit_cells_box(b)
                elim = []
                for rr, cc in box_cells:
                    if cc != c:
                        key = rc_to_key(rr, cc)
                        if key in candidates and d in candidates[key]:
                            elim.append(key)
                if elim:
                    moves.append(
                        {
                            "technique": "locked_candidates_claiming",
                            "type": "elimination",
                            "digit": d,
                            "box": f"b{b}",
                            "line": f"c{c}",
                            "eliminate": elim,
                            "in_line": locs,
                            "explanation": {
                                "why": f"In column {c}, digit {d}'s candidates are confined to box {b}. Eliminate {d} from other cells in box {b}."
                            },
                            "highlights": {
                                "box": f"b{b}",
                                "col": f"c{c}",
                                "cells": elim,
                                "in_line": locs,
                            },
                        }
                    )
    return moves


def apply_move(grid: Grid, move: dict) -> Grid:
    """Placement only; returns a new grid with the digit placed."""
    r, c = key_to_rc(move["cell"])
    g2 = clone_grid(grid)
    g2[r - 1][c - 1] = move["digit"]
    return g2
