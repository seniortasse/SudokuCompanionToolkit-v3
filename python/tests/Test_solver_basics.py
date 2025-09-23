# tests/test_solver_basics.py
from solver.sudoku_tools import compute_candidates_tool, next_moves


def test_naked_hidden_singles_basic():
    # Simple grid with one obvious single at r1c1 = 5
    grid = [
        [0, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9],
    ]
    cand = compute_candidates_tool(grid)["candidates"]
    moves = next_moves(grid, cand, max_difficulty="locked", max_moves=3, chain=True)
    # Ensure we got at least one move and it has expected fields
    assert "moves" in moves or isinstance(moves, dict)  # tolerate dict or list wrapper
    seq = moves["moves"] if isinstance(moves, dict) else moves
    assert len(seq) >= 1
    m0 = seq[0]
    assert m0["type"] in ("placement", "elimination")
    # If the first move is a placement, it must have "cell" and "digit"
    if m0["type"] == "placement":
        assert "cell" in m0 and "digit" in m0
