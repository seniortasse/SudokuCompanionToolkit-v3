"""Support routines for the demo mode: deterministic mock classifier to simulate reading digits and building a plausible Sudoku state without ML."""

# demo_cli.py
# End-to-end demo:
# - Takes a photo path
# - Runs OpenCV rectification to extract 81 cells
# - Uses a MOCK classifier to produce a "current" grid (random digits with blanks)
# - Computes candidates
# - Calls next_moves with chaining to show suggested steps
#
# Usage:
#   python demo_cli.py --image /path/to/photo.jpg --mode random --seed 123

import argparse
import json
import random
from pathlib import Path

from opencv_rectify import process as rectify_process
from sudoku_tools import compute_candidates_tool, next_moves


def mock_classify_cells(cells_json, mode="random", seed=123):
    """Return (original, current) 9x9 grids using a simple mock strategy.
    - 'random': 65% blanks, else random digit as 'printed' (original+current).
    - 'demo': a small fixed grid.
    - 'blank': all zeros.
    """
    random.seed(seed)
    original = [[0] * 9 for _ in range(9)]
    current = [[0] * 9 for _ in range(9)]
    if mode == "blank":
        return original, current
    if mode == "demo":
        demo = [
            [6, 2, 4, 3, 8, 0, 1, 0, 9],
            [5, 0, 0, 0, 9, 0, 0, 0, 7],
            [1, 0, 9, 0, 2, 6, 4, 3, 8],
            [8, 0, 0, 0, 6, 0, 0, 0, 1],
            [9, 1, 5, 8, 7, 0, 6, 0, 2],
            [4, 0, 0, 0, 1, 0, 0, 0, 3],
            [3, 0, 1, 0, 4, 2, 8, 7, 5],
            [7, 0, 0, 0, 3, 0, 0, 0, 4],
            [2, 4, 8, 7, 5, 0, 3, 0, 6],
        ]
        return demo, [row[:] for row in demo]
    # random
    with open(cells_json, encoding="utf-8") as f:
        cells = json.load(f)
    for cell in cells:
        r = cell["r"] - 1
        c = cell["c"] - 1
        if random.random() < 0.65:
            continue
        d = random.randint(1, 9)
        original[r][c] = d
        current[r][c] = d
    return original, current


def main(args):
    export_dir = Path(args.out)
    export_dir.mkdir(parents=True, exist_ok=True)
    info = rectify_process(args.image, export_dir)
    original, current = mock_classify_cells(info["cells_json"], mode=args.mode, seed=args.seed)
    cands = compute_candidates_tool(current)["candidates"]
    result = next_moves(current, cands, max_difficulty="locked", max_moves=5, chain=True)
    payload = {
        "rectify": info,
        "original": original,
        "current": current,
        "candidates_count": sum(len(v) for v in cands.values()),
        "moves": result["moves"],
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", type=str, default="demo_export")
    ap.add_argument("--mode", type=str, default="random", choices=["random", "demo", "blank"])
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()
    main(args)
