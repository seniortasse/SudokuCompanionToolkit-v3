from __future__ import annotations
from typing import List, Dict, Optional, Tuple, Any
from types_sudoku import Grid, Candidates, Move
"""CLI orchestrator for the demo pipeline. Loads an image, runs OpenCV rectification, obtains a (mock) board state, computes candidates and next moves with chaining, renders overlay images per move, and prints a JSON report to stdout."""


# demo_cli_overlay.py
# Demo CLI with visual overlays per move.
# Usage:
#   python demo_cli_overlay.py --image /path/to/photo.jpg --out demo_export --mode demo --max_moves 5
import argparse, json
from pathlib import Path

from vision.rectify.opencv_rectify import process as rectify_process
from solver.sudoku_tools import compute_candidates_tool, next_moves
from .overlay_renderer import draw_move
from .demo_cli import mock_classify_cells  # reuse

def main(args=None) -> None:
    if args is None:
        # Parse here if caller didn’t pass args
        ap = argparse.ArgumentParser()
        ap.add_argument("--image", required=True)
        ap.add_argument("--out", type=str, default="demo_export")
        ap.add_argument("--mode", type=str, default="demo", choices=["demo","random","blank"])
        ap.add_argument("--seed", type=int, default=123)
        ap.add_argument("--max_moves", type=int, default=5)
        ap.add_argument("--json", type=str, default=None)
        args = ap.parse_args()

    export_dir = Path(args.out)
    export_dir.mkdir(parents=True, exist_ok=True)

    info = rectify_process(args.image, export_dir)
    original, current = mock_classify_cells(info["cells_json"], mode=args.mode, seed=args.seed)

    cands  = compute_candidates_tool(current)["candidates"]
    result = next_moves(current, cands, max_difficulty="locked", max_moves=args.max_moves, chain=True)

    warped = info["warped"]
    moves  = result["moves"]

    overlays = []
    for i, mv in enumerate(moves, 1):
        out = export_dir / f"overlay_move_{i:02d}.jpg"
        draw_move(warped, mv, str(out))
        overlays.append(str(out))

    payload = {"rectify": info, "moves": moves, "overlays": overlays}

    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2))
    


if __name__ == "__main__":
    main()  # <- no args passed; main() will parse