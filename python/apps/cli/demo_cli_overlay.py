# apps/cli/demo_cli_overlay.py
from __future__ import annotations
"""
Demo CLI with visual overlays per move.

Now supports TWO modes:
- Classifier mode (default if --model is provided): runs your CNN on the 81 cell crops
  produced by rectification and uses those digits as the board state.
- Demo/mock mode (fallback if no --model): uses the existing mock generator.

Also adds:
--calib <path/to/calibration.json>
--inner-crop <fraction> (e.g., 0.92 to keep the central 92% of each tile)
--low, --margin thresholds to compute/print low-confidence counts.
"""

import argparse
import json
from pathlib import Path

from solver.sudoku_tools import compute_candidates_tool, next_moves
from vision.rectify.opencv_rectify import process as rectify_process
from vision.infer.classify_cells_model import predict_folder

# Fallback demo generator (unchanged)
from .demo_cli import mock_classify_cells
from .overlay_renderer import draw_move


def run_classifier_on_export(cells_dir: Path,
                             model: str,
                             img: int,
                             device: str,
                             calib: str | None,
                             low: float | None,
                             margin: float | None,
                             inner_crop: float) -> dict:
    """Runs the CNN on export_dir/cells and returns the predict_folder() payload."""
    out = predict_folder(
        str(cells_dir),
        model_path=model,
        img_size=img,
        device=device,
        calib=calib,
        low=low,
        margin_thr=margin,
        inner_crop=inner_crop,
    )
    return out


def main(args=None) -> None:
    if args is None:
        ap = argparse.ArgumentParser()
        ap.add_argument("--image", required=True, help="Path to a Sudoku photo")
        ap.add_argument("--out", type=str, default="demo_export", help="Export folder")
        # Solver/demo options
        ap.add_argument("--mode", type=str, default="demo",
                        choices=["demo", "random", "blank"],
                        help="Used only if --model is not provided (mock mode)")
        ap.add_argument("--seed", type=int, default=123)
        ap.add_argument("--max_moves", type=int, default=5)

        # Classifier options
        ap.add_argument("--model", type=str, default="", help="Path to best.pt (enable classifier mode)")
        ap.add_argument("--device", type=str, default="cpu")
        ap.add_argument("--img", type=int, default=28)
        ap.add_argument("--calib", type=str, default="", help="Path to calibration.json")
        ap.add_argument("--inner-crop", type=float, default=1.0, help="Center-crop fraction (e.g., 0.92)")
        ap.add_argument("--low", type=float, default=None, help="Flag low if top-1 prob < this")
        ap.add_argument("--margin", type=float, default=None, help="Flag low if (p1-p2) < this")

        # Output
        ap.add_argument("--json", type=str, default=None, help="Write full payload JSON here")
        args = ap.parse_args()

    export_dir = Path(args.out)
    export_dir.mkdir(parents=True, exist_ok=True)

    # 1) Rectify -> writes export_dir/warped.jpg and export_dir/cells/r#c#.png
    info = rectify_process(args.image, export_dir, precompress=True)

    # 2) Detect digits: classifier OR demo fallback
    clf_payload = None
    if args.model:
        cells_dir = export_dir / "cells"
        clf_payload = run_classifier_on_export(
            cells_dir=cells_dir,
            model=args.model,
            img=args.img,
            device=args.device,
            calib=(args.calib or None),
            low=(args.low if args.low is not None else None),
            margin=(args.margin if args.margin is not None else None),
            inner_crop=args.inner_crop,
        )
        original = clf_payload["grid"]  # 9x9 ints (0..9, where 0=blank)
        current  = clf_payload["grid"]  # MVP: same as original
        # Friendly log about low-confidence (if thresholds were provided)
        lows = 0
        if "low_conf" in clf_payload:
            lows = int(sum(sum(row) for row in clf_payload["low_conf"]))
        board_name = Path(args.image).stem
        if args.low is not None or args.margin is not None:
            print(f"{board_name}: {lows} low-confidence cells out of 81")
        else:
            print(f"{board_name}: 81 tiles (no thresholds set)")
    else:
        # fallback demo
        original, current = mock_classify_cells(info["cells_json"], mode=args.mode, seed=args.seed)
        print("Demo/mock mode (no --model): generated a board state")

    # 3) Candidates + next moves
    cands = compute_candidates_tool(current)["candidates"]
    result = next_moves(
        current, cands, max_difficulty="locked", max_moves=args.max_moves, chain=True
    )
    moves = result["moves"]

    # 4) Render solver overlays on the rectified board
    warped = info["warped"]
    overlays = []
    for i, mv in enumerate(moves, 1):
        out = export_dir / f"overlay_move_{i:02d}.jpg"
        draw_move(warped, mv, str(out))
        overlays.append(str(out))

    # 5) Emit a single JSON payload (includes rectification, classifier meta, moves, overlays)
    payload = {
        "rectify": info,
        "classifier": clf_payload,  # None in demo/mock mode
        "board": {
            "original": original,
            "current": current
        },
        "moves": moves,
        "overlays": overlays,
    }

    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()