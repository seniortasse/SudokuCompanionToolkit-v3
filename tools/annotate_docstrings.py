# tools/annotate_docstrings.py
"""Docstring Annotator.
-------------------
Adds concise, helpful docstrings to key modules and functions in the Sudoku Companion Toolkit.
- Safe by default: does not overwrite existing docstrings.
- Supports --dry-run to preview changes and --apply to write in-place.
- Uses simple regex-based insertion; it will skip functions it cannot confidently match.

Usage (from the project root):
  python tools/annotate_docstrings.py --dry-run
  python tools/annotate_docstrings.py --apply

If your local folder names differ, edit the TARGETS list or provide --base to point to a different root.

Limitations:
- Only inserts into the commonly shared files we expect. If a file or function is missing, it is skipped and reported.
- For complex decorators or unusual formatting, the insertion may skip to stay safe.
"""
import argparse
import json
import re
from pathlib import Path

# Files we expect and the functions/classes we aim to annotate.
TARGETS = {
    "apps/cli/demo_cli_overlay.py": {
        "module": "CLI orchestrator for the demo pipeline. Loads an image, runs OpenCV rectification, obtains a (mock) board state, computes candidates and next moves with chaining, renders overlay images per move, and prints a JSON report to stdout.",
        "functions": {
            "parse_args": "Parse CLI arguments: --image (path), --out (output directory), --mode (demo/real), --max_moves (int), --seed (optional). Returns an argparse.Namespace.",
            "main": "End-to-end demo runner. Steps: (1) rectify board image; (2) classify 81 cells (mock in demo mode); (3) compute candidates; (4) call next_moves with chaining; (5) render per-move overlays; (6) assemble and print a JSON report to stdout.",
        },
    },
    "apps/cli/demo_cli.py": {
        "module": "Support routines for the demo mode: deterministic mock classifier to simulate reading digits and building a plausible Sudoku state without ML.",
        "functions": {
            "mock_classify_cells": "Return a tuple (original, current) 9x9 lists. Simulates printed givens and current big digits for demo purposes based on random seed or fixed pattern."
        },
    },
    "apps/cli/overlay_renderer.py": {
        "module": "Rendering utilities to draw move highlights over the rectified board image (rows/cols/boxes/digits). Produces overlay_move_XX.jpg artifacts used by storyboard and animations.",
        "functions": {
            "draw_move": "Render a single move overlay. Inputs: base board image path, move dict (technique, type, cell/digit/eliminate, highlights), output path. Draws colored shapes/text and saves a JPEG."
        },
    },
    "apps/cli/storyboard_sheet.py": {
        "module": "Build a printable storyboard sheet (PNG/PDF) that tiles overlay images with captions. Optionally reads moves.json to include technique names and captions.",
        "functions": {
            "read_moves_json": "Load and parse a moves.json file produced by the demo CLI. Returns a dict with keys like moves, board_image, etc.",
            "layout_grid": "Compute tiling layout based on paper size, DPI, margins, and number of columns/rows. Returns positions for each tile.",
            "main": "CLI entrypoint. Finds overlay images in a directory, reads optional moves.json, composes a tiled page, writes storyboard.png and storyboard.pdf to the output path.",
        },
    },
    "apps/cli/animate_gif.py": {
        "module": "Create an animated GIF from the sequence of overlay_move_XX.jpg frames.",
        "functions": {
            "main": "CLI entrypoint. Scans a directory for overlay_move_*.jpg frames, encodes them into an animated GIF at a modest framerate, and saves to --out."
        },
    },
    "apps/cli/animate_mp4.py": {
        "module": "Create an MP4 animation from overlay frames using imageio-ffmpeg if available (or save raw frames and a how-to file as a fallback).",
        "functions": {
            "main": "CLI entrypoint. Scans for overlay frames, writes an MP4 using FFmpeg via imageio; on failure, emits PNG frames and HOW_TO_FFMPEG.txt with an exact command to run."
        },
    },
    "vision/rectify/opencv_rectify.py": {
        "module": "OpenCV-based rectification: detect the Sudoku grid, perform perspective warp, optional line removal, and crop 81 cell tiles. This normalizes arbitrary phone photos to a clean board canvas.",
        "functions": {
            "process": "High-level rectification pipeline. Inputs: image path and options. Returns (board_image_path, cells) where cells is a list/array of 81 grayscale crops.",
            "find_grid_contour": "Detect the outermost grid contour using edge detection (Canny) + contour filtering (largest convex quadrilateral).",
            "perspective_warp": "Warp the detected quadrilateral to a square canvas (e.g., 900x900) using getPerspectiveTransform/warpPerspective.",
            "remove_grid_lines": "Optionally suppress grid lines via morphological ops and/or adaptive thresholding so digits stand out.",
            "crop_cells": "Cut the warped board into a 9x9 set of tiles (resize to 64x64). Returns list of 81 arrays or image paths.",
        },
    },
    "solver/sudoku_tools.py": {
        "module": "Human-style solving helpers: candidates calculation and next-move search over common techniques (singles, locked candidates), including auto follow-up (chaining). Also provides a tool-friendly interface for the demo CLI.",
        "functions": {
            "compute_candidates_tool": "Compute candidate digits for each empty cell in the current grid. Returns a dict like {'r1c2':[1,2,5], ...}.",
            "find_naked_singles": "Find cells whose candidate set has size 1. Returns a list of placement moves with metadata for visualization.",
            "find_hidden_singles": "Find digits that appear exactly once in a house (row/col/box). Returns placement moves.",
            "find_locked_candidates_pointing": "Locked Candidates (Pointing): if a digit's candidates in a box lie in a single row/col, eliminate that digit from the rest of that row/col outside the box. Returns elimination moves.",
            "chain_followup_singles": "After applying a move, recompute candidates and append any newly created Singles to the move list; repeat up to a cap to avoid runaway loops.",
            "next_moves": "Top-level technique dispatcher. In order: singles, locked candidates; applies chaining between steps. Returns at most max_moves moves with visualization-friendly fields.",
        },
    },
    "solver/solver_core.py": {
        "module": "Core Sudoku utilities used by higher-level techniques: index math, peers, house iterators, and grid mutation helpers.",
        "functions": {
            "rc_to_idx": "Convert (row, col) to a linear index and back. Keeps addressing consistent across modules.",
            "peers": "Return the set of peer coordinates for a given cell (same row, column, and 3x3 box).",
            "house_iter": "Iterate over cells of a given house (row/col/box). Yields coordinates or indices as needed.",
            "place_digit": "Apply a placement to the grid safely (with sanity checks) and return the updated grid.",
            "eliminate_candidate": "Remove a candidate digit from a cell's candidate set. Utility to keep elimination logic centralized.",
        },
    },
}

TRIQUOTE = '"""'


def ensure_module_docstring(text: str, module_doc: str) -> str:
    # If first non-empty non-comment line isn't a triple-quoted string, insert module docstring at top.
    lines = text.splitlines()
    i = 0
    # Skip shebang and encoding comments
    while i < len(lines) and (
        lines[i].strip().startswith("#!") or lines[i].strip().startswith("# -*-")
    ):
        i += 1
    # Skip blank lines and pure comments
    j = i
    while j < len(lines) and (lines[j].strip() == "" or lines[j].strip().startswith("#")):
        j += 1
    if j < len(lines) and lines[j].lstrip().startswith(TRIQUOTE):
        return text  # already has a module docstring
    # Insert
    module_block = f"{TRIQUOTE}{module_doc}{TRIQUOTE}\n\n"
    return "\n".join(lines[:i] + [module_block] + lines[i:])


def insert_function_docstrings(text: str, fn_docs: dict) -> tuple[str, list]:
    """Insert docstrings for functions listed in fn_docs where missing.
    Returns (new_text, applied:list[str]).
    """
    applied = []
    lines = text.splitlines()
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        # Match a simple def pattern at start of line or after indentation
        m = re.match(r"^(\s*)def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", line)
        if m:
            indent, name = m.group(1), m.group(2)
            if name in fn_docs:
                # Look ahead to see if a docstring exists as the next logical statement
                k = i + 1
                # skip blank lines
                while k < len(lines) and lines[k].strip() == "":
                    out.append(lines[k])
                    k += 1
                has_doc = False
                if k < len(lines):
                    nxt = lines[k].lstrip()
                    if nxt.startswith(TRIQUOTE) or nxt.startswith("r" + TRIQUOTE):
                        has_doc = True
                if not has_doc:
                    doc = fn_docs[name].strip().replace('"""', '\\"\\"\\"')
                    doc_block = indent + "    " + f"{TRIQUOTE}{doc}{TRIQUOTE}"
                    out.append(doc_block)
                    applied.append(name)
        i += 1
    return ("\n".join(out), applied)


def process_file(base: Path, rel: str, plan: dict, apply: bool) -> dict:
    p = base / rel
    report = {"file": rel, "exists": p.exists(), "inserted": [], "skipped": [], "module_doc": False}
    if not p.exists():
        return report
    text = p.read_text(encoding="utf-8")
    new_text = text
    # Module docstring
    if plan.get("module"):
        newer = ensure_module_docstring(new_text, plan["module"])
        if newer != new_text:
            new_text = newer
            report["module_doc"] = True
    # Function docstrings
    if plan.get("functions"):
        newer, applied = insert_function_docstrings(new_text, plan["functions"])
        if newer != new_text:
            new_text = newer
        # Track applied and skipped
        report["inserted"] = applied
        for fn in plan["functions"]:
            if fn not in applied:
                report["skipped"].append(fn)
    # Write back
    if apply and new_text != text:
        p.write_text(new_text, encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=".", help="Project root (default: current directory)")
    ap.add_argument(
        "--apply", action="store_true", help="Write changes to files (default is dry-run)"
    )
    ap.add_argument(
        "--report", default="docs/annotation_report.json", help="Where to write JSON report"
    )
    args = ap.parse_args()

    base = Path(args.base)
    results = []
    for rel, plan in TARGETS.items():
        results.append(process_file(base, rel, plan, apply=args.apply))

    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] Annotated {sum(1 for r in results if r['exists'])} existing files.")
    print(f"Report written to {out}")


if __name__ == "__main__":
    main()
