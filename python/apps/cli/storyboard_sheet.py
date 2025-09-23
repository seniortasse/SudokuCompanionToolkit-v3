from __future__ import annotations

from typing import Any

"""Build a printable storyboard sheet (PNG/PDF) that tiles overlay images with captions. Optionally reads moves.json to include technique names and captions."""


# storyboard_sheet.py
# Build a combined storyboard sheet (PNG + PDF) from per-move overlay images.
# Usage:
#   python storyboard_sheet.py --dir demo_export --out demo_export/storyboard --paper letter --cols 2 --title "Puzzle L-2-231"
#
# If you pass --json <path>, the script will use it to caption each move (technique, placement/elimination).
# Otherwise, filenames will be used as captions.

import argparse
import glob
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

PAPERS = {"letter": (2550, 3300), "a4": (2480, 3508)}  # 8.5x11 @ 300dpi  # 210x297mm @ 300dpi


def load_font(size):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except:
        return ImageFont.load_default()


def caption_for_move(move, idx):
    tech = move.get("technique", "?")
    if move.get("type", "placement") == "placement":
        return f"#{idx}  {tech}: {move.get('cell','')} = {move.get('digit','')}"
    else:
        elims = len(move.get("eliminate", []))
        return f"#{idx}  {tech}: eliminate {move.get('digit','')} from {elims} cell(s)"


def gather_overlays(dir_path):
    files = sorted(glob.glob(str(Path(dir_path) / "overlay_move_*.jpg")))
    return files


def read_moves_json(json_path: str) -> dict[str, Any]:
    """Load and parse a moves.json file produced by the demo CLI. Returns a dict with keys like moves, board_image, etc."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    # Accept either the demo_cli_overlay output or a bare {"moves":[...]} file
    if "moves" in data:
        return data["moves"]
    return data


def build_storyboard(
    overlay_paths, out_prefix, paper="letter", cols=2, margin=80, gutter=40, title=None, moves=None
):
    W, H = PAPERS[paper]
    page = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(page)

    # Title
    y = margin
    if title:
        f = load_font(64)
        tw, th = draw.textbbox((0, 0), title, font=f)[2:]
        draw.text(((W - tw) // 2, y), title, fill=(0, 0, 0), font=f)
        y += th + margin // 2

    # Grid layout
    rows = math.ceil(len(overlay_paths) / cols) if overlay_paths else 0
    if rows == 0:
        # nothing to draw
        page.save(out_prefix + ".png")
        page.save(out_prefix + ".pdf")
        return out_prefix + ".png", out_prefix + ".pdf"

    avail_h = H - y - margin
    avail_w = W - 2 * margin
    cell_w = (avail_w - (cols - 1) * gutter) // cols
    # reserve caption area under each tile
    cap_h = 60
    cell_h = ((avail_h - (rows - 1) * gutter) // rows) - cap_h
    thumb_w = thumb_h = min(cell_w, cell_h)

    fcap = load_font(28)

    i = 0
    for r in range(rows):
        x = margin
        for c in range(cols):
            if i >= len(overlay_paths):
                break
            img = Image.open(overlay_paths[i]).convert("RGB")
            img = ImageOps.fit(img, (thumb_w, thumb_h), method=Image.BICUBIC)
            page.paste(img, (x, y))
            # Caption
            cap = None
            if moves and i < len(moves):
                cap = caption_for_move(moves[i], i + 1)
            else:
                cap = f"#{i+1}  {Path(overlay_paths[i]).name}"
            draw.text((x, y + thumb_h + 6), cap, fill=(0, 0, 0), font=fcap)
            x += cell_w + gutter
            i += 1
        y += thumb_h + cap_h + gutter

    # Save
    png_path = out_prefix + ".png"
    pdf_path = out_prefix + ".pdf"
    page.save(png_path)
    page.save(pdf_path)
    return png_path, pdf_path


def main() -> None:
    """CLI entrypoint. Finds overlay images in a directory, reads optional moves.json, composes a tiled page, writes storyboard.png and storyboard.pdf to the output path."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default="demo_export")
    ap.add_argument("--out", type=str, default="demo_export/storyboard")
    ap.add_argument("--paper", type=str, default="letter", choices=["letter", "a4"])
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--title", type=str, default="Sudoku Companion — Move Storyboard")
    ap.add_argument("--json", type=str, default="")  # optional moves JSON
    args = ap.parse_args()

    overlays = gather_overlays(args.dir)
    moves = read_moves_json(args.json) if args.json else None
    png, pdf = build_storyboard(
        overlays, args.out, paper=args.paper, cols=args.cols, title=args.title, moves=moves
    )
    print(json.dumps({"png": png, "pdf": pdf, "count": len(overlays)}, indent=2))


if __name__ == "__main__":
    main()
