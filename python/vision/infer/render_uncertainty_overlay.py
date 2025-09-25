
# vision/infer/render_uncertainty_overlay.py
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List
from PIL import Image, ImageDraw, ImageFont

def draw_overlay(grid, probs, top2, prob2, margin, low_conf, meta: dict, out_path: Path,
                 cell:int=64, pad:int=4, show_alt:bool=True):
    """Draw a 9x9 board with per-cell prob and low_conf highlighting."""
    W = 9*cell + 8*pad
    H = W + 40  # extra footer for meta
    im = Image.new("RGB", (W, H), (255,255,255))
    draw = ImageDraw.Draw(im)

    # fonts
    font_big = font_small = font_footer = None
    for cand in ["arial.ttf", "DejaVuSans.ttf"]:
        try:
            font_big = ImageFont.truetype(cand, 30)
            font_small = ImageFont.truetype(cand, 14)
            font_footer = ImageFont.truetype(cand, 13)
            break
        except Exception:
            continue
    if font_big is None:
        font_big = ImageFont.load_default()
        font_small = ImageFont.load_default()
        font_footer = ImageFont.load_default()

    # cells
    for r in range(9):
        for c in range(9):
            x = c*(cell+pad)
            y = r*(cell+pad)
            low = bool(low_conf[r][c])
            # bg highlight
            if low:
                draw.rectangle([x,y,x+cell,y+cell], fill=(255,230,230))
            # border
            draw.rectangle([x,y,x+cell,y+cell], outline=(255,0,0) if low else (0,0,0), width=2 if low else 1)

            # big digit
            d = str(grid[r][c])
            col = (200,0,0) if low else (0,0,0)
            bbox = draw.textbbox((0,0), d, font=font_big)
            tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
            draw.text((x + (cell-tw)//2, y + 6 + (cell-th)//2), d, fill=col, font=font_big)

            # prob in bottom-right
            ptxt = f"{probs[r][c]:.2f}"
            bbox2 = draw.textbbox((0,0), ptxt, font=font_small)
            draw.text((x + cell - bbox2[2] - 4, y + cell - bbox2[3] - 2), ptxt, fill=col, font=font_small)

            # optional alt
            if show_alt:
                alt_txt = f"alt {top2[r][c]}:{prob2[r][c]:.2f}"
                draw.text((x + 4, y + 2), alt_txt, fill=(80,80,80), font=font_small)

    # footer meta
    footer = f"T={meta.get('temperature',1.0):.3f}  thresholds: low={meta.get('thresholds',{}).get('low')}  margin={meta.get('thresholds',{}).get('margin')}  model={Path(meta.get('model','')).name}"
    draw.text((4, W + 8), footer, fill=(0,0,0), font=font_footer)

    im.save(out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="JSON from classify_cells_model.py")
    ap.add_argument("--out", required=True, help="PNG path to write")
    ap.add_argument("--cell", type=int, default=64)
    ap.add_argument("--pad", type=int, default=4)
    ap.add_argument("--no-alt", action="store_true", help="Hide top-2 info in each cell")
    args = ap.parse_args()

    obj = json.loads(Path(args.json).read_text(encoding="utf-8"))
    grid = obj["grid"]; probs = obj["probs"]; top2 = obj["top2"]; prob2 = obj["prob2"]
    margin = obj["margin"]; low_conf = obj.get("low_conf") or [[0]*9 for _ in range(9)]
    meta = obj.get("meta", {})

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    draw_overlay(grid, probs, top2, prob2, margin, low_conf, meta, out_path, cell=args.cell, pad=args.pad, show_alt=not args.no_alt)
    print("Wrote:", out_path)

if __name__ == "__main__":
    main()
