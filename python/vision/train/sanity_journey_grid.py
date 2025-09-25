# sanity_journey_grid.py
# Quick visual check for the Sudoku Journey generator (train_corners.synth_sudoku_journey).
# Saves mosaics under ./runs/journey_sanity by default.
#
# Examples:
#   python sanity_journey_grid.py
#   python sanity_journey_grid.py --also-d1to8 --scale 2 --outdir ./runs/journey_sanity
#   python sanity_journey_grid.py --debug-overlay --overlay-keys header,fill_mode,link_mode,distortion,config,footprint
#   python sanity_journey_grid.py --clean --no-title       # grids only (no dots/tags/title)

import os
import math
import random
import argparse
import numpy as np
import cv2

# Your updated train file must define synth_sudoku_journey.
# It MAY return (inp, hms, xy) or (inp, hms, xy, meta). We handle both.
from train_corners import synth_sudoku_journey

COLORS = [
    (0,255,0),    # TL
    (0,255,255),  # TR
    (255,255,0),  # BR
    (255,0,255),  # BL
]

def _poly_area(xy: np.ndarray) -> float:
    # xy: [4,2]
    x = xy[:,0]; y = xy[:,1]
    return 0.5*abs(np.dot(x, np.roll(y,-1)) - np.dot(y, np.roll(x,-1)))

def _quad_width_height(xy: np.ndarray) -> tuple[float,float]:
    # approximate width/height by averaging opposite side lengths
    def dist(a,b): return float(np.linalg.norm(a-b))
    tl,tr,br,bl = xy
    top = dist(tl,tr); bot = dist(bl,br)
    left = dist(tl,bl); right = dist(tr,br)
    w = 0.5*(top+bot); h = 0.5*(left+right)
    return max(w,1e-6), max(h,1e-6)

def _footprint_bucket(area_frac: float) -> str:
    # expected ranges: small: 0.15–0.30, medium: 0.30–0.60, large: 0.60–0.90
    if area_frac < 0.30: return "S"
    if area_frac < 0.60: return "M"
    return "L"

def _put_label(panel: np.ndarray, lines: list[str], scale: int = 1,
               color=(0,0,0), bg=(255,255,255)) -> None:
    # draw multi-line label with a simple background strip at the top-left
    h = 12*scale
    pad = 3*scale
    total_h = pad + len(lines)*(h+pad)
    max_w = 0
    for t in lines:
        (w, _), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.35*scale, 1)
        max_w = max(max_w, w)
    bg_w = max_w + 2*pad
    bg_h = total_h
    if bg_h <= 0 or bg_w <= 0:
        return
    cv2.rectangle(panel, (0,0), (bg_w, bg_h), bg, -1)
    y = pad + int(0.8*h)
    for t in lines:
        cv2.putText(panel, t, (pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35*scale, color, 1, cv2.LINE_AA)
        y += h + pad

def draw_sample_tile(inp: np.ndarray,
                     gt_xy: np.ndarray,
                     difficulty: int,
                     scale: int = 1,
                     show_tags: bool = True,
                     show_dots: bool = True,
                     overlay: dict | None = None) -> np.ndarray:
    """
    inp: [1,H,W] float32 0..1
    gt_xy: [4,2] or all-negative for negatives
    overlay: optional dict of extra strings to add (computed here and/or provided via meta)
    """
    g = (inp[0] * 255.0).astype(np.uint8)
    tile = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

    valid = (gt_xy >= 0).all()

    # draw dots only if requested
    if valid and show_dots:
        for i,(x,y) in enumerate(gt_xy):
            cv2.circle(tile, (int(round(x)), int(round(y))), 2, COLORS[i%4], -1, lineType=cv2.LINE_AA)

    # tags / overlay
    if show_tags or overlay:
        base = []
        if show_tags:
            base.append(f"D{difficulty}")
            if not valid:
                base.append("NEG")

        # geometry + meta overlay (if any)
        if (overlay or show_tags) and valid:
            H, W = g.shape
            area = _poly_area(gt_xy)
            area_frac = float(area) / float(W*H)
            w,h = _quad_width_height(gt_xy)
            ar = w / h
            base.append(f"{_footprint_bucket(area_frac)} {area_frac*100:.0f}%")
            base.append(f"AR {ar:.2f}")

        if overlay:
            for k, v in overlay.items():
                if isinstance(v, float):
                    base.append(f"{k}:{v:.2f}")
                else:
                    base.append(f"{k}:{v}")

        if base:
            _put_label(tile, base, scale=max(1,scale))

    if scale != 1:
        tile = cv2.resize(tile, (tile.shape[1]*scale, tile.shape[0]*scale), interpolation=cv2.INTER_NEAREST)
    return tile

def make_grid_image(rows: int,
                    cols: int,
                    difficulties,
                    seed: int = 1234,
                    img_size: int = 128,
                    sigma: float = 1.6,
                    neg_frac: float = 0.03,
                    occlusion_prob: float = 0.05,
                    scale: int = 1,
                    title: str | None = None,
                    debug_overlay: bool = False,
                    overlay_keys: list[str] | None = None,
                    show_tags: bool = True,
                    show_dots: bool = True) -> np.ndarray:
    """
    Build a mosaic grid preview.
    `difficulties` can be:
      - int  → single difficulty for all tiles
      - list → rotated across tiles or per row if len == rows
    """
    rng = random.Random(seed)
    tiles = []
    for r in range(rows):
        row_tiles = []
        for c in range(cols):
            # choose difficulty for this tile
            if isinstance(difficulties, int):
                d = difficulties
            elif isinstance(difficulties, (list, tuple)):
                if len(difficulties) == rows:
                    d = difficulties[r]
                else:
                    d = difficulties[(r*cols + c) % len(difficulties)]
            else:
                d = rng.randint(1, 8)

            # deterministic per-tile RNG for reproducibility
            local_rng = random.Random(seed*100003 + r*1009 + c*9173 + d*37)

            # Call generator, supporting optional meta
            meta = None
            try:
                out = synth_sudoku_journey(
                    local_rng,
                    img_size=img_size,
                    difficulty=d,
                    sigma=sigma,
                    neg_frac=neg_frac,
                    occlusion_prob=occlusion_prob,
                    return_meta=True  # if your generator supports this
                )
                if len(out) == 4:
                    inp, hms, xy, meta = out
                else:
                    inp, hms, xy = out
            except TypeError:
                # older signature without return_meta
                inp, hms, xy = synth_sudoku_journey(
                    local_rng,
                    img_size=img_size,
                    difficulty=d,
                    sigma=sigma,
                    neg_frac=neg_frac,
                    occlusion_prob=occlusion_prob
                )

            tile_overlay = None
            if debug_overlay:
                tile_overlay = {}
                if meta and isinstance(meta, dict):
                    keys = overlay_keys or []
                    for k in keys:
                        if k in meta:
                            v = meta[k]
                            tile_overlay[k] = v

            tile = draw_sample_tile(
                inp, xy, d, scale=scale,
                show_tags=show_tags,
                show_dots=show_dots,
                overlay=tile_overlay
            )
            row_tiles.append(tile)
        tiles.append(np.hstack(row_tiles))
    grid = np.vstack(tiles)

    if title:
        pad = 24 * scale
        canvas = np.full((grid.shape[0] + pad, grid.shape[1], 3), 255, np.uint8)
        canvas[pad:, :] = grid
        cv2.putText(canvas, title, (8*scale, int(16*scale)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5*scale, (0,0,0), 1, cv2.LINE_AA)
        grid = canvas
    return grid

def _parse_overlay_keys(s: str | None) -> list[str]:
    if not s:
        return []
    return [k.strip() for k in s.split(",") if k.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="./runs/journey_sanity",
                    help="Directory to save preview mosaics (default: ./runs/journey_sanity)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--scale", type=int, default=1, help="Integer upscaling for visibility in the mosaic")
    ap.add_argument("--sigma", type=float, default=1.6)
    ap.add_argument("--neg-frac", type=float, default=0.03)
    ap.add_argument("--occlusion-prob", type=float, default=0.05)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--also-d1to8", action="store_true", help="Also produce an 8x6 grid, one row per difficulty.")

    # Overlays and cleanliness toggles
    ap.add_argument("--debug-overlay", action="store_true",
                    help="Show per-tile overlay (footprint/AR + optional meta keys).")
    ap.add_argument("--overlay-keys", type=str, default="header,fill_mode,link_mode,distortion,config,footprint",
                    help="Comma-separated meta keys to request from the generator (if available).")
    ap.add_argument("--no-tags", action="store_true", help="Hide the D#/NEG tile labels.")
    ap.add_argument("--no-dots", action="store_true", help="Hide the colored corner dots.")
    ap.add_argument("--clean", action="store_true",
                    help="Shortcut: hide tags and dots, and force overlays off.")
    ap.add_argument("--no-title", action="store_true", help="Hide the mosaic title banner.")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Resolve toggles
    show_tags = not (args.no_tags or args.clean)
    show_dots = not (args.no_dots or args.clean)
    debug_overlay_effective = args.debug_overlay and not args.clean
    keys = _parse_overlay_keys(args.overlay_keys)

    # Titles can be hidden
    title1 = None if args.no_title else f"Sudoku Journey — {args.rows}x{args.cols} mixed (D1–D8)"

    # 6x6 mixed difficulties (uniform over 1..8)
    mix_diffs = [1,2,3,4,5,6,7,8]
    grid_mix = make_grid_image(
        rows=args.rows, cols=args.cols, difficulties=mix_diffs,
        seed=args.seed, img_size=args.img_size, sigma=args.sigma,
        neg_frac=args.neg_frac, occlusion_prob=args.occlusion_prob,
        scale=args.scale, title=title1,
        debug_overlay=debug_overlay_effective, overlay_keys=keys,
        show_tags=show_tags, show_dots=show_dots
    )
    p1 = os.path.join(args.outdir, "journey_grid_6x6.png")
    cv2.imwrite(p1, grid_mix)

    # Optional: 8 rows (D1..D8), 6 columns each
    if args.also_d1to8:
        per_row = [1,2,3,4,5,6,7,8]
        title2 = None if args.no_title else "Sudoku Journey — rows D1→D8 (6 per row)"
        grid_d = make_grid_image(
            rows=8, cols=max(6, args.cols), difficulties=per_row,
            seed=args.seed+999, img_size=args.img_size, sigma=args.sigma,
            neg_frac=args.neg_frac, occlusion_prob=args.occlusion_prob,
            scale=args.scale, title=title2,
            debug_overlay=debug_overlay_effective, overlay_keys=keys,
            show_tags=show_tags, show_dots=show_dots
        )
        p2 = os.path.join(args.outdir, "journey_grid_D1toD8.png")
        cv2.imwrite(p2, grid_d)
    else:
        p2 = None

    print("Saved:")
    print(f"  {p1}")
    if p2: print(f"  {p2}")

if __name__ == "__main__":
    main()