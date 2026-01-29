"""
Quick dataset visualizer for Sudoku grid data.

- Reads a JSONL manifest (synth or real).
- For each record, builds an overlay of label channels (A/H/V/J) on top of the image.
- Writes per-sample overlays and optional montages.

Outputs:
  debug/grid_intersection/overlays/<stem>_overlay.jpg     # per-sample
  debug/grid_intersection/overlays/montage_000.jpg        # N×M collage (optional)

Usage:
  python tools/grid_intersection/visualize_dataset.py \
    --manifest datasets/grids/synth/manifests/val_synth.jsonl \
    --outdir   debug/grid_intersection/overlays \
    --limit 64 --montage_rows 4 --montage_cols 4
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List, Dict
import numpy as np
import cv2

COL_A = (0, 255, 255)   # yellow
COL_H = (255, 128, 0)   # orange
COL_V = (0, 200, 255)   # cyan
COL_J = (255, 0, 255)   # magenta

def read_jsonl(p: str) -> List[Dict]:
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def heat_overlay(gray: np.ndarray, A=None, H=None, V=None, J=None) -> np.ndarray:
    """
    Create a colored overlay showing any available channels.
    Inputs can be float [0,1], uint8 [0,255], or None.
    """
    if gray.ndim == 2:
        base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        base = gray.copy()

    def norm255(x):
        if x is None: return None
        xf = x.astype(np.float32)
        if xf.max() <= 1.5: xf *= 255.0
        return np.clip(xf, 0, 255).astype(np.uint8)

    A = norm255(A); H = norm255(H); V = norm255(V); J = norm255(J)

    # Blend each channel with a solid color mask
    out = base
    for chan, col in [(A, COL_A), (H, COL_H), (V, COL_V), (J, COL_J)]:
        if chan is None: continue
        color = np.zeros_like(out)
        color[:] = col
        alpha = (chan / 255.0)[..., None]  # HxWx1
        out = cv2.convertScaleAbs(out * (1 - 0.40 * alpha) + color * (0.40 * alpha))
    return out

def put_sticker(img: np.ndarray, text: str, color=(0,0,0)) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (6, 6), (6+210, 6+28), (255,255,255), -1)
    cv2.putText(out, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return out

def make_montage(images: List[np.ndarray], rows: int, cols: int, pad: int=8, bg=220) -> np.ndarray:
    assert len(images) <= rows*cols
    if not images: return None
    h, w = images[0].shape[:2]
    canvas = np.full((rows*h + (rows+1)*pad, cols*w + (cols+1)*pad, 3), bg, np.uint8)
    k = 0
    for r in range(rows):
        for c in range(cols):
            if k >= len(images): break
            y = pad + r*(h+pad); x = pad + c*(w+pad)
            tile = images[k]
            if tile.ndim == 2: tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
            canvas[y:y+h, x:x+w] = tile
            k += 1
    return canvas

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--resize", type=int, default=768, help="Resize short side to this for visualization")
    ap.add_argument("--limit", type=int, default=64)
    ap.add_argument("--montage_rows", type=int, default=0)
    ap.add_argument("--montage_cols", type=int, default=0)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    recs = read_jsonl(args.manifest)[:args.limit]
    thumbs = []

    for r in recs:
        img = cv2.imread(r["image_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[skip] unreadable {r['image_path']}")
            continue

        # Optional labels
        A=H=V=J=None
        lp = r.get("label_path", None)
        if lp and Path(lp).exists():
            lab = np.load(lp)
            A = lab.get("A"); H = lab.get("H"); V = lab.get("V"); J = lab.get("J")

        # Resize proportionally (short side = args.resize)
        h, w = img.shape[:2]
        scale = float(args.resize) / min(h, w)
        new_w = int(round(w * scale)); new_h = int(round(h * scale))
        img_r = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        A_r = cv2.resize(A, (new_w, new_h)) if A is not None else None
        H_r = cv2.resize(H, (new_w, new_h)) if H is not None else None
        V_r = cv2.resize(V, (new_w, new_h)) if V is not None else None
        J_r = cv2.resize(J, (new_w, new_h)) if J is not None else None

        ov = heat_overlay(img_r, A_r, H_r, V_r, J_r)
        stem = Path(r["image_path"]).stem
        ov = put_sticker(ov, f"{stem}", (10,10,10))
        cv2.imwrite(str(outdir / f"{stem}_overlay.jpg"), ov)

        # Square thumbnail for montage
        side = min(512, min(ov.shape[:2]))
        th = cv2.resize(ov, (side, side), interpolation=cv2.INTER_AREA)
        thumbs.append(th)

    # Montage (optional)
    if args.montage_rows > 0 and args.montage_cols > 0:
        thumbs = thumbs[:args.montage_rows * args.montage_cols]
        if thumbs:
            grid = make_montage(thumbs, args.montage_rows, args.montage_cols)
            if grid is not None:
                cv2.imwrite(str(outdir / "montage_000.jpg"), grid)
                print(f"[visualize_dataset] wrote montage → {outdir/'montage_000.jpg'}")

    print(f"[visualize_dataset] wrote overlays → {outdir}")

if __name__ == "__main__":
    main()