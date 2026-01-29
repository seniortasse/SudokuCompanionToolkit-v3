# tools/grid_intersection/overlay_preds.py
# ------------------------------------------------------------
# Visualization helpers:
#  - Heatmap overlays for A/H/V
#  - Junction peaks + lattice ordering indices
#  - Polylines (10 H + 10 V)
#  - Optional: sample 4 cells with Coons meshes to verify "no eating"
#
# This script expects model outputs and postproc utilities to be available.
# You can adapt the loader to your environment (npz, torch output, etc.).
# ------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import numpy as np
import cv2
import json
import random

# Import your postproc API
from python.vision.train.grid_intersection.postproc import (
    PPConfig, bin_maps, junction_nms, cluster_grid, assemble_polylines,
    estimate_halfwidth, offset_inward, coons_patch_grid
)

@dataclass
class OverlayConfig:
    thr_strong: float = 0.5
    thr_weak: float = 0.3
    j_nms_radius: int = 3
    j_topk: int = 120
    resample_K: int = 64
    inward_margin_min: int = 2
    halfwidth_cap_px: int = 5
    font: int = cv2.FONT_HERSHEY_SIMPLEX

def _cmap_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def overlay_heatmap(base: np.ndarray, prob: np.ndarray, alpha: float=0.5, channel: str="A") -> np.ndarray:
    base3 = _cmap_gray(base.copy())
    pf = prob.astype(np.float32)
    if prob.dtype == np.uint8:
        pf = pf / 255.0
    hm = (pf * 255).clip(0,255).astype(np.uint8)
    hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    out = cv2.addWeighted(base3, 1.0, hm, alpha, 0)
    cv2.putText(out, f"{channel}", (12,24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    return out

def draw_junctions(img: np.ndarray, jpts: List[Tuple[float,float,float]], color=(0,255,255)) -> np.ndarray:
    out = img.copy()
    for k,(x,y,s) in enumerate(jpts):
        cv2.circle(out, (int(round(x)), int(round(y))), 3, color, -1, cv2.LINE_AA)
    return out

def draw_lattice_indices(img: np.ndarray, ylv: np.ndarray, xlv: np.ndarray,
                         grid: List[List[Tuple[float,float]]]) -> np.ndarray:
    out = img.copy()
    # draw horizontal level indices
    for i,y in enumerate(ylv):
        cv2.putText(out, f"r{i}", (5, int(round(y))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,255), 1, cv2.LINE_AA)
    for j,x in enumerate(xlv):
        cv2.putText(out, f"c{j}", (int(round(x)), 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,200), 1, cv2.LINE_AA)
    # draw grid points
    for i in range(10):
        for j in range(10):
            x,y = grid[i][j]
            cv2.circle(out, (int(round(x)), int(round(y))), 2, (255,255,0), -1, cv2.LINE_AA)
    return out

def draw_polylines(img: np.ndarray, H_lines: List[np.ndarray], V_lines: List[np.ndarray]) -> np.ndarray:
    out = _cmap_gray(img.copy())
    # color palette
    colorsH = [(255, 120, 120), (255,160,120), (255,200,120), (255,240,120),
               (220,255,120), (180,255,120), (140,255,120), (120,255,180),
               (120,255,220), (120,255,255)]
    colorsV = [(120,120,255), (120,160,255), (120,200,255), (120,240,255),
               (120,255,220), (120,255,180), (120,255,140), (180,255,120),
               (220,255,120), (255,255,120)]
    for i,pl in enumerate(H_lines):
        for a,b in zip(pl[:-1], pl[1:]):
            cv2.line(out, (int(round(a[0])), int(round(a[1]))),
                          (int(round(b[0])), int(round(b[1]))),
                          colorsH[i%len(colorsH)], 2, cv2.LINE_AA)
    for j,pl in enumerate(V_lines):
        for a,b in zip(pl[:-1], pl[1:]):
            cv2.line(out, (int(round(a[0])), int(round(a[1]))),
                          (int(round(b[0])), int(round(b[1]))),
                          colorsV[j%len(colorsV)], 2, cv2.LINE_AA)
    return out

def sample_cell_meshes(img: np.ndarray,
                       H_lines: List[np.ndarray],
                       V_lines: List[np.ndarray],
                       n_samples: int=4,
                       tile_size: int=96) -> np.ndarray:
    """
    Pick up to n_samples random cells and draw a coarse Coons mesh on top.
    """
    out = _cmap_gray(img.copy())
    Hn, Vn = len(H_lines), len(V_lines)
    cand = [(i,j) for i in range(Hn-1) for j in range(Vn-1)]
    random.shuffle(cand)
    cand = cand[:n_samples]
    for (i,j) in cand:
        # corners:
        # (i,j) (i,j+1)
        # (i+1,j) (i+1,j+1)
        # Extract boundaries
        # segment slicing by index range
        def seg_between(poly, p, q):
            # naive: pick closest points and take slice
            P = poly
            d = np.sqrt(((P - p)**2).sum(axis=1))
            a = int(np.argmin(d))
            d2 = np.sqrt(((P - q)**2).sum(axis=1))
            b = int(np.argmin(d2))
            if a <= b:
                return P[a:b+1]
            else:
                return P[b:a+1][::-1]

        p00 = V_lines[j][0]   # rough; we’ll find nearest in seg_between
        p10 = V_lines[j+1][-1]

        # Using lattice logic from polylines: use nearest points by coordinates
        top    = seg_between(H_lines[i],     V_lines[j][0],   V_lines[j+1][0])
        bottom = seg_between(H_lines[i+1],   V_lines[j][-1],  V_lines[j+1][-1])
        left   = seg_between(V_lines[j],     H_lines[i][0],   H_lines[i+1][0])
        right  = seg_between(V_lines[j+1],   H_lines[i][-1],  H_lines[i+1][-1])

        grid = coons_patch_grid(top, bottom, left, right, out_size=tile_size)
        # draw coarse mesh (every 12 px)
        step = max(12, tile_size//8)
        for u in range(0, tile_size, step):
            for v in range(0, tile_size, step):
                if u+step<tile_size:
                    a = grid[v, u]; b = grid[v, u+step]
                    cv2.line(out, (int(a[0]),int(a[1])), (int(b[0]),int(b[1])), (0,255,255), 1, cv2.LINE_AA)
                if v+step<tile_size:
                    a = grid[v, u]; b = grid[v+step, u]
                    cv2.line(out, (int(a[0]),int(a[1])), (int(b[0]),int(b[1])), (0,255,255), 1, cv2.LINE_AA)
    return out

# ------------------------------ Driver --------------------------------

def make_overlays(image: np.ndarray,
                  A: np.ndarray, H: np.ndarray, V: np.ndarray, J: np.ndarray, O: np.ndarray,
                  pp_cfg: PPConfig = PPConfig(),
                  n_cell_meshes: int = 4) -> dict:
    """
    Returns a dict of BGR images:
      - heat_A, heat_H, heat_V
      - juncs
      - lattice
      - polylines
      - cells_mesh
    """
    # Heatmaps
    out = {}
    out["heat_A"] = overlay_heatmap(image, A, 0.5, "A")
    out["heat_H"] = overlay_heatmap(image, H, 0.5, "H")
    out["heat_V"] = overlay_heatmap(image, V, 0.5, "V")

    # Junctions + lattice
    A_bin, H_bin, V_bin = bin_maps(A, H, V, pp_cfg)
    jpts = junction_nms(J, pp_cfg)
    img_j = draw_junctions(_cmap_gray(image), jpts, (0,255,255))

    ylv, xlv, grid = cluster_grid(jpts, pp_cfg, image.shape[0], image.shape[1])
    img_lat = draw_lattice_indices(img_j, ylv, xlv, grid)

    # Polylines
    A01 = A.astype(np.float32) / (255.0 if A.dtype==np.uint8 else 1.0)
    H_lines, V_lines = assemble_polylines(grid, A01, O, pp_cfg)
    img_poly = draw_polylines(image, H_lines, V_lines)

    # Cell meshes
    img_cells = sample_cell_meshes(img_poly, H_lines, V_lines, n_samples=n_cell_meshes)

    return {
        "heat_A": out["heat_A"],
        "heat_H": out["heat_H"],
        "heat_V": out["heat_V"],
        "juncs": img_j,
        "lattice": img_lat,
        "polylines": img_poly,
        "cells_mesh": img_cells
    }

# ------------------------------- CLI ----------------------------------

if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True, help="Path to a .npz containing A,H,V,J,O and an image .jpg alongside.")
    parser.add_argument("--image", required=False, help="Optional explicit image path; if omitted, swaps .npz->.jpg")
    parser.add_argument("--outdir", required=True, help="Dir to write overlay jpgs.")
    args = parser.parse_args()

    npz_path = Path(args.npz)
    img_path = Path(args.image) if args.image else npz_path.with_suffix(".jpg")
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    data = np.load(str(npz_path))
    # Expect A,H,V,J,O in the npz (same keys as synth_renderer)
    A = data["A"]; H = data["H"]; V = data["V"]; J = data["J"]; O = data["O"]

    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {img_path}")

    pp_cfg = PPConfig()
    overlays = make_overlays(img, A,H,V,J,O, pp_cfg, n_cell_meshes=4)

    for k,im in overlays.items():
        cv2.imwrite(str(outdir / f"{npz_path.stem}_{k}.jpg"), im)
    print(f"[overlay_preds] wrote overlays to {outdir}")