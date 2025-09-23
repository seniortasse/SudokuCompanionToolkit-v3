# opencv_rectify.py — robust Sudoku rectifier:
# - auto deskew
# - oriented masks with axis-aware gap-bridging
# - collinear-component grouping (repairs broken lines)
# - grid-aware 8×8 internal-line selection (progressive gates + centrality window)
# - orientation-aware border fit
# - ROI-clipped 10×10 lattice

from __future__ import annotations
from typing import Any, List, Tuple
from pathlib import Path
import json
import numpy as np
import cv2

# ───────── helpers ─────────
def _ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)
def _to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
def _save(p: Path, im: np.ndarray): cv2.imwrite(str(p), im)
def _rotate_image(img: np.ndarray, angle_deg: float) -> np.ndarray:
    H, W = img.shape[:2]
    M = cv2.getRotationMatrix2D((W/2.0, H/2.0), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

# ───────── oriented masks ─────────
def build_oriented_masks(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    H, W = gray.shape[:2]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    eq = clahe.apply(gray)

    block = max(21, (min(H, W)//32) | 1)  # ~3% of min side
    bw = cv2.adaptiveThreshold(eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, blockSize=block, C=6)

    approx_cell = min(H, W)/9.0

    # tilt-tolerant: erode with long bar, then dilate wider
    L = max(15, int(0.85*approx_cell))
    k_h1 = cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1))
    k_hd = cv2.getStructuringElement(cv2.MORPH_RECT, (L+2, 3))
    k_v1 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, L))
    k_vd = cv2.getStructuringElement(cv2.MORPH_RECT, (3, L+2))

    mask_h = cv2.dilate(cv2.erode(bw, k_h1, 1), k_hd, 1)
    mask_v = cv2.dilate(cv2.erode(bw, k_v1, 1), k_vd, 1)

    # heal tiny gaps (NEW: axis-aware thin closing)
    gap = max(5, int(0.28*approx_cell))
    mask_h = cv2.morphologyEx(mask_h, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (gap, 1)), 1)
    mask_v = cv2.morphologyEx(mask_v, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, gap)), 1)

    # light final close to smooth edges
    mask_h = cv2.morphologyEx(mask_h, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), 1)
    mask_v = cv2.morphologyEx(mask_v, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), 1)

    return mask_h, mask_v, cv2.bitwise_or(mask_h, mask_v)

# ───────── ROI detection ─────────
def sudoku_roi_from_masks(mask_h: np.ndarray, mask_v: np.ndarray):
    H, W = mask_h.shape[:2]
    fused = cv2.bitwise_or(mask_h, mask_v)
    d = max(7, int(0.06*min(H, W)))
    fused = cv2.dilate(fused, cv2.getStructuringElement(cv2.MORPH_RECT, (d, d)), 1)

    num, lab = cv2.connectedComponents((fused > 0).astype(np.uint8))
    if num <= 1:
        y0, y1, x0, x1 = 0, H, 0, W
    else:
        bestA, best = -1, (0, H, 0, W)
        for t in range(1, num):
            ys, xs = np.where(lab == t)
            if xs.size == 0: continue
            x0c, x1c = int(xs.min()), int(xs.max())
            y0c, y1c = int(ys.min()), int(ys.max())
            A = (x1c-x0c+1)*(y1c-y0c+1)
            if A > bestA:
                bestA, best = A, (y0c, y1c, x0c, x1c)
        y0, y1, x0, x1 = best

    return (mask_h[y0:y1, x0:x1], mask_v[y0:y1, x0:x1]), (y0, y1, x0, x1)

# ───────── components & utilities ─────────
def _components_from_mask(mask: np.ndarray):
    num, lab = cv2.connectedComponents((mask > 0).astype(np.uint8))
    comps = []
    for k in range(1, num):
        ys, xs = np.where(lab == k)
        if xs.size == 0: continue
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        comps.append((k, (x0, y0, x1, y1)))
    return comps, lab

def _extract_mask_for_component(lab: np.ndarray, label_id: int) -> np.ndarray:
    return (lab == label_id).astype(np.uint8) * 255

def _line_centers(line_masks: List[np.ndarray], axis: str) -> np.ndarray:
    coords = []
    for m in line_masks:
        ys, xs = np.where(m > 0)
        coord = float(np.median(ys)) if axis == 'h' else float(np.median(xs)) if xs.size else 0.0
        coords.append(coord)
    return np.array(sorted(coords))

def _cv_gaps(coords: np.ndarray) -> float:
    if len(coords) < 2: return 1e6
    gaps = np.diff(coords); mu = float(np.mean(gaps))
    return 1e6 if mu <= 1e-6 else float(np.std(gaps)/mu)

# ───────── crossings (central band) ─────────
def _count_crossings_central_band(
    line_mask: np.ndarray,
    other_mask: np.ndarray,
    axis: str,
    dilate_px: int,
    ignore_margin_px: int,
    band_frac: float,
) -> int:
    H, W = line_mask.shape[:2]
    band = np.zeros_like(line_mask)
    if axis == 'h':
        y0 = int(band_frac*H); y1 = int((1.0-band_frac)*H)
        band[y0:y1, :] = 255
    else:
        x0 = int(band_frac*W); x1 = int((1.0-band_frac)*W)
        band[:, x0:x1] = 255
    cand = cv2.bitwise_and(line_mask, band)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilate_px+1, 2*dilate_px+1))
    a = cv2.dilate(cand, k, 1)
    b = cv2.dilate(other_mask, k, 1)
    inter = cv2.bitwise_and(a, b)

    if ignore_margin_px > 0:
        if axis == 'h':
            inter[:ignore_margin_px,:] = 0
            inter[H-ignore_margin_px:,:] = 0
        else:
            inter[:, :ignore_margin_px] = 0
            inter[:, W-ignore_margin_px:] = 0

    num, _ = cv2.connectedComponents((inter > 0).astype(np.uint8))
    return max(0, num - 1)

# ───────── NEW: collinear grouping ─────────
def _group_collinear_components(mask: np.ndarray, axis: str, approx_cell: float):
    """
    Merge components that belong to the same line (similar x for 'v' or y for 'h').
    Returns a list of tuples: (merged_mask, bbox, coord)
    """
    comps, lab = _components_from_mask(mask)
    if not comps:
        return []

    # prepare sortable entries by axis coordinate
    entries = []
    for k, (x0, y0, x1, y1) in comps:
        coord = 0.5*(x0+x1) if axis == 'v' else 0.5*(y0+y1)
        entries.append((coord, k, (x0, y0, x1, y1)))
    entries.sort(key=lambda t: t[0])

    tol = 0.18*approx_cell  # must be well below cell size; avoids merging adjacent internals
    groups: List[List[Tuple[int, Tuple[int,int,int,int], float]]] = []
    curr: List[Tuple[int, Tuple[int,int,int,int], float]] = []
    last_c = None
    for c, k, b in entries:
        if last_c is None or abs(c - last_c) <= tol:
            curr.append((k, b, c))
        else:
            groups.append(curr); curr = [(k, b, c)]
        last_c = c
    if curr: groups.append(curr)

    merged = []
    for g in groups:
        # build mask union for this group
        gm = np.zeros_like(mask, np.uint8)
        xs0, ys0, xs1, ys1 = [], [], [], []
        coords = []
        for k, (x0, y0, x1, y1), c in g:
            gm[lab == k] = 255
            xs0.append(x0); xs1.append(x1); ys0.append(y0); ys1.append(y1)
            coords.append(c)
        bbox = (min(xs0), min(ys0), max(xs1), max(ys1))
        merged.append((gm, bbox, float(np.median(coords))))
    return merged

# ───────── select 8 internal lines (robust) ─────────
def _select_8_lines(mask: np.ndarray, axis: str, approx_cell: float, other_mask: np.ndarray) -> List[np.ndarray]:
    """
    Progressive gating with central-band crossing test + centrality-aware window,
    operating on collinearly merged components. Always returns 8 if possible.
    """
    H, W = mask.shape[:2]
    merged = _group_collinear_components(mask, axis, approx_cell)

    # span/thickness relax schedule
    span_thick = [
        (0.55, 0.33),
        (0.45, 0.45),
        (0.35, 0.60),
        (0.30, 0.80),
        (0.25, 0.90),
    ]
    # crossing-gate schedule (center margin, band, ignore strip, min crossings)
    gates = [
        (0.50, 0.18, 0.40, 6),
        (0.45, 0.16, 0.35, 5),
        (0.40, 0.14, 0.30, 5),
        (0.38, 0.12, 0.25, 4),
    ]

    def choose_window(cands):
        """Pick 8 contiguous lines with best spacing *and* farthest from extremes."""
        cands.sort(key=lambda t: t[0])
        coords_all = [t[0] for t in cands]
        minc, maxc = coords_all[0], coords_all[-1]
        mean_gap_all = float(np.mean(np.diff(coords_all))) if len(coords_all) > 1 else 1.0
        best_i, best_score = 0, 1e9
        for i in range(0, len(cands)-7):
            coords = coords_all[i:i+8]
            gaps = np.diff(coords); mu = float(np.mean(gaps))
            cv = (float(np.std(gaps))/mu) if mu > 1e-6 else 1e6
            left_margin  = coords[0] - minc
            right_margin = maxc - coords[-1]
            margin = min(left_margin, right_margin) / (mean_gap_all + 1e-6)
            score = cv - 0.15*margin
            if score < best_score: best_i, best_score = i, score
        return [cands[j][1] for j in range(best_i, best_i+8)]

    for min_span_frac, thick_frac in span_thick:
        max_thick_px = max(5, int(thick_frac*approx_cell))
        def ok(b):
            x0,y0,x1,y1 = b; w,h = (x1-x0+1), (y1-y0+1)
            return (w >= min_span_frac*W and h <= max_thick_px) if axis == 'h' else (h >= min_span_frac*H and w <= max_thick_px)
        base = [(m,b,c) for (m,b,c) in merged if ok(b)]
        if len(base) < 8: continue

        for center_margin_frac, band_frac, ignore_frac, crossings_min in gates:
            center_margin = center_margin_frac*approx_cell
            ignore_px = int(ignore_frac*approx_cell)
            cands = []
            for (cmask, b, coord) in base:
                # interior-only gate relative to ROI edges
                if axis == 'h':
                    if coord < center_margin or (H - coord) < center_margin: continue
                else:
                    if coord < center_margin or (W - coord) < center_margin: continue
                crossings = _count_crossings_central_band(
                    cmask, other_mask, axis=axis, dilate_px=2,
                    ignore_margin_px=ignore_px, band_frac=band_frac
                )
                if crossings >= crossings_min:
                    cands.append((coord, cmask))
            if len(cands) >= 8:
                return choose_window(cands)

        # fallback on this span/thickness: ignore crossings, just pick most uniform central window
        cands = [(coord, cmask) for (cmask, b, coord) in base]
        if len(cands) >= 8:
            return choose_window(cands)

    # last resort across all merged groups: 8 most central by coordinate
    if len(merged) >= 8:
        center = H/2.0 if axis=='h' else W/2.0
        merged.sort(key=lambda t: abs(center - t[2]))
        return [merged[i][0] for i in range(8)]

    raise RuntimeError(f"Not enough components to select 8 {axis}-lines (have {len(merged)} after grouping).")

# ───────── endpoints & intersections ─────────
def _endpoints_from_mask(line_mask: np.ndarray, axis: str):
    ys, xs = np.where(line_mask > 0)
    if xs.size == 0:
        return np.array([0,0], np.float32), np.array([0,0], np.float32)
    if axis == 'h':
        xl, xr = xs.min(), xs.max()
        yl = float(np.median(ys[xs <= xl+2])) if (xs <= xl+2).any() else float(ys.min())
        yr = float(np.median(ys[xs >= xr-2])) if (xs >= xr-2).any() else float(ys.max())
        return np.array([xl, yl], np.float32), np.array([xr, yr], np.float32)
    else:
        yt, yb = ys.min(), ys.max()
        xt = float(np.median(xs[ys <= yt+2])) if (ys <= yt+2).any() else float(xs.min())
        xb = float(np.median(xs[ys >= yb-2])) if (ys >= yb-2).any() else float(xs.max())
        return np.array([xt, yt], np.float32), np.array([xb, yb], np.float32)

def _intersections_from_masks(h_masks: List[np.ndarray], v_masks: List[np.ndarray], dilate_px: int = 3) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilate_px+1, 2*dilate_px+1))
    v_d = [cv2.dilate(m, k, 1) for m in v_masks]
    pts = np.zeros((8,8,2), np.float32)
    for i, hm in enumerate(h_masks):
        h_d = cv2.dilate(hm, k, 1)
        for j, vm in enumerate(v_d):
            inter = cv2.bitwise_and(h_d, vm)
            ys, xs = np.where(inter > 0)
            if xs.size:
                pts[i, j] = [float(np.median(xs)), float(np.median(ys))]
            else:
                dt = cv2.distanceTransform(cv2.bitwise_not(inter), cv2.DIST_L2, 3)
                yx = np.unravel_index(np.argmax(dt), dt.shape)
                pts[i, j] = [float(yx[1]), float(yx[0])]
    return pts

# ───────── orientation-aware fits ─────────
def _fit_h(points: np.ndarray) -> Tuple[float,float]:  # y = a*x + b
    x = points[:,0].astype(np.float64); y = points[:,1].astype(np.float64)
    if len(points) < 2: return 0.0, float(np.median(y) if len(y) else 0.0)
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(a), float(b)

def _fit_v(points: np.ndarray) -> Tuple[float,float]:  # x = a*y + b
    x = points[:,0].astype(np.float64); y = points[:,1].astype(np.float64)
    if len(points) < 2: return 0.0, float(np.median(x) if len(x) else 0.0)
    A = np.vstack([y, np.ones_like(y)]).T
    a, b = np.linalg.lstsq(A, x, rcond=None)[0]
    return float(a), float(b)

def _intersect_h_v(h: Tuple[float,float], v: Tuple[float,float]) -> np.ndarray:
    a_h, b_h = h; a_v, b_v = v
    denom = (1.0 - a_h*a_v)
    y = (a_h*b_v + b_h) / denom if abs(denom) > 1e-9 else b_h
    x = a_v*y + b_v
    return np.array([x, y], np.float32)

def _order_quad(q: np.ndarray) -> np.ndarray:
    q = q[np.argsort(q[:,1])]
    top, bot = q[:2], q[2:]
    TL, TR = top[np.argsort(top[:,0])]
    BL, BR = bot[np.argsort(bot[:,0])]
    return np.array([TL, TR, BR, BL], np.float32)

def _warp_cell(gray_img: np.ndarray, P: np.ndarray, r: int, c: int, out: int = 64, shrink: float = 0.14) -> np.ndarray:
    q = np.array([P[r,c], P[r,c+1], P[r+1,c+1], P[r+1,c]], np.float32)
    ctr = q.mean(axis=0); q = ctr + (q - ctr)*(1.0 - shrink)
    M = cv2.getPerspectiveTransform(_order_quad(q), np.array([[0,0],[out-1,0],[out-1,out-1],[0,out-1]], np.float32))
    return cv2.warpPerspective(gray_img, M, (out, out), flags=cv2.INTER_LINEAR)

def _overlay_masks_and_points(gray_roi: np.ndarray, h_masks: List[np.ndarray], v_masks: List[np.ndarray], G: np.ndarray) -> np.ndarray:
    vis = cv2.cvtColor(gray_roi, cv2.COLOR_GRAY2BGR)
    for m in h_masks:
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (0,255,0), 1)
    for m in v_masks:
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (0,165,255), 1)
    for r in range(10):
        for c in range(10):
            x, y = int(G[r,c,0]), int(G[r,c,1])
            cv2.circle(vis, (x,y), 3, (0,0,255), -1)
    return vis

def _clip_grid_inplace(G: np.ndarray, H: int, W: int):
    G[:,:,0] = np.clip(G[:,:,0], 0, W-1)
    G[:,:,1] = np.clip(G[:,:,1], 0, H-1)

# ───────── main ─────────
def process(image_path: str, out_dir: str, tile_size: int = 64) -> dict[str, Any]:
    out = Path(out_dir); _ensure_dir(out)
    debug = out / "rectify_debug"; _ensure_dir(debug)

    img0 = cv2.imread(str(image_path))
    if img0 is None: raise RuntimeError("Cannot read image")
    gray0 = _to_gray(img0)

    # Angle sweep — require 8+8, selector has fallbacks
    angle_candidates = list(range(-10, 11, 2))
    best = None; best_score = -1e9
    for ang in angle_candidates:
        gray = _rotate_image(gray0, ang) if ang else gray0
        mh_all, mv_all, _ = build_oriented_masks(gray)
        (mh, mv), roi = sudoku_roi_from_masks(mh_all, mv_all)
        Hroi, Wroi = mh.shape[:2]
        if Hroi <= 0 or Wroi <= 0: continue
        approx_cell = min(Hroi, Wroi)/9.0
        try:
            hsel = _select_8_lines(mh, 'h', approx_cell, other_mask=mv)
            vsel = _select_8_lines(mv, 'v', approx_cell, other_mask=mh)
        except RuntimeError:
            continue
        # score by uniformity of gaps (lower is better) + mild angle regularization
        cvh = _cv_gaps(_line_centers(hsel, 'h'))
        cvv = _cv_gaps(_line_centers(vsel, 'v'))
        score = 100.0 - 100.0*(cvh + cvv) - 0.5*abs(ang)
        if score > best_score:
            best_score = score
            best = (ang, gray, mh, mv, roi, hsel, vsel)

    if best is None:
        raise RuntimeError("Failed to find angle with 8×8 internal lines")

    ang, gray, mh, mv, (y0,y1,x0,x1), h_masks, v_masks = best
    img = _rotate_image(img0, ang) if ang else img0
    gray_roi = gray[y0:y1, x0:x1]

    # S1 debug for chosen angle
    mask_h, mask_v, mask_all = build_oriented_masks(gray)
    _save(debug/"S1_lines_h.png", mask_h)
    _save(debug/"S1_lines_v.png", mask_v)
    _save(debug/"S1_grid_mask.png", mask_all)
    roi_draw = img.copy(); cv2.rectangle(roi_draw, (x0,y0), (x1,y1), (0,255,0), 2)
    _save(debug/"S1_roi_rect.png", roi_draw)
    _save(debug/"S1_gray_roi.png", gray_roi)

    Hroi2, Wroi2 = gray_roi.shape[:2]

    # 8×8 interior intersections
    P8 = _intersections_from_masks(h_masks, v_masks, dilate_px=3)

    # endpoints and oriented line models for internals
    Ls, Rs, Ts, Bs = [], [], [], []
    h_lines, v_lines = [], []
    for hm in h_masks:
        L, R = _endpoints_from_mask(hm, 'h'); Ls.append(L); Rs.append(R)
        h_lines.append(_fit_h(np.array([L, R], np.float32)))
    for vm in v_masks:
        T, B = _endpoints_from_mask(vm, 'v'); Ts.append(T); Bs.append(B)
        v_lines.append(_fit_v(np.array([T, B], np.float32)))
    Ls, Rs, Ts, Bs = map(lambda x: np.array(x, np.float32), (Ls, Rs, Ts, Bs))

    # robust borders (orientation-aware)
    left_border   = _fit_v(Ls)
    right_border  = _fit_v(Rs)
    top_border    = _fit_h(Ts)
    bottom_border = _fit_h(Bs)

    # 10×10 lattice with border-fit
    G = np.zeros((10,10,2), np.float32)
    G[1:9,1:9,:] = P8
    for c in range(8):
        G[0,  c+1] = _intersect_h_v(top_border,    v_lines[c])
        G[9,  c+1] = _intersect_h_v(bottom_border, v_lines[c])
    for r in range(8):
        G[r+1, 0] = _intersect_h_v(h_lines[r], left_border)
        G[r+1, 9] = _intersect_h_v(h_lines[r], right_border)
    G[0,0] = _intersect_h_v(top_border, left_border)
    G[0,9] = _intersect_h_v(top_border, right_border)
    G[9,0] = _intersect_h_v(bottom_border, left_border)
    G[9,9] = _intersect_h_v(bottom_border, right_border)

    # keep lattice inside ROI
    _clip_grid_inplace(G, Hroi2, Wroi2)

    # overlay
    overlay = _overlay_masks_and_points(gray_roi, h_masks, v_masks, G)
    _save(debug/"S2_lines_and_points.png", overlay)

    inside = int(np.sum((G[:,:,0] >= 0) & (G[:,:,0] < Wroi2) &
                        (G[:,:,1] >= 0) & (G[:,:,1] < Hroi2)))
    print(f"[new-grid] angle={ang:+.1f}° | lattice_points={inside}/100")

    # tiles
    cells_dir = out/"cells"; _ensure_dir(cells_dir)
    cell_paths: List[str] = []
    for r in range(9):
        for c in range(9):
            tile = _warp_cell(gray_roi, G, r, c, out=tile_size, shrink=0.14)
            p = cells_dir/f"r{r+1}c{c+1}.png"; _save(p, tile); cell_paths.append(str(p))
    _save(out/"board_warped.png", gray_roi)
    _save(out/"board_clean.png", gray_roi)
    (out/"cells.json").write_text(json.dumps({
        "tiles": cell_paths,
        "roi": {"y0": int(y0), "y1": int(y1), "x0": int(x0), "x1": int(x1)}
    }, indent=2), encoding="utf-8")

    return {
        "warped": str(out/"board_warped.png"),
        "clean": str(out/"board_clean.png"),
        "cells_dir": str(cells_dir),
        "cells_json": str(out/"cells.json"),
        "cells_count": len(cell_paths),
        "rotation_deg": float(ang),
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python opencv_rectify.py <image_path> <export_dir>")
        raise SystemExit(1)
    print(process(sys.argv[1], sys.argv[2]))