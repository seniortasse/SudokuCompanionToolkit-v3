"""
===============================================================================
Sudoku Photo → 81 Clean Tiles — Internal-Lines Rectifier (Production-ready)
===============================================================================

WHY (the problem)
-----------------
A phone snapshot of a printed Sudoku is messy: perspective skew, bowed lines from
lens curvature, headers/footers that compete with the grid, uneven lighting, and
thick borders that can confuse “biggest quadrilateral” approaches. We need a
robust, *explainable* rectifier that returns 81 correctly ordered tiles (64×64)
and rich debug artifacts so failures are obvious and fixable.

WHAT (this module does)
-----------------------
Given an image path, we:
  1) Find the Sudoku board region of interest (ROI) without trusting outer borders.
  2) Detect oriented masks (H/V) and select exactly 8 internal horizontal and
     8 internal vertical rails (ignoring page frames and headers).
  3) Repair short/broken rails, then compute the 8×8 intersection lattice.
  4) Extrapolate an accurate outer ring (10×10 lattice) with fitted borders
     and corner locking to prevent drift.
  5) Warp each cell to a fixed square tile, with a configurable “shrink” to
     keep inked grid lines away from tile edges.
  6) Emit clear debug images (S1/S2) and reproducible outputs for QA.

HOW (core ideas & guarantees)
-----------------------------
• Internal-lines strategy — We detect and rank only *internal* rails by span,
  thickness, crossings and central coverage. This avoids depending on page
  borders and tolerates headers/footers.

• Collinear fragment merging — Repairs broken lines before selection.

• Adaptive ring extrapolation — From the 8×8 intersection grid, we fit border
  lines and clamp the 10×10 ring to those fitted borders. Corners are computed
  from border intersections to stop edge drift.

• Scale-aware rendering — Debug overlays draw with thickness/radius scaled to
  image size so artifacts are readable at any resolution.

WHERE (intended deployment)
---------------------------
Works on device or server. Optional pre-compression keeps processing snappy
on mobile. All steps are deterministic and log their artifacts for debugging.

WHEN (use and extend)
---------------------
Use for production rectification now. Extend with optional inpainting/grid
removal *after* lattice construction if you want fully border-free tiles.

FILE NAVIGATION (sections)
--------------------------
  1. Basic helpers
  2. Optional precompression / downscale
  3. Oriented masks (H/V)
  4. Sudoku ROI from masks
  5. Components & utilities
  6. Crossings & central-coverage tests
  7. Collinear grouping (rail repairs)
  8. Select exactly 8 internal lines
  9. Refit/extend short selected lines
 10. Intersections (8×8)
 11. Adaptive outer ring & corners (10×10)
 12. Overlay / warp helpers
 13. Main entry: process()

OUTPUTS (by default)
--------------------
out/
  board_warped.png   – the working ROI used for cropping
  board_clean.png    – identical to warped (reserved for future “cleaning” step)
  cells/ r{1..9}c{1..9}.png  – 81 tiles (top-left r1c1 … bottom-right r9c9)
  cells.json         – tile paths + ROI box in the original image
  rectify_debug/
    S1_lines_h.png   – horizontal mask (full image)
    S1_lines_v.png   – vertical mask (full image)
    S1_grid_mask.png – H|V fused mask (full image)
    S1_roi_rect.png  – ROI box over the original image
    S1_gray_roi.png  – grayscale ROI crop
    S2_lines_and_points.png – selected rails + 10×10 lattice points overlay

TUNABLES (high-level)
---------------------
• tile_size (default 64)          – square tile output dimension
• shrink (default 0.14)           – inward shrink before warping to keep rails off edges
• angle sweep range               – coarse rotation search to stabilize rail selection
• morphological kernel sizes      – scale with approx_cell for robust masks

USAGE (CLI)
-----------
$ python opencv_rectify.py <image_path> <export_dir>

RETURN (programmatic)
---------------------
process(...) → dict with paths, rotation_deg, precompressed flag.

Design note
-----------
The code favors *explainability*: every decision (line selection, lattice points)
can be inspected via S1/S2 artifacts. Where possible, thresholds are tied to
“approx_cell” (min(ROI.height, ROI.width) / 9) for scale invariance.
"""

from __future__ import annotations
from typing import Any, List, Tuple
from pathlib import Path
import json
import numpy as np
import cv2

# ──────────────────────────────────────────────────────────────────────────────
# 1) BASIC HELPERS
#    Small utilities for filesystem, color conversions, saving, rotation, resize.
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_dir(p: Path):
    """Create directory p (parents too) if it does not exist."""
    p.mkdir(parents=True, exist_ok=True)

def _to_gray(img: np.ndarray) -> np.ndarray:
    """Return a single-channel grayscale image (no copy if already gray)."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

def _save(p: Path, im: np.ndarray):
    """Write image to disk (path → OpenCV)."""
    cv2.imwrite(str(p), im)

def _rotate_image(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate image about center by angle_deg (deg). Border is replicated."""
    H, W = img.shape[:2]
    M = cv2.getRotationMatrix2D((W/2.0, H/2.0), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def _resize_longside(img: np.ndarray, long_side: int = 1280) -> Tuple[np.ndarray, float]:
    """
    Resize image so max(H, W) <= long_side.
    Returns (resized_img, scale) where 'scale' is the factor applied to width/height.
    """
    H, W = img.shape[:2]
    L = max(H, W)
    if L <= long_side:
        return img, 1.0
    s = long_side / float(L)
    return cv2.resize(img, (int(W*s), int(H*s)), interpolation=cv2.INTER_AREA), s


# ──────────────────────────────────────────────────────────────────────────────
# 2) OPTIONAL PRECOMPRESSION / DOWNSCALE
#    Keep mobile latency low by bounding input size and JPEG budget.
# ──────────────────────────────────────────────────────────────────────────────

def downscale_for_work(img: np.ndarray, max_long_side: int = 1600) -> np.ndarray:
    """Resize so max(H, W) <= max_long_side; returns the resized image."""
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_long_side:
        return img
    s = max_long_side / float(m)
    return cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)

def compress_to_budget(img_bgr: np.ndarray,
                       target_kb: int = 100,
                       max_long_side: int = 1600,
                       min_quality: int = 40) -> tuple[np.ndarray, bytes]:
    """
    Resize to max_long_side, then JPEG-encode via binary search on quality to
    meet target_kb. Returns (decoded_image, jpeg_bytes).
    """
    work = downscale_for_work(img_bgr, max_long_side=max_long_side)

    lo, hi = min_quality, 95
    best = None
    while lo <= hi:
        q = (lo + hi) // 2
        ok, buf = cv2.imencode(".jpg", work, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
        if not ok:
            break
        size_kb = buf.size / 1024.0
        if size_kb <= target_kb:
            best = buf  # keep this and try higher quality
            lo = q + 1
        else:
            hi = q - 1

    if best is None:
        ok, best = cv2.imencode(".jpg", work, [cv2.IMWRITE_JPEG_QUALITY, min_quality])

    dec = cv2.imdecode(best, cv2.IMREAD_COLOR)
    return dec, best.tobytes()


# ──────────────────────────────────────────────────────────────────────────────
# 3) ORIENTED MASKS (H/V)
#    Robust, scale-aware morphological extraction of horizontal/vertical ink.
#    Produces:
#      mask_h: strong horizontal strokes
#      mask_v: strong vertical strokes
#      mask_all: union of both
# ──────────────────────────────────────────────────────────────────────────────

def build_oriented_masks(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    CLAHE equalization → adaptive threshold → oriented morphology with
    gap healing. Kernel sizes scale with approx_cell so behavior is
    consistent across resolutions.
    """
    H, W = gray.shape[:2]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    eq = clahe.apply(gray)

    # Adaptive binarization (odd block ~3% of min side)
    block = max(21, (min(H, W)//32) | 1)
    bw = cv2.adaptiveThreshold(eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, blockSize=block, C=6)

    approx_cell = min(H, W)/9.0

    # Long kernels along main direction; slim kernels orthogonal
    L = max(15, int(0.85*approx_cell))
    k_h1 = cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1))
    k_hd = cv2.getStructuringElement(cv2.MORPH_RECT, (L+2, 3))
    k_v1 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, L))
    k_vd = cv2.getStructuringElement(cv2.MORPH_RECT, (3, L+2))

    mask_h = cv2.dilate(cv2.erode(bw, k_h1, 1), k_hd, 1)
    mask_v = cv2.dilate(cv2.erode(bw, k_v1, 1), k_vd, 1)

    # Heal gaps along each axis
    gap = max(5, int(0.28*approx_cell))
    mask_h = cv2.morphologyEx(mask_h, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (gap, 1)), 1)
    mask_v = cv2.morphologyEx(mask_v, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (1, gap)), 1)

    # Small close to clean speckles
    mask_h = cv2.morphologyEx(mask_h, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), 1)
    mask_v = cv2.morphologyEx(mask_v, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), 1)

    return mask_h, mask_v, cv2.bitwise_or(mask_h, mask_v)


# ──────────────────────────────────────────────────────────────────────────────
# 4) SUDOKU ROI FROM MASKS
#    Fuse H|V masks, dilate to bridge gaps, select largest connected component.
# ──────────────────────────────────────────────────────────────────────────────

def sudoku_roi_from_masks(mask_h: np.ndarray, mask_v: np.ndarray):
    """
    Return ((mask_h_roi, mask_v_roi), (y0,y1,x0,x1)) where the ROI is the
    bounding box of the largest fused component after a scale-aware dilation.
    """
    H, W = mask_h.shape[:2]
    fused = cv2.bitwise_or(mask_h, mask_v)
    d = max(7, int(0.06*min(H, W)))
    fused = cv2.dilate(fused, cv2.getStructuringElement(cv2.MORPH_RECT, (d, d)), 1)

    num, lab = cv2.connectedComponents((fused > 0).astype(np.uint8))
    if num <= 1:
        return (mask_h, mask_v), (0, H, 0, W)

    bestA, best = -1, (0, H, 0, W)
    for t in range(1, num):
        ys, xs = np.where(lab == t)
        if xs.size == 0: continue
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        A = (x1-x0+1)*(y1-y0+1)
        if A > bestA: bestA, best = A, (y0, y1, x0, x1)
    y0, y1, x0, x1 = best
    return (mask_h[y0:y1, x0:x1], mask_v[y0:y1, x0:x1]), (y0, y1, x0, x1)


# ──────────────────────────────────────────────────────────────────────────────
# 5) COMPONENTS & UTILITIES
#    Connected components; median centers for spacing/cv; small helpers.
# ──────────────────────────────────────────────────────────────────────────────

def _components_from_mask(mask: np.ndarray):
    """Return (components, labels) where components = [(id, bbox), ...]."""
    num, lab = cv2.connectedComponents((mask > 0).astype(np.uint8))
    comps = []
    for k in range(1, num):
        ys, xs = np.where(lab == k)
        if xs.size == 0: continue
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        comps.append((k, (x0, y0, x1, y1)))
    return comps, lab

def _line_centers(line_masks: List[np.ndarray], axis: str) -> np.ndarray:
    """Median coordinate per mask along the *orthogonal* axis; sorted list."""
    coords = []
    for m in line_masks:
        ys, xs = np.where(m > 0)
        coord = float(np.median(ys)) if axis == 'h' else float(np.median(xs)) if xs.size else 0.0
        coords.append(coord)
    return np.array(sorted(coords))

def _cv_gaps(coords: np.ndarray) -> float:
    """Coefficient of variation of consecutive gaps (spacing regularity metric)."""
    if len(coords) < 2: return 1e6
    gaps = np.diff(coords); mu = float(np.mean(gaps))
    return 1e6 if mu <= 1e-6 else float(np.std(gaps)/mu)


# ──────────────────────────────────────────────────────────────────────────────
# 6) CROSSINGS & CENTRAL-COVERAGE TESTS
#    Heuristics to gate “real” rails vs noise/header by requiring overlap and
#    crossings in a central band; margins near ROI edges are ignored.
# ──────────────────────────────────────────────────────────────────────────────

def _count_crossings_central_band(line_mask: np.ndarray, other_mask: np.ndarray,
                                  axis: str, dilate_px: int, ignore_margin_px: int, band_frac: float) -> int:
    """Count connected crossings with the other axis inside a central band."""
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
    inter = cv2.bitwise_and(cv2.dilate(cand, k, 1), cv2.dilate(other_mask, k, 1))
    if ignore_margin_px > 0:
        if axis == 'h':
            inter[:ignore_margin_px,:] = 0
            inter[H-ignore_margin_px:,:] = 0
        else:
            inter[:, :ignore_margin_px] = 0
            inter[:, W-ignore_margin_px:] = 0
    num, _ = cv2.connectedComponents((inter > 0).astype(np.uint8))
    return max(0, num - 1)

def _central_coverage(line_mask: np.ndarray, axis: str, band_frac: float) -> float:
    """Fraction of columns/rows showing ink inside the central band."""
    H, W = line_mask.shape[:2]
    if axis == 'h':
        y0 = int(band_frac*H); y1 = int((1.0-band_frac)*H)
        roi = line_mask[y0:y1, :]
        covered = np.count_nonzero((roi > 0).any(axis=0))  # columns with ink
        return covered / float(max(1, W))
    else:
        x0 = int(band_frac*W); x1 = int((1.0-band_frac)*W)
        roi = line_mask[:, x0:x1]
        covered = np.count_nonzero((roi > 0).any(axis=1))  # rows with ink
        return covered / float(max(1, H))


# ──────────────────────────────────────────────────────────────────────────────
# 7) COLLINEAR GROUPING (RAIL REPAIRS)
#    Merge components that lie along the same rail using a tolerance on the
#    orthogonal coordinate; return merged masks + bbox + median coord.
# ──────────────────────────────────────────────────────────────────────────────

def _group_collinear_components(mask: np.ndarray, axis: str, approx_cell: float):
    """Group near-collinear components and return merged rail candidates."""
    comps, lab = _components_from_mask(mask)
    if not comps: return []
    entries = []
    for k, (x0, y0, x1, y1) in comps:
        coord = 0.5*(x0+x1) if axis == 'v' else 0.5*(y0+y1)
        entries.append((coord, k, (x0, y0, x1, y1)))
    entries.sort(key=lambda t: t[0])

    tol = 0.18*approx_cell  # tolerance for grouping along the orthogonal coordinate
    groups: List[List[Tuple[int, Tuple[int,int,int,int], float]]] = []
    curr = []; last = None
    for c, k, b in entries:
        if last is None or abs(c - last) <= tol:
            curr.append((k, b, c))
        else:
            groups.append(curr); curr = [(k, b, c)]
        last = c
    if curr: groups.append(curr)

    merged = []
    for g in groups:
        gm = np.zeros_like(mask, np.uint8)
        xs0, ys0, xs1, ys1, coords = [], [], [], [], []
        for k, (x0, y0, x1, y1), c in g:
            gm[lab == k] = 255
            xs0.append(x0); xs1.append(x1); ys0.append(y0); ys1.append(y1)
            coords.append(c)
        bbox = (min(xs0), min(ys0), max(xs1), max(ys1))
        merged.append((gm, bbox, float(np.median(coords))))
    return merged


# ──────────────────────────────────────────────────────────────────────────────
# 8) SELECT EXACTLY 8 INTERNAL LINES
#    Multi-stage gating: span/thickness → central coverage → crossings →
#    spacing regularity, with a “best window of 8” selector if more survive.
# ──────────────────────────────────────────────────────────────────────────────

def _select_8_lines(mask: np.ndarray, axis: str, approx_cell: float, other_mask: np.ndarray) -> List[np.ndarray]:
    """
    Select precisely 8 rails along one axis.
      • Filters by span (≥ fraction of ROI) and thickness (≤ px).
      • Requires coverage in a central band.
      • Counts crossings with the other axis near the center.
      • Chooses the best contiguous window of 8 by spacing regularity + margin.
    """
    H, W = mask.shape[:2]
    merged = _group_collinear_components(mask, axis, approx_cell)

    # progressively relax span/thickness; then gates on crossings/coverage
    span_thick = [(0.55,0.33),(0.45,0.45),(0.35,0.60),(0.30,0.80),(0.25,0.90)]
    gates = [ (0.50,0.18,0.40,6), (0.45,0.16,0.35,5), (0.40,0.14,0.30,5), (0.38,0.12,0.25,4) ]
    min_cov = 0.22

    def choose_window(cands):
        """Pick the most even 8 rails (lowest spacing CV; slight center bias)."""
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
            return (w >= min_span_frac*W and h <= max_thick_px) if axis=='h' else (h >= min_span_frac*H and w <= max_thick_px)
        base = [(m,b,c) for (m,b,c) in merged if ok(b)]
        if len(base) < 8: continue

        for center_margin_frac, band_frac, ignore_frac, crossings_min in gates:
            center_margin = center_margin_frac*approx_cell
            ignore_px = int(ignore_frac*approx_cell)
            cands = []
            for (cmask, b, coord) in base:
                if axis == 'h':
                    if coord < center_margin or (H - coord) < center_margin: continue
                else:
                    if coord < center_margin or (W - coord) < center_margin: continue
                if _central_coverage(cmask, axis, band_frac) < min_cov: continue
                crossings = _count_crossings_central_band(
                    cmask, other_mask, axis=axis, dilate_px=2,
                    ignore_margin_px=ignore_px, band_frac=band_frac
                )
                if crossings >= crossings_min:
                    cands.append((coord, cmask))
            if len(cands) >= 8: return choose_window(cands)

        # fallback: coverage + spacing only
        fallback_band = gates[-1][1]
        cands = []
        for (cmask, b, coord) in base:
            if _central_coverage(cmask, axis, fallback_band) >= min_cov:
                cands.append((coord, cmask))
        if len(cands) >= 8: return choose_window(cands)

    # final resort: most central with coverage
    if len(merged) >= 8:
        H, W = mask.shape[:2]
        center = H/2.0 if axis=='h' else W/2.0
        fb = gates[-1][1]
        filt = [(abs(center-c), m) for (m,b,c) in merged if _central_coverage(m, axis, fb) >= min_cov]
        if len(filt) >= 8:
            filt.sort(key=lambda t: t[0])
            return [filt[i][1] for i in range(8)]
    raise RuntimeError(f"Not enough components to select 8 {axis}-lines (have {len(merged)} after grouping/support).")


# ──────────────────────────────────────────────────────────────────────────────
# 9) REFIT/EXTEND SHORT SELECTED LINES
#    For rails that don’t span the ROI, refit as polyline (quadratic if it helps)
#    and redraw with scale-aware thickness to follow bow/tilt.
# ──────────────────────────────────────────────────────────────────────────────

def _refit_mask_line(m: np.ndarray, axis: str, W: int, H: int, approx_cell: float) -> np.ndarray:
    """Refit a line mask with a 1st/2nd-order poly model (axis-aware), extend to ROI."""
    pts = cv2.findNonZero(m)
    if pts is None or len(pts) < 50:
        return m
    pts = pts[:,0,:].astype(np.float64)
    xs, ys = pts[:,0], pts[:,1]
    thickness = max(2, int(0.12*approx_cell))
    if axis == 'h':
        try: coeff = np.polyfit(xs, ys, 2)
        except np.linalg.LinAlgError: coeff = np.polyfit(xs, ys, 1)
        poly = np.poly1d(coeff)
        xx = np.linspace(0, W-1, num=W, dtype=np.float32)
        yy = np.clip(poly(xx), 0, H-1).astype(np.int32)
        newm = np.zeros_like(m)
        pts_poly = np.stack([xx.astype(np.int32), yy], axis=1).reshape(-1,1,2)
        cv2.polylines(newm, [pts_poly], False, 255, thickness)
        return newm
    else:
        try: coeff = np.polyfit(ys, xs, 2)
        except np.linalg.LinAlgError: coeff = np.polyfit(ys, xs, 1)
        poly = np.poly1d(coeff)
        yy = np.linspace(0, H-1, num=H, dtype=np.float32)
        xx = np.clip(poly(yy), 0, W-1).astype(np.int32)
        newm = np.zeros_like(m)
        pts_poly = np.stack([xx, yy.astype(np.int32)], axis=1).reshape(-1,1,2)
        cv2.polylines(newm, [pts_poly], False, 255, thickness)
        return newm

def _refine_selected_masks(masks: List[np.ndarray], axis: str, W: int, H: int, approx_cell: float) -> List[np.ndarray]:
    """Refit each selected rail iff its span is short vs ROI; preserves long rails."""
    refined = []
    for m in masks:
        ys, xs = np.where(m > 0)
        if xs.size == 0:
            refined.append(m); continue
        if axis == 'h':
            span = xs.max() - xs.min()
            need_refit = span < 0.90*W
        else:
            span = ys.max() - ys.min()
            need_refit = span < 0.90*H
        refined.append(_refit_mask_line(m, axis, W, H, approx_cell) if need_refit else m)
    return refined


# ──────────────────────────────────────────────────────────────────────────────
# 10) INTERSECTIONS (8×8)
#     Compute median intersection points between each selected H/V pair.
# ──────────────────────────────────────────────────────────────────────────────

def _intersections_from_masks(h_masks: List[np.ndarray], v_masks: List[np.ndarray], dilate_px: int = 3) -> np.ndarray:
    """Return P8[8,8,2] = (x,y) lattice from dilated rail overlaps."""
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
                # very rare: fall back to distance-transform peak
                dt = cv2.distanceTransform(cv2.bitwise_not(inter), cv2.DIST_L2, 3)
                yx = np.unravel_index(np.argmax(dt), dt.shape)
                pts[i, j] = [float(yx[1]), float(yx[0])]
    return pts


# ──────────────────────────────────────────────────────────────────────────────
# 11) ADAPTIVE OUTER RING & CORNERS (10×10)
#     Extrapolate top/bottom from columns, left/right from rows; clamp outliers; 
#     compute corners from fitted border intersections.
# ──────────────────────────────────────────────────────────────────────────────

def _fit_mse(values: np.ndarray, deg: int) -> Tuple[np.poly1d, float]:
    """Fit polynomial of degree 'deg' to index→value and return (poly, mse)."""
    idx = np.arange(1, 9, dtype=np.float64)
    coeff = np.polyfit(idx, values.astype(np.float64), deg)
    f = np.poly1d(coeff)
    mse = float(np.mean((f(idx) - values.astype(np.float64))**2))
    return f, mse

def _extrap_adaptive_xy(xs: np.ndarray, ys: np.ndarray, improve_ratio: float = 0.85) -> Tuple[float,float,float,float]:
    """Pick linear vs quadratic extrapolation column/row-wise based on MSE."""
    f1x, m1x = _fit_mse(xs, 1); f2x, m2x = _fit_mse(xs, 2)
    f1y, m1y = _fit_mse(ys, 1); f2y, m2y = _fit_mse(ys, 2)
    use_quad = (m2x < m1x*improve_ratio) or (m2y < m1y*improve_ratio)
    fx, fy = (f2x, f2y) if use_quad else (f1x, f1y)
    return float(fx(0.0)), float(fx(9.0)), float(fy(0.0)), float(fy(9.0))

def _clamp_extrap(p_ref: np.ndarray, p_next: np.ndarray, p_pred: np.ndarray, factor: float = 1.6) -> np.ndarray:
    """Limit extrapolation step vs preceding step length to curb runaway edges."""
    step = p_ref - p_next
    ref_to_pred = p_pred - p_ref
    nstep = np.linalg.norm(step) + 1e-6
    nref  = np.linalg.norm(ref_to_pred)
    if nref > factor * nstep:
        ref_to_pred = ref_to_pred * (factor * nstep / nref)
    return p_ref + ref_to_pred

def _complete_lattice_adaptive(P8: np.ndarray) -> np.ndarray:
    """Complete 8×8 internal lattice to a 10×10 ring with fitted borders & corners."""
    G = np.zeros((10,10,2), np.float32)
    G[1:9,1:9,:] = P8

    # Top/bottom via columns
    for j in range(8):
        xs = P8[:, j, 0]; ys = P8[:, j, 1]
        xt, xb, yt, yb = _extrap_adaptive_xy(xs, ys)
        top = _clamp_extrap(P8[0, j], P8[1, j], np.array([xt, yt], np.float32), 1.6)
        bot = _clamp_extrap(P8[7, j], P8[6, j], np.array([xb, yb], np.float32), 1.6)
        G[0, j+1] = top; G[9, j+1] = bot

    # Left/right via rows
    for i in range(8):
        xs = P8[i, :, 0]; ys = P8[i, :, 1]
        xl, xr, yl, yr = _extrap_adaptive_xy(xs, ys)
        lef = _clamp_extrap(P8[i, 0], P8[i, 1], np.array([xl, yl], np.float32), 1.6)
        rig = _clamp_extrap(P8[i, 7], P8[i, 6], np.array([xr, yr], np.float32), 1.6)
        G[i+1, 0] = lef; G[i+1, 9] = rig

    # Corners from fitted border intersections
    def _fit_h(points: np.ndarray) -> Tuple[float,float]:  # y = a*x + b
        x = points[:,0].astype(np.float64); y = points[:,1].astype(np.float64)
        if len(points) < 2: return 0.0, float(np.median(y) if len(y) else 0.0)
        A = np.vstack([x, np.ones_like(x)]).T
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
        return float(a), float(b)
    def _fit_v(points: np.ndarray) -> Tuple[float,float]:  # x = a*y + b
        x = points[:,0].astype(np.float64); y = points[:,1].astype(np.float64)
        if len(points) < 2: return 0.0, float(np.median(x) if len(x) else 0.0)
        A = np.vstack([y, np.ones_like(x)]).T
        a, b = np.linalg.lstsq(A, x, rcond=None)[0]
        return float(a), float(b)
    def _intersect_h_v(h: Tuple[float,float], v: Tuple[float,float]) -> np.ndarray:
        a_h, b_h = h; a_v, b_v = v
        denom = (1.0 - a_h*a_v)
        y = (a_h*b_v + b_h) / denom if abs(denom) > 1e-9 else b_h
        x = a_v*y + b_v
        return np.array([x, y], np.float32)

    top_pts    = G[0, 1:9]
    bottom_pts = G[9, 1:9]
    left_pts   = G[1:9, 0]
    right_pts  = G[1:9, 9]

    def _first3(a): return a[:3] if len(a) >= 3 else a
    def _last3(a):  return a[-3:] if len(a) >= 3 else a

    G[0,0] = _intersect_h_v(_fit_h(_first3(top_pts)),    _fit_v(_first3(left_pts)))
    G[0,9] = _intersect_h_v(_fit_h(_last3(top_pts)),     _fit_v(_first3(right_pts)))
    G[9,0] = _intersect_h_v(_fit_h(_first3(bottom_pts)), _fit_v(_last3(left_pts)))
    G[9,9] = _intersect_h_v(_fit_h(_last3(bottom_pts)),  _fit_v(_last3(right_pts)))

    return G


# ──────────────────────────────────────────────────────────────────────────────
# 12) OVERLAY / WARP HELPERS
#     Visualization and per-cell perspective warps (with “shrink” guard).
# ──────────────────────────────────────────────────────────────────────────────

def _overlay_masks_and_points(gray_roi: np.ndarray, h_masks: List[np.ndarray], v_masks: List[np.ndarray], G: np.ndarray) -> np.ndarray:
    """Draw rails (green/orange) and 10×10 lattice points (red) over ROI."""
    vis = cv2.cvtColor(gray_roi, cv2.COLOR_GRAY2BGR)
    H, W = gray_roi.shape[:2]
    # scale-aware ink
    thick = max(2, int(min(H, W) * 0.003))     # contours
    radius = max(2, int(min(H, W) * 0.008))    # red dots
    for m in h_masks:
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (0,255,0), thick)
    for m in v_masks:
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (0,165,255), thick)
    for r in range(10):
        for c in range(10):
            x, y = int(G[r,c,0]), int(G[r,c,1])
            cv2.circle(vis, (x,y), radius, (0,0,255), -1)
    return vis

def _clip_grid_inplace(G: np.ndarray, H: int, W: int):
    """Clamp lattice coordinates to ROI bounds (safety against extrap overshoot)."""
    G[:,:,0] = np.clip(G[:,:,0], 0, W-1)
    G[:,:,1] = np.clip(G[:,:,1], 0, H-1)

def _warp_cell(gray_img: np.ndarray, P: np.ndarray, r: int, c: int, out: int = 64, shrink: float = 0.14) -> np.ndarray:
    """
    Warp a cell quad [P[r,c] ... P[r+1,c]] → out×out tile.
    We shrink the quad toward its center so border ink doesn’t leak into tiles.
    """
    quad = np.array([P[r,c], P[r,c+1], P[r+1,c+1], P[r+1,c]], np.float32)
    ctr = quad.mean(axis=0); quad = ctr + (quad - ctr)*(1.0 - shrink)
    M = cv2.getPerspectiveTransform(quad, np.array([[0,0],[out-1,0],[out-1,out-1],[0,out-1]], np.float32))
    return cv2.warpPerspective(gray_img, M, (out, out), flags=cv2.INTER_LINEAR)


# ──────────────────────────────────────────────────────────────────────────────
# 13) MAIN ENTRY — process()
#     Run the whole pipeline, save artifacts, return useful paths & metadata.
# ──────────────────────────────────────────────────────────────────────────────

def process(image_path: str,
            out_dir: str,
            tile_size: int = 64,
            *,
            precompress: bool = False,
            target_kb: int = 100,
            work_max_long_side: int = 1600,
            min_jpeg_quality: int = 40) -> dict[str, Any]:
    """
    Parameters
    ----------
    image_path : str
        Input photo path.
    out_dir : str
        Output directory (created if missing).
    tile_size : int
        Output tile side (pixels).
    precompress : bool
        If True, downscale + JPEG budget before rectifying (mobile-friendly).
    target_kb : int
        JPEG size budget when precompressing.
    work_max_long_side : int
        Long side cap for work image.
    min_jpeg_quality : int
        Lower bound in binary-search for JPEG quality.

    Returns
    -------
    dict
        Paths to outputs, rotation_deg used, and other flags.
    """
    out = Path(out_dir); _ensure_dir(out)
    debug = out / "rectify_debug"; _ensure_dir(debug)

    img0 = cv2.imread(str(image_path))
    if img0 is None: raise RuntimeError("Cannot read image")

    # ---- (opt) precompression for fast work on mobile ----
    if precompress:
        img_comp, jpeg_bytes = compress_to_budget(
            img0, target_kb=target_kb, max_long_side=work_max_long_side, min_quality=min_jpeg_quality
        )
        (debug / "compressed_input.jpg").write_bytes(jpeg_bytes)
        img0 = img_comp
    # else:
    #     img0 = downscale_for_work(img0, max_long_side=work_max_long_side)

    gray0 = _to_gray(img0)

    # ── 1) FAST ANGLE SWEEP (small image) ──
    # Try a small set of rotations to stabilize rail selection; score by uniformity.
    small, scale = _resize_longside(gray0, 1280)
    angle_candidates = list(range(-10, 11, 2))
    best = None; best_score = -1e9
    for ang in angle_candidates:
        g = _rotate_image(small, ang) if ang else small
        mh_all, mv_all, _ = build_oriented_masks(g)
        (mh, mv), roi = sudoku_roi_from_masks(mh_all, mv_all)
        Hroi, Wroi = mh.shape[:2]
        if Hroi <= 0 or Wroi <= 0: continue
        approx_cell = min(Hroi, Wroi)/9.0
        try:
            hsel = _select_8_lines(mh, 'h', approx_cell, other_mask=mv)
            vsel = _select_8_lines(mv, 'v', approx_cell, other_mask=mh)
        except RuntimeError:
            continue
        cvh = _cv_gaps(_line_centers(hsel, 'h'))
        cvv = _cv_gaps(_line_centers(vsel, 'v'))
        score = 100.0 - 100.0*(cvh + cvv) - 0.5*abs(ang)  # even spacing + low rotation penalty
        if score > best_score:
            best_score = score
            best = ang

    if best is None:
        raise RuntimeError("Failed to find angle with 8×8 internal lines")

    ang = best

    # ── 2) FULL-RES (or compressed) ONCE AT BEST ANGLE ──
    img = _rotate_image(img0, ang) if ang else img0
    gray = _to_gray(img)
    mask_h_all, mask_v_all, mask_all = build_oriented_masks(gray)
    (mask_h, mask_v), (y0,y1,x0,x1) = sudoku_roi_from_masks(mask_h_all, mask_v_all)
    gray_roi = gray[y0:y1, x0:x1]

    # S1 artifacts — masks + ROI crop
    _save(debug/"S1_lines_h.png", mask_h_all)
    _save(debug/"S1_lines_v.png", mask_v_all)
    _save(debug/"S1_grid_mask.png", mask_all)
    roi_draw = img.copy(); cv2.rectangle(roi_draw, (x0,y0), (x1,y1), (0,255,0), 2)
    _save(debug/"S1_roi_rect.png", roi_draw)
    _save(debug/"S1_gray_roi.png", gray_roi)

    Hroi2, Wroi2 = gray_roi.shape[:2]
    approx_cell = min(Hroi2, Wroi2)/9.0

    # Select 8 H + 8 V internal rails
    h_masks = _select_8_lines(mask_h, 'h', approx_cell, other_mask=mask_v)
    v_masks = _select_8_lines(mask_v, 'v', approx_cell, other_mask=mask_h)

    # Refit/extend rails where needed (bowed/gridded pages)
    h_masks_ref = _refine_selected_masks(h_masks, 'h', Wroi2, Hroi2, approx_cell)
    v_masks_ref = _refine_selected_masks(v_masks, 'v', Wroi2, Hroi2, approx_cell)

    # Intersections + adaptive ring + corner locking
    P8 = _intersections_from_masks(h_masks_ref, v_masks_ref, dilate_px=3)
    G = _complete_lattice_adaptive(P8)
    _clip_grid_inplace(G, Hroi2, Wroi2)

    # S2 overlay (scale-aware)
    overlay = _overlay_masks_and_points(gray_roi, h_masks_ref, v_masks_ref, G)
    _save(debug/"S2_lines_and_points.png", overlay)

    inside = int(np.sum((G[:,:,0] >= 0) & (G[:,:,0] < Wroi2) & (G[:,:,1] >= 0) & (G[:,:,1] < Hroi2)))
    print(f"[adaptive-grid] angle={ang:+.1f}° | lattice_points={inside}/100")

    # ── 3) PER-CELL CROPS ──
    cells_dir = out/"cells"; _ensure_dir(cells_dir)
    paths: List[str] = []
    for r in range(9):
        for c in range(9):
            # Shrink keeps grid ink off tile borders (0.14 recommended for bold frames)
            tile = _warp_cell(gray_roi, G, r, c, out=tile_size, shrink=0.14)
            p = cells_dir/f"r{r+1}c{c+1}.png"; _save(p, tile); paths.append(str(p))

    # Output board previews (warped/clean)
    _save(out/"board_warped.png", gray_roi)
    _save(out/"board_clean.png", gray_roi)

    # Index JSON
    (out/"cells.json").write_text(json.dumps({
        "tiles": paths,
        "roi": {"y0": int(y0), "y1": int(y1), "x0": int(x0), "x1": int(x1)}
    }, indent=2), encoding="utf-8")

    return {
        "warped": str(out/"board_warped.png"),
        "clean": str(out/"board_clean.png"),
        "cells_dir": str(cells_dir),
        "cells_json": str(out/"cells.json"),
        "cells_count": len(paths),
        "rotation_deg": float(ang),
        "precompressed": bool(precompress),
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI entrypoint for quick testing
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python opencv_rectify.py <image_path> <export_dir>")
        raise SystemExit(1)
    # Example: python opencv_rectify.py img.jpg out_dir
    print(process(sys.argv[1], sys.argv[2]))