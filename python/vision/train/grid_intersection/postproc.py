# ------------------------------------------------------------
# Post-processing for [A,H,V,J,Ox,Oy] model outputs:
#  - hysteresis binarization
#  - junction NMS + subpixel refine (with threshold + topK fallback)
#  - 10x10 lattice clustering (x- and y-bands)
#  - constrained path tracing between junctions using A*/Dijkstra
#    with orientation alignment cost
#  - smoothing + resampling
#  - inward offsetting (to avoid eating cell interiors)
#  - Coons patch tiling grid for a single cell
# ------------------------------------------------------------

from dataclasses import dataclass
import numpy as np
import cv2
from scipy.signal import savgol_filter
from typing import List, Tuple
import math, heapq

@dataclass
class PPConfig:
    # A/H/V hysteresis thresholds (on probabilities)
    thr_strong: float = 0.5
    thr_weak:   float = 0.3

    # Junction extraction
    j_peak_thr: float = 0.02     # NEW: low default because your J means are ~0.001–0.003
    j_nms_radius: int = 2
    j_topk_min: int = 120        # ensure we always have enough candidates
    j_topk_cap: int = 180        # safety cap
    j_topk: int = 120            # target number after NMS (pre-fallback)

    # Lattice clustering
    cluster_band_y: float = 0.016  # fractions of H/W
    cluster_band_x: float = 0.016

    # Path tracing
    path_band_px: int = 10
    path_alpha: float = 1.0
    path_beta:  float = 0.5

    # Poly sampling / offsets
    resample_K: int = 64
    inward_margin_min: int = 2
    halfwidth_cap_px: int = 5

# ------------------------ Thresholding & skeleton ----------------------

def bin_maps(A: np.ndarray, H: np.ndarray, V: np.ndarray, cfg: PPConfig):
    """Return binary maps via hysteresis. Inputs are float [0..1] or uint8 0..255."""
    def _norm(x):
        return x.astype(np.float32) / (255.0 if x.dtype==np.uint8 else 1.0)
    A, H, V = map(_norm, (A,H,V))
    strong = (A >= cfg.thr_strong).astype(np.uint8)*255
    weak   = (A >= cfg.thr_weak).astype(np.uint8)*255
    # connect weak that touch strong (OpenCV hysteresis-like)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    prev = np.zeros_like(strong)
    cur = strong.copy()
    while True:
        dil = cv2.dilate(cur, kernel)
        cur = cv2.bitwise_or(cur, cv2.bitwise_and(weak, dil))
        if np.array_equal(cur, prev): break
        prev = cur.copy()
    A_bin = cur
    H_bin = (H >= cfg.thr_strong).astype(np.uint8)*255
    V_bin = (V >= cfg.thr_strong).astype(np.uint8)*255
    return A_bin, H_bin, V_bin

# --------------------------- Junction NMS ------------------------------

def junction_nms(J: np.ndarray, cfg: PPConfig):
    """
    Return [(x,y,score), ...] with subpixel refinement.

    Changes:
      - Apply a probability threshold (cfg.j_peak_thr) before NMS.
      - Keep up to cfg.j_topk after NMS.
      - If nothing (or too few) survives, fall back to global top-K to
        guarantee we have at least cfg.j_topk_min candidates (capped).
    """
    # normalize to [0,1]
    Jf = J.astype(np.float32) / (255.0 if J.dtype==np.uint8 else 1.0)
    h, w = Jf.shape

    # ---- 1) Candidate mask by threshold
    cand = (Jf >= float(cfg.j_peak_thr))

    # ---- 2) Local maxima (max-pooling NMS within radius)
    k = cfg.j_nms_radius*2 + 1
    Jmax = cv2.dilate(Jf, cv2.getStructuringElement(cv2.MORPH_RECT, (k,k)))
    is_max = (Jf >= (Jmax - 1e-8))
    mask = cand & is_max

    ys, xs = np.where(mask)
    scores = Jf[ys, xs]

    # ---- 3) Sort by score and keep topk
    if scores.size > 0:
        order = np.argsort(scores)[::-1]
        keep = min(len(order), int(cfg.j_topk))
        ys = ys[order[:keep]]
        xs = xs[order[:keep]]
        scores = scores[order[:keep]]

    # ---- 4) Fallback: if too few (or zero), take global top-K by score
    # This is crucial when J map is very dark: threshold+NMS might produce 0.
    if scores.size < int(cfg.j_topk_min):
        # take global top-K ignoring threshold
        flat = Jf.reshape(-1)
        topk = min(int(cfg.j_topk_cap), int(max(cfg.j_topk_min, cfg.j_topk)))
        # If the map is entirely zeros, torch.topk would be undefined here; use numpy:
        idx = np.argpartition(flat, -topk)[-topk:]
        idx = idx[np.argsort(flat[idx])[::-1]]  # sorted descending
        ys = (idx // w).astype(np.int64)
        xs = (idx %  w).astype(np.int64)
        scores = Jf[ys, xs]

    # ---- 5) Subpixel via center-of-mass in 5x5 window
    pts = []
    for y, x, s in zip(ys, xs, scores):
        x0, x1 = max(0, x-2), min(w-1, x+2)
        y0, y1 = max(0, y-2), min(h-1, y+2)
        patch = Jf[y0:y1+1, x0:x1+1]
        py, px = np.mgrid[y0:y1+1, x0:x1+1]
        ss = float(patch.sum()) + 1e-6
        cx = float((patch*px).sum() / ss)
        cy = float((patch*py).sum() / ss)
        pts.append((cx, cy, float(s)))

    return pts  # (x,y,score)

# -------------------------- Lattice clustering ------------------------

def _cluster_1d(vals: np.ndarray, n_groups: int, band_frac: float, size_px: int):
    """Greedy 1D clustering into n_groups using bandwidth = band_frac*size."""
    vals = np.array(sorted(vals))
    band = max(1.0, band_frac * size_px)
    groups = []
    cur = [vals[0]]
    for v in vals[1:]:
        if abs(v - cur[-1]) <= band:
            cur.append(v)
        else:
            groups.append(cur); cur = [v]
    groups.append(cur)
    # if too many groups, merge nearest until n_groups
    while len(groups) > n_groups:
        dmin, idx = 1e9, -1
        for i in range(len(groups)-1):
            d = abs(np.mean(groups[i+1]) - np.mean(groups[i]))
            if d < dmin: dmin, idx = d, i
        groups[idx] = groups[idx] + groups[idx+1]
        groups.pop(idx+1)
    # if too few, split the widest group
    while len(groups) < n_groups:
        lens = [np.ptp(g) if len(g)>1 else 0 for g in groups]
        k = int(np.argmax(lens))
        g = groups[k]
        if len(g) >= 2:
            mid = len(g)//2
            groups[k] = g[:mid]
            groups.insert(k+1, g[mid:])
        else:
            groups.append(g.copy())
    centers = np.array([np.mean(g) for g in groups])
    order = np.argsort(centers)
    centers = centers[order]
    return centers

def cluster_grid(jpts: List[Tuple[float,float,float]], cfg: PPConfig, h: int, w: int):
    """Return lattice dict with 10 ordered y-levels and x-levels and a 10x10 grid of points.
       Missing points are interpolated."""
    xs = np.array([p[0] for p in jpts])
    ys = np.array([p[1] for p in jpts])
    # 10 bands in y and 10 bands in x
    ylv = _cluster_1d(ys, 10, cfg.cluster_band_y, h)
    xlv = _cluster_1d(xs, 10, cfg.cluster_band_x, w)
    # Assign each point to nearest (i,j)
    grid = [[None]*10 for _ in range(10)]
    for x,y,_ in jpts:
        i = int(np.argmin(np.abs(ylv - y)))
        j = int(np.argmin(np.abs(xlv - x)))
        if grid[i][j] is None or abs(ylv[i]-y)+abs(xlv[j]-x) < abs(ylv[i]-grid[i][j][1])+abs(xlv[j]-grid[i][j][0]):
            grid[i][j] = (x,y)
    # Fill missing by bilinear interpolation from band centers
    for i in range(10):
        for j in range(10):
            if grid[i][j] is None:
                yi = ylv[i]; xj = xlv[j]
                grid[i][j] = (float(xj), float(yi))
    return np.array(ylv), np.array(xlv), grid  # y-levels, x-levels, 10x10 [(x,y)]

# ------------------------ Path tracing (A* in band) -------------------

def _angle_cost(OxOy, dir_vec):
    """1 - cos(theta) between orientation and desired direction [0..2]."""
    ox, oy = OxOy
    dot = ox*dir_vec[0] + oy*dir_vec[1]
    return float(1.0 - max(-1.0, min(1.0, dot)))

def trace_between(p0, p1, A_map: np.ndarray, O: np.ndarray, cfg: PPConfig):
    """Constrained A* from p0->p1 on a narrow tube around the segment with cost:
       alpha*(1-A) + beta*angle_mismatch."""
    h, w = A_map.shape
    x0, y0 = p0; x1, y1 = p1
    # band mask
    cx, cy = (x0+x1)/2, (y0+y1)/2
    dx, dy = x1-x0, y1-y0
    L = max(1.0, np.hypot(dx, dy))
    nx, ny = -dy/L, dx/L
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    proj = ((xs - x0) * dx + (ys - y0) * dy) / (L*L)
    dist = np.abs((xs - x0) * ny - (ys - y0) * nx)
    band = (proj >= -0.1) & (proj <= 1.1) & (dist <= cfg.path_band_px)

    # A*, 8-neighborhood
    A = A_map.astype(np.float32)
    start = (int(round(y0)), int(round(x0)))
    goal  = (int(round(y1)), int(round(x1)))
    if not band[start] or not band[goal]:
        band[start] = True; band[goal] = True

    def hfun(p):
        y,x = p
        return math.hypot(x - goal[1], y - goal[0])

    neighbors = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    openpq = []
    g = {start: 0.0}
    came = {}
    heapq.heappush(openpq, (hfun(start), start))
    visited = set()

    while openpq:
        _, cur = heapq.heappop(openpq)
        if cur in visited: continue
        visited.add(cur)
        if cur == goal: break
        cy_, cx_ = cur
        for dy_, dx_ in neighbors:
            ny_, nx_ = cy_+dy_, cx_+dx_
            if ny_ < 0 or ny_ >= h or nx_ < 0 or nx_ >= w: continue
            if not band[ny_, nx_]: continue
            # cost
            a = A[ny_, nx_]
            alpha = cfg.path_alpha * (1.0 - a)  # prefer high A
            # desired local dir = from cur to (nx_,ny_)
            dv = np.array([nx_ - cx_, ny_ - cy_], dtype=np.float32)
            nrm = np.linalg.norm(dv) + 1e-6
            dv /= nrm
            o = O[ny_, nx_, :] if O.ndim==3 else np.array([0.0, 0.0], np.float32)
            beta = cfg.path_beta * _angle_cost(o, dv)
            step_cost = alpha + beta + (1.41 if dx_*dy_!=0 else 1.0)*0.01
            cand = g[cur] + step_cost
            nxt = (ny_, nx_)
            if cand < g.get(nxt, 1e9):
                g[nxt] = cand
                came[nxt] = cur
                heapq.heappush(openpq, (cand + hfun(nxt), nxt))

    # reconstruct
    path = []
    cur = goal
    if cur not in came:
        # fallback straight line sampling
        K = int(max(2, round(L)))
        xs = np.linspace(x0, x1, K)
        ys = np.linspace(y0, y1, K)
        return np.stack([xs, ys], axis=1)
    while cur != start:
        y,x = cur
        path.append([x,y])
        cur = came[cur]
    path.append([start[1], start[0]])
    path = np.array(path[::-1], dtype=np.float32)
    return path

# --------------------- Smooth & resample & offset ---------------------

def smooth_and_resample(poly: np.ndarray, K: int):
    if len(poly) < 5:
        # upsample linearly
        t = np.linspace(0,1,max(K,len(poly)))
        xi = np.interp(t, np.linspace(0,1,len(poly)), poly[:,0])
        yi = np.interp(t, np.linspace(0,1,len(poly)), poly[:,1])
        return np.stack([xi, yi], axis=1)[:K]
    # Savitzky–Golay smoothing
    w = min(17, len(poly) // 2 * 2 + 1)
    x = savgol_filter(poly[:,0], w, 3, mode='interp')
    y = savgol_filter(poly[:,1], w, 3, mode='interp')
    # resample by arc length
    p = np.stack([x,y], axis=1)
    d = np.sqrt(((p[1:]-p[:-1])**2).sum(axis=1))
    s = np.concatenate([[0], np.cumsum(d)])
    if s[-1] < 1e-6:
        return np.repeat(p[:1], K, axis=0)
    t = np.linspace(0, s[-1], K)
    xi = np.interp(t, s, p[:,0])
    yi = np.interp(t, s, p[:,1])
    return np.stack([xi, yi], axis=1)

def estimate_halfwidth(A_bin: np.ndarray, poly: np.ndarray, cap_px: int):
    """Sample distance transform along the curve; return median halfwidth (px)."""
    dist = cv2.distanceTransform(255 - A_bin, cv2.DIST_L2, 3)
    pts = np.round(poly).astype(int)
    pts[:,0] = np.clip(pts[:,0], 0, A_bin.shape[1]-1)
    pts[:,1] = np.clip(pts[:,1], 0, A_bin.shape[0]-1)
    vals = dist[pts[:,1], pts[:,0]]
    hw = float(np.median(vals))
    return min(hw, float(cap_px))

def offset_inward(poly: np.ndarray, margin_px: float, inward=True):
    """Offset polyline by margin along local normals. inward=True assumes
       normal points *inside* the cell when used segment-wise (you’ll pick side)."""
    p = poly
    # estimate tangents
    t = np.zeros_like(p)
    t[1:-1] = p[2:] - p[:-2]
    t[0] = p[1] - p[0]
    t[-1] = p[-1] - p[-2]
    nrm = np.sqrt((t**2).sum(axis=1, keepdims=True)) + 1e-6
    t = t / nrm
    n = np.stack([-t[:,1], t[:,0]], axis=1)  # left normal
    off = p + ( -margin_px if inward else margin_px ) * n
    return off

# --------------------------- Coons patch ------------------------------

def coons_patch_grid(top: np.ndarray, bottom: np.ndarray, left: np.ndarray, right: np.ndarray, out_size: int=96):
    """Return sampling grid (u,v)->(x,y) of shape (out_size,out_size,2)."""
    def resample(curve, K):
        return smooth_and_resample(curve, K)
    K = out_size
    T = resample(top, K); B = resample(bottom, K)
    L = resample(left, K); R = resample(right, K)
    # corners
    P00, P10 = T[0], T[-1]
    P01, P11 = B[0], B[-1]
    # build grid
    u = np.linspace(0,1,K)
    v = np.linspace(0,1,K)
    U,V = np.meshgrid(u,v)
    # bilinear corners
    BL = (1-U)*(1-V)[:,None]*P00 + U*(1-V)[:,None]*P10 + (1-U)*V[:,None]*P01 + U*V[:,None]*P11
    # Coons
    topv    = (1-V)[:,:,None]*T[None,:,:]
    bottomv = V[:,:,None]*B[None,:,:]
    leftu   = (1-U)[:,:,None]*L[:,None,:]
    rightu  = U[:,:,None]*R[:,None,:]
    S = topv + bottomv + leftu + rightu - BL
    return S.astype(np.float32)  # (K,K,2) with x,y

# ----------------------- Assemble polylines ---------------------------

def assemble_polylines(lattice_xy: List[List[Tuple[float,float]]],
                       A_map: np.ndarray, O: np.ndarray, cfg: PPConfig):
    """From 10x10 lattice of crossings, trace 10 horizontals and 10 verticals."""
    # horizontals
    H_lines = []
    for i in range(10):
        segs = []
        for j in range(9):
            p0 = lattice_xy[i][j]; p1 = lattice_xy[i][j+1]
            path = trace_between(p0, p1, A_map, O, cfg)
            segs.append(path)
        line = np.concatenate(segs, axis=0)
        H_lines.append(smooth_and_resample(line, cfg.resample_K))
    # verticals
    V_lines = []
    for j in range(10):
        segs = []
        for i in range(9):
            p0 = lattice_xy[i][j]; p1 = lattice_xy[i+1][j]
            path = trace_between(p0, p1, A_map, O, cfg)
            segs.append(path)
        line = np.concatenate(segs, axis=0)
        V_lines.append(smooth_and_resample(line, cfg.resample_K))
    return H_lines, V_lines

# ------------------------------- Demo ---------------------------------

if __name__ == "__main__":
    # Tiny demo with fake maps for shape sanity (not a full example).
    h = w = 512
    A = np.zeros((h,w), np.uint8); cv2.line(A,(50,50),(450,450),255,2)
    H = A.copy(); V = A.copy()
    J = np.zeros((h,w), np.uint8)
    for k in range(10):
        x = 50 + k*40; y = 50 + k*40
        cv2.circle(J,(x,y),2,255,-1)
    O = np.zeros((h,w,2), np.float32); O[...,0]=1.0
    cfg = PPConfig()
    A_bin, H_bin, V_bin = bin_maps(A,H,V,cfg)
    jpts = junction_nms(J, cfg)
    ylv, xlv, grid = cluster_grid(jpts, cfg, h, w)
    H_lines, V_lines = assemble_polylines(grid, A_bin.astype(np.float32)/255.0, O, cfg)
    print(f"Built {len(H_lines)} H and {len(V_lines)} V lines; J pts = {len(jpts)}")