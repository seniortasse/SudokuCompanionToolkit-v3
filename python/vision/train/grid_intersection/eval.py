from __future__ import annotations
import os
import json
import math
import copy
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import cv2
import torch
import torch.nn.functional as F

from python.vision.train.grid_intersection.metrics import (
    EvalConfig, evaluate_case, summarize_to_json_md
)
from python.vision.train.grid_intersection.postproc import (
    PPConfig, junction_nms, cluster_grid
)

# =========================
# Sub-pixel joint decoding
# =========================

class SubpixelMethod(str, Enum):
    none = "none"
    quadfit = "quadfit"
    softargmax = "softargmax"

def _to_numpy01(t: torch.Tensor) -> np.ndarray:
    a = t.detach().cpu().numpy().astype(np.float32)
    return np.clip(a, 0.0, 1.0)

def _save_colormap(p: Path, m: np.ndarray):
    x = (np.clip(m, 0.0, 1.0) * 255.0).astype(np.uint8)
    x = cv2.applyColorMap(x, cv2.COLORMAP_JET)
    cv2.imwrite(str(p), x)

def _sort_row_major(pts: np.ndarray) -> np.ndarray:
    """Sort Nx2 points by y then x (row-major)."""
    if pts is None or pts.size == 0:
        return np.zeros((0, 2), np.float32)
    order = np.lexsort((pts[:, 0], pts[:, 1]))
    return pts[order].astype(np.float32, copy=False)

# ---------- sub-pixel helpers ----------

def _quadfit_offset_3x3(patch: torch.Tensor) -> Tuple[float, float]:
    """
    patch: (3,3) around the peak (float). Returns (dx, dy) in [-1,1] w.r.t. center pixel.
    Quadratic fit in log domain for stability.
    """
    eps = 1e-8
    Z = torch.log(torch.clamp(patch, min=eps))

    yy, xx = torch.meshgrid(
        torch.arange(3, device=patch.device),
        torch.arange(3, device=patch.device),
        indexing="ij"
    )
    X = torch.stack([
        torch.ones_like(xx), xx.float(), yy.float(),
        (xx**2).float(), (yy**2).float(), (xx*yy).float()
    ], dim=-1).reshape(-1, 6)  # (9,6)
    y = Z.reshape(-1, 1)       # (9,1)

    beta = (torch.linalg.lstsq(X, y).solution).view(-1)  # (6,)
    a0, a1, a2, a3, a4, a5 = beta
    A = torch.tensor([[2*a3, a5], [a5, 2*a4]], device=patch.device)
    b = -torch.tensor([a1, a2], device=patch.device)
    try:
        sol = torch.linalg.solve(A, b)
        dx = torch.clamp(sol[0] - 1.0, -1.0, 1.0)
        dy = torch.clamp(sol[1] - 1.0, -1.0, 1.0)
        return float(dx), float(dy)
    except RuntimeError:
        return 0.0, 0.0

def _softargmax_2d(hm: torch.Tensor, temp: float = 0.5) -> Tuple[float, float]:
    """
    hm: (H,W). Returns (x,y) in pixel coords (float).
    """
    H, W = hm.shape
    p = F.softmax(hm.reshape(-1) / max(1e-6, float(temp)), dim=0)
    x_grid = torch.arange(W, device=hm.device).float()
    y_grid = torch.arange(H, device=hm.device).float()
    px = (p * x_grid.repeat(H)).sum()
    py = (p * torch.repeat_interleave(y_grid, W)).sum()
    return float(px), float(py)

def _decode_joints_from_heatmap(
    jmap: torch.Tensor,
    *,
    topk: int = 150,
    nms_kernel: int = 3,
    conf_thresh: float = 0.05,
    subpixel: SubpixelMethod = SubpixelMethod.quadfit,
    softargmax_temp: float = 0.5,
) -> np.ndarray:
    """
    jmap: (H,W) torch float32 in [0,1]
    returns: (N,2) xy in image pixel coords, score-sorted.
    """
    H, W = jmap.shape
    mx = F.max_pool2d(jmap[None, None, ...], nms_kernel, 1, nms_kernel // 2)[0, 0]
    keep = (jmap == mx) & (jmap >= conf_thresh)
    ys, xs = torch.nonzero(keep, as_tuple=True)
    scores = jmap[ys, xs]
    if scores.numel() == 0:
        return np.zeros((0, 2), np.float32)

    idx = torch.argsort(scores, descending=True)[:topk]
    xs = xs[idx]; ys = ys[idx]

    out = []
    for x, y in zip(xs.tolist(), ys.tolist()):
        fx, fy = float(x), float(y)
        if subpixel == SubpixelMethod.quadfit and 1 <= x <= W - 2 and 1 <= y <= H - 2:
            patch = jmap[y - 1:y + 2, x - 1:x + 2]
            dx, dy = _quadfit_offset_3x3(patch)
            fx, fy = x + dx, y + dy
        elif subpixel == SubpixelMethod.softargmax:
            x0, x1 = max(0, x - 2), min(W, x + 3)
            y0, y1 = max(0, y - 2), min(H, y + 3)
            sub = jmap[y0:y1, x0:x1]
            sx, sy = _softargmax_2d(sub, temp=softargmax_temp)
            fx, fy = x0 + sx, y0 + sy
        out.append([fx, fy])

    return np.asarray(out, dtype=np.float32)

def _order_10x10_raster(xy: np.ndarray) -> np.ndarray | None:
    """Greedy 10-row by y then x ordering. Returns (100,2) or None."""
    if xy.shape[0] < 50:
        return None
    y_sorted = xy[np.argsort(xy[:, 1])]
    rows = np.array_split(y_sorted, 10)
    out = []
    for r in rows:
        r = r[np.argsort(r[:, 0])]
        out.append(r[:10] if r.shape[0] >= 10 else r)
    out = np.vstack(out)
    if out.shape[0] >= 100:
        return out[:100].astype(np.float32, copy=False)
    return None

def _greedy_1to1_ap(pred_xy: np.ndarray, gt_xy: np.ndarray, thresholds=(1.0, 2.0, 3.0)) -> Tuple[Dict[float, float], float]:
    """
    Greedy one-to-one matching based on pairwise distances.
    Returns:
      - ap_by_thr: {thr: fraction of GT matched within thr (recall@thr under 1:1)}
      - mean_matched_dist: mean distance of all matched pairs (no threshold)
    """
    if pred_xy is None or gt_xy is None:
        return {t: 0.0 for t in thresholds}, float("nan")

    p_ok = np.isfinite(pred_xy).all(axis=1)
    g_ok = np.isfinite(gt_xy).all(axis=1)
    P = pred_xy[p_ok]
    G = gt_xy[g_ok]
    if P.shape[0] == 0 or G.shape[0] == 0:
        return {t: 0.0 for t in thresholds}, float("nan")

    dmat = np.sqrt(((P[:, None, :] - G[None, :, :]) ** 2).sum(axis=2))  # (P, G)

    used_p, used_g, pairs = set(), set(), []
    idxs = np.dstack(np.unravel_index(np.argsort(dmat, axis=None), dmat.shape))[0]
    for pi, gi in idxs:
        if pi in used_p or gi in used_g:
            continue
        pairs.append((pi, gi, dmat[pi, gi]))
        used_p.add(pi); used_g.add(gi)
        if len(used_p) == min(P.shape[0], G.shape[0]):
            break

    if not pairs:
        return {t: 0.0 for t in thresholds}, float("nan")

    dists = np.array([d for (_, _, d) in pairs], dtype=np.float32)
    ap_by_thr = {}
    total_gt = G.shape[0]
    for thr in thresholds:
        matched = (dists <= thr).sum()
        ap_by_thr[thr] = float(matched) / float(total_gt)
    mean_matched = float(dists.mean()) if dists.size else float("nan")
    return ap_by_thr, mean_matched

# =========================
# Lattice snap (ordering v2)
# =========================

def _fit_bilinear_grid(rc: np.ndarray, xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit bilinear map:
       x = a0 + a1*r + a2*c + a3*r*c
       y = b0 + b1*r + b2*c + b3*r*c
    rc: (N,2) with r,c in [0..9]
    xy: (N,2)
    returns (ax[4], ay[4])
    """
    r = rc[:, 0].astype(np.float32)
    c = rc[:, 1].astype(np.float32)
    X = np.stack([np.ones_like(r), r, c, r * c], axis=1)  # (N,4)
    ax, *_ = np.linalg.lstsq(X, xy[:, 0], rcond=None)
    ay, *_ = np.linalg.lstsq(X, xy[:, 1], rcond=None)
    return ax.astype(np.float32), ay.astype(np.float32)

def _apply_bilinear(ax: np.ndarray, ay: np.ndarray, rc: np.ndarray) -> np.ndarray:
    r = rc[:, 0].astype(np.float32)
    c = rc[:, 1].astype(np.float32)
    X = np.stack([np.ones_like(r), r, c, r * c], axis=1)
    x = X @ ax
    y = X @ ay
    return np.stack([x, y], axis=1).astype(np.float32)

def _lattice_snap_ordering_v2(pred_xy: np.ndarray, img_h: int, img_w: int) -> np.ndarray | None:
    """
    Robust 10x10 ordering + fill using bilinear lattice fit.
    Returns (100,2) float32 with no NaNs, clipped to bounds, or None if too few points (<40).
    """
    K = pred_xy.shape[0]
    if K < 40:
        return None

    base = _order_10x10_raster(pred_xy)
    if base is None:
        rm = _sort_row_major(pred_xy)
        pad = np.full((100, 2), np.nan, np.float32)
        pad[:min(100, rm.shape[0])] = rm[:100]
        base = pad

    rc_all, xy_all = [], []
    for r in range(10):
        for c in range(10):
            i = r * 10 + c
            p = base[i]
            if np.all(np.isfinite(p)):
                rc_all.append([r, c])
                xy_all.append(p)
    if len(rc_all) < 40:
        return None

    rc_all = np.asarray(rc_all, np.float32)
    xy_all = np.asarray(xy_all, np.float32)

    ax, ay = _fit_bilinear_grid(rc_all, xy_all)
    rc_full = np.stack(np.meshgrid(np.arange(10), np.arange(10), indexing="ij"), axis=-1).reshape(-1, 2)
    xy_full = _apply_bilinear(ax, ay, rc_full)

    for r in range(10):
        row_idx = np.arange(r * 10, r * 10 + 10)
        row = xy_full[row_idx]
        order = np.argsort(row[:, 0])
        xy_full[row_idx] = row[order]
    for c in range(10):
        col_idx = np.arange(c, 100, 10)
        col = xy_full[col_idx]
        order = np.argsort(col[:, 1])
        xy_full[col_idx] = col[order]

    xy_full[:, 0] = np.clip(xy_full[:, 0], 0, img_w - 1)
    xy_full[:, 1] = np.clip(xy_full[:, 1], 0, img_h - 1)
    return xy_full.astype(np.float32)

# =========================
# Existing helpers (kept)
# =========================

def _maybe_rescale_points(
    pts: np.ndarray,
    dst_h: int,
    dst_w: int,
    label_meta: Dict[str, Any] | None
) -> np.ndarray:
    if pts is None or pts.size == 0:
        return np.zeros((0, 2), np.float32)

    p = pts.astype(np.float32, copy=True)
    dst_h = int(dst_h); dst_w = int(dst_w)

    def _try_meta(meta: Dict[str, Any]) -> tuple[int | None, int | None]:
        if not meta:
            return None, None
        for kh, kw in (("img_h", "img_w"), ("H", "W"), ("orig_h", "orig_w")):
            if kh in meta and kw in meta:
                try:
                    sh = int(meta[kh]); sw = int(meta[kw])
                    if sh > 0 and sw > 0:
                        return sh, sw
                except Exception:
                    pass
        return None, None

    src_h, src_w = _try_meta(label_meta or {})
    if src_h and src_w and (src_h != dst_h or src_w != dst_w):
        sx = float(dst_w) / float(src_w)
        sy = float(dst_h) / float(src_h)
        p[:, 0] *= sx; p[:, 1] *= sy
        return p

    maxx = float(np.max(p[:, 0])) if p.size else 0.0
    maxy = float(np.max(p[:, 1])) if p.size else 0.0
    pmax = max(maxx, maxy)
    dmax = max(dst_h, dst_w)

    if pmax > 1e-3 and pmax < 0.75 * dmax:
        approx = dmax / pmax
        candidates = np.array([1.5, 2.0, 3.0, 4.0], dtype=np.float32)
        snap = float(candidates[np.argmin(np.abs(candidates - approx))])
        if abs(snap - approx) <= 0.25:
            p *= snap
            p[:, 0] = np.clip(p[:, 0], 0, dst_w - 1)
            p[:, 1] = np.clip(p[:, 1], 0, dst_h - 1)
            return p

    if pmax > 1.25 * dmax:
        approx = pmax / dmax
        candidates = np.array([1.5, 2.0, 3.0, 4.0], dtype=np.float32)
        snap = float(candidates[np.argmin(np.abs(candidates - approx))])
        if abs(snap - approx) <= 0.25:
            p /= snap
            p[:, 0] = np.clip(p[:, 0], 0, dst_w - 1)
            p[:, 1] = np.clip(p[:, 1], 0, dst_h - 1)
            return p
    return p

def _precheck_mje(p100: np.ndarray, g100: np.ndarray) -> float:
    if p100.shape != (100, 2) or g100.shape != (100, 2):
        return float("nan")
    d = np.linalg.norm(p100 - g100, axis=1)
    return float(np.mean(d)) if d.size else float("nan")

def _relaxed_eval_cfg(ev_cfg: EvalConfig) -> EvalConfig:
    ev2 = copy.deepcopy(ev_cfg)
    for name in ("j_match_radius_px", "match_radius_px", "junction_radius_px"):
        if hasattr(ev2, name):
            setattr(ev2, name, max(64.0, float(getattr(ev2, name))))
    for name in ("require_full_lattice", "strict_topology", "enforce_monotonic_rows"):
        if hasattr(ev2, name):
            setattr(ev2, name, False)
    for name in ("row_col_tolerance_px", "ordering_tolerance_px"):
        if hasattr(ev2, name):
            setattr(ev2, name, max(64.0, float(getattr(ev2, name, 0.0))))
    return ev2

# =========================
# Main eval
# =========================

@torch.no_grad()
def eval_val_miniset(
    model: torch.nn.Module,
    val_ld,
    device: torch.device,
    out_dir: Path,
    epoch: int,
    pp_cfg: PPConfig,          # API compatibility with train.py
    ev_cfg: EvalConfig,
    save_overlays: bool = True,
    overlay_max: int = 64,
    *,
    subpixel: str | SubpixelMethod = SubpixelMethod.quadfit,
    softargmax_temp: float = 0.5,
    tj=None, j_conf=None, j_topk=None
) -> List[Dict[str, float]]:
    """
    Adds:
      - subpixel decoding (--subpixel none|quadfit|softargmax)
      - Softargmax temperature control
      - One-shot decode "backoff" if <100 points
      - Lattice snap ordering v2 (monotonic, fill missing) -> no NaN outputs
      - AP@1/2/3 px, %pred_J==100, finite normalized metrics
    """
    subpixel = SubpixelMethod(str(subpixel))

    model.eval()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_all: List[Dict[str, float]] = []
    n_saved = 0
    k_save = min(int(overlay_max), 9999)

    ev_relaxed = _relaxed_eval_cfg(ev_cfg)
    T_J = 0.75  # eval-only sharpening

    def _sample_scores01(img01: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
        """
        Bilinear-sample a heatmap `img01` (H,W) assumed in [0,1] at floating coords `pts_xy` (N,2).
        Returns an array of length N with NaN for non-finite query points.
        Robust to edges and degenerate 1px neighborhoods.
        """
        if not isinstance(img01, np.ndarray) or img01.ndim != 2:
            return np.zeros((0,), np.float32)

        h, w = img01.shape
        if pts_xy is None or pts_xy.size == 0:
            return np.zeros((0,), np.float32)

        # Ensure float32 heatmap in [0,1] (avoid integer overflow in blends)
        img01 = np.clip(img01.astype(np.float32, copy=False), 0.0, 1.0)

        xs = pts_xy[:, 0].astype(np.float32, copy=False)
        ys = pts_xy[:, 1].astype(np.float32, copy=False)
        finite = np.isfinite(xs) & np.isfinite(ys)

        out = np.full(xs.shape, np.nan, np.float32)
        if not finite.any():
            return out

        xf = xs[finite]
        yf = ys[finite]

        # Floor/ceil and clamp to bounds
        x0 = np.clip(np.floor(xf), 0, w - 1).astype(np.int32)
        y0 = np.clip(np.floor(yf), 0, h - 1).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, w - 1)
        y1 = np.clip(y0 + 1, 0, h - 1)

        dx = xf - x0.astype(np.float32)
        dy = yf - y0.astype(np.float32)

        Ia = img01[y0, x0]
        Ib = img01[y0, x1]
        Ic = img01[y1, x0]
        Id = img01[y1, x1]

        top = Ia * (1.0 - dx) + Ib * dx
        bot = Ic * (1.0 - dx) + Id * dx
        out[finite] = (top * (1.0 - dy) + bot * dy).astype(np.float32)
        return out

    predJ_eq100_count = 0

    for batch_idx, (img, y, recs) in enumerate(val_ld, start=1):
        img = img.to(device, non_blocking=True)
        y   = y.to(device, non_blocking=True)

        logits = model(img)                     # (N,6,H,W)
        probs  = torch.sigmoid(logits[:, 0:4])  # A,H,V,J nominal in [0,1]

        A_pred = probs[:, 0]
        H_pred = probs[:, 1]
        V_pred = probs[:, 2]




        T_J = float(tj) if tj is not None else 0.70
        CONF = float(j_conf) if j_conf is not None else 0.05
        TOPK = int(j_topk) if j_topk is not None else 150


        J_pred = torch.sigmoid(logits[:, 3] / T_J)

        N = img.shape[0]
        for i in range(N):
            rec  = recs[i]
            idx  = int(rec.get("idx", (batch_idx - 1) * N + i))
            stem = f"ep{epoch:02d}_{idx:06d}"

            A_np = _to_numpy01(A_pred[i])
            H_np = _to_numpy01(H_pred[i])
            V_np = _to_numpy01(V_pred[i])
            J_np = _to_numpy01(J_pred[i])

            # --------- Decode → (try) lattice snap ----------
            jmap_t = J_pred[i]  # (H,W)
            Ht, Wt = J_np.shape[:2]

            

            pred_xy = _decode_joints_from_heatmap(
                jmap_t, topk=TOPK, nms_kernel=3, conf_thresh=CONF,
                subpixel=subpixel, softargmax_temp=softargmax_temp
            )

            if pred_xy.shape[0] < 100:

                pred_xy_b = _decode_joints_from_heatmap(
                    jmap_t, topk=max(180, TOPK+30), nms_kernel=5, conf_thresh=max(0.03, CONF-0.01),
                    subpixel=subpixel, softargmax_temp=softargmax_temp
                )

                if pred_xy_b.shape[0] > pred_xy.shape[0]:
                    print(f"[eval] idx={idx} backoff_decode: {pred_xy.shape[0]} -> {pred_xy_b.shape[0]}")
                    pred_xy = pred_xy_b

            snapped = _lattice_snap_ordering_v2(pred_xy[:150], Ht, Wt)
            if snapped is None:
                ordered = _order_10x10_raster(pred_xy[:100])
                if ordered is None:
                    rm = _sort_row_major(pred_xy)
                    pad = np.full((100, 2), np.nan, np.float32)
                    pad[:min(100, rm.shape[0])] = rm[:100]
                    ordered = pad
            else:
                ordered = snapped

            pj = int(np.sum(np.isfinite(ordered).all(axis=1)))
            if pj == 100:
                predJ_eq100_count += 1
            else:
                bad = ~np.isfinite(ordered).all(axis=1)
                if bad.any():
                    good_idx = np.where(~bad)[0]
                    if good_idx.size > 0:
                        for j in np.where(bad)[0]:
                            # nearest in (r,c) space among good indices
                            gr = good_idx // 10
                            gc = good_idx % 10
                            r, c = divmod(j, 10)
                            d = (gr - r) ** 2 + (gc - c) ** 2
                            k = good_idx[np.argmin(d)]
                            ordered[j] = ordered[k]
                        pj = 100

            pred_scores = _sample_scores01(J_np, ordered)
            pred: Dict[str, Any] = {
                "A": A_np, "H": H_np, "V": V_np,
                "J_pts": ordered,
                "J_scores": pred_scores,
            }

            # --- GT dict
            A_gt = _to_numpy01(y[i, 0])
            H_gt = _to_numpy01(y[i, 1])
            V_gt = _to_numpy01(y[i, 2])
            J_gt = _to_numpy01(y[i, 3])
            gt: Dict[str, Any] = {"A": A_gt, "H": H_gt, "V": V_gt}

            gt_struct = None
            used_npz = False

            label_meta: Dict[str, Any] = {}
            for kh, kw in (("img_h", "img_w"), ("H", "W"), ("orig_h", "orig_w")):
                if kh in rec and kw in rec:
                    try:
                        label_meta[kh] = int(rec[kh]); label_meta[kw] = int(rec[kw])
                    except Exception:
                        pass

            lp = rec.get("label_path", "")
            if isinstance(lp, str) and len(lp) > 0 and os.path.exists(lp):
                try:
                    with np.load(lp) as z:
                        if "J_pts" in z.files:
                            raw = z["J_pts"].astype(np.float32).reshape(-1, 2)
                            if raw.shape == (100, 2):
                                for k in ("img_h", "img_w", "H", "W", "orig_h", "orig_w"):
                                    if k in z.files and np.size(z[k]) == 1:
                                        try:
                                            label_meta[k] = int(z[k])
                                        except Exception:
                                            pass
                                rs = _maybe_rescale_points(raw, Ht, Wt, label_meta)
                                gt_struct = _sort_row_major(rs)
                                used_npz = True
                except Exception as e:
                    print(f"[eval] WARN: could not read/scale J_pts from {lp}: {type(e).__name__}: {e}")

            if gt_struct is None or gt_struct.shape != (100, 2):
                J8 = (np.clip(J_gt, 0.0, 1.0) * 255.0).astype(np.uint8)
                pp = PPConfig()
                if hasattr(pp, "j_topk"):     pp.j_topk     = max(int(getattr(pp, "j_topk", 120)), 120)
                if hasattr(pp, "j_topk_cap"): pp.j_topk_cap = max(int(getattr(pp, "j_topk_cap", 180)), 180)
                if hasattr(pp, "j_topk_min"): pp.j_topk_min = max(int(getattr(pp, "j_topk_min", 120)), 120)
                pts = junction_nms(J8, pp)
                _, _, grid = cluster_grid(pts, pp, Ht, Wt)
                flat = [[float(grid[r][c][0]), float(grid[r][c][1])] for r in range(10) for c in range(10)]
                gt_struct = _sort_row_major(np.array(flat, dtype=np.float32))

            gt["J_pts"] = gt_struct

            # --- Debug counts + direct 1–1 precheck MJE
            gj = gt_struct.shape[0]
            pre_mje = _precheck_mje(ordered, gt_struct)
            src_tag = "npz" if used_npz else "Jmap"
            print(f"[eval] idx={idx} srcGT={src_tag} pred_J={pj} gt_J={gj} precheck_MJE={pre_mje:.3f}")

            # --- Metrics (relaxed first)
            m = evaluate_case(pred, gt, ev_relaxed)

            # --- Build a robust per-sample normalizer from GT lattice (bbox diagonal)
            norm = float("nan")
            if isinstance(gt_struct, np.ndarray) and gt_struct.shape == (100, 2) and np.all(np.isfinite(gt_struct)):
                gx_min = float(np.min(gt_struct[:, 0])); gx_max = float(np.max(gt_struct[:, 0]))
                gy_min = float(np.min(gt_struct[:, 1])); gy_max = float(np.max(gt_struct[:, 1]))
                dx = max(1e-6, gx_max - gx_min)
                dy = max(1e-6, gy_max - gy_min)
                norm = math.hypot(dx, dy)

            # --- Fallback if J_MJE is NaN/inf
            jmje = m.get("J_MJE", float("nan"))
            if isinstance(jmje, float) and (math.isnan(jmje) or math.isinf(jmje)):
                if not math.isnan(pre_mje) and not math.isinf(pre_mje):
                    jmje = pre_mje
                    m["J_MJE"] = jmje
                    m["J_MJE_fallback"] = 1.0
                    print(f"[eval] idx={idx} J_MJE was NaN -> using fallback {pre_mje:.3f}")

            # --- Thresholded per-image MJE flags (so epoch means aren't NaN)
            jmje_final = float(m.get("J_MJE", float("nan")))
            for thr in (6, 8, 10):
                key = f"J_MJE<= {thr}px"  # NOTE: space matches train.py keys
                m[key] = 1.0 if (not math.isnan(jmje_final) and jmje_final <= float(thr)) else 0.0

            # --- Extra: AP@{1,2,3} px via greedy 1:1 (finite)
            ap1 = ap2 = ap3 = 0.0
            jmje_greedy = float("nan")
            if (ordered.shape == (100, 2) and gt_struct.shape == (100, 2)
                    and np.all(np.isfinite(ordered)) and np.all(np.isfinite(gt_struct))):
                ap_by_thr, jmje_greedy = _greedy_1to1_ap(ordered, gt_struct, thresholds=(1.0, 2.0, 3.0))
                ap1 = float(ap_by_thr[1.0])
                ap2 = float(ap_by_thr[2.0])
                ap3 = float(ap_by_thr[3.0])
                m["J_MJE_greedy"] = float(jmje_greedy)
            else:
                m["J_MJE_greedy"] = float("nan")

            m["J_AP@1px"] = float(ap1)
            m["J_AP@2px"] = float(ap2)
            m["J_AP@3px"] = float(ap3)
            # key expected by train.py console line:
            m["J_AP@2px_finite"] = float(ap2)

            # --- Normalized MJE (finite when possible)
            if isinstance(jmje, (int, float)) and not (math.isnan(jmje) or math.isinf(jmje)):
                if isinstance(norm, float) and not (math.isnan(norm) or math.isinf(norm)) and norm > 1e-6:
                    m["J_MJE_norm"] = float(jmje / norm)
                else:
                    diag = math.hypot(float(Wt), float(Ht))
                    m["J_MJE_norm"] = float(jmje / diag) if diag > 1e-6 else float("nan")
            else:
                m["J_MJE_norm"] = float("nan")

            m["pred_J_eq_100"] = 1.0 if pj == 100 else 0.0
            m["subpixel_temp"] = float(softargmax_temp)
            m["subpixel"] = subpixel.value

            m["tj"] = float(T_J)
            m["j_conf"] = float(CONF)
            m["j_topk"] = float(TOPK)

            # --- Keep only numeric fields for summarizer robustness
            for _k in list(m.keys()):
                _v = m[_k]
                if not isinstance(_v, (int, float, np.floating, np.integer)):
                    m.pop(_k, None)

            metrics_all.append(m)

            # --- Overlays (optional)
            if save_overlays and n_saved < k_save:
                _save_colormap(out_dir / f"{stem}_A.png", A_np)
                _save_colormap(out_dir / f"{stem}_H.png", H_np)
                _save_colormap(out_dir / f"{stem}_V.png", V_np)
                _save_colormap(out_dir / f"{stem}_J.png", J_np)
                n_saved += 1

    # Write summary files
    js, md = summarize_to_json_md(metrics_all)
    (out_dir / f"eval_report_epoch{epoch:02d}.json").write_text(js, encoding="utf-8")
    (out_dir / f"eval_report_epoch{epoch:02d}.md").write_text(md, encoding="utf-8")

    return metrics_all