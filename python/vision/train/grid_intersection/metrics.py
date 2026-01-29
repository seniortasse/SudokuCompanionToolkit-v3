# python/vision/train/grid_intersection/metrics.py
# ------------------------------------------------------------
# Metrics for the 5-map grid-line model:
#  - Junction AP@r and Mean Junction Error (MJE)
#  - Pixel IoU / F1 / Dice for A, H, V
#  - Polyline deviation (mean, P95) w.r.t. GT curves
#  - Break count per line (from skeleton gaps)
#  - False-positive (FP) area outside an allowed grid band
#  - Aggregator that emits JSON + Markdown summaries
#
# Expected GT availability:
#  - Full synth: masks (A,H,V), junctions (100 points), optional GT polylines
#  - Real: often only junctions (recommended) and maybe sparse polylines
# The functions below gracefully accept missing GT pieces.
# ------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
import json
import math

# ----------------------------- Utilities ------------------------------

def _to_float01(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.uint8:
        return (x.astype(np.float32) / 255.0)
    return x.astype(np.float32)

def _binarize(x: np.ndarray, thr: float=0.5) -> np.ndarray:
    xf = _to_float01(x)
    return (xf >= thr).astype(np.uint8) * 255

def _safe_div(a: float, b: float, eps: float=1e-9) -> float:
    return float(a / (b + eps))

def _pairwise_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: (Na,2), b: (Nb,2) -> (Na,Nb)
    da = a[:, None, :] - b[None, :, :]
    return np.sqrt((da ** 2).sum(axis=2))

# ----------------------------- Junctions ------------------------------

def match_junctions(pred_xy: np.ndarray,
                    gt_xy: np.ndarray,
                    r_px: float) -> Tuple[int,int,int, np.ndarray]:
    """
    Greedy bipartite match within r_px.
    Returns: (TP, FP, FN, per-gt error vector [matched gt only])
    """
    if len(gt_xy) == 0 and len(pred_xy) == 0:
        return 0,0,0, np.array([], np.float32)
    if len(gt_xy) == 0:
        return 0, len(pred_xy), 0, np.array([], np.float32)
    if len(pred_xy) == 0:
        return 0, 0, len(gt_xy), np.array([], np.float32)

    D = _pairwise_l2(pred_xy, gt_xy)
    Na, Nb = D.shape
    # Greedy: smallest distance first
    pairs = []
    used_a = set()
    used_b = set()
    flat = [(D[i,j], i, j) for i in range(Na) for j in range(Nb)]
    flat.sort(key=lambda t: t[0])
    for d,i,j in flat:
        if d > r_px: break
        if i in used_a or j in used_b: continue
        used_a.add(i); used_b.add(j)
        pairs.append((i,j,d))
    TP = len(pairs)
    FP = Na - TP
    FN = Nb - TP
    errs = np.array([d for (_,_,d) in pairs], np.float32)
    return TP, FP, FN, errs

def junction_ap_curve(pred_xy: np.ndarray,
                      gt_xy: np.ndarray,
                      radii_px: List[float]) -> Dict[str, float]:
    """AP@r via F1 as proxy (precision/recall at fixed distance threshold)."""
    out = {}
    for r in radii_px:
        TP, FP, FN, _ = match_junctions(pred_xy, gt_xy, r)
        prec = _safe_div(TP, TP + FP)
        rec  = _safe_div(TP, TP + FN)
        f1   = _safe_div(2*prec*rec, prec + rec)
        out[f"J_AP@{int(r)}px"] = f1   # Using F1 as AP proxy at fixed r
    return out

def mean_junction_error(pred_xy: np.ndarray,
                        gt_xy: np.ndarray,
                        r_px: float=3.0) -> float:
    """Mean error over matched pairs only (<= r_px). Unmatched are ignored."""
    _, _, _, errs = match_junctions(pred_xy, gt_xy, r_px)
    if errs.size == 0:
        return float("nan")
    return float(errs.mean())

# ----------------------------- Masks ----------------------------------

def iou_f1_dice(pred_bin: np.ndarray, gt_bin: np.ndarray) -> Dict[str, float]:
    pred = (pred_bin > 0).astype(np.uint8)
    gt   = (gt_bin   > 0).astype(np.uint8)
    inter = int((pred & gt).sum())
    union = int((pred | gt).sum())
    iou = _safe_div(inter, union)
    tp = inter
    fp = int((pred & (1-gt)).sum())
    fn = int(((1-pred) & gt).sum())
    prec = _safe_div(tp, tp+fp)
    rec  = _safe_div(tp, tp+fn)
    f1   = _safe_div(2*prec*rec, prec+rec)
    # Dice = 2*inter / (|pred|+|gt|)
    dice = _safe_div(2*inter, int(pred.sum()+gt.sum()))
    return {"IoU": iou, "F1": f1, "Dice": dice}

# -------------------------- Polyline metrics --------------------------

def polyline_deviation(pred_xy: np.ndarray,
                       gt_xy: np.ndarray,
                       K: int=128) -> Dict[str, float]:
    """
    Resample both polylines to K points by arc length, then compute
    pointwise distances and aggregate mean / P95.
    """
    def resample_arclen(P, K):
        if len(P) < 2:
            return np.repeat(P[:1], K, axis=0)
        d = np.sqrt(((P[1:]-P[:-1])**2).sum(axis=1))
        s = np.concatenate([[0], np.cumsum(d)])
        if s[-1] < 1e-6:
            return np.repeat(P[:1], K, axis=0)
        t = np.linspace(0, s[-1], K)
        x = np.interp(t, s, P[:,0]); y = np.interp(t, s, P[:,1])
        return np.stack([x,y], axis=1)

    P = resample_arclen(pred_xy.astype(np.float32), K)
    G = resample_arclen(gt_xy.astype(np.float32),   K)
    d = np.sqrt(((P-G)**2).sum(axis=1))
    return {"poly_mean": float(d.mean()),
            "poly_p95":  float(np.percentile(d, 95))}

def break_count_from_skeleton(pred_bin: np.ndarray) -> int:
    """
    Approximate 'breaks' as number of connected components minus 1
    on a thinned line mask (ignores tiny specks).
    """
    # OpenCV doesn't ship Zhang-Suen skeleton; approximate by thinning with morphology
    skel = np.zeros_like(pred_bin)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    done = False
    img = (pred_bin>0).astype(np.uint8)*255
    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        done = (cv2.countNonZero(img) == 0)

    # clean tiny specks
    num, labels = cv2.connectedComponents((skel>0).astype(np.uint8))
    # 'breaks' = components - 1, ignoring empty background and very small noise
    breaks = max(0, num - 2)  # background + main skeleton → subtract 2
    return breaks

def false_positive_rate(pred_bin: np.ndarray,
                        allowed_band: Optional[np.ndarray]) -> float:
    """
    FP area outside an allowed band, as fraction of total predicted positive area.
    If allowed_band is None, returns 0.0.
    """
    pred = (pred_bin>0).astype(np.uint8)
    pos = pred.sum()
    if pos == 0 or allowed_band is None:
        return 0.0
    outside = (pred & (1 - (allowed_band>0).astype(np.uint8))).sum()
    return _safe_div(outside, pos)

# ---------------------------- Aggregator ------------------------------

@dataclass
class EvalConfig:
    # thresholds & radii
    bin_thr: float = 0.5
    j_ap_radii: Tuple[int,int,int] = (1,2,3)  # px
    mje_radius: float = 3.0
    poly_K: int = 128

def evaluate_case(pred: Dict[str,np.ndarray],
                  gt: Dict[str, np.ndarray],
                  cfg: EvalConfig) -> Dict[str, float]:
    """
    pred: dict with keys possible: A,H,V,J_pts (Nx2), H_lines(list[Nx2]), V_lines(list[Nx2])
          where A/H/V are float/uint8 maps in model space.
    gt:   dict with any subset of: A,H,V, J_pts (Nx2), H_lines(list), V_lines(list)
    """
    metrics = {}

    # Junctions
    if "J_pts" in pred and "J_pts" in gt:
        j_pred = np.array(pred["J_pts"], dtype=np.float32).reshape(-1,2)
        j_gt   = np.array(gt["J_pts"],   dtype=np.float32).reshape(-1,2)
        for r in cfg.j_ap_radii:
            metrics.update(junction_ap_curve(j_pred, j_gt, [r]))
        metrics["J_MJE"] = mean_junction_error(j_pred, j_gt, cfg.mje_radius)

    # Masks
    for key in ["A","H","V"]:
        if key in pred and key in gt and gt[key] is not None:
            pb = _binarize(pred[key], cfg.bin_thr)
            gb = _binarize(gt[key],   0.5)
            scores = iou_f1_dice(pb, gb)
            for k,v in scores.items():
                metrics[f"{key}_{k}"] = v

    # Polylines (if both sides provide them)
    if "H_lines" in pred and "H_lines" in gt:
        devs = []
        for pi,gi in zip(pred["H_lines"], gt["H_lines"]):
            devs.append(polyline_deviation(np.asarray(pi), np.asarray(gi), K=cfg.poly_K))
        if devs:
            metrics["H_poly_mean"] = float(np.mean([d["poly_mean"] for d in devs]))
            metrics["H_poly_p95"]  = float(np.mean([d["poly_p95"]  for d in devs]))
    if "V_lines" in pred and "V_lines" in gt:
        devs = []
        for pj,gj in zip(pred["V_lines"], gt["V_lines"]):
            devs.append(polyline_deviation(np.asarray(pj), np.asarray(gj), K=cfg.poly_K))
        if devs:
            metrics["V_poly_mean"] = float(np.mean([d["poly_mean"] for d in devs]))
            metrics["V_poly_p95"]  = float(np.mean([d["poly_p95"]  for d in devs]))

    # Break count (approx) — apply to A
    if "A" in pred:
        pb = _binarize(pred["A"], cfg.bin_thr)
        metrics["A_breaks"] = float(break_count_from_skeleton(pb))

    # False positives outside band (optional)
    if "A" in pred and "allowed_band" in gt and gt["allowed_band"] is not None:
        pb = _binarize(pred["A"], cfg.bin_thr)
        metrics["A_fp_outside"] = false_positive_rate(pb, gt["allowed_band"])

    return metrics



def summarize_to_json_md(all_case_metrics):
    """
    Aggregate a list of per-case metrics into summary JSON & Markdown strings.
    - Numeric-only aggregation (ints/floats); non-numerics are skipped and counted.
    - Mean is computed over finite values only.
    - Reports N total, N_finite, N_NaN (within numeric), and N_non_numeric.
    Returns (json_str, md_str).
    """
    import json
    import math
    import numpy as np

    def _is_number(x):
        return isinstance(x, (int, float, np.integer, np.floating))

    # collect keys
    keys = sorted({k for m in all_case_metrics for k in m.keys()})

    agg = {}
    lines = []
    lines.append("| Metric | Mean | N_total | N_finite | N_NaN | N_non_numeric |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    for k in keys:
        raw_vals = [m[k] for m in all_case_metrics if k in m]

        numeric_vals = []
        n_non_numeric = 0
        for v in raw_vals:
            if _is_number(v):
                numeric_vals.append(float(v))
            else:
                n_non_numeric += 1

        if numeric_vals:
            arr = np.array(numeric_vals, dtype=np.float32)
            finite = np.isfinite(arr)
            n_total = len(raw_vals)
            n_finite = int(finite.sum())
            n_nan = int((~finite).sum())
            mean = float(arr[finite].mean()) if n_finite > 0 else float("nan")
        else:
            n_total = len(raw_vals)
            n_finite = 0
            n_nan = 0
            mean = float("nan")

        agg[k] = {
            "mean": None if not math.isfinite(mean) else mean,
            "n": int(n_total),
            "n_finite": int(n_finite),
            "n_nan": int(n_nan),
            "non_numeric": int(n_non_numeric),
        }

        mean_str = f"{mean:.4f}" if math.isfinite(mean) else "—"
        lines.append(f"| {k} | {mean_str} | {n_total} | {n_finite} | {n_nan} | {n_non_numeric} |")

    json_str = json.dumps(agg, indent=2)
    md_str = "\n".join(lines)
    return json_str, md_str




# ----------------------------- Demo -----------------------------------

if __name__ == "__main__":
    # Tiny synthetic demo to show usage
    h=w=512
    gtJ = np.stack([np.linspace(80,430,10).repeat(10),
                    np.tile(np.linspace(80,430,10),10)], axis=1)
    prJ = gtJ + np.random.randn(*gtJ.shape).astype(np.float32)*0.6

    pred = {"J_pts": prJ}
    gt   = {"J_pts": gtJ}

    cfg = EvalConfig()
    m = evaluate_case(pred, gt, cfg)
    js, md = summarize_to_json_md([m])
    print(js)
    print(md)