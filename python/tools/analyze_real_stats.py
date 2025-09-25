# tools/analyze_real_stats.py
# Compute augmentation distributions from a labeled dataset (images/ + labels.jsonl)
# and emit JSON stats + suggested synthetic generator ranges.

import os, json, math, argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np

# ----------------------------- small utils -----------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_labels(labels_path: Path) -> List[Dict[str, Any]]:
    recs = []
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            fn = obj.get("file_name") or obj.get("file") or obj.get("image") or obj.get("path")
            corners = obj.get("corners", None)
            if fn is None or corners is None:
                continue
            c = np.asarray(corners, dtype=np.float32)
            if c.shape != (4, 2):
                continue
            recs.append({"file_name": fn, "corners": c, "mode": obj.get("mode")})
    if not recs:
        raise RuntimeError(f"No valid rows in {labels_path}")
    return recs

def order_tl_tr_br_bl(c: np.ndarray) -> np.ndarray:
    # sort top2/bottom2 by y, then by x inside each row
    idx = np.argsort(c[:, 1])
    top, bot = c[idx[:2]], c[idx[2:]]
    top = top[np.argsort(top[:, 0])]
    bot = bot[np.argsort(bot[:, 0])]
    tl, tr, bl, br = top[0], top[1], bot[0], bot[1]
    return np.stack([tl, tr, br, bl], 0)

def line_angle_deg(p0: np.ndarray, p1: np.ndarray) -> float:
    v = (p1 - p0).astype(np.float64)
    a = math.degrees(math.atan2(v[1], v[0]))
    # wrap to [-90,90] for easier interpretation (horizontal-ish)
    while a > 90: a -= 180
    while a < -90: a += 180
    return a

def side_len(p0: np.ndarray, p1: np.ndarray) -> float:
    v = (p1 - p0).astype(np.float64)
    return float(np.linalg.norm(v))

def homography_from_square(tl, tr, br, bl, W, H):
    # canonical square corners (0,0)-(1,1) → image px
    src = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=np.float32)
    dst = np.array([tl, tr, br, bl], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)

def percentiles(x: np.ndarray, ps=(5,10,25,50,75,90,95)):
    return {f"p{p}": float(np.percentile(x, p)) for p in ps}

def add_hist_plot(values: np.ndarray, out_path: Path, title: str):
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.hist(values, bins=40)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(str(out_path))
        plt.close()
    except Exception as e:
        # plotting is optional; silently skip if matplotlib isn't present
        pass

# ----------------------------- main analysis -----------------------------

def analyze_dataset(root: Path, out_dir: Path, make_plots: bool = True) -> Dict[str, Any]:
    img_dir = root / "images"
    labels_path = root / "labels.jsonl"
    ensure_dir(out_dir)
    plots_dir = out_dir / "plots"
    if make_plots:
        ensure_dir(plots_dir)

    recs = load_labels(labels_path)

    # accumulators
    m_brightness, s_contrast = [], []
    lap_var, sobel_med = [], []

    # illumination gradient (proxy for shadow)
    illum_grad = []

    # geometry/warp
    top_bot_ratio, left_right_ratio = [], []
    top_ang, bot_ang, left_ang, right_ang = [], [], [], []
    parallelogram_angle_err = []  # deviation from 90 deg at corners
    proj_p0, proj_p1 = [], []
    border_margins = []  # min margin to border relative to min(H,W)

    for r in recs:
        fp = img_dir / r["file_name"]
        img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        # --- brightness/contrast ---
        m_brightness.append(float(gray.mean()))
        s_contrast.append(float(gray.std()))

        # --- blur/noise proxies ---
        lap = cv2.Laplacian((gray*255).astype(np.uint8), cv2.CV_64F)
        lap_var.append(float(lap.var()))
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx*gx + gy*gy)
        sobel_med.append(float(np.median(mag)))

        # illumination gradient magnitude after heavy blur
        big = cv2.GaussianBlur(gray, (0,0), sigmaX=15, sigmaY=15)
        gy2, gx2 = np.gradient(big)
        illum_grad.append(float(np.median(np.hypot(gx2, gy2))))

        # --- geometry using labeled corners ---
        c = order_tl_tr_br_bl(np.array(r["corners"], dtype=np.float32))
        tl, tr, br, bl = c[0], c[1], c[2], c[3]

        # side lengths and ratios
        L_top = side_len(tl, tr)
        L_bot = side_len(bl, br)
        L_left = side_len(tl, bl)
        L_right = side_len(tr, br)

        if L_bot > 1e-6:
            top_bot_ratio.append(float(L_top / L_bot))
        if L_right > 1e-6:
            left_right_ratio.append(float(L_left / L_right))

        # edge angles
        top_ang.append(line_angle_deg(tl, tr))
        bot_ang.append(line_angle_deg(bl, br))
        left_ang.append(line_angle_deg(tl, bl) + 90.0)   # shift vertical to ~0
        right_ang.append(line_angle_deg(tr, br) + 90.0)

        # corner right-angle deviation (avg of 4)
        v_t = tr - tl; v_l = bl - tl
        v_b = br - bl; v_r = br - tr
        def angle(u, v):
            nu = np.linalg.norm(u); nv = np.linalg.norm(v)
            if nu < 1e-6 or nv < 1e-6: return 0.0
            cos = float(np.dot(u, v) / (nu*nv))
            cos = max(-1.0, min(1.0, cos))
            return math.degrees(math.acos(cos))
        ang_tl = angle(v_t, v_l)
        ang_tr = angle(tl - tr, br - tr)
        ang_br = angle(tr - br, bl - br)
        ang_bl = angle(br - bl, tl - bl)
        parallelogram_angle_err.append(float(
            np.mean([abs(ang_tl-90), abs(ang_tr-90), abs(ang_br-90), abs(ang_bl-90)])
        ))

        # projective strength from homography bottom row
        Hm = homography_from_square(tl, tr, br, bl, W, H)
        p0 = Hm[2,0] / max(Hm[2,2], 1e-9)
        p1 = Hm[2,1] / max(Hm[2,2], 1e-9)
        proj_p0.append(float(p0))
        proj_p1.append(float(p1))

        # border margin (closest corner to image border)
        dists = [
            min(tl[0], W-1-tl[0], tl[1], H-1-tl[1]),
            min(tr[0], W-1-tr[0], tr[1], H-1-tr[1]),
            min(br[0], W-1-br[0], br[1], H-1-br[1]),
            min(bl[0], W-1-bl[0], bl[1], H-1-bl[1]),
        ]
        margin_rel = np.array(dists).min() / max(min(H, W), 1e-6)
        border_margins.append(float(margin_rel))

    def pack(values: List[float], name: str):
        x = np.array(values, dtype=np.float32)
        out = {
            "count": int(x.size),
            "mean": float(x.mean()),
            "std": float(x.std(ddof=0)),
            "min": float(x.min()),
            "max": float(x.max()),
        }
        out.update(percentiles(x))
        return out, x

    stats: Dict[str, Any] = {}
    raw_cache: Dict[str, np.ndarray] = {}

    for label, arr in [
        ("brightness_mean", m_brightness),
        ("contrast_std", s_contrast),
        ("laplacian_var", lap_var),
        ("sobel_median", sobel_med),
        ("illum_gradient", illum_grad),
        ("ratio_top_over_bot", top_bot_ratio),
        ("ratio_left_over_right", left_right_ratio),
        ("top_edge_angle_deg", top_ang),
        ("bottom_edge_angle_deg", bot_ang),
        ("left_edge_angle_deg", left_ang),
        ("right_edge_angle_deg", right_ang),
        ("corner_right_angle_dev_deg", parallelogram_angle_err),
        ("proj_p0", proj_p0),
        ("proj_p1", proj_p1),
        ("border_margin_rel", border_margins),
    ]:
        s, x = pack(arr, label)
        stats[label] = s
        raw_cache[label] = x
        if make_plots:
            add_hist_plot(x, plots_dir / f"{label}.png", label)

    # Suggested synth configuration: use middle mass (10th-90th percentiles)
    def q(label, qlo="p10", qhi="p90"):
        return stats[label][qlo], stats[label][qhi]

    sugg = {
        "brightness_jitter":        {"range": [float(q('brightness_mean')[0]-0.07), float(q('brightness_mean')[1]+0.07)]},
        "contrast_jitter":          {"range": list(q("contrast_std"))},
        "blur_gaussian_sigma_px":   {"range": [0.0, max(0.8, float(np.interp( # map lap_var→rough sigma
                                        np.median(raw_cache["laplacian_var"]),
                                        [stats["laplacian_var"]["p10"], stats["laplacian_var"]["p90"]],
                                        [0.6, 2.2])))]},
        "shadow_strength":          {"range": list(q("illum_gradient"))},

        # Geometry:
        "perspective_p0":           {"range": list(q("proj_p0"))},
        "perspective_p1":           {"range": list(q("proj_p1"))},
        "edge_angle_top_deg":       {"range": list(q("top_edge_angle_deg"))},
        "edge_angle_left_deg":      {"range": list(q("left_edge_angle_deg"))},
        "parallelogram_angle_dev":  {"range": list(q("corner_right_angle_dev_deg"))},

        # Cropping safety (how close corners are to the border)
        "min_border_margin_rel":    {"range": [float(stats["border_margin_rel"]["p10"]), float(stats["border_margin_rel"]["p50"])]},
        # Ratios tell you unequal top/bottom or left/right lengths (perspective squash)
        "ratio_top_over_bot":       {"range": list(q("ratio_top_over_bot"))},
        "ratio_left_over_right":    {"range": list(q("ratio_left_over_right"))},
    }

    return {"stats": stats, "suggested_synth_config": sugg}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Dataset root with images/ and labels.jsonl")
    ap.add_argument("--out", type=str, default="runs/real_stats", help="Output directory for stats & plots")
    ap.add_argument("--no-plots", action="store_true", help="Disable histogram PNGs")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    ensure_dir(out_dir)

    result = analyze_dataset(root, out_dir, make_plots=(not args.no_plots))

    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(result["stats"], f, indent=2)

    with open(out_dir / "suggested_synth_config.json", "w", encoding="utf-8") as f:
        json.dump(result["suggested_synth_config"], f, indent=2)

    print(f"Saved:\n- {out_dir/'stats.json'}\n- {out_dir/'suggested_synth_config.json'}")
    if not args.no_plots:
        print(f"- Plots under {out_dir/'plots'}")

if __name__ == "__main__":
    main()