# tools/grid_intersection/synth_renderer.py
# ------------------------------------------------------------
# Synthetic Sudoku grid renderer producing:
#  - A,H,V,J maps (uint8 0..255)
#  - O = (Ox,Oy) orientation field (float32, unit vectors)
#  - Rendered image with page-like artifacts (grayscale by default).
#  - J_pts = 100 junction coordinates (float32, in image space)
#
# Writes paired image/label files and JSONL manifests.
# ------------------------------------------------------------

from dataclasses import dataclass, asdict
from pathlib import Path
import argparse, json, math
import numpy as np
import cv2
from typing import List, Tuple

# ------------------------------- Config -------------------------------

@dataclass
class SynthConfig:
    out_root: str = "datasets/grids/synth"
    n_train: int = 1000
    n_val: int = 200
    img_size: int = 768

    # Geometry & line model
    jitter_px: float = 0.6
    curve_amp_px: float = 2.5
    curve_amp_jitter: float = 2.0
    base_inner_px: int = 3
    base_border_px: int = 6
    thin_line_p: float = 0.25
    very_thin_p: float = 0.10
    faint_line_p: float = 0.25
    border_bias_px: int = 2

    # Lighting & artifacts
    blur_p: float = 0.35
    noise_p: float = 0.35
    vignette_p: float = 0.4
    shadow_p: float = 0.30
    occluder_density: float = 0.6
    grad_strength: float = 0.25
    jpeg_q_min: int = 70
    jpeg_q_max: int = 95
    grayscale: bool = True

    # Warps
    persp_strength: float = 0.010
    elastic_alpha: float = 12.0
    elastic_sigma: float = 5.0

    # Junction map
    gauss_j_sigma: float = 1.2

    # Header bands
    header_band_p: float = 0.35
    header_halfwidth_p: float = 0.4
    header_h_min_px: int = 10
    header_h_max_px: int = 60
    header_gray_min: int = 140
    header_gray_max: int = 220

    # RNG
    seed: int = 1337


# ---------------------------- Utilities -------------------------------

def _rng(seed=None):
    return np.random.default_rng(seed)

def _make_dirs(root: Path):
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    (root / "manifests").mkdir(parents=True, exist_ok=True)

def _draw_polyline(img, pts, thickness, color, lineType=cv2.LINE_AA):
    for i in range(len(pts) - 1):
        p0 = tuple(np.round(pts[i]).astype(int))
        p1 = tuple(np.round(pts[i + 1]).astype(int))
        cv2.line(img, p0, p1, color, thickness=thickness, lineType=lineType)

def _gaussian_disk(h, w, center, sigma):
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = center
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    g = (255 * g / (g.max() + 1e-6)).astype(np.uint8)
    return g

def _elastic_deform(img, alpha, sigma, rng):
    h, w = img.shape[:2]
    dx = cv2.GaussianBlur((rng.random((h, w)) * 2 - 1), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((rng.random((h, w)) * 2 - 1), (0, 0), sigma) * alpha
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (xx + dx).astype(np.float32)
    map_y = (yy + dy).astype(np.float32)
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)

def _apply_vignette(img, rng):
    h, w = img.shape[:2]
    kx = cv2.getGaussianKernel(w, w / 2.5)
    ky = cv2.getGaussianKernel(h, h / 2.5)
    mask = ky @ kx.T
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-6)
    imgf = img.astype(np.float32)
    v = 0.65 + 0.35 * mask
    if img.ndim == 3:
        v = np.stack([v] * 3, axis=2)
    out = (imgf * v).clip(0, 255).astype(img.dtype)
    return out

def _apply_shadow(img, rng, occluder_density):
    h, w = img.shape[:2]
    n_shapes = 1 + int(occluder_density * 3)
    out = img.copy()
    for _ in range(n_shapes):
        n = rng.integers(3, 9)
        poly = np.stack([rng.uniform(0, w, n), rng.uniform(0, h, n)], axis=1).astype(np.int32)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        ksz = int(max(31, 0.05 * min(h, w)) // 2 * 2 + 1)
        mask = cv2.GaussianBlur(mask, (ksz, ksz), ksz * 0.15)
        alpha = rng.uniform(0.35, 0.8)
        if out.ndim == 2:
            out[mask > 0] = (out[mask > 0].astype(np.float32) * alpha).astype(np.uint8)
        else:
            out[mask > 0, :] = (out[mask > 0, :].astype(np.float32) * alpha).astype(np.uint8)
    return out

def _apply_linear_gradient(img, rng, strength: float):
    if strength <= 0:
        return img
    h, w = img.shape[:2]
    theta = rng.uniform(0, 2 * math.pi)
    dx, dy = math.cos(theta), math.sin(theta)
    yy, xx = np.mgrid[0:h, 0:w]
    xx = (xx / (w - 1) - 0.5) * 2
    yy = (yy / (h - 1) - 0.5) * 2
    g = (xx * dx + yy * dy)
    g = (g - g.min()) / (g.max() - g.min() + 1e-6)
    mul = 1.0 - strength * g
    imgf = img.astype(np.float32)
    if img.ndim == 3:
        mul = np.stack([mul] * 3, axis=2)
    out = (imgf * mul).clip(0, 255).astype(img.dtype)
    return out

def _jpeg_artifacts(img, q, grayscale):
    flag = [cv2.IMWRITE_JPEG_QUALITY, int(q)]
    _, enc = cv2.imencode(".jpg", img, flag)
    dec = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
    return dec


# ---------------------- Core rendering pipeline -----------------------

def _make_grid_geometry(cfg: SynthConfig, rng):
    s = cfg.img_size
    pad = int(0.08 * s)
    x0, y0 = pad, pad
    x1, y1 = s - pad, s - pad

    if rng.random() < 0.35:
        aspect = rng.uniform(0.90, 1.10)
        if aspect >= 1.0:
            x1 = int(x0 + (y1 - y0) * aspect); x1 = min(x1, s - pad)
        else:
            y1 = int(y0 + (x1 - x0) / aspect); y1 = min(y1, s - pad)

    ys = np.linspace(y0, y1, 10) + rng.normal(0, cfg.jitter_px, size=10)
    xs = np.linspace(x0, x1, 10) + rng.normal(0, cfg.jitter_px, size=10)

    K = 64
    tt = np.linspace(0, 1, K)
    H_lines, V_lines = [], []
    amp = max(0.0, cfg.curve_amp_px + rng.normal(0, cfg.curve_amp_jitter))

    for i in range(10):
        yi = ys[i]
        xs_curve = x0 + (x1 - x0) * tt
        yi_curve = yi + amp * np.sin(2 * math.pi * (tt + rng.uniform(0, 1)) * rng.integers(1, 3))
        H_lines.append(np.stack([xs_curve, yi_curve], axis=1).astype(np.float32))

        xi = xs[i]
        ys_curve = y0 + (y1 - y0) * tt
        xi_curve = xi + amp * np.sin(2 * math.pi * (tt + rng.uniform(0, 1)) * rng.integers(1, 3))
        V_lines.append(np.stack([xi_curve, ys_curve], axis=1).astype(np.float32))

    return H_lines, V_lines


def _sample_line_styles(cfg: SynthConfig, rng):
    th_h, th_v = [], []
    ink_h, ink_v = [], []

    def sample_one(is_border: bool):
        base = cfg.base_border_px if is_border else cfg.base_inner_px
        t = max(1, int(base + rng.integers(-1, 2)))
        if not is_border:
            if rng.random() < cfg.thin_line_p:
                t = max(1, int(t * rng.uniform(0.25, 0.7)))
                if rng.random() < cfg.very_thin_p:
                    t = max(1, int(t * rng.uniform(0.3, 0.6)))
        if is_border:
            t += cfg.border_bias_px
        ink = rng.integers(15, 60)
        if not is_border and rng.random() < cfg.faint_line_p:
            ink = rng.integers(80, 190)
        return t, int(ink)

    for i in range(10):
        t, ink = sample_one(i in (0, 9))
        th_h.append(t); ink_h.append(ink)
    for j in range(10):
        t, ink = sample_one(j in (0, 9))
        th_v.append(t); ink_v.append(ink)

    return th_h, th_v, ink_h, ink_v


def _rasterize_lines(h, w, H_lines, V_lines, cfg: SynthConfig, rng):
    A = np.zeros((h, w), np.uint8)
    Hm = np.zeros((h, w), np.uint8)
    Vm = np.zeros((h, w), np.uint8)

    th_h, th_v, ink_h, ink_v = _sample_line_styles(cfg, rng)

    for i, pl in enumerate(H_lines):
        _draw_polyline(Hm, pl, thickness=int(th_h[i]), color=255)
    for j, pl in enumerate(V_lines):
        _draw_polyline(Vm, pl, thickness=int(th_v[j]), color=255)

    A = np.maximum(Hm, Vm)

    dist = cv2.distanceTransform(255 - A, cv2.DIST_L2, 3)
    dist = cv2.GaussianBlur(dist, (0, 0), 1.0)
    gx = cv2.Sobel(dist, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(dist, cv2.CV_32F, 0, 1, ksize=3)
    Ox, Oy = -gy, gx
    mag = np.sqrt(Ox * Ox + Oy * Oy) + 1e-6
    Ox /= mag; Oy /= mag
    O = np.stack([Ox, Oy], axis=2).astype(np.float32)

    return A, Hm, Vm, O, (th_h, th_v, ink_h, ink_v)


def _junction_map(h, w, H_lines, V_lines, sigma):
    K = len(H_lines[0])
    J = np.zeros((h, w), np.uint8)
    for i in range(10):
        for j in range(10):
            ph = H_lines[i][K // 2]
            pv = V_lines[j][K // 2]
            cx, cy = (ph[0] + pv[0]) * 0.5, (ph[1] + pv[1]) * 0.5
            g = _gaussian_disk(h, w, (cx, cy), sigma)
            J = np.maximum(J, g)
    return J


def _render_base_image(h, w, H_lines, V_lines, style, grayscale=True):
    th_h, th_v, ink_h, ink_v = style
    paper = np.full((h, w, 3), 242, np.uint8)
    canvas = paper.copy()

    for i, pl in enumerate(H_lines):
        col = (ink_h[i],) * 3
        _draw_polyline(canvas, pl, thickness=int(th_h[i]), color=col)
    for j, pl in enumerate(V_lines):
        col = (ink_v[j],) * 3
        _draw_polyline(canvas, pl, thickness=int(th_v[j]), color=col)

    noise = np.random.default_rng().normal(0, 7, size=canvas.shape).astype(np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    if grayscale:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    return canvas


def _add_header_band(img, cfg: SynthConfig, rng):
    if rng.random() >= cfg.header_band_p:
        return img, None

    h, w = img.shape[:2]
    bh = int(rng.integers(cfg.header_h_min_px, cfg.header_h_max_px + 1))
    x0, x1 = 0, w
    if rng.random() < cfg.header_halfwidth_p:
        if rng.random() < 0.5:
            x1 = w // 2 + rng.integers(-w // 8, w // 8 + 1)
        else:
            x0 = w // 2 + rng.integers(-w // 8, w // 8 + 1)

    gray = int(rng.integers(cfg.header_gray_min, cfg.header_gray_max + 1))
    out = img.copy()
    cv2.rectangle(out, (x0, 0), (x1, bh), gray, thickness=-1)
    return out, {"x0": int(x0), "x1": int(x1), "h": int(bh), "gray": int(gray)}


def _apply_global_warps(img, A, Hm, Vm, J, O, cfg: SynthConfig, rng):
    h, w = img.shape[:2]
    if cfg.persp_strength > 0:
        m = cfg.persp_strength
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = src + rng.uniform(-m * w, m * w, size=(4, 2)).astype(np.float32)
        P = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, P, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        A   = cv2.warpPerspective(A,   P, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        Hm  = cv2.warpPerspective(Hm,  P, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        Vm  = cv2.warpPerspective(Vm,  P, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        J   = cv2.warpPerspective(J,   P, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        Ox = cv2.warpPerspective(O[..., 0], P, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        Oy = cv2.warpPerspective(O[..., 1], P, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        mag = np.sqrt(Ox * Ox + Oy * Oy) + 1e-6
        O = np.stack([Ox / mag, Oy / mag], axis=2)

    if cfg.elastic_alpha > 0 and cfg.elastic_sigma > 0:
        img = _elastic_deform(img, cfg.elastic_alpha, cfg.elastic_sigma, _rng(rng.integers(1 << 30)))
        A   = _elastic_deform(A,   cfg.elastic_alpha, cfg.elastic_sigma, _rng(rng.integers(1 << 30)))
        Hm  = _elastic_deform(Hm,  cfg.elastic_alpha, cfg.elastic_sigma, _rng(rng.integers(1 << 30)))
        Vm  = _elastic_deform(Vm,  cfg.elastic_alpha, cfg.elastic_sigma, _rng(rng.integers(1 << 30)))
        J   = _elastic_deform(J,   cfg.elastic_alpha, cfg.elastic_sigma, _rng(rng.integers(1 << 30)))
        Ox  = _elastic_deform(O[..., 0], cfg.elastic_alpha, cfg.elastic_sigma, _rng(rng.integers(1 << 30)))
        Oy  = _elastic_deform(O[..., 1], cfg.elastic_alpha, cfg.elastic_sigma, _rng(rng.integers(1 << 30)))
        mag = np.sqrt(Ox * Ox + Oy * Oy) + 1e-6
        O = np.stack([Ox / mag, Oy / mag], axis=2).astype(np.float32)

    return img, A, Hm, Vm, J, O


def _final_augs(img, cfg: SynthConfig, rng):
    img = _apply_linear_gradient(img, rng, cfg.grad_strength)

    if rng.random() < cfg.blur_p:
        k = int(rng.integers(3, 7)); img = cv2.GaussianBlur(img, (k | 1, k | 1), 0)
    if rng.random() < cfg.noise_p:
        noise = rng.normal(0, 5, size=img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < cfg.vignette_p:
        img = _apply_vignette(img, rng)
    if rng.random() < cfg.shadow_p:
        img = _apply_shadow(img, rng, cfg.occluder_density)
    q = int(np.clip(rng.integers(cfg.jpeg_q_min, cfg.jpeg_q_max + 1), 1, 100))
    img = _jpeg_artifacts(img, q, grayscale=img.ndim == 2)
    return img

# --------------------- Junction points extraction ---------------------

def _extract_junction_points(J: np.ndarray, expect_n: int = 100) -> np.ndarray:
    """
    Extract approximate junction centers from the already-warped J map.
    Returns (N,2) float32, sorted top-to-bottom, left-to-right.
    """
    J8 = J.astype(np.uint8)
    if J8.max() == 0:
        return np.zeros((0, 2), np.float32)

    thr = max(30, int(0.3 * int(J8.max())))
    _, bw = cv2.threshold(J8, thr, 255, cv2.THRESH_BINARY)
    num, labels = cv2.connectedComponents(bw)

    pts = []
    for k in range(1, num):
        ys, xs = np.where(labels == k)
        if ys.size == 0:
            continue
        w = J8[ys, xs].astype(np.float32)
        cx = float((xs * w).sum() / (w.sum() + 1e-6))
        cy = float((ys * w).sum() / (w.sum() + 1e-6))
        pts.append((cx, cy))

    if len(pts) != expect_n:
        # fallback: use goodFeaturesToTrack to reach up to expect_n salient peaks
        corners = cv2.goodFeaturesToTrack(J8, maxCorners=expect_n, qualityLevel=0.01,
                                          minDistance=max(1, J8.shape[0] // 20))
        if corners is not None:
            extra = [(float(x), float(y)) for [[x, y]] in corners]
            pts = extra  # replace; they’re usually close to the true centers

    if not pts:
        return np.zeros((0, 2), np.float32)

    arr = np.array(pts, dtype=np.float32).reshape(-1, 2)
    # Sort row-major (y then x) for stable ordering
    order = np.lexsort((arr[:, 0], arr[:, 1]))
    return arr[order]

# ------------------------------- Public API ---------------------------

def generate_sample(cfg: SynthConfig, idx: int, rng):
    h = w = cfg.img_size
    H_lines, V_lines = _make_grid_geometry(cfg, rng)

    A, Hm, Vm, O, style = _rasterize_lines(h, w, H_lines, V_lines, cfg, rng)
    J = _junction_map(h, w, H_lines, V_lines, cfg.gauss_j_sigma)

    img = _render_base_image(h, w, H_lines, V_lines, style, grayscale=cfg.grayscale)
    img, header_meta = _add_header_band(img, cfg, rng)

    # Apply global warps to image + labels
    img, A, Hm, Vm, J, O = _apply_global_warps(img, A, Hm, Vm, J, O, cfg, rng)

    # Photographic augs
    img = _final_augs(img, cfg, rng)

    # Slight label dilations
    dil = 1
    A  = cv2.dilate(A,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil * 2 + 1, dil * 2 + 1)))
    Hm = cv2.dilate(Hm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil * 2 + 1, dil * 2 + 1)))
    Vm = cv2.dilate(Vm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil * 2 + 1, dil * 2 + 1)))

    # NEW: extract junction points AFTER warps (so GT is perfectly aligned)
    J_pts = _extract_junction_points(J, expect_n=100)  # (N,2) float32

    th_h, th_v, ink_h, ink_v = style
    log = {
        "curv_amp": float(cfg.curve_amp_px),
        "header": header_meta,
        "min_th": int(min(min(th_h), min(th_v))),
        "max_th": int(max(max(th_h), max(th_v))),
        "min_ink": int(min(min(ink_h), min(ink_v))),
        "max_ink": int(max(max(ink_h), max(ink_v))),
    }
    return img, A, Hm, Vm, J, O, J_pts, log


def save_sample(root: Path, idx: int, img, A, H, V, J, O, J_pts, grayscale=True):
    img_name = f"{idx:06d}.jpg"
    lab_name = f"{idx:06d}.npz"
    img_p = root / "images" / img_name
    lab_p = root / "labels" / lab_name
    cv2.imwrite(str(img_p), img)
    np.savez_compressed(str(lab_p), A=A, H=H, V=V, J=J, O=O.astype(np.float32),
                        J_pts=J_pts.astype(np.float32))
    return str(img_p), str(lab_p)


def write_manifest(manifest_path: Path, records):
    with open(manifest_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_generate(cfg: SynthConfig):
    rng_master = _rng(cfg.seed)
    root = Path(cfg.out_root)
    _make_dirs(root)

    recs_train, recs_val = [], []

    total = cfg.n_train + cfg.n_val
    print(f"[synth] out_root={root}  size={cfg.img_size}  seed={cfg.seed}")
    print(f"[synth] n_train={cfg.n_train}  n_val={cfg.n_val}")
    print("[synth] starting sample generation ...")

    for i in range(total):
        img, A, H, V, J, O, J_pts, slog = generate_sample(cfg, i, _rng(rng_master.integers(1 << 30)))
        ip, lp = save_sample(root, i, img, A, H, V, J, O, J_pts, grayscale=cfg.grayscale)

        split = "train" if i < cfg.n_train else "val"
        rec = {
            "image_path": ip,
            "label_path": lp,
            "split": split,
            "height": cfg.img_size,
            "width": cfg.img_size,
            "meta": {"source": "synth", "log": slog}
        }
        (recs_train if split == "train" else recs_val).append(rec)

        if (i + 1) % 25 == 0 or (i + 1) == total:
            hb = slog["header"]
            hdr = "none" if hb is None else f"h={hb['h']}, x0={hb['x0']}, x1={hb['x1']}, gray={hb['gray']}"
            print(f"[synth] {i+1:5d}/{total}  split={split:5s}  "
                  f"th[min,max]={slog['min_th']},{slog['max_th']}  "
                  f"ink[min,max]={slog['min_ink']},{slog['max_ink']}  "
                  f"header={hdr}")

    write_manifest(root / "manifests" / "train_synth.jsonl", recs_train)
    write_manifest(root / "manifests" / "val_synth.jsonl", recs_val)
    print(f"[synth] wrote train manifest: {root/'manifests'/'train_synth.jsonl'} ({len(recs_train)} recs)")
    print(f"[synth] wrote   val manifest: {root/'manifests'/'val_synth.jsonl'} ({len(recs_val)} recs)")
    print("[synth] done.")

# ------------------------------- CLI ---------------------------------

def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="datasets/grids/synth")
    ap.add_argument("--n_train", type=int, default=1000)
    ap.add_argument("--n_val", type=int, default=200)
    ap.add_argument("--img_size", type=int, default=768)
    ap.add_argument("--seed", type=int, default=1337)

    # Geometry
    ap.add_argument("--jitter_px", type=float, default=0.6)
    ap.add_argument("--curve_amp_px", type=float, default=2.5)
    ap.add_argument("--curve_amp_jitter", type=float, default=2.0)

    ap.add_argument("--base_inner_px", type=int, default=3)
    ap.add_argument("--base_border_px", type=int, default=6)
    ap.add_argument("--border_bias_px", type=int, default=2)
    ap.add_argument("--thin_line_p", type=float, default=0.25)
    ap.add_argument("--very_thin_p", type=float, default=0.10)
    ap.add_argument("--faint_line_p", type=float, default=0.25)

    # Lighting & artifacts
    ap.add_argument("--blur_p", type=float, default=0.35)
    ap.add_argument("--noise_p", type=float, default=0.35)
    ap.add_argument("--vignette_p", type=float, default=0.4)
    ap.add_argument("--shadow_p", type=float, default=0.30)
    ap.add_argument("--occluder_density", type=float, default=0.6)
    ap.add_argument("--grad_strength", type=float, default=0.25)
    ap.add_argument("--jpeg_q_min", type=int, default=70)
    ap.add_argument("--jpeg_q_max", type=int, default=95)
    ap.add_argument("--grayscale", action="store_true")

    # Warps
    ap.add_argument("--persp_strength", type=float, default=0.010)
    ap.add_argument("--elastic_alpha", type=float, default=12.0)
    ap.add_argument("--elastic_sigma", type=float, default=5.0)

    # Junction sigma
    ap.add_argument("--gauss_j_sigma", type=float, default=1.2)

    # Headers
    ap.add_argument("--header_band_p", type=float, default=0.35)
    ap.add_argument("--header_halfwidth_p", type=float, default=0.4)
    ap.add_argument("--header_h_min_px", type=int, default=10)
    ap.add_argument("--header_h_max_px", type=int, default=60)
    ap.add_argument("--header_gray_min", type=int, default=140)
    ap.add_argument("--header_gray_max", type=int, default=220)

    return ap


if __name__ == "__main__":
    ap = build_argparser()
    args = ap.parse_args()

    cfg = SynthConfig(
        out_root=args.out_root,
        n_train=args.n_train,
        n_val=args.n_val,
        img_size=args.img_size,
        jitter_px=args.jitter_px,
        curve_amp_px=args.curve_amp_px,
        curve_amp_jitter=args.curve_amp_jitter,
        base_inner_px=args.base_inner_px,
        base_border_px=args.base_border_px,
        border_bias_px=args.border_bias_px,
        thin_line_p=args.thin_line_p,
        very_thin_p=args.very_thin_p,
        faint_line_p=args.faint_line_p,
        blur_p=args.blur_p,
        noise_p=args.noise_p,
        vignette_p=args.vignette_p,
        shadow_p=args.shadow_p,
        occluder_density=args.occluder_density,
        grad_strength=args.grad_strength,
        jpeg_q_min=args.jpeg_q_min,
        jpeg_q_max=args.jpeg_q_max,
        grayscale=args.grayscale,
        persp_strength=args.persp_strength,
        elastic_alpha=args.elastic_alpha,
        elastic_sigma=args.elastic_sigma,
        gauss_j_sigma=args.gauss_j_sigma,
        header_band_p=args.header_band_p,
        header_halfwidth_p=args.header_halfwidth_p,
        header_h_min_px=args.header_h_min_px,
        header_h_max_px=args.header_h_max_px,
        header_gray_min=args.header_gray_min,
        header_gray_max=args.header_gray_max,
        seed=args.seed
    )
    run_generate(cfg)