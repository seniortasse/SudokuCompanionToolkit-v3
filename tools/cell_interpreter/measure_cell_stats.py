import argparse, json, math, sys, time
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def read_jsonl(p: Path) -> List[Dict]:
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except Exception:
                # Skip badly formatted lines, but keep going
                pass
    return rows

def resolve_path(p: str, project_root: Path) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (project_root / pp).resolve()

def var_laplacian(img_gray):
    return float(cv2.Laplacian(img_gray, cv2.CV_64F).var())

def otsu_binarize(img_gray):
    blur = cv2.GaussianBlur(img_gray, (3,3), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw

def distance_transform_width(bw_fg255):
    fg = (bw_fg255 > 0).astype(np.uint8)
    if fg.sum() == 0:
        return 0.0
    dist = cv2.distanceTransform(fg, distanceType=cv2.DIST_L2, maskSize=3)
    mean_radius = float(dist[fg > 0].mean())
    return 2.0 * mean_radius

def fg_stats(img_gray, bw_fg255):
    h, w = img_gray.shape[:2]
    n = h * w
    fg = (bw_fg255 > 0)
    bg = ~fg
    coverage = float(fg.sum()) / float(max(1, n))
    ink_mean = float(img_gray[fg].mean()) if fg.sum() > 0 else float(img_gray.mean())
    if bg.sum() > 0:
        bg_std = float(img_gray[bg].std())
        gx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
        gradmag = np.sqrt(gx*gx + gy*gy)
        bg_gradmag = float(gradmag[bg].mean())
    else:
        bg_std = 0.0
        bg_gradmag = 0.0
    return coverage, ink_mean, bg_std, bg_gradmag

def line_energy(img_gray):
    edges = cv2.Canny(img_gray, 50, 150)
    k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (9,1))
    k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1,9))
    h_resp = cv2.morphologyEx(edges, cv2.MORPH_OPEN, k_h)
    v_resp = cv2.morphologyEx(edges, cv2.MORPH_OPEN, k_v)
    total = float(np.count_nonzero(edges) + 1e-6)
    e_h = float(np.count_nonzero(h_resp)) / total
    e_v = float(np.count_nonzero(v_resp)) / total
    return e_h, e_v

def centroid_and_orientation(bw_fg255):
    fg = (bw_fg255 > 0).astype(np.uint8)
    if fg.sum() == 0:
        return 0.0, 0.0, 0.0
    m = cv2.moments(fg, binaryImage=True)
    h, w = fg.shape[:2]
    if m['m00'] <= 1e-6:
        return 0.0, 0.0, 0.0
    cx = m['m10'] / m['m00']
    cy = m['m01'] / m['m00']
    cx_off = (cx - (w/2.0)) / (w/2.0 + 1e-6)
    cy_off = (cy - (h/2.0)) / (h/2.0 + 1e-6)
    mu11 = m['mu11']; mu20 = m['mu20']; mu02 = m['mu02']
    angle = 0.5 * np.degrees(np.arctan2(2.0*mu11, (mu20 - mu02 + 1e-12)))
    return float(cx_off), float(cy_off), float(angle)

def ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(a.astype(np.float64))
    b = np.sort(b.astype(np.float64))
    xs = np.unique(np.concatenate([a, b], axis=0))
    def ecdf(x, s):
        return np.searchsorted(s, x, side='right') / s.size
    fa = np.array([ecdf(x, a) for x in xs])
    fb = np.array([ecdf(x, b) for x in xs])
    return float(np.max(np.abs(fa - fb)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", nargs="+", required=True)
    ap.add_argument("--tag", nargs="+", required=True)
    ap.add_argument("--project-root", type=str, default=".")
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=1000,
                    help="Print a progress heartbeat every N processed images per tag (default: 1000).")
    args = ap.parse_args()

    if len(args.jsonl) != len(args.tag):
        print("[error] --jsonl and --tag must have the same length.", file=sys.stderr, flush=True)
        sys.exit(2)

    proj = Path(args.project_root).resolve()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    csv_rows = []
    try:
        for jpath, tag in zip(args.jsonl, args.tag):
            rows = read_jsonl(Path(jpath))
            total_planned = len(rows) if args.max_images <= 0 else min(len(rows), args.max_images)
            if args.max_images > 0:
                rows = rows[:args.max_images]

            print(f"[start] tag={tag} jsonl={jpath} total_images={total_planned}", flush=True)

            processed = 0
            t0 = time.time()
            for r in rows:
                p = r.get("path") or ""
                if not p:
                    continue
                ipath = (proj / p).resolve() if not Path(p).is_absolute() else Path(p)
                if not ipath.exists():
                    print(f"[warn] missing image: {ipath}", flush=True)
                    continue
                img = cv2.imread(str(ipath), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"[warn] unreadable image: {ipath}", flush=True)
                    continue

                bw = otsu_binarize(img)
                coverage, ink_mean, bg_std, bg_gradmag = fg_stats(img, bw)
                stroke_w = distance_transform_width(bw)
                sharp = var_laplacian(img)
                e_h, e_v = line_energy(img)
                cx_off, cy_off, orient_deg = centroid_and_orientation(bw)

                csv_rows.append({
                    "tag": tag,
                    "path": str(ipath),
                    "coverage_fg": coverage,
                    "stroke_width_px": stroke_w,
                    "sharpness_varlap": sharp,
                    "ink_mean": ink_mean,
                    "bg_std": bg_std,
                    "bg_gradmag": bg_gradmag,
                    "line_energy_h": e_h,
                    "line_energy_v": e_v,
                    "cx_off_norm": cx_off,
                    "cy_off_norm": cy_off,
                    "orient_deg": orient_deg,
                })
                processed += 1

                if processed % max(1, args.log_every) == 0:
                    dt = time.time() - t0
                    ips = processed / dt if dt > 0 else 0.0
                    pct = 100.0 * processed / max(1, total_planned)
                    print(f"[{tag}] processed={processed}/{total_planned} ({pct:.1f}%)  ~{ips:.1f} img/s", flush=True)

            dt = time.time() - t0
            ips = processed / dt if dt > 0 else 0.0
            print(f"[info] {tag}: processed {processed}/{total_planned} images in {dt:.1f}s  (~{ips:.1f} img/s)", flush=True)

    except KeyboardInterrupt:
        print("[abort] Caught KeyboardInterrupt. Writing whatever has been processed so far...", flush=True)

    # Write CSV
    import csv
    csv_path = outdir / "cell_stats.csv"
    if not csv_rows:
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("")
        print(f"[done] wrote empty CSV -> {csv_path}", flush=True)
        return

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        for r in csv_rows:
            w.writerow(r)
    print(f"[done] wrote CSV -> {csv_path}", flush=True)

    # Build arrays per metric per tag
    metrics = [
        "coverage_fg","stroke_width_px","sharpness_varlap","ink_mean",
        "bg_std","bg_gradmag","line_energy_h","line_energy_v",
        "cx_off_norm","cy_off_norm","orient_deg",
    ]
    by_tag = {m:{} for m in metrics}
    for r in csv_rows:
        t = r["tag"]
        for m in metrics:
            by_tag[m].setdefault(t, []).append(float(r[m]))
    for m in metrics:
        for t in list(by_tag[m].keys()):
            by_tag[m][t] = np.asarray(by_tag[m][t], dtype=np.float64)

    # Plots + KS report
    report_lines = []
    for m in metrics:
        plt.figure()
        for t, arr in by_tag[m].items():
            plt.hist(arr, bins=40, alpha=0.5, label=f"{t} (n={arr.size})", density=True)
        plt.title(m)
        plt.legend()
        plt.tight_layout()
        plt.savefig(str(outdir / f"{m}.png"))
        plt.close()

        # KS if >=2 tags
        tags = list(by_tag[m].keys())
        if len(tags) >= 2:
            for i in range(len(tags)):
                for j in range(i+1, len(tags)):
                    a, b = by_tag[m][tags[i]], by_tag[m][tags[j]]
                    if a.size > 0 and b.size > 0:
                        d = ks_distance(a, b)
                        report_lines.append(f"{m}: KS({tags[i]} vs {tags[j]}) = {d:.3f}")

    report_path = outdir / "report_ks.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) if report_lines else "No pairwise comparisons computed.\n")
    print(f"[done] wrote report -> {report_path}", flush=True)

if __name__ == "__main__":
    main()