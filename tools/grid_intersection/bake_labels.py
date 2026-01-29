from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List, Tuple
import numpy as np
import cv2

def load_ann(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def gaussian_disk(h: int, w: int, cx: float, cy: float, sigma: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    g = (255 * g / (g.max() + 1e-6)).astype(np.uint8)
    return g

def band_from_points(h: int, w: int, pts: List[Tuple[float, float]], thr: int = 8) -> np.ndarray:
    mask = np.zeros((h, w), np.uint8)
    if len(pts) < 3:
        return mask
    hull = cv2.convexHull(np.array(pts, dtype=np.float32))
    cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thr*2+1, thr*2+1))
    mask = cv2.dilate(mask, k)
    return mask

def bake_for_image(img_path: Path, ann_path: Path, out_npz: Path, j_sigma: float = 1.2,
                   make_coarse_AHV: bool = False) -> str:
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(img_path)
    h, w = img.shape[:2]
    ann = load_ann(ann_path)
    juncs = ann.get("junctions", [])  # list of [x,y]

    # J map
    J = np.zeros((h, w), np.uint8)
    for (x, y) in juncs:
        J = np.maximum(J, gaussian_disk(h, w, float(x), float(y), j_sigma))

    # Optional coarse A/H/V
    if make_coarse_AHV and len(juncs) >= 25:
        xs = sorted(set(int(round(p[0])) for p in juncs))
        ys = sorted(set(int(round(p[1])) for p in juncs))
        A = np.zeros((h, w), np.uint8)
        H = np.zeros((h, w), np.uint8)
        V = np.zeros((h, w), np.uint8)
        for y in ys:
            cv2.line(H, (0, y), (w-1, y), 255, 2, cv2.LINE_AA)
        for x in xs:
            cv2.line(V, (x, 0), (x, h-1), 255, 2, cv2.LINE_AA)
        A = np.maximum(H, V)
    else:
        A = np.zeros((h, w), np.uint8)
        H = np.zeros((h, w), np.uint8)
        V = np.zeros((h, w), np.uint8)

    # Orientation unknown for real → zeros
    O = np.zeros((h, w, 2), np.float32)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_npz), A=A, H=H, V=V, J=J, O=O)

    # Allowed band (for FP metric)
    allowed_band = band_from_points(h, w, juncs, thr=12)
    np.save(str(out_npz.with_suffix(".band.npy")), allowed_band)
    return str(out_npz)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True, help="datasets/grids/real/images/")
    ap.add_argument("--ann_dir", required=True, help="datasets/grids/real/annotations/")
    ap.add_argument("--out_labels", required=True, help="datasets/grids/real/labels/")
    ap.add_argument("--manifest_in", required=True, help="Input JSONL manifest to update (label_path)")
    ap.add_argument("--manifest_out", required=True, help="Output JSONL manifest with updated label_path")
    ap.add_argument("--j_sigma", type=float, default=1.2)
    ap.add_argument("--weak_AHV", action="store_true", help="Also draw coarse A/H/V from rows/cols")
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    ann_dir = Path(args.ann_dir)
    out_labels = Path(args.out_labels)

    # Load manifest
    records = []
    with open(args.manifest_in, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Bake labels when matching annotation JSON exists
    updated = []
    for r in records:
        ip = Path(r["image_path"])
        annp = ann_dir / f"{ip.stem}.json"
        if annp.exists():
            out_npz = out_labels / f"{ip.stem}.npz"
            lab_path = bake_for_image(ip, annp, out_npz, args.j_sigma, args.weak_AHV)
            r["label_path"] = lab_path
        else:
            r["label_path"] = r.get("label_path", None)
        updated.append(r)

    Path(args.manifest_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest_out, "w", encoding="utf-8") as f:
        for r in updated:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[bake_labels] updated manifest → {args.manifest_out}")

if __name__ == "__main__":
    main()