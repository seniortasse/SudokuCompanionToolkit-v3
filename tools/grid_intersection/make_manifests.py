from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List
import random
import cv2

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def list_images(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS])

def write_jsonl(path: Path, records: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def guess_label_path(image_path: Path, labels_root: Path | None) -> str | None:
    if labels_root is None:
        return None
    stem = image_path.stem
    cand = labels_root / f"{stem}.npz"
    return str(cand) if cand.exists() else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True, help="datasets/grids/real/images/")
    ap.add_argument("--out_dir", required=True, help="datasets/grids/real/manifests/")
    ap.add_argument("--labels_dir", default="", help="Optional labels dir with NPZ files")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    images_root = Path(args.images_dir)
    out_root = Path(args.out_dir)
    labels_root = Path(args.labels_dir) if args.labels_dir else None

    ims = list_images(images_root)
    if not ims:
        raise SystemExit(f"No images found in {images_root}")

    recs = []
    for p in ims:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[skip] unreadable: {p}")
            continue
        h, w = img.shape[:2]
        labp = guess_label_path(p, labels_root)
        recs.append({
            "image_path": str(p.resolve()),
            "label_path": labp,   # may be None
            "split": "train",     # set after shuffle
            "height": int(h),
            "width": int(w),
            "meta": {"source": "real"}
        })

    random.Random(args.seed).shuffle(recs)
    n = len(recs)
    n_val = max(1, int(round(n * args.val_ratio)))
    val = recs[:n_val]
    train = recs[n_val:]
    for r in val: r["split"] = "val"

    write_jsonl(out_root / "train_real.jsonl", train)
    write_jsonl(out_root / "val_real.jsonl", val)
    print(f"[make_manifests] wrote {len(train)} train and {len(val)} val records → {out_root}")

if __name__ == "__main__":
    main()