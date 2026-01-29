# collect_digits96.py — Build a clean 96×96 handwritten-digit corpus + manifest
# ============================================================================
# STORY / CONTEXT
# ----------------------------------------------------------------------------
# This script is the **front door** to our synthetic Sudoku cell pipeline:
# we curate a high‑quality library of **handwritten digit patches** (0–9) that
# will later be pasted into 64×64 Sudoku cells as (a) candidate glyphs and (b)
# solution digits. Good patches here → better realism later.
#
# Sources supported out of the box:
#   • **EMNIST**  (provided as .mat files) under datasets/digits/matlab/*
#   • **NIST SD‑19** "by_class"  under datasets/digits/nist_sd19/by_class
#   • **NIST SD‑19** "by_write"  under datasets/digits/nist_sd19/by_write
#
# What the script does:
#   1) Loads digits from each source (robust parsing for EMNIST .mat).
#   2) Normalizes each image into a **96×96, white‑background, grayscale**
#      canvas, with strokes dark (black-ish) and aspect ratio preserved.
#   3) Writes files to an organized tree:
#        <out_root>/<source>/<digit>/<filename>.png
#      For example: datasets/digits96/nist_by_class/7/nist_byclass_7_000012.png
#   4) Produces a JSONL **manifest** containing one row per image:
#        {"path": "<relative path>", "digit": int, "source": "<source>"}
#
# Why 96×96?
#   • It’s a comfy resolution: large enough to preserve stroke details,
#     small enough to keep storage and I/O light. Our synthesizer will scale
#     these patches down to ~18–40 px when pasting into 64×64 Sudoku cells.
#
# Design principles:
#   • **Uniform normalization** across sources → consistent brightness/contrast.
#   • **Dark-on-light convention** so downstream compositing/blending is simple.
#   • **Tight crop + padded center fit** so digits are well-centered.
#   • **Deterministic structure** (stable paths, digit-aware directories)
#     to make downstream sampling trivial and debuggable.
#
# Expected consumers:
#   • tools/synthesize_cells_from_digits.py (uses this manifest to assemble
#     synthetic Sudoku cells with various layouts and jitters).
#
# Dependencies: pillow, numpy, scipy (for EMNIST), tqdm
#   pip install pillow numpy scipy tqdm
# ============================================================================

from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

# Lazy import for EMNIST .mat reading. If SciPy is missing, we gracefully skip.
try:
    from scipy.io import loadmat
except Exception:  # pragma: no cover
    loadmat = None

# Recognized raster formats when scanning NIST trees.
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pct"}


# ----------------------------------------------------------------------------
# Small filesystem / imaging utilities
# ----------------------------------------------------------------------------

def ensure_dir(p: Path) -> None:
    """Create folder *p* (and parents) if it does not exist."""
    p.mkdir(parents=True, exist_ok=True)


def to_grayscale(im: Image.Image) -> Image.Image:
    """Return grayscale (mode 'L'); tolerant to common input modes.
    
    We normalize everything to single‑channel grayscale so later math
    (cropping, inversion heuristics, resizing) is predictable.
    """
    if im.mode == "L":
        return im
    if im.mode in ("RGB", "RGBA", "LA", "P"):
        return im.convert("L")
    return im.convert("L")


def maybe_invert(arr: np.ndarray) -> np.ndarray:
    """Ensure **dark ink on light background**.
    Heuristic: if the image median is < 128 (mostly dark), we invert.
    This gives us a consistent convention across datasets.
    """
    med = np.median(arr)
    return (255 - arr) if med < 128 else arr


def tight_crop(arr: np.ndarray, thr: int = 250, pad_frac: float = 0.06) -> np.ndarray:
    """Crop to the smallest box that contains non‑white pixels (< thr),
    then add symmetric padding as a fraction of max(H,W) to avoid over‑tight crops.
    If no ink is found, return the original array.
    """
    h, w = arr.shape
    mask = arr < thr
    if not mask.any():
        return arr
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    pad = int(round(max(h, w) * pad_frac))
    y0 = max(0, y0 - pad); y1 = min(h - 1, y1 + pad)
    x0 = max(0, x0 - pad); x1 = min(w - 1, x1 + pad)
    return arr[y0 : y1 + 1, x0 : x1 + 1]


def normalize_to_96(im: Image.Image, size: int = 96) -> Image.Image:
    """Full normalization pipeline → **96×96 white canvas**.
    Steps:
      1) convert to grayscale
      2) invert if needed (dark strokes on white)
      3) tight crop with small padding
      4) resize to fit within *size* while preserving aspect
      5) center paste on a white *size×size* canvas
    """
    im = to_grayscale(im)
    arr = np.array(im, dtype=np.uint8)
    arr = maybe_invert(arr)
    arr = tight_crop(arr, thr=250, pad_frac=0.08)

    # Preserve aspect ratio when fitting to square
    h, w = arr.shape
    scale = min(size / h, size / w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    arr_resized = np.array(
        Image.fromarray(arr).resize((new_w, new_h), Image.BICUBIC), dtype=np.uint8
    )

    canvas = np.full((size, size), 255, dtype=np.uint8)  # white background
    y0 = (size - new_h) // 2
    x0 = (size - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = arr_resized

    return Image.fromarray(canvas, mode="L")


# ----------------------------------------------------------------------------
# EMNIST (.mat) discovery & parsing
# ----------------------------------------------------------------------------

def find_emnist_mat_files(root: Path) -> List[Path]:
    """Recursively collect all .mat files under *root* (if present)."""
    if not root.exists():
        return []
    return sorted(root.rglob("*.mat"))


def emnist_extract_images_and_labels(mat_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Extract `(images[N,28,28], labels[N])` from a variety of EMNIST layouts.
    
    EMNIST sometimes nests arrays under `dataset/train|test/images|labels`.
    Other times we just get top‑level arrays. We handle both, then apply the
    conventional EMNIST orientation fix (rotate + flip) so digits are upright.
    """
    if loadmat is None:
        raise RuntimeError("scipy is required to read EMNIST .mat files (pip install scipy).")

    md = loadmat(str(mat_path))

    images_all, labels_all = [], []

    # Preferred nested structure
    dataset = md.get("dataset", None)
    if dataset is not None:
        for split in ("train", "test"):
            try:
                node = dataset[0, 0][split][0, 0]
                imgs = node["images"]
                labs = node["labels"].reshape(-1)

                # Normalize shapes → (N, 28, 28)
                if imgs.ndim == 2:
                    if imgs.shape[1] == 784:
                        imgs = imgs.reshape((-1, 28, 28))
                    elif imgs.shape[0] == 784:
                        imgs = imgs.T.reshape((-1, 28, 28))
                elif imgs.ndim == 3 and imgs.shape[1:] == (28, 28):
                    pass
                else:
                    continue

                # Orientation fix (rotate then flip)
                imgs = np.rot90(imgs, k=1, axes=(1, 2))
                imgs = np.fliplr(imgs)

                images_all.append(imgs.astype(np.uint8))
                labels_all.append(labs.astype(np.int64))
            except Exception:
                # If one split is missing/odd, keep going; we’ll try other paths
                pass

    # Fallback: look for any arrays that look like images/labels
    if not images_all:
        for k, v in md.items():
            if not isinstance(v, np.ndarray):
                continue
            if "images" in k and v.ndim >= 2:
                imgs = v
                labs = None
                k2 = k.replace("images", "labels")
                if k2 in md:
                    labs = md[k2].reshape(-1)
                if labs is None:
                    continue

                if imgs.ndim == 2:
                    if imgs.shape[1] == 784:
                        imgs = imgs.reshape((-1, 28, 28))
                    elif imgs.shape[0] == 784:
                        imgs = imgs.T.reshape((-1, 28, 28))
                elif imgs.ndim == 3 and imgs.shape[1:] == (28, 28):
                    pass
                else:
                    continue

                imgs = np.rot90(imgs, k=1, axes=(1, 2))
                imgs = np.fliplr(imgs)

                images_all.append(imgs.astype(np.uint8))
                labels_all.append(labs.astype(np.int64))

    if not images_all:
        raise RuntimeError(f"Could not parse EMNIST .mat structure in {mat_path}")

    images = np.concatenate(images_all, axis=0)
    labels = np.concatenate(labels_all, axis=0)
    return images, labels


def is_digit_label(l) -> bool:
    """Return True iff *l* is an integer 0..9."""
    try:
        li = int(l)
        return 0 <= li <= 9
    except Exception:
        return False


# ----------------------------------------------------------------------------
# NIST SD‑19 scanning
# ----------------------------------------------------------------------------

def scan_images_with_digit_label(root: Path) -> Iterable[Tuple[Path, int]]:
    """Walk *root* and yield `(image_path, digit)` whenever one path component
    is the single character '0'..'9'. This works for both `by_class` and
    `by_write` trees (we don’t assume exact depth, just the presence of a
    digit‑named folder in the path).
    """
    if not root.exists():
        return
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            d = None
            for part in p.parts[::-1]:  # scan from leaf upward
                if part.isdigit() and len(part) == 1:
                    d = int(part); break
            if d is not None and 0 <= d <= 9:
                yield p, d


# ----------------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------------

def main() -> None:
    # CLI — keep defaults aligned with repo layout
    ap = argparse.ArgumentParser()
    ap.add_argument("--emnist-root", type=Path, default=Path("datasets/digits/matlab"),
                    help="Folder containing EMNIST .mat files (any depth).")
    ap.add_argument("--nist-byclass-root", type=Path, default=Path("datasets/digits/nist_sd19/by_class"))
    ap.add_argument("--nist-bywrite-root", type=Path, default=Path("datasets/digits/nist_sd19/by_write"))
    ap.add_argument("--out-root", type=Path, default=Path("datasets/digits96"),
                    help="Output root for normalized 96x96 PNGs.")
    ap.add_argument("--manifest", type=Path, default=Path("datasets/digits96/digits96_manifest.jsonl"))
    ap.add_argument("--max-per-digit", type=int, default=0,
                    help="Optional cap per digit per source (0 = no cap).")
    ap.add_argument("--overwrite", action="store_true", help="Recreate files even if they exist.")
    ap.add_argument("--append", action="store_true",
                    help="Append to an existing manifest (de‑dupe by path).")
    args = ap.parse_args()

    # Prep outputs
    ensure_dir(args.out_root)
    ensure_dir(args.manifest.parent)

    records: List[dict] = []  # accumulate rows before de‑dupe/append logic

    print("[info] Starting digit collection…")
    print(f"[info] out_root   = {args.out_root}")
    print(f"[info] manifest   = {args.manifest}")

    # 1) EMNIST — optional if SciPy is available and .mat files exist
    emnist_mats = find_emnist_mat_files(args.emnist_root)
    if emnist_mats and loadmat is None:
        print("[warn] EMNIST .mat found but scipy not installed; skipping EMNIST.", file=sys.stderr)

    if emnist_mats and loadmat is not None:
        cap = {d: 0 for d in range(10)}
        for matp in emnist_mats:
            try:
                imgs, labs = emnist_extract_images_and_labels(matp)
            except Exception as e:
                print(f"[warn] skipping {matp}: {e}", file=sys.stderr)
                continue

            for i in tqdm(range(len(labs)), desc=f"[EMNIST] {matp.name}", unit="img"):
                lab = int(labs[i])
                if not is_digit_label(lab):
                    continue
                if args.max_per_digit > 0 and cap[lab] >= args.max_per_digit:
                    continue

                arr = imgs[i]
                im = Image.fromarray(arr, mode="L")
                out_im = normalize_to_96(im, size=96)

                out_dir = args.out_root / "emnist" / str(lab)
                ensure_dir(out_dir)
                fname = f"emnist_{matp.stem}_{lab}_{cap[lab]:06d}.png"
                out_path = out_dir / fname
                if args.overwrite or not out_path.exists():
                    out_im.save(out_path, format="PNG")

                records.append({"path": out_path.as_posix(), "digit": lab, "source": "emnist"})
                cap[lab] += 1

    # 2) NIST SD‑19 by_class — directory tree with digit‑named folders
    if args.nist_byclass_root.exists():
        cap = {d: 0 for d in range(10)}
        items = list(scan_images_with_digit_label(args.nist_byclass_root))
        for p, d in tqdm(items, desc="[NIST by_class]", unit="img"):
            if args.max_per_digit > 0 and cap[d] >= args.max_per_digit:
                continue
            try:
                im = Image.open(p)
            except Exception:
                continue
            out_im = normalize_to_96(im, size=96)
            out_dir = args.out_root / "nist_by_class" / str(d)
            ensure_dir(out_dir)
            fname = f"nist_byclass_{d}_{cap[d]:06d}.png"
            out_path = out_dir / fname
            if args.overwrite or not out_path.exists():
                out_im.save(out_path, format="PNG")
            records.append({"path": out_path.as_posix(), "digit": d, "source": "nist_by_class"})
            cap[d] += 1
    else:
        print(f"[info] Skipping NIST by_class (missing: {args.nist_byclass_root})")

    # 3) NIST SD‑19 by_write — alternate tree (writer‑organized) but still
    #    has digit‑named components we can parse.
    if args.nist_bywrite_root.exists():
        cap = {d: 0 for d in range(10)}
        items = list(scan_images_with_digit_label(args.nist_bywrite_root))
        for p, d in tqdm(items, desc="[NIST by_write]", unit="img"):
            if args.max_per_digit > 0 and cap[d] >= args.max_per_digit:
                continue
            try:
                im = Image.open(p)
            except Exception:
                continue
            out_im = normalize_to_96(im, size=96)
            out_dir = args.out_root / "nist_by_write" / str(d)
            ensure_dir(out_dir)
            fname = f"nist_bywrite_{d}_{cap[d]:06d}.png"
            out_path = out_dir / fname
            if args.overwrite or not out_path.exists():
                out_im.save(out_path, format="PNG")
            records.append({"path": out_path.as_posix(), "digit": d, "source": "nist_by_write"})
            cap[d] += 1
    else:
        print(f"[info] Skipping NIST by_write (missing: {args.nist_bywrite_root})")

    # Manifest writing — supports --append (de‑dupe by path)
    existing: dict[str, dict] = {}
    if args.append and args.manifest.exists():
        with args.manifest.open("r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    obj = json.loads(ln)
                    p = obj.get("path", "")
                    if p:
                        existing[p] = obj
                except Exception:
                    pass

    for r in records:
        existing[r["path"]] = r  # last one wins; ensures de‑dupe by path

    ensure_dir(args.manifest.parent)
    with args.manifest.open("w", encoding="utf-8") as f:
        for r in existing.values():
            f.write(json.dumps(r) + "\n")

    print(f"[done] wrote {len(existing)} total records to {args.manifest}")


if __name__ == "__main__":
    main()
