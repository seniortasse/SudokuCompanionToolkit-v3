# -*- coding: utf-8 -*-
"""
Synthesize Sudoku Cells from Handwritten Digit Patches (Story + Annotations)
=============================================================================

WHY THIS SCRIPT EXISTS
----------------------
Phase 1 of the Sudoku Companion pipeline needs a *large, diverse, controllable* stream of
training examples for the **Cell Interpreter** model (CellNet). Real, labeled cells are scarce,
so we bootstrap with *synthetic cells* built from real handwritten digit patches (your
digits96 set) and optional blank backgrounds. This script:

  • Stitches together 64×64 grayscale cell images that look like crops from real puzzles.
  • Randomizes stylistic details (font choice for givens, ink gray, jitter, smudge, grid hints,
    brightness/contrast/noise) to mimic camera/printing variability.
  • Emits a JSONL manifest that pairs each generated image with labels used by CellNet:
        - given_digit (printed font, like a puzzle clue)
        - solution_digit (handwritten, center)
        - candidates (list[int] 1..9 scattered around)

MODEL-LEVEL INTENT
------------------
For the **candidates head**, we especially care about spatial variety: candidate digits should
appear near edges/corners or clustered, sometimes partially overlapping—but never so much
that they become indistinguishable. For **solution** we paste a central handwritten patch.
For **given**, we draw a clean printed font.

OUTPUT CONTRACT
---------------
• IMAGES → datasets/cells/cell_interpreter/<out_tag>/img_XXXXXX.png
• JSONL  → datasets/cells/cell_interpreter/<out_tag>/cells_<out_tag>.jsonl

Each JSON line has:
  { "path": str, "given_digit": int, "solution_digit": int, "candidates": List[int], "source": str }

HOW TO USE (quick start)
------------------------
    python tools/synthesize_cells_from_digits.py \
        --digits-manifest datasets/digits96/digits96_manifest.jsonl \
        --out-tag synth_v5_from_digits_train \
        --num-cells 50000

TUNABLES
--------
• Mixture of cell types (candidate-only / solution-only / solution+candidate / given-only)
• Printed font discovery on Windows machines (auto); override via --font-printed
• Candidate placement controls (Poisson-like sampling + repulsion)
• Camera-style jitter (contrast/brightness/noise) and light smudge
• Optional grid occlusion to hint thick borders/midlines

Notes
-----
All heavy lifting is pure-Pillow/NumPy. The result is deterministic for a given --seed.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Imports — what and why
# ──────────────────────────────────────────────────────────────────────────────
# argparse/json/pathlib: clean CLIs + robust paths
# random/math/numpy: placement + geometry + noise
# PIL: image IO and drawing; tqdm: progress bar
import argparse, json, random, math, os
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Small utilities — safe dirs, manifests, path handling
# ──────────────────────────────────────────────────────────────────────────────

def ensure_dir(p: Path) -> None:
    """Create directory tree if missing (idempotent)."""
    p.mkdir(parents=True, exist_ok=True)


def load_manifest(manifest_path: Path) -> List[Dict]:
    """Load a JSONL manifest of handwritten digit patches.

    Expected record per line: {"path": str, "digit": int, ...}
    We filter out anything without a valid 0..9 digit or missing path.
    """
    rows: List[Dict] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except Exception:
                # Be forgiving: skip malformed lines without aborting the run.
                pass
    # keep only 0..9
    rows = [r for r in rows if "digit" in r and isinstance(r["digit"], int)
            and 0 <= r["digit"] <= 9 and "path" in r]
    return rows


def pick_patches_by_digit(rows: List[Dict]) -> Dict[int, List[str]]:
    """Bucket patch paths by their labeled digit for fast sampling."""
    bins: Dict[int, List[str]] = {d: [] for d in range(10)}
    for r in rows:
        p = r["path"]
        bins[r["digit"]].append(p)
    return bins


def resolve_image(path_str: str, project_root: Path) -> Image.Image:
    """Resolve a possibly relative path against project_root and open as grayscale."""
    p = Path(path_str)
    if not p.is_absolute():
        p = (project_root / p).resolve()
    return Image.open(p).convert("L")  # grayscale


# ──────────────────────────────────────────────────────────────────────────────
# Text helpers — measuring, clamping, robust font loading
# ──────────────────────────────────────────────────────────────────────────────

def pil_textsize(draw: ImageDraw.Draw, text: str, font: ImageFont.FreeTypeFont) -> Tuple[int,int]:
    bbox = draw.textbbox((0,0), text, font=font)
    return bbox[2]-bbox[0], bbox[3]-bbox[1]


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


_missing_font_cache = set()


def make_font(path: str, px: int) -> ImageFont.FreeTypeFont:
    """Robustly create a font. Falls back to Pillow default if unavailable.

    We memoize missing paths to avoid repeating warnings.
    """
    try:
        if path and Path(path).exists():
            return ImageFont.truetype(path, px)
    except Exception:
        pass
    if path and path not in _missing_font_cache:
        print(f"[warn] missing/unreadable font: {path} -> using default")
        _missing_font_cache.add(path)
    return ImageFont.load_default()


def discover_windows_fonts_by_keywords(keywords: List[str]) -> List[str]:
    """Best-effort font discovery on Windows via keyword search in C:/Windows/Fonts.

    Useful for printed givens; avoids hardcoding a specific typeface.
    """
    out: List[str] = []
    fonts_dir = Path("C:/Windows/Fonts")
    if fonts_dir.exists():
        for p in fonts_dir.glob("*"):
            name = p.name.lower()
            if p.suffix.lower() in (".ttf", ".ttc", ".otf"):
                if any(k in name for k in keywords):
                    out.append(str(p))
    return sorted(set(out))


# ──────────────────────────────────────────────────────────────────────────────
# Geometry + sampling for candidate placement
# ──────────────────────────────────────────────────────────────────────────────

def _boxes_overlap_frac(a, b) -> float:
    """Intersection over smaller area; returns [0..1]."""
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    area_a = (a[2]-a[0])*(a[3]-a[1])
    area_b = (b[2]-b[0])*(b[3]-b[1])
    return inter / max(1.0, min(area_a, area_b))


def _poisson_sample_in_cell(
    W:int, H:int,
    n:int,
    bb_size:int,
    edge_bias:float=0.45,
    min_gap:int=6,
    max_tries:int=2000,
    avoid_rect:Tuple[int,int,int,int]=None,
):
    """Edge-biased Poisson-ish sampler for candidate centers.

    We sprinkle points with:
      • Slight edge preference (candidates often hug borders in real puzzles).
      • A min-gap constraint to reduce overlaps.
      • Optional exclusion of a center rectangle (reserve it for the solution digit).
    Returns a list of (cx, cy) integer centers.
    """
    pts: List[Tuple[int,int]] = []
    tries = 0
    r = max(2, bb_size//2)
    min_d2 = (min_gap + r)**2

    while len(pts) < n and tries < max_tries:
        tries += 1
        # Edge bias / interior split
        u = random.random()
        if u < edge_bias:
            side = random.choice(("L","R","T","B"))
            inset = random.randint(2, 8)
            if side == "L":
                cx = random.randint(inset, inset + max(3, r))
                cy = random.randint(r, H - r - 1)
            elif side == "R":
                cx = random.randint(W - inset - max(3, r), W - inset)
                cy = random.randint(r, H - r - 1)
            elif side == "T":
                cy = random.randint(inset, inset + max(3, r))
                cx = random.randint(r, W - r - 1)
            else:
                cy = random.randint(H - inset - max(3, r), H - inset)
                cx = random.randint(r, W - r - 1)
        else:
            cx = random.randint(r, W - r - 1)
            cy = random.randint(r, H - r - 1)

        x0, y0 = cx - r, cy - r
        x1, y1 = cx + r, cy + r
        if x0 < 0 or y0 < 0 or x1 >= W or y1 >= H:
            continue

        if avoid_rect is not None:
            if _boxes_overlap_frac((x0,y0,x1,y1), avoid_rect) > 0.01:
                continue

        ok = True
        for (px, py) in pts:
            if (px - cx)**2 + (py - cy)**2 < min_d2:
                ok = False; break
        if ok:
            pts.append((cx, cy))

    return pts


def _layout_candidates(
    cell_W:int,
    cell_H:int,
    k:int,
    glyph_box_side:int,
    center_reserved_frac:float=0.30,
    min_gap:int=6,
    edge_bias:float=0.50,
    repel_steps:int=8,
    repel_strength:float=0.35,
) -> List[Tuple[int,int]]:
    """Place k candidate centers with (1) edge bias, (2) center reservation, (3) repulsion.

    This balances realism (edge clustering) with readability (limited overlap).
    """
    cx0 = int(cell_W * 0.5)
    cy0 = int(cell_H * 0.5)
    half = int(min(cell_W, cell_H) * center_reserved_frac * 0.5)
    center_rect = (cx0 - half, cy0 - half, cx0 + half, cy0 + half)

    pts = _poisson_sample_in_cell(
        cell_W, cell_H, k, glyph_box_side,
        edge_bias=edge_bias, min_gap=min_gap,
        avoid_rect=center_rect if k <= 6 else None,
    )

    # Repulsion sweeps to reduce crowding
    r = max(2, glyph_box_side//2)
    for _ in range(repel_steps):
        moved = False
        for i in range(len(pts)):
            x, y = pts[i]
            fx = fy = 0.0
            for j in range(len(pts)):
                if i == j: continue
                x2, y2 = pts[j]
                dx = x - x2; dy = y - y2
                d2 = dx*dx + dy*dy
                if d2 == 0:
                    fx += (random.random()-0.5); fy += (random.random()-0.5)
                    continue
                if d2 < (min_gap + r)**2:
                    inv = 1.0 / math.sqrt(d2)
                    fx += dx * inv
                    fy += dy * inv
            if fx != 0 or fy != 0:
                nx = int(round(x + repel_strength * fx))
                ny = int(round(y + repel_strength * fy))
                nx = max(r, min(cell_W - r - 1, nx))
                ny = max(r, min(cell_H - r - 1, ny))
                if (nx, ny) != (x, y):
                    pts[i] = (nx, ny); moved = True
        if not moved:
            break
    return pts


# ──────────────────────────────────────────────────────────────────────────────
# Patch pasting — estimate bbox, scale, alpha blend onto canvas
# ──────────────────────────────────────────────────────────────────────────────

def estimate_patch_bbox(patch: Image.Image, target_px: int, cx: int, cy: int) -> Tuple[int,int,int,int]:
    """Estimate the bounding box after scaling a patch to target pixel height.

    We compute a crude ink mask to crop extra whitespace before scaling—helps consistency.
    """
    arr = np.array(patch.convert("L"), dtype=np.float32)
    # Normalize polarity: assume darker ink on lighter paper
    if np.median(arr) < 128:
        arr = 255.0 - arr
    ink_mask = arr < 250
    if ink_mask.any():
        ys, xs = np.where(ink_mask)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        arr = arr[y0:y1+1, x0:x1+1]
    h, w = arr.shape
    scale = max(1e-6, target_px / float(h))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    x0 = int(cx - new_w / 2)
    y0 = int(cy - new_h / 2)
    return (x0, y0, x0 + new_w, y0 + new_h)


def paste_scaled_patch(canvas: Image.Image, patch: Image.Image, cx: int, cy: int,
                       target_px: int, gray_min: int, gray_max: int) -> None:
    """Paste a handwriting patch onto canvas with simple matte compositing.

    Steps: polarity normalize → crop ink → scale → build alpha from inverted luminance →
    gray-ink matte over background.
    """
    arr = np.array(patch.convert("L"), dtype=np.float32)
    if np.median(arr) < 128:
        arr = 255.0 - arr

    ink_mask = arr < 250
    if ink_mask.any():
        ys, xs = np.where(ink_mask)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        arr = arr[y0:y1+1, x0:x1+1]

    h, w = arr.shape
    scale = max(1e-6, target_px / float(h))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    arr = np.array(Image.fromarray(arr.astype(np.uint8)).resize((new_w, new_h),
                                                                Image.BICUBIC), dtype=np.float32)

    # Alpha from darkness, with threshold to reduce paper noise
    alpha = (255.0 - arr) / 255.0
    thresh = 0.12
    alpha = np.clip((alpha - thresh) / (1.0 - thresh), 0.0, 1.0)

    ink_gray = float(np.random.randint(int(gray_min), int(gray_max)+1))
    patch_rgb = np.full_like(arr, ink_gray, dtype=np.float32)

    x0 = int(cx - new_w/2)
    y0 = int(cy - new_h/2)
    x1 = x0 + new_w
    y1 = y0 + new_h

    cw, ch = canvas.width, canvas.height
    if x1 <= 0 or y1 <= 0 or x0 >= cw or y0 >= ch:
        return

    ox0 = max(0, x0); oy0 = max(0, y0)
    ox1 = min(cw, x1); oy1 = min(ch, y1)
    px0 = ox0 - x0; py0 = oy0 - y0
    px1 = px0 + (ox1 - ox0); py1 = py0 + (oy1 - oy0)

    base = np.array(canvas, dtype=np.float32)
    region = base[oy0:oy1, ox0:ox1]
    a = alpha[py0:py1, px0:px1][..., None]
    ink = patch_rgb[py0:py1, px0:px1][..., None]
    bg = region[..., None]

    blended = (1.0 - a) * bg + a * ink
    blended = blended[..., 0]
    base[oy0:oy1, ox0:ox1] = blended.clip(0, 255).astype(np.uint8)
    canvas.paste(Image.fromarray(base.astype(np.uint8), mode="L"))


# ──────────────────────────────────────────────────────────────────────────────
# Light degradations — smudge, grid hints, camera jitter
# ──────────────────────────────────────────────────────────────────────────────

def light_smudge(im: Image.Image, prob: float, radius_lo: float, radius_hi: float) -> Image.Image:
    """Occasional Gaussian blur to mimic motion or defocus."""
    if random.random() > prob:
        return im
    r = random.uniform(radius_lo, radius_hi)
    if r <= 0:
        return im
    return im.filter(ImageFilter.GaussianBlur(radius=r))


def draw_grid_occlusion(im: Image.Image, prob: float,
                        border_gray: int, border_w: int) -> Image.Image:
    """Optionally draw a thin outer border and midlines to hint a Sudoku grid.

    This helps the model tolerate faint grid artifacts.
    """
    if random.random() > prob:
        return im
    draw = ImageDraw.Draw(im)
    s = im.width
    margin = max(1, s//32)
    draw.rectangle([margin, margin, s-1-margin, s-1-margin],
                   outline=border_gray, width=border_w)
    if random.random() < 0.35:
        mid = s//2 + random.randint(-2,2)
        draw.line([(margin, mid), (s-1-margin, mid)], fill=border_gray, width=1)
    if random.random() < 0.35:
        mid = s//2 + random.randint(-2,2)
        draw.line([(mid, margin), (mid, s-1-margin)], fill=border_gray, width=1)
    return im


# ──────────────────────────────────────────────────────────────────────────────
# Misc helpers — polar jitter, printed digits (givens)
# ──────────────────────────────────────────────────────────────────────────────
from typing import Tuple

def sample_in_disc(cx: int, cy: int, radius: int) -> Tuple[int, int]:
    """Uniform sample inside a disc centered at (cx, cy). Radius 0 → exact center."""
    if radius <= 0:
        return cx, cy
    u = random.random()
    r = int(round(radius * math.sqrt(u)))
    a = 2 * math.pi * random.random()
    dx = int(round(r * math.cos(a)))
    dy = int(round(r * math.sin(a)))
    return cx + dx, cy + dy


def render_text_mask(text: str, font: ImageFont.FreeTypeFont) -> Image.Image:
    """Rasterize text to a tight mask (used as alpha if anchor="bbox")."""
    tmp = Image.new("L", (256, 256), 0)
    d = ImageDraw.Draw(tmp)
    ox, oy = 128, 128
    d.text((ox, oy), text, font=font, fill=255, anchor="mm")
    arr = np.array(tmp)
    ys, xs = np.where(arr > 0)
    if len(xs) == 0:
        return tmp.crop((0, 0, 1, 1))
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return tmp.crop((x0, y0, x1+1, y1+1))


def draw_given_digit(im: Image.Image, digit: int, font_path: str,
                     px: int, jitter_pct: float,
                     gray_min: int, gray_max: int,
                     center_frac_xy: Tuple[float, float],
                     pos_jitter_px: int,
                     center_jitter_radius: int,
                     anchor_mode: str = "mm") -> None:
    """Draw a clean, printed given digit with jitter and font variety.

    • anchor_mode="mm" tries modern Pillow centering; falls back to mask-based paste.
    • Gray range is intentionally dark (near black) for printed clues.
    """
    s = im.width
    text = str(digit)
    # Randomize font size slightly to avoid lockstep visuals
    px2 = int(round(px * (1.0 + random.uniform(-jitter_pct, +jitter_pct))))
    px2 = max(6, px2)
    font = make_font(font_path, px2)

    cx0 = int(center_frac_xy[0] * s)
    cy0 = int(center_frac_xy[1] * s)
    cx, cy = sample_in_disc(cx0, cy0, center_jitter_radius)

    if pos_jitter_px > 0:
        cx += random.randint(-pos_jitter_px, pos_jitter_px)
        cy += random.randint(-pos_jitter_px, pos_jitter_px)

    col = random.randint(gray_min, gray_max)
    draw = ImageDraw.Draw(im)

    if anchor_mode == "mm":
        try:
            draw.text((cx, cy), text, font=font, fill=col, anchor="mm")
            return
        except TypeError:
            # Older Pillow: no anchor support → fall through to mask paste
            pass

    # Fallback: explicit mask composition, centered manually
    mask = render_text_mask(text, font)
    w, h = mask.size
    x = cx - w // 2
    y = cy - h // 2
    base = np.array(im, dtype=np.uint8)
    region = base[max(0,y):min(s,y+h), max(0,x):min(s,x+w)]
    if region.size > 0:
        mx0 = max(0, -x); my0 = max(0, -y)
        mx1 = mx0 + region.shape[1]; my1 = my0 + region.shape[0]
        m = np.array(mask, dtype=np.float32)[my0:my1, mx0:mx1] / 255.0
        ink = np.full_like(region, fill_value=col, dtype=np.float32)
        blended = (1.0 - m) * region.astype(np.float32) + m * ink
        base[max(0,y):min(s,y+h), max(0,x):min(s,x+w)] = blended.clip(0,255).astype(np.uint8)
        im.paste(Image.fromarray(base, mode="L"))


# ──────────────────────────────────────────────────────────────────────────────
# Camera-style jitter — global contrast/brightness/noise
# ──────────────────────────────────────────────────────────────────────────────

def apply_global_jitter(im: Image.Image,
                        contrast_lo: float, contrast_hi: float,
                        brightness_lo: float, brightness_hi: float,
                        noise_std: float) -> Image.Image:
    """Affine intensity transform + Gaussian noise in pixel space.

    Produces small lighting/exposure variations; keeps values in [0,255].
    """
    arr = np.array(im, dtype=np.float32)
    c = random.uniform(contrast_lo, contrast_hi)
    b = random.uniform(brightness_lo, brightness_hi)
    arr = arr * c + b
    if noise_std > 0:
        arr += np.random.normal(0.0, noise_std, size=arr.shape).astype(np.float32)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


# ──────────────────────────────────────────────────────────────────────────────
# Main synthesizer — orchestrates the full data generation loop
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # CLI definition — every flag is documented by its help string.
    ap = argparse.ArgumentParser()

    # IO
    ap.add_argument("--digits-manifest", type=str,
                    default="datasets/digits96/digits96_manifest.jsonl",
                    help="JSONL with records {path, digit, source}.")
    ap.add_argument("--out-tag", type=str, default="synth_v5_from_digits_train",
                    help="Folder name under datasets/cells/cell_interpreter.")
    ap.add_argument("--num-cells", type=int, default=50000)
    ap.add_argument("--project-root", type=str, default=".",
                    help="Base dir to resolve relative manifest paths.")
    ap.add_argument("--cell-size", type=int, default=64,
                    help="Output cell size in pixels (width=height).")

    # optional real blank backgrounds
    ap.add_argument("--blank-bg-list", type=str, default="",
                    help="Optional txt file with one image path per line to use as backgrounds.")

    # mixture of cell types
    ap.add_argument("--p-type-candonly", type=float, default=0.50)
    ap.add_argument("--p-type-solonly", type=float, default=0.10)
    ap.add_argument("--p-type-solpluscand", type=float, default=0.30)
    ap.add_argument("--p-type-givenonly", type=float, default=0.10)

    # candidate placement mode probabilities (kept for future use; main layout uses _layout_candidates)
    ap.add_argument("--p-cand-top", type=float, default=0.12)
    ap.add_argument("--p-cand-bottom", type=float, default=0.12)
    ap.add_argument("--p-cand-left", type=float, default=0.12)
    ap.add_argument("--p-cand-right", type=float, default=0.12)
    ap.add_argument("--p-cand-edges", type=float, default=0.20)
    ap.add_argument("--p-cand-combo2", type=float, default=0.12)
    ap.add_argument("--p-cand-combo3", type=float, default=0.10)
    ap.add_argument("--p-cand-random", type=float, default=0.10)

    # candidate content
    ap.add_argument("--cand-k-min", type=int, default=2)
    ap.add_argument("--cand-k-max", type=int, default=7)
    ap.add_argument("--cand-scale-mean", type=float, default=18.0,
                    help="Target pixel height of pasted candidate patch.")
    ap.add_argument("--cand-scale-jitter", type=float, default=0.25)
    ap.add_argument("--cand-gray-min", type=int, default=110)
    ap.add_argument("--cand-gray-max", type=int, default=190)
    ap.add_argument("--cand-band-frac", type=float, default=0.10)
    ap.add_argument("--cand-pos-jitter", type=int, default=3)
    ap.add_argument("--cand-avoid-solution", action="store_true")
    ap.add_argument("--cand-allow-duplicates", action="store_true")

    # solution (handwritten, from patch)
    ap.add_argument("--sol-scale-mean", type=float, default=40.0,
                    help="Pixel height for solution patch (smaller for 64x64).")
    ap.add_argument("--sol-scale-jitter", type=float, default=0.12)
    ap.add_argument("--sol-gray-min", type=int, default=60)
    ap.add_argument("--sol-gray-max", type=int, default=140)
    ap.add_argument("--sol-center-x", type=float, default=0.50)
    ap.add_argument("--sol-center-y", type=float, default=0.50)
    ap.add_argument("--sol-pos-jitter", type=int, default=2)

    # given (printed font)
    ap.add_argument("--given-font-size", type=int, default=44)
    ap.add_argument("--given-font-jitterpct", type=float, default=0.10)
    ap.add_argument("--given-gray-min", type=int, default=0)
    ap.add_argument("--given-gray-max", type=int, default=40)
    ap.add_argument("--given-center-x", type=float, default=0.50)
    ap.add_argument("--given-center-y", type=float, default=0.50)
    ap.add_argument("--given-center-jitter-radius", type=int, default=4)
    ap.add_argument("--given-pos-jitter", type=int, default=1)
    ap.add_argument("--given-anchor", type=str, default="mm", choices=["mm","bbox"])

    # printed fonts (auto-discover if not provided)
    ap.add_argument("--font-printed", type=str, default="",
                    help="Comma-separated font paths (for givens). If empty, auto-discover.")
    ap.add_argument("--auto-font-keys", type=str,
                    default="arial,calibri,cambria,candara,consola,cour,georgia,segoeui,tahoma,trebuc,verdana,bahnschrift,framd,pala",
                    help="Keywords to auto-discover fonts in C:/Windows/Fonts.")

    # light grid occlusion / smudge
    ap.add_argument("--grid-occlusion-prob", type=float, default=0.35)
    ap.add_argument("--grid-occlusion-gray", type=int, default=130)
    ap.add_argument("--grid-occlusion-width", type=int, default=1)
    ap.add_argument("--smudge-prob", type=float, default=0.20)
    ap.add_argument("--smudge-radius-lo", type=float, default=0.0)
    ap.add_argument("--smudge-radius-hi", type=float, default=0.8)

    # global jitter (camera style)
    ap.add_argument("--jitter-contrast-lo", type=float, default=0.9)
    ap.add_argument("--jitter-contrast-hi", type=float, default=1.1)
    ap.add_argument("--jitter-brightness-lo", type=float, default=-10.0)
    ap.add_argument("--jitter-brightness-hi", type=float, default=10.0)
    ap.add_argument("--jitter-noise-std", type=float, default=4.0)

    # reproducibility
    ap.add_argument("--seed", type=int, default=1337)

    # candidate layout controls
    ap.add_argument("--min-cand-gap", type=int, default=7)
    ap.add_argument("--edge-bias", type=float, default=0.65)
    ap.add_argument("--center-reserved-frac", type=float, default=0.30)
    ap.add_argument("--max-overlap-frac", type=float, default=0.15)
    ap.add_argument("--repel-steps", type=int, default=8)
    ap.add_argument("--repel-strength", type=float, default=0.35)

    args = ap.parse_args()

    # Seeds → reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    S = int(args.cell_size)

    # Printed fonts pool — either explicit list or auto-discovery by keywords
    printed_fonts: List[str] = []
    if args.font_printed.strip():
        printed_fonts = [x.strip() for x in args.font_printed.split(",") if x.strip()]
    else:
        keys = [k.strip().lower() for k in args.auto_font_keys.split(",") if k.strip()]
        printed_fonts = discover_windows_fonts_by_keywords(keys)
    if not printed_fonts:
        print("[warn] No printed fonts found; will fall back to Pillow default for givens.")

    # Load digits manifest and bin by digit
    project_root = Path(args.project_root).resolve()
    rows = load_manifest(Path(args.digits_manifest))
    if not rows:
        raise SystemExit(f"[error] No digit records found in {args.digits_manifest}")
    bins = pick_patches_by_digit(rows)
    for d in range(10):
        if len(bins[d]) == 0:
            raise SystemExit(f"[error] No samples for digit {d} in manifest!")

    # Optional blank backgrounds
    blank_paths: List[str] = []
    if args.blank_bg_list.strip():
        p = Path(args.blank_bg_list)
        if p.is_file():
            with p.open("r", encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if ln:
                        blank_paths.append(ln)
            print(f"[bg] loaded {len(blank_paths)} blank backgrounds from {p}")
        else:
            print(f"[bg] WARNING: blank-bg-list {p} not found; using plain backgrounds.")

    # Output layout: one folder per tag, plus its JSONL manifest
    out_root = Path("datasets") / "cells" / "cell_interpreter" / args.out_tag
    ensure_dir(out_root)
    manifest_path = out_root / f"cells_{args.out_tag}.jsonl"
    fout = open(manifest_path, "w", encoding="utf-8")

    # Cell-type mixture (probabilities should sum to ~1.0)
    p_types = {
        "candonly": args.p_type_candonly,
        "solonly": args.p_type_solonly,
        "solpluscand": args.p_type_solpluscand,
        "givenonly": args.p_type_givenonly,
    }

    print(f"[config] cell_size={S} ; num_cells={args.num_cells}")

    # ──────────────────────────────────────────────────────────────────────────
    # Main generation loop
    # ──────────────────────────────────────────────────────────────────────────
    for idx in tqdm(range(args.num_cells), desc="[synth] cells", unit="cell"):
        # 1) Sample cell type by mixture
        cell_type = random.choices(
            population=list(p_types.keys()),
            weights=list(p_types.values()),
            k=1
        )[0]

        # 2) Background: either a real blank or a plain light gray canvas
        if blank_paths:
            bg_path = random.choice(blank_paths)
            try:
                im = Image.open(bg_path).convert("L").resize((S, S), Image.BICUBIC)
            except Exception:
                # If any path fails, fallback to plain background for robustness
                im = Image.fromarray(
                    np.full((S, S), random.randint(230, 255), dtype=np.uint8),
                    mode="L"
                )
        else:
            base_gray = random.randint(235, 255)
            im = Image.fromarray(np.full((S, S), base_gray, dtype=np.uint8), mode="L")

        given_digit = 0
        solution_digit = 0
        candidates: List[int] = []

        # 3) Solution (centered handwritten patch)
        if cell_type in ("solonly","solpluscand"):
            solution_digit = random.randint(1,9)
            src = random.choice(bins[solution_digit])
            patch = resolve_image(src, project_root)
            h_mean = args.sol_scale_mean
            h = int(round(h_mean * (1.0 + random.uniform(-args.sol_scale_jitter,
                                                         +args.sol_scale_jitter))))
            h = clamp(h, 10, S-4)
            cx = int(args.sol_center_x * S) + random.randint(-args.sol_pos_jitter,
                                                             args.sol_pos_jitter)
            cy = int(args.sol_center_y * S) + random.randint(-args.sol_pos_jitter,
                                                             args.sol_pos_jitter)
            cx = clamp(cx, 0, S-1)
            cy = clamp(cy, 0, S-1)
            paste_scaled_patch(im, patch, cx, cy, h,
                               args.sol_gray_min, args.sol_gray_max)

        # 4) Candidates (scattered small patches, possibly avoiding solution)
        if cell_type in ("candonly","solpluscand"):
            k = random.randint(args.cand_k_min, args.cand_k_max)
            all_digits = list(range(1, 10))
            if args.cand_avoid_solution and solution_digit != 0 and solution_digit in all_digits:
                all_digits.remove(solution_digit)

            if args.cand_allow_duplicates:
                cand_digits = [random.choice(all_digits) for _ in range(k)]
            else:
                k = min(k, len(all_digits))
                cand_digits = random.sample(all_digits, k)

            approx_side = int(max(6, min(S//2, args.cand_scale_mean)))
            centers = _layout_candidates(
                cell_W=S, cell_H=S, k=len(cand_digits),
                glyph_box_side=approx_side,
                center_reserved_frac=args.center_reserved_frac,
                min_gap=args.min_cand_gap,
                edge_bias=args.edge_bias,
                repel_steps=args.repel_steps,
                repel_strength=args.repel_strength,
            )

            placed_bboxes: List[Tuple[int,int,int,int]] = []
            placed_digits: List[int] = []

            for d, (cx, cy) in zip(cand_digits, centers):
                src = random.choice(bins[d])
                patch = resolve_image(src, project_root)
                h_mean = args.cand_scale_mean
                h = int(round(h_mean * (1.0 + random.uniform(-args.cand_scale_jitter,
                                                             +args.cand_scale_jitter))))
                h = clamp(h, 6, S//2)

                accepted = False
                for _try in range(6):
                    if _try > 0:
                        jx = random.randint(-args.cand_pos_jitter, args.cand_pos_jitter)
                        jy = random.randint(-args.cand_pos_jitter, args.cand_pos_jitter)
                        cx_try = clamp(cx + jx, 0, S-1)
                        cy_try = clamp(cy + jy, 0, S-1)
                    else:
                        cx_try, cy_try = cx, cy

                    bbox = estimate_patch_bbox(patch, h, cx_try, cy_try)
                    too_much = any(_boxes_overlap_frac(bbox, b) > args.max_overlap_frac
                                   for b in placed_bboxes)
                    if not too_much:
                        paste_scaled_patch(im, patch, cx_try, cy_try, h,
                                           args.cand_gray_min, args.cand_gray_max)
                        placed_bboxes.append(bbox)
                        placed_digits.append(d)
                        accepted = True
                        break

            candidates = (sorted(set(placed_digits))
                          if not args.cand_allow_duplicates else placed_digits)

        # 5) Given (printed digit in clean font)
        if cell_type == "givenonly":
            given_digit = random.randint(1,9)
            font_path = random.choice(printed_fonts) if printed_fonts else ""
            draw_given_digit(
                im, given_digit, font_path,
                px=args.given_font_size,
                jitter_pct=args.given_font_jitterpct,
                gray_min=args.given_gray_min, gray_max=args.given_gray_max,
                center_frac_xy=(args.given_center_x, args.given_center_y),
                pos_jitter_px=args.given_pos_jitter,
                center_jitter_radius=args.given_center_jitter_radius,
                anchor_mode=args.given_anchor
            )

        # 6) Light degradations: smudge then grid hints
        im = light_smudge(im, args.smudge_prob,
                        args.smudge_radius_lo, args.smudge_radius_hi)
        im = draw_grid_occlusion(im, args.grid_occlusion_prob,
                                args.grid_occlusion_gray,
                                args.grid_occlusion_width)

        # 7) Camera-style jitter last (global intensity + noise)
        im = apply_global_jitter(
            im,
            contrast_lo=args.jitter_contrast_lo,
            contrast_hi=args.jitter_contrast_hi,
            brightness_lo=args.jitter_brightness_lo,
            brightness_hi=args.jitter_brightness_hi,
            noise_std=args.jitter_noise_std,
        )

        # 8) Persist image + JSONL record
        img_name = f"img_{idx:06d}.png"
        out_img = (out_root / img_name)
        im.save(out_img, format="PNG")

        rec = {
            "path": str(out_img).replace("\\","/"),
            "given_digit": int(given_digit),
            "solution_digit": int(solution_digit),
            "candidates": [int(x) for x in candidates],
            "source": args.out_tag,
        }
        fout.write(json.dumps(rec) + "\n")

    # Cleanup
    fout.close()
    print(f"[done] wrote {args.num_cells} cells to {out_root}")
    print(f"[done] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
