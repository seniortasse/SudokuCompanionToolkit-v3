# Synthetic Sudoku Cell Generator (v4-edge)
# -----------------------------------------
# Changes vs v3:
# - Candidates placed anywhere (uniform XY) with edge-hugging bias (40–50% near border).
# - More faint pencil strokes + increased eraser marks when candidates exist.
# - Partial occlusion by faint grid lines drawn after digits.
# - Keeps your given/solution logic and overall rendering pipeline.

from __future__ import annotations
import argparse, json, random
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter


# =========================================================
# Scenario sampling
# =========================================================

def sample_scenario() -> dict:
    r = random.random()
    if r < 0.20:
        return {"given_digit": 0, "solution_digit": 0, "candidates": []}
    elif r < 0.45:
        d = random.randint(1, 9)
        return {"given_digit": d, "solution_digit": 0, "candidates": []}
    elif r < 0.70:
        d = random.randint(1, 9)
        return {"given_digit": 0, "solution_digit": d, "candidates": []}
    elif r < 0.85:
        k = random.randint(1, 5)
        cand = sorted(random.sample(range(1, 10), k))
        return {"given_digit": 0, "solution_digit": 0, "candidates": cand}
    else:
        sol = random.randint(1, 9)
        pool = [d for d in range(1, 10) if d != sol]
        k = random.randint(1, min(4, len(pool)))
        cand = sorted(random.sample(pool, k))
        return {"given_digit": 0, "solution_digit": sol, "candidates": cand}


def sample_style_mode() -> int:
    """
    1 = upright (30%)
    2 = slight slant (40%)
    3 = full realism (30%)
    """
    r = random.random()
    if r < 0.30:
        return 1
    elif r < 0.70:
        return 2
    else:
        return 3


# =========================================================
# Font & drawing helpers
# =========================================================

def parse_font_list(arg: str) -> List[str]:
    if not arg:
        return []
    parts = [p.strip() for p in arg.split(",")]
    return [p for p in parts if p]


def choose_font_path(font_paths: List[str]) -> Optional[str]:
    if not font_paths:
        return None
    return random.choice(font_paths)


_missing_font_cache = set()


def make_font(path: Optional[str], px: int) -> ImageFont.FreeTypeFont:
    """
    Safe font loader:
      - Tries to load the requested font.
      - If missing/corrupt → falls back to ImageFont.load_default().
      - Warns only once per missing font path.
    """
    if path is None or path == "":
        return ImageFont.load_default()

    try:
        return ImageFont.truetype(path, size=px)
    except Exception:
        if path not in _missing_font_cache:
            print(f"[warn] font not found or unreadable, using default: {path}")
            _missing_font_cache.add(path)
        return ImageFont.load_default()


def measure_text(draw: ImageDraw.Draw, text: str, font: ImageFont.FreeTypeFont) -> Tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def is_script_font(path: Optional[str]) -> bool:
    """
    Heuristic: detect handwriting/script fonts by filename.
    Used to slightly thin strokes so they look more pencil-like.
    """
    if not path:
        return False
    lower = path.lower()
    script_keys = [
        "segoepr",   # Segoe Print
        "segoesc",   # Segoe Script
        "lhandw",    # Lucida Handwriting
        "bradhitc",  # Bradley Hand ITC
        "inkfree",   # Ink Free
        "itckrist",  # Kristen ITC
        "mistral",   # Mistral
    ]
    return any(k in lower for k in script_keys)


# =========================================================
# Pencil profiles / colors
# =========================================================

def sample_pencil_profile() -> Dict[str, float]:
    """
    Per-cell "pencil hardness" model.

    25% of the time: classic dark ink
    75% of the time: pencil with different hardness levels.
    """
    r = random.random()
    if r < 0.25:
        return {
            "mode": "ink",
            "name": "INK",
            "fg_min": 0,
            "fg_max": 45,
            "stroke_mult": 1.1,
            "smudge_prob": 0.10,
            "smudge_strength": 10.0,
            "eraser_prob": 0.10,
        }

    r2 = random.random()
    if r2 < 0.25:
        # H: harder, lighter, thinner
        return {
            "mode": "pencil",
            "name": "H",
            "fg_min": 140,   # slightly lighter band
            "fg_max": 220,
            "stroke_mult": 0.7,
            "smudge_prob": 0.18,  # a bit more smudge overall
            "smudge_strength": 8.0,
            "eraser_prob": 0.22,  # more frequent eraser
        }
    elif r2 < 0.55:
        # HB: medium hardness / darkness
        return {
            "mode": "pencil",
            "name": "HB",
            "fg_min": 95,
            "fg_max": 185,
            "stroke_mult": 0.9,
            "smudge_prob": 0.22,
            "smudge_strength": 10.0,
            "eraser_prob": 0.26,
        }
    elif r2 < 0.80:
        # B: softer, darker, thicker
        return {
            "mode": "pencil",
            "name": "B",
            "fg_min": 65,
            "fg_max": 155,
            "stroke_mult": 1.1,
            "smudge_prob": 0.27,
            "smudge_strength": 12.0,
            "eraser_prob": 0.28,
        }
    else:
        # 2B: very soft, dark, thicker lines, smudges easily
        return {
            "mode": "pencil",
            "name": "2B",
            "fg_min": 45,
            "fg_max": 125,
            "stroke_mult": 1.3,
            "smudge_prob": 0.32,
            "smudge_strength": 14.0,
            "eraser_prob": 0.32,
        }


def sample_handwritten_color(profile: Dict[str, float]) -> int:
    """
    Sample a grayscale value for handwritten digits based on profile.
    """
    if profile["mode"] == "ink":
        return random.randint(0, 45)
    return random.randint(int(profile["fg_min"]), int(profile["fg_max"]))


# =========================================================
# Image effects (background, shadows, smudges, eraser)
# =========================================================

def make_base_background(size: int) -> np.ndarray:
    bg_val = random.randint(235, 255)
    arr = np.full((size, size), float(bg_val), dtype=np.float32)

    # Lighting gradient
    if random.random() < 0.9:
        mode = random.choice(["horizontal", "vertical", "diag1", "diag2"])
        amp = random.uniform(-25.0, 25.0)
        xs = np.linspace(-1.0, 1.0, size, dtype=np.float32)
        ys = np.linspace(-1.0, 1.0, size, dtype=np.float32)
        if mode == "horizontal":
            grad = np.tile(xs, (size, 1))
        elif mode == "vertical":
            grad = np.tile(ys[:, None], (1, size))
        elif mode == "diag1":
            grad = (np.add.outer(ys, xs)) / 2.0
        else:  # diag2
            grad = (np.add.outer(ys, -xs)) / 2.0
        arr += amp * grad

    # Paper grain / noise
    noise_scale = random.uniform(0.15, 0.40)
    noise = np.random.normal(0.0, noise_scale * 12.0, arr.shape).astype(np.float32)
    arr += noise

    arr = np.clip(arr, 0.0, 255.0)
    return arr


def apply_shadow(arr: np.ndarray) -> np.ndarray:
    size = arr.shape[0]
    if random.random() < 0.5:
        return arr

    direction = random.choice(["top", "bottom", "left", "right"])
    strength = random.uniform(8.0, 25.0)

    xs = np.linspace(0.0, 1.0, size, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, size, dtype=np.float32)

    if direction == "top":
        mask = np.tile(ys.reshape(-1, 1), (1, size))
        mask = 1.0 - mask
    elif direction == "bottom":
        mask = np.tile(ys.reshape(-1, 1), (1, size))
    elif direction == "left":
        mask = np.tile(xs.reshape(1, -1), (size, 1))
        mask = 1.0 - mask
    else:  # right
        mask = np.tile(xs.reshape(1, -1), (size, 1))

    arr = arr - strength * mask
    arr = np.clip(arr, 0.0, 255.0)
    return arr


def apply_perspective(im: Image.Image, max_jitter_frac: float = 0.02, prob: float = 0.6) -> Image.Image:
    if random.random() > prob:
        return im

    w, h = im.size
    jitter = min(w, h) * max_jitter_frac

    def j(x, y):
        return x + random.uniform(-jitter, jitter), y + random.uniform(-jitter, jitter)

    tl = j(0.0, 0.0)
    tr = j(float(w), 0.0)
    br = j(float(w), float(h))
    bl = j(0.0, float(h))

    quad = [tl[0], tl[1], tr[0], tr[1], br[0], br[1], bl[0], bl[1]]
    im = im.transform((w, h), Image.QUAD, quad, resample=Image.BICUBIC)
    return im


def apply_slant(im: Image.Image, max_angle_deg: float = 3.0, min_angle_abs: float = 0.2) -> Image.Image:
    angle = random.uniform(-max_angle_deg, max_angle_deg)
    if abs(angle) < min_angle_abs:
        return im
    bg = int(np.median(np.array(im)))
    return im.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=bg)


def apply_eraser_marks(im: Image.Image, profile: Dict[str, float], extra_boost: float = 0.0) -> Image.Image:
    """
    Lighten small patches as if the user erased something.
    extra_boost increases probability when candidates exist.
    """
    base_prob = profile.get("eraser_prob", 0.15)
    prob = min(0.95, base_prob + extra_boost)  # clamp
    if random.random() > prob:
        return im

    arr = np.array(im, dtype=np.float32)
    h, w = arr.shape

    num_marks = random.randint(1, 3)
    for _ in range(num_marks):
        cx = random.randint(int(w * 0.1), int(w * 0.9))
        cy = random.randint(int(h * 0.1), int(h * 0.9))
        radius = random.randint(max(3, h // 10), max(4, h // 4))

        yy, xx = np.ogrid[:h, :w]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2

        lighten = random.uniform(8.0, 28.0)
        arr[mask] = np.clip(arr[mask] + lighten, 0.0, 255.0)

    return Image.fromarray(arr.astype(np.uint8), mode="L")


def apply_smudges(im: Image.Image, profile: Dict[str, float]) -> Image.Image:
    prob = profile.get("smudge_prob", 0.2)
    strength = profile.get("smudge_strength", 10.0)

    if random.random() > prob:
        return im

    arr = np.array(im, dtype=np.float32)
    h, w = arr.shape

    num_smudges = random.randint(1, 2)
    for _ in range(num_smudges):
        cx = random.randint(int(w * 0.2), int(w * 0.8))
        cy = random.randint(int(h * 0.2), int(h * 0.8))
        radius = random.randint(max(4, h // 6), max(6, h // 3))

        yy, xx = np.ogrid[:h, :w]
        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
        mask = dist2 <= radius ** 2

        darken = random.uniform(strength * 0.5, strength * 1.2)
        arr[mask] = np.clip(arr[mask] - darken, 0.0, 255.0)

    return Image.fromarray(arr.astype(np.uint8), mode="L")


# =========================================================
# Candidate placement utilities
# =========================================================

def sample_candidate_positions(
    size: int,
    n_digits: int,
    edge_frac: float = None,
    edge_band_min: float = 0.10,
    edge_band_max: float = 0.15,
) -> List[Tuple[int, int]]:
    """
    Return 'n_digits' (x,y) centers for candidate digits.
    - 'edge_frac': portion placed near edges (if None, random in [0.40,0.50])
    - Edge band is a ring inside the cell within 10–15% from the border.
    """
    if edge_frac is None:
        edge_frac = random.uniform(0.40, 0.50)

    k_edge = int(round(n_digits * edge_frac))
    k_free = n_digits - k_edge

    # Free placements: uniform over full cell with small safety margin
    margin = int(0.06 * size)  # avoid clipping glyphs
    free_pts = []
    for _ in range(max(0, k_free)):
        x = random.randint(margin, size - margin)
        y = random.randint(margin, size - margin)
        free_pts.append((x, y))

    # Edge placements: centers constrained to an inner "ring"
    band = random.uniform(edge_band_min, edge_band_max)
    t = int(band * size)
    edge_pts = []
    for _ in range(max(0, k_edge)):
        side = random.choice(["top", "bottom", "left", "right"])
        if side in ("top", "bottom"):
            x = random.randint(margin, size - margin)
            y = random.randint(margin, margin + t) if side == "top" else random.randint(size - margin - t, size - margin)
        else:
            y = random.randint(margin, size - margin)
            x = random.randint(margin, margin + t) if side == "left" else random.randint(size - margin - t, size - margin)
        edge_pts.append((x, y))

    pts = free_pts + edge_pts
    random.shuffle(pts)
    return pts[:n_digits]


def overlay_faint_grid_occluders(im: Image.Image) -> Image.Image:
    """
    Draw a couple of faint lines over the cell AFTER digits,
    to partially occlude strokes (simulating grid lines / paper artifacts).
    """
    draw = ImageDraw.Draw(im)
    w, h = im.size

    # choose 1–3 faint lines
    n_lines = random.randint(1, 3)
    for _ in range(n_lines):
        # vertical or horizontal
        if random.random() < 0.5:
            # vertical
            x = random.randint(int(0.15*w), int(0.85*w))
            col = random.randint(170, 210)  # faint
            width = random.choice([1, 1, 2])
            draw.line([(x, 0), (x, h)], fill=col, width=width)
        else:
            # horizontal
            y = random.randint(int(0.15*h), int(0.85*h))
            col = random.randint(170, 210)
            width = random.choice([1, 1, 2])
            draw.line([(0, y), (w, y)], fill=col, width=width)

    return im


# =========================================================
# Cell rendering
# =========================================================

def render_cell(
    size: int,
    given_digit: int,
    solution_digit: int,
    candidates: List[int],
    printed_fonts: List[str],
    hand_fonts: List[str],
    style_mode: int,
) -> Image.Image:
    """
    style_mode:
      1 = upright (no slant/perspective)
      2 = slight digit slant
      3 = full realism (slant + perspective)
    """
    pencil_profile = sample_pencil_profile()

    # 1) base paper
    arr = make_base_background(size)
    arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    arr = apply_shadow(arr)

    im = Image.fromarray(arr, mode="L")
    draw = ImageDraw.Draw(im)

    # 2) border / grid line (under everything)
    border_val = random.randint(80, 130)
    margin = max(1, size // 32)
    draw.rectangle(
        [margin, margin, size - 1 - margin, size - 1 - margin],
        outline=border_val,
        width=max(1, size // 64),
    )

    cx = size // 2
    cy = size // 2

    # Jitter scales per style
    if style_mode == 1:
        given_jitter = size // 64
        sol_jitter = size // 48
        cand_jitter = size // 48
        max_slant = 0.0
        persp_jitter = 0.0
    elif style_mode == 2:
        given_jitter = size // 48
        sol_jitter = size // 32
        cand_jitter = size // 32
        max_slant = 1.8
        persp_jitter = 0.0
    else:  # style_mode == 3
        given_jitter = size // 48
        sol_jitter = size // 32
        cand_jitter = size // 32
        max_slant = 2.5
        persp_jitter = 0.02  # up to 2%

    # --------------------------
    # GIVEN DIGIT (printed)
    # --------------------------
    if given_digit != 0:
        text = str(given_digit)
        px = int(size * random.uniform(0.55, 0.72))
        font_path = choose_font_path(printed_fonts)
        font_p = make_font(font_path, px)

        w, h = measure_text(draw, text, font_p)
        dx = random.randint(-given_jitter, given_jitter)
        dy = random.randint(-given_jitter, given_jitter)
        x = cx - w // 2 + dx
        y = cy - h // 2 + dy

        fg = random.randint(0, 50)
        draw.text((x, y), text, font=font_p, fill=fg)

    # --------------------------
    # SOLUTION DIGIT (handwritten)
    # --------------------------
    if solution_digit != 0:
        text = str(solution_digit)
        px = int(size * random.uniform(0.58, 0.72))
        font_path = choose_font_path(hand_fonts)
        font_h = make_font(font_path, px)

        w, h = measure_text(draw, text, font_h)
        dx = random.randint(-sol_jitter, sol_jitter)
        dy = random.randint(-sol_jitter, sol_jitter)
        x = cx - w // 2 + dx
        y = cy - h // 2 + dy

        base_stroke = random.uniform(0.010, 0.030) * pencil_profile["stroke_mult"]
        stroke_w = max(1, int(size * base_stroke))
        if is_script_font(font_path):
            stroke_w = max(1, int(stroke_w * 0.8))

        fg = sample_handwritten_color(pencil_profile)
        draw.text(
            (x, y),
            text,
            font=font_h,
            fill=fg,
            stroke_width=stroke_w,
            stroke_fill=fg,
        )

    # --------------------------
    # CANDIDATES (full-cell, with edge bias)
    # --------------------------
    if candidates:
        # Slightly smaller + lighter than center solution
        px = int(size * random.uniform(0.17, 0.28))
        font_path = choose_font_path(hand_fonts)
        font_c = make_font(font_path, px)

        # More faintness when candidates exist (to mimic pencil notes)
        # We do this by biasing grayscale brighter and reducing stroke thickness a touch.
        # We'll also raise eraser mark probability later.
        candidate_color_bias = random.uniform(0.05, 0.18)  # extra brightness
        cand_pts = sample_candidate_positions(
            size=size,
            n_digits=len(candidates),
            edge_frac=None,  # random in [0.40, 0.50]
            edge_band_min=0.10,
            edge_band_max=0.15,
        )

        for d, (cx_d, cy_d) in zip(candidates, cand_pts):
            text = str(d)
            w, h = measure_text(draw, text, font_c)

            dx = random.randint(-cand_jitter, cand_jitter)
            dy = random.randint(-cand_jitter, cand_jitter)
            x = int(cx_d - w / 2 + dx)
            y = int(cy_d - h / 2 + dy)

            # Color: push pencil to lighter range
            fg = sample_handwritten_color(pencil_profile)
            if pencil_profile["mode"] == "pencil":
                fg = int(np.clip(fg + candidate_color_bias * 255, 0, 255))

            draw.text((x, y), text, font=font_c, fill=fg)

    # --------------------------
    # Random scribbles
    # --------------------------
    if random.random() < 0.18:
        n = random.randint(1, 3)
        for _ in range(n):
            x0 = random.randint(0, size)
            y0 = random.randint(0, size)
            x1 = random.randint(0, size)
            y1 = random.randint(0, size)
            val = random.randint(110, 200)
            width = max(1, size // random.choice([80, 96, 128]))
            draw.line((x0, y0, x1, y1), fill=val, width=width)

    # --------------------------
    # Eraser marks & smudges (after digits)
    # If there are candidates, boost eraser probability
    # --------------------------
    extra_eraser = 0.10 if candidates else 0.0
    im = apply_eraser_marks(im, pencil_profile, extra_boost=extra_eraser)
    im = apply_smudges(im, pencil_profile)

    # --------------------------
    # Partial occlusion by faint lines (AFTER digits)
    # --------------------------
    if random.random() < 0.60:
        im = overlay_faint_grid_occluders(im)

    # --------------------------
    # Slant & perspective per style
    # --------------------------
    if style_mode in (2, 3) and max_slant > 0.0:
        im = apply_slant(im, max_angle_deg=max_slant, min_angle_abs=0.3)

    if style_mode == 3 and persp_jitter > 0.0:
        im = apply_perspective(im, max_jitter_frac=persp_jitter, prob=0.8)

    # Slight blur sometimes
    if random.random() < 0.35:
        im = im.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.0, 0.8)))

    return im


# =========================================================
# Main
# =========================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-tag", type=str, default="synth_v4_edge",
                    help="Name for this synthetic dataset variant")
    ap.add_argument("--num-cells", type=int, default=20000,
                    help="Number of cells to generate")
    ap.add_argument("--img-size", type=int, default=64,
                    help="Cell image size (H=W)")
    ap.add_argument("--font-printed", type=str, default="",
                    help="Comma-separated TTF paths for printed digits")
    ap.add_argument("--font-hand", type=str, default="",
                    help="Comma-separated TTF paths for handwritten digits")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    printed_fonts = parse_font_list(args.font_printed)
    hand_fonts = parse_font_list(args.font_hand)

    if not printed_fonts:
        print("[warn] No printed fonts provided; using Pillow default font.")
    if not hand_fonts:
        print("[warn] No handwritten fonts provided; using Pillow default font.")

    root = Path("datasets") / "cells" / "cell_interpreter" / args.out_tag
    root.mkdir(parents=True, exist_ok=True)

    manifest_path = root / f"cells_{args.out_tag}.jsonl"

    print(f"[gen] Writing images under: {root}")
    print(f"[gen] JSONL manifest:      {manifest_path}")
    print(f"[gen] num_cells={args.num_cells}, size={args.img_size}")

    with manifest_path.open("w", encoding="utf-8") as fout:
        for idx in range(args.num_cells):
            scenario = sample_scenario()
            style_mode = sample_style_mode()

            im = render_cell(
                size=args.img_size,
                given_digit=scenario["given_digit"],
                solution_digit=scenario["solution_digit"],
                candidates=scenario["candidates"],
                printed_fonts=printed_fonts,
                hand_fonts=hand_fonts,
                style_mode=style_mode,
            )

            img_name = f"img_{idx:06d}.png"
            img_rel = Path("datasets") / "cells" / "cell_interpreter" / args.out_tag / img_name
            img_abs = img_rel.resolve()
            img_abs.parent.mkdir(parents=True, exist_ok=True)
            im.save(img_abs, format="PNG")

            rec = {
                "path": str(img_rel).replace("\\", "/"),
                "given_digit": int(scenario["given_digit"]),
                "solution_digit": int(scenario["solution_digit"]),
                "candidates": [int(d) for d in scenario["candidates"]],
                "source": args.out_tag,
            }
            fout.write(json.dumps(rec) + "\n")

            if (idx + 1) % 1000 == 0:
                print(f"  [{idx+1}/{args.num_cells}]")

    print("[gen] Done.")


if __name__ == "__main__":
    main()