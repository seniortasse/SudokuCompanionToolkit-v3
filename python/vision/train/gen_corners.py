# ===============================================================================
# Synthetic Sudoku Corner Heatmaps — Dataset Generator (Mode-Aware)
# ===============================================================================

from __future__ import annotations
import argparse, json, random, os, io, time
from pathlib import Path
from typing import Tuple, List, Optional, Dict

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# ──────────────────────────────────────────────────────────────────────────────
# 1) HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def safe_save_npy(path, arr, retries=3, sleep=0.1):
    path = str(path)
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    tmp = path + ".tmp"
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    data = buf.getvalue()
    for k in range(retries):
        try:
            with open(tmp, "wb", buffering=0) as f:
                f.write(data); f.flush(); os.fsync(f.fileno())
            os.replace(tmp, path)
            return
        except Exception:
            try:
                if os.path.exists(tmp): os.remove(tmp)
            except: pass
            if k == retries - 1: raise
            time.sleep(sleep * (k + 1))

def try_load_ttf(size: int) -> Optional[ImageFont.FreeTypeFont]:
    candidates = [
        "DejaVuSans-Bold.ttf", "DejaVuSans.ttf",
        "Arial.ttf", "Arial Bold.ttf", "LiberationSans-Bold.ttf",
        "SegoeUI-Bold.ttf", "Segoe UI Bold.ttf", "ComicSansMS.ttf",
        "Verdana.ttf", "Tahoma.ttf"
    ]
    for name in candidates:
        try: return ImageFont.truetype(name, size=size)
        except Exception: pass
    return None

def draw_text_scaled(base_img: Image.Image, text: str, xy: tuple[int, int],
                     target_h: int, color=(0, 0, 0)):
    draw = ImageDraw.Draw(base_img)
    font = try_load_ttf(size=max(8, target_h))
    if font is not None:
        draw.text(xy, text, fill=color, font=font); return
    # Fallback: bitmap upscale
    tmp = Image.new("RGBA", (target_h * 2, target_h * 2), (0, 0, 0, 0))
    d2 = ImageDraw.Draw(tmp)
    fdef = ImageFont.load_default()
    d2.text((2, 2), text, fill=(*color, 255), font=fdef)
    bbox = d2.textbbox((2, 2), text, font=fdef)
    th = max(1, bbox[3] - bbox[1])
    scale = max(1, int(target_h / th))
    scaled = tmp.resize((tmp.width * scale, tmp.height * scale), Image.NEAREST)
    base_img.alpha_composite(scaled, dest=xy) if base_img.mode == "RGBA" else base_img.paste(scaled, xy, scaled)

def _normalize_ratios(d: Dict[str, float]) -> Dict[str, float]:
    total = float(sum(max(0.0, v) for v in d.values()))
    if total <= 0.0:
        return {k: 0.0 for k in d.keys()}
    return {k: max(0.0, v) / total for k, v in d.items()}

def _pick_from_ratios(rng: random.Random, ratios: Dict[str, float]) -> str:
    r = rng.random()
    acc = 0.0
    for k, v in ratios.items():
        acc += v
        if r <= acc:
            return k
    # numerical tail
    return next(iter(ratios.keys()))

# ──────────────────────────────────────────────────────────────────────────────
# 2) GENERATOR
# ──────────────────────────────────────────────────────────────────────────────

class SudokuCornerGenerator:
    """
    Mode-aware synthetic generator.

    Primary modes (mutually exclusive, one per sample):
      - straight_strict   : axis-aligned rectangle with margin (no perspective/curvature)
      - straight_loose    : small inside jitter (mild perspective, fully inside)
      - warped_inside     : perspective warp inside frame
      - warped_partial    : perspective warp with partial out-of-frame
      - curved_inside     : perspective + page curvature (inside frame)
      - curved_partial    : perspective + curvature + partial out-of-frame
    """
    ALL_MODES = [
        "straight_strict",
        "straight_loose",
        "warped_inside",
        "warped_partial",
        "curved_inside",
        "curved_partial",
    ]

    def __init__(self,
        img_size: int = 128,
        tile_size: int = 512,
        seed: Optional[int] = None,

        # Legacy knobs (used only when no explicit --mode-ratios provided)
        p_warp: float = 0.50,
        p_attached_header: float = 0.10,
        strict_axis_straight: bool = False,
        strict_margin_frac: float = 0.08,

        # Heatmap Gaussian sigma controls
        sigma: float = 2.2,
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,

        # New: explicit mode mixture (normalized inside ctor)
        mode_ratios: Optional[Dict[str, float]] = None,
    ):
        self.img_size = img_size
        self.tile_size = tile_size
        self.p_warp = float(p_warp)
        self.p_attached_header = p_attached_header
        self.strict_axis_straight = strict_axis_straight
        self.strict_margin_frac = strict_margin_frac

        self.sigma = float(sigma)
        self.sigma_min = float(sigma_min) if sigma_min is not None else None
        self.sigma_max = float(sigma_max) if sigma_max is not None else None

        self.rng = random.Random(seed)
        if seed is not None:
            random.seed(seed); np.random.seed(seed)

        # Normalize & validate mode ratios
        self.mode_ratios = None
        if mode_ratios:
            # keep only supported modes, normalize
            filt = {k: float(mode_ratios[k]) for k in mode_ratios if k in self.ALL_MODES}
            if not filt:
                raise ValueError("mode_ratios provided, but none match supported modes.")
            self.mode_ratios = _normalize_ratios(filt)
            # Deterministic order for sampling
            self.mode_ratios = {k: self.mode_ratios[k] for k in self.ALL_MODES if k in self.mode_ratios}

    # ---------------- Core sampling ----------------

    def sample(self) -> Tuple[np.ndarray, List[Tuple[int, int]], np.ndarray, str]:
        """
        Returns:
          img (H,W,3) uint8, corners [(x,y)*4], heatmaps (4,H,W) float32, mode_name (str)
        """
        base = self._draw_grid()
        base = self._fill_cells(base)

        # Decide primary mode
        if self.mode_ratios is not None:
            mode = _pick_from_ratios(self.rng, self.mode_ratios)
        else:
            # Legacy fallback: straight vs warped by p_warp; “curved” only when warped
            if self.rng.random() < self.p_warp:
                mode = "warped_inside"
                # 25% of warped become curved_inside (soft legacy flavor)
                if self.rng.random() < 0.25:
                    mode = "curved_inside"
            else:
                mode = "straight_strict" if self.strict_axis_straight else "straight_loose"

        # Apply geometry for the chosen mode
        base, corners = self._apply_geometry(base, mode)

        # Orthogonal attributes (may or may not apply; not part of "mode")
        if self.rng.random() < self.p_attached_header:
            base = self._add_attached_header(base, corners)

        # Background + photo noise/blur/compression
        base = self._add_paper_background(base)
        base = self._add_noise(base, force_curvature=("curved" in mode))

        # Resize to output, scale corners
        base = cv2.resize(base, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        scale = self.img_size / float(self.tile_size)
        corners = [(int(np.clip(x * scale, 0, self.img_size - 1)),
                    int(np.clip(y * scale, 0, self.img_size - 1))) for (x, y) in corners]

        heatmaps = self._make_heatmaps(corners)
        return base, corners, heatmaps, mode

    # ---------------- Drawing primitives ----------------

    def _draw_grid(self) -> np.ndarray:
        base_tone = self.rng.randint(236, 248)
        img = np.ones((self.tile_size, self.tile_size, 3), np.uint8) * base_tone
        thin = self.rng.randint(1, 3)
        thick = thin + self.rng.randint(1, 3)
        step = self.tile_size // 9
        for i in range(10):
            w = thick if i % 3 == 0 else thin
            y = i * step; x = i * step
            cv2.line(img, (0, y), (self.tile_size, y), (0, 0, 0), w)
            cv2.line(img, (x, 0), (x, self.tile_size), (0, 0, 0), w)
        return img

    def _fill_cells(self, img: np.ndarray) -> np.ndarray:
        H, W = img.shape[:2]; step = W // 9
        pil_img = Image.fromarray(img).convert("RGBA")
        fill_mode = self.rng.choices(["empty", "partial", "full"], weights=[0.10, 0.60, 0.30], k=1)[0]
        if fill_mode == "empty":
            return np.array(pil_img.convert("RGB"))
        if fill_mode == "partial":
            target_digits = self.rng.randint(20, 34)
            cells = [(r, c) for r in range(9) for c in range(9)]
            self.rng.shuffle(cells)
            chosen = set(cells[:target_digits])
        else:
            chosen = {(r, c) for r in range(9) for c in range(9)}
        for r in range(9):
            for c in range(9):
                x0, y0 = c * step, r * step
                if (r, c) not in chosen:
                    if self.rng.random() < 0.15: self._draw_notes(pil_img, x0, y0, step)
                    continue
                mode = self.rng.choices(["digit", "notes", "both"], weights=[0.7, 0.2, 0.1])[0]
                if mode in ["digit", "both"]:
                    num = str(self.rng.randint(1, 9))
                    jitter_x = self.rng.randint(step // 8, step // 3)
                    jitter_y = self.rng.randint(step // 8, step // 3)
                    color = (0, 0, 0) if self.rng.random() < 0.9 else (70, 70, 70)
                    draw_text_scaled(pil_img, num, (x0 + jitter_x, y0 + jitter_y),
                                     target_h=int(step * 0.62), color=color)
                if mode in ["notes", "both"]:
                    self._draw_notes(pil_img, x0, y0, step)
        return np.array(pil_img.convert("RGB"))

    @staticmethod
    def _draw_notes(pil_img: Image.Image, x0: int, y0: int, step: int):
        rng = random
        note_digits = rng.sample("123456789", k=rng.randint(2, 4))
        positions = [
            (x0 + 3, y0 + int(step * 0.28)),
            (x0 + step - int(step * 0.35), y0 + int(step * 0.28)),
            (x0 + 3, y0 + step - int(step * 0.10)),
            (x0 + step - int(step * 0.35), y0 + step - int(step * 0.10)),
        ]
        chosen_pos = rng.sample(positions, k=len(note_digits))
        for nd, pos in zip(note_digits, chosen_pos):
            draw_text_scaled(pil_img, nd, (int(pos[0]), int(pos[1])),
                             target_h=max(10, int(step * 0.28)), color=(70, 70, 70))

    # ---------------- Geometry modes ----------------

    def _apply_geometry(self, img: np.ndarray, mode: str) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        H, W = img.shape[:2]
        pts_src = np.float32([[0, 0], [W - 1, 0], [0, H - 1], [W - 1, H - 1]])

        def axis_aligned_inset(margin_frac: float) -> np.ndarray:
            m = int(min(H, W) * margin_frac)
            x0, y0 = m, m
            x1, y1 = W - 1 - m, H - 1 - m
            return np.float32([[x0, y0], [x1, y0], [x0, y1], [x1, y1]])

        def inside_variant(jratio_low=0.05, jratio_high=0.15, jitter_frac=0.12):
            m = self.rng.randint(int(min(H, W) * jratio_low), int(min(H, W) * jratio_high))
            x0, y0 = m, m; x1, y1 = W - 1 - m, H - 1 - m
            rect = np.float32([[x0, y0], [x1, y0], [x0, y1], [x1, y1]])
            j = int(jitter_frac * min(H, W))
            def jit(v): return float(np.clip(v + self.rng.uniform(-j, j), 0, W - 1))
            def jit_y(v): return float(np.clip(v + self.rng.uniform(-j, j), 0, H - 1))
            pts = rect.copy()
            for i in range(4):
                pts[i, 0] = jit(pts[i, 0])
                pts[i, 1] = jit_y(pts[i, 1])
            return pts

        def partial_out_variant(margin_frac=0.20):
            margin = int(margin_frac * min(H, W))
            def jx(v): return float(v + self.rng.uniform(-margin, margin))
            def jy(v): return float(v + self.rng.uniform(-margin, margin))
            return np.float32([[jx(0), jy(0)],
                               [jx(W - 1), jy(0)],
                               [jx(0), jy(H - 1)],
                               [jx(W - 1), jy(H - 1)]])

        if mode == "straight_strict":
            pts_dst = axis_aligned_inset(self.strict_margin_frac)
        elif mode == "straight_loose":
            # Mild inside jitter, less than warped_inside
            pts_dst = inside_variant(jratio_low=0.05, jratio_high=0.08, jitter_frac=0.06)
        elif mode == "warped_inside":
            pts_dst = inside_variant(jratio_low=0.05, jratio_high=0.15, jitter_frac=0.12)
        elif mode == "warped_partial":
            pts_dst = partial_out_variant(margin_frac=0.20)
        elif mode == "curved_inside":
            pts_dst = inside_variant(jratio_low=0.05, jratio_high=0.15, jitter_frac=0.12)
        elif mode == "curved_partial":
            pts_dst = partial_out_variant(margin_frac=0.20)
        else:
            raise ValueError(f"Unknown mode '{mode}'")

        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        warped = cv2.warpPerspective(img, M, (W, H), borderValue=(255, 255, 255))
        pts_dst_clipped = [(int(np.clip(x, 0, W - 1)), int(np.clip(y, 0, H - 1))) for (x, y) in pts_dst]
        return warped, pts_dst_clipped

    def _add_attached_header(self, img: np.ndarray, corners: List[Tuple[int, int]]) -> np.ndarray:
        H, W = img.shape[:2]
        min_y = min(y for (_, y) in corners)
        band_h = self.rng.randint(H // 16, H // 10)
        y0 = max(0, min_y - band_h)
        color = self._choice([(30, 30, 30), (60, 60, 60), (100, 100, 100), (180, 180, 180)])
        cv2.rectangle(img, (0, y0), (W, min_y), color, -1)
        pil_img = Image.fromarray(img).convert("RGBA")
        txt = self._choice([f"Hard #{self.rng.randint(1, 600)}", "Sudoku Expert", "Logic Puzzle"])
        txt_color = (255, 255, 255) if sum(color) < 120 else (0, 0, 0)
        draw_text_scaled(pil_img, txt, (10, y0 + max(10, band_h - 6)), target_h=int(band_h * 0.6), color=txt_color)
        return np.array(pil_img.convert("RGB"))

    def _choice(self, seq):
        return seq[self.rng.randrange(len(seq))]

    def _add_paper_background(self, img: np.ndarray) -> np.ndarray:
        H, W = img.shape[:2]
        tint = self.rng.randint(238, 248)
        bg = np.ones((H, W, 3), np.uint8) * tint
        noise = cv2.GaussianBlur((np.random.randn(H, W).astype(np.float32) * 4.0), (0, 0), sigmaX=W // 12)
        noise = cv2.normalize(noise, None, -6, 6, cv2.NORM_MINMAX)
        bg = np.clip(bg.astype(np.float32) + noise[..., None], 0, 255).astype(np.uint8)
        img = cv2.addWeighted(img, 1.0, bg, 0.0, 0)  # keep grid as source
        Y, X = np.ogrid[:H, :W]; cx, cy = W / 2, H / 2
        r2 = ((X - cx) ** 2 + (Y - cy) ** 2) / (max(H, W) ** 2)
        vignette = 1.0 - 0.08 * r2
        return np.clip(img.astype(np.float32) * vignette[..., None], 0, 255).astype(np.uint8)

    def _add_noise(self, img: np.ndarray, force_curvature: bool) -> np.ndarray:
        H, W = img.shape[:2]

        # white rectangle occluder (e.g., sticker)
        if self.rng.random() < 0.35:
            bx, by = self.rng.randint(0, W // 5), self.rng.randint(0, H - 50)
            bw, bh = self.rng.randint(W // 10, W // 4), self.rng.randint(30, H // 3)
            color = (255, 255, 255) if self.rng.random() < 0.7 else (210, 210, 210)
            cv2.rectangle(img, (bx, by), (bx + bw, by + bh), color, -1)

        # smudges / stains
        for _ in range(self.rng.randint(0, 2)):
            cx, cy = self.rng.randint(0, W), self.rng.randint(0, H)
            rx, ry = self.rng.randint(10, 40), self.rng.randint(5, 20)
            overlay = img.copy()
            cv2.ellipse(overlay, (cx, cy), (rx, ry), angle=self.rng.randint(0, 180),
                        startAngle=0, endAngle=360, color=(self.rng.randint(120, 200),) * 3, thickness=-1)
            a = self.rng.uniform(0.2, 0.5)
            img = cv2.addWeighted(overlay, a, img, 1 - a, 0)

        # page curvature (forced for curved_* modes, optional otherwise)
        if force_curvature or self.rng.random() < 0.55:
            if force_curvature or self.rng.random() < 0.50:
                bend_strength = self.rng.uniform(5, 15)
                map_x = np.zeros((H, W), np.float32); map_y = np.zeros((H, W), np.float32)
                for y in range(H):
                    shift = int(np.sin(2 * np.pi * y / H) * bend_strength)
                    map_x[y, :] = np.arange(W) + shift; map_y[y, :] = y
                img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # lighting band
        if self.rng.random() < 0.5:
            mask = np.ones((H, W), np.float32)
            x0, y0 = self.rng.randint(0, W), self.rng.randint(0, H)
            x1, y1 = self.rng.randint(0, W), self.rng.randint(0, H)
            cv2.line(mask, (x0, y0), (x1, y1), (0,), thickness=self.rng.randint(80, 150))
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=W // 6)
            mask = cv2.normalize(mask, None, 0.6, 1.08, cv2.NORM_MINMAX)
            img = np.clip(img.astype(np.float32) * mask[:, :, None], 0, 255).astype(np.uint8)

        # blur / JPEG
        if self.rng.random() < 0.3:
            img = cv2.GaussianBlur(img, (self._choice([3, 5]),) * 2, 0)
        if self.rng.random() < 0.25:
            _, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), self.rng.randint(40, 90)])
            img = cv2.imdecode(enc, 1)
        return img

    def _pick_sigma(self) -> float:
        if self.sigma_min is not None and self.sigma_max is not None:
            return float(np.random.uniform(self.sigma_min, self.sigma_max))
        return self.sigma

    def _make_heatmaps(self, corners: List[Tuple[int, int]]) -> np.ndarray:
        H = W = self.img_size
        heatmaps = np.zeros((4, H, W), np.float32)
        xs, ys = np.meshgrid(np.arange(W), np.arange(H))
        sigma = self._pick_sigma()
        denom = 2.0 * (sigma ** 2)
        for i, (cx, cy) in enumerate(corners):
            d2 = (xs - cx) ** 2 + (ys - cy) ** 2
            heatmaps[i] = np.exp(-d2 / denom)
        return heatmaps

# ──────────────────────────────────────────────────────────────────────────────
# 3) CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_mode_ratios(arg: Optional[str]) -> Optional[Dict[str, float]]:
    if not arg:
        return None
    out: Dict[str, float] = {}
    for part in arg.split(","):
        part = part.strip()
        if not part: continue
        if "=" not in part:
            raise ValueError(f"Bad --mode-ratios token: '{part}'. Expected name=ratio.")
        k, v = part.split("=", 1)
        k = k.strip()
        v = float(v.strip())
        out[k] = v
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--img-size", type=int, default=128)

    # Legacy knobs (kept for backward compatibility)
    ap.add_argument("--p-warp", type=float, default=0.50,
                    help="[LEGACY] fraction of warped/curved grids if --mode-ratios is NOT provided")
    ap.add_argument("--p-attached-header", type=float, default=0.10,
                    help="probability of a band touching the top edge (orthogonal attribute)")
    ap.add_argument("--strict-axis-straight", action="store_true",
                    help="When generating 'straight' (legacy), use axis-aligned rectangle with margin.")
    ap.add_argument("--strict-margin-frac", type=float, default=0.08,
                    help="Margin fraction for strict straight grids (inset, default 0.08).")

    # Heatmap sigma
    ap.add_argument("--sigma", type=float, default=2.2,
                    help="Gaussian sigma (px) for corner heatmaps (default 2.2).")
    ap.add_argument("--sigma-min", type=float, default=None,
                    help="Lower bound for randomized sigma (px). Used only if both min & max are set.")
    ap.add_argument("--sigma-max", type=float, default=None,
                    help="Upper bound for randomized sigma (px). Used only if both min & max are set.")

    # NEW: explicit mode mix
    ap.add_argument("--mode-ratios", type=str, default=None,
                    help=("Comma-separated list of name=ratio. "
                          "Supported: straight_strict,straight_loose,warped_inside,warped_partial,curved_inside,curved_partial "
                          "Example: 'straight_strict=0.25,straight_loose=0.25,warped_inside=0.30,warped_partial=0.10,curved_inside=0.10'"))

    ap.add_argument("--seed", type=int, default=None)

    args = ap.parse_args()

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "heatmaps").mkdir(parents=True, exist_ok=True)

    ratios = parse_mode_ratios(args.mode_ratios)
    gen = SudokuCornerGenerator(
        img_size=args.img_size,
        seed=args.seed,
        p_warp=args.p_warp,
        p_attached_header=args.p_attached_header,
        strict_axis_straight=args.strict_axis_straight,
        strict_margin_frac=args.strict_margin_frac,
        sigma=args.sigma,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        mode_ratios=ratios,
    )

    # If ratios provided, print normalized for sanity
    if ratios:
        norm = _normalize_ratios({k: ratios.get(k, 0.0) for k in SudokuCornerGenerator.ALL_MODES if k in ratios})
        print("Using mode ratios (normalized):")
        for k, v in norm.items():
            print(f"  - {k:15s}: {v:.3f}")

    labels = []
    for i in range(args.n):
        img, corners, heatmaps, mode = gen.sample()
        fname = f"sample_{i:05d}.png"
        cv2.imwrite(str(out / "images" / fname), img)
        safe_save_npy(out / "heatmaps" / fname.replace(".png", ".npy"), heatmaps)
        labels.append({"file": fname, "corners": corners, "mode": mode})
        if i % 100 == 0:
            print(f"[{i}/{args.n}] generated")

    with open(out / "labels.jsonl", "w", encoding="utf-8") as f:
        for row in labels:
            f.write(json.dumps(row) + "\n")
    print(f"✅ Done. {args.n} samples → {out}")

if __name__ == "__main__":
    main()