"""
train_corners.py — end‑to‑end training script for Sudoku corner heatmaps
=======================================================================

WHY
----
Detecting the four outer corners of a Sudoku grid reliably is the gateway to
everything that follows (perspective rectification, cell parsing, digit OCR).
Real‑world images are messy: borders can be faint, inner lines can be darker
than the frame, pages warp, photographs are shadowed, and sometimes there is
no grid at all. A small model must learn to lock on the *true* corners while
resisting many tempting distractors.

WHAT
-----
This module trains (and evaluates) a tiny heatmap network that outputs four
probability maps — one per corner: TL, TR, BR, BL. The training pipeline
supports a rich synthetic generator ("Sudoku Journey") with a curriculum
(D1→D8), optional real data, and a set of targeted loss terms that shape the
heatmaps:

- BCE / Focal‑BCE against Gaussian targets

# -----------------------------------------------------------------------------
# Imports overview
# -----------------------------------------------------------------------------
# - Built‑ins: os/json/csv/time/random/math/pathlib/dataclasses/typing for I/O,
#   config, and small geometric computations.
# - NumPy / OpenCV: image synthesis, drawing, warps, and numeric helpers.
# - PyTorch: model, losses, data loaders, training loop machinery.
# - TQDM: progress bars to keep long runs honest.
# 
# The order is deliberate: light stdlib first, then heavy deps, and finally
# evaluation / convenience tools.
# -----------------------------------------------------------------------------

- Pixel‑wise cross‑entropy to soften supervision
- Coordinate loss measured via a temperatured soft‑argmax
- *Inset*/*Outset* strip suppressions that reduce false peaks along the first
  cell inside/outside each corner
- A negative‑only peak suppression to keep "no‑grid" samples quiet
- (Optional) border‑ring suppression to avoid "frame snapping"

HOW (map to code)
-----------------
The file is divided into clear sections you can skim top‑to‑bottom:

1. **Utilities** — small geometry helpers, heatmap tools, PCK/softargmax.
2. **Synthetic generator (D1..D8)** — the curriculum “story engine”.
   Start at `synth_sudoku_journey` which orchestrates rendering, then jump
   to helpers such as `_draw_grid_lines`, `_photographic_layer`,
   `_add_soft_shadow`, and the projective/curvature warps.
3. **Iterable synthetic stream** — a tiny dataset that yields endless samples.
4. **Model** — `TinyCornerNet`: encoder (+ CoordConv) with optional U‑Net head.
5. **Visualization** — convenience panels for quick qualitative checks.
6. **Losses & evaluation** — BCE/CE, PCK, masked metrics, and the suppression
   helpers: `inset_suppression_max`, `outset_suppression_max`,
   `border_ring_suppression`.
7. **Training** — the full loop (`train`) with LR/plateau logic and CSV logs.
8. **CLI** — a well‑documented interface; see the big block at the bottom for
   argument groups, advice, and ready‑to‑run PowerShell examples.

Reading tips
------------
- Prefer **docstrings** at the top of each function/class for the “why/what/how”;
  inline comments inside mark the main blocks.
- The generator’s comments describe *how each difficulty level escalates realism*.
- The training docstring groups arguments by theme; jump straight there when
  tuning runs.
- Examples near the CLI mirror our best performing commands and typical knobs.

"""
# train_corners.py
# Corner heatmap model training for Sudoku Journey (D1..D8).
# - Synthetic generator that follows the 8-stage curriculum
# - Optional negatives and occlusions
# - Mask-safe metrics/loss for invalid corners
# - CoordConv + tiny U-Net-ish decoder
# - CSV logging + optional viz panels

import os, json, math, csv, random, time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, IterableDataset
from tqdm import tqdm




# ---- Real dataset for heatmaps (top-level, picklable) ----
class CornerHeatmapDataset(torch.utils.data.Dataset):
    """
    Expects a folder structure:
      <root>/
        images/
          <filename>        (RGB/BGR or grayscale)
        labels.jsonl         (one JSON per line)
    Each JSON has a 'file_name' (or 'file'/'image'/'path') and
    'corners': [[x,y], [x,y], [x,y], [x,y]] in absolute pixels or [0..1] normalized.
    Returns:
      inp:  [1,H,W] float32 in [0,1]
      hms:  [4,H,W] float32 heatmaps rendered with given sigma
      xy:   [4,2]   float32 target corner coordinates (ordered TL,TR,BR,BL) in resized space
    """
    def __init__(self, root: str, img_size: int = 128, sigma: float = 3.0, augment: bool = False):
        self.root = Path(root)
        self.img_dir = self.root / "images"
        self.labels_path = self.root / "labels.jsonl"
        self.img_size = int(img_size)
        self.sigma = float(sigma)
        self.augment = augment

        assert self.img_dir.is_dir(), f"Missing images dir: {self.img_dir}"
        assert self.labels_path.is_file(), f"Missing labels: {self.labels_path}"

        self.samples = []
        with open(self.labels_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        fname = s.get("file_name", s.get("file", s.get("image", s.get("path"))))
        if fname is None:
            raise KeyError("No image filename key found in labels.jsonl")

        fp = self.img_dir / fname
        img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(fp)

        H0, W0 = img.shape[:2]
        corners = np.array(s["corners"], dtype=np.float32)  # (4,2)

        # support normalized labels
        if float(np.max(corners)) <= 1.5:
            corners = corners * np.array([[W0, H0]], dtype=np.float32)

        # (very) light augmentation: small in-plane rotation
        if self.augment and random.random() < 0.3:
            ang = random.uniform(-2, 2)
            M = cv2.getRotationMatrix2D((W0 / 2, H0 / 2), ang, 1.0)
            img = cv2.warpAffine(
                img, M, (W0, H0),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )
            ones = np.ones((4, 1), np.float32)
            corners = (M @ np.concatenate([corners, ones], axis=1).T).T

        # ensure canonical order TL,TR,BR,BL (matches the rest of your pipeline)
        corners = order_corners_tl_tr_br_bl(corners)

        # resize + scale corners
        img_rs = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        scale_x = self.img_size / W0
        scale_y = self.img_size / H0
        corners_rs = corners.copy()
        corners_rs[:, 0] *= scale_x
        corners_rs[:, 1] *= scale_y

        # grayscale input in [0,1]
        g = to_grayscale_01(img_rs)                 # [H,W], float32
        inp = np.expand_dims(g, 0).astype(np.float32)  # [1,H,W]

        # target heatmaps
        hms = render_heatmaps(self.img_size, self.img_size, corners_rs, self.sigma)  # [4,H,W]

        return (
            torch.from_numpy(inp),
            torch.from_numpy(hms),
            torch.from_numpy(corners_rs),
        )


# =========================================================
# Utilities
# =========================================================



# ---- helpers ---------------------------------------------------------------


def collate_corners_batch(batch):
    """
    Collate function that accepts either:
      - (inp, hm, xy) tuples   OR
      - (inp, hm, xy, meta) tuples (where meta has header/footer rects lists)

    It returns:
      (imgs[B,1,H,W], hms[B,4,H,W], xys[B,4,2])      -- when no meta present in batch
    or
      (imgs, hms, xys, batch_meta_dict)              -- when any meta present

    batch_meta_dict contains:
      "header_text_rects_list": List[List[List[int]]] length = B
      "footer_text_rects_list": List[List[List[int]]] length = B
    """
    has_meta = isinstance(batch[0], (list, tuple)) and len(batch[0]) == 4

    if not has_meta:
        imgs = torch.stack([torch.as_tensor(b[0]) for b in batch], dim=0)
        hms  = torch.stack([torch.as_tensor(b[1]) for b in batch], dim=0)
        xys  = torch.stack([torch.as_tensor(b[2]) for b in batch], dim=0)
        return imgs, hms, xys

    imgs, hms, xys, metas = zip(*batch)  # lists of length B
    imgs = torch.stack([torch.as_tensor(t) for t in imgs], dim=0)
    hms  = torch.stack([torch.as_tensor(t) for t in hms],  dim=0)
    xys  = torch.stack([torch.as_tensor(t) for t in xys],  dim=0)

    header_list = []
    footer_list = []
    for m in metas:
        header_list.append(list(m.get("header_text_rects", [])))
        footer_list.append(list(m.get("footer_text_rects", [])))

    batch_meta = {
        "header_text_rects_list": header_list,
        "footer_text_rects_list": footer_list,
    }
    return imgs, hms, xys, batch_meta


def _random_phrase(rng: random.Random) -> str:
    # short, high-contrast words/sentences so they can span the grid width when repeated
    bank = [
        "Daily Sudoku Challenge", "Keep your brain sharp", "Puzzle time",
        "Classic edition", "Logic over luck", "Solve and relax",
        "Focus then solve", "Practice makes progress", "Number puzzle",
        "Train attention", "Steady and calm", "Concentration mode",
        "Sharpen your skills", "Mindful break", "Sudoku strategies",
    ]
    return rng.choice(bank)

def _build_mask_from_rects(rects, H, W, device):
    m = torch.zeros((H, W), device=device, dtype=torch.float32)
    for x0,y0,x1,y1 in rects:
        x0 = max(0, min(W-1, int(x0))); x1 = max(0, min(W-1, int(x1)))
        y0 = max(0, min(H-1, int(y0))); y1 = max(0, min(H-1, int(y1)))
        if x1 > x0 and y1 > y0:
            m[y0:y1+1, x0:x1+1] = 1.0
    return m

def _disc_mask(H, W, cx, cy, r, device):
    yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    return (((xx - cx)**2 + (yy - cy)**2) <= (r*r)).float()



def gutter_suppression_loss(
    pred_probs: torch.Tensor,   # [B,4,H,W]
    gt_xy: torch.Tensor,        # [B,4,2]
    thickness: int = 6          # strip height above top and below bottom borders
):
    """
    Penalize probability mass for top-corner channels in a band above the top
    border, and bottom-corner channels in a band below the bottom border.
    The band is anchored from the y of the GT corners for each sample.
    """
    device = pred_probs.device
    B, C, H, W = pred_probs.shape
    assert C == 4

    yy = torch.arange(H, device=device).view(H, 1).expand(H, W)

    loss = pred_probs.new_tensor(0.0)
    for i in range(B):
        y_top = torch.clamp(gt_xy[i, 0:2, 1].mean().round().long(), 0, H - 1)   # avg TL/TR y
        y_bot = torch.clamp(gt_xy[i, 2:4, 1].mean().round().long(), 0, H - 1)   # avg BL/BR y

        # top band for TL/TR
        y0t = int(max(0, y_top.item() - thickness))
        y1t = int(min(H - 1, y_top.item() - 1))
        if y1t >= y0t:
            loss = loss + pred_probs[i, 0, y0t:y1t+1, :].amax()  # TL
            loss = loss + pred_probs[i, 1, y0t:y1t+1, :].amax()  # TR

        # bottom band for BL/BR
        y0b = int(min(H - 1, y_bot.item() + 1))
        y1b = int(min(H - 1, y_bot.item() + thickness))
        if y1b >= y0b:
            loss = loss + pred_probs[i, 2, y0b:y1b+1, :].amax()  # BL
            loss = loss + pred_probs[i, 3, y0b:y1b+1, :].amax()  # BR

    return loss / max(1, B)



def text_suppression_max(
    pred_probs: torch.Tensor,                # [Bp, 4, H, W] probabilities in [0,1]
    gt_xy: torch.Tensor,                     # [Bp, 4, 2] corner coords (x,y)
    header_rects_list: list,                 # list (len ≥ Bp allowed; missing entries OK)
    footer_rects_list: list,                 # list (len ≥ Bp allowed; missing entries OK)
    exempt_radius: int = 6,
) -> torch.Tensor:
    """
    Penalize high corner probability within header/footer text rectangles.
    For each positive sample i, and for each provided rectangle r in header/footer,
    we take max(prob) inside r (per channel), after zeroing a small disc around the
    true corner to avoid conflicting with the supervised peak. We average over rects,
    channels, and batch.

    The function is robust to:
      - header_rects_list/footer_rects_list being empty or None
      - lists shorter than Bp (missing entries treated as [])
      - non-list entries (treated as [])
    """
    if pred_probs.numel() == 0:
        return torch.tensor(0.0, device=gt_xy.device, dtype=gt_xy.dtype)

    Bp, C, H, W = pred_probs.shape
    device = pred_probs.device
    dtype = pred_probs.dtype

    # Helpers to safely fetch per-sample rects
    def _safe_rects(src, i):
        if not isinstance(src, (list, tuple)) or len(src) == 0:
            return []
        if i >= len(src):
            return []
        v = src[i]
        return v if isinstance(v, (list, tuple)) else []

    # Helper to zero-out a small disc around a (x,y) integer center
    def _zero_disc_(m: torch.Tensor, x: int, y: int, r: int):
        if r <= 0: 
            return
        # clamp disc bbox
        x0 = max(0, x - r); x1 = min(W - 1, x + r)
        y0 = max(0, y - r); y1 = min(H - 1, y + r)
        if x0 > x1 or y0 > y1:
            return
        ys = torch.arange(y0, y1 + 1, device=m.device, dtype=torch.int64).view(-1, 1)
        xs = torch.arange(x0, x1 + 1, device=m.device, dtype=torch.int64).view(1, -1)
        # boolean disc mask
        disc = (xs - x)**2 + (ys - y)**2 <= (r * r)
        # m is [4, H, W]; zero on all channels
        m[:, y0:y1 + 1, x0:x1 + 1] = torch.where(disc, torch.zeros(1, dtype=m.dtype, device=m.device), m[:, y0:y1 + 1, x0:x1 + 1])

    # Compute the loss
    total = torch.zeros((), device=device, dtype=dtype)
    count = 0

    for i in range(Bp):
        # Work on a copy we can zero discs on
        m = pred_probs[i].clone()  # [4, H, W]

        # Exempt discs at true corner coords (avoid penalizing the true peaks)
        for k in range(4):
            # Safe int conversions (work even if gt_xy is float/double)
            xk = int(round(float(gt_xy[i, k, 0])))
            yk = int(round(float(gt_xy[i, k, 1])))
            _zero_disc_(m, xk, yk, exempt_radius)

        # Now compute maxima inside any provided rects
        # Header rects
        hdr_rects = _safe_rects(header_rects_list, i)
        for r in hdr_rects:
            if not isinstance(r, (list, tuple)) or len(r) != 4:
                continue
            x0, y0, x1, y1 = r
            x0 = max(0, min(W - 1, int(x0))); x1 = max(0, min(W - 1, int(x1)))
            y0 = max(0, min(H - 1, int(y0))); y1 = max(0, min(H - 1, int(y1)))
            if x1 < x0 or y1 < y0:
                continue
            # max prob per channel in the rect
            mx = m[:, y0:y1 + 1, x0:x1 + 1].amax(dim=-1).amax(dim=-1)  # [4]
            total = total + mx.mean()
            count += 1

        # Footer rects
        ftr_rects = _safe_rects(footer_rects_list, i)
        for r in ftr_rects:
            if not isinstance(r, (list, tuple)) or len(r) != 4:
                continue
            x0, y0, x1, y1 = r
            x0 = max(0, min(W - 1, int(x0))); x1 = max(0, min(W - 1, int(x1)))
            y0 = max(0, min(H - 1, int(y0))); y1 = max(0, min(H - 1, int(y1)))
            if x1 < x0 or y1 < y0:
                continue
            mx = m[:, y0:y1 + 1, x0:x1 + 1].amax(dim=-1).amax(dim=-1)  # [4]
            total = total + mx.mean()
            count += 1

    if count == 0:
        return torch.tensor(0.0, device=device, dtype=dtype)
    return total / float(count)






def _add_decoy_neighbor_grid(
    img: np.ndarray,
    rng: random.Random,
    main_rect: tuple[int, int, int, int],  # (x0, y0, w, h) of the target grid
    cells_h: int,
    cells_w: int,
    *,
    side: str,  # "top" | "bottom" | "left" | "right"
    gap_factor_range: tuple[float, float] = (0.5, 1.5),
    t_border: int = 2,
    t_div: list[int] | None = None,
    t_std: list[float] | None = None,
    link_mode: bool = True,
    broken_border: float = 0.0,
    broken_std: float = 0.0,
    distortion_level: float = 0.0,
    distort_style: str = "S",
    col_border: int | None = None,
    col_div: int | None = None,
    col_std: int | None = None,
) -> None:
    """
    Draw a duplicate Sudoku grid adjacent to the main one, offset by a gap that is
    0.5–1.5× the cell size along the corresponding axis. The grid may be partially
    outside the canvas; OpenCV will clip appropriately.
    """
    H, W = img.shape[:2]
    x0, y0, w, h = main_rect
    x1, y1 = x0 + w, y0 + h
    cw = max(1, cells_w)
    ch = max(1, cells_h)

    cell_w = max(1.0, w / float(cw))
    cell_h = max(1.0, h / float(ch))

    # choose gap in pixels based on the requested side
    if side in ("left", "right"):
        gap = rng.uniform(*gap_factor_range) * cell_w
    else:
        gap = rng.uniform(*gap_factor_range) * cell_h

    # top-left of the decoy
    if side == "top":
        dx, dy = 0.0, -(gap + h)
    elif side == "bottom":
        dx, dy = 0.0, (gap + 0.0)
    elif side == "left":
        dx, dy = -(gap + w), 0.0
    else:  # "right"
        dx, dy = (gap + 0.0), 0.0

    dec_x0 = int(round(x0 + dx))
    dec_y0 = int(round(y0 + dy))
    dec_rect = (dec_x0, dec_y0, w, h)

    # four corners of the decoy rectangle (TL, TR, BR, BL)
    dec_corners = np.array([
        [dec_x0,         dec_y0        ],
        [dec_x0 + w - 1, dec_y0        ],
        [dec_x0 + w - 1, dec_y0 + h - 1],
        [dec_x0,         dec_y0 + h - 1],
    ], dtype=np.float32)

    # draw the same grid styling as the main one (kept lightweight)
    _draw_grid_lines(
        img, rng, dec_corners, ch, cw,
        t_border=t_border,
        t_div=t_div if t_div is not None else [],
        t_std=t_std if t_std is not None else [],
        link_mode=link_mode,
        broken_border=broken_border,
        broken_std=broken_std,
        distortion_level=distortion_level,
        distort_style=distort_style,
        div_h_count=ch if (t_div is not None and len(t_div) > 0) else ch,
        div_v_count=cw if (t_div is not None and len(t_div) > 0) else cw,
        col_border=col_border, col_div=col_div, col_std=col_std,
    )

    # (Optional) very light digits in the decoy, if you want extra realism.
    # You can comment this block out if you prefer blank decoys.
    if rng.random() < 0.35:
        #_digits_on_grid(img, rng, dec_rect, ch, cw, fill_mode=rng.choice(["empty","normal","dense"]))
        _digits_on_grid(img, rng, dec_rect, ch, cw, mode=rng.choice(["empty","normal","dense"]))
        # If your _digits_on_grid takes it positionally, this also works:
        # _digits_on_grid(img, rng, dec_rect, ch, cw, rng.choice(["empty","normal","dense"]))



def _safe_unit(v, eps=1e-6):
    # v: [..., 2]
    """
    Return a numerically safe unit vector per row.

    Inputs
      v : tensor [..., 2]

    Notes
      We add epsilon to the denominator to avoid NaNs when norms are tiny.
    """
    return v / (torch.norm(v, dim=-1, keepdim=True) + eps)

def _arm_dirs_inward(gt_xy):
    """
    gt_xy: [B,4,2] (TL,TR,BR,BL)
    returns dirs_in: [B,4,2,2]  (per corner: two inward arm unit vectors)
    """
    tl, tr, br, bl = gt_xy[:, 0], gt_xy[:, 1], gt_xy[:, 2], gt_xy[:, 3]
    v_top, v_right, v_bottom, v_left = tr - tl, br - tr, bl - br, tl - bl

    d_tl = torch.stack([ _safe_unit(v_top),    _safe_unit(v_left)   ], dim=1)  # right, down
    d_tr = torch.stack([ _safe_unit(-v_top),   _safe_unit(v_right)  ], dim=1)  # left,  down
    d_br = torch.stack([ _safe_unit(-v_right), _safe_unit(-v_bottom)], dim=1)  # up,    left
    d_bl = torch.stack([ _safe_unit(v_left),   _safe_unit(-v_bottom)], dim=1)  # right, up
    return torch.stack([d_tl, d_tr, d_br, d_bl], dim=1)  # [B,4,2,2]

def _arm_length(gt_xy, frac, eps=1e-6):
    # use min(grid_w, grid_h) for scale
    """
    Scalar arm length per image ∝ min(grid_w, grid_h).

    Why
      When sampling along “first‑cell” corridors we want a distance that
      adapts to grid size yet is comparable across examples.

    Params
      frac : multiply min(grid_w, grid_h) by this fraction.
    """
    tl, tr, br, bl = gt_xy[:, 0], gt_xy[:, 1], gt_xy[:, 2], gt_xy[:, 3]
    grid_w = torch.norm(tr - tl, dim=-1) + eps
    grid_h = torch.norm(tl - bl, dim=-1) + eps
    return frac * torch.minimum(grid_w, grid_h)  # [B]

def _sample_strip_max(pred_probs, centers, dirs, arm_len, n_steps, radius):
    """
    pred_probs: [B,4,H,W]  (probabilities)
    centers:    [B,4,2]    (corner coords)
    dirs:       [B,4,2,2]  (two unit vectors per corner)
    arm_len:    [B]        (scalar per image)
    returns:    [B,4]      max response found along both arms within radius windows
    """
    B, C, H, W = pred_probs.shape
    device = pred_probs.device
    dtype  = pred_probs.dtype

    # pre-maxpool so a local window [2r+1]^2 becomes one read
    pooled = F.max_pool2d(pred_probs, kernel_size=2*radius+1, stride=1, padding=radius)  # [B,4,H,W]

    # steps along each arm (scaled by per-image arm_len)
    t = torch.arange(1, n_steps + 1, device=device, dtype=dtype) / float(n_steps)  # [S]
    t = (arm_len[:, None] * t[None, :])  # [B,S]

    # centers -> [B,4,1,1,2]; dirs -> [B,4,2,1,2]; t -> [B,1,1,S,1]
    centers = centers[:, :, None, None, :]                    # [B,4,1,1,2]
    dirs    = dirs[:, :, :, None, :]                          # [B,4,2,1,2]
    steps   = t[:, None, None, :, None]                       # [B,1,1,S,1]

    pos = centers + dirs * steps                              # [B,4,2,S,2]
    xs = pos[..., 0].round().long().clamp(0, W - 1)           # [B,4,2,S]
    ys = pos[..., 1].round().long().clamp(0, H - 1)           # [B,4,2,S]
    lin = ys * W + xs                                         # [B,4,2,S]

    # gather from pooled maps (flatten H*W axis). We loop k=0..3 only (tiny).
    flat = pooled.view(B, C, -1)                              # [B,4,HW]
    vals = []
    for k in range(4):  # small, harmless loop
        v = torch.gather(flat[:, k], -1, lin[:, k].view(B, -1))          # [B,2*S]
        vals.append(v.view(B, 2, -1))                                     # [B,2,S]
    vals = torch.stack(vals, dim=1)                                       # [B,4,2,S]
    return vals.amax(dim=(-1, -2))                                        # [B,4]
# ---------------------------------------------------------------------------


def inset_suppression_max(
    pred_probs: torch.Tensor,
    gt_xy: torch.Tensor,
    valid_mask: torch.Tensor,
    frac: float = 0.12,
    radius: int = 3,
    n_steps: int = 6,
) -> torch.Tensor:
    """
    Vectorized: penalize peaks along the first-cell-IN corridor from each corner.
    Returns a scalar (mean over valid corners in positives).
    """
    B, C, H, W = pred_probs.shape
    dirs_in    = _arm_dirs_inward(gt_xy)                  # [B,4,2,2]
    arm_len    = _arm_length(gt_xy, frac)                 # [B]
    strip_max  = _sample_strip_max(pred_probs, gt_xy, dirs_in, arm_len, n_steps, radius)  # [B,4]

    if valid_mask is not None:
        m = valid_mask.bool()
        if not m.any():
            return pred_probs.new_tensor(0.0)
        return strip_max[m].mean()
    return strip_max.mean()


def outset_suppression_max(
    pred_probs: torch.Tensor,
    gt_xy: torch.Tensor,
    valid_mask: torch.Tensor,
    frac: float = 0.12,
    radius: int = 3,
    n_steps: int = 6,
) -> torch.Tensor:
    """
    Vectorized: penalize peaks just OUTSIDE the grid along outward arms.
    (Same directions as inset, but negated.)
    """
    B, C, H, W = pred_probs.shape
    dirs_out   = -_arm_dirs_inward(gt_xy)                 # [B,4,2,2]
    arm_len    = _arm_length(gt_xy, frac)                 # [B]
    strip_max  = _sample_strip_max(pred_probs, gt_xy, dirs_out, arm_len, n_steps, radius)  # [B,4]

    if valid_mask is not None:
        m = valid_mask.bool()
        if not m.any():
            return pred_probs.new_tensor(0.0)
        return strip_max[m].mean()
    return strip_max.mean()


def border_ring_suppression(
    pred_probs: torch.Tensor,
    gt_xy: torch.Tensor,
    valid_mask: torch.Tensor,
    ring_thickness: int = 8,
    exempt_radius: int = 5,
) -> torch.Tensor:
    """
    Cheap global guard: penalize channel-wise max along the image border ring,
    while exempting small discs around the true corners.
    """
    B, C, H, W = pred_probs.shape
    device = pred_probs.device
    dtype  = pred_probs.dtype

    # ring mask [H,W]
    ring = torch.zeros((H, W), device=device, dtype=dtype)
    ring[:ring_thickness, :] = 1
    ring[-ring_thickness:, :] = 1
    ring[:, :ring_thickness] = 1
    ring[:, -ring_thickness:] = 1
    ring = ring[None, None].expand(B, C, H, W)  # [B,4,H,W]

    # exempt discs around corners → [B,1,H,W]
    xs = torch.arange(W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)
    ys = torch.arange(H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
    r2 = float(exempt_radius ** 2)

    exempt = torch.zeros((B, 1, H, W), device=device, dtype=torch.bool)
    for k in range(4):
        xk = gt_xy[:, k, 0].view(B, 1, 1, 1)
        yk = gt_xy[:, k, 1].view(B, 1, 1, 1)
        d2 = (xs - xk)**2 + (ys - yk)**2
        exempt |= (d2 <= r2)

    ring = ring * (~exempt).float()                       # [B,4,H,W]
    masked = pred_probs * ring
    per_chan_max = masked.amax(dim=(-1, -2))              # [B,4]

    if valid_mask is not None:
        m = valid_mask.bool()
        if not m.any():
            return pred_probs.new_tensor(0.0)
        return per_chan_max[m].mean()
    return per_chan_max.mean()




def _pick_font(rng: random.Random) -> int:
    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_COMPLEX_SMALL,
        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_SCRIPT_COMPLEX
    ]
    return rng.choice(fonts)

def _draw_text_block_aligned(
    img: np.ndarray,
    rng: random.Random,
    grid_rect: tuple[int,int,int,int],
    cells_h: int,
    where: str,                     # 'header' or 'footer'
    n_lines: int,
    height_mult_range: tuple[float,float] = (0.8, 1.5),  # ← 0.8–1.5× digit height
    gap_px_range: tuple[int,int] = (2, 5),               # ← 2–5 px gap to grid
    align: str = "left",
    jitter_px: int = 0
) -> list[list[int]]:
    """
    Draw left-aligned text just above ('header') or just below ('footer') the grid.
    The last header line sits 2–5 px above the top border; the first footer line
    sits 2–5 px below the bottom border. Each line attempts to cover ~full grid
    width. Returns a list of rectangles [x0,y0,x1,y1] for every rendered line.
    """
    H, W = img.shape[:2]
    x, y, w, h = grid_rect
    ch = max(1.0, float(h) / max(1, cells_h))           # cell height (px)

    # estimate digit visual height (~65% of a cell)
    digit_h = 0.65 * ch
    target_h = rng.uniform(*height_mult_range) * digit_h
    scale = float(np.clip(target_h / 18.0, 0.3, 4.0))
    font  = _pick_font(rng)
    thick = rng.randint(1, 2)

    vspace = max(3, int(round(0.15 * target_h)))        # ~15% line spacing

    def make_line():
        # build up a string until it spans the grid width at current scale
        parts = []
        while True:
            parts.append(_random_phrase(rng))
            txt = " ".join(parts)
            (tw, th), base = cv2.getTextSize(txt, font, scale, thick)
            if tw >= (w - 6) or len(parts) > 6:
                while tw > (w - 4) and len(txt) > 3:
                    txt = txt[:-1]
                    (tw, th), base = cv2.getTextSize(txt, font, scale, thick)
                return txt, tw, th, base

    gap   = rng.randint(*gap_px_range)
    lines = [make_line() for _ in range(n_lines)]

    # Make text typically lighter than dark borders (which are ~0–25).
    color = int(rng.uniform(80, 180))

    def x_anchor_for_width(tw: int) -> int:
        base_x = x
        if align == "right":
            base_x = x + (w - tw)
        elif align == "center":
            base_x = x + (w - tw) // 2
        if jitter_px:
            base_x += rng.randint(-jitter_px, jitter_px)
        return int(np.clip(base_x, 0, W - 1))

    rects: list[list[int]] = []

    if where == "header":
        y_base = y - gap
        for txt, tw, th, base in reversed(lines):
            tx = x_anchor_for_width(tw)
            ty = y_base
            # clamp vertically if needed
            top = ty - th
            if top < 0:
                ty -= top
            cv2.putText(img, txt, (tx, ty), font, scale, (color,), thick, cv2.LINE_AA)
            # record rectangle
            x0, y0 = tx, max(0, ty - th)
            x1, y1 = min(W - 1, tx + tw), min(H - 1, ty + base)
            rects.append([x0, y0, x1, y1])
            y_base -= (th + vspace)
        rects.reverse()  # keep top→bottom order
    else:  # 'footer'
        y_base = y + h + gap
        for txt, tw, th, base in lines:
            tx = x_anchor_for_width(tw)
            ty = y_base
            bottom = ty + base
            if bottom >= H:
                ty = min(ty, H - base - 1)
            cv2.putText(img, txt, (tx, ty), font, scale, (color,), thick, cv2.LINE_AA)
            x0, y0 = tx, max(0, ty - th)
            x1, y1 = min(W - 1, tx + tw), min(H - 1, ty + base)
            rects.append([x0, y0, x1, y1])
            y_base += (th + vspace)

    # Optional: lightly blur the whole text block region (simulates softer print)
    if rects and rng.random() < 0.30:
        x0u = max(0, min(r[0] for r in rects))
        y0u = max(0, min(r[1] for r in rects))
        x1u = min(W - 1, max(r[2] for r in rects))
        y1u = min(H - 1, max(r[3] for r in rects))
        if x1u > x0u and y1u > y0u:
            k = rng.choice([3, 5])
            roi = img[y0u:y1u+1, x0u:x1u+1]
            img[y0u:y1u+1, x0u:x1u+1] = cv2.GaussianBlur(roi, (k, k), 0)

    return rects





def _shadow_blob_schedule(d: int, rng: random.Random):
    """
    Difficulty ramp for intrusive blob:
      D6 → 30% chance, medium size/darkness
      D7 → 45% chance, larger & darker possible
      D8 → 60% chance, can get very large & dark
    Returns (prob, area_frac_range, max_attn_range).
    """
    if d < 6:
        return 0.0, (0.0, 0.0), (0.0, 0.0)

    # 6→0.0, 7→0.5, 8→1.0
    t = min(1.0, max(0.0, (d - 6) / 2.0))

    # presence probability: 0.30 → 0.60
    prob = 0.30 + 0.30 * t

    # size range (fraction of image area):
    # start ~10–35% at D6 → up to ~18–80% by D8
    area_lo = 0.10 + (0.18 - 0.10) * t
    area_hi = 0.35 + (0.80 - 0.35) * t

    # rare “huge” blob tail (only at high difficulty)
    if rng.random() < 0.10 * t:
        area_hi = max(area_hi, 0.85)

    # darkness (attenuation at the blob center):
    # 0.25–0.45 at D6 → 0.35–0.75 at D8
    dark_lo = 0.25 + (0.35 - 0.25) * t
    dark_hi = 0.45 + (0.75 - 0.45) * t

    return prob, (area_lo, area_hi), (dark_lo, dark_hi)

def _paper_tone(H: int, W: int, rng: random.Random) -> np.uint8:
    """
    Returns a grayscale 'paper' background with optional gentle shading and texture.
    Produces off-white / light grey / newsprint-like tones.
    """
    r = rng.random()
    if r < 0.5:         # off-white
        base_val = rng.randint(220, 245)
    elif r < 0.8:       # light cream/tan-ish (grayscale proxy)
        base_val = rng.randint(195, 220)
    else:               # grey/newsprint
        base_val = rng.randint(165, 200)

    base = np.ones((H, W), np.float32) * base_val

    # large-scale paper shading (very low freq)
    if rng.random() < 0.9:
        gx = cv2.getGaussianKernel(W, rng.uniform(90, 180))
        gy = cv2.getGaussianKernel(H, rng.uniform(90, 180))
        field = (gy @ gx.T)
        field = (field - field.min()) / (field.max() - field.min() + 1e-6)
        amp = rng.uniform(-10, 10)
        base = np.clip(base + amp * (field - 0.5), 0, 255)

    # subtle paper texture / fiber speckle
    if rng.random() < 0.6:
        tex = cv2.GaussianBlur(
            np.random.randn(H, W).astype(np.float32),
            (0, 0),
            rng.uniform(2.0, 6.0)
        )
        tex = (tex - tex.min()) / (tex.max() - tex.min() + 1e-6)
        amp = rng.uniform(3.0, 12.0)
        base = np.clip(base + amp * (tex - 0.5), 0, 255)

    return base.astype(np.uint8)


def _fit_text_scale_and_bbox(
    text: str,
    font_face: int,
    thickness: int,
    target_w: int,
    target_h: int,
    target_height_frac: float,
) -> tuple[float, tuple[int, int], int]:
    """
    Compute a scale s so that the rendered text fits inside (target_w x target_h)
    and its glyph height is ~ target_height_frac of the target_h.
    Returns (scale, (text_w, text_h), baseline).
    """
    (w1, h1), base1 = cv2.getTextSize(text, font_face, 1.0, thickness)
    if w1 == 0 or h1 == 0:
        return 0.5, (0, 0), 0

    # Max scale that fits inside the inner box
    s_fit_w = target_w / max(1, w1)
    s_fit_h = target_h / max(1, h1 + base1)  # include baseline in total box
    s_fit = min(s_fit_w, s_fit_h)

    # Target a fraction of the inner height for the glyph (not counting baseline)
    s_target_h = (target_height_frac * target_h) / max(1, h1)

    # Take the smaller of the two and keep a small margin
    s = max(0.2, min(s_fit, s_target_h) * 0.95)

    (tw, th), base = cv2.getTextSize(text, font_face, s, thickness)
    return float(s), (int(tw), int(th)), int(base)




def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def set_seed(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def order_corners_tl_tr_br_bl(corners_np):
    c = np.asarray(corners_np, dtype=np.float32)
    idx_y = np.argsort(c[:, 1])
    top = c[idx_y[:2]]
    bot = c[idx_y[2:]]
    top = top[np.argsort(top[:, 0])]
    bot = bot[np.argsort(bot[:, 0])]
    tl, tr = top[0], top[1]
    bl, br = bot[0], bot[1]
    return np.stack([tl, tr, br, bl], axis=0)

def to_grayscale_01(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.ndim == 2:
        g = img_bgr
    else:
        g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return g.astype(np.float32) / 255.0

def gaussian_heatmap(H: int, W: int, cx: float, cy: float, sigma: float) -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    hm = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma ** 2))
    return hm

def render_heatmaps(H: int, W: int, corners: np.ndarray, sigma: float, valid=None) -> np.ndarray:
    """corners: [4,2], valid: [4] bool or None (if provided, channels where valid==False are zeros)."""
    hms = []
    for i, (x, y) in enumerate(corners):
        if valid is not None and not bool(valid[i]):
            hms.append(np.zeros((H, W), np.float32))
        else:
            hms.append(gaussian_heatmap(H, W, float(x), float(y), sigma))
    return np.stack(hms, axis=0).astype(np.float32)  # [4,H,W]

def _normalize_hm_layout_4d(x: torch.Tensor, name: str) -> torch.Tensor:
    if x.dim() != 4:
        raise RuntimeError(f"{name}: expected 4D, got {x.dim()}D with shape {tuple(x.shape)}")
    B, d1, d2, d3 = x.shape
    if d1 == 4:
        return x
    last3 = [d1, d2, d3]
    if 4 in last3:
        idx = last3.index(4)
        others = [i+1 for i in range(3) if i != idx]
        perm = [0, idx+1, others[0], others[1]]
        return x.permute(*perm).contiguous()
    raise RuntimeError(f"{name}: cannot find channel dim of size 4 in shape {tuple(x.shape)}")

def ensure_chw_4(t: torch.Tensor, name: str = "tensor") -> torch.Tensor:
    if t.dim() == 4:
        return _normalize_hm_layout_4d(t, name)
    elif t.dim() == 3:
        t4 = _normalize_hm_layout_4d(t.unsqueeze(0), name + "(3D->4D)")
        return t4.squeeze(0)
    else:
        raise RuntimeError(f"{name}: expected 3D/4D heatmap tensor, got {t.dim()}D with shape {tuple(t.shape)}")

def spatial_softargmax(hm: torch.Tensor, tau: float = 0.6) -> torch.Tensor:
    """hm: [B,4,H,W] probs (0..1) or logits (we'll assume probs here)"""
    B, C, H, W = hm.shape
    flat = hm.view(B, C, -1) / max(tau, 1e-6)
    prob = torch.softmax(flat, dim=-1)  # [B,4,H*W]
    xs = torch.linspace(0, W - 1, W, device=hm.device)
    ys = torch.linspace(0, H - 1, H, device=hm.device)
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    x = torch.sum(prob * X, dim=-1)
    y = torch.sum(prob * Y, dim=-1)
    return torch.stack([x, y], dim=-1)  # [B,4,2]

def valid_corner_mask(gt_xy: torch.Tensor) -> torch.Tensor:
    """
    gt_xy: [B,4,2] with (-1,-1) for invalid corners (negatives/occlusions).
    Returns [B,4] boolean mask of valid corners.
    """
    return (gt_xy[..., 0] >= 0) & (gt_xy[..., 1] >= 0)

def pck(pred_xy: torch.Tensor, gt_xy: torch.Tensor, thresh: float, valid_mask: Optional[torch.Tensor] = None) -> float:
    """
    Percentage of Correct Keypoints within 'thresh' pixels.
    pred_xy, gt_xy: [N,2] OR [B,4,2]. We internally flatten.
    valid_mask (optional): same semantic shape as [B,4] or flat [B*4] or [N].
    """
    # normalize shapes to flat
    if pred_xy.dim() == 3:
        B, K, _ = pred_xy.shape
        pred_flat = pred_xy.reshape(B * K, 2)
        gt_flat = gt_xy.reshape(B * K, 2)
    else:
        pred_flat = pred_xy.reshape(-1, 2)
        gt_flat = gt_xy.reshape(-1, 2)

    d = torch.linalg.norm(pred_flat - gt_flat, dim=-1)  # [B*K]

    if valid_mask is not None:
        if valid_mask.dim() == 2:  # [B,4]
            m = valid_mask.reshape(-1)
        else:
            m = valid_mask.reshape(-1)
        if m.numel() != d.numel():
            # try to broadcast if mask is [B,] where B*4==len(d) not true; fallback safe
            raise RuntimeError(f"pck: mask numel ({m.numel()}) != num distances ({d.numel()})")
        d = d[m]
        if d.numel() == 0:
            return 0.0

    return (d <= thresh).float().mean().item()

def soft_targets_from_heatmaps(gt_hm: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Turn per‑pixel Gaussian targets into normalized distributions suitable for
    pixel‑wise cross‑entropy (one‑hot softened by the Gaussian)."""
    B, C, H, W = gt_hm.shape
    flat = gt_hm.view(B, C, -1)
    flat = flat + eps
    flat = flat / flat.sum(dim=-1, keepdim=True)
    return flat

# =========================================================
# Sudoku Journey synthetic generator (D1..D8)
# =========================================================

# Footprint buckets
FOOTPRINTS = {
    "S": (0.15, 0.30),
    "M": (0.30, 0.60),
    "L": (0.60, 0.90),
}
FOOT_KEYS = ["S", "M", "L"]

CONFIGS_D2 = [
    # (cells_h, cells_w, std_h, std_v, div_h, div_v)
    (4, 4, 2, 2, 1, 1),
    (6, 6, 4, 3, 1, 2),
    (6, 6, 3, 4, 2, 1),
    (8, 8, 6, 4, 1, 3),
    (8, 8, 4, 6, 3, 1),
    (9, 9, 6, 6, 2, 2),  # favored 50% of the time
]

def _choose_config_d2(rng: random.Random):
    if rng.random() < 0.5:
        return CONFIGS_D2[-1]
    idx = rng.randrange(len(CONFIGS_D2) - 1)
    return CONFIGS_D2[idx]

def _draw_polyline(img, pts, color, thickness, broken_pct, rng: random.Random):
    if broken_pct <= 0:
        cv2.polylines(img, [pts.astype(np.int32)], False, color, thickness, lineType=cv2.LINE_AA)
        return
    # dashed by skipping short spans
    keep_prob = 1.0 - broken_pct
    for i in range(len(pts)-1):
        p0, p1 = pts[i], pts[i+1]
        seg_len = float(np.linalg.norm(p1 - p0))
        n = max(1, int(seg_len/6))
        for k in range(n):
            t0 = k/n
            t1 = (k+1)/n
            s0 = p0*(1-t0) + p1*t0
            s1 = p0*(1-t1) + p1*t1
            if rng.random() < keep_prob:
                cv2.line(img, tuple(np.int32(s0)), tuple(np.int32(s1)), color, thickness, cv2.LINE_AA)

def _edge_samples(style: str, p0: np.ndarray, p1: np.ndarray, rng: random.Random, severity=0.0):
    """Return Nx2 points for an edge; style in {'S','W','B','Sh'}; severity in [0..1]."""
    L = float(np.linalg.norm(p1 - p0) + 1e-6)
    n = max(12, int(L / 3))
    t = np.linspace(0, 1, n, dtype=np.float32)
    pts = p0[None, :] * (1 - t)[:, None] + p1[None, :] * t[:, None]
    if style == "S" or severity <= 0:
        return pts
    v = (p1 - p0).astype(np.float32)
    nrm = np.array([-v[1], v[0]], dtype=np.float32)
    nrm = nrm / (np.linalg.norm(nrm) + 1e-6)
    taper = np.minimum(t, 1.0 - t) * 5.0
    taper = np.clip(taper, 0.0, 1.0)
    if style == "W":
        amp = (0.5 + 2.0*severity)  # ~0.5..2.5
        freq = 1.0 + 3.0*severity
        phase = rng.uniform(0, 2*np.pi)
        disp = (amp * np.sin(2*np.pi*freq*t + phase)).astype(np.float32) * taper
        pts = pts + nrm[None, :] * disp[:, None]
        return pts
    if style == "B":
        knee_t = 0.35 + 0.3*rng.random()
        knee = p0*(1-knee_t) + p1*knee_t
        defl = rng.uniform(-0.15, 0.15) * L * 0.1 * (0.5 + severity)
        knee = knee + nrm * defl
        t1 = t[t<=knee_t]; t2 = t[t>=knee_t]
        pts1 = p0[None,:]*(1-(t1/max(knee_t,1e-6)))[:,None] + knee[None,:]*((t1/max(knee_t,1e-6)))[:,None]
        denom = max(1e-6, (1-knee_t))
        pts2 = knee[None,:]*(1-((t2-knee_t)/denom))[:,None] + p1[None,:]*(((t2-knee_t)/denom))[:,None]
        return np.vstack([pts1, pts2])
    if style == "Sh":
        amp = (0.3 + 1.0*severity)
        for i in range(1, len(pts)-1):
            jitter = rng.normalvariate(0.0, amp) * taper[i]
            pts[i] = pts[i] + nrm * jitter
        return pts
    return pts

def _photographic_layer(
    img: np.ndarray,
    rng: random.Random,
    allow_low_contrast: bool = True,
    strength: float = 1.0,   # NEW: 0 (very mild) … 1 (as before)
):
    """Apply camera‑like post‑processing to a rendered grayscale image.

    A blend of uneven illumination, contrast/brightness tweaks, blur, noise,
    stains, and a gentle vignette. The `strength` parameter scales all effects."""
    H, W = img.shape[:2]
    g0 = img.astype(np.float32)  # keep original for a safety blend
    g = g0.copy()

    # helper: linear scale with strength
    s = float(max(0.0, min(1.0, strength)))
    def U(lo, hi):                  # uniform scaled by strength
        return rng.uniform(lo, lo + s*(hi - lo))
    def P(p_lo, p_hi):              # probability scaled by strength
        return rng.random() < (p_lo + s*(p_hi - p_lo))

    # 1) uneven illumination (prob ~ 0.5→0.9 as s grows)
    if P(0.5, 0.9):
        gx = cv2.getGaussianKernel(W, U(70, 120))
        gy = cv2.getGaussianKernel(H, U(70, 120))
        illum = (gy @ gx.T)
        illum = (illum - illum.min()) / (illum.max() - illum.min() + 1e-6)
        delta = U(4, 22)            # was 8..22; softer at low s
        g = np.clip(g - delta*illum, 0, 255)

    # 2) contrast / brightness (smaller excursions when s is small)
    alpha = 1.0 + U(-0.10, 0.20)    # was 0.85..1.20
    beta  = U(-8, 16)               # was -16..16
    g = np.clip(alpha*g + beta, 0, 255)

    # 3) blur (prob ~ 0.3→0.7; sigma capped by s)
    if P(0.3, 0.7):
        if rng.random() < 0.6:
            k = rng.choice([3,5,7])
            g = cv2.GaussianBlur(g, (k,k), U(0.15, 1.2))   # was 0.3..1.2
        else:
            ksize = rng.choice([5,7])
            kern = np.zeros((ksize, ksize), np.float32)
            if rng.random() < 0.5: kern[ksize//2,:] = 1.0/ksize
            else:                  kern[:,ksize//2] = 1.0/ksize
            # motion blur effect—kept mild by earlier alpha/beta ranges
            g = cv2.filter2D(g, -1, kern)

    # 4) noise (prob ~ 0.5→0.8; amplitude smaller at low s)
    if P(0.5, 0.8):
        noise = np.random.randn(H, W).astype(np.float32) * U(0.5, 8.0)
        g = np.clip(g + noise, 0, 255)

    # 5) stains / scribbles (prob ~ 0.15→0.40; weaker at low s)
    if P(0.15, 0.40):
        for _ in range(rng.randint(1,3)):
            x0, y0 = rng.randint(0, W-1), rng.randint(0, H-1)
            r = int(U(5, 18))
            c = U(-12, 20)
            cv2.circle(g, (x0,y0), r, (c,), -1, lineType=cv2.LINE_AA)

    # 6) vignette (strength ~ 0.05→0.22 with s)
    if P(0.5, 0.8):
        cx, cy = (W-1)/2, (H-1)/2
        xs = np.linspace(0, W-1, W, dtype=np.float32)
        ys = np.linspace(0, H-1, H, dtype=np.float32)
        X, Y = np.meshgrid(xs, ys)
        r = np.sqrt(((X - cx)/(W/2))**2 + ((Y - cy)/(H/2))**2)
        v = U(0.05, 0.22)
        mask = 1.0 - v * np.clip(r, 0, 1)
        g *= mask

    # 7) optional “low contrast remap” (very rare + milder)
    if allow_low_contrast and rng.random() < (0.1 + 0.3*s):
        lo = U(90, 110)             # was 80..115
        hi = U(160, 185)            # was 140..185 (too aggressive at the low end)
        if hi - lo < 60:            # safety: don’t crush dynamic range
            hi = lo + 60
        g = np.clip((g - lo) * (255.0 / max(5.0, hi - lo)), 0, 255)

    # visibility safeguard: if global contrast collapsed, blend back a bit
    p1, p99 = np.percentile(g, (1, 99))
    if (p99 - p1) < 90:             # too flat → rescue detail
        w = 0.30 * (1.0 - s)        # small pull towards original
        g = (1-w)*g + w*g0

    return np.clip(g, 0, 255).astype(np.uint8)

def _small_text(img: np.ndarray, s: str, org: Tuple[int,int], scale: float, color: int, thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (int(color),), thick, cv2.LINE_AA)





def _header_band(img: np.ndarray, rng: random.Random, grid_rect: Tuple[int,int,int,int], cells_h: int, band_color: int):
    """
    Attach a header band to the top of the grid; returns (band_rect or None).

    Tweaks:
      - heavy bands are much more common (≈60%) and prefer full width
      - very dark bands favored when heavy
      - text scale fills ~62–96% of band height
      - label pool biased toward 'STEP 5' style
    """
    H, W = img.shape[:2]
    x, y, w, h = grid_rect
    cell_h = max(1.0, h / max(1, cells_h))

    # Heavier bands ~60% of the time (magazine/book style)
    heavy = (rng.random() < 0.60)

    # Thickness: heavy = 0.9..2.1 cell_h ; normal = 0.25..1.2 cell_h
    if heavy:
        band_h = int(np.clip(rng.uniform(0.90, 2.10) * cell_h, 6, H))
    else:
        band_h = int(np.clip(rng.uniform(0.25, 1.20) * cell_h, 4, H))

    # Full width strongly preferred when heavy (90%), else 70%
    if heavy or (rng.random() < 0.70):
        bx0, bx1 = x, x + w
    else:
        min_w = max(4, int(0.25 * w))
        bw = rng.randint(min_w, w)
        bx0 = x + rng.randint(0, max(0, w - bw))
        bx1 = bx0 + bw

    by0 = max(0, y - band_h)
    by1 = y

    # Very dark when heavy (≈75%), else sometimes colored gray
    if (heavy and rng.random() < 0.75):
        band_col = 0
    else:
        band_col = int(band_color)

    cv2.rectangle(img, (bx0, by0), (bx1, by1), int(band_col), -1)

    # ---- Label (bias toward STEP 5 look) ----
    # ~40% exact "STEP 5", ~20% STEP N (3..7), otherwise other titles
    r = rng.random()
    if r < 0.40:
        label = "STEP 5"
    elif r < 0.60:
        label = f"STEP {rng.randint(3,7)}"
    else:
        label = rng.choice([
            "SUDOKU", f"Sudoku Puzzle #{rng.randint(1000,9999)}",
            "Daily Sudoku", "Sudoku Classic", "Hard #203"
        ])

    fg = 255 if band_col < 128 else 0

    # Font size ~62–96% of band height (heavier bands → bigger text)
    scale = np.clip((rng.uniform(0.62, 0.96) * band_h) / 18.0, 0.35, 3.0)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)

    # Centered 75% of the time; otherwise slightly left-biased like many mags
    if rng.random() < 0.75:
        tx = int(np.clip((bx0 + bx1 - tw) / 2, bx0 + 3, max(bx0 + 3, bx1 - tw - 3)))
    else:
        tx = int(np.clip(bx0 + rng.randint(6, max(6, w // 8)), bx0 + 3, max(bx0 + 3, bx1 - tw - 3)))
    ty = by0 + int((band_h + th) / 2)

    cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, (fg,), 1, cv2.LINE_AA)
    return (bx0, by0, bx1 - bx0, by1 - by0)

"""
def _page_frame_decoy(img: np.ndarray, rng: random.Random):
    Draw a faint, possibly warped/curved 'page frame' near image edges.
    H, W = img.shape[:2]
    margin = rng.randint(2, max(2, int(0.06*min(H, W))))
    x0, y0 = margin, margin
    x1, y1 = W - 1 - margin, H - 1 - margin

    # thickness & color: faint to medium grey
    t = rng.randint(1, 3)
    col = rng.randint(80, 170)

    # sometimes broken/lightly wavy
    broken = rng.uniform(0.0, 0.25)
    sev = rng.uniform(0.0, 0.5)
    style = rng.choice(["S","W","B","Sh"])

    def draw(a, b):
        pts = _edge_samples(style, a.astype(np.float32), b.astype(np.float32), rng, severity=sev)
        _draw_polyline(img, pts, color=col, thickness=t, broken_pct=broken, rng=rng)

    draw(np.array([x0, y0]), np.array([x1, y0]))
    draw(np.array([x1, y0]), np.array([x1, y1]))
    draw(np.array([x1, y1]), np.array([x0, y1]))
    draw(np.array([x0, y1]), np.array([x0, y0]))
"""



def _page_frame_decoy(img: np.ndarray, rng: random.Random):
    H, W = img.shape[:2]
    # mild gray rectangular frame with slight randomly broken edges
    pad = rng.randint(1, 3)
    col = rng.randint(180, 210)
    th  = rng.randint(1, 2)
    cv2.rectangle(img, (pad, pad), (W-1-pad, H-1-pad), col, th)
    # tiny gaps to make it less perfect
    for _ in range(rng.randint(2, 5)):
        if rng.random() < 0.5:
            x0 = rng.randint(pad, W-1-pad)
            x1 = min(W-1-pad, x0 + rng.randint(5, 20))
            y  = rng.choice([pad, H-1-pad])
            cv2.line(img, (x0, y), (x1, y), int(col + rng.randint(-10, 10)), thickness=th)
        else:
            y0 = rng.randint(pad, H-1-pad)
            y1 = min(H-1-pad, y0 + rng.randint(5, 20))
            x  = rng.choice([pad, W-1-pad])
            cv2.line(img, (x, y0), (x, y1), int(col + rng.randint(-10, 10)), thickness=th)







def _add_soft_shadow(
    img: np.ndarray,
    rng: random.Random,
    *,
    max_attn: float = 0.55,
    gated: bool = True,
    coverage_range: tuple[float, float] = (0.30, 0.85),
    penumbra_range: tuple[float, float] = (28, 120),
    side_weights: dict[str, float] | None = None
) -> np.uint8:
    """
    Half-plane shadow with a gentle penumbra.

    CHANGE:
      - Default side_weights are now balanced (no left/right preference).
      - Slightly narrowed ranges to avoid extreme cases at D6.
    """
    H, W = img.shape[:2]
    g = img.astype(np.float32)

    # ---------- choose side ----------
    if side_weights is None:
        # perfectly balanced weights
        side_weights = {"left": 0.25, "right": 0.25, "top": 0.25, "bottom": 0.25}

    tot = sum(max(0.0, float(v)) for v in side_weights.values())
    r = rng.random() * (tot if tot > 0 else 1.0)
    acc = 0.0
    origin_side = "right"
    for k, w in side_weights.items():
        acc += max(0.0, float(w))
        if r <= acc:
            origin_side = k
            break

    # angle of the boundary's normal (theta)
    if origin_side in ("left", "right"):
        theta = rng.uniform(-12, +12) * math.pi / 180.0  # near-vertical boundary
    else:
        theta = (90 + rng.uniform(-12, +12)) * math.pi / 180.0

    n = np.array([math.cos(theta), math.sin(theta)], np.float32)
    xs = np.linspace(0, W - 1, W, dtype=np.float32)
    ys = np.linspace(0, H - 1, H, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)

    # boundary placement 'c'
    if origin_side == "right":
        c = rng.uniform(0.50 * W, 0.85 * W)
    elif origin_side == "left":
        c = rng.uniform(0.15 * W, 0.50 * W)
    elif origin_side == "top":
        c = rng.uniform(0.15 * H, 0.50 * H)
    else:  # bottom
        c = rng.uniform(0.50 * H, 0.85 * H)

    # logistic falloff (penumbra)
    width = rng.uniform(*penumbra_range)
    d = (X * n[0] + Y * n[1]) - c
    attn = (1.0 / (1.0 + np.exp(d / max(1.0, width)))).astype(np.float32)  # 0..1

    # optional "gate" to a partial region near the emitting side
    if gated:
        gate = np.zeros_like(attn, np.float32)
        frac = float(np.clip(rng.uniform(*coverage_range), 0.05, 1.0))
        if origin_side in ("left", "right"):
            w_gate = int(round(frac * W))
            if origin_side == "left":
                gate[:, :w_gate] = 1.0
            else:
                gate[:, W - w_gate:] = 1.0
        else:
            h_gate = int(round(frac * H))
            if origin_side == "top":
                gate[:h_gate, :] = 1.0
            else:
                gate[H - h_gate:, :] = 1.0
        gate = cv2.GaussianBlur(gate, (0, 0), rng.uniform(10, 36))
        gate = np.clip(gate, 0, 1)
        attn *= gate

    a = rng.uniform(0.12, max_attn)
    out = g * (1.0 - a * attn)
    return np.clip(out, 0, 255).astype(np.uint8)


def _choose_header_footer_mode(rng: random.Random):
    """
    Decide D4+ header/footer decoration.
      - 60% of images get a decoration at all.
      - Of those, 50% a classic band; 50% text-based (header/footer/both, equal split).
      - Text line counts: header 1–2, footer 1–3.

    Returns: (mode, header_lines, footer_lines)
      mode ∈ {'none','band','header_text','footer_text','both_text'}
    """
    if rng.random() >= 0.60:
        return "none", 0, 0

    if rng.random() < 0.50:
        return "band", 0, 0

    # text family: equally likely header/footer/both
    r = rng.random()
    if r < (1.0 / 3.0):
        return "header_text", rng.randint(1, 2), 0
    elif r < (2.0 / 3.0):
        return "footer_text", 0, rng.randint(1, 3)
    else:
        return "both_text", rng.randint(1, 2), rng.randint(1, 3)




def _add_local_shadow_patches(
    img: np.ndarray,
    rng: random.Random,
    *,
    area_range: tuple[float, float] = (0.08, 0.60),   # total covered fraction of image
    n_range: tuple[int, int] = (1, 3),                # how many blobs
    darkness_range: tuple[float, float] = (0.20, 0.85),
    softness_range: tuple[float, float] = (8.0, 60.0) # Gaussian sigma px
) -> np.uint8:
    """
    Adds 1..N elliptical (optionally partly off-frame) soft shadow blobs.
    Each blob is fully dark inside and feathers out; overlaps add up.

    Returns uint8 image.
    """
    H, W = img.shape[:2]
    g = img.astype(np.float32)
    mask = np.zeros((H, W), np.float32)

    total_target = float(np.clip(rng.uniform(*area_range), 0.01, 0.9)) * (H * W)
    N = rng.randint(n_range[0], n_range[1])
    target_each = total_target / float(N)

    for _ in range(N):
        aspect = rng.uniform(0.5, 2.0)
        ry = math.sqrt(target_each / (math.pi * max(1e-6, aspect)))
        rx = aspect * ry
        rx *= rng.uniform(0.8, 1.2)
        ry *= rng.uniform(0.8, 1.2)

        # allow partly off-frame placement
        cx = rng.uniform(-0.15 * W, 1.15 * W)
        cy = rng.uniform(-0.15 * H, 1.15 * H)
        ang = rng.uniform(0, 180)

        tmp = np.zeros((H, W), np.uint8)
        cv2.ellipse(tmp, (int(round(cx)), int(round(cy))), (int(round(rx)), int(round(ry))),
                    angle=ang, startAngle=0, endAngle=360, color=255, thickness=-1, lineType=cv2.LINE_AA)
        # feather
        sigma = rng.uniform(*softness_range)
        feather = cv2.GaussianBlur(tmp.astype(np.float32) / 255.0, (0, 0), sigma)
        feather = np.clip(feather, 0.0, 1.0)

        # slight irregularity (speckle) so it’s not a perfect ellipse
        if rng.random() < 0.5:
            noise = cv2.GaussianBlur(np.random.rand(H, W).astype(np.float32), (0, 0), rng.uniform(6, 18))
            feather *= (0.85 + 0.30 * noise)

        mask += feather

    mask = np.clip(mask, 0.0, 1.0)
    a = rng.uniform(*darkness_range)
    out = g * (1.0 - a * mask)
    return np.clip(out, 0, 255).astype(np.uint8)




def _apply_cylindrical_page_warp(
    img: np.ndarray,
    corners: np.ndarray,
    rng: random.Random,
    *,
    axis: str | None = None,           # 'horizontal' bows vertical lines; 'vertical' bows horizontals
    strength: float = 0.0              # 0..1, typical 0.2..0.7
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cylindrical warp to mimic page curvature.
    axis='vertical' -> horizontal lines bow; axis='horizontal' -> vertical lines bow.
    """
    H, W = img.shape[:2]
    if axis is None:
        axis = rng.choice(["vertical", "horizontal"])
    k = (0.005 + 0.035 * float(np.clip(strength, 0.0, 1.0)))  # ~0.5%..4% of size

    xs = np.arange(W, dtype=np.float32)[None, :].repeat(H, 0)
    ys = np.arange(H, dtype=np.float32)[:, None].repeat(W, 1)
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0

    if axis == "vertical":
        y_norm = (ys - cy) / max(1.0, H)
        shift = (y_norm ** 2) * (1.0 if rng.random() < 0.5 else -1.0)  # bow in or out
        map_x = (xs + k * shift * W).astype(np.float32)
        map_y = ys.astype(np.float32)
        warped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        # transform corners (x' = x + k * ((y-cy)/H)^2 * W * sign)
        sgn = 1.0 if (shift[0,0] >= 0) else -1.0
        c = corners.copy().astype(np.float32)
        yn = (c[:, 1] - cy) / max(1.0, H)
        c[:, 0] = c[:, 0] + k * (yn ** 2) * W * sgn
        return warped, order_corners_tl_tr_br_bl(c)
    else:
        x_norm = (xs - cx) / max(1.0, W)
        shift = (x_norm ** 2) * (1.0 if rng.random() < 0.5 else -1.0)
        map_x = xs.astype(np.float32)
        map_y = (ys + k * shift * H).astype(np.float32)
        warped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        sgn = 1.0 if (shift[0,0] >= 0) else -1.0
        c = corners.copy().astype(np.float32)
        xn = (c[:, 0] - cx) / max(1.0, W)
        c[:, 1] = c[:, 1] + k * (xn ** 2) * H * sgn
        return warped, order_corners_tl_tr_br_bl(c)



def _perspective_pts(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply homography M to N×2 points."""
    pts_h = np.hstack([pts.astype(np.float32), np.ones((pts.shape[0],1), np.float32)])
    t = (M @ pts_h.T).T
    t = t[:, :2] / np.maximum(1e-6, t[:, 2:3])
    return t.astype(np.float32)

def _apply_projective_keystone(
    img: np.ndarray,
    corners: np.ndarray,
    rng: random.Random,
    strength: float = 0.45
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a mild-to-strong keystone warp to the whole image and transform corners.
    strength ∈ [0..1] controls max deviation.

    CHANGE: removed right-leaning bias; left/right/top/bottom are now equally likely.
    """
    H, W = img.shape[:2]
    src = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], np.float32)

    # 50/50 horizontal vs vertical keystone
    horizontal = (rng.random() < 0.5)

    s = float(np.clip(strength, 0.0, 1.0))
    max_dx = 0.15 * s * W
    max_dy = 0.08 * s * H

    dst = src.copy()
    if horizontal:
        # Randomly choose which side is "far" with no bias
        right_far = (rng.random() < 0.5)
        dx = rng.uniform(0.35, 1.0) * max_dx
        dy = rng.uniform(0.20, 1.0) * max_dy
        if right_far:
            dst[1,0] -= dx;  dst[2,0] -= dx
            dst[1,1] += dy;  dst[2,1] -= dy
        else:
            dst[0,0] += dx;  dst[3,0] += dx
            dst[0,1] += dy;  dst[3,1] -= dy
    else:
        top_far = (rng.random() < 0.5)
        dx = rng.uniform(0.20, 0.8) * max_dx
        dy = rng.uniform(0.35, 1.0) * max_dy
        if top_far:
            dst[0,1] += dy; dst[1,1] += dy
            dst[0,0] += dx; dst[1,0] -= dx
        else:
            dst[2,1] -= dy; dst[3,1] -= dy
            dst[2,0] -= dx; dst[3,0] += dx

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        img, M, (W, H),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    new_c = _perspective_pts(corners.astype(np.float32), M)
    return warped, order_corners_tl_tr_br_bl(new_c)



def _digits_on_grid(img: np.ndarray, rng: random.Random, grid_rect: tuple[int,int,int,int],
                    cells_h: int, cells_w: int, mode: str):
    """
    Draw digits centered in each cell.
    mode in {'empty','normal','dense'}
    - One random font picked for the whole grid
    - One random 'boldness' (thickness) picked for the whole grid
    - Each digit size is fit to 50–80% of the cell height and clamped to the cell's inner box
    """
    x, y, w, h = grid_rect
    if cells_h <= 0 or cells_w <= 0:
        return

    cw = float(w) / float(cells_w)
    ch = float(h) / float(cells_h)

    # Per-grid font and boldness (constant across the grid)
    font = rng.choice([
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX,
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
    ])
    # Thickness tied to cell size but kept small; same for all digits in this grid
    cell_min = min(cw, ch)
    thickness = 1 if cell_min < 12 else (2 if cell_min < 24 else rng.choice([2, 3]))

    # Digits fill probability per mode
    p = {"empty": 0.0, "normal": 0.30, "dense": 0.85}[mode]

    # Margin to stay clear of grid lines (proportional to cell size)
    pad = int(max(1, round(0.12 * cell_min)))  # ~12% inset
    inner_w_nom = max(1, int(round(cw)) - 2 * pad)
    inner_h_nom = max(1, int(round(ch)) - 2 * pad)

    for i in range(cells_h):
        for j in range(cells_w):
            if rng.random() >= p:
                continue

            # Choose number
            txt = str(rng.randint(1, 9))

            # Target coverage 50–80% of the inner box height
            h_frac = rng.uniform(0.50, 0.80)

            # Center of the cell
            cx = x + int(round((j + 0.5) * cw))
            cy = y + int(round((i + 0.5) * ch))

            # Fit scale inside the inner box
            s, (tw, th), base = _fit_text_scale_and_bbox(
                txt, font, thickness, inner_w_nom, inner_h_nom, target_height_frac=h_frac
            )

            # Compute baseline so the text's *bounding box* is centered at (cx, cy)
            # For OpenCV: top = y_base - th, bottom = y_base + base
            # Center condition ⇒ y_base = cy + (th - base) / 2
            x_text = int(round(cx - tw / 2))
            y_base = int(round(cy + (th - base) / 2))

            # Draw
            cv2.putText(img, txt, (x_text, y_base), font, s, (0,), thickness, cv2.LINE_AA)



def _draw_grid_lines(
    img: np.ndarray,
    rng: random.Random,
    corners: np.ndarray,
    cells_h: int,
    cells_w: int,
    t_border: List[int],
    t_div: List[int],
    t_std: List[int],
    link_mode: bool,
    broken_border: float = 0.0,
    broken_std: float = 0.0,
    distortion_level: float = 0.0,
    distort_style: str = "S",
    *,
    div_h_count: Optional[int] = None,
    div_v_count: Optional[int] = None,

    col_border: int = 0,
    col_div: Optional[int] = None,
    col_std: Optional[int] = None,
):
    """
    Draw a rectangular Sudoku grid inside 'corners' (axis-aligned expected).

    NEW:
      - col_border/col_div/col_std let us simulate very light inner lines,
        or dividers that rival the border thickness/contrast.
    """
    def _canonical_block_indices(cells: int, n_div: int | None) -> set[int]:
        if not n_div:
            return set()
        if cells == 4 and n_div == 1: return {2}
        if cells == 6:                 return {3} if n_div == 1 else {2, 4}
        if cells == 8:                 return {4} if n_div == 1 else {2, 4, 6}
        if cells == 9 and n_div == 2:  return {3, 6}
        step = max(1, cells // (n_div + 1))
        return {k * step for k in range(1, n_div + 1) if 1 <= k * step <= cells - 1}

    H, W = img.shape[:2]
    tl, tr, br, bl = corners
    x0, y0 = int(tl[0]), int(tl[1])
    x1, y1 = int(br[0]), int(br[1])
    w = x1 - x0; h = y1 - y0

    # default colors if not provided
    col_div = 0 if col_div is None else int(np.clip(col_div, 0, 255))
    col_std = 0 if col_std is None else int(np.clip(col_std, 0, 255))
    col_border = int(np.clip(col_border, 0, 255))

    def edge(a, b, thick, color):
        pts = _edge_samples(distort_style, a.astype(np.float32), b.astype(np.float32),
                            rng, severity=distortion_level)
        _draw_polyline(
            img, pts, color=int(color),
            thickness=int(max(1, round(thick))),
            broken_pct=broken_border, rng=rng
        )

    # Border (clockwise)
    edge(np.array([x0, y0]), np.array([x1, y0]), t_border[0], col_border)  # top
    edge(np.array([x1, y0]), np.array([x1, y1]), t_border[1], col_border)  # right
    edge(np.array([x1, y1]), np.array([x0, y1]), t_border[2], col_border)  # bottom
    edge(np.array([x0, y1]), np.array([x0, y0]), t_border[3], col_border)  # left

    # Internal lines (equally spaced)
    cw = w / max(1, cells_w)
    ch = h / max(1, cells_h)

    divH_idx = _canonical_block_indices(cells_h, div_h_count)
    divV_idx = _canonical_block_indices(cells_w, div_v_count)

    # Horizontals
    k_std = k_div = 0
    for k in range(1, cells_h):
        y = int(round(y0 + k * ch))
        is_div = (k in divH_idx)
        thick = (t_div[min(k_div, len(t_div)-1)] if (is_div and len(t_div) > 0)
                 else t_std[min(k_std, len(t_std)-1)] if len(t_std) > 0 else 1)
        color = (col_div if is_div else col_std)
        if is_div: k_div += 1
        else:      k_std += 1
        p0 = np.array([x0, y], np.float32); p1 = np.array([x1, y], np.float32)
        pts = _edge_samples("S" if distortion_level == 0 else "W",
                            p0, p1, rng, severity=distortion_level * 0.8)
        _draw_polyline(img, pts, color=int(color),
                       thickness=int(max(1, round(thick))),
                       broken_pct=broken_std, rng=rng)

    # Verticals
    k_std = k_div = 0
    for k in range(1, cells_w):
        x = int(round(x0 + k * cw))
        is_div = (k in divV_idx)
        thick = (t_div[min(k_div, len(t_div)-1)] if (is_div and len(t_div) > 0)
                 else t_std[min(k_std, len(t_std)-1)] if len(t_std) > 0 else 1)
        color = (col_div if is_div else col_std)
        if is_div: k_div += 1
        else:      k_std += 1
        p0 = np.array([x, y0], np.float32); p1 = np.array([x, y1], np.float32)
        pts = _edge_samples("S" if distortion_level == 0 else "W",
                            p0, p1, rng, severity=distortion_level * 0.8)
        _draw_polyline(img, pts, color=int(color),
                       thickness=int(max(1, round(thick))),
                       broken_pct=broken_std, rng=rng)







def _randn_clamped(rng: random.Random, mean: float, std: float, lo: float, hi: float) -> float:
    val = rng.normalvariate(mean, std)
    return float(np.clip(val, lo, hi))

def _thickness_ratios(rng: random.Random, n: int, mode: str) -> List[float]:
    """mode: 'equal' (all=1), 'family' (a:b:c:d etc ~ U[1,2])"""
    if mode == "equal":
        return [1.0]*n
    return [rng.uniform(1.0, 2.0) for _ in range(n)]

def _thickness_caps(cell_min: float) -> tuple[int, int]:
    """
    Returns (max_std_px, max_div_px) caps derived from cell size.
    Keeps lines realistic and avoids bar-like dividers on small cells.
    """
    max_std_px = max(1, min(3, int(round(0.25 * cell_min))))  # ≤3 px and ≤25% of cell
    max_div_px = max(2, min(6, int(round(0.40 * cell_min))))  # ≤6 px and ≤40% of cell
    return max_std_px, max_div_px




def synth_sudoku_journey(
    rng: random.Random,
    img_size: int = 128,
    difficulty: int = 1,
    sigma: float = 1.6,
    neg_frac: float = 0.03,
    occlusion_prob: float = 0.05,
    return_meta: bool = False
):
    """
    Returns (inp[1,H,W], heatmaps[4,H,W], gt_xy[4,2]) and optional meta dict.
    """
    H = W = int(img_size)

    # -------- Negatives --------
    if rng.random() < max(0.0, min(1.0, neg_frac)):
        if rng.random() < 0.35:
            base = _paper_tone(H, W, rng)
        else:
            base = np.ones((H, W), np.uint8) * rng.randint(235, 255)
        base = _photographic_layer(base, rng, allow_low_contrast=True)
        inp = np.expand_dims(base.astype(np.float32)/255.0, 0)
        hms = np.zeros((4, H, W), np.float32)
        xy = np.full((4,2), -1.0, np.float32)
        out = (inp.astype(np.float32), hms.astype(np.float32), xy.astype(np.float32))
        if return_meta:
            return (*out, {"negative": True})
        return out

    # -------- Difficulty selection --------
    d = int(np.clip(difficulty, 1, 8))
    base_d = d
    apply_photo = False
    if d == 1:
        base_d = 1; apply_photo = False
    elif 2 <= d <= 7:
        lower = list(range(1, d))
        weights_lower = [0.5 / (d - 1)] * len(lower)
        choices = lower + [d]; weights = weights_lower + [0.5]
        base_d = rng.choices(choices, weights=weights, k=1)[0]
        apply_photo = False
    else:  # d == 8
        base_d = rng.randint(1, 7)
        apply_photo = (rng.random() < 0.5)

    # -------- Canvas --------
    base = np.ones((H, W), np.uint8) * rng.randint(235, 255)

    # Footprint bucket (equal share)
    bucket = FOOT_KEYS[rng.randrange(3)]
    fa = FOOTPRINTS[bucket]
    target_area_frac = rng.uniform(fa[0], fa[1])
    target_area = target_area_frac * (H * W)

    # Aspect ratio (square until D4; near-square favored after)
    if base_d <= 4:
        ar = 1.0
    else:
        r = rng.random()
        if r < 0.60:
            ar = rng.uniform(0.90, 1.10)
        elif r < 0.80:
            ar = rng.uniform(0.60, 0.90)
        else:
            ar = rng.uniform(1.10, 1.80)

    h = int(round(math.sqrt(target_area / max(ar, 1e-6))))
    w = int(round(ar * h))
    w = max(20, min(W - 6, w))
    h = max(20, min(H - 6, h))

    # --- Placement: tight framing / clipping ---
    tight = (rng.random() < 0.55)
    clip  = (rng.random() < 0.25)
    if tight:
        hug_top    = (rng.random() < 0.5)
        hug_left   = (rng.random() < 0.5)
        hug_bottom = (not hug_top) and (rng.random() < 0.25)
        hug_right  = (not hug_left) and (rng.random() < 0.25)
        x0 = rng.randint(0 if hug_left else 3, (W - w - 1) if not hug_right else max(0, 3))
        y0 = rng.randint(0 if hug_top  else 3, (H - h - 1) if not hug_bottom else max(0, 3))
        if clip:
            if hug_left and rng.random() < 0.8:  x0 -= rng.randint(1, 6)
            if hug_top  and rng.random() < 0.8:  y0 -= rng.randint(1, 6)
            if hug_right and rng.random() < 0.5: x0 += rng.randint(1, 6)
            if hug_bottom and rng.random() < 0.5: y0 += rng.randint(1, 6)
    else:
        x0 = rng.randint(3, W - w - 3)
        y0 = rng.randint(3, H - h - 3)

    x1 = x0 + w
    y1 = y0 + h
    corners = order_corners_tl_tr_br_bl(np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]], np.float32))
    valid = np.array([True, True, True, True], np.bool_)

    # -------- D1 border thickness --------
    r = rng.random()
    if   r < 0.25: border_base = 1
    elif r < 0.60: border_base = 2
    elif r < 0.85: border_base = 3
    elif r < 0.95: border_base = 4
    else:          border_base = 5
    border_ratios = [1.0, 1.0, 1.0, 1.0] if base_d == 1 else _thickness_ratios(rng, 4, "family")
    t_border = [max(1, int(round(border_base * r))) for r in border_ratios]


    # ---- Hard-combo flag for D6 (25% of the time) ----
    hard_combo = (base_d == 6) and (rng.random() < 0.25)

    # If we're in the hard-combo branch, force **thin** borders
    if hard_combo:
        border_base = 1
        t_border = [1, 1, 1, 1]


    # Defaults for later stages
    cells_h, cells_w = 0, 0
    t_div, t_std = [], []
    link_mode = True
    config_name = "D1"
    nh = nv = 0

    col_border, col_div, col_std = 0, None, None

    # -------- D2 internal lines (configs) --------
    if base_d >= 2:
        ch, cw, std_h, std_v, div_h, div_v = _choose_config_d2(rng)
        cells_h, cells_w = ch, cw
        nh, nv = int(div_h), int(div_v)
        link_mode = (rng.random() < 0.7)

        # cell-aware caps
        cell_h = (h / max(1, ch)) if ch > 0 else (h / 4.0)
        cell_w = (w / max(1, cw)) if cw > 0 else (w / 4.0)
        cell_min = max(1.0, min(cell_h, cell_w))
        max_std_px, max_div_px = _thickness_caps(cell_min)

        # ---- line color hierarchy: border darkest, div mid, std light ----
        col_border = rng.randint(0, 25)                               # very dark
        col_div    = min(200, col_border + rng.randint(25, 55))       # mid-dark
        col_std    = min(230, col_div    + rng.randint(20, 55))       # lightest
        

        



        if link_mode:
            t_border = [border_base]*4
            n_div_pool = max(nh, nv) if (nh or nv) else 0
            mul_div = [_randn_clamped(rng, 1.0, 1.0/6.0, 0.1, 5.0) for _ in range(n_div_pool)]
            div_rel = rng.uniform(0.6, 0.9)  # keep dividers thinner than border
            t_div = [int(np.clip(int(round(border_base * div_rel * m)), 2, max_div_px)) for m in mul_div]
            m_std = _randn_clamped(rng, 0.3, 0.3, 0.1, 2.0)
            t_std_val = int(np.clip(int(round(border_base * m_std)), 1, max_std_px))
            t_std = [t_std_val]
        else:
            n_div_pool = max(nh, nv) if (nh or nv) else 0
            div_rat = _thickness_ratios(rng, n_div_pool, "family")
            t_div = [int(np.clip(int(round(border_base*r)), 2, max_div_px)) for r in div_rat]
            n_std_pool = max(max(0, ch - 1), max(0, cw - 1))
            n_std_pool = max(1, n_std_pool)
            t_std = [rng.randint(1, min(3, max_std_px)) for _ in range(n_std_pool)]

        config_name = f"{ch}x{cw}"





        # D6 hardcase
        # If we're in the hard-combo branch, enforce a **light** grid theme
        if hard_combo:
            # Make standard and divider lines on the lighter side
            col_border = rng.randint(0, 15)         # keep frame very dark/contrasty
            col_div    = rng.randint(140, 190)      # fairly light
            col_std    = rng.randint(190, 235)      # lightest

            # Narrow dividers/standard lines (respecting caps you calculated)
            if link_mode:
                div_rel = rng.uniform(0.6, 0.8)
                if len(t_div) > 0:
                    t_div = [max(1, min(max_div_px, int(round(border_base * div_rel)))) for _ in t_div]
                if len(t_std) > 0:
                    t_std = [max(1, min(max_std_px, 1))]   # keep standard lines very thin
            else:
                if len(t_div) > 0:
                    t_div = [max(1, min(max_div_px, 1)) for _ in t_div]
                if len(t_std) > 0:
                    t_std = [max(1, min(max_std_px, 1)) for _ in t_std]



    # -------- D3 digits --------
    fill_mode = None
    if base_d >= 3:
        r = rng.random()
        if r < 0.05: fill_mode = "empty"
        elif r < 0.6: fill_mode = "normal"
        else: fill_mode = "dense"

    
    
    # -------- D4 header/footer selection (defer drawing until after warps) --------
    header_footer_mode = "none"     # {"none","band","header_text","footer_text","both_text"}
    header_lines = 0
    footer_lines = 0
    band_color = None

    # If you created the D6 hard-case switch earlier, reuse it here.
    # (Safe guard so this block runs even if the name isn't defined.)
    _hard_combo = ('hard_combo' in locals() and hard_combo) or ('hard_combo' in globals() and hard_combo)

    if base_d >= 4:
        if _hard_combo:
            # Dense header for hard-combo: force a DARK band to dominate a light grid
            header_footer_mode = "band"
            band_color = int(rng.randint(0, 40))   # very dark band (high ink)
            # text modes not used in hard-combo; keep lines = 0
            header_lines = 0
            footer_lines = 0
        else:
            # Your original sampling logic (kept intact)
            if rng.random() < 0.60:  # 60% of images at D4+
                if rng.random() < 0.50:
                    # classic header band (~30% overall)
                    header_footer_mode = "band"
                    band_color = 0 if (rng.random() < 0.8) else rng.randint(40, 140)
                else:
                    # text distractors (~30% overall), split evenly
                    r = rng.random()
                    if r < (1.0/3.0):
                        header_footer_mode = "header_text"
                        header_lines = rng.randint(1, 2)
                    elif r < (2.0/3.0):
                        header_footer_mode = "footer_text"
                        footer_lines = rng.randint(1, 3)
                    else:
                        header_footer_mode = "both_text"
                        header_lines = rng.randint(1, 2)
                        footer_lines = rng.randint(1, 3)
    # else: keep "none"

    # rects are produced LATER (after rotation/curvature/keystone) in final coords
    header_rects: list[list[int]] = []
    footer_rects: list[list[int]] = []

    if base_d >= 4:
        # pick mode + line counts; drawing happens AFTER all warps
        header_footer_mode, header_lines, footer_lines = _choose_header_footer_mode(rng)
        if header_footer_mode == "band":
            # decide band color now; draw later
            band_color = 0 if rng.random() < 0.8 else rng.randint(40, 140)




    # -------- D6 distort/bend/rotate edges --------
    distortion_level = 0.0
    distort_style = "S"
    broken_border = 0.0
    broken_std = 0.0
    if base_d >= 6:
        r = rng.random()
        if r < 0.3: sev = 0.25
        elif r < 0.7: sev = 0.55
        else: sev = 0.85
        distortion_level = sev
        distort_style = rng.choice(["W","B","Sh"])
        broken_border = rng.uniform(0.0, 0.15 if sev<0.6 else 0.35)
        broken_std    = rng.uniform(0.0, 0.10 if sev<0.6 else 0.25)

    if base_d >= 7:
        broken_border = max(broken_border, rng.uniform(0.05, 0.35))
        broken_std    = max(broken_std, rng.uniform(0.05, 0.25))

    # -------- Draw grid / extras --------
    # Optional page/frame decoy (competing rectangle) for D4+ (60% chance)
    if base_d >= 4 and rng.random() < 0.60:
        _page_frame_decoy(base, rng)
     
    

    # D7: surrounding-grid decoy (replaces ambient text/clutter idea) --------
    # At D7+ add ONE neighboring duplicate grid (25% each: top/bottom/left/right),
    # placed 0.5–1.5× cell size away from the target grid edge. It may be partially out of frame.
    if base_d >= 7:
        side = rng.choice(["top", "bottom", "left", "right"])
        _add_decoy_neighbor_grid(
            base, rng,
            (x0, y0, w, h),
            max(1, cells_h), max(1, cells_w),
            side=side,
            gap_factor_range=(0.5, 1.5),
            # reuse the same styling knobs used for the main grid so it truly looks like a neighbor puzzle
            t_border=t_border,
            t_div=t_div if base_d >= 2 else [],
            t_std=t_std if base_d >= 2 else [],
            link_mode=link_mode,
            broken_border=broken_border,
            broken_std=broken_std,
            distortion_level=distortion_level,
            distort_style=distort_style,
            col_border=col_border, col_div=col_div, col_std=col_std,
        )






    # -------------------------------
    # Weak-outer stressor (1-cell-in cure)
    # -------------------------------
    weak_outer = (base_d >= 2) and (rng.random() < 0.65)  # was 0.40

    # Keep the D2 color hierarchy unless we actually apply the stressor.
    if weak_outer:
        # lighter border, darker internals
        col_border = rng.randint(110, 190)
        col_std    = rng.randint(0, 50)
        col_div    = rng.randint(0, 40) if (nh or nv) else col_std

        # slightly thinner/broken border vs inners
        t_border = [max(1, int(round(b * rng.uniform(0.45, 0.90)))) for b in t_border]
        if len(t_std) > 0:
            max_std_px, _ = _thickness_caps(min(h/max(1,cells_h), w/max(1,cells_w)))
            t_std = [min(s + rng.randint(0, 1), max_std_px) for s in t_std]
        if len(t_div) > 0:
            _, max_div_px = _thickness_caps(min(h/max(1,cells_h), w/max(1,cells_w)))
            t_div = [min(d + rng.randint(0, 1), max_div_px) for d in t_div]

        broken_border = max(broken_border, rng.uniform(0.05, 0.25))
        broken_std    = min(broken_std, rng.uniform(0.0, 0.04))  # keep internals crisp
    # -------------------------------

    # soften perimeter a bit to reduce corner snaps to full image bounds
    weak_outer = (base_d >= 4) and (rng.random() < (0.55 if base_d <= 5 else 0.35))


    _draw_grid_lines(
        base, rng, corners, max(1,cells_h), max(1,cells_w),
        t_border=t_border,
        t_div=t_div if base_d >= 2 else [],
        t_std=t_std if base_d >= 2 else [],
        link_mode=link_mode,
        broken_border=broken_border,
        broken_std=broken_std,
        distortion_level=distortion_level,
        distort_style=distort_style,
        div_h_count=nh if base_d >= 2 else 0,
        div_v_count=nv if base_d >= 2 else 0,
        col_border=col_border, col_div=col_div, col_std=col_std,
    )

    # subtle boundary washout (drawn AFTER the grid so it actually fades it)
    if weak_outer and rng.random() < 0.65:
        k = rng.randint(1, 2)                      # 1–2 px
        cv2.rectangle(base, (0,0), (W-1,H-1), rng.randint(185, 230), thickness=k)



    #if header:
    #    header_rect = _header_band(base, rng, (x0, y0, w, h), max(1,cells_h), band_color)


    if base_d >= 4 and header_footer_mode != "none":
        if header_footer_mode == "band":
            header_rect = _header_band(base, rng, (x0, y0, w, h), max(1, cells_h), band_color)

        header_rects = []
        footer_rects = []

        # remove left-only bias: randomize alignment and add tiny positional jitter
        align = rng.choices(["left", "right", "center"], weights=[0.4, 0.4, 0.2])[0]

        if header_footer_mode in ("header_text", "both_text"):
            header_rects = _draw_text_block_aligned(
                base, rng, (x0, y0, w, h), max(1, cells_h),
                where="header", n_lines=header_lines,
                height_mult_range=(0.8, 1.5), gap_px_range=(2, 5),
                align=align, jitter_px=3
            )
        if header_footer_mode in ("footer_text", "both_text"):
            footer_rects = _draw_text_block_aligned(
                base, rng, (x0, y0, w, h), max(1, cells_h),
                where="footer", n_lines=footer_lines,
                height_mult_range=(0.8, 1.5), gap_px_range=(2, 5),
                align=align, jitter_px=3
            )


    if fill_mode is not None and cells_h > 0 and cells_w > 0:
        _digits_on_grid(base, rng, (x0, y0, w, h), cells_h, cells_w, fill_mode)
    
    """
    # Ambient text/clutter at D7 - Now obsolete
    if base_d >= 7:
        for _ in range(rng.randint(2,5)):
            s = rng.choice(["SUDOKU", f"PUZZLE #{rng.randint(1000,9999)}", "DATE: __/__/__", "NOTES:"])
            org = (rng.randint(0, W-50), rng.randint(0, H-5))
            _small_text(base, s, org, scale=rng.uniform(0.3,0.45), color=rng.randint(0,30), thick=1)
    """

    # --- Global rotation (rigid) ---
    if base_d <= 7 and rng.random() < 0.35:
        max_deg = 5.0
        ang = rng.uniform(-max_deg, max_deg)
        for s in [1.0, 0.97, 0.95, 0.93, 0.90]:
            cx, cy = (W - 1) / 2, (H - 1) / 2
            M = cv2.getRotationMatrix2D((cx, cy), ang, s)
            rot = cv2.warpAffine(base, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            ones = np.ones((4, 1), np.float32)
            corners_rot = (M @ np.hstack([corners, ones]).T).T.astype(np.float32)
            if (corners_rot[:, 0].min() >= -2 and corners_rot[:, 0].max() <= (W+1) and
                corners_rot[:, 1].min() >= -2 and corners_rot[:, 1].max() <= (H+1)):
                base = rot
                corners = order_corners_tl_tr_br_bl(corners_rot)
                break

    # --- Page curvature (book bow) for D3+ ---
    if base_d >= 3 and rng.random() < (0.25 if base_d < 6 else 0.40):
        strength = rng.uniform(0.25, 0.70)
        axis = rng.choice(["vertical", "horizontal"])
        base, corners = _apply_cylindrical_page_warp(base, corners, rng, axis=axis, strength=strength)

    # --- Keystone (D6+) ---
    if base_d >= 6 and rng.random() < 0.50:
        strength = rng.uniform(0.25, 0.60)
        base, corners = _apply_projective_keystone(base, corners, rng, strength=strength)

    # --- Mild photographic "capture" layer for D4+ (NEW) ---
    if base_d >= 4 and rng.random() < 0.20:
        base = _photographic_layer(base, rng, allow_low_contrast=True, strength=0.25)

    # --- Shadows: half-plane + local blobs (bottom-heavy & rare combo) ---
    p_shadow = 0.20 if base_d <= 3 else (0.28 if base_d <= 5 else 0.35)
    if rng.random() < p_shadow:
        blob_share = [0.20, 0.25, 0.35, 0.45, 0.55, 0.65, 0.70, 0.75][base_d - 1]
        use_blob = (rng.random() < blob_share)
        combo = (rng.random() < 0.10)  # 10%: do BOTH

        # Bottom-heavy side weights
        side_w = {"left": 0.15, "right": 0.20, "top": 0.10, "bottom": 0.55}

        if (not use_blob) or combo:
            # primary pass: bottom-biased half-plane
            base = _add_soft_shadow(
                base, rng,
                max_attn=0.60, gated=True,
                coverage_range=(0.35, 0.95),
                penumbra_range=(25, 140),
                side_weights=side_w
            )
            # optional second light lateral pass (adds realism)
            if rng.random() < 0.50:
                base = _add_soft_shadow(
                    base, rng,
                    max_attn=0.35, gated=True,
                    coverage_range=(0.25, 0.70),
                    penumbra_range=(18, 90),
                    side_weights={"left":0.45, "right":0.45, "top":0.05, "bottom":0.05}
                )

        if use_blob or combo:
            # local blobs (smaller if we're in combo mode)
            area_hi = 0.60 if base_d <= 5 else 0.70
            area_range = (0.05, 0.25) if combo else (0.08, area_hi)
            n_range = (1, 1) if combo else (1, 3 if base_d <= 5 else 4)
            base = _add_local_shadow_patches(
                base, rng,
                area_range=area_range,
                n_range=n_range,
                darkness_range=(0.20, 0.85),
                softness_range=(8.0, 60.0)
            )

    # --- D8 photographic effects ---
    if d == 8 and apply_photo:
        base = _photographic_layer(base, rng, allow_low_contrast=True, strength=0.5)

    # Optional occlusions
    occ_mask = valid.copy()
    if rng.random() < max(0.0, min(1.0, occlusion_prob)):
        n_occ = rng.randint(1, 2)
        idxs = rng.sample([0,1,2,3], n_occ)
        for k in idxs:
            occ_mask[k] = False
        for _ in range(n_occ):
            rx0 = rng.randint(max(0,x0-3), min(W-5,x1))
            ry0 = rng.randint(max(0,y0-3), min(H-5,y1))
            rx1 = rng.randint(rx0+3, min(W-1, x1+3))
            ry1 = rng.randint(ry0+3, min(H-1, y1+3))
            cv2.rectangle(base, (rx0,ry0), (rx1,ry1), rng.randint(0,30), -1)

    # tensors / targets
    corners[:, 0] = np.clip(corners[:, 0], 0, W-1)
    corners[:, 1] = np.clip(corners[:, 1], 0, H-1)

    g = base.astype(np.float32) / 255.0
    inp = np.expand_dims(g, 0).astype(np.float32)
    gt_xy = corners.astype(np.float32)
    Hm = render_heatmaps(H, W, gt_xy, sigma=sigma, valid=occ_mask)

    xy_out = gt_xy.copy()
    for i in range(4):
        if not occ_mask[i]:
            xy_out[i] = [-1.0, -1.0]

    
    """
    meta = {
        "difficulty": d,
        "base_d": base_d,
        "footprint": bucket,
        "config": config_name,
        "link_mode": "linked" if link_mode else "free",

        # NEW: explicit header/footer descriptors (replaces old "header")
        "header_footer_mode": header_footer_mode,                  # {"none","band","header_text","footer_text","both_text"}
        "has_header_band": (header_footer_mode == "band"),
        "has_header_text": (header_footer_mode in ("header_text", "both_text")),
        "has_footer_text": (header_footer_mode in ("footer_text", "both_text")),

        "fill_mode": fill_mode if fill_mode else "none",
        "distortion": distortion_level,
        "neg": False,

        "header_text_rects": header_rects,   # list of [x0,y0,x1,y1]
        "footer_text_rects": footer_rects,
    }
    """

    meta = {
        "difficulty": d,
        "base_d": base_d,
        "footprint": bucket,
        "config": config_name,
        "link_mode": "linked" if link_mode else "free",
        "fill_mode": fill_mode if fill_mode else "none",
        "distortion": distortion_level,
        "neg": False,
        # NEW: pass the rectangles for suppression losses
        "header_text_rects": header_rects,
        "footer_text_rects": footer_rects,
    }

    if return_meta:
        return inp, Hm, xy_out.astype(np.float32), meta
    else:
        return inp, Hm, xy_out.astype(np.float32)







# =========================================================
# Iterable synthetic stream
# =========================================================

class SyntheticIterable(torch.utils.data.IterableDataset):
    """
    Streams synthetic Sudoku crops for training/eval.

    When return_meta=True it yields (inp, hm, xy, meta) where meta contains:
      meta["header_text_rects"]: List[List[int]]  (x0,y0,x1,y1) per header line
      meta["footer_text_rects"]: List[List[int]]  (x0,y0,x1,y1) per footer line
      ...plus any other fields you already write in synth_sudoku_journey(meta)
    Otherwise it yields the classic (inp, hm, xy).
    """
    def __init__(self,
                 img_size: int,
                 sigma: float,
                 difficulty: int,
                 seed: int = 1234,
                 neg_frac: float = 0.03,
                 occlusion_prob: float = 0.05,
                 return_meta: bool = False):
        super().__init__()
        self.img_size = img_size
        self.sigma = sigma
        self.difficulty = difficulty
        self.seed = seed
        self.neg_frac = neg_frac
        self.occlusion_prob = occlusion_prob
        self.return_meta = return_meta

    def __iter__(self):
        # Keep a Python RNG for synth_sudoku_journey (needs .choices/.uniform/etc.)
        rng_py = random.Random(self.seed + int(time.time() * 1000) % 10_000_000)
        # Optional: a NumPy RNG if inside journey you want np-level sampling,
        # but DO NOT pass this into synth_sudoku_journey directly.
        np_rng = np.random.RandomState(rng_py.randint(0, 2**31 - 1))

        # If you want synth_sudoku_journey to occasionally use NumPy randomness,
        # you can expose it via an attribute so your code can access rng.np
        # (only if synth_sudoku_journey checks for it).
        setattr(rng_py, "np", np_rng)

        while True:
            if self.return_meta:
                inp, hm, xy, meta = synth_sudoku_journey(
                    rng_py,                     # <<— IMPORTANT: python RNG, not np RNG
                    self.img_size, self.difficulty, self.sigma,
                    neg_frac=self.neg_frac, occlusion_prob=self.occlusion_prob,
                    return_meta=True
                )
                meta = meta if isinstance(meta, dict) else {}
                meta.setdefault("header_text_rects", [])
                meta.setdefault("footer_text_rects", [])
                yield (torch.from_numpy(inp), torch.from_numpy(hm), torch.from_numpy(xy), meta)
            else:
                inp, hm, xy = synth_sudoku_journey(
                    rng_py,                     # <<— pass python RNG here too
                    self.img_size, self.difficulty, self.sigma,
                    neg_frac=self.neg_frac, occlusion_prob=self.occlusion_prob,
                    return_meta=False
                )
                yield (torch.from_numpy(inp), torch.from_numpy(hm), torch.from_numpy(xy))





# =========================================================
# Model
# =========================================================

class TinyCornerNet(nn.Module):
    """
    TinyCornerNet
    -------------
    A compact heatmap head for 4 Sudoku corners.

    Why this shape?
      - Small encoder keeps training/inference fast.
      - Optional U‑Net head provides skip connections for cleaner peaks.
      - CoordConv channels (x,y) give a light positional prior without
        hard constraints.

    Structure
      enc1..enc4  : strided conv blocks that downsample to 1/16 resolution.
      up / up*    : deconvs (with optional skips) to return to 1× resolution.
      head        : 1×1 conv producing 4 heatmaps.

    Forward path
      If `use_coordconv`, two channels with normalized coordinates are
      concatenated to the input before the encoder.
    """
    def __init__(self, in_ch_gray: int = 1, out_ch: int = 4, base: int = 24,
                 use_coordconv: bool = True, unet_head: bool = False):
        super().__init__()
        self.use_coordconv = use_coordconv
        self.unet_head = unet_head
        eff_in = in_ch_gray + (2 if use_coordconv else 0)
        c1, c2, c3, c4 = base, base*2, base*3, base*4

        self.enc1 = nn.Sequential(nn.Conv2d(eff_in, c1, 3, 2, 1), nn.BatchNorm2d(c1), nn.ReLU(inplace=True))  # 64
        self.enc2 = nn.Sequential(nn.Conv2d(c1, c2, 3, 2, 1), nn.BatchNorm2d(c2), nn.ReLU(inplace=True))      # 32
        self.enc3 = nn.Sequential(nn.Conv2d(c2, c3, 3, 2, 1), nn.BatchNorm2d(c3), nn.ReLU(inplace=True))      # 16
        self.enc4 = nn.Sequential(nn.Conv2d(c3, c4, 3, 2, 1), nn.BatchNorm2d(c4), nn.ReLU(inplace=True))      # 8

        if unet_head:
            self.up4 = nn.Sequential(nn.ConvTranspose2d(c4, c3, 2, 2), nn.ReLU(inplace=True))   # 16
            self.up3 = nn.Sequential(nn.ConvTranspose2d(c3*2, c2, 2, 2), nn.ReLU(inplace=True)) # 32
            self.up2 = nn.Sequential(nn.ConvTranspose2d(c2*2, c1, 2, 2), nn.ReLU(inplace=True)) # 64
            self.up1 = nn.Sequential(nn.ConvTranspose2d(c1*2, c1, 2, 2), nn.ReLU(inplace=True)) # 128
            self.head = nn.Conv2d(c1, out_ch, 1)
        else:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(c4, c3, 2, 2), nn.ReLU(inplace=True),  # 16
                nn.ConvTranspose2d(c3, c2, 2, 2), nn.ReLU(inplace=True),  # 32
                nn.ConvTranspose2d(c2, c1, 2, 2), nn.ReLU(inplace=True),  # 64
                nn.ConvTranspose2d(c1, c1, 2, 2), nn.ReLU(inplace=True),  # 128
            )
            self.head = nn.Conv2d(c1, out_ch, 1)

    @staticmethod
    def _coord_channels(B: int, H: int, W: int, device):
        xs = torch.linspace(0.0, 1.0, W, device=device).view(1, 1, 1, W).expand(B, -1, H, -1)
        ys = torch.linspace(0.0, 1.0, H, device=device).view(1, 1, H, 1).expand(B, -1, -1, W)
        return xs, ys

    def forward(self, x):
        if self.use_coordconv:
            B, _, H, W = x.shape
            xs, ys = self._coord_channels(B, H, W, x.device)
            x = torch.cat([x, xs, ys], dim=1)  # [B,3,H,W]
        e1 = self.enc1(x); e2 = self.enc2(e1); e3 = self.enc3(e2); e4 = self.enc4(e3)
        if self.unet_head:
            u4 = self.up4(e4)
            u3 = self.up3(torch.cat([u4, e3], dim=1))
            u2 = self.up2(torch.cat([u3, e2], dim=1))
            u1 = self.up1(torch.cat([u2, e1], dim=1))
            return self.head(u1)
        else:
            x = self.up(e4)
            return self.head(x)

# =========================================================
# Visualization
# =========================================================

def colorize(hm: np.ndarray) -> np.ndarray:
    """Map a floating heatmap (0..1) to a JET color image for visualization."""
    hm8 = np.clip(hm * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(hm8, cv2.COLORMAP_JET)

def save_viz_samples(save_dir: str,
                     epoch: int,
                     imgs: torch.Tensor,
                     gt_hms: torch.Tensor,
                     pred_hms: torch.Tensor,
                     gt_xy: torch.Tensor,
                     pr_xy: torch.Tensor,
                     max_samples: int = 64):
    """Create a side‑by‑side panel for quick qualitative inspection.

    Left: grayscale input with GT corners (colored) and predictions (red).
    Right: 4×2 grid of (GT heatmap, predicted heatmap) per corner."""
    ensure_dir(save_dir)
    imgs = imgs.cpu().numpy()
    gt_hms = gt_hms.cpu().numpy()
    pred_hms = pred_hms.detach().cpu().numpy()
    gt_xy = gt_xy.cpu().numpy()
    pr_xy = pr_xy.detach().cpu().numpy()

    N, _, H, W = imgs.shape
    K = min(max_samples, N)
    colors = [(0,255,0),(0,255,255),(255,255,0),(255,0,255)]

    for i in range(K):
        g = imgs[i, 0]
        gt_pts = [(gt_xy[i, j, 0], gt_xy[i, j, 1]) for j in range(4)]
        pr_pts = [(pr_xy[i, j, 0], pr_xy[i, j, 1]) for j in range(4)]

        left = cv2.cvtColor((g * 255.0).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for j,(x,y) in enumerate(gt_pts):
            if x >= 0 and y >= 0:
                cv2.circle(left, (int(round(x)), int(round(y))), 3, colors[j], -1, lineType=cv2.LINE_AA)
        for j,(x,y) in enumerate(pr_pts):
            cv2.circle(left, (int(round(x)), int(round(y))), 3, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        left = cv2.resize(left, (256, 256), interpolation=cv2.INTER_NEAREST)

        rows = []
        for j in range(4):
            gt_c = colorize(gt_hms[i, j])
            pr_c = colorize(pred_hms[i, j])
            gt_c = cv2.resize(gt_c, (128, 128), interpolation=cv2.INTER_NEAREST)
            pr_c = cv2.resize(pr_c, (128, 128), interpolation=cv2.INTER_NEAREST)
            rows.append(np.hstack([gt_c, pr_c]))
        right = np.vstack(rows)
        right = cv2.resize(right, (256, 256), interpolation=cv2.INTER_AREA)

        panel = np.hstack([left, right])
        out_path = os.path.join(save_dir, f"ep{epoch:03d}_sample{i:02d}.png")
        cv2.imwrite(out_path, panel)

# =========================================================
# Losses & eval
# =========================================================

def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    pos_weight: torch.Tensor | None = None,  # [1,C,1,1]
    reduction: str = "mean",
):
    """Classic focal BCE with logits (alpha, gamma, pos_weight).

    We keep this version for evaluation consistency; training usually uses the
    implementation wired directly in the loop for speed."""
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=pos_weight
    )
    p = torch.sigmoid(logits)
    p_t = p*targets + (1 - p)*(1 - targets)
    alpha_t = alpha*targets + (1 - alpha)*(1 - targets)
    focal = alpha_t * ((1.0 - p_t) ** gamma) * bce
    if reduction == "mean":
        return focal.mean()
    elif reduction == "sum":
        return focal.sum()
    else:
        return focal

def _hm_peak_metrics(pred_hm: torch.Tensor) -> Tuple[float,float]:
    """Two quick scalars from predicted heatmaps: channel‑wise peak height and
    average "sharpness" (peak minus its 3×3 average). Good sanity checks."""
    with torch.no_grad():
        B,C,H,W = pred_hm.shape
        mx = torch.amax(pred_hm, dim=(-1,-2))
        pad = (1,1,1,1)
        avg = F.avg_pool2d(F.pad(pred_hm, pad, mode="replicate"), kernel_size=3, stride=1)
        flat = pred_hm.view(B,C,-1)
        idx = torch.argmax(flat, dim=-1)
        av_flat = avg.view(B,C,-1)
        av_at = torch.gather(av_flat, -1, idx.unsqueeze(-1)).squeeze(-1)
        sharp = (mx - av_at).mean().item()
        mx_mean = mx.mean().item()
        return mx_mean, sharp

def evaluate(model, loader, device, tau_eval: float,
             use_focal_bce: bool, pos_weight: float,
             focal_alpha: float, focal_gamma: float,
             lambda_bce: float, lambda_ce: float,
             max_eval_batches: Optional[int] = None):
    """Run the model in eval mode and compute scalar metrics.

    We compute a validation loss (BCE/CE mix) plus px error and PCK@{2,3,5,10}.
    For masked/occluded corners we skip coordinate terms so negatives don't pollute
    the averages."""
    model.eval()
    pos_w = torch.tensor([pos_weight]*4, device=device, dtype=torch.float32).view(1, 4, 1, 1)

    def bce_term_eval(logits, targets):
        logits  = ensure_chw_4(logits,  name="logits(eval-bce)").float()
        targets = ensure_chw_4(targets, name="targets(eval-bce)").float()
        if use_focal_bce:
            return focal_bce_with_logits(logits, targets, alpha=focal_alpha, gamma=focal_gamma, pos_weight=pos_w, reduction="mean")
        else:
            return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_w, reduction="mean")

    tot_loss, n_pix = 0.0, 0
    all_err = []
    all_pred_pts = []
    all_gt_pts = []
    peak_mx_vals, peak_sharp_vals = [], []

    with torch.no_grad():
        for b_i, batch in enumerate(loader):
            if (max_eval_batches is not None) and (b_i >= max_eval_batches):
                break

            # ---- Flexible unpack: accept (inp, gt_hm, gt_xy) or (inp, gt_hm, gt_xy, meta)
            if isinstance(batch, (list, tuple)):
                if len(batch) >= 3:
                    inp, gt_hm, gt_xy = batch[0], batch[1], batch[2]
                else:
                    # malformed batch; skip
                    continue
            elif isinstance(batch, dict):
                # Optional support if a dict-based loader is used
                try:
                    inp, gt_hm, gt_xy = batch["inp"], batch["gt_hm"], batch["gt_xy"]
                except Exception:
                    continue
            else:
                # Unknown structure; skip
                continue
            # ---------------------------------------------

            inp = inp.to(device)
            gt_hm = ensure_chw_4(gt_hm.to(device).float(), name="gt_hm(eval)")
            gt_xy = gt_xy.to(device).float()

            logits = model(inp)
            logits = ensure_chw_4(logits, name="logits(eval)")

            loss_bce = bce_term_eval(logits, gt_hm)

            B, C, Hm, Wm = logits.shape
            log_probs = torch.log_softmax(logits.view(B, C, -1), dim=-1)
            target = soft_targets_from_heatmaps(gt_hm)
            loss_ce = -(target * log_probs).sum(dim=-1).mean()

            pred_hm = torch.sigmoid(logits)
            pr_xy = spatial_softargmax(pred_hm, tau=tau_eval)

            # Masked coordinate loss (for logging total loss)
            msk = valid_corner_mask(gt_xy)  # [B,4]
            if msk.any():
                err_mat = torch.linalg.norm(pr_xy - gt_xy, dim=-1)  # [B,4]
                coord_loss = err_mat[msk].mean()
            else:
                coord_loss = torch.tensor(0.0, device=gt_xy.device)

            # coord not in eval loss; px_err is reported separately
            loss = lambda_bce*loss_bce + lambda_ce*loss_ce + coord_loss*0.0
            tot_loss += loss.item() * inp.size(0)
            n_pix += inp.size(0)

            # Collect masked errors and points for PCK after loop
            err_flat = torch.linalg.norm(pr_xy - gt_xy, dim=-1).reshape(-1)  # [B*4]
            all_err.append(err_flat[msk.reshape(-1)])

            all_pred_pts.append(pr_xy.reshape(-1, 2)[msk.reshape(-1)])
            all_gt_pts.append(gt_xy.reshape(-1, 2)[msk.reshape(-1)])

            mx, sharp = _hm_peak_metrics(pred_hm)
            peak_mx_vals.append(mx); peak_sharp_vals.append(sharp)

    if n_pix == 0:
        return {"loss": 0, "px_err": 0, "pck2": 0, "pck3": 0, "pck5": 0, "pck10": 0, "hm_peak": 0, "hm_sharp": 0}

    if len(all_err) == 0 or sum([e.numel() for e in all_err]) == 0:
        return {"loss": tot_loss / n_pix, "px_err": 0.0, "pck2": 0.0, "pck3": 0.0, "pck5": 0.0, "pck10": 0.0,
                "hm_peak": float(np.mean(peak_mx_vals)) if peak_mx_vals else 0.0,
                "hm_sharp": float(np.mean(peak_sharp_vals)) if peak_sharp_vals else 0.0}

    errs = torch.cat([e.reshape(-1) for e in all_err], dim=0)            # only valid
    pr_pts = torch.cat(all_pred_pts, dim=0) if all_pred_pts else torch.empty(0,2, device=device)
    gt_pts = torch.cat(all_gt_pts, dim=0) if all_gt_pts else torch.empty(0,2, device=device)

    def _pck(pred, gt, t):
        if pred.numel() == 0:
            return 0.0
        d = torch.linalg.norm(pred - gt, dim=-1)
        return (d <= t).float().mean().item()

    return {
        "loss": tot_loss / n_pix,
        "px_err": (errs.mean().item() if errs.numel() > 0 else 0.0),
        "pck2": _pck(pr_pts, gt_pts, 2.0),
        "pck3": _pck(pr_pts, gt_pts, 3.0),
        "pck5": _pck(pr_pts, gt_pts, 5.0),
        "pck10": _pck(pr_pts, gt_pts, 10.0),
        "hm_peak": float(np.mean(peak_mx_vals)) if peak_mx_vals else 0.0,
        "hm_sharp": float(np.mean(peak_sharp_vals)) if peak_sharp_vals else 0.0,
    }

# =========================================================
# Debug GT
# =========================================================

def debug_gt_samples(ds, out_dir="runs/corners_debug_gt", k=8):
    """Save a few raw GT renders to disk to sanity‑check the generator and labels."""
    ensure_dir(out_dir)
    for i in range(min(k, len(ds))):
        inp, gt_hm, gt_xy = ds[i]
        g = (inp[0].numpy() * 255).astype(np.uint8)
        g_bgr = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        cols = [(0,255,0),(0,255,255),(255,255,0),(255,0,255)]
        for j,(x,y) in enumerate(gt_xy.numpy()):
            if x >= 0 and y >= 0:
                cv2.circle(g_bgr, (int(x),int(y)), 3, cols[j], -1, lineType=cv2.LINE_AA)
        cv2.imwrite(f"{out_dir}/sample_{i:02d}.png", g_bgr)





# =========================================================
# Training
# =========================================================

def train(data_root: Optional[str],
          out_dir: str = "runs/corners_exp",
          epochs: int = 12,
          batch_size: int = 96,
          lr: float = 3e-4,
          img_size: int = 128,
          sigma: float = 1.6,
          base: int = 24,
          val_split: float = 0.1,
          tau: float = 0.5,
          tau_final: Optional[float] = None,
          augment: bool = True,
          viz_out: str = None,
          viz_every: int = 1,
          viz_count: int = 64,
          seed: int = 1234,
          lambda_coord: float = 0.25,
          lambda_bce: float = 1.0,
          lambda_ce: float = 0.0,
          pos_weight: float = 1.0,
          use_focal_bce: bool = False,
          focal_alpha: float = 0.25,
          focal_gamma: float = 2.0,
          unet_head: bool = False,
          plateau_patience: int = 3,
          debug_shapes_once: bool = True,
          save_best_by: str = "val_loss",
          # automation
          monitor_metric: str = "px_err",
          stall_window: int = 8,
          min_improve_px: float = 0.25,
          lr_decay_factor: float = 0.5,
          lr_min: float = 1e-5,
          max_lr_decays: int = 2,
          lcoord_bump_after_frac: float = 0.66,
          lcoord_bump_factor: float = 1.25,
          lambda_coord_max: float = 6.0,
          max_lcoord_bumps: int = 1,
          early_stop_after: int = 24,
          # synthetic controls
          use_synth: bool = False,
          synth_difficulty: int = 1,
          real_frac: float = 0.0,
          steps_per_epoch: Optional[int] = None,
          # journey extras
          neg_frac: float = 0.03,
          occlusion_prob: float = 0.05,
          resume: Optional[str] = None,
          lambda_negmax: float = 0.0,
          lambda_inset: float = 0.0,
          lambda_outset: float = 0.0,
          lambda_ring: float = 0.0,
          inset_frac: float = 0.12,
          inset_radius: int = 3,
          lambda_textslab: float = 0.0,   # NEW
          ):
    """Main training entry point used by the CLI."""
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ensure_dir(out_dir)

    # --- Real dataset (optional) ---
    ds = None
    loader_va = None
    real_loader = None
    real_iter = None
    tr_idx = []

    if data_root:
        ds = CornerHeatmapDataset(data_root, img_size=img_size, sigma=sigma, augment=augment)

        # split train/val
        n = len(ds)
        idxs = list(range(n))
        random.Random(seed).shuffle(idxs)
        n_val = max(1, int(n * val_split))
        val_idx = idxs[:n_val]
        tr_idx  = idxs[n_val:]

        ds_tr = torch.utils.data.Subset(ds, tr_idx)
        ds_va = torch.utils.data.Subset(ds, val_idx)

        # small real-val loader (no replacement)
        loader_va = DataLoader(
            ds_va,
            batch_size=min(batch_size, 64),
            shuffle=False,
            num_workers=(2 if os.name != "nt" else 0),
            persistent_workers=False,
            pin_memory=False,
            drop_last=False,
        )

        # REAL train loader with replacement so an epoch never runs out
        from torch.utils.data import RandomSampler
        samples_per_epoch = (steps_per_epoch or 100) * batch_size

        real_sampler = RandomSampler(
            ds_tr,
            replacement=True,
            num_samples=samples_per_epoch,
        )

        real_loader = DataLoader(
            ds_tr,
            batch_size=batch_size,
            sampler=real_sampler,   # <- use sampler; do NOT also set shuffle
            num_workers=(2 if os.name != "nt" else 0),
            persistent_workers=False,
            pin_memory=False,
            drop_last=False,
        )
    else:
        print("ℹ️ No real dataset provided; training will use synthetic only if --use-synth is set.")

    # --- Synthetic stream (optional) ---
    synth_loader = None
    synth_iter = None
    if use_synth:
        synth_loader = DataLoader(
            SyntheticIterable(
                img_size=img_size, sigma=sigma, difficulty=synth_difficulty,
                seed=seed, neg_frac=neg_frac, occlusion_prob=occlusion_prob,
                return_meta=True,                    # IMPORTANT
            ),
            batch_size=batch_size,
            num_workers=0,
            collate_fn=collate_corners_batch,       # IMPORTANT
        )
        synth_iter = iter(synth_loader)

    # --- Model/opt
    model = TinyCornerNet(in_ch_gray=1, out_ch=4, base=base, use_coordconv=True, unet_head=unet_head).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    # LOAD RESUME
    if resume:
        ckpt = torch.load(resume, map_location=device)
        state = ckpt.get("model", ckpt)  # support raw state_dict or {"model": ...}
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"↪️  Resumed weights from {resume} | missing={len(missing)} unexpected={len(unexpected)}")

    # --- Loss helpers
    pos_w_vec = torch.tensor([pos_weight] * 4, device=device, dtype=torch.float32)
    pos_w_bc  = pos_w_vec.view(1, 4, 1, 1)

    def _norm4(x: torch.Tensor) -> torch.Tensor:
        x = ensure_chw_4(x)
        if x.dtype != torch.float32:
            x = x.float()
        return x.contiguous()

    def bce_term(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits  = _norm4(logits)
        targets = _norm4(targets).type_as(logits)
        try:
            return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_w_vec, reduction="mean")
        except RuntimeError:
            return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_w_bc, reduction="mean")

    def focal_term(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits  = _norm4(logits)
        targets = _norm4(targets).type_as(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_w_bc, reduction="none")
        p       = torch.sigmoid(logits)
        p_t     = p * targets + (1 - p) * (1 - targets)
        alpha_t = focal_alpha * targets + (1 - focal_alpha) * (1 - targets)
        focal   = alpha_t * ((1.0 - p_t) ** focal_gamma) * bce
        return focal.mean()

    bce_descr = (f"FocalBCE(alpha={focal_alpha}, gamma={focal_gamma}, pos_w={pos_weight})"
                 if use_focal_bce else f"BCEWithLogits(pos_weight={pos_weight})")

    # --- Logging
    csv_path = Path(out_dir) / "metrics.csv"
    ensure_dir(out_dir)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch","train_loss","val_loss","px_err","pck2","pck3","pck5","pck10",
                    "hm_peak","hm_sharp","lr","tau","lambda_coord","notes"])
    print(f"🔧 Using {bce_descr} | λ_coord={lambda_coord} λ_bce={lambda_bce} λ_ce={lambda_ce} | "
          f"{'U-Net head ON' if unet_head else 'U-Net head OFF'} | "
          f"τ_init={tau}" + (f" → τ_final={tau_final}" if tau_final is not None else ""))
    print(f"💾 Checkpoint policy: save_best_by = {save_best_by} (also tracking best_by_loss.pt and best_by_px.pt)")

    # --- Best tracking
    best_metric = float("inf")
    best_metric_ep = 0
    best_loss_val = float("inf")
    best_loss_ep = 0
    best_px_val = float("inf")
    best_px_ep = 0

    epochs_since_primary_best = 0
    printed_debug = False

    # --- Stall state
    px_history: list[float] = []
    lr_decays_used = 0
    lcoord_bumps_used = 0

    def current_lr() -> float:
        return float(opt.param_groups[0]["lr"])

    def set_lr(new_lr: float):
        for g in opt.param_groups:
            g["lr"] = new_lr

    # --- helper to draw next batch from (real,synth) based on real_frac
    def next_batch():
        nonlocal real_iter, real_loader, synth_iter, synth_loader

        def _next_real():
            nonlocal real_iter, real_loader
            if real_iter is None:
                if real_loader is None:
                    raise RuntimeError("Real loader not available")
                real_iter = iter(real_loader)
            try:
                return next(real_iter)
            except StopIteration:
                real_iter = iter(real_loader)
                return next(real_iter)

        def _next_synth():
            nonlocal synth_iter, synth_loader
            if synth_iter is None:
                if synth_loader is None:
                    raise RuntimeError("Synthetic iterator not available")
                synth_iter = iter(synth_loader)
            try:
                return next(synth_iter)
            except StopIteration:
                # recreate your synthetic iterator here
                synth_iter = iter(synth_loader)
                return next(synth_iter)

        has_real = real_loader is not None
        has_synth = synth_loader is not None

        if has_real and has_synth:
            # Mix by real_frac
            return _next_real() if random.random() < real_frac else _next_synth()
        elif has_real:
            return _next_real()
        elif has_synth:
            return _next_synth()
        else:
            raise RuntimeError("No data streams available: provide --data and/or --use-synth")

    # =====================================================
    # Training loop
    # =====================================================
    for ep in range(1, epochs + 1):

        # Reset iterators (safety for finite loaders)
        if real_loader is not None:
            real_iter = iter(real_loader)
        else:
            real_iter = None
        if synth_loader is not None:
            synth_iter = iter(synth_loader)

        # τ annealing
        if tau_final is not None and epochs > 1:
            t = (ep - 1) / (epochs - 1)
            tau_cur = (1 - t) * tau + t * tau_final
        else:
            tau_cur = tau

        model.train()
        run_loss = 0.0
        seen = 0

        # define steps this epoch
        if steps_per_epoch is None:
            steps = 100 if ds is None else max(1, math.ceil(len(tr_idx) / batch_size))
        else:
            steps = int(steps_per_epoch)

        pbar = tqdm(range(steps), desc=f"Epoch {ep}/{epochs}")
        for _ in pbar:
            batch = next_batch()
            if isinstance(batch, (list, tuple)) and len(batch) == 4:
                inp, gt_hm, gt_xy, batch_meta = batch
            else:
                inp, gt_hm, gt_xy = batch
                batch_meta = {"header_text_rects_list": [], "footer_text_rects_list": []}

            inp   = inp.to(device)
            gt_hm = ensure_chw_4(gt_hm.to(device).float())
            gt_xy = gt_xy.to(device).float()

            opt.zero_grad()
            logits = model(inp)
            logits = ensure_chw_4(logits)

            if debug_shapes_once and not printed_debug:
                print(f"DEBUG shapes: inp {tuple(inp.shape)} logits {tuple(logits.shape)} gt_hm {tuple(gt_hm.shape)}")
                printed_debug = True

            # ---- losses ----
            # BCE / focal
            if use_focal_bce:
                loss_b = focal_term(logits, gt_hm)
            else:
                loss_b = bce_term(logits, gt_hm)

            # pixelwise CE to soft-targets
            B, C, Hm, Wm = logits.shape
            log_probs = torch.log_softmax(logits.view(B, C, -1), dim=-1)
            target = soft_targets_from_heatmaps(gt_hm)
            loss_ce = -(target * log_probs).sum(dim=-1).mean()

            # predictions & coords
            pred_probs = torch.sigmoid(logits)
            pr_xy = spatial_softargmax(pred_probs, tau=tau_cur)

            # masked coordinate loss
            mask = valid_corner_mask(gt_xy)  # [B,4] bool
            if mask.any():
                err = torch.linalg.norm(pr_xy - gt_xy, dim=-1)  # [B,4]
                coord_loss = err[mask].mean()
            else:
                coord_loss = torch.tensor(0.0, device=gt_xy.device)

            # ---- base loss (build FIRST) ----
            loss = lambda_bce * loss_b + lambda_ce * loss_ce + lambda_coord * coord_loss

            pos_img_mask = (gt_hm.sum(dim=(1, 2, 3)) > 0)  # [B]

            # ---- header/footer text suppression (positives only) ----
            if lambda_textslab > 0.0 and pos_img_mask.any():
                B = inp.size(0)
                # Try to get per-sample lists; if missing/short, synthesize empty ones of length B
                hdr_full = batch_meta.get("header_text_rects_list", None)
                ftr_full = batch_meta.get("footer_text_rects_list", None)
                if not isinstance(hdr_full, list) or len(hdr_full) != B:
                    hdr_full = [[] for _ in range(B)]
                if not isinstance(ftr_full, list) or len(ftr_full) != B:
                    ftr_full = [[] for _ in range(B)]

                # Indices of the positive subset (the tensors we pass to the loss)
                pos_idx = pos_img_mask.nonzero(as_tuple=False).view(-1).tolist()

                # Slice the rect-lists to match exactly the positive subset length
                hdr_pos = [hdr_full[i] for i in pos_idx]
                ftr_pos = [ftr_full[i] for i in pos_idx]

                txt_loss = text_suppression_max(
                    pred_probs[pos_img_mask],    # [Bp,4,H,W]
                    gt_xy[pos_img_mask],         # [Bp,4,2]
                    hdr_pos,                     # list length == Bp
                    ftr_pos,                     # list length == Bp
                    exempt_radius=8              # feel free to use 6–8; 8 is a bit more forgiving
                )
                loss = loss + lambda_textslab * txt_loss

            # ---- generic gutter suppression (positives only) ----
            # small constant weight; discourages peaks in narrow bands above/below grid borders
            if pos_img_mask.any():
                gutter_lambda = 0.10
                g_loss = gutter_suppression_loss(
                    pred_probs[pos_img_mask],
                    gt_xy[pos_img_mask],
                    thickness=6
                )
                loss = loss + gutter_lambda * g_loss

            # ---- inset suppression (positives only) ----
            if lambda_inset > 0.0 and pos_img_mask.any():
                inset_loss = inset_suppression_max(
                    pred_probs[pos_img_mask],                 # [Bp,4,H,W]
                    gt_xy[pos_img_mask],                      # [Bp,4,2]
                    valid_corner_mask(gt_xy[pos_img_mask]),   # [Bp,4] bool
                    frac=inset_frac,
                    radius=inset_radius,
                )
                loss = loss + lambda_inset * inset_loss

            # outward corridors (fights page-frame/header snaps)
            if lambda_outset > 0.0 and pos_img_mask.any():
                outset_loss = outset_suppression_max(
                    pred_probs[pos_img_mask],
                    gt_xy[pos_img_mask],
                    valid_corner_mask(gt_xy[pos_img_mask]),
                    frac=inset_frac, radius=inset_radius
                )
                loss = loss + lambda_outset * outset_loss

            # ---- negatives-only max-peak suppression ----
            neg_img_mask = (gt_hm.sum(dim=(1, 2, 3)) == 0)  # [B]
            if lambda_negmax > 0.0 and neg_img_mask.any():
                neg_probs = pred_probs[neg_img_mask]             # [Bn,4,H,W]
                neg_max = neg_probs.amax(dim=(-1, -2)).mean()    # mean of per-channel maxima
                loss = loss + lambda_negmax * neg_max

            # backward
            loss.backward()
            opt.step()

            bs = inp.size(0)
            seen += bs
            run_loss += loss.item() * bs
            pbar.set_postfix(loss=f"{(run_loss/seen):.4f}", tau=f"{tau_cur:.3f}")

        tr_loss = run_loss / max(1, seen)

        # Eval (real val if provided; else small synth eval)
        if loader_va is not None:
            eval_loader = loader_va
        else:
            eval_loader = DataLoader(
                SyntheticIterable(
                    img_size=img_size, sigma=sigma, difficulty=synth_difficulty,
                    seed=seed+999, neg_frac=neg_frac, occlusion_prob=occlusion_prob,
                    return_meta=True,                        # IMPORTANT
                ),
                batch_size=min(batch_size, 64),
                num_workers=0,
                collate_fn=collate_corners_batch,           # IMPORTANT
            )

        metrics = evaluate(
            model, eval_loader, device,
            tau_eval=tau_cur,
            use_focal_bce=use_focal_bce, pos_weight=pos_weight,
            focal_alpha=focal_alpha, focal_gamma=focal_gamma,
            lambda_bce=lambda_bce, lambda_ce=lambda_ce,
            max_eval_batches=(None if loader_va is not None else 32)
        )

        # Save best
        current_primary = metrics["loss"] if save_best_by == "val_loss" else metrics["px_err"]
        improved_primary = current_primary < best_metric - 1e-8
        notes = ""
        if improved_primary:
            prev = best_metric
            best_metric = current_primary
            best_metric_ep = ep
            epochs_since_primary_best = 0
            ckpt_path = Path(out_dir) / "best.pt"
            torch.save(
                {"model": model.state_dict(),
                "meta": {"img_size": img_size, "sigma": sigma, "base": base,
                        "use_coordconv": True, "unet_head": unet_head}},
                ckpt_path
            )
            print(f"💾 Saved new BEST ({save_best_by}) at epoch {ep}: {best_metric:.6f} (Δ={prev - best_metric:+.6f}) → {ckpt_path}")
            notes += "[best_primary] "
        else:
            epochs_since_primary_best += 1
            print(f"• No improvement on {save_best_by} this epoch "
                f"(best@{best_metric_ep}={best_metric:.6f}, no-improve={epochs_since_primary_best}).")

        if metrics["loss"] < best_loss_val - 1e-8:
            prev = best_loss_val
            best_loss_val = metrics["loss"]; best_loss_ep = ep
            path_loss = Path(out_dir) / "best_by_loss.pt"
            torch.save({"model": model.state_dict(),
                        "meta": {"img_size": img_size, "sigma": sigma, "base": base,
                                "use_coordconv": True, "unet_head": unet_head}},
                    path_loss)
            print(f"💾 Saved best_by_loss.pt at epoch {ep}: val_loss {best_loss_val:.6f} (Δ={prev - best_loss_val:+.6f})")
            notes += "[best_loss] "
        if metrics["px_err"] < best_px_val - 1e-8:
            prev = best_px_val
            best_px_val = metrics["px_err"]; best_px_ep = ep
            path_px = Path(out_dir) / "best_by_px.pt"
            torch.save({"model": model.state_dict(),
                        "meta": {"img_size": img_size, "sigma": sigma, "base": base,
                                "use_coordconv": True, "unet_head": unet_head}},
                    path_px)
            print(f"💾 Saved best_by_px.pt at epoch {ep}: px_err {best_px_val:.6f} (Δ={prev - best_px_val:+.6f})")
            notes += "[best_px] "

        # Stall detection (px_err)
        px_history.append(metrics["px_err"])
        if len(px_history) >= stall_window:
            window_vals = px_history[-stall_window:]
            window_best = min(window_vals[:-1]) if len(window_vals) > 1 else float("inf")
            last_val = window_vals[-1]
            improved_enough = (window_best - last_val) >= min_improve_px
            if not improved_enough:
                print(f"⏸️  Stall detected on px_err over last {stall_window} epochs: "
                    f"best={window_best:.3f} → last={last_val:.3f} (min_improve={min_improve_px:.2f})")
                if lr_decays_used < max_lr_decays:
                    old = current_lr()
                    new = max(lr_min, old * lr_decay_factor)
                    if new < old - 1e-12:
                        set_lr(new)
                        lr_decays_used += 1
                        print(f"🔻 LR decay {lr_decays_used}/{max_lr_decays}: {old:.6g} → {new:.6g}")
                        notes += f"[lr_decay {old:.2e}->{new:.2e}] "
                elif (use_focal_bce and (ep >= int(lcoord_bump_after_frac * epochs)) and
                    (lcoord_bumps_used < max_lcoord_bumps)):
                    old_l = lambda_coord
                    lambda_coord = min(lambda_coord_max, lambda_coord * lcoord_bump_factor)
                    lcoord_bumps_used += 1
                    print(f"⬆️  λ_coord bump {lcoord_bumps_used}/{max_lcoord_bumps}: {old_l:.3f} → {lambda_coord:.3f} "
                        f"(triggered after {int(lcoord_bump_after_frac*100)}% of epochs)")
                    notes += f"[lambda_coord {old_l:.3f}->{lambda_coord:.3f}] "
                else:
                    print("ℹ️  Stall detected but no interventions left/applicable this epoch.")

        if epochs_since_primary_best > 0 and epochs_since_primary_best % plateau_patience == 0:
            print(f"⚠️  Plateau: no new best for {epochs_since_primary_best} epochs "
                f"(best@{best_metric_ep} {save_best_by}={best_metric:.6f}). "
                f"Interventions used: lr={lr_decays_used}/{max_lr_decays}, "
                f"λ_bumps={lcoord_bumps_used}/{max_lcoord_bumps}.")

        if (epochs_since_primary_best >= early_stop_after and
            lr_decays_used >= max_lr_decays and
            lcoord_bumps_used >= max_lcoord_bumps):
            print(f"⛔ Early stop at epoch {ep}: no new best for {epochs_since_primary_best} epochs "
                f"and all interventions exhausted (lr decays={lr_decays_used}, λ bumps={lcoord_bumps_used}).")
            with open(csv_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([ep, f"{tr_loss:.6f}", f"{metrics['loss']:.6f}",
                            f"{metrics['px_err']:.6f}",
                            f"{metrics['pck2']:.6f}", f"{metrics['pck3']:.6f}", f"{metrics['pck5']:.6f}", f"{metrics['pck10']:.6f}",
                            f"{metrics['hm_peak']:.6f}", f"{metrics['hm_sharp']:.6f}",
                            f"{current_lr():.6f}", f"{tau_cur:.4f}", f"{lambda_coord:.3f}", notes.strip()])
            break

        # Log CSV
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([ep, f"{tr_loss:.6f}", f"{metrics['loss']:.6f}",
                        f"{metrics['px_err']:.6f}",
                        f"{metrics['pck2']:.6f}", f"{metrics['pck3']:.6f}", f"{metrics['pck5']:.6f}", f"{metrics['pck10']:.6f}",
                        f"{metrics['hm_peak']:.6f}", f"{metrics['hm_sharp']:.6f}",
                        f"{current_lr():.6f}", f"{tau_cur:.4f}", f"{lambda_coord:.3f}", notes.strip()])

        print(f"[ep {ep}] train_loss={tr_loss:.4f} | val_loss={metrics['loss']:.4f} | px_err={metrics['px_err']:.2f} | "
            f"PCK@2={metrics['pck2']:.3f} PCK@3={metrics['pck3']:.3f} PCK@5={metrics['pck5']:.3f} PCK@10={metrics['pck10']:.3f} | "
            f"hm_peak={metrics['hm_peak']:.3f} hm_sharp={metrics['hm_sharp']:.3f} | "
            f"τ={tau_cur:.3f} | lr={current_lr():.2e} | λ_coord={lambda_coord:.3f} | {notes}")

        # Visualize
        if viz_out and (ep % max(1, viz_every) == 0):
            model.eval()
            with torch.no_grad():
                synth_prev_loader = DataLoader(
                    SyntheticIterable(
                        img_size=img_size, sigma=sigma, difficulty=synth_difficulty,
                        seed=seed + 2024 + ep, neg_frac=neg_frac, occlusion_prob=occlusion_prob,
                        return_meta=True,                    # IMPORTANT
                    ),
                    batch_size=min(batch_size, 64),
                    num_workers=0,
                    collate_fn=collate_corners_batch,       # IMPORTANT
                )

                for batch in synth_prev_loader:
                    # Accept either (inp, gt_hm, gt_xy) or (inp, gt_hm, gt_xy, meta)
                    if isinstance(batch, (list, tuple)) and len(batch) == 4:
                        inp, gt_hm, gt_xy, _meta = batch
                    else:
                        inp, gt_hm, gt_xy = batch

                    logits = model(inp.to(device))
                    logits = ensure_chw_4(logits)
                    pr_probs = torch.sigmoid(logits).cpu()
                    pr_xy = spatial_softargmax(pr_probs.to(device), tau=tau_cur).cpu()

                    save_viz_samples(
                        save_dir=str(Path(viz_out) / "synth"),
                        epoch=ep,
                        imgs=inp.cpu(),
                        gt_hms=ensure_chw_4(gt_hm),
                        pred_hms=pr_probs,
                        gt_xy=gt_xy,
                        pr_xy=pr_xy,
                        max_samples=max(4, min(viz_count, 64))
                    )
                    break
                print(f"🖼️  Saved viz panels to {Path(viz_out)/'synth'} [epoch {ep}]")

    print(f"🏁 Training done. Primary best ({save_best_by}) = {best_metric:.6f} @ epoch {best_metric_ep} | "
        f"best val_loss = {best_loss_val:.6f} @ {best_loss_ep} | "
        f"best px_err = {best_px_val:.6f} @ {best_px_ep} | artifacts in {out_dir}")








# =========================================================
# CLI
# =========================================================


# ============================================================================
# Command‑line interface (CLI)
# ----------------------------------------------------------------------------
# The CLI exposes the full training/evaluation pipeline without touching code.
# Arguments are grouped conceptually below; for a quick tour:
#
#   DATA & OUTPUTS
#   ---------------
#   --data                 Path to real dataset root (images/ + labels.jsonl).
#   --out                  Where to store checkpoints, CSV logs and visuals.
#   --resume               Load weights before training (warm‑start).
#
#   OPTIMIZATION
#   -------------
#   --epochs, --batch, --lr   Usual knobs. Small batches keep GPUs happy.
#   --plateau-patience        How many non‑improving epochs before warnings.
#   --save-best-by            Decide what defines "best": val_loss or px_err.
#
#   ARCHITECTURE
#   -------------
#   --base, --unet-head       Channel scaling and whether to use skip‑U‑Net.
#   --img-size, --sigma       Input resolution and target Gaussian sigma.
#
#   LOSSES (weights)
#   -----------------
#   --lambda-bce, --lambda-ce, --lambda-coord
#       Mix the core loss terms. BCE dominates most of the time.
#   --use-focal-bce, --focal-alpha, --focal-gamma, --pos-weight
#       Switch to focal BCE to focus on hard pixels; increase pos_weight if
#       positive Gaussians are rare compared to background.
#
#   TARGETED SUPPRESSIONS
#   ----------------------
#   --lambda-inset, --lambda-outset, --inset-frac, --inset-radius
#       Penalize peaks just inside/outside corners; good against page frames
#       and "first‑cell" artifacts.
#   --lambda-ring
#       Global guardrail against frame snapping; keep small (0.1–0.3).
#   --lambda-negmax
#       Suppress spurious peaks on pure negatives (no grid in the image).
#
#   SYNTHETIC CONTROLS
#   -------------------
#   --use-synth, --synth-difficulty (1..8)
#       Turn on the Sudoku Journey generator and pick a curriculum level.
#       Real‑world mixes can be emulated with 5–7. Difficulty 8 enables
#       heavy photographic effects.
#   --neg-frac, --occlusion-prob
#       Probability of drawing a negative sample or occluding corners.
#
#   DATA MIXING
#   ------------
#   --real-frac
#       Fraction of steps that should use real data (if --data is set); the
#       remainder pulls from the synthetic stream.
#
#   VISUALIZATION & EVAL
#   ---------------------
#   --viz-out, --viz-every, --viz-count
#       Save qualitative panels during training.
#   --tau, --tau-final
#       Soft‑argmax temperature (constant or annealed) used for both training
#       coordinate loss and evaluation.
#
#   REPRODUCIBILITY & EARLY EXIT
#   -----------------------------
#   --seed, --early-stop-after
#
# EXAMPLES (PowerShell)
# ----------------------
# 1) Resume from a checkpoint and train mostly on synthetic D5 with inset/outset:
#    python .\python\vision\train\train_corners.py `
#       --resume runs\level_4_unet_v4h\best.pt `
#       --out runs\level_4_unet_v4i `
#       --use-synth --synth-difficulty 5 `
#       --epochs 4 --steps-per-epoch 300 --batch 96 `
#       --lr 1.5e-4 --img-size 128 --sigma 1.6 `
#       --tau 0.14 --tau-final 0.05 `
#       --use-focal-bce --focal-alpha 0.5 --focal-gamma 2.0 `
#       --pos-weight 16 `
#       --lambda-bce 1.0 --lambda-ce 0.5 --lambda-coord 8.0 `
#       --lambda-inset 0.45 --inset-frac 0.14 --inset-radius 4 `
#       --lambda-outset 0.30 `
#       --unet-head `
#       --save-best-by px_err `
#       --viz-out runs\corners_viz_level_4_unet_v4i `
#       --viz-every 1 --viz-count 16 `
#       --seed 1234 `
#       --neg-frac 0.10 --lambda-negmax 0.30 --occlusion-prob 0.03
#
#    Typical outcomes on 84 real samples after 4 epochs (τ≈0.05):
#    median px_err ≈ 2.7–3.0, mean px_err ≈ 5.0–5.5, PCK@5 ≈ 0.68–0.74.
#
# 2) Lightweight baseline (no focal BCE, no suppressions), pure synth D3:
#    python .\python\vision\train\train_corners.py --use-synth --synth-difficulty 3 `
#       --epochs 2 --steps-per-epoch 200 --batch 96 --lr 3e-4 --img-size 128 `
#       --lambda-bce 1.0 --lambda-ce 0.2 --lambda-coord 2.0 --save-best-by val_loss
#
# Tweak notes:
# - Increase `pos_weight` when using focal BCE to balance tiny Gaussians.
# - Start λ_inset around 0.3–0.6 and λ_outset around 0.2–0.4, then watch
#   the qualitative panels: peaks should become tighter and less attracted to
#   outside frames.
# ============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None, help="Path to dataset root (images/ + labels.jsonl). Optional if using --use-synth")
    parser.add_argument("--out", type=str, default="runs/corners_exp")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=96)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--sigma", type=float, default=1.6)
    parser.add_argument("--base", type=int, default=24)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.48)
    parser.add_argument("--tau-final", type=float, default=None)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--viz-out", type=str, default="runs/corners_viz")
    parser.add_argument("--viz-every", type=int, default=1)
    parser.add_argument("--viz-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--lambda-coord", type=float, default=0.25)
    parser.add_argument("--lambda-bce", type=float, default=1.0)
    parser.add_argument("--lambda-ce", type=float, default=0.0)
    parser.add_argument("--pos-weight", type=float, default=1.0)

    parser.add_argument("--use-focal-bce", action="store_true")
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--unet-head", action="store_true")

    parser.add_argument("--plateau-patience", type=int, default=3)
    parser.add_argument("--debug-shapes-once", action="store_true")

    parser.add_argument("--save-best-by", type=str, default="val_loss", choices=["val_loss", "px_err"])

    # automation
    parser.add_argument("--monitor-metric", type=str, default="px_err", choices=["px_err"])
    parser.add_argument("--stall-window", type=int, default=8)
    parser.add_argument("--min-improve-px", type=float, default=0.25)
    parser.add_argument("--lr-decay-factor", type=float, default=0.5)
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--max-lr-decays", type=int, default=2)
    parser.add_argument("--lcoord-bump-after-frac", type=float, default=0.66)
    parser.add_argument("--lcoord-bump-factor", type=float, default=1.25)
    parser.add_argument("--lambda-coord-max", type=float, default=6.0)
    parser.add_argument("--max-lcoord-bumps", type=int, default=1)
    parser.add_argument("--early-stop-after", type=int, default=24)

    # synthetic controls
    parser.add_argument("--use-synth", action="store_true", help="Use on-the-fly Sudoku Journey generator")
    parser.add_argument("--synth-difficulty", type=int, default=1, help="1..8")
    parser.add_argument("--real-frac", type=float, default=0.0, help="Fraction of steps per epoch taken from real train set (0..1)")
    parser.add_argument("--steps-per-epoch", type=int, default=None, help="Force a fixed number of training steps per epoch")

    # journey extras
    parser.add_argument("--neg-frac", type=float, default=0.03, help="Probability of negative (no grid) samples in synth.")
    parser.add_argument("--occlusion-prob", type=float, default=0.05, help="Probability to occlude corners in synth.")
    
    

    parser.add_argument("--lambda-negmax", type=float, default=0.0, help="Extra penalty weight on max heatmap peak for negative samples (no grid).")


    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt to load model weights from before training")
    


    parser.add_argument("--lambda-inset", type=float, default=0.0,
                    help="Weight for 1-cell-in suppression around each corner (positives only).")
    parser.add_argument("--inset-frac", type=float, default=0.12,
                        help="Fraction of min(grid_w,grid_h) to step inward along diagonal.")
    parser.add_argument("--inset-radius", type=int, default=3,
                        help="Half-window radius for local max around inset point.")

    parser.add_argument("--lambda-outset", type=float, default=0.0,
                        help="Weight for 1-cell-OUT suppression (positives only).")
    parser.add_argument("--lambda-ring", type=float, default=0.0,
                        help="Weight for border ring suppression (positives only).")
    

    parser.add_argument("--lambda-textslab", type=float, default=0.0,
                        help="Weight for suppression of peaks inside header/footer text rectangles (D4+ generator).")

    args = parser.parse_args()

    train(
        data_root=args.data,
        out_dir=args.out,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        img_size=args.img_size,
        sigma=args.sigma,
        base=args.base,
        val_split=args.val_split,
        tau=args.tau,
        tau_final=args.tau_final,
        augment=(not args.no_augment),
        viz_out=args.viz_out,
        viz_every=args.viz_every,
        viz_count=args.viz_count,
        seed=args.seed,
        lambda_coord=args.lambda_coord,
        lambda_bce=args.lambda_bce,
        lambda_ce=args.lambda_ce,
        pos_weight=args.pos_weight,
        use_focal_bce=args.use_focal_bce,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        unet_head=args.unet_head,
        plateau_patience=args.plateau_patience,
        debug_shapes_once=args.debug_shapes_once,
        save_best_by=args.save_best_by,
        monitor_metric=args.monitor_metric,
        stall_window=args.stall_window,
        min_improve_px=args.min_improve_px,
        lr_decay_factor=args.lr_decay_factor,
        lr_min=args.lr_min,
        max_lr_decays=args.max_lr_decays,
        lcoord_bump_after_frac=args.lcoord_bump_after_frac,
        lcoord_bump_factor=args.lcoord_bump_factor,
        lambda_coord_max=args.lambda_coord_max,
        max_lcoord_bumps=args.max_lcoord_bumps,
        early_stop_after=args.early_stop_after,
        use_synth=args.use_synth,
        synth_difficulty=args.synth_difficulty,
        real_frac=args.real_frac,
        steps_per_epoch=args.steps_per_epoch,
        neg_frac=args.neg_frac,
        occlusion_prob=args.occlusion_prob,
        resume=args.resume,
        lambda_negmax=args.lambda_negmax,
        lambda_inset=args.lambda_inset,
        inset_frac=args.inset_frac,
        inset_radius=args.inset_radius,
        lambda_outset=args.lambda_outset,   # NEW
        lambda_ring=args.lambda_ring,       # NEW
        lambda_textslab=args.lambda_textslab,
    )