# =========================================================
# eval_corners.py — Evaluating a Sudoku corner detector
# =========================================================
#
# Why this module exists
# ----------------------
# In training we teach a model to emit four heatmaps (TL, TR, BR, BL). At
# deployment time we need a *reliable and reproducible* way to turn those
# heatmaps into points, compare them against labeled corners, and export
# evidence for inspection. This file is the single place that does that.
#
# What it does
# ------------
# * Loads a checkpoint and rebuilds the exact architecture used at train time
#   (reading architectural hints saved in the checkpoint meta).
# * Reads a small, explicit dataset format (images/ + labels.jsonl), resizes
#   to a unified resolution, and keeps the ground‑truth ordering TL→TR→BR→BL.
# * Runs inference, decodes the four corners from predicted heatmaps using a
#   soft‑argmax with a safety fallback to discrete argmax when peaks are flat.
# * Writes per‑image CSV metrics, saves qualitative overlays (and optional
#   heatmap panels), and prints an aggregate summary.
#
# How to navigate the file
# ------------------------
# 1) Utilities:
#    - tiny geometry/numpy helpers and the soft‑argmax decoder.
# 2) Dataset:
#    - thin reader for root/{images/, labels.jsonl} with (x,y)×4 corners.
# 3) Model:
#    - the same TinyCornerNet used in training (kept here to avoid imports).
# 4) Visualization & Metrics:
#    - overlay/panel writers and PCK/px‑error summaries.
# 5) main():
#    - the evaluation pipeline + CLI argument parsing.
#
# A note on style
# ---------------
# The comments here tell a story rather than restating the obvious. Each
# function’s header explains the purpose and *how to read* the function; key
# blocks inside the function are annotated with intent.
# =========================================================

from __future__ import annotations

# =========================================================
# Imports — what we pull in and why
# =========================================================
# os/json/csv/Path/argparse  : file system, JSON lines, CSV exports, CLI
# typing                     : light type hints for readability
# numpy, cv2                 : image I/O, color transforms, geometry
# torch, nn, DataLoader      : model definition and batched inference
# tqdm                       : progress bar for friendly feedback
# torch.nn.functional as F   : pooling and softmax used by decoders

import os, json, csv, argparse
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random

import torch.nn.functional as F

# =========================================================
# Small utilities
# =========================================================


def ensure_dir(p: str | Path) -> None:
    """Create directory *p* (and parents) if it does not already exist.
    Convenience wrapper to keep the calling code noise‑free.
    """
    Path(p).mkdir(parents=True, exist_ok=True)


def order_corners_tl_tr_br_bl(corners_np: np.ndarray) -> np.ndarray:
    """Return corners ordered TL→TR→BR→BL.

    Why: The dataset may provide corners in any order. Many parts of the
    pipeline (heatmap channel assignment, color coding, metrics) assume a
    consistent order. We sort by y to split top/bottom pairs, then by x within
    each pair.
    """
    c = np.asarray(corners_np, dtype=np.float32)
    idx_y = np.argsort(c[:, 1])              # sort by y → top two, bottom two
    top = c[idx_y[:2]]; bot = c[idx_y[2:]]
    top = top[np.argsort(top[:, 0])]; bot = bot[np.argsort(bot[:, 0])]
    tl, tr = top[0], top[1]
    bl, br = bot[0], bot[1]
    return np.stack([tl, tr, br, bl], axis=0)


def _normalize_hm_layout_4d(x: torch.Tensor, name: str) -> torch.Tensor:
    """Ensure a 4‑D tensor has the corner *channel* in dim=1.

    Accept shapes like [B,4,H,W] (ideal) or [B,H,W,4] (NHWC) and permute as
    needed. Clear error messages when the channel cannot be located.
    """
    if x.dim() != 4:
        raise RuntimeError(f"{name}: expected 4D, got {x.dim()}D shape={tuple(x.shape)}")
    B, d1, d2, d3 = x.shape
    if d1 == 4:  # already [B,4,H,W]
        return x
    last3 = [d1, d2, d3]
    if 4 in last3:
        idx = last3.index(4)                        # where the channel lives now
        others = [i + 1 for i in range(3) if i != idx]
        perm = [0, idx + 1, others[0], others[1]]   # bring it to dim=1
        return x.permute(*perm).contiguous()
    raise RuntimeError(f"{name}: cannot find channel=4 in {tuple(x.shape)}")


def ensure_chw_4(t: torch.Tensor, name: str = "tensor") -> torch.Tensor:
    """Accept [B,4,H,W] or [4,H,W] and return [B,4,H,W]."""
    if t.dim() == 4:
        return _normalize_hm_layout_4d(t, name)
    if t.dim() == 3:
        return _normalize_hm_layout_4d(t.unsqueeze(0), name + "(3D->4D)").squeeze(0)
    raise RuntimeError(f"{name}: expected 3D/4D tensor, got {t.dim()}D {tuple(t.shape)}")


def to_grayscale_01(img_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR uint8 → grayscale float in [0,1]."""
    if img_bgr.ndim == 2:
        g = img_bgr
    else:
        g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return g.astype(np.float32) / 255.0


# ---- heatmap → point decoders ---------------------------------------------

def spatial_softargmax(hm: torch.Tensor, tau: float = 0.6) -> torch.Tensor:
    """Spatial soft‑argmax over each channel.

    Inputs
    ------
    hm  : [B,4,H,W] *probabilities* (already passed through sigmoid).
    tau : temperature. Lower → peaky distributions, higher → smoother.

    How it works
    ------------
    1) Flatten each heatmap to a categorical distribution with temperature τ.
    2) Compute the expected x/y under that distribution.

    Note: we create a meshgrid with indexing="xy" and then accumulate carefully
    so x corresponds to columns (width) and y to rows (height).
    """
    B, C, H, W = hm.shape
    flat = hm.view(B, C, -1) / max(tau, 1e-6)      # temperature scaling
    prob = torch.softmax(flat, dim=-1)             # [B,4,H*W]
    xs = torch.linspace(0, W - 1, W, device=hm.device)
    ys = torch.linspace(0, H - 1, H, device=hm.device)
    Y, X = torch.meshgrid(xs, ys, indexing="xy")  # Y=x, X=y named for clarity
    X = X.reshape(-1); Y = Y.reshape(-1)
    x = torch.sum(prob * Y, dim=-1)                # expected column
    y = torch.sum(prob * X, dim=-1)                # expected row
    return torch.stack([x, y], dim=-1)             # [B,4,2]


def softargmax_with_fallback(
    pred_hm: torch.Tensor,
    tau: float = 0.10,
    sharp_thresh: float = 0.02,
) -> torch.Tensor:
    """Decode points with a safety net.

    Why: On some real images a channel can become *flat-ish* (no crisp peak):
    soft‑argmax will then drift between modes. Here we measure peak sharpness
    as (peak − local‑avg‑around‑peak). If it’s below *sharp_thresh*, we trust a
    plain argmax for that channel; otherwise we keep the soft‑argmax result.

    How to read the code
    --------------------
    • First we compute the soft prediction.
    • Then we estimate sharpness via a 3×3 average around the peak.
    • Finally we blend per channel between (argmax) and (soft) based on the
      sharpness test.
    """
    B, C, H, W = pred_hm.shape

    # 1) soft prediction
    pr_soft = spatial_softargmax(pred_hm, tau=tau)  # [B,4,2]

    # 2) peak sharpness: (value at peak) − (3×3 neighbor average at the peak)
    pad = (1, 1, 1, 1)
    avg = F.avg_pool2d(F.pad(pred_hm, pad, mode="replicate"), kernel_size=3, stride=1)
    flat = pred_hm.view(B, C, -1)
    idx = torch.argmax(flat, dim=-1)                        # [B,4] linear indices
    mx = torch.gather(flat, -1, idx.unsqueeze(-1)).squeeze(-1)               # [B,4]
    av = torch.gather(avg.view(B, C, -1), -1, idx.unsqueeze(-1)).squeeze(-1) # [B,4]
    sharp = mx - av

    # 3) hard argmax coordinates (y = idx//W, x = idx%W)
    y = (idx // W).float(); x = (idx % W).float()
    pr_arg = torch.stack([x, y], dim=-1)

    use_arg = (sharp < sharp_thresh).unsqueeze(-1)          # select where flat
    return torch.where(use_arg, pr_arg, pr_soft)


def pck(pred_xy: torch.Tensor, gt_xy: torch.Tensor, thresh: float) -> float:
    """Percentage of Correct Keypoints within *thresh* pixels.
    Numbers close to 1.0 mean the model is placing points within that radius
    almost all the time; we report PCK@{2,3,5,10} in the summary.
    """
    d = torch.linalg.norm(pred_xy - gt_xy, dim=-1)
    return (d <= thresh).float().mean().item()


def colorize(hm: np.ndarray) -> np.ndarray:
    """Utility for making a heatmap pretty (JET colormap)."""
    hm8 = np.clip(hm * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(hm8, cv2.COLORMAP_JET)


# =========================================================
# Dataset — small, explicit, dependency‑free
# =========================================================

class CornerEvalDataset(Dataset):
    """Labeled dataset reader for evaluation.

    Format
    ------
    root/
      images/           # image files (any OpenCV‑readable format)
      labels.jsonl      # one JSON object per line with fields:
                        #   file_name: string (relative to images/)
                        #   corners  : [[x,y]×4] in *original* pixel coords
                        #   mode     : optional string for per‑mode stats

    What it returns
    ---------------
    • a 1×H×W grayscale tensor in [0,1]
    • the ground‑truth corners (TL→TR→BR→BL) *rescaled* to the eval resolution
    • a small meta dict (file name, mode, original H×W) used for reporting
    """

    def __init__(self, root: str, img_size: int = 128):
        self.root = Path(root)
        self.img_dir = self.root / "images"
        self.labels_path = self.root / "labels.jsonl"
        self.img_size = int(img_size)
        assert self.img_dir.is_dir(), f"Missing images dir: {self.img_dir}"
        assert self.labels_path.is_file(), f"Missing labels: {self.labels_path}"

        # Parse JSONL into a compact list of records we can index quickly.
        self.records: List[Dict[str, object]] = []
        with open(self.labels_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                # Be permissive about the image key; keep only well‑formed rows.
                fname = (
                    rec.get("file_name")
                    or rec.get("file")
                    or rec.get("image")
                    or rec.get("path")
                )
                if fname is None or "corners" not in rec:
                    continue
                corners = np.array(rec["corners"], dtype=np.float32)
                if corners.shape != (4, 2):
                    continue
                mode = rec.get("mode", None)
                self.records.append({"file_name": fname, "corners": corners, "mode": mode})
        assert len(self.records) > 0, "No valid labeled records found."

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        # --- load image and corners from disk
        rec = self.records[idx]
        fp = self.img_dir / rec["file_name"]
        img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(fp)

        H0, W0 = img.shape[:2]
        corners = order_corners_tl_tr_br_bl(rec["corners"])  # keep canonical order

        # --- resize image and scale corners to the new resolution
        img_rs = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        sx = self.img_size / W0; sy = self.img_size / H0
        gt_xy = corners.copy(); gt_xy[:, 0] *= sx; gt_xy[:, 1] *= sy

        # --- grayscale input in [0,1] with a channels‑first shape
        g = to_grayscale_01(img_rs)
        inp = np.expand_dims(g, 0).astype(np.float32)  # [1,H,W]

        # --- small meta used by the writer
        meta = {"file_name": rec["file_name"], "mode": rec.get("mode", None), "orig_hw": (H0, W0)}
        return torch.from_numpy(inp), torch.from_numpy(gt_xy), meta


# =========================================================
# Model — same TinyCornerNet used in training
# =========================================================

class TinyCornerNet(nn.Module):
    """Compact CNN that produces 4 heatmaps at the input resolution.

    Why here and not imported: Keeping the definition local avoids import
    tight‑coupling and lets eval work as a single file. Architectural choices
    must match training; we read them from the checkpoint meta below.

    Structure
    ---------
    • Encoder: 4 strided conv blocks (down to 1/16 resolution).
    • Decoder: either a plain deconv stack or a U‑Net‑like head with skips.
    • Head   : 1×1 conv → 4 logits (one per corner channel).
    """

    def __init__(self, in_ch_gray: int = 1, out_ch: int = 4, base: int = 24,
                 use_coordconv: bool = True, unet_head: bool = False):
        super().__init__()
        self.use_coordconv = use_coordconv
        self.unet_head = unet_head
        eff_in = in_ch_gray + (2 if use_coordconv else 0)   # +2 for (x,y) coords
        c1, c2, c3, c4 = base, base * 2, base * 3, base * 4

        # --- Encoder (strided conv + BN + ReLU)
        self.enc1 = nn.Sequential(nn.Conv2d(eff_in, c1, 3, 2, 1), nn.BatchNorm2d(c1), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(nn.Conv2d(c1, c2, 3, 2, 1), nn.BatchNorm2d(c2), nn.ReLU(inplace=True))
        self.enc3 = nn.Sequential(nn.Conv2d(c2, c3, 3, 2, 1), nn.BatchNorm2d(c3), nn.ReLU(inplace=True))
        self.enc4 = nn.Sequential(nn.Conv2d(c3, c4, 3, 2, 1), nn.BatchNorm2d(c4), nn.ReLU(inplace=True))

        # --- Decoder (two flavors)
        if unet_head:
            # U‑Net variant with skip connections (better spatial fidelity)
            self.up4 = nn.Sequential(nn.ConvTranspose2d(c4, c3, 2, 2), nn.ReLU(inplace=True))
            self.up3 = nn.Sequential(nn.ConvTranspose2d(c3 * 2, c2, 2, 2), nn.ReLU(inplace=True))
            self.up2 = nn.Sequential(nn.ConvTranspose2d(c2 * 2, c1, 2, 2), nn.ReLU(inplace=True))
            self.up1 = nn.Sequential(nn.ConvTranspose2d(c1 * 2, c1, 2, 2), nn.ReLU(inplace=True))
            self.head = nn.Conv2d(c1, out_ch, 1)
        else:
            # Plain deconvolutional ladder back to full resolution
            self.up = nn.Sequential(
                nn.ConvTranspose2d(c4, c3, 2, 2), nn.ReLU(inplace=True),
                nn.ConvTranspose2d(c3, c2, 2, 2), nn.ReLU(inplace=True),
                nn.ConvTranspose2d(c2, c1, 2, 2), nn.ReLU(inplace=True),
                nn.ConvTranspose2d(c1, c1, 2, 2), nn.ReLU(inplace=True),
            )
            self.head = nn.Conv2d(c1, out_ch, 1)

    @staticmethod
    def _coord_channels(B: int, H: int, W: int, device):
        # Normalized [0,1] coordinate maps help the network localize.
        xs = torch.linspace(0.0, 1.0, W, device=device).view(1, 1, 1, W).expand(B, -1, H, -1)
        ys = torch.linspace(0.0, 1.0, H, device=device).view(1, 1, H, 1).expand(B, -1, -1, W)
        return xs, ys

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Optional CoordConv: concatenate (x,y) maps to the grayscale image.
        if self.use_coordconv:
            B, _, H, W = x.shape
            xs, ys = self._coord_channels(B, H, W, x.device)
            x = torch.cat([x, xs, ys], dim=1)
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
# Visualization helpers
# =========================================================

def save_overlay(out_path: Path, img_gray01: np.ndarray, gt_xy: np.ndarray, pr_xy: np.ndarray) -> None:
    """Save a side‑by‑side overlay of ground truth (green) and prediction (red).
    This is the quickest way to *see* what the model is doing.
    """
    g = (img_gray01 * 255.0).astype(np.uint8)
    bgr = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    for (x, y) in gt_xy:
        cv2.circle(bgr, (int(round(x)), int(round(y))), 3, (0, 255, 0), -1, lineType=cv2.LINE_AA)
    for (x, y) in pr_xy:
        cv2.circle(bgr, (int(round(x)), int(round(y))), 3, (0, 0, 255), -1, lineType=cv2.LINE_AA)
    cv2.imwrite(str(out_path), bgr)


def save_panel_with_heatmaps(
    out_path: Path,
    img_gray01: np.ndarray,
    gt_xy: np.ndarray,
    pr_xy: np.ndarray,
    pred_hm: np.ndarray,
) -> None:
    """Save a 2×2 panel with the four heatmaps next to the image overlay.
    Use `--save-heatmaps` to enable this heavier visualization.
    """
    g = (img_gray01 * 255.0).astype(np.uint8)
    left = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    for (x, y) in gt_xy:
        cv2.circle(left, (int(round(x)), int(round(y))), 3, (0, 255, 0), -1, lineType=cv2.LINE_AA)
    for (x, y) in pr_xy:
        cv2.circle(left, (int(round(x)), int(round(y))), 3, (0, 0, 255), -1, lineType=cv2.LINE_AA)
    left = cv2.resize(left, (256, 256), interpolation=cv2.INTER_NEAREST)

    tiles = []
    for j in range(4):
        cimg = colorize(pred_hm[j])
        cimg = cv2.resize(cimg, (128, 128), interpolation=cv2.INTER_NEAREST)
        tiles.append(cimg)
    right = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    cv2.imwrite(str(out_path), np.hstack([left, right]))


# =========================================================
# Metrics
# =========================================================

def compute_errors(pr_xy: torch.Tensor, gt_xy: torch.Tensor) -> np.ndarray:
    """Per‑image per‑corner Euclidean distances in pixels (numpy array)."""
    d = torch.linalg.norm(pr_xy - gt_xy, dim=-1)
    return d.cpu().numpy()


def summarize(errors_px: np.ndarray, pred_xy: torch.Tensor, gt_xy: torch.Tensor) -> Dict[str, float]:
    """Aggregate summary (mean/median/std and PCK@2/3/5/10)."""
    flat = errors_px.reshape(-1)
    return {
        "px_mean": float(flat.mean()),
        "px_median": float(np.median(flat)),
        "px_std": float(flat.std(ddof=0)),
        "pck2": pck(pred_xy, gt_xy, 2.0),
        "pck3": pck(pred_xy, gt_xy, 3.0),
        "pck5": pck(pred_xy, gt_xy, 5.0),
        "pck10": pck(pred_xy, gt_xy, 10.0),
    }


def per_mode_summary(
    modes: List[Optional[str]],
    errors_px: np.ndarray,
    pred_xy: torch.Tensor,
    gt_xy: torch.Tensor,
) -> Dict[str, Dict[str, float]]:
    """Break the summary out by the optional `mode` field in labels.jsonl.
    Useful to see, for example, `curved_inside` vs `straight_strict` behavior.
    """
    out: Dict[str, Dict[str, float]] = {}
    modes_arr = np.array([m if m is not None else "(none)" for m in modes])
    for m in sorted(set(modes_arr.tolist())):
        idx = np.where(modes_arr == m)[0]
        if len(idx) == 0:
            continue
        out[m] = summarize(errors_px[idx], pred_xy[idx], gt_xy[idx])
    return out


# =========================================================
# Evaluation pipeline
# =========================================================

def main() -> None:
    # -----------------------------------------------------
    # CLI
    # -----------------------------------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Dataset root (images/ + labels.jsonl)")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to a trained checkpoint .pt")
    ap.add_argument("--viz-out", type=str, default="runs/corners_eval_viz", help="Where to save overlays/panels")
    ap.add_argument("--csv-out", type=str, default="runs/corners_eval_metrics.csv", help="Per‑image metrics CSV path")
    ap.add_argument("--batch", type=int, default=96, help="Batch size for inference")
    ap.add_argument("--tau", type=float, default=0.18, help="Soft‑argmax temperature (see docs below)")
    ap.add_argument("--limit", type=int, default=None, help="Evaluate only the first N samples (for quick checks)")
    ap.add_argument("--save-heatmaps", action="store_true", help="Also save 2×2 heatmap panels per image")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load checkpoint + rebuild the *exact* architecture used at train time
    ckpt = torch.load(args.ckpt, map_location=device)
    if "model" not in ckpt:
        raise RuntimeError(f"Checkpoint {args.ckpt} has no 'model' state_dict")
    meta = ckpt.get("meta", {})
    print("Loaded ckpt meta:", meta)

    img_size = int(meta.get("img_size", 128))
    base = int(meta.get("base", 24))
    use_coordconv = bool(meta.get("use_coordconv", True))
    unet_head = bool(meta.get("unet_head", False))

    model = TinyCornerNet(
        in_ch_gray=1,
        out_ch=4,
        base=base,
        use_coordconv=use_coordconv,
        unet_head=unet_head,
    ).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    if missing or unexpected:
        print("⚠️  Mismatch between ckpt and eval architecture. Double‑check base/use_coordconv/unet_head.")

    model.eval()

    # 2) Dataset/loader (optionally trim with --limit for quick iterations)
    ds_full = CornerEvalDataset(args.data, img_size=img_size)
    if args.limit is not None:
        ds = torch.utils.data.Subset(ds_full, list(range(min(args.limit, len(ds_full)))))
    else:
        ds = ds_full
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    # 3) Prepare outputs (folders + CSV header)
    ensure_dir(args.viz_out)
    ensure_dir(Path(args.csv_out).parent)

    csv_path = Path(args.csv_out)
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(
            [
                "file_name",
                "mode",
                "err_tl",
                "err_tr",
                "err_br",
                "err_bl",
                "err_mean",
                "pck3_img",
                "pck5_img",
                "pck10_img",
            ]
        )

    # 4) Inference loop with decoding and visualization
    all_errs: List[np.ndarray] = []
    all_gt: List[torch.Tensor] = []
    all_pr: List[torch.Tensor] = []
    all_modes: List[Optional[str]] = []
    first_batch_stats_printed = False

    with torch.no_grad():
        for (inp, gt_xy, meta_batch) in tqdm(loader, desc="Eval"):
            inp = inp.to(device)              # [B,1,H,W]
            gt_xy = gt_xy.to(device).float()  # [B,4,2]

            logits = model(inp)               # [B,4,H,W] — raw logits
            logits = ensure_chw_4(logits, "logits(eval)")
            pred_hm = torch.sigmoid(logits)   # turn into probabilities

            if not first_batch_stats_printed:
                stats = [(float(pred_hm[:, c].mean()), float(pred_hm[:, c].std())) for c in range(4)]
                print("pred_hm mean/std per channel:", stats)
                first_batch_stats_printed = True

            # Soft‑argmax with a guard for flat maps (see function above)
            pr_xy = softargmax_with_fallback(pred_hm, tau=args.tau, sharp_thresh=0.02)

            # Per‑image per‑corner errors (for CSV) and stash for summary
            errs = torch.linalg.norm(pr_xy - gt_xy, dim=-1)  # [B,4]
            errs_np = errs.cpu().numpy()
            all_errs.append(errs_np)
            all_gt.append(gt_xy.cpu())
            all_pr.append(pr_xy.cpu())

            # Write CSV row + save visuals for each image in the batch
            B = inp.size(0)
            for i in range(B):
                file_name = meta_batch["file_name"][i]
                mode = meta_batch["mode"][i] if "mode" in meta_batch else None
                all_modes.append(mode)

                d = errs[i]
                p3 = float((d <= 3.0).float().mean().item())
                p5 = float((d <= 5.0).float().mean().item())
                p10 = float((d <= 10.0).float().mean().item())

                with open(csv_path, "a", newline="") as f:
                    csv.writer(f).writerow(
                        [
                            file_name,
                            mode,
                            f"{float(d[0]):.4f}",
                            f"{float(d[1]):.4f}",
                            f"{float(d[2]):.4f}",
                            f"{float(d[3]):.4f}",
                            f"{float(d.mean().item()):.4f}",
                            f"{p3:.3f}",
                            f"{p5:.3f}",
                            f"{p10:.3f}",
                        ]
                    )

                g = inp[i, 0].cpu().numpy()
                gt = gt_xy[i].cpu().numpy()
                pr = pr_xy[i].cpu().numpy()
                save_overlay(Path(args.viz_out) / (Path(file_name).stem + "_overlay.png"), g, gt, pr)
                if args.save_heatmaps:
                    save_panel_with_heatmaps(
                        Path(args.viz_out) / (Path(file_name).stem + "_panel.png"),
                        g,
                        gt,
                        pr,
                        pred_hm[i].cpu().numpy(),
                    )

    # 5) Aggregate results and print a compact summary
    E = np.concatenate(all_errs, axis=0)
    gt_xy_all = torch.cat(all_gt, dim=0)
    pr_xy_all = torch.cat(all_pr, dim=0)

    overall = summarize(E, pr_xy_all, gt_xy_all)
    per_corner = {
        **{
            f"pck3_c{j}": float(
                (torch.linalg.norm(pr_xy_all[:, j] - gt_xy_all[:, j], dim=-1) <= 3.0)
                .float()
                .mean()
                .item()
            )
            for j in range(4)
        },
        **{
            f"pck5_c{j}": float(
                (torch.linalg.norm(pr_xy_all[:, j] - gt_xy_all[:, j], dim=-1) <= 5.0)
                .float()
                .mean()
                .item()
            )
            for j in range(4)
        },
    }
    mode_break = per_mode_summary(all_modes, E, pr_xy_all, gt_xy_all)

    print("\n================ Summary ================")
    print(f"Samples evaluated: {len(E)}")
    print(f"Mean px error:  {overall['px_mean']:.3f}")
    print(f"Median px err:  {overall['px_median']:.3f}")
    print(f"Std px err:     {overall['px_std']:.3f}")
    print(
        f"PCK@2: {overall['pck2']:.3f} | PCK@3: {overall['pck3']:.3f} | "
        f"PCK@5: {overall['pck5']:.3f} | PCK@10: {overall['pck10']:.3f}"
    )
    print("Per-corner PCKs:")
    print("  " + "  ".join([f"{k}: {v:.3f}" for k, v in per_corner.items() if k.startswith("pck3")]))
    print("  " + "  ".join([f"{k}: {v:.3f}" for k, v in per_corner.items() if k.startswith("pck5")]))
    if len(mode_break) > 0:
        print("\nPer-mode breakdown:")
        for m, s in mode_break.items():
            print(
                f"  {m:16s}  mean={s['px_mean']:.3f}  med={s['px_median']:.3f}  "
                f"PCK@3={s['pck3']:.3f}  PCK@5={s['pck5']:.3f}  PCK@10={s['pck10']:.3f}"
            )
    print(f"\nPer-image CSV: {args.csv_out}")
    print("Overlays saved under:", str(Path(args.viz_out).resolve()))
    print("=========================================\n")


# =========================================================
# CLI
# =========================================================

# ============================================================================
# Command‑line interface (CLI)
# ----------------------------------------------------------------------------
# The CLI exposes the full evaluation pipeline without touching code. Arguments
# are grouped conceptually below; skim the groups, then peek at the examples.
#
#   DATA & CHECKPOINT
#   ------------------
#   --data                 Path to real dataset root (images/ + labels.jsonl).
#   --ckpt                 Path to a trained checkpoint (.pt) to evaluate.
#
#   INFERENCE & DECODING
#   ---------------------
#   --batch                Batch size during inference; 64–128 is typical.
#   --tau                  Soft‑argmax temperature. Smaller → sharper but may
#                          snap to noise; larger → smoother but slightly biased.
#                          We also guard with an argmax fallback when peaks are
#                          too flat (see `softargmax_with_fallback`).
#   --limit                Evaluate only the first N samples (quick smoke test).
#
#   OUTPUTS
#   -------
#   --viz-out              Directory for image overlays (and panels below).
#   --csv-out              Per‑image metrics CSV path.
#   --save-heatmaps        Also write 2×2 heatmap panels (heavier but useful).
#
# EXAMPLES (PowerShell)
# ----------------------
# 1) Evaluate the latest training run on the 84‑image real set (overlays + CSV):
#    python .\python\vision\train\eval_corners.py `
#       --data .\datasets\sudoku_corners_real `
#       --ckpt .\runs\level_4_unet_v4i\best.pt `
#       --viz-out .\runs\eval_unet_v4i `
#       --csv-out .\runs\eval_unet_v4i\metrics.csv `
#       --batch 96 --tau 0.05 --save-heatmaps
#
#    Typical outcomes for a strong model (from recent runs):
#    median px_err ≈ 2.7–3.0, mean px_err ≈ 5.0–5.5, PCK@5 ≈ 0.68–0.74.
#
# 2) Quick sanity check on just 12 images (faster, no panels):
#    python .\python\vision\train\eval_corners.py `
#       --data .\datasets\sudoku_corners_real `
#       --ckpt .\runs\level_4_unet_v4i\best.pt `
#       --viz-out .\runs\eval_unet_v4i_quick `
#       --csv-out .\runs\eval_unet_v4i_quick\metrics.csv `
#       --batch 64 --tau 0.10 --limit 12
#
# Tweak notes:
# - If overlays show red points drifting when the heatmap looks flat, try a
#   slightly lower --tau (e.g., 0.05–0.10). The fallback already helps a lot.
# - Keep --save-heatmaps on when you’re tuning the decoder; it’s the best
#   way to *see* why the numbers move.
# ============================================================================


if __name__ == "__main__":
    main()
