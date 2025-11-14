import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.utils.data import ConcatDataset

# ---- robust import: works as module or as script
try:
    from ..models.unet_lite import UNetLite  # when run with: python -m python.vision.train.train_intersections
except Exception:
    # allow running as a loose script: python python/vision/train/train_intersections.py
    import sys as _sys
    _ROOT = Path(__file__).resolve().parents[2]  # repo root (folder containing "python/")
    if str(_ROOT) not in _sys.path:
        _sys.path.append(str(_ROOT))
    from python.vision.models.unet_lite import UNetLite  # noqa: E402


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def draw_points_bgr(im_g: np.ndarray, pts_yx: np.ndarray, color=(0, 0, 255), r=2):
    """Draw small circles at [y,x] on a gray image and return BGR."""
    vis = cv2.cvtColor(im_g, cv2.COLOR_GRAY2BGR)
    for (y, x) in pts_yx:
        cv2.circle(vis, (int(round(x)), int(round(y))), r, color, -1, lineType=cv2.LINE_AA)
    return vis

def save_viz_peaks(out_dir: Path, img: torch.Tensor, pred_logits: torch.Tensor, json_path: str, tag: str,
                   thr_pred: float = 0.2, topk: int = 120):
    """
    Save two extra images:
      - *_pred_peaks.png  : predicted peaks (red) over input
      - *_pred_vs_gt.png  : GT points (green) + predicted peaks (red)
    """
    ensure_dir(out_dir)
    im = (img.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
    ph = torch.sigmoid(pred_logits).squeeze(0).squeeze(0).cpu().numpy()

    # Pred peaks via NMS
    pred_xy, _ = _local_maxima(ph, thr=thr_pred, pool=3, topk=topk)  # (K,2) [x,y]
    if pred_xy.size == 0:
        pred_yx = np.zeros((0, 2), dtype=np.float32)
    else:
        pred_yx = np.stack([pred_xy[:, 1], pred_xy[:, 0]], axis=1)  # to [y,x]

    # GT points from JSON
    jp = Path(json_path).with_suffix(".json")
    gt = []
    if jp.exists():
        data = json.loads(jp.read_text())
        gt = np.array(data.get("points", []), dtype=np.float32)  # [y,x]
    gt = gt if gt is not None and gt.size else np.zeros((0, 2), dtype=np.float32)

    # Save: predicted peaks only
    vis_pred = draw_points_bgr(im, pred_yx, color=(0, 0, 255), r=2)
    cv2.imwrite(str(out_dir / f"{tag}_pred_peaks.png"), vis_pred)

    # Save: GT (green) + Pred (red)
    vis_both = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    for (y, x) in gt:
        cv2.circle(vis_both, (int(round(x)), int(round(y))), 2, (0, 255, 0), -1, lineType=cv2.LINE_AA)
    for (y, x) in pred_yx:
        cv2.circle(vis_both, (int(round(x)), int(round(y))), 2, (0, 0, 255), -1, lineType=cv2.LINE_AA)
    cv2.imwrite(str(out_dir / f"{tag}_pred_vs_gt.png"), vis_both)


def resize_with_points(img: np.ndarray, pts_yx: np.ndarray, out_size: int) -> Tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    scale_y = out_size / h
    scale_x = out_size / w
    img_r = cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_AREA)
    pts_r = pts_yx.copy()
    if pts_r.size:
        pts_r[:, 0] = pts_r[:, 0] * scale_y
        pts_r[:, 1] = pts_r[:, 1] * scale_x
    return img_r, pts_r


def rasterize_heatmap(size: int, pts_yx: np.ndarray, sigma: float) -> np.ndarray:
    """ Sum-of-Gaussians heatmap, clipped to [0,1]. pts_yx is (N,2) [y,x]. """
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
    hm = np.zeros((size, size), dtype=np.float32)
    if pts_yx.size == 0:
        return hm
    for (y0, x0) in pts_yx:
        g = np.exp(-((yy - y0) ** 2 + (xx - x0) ** 2) / (2.0 * sigma ** 2))
        hm += g.astype(np.float32)
    hm = np.clip(hm, 0.0, 1.0)
    return hm


class IntersectionDataset(Dataset):
    """
    Folder with:
      - images (*.png/jpg/jpeg)
      - same-name JSON annotations with {"points": [[y,x], ...]}.
    """
    def __init__(self, root: str, img_size: int = 128, sigma: float = 1.8):
        self.root = Path(root)
        self.img_size = img_size
        self.sigma = sigma
        exts = {".png", ".jpg", ".jpeg"}
        self.images: List[Path] = sorted([p for p in self.root.rglob("*") if p.suffix.lower() in exts])
        if not self.images:
            raise FileNotFoundError(f"No images under {self.root}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        ip = self.images[idx]
        jp = ip.with_suffix(".json")
        if not jp.exists():
            raise FileNotFoundError(f"Missing JSON next to {ip.name}")

        img = read_gray(ip)
        data = json.loads(jp.read_text())
        pts = np.array(data.get("points", []), dtype=np.float32)  # (N,2) [y,x]

        img_r, pts_r = resize_with_points(img, pts, self.img_size)
        hm = rasterize_heatmap(self.img_size, pts_r, self.sigma)

        # To tensors
        im_t = torch.from_numpy(img_r).unsqueeze(0).float() / 255.0  # (1,H,W)
        hm_t = torch.from_numpy(hm).unsqueeze(0).float()             # (1,H,W)
        return im_t, hm_t, str(ip)


# ---------- metrics helpers (no SciPy dependency) ----------
def _local_maxima(hm: np.ndarray, thr: float, pool: int = 3, topk: Optional[int] = None):
    """
    hm: (H,W) in [0,1]
    Returns XY array of peak coords (K,2) and their scores (K,)
    """
    import torch as _t
    t = _t.from_numpy(hm[None, None])  # 1x1xH xW
    m = F.max_pool2d(t, pool, stride=1, padding=pool // 2)
    keep = (t >= m) & (t > thr)
    ys, xs = _t.where(keep[0, 0])
    vals = t[0, 0, ys, xs]
    # sort by value desc
    order = _t.argsort(vals, descending=True)
    if topk is not None:
        order = order[:topk]
    xs = xs[order].cpu().numpy()
    ys = ys[order].cpu().numpy()
    vals = vals[order].cpu().numpy()
    xy = np.stack([xs, ys], axis=1)
    return xy, vals


def _match_stats_1to1(pred_xy: np.ndarray, gt_xy: np.ndarray, px: int, return_pairs: bool = False):
    """
    One-to-one greedy matching within radius px.
    Returns (tp, fp, fn) by default.
    If return_pairs=True, also returns 'pairs' (list[(pi,gi)]) and 'dists' (list[float, in px]).
    """
    P = pred_xy.shape[0]; G = gt_xy.shape[0]
    if P == 0 and G == 0:
        return (0, 0, 0, [], []) if return_pairs else (0, 0, 0)
    if P == 0:
        return (0, 0, G, [], []) if return_pairs else (0, 0, G)
    if G == 0:
        return (0, P, 0, [], []) if return_pairs else (0, P, 0)

    # pairwise distances (P,G)
    diff = pred_xy[:, None, :] - gt_xy[None, :, :]
    d = np.sqrt((diff * diff).sum(axis=2))

    used_p = np.zeros(P, dtype=bool)
    used_g = np.zeros(G, dtype=bool)
    tp = 0
    big = 1e9
    d_work = d.copy()
    pairs, dists = [], []

    while True:
        i, j = np.unravel_index(np.argmin(d_work), d_work.shape)
        if d_work[i, j] > px:  # smallest left is too far
            break
        if used_p[i] or used_g[j]:
            d_work[i, j] = big
            continue
        used_p[i] = True
        used_g[j] = True
        tp += 1
        pairs.append((i, j))
        dists.append(float(d[i, j]))
        d_work[i, :] = big
        d_work[:, j] = big

    fp = int((~used_p).sum())
    fn = int((~used_g).sum())
    if return_pairs:
        return tp, fp, fn, pairs, dists
    return tp, fp, fn


def _chamfer(pred_xy: np.ndarray, gt_xy: np.ndarray):
    """
    Symmetric Chamfer distance stats (pixels).
    Returns dict with mean/p95 in each direction and symmetric mean.
    """
    if pred_xy.shape[0] == 0 and gt_xy.shape[0] == 0:
        return {"chamfer_mean": 0.0, "chamfer_p95": 0.0, "p2g_mean": 0.0, "g2p_mean": 0.0}
    if pred_xy.shape[0] == 0 or gt_xy.shape[0] == 0:
        infv = float("inf")
        return {"chamfer_mean": infv, "chamfer_p95": infv, "p2g_mean": infv, "g2p_mean": infv}

    diff = pred_xy[:, None, :] - gt_xy[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2) + 1e-9)  # (P,G)
    p2g = d.min(axis=1)  # (P,)
    g2p = d.min(axis=0)  # (G,)
    sym = np.concatenate([p2g, g2p], axis=0)
    return {
        "chamfer_mean": float(sym.mean()),
        "chamfer_p95":  float(np.percentile(sym, 95)),
        "p2g_mean":     float(p2g.mean()),
        "g2p_mean":     float(g2p.mean()),
    }


def compute_metrics_batch(pred_logits: torch.Tensor, gt_hm: torch.Tensor,
                          px_list=(2, 3, 5), thr_pred=0.2, thr_gt=0.5):
    """
    pred_logits, gt_hm: (B,1,H,W)
    Returns dict with PCK@k (1–1 matching), PREC/REC/F1 (px=3), and CHAMFER (px mean, both ways).
    """
    B, _, H, W = pred_logits.shape
    pck_hits = {px: 0 for px in px_list}
    pck_total = 0
    pr_tp = pr_fp = pr_fn = 0
    chamfer_sum = 0.0
    chamfer_count = 0

    pred_hm = torch.sigmoid(pred_logits).cpu().numpy()
    gt_hm = gt_hm.cpu().numpy()

    for b in range(B):
        ph = pred_hm[b, 0]
        gh = gt_hm[b, 0]
        pred_xy, _ = _local_maxima(ph, thr=thr_pred, pool=3, topk=120)  # (K,2) [x,y]
        gt_xy,   _ = _local_maxima(gh, thr=thr_gt, pool=3, topk=120)

        # PCK (1–1)
        pck_total += gt_xy.shape[0]
        for px in px_list:
            tp, _, fn = _match_stats_1to1(pred_xy, gt_xy, px=px)
            # PCK is TPs over #GT
            pck_hits[px] += tp

        # PR/F1 @ px=3 (1–1)
        tp, fp, fn = _match_stats_1to1(pred_xy, gt_xy, px=3)
        pr_tp += tp; pr_fp += fp; pr_fn += fn

        # Symmetric Chamfer distance (mean nearest-neighbor, both directions)
        if pred_xy.shape[0] and gt_xy.shape[0]:
            # forward: pred->gt
            d_fw = np.sqrt(((pred_xy[:, None, :] - gt_xy[None, :, :]) ** 2).sum(axis=2)).min(axis=1).mean()
            # backward: gt->pred
            d_bw = np.sqrt(((gt_xy[:, None, :] - pred_xy[None, :, :]) ** 2).sum(axis=2)).min(axis=1).mean()
            chamfer_sum += 0.5 * (d_fw + d_bw)
            chamfer_count += 1

    metrics = {}
    for px in px_list:
        metrics[f"PCK@{px}"] = (pck_hits[px] / max(pck_total, 1)) if pck_total else 0.0

    prec = pr_tp / max(pr_tp + pr_fp, 1)
    rec  = pr_tp / max(pr_tp + pr_fn, 1)
    f1   = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    metrics.update(dict(PREC=prec, REC=rec, F1=f1))

    if chamfer_count:
        metrics["CHAMFER"] = chamfer_sum / chamfer_count
    else:
        metrics["CHAMFER"] = float("nan")
    return metrics


def _overlay(img_g: np.ndarray, hm: np.ndarray):
    im_bgr = cv2.cvtColor(img_g, cv2.COLOR_GRAY2BGR)
    heat = (hm * 255).astype(np.uint8)
    cm = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    return cv2.addWeighted(im_bgr, 0.6, cm, 0.4, 0)


def save_viz(out_dir: Path, img: torch.Tensor, pred: torch.Tensor, gt: torch.Tensor, tag: str):
    ensure_dir(out_dir)
    im = (img.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
    ph = torch.sigmoid(pred).squeeze(0).squeeze(0).cpu().numpy()
    gh = gt.squeeze(0).squeeze(0).cpu().numpy()
    cv2.imwrite(str(out_dir / f"{tag}_img.png"), im)
    cv2.imwrite(str(out_dir / f"{tag}_pred.png"), _overlay(im, ph))
    cv2.imwrite(str(out_dir / f"{tag}_gt.png"), _overlay(im, gh))


def train_one_epoch(model, loader, opt, device, loss_fn, scaler=None, log_interval=50):
    model.train()
    total = 0.0
    nimg = 0
    t0 = time.time()
    for step, (imgs, hms, _) in enumerate(loader, 1):
        imgs = imgs.to(device, non_blocking=True)
        hms = hms.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        if scaler:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(imgs)
                loss = loss_fn(logits, hms)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            logits = model(imgs)
            loss = loss_fn(logits, hms)
            loss.backward()
            opt.step()

        total += float(loss.detach().cpu())
        nimg += imgs.shape[0]
        if step % log_interval == 0:
            dt = time.time() - t0
            ips = nimg / max(dt, 1e-6)
            lr = opt.param_groups[0]["lr"]
            print(f"  step {step:05d}/{len(loader):05d} | loss={total/step:.4f} | {ips:.1f} img/s | lr={lr:g}")
    avg = total / max(len(loader), 1)
    return avg


def _bbox_scale_offset(pred_xy: np.ndarray, gt_xy: np.ndarray):
    # returns (scale_err, offset_err) in pixels where scale_err is relative
    if pred_xy.shape[0] == 0 or gt_xy.shape[0] == 0:
        return np.nan, np.nan
    px_min, py_min = pred_xy.min(axis=0); px_max, py_max = pred_xy.max(axis=0)
    gx_min, gy_min = gt_xy.min(axis=0);   gx_max, gy_max = gt_xy.max(axis=0)
    pw, ph = max(px_max - px_min, 1e-6), max(py_max - py_min, 1e-6)
    gw, gh = max(gx_max - gx_min, 1e-6), max(gy_max - gy_min, 1e-6)
    scale_err = 0.5 * (abs(pw - gw)/gw + abs(ph - gh)/gh)
    offx = 0.5*((px_min + px_max) - (gx_min + gx_max))
    offy = 0.5*((py_min + py_max) - (gy_min + gy_max))
    offset_err = float(np.hypot(offx, offy))
    return float(scale_err), offset_err

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    loss_fn,
    viz_dir: Optional[Path] = None,
    viz_max: int = 2,
    thr_pred: float = 0.2,
    thr_gt: float = 0.5,
    per_sample_csv: Optional[Path] = None,
):
    model.eval()
    total = 0.0
    saved = 0

    agg_metrics = {"PCK@2": 0.0, "PCK@3": 0.0, "PCK@5": 0.0, "PREC": 0.0, "REC": 0.0, "F1": 0.0, "CHAMFER": 0.0}
    count_batches = 0
    chamfer_batches = 0

    # optional per-sample CSV
    csv_writer = None
    f_csv = None
    if per_sample_csv is not None:
        per_sample_csv.parent.mkdir(parents=True, exist_ok=True)
        f_csv = open(per_sample_csv, "w", newline="")
        csv_writer = csv.writer(f_csv)
        csv_writer.writerow(["file", "n_pred", "n_gt", "PCK@2", "PCK@3", "PCK@5", "PREC", "REC", "F1", "CHAMFER"])

    for imgs, hms, tags in loader:
        imgs = imgs.to(device)
        hms = hms.to(device)
        logits = model(imgs)
        loss = loss_fn(logits, hms)
        total += float(loss.detach().cpu())

        # ---- batch-level aggregate (fast) ----
        m_batch = compute_metrics_batch(logits, hms, px_list=(2, 3, 5), thr_pred=thr_pred, thr_gt=thr_gt)
        for k in ("PCK@2", "PCK@3", "PCK@5", "PREC", "REC", "F1"):
            agg_metrics[k] += float(m_batch[k])
        if not np.isnan(m_batch.get("CHAMFER", np.nan)):
            agg_metrics["CHAMFER"] += float(m_batch["CHAMFER"])
            chamfer_batches += 1
        count_batches += 1

        # ---- per-sample (optional) + viz ----
        B = imgs.size(0)
        for i in range(B):
            if csv_writer is not None:
                # compute single-sample metrics by slicing to batch=1
                m1 = compute_metrics_batch(
                    logits[i:i+1], hms[i:i+1], px_list=(2, 3, 5), thr_pred=thr_pred, thr_gt=thr_gt
                )
                # count points for convenience
                ph = torch.sigmoid(logits[i:i+1])[0,0].cpu().numpy()
                gh = hms[i:i+1][0,0].cpu().numpy()
                pred_xy, _ = _local_maxima(ph, thr=thr_pred, pool=3, topk=120)
                gt_xy, _   = _local_maxima(gh, thr=thr_gt,   pool=3, topk=120)

                # CHAMFER per-sample
                if pred_xy.shape[0] and gt_xy.shape[0]:
                    d_fw = np.sqrt(((pred_xy[:, None, :] - gt_xy[None, :, :])**2).sum(axis=2)).min(axis=1).mean()
                    d_bw = np.sqrt(((gt_xy[:, None, :] - pred_xy[None, :, :])**2).sum(axis=2)).min(axis=1).mean()
                    chamfer = 0.5 * (float(d_fw) + float(d_bw))
                else:
                    chamfer = float("nan")

                csv_writer.writerow([
                    str(tags[i]),
                    int(pred_xy.shape[0]),
                    int(gt_xy.shape[0]),
                    f"{m1['PCK@2']:.6f}",
                    f"{m1['PCK@3']:.6f}",
                    f"{m1['PCK@5']:.6f}",
                    f"{m1['PREC']:.6f}",
                    f"{m1['REC']:.6f}",
                    f"{m1['F1']:.6f}",
                    f"{chamfer:.6f}" if not np.isnan(chamfer) else "nan",
                ])

            # visualizations
            if viz_dir is not None and saved < viz_max:
                stem = Path(tags[i]).stem
                save_viz(viz_dir, imgs[i], logits[i:i+1], hms[i:i+1], stem)
                save_viz_peaks(viz_dir, imgs[i], logits[i:i+1], tags[i], stem, thr_pred=thr_pred, topk=120)
                saved += 1

    if f_csv is not None:
        f_csv.close()

    avg_loss = total / max(len(loader), 1)
    # average metrics across batches
    for k in ("PCK@2", "PCK@3", "PCK@5", "PREC", "REC", "F1"):
        agg_metrics[k] = agg_metrics[k] / max(count_batches, 1)
    agg_metrics["CHAMFER"] = (agg_metrics["CHAMFER"] / chamfer_batches) if chamfer_batches else float("nan")
    return avg_loss, agg_metrics


class FocalBCE(nn.Module):
    """Custom focal BCE (no torchvision dependency)"""
    def __init__(self, alpha=0.5, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        p = torch.sigmoid(logits)
        pt = targets * p + (1 - targets) * (1 - p)
        w = self.alpha * (1 - pt).pow(self.gamma)
        return (w * bce).mean()


def main():
    ap = argparse.ArgumentParser()
    #ap.add_argument("--data", default=None, help="Train root (images + JSONs). Optional in --eval-only.")
    ap.add_argument("--data", nargs="+", default=None,
                help="One or more train roots (images+JSON). Optional in --eval-only.")
    ap.add_argument("--val", default=None, help="Optional val/eval root")
    ap.add_argument("--out", required=True, help="Run folder")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--sigma", type=float, default=1.8)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--resume", default=None, help="Resume training from checkpoint")
    ap.add_argument("--focal", action="store_true", help="Use Focal BCE; else BCEWithLogits")
    ap.add_argument("--viz-every", type=int, default=1, help="Save visualizations every N epochs (0 disables)")
    ap.add_argument("--viz-max", type=int, default=8, help="Max val samples to visualize per eval")
    ap.add_argument("--log-interval", type=int, default=50)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=123)

    # eval-only & metrics knobs
    ap.add_argument("--eval-only", action="store_true", help="Skip training; evaluate a checkpoint on --val")
    ap.add_argument("--weights", default=None, help="Checkpoint (.pt) to evaluate; if set, overrides --resume")
    ap.add_argument("--per-sample-csv", action="store_true", help="Write detailed per-sample metrics CSV")
    ap.add_argument("--thr-pred", type=float, default=0.2, help="Peak threshold on predicted heatmaps")
    ap.add_argument("--thr-gt",   type=float, default=0.5, help="Peak threshold on GT heatmaps")

    args = ap.parse_args()

    # seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out)
    ensure_dir(out_dir)
    (out_dir / "weights").mkdir(exist_ok=True, parents=True)
    (out_dir / "viz").mkdir(exist_ok=True, parents=True)

    # device info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] torch={torch.__version__} | device={device.type} | cuda={torch.cuda.is_available()}")
    if device.type == "cuda":
        print(f"[env] cuda name={torch.cuda.get_device_name(0)}")
    print(f"[data] train={args.data} | val={args.val or '(none)'}")
    print(f"[cfg] epochs={args.epochs} batch={args.batch} lr={args.lr} base={args.base} img={args.img_size} sigma={args.sigma}")

    # model
    model = UNetLite(in_ch=1, base=args.base, out_ch=1).to(device)

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        print(f"[resume] Loaded {args.resume}")

    
    # also support direct --weights for eval or train init
    if args.weights:
        ckpt = torch.load(args.weights, map_location="cpu")
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        print(f"[weights] Loaded {args.weights}")



    # loss (must exist for both training and eval-only)
    loss_fn = FocalBCE(alpha=0.5, gamma=2.0) if args.focal else nn.BCEWithLogitsLoss()
    
     


    

    # -------- eval-only short-circuit (return before building train_ds) --------
    if args.eval_only:
        if not args.val:
            raise ValueError("--eval-only requires --val to point to your evaluation set")

        # Build only the val loader
        val_ds = IntersectionDataset(args.val, img_size=args.img_size, sigma=args.sigma)
        val_ld = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
        print(f"[sizes] val={len(val_ds)}")

        viz_dir = Path(args.out) / "viz" / "eval"
        val_loss, metrics = evaluate(
            model, val_ld, device, loss_fn,
            viz_dir=viz_dir,
            viz_max=args.viz_max,
            thr_pred=args.thr_pred,
            thr_gt=args.thr_gt,
            per_sample_csv=(Path(args.out) / "per_sample.csv") if args.per_sample_csv else None,
        )

        # Write a simple summary CSV (include CHAMFER)
        csv_path = Path(args.out) / "metrics.csv"
        ensure_dir(csv_path.parent)
        with open(csv_path, "w", newline="") as fsum:
            w = csv.writer(fsum)
            w.writerow(["mode", "val_loss", "PCK@2", "PCK@3", "PCK@5", "PREC", "REC", "F1", "CHAMFER"])
            w.writerow(["eval-only",
                        f"{val_loss:.6f}",
                        f"{metrics['PCK@2']:.6f}", f"{metrics['PCK@3']:.6f}", f"{metrics['PCK@5']:.6f}",
                        f"{metrics['PREC']:.6f}", f"{metrics['REC']:.6f}", f"{metrics['F1']:.6f}",
                        f"{metrics['CHAMFER']:.6f}" if not np.isnan(metrics['CHAMFER']) else "nan"])

        print(f"[eval] loss={val_loss:.4f} | "
            f"PCK2={metrics['PCK@2']:.3f} PCK3={metrics['PCK@3']:.3f} PCK5={metrics['PCK@5']:.3f} | "
            f"P={metrics['PREC']:.3f} R={metrics['REC']:.3f} F1={metrics['F1']:.3f} | "
            f"Chamfer={metrics['CHAMFER']:.3f}")
        return
    


    # ----- Guard: training requires --data -----
    if not args.eval_only and not args.data:
        raise ValueError("Training mode requires --data (or pass --eval-only to skip training).")
    

    # data
    if args.data:
        ds_list = [IntersectionDataset(p, img_size=args.img_size, sigma=args.sigma) for p in args.data]
        train_ds = ConcatDataset(ds_list) if len(ds_list) > 1 else ds_list[0]
        train_ld = DataLoader(
            train_ds, batch_size=args.batch, shuffle=True,
            num_workers=args.num_workers, pin_memory=True
        )
    

    #train_ds = IntersectionDataset(args.data, img_size=args.img_size, sigma=args.sigma)
    #train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
    #                      num_workers=args.num_workers, pin_memory=True)
    


    val_ld = None
    if args.val:
        val_ds = IntersectionDataset(args.val, img_size=args.img_size, sigma=args.sigma)
        val_ld = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
        print(f"[sizes] train={len(train_ds)} | val={len(val_ds)}")
    else:
        print(f"[sizes] train={len(train_ds)} | val=(none)")

    

    

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # CSV logger
    csv_path = out_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "PCK@2", "PCK@3", "PCK@5", "PREC", "REC", "F1", "lr", "sec"])

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_ld, opt, device, loss_fn, scaler if args.amp else None, log_interval=args.log_interval)

        # evaluate
        if val_ld:
            # only save visualizations on epochs that match --viz-every; set to 0 to disable saving
            should_viz = (args.viz_every > 0) and (epoch % args.viz_every == 0)
            viz_dir = (out_dir / "viz" / f"ep{epoch:03d}") if should_viz else None
            
            
            #val_loss, metrics = evaluate(model, val_ld, device, loss_fn,
            #                           viz_dir=viz_dir, viz_max=args.viz_max)
            
            val_loss, metrics = evaluate(
                model, val_ld, device, loss_fn,
                viz_dir=viz_dir, viz_max=args.viz_max,
                thr_pred=args.thr_pred, thr_gt=args.thr_gt
            )

            


        else:
            val_loss, metrics = tr_loss, {"PCK@2": 0.0, "PCK@3": 0.0, "PCK@5": 0.0, "PREC": 0.0, "REC": 0.0, "F1": 0.0}

                
        dt = time.time() - t0
        lr = opt.param_groups[0]["lr"]
        print(f"[{epoch:03d}/{args.epochs}] train={tr_loss:.4f} val={val_loss:.4f} "
              f"| PCK2={metrics['PCK@2']:.3f} PCK3={metrics['PCK@3']:.3f} PCK5={metrics['PCK@5']:.3f} "
              f"| P={metrics['PREC']:.3f} R={metrics['REC']:.3f} F1={metrics['F1']:.3f} "
              f"| {dt:.1f}s")

        # append CSV
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([epoch, f"{tr_loss:.6f}", f"{val_loss:.6f}",
                        f"{metrics['PCK@2']:.6f}", f"{metrics['PCK@3']:.6f}", f"{metrics['PCK@5']:.6f}",
                        f"{metrics['PREC']:.6f}", f"{metrics['REC']:.6f}", f"{metrics['F1']:.6f}",
                        f"{lr:.8f}", f"{dt:.3f}"])

        # Save latest
        latest = {"model": model.state_dict(), "epoch": epoch, "args": vars(args)}
        torch.save(latest, out_dir / "weights" / "latest.pt")

        # Save best (by val loss)
        if val_loss <= best_val:
            best_val = val_loss
            torch.save(latest, out_dir / "weights" / "best.pt")

    print(f"[done] Best val loss: {best_val:.4f}. Weights: {out_dir / 'weights'}")


if __name__ == "__main__":
    main()