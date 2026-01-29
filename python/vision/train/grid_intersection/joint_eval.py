# joint_eval.py
import math
import numpy as np
import torch
import torch.nn.functional as F

# ---------- Sub-pixel refinements ----------

def _quadfit_offset_3x3(patch: torch.Tensor):
    """
    patch: (3,3) around the peak (float). Returns (dx, dy) in [-1,1] w.r.t. center pixel.
    Uses quadratic fit on log-intensities (more stable for peaked maps).
    """
    # stabilize
    eps = 1e-8
    Z = torch.log(torch.clamp(patch, min=eps))
    # coordinates
    yy, xx = torch.meshgrid(torch.arange(3, device=patch.device),
                            torch.arange(3, device=patch.device), indexing="ij")
    X = torch.stack([torch.ones_like(xx), xx.float(), yy.float(),
                     (xx**2).float(), (yy**2).float(), (xx*yy).float()], dim=-1)  # (3,3,6)
    X = X.reshape(-1, 6)       # (9,6)
    y = Z.reshape(-1, 1)       # (9,1)
    # least squares
    beta, _ = torch.lstsq(y, X) if hasattr(torch, "lstsq") else torch.linalg.lstsq(X, y).solution  # (6,1)
    a0,a1,a2,a3,a4,a5 = beta.view(-1)
    # extremum of quadratic surface z = a0 + a1 x + a2 y + a3 x^2 + a4 y^2 + a5 xy
    # solve grad=0: [2a3  a5; a5  2a4] [x;y] = [-a1; -a2]
    A = torch.tensor([[2*a3, a5],[a5, 2*a4]], device=patch.device)
    b = -torch.tensor([a1, a2], device=patch.device)
    try:
        sol = torch.linalg.solve(A, b)
        # convert from absolute 0..2 to offset around center (1,1)
        dx = sol[0] - 1.0
        dy = sol[1] - 1.0
        # clamp to be safe
        dx = torch.clamp(dx, -1.0, 1.0)
        dy = torch.clamp(dy, -1.0, 1.0)
        return dx.item(), dy.item()
    except RuntimeError:
        return 0.0, 0.0

def _softargmax_2d(hm: torch.Tensor, temp: float = 1.0):
    """
    hm: (H,W). Returns (x,y) in pixel coordinates (float).
    """
    H, W = hm.shape
    x_grid = torch.arange(W, device=hm.device).float()
    y_grid = torch.arange(H, device=hm.device).float()
    p = F.softmax(hm.reshape(-1) / temp, dim=0)
    px = (p * x_grid.repeat(H)).sum()
    py = (p * torch.repeat_interleave(y_grid, W)).sum()
    return px.item(), py.item()

# ---------- Peak finding ----------

def decode_joints_from_heatmap(
    jmap: torch.Tensor,
    topk: int = 100,
    nms_kernel: int = 3,
    method: str = "quadfit",  # "quadfit" or "softargmax"
    conf_thresh: float = 0.05,
):
    """
    jmap: (H,W) torch float32 in [0,1]
    returns: (xy: (N,2) float, scores: (N,)) in image pixel coords
    """
    H, W = jmap.shape
    # NMS via maxpool
    maxp = F.max_pool2d(jmap[None, None, ...], nms_kernel, 1, nms_kernel//2)[0,0]
    keep = (jmap == maxp) & (jmap >= conf_thresh)
    ys, xs = torch.nonzero(keep, as_tuple=True)
    scores = jmap[ys, xs]
    if scores.numel() == 0:
        return np.zeros((0,2), np.float32), np.zeros((0,), np.float32)

    # sort by score
    idx = torch.argsort(scores, descending=True)[:topk]
    xs = xs[idx]; ys = ys[idx]; scores = scores[idx]

    xy = []
    for x, y in zip(xs.tolist(), ys.tolist()):
        if method == "quadfit" and 1 <= x <= W-2 and 1 <= y <= H-2:
            patch = jmap[y-1:y+2, x-1:x+2].clone()
            dx, dy = _quadfit_offset_3x3(patch)
            fx = x + dx
            fy = y + dy
        elif method == "softargmax":
            # small window around peak for stability
            x0, x1 = max(0, x-2), min(W, x+3)
            y0, y1 = max(0, y-2), min(H, y+3)
            sub = jmap[y0:y1, x0:x1]
            sx, sy = _softargmax_2d(sub, temp=0.5)
            fx = x0 + sx; fy = y0 + sy
        else:
            fx, fy = float(x), float(y)
        xy.append([fx, fy])

    xy = np.array(xy, dtype=np.float32)
    return xy, scores.cpu().numpy().astype(np.float32)

# ---------- 10×10 ordering + metrics ----------

def order_10x10_raster(xy: np.ndarray):
    """
    xy: (N,2) float. Returns (100,2) ordered by row-major if possible.
    Strategy: k-means 10 rows by y, sort within row by x; fallback to nearest grid.
    """
    if xy.shape[0] < 50:   # too few to trust ordering
        return None

    y_sorted = xy[np.argsort(xy[:,1])]
    rows = np.array_split(y_sorted, 10)
    out = []
    for r in rows:
        r = r[np.argsort(r[:,0])]
        out.append(r[:10] if r.shape[0] >= 10 else r)
    out = np.vstack(out)
    if out.shape[0] >= 100:
        return out[:100]
    return None

def joint_metrics(gt_xy: np.ndarray, pred_xy: np.ndarray, img_size: int = 384):
    """
    gt_xy, pred_xy: (100,2) arrays in pixel coords.
    Returns dict with MJE and AP@kpx.
    """
    # assume both are aligned 10x10 raster; if not, do nearest-neighbor assignment
    if pred_xy.shape[0] != 100:
        raise ValueError("pred_xy must be (100,2)")

    diffs = pred_xy - gt_xy
    d = np.sqrt((diffs**2).sum(axis=1))  # (100,)
    mje = float(d.mean())
    ap1 = float((d <= 1.0).mean())
    ap2 = float((d <= 2.0).mean())
    ap3 = float((d <= 3.0).mean())
    return {"J_MJE": mje, "AP@1": ap1, "AP@2": ap2, "AP@3": ap3}