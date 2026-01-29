# python/vision/train_v2/metrics.py
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple

@torch.no_grad()
def eval_heads(model: nn.Module, loader: DataLoader, device: torch.device, cand_thr: float) -> Dict:
    model.eval()
    total = 0
    correct_g = 0
    correct_s = 0

    total_g_nz = 0
    correct_g_nz = 0
    total_s_nz = 0
    correct_s_nz = 0

    tp = np.zeros(10, dtype=np.int64)
    fp = np.zeros(10, dtype=np.int64)
    fn = np.zeros(10, dtype=np.int64)

    tp_ne = np.zeros(10, dtype=np.int64)
    fp_ne = np.zeros(10, dtype=np.int64)
    fn_ne = np.zeros(10, dtype=np.int64)

    cm_g = np.zeros((10,10), dtype=np.int64)
    cm_s = np.zeros((10,10), dtype=np.int64)

    for x, y_g, y_s, y_c in loader:
        x = x.to(device, non_blocking=True)
        y_g = y_g.to(device, non_blocking=True)
        y_s = y_s.to(device, non_blocking=True)
        y_c = y_c.to(device, non_blocking=True)

        out = model(x)
        pg = out["logits_given"].argmax(1)
        ps = out["logits_solution"].argmax(1)

        bs = x.size(0)
        total += bs
        correct_g += int((pg == y_g).sum().item())
        correct_s += int((ps == y_s).sum().item())

        # non-zero slices
        m_g = (y_g != 0)
        m_s = (y_s != 0)
        if m_g.any():
            total_g_nz += int(m_g.sum().item())
            correct_g_nz += int(((pg == y_g) & m_g).sum().item())
        if m_s.any():
            total_s_nz += int(m_s.sum().item())
            correct_s_nz += int(((ps == y_s) & m_s).sum().item())

        # confusion matrices
        yg = y_g.cpu().numpy()
        pgc = pg.cpu().numpy()
        ys = y_s.cpu().numpy()
        psc = ps.cpu().numpy()
        for t, p in zip(yg, pgc): cm_g[t, p] += 1
        for t, p in zip(ys, psc): cm_s[t, p] += 1

        # candidates
        probs = torch.sigmoid(out["logits_candidates"])
        pred = (probs >= cand_thr).float()

        yt = y_c.cpu().numpy().astype(bool)
        yp = pred.cpu().numpy().astype(bool)

        for d in range(10):
            tp[d] += int(np.logical_and(yt[:,d],  yp[:,d]).sum())
            fp[d] += int(np.logical_and(~yt[:,d], yp[:,d]).sum())
            fn[d] += int(np.logical_and(yt[:,d],  ~yp[:,d]).sum())

        mask_ne = yt.any(axis=1)
        if mask_ne.any():
            yt2 = yt[mask_ne]; yp2 = yp[mask_ne]
            for d in range(10):
                tp_ne[d] += int(np.logical_and(yt2[:,d],  yp2[:,d]).sum())
                fp_ne[d] += int(np.logical_and(~yt2[:,d], yp2[:,d]).sum())
                fn_ne[d] += int(np.logical_and(yt2[:,d],  ~yp2[:,d]).sum())

    def f1(tp, fp, fn):
        f = np.zeros(10, dtype=np.float32)
        for d in range(10):
            prec = tp[d] / max(1, tp[d] + fp[d])
            rec  = tp[d] / max(1, tp[d] + fn[d])
            f[d] = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
        return f

    f1_all = f1(tp, fp, fn)
    f1_ne  = f1(tp_ne, fp_ne, fn_ne)

    return {
        "acc_given": correct_g / max(1,total),
        "acc_solution": correct_s / max(1,total),
        "acc_given_non0": correct_g_nz / max(1,total_g_nz),
        "acc_solution_non0": correct_s_nz / max(1,total_s_nz),
        "f1_candidates": f1_all,
        "f1_candidates_nonempty": f1_ne,
        "cm_given": cm_g,
        "cm_solution": cm_s,
    }

@torch.no_grad()
def scan_candidate_thresholds(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    thr_list: List[float]
) -> Tuple[float, Dict[float, float]]:
    """
    Return (best_thr, {thr: f1_nonempty_macro})
    """
    scores = {}
    best_thr = thr_list[0]
    best = -1.0
    for thr in thr_list:
        m = eval_heads(model, loader, device, cand_thr=thr)
        f1_ne = float(m["f1_candidates_nonempty"].mean())
        scores[thr] = f1_ne
        if f1_ne > best:
            best = f1_ne
            best_thr = thr
    return best_thr, scores