"""
Calibration for CellNet
-----------------------

Fits temperature scaling parameters for:

  - Given head (softmax)
  - Solution head (softmax)
  - Candidates head (sigmoid)

Given a trained CellNet and a calibration JSONL set, we compute:

  T_given, T_solution, T_candidates

and store them in:

  models/cell_interpreter/cell_calibration.json

At inference:
  probs_given    = softmax(logits_given / T_given)
  probs_solution = softmax(logits_solution / T_solution)
  probs_cand     = sigmoid(logits_candidates / T_candidates)
"""

from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from vision.models.cell_net import CellNet
from vision.train.train_cell_interpreter import JsonlCellList, CellDataset


def collect_logits_labels(
    model: nn.Module,
    ds: CellDataset,
    device: torch.device,
    batch_size: int = 1024,
):
    """Collect logits and labels for all samples in ds."""
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model.eval()
    all_log_g, all_log_s, all_log_c = [], [], []
    all_y_g, all_y_s, all_y_c = [], [], []

    with torch.no_grad():
        for x, y_g, y_s, y_c in loader:
            x = x.to(device, non_blocking=True)
            y_g = y_g.to(device, non_blocking=True)
            y_s = y_s.to(device, non_blocking=True)
            y_c = y_c.to(device, non_blocking=True)

            out = model(x)
            lg = out["logits_given"]      # [B,10]
            ls = out["logits_solution"]   # [B,10]
            lc = out["logits_candidates"] # [B,10]

            all_log_g.append(lg.detach().cpu())
            all_log_s.append(ls.detach().cpu())
            all_log_c.append(lc.detach().cpu())

            all_y_g.append(y_g.detach().cpu())
            all_y_s.append(y_s.detach().cpu())
            all_y_c.append(y_c.detach().cpu())

    logits_given = torch.cat(all_log_g, dim=0)      # [N,10]
    logits_solution = torch.cat(all_log_s, dim=0)   # [N,10]
    logits_cand = torch.cat(all_log_c, dim=0)       # [N,10]

    y_given = torch.cat(all_y_g, dim=0).long()      # [N]
    y_solution = torch.cat(all_y_s, dim=0).long()   # [N]
    y_cand = torch.cat(all_y_c, dim=0).float()      # [N,10]

    return logits_given, logits_solution, logits_cand, y_given, y_solution, y_cand


def _optimize_temperature_softmax(
    logits: torch.Tensor,
    labels: torch.Tensor,
    init_T: float = 1.0,
    max_iter: int = 200,
) -> float:
    """
    Temperature scaling for a softmax classifier.
    Minimizes NLL(softmax(logits/T), labels).
    """
    device = logits.device
    T = torch.ones(1, device=device) * init_T
    T.requires_grad_(True)

    ce = nn.CrossEntropyLoss()
    optimizer = optim.LBFGS([T], lr=0.1, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        # logits_scaled = logits / T
        logits_scaled = logits / T.clamp(1e-3, 100.0)
        loss = ce(logits_scaled, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    T_opt = T.detach().cpu().item()
    return float(max(1e-3, min(T_opt, 100.0)))


def _optimize_temperature_sigmoid(
    logits: torch.Tensor,
    labels: torch.Tensor,
    init_T: float = 1.0,
    max_iter: int = 200,
) -> float:
    """
    Temperature scaling for sigmoid multi-label:
    Minimizes BCEWithLogitsLoss(logits/T, labels).
    """
    device = logits.device
    T = torch.ones(1, device=device) * init_T
    T.requires_grad_(True)

    bce = nn.BCEWithLogitsLoss()
    optimizer = optim.LBFGS([T], lr=0.1, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        logits_scaled = logits / T.clamp(1e-3, 100.0)
        loss = bce(logits_scaled, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    T_opt = T.detach().cpu().item()
    return float(max(1e-3, min(T_opt, 100.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-ckpt",
        type=str,
        required=True,
        help="Path to trained CellNet checkpoint (.pt)",
    )
    ap.add_argument(
        "--calib-manifest",
        type=str,
        required=True,
        help="JSONL manifest for calibration set",
    )
    ap.add_argument(
        "--img",
        type=int,
        default=64,
        help="Input size (H=W) used during training",
    )
    ap.add_argument(
        "--inner-crop",
        type=float,
        default=1.0,
        help="Center crop fraction used during training",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    ap.add_argument(
        "--out-json",
        type=str,
        default="models/cell_interpreter/cell_calibration.json",
        help="Output JSON path for calibration parameters",
    )
    ap.add_argument(
        "--batch",
        type=int,
        default=1024,
        help="Batch size when collecting logits",
    )
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt_path = Path(args.model_ckpt)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("[calib] Loading model from:", ckpt_path)
    model = CellNet(num_classes=10).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state", ckpt)
    msg = model.load_state_dict(state, strict=False)
    print("[calib] load_state_dict:", msg)

    print("[calib] Loading calibration dataset:", args.calib_manifest)
    cell_list = JsonlCellList(args.calib_manifest)
    calib_ds = CellDataset(
        cell_list,
        img_size=args.img,
        inner_crop=args.inner_crop,
        train=False,
    )

    print("[calib] Collecting logits/labels...")
    (
        logits_given,
        logits_solution,
        logits_cand,
        y_given,
        y_solution,
        y_cand,
    ) = collect_logits_labels(
        model.to(device),
        calib_ds,
        device=device,
        batch_size=args.batch,
    )

    logits_given = logits_given.to(device)
    logits_solution = logits_solution.to(device)
    logits_cand = logits_cand.to(device)
    y_given = y_given.to(device)
    y_solution = y_solution.to(device)
    y_cand = y_cand.to(device)

    print("[calib] Optimizing temperature for given head (softmax)...")
    T_given = _optimize_temperature_softmax(logits_given, y_given)

    print("[calib] Optimizing temperature for solution head (softmax)...")
    T_solution = _optimize_temperature_softmax(logits_solution, y_solution)

    print("[calib] Optimizing temperature for candidates head (sigmoid)...")
    T_cand = _optimize_temperature_sigmoid(logits_cand, y_cand)

    print(f"[calib] T_given    = {T_given:.4f}")
    print(f"[calib] T_solution = {T_solution:.4f}")
    print(f"[calib] T_cand     = {T_cand:.4f}")

    calib_dict = {
        "mode": "multihead",
        "input": {
            "layout": "NCHW",
            "height": args.img,
            "width": args.img,
            "mean": 0.5,
            "std": 0.5,
        },
        "heads": {
            "given": {
                "name": "logits_given",
                "temperature": T_given,
                "activation": "softmax",
            },
            "solution": {
                "name": "logits_solution",
                "temperature": T_solution,
                "activation": "softmax",
            },
            "candidates": {
                "name": "logits_candidates",
                "temperature": T_cand,
                "activation": "sigmoid",
            },
        },
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(calib_dict, f, indent=2)

    print("[calib] Saved calibration JSON to:", out_path)


if __name__ == "__main__":
    main()