from __future__ import annotations
import argparse, json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Reuse dataset helpers from your v2 trainer
from python.vision.train_v2.train_cell_interpreter_v2 import (
    JsonlCellList, CellDataset
)
from python.vision.models.cell_net import CellNet
from python.vision.train_v2.metrics import scan_candidate_thresholds


def parse_thr_grid(s: str) -> list[float]:
    return [float(t.strip()) for t in s.split(",") if t.strip()]


def main():
    ap = argparse.ArgumentParser(description="Re-scan candidate threshold on a saved CellNet checkpoint.")
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint (.pt) saved by the trainer (contains model_state).")
    ap.add_argument("--val-manifest", required=True, help="JSONL with validation cells (REAL).")
    ap.add_argument("--img", type=int, default=96)
    ap.add_argument("--inner-crop", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--thr-grid", type=str, default="0.05,0.10,0.15,0.20,0.25,0.30,0.35")
    ap.add_argument("--out", type=str, default="", help="Optional path to write best threshold text file (default: alongside ckpt).")
    args = ap.parse_args()

    device = torch.device(args.device)

    # Data
    val_cells = JsonlCellList(args.val_manifest)
    val_ds = CellDataset(val_cells, img_size=args.img, inner_crop=args.inner_crop, train=False)
    val_ld = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        num_workers=(0 if (device.type == "cpu") else 2), pin_memory=True)

    # Model
    model = CellNet(num_classes=10).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("model_state", ckpt)
    missing = model.load_state_dict(state, strict=False)
    print(f"[load] checkpoint={args.ckpt}")
    if getattr(missing, "missing_keys", None) or getattr(missing, "unexpected_keys", None):
        print(f"[load] missing={getattr(missing, 'missing_keys', [])} unexpected={getattr(missing, 'unexpected_keys', [])}")

    thr_list = parse_thr_grid(args.thr_grid)
    best_thr, scores = scan_candidate_thresholds(model, val_ld, device, thr_list)
    print("[thr-scan] cand F1_nonempty by thr:", " ".join(f"{t:.2f}:{scores[t]:.3f}" for t in thr_list))
    print(f"[thr-scan] best_thr={best_thr:.2f}")

    # Write sidecar result
    out_path = Path(args.out) if args.out else (Path(args.ckpt).with_suffix(".thr.txt"))
    out_json = out_path.with_suffix(".thr.json")
    out_path.write_text(f"{best_thr:.6f}\n", encoding="utf-8")
    out_json.write_text(json.dumps({"best_thr": best_thr, "scores": scores}, indent=2), encoding="utf-8")
    print(f"[write] {out_path}")
    print(f"[write] {out_json}")


if __name__ == "__main__":
    main()