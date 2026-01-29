# python/vision/train_v2/eval_cell_interpreter.py
from __future__ import annotations
import argparse, json, os, csv
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Reuse your existing project bits
from python.vision.models.cell_net import CellNet
from python.vision.train_v2.train_cell_interpreter_v2 import (
    JsonlCellList, CellDataset,
    cand_confusion_matrix, cls_confusion_matrix
)
from python.vision.train_v2.metrics import eval_heads


def load_best_thr(weights_path: Path, fallback: float = 0.25) -> float:
    """Try to read a sidecar threshold file next to best.pt (e.g., best.pt.thr.txt)."""
    txt = Path(str(weights_path) + ".thr.txt")
    if txt.exists():
        try:
            return float(txt.read_text().strip())
        except Exception:
            pass
    return fallback


def to_jsonable(v):
    """Robust conversion of tensors/arrays to plain Python for JSON dumps."""
    import numpy as _np
    import torch as _torch
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    if isinstance(v, _torch.Tensor):
        v = v.detach().cpu()
        if v.ndim == 0:
            return float(v.item())
        return v.numpy().tolist()
    if isinstance(v, _np.ndarray):
        if v.ndim == 0:
            return float(v.reshape(()).item())
        return v.tolist()
    # Last resort: try numpy view, else string
    try:
        return float(v)
    except Exception:
        try:
            return _np.asarray(v).tolist()
        except Exception:
            return str(v)


def pretty_metric(v):
    """Readable console formatting."""
    import torch as _torch
    import numpy as _np
    if isinstance(v, (int, float)):
        return f"{v:.6f}" if isinstance(v, float) else str(v)
    if isinstance(v, _torch.Tensor):
        v = v.detach().cpu().numpy()
    if isinstance(v, _np.ndarray):
        if v.ndim == 0:
            return f"{v.reshape(()).item():.6f}"
        # small arrays inline, bigger summarized
        if v.size <= 10:
            return _np.array2string(v, precision=4, suppress_small=True)
        return f"array(shape={tuple(v.shape)}, min={v.min():.4f}, max={v.max():.4f})"
    return str(v)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="JSONL with {path,given_digit,solution_digit,candidates}")
    ap.add_argument("--weights", required=True, help="best.pt (contains model_state or full state dict)")
    ap.add_argument("--img", type=int, default=64)
    ap.add_argument("--inner-crop", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cand-thr", type=float, default=-1.0, help="override; if <0, auto-load from .thr.txt")
    ap.add_argument("--out-dir", type=str, default="runs/cell_eval")
    args = ap.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Data
    cells = JsonlCellList(args.manifest)
    ds = CellDataset(cells, img_size=args.img, inner_crop=args.inner_crop, train=False)
    ld = DataLoader(
        ds, batch_size=args.batch, shuffle=False,
        num_workers=(0 if os.name == "nt" else 2), pin_memory=True
    )

    # Model
    model = CellNet(num_classes=10, lse_tau=0.40).to(device)

    # The warning you saw is benign for your own checkpoints; we still load as usual.
    ckpt = torch.load(args.weights, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()

    # Candidate threshold
    cand_thr = args.cand_thr if args.cand_thr >= 0.0 else load_best_thr(Path(args.weights), fallback=0.25)

    # --- Aggregate metrics (same helpers as in training) ---
    metrics = eval_heads(model, ld, device, cand_thr=cand_thr)

    # Save metrics.json with robust serialization
    metrics_json = {"cand_thr": cand_thr, **{k: to_jsonable(v) for k, v in metrics.items()}}
    (out_dir / "metrics.json").write_text(json.dumps(metrics_json, indent=2), encoding="utf-8")

    # --- Confusion matrices (candidates, solution, given) ---
    cm_counts, _, cm_row_norm = cand_confusion_matrix(model, ld, device, thr=cand_thr)
    np.savetxt(out_dir / "cand_confusion_counts.csv", cm_counts, fmt="%d", delimiter=",")
    np.savetxt(out_dir / "cand_confusion_rownorm.csv", cm_row_norm, fmt="%.6f", delimiter=",")

    solC, _, solRN = cls_confusion_matrix(model, ld, device, head="solution")
    np.savetxt(out_dir / "solution_confusion_counts.csv", solC, fmt="%d", delimiter=",")
    np.savetxt(out_dir / "solution_confusion_rownorm.csv", solRN, fmt="%.6f", delimiter=",")

    givC, _, givRN = cls_confusion_matrix(model, ld, device, head="given")
    np.savetxt(out_dir / "given_confusion_counts.csv", givC, fmt="%d", delimiter=",")
    np.savetxt(out_dir / "given_confusion_rownorm.csv", givRN, fmt="%.6f", delimiter=",")

    # --- Per-cell dump to line up with Android capture_debug.csv ---
    model.eval()
    with open(out_dir / "per_cell.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = [
            "idx","path","given_label","solution_label",
            "pred_given","pred_solution"
        ]
        # Probabilities
        header += [f"pG_{d}" for d in range(10)]     # Given softmax
        header += [f"pS_{d}" for d in range(10)]     # Solution softmax
        header += [f"pC_{d}" for d in range(10)]     # Candidates sigmoid (probabilities)
        # Candidate bits @ thr
        header += [f"cand_{d}@{cand_thr:.2f}" for d in range(10)]
        # Raw logits for all heads
        header += [f"logits_given_{d}" for d in range(10)]
        header += [f"logits_solution_{d}" for d in range(10)]
        header += [f"logits_candidates_{d}" for d in range(10)]
        w.writerow(header)

        idx_base = 0
        for (x, y_g, y_s, _) in ld:
            x = x.to(device)
            out = model(x)
            lg = out["logits_given"]        # [B,10]
            ls = out["logits_solution"]     # [B,10]
            lc = out["logits_candidates"]   # [B,10]

            # Argmax predictions
            pred_g = torch.argmax(lg, dim=1).cpu().numpy()
            pred_s = torch.argmax(ls, dim=1).cpu().numpy()

            # Probabilities
            pG = F.softmax(lg, dim=1).cpu().numpy()
            pS = F.softmax(ls, dim=1).cpu().numpy()
            pC = torch.sigmoid(lc).cpu().numpy()

            # Candidate bits at threshold
            cand_bits = (pC >= cand_thr).astype(np.int32)

            # Raw logits
            LG = lg.detach().cpu().numpy()
            LS = ls.detach().cpu().numpy()
            LC = lc.detach().cpu().numpy()

            B = x.size(0)
            for b in range(B):
                row = [
                    idx_base + b,
                    cells.rows[idx_base + b][0],
                    int(y_g[b].item()),
                    int(y_s[b].item()),
                    int(pred_g[b]),
                    int(pred_s[b]),
                ]
                # Probabilities
                row += [float(pG[b, d]) for d in range(10)]
                row += [float(pS[b, d]) for d in range(10)]
                row += [float(pC[b, d]) for d in range(10)]
                # Candidate bits
                row += [int(cand_bits[b, d]) for d in range(10)]
                # Raw logits
                row += [float(LG[b, d]) for d in range(10)]
                row += [float(LS[b, d]) for d in range(10)]
                row += [float(LC[b, d]) for d in range(10)]

                w.writerow(row)
            idx_base += B

    # --- Console summary (clear & compact) ---
    print("")
    print("==== Cell Interpreter Evaluation ====")
    print(f"Manifest      : {args.manifest}")
    print(f"Weights       : {args.weights}")
    print(f"Images        : {len(ds)}")
    print(f"IMG size      : {args.img}  (inner-crop={args.inner_crop})")
    print(f"Batch size    : {args.batch}")
    print(f"Device        : {device}")
    print(f"Cand. Thr     : {cand_thr:.4f}")
    print("")
    # Print top-level metrics in a stable order if present
    keys_order = [
        "acc_solution", "acc_solution_non0",
        "acc_given", "acc_given_non0",
        "f1_candidates", "f1_candidates_non0",
        "precision_candidates", "recall_candidates",
        "n_nonzero_solution", "n_nonzero_given"
    ]
    for k in keys_order:
        if k in metrics:
            print(f"{k:24s}: {pretty_metric(metrics[k])}")
    # Print any additional keys
    for k, v in metrics.items():
        if k not in keys_order:
            print(f"{k:24s}: {pretty_metric(v)}")
    print("")
    print(f"Wrote outputs : {out_dir}")
    print("  - metrics.json")
    print("  - cand_confusion_counts.csv / cand_confusion_rownorm.csv")
    print("  - solution_confusion_counts.csv / solution_confusion_rownorm.csv")
    print("  - given_confusion_counts.csv / given_confusion_rownorm.csv")
    print("  - per_cell.csv")
    print("=====================================")


if __name__ == "__main__":
    main()