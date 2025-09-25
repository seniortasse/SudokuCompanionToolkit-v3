# probe_csv_vs_phone.py
import json, argparse
from pathlib import Path
import numpy as np
import torch

def load_phone_full(session_dir: Path):
    jf = session_dir / "preds_full.json"
    with jf.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    cells = obj.get("cells") or []
    if not cells:
        raise RuntimeError("preds_full.json has no 'cells' array")
    names   = [c["name"] for c in cells]
    logits  = [c["logits"] for c in cells]
    top1    = [c["top1"]   for c in cells]
    return names, logits, top1

def read_csv_28x28(csv_path: Path) -> np.ndarray:
    # CSV saved by phone is the exact 28x28 tensor the model sees ([-1, 1]).
    a = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)
    if a.shape != (28, 28):
        raise RuntimeError(f"Bad CSV shape {a.shape} at {csv_path}")
    return a

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to cell_cnn28_logits.ptl")
    ap.add_argument("--session", required=True, help="phone export session dir (contains preds_full.json, pre/)")
    ap.add_argument("--eps", type=float, default=1e-3, help="∞-norm tolerance for logit equality")
    ap.add_argument("--show", type=int, default=20, help="max mismatches to print")
    args = ap.parse_args()

    sess = Path(args.session)
    names, phone_logits, phone_top1 = load_phone_full(sess)

    # Load TorchScript model
    m = torch.jit.load(args.model, map_location="cpu")
    m.eval()
    torch.set_grad_enabled(False)

    deltas = []     # (l2, linf)
    bad    = []     # (i, name, phone_top, pc_top, linf, l2)

    for i, name in enumerate(names):
        stem = Path(name).stem            # e.g., "r8c9"
        csv_path = sess / "pre" / f"{stem}_28x28.csv"
        x = read_csv_28x28(csv_path)      # (28, 28) in [-1,1]
        t = torch.from_numpy(x).view(1, 1, 28, 28)

        out = m(t).squeeze(0).numpy()     # (10,) logits on desktop
        ph  = np.asarray(phone_logits[i], dtype=np.float32)  # (10,) logits from phone

        l2   = float(np.sqrt(np.mean((out - ph) ** 2)))
        linf = float(np.max(np.abs(out - ph)))
        deltas.append((l2, linf))

        d_top = int(np.argmax(out))
        p_top = int(phone_top1[i])

        # flag if class differs OR logits differ more than eps
        if (d_top != p_top) or (linf > args.eps):
            bad.append((i, name, p_top, d_top, linf, l2))

    # Report
    max_linf = max(d[1] for d in deltas) if deltas else 0.0
    mean_l2  = float(np.mean([d[0] for d in deltas])) if deltas else 0.0
    print(f"Checked {len(names)} cells")
    print(f"Max |Δlogit| = {max_linf:.6f}    Mean L2 = {mean_l2:.6f}")
    print(f"Mismatches (top1 or |Δ|_inf > {args.eps:g}): {len(bad)}")
    for i, name, p_top, d_top, linf, l2 in bad[:args.show]:
        r = i // 9 + 1
        c = i % 9 + 1
        print(f"  {name:>10s} (r{r}c{c})   phone_top={p_top}   pc_top={d_top}   |Δ|_inf={linf:.4g}   L2={l2:.4g}")

if __name__ == "__main__":
    main()