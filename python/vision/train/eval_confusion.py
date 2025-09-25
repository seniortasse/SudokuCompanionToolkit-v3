
# vision/train/eval_confusion.py
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from vision.models.cnn_small import CNN28
from vision.train.train_cells import JsonlList, ImageDataset

def confusion_matrix(y_true, y_pred, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--val-manifest", required=True)
    ap.add_argument("--img", type=int, default=28)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--inner-crop", type=float, default=1.0, help="Center-crop fraction before resize (e.g., 0.9)")
    ap.add_argument("--out", type=str, default="runs/eval")
    args = ap.parse_args()

    device = args.device
    model = CNN28().to(device)
    ck = torch.load(args.model, map_location="cpu")
    sd = ck["state_dict"] if "state_dict" in ck else ck
    model.load_state_dict(sd, strict=False)
    model.eval()

    lst = JsonlList(Path(args.val_manifest))
    ds = ImageDataset(lst, img_size=args.img, train=False, inner_crop=args.inner_crop)
    y_true, y_pred = [], []

    for i in range(len(ds)):
        (x, y) = ds[i]
        x = x.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            prob = F.softmax(logits, dim=1).cpu().numpy().squeeze(0)
        y_true.append(int(y.item()))
        y_pred.append(int(np.argmax(prob)))

    cm = confusion_matrix(y_true, y_pred, 10)
    acc_micro = (cm.trace() / cm.sum()) if cm.sum() > 0 else 0.0
    recalls = []
    for d in range(1,10):
        denom = cm[d,:].sum()
        recalls.append((cm[d,d] / denom) if denom > 0 else 0.0)
    acc_macro_1to9 = float(np.mean(recalls)) if recalls else 0.0

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    fig, ax = plt.subplots(figsize=(8,8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.colorbar(im, ax=ax)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    print(f"Eval accuracy (micro): {acc_micro:.4f}  ({cm.trace()}/{cm.sum()})")
    print(f"Macro accuracy 1..9  : {acc_macro_1to9:.4f}")
    print(f"Saved confusion matrix to {out_dir}")

if __name__ == "__main__":
    main()
