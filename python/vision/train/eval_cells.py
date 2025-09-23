
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
import numpy as np
import matplotlib.pyplot as plt
import itertools
from PIL import Image
import torch
import torch.nn.functional as F

from vision.infer.classify_cells_model import load_model, _load_tile

DIGITS = list("0123456789")

def iter_dataset(root: Path):
    for cls_name in DIGITS:
        cls_dir = root / cls_name
        if not cls_dir.exists():
            continue
        for p in cls_dir.glob("*.png"):
            yield int(cls_name), p
        for p in cls_dir.glob("*.jpg"):
            yield int(cls_name), p
        for p in cls_dir.glob("*.jpeg"):
            yield int(cls_name), p

def confusion_matrix(y_true, y_pred, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def plot_cm(cm: np.ndarray, out_png: Path):
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title('Confusion Matrix')
    plt.colorbar(im, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels(DIGITS); ax.set_yticklabels(DIGITS)
    thresh = cm.max() * 0.6
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        v = cm[i,j]
        ax.text(j, i, str(v), ha="center", va="center", color="white" if v > thresh else "black", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to best.pt")
    ap.add_argument("--data", required=True, help="Root of val set with folders 0..9")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="runs/eval")
    args = ap.parse_args()

    model = load_model(Path(args.model), device=args.device)

    xs, ys = [], []
    for y, p in iter_dataset(Path(args.data)):
        arr = _load_tile(p)
        xs.append(arr)
        ys.append(y)
    if not xs:
        raise SystemExit("No images found in val data.")

    X = torch.from_numpy(np.stack(xs, 0)).to(args.device)  # (N,1,28,28)
    Y = np.array(ys, dtype=np.int64)
    with torch.no_grad():
        logits = model(X)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)

    acc = float((preds == Y).mean()) * 100.0
    per_class = []
    for c in range(10):
        idx = np.where(Y == c)[0]
        a = float((preds[idx] == Y[idx]).mean()) * 100.0 if len(idx) else 0.0
        per_class.append(a)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(Y, preds, 10)
    plot_cm(cm, out_dir/"confusion_matrix.png")

    # Save a simple report
    (out_dir/"report.txt").write_text(
        "Overall acc: {:.2f}%\n".format(acc) +
        "\n".join(f"class {d}: {a:.2f}%" for d, a in zip(DIGITS, per_class)),
        encoding="utf-8"
    )

    print(f"Overall acc: {acc:.2f}%")
    print("Per-class:", ", ".join(f"{d}:{a:.1f}%" for d,a in zip(DIGITS, per_class)))
    print(f"Confusion matrix saved to {out_dir/'confusion_matrix.png'}")

if __name__ == "__main__":
    main()
