
# vision/train/train_cells.py
from __future__ import annotations
import argparse, json, os, random
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from vision.models.cnn_small import CNN28

def set_seed(seed: int = 1337):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ensure_abs(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())

class CenterSquare:
    def __call__(self, im: Image.Image) -> Image.Image:
        w, h = im.size
        if w == h: return im
        side = min(w, h)
        l = (w - side)//2; t = (h - side)//2
        return im.crop((l, t, l+side, t+side))

class CenterFrac:
    def __init__(self, frac: float = 1.0):
        self.frac = float(frac)
    def __call__(self, im: Image.Image) -> Image.Image:
        f = self.frac
        if f >= 0.999: return im
        w,h = im.size
        side = min(w,h)
        keep = max(1, int(round(side * f)))
        l = (w - keep)//2; t = (h - keep)//2
        return im.crop((l,t,l+keep,t+keep))

# ---------------- JSONL-backed datasets ----------------

class JsonlList:
    """Holds (path, label, source) triplets loaded from a JSONL file."""
    def __init__(self, jsonl_path: Path):
        self.items: List[Tuple[str, int, str]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                obj = json.loads(line)
                p = ensure_abs(obj["path"])
                lab = int(obj["label"])
                src = str(obj.get("source", "unknown"))
                self.items.append((p, lab, src))
        if not self.items:
            raise ValueError(f"No entries in {jsonl_path}")

    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]

class ImageDataset(Dataset):
    """Basic image classification dataset from a JsonlList."""
    def __init__(self, listing: JsonlList, img_size: int = 28, train: bool = False, inner_crop: float = 1.0):
        self.listing = listing
        self.inner_crop = float(inner_crop)
        tfms = []
        tfms.append(transforms.Grayscale(num_output_channels=1))
        tfms.append(CenterSquare())
        tfms.append(CenterFrac(self.inner_crop))
        tfms.append(transforms.Resize((img_size, img_size)))
        if train:
            tfms.append(transforms.RandomAffine(
                degrees=6, translate=(0.08, 0.08), scale=(0.9, 1.1), fill=255))
        tfms.append(transforms.ToTensor())
        tfms.append(transforms.Normalize((0.5,), (0.5,)))
        self.tfm = transforms.Compose(tfms)

    def __len__(self): return len(self.listing)

    def __getitem__(self, idx):
        path, lab, _ = self.listing[idx]
        im = Image.open(path).convert("L")
        x = self.tfm(im)
        y = torch.tensor(lab, dtype=torch.long)
        return x, y

class MixedDataset(Dataset):
    """
    Draws samples on-the-fly from multiple datasets according to a ratio.
    __len__ defines the nominal epoch size (default: sum of each dataset length).
    """
    def __init__(self, datasets: List[ImageDataset], ratios: List[float], epoch_size: Optional[int] = None):
        assert len(datasets) == len(ratios) and len(datasets) > 0
        self.datasets = datasets
        s = sum(ratios); self.ratios = [r/s for r in ratios]
        self.epoch_size = epoch_size or sum(len(ds) for ds in datasets)
        import numpy as _np
        self.cum = _np.cumsum(self.ratios)

    def __len__(self): return self.epoch_size

    def __getitem__(self, idx):
        import numpy as _np, random as _rnd
        u = _rnd.random()
        src_idx = int(_np.searchsorted(self.cum, u, side="right"))
        if src_idx >= len(self.datasets): src_idx = len(self.datasets)-1
        ds = self.datasets[src_idx]
        ridx = _rnd.randrange(len(ds))
        return ds[ridx]

# ---------------- Metrics & saving ----------------

def per_class_metrics(model: nn.Module, loader: DataLoader, device: str) -> Dict[int, float]:
    model.eval()
    correct = {i: 0 for i in range(10)}
    total   = {i: 0 for i in range(10)}
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            pred = model(x).argmax(1)
            for i in range(10):
                mask = (y == i)
                total[i]  += int(mask.sum().item())
                correct[i]+= int(((pred == i) & mask).sum().item())
    acc = {i: (correct[i] / total[i] * 100.0) if total[i] > 0 else 0.0 for i in range(10)}
    return acc

def compute_class_weights(listings: List[JsonlList]) -> torch.Tensor:
    counts = np.zeros(10, dtype=np.int64)
    for lst in listings:
        for _, lab, _ in lst.items:
            counts[lab] += 1
    counts = np.maximum(counts, 1)
    inv = 1.0 / counts
    weights = inv / inv.sum() * 10.0
    return torch.tensor(weights, dtype=torch.float32)

def save_ckpt(model: nn.Module, out_dir: Path, tag: str = "best.pt"):
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / tag
    torch.save({"state_dict": model.state_dict()}, ckpt)
    print(f"  saved {ckpt}")

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-manifests", type=str, default="", help="Comma-separated JSONL paths for training")
    ap.add_argument("--train-mix", type=str, default="", help="Comma-separated ratios matching --train-manifests")
    ap.add_argument("--val-manifest", type=str, default="", help="JSONL path for validation (or comma-separated)")
    ap.add_argument("--img", type=int, default=28, help="Input size")
    ap.add_argument("--inner-crop", type=float, default=1.0, help="Center-crop fraction before resize (e.g., 0.9)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save-dir", type=str, default="vision/train/checkpoints")
    ap.add_argument("--warm-start", type=str, default="", help="Path to a .pt checkpoint to initialize weights.")
    ap.add_argument("--class-weights", type=str, default="", choices=["", "auto"], help="'auto' weights loss inverse by class freq")
    ap.add_argument("--workers", type=int, default=(0 if os.name == "nt" else 2), help="DataLoader workers; 0 on Windows")
    args = ap.parse_args()

    set_seed(args.seed)
    device = args.device
    out_dir = Path(args.save_dir)

    if not args.train_manifests:
        raise SystemExit("Please provide --train-manifests (comma-separated JSONL paths).")
    train_paths = [Path(s.strip()) for s in args.train_manifests.split(",") if s.strip()]
    listings_train = [JsonlList(p) for p in train_paths]

    if args.train_mix:
        ratios = [float(x) for x in args.train_mix.split(",")]
        if len(ratios) != len(listings_train):
            raise SystemExit("--train-mix must match number of train manifests")
    else:
        ratios = [1.0] * len(listings_train)

    ds_trains = [ImageDataset(lst, img_size=args.img, train=True, inner_crop=args.inner_crop) for lst in listings_train]
    mix_epoch_size = sum(len(ds) for ds in ds_trains)
    train_ds = MixedDataset(ds_trains, ratios, epoch_size=mix_epoch_size)

    # Validation (concat if multiple manifests)
    val_listings: List[JsonlList] = []
    if args.val_manifest:
        for p in [Path(s.strip()) for s in args.val_manifest.split(",") if s.strip()]:
            val_listings.append(JsonlList(p))

    if val_listings:
        ds_vals = [ImageDataset(lst, img_size=args.img, train=False, inner_crop=args.inner_crop) for lst in val_listings]
        class ConcatDS(Dataset):
            def __init__(self, parts: List[ImageDataset]):
                self.parts = parts
                self.offsets = []
                off = 0
                for d in parts:
                    self.offsets.append(off); off += len(d)
                self.total = off
            def __len__(self): return self.total
            def __getitem__(self, idx):
                for d, off in zip(self.parts, self.offsets):
                    if idx < off + len(d): return d[idx - off]
                return self.parts[-1][idx - self.offsets[-1]]
        val_ds = ConcatDS(ds_vals)
    else:
        val_ds = None

    pin = device.startswith("cuda")
    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=pin)
    val_ld = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=pin) if val_ds else None

    model = CNN28().to(device)

    if args.warm_start:
        ck = torch.load(args.warm_start, map_location="cpu")
        sd = ck["state_dict"] if "state_dict" in ck else ck
        model.load_state_dict(sd, strict=False)
        print(f"Loaded warm-start weights from {args.warm_start}")

    if args.class_weights == "auto":
        cw = compute_class_weights(listings_train).to(device)
        print("Using auto class weights:", cw.cpu().numpy())
        criterion = nn.CrossEntropyLoss(weight=cw)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = -1.0
    out_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in train_ld:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward(); optimizer.step()
            loss_sum += float(loss.item()) * x.size(0)
            correct += int((logits.argmax(1) == y).sum().item())
            total += x.size(0)
        train_acc = correct / max(1, total)
        train_loss = loss_sum / max(1, total)

        if val_ld is not None:
            model.eval()
            v_total, v_correct = 0, 0
            with torch.no_grad():
                for x, y in val_ld:
                    x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
                    pred = model(x).argmax(1)
                    v_correct += int((pred == y).sum().item()); v_total += x.size(0)
            val_acc = v_correct / max(1, v_total)
            per_cls = per_class_metrics(model, val_ld, device)
            per_cls_str = ", ".join([f"{d}:{per_cls[d]:.1f}%" for d in range(10)])
            print(f"Epoch {ep}/{args.epochs}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  loss={train_loss:.4f}")
            print(f"  Val per-class: {per_cls_str}")
            if val_acc > best_val:
                best_val = val_acc; save_ckpt(model, out_dir, "best.pt")
        else:
            print(f"Epoch {ep}/{args.epochs}  train_acc={train_acc:.4f}  loss={train_loss:.4f}")
            save_ckpt(model, out_dir, f"epoch{ep:02d}.pt")

    print("Done. Best val acc:", f"{best_val:.4f}" if best_val >= 0 else "N/A")

if __name__ == "__main__":
    main()
