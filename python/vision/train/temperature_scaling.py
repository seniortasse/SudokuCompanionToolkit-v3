
# vision/train/temperature_scaling.py
r"""
Fit temperature scaling on a validation manifest and save the temperature (for probability calibration).

Usage (PowerShell):
python ".\vision\train\temperature_scaling.py" ^
  --model ".\vision\train\checkpoints\best.pt" ^
  --val-manifest ".\vision\data\real\meta\val.jsonl" ^
  --img 28 ^
  --device cpu ^
  --out ".\vision\train\calibration.json"

It prints pre-/post-calibration NLL and accuracy, and writes:
{
  "temperature": T
}
You can later apply probabilities as softmax(logits / T).
"""

import argparse, json, sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Ensure vision.* importable
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vision.models.cnn_small import CNN28

class CenterSquare:
    def __call__(self, im: Image.Image):
        w,h = im.size
        if w==h: return im
        side=min(w,h); l=(w-side)//2; t=(h-side)//2
        return im.crop((l,t,l+side,t+side))

class JsonlList:
    def __init__(self, jsonl_path: Path):
        self.items = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line: continue
                obj = json.loads(line)
                self.items.append((str(Path(obj["path"]).resolve()), int(obj["label"])))
        if not self.items: raise ValueError(f"No entries in {jsonl_path}")
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

class ImageDataset(Dataset):
    def __init__(self, listing: JsonlList, img_size:int=28):
        self.listing = listing
        self.tfm = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            CenterSquare(),
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    def __len__(self): return len(self.listing)
    def __getitem__(self,i):
        p, y = self.listing[i]
        im = Image.open(p).convert("L")
        x = self.tfm(im)
        return x, y

def gather_logits(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    logits_list=[]; labels=[]
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            l = model(x).cpu()
            logits_list.append(l)
            labels.append(y)
    return torch.cat(logits_list, dim=0), torch.cat(labels, dim=0)

def nll_from_logits(logits: torch.Tensor, labels: torch.Tensor, T: float=1.0):
    l = logits / T
    return F.cross_entropy(l, labels, reduction="mean").item()

def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor, T: float=1.0):
    l = logits / T
    pred = l.argmax(1)
    return (pred==labels).float().mean().item()

def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, init_T: float=1.0, max_iter:int=200):
    T = torch.tensor([init_T], dtype=torch.float32, requires_grad=True)
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / T.clamp_min(1e-3), labels, reduction="mean")
        loss.backward()
        return loss

    opt.step(closure)
    T_opt = float(T.detach().clamp_min(1e-3).item())
    return T_opt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--val-manifest", required=True)
    ap.add_argument("--img", type=int, default=28)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out", type=str, default="vision/train/calibration.json")
    args = ap.parse_args()

    # load model
    ck = torch.load(args.model, map_location="cpu")
    sd = ck["state_dict"] if "state_dict" in ck else ck
    model = CNN28().to(args.device); model.load_state_dict(sd, strict=False)

    # data
    lst = JsonlList(Path(args.val_manifest))
    ds = ImageDataset(lst, img_size=args.img)
    ld = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)

    logits, labels = gather_logits(model, ld, args.device)
    pre_nll = nll_from_logits(logits, labels, 1.0)
    pre_acc = accuracy_from_logits(logits, labels, 1.0)
    T = fit_temperature(logits, labels, init_T=1.0, max_iter=200)
    post_nll = nll_from_logits(logits, labels, T)
    post_acc = accuracy_from_logits(logits, labels, T)

    print(f"Pre-calibration:  NLL={pre_nll:.4f}  Acc={pre_acc:.4f}")
    print(f"Temperature T*:   {T:.4f}")
    print(f"Post-calibration: NLL={post_nll:.4f}  Acc={post_acc:.4f} (accuracy should not change)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"temperature": T}, indent=2), encoding="utf-8")
    print(f"Wrote calibration -> {out_path}")

if __name__ == "__main__":
    main()
