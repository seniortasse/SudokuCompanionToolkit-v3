
# tools/mining/mine_pairs.py
r"""
Mine ambiguous tiles for specified confusing pairs (e.g., 7↔1, 5↔6, 8↔3).

Usage (PowerShell):
python ".\tools\mining\mine_pairs.py" ^
  --src ".\demo_export" ^
  --model ".\vision\train\checkpoints\best.pt" ^
  --out ".\vision\data\pairs_focus" ^
  --pairs "7,1 5,6 8,3" ^
  --img 28 ^
  --hi-thresh 0.95 ^
  --margin 0.10 ^
  --loose-thresh 0.70

Selection logic (union of conditions):
A) Top-2 are the pair and the margin is small:  (top1 in {a,b} AND top2 is the other) AND (top1 - top2 < --margin)
B) Top1 is in the pair but not very confident: (top1 in {a,b}) AND (prob_top1 < --hi-thresh)
C) Globally low confidence: (max_prob < --loose-thresh)

Outputs:
- Copies selected tiles into: <out>/queue/<pair_name>/
  where pair_name is like "7_1". Filenames include a short hash to avoid collisions.
"""

import argparse, hashlib, sys, re
from pathlib import Path
from typing import Dict, Tuple, List, Set

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

# Ensure vision.* is importable regardless of CWD
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vision.models.cnn_small import CNN28

SUPP = {".png", ".jpg", ".jpeg"}

class CenterSquare:
    def __call__(self, im: Image.Image):
        w,h = im.size
        if w==h: return im
        side = min(w,h)
        l=(w-side)//2; t=(h-side)//2
        return im.crop((l,t,l+side,t+side))

def tfm(img_size:int=28):
    from torchvision import transforms as T
    return T.Compose([
        T.Grayscale(num_output_channels=1),
        CenterSquare(),
        T.Resize((img_size,img_size)),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,)),
    ])

def parse_pairs(arg: str) -> List[Tuple[int,int]]:
    pairs = []
    for tok in arg.strip().split():
        if "," in tok:
            a,b = tok.split(",")
        elif "/" in tok:
            a,b = tok.split("/")
        elif "-" in tok:
            a,b = tok.split("-")
        else:
            continue
        pairs.append((int(a), int(b)))
    # dedup symmetrical
    uniq=set()
    out=[]
    for a,b in pairs:
        key = tuple(sorted((a,b)))
        if key not in uniq:
            uniq.add(key); out.append((a,b))
    return out

def find_tiles(src: Path) -> List[Path]:
    return [p for p in src.rglob("*") if p.suffix.lower() in SUPP]

def short_hash(p: Path) -> str:
    h = hashlib.md5(str(p.resolve()).encode("utf-8")).hexdigest()[:8]
    return h

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Root folder with boards (recurses to find tiles).")
    ap.add_argument("--model", required=True, help="Path to best.pt")
    ap.add_argument("--out", required=True, help="Output root for queue")
    ap.add_argument("--pairs", required=True, help='Space-separated list of pairs, e.g. "7,1 5,6 8,3"')
    ap.add_argument("--img", type=int, default=28)
    ap.add_argument("--hi-thresh", type=float, default=0.95)
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--loose-thresh", type=float, default=0.70)
    args = ap.parse_args()

    pairs = parse_pairs(args.pairs)
    if not pairs:
        print("No valid pairs parsed from --pairs"); return
    pair_set: Set[Tuple[int,int]] = set(tuple(sorted(x)) for x in pairs)

    # load model
    ck = torch.load(args.model, map_location="cpu")
    sd = ck["state_dict"] if "state_dict" in ck else ck
    model = CNN28(); model.load_state_dict(sd, strict=False); model.eval()

    transform = tfm(args.img)
    tiles = find_tiles(Path(args.src))
    if not tiles:
        print("No tiles found under", args.src); return

    out_root = Path(args.out) / "queue"
    total=0; queued=0
    for p in tiles:
        try:
            im = Image.open(p).convert("L")
        except Exception:
            continue
        x = transform(im).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
            prob = F.softmax(logits, dim=1).numpy().squeeze(0)  # [10]
        order = np.argsort(prob)
        top1 = int(order[-1]); top2 = int(order[-2])
        p1 = float(prob[top1]); p2 = float(prob[top2])
        total += 1

        condA = (tuple(sorted((top1, top2))) in pair_set) and ((p1 - p2) < args.margin)
        condB = ((top1, top2) in pairs or (top2, top1) in pairs) and (p1 < args.hi_thresh)
        condC = (p1 < args.loose_thresh)

        if condA or condB or condC:
            key = tuple(sorted((top1, top2)))
            # choose an existing pair folder name, order normalized small_big
            pa, pb = key
            pair_name = f"{pa}_{pb}"
            out_dir = out_root / pair_name
            out_dir.mkdir(parents=True, exist_ok=True)
            # stable, de-duplicated name
            out_file = out_dir / f"{p.stem}_{short_hash(p)}{p.suffix.lower()}"
            try:
                from shutil import copy2
                copy2(p, out_file)
                queued += 1
            except Exception:
                pass

    print(f"Scanned {total} tiles; queued {queued} tiles across pairs -> {out_root}")

if __name__ == "__main__":
    main()
