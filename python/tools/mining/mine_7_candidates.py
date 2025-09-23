
# tools/mining/mine_7_candidates.py
"""
Mine tiles that are likely 7s (or confused with 1) so you can label just those.

Strategy
- Load your current model (CNN28).
- Recursively scan --src for image tiles (*.png/jpg/jpeg).
- Apply the same preprocessing as training (center square, resize, grayscale, normalize).
- Select candidates if any of the following holds:
  A) pred == 7 and prob7 < hi_thresh  (uncertain 7s)
  B) pred == 1 and prob7 > lo_thresh  (likely confused with 7)
  C) max_prob < loose_thresh          (overall uncertain)
  D) top-2 are {7,1} and |p7 - p1| < margin

Copies selected tiles into --out/queue preserving a unique name, so you can run the labeler on that folder only.
"""

import argparse, hashlib, sys
from pathlib import Path
from typing import Iterable
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

# --- Make 'vision' importable regardless of current working dir ---
REPO_ROOT = Path(__file__).resolve().parents[2]  # SudokuCompanionToolkit/
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vision.models.cnn_small import CNN28  # now resolves

SUPP_SUFFIXES = {".png", ".jpg", ".jpeg"}

class CenterSquare:
    def __call__(self, im: Image.Image) -> Image.Image:
        w, h = im.size
        if w == h:
            return im
        side = min(w, h)
        l = (w - side) // 2
        t = (h - side) // 2
        return im.crop((l, t, l+side, t+side))

def build_transform(img_size: int = 28):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        CenterSquare(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

def iter_images(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.suffix.lower() in SUPP_SUFFIXES:
            yield p

def unique_name(p: Path) -> str:
    # Stable name including a short hash to avoid collisions
    h = hashlib.md5(str(p.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{p.stem}_{h}{p.suffix.lower()}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Folder containing many boards (recurses).")
    ap.add_argument("--model", required=True, help="Path to best.pt (state_dict or whole state).")
    ap.add_argument("--out", required=True, help="Output folder where 'queue' will be created.")
    ap.add_argument("--img", type=int, default=28)
    # thresholds
    ap.add_argument("--hi-thresh", type=float, default=0.95, help="If pred==7 and prob7 < hi_thresh -> keep (uncertain 7).")
    ap.add_argument("--lo-thresh", type=float, default=0.15, help="If pred==1 and prob7 > lo_thresh -> keep (likely 7).")
    ap.add_argument("--loose-thresh", type=float, default=0.70, help="If max prob < loose_thresh -> keep (uncertain).")
    ap.add_argument("--margin", type=float, default=0.10, help="If top-2 are {7,1} and |p7-p1| < margin -> keep.")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    queue = out / "queue"; queue.mkdir(parents=True, exist_ok=True)

    # Load model
    ck = torch.load(args.model, map_location="cpu")
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    model = CNN28()
    model.load_state_dict(sd, strict=False)
    model.eval()

    tfm = build_transform(args.img)

    kept = 0
    total = 0
    for p in iter_images(src):
        total += 1
        try:
            im = Image.open(p).convert("L")
        except Exception:
            continue
        x = tfm(im).unsqueeze(0)  # 1x1xHxW
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy().squeeze(0)  # [10]
            pred = int(np.argmax(probs))
        p7 = float(probs[7]); p1 = float(probs[1]); m = float(np.max(probs))

        keep = False
        # A) pred is 7 but uncertain
        if pred == 7 and p7 < args.hi_thresh:
            keep = True
        # B) pred is 1 but prob 7 is not tiny
        elif pred == 1 and p7 > args.lo_thresh:
            keep = True
        # C) overall uncertain
        elif m < args.loose_thresh:
            keep = True
        else:
            # D) top-2 are {7,1} and close
            top2 = np.argsort(probs)[-2:]
            if set(top2) == {1,7} and abs(p7 - p1) < args.margin:
                keep = True

        if keep:
            dst = queue / unique_name(p)
            try:
                im.save(dst)
                kept += 1
            except Exception:
                pass

    print(f"Scanned {total} tiles; queued {kept} likely-7/ambiguous tiles -> {queue}")

if __name__ == "__main__":
    main()
