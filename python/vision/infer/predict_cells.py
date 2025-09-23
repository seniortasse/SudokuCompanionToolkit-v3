
# vision/infer/predict_cells.py
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Ensure repo root on path
_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
from torchvision import transforms

from vision.models.cnn_small import CNN28

class CenterSquare:
    def __call__(self, im: Image.Image):
        w,h = im.size
        if w==h: return im
        side=min(w,h); l=(w-side)//2; t=(h-side)//2
        return im.crop((l,t,l+side,t+side))

class CenterFrac:
    def __init__(self, frac: float = 1.0):
        self.frac = float(frac)
    def __call__(self, im: Image.Image):
        f = self.frac
        if f >= 0.999: return im
        w,h = im.size
        side = min(w,h)
        keep = max(1, int(round(side * f)))
        l = (w - keep)//2; t = (h - keep)//2
        return im.crop((l,t,l+keep,t+keep))

def make_tfm(img_size:int=28, inner:float=1.0):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        CenterSquare(),
        CenterFrac(inner),
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

def load_temperature(calib_path: str | None) -> float:
    if not calib_path: return 1.0
    try:
        obj = json.loads(Path(calib_path).read_text(encoding="utf-8"))
        T = float(obj.get("temperature", 1.0))
        return 1.0 if T <= 0 else T
    except Exception:
        return 1.0

def list_board_cells(root: Path) -> List[Tuple[str, List[Path]]]:
    boards = []
    for cells_dir in root.rglob("cells"):
        board = cells_dir.parent.name
        paths = []
        for r in range(1,10):
            for c in range(1,10):
                p = None
                for ext in (".png", ".jpg", ".jpeg"):
                    cand = cells_dir / f"r{r}c{c}{ext}"
                    if cand.exists(): p = cand; break
                if p is None: paths = []; break
                paths.append(p)
            if not paths: break
        if len(paths)==81:
            boards.append((board, paths))
    return sorted(boards, key=lambda t: t[0])

def render_grid(rows: List[Dict[str,Any]], out_png: Path):
    W,H = 9*64+8*4, 9*64+8*4+40
    im = Image.new("RGB",(W,H),(255,255,255))
    d = ImageDraw.Draw(im)
    try:
        font_big = ImageFont.truetype("DejaVuSans.ttf", 30)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 14)
        font_footer = ImageFont.truetype("DejaVuSans.ttf", 13)
    except Exception:
        font_big = font_small = font_footer = ImageFont.load_default()

    for row in rows:
        r,c = row["r"]-1, row["c"]-1
        x = c*(64+4); y = r*(64+4)
        low = bool(row["low_conf"])
        if low: d.rectangle([x,y,x+64,y+64], fill=(255,230,230))
        d.rectangle([x,y,x+64,y+64], outline=(255,0,0) if low else (0,0,0), width=2 if low else 1)

        txt = str(row["pred"])
        bbox = d.textbbox((0,0), txt, font=font_big)
        d.text((x + (64-(bbox[2]-bbox[0]))//2, y + 6 + (64-(bbox[3]-bbox[1]))//2), txt, fill=(200,0,0) if low else (0,0,0), font=font_big)
        ptxt = f"{row['prob']:.2f}"
        bbox2 = d.textbbox((0,0), ptxt, font=font_small)
        d.text((x + 64 - bbox2[2] - 4, y + 64 - bbox2[3] - 2), ptxt, fill=(200,0,0) if low else (0,0,0), font=font_small)
        alt = f"alt {row['top2']}:{row['prob2']:.2f}"
        d.text((x+4, y+2), alt, fill=(80,80,80), font=font_small)

    meta = rows[0].get("_meta", {})
    footer = f"T={meta.get('temperature',1.0):.3f}  low={meta.get('low')}  margin={meta.get('margin')}  inner={meta.get('inner_crop')}  model={meta.get('model_name','')}"
    d.text((4, 9*64+8*4 + 8), footer, fill=(0,0,0), font=font_footer)
    im.save(out_png)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Root containing many boards/**/cells")
    ap.add_argument("--model", required=True)
    ap.add_argument("--img", type=int, default=28)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--calib", type=str, default="")
    ap.add_argument("--low", type=float, default=0.85)
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--inner-crop", type=float, default=1.0, help="Center-crop fraction before resize (e.g., 0.9)")
    ap.add_argument("--out", type=str, default="runs/infer")
    args = ap.parse_args()

    T = load_temperature(args.calib or None)
    tfm = make_tfm(args.img, inner=args.inner_crop)

    device = args.device
    model = CNN28().to(device)
    ck = torch.load(args.model, map_location="cpu")
    sd = ck["state_dict"] if "state_dict" in ck else ck
    model.load_state_dict(sd, strict=False)
    model.eval()

    root = Path(args.src)
    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)

    boards = list_board_cells(root)
    all_rows = []
    low_total = 0

    for board, paths in boards:
        rows = []
        for i,p in enumerate(paths):
            r,c = i//9+1, i%9+1
            im = Image.open(p).convert("L")
            x = tfm(im).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(x)
                if T != 1.0: logits = logits / T
                prob = F.softmax(logits, dim=1).cpu().numpy().squeeze(0)
            order = np.argsort(prob)
            k1,k2 = int(order[-1]), int(order[-2])
            p1,p2 = float(prob[k1]), float(prob[k2])
            m = p1-p2
            low_by_prob = int(p1 < args.low)
            low_by_margin = int(m < args.margin)
            low_conf = int(low_by_prob or low_by_margin)
            if low_conf: low_total += 1

            rows.append({
                "board": board, "r": r, "c": c, "path": str(p),
                "pred": k1, "prob": p1, "top2": k2, "prob2": p2, "margin": m,
                "low_conf": low_conf, "low_by_prob": low_by_prob, "low_by_margin": low_by_margin,
                "_meta": {"temperature": T, "low": args.low, "margin": args.margin, "inner_crop": args.inner_crop,
                          "model_name": Path(args.model).name}
            })
            all_rows.append(rows[-1])

        # per-board outputs
        board_dir = out_root / board; board_dir.mkdir(exist_ok=True, parents=True)
        # csv
        with open(board_dir / f"{board}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[k for k in rows[0].keys() if k!="_meta"])
            w.writeheader(); w.writerows([{k:v for k,v in r.items() if k!="_meta"} for r in rows])
        # png
        render_grid(rows, board_dir / f"{board}_grid.png")

        # log
        lows = sum(r["low_conf"] for r in rows)
        print(f"{board}: {lows} low-confidence cells out of {len(rows)}")

    # global CSV
    try:
        import pandas as pd
        df = pd.DataFrame([{k:v for k,v in r.items() if k!="_meta"} for r in all_rows])
        df.to_csv(out_root / "all_preds.csv", index=False, encoding="utf-8")
    except Exception:
        # fallback CSV writer if pandas not available
        import csv as _csv
        with open(out_root / "all_preds.csv", "w", newline="", encoding="utf-8") as f:
            keys = [k for k in all_rows[0].keys() if k!="_meta"]
            w = _csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in all_rows:
                w.writerow({k:v for k,v in r.items() if k!="_meta"})

    # meta.json
    meta = {
        "script_version": "predict_cells_v2",
        "src": str(root.resolve()),
        "out": str(out_root.resolve()),
        "model": str(Path(args.model).resolve()),
        "img": args.img, "device": args.device,
        "calibration_path": args.calib or "", "temperature": T,
        "thresholds": {"low": args.low, "margin": args.margin},
        "inner_crop": args.inner_crop,
        "boards": len(boards), "tiles": len(all_rows),
        "low_total": int(low_total), "low_rate": float(low_total)/max(1,len(all_rows))
    }
    (out_root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"TOTAL: {low_total} low-confidence cells out of {len(all_rows)}")

if __name__ == "__main__":
    main()
