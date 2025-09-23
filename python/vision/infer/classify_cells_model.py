# vision/infer/classify_cells_model.py
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Dict, Any, List

# Ensure repo root on path
_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import hashlib, pathlib

from vision.models.cnn_small import CNN28  # used for .pt checkpoints





# ---------- image transforms ----------

class FlattenAlphaToWhite:
    def __call__(self, im: Image.Image) -> Image.Image:
        # Convert P/LA/RGBA (with possible transparency) to RGB over white
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im)
            im = im.convert("RGB")
        else:
            im = im.convert("RGB")
        return im
    
class CenterSquare:
    def __call__(self, im: Image.Image):
        w, h = im.size
        if w == h:
            return im
        side = min(w, h)
        l = (w - side) // 2
        t = (h - side) // 2
        return im.crop((l, t, l + side, t + side))

class CenterFrac:
    """Center-crop to a given fraction of the (square) image side, e.g., 0.9 keeps the central 90%."""
    def __init__(self, frac: float = 1.0):
        self.frac = float(frac)
    def __call__(self, im: Image.Image):
        f = self.frac
        if f >= 0.999:  # no-op
            return im
        w, h = im.size
        side = min(w, h)
        keep = max(1, int(round(side * f)))
        l = (w - keep) // 2
        t = (h - keep) // 2
        return im.crop((l, t, l + keep, t + keep))

def tfm(img_size:int=28, inner:float=1.0):
    return transforms.Compose([
        FlattenAlphaToWhite(),           # <-- new: match Android "composite over white"
        transforms.Grayscale(num_output_channels=1),
        CenterSquare(),
        CenterFrac(inner),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

# ---------- calibration & model helpers ----------
def _is_torchscript(path: str) -> bool:
    p = str(path).lower()
    return p.endswith(".ptl") or p.endswith(".jit")

def _load_calibration(calib_path: str | None) -> tuple[str, float]:
    """
    Returns (mode, T_effective).

    If calibration.json has {"mode":"logits","temperature":T_raw}, use T_effective = T_raw.
    If it has {"mode":"prob","temperature":T_raw}, that T was fit on probabilities, so invert
    to operate on logits:  T_effective = 1.0 / T_raw.
    Missing/invalid file -> ("logits", 1.0)
    """
    mode = "logits"
    T_eff = 1.0
    if not calib_path:
        return mode, T_eff
    try:
        obj = json.loads(Path(calib_path).read_text(encoding="utf-8"))
        raw_T = float(obj.get("temperature", 1.0))
        mode = str(obj.get("mode", "logits")).lower()
        if raw_T <= 0:
            T_eff = 1.0
        else:
            T_eff = raw_T if mode == "logits" else (1.0 / raw_T)
    except Exception:
        mode, T_eff = "logits", 1.0
    return mode, T_eff

def _softmax_with_temp(logits: torch.Tensor, T: float) -> torch.Tensor:
    """logits [N,10] -> probs [N,10] using a single temperature-scaled softmax."""
    if T <= 0:
        T = 1.0
    return F.softmax(logits / T, dim=1)

def _load_model_any(model_path: str, device: str = "cpu"):
    """
    Load either TorchScript (.ptl/.jit) or a standard PyTorch checkpoint (.pt).
    Returns a module that outputs LOGITS for 10 classes.
    """
    mp = str(model_path)
    if _is_torchscript(mp):
        m = torch.jit.load(mp, map_location=device)
        m.eval().to(device)
        return m

    # Regular checkpoint path
    m = CNN28(num_classes=10)  # adjust if your ctor differs
    ck = torch.load(mp, map_location="cpu")
    # support a few common layouts
    if isinstance(ck, dict):
        if "model" in ck and isinstance(ck["model"], dict):
            sd = ck["model"]
        elif "state_dict" in ck and isinstance(ck["state_dict"], dict):
            sd = ck["state_dict"]
        else:
            sd = ck
    else:
        sd = ck
    m.load_state_dict(sd, strict=False)
    m.to(device).eval()
    return m

# ---------- IO ----------
def load_tiles(folder: Path) -> List[Path]:
    """Load exactly r1c1..r9c9 (png/jpg/jpeg). Raises if missing."""
    paths = []
    for r in range(1, 10):
        for c in range(1, 10):
            p = None
            for ext in (".png", ".jpg", ".jpeg"):
                cand = folder / f"r{r}c{c}{ext}"
                if cand.exists():
                    p = cand
                    break
            if p is None:
                raise FileNotFoundError(f"Missing tile for r{r}c{c} in {folder}")
            paths.append(p)
    return paths

# ---------- main API ----------
def predict_folder(
    folder: str,
    model_path: str,
    img_size: int = 28,
    device: str = "cpu",
    calib: str | None = None,
    low: float | None = None,
    margin_thr: float | None = None,
    inner_crop: float = 1.0
) -> Dict[str, Any]:
    """
    Classify a canonical 9x9 folder; return top1 grid plus probabilities and (optional) low-confidence masks.
    Uses ONE temperature-scaled softmax, matching the mobile pipeline.
    """
    # TO DELETE LATER - JUST FOR A QUICK TEST
    p = pathlib.Path("C:/Users/17347/desktop/contextionary/1- sudoku companion/sudokucompaniontoolkit v3/artifacts/cell_cnn28_logits.ptl")
    print("compute SHA-256 of the .ptl I use:")
    print(hashlib.sha256(p.read_bytes()).hexdigest())
    # DELETE THE ABOVE
    device = str(device)
    model = _load_model_any(model_path, device=device)
    mode, T_eff = _load_calibration(calib)

    folder = Path(folder)
    paths = load_tiles(folder)
    tr = tfm(img_size, inner=inner_crop)

    grid   = np.zeros((9, 9), dtype=int)
    probs  = np.zeros((9, 9), dtype=float)   # top-1 prob
    top2   = np.zeros((9, 9), dtype=int)     # second-best class
    prob2  = np.zeros((9, 9), dtype=float)   # second-best prob
    margin = np.zeros((9, 9), dtype=float)   # top1 - top2

    low_by_prob   = np.zeros((9, 9), dtype=int)
    low_by_margin = np.zeros((9, 9), dtype=int)
    low_conf      = np.zeros((9, 9), dtype=int)

    for i, p in enumerate(paths):
        r, c = i // 9, i % 9
        im = Image.open(p).convert("L")
        x = tr(im).unsqueeze(0).to(device)  # [1,1,H,W]

        with torch.no_grad():
            logits = model(x)               # [1,10] (TorchScript or nn.Module)
            prob_t = _softmax_with_temp(logits, T_eff)  # single softmax with effective T
            prob = prob_t.cpu().numpy().squeeze(0)      # [10]

        order = np.argsort(prob)
        k1, k2 = int(order[-1]), int(order[-2])
        p1, p2 = float(prob[k1]), float(prob[k2])
        m = p1 - p2

        grid[r, c]   = k1
        probs[r, c]  = p1
        top2[r, c]   = k2
        prob2[r, c]  = p2
        margin[r, c] = m

        # thresholds (if provided)
        lbp = int(low is not None and p1 < low)
        lbm = int(margin_thr is not None and m < margin_thr)
        lcf = int(lbp or lbm)

        low_by_prob[r, c]   = lbp
        low_by_margin[r, c] = lbm
        low_conf[r, c]      = lcf

    out = {
        "grid": grid.tolist(),
        "probs": probs.tolist(),
        "top2": top2.tolist(),
        "prob2": prob2.tolist(),
        "margin": margin.tolist(),
        "low_conf": low_conf.tolist(),
        "low_by_prob": low_by_prob.tolist(),
        "low_by_margin": low_by_margin.tolist(),
        "paths": [str(p) for p in paths],
        "meta": {
            "model": str(Path(model_path).resolve()),
            "img": int(img_size),
            "device": device,
            "calibration_path": calib or "",
            "calibration_mode": mode,
            "temperature_effective": float(T_eff),
            "thresholds": {
                "low": float(low) if low is not None else None,
                "margin": float(margin_thr) if margin_thr is not None else None
            },
            "inner_crop": float(inner_crop),
        }
    }
    return out

# ---------- CLI ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Folder with r#c#.png tiles")
    ap.add_argument("--model", required=True, help=".pt (checkpoint) or .ptl/.jit (TorchScript)")
    ap.add_argument("--img", type=int, default=28)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--calib", type=str, default="", help="Path to calibration.json (with {'temperature': T, 'mode': 'logits'|'prob'})")
    ap.add_argument("--low", type=float, default=None, help="Flag low if top-1 prob < this")
    ap.add_argument("--margin", type=float, default=None, help="Flag low if (top1 - top2) < this")
    ap.add_argument("--inner-crop", type=float, default=1.0, help="Center-crop fraction (e.g., 0.9 keeps central 90%)")
    ap.add_argument("--out", type=str, default="runs/classify_folder.json")
    args = ap.parse_args()

    out = predict_folder(
        args.src, args.model,
        img_size=args.img, device=args.device,
        calib=args.calib or None,
        low=args.low, margin_thr=args.margin,
        inner_crop=args.inner_crop
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Wrote:", out_path)