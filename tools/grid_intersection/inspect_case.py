"""
Single-image deep dive:
- Load an image (and optional label .npz)
- Run model → A/H/V/J/O
- Build lattice & polylines
- Emit a small gallery: heatmaps, junctions, lattice, polylines, and a few cell Coons meshes
"""

from __future__ import annotations
import argparse, os
from pathlib import Path
import numpy as np
import cv2
import torch

from python.vision.models.grid_intersection_net import GridIntersectionNet
from python.vision.train.grid_intersection.postproc import PPConfig
from tools.grid_intersection.overlay_preds import make_overlays

@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Input image path")
    ap.add_argument("--checkpoint", required=True, help="Model .pt")
    ap.add_argument("--outdir", required=True, help="Output dir for overlays")
    ap.add_argument("--image_size", type=int, default=768)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(args.image)

    S = args.image_size
    inp = cv2.resize(img, (S, S), interpolation=cv2.INTER_AREA)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = GridIntersectionNet(in_channels=1, out_channels=6, base_ch=64).to(device)
    ck = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()

    x = torch.from_numpy(inp[None, None].astype(np.float32)/255.0).to(device)
    y = model(x)

    A = torch.sigmoid(y[:,0:1])[0,0].cpu().numpy().astype(np.float32)
    H = torch.sigmoid(y[:,1:2])[0,0].cpu().numpy().astype(np.float32)
    V = torch.sigmoid(y[:,2:3])[0,0].cpu().numpy().astype(np.float32)
    J = torch.sigmoid(y[:,3:4])[0,0].cpu().numpy().astype(np.float32)
    Ox = y[0,4].cpu().numpy()
    Oy = y[0,5].cpu().numpy()
    mag = np.sqrt(Ox*Ox+Oy*Oy)+1e-6
    O = np.stack([Ox/mag, Oy/mag], axis=2).astype(np.float32)

    overlays = make_overlays(inp, (A*255).astype(np.uint8),
                                  (H*255).astype(np.uint8),
                                  (V*255).astype(np.uint8),
                                  (J*255).astype(np.uint8),
                                  O, PPConfig(), n_cell_meshes=6)

    stem = Path(args.image).stem
    for k, im in overlays.items():
        cv2.imwrite(str(outdir / f"{stem}_{k}.jpg"), im)
    print(f"[inspect_case] wrote overlays → {outdir}")

if __name__ == "__main__":
    main()