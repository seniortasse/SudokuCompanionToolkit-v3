"""
Export predicted polylines to JSON and/or CSV for downstream rectifier,
with optional preview images.

Usage:
  python tools/grid_intersection/export_polylines.py \
    --manifest datasets/grids/real/manifests/val_real.jsonl \
    --checkpoint models/grid_intersection/checkpoints/run_.../best_mje.pt \
    --outdir exports/grid_intersection/polylines \
    --image_size 768 \
    --coord_space original \
    --json --csv \
    --preview_k 12 --preview_dir exports/grid_intersection/polylines/preview
"""

from __future__ import annotations
import argparse, json, os, random
from pathlib import Path
from typing import Dict, List
import numpy as np
import cv2
import torch

from python.vision.models.grid_intersection_net import GridIntersectionNet
from python.vision.train.grid_intersection.postproc import (
    PPConfig, bin_maps, junction_nms, cluster_grid, assemble_polylines
)

def read_jsonl(p: str) -> List[Dict]:
    lst = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                lst.append(json.loads(line))
    return lst

def rescale_polyline(pl: np.ndarray, sx: float, sy: float) -> np.ndarray:
    out = pl.copy().astype(np.float32)
    out[:,0] = out[:,0] / sx
    out[:,1] = out[:,1] / sy
    return out

def draw_polylines_on(img_bgr: np.ndarray, H_lines: List[List[List[float]]], V_lines: List[List[List[float]]]) -> np.ndarray:
    out = img_bgr.copy()
    # H in orange, V in cyan
    colH = (0, 140, 255)
    colV = (255, 200, 0)
    for lines, col in [(H_lines, colH), (V_lines, colV)]:
        for pl in lines:
            pts = np.asarray(pl, dtype=np.float32)
            for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
                cv2.line(out, (int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1))), col, 2, cv2.LINE_AA)
    return out

@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--image_size", type=int, default=768)
    ap.add_argument("--base_ch", type=int, default=64)
    ap.add_argument("--coord_space", choices=["resized", "original"], default="original",
                    help="Export coordinates in resized (S×S) or original image size")
    ap.add_argument("--json", action="store_true", help="Write per-image JSON files")
    ap.add_argument("--csv", action="store_true", help="Write a tall CSV file")
    ap.add_argument("--device", default="cuda")
    # Preview controls
    ap.add_argument("--preview_k", type=int, default=0, help="If >0, save K random preview images")
    ap.add_argument("--preview_dir", default="", help="Directory to write previews (defaults inside outdir)")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    random.seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    json_dir = outdir / "json"; json_dir.mkdir(exist_ok=True)
    csv_path = outdir / "polylines.csv"

    if args.preview_k > 0:
        preview_dir = Path(args.preview_dir) if args.preview_dir else (outdir / "preview")
        preview_dir.mkdir(parents=True, exist_ok=True)
    else:
        preview_dir = None

    # Model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = GridIntersectionNet(in_channels=1, out_channels=6, base_ch=args.base_ch).to(device)
    ck = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()

    pp_cfg = PPConfig()
    recs = read_jsonl(args.manifest)

    # Optionally pick K samples for preview upfront
    preview_indices = set()
    if preview_dir is not None:
        idxs = list(range(len(recs)))
        random.shuffle(idxs)
        preview_indices = set(idxs[:min(args.preview_k, len(idxs))])

    # CSV collector
    csv_rows = [["image_stem","axis","line_index","point_index","x","y"]]

    for i, r in enumerate(recs):
        img = cv2.imread(r["image_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[skip] unreadable {r['image_path']}")
            continue

        H0, W0 = img.shape[:2]
        S = args.image_size
        im_resized = cv2.resize(img, (S,S), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(im_resized[None,None].astype(np.float32)/255.0).to(device)

        # Forward
        y = model(x)
        A  = torch.sigmoid(y[:,0:1])[0,0].cpu().numpy().astype(np.float32)
        Hm = torch.sigmoid(y[:,1:2])[0,0].cpu().numpy().astype(np.float32)
        Vm = torch.sigmoid(y[:,2:3])[0,0].cpu().numpy().astype(np.float32)
        J  = torch.sigmoid(y[:,3:4])[0,0].cpu().numpy().astype(np.float32)
        Ox = y[0,4].cpu().numpy(); Oy = y[0,5].cpu().numpy()
        mag = np.sqrt(Ox*Ox+Oy*Oy)+1e-6
        O  = np.stack([Ox/mag, Oy/mag], axis=2).astype(np.float32)

        # Lattice & polylines in resized coord space
        _ = bin_maps((A*255).astype(np.uint8), (Hm*255).astype(np.uint8), (Vm*255).astype(np.uint8), pp_cfg)
        jpts = junction_nms((J*255).astype(np.uint8), pp_cfg)
        ylv, xlv, grid = cluster_grid(jpts, pp_cfg, S, S)
        H_lines_r, V_lines_r = assemble_polylines(grid, A, O, pp_cfg)  # lists of (N_i×2) in resized space

        # Prepare export copies (possibly converted to original coord space)
        sx = S / float(W0)
        sy = S / float(H0)
        if args.coord_space == "original":
            H_export = [rescale_polyline(np.asarray(pl), sx, sy).tolist() for pl in H_lines_r]
            V_export = [rescale_polyline(np.asarray(pl), sx, sy).tolist() for pl in V_lines_r]
        else:
            H_export = [np.asarray(pl, np.float32).tolist() for pl in H_lines_r]
            V_export = [np.asarray(pl, np.float32).tolist() for pl in V_lines_r]

        # JSON
        if args.json:
            payload = {
                "image_path": r["image_path"],
                "width": int(W0), "height": int(H0),
                "scale_x": sx, "scale_y": sy,
                "coord_space": args.coord_space,
                "H": H_export,
                "V": V_export
            }
            stem = Path(r["image_path"]).stem
            (json_dir / f"{stem}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        # CSV (tall)
        if args.csv:
            stem = Path(r["image_path"]).stem
            for axis, lines in [("H", H_export), ("V", V_export)]:
                for li, pl in enumerate(lines):
                    for pi, (xv, yv) in enumerate(pl):
                        csv_rows.append([stem, axis, str(li), str(pi), f"{xv:.3f}", f"{yv:.3f}"])

        # Preview (draw on original-size image for readability)
        if i in preview_indices and preview_dir is not None:
            # Ensure we draw in original coordinates
            if args.coord_space == "original":
                H_draw, V_draw = H_export, V_export
            else:
                # Convert resized→original for the preview
                H_draw = [rescale_polyline(np.asarray(pl), sx, sy).tolist() for pl in H_lines_r]
                V_draw = [rescale_polyline(np.asarray(pl), sx, sy).tolist() for pl in V_lines_r]

            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            preview = draw_polylines_on(bgr, H_draw, V_draw)
            stem = Path(r["image_path"]).stem
            cv2.imwrite(str(preview_dir / f"{stem}_preview.jpg"), preview)

    # Write CSV once
    if args.csv and len(csv_rows) > 1:
        with open(csv_path, "w", encoding="utf-8") as f:
            for row in csv_rows:
                f.write(",".join(row) + "\n")
        print(f"[export_polylines] CSV → {csv_path}")

    if preview_dir is not None:
        print(f"[export_polylines] previews → {preview_dir}")

    print(f"[export_polylines] done → {outdir}")

if __name__ == "__main__":
    main()