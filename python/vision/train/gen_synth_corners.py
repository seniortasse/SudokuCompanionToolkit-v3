import argparse, csv, os, random
from pathlib import Path
import numpy as np
import torch
import cv2

# Reuse your generator pieces from train_corners.py
from train_corners import (
    ensure_dir,
    SyntheticIterable,               # iterable wrapper over synth_sudoku_journey
    collate_corners_batch,           # keeps meta lists aligned
)

def main():
    ap = argparse.ArgumentParser("Export synthetic Sudoku corner samples")
    ap.add_argument("--out", required=True, help="Output folder for this difficulty (e.g. datasets/synth/D3)")
    ap.add_argument("--count", type=int, default=1000, help="Number of samples to export")
    ap.add_argument("--difficulty", type=int, required=True, help="Generator difficulty (3–7)")
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--sigma", type=float, default=1.6)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--save-hm", action="store_true", help="Also save heatmaps as .npy (float32 in [0,1])")
    ap.add_argument("--save-meta", action="store_true", help="Also save header/footer rect meta as .npz")
    ap.add_argument("--save-overlay", action="store_true", help="Also save debug overlay PNGs")
    args = ap.parse_args()

    out_dir = Path(args.out)
    ensure_dir(out_dir)

    # Subfolders
    img_dir  = out_dir / "images"
    hm_dir   = out_dir / "heatmaps"
    meta_dir = out_dir / "meta"
    ov_dir   = out_dir / "overlays"
    ensure_dir(img_dir)
    if args.save_hm:      ensure_dir(hm_dir)
    if args.save_meta:    ensure_dir(meta_dir)
    if args.save_overlay: ensure_dir(ov_dir)

    # annotations.csv: filename,x0,y0,x1,y1,x2,y2,x3,y3
    ann_csv = out_dir / "annotations.csv"
    with open(ann_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename","x0","y0","x1","y1","x2","y2","x3","y3"])

    # Build an iterable synthetic stream
    # return_meta=True ensures we get header/footer rect lists if present
    ds = SyntheticIterable(
        img_size=args.img_size,
        sigma=args.sigma,
        difficulty=args.difficulty,
        seed=args.seed,
        neg_frac=0.0,                 # for a labeled dataset, you usually want positives
        occlusion_prob=0.0,           # set >0.0 if you want occluded samples
        return_meta=True,
    )

    loader = torch.utils.data.DataLoader(
        ds, batch_size=32, num_workers=0, collate_fn=collate_corners_batch
    )

    saved = 0
    rng = random.Random(args.seed)

    def denorm_to_u8(x: np.ndarray) -> np.ndarray:
        """
        x is float32 in [0,1] (model input convention).
        Returns HxW uint8 (0..255).
        Accepts [H,W], [1,H,W], or [H,W,1].
        """
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 3 and x.shape[0] == 1:
            x = x[0]
        if x.ndim == 3 and x.shape[-1] == 1:
            x = x[..., 0]
        x = np.clip(x * 255.0, 0.0, 255.0)
        return x.astype(np.uint8)

    for batch in loader:
        # Accept either (inp, hm, xy, meta) or (inp, hm, xy)
        if isinstance(batch, (list, tuple)) and len(batch) == 4:
            inp, gt_hm, gt_xy, meta = batch
        else:
            inp, gt_hm, gt_xy = batch
            meta = {"header_text_rects_list": [[]]*len(inp), "footer_text_rects_list": [[]]*len(inp)}

        # NOTE: inputs/heatmaps are float in [0,1]; xy are pixel coords
        inp   = inp.detach().cpu().numpy()     # [B,1,H,W], float32 in [0,1]
        gt_xy = gt_xy.detach().cpu().numpy()   # [B,4,2], float32 pixel coordinates
        gt_hm = gt_hm.detach().cpu().numpy()   # [B,4,H,W], float32 in [0,1]

        B = inp.shape[0]
        for i in range(B):
            if saved >= args.count:
                break

            # ----- image -----
            img8  = denorm_to_u8(inp[i])                  # HxW uint8
            fname = f"{saved:06d}.png"
            cv2.imwrite(str(img_dir / fname), img8)

            # ----- heatmaps (optional; keep as float [0,1]) -----
            if args.save_hm:
                np.save(hm_dir / f"{saved:06d}.npy", gt_hm[i].astype(np.float32))

            # ----- meta (optional) -----
            if args.save_meta:
                header_list = meta.get("header_text_rects_list", [])
                footer_list = meta.get("footer_text_rects_list", [])
                header_rects = header_list[i] if i < len(header_list) else []
                footer_rects = footer_list[i] if i < len(footer_list) else []
                np.savez_compressed(
                    meta_dir / f"{saved:06d}.npz",
                    header=np.array(header_rects, dtype=np.int32),
                    footer=np.array(footer_rects, dtype=np.int32)
                )

            # ----- overlay (optional debug) -----
            if args.save_overlay:
                ov = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
                # draw gt corners
                xy = gt_xy[i]  # [4,2] in pixel coords
                colors = [(0,0,255),(0,255,0),(255,0,0),(0,255,255)]  # TL, TR, BR, BL (or your convention)
                for k,(x,y) in enumerate(xy):
                    if x >= 0 and y >= 0:
                        cv2.circle(ov, (int(round(x)), int(round(y))), 2, colors[k], -1, lineType=cv2.LINE_AA)
                cv2.imwrite(str(ov_dir / fname), ov)

            # ----- append to CSV -----
            with open(ann_csv, "a", newline="") as f:
                w = csv.writer(f)
                x0,y0 = gt_xy[i,0,0], gt_xy[i,0,1]
                x1,y1 = gt_xy[i,1,0], gt_xy[i,1,1]
                x2,y2 = gt_xy[i,2,0], gt_xy[i,2,1]
                x3,y3 = gt_xy[i,3,0], gt_xy[i,3,1]
                w.writerow([fname, f"{x0:.3f}", f"{y0:.3f}", f"{x1:.3f}", f"{y1:.3f}",
                            f"{x2:.3f}", f"{y2:.3f}", f"{x3:.3f}", f"{y3:.3f}"])

            saved += 1
        if saved >= args.count:
            break

    print(f"✅ Wrote {saved} samples to {out_dir}")
    print(f"   - images/: PNGs (uint8)")
    if args.save_hm:      print("   - heatmaps/: per-sample 4xHxW .npy (float32 in [0,1])")
    if args.save_meta:    print("   - meta/: header/footer rects .npz")
    if args.save_overlay: print("   - overlays/: corner overlays")
    print(f"   - annotations.csv (filename + 4 corner (x,y))")

if __name__ == "__main__":
    main()