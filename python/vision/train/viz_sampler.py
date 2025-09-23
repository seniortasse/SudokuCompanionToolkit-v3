# python/vision/train/viz_sampler.py
import os, argparse, random, sys
from pathlib import Path
import numpy as np
import cv2

# allow importing sibling train_corners.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_corners import synth_sudoku_journey

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--difficulty", type=int, default=5)
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--sigma", type=float, default=1.6)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--force-footer", action="store_true")
    args = ap.parse_args()

    if args.force_footer:
        os.environ["FORCE_FOOTER"] = "1"

    ensure_dir(args.out)
    rng0 = random.Random(args.seed)

    counts = {"footer": 0, "page_frame": 0, "decoy_grid": 0, "neg": 0}

    for i in range(args.n):
        # per-sample RNG seeded from a master stream for diversity
        rng = random.Random(rng0.randint(0, 2**31 - 1))
        inp, hm, xy, meta = synth_sudoku_journey(
            rng,
            img_size=args.img_size,
            difficulty=args.difficulty,
            sigma=args.sigma,
            return_meta=True,
        )

        # count flags (support both 'neg' and legacy 'negative')
        if bool(meta.get("neg", meta.get("negative", False))):
            counts["neg"] += 1
        for k in ("footer", "page_frame", "decoy_grid"):
            if bool(meta.get(k, False)):
                counts[k] += 1

        # save the grayscale input for a quick look
        g = (inp[0] * 255.0).astype(np.uint8)  # [H,W]
        cv2.imwrite(str(Path(args.out) / f"sample_{i:03d}.png"), g)

    print(f"Saved {args.n} samples to: {args.out}")
    for k in ("footer", "page_frame", "decoy_grid"):
        v = counts[k]
        print(f"{k:>10}: {v:3d} / {args.n}  ({100.0*v/args.n:5.1f}%)")

if __name__ == "__main__":
    main()