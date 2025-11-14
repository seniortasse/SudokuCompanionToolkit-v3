import argparse
import subprocess
import json
import shutil
import sys          # <-- add this
from pathlib import Path
import cv2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="python/samples", help="Folder with unlabeled photos")
    ap.add_argument("--work", default="python/demo_export", help="Temp per-image exports by rectifier")
    ap.add_argument("--out", default="python/vision/data/real_bootstrap", help="Output folder for 128x128 pairs")
    ap.add_argument("--rectify", default="python/vision/rectify/opencv_rectify.py")
    args = ap.parse_args()

    samples = sorted(list(Path(args.samples).glob("*.*")))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    for sp in samples:
        name = sp.stem
        exp_dir = Path(args.work) / name
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Run your rectifier CLI
        #cmd = ["python", args.rectify, str(sp), str(exp_dir)]
        cmd = [sys.executable, args.rectify, str(sp), str(exp_dir)]
        print("[rectify]", " ".join(cmd))
        subprocess.run(cmd, check=True)

        # Expect board_clean.png and points_10x10.json (from our small patch)
        board = exp_dir / "board_clean.png"
        ptsj  = exp_dir / "points_10x10.json"
        if not board.exists() or not ptsj.exists():
            print(f"[skip] Missing outputs for {sp.name}")
            continue

        # Load and resize to 128, scale points
        im = cv2.imread(str(board), cv2.IMREAD_GRAYSCALE)
        h, w = im.shape[:2]
        im128 = cv2.resize(im, (128,128), interpolation=cv2.INTER_AREA)

        data = json.loads(ptsj.read_text())
        pts = data["points"]  # [ [y,x], ... ] in ROI
        sy, sx = 128.0/h, 128.0/w
        pts128 = [[float(y)*sy, float(x)*sx] for (y,x) in pts]

        # Write pair
        out_png = out / f"{name}.png"
        out_json = out / f"{name}.json"
        cv2.imwrite(str(out_png), im128)
        out_json.write_text(json.dumps({"points": pts128}, indent=2), encoding="utf-8")
        print(f"[ok] {out_png.name}")

    print(f"[done] Bootstrapped into {args.out}")

if __name__ == "__main__":
    main()