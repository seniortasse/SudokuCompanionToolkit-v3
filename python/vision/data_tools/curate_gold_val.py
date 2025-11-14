import argparse, json, random
from pathlib import Path
import cv2
import numpy as np

def draw_points(im, pts):
    vis = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    for (y,x) in pts:
        cv2.circle(vis, (int(x), int(y)), 2, (0,0,255), -1)
    return vis

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="python/vision/data/real_bootstrap", help="Where bootstrapped pairs live")
    ap.add_argument("--out",  default="python/vision/data/gold_val", help="Accepted gold pairs go here")
    ap.add_argument("--sample", type=int, default=400)
    args = ap.parse_args()

    pool = Path(args.pool)
    out  = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    imgs = sorted(pool.glob("*.png"))
    random.shuffle(imgs)
    imgs = imgs[:args.sample]

    print("[info] Controls: 'y' accept, 'n' reject, 'q' quit")
    for ip in imgs:
        jp = ip.with_suffix(".json")
        if not jp.exists(): continue
        im = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        pts = json.loads(jp.read_text())["points"]

        vis = draw_points(im, pts)
        cv2.imshow("check (y/n/q)", vis)
        k = cv2.waitKey(0) & 0xFF
        if k in (ord('q'), 27): break
        elif k == ord('y'):
            # copy to gold
            (out / ip.name).write_bytes(ip.read_bytes())
            (out / jp.name).write_text(json.dumps({"points": pts}, indent=2), encoding="utf-8")
            print("[accepted]", ip.name)
        else:
            print("[rejected]", ip.name)
    cv2.destroyAllWindows()
    print(f"[done] Gold val in {args.out}")

if __name__ == "__main__":
    main()