import json, os, sys, math, glob
from pathlib import Path
import numpy as np
import cv2

def load_points(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    pts = np.array(d["points"], dtype=np.float32).reshape(100,2)  # [y,x]
    return pts

def pck(ptsA, ptsB, thr_px):
    err = np.linalg.norm(ptsA - ptsB, axis=1)
    return float(np.mean(err <= thr_px)), err

def ssim_psnr(a,b):
    # both uint8 64x64
    if a.shape != b.shape: 
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
    # SSIM (simple windowless per Wang 2004 constants)
    a32 = a.astype(np.float32); b32 = b.astype(np.float32)
    muA = a32.mean(); muB = b32.mean()
    varA = a32.var(); varB = b32.var()
    cov  = ((a32 - muA)*(b32 - muB)).mean()
    c1 = (0.01*255)**2; c2 = (0.03*255)**2
    ssim = ((2*muA*muB + c1)*(2*cov + c2))/((muA**2 + muB**2 + c1)*(varA + varB + c2))
    # PSNR
    mse = np.mean((a32 - b32)**2) + 1e-8
    psnr = 10.0 * math.log10((255.0**2)/mse)
    return float(ssim), float(psnr)

def compare_case(case_dir):
    case = Path(case_dir)
    android = case/"android"
    python  = case/"python"
    report = {"case": str(case), "ok": False}

    # geometry
    pa = android/"points_10x10.json"
    pp = python/"points_10x10.json"
    if not pa.exists() or not pp.exists():
        report["error"] = "missing points_10x10.json for one engine"
        return report

    A = load_points(pa)  # [100,2] yx
    B = load_points(pp)
    # threshold: max(5px, min(H,W)/400)
    # infer H,W from board_warped
    bw = cv2.imread(str(android/"board_warped.png"), cv2.IMREAD_GRAYSCALE)
    if bw is None:
        bw = cv2.imread(str(python/"board_warped.png"), cv2.IMREAD_GRAYSCALE)
    H,W = bw.shape[:2]
    thr = max(5.0, min(H,W)/400.0)
    pck_val, perr = pck(A, B, thr)

    # tiles
    good_tiles = 0
    total_tiles = 81
    ssim_grid = np.zeros((9,9), np.float32)
    for r in range(1,10):
        for c in range(1,10):
            na = android/f"cells/r{r}c{c}.png"
            npy = python/f"cells/r{r}c{c}.png"
            ta = cv2.imread(str(na), cv2.IMREAD_GRAYSCALE)
            tb = cv2.imread(str(npy), cv2.IMREAD_GRAYSCALE)
            if ta is None or tb is None:
                continue
            ssim, psnr = ssim_psnr(ta,tb)
            ssim_grid[r-1,c-1] = ssim
            if ssim >= 0.92 or psnr >= 25.0:
                good_tiles += 1

    tiles_ok = (good_tiles >= int(0.90*total_tiles))

    report.update({
        "pck_thr": thr,
        "pck": pck_val,
        "tiles_ok_count": good_tiles,
        "tiles_total": total_tiles,
        "tiles_ok": tiles_ok,
    })
    report["ok"] = (pck_val >= 0.95) and tiles_ok
    return report

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", help="Path to runs/grid_rectification/<case>")
    ap.add_argument("--batch", help="Folder containing many cases")
    args = ap.parse_args()

    cases = []
    if args.case:
        cases = [args.case]
    elif args.batch:
        for d in sorted(glob.glob(os.path.join(args.batch, "*"))):
            if os.path.isdir(d):
                cases.append(d)
    else:
        print("Provide --case or --batch")
        sys.exit(2)

    results = []
    ok_count = 0
    for c in cases:
        rep = compare_case(c)
        results.append(rep)
        print(f"{'PASS' if rep.get('ok') else 'FAIL'}  {c}  PCK={rep.get('pck',0):.3f}  tiles_ok={rep.get('tiles_ok_count',0)}/81")

        if rep.get("ok"):
            ok_count += 1

    summary = {
        "total": len(cases),
        "pass": ok_count,
        "pass_rate": (ok_count/len(cases)) if cases else 0.0
    }
    Path("parity_summary.json").write_text(json.dumps({"cases":results,"summary":summary}, indent=2), encoding="utf-8")
    print(f"\nBatch summary: {ok_count}/{len(cases)} passed  ({summary['pass_rate']*100:.1f}%)")

if __name__ == "__main__":
    main()