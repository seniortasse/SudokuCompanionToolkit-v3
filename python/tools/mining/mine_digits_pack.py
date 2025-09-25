
# tools/mining/mine_digits_pack.py
"""
Mine "interesting" tiles for specific digits (or a strict pair) from one or more all_preds.csv files.

Usage (PowerShell):
  python ".\tools\mining\mine_digits_pack.py" ^
    --preds ".\runs\infer_loop\all_preds.csv" ^
    --digits "3,1" ^
    --out ".\vision\data\mine_3_1" ^
    --val-every 10 ^
    --include-low ^
    --wide-margin 0.15 ^
    --pair-only

Selection:
- Base: (pred in digits OR top2 in digits)
- Hardness gate: (low_conf == 1) OR (margin < wide_margin)
- With --pair-only: require EXACTLY two digits and keep only rows whose {pred, top2} == {d1, d2} (in any order), with pred != top2.

Output:
  <out>/queue/train/*.png, <out>/queue/val/*.png  (stable names: <board>_r#c#.png)
"""
from __future__ import annotations
import argparse, shutil, sys
from pathlib import Path
import pandas as pd

def copy_rows(rows: pd.DataFrame, out_root: Path, val_every:int=10) -> int:
    qdir = out_root / "queue"
    train_dir = qdir / "train"; val_dir = qdir / "val"
    train_dir.mkdir(parents=True, exist_ok=True); val_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for _, row in rows.iterrows():
        src = Path(str(row["path"]))
        if not src.exists():
            continue
        board = str(row.get("board","board"))
        r, c = int(row.get("r", 0)), int(row.get("c", 0))
        ext = src.suffix.lower() or ".png"
        dest_name = f"{board}_r{r}c{c}{ext}"
        split_dir = val_dir if (val_every>0 and ((count+1) % val_every)==0) else train_dir
        shutil.copy2(src, split_dir / dest_name)
        count += 1
    return count

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="Comma-separated paths to all_preds.csv files")
    ap.add_argument("--digits", required=True, help="Comma-separated target digits, e.g., '3,1' or '3,1,2,5'")
    ap.add_argument("--out", required=True, help="Output root (will create queue/train,val)")
    ap.add_argument("--val-every", type=int, default=10)
    ap.add_argument("--include-low", action="store_true", help="Include rows where low_conf==1 (if column exists)")
    ap.add_argument("--wide-margin", type=float, default=None, help="Also include rows with margin < this (if column exists)")
    ap.add_argument("--pair-only", action="store_true", help="Require EXACTLY two digits and keep rows whose {pred,top2} equals that pair")
    args = ap.parse_args()

    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)

    # Load CSVs
    frames = []
    for p in [Path(s.strip()) for s in args.preds.split(",") if s.strip()]:
        if p.exists():
            frames.append(pd.read_csv(p))
        else:
            print(f"[mine_digits_pack] WARN: missing {p}", file=sys.stderr)
    if not frames:
        print("[mine_digits_pack] ERROR: no valid all_preds.csv provided.", file=sys.stderr); return
    df = pd.concat(frames, ignore_index=True)

    # Parse target digits
    target = [int(x) for x in args.digits.split(",") if x.strip()]
    if len(target) == 0:
        print("[mine_digits_pack] ERROR: --digits is empty.", file=sys.stderr); return

    # Base filter: pred/top2 in target
    in_target = df["pred"].isin(target) | df["top2"].isin(target)

    # Optional pair-only constraint
    if args.pair_only:
        if len(target) != 2:
            print("[mine_digits_pack] ERROR: --pair-only requires EXACTLY two digits in --digits.", file=sys.stderr)
            return
        a, b = set(target)
        pair_mask = ((df["pred"].isin({a,b})) & (df["top2"].isin({a,b})) & (df["pred"] != df["top2"]))
        in_target = in_target & pair_mask

    # Hardness condition: (low_conf) OR (margin small)
    cond = in_target
    if args.include_low and "low_conf" in df.columns:
        cond = cond & (df["low_conf"] == 1)
    if args.wide_margin is not None and "margin" in df.columns:
        cond = cond | (df["margin"] < float(args.wide_margin))

    sel = df[cond].drop_duplicates(subset=["path"])

    # Summary
    print(f"[mine_digits_pack] scanned {len(df)} rows; selected {len(sel)} tiles.")
    if len(sel) > 0:
        try:
            print("[mine_digits_pack] selection breakdown (pred -> count):")
            print(sel["pred"].value_counts().sort_index())
        except Exception:
            pass

    n = copy_rows(sel, out_root, val_every=args.val_every)
    print(f"[mine_digits_pack] copied {n} tiles -> {out_root/'queue'}")
    if n == 0:
        print("[mine_digits_pack] HINT: increase --wide-margin (e.g., 0.20) or re-run inference on more boards.")

if __name__ == "__main__":
    main()
