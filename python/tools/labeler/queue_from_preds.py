
# tools/labeler/queue_from_preds.py  (pandas-free, with stable names + auto split + filters)
r"""
Build a labeling queue directly from an inference CSV (all_preds.csv), with options to
- rename tiles using a stable pattern: <board>_r{r}c{c}.ext
- auto-split selected tiles into queue/train and queue/val (every Nth goes to val)
- filter by board name(s) and/or by predicted digit(s)

Usage (PowerShell):
python ".\tools\labeler\queue_from_preds.py" ^
  --preds ".\runs\infer\all_preds.csv" ^
  --out ".\vision\data\queue_from_preds" ^
  --pairs "7,1 5,6 8,3" ^
  --min-prob 0.95 ^
  --max-margin 0.15 ^
  --only-low-conf 0 ^
  --stable-names 1 ^
  --val-every 10 ^
  --only-boards "predict_4,predict_5" ^
  --also-pred "7,1" ^
  --pred-min-prob 0.00 ^
  --pred-max-margin 1.00 ^
  --max-per-board 0

Selection (union):
- (pred, top2) matches any requested pair AND (prob - prob2) < --max-margin
- (pred, top2) matches any pair AND prob > --min-prob AND (prob - prob2) < 1.5*--max-margin
- If --only-low-conf=1, any row with low_conf=1 is included regardless of pair
- If --also-pred is provided, any row with pred in that list and passing --pred-min-prob/--pred-max-margin is included

Notes
- If --val-every > 0, images are placed under <out>/queue/train/ or <out>/queue/val/
  with every Nth selected image routed to val (1-based counting within this run).
- If --stable-names=1 and the CSV contains 'board','r','c', filenames become
  <board>_r{r}c{c}.ext; otherwise the original basename is preserved.
- --only-boards filters by substring match (case-insensitive) on the 'board' column.
- --max-per-board caps the number of files copied per board (0 = no cap).
"""

import argparse, csv, shutil
from pathlib import Path

REQUIRED_COLS = {"pred","prob","top2","prob2","path","low_conf"}

def parse_pairs(arg: str):
    pairs=set()
    for tok in arg.strip().split():
        if "," in tok: a,b = tok.split(",")
        elif "/" in tok: a,b = tok.split("/")
        elif "-" in tok: a,b = tok.split("-")
        else: continue
        a=int(a); b=int(b)
        pairs.add(tuple(sorted((a,b))))
    return pairs

def parse_digits(arg: str):
    out=set()
    for tok in arg.replace(",", " ").split():
        tok=tok.strip()
        if tok=="": continue
        try:
            out.add(int(tok))
        except Exception:
            pass
    return out

def as_float(v):
    try:
        return float(v)
    except Exception:
        return 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="Path to all_preds.csv")
    ap.add_argument("--out", required=True, help="Output root")
    ap.add_argument("--pairs", type=str, default="7,1 5,6 8,3")
    ap.add_argument("--min-prob", type=float, default=0.95)
    ap.add_argument("--max-margin", type=float, default=0.15)
    ap.add_argument("--only-low-conf", type=int, default=0)
    ap.add_argument("--stable-names", type=int, default=1, help="1=use <board>_r{r}c{c} naming if possible")
    ap.add_argument("--val-every", type=int, default=0, help="Every Nth selected -> val; others -> train. 0 = no split")
    # New filters
    ap.add_argument("--only-boards", type=str, default="", help="Comma-separated substrings to match board names (case-insensitive)")
    ap.add_argument("--also-pred", type=str, default="", help="Digits to also select by predicted class (e.g., '7,1')")
    ap.add_argument("--pred-min-prob", type=float, default=0.0, help="Min prob for --also-pred selections")
    ap.add_argument("--pred-max-margin", type=float, default=1.0, help="Max (prob-prob2) for --also-pred selections")
    ap.add_argument("--max-per-board", type=int, default=0, help="Cap number of files per board (0=no cap)")
    args = ap.parse_args()

    pairs = parse_pairs(args.pairs)
    also_pred = parse_digits(args.also_pred)
    board_filters = [s.strip().lower() for s in args.only_boards.split(",") if s.strip()] if hasattr(args, "only_boards") else []

    src = Path(args.preds)
    if not src.exists():
        print("File not found:", src); return

    base_out = Path(args.out) / "queue"
    if args.val_every and args.val_every > 0:
        out_train = base_out / "train"
        out_val   = base_out / "val"
        out_train.mkdir(parents=True, exist_ok=True)
        out_val.mkdir(parents=True, exist_ok=True)
    else:
        base_out.mkdir(parents=True, exist_ok=True)

    total=0; copied=0; selected=0
    per_board_copied = {}

    with open(src, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        if not REQUIRED_COLS.issubset(cols):
            print("CSV missing required columns; found:", cols, "needed:", REQUIRED_COLS)
            return

        have_board_rc = {"board","r","c"}.issubset(cols)

        for row in reader:
            total += 1
            board = (row.get("board","") or "").strip()
            board_l = board.lower()

            # Board filter
            if board_filters:
                if not any(s in board_l for s in board_filters):
                    continue

            try:
                pr = int(float(row["pred"]))
                tp2 = int(float(row["top2"]))
            except Exception:
                continue
            prob = as_float(row.get("prob", "0"))
            prob2 = as_float(row.get("prob2", "0"))
            margin = prob - prob2
            low = str(row.get("low_conf","0")).strip() in {"1","true","True"}

            key = tuple(sorted((pr, tp2)))

            sel = False
            # Pair-based rules
            if key in pairs and margin < args.max_margin:
                sel = True
            if key in pairs and prob > args.min_prob and margin < (args.max_margin*1.5):
                sel = True
            # Low-conf rule
            if args.only_low_conf and low:
                sel = True
            # Also-pred rule
            if pr in also_pred and (prob >= args.pred_min_prob) and (margin <= args.pred_max_margin):
                sel = True

            if not sel:
                continue

            # Per-board cap
            if args.max_per_board and args.max_per_board > 0:
                if per_board_copied.get(board, 0) >= args.max_per_board:
                    continue

            selected += 1

            p = Path(row["path"])
            if not p.exists():
                continue

            # Determine destination folder (train/val or flat)
            if args.val_every and args.val_every > 0:
                # 1-based counting; every Nth goes to val
                if (selected % args.val_every) == 0:
                    dest_dir = out_val
                else:
                    dest_dir = out_train
            else:
                dest_dir = base_out

            # Stable name if possible
            if args.stable_names and have_board_rc:
                try:
                    r = int(float(row["r"])); c = int(float(row["c"]))
                    stem = f"{board}_r{r}c{c}"
                except Exception:
                    stem = p.stem
            else:
                stem = p.stem

            dst = dest_dir / f"{stem}{p.suffix.lower()}"
            i=1
            while dst.exists():
                dst = dest_dir / f"{stem}_{i}{p.suffix.lower()}"
                i += 1

            try:
                shutil.copy2(p, dst)
                copied += 1
                per_board_copied[board] = per_board_copied.get(board, 0) + 1
            except Exception:
                pass

    where = (base_out / "train") if (args.val_every and args.val_every>0) else base_out
    print(f"Scanned {total} rows; selected {selected}; copied {copied} tiles -> {base_out}")
    if args.val_every and args.val_every>0:
        print(f"  Split into: {out_train} and {out_val} (val every {args.val_every})")

if __name__ == "__main__":
    main()
