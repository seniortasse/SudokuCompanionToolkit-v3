# tools/visualize_hard_examples.py
# ------------------------------------------------------------
# Create a folder gallery for hard examples, grouped by
# (true_digit, predicted_digit) for the solution head.
# ------------------------------------------------------------
import argparse, json, shutil
from pathlib import Path

from PIL import Image  # only to sanity-check readability

def read_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hard-manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--only-true", default="", help="Optional comma-separated true digits to keep, e.g. '1,7'")
    ap.add_argument("--only-pred", default="", help="Optional comma-separated predicted digits to keep, e.g. '1,7'")
    args = ap.parse_args()

    rows = read_jsonl(args.hard_manifest)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    only_true = {int(x) for x in args.only_true.split(",") if x.strip()} if args.only_true else None
    only_pred = {int(x) for x in args.only_pred.split(",") if x.strip()} if args.only_pred else None

    for i, r in enumerate(rows):
        hi = r.get("_hard_info", {}).get("solution", None)
        if not hi:
            continue
        y = int(hi.get("y", 0))
        y_pred = int(hi.get("y_pred", 0))
        if y == y_pred:
            continue  # skip "no-op" rows (shouldn't normally happen here)

        # Optional filtering
        if only_true is not None and y not in only_true:
            continue
        if only_pred is not None and y_pred not in only_pred:
            continue

        # Group folder: e.g. sol_y1_yp7
        group_dir = out_root / f"sol_y{y}_yp{y_pred}"
        group_dir.mkdir(parents=True, exist_ok=True)

        src = Path(r["path"]).expanduser().resolve()

        # Name target file as index + original name
        dst = group_dir / f"{i:06d}_{src.name}"

        try:
            # Just copy the image for now (could also use hard links or symlinks).
            shutil.copy2(src, dst)
            # Optional sanity check: open & close once
            Image.open(dst).close()
        except Exception as e:
            print(f"[warn] failed copying {src} -> {dst}: {e}")

    print(f"[done] Gallery written under: {out_root}")

if __name__ == "__main__":
    main()