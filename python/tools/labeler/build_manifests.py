
"""
Build JSONL manifests from a 0..9 folder tree.
Usage:
  python tools/labeler/build_manifests.py --root vision/data/real --out vision/data/real/meta
Produces:
  train.jsonl, val.jsonl with fields {path, label, source}.
"""
import argparse, json
from pathlib import Path

DIGITS = list("0123456789")

def write_manifest(split_root: Path, out_jsonl: Path, source: str):
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_jsonl.open("w", encoding="utf-8") as f:
        for d in DIGITS:
            ddir = split_root / d
            if not ddir.exists():
                continue
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                for p in ddir.glob(ext):
                    f.write(json.dumps({"path": str(p), "label": int(d), "source": source}) + "\n")
                    n += 1
    return n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root that contains train/ and val/ with 0..9 subfolders.")
    ap.add_argument("--out", required=True, help="Output folder for jsonl files.")
    ap.add_argument("--source", default="real")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    ntr = write_manifest(root/"train", out/"train.jsonl", args.source)
    nva = write_manifest(root/"val", out/"val.jsonl", args.source)
    print(f"Wrote {ntr} train and {nva} val entries to {out}")

if __name__ == "__main__":
    main()
