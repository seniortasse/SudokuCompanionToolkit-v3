#!/usr/bin/env python
import json
from pathlib import Path
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, required=True,
                    help="JSONL manifest with fields path, given_digit, solution_digit, candidates.")
    ap.add_argument("--project-root", type=str, default=".",
                    help="Base dir to resolve relative paths.")
    ap.add_argument("--out", type=str, required=True,
                    help="Output text file: one absolute image path per line.")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    project_root = Path(args.project_root).resolve()
    out_path = Path(args.out)

    blanks = []
    n_total = 0

    with manifest_path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            n_total += 1
            try:
                r = json.loads(ln)
            except Exception:
                continue

            gd = int(r.get("given_digit", 0))
            sd = int(r.get("solution_digit", 0))
            cands = r.get("candidates", [])
            if gd != 0 or sd != 0:
                continue
            if cands:
                continue

            p = Path(r["path"])
            if not p.is_absolute():
                p = (project_root / p).resolve()
            if p.is_file():
                blanks.append(str(p))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for p in blanks:
            f.write(p + "\n")

    print(f"[done] scanned {n_total} records, found {len(blanks)} blank cells")
    print(f"[done] backgrounds written to: {out_path}")

if __name__ == "__main__":
    main()