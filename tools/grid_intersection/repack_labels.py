# tools/grid_intersection/repack_labels.py
import argparse, json, os, zipfile
from pathlib import Path
import numpy as np

def is_uncompressed_npz(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path, "r") as zf:
            # All members must be stored without deflate
            return all(info.compress_type == zipfile.ZIP_STORED for info in zf.infolist())
    except Exception:
        return False  # if broken/unreadable, let caller try to rewrite

def repack_npz_in_place(npz_path: Path, to: str):
    tmp = npz_path.with_suffix(".tmp.npz")
    # If tmp from previous failure exists, remove it
    if tmp.exists():
        try:
            tmp.unlink()
        except Exception:
            pass

    with np.load(str(npz_path), allow_pickle=False) as d:
        arrays = {k: d[k] for k in d.files}
    if to == "uncompressed":
        np.savez(str(tmp), **arrays)               # ZIP_STORED (no deflate)
    elif to == "compressed":
        np.savez_compressed(str(tmp), **arrays)    # ZIP_DEFLATED
    else:
        raise ValueError("--to must be 'compressed' or 'uncompressed'")
    os.replace(tmp, npz_path)  # atomic replace

def main():
    ap = argparse.ArgumentParser(description="Repack label NPZ files referenced by a manifest.")
    ap.add_argument("--manifest", required=True, help="JSONL manifest with label_path fields")
    ap.add_argument("--to", required=True, choices=["compressed", "uncompressed"])
    ap.add_argument("--dry_run", action="store_true", help="List what would be changed")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip files already in the requested format")
    ap.add_argument("--offset", type=int, default=0, help="Skip first N records in manifest")
    ap.add_argument("--limit", type=int, default=0, help="Process at most N records")
    args = ap.parse_args()

    mpath = Path(args.manifest)
    records = [json.loads(l) for l in mpath.read_text(encoding="utf-8").splitlines() if l.strip()]

    if args.offset:
        records = records[args.offset:]
    if args.limit and args.limit > 0:
        records = records[:args.limit]

    total = 0
    changed = 0
    for r in records:
        lp = r.get("label_path")
        if not lp:
            continue
        p = Path(lp)
        if not p.exists():
            print(f"[warn] missing label_path: {p}")
            continue

        total += 1
        if args.skip_existing:
            already_uncompressed = is_uncompressed_npz(p)
            if args.to == "uncompressed" and already_uncompressed:
                continue
            if args.to == "compressed" and not already_uncompressed:
                # crude check; in practice you’d check ZIP_DEFLATED
                continue

        if args.dry_run:
            print(f"[dry] {p} -> {args.to}")
            continue

        repack_npz_in_place(p, args.to)
        changed += 1
        if changed % 100 == 0:
            print(f"[info] repacked {changed} files...")

    print(f"[done] scanned {total} records, repacked {changed} to {args.to}.")

if __name__ == "__main__":
    main()