# tools/build_mixed_manifest.py
# Build a mixed train manifest with a target synth:real ratio (default 80:20).
# Recursively scans provided folders for JSONL manifests:
#   - prefers "cells_*.jsonl" that do NOT contain 'val'/'test' in the filename
#   - otherwise any *.jsonl that do NOT contain 'val'/'test'
#   - finally, any *.jsonl
# Pools synth manifests together and real manifests together, de-dupes by 'path',
# shuffles, downsamples to match the requested ratio, and writes a single JSONL.
#
# Usage (PowerShell):
#   python -m tools.build_mixed_manifest `
#     --synth-folders `
#       datasets/cells/cell_interpreter/synth_train_candidates_only `
#       datasets/cells/cell_interpreter/synth_train_given_only `
#       datasets/cells/cell_interpreter/synth_train_solpluscand_only `
#       datasets/cells/cell_interpreter/synth_train_solution_only `
#     --real-folders `
#       datasets/cells/cell_interpreter/real_train `
#     --out datasets/cells/cell_interpreter/mixed_train_80s20r.jsonl `
#     --ratio-synth 0.80 `
#     --seed 1337
#
from __future__ import annotations
import argparse, json, random
from pathlib import Path
from typing import List, Dict, Iterable


def _rank_key(p: Path) -> tuple[int, str]:
    """Rank manifests: cells_*.jsonl (best), other non-val/test, else anything."""
    name = p.name.lower()
    # Lower rank value = better
    if name.startswith("cells_") and ("val" not in name) and ("test" not in name):
        return (0, name)
    if ("val" not in name) and ("test" not in name) and name.endswith(".jsonl"):
        return (1, name)
    return (2, name)


def find_manifests(folder: Path, recursive: bool = True) -> List[Path]:
    """
    Return a ranked list of JSONL manifests inside 'folder'.
    If recursive, scans subfolders too (e.g., real_train/**/cells_real_labeled.jsonl).
    """
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    pats = ["*.jsonl"]
    files: List[Path] = []
    if recursive:
        for pat in pats:
            files.extend(folder.rglob(pat))
    else:
        for pat in pats:
            files.extend(folder.glob(pat))

    files = [p for p in files if p.is_file()]
    if not files:
        return []

    # Sort by our preference ranking
    files_sorted = sorted(files, key=_rank_key)
    return files_sorted


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except Exception:
                # skip malformed lines
                pass
    return rows


def dedupe_by_path(rows: Iterable[Dict]) -> List[Dict]:
    out: List[Dict] = []
    seen = set()
    for r in rows:
        p = (r.get("path") or "").replace("\\", "/")
        if not p or p in seen:
            continue
        seen.add(p)
        out.append(r)
    return out


def collect_group(folders: List[str], group_name: str) -> List[Dict]:
    """Collect and merge rows from a list of folders (recursively)."""
    all_rows: List[Dict] = []
    total_manifests = 0

    for folder in folders:
        base = Path(folder)
        mans = find_manifests(base, recursive=True)
        if not mans:
            print(f"[warn] No JSONL found under {base} (recursive).")
            continue

        # Favor earlier-ranked manifests; but in a 'folder of folders' we likely want all *rank-0* first,
        # then rank-1 (non-val/test), and avoid val/test. We'll read them in rank order and just pool.
        for m in mans:
            name = m.name.lower()
            if "val" in name or "test" in name:
                continue  # skip validation/test files
            rows = read_jsonl(m)
            if rows:
                all_rows.extend(rows)
                total_manifests += 1

    print(f"[info] {group_name}: loaded {len(all_rows)} rows from {total_manifests} manifest(s).")
    return all_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-folders", nargs="+", required=True,
                    help="List of synth train folders (will be scanned recursively).")
    ap.add_argument("--real-folders", nargs="+", required=True,
                    help="List of real train folders (will be scanned recursively).")
    ap.add_argument("--out", type=str, required=True,
                    help="Output mixed JSONL path.")
    ap.add_argument("--ratio-synth", type=float, default=0.80,
                    help="Target fraction of synthetic samples in the mix (default 0.80).")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    random.seed(args.seed)

    # Gather synth + real recursively
    synth_rows = collect_group(args.synth_folders, "synth")
    real_rows  = collect_group(args.real_folders,  "real")

    # De-dupe by path and shuffle
    synth_rows = dedupe_by_path(synth_rows)
    real_rows  = dedupe_by_path(real_rows)

    random.shuffle(synth_rows)
    random.shuffle(real_rows)

    n_synth = len(synth_rows)
    n_real  = len(real_rows)

    if n_synth == 0 or n_real == 0:
        raise SystemExit(f"[error] Need both synth and real > 0 (got synth={n_synth}, real={n_real}).")

    r = max(0.0, min(1.0, float(args.ratio_synth)))

    # Degenerate cases: just concat + shuffle
    if r in (0.0, 1.0):
        mix = synth_rows + real_rows
        random.shuffle(mix)
    else:
        # We want synth:real ≈ r : (1-r)
        synth_per_real = r / (1.0 - r)
        # Default: keep ALL real, downsample synth to match ratio
        n_synth_keep = min(n_synth, int(n_real * synth_per_real))
        n_real_keep  = int(round(n_synth_keep / synth_per_real))
        n_real_keep  = min(n_real_keep, n_real)
        n_synth_keep = min(n_synth_keep, n_synth)

        # If synth is the limiter, recompute real to preserve ratio
        if n_synth_keep < int(n_real * synth_per_real):
            n_real_keep = int(round(n_synth_keep / synth_per_real))
            n_real_keep = min(n_real_keep, n_real)

        synth_rows = synth_rows[:n_synth_keep]
        real_rows  = real_rows [:n_real_keep]
        mix = synth_rows + real_rows
        random.shuffle(mix)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rcd in mix:
            f.write(json.dumps(rcd) + "\n")

    print(f"[done] wrote {len(mix)} rows → {out_path}")
    print(f"[info] kept synth={len(synth_rows)}  real={len(real_rows)}  ratio_synth≈{len(synth_rows)/max(1,len(mix)):.3f}")


if __name__ == "__main__":
    main()