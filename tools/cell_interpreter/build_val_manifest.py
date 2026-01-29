import argparse, json
from pathlib import Path

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception:
                pass

def read_json(path: Path):
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            for r in obj:
                yield r
    except Exception:
        pass

def normalize_path(p, project_root: Path) -> str:
    p = str(p or "").strip()
    if not p:
        return ""
    ps = Path(p)
    # Keep relative paths as-is (just normalize slashes)
    if not ps.is_absolute():
        return str(ps).replace("\\", "/")
    # If absolute and under project_root, rebase to relative
    try:
        rel = ps.relative_to(project_root)
        return str(rel).replace("\\", "/")
    except Exception:
        # Different drive or outside tree: keep absolute, normalized
        return str(ps).replace("\\", "/")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path,
                    help="Parent folder with subfolders each containing cells_real_labeled.jsonl/.json")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output JSONL manifest")
    ap.add_argument("--project-root", type=Path, default=Path("."),
                    help="Project root; used to normalize absolute paths to relative")
    ap.add_argument("--report-missing", type=Path, default=None,
                    help="Optional text file to list missing image paths")
    ap.add_argument("--drop-missing", action="store_true",
                    help="Drop rows whose image file is missing")
    args = ap.parse_args()

    project_root = args.project_root.resolve()

    rows = []
    # Collect from both .jsonl and .json
    for p in args.root.rglob("cells_real_labeled.jsonl"):
        rows.extend(list(read_jsonl(p)))
    for p in args.root.rglob("cells_real_labeled.json"):
        rows.extend(list(read_json(p)))

    if not rows:
        raise SystemExit(f"No labeled files found under {args.root}")

    out_rows = []
    missing = []

    for r in rows:
        raw_path = (r.get("path") or "").strip()
        if not raw_path:
            continue

        path_norm = normalize_path(raw_path, project_root)

        # If drop-missing, check existence (resolve relative to project root)
        if args.drop_missing:
            ps = Path(path_norm)
            if not ps.is_absolute():
                ps = project_root / ps
            if not ps.exists():
                missing.append(path_norm)
                continue

        rec = {
            "path": path_norm,
            "given_digit": int(r.get("given_digit", 0)),
            "solution_digit": int(r.get("solution_digit", 0)),
            "candidates": [int(x) for x in r.get("candidates", [])],
            "source": r.get("source", "real_val"),
        }
        out_rows.append(rec)

    # De-dupe by path (keep first)
    seen = set()
    deduped = []
    for r in out_rows:
        p = r["path"]
        if p in seen:
            continue
        seen.add(p)
        deduped.append(r)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in deduped:
            f.write(json.dumps(r) + "\n")

    if args.report_missing is not None and missing:
        args.report_missing.parent.mkdir(parents=True, exist_ok=True)
        args.report_missing.write_text("\n".join(sorted(set(missing))), encoding="utf-8")

    print(f"[done] wrote {len(deduped)} rows to {args.out}")
    if missing:
        print(f"[warn] missing images: {len(set(missing))} (see {args.report_missing or 'no report path provided'})")

if __name__ == "__main__":
    main()