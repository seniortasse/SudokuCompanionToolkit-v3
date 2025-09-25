# vision/label/collect_cells.py
from __future__ import annotations
from pathlib import Path
import json, shutil, time
from typing import Any, List

def _looks_like_path(s: str) -> bool:
    s = s.lower()
    return any(s.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))

def _try_candidates(cands: List[Path]) -> Path | None:
    for c in cands:
        try:
            q = c.resolve()
        except Exception:
            q = c
        if q.exists():
            return q
    return None

def _coerce_path(s: str, base: Path) -> Path:
    """
    Resolve `s` (which may be absolute or relative) to an existing file.
    If `s` is relative and starts with the base folder name, strip that prefix
    to avoid joining base/base/…
    Try (in order): base/s, base.parent/s, cwd/s.
    """
    p = Path(s)
    if p.is_absolute():
        return p

    # If s starts with the base folder name (e.g., "demo_export/cells/r1c1.png"),
    # drop that first component to avoid base/base duplication.
    parts = p.parts
    if parts and parts[0].lower() == base.name.lower():
        p = Path(*parts[1:])  # strip leading "demo_export"

    candidates = [
        (base / p),
        (base.parent / p),
        (Path.cwd() / p),
    ]
    resolved = _try_candidates(candidates)
    return resolved if resolved is not None else (base / p)  # fall back (may not exist)

def _extract_paths(obj: Any, base: Path, out: List[Path]) -> None:
    """Collect image paths from many common JSON shapes."""
    if isinstance(obj, str) and _looks_like_path(obj):
        out.append(_coerce_path(obj, base)); return

    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, str) and _looks_like_path(item):
                out.append(_coerce_path(item, base))
            elif isinstance(item, dict):
                # try common keys
                for key in ("path", "file", "filepath", "relpath"):
                    v = item.get(key)
                    if isinstance(v, str) and _looks_like_path(v):
                        out.append(_coerce_path(v, base))
                        break
                else:
                    # fallback: scan nested values
                    for v in item.values():
                        if isinstance(v, str) and _looks_like_path(v):
                            out.append(_coerce_path(v, base)); break
                        elif isinstance(v, (dict, list)):
                            _extract_paths(v, base, out)
            elif isinstance(item, (dict, list)):
                _extract_paths(item, base, out)
        return

    if isinstance(obj, dict):
        for key in ("paths", "cells", "items"):
            if key in obj:
                _extract_paths(obj[key], base, out)
        for v in obj.values():
            if isinstance(v, str) and _looks_like_path(v):
                out.append(_coerce_path(v, base))
            elif isinstance(v, (dict, list)):
                _extract_paths(v, base, out)
        return
    # else: ignore other types

def collect(cells_json: str, out_dir: str = "vision/data/real/inbox") -> None:
    cj = Path(cells_json)
    base = cj.parent
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    raw = json.loads(cj.read_text(encoding="utf-8"))
    found: List[Path] = []
    _extract_paths(raw, base, found)

    # de-dup and sanity-check existence
    unique: List[Path] = []
    seen = set()
    missing: List[Path] = []
    for p in found:
        q = p.resolve()
        if q in seen:
            continue
        if not q.exists():
            missing.append(q)
        else:
            unique.append(q); seen.add(q)

    if missing:
        # print only a few to keep output readable
        sample = "\n  ".join(str(m) for m in missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} referenced image(s) not found. First few:\n  {sample}\n"
            f"Tip: Your JSON paths appear to be relative to the project root. "
            f"If so, either (a) update the rectifier to write paths relative to {base}, "
            f"or (b) keep this collector, which already strips a leading '{base.name}\\' if present."
        )

    # Some pipelines may include >81 paths; keep first 81 in order.
    if len(unique) < 81:
        raise ValueError(f"Found only {len(unique)} image paths; need 81.")
    if len(unique) > 81:
        unique = unique[:81]

    ts = int(time.time())
    for i, src in enumerate(unique):
        dst = out / f"cell_{ts}_{i:02d}{src.suffix.lower()}"
        shutil.copy2(src, dst)

    print(f"Copied {len(unique)} cells to {out}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python vision/label/collect_cells.py demo_export/cells.json [out_dir]")
        raise SystemExit(1)
    cells_json = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) >= 3 else "vision/data/real/inbox"
    collect(cells_json, out_dir)