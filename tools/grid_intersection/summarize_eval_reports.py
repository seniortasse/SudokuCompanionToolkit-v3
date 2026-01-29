#!/usr/bin/env python3
"""
summarize_eval_reports.py

Scan a sweep root for run folders and aggregate evaluation summaries
into a single CSV (and JSON) for ranking & comparison.

What it reads (best effort; all optional):
- <run>/eval_report.json                   (from tools/grid_intersection/eval_suite.py)
- <run>/eval_report_epoch*.json            (from train.py --eval_only or per-epoch val mini-pass)
- <run>/config_used.yaml or config_used.json (effective config dumped by train.py)
- Hyperparameters encoded in the folder name, e.g. "mode-softargmax_temp-0.5_tj-0.70_conf-0.05_topk-180_img-768"

What it writes:
- <out>/summary.csv
- <out>/summary.json

Usage
-----
python summarize_eval_reports.py --root <sweep_root> [--out <outdir>]
"""
from __future__ import annotations

import argparse, json, csv, re, os
from pathlib import Path
from typing import Dict, Any, List, Optional

def _read_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def _load_config_used(run_dir: Path) -> Dict[str, Any]:
    for name in ("config_used.yaml", "config_used.json"):
        p = run_dir / name
        if p.exists():
            try:
                if p.suffix == ".yaml":
                    import yaml  # optional
                    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                else:
                    return json.loads(p.read_text(encoding="utf-8")) or {}
            except Exception:
                pass
    return {}

def _parse_from_dirname(name: str) -> Dict[str, Any]:
    # Extract key=value-ish patterns commonly used in these sweeps
    # e.g., mode-softargmax_temp-0.5_tj-0.70_conf-0.05_topk-180_img-768
    kv: Dict[str, Any] = {}
    # break into tokens separated by '_'
    for tok in name.split('_'):
        if '-' in tok:
            k, v = tok.split('-', 1)
            if re.fullmatch(r"-?\d+(\.\d+)?", v):
                try:
                    kv[k] = float(v) if ('.' in v) else int(v)
                    continue
                except Exception:
                    pass
            kv[k] = v
    return kv

def _best_eval_json(run_dir: Path) -> Optional[Path]:
    """
    Prefer eval_suite's eval_report.json if present (single report per run).
    Otherwise, pick the latest eval_report_epochXX.json.
    """
    p = run_dir / "eval_report.json"
    if p.exists():
        return p
    candidates = sorted(run_dir.glob("eval_report_epoch*.json"))
    if candidates:
        return candidates[-1]
    # Some users put reports directly in a preds folder
    candidates = sorted((run_dir / "preds_val_epoch01").glob("*.json")) if (run_dir / "preds_val_epoch01").exists() else []
    return candidates[-1] if candidates else None

def _flatten_metrics(m: Dict[str, Any]) -> Dict[str, Any]:
    """Pick the common, comparable keys; rename a few for consistency."""
    out: Dict[str, Any] = {}
    # Common keys across both report styles (best effort)
    key_map = {
        "IoU_mean": "IoU_mean",
        "J_MJE": "J_MJE",
        "J_MJE_norm": "J_MJE_norm",
        "J_AP@2px_finite": "AP2",
        "J_MJE<= 6px": "LE6",
        "J_MJE<= 8px": "LE8",
        "J_MJE<= 10px": "LE10",
        "pred_J_eq_100": "predJ100",
        "cases": "cases",
        "errors": "errors",
    }
    # Some reports nest means under "means" or "summary"
    merged = {}
    merged.update(m)
    for k in ("means", "summary"):
        if isinstance(m.get(k), dict):
            merged.update(m[k])
    for k_src, k_dst in key_map.items():
        if k_src in merged:
            out[k_dst] = merged[k_src]
    # If the report provides per-case arrays, compute means quickly
    if "cases" not in out and isinstance(merged.get("case_count"), (int, float)):
        out["cases"] = merged["case_count"]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Sweep root: directory that contains many run subfolders")
    ap.add_argument("--out", default=None, help="Output folder for the summary CSV/JSON (default: <root>)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    outdir = Path(args.out).resolve() if args.out else root
    outdir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for run_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        eval_json = _best_eval_json(run_dir)
        if not eval_json:
            continue
        report = _read_json(eval_json)
        if not report:
            continue

        # Collate metrics
        flat = _flatten_metrics(report)

        # Attach config hints: from config_used.* + folder name
        cfg = _load_config_used(run_dir)
        # CLI knobs that matter most for the sweep
        # Try to fetch from cfg first, then from dirname tokens
        derived = _parse_from_dirname(run_dir.name)
        row: Dict[str, Any] = {
            "run_dir": run_dir.name,
            "eval_file": eval_json.name,
            "image_size": cfg.get("image_size", derived.get("img")),
            "base_ch": cfg.get("model", {}).get("base_ch", derived.get("base_ch", None)),
            "subpixel": cfg.get("subpixel", derived.get("mode", None)),
            "softargmax_temp": cfg.get("softargmax_temp", derived.get("temp", None)),
            "tj": derived.get("tj", None),
            "j_conf": derived.get("conf", None),
            "j_topk": derived.get("topk", None),
        }
        row.update(flat)
        rows.append(row)

    # Sort by a composite you care about (lower MJE, higher AP2, higher LE8)
    def _score(r: Dict[str, Any]) -> float:
        mje = r.get("J_MJE", float("inf"))
        ap2 = r.get("AP2", 0.0) or 0.0
        le8 = r.get("LE8", 0.0) or 0.0
        # normalize terms to similar range; tweak weights to taste
        return (-ap2 * 10.0) + (mje) + (-le8 * 5.0)

    rows.sort(key=_score)

    # Write CSV
    csv_path = outdir / "summary.csv"
    if rows:
        fieldnames = list(rows[0].keys()) + ["score"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                r2 = dict(r)
                r2["score"] = _score(r)
                w.writerow(r2)

    # Write JSON
    json_path = outdir / "summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"[summarize] wrote {csv_path}")
    print(f"[summarize] wrote {json_path}")

if __name__ == "__main__":
    main()
