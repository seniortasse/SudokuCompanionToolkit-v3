
# vision/infer/json_to_table.py
from __future__ import annotations
import argparse, json, csv
from pathlib import Path

def flatten(json_path: Path, out_csv: Path):
    obj = json.loads(json_path.read_text(encoding="utf-8"))
    grid = obj["grid"]; probs = obj["probs"]; top2 = obj["top2"]; prob2 = obj["prob2"]
    margin = obj["margin"]; low_conf = obj.get("low_conf") or [[0]*9 for _ in range(9)]
    paths = obj.get("paths", [[""]*9 for _ in range(9)])

    rows = []
    for r in range(9):
        for c in range(9):
            rows.append({
                "r": r+1, "c": c+1,
                "pred": grid[r][c],
                "prob": probs[r][c],
                "top2": top2[r][c],
                "prob2": prob2[r][c],
                "margin": margin[r][c],
                "low_conf": low_conf[r][c],
                "path": paths[r*9+c] if isinstance(paths, list) else ""
            })
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Wrote:", out_csv)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="JSON from classify_cells_model.py")
    ap.add_argument("--out", required=True, help="CSV path")
    args = ap.parse_args()
    flatten(Path(args.json), Path(args.out))

if __name__ == "__main__":
    main()
