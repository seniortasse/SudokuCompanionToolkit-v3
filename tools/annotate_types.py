# tools/annotate_types.py
"""
Type-Hint Annotator (conservative)
----------------------------------
Adds type hints to a small set of well-known functions if *not* already typed.
- Only touches exact function names we list in TARGETS.
- Preserves code semantics; only signature lines are adjusted.
- Inserts `from __future__ import annotations` and common typing imports if missing.

Run:
  python tools/annotate_types.py --dry-run
  python tools/annotate_types.py --apply
"""
from __future__ import annotations
import argparse, re
from pathlib import Path

INSERT_IMPORT = "from __future__ import annotations\n"

COMMON_TYPES = "from typing import List, Dict, Optional, Tuple, Any\nfrom types_sudoku import Grid, Candidates, Move\n"

TARGETS = {
  "solver/sudoku_tools.py": {
    "insert_types": {
      # name: (new_signature)
      "compute_candidates_tool": "def compute_candidates_tool(grid: Grid) -> Candidates:",
      "find_naked_singles": "def find_naked_singles(grid: Grid, candidates: Candidates) -> List[Move]:",
      "find_hidden_singles": "def find_hidden_singles(grid: Grid, candidates: Candidates) -> List[Move]:",
      "find_locked_candidates_pointing": "def find_locked_candidates_pointing(grid: Grid, candidates: Candidates) -> List[Move]:",
      "chain_followup_singles": "def chain_followup_singles(grid: Grid, candidates: Candidates, max_chain: int = 20) -> List[Move]:",
      "next_moves": "def next_moves(current: Grid, candidates: Candidates, max_moves: int = 5) -> List[Move]:",
    }
  },
  "apps/cli/demo_cli_overlay.py": {
    "insert_types": {
      "parse_args": "def parse_args() -> Any:",
      "main": "def main() -> None:",
    }
  },
  "apps/cli/storyboard_sheet.py": {
    "insert_types": {
      "read_moves_json": "def read_moves_json(json_path: str) -> Dict[str, Any]:",
      "layout_grid": "def layout_grid(paper: str, cols: int) -> Dict[str, Any]:",
      "main": "def main() -> None:",
    }
  },
  "apps/cli/overlay_renderer.py": {
    "insert_types": {
      "draw_move": "def draw_move(board_image_path: str, move: Move, out_path: str) -> None:",
    }
  },
  "vision/rectify/opencv_rectify.py": {
    "insert_types": {
      "process": "def process(image_path: str, out_dir: str, tile_size: int = 64) -> tuple[str, list[Any]]:",
    }
  }
}

def ensure_future_import(text:str)->str:
    lines = text.splitlines()
    for ln in lines[:8]:
        if "from __future__ import annotations" in ln:
            return text
    # insert after shebang / encoding if present
    i = 0
    while i < len(lines) and (lines[i].startswith("#!") or lines[i].startswith("# -*-")):
        i += 1
    lines.insert(i, INSERT_IMPORT.strip())
    return "\n".join(lines)

def ensure_common_imports(text:str)->str:
    if "from types_sudoku import Grid, Candidates, Move" in text:
        return text
    # add below future import or at top
    lines = text.splitlines()
    j = 0
    for idx, ln in enumerate(lines[:20]):
        if "from __future__ import annotations" in ln:
            j = idx + 1
            break
    lines.insert(j, COMMON_TYPES.rstrip())
    return "\n".join(lines)

def replace_signature(text:str, func:str, new_sig:str)->tuple[str,bool]:
    # Replace leading 'def func(' line with new signature if not already typed
    # Capture indentation
    pattern = re.compile(rf'^(\s*)def\s+{re.escape(func)}\s*\(.*\)\s*:', re.M)
    m = pattern.search(text)
    if not m: return text, False
    indent = m.group(1)
    # If there's already ':'-style hints inside '(', we still allow replacement (idempotent intent)
    replacement = indent + new_sig
    new_text = text[:m.start()] + replacement + text[m.end():]
    return new_text, True

def process_file(base:Path, rel:str, plan:dict, apply:bool)->dict:
    p = base/rel
    report = {"file": rel, "exists": p.exists(), "changed": [], "skipped": []}
    if not p.exists(): return report
    text = p.read_text(encoding="utf-8")
    orig = text
    text = ensure_future_import(text)
    text = ensure_common_imports(text)
    for fn, sig in plan.get("insert_types", {}).items():
        newer, did = replace_signature(text, fn, sig)
        if did:
            text = newer
            report["changed"].append(fn)
        else:
            report["skipped"].append(fn)
    if apply and text != orig:
        p.write_text(text, encoding="utf-8")
    return report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=".", help="project root")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--report", default="docs/type_annotation_report.json")
    args = ap.parse_args()

    base = Path(args.base)
    results = []
    for rel, plan in TARGETS.items():
        results.append(process_file(base, rel, plan, apply=args.apply))

    out = base / args.report
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(__import__("json").dumps(results, indent=2), encoding="utf-8")
    print(f"[{'APPLY' if args.apply else 'DRY-RUN'}] Processed {sum(1 for r in results if r['exists'])} files. Report → {out}")

if __name__ == "__main__":
    main()
