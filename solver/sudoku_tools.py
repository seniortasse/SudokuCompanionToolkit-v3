from __future__ import annotations
from typing import List, Dict, Optional, Tuple, Any
from types_sudoku import Grid, Candidates, Move
"""Human-style solving helpers: candidates calculation and next-move search over common techniques (singles, locked candidates), including auto follow-up (chaining). Also provides a tool-friendly interface for the demo CLI."""


# sudoku_tools.py (chaining enabled)
from typing import Dict, List, Tuple
from copy import deepcopy

from .solver_core import (
    Grid, compute_candidates, find_naked_singles, find_hidden_singles,
    find_locked_candidates_pointing, find_locked_candidates_claiming,
    apply_move as _apply_move
)

def sanity_check(original:Grid, current:Grid)->Dict:
    issues = []
    for r in range(1,10):
        for c in range(1,10):
            if original[r-1][c-1] != 0 and current[r-1][c-1] not in (0, original[r-1][c-1]):
                issues.append({"type":"given_overwritten","cell":f"r{r}c{c}",
                               "given": original[r-1][c-1], "found": current[r-1][c-1]})
    def duplicates_in_unit(vals):
        seen=set(); dups=set()
        for v in vals:
            if v==0: continue
            if v in seen: dups.add(v)
            seen.add(v)
        return dups
    # rows
    for r in range(1,10):
        dups = duplicates_in_unit(current[r-1])
        if dups:
            cells = [f"r{r}c{c}" for c in range(1,10) if current[r-1][c-1] in dups]
            issues.append({"type":"duplicate","unit":f"r{r}","digits":sorted(list(dups)),"cells":cells})
    # cols
    for c in range(1,10):
        col = [current[r-1][c-1] for r in range(1,10)]
        dups = duplicates_in_unit(col)
        if dups:
            cells = [f"r{r}c{c}" for r in range(1,10) if current[r-1][c-1] in dups]
            issues.append({"type":"duplicate","unit":f"c{c}","digits":sorted(list(dups)),"cells":cells})
    # boxes
    for b in range(1,10):
        br = (b-1)//3; bc=(b-1)%3
        cells = []
        vals = []
        for i in range(3):
            for j in range(3):
                r = 3*br+i+1; c = 3*bc+j+1
                cells.append(f"r{r}c{c}")
                vals.append(current[r-1][c-1])
        dups = duplicates_in_unit(vals)
        if dups:
            bad = [cells[i] for i,v in enumerate(vals) if v in dups]
            issues.append({"type":"duplicate","unit":f"b{b}","digits":sorted(list(dups)),"cells":bad})
    return {"ok": len(issues)==0, "issues": issues}

def compute_candidates_tool(current:Grid)->Dict:
    """Compute candidate digits for each empty cell in the current grid. Returns a dict like {'r1c2':[1,2,5], ...}."""
    return {"candidates": compute_candidates(current)}

def _append_unique(moves:List[Dict], new_moves:List[Dict]):
    seen = set((m.get("technique"), m.get("cell"), m.get("digit"), tuple(sorted(m.get("eliminate",[])))) for m in moves)
    for m in new_moves:
        key = (m.get("technique"), m.get("cell"), m.get("digit"), tuple(sorted(m.get("eliminate",[]))))
        if key not in seen:
            moves.append(m); seen.add(key)

def next_moves(
    current: Grid,
    candidates: Candidates | None = None,
    max_difficulty: str = "locked",
    max_moves: int = 5,
    chain: bool = True,
) -> list[Move]:
    """Top-level technique dispatcher. In order: singles, locked candidates;
    applies chaining between steps. Returns at most `max_moves` moves with
    visualization-friendly fields.
    """
    """
    Returns up to max_moves moves. If `chain=True`, applies eliminations and placements
    to discover follow-up singles (naked/hidden), updating `current` and `candidates` on the fly.
    """
    if candidates is None:
        candidates = compute_candidates(current)
    # Work on copies
    cur = [row[:] for row in current]
    cands = {k:v[:] for k,v in candidates.items()}
    out_moves: List[Dict] = []

    # 1) Start with singles (immediate wins)
    singles = find_naked_singles(cur, cands) + find_hidden_singles(cur, cands)
    _append_unique(out_moves, singles)

    # If chaining, apply placements as we go
    def apply_placement(move):
        nonlocal cur, cands
        cur = _apply_move(cur, move)
        cands = compute_candidates(cur)

    for m in list(out_moves):
        if m.get("type","placement")=="placement" and chain and len(out_moves) <= max_moves:
            apply_placement(m)

    # 2) Locked candidates (eliminations), then re-scan for singles
    if max_difficulty in ("locked","xwing","advanced") and len(out_moves) < max_moves:
        # find eliminations
        elims = find_locked_candidates_pointing(cur, cands) + find_locked_candidates_claiming(cur, cands)
        # apply each elimination and test for new singles
        for e in elims:
            if len(out_moves) >= max_moves: break
            # simulate elimination effect
            changed = False
            for key in e.get("eliminate", []):
                if key in cands and e["digit"] in cands[key]:
                    changed = True
                    cands[key] = [d for d in cands[key] if d != e["digit"]]
            if not changed:
                continue
            # record the elimination
            _append_unique(out_moves, [e])
            if len(out_moves) >= max_moves: break
            # re-scan singles after elimination
            new_singles = find_naked_singles(cur, cands) + find_hidden_singles(cur, cands)
            if new_singles:
                for s in new_singles:
                    if len(out_moves) >= max_moves: break
                    _append_unique(out_moves, [s])
                    if chain:
                        apply_placement(s)

    # 3) Final truncate and return the current/candidates snapshot if needed
    out_moves = out_moves[:max_moves]
    return {"moves": out_moves, "snapshot": {"current": cur, "candidates": cands}}

def apply_action(current:Grid, candidates:Dict[str, List[int]] , move:Dict):
    if move.get("type","placement") == "placement":
        g2 = _apply_move(current, move)
        c2 = compute_candidates(g2)
        return {"current": g2, "candidates": c2}
    else:
        from copy import deepcopy
        c2 = deepcopy(candidates) if candidates is not None else compute_candidates(current)
        for key in move.get("eliminate", []):
            if key in c2 and move["digit"] in c2[key]:
                c2[key] = [d for d in c2[key] if d != move["digit"]]
        return {"current": current, "candidates": c2}

# Backward-compat
def apply_move(current:Grid, move:Dict)->Dict:
    return apply_action(current, None, move)