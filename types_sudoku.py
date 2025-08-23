# types_sudoku.py
from __future__ import annotations
from typing import TypedDict, Dict, List, Optional, Tuple, Any

Grid = List[List[int]]
"""A 9x9 Sudoku grid as rows of integers (0 = empty)."""

Candidates = Dict[str, List[int]]
"""Map from cell key (e.g., 'r1c1') to a list of candidate digits (1..9)."""

class Move(TypedDict, total=False):
    """A single human-style solving action used by the demo & UI layers."""
    index: int                 # 1-based order in the sequence
    technique: str             # e.g., 'naked_single', 'hidden_single', 'locked_candidates_pointing'
    type: str                  # 'placement' or 'elimination'
    digit: int                 # the digit being placed or eliminated
    cell: str                  # for placements, target cell (e.g., 'r4c7')
    eliminate: List[str]       # for eliminations, list of cells to clear that digit from
    highlights: Dict[str, Any] # UI hints (row/col/box/cells) for overlay rendering
    overlay: str               # path to rendered overlay frame (filled later by CLI)
    caption: str               # human-friendly explanation
