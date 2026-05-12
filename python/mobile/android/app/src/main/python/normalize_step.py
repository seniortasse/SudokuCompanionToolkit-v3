# normalize_step.py
# Engine -> SolveStepV2 normalizer (Android-ready)
#
# Rewritten as a 3-stage adapter:
#
#   Stage A — ingest raw engine output
#   Stage B — canonical semantic extraction
#   Stage C — build SolveStepV2 from canonical + JSON-safe debug
#
# Core rule:
#   Every outgoing field must be either:
#     - canonical semantic data, or
#     - sanitized debug data
#   Never raw Python-native structures.
#
# This module is intentionally defensive:
# - fully JSON-safe output
# - family-driven semantic normalization
# - explicit semantic completeness
# - robust placement parsing for multiple engine hit shapes
#
# Notes:
# - Keeps rich technique metadata from the previous module
# - Keeps pre/candidate snapshots and explicit target
# - Preserves debug richness via engine_debug_summary + engine_debug_sanitized
# - Supports singles as first-class canonical steps
# - Supports non-single technique families through family normalizers + safe fallback

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Set, Callable

from generator.model import Instance, EMPTY_CHAR
from generator.algo_human import TECHNIQUES
from generator.weights import WEIGHTS
from generator.techniques.options import determine_options_per_cell

DIGITS = "123456789"

# ============================================================================
# Technique metadata
# ============================================================================

TECHNIQUE_FAMILY_DESCRIPTION: Dict[str, str] = {
    "single": (
        "Singles are forced moves found either by candidate elimination in a cell "
        "(naked single) or by digit-position uniqueness in a house "
        "(hidden single / full house)."
    ),
    "box_line_interaction": (
        "Box/Line interactions eliminate candidates by noticing a digit is confined "
        "to a line within a box (pointing) or confined to a box within a line "
        "(claiming)."
    ),
    "multiple_naked": (
        "Naked subsets (pair/triple/quad) occur when a small set of cells in a house "
        "contains a small set of digits; those digits can be removed from the other "
        "cells of that house."
    ),
    "multiple_hidden": (
        "Hidden subsets (pair/triple/quad) occur when a small set of digits can only "
        "go in a small set of cells in a house; those cells can be restricted to "
        "those digits."
    ),
    "multiple_hidden_boxed": (
        "Boxed hidden subsets are engine-specific variants where external constraints "
        "force a small digit set to be the only viable set for a specific cell "
        "(or small structure) inside a box."
    ),
    "fish": (
        "Fish patterns (X-Wing, Swordfish, Jellyfish) eliminate candidates by matching "
        "digit positions across multiple rows/columns forming a constrained cover."
    ),
    "wing": (
        "Wing patterns (e.g., XY-Wing) use bivalue relationships to eliminate a digit "
        "from cells that see both wings."
    ),
    "chain": (
        "Chains propagate alternating candidate truth along linked bivalue cells to "
        "produce eliminations or force a choice."
    ),
    "ring": (
        "Rings/loops are closed bivalue cycles that create eliminations due to "
        "alternation constraints."
    ),
    "boxed_pattern": (
        "Boxed patterns are advanced box-centric structures where box geometry plus "
        "arm constraints produce eliminations or strong restrictions."
    ),
    "leftovers": (
        "Leftovers are engine-specific techniques for irregular/custom box layouts: "
        "a region is split into inside/outside groups across overlapping boxes, and "
        "digit sets are balanced between the groups to remove impossible digits."
    ),
    "unknown": "Unknown/unsupported family.",
}

TECHNIQUE_FRIENDLY_NAME: Dict[str, str] = {
    "singles-1": "Single (Level 1)",
    "singles-2": "Single (Level 2)",
    "singles-naked-2": "Naked Single (Level 2)",
    "singles-3": "Single (Level 3)",
    "singles-naked-3": "Naked Single (Level 3)",
    "doubles-naked": "Naked Pair",
    "leftovers-1": "Leftovers (Level 1)",
    "triplets-naked": "Naked Triplet",
    "quads-naked": "Naked Quad",
    "leftovers-2": "Leftovers (Level 2)",
    "singles-pointing": "Pointing Single",
    "leftovers-3": "Leftovers (Level 3)",
    "singles-boxed": "Boxed Single",
    "doubles": "Hidden Pair",
    "triplets": "Hidden Triplet",
    "quads": "Hidden Quad",
    "x-wings": "X-Wing",
    "leftovers-4": "Leftovers (Level 4)",
    "y-wings": "Y-Wing",
    "leftovers-5": "Leftovers (Level 5)",
    "remote-pairs": "Remote Pairs",
    "leftovers-6": "Leftovers (Level 6)",
    "boxed-doubles": "Boxed Hidden Pair",
    "leftovers-7": "Leftovers (Level 7)",
    "boxed-triplets": "Boxed Hidden Triplet",
    "leftovers-8": "Leftovers (Level 8)",
    "boxed-wings": "Boxed Wings",
    "boxed-rays": "Boxed Rays",
    "ab-rings": "A/B Rings",
    "ab-chains": "A/B Chains",
    "x-wings-3": "X-Wing (Size 3)",
    "boxed-quads": "Boxed Hidden Quad",
    "x-wings-4": "X-Wing (Size 4)",
    "leftovers-9": "Leftovers (Level 9)",
}

TECHNIQUE_FAMILY: Dict[str, str] = {
    "singles": "single",
    "singles-1": "single",
    "singles-2": "single",
    "singles-3": "single",
    "singles-naked-2": "single",
    "singles-naked-3": "single",

    "doubles-naked": "multiple_naked",
    "triplets-naked": "multiple_naked",
    "quads-naked": "multiple_naked",

    "doubles": "multiple_hidden",
    "triplets": "multiple_hidden",
    "quads": "multiple_hidden",

    "boxed-doubles": "multiple_hidden_boxed",
    "boxed-triplets": "multiple_hidden_boxed",
    "boxed-quads": "multiple_hidden_boxed",

    "singles-pointing": "box_line_interaction",
    "singles-boxed": "box_line_interaction",

    "x-wings": "fish",
    "x-wings-3": "fish",
    "x-wings-4": "fish",

    "y-wings": "wing",

    "remote-pairs": "chain",
    "ab-chains": "chain",

    "ab-rings": "ring",

    "boxed-wings": "boxed_pattern",
    "boxed-rays": "boxed_pattern",

    "leftovers-1": "leftovers",
    "leftovers-2": "leftovers",
    "leftovers-3": "leftovers",
    "leftovers-4": "leftovers",
    "leftovers-5": "leftovers",
    "leftovers-6": "leftovers",
    "leftovers-7": "leftovers",
    "leftovers-8": "leftovers",
    "leftovers-9": "leftovers",
}

BASE_TECHNIQUES: Set[str] = {
    "singles-1",
    "singles-2",
    "singles-naked-2",
    "singles-3",
    "singles-naked-3",
}

TECHNIQUE_REAL_NAME: Dict[str, str] = {
    "singles-1": "Full House",
    "singles-2": "Hidden Single (2-house scan)",
    "singles-naked-2": "Naked Single (2-house options)",
    "singles-3": "Hidden Single",
    "singles-naked-3": "Naked Single",

    "doubles-naked": "Naked Pair",
    "triplets-naked": "Naked Triple",
    "quads-naked": "Naked Quad",

    "singles-pointing": "Pointing Pair/Triple",
    "singles-boxed": "Claiming Pair/Triple",

    "doubles": "Hidden Pair",
    "triplets": "Hidden Triple",
    "quads": "Hidden Quad",

    "x-wings": "X-Wing",
    "x-wings-3": "Swordfish",
    "x-wings-4": "Jellyfish",

    "y-wings": "XY-Wing",

    "remote-pairs": "Remote Pairs",
    "ab-chains": "XY-Chain (A/B Chain)",
    "ab-rings": "XY-Loop (A/B Ring)",

    "boxed-doubles": "Boxed Pair Lock (engine variant)",
    "boxed-triplets": "Boxed Triple Lock (engine variant)",
    "boxed-quads": "Boxed Quad Lock (engine variant)",
    "boxed-wings": "Boxed XY-Wing (engine variant)",
    "boxed-rays": "Empty Rectangle / Bent-Line Box Interaction (engine variant)",

    "leftovers-1": "Leftovers (Irregular Boxes) — Level 1",
    "leftovers-2": "Leftovers (Irregular Boxes) — Level 2",
    "leftovers-3": "Leftovers (Irregular Boxes) — Level 3",
    "leftovers-4": "Leftovers (Irregular Boxes) — Level 4",
    "leftovers-5": "Leftovers (Irregular Boxes) — Level 5",
    "leftovers-6": "Leftovers (Irregular Boxes) — Level 6",
    "leftovers-7": "Leftovers (Irregular Boxes) — Level 7",
    "leftovers-8": "Leftovers (Irregular Boxes) — Level 8",
    "leftovers-9": "Leftovers (Irregular Boxes) — Level 9",
}

TECHNIQUE_DIFFICULTY_LEVEL: Dict[str, str] = {
    "singles-1": "Easy",
    "singles-2": "Easy",
    "singles-naked-2": "Easy",
    "singles-3": "Easy",
    "singles-naked-3": "Easy",

    "doubles-naked": "Medium",
    "doubles": "Medium",
    "singles-pointing": "Medium",
    "singles-boxed": "Medium",

    "triplets-naked": "Hard",
    "triplets": "Hard",
    "x-wings": "Hard",

    "quads-naked": "Expert",
    "quads": "Expert",
    "y-wings": "Expert",
    "remote-pairs": "Expert",
    "ab-chains": "Expert",
    "ab-rings": "Expert",
    "x-wings-3": "Expert",
    "x-wings-4": "Expert",

    "boxed-doubles": "Expert",
    "boxed-triplets": "Expert",
    "boxed-quads": "Expert",
    "boxed-wings": "Expert",
    "boxed-rays": "Expert",

    "leftovers-1": "Hard",
    "leftovers-2": "Hard",
    "leftovers-3": "Hard",
    "leftovers-4": "Expert",
    "leftovers-5": "Expert",
    "leftovers-6": "Expert",
    "leftovers-7": "Expert",
    "leftovers-8": "Expert",
    "leftovers-9": "Expert",
}

TECHNIQUE_DEFINITION: Dict[str, Dict[str, Any]] = {
    "singles-1": {
        "definition": {
            "what": "A house has exactly one empty cell, so the missing digit is forced.",
            "why": "No ambiguity remains in that house.",
            "when": "You count eight placed digits in a row/col/box and see one empty cell.",
            "how": "List the missing digit in the house and place it in the empty cell.",
        },
        "comments": {
            "synonyms": ["Full House", "Last Remaining Cell"],
            "notes": "Simplest deterministic placement.",
            "interesting_facts": [
                "Often grouped with hidden singles in apps, but conceptually even simpler."
            ],
        },
    },
    "singles-2": {
        "definition": {
            "what": "A hidden single found with a partial constraint check.",
            "why": "A faster scan can spot forced placements early.",
            "when": "Only one cell survives an extra constraint in a unit.",
            "how": "Scan a house and test helper constraints until one location remains.",
        },
        "comments": {
            "synonyms": ["Hidden Single (quick scan)"],
            "notes": "Engine-level staged scan.",
            "interesting_facts": ["Can reduce work before full option computation."],
        },
    },
    "singles-naked-2": {
        "definition": {
            "what": "A naked single detected from two of the three houses.",
            "why": "Sometimes the third house is unnecessary.",
            "when": "A cell’s allowed digits collapse to one under 2-house constraints.",
            "how": "Compute allowed digits using two constraints; if one remains, place it.",
        },
        "comments": {
            "synonyms": ["Naked Single (partial)"],
            "notes": "Engine staging variant.",
            "interesting_facts": ["Still logically valid if implemented carefully."],
        },
    },
    "singles-3": {
        "definition": {
            "what": "A digit can go in exactly one cell within a row/col/box.",
            "why": "Digit-position uniqueness forces the placement.",
            "when": "All other empty cells in a house are blocked for that digit.",
            "how": "For each house and digit, list candidate cells; if length is 1, place it.",
        },
        "comments": {
            "synonyms": ["Hidden Single"],
            "notes": "Core single technique.",
            "interesting_facts": ["Repeated hidden-single scans cascade quickly."],
        },
    },
    "singles-naked-3": {
        "definition": {
            "what": "A cell has exactly one candidate considering row, column, and box.",
            "why": "Only one digit fits the cell.",
            "when": "Eliminations reduce a cell’s candidate set to size 1.",
            "how": "Compute all allowed digits in the cell; if only one remains, place it.",
        },
        "comments": {
            "synonyms": ["Naked Single", "Single Candidate"],
            "notes": "Modeled canonically as cell-centric proof.",
            "interesting_facts": ["Often a consequence of deeper logic."],
        },
    },
    "doubles-naked": {
        "definition": {
            "what": "Two cells in a house contain only the same two digits.",
            "why": "Those digits are locked there and can be removed elsewhere in the house.",
            "when": "Two cells have identical size-2 candidate sets, or equivalent pair lock.",
            "how": "Identify the pair, then remove those digits from the other cells in the house.",
        },
        "comments": {
            "synonyms": ["Naked Pair"],
            "notes": "Common mid-level technique.",
            "interesting_facts": ["Frequently triggers singles afterward."],
        },
    },
    "triplets-naked": {
        "definition": {
            "what": "Three cells in a house collectively contain only three digits.",
            "why": "Those digits are locked to those cells.",
            "when": "Three cells have candidate union size 3.",
            "how": "Remove those digits from the other cells in the house.",
        },
        "comments": {
            "synonyms": ["Naked Triple", "Naked Triplet"],
            "notes": "Union-based subset.",
            "interesting_facts": ["Can appear as distributed pairs like {1,2}, {1,3}, {2,3}."],
        },
    },
    "quads-naked": {
        "definition": {
            "what": "Four cells in a house collectively contain only four digits.",
            "why": "Those digits are locked to those cells.",
            "when": "Four cells have candidate union size 4.",
            "how": "Remove those digits from the other cells in the house.",
        },
        "comments": {
            "synonyms": ["Naked Quad"],
            "notes": "Rare but powerful.",
            "interesting_facts": ["Can produce big cleanup in hard puzzles."],
        },
    },
    "singles-pointing": {
        "definition": {
            "what": "In a box, all candidates for a digit lie in one line.",
            "why": "That digit must occupy that line within the box.",
            "when": "Digit candidates in a box are confined to one row or column.",
            "how": "Eliminate the digit from the rest of that row/column outside the box.",
        },
        "comments": {
            "synonyms": ["Pointing Pair", "Pointing Triple", "Pointing"],
            "notes": "Classic box-line interaction.",
            "interesting_facts": ["Mirror family of claiming."],
        },
    },
    "singles-boxed": {
        "definition": {
            "what": "In a line, all candidates for a digit lie in one box.",
            "why": "That line must place the digit within that box.",
            "when": "Digit candidates in a row/col are confined to one box.",
            "how": "Eliminate the digit from the rest of that box outside the line.",
        },
        "comments": {
            "synonyms": ["Claiming Pair", "Claiming Triple", "Claiming"],
            "notes": "Classic line-box interaction.",
            "interesting_facts": ["Mirror image of pointing."],
        },
    },
    "doubles": {
        "definition": {
            "what": "Two digits in a house can only go in the same two cells.",
            "why": "Those two cells can be restricted to those digits.",
            "when": "Digit-location scan shows exactly two shared support cells.",
            "how": "Remove all other digits from those two cells.",
        },
        "comments": {
            "synonyms": ["Hidden Pair"],
            "notes": "Digit-location driven, not cell-union driven.",
            "interesting_facts": ["Best visualized by candidate locations per digit."],
        },
    },
    "triplets": {
        "definition": {
            "what": "Three digits in a house can only go in the same three cells.",
            "why": "Those cells can be restricted to those digits.",
            "when": "Digit-location scan yields a 3-digit/3-cell cover.",
            "how": "Remove other candidates from those support cells.",
        },
        "comments": {
            "synonyms": ["Hidden Triple", "Hidden Triplet"],
            "notes": "Digit-location subset.",
            "interesting_facts": ["Easy to confuse with naked triples."],
        },
    },
    "quads": {
        "definition": {
            "what": "Four digits in a house can only go in the same four cells.",
            "why": "Those cells can be restricted to those digits.",
            "when": "Digit-location scan yields a 4-digit/4-cell cover.",
            "how": "Remove other candidates from those support cells.",
        },
        "comments": {
            "synonyms": ["Hidden Quad"],
            "notes": "Rare expert subset.",
            "interesting_facts": ["Hard to spot without candidate visualization."],
        },
    },
    "x-wings": {
        "definition": {
            "what": "Two base houses restrict a digit to the same two cover houses.",
            "why": "That rectangle lets you eliminate the digit elsewhere in the cover houses.",
            "when": "A digit appears twice in two rows with matching columns, or vice versa.",
            "how": "Find the fish pattern and eliminate outside the fish.",
        },
        "comments": {
            "synonyms": ["X-Wing", "Fish (size 2)"],
            "notes": "First fish most players learn.",
            "interesting_facts": ["Geometrically simple fish pattern."],
        },
    },
    "x-wings-3": {
        "definition": {
            "what": "A fish pattern of size 3.",
            "why": "Three base houses restrict the digit to three cover houses.",
            "when": "Union of candidate columns/rows across three houses is size 3.",
            "how": "Eliminate outside the fish cover.",
        },
        "comments": {
            "synonyms": ["Swordfish", "Fish (size 3)"],
            "notes": "Standard real name is Swordfish.",
            "interesting_facts": ["Usually appears after easier logic is exhausted."],
        },
    },
    "x-wings-4": {
        "definition": {
            "what": "A fish pattern of size 4.",
            "why": "Four base houses restrict the digit to four cover houses.",
            "when": "Union of candidate columns/rows across four houses is size 4.",
            "how": "Eliminate outside the fish cover.",
        },
        "comments": {
            "synonyms": ["Jellyfish", "Fish (size 4)"],
            "notes": "Standard real name is Jellyfish.",
            "interesting_facts": ["Rare but legitimate in expert puzzles."],
        },
    },
    "y-wings": {
        "definition": {
            "what": "A pivot (A/B) sees two wings (A/C and B/C).",
            "why": "One wing must become C, so C can be removed from cells seeing both wings.",
            "when": "Three bivalue cells satisfy the XY-Wing geometry.",
            "how": "Identify pivot and wings, then eliminate the shared wing digit.",
        },
        "comments": {
            "synonyms": ["XY-Wing", "Y-Wing"],
            "notes": "Strict visibility matters.",
            "interesting_facts": ["An approachable introduction to chain-style reasoning."],
        },
    },
    "remote-pairs": {
        "definition": {
            "what": "A chain of same-pair bivalue cells creates alternating truth.",
            "why": "Cells seeing both endpoints can lose a candidate.",
            "when": "A visible chain of repeated two-digit cells exists.",
            "how": "Use endpoint logic to eliminate from intersecting cells.",
        },
        "comments": {
            "synonyms": ["Remote Pairs"],
            "notes": "Specialized alternating chain.",
            "interesting_facts": ["A coloring-flavored technique."],
        },
    },
    "ab-chains": {
        "definition": {
            "what": "A bivalue forcing chain propagates implications through links.",
            "why": "Endpoint logic creates eliminations or forces.",
            "when": "A valid alternating link chain exists.",
            "how": "Follow implications and use endpoint consequences.",
        },
        "comments": {
            "synonyms": ["XY-Chain", "A/B Chain"],
            "notes": "Your engine label differs from the common human label.",
            "interesting_facts": ["XY-Wing is a short XY-Chain."],
        },
    },
    "ab-rings": {
        "definition": {
            "what": "A closed bivalue loop creates parity-based eliminations.",
            "why": "Alternation around the loop fixes relationships.",
            "when": "A consistent closed chain exists.",
            "how": "Construct the loop and eliminate via loop rules.",
        },
        "comments": {
            "synonyms": ["XY-Loop", "A/B Ring"],
            "notes": "Loop correctness matters.",
            "interesting_facts": ["Many advanced techniques can be graph-loop views."],
        },
    },
    "boxed-wings": {
        "definition": {
            "what": "A box-centric wing pattern creates eliminations via two arms.",
            "why": "The box plus arm constraints force a shared consequence.",
            "when": "A valid box-and-arms structure exists.",
            "how": "Identify box context, two wings, and eliminate from intersections.",
        },
        "comments": {
            "synonyms": ["Boxed XY-Wing (variant)"],
            "notes": "Engine-specific variant.",
            "interesting_facts": ["Best taught visually."],
        },
    },
    "boxed-rays": {
        "definition": {
            "what": "A bent-line box interaction comparable to Empty Rectangle ideas.",
            "why": "The box bend plus arm constraints eliminates a digit elsewhere.",
            "when": "A row/column bend exists inside a box with valid arm conditions.",
            "how": "Identify bend geometry and eliminate in the target region.",
        },
        "comments": {
            "synonyms": ["Empty Rectangle (variant)"],
            "notes": "Engine-specific naming.",
            "interesting_facts": ["Shape-based explanation works best."],
        },
    },
    "boxed-doubles": {
        "definition": {
            "what": "An engine-specific box-centric hidden pair lock.",
            "why": "External constraints restrict a box target to two digits.",
            "when": "Arms converge on a box target cell or structure.",
            "how": "Use external support to restrict the target.",
        },
        "comments": {
            "synonyms": ["Boxed Hidden Pair (variant)"],
            "notes": "Engine-specific.",
            "interesting_facts": ["Better explained as candidate restriction, not raw arm mechanics."],
        },
    },
    "boxed-triplets": {
        "definition": {
            "what": "An engine-specific box-centric hidden triple lock.",
            "why": "External constraints restrict a target to three digits.",
            "when": "Arms and box context force a 3-digit restriction.",
            "how": "Build the support set and restrict the target.",
        },
        "comments": {
            "synonyms": ["Boxed Hidden Triple (variant)"],
            "notes": "Engine-specific.",
            "interesting_facts": ["Outcome-driven explanation is better than internal mechanics."],
        },
    },
    "boxed-quads": {
        "definition": {
            "what": "An engine-specific box-centric hidden quad lock.",
            "why": "External constraints restrict a target to four digits.",
            "when": "Arms and box context force a 4-digit restriction.",
            "how": "Build the support set and restrict the target.",
        },
        "comments": {
            "synonyms": ["Boxed Hidden Quad (variant)"],
            "notes": "Engine-specific.",
            "interesting_facts": ["Useful to preserve as semantic restriction even without immediate placement."],
        },
    },
}

def _leftovers_definition(level: int) -> Dict[str, Any]:
    return {
        "definition": {
            "what": (
                "An engine-specific irregular-box technique where a region is split into "
                "inside/outside groups across overlapping custom boxes."
            ),
            "why": (
                "Custom box overlaps create balanced digit-set constraints that remove "
                "impossible digits."
            ),
            "when": (
                "A custom-layout leftovers pattern exists "
                f"(engine level {level} reflects group/application size)."
            ),
            "how": (
                "Compute inside/outside groups, compare digit support, and remove digits "
                "that cannot stay on one side."
            ),
        },
        "comments": {
            "synonyms": ["Leftovers (Irregular Boxes)", "Region Balance (variant)"],
            "notes": f"Level {level} is engine-defined.",
            "interesting_facts": [
                "Best explained visually with highlighted groups.",
                "Not a mainstream textbook label, but a valid constraint pattern.",
            ],
        },
    }

for _k in range(1, 10):
    TECHNIQUE_DEFINITION[f"leftovers-{_k}"] = _leftovers_definition(_k)

def technique_meta(technique_id: str) -> Dict[str, Any]:
    tid = (technique_id or "unknown").strip() or "unknown"

    try:
        priority_rank = TECHNIQUES.index(tid) + 1
    except Exception:
        priority_rank = 0

    difficulty_weight = int(WEIGHTS.get(tid, 0))
    family = TECHNIQUE_FAMILY.get(tid, "unknown")
    family_desc = TECHNIQUE_FAMILY_DESCRIPTION.get(family, "")
    friendly_name = TECHNIQUE_FRIENDLY_NAME.get(tid, tid)
    real_name = TECHNIQUE_REAL_NAME.get(tid, friendly_name)
    difficulty_level = TECHNIQUE_DIFFICULTY_LEVEL.get(tid, "Unknown")

    def_pack = TECHNIQUE_DEFINITION.get(tid, {})
    definition = def_pack.get("definition", {})
    comments = def_pack.get("comments", {})
    is_base = tid in BASE_TECHNIQUES

    return {
        "technique_id": tid,
        "technique_name": friendly_name,
        "family": family,
        "is_base": is_base,
        "priority_rank": int(priority_rank),
        "difficulty_weight": int(difficulty_weight),

        "app_name": tid,
        "real_name": real_name,
        "family_description": family_desc,
        "difficulty_level": difficulty_level,
        "definition": {
            "what": str(definition.get("what", "")),
            "why": str(definition.get("why", "")),
            "when": str(definition.get("when", "")),
            "how": str(definition.get("how", "")),
        },
        "comments": {
            "synonyms": list(comments.get("synonyms", []) or []),
            "notes": str(comments.get("notes", "")),
            "interesting_facts": list(comments.get("interesting_facts", []) or []),
        },

        # Compat aliases
        "appName": tid,
        "realName": real_name,
        "familyDescription": family_desc,
        "difficultyLevel": difficulty_level,
        "isBase": is_base,
        "priorityRank": int(priority_rank),
        "difficultyWeight": int(difficulty_weight),
    }

# ============================================================================
# Basic helpers
# ============================================================================

def sha12(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]

def rc_to_cell_index(r_1to9: int, c_1to9: int) -> int:
    return (r_1to9 - 1) * 9 + (c_1to9 - 1)

def cell_index_to_rc(cell_index: int) -> Tuple[int, int]:
    return (cell_index // 9) + 1, (cell_index % 9) + 1

def box_index_1to9(r_1to9: int, c_1to9: int) -> int:
    return ((r_1to9 - 1) // 3) * 3 + ((c_1to9 - 1) // 3) + 1

def char_to_digit(ch: Any) -> Optional[int]:
    if ch is None:
        return None
    s = str(ch)
    if s in DIGITS:
        return int(s)
    return None

def candidates_set_to_mask(cands: Set[str]) -> int:
    mask = 0
    for ch in cands:
        d = char_to_digit(ch)
        if d is not None:
            mask |= (1 << (d - 1))
    return mask

def normalize_house(house_type: str, index_1to9: int) -> Dict[str, Any]:
    return {"type": str(house_type), "index1to9": int(index_1to9)}

def house_label(house: Optional[Dict[str, Any]]) -> str:
    if not isinstance(house, dict):
        return "the house"

    h_type = str(house.get("type", "")).strip().lower()
    idx = house.get("index1to9")

    try:
        idx1 = int(idx)
    except Exception:
        idx1 = None

    if h_type == "row" and idx1 in range(1, 10):
        return f"row {idx1}"
    if h_type in {"col", "column"} and idx1 in range(1, 10):
        return f"column {idx1}"
    if h_type == "box" and idx1 in range(1, 10):
        return f"box {idx1}"
    if h_type == "region" and idx is not None:
        return f"region {idx}"

    if idx1 in range(1, 10) and h_type:
        return f"{h_type} {idx1}"
    if h_type:
        return h_type
    return "the house"

def normalize_custom_region(region_id: Any) -> Dict[str, Any]:
    return {"type": "region", "regionId": str(region_id)}

def cell_ref_from_rc0(i1: int, i2: int) -> Dict[str, Any]:
    r, c = int(i1) + 1, int(i2) + 1
    return {"cellIndex": rc_to_cell_index(r, c), "r": r, "c": c}

def cell_ref_from_rc1(r: int, c: int) -> Dict[str, Any]:
    return {"cellIndex": rc_to_cell_index(int(r), int(c)), "r": int(r), "c": int(c)}

def cell_ref_from_index(ci: int) -> Dict[str, Any]:
    r, c = cell_index_to_rc(int(ci))
    return {"cellIndex": int(ci), "r": r, "c": c}

def cell_ref_label(cell: Optional[Dict[str, Any]]) -> str:
    if not isinstance(cell, dict):
        return "the cell"

    try:
        r = int(cell.get("r"))
        c = int(cell.get("c"))
        if r in range(1, 10) and c in range(1, 10):
            return f"r{r}c{c}"
    except Exception:
        pass

    ci = cell.get("cellIndex")
    try:
        ci_int = int(ci)
        if 0 <= ci_int <= 80:
            r, c = cell_index_to_rc(ci_int)
            return f"r{r}c{c}"
    except Exception:
        pass

    return "the cell"

def instance_to_grid81(inst: Instance) -> str:
    out: List[str] = []
    for r in range(9):
        for c in range(9):
            ch = None
            try:
                ch = inst[r][c]
            except Exception:
                pass
            if ch is None:
                try:
                    ch = inst.grid[r][c]
                except Exception:
                    ch = EMPTY_CHAR
            out.append("." if ch == EMPTY_CHAR else str(ch))
    return "".join(out)

def houses_for_cell(ci: int) -> List[Dict[str, Any]]:
    r, c = cell_index_to_rc(ci)
    return [
        normalize_house("row", r),
        normalize_house("col", c),
        normalize_house("box", box_index_1to9(r, c)),
    ]

def _house_cells(h_type: str, idx1: int) -> List[int]:
    if h_type == "row":
        return [rc_to_cell_index(idx1, c) for c in range(1, 10)]
    if h_type == "col":
        return [rc_to_cell_index(r, idx1) for r in range(1, 10)]
    b = idx1
    br = ((b - 1) // 3) * 3 + 1
    bc = ((b - 1) % 3) * 3 + 1
    out: List[int] = []
    for dr in range(3):
        for dc in range(3):
            out.append(rc_to_cell_index(br + dr, bc + dc))
    return out

def _digits_from_mask(mask: int) -> List[int]:
    return [d for d in range(1, 10) if mask & (1 << (d - 1))]

def _mask_for_cell(options_all_masks: Dict[str, int], cell_index: int) -> int:
    return int(options_all_masks.get(str(cell_index), 0))

# ============================================================================
# JSON-safe sanitizer + debug summaries
# ============================================================================

def _looks_like_rc0_tuple(x: Any) -> bool:
    return (
        isinstance(x, (tuple, list))
        and len(x) == 2
        and all(isinstance(v, int) for v in x)
        and 0 <= int(x[0]) <= 8
        and 0 <= int(x[1]) <= 8
    )

def _looks_like_rc1_tuple(x: Any) -> bool:
    return (
        isinstance(x, (tuple, list))
        and len(x) == 2
        and all(isinstance(v, int) for v in x)
        and 1 <= int(x[0]) <= 9
        and 1 <= int(x[1]) <= 9
    )

def to_json_safe(value: Any, path: str = "root") -> Any:
    """
    Fully recursive JSON-safe sanitizer.
    Rules:
      - tuple/list/set -> JSON-safe list or structured wrapper
      - dict with non-string keys -> map_entries wrapper
      - coordinate tuples recognized -> cell refs where appropriate
      - custom objects -> repr wrapper
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    # Prefer cell semantic conversion for coordinate pairs in debug too
    if _looks_like_rc0_tuple(value):
        return {
            "__kind__": "cell_rc0",
            "cell": cell_ref_from_rc0(int(value[0]), int(value[1])),
        }

    if _looks_like_rc1_tuple(value):
        return {
            "__kind__": "cell_rc1",
            "cell": cell_ref_from_rc1(int(value[0]), int(value[1])),
        }

    if isinstance(value, list):
        return [to_json_safe(v, f"{path}[{i}]") for i, v in enumerate(value)]

    if isinstance(value, tuple):
        return {
            "__kind__": "tuple",
            "items": [to_json_safe(v, f"{path}[{i}]") for i, v in enumerate(value)],
        }

    if isinstance(value, set):
        items = list(value)
        try:
            items = sorted(items, key=lambda x: str(x))
        except Exception:
            pass
        return {
            "__kind__": "set",
            "items": [to_json_safe(v, f"{path}{{set}}") for v in items],
        }

    if isinstance(value, dict):
        if all(isinstance(k, str) for k in value.keys()):
            return {str(k): to_json_safe(v, f"{path}.{k}") for k, v in value.items()}
        entries: List[Dict[str, Any]] = []
        for k, v in value.items():
            entries.append({
                "key": to_json_safe(k, f"{path}.<key>"),
                "value": to_json_safe(v, f"{path}.{str(k)}"),
            })
        return {"__kind__": "map_entries", "entries": entries}

    # Fallback for unknown/custom objects
    return {
        "__kind__": "repr",
        "python_type": type(value).__name__,
        "repr": repr(value),
    }

def debug_shape_summary(value: Any) -> Dict[str, Any]:
    """
    Cheap structured summary of a raw engine-native object.
    """
    summary = {
        "python_type": type(value).__name__,
        "is_none": value is None,
        "tuple_arity": len(value) if isinstance(value, tuple) else None,
        "list_len": len(value) if isinstance(value, list) else None,
        "set_len": len(value) if isinstance(value, set) else None,
        "dict_len": len(value) if isinstance(value, dict) else None,
        "contains_sets": False,
        "contains_tuple_keys": False,
        "contains_tuple_values": False,
        "cell_like_count": 0,
        "digit_like_count": 0,
        "top_level_preview_types": [],
    }

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                if isinstance(k, tuple):
                    summary["contains_tuple_keys"] = True
                if _looks_like_rc0_tuple(k) or _looks_like_rc1_tuple(k):
                    summary["cell_like_count"] += 1
                walk(k)
                walk(v)
        elif isinstance(x, tuple):
            summary["contains_tuple_values"] = True
            if _looks_like_rc0_tuple(x) or _looks_like_rc1_tuple(x):
                summary["cell_like_count"] += 1
            for it in x:
                walk(it)
        elif isinstance(x, list):
            for it in x:
                walk(it)
        elif isinstance(x, set):
            summary["contains_sets"] = True
            for it in x:
                walk(it)
        else:
            d = char_to_digit(x)
            if d is not None:
                summary["digit_like_count"] += 1

    walk(value)

    if isinstance(value, (list, tuple, set)):
        try:
            preview_items = list(value)[:8]
        except Exception:
            preview_items = []
        summary["top_level_preview_types"] = [type(x).__name__ for x in preview_items]
    elif isinstance(value, dict):
        preview_items = list(value.items())[:8]
        summary["top_level_preview_types"] = [
            f"{type(k).__name__}->{type(v).__name__}" for k, v in preview_items
        ]

    return summary

# ============================================================================
# Stage A — raw adapters
# ============================================================================

@dataclass
class PlacementHit:
    cellIndex: int
    r: int
    c: int
    digit: int
    dimension: Optional[str] = None
    idx_dim_0based: Optional[int] = None
    source_shape: str = "unknown"
    raw_debug: Any = None

@dataclass
class RawApplication:
    technique_id: str
    name_application: str
    application_details: Any
    removed_chars_updated: Any
    engine_debug_summary: Dict[str, Any]
    engine_debug_sanitized: Any

@dataclass
class RawCleanupStep:
    technique_latest: Any
    technique_cleanup: Any
    iteration: Any
    options_before: Any
    applications: List[RawApplication]

def parse_placements_from_new_values(new_values: Any) -> Tuple[List[PlacementHit], List[Dict[str, Any]]]:
    """
    Robust placement parser with diagnostics.

    Recognized shapes:
      A) ((i1,i2), char, dim)
      B) (dim, idx_dim, [(i1,i2), ...], char)
      C) ((i1,i2), char, dim, details)
      D) (dim, idx_dim, [(i1,i2), ...], char, details)
    """
    out: List[PlacementHit] = []
    diagnostics: List[Dict[str, Any]] = []

    if not new_values:
        return out, diagnostics

    for idx, item in enumerate(new_values):
        parsed_any = False

        # Shape A
        if isinstance(item, (list, tuple)) and len(item) == 3:
            idx_part, ch, dim = item
            if isinstance(idx_part, (list, tuple)) and len(idx_part) == 2:
                d = char_to_digit(ch)
                if d is not None:
                    i1, i2 = int(idx_part[0]), int(idx_part[1])
                    ref = cell_ref_from_rc0(i1, i2)
                    out.append(PlacementHit(
                        cellIndex=ref["cellIndex"],
                        r=ref["r"],
                        c=ref["c"],
                        digit=d,
                        dimension=str(dim),
                        idx_dim_0based=None,
                        source_shape="shape_A_direct_hit",
                        raw_debug=to_json_safe(item),
                    ))
                    parsed_any = True

        # Shape B or C
        if not parsed_any and isinstance(item, (list, tuple)) and len(item) == 4:
            a, b, c, d = item

            # Shape C: ((i1,i2), char, dim, details)
            if isinstance(a, (list, tuple)) and len(a) == 2:
                digit = char_to_digit(b)
                if digit is not None:
                    i1, i2 = int(a[0]), int(a[1])
                    ref = cell_ref_from_rc0(i1, i2)
                    out.append(PlacementHit(
                        cellIndex=ref["cellIndex"],
                        r=ref["r"],
                        c=ref["c"],
                        digit=digit,
                        dimension=str(c),
                        idx_dim_0based=None,
                        source_shape="shape_C_direct_hit_with_details",
                        raw_debug=to_json_safe(item),
                    ))
                    parsed_any = True

            # Shape B: (dim, idx_dim, idxs, char)
            if not parsed_any:
                dim, idx_dim, idxs, ch = a, b, c, d
                digit = char_to_digit(ch)
                if digit is not None and isinstance(idxs, (list, tuple)):
                    local_parsed = False
                    for cellish in idxs:
                        if isinstance(cellish, (list, tuple)) and len(cellish) == 2:
                            i1, i2 = int(cellish[0]), int(cellish[1])
                            ref = cell_ref_from_rc0(i1, i2)
                            out.append(PlacementHit(
                                cellIndex=ref["cellIndex"],
                                r=ref["r"],
                                c=ref["c"],
                                digit=digit,
                                dimension=str(dim),
                                idx_dim_0based=int(idx_dim) if isinstance(idx_dim, int) else None,
                                source_shape="shape_B_grouped_hit",
                                raw_debug=to_json_safe(item),
                            ))
                            local_parsed = True
                    parsed_any = local_parsed

        # Shape D
        if not parsed_any and isinstance(item, (list, tuple)) and len(item) == 5:
            dim, idx_dim, idxs, ch, _details = item
            digit = char_to_digit(ch)
            if digit is not None and isinstance(idxs, (list, tuple)):
                local_parsed = False
                for cellish in idxs:
                    if isinstance(cellish, (list, tuple)) and len(cellish) == 2:
                        i1, i2 = int(cellish[0]), int(cellish[1])
                        ref = cell_ref_from_rc0(i1, i2)
                        out.append(PlacementHit(
                            cellIndex=ref["cellIndex"],
                            r=ref["r"],
                            c=ref["c"],
                            digit=digit,
                            dimension=str(dim),
                            idx_dim_0based=int(idx_dim) if isinstance(idx_dim, int) else None,
                            source_shape="shape_D_grouped_hit_with_details",
                            raw_debug=to_json_safe(item),
                        ))
                        local_parsed = True
                parsed_any = local_parsed

        if not parsed_any:
            diagnostics.append({
                "index": idx,
                "code": "unparsed_new_values_item",
                "shape_summary": debug_shape_summary(item),
                "raw_sanitized": to_json_safe(item),
            })

    return out, diagnostics

def parse_cleanup_steps(cleanup_steps: Any, grid81_before: str = "") -> List[RawCleanupStep]:
    out: List[RawCleanupStep] = []
    for cs in (cleanup_steps or []):
        try:
            tech_latest, tech_cleanup, iteration, options_before, details_updated = cs
        except Exception:
            continue

        apps: List[RawApplication] = []
        for du in (details_updated or []):
            try:
                (t_id, name_app), app_details, removed_chars_updated = du
            except Exception:
                # Keep malformed record as debug-only pseudo-app if desired
                apps.append(RawApplication(
                    technique_id=str(tech_cleanup or "unknown"),
                    name_application="malformed_cleanup_detail",
                    application_details=du,
                    removed_chars_updated=[],
                    engine_debug_summary=debug_shape_summary(du),
                    engine_debug_sanitized=to_json_safe(du),
                ))
                continue

            apps.append(RawApplication(
                technique_id=str(t_id),
                name_application=str(name_app),
                application_details=app_details,
                removed_chars_updated=removed_chars_updated,
                engine_debug_summary={
                    "application_details": debug_shape_summary(app_details),
                    "removed_chars_updated": debug_shape_summary(removed_chars_updated),
                    "grid81_before": grid81_before or "",
                },
                engine_debug_sanitized={
                    "application_details": to_json_safe(app_details),
                    "removed_chars_updated": to_json_safe(removed_chars_updated),
                },
            ))

        out.append(RawCleanupStep(
            technique_latest=tech_latest,
            technique_cleanup=tech_cleanup,
            iteration=iteration,
            options_before=options_before,
            applications=apps,
        ))
    return out


def parse_engine_step_trace(engine_step_trace: Any) -> Dict[str, Any]:
    """
    Series 3 compatibility seam:
    - old engine logs may not have a trace
    - new engine logs append trace as the 7th tuple element
    """
    if not isinstance(engine_step_trace, dict):
        return {}
    return to_json_safe(engine_step_trace)

# ============================================================================
# Stage B — canonical semantic extraction
# ============================================================================

@dataclass
class CanonicalEffectPlacement:
    cell: Dict[str, Any]
    digit: int
    source: str = "technique_effect"

@dataclass
class CanonicalEffectElimination:
    cell: Dict[str, Any]
    digit: int
    source: str = "technique_effect"

@dataclass
class CanonicalEffectRestriction:
    cell: Dict[str, Any]
    allowed_digits: List[int]
    removed_digits: List[int]
    source: str = "technique_effect"

@dataclass
class CanonicalTechniqueApplication:
    application_id: str
    technique_id: str
    technique_family: str
    technique_real_name: str
    application_kind: str  # placement/elimination/restriction/mixed
    semantic_completeness: str  # full/partial/debug_only

    # Main semantic actors
    focus_cells: List[Dict[str, Any]] = field(default_factory=list)
    pattern_cells: List[Dict[str, Any]] = field(default_factory=list)
    peer_cells: List[Dict[str, Any]] = field(default_factory=list)
    target_cells: List[Dict[str, Any]] = field(default_factory=list)
    witness_cells: List[Dict[str, Any]] = field(default_factory=list)
    houses: List[Dict[str, Any]] = field(default_factory=list)
    digits: List[int] = field(default_factory=list)

    # Effects
    placements: List[CanonicalEffectPlacement] = field(default_factory=list)
    candidate_eliminations: List[CanonicalEffectElimination] = field(default_factory=list)
    candidate_restrictions: List[CanonicalEffectRestriction] = field(default_factory=list)
    cell_value_forces: List[CanonicalEffectPlacement] = field(default_factory=list)

    # Pattern semantics
    pattern_type: str = "unknown"
    pattern_subtype: Optional[str] = None
    anchors: List[Dict[str, Any]] = field(default_factory=list)
    units_scanned: List[Dict[str, Any]] = field(default_factory=list)
    cover_sets: List[Dict[str, Any]] = field(default_factory=list)
    constraint_explanation: List[str] = field(default_factory=list)
    roles: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    # Canonical support links (especially important for singles)
    explanation_links: List[Dict[str, Any]] = field(default_factory=list)

    # Narration support
    narrative_archetype: str = "UNKNOWN"
    trigger_facts: List[str] = field(default_factory=list)
    confrontation_facts: List[str] = field(default_factory=list)
    resolution_facts: List[str] = field(default_factory=list)
    narrative_role: str = "setup"
    summary_fact: str = ""

    # Safe debug
    engine_debug_summary: Dict[str, Any] = field(default_factory=dict)
    engine_debug_sanitized: Any = None
    final_canonical_proof: Dict[str, Any] = field(default_factory=dict)

# ----------------------------------------------------------------------------
# Candidate snapshots helpers
# ----------------------------------------------------------------------------

def options_snapshot_all_cells(options_grid: Any, grid81_before: str) -> Dict[str, int]:
    snap: Dict[str, int] = {}
    for ci in range(81):
        if ci < len(grid81_before) and grid81_before[ci] in DIGITS:
            snap[str(ci)] = 0
            continue
        i1, i2 = ci // 9, ci % 9
        try:
            cset = options_grid[i1][i2]
            snap[str(ci)] = candidates_set_to_mask(set(cset))
        except Exception:
            snap[str(ci)] = 0
    return snap

def digit_status_from_grid81(grid81: str) -> Dict[str, Any]:
    status: Dict[str, Any] = {}
    for d in range(1, 10):
        solved_cells: List[int] = [i for i, ch in enumerate(grid81) if ch == str(d)]
        solved = len(solved_cells) >= 9
        by_row: Dict[str, List[int]] = {}
        by_col: Dict[str, List[int]] = {}
        by_box: Dict[str, List[int]] = {}
        for ci in solved_cells:
            r, c = cell_index_to_rc(ci)
            b = box_index_1to9(r, c)
            by_row.setdefault(str(r), []).append(ci)
            by_col.setdefault(str(c), []).append(ci)
            by_box.setdefault(str(b), []).append(ci)
        status[str(d)] = {
            "digit": d,
            "status": "SOLVED" if solved else "UNSOLVED",
            "count": len(solved_cells),
            "solved_cells": solved_cells,
            "solved_by_house": {"row": by_row, "col": by_col, "box": by_box},
        }
    return status

def candidate_cells_by_house_from_masks(options_all_masks: Dict[str, int], grid81: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    empty = [i for i, ch in enumerate(grid81) if ch == "."]

    rows = {r: [rc_to_cell_index(r, c) for c in range(1, 10)] for r in range(1, 10)}
    cols = {c: [rc_to_cell_index(r, c) for r in range(1, 10)] for c in range(1, 10)}
    boxes: Dict[int, List[int]] = {b: [] for b in range(1, 10)}
    for r in range(1, 10):
        for c in range(1, 10):
            boxes[box_index_1to9(r, c)].append(rc_to_cell_index(r, c))

    for d in range(1, 10):
        bit = 1 << (d - 1)
        by_row: Dict[str, List[int]] = {}
        by_col: Dict[str, List[int]] = {}
        by_box: Dict[str, List[int]] = {}
        for r, cells in rows.items():
            by_row[str(r)] = [ci for ci in cells if ci in empty and (int(options_all_masks.get(str(ci), 0)) & bit)]
        for c, cells in cols.items():
            by_col[str(c)] = [ci for ci in cells if ci in empty and (int(options_all_masks.get(str(ci), 0)) & bit)]
        for b, cells in boxes.items():
            by_box[str(b)] = [ci for ci in cells if ci in empty and (int(options_all_masks.get(str(ci), 0)) & bit)]
        out[str(d)] = {"digit": d, "candidate_cells_by_house": {"row": by_row, "col": by_col, "box": by_box}}
    return out

# ----------------------------------------------------------------------------
# Removed-char / generic cleanup extraction helpers
# ----------------------------------------------------------------------------

def _elim_entries_from_removed_chars(removed_chars_updated: Any) -> List[CanonicalEffectElimination]:
    out: List[CanonicalEffectElimination] = []
    for item in (removed_chars_updated or []):
        try:
            (i1, i2), ch = item
        except Exception:
            continue
        d = char_to_digit(ch)
        if d is None:
            continue
        ref = cell_ref_from_rc0(int(i1), int(i2))
        out.append(CanonicalEffectElimination(cell=ref, digit=d))
    return out

def _cell_set_from_eliminations(elims: List[CanonicalEffectElimination]) -> List[Dict[str, Any]]:
    seen: Dict[int, Dict[str, Any]] = {}
    for e in elims:
        seen[e.cell["cellIndex"]] = e.cell
    return [seen[k] for k in sorted(seen.keys())]

def _digits_from_eliminations(elims: List[CanonicalEffectElimination]) -> List[int]:
    return sorted({e.digit for e in elims})

def _houses_from_cells(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Set[Tuple[str, int]] = set()
    out: List[Dict[str, Any]] = []
    for c in cells:
        ci = int(c["cellIndex"])
        for h in houses_for_cell(ci):
            key = (str(h["type"]), int(h["index1to9"]))
            if key not in seen:
                seen.add(key)
                out.append(h)
    return out

def _best_common_house(cells: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not cells:
        return None
    cis = [int(c["cellIndex"]) for c in cells]
    house_counts: Dict[Tuple[str, int], int] = {}
    for ci in cis:
        for h in houses_for_cell(ci):
            key = (str(h["type"]), int(h["index1to9"]))
            house_counts[key] = house_counts.get(key, 0) + 1
    full = [(k, v) for k, v in house_counts.items() if v == len(cis)]
    if not full:
        return None
    # Prefer row/col/box order for determinism
    order = {"row": 0, "col": 1, "box": 2}
    full.sort(key=lambda kv: (order.get(kv[0][0], 99), kv[0][1]))
    (h_type, idx1), _ = full[0]
    return normalize_house(h_type, idx1)

def _detect_family(technique_id: str) -> str:
    return TECHNIQUE_FAMILY.get((technique_id or "").strip(), "unknown")

def _detect_archetype(technique_id: str) -> str:
    family = _detect_family(technique_id)
    if family == "single":
        tid = (technique_id or "").strip()
        if tid in {"singles-naked-2", "singles-naked-3"}:
            return "NAKED_SINGLES"
        return "HIDDEN_SINGLES"
    if family in {"multiple_naked", "multiple_hidden", "multiple_hidden_boxed"}:
        return "SUBSETS"
    if family == "box_line_interaction":
        return "INTERSECTIONS"
    if family == "fish":
        return "FISH"
    if family in {"wing", "boxed_pattern"}:
        return "WINGS"
    if family in {"chain", "ring"}:
        return "CHAINS"
    return "UNKNOWN"

# ----------------------------------------------------------------------------
# Singles support / impacts (kept and hardened)
# ----------------------------------------------------------------------------

def _find_placed_digit_in_house(grid81: str, h_type: str, idx1: int, digit: int) -> Optional[int]:
    dch = str(digit)
    for ci in _house_cells(h_type, idx1):
        if 0 <= ci < len(grid81) and grid81[ci] == dch:
            return ci
    return None

def _witness_for_peer_cell(
    *,
    grid81: str,
    peer_ci: int,
    digit: int,
    primary_house_type: str,
) -> Optional[int]:
    r, c = cell_index_to_rc(peer_ci)
    b = box_index_1to9(r, c)

    if primary_house_type == "row":
        return _find_placed_digit_in_house(grid81, "col", c, digit) or _find_placed_digit_in_house(grid81, "box", b, digit)
    if primary_house_type == "col":
        return _find_placed_digit_in_house(grid81, "row", r, digit) or _find_placed_digit_in_house(grid81, "box", b, digit)
    return _find_placed_digit_in_house(grid81, "row", r, digit) or _find_placed_digit_in_house(grid81, "col", c, digit)

def _parse_single_dimension_tokens(dimension: str) -> List[str]:
    dim = (dimension or "").strip().lower()
    if not dim:
        return []
    raw_parts = [p.strip() for p in dim.replace("+", " + ").replace("&", " & ").split()]
    out: List[str] = []
    seen = set()
    for p in raw_parts:
        if p in {"+", "&"}:
            continue
        if p in {"row", "col", "box"} and p not in seen:
            out.append(p)
            seen.add(p)
    return out

def _is_naked_single_technique(technique_id: str) -> bool:
    tid = (technique_id or "").strip().lower()
    return tid in {"singles-naked-2", "singles-naked-3"}

def _primary_house_keys_for_single_dimension(
    *,
    dimension: str,
    target_ci: int,
) -> List[Tuple[str, int]]:
    r, c = cell_index_to_rc(target_ci)
    b = box_index_1to9(r, c)
    idx_by_type = {"row": r, "col": c, "box": b}
    toks = _parse_single_dimension_tokens(dimension) or ["row", "col", "box"]
    return [(t, int(idx_by_type[t])) for t in toks]

def _house_keys_for_naked_single_dimension(
    *,
    dimension: str,
    target_ci: int,
) -> List[Tuple[str, int]]:
    r, c = cell_index_to_rc(target_ci)
    b = box_index_1to9(r, c)
    idx_by_type = {"row": r, "col": c, "box": b}
    toks = _parse_single_dimension_tokens(dimension) or ["row", "col", "box"]
    return [(t, int(idx_by_type[t])) for t in toks]

def _witness_for_digit_in_target_cell_via_dimension_houses(
    *,
    grid81: str,
    target_ci: int,
    digit: int,
    house_keys: List[Tuple[str, int]],
) -> Optional[int]:
    for (h_type, idx1) in house_keys:
        w = _find_placed_digit_in_house(grid81, h_type, idx1, digit)
        if isinstance(w, int) and w != target_ci:
            return w
    return None

def _synthetic_support_for_hidden_single(
    *,
    grid_before: str,
    target_ci: int,
    target_digit: int,
    dimension: str,
) -> Dict[str, Any]:
    house_keys = _primary_house_keys_for_single_dimension(dimension=dimension, target_ci=target_ci)
    h_type, idx1 = house_keys[0]
    house_cells = _house_cells(h_type, idx1)
    empties_in_house = [ci for ci in house_cells if 0 <= ci < len(grid_before) and grid_before[ci] == "."]
    if target_ci not in empties_in_house:
        empties_in_house.append(target_ci)
        empties_in_house = sorted(set(empties_in_house))
    peer_cells = [ci for ci in empties_in_house if ci != target_ci]

    witness_by_peer: Dict[str, Optional[int]] = {}
    witness_cells_set: Set[int] = set()
    for pci in peer_cells:
        w = _witness_for_peer_cell(
            grid81=grid_before,
            peer_ci=pci,
            digit=target_digit,
            primary_house_type=h_type,
        )
        witness_by_peer[str(pci)] = w
        if isinstance(w, int):
            witness_cells_set.add(w)

    return {
        "pattern_cells": [cell_ref_from_index(ci) for ci in sorted(empties_in_house)],
        "focus_cells": [cell_ref_from_index(target_ci)],
        "peer_cells": [cell_ref_from_index(ci) for ci in sorted(peer_cells)],
        "witness_cells": [cell_ref_from_index(ci) for ci in sorted(witness_cells_set)],
        "witness_by_peer": {k: (cell_ref_from_index(v) if isinstance(v, int) else None) for k, v in witness_by_peer.items()},
    }

def _synthetic_support_for_naked_single(
    *,
    grid_before: str,
    options_all_masks: Dict[str, int],
    target_ci: int,
    target_digit: int,
    dimension: str,
) -> Dict[str, Any]:
    house_keys = _house_keys_for_naked_single_dimension(dimension=dimension, target_ci=target_ci)
    cell_mask = _mask_for_cell(options_all_masks, target_ci)
    default_digits = _digits_from_mask(cell_mask)

    eliminated_digits: List[int] = []
    witness_by_digit: Dict[str, Optional[Dict[str, Any]]] = {}
    witness_cells_set: Set[int] = set()

    for d in range(1, 10):
        if d == target_digit:
            continue
        w = _witness_for_digit_in_target_cell_via_dimension_houses(
            grid81=grid_before,
            target_ci=target_ci,
            digit=d,
            house_keys=house_keys,
        )
        if isinstance(w, int):
            eliminated_digits.append(d)
            witness_by_digit[str(d)] = cell_ref_from_index(w)
            witness_cells_set.add(w)

    return {
        "pattern_cells": [cell_ref_from_index(target_ci)],
        "focus_cells": [cell_ref_from_index(target_ci)],
        "peer_cells": [],
        "witness_cells": [cell_ref_from_index(ci) for ci in sorted(witness_cells_set)],
        "proof_kind": "CELL_FOR_DIGIT",
        "focus_cell": cell_ref_from_index(target_ci),
        "dimension_houses": [normalize_house(h_type, idx1) for (h_type, idx1) in house_keys],
        "default_candidate_digits": default_digits,
        "eliminated_digits": eliminated_digits,
        "witness_by_digit": witness_by_digit,
    }

def _synthetic_impacts_for_hidden_single(
    *,
    options_all_masks: Dict[str, int],
    grid_before: str,
    target_ci: int,
    target_digit: int,
    dimension: str,
) -> Dict[str, Any]:
    cell_mask = _mask_for_cell(options_all_masks, target_ci)
    default_digits = _digits_from_mask(cell_mask)
    r, c = cell_index_to_rc(target_ci)
    house_keys = _primary_house_keys_for_single_dimension(dimension=dimension, target_ci=target_ci)

    def default_candidate_cells_for_digit(h_type: str, idx1: int, d: int) -> List[int]:
        bit = 1 << (d - 1)
        cells: List[int] = []
        if h_type == "row":
            rr = idx1
            for cc in range(1, 10):
                ci = rc_to_cell_index(rr, cc)
                if grid_before[ci] == "." and (_mask_for_cell(options_all_masks, ci) & bit):
                    cells.append(ci)
        elif h_type == "col":
            cc = idx1
            for rr in range(1, 10):
                ci = rc_to_cell_index(rr, cc)
                if grid_before[ci] == "." and (_mask_for_cell(options_all_masks, ci) & bit):
                    cells.append(ci)
        else:
            bb = idx1
            br = ((bb - 1) // 3) * 3 + 1
            bc = ((bb - 1) % 3) * 3 + 1
            for dr in range(3):
                for dc in range(3):
                    rr = br + dr
                    cc = bc + dc
                    ci = rc_to_cell_index(rr, cc)
                    if grid_before[ci] == "." and (_mask_for_cell(options_all_masks, ci) & bit):
                        cells.append(ci)
        return cells

    house_impacts: List[Dict[str, Any]] = []
    for (h_type, idx1) in house_keys:
        default_cells = default_candidate_cells_for_digit(h_type, idx1, target_digit)
        if target_ci not in default_cells:
            default_cells = list(sorted(set(default_cells + [target_ci])))
        claimed = [ci for ci in default_cells if ci != target_ci]
        remaining = [target_ci]
        house_impacts.append({
            "house": normalize_house(h_type, idx1),
            "digit": target_digit,
            "impact": "FINAL",
            "default_candidate_cells": default_cells,
            "claimed_candidate_cells": claimed,
            "remaining_candidate_cells": remaining,
            "synthetic": True,
            "synthetic_reason": "hidden_single_eliminates_other_cells",
        })

    cell_impacts = [{
        "cellIndex": target_ci,
        "r": r,
        "c": c,
        "impact": "FINAL",
        "default_candidate_digits": default_digits,
        "claimed_candidate_digits": [],
        "remaining_candidate_digits": [target_digit],
        "synthetic": True,
        "synthetic_reason": "cell_final_by_hidden_single",
    }]
    return {"cell_impacts": cell_impacts, "house_impacts": house_impacts}

def _synthetic_impacts_for_naked_single(
    *,
    options_all_masks: Dict[str, int],
    target_ci: int,
    target_digit: int,
    dimension: str,
) -> Dict[str, Any]:
    cell_mask = _mask_for_cell(options_all_masks, target_ci)
    default_digits = _digits_from_mask(cell_mask)
    claimed = [d for d in default_digits if d != target_digit]
    r, c = cell_index_to_rc(target_ci)
    house_keys = _house_keys_for_naked_single_dimension(dimension=dimension, target_ci=target_ci)

    return {
        "cell_impacts": [{
            "cellIndex": target_ci,
            "r": r,
            "c": c,
            "impact": "FINAL",
            "default_candidate_digits": default_digits,
            "claimed_candidate_digits": claimed,
            "remaining_candidate_digits": [target_digit],
            "synthetic": True,
            "synthetic_reason": "naked_single_eliminates_other_digits_via_dimension_houses",
            "dimension_houses": [normalize_house(h_type, idx1) for (h_type, idx1) in house_keys],
        }],
        "house_impacts": [],
    }



def _single_explanation_links(
    *,
    technique_id: str,
    support: Dict[str, Any],
    target_ci: int,
    target_digit: int,
) -> List[Dict[str, Any]]:
    """
    Canonical explanation links for singles.

    Hidden single:
      kind = "peer_witness"
      peer_cell -> witness_cell for the placed digit

    Naked single:
      kind = "digit_witness"
      focus_cell -> witness_cell for each eliminated digit
    """
    out: List[Dict[str, Any]] = []
    focus_cell = cell_ref_from_index(target_ci)

    if _is_naked_single_technique(technique_id):
        witness_by_digit = support.get("witness_by_digit", {}) or {}
        for k, witness_cell in sorted(witness_by_digit.items(), key=lambda kv: int(kv[0])):
            try:
                eliminated_digit = int(k)
            except Exception:
                continue
            if not isinstance(witness_cell, dict):
                continue
            out.append({
                "kind": "digit_witness",
                "focus_cell": focus_cell,
                "eliminated_digit": eliminated_digit,
                "witness_cell": witness_cell,
                "placed_digit": int(target_digit),
            })
        return out

    witness_by_peer = support.get("witness_by_peer", {}) or {}
    for k, witness_cell in sorted(witness_by_peer.items(), key=lambda kv: int(kv[0])):
        try:
            peer_ci = int(k)
        except Exception:
            continue
        if not isinstance(witness_cell, dict):
            continue
        out.append({
            "kind": "peer_witness",
            "peer_cell": cell_ref_from_index(peer_ci),
            "witness_cell": witness_cell,
            "digit": int(target_digit),
            "focus_cell": focus_cell,
        })
    return out


def _build_single_application(
    *,
    technique_id: str,
    placement: PlacementHit,
    grid_before: str,
    options_all_masks: Dict[str, int],
) -> CanonicalTechniqueApplication:
    family = _detect_family(technique_id)
    real_name = technique_meta(technique_id).get("real_name", technique_id)
    target_ci = int(placement.cellIndex)
    target_digit = int(placement.digit)
    dim = str(placement.dimension or "")
    focus = cell_ref_from_index(target_ci)

    if _is_naked_single_technique(technique_id):
        support = _synthetic_support_for_naked_single(
            grid_before=grid_before,
            options_all_masks=options_all_masks,
            target_ci=target_ci,
            target_digit=target_digit,
            dimension=dim,
        )
        impacts = _synthetic_impacts_for_naked_single(
            options_all_masks=options_all_masks,
            target_ci=target_ci,
            target_digit=target_digit,
            dimension=dim,
        )
        pattern_type = "single_in_cell"
        archetype = "NAKED_SINGLES"
        summary = f"Cell r{placement.r}c{placement.c} is forced to {target_digit}."
        houses = [{
            "type": "cell",
            "cell": focus,
        }]
        units_scanned = support.get("dimension_houses", [])
        trigger_facts = [f"Technique: {real_name}."]
        confrontation_facts = [
            "Other candidates in the cell are ruled out by house constraints."
        ]
        resolution_facts = [summary]
    else:
        support = _synthetic_support_for_hidden_single(
            grid_before=grid_before,
            target_ci=target_ci,
            target_digit=target_digit,
            dimension=dim,
        )
        impacts = _synthetic_impacts_for_hidden_single(
            options_all_masks=options_all_masks,
            grid_before=grid_before,
            target_ci=target_ci,
            target_digit=target_digit,
            dimension=dim,
        )

        norm_dim = dim.lower().replace(" ", "_").replace("&", "").replace("__", "_")
        pattern_type = f"single_in_{norm_dim}" if norm_dim else "single_in_house"

        if technique_id == "singles-1":
            archetype = "FULL_HOUSE"
            summary = f"Only one cell remains open in the house, so r{placement.r}c{placement.c} must be {target_digit}."
        else:
            archetype = "HIDDEN_SINGLES"
            summary = f"Digit {target_digit} is forced at r{placement.r}c{placement.c}."

        primary_house_keys = _primary_house_keys_for_single_dimension(
            dimension=dim,
            target_ci=target_ci,
        )
        units_scanned = [normalize_house(h_type, idx1) for (h_type, idx1) in primary_house_keys]
        houses = units_scanned[:] if units_scanned else houses_for_cell(target_ci)

        trigger_facts = [f"Technique: {real_name}."]
        confrontation_facts = [
            "Other candidates or other cells are ruled out by house constraints."
        ]
        resolution_facts = [summary]

        if technique_id == "singles-1":
            trigger_facts = [
                "Technique: Full House.",
                "The house is nearly complete and only one seat is still open.",
            ]
            confrontation_facts = [
                "Full House now steps in to claim the last open seat in the house.",
            ]
            resolution_facts = [
                summary,
                "The lesson is to scan houses that are almost complete: the final seat often gives itself away.",
            ]

    explanation_links = _single_explanation_links(
        technique_id=technique_id,
        support=support,
        target_ci=target_ci,
        target_digit=target_digit,
    )

    return CanonicalTechniqueApplication(
        application_id="app:" + sha12(f"single|{technique_id}|{target_ci}|{target_digit}|{dim}"),
        technique_id=technique_id,
        technique_family=family,
        technique_real_name=real_name,
        application_kind="placement",
        semantic_completeness="full",
        focus_cells=support.get("focus_cells", [focus]),
        pattern_cells=support.get("pattern_cells", [focus]),
        peer_cells=support.get("peer_cells", []),
        witness_cells=support.get("witness_cells", []),
        target_cells=[focus],
        houses=houses,
        digits=[target_digit],
        placements=[CanonicalEffectPlacement(cell=focus, digit=target_digit, source="single_resolution")],
        cell_value_forces=[CanonicalEffectPlacement(cell=focus, digit=target_digit, source="single_resolution")],
        pattern_type=pattern_type,
        pattern_subtype=dim or None,
        anchors=[focus],
        units_scanned=units_scanned,
        cover_sets=[],
        constraint_explanation=[
            f"Single-technique resolution using dimension '{dim}'." if dim else "Single-technique resolution."
        ],
        roles={
            "focus": [focus],
            "peer": support.get("peer_cells", []),
            "witness": support.get("witness_cells", []),
        },
        explanation_links=explanation_links,
        narrative_archetype=archetype,
        trigger_facts=trigger_facts,
        confrontation_facts=confrontation_facts,
        resolution_facts=resolution_facts,
        narrative_role="trigger",
        summary_fact=summary,
        engine_debug_summary={
            "synthetic_single": True,
            "placement_shape": placement.source_shape,
            "support_shape": debug_shape_summary(support),
            "impacts_shape": debug_shape_summary(impacts),
            "explanation_links_count": len(explanation_links),
            "single_kind": "naked" if _is_naked_single_technique(technique_id) else "hidden",
        },
        engine_debug_sanitized={
            "placement": to_json_safe(asdict(placement)),
            "support": to_json_safe(support),
            "impacts": to_json_safe(impacts),
            "explanation_links": to_json_safe(explanation_links),
        },
    )



# ----------------------------------------------------------------------------
# Family normalizers
# ----------------------------------------------------------------------------

def _normalize_subset_application(raw_app: RawApplication) -> CanonicalTechniqueApplication:
    tid = raw_app.technique_id
    family = _detect_family(tid)
    real_name = technique_meta(tid).get("real_name", tid)
    elims = _elim_entries_from_removed_chars(raw_app.removed_chars_updated)
    elim_cells = _cell_set_from_eliminations(elims)
    digits = _digits_from_eliminations(elims)
    common_house = _best_common_house(elim_cells)
    subset_cells: List[Dict[str, Any]] = []
    other_cells: List[Dict[str, Any]] = []

    # Try to infer subset cells from raw app_details in a generic way:
    # collect cell-like tuples from sanitized raw if possible; keep only first small cluster
    def collect_cell_like(x: Any, acc: List[Dict[str, Any]]) -> None:
        if _looks_like_rc0_tuple(x):
            acc.append(cell_ref_from_rc0(int(x[0]), int(x[1])))
            return
        if isinstance(x, dict):
            for k, v in x.items():
                collect_cell_like(k, acc)
                collect_cell_like(v, acc)
        elif isinstance(x, (list, tuple, set)):
            for it in x:
                collect_cell_like(it, acc)

    tmp_cells: List[Dict[str, Any]] = []
    collect_cell_like(raw_app.application_details, tmp_cells)
    seen: Dict[int, Dict[str, Any]] = {}
    for c in tmp_cells:
        seen[c["cellIndex"]] = c
    dedup_cells = [seen[k] for k in sorted(seen.keys())]

    # Heuristic subset members: first 2/3/4 unique cells depending on technique
    subset_size = 2 if tid in {"doubles-naked", "doubles", "boxed-doubles"} else \
                  3 if tid in {"triplets-naked", "triplets", "boxed-triplets"} else \
                  4 if tid in {"quads-naked", "quads", "boxed-quads"} else 0

    if subset_size > 0:
        subset_cells = dedup_cells[:subset_size]

    if common_house:
        house_cells = [cell_ref_from_index(ci) for ci in _house_cells(common_house["type"], int(common_house["index1to9"]))]
        subset_ci = {c["cellIndex"] for c in subset_cells}
        other_cells = [c for c in house_cells if c["cellIndex"] not in subset_ci]

    if family == "multiple_naked":
        if tid == "doubles-naked":
            subtype = "naked_pair"
        elif tid == "triplets-naked":
            subtype = "naked_triplet"
        else:
            subtype = "naked_quad"
        pattern_type = "subset"
        summary = f"{real_name} constrains digits {digits or []} within a small cell set."
        kind = "elimination" if elims else "mixed"
        roles = {
                    "subset_member": subset_cells,
                    "target": elim_cells,
                    "elimination_target": elim_cells,
                }
    else:
        if tid in {"doubles", "boxed-doubles"}:
            subtype = "hidden_pair"
        elif tid in {"triplets", "boxed-triplets"}:
            subtype = "hidden_triplet"
        else:
            subtype = "hidden_quad"
        pattern_type = "hidden_subset"
        summary = f"{real_name} restricts a small support cell set."
        kind = "restriction" if not elims else "mixed"
        roles = {
                    "supporting_cell": subset_cells,
                    "target": subset_cells,
                    "restriction_target": subset_cells,
                    "elimination_target": elim_cells,
                }

    if tid.startswith("boxed-"):
        if common_house:
            box_context = common_house if common_house.get("type") == "box" else None
        else:
            box_context = None
    else:
        box_context = None

    app_id = "app:" + sha12(f"{tid}|{raw_app.name_application}|{repr(raw_app.application_details)}")
    explanation: List[str] = []
    if common_house:
        explanation.append(
            f"Pattern operates in {common_house['type']} {common_house.get('index1to9')}."
        )
    if subset_cells:
        explanation.append(f"Subset/support cells count = {len(subset_cells)}.")
    if digits:
        explanation.append(f"Locked/support digits = {digits}.")
    if family == "multiple_naked" and elim_cells:
        explanation.append("Those locked digits can be swept from other cells in the same house.")
    elif family in {"multiple_hidden", "multiple_hidden_boxed"} and subset_cells:
        explanation.append("Those hidden digits tighten which values can remain in the support cells.")

    completeness = "full" if common_house is not None else "partial"

    app = CanonicalTechniqueApplication(
        application_id=app_id,
        technique_id=tid,
        technique_family=family,
        technique_real_name=real_name,
        application_kind=kind,
        semantic_completeness=completeness,
        focus_cells=subset_cells or elim_cells,
        pattern_cells=subset_cells,
        peer_cells=other_cells,
        target_cells=elim_cells if family == "multiple_naked" else subset_cells,
        witness_cells=[],
        houses=[common_house] if common_house else [],
        digits=digits,
        candidate_eliminations=elims,
        pattern_type=pattern_type,
        pattern_subtype=subtype,
        anchors=subset_cells[:],
        units_scanned=[common_house] if common_house else [],
        cover_sets=[{
                    "subset_kind": subtype,
                    "house": common_house,
                    "subset_digits": digits,
                    "subset_cells": subset_cells,
                    "other_cells_in_house": other_cells,
                    "sweep_cells": elim_cells if family == "multiple_naked" else subset_cells,
                    "target_relation": {
                        "kind": "cleanup_target_cell" if family == "multiple_naked" else "cleanup_target_house",
                        "target_cells": elim_cells if family == "multiple_naked" else subset_cells,
                        "locked_digits": digits,
                    },
                    "sweep_relation": {
                        "house": common_house,
                        "sweep_cells": elim_cells if family == "multiple_naked" else subset_cells,
                        "locked_digits": digits,
                    },
                    "box_context": box_context,
                }],
        constraint_explanation=explanation,
        roles=roles,
        narrative_archetype="SUBSETS",
        trigger_facts=[
                    f"Technique: {real_name}.",
                    "A small set of digits is confined to a small set of cells in one house.",
                ],
                confrontation_facts=[
                    "First identify the subset pattern and its locked digits.",
                    "Then use that confinement to remove or restrict candidates in related cells.",
                ],
                resolution_facts=[summary],
        narrative_role="setup",
        summary_fact=summary,
        engine_debug_summary=raw_app.engine_debug_summary,
        engine_debug_sanitized=raw_app.engine_debug_sanitized,
    )

    # Hidden subsets prefer restrictions; if we observed eliminations in those same support cells,
    # convert a light restriction view.
    if family in {"multiple_hidden", "multiple_hidden_boxed"} and subset_cells and digits:
        restricts: List[CanonicalEffectRestriction] = []
        subset_ci = {c["cellIndex"] for c in subset_cells}
        by_cell_removed: Dict[int, List[int]] = {}
        for e in elims:
            if e.cell["cellIndex"] in subset_ci:
                by_cell_removed.setdefault(e.cell["cellIndex"], []).append(e.digit)
        for c in subset_cells:
            removed = sorted(by_cell_removed.get(c["cellIndex"], []))
            restricts.append(CanonicalEffectRestriction(
                cell=c,
                allowed_digits=digits,
                removed_digits=removed,
                source="hidden_subset_restriction",
            ))
        app.candidate_restrictions = restricts

    return app

def _dedup_cells_by_index(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Dict[int, Dict[str, Any]] = {}
    for c in cells:
        try:
            ci = int(c["cellIndex"])
        except Exception:
            continue
        seen[ci] = c
    return [seen[k] for k in sorted(seen.keys())]


def _append_house_once(out: List[Dict[str, Any]], house: Optional[Dict[str, Any]]) -> None:
    if house is None:
        return
    key = (str(house.get("type")), int(house.get("index1to9", 0)))
    existing = {(str(h.get("type")), int(h.get("index1to9", 0))) for h in out}
    if key not in existing:
        out.append(house)


def _orientation_from_line_type(line_type: Optional[str]) -> Optional[str]:
    if line_type == "row":
        return "horizontal"
    if line_type == "col":
        return "vertical"
    return None


def _common_house_of_type(
    cells: List[Dict[str, Any]],
    allowed_types: Set[str],
) -> Optional[Dict[str, Any]]:
    if not cells:
        return None
    cis = [int(c["cellIndex"]) for c in cells]
    house_counts: Dict[Tuple[str, int], int] = {}
    for ci in cis:
        for h in houses_for_cell(ci):
            h_type = str(h["type"])
            if h_type not in allowed_types:
                continue
            key = (h_type, int(h["index1to9"]))
            house_counts[key] = house_counts.get(key, 0) + 1
    full = [(k, v) for k, v in house_counts.items() if v == len(cis)]
    if not full:
        return None
    order = {"row": 0, "col": 1, "box": 2}
    full.sort(key=lambda kv: (order.get(kv[0][0], 99), kv[0][1]))
    (h_type, idx1), _ = full[0]
    return normalize_house(h_type, idx1)


def _collect_rc0_cells_recursive(value: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    def walk(x: Any) -> None:
        if _looks_like_rc0_tuple(x):
            out.append(cell_ref_from_rc0(int(x[0]), int(x[1])))
            return
        if isinstance(x, dict):
            for k, v in x.items():
                walk(k)
                walk(v)
            return
        if isinstance(x, (list, tuple, set)):
            for it in x:
                walk(it)

    walk(value)
    return _dedup_cells_by_index(out)


def _parse_box_line_structure(
    raw_app: RawApplication,
    interaction_kind: str,
    elim_cells: List[Dict[str, Any]],
    digit_from_elims: Optional[int],
    grid_before: str,
) -> Dict[str, Any]:
    details = raw_app.application_details

    box_house: Optional[Dict[str, Any]] = None
    line_house: Optional[Dict[str, Any]] = None
    line_type: Optional[str] = None
    locked_cells: List[Dict[str, Any]] = []

    # ------------------------------------------------------------
    # Preferred path: parse the actual engine tuple shapes
    # ------------------------------------------------------------
    if interaction_kind == "pointing":
        # generator/techniques/singles_pointing.py
        # (char, idx_box, idxs_possible, "hor"/"ver", idx_row_or_col)
        if isinstance(details, (list, tuple)) and len(details) >= 5:
            char, idx_box, idxs_possible, ray_kind, idx_line = details[:5]
            d = char_to_digit(char)
            if digit_from_elims is None and d is not None:
                digit_from_elims = d

            try:
                box_house = normalize_house("box", int(idx_box) + 1)
            except Exception:
                box_house = None

            rk = str(ray_kind).strip().lower()
            if rk == "hor":
                line_type = "row"
            elif rk == "ver":
                line_type = "col"

            if line_type is not None:
                try:
                    line_house = normalize_house(line_type, int(idx_line) + 1)
                except Exception:
                    line_house = None

            if isinstance(idxs_possible, (list, tuple, set)):
                locked_cells = _dedup_cells_by_index([
                    cell_ref_from_rc0(int(rc[0]), int(rc[1]))
                    for rc in idxs_possible
                    if _looks_like_rc0_tuple(rc)
                ])

    elif interaction_kind == "claiming":
        # generator/techniques/singles_boxed.py
        # ("row"/"col", char, idx_line, idx_box, idxs_possible)
        if isinstance(details, (list, tuple)) and len(details) >= 5:
            raw_line_type, char, idx_line, idx_box, idxs_possible = details[:5]
            d = char_to_digit(char)
            if digit_from_elims is None and d is not None:
                digit_from_elims = d

            lt = str(raw_line_type).strip().lower()
            if lt in {"row", "col"}:
                line_type = lt

            if line_type is not None:
                try:
                    line_house = normalize_house(line_type, int(idx_line) + 1)
                except Exception:
                    line_house = None

            try:
                box_house = normalize_house("box", int(idx_box) + 1)
            except Exception:
                box_house = None

            if isinstance(idxs_possible, (list, tuple, set)):
                locked_cells = _dedup_cells_by_index([
                    cell_ref_from_rc0(int(rc[0]), int(rc[1]))
                    for rc in idxs_possible
                    if _looks_like_rc0_tuple(rc)
                ])

    # ------------------------------------------------------------
    # Fallback path: infer from generic cell scrape
    # ------------------------------------------------------------
    if not locked_cells:
        inferred_cells = _collect_rc0_cells_recursive(details)
        elim_ci = {int(c["cellIndex"]) for c in elim_cells}
        locked_cells = [c for c in inferred_cells if int(c["cellIndex"]) not in elim_ci]
        locked_cells = _dedup_cells_by_index(locked_cells)

    if box_house is None:
        box_house = _common_house_of_type(locked_cells, {"box"})
    if line_house is None:
        line_house = _common_house_of_type(locked_cells, {"row", "col"})
    if line_type is None and line_house is not None:
        lh_type = str(line_house.get("type"))
        if lh_type in {"row", "col"}:
            line_type = lh_type

    source_house = box_house if interaction_kind == "pointing" else line_house
    cross_house = line_house if interaction_kind == "pointing" else box_house
    orientation = _orientation_from_line_type(line_type)

    overlap_cells = _dedup_cells_by_index(locked_cells)



    overlap_ci = {
        int(c["cellIndex"])
        for c in overlap_cells
        if isinstance(c, dict) and c.get("cellIndex") is not None
    }

    def _is_open_ci(ci: int) -> bool:
        if not (0 <= int(ci) < len(grid_before)):
            return False
        ch = str(grid_before[int(ci)])
        return ch not in DIGITS

    def _candidate_digits_before(ci: int) -> List[int]:
        return _candidate_digits_before_for_cell(grid_before, int(ci))

    source_outside_overlap_all_cells: List[Dict[str, Any]] = []
    source_outside_overlap_open_cells: List[Dict[str, Any]] = []
    source_outside_overlap_open_candidate_cells: List[Dict[str, Any]] = []
    source_outside_overlap_open_noncandidate_cells: List[Dict[str, Any]] = []

    if isinstance(source_house, dict) and digit_from_elims in range(1, 10):
        for ci in _house_cells(str(source_house.get("type")), int(source_house.get("index1to9"))):
            if int(ci) in overlap_ci:
                continue
            cell = cell_ref_from_index(int(ci))
            source_outside_overlap_all_cells.append(cell)

            if not _is_open_ci(int(ci)):
                continue

            source_outside_overlap_open_cells.append(cell)
            if int(digit_from_elims) in _candidate_digits_before(int(ci)):
                source_outside_overlap_open_candidate_cells.append(cell)
            else:
                source_outside_overlap_open_noncandidate_cells.append(cell)

    cross_outside_overlap_all_cells: List[Dict[str, Any]] = []
    cross_outside_overlap_open_cells: List[Dict[str, Any]] = []
    cross_outside_overlap_open_candidate_cells: List[Dict[str, Any]] = []
    cross_outside_overlap_open_noncandidate_cells: List[Dict[str, Any]] = []

    if isinstance(cross_house, dict) and digit_from_elims in range(1, 10):
        for ci in _house_cells(str(cross_house.get("type")), int(cross_house.get("index1to9"))):
            if int(ci) in overlap_ci:
                continue
            cell = cell_ref_from_index(int(ci))
            cross_outside_overlap_all_cells.append(cell)

            if not _is_open_ci(int(ci)):
                continue

            cross_outside_overlap_open_cells.append(cell)
            if int(digit_from_elims) in _candidate_digits_before(int(ci)):
                cross_outside_overlap_open_candidate_cells.append(cell)
            else:
                cross_outside_overlap_open_noncandidate_cells.append(cell)

    elimination_cells = _dedup_cells_by_index(elim_cells)
    elimination_ci = {int(c["cellIndex"]) for c in elimination_cells if c.get("cellIndex") is not None}
    forbidden_cross_cells = [
        c for c in cross_outside_overlap_open_cells
        if int(c["cellIndex"]) in elimination_ci
    ]




    cardinality = len(overlap_cells)
    pattern_subtype = f"{interaction_kind}_{'pair' if cardinality == 2 else 'triple' if cardinality == 3 else 'group'}"

    return {
        "digit": digit_from_elims,
        "box_house": box_house,
        "line_house": line_house,
        "line_type": line_type,
        "orientation": orientation,
        "direction_mode": interaction_kind,
        "source_house": source_house,
        "cross_house": cross_house,
        "target_house": cross_house,  # backward-compatible alias
        "overlap_cells": overlap_cells,
        "locked_cells": overlap_cells,  # backward-compatible alias
        "pattern_cells": overlap_cells,

        # Setup-facing truth: the legacy source_outside_overlap_cells field is now
        # narrowed to OUTSIDE OPEN SEATS only, because setup narration must reason
        # about open seats, not filled cells.
        "source_outside_overlap_cells": _dedup_cells_by_index(source_outside_overlap_open_cells),
        "source_outside_overlap_all_cells": _dedup_cells_by_index(source_outside_overlap_all_cells),
        "source_outside_overlap_open_cells": _dedup_cells_by_index(source_outside_overlap_open_cells),
        "source_outside_overlap_open_candidate_cells": _dedup_cells_by_index(source_outside_overlap_open_candidate_cells),
        "source_outside_overlap_open_noncandidate_cells": _dedup_cells_by_index(source_outside_overlap_open_noncandidate_cells),

        "cross_outside_overlap_cells": _dedup_cells_by_index(cross_outside_overlap_open_cells),
        "cross_outside_overlap_all_cells": _dedup_cells_by_index(cross_outside_overlap_all_cells),
        "cross_outside_overlap_open_cells": _dedup_cells_by_index(cross_outside_overlap_open_cells),
        "cross_outside_overlap_open_candidate_cells": _dedup_cells_by_index(cross_outside_overlap_open_candidate_cells),
        "cross_outside_overlap_open_noncandidate_cells": _dedup_cells_by_index(cross_outside_overlap_open_noncandidate_cells),

        "forbidden_cross_cells": _dedup_cells_by_index(forbidden_cross_cells),


        "sweep_cells": elimination_cells,
        "elimination_cells": elimination_cells,
        "cardinality": cardinality,
        "pattern_subtype": pattern_subtype,
    }


def _normalize_box_line_application(raw_app: RawApplication) -> CanonicalTechniqueApplication:
    tid = raw_app.technique_id
    real_name = technique_meta(tid).get("real_name", tid)
    elims = _elim_entries_from_removed_chars(raw_app.removed_chars_updated)
    elim_cells = _cell_set_from_eliminations(elims)
    elim_digits = _digits_from_eliminations(elims)

    interaction_kind = "pointing" if tid == "singles-pointing" else "claiming"
    digit0 = elim_digits[0] if len(elim_digits) == 1 else None

    parsed = _parse_box_line_structure(
        raw_app=raw_app,
        interaction_kind=interaction_kind,
        elim_cells=elim_cells,
        digit_from_elims=digit0,
        grid_before=raw_app.engine_debug_summary.get("grid81_before", ""),
    )



    digit = parsed["digit"]
    digits = [digit] if digit is not None else elim_digits[:]

    box_house = parsed["box_house"]
    line_house = parsed["line_house"]
    line_type = parsed["line_type"]
    orientation = parsed["orientation"]
    direction_mode = parsed["direction_mode"]
    source_house = parsed["source_house"]
    cross_house = parsed["cross_house"]
    target_house = parsed["target_house"]  # backward-compatible alias
    overlap_cells = parsed["overlap_cells"]
    pattern_cells = parsed["pattern_cells"]
    locked_cells = parsed["locked_cells"]  # backward-compatible alias
    source_outside_overlap_cells = parsed["source_outside_overlap_cells"]
    cross_outside_overlap_cells = parsed["cross_outside_overlap_cells"]
    forbidden_cross_cells = parsed["forbidden_cross_cells"]
    sweep_cells = parsed["sweep_cells"]
    elimination_cells = parsed["elimination_cells"]
    cardinality = parsed["cardinality"]
    pattern_subtype = parsed["pattern_subtype"]

    houses: List[Dict[str, Any]] = []
    _append_house_once(houses, box_house)
    _append_house_once(houses, line_house)
    _append_house_once(houses, source_house)
    _append_house_once(houses, cross_house)

    locked_count = len(overlap_cells)
    completeness = "full" if (
        digit is not None
        and box_house is not None
        and line_house is not None
        and source_house is not None
        and cross_house is not None
        and locked_count > 0
    ) else "partial"

    if digit is not None and source_house is not None and cross_house is not None and overlap_cells:
        summary = (
            f"{real_name} traps digit {digit} in the overlap between "
            f"{source_house.get('type')} {source_house.get('index1to9')} and "
            f"{cross_house.get('type')} {cross_house.get('index1to9')}, "
            f"so the crossing house must give it up elsewhere."
        )
    else:
        summary = f"{real_name} applies territorial control through a box-line overlap."

    explanation: List[str] = [f"Interaction kind: {interaction_kind}."]
    if digit is not None:
        explanation.append(f"Forced digit = {digit}.")
    if source_house is not None:
        explanation.append(
            f"Source house = {source_house.get('type')} {source_house.get('index1to9')}."
        )
    if cross_house is not None:
        explanation.append(
            f"Cross house = {cross_house.get('type')} {cross_house.get('index1to9')}."
        )
    if overlap_cells:
        explanation.append(f"Overlap carrier cells count = {len(overlap_cells)}.")
    if source_outside_overlap_cells:
        explanation.append(
            f"Source outside-overlap cells count = {len(source_outside_overlap_cells)}."
        )
    if forbidden_cross_cells:
        explanation.append(
            f"Cross-house forbidden cells count = {len(forbidden_cross_cells)}."
        )

    trigger_facts: List[str] = []
    if digit is not None and source_house is not None and cross_house is not None:
        trigger_facts = [
            f"Technique: {real_name}.",
            f"Digit {digit} is forced into the overlap between "
            f"{source_house.get('type')} {source_house.get('index1to9')} and "
            f"{cross_house.get('type')} {cross_house.get('index1to9')}.",
            "The source house has run out of room for that digit outside the overlap.",
        ]
    else:
        trigger_facts = [f"Technique: {real_name}."]

    confrontation_facts: List[str] = [
        "Once the digit is confined to the overlap, the crossing house loses permission to keep that digit elsewhere.",
        "The decisive exclusion should be narrated as a territorial permission change, not as an isolated removal.",
    ]

    resolution_facts: List[str] = [
        summary,
        "The intersection pattern contributes indirectly by redrawing the legal map around the overlap.",
    ]

    app_id = "app:" + sha12(f"{tid}|{raw_app.name_application}|{repr(raw_app.application_details)}")
    return CanonicalTechniqueApplication(
        application_id=app_id,
        technique_id=tid,
        technique_family="box_line_interaction",
        technique_real_name=real_name,
        application_kind="elimination" if elims else "mixed",
        semantic_completeness=completeness,

        focus_cells=overlap_cells or sweep_cells,
        pattern_cells=pattern_cells,
        peer_cells=sweep_cells,
        target_cells=elimination_cells,
        witness_cells=overlap_cells,
        houses=houses,
        digits=digits,

        candidate_eliminations=elims,

        pattern_type="intersection",
        pattern_subtype=pattern_subtype,
        anchors=overlap_cells,
        units_scanned=houses,
        cover_sets=[{
            "interaction_kind": interaction_kind,
            "direction_mode": direction_mode,
            "digit": digit,
            "cardinality": cardinality,
            "pattern_subtype": pattern_subtype,
            "box_house": box_house,
            "line_house": line_house,
            "source_house": source_house,
            "cross_house": cross_house,
            "target_house": target_house,  # backward-compatible alias
            "line_type": line_type,
            "orientation": orientation,
            "overlap_cells": overlap_cells,
            "locked_cells": overlap_cells,      # backward-compatible alias
            "pattern_cells": pattern_cells,
            "constrained_cells": overlap_cells, # backward-compatible alias for current Kotlin reader
            "source_outside_overlap_cells": source_outside_overlap_cells,
            "cross_outside_overlap_cells": cross_outside_overlap_cells,
            "forbidden_cross_cells": forbidden_cross_cells,
            "sweep_cells": sweep_cells,
            "elimination_cells": elimination_cells,
            "locked_count": locked_count,
        }],
        constraint_explanation=explanation,
        roles={
            "overlap_cell": overlap_cells,
            "locked_cell": overlap_cells,  # backward-compatible alias
            "pattern_cell": pattern_cells,
            "source_outside_cell": source_outside_overlap_cells,
            "cross_outside_cell": cross_outside_overlap_cells,
            "forbidden_cross_cell": forbidden_cross_cells,
            "sweep_target": sweep_cells,
            "source_house": [source_house] if source_house else [],
            "cross_house": [cross_house] if cross_house else [],
            "target_house": [target_house] if target_house else [],
            # backward-compatible aliases
            "supporting_cell": overlap_cells,
            "elimination_target": sweep_cells,
        },

        explanation_links=[],
        narrative_archetype="INTERSECTIONS",
        trigger_facts=trigger_facts,
        confrontation_facts=confrontation_facts,
        resolution_facts=resolution_facts,
        narrative_role="setup",
        summary_fact=summary,
        engine_debug_summary=raw_app.engine_debug_summary,
        engine_debug_sanitized=raw_app.engine_debug_sanitized,
    )



def _normalize_fish_application(raw_app: RawApplication) -> CanonicalTechniqueApplication:
    tid = raw_app.technique_id
    real_name = technique_meta(tid).get("real_name", tid)
    elims = _elim_entries_from_removed_chars(raw_app.removed_chars_updated)
    elim_cells = _cell_set_from_eliminations(elims)
    digits = _digits_from_eliminations(elims)
    fish_kind = "x_wing" if tid == "x-wings" else "swordfish" if tid == "x-wings-3" else "jellyfish"
    digit = digits[0] if len(digits) == 1 else None

    # Collect candidate structural cells from raw details
    cell_like: List[Dict[str, Any]] = []

    def collect_cell_like(x: Any) -> None:
        if _looks_like_rc0_tuple(x):
            cell_like.append(cell_ref_from_rc0(int(x[0]), int(x[1])))
            return
        if isinstance(x, dict):
            for k, v in x.items():
                collect_cell_like(k)
                collect_cell_like(v)
        elif isinstance(x, (list, tuple, set)):
            for it in x:
                collect_cell_like(it)

    collect_cell_like(raw_app.application_details)

    dedup: Dict[int, Dict[str, Any]] = {}
    for c in cell_like:
        dedup[c["cellIndex"]] = c
    fish_cells = [dedup[k] for k in sorted(dedup.keys())]

    if not fish_cells:
        fish_cells = []

    rows = sorted({c["r"] for c in fish_cells})
    cols = sorted({c["c"] for c in fish_cells})

    orientation: Optional[str]
    base_houses: List[Dict[str, Any]]
    cover_houses: List[Dict[str, Any]]

    if rows and cols:
        if len(rows) <= len(cols):
            orientation = "row_based"
            base_houses = [normalize_house("row", r) for r in rows]
            cover_houses = [normalize_house("col", c) for c in cols]
        else:
            orientation = "col_based"
            base_houses = [normalize_house("col", c) for c in cols]
            cover_houses = [normalize_house("row", r) for r in rows]
    else:
        orientation = None
        base_houses = []
        cover_houses = []

    summary = f"{real_name} forms a fish pattern eliminating digit {digits}."

    app_id = "app:" + sha12(f"{tid}|{raw_app.name_application}|{repr(raw_app.application_details)}")
    return CanonicalTechniqueApplication(
        application_id=app_id,
        technique_id=tid,
        technique_family="fish",
        technique_real_name=real_name,
        application_kind="elimination" if elims else "mixed",
        semantic_completeness="full" if fish_cells and base_houses and cover_houses else "partial",
        focus_cells=fish_cells or elim_cells,
        pattern_cells=fish_cells,
        peer_cells=elim_cells,
        target_cells=elim_cells,
        witness_cells=[],
        houses=base_houses + [h for h in cover_houses if h not in base_houses],
        digits=digits,
        candidate_eliminations=elims,
        pattern_type="fish",
        pattern_subtype=fish_kind,
        anchors=fish_cells,
        units_scanned=base_houses + cover_houses,
        cover_sets=[{
            "fish_kind": fish_kind,
            "digit": digit,
            "orientation": orientation,
            "base_houses": base_houses,
            "cover_houses": cover_houses,
            "fish_cells": fish_cells,
            "elimination_cells": elim_cells,
        }],
        constraint_explanation=[
            "Fish pattern semantics preserved.",
            "Base houses restrict the digit to matching cover houses.",
        ],
        roles={
            "cover_cell": fish_cells,
            "elimination_target": elim_cells,
        },
        explanation_links=[],
        narrative_archetype="FISH",
        trigger_facts=[f"Technique: {real_name}."],
        confrontation_facts=["A digit forms a multi-house cover pattern."],
        resolution_facts=[summary],
        summary_fact=summary,
        engine_debug_summary=raw_app.engine_debug_summary,
        engine_debug_sanitized=raw_app.engine_debug_sanitized,
    )

def _normalize_wing_application(raw_app: RawApplication) -> CanonicalTechniqueApplication:
    tid = raw_app.technique_id
    real_name = technique_meta(tid).get("real_name", tid)
    elims = _elim_entries_from_removed_chars(raw_app.removed_chars_updated)
    elim_cells = _cell_set_from_eliminations(elims)
    digits = _digits_from_eliminations(elims)
    wing_kind = "xy_wing" if tid == "y-wings" else "boxed_wing"
    summary = f"{real_name} creates a wing-based elimination."

    # Heuristic pivot/wing extraction from raw details
    cell_like: List[Dict[str, Any]] = []
    def collect_cell_like(x: Any) -> None:
        if _looks_like_rc0_tuple(x):
            cell_like.append(cell_ref_from_rc0(int(x[0]), int(x[1])))
            return
        if isinstance(x, dict):
            for k, v in x.items():
                collect_cell_like(k)
                collect_cell_like(v)
        elif isinstance(x, (list, tuple, set)):
            for it in x:
                collect_cell_like(it)
    collect_cell_like(raw_app.application_details)
    dedup = {}
    for c in cell_like:
        dedup[c["cellIndex"]] = c
    cells = [dedup[k] for k in sorted(dedup.keys())]
    pivot = cells[:1]
    wings = cells[1:3]

    app_id = "app:" + sha12(f"{tid}|{raw_app.name_application}|{repr(raw_app.application_details)}")
    return CanonicalTechniqueApplication(
        application_id=app_id,
        technique_id=tid,
        technique_family=_detect_family(tid),
        technique_real_name=real_name,
        application_kind="elimination" if elims else "mixed",
        semantic_completeness="partial",
        focus_cells=pivot or wings or elim_cells,
        pattern_cells=cells,
        target_cells=elim_cells,
        digits=digits,
        candidate_eliminations=elims,
        pattern_type="wing",
        pattern_subtype=wing_kind,
        anchors=pivot,
        cover_sets=[{
            "wing_kind": wing_kind,
            "pivot_cell": pivot[0] if pivot else None,
            "wing_cells": wings,
            "digits_by_cell": [],
            "elimination_digit": digits[0] if len(digits) == 1 else None,
            "elimination_targets": elim_cells,
        }],
        roles={
            "pivot": pivot,
            "wing": wings,
            "elimination_target": elim_cells,
        },
        constraint_explanation=["Wing relation preserved semantically; exact arm internals remain in debug."],
        narrative_archetype="WINGS",
        trigger_facts=[f"Technique: {real_name}."],
        confrontation_facts=["A pivot-and-wings relation forces a shared consequence."],
        resolution_facts=[summary],
        summary_fact=summary,
        engine_debug_summary=raw_app.engine_debug_summary,
        engine_debug_sanitized=raw_app.engine_debug_sanitized,
    )

def _normalize_chain_application(raw_app: RawApplication) -> CanonicalTechniqueApplication:
    tid = raw_app.technique_id
    real_name = technique_meta(tid).get("real_name", tid)
    elims = _elim_entries_from_removed_chars(raw_app.removed_chars_updated)
    elim_cells = _cell_set_from_eliminations(elims)
    digits = _digits_from_eliminations(elims)
    chain_kind = "remote_pairs" if tid == "remote-pairs" else "ab_chain"
    summary = f"{real_name} uses chain logic to eliminate candidates."

    cell_like: List[Dict[str, Any]] = []
    def collect_cell_like(x: Any) -> None:
        if _looks_like_rc0_tuple(x):
            cell_like.append(cell_ref_from_rc0(int(x[0]), int(x[1])))
            return
        if isinstance(x, dict):
            for k, v in x.items():
                collect_cell_like(k)
                collect_cell_like(v)
        elif isinstance(x, (list, tuple, set)):
            for it in x:
                collect_cell_like(it)
    collect_cell_like(raw_app.application_details)
    dedup = {}
    for c in cell_like:
        dedup[c["cellIndex"]] = c
    chain_cells = [dedup[k] for k in sorted(dedup.keys())]

    chain_nodes: List[Dict[str, Any]] = []
    for i, c in enumerate(chain_cells):
        role = "endpoint" if i in {0, len(chain_cells)-1} and len(chain_cells) >= 2 else "chain_node"
        chain_nodes.append({
            "index": i,
            "cell": c,
            "candidate_digits": [],
            "role": role,
        })

    chain_links: List[Dict[str, Any]] = []
    for i in range(max(0, len(chain_nodes)-1)):
        chain_links.append({
            "from_node": i,
            "to_node": i + 1,
            "digit": None,
            "link_type": None,
        })

    endpoint_cells = [n["cell"] for n in chain_nodes if n["role"] == "endpoint"]

    app_id = "app:" + sha12(f"{tid}|{raw_app.name_application}|{repr(raw_app.application_details)}")
    return CanonicalTechniqueApplication(
        application_id=app_id,
        technique_id=tid,
        technique_family="chain",
        technique_real_name=real_name,
        application_kind="elimination" if elims else "mixed",
        semantic_completeness="partial",
        focus_cells=chain_cells or elim_cells,
        pattern_cells=chain_cells,
        target_cells=elim_cells,
        digits=digits,
        candidate_eliminations=elims,
        pattern_type="chain",
        pattern_subtype=chain_kind,
        anchors=endpoint_cells,
        cover_sets=[{
            "chain_kind": chain_kind,
            "chain_nodes": chain_nodes,
            "chain_links": chain_links,
            "endpoint_cells": endpoint_cells,
            "elimination_digit": digits[0] if len(digits) == 1 else None,
            "elimination_targets": elim_cells,
            "loop": False,
        }],
        roles={
            "chain_node": chain_cells,
            "endpoint": endpoint_cells,
            "elimination_target": elim_cells,
        },
        constraint_explanation=["Chain semantics preserved; exact strong/weak link typing may remain debug-only."],
        narrative_archetype="CHAINS",
        trigger_facts=[f"Technique: {real_name}."],
        confrontation_facts=["Alternating implications propagate along a cell chain."],
        resolution_facts=[summary],
        summary_fact=summary,
        engine_debug_summary=raw_app.engine_debug_summary,
        engine_debug_sanitized=raw_app.engine_debug_sanitized,
    )

def _normalize_ring_application(raw_app: RawApplication) -> CanonicalTechniqueApplication:
    tid = raw_app.technique_id
    real_name = technique_meta(tid).get("real_name", tid)
    elims = _elim_entries_from_removed_chars(raw_app.removed_chars_updated)
    elim_cells = _cell_set_from_eliminations(elims)
    digits = _digits_from_eliminations(elims)
    summary = f"{real_name} uses loop/ring logic to eliminate candidates."

    cell_like: List[Dict[str, Any]] = []
    def collect_cell_like(x: Any) -> None:
        if _looks_like_rc0_tuple(x):
            cell_like.append(cell_ref_from_rc0(int(x[0]), int(x[1])))
            return
        if isinstance(x, dict):
            for k, v in x.items():
                collect_cell_like(k)
                collect_cell_like(v)
        elif isinstance(x, (list, tuple, set)):
            for it in x:
                collect_cell_like(it)
    collect_cell_like(raw_app.application_details)
    dedup = {}
    for c in cell_like:
        dedup[c["cellIndex"]] = c
    ring_cells = [dedup[k] for k in sorted(dedup.keys())]

    ring_nodes: List[Dict[str, Any]] = []
    for i, c in enumerate(ring_cells):
        ring_nodes.append({"index": i, "cell": c, "candidate_digits": [], "role": "chain_node"})
    ring_links: List[Dict[str, Any]] = []
    for i in range(max(0, len(ring_nodes)-1)):
        ring_links.append({"from_node": i, "to_node": i + 1, "digit": None, "link_type": None})
    if len(ring_nodes) >= 2:
        ring_links.append({"from_node": len(ring_nodes)-1, "to_node": 0, "digit": None, "link_type": None})

    app_id = "app:" + sha12(f"{tid}|{raw_app.name_application}|{repr(raw_app.application_details)}")
    return CanonicalTechniqueApplication(
        application_id=app_id,
        technique_id=tid,
        technique_family="ring",
        technique_real_name=real_name,
        application_kind="elimination" if elims else "mixed",
        semantic_completeness="partial",
        focus_cells=ring_cells or elim_cells,
        pattern_cells=ring_cells,
        target_cells=elim_cells,
        digits=digits,
        candidate_eliminations=elims,
        pattern_type="ring",
        pattern_subtype="xy_loop",
        cover_sets=[{
            "chain_kind": "ring",
            "chain_nodes": ring_nodes,
            "chain_links": ring_links,
            "endpoint_cells": [],
            "elimination_digit": digits[0] if len(digits) == 1 else None,
            "elimination_targets": elim_cells,
            "loop": True,
        }],
        roles={
            "chain_node": ring_cells,
            "elimination_target": elim_cells,
        },
        constraint_explanation=["Closed alternating loop semantics preserved."],
        narrative_archetype="CHAINS",
        trigger_facts=[f"Technique: {real_name}."],
        confrontation_facts=["A closed loop fixes parity relationships."],
        resolution_facts=[summary],
        summary_fact=summary,
        engine_debug_summary=raw_app.engine_debug_summary,
        engine_debug_sanitized=raw_app.engine_debug_sanitized,
    )

def _normalize_leftovers_application(raw_app: RawApplication) -> CanonicalTechniqueApplication:
    tid = raw_app.technique_id
    real_name = technique_meta(tid).get("real_name", tid)
    elims = _elim_entries_from_removed_chars(raw_app.removed_chars_updated)
    elim_cells = _cell_set_from_eliminations(elims)
    digits = _digits_from_eliminations(elims)
    level = int(tid.split("-")[-1]) if "-" in tid and tid.split("-")[-1].isdigit() else None
    summary = f"{real_name} balances inside/outside groups to eliminate candidates."

    cell_like: List[Dict[str, Any]] = []
    def collect_cell_like(x: Any) -> None:
        if _looks_like_rc0_tuple(x):
            cell_like.append(cell_ref_from_rc0(int(x[0]), int(x[1])))
            return
        if isinstance(x, dict):
            for k, v in x.items():
                collect_cell_like(k)
                collect_cell_like(v)
        elif isinstance(x, (list, tuple, set)):
            for it in x:
                collect_cell_like(it)
    collect_cell_like(raw_app.application_details)
    dedup = {}
    for c in cell_like:
        dedup[c["cellIndex"]] = c
    groups = [dedup[k] for k in sorted(dedup.keys())]
    half = len(groups) // 2 if len(groups) >= 2 else 0
    inside = groups[:half]
    outside = groups[half:]

    app_id = "app:" + sha12(f"{tid}|{raw_app.name_application}|{repr(raw_app.application_details)}")
    return CanonicalTechniqueApplication(
        application_id=app_id,
        technique_id=tid,
        technique_family="leftovers",
        technique_real_name=real_name,
        application_kind="elimination" if elims else "mixed",
        semantic_completeness="partial",
        focus_cells=inside or outside or elim_cells,
        pattern_cells=groups,
        target_cells=elim_cells,
        digits=digits,
        candidate_eliminations=elims,
        pattern_type="leftovers",
        pattern_subtype=f"level_{level}" if level is not None else None,
        cover_sets=[{
            "leftovers_level": level,
            "region_type": "custom_or_irregular",
            "inside_group_cells": inside,
            "outside_group_cells": outside,
            "balanced_digits": digits,
            "eliminations": elim_cells,
            "overlapping_regions": [],
        }],
        roles={
            "inside_group": inside,
            "outside_group": outside,
            "elimination_target": elim_cells,
        },
        constraint_explanation=["Inside/outside group balance preserved semantically; exact irregular region internals stay in debug."],
        narrative_archetype="UNKNOWN",
        trigger_facts=[f"Technique: {real_name}."],
        confrontation_facts=["Inside/outside region groups constrain digit balance."],
        resolution_facts=[summary],
        summary_fact=summary,
        engine_debug_summary=raw_app.engine_debug_summary,
        engine_debug_sanitized=raw_app.engine_debug_sanitized,
    )

def _normalize_fallback_application(raw_app: RawApplication) -> CanonicalTechniqueApplication:
    tid = raw_app.technique_id
    family = _detect_family(tid)
    real_name = technique_meta(tid).get("real_name", tid)
    elims = _elim_entries_from_removed_chars(raw_app.removed_chars_updated)
    elim_cells = _cell_set_from_eliminations(elims)
    digits = _digits_from_eliminations(elims)
    common_house = _best_common_house(elim_cells)
    kind = "elimination" if elims else "mixed"

    app_id = "app:" + sha12(f"fallback|{tid}|{raw_app.name_application}|{repr(raw_app.application_details)}")
    return CanonicalTechniqueApplication(
        application_id=app_id,
        technique_id=tid,
        technique_family=family,
        technique_real_name=real_name,
        application_kind=kind,
        semantic_completeness="partial" if elims else "debug_only",
        focus_cells=elim_cells,
        pattern_cells=[],
        target_cells=elim_cells,
        houses=[common_house] if common_house else [],
        digits=digits,
        candidate_eliminations=elims,
        pattern_type="fallback",
        pattern_subtype=None,
        cover_sets=[{
            "raw_family": family,
            "elimination_cells": elim_cells,
            "digits": digits,
        }],
        roles={"elimination_target": elim_cells},
        constraint_explanation=["Coarse fallback summary only; raw semantics preserved in debug."],
        narrative_archetype=_detect_archetype(tid),
        trigger_facts=[f"Technique: {real_name}."],
        confrontation_facts=["Technique-specific detail could not be fully canonically extracted."],
        resolution_facts=[f"Observed {len(elims)} candidate eliminations." if elims else "No explicit eliminations parsed."],
        summary_fact=f"{real_name} application summarized with partial semantics.",
        engine_debug_summary=raw_app.engine_debug_summary,
        engine_debug_sanitized=raw_app.engine_debug_sanitized,
    )

def _normalize_application_by_family(raw_app: RawApplication) -> CanonicalTechniqueApplication:
    tid = raw_app.technique_id
    family = _detect_family(tid)
    if family in {"multiple_naked", "multiple_hidden", "multiple_hidden_boxed"}:
        return _normalize_subset_application(raw_app)
    if family == "box_line_interaction":
        return _normalize_box_line_application(raw_app)
    if family == "fish":
        return _normalize_fish_application(raw_app)
    if family == "wing" or tid in {"boxed-wings"}:
        return _normalize_wing_application(raw_app)
    if family == "chain":
        return _normalize_chain_application(raw_app)
    if family == "ring":
        return _normalize_ring_application(raw_app)
    if family == "leftovers":
        return _normalize_leftovers_application(raw_app)
    if tid == "boxed-rays":
        # Closest semantics = boxed pattern / shape interaction
        return _normalize_wing_application(raw_app)
    return _normalize_fallback_application(raw_app)

# ============================================================================
# Final canonical proof synthesis
# ============================================================================

def _candidate_digits_from_mask(mask: Any) -> List[int]:
    try:
        m = int(mask)
    except Exception:
        return []
    return [d for d in range(1, 10) if m & (1 << (d - 1))]


def _candidate_digits_before_for_cell_from_masks(options_all_masks: Dict[str, int], cell_index: int) -> List[int]:
    return _candidate_digits_from_mask(options_all_masks.get(str(int(cell_index)), 0))


def _candidate_digits_before_for_cell(grid81: str, cell_index: int) -> List[int]:
    try:
        ci = int(cell_index)
    except Exception:
        return []

    if not (isinstance(grid81, str) and len(grid81) >= 81 and 0 <= ci < 81):
        return []

    ch = str(grid81[ci])
    if not _grid81_char_is_unsolved(ch):
        try:
            d = int(ch)
            return [d] if d in range(1, 10) else []
        except Exception:
            return []

    used: Set[int] = set()
    for house in houses_for_cell(ci):
        h_type = str(house.get("type", ""))
        try:
            idx1 = int(house.get("index1to9"))
        except Exception:
            continue

        for peer_ci in _house_cells(h_type, idx1):
            if not (0 <= peer_ci < len(grid81)):
                continue
            peer_ch = str(grid81[peer_ci])
            if peer_ch in DIGITS:
                try:
                    used.add(int(peer_ch))
                except Exception:
                    pass

    return [d for d in range(1, 10) if d not in used]


def _grid81_char_is_unsolved(ch: str) -> bool:
    return ch in {EMPTY_CHAR, ".", "0"}


def _candidate_cells_before_for_house_digit_from_masks(
    options_all_masks: Dict[str, int],
    grid81: str,
    house: Dict[str, Any],
    digit: int,
) -> List[int]:
    if digit not in range(1, 10) or not isinstance(house, dict):
        return []
    h_type = str(house.get("type", ""))
    if h_type not in {"row", "col", "box"}:
        return []
    try:
        idx1 = int(house.get("index1to9"))
    except Exception:
        return []
    bit = 1 << (digit - 1)
    out: List[int] = []
    for ci in _house_cells(h_type, idx1):
        if (
            0 <= ci < len(grid81)
            and _grid81_char_is_unsolved(str(grid81[ci]))
            and (int(options_all_masks.get(str(ci), 0)) & bit)
        ):
            out.append(ci)
    return sorted(set(out))


def relation_between_cells(a_ci: int, b_ci: int) -> str:
    """
    Match the Kotlin canonical-reader semantics used by NarrativeAtomModelsV1:
      - SAME_ROW
      - SAME_COL
      - SAME_BOX
      - RELATED
    """
    try:
        a_ci = int(a_ci)
        b_ci = int(b_ci)
    except Exception:
        return "RELATED"

    if not (0 <= a_ci <= 80 and 0 <= b_ci <= 80):
        return "RELATED"

    ar, ac = cell_index_to_rc(a_ci)
    br, bc = cell_index_to_rc(b_ci)

    if ar == br:
        return "SAME_ROW"
    if ac == bc:
        return "SAME_COL"
    if box_index_1to9(ar, ac) == box_index_1to9(br, bc):
        return "SAME_BOX"
    return "RELATED"


def _single_cell_witness_payload(grid81: str, focus_ci: int, digit: int) -> Optional[Dict[str, Any]]:
    if digit not in range(1, 10):
        return None
    r, c = cell_index_to_rc(focus_ci)
    b = box_index_1to9(r, c)
    witness_ci = (
        _find_placed_digit_in_house(grid81, "row", r, digit)
        or _find_placed_digit_in_house(grid81, "col", c, digit)
        or _find_placed_digit_in_house(grid81, "box", b, digit)
    )
    if witness_ci is None:
        return None
    wref = cell_ref_from_index(witness_ci)
    return {
        "kind": "single_cell",
        "digit": digit,
        "cell": wref,
        "relation": relation_between_cells(focus_ci, witness_ci),
    }


def _subset_group_witness_payload(app: CanonicalTechniqueApplication) -> Optional[Dict[str, Any]]:
    if app.pattern_type not in {"subset", "hidden_subset"}:
        return None
    house = app.houses[0] if app.houses else _best_common_house(app.pattern_cells)
    if not isinstance(house, dict):
        return None
    digits = sorted({int(d) for d in (app.digits or []) if int(d) in range(1, 10)})
    cells = sorted({int(c["cellIndex"]) for c in (app.pattern_cells or []) if isinstance(c, dict) and "cellIndex" in c})
    if not cells or not digits:
        return None
    return {
        "kind": "subset_group",
        "subset_kind": app.pattern_subtype or app.technique_id,
        "digits": digits,
        "cells": [cell_ref_from_index(ci) for ci in cells],
        "house": house,
    }


def _target_cell_ref_for_app(
    app: CanonicalTechniqueApplication,
    selected_placement: Optional[PlacementHit],
) -> Optional[Dict[str, Any]]:
    if selected_placement is not None:
        return cell_ref_from_index(selected_placement.cellIndex)
    if app.cell_value_forces:
        return app.cell_value_forces[0].cell
    if app.placements:
        return app.placements[0].cell
    if app.candidate_restrictions:
        return app.candidate_restrictions[0].cell
    if app.target_cells:
        return app.target_cells[0]
    if app.focus_cells:
        return app.focus_cells[0]
    return None


def _target_digit_for_app(
    app: CanonicalTechniqueApplication,
    selected_placement: Optional[PlacementHit],
) -> Optional[int]:
    if selected_placement is not None:
        return selected_placement.digit
    if app.cell_value_forces:
        return app.cell_value_forces[0].digit
    if app.placements:
        return app.placements[0].digit
    digits = [int(d) for d in (app.digits or []) if int(d) in range(1, 10)]
    if len(digits) == 1:
        return digits[0]
    if app.candidate_eliminations:
        uniq = sorted({e.digit for e in app.candidate_eliminations if e.digit in range(1, 10)})
        if len(uniq) == 1:
            return uniq[0]
    return None


def _prefer_primary_house_for_hidden_like(
    app: CanonicalTechniqueApplication,
    target_ci: Optional[int],
) -> Optional[Dict[str, Any]]:
    if target_ci is None:
        return app.houses[0] if app.houses else None
    if app.houses:
        for h in app.houses:
            if not isinstance(h, dict):
                continue
            if h.get("type") in {"row", "col", "box"}:
                idx1 = int(h.get("index1to9", -1))
                if target_ci in _house_cells(str(h.get("type")), idx1):
                    return h
        return app.houses[0]
    if app.focus_cells:
        return _best_common_house(app.focus_cells)
    return None


def _infer_final_resolution_kind(
    *,
    app: CanonicalTechniqueApplication,
    selected_placement: Optional[PlacementHit],
    target_cell: Optional[Dict[str, Any]],
    target_digit: Optional[int],
    grid_before: str,
    options_all_masks: Dict[str, int],
) -> str:
    tid = (app.technique_id or "").strip().lower()
    family = app.technique_family
    target_ci = int(target_cell["cellIndex"]) if isinstance(target_cell, dict) and "cellIndex" in target_cell else None

    if tid in {"singles-naked-2", "singles-naked-3"}:
        return "CELL_CANDIDATE_DIGITS"
    if tid in {"singles-1", "singles-2", "singles-3"}:
        return "HOUSE_CANDIDATE_CELLS_FOR_DIGIT"

    if family == "multiple_naked":
        if selected_placement is not None or app.cell_value_forces or app.placements:
            return "CELL_CANDIDATE_DIGITS"
        if target_ci is not None and any(e.cell["cellIndex"] == target_ci for e in app.candidate_eliminations):
            return "CELL_CANDIDATE_DIGITS"
        return "CELL_CANDIDATE_DIGITS"

    if family in {"multiple_hidden", "multiple_hidden_boxed"}:
        if selected_placement is not None and target_ci is not None:
            return "CELL_CANDIDATE_DIGITS"
        if target_digit is not None:
            primary_house = _prefer_primary_house_for_hidden_like(app, target_ci)
            if primary_house is not None:
                defaults = _candidate_cells_before_for_house_digit_from_masks(options_all_masks, grid_before, primary_house, target_digit)
                if defaults:
                    return "HOUSE_CANDIDATE_CELLS_FOR_DIGIT"
        return "HOUSE_CANDIDATE_CELLS_FOR_DIGIT"

    if selected_placement is not None:
        if target_ci is not None and any(e.cell["cellIndex"] == target_ci for e in app.candidate_eliminations):
            return "CELL_CANDIDATE_DIGITS"
        if target_ci is not None and any(r.cell["cellIndex"] == target_ci for r in app.candidate_restrictions):
            return "CELL_CANDIDATE_DIGITS"

    if target_digit is not None and app.houses:
        house = app.houses[0]
        defaults = _candidate_cells_before_for_house_digit_from_masks(options_all_masks, grid_before, house, target_digit)
        if len(defaults) >= 2 and app.candidate_eliminations:
            same_digit = [e for e in app.candidate_eliminations if e.digit == target_digit]
            if same_digit:
                return "HOUSE_CANDIDATE_CELLS_FOR_DIGIT"

    if target_ci is not None:
        return "CELL_CANDIDATE_DIGITS"
    return "HOUSE_CANDIDATE_CELLS_FOR_DIGIT"


def _build_canonical_cell_proof(
    *,
    app: CanonicalTechniqueApplication,
    grid_before: str,
    options_all_masks: Dict[str, int],
    target_cell: Dict[str, Any],
    target_digit: Optional[int],
) -> Dict[str, Any]:
    focus_ci = int(target_cell["cellIndex"])
    default_digits = _candidate_digits_before_for_cell_from_masks(options_all_masks, focus_ci)
    actual_remaining = None
    for r in app.candidate_restrictions:
        if int(r.cell["cellIndex"]) == focus_ci and r.allowed_digits:
            actual_remaining = sorted({int(d) for d in r.allowed_digits if int(d) in range(1, 10)})
            break
    if target_digit is not None:
        remaining_digits = [int(target_digit)]
    elif actual_remaining is not None:
        remaining_digits = actual_remaining
    else:
        remaining_digits = default_digits[:] if default_digits else []
        if len(remaining_digits) > 1:
            remaining_digits = remaining_digits[:]
    if not remaining_digits and target_digit is not None:
        remaining_digits = [int(target_digit)]
    remaining_digits = sorted({int(d) for d in remaining_digits if int(d) in range(1, 10)})
    universe_digits = list(range(1, 10))
    claimed_digits = [d for d in universe_digits if d not in remaining_digits]

    explicit_elims = {e.digit for e in app.candidate_eliminations if int(e.cell["cellIndex"]) == focus_ci and e.digit in range(1, 10)}
    explicit_removed = set()
    for r in app.candidate_restrictions:
        if int(r.cell["cellIndex"]) == focus_ci:
            explicit_removed.update(int(d) for d in (r.removed_digits or []) if int(d) in range(1, 10))
    subset_witness = _subset_group_witness_payload(app)
    subset_digits = set(subset_witness.get("digits", [])) if isinstance(subset_witness, dict) else set()
    subset_focus_applies = focus_ci in {int(c["cellIndex"]) for c in (app.target_cells or [])} or focus_ci in {int(c["cellIndex"]) for c in (app.peer_cells or [])}
    if app.pattern_type in {"subset", "hidden_subset"} and not subset_focus_applies and app.focus_cells:
        subset_focus_applies = focus_ci in {int(c["cellIndex"]) for c in app.focus_cells}

    witness_cells: Dict[int, Dict[str, Any]] = {}
    witness_by_digit: List[Dict[str, Any]] = []
    support_provenance: Dict[str, str] = {}

    for digit in claimed_digits:
        witness = None
        provenance = "sudoku_peer"
        if subset_witness is not None and subset_focus_applies and digit in subset_digits and (digit in explicit_elims or digit in explicit_removed or app.technique_family.startswith("multiple_")):
            witness = subset_witness
            provenance = "technique_witness"
            for c in subset_witness.get("cells", []):
                if isinstance(c, dict) and "cellIndex" in c:
                    witness_cells[int(c["cellIndex"])] = c
        else:
            w = _single_cell_witness_payload(grid_before, focus_ci, digit)
            if w is not None:
                witness = w
                c = w.get("cell")
                if isinstance(c, dict) and "cellIndex" in c:
                    witness_cells[int(c["cellIndex"])] = c
            elif digit in explicit_elims or digit in explicit_removed:
                provenance = "technique_effect_without_explicit_witness"
                witness = {"kind": "unknown", "digit": digit}
            else:
                provenance = "preexcluded_or_unknown"
                witness = {"kind": "unknown", "digit": digit}
        support_provenance[str(digit)] = provenance
        witness_by_digit.append({"digit": digit, "witness": witness})

    technique_eliminated_digits: List[int] = []
    peer_eliminated_digits: List[int] = []
    unknown_eliminated_digits: List[int] = []
    blocker_rows: List[Dict[str, Any]] = []

    for row in witness_by_digit:
        digit = int(row.get("digit", -1))
        witness = row.get("witness") or {}
        kind = witness.get("kind", "unknown")

        if kind == "subset_group":
            technique_eliminated_digits.append(digit)
            blocker_rows.append({
                "digit": digit,
                "source_kind": "subset_witness",
                "subset_kind": witness.get("subset_kind"),
                "subset_digits": to_json_safe(witness.get("digits", [])),
                "subset_cells": to_json_safe(witness.get("cells", [])),
                "house": to_json_safe(witness.get("house")),
            })
        elif kind == "single_cell":
            peer_eliminated_digits.append(digit)
            blocker_rows.append({
                "digit": digit,
                "source_kind": "peer_witness",
                "witness_cell": to_json_safe(witness.get("cell")),
                "relation": witness.get("relation"),
            })
        else:
            unknown_eliminated_digits.append(digit)
            blocker_rows.append({
                "digit": digit,
                "source_kind": "unknown",
            })

    technique_eliminated_digits = sorted({int(d) for d in technique_eliminated_digits if int(d) in range(1, 10)})
    peer_eliminated_digits = sorted({int(d) for d in peer_eliminated_digits if int(d) in range(1, 10)})
    unknown_eliminated_digits = sorted({int(d) for d in unknown_eliminated_digits if int(d) in range(1, 10)})

    if technique_eliminated_digits and peer_eliminated_digits:
        forcing_note = "Technique cleanup first, ordinary peer blockers second."
    elif technique_eliminated_digits:
        forcing_note = "Technique cleanup is the decisive reason this target collapses."
    else:
        forcing_note = "Ordinary peer blockers complete the collapse around the target."

    return {
        "version": "canonical_final_proof_v1",
        "elimination_kind": "CELL_CANDIDATE_DIGITS",
        "primary_house": {"type": "cell", "cell": target_cell},
        "focus_cell": target_cell,
        "digit": remaining_digits[0] if len(remaining_digits) == 1 else target_digit,
        "proof_payload": {
            "cell_outcome": {
                "cell": target_cell,
                "default_candidate_digits": default_digits,
                "universe_candidate_digits": universe_digits,
                "claimed_candidate_digits": claimed_digits,
                "remaining_candidate_digits": remaining_digits,
            },
            "support": {
                "witness_cells": [witness_cells[k] for k in sorted(witness_cells.keys())],
                "witness_by_digit": witness_by_digit,
                "eliminated_digits": claimed_digits,
                "support_provenance": support_provenance,
                "technique_eliminated_digits": technique_eliminated_digits,
                "peer_eliminated_digits": peer_eliminated_digits,
                "unknown_eliminated_digits": unknown_eliminated_digits,
                "blocker_rows": blocker_rows,
                "forcing_summary": {
                    "technique_eliminated_digits": technique_eliminated_digits,
                    "peer_eliminated_digits": peer_eliminated_digits,
                    "unknown_eliminated_digits": unknown_eliminated_digits,
                    "remaining_candidate_digits": remaining_digits,
                    "forcing_note": forcing_note,
                },
            },
        },
    }


def _build_canonical_house_proof(
    *,
    app: CanonicalTechniqueApplication,
    grid_before: str,
    options_all_masks: Dict[str, int],
    target_cell: Optional[Dict[str, Any]],
    target_digit: Optional[int],
    target_house: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if target_digit not in range(1, 10):
        digits = [int(d) for d in (app.digits or []) if int(d) in range(1, 10)]
        target_digit = digits[0] if digits else None

    primary_house = to_json_safe(target_house or {})
    if not isinstance(primary_house, dict) or not primary_house.get("type"):
        primary_house = _prefer_primary_house_for_hidden_like(
            app,
            int(target_cell["cellIndex"]) if isinstance(target_cell, dict) and "cellIndex" in target_cell else None,
        )
    if primary_house is None:
        primary_house = app.houses[0] if app.houses else None

    if primary_house is None or target_digit not in range(1, 10):
        return {
            "version": "canonical_final_proof_v1",
            "elimination_kind": "HOUSE_CANDIDATE_CELLS_FOR_DIGIT",
            "primary_house": primary_house or {},
            "target_house": primary_house or {},
            "digit": target_digit,
            "proof_payload": {
                "house_claim": {
                    "digit": target_digit,
                    "house": primary_house or {},
                    "default_candidate_cells": [],
                    "claimed_candidate_cells": [],
                    "remaining_candidate_cells": [target_cell] if target_cell else [],
                },
                "support": {"peer_cells": [], "witness_by_cell": [], "support_provenance": {}},
            },
        }

    default_cells = _candidate_cells_before_for_house_digit_from_masks(options_all_masks, grid_before, primary_house, int(target_digit))
    if isinstance(target_cell, dict) and int(target_cell["cellIndex"]) not in default_cells and target_cell.get("cellIndex") is not None:
        default_cells = sorted(set(default_cells + [int(target_cell["cellIndex"])]))

    explicit_claimed = sorted({
        int(e.cell["cellIndex"])
        for e in app.candidate_eliminations
        if e.digit == int(target_digit) and int(e.cell["cellIndex"]) in default_cells
    })
    if not explicit_claimed and target_cell is not None:
        remaining_cells = [int(target_cell["cellIndex"])]
    elif target_cell is not None and int(target_cell["cellIndex"]) in default_cells:
        remaining_cells = [int(target_cell["cellIndex"])]
    elif app.pattern_type == "hidden_subset" and app.pattern_cells:
        remaining_cells = [int(c["cellIndex"]) for c in app.pattern_cells if int(c["cellIndex"]) in default_cells]
    else:
        remaining_cells = [ci for ci in default_cells if ci not in explicit_claimed]
    remaining_cells = sorted(set(remaining_cells))
    if not remaining_cells and default_cells:
        remaining_cells = [default_cells[0]]
    claimed_cells = [ci for ci in default_cells if ci not in remaining_cells]

    witness_rows: List[Dict[str, Any]] = []
    peer_cells: List[Dict[str, Any]] = []
    support_provenance: Dict[str, str] = {}
    subset_witness = _subset_group_witness_payload(app)
    subset_digits = set(subset_witness.get("digits", [])) if isinstance(subset_witness, dict) else set()

    for ci in claimed_cells:
        claimed_ref = cell_ref_from_index(ci)
        peer_cells.append(claimed_ref)
        witness = None
        provenance = "sudoku_peer"
        if subset_witness is not None and int(target_digit) in subset_digits:
            witness = subset_witness
            provenance = "technique_witness"
        else:
            witness_ci = _witness_for_peer_cell(
                grid81=grid_before,
                peer_ci=ci,
                digit=int(target_digit),
                primary_house_type=str(primary_house.get("type", "")),
            )
            if witness_ci is not None:
                witness = {
                    "kind": "single_cell",
                    "digit": int(target_digit),
                    "cell": cell_ref_from_index(witness_ci),
                    "relation": relation_between_cells(ci, witness_ci),
                }
            else:
                provenance = "preexcluded_or_unknown"
                witness = {"kind": "unknown", "digit": int(target_digit)}
        support_provenance[str(ci)] = provenance
        witness_rows.append({"claimed_cell": claimed_ref, "witness": witness})

    return {
        "version": "canonical_final_proof_v1",
        "elimination_kind": "HOUSE_CANDIDATE_CELLS_FOR_DIGIT",
        "primary_house": primary_house,
        "digit": int(target_digit),
        "focus_cell": target_cell,
        "proof_payload": {
            "house_claim": {
                "digit": int(target_digit),
                "house": primary_house,
                "default_candidate_cells": [cell_ref_from_index(ci) for ci in default_cells],
                "claimed_candidate_cells": [cell_ref_from_index(ci) for ci in claimed_cells],
                "remaining_candidate_cells": [cell_ref_from_index(ci) for ci in remaining_cells],
            },
            "support": {
                "peer_cells": peer_cells,
                "witness_by_cell": witness_rows,
                "support_provenance": support_provenance,
            },
        },
    }


def _validate_final_canonical_proof(final_proof: Dict[str, Any]) -> Dict[str, Any]:
    result = {"ok": True, "errors": []}
    if not isinstance(final_proof, dict):
        return {"ok": False, "errors": ["final_proof_not_dict"]}
    kind = str(final_proof.get("elimination_kind", ""))
    payload = final_proof.get("proof_payload") or {}
    if kind == "CELL_CANDIDATE_DIGITS":
        cell_outcome = payload.get("cell_outcome") or {}
        remaining = sorted({int(d) for d in (cell_outcome.get("remaining_candidate_digits") or []) if int(d) in range(1, 10)})
        claimed = sorted({int(d) for d in (cell_outcome.get("claimed_candidate_digits") or []) if int(d) in range(1, 10)})
        if not remaining:
            result["ok"] = False
            result["errors"].append("cell_remaining_digits_empty")
        if set(remaining) & set(claimed):
            result["ok"] = False
            result["errors"].append("cell_claimed_remaining_overlap")
        if set(remaining) | set(claimed) != set(range(1, 10)):
            result["ok"] = False
            result["errors"].append("cell_claimed_plus_remaining_not_full_universe")
    elif kind == "HOUSE_CANDIDATE_CELLS_FOR_DIGIT":
        house_claim = payload.get("house_claim") or {}
        default_cells = {int(c.get("cellIndex")) for c in (house_claim.get("default_candidate_cells") or []) if isinstance(c, dict) and c.get("cellIndex") is not None}
        claimed_cells = {int(c.get("cellIndex")) for c in (house_claim.get("claimed_candidate_cells") or []) if isinstance(c, dict) and c.get("cellIndex") is not None}
        remaining_cells = {int(c.get("cellIndex")) for c in (house_claim.get("remaining_candidate_cells") or []) if isinstance(c, dict) and c.get("cellIndex") is not None}
        if not remaining_cells:
            result["ok"] = False
            result["errors"].append("house_remaining_cells_empty")
        if claimed_cells & remaining_cells:
            result["ok"] = False
            result["errors"].append("house_claimed_remaining_overlap")
        if default_cells and (claimed_cells | remaining_cells) != default_cells:
            result["ok"] = False
            result["errors"].append("house_claimed_plus_remaining_not_default")
    else:
        result["ok"] = False
        result["errors"].append("unknown_elimination_kind")
    return result


def _build_technique_witness_snapshot(app: CanonicalTechniqueApplication) -> Dict[str, Any]:
    return {
        "technique_id": app.technique_id,
        "technique_family": app.technique_family,
        "pattern_type": app.pattern_type,
        "pattern_subtype": app.pattern_subtype,
        "houses": to_json_safe(app.houses),
        "digits": to_json_safe(app.digits),
        "pattern_cells": to_json_safe(app.pattern_cells),
        "focus_cells": to_json_safe(app.focus_cells),
        "target_cells": to_json_safe(app.target_cells),
        "witness_cells": to_json_safe(app.witness_cells),
        "roles": to_json_safe(app.roles),
    }



def _find_single_cell_blocking_witness_for_digit(
    *,
    grid_before: str,
    target_ci: int,
    digit: int,
) -> Optional[Dict[str, Any]]:
    if digit not in range(1, 10):
        return None
    if not (0 <= int(target_ci) <= 80):
        return None

    # Setup witness reconstruction only makes sense for an OPEN target seat.
    # If the square is already filled, it is not a candidate seat and must not
    # be narrated as "blocked".
    if not (0 <= int(target_ci) < len(grid_before)):
        return None
    target_ch = str(grid_before[int(target_ci)])
    if target_ch in DIGITS:
        return None

    for house in houses_for_cell(int(target_ci)):
        h_type = str(house.get("type", ""))
        if h_type not in {"row", "col", "box"}:
            continue
        try:
            idx1 = int(house.get("index1to9"))
        except Exception:
            continue

        for peer_ci in _house_cells(h_type, idx1):
            if int(peer_ci) == int(target_ci):
                continue
            if 0 <= int(peer_ci) < len(grid_before) and grid_before[int(peer_ci)] == str(digit):
                return {
                    "reason_kind": "single_cell_witness",
                    "witness_cell": cell_ref_from_index(int(peer_ci)),
                    "relation": relation_between_cells(int(target_ci), int(peer_ci)),
                    "house": normalize_house(h_type, idx1),
                }

    return None


def _build_intersection_outside_seat_witness_row(
    *,
    audited_ci: int,
    digit: int,
    source_house: Dict[str, Any],
    witness_reason: Optional[Dict[str, Any]],
    digit_is_live_here: bool,
) -> Dict[str, Any]:
    audited_cell = cell_ref_from_index(int(audited_ci))
    audited_label = cell_ref_label(audited_cell)

    row: Dict[str, Any] = {
        "source_kind": "intersection_outside_witness",
        "digit": int(digit),
        "cell": audited_cell,
        "claimed_cell": audited_cell,
        "audited_cell": audited_cell,
        "cell_label": audited_label,
        "audited_cell_label": audited_label,
        "seat_state": "open",
        "digit_live_here": bool(digit_is_live_here),
        "narration_role": "source_house_outside_open_seat",
        "narratable_in_setup": True,
        "source_house": to_json_safe(source_house),
        "source_house_label": house_label(source_house) if isinstance(source_house, dict) else "",
        "proof_family": "intersection_source_house_outside_audit",
        "proof_strength": "explicit_witness",
    }

    if isinstance(witness_reason, dict):
        witness_cell = witness_reason.get("witness_cell")
        witness_house = witness_reason.get("house")
        relation = witness_reason.get("relation")

        row["reason_kind"] = "single_cell_witness"
        row["closure_kind"] = "witness_blocked"
        row["relation"] = relation

        if isinstance(witness_cell, dict):
            row["witness_cell"] = to_json_safe(witness_cell)
            row["witness_cell_label"] = cell_ref_label(witness_cell)

        if isinstance(witness_house, dict):
            row["house"] = to_json_safe(witness_house)
            row["house_scope"] = to_json_safe(witness_house)
            row["house_label"] = house_label(witness_house)

        if isinstance(witness_cell, dict) and isinstance(witness_house, dict):
            witness_label = cell_ref_label(witness_cell)
            house_lbl = house_label(witness_house)
            row["because"] = (
                f"{audited_label} is blocked for {int(digit)} because "
                f"{witness_label} already places {int(digit)} in {house_lbl}."
            )
            row["spoken_reason"] = row["because"]
            row["setup_spoken_line"] = (
                f"{audited_label} is blocked by the {int(digit)} already sitting at "
                f"{witness_label} in {house_lbl}."
            )
            row["overlay_line"] = row["setup_spoken_line"]
            row["setup_priority"] = 100 if str(relation or "") in {"same_col", "same_row", "same_box"} else 90
        else:
            row["because"] = (
                f"{audited_label} is blocked for {int(digit)} by an already occupied peer house."
            )
            row["spoken_reason"] = row["because"]
            row["setup_spoken_line"] = row["because"]
            row["overlay_line"] = row["setup_spoken_line"]
            row["setup_priority"] = 80

        return row

    row["reason_kind"] = "candidate_absent" if not digit_is_live_here else "unknown"
    row["closure_kind"] = "digit_absent" if not digit_is_live_here else "unknown"
    row["relation"] = None
    row["proof_strength"] = "fallback"

    if not digit_is_live_here:
        row["because"] = (
            f"{audited_label} is open, but {int(digit)} is no longer one of its live candidates."
        )
        row["spoken_reason"] = row["because"]
        row["setup_spoken_line"] = row["because"]
        row["overlay_line"] = row["setup_spoken_line"]
        row["setup_priority"] = 40
    else:
        row["because"] = (
            f"{audited_label} is open, but the blocker for {int(digit)} could not be reconstructed cleanly."
        )
        row["spoken_reason"] = row["because"]
        row["setup_spoken_line"] = row["because"]
        row["overlay_line"] = row["setup_spoken_line"]
        row["setup_priority"] = 10

    return row





def _build_intersection_source_outside_setup_audit(
    *,
    grid_before: str,
    options_all_masks: Dict[str, int],
    digit: int,
    source_house: Dict[str, Any],
    source_outside_overlap_open_cells: List[Any],
) -> Tuple[List[Dict[str, Any]], str]:
    audit_rows: List[Dict[str, Any]] = []
    semantic_completeness = "full"

    for raw_cell in source_outside_overlap_open_cells:
        ci: Optional[int] = None

        if isinstance(raw_cell, dict):
            if raw_cell.get("cellIndex") is not None:
                try:
                    ci = int(raw_cell.get("cellIndex"))
                except Exception:
                    ci = None
            elif isinstance(raw_cell.get("cell"), dict) and raw_cell["cell"].get("cellIndex") is not None:
                try:
                    ci = int(raw_cell["cell"].get("cellIndex"))
                except Exception:
                    ci = None
        elif isinstance(raw_cell, int):
            ci = int(raw_cell)

        if ci is None or not (0 <= int(ci) <= 80):
            semantic_completeness = "partial"
            continue

        # Outside setup audit is about OPEN seats only.
        if not (0 <= int(ci) < len(grid_before)):
            semantic_completeness = "partial"
            continue
        if str(grid_before[int(ci)]) in DIGITS:
            semantic_completeness = "partial"
            continue

        mask_digits = _candidate_digits_before_for_cell_from_masks(options_all_masks, int(ci))
        digit_is_live_here = int(digit) in mask_digits

        witness_reason = _find_single_cell_blocking_witness_for_digit(
            grid_before=grid_before,
            target_ci=int(ci),
            digit=int(digit),
        )

        row = _build_intersection_outside_seat_witness_row(
            audited_ci=int(ci),
            digit=int(digit),
            source_house=source_house,
            witness_reason=witness_reason,
            digit_is_live_here=bool(digit_is_live_here),
        )

        if str(row.get("closure_kind", "")) not in {"witness_blocked", "digit_absent"}:
            semantic_completeness = "partial"

        audit_rows.append(row)

    audit_rows.sort(
        key=lambda r: (
            -int(r.get("setup_priority", 0) or 0),
            int((((r.get("cell") or {}).get("cellIndex")) if isinstance(r.get("cell"), dict) else 999) or 999),
        )
    )

    return audit_rows, semantic_completeness


def _build_intersection_source_confinement_proof(
    *,
    app: CanonicalTechniqueApplication,
    grid_before: str,
    options_all_masks: Dict[str, int],
) -> Dict[str, Any]:
    if app.technique_family != "box_line_interaction":
        return {}

    cover = app.cover_sets[0] if app.cover_sets else {}
    if not isinstance(cover, dict):
        return {}

    source_house = cover.get("source_house")
    cross_house = cover.get("cross_house", cover.get("target_house"))
    box_house = cover.get("box_house")
    line_house = cover.get("line_house")
    direction_mode = str(cover.get("direction_mode", cover.get("interaction_kind", "")) or "")
    pattern_subtype = str(cover.get("pattern_subtype", app.pattern_subtype or "") or "")
    digit = cover.get("digit")

    overlap_cells = to_json_safe(cover.get("overlap_cells", cover.get("locked_cells", app.pattern_cells)))
    pattern_cells = to_json_safe(cover.get("pattern_cells", overlap_cells))

    source_outside_overlap_all_cells = to_json_safe(cover.get("source_outside_overlap_all_cells", []))
    source_outside_overlap_open_cells = to_json_safe(
        cover.get("source_outside_overlap_open_cells", cover.get("source_outside_overlap_cells", []))
    )
    source_outside_overlap_open_candidate_cells = to_json_safe(
        cover.get("source_outside_overlap_open_candidate_cells", [])
    )
    source_outside_overlap_open_noncandidate_cells = to_json_safe(
        cover.get("source_outside_overlap_open_noncandidate_cells", [])
    )

    cross_outside_overlap_all_cells = to_json_safe(cover.get("cross_outside_overlap_all_cells", []))
    cross_outside_overlap_open_cells = to_json_safe(
        cover.get("cross_outside_overlap_open_cells", cover.get("cross_outside_overlap_cells", []))
    )
    cross_outside_overlap_open_candidate_cells = to_json_safe(
        cover.get("cross_outside_overlap_open_candidate_cells", [])
    )
    cross_outside_overlap_open_noncandidate_cells = to_json_safe(
        cover.get("cross_outside_overlap_open_noncandidate_cells", [])
    )

    forbidden_cross_cells = to_json_safe(cover.get("forbidden_cross_cells", []))
    sweep_cells = to_json_safe(cover.get("sweep_cells", app.peer_cells))

    if not isinstance(source_house, dict) or digit not in range(1, 10):
        return {}

    overlap_indices = {
        int(c.get("cellIndex"))
        for c in overlap_cells
        if isinstance(c, dict) and c.get("cellIndex") is not None
    }

    source_candidate_indices = _candidate_cells_before_for_house_digit_from_masks(
        options_all_masks=options_all_masks,
        grid81=grid_before,
        house=source_house,
        digit=int(digit),
    )

    surviving_cells = [
        cell_ref_from_index(ci)
        for ci in source_candidate_indices
        if ci in overlap_indices
    ]

    if not surviving_cells and overlap_indices:
        surviving_cells = [cell_ref_from_index(ci) for ci in sorted(overlap_indices)]

    audit_rows, audit_semantic_completeness = _build_intersection_source_outside_setup_audit(
        grid_before=grid_before,
        options_all_masks=options_all_masks,
        digit=int(digit),
        source_house=source_house,
        source_outside_overlap_open_cells=source_outside_overlap_open_cells,
    )
    semantic_completeness = audit_semantic_completeness

    explicit_outside_witness_rows = [
        r for r in audit_rows
        if str(r.get("source_kind", "")) == "intersection_outside_witness"
    ]
    witness_closure_rows = [
        r for r in explicit_outside_witness_rows
        if str(r.get("closure_kind", "")) == "witness_blocked"
    ]
    open_noncandidate_rows = [
        r for r in explicit_outside_witness_rows
        if str(r.get("closure_kind", "")) == "digit_absent"
    ]

    setup_preferred_witness_rows = witness_closure_rows[:3]
    if len(setup_preferred_witness_rows) < 2:
        setup_preferred_witness_rows.extend(
            open_noncandidate_rows[: max(0, 2 - len(setup_preferred_witness_rows))]
        )

    setup_preferred_audit_rows = list(setup_preferred_witness_rows)

    if len(surviving_cells) != len(overlap_indices):
        semantic_completeness = "partial"

    trigger_pattern = {
        "interaction_kind": direction_mode,
        "pattern_subtype": pattern_subtype,
        "digit": int(digit),
        "source_house": to_json_safe(source_house),
        "cross_house": to_json_safe(cross_house),
        "box_house": to_json_safe(box_house),
        "line_house": to_json_safe(line_house),
        "overlap_cells": to_json_safe(overlap_cells),
        "pattern_cells": to_json_safe(pattern_cells),

        "source_outside_overlap_cells": to_json_safe(source_outside_overlap_open_cells),
        "source_outside_overlap_all_cells": to_json_safe(source_outside_overlap_all_cells),
        "source_outside_overlap_open_cells": to_json_safe(source_outside_overlap_open_cells),
        "source_outside_overlap_open_candidate_cells": to_json_safe(source_outside_overlap_open_candidate_cells),
        "source_outside_overlap_open_noncandidate_cells": to_json_safe(source_outside_overlap_open_noncandidate_cells),

        "cross_outside_overlap_cells": to_json_safe(cross_outside_overlap_open_cells),
        "cross_outside_overlap_all_cells": to_json_safe(cross_outside_overlap_all_cells),
        "cross_outside_overlap_open_cells": to_json_safe(cross_outside_overlap_open_cells),
        "cross_outside_overlap_open_candidate_cells": to_json_safe(cross_outside_overlap_open_candidate_cells),
        "cross_outside_overlap_open_noncandidate_cells": to_json_safe(cross_outside_overlap_open_noncandidate_cells),

        "forbidden_cross_cells": to_json_safe(forbidden_cross_cells),
        "cardinality": len(overlap_cells),
    }

    trigger_explanation = {
        "setup_source_audit_rows": to_json_safe(audit_rows),
        "outside_open_seat_audit_rows": to_json_safe(audit_rows),
        "outside_explicit_witness_rows": to_json_safe(explicit_outside_witness_rows),
        "source_house_outside_witness_rows": to_json_safe(explicit_outside_witness_rows),
        "outside_witness_closure_rows": to_json_safe(witness_closure_rows),
        "outside_open_noncandidate_rows": to_json_safe(open_noncandidate_rows),
        "setup_preferred_witness_rows": to_json_safe(setup_preferred_witness_rows),
        "setup_preferred_audit_rows": to_json_safe(setup_preferred_audit_rows),

        "outside_open_seat_cells": to_json_safe(source_outside_overlap_open_cells),
        "outside_open_candidate_cells": to_json_safe(source_outside_overlap_open_candidate_cells),
        "outside_open_noncandidate_cells": to_json_safe(source_outside_overlap_open_noncandidate_cells),

        "outside_open_seat_count": len(source_outside_overlap_open_cells),
        "outside_explicit_witness_count": len(explicit_outside_witness_rows),
        "outside_witness_closure_count": len(witness_closure_rows),
        "outside_open_noncandidate_count": len(open_noncandidate_rows),

        "overlap_survivor_cells": to_json_safe(surviving_cells),
        "forced_inward_reason": (
            f"Among the open seats outside the overlap, {house_label(source_house)} has nowhere left to place {int(digit)}."
            if isinstance(source_house, dict) else ""
        ),
        "forced_into_overlap_summary": (
            f"Among the open seats outside the overlap, {house_label(source_house)} has nowhere left to place {int(digit)}."
            if isinstance(source_house, dict) else ""
        ),
        "pattern_reveal_moment": (
            f"That forces {int(digit)} into the overlap cells, and that is exactly the {pattern_subtype.replace('_', ' ')} pattern."
            if pattern_subtype else ""
        ),
    }

    trigger_bridge = {
        "cross_house_now_restricted": True,
        "cross_house": to_json_safe(cross_house),
        "forbidden_elsewhere_cells": to_json_safe(forbidden_cross_cells),
        "cleanup_digits": [int(digit)],
        "cross_house_permission_change": (
            f"Once {int(digit)} is trapped in the overlap, {house_label(cross_house)} cannot keep {int(digit)} anywhere else."
            if isinstance(cross_house, dict) else ""
        ),
    }

    return {
        "semantic_completeness": semantic_completeness,
        "digit": int(digit),
        "interaction_kind": direction_mode,
        "pattern_subtype": pattern_subtype,
        "source_house": to_json_safe(source_house),
        "cross_house": to_json_safe(cross_house),
        "overlap_cells": to_json_safe(overlap_cells),
        "surviving_cells": to_json_safe(surviving_cells),

        "outside_audit_rows": to_json_safe(audit_rows),
        "outside_open_seat_audit_rows": to_json_safe(audit_rows),
        "outside_explicit_witness_rows": to_json_safe(explicit_outside_witness_rows),
        "source_house_outside_witness_rows": to_json_safe(explicit_outside_witness_rows),
        "outside_witness_closure_rows": to_json_safe(witness_closure_rows),
        "outside_open_noncandidate_rows": to_json_safe(open_noncandidate_rows),
        "setup_preferred_witness_rows": to_json_safe(setup_preferred_witness_rows),
        "setup_preferred_audit_rows": to_json_safe(setup_preferred_audit_rows),

        "source_outside_overlap_all_cells": to_json_safe(source_outside_overlap_all_cells),
        "source_outside_overlap_open_cells": to_json_safe(source_outside_overlap_open_cells),
        "source_outside_overlap_open_candidate_cells": to_json_safe(source_outside_overlap_open_candidate_cells),
        "source_outside_overlap_open_noncandidate_cells": to_json_safe(source_outside_overlap_open_noncandidate_cells),

        "trigger_pattern": to_json_safe(trigger_pattern),
        "trigger_explanation": to_json_safe(trigger_explanation),
        "trigger_bridge": to_json_safe(trigger_bridge),

        "conclusion": {
            "kind": "digit_confined_to_overlap",
            "locked_cells": to_json_safe(overlap_cells),
            "permission_change": trigger_bridge.get("cross_house_permission_change", ""),
        },
    }





def _explicit_downstream_placement_for_intersection(
    app: CanonicalTechniqueApplication,
    selected_placement: Optional[PlacementHit],
) -> Optional[Tuple[Dict[str, Any], int]]:
    if selected_placement is not None:
        return cell_ref_from_index(int(selected_placement.cellIndex)), int(selected_placement.digit)

    for p in app.cell_value_forces:
        if isinstance(p.cell, dict) and p.cell.get("cellIndex") is not None and p.digit in range(1, 10):
            return to_json_safe(p.cell), int(p.digit)

    for p in app.placements:
        if isinstance(p.cell, dict) and p.cell.get("cellIndex") is not None and p.digit in range(1, 10):
            return to_json_safe(p.cell), int(p.digit)

    return None


def _houses_through_cell_ref(cell: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(cell, dict):
        return []

    r = cell.get("r")
    c = cell.get("c")
    if r not in range(1, 10) or c not in range(1, 10):
        ci = cell.get("cellIndex")
        try:
            if ci is None:
                return []
            r, c = cell_index_to_rc(int(ci))
        except Exception:
            return []

    r = int(r)
    c = int(c)
    b = ((r - 1) // 3) * 3 + ((c - 1) // 3) + 1
    return [
        {"type": "row", "index1to9": r},
        {"type": "col", "index1to9": c},
        {"type": "box", "index1to9": b},
    ]


def _infer_intersection_final_resolution(
    *,
    app: CanonicalTechniqueApplication,
    selected_placement: Optional[PlacementHit],
    grid_before: str,
    options_all_masks: Dict[str, int],
) -> Dict[str, Any]:
    target_cell = _target_cell_ref_for_app(app, selected_placement)
    target_digit = _target_digit_for_app(app, selected_placement)
    target_ci = int(target_cell["cellIndex"]) if isinstance(target_cell, dict) and target_cell.get("cellIndex") is not None else None

    if target_ci is not None and target_digit in range(1, 10):
        for house in _houses_through_cell_ref(target_cell):
            default_cells = _candidate_cells_before_for_house_digit_from_masks(
                options_all_masks,
                grid_before,
                house,
                int(target_digit),
            )
            if target_ci not in default_cells or len(default_cells) < 2:
                continue

            explicit_claimed = sorted({
                int(e.cell["cellIndex"])
                for e in app.candidate_eliminations
                if e.digit == int(target_digit) and int(e.cell["cellIndex"]) in default_cells
            })
            if not explicit_claimed:
                continue

            remaining_cells = [ci for ci in default_cells if ci not in explicit_claimed]
            if remaining_cells == [target_ci]:
                return {
                    "kind": "HOUSE_CANDIDATE_CELLS_FOR_DIGIT",
                    "target_house": to_json_safe(house),
                    "focus_cell": to_json_safe(target_cell),
                    "target_digit": int(target_digit),
                }

    fallback_kind = _infer_final_resolution_kind(
        app=app,
        selected_placement=selected_placement,
        target_cell=target_cell,
        target_digit=target_digit,
        grid_before=grid_before,
        options_all_masks=options_all_masks,
    )
    target_house = {"type": "cell", "cell": to_json_safe(target_cell)} if fallback_kind == "CELL_CANDIDATE_DIGITS" else {}
    return {
        "kind": str(fallback_kind or ""),
        "target_house": target_house,
        "focus_cell": to_json_safe(target_cell or {}),
        "target_digit": int(target_digit) if target_digit in range(1, 10) else None,
    }




def _synthesize_final_canonical_proof(
    *,
    app: CanonicalTechniqueApplication,
    grid_before: str,
    options_all_masks: Dict[str, int],
    selected_placement: Optional[PlacementHit],
) -> Dict[str, Any]:
    """
    Advanced-technique final proof synthesis.

    MECE structure by narrative archetype:
      - INTERSECTIONS
      - SUBSETS
      - FISH
      - WINGS
      - CHAINS
      - UNKNOWN

    Rule:
      Each archetype gets its own explicit branch.
      Shared logic is allowed only when the treatment is truly identical.

    Wave-0 constitution for INTERSECTIONS:
      - treat intersections as an advanced territorial-control family
      - setup truth must ultimately support an explicit source-house-outside-
        overlap audit before pattern naming
      - confrontation truth must preserve two-actor ownership:
            ordinary witnesses first, intersection hero second
      - resolution truth must distinguish indirect pattern power from the final
        direct survivor

    Wave-0 note:
      This function is not yet the full story-shaped intersection pipeline.
      Those richer payloads arrive in Wave-1+; this comment pins the intended
      family law before behavior changes begin.
    """

    archetype = _detect_archetype(app.technique_id)

    # -----------------------------------------------------------------
    # ARCHETYPE — INTERSECTIONS
    #
    # Native truth:
    #   - intersection witness
    #   - sweep eliminations
    #
    # Optional downstream metadata:
    #   - final canonical proof for a resulting placement/restriction
    #   - source-house confinement proof for narration/origin honesty
    # -----------------------------------------------------------------
    if archetype == "INTERSECTIONS":
        source_confinement_proof = _build_intersection_source_confinement_proof(
            app=app,
            grid_before=grid_before,
            options_all_masks=options_all_masks,
        )

        cover = app.cover_sets[0] if app.cover_sets else {}
        if not isinstance(cover, dict):
            cover = {}


        downstream = _explicit_downstream_placement_for_intersection(app, selected_placement)
        inferred_resolution = _infer_intersection_final_resolution(
            app=app,
            selected_placement=selected_placement,
            grid_before=grid_before,
            options_all_masks=options_all_masks,
        )

        confrontation_summary = {
            "target_frame_kind": str(inferred_resolution.get("kind", "") or "") if downstream is not None else "",
            "target_cell": to_json_safe(downstream[0]) if downstream is not None else {},
            "target_digit": int(downstream[1]) if downstream is not None else None,
            "ordinary_narrowing_count": 0,
            "hero_elimination_count": len(to_json_safe(cover.get("forbidden_cross_cells", []))),
            "territorial_spillover_summary": str(
                (
                    source_confinement_proof.get("trigger_bridge", {}) or {}
                ).get("cross_house_permission_change", "")
                or ""
            ),
            "honesty_line": (
                "The intersection pattern does not usually place the digit by itself; "
                "it redraws what the crossing house is allowed to keep."
            ),
        }



        if downstream is None:
            return {
                "trigger_pattern": to_json_safe(source_confinement_proof.get("trigger_pattern", {})),
                "trigger_explanation": to_json_safe(source_confinement_proof.get("trigger_explanation", {})),
                "trigger_bridge": to_json_safe(source_confinement_proof.get("trigger_bridge", {})),
                "confrontation_summary": confrontation_summary,
                "proof_payload": {
                    "kind": "INTERSECTION_CONTROL_ONLY",
                    "support": {
                        "overlap_cells": to_json_safe(source_confinement_proof.get("overlap_cells", [])),
                        "source_outside_overlap_cells": to_json_safe(source_confinement_proof.get("source_outside_overlap_cells", [])),
                        "cross_outside_overlap_cells": to_json_safe(source_confinement_proof.get("cross_outside_overlap_cells", [])),
                        "forbidden_cross_cells": to_json_safe(source_confinement_proof.get("forbidden_cross_cells", [])),
                        "outside_audit_rows": to_json_safe(source_confinement_proof.get("outside_audit_rows", [])),
                    },
                },
                "final_resolution": {
                    "kind": "INTERSECTION_ELIMINATION_ONLY",
                    "pattern_contribution_summary": str(
                        (
                            source_confinement_proof.get("trigger_bridge", {}) or {}
                        ).get("cross_house_permission_change", "")
                        or ""
                    ),
                    "territorial_takeaway": (
                        "The digit is trapped in the overlap, so the crossing house must surrender it elsewhere."
                    ),
                },
                "source_confinement_proof": source_confinement_proof,
                "technique_witness": _build_technique_witness_snapshot(app),
                "validation": {
                    "ok": True,
                    "reason": "intersection_without_downstream_placement",
                },
            }

        target_cell, target_digit = downstream

        if str(inferred_resolution.get("kind", "") or "") == "HOUSE_CANDIDATE_CELLS_FOR_DIGIT":
            final_proof = _build_canonical_house_proof(
                app=app,
                grid_before=grid_before,
                options_all_masks=options_all_masks,
                target_cell=target_cell,
                target_digit=target_digit,
                target_house=to_json_safe(inferred_resolution.get("target_house", {})),
            )
        else:
            final_proof = _build_canonical_cell_proof(
                app=app,
                grid_before=grid_before,
                options_all_masks=options_all_masks,
                target_cell=target_cell,
                target_digit=target_digit,
            )




        proof_payload = final_proof.get("proof_payload", {}) or {}
        if not isinstance(proof_payload.get("support"), dict):
            proof_payload["support"] = {}
        support = proof_payload["support"]

        house_claim = proof_payload.get("house_claim", {}) or {}

        owner_source_house = to_json_safe(cover.get("source_house", {}))
        owner_cross_house = to_json_safe(cover.get("cross_house", cover.get("target_house", {})))
        resolved_target_house = to_json_safe(
            final_proof.get("target_house", inferred_resolution.get("target_house", final_proof.get("primary_house", {})))
        )
        resolved_focus_cell = to_json_safe(
            final_proof.get("focus_cell", inferred_resolution.get("focus_cell", {}))
        )
        final_kind = str(inferred_resolution.get("kind", final_proof.get("elimination_kind", "")) or "")

        def _relation_house_from_pair(relation: str, witness_cell: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            if not isinstance(witness_cell, dict):
                return {}
            if relation == "SAME_ROW":
                r = int(witness_cell.get("r", 0) or 0)
                return {"type": "row", "index1to9": r} if r in range(1, 10) else {}
            if relation == "SAME_COL":
                c = int(witness_cell.get("c", 0) or 0)
                return {"type": "col", "index1to9": c} if c in range(1, 10) else {}
            if relation == "SAME_BOX":
                r = int(witness_cell.get("r", 0) or 0)
                c = int(witness_cell.get("c", 0) or 0)
                if r in range(1, 10) and c in range(1, 10):
                    return {"type": "box", "index1to9": ((r - 1) // 3) * 3 + ((c - 1) // 3) + 1}
            return {}

        peer_witness_pairs: List[Dict[str, Any]] = []
        peer_witness_rows: List[Dict[str, Any]] = []
        peer_blocker_rows: List[Dict[str, Any]] = []
        ordinary_group: List[Dict[str, Any]] = []

        for row in to_json_safe(support.get("witness_by_cell", [])):
            if not isinstance(row, dict):
                continue
            claimed_cell = to_json_safe(row.get("claimed_cell", {}))
            witness = to_json_safe(row.get("witness", {}))
            witness_cell = to_json_safe(witness.get("cell", {}))
            relation = str(witness.get("relation", "") or "")
            relation_house = _relation_house_from_pair(relation, witness_cell)

            pair_row = {
                "peer_cell": claimed_cell,
                "witness_cell": witness_cell,
                "relation": relation,
                "digit": int(target_digit),
                "house_scope": relation_house,
            }
            peer_witness_pairs.append(pair_row)

            peer_witness_rows.append({
                "digit": int(target_digit),
                "witness_cell": witness_cell,
                "relation": relation,
                "relation_house": relation_house,
            })

            blocker_row = {
                "digit": int(target_digit),
                "claimed_cell": claimed_cell,
                "cell": witness_cell,
                "witness_cell": witness_cell,
                "relation": relation,
                "house_scope": relation_house,
                "because": (
                    f"{cell_ref_label(claimed_cell)} is blocked for {int(target_digit)} because "
                    f"{cell_ref_label(witness_cell)} already places {int(target_digit)} in {house_label(relation_house)}."
                    if isinstance(claimed_cell, dict) and isinstance(witness_cell, dict) and relation_house else
                    ""
                ),
                "spoken_line": (
                    f"{cell_ref_label(claimed_cell)} cannot be {int(target_digit)}, because "
                    f"{house_label(relation_house)} already has {int(target_digit)} at {cell_ref_label(witness_cell)}."
                    if isinstance(claimed_cell, dict) and isinstance(witness_cell, dict) and relation_house else
                    ""
                ),
                "source_kind": "peer_witness",
            }
            peer_blocker_rows.append(blocker_row)
            ordinary_group.append(blocker_row)

        if not ordinary_group:
            for row in to_json_safe(support.get("witness_by_digit", [])):
                ordinary_group.append(to_json_safe(row))


        hero_group: List[Dict[str, Any]] = []
        for fc in to_json_safe(cover.get("forbidden_cross_cells", [])):
            hero_group.append({
                "digit": int(target_digit),
                "claimed_cell": to_json_safe(fc),
                "forbidden_cell": to_json_safe(fc),
                "cell": None,
                "source_house": owner_source_house,
                "cross_house": owner_cross_house,
                "house_scope": owner_cross_house,
                "overlap_cells": to_json_safe(cover.get("overlap_cells", cover.get("locked_cells", []))),
                "pattern_subtype": str(cover.get("pattern_subtype", app.pattern_subtype or "") or ""),
                "because": (
                    f"Because {int(target_digit)} is already trapped in the overlap, "
                    f"{cell_ref_label(fc)} cannot keep {int(target_digit)} anywhere else in {house_label(owner_cross_house)}."
                    if isinstance(fc, dict) and owner_cross_house else
                    ""
                ),
                "spoken_line": (
                    f"So {cell_ref_label(fc)}, which sits in {house_label(owner_cross_house)} but outside the owning overlap, loses {int(target_digit)}."
                    if isinstance(fc, dict) and owner_cross_house else
                    ""
                ),
                "source_kind": "intersection_technique",
            })



        support["peer_witness_pairs"] = peer_witness_pairs
        support["peer_witness_rows"] = peer_witness_rows
        support["peer_blocker_rows"] = peer_blocker_rows
        support["technique_blocker_rows"] = hero_group
        support["ordinary_witness_first_required"] = len(peer_blocker_rows) > 0
        support["technique_finishing_cut_required"] = len(hero_group) > 0
        support["actor_structure"] = (
            "First the ordinary Sudoku blockers clear the obvious rivals. Then the intersection pattern delivers the decisive territorial cut."
            if ordinary_group and hero_group else
            "The intersection pattern delivers the decisive territorial cut."
        )

        if final_kind == "HOUSE_CANDIDATE_CELLS_FOR_DIGIT":
            remaining_candidate_cells = to_json_safe(
                house_claim.get("remaining_candidate_cells", [])
            )
            support["blocker_rows"] = peer_blocker_rows + hero_group
            support["ordered_proof_ladder"] = [
                {
                    "step_kind": "ordinary_witness_ensemble",
                    "summary": (
                        f"First, the standard Sudoku blockers thin the field in {house_label(resolved_target_house)}."
                        if resolved_target_house else
                        "First, the standard Sudoku blockers thin the field."
                    ),
                    "supporting_rows": peer_blocker_rows,
                },
                {
                    "step_kind": "technique_finishing_cut",
                    "summary": (
                        "Then the intersection pattern cashes in its overlap claim and removes the last competing seat."
                        if hero_group else
                        ""
                    ),
                    "supporting_rows": hero_group,
                },
                {
                    "step_kind": "target_collapse",
                    "summary": (
                        f"Once that decisive rival falls, only {cell_ref_label(resolved_focus_cell)} remains for {int(target_digit)} in {house_label(resolved_target_house)}."
                        if isinstance(resolved_focus_cell, dict) and resolved_target_house else
                        ""
                    ),
                    "surviving_digit": int(target_digit),
                    "surviving_cell": resolved_focus_cell,
                    "remaining_candidate_cells": remaining_candidate_cells,
                },
            ]
            support["target_spotlight_line"] = (
                f"Now let’s move to the battlefield: {house_label(resolved_target_house)}. "
                f"We are looking for one thing only: where can {int(target_digit)} still sit there?"
                if resolved_target_house else
                ""
            )
            support["survivor_reveal_line"] = (
                f"And now the standoff is over. In {house_label(resolved_target_house)}, only {cell_ref_label(resolved_focus_cell)} is left for {int(target_digit)}."
                if isinstance(resolved_focus_cell, dict) and resolved_target_house else
                ""
            )
            support["house_battlefield_proof"] = {
                "proof_kind": "intersection_house_battlefield",
                "birthplace_house": owner_source_house,
                "birthplace_overlap_cells": to_json_safe(
                    cover.get("overlap_cells", cover.get("locked_cells", []))
                ),
                "battlefield_house": resolved_target_house,
                "battlefield_digit": int(target_digit),
                "battlefield_focus_cell": resolved_focus_cell,
                "ordinary_blocker_rows": peer_blocker_rows,
                "technique_finishing_rows": hero_group,
                "ordered_proof_ladder": to_json_safe(
                    support.get("ordered_proof_ladder", [])
                ),
                "battlefield_intro_line": support.get("target_spotlight_line", ""),
                "standoff_line": (
                    f"After that cleanup, the real standoff is inside {house_label(resolved_target_house)}."
                    if resolved_target_house else
                    ""
                ),
                "survivor_reveal_line": support.get("survivor_reveal_line", ""),
                "remaining_candidate_cells": remaining_candidate_cells,
            }
        else:
            support["blocker_rows"] = ordinary_group + hero_group
            support["ordered_proof_ladder"] = [
                {
                    "step_kind": "ordinary_witness_ensemble",
                    "summary": "First, the standard Sudoku blockers strip away the obvious wrong digits around the target cell.",
                    "supporting_rows": ordinary_group,
                },
                {
                    "step_kind": "technique_finishing_cut",
                    "summary": (
                        "Then the intersection pattern removes the last protected rival."
                        if hero_group else
                        ""
                    ),
                    "supporting_rows": hero_group,
                },
                {
                    "step_kind": "target_collapse",
                    "summary": (
                        f"When the dust settles, only {int(target_digit)} is left standing in {cell_ref_label(resolved_focus_cell)}."
                        if isinstance(resolved_focus_cell, dict) else
                        ""
                    ),
                    "surviving_digit": int(target_digit),
                    "surviving_cell": resolved_focus_cell,
                },
            ]
            support["target_spotlight_line"] = (
                f"Let’s bring the spotlight onto {cell_ref_label(resolved_focus_cell)} and see which digit can still survive there."
                if isinstance(resolved_focus_cell, dict) else
                ""
            )
            support["survivor_reveal_line"] = (
                f"When the dust settles, only {int(target_digit)} is left standing in {cell_ref_label(resolved_focus_cell)}."
                if isinstance(resolved_focus_cell, dict) else
                ""
            )

        confrontation_summary["ordinary_narrowing_count"] = len(ordinary_group)



        final_resolution = {
            "kind": final_kind,
            "focus_cell": resolved_focus_cell,
            "target_digit": int(target_digit),




            # Final-resolution house (downstream finishing house)
            "target_house": resolved_target_house,
            "primary_house": resolved_target_house,
            "battlefield_house": resolved_target_house,

            # Pattern-owner houses kept separately and explicitly
            "owner_houses": {
                "source_house": owner_source_house,
                "cross_house": owner_cross_house,
            },

            "pattern_contribution_summary": str(
                (
                    source_confinement_proof.get("trigger_bridge", {}) or {}
                ).get("cross_house_permission_change", "")
                or ""
            ),
            "ordinary_support_summary": (
                f"{len(ordinary_group)} ordinary witness group(s) narrow the scene before the intersection delivers the decisive exclusion."
                if ordinary_group else
                "The intersection itself carries the decisive territorial exclusion."
            ),
            "territorial_takeaway": (
                "Intersection techniques trap a digit in the overlap between a box and a line, "
                "then use that territorial control to redraw what the crossing house may keep."
            ),
            "origin_story": "INTERSECTION",
        }







        final_proof["proof_payload"]["support"] = support
        final_proof["trigger_pattern"] = to_json_safe(source_confinement_proof.get("trigger_pattern", {}))
        final_proof["trigger_explanation"] = to_json_safe(source_confinement_proof.get("trigger_explanation", {}))
        final_proof["trigger_bridge"] = to_json_safe(source_confinement_proof.get("trigger_bridge", {}))
        final_proof["confrontation_summary"] = confrontation_summary

        final_proof["proof_payload"]["intersection_control"] = {
            "ordinary_witness_group": ordinary_group,
            "hero_elimination_group": hero_group,
            "territorial_causality_chain": str(
                (
                    source_confinement_proof.get("trigger_bridge", {}) or {}
                ).get("cross_house_permission_change", "")
                or ""
            ),
            "outside_audit_rows": to_json_safe(source_confinement_proof.get("outside_audit_rows", [])),
            "birthplace_house": owner_source_house,
            "birthplace_overlap_cells": to_json_safe(
                cover.get("overlap_cells", cover.get("locked_cells", []))
            ),
            "battlefield_house": resolved_target_house,
            "battlefield_digit": int(target_digit),
            "battlefield_focus_cell": resolved_focus_cell,
            "house_battlefield_proof": to_json_safe(support.get("house_battlefield_proof", {})),
            "target_spotlight_line": support.get("target_spotlight_line", ""),
            "survivor_reveal_line": support.get("survivor_reveal_line", ""),
        }





        final_proof["final_resolution"] = final_resolution
        final_proof["source_confinement_proof"] = source_confinement_proof
        final_proof["technique_witness"] = _build_technique_witness_snapshot(app)
        final_proof["validation"] = _validate_final_canonical_proof(final_proof)
        return final_proof

    # -----------------------------------------------------------------
    # ARCHETYPE — SUBSETS
    #
    # Current intended behavior:
    #   - infer final downstream resolution kind
    #   - build either canonical cell proof or canonical house proof
    #   - attach technique witness + validation
    # -----------------------------------------------------------------
    if archetype == "SUBSETS":
        target_cell = _target_cell_ref_for_app(app, selected_placement)
        target_digit = _target_digit_for_app(app, selected_placement)

        elimination_kind = _infer_final_resolution_kind(
            app=app,
            selected_placement=selected_placement,
            target_cell=target_cell,
            target_digit=target_digit,
            grid_before=grid_before,
            options_all_masks=options_all_masks,
        )

        if elimination_kind == "CELL_CANDIDATE_DIGITS":
            if target_cell is None:
                return {
                    "technique_witness": _build_technique_witness_snapshot(app),
                    "validation": {
                        "ok": False,
                        "reason": "missing_target_cell_for_cell_candidate_digits",
                    },
                }

            final_proof = _build_canonical_cell_proof(
                app=app,
                grid_before=grid_before,
                options_all_masks=options_all_masks,
                target_cell=target_cell,
                target_digit=target_digit,
            )

        elif elimination_kind == "HOUSE_CANDIDATE_CELLS_FOR_DIGIT":
            final_proof = _build_canonical_house_proof(
                app=app,
                grid_before=grid_before,
                options_all_masks=options_all_masks,
                target_cell=target_cell,
                target_digit=target_digit,
            )

        else:
            return {
                "technique_witness": _build_technique_witness_snapshot(app),
                "validation": {
                    "ok": False,
                    "reason": f"unsupported_elimination_kind:{elimination_kind}",
                },
            }

        final_proof["technique_witness"] = _build_technique_witness_snapshot(app)
        final_proof["validation"] = _validate_final_canonical_proof(final_proof)
        return final_proof

    # -----------------------------------------------------------------
    # ARCHETYPE — FISH
    #
    # Placeholder branch for future explicit family-specific treatment.
    # For now, keep the same generic downstream proof-shape behavior.
    # -----------------------------------------------------------------
    if archetype == "FISH":
        target_cell = _target_cell_ref_for_app(app, selected_placement)
        target_digit = _target_digit_for_app(app, selected_placement)

        elimination_kind = _infer_final_resolution_kind(
            app=app,
            selected_placement=selected_placement,
            target_cell=target_cell,
            target_digit=target_digit,
            grid_before=grid_before,
            options_all_masks=options_all_masks,
        )

        if elimination_kind == "CELL_CANDIDATE_DIGITS":
            if target_cell is None:
                return {
                    "technique_witness": _build_technique_witness_snapshot(app),
                    "validation": {
                        "ok": False,
                        "reason": "missing_target_cell_for_cell_candidate_digits",
                    },
                }

            final_proof = _build_canonical_cell_proof(
                app=app,
                grid_before=grid_before,
                options_all_masks=options_all_masks,
                target_cell=target_cell,
                target_digit=target_digit,
            )

        elif elimination_kind == "HOUSE_CANDIDATE_CELLS_FOR_DIGIT":
            final_proof = _build_canonical_house_proof(
                app=app,
                grid_before=grid_before,
                options_all_masks=options_all_masks,
                target_cell=target_cell,
                target_digit=target_digit,
            )

        else:
            return {
                "technique_witness": _build_technique_witness_snapshot(app),
                "validation": {
                    "ok": False,
                    "reason": f"unsupported_elimination_kind:{elimination_kind}",
                },
            }

        final_proof["technique_witness"] = _build_technique_witness_snapshot(app)
        final_proof["validation"] = _validate_final_canonical_proof(final_proof)
        return final_proof

    # -----------------------------------------------------------------
    # ARCHETYPE — WINGS
    #
    # Placeholder branch for future explicit family-specific treatment.
    # For now, keep the same generic downstream proof-shape behavior.
    # -----------------------------------------------------------------
    if archetype == "WINGS":
        target_cell = _target_cell_ref_for_app(app, selected_placement)
        target_digit = _target_digit_for_app(app, selected_placement)

        elimination_kind = _infer_final_resolution_kind(
            app=app,
            selected_placement=selected_placement,
            target_cell=target_cell,
            target_digit=target_digit,
            grid_before=grid_before,
            options_all_masks=options_all_masks,
        )

        if elimination_kind == "CELL_CANDIDATE_DIGITS":
            if target_cell is None:
                return {
                    "technique_witness": _build_technique_witness_snapshot(app),
                    "validation": {
                        "ok": False,
                        "reason": "missing_target_cell_for_cell_candidate_digits",
                    },
                }

            final_proof = _build_canonical_cell_proof(
                app=app,
                grid_before=grid_before,
                options_all_masks=options_all_masks,
                target_cell=target_cell,
                target_digit=target_digit,
            )

        elif elimination_kind == "HOUSE_CANDIDATE_CELLS_FOR_DIGIT":
            final_proof = _build_canonical_house_proof(
                app=app,
                grid_before=grid_before,
                options_all_masks=options_all_masks,
                target_cell=target_cell,
                target_digit=target_digit,
            )

        else:
            return {
                "technique_witness": _build_technique_witness_snapshot(app),
                "validation": {
                    "ok": False,
                    "reason": f"unsupported_elimination_kind:{elimination_kind}",
                },
            }

        final_proof["technique_witness"] = _build_technique_witness_snapshot(app)
        final_proof["validation"] = _validate_final_canonical_proof(final_proof)
        return final_proof

    # -----------------------------------------------------------------
    # ARCHETYPE — CHAINS
    #
    # Placeholder branch for future explicit family-specific treatment.
    # For now, keep the same generic downstream proof-shape behavior.
    # -----------------------------------------------------------------
    if archetype == "CHAINS":
        target_cell = _target_cell_ref_for_app(app, selected_placement)
        target_digit = _target_digit_for_app(app, selected_placement)

        elimination_kind = _infer_final_resolution_kind(
            app=app,
            selected_placement=selected_placement,
            target_cell=target_cell,
            target_digit=target_digit,
            grid_before=grid_before,
            options_all_masks=options_all_masks,
        )

        if elimination_kind == "CELL_CANDIDATE_DIGITS":
            if target_cell is None:
                return {
                    "technique_witness": _build_technique_witness_snapshot(app),
                    "validation": {
                        "ok": False,
                        "reason": "missing_target_cell_for_cell_candidate_digits",
                    },
                }

            final_proof = _build_canonical_cell_proof(
                app=app,
                grid_before=grid_before,
                options_all_masks=options_all_masks,
                target_cell=target_cell,
                target_digit=target_digit,
            )

        elif elimination_kind == "HOUSE_CANDIDATE_CELLS_FOR_DIGIT":
            final_proof = _build_canonical_house_proof(
                app=app,
                grid_before=grid_before,
                options_all_masks=options_all_masks,
                target_cell=target_cell,
                target_digit=target_digit,
            )

        else:
            return {
                "technique_witness": _build_technique_witness_snapshot(app),
                "validation": {
                    "ok": False,
                    "reason": f"unsupported_elimination_kind:{elimination_kind}",
                },
            }

        final_proof["technique_witness"] = _build_technique_witness_snapshot(app)
        final_proof["validation"] = _validate_final_canonical_proof(final_proof)
        return final_proof

    # -----------------------------------------------------------------
    # ARCHETYPE — UNKNOWN
    #
    # Placeholder branch for not-yet-separated advanced families.
    # -----------------------------------------------------------------
    if archetype == "UNKNOWN":
        target_cell = _target_cell_ref_for_app(app, selected_placement)
        target_digit = _target_digit_for_app(app, selected_placement)

        elimination_kind = _infer_final_resolution_kind(
            app=app,
            selected_placement=selected_placement,
            target_cell=target_cell,
            target_digit=target_digit,
            grid_before=grid_before,
            options_all_masks=options_all_masks,
        )

        if elimination_kind == "CELL_CANDIDATE_DIGITS":
            if target_cell is None:
                return {
                    "technique_witness": _build_technique_witness_snapshot(app),
                    "validation": {
                        "ok": False,
                        "reason": "missing_target_cell_for_cell_candidate_digits",
                    },
                }

            final_proof = _build_canonical_cell_proof(
                app=app,
                grid_before=grid_before,
                options_all_masks=options_all_masks,
                target_cell=target_cell,
                target_digit=target_digit,
            )

        elif elimination_kind == "HOUSE_CANDIDATE_CELLS_FOR_DIGIT":
            final_proof = _build_canonical_house_proof(
                app=app,
                grid_before=grid_before,
                options_all_masks=options_all_masks,
                target_cell=target_cell,
                target_digit=target_digit,
            )

        else:
            return {
                "technique_witness": _build_technique_witness_snapshot(app),
                "validation": {
                    "ok": False,
                    "reason": f"unsupported_elimination_kind:{elimination_kind}",
                },
            }

        final_proof["technique_witness"] = _build_technique_witness_snapshot(app)
        final_proof["validation"] = _validate_final_canonical_proof(final_proof)
        return final_proof

    # -----------------------------------------------------------------
    # Defensive fallback
    # -----------------------------------------------------------------
    return {
        "technique_witness": _build_technique_witness_snapshot(app),
        "validation": {
            "ok": False,
            "reason": f"unhandled_archetype:{archetype}",
        },
    }






# ============================================================================
# Stage C — SolveStepV2 builder
# ============================================================================

def _support_projection_for_intersection(app: CanonicalTechniqueApplication) -> Dict[str, Any]:
    cover = app.cover_sets[0] if app.cover_sets else {}
    if not isinstance(cover, dict):
        cover = {}

    interaction_kind = cover.get("interaction_kind")
    direction_mode = cover.get("direction_mode", interaction_kind)
    digit = cover.get("digit")
    cardinality = cover.get("cardinality")
    pattern_subtype = str(cover.get("pattern_subtype", app.pattern_subtype or "") or "")

    source_house = to_json_safe(cover.get("source_house", {}))
    cross_house = to_json_safe(cover.get("cross_house", cover.get("target_house", {})))
    target_house = to_json_safe(cover.get("target_house", {}))  # backward-compatible alias
    box_house = to_json_safe(cover.get("box_house", {}))
    line_house = to_json_safe(cover.get("line_house", {}))
    line_type = cover.get("line_type")
    orientation = cover.get("orientation")

    overlap_cells = to_json_safe(
        cover.get("overlap_cells", cover.get("locked_cells", cover.get("constrained_cells", app.pattern_cells)))
    )
    source_outside_overlap_cells = to_json_safe(cover.get("source_outside_overlap_cells", []))
    forbidden_cross_cells = to_json_safe(cover.get("forbidden_cross_cells", []))

    locked_cells = to_json_safe(cover.get("locked_cells", overlap_cells))  # backward-compatible alias
    sweep_cells = to_json_safe(cover.get("sweep_cells", app.peer_cells))
    witness_cells = to_json_safe(app.witness_cells or overlap_cells)
    locked_count = cover.get("locked_count")
    if locked_count is None:
        try:
            locked_count = len(overlap_cells)
        except Exception:
            locked_count = 0

    explanation_links: List[Dict[str, Any]] = []
    for sweep_cell in sweep_cells or []:
        explanation_links.append({
            "kind": "intersection_witness",
            "digit": digit,
            "interaction_kind": interaction_kind,
            "direction_mode": direction_mode,
            "pattern_subtype": pattern_subtype,
            "overlap_cells": overlap_cells,
            "locked_cells": locked_cells,
            "sweep_cell": to_json_safe(sweep_cell),
            "source_house": source_house,
            "cross_house": cross_house,
            "target_house": target_house,
            "box_house": box_house,
            "line_house": line_house,
            "line_type": line_type,
            "orientation": orientation,
        })

    final_proof = app.final_canonical_proof or {}
    resolved_final_resolution = to_json_safe(final_proof.get("final_resolution", {}))
    resolved_target_house = to_json_safe(
        resolved_final_resolution.get("target_house", final_proof.get("primary_house", {}))
    )

    raw_source_confinement = final_proof.get("source_confinement_proof", {}) or {}
    compact_source_confinement_proof = {
        "semantic_completeness": raw_source_confinement.get("semantic_completeness", ""),
        "digit": raw_source_confinement.get("digit", digit),
        "interaction_kind": raw_source_confinement.get("interaction_kind", direction_mode),
        "pattern_subtype": raw_source_confinement.get("pattern_subtype", pattern_subtype),
        "source_house": to_json_safe(raw_source_confinement.get("source_house", source_house)),
        "cross_house": to_json_safe(raw_source_confinement.get("cross_house", cross_house)),
        "overlap_cells": to_json_safe(raw_source_confinement.get("overlap_cells", overlap_cells)),
        "surviving_cells": to_json_safe(raw_source_confinement.get("surviving_cells", overlap_cells)),
        "outside_audit_rows": to_json_safe(raw_source_confinement.get("outside_audit_rows", [])),
        "conclusion": to_json_safe(raw_source_confinement.get("conclusion", {})),
    }

    raw_proof_payload = final_proof.get("proof_payload", {}) or {}
    compact_final_canonical_proof = {
        "elimination_kind": str(final_proof.get("elimination_kind", "") or ""),
        "digit": final_proof.get("digit", digit),
        "focus_cell": to_json_safe(final_proof.get("focus_cell", {})),
        "primary_house": to_json_safe(final_proof.get("primary_house", resolved_target_house)),
        "proof_payload": {
            "house_claim": to_json_safe(raw_proof_payload.get("house_claim", {})),
            "cell_outcome": to_json_safe(raw_proof_payload.get("cell_outcome", {})),
        },
    }

    return {
        "interaction_kind": interaction_kind,
        "direction_mode": direction_mode,
        "pattern_subtype": pattern_subtype,
        "cardinality": cardinality,
        "digit": digit,

        # Owner-side houses for the pattern itself
        "source_house": source_house,
        "cross_house": cross_house,
        "owner_houses": {
            "source_house": source_house,
            "cross_house": cross_house,
        },

        # Downstream finishing house for the final resolution
        "target_house": resolved_target_house,

        "box_house": box_house,
        "line_house": line_house,
        "line_type": line_type,
        "orientation": orientation,

        # Keep only the cell lists still used by the current narrative / overlay stack
        "overlap_cells": overlap_cells,
        "locked_cells": locked_cells,  # backward-compatible alias
        "source_outside_overlap_cells": source_outside_overlap_cells,
        "forbidden_cross_cells": forbidden_cross_cells,
        "sweep_cells": sweep_cells,
        "witness_cells": witness_cells,
        "locked_count": locked_count,

        "explanation_links": explanation_links,
        "source_confinement_proof": compact_source_confinement_proof,

        # Story-shaped sections
        "trigger_pattern": to_json_safe(final_proof.get("trigger_pattern", {})),
        "trigger_explanation": to_json_safe(final_proof.get("trigger_explanation", {})),
        "trigger_bridge": to_json_safe(final_proof.get("trigger_bridge", {})),
        "confrontation_summary": to_json_safe(final_proof.get("confrontation_summary", {})),
        "final_resolution": resolved_final_resolution,

        # Explicit semantic split: owner-side pattern houses vs downstream target house
        "owner_houses": to_json_safe((final_proof.get("final_resolution", {}) or {}).get("owner_houses", {})),
        "target_house": to_json_safe(
            (final_proof.get("final_resolution", {}) or {}).get("target_house", final_proof.get("primary_house", {}))
        ),

        # Compact compatibility payloads during rollout
        "final_canonical_proof": compact_final_canonical_proof,
        "elimination_kind": str(final_proof.get("elimination_kind", "") or ""),
        "primary_house": to_json_safe(
            (final_proof.get("final_resolution", {}) or {}).get("target_house", final_proof.get("primary_house", {}))
        ),
        "focus_cell": to_json_safe(final_proof.get("focus_cell", {})),
    }


def _support_projection_from_final_canonical_proof(app: CanonicalTechniqueApplication) -> Dict[str, Any]:
    if app.technique_family == "box_line_interaction":
        return _support_projection_for_intersection(app)

    final_proof = app.final_canonical_proof or {}
    proof_payload = final_proof.get("proof_payload", {}) or {}
    support = proof_payload.get("support", {}) or {}
    elimination_kind = str(final_proof.get("elimination_kind", "") or "")
    primary_house = to_json_safe(final_proof.get("primary_house", {}))
    focus_cell = to_json_safe(final_proof.get("focus_cell", {}))

    # Restore N-2 support serialization behavior for singles only.
    # Keep subsets / intersections / all other families on the newer projection path.
    legacy_single_support = (app.technique_family == "single")

    projected: Dict[str, Any] = {
        "focus_cells": to_json_safe(app.focus_cells),
        "peer_cells": to_json_safe(app.peer_cells),
        "witness_cells": to_json_safe(app.witness_cells if legacy_single_support else support.get("witness_cells", app.witness_cells)),
        "explanation_links": [],
        "elimination_kind": elimination_kind,
        "primary_house": primary_house,
        "focus_cell": focus_cell,
        "final_canonical_proof": to_json_safe(final_proof),
        "technique_witness": to_json_safe(final_proof.get("technique_witness", {})),
    }

    # For hidden singles + naked singles, serialize support exactly like N-2:
    # trust the already-built single support graph instead of rebuilding it from
    # final_canonical_proof support rows.
    if legacy_single_support:
        if elimination_kind == "CELL_CANDIDATE_DIGITS":
            cell_outcome = proof_payload.get("cell_outcome", {}) or {}
            projected["default_candidate_digits"] = to_json_safe(cell_outcome.get("default_candidate_digits", []))
            projected["universe_candidate_digits"] = to_json_safe(cell_outcome.get("universe_candidate_digits", []))
            projected["claimed_candidate_digits"] = to_json_safe(cell_outcome.get("claimed_candidate_digits", []))
            projected["remaining_candidate_digits"] = to_json_safe(cell_outcome.get("remaining_candidate_digits", []))
        elif elimination_kind == "HOUSE_CANDIDATE_CELLS_FOR_DIGIT":
            house_claim = proof_payload.get("house_claim", {}) or {}
            projected["digit"] = house_claim.get("digit")
            projected["default_candidate_cells"] = to_json_safe(house_claim.get("default_candidate_cells", []))
            projected["claimed_candidate_cells"] = to_json_safe(house_claim.get("claimed_candidate_cells", []))
            projected["remaining_candidate_cells"] = to_json_safe(house_claim.get("remaining_candidate_cells", []))

        projected["explanation_links"] = to_json_safe(app.explanation_links)
        return projected

    if elimination_kind == "CELL_CANDIDATE_DIGITS":
        cell_outcome = proof_payload.get("cell_outcome", {}) or {}
        projected["default_candidate_digits"] = to_json_safe(cell_outcome.get("default_candidate_digits", []))
        projected["universe_candidate_digits"] = to_json_safe(cell_outcome.get("universe_candidate_digits", []))
        projected["claimed_candidate_digits"] = to_json_safe(cell_outcome.get("claimed_candidate_digits", []))
        projected["remaining_candidate_digits"] = to_json_safe(cell_outcome.get("remaining_candidate_digits", []))

        explanation_links: List[Dict[str, Any]] = []
        for row in support.get("witness_by_digit", []) or []:
            if not isinstance(row, dict):
                continue
            digit = row.get("digit")
            witness = row.get("witness", {}) or {}
            kind = str(witness.get("kind", "") or "")

            if kind == "single_cell":
                explanation_links.append({
                    "kind": "digit_witness",
                    "focus_cell": focus_cell,
                    "eliminated_digit": digit,
                    "witness_cell": to_json_safe(witness.get("cell")),
                    "witness_kind": "single_cell",
                    "placed_digit": final_proof.get("digit"),
                    "relation": witness.get("relation"),
                })
            elif kind == "subset_group":
                explanation_links.append({
                    "kind": "digit_witness",
                    "focus_cell": focus_cell,
                    "eliminated_digit": digit,
                    "witness_kind": "subset_group",
                    "subset_kind": witness.get("subset_kind"),
                    "digits": to_json_safe(witness.get("digits", [])),
                    "cells": to_json_safe(witness.get("cells", [])),
                    "house": to_json_safe(witness.get("house", {})),
                    "placed_digit": final_proof.get("digit"),
                })
            else:
                explanation_links.append({
                    "kind": "digit_witness",
                    "focus_cell": focus_cell,
                    "eliminated_digit": digit,
                    "witness_kind": "unknown",
                    "placed_digit": final_proof.get("digit"),
                })

        projected["explanation_links"] = explanation_links

    elif elimination_kind == "HOUSE_CANDIDATE_CELLS_FOR_DIGIT":
        house_claim = proof_payload.get("house_claim", {}) or {}
        projected["digit"] = house_claim.get("digit")
        projected["default_candidate_cells"] = to_json_safe(house_claim.get("default_candidate_cells", []))
        projected["claimed_candidate_cells"] = to_json_safe(house_claim.get("claimed_candidate_cells", []))
        projected["remaining_candidate_cells"] = to_json_safe(house_claim.get("remaining_candidate_cells", []))

        explanation_links: List[Dict[str, Any]] = []
        for row in support.get("witness_by_cell", []) or []:
            if not isinstance(row, dict):
                continue
            claimed_cell = to_json_safe(row.get("claimed_cell"))
            witness = row.get("witness", {}) or {}
            kind = str(witness.get("kind", "") or "")

            if kind == "single_cell":
                explanation_links.append({
                    "kind": "peer_witness",
                    "peer_cell": claimed_cell,
                    "witness_cell": to_json_safe(witness.get("cell")),
                    "digit": house_claim.get("digit"),
                    "witness_kind": "single_cell",
                    "relation": witness.get("relation"),
                })
            elif kind == "subset_group":
                explanation_links.append({
                    "kind": "peer_witness",
                    "peer_cell": claimed_cell,
                    "digit": house_claim.get("digit"),
                    "witness_kind": "subset_group",
                    "subset_kind": witness.get("subset_kind"),
                    "digits": to_json_safe(witness.get("digits", [])),
                    "cells": to_json_safe(witness.get("cells", [])),
                    "house": to_json_safe(witness.get("house", {})),
                })
            else:
                explanation_links.append({
                    "kind": "peer_witness",
                    "peer_cell": claimed_cell,
                    "digit": house_claim.get("digit"),
                    "witness_kind": "unknown",
                })

        projected["explanation_links"] = explanation_links

    else:
        projected["explanation_links"] = to_json_safe(app.explanation_links)

    return projected


def _serialize_canonical_app(
    app: CanonicalTechniqueApplication,
    *,
    application_index: Optional[int] = None,
    lead_idx: Optional[int] = None,
    causal_indices: Optional[List[int]] = None,
    lead_owner_source: Optional[str] = None,
) -> Dict[str, Any]:
    support_block = _support_projection_from_final_canonical_proof(app)
    causal_indices = [int(i) for i in (causal_indices or []) if isinstance(i, int)]

    is_lead = isinstance(application_index, int) and isinstance(lead_idx, int) and application_index == lead_idx
    is_causal = isinstance(application_index, int) and application_index in causal_indices

    return {
        "identity": {
            "application_id": app.application_id,
            "technique_id": app.technique_id,
            "family": app.technique_family,
            "kind": app.application_kind,
            "semantic_completeness": app.semantic_completeness,
        },
        "ownership": {
            "application_index": application_index,
            "is_lead": bool(is_lead),
            "is_causal": bool(is_causal),
            "lead_owner_source": lead_owner_source,
        },
        "pattern": {
            "pattern_type": app.pattern_type,
            "pattern_subtype": app.pattern_subtype,
            "roles": app.roles,
            "cells": {
                "focus_cells": app.focus_cells,
                "pattern_cells": app.pattern_cells,
                "peer_cells": app.peer_cells,
                "target_cells": app.target_cells,
                "witness_cells": app.witness_cells,
                "anchors": app.anchors,
            },
            "houses": app.houses,
            "digits": app.digits,
            "units_scanned": app.units_scanned,
            "cover_sets": app.cover_sets,
            "constraint_explanation": app.constraint_explanation,
        },
        "effects": {
            "placements": [
                {"cell": p.cell, "digit": p.digit, "source": p.source}
                for p in app.placements
            ],
            "candidate_eliminations": [
                {"cell": e.cell, "digit": e.digit, "source": e.source}
                for e in app.candidate_eliminations
            ],
            "candidate_restrictions": [
                {
                    "cell": r.cell,
                    "allowed_digits": r.allowed_digits,
                    "removed_digits": r.removed_digits,
                    "source": r.source,
                }
                for r in app.candidate_restrictions
            ],
            "cell_value_forces": [
                {"cell": p.cell, "digit": p.digit, "source": p.source}
                for p in app.cell_value_forces
            ],
        },
        "support": support_block,
        "narrative": {
            "archetype": app.narrative_archetype,
            "role": app.narrative_role,
            "summary_fact": app.summary_fact,
            "trigger_facts": app.trigger_facts,
            "confrontation_facts": app.confrontation_facts,
            "resolution_facts": app.resolution_facts,
        },
        "debug": {
            "engine_debug_summary": to_json_safe(app.engine_debug_summary),
            "engine_debug_sanitized": to_json_safe(app.engine_debug_sanitized),
        },
    }

def _application_summary_payload(
    app: CanonicalTechniqueApplication,
    *,
    application_index: int,
    lead_owner_source: Optional[str] = None,
    is_lead: bool = False,
    is_causal: bool = False,
) -> Dict[str, Any]:
    return {
        "application_index": application_index,
        "application_id": app.application_id,
        "technique_id": app.technique_id,
        "technique_real_name": app.technique_real_name,
        "family": app.technique_family,
        "archetype": app.narrative_archetype,
        "pattern_type": app.pattern_type,
        "pattern_subtype": app.pattern_subtype,
        "role": app.narrative_role,
        "is_lead": bool(is_lead),
        "is_causal": bool(is_causal),
        "lead_owner_source": lead_owner_source,
        "focus_cells": app.focus_cells,
        "pattern_cells": app.pattern_cells,
        "target_cells": app.target_cells,
        "houses": app.houses,
        "digits": app.digits,
        "summary_fact": app.summary_fact,
    }


def _validate_owner_resolution(
    *,
    applications: List[CanonicalTechniqueApplication],
    lead_idx: Optional[int],
    causal_indices: Optional[List[int]],
    lead_owner_source: Optional[str],
    engine_trace: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    causal_indices = [int(i) for i in (causal_indices or []) if isinstance(i, int)]
    problems: List[str] = []

    if lead_idx is None and applications:
        problems.append("lead_idx_missing_with_nonempty_applications")

    if isinstance(lead_idx, int) and not (0 <= lead_idx < len(applications)):
        problems.append("lead_idx_out_of_range")

    for idx in causal_indices:
        if not (0 <= idx < len(applications)):
            problems.append(f"causal_idx_out_of_range:{idx}")

    if isinstance(lead_idx, int) and causal_indices and lead_idx not in causal_indices:
        problems.append("lead_not_in_causal_indices")

    if lead_owner_source == "engine_trace" and not engine_trace:
        problems.append("lead_owner_source_engine_trace_but_trace_missing")

    if lead_owner_source == "engine_trace":
        selected_hit_source = (engine_trace or {}).get("selected_hit_source") or {}
        causal_resolution = (engine_trace or {}).get("causal_resolution") or {}
        if not selected_hit_source:
            problems.append("engine_trace_owner_missing_selected_hit_source")
        if not causal_resolution:
            problems.append("engine_trace_owner_missing_causal_resolution")

    return {
        "status": "ok" if not problems else "warning",
        "problem_count": len(problems),
        "problems": problems,
    }


def _build_step_outcome(
    *,
    selected_placement: Optional[PlacementHit],
    applications: List[CanonicalTechniqueApplication],
    lead_idx: Optional[int] = None,
    causal_indices: Optional[List[int]] = None,
    lead_owner_source: Optional[str] = None,
) -> Dict[str, Any]:
    placements = []
    eliminations = []
    restrictions = []
    for app in applications:
        placements.extend(app.placements)
        eliminations.extend(app.candidate_eliminations)
        restrictions.extend(app.candidate_restrictions)

    if selected_placement is not None:
        primary_outcome_kind = "placement"
        primary_target = {
            "kind": "placement",
            "cell": {
                "cellIndex": selected_placement.cellIndex,
                "r": selected_placement.r,
                "c": selected_placement.c,
            },
            "digit": selected_placement.digit,
            "source_shape": selected_placement.source_shape,
        }
    elif eliminations and restrictions:
        primary_outcome_kind = "mixed"
        e0 = eliminations[0]
        primary_target = {"kind": "elimination", "cell": e0.cell, "digit": e0.digit}
    elif eliminations:
        primary_outcome_kind = "elimination"
        e0 = eliminations[0]
        primary_target = {"kind": "elimination", "cell": e0.cell, "digit": e0.digit}
    elif restrictions:
        primary_outcome_kind = "restriction"
        r0 = restrictions[0]
        primary_target = {
            "kind": "restriction",
            "cell": r0.cell,
            "allowed_digits": r0.allowed_digits,
        }
    else:
        primary_outcome_kind = "noop"
        primary_target = {"kind": "noop"}



    causal_indices = [int(i) for i in (causal_indices or []) if isinstance(i, int)]

    lead_application_summary = None
    if isinstance(lead_idx, int) and 0 <= lead_idx < len(applications):
        lead_application_summary = _application_summary_payload(
            applications[lead_idx],
            application_index=lead_idx,
            lead_owner_source=lead_owner_source,
            is_lead=True,
            is_causal=lead_idx in causal_indices,
        )

    causal_application_summaries = [
        _application_summary_payload(
            applications[idx],
            application_index=idx,
            lead_owner_source=lead_owner_source,
            is_lead=(idx == lead_idx),
            is_causal=True,
        )
        for idx in causal_indices
        if 0 <= idx < len(applications)
    ]

    return {
        "primary_outcome_kind": primary_outcome_kind,
        "primary_target": primary_target,
        "lead_application_summary": lead_application_summary,
        "causal_application_summaries": causal_application_summaries,
        "secondary_effects": {
            "placements": [{"cell": p.cell, "digit": p.digit, "source": p.source} for p in placements],
            "candidate_eliminations": [{"cell": e.cell, "digit": e.digit, "source": e.source} for e in eliminations],
            "candidate_restrictions": [
                {"cell": r.cell, "allowed_digits": r.allowed_digits, "removed_digits": r.removed_digits, "source": r.source}
                for r in restrictions
            ],
        },
    }




def _build_combo_story(
    *,
    selected_placement: Optional[PlacementHit],
    applications: List[CanonicalTechniqueApplication],
    lead_idx: Optional[int] = None,
    causal_indices: Optional[List[int]] = None,
) -> Dict[str, Any]:
    combo_story: Dict[str, Any] = {"trigger": None, "ladder": [], "summary": None}
    if not applications:
        return combo_story

    resolved_lead_idx = lead_idx if isinstance(lead_idx, int) and 0 <= lead_idx < len(applications) else 0
    lead = applications[resolved_lead_idx]

    causal_indices = [
        int(i) for i in (causal_indices or [])
        if isinstance(i, int) and 0 <= i < len(applications)
    ]

    if causal_indices:
        combo_story["trigger"] = {
            "application_id": lead.application_id,
            "trigger_type": "causal_application_set" if len(causal_indices) > 1 else "technique_application",
            "technique_id": lead.technique_id,
            "archetype": lead.narrative_archetype,
            "lead_application_index": resolved_lead_idx,
            "causal_application_indices": causal_indices,
            "causal_application_ids": [applications[i].application_id for i in causal_indices],
        }
    else:
        combo_story["trigger"] = {
            "application_id": lead.application_id,
            "trigger_type": "technique_application",
            "technique_id": lead.technique_id,
            "archetype": lead.narrative_archetype,
            "lead_application_index": resolved_lead_idx,
            "causal_application_indices": [],
            "causal_application_ids": [],
        }

    for app_idx, app in enumerate(applications):
        final_conditions: List[Dict[str, Any]] = []
        for p in app.placements:
            final_conditions.append({"type": "placement", "cellIndex": p.cell["cellIndex"], "digit": p.digit})
        for e in app.candidate_eliminations:
            final_conditions.append({"type": "elimination", "cellIndex": e.cell["cellIndex"], "digit": e.digit})
        for r in app.candidate_restrictions:
            final_conditions.append({"type": "restriction", "cellIndex": r.cell["cellIndex"], "allowed_digits": r.allowed_digits})
        combo_story["ladder"].append({
            "application_id": app.application_id,
            "application_index": app_idx,
            "is_lead": app_idx == resolved_lead_idx,
            "is_causal": app_idx in causal_indices,
            "because": app.trigger_facts[0] if app.trigger_facts else f"Because of {app.technique_real_name}",
            "therefore": app.summary_fact or f"Therefore {app.technique_real_name} changes the candidate landscape.",
            "final_conditions": final_conditions,
        })

    if selected_placement is not None:
        combo_story["summary"] = (
            f"Technique pattern -> cleanup -> final resolution at "
            f"r{selected_placement.r}c{selected_placement.c}={selected_placement.digit}."
        )
    else:
        combo_story["summary"] = "Technique pattern -> cleanup/restriction summarized."

    return combo_story

# ============================================================================
# Main entry
# ============================================================================

def normalize_logged_step(
    logged_step: Tuple[Any, ...],
    step_index: int = 1,
    include_grids: bool = True,
    style: str = "full",
    *,
    engine_version: str = "unknown",
    use_cleanup_method: Optional[bool] = None,
    include_magic_technique: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Main adapter:
      Stage A — raw ingest
      Stage B — canonical semantic extraction
      Stage C — SolveStepV2 build

    Architectural rule:
      - INTERSECTIONS keeps its explicit special treatment
      - SUBSETS keeps its explicit existing treatment
      - other archetypes get explicit placeholder branches
      - shared logic exists only where all archetypes truly receive the same treatment
    """
    engine_step_trace_raw = None

    if isinstance(logged_step, (list, tuple)) and len(logged_step) == 6:
        (inst_before, inst_after, new_values, technique_used, is_cleanup_issue, cleanup_steps) = logged_step
    elif isinstance(logged_step, (list, tuple)) and len(logged_step) == 7:
        (
            inst_before,
            inst_after,
            new_values,
            technique_used,
            is_cleanup_issue,
            cleanup_steps,
            engine_step_trace_raw,
        ) = logged_step
    else:
        raise ValueError(
            f"normalize_logged_step expected logged_step tuple of len 6 or 7, got "
            f"{type(logged_step).__name__} with len="
            f"{len(logged_step) if isinstance(logged_step, (list, tuple)) else 'n/a'}"
        )

    # ---------------------------------------------------------------------
    # Stage A — raw ingest
    # ---------------------------------------------------------------------
    grid_before = instance_to_grid81(inst_before)
    grid_after = instance_to_grid81(inst_after)
    placements_all, placement_diagnostics = parse_placements_from_new_values(new_values)
    placements_selected = placements_all[:1]
    selected_placement = placements_selected[0] if placements_selected else None
    raw_cleanup = parse_cleanup_steps(cleanup_steps, grid81_before=grid_before)
    engine_step_trace = parse_engine_step_trace(engine_step_trace_raw)

    tmeta = technique_meta(technique_used or "unknown")





    # ---------------------------------------------------------------------
    # Pre / options snapshot
    # ---------------------------------------------------------------------
    snapshot_seed_options = None
    if raw_cleanup:
        for rcs in raw_cleanup:
            if rcs.options_before is not None:
                snapshot_seed_options = rcs.options_before
                break
    if snapshot_seed_options is None:
        try:
            snapshot_seed_options = determine_options_per_cell(inst_before)
        except Exception:
            snapshot_seed_options = None

    if snapshot_seed_options is not None:
        all_masks = options_snapshot_all_cells(snapshot_seed_options, grid_before)
        pre_section = {
            "candidates": {"all_cells": {"cell_candidates_mask": all_masks}},
            "digits": {
                "status": digit_status_from_grid81(grid_before),
                "candidate_cells_by_house": candidate_cells_by_house_from_masks(all_masks, grid_before),
            }
        }
    else:
        all_masks = {}
        pre_section = {
            "candidates": {"all_cells": {"cell_candidates_mask": {}}},
            "digits": {"status": digit_status_from_grid81(grid_before), "candidate_cells_by_house": {}},
        }

    # ---------------------------------------------------------------------
    # Stage B — canonical semantic extraction
    # ---------------------------------------------------------------------
    canonical_apps: List[CanonicalTechniqueApplication] = []
    relevant_cells: Dict[int, Dict[str, Any]] = {}
    houses_involved: Dict[Tuple[str, Any], Dict[str, Any]] = {}

    def _register_cell_obj(cell_obj: Optional[Dict[str, Any]]) -> None:
        if not isinstance(cell_obj, dict):
            return
        ci = cell_obj.get("cellIndex")
        if isinstance(ci, int):
            relevant_cells[ci] = cell_obj

    def _register_house_obj(h: Optional[Dict[str, Any]]) -> None:
        if not isinstance(h, dict):
            return

        h_type = str(h.get("type", ""))

        if h_type == "region":
            houses_involved[("region", h.get("regionId"))] = h
            return

        if h_type == "cell":
            cell_obj = h.get("cell")
            if isinstance(cell_obj, dict):
                ci = cell_obj.get("cellIndex")
                if isinstance(ci, int):
                    houses_involved[("cell", ci)] = h
            return

        if "index1to9" in h:
            houses_involved[(h_type, int(h["index1to9"]))] = h

    def _register_app_geometry(app: CanonicalTechniqueApplication) -> None:
        for coll in [app.focus_cells, app.pattern_cells, app.peer_cells, app.target_cells, app.witness_cells]:
            for c in coll:
                _register_cell_obj(c)
        for h in app.houses:
            _register_house_obj(h)

    for cs in raw_cleanup:
        for raw_app in cs.applications:
            app = _normalize_application_by_family(raw_app)
            canonical_apps.append(app)
            _register_app_geometry(app)

    # Singles fallback only when cleanup yielded no canonical applications
    if not canonical_apps and selected_placement is not None:
        single_app = _build_single_application(
            technique_id=technique_used or "unknown",
            placement=selected_placement,
            grid_before=grid_before,
            options_all_masks=all_masks if isinstance(all_masks, dict) else {},
        )
        canonical_apps.append(single_app)
        _register_app_geometry(single_app)

    # ---------------------------------------------------------------------
    # Stage C — lead owner selection by archetype
    # ---------------------------------------------------------------------
    technique_archetype = _detect_archetype(technique_used or "unknown")
    lead_app_idx: Optional[int] = 0 if canonical_apps else None

    def _all_app_cell_indices(app: CanonicalTechniqueApplication) -> Set[int]:
        out: Set[int] = set()

        for coll in (
            app.focus_cells,
            app.pattern_cells,
            app.peer_cells,
            app.target_cells,
            app.witness_cells,
        ):
            for c in coll:
                if isinstance(c, dict) and c.get("cellIndex") is not None:
                    out.add(int(c["cellIndex"]))

        for e in app.candidate_eliminations:
            if isinstance(e.cell, dict) and e.cell.get("cellIndex") is not None:
                out.add(int(e.cell["cellIndex"]))

        for p in (app.placements + app.cell_value_forces):
            if isinstance(p.cell, dict) and p.cell.get("cellIndex") is not None:
                out.add(int(p.cell["cellIndex"]))

        return out

    def _base_owner_score(app: CanonicalTechniqueApplication) -> int:
        score = 0

        step_tid = str(technique_used or "").strip().lower()
        selected_ci = int(selected_placement.cellIndex) if selected_placement is not None else None
        selected_digit = int(selected_placement.digit) if selected_placement is not None else None

        if step_tid and str(app.technique_id or "").strip().lower() == step_tid:
            score += 100

        if selected_ci is not None and selected_ci in _all_app_cell_indices(app):
            score += 50

        if selected_digit is not None and selected_digit in [int(d) for d in app.digits if isinstance(d, int)]:
            score += 10

        return score







    def _choose_best_idx(
        *,
        family_filter: Optional[str],
        archetype_bonus: Optional[str],
    ) -> Optional[int]:
        if not canonical_apps:
            return None

        best_score: Optional[int] = None
        best_idx: Optional[int] = None

        for idx, app in enumerate(canonical_apps):
            if family_filter is not None and app.technique_family != family_filter:
                continue

            score = _base_owner_score(app)

            if archetype_bonus is not None and app.narrative_archetype == archetype_bonus:
                score += 25

            if best_score is None or score > best_score:
                best_score = score
                best_idx = idx

        return best_idx if best_idx is not None else (0 if canonical_apps else None)

    def _map_engine_trace_causal_indices_to_canonical_indices() -> Tuple[Optional[int], List[int], Dict[str, Any]]:
        """
        Map engine trace causal application indices (which are relative to the
        winning cleanup batch's effective application list) onto the flattened
        canonical_apps list built from raw_cleanup.
        """
        debug: Dict[str, Any] = {
            "present": bool(engine_step_trace),
            "matched_batch": None,
            "raw_causal_indices": [],
            "mapped_canonical_indices": [],
            "reason": "trace_absent",
        }

        if not engine_step_trace or not raw_cleanup or not canonical_apps:
            return None, [], debug

        selected_hit_source = engine_step_trace.get("selected_hit_source") or {}
        causal_resolution = engine_step_trace.get("causal_resolution") or {}
        raw_causal_indices = causal_resolution.get("causal_application_indices") or []
        raw_causal_indices = [int(i) for i in raw_causal_indices if isinstance(i, int)]

        debug["raw_causal_indices"] = raw_causal_indices

        if not raw_causal_indices:
            debug["reason"] = "trace_present_but_no_causal_indices"
            return None, [], debug

        target_latest = str(
            selected_hit_source.get("technique_latest")
            or selected_hit_source.get("technique_id")
            or ""
        )
        target_cleanup = str(
            selected_hit_source.get("technique_cleanup")
            or selected_hit_source.get("technique_id")
            or ""
        )
        target_iteration = int(selected_hit_source.get("iteration", 0))

        running_offset = 0
        for batch_idx, cs in enumerate(raw_cleanup):
            batch_count = len(cs.applications)

            batch_match = (
                str(cs.technique_latest) == target_latest and
                str(cs.technique_cleanup) == target_cleanup and
                int(cs.iteration) == target_iteration
            )

            if batch_match:
                mapped: List[int] = []
                for rel_idx in raw_causal_indices:
                    if 0 <= rel_idx < batch_count:
                        mapped.append(running_offset + rel_idx)

                debug["matched_batch"] = {
                    "batch_index": batch_idx,
                    "technique_latest": cs.technique_latest,
                    "technique_cleanup": cs.technique_cleanup,
                    "iteration": cs.iteration,
                    "running_offset": running_offset,
                    "batch_count": batch_count,
                }
                debug["mapped_canonical_indices"] = mapped

                if mapped:
                    debug["reason"] = "mapped_from_engine_trace"
                    return mapped[0], mapped, debug

                debug["reason"] = "matched_batch_but_indices_out_of_range"
                return None, [], debug

            running_offset += batch_count

        debug["reason"] = "no_matching_batch_for_engine_trace"
        return None, [], debug




        # ---------------------------------------------------------------------
        # ARCHETYPE — INTERSECTIONS
        #
        # Existing explicit treatment:
        # choose which intersection application owns the downstream placement.
        #
        # Wave-0 family note:
        # intersections are the box-line territorial-control family
        # (claiming pair/triple + pointing pair/triple). Later waves will make the
        # owner emit richer story-shaped truth, including:
        #   - source house
        #   - cross house
        #   - overlap cells
        #   - explicit source-house outside-overlap audit
        #   - territorial permission change
        #   - downstream target contribution
        # ---------------------------------------------------------------------

    heuristic_lead_app_idx: Optional[int] = None

    if technique_archetype == "INTERSECTIONS":
        heuristic_lead_app_idx = _choose_best_idx(
            family_filter="box_line_interaction",
            archetype_bonus="INTERSECTIONS",
        )

    # ---------------------------------------------------------------------
    # ARCHETYPE — SUBSETS
    #
    # Explicit branch for subsets.
    # Structure-only refactor: keep same owner-selection style as current
    # generic flow, but make the branch visible and independently editable.
    # ---------------------------------------------------------------------
    elif technique_archetype == "SUBSETS":
        heuristic_lead_app_idx = _choose_best_idx(
            family_filter="subset",
            archetype_bonus="SUBSETS",
        )

    # ---------------------------------------------------------------------
    # ARCHETYPE — FISH
    #
    # Placeholder branch for future family-specific treatment.
    # For now, explicit but conservative.
    # ---------------------------------------------------------------------
    elif technique_archetype == "FISH":
        heuristic_lead_app_idx = _choose_best_idx(
            family_filter="fish",
            archetype_bonus="FISH",
        )

    # ---------------------------------------------------------------------
    # ARCHETYPE — WINGS
    #
    # Placeholder branch for future family-specific treatment.
    # ---------------------------------------------------------------------
    elif technique_archetype == "WINGS":
        heuristic_lead_app_idx = _choose_best_idx(
            family_filter="wing",
            archetype_bonus="WINGS",
        )

    # ---------------------------------------------------------------------
    # ARCHETYPE — CHAINS
    #
    # Placeholder branch for future family-specific treatment.
    # ---------------------------------------------------------------------
    elif technique_archetype == "CHAINS":
        heuristic_lead_app_idx = _choose_best_idx(
            family_filter="chain",
            archetype_bonus="CHAINS",
        )

    # ---------------------------------------------------------------------
    # ARCHETYPE — UNKNOWN
    #
    # Placeholder branch for not-yet-separated advanced families.
    # ---------------------------------------------------------------------
    elif technique_archetype == "UNKNOWN":
        heuristic_lead_app_idx = 0 if canonical_apps else None

    # ---------------------------------------------------------------------
    # Defensive fallback
    # ---------------------------------------------------------------------
    else:
        heuristic_lead_app_idx = 0 if canonical_apps else None

    engine_trace_lead_app_idx, engine_trace_causal_app_indices, engine_trace_owner_debug = (
        _map_engine_trace_causal_indices_to_canonical_indices()
    )

    if engine_trace_lead_app_idx is not None:
        lead_app_idx = engine_trace_lead_app_idx
        lead_owner_source = "engine_trace"
        causal_app_indices = list(engine_trace_causal_app_indices)
    else:
        lead_app_idx = heuristic_lead_app_idx
        lead_owner_source = "heuristic_fallback"
        causal_app_indices = []

    # Harden the owner picture so downstream consumers always get a coherent shape.
    causal_app_indices = sorted(set(
        idx for idx in causal_app_indices
        if isinstance(idx, int) and 0 <= idx < len(canonical_apps)
    ))

    if isinstance(lead_app_idx, int) and 0 <= lead_app_idx < len(canonical_apps):
        if lead_app_idx not in causal_app_indices:
            causal_app_indices = [lead_app_idx] + causal_app_indices
    else:
        lead_app_idx = 0 if canonical_apps else None
        if isinstance(lead_app_idx, int):
            lead_owner_source = f"{lead_owner_source}_normalized"
            causal_app_indices = [lead_app_idx] if lead_app_idx not in causal_app_indices else causal_app_indices

    if lead_app_idx is not None and 0 <= lead_app_idx < len(canonical_apps):
        canonical_apps[lead_app_idx].narrative_role = "trigger"

    for idx, app in enumerate(canonical_apps):
        if idx in causal_app_indices and idx != lead_app_idx:
            app.narrative_role = "causal_support"

    owner_validation = _validate_owner_resolution(
        applications=canonical_apps,
        lead_idx=lead_app_idx,
        causal_indices=causal_app_indices,
        lead_owner_source=lead_owner_source,
        engine_trace=engine_step_trace,
    )




    # ---------------------------------------------------------------------
    # Shared post-processing for ALL canonical applications
    #
    # This is truly shared:
    # every archetype attaches the selected downstream resolution to the
    # chosen lead app in the same way.
    # ---------------------------------------------------------------------
    if selected_placement is not None and canonical_apps and lead_app_idx is not None:
        follow_ref = cell_ref_from_index(selected_placement.cellIndex)
        lead = canonical_apps[lead_app_idx]

        lead.placements.append(CanonicalEffectPlacement(
            cell=follow_ref,
            digit=selected_placement.digit,
            source="derived_resolution",
        ))
        lead.cell_value_forces.append(CanonicalEffectPlacement(
            cell=follow_ref,
            digit=selected_placement.digit,
            source="derived_resolution",
        ))

        if follow_ref["cellIndex"] not in {c["cellIndex"] for c in lead.target_cells}:
            lead.target_cells.append(follow_ref)

        _register_cell_obj(follow_ref)

        for h in houses_for_cell(selected_placement.cellIndex):
            _register_house_obj(h)

        lead.resolution_facts.append(
            f"Derived resolution places {selected_placement.digit} at r{selected_placement.r}c{selected_placement.c}."
        )

        if lead.summary_fact:
            lead.summary_fact = (
                f"{lead.summary_fact} Derived resolution: r{selected_placement.r}c{selected_placement.c}={selected_placement.digit}."
            )







    # ---------------------------------------------------------------------
    # Final canonical proof synthesis — LEAD application only
    #
    # The conversation packet now carries only the owner application.
    # Non-lead applications still keep their pattern/effects/debug summaries,
    # but they no longer receive a full synthesized canonical proof payload.
    # ---------------------------------------------------------------------
    for idx, app in enumerate(canonical_apps):
        should_build_full_proof = (
            isinstance(lead_app_idx, int) and idx == lead_app_idx
        ) or (
            lead_app_idx is None and idx == 0
        )

        if should_build_full_proof:
            if selected_placement is None:
                app_selected = None
            elif isinstance(lead_app_idx, int) and idx == lead_app_idx:
                app_selected = selected_placement
            else:
                app_selected = None

            app.final_canonical_proof = _synthesize_final_canonical_proof(
                app=app,
                grid_before=grid_before,
                options_all_masks=all_masks if isinstance(all_masks, dict) else {},
                selected_placement=app_selected,
            )
        else:
            app.final_canonical_proof = {}

    # ---------------------------------------------------------------------
    # Candidate snapshots on relevant cells
    # ---------------------------------------------------------------------
    rel_cells_sorted = sorted(relevant_cells.keys())
    snapshot_before: Dict[str, int] = {}
    if snapshot_seed_options is not None:
        for ci in rel_cells_sorted:
            i1, i2 = ci // 9, ci % 9
            try:
                snapshot_before[str(ci)] = candidates_set_to_mask(set(snapshot_seed_options[i1][i2]))
            except Exception:
                snapshot_before[str(ci)] = 0
    snapshot_after = dict(snapshot_before)

    # Apply eliminations/restrictions/placements to after snapshot
    for app in canonical_apps:
        for e in app.candidate_eliminations:
            key = str(e.cell["cellIndex"])
            if key in snapshot_after:
                snapshot_after[key] = snapshot_after[key] & ~(1 << (e.digit - 1))
        for r in app.candidate_restrictions:
            key = str(r.cell["cellIndex"])
            if key in snapshot_after:
                mask = 0
                for d in r.allowed_digits:
                    mask |= (1 << (d - 1))
                snapshot_after[key] = mask
        for p in app.placements:
            snapshot_after[str(p.cell["cellIndex"])] = 0

    # ---------------------------------------------------------------------
    # Step outcome and target
    # ---------------------------------------------------------------------
    step_outcome = _build_step_outcome(
        selected_placement=selected_placement,
        applications=canonical_apps,
        lead_idx=lead_app_idx,
        causal_indices=causal_app_indices,
        lead_owner_source=lead_owner_source,
    )

    primary_target = step_outcome["primary_target"]
    if primary_target.get("kind") == "placement":
        target = {
            "kind": "placement",
            "cell": primary_target["cell"],
            "digit": primary_target["digit"],
        }
    elif primary_target.get("kind") == "elimination":
        target = {
            "kind": "elimination",
            "cell": primary_target["cell"],
            "digit": primary_target["digit"],
        }
    elif primary_target.get("kind") == "restriction":
        target = {
            "kind": "restriction",
            "cell": primary_target["cell"],
            "allowed_digits": primary_target["allowed_digits"],
        }
    else:
        target = {"kind": "noop"}

    # ---------------------------------------------------------------------
    # Proof sections
    # ---------------------------------------------------------------------
    placements_out = []
    if selected_placement is not None:
        placements_out.append({
            "cellIndex": selected_placement.cellIndex,
            "r": selected_placement.r,
            "c": selected_placement.c,
            "digit": selected_placement.digit,
            "source": "final_hit_selected",
            "dimension": selected_placement.dimension,
            "idx_dim_0based": selected_placement.idx_dim_0based,
            "source_shape": selected_placement.source_shape,
        })

    all_possible_placements_out = [{
        "cellIndex": p.cellIndex,
        "r": p.r,
        "c": p.c,
        "digit": p.digit,
        "source": "final_hit_parallel",
        "dimension": p.dimension,
        "idx_dim_0based": p.idx_dim_0based,
        "source_shape": p.source_shape,
    } for p in placements_all]

    eliminations_out: List[Dict[str, Any]] = []
    for app in canonical_apps:
        for e in app.candidate_eliminations:
            eliminations_out.append({
                "cellIndex": e.cell["cellIndex"],
                "r": e.cell["r"],
                "c": e.cell["c"],
                "digit": e.digit,
                "reason_code": app.technique_id,
            })

    houses_dedup = list(houses_involved.values())




    canonical_application_summaries_v2 = [
        _application_summary_payload(
            app,
            application_index=idx,
            lead_owner_source=lead_owner_source,
            is_lead=(isinstance(lead_app_idx, int) and idx == lead_app_idx),
            is_causal=(idx in causal_app_indices),
        )
        for idx, app in enumerate(canonical_apps)
    ]

    lead_application_v2 = (
        _serialize_canonical_app(
            canonical_apps[lead_app_idx],
            application_index=lead_app_idx,
            lead_idx=lead_app_idx,
            causal_indices=causal_app_indices,
            lead_owner_source=lead_owner_source,
        )
        if isinstance(lead_app_idx, int) and 0 <= lead_app_idx < len(canonical_apps)
        else None
    )

    applications_v2 = [lead_application_v2] if isinstance(lead_application_v2, dict) else []

    causal_application_summaries_v2 = [
        canonical_application_summaries_v2[idx]
        for idx in sorted(set(causal_app_indices))
        if 0 <= idx < len(canonical_application_summaries_v2)
    ]





    causal_applications_v2 = causal_application_summaries_v2
    applications_for_proof = applications_v2

    combo_story = _build_combo_story(
        selected_placement=selected_placement,
        applications=canonical_apps,
        lead_idx=lead_app_idx,
        causal_indices=causal_app_indices,
    )
    if engine_step_trace:
        combo_story["engine_trace_summary"] = {
            "selected_hit": engine_step_trace.get("selected_hit"),
            "selected_hit_source": engine_step_trace.get("selected_hit_source"),
            "causal_resolution": engine_step_trace.get("causal_resolution"),
            "causal_applications": engine_step_trace.get("causal_applications", []),
        }

    combo_story["owner_summary"] = {
        "lead_owner_source": lead_owner_source,
        "lead_app_idx": lead_app_idx,
        "heuristic_lead_app_idx": heuristic_lead_app_idx,
        "engine_trace_lead_app_idx": engine_trace_lead_app_idx,
        "causal_app_indices": causal_app_indices,
        "engine_trace_owner_debug": engine_trace_owner_debug,
        "owner_validation": owner_validation,
        "lead_application_summary": (
            canonical_application_summaries_v2[lead_app_idx]
            if isinstance(lead_app_idx, int) and 0 <= lead_app_idx < len(canonical_application_summaries_v2)
            else None
        ),
        "causal_application_summaries": causal_application_summaries_v2,
    }






    # ---------------------------------------------------------------------
    # IDs / engine / presentation
    # ---------------------------------------------------------------------
    grid_hash_before = sha12(grid_before)
    grid_hash_after = sha12(grid_after)
    step_id = sha12(grid_before + "|" + (technique_used or "unknown") + "|" + grid_after)

    engine_flags = {
        "name": "generator.algo_human",
        "engine_version": engine_version,
        "use_cleanup_method": True if use_cleanup_method is None else bool(use_cleanup_method),
        "include_magic_technique": False if include_magic_technique is None else bool(include_magic_technique),
    }

    # ---------------------------------------------------------------------
    # Engine-native debug — sanitized only
    # ---------------------------------------------------------------------
    cleanup_steps_debug = []
    for cs in raw_cleanup:
        cleanup_steps_debug.append({
            "technique_latest": to_json_safe(cs.technique_latest),
            "technique_cleanup": to_json_safe(cs.technique_cleanup),
            "iteration": to_json_safe(cs.iteration),
            "details_updated": [
                {
                    "technique_id": app.technique_id,
                    "name_application": app.name_application,
                    "engine_debug_summary": to_json_safe(app.engine_debug_summary),
                    "engine_debug_sanitized": to_json_safe(app.engine_debug_sanitized),
                }
                for app in cs.applications
            ],
        })

    # ---------------------------------------------------------------------
    # Final SolveStepV2
    # ---------------------------------------------------------------------
    out: Dict[str, Any] = {
        "schema_version": "solve_step_v2",
        "engine": engine_flags,
        "pre": pre_section,
        "presentation": {
            "spoiler_policy": "hide_digit_until_reveal",
            "default_style": "mini" if style == "mini" else "full",
            "has_reveal": bool(selected_placement is not None),
            "has_hint_ladder": False,
        },
        "ids": {
            "grid_hash12_before": grid_hash_before,
            "grid_hash12_after": grid_hash_after,
            "step_id": step_id,
            "step_index": step_index,
        },
        "technique": tmeta,
        "target": target,
        "step_outcome": step_outcome,
        "engine_trace_summary": {
            "present": bool(engine_step_trace),
            "selected_hit": engine_step_trace.get("selected_hit") if engine_step_trace else None,
            "selected_hit_source": engine_step_trace.get("selected_hit_source") if engine_step_trace else None,
            "causal_resolution": engine_step_trace.get("causal_resolution") if engine_step_trace else None,
            "causal_applications": engine_step_trace.get("causal_applications", []) if engine_step_trace else [],
        },


        "owner_summary": {
            "lead_owner_source": lead_owner_source,
            "lead_app_idx": lead_app_idx,
            "heuristic_lead_app_idx": heuristic_lead_app_idx,
            "engine_trace_lead_app_idx": engine_trace_lead_app_idx,
            "causal_app_indices": causal_app_indices,
            "engine_trace_owner_debug": engine_trace_owner_debug,
            "owner_validation": owner_validation,
        },



        "lead_application_summary": step_outcome.get("lead_application_summary"),
        "causal_application_summaries": step_outcome.get("causal_application_summaries", []),
        "validation": {
            "owner_validation": owner_validation,
        },
        "proof": {




            "placements": placements_out,
            "all_possible_placements": all_possible_placements_out,
            "eliminations": eliminations_out,
            "houses_involved": houses_dedup,
            "candidates": {
                "snapshot_before": snapshot_before,
                "snapshot_after": snapshot_after,
                "relevant_cells": rel_cells_sorted,
            },


            "applications": applications_v2,

            "lead_application": lead_application_v2,
            "causal_applications": causal_application_summaries_v2,
            "causal_application_summaries": causal_application_summaries_v2,

            "combo_story": combo_story,
        },
        "hint_ladder": [],
        "overlay_frames": [],
        "teaching": {
            "recognition": [],
            "application": [],
            "pitfalls": [],
            "glossary_keys": [],
        },
        "engine_native": {
            "log_technique_used": str(technique_used),
            "log_new_values_found_summary": debug_shape_summary(new_values),
            "log_new_values_found_sanitized": to_json_safe(new_values),
            "log_new_values_parse_diagnostics": to_json_safe(placement_diagnostics),
            "is_cleanup_issue": bool(is_cleanup_issue),
            "cleanup_steps": cleanup_steps_debug,
            "engine_step_trace_present": bool(engine_step_trace),
            "engine_step_trace": engine_step_trace,

            "owner_resolution": {
                "lead_owner_source": lead_owner_source,
                "lead_app_idx": lead_app_idx,
                "heuristic_lead_app_idx": heuristic_lead_app_idx,
                "engine_trace_lead_app_idx": engine_trace_lead_app_idx,
                "causal_app_indices": causal_app_indices,
                "engine_trace_owner_debug": engine_trace_owner_debug,
                "owner_validation": owner_validation,
            },
            "canonical_application_summaries_debug": canonical_application_summaries_v2,


            "canonical_application_summaries_debug": canonical_application_summaries_v2,



        },
    }

    if include_grids:
        out["grids"] = {
            "grid81_before": grid_before,
            "grid81_after": grid_after,
        }

    return out