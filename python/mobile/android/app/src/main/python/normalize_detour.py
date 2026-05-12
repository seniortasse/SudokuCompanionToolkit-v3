from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from generator.model import Instance, EMPTY_CHAR
from generator.algo_human import solve_using_human_techniques, TECHNIQUES
from generator.techniques.options import determine_options_per_cell

from normalize_step import (
    normalize_logged_step,
    options_snapshot_all_cells,
    candidate_cells_by_house_from_masks,
    _candidate_digits_before_for_cell_from_masks,
    rc_to_cell_index,
    cell_index_to_rc,
    box_index_1to9,
    _find_placed_digit_in_house,
)

DIGITS = "123456789"


# ============================================================================
# Shared helpers
# ============================================================================

def _grid81_to_instance(grid81: str) -> Instance:
    g = (grid81 or "").strip()
    if len(g) != 81:
        raise ValueError(f"grid81 must be length 81, got {len(g)}")

    rows = []
    for r in range(9):
        row = []
        for c in range(9):
            ch = g[r * 9 + c]
            if ch in ("0", "."):
                row.append(EMPTY_CHAR)
            elif ch in DIGITS:
                row.append(ch)
            else:
                row.append(EMPTY_CHAR)
        rows.append(row)

    inst = Instance(rows, is_chars=True)
    inst.chars = set(DIGITS)
    return inst


def _instance_to_grid81(inst: Instance) -> str:
    out = []
    for r in range(9):
        for c in range(9):
            ch = inst[r][c]
            out.append("." if ch == EMPTY_CHAR else str(ch))
    return "".join(out)


def _run_next_step_obj(grid81: str) -> Dict[str, Any]:
    inst = _grid81_to_instance(grid81)
    solved_inst, (_counts, logs) = solve_using_human_techniques(
        inst,
        use_techniques=TECHNIQUES,
        use_cleanup_method=True,
        include_magic_technique=False,
        magic_solution=None,
        max_number_iterations=1,
        show_logs=False
    )

    before = grid81
    after = _instance_to_grid81(solved_inst)

    if before == after:
        if "." not in before:
            return {"ok": True, "status": "solved"}
        return {"ok": True, "status": "stuck"}

    steps = (logs or {}).get("steps") or []
    if not steps:
        return {
            "ok": False,
            "status": "error",
            "error": {"code": "missing_step_log", "msg": "Engine progressed but produced no step log."}
        }

    step_v2 = normalize_logged_step(
        logged_step=steps[0],
        step_index=1,
        include_grids=True,
        style="full"
    )
    return {"ok": True, "status": "ok", "step": step_v2}


def _parse_cell_ref(cell: Any) -> Optional[int]:
    if cell is None:
        return None
    if isinstance(cell, int):
        return int(cell) if 0 <= int(cell) <= 80 else None

    s = str(cell).strip().lower()
    if not s:
        return None

    if s.isdigit():
        ci = int(s)
        return ci if 0 <= ci <= 80 else None

    if s.startswith("r") and "c" in s:
        try:
            r_part, c_part = s[1:].split("c", 1)
            r = int(r_part)
            c = int(c_part)
            if 1 <= r <= 9 and 1 <= c <= 9:
                return rc_to_cell_index(r, c)
        except Exception:
            return None

    return None


def _cell_ref_string(cell_index: int) -> str:
    r, c = cell_index_to_rc(int(cell_index))
    return f"r{r}c{c}"


def _parse_digit_list(raw: Any) -> List[int]:
    if raw is None:
        return []
    if isinstance(raw, list):
        out: List[int] = []
        for v in raw:
            try:
                d = int(v)
            except Exception:
                continue
            if d in range(1, 10) and d not in out:
                out.append(d)
        return out

    s = str(raw).strip()
    if not s:
        return []

    out: List[int] = []
    token = ""
    for ch in s:
        if ch.isdigit():
            token += ch
        else:
            if token:
                try:
                    d = int(token)
                    if d in range(1, 10) and d not in out:
                        out.append(d)
                except Exception:
                    pass
                token = ""
    if token:
        try:
            d = int(token)
            if d in range(1, 10) and d not in out:
                out.append(d)
        except Exception:
            pass
    return out


def _compute_options_masks(grid81: str) -> Dict[str, int]:
    inst = _grid81_to_instance(grid81)
    options_grid = determine_options_per_cell(inst)
    return options_snapshot_all_cells(options_grid, grid81)


def _candidate_payload_for_cell(grid81: str, options_all_masks: Dict[str, int], cell_index: int) -> Dict[str, Any]:
    digits = _candidate_digits_before_for_cell_from_masks(options_all_masks, int(cell_index))
    r, c = cell_index_to_rc(int(cell_index))
    b = box_index_1to9(r, c)
    mask = int(options_all_masks.get(str(int(cell_index)), 0))
    return {
        "cell": _cell_ref_string(cell_index),
        "cell_index": int(cell_index),
        "row": r,
        "col": c,
        "box": b,
        "mask": mask,
        "digits": digits,
    }


def _digit_blockers_for_cell(grid81: str, cell_index: int, digit: int) -> Dict[str, Any]:
    r, c = cell_index_to_rc(int(cell_index))
    b = box_index_1to9(r, c)
    row_w = _find_placed_digit_in_house(grid81, "row", r, digit)
    col_w = _find_placed_digit_in_house(grid81, "col", c, digit)
    box_w = _find_placed_digit_in_house(grid81, "box", b, digit)

    def witness_payload(ci: Optional[int], relation: str) -> Optional[Dict[str, Any]]:
        if ci is None:
            return None
        return {
            "cell": _cell_ref_string(ci),
            "cell_index": int(ci),
            "relation": relation
        }

    blockers = {
        "row": witness_payload(row_w, "SAME_ROW"),
        "col": witness_payload(col_w, "SAME_COL"),
        "box": witness_payload(box_w, "SAME_BOX"),
    }
    blockers_present = [k for k, v in blockers.items() if v is not None]

    return {
        "digit": int(digit),
        "blocked": len(blockers_present) > 0,
        "blockers": blockers,
        "blocking_houses": blockers_present
    }


def _parse_house_ref(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    h_type = str(raw.get("type") or "").strip().lower()
    try:
        idx1 = int(raw.get("index1to9"))
    except Exception:
        return None
    if h_type not in {"row", "col", "box"}:
        return None
    if idx1 not in range(1, 10):
        return None
    return {"type": h_type, "index1to9": idx1}


def _build_anchor(query: Dict[str, Any], step_obj: Dict[str, Any]) -> Dict[str, Any]:
    step = step_obj.get("step") or {}
    technique_meta = (step.get("technique_meta") or {})
    technique = (step.get("technique") or {})
    current_target = {
        "cell": query.get("anchor_target_cell") or step.get("target_cell"),
        "digit": query.get("anchor_target_digit") or step.get("target_digit"),
    }
    return {
        "step_id": query.get("anchor_step_id"),
        "story_stage": query.get("anchor_stage") or "UNKNOWN",
        "canonical_position_kind": query.get("anchor_stage") or "UNKNOWN",
        "technique": {
            "id": technique_meta.get("id") or query.get("anchor_technique_id"),
            "name": technique.get("name") or step.get("technique"),
            "real_name": technique.get("real_name"),
            "family": technique.get("family"),
            "archetype": technique.get("archetype"),
        },
        "paused_route_checkpoint_id": query.get("paused_route_checkpoint_id"),
        "current_target": current_target
    }


def _build_route_context(query: Dict[str, Any], default_return_line: Optional[str] = None) -> Dict[str, Any]:
    result = {
        "is_route_aligned": True,
        "route_relation": "ON_CURRENT_ROUTE",
        "may_switch_route": False,
        "recommended_handover_mode": "RETURN_TO_CURRENT_MOVE",
        "may_offer_return_now": True,
        "may_offer_one_followup": True,
        "preferred_return_style": "GENTLE",
        "preferred_followup_style": "BOUNDED",
    }
    if default_return_line:
        result["spoken_return_line"] = default_return_line
    return result


def _default_overlay_mode_for_story_kind(overlay_story_kind: str) -> str:
    kind = (overlay_story_kind or "").strip().upper()
    if kind in (
        "LOCAL_CONTRADICTION_SPOTLIGHT",
        "LOCAL_PERMISSIBILITY_SCAN",
        "SURVIVOR_LADDER",
        "CONTRAST_DUEL",
        "PATTERN_LEGITIMACY",
    ):
        return "REPLACE"
    return "AUGMENT"


def _build_overlay_context(
    focus_cells: List[str],
    focus_houses: List[str],
    reason_for_focus: str,
    overlay_mode: Optional[str] = None,
    overlay_story_kind: str = "LOCAL_PROOF_SPOTLIGHT",
    secondary_focus_cells: Optional[List[str]] = None,
    deemphasize_cells: Optional[List[str]] = None,
    highlight_roles: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    resolved_overlay_mode = (
        overlay_mode
        if overlay_mode is not None and str(overlay_mode).strip()
        else _default_overlay_mode_for_story_kind(overlay_story_kind)
    )
    return {
        "focus_cells": focus_cells,
        "focus_houses": focus_houses,
        "secondary_focus_cells": secondary_focus_cells or [],
        "deemphasize_cells": deemphasize_cells or [],
        "reason_for_focus": reason_for_focus,
        "overlay_mode": resolved_overlay_mode,
        "overlay_story_kind": overlay_story_kind,
        "highlight_roles": highlight_roles or {},
    }


def _build_support(has_primary_truth: bool, has_bounded_evidence: bool) -> Dict[str, Any]:
    confidence = "HIGH" if has_primary_truth and has_bounded_evidence else "MEDIUM" if has_primary_truth else "LOW"
    return {
        "completeness": {
            "has_anchor": True,
            "has_scope": True,
            "has_primary_truth": has_primary_truth,
            "has_bounded_evidence": has_bounded_evidence,
            "has_route_context": True
        },
        "confidence": confidence,
        "notes": []
    }


def _proof_ladder_row(
    step_kind: str,
    actor_type: str,
    actor_ref: str,
    target_ref: str,
    supported_claim: str,
    spoken_line_seed: str,
    rival_digit: Optional[int] = None,
    fact_basis: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "step_kind": step_kind,
        "actor_type": actor_type,
        "actor_ref": actor_ref,
        "target_ref": target_ref,
        "rival_digit": rival_digit,
        "supported_claim": supported_claim,
        "fact_basis": fact_basis or {},
        "spoken_line_seed": spoken_line_seed,
    }


def _house_type_for_relation(relation: str) -> str:
    rel = str(relation or "").strip().upper()
    if rel == "SAME_ROW":
        return "ROW"
    if rel == "SAME_COL":
        return "COLUMN"
    if rel == "SAME_BOX":
        return "BOX"
    return "CELL"


def _house_ref_for_relation(cell_ref: Optional[str], relation: str) -> str:
    if not cell_ref:
        return ""
    ci = _parse_cell_ref(cell_ref)
    if ci is None:
        return cell_ref
    r, c = cell_index_to_rc(int(ci))
    b = box_index_1to9(r, c)
    rel = str(relation or "").strip().upper()
    if rel == "SAME_ROW":
        return f"row{r}"
    if rel == "SAME_COL":
        return f"col{c}"
    if rel == "SAME_BOX":
        return f"box{b}"
    return cell_ref


def _blocker_rows_for_digit_target(
    target_cell: Optional[str],
    target_digit: Optional[int],
    witness_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not target_cell or target_digit not in range(1, 10):
        return rows

    target_ref = f"candidate {target_digit} in {target_cell}"
    for witness in witness_rows:
        relation = witness.get("relation")
        blocker_cell = witness.get("cell")
        actor_type = _house_type_for_relation(relation)
        actor_ref = _house_ref_for_relation(target_cell, relation)
        rows.append(
            _proof_ladder_row(
                step_kind="BLOCKER_RECEIPT",
                actor_type=actor_type,
                actor_ref=actor_ref,
                target_ref=target_ref,
                rival_digit=target_digit,
                supported_claim="DIGIT_BLOCKED",
                fact_basis={
                    "blocking_house": actor_ref,
                    "blocker_cell": blocker_cell,
                    "blocker_digit": target_digit,
                    "relation": relation,
                },
                spoken_line_seed=f"{actor_ref} already contains {target_digit} at {blocker_cell}."
            )
        )
    return rows


def _house_cell_indexes(h_type: str, idx1: int) -> List[int]:
    if h_type == "row":
        return [rc_to_cell_index(idx1, c) for c in range(1, 10)]
    if h_type == "col":
        return [rc_to_cell_index(r, idx1) for r in range(1, 10)]
    if h_type == "box":
        box = int(idx1)
        r0 = ((box - 1) // 3) * 3 + 1
        c0 = ((box - 1) % 3) * 3 + 1
        return [rc_to_cell_index(r0 + dr, c0 + dc) for dr in range(3) for dc in range(3)]
    return []


def _placed_digit_in_cell(grid81: str, cell_index: Optional[int]) -> Optional[int]:
    if cell_index is None:
        return None
    try:
        ch = (grid81 or "")[int(cell_index)]
    except Exception:
        return None
    if ch in DIGITS:
        return int(ch)
    return None


def _fixed_cell_payload(grid81: str, cell_index: int) -> Dict[str, Any]:
    placed_digit = _placed_digit_in_cell(grid81, cell_index)
    r, c = cell_index_to_rc(int(cell_index))
    b = box_index_1to9(r, c)
    return {
        "cell": _cell_ref_string(cell_index),
        "cell_index": int(cell_index),
        "row": r,
        "col": c,
        "box": b,
        "mask": 0,
        "digits": [placed_digit] if placed_digit in range(1, 10) else [],
        "placed_value": placed_digit,
        "is_filled": placed_digit in range(1, 10),
    }


def _house_existing_digit_cell(grid81: str, house: Optional[Dict[str, Any]], digit: Optional[int]) -> Optional[str]:
    if not house or digit not in range(1, 10):
        return None
    ci = _find_placed_digit_in_house(grid81, str(house.get("type")), int(house.get("index1to9")), int(digit))
    return _cell_ref_string(ci) if ci is not None else None


def _cell_already_filled_geometry(grid81: str, cell_index: int, asked_digit: Optional[int]) -> Dict[str, Any]:
    payload = _fixed_cell_payload(grid81, cell_index)
    target_cell = payload.get("cell")
    placed_value = payload.get("placed_value")
    row = int(payload.get("row"))
    col = int(payload.get("col"))
    box = int(payload.get("box"))

    return {
        "geometry_kind": "CELL_ALREADY_FILLED",
        "target_cell": target_cell,
        "asked_digit": asked_digit,
        "placed_value": placed_value,
        "houses": [f"row{row}", f"col{col}", f"box{box}"],
        "filled_state": {
            "cell": target_cell,
            "placed_value": placed_value,
            "row": row,
            "col": col,
            "box": box,
        },
        "scan_order": ["FILLED_CELL_FACT"],
        "primary_spotlight": target_cell,
    }


def _house_digit_already_placed_geometry(
    grid81: str,
    house: Dict[str, Any],
    digit: int,
    existing_cell: str,
) -> Dict[str, Any]:
    h_type = str(house.get("type"))
    idx1 = int(house.get("index1to9"))
    existing_index = _parse_cell_ref(existing_cell)
    open_seat_rows: List[Dict[str, Any]] = []

    for ci in _house_cell_indexes(h_type, idx1):
        if existing_index is not None and int(ci) == int(existing_index):
            continue
        if grid81[int(ci)] in DIGITS:
            continue

        seat_ref = _cell_ref_string(ci)
        blockers = _digit_blockers_for_cell(grid81, ci, digit)
        blocker_receipts = []
        for _, blocker in (blockers.get("blockers") or {}).items():
            if blocker is None:
                continue
            blocker_receipts.append({
                "blocking_house": _house_ref_for_relation(seat_ref, blocker.get("relation")),
                "blocker_cell": blocker.get("cell"),
                "relation": blocker.get("relation"),
            })

        open_seat_rows.append({
            "seat": seat_ref,
            "closed_for_digit": True,
            "blockers": blocker_receipts,
        })

    return {
        "geometry_kind": "HOUSE_DIGIT_ALREADY_PLACED",
        "target_house": f"{h_type}{idx1}",
        "target_digit": digit,
        "existing_digit_cell": existing_cell,
        "open_seat_rows": open_seat_rows,
        "scan_order": ["HOUSE_OCCUPANCY_FACT", "OPEN_SEAT_CONFIRMATION"],
        "primary_spotlight": f"{h_type}{idx1}",
    }


def _cell_three_house_universe_geometry(
    grid81: str,
    options_all_masks: Dict[str, int],
    cell_index: int,
) -> Dict[str, Any]:
    payload = _candidate_payload_for_cell(grid81, options_all_masks, cell_index)
    target_cell = payload.get("cell")
    row = int(payload.get("row"))
    col = int(payload.get("col"))
    box = int(payload.get("box"))
    survivors = list(payload.get("digits") or [])

    blocked_digits_by_house: Dict[str, List[Dict[str, Any]]] = {
        f"row{row}": [],
        f"col{col}": [],
        f"box{box}": [],
    }
    merged_blocked_digits: List[int] = []
    blocker_receipts: List[Dict[str, Any]] = []

    for digit in range(1, 10):
        blockers = _digit_blockers_for_cell(grid81, cell_index, digit)
        present = False
        for house_key, blocker in (blockers.get("blockers") or {}).items():
            if blocker is None:
                continue
            present = True
            house_ref = f"row{row}" if house_key == "row" else f"col{col}" if house_key == "col" else f"box{box}"
            receipt = {
                "digit": digit,
                "blocking_house": house_ref,
                "blocker_cell": blocker.get("cell"),
                "relation": blocker.get("relation"),
            }
            blocked_digits_by_house[house_ref].append(receipt)
            blocker_receipts.append(receipt)
        if present:
            merged_blocked_digits.append(digit)

    merged_blocked_digits = sorted(set(merged_blocked_digits))

    candidate_status_map = []
    for digit in range(1, 10):
        status = (
            "SURVIVES" if digit in survivors else
            "BLOCKED" if digit in merged_blocked_digits else
            "NOT_CANDIDATE"
        )
        candidate_status_map.append({
            "digit": digit,
            "status": status,
        })

    return {
        "geometry_kind": "CELL_THREE_HOUSE_UNIVERSE",
        "target_cell": target_cell,
        "asked_digit": None,
        "houses": [f"row{row}", f"col{col}", f"box{box}"],
        "blocked_digits_by_house": blocked_digits_by_house,
        "merged_blocked_digits": merged_blocked_digits,
        "surviving_digits": survivors,
        "candidate_status_map": candidate_status_map,
        "blocker_receipts": blocker_receipts,
        "scan_order": ["ROW", "COLUMN", "BOX"],
        "primary_spotlight": target_cell,
    }


def _house_digit_seat_map_geometry(
    grid81: str,
    options_all_masks: Dict[str, int],
    house: Dict[str, Any],
    digit: int,
) -> Dict[str, Any]:
    h_type = str(house.get("type"))
    idx1 = int(house.get("index1to9"))
    full_map = candidate_cells_by_house_from_masks(options_all_masks, grid81)
    by_house = (((full_map.get(str(digit), {}) or {}).get("candidate_cells_by_house") or {}).get(h_type) or {})
    candidate_indexes = [int(ci) for ci in (by_house.get(str(idx1), []) or [])]
    candidate_cells = [_cell_ref_string(ci) for ci in candidate_indexes]

    seat_rows: List[Dict[str, Any]] = []
    surviving_seats: List[str] = []
    eliminated_seats: List[Dict[str, Any]] = []

    for ci in _house_cell_indexes(h_type, idx1):
        if grid81[int(ci)] in DIGITS:
            continue
        seat_ref = _cell_ref_string(ci)
        blockers = _digit_blockers_for_cell(grid81, ci, digit)
        blocker_receipts = []
        for _, blocker in (blockers.get("blockers") or {}).items():
            if blocker is None:
                continue
            blocker_receipts.append({
                "blocking_house": _house_ref_for_relation(seat_ref, blocker.get("relation")),
                "blocker_cell": blocker.get("cell"),
                "relation": blocker.get("relation"),
            })

        survives = seat_ref in candidate_cells
        if survives:
            surviving_seats.append(seat_ref)
        else:
            eliminated_seats.append({
                "seat": seat_ref,
                "blockers": blocker_receipts,
            })

        seat_rows.append({
            "seat": seat_ref,
            "survives": survives,
            "blockers": blocker_receipts,
        })

    return {
        "geometry_kind": "HOUSE_DIGIT_SEAT_MAP",
        "target_house": f"{h_type}{idx1}",
        "target_digit": digit,
        "seat_rows": seat_rows,
        "surviving_seats": surviving_seats,
        "eliminated_seats": eliminated_seats,
        "only_place": len(surviving_seats) == 1,
        "scan_order": ["HOUSE_SEATS"],
        "primary_spotlight": f"{h_type}{idx1}",
    }


def _rival_comparison_frame_geometry(
    grid81: str,
    options_all_masks: Dict[str, int],
    primary_cell: Optional[str],
    rival_cell: Optional[str],
    digit: Optional[int],
) -> Dict[str, Any]:
    primary_index = _parse_cell_ref(primary_cell) if primary_cell else None
    rival_index = _parse_cell_ref(rival_cell) if rival_cell else None
    primary_before = _candidate_payload_for_cell(grid81, options_all_masks, primary_index) if primary_index is not None else {}
    rival_before = _candidate_payload_for_cell(grid81, options_all_masks, rival_index) if rival_index is not None else {}
    primary_blockers = _digit_blockers_for_cell(grid81, primary_index, digit) if primary_index is not None and digit in range(1, 10) else {"blocked": False, "blockers": {}}
    rival_blockers = _digit_blockers_for_cell(grid81, rival_index, digit) if rival_index is not None and digit in range(1, 10) else {"blocked": False, "blockers": {}}

    return {
        "geometry_kind": "RIVAL_COMPARISON_FRAME",
        "target_digit": digit,
        "primary_cell": primary_cell,
        "rival_cell": rival_cell,
        "shared_standard": "LOCAL_BLOCKER_TEST",
        "primary": {
            "cell": primary_cell,
            "current_candidates": primary_before.get("digits") or [],
            "blocked": bool(primary_blockers.get("blocked")),
            "blockers": [b for b in (primary_blockers.get("blockers") or {}).values() if b is not None],
        },
        "rival": {
            "cell": rival_cell,
            "current_candidates": rival_before.get("digits") or [],
            "blocked": bool(rival_blockers.get("blocked")),
            "blockers": [b for b in (rival_blockers.get("blockers") or {}).values() if b is not None],
        },
        "primary_spotlight": primary_cell,
    }


def _pattern_structure_frame_geometry(
    anchor: Dict[str, Any],
    technique_legitimacy: Dict[str, Any],
    focus_cells: List[str],
    focus_houses: List[str],
) -> Dict[str, Any]:
    technique = (anchor.get("technique") or {})
    return {
        "geometry_kind": "PATTERN_STRUCTURE_FRAME",
        "claimed_technique_id": technique_legitimacy.get("claimed_technique_id") or technique.get("id"),
        "anchor_technique_id": technique_legitimacy.get("anchor_technique_id") or technique.get("id"),
        "structure_is_present": bool(technique_legitimacy.get("structure_is_present")),
        "pattern_members": focus_cells,
        "governing_houses": focus_houses,
        "consequence_scope": technique_legitimacy.get("consequence_scope") or anchor.get("target_cell"),
        "primary_spotlight": technique_legitimacy.get("claimed_technique_id") or technique.get("name"),
    }


def _build_local_proof_geometry_for_move_proof(
    grid81: str,
    options_all_masks: Dict[str, int],
    query: Dict[str, Any],
    anchor: Dict[str, Any],
    proof_object: str,
    challenge_lane: str,
    cell_index: Optional[int],
    target_cell: Optional[str],
    asked_digit: Optional[int],
    rival_cell: Optional[str],
    technique_legitimacy: Dict[str, Any],
    focus_cells: List[str],
    focus_houses: List[str],
) -> Dict[str, Any]:
    house = _parse_house_ref(query.get("house"))

    if challenge_lane == "RIVAL_COMPARISON" and asked_digit in range(1, 10) and target_cell and rival_cell:
        return _rival_comparison_frame_geometry(
            grid81=grid81,
            options_all_masks=options_all_masks,
            primary_cell=target_cell,
            rival_cell=rival_cell,
            digit=asked_digit,
        )

    if challenge_lane == "TECHNIQUE_LEGITIMACY" or proof_object == "TECHNIQUE_CLAIM_IS_VALID" or query.get("claimed_technique_id"):
        return _pattern_structure_frame_geometry(
            anchor=anchor,
            technique_legitimacy=technique_legitimacy,
            focus_cells=focus_cells,
            focus_houses=focus_houses,
        )

    if house is not None and asked_digit in range(1, 10):
        return _house_digit_seat_map_geometry(
            grid81=grid81,
            options_all_masks=options_all_masks,
            house=house,
            digit=asked_digit,
        )

    if cell_index is not None:
        geom = _cell_three_house_universe_geometry(
            grid81=grid81,
            options_all_masks=options_all_masks,
            cell_index=cell_index,
        )
        geom["asked_digit"] = asked_digit if asked_digit in range(1, 10) else None
        return geom

    return {
        "geometry_kind": "NONE",
        "primary_spotlight": target_cell,
    }



def _move_proof_archetype_for_truth(
    method_family: str,
    proof_object: str,
    challenge_lane: str,
    answer_polarity: str,
    nonproof_reason: str,
    rival_cell: Optional[str],
    claimed_technique_id: Optional[str],
    geometry_kind: Optional[str] = None,
) -> str:
    mf = str(method_family or "").strip().upper()
    po = str(proof_object or "").strip().upper()
    lane = str(challenge_lane or "").strip().upper()
    polarity = str(answer_polarity or "").strip().upper()
    reason = str(nonproof_reason or "").strip().upper()
    geom = str(geometry_kind or "").strip().upper()
    has_rival_cell = bool(str(rival_cell or "").strip())
    has_claimed_technique = bool(str(claimed_technique_id or "").strip())

    if geom == "PATTERN_STRUCTURE_FRAME" or mf in ("TECHNIQUE_LEGITIMACY",) or lane == "TECHNIQUE_LEGITIMACY" or has_claimed_technique:
        return "PATTERN_LEGITIMACY_CHECK"

    if geom == "RIVAL_COMPARISON_FRAME" or mf == "CONTRAST_TEST" or po == "CELL_A_WINS_OVER_CELL_B_FOR_DIGIT" or has_rival_cell:
        return "CONTRAST_DUEL"

    if geom == "HOUSE_DIGIT_SEAT_MAP":
        return "SURVIVOR_LADDER"

    if (
        geom == "CELL_THREE_HOUSE_UNIVERSE"
        and polarity == "NOT_LOCALLY_PROVED"
        and lane in ("ELIMINATION_LEGITIMACY", "CANDIDATE_POSSIBILITY")
        and po in ("ELIMINATION_IS_LEGAL", "CELL_CAN_BE_DIGIT", "LOCAL_PROOF_INSUFFICIENT")
    ):
        return "LOCAL_PERMISSIBILITY_SCAN"

    if mf in ("HOUSE_UNIQUENESS", "RIVAL_ELIMINATION_LADDER") or po in (
        "CELL_IS_ONLY_PLACE_FOR_DIGIT_IN_HOUSE",
        "DIGIT_SURVIVES_RIVAL_CANDIDATES_IN_CELL",
        "CELL_CAN_BE_DIGIT",
    ):
        return "SURVIVOR_LADDER"

    if mf in ("DIRECT_CONTRADICTION", "ACTION_LEGITIMACY") or po in (
        "CELL_CANNOT_BE_DIGIT",
        "HOUSE_BLOCKS_DIGIT_FOR_TARGET",
        "ELIMINATION_IS_LEGAL",
    ):
        return "LOCAL_CONTRADICTION_SPOTLIGHT"

    if polarity == "NOT_LOCALLY_PROVED":
        if reason in (
            "ROUTE_DEPENDENT_NOT_LOCALLY_VISIBLE",
            "ELIMINATION_SUPPORT_NOT_LOCALLY_VISIBLE",
        ) and has_claimed_technique:
            return "PATTERN_LEGITIMACY_CHECK"
        return "HONEST_INSUFFICIENCY_ANSWER"

    if lane == "FORCEDNESS_OR_UNIQUENESS":
        return "SURVIVOR_LADDER"

    if lane in ("CANDIDATE_IMPOSSIBILITY", "ELIMINATION_LEGITIMACY"):
        return "LOCAL_CONTRADICTION_SPOTLIGHT"

    return "HONEST_INSUFFICIENCY_ANSWER"


def _move_proof_doctrine_for_archetype(archetype: str) -> str:
    arc = str(archetype or "").strip().upper()
    if arc == "LOCAL_CONTRADICTION_SPOTLIGHT":
        return "contradiction_spotlight_v1"
    if arc == "LOCAL_PERMISSIBILITY_SCAN":
        return "local_permissibility_scan_v1"
    if arc == "SURVIVOR_LADDER":
        return "survivor_ladder_v1"
    if arc == "CONTRAST_DUEL":
        return "contrast_duel_v1"
    if arc == "PATTERN_LEGITIMACY_CHECK":
        return "pattern_legitimacy_v1"
    return "honest_insufficiency_v1"


def _move_proof_overlay_story_kind_for_archetype(archetype: str) -> str:
    arc = str(archetype or "").strip().upper()
    if arc == "LOCAL_CONTRADICTION_SPOTLIGHT":
        return "CONTRADICTION_SPOTLIGHT"
    if arc == "LOCAL_PERMISSIBILITY_SCAN":
        return "LOCAL_PERMISSIBILITY_SCAN"
    if arc == "SURVIVOR_LADDER":
        return "SURVIVOR_LADDER"
    if arc == "CONTRAST_DUEL":
        return "CONTRAST_DUEL"
    if arc == "PATTERN_LEGITIMACY_CHECK":
        return "PATTERN_LEGITIMACY"
    return "HONEST_INSUFFICIENCY"


def _proof_motion_type_for_archetype(archetype: str) -> str:
    arc = str(archetype or "").strip().upper()
    if arc == "LOCAL_CONTRADICTION_SPOTLIGHT":
        return "BLOCKER_COLLAPSE"
    if arc == "LOCAL_PERMISSIBILITY_SCAN":
        return "BLOCKER_SCAN_TO_SURVIVORS"
    if arc == "SURVIVOR_LADDER":
        return "RIVAL_ELIMINATION"
    if arc == "CONTRAST_DUEL":
        return "SHARED_STANDARD_COMPARISON"
    if arc == "PATTERN_LEGITIMACY_CHECK":
        return "STRUCTURE_VALIDATION"
    return "BOUNDED_HONEST_LIMIT"


def _visible_tension_type_for_truth(
    archetype: str,
    answer_polarity: str,
    challenge_lane: str,
    proof_object: str,
) -> str:
    arc = str(archetype or "").strip().upper()
    polarity = str(answer_polarity or "").strip().upper()
    lane = str(challenge_lane or "").strip().upper()
    obj = str(proof_object or "").strip().upper()

    if arc == "LOCAL_CONTRADICTION_SPOTLIGHT":
        return "CAN_THIS_STAND"
    if arc == "LOCAL_PERMISSIBILITY_SCAN":
        return "WHAT_SURVIVES_THE_LOCAL_SCAN"
    if arc == "SURVIVOR_LADDER":
        return "WHICH_OPTION_SURVIVES"
    if arc == "CONTRAST_DUEL":
        return "WHICH_RIVAL_WINS"
    if arc == "PATTERN_LEGITIMACY_CHECK":
        return "IS_THE_PATTERN_REAL"
    if polarity == "NOT_LOCALLY_PROVED":
        return "HOW_FAR_LOCAL_PROOF_REACHES"
    if lane == "FORCEDNESS_OR_UNIQUENESS" or obj == "CELL_IS_ONLY_PLACE_FOR_DIGIT_IN_HOUSE":
        return "WHICH_HOME_REMAINS"
    return "WHAT_THE_LOCAL_PICTURE_TRULY_ESTABLISHES"


def _move_proof_speech_skeleton_for_archetype(archetype: str) -> List[str]:
    arc = str(archetype or "").strip().upper()
    if arc == "LOCAL_CONTRADICTION_SPOTLIGHT":
        return ["focus_target", "blocker", "contradiction", "local_conclusion"]
    if arc == "LOCAL_PERMISSIBILITY_SCAN":
        return ["focus_target", "scan_arena", "house_pressure", "survivor_reveal", "permissibility_conclusion"]
    if arc == "SURVIVOR_LADDER":
        return ["scope", "rival_failures", "survivor", "bounded_conclusion"]
    if arc == "CONTRAST_DUEL":
        return ["rivalry_frame", "rival_a_result", "rival_b_result", "winner"]
    if arc == "PATTERN_LEGITIMACY_CHECK":
        return ["claimed_technique", "qualifying_structure", "local_consequence"]
    return ["direct_answer", "what_is_not_proved", "current_local_state"]


def _ordered_house_scan_refs(local_proof_geometry: Dict[str, Any]) -> List[str]:
    geom = local_proof_geometry if isinstance(local_proof_geometry, dict) else {}
    houses = [str(h) for h in (geom.get("houses") or []) if str(h).strip()]
    if not houses:
        return []

    scan_order = [str(x).strip().upper() for x in (geom.get("scan_order") or []) if str(x).strip()]
    prefix_for_scan = {
        "ROW": "row",
        "COLUMN": "col",
        "COL": "col",
        "BOX": "box",
    }

    ordered: List[str] = []
    for token in scan_order:
        prefix = prefix_for_scan.get(token)
        if not prefix:
            continue
        match = next((house for house in houses if house.startswith(prefix) and house not in ordered), None)
        if match:
            ordered.append(match)

    for house in houses:
        if house not in ordered:
            ordered.append(house)

    return ordered


def _survivor_digits_from_geometry(local_proof_geometry: Dict[str, Any]) -> List[int]:
    geom = local_proof_geometry if isinstance(local_proof_geometry, dict) else {}
    survivors = geom.get("surviving_digits")
    if isinstance(survivors, list):
        out: List[int] = []
        for value in survivors:
            try:
                digit = int(value)
            except Exception:
                continue
            if digit in range(1, 10):
                out.append(digit)
        return out

    out: List[int] = []
    for row in (geom.get("candidate_status_map") or []):
        try:
            digit = int(row.get("digit"))
        except Exception:
            continue
        if digit not in range(1, 10):
            continue
        if str(row.get("status") or "").strip().upper() == "SURVIVES":
            out.append(digit)
    return out


def _local_permissibility_stage_rows(
    local_proof_geometry: Dict[str, Any],
    target_cell: Optional[str],
    asked_digit: Optional[int],
) -> List[Dict[str, Any]]:
    geom = local_proof_geometry if isinstance(local_proof_geometry, dict) else {}
    if not target_cell:
        return []

    blocked_by_house = geom.get("blocked_digits_by_house") or {}
    ordered_houses = _ordered_house_scan_refs(geom)
    survivor_digits = _survivor_digits_from_geometry(geom)
    merged_blocked = [
        int(d)
        for d in (geom.get("merged_blocked_digits") or [])
        if isinstance(d, int) and d in range(1, 10)
    ]

    target_ref = (
        f"candidate {asked_digit} in {target_cell}"
        if isinstance(asked_digit, int) and asked_digit in range(1, 10)
        else target_cell
    )

    rows: List[Dict[str, Any]] = [
        _proof_ladder_row(
            step_kind="SPOTLIGHT_TARGET",
            actor_type="CELL",
            actor_ref=target_cell,
            target_ref=target_ref,
            rival_digit=asked_digit if isinstance(asked_digit, int) and asked_digit in range(1, 10) else None,
            supported_claim="LOCAL_ARENA_DEFINED",
            fact_basis={
                "target_cell": target_cell,
                "houses_under_scan": ordered_houses,
                "scan_order": geom.get("scan_order") or [],
            },
            spoken_line_seed=(
                f"We test {target_cell} against its row, column, and box before we decide whether {asked_digit} can be removed."
                if isinstance(asked_digit, int) and asked_digit in range(1, 10)
                else f"We test {target_cell} against its row, column, and box before making the local call."
            ),
        )
    ]

    for house_ref in ordered_houses:
        receipts = blocked_by_house.get(house_ref) or []
        blocked_digits: List[int] = []
        normalized_receipts: List[Dict[str, Any]] = []
        for receipt in receipts:
            try:
                digit = int(receipt.get("digit"))
            except Exception:
                digit = None
            if digit in range(1, 10):
                blocked_digits.append(digit)
            normalized_receipts.append({
                "digit": digit,
                "blocking_house": receipt.get("blocking_house"),
                "blocker_cell": receipt.get("blocker_cell"),
                "relation": receipt.get("relation"),
            })

        blocked_digits = sorted(set(d for d in blocked_digits if d in range(1, 10)))
        rows.append(
            _proof_ladder_row(
                step_kind="HOUSE_SCAN",
                actor_type="HOUSE",
                actor_ref=house_ref,
                target_ref=target_ref,
                rival_digit=asked_digit if isinstance(asked_digit, int) and asked_digit in range(1, 10) else None,
                supported_claim="HOUSE_PRESSURE_APPLIED",
                fact_basis={
                    "house_ref": house_ref,
                    "blocked_digits": blocked_digits,
                    "receipt_count": len(normalized_receipts),
                    "receipts": normalized_receipts,
                },
                spoken_line_seed=(
                    f"{house_ref} rules out {', '.join(str(d) for d in blocked_digits)} around {target_cell}."
                    if blocked_digits
                    else f"{house_ref} adds no direct blocker against {asked_digit}."
                ),
            )
        )

    rows.append(
        _proof_ladder_row(
            step_kind="SURVIVOR_REVEAL",
            actor_type="CELL",
            actor_ref=target_cell,
            target_ref=target_ref,
            rival_digit=asked_digit if isinstance(asked_digit, int) and asked_digit in range(1, 10) else None,
            supported_claim="SURVIVORS_AFTER_LOCAL_SCAN",
            fact_basis={
                "merged_blocked_digits": merged_blocked,
                "surviving_digits": survivor_digits,
                "asked_digit_survives": (
                    isinstance(asked_digit, int)
                    and asked_digit in range(1, 10)
                    and asked_digit in survivor_digits
                ),
            },
            spoken_line_seed=(
                f"After the local scan, {target_cell} still keeps {', '.join(str(d) for d in survivor_digits)} alive."
                if survivor_digits
                else f"After the local scan, the asked digit is not removed by any direct local blocker."
            ),
        )
    )

    return rows


def _build_move_proof_story_arc(
    narrative_archetype: str,
    answer_polarity: str,
    challenge_lane: str,
    proof_object: str,
    decisive_fact: str,
    proof_claim: str,
    proof_ladder_rows: List[Dict[str, Any]],
    local_proof_geometry: Dict[str, Any],
    target_cell: Optional[str],
    asked_digit: Optional[int],
) -> Dict[str, Any]:
    arc = str(narrative_archetype or "").strip().upper()
    survivors = _survivor_digits_from_geometry(local_proof_geometry)
    ordered_houses = _ordered_house_scan_refs(local_proof_geometry)

    story_arc = {
        "opening_mode":
            "SPOTLIGHT_TARGET" if arc == "LOCAL_CONTRADICTION_SPOTLIGHT" else
            "SCAN_TARGET_CELL" if arc == "LOCAL_PERMISSIBILITY_SCAN" else
            "SHOW_LIVE_CONTENDERS" if arc == "SURVIVOR_LADDER" else
            "FRAME_DUEL" if arc == "CONTRAST_DUEL" else
            "NAME_PATTERN" if arc == "PATTERN_LEGITIMACY_CHECK" else
            "SPOTLIGHT_LOCAL_QUESTION",
        "motion_mode":
            "BLOCKER_TO_CONTRADICTION" if arc == "LOCAL_CONTRADICTION_SPOTLIGHT" else
            "BLOCKER_SCAN_TO_SURVIVORS" if arc == "LOCAL_PERMISSIBILITY_SCAN" else
            "RIVAL_FAILURES_TO_SURVIVOR" if arc == "SURVIVOR_LADDER" else
            "COMPARE_UNDER_SHARED_STANDARD" if arc == "CONTRAST_DUEL" else
            "STRUCTURE_TO_CONSEQUENCE" if arc == "PATTERN_LEGITIMACY_CHECK" else
            "BOUNDED_HONEST_READOUT",
        "proof_motion_type": _proof_motion_type_for_archetype(arc),
        "visible_tension_type": _visible_tension_type_for_truth(
            arc,
            answer_polarity,
            challenge_lane,
            proof_object,
        ),
        "local_landing_line": decisive_fact or proof_claim,
        "proof_row_count": len(proof_ladder_rows),
    }

    if arc == "LOCAL_PERMISSIBILITY_SCAN":
        story_arc.update({
            "opening_mode": "SPOTLIGHT_LOCAL_SCAN",
            "motion_mode": "BLOCKER_SCAN_TO_SURVIVORS",
            "proof_motion_type": "BLOCKER_SCAN_TO_SURVIVORS",
            "setup_beat": "SPOTLIGHT_TARGET_AND_DOUBT",
            "confrontation_beat": "ROW_COLUMN_BOX_PRESSURE",
            "resolution_beat": "SURVIVOR_REVEAL",
            "scan_order": local_proof_geometry.get("scan_order") or ["ROW", "COLUMN", "BOX"],
            "ordered_houses": ordered_houses,
            "reveal_target_digit": asked_digit if isinstance(asked_digit, int) and asked_digit in range(1, 10) else None,
            "surviving_digits": survivors,
            "delay_reveal_until_resolution": True,
            "must_not_open_with_merged_summary": True,
            "must_stage_house_pressure": True,
            "must_land_on_asked_digit_survival": (
                isinstance(asked_digit, int)
                and asked_digit in range(1, 10)
                and asked_digit in survivors
            ),
            "scan_arena_line": (
                f"We test {target_cell} by the three judges around it: its row, its column, and its box."
                if target_cell else
                "We test the target by the three local judges around it: row, column, and box."
            ),
            "resolution_landing_mode": "ASKED_DIGIT_SURVIVES_LOCAL_SCAN",
        })

    return story_arc


def _build_move_proof_micro_stage_plan(
    story_arc: Dict[str, Any],
    story_focus: Dict[str, Any],
    story_question: Dict[str, Any],
    narrative_archetype: str,
    local_proof_geometry: Dict[str, Any],
    proof_ladder_rows: List[Dict[str, Any]],
    asked_digit: Optional[int],
) -> Dict[str, Any]:
    arc = str(narrative_archetype or "").strip().upper()
    proof_row_count = len(proof_ladder_rows)

    plan = {
        "micro_setup": {
            "goal": "SPOTLIGHT_LOCAL_CHALLENGE",
            "opening_mode": story_arc.get("opening_mode"),
            "focus_scope": story_focus.get("scope"),
            "spotlight_object": story_question.get("spotlight_object"),
            "must_name_local_question": True,
        },
        "micro_confrontation": {
            "goal": "WALK_LOCAL_PROOF",
            "motion_mode": story_arc.get("motion_mode"),
            "proof_motion_type": story_arc.get("proof_motion_type"),
            "visible_tension_type": story_arc.get("visible_tension_type"),
            "proof_row_count": story_arc.get("proof_row_count", 0),
            "prefer_ordered_ladder_when_present": True,
        },
        "micro_resolution": {
            "goal": "LAND_LOCAL_RESULT",
            "landing_line": story_arc.get("local_landing_line"),
            "must_not_commit_move": True,
            "must_close_boundedly": True,
        },
        "compression_mode":
            "FULL_THREE_BEAT" if proof_row_count >= 3 else
            "LIGHT_THREE_BEAT" if proof_row_count > 0 else
            "MINIMAL_THREE_BEAT",
    }

    if arc == "LOCAL_PERMISSIBILITY_SCAN":
        plan["micro_setup"].update({
            "goal": "SPOTLIGHT_TARGET_AND_LOCAL_DOUBT",
            "opening_mode": "SPOTLIGHT_LOCAL_SCAN",
            "must_name_local_question": True,
            "must_name_target_cell_first": True,
            "must_define_local_arena": True,
            "scan_arena": local_proof_geometry.get("houses") or [],
        })
        plan["micro_confrontation"].update({
            "goal": "WALK_ROW_COLUMN_BOX_PRESSURE",
            "motion_mode": "BLOCKER_SCAN_TO_SURVIVORS",
            "proof_motion_type": "BLOCKER_SCAN_TO_SURVIVORS",
            "prefer_ordered_ladder_when_present": True,
            "must_follow_scan_order": True,
            "scan_order": story_arc.get("scan_order") or ["ROW", "COLUMN", "BOX"],
            "ordered_houses": story_arc.get("ordered_houses") or _ordered_house_scan_refs(local_proof_geometry),
            "must_not_open_with_merged_summary": True,
            "must_let_blocked_digits_fall_away_progressively": True,
        })
        plan["micro_resolution"].update({
            "goal": "REVEAL_SURVIVING_DIGIT_AND_LIMIT",
            "must_not_commit_move": True,
            "must_close_boundedly": True,
            "must_land_on_asked_digit_survival": True,
            "asked_digit": asked_digit if isinstance(asked_digit, int) and asked_digit in range(1, 10) else None,
            "must_distinguish_survival_from_placement": True,
        })
        plan["compression_mode"] = (
            "FULL_THREE_BEAT" if proof_row_count >= 4 else
            "LIGHT_THREE_BEAT" if proof_row_count >= 2 else
            "MINIMAL_THREE_BEAT"
        )

    return plan


def _canonical_method_family_for_move_proof(method_family: str) -> str:
    mf = str(method_family or "").strip().upper()
    if mf == "DIRECT_CONTRADICTION":
        return "DIRECT_SOLVED_PEER_CONTRADICTION"
    if mf == "HOUSE_UNIQUENESS":
        return "HOUSE_UNIQUENESS"
    if mf == "RIVAL_ELIMINATION_LADDER":
        return "CELL_SURVIVOR"
    if mf == "CONTRAST_TEST":
        return "RIVAL_COMPARISON"
    if mf == "TECHNIQUE_LEGITIMACY":
        return "PATTERN_STRUCTURE"
    if mf == "ACTION_LEGITIMACY":
        return "DIRECT_SOLVED_PEER_CONTRADICTION"
    return "BOUNDED_INSUFFICIENCY"

# ============================================================================
# Family normalizers
# ============================================================================



def _normalize_move_proof_query(grid81: str, query: Dict[str, Any]) -> Dict[str, Any]:
    options_all_masks = _compute_options_masks(grid81)
    step_obj = _run_next_step_obj(grid81)
    anchor = _build_anchor(query, step_obj)

    requested_cell = query.get("requested_cell") or query.get("anchor_target_cell")
    cell_index = _parse_cell_ref(requested_cell)
    target_cell = _cell_ref_string(cell_index) if cell_index is not None else query.get("anchor_target_cell")

    requested_digit = query.get("requested_digit")
    try:
        requested_digit = int(requested_digit) if requested_digit is not None else None
    except Exception:
        requested_digit = None

    anchor_target_digit = query.get("anchor_target_digit")
    try:
        anchor_target_digit = int(anchor_target_digit) if anchor_target_digit is not None else None
    except Exception:
        anchor_target_digit = None

    question_profile = str(query.get("query_profile") or "").strip().upper() or "PROVE_ELIMINATION"
    ask_kind = str(query.get("proof_ask_kind") or query.get("user_ask_kind") or question_profile).strip().upper()
    challenge_lane = str(query.get("challenge_lane") or "").strip().upper() or "ELIMINATION_LEGITIMACY"
    proof_object_hint = str(query.get("proof_object_hint") or "").strip().upper() or None
    claimed_technique_id = query.get("claimed_technique_id")
    rival_cell = query.get("rival_cell")
    rival_digit = query.get("rival_digit")
    try:
        rival_digit = int(rival_digit) if rival_digit is not None else None
    except Exception:
        rival_digit = None

    target_digit = anchor_target_digit
    asked_digit = requested_digit

    target_before: Dict[str, Any] = {}
    witness_rows: List[Dict[str, Any]] = []
    decisive_fact = ""
    proof_claim = ""
    focus_houses: List[str] = []
    focus_cells: List[str] = [target_cell] if target_cell else []
    secondary_focus_cells: List[str] = [rival_cell] if isinstance(rival_cell, str) and rival_cell.strip() else []

    proof_object = proof_object_hint or "LOCAL_PROOF_INSUFFICIENT"
    method_family = "PERSISTENCE_OR_INSUFFICIENCY"
    answer_polarity = "NOT_LOCALLY_PROVED"
    local_truth_status = "NOT_LOCALLY_ESTABLISHED"
    proof_ladder_rows: List[Dict[str, Any]] = []
    nonproof_reason: Optional[str] = None
    survivor_summary: Dict[str, Any] = {}
    actor_model = "LOCAL_SINGLE_SCOPE"
    contrast_summary: Dict[str, Any] = {}
    technique_legitimacy: Dict[str, Any] = {}

    house = _parse_house_ref(query.get("house"))
    filled_cell_digit = _placed_digit_in_cell(grid81, cell_index)
    existing_house_digit_cell = _house_existing_digit_cell(grid81, house, asked_digit)
    guard_mode: Optional[str] = None

    if house is not None and asked_digit in range(1, 10) and existing_house_digit_cell:
        guard_mode = "HOUSE_ALREADY_CONTAINS_DIGIT"
        target_cell = existing_house_digit_cell
        cell_index = _parse_cell_ref(existing_house_digit_cell)
        target_before = _fixed_cell_payload(grid81, cell_index) if cell_index is not None else {}
        focus_cells = [existing_house_digit_cell]
        focus_houses = [f"{house['type']}{int(house['index1to9'])}"]

        proof_object = "HOUSE_ALREADY_CONTAINS_DIGIT"
        method_family = "SATISFIED_CONSTRAINT"
        answer_polarity = "ALREADY_PLACED"
        local_truth_status = "HOUSE_ALREADY_HAS_DIGIT"
        actor_model = "HOUSE_OCCUPANCY_READOUT"
        decisive_fact = (
            f"{focus_houses[0]} already contains {asked_digit} at {existing_house_digit_cell}."
        )
        proof_claim = (
            f"{focus_houses[0]} does not still need a home for {asked_digit} because that digit is already placed at {existing_house_digit_cell}."
        )

        open_seats = [
            _cell_ref_string(ci)
            for ci in _house_cell_indexes(str(house["type"]), int(house["index1to9"]))
            if grid81[int(ci)] not in DIGITS and _cell_ref_string(ci) != existing_house_digit_cell
        ]

        proof_ladder_rows = [
            _proof_ladder_row(
                step_kind="HOUSE_OCCUPANCY_FACT",
                actor_type=str(house["type"]).upper(),
                actor_ref=focus_houses[0],
                target_ref=f"digit {asked_digit} in {focus_houses[0]}",
                rival_digit=asked_digit,
                supported_claim="HOUSE_ALREADY_HAS_DIGIT",
                fact_basis={
                    "house": focus_houses[0],
                    "digit": asked_digit,
                    "existing_digit_cell": existing_house_digit_cell,
                },
                spoken_line_seed=f"{focus_houses[0]} already has {asked_digit} at {existing_house_digit_cell}."
            )
        ]

        if open_seats:
            proof_ladder_rows.append(
                _proof_ladder_row(
                    step_kind="OPEN_SEATS_CLOSED_BY_OCCUPANCY",
                    actor_type=str(house["type"]).upper(),
                    actor_ref=focus_houses[0],
                    target_ref=f"digit {asked_digit} in {focus_houses[0]}",
                    rival_digit=asked_digit,
                    supported_claim="OTHER_SEATS_CLOSED_TO_DUPLICATE",
                    fact_basis={
                        "house": focus_houses[0],
                        "digit": asked_digit,
                        "existing_digit_cell": existing_house_digit_cell,
                        "remaining_open_seats": open_seats,
                    },
                    spoken_line_seed=(
                        f"That is why the other open seats in {focus_houses[0]} cannot take another {asked_digit}: {open_seats}."
                    )
                )
            )

    elif cell_index is not None and filled_cell_digit in range(1, 10):
        guard_mode = "CELL_ALREADY_FILLED"
        target_before = _fixed_cell_payload(grid81, cell_index)
        row = target_before.get("row")
        col = target_before.get("col")
        box = target_before.get("box")
        focus_houses = [f"row{row}", f"col{col}", f"box{box}"]
        focus_cells = [target_cell] if target_cell else []

        proof_object = "CELL_ALREADY_FILLED"
        method_family = "SATISFIED_CONSTRAINT"
        answer_polarity = "ALREADY_FILLED"
        local_truth_status = "CELL_ALREADY_OCCUPIED"
        actor_model = "CELL_OCCUPANCY_READOUT"
        decisive_fact = f"{target_cell} is already filled with {filled_cell_digit}."
        proof_claim = (
            f"This is not a live candidate question because {target_cell} is already occupied by {filled_cell_digit}."
        )

        proof_ladder_rows = [
            _proof_ladder_row(
                step_kind="CELL_OCCUPANCY_FACT",
                actor_type="CELL",
                actor_ref=target_cell,
                target_ref=target_cell or "target_cell",
                rival_digit=asked_digit if asked_digit in range(1, 10) else None,
                supported_claim="CELL_ALREADY_OCCUPIED",
                fact_basis={
                    "target_cell": target_cell,
                    "placed_value": filled_cell_digit,
                    "asked_digit": asked_digit,
                },
                spoken_line_seed=f"{target_cell} is already filled with {filled_cell_digit}."
            )
        ]

        if asked_digit in range(1, 10):
            proof_ladder_rows.append(
                _proof_ladder_row(
                    step_kind="CANDIDATE_QUERY_SUPERSEDED_BY_FILLED_CELL",
                    actor_type="CELL",
                    actor_ref=target_cell,
                    target_ref=f"digit {asked_digit} in {target_cell}",
                    rival_digit=asked_digit,
                    supported_claim="CANDIDATE_QUERY_NO_LONGER_APPLIES",
                    fact_basis={
                        "target_cell": target_cell,
                        "placed_value": filled_cell_digit,
                        "asked_digit": asked_digit,
                    },
                    spoken_line_seed=(
                        f"Once {target_cell} is filled with {filled_cell_digit}, it is no longer an open seat for candidate testing."
                    )
                )
            )

    elif cell_index is not None:


        target_before = _candidate_payload_for_cell(grid81, options_all_masks, cell_index)
        row = target_before.get("row")
        col = target_before.get("col")
        box = target_before.get("box")
        focus_houses = [f"row{row}", f"col{col}", f"box{box}"]

        house = _parse_house_ref(query.get("house"))
        candidate_digits_now = list(target_before.get("digits") or [])

        if challenge_lane == "TECHNIQUE_LEGITIMACY":
            proof_object = "TECHNIQUE_CLAIM_IS_VALID"
            method_family = "TECHNIQUE_LEGITIMACY"
            actor_model = "PATTERN_STRUCTURE_CHECK"

            technique_name = str(claimed_technique_id or anchor.get("technique_id") or "the current technique").strip()
            anchor_stage = str(query.get("anchor_stage") or "").strip().upper()
            anchor_step_id = query.get("anchor_step_id")

            decisive_fact = (
                f"This question is about whether the current move is legitimately being treated as {technique_name}."
            )
            proof_claim = (
                f"The technique claim is evaluated by checking whether the required local structure for {technique_name} is the one anchoring the current move."
            )

            proof_ladder_rows = [
                _proof_ladder_row(
                    step_kind="PATTERN_CLAIM",
                    actor_type="TECHNIQUE",
                    actor_ref=technique_name,
                    target_ref=target_cell or (anchor.get("target_cell") or "current_move"),
                    supported_claim="TECHNIQUE_CLAIM_UNDER_REVIEW",
                    fact_basis={
                        "claimed_technique_id": technique_name,
                        "anchor_step_id": anchor_step_id,
                        "anchor_stage": anchor_stage,
                        "anchor_target_cell": anchor.get("target_cell"),
                    },
                    spoken_line_seed=f"The current move is being checked as an instance of {technique_name}."
                )
            ]

            if anchor.get("technique_id"):
                answer_polarity = "PATTERN_CONFIRMED"
                local_truth_status = "STRUCTURE_ALIGNS_WITH_CURRENT_MOVE"
                technique_legitimacy = {
                    "claimed_technique_id": technique_name,
                    "anchor_technique_id": anchor.get("technique_id"),
                    "technique_matches_anchor": str(anchor.get("technique_id")).strip() == technique_name,
                    "structure_is_present": True,
                    "consequence_scope": anchor.get("target_cell"),
                }
            else:
                answer_polarity = "NOT_LOCALLY_PROVED"
                local_truth_status = "NOT_LOCALLY_ESTABLISHED"
                nonproof_reason = "TECHNIQUE_STRUCTURE_NOT_LOCALLY_EXTRACTED"
                technique_legitimacy = {
                    "claimed_technique_id": technique_name,
                    "anchor_technique_id": None,
                    "technique_matches_anchor": False,
                    "structure_is_present": False,
                    "consequence_scope": anchor.get("target_cell"),
                }

        elif challenge_lane == "RIVAL_COMPARISON" and asked_digit in range(1, 10):
            proof_object = "CELL_A_WINS_OVER_CELL_B_FOR_DIGIT"
            method_family = "CONTRAST_TEST"
            actor_model = "CONTRAST_DUEL"

            blockers = _digit_blockers_for_cell(grid81, cell_index, asked_digit)
            target_blocked = bool(blockers.get("blocked"))
            target_blocker_rows = _blocker_rows_for_digit_target(target_cell, asked_digit, [
                {
                    "kind": "DIGIT_BLOCKER",
                    "relation": payload.get("relation"),
                    "cell": payload.get("cell"),
                    "digit": asked_digit,
                }
                for rel, payload in (blockers.get("blockers") or {}).items()
                if payload is not None
            ])

            rival_index = _parse_cell_ref(rival_cell) if rival_cell else None
            rival_before = _candidate_payload_for_cell(grid81, options_all_masks, rival_index) if rival_index is not None else {}
            rival_digits_now = list(rival_before.get("digits") or [])
            rival_blockers = (
                _digit_blockers_for_cell(grid81, rival_index, asked_digit)
                if rival_index is not None and asked_digit in range(1, 10)
                else {"blocked": False, "blockers": {}}
            )
            rival_blocked = bool(rival_blockers.get("blocked"))

            if rival_cell:
                secondary_focus_cells = [rival_cell]

            contrast_summary = {
                "primary_cell": target_cell,
                "rival_cell": rival_cell,
                "digit_under_test": asked_digit,
                "primary_blocked": target_blocked,
                "rival_blocked": rival_blocked,
                "primary_candidates": candidate_digits_now,
                "rival_candidates": rival_digits_now,
            }

            proof_ladder_rows = [
                _proof_ladder_row(
                    step_kind="CONTRAST_TEST_PRIMARY",
                    actor_type="CELL",
                    actor_ref=target_cell,
                    target_ref=f"digit {asked_digit}",
                    rival_digit=asked_digit,
                    supported_claim="PRIMARY_CELL_TESTED",
                    fact_basis={
                        "blocked": target_blocked,
                        "candidate_digits": candidate_digits_now,
                    },
                    spoken_line_seed=(
                        f"{target_cell} {'fails' if target_blocked else 'survives'} the local test for digit {asked_digit}."
                    )
                )
            ]

            if rival_cell:
                proof_ladder_rows.append(
                    _proof_ladder_row(
                        step_kind="CONTRAST_TEST_RIVAL",
                        actor_type="CELL",
                        actor_ref=rival_cell,
                        target_ref=f"digit {asked_digit}",
                        rival_digit=asked_digit,
                        supported_claim="RIVAL_CELL_TESTED",
                        fact_basis={
                            "blocked": rival_blocked,
                            "candidate_digits": rival_digits_now,
                        },
                        spoken_line_seed=(
                            f"{rival_cell} {'fails' if rival_blocked else 'survives'} the local test for digit {asked_digit}."
                        )
                    )
                )

            if rival_cell and (target_blocked != rival_blocked):
                answer_polarity = "SURVIVES_RIVALS" if not target_blocked else "RIVAL_SURVIVES"
                local_truth_status = "PROVED_BY_CONTRAST"
                decisive_fact = (
                    f"Under the same local test for digit {asked_digit}, one rival fails and the other survives."
                )
                proof_claim = (
                    f"The contrast answer comes from testing {target_cell} and {rival_cell} under the same blocker standard for digit {asked_digit}."
                )
            else:
                answer_polarity = "NOT_LOCALLY_PROVED"
                local_truth_status = "NOT_LOCALLY_ESTABLISHED"
                nonproof_reason = "RIVALS_NOT_SEPARATED_BY_LOCAL_TEST"
                decisive_fact = (
                    f"The current local facts do not separate the rivals decisively for digit {asked_digit}."
                )
                proof_claim = (
                    f"There is not yet a bounded local contrast proof deciding between the rival options for digit {asked_digit}."
                )

            witness_rows = [
                {
                    "kind": "DIGIT_BLOCKER",
                    "relation": payload.get("relation"),
                    "cell": payload.get("cell"),
                    "digit": asked_digit
                }
                for rel, payload in (blockers.get("blockers") or {}).items()
                if payload is not None
            ]

        elif challenge_lane == "CANDIDATE_POSSIBILITY" and asked_digit in range(1, 10):
            proof_object = "CELL_CAN_BE_DIGIT"
            method_family = "PERSISTENCE_OR_INSUFFICIENCY"
            actor_model = "LOCAL_SINGLE_SCOPE"
            blockers = _digit_blockers_for_cell(grid81, cell_index, asked_digit)

            for rel, payload in (blockers.get("blockers") or {}).items():
                if payload is not None:
                    witness_rows.append({
                        "kind": "DIGIT_BLOCKER",
                        "relation": payload.get("relation"),
                        "cell": payload.get("cell"),
                        "digit": asked_digit
                    })

            if asked_digit in candidate_digits_now:
                answer_polarity = "STILL_POSSIBLE"
                local_truth_status = "STILL_OPEN_LOCALLY"
                decisive_fact = f"{asked_digit} is still present in the current candidate set for {target_cell}."
                proof_claim = (
                    f"{target_cell} can still be {asked_digit} because the local candidate state has not ruled that digit out."
                )
                proof_ladder_rows = [
                    _proof_ladder_row(
                        step_kind="NONPROOF_CLARIFICATION",
                        actor_type="CELL",
                        actor_ref=target_cell,
                        target_ref=f"candidate {asked_digit} in {target_cell}",
                        rival_digit=asked_digit,
                        supported_claim="DIGIT_STILL_ALLOWED",
                        fact_basis={
                            "current_candidates": candidate_digits_now,
                            "blocker_found": False
                        },
                        spoken_line_seed=f"There is no current local blocker removing {asked_digit} from {target_cell}."
                    )
                ]
            else:
                answer_polarity = "NOT_LOCALLY_PROVED"
                local_truth_status = "NOT_LOCALLY_ESTABLISHED"
                nonproof_reason = "DIGIT_NOT_PRESENT_IN_CURRENT_CANDIDATE_STATE"
                decisive_fact = f"{asked_digit} is not present in the current candidate set for {target_cell}."
                proof_claim = (
                    f"{target_cell} is not currently supported as {asked_digit} by the present local candidate state."
                )
                proof_ladder_rows = [
                    _proof_ladder_row(
                        step_kind="NONPROOF_CLARIFICATION",
                        actor_type="CELL",
                        actor_ref=target_cell,
                        target_ref=f"candidate {asked_digit} in {target_cell}",
                        rival_digit=asked_digit,
                        supported_claim="DIGIT_NOT_CURRENTLY_SUPPORTED",
                        fact_basis={
                            "current_candidates": candidate_digits_now
                        },
                        spoken_line_seed=f"{asked_digit} is not currently present in the candidate set for {target_cell}."
                    )
                ]

        elif challenge_lane == "FORCEDNESS_OR_UNIQUENESS" and asked_digit in range(1, 10):
            house = _parse_house_ref(query.get("house"))
            actor_model = "SURVIVOR_LADDER"

            if house is not None:
                full_map = candidate_cells_by_house_from_masks(options_all_masks, grid81)
                h_type = house["type"]
                idx1 = int(house["index1to9"])
                by_house = (((full_map.get(str(asked_digit), {}) or {}).get("candidate_cells_by_house") or {}).get(h_type) or {})
                cell_indexes = by_house.get(str(idx1), []) or []
                candidate_cells = [_cell_ref_string(int(ci)) for ci in cell_indexes]
                focus_houses = [f"{h_type}{idx1}"]
                proof_object = "CELL_IS_ONLY_PLACE_FOR_DIGIT_IN_HOUSE"
                method_family = "HOUSE_UNIQUENESS"

                if len(candidate_cells) == 1 and target_cell in candidate_cells:
                    answer_polarity = "ONLY_PLACE"
                    local_truth_status = "PROVED_BY_HOUSE_UNIQUENESS"
                    decisive_fact = f"In {h_type} {idx1}, digit {asked_digit} has only one remaining candidate cell: {target_cell}."
                    proof_claim = f"{target_cell} is the only place left for {asked_digit} in {h_type} {idx1}."
                    survivor_summary = {
                        "surviving_digit": asked_digit,
                        "only_place_in_house": True,
                        "surviving_cell": target_cell,
                    }
                    proof_ladder_rows = [
                        _proof_ladder_row(
                            step_kind="SURVIVOR_DECLARATION",
                            actor_type="CELL",
                            actor_ref=target_cell,
                            target_ref=f"digit {asked_digit} in {h_type}{idx1}",
                            rival_digit=asked_digit,
                            supported_claim="ONLY_PLACE_REMAINS",
                            fact_basis={
                                "house": f"{h_type}{idx1}",
                                "candidate_cells": candidate_cells,
                                "surviving_cell": target_cell
                            },
                            spoken_line_seed=f"In {h_type} {idx1}, {target_cell} is the only place left for {asked_digit}."
                        )
                    ]
                else:
                    answer_polarity = "NOT_LOCALLY_PROVED"
                    local_truth_status = "NOT_LOCALLY_ESTABLISHED"
                    nonproof_reason = "HOUSE_HAS_MULTIPLE_LIVE_CELLS"
                    decisive_fact = f"In {h_type} {idx1}, digit {asked_digit} can currently go in {candidate_cells}."
                    proof_claim = (
                        f"The local uniqueness claim is not yet established because {asked_digit} still has multiple live cells in {h_type} {idx1}."
                    )
                    proof_ladder_rows = [
                        _proof_ladder_row(
                            step_kind="NONPROOF_CLARIFICATION",
                            actor_type=h_type.upper(),
                            actor_ref=f"{h_type}{idx1}",
                            target_ref=f"digit {asked_digit} in {h_type}{idx1}",
                            rival_digit=asked_digit,
                            supported_claim="MULTIPLE_CELLS_STILL_LIVE",
                            fact_basis={
                                "house": f"{h_type}{idx1}",
                                "candidate_cells": candidate_cells
                            },
                            spoken_line_seed=f"{asked_digit} still has multiple candidate cells in {h_type} {idx1}: {candidate_cells}."
                        )
                    ]
            else:
                proof_object = "DIGIT_SURVIVES_RIVAL_CANDIDATES_IN_CELL"
                method_family = "RIVAL_ELIMINATION_LADDER"

                if asked_digit in candidate_digits_now:
                    rivals = [d for d in candidate_digits_now if d != asked_digit]
                    answer_polarity = "FORCED" if len(candidate_digits_now) == 1 else "SURVIVES_RIVALS"
                    local_truth_status = (
                        "PROVED_BY_CELL_SURVIVOR"
                        if len(candidate_digits_now) == 1 else
                        "PARTIALLY_SUPPORTED_BY_CELL_SURVIVOR"
                    )
                    decisive_fact = (
                        f"{target_cell} currently carries candidates {candidate_digits_now}, with {asked_digit} surviving inside the cell's live candidate set."
                    )
                    proof_claim = (
                        f"The local survivor story is about rival candidates inside {target_cell}; {asked_digit} remains alive while the cell narrows around it."
                    )
                    survivor_summary = {
                        "surviving_digit": asked_digit,
                        "rival_digits": rivals,
                        "only_place_in_house": False,
                        "surviving_cell": target_cell,
                    }
                    proof_ladder_rows = [
                        _proof_ladder_row(
                            step_kind="CELL_SURVIVOR_SCOPE",
                            actor_type="CELL",
                            actor_ref=target_cell,
                            target_ref=f"digit {asked_digit} in {target_cell}",
                            rival_digit=asked_digit,
                            supported_claim="CELL_SURVIVOR_UNDER_REVIEW",
                            fact_basis={
                                "current_candidates": candidate_digits_now,
                                "rival_digits": rivals
                            },
                            spoken_line_seed=f"In {target_cell}, the rival digits are {rivals}, and {asked_digit} is the candidate being tested to survive."
                        )
                    ]
                else:
                    answer_polarity = "NOT_LOCALLY_PROVED"
                    local_truth_status = "NOT_LOCALLY_ESTABLISHED"
                    nonproof_reason = "DIGIT_NOT_PRESENT_IN_CURRENT_CANDIDATE_STATE"
                    decisive_fact = f"{asked_digit} is not currently among the live candidates in {target_cell}."
                    proof_claim = (
                        f"The requested survivor claim is not established because {asked_digit} is not currently present in {target_cell}'s candidate state."
                    )
                    proof_ladder_rows = [
                        _proof_ladder_row(
                            step_kind="NONPROOF_CLARIFICATION",
                            actor_type="CELL",
                            actor_ref=target_cell,
                            target_ref=f"digit {asked_digit} in {target_cell}",
                            rival_digit=asked_digit,
                            supported_claim="DIGIT_NOT_CURRENTLY_SUPPORTED",
                            fact_basis={
                                "current_candidates": candidate_digits_now
                            },
                            spoken_line_seed=f"{asked_digit} is not currently present in the candidate set for {target_cell}."
                        )
                    ]

        elif challenge_lane == "HOUSE_BLOCKER" and asked_digit in range(1, 10):
            proof_object = "HOUSE_BLOCKS_DIGIT_FOR_TARGET"
            method_family = "DIRECT_CONTRADICTION"
            actor_model = "LOCAL_CONTRADICTION_SPOTLIGHT"
            blockers = _digit_blockers_for_cell(grid81, cell_index, asked_digit)

            for rel, payload in (blockers.get("blockers") or {}).items():
                if payload is not None:
                    witness_rows.append({
                        "kind": "DIGIT_BLOCKER",
                        "relation": payload.get("relation"),
                        "cell": payload.get("cell"),
                        "digit": asked_digit
                    })

            if witness_rows:
                answer_polarity = "RULED_OUT"
                local_truth_status = "DIRECTLY_PROVED"
                decisive_fact = f"The blocking story for {asked_digit} at {target_cell} is carried by linked-house blockers."
                proof_claim = f"The local blocker evidence explains why {asked_digit} cannot survive at {target_cell}."
                proof_ladder_rows = _blocker_rows_for_digit_target(target_cell, asked_digit, witness_rows)
            else:
                method_family = "PERSISTENCE_OR_INSUFFICIENCY"
                answer_polarity = "NOT_LOCALLY_PROVED"
                local_truth_status = "NOT_LOCALLY_ESTABLISHED"
                nonproof_reason = "NO_DIRECT_LOCAL_BLOCKER_PRESENT"
                decisive_fact = f"There is no direct solved-peer blocker currently proving {asked_digit} cannot survive at {target_cell}."
                proof_claim = f"The blocker story for {asked_digit} at {target_cell} is not established by solved peers alone."
                proof_ladder_rows = [
                    _proof_ladder_row(
                        step_kind="NONPROOF_CLARIFICATION",
                        actor_type="CELL",
                        actor_ref=target_cell,
                        target_ref=f"candidate {asked_digit} in {target_cell}",
                        rival_digit=asked_digit,
                        supported_claim="NO_DIRECT_LOCAL_BLOCKER",
                        fact_basis={
                            "blocker_found": False
                        },
                        spoken_line_seed=f"There is no direct local blocker currently ruling {asked_digit} out of {target_cell}."
                    )
                ]

        elif challenge_lane == "NON_PROOF_OR_NOT_ESTABLISHED" and asked_digit in range(1, 10):
            proof_object = "LOCAL_PROOF_INSUFFICIENT"
            method_family = "PERSISTENCE_OR_INSUFFICIENCY"
            actor_model = "HONEST_INSUFFICIENCY_ANSWER"

            blockers = _digit_blockers_for_cell(grid81, cell_index, asked_digit)
            for rel, payload in (blockers.get("blockers") or {}).items():
                if payload is not None:
                    witness_rows.append({
                        "kind": "DIGIT_BLOCKER",
                        "relation": payload.get("relation"),
                        "cell": payload.get("cell"),
                        "digit": asked_digit
                    })

            answer_polarity = "NOT_LOCALLY_PROVED"
            local_truth_status = "NOT_LOCALLY_ESTABLISHED"
            nonproof_reason = "ROUTE_DEPENDENT_NOT_LOCALLY_VISIBLE" if anchor.get("step_id") else "NO_DIRECT_LOCAL_BLOCKER_PRESENT"
            decisive_fact = (
                f"The current bounded local facts do not by themselves prove the challenged claim for {asked_digit} at {target_cell}."
            )
            proof_claim = (
                f"This specific local challenge is not established on the currently visible local facts alone."
            )
            proof_ladder_rows = [
                _proof_ladder_row(
                    step_kind="NONPROOF_CLARIFICATION",
                    actor_type="CELL",
                    actor_ref=target_cell,
                    target_ref=f"candidate {asked_digit} in {target_cell}",
                    rival_digit=asked_digit,
                    supported_claim="LOCAL_PROOF_INSUFFICIENT",
                    fact_basis={
                        "current_candidates": candidate_digits_now,
                        "direct_blocker_found": bool(blockers.get("blocked")),
                        "anchor_step_id": anchor.get("step_id"),
                    },
                    spoken_line_seed=f"The local evidence here is bounded, and that bounded evidence does not yet prove the challenged claim."
                )
            ]

        elif challenge_lane == "ELIMINATION_LEGITIMACY" and asked_digit in range(1, 10):
            proof_object = "ELIMINATION_IS_LEGAL"
            actor_model = "LOCAL_CONTRADICTION_SPOTLIGHT"
            blockers = _digit_blockers_for_cell(grid81, cell_index, asked_digit)

            for rel, payload in (blockers.get("blockers") or {}).items():
                if payload is not None:
                    witness_rows.append({
                        "kind": "DIGIT_BLOCKER",
                        "relation": payload.get("relation"),
                        "cell": payload.get("cell"),
                        "digit": asked_digit
                    })

            if blockers.get("blocked"):
                method_family = "ACTION_LEGITIMACY"
                answer_polarity = "LEGAL"
                local_truth_status = "DIRECTLY_PROVED"
                decisive_fact = f"The elimination of {asked_digit} from {target_cell} is justified by a direct house conflict."
                proof_claim = (
                    f"Removing {asked_digit} from {target_cell} is legal because Sudoku already places {asked_digit} in a linked house."
                )
                proof_ladder_rows = _blocker_rows_for_digit_target(target_cell, asked_digit, witness_rows)
            else:
                method_family = "PERSISTENCE_OR_INSUFFICIENCY"
                answer_polarity = "NOT_LOCALLY_PROVED"
                local_truth_status = "NOT_LOCALLY_ESTABLISHED"
                nonproof_reason = "ELIMINATION_SUPPORT_NOT_LOCALLY_VISIBLE"
                decisive_fact = f"There is no direct local blocker currently justifying the elimination of {asked_digit} from {target_cell}."
                proof_claim = (
                    f"On the current bounded local facts alone, the elimination of {asked_digit} from {target_cell} is not directly established."
                )
                proof_ladder_rows = [
                    _proof_ladder_row(
                        step_kind="NONPROOF_CLARIFICATION",
                        actor_type="CELL",
                        actor_ref=target_cell,
                        target_ref=f"candidate {asked_digit} in {target_cell}",
                        rival_digit=asked_digit,
                        supported_claim="ELIMINATION_NOT_LOCALLY_ESTABLISHED",
                        fact_basis={
                            "current_candidates": candidate_digits_now,
                            "blocker_found": False
                        },
                        spoken_line_seed=f"There is no direct local blocker currently justifying that removal."
                    )
                ]

        elif asked_digit in range(1, 10):
            blockers = _digit_blockers_for_cell(grid81, cell_index, asked_digit)

            for rel, payload in (blockers.get("blockers") or {}).items():
                if payload is not None:
                    witness_rows.append({
                        "kind": "DIGIT_BLOCKER",
                        "relation": payload.get("relation"),
                        "cell": payload.get("cell"),
                        "digit": asked_digit
                    })

            if blockers.get("blocked"):
                proof_object = "CELL_CANNOT_BE_DIGIT"
                method_family = "DIRECT_CONTRADICTION"
                actor_model = "LOCAL_CONTRADICTION_SPOTLIGHT"
                answer_polarity = "RULED_OUT"
                local_truth_status = "DIRECTLY_PROVED"
                decisive_fact = f"{target_cell} sees an existing {asked_digit} in one of its houses."
                proof_claim = (
                    f"{target_cell} cannot be {asked_digit} because Sudoku already places {asked_digit} in a linked house."
                )
                proof_ladder_rows = _blocker_rows_for_digit_target(target_cell, asked_digit, witness_rows)
            else:
                proof_object = "LOCAL_PROOF_INSUFFICIENT"
                method_family = "PERSISTENCE_OR_INSUFFICIENCY"
                actor_model = "HONEST_INSUFFICIENCY_ANSWER"
                answer_polarity = "NOT_LOCALLY_PROVED"
                local_truth_status = "NOT_LOCALLY_ESTABLISHED"
                nonproof_reason = "NO_DIRECT_LOCAL_BLOCKER_PRESENT"
                decisive_fact = f"{asked_digit} is not immediately blocked by a solved peer for {target_cell}."
                proof_claim = (
                    f"There is no direct solved-peer blocker proving {target_cell} cannot be {asked_digit}."
                )
                proof_ladder_rows = [
                    _proof_ladder_row(
                        step_kind="NONPROOF_CLARIFICATION",
                        actor_type="CELL",
                        actor_ref=target_cell,
                        target_ref=f"candidate {asked_digit} in {target_cell}",
                        rival_digit=asked_digit,
                        supported_claim="NO_DIRECT_LOCAL_BLOCKER",
                        fact_basis={
                            "current_candidates": candidate_digits_now,
                            "blocker_found": False
                        },
                        spoken_line_seed=f"There is no direct local blocker currently ruling {asked_digit} out of {target_cell}."
                    )
                ]
        else:
            candidates_now = target_before.get("digits") or []
            proof_object = "PROOF_SCOPE_ORIGIN"
            method_family = "PERSISTENCE_OR_INSUFFICIENCY"
            actor_model = "HONEST_INSUFFICIENCY_ANSWER"
            answer_polarity = "NOT_LOCALLY_PROVED"
            local_truth_status = "NOT_LOCALLY_ESTABLISHED"
            nonproof_reason = "TARGET_DIGIT_MISSING"
            decisive_fact = f"{target_cell} currently has candidates {candidates_now}."
            proof_claim = f"The local proof question is anchored on {target_cell} and its current candidate state."
            proof_ladder_rows = [
                _proof_ladder_row(
                    step_kind="NONPROOF_CLARIFICATION",
                    actor_type="CELL",
                    actor_ref=target_cell,
                    target_ref=target_cell or "target_cell",
                    rival_digit=None,
                    supported_claim="TARGET_STATE_READOUT",
                    fact_basis={
                        "current_candidates": candidates_now
                    },
                    spoken_line_seed=f"{target_cell} currently has candidates {candidates_now}."
                )
            ]

    target_after: Dict[str, Any] = {}
    if target_before:
        before_digits = list(target_before.get("digits") or [])
        if answer_polarity in ("ONLY_PLACE", "FORCED") and asked_digit in range(1, 10):
            target_after = {"cell": target_cell, "digits": [asked_digit]}
        elif target_digit in range(1, 10) and target_digit in before_digits:
            if not survivor_summary:
                survivor_summary = {"surviving_digit": target_digit}
            target_after = {"cell": target_cell, "digits": [target_digit]}
        else:
            target_after = {"cell": target_cell, "digits": before_digits}






    scope = {
        "kind":
            "HOUSE_ALREADY_OCCUPIED" if proof_object == "HOUSE_ALREADY_CONTAINS_DIGIT" else
            "FILLED_CELL_FACT" if proof_object == "CELL_ALREADY_FILLED" else
            "TARGET_LOCAL_PROOF",
        "ref":
            (focus_houses[0] if focus_houses else target_cell) if proof_object == "HOUSE_ALREADY_CONTAINS_DIGIT" else
            target_cell,
        "cells": [target_cell] if target_cell else [],
        "houses": focus_houses,
        "digits": [d for d in [asked_digit, target_digit, rival_digit] if isinstance(d, int) and d in range(1, 10)]
    }

    if proof_object == "HOUSE_ALREADY_CONTAINS_DIGIT" and house is not None and asked_digit in range(1, 10) and target_cell:
        local_proof_geometry = _house_digit_already_placed_geometry(
            grid81=grid81,
            house=house,
            digit=asked_digit,
            existing_cell=target_cell,
        )
    elif proof_object == "CELL_ALREADY_FILLED" and cell_index is not None:
        local_proof_geometry = _cell_already_filled_geometry(
            grid81=grid81,
            cell_index=cell_index,
            asked_digit=asked_digit,
        )
    else:
        local_proof_geometry = _build_local_proof_geometry_for_move_proof(
            grid81=grid81,
            options_all_masks=options_all_masks,
            query=query,
            anchor=anchor,
            proof_object=proof_object,
            challenge_lane=challenge_lane,
            cell_index=cell_index,
            target_cell=target_cell,
            asked_digit=asked_digit,
            rival_cell=rival_cell,
            technique_legitimacy=technique_legitimacy,
            focus_cells=focus_cells,
            focus_houses=focus_houses,
        )

    narrative_archetype = _move_proof_archetype_for_truth(
        method_family=method_family,
        proof_object=proof_object,
        challenge_lane=challenge_lane,
        answer_polarity=answer_polarity,
        nonproof_reason=nonproof_reason,
        rival_cell=rival_cell,
        claimed_technique_id=claimed_technique_id,
        geometry_kind=local_proof_geometry.get("geometry_kind"),
    )

    if narrative_archetype == "LOCAL_PERMISSIBILITY_SCAN":
        staged_rows = _local_permissibility_stage_rows(
            local_proof_geometry=local_proof_geometry,
            target_cell=target_cell,
            asked_digit=asked_digit,
        )
        if staged_rows:
            proof_ladder_rows = staged_rows





    doctrine_id = _move_proof_doctrine_for_archetype(narrative_archetype)
    overlay_story_kind = _move_proof_overlay_story_kind_for_archetype(narrative_archetype)
    canonical_method_family = _canonical_method_family_for_move_proof(method_family)
    speech_skeleton = _move_proof_speech_skeleton_for_archetype(narrative_archetype)




    story_focus = {
        "scope": scope.get("kind") or scope.get("ref"),
        "primary_cell": target_cell,
        "asked_digit": asked_digit,
        "house_scope": focus_houses[0] if focus_houses else None,
        "focus_cells": scope.get("cells") or [],
        "focus_houses": scope.get("houses") or [],
        "overlay_story_kind": overlay_story_kind,
        "geometry_kind": local_proof_geometry.get("geometry_kind"),
    }

    story_question = {
        "user_ask_kind": ask_kind,
        "central_question": query.get("user_question") or query.get("question_text") or proof_claim,
        "local_story_question": (
            query.get("user_question")
            or query.get("question_text")
            or proof_claim
            or decisive_fact
        ),
        "challenge_lane": challenge_lane,
        "proof_object": proof_object,
        "spotlight_object":
            local_proof_geometry.get("primary_spotlight")
            or target_cell
            or (focus_houses[0] if focus_houses else None),
        "doctrine_hint": doctrine_id,
        "visible_tension_type": _visible_tension_type_for_truth(
            narrative_archetype,
            answer_polarity,
            challenge_lane,
            proof_object,
        ),
        "target_relation":
            "BLOCK_OR_RULE_OUT" if narrative_archetype == "LOCAL_CONTRADICTION_SPOTLIGHT" else
            "SCAN_LOCAL_PERMISSIBILITY" if narrative_archetype == "LOCAL_PERMISSIBILITY_SCAN" else
            "SURVIVE_OR_ONLY_PLACE" if narrative_archetype == "SURVIVOR_LADDER" else
            "COMPARE_RIVALS" if narrative_archetype == "CONTRAST_DUEL" else
            "VALIDATE_PATTERN" if narrative_archetype == "PATTERN_LEGITIMACY_CHECK" else
            "ESTABLISH_LOCAL_LIMIT",
    }



    story_actors = {
        "target_cell": target_cell,
        "asked_digit": asked_digit,
        "rival_cell": rival_cell,
        "rival_digit": rival_digit if isinstance(rival_digit, int) and rival_digit in range(1, 10) else None,
        "blocker_house": focus_houses[0] if focus_houses else None,
        "survivor_summary": survivor_summary,
        "contrast_summary": contrast_summary,
        "technique_legitimacy": technique_legitimacy,
        "actor_roles": {
            "spotlight_target": target_cell,
            "blocked_candidate": asked_digit if narrative_archetype == "LOCAL_CONTRADICTION_SPOTLIGHT" else None,
            "survivor_candidate": survivor_summary.get("surviving_digit") if isinstance(survivor_summary, dict) else None,
            "rival_candidate": rival_digit if isinstance(rival_digit, int) and rival_digit in range(1, 10) else None,
            "rival_cell": rival_cell,
            "blocking_house": focus_houses[0] if focus_houses else None,
            "claimed_pattern": (
                query.get("claimed_technique_id")
                or (technique_legitimacy.get("claimed_technique_id") if isinstance(technique_legitimacy, dict) else None)
            ),
        },
    }


    story_arc = _build_move_proof_story_arc(
        narrative_archetype=narrative_archetype,
        answer_polarity=answer_polarity,
        challenge_lane=challenge_lane,
        proof_object=proof_object,
        decisive_fact=decisive_fact,
        proof_claim=proof_claim,
        proof_ladder_rows=proof_ladder_rows,
        local_proof_geometry=local_proof_geometry,
        target_cell=target_cell,
        asked_digit=asked_digit,
    )





    micro_stage_plan = _build_move_proof_micro_stage_plan(
        story_arc=story_arc,
        story_focus=story_focus,
        story_question=story_question,
        narrative_archetype=narrative_archetype,
        local_proof_geometry=local_proof_geometry,
        proof_ladder_rows=proof_ladder_rows,
        asked_digit=asked_digit,
    )



    closure_contract = {
        "closure_mode": "AUTHORED_NATURAL_CLOSURE",
        "return_target_kind": "CURRENT_MOVE",
        "return_style": "GENTLE_ROUTE_RETURN",
        "may_offer_followup": True,
        "followup_style": "ONE_BOUNDED_OPTION",
        "may_offer_return_now": True,
        "must_not_sound_procedural": True,
        "must_not_emit_stock_handback": True,
        "must_not_use_internal_route_jargon": True,
        "must_land_local_result_before_return_offer": True,
    }



    visual_language = {
        "may_use_spotlight_language": True,
        "may_use_scene_language": True,
        "may_use_actor_language": True,
        "may_use_scan_language": narrative_archetype == "LOCAL_PERMISSIBILITY_SCAN",
        "may_use_survival_language": narrative_archetype in ("LOCAL_PERMISSIBILITY_SCAN", "SURVIVOR_LADDER"),
        "may_use_duel_language": narrative_archetype == "CONTRAST_DUEL",
        "may_use_pattern_visibility_language": narrative_archetype == "PATTERN_LEGITIMACY_CHECK",
        "must_stay_grounded_in_packet_truth": True,
    }


    result = {
        "schema_version": "normalized_detour_v1",
        "query_family": "MOVE_PROOF",
        "query_profile": question_profile,
        "challenge_lane": challenge_lane,
        "proof_object": proof_object,
        "method_family": method_family,
        "canonical_method_family": canonical_method_family,
        "narrative_archetype": narrative_archetype,
        "doctrine_id": doctrine_id,
        "speech_skeleton": speech_skeleton,
        "actor_model": actor_model,


        "story_focus": story_focus,
        "story_question": story_question,
        "story_actors": story_actors,
        "story_arc": story_arc,
        "micro_stage_plan": micro_stage_plan,
        "local_proof_geometry": local_proof_geometry,
        "local_story_question": story_question.get("local_story_question"),
        "spotlight_object": story_question.get("spotlight_object"),
        "doctrine_hint": story_question.get("doctrine_hint"),
        "proof_motion_type": story_arc.get("proof_motion_type"),
        "visible_tension_type": story_arc.get("visible_tension_type"),
        "actor_roles": story_actors.get("actor_roles"),
        "closure_contract": closure_contract,
        "visual_language": visual_language,


        "answer_truth": {
            "answer_polarity": answer_polarity,
            "short_answer": proof_claim,
            "one_sentence_claim": decisive_fact or proof_claim,
            "local_truth_status": local_truth_status,
        },
        "anchor": anchor,
        "scope": scope,
        "question": {
            "user_ask_kind": ask_kind,
            "central_question": query.get("user_question") or query.get("question_text") or proof_claim,
            "target_cell": target_cell,
            "target_digit": target_digit,
            "asked_digit": asked_digit,
            "contrast_digit": rival_digit if rival_digit in range(1, 10) else (target_digit if asked_digit != target_digit else None),
            "rival_cell": rival_cell,
            "claimed_technique_id": claimed_technique_id,
        },
        "proof_truth": {


        "claim_kind":
            "HOUSE_ALREADY_CONTAINS_DIGIT" if proof_object == "HOUSE_ALREADY_CONTAINS_DIGIT" else
            "CELL_ALREADY_FILLED" if proof_object == "CELL_ALREADY_FILLED" else
            "CELL_CAN_BE_DIGIT" if proof_object == "CELL_CAN_BE_DIGIT" else
            "HOUSE_ONLY_PLACE_FOR_DIGIT" if proof_object == "CELL_IS_ONLY_PLACE_FOR_DIGIT_IN_HOUSE" else
            "LOCAL_PROOF_INSUFFICIENT" if proof_object == "LOCAL_PROOF_INSUFFICIENT" else
            "HOUSE_BLOCKS_DIGIT_FOR_TARGET" if proof_object == "HOUSE_BLOCKS_DIGIT_FOR_TARGET" else
            "TECHNIQUE_CLAIM_IS_VALID" if proof_object == "TECHNIQUE_CLAIM_IS_VALID" else
            "ELIMINATION_IS_LEGAL" if proof_object == "ELIMINATION_IS_LEGAL" else
            "CELL_A_WINS_OVER_CELL_B_FOR_DIGIT" if proof_object == "CELL_A_WINS_OVER_CELL_B_FOR_DIGIT" else
            "DIGIT_SURVIVES_RIVAL_CANDIDATES_IN_CELL" if proof_object == "DIGIT_SURVIVES_RIVAL_CANDIDATES_IN_CELL" else
            "CELL_CANNOT_BE_DIGIT" if asked_digit in range(1, 10) else
            "WHY_TARGET",

            "challenge_lane": challenge_lane,
            "proof_object": proof_object,
            "method_family": method_family,
            "canonical_method_family": canonical_method_family,
            "narrative_archetype": narrative_archetype,
            "doctrine_id": doctrine_id,

            "speech_skeleton": speech_skeleton,
            "actor_model": actor_model,


            "story_focus": story_focus,
            "story_question": story_question,
            "story_actors": story_actors,
            "story_arc": story_arc,
            "micro_stage_plan": micro_stage_plan,
            "local_proof_geometry": local_proof_geometry,
            "local_story_question": story_question.get("local_story_question"),
            "spotlight_object": story_question.get("spotlight_object"),
            "doctrine_hint": story_question.get("doctrine_hint"),
            "proof_motion_type": story_arc.get("proof_motion_type"),
            "visible_tension_type": story_arc.get("visible_tension_type"),
            "actor_roles": story_actors.get("actor_roles"),
            "closure_contract": closure_contract,
            "visual_language": visual_language,



            "answer_polarity": answer_polarity,
            "local_truth_status": local_truth_status,
            "proof_claim": proof_claim,
            "decisive_fact": decisive_fact,

            "house_scope": focus_houses[0] if focus_houses else None,
            "elimination_kind":
                "TARGET_EXPLANATION" if challenge_lane in ("CANDIDATE_POSSIBILITY", "FORCEDNESS_OR_UNIQUENESS") else
                "TECHNIQUE_VALIDATION" if challenge_lane == "TECHNIQUE_LEGITIMACY" else
                "CELL_CANDIDATE_REMOVAL" if asked_digit in range(1, 10) else
                "TARGET_EXPLANATION",
            "target_before_state": target_before,
            "target_after_state": target_after,
            "survivor_summary": survivor_summary,
            "contrast_summary": contrast_summary,
            "technique_legitimacy": technique_legitimacy,
            "bounded_proof_rows": proof_ladder_rows,
            "witness_rows": witness_rows,
            "peer_rows": [],
            "pattern_rows": [],
            "proof_ladder": {
                "rows": proof_ladder_rows
            },
            "proof_outcome": {
                "surviving_digit": survivor_summary.get("surviving_digit"),
                "remaining_candidates": list(target_before.get("digits") or []),
                "only_place_in_house": True if survivor_summary.get("only_place_in_house") else None,
                "forced_cell_value": asked_digit if answer_polarity in ("FORCED", "ONLY_PLACE") else None,
                "winning_cell": target_cell if answer_polarity in ("ONLY_PLACE", "SURVIVES_RIVALS") else None,
                "winning_digit": asked_digit if answer_polarity in ("ONLY_PLACE", "SURVIVES_RIVALS", "STILL_POSSIBLE", "FORCED") else None,
                "nonproof_reason": nonproof_reason,
                "overlay_story_kind": overlay_story_kind,
                "challenge_lane": challenge_lane,
            },
            "allowed_stage_boundary": "STOP_BEFORE_COMMIT"
        },
        "route_context": _build_route_context(
            query,
            None
        ),

        "overlay_context": _build_overlay_context(
            focus_cells=focus_cells,
            focus_houses=focus_houses,
            reason_for_focus="move_proof_focus",
            overlay_story_kind=overlay_story_kind,
            secondary_focus_cells=list(dict.fromkeys(
                list(secondary_focus_cells or []) +
                [
                    row.get("blocker_cell")
                    for row in (local_proof_geometry.get("blocker_receipts") or [])
                    if isinstance(row, dict) and row.get("blocker_cell")
                ] +
                [
                    row.get("existing_digit_cell")
                    for row in [local_proof_geometry]
                    if isinstance(row, dict) and row.get("existing_digit_cell")
                ] +
                [
                    row.get("rival_cell")
                    for row in [local_proof_geometry]
                    if isinstance(row, dict) and row.get("rival_cell")
                ]
            )),
            highlight_roles={
                "target_cell": target_cell,
                "asked_digit": asked_digit,
                "rival_cell": rival_cell,
                "rival_digit": rival_digit,
                "house_scope": focus_houses[0] if focus_houses else None,
                "claimed_technique_id": claimed_technique_id,
                "surviving_digits": local_proof_geometry.get("surviving_digits") or [],
                "surviving_seats": local_proof_geometry.get("surviving_seats") or [],
                "primary_spotlight": local_proof_geometry.get("primary_spotlight"),
                "geometry_kind": local_proof_geometry.get("geometry_kind"),
            },
        ),

        "support": _build_support(
            has_primary_truth=bool(proof_claim or decisive_fact),
            has_bounded_evidence=bool(proof_ladder_rows or witness_rows)
        ),

        "debug": {
            "normalizer_path": "normalize_detour.py:_normalize_move_proof_query",
            "source_summary": {
                "used_next_step": bool(step_obj.get("ok")),
                "used_options_masks": True,
                "used_local_proof_geometry": True,
                "used_guard_mode": bool(guard_mode),
            },
            "sanitized_inputs": {
                "requested_cell": requested_cell,
                "requested_digit": requested_digit,
                "anchor_target_digit": anchor_target_digit,
                "challenge_lane": challenge_lane,
                "proof_object_hint": proof_object_hint,
                "claimed_technique_id": claimed_technique_id,
                "rival_cell": rival_cell,
                "rival_digit": rival_digit,
                "guard_mode": guard_mode,
                "filled_cell_digit": filled_cell_digit,
                "existing_house_digit_cell": existing_house_digit_cell,
            }
        }


    }
    return result


def _normalize_local_inspection_query(grid81: str, query: Dict[str, Any]) -> Dict[str, Any]:
    options_all_masks = _compute_options_masks(grid81)
    step_obj = _run_next_step_obj(grid81)
    anchor = _build_anchor(query, step_obj)

    profile = str(query.get("query_profile") or "").strip().upper() or "CELL_CANDIDATES"



    requested_cell = query.get("requested_cell") or query.get("anchor_target_cell")
    compare_cell = query.get("compare_cell")
    scope_kind = str(query.get("scope_kind") or "").strip().upper()
    scope_ref = query.get("scope_ref")

    focus_cells: List[str] = []
    focus_houses: List[str] = []
    candidate_state: Dict[str, Any] = {}
    digit_locations: List[Dict[str, Any]] = []
    house_missing_digits: List[int] = []
    local_delta: Dict[str, Any] = {}
    nearby_effects_summary: Dict[str, Any] = {}
    local_constraints_summary: Dict[str, Any] = {}
    shared_candidates_summary: Dict[str, Any] = {}




    if requested_cell:
        cell_index = _parse_cell_ref(requested_cell)
        if cell_index is not None:
            candidate_state = _candidate_payload_for_cell(grid81, options_all_masks, cell_index)
            focus_cells = [candidate_state["cell"]]
            focus_houses = [f"row{candidate_state['row']}", f"col{candidate_state['col']}", f"box{candidate_state['box']}"]
            local_constraints_summary = {
                candidate_state["cell"]: f"Cell currently has candidates {candidate_state.get('digits', [])}."
            }

            if profile == "COMPARE_CANDIDATES" and compare_cell:
                compare_index = _parse_cell_ref(compare_cell)
                if compare_index is not None:
                    compare_state = _candidate_payload_for_cell(grid81, options_all_masks, compare_index)
                    focus_cells = list(dict.fromkeys([candidate_state["cell"], compare_state["cell"]]))
                    shared = sorted(list(set(candidate_state.get("digits", [])) & set(compare_state.get("digits", []))))
                    shared_candidates_summary = {
                        "anchor_cell": compare_state["cell"],
                        "anchor_digits": compare_state.get("digits", []),
                        "requested_cell": candidate_state["cell"],
                        "requested_digits": candidate_state.get("digits", []),
                        "shared_digits": shared
                    }
                    local_constraints_summary[compare_state["cell"]] = f"Cell currently has candidates {compare_state.get('digits', [])}."



    house = _parse_house_ref(query.get("house"))
    if house is not None:
        full_map = candidate_cells_by_house_from_masks(options_all_masks, grid81)
        h_type = house["type"]
        idx1 = int(house["index1to9"])
        focus_houses = [f"{h_type}{idx1}"]
        for d in range(1, 10):
            digit_entry = full_map.get(str(d), {})
            by_house = ((digit_entry.get("candidate_cells_by_house") or {}).get(h_type) or {})
            cell_indexes = by_house.get(str(idx1), []) or []
            if cell_indexes:
                digit_locations.append({
                    "digit": d,
                    "cells": [_cell_ref_string(int(ci)) for ci in cell_indexes]
                })
        missing = []
        for d in range(1, 10):
            found = False
            if h_type == "row":
                for c in range(1, 10):
                    if grid81[(idx1 - 1) * 9 + (c - 1)] == str(d):
                        found = True
                        break
            elif h_type == "col":
                for r in range(1, 10):
                    if grid81[(r - 1) * 9 + (idx1 - 1)] == str(d):
                        found = True
                        break
            elif h_type == "box":
                # cheap approximation through candidate map + solved peers
                pass
            if not found:
                missing.append(d)
        house_missing_digits = missing

    if not scope_kind:
        if house is not None:
            scope_kind = house["type"].upper()
            scope_ref = f"{house['type']}{house['index1to9']}"
        elif focus_cells:
            scope_kind = "CELL"
            scope_ref = focus_cells[0]
        else:
            scope_kind = "LOCAL_REGION"
            scope_ref = query.get("anchor_target_cell")

    result = {
        "schema_version": "normalized_detour_v1",
        "query_family": "LOCAL_INSPECTION",
        "query_profile": profile,
        "anchor": anchor,
        "scope": {
            "kind": scope_kind,
            "ref": scope_ref,
            "cells": focus_cells,
            "houses": focus_houses,
            "digits": _parse_digit_list(query.get("digits"))
        },
        "question": {
            "user_ask_kind": query.get("user_ask_kind") or profile,
            "central_question": query.get("user_question") or query.get("question_text") or "Read the local board state."
        },
        "inspection_truth": {
            "state_read_kind": profile,
            "candidate_state": candidate_state,
            "digit_locations": digit_locations,
            "house_missing_digits": house_missing_digits,
            "local_delta": local_delta,
            "nearby_effects_summary": nearby_effects_summary,
            "shared_candidates_summary": shared_candidates_summary,
            "local_constraints_summary": local_constraints_summary,
            "why_it_matters": "This bounded readout explains the local state without changing the paused route."
        },
        "route_context": _build_route_context(
            query,
            "That gives the local board readout; we can return to the paused move whenever you want."
        ),
        "overlay_context": _build_overlay_context(
            focus_cells=focus_cells,
            focus_houses=focus_houses,
            reason_for_focus="local_inspection_focus"
        ),
        "support": _build_support(
            has_primary_truth=bool(candidate_state or digit_locations or local_constraints_summary),
            has_bounded_evidence=bool(candidate_state or digit_locations)
        ),
        "debug": {
            "normalizer_path": "normalize_detour.py:_normalize_local_inspection_query",
            "source_summary": {
                "used_next_step": bool(step_obj.get("ok")),
                "used_options_masks": True
            },
            "sanitized_inputs": {
                "requested_cell": requested_cell,
                "scope_kind": scope_kind,
                "scope_ref": scope_ref
            }
        }
    }
    return result


def _normalize_proposal_verdict_query(grid81: str, query: Dict[str, Any]) -> Dict[str, Any]:
    options_all_masks = _compute_options_masks(grid81)
    step_obj = _run_next_step_obj(grid81)
    anchor = _build_anchor(query, step_obj)

    requested_cell = query.get("requested_cell") or query.get("anchor_target_cell")
    cell_index = _parse_cell_ref(requested_cell)
    claimed_digits = _parse_digit_list(query.get("claimed_digits"))
    actual_digits: List[int] = []
    if cell_index is not None:
        actual_digits = _candidate_digits_before_for_cell_from_masks(options_all_masks, cell_index)

    extra_digits = [d for d in claimed_digits if d not in actual_digits]
    missing_digits = [d for d in actual_digits if d not in claimed_digits]

    if claimed_digits == actual_digits and claimed_digits:
        verdict = "VALID"
    elif claimed_digits and (extra_digits or missing_digits):
        verdict = "INVALID"
    elif not claimed_digits:
        verdict = "UNKNOWN"
    else:
        verdict = "PARTIALLY_VALID"

    proposal_text = query.get("proposal_text") or query.get("user_question") or query.get("question_text")
    proposal_kind = str(query.get("query_profile") or "").strip().upper() or "GENERAL_REASONING_CHECK"

    what_is_correct: List[str] = []
    what_is_incorrect: List[str] = []
    missing_condition = ""

    if verdict == "VALID":
        what_is_correct.append(f"The claimed digits match the current candidate set {actual_digits}.")
    elif verdict == "INVALID":
        if extra_digits:
            what_is_incorrect.append(f"These claimed digits are not actually available: {extra_digits}.")
        if missing_digits:
            what_is_incorrect.append(f"These current candidates were omitted: {missing_digits}.")
        missing_condition = "The proposal must match the true current candidate state before it can support a valid conclusion."
    elif verdict == "PARTIALLY_VALID":
        what_is_correct.append("Part of the proposal overlaps with the true candidate state.")
        if missing_digits:
            what_is_incorrect.append(f"Some current candidates are still missing from the proposal: {missing_digits}.")
        missing_condition = "The proposal needs the full current candidate picture."

    focus_cells = [_cell_ref_string(cell_index)] if cell_index is not None else []
    focus_houses: List[str] = []

    result = {
        "schema_version": "normalized_detour_v1",
        "query_family": "PROPOSAL_VERDICT",
        "query_profile": proposal_kind,
        "anchor": anchor,
        "scope": {
            "kind": "USER_PROPOSAL_SCOPE",
            "ref": proposal_text or requested_cell,
            "cells": focus_cells,
            "houses": focus_houses,
            "digits": claimed_digits
        },
        "question": {
            "user_ask_kind": "REQUEST_REASONING_CHECK",
            "central_question": "Is my idea valid?",
            "proposal_text": proposal_text,
            "proposal_summary": proposal_text or "User proposal requires verdict."
        },
        "proposal_truth": {
            "proposal_kind": proposal_kind,
            "verdict": verdict,
            "verdict_reason": (
                "The proposal matches the current candidate state."
                if verdict == "VALID" else
                "The proposal does not match the current candidate state."
                if verdict == "INVALID" else
                "The proposal is incomplete relative to the current candidate state."
                if verdict == "PARTIALLY_VALID" else
                "Not enough concrete proposal detail was available to validate."
            ),
            "what_is_correct": what_is_correct,
            "what_is_incorrect": what_is_incorrect,
            "missing_condition": missing_condition,
            "route_alignment": "ON_CURRENT_ROUTE" if verdict == "VALID" else "UNKNOWN",
            "support_rows": [{
                "cell": _cell_ref_string(cell_index),
                "actual_digits": actual_digits,
                "claimed_digits": claimed_digits
            }] if cell_index is not None else [],
            "closest_valid_version": {
                "cell": _cell_ref_string(cell_index) if cell_index is not None else None,
                "actual_digits": actual_digits
            } if cell_index is not None else {},
            "recommended_next_action": "Return to the paused move after checking the candidate truth."
        },
        "route_context": _build_route_context(
            query,
            "That checks your idea against the current board truth; the paused route is still available."
        ),
        "overlay_context": _build_overlay_context(
            focus_cells=focus_cells,
            focus_houses=focus_houses,
            reason_for_focus="proposal_verdict_focus"
        ),
        "support": _build_support(
            has_primary_truth=bool(verdict),
            has_bounded_evidence=cell_index is not None
        ),
        "debug": {
            "normalizer_path": "normalize_detour.py:_normalize_proposal_verdict_query",
            "source_summary": {
                "used_next_step": bool(step_obj.get("ok")),
                "used_options_masks": True
            },
            "sanitized_inputs": {
                "requested_cell": requested_cell,
                "claimed_digits": claimed_digits
            }
        }
    }
    return result


# ============================================================================
# Public entrypoint
# ============================================================================

def normalize_detour_query(payload_json: str) -> str:
    try:
        payload = json.loads(payload_json or "{}")
        grid81 = str(payload.get("grid81") or "").strip()
        query = payload.get("query") or {}
        family = str(query.get("query_family") or "").strip().upper()

        if len(grid81) != 81:
            return json.dumps({
                "ok": False,
                "status": "error",
                "op": "normalize_detour_query",
                "error": {
                    "code": "invalid_grid81",
                    "msg": f"grid81 must be length 81, got {len(grid81)}"
                }
            }, ensure_ascii=False)

        if family == "MOVE_PROOF":
            result = _normalize_move_proof_query(grid81, query)
        elif family == "LOCAL_INSPECTION":
            result = _normalize_local_inspection_query(grid81, query)
        elif family == "PROPOSAL_VERDICT":
            result = _normalize_proposal_verdict_query(grid81, query)
        else:
            return json.dumps({
                "ok": False,
                "status": "error",
                "op": "normalize_detour_query",
                "error": {
                    "code": "unsupported_query_family",
                    "msg": f"Unsupported detour query_family: {family}"
                }
            }, ensure_ascii=False)

        return json.dumps({
            "ok": True,
            "status": "ok",
            "op": "normalize_detour_query",
            "result": result
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "ok": False,
            "status": "error",
            "op": "normalize_detour_query",
            "error": {
                "code": "normalize_detour_exception",
                "msg": str(e)[:240]
            }
        }, ensure_ascii=False)