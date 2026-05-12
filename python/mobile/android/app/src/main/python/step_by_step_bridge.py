# step_by_step_bridge.py
# Chaquopy entrypoint used by Android to request ONE next human-style step.
#
# ✅ V2-only: returns SolveStepV2 envelope (single source of truth)
#
# Requires: generator/ folder present in the same Python src root.
# Requires: colorama installed (or vendored) because algo_human imports it.

import json
import hashlib
from typing import Any, Dict, List, Optional

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
from normalize_detour import normalize_detour_query

from generator.model import Instance, EMPTY_CHAR
from generator.algo_human import solve_using_human_techniques, TECHNIQUES
from generator.techniques.options import determine_options_per_cell

DIGITS = "123456789"


def _sha12(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


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
    inst.chars = set(DIGITS)  # avoids engine "dummy char" behavior
    return inst


def _instance_to_grid81(inst: Instance) -> str:
    out = []
    for r in range(9):
        for c in range(9):
            ch = inst[r][c]
            out.append("." if ch == EMPTY_CHAR else str(ch))
    return "".join(out)


def next_step(payload_json: str) -> str:
    """
    Input JSON:
      {
        "grid81": "....81 chars....",
        "options": {
          "use_cleanup_method": true,
          "include_magic_technique": false,
          "step_style": "full"      // "full"|"mini"
        }
      }

    Output JSON (V2-only):
      {
        "ok": true|false,
        "status": "ok"|"stuck"|"solved"|"error",
        "step": <SolveStepV2>,          // present when status="ok"
        "error": {"code":"...","msg":"..."}   // only if ok=false
      }
    """
    try:
        payload = json.loads(payload_json or "{}")
        grid81 = (payload.get("grid81") or "").strip()
        options = payload.get("options") or {}

        inst = _grid81_to_instance(grid81)

        use_cleanup_method = bool(options.get("use_cleanup_method", True))
        include_magic_technique = bool(options.get("include_magic_technique", False))
        step_style = str(options.get("step_style", "full")).strip().lower()
        if step_style not in ("full", "mini"):
            step_style = "full"

        solved_inst, (_counts, logs) = solve_using_human_techniques(
            inst,
            use_techniques=TECHNIQUES,             # ✅ ALL techniques enabled
            use_cleanup_method=use_cleanup_method,
            include_magic_technique=include_magic_technique,
            magic_solution=None,
            max_number_iterations=1,               # ✅ ONE step only
            show_logs=False
        )

        before = grid81
        after = _instance_to_grid81(solved_inst)

        if before == after:
            if "." not in before:
                return json.dumps({"ok": True, "status": "solved"}, ensure_ascii=False)
            return json.dumps({"ok": True, "status": "stuck"}, ensure_ascii=False)

        steps = (logs or {}).get("steps") or []
        if not steps:
            return json.dumps({
                "ok": False,
                "status": "error",
                "error": {"code": "missing_step_log", "msg": "Engine progressed but produced no step log."}
            }, ensure_ascii=False)

        logged_step = steps[0]
        step_v2 = normalize_logged_step(
            logged_step=logged_step,
            step_index=1,
            include_grids=True,
            style=step_style
        )

        return json.dumps({"ok": True, "status": "ok", "step": step_v2}, ensure_ascii=False)

    except Exception as e:
            return json.dumps({
                "ok": False,
                "status": "error",
                "error": {"code": "engine_exception", "msg": str(e)[:240]}
            }, ensure_ascii=False)


def _cell_ref_string(cell_index: int) -> str:
    r, c = cell_index_to_rc(int(cell_index))
    return f"r{r}c{c}"


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


def _parse_house_ref(house: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(house, dict):
        return None

    h_type = str(house.get("type") or "").strip().lower()
    try:
        idx1 = int(house.get("index1to9"))
    except Exception:
        return None

    if h_type not in {"row", "col", "box"}:
        return None
    if idx1 not in range(1, 10):
        return None

    return {"type": h_type, "index1to9": idx1}


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
        "digits": digits
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


def _run_next_step_obj(grid81: str) -> Dict[str, Any]:
    raw = next_step(json.dumps({
        "grid81": grid81,
        "options": {
            "use_cleanup_method": True,
            "include_magic_technique": False,
            "step_style": "full"
        }
    }, ensure_ascii=False))
    try:
        return json.loads(raw or "{}")
    except Exception:
        return {
            "ok": False,
            "status": "error",
            "error": {"code": "bad_next_step_json", "msg": "next_step returned invalid JSON"}
        }


def _normalize_technique_hint(raw: Any) -> str:
    return str(raw or "").strip().lower()


def _next_step_summary(step_obj: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "technique": step_obj.get("technique"),
        "label": step_obj.get("label"),
        "target_cell": step_obj.get("target_cell"),
        "target_digit": step_obj.get("target_digit"),
        "focus_cell": step_obj.get("focus_cell"),
        "focus_digit": step_obj.get("focus_digit"),
    }


def _technique_matches_hint(step_obj: Dict[str, Any], hint: str) -> bool:
    h = _normalize_technique_hint(hint)
    if not h:
        return False

    candidates = [
        _normalize_technique_hint(step_obj.get("technique")),
        _normalize_technique_hint(step_obj.get("label")),
    ]
    return h in [c for c in candidates if c]


def _cell_matches_scope(cell_ref: Any, scope: Dict[str, Any]) -> bool:
    if not isinstance(scope, dict):
        return True
    ci = _parse_cell_ref(cell_ref)
    if ci is None:
        return False
    r, c = cell_index_to_rc(ci)
    b = box_index_1to9(r, c)

    scope_type = str(scope.get("type") or "").strip().lower()
    try:
        idx1 = int(scope.get("index1to9"))
    except Exception:
        idx1 = -1

    if scope_type == "row":
        return r == idx1
    if scope_type == "col":
        return c == idx1
    if scope_type == "box":
        return b == idx1
    if scope_type == "cell":
        scope_ci = _parse_cell_ref(scope.get("cell"))
        return scope_ci == ci
    return True


def _summarize_step_for_scope(step_obj: Dict[str, Any], scope: Dict[str, Any]) -> Dict[str, Any]:
    summary = _next_step_summary(step_obj)
    target_cell = summary.get("target_cell")
    focus_cell = summary.get("focus_cell")
    summary["in_scope"] = _cell_matches_scope(target_cell, scope) or _cell_matches_scope(focus_cell, scope)
    return summary


def _normalize_cell_ref_string(raw: Any) -> str:
    ci = _parse_cell_ref(raw)
    if ci is None:
        return ""
    return _cell_ref_string(ci)


def _compare_current_route_to_next_step(
    current_route: Dict[str, Any],
    next_step: Dict[str, Any],
    scope: Dict[str, Any],
) -> Dict[str, Any]:
    current_technique = _normalize_technique_hint(current_route.get("technique"))
    current_target_cell = _normalize_cell_ref_string(current_route.get("target_cell"))
    current_focus_cell = _normalize_cell_ref_string(current_route.get("focus_cell"))

    next_technique = _normalize_technique_hint(next_step.get("technique"))
    next_target_cell = _normalize_cell_ref_string(next_step.get("target_cell"))
    next_focus_cell = _normalize_cell_ref_string(next_step.get("focus_cell"))

    current_in_scope = _cell_matches_scope(current_target_cell or current_focus_cell, scope)
    next_in_scope = bool(next_step.get("in_scope"))

    relation = "UNKNOWN"
    if current_technique and next_technique and current_technique == next_technique:
        if current_target_cell and current_target_cell == next_target_cell:
            relation = "SAME_ROUTE"
        elif current_focus_cell and current_focus_cell == next_focus_cell:
            relation = "SAME_ROUTE"
        else:
            relation = "SAME_TECHNIQUE_DIFFERENT_TARGET"
    elif current_in_scope and next_in_scope:
        relation = "DIFFERENT_ROUTE_SAME_SCOPE"
    elif current_in_scope and not next_in_scope:
        relation = "CURRENT_ROUTE_IN_SCOPE_NEXT_STEP_OUT_OF_SCOPE"
    elif not current_in_scope and next_in_scope:
        relation = "NEXT_STEP_IN_SCOPE_CURRENT_ROUTE_OUT_OF_SCOPE"

    return {
        "relation": relation,
        "current_route_in_scope": current_in_scope,
        "next_step_in_scope": next_in_scope,
        "current_route": {
            "technique": current_route.get("technique"),
            "target_cell": current_route.get("target_cell"),
            "focus_cell": current_route.get("focus_cell"),
        },
        "next_step": next_step,
    }


def _scoped_support_snapshot(
    grid81: str,
    options_all_masks: Dict[str, int],
    scope: Dict[str, Any],
) -> Dict[str, Any]:
    scope_type = str(scope.get("type") or "").strip().lower()

    if scope_type == "cell":
        ci = _parse_cell_ref(scope.get("cell"))
        if ci is None:
            return {"scope_type": scope_type, "support": {}}
        return {
            "scope_type": scope_type,
            "support": {
                "cell_candidates": _candidate_payload_for_cell(grid81, options_all_masks, ci)
            }
        }

    if scope_type in {"row", "col", "box"}:
        try:
            idx1 = int(scope.get("index1to9"))
        except Exception:
            return {"scope_type": scope_type, "support": {}}

        full_map = candidate_cells_by_house_from_masks(options_all_masks, grid81)
        per_digit: Dict[str, List[str]] = {}
        for d in range(1, 10):
            digit_entry = full_map.get(str(d), {})
            by_house = ((digit_entry.get("candidate_cells_by_house") or {}).get(scope_type) or {})
            cell_indexes = by_house.get(str(idx1), []) or []
            per_digit[str(d)] = [_cell_ref_string(int(ci)) for ci in cell_indexes]

        return {
            "scope_type": scope_type,
            "support": {
                "house": {"type": scope_type, "index1to9": idx1},
                "house_candidate_map": per_digit
            }
        }

    return {
        "scope_type": scope_type or "global",
        "support": {}
    }



def detour_query(payload_json: str) -> str:
    """
    Generic solver-backed detour query bridge.

        Supported ops in SV-2 / Wave-1 permanent-design:
          - ping
          - bridge_info
          - get_cell_candidates
          - get_cells_candidates
          - get_house_candidate_map
          - get_digit_blockers_for_cell
          - validate_candidate_claim
          - get_next_step_summary
          - compare_requested_technique_to_next_step
          - check_technique_in_scope
          - search_local_moves
          - compare_current_route_to_scope
          - normalize_detour_query
    """
    try:
        payload = json.loads(payload_json or "{}")
        op = str(payload.get("op") or "").strip()
        grid81 = str(payload.get("grid81") or "").strip()
        query = payload.get("query") or {}

        if not op:
            return json.dumps({
                "ok": False,
                "status": "error",
                "error": {
                    "code": "missing_op",
                    "msg": "detour_query requires non-empty 'op'."
                }
            }, ensure_ascii=False)

        if op == "ping":
            return json.dumps({
                "ok": True,
                "status": "ok",
                "op": op,
                "result": {
                    "bridge": "detour_query",
                    "bridge_version": 2,
                    "supported_ops": [
                        "ping",
                        "bridge_info",
                        "get_cell_candidates",
                        "get_cells_candidates",
                        "get_house_candidate_map",
                        "get_digit_blockers_for_cell",
                        "validate_candidate_claim",
                        "get_next_step_summary",
                        "compare_requested_technique_to_next_step",
                        "check_technique_in_scope",
                        "search_local_moves",
                        "compare_current_route_to_scope",
                        "normalize_detour_query"
                    ]
                }
            }, ensure_ascii=False)

        if op == "bridge_info":
            return json.dumps({
                "ok": True,
                "status": "ok",
                "op": op,
                "result": {
                    "bridge": "detour_query",
                    "bridge_version": 2,
                    "step_entrypoint": "next_step",
                    "supported_ops": [
                        "ping",
                        "bridge_info",
                        "get_cell_candidates",
                        "get_cells_candidates",
                        "get_house_candidate_map",
                        "get_digit_blockers_for_cell",
                        "validate_candidate_claim",
                        "get_next_step_summary",
                        "compare_requested_technique_to_next_step",
                        "check_technique_in_scope",
                        "search_local_moves",
                        "compare_current_route_to_scope",
                        "normalize_detour_query"
                    ]
                }
            }, ensure_ascii=False)


        if op == "normalize_detour_query":
            return normalize_detour_query(payload_json)

        if len(grid81) != 81:
            return json.dumps({
                "ok": False,
                "status": "error",
                "op": op,
                "error": {
                    "code": "invalid_grid81",
                    "msg": f"grid81 must be length 81, got {len(grid81)}"
                }
            }, ensure_ascii=False)

        options_all_masks = _compute_options_masks(grid81)

        if op == "get_cell_candidates":
            cell_index = _parse_cell_ref(query.get("cell"))
            if cell_index is None:
                return json.dumps({
                    "ok": False,
                    "status": "error",
                    "op": op,
                    "error": {"code": "missing_cell", "msg": "query.cell is required"}
                }, ensure_ascii=False)

            result = _candidate_payload_for_cell(grid81, options_all_masks, cell_index)
            return json.dumps({
                "ok": True,
                "status": "ok",
                "op": op,
                "result": result
            }, ensure_ascii=False)

        if op == "get_cells_candidates":
            cells = query.get("cells") or []
            out: List[Dict[str, Any]] = []
            for raw in cells:
                ci = _parse_cell_ref(raw)
                if ci is None:
                    continue
                out.append(_candidate_payload_for_cell(grid81, options_all_masks, ci))

            return json.dumps({
                "ok": True,
                "status": "ok",
                "op": op,
                "result": {
                    "cells": out,
                    "count": len(out)
                }
            }, ensure_ascii=False)

        if op == "get_house_candidate_map":
            house = _parse_house_ref(query.get("house"))
            if house is None:
                return json.dumps({
                    "ok": False,
                    "status": "error",
                    "op": op,
                    "error": {
                        "code": "missing_house",
                        "msg": "query.house {type,index1to9} is required"
                    }
                }, ensure_ascii=False)

            full_map = candidate_cells_by_house_from_masks(options_all_masks, grid81)
            h_type = house["type"]
            idx1 = int(house["index1to9"])

            per_digit: Dict[str, List[str]] = {}
            for d in range(1, 10):
                digit_entry = full_map.get(str(d), {})
                by_house = ((digit_entry.get("candidate_cells_by_house") or {}).get(h_type) or {})
                cell_indexes = by_house.get(str(idx1), []) or []
                per_digit[str(d)] = [_cell_ref_string(int(ci)) for ci in cell_indexes]

            return json.dumps({
                "ok": True,
                "status": "ok",
                "op": op,
                "result": {
                    "house": house,
                    "map": per_digit
                }
            }, ensure_ascii=False)

        if op == "get_digit_blockers_for_cell":
            cell_index = _parse_cell_ref(query.get("cell"))
            try:
                digit = int(query.get("digit"))
            except Exception:
                digit = -1

            if cell_index is None:
                return json.dumps({
                    "ok": False,
                    "status": "error",
                    "op": op,
                    "error": {"code": "missing_cell", "msg": "query.cell is required"}
                }, ensure_ascii=False)

            if digit not in range(1, 10):
                return json.dumps({
                    "ok": False,
                    "status": "error",
                    "op": op,
                    "error": {"code": "missing_digit", "msg": "query.digit must be 1..9"}
                }, ensure_ascii=False)

            allowed_digits = _candidate_digits_before_for_cell_from_masks(options_all_masks, cell_index)
            blockers = _digit_blockers_for_cell(grid81, cell_index, digit)

            return json.dumps({
                "ok": True,
                "status": "ok",
                "op": op,
                "result": {
                    "cell": _cell_ref_string(cell_index),
                    "cell_index": cell_index,
                    "digit": digit,
                    "is_candidate": digit in allowed_digits,
                    "candidates_now": allowed_digits,
                    "blocker_analysis": blockers
                }
            }, ensure_ascii=False)

        if op == "validate_candidate_claim":
            cell_index = _parse_cell_ref(query.get("cell"))
            if cell_index is None:
                return json.dumps({
                    "ok": False,
                    "status": "error",
                    "op": op,
                    "error": {"code": "missing_cell", "msg": "query.cell is required"}
                }, ensure_ascii=False)

            claimed_digits = _parse_digit_list(query.get("claimed_digits"))
            actual_digits = _candidate_digits_before_for_cell_from_masks(options_all_masks, cell_index)

            extra_digits = [d for d in claimed_digits if d not in actual_digits]
            missing_digits = [d for d in actual_digits if d not in claimed_digits]

            if claimed_digits == actual_digits:
                verdict = "VALID"
            elif claimed_digits and (extra_digits or missing_digits):
                verdict = "INVALID"
            elif not claimed_digits:
                verdict = "UNKNOWN"
            else:
                verdict = "PARTIALLY_VALID"

            return json.dumps({
                "ok": True,
                "status": "ok",
                "op": op,
                "result": {
                    "cell": _cell_ref_string(cell_index),
                    "cell_index": cell_index,
                    "claimed_digits": claimed_digits,
                    "actual_digits": actual_digits,
                    "extra_digits": extra_digits,
                    "missing_digits": missing_digits,
                    "verdict": verdict
                }
            }, ensure_ascii=False)

        if op == "get_next_step_summary":
            step_obj = _run_next_step_obj(grid81)
            if not bool(step_obj.get("ok", False)):
                return json.dumps({
                    "ok": False,
                    "status": "error",
                    "op": op,
                    "error": step_obj.get("error", {
                        "code": "next_step_failed",
                        "msg": "next_step failed"
                    })
                }, ensure_ascii=False)

            return json.dumps({
                "ok": True,
                "status": "ok",
                "op": op,
                "result": {
                    "next_step": _next_step_summary(step_obj)
                }
            }, ensure_ascii=False)

        if op == "compare_requested_technique_to_next_step":
            requested_technique = query.get("requested_technique")
            requested_cell = query.get("cell")

            step_obj = _run_next_step_obj(grid81)
            if not bool(step_obj.get("ok", False)):
                return json.dumps({
                    "ok": False,
                    "status": "error",
                    "op": op,
                    "error": step_obj.get("error", {
                        "code": "next_step_failed",
                        "msg": "next_step failed"
                    })
                }, ensure_ascii=False)

            next_step = _next_step_summary(step_obj)
            relation = "UNKNOWN"

            if _technique_matches_hint(step_obj, requested_technique):
                relation = "SAME_MOVE"
            elif requested_technique:
                relation = "ALTERNATIVE_MOVE"

            cell_alignment = "UNSPECIFIED"
            if requested_cell:
                req_ci = _parse_cell_ref(requested_cell)
                target_ci = _parse_cell_ref(next_step.get("target_cell"))
                focus_ci = _parse_cell_ref(next_step.get("focus_cell"))
                if req_ci is not None and (req_ci == target_ci or req_ci == focus_ci):
                    cell_alignment = "MATCH"
                elif req_ci is not None:
                    cell_alignment = "DIFFERENT"

            return json.dumps({
                "ok": True,
                "status": "ok",
                "op": op,
                "result": {
                    "requested_technique": requested_technique,
                    "requested_cell": requested_cell,
                    "next_step": next_step,
                    "relation_to_next_step": relation,
                    "cell_alignment": cell_alignment
                }
            }, ensure_ascii=False)

        if op == "check_technique_in_scope":
            requested_technique = query.get("requested_technique")
            scope = query.get("scope") or {}

            step_obj = _run_next_step_obj(grid81)
            if not bool(step_obj.get("ok", False)):
                return json.dumps({
                    "ok": False,
                    "status": "error",
                    "op": op,
                    "error": step_obj.get("error", {
                        "code": "next_step_failed",
                        "msg": "next_step failed"
                    })
                }, ensure_ascii=False)

            scoped_summary = _summarize_step_for_scope(step_obj, scope)
            technique_match = _technique_matches_hint(step_obj, requested_technique)
            support_snapshot = _scoped_support_snapshot(grid81, options_all_masks, scope)

            verdict = "NOT_FOUND"
            if scoped_summary.get("in_scope") and technique_match:
                verdict = "FOUND"
            elif technique_match:
                verdict = "FOUND_OUT_OF_SCOPE"
            elif scoped_summary.get("in_scope"):
                verdict = "DIFFERENT_TECHNIQUE_IN_SCOPE"

            return json.dumps({
                "ok": True,
                "status": "ok",
                "op": op,
                "result": {
                    "requested_technique": requested_technique,
                    "scope": scope,
                    "next_step": scoped_summary,
                    "verdict": verdict,
                    "search_mode": "SCOPED_PROXY_NOT_EXHAUSTIVE",
                    "support_snapshot": support_snapshot
                }
            }, ensure_ascii=False)

        if op == "search_local_moves":
            scope = query.get("scope") or {}
            try:
                max_results = int(query.get("max_results", 1))
            except Exception:
                max_results = 1
            max_results = max(1, min(max_results, 3))

            step_obj = _run_next_step_obj(grid81)
            if not bool(step_obj.get("ok", False)):
                return json.dumps({
                    "ok": False,
                    "status": "error",
                    "op": op,
                    "error": step_obj.get("error", {
                        "code": "next_step_failed",
                        "msg": "next_step failed"
                    })
                }, ensure_ascii=False)

            scoped_summary = _summarize_step_for_scope(step_obj, scope)
            support_snapshot = _scoped_support_snapshot(grid81, options_all_masks, scope)

            moves: List[Dict[str, Any]] = []
            if scoped_summary.get("in_scope"):
                moves.append(scoped_summary)

            if len(moves) > max_results:
                moves = moves[:max_results]

            return json.dumps({
                "ok": True,
                "status": "ok",
                "op": op,
                "result": {
                    "scope": scope,
                    "moves": moves,
                    "count": len(moves),
                    "max_results": max_results,
                    "search_mode": "SCOPED_PROXY_NOT_EXHAUSTIVE",
                    "support_snapshot": support_snapshot
                }
            }, ensure_ascii=False)

        if op == "compare_current_route_to_scope":
            scope = query.get("scope") or {}
            current_route = query.get("current_route") or {}

            step_obj = _run_next_step_obj(grid81)
            if not bool(step_obj.get("ok", False)):
                return json.dumps({
                    "ok": False,
                    "status": "error",
                    "op": op,
                    "error": step_obj.get("error", {
                        "code": "next_step_failed",
                        "msg": "next_step failed"
                    })
                }, ensure_ascii=False)

            scoped_summary = _summarize_step_for_scope(step_obj, scope)
            comparison = _compare_current_route_to_next_step(current_route, scoped_summary, scope)

            return json.dumps({
                "ok": True,
                "status": "ok",
                "op": op,
                "result": {
                    "scope": scope,
                    "comparison": comparison
                }
            }, ensure_ascii=False)

        return json.dumps({
            "ok": False,
            "status": "unsupported_op",
            "op": op,
            "error": {
                "code": "unsupported_op",
                "msg": f"detour_query op not implemented yet: {op}"
            }
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "ok": False,
            "status": "error",
            "error": {
                "code": "detour_query_exception",
                "msg": str(e)[:240]
            }
        }, ensure_ascii=False)