from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# =============================================================================
# Helpers: loading + sorting
# =============================================================================

def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
                if isinstance(o, dict):
                    o["_src_file"] = str(path)
                    o["_src_line"] = ln
                    yield o
            except Exception:
                # Robust to corrupted line(s)
                continue


def _collect_inputs(inputs: List[str]) -> List[Path]:
    out: List[Path] = []
    for s in inputs:
        p = Path(s)
        if p.is_dir():
            out.extend(sorted(p.glob("*.jsonl")))
        elif p.is_file():
            out.append(p)

    uniq: List[Path] = []
    seen = set()
    for p in out:
        rp = str(p.resolve())
        if rp not in seen:
            uniq.append(p)
            seen.add(rp)
    return uniq


def _sort_key(ev: Dict[str, Any]) -> Tuple[int, int, int, int]:
    """
    Prefer stable sort:
      1) ts_epoch_ms
      2) seq
      3) turn_id
      4) policy_req_seq
    """
    def _int(x: Any) -> int:
        try:
            return int(x)
        except Exception:
            return 0

    ts_i = _int(ev.get("ts_epoch_ms"))
    seq_i = _int(ev.get("seq"))
    turn_i = _int(ev.get("turn_id"))
    req_i = _int(ev.get("policy_req_seq"))
    return (ts_i, seq_i, turn_i, req_i)


def _to_iso(ts_epoch_ms: Optional[int]) -> Optional[str]:
    if ts_epoch_ms is None:
        return None
    try:
        dt = datetime.fromtimestamp(int(ts_epoch_ms) / 1000.0, tz=timezone.utc)
        return dt.isoformat()
    except Exception:
        return None


def _cap_text(text: str, max_chars: int) -> str:
    t = (text or "").replace("\r", "\n").strip()
    if len(t) <= max_chars:
        return t
    return t[:max_chars].rstrip() + "…"


# =============================================================================
# Domain objects (existing)
# =============================================================================

@dataclass
class GridContext:
    raw_text: str = ""
    solvability: Optional[str] = None  # unique|multiple|none
    mismatch_indices: List[int] = dataclasses.field(default_factory=list)
    unresolved_indices: List[int] = dataclasses.field(default_factory=list)
    low_confidence_indices: List[int] = dataclasses.field(default_factory=list)
    auto_changed_indices: List[int] = dataclasses.field(default_factory=list)

    confirmed_indices: List[int] = dataclasses.field(default_factory=list)
    confirmed_count: Optional[int] = None

    conflicts_details_lines: Optional[List[str]] = None
    status_line: Optional[str] = None
    parse_warnings: List[str] = dataclasses.field(default_factory=list)


@dataclass
class ToolPlanFinal:
    reply_len: Optional[int] = None
    ops: List[str] = dataclasses.field(default_factory=list)
    control: Optional[str] = None
    out_names: List[str] = dataclasses.field(default_factory=list)


@dataclass
class ControlArgs:
    row: Optional[int] = None
    col: Optional[int] = None
    cell_index: Optional[int] = None
    digit: Optional[int] = None
    source_event_type: Optional[str] = None

    def to_idx(self) -> Optional[int]:
        if self.cell_index is not None:
            return self.cell_index
        if self.row is not None and self.col is not None:
            if 1 <= self.row <= 9 and 1 <= self.col <= 9:
                return (self.row - 1) * 9 + (self.col - 1)
        return None


@dataclass
class CanonicalMessage:
    role: str
    content_preview: str
    content_len: int


@dataclass
class ObservedToolCall:
    name: str
    idx: Optional[int] = None
    row: Optional[int] = None
    col: Optional[int] = None
    digit: Optional[int] = None
    ev_type: Optional[str] = None


# =============================================================================
# New v2 domain objects: assistant_turn -> ticks[]
# =============================================================================

@dataclass
class TickAudit:
    tick_id: int
    turn_id: int
    policy_req_seq: Optional[int] = None
    mode: Optional[str] = None

    # Window timing
    window_start_ts_iso: Optional[str] = None
    window_end_ts_iso: Optional[str] = None
    window_start_seq: Optional[int] = None
    window_end_seq: Optional[int] = None

    # Prompt snapshot
    prompt_req_id: Optional[str] = None
    prompt_snapshot_note: Optional[str] = None

    grid: Optional[GridContext] = None
    toolplan: Optional[ToolPlanFinal] = None
    control_args: Optional[ControlArgs] = None

    # reply + canonical history
    reply_text: Optional[str] = None
    reply_text_len: Optional[int] = None
    canonical_history: Optional[List[CanonicalMessage]] = None
    canonical_history_count: Optional[int] = None
    canonical_history_sha256: Optional[str] = None

    # Observed tools inside policy window
    observed_tools: Optional[List[ObservedToolCall]] = None
    applied_edit_indices: List[int] = dataclasses.field(default_factory=list)
    confirmed_this_tick_indices: List[int] = dataclasses.field(default_factory=list)

    violations: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    notes: List[str] = dataclasses.field(default_factory=list)


@dataclass
class AssistantTurnAudit:
    turn_id: int
    # Usually stable across ticks, but we take from the first tick begin event
    mode: Optional[str] = None
    user_text: Optional[str] = None
    state_header_preview: Optional[str] = None

    ticks: List[TickAudit] = dataclasses.field(default_factory=list)

    # Cross-tick notes/violations (Design A “don’t repeat” etc.)
    violations: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    notes: List[str] = dataclasses.field(default_factory=list)


@dataclass
class AuditReportV2:
    audit_version: str
    generated_at_iso: str
    inputs: Dict[str, Any]
    session_id: Optional[str]
    telemetry_id: Optional[str]
    start_ts_iso: Optional[str]
    end_ts_iso: Optional[str]

    assistant_turns: List[AssistantTurnAudit]
    summary: Dict[str, Any]


# =============================================================================
# Parsing: GRID_CONTEXT from prompt dumps (unchanged logic)
# =============================================================================

_RE_STATUS_LIST_LINE = re.compile(r"(?m)^\s*-\s*([A-Za-z0-9_ ()/.-]+?)\s*:\s*\[(.*?)\]\s*$")
_RE_STATUS_INT_LINE = re.compile(r"(?m)^\s*-\s*([A-Za-z0-9_ ()/.-]+?)\s*:\s*(\d+)\s*$")


def _parse_int_list_brackets(inner: str) -> List[int]:
    inner = inner.strip()
    if not inner:
        return []
    out: List[int] = []
    for part in inner.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except Exception:
            pass
    return out


def _extract_grid_context_block(text: str) -> Optional[str]:
    m = re.search(r"BEGIN_GRID_CONTEXT\s*\n(.*?)\nEND_GRID_CONTEXT", text, flags=re.DOTALL)
    if not m:
        return None
    return m.group(1)


def _parse_grid_context(grid_ctx: str) -> GridContext:
    gc = GridContext(raw_text=grid_ctx)

    m = re.search(
        r"(?m)^\s*-\s*solvability_of_current_display:\s*(unique|multiple|none)\b",
        grid_ctx
    )
    if m:
        gc.solvability = m.group(1).lower()
        gc.status_line = m.group(0).strip()
    else:
        gc.parse_warnings.append("No STATUS solvability_of_current_display line found.")

    lists: Dict[str, List[int]] = {}
    for mm in _RE_STATUS_LIST_LINE.finditer(grid_ctx):
        key = mm.group(1).strip()
        inner = mm.group(2)
        lists[key] = _parse_int_list_brackets(inner)

    ints: Dict[str, int] = {}
    for mm in _RE_STATUS_INT_LINE.finditer(grid_ctx):
        key = mm.group(1).strip()
        val = mm.group(2)
        try:
            ints[key] = int(val)
        except Exception:
            pass

    def pick_list(*keys: str) -> List[int]:
        for k in keys:
            if k in lists:
                return lists[k]
        return []

    def pick_int(*keys: str) -> Optional[int]:
        for k in keys:
            if k in ints:
                return ints[k]
        return None

    gc.unresolved_indices = pick_list("unresolved_indices")
    gc.low_confidence_indices = pick_list("low_confidence_indices")
    gc.auto_changed_indices = pick_list("auto_changed_indices", "autoChanged_indices")
    gc.mismatch_indices = pick_list("mismatch_indices_vs_deduced (only if unique)", "mismatch_indices_vs_deduced")

    gc.confirmed_indices = pick_list("confirmed_indices")
    gc.confirmed_count = pick_int("confirmed_count")
    if gc.confirmed_count is None:
        gc.confirmed_count = len(gc.confirmed_indices)

    header = re.search(r"(?m)^CONFLICTS_DETAILS\b.*?:\s*$", grid_ctx)
    if header:
        tail = grid_ctx[header.end():]
        nxt = re.search(r"(?m)^[A-Z0-9_ ]{3,}:\s*$", tail)
        block = tail[:nxt.start()] if nxt else tail

        lines: List[str] = []
        for line in block.splitlines():
            line = line.strip()
            if line.startswith("- "):
                lines.append(line[2:].strip())
        gc.conflicts_details_lines = lines
    else:
        gc.parse_warnings.append("No CONFLICTS_DETAILS header found (accepted forms include parentheses + colon).")

    return gc


def _extract_grid_context_from_messages_pretty_json(messages_json_text: str) -> Optional[GridContext]:
    try:
        msgs = json.loads(messages_json_text)
    except Exception:
        return None
    if not isinstance(msgs, list):
        return None

    contents: List[str] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        c = m.get("content")
        if isinstance(c, str) and "BEGIN_GRID_CONTEXT" in c and "END_GRID_CONTEXT" in c:
            contents.append(c)

    if not contents:
        return None

    text = contents[-1]
    block = _extract_grid_context_block(text)
    if not block:
        return None

    return _parse_grid_context(block)


# =============================================================================
# Canonical history extraction (from messages_pretty_json)
# =============================================================================

def _extract_canonical_history(messages_json_text: str, per_msg_cap: int = 700) -> Tuple[Optional[List[CanonicalMessage]], Optional[int], Optional[str]]:
    if not messages_json_text:
        return None, None, None

    sha = hashlib.sha256(messages_json_text.encode("utf-8", errors="replace")).hexdigest()

    try:
        msgs = json.loads(messages_json_text)
    except Exception:
        return None, None, sha
    if not isinstance(msgs, list):
        return None, None, sha

    out: List[CanonicalMessage] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = (m.get("role") or "?")
        if not isinstance(role, str):
            role = str(role)
        content = m.get("content")
        if content is None:
            content = ""
        if not isinstance(content, str):
            try:
                content = json.dumps(content, ensure_ascii=False)
            except Exception:
                content = str(content)
        out.append(CanonicalMessage(role=role, content_preview=_cap_text(content, per_msg_cap), content_len=len(content)))

    return out, len(out), sha


# =============================================================================
# Parsing: LLM_PROMPT_DUMP chunk assembly
# =============================================================================

def _assemble_chunks(events: List[Dict[str, Any]], label: str) -> Dict[str, str]:
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for e in events:
        if e.get("type") != "LLM_PROMPT_DUMP":
            continue
        if e.get("label") != label:
            continue
        req_id = e.get("req_id")
        if not req_id:
            continue
        buckets.setdefault(req_id, []).append(e)

    out: Dict[str, str] = {}
    for req_id, chunks in buckets.items():
        chunks_sorted = sorted(chunks, key=lambda x: int(x.get("chunk_index") or 0))
        out[req_id] = "".join([(c.get("text_part") or "") for c in chunks_sorted])
    return out


# =============================================================================
# Windowing: policy call spans
# =============================================================================

def _policy_windows(events: List[Dict[str, Any]]) -> List[Tuple[int, int, Dict[str, Any], Dict[str, Any]]]:
    begins: Dict[int, int] = {}
    windows: List[Tuple[int, int, Dict[str, Any], Dict[str, Any]]] = []

    for i, e in enumerate(events):
        t = e.get("type")
        if t == "LLM_CALLPOLICY_BEGIN":
            seq = e.get("policy_req_seq")
            if isinstance(seq, int):
                begins[seq] = i
        elif t in ("LLM_CALLPOLICY_END_OK", "LLM_CALLPOLICY_END_ERR"):
            seq = e.get("policy_req_seq")
            if isinstance(seq, int) and seq in begins:
                b_i = begins.pop(seq)
                windows.append((b_i, i, events[b_i], e))

    return windows


# =============================================================================
# Best-effort extraction of tool args / reply text from window events
# =============================================================================

def _parse_args_json(s: str) -> Optional[Dict[str, Any]]:
    if not isinstance(s, str):
        return None
    ss = s.strip()
    if not (ss.startswith("{") and ss.endswith("}")):
        return None
    try:
        o = json.loads(ss)
        return o if isinstance(o, dict) else None
    except Exception:
        return None


def _iter_tool_calls_from_event(e: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any], str]]:
    ev_type = e.get("type") or "?"
    for key in ("tools", "tool_calls", "parsed_tools", "toolplan", "toolPlan"):
        tl = e.get(key)
        if isinstance(tl, list):
            for tc in tl:
                if not isinstance(tc, dict):
                    continue
                name = tc.get("name") or tc.get("tool") or tc.get("wire_name")
                if not isinstance(name, str) or not name:
                    continue
                args = tc.get("args") or tc.get("arguments") or {}
                if isinstance(args, dict):
                    yield name, args, ev_type
                elif isinstance(args, str):
                    args_obj = _parse_args_json(args)
                    if isinstance(args_obj, dict):
                        yield name, args_obj, ev_type

    names = e.get("raw_tool_names")
    args_list = e.get("raw_tool_args_json")
    if isinstance(names, list) and isinstance(args_list, list) and len(names) == len(args_list):
        for n, a in zip(names, args_list):
            if not isinstance(n, str):
                continue
            if isinstance(a, dict):
                yield n, a, ev_type
            elif isinstance(a, str):
                args_obj = _parse_args_json(a)
                if isinstance(args_obj, dict):
                    yield n, args_obj, ev_type

    for key in ("raw_tool_args_json", "tool_args_json", "args_json", "arguments_json"):
        s = e.get(key)
        if isinstance(s, str):
            args_obj = _parse_args_json(s)
            if isinstance(args_obj, dict):
                yield "", args_obj, ev_type

    ca = e.get("control_args")
    if isinstance(ca, dict):
        yield "", ca, ev_type


def _extract_idx_row_col_digit(args: Dict[str, Any]) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    idx = None
    for k in ("cellIndex", "cell_index", "cellIdx", "idx", "index"):
        if isinstance(args.get(k), int):
            idx = int(args[k])
            break

    row = args.get("row")
    col = args.get("col")
    if not isinstance(row, int):
        row = None
    if not isinstance(col, int):
        col = None

    digit = None
    for k in ("digit", "value", "proposedDigit", "givenDigit"):
        if isinstance(args.get(k), int):
            digit = int(args[k])
            break

    if idx is None and row is not None and col is not None:
        if 1 <= row <= 9 and 1 <= col <= 9:
            idx = (row - 1) * 9 + (col - 1)

    return idx, row, col, digit


def _extract_observed_tool_calls(window_events: List[Dict[str, Any]], cap: int = 60) -> List[ObservedToolCall]:
    out: List[ObservedToolCall] = []
    for e in window_events:
        ev_type = e.get("type") or "?"
        for name, args, _ in _iter_tool_calls_from_event(e):
            if not isinstance(name, str):
                continue
            n = name.strip()
            if not n:
                continue
            if not isinstance(args, dict):
                continue
            idx, row, col, digit = _extract_idx_row_col_digit(args)
            out.append(ObservedToolCall(name=n, idx=idx, row=row, col=col, digit=digit, ev_type=ev_type))
            if len(out) >= cap:
                return out
    return out


def _try_extract_control_args(window_events: List[Dict[str, Any]], control_name: Optional[str]) -> Optional[ControlArgs]:
    if not control_name:
        return None

    def parse_from_args(args: Dict[str, Any], ev_type: str) -> ControlArgs:
        ca = ControlArgs(source_event_type=ev_type)
        idx, row, col, digit = _extract_idx_row_col_digit(args)
        ca.cell_index = idx
        ca.row = row
        ca.col = col
        ca.digit = digit
        return ca

    for e in reversed(window_events):
        ev_type = e.get("type") or "?"
        for name, args, _ in _iter_tool_calls_from_event(e):
            if name == control_name:
                return parse_from_args(args, ev_type)

    return None


def _try_extract_reply_text(window_events: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[int]]:
    for e in reversed(window_events):
        for name, args, _ in _iter_tool_calls_from_event(e):
            if (name or "").lower() != "reply":
                continue
            txt = args.get("text")
            if isinstance(txt, str) and txt.strip():
                return txt, len(txt)
    return None, None


# =============================================================================
# Cell-question detection (used in your existing policy checks)
# =============================================================================

_RC_PATTERNS = [
    re.compile(r"\br\s*([1-9])\s*c\s*([1-9])\b", re.IGNORECASE),
    re.compile(r"\brow\s*([1-9])\s*(?:,|\s)*col(?:umn)?\s*([1-9])\b", re.IGNORECASE),
    re.compile(r"\brow\s*([1-9])\s*(?:,|\s)*column\s*([1-9])\b", re.IGNORECASE),
]


def _idx_from_rc(row: int, col: int) -> int:
    return (row - 1) * 9 + (col - 1)


def extract_cell_ref_from_text(text: Optional[str]) -> Optional[Tuple[int, int, int]]:
    if not text:
        return None
    hits: List[Tuple[int, int, int, int]] = []
    for pat in _RC_PATTERNS:
        for m in pat.finditer(text):
            r = int(m.group(1))
            c = int(m.group(2))
            hits.append((m.start(), _idx_from_rc(r, c), r, c))
    if not hits:
        return None
    _, idx, r, c = max(hits, key=lambda x: x[0])
    return (idx, r, c)


def text_looks_like_cell_question(text: Optional[str]) -> bool:
    if not text:
        return False
    if extract_cell_ref_from_text(text) is None:
        return False
    t = text.lower()
    if "?" in t:
        return True
    return any(w in t for w in ["confirm", "should", "blank", "value", "what is in", "what's in", "is it"])


# =============================================================================
# Checks (ported to TickAudit; same logic, but renamed args)
# =============================================================================

def _check_allowed_next_step(tick: TickAudit) -> None:
    if not tick.grid or not tick.toolplan:
        tick.notes.append("Skipping next-step checks: missing GRID_CONTEXT or toolplan.")
        return

    solv = (tick.grid.solvability or "").lower()
    mismatch = tick.grid.mismatch_indices or []
    unresolved = tick.grid.unresolved_indices or []
    confirmed = set(tick.grid.confirmed_indices or [])

    control = tick.toolplan.control
    if control is None:
        return

    is_cell_check = control in ("ask_confirm_cell_rc", "ask_confirm_cell")
    asked_idx = tick.control_args.to_idx() if tick.control_args else None

    reply_ref = extract_cell_ref_from_text(getattr(tick, "reply_text", None))
    reply_asks_cell = text_looks_like_cell_question(getattr(tick, "reply_text", None))
    reply_asked_idx = reply_ref[0] if (reply_ref and reply_asks_cell) else None

    def fail(code: str, msg: str, details: Dict[str, Any]) -> None:
        tick.violations.append({"kind": "NEXT_STEP_POLICY", "code": code, "message": msg, "details": details})

    if is_cell_check and asked_idx is not None and asked_idx in confirmed:
        fail(
            "ASKED_CONFIRMED_CELL",
            "Control tool asked a cell that is already in confirmed_indices (should not re-ask).",
            {"asked_idx": asked_idx, "confirmed_count": len(confirmed), "control": control},
        )

    resolved_this_tick = set(tick.applied_edit_indices or [])
    mismatch_set = set(mismatch)

    if mismatch:
        mismatch_remaining_after_resolve = mismatch_set - resolved_this_tick

        if control in ("recommend_validate", "recommend_retake") and not mismatch_remaining_after_resolve:
            return

        if control in ("recommend_validate", "recommend_retake"):
            fail(
                "CASE1_WRONG_CONTROL",
                "Mismatch exists in the pre-tick snapshot and is not fully resolved within this same policy window; control should be a cell confirmation.",
                {
                    "solvability": solv,
                    "mismatch_count": len(mismatch),
                    "mismatch_indices": mismatch[:30],
                    "resolved_this_tick_indices": sorted(resolved_this_tick)[:30],
                    "control": control,
                },
            )
            return

        if is_cell_check and asked_idx is not None and asked_idx not in mismatch_set:
            fail(
                "CASE1_WRONG_CELL",
                "Mismatch exists; asked cell is not in mismatch_indices_vs_deduced.",
                {"asked_idx": asked_idx, "mismatch_indices_sample": mismatch[:30], "control": control},
            )
        return

    if solv == "none":
        if control == "recommend_validate":
            fail(
                "CASE2_VALIDATE_IN_NONE",
                "solvability==none: recommend_validate is usually a dead step; should confirm an unresolved cell or retake.",
                {"unresolved_count": len(unresolved), "control": control},
            )
        if is_cell_check and asked_idx is not None and unresolved and asked_idx not in set(unresolved):
            fail(
                "CASE2_WRONG_CELL",
                "solvability==none: asked cell is not in unresolved_indices.",
                {"asked_idx": asked_idx, "unresolved_indices_sample": unresolved[:30], "control": control},
            )
        return

    if solv == "unique":
        if control in ("recommend_validate", "recommend_retake") and reply_asked_idx is not None:
            fail(
                "VALIDATE_REPLY_ASKS_CELL",
                "solvability==unique: control recommends validate/retake, but reply text asks about a specific cell.",
                {
                    "control": control,
                    "reply_asked_idx": reply_asked_idx,
                    "reply_excerpt": (getattr(tick, "reply_text", "") or "")[:220],
                    "confirmed_this_tick_indices": tick.confirmed_this_tick_indices,
                },
            )
            if tick.confirmed_this_tick_indices and reply_asked_idx not in set(tick.confirmed_this_tick_indices):
                fail(
                    "REPLY_CELL_DRIFT_FROM_CONFIRMED",
                    "Reply asks about a different cell than the one confirmed this tick.",
                    {
                        "confirmed_this_tick_indices": tick.confirmed_this_tick_indices,
                        "reply_asked_idx": reply_asked_idx,
                        "reply_excerpt": (getattr(tick, "reply_text", "") or "")[:220],
                    },
                )

        if is_cell_check:
            fail(
                "CASE3_CELL_CHECK_FORBIDDEN",
                "solvability==unique and mismatch empty: should stop cell-by-cell checks and use recommend_validate.",
                {"control": control, "asked_idx": asked_idx},
            )
        return

    if solv == "multiple":
        if is_cell_check:
            fail(
                "CASE4_CELL_CHECK_FORBIDDEN",
                "solvability==multiple and mismatch empty: should not do cell-by-cell verification; ask for match via recommend_validate.",
                {"control": control, "asked_idx": asked_idx},
            )
        return

    if solv not in ("unique", "multiple", "none", ""):
        tick.notes.append(f"Unknown solvability value: {solv!r}")


# =============================================================================
# Toolplan indexing (robust to variants)
# =============================================================================

def _index_toolplans(events: List[Dict[str, Any]]) -> Dict[Tuple[int, int], ToolPlanFinal]:
    """
    Prefer (turn_id, policy_req_seq) if present; fall back to (turn_id, 0) if not.
    """
    out: Dict[Tuple[int, int], ToolPlanFinal] = {}

    for e in events:
        t = e.get("type")
        if t not in ("LLM_TOOLPLAN_FINALIZED", "LLM_TOOLPLAN_OK", "LLM_TOOLPLAN_FINAL", "LLM_TOOLPLAN"):
            continue

        tid = e.get("turn_id")
        if not isinstance(tid, int):
            continue

        prs = e.get("policy_req_seq")
        prs_i = int(prs) if isinstance(prs, int) else 0

        # Normalize fields across types
        reply_len = e.get("reply_len")
        ops = e.get("ops") or e.get("op_names") or []
        control = e.get("control") or e.get("control_tool") or e.get("controlName")
        out_names = e.get("out_names") or e.get("tool_names") or []

        if isinstance(ops, str):
            ops = [p.strip() for p in ops.split(",") if p.strip()]
        if isinstance(out_names, str):
            out_names = [p.strip() for p in out_names.split(",") if p.strip()]

        if not isinstance(ops, list):
            ops = []
        if not isinstance(out_names, list):
            out_names = []

        out[(tid, prs_i)] = ToolPlanFinal(
            reply_len=int(reply_len) if isinstance(reply_len, int) else None,
            ops=[str(x) for x in ops if str(x)],
            control=str(control) if isinstance(control, str) and control else None,
            out_names=[str(x) for x in out_names if str(x)],
        )

    return out


# =============================================================================
# Cross-tick “Design A” guard: don’t restate applied edit on tick>0
# =============================================================================

_RE_RESTATE_HINT = re.compile(
    r"\b(i\s+(updated|set|changed|applied)|just\s+(updated|set|changed)|"
    r"i\s+have\s+(updated|set|changed)|done|great,\s*i\s+set)\b",
    re.IGNORECASE
)

def _check_dont_repeat_guard(turn: AssistantTurnAudit) -> None:
    """
    Heuristic:
    - If tick 0 applied an edit (apply_user_edit*), then tick 1+ reply should not
      “re-acknowledge” the same applied edit.
    """
    if len(turn.ticks) <= 1:
        return

    applied_by_tick: Dict[int, List[int]] = {}
    for tk in turn.ticks:
        if tk.applied_edit_indices:
            applied_by_tick[tk.tick_id] = tk.applied_edit_indices[:]

    if not applied_by_tick:
        return

    # Most common pattern: tick0 applies, tick1 continues.
    # We’ll check tick>=1 for “restatement language”.
    first_apply_tick = min(applied_by_tick.keys())
    applied_cells = set(applied_by_tick[first_apply_tick])

    for tk in turn.ticks:
        if tk.tick_id <= first_apply_tick:
            continue
        txt = tk.reply_text or ""
        if not txt.strip():
            continue
        if _RE_RESTATE_HINT.search(txt):
            # If it also mentions the same cell, confidence increases
            ref = extract_cell_ref_from_text(txt)
            ref_idx = ref[0] if ref else None
            suspicious = (ref_idx is None) or (ref_idx in applied_cells)
            if suspicious:
                turn.violations.append({
                    "kind": "DESIGN_A_GUARD",
                    "code": "DONT_REPEAT_APPLIED_EDIT",
                    "message": "Later tick reply looks like it restates an edit that was already applied earlier in the same assistant_turn.",
                    "details": {
                        "first_apply_tick": first_apply_tick,
                        "applied_cells": sorted(applied_cells)[:30],
                        "tick_id": tk.tick_id,
                        "reply_excerpt": _cap_text(txt, 260),
                        "reply_ref_idx": ref_idx,
                    }
                })


# =============================================================================
# Rendering: timeline view + markdown report
# =============================================================================

def _render_timeline_md(report: AuditReportV2) -> str:
    """
    A human-first “timeline view”:
    - assistant_turn sections
    - each tick summarized in one line + optional compact tool list + violations
    """
    L: List[str] = []
    L.append("# Telemetry Timeline View")
    L.append("")
    L.append(f"- Generated: `{report.generated_at_iso}`")
    L.append(f"- Audit version: `{report.audit_version}`")
    L.append(f"- Session: `{report.session_id}`  Telemetry: `{report.telemetry_id}`")
    L.append(f"- Range: `{report.start_ts_iso}` → `{report.end_ts_iso}`")
    L.append("")

    for at in report.assistant_turns:
        L.append(f"## Assistant turn {at.turn_id} ({at.mode})")
        if at.user_text:
            L.append(f"- User: {at.user_text}")
        if at.state_header_preview:
            L.append(f"- state: `{_cap_text(at.state_header_preview, 240)}`")
        if at.violations:
            L.append("")
            L.append("**Turn-level violations:**")
            for v in at.violations:
                L.append(f"- [{v['kind']}] {v['code']}: {v['message']}")

        L.append("")
        L.append("| tick | policy_req_seq | window_ts | control | reply_preview | tools/effects | violations |")
        L.append("|---:|---:|---|---|---|---|---:|")

        for tk in at.ticks:
            control = tk.toolplan.control if tk.toolplan else None
            reply_prev = _cap_text(tk.reply_text or "", 120) if tk.reply_text else ""
            tools_compact: List[str] = []
            for oc in (tk.observed_tools or [])[:10]:
                if oc.idx is not None:
                    tools_compact.append(f"{oc.name}@{oc.idx}")
                else:
                    tools_compact.append(oc.name)
            eff = []
            if tk.applied_edit_indices:
                eff.append(f"apply={tk.applied_edit_indices}")
            if tk.confirmed_this_tick_indices:
                eff.append(f"confirm={tk.confirmed_this_tick_indices}")
            tools_s = ", ".join(tools_compact)
            eff_s = " ".join(eff)
            te = (tools_s + ("; " if tools_s and eff_s else "") + eff_s).strip()

            wts = f"`{tk.window_start_ts_iso}` → `{tk.window_end_ts_iso}`"
            vcount = len(tk.violations)
            L.append(f"| {tk.tick_id} | {tk.policy_req_seq or ''} | {wts} | `{control or ''}` | `{reply_prev}` | `{_cap_text(te, 140)}` | {vcount} |")

        # Details per tick
        for tk in at.ticks:
            L.append("")
            L.append(f"### Turn {at.turn_id} tick {tk.tick_id}")
            L.append(f"- policy_req_seq: {tk.policy_req_seq}")
            L.append(f"- window_seq: {tk.window_start_seq}..{tk.window_end_seq}")
            L.append(f"- window_ts: `{tk.window_start_ts_iso}` → `{tk.window_end_ts_iso}`")
            if tk.prompt_req_id:
                L.append(f"- prompt_req_id: `{tk.prompt_req_id}`")
            if tk.grid:
                L.append(
                    f"- GRID: solvability=`{tk.grid.solvability}` "
                    f"mismatch={len(tk.grid.mismatch_indices)} unresolved={len(tk.grid.unresolved_indices)} "
                    f"confirmed={tk.grid.confirmed_count}"
                )
            if tk.toolplan:
                L.append(f"- toolplan: control=`{tk.toolplan.control}` ops={tk.toolplan.ops} out={tk.toolplan.out_names}")
            if tk.reply_text:
                L.append(f"- reply: `{_cap_text(tk.reply_text, 360)}`")
            if tk.applied_edit_indices or tk.confirmed_this_tick_indices:
                L.append(f"- effects: applied={tk.applied_edit_indices} confirmed={tk.confirmed_this_tick_indices}")

            if tk.violations:
                L.append("")
                L.append("**Violations:**")
                for v in tk.violations:
                    L.append(f"- [{v['kind']}] {v['code']}: {v['message']}")
            if tk.notes:
                L.append("")
                L.append("Notes:")
                for n in tk.notes:
                    L.append(f"- {n}")

        L.append("")

    return "\n".join(L)


def _render_audit_md(report: AuditReportV2) -> str:
    L: List[str] = []
    L.append("# Telemetry Audit Report (v2)")
    L.append("")
    L.append(f"- Generated: `{report.generated_at_iso}`")
    L.append(f"- Audit version: `{report.audit_version}`")
    L.append(f"- Session: `{report.session_id}`  Telemetry: `{report.telemetry_id}`")
    L.append(f"- Range: `{report.start_ts_iso}` → `{report.end_ts_iso}`")
    L.append("")
    L.append("## Reading guide (timing semantics)")
    L.append("")
    L.append("- Each **tick** corresponds to one `LLM_CALLPOLICY_BEGIN..END` window.")
    L.append("- An **assistant_turn** groups all ticks that share the same `turn_id` (your new structure: `assistant_turn -> ticks[]`).")
    L.append("- `GRID_CONTEXT` is parsed from `messages_pretty_json` **inside each tick** and represents the **pre-decision snapshot** for that policy call.")
    L.append("- Tool calls in `observed_tools` happen **during** that tick.")
    L.append("")

    L.append("## Summary")
    L.append("")
    for k, v in report.summary.items():
        L.append(f"- **{k}**: {v}")
    L.append("")

    L.append("## Assistant turns")
    L.append("")
    for at in report.assistant_turns:
        L.append(f"### Assistant turn {at.turn_id} ({at.mode})")
        if at.user_text:
            L.append(f"- User: {at.user_text}")
        if at.violations:
            L.append("")
            L.append("**Turn-level violations:**")
            for v in at.violations:
                L.append(f"- [{v['kind']}] {v['code']}: {v['message']}")

        for tk in at.ticks:
            L.append("")
            L.append(f"#### Tick {tk.tick_id} (policy_req_seq={tk.policy_req_seq})")
            L.append(f"- window_ts: `{tk.window_start_ts_iso}` → `{tk.window_end_ts_iso}` (seq {tk.window_start_seq}..{tk.window_end_seq})")
            if tk.reply_text:
                L.append(f"- reply_preview: `{_cap_text(tk.reply_text, 220)}`")
            if tk.grid:
                L.append(
                    f"- solvability: `{tk.grid.solvability}` mismatch:{len(tk.grid.mismatch_indices)} "
                    f"unresolved:{len(tk.grid.unresolved_indices)} confirmed:{tk.grid.confirmed_count}"
                )
            if tk.toolplan:
                L.append(f"- toolplan: control=`{tk.toolplan.control}` ops={tk.toolplan.ops} out={tk.toolplan.out_names}")
            if tk.applied_edit_indices or tk.confirmed_this_tick_indices:
                L.append(f"- effects: applied={tk.applied_edit_indices} confirmed={tk.confirmed_this_tick_indices}")

            if tk.violations:
                L.append("")
                L.append("**Violations:**")
                for v in tk.violations:
                    L.append(f"- [{v['kind']}] {v['code']}: {v['message']}")

        L.append("")

    return "\n".join(L)


# =============================================================================
# Main audit pipeline (v2)
# =============================================================================

def run_audit_v2(input_paths: List[Path]) -> AuditReportV2:
    events: List[Dict[str, Any]] = []
    for p in input_paths:
        events.extend(list(_iter_jsonl(p)))
    events.sort(key=_sort_key)

    session_id = events[0].get("session_id") if events else None
    telemetry_id = events[0].get("telemetry_id") if events else None
    start_ts_iso = _to_iso(events[0].get("ts_epoch_ms")) if events else None
    end_ts_iso = _to_iso(events[-1].get("ts_epoch_ms")) if events else None

    messages_by_req = _assemble_chunks(events, label="messages_pretty_json")

    toolplans_by_turn_req = _index_toolplans(events)

    windows = _policy_windows(events)

    # 1) Build ticks
    ticks_by_turn: Dict[int, List[TickAudit]] = {}

    for (b_i, e_i, b_ev, e_ev) in windows:
        turn_id = b_ev.get("turn_id")
        if not isinstance(turn_id, int):
            continue

        policy_req_seq = b_ev.get("policy_req_seq") if isinstance(b_ev.get("policy_req_seq"), int) else None

        window_events = events[b_i:e_i + 1]

        tk = TickAudit(
            tick_id=-1,  # assigned after grouping
            turn_id=turn_id,
            policy_req_seq=policy_req_seq,
            mode=b_ev.get("mode"),
            window_start_ts_iso=_to_iso(b_ev.get("ts_epoch_ms")),
            window_end_ts_iso=_to_iso(e_ev.get("ts_epoch_ms")),
            window_start_seq=b_ev.get("seq") if isinstance(b_ev.get("seq"), int) else None,
            window_end_seq=e_ev.get("seq") if isinstance(e_ev.get("seq"), int) else None,
        )

        # Find latest req_id in this window that has messages_pretty_json
        req_ids: List[str] = []
        for we in window_events:
            rid = we.get("req_id")
            if rid and rid in messages_by_req:
                req_ids.append(rid)

        if req_ids:
            rid = req_ids[-1]
            tk.prompt_req_id = rid
            msgs_text = messages_by_req[rid]

            tk.prompt_snapshot_note = (
                "GRID_CONTEXT parsed from messages_pretty_json in this policy window. "
                "Interpret as PRE-DECISION snapshot for this tick."
            )

            hist, cnt, sha = _extract_canonical_history(msgs_text)
            tk.canonical_history = hist
            tk.canonical_history_count = cnt
            tk.canonical_history_sha256 = sha

            tk.grid = _extract_grid_context_from_messages_pretty_json(msgs_text)
        else:
            tk.notes.append("No messages_pretty_json found in this policy window; cannot parse GRID_CONTEXT or canonical history.")

        # Toolplan: prefer (turn_id, policy_req_seq), fallback (turn_id,0)
        if policy_req_seq is not None and (turn_id, policy_req_seq) in toolplans_by_turn_req:
            tk.toolplan = toolplans_by_turn_req[(turn_id, policy_req_seq)]
        else:
            tk.toolplan = toolplans_by_turn_req.get((turn_id, 0))

        rt, rtl = _try_extract_reply_text(window_events)
        tk.reply_text = rt
        tk.reply_text_len = rtl

        if tk.toolplan and tk.toolplan.control:
            tk.control_args = _try_extract_control_args(window_events, tk.toolplan.control)

        tk.observed_tools = _extract_observed_tool_calls(window_events)

        applied = set()
        confirmed_now = set()
        for oc in tk.observed_tools or []:
            n = (oc.name or "").lower()
            if oc.idx is None:
                continue
            if "apply_user_edit" in n:
                applied.add(oc.idx)
            if "confirm_cell_value" in n:
                confirmed_now.add(oc.idx)
        tk.applied_edit_indices = sorted(applied)
        tk.confirmed_this_tick_indices = sorted(confirmed_now)

        # Checks
        _check_allowed_next_step(tk)

        ticks_by_turn.setdefault(turn_id, []).append(tk)

    # 2) Build assistant_turns + assign tick_id
    assistant_turns: List[AssistantTurnAudit] = []
    for turn_id, tks in sorted(ticks_by_turn.items(), key=lambda x: x[0]):
        # Sort ticks inside a turn by policy_req_seq (best) then window_start_seq
        def _tick_sort_key(tk: TickAudit) -> Tuple[int, int]:
            prs = tk.policy_req_seq if isinstance(tk.policy_req_seq, int) else 0
            ss = tk.window_start_seq if isinstance(tk.window_start_seq, int) else 0
            return (prs, ss)

        tks_sorted = sorted(tks, key=_tick_sort_key)
        for i, tk in enumerate(tks_sorted):
            tk.tick_id = i

        # Pull turn metadata from earliest begin event for that turn_id
        # (We’ll scan events for first LLM_CALLPOLICY_BEGIN matching turn_id.)
        mode = None
        user_text = None
        state_header_preview = None
        for e in events:
            if e.get("type") == "LLM_CALLPOLICY_BEGIN" and e.get("turn_id") == turn_id:
                mode = e.get("mode")
                user_text = e.get("user_text")
                state_header_preview = e.get("state_header_preview")
                break

        at = AssistantTurnAudit(
            turn_id=turn_id,
            mode=mode or (tks_sorted[0].mode if tks_sorted else None),
            user_text=user_text,
            state_header_preview=state_header_preview,
            ticks=tks_sorted
        )

        # Cross-tick checks for Design A
        _check_dont_repeat_guard(at)

        assistant_turns.append(at)

    # 3) Summary
    tick_count = sum(len(at.ticks) for at in assistant_turns)
    v_total = 0
    by_kind: Dict[str, int] = {}
    by_code: Dict[str, int] = {}

    for at in assistant_turns:
        for v in at.violations:
            v_total += 1
            by_kind[v["kind"]] = by_kind.get(v["kind"], 0) + 1
            by_code[v["code"]] = by_code.get(v["code"], 0) + 1

        for tk in at.ticks:
            v_total += len(tk.violations)
            for v in tk.violations:
                by_kind[v["kind"]] = by_kind.get(v["kind"], 0) + 1
                by_code[v["code"]] = by_code.get(v["code"], 0) + 1

    report = AuditReportV2(
        audit_version="2.0.0-assistant_turn_ticks",
        generated_at_iso=datetime.now(tz=timezone.utc).isoformat(),
        inputs={
            "paths": [str(p) for p in input_paths],
            "file_count": len(input_paths),
            "event_count": len(events),
        },
        session_id=session_id,
        telemetry_id=telemetry_id,
        start_ts_iso=start_ts_iso,
        end_ts_iso=end_ts_iso,
        assistant_turns=assistant_turns,
        summary={
            "assistant_turns": len(assistant_turns),
            "ticks_total": tick_count,
            "violations_total": v_total,
            "violations_by_kind": by_kind,
            "violations_by_code": by_code,
        },
    )
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", nargs="+", required=True, help="Input .jsonl files and/or directories")
    ap.add_argument("--out", dest="out_dir", required=True, help="Output directory (will be created)")
    args = ap.parse_args()

    input_paths = _collect_inputs(args.inputs)
    if not input_paths:
        print("No .jsonl inputs found.", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = run_audit_v2(input_paths)

    audit_json_path = out_dir / "audit_v2.json"
    audit_md_path = out_dir / "audit_v2.md"
    timeline_md_path = out_dir / "timeline.md"

    with audit_json_path.open("w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(report), f, ensure_ascii=False, indent=2)

    with audit_md_path.open("w", encoding="utf-8") as f:
        f.write(_render_audit_md(report))

    with timeline_md_path.open("w", encoding="utf-8") as f:
        f.write(_render_timeline_md(report))

    print(f"Wrote: {audit_json_path}")
    print(f"Wrote: {audit_md_path}")
    print(f"Wrote: {timeline_md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())