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


# -----------------------------
# Helpers: loading + sorting
# -----------------------------

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
                # Keep going; audit should be robust to a corrupted line or two.
                continue


def _collect_inputs(inputs: List[str]) -> List[Path]:
    out: List[Path] = []
    for s in inputs:
        p = Path(s)
        if p.is_dir():
            out.extend(sorted(p.glob("*.jsonl")))
        elif p.is_file():
            out.append(p)
    # de-dupe
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


# -----------------------------
# Domain objects
# -----------------------------

@dataclass
class GridContext:
    raw_text: str = ""          # full extracted grid context (BEGIN..END)
    solvability: Optional[str] = None  # unique|multiple|none
    mismatch_indices: List[int] = dataclasses.field(default_factory=list)
    unresolved_indices: List[int] = dataclasses.field(default_factory=list)
    low_confidence_indices: List[int] = dataclasses.field(default_factory=list)
    auto_changed_indices: List[int] = dataclasses.field(default_factory=list)

    # confirmation facts (best effort)
    confirmed_indices: List[int] = dataclasses.field(default_factory=list)
    confirmed_count: Optional[int] = None

    conflicts_details_lines: Optional[List[str]] = None
    status_line: Optional[str] = None  # exact STATUS solvability line, if found
    parse_warnings: List[str] = dataclasses.field(default_factory=list)


@dataclass
class ToolPlanFinal:
    reply_len: Optional[int] = None
    ops: List[str] = dataclasses.field(default_factory=list)
    control: Optional[str] = None
    out_names: List[str] = dataclasses.field(default_factory=list)


@dataclass
class ControlArgs:
    """
    Best-effort extracted arguments for the control tool.
    Not always available from telemetry.
    """
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


# --- New: detect cell references/questions in reply text (audit robustness) ---
_RC_PATTERNS = [
    re.compile(r"\br\s*([1-9])\s*c\s*([1-9])\b", re.IGNORECASE),
    re.compile(r"\brow\s*([1-9])\s*(?:,|\s)*col(?:umn)?\s*([1-9])\b", re.IGNORECASE),
    re.compile(r"\brow\s*([1-9])\s*(?:,|\s)*column\s*([1-9])\b", re.IGNORECASE),
]

def _idx_from_rc(row: int, col: int) -> int:
    return (row - 1) * 9 + (col - 1)

def extract_cell_ref_from_text(text: Optional[str]) -> Optional[Tuple[int, int, int]]:
    """Return (idx,row,col) for the *most relevant* cell reference in free-form text.

    Heuristic: pick the *last* (row,col) mention, because the question/prompt is
    typically at the end of the reply (e.g., "Next, check row 1 col 2?").
    """
    if not text:
        return None
    hits: List[Tuple[int, int, int, int]] = []  # (pos, idx, row, col)
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
    """Heuristic: True if text appears to be asking to confirm/clarify a specific cell."""
    if not text:
        return False
    if extract_cell_ref_from_text(text) is None:
        return False
    t = text.lower()
    if "?" in t:
        return True
    return any(w in t for w in ["confirm", "should", "blank", "value", "what is in", "what's in", "is it"])


@dataclass
class CanonicalMessage:
    role: str
    content_preview: str
    content_len: int


@dataclass
class ObservedToolCall:
    """
    Tool calls discovered in the policy window (best-effort).
    We keep these small & bounded for troubleshooting.
    """
    name: str
    idx: Optional[int] = None
    row: Optional[int] = None
    col: Optional[int] = None
    digit: Optional[int] = None
    ev_type: Optional[str] = None


@dataclass
class TurnAudit:
    turn_id: int
    policy_req_seq: Optional[int] = None
    mode: Optional[str] = None
    user_text: Optional[str] = None
    state_header_preview: Optional[str] = None

    # Window timing
    window_start_ts_iso: Optional[str] = None
    window_end_ts_iso: Optional[str] = None
    window_start_seq: Optional[int] = None
    window_end_seq: Optional[int] = None

    # Prompt snapshot
    prompt_req_id: Optional[str] = None
    prompt_snapshot_note: Optional[str] = None  # explicitly state timing semantics

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
    confirmed_this_turn_indices: List[int] = dataclasses.field(default_factory=list)

    violations: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    notes: List[str] = dataclasses.field(default_factory=list)


@dataclass
class AuditReport:
    audit_version: str
    generated_at_iso: str
    inputs: Dict[str, Any]
    session_id: Optional[str]
    telemetry_id: Optional[str]
    start_ts_iso: Optional[str]
    end_ts_iso: Optional[str]
    turns: List[TurnAudit]
    summary: Dict[str, Any]


# -----------------------------
# Parsing: GRID_CONTEXT from prompt dumps
# -----------------------------

# Matches: - unresolved_indices: [8, 15, ...]
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

    # Parse solvability strictly from STATUS line:
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

    # Parse CONFLICTS_DETAILS block.
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


# -----------------------------
# Canonical history extraction (from messages_pretty_json)
# -----------------------------

def _cap_text(text: str, max_chars: int) -> str:
    t = (text or "").replace("\r", "\n")
    t = t.strip()
    if len(t) <= max_chars:
        return t
    return t[:max_chars].rstrip() + "…"


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


# -----------------------------
# Parsing: LLM_PROMPT_DUMP chunk assembly
# -----------------------------

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


# -----------------------------
# Windowing: policy call spans
# -----------------------------

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


# -----------------------------
# Best-effort extraction of tool args / reply text from window events
# -----------------------------

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
    # 1) tool list fields that already contain dicts
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

    # 2) POLICY_TRACE shape: parallel name + args-json lists
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

    # 3) raw args JSON fields (single tool)
    for key in ("raw_tool_args_json", "tool_args_json", "args_json", "arguments_json"):
        s = e.get(key)
        if isinstance(s, str):
            args_obj = _parse_args_json(s)
            if isinstance(args_obj, dict):
                yield "", args_obj, ev_type

    # 4) explicit control args
    ca = e.get("control_args")
    if isinstance(ca, dict):
        yield "", ca, ev_type


def _extract_idx_row_col_digit(args: Dict[str, Any]) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    """
    Normalize common arg shapes:
      - cellIndex / cell_index / idx / index
      - row/col (1-based)
      - digit / value
    """
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

    # If idx missing but row/col present, derive idx (expects 1..9)
    if idx is None and row is not None and col is not None:
        if 1 <= row <= 9 and 1 <= col <= 9:
            idx = (row - 1) * 9 + (col - 1)

    return idx, row, col, digit


def _extract_observed_tool_calls(window_events: List[Dict[str, Any]], cap: int = 40) -> List[ObservedToolCall]:
    """
    Collect a bounded list of observed tool calls inside the policy window.
    """
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
            if name != "reply":
                continue
            txt = args.get("text")
            if isinstance(txt, str) and txt.strip():
                return txt, len(txt)
    return None, None


# -----------------------------
# Checks: allowed next-check sets (your 4-case policy)
# -----------------------------

def _check_allowed_next_step(turn: TurnAudit) -> None:
    """
    Important timing rule (for humans reading the audit):
    - GRID_CONTEXT is extracted from messages_pretty_json inside THIS policy window.
    - That snapshot is the *pre-decision* state (i.e., at start of the policy call),
      before applying any tool ops in the same window (including consuming the user's
      answer, applying edits, autocorrect passes, etc.).
    Therefore:
    - It is valid for mismatch_indices to still include "the cell we are resolving
      right now", and the toolplan may BOTH resolve that mismatch and then
      recommend_validate in the same turn.
    """
    if not turn.grid or not turn.toolplan:
        turn.notes.append("Skipping next-step checks: missing GRID_CONTEXT or toolplan.")
        return

    solv = (turn.grid.solvability or "").lower()
    mismatch = turn.grid.mismatch_indices or []
    unresolved = turn.grid.unresolved_indices or []
    confirmed = set(turn.grid.confirmed_indices or [])

    control = turn.toolplan.control
    if control is None:
        return

    is_cell_check = control in ("ask_confirm_cell_rc", "ask_confirm_cell")
    asked_idx = turn.control_args.to_idx() if turn.control_args else None

    # New: detect whether the reply text is asking about a specific cell (independent of control).
    reply_ref = extract_cell_ref_from_text(getattr(turn, "reply_text", None))
    reply_asks_cell = text_looks_like_cell_question(getattr(turn, "reply_text", None))
    reply_asked_idx = reply_ref[0] if (reply_ref and reply_asks_cell) else None

    def fail(code: str, msg: str, details: Dict[str, Any]) -> None:
        turn.violations.append({"kind": "NEXT_STEP_POLICY", "code": code, "message": msg, "details": details})

    # Never re-ask confirmed cells (when we can identify idx)
    if is_cell_check and asked_idx is not None and asked_idx in confirmed:
        fail(
            "ASKED_CONFIRMED_CELL",
            "Control tool asked a cell that is already in confirmed_indices (should not re-ask).",
            {"asked_idx": asked_idx, "confirmed_count": len(confirmed), "control": control},
        )

    # Best-effort: treat mismatches that are being *resolved in this same policy window* as "addressed".
    # We primarily trust apply_user_edit*, because that implies the app accepted the user's confirmation/edit.
    resolved_this_turn = set(turn.applied_edit_indices or [])
    mismatch_set = set(mismatch)

    # ---- Case 1: mismatch has priority (with the timing-aware exception above)
    if mismatch:
        # If the only mismatch cells are being resolved in this same window,
        # then it is OK for the control to be recommend_validate.
        mismatch_remaining_after_resolve = mismatch_set - resolved_this_turn

        if control in ("recommend_validate", "recommend_retake") and not mismatch_remaining_after_resolve:
            # Allowed: resolve mismatch then validate/retake (same turn).
            return

        if control in ("recommend_validate", "recommend_retake"):
            fail(
                "CASE1_WRONG_CONTROL",
                "Mismatch exists in the pre-turn snapshot and is not fully resolved within this same policy window; control should be a cell confirmation.",
                {
                    "solvability": solv,
                    "mismatch_count": len(mismatch),
                    "mismatch_indices": mismatch[:30],
                    "resolved_this_turn_indices": sorted(resolved_this_turn)[:30],
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

    # ---- Case 2: solvability none
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

    # ---- Case 3: unique
    if solv == "unique":
        # New: if control recommends validate/retake, the reply should not ask to confirm a specific cell.
        if control in ("recommend_validate", "recommend_retake") and reply_asked_idx is not None:
            fail(
                "VALIDATE_REPLY_ASKS_CELL",
                "solvability==unique: control recommends validate/retake, but reply text asks about a specific cell.",
                {
                    "control": control,
                    "reply_asked_idx": reply_asked_idx,
                    "reply_excerpt": (getattr(turn, "reply_text", "") or "")[:220],
                    "confirmed_this_turn_indices": turn.confirmed_this_turn_indices,
                },
            )
            if turn.confirmed_this_turn_indices and reply_asked_idx not in set(turn.confirmed_this_turn_indices):
                fail(
                    "REPLY_CELL_DRIFT_FROM_CONFIRMED",
                    "Reply asks about a different cell than the one confirmed this turn.",
                    {
                        "confirmed_this_turn_indices": turn.confirmed_this_turn_indices,
                        "reply_asked_idx": reply_asked_idx,
                        "reply_excerpt": (getattr(turn, "reply_text", "") or "")[:220],
                    },
                )

        if is_cell_check:
            fail(
                "CASE3_CELL_CHECK_FORBIDDEN",
                "solvability==unique and mismatch empty: should stop cell-by-cell checks and use recommend_validate.",
                {"control": control, "asked_idx": asked_idx},
            )
        return

    # ---- Case 4: multiple
    if solv == "multiple":
        if is_cell_check:
            fail(
                "CASE4_CELL_CHECK_FORBIDDEN",
                "solvability==multiple and mismatch empty: should not do cell-by-cell verification; ask for match via recommend_validate.",
                {"control": control, "asked_idx": asked_idx},
            )
        return

    if solv not in ("unique", "multiple", "none", ""):
        turn.notes.append(f"Unknown solvability value: {solv!r}")


# -----------------------------
# Optional: lightweight grounding heuristics (regex based)
# -----------------------------

def _check_grounding_heuristics(turn: TurnAudit, assistant_text: Optional[str]) -> None:
    if not assistant_text or not turn.grid:
        return

    text = assistant_text.lower()
    solv = (turn.grid.solvability or "").lower()
    conflicts = turn.grid.conflicts_details_lines or []

    def fail(code: str, msg: str, details: Dict[str, Any]) -> None:
        turn.violations.append({"kind": "GROUNDING", "code": code, "message": msg, "details": details})

    if "unique" in text and solv != "unique":
        fail(
            "CLAIM_UNIQUE_NOT_SUPPORTED",
            "Assistant text mentions 'unique' but GRID_CONTEXT solvability is not unique.",
            {"solvability": solv, "status_line": turn.grid.status_line},
        )

    if "multiple" in text and solv != "multiple":
        fail(
            "CLAIM_MULTIPLE_NOT_SUPPORTED",
            "Assistant text mentions 'multiple solutions' but GRID_CONTEXT solvability is not multiple.",
            {"solvability": solv, "status_line": turn.grid.status_line},
        )

    contradiction_words = ("contradiction", "conflict", "duplicate", "two ")
    if any(w in text for w in contradiction_words):
        if len(conflicts) == 1 and conflicts[0].strip().lower() in ("(none)", "- (none)", "none"):
            fail(
                "CLAIM_CONFLICT_WITH_NONE",
                "Assistant implies a conflict but CONFLICTS_DETAILS indicates none.",
                {"conflicts_details": conflicts},
            )


# -----------------------------
# Report rendering
# -----------------------------

def _to_iso(ts_epoch_ms: Optional[int]) -> Optional[str]:
    if ts_epoch_ms is None:
        return None
    try:
        dt = datetime.fromtimestamp(int(ts_epoch_ms) / 1000.0, tz=timezone.utc)
        return dt.isoformat()
    except Exception:
        return None


def _render_markdown(report: AuditReport) -> str:
    lines: List[str] = []
    lines.append("# Telemetry Audit Report")
    lines.append("")
    lines.append(f"- Generated: `{report.generated_at_iso}`")
    lines.append(f"- Audit version: `{report.audit_version}`")
    lines.append(f"- Session: `{report.session_id}`  Telemetry: `{report.telemetry_id}`")
    lines.append(f"- Range: `{report.start_ts_iso}` → `{report.end_ts_iso}`")
    lines.append("")
    lines.append("## Reading guide (timing semantics)")
    lines.append("")
    lines.append("- Each **Turn** corresponds to one `LLM_CALLPOLICY_BEGIN..END` window.")
    lines.append("- `GRID_CONTEXT` is parsed from `messages_pretty_json` **inside that same window** and represents the **pre-decision snapshot** (state at the moment the policy call is made).")
    lines.append("- Tool calls listed under `observed_tools` happen **during** that window; they may resolve items that are still present in the pre-decision `mismatch_indices` / `unresolved_indices` lists.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    for k, v in report.summary.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("## Turns")
    lines.append("")
    for t in report.turns:
        lines.append(f"### Turn {t.turn_id} ({t.mode})")
        if t.policy_req_seq is not None:
            lines.append(f"- policy_req_seq: {t.policy_req_seq}")
        if t.window_start_ts_iso or t.window_end_ts_iso:
            lines.append(f"- window_ts: `{t.window_start_ts_iso}` → `{t.window_end_ts_iso}` (seq {t.window_start_seq}..{t.window_end_seq})")
        if t.user_text:
            lines.append(f"- User: {t.user_text}")
        if t.reply_text:
            lines.append(f"- reply_len: {t.reply_text_len}  reply_preview: `{_cap_text(t.reply_text, 220)}`")
        if t.canonical_history_count is not None:
            lines.append(f"- canonical_history: count={t.canonical_history_count} sha256={t.canonical_history_sha256}")
        if t.prompt_req_id:
            lines.append(f"- prompt_req_id_used: {t.prompt_req_id}")
        if t.prompt_snapshot_note:
            lines.append(f"- snapshot_note: {t.prompt_snapshot_note}")
        if t.grid:
            lines.append(
                f"- solvability: `{t.grid.solvability}`  "
                f"mismatch:{len(t.grid.mismatch_indices)} unresolved:{len(t.grid.unresolved_indices)} "
                f"confirmed:{t.grid.confirmed_count}"
            )
            if t.grid.status_line:
                lines.append(f"- status_line: `{t.grid.status_line}`")
            if t.grid.mismatch_indices:
                lines.append(f"- mismatch_indices: {t.grid.mismatch_indices}")
            if t.grid.unresolved_indices:
                lines.append(f"- unresolved_indices: {t.grid.unresolved_indices[:50]}" + (" …" if len(t.grid.unresolved_indices) > 50 else ""))
            if t.grid.parse_warnings:
                lines.append("- grid_parse_warnings:")
                for w in t.grid.parse_warnings:
                    lines.append(f"  - {w}")
        if t.toolplan:
            lines.append(f"- toolplan: control=`{t.toolplan.control}` ops={t.toolplan.ops} out={t.toolplan.out_names}")
        if t.control_args and (t.control_args.row or t.control_args.col or t.control_args.cell_index is not None):
            lines.append(
                f"- control_args: row={t.control_args.row} col={t.control_args.col} "
                f"cell_index={t.control_args.cell_index} digit={t.control_args.digit} "
                f"(source={t.control_args.source_event_type})"
            )
        if t.applied_edit_indices or t.confirmed_this_turn_indices:
            lines.append(f"- window_effects: applied_edits={t.applied_edit_indices} confirmed={t.confirmed_this_turn_indices}")

        if t.observed_tools:
            # show compact (name@idx)
            compact = []
            for oc in t.observed_tools[:25]:
                if oc.idx is not None:
                    compact.append(f"{oc.name}@{oc.idx}")
                elif oc.row is not None and oc.col is not None:
                    compact.append(f"{oc.name}@r{oc.row}c{oc.col}")
                else:
                    compact.append(f"{oc.name}")
            lines.append(f"- observed_tools_sample: {compact}" + (" …" if len(t.observed_tools) > 25 else ""))

        if t.violations:
            lines.append("")
            lines.append("**Violations:**")
            for v in t.violations:
                lines.append(f"- [{v['kind']}] {v['code']}: {v['message']}")
        if t.notes:
            lines.append("")
            lines.append("Notes:")
            for n in t.notes:
                lines.append(f"- {n}")
        lines.append("")
    return "\n".join(lines)


# -----------------------------
# Main audit pipeline
# -----------------------------

def run_audit(input_paths: List[Path]) -> AuditReport:
    events: List[Dict[str, Any]] = []
    for p in input_paths:
        events.extend(list(_iter_jsonl(p)))
    events.sort(key=_sort_key)

    session_id = events[0].get("session_id") if events else None
    telemetry_id = events[0].get("telemetry_id") if events else None
    start_ts_iso = _to_iso(events[0].get("ts_epoch_ms")) if events else None
    end_ts_iso = _to_iso(events[-1].get("ts_epoch_ms")) if events else None

    # Pre-assemble prompt dumps (messages_pretty_json)
    messages_by_req = _assemble_chunks(events, label="messages_pretty_json")

    # Index toolplans by turn_id
    toolplan_by_turn: Dict[int, ToolPlanFinal] = {}
    for e in events:
        if e.get("type") == "LLM_TOOLPLAN_FINALIZED":
            tid = e.get("turn_id")
            if isinstance(tid, int):
                toolplan_by_turn[tid] = ToolPlanFinal(
                    reply_len=e.get("reply_len"),
                    ops=list(e.get("ops") or []),
                    control=e.get("control"),
                    out_names=list(e.get("out_names") or []),
                )

    turns: List[TurnAudit] = []
    windows = _policy_windows(events)

    for (b_i, e_i, b_ev, e_ev) in windows:
        turn_id = b_ev.get("turn_id")
        if not isinstance(turn_id, int):
            continue

        ta = TurnAudit(
            turn_id=turn_id,
            policy_req_seq=b_ev.get("policy_req_seq") if isinstance(b_ev.get("policy_req_seq"), int) else None,
            mode=b_ev.get("mode"),
            user_text=b_ev.get("user_text"),
            state_header_preview=b_ev.get("state_header_preview"),
            window_start_ts_iso=_to_iso(b_ev.get("ts_epoch_ms")),
            window_end_ts_iso=_to_iso(e_ev.get("ts_epoch_ms")),
            window_start_seq=b_ev.get("seq") if isinstance(b_ev.get("seq"), int) else None,
            window_end_seq=e_ev.get("seq") if isinstance(e_ev.get("seq"), int) else None,
        )

        window_events = events[b_i:e_i + 1]

        # Find latest req_id in this window that has messages_pretty_json to parse GRID_CONTEXT + history
        req_ids: List[str] = []
        for we in window_events:
            rid = we.get("req_id")
            if rid and rid in messages_by_req:
                req_ids.append(rid)

        if req_ids:
            rid = req_ids[-1]
            ta.prompt_req_id = rid
            msgs_text = messages_by_req[rid]

            ta.prompt_snapshot_note = (
                "GRID_CONTEXT parsed from messages_pretty_json in this policy window. "
                "Interpret as PRE-DECISION snapshot (state at the moment policy is called), "
                "before any tool ops in the same window apply the user's answer / edits."
            )

            # canonical history
            hist, cnt, sha = _extract_canonical_history(msgs_text)
            ta.canonical_history = hist
            ta.canonical_history_count = cnt
            ta.canonical_history_sha256 = sha

            # GRID_CONTEXT
            ta.grid = _extract_grid_context_from_messages_pretty_json(msgs_text)
        else:
            ta.notes.append("No messages_pretty_json found in this policy window; cannot parse GRID_CONTEXT or canonical history.")

        ta.toolplan = toolplan_by_turn.get(turn_id)

        # Reply text (from POLICY_TRACE tools-parsed, best effort)
        rt, rtl = _try_extract_reply_text(window_events)
        ta.reply_text = rt
        ta.reply_text_len = rtl

        # Try to extract control args for precise allowed-set membership checks
        if ta.toolplan and ta.toolplan.control:
            ta.control_args = _try_extract_control_args(window_events, ta.toolplan.control)

        # Observed tools (useful when a pre-turn snapshot still contains the cell being resolved)
        ta.observed_tools = _extract_observed_tool_calls(window_events)

        # Derive "effects" inside window:
        applied = set()
        confirmed_now = set()
        for oc in ta.observed_tools or []:
            n = (oc.name or "").lower()
            if oc.idx is None:
                continue
            if "apply_user_edit" in n:
                applied.add(oc.idx)
            if "confirm_cell_value" in n:
                confirmed_now.add(oc.idx)
        ta.applied_edit_indices = sorted(applied)
        ta.confirmed_this_turn_indices = sorted(confirmed_now)

        # Checks
        _check_allowed_next_step(ta)

        # Optional grounding checks (needs assistant text; keep disabled by default)
        _check_grounding_heuristics(ta, assistant_text=None)

        turns.append(ta)

    # Summary
    vcount = 0
    by_kind: Dict[str, int] = {}
    by_code: Dict[str, int] = {}
    for t in turns:
        vcount += len(t.violations)
        for v in t.violations:
            by_kind[v["kind"]] = by_kind.get(v["kind"], 0) + 1
            by_code[v["code"]] = by_code.get(v["code"], 0) + 1

    report = AuditReport(
        audit_version="1.3.0+cellaskfix",
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
        turns=turns,
        summary={
            "turns_audited": len(turns),
            "violations_total": vcount,
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

    report = run_audit(input_paths)

    audit_json_path = out_dir / "audit.json"
    audit_md_path = out_dir / "audit.md"

    with audit_json_path.open("w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(report), f, ensure_ascii=False, indent=2)

    with audit_md_path.open("w", encoding="utf-8") as f:
        f.write(_render_markdown(report))

    print(f"Wrote: {audit_json_path}")
    print(f"Wrote: {audit_md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())