# tools/telemetry_audit/telemetry_audit.py
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -------------------------
# Small helpers
# -------------------------

def _safe_read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def _try_json_loads(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        return None


def _iso_min(a: Optional[str], b: Optional[str]) -> Optional[str]:
    if not a:
        return b
    if not b:
        return a
    return a if a <= b else b


def _iso_max(a: Optional[str], b: Optional[str]) -> Optional[str]:
    if not a:
        return b
    if not b:
        return a
    return a if a >= b else b


def _md_escape(s: str) -> str:
    return (s or "").replace("\r", "")


def _json_pretty(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=False)


def _as_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _parse_preview_json_or_raw(preview: Any) -> Dict[str, Any]:
    if isinstance(preview, (dict, list)):
        return {"parsed": preview}
    if not isinstance(preview, str) or not preview.strip():
        return {"raw": ""}
    parsed = _try_json_loads(preview)
    if parsed is not None:
        return {"parsed": parsed}
    return {"raw": preview}


def _first_nonempty_str(*vals: Any) -> Optional[str]:
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v
    return None


def _safe_get_dict(d: Any) -> Dict[str, Any]:
    return d if isinstance(d, dict) else {}


def _safe_get_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []


def _cell_ref(ci: int) -> str:
    r = (ci // 9) + 1
    c = (ci % 9) + 1
    return f"r{r}c{c}"


def _cell_ref_verbose(ci: int) -> str:
    return f"{_cell_ref(ci)} ({ci})"


def _digits_from_mask(mask: int) -> List[int]:
    out: List[int] = []
    for d in range(1, 10):
        if mask & (1 << (d - 1)):
            out.append(d)
    return out


def _bitcount(mask: int) -> int:
    try:
        return int(mask).bit_count()  # type: ignore[attr-defined]
    except Exception:
        m = int(mask)
        c = 0
        while m:
            m &= m - 1
            c += 1
        return c


def _displayed81_to_grid81(displayed81: str) -> str:
    s = (displayed81 or "").strip()
    if len(s) != 81:
        return s
    return "".join("." if ch in ("0",) else ch for ch in s)


def _pick_house_name(h: Dict[str, Any]) -> str:
    ht = str(h.get("type") or "")
    if ht == "cell":
        cell = _safe_get_dict(h.get("cell"))
        ci = _as_int(cell.get("cellIndex"), -1)
        return f"cell {_cell_ref(ci)}" if 0 <= ci <= 80 else "cell"
    idx = _as_int(h.get("index1to9"), -1)
    if ht == "row":
        return f"row {idx}"
    if ht == "col":
        return f"col {idx}"
    if ht == "box":
        return f"box {idx}"
    if ht == "region":
        rid = h.get("regionId")
        return f"region {rid}"
    return f"{ht} {idx}"


def _house_compact(h: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(h, dict):
        return None
    ht = str(h.get("type") or "")
    if not ht:
        return None
    if ht == "cell":
        cell = _parse_cell_ref_like(h.get("cell"))
        if not cell:
            return None
        return {"type": "cell", "cell": cell}
    out: Dict[str, Any] = {"type": ht}
    if ht == "region":
        out["regionId"] = h.get("regionId")
        return out
    idx = _as_int(h.get("index1to9"), -1)
    if idx in range(1, 10):
        out["index1to9"] = idx
        return out
    return None


def _parse_cell_ref_like(raw: Any) -> Optional[Dict[str, Any]]:
    if isinstance(raw, int):
        if 0 <= raw <= 80:
            return {"cellIndex": raw, "r": (raw // 9) + 1, "c": (raw % 9) + 1}
        return None

    if not isinstance(raw, dict):
        return None

    if isinstance(raw.get("cell"), dict):
        return _parse_cell_ref_like(raw.get("cell"))

    ci = _as_int(raw.get("cellIndex", raw.get("cell_index")), -1)
    if ci not in range(0, 81):
        return None

    r = _as_int(raw.get("r"), (ci // 9) + 1)
    c = _as_int(raw.get("c"), (ci % 9) + 1)
    if r not in range(1, 10) or c not in range(1, 10):
        return None

    return {"cellIndex": ci, "r": r, "c": c}


def _cell_to_verbose(cell: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(cell, dict):
        return None
    ci = _as_int(cell.get("cellIndex"), -1)
    if ci not in range(0, 81):
        return None
    return _cell_ref_verbose(ci)


def _cells_to_verbose(raw_cells: Any) -> List[str]:
    out: List[str] = []
    items = raw_cells if isinstance(raw_cells, list) else []
    for raw in items:
        cell = _parse_cell_ref_like(raw)
        s = _cell_to_verbose(cell)
        if s:
            out.append(s)
    return out


def _cells_to_indices(raw_cells: Any) -> List[int]:
    out: List[int] = []
    items = raw_cells if isinstance(raw_cells, list) else []
    for raw in items:
        cell = _parse_cell_ref_like(raw)
        if isinstance(cell, dict):
            ci = _as_int(cell.get("cellIndex"), -1)
            if 0 <= ci <= 80:
                out.append(ci)
    return out


def _parse_house_like(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    return _house_compact(raw)


def _house_list_compact(raw: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    items = raw if isinstance(raw, list) else []
    for x in items:
        h = _parse_house_like(x)
        if h:
            out.append(h)
    return out


def _pairwise_dedup_dicts(xs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for x in xs:
        key = json.dumps(x, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(x)
    return out


# -------------------------
# Telemetry indexing
# -------------------------

@dataclass
class Event:
    raw: Dict[str, Any]

    @property
    def type(self) -> str:
        return self.raw.get("type", "")

    @property
    def tag(self) -> str:
        return self.raw.get("tag", "")

    @property
    def seq(self) -> int:
        return _as_int(self.raw.get("seq", 0), 0)

    @property
    def ts_iso(self) -> Optional[str]:
        return self.raw.get("ts_iso")

    @property
    def turn_id(self) -> Optional[int]:
        v = self.raw.get("turn_id")
        if isinstance(v, int):
            return v
        if isinstance(v, str) and v.isdigit():
            return int(v)
        return None

    @property
    def tick_id(self) -> Optional[int]:
        v = self.raw.get("tick_id")
        if isinstance(v, int):
            return v
        if isinstance(v, str) and v.isdigit():
            return int(v)
        return None

    def get(self, k: str, default=None):
        return self.raw.get(k, default)


def _parse_jsonl_lenient(text: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    buf = ""

    for line in text.splitlines():
        line = line.rstrip("\n")
        if not line.strip():
            continue

        if not buf:
            buf = line
        else:
            buf += "\n" + line

        try:
            obj = json.loads(buf)
            if isinstance(obj, dict):
                out.append(obj)
            buf = ""
        except json.JSONDecodeError:
            continue
        except Exception:
            buf = ""

    return out


def load_events_from_folder(folder: Path) -> List[Event]:
    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"--in must be a folder. Not found: {folder}")

    files: List[Path] = []
    for ext in ("*.jsonl", "*.json"):
        files.extend(sorted(folder.glob(ext)))

    if not files:
        raise SystemExit(f"No .jsonl/.json telemetry files found under: {folder}")

    out: List[Event] = []
    for fp in files:
        text = _safe_read_text(fp).strip()
        if not text:
            continue

        if fp.suffix.lower() == ".jsonl":
            objs = _parse_jsonl_lenient(text)
            for obj in objs:
                if isinstance(obj, dict):
                    obj.setdefault("_source_file", str(fp))
                    out.append(Event(obj))
            continue

        obj = _try_json_loads(text)
        if isinstance(obj, list):
            for it in obj:
                if isinstance(it, dict):
                    it.setdefault("_source_file", str(fp))
                    out.append(Event(it))
        elif isinstance(obj, dict):
            obj.setdefault("_source_file", str(fp))
            out.append(Event(obj))
        else:
            objs = _parse_jsonl_lenient(text)
            for obj2 in objs:
                if isinstance(obj2, dict):
                    obj2.setdefault("_source_file", str(fp))
                    out.append(Event(obj2))

    out.sort(key=lambda e: (e.seq, e.ts_iso or "", e.type, e.tag))
    return out


def find_session_id(events: List[Event]) -> str:
    for e in events:
        sid = e.get("session_id")
        if isinstance(sid, str) and sid.strip():
            return sid.strip()
    return "unknown_session"


def compute_time_bounds(events: List[Event]) -> Tuple[Optional[str], Optional[str]]:
    start = None
    end = None
    for e in events:
        ts = e.ts_iso
        if not ts:
            continue
        start = _iso_min(start, ts)
        end = _iso_max(end, ts)
    return start, end


def group_by_turn(events: List[Event]) -> Dict[int, List[Event]]:
    turns: Dict[int, List[Event]] = {}
    for e in events:
        tid = e.turn_id
        if tid is None:
            continue
        turns.setdefault(tid, []).append(e)
    return dict(sorted(turns.items(), key=lambda kv: kv[0]))


# -------------------------
# Extraction: agenda audit
# -------------------------

def extract_agenda_audit_events(turn_events: List[Event]) -> Dict[str, Any]:
    wanted = {
        "AGENDA_SELECTED",
        "CTA_SELECTED",
        "AGENDA_TRANSITION",
        "OPS_PLANNED",
        "SOLVING_AGENDA_TRANSITION",
        "AGENDA_STATE_SNAPSHOT",
        "SOLVING_AGENDA_STATE_SNAPSHOT",
    }
    events_out: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {k: 0 for k in wanted}

    for e in turn_events:
        if e.type not in wanted:
            continue

        counts[e.type] = counts.get(e.type, 0) + 1

        payload = dict(e.raw)
        payload.pop("_source_file", None)

        events_out.append({
            "type": e.type,
            "tag": e.tag or None,
            "turn_id": e.turn_id,
            "tick_id": e.tick_id,
            "seq": e.seq,
            "ts_iso": e.ts_iso,
            "payload": payload,
        })

    events_out.sort(key=lambda x: (_as_int(x.get("seq"), 0), x.get("ts_iso") or "", x.get("type") or ""))
    counts_trimmed = {k: v for k, v in counts.items() if v}

    return {"events": events_out, "counts": counts_trimmed}


# -------------------------
# Extraction: turn context
# -------------------------

def _pick_first(
    events: List[Event],
    t_or_tag: str,
    tick: Optional[int] = None,
    payload_kind: Optional[str] = None,
) -> Optional[Event]:
    for e in events:
        is_match = (e.type == t_or_tag) or (e.type == "POLICY_TRACE" and e.tag == t_or_tag)
        if not is_match:
            continue
        if tick is not None and e.tick_id != tick:
            continue
        if payload_kind is not None and e.get("payload_kind") != payload_kind:
            continue
        return e
    return None


def _pick_last(events: List[Event], t_or_tag: str, tick: Optional[int] = None) -> Optional[Event]:
    last: Optional[Event] = None
    for e in events:
        is_match = (e.type == t_or_tag) or (e.type == "POLICY_TRACE" and e.tag == t_or_tag)
        if not is_match:
            continue
        if tick is not None and e.tick_id != tick:
            continue
        last = e
    return last


def extract_turn_context_v1(turn_events: List[Event]) -> Dict[str, Any]:
    ev = _pick_last(turn_events, "TURN_CONTEXT_V1", tick=1)
    if not ev:
        return {}

    raw = _first_nonempty_str(
        ev.get("turn_ctx_json"),
        ev.get("turn_ctx_preview"),
        ev.get("payload_text"),
        ev.get("payload_preview"),
    )
    parsed = _try_json_loads(raw) if isinstance(raw, str) and raw.strip() else None

    def _parse_json_field(x: Any) -> Optional[Dict[str, Any]]:
        if isinstance(x, dict):
            return x
        if isinstance(x, str) and x.strip():
            y = _try_json_loads(x)
            if isinstance(y, dict):
                return y
        return None

    # TURN_CONTEXT_V1 currently does not carry relationship_memory in your app telemetry.
    # Enrich it from TURN_SUMMARY_SNAPSHOT for the same turn.
    summary_ev = _pick_last(turn_events, "TURN_SUMMARY_SNAPSHOT")

    summary_user_tally = _parse_json_field(summary_ev.get("user_tally_json")) if summary_ev else None
    summary_assistant_tally = _parse_json_field(summary_ev.get("assistant_tally_json")) if summary_ev else None
    summary_relationship_memory = _parse_json_field(summary_ev.get("relationship_memory_json")) if summary_ev else None
    summary_recent_turns = None
    if summary_ev:
        rt = summary_ev.get("recent_turns_json")
        if isinstance(rt, str) and rt.strip():
            maybe_rt = _try_json_loads(rt)
            if isinstance(maybe_rt, list):
                summary_recent_turns = maybe_rt

    parsed_enriched: Optional[Dict[str, Any]] = None
    if isinstance(parsed, dict):
        parsed_enriched = dict(parsed)

        tally_obj = _safe_get_dict(parsed_enriched.get("tally"))
        if summary_user_tally or summary_assistant_tally:
            tally_obj = dict(tally_obj)
            if summary_user_tally:
                tally_obj["user_tally"] = summary_user_tally
            if summary_assistant_tally:
                tally_obj["assistant_tally"] = summary_assistant_tally
            parsed_enriched["tally"] = tally_obj

        if summary_relationship_memory:
            parsed_enriched["relationship_memory"] = summary_relationship_memory

        if summary_recent_turns is not None and "recent_turns" not in parsed_enriched:
            parsed_enriched["recent_turns"] = summary_recent_turns

    curated: Dict[str, Any] = {}
    if isinstance(parsed_enriched, dict):
        curated = {
            "schema": parsed_enriched.get("schema"),
            "v": parsed_enriched.get("v"),
            "turn_id": parsed_enriched.get("turn_id"),
            "mode": parsed_enriched.get("mode"),
            "phase": parsed_enriched.get("phase"),
            "user_text": parsed_enriched.get("user_text"),
            "asr": _safe_get_dict(parsed_enriched.get("asr")) if parsed_enriched.get("asr") is not None else None,
            "pending": _safe_get_dict(parsed_enriched.get("pending")) if parsed_enriched.get("pending") is not None else None,
            "focus": _safe_get_dict(parsed_enriched.get("focus")) if parsed_enriched.get("focus") is not None else None,
            "discourse_state": _safe_get_dict(parsed_enriched.get("discourse_state")) if parsed_enriched.get("discourse_state") is not None else None,
            "grid_context": _safe_get_dict(parsed_enriched.get("grid_context")) if parsed_enriched.get("grid_context") is not None else None,
            "recent_turns_count": len(_safe_get_list(parsed_enriched.get("recent_turns"))),
            "tally": _safe_get_dict(parsed_enriched.get("tally")) if parsed_enriched.get("tally") is not None else None,
            "relationship_memory": _safe_get_dict(parsed_enriched.get("relationship_memory")) if parsed_enriched.get("relationship_memory") is not None else None,
        }

        if isinstance(curated.get("asr"), dict):
            asr = curated["asr"]
            curated["asr"] = {
                "lang_hint": asr.get("lang_hint"),
                "confidence": asr.get("confidence"),
                "nbest": asr.get("nbest") if isinstance(asr.get("nbest"), list) else None,
            }

        if isinstance(curated.get("recent_turns_count"), int) and curated["recent_turns_count"] == 0:
            curated.pop("recent_turns_count", None)

        if not curated.get("tally"):
            curated.pop("tally", None)

        if not curated.get("relationship_memory"):
            curated.pop("relationship_memory", None)

    return {
        "meta": {
            "type_or_tag": ev.type if ev.type != "POLICY_TRACE" else (ev.tag or ev.type),
            "turn_id": ev.get("turn_id"),
            "tick_id": ev.get("tick_id"),
            "policy_req_seq": ev.get("policy_req_seq"),
            "correlation_id": ev.get("correlation_id"),
            "model_call_id": ev.get("model_call_id"),
            "toolplan_id": ev.get("toolplan_id"),
            "seq": ev.get("seq"),
            "ts_iso": ev.get("ts_iso"),
            "reason": ev.get("reason"),
            "turn_ctx_sha12": ev.get("turn_ctx_sha12"),
            "turn_ctx_len": ev.get("turn_ctx_len"),
            "relationship_memory_source": (
                "TURN_SUMMARY_SNAPSHOT" if summary_relationship_memory else None
            ),
        },
        "parsed": parsed_enriched if isinstance(parsed_enriched, dict) else (parsed if isinstance(parsed, dict) else None),
        "curated": curated if curated else None,
        "raw_preview": (raw[:900] + ("…" if isinstance(raw, str) and len(raw) > 900 else "")) if isinstance(raw, str) else "",
    }


# -------------------------
# Extraction: transcript
# -------------------------

def extract_spoken_transcript(events: List[Event]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for e in events:
        if e.type == "USER_SAY":
            items.append({"ts_iso": e.ts_iso, "speaker": "User", "text": e.get("text", "")})
        elif e.type == "ASSISTANT_SAY":
            items.append({"ts_iso": e.ts_iso, "speaker": "Assistant", "text": e.get("text", "")})
    return items



def extract_turn_conversation_transcript(turn_events: List[Event]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    # Preferred source: explicit USER_SAY / ASSISTANT_SAY events for this turn.
    for e in turn_events:
        if e.type == "USER_SAY":
            txt = e.get("text", "")
            if isinstance(txt, str) and txt.strip():
                items.append({"speaker": "User", "text": txt.strip()})
        elif e.type == "ASSISTANT_SAY":
            txt = e.get("text", "")
            if isinstance(txt, str) and txt.strip():
                items.append({"speaker": "Assistant", "text": txt.strip()})

    if items:
        return items

    # Fallback 1: Turn context user_text + reply model payload-in response text.
    turn_ctx = extract_turn_context_v1(turn_events)
    parsed_ctx = turn_ctx.get("parsed") if isinstance(turn_ctx, dict) else None

    user_text = None
    if isinstance(parsed_ctx, dict):
        ut = parsed_ctx.get("user_text")
        if isinstance(ut, str) and ut.strip() and not ut.strip().startswith("[EVENT]"):
            user_text = ut.strip()

    assistant_text = None
    reply_block = extract_model_call_block(turn_events, channel="REPLY", tick_id=2)
    payload_in = _safe_get_dict(reply_block.get("payload_in")).get("response_text")
    if isinstance(payload_in, str) and payload_in.strip():
        # Some logs store plain assistant text directly.
        maybe = payload_in.strip()
        parsed = _try_json_loads(maybe)
        if isinstance(parsed, str):
            assistant_text = parsed.strip()
        else:
            assistant_text = maybe

    if user_text:
        items.append({"speaker": "User", "text": user_text})
    if assistant_text:
        items.append({"speaker": "Assistant", "text": assistant_text})

    return items



# -------------------------
# Extraction: model call blocks
# -------------------------

def _extract_assistant_content_from_openai_response(response_text: str) -> Tuple[Optional[str], Optional[Any]]:
    body = _try_json_loads(response_text)
    if not isinstance(body, dict):
        return None, None

    choices = body.get("choices")
    if not (isinstance(choices, list) and choices):
        return None, None

    msg = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(msg, dict):
        return None, None

    content = msg.get("content")
    if not isinstance(content, str):
        return None, None

    parsed = _try_json_loads(content)
    return content, parsed


def extract_model_call_block(turn_events: List[Event], channel: str, tick_id: int) -> Dict[str, Any]:
    expected_kind = None
    if channel == "MEANING":
        expected_kind = "meaning_extract_v1"
    elif channel == "REPLY":
        expected_kind = "reply_generate_v1"

    # First try the strict selector.
    out_ev = _pick_first(turn_events, "MODEL_PAYLOAD_OUT", tick=tick_id, payload_kind=expected_kind)
    in_ev = _pick_first(turn_events, "MODEL_PAYLOAD_IN", tick=tick_id, payload_kind=expected_kind)

    # Fallback: some telemetry files do not populate payload_kind on MODEL_PAYLOAD_*.
    if out_ev is None:
        out_ev = _pick_first(turn_events, "MODEL_PAYLOAD_OUT", tick=tick_id)
    if in_ev is None:
        in_ev = _pick_first(turn_events, "MODEL_PAYLOAD_IN", tick=tick_id)

    block: Dict[str, Any] = {
        "channel": channel,
        "tick_id": tick_id,
        "payload_kind": expected_kind,
        "ids": {},
        "req": {},
        "res": {},
        "payload_out": {},
        "payload_in": {},
        "extracted_assistant_content": None,
        "extracted_assistant_content_parsed": None,
    }

    src = out_ev or in_ev
    if src:
        block["ids"] = {
            "model_call_id": src.get("model_call_id"),
            "toolplan_id": src.get("toolplan_id"),
            "correlation_id": src.get("correlation_id"),
            "policy_req_seq": src.get("policy_req_seq"),
            "turn_id": src.get("turn_id"),
            "tick_id": src.get("tick_id"),
        }

    if out_ev:
        block["req"] = {
            "req_id": out_ev.get("req_id"),
            "payload_sha12": out_ev.get("payload_sha12"),
            "payload_len": out_ev.get("payload_len"),
            "payload_kind": out_ev.get("payload_kind"),
        }
        block["payload_out"] = {
            "payload_text": out_ev.get("payload_text") or out_ev.get("payload_preview"),
            "payload_preview": out_ev.get("payload_preview"),
        }

    if in_ev:
        block["res"] = {
            "http_code": in_ev.get("http_code"),
            "dt_ms": in_ev.get("dt_ms"),
            "response_sha12": in_ev.get("response_sha12"),
            "response_len": in_ev.get("response_len"),
            "parse_ok": in_ev.get("parse_ok"),
            "parse_errors": in_ev.get("parse_errors"),
            "req_id": in_ev.get("req_id"),
            "payload_sha12": in_ev.get("payload_sha12"),
            "payload_kind": in_ev.get("payload_kind"),
        }
        block["payload_in"] = {
            "response_text": in_ev.get("response_text") or in_ev.get("payload_preview"),
            "payload_preview": in_ev.get("payload_preview"),
        }

        resp_text = in_ev.get("response_text") or in_ev.get("payload_preview")
        if isinstance(resp_text, str) and resp_text.strip():
            raw_content, parsed = _extract_assistant_content_from_openai_response(resp_text)
            block["extracted_assistant_content"] = raw_content
            block["extracted_assistant_content_parsed"] = parsed

    return block


# -------------------------
# Extraction: grid facts snapshots
# -------------------------

_GRID_FACTS_TAG_CANDIDATES = [
    "GRID_FACTS_SNAPSHOT_V1",
    "grid_facts_snapshot_v1",
]


def _curate_grid_context(parsed: Dict[str, Any]) -> Dict[str, Any]:
    keep_keys = [
        "displayed81",
        "truth_givens81",
        "truth_solution81",
        "deduced_unique_solution81",
        "solvability",
        "is_structurally_valid",
        "severity",
        "retake_recommendation",
        "confirmed_indices",
        "unresolved_indices",
        "mismatch_indices",
        "conflict_indices",
        "low_confidence_indices",
        "manual_corrected_indices",
        "auto_changed_indices",
        "unresolved_count",
        "confirmed_indices_count",
        "unresolved_indices_count",
        "mismatch_indices_count",
        "conflict_indices_count",
        "low_confidence_indices_count",
        "manual_corrected_indices_count",
        "auto_changed_indices_count",
        "display_rows_9",
        "givens_rows_9",
        "solution_rows_9",
        "mismatch_details",
        "conflict_details",
    ]
    out: Dict[str, Any] = {}
    for k in keep_keys:
        if k in parsed:
            out[k] = parsed.get(k)

    if isinstance(out.get("conflict_details"), list) and len(out["conflict_details"]) > 12:
        out["conflict_details"] = out["conflict_details"][:12] + [{"note": f"… trimmed ({len(parsed.get('conflict_details', []))} total)"}]

    for lk in ["confirmed_indices", "unresolved_indices", "mismatch_indices", "conflict_indices", "low_confidence_indices"]:
        v = out.get(lk)
        if isinstance(v, list) and len(v) > 80:
            out[lk] = v[:80] + ["…trimmed"]

    return out


def extract_grid_facts_snapshots_by_stage(turn_events: List[Event]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"stages": {}, "count": 0, "warnings": []}
    candidates: List[Event] = []

    for e in turn_events:
        if e.type == "GRID_FACTS_SNAPSHOT":
            candidates.append(e)

    for tag in _GRID_FACTS_TAG_CANDIDATES:
        for e in turn_events:
            if (e.type == tag) or (e.type == "POLICY_TRACE" and e.tag == tag):
                candidates.append(e)

    if not candidates:
        return {}

    def _get_stage(ev: Event) -> str:
        st = ev.get("stage")
        if isinstance(st, str) and st.strip():
            return st.strip()
        if ev.tick_id == 1:
            return "PRE_TICK1"
        return "UNKNOWN"

    tick1_idxs = [i for i, ev in enumerate(candidates) if ev.tick_id == 1]
    if len(tick1_idxs) >= 2:
        unknown_tick1 = [
            i for i in tick1_idxs
            if not (isinstance(candidates[i].get("stage"), str) and candidates[i].get("stage").strip())
        ]
        if len(unknown_tick1) >= 2:
            first_i = unknown_tick1[0]
            last_i = unknown_tick1[-1]
            stage_override = {first_i: "PRE_TICK1", last_i: "POST_TICK1"}
        else:
            stage_override = {}
    else:
        stage_override = {}

    for idx, ev in enumerate(candidates):
        stage = stage_override.get(idx) or _get_stage(ev)

        parsed_obj: Optional[Dict[str, Any]] = None
        raw_preview = ""

        payload = ev.get("payload")
        if isinstance(payload, dict):
            parsed_obj = payload
        else:
            blob = _first_nonempty_str(
                ev.get("payload_text"),
                ev.get("payload_json"),
                ev.get("snapshot_json"),
                ev.get("facts_json"),
                ev.get("grid_facts_json"),
                ev.get("payload_preview"),
            )
            if isinstance(blob, str) and blob.strip():
                raw_preview = blob[:900] + ("…" if len(blob) > 900 else "")
                maybe = _try_json_loads(blob)
                if isinstance(maybe, dict):
                    parsed_obj = maybe
            elif isinstance(payload, str) and payload.strip():
                raw_preview = payload[:900] + ("…" if len(payload) > 900 else "")
                maybe = _try_json_loads(payload)
                if isinstance(maybe, dict):
                    parsed_obj = maybe

        curated = _curate_grid_context(parsed_obj) if isinstance(parsed_obj, dict) else None

        out["stages"][stage] = {
            "type_or_tag": ev.type if ev.type != "POLICY_TRACE" else (ev.tag or ev.type),
            "turn_id": ev.get("turn_id"),
            "tick_id": ev.get("tick_id"),
            "policy_req_seq": ev.get("policy_req_seq"),
            "model_call_id": ev.get("model_call_id"),
            "correlation_id": ev.get("correlation_id"),
            "toolplan_id": ev.get("toolplan_id"),
            "seq": ev.get("seq"),
            "ts_iso": ev.get("ts_iso"),
            "stage": stage,
            "parsed": parsed_obj,
            "curated": curated,
            "raw_preview": raw_preview,
        }

    out["count"] = len(out["stages"])

    if "PRE_TICK1" in out["stages"] and not any(k in out["stages"] for k in ("POST_TICK1", "POST_DECISION", "POST_APPLY")):
        out["warnings"].append(
            "grid_facts_post_stage_missing (only PRE_TICK1 was found; emit POST_TICK1/POST_DECISION/POST_APPLY in app)"
        )

    return out


# -------------------------
# Solving truth: extraction
# -------------------------

def _extract_step_json_from_event(ev: Event) -> Optional[Dict[str, Any]]:
    raw = _first_nonempty_str(
        ev.get("step_json"),
        ev.get("payload_text"),
        ev.get("payload_json"),
        ev.get("payload_preview"),
    )
    if not isinstance(raw, str) or not raw.strip():
        return None

    parsed = _try_json_loads(raw)
    if not isinstance(parsed, dict):
        return None

    if parsed.get("schema_version") == "solve_step_v2":
        return {"envelope": {"ok": True, "status": "ok", "step": parsed}, "step": parsed}

    step = parsed.get("step") if isinstance(parsed.get("step"), dict) else None
    if isinstance(step, dict) and step.get("schema_version") == "solve_step_v2":
        return {"envelope": parsed, "step": step}

    return {"envelope": parsed, "step": step if isinstance(step, dict) else None}


def _extract_solving_step_packet_ready_payload(ev: Event) -> Optional[Dict[str, Any]]:
    raw = _first_nonempty_str(
        ev.get("packet_payload_json"),
        ev.get("packet_bundle_json"),
        ev.get("payload_json"),
        ev.get("payload_text"),
        ev.get("payload_preview"),
    )
    if not isinstance(raw, str) or not raw.strip():
        return None
    parsed = _try_json_loads(raw)
    return parsed if isinstance(parsed, dict) else None


def extract_solving_step_packet_ready(turn_events: List[Event]) -> Dict[str, Any]:
    ev = _pick_last(turn_events, "SOLVING_STEP_PACKET_READY")
    if not ev:
        return {}

    payload = _extract_solving_step_packet_ready_payload(ev)
    if not isinstance(payload, dict):
        return {
            "meta": {
                "type": ev.type,
                "turn_id": ev.get("turn_id"),
                "tick_id": ev.get("tick_id"),
                "seq": ev.get("seq"),
                "ts_iso": ev.get("ts_iso"),
                "phase": ev.get("phase"),
                "grid_hash12": ev.get("grid_hash12"),
                "step_id": ev.get("step_id"),
                "packet_payload_sha12": ev.get("packet_payload_sha12"),
                "packet_payload_len": ev.get("packet_payload_len"),
            },
            "packet_payload": None,
            "packet_step": None,
            "evidence": None,
            "technique_info": None,
            "narrative_atoms_v1": None,
            "warning": "solving_step_packet_ready_present_but_payload_not_parsed",
        }

    packet_step = payload.get("step") if isinstance(payload.get("step"), dict) else None
    evidence = payload.get("evidence") if isinstance(payload.get("evidence"), dict) else None
    technique_info = packet_step.get("technique_info") if isinstance(packet_step, dict) and isinstance(packet_step.get("technique_info"), dict) else None
    nav = evidence.get("narrative_atoms_v1") if isinstance(evidence, dict) and isinstance(evidence.get("narrative_atoms_v1"), dict) else None
    truth_v2 = evidence.get("narrative_truth_v2") if isinstance(evidence, dict) and isinstance(evidence.get("narrative_truth_v2"), dict) else None

    return {
        "meta": {
            "type": ev.type,
            "turn_id": ev.get("turn_id"),
            "tick_id": ev.get("tick_id"),
            "seq": ev.get("seq"),
            "ts_iso": ev.get("ts_iso"),
            "phase": ev.get("phase"),
            "grid_hash12": ev.get("grid_hash12"),
            "step_id": ev.get("step_id"),
            "packet_payload_sha12": ev.get("packet_payload_sha12"),
            "packet_payload_len": ev.get("packet_payload_len"),
            "technique_info_present": bool(technique_info),
            "narrative_atoms_present": bool(nav),
            "narrative_truth_v2_present": bool(truth_v2),
        },
        "packet_payload": payload,
        "packet_step": packet_step,
        "evidence": evidence,
        "technique_info": technique_info,
        "narrative_atoms_v1": nav,
        "narrative_truth_v2": truth_v2,
    }


def _curate_technique_info(step_technique: Dict[str, Any], packet_technique_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    src = packet_technique_info if isinstance(packet_technique_info, dict) and packet_technique_info else step_technique
    source = "packet.step.technique_info" if src is packet_technique_info else "solve_step.technique"

    return {
        "source_of_truth": source,
        "id": src.get("technique_id") or src.get("id") or src.get("app_name") or src.get("appName"),
        "name": src.get("technique_name") or src.get("name"),
        "app_name": src.get("app_name") or src.get("appName"),
        "real_name": src.get("real_name") or src.get("realName"),
        "family": src.get("family"),
        "family_description": src.get("family_description") or src.get("familyDescription"),
        "difficulty_level": src.get("difficulty_level") or src.get("difficultyLevel"),
        "is_base": src.get("is_base") if "is_base" in src else src.get("isBase"),
        "priority_rank": src.get("priority_rank") or src.get("priorityRank"),
        "difficulty_weight": src.get("difficulty_weight") or src.get("difficultyWeight"),
        "definition": src.get("definition") if isinstance(src.get("definition"), dict) else None,
        "comments": src.get("comments") if isinstance(src.get("comments"), dict) else None,
    }


def _extract_selected_placement(step: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    proof = _safe_get_dict(step.get("proof"))
    target = _safe_get_dict(step.get("target"))
    target_cell = _safe_get_dict(target.get("cell"))

    placements = proof.get("placements") if isinstance(proof.get("placements"), list) else []
    if placements:
        p0 = placements[0] if isinstance(placements[0], dict) else {}
        ci = _as_int(p0.get("cellIndex"), _as_int(target_cell.get("cellIndex"), -1))
        if ci in range(0, 81):
            return {
                "r": _as_int(p0.get("r"), (ci // 9) + 1),
                "c": _as_int(p0.get("c"), (ci % 9) + 1),
                "cellIndex": ci,
                "digit": _as_int(p0.get("digit"), _as_int(target.get("digit"), -1)),
                "dimension": p0.get("dimension"),
                "source_shape": p0.get("source_shape"),
            }

    ci = _as_int(target_cell.get("cellIndex"), -1)
    d = _as_int(target.get("digit"), -1)
    if ci in range(0, 81) and d in range(1, 10):
        return {
            "r": _as_int(target_cell.get("r"), (ci // 9) + 1),
            "c": _as_int(target_cell.get("c"), (ci % 9) + 1),
            "cellIndex": ci,
            "digit": d,
            "dimension": None,
            "source_shape": None,
        }

    return None


def _extract_parallel_placements(step: Dict[str, Any]) -> List[Dict[str, Any]]:
    proof = _safe_get_dict(step.get("proof"))
    all_pos = proof.get("all_possible_placements")
    all_pos_list = all_pos if isinstance(all_pos, list) else []
    parallel: List[Dict[str, Any]] = []
    for it in all_pos_list[:12]:
        if not isinstance(it, dict):
            continue
        ci = _as_int(it.get("cellIndex"), -1)
        if ci not in range(0, 81):
            continue
        parallel.append({
            "r": _as_int(it.get("r"), (ci // 9) + 1),
            "c": _as_int(it.get("c"), (ci % 9) + 1),
            "cellIndex": ci,
            "digit": _as_int(it.get("digit"), -1),
            "dimension": it.get("dimension"),
            "source_shape": it.get("source_shape"),
        })
    return parallel


def _extract_default_candidates_summary(step: Dict[str, Any]) -> Dict[str, Any]:
    pre = _safe_get_dict(step.get("pre"))
    pre_candidates = _safe_get_dict(_safe_get_dict(_safe_get_dict(pre.get("candidates")).get("all_cells")).get("cell_candidates_mask"))
    pre_digits = _safe_get_dict(pre.get("digits"))
    candidate_cells_by_house = _safe_get_dict(pre_digits.get("candidate_cells_by_house"))

    grids = _safe_get_dict(step.get("grids"))
    grid81_before = grids.get("grid81_before") if isinstance(grids.get("grid81_before"), str) else None

    empty_cells = None
    if isinstance(grid81_before, str) and len(grid81_before) == 81:
        empty_cells = sum(1 for ch in grid81_before if ch == ".")
    if empty_cells is None:
        empty_cells = sum(1 for _, v in pre_candidates.items() if isinstance(v, int) and v != 0)

    hist: Dict[str, int] = {}
    low: List[Dict[str, Any]] = []
    for k, v in pre_candidates.items():
        if not isinstance(k, str):
            continue
        ci = _as_int(k, -1)
        if ci not in range(0, 81):
            continue
        mask = _as_int(v, 0)
        if mask == 0:
            continue
        bc = _bitcount(mask)
        hist[str(bc)] = hist.get(str(bc), 0) + 1
        if bc <= 2:
            low.append({"cell": _cell_ref(ci), "cellIndex": ci, "cands": _digits_from_mask(mask)})

    low.sort(key=lambda x: (len(x.get("cands") or []), x.get("cellIndex", 999)))
    low = low[:12]

    low_house_candidates: List[Dict[str, Any]] = []
    try:
        for d_str, dpack in candidate_cells_by_house.items():
            if not isinstance(dpack, dict):
                continue
            d = _as_int(dpack.get("digit"), _as_int(d_str, -1))
            ccbh = dpack.get("candidate_cells_by_house")
            if not isinstance(ccbh, dict):
                continue

            for house_type in ("row", "col", "box"):
                hmap = ccbh.get(house_type)
                if not isinstance(hmap, dict):
                    continue
                for idx_str, cells in hmap.items():
                    if not isinstance(cells, list):
                        continue
                    if 1 <= len(cells) <= 2:
                        idx1 = _as_int(idx_str, -1)
                        cell_refs = [f"{_cell_ref(ci)} ({ci})" for ci in cells if isinstance(ci, int)]
                        low_house_candidates.append({
                            "digit": d,
                            "house": f"{house_type} {idx1}",
                            "cells": cell_refs,
                            "count": len(cell_refs),
                        })
    except Exception:
        low_house_candidates = []

    low_house_candidates.sort(
        key=lambda x: (_as_int(x.get("count"), 99), _as_int(x.get("digit"), 99), str(x.get("house") or ""))
    )
    low_house_candidates = low_house_candidates[:12]

    return {
        "empty_cells": empty_cells,
        "candidate_count_distribution": {k: hist[k] for k in sorted(hist.keys(), key=lambda x: _as_int(x, 0))},
        "low_candidate_cells_(<=2)": low,
        "low_candidate_houses_(<=2_cells_for_digit)": low_house_candidates,
    }


def _extract_digits_status(step: Dict[str, Any]) -> Dict[str, Any]:
    pre = _safe_get_dict(step.get("pre"))
    pre_digits = _safe_get_dict(pre.get("digits"))
    digit_status = _safe_get_dict(pre_digits.get("status"))

    digits_status_full: Dict[str, Any] = {}
    for d in range(1, 10):
        ds = digit_status.get(str(d))
        if not isinstance(ds, dict):
            continue
        digits_status_full[str(d)] = {
            "digit": _as_int(ds.get("digit"), d),
            "status": ds.get("status"),
            "count": _as_int(ds.get("count"), 0),
            "solved_cells": ds.get("solved_cells") if isinstance(ds.get("solved_cells"), list) else [],
            "solved_by_house": ds.get("solved_by_house") if isinstance(ds.get("solved_by_house"), dict) else {},
        }
    return digits_status_full


def _choose_lead_application(applications: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    first: Optional[Dict[str, Any]] = None
    for app in applications:
        if not isinstance(app, dict):
            continue
        if first is None:
            first = app
        narrative = _safe_get_dict(app.get("narrative"))
        role = str(narrative.get("role") or "")
        if role.lower() == "trigger":
            return app
    return first


def _extract_canonical_applications(step: Dict[str, Any]) -> List[Dict[str, Any]]:
    proof = _safe_get_dict(step.get("proof"))
    apps = proof.get("applications")
    return [a for a in apps if isinstance(a, dict)] if isinstance(apps, list) else []


def _curate_lead_application(step: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    apps = _extract_canonical_applications(step)
    lead = _choose_lead_application(apps)
    if not isinstance(lead, dict):
        return None

    identity = _safe_get_dict(lead.get("identity"))
    pattern = _safe_get_dict(lead.get("pattern"))
    pattern_cells = _safe_get_dict(pattern.get("cells"))
    effects = _safe_get_dict(lead.get("effects"))
    support = _safe_get_dict(lead.get("support"))
    narrative = _safe_get_dict(lead.get("narrative"))

    placements = []
    for p in _safe_get_list(effects.get("placements")):
        if not isinstance(p, dict):
            continue
        cell = _parse_cell_ref_like(p.get("cell"))
        if not cell:
            continue
        placements.append({
            "cell": _cell_to_verbose(cell),
            "digit": _as_int(p.get("digit"), -1),
            "source": p.get("source"),
        })

    candidate_elims = []
    for e in _safe_get_list(effects.get("candidate_eliminations")):
        if not isinstance(e, dict):
            continue
        cell = _parse_cell_ref_like(e.get("cell"))
        if not cell:
            continue
        candidate_elims.append({
            "cell": _cell_to_verbose(cell),
            "digit": _as_int(e.get("digit"), -1),
            "source": e.get("source"),
        })

    candidate_restrictions = []
    for r in _safe_get_list(effects.get("candidate_restrictions")):
        if not isinstance(r, dict):
            continue
        cell = _parse_cell_ref_like(r.get("cell"))
        if not cell:
            continue
        candidate_restrictions.append({
            "cell": _cell_to_verbose(cell),
            "allowed_digits": r.get("allowed_digits") if isinstance(r.get("allowed_digits"), list) else [],
            "removed_digits": r.get("removed_digits") if isinstance(r.get("removed_digits"), list) else [],
            "source": r.get("source"),
        })

    explanation_links = _safe_get_list(support.get("explanation_links"))

    return {
        "identity": {
            "application_id": identity.get("application_id"),
            "technique_id": identity.get("technique_id"),
            "family": identity.get("family"),
            "kind": identity.get("kind"),
            "semantic_completeness": identity.get("semantic_completeness"),
        },
        "pattern": {
            "pattern_type": pattern.get("pattern_type"),
            "pattern_subtype": pattern.get("pattern_subtype"),
            "digits": pattern.get("digits") if isinstance(pattern.get("digits"), list) else [],
            "houses": _house_list_compact(pattern.get("houses")),
            "units_scanned": _house_list_compact(pattern.get("units_scanned")),
            "focus_cells": _cells_to_verbose(pattern_cells.get("focus_cells")),
            "pattern_cells": _cells_to_verbose(pattern_cells.get("pattern_cells")),
            "peer_cells": _cells_to_verbose(pattern_cells.get("peer_cells")),
            "target_cells": _cells_to_verbose(pattern_cells.get("target_cells")),
            "witness_cells": _cells_to_verbose(pattern_cells.get("witness_cells")),
            "anchors": _cells_to_verbose(pattern_cells.get("anchors")),
            "roles": pattern.get("roles") if isinstance(pattern.get("roles"), dict) else {},
            "cover_sets": pattern.get("cover_sets") if isinstance(pattern.get("cover_sets"), list) else [],
            "constraint_explanation": pattern.get("constraint_explanation") if isinstance(pattern.get("constraint_explanation"), list) else [],
        },
        "support": {
            "focus_cells": _cells_to_verbose(support.get("focus_cells")),
            "peer_cells": _cells_to_verbose(support.get("peer_cells")),
            "witness_cells": _cells_to_verbose(support.get("witness_cells")),
            "explanation_links": explanation_links,
        },
        "effects": {
            "placements": placements,
            "candidate_eliminations": candidate_elims,
            "candidate_restrictions": candidate_restrictions,
            "cell_value_forces": [
                {
                    "cell": _cell_to_verbose(_parse_cell_ref_like(p.get("cell"))),
                    "digit": _as_int(p.get("digit"), -1),
                    "source": p.get("source"),
                }
                for p in _safe_get_list(effects.get("cell_value_forces"))
                if isinstance(p, dict) and _parse_cell_ref_like(p.get("cell"))
            ],
        },
        "narrative": {
            "archetype": narrative.get("archetype"),
            "role": narrative.get("role"),
            "summary_fact": narrative.get("summary_fact"),
            "trigger_facts": narrative.get("trigger_facts") if isinstance(narrative.get("trigger_facts"), list) else [],
            "confrontation_facts": narrative.get("confrontation_facts") if isinstance(narrative.get("confrontation_facts"), list) else [],
            "resolution_facts": narrative.get("resolution_facts") if isinstance(narrative.get("resolution_facts"), list) else [],
        },
    }


def _derive_hidden_single_truth_from_atoms(nav: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    atoms = nav.get("atoms") if isinstance(nav.get("atoms"), list) else []
    if not atoms:
        return None

    archetype = str(nav.get("archetype") or "").upper()
    if archetype != "HIDDEN_SINGLES":
        return None

    a0 = atoms[0] if isinstance(atoms[0], dict) else {}
    focus = _safe_get_dict(a0.get("focus"))
    target_cell = _parse_cell_ref_like(focus.get("target_cell"))
    primary_house = _parse_house_like(focus.get("primary_house"))
    digit = _as_int(focus.get("target_digit"), -1)

    if not target_cell or digit not in range(1, 10):
        return None

    peer_witness_pairs: List[Dict[str, Any]] = []
    witness_cells: List[str] = []
    peer_cells: List[str] = []
    remaining_cell = _cell_to_verbose(target_cell)

    for atom in atoms:
        if not isinstance(atom, dict):
            continue
        if str(atom.get("beat_kind") or "") != "WITNESS_ELIMINATION":
            continue
        claim = _safe_get_dict(atom.get("claim"))
        if str(claim.get("code") or "") != "CELL_CANNOT_BE_DIGIT":
            continue

        witnesses = atom.get("witnesses") if isinstance(atom.get("witnesses"), list) else []
        w0 = witnesses[0] if (witnesses and isinstance(witnesses[0], dict)) else {}
        because = _safe_get_dict(w0.get("because"))

        peer = _parse_cell_ref_like(because.get("explains_peer"))
        witness = _parse_cell_ref_like(because.get("witness_cell"))
        relation = because.get("relation")

        peer_s = _cell_to_verbose(peer)
        witness_s = _cell_to_verbose(witness)
        if peer_s and witness_s:
            peer_cells.append(peer_s)
            witness_cells.append(witness_s)
            peer_witness_pairs.append({
                "peer_cell": peer_s,
                "witness_cell": witness_s,
                "relation": relation,
                "digit": digit,
            })

    peer_cells = sorted(set(peer_cells))
    witness_cells = sorted(set(witness_cells))

    return {
        "source_of_truth": "packet.evidence.narrative_atoms_v1",
        "kind": "HIDDEN_SINGLE",
        "digit": digit,
        "focus_cell": remaining_cell,
        "primary_house": primary_house,
        "elimination_kind": "HOUSE_CANDIDATE_CELLS_FOR_DIGIT",
        "house_claim": {
            "digit": digit,
            "house": _pick_house_name(primary_house or {}),
            "remaining_candidate_cells": [remaining_cell] if remaining_cell else [],
            "claimed_candidate_cells": peer_cells,
            "default_candidate_cells": peer_cells + ([remaining_cell] if remaining_cell else []),
        },
        "support": {
            "peer_cells": peer_cells,
            "witness_cells": witness_cells,
            "peer_witness_pairs": peer_witness_pairs,
        },
    }


def _derive_naked_single_truth_from_atoms(nav: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    atoms = nav.get("atoms") if isinstance(nav.get("atoms"), list) else []
    if not atoms:
        return None

    archetype = str(nav.get("archetype") or "").upper()
    if archetype != "NAKED_SINGLES":
        return None

    a0 = atoms[0] if isinstance(atoms[0], dict) else {}
    focus = _safe_get_dict(a0.get("focus"))
    target_cell = _parse_cell_ref_like(focus.get("target_cell"))
    digit = _as_int(focus.get("target_digit"), -1)
    if not target_cell or digit not in range(1, 10):
        return None

    focus_cell = _cell_to_verbose(target_cell)
    witness_by_digit: List[Dict[str, Any]] = []
    witness_cells = set()
    eliminated_digits = set()

    for a in atoms:
        if not isinstance(a, dict):
            continue
        if str(a.get("beat_kind") or "") != "WITNESS_ELIMINATION":
            continue
        claim = _safe_get_dict(a.get("claim"))
        if str(claim.get("code") or "") != "CELL_CANNOT_BE_DIGIT":
            continue

        args = _safe_get_dict(claim.get("args"))
        d = _as_int(args.get("digit"), -1)
        if d not in range(1, 10):
            continue

        ws = a.get("witnesses") if isinstance(a.get("witnesses"), list) else []
        w0 = ws[0] if (ws and isinstance(ws[0], dict)) else {}
        because = _safe_get_dict(w0.get("because"))
        wcell = _parse_cell_ref_like(because.get("witness_cell"))
        wref = _cell_to_verbose(wcell)

        eliminated_digits.add(d)
        if wref:
            witness_cells.add(wref)
        witness_by_digit.append({"digit": d, "witness_cell": wref})

    witness_by_digit.sort(key=lambda x: _as_int(x.get("digit"), 99))

    lock_atom = None
    for a in atoms:
        if isinstance(a, dict) and str(a.get("beat_kind") or "") == "LOCK_IN":
            lock_atom = a
            break

    lock_eliminated_digits = []
    if isinstance(lock_atom, dict):
        witnesses = lock_atom.get("witnesses") if isinstance(lock_atom.get("witnesses"), list) else []
        w0 = witnesses[0] if (witnesses and isinstance(witnesses[0], dict)) else {}
        because = _safe_get_dict(w0.get("because"))
        lock_eliminated_digits = [d for d in because.get("eliminated_digits", []) if isinstance(d, int)]

    if lock_eliminated_digits:
        eliminated_digits = set(lock_eliminated_digits)

    return {
        "source_of_truth": "packet.evidence.narrative_atoms_v1",
        "kind": "NAKED_SINGLE",
        "digit": digit,
        "focus_cell": focus_cell,
        "primary_house": {"type": "cell", "cell": target_cell},
        "elimination_kind": "CELL_CANDIDATE_DIGITS",
        "cell_outcome": {
            "cell": focus_cell,
            "remaining_candidate_digits": [digit],
            "claimed_candidate_digits": sorted(eliminated_digits),
        },
        "support": {
            "witness_cells": sorted(witness_cells),
            "witness_by_digit": witness_by_digit,
            "eliminated_digits": sorted(eliminated_digits),
        },
    }


def _derive_intersection_truth_from_atoms(nav: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    atoms = nav.get("atoms") if isinstance(nav.get("atoms"), list) else []
    if not atoms or str(nav.get("archetype") or "").upper() != "INTERSECTIONS":
        return None

    lock_atom = None
    sweep_atom = None
    for a in atoms:
        if not isinstance(a, dict):
            continue
        claim = _safe_get_dict(a.get("claim"))
        code = str(claim.get("code") or "")
        if code == "DIGIT_LOCKED_TO_LINE_IN_BOX":
            lock_atom = a
        elif code == "INTERSECTION_SWEEP":
            sweep_atom = a

    if not isinstance(lock_atom, dict):
        return None

    claim = _safe_get_dict(lock_atom.get("claim"))
    args = _safe_get_dict(claim.get("args"))
    digit = _as_int(args.get("digit"), -1)
    source_house = _parse_house_like(args.get("source_house"))
    target_house = _parse_house_like(args.get("target_house"))
    constrained_cells = [_cell_ref_verbose(ci) for ci in args.get("constrained_cells", []) if isinstance(ci, int)]

    sweep_cells: List[str] = []
    if isinstance(sweep_atom, dict):
        sargs = _safe_get_dict(_safe_get_dict(sweep_atom.get("claim")).get("args"))
        sweep_cells = [_cell_ref_verbose(ci) for ci in sargs.get("sweep_cells", []) if isinstance(ci, int)]

    return {
        "source_of_truth": "packet.evidence.narrative_atoms_v1",
        "kind": "INTERSECTION",
        "digit": digit,
        "source_house": source_house,
        "target_house": target_house,
        "interaction_kind": args.get("interaction_kind"),
        "constrained_cells": constrained_cells,
        "sweep_cells": sweep_cells,
    }


def _derive_subset_truth_from_atoms(
    nav: Dict[str, Any],
    solve_step_pack: Optional[Dict[str, Any]] = None,
    truth_v2: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    atoms = nav.get("atoms") if isinstance(nav.get("atoms"), list) else []
    if not atoms or str(nav.get("archetype") or "").upper() != "SUBSETS":
        return None

    identify = None
    action = None
    lock_atom = None
    commit_atom = None

    for a in atoms:
        if not isinstance(a, dict):
            continue

        code = str(_safe_get_dict(a.get("claim")).get("code") or "")
        beat = str(a.get("beat_kind") or "")

        if code == "SUBSET_LOCKS_DIGITS" and identify is None:
            identify = a
        elif code in {"SUBSET_SWEEP", "HIDDEN_SUBSET_RESTRICT"} and action is None:
            action = a

        if beat == "LOCK_IN" and lock_atom is None:
            lock_atom = a
        elif beat == "COMMIT" and commit_atom is None:
            commit_atom = a

    if not isinstance(identify, dict):
        return None

    iargs = _safe_get_dict(_safe_get_dict(identify.get("claim")).get("args"))
    house = _parse_house_like(iargs.get("house"))
    subset_cells_verbose = [_cell_ref_verbose(ci) for ci in iargs.get("subset_cells", []) if isinstance(ci, int)]
    locked_digits = [d for d in iargs.get("locked_digits", []) if isinstance(d, int)]
    subset_mode = iargs.get("subset_mode")
    subset_subtype = iargs.get("subset_subtype")

    action_summary = None
    sweep_cells_verbose: List[str] = []
    support_cells_verbose: List[str] = []
    removed_digits: List[int] = []

    if isinstance(action, dict):
        aclaim = _safe_get_dict(action.get("claim"))
        aargs = _safe_get_dict(aclaim.get("args"))
        sweep_cells_verbose = [_cell_ref_verbose(ci) for ci in aargs.get("sweep_cells", []) if isinstance(ci, int)]
        support_cells_verbose = [_cell_ref_verbose(ci) for ci in aargs.get("support_cells", []) if isinstance(ci, int)]
        removed_digits = [d for d in aargs.get("removed_digits", []) if isinstance(d, int)]

        action_summary = {
            "claim_code": aclaim.get("code"),
            "sweep_cells": sweep_cells_verbose,
            "support_cells": support_cells_verbose,
            "removed_digits": removed_digits,
        }

    # Baseline generic subset witness packet
    base_packet: Dict[str, Any] = {
        "source_of_truth": "packet.evidence.narrative_atoms_v1",
        "kind": "SUBSET",
        "house": house,
        "subset_mode": subset_mode,
        "subset_subtype": subset_subtype,
        "subset_cells": subset_cells_verbose,
        "locked_digits": locked_digits,
        "action": action_summary,
    }

    # If truth_v2 is not available, fall back to the generic packet
    if not isinstance(truth_v2, dict) or not truth_v2:
        return base_packet

    # Upgrade the legacy view into elimination-shaped truth
    focus = _safe_get_dict(truth_v2.get("focus"))
    focus_cell = _parse_cell_ref_like(focus.get("focus_cell"))
    focus_cell_verbose = _cell_to_verbose(focus_cell)
    focus_digit = _as_int(focus.get("digit"), -1)
    resolution_kind = str(truth_v2.get("resolution_kind") or "")
    primary_house = _safe_get_dict(truth_v2.get("primary_house"))

    proof_payload = _safe_get_dict(truth_v2.get("proof_payload"))
    support = _safe_get_dict(proof_payload.get("support"))

    out: Dict[str, Any] = {
        "source_of_truth": "packet.evidence.narrative_atoms_v1",
        "kind": "SUBSET",
        "subset_mode": subset_mode,
        "subset_subtype": subset_subtype,
        "house": house,
        "subset_cells": subset_cells_verbose,
        "locked_digits": locked_digits,
        "focus_cell": focus_cell_verbose,
        "primary_house": primary_house if primary_house else None,
        "elimination_kind": resolution_kind or None,
    }

    if resolution_kind == "CELL_CANDIDATE_DIGITS":
        cell_outcome = _safe_get_dict(proof_payload.get("cell_outcome"))

        remaining_candidate_digits = [
            d for d in _safe_get_list(cell_outcome.get("remaining_candidate_digits"))
            if isinstance(d, int)
        ]
        claimed_candidate_digits = [
            d for d in _safe_get_list(cell_outcome.get("claimed_candidate_digits"))
            if isinstance(d, int)
        ]

        witness_cells_verbose: List[str] = []
        for raw in _safe_get_list(support.get("witness_cells")):
            s = _cell_to_verbose(_parse_cell_ref_like(raw))
            if s:
                witness_cells_verbose.append(s)

        witness_by_digit: List[Dict[str, Any]] = []
        for row in _safe_get_list(support.get("witness_by_digit")):
            if not isinstance(row, dict):
                continue
            d = _as_int(row.get("digit"), -1)
            if d not in range(1, 10):
                continue

            w = _safe_get_dict(row.get("witness"))
            wk = str(w.get("kind") or "")

            if wk == "single_cell":
                wcell = _cell_to_verbose(_parse_cell_ref_like(w.get("cell")))
                witness_by_digit.append({
                    "digit": d,
                    "witness_cell": wcell,
                })
            elif wk == "subset_group":
                witness_by_digit.append({
                    "digit": d,
                    "witness": {
                        "kind": "subset_group",
                        "subset_kind": w.get("subset_kind"),
                        "digits": [x for x in _safe_get_list(w.get("digits")) if isinstance(x, int)],
                        "cells": _cells_to_verbose(w.get("cells")),
                        "house": _parse_house_like(w.get("house")),
                    }
                })
            else:
                witness_by_digit.append({
                    "digit": d,
                    "witness": {"kind": wk or "unknown"},
                })

        out["cell_outcome"] = {
            "cell": focus_cell_verbose,
            "remaining_candidate_digits": remaining_candidate_digits,
            "claimed_candidate_digits": claimed_candidate_digits,
        }
        out["support"] = {
            "witness_cells": sorted(set(witness_cells_verbose)),
            "witness_by_digit": witness_by_digit,
            "eliminated_digits": [
                d for d in _safe_get_list(support.get("eliminated_digits"))
                if isinstance(d, int)
            ],
        }
        return out

    if resolution_kind == "HOUSE_CANDIDATE_CELLS_FOR_DIGIT":
        house_claim = _safe_get_dict(proof_payload.get("house_claim"))

        witness_cells_verbose: List[str] = []
        for raw in _safe_get_list(support.get("witness_cells")):
            s = _cell_to_verbose(_parse_cell_ref_like(raw))
            if s:
                witness_cells_verbose.append(s)

        peer_cells_verbose: List[str] = []
        for raw in _safe_get_list(support.get("peer_cells")):
            s = _cell_to_verbose(_parse_cell_ref_like(raw))
            if s:
                peer_cells_verbose.append(s)

        peer_witness_pairs: List[Dict[str, Any]] = []
        for row in _safe_get_list(support.get("witness_by_cell")):
            if not isinstance(row, dict):
                continue

            claimed_cell = _cell_to_verbose(_parse_cell_ref_like(row.get("claimed_cell")))
            w = _safe_get_dict(row.get("witness"))
            wk = str(w.get("kind") or "")

            if wk == "single_cell":
                witness_cell = _cell_to_verbose(_parse_cell_ref_like(w.get("cell")))
                peer_witness_pairs.append({
                    "peer_cell": claimed_cell,
                    "witness_cell": witness_cell,
                    "relation": w.get("relation"),
                    "digit": focus_digit if focus_digit in range(1, 10) else None,
                })
            elif wk == "subset_group":
                peer_witness_pairs.append({
                    "peer_cell": claimed_cell,
                    "witness": {
                        "kind": "subset_group",
                        "subset_kind": w.get("subset_kind"),
                        "digits": [x for x in _safe_get_list(w.get("digits")) if isinstance(x, int)],
                        "cells": _cells_to_verbose(w.get("cells")),
                        "house": _parse_house_like(w.get("house")),
                    },
                    "digit": focus_digit if focus_digit in range(1, 10) else None,
                })

        out["house_claim"] = {
            "digit": _as_int(house_claim.get("digit"), focus_digit),
            "house": _pick_house_name(_safe_get_dict(house_claim.get("house"))),
            "remaining_candidate_cells": _cells_to_verbose(house_claim.get("remaining_candidate_cells")),
            "claimed_candidate_cells": _cells_to_verbose(house_claim.get("claimed_candidate_cells")),
            "default_candidate_cells": _cells_to_verbose(house_claim.get("default_candidate_cells")),
        }
        out["support"] = {
            "peer_cells": sorted(set(peer_cells_verbose)),
            "witness_cells": sorted(set(witness_cells_verbose)),
            "peer_witness_pairs": peer_witness_pairs,
        }
        return out

    return base_packet


def _derive_fish_truth_from_atoms(nav: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    atoms = nav.get("atoms") if isinstance(nav.get("atoms"), list) else []
    if not atoms or str(nav.get("archetype") or "").upper() != "FISH":
        return None

    pattern_atom = None
    sweep_atom = None
    for a in atoms:
        if not isinstance(a, dict):
            continue
        code = str(_safe_get_dict(a.get("claim")).get("code") or "")
        if code == "FISH_LOCKS_DIGIT":
            pattern_atom = a
        elif code == "FISH_SWEEP":
            sweep_atom = a

    if not isinstance(pattern_atom, dict):
        return None

    pargs = _safe_get_dict(_safe_get_dict(pattern_atom.get("claim")).get("args"))
    base = _safe_get_dict(pargs.get("base"))
    cover = _safe_get_dict(pargs.get("cover"))
    sweep_cells = []
    if isinstance(sweep_atom, dict):
        sargs = _safe_get_dict(_safe_get_dict(sweep_atom.get("claim")).get("args"))
        sweep_cells = [_cell_ref_verbose(ci) for ci in sargs.get("sweep_cells", []) if isinstance(ci, int)]

    return {
        "source_of_truth": "packet.evidence.narrative_atoms_v1",
        "kind": "FISH",
        "fish_kind": pargs.get("fish_kind"),
        "digit": _as_int(pargs.get("digit"), -1),
        "base": {"type": base.get("type"), "indices": base.get("indices") if isinstance(base.get("indices"), list) else []},
        "cover": {"type": cover.get("type"), "indices": cover.get("indices") if isinstance(cover.get("indices"), list) else []},
        "corners": [_cell_ref_verbose(ci) for ci in pargs.get("corners", []) if isinstance(ci, int)],
        "sweep_cells": sweep_cells,
    }


def _derive_wing_truth_from_atoms(nav: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    atoms = nav.get("atoms") if isinstance(nav.get("atoms"), list) else []
    if not atoms or str(nav.get("archetype") or "").upper() != "WINGS":
        return None

    either_way = None
    elim = None
    for a in atoms:
        if not isinstance(a, dict):
            continue
        code = str(_safe_get_dict(a.get("claim")).get("code") or "")
        if code == "EITHER_WAY_ELIMINATION":
            either_way = a
        elif code == "CELL_CANNOT_BE_DIGIT":
            elim = a

    if not isinstance(either_way, dict):
        return None

    args = _safe_get_dict(_safe_get_dict(either_way.get("claim")).get("args"))
    return {
        "source_of_truth": "packet.evidence.narrative_atoms_v1",
        "kind": "WING",
        "digit": _as_int(args.get("digit"), -1),
        "hinge": _cell_ref_verbose(_as_int(args.get("hinge"), -1)) if _as_int(args.get("hinge"), -1) in range(0, 81) else None,
        "pincers": [_cell_ref_verbose(ci) for ci in args.get("pincers", []) if isinstance(ci, int)],
        "target_eliminate": _cell_ref_verbose(_as_int(args.get("target_eliminate"), -1)) if _as_int(args.get("target_eliminate"), -1) in range(0, 81) else None,
        "elimination_atom_present": bool(elim),
    }


def _derive_chain_truth_from_atoms(nav: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    atoms = nav.get("atoms") if isinstance(nav.get("atoms"), list) else []
    if not atoms or str(nav.get("archetype") or "").upper() != "CHAINS":
        return None

    color = None
    contradiction = None
    for a in atoms:
        if not isinstance(a, dict):
            continue
        code = str(_safe_get_dict(a.get("claim")).get("code") or "")
        if code == "CHAIN_COLORING":
            color = a
        elif code == "CONTRADICTION_IMPLES_NOT":
            contradiction = a

    if not isinstance(color, dict):
        return None

    carg = _safe_get_dict(_safe_get_dict(color.get("claim")).get("args"))
    xarg = _safe_get_dict(_safe_get_dict(contradiction.get("claim")).get("args")) if isinstance(contradiction, dict) else {}
    elim_ci = _as_int(xarg.get("eliminate_cell", carg.get("eliminate_cell")), -1)

    return {
        "source_of_truth": "packet.evidence.narrative_atoms_v1",
        "kind": "CHAIN",
        "digit": _as_int(carg.get("digit"), -1),
        "colorA": [_cell_ref_verbose(ci) for ci in carg.get("colorA", []) if isinstance(ci, int)],
        "colorB": [_cell_ref_verbose(ci) for ci in carg.get("colorB", []) if isinstance(ci, int)],
        "eliminate_cell": _cell_ref_verbose(elim_ci) if elim_ci in range(0, 81) else None,
    }


def _derive_narrative_truth_from_atoms(
    nav: Optional[Dict[str, Any]],
    solve_step_pack: Optional[Dict[str, Any]] = None,
    truth_v2: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if not isinstance(nav, dict) or not nav:
        return None

    archetype = str(nav.get("archetype") or "").upper()

    if archetype == "HIDDEN_SINGLES":
        return _derive_hidden_single_truth_from_atoms(nav)

    if archetype == "NAKED_SINGLES":
        return _derive_naked_single_truth_from_atoms(nav)

    if archetype == "INTERSECTIONS":
        return _derive_intersection_truth_from_atoms(nav)

    if archetype == "SUBSETS":
        return _derive_subset_truth_from_atoms(
            nav=nav,
            solve_step_pack=solve_step_pack,
            truth_v2=truth_v2,
        )

    if archetype == "FISH":
        return _derive_fish_truth_from_atoms(nav)

    if archetype == "WINGS":
        return _derive_wing_truth_from_atoms(nav)

    if archetype == "CHAINS":
        return _derive_chain_truth_from_atoms(nav)

    return {
        "source_of_truth": "packet.evidence.narrative_atoms_v1",
        "kind": archetype or "UNKNOWN",
        "note": "No specialized narrative truth reader for this archetype.",
    }


def _compact_witness_by_digit(rows: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for row in _safe_get_list(rows):
        if not isinstance(row, dict):
            continue

        d = _as_int(row.get("digit"), -1)
        if d not in range(1, 10):
            continue

        # Preferred shape for advanced trigger explanations:
        # {
        #   "digit": 1,
        #   "witness_cell": {...},
        #   "via_house": {...},
        #   "relation": "SAME_ROW"   # optional
        # }
        direct_witness_cell = _parse_cell_ref_like(row.get("witness_cell"))
        direct_via_house = _parse_house_like(row.get("via_house"))
        direct_relation = row.get("relation")

        if direct_witness_cell or direct_via_house or direct_relation:
            item: Dict[str, Any] = {
                "digit": d,
                "witness_cell": _cell_to_verbose(direct_witness_cell),
                "via_house": direct_via_house,
            }
            if direct_relation is not None:
                item["relation"] = direct_relation
            out.append(item)
            continue

        # Fallback shape used in some proof payloads:
        # {
        #   "digit": 4,
        #   "witness": {
        #       "kind": "single_cell" | "subset_group" | ...
        #   }
        # }
        witness = _safe_get_dict(row.get("witness"))
        kind = str(witness.get("kind") or "")

        if kind == "single_cell":
            item = {
                "digit": d,
                "witness_cell": _cell_to_verbose(_parse_cell_ref_like(witness.get("cell"))),
                "via_house": _parse_house_like(witness.get("via_house")),
            }
            if witness.get("relation") is not None:
                item["relation"] = witness.get("relation")
            out.append(item)

        elif kind == "subset_group":
            out.append({
                "digit": d,
                "witness": {
                    "kind": "subset_group",
                    "subset_kind": witness.get("subset_kind"),
                    "digits": [x for x in _safe_get_list(witness.get("digits")) if isinstance(x, int)],
                    "cells": _cells_to_verbose(witness.get("cells")),
                    "house": _parse_house_like(witness.get("house")),
                }
            })

        else:
            # Last-resort fallback for shapes we do not yet know.
            fallback: Dict[str, Any] = {"digit": d}
            if kind:
                fallback["witness_kind"] = kind
            else:
                fallback["witness_kind"] = "unknown"

            # Keep any direct evidence we can still surface.
            fallback_witness_cell = _cell_to_verbose(_parse_cell_ref_like(row.get("cell")))
            if fallback_witness_cell:
                fallback["witness_cell"] = fallback_witness_cell

            fallback_via_house = _parse_house_like(row.get("house"))
            if fallback_via_house:
                fallback["via_house"] = fallback_via_house

            out.append(fallback)

    return out


def _compact_pattern_member_proofs(trigger_explanation: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for row in _safe_get_list(trigger_explanation.get("pattern_member_proofs")):
        if not isinstance(row, dict):
            continue

        item: Dict[str, Any] = {
            "cell": _cell_to_verbose(_parse_cell_ref_like(row.get("cell"))),
            "explanation_kind": row.get("explanation_kind"),
            "remaining_candidate_digits": [
                d for d in _safe_get_list(row.get("remaining_candidate_digits"))
                if isinstance(d, int)
            ],
            "claimed_candidate_digits": [
                d for d in _safe_get_list(row.get("claimed_candidate_digits"))
                if isinstance(d, int)
            ],
            "witness_by_digit": _compact_witness_by_digit(row.get("witness_by_digit")),
            "status": row.get("status"),
        }

        via_houses = _house_list_compact(row.get("via_houses"))
        if via_houses:
            item["via_houses"] = via_houses

        out.append(item)

    return out


def _curate_narrative_truth_story_v2(
    truth_v2: Dict[str, Any],
    nav: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    tech = _safe_get_dict(truth_v2.get("technique"))
    focus = _safe_get_dict(truth_v2.get("focus"))
    proof_payload = _safe_get_dict(truth_v2.get("proof_payload"))
    trigger_pattern = _safe_get_dict(truth_v2.get("trigger_pattern"))
    trigger_explanation = _safe_get_dict(truth_v2.get("trigger_explanation"))
    trigger_bridge = _safe_get_dict(truth_v2.get("trigger_bridge"))
    final_resolution = _safe_get_dict(truth_v2.get("final_resolution"))

    cell_outcome = _safe_get_dict(proof_payload.get("cell_outcome"))
    house_claim = _safe_get_dict(proof_payload.get("house_claim"))

    audit = _safe_get_dict(nav.get("audit")) if isinstance(nav, dict) else {}
    atom0_snapshot = _safe_get_dict(audit.get("atom0_snapshot"))
    audit_summary = _safe_get_dict(audit.get("summary"))
    validation = _safe_get_dict(nav.get("validation")) if isinstance(nav, dict) else {}

    story: Dict[str, Any] = {
        "technique": {
            "id": tech.get("technique_id"),
            "name": tech.get("technique_name"),
            "family": tech.get("family"),
            "archetype": tech.get("archetype"),
            "is_base": tech.get("is_base"),
        },
        "target": {
            "focus_cell": _cell_to_verbose(_parse_cell_ref_like(focus.get("focus_cell"))),
            "digit": _as_int(focus.get("digit"), -1),
            "final_resolution_kind": final_resolution.get("kind"),
            "primary_house": _parse_house_like(final_resolution.get("primary_house")),
        },
    }

    if tech.get("is_base") is False:
        story["trigger_story"] = {
            "pattern_identity": {
                "kind": trigger_pattern.get("kind"),
                "subset_mode": trigger_pattern.get("subset_mode"),
                "subset_subtype": trigger_pattern.get("subset_subtype"),
                "interaction_kind": trigger_pattern.get("interaction_kind"),
            },
            "pattern_structure": {
                "house": _parse_house_like(trigger_pattern.get("house")),
                "source_house": _parse_house_like(trigger_pattern.get("source_house")),
                "target_house": _parse_house_like(trigger_pattern.get("target_house")),
                "subset_cells": _cells_to_verbose(trigger_pattern.get("subset_cells")),
                "locked_cells": _cells_to_verbose(trigger_pattern.get("locked_cells")),
                "locked_digits": [
                    d for d in _safe_get_list(trigger_pattern.get("locked_digits"))
                    if isinstance(d, int)
                ],
                "sweep_cells": _cells_to_verbose(trigger_pattern.get("sweep_cells")),
            },
            "pattern_explanation": {
                "kind": trigger_explanation.get("kind"),
                "member_proofs": _compact_pattern_member_proofs(trigger_explanation),
                "source_confinement_proof": (
                    trigger_explanation.get("source_confinement_proof")
                    if isinstance(trigger_explanation.get("source_confinement_proof"), dict)
                    else None
                ),
            },
            "pattern_to_target_bridge": {
                "kind": trigger_bridge.get("kind"),
                "why_this_matters": trigger_bridge.get("why_this_matters"),
                "target_relation": (
                    trigger_bridge.get("target_relation")
                    if isinstance(trigger_bridge.get("target_relation"), dict)
                    else None
                ),
                "sweep_relation": (
                    trigger_bridge.get("sweep_relation")
                    if isinstance(trigger_bridge.get("sweep_relation"), dict)
                    else None
                ),
            },
        }

    if cell_outcome:
        story["final_resolution"] = {
            "cell_outcome": {
                "cell": _cell_to_verbose(_parse_cell_ref_like(cell_outcome.get("cell"))),
                "remaining_candidate_digits": [
                    d for d in _safe_get_list(cell_outcome.get("remaining_candidate_digits"))
                    if isinstance(d, int)
                ],
                "claimed_candidate_digits": [
                    d for d in _safe_get_list(cell_outcome.get("claimed_candidate_digits"))
                    if isinstance(d, int)
                ],
            }
        }
    elif house_claim:
        story["final_resolution"] = {
            "house_claim": {
                "digit": _as_int(house_claim.get("digit"), -1),
                "house": _parse_house_like(house_claim.get("house")),
                "remaining_candidate_cells": _cells_to_verbose(house_claim.get("remaining_candidate_cells")),
                "claimed_candidate_cells": _cells_to_verbose(house_claim.get("claimed_candidate_cells")),
            }
        }
    else:
        story["final_resolution"] = final_resolution

    story["audit"] = {
        "validation_status": validation.get("status"),
        "validation_problem_count": len(_safe_get_list(validation.get("problems"))),
        "atom0_has_trigger_pattern": atom0_snapshot.get("atom0_has_trigger_pattern"),
        "atom0_has_trigger_explanation": atom0_snapshot.get("atom0_has_trigger_explanation"),
        "atom0_has_trigger_bridge": atom0_snapshot.get("atom0_has_trigger_bridge"),
        "atom0_has_trigger_packet": atom0_snapshot.get("atom0_has_trigger_packet"),
        "atom0_has_confrontation_summary": atom0_snapshot.get("atom0_has_confrontation_summary"),
        "truth_vs_atom0_trigger_kind_match": atom0_snapshot.get("truth_vs_atom0_trigger_kind_match"),
        "truth_vs_atom0_resolution_kind_match": atom0_snapshot.get("truth_vs_atom0_resolution_kind_match"),
        "validation_summary": audit_summary,
    }

    return story



def _curate_engine_step_summary(step: Dict[str, Any], packet_technique_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    grids = _safe_get_dict(step.get("grids"))
    technique = _safe_get_dict(step.get("technique"))
    grid81_before = grids.get("grid81_before") if isinstance(grids.get("grid81_before"), str) else None

    lead = _curate_lead_application(step) or {}
    lead_identity = _safe_get_dict(lead.get("identity"))
    lead_pattern = _safe_get_dict(lead.get("pattern"))
    lead_narrative = _safe_get_dict(lead.get("narrative"))
    lead_effects = _safe_get_dict(lead.get("effects"))

    return {
        "grid81_before": grid81_before,
        "technique": _curate_technique_info(technique, packet_technique_info),
        "selected_placement": _extract_selected_placement(step),
        "parallel_forced_placements_count": len(_extract_parallel_placements(step)),
        "lead_application_summary": {
            "application_id": lead_identity.get("application_id"),
            "technique_id": lead_identity.get("technique_id"),
            "pattern_type": lead_pattern.get("pattern_type"),
            "pattern_subtype": lead_pattern.get("pattern_subtype"),
            "houses": lead_pattern.get("houses") if isinstance(lead_pattern.get("houses"), list) else [],
            "focus_cells": lead_pattern.get("focus_cells") if isinstance(lead_pattern.get("focus_cells"), list) else [],
            "target_cells": lead_pattern.get("target_cells") if isinstance(lead_pattern.get("target_cells"), list) else [],
            "summary_fact": lead_narrative.get("summary_fact"),
            "trigger_facts": lead_narrative.get("trigger_facts") if isinstance(lead_narrative.get("trigger_facts"), list) else [],
            "resolution_facts": lead_narrative.get("resolution_facts") if isinstance(lead_narrative.get("resolution_facts"), list) else [],
            "placements_count": len(lead_effects.get("placements") if isinstance(lead_effects.get("placements"), list) else []),
            "candidate_eliminations_count": len(lead_effects.get("candidate_eliminations") if isinstance(lead_effects.get("candidate_eliminations"), list) else []),
            "candidate_restrictions_count": len(lead_effects.get("candidate_restrictions") if isinstance(lead_effects.get("candidate_restrictions"), list) else []),
        },
    }


def extract_solve_step_json_ready(
    turn_events: List[Event],
    packet_technique_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ev = _pick_last(turn_events, "SOLVE_STEP_JSON_READY")
    if not ev:
        return {}

    box = _extract_step_json_from_event(ev)
    if not box:
        return {}

    step = box.get("step")
    env = box.get("envelope")

    if not isinstance(step, dict):
        return {
            "meta": {
                "type": ev.type,
                "turn_id": ev.get("turn_id"),
                "tick_id": ev.get("tick_id"),
                "seq": ev.get("seq"),
                "ts_iso": ev.get("ts_iso"),
                "phase": ev.get("phase"),
                "grid_hash12": ev.get("grid_hash12"),
                "step_id": ev.get("step_id"),
                "step_json_len": ev.get("step_json_len"),
                "step_json_sha12": ev.get("step_json_sha12"),
            },
            "raw_envelope": env,
            "step": None,
            "curated": None,
            "warning": "solve_step_present_but_not_parsed_as_solve_step_v2",
        }

    curated = _curate_engine_step_summary(step, packet_technique_info)

    return {
        "meta": {
            "type": ev.type,
            "turn_id": ev.get("turn_id"),
            "tick_id": ev.get("tick_id"),
            "seq": ev.get("seq"),
            "ts_iso": ev.get("ts_iso"),
            "phase": ev.get("phase"),
            "grid_hash12": ev.get("grid_hash12"),
            "step_id": ev.get("step_id"),
            "step_json_len": ev.get("step_json_len"),
            "step_json_sha12": ev.get("step_json_sha12"),
        },
        "raw_envelope": env,
        "step": step,
        "curated": curated,
    }


def solve_step_grid_matches_displayed(grid_facts_pack: Dict[str, Any], solve_step_pack: Dict[str, Any]) -> Optional[bool]:
    if not isinstance(grid_facts_pack, dict) or not isinstance(solve_step_pack, dict):
        return None
    stages = grid_facts_pack.get("stages")
    if not isinstance(stages, dict):
        return None
    pre = stages.get("PRE_TICK1") or {}
    curated = pre.get("curated")
    if not isinstance(curated, dict):
        return None
    displayed81 = curated.get("displayed81")
    if not isinstance(displayed81, str) or len(displayed81.strip()) != 81:
        return None

    step = solve_step_pack.get("step")
    if not isinstance(step, dict):
        return None
    grids = step.get("grids")
    if not isinstance(grids, dict):
        return None
    g81 = grids.get("grid81_before")
    if not isinstance(g81, str) or len(g81) != 81:
        return None

    return _displayed81_to_grid81(displayed81) == g81


# -------------------------
# Narrative atoms + reconstructed overlays
# -------------------------

def _reconstruct_overlay_frame_from_atom(atom: Dict[str, Any], style: str = "full") -> Dict[str, Any]:
    atom_index = _as_int(atom.get("index"), -1)
    overlay = _safe_get_dict(atom.get("overlay"))
    fid = overlay.get("frame_id")
    if not (isinstance(fid, str) and fid.strip()):
        fid = f"ov:atom:{atom_index}"

    focus = _safe_get_dict(atom.get("focus"))
    target_cell = _parse_cell_ref_like(focus.get("target_cell"))
    focus_idx = _as_int(target_cell.get("cellIndex"), -1) if isinstance(target_cell, dict) else -1

    primary_house = _parse_house_like(focus.get("primary_house"))
    ph_type = primary_house.get("type") if isinstance(primary_house, dict) else ""
    ph_idx = _as_int(primary_house.get("index1to9"), -1) if isinstance(primary_house, dict) else -1

    def _house_obj(h_type: str, idx1to9: int, role: str) -> Dict[str, Any]:
        return {"kind": "house", "role": role, "house": {"type": h_type, "index1to9": idx1to9}}

    def _cell_obj(ci: int, role: str, digit: Optional[int] = None) -> Dict[str, Any]:
        o = {"kind": "cell", "cellIndex": ci, "role": role}
        if digit is not None:
            o["digit"] = digit
        return o

    def _link_obj(frm: int, to: int, role: str, digit: Optional[int] = None) -> Dict[str, Any]:
        o = {"kind": "link", "fromCellIndex": frm, "toCellIndex": to, "role": role}
        if digit is not None:
            o["digit"] = digit
        return o

    hi: List[Dict[str, Any]] = []

    if 0 <= focus_idx <= 80:
        hi.append(_cell_obj(focus_idx, "focus"))

    if isinstance(primary_house, dict) and ph_type in {"row", "col", "box"} and 1 <= ph_idx <= 9:
        hi.append(_house_obj(ph_type, ph_idx, "primary_house"))

    beat_kind = str(atom.get("beat_kind") or "")
    archetype = str(atom.get("archetype") or "")
    claim = _safe_get_dict(atom.get("claim"))
    claim_code = str(claim.get("code") or "")
    claim_args = _safe_get_dict(claim.get("args"))
    witnesses = atom.get("witnesses") if isinstance(atom.get("witnesses"), list) else []

    if beat_kind == "WITNESS_ELIMINATION" and claim_code == "CELL_CANNOT_BE_DIGIT":
        d = _as_int(claim_args.get("digit"), -1)
        peer = _parse_cell_ref_like(claim_args.get("cell"))
        peer_idx = _as_int(peer.get("cellIndex"), -1) if isinstance(peer, dict) else -1
        if 0 <= peer_idx <= 80:
            hi.append(_cell_obj(peer_idx, "peer"))
            if 1 <= d <= 9:
                hi.append(_cell_obj(peer_idx, "eliminate_digit", d))

        w0 = witnesses[0] if (witnesses and isinstance(witnesses[0], dict)) else {}
        because = _safe_get_dict(w0.get("because"))
        wcell = _parse_cell_ref_like(because.get("witness_cell"))
        w_idx = _as_int(wcell.get("cellIndex"), -1) if isinstance(wcell, dict) else -1
        if 0 <= w_idx <= 80:
            hi.append(_cell_obj(w_idx, "witness", d if 1 <= d <= 9 else None))

        relation = str(because.get("relation") or "")
        if 0 <= peer_idx <= 80:
            pr = (peer_idx // 9) + 1
            pc = (peer_idx % 9) + 1
            if relation == "SAME_COL":
                hi.append(_house_obj("col", pc, "secondary_house"))
            elif relation == "SAME_ROW":
                hi.append(_house_obj("row", pr, "secondary_house"))
            elif relation == "SAME_BOX":
                bi = (((pr - 1) // 3) * 3 + ((pc - 1) // 3) + 1)
                hi.append(_house_obj("box", bi, "secondary_house"))

    elif claim_code == "DIGIT_LOCKED_TO_LINE_IN_BOX":
        d = _as_int(claim_args.get("digit"), -1)
        source_house = _parse_house_like(claim_args.get("source_house"))
        target_house = _parse_house_like(claim_args.get("target_house"))
        constrained = claim_args.get("constrained_cells") if isinstance(claim_args.get("constrained_cells"), list) else []

        if isinstance(source_house, dict):
            sht = str(source_house.get("type") or "")
            shi = _as_int(source_house.get("index1to9"), -1)
            if sht in {"row", "col", "box"} and 1 <= shi <= 9:
                hi.append(_house_obj(sht, shi, "primary_house"))

        if isinstance(target_house, dict):
            tht = str(target_house.get("type") or "")
            thi = _as_int(target_house.get("index1to9"), -1)
            if tht in {"row", "col", "box"} and 1 <= thi <= 9:
                hi.append(_house_obj(tht, thi, "secondary_house"))

        for ci in constrained:
            idx = _as_int(ci, -1)
            if 0 <= idx <= 80:
                hi.append(_cell_obj(idx, "peer"))
                if 1 <= d <= 9:
                    hi.append(_cell_obj(idx, "lock_digit", d))

    elif claim_code == "INTERSECTION_SWEEP":
        d = _as_int(claim_args.get("digit"), -1)
        sweep_cells = claim_args.get("sweep_cells") if isinstance(claim_args.get("sweep_cells"), list) else []
        source_house = _parse_house_like(claim_args.get("source_house"))
        target_house = _parse_house_like(claim_args.get("target_house"))

        if isinstance(source_house, dict):
            sht = str(source_house.get("type") or "")
            shi = _as_int(source_house.get("index1to9"), -1)
            if sht in {"row", "col", "box"} and 1 <= shi <= 9:
                hi.append(_house_obj(sht, shi, "primary_house"))

        if isinstance(target_house, dict):
            tht = str(target_house.get("type") or "")
            thi = _as_int(target_house.get("index1to9"), -1)
            if tht in {"row", "col", "box"} and 1 <= thi <= 9:
                hi.append(_house_obj(tht, thi, "secondary_house"))

        for ci in sweep_cells:
            idx = _as_int(ci, -1)
            if 0 <= idx <= 80:
                hi.append(_cell_obj(idx, "peer"))
                if 1 <= d <= 9:
                    hi.append(_cell_obj(idx, "eliminate_digit", d))

    elif claim_code == "SUBSET_LOCKS_DIGITS":
        subset_cells = claim_args.get("subset_cells") if isinstance(claim_args.get("subset_cells"), list) else []
        locked_digits = claim_args.get("locked_digits") if isinstance(claim_args.get("locked_digits"), list) else []
        house = _parse_house_like(claim_args.get("house"))
        if isinstance(house, dict):
            ht = str(house.get("type") or "")
            hi9 = _as_int(house.get("index1to9"), -1)
            if ht in {"row", "col", "box"} and 1 <= hi9 <= 9:
                hi.append(_house_obj(ht, hi9, "primary_house"))
        for ci in subset_cells:
            idx = _as_int(ci, -1)
            if 0 <= idx <= 80:
                hi.append(_cell_obj(idx, "peer"))
                for dd in locked_digits:
                    d = _as_int(dd, -1)
                    if 1 <= d <= 9:
                        hi.append(_cell_obj(idx, "lock_digit", d))

    elif claim_code == "SUBSET_SWEEP":
        sweep_cells = claim_args.get("sweep_cells") if isinstance(claim_args.get("sweep_cells"), list) else []
        locked_digits = claim_args.get("locked_digits") if isinstance(claim_args.get("locked_digits"), list) else []
        for ci in sweep_cells:
            idx = _as_int(ci, -1)
            if 0 <= idx <= 80:
                hi.append(_cell_obj(idx, "peer"))
                for dd in locked_digits:
                    d = _as_int(dd, -1)
                    if 1 <= d <= 9:
                        hi.append(_cell_obj(idx, "eliminate_digit", d))

    elif claim_code == "HIDDEN_SUBSET_RESTRICT":
        support_cells = claim_args.get("support_cells") if isinstance(claim_args.get("support_cells"), list) else []
        removed_digits = claim_args.get("removed_digits") if isinstance(claim_args.get("removed_digits"), list) else []
        for ci in support_cells:
            idx = _as_int(ci, -1)
            if 0 <= idx <= 80:
                hi.append(_cell_obj(idx, "peer"))
                for dd in removed_digits:
                    d = _as_int(dd, -1)
                    if 1 <= d <= 9:
                        hi.append(_cell_obj(idx, "eliminate_digit", d))

    elif claim_code == "FISH_LOCKS_DIGIT":
        d = _as_int(claim_args.get("digit"), -1)
        base = _safe_get_dict(claim_args.get("base"))
        cover = _safe_get_dict(claim_args.get("cover"))
        corners = claim_args.get("corners") if isinstance(claim_args.get("corners"), list) else []

        bt = base.get("type") if isinstance(base.get("type"), str) else ""
        bidxs = base.get("indices") if isinstance(base.get("indices"), list) else []
        for h in bidxs:
            hv = _as_int(h, -1)
            if bt and 1 <= hv <= 9:
                hi.append(_house_obj(bt, hv, "primary_house"))

        ct = cover.get("type") if isinstance(cover.get("type"), str) else ""
        cidxs = cover.get("indices") if isinstance(cover.get("indices"), list) else []
        for h in cidxs:
            hv = _as_int(h, -1)
            if ct and 1 <= hv <= 9:
                hi.append(_house_obj(ct, hv, "secondary_house"))

        corner_idxs: List[int] = []
        for ci in corners:
            idx = _as_int(ci, -1)
            if 0 <= idx <= 80:
                corner_idxs.append(idx)
                hi.append(_cell_obj(idx, "link_a", d if 1 <= d <= 9 else None))

        for i in range(0, max(0, len(corner_idxs) - 1)):
            hi.append(_link_obj(corner_idxs[i], corner_idxs[i + 1], "inference_link", d if 1 <= d <= 9 else None))

    elif claim_code == "FISH_SWEEP":
        d = _as_int(claim_args.get("digit"), -1)
        sweep = claim_args.get("sweep_cells") if isinstance(claim_args.get("sweep_cells"), list) else []
        for ci in sweep:
            idx = _as_int(ci, -1)
            if 0 <= idx <= 80:
                hi.append(_cell_obj(idx, "peer"))
                if 1 <= d <= 9:
                    hi.append(_cell_obj(idx, "eliminate_digit", d))

    elif claim_code == "EITHER_WAY_ELIMINATION":
        d = _as_int(claim_args.get("digit"), -1)
        hinge = _as_int(claim_args.get("hinge"), -1)
        pincers = claim_args.get("pincers") if isinstance(claim_args.get("pincers"), list) else []
        target = _as_int(claim_args.get("target_eliminate"), -1)

        if 0 <= hinge <= 80:
            hi.append(_cell_obj(hinge, "link_a", d if 1 <= d <= 9 else None))
        for p in pincers:
            pi = _as_int(p, -1)
            if 0 <= pi <= 80:
                hi.append(_cell_obj(pi, "link_b", d if 1 <= d <= 9 else None))
                if 0 <= hinge <= 80:
                    hi.append(_link_obj(hinge, pi, "inference_link", d if 1 <= d <= 9 else None))
        if 0 <= target <= 80:
            hi.append(_cell_obj(target, "peer"))
            if 1 <= d <= 9:
                hi.append(_cell_obj(target, "eliminate_digit", d))

    elif claim_code in ("CHAIN_COLORING", "CONTRADICTION_IMPLES_NOT"):
        d = _as_int(claim_args.get("digit"), -1)
        a = claim_args.get("colorA") if isinstance(claim_args.get("colorA"), list) else []
        b = claim_args.get("colorB") if isinstance(claim_args.get("colorB"), list) else []
        for ci in a:
            idx = _as_int(ci, -1)
            if 0 <= idx <= 80:
                hi.append(_cell_obj(idx, "link_a", d if 1 <= d <= 9 else None))
        for ci in b:
            idx = _as_int(ci, -1)
            if 0 <= idx <= 80:
                hi.append(_cell_obj(idx, "link_b", d if 1 <= d <= 9 else None))

        elim = _as_int(claim_args.get("eliminate_cell"), -1)
        if 0 <= elim <= 80:
            hi.append(_cell_obj(elim, "peer"))
            if 1 <= d <= 9:
                hi.append(_cell_obj(elim, "eliminate_digit", d))

    elif beat_kind == "LOCK_IN":
        td = _as_int(focus.get("target_digit"), -1)
        w0 = witnesses[0] if (witnesses and isinstance(witnesses[0], dict)) else {}
        because = _safe_get_dict(w0.get("because"))

        eliminated_peers = because.get("eliminated_peer_cell_indices") if isinstance(because.get("eliminated_peer_cell_indices"), list) else []
        eliminated_digits = because.get("eliminated_digits") if isinstance(because.get("eliminated_digits"), list) else []

        for ci in eliminated_peers:
            idx = _as_int(ci, -1)
            if 0 <= idx <= 80:
                hi.append(_cell_obj(idx, "peer"))
                if 1 <= td <= 9:
                    hi.append(_cell_obj(idx, "eliminate_digit", td))

        if 0 <= focus_idx <= 80:
            for dd in eliminated_digits:
                d2 = _as_int(dd, -1)
                if 1 <= d2 <= 9:
                    hi.append(_cell_obj(focus_idx, "peer"))
                    hi.append(_cell_obj(focus_idx, "eliminate_digit", d2))

    elif beat_kind == "COMMIT":
        d = _as_int(focus.get("target_digit"), -1)
        if 0 <= focus_idx <= 80:
            hi.append(_cell_obj(focus_idx, "result_place", d if 1 <= d <= 9 else None))

    hi = _pairwise_dedup_dicts(hi)

    return {
        "v": 1,
        "meta": {
            "reason": "derived_from_narrative_atoms_v1",
            "style": style,
            "frame_id": fid,
            "beat_kind": beat_kind,
            "claim_code": claim_code,
            "archetype": archetype,
        },
        "focus": {"cellIndex": focus_idx if 0 <= focus_idx <= 80 else None},
        "highlights": hi,
    }


def reconstruct_overlay_frames_from_atoms(narrative_atoms_v1: Dict[str, Any], style: str = "full") -> List[Dict[str, Any]]:
    atoms = narrative_atoms_v1.get("atoms") if isinstance(narrative_atoms_v1.get("atoms"), list) else []
    out: List[Dict[str, Any]] = []
    for a in atoms:
        if not isinstance(a, dict):
            continue
        out.append(_reconstruct_overlay_frame_from_atom(a, style=style))
    return out


def _truth_v2_focus_cell_index(truth_v2: Dict[str, Any]) -> Optional[int]:
    focus = _safe_get_dict(truth_v2.get("focus"))
    cell = _parse_cell_ref_like(focus.get("focus_cell"))
    if isinstance(cell, dict):
        ci = _as_int(cell.get("cellIndex"), -1)
        if ci in range(0, 81):
            return ci
    return None


def _truth_v2_focus_digit(truth_v2: Dict[str, Any]) -> Optional[int]:
    d = _as_int(_safe_get_dict(truth_v2.get("focus")).get("digit"), -1)
    return d if d in range(1, 10) else None


def _truth_v2_archetype(truth_v2: Dict[str, Any]) -> Optional[str]:
    tech = _safe_get_dict(truth_v2.get("technique"))
    a = tech.get("archetype")
    return str(a).upper() if isinstance(a, str) and a.strip() else None


def _truth_v2_commit_cell_and_digit(truth_v2: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    dr = _safe_get_dict(truth_v2.get("downstream_resolution"))
    placement = _safe_get_dict(dr.get("placement"))
    cell = _parse_cell_ref_like(placement.get("cell"))
    digit = _as_int(placement.get("digit"), -1)
    ci = _as_int(cell.get("cellIndex"), -1) if isinstance(cell, dict) else -1
    return (ci if ci in range(0, 81) else None, digit if digit in range(1, 10) else None)


def _truth_v2_expected_lockin_claim_code(truth_v2: Dict[str, Any]) -> Optional[str]:
    rk = str(truth_v2.get("resolution_kind") or "")
    if rk == "CELL_CANDIDATE_DIGITS":
        return "ONLY_DIGIT_LEFT_FOR_CELL"
    if rk == "HOUSE_CANDIDATE_CELLS_FOR_DIGIT":
        return "ONLY_CELL_LEFT_FOR_DIGIT_IN_HOUSE"
    return None


def _count_subset_group_witnesses_in_truth_v2(truth_v2: Dict[str, Any]) -> int:
    proof = _safe_get_dict(truth_v2.get("proof_payload"))
    support = _safe_get_dict(proof.get("support"))
    count = 0

    for row in _safe_get_list(support.get("witness_by_digit")):
        if not isinstance(row, dict):
            continue
        w = _safe_get_dict(row.get("witness"))
        if str(w.get("kind") or "") == "subset_group":
            count += 1

    for row in _safe_get_list(support.get("witness_by_cell")):
        if not isinstance(row, dict):
            continue
        w = _safe_get_dict(row.get("witness"))
        if str(w.get("kind") or "") == "subset_group":
            count += 1

    return count


def _count_subset_group_atoms(nav: Dict[str, Any]) -> int:
    atoms = nav.get("atoms") if isinstance(nav.get("atoms"), list) else []
    count = 0
    for a in atoms:
        if not isinstance(a, dict):
            continue
        ws = a.get("witnesses") if isinstance(a.get("witnesses"), list) else []
        for w in ws:
            if not isinstance(w, dict):
                continue
            if str(w.get("kind") or "") == "SUBSET_DEFINITION":
                count += 1
                break
    return count


def build_truth_atoms_consistency_report(
    truth_v2: Optional[Dict[str, Any]],
    narrative_atoms_v1: Optional[Dict[str, Any]],
    overlay_frames: Optional[List[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    if not isinstance(truth_v2, dict) or not truth_v2:
        return None
    if not isinstance(narrative_atoms_v1, dict) or not narrative_atoms_v1:
        return {
            "status": "MISSING_ATOMS",
            "checks": [],
            "summary": {
                "ok": 0,
                "warn": 0,
                "fail": 1,
            },
            "notes": ["narrative_truth_v2 present but narrative_atoms_v1 missing"],
        }

    atoms = narrative_atoms_v1.get("atoms") if isinstance(narrative_atoms_v1.get("atoms"), list) else []
    overlays = overlay_frames if isinstance(overlay_frames, list) else []

    checks: List[Dict[str, Any]] = []

    def add_check(name: str, ok: bool, expected: Any, observed: Any, severity: str = "fail", note: Optional[str] = None) -> None:
        checks.append({
            "name": name,
            "ok": bool(ok),
            "severity": severity if not ok else "ok",
            "expected": expected,
            "observed": observed,
            "note": note,
        })

    truth_arch = _truth_v2_archetype(truth_v2)
    atoms_arch = str(narrative_atoms_v1.get("archetype") or "").upper() or None
    add_check(
        name="archetype_match",
        ok=(truth_arch == atoms_arch),
        expected=truth_arch,
        observed=atoms_arch,
    )

    truth_focus_ci = _truth_v2_focus_cell_index(truth_v2)
    truth_focus_digit = _truth_v2_focus_digit(truth_v2)

    first_atom = atoms[0] if (atoms and isinstance(atoms[0], dict)) else {}
    first_focus = _safe_get_dict(first_atom.get("focus"))
    first_focus_cell = _parse_cell_ref_like(first_focus.get("target_cell"))
    first_focus_ci = _as_int(first_focus_cell.get("cellIndex"), -1) if isinstance(first_focus_cell, dict) else -1
    first_focus_digit = _as_int(first_focus.get("target_digit"), -1)

    add_check(
        name="focus_cell_match",
        ok=(truth_focus_ci == first_focus_ci if truth_focus_ci is not None and first_focus_ci in range(0, 81) else False),
        expected=truth_focus_ci,
        observed=first_focus_ci if first_focus_ci in range(0, 81) else None,
    )

    add_check(
        name="focus_digit_match",
        ok=(truth_focus_digit == first_focus_digit if truth_focus_digit is not None and first_focus_digit in range(1, 10) else False),
        expected=truth_focus_digit,
        observed=first_focus_digit if first_focus_digit in range(1, 10) else None,
    )

    expected_lockin = _truth_v2_expected_lockin_claim_code(truth_v2)
    lock_atom = None
    for a in atoms:
        if isinstance(a, dict) and str(a.get("beat_kind") or "") == "LOCK_IN":
            lock_atom = a
            break
    observed_lockin = None
    if isinstance(lock_atom, dict):
        observed_lockin = _safe_get_dict(lock_atom.get("claim")).get("code")

    add_check(
        name="lockin_claim_matches_resolution_kind",
        ok=(expected_lockin == observed_lockin if expected_lockin is not None else False),
        expected=expected_lockin,
        observed=observed_lockin,
    )

    expected_commit_ci, expected_commit_digit = _truth_v2_commit_cell_and_digit(truth_v2)
    commit_atom = None
    for a in atoms:
        if isinstance(a, dict) and str(a.get("beat_kind") or "") == "COMMIT":
            commit_atom = a
            break

    observed_commit_ci = None
    observed_commit_digit = None
    if isinstance(commit_atom, dict):
        f = _safe_get_dict(commit_atom.get("focus"))
        cell = _parse_cell_ref_like(f.get("target_cell"))
        if isinstance(cell, dict):
            ci = _as_int(cell.get("cellIndex"), -1)
            if ci in range(0, 81):
                observed_commit_ci = ci
        d = _as_int(f.get("target_digit"), -1)
        if d in range(1, 10):
            observed_commit_digit = d

    add_check(
        name="commit_cell_matches_truth",
        ok=(expected_commit_ci == observed_commit_ci if expected_commit_ci is not None else False),
        expected=expected_commit_ci,
        observed=observed_commit_ci,
    )

    add_check(
        name="commit_digit_matches_truth",
        ok=(expected_commit_digit == observed_commit_digit if expected_commit_digit is not None else False),
        expected=expected_commit_digit,
        observed=observed_commit_digit,
    )

    add_check(
        name="overlay_frame_count_matches_atom_count",
        ok=(len(overlays) == len(atoms)),
        expected=len(atoms),
        observed=len(overlays),
        severity="warn",
    )

    if truth_arch == "SUBSETS":
        expected_subset_group_count = _count_subset_group_witnesses_in_truth_v2(truth_v2)
        observed_subset_group_atoms = _count_subset_group_atoms(narrative_atoms_v1)
        add_check(
            name="subset_group_witness_presence",
            ok=(observed_subset_group_atoms >= min(1, expected_subset_group_count) if expected_subset_group_count > 0 else True),
            expected={"subset_group_witnesses_in_truth_v2": expected_subset_group_count},
            observed={"subset_definition_atoms": observed_subset_group_atoms},
            severity="warn",
            note="For SUBSETS, at least one subset-group witness atom should appear when truth_v2 contains subset_group witnesses.",
        )

    ok_n = sum(1 for c in checks if c.get("ok") is True)
    warn_n = sum(1 for c in checks if c.get("ok") is False and c.get("severity") == "warn")
    fail_n = sum(1 for c in checks if c.get("ok") is False and c.get("severity") != "warn")

    status = "PASS"
    if fail_n > 0:
        status = "FAIL"
    elif warn_n > 0:
        status = "WARN"

    return {
        "status": status,
        "summary": {
            "ok": ok_n,
            "warn": warn_n,
            "fail": fail_n,
        },
        "checks": checks,
        "notes": [
            "This report compares packet.evidence.narrative_truth_v2 against packet.evidence.narrative_atoms_v1 and derived overlay frames.",
        ],
    }


# -------------------------
# Extraction: reply_request_v1 + toolplan summary
# -------------------------

def extract_toolplan_planned_v1(turn_events: List[Event]) -> Dict[str, Any]:
    planned = [e for e in turn_events if e.type == "TOOLPLAN_PLANNED_V1"]
    if not planned:
        return {}

    planned_t1 = [e for e in planned if e.tick_id == 1]
    ev = planned_t1[-1] if planned_t1 else planned[-1]

    return {
        "turn_id": ev.get("turn_id"),
        "tick_id": ev.get("tick_id"),
        "policy_req_seq": ev.get("policy_req_seq"),
        "model_call_id": ev.get("model_call_id"),
        "correlation_id": ev.get("correlation_id"),
        "toolplan_id": ev.get("toolplan_id"),
        "decision_kind": ev.get("decision_kind"),
        "planned_tools": ev.get("planned_tools") or [],
        "planned_tools_count": ev.get("planned_tools_count"),
        "planned_control": ev.get("planned_control"),
        "rationale": ev.get("rationale"),
        "seq": ev.get("seq"),
        "ts_iso": ev.get("ts_iso"),
    }


def extract_reply_request_details(turn_events: List[Event]) -> Dict[str, Any]:
    rr = _pick_first(turn_events, "REPLY_REQUEST_V1_OUT", tick=2)
    if not rr:
        return {}

    payload_text = _first_nonempty_str(
        rr.get("rr_json"),
        rr.get("payload_text"),
        rr.get("payload_json"),
        rr.get("payload_preview"),
    )

    if not isinstance(payload_text, str) or not payload_text.strip():
        return {
            "reply_request_v1": None,
            "turn_ctx": {},
            "decision": {},
            "warning": "reply_request_payload_missing",
        }

    req = _try_json_loads(payload_text)
    if not isinstance(req, dict):
        return {
            "reply_request_v1": None,
            "turn_ctx": {},
            "decision": {},
            "warning": "reply_request_payload_not_json (likely truncated preview)",
            "payload_preview": payload_text,
        }

    turn = req.get("turn") if isinstance(req.get("turn"), dict) else {}
    decision = req.get("decision") if isinstance(req.get("decision"), dict) else {}
    return {"reply_request_v1": req, "turn_ctx": turn, "decision": decision}


def extract_toolplan_tools_from_reply_request(rr_req: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = {
        "operation_tools": [],
        "control_tools": [],
        "note": "",
        "source_path": None,
    }

    if not isinstance(rr_req, dict):
        out["note"] = "reply_request_v1 missing/unparsed; toolplan tools unavailable"
        return out

    candidates = []

    toolplan = rr_req.get("toolplan")
    if isinstance(toolplan, dict):
        candidates.append(("toolplan", toolplan))

    decision = rr_req.get("decision")
    if isinstance(decision, dict):
        candidates.append(("decision", decision))

    def _as_list(x):
        return x if isinstance(x, list) else None

    for base_name, base in candidates:
        op = _as_list(base.get("operation_tools")) or _as_list(base.get("op_tools")) or _as_list(base.get("ops_tools"))
        ctrl = _as_list(base.get("control_tools")) or _as_list(base.get("control")) or _as_list(base.get("ctrl_tools"))
        if op is not None or ctrl is not None:
            out["operation_tools"] = op or []
            out["control_tools"] = ctrl or []
            out["source_path"] = f"{base_name}.(operation_tools/control_tools)"
            return out

        tools = base.get("tools")
        if isinstance(tools, dict):
            op2 = _as_list(tools.get("operation")) or _as_list(tools.get("ops"))
            ctrl2 = _as_list(tools.get("control")) or _as_list(tools.get("ctrl"))
            if op2 is not None or ctrl2 is not None:
                out["operation_tools"] = op2 or []
                out["control_tools"] = ctrl2 or []
                out["source_path"] = f"{base_name}.tools.(operation/control)"
                return out

        tcs = base.get("tool_calls")
        if isinstance(tcs, list) and tcs:
            ops: List[str] = []
            ctrls: List[str] = []
            for tc in tcs:
                if not isinstance(tc, dict):
                    continue
                name = tc.get("name") or tc.get("tool") or tc.get("id") or ""
                kind = tc.get("kind") or tc.get("category") or ""
                if isinstance(name, str) and name:
                    if isinstance(kind, str) and kind.lower() in ("control", "ctrl"):
                        ctrls.append(name)
                    elif isinstance(kind, str) and kind.lower() in ("operation", "op", "ops"):
                        ops.append(name)
                    else:
                        ops.append(name)
            out["operation_tools"] = ops
            out["control_tools"] = ctrls
            out["source_path"] = f"{base_name}.tool_calls"
            out["note"] = "tool_calls lacked explicit kind for some entries; defaulted to operation"
            return out

    out["note"] = "toolplan not present in telemetry payloads (or only truncated previews were logged)"
    return out


# -------------------------
# Contract extraction + reshaping
# -------------------------

def _extract_contract_event(turn_events: List[Event], event_type: str, tick: Optional[int] = None) -> Optional[Event]:
    ev = _pick_last(turn_events, event_type, tick=tick)
    if ev:
        return ev
    return None


def _shape_tick1_intent_envelope_expected(
    tev: List[Event],
    user_text: str,
    meaning_block: Dict[str, Any],
    fallback_turn_state: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    warnings: List[str] = []

    ev = _extract_contract_event(tev, "TICK1_INTENT_ENVELOPE", tick=1)
    if ev:
        payload = ev.get("payload")
        p = payload if isinstance(payload, dict) else {}
        ids = {
            "policy_req_seq": ev.get("policy_req_seq"),
            "correlation_id": ev.get("correlation_id"),
            "model_call_id": ev.get("model_call_id"),
            "toolplan_id": ev.get("toolplan_id"),
        }

        if "raw" in p and "compat" in p and "ids" in p and "input" in p:
            shaped = dict(p)
            shaped.setdefault("type", "TICK1_INTENT_ENVELOPE")
            shaped.setdefault("turn_id", ev.get("turn_id"))
            shaped.setdefault("tick_id", ev.get("tick_id"))
            shaped.setdefault("tag", ev.tag or "tick1_ok")
            shaped["ids"] = shaped.get("ids") or ids
            return shaped, warnings

        compat = p.get("compat") if isinstance(p.get("compat"), dict) else {}
        raw_intents = p.get("intents") if isinstance(p.get("intents"), list) else []
        shaped = {
            "type": "TICK1_INTENT_ENVELOPE",
            "tag": ev.tag or "tick1_ok",
            "turn_id": ev.get("turn_id"),
            "tick_id": ev.get("tick_id"),
            "ids": ids,
            "input": {
                "user_text": p.get("raw_user_text") or user_text,
                "mode": (fallback_turn_state.get("mode") or p.get("mode") or "GRID_SESSION"),
                "phase": (fallback_turn_state.get("phase") or p.get("phase") or None),
                "pending_before": p.get("pending_before") or None,
                "focus_before": p.get("focus_before") if p.get("focus_before") is not None else fallback_turn_state.get("focus_idx"),
            },
            "raw": {
                "version": p.get("version") or "intent_envelope_v1",
                "intents": raw_intents,
                "is_unclear": p.get("needs_clarification") if p.get("needs_clarification") is not None else p.get("is_unclear"),
                "notes": p.get("notes"),
            },
            "compat": compat,
        }
        return shaped, warnings

    warnings.append("missing_contract_event: TICK1_INTENT_ENVELOPE (tick 1)")

    compat_parsed = meaning_block.get("extracted_assistant_content_parsed")
    if not isinstance(compat_parsed, dict):
        warnings.append("tick1_intent_envelope_synthesis_failed: meaning_v1 content not parsed as JSON")
        return None, warnings

    intent = compat_parsed.get("intent")
    confidence = compat_parsed.get("confidence")
    needs_clarification = compat_parsed.get("needs_clarification")
    slots = compat_parsed.get("slots") if isinstance(compat_parsed.get("slots"), dict) else {}

    raw_intent = {
        "id": "t1:0",
        "type": intent or "UNKNOWN",
        "confidence": confidence if isinstance(confidence, (int, float)) else None,
        "entities": {
            "cell": slots.get("cell") or slots.get("about_cell") or None,
            "cell_index": slots.get("cell_index") or None,
            "digit": slots.get("digit") or None,
            "region": slots.get("region") or None,
            "digits": slots.get("digits") or None,
            "yesno": slots.get("yesno") or None,
            "meta": slots.get("meta") or None,
            "question_type": slots.get("question_type") or None,
        },
        "uncertainty": {
            "needs_clarification": bool(needs_clarification) if needs_clarification is not None else False,
            "missing": [],
            "ambiguity_reason": compat_parsed.get("clarify_question"),
            "alternatives": [],
        },
        "addresses_user_agenda_id": None,
    }

    shaped = {
        "type": "TICK1_INTENT_ENVELOPE",
        "tag": "tick1_synth_from_meaning_v1",
        "turn_id": meaning_block.get("ids", {}).get("turn_id"),
        "tick_id": 1,
        "ids": {
            "policy_req_seq": meaning_block.get("ids", {}).get("policy_req_seq"),
            "correlation_id": meaning_block.get("ids", {}).get("correlation_id"),
            "model_call_id": meaning_block.get("ids", {}).get("model_call_id"),
            "toolplan_id": meaning_block.get("ids", {}).get("toolplan_id"),
        },
        "input": {
            "user_text": user_text,
            "mode": fallback_turn_state.get("mode") or "GRID_SESSION",
            "phase": fallback_turn_state.get("phase") or None,
            "pending_before": fallback_turn_state.get("pending") or None,
            "focus_before": fallback_turn_state.get("focus_idx"),
        },
        "raw": {
            "version": "intent_envelope_v1",
            "intents": [raw_intent],
            "is_unclear": bool(needs_clarification) if needs_clarification is not None else False,
            "notes": "SYNTHESIZED from meaning_v1 response because TICK1_INTENT_ENVELOPE event was not present in telemetry.",
        },
        "compat": {
            "meaning_v1": compat_parsed,
            "derived": {
                "has_any_grid_question_intent": True if (isinstance(intent, str) and "ASK" in intent) else False,
                "has_any_mutation_intent": True if (isinstance(intent, str) and ("EDIT" in intent or "MUTATION" in intent)) else False,
                "has_any_cell_target": bool(raw_intent["entities"].get("cell") or raw_intent["entities"].get("cell_index")),
                "has_any_digit": raw_intent["entities"].get("digit") is not None,
                "top_intent_type": intent or "UNKNOWN",
            }
        }
    }
    return shaped, warnings


def _infer_turn_state_from_anywhere(
    turn_events: List[Event],
    rr_turn_ctx: Dict[str, Any],
    tick2_contract: Optional[Dict[str, Any]],
    turn_context_v1: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if isinstance(tick2_contract, dict):
        t = tick2_contract.get("turn")
        if isinstance(t, dict):
            return {
                "mode": t.get("mode"),
                "phase": t.get("phase"),
                "focus_idx": t.get("focus_after") if t.get("focus_after") is not None else t.get("focus_before"),
                "pending": t.get("pending_after") if t.get("pending_after") is not None else t.get("pending_before"),
            }

    if isinstance(rr_turn_ctx, dict) and rr_turn_ctx:
        return {
            "mode": rr_turn_ctx.get("mode"),
            "phase": rr_turn_ctx.get("phase"),
            "focus_idx": rr_turn_ctx.get("focus_after") if rr_turn_ctx.get("focus_after") is not None else rr_turn_ctx.get("focus_before"),
            "pending": rr_turn_ctx.get("pending_after") if rr_turn_ctx.get("pending_after") is not None else rr_turn_ctx.get("pending_before"),
        }

    if isinstance(turn_context_v1, dict):
        parsed = turn_context_v1.get("parsed")
        if isinstance(parsed, dict):
            mode = parsed.get("mode")
            phase = parsed.get("phase")

            pending_before = None
            p = parsed.get("pending")
            if isinstance(p, dict):
                pending_before = p.get("pending_before") or p.get("pending_kind") or p.get("kind")

            focus_idx = None
            f = parsed.get("focus")
            if isinstance(f, dict):
                cell = f.get("cell")
                if isinstance(cell, dict):
                    r = cell.get("r")
                    c = cell.get("c")
                    if isinstance(r, int) and isinstance(c, int) and 1 <= r <= 9 and 1 <= c <= 9:
                        focus_idx = (r - 1) * 9 + (c - 1)

            return {
                "mode": mode,
                "phase": phase,
                "focus_idx": focus_idx,
                "pending": pending_before,
            }

    ts = _pick_last(turn_events, "TURN_STATE")
    if ts:
        payload = ts.get("payload")
        p = payload if isinstance(payload, dict) else {}
        return {
            "mode": p.get("mode") or ts.get("mode"),
            "phase": p.get("phase") or ts.get("phase"),
            "focus_idx": p.get("focus_cell_index") or p.get("focus_idx"),
            "pending": p.get("pending") or p.get("pending_kind"),
        }

    return {}


def _shape_app_plan_expected(
    ev: Optional[Event],
    tick1_env: Optional[Dict[str, Any]],
    inferred_state_before: Dict[str, Any],
    grid_facts_present: bool,
) -> Optional[Dict[str, Any]]:
    if not ev:
        return None

    payload = ev.get("payload")
    p = payload if isinstance(payload, dict) else {}

    ids = {
        "policy_req_seq": ev.get("policy_req_seq"),
        "correlation_id": ev.get("correlation_id"),
        "model_call_id": ev.get("model_call_id"),
        "toolplan_id": ev.get("toolplan_id"),
    }

    input_block = {
        "mode": inferred_state_before.get("mode") or "GRID_SESSION",
        "phase": inferred_state_before.get("phase") or None,
        "pending_before": inferred_state_before.get("pending") or None,
        "focus_before": inferred_state_before.get("focus_idx"),
        "user_text": (tick1_env.get("input", {}).get("user_text") if isinstance(tick1_env, dict) else None),
    }

    ops = _safe_get_list(p.get("ops"))
    evidence = _safe_get_dict(p.get("evidence_selection"))
    fact_types = evidence.get("fact_types") if isinstance(evidence.get("fact_types"), list) else []
    facts_count = evidence.get("facts_selected_count")

    st = _safe_get_dict(p.get("state_transitions"))
    pending = _safe_get_dict(p.get("pending"))
    no_pending_reason = pending.get("no_pending_reason")

    plan_block = {
        "summary": ev.get("summary") or f"planned_ops={len(ops)}",
        "ops": ops,
        "facts": [],
        "facts_count": facts_count if isinstance(facts_count, int) else (len(fact_types) if isinstance(fact_types, list) else 0),
        "fact_types": fact_types,
        "mutations": {
            "new_mode": st.get("mode_after"),
            "new_phase": st.get("phase_after"),
            "new_focus_idx": st.get("focus_after"),
            "new_pending": pending.get("after"),
        },
        "no_pending_reason": None,
    }

    if (pending.get("after") in (None, "none", "NONE")) and no_pending_reason:
        plan_block["no_pending_reason"] = {
            "code": no_pending_reason,
            "explain": "Planner ended with no pending and empty queues." if no_pending_reason == "NO_APP_AGENDA_AND_NO_USER_AGENDA" else "Planner ended with no pending (see code).",
            "extra": {"app_agenda_n": None, "user_agenda_n": None},
        }

    shaped = {
        "type": "APP_PLAN_V1",
        "tag": ev.tag or "plan_built",
        "turn_id": ev.get("turn_id"),
        "tick_id": ev.get("tick_id"),
        "ids": ids,
        "input": input_block,
        "plan": plan_block,
        "grid_state": {
            "grid_present_in_state": grid_facts_present,
            "llm_snapshot_present": grid_facts_present,
        }
    }
    return shaped


def _shape_tick2_request_contract_expected(ev: Optional[Event]) -> Optional[Dict[str, Any]]:
    if not ev:
        return None
    payload = ev.get("payload")
    p = payload if isinstance(payload, dict) else {}
    shaped = {
        "type": "TICK2_REQUEST_CONTRACT",
        "tag": ev.tag or "tick2_sent",
        "turn_id": ev.get("turn_id"),
        "tick_id": ev.get("tick_id"),
        "ids": {
            "policy_req_seq": ev.get("policy_req_seq"),
            "correlation_id": ev.get("correlation_id"),
            "model_call_id": ev.get("model_call_id"),
            "toolplan_id": ev.get("toolplan_id"),
        },
        "payload": p,
    }
    return shaped







def _event_payload_or_raw(ev: Optional[Event]) -> Dict[str, Any]:
    if not ev:
        return {}
    payload = ev.get("payload")
    if isinstance(payload, dict):
        return payload
    return dict(ev.raw)


def _extract_reply_demand_resolved(turn_events: List[Event]) -> Dict[str, Any]:
    ev = _extract_contract_event(turn_events, "REPLY_DEMAND_RESOLVED", tick=2)
    if not ev:
        return {}
    p = _event_payload_or_raw(ev)
    return {
        "turn_id": ev.get("turn_id"),
        "tick_id": ev.get("tick_id"),
        "category": p.get("category") or p.get("demand_category"),
        "reason": p.get("reason"),
        "phase": p.get("phase"),
        "pending_kind": p.get("pendingKind") or p.get("pending_kind"),
        "story_stage": p.get("storyStage") or p.get("story_stage"),
        "opening_turn": p.get("openingTurn") if p.get("openingTurn") is not None else p.get("opening_turn"),
        "setup_profile": p.get("setupProfile") or p.get("setup_profile"),
        "confrontation_proof_profile": (
            p.get("confrontationProofProfile")
            or p.get("confrontation_proof_profile")
        ),
        "resolution_profile": (
            p.get("resolutionProfile")
            or p.get("resolution_profile")
        ),
    }


def _extract_reply_assembly_plan(turn_events: List[Event]) -> Dict[str, Any]:
    ev = _extract_contract_event(turn_events, "REPLY_ASSEMBLY_PLAN", tick=2)
    if not ev:
        return {}
    p = _event_payload_or_raw(ev)

    required_prompt_modules = p.get("requiredPromptModules") if isinstance(p.get("requiredPromptModules"), list) else p.get("required_prompt_modules")
    required_channels = p.get("requiredChannels") if isinstance(p.get("requiredChannels"), list) else p.get("required_channels")
    optional_channels = p.get("optionalChannels") if isinstance(p.get("optionalChannels"), list) else p.get("optional_channels")
    forbidden_channels = p.get("forbiddenChannels") if isinstance(p.get("forbiddenChannels"), list) else p.get("forbidden_channels")
    selected_prompt_modules = p.get("selectedPromptModules") if isinstance(p.get("selectedPromptModules"), list) else p.get("selected_prompt_modules")
    selected_channels = p.get("selectedChannels") if isinstance(p.get("selectedChannels"), list) else p.get("selected_channels")

    return {
        "turn_id": ev.get("turn_id"),
        "tick_id": ev.get("tick_id"),
        "demand_category": p.get("demandCategory") or p.get("demand_category"),
        "required_prompt_modules": required_prompt_modules or [],
        "required_channels": required_channels or [],
        "optional_channels": optional_channels or [],
        "forbidden_channels": forbidden_channels or [],
        "selected_prompt_modules": selected_prompt_modules or [],
        "selected_channels": selected_channels or [],
        "rollout_mode": p.get("rolloutMode") or p.get("rollout_mode"),
        "soft_char_budget": p.get("softCharBudget") or p.get("soft_char_budget"),
        "soft_token_budget": p.get("softTokenBudget") or p.get("soft_token_budget"),
        "notes": p.get("notes"),
    }


def _parse_projected_payload(raw: Any) -> Any:
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, str) and raw.strip():
        parsed = _try_json_loads(raw)
        if parsed is not None:
            return parsed
        return raw
    return None


def _extract_reply_projected_channels(turn_events: List[Event]) -> Dict[str, Any]:
    ev = _extract_contract_event(turn_events, "REPLY_PROJECTED_CHANNELS", tick=2)
    if not ev:
        return {}

    p = _event_payload_or_raw(ev)

    channels: List[Dict[str, Any]] = []

    # Real telemetry shape:
    # "channels": {
    #   "CHANNEL_NAME": "{\"json\":\"string\"}",
    #   ...
    # }
    raw_channels_dict = p.get("channels")
    if isinstance(raw_channels_dict, dict):
        for channel_name, raw_payload in raw_channels_dict.items():
            payload = _parse_projected_payload(raw_payload)

            payload_len = 0
            if isinstance(raw_payload, str):
                payload_len = len(raw_payload)
            elif payload is not None:
                try:
                    payload_len = len(json.dumps(payload, ensure_ascii=False))
                except Exception:
                    payload_len = len(str(payload))

            channels.append({
                "channel": channel_name,
                "payload": payload,
                "payload_raw": raw_payload,
                "payload_len": payload_len,
                "payload_sha12": None,
            })

        return {
            "turn_id": ev.get("turn_id"),
            "tick_id": ev.get("tick_id"),
            "channels": channels,
        }

    # Legacy / alternate shape support:
    raw_channels_list = (
        p.get("channels")
        if isinstance(p.get("channels"), list)
        else p.get("projected_channels")
        if isinstance(p.get("projected_channels"), list)
        else []
    )

    for item in raw_channels_list:
        if not isinstance(item, dict):
            continue

        channel = item.get("channel") or item.get("name") or item.get("type")
        raw_payload = (
            item.get("payload")
            if item.get("payload") is not None
            else item.get("payload_preview")
        )
        payload = _parse_projected_payload(raw_payload)

        payload_len = item.get("payload_len")
        if not isinstance(payload_len, int):
            if isinstance(raw_payload, str):
                payload_len = len(raw_payload)
            else:
                try:
                    payload_len = len(json.dumps(payload, ensure_ascii=False)) if payload is not None else 0
                except Exception:
                    payload_len = len(str(payload)) if payload is not None else 0

        channels.append({
            "channel": channel,
            "payload": payload,
            "payload_raw": raw_payload,
            "payload_len": payload_len,
            "payload_sha12": item.get("payload_sha12"),
        })

    return {
        "turn_id": ev.get("turn_id"),
        "tick_id": ev.get("tick_id"),
        "channels": channels,
    }


def _extract_reply_waste_audit(turn_events: List[Event]) -> Dict[str, Any]:
    ev = _extract_contract_event(turn_events, "REPLY_WASTE_AUDIT", tick=2)
    if not ev:
        return {}
    p = _event_payload_or_raw(ev)
    return dict(p)


def _extract_setup_reply_coverage(turn_events: List[Event]) -> Dict[str, Any]:
    ev = _extract_contract_event(turn_events, "SETUP_REPLY_COVERAGE", tick=2)
    if not ev:
        return {}
    p = _event_payload_or_raw(ev)
    return dict(p)


def _extract_confrontation_reply_coverage(turn_events: List[Event]) -> Dict[str, Any]:
    ev = _extract_contract_event(turn_events, "CONFRONTATION_REPLY_COVERAGE", tick=2)
    if not ev:
        return {}
    p = _event_payload_or_raw(ev)
    return dict(p)


def _extract_resolution_reply_coverage(turn_events: List[Event]) -> Dict[str, Any]:
    ev = _extract_contract_event(turn_events, "RESOLUTION_REPLY_COVERAGE", tick=2)
    if not ev:
        return {}
    p = dict(_event_payload_or_raw(ev))
    if "resolution_profile" not in p:
        p["resolution_profile"] = (
            p.get("resolutionProfile")
            or p.get("profile")
        )
    return p


def _extract_prompt_surface_from_reply_block(reply_block: Dict[str, Any]) -> Dict[str, Any]:
    def _content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if content is None:
            return ""
        try:
            return json.dumps(content, ensure_ascii=False, indent=2)
        except Exception:
            return str(content)

    out: Dict[str, Any] = {
        "system_prompt_len": None,
        "developer_prompt_len": None,
        "user_message_len": None,
        "messages_count": None,
        "message_roles": [],
        "estimated_total_prompt_len": None,
        "parsed_messages_total_content_len": None,
        "system_prompt_text": None,
        "developer_prompt_text": None,
        "user_message_text": None,
        "message_blocks": [],
    }

    if not isinstance(reply_block, dict):
        return out

    # Prefer top-level payload_len from the actual MODEL_PAYLOAD_OUT event.
    payload_len_meta = reply_block.get("payload_len")
    if not isinstance(payload_len_meta, int):
        req_meta = _safe_get_dict(reply_block.get("req"))
        payload_len_meta = req_meta.get("payload_len")
    out["estimated_total_prompt_len"] = payload_len_meta if isinstance(payload_len_meta, int) else None

    # Try all known locations in order.
    candidate_payload_texts: List[str] = []

    nested_payload_out = _safe_get_dict(reply_block.get("payload_out"))
    for candidate in (
        nested_payload_out.get("payload_text"),
        reply_block.get("payload_text"),
        nested_payload_out.get("payload_preview"),
        reply_block.get("payload_preview"),
    ):
        if isinstance(candidate, str) and candidate.strip():
            candidate_payload_texts.append(candidate)

    req = None
    for raw in candidate_payload_texts:
        parsed = _try_json_loads(raw)
        if isinstance(parsed, dict) and isinstance(parsed.get("messages"), list):
            req = parsed
            break

    if not isinstance(req, dict):
        return out

    msgs = req.get("messages")
    if not isinstance(msgs, list):
        return out

    out["messages_count"] = len(msgs)

    total_len = 0
    system_len = 0
    developer_len = 0
    user_len = 0
    roles: List[str] = []

    system_blocks: List[str] = []
    developer_blocks: List[str] = []
    user_blocks: List[str] = []
    message_blocks: List[Dict[str, Any]] = []

    for i, m in enumerate(msgs):
        if not isinstance(m, dict):
            continue

        role = m.get("role")
        role_name = role if isinstance(role, str) else "unknown"
        roles.append(role_name)

        content_text = _content_to_text(m.get("content"))
        content_len = len(content_text)
        total_len += content_len

        if role_name == "system":
            system_len += content_len
            system_blocks.append(content_text)
        elif role_name == "developer":
            developer_len += content_len
            developer_blocks.append(content_text)
        elif role_name == "user":
            user_len += content_len
            user_blocks.append(content_text)

        message_blocks.append({
            "index": i,
            "role": role_name,
            "content_len": content_len,
            "content": content_text,
        })

    out["message_roles"] = roles
    out["system_prompt_len"] = system_len
    out["developer_prompt_len"] = developer_len
    out["user_message_len"] = user_len
    out["parsed_messages_total_content_len"] = total_len
    out["system_prompt_text"] = "\n\n".join([b for b in system_blocks if b]).strip() or None
    out["developer_prompt_text"] = "\n\n".join([b for b in developer_blocks if b]).strip() or None
    out["user_message_text"] = "\n\n".join([b for b in user_blocks if b]).strip() or None
    out["message_blocks"] = message_blocks

    if out["estimated_total_prompt_len"] is None:
        out["estimated_total_prompt_len"] = total_len

    return out





def _receipt_for_channel(channel: str, payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {
            "semantic_summary": "Channel payload was not structured JSON.",
            "field_receipt": {"raw": payload},
        }

    if channel == "SETUP_REPLY_PACKET":
        technique = _safe_get_dict(payload.get("technique"))
        target = _safe_get_dict(payload.get("target"))
        orientation = _safe_get_dict(payload.get("orientation"))
        lens = _safe_get_dict(payload.get("lens"))
        trigger_overview = _safe_get_dict(payload.get("trigger_overview"))
        trigger_statement = _safe_get_dict(payload.get("trigger_statement"))
        bridge = _safe_get_dict(payload.get("bridge"))
        setup_only_line = _safe_get_dict(payload.get("setup_only_line"))
        cta = _safe_get_dict(payload.get("cta"))
        support = _safe_get_dict(payload.get("support"))
        bounded_rows = _safe_get_list(payload.get("bounded_trigger_rows"))
        pattern_member_proof_rows = _safe_get_list(payload.get("pattern_member_proof_rows"))

        support_count = _as_int(support.get("bounded_trigger_row_count"), 0)
        top_level_count = len(bounded_rows)

        support_pattern_count = _as_int(support.get("pattern_member_proof_row_count"), 0)
        top_level_pattern_count = len(pattern_member_proof_rows)

        return {
            "semantic_summary": "Canonical setup packet supplied to the model.",
            "field_receipt": {
                "setup_profile": payload.get("setup_profile"),
                "archetype": payload.get("archetype"),
                "technique": {
                    "id": technique.get("id"),
                    "technique_name": technique.get("technique_name") or technique.get("name"),
                    "name": technique.get("name") or technique.get("technique_name"),
                    "real_name": technique.get("real_name"),
                    "family": technique.get("family"),
                    "archetype": technique.get("archetype"),
                    "is_base": technique.get("is_base"),
                    "difficulty_level": technique.get("difficulty_level"),
                    "short_definition_summary": technique.get("short_definition_summary"),
                },
                "target": {
                    "cell": target.get("cell"),
                    "primary_house": target.get("primary_house"),
                    "target_digit": target.get("target_digit"),
                    "can_say_target_digit": target.get("can_say_target_digit"),
                    "spoiler_level": target.get("spoiler_level"),
                },
                "orientation": {
                    "summary": orientation.get("summary"),
                },
                "lens": {
                    "summary": lens.get("summary"),
                    "setup_role": lens.get("setup_role"),
                    "trigger_kind": lens.get("trigger_kind"),
                    "must_include_bounded_trigger_member_explanation": lens.get(
                        "must_include_bounded_trigger_member_explanation"
                    ),
                },
                "trigger_overview": {
                    "summary": trigger_overview.get("summary"),
                },
                "bounded_trigger_rows_count": top_level_count,
                "bounded_trigger_rows": bounded_rows,
                "pattern_member_proof_rows_count": top_level_pattern_count,
                "pattern_member_proof_rows": pattern_member_proof_rows,
                "trigger_statement": {
                    "summary": trigger_statement.get("summary"),
                    "pattern_kind": trigger_statement.get("pattern_kind"),
                },
                "bridge": {
                    "summary": bridge.get("summary"),
                    "why_this_matters": bridge.get("why_this_matters"),
                    "final_resolution_setup_summary": bridge.get("final_resolution_setup_summary"),
                },
                "setup_only_line": {
                    "summary": setup_only_line.get("summary"),
                    "spoiler_level": setup_only_line.get("spoiler_level"),
                    "claim_code": setup_only_line.get("claim_code"),
                },
                "cta": {
                    "kind": cta.get("kind"),
                },
                "support": support,
                "consistency_checks": {
                    "bounded_trigger_rows_count_matches_support": top_level_count == support_count,
                    "top_level_bounded_trigger_rows_count": top_level_count,
                    "support_bounded_trigger_row_count": support_count,
                    "pattern_member_proof_rows_count_matches_support": top_level_pattern_count == support_pattern_count,
                    "top_level_pattern_member_proof_rows_count": top_level_pattern_count,
                    "support_pattern_member_proof_row_count": support_pattern_count,
                },
            },
        }

    if channel == "CONFRONTATION_REPLY_PACKET":
        technique = _safe_get_dict(payload.get("technique"))
        target = _safe_get_dict(payload.get("target"))
        trigger_reference = _safe_get_dict(payload.get("trigger_reference"))
        trigger_effect = _safe_get_dict(payload.get("trigger_effect"))
        target_resolution_truth = _safe_get_dict(payload.get("target_resolution_truth"))
        collapse = _safe_get_dict(payload.get("collapse"))
        pre_commit_line = _safe_get_dict(payload.get("pre_commit_line"))
        cta = _safe_get_dict(payload.get("cta"))
        support = _safe_get_dict(payload.get("support"))
        proof_row_policy = _safe_get_dict(payload.get("proof_row_policy"))
        target_rows = _safe_get_list(payload.get("target_proof_rows"))
        technique_blocker_rows = _safe_get_list(payload.get("technique_blocker_rows"))
        peer_blocker_rows = _safe_get_list(payload.get("peer_blocker_rows"))
        ordered_proof_ladder = _safe_get_list(payload.get("ordered_proof_ladder"))

        support_count = _as_int(support.get("target_proof_row_count"), 0)
        raw_count = _as_int(support.get("raw_target_proof_row_count"), support_count)
        row_limit = _as_int(support.get("proof_row_limit"), 0)
        truncated = bool(support.get("proof_rows_truncated"))

        support_technique_count = _as_int(support.get("technique_blocker_row_count"), 0)
        support_peer_count = _as_int(support.get("peer_blocker_row_count"), 0)
        support_ladder_count = _as_int(support.get("ordered_proof_ladder_step_count"), 0)

        return {
            "semantic_summary": "Canonical confrontation packet supplied to the model.",
            "field_receipt": {
                "proof_profile": payload.get("proof_profile"),
                "archetype": payload.get("archetype"),
                "technique": {
                    "id": technique.get("id"),
                    "technique_name": technique.get("technique_name") or technique.get("name"),
                    "name": technique.get("name") or technique.get("technique_name"),
                    "real_name": technique.get("real_name"),
                    "family": technique.get("family"),
                    "archetype": technique.get("archetype"),
                    "is_base": technique.get("is_base"),
                    "difficulty_level": technique.get("difficulty_level"),
                    "short_definition_summary": technique.get("short_definition_summary"),
                },
                "target": {
                    "cell": target.get("cell"),
                    "primary_house": target.get("primary_house"),
                    "target_digit": target.get("target_digit"),
                    "can_say_target_digit": target.get("can_say_target_digit"),
                    "spoiler_level": target.get("spoiler_level"),
                },
                "trigger_reference": {
                    "summary": trigger_reference.get("summary"),
                    "trigger_kind": trigger_reference.get("trigger_kind"),
                    "reference_only": trigger_reference.get("reference_only"),
                    "pattern_reteach_forbidden": trigger_reference.get("pattern_reteach_forbidden"),
                },
                "trigger_effect": {
                    "summary": trigger_effect.get("summary"),
                },
                "target_resolution_truth": target_resolution_truth,
                "target_proof_rows_count": len(target_rows),
                "target_proof_rows": target_rows,
                "technique_blocker_rows_count": len(technique_blocker_rows),
                "technique_blocker_rows": technique_blocker_rows,
                "peer_blocker_rows_count": len(peer_blocker_rows),
                "peer_blocker_rows": peer_blocker_rows,
                "ordered_proof_ladder_step_count": len(ordered_proof_ladder),
                "ordered_proof_ladder": ordered_proof_ladder,
                "proof_row_policy": proof_row_policy,
                "collapse": {
                    "summary": collapse.get("summary"),
                    "remaining_candidate_digits": collapse.get("remaining_candidate_digits"),
                    "surviving_digit": collapse.get("surviving_digit"),
                    "can_say_target_digit": collapse.get("can_say_target_digit"),
                    "is_single_after_cleanup": collapse.get("is_single_after_cleanup"),
                    "two_layer_honesty_line": collapse.get("two_layer_honesty_line"),
                },
                "pre_commit_line": {
                    "summary": pre_commit_line.get("summary"),
                    "claim_code": pre_commit_line.get("claim_code"),
                    "spoiler_level": pre_commit_line.get("spoiler_level"),
                },
                "cta": {
                    "kind": cta.get("kind"),
                },
                "support": support,
                "consistency_checks": {
                    "target_proof_rows_count_matches_support": len(target_rows) == support_count,
                    "top_level_target_proof_rows_count": len(target_rows),
                    "support_target_proof_row_count": support_count,
                    "support_raw_target_proof_row_count": raw_count,
                    "proof_row_limit": row_limit,
                    "proof_rows_truncated": truncated,
                    "technique_blocker_rows_count_matches_support": len(technique_blocker_rows) == support_technique_count,
                    "top_level_technique_blocker_rows_count": len(technique_blocker_rows),
                    "support_technique_blocker_row_count": support_technique_count,
                    "peer_blocker_rows_count_matches_support": len(peer_blocker_rows) == support_peer_count,
                    "top_level_peer_blocker_rows_count": len(peer_blocker_rows),
                    "support_peer_blocker_row_count": support_peer_count,
                    "ordered_proof_ladder_count_matches_support": len(ordered_proof_ladder) == support_ladder_count,
                    "top_level_ordered_proof_ladder_step_count": len(ordered_proof_ladder),
                    "support_ordered_proof_ladder_step_count": support_ladder_count,
                },
            },
        }

    if channel == "RESOLUTION_REPLY_PACKET":
        technique = _safe_get_dict(payload.get("technique"))
        commit = _safe_get_dict(payload.get("commit"))
        recap = _safe_get_dict(payload.get("recap"))
        technique_contribution = _safe_get_dict(payload.get("technique_contribution"))
        final_forcing = _safe_get_dict(payload.get("final_forcing"))
        honesty = _safe_get_dict(payload.get("honesty"))
        present_state_line = _safe_get_dict(payload.get("present_state_line"))
        post_commit = _safe_get_dict(payload.get("post_commit"))
        cta = _safe_get_dict(payload.get("cta"))
        support = _safe_get_dict(payload.get("support"))

        return {
            "semantic_summary": "Canonical resolution packet supplied to the model.",
            "field_receipt": {
                "resolution_profile": payload.get("resolution_profile"),
                "archetype": payload.get("archetype"),
                "technique": {
                    "id": technique.get("id"),
                    "technique_name": technique.get("technique_name") or technique.get("name"),
                    "name": technique.get("name") or technique.get("technique_name"),
                    "real_name": technique.get("real_name"),
                    "family": technique.get("family"),
                    "archetype": technique.get("archetype"),
                    "is_base": technique.get("is_base"),
                    "difficulty_level": technique.get("difficulty_level"),
                    "short_definition_summary": technique.get("short_definition_summary"),
                },
                "commit": {
                    "cell": commit.get("cell"),
                    "digit": commit.get("digit"),
                    "authorized": commit.get("authorized"),
                    "present_state_language_required": commit.get("present_state_language_required"),
                    "commit_truth_source": commit.get("commit_truth_source"),
                    "claim_code": commit.get("claim_code"),
                    "spoiler_level": commit.get("spoiler_level"),
                },
                "recap": {
                    "summary": recap.get("summary"),
                    "compact_mode": recap.get("compact_mode"),
                    "max_beats": recap.get("max_beats"),
                },
                "technique_contribution": {
                    "summary": technique_contribution.get("summary"),
                    "kind": technique_contribution.get("kind"),
                },
                "final_forcing": {
                    "summary": final_forcing.get("summary"),
                    "resolution_kind": final_forcing.get("resolution_kind"),
                    "surviving_digit": final_forcing.get("surviving_digit"),
                },
                "honesty": {
                    "two_layer_honesty_line": honesty.get("two_layer_honesty_line"),
                    "must_distinguish_technique_from_finish": honesty.get("must_distinguish_technique_from_finish"),
                },
                "present_state_line": {
                    "summary": present_state_line.get("summary"),
                },
                "post_commit": {
                    "board_delta_summary": post_commit.get("board_delta_summary"),
                    "placement_count": post_commit.get("placement_count"),
                },
                "cta": {
                    "kind": cta.get("kind"),
                },
                "support": support,
            },
        }

    if channel == "CONTINUITY_SHORT":
        user_tally = _safe_get_dict(payload.get("user_tally"))
        assistant_tally = _safe_get_dict(payload.get("assistant_tally"))
        recent_turns = _safe_get_list(payload.get("recent_turns"))
        return {
            "semantic_summary": "Small continuity packet supplied.",
            "field_receipt": {
                "user_tally.name": user_tally.get("name"),
                "assistant_tally.name": assistant_tally.get("name"),
                "assistant_tally.personality": assistant_tally.get("personality"),
                "transition_hint": payload.get("transition_hint"),
                "recent_turns_count": len(recent_turns),
                "recent_turns": recent_turns,
            },
        }

    if channel == "HANDOVER_NOTE_MINI":
        return {
            "semantic_summary": "Tiny transition bridge supplied.",
            "field_receipt": {
                "previous_technique_short_label": payload.get("previous_technique_short_label"),
                "next_technique_short_label": payload.get("next_technique_short_label"),
                "relation": payload.get("relation"),
                "bridge_hint": payload.get("bridge_hint"),
            },
        }

    if channel == "GLOSSARY_MINI":
        return {
            "semantic_summary": "Small glossary/support packet supplied.",
            "field_receipt": {
                "technique_player_name": payload.get("technique_player_name"),
                "family_description_short": payload.get("family_description_short"),
                "relevant_term": payload.get("relevant_term"),
                "say": payload.get("say"),
                "dont_say": payload.get("dont_say"),
            },
        }

    if channel == "CTA_CONTEXT":
        return {
            "semantic_summary": "CTA context supplied.",
            "field_receipt": payload,
        }

    if channel == "TURN_HEADER_MINI":
        return {
            "semantic_summary": "Turn header supplied.",
            "field_receipt": payload,
        }

    if channel == "STYLE_MINI":
        return {
            "semantic_summary": "Style packet supplied.",
            "field_receipt": payload,
        }

    if channel == "DECISION_SUMMARY_MINI":
        return {
            "semantic_summary": "Decision summary supplied.",
            "field_receipt": payload,
        }

    if channel in {
        "STEP_CLARIFICATION_PACKET",
        "PROOF_CHALLENGE_PACKET",
        "USER_REASONING_CHECK_PACKET",
        "ALTERNATIVE_TECHNIQUE_PACKET",
        "TARGET_CELL_QUERY_PACKET",
        "CANDIDATE_STATE_PACKET",
        "NEIGHBOR_CELL_QUERY_PACKET",
        "RETURN_TO_ROUTE_PACKET",
    }:
        return {
            "semantic_summary": "Heuristic detour packet supplied.",
            "field_receipt": payload,
        }

    if channel == "SOLVER_CELL_CANDIDATES_PACKET":
        result = _safe_get_dict(payload.get("result"))
        return {
            "semantic_summary": "Solver-backed single-cell candidate packet supplied.",
            "field_receipt": {
                "cell": result.get("cell"),
                "cell_index": result.get("cell_index"),
                "row": result.get("row"),
                "col": result.get("col"),
                "box": result.get("box"),
                "digits": result.get("digits"),
                "mask": result.get("mask"),
            },
        }

    if channel == "SOLVER_CELLS_CANDIDATES_PACKET":
        result = _safe_get_dict(payload.get("result"))
        cells = _safe_get_list(result.get("cells"))
        return {
            "semantic_summary": "Solver-backed multi-cell candidates packet supplied.",
            "field_receipt": {
                "count": result.get("count"),
                "cells": cells,
            },
        }

    if channel == "SOLVER_HOUSE_CANDIDATE_MAP_PACKET":
        result = _safe_get_dict(payload.get("result"))
        return {
            "semantic_summary": "Solver-backed house candidate map supplied.",
            "field_receipt": {
                "house": result.get("house"),
                "map": result.get("map"),
            },
        }

    if channel == "SOLVER_CELL_DIGIT_BLOCKERS_PACKET":
        result = _safe_get_dict(payload.get("result"))
        blocker_analysis = _safe_get_dict(result.get("blocker_analysis"))
        return {
            "semantic_summary": "Solver-backed blocker analysis for one cell/digit supplied.",
            "field_receipt": {
                "cell": result.get("cell"),
                "cell_index": result.get("cell_index"),
                "digit": result.get("digit"),
                "is_candidate": result.get("is_candidate"),
                "candidates_now": result.get("candidates_now"),
                "blocker_analysis": blocker_analysis,
            },
        }

    if channel == "SOLVER_REASONING_CHECK_PACKET":
        result = _safe_get_dict(payload.get("result"))
        return {
            "semantic_summary": "Solver-backed candidate-claim validation packet supplied.",
            "field_receipt": {
                "cell": result.get("cell"),
                "cell_index": result.get("cell_index"),
                "claimed_digits": result.get("claimed_digits"),
                "actual_digits": result.get("actual_digits"),
                "extra_digits": result.get("extra_digits"),
                "missing_digits": result.get("missing_digits"),
                "verdict": result.get("verdict"),
            },
        }

    if channel == "SOLVER_ALTERNATIVE_TECHNIQUE_PACKET":
        result = _safe_get_dict(payload.get("result"))
        return {
            "semantic_summary": "Solver-backed alternative-technique comparison packet supplied.",
            "field_receipt": result,
        }

    if channel == "SOLVER_TECHNIQUE_SCOPE_CHECK_PACKET":
        result = _safe_get_dict(payload.get("result"))
        return {
            "semantic_summary": "Solver-backed technique-in-scope check supplied.",
            "field_receipt": {
                "requested_technique": result.get("requested_technique"),
                "scope": result.get("scope"),
                "next_step": result.get("next_step"),
                "verdict": result.get("verdict"),
                "search_mode": result.get("search_mode"),
            },
        }

    if channel == "SOLVER_LOCAL_MOVE_SEARCH_PACKET":
        result = _safe_get_dict(payload.get("result"))
        return {
            "semantic_summary": "Solver-backed scoped local move search packet supplied.",
            "field_receipt": {
                "scope": result.get("scope"),
                "count": result.get("count"),
                "max_results": result.get("max_results"),
                "search_mode": result.get("search_mode"),
                "moves": result.get("moves"),
            },
        }

    if channel == "SOLVER_ROUTE_COMPARISON_PACKET":
        result = _safe_get_dict(payload.get("result"))
        return {
            "semantic_summary": "Solver-backed route-vs-scope comparison packet supplied.",
            "field_receipt": {
                "scope": result.get("scope"),
                "comparison": result.get("comparison"),
            },
        }

    if channel == "SOLVER_SCOPED_SUPPORT_PACKET":
        result = _safe_get_dict(payload.get("result"))
        return {
            "semantic_summary": "Solver-backed scoped support snapshot supplied.",
            "field_receipt": result,
        }

    return {
        "semantic_summary": "Projected payload supplied.",
        "field_receipt": payload,
    }









def _build_selected_supply_summary(
    demand: Dict[str, Any],
    plan: Dict[str, Any],
    projected: Dict[str, Any],
    waste: Dict[str, Any],
) -> List[Dict[str, Any]]:
    required = set(plan.get("required_channels") or [])
    optional = set(plan.get("optional_channels") or [])
    forbidden = set(plan.get("forbidden_channels") or [])
    selected = list(plan.get("selected_channels") or [])

    projected_map = {
        c.get("channel"): c
        for c in (projected.get("channels") or [])
        if isinstance(c, dict) and c.get("channel")
    }

    waste_chars = waste.get("channel_chars") if isinstance(waste.get("channel_chars"), dict) else {}

    ordered_names: List[str] = []
    for ch in (plan.get("required_channels") or []) + (plan.get("optional_channels") or []) + selected + list(projected_map.keys()):
        if ch not in ordered_names:
            ordered_names.append(ch)

    solver_channels = {
        "SOLVER_CELL_CANDIDATES_PACKET",
        "SOLVER_CELLS_CANDIDATES_PACKET",
        "SOLVER_HOUSE_CANDIDATE_MAP_PACKET",
        "SOLVER_CELL_DIGIT_BLOCKERS_PACKET",
        "SOLVER_REASONING_CHECK_PACKET",
        "SOLVER_ALTERNATIVE_TECHNIQUE_PACKET",
        "SOLVER_TECHNIQUE_SCOPE_CHECK_PACKET",
        "SOLVER_LOCAL_MOVE_SEARCH_PACKET",
        "SOLVER_ROUTE_COMPARISON_PACKET",
        "SOLVER_SCOPED_SUPPORT_PACKET",
    }

    heuristic_detour_channels = {
        "STEP_CLARIFICATION_PACKET",
        "PROOF_CHALLENGE_PACKET",
        "USER_REASONING_CHECK_PACKET",
        "ALTERNATIVE_TECHNIQUE_PACKET",
        "TARGET_CELL_QUERY_PACKET",
        "CANDIDATE_STATE_PACKET",
        "NEIGHBOR_CELL_QUERY_PACKET",
        "RETURN_TO_ROUTE_PACKET",
    }

    rows: List[Dict[str, Any]] = []
    for ch in ordered_names:
        proj = projected_map.get(ch) or {}
        payload = proj.get("payload")
        projected_yes = ch in projected_map or ch in waste_chars

        payload_len = proj.get("payload_len")
        if not isinstance(payload_len, int) or payload_len <= 0:
            payload_len = _as_int(waste_chars.get(ch), 0)

        selected_yes = ch in selected

        status = "OPTIONAL_MISSING"
        if ch in forbidden and selected_yes:
            status = "FORBIDDEN_PRESENT"
        elif ch in required and not selected_yes:
            status = "REQUIRED_MISSING"
        elif selected_yes and projected_yes and payload_len > 0:
            status = "SELECTED_OK"
        elif selected_yes and projected_yes and payload_len == 0:
            status = "SELECTED_EMPTY"
        elif selected_yes and not projected_yes:
            status = "SELECTED_NOT_PROJECTED"
        elif (not selected_yes) and projected_yes and payload_len > 0:
            status = "PROJECTED_ONLY"
        elif ch in optional and not selected_yes:
            status = "OPTIONAL_MISSING"

        family = "other"
        if ch in solver_channels:
            family = "solver_backed_detour"
        elif ch in heuristic_detour_channels:
            family = "heuristic_detour"
        elif ch.endswith("_REPLY_PACKET"):
            family = "packet_core"
        elif ch.endswith("_MINI"):
            family = "mini"
        elif ch == "CONTINUITY_SHORT":
            family = "continuity"

        rows.append({
            "channel_name": ch,
            "role": "required" if ch in required else "optional" if ch in optional else "other",
            "family": family,
            "selected": selected_yes,
            "projected": projected_yes,
            "payload_len": payload_len,
            "payload_sha12": proj.get("payload_sha12"),
            "status": status,
            "payload": payload,
            "payload_raw": proj.get("payload_raw"),
        })

    return rows


def _build_reality_vs_contract_verdict(
    demand: Dict[str, Any],
    plan: Dict[str, Any],
    supply_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    required_modules = set(plan.get("required_prompt_modules") or [])
    selected_modules = set(plan.get("selected_prompt_modules") or [])
    required_channels = set(plan.get("required_channels") or [])
    selected_channels = set(plan.get("selected_channels") or [])
    forbidden_channels = set(plan.get("forbidden_channels") or [])

    missing_required_channels = [r["channel_name"] for r in supply_rows if r["status"] == "REQUIRED_MISSING"]
    selected_empty = [r["channel_name"] for r in supply_rows if r["status"] == "SELECTED_EMPTY"]
    forbidden_present = sorted(list(selected_channels & forbidden_channels))

    solver_rows = [r for r in supply_rows if r.get("family") == "solver_backed_detour" and _as_int(r.get("payload_len"), 0) > 0]
    heuristic_detour_rows = [r for r in supply_rows if r.get("family") == "heuristic_detour" and _as_int(r.get("payload_len"), 0) > 0]

    solver_backed_packet_types = [r.get("channel_name") for r in solver_rows]
    heuristic_detour_packet_types = [r.get("channel_name") for r in heuristic_detour_rows]

    setup_packet_len = 0
    confrontation_packet_len = 0
    resolution_packet_len = 0
    continuity_len = 0

    for r in supply_rows:
        if r["channel_name"] == "SETUP_REPLY_PACKET":
            setup_packet_len = _as_int(r.get("payload_len"), 0)
        if r["channel_name"] == "CONFRONTATION_REPLY_PACKET":
            confrontation_packet_len = _as_int(r.get("payload_len"), 0)
        if r["channel_name"] == "RESOLUTION_REPLY_PACKET":
            resolution_packet_len = _as_int(r.get("payload_len"), 0)
        if r["channel_name"] == "CONTINUITY_SHORT":
            continuity_len = _as_int(r.get("payload_len"), 0)

    packet_centered_setup_active = None
    packet_centered_confrontation_active = None
    packet_centered_resolution_active = None
    continuity_subordinate = None

    if demand.get("category") == "SOLVING_SETUP":
        packet_centered_setup_active = "SETUP_REPLY_PACKET" in selected_channels and setup_packet_len > 0
        continuity_subordinate = (
            continuity_len <= int(setup_packet_len * 0.35)
        ) if setup_packet_len > 0 else None

    if demand.get("category") == "SOLVING_CONFRONTATION":
        packet_centered_confrontation_active = (
            "CONFRONTATION_REPLY_PACKET" in selected_channels and confrontation_packet_len > 0
        )
        continuity_subordinate = (
            continuity_len <= int(confrontation_packet_len * 0.35)
        ) if confrontation_packet_len > 0 else None

    if demand.get("category") == "SOLVING_RESOLUTION":
        packet_centered_resolution_active = (
            "RESOLUTION_REPLY_PACKET" in selected_channels and resolution_packet_len > 0
        )
        continuity_subordinate = (
            continuity_len <= int(resolution_packet_len * 0.35)
        ) if resolution_packet_len > 0 else None

    risk_flags: List[str] = []
    if missing_required_channels:
        risk_flags.append("missing_required_channels")
    if forbidden_present:
        risk_flags.append("forbidden_channel_leak")
    if selected_empty:
        risk_flags.append("selected_empty_channel")

    if demand.get("category") == "SOLVING_SETUP" and packet_centered_setup_active is False:
        risk_flags.append("setup_starvation")
    if demand.get("category") == "SOLVING_CONFRONTATION" and packet_centered_confrontation_active is False:
        risk_flags.append("confrontation_starvation")
    if demand.get("category") == "SOLVING_RESOLUTION" and packet_centered_resolution_active is False:
        risk_flags.append("resolution_starvation")

    if continuity_subordinate is False:
        if demand.get("category") == "SOLVING_SETUP":
            risk_flags.append("continuity_dominates_setup")
        elif demand.get("category") == "SOLVING_CONFRONTATION":
            risk_flags.append("continuity_dominates_confrontation")
        elif demand.get("category") == "SOLVING_RESOLUTION":
            risk_flags.append("continuity_dominates_resolution")

    detour_mode = bool(
        solver_backed_packet_types
        or heuristic_detour_packet_types
        or (demand.get("detour_question_class"))
    )
    solver_backed_detour_mode = bool(solver_backed_packet_types)
    heuristic_detour_mode = bool(heuristic_detour_packet_types) and not solver_backed_detour_mode

    if detour_mode and not solver_backed_detour_mode:
        risk_flags.append("detour_without_solver_backing")
    if detour_mode and heuristic_detour_mode and solver_backed_detour_mode:
        risk_flags.append("mixed_heuristic_and_solver_detour_packets")

    return {
        "required_prompt_modules_satisfied": required_modules.issubset(selected_modules) if required_modules else True,
        "required_channels_satisfied": len(missing_required_channels) == 0,
        "missing_required_channels": missing_required_channels,
        "forbidden_channels_clean": len(forbidden_present) == 0,
        "forbidden_present": forbidden_present,
        "selected_empty_channels": selected_empty,
        "packet_centered_setup_active": packet_centered_setup_active,
        "packet_centered_confrontation_active": packet_centered_confrontation_active,
        "packet_centered_resolution_active": packet_centered_resolution_active,
        "continuity_subordinate": continuity_subordinate,
        "detour_mode": detour_mode,
        "heuristic_detour_mode": heuristic_detour_mode,
        "solver_backed_detour_mode": solver_backed_detour_mode,
        "solver_backed_packet_present": solver_backed_detour_mode,
        "solver_backed_packet_types": solver_backed_packet_types,
        "heuristic_detour_packet_present": bool(heuristic_detour_packet_types),
        "heuristic_detour_packet_types": heuristic_detour_packet_types,
        "fallback_risk_flags": [f for f in risk_flags if "detour" in f or "solver" in f],
        "risk_flags": risk_flags,
    }


def _build_waste_dominance_metrics(
    demand: Dict[str, Any],
    supply_rows: List[Dict[str, Any]],
    prompt_surface: Dict[str, Any],
    waste: Dict[str, Any],
) -> Dict[str, Any]:
    total_projected = _as_int(
        waste.get("total_dynamic_chars"),
        sum(_as_int(r.get("payload_len"), 0) for r in supply_rows if r.get("selected"))
    )
    total_prompt = _as_int(prompt_surface.get("estimated_total_prompt_len"), 0)

    ranked: List[Dict[str, Any]] = []
    selected_rows = [r for r in supply_rows if r.get("selected")]
    selected_rows.sort(key=lambda r: _as_int(r.get("payload_len"), 0), reverse=True)

    for idx, r in enumerate(selected_rows, start=1):
        payload_len = _as_int(r.get("payload_len"), 0)
        ranked.append({
            "rank": idx,
            "channel": r.get("channel_name"),
            "payload_len": payload_len,
            "pct_of_projected_body": round((payload_len / total_projected) * 100.0, 1) if total_projected > 0 else 0.0,
            "pct_of_total_prompt": round((payload_len / total_prompt) * 100.0, 1) if total_prompt > 0 else 0.0,
        })

    warnings: List[str] = []

    channel_len = {r.get("channel_name"): _as_int(r.get("payload_len"), 0) for r in selected_rows}

    if demand.get("category") == "SOLVING_SETUP":
        setup_len = channel_len.get("SETUP_REPLY_PACKET", 0)
        continuity_len = channel_len.get("CONTINUITY_SHORT", 0)

        if setup_len <= 0:
            warnings.append("critical: SETUP_REPLY_PACKET missing on setup turn")
        if setup_len > 0 and continuity_len > int(setup_len * 0.35):
            warnings.append("warning: CONTINUITY_SHORT exceeds 35% of SETUP_REPLY_PACKET on setup turn")

    if demand.get("category") == "SOLVING_CONFRONTATION":
        confrontation_len = channel_len.get("CONFRONTATION_REPLY_PACKET", 0)
        continuity_len = channel_len.get("CONTINUITY_SHORT", 0)

        if confrontation_len <= 0:
            warnings.append("critical: CONFRONTATION_REPLY_PACKET missing on confrontation turn")
        if confrontation_len > 0 and continuity_len > int(confrontation_len * 0.35):
            warnings.append("warning: CONTINUITY_SHORT exceeds 35% of CONFRONTATION_REPLY_PACKET on confrontation turn")

    if demand.get("category") == "SOLVING_RESOLUTION":
        resolution_len = channel_len.get("RESOLUTION_REPLY_PACKET", 0)
        continuity_len = channel_len.get("CONTINUITY_SHORT", 0)

        if resolution_len <= 0:
            warnings.append("critical: RESOLUTION_REPLY_PACKET missing on resolution turn")
        if resolution_len > 0 and continuity_len > int(resolution_len * 0.35):
            warnings.append("warning: CONTINUITY_SHORT exceeds 35% of RESOLUTION_REPLY_PACKET on resolution turn")

    if channel_len.get("HANDOVER_NOTE_MINI", 0) > 250:
        warnings.append("warning: HANDOVER_NOTE_MINI exceeds 250 chars")

    if channel_len.get("GLOSSARY_MINI", 0) > 350:
        warnings.append("warning: GLOSSARY_MINI exceeds 350 chars")

    return {
        "projected_body_total_len": total_projected,
        "estimated_total_prompt_len": total_prompt,
        "ranked_channels": ranked,
        "warnings": warnings,
    }


def _infer_developer_prompt_variant(
    demand: Dict[str, Any],
    plan: Dict[str, Any],
) -> str:
    category = demand.get("category") or plan.get("demand_category")
    if category == "ONBOARDING_OPENING":
        return "composeTick2OnboardingDeveloperPrompt"
    if category == "CONFIRMING_VALIDATION_SUMMARY":
        return "composeTick2ConfirmingDeveloperPrompt"
    if category == "SOLVING_SETUP":
        return "composeTick2SolvingSetupDeveloperPrompt"
    if category == "SOLVING_CONFRONTATION":
        return "composeTick2SolvingConfrontationDeveloperPrompt"
    if category == "SOLVING_RESOLUTION":
        return "composeTick2SolvingResolutionDeveloperPrompt"
    if category == "REPAIR_CONTRADICTION":
        return "composeTick2RepairDeveloperPrompt"
    return "composeTick2DeveloperPrompt"


def _build_confrontation_scorecard(
    turn_header: Dict[str, Any],
    demand: Dict[str, Any],
    plan: Dict[str, Any],
    supply_rows: List[Dict[str, Any]],
    verdict: Dict[str, Any],
    dominance: Dict[str, Any],
    confrontation_cov: Dict[str, Any],
    waste: Dict[str, Any],
) -> Dict[str, Any]:
    if (turn_header.get("demand_category") or demand.get("category")) != "SOLVING_CONFRONTATION":
        return {}

    selected_channels = set(plan.get("selected_channels") or [])
    risk_flags = set(verdict.get("risk_flags") or [])
    waste_warnings = [w for w in (waste.get("warnings") or []) if isinstance(w, str)]

    legacy_story_selected = "CONFRONTATION_STORY_SLICE" in selected_channels
    legacy_step_selected = "CONFRONTATION_STEP_SLICE" in selected_channels

    packet_selected = bool(confrontation_cov.get("confrontation_packet_selected"))
    packet_projected = bool(confrontation_cov.get("confrontation_packet_projected"))
    proof_profile_present = bool(
        confrontation_cov.get("confrontation_proof_profile")
        or turn_header.get("confrontation_proof_profile")
        or demand.get("confrontation_proof_profile")
    )

    bounded_rows = _as_int(confrontation_cov.get("confrontation_packet_bounded_proof_row_count"), 0)
    raw_rows = _as_int(confrontation_cov.get("confrontation_packet_raw_proof_row_count"), 0)
    row_limit = _as_int(confrontation_cov.get("confrontation_packet_proof_row_limit"), 0)

    checks: List[Dict[str, Any]] = []

    def add_check(
        name: str,
        ok: bool,
        severity: str = "fail",
        observed: Any = None,
        expected: Any = None,
        note: Optional[str] = None,
    ) -> None:
        checks.append({
            "name": name,
            "ok": bool(ok),
            "severity": "ok" if ok else severity,
            "observed": observed,
            "expected": expected,
            "note": note,
        })

    add_check(
        name="packet_centered_confrontation_active",
        ok=bool(verdict.get("packet_centered_confrontation_active") is True),
        severity="fail",
        observed=verdict.get("packet_centered_confrontation_active"),
        expected=True,
        note="Confrontation should be packet-first, not slice-first.",
    )

    add_check(
        name="confrontation_packet_selected",
        ok=packet_selected,
        severity="fail",
        observed=packet_selected,
        expected=True,
    )

    add_check(
        name="confrontation_packet_projected",
        ok=packet_projected,
        severity="fail",
        observed=packet_projected,
        expected=True,
    )

    add_check(
        name="proof_profile_present",
        ok=proof_profile_present,
        severity="fail",
        observed=(
            confrontation_cov.get("confrontation_proof_profile")
            or turn_header.get("confrontation_proof_profile")
            or demand.get("confrontation_proof_profile")
        ),
        expected="non-empty proof profile",
    )

    add_check(
        name="packet_has_target",
        ok=bool(confrontation_cov.get("confrontation_packet_has_target")),
        severity="fail",
        observed=confrontation_cov.get("confrontation_packet_has_target"),
        expected=True,
    )

    add_check(
        name="packet_has_trigger_reference",
        ok=bool(confrontation_cov.get("confrontation_packet_has_trigger_reference")),
        severity="fail",
        observed=confrontation_cov.get("confrontation_packet_has_trigger_reference"),
        expected=True,
    )

    add_check(
        name="packet_has_trigger_effect",
        ok=bool(confrontation_cov.get("confrontation_packet_has_trigger_effect")),
        severity="warn",
        observed=confrontation_cov.get("confrontation_packet_has_trigger_effect"),
        expected=True,
        note="Some flows may stay brief, but the trigger effect should usually be explicit.",
    )

    add_check(
        name="packet_has_collapse",
        ok=bool(confrontation_cov.get("confrontation_packet_has_collapse")),
        severity="fail",
        observed=confrontation_cov.get("confrontation_packet_has_collapse"),
        expected=True,
    )

    add_check(
        name="packet_has_pre_commit_line",
        ok=bool(confrontation_cov.get("confrontation_packet_has_pre_commit_line")),
        severity="fail",
        observed=confrontation_cov.get("confrontation_packet_has_pre_commit_line"),
        expected=True,
    )

    add_check(
        name="packet_has_cta",
        ok=bool(confrontation_cov.get("confrontation_packet_has_cta")),
        severity="fail",
        observed=confrontation_cov.get("confrontation_packet_has_cta"),
        expected=True,
    )

    add_check(
        name="bounded_target_proof_rows_present",
        ok=(bounded_rows > 0),
        severity="fail",
        observed=bounded_rows,
        expected="> 0",
    )

    add_check(
        name="proof_row_limit_present",
        ok=(row_limit > 0),
        severity="warn",
        observed=row_limit,
        expected="> 0",
    )

    add_check(
        name="proof_row_bounding_active",
        ok=(raw_rows == 0 and bounded_rows > 0) or (raw_rows >= bounded_rows),
        severity="warn",
        observed={"raw": raw_rows, "bounded": bounded_rows},
        expected="raw >= bounded, with bounded rows present",
    )

    add_check(
        name="no_legacy_confrontation_story_slice",
        ok=not legacy_story_selected,
        severity="fail",
        observed=legacy_story_selected,
        expected=False,
    )

    add_check(
        name="no_legacy_confrontation_step_slice",
        ok=not legacy_step_selected,
        severity="fail",
        observed=legacy_step_selected,
        expected=False,
    )

    add_check(
        name="no_confrontation_starvation_flag",
        ok=("confrontation_starvation" not in risk_flags),
        severity="fail",
        observed=("confrontation_starvation" in risk_flags),
        expected=False,
    )

    add_check(
        name="continuity_subordinate_to_confrontation",
        ok=("continuity_dominates_confrontation" not in risk_flags),
        severity="warn",
        observed=("continuity_dominates_confrontation" in risk_flags),
        expected=False,
    )

    add_check(
        name="char_budget_not_exceeded",
        ok=not any(w.startswith("char_budget_exceeded:") for w in waste_warnings),
        severity="warn",
        observed=[w for w in waste_warnings if w.startswith("char_budget_exceeded:")],
        expected=[],
    )

    add_check(
        name="token_budget_not_exceeded",
        ok=not any(w.startswith("token_budget_exceeded:") for w in waste_warnings),
        severity="warn",
        observed=[w for w in waste_warnings if w.startswith("token_budget_exceeded:")],
        expected=[],
    )

    add_check(
        name="no_missing_collapse_truth_warning",
        ok=("confrontation_packet_missing_collapse_truth" not in waste_warnings),
        severity="fail",
        observed=("confrontation_packet_missing_collapse_truth" in waste_warnings),
        expected=False,
    )

    add_check(
        name="no_missing_pre_commit_boundary_warning",
        ok=("confrontation_packet_missing_pre_commit_boundary" not in waste_warnings),
        severity="fail",
        observed=("confrontation_packet_missing_pre_commit_boundary" in waste_warnings),
        expected=False,
    )

    add_check(
        name="no_missing_cta_warning",
        ok=("confrontation_packet_missing_cta" not in waste_warnings),
        severity="fail",
        observed=("confrontation_packet_missing_cta" in waste_warnings),
        expected=False,
    )

    add_check(
        name="no_missing_packet_channel_warning",
        ok=("missing_confrontation_reply_packet_channel" not in waste_warnings),
        severity="fail",
        observed=("missing_confrontation_reply_packet_channel" in waste_warnings),
        expected=False,
    )

    add_check(
        name="no_missing_packet_projection_warning",
        ok=("missing_confrontation_reply_packet_projection" not in waste_warnings),
        severity="fail",
        observed=("missing_confrontation_reply_packet_projection" in waste_warnings),
        expected=False,
    )

    add_check(
        name="rollout_mode_is_packet_first",
        ok=str(turn_header.get("rollout_mode") or "").strip() == "solving_confrontation_packet_v1",
        severity="warn",
        observed=turn_header.get("rollout_mode"),
        expected="solving_confrontation_packet_v1",
    )

    ok_n = sum(1 for c in checks if c["ok"] is True)
    warn_n = sum(1 for c in checks if c["ok"] is False and c["severity"] == "warn")
    fail_n = sum(1 for c in checks if c["ok"] is False and c["severity"] != "warn")
    total_n = len(checks)

    score_pct = round((ok_n / total_n) * 100.0, 1) if total_n > 0 else 0.0

    overall = "PASS"
    if fail_n > 0:
        overall = "FAIL"
    elif warn_n > 0:
        overall = "WARN"

    return {
        "overall": overall,
        "score_pct": score_pct,
        "summary": {
            "ok": ok_n,
            "warn": warn_n,
            "fail": fail_n,
            "total": total_n,
        },
        "proof_profile": (
            confrontation_cov.get("confrontation_proof_profile")
            or turn_header.get("confrontation_proof_profile")
            or demand.get("confrontation_proof_profile")
        ),
        "packet_stats": {
            "packet_len": confrontation_cov.get("confrontation_packet_len"),
            "raw_proof_row_count": raw_rows,
            "bounded_proof_row_count": bounded_rows,
            "proof_row_limit": row_limit,
            "proof_rows_truncated": confrontation_cov.get("confrontation_packet_proof_rows_truncated"),
            "overlay_variant": confrontation_cov.get("confrontation_overlay_variant"),
        },
        "checks": checks,
        "notes": [
            "This scorecard audits whether the confrontation turn is packet-first, proof-complete, bounded, and pre-commit.",
            "FAIL means a structural confrontation contract breach. WARN means soft quality drift or budget pressure.",
        ],
    }







def _build_resolution_scorecard(
    turn_header: Dict[str, Any],
    demand: Dict[str, Any],
    plan: Dict[str, Any],
    supply_rows: List[Dict[str, Any]],
    verdict: Dict[str, Any],
    dominance: Dict[str, Any],
    resolution_cov: Dict[str, Any],
    waste: Dict[str, Any],
) -> Dict[str, Any]:
    if (turn_header.get("demand_category") or demand.get("category")) != "SOLVING_RESOLUTION":
        return {}

    selected_channels = set(plan.get("selected_channels") or [])
    risk_flags = set(verdict.get("risk_flags") or [])
    waste_warnings = [w for w in (waste.get("warnings") or []) if isinstance(w, str)]

    setup_packet_selected = "SETUP_REPLY_PACKET" in selected_channels
    confrontation_packet_selected = "CONFRONTATION_REPLY_PACKET" in selected_channels
    resolution_packet_selected = "RESOLUTION_REPLY_PACKET" in selected_channels

    setup_story_selected = "SETUP_STORY_SLICE" in selected_channels
    setup_step_selected = "SETUP_STEP_SLICE" in selected_channels
    confrontation_story_selected = "CONFRONTATION_STORY_SLICE" in selected_channels
    confrontation_step_selected = "CONFRONTATION_STEP_SLICE" in selected_channels
    resolution_story_selected = "RESOLUTION_STORY_SLICE" in selected_channels
    resolution_step_selected = "RESOLUTION_STEP_SLICE" in selected_channels

    continuity_len = 0
    resolution_packet_len = 0
    for r in supply_rows:
        ch = r.get("channel_name")
        if ch == "CONTINUITY_SHORT":
            continuity_len = _as_int(r.get("payload_len"), 0)
        elif ch == "RESOLUTION_REPLY_PACKET":
            resolution_packet_len = _as_int(r.get("payload_len"), 0)

    checks: List[Dict[str, Any]] = []

    def add_check(
        name: str,
        ok: bool,
        severity: str = "fail",
        observed: Any = None,
        expected: Any = None,
        note: Optional[str] = None,
    ) -> None:
        checks.append({
            "name": name,
            "ok": bool(ok),
            "severity": "ok" if ok else severity,
            "observed": observed,
            "expected": expected,
            "note": note,
        })

    add_check(
        name="resolution_packet_selected",
        ok=resolution_packet_selected,
        severity="fail",
        observed=resolution_packet_selected,
        expected=True,
        note="Resolution should now be packet-first.",
    )

    add_check(
        name="resolution_packet_projected",
        ok=bool(resolution_cov.get("resolution_packet_projected")),
        severity="fail",
        observed=resolution_cov.get("resolution_packet_projected"),
        expected=True,
    )

    add_check(
        name="resolution_profile_present",
        ok=bool(
            resolution_cov.get("resolution_profile")
            or turn_header.get("resolution_profile")
        ),
        severity="warn",
        observed=(
            resolution_cov.get("resolution_profile")
            or turn_header.get("resolution_profile")
        ),
        expected="non-empty resolution profile",
    )

    add_check(
        name="packet_has_commit",
        ok=bool(resolution_cov.get("resolution_packet_has_commit")),
        severity="fail",
        observed=resolution_cov.get("resolution_packet_has_commit"),
        expected=True,
    )

    add_check(
        name="packet_has_recap",
        ok=bool(resolution_cov.get("resolution_packet_has_recap")),
        severity="fail",
        observed=resolution_cov.get("resolution_packet_has_recap"),
        expected=True,
    )

    add_check(
        name="packet_has_technique_contribution",
        ok=bool(resolution_cov.get("resolution_packet_has_technique_contribution")),
        severity="fail",
        observed=resolution_cov.get("resolution_packet_has_technique_contribution"),
        expected=True,
    )

    add_check(
        name="packet_has_final_forcing",
        ok=bool(resolution_cov.get("resolution_packet_has_final_forcing")),
        severity="fail",
        observed=resolution_cov.get("resolution_packet_has_final_forcing"),
        expected=True,
    )

    add_check(
        name="packet_has_honesty",
        ok=bool(resolution_cov.get("resolution_packet_has_honesty")),
        severity="warn",
        observed=resolution_cov.get("resolution_packet_has_honesty"),
        expected=True,
        note="Most mature two-layer resolution turns should carry an honesty line.",
    )

    add_check(
        name="packet_has_present_state_line",
        ok=bool(resolution_cov.get("resolution_packet_has_present_state_line")),
        severity="fail",
        observed=resolution_cov.get("resolution_packet_has_present_state_line"),
        expected=True,
    )

    add_check(
        name="packet_has_post_commit_summary",
        ok=bool(resolution_cov.get("resolution_packet_has_post_commit_summary")),
        severity="warn",
        observed=resolution_cov.get("resolution_packet_has_post_commit_summary"),
        expected=True,
    )

    add_check(
        name="packet_has_cta",
        ok=bool(resolution_cov.get("resolution_packet_has_cta")),
        severity="fail",
        observed=resolution_cov.get("resolution_packet_has_cta"),
        expected=True,
    )

    add_check(
        name="recap_max_beats_present",
        ok=_as_int(resolution_cov.get("resolution_packet_recap_max_beats"), 0) > 0,
        severity="warn",
        observed=resolution_cov.get("resolution_packet_recap_max_beats"),
        expected="> 0",
    )

    add_check(
        name="compact_mode_present",
        ok=bool(resolution_cov.get("resolution_packet_compact_mode")),
        severity="warn",
        observed=resolution_cov.get("resolution_packet_compact_mode"),
        expected="non-empty compact mode",
    )

    add_check(
        name="no_setup_reply_packet_selected",
        ok=not setup_packet_selected,
        severity="fail",
        observed=setup_packet_selected,
        expected=False,
    )

    add_check(
        name="no_confrontation_reply_packet_selected",
        ok=not confrontation_packet_selected,
        severity="fail",
        observed=confrontation_packet_selected,
        expected=False,
    )

    add_check(
        name="no_setup_story_slice_selected",
        ok=not setup_story_selected,
        severity="fail",
        observed=setup_story_selected,
        expected=False,
    )

    add_check(
        name="no_setup_step_slice_selected",
        ok=not setup_step_selected,
        severity="fail",
        observed=setup_step_selected,
        expected=False,
    )

    add_check(
        name="no_confrontation_story_slice_selected",
        ok=not confrontation_story_selected,
        severity="fail",
        observed=confrontation_story_selected,
        expected=False,
    )

    add_check(
        name="no_confrontation_step_slice_selected",
        ok=not confrontation_step_selected,
        severity="fail",
        observed=confrontation_step_selected,
        expected=False,
    )

    add_check(
        name="no_legacy_resolution_story_slice_selected",
        ok=not resolution_story_selected,
        severity="fail",
        observed=resolution_story_selected,
        expected=False,
    )

    add_check(
        name="no_legacy_resolution_step_slice_selected",
        ok=not resolution_step_selected,
        severity="fail",
        observed=resolution_step_selected,
        expected=False,
    )

    continuity_subordinate = (
        continuity_len <= int(resolution_packet_len * 0.35)
    ) if resolution_packet_len > 0 else None

    add_check(
        name="continuity_subordinate_to_resolution",
        ok=(continuity_subordinate is not False),
        severity="warn",
        observed=continuity_subordinate,
        expected=True,
    )

    add_check(
        name="required_channels_satisfied",
        ok=bool(verdict.get("required_channels_satisfied") is True),
        severity="fail",
        observed=verdict.get("required_channels_satisfied"),
        expected=True,
    )

    add_check(
        name="no_forbidden_channel_leak",
        ok=not bool(verdict.get("forbidden_channels_leaked")),
        severity="fail",
        observed=verdict.get("forbidden_channels_leaked"),
        expected=[],
    )

    add_check(
        name="no_selected_empty_channels",
        ok=not bool(verdict.get("selected_but_empty_channels")),
        severity="warn",
        observed=verdict.get("selected_but_empty_channels"),
        expected=[],
    )

    add_check(
        name="char_budget_not_exceeded",
        ok=not any(w.startswith("char_budget_exceeded:") for w in waste_warnings),
        severity="warn",
        observed=[w for w in waste_warnings if w.startswith("char_budget_exceeded:")],
        expected=[],
    )

    add_check(
        name="token_budget_not_exceeded",
        ok=not any(w.startswith("token_budget_exceeded:") for w in waste_warnings),
        severity="warn",
        observed=[w for w in waste_warnings if w.startswith("token_budget_exceeded:")],
        expected=[],
    )

    add_check(
        name="no_missing_packet_channel_warning",
        ok=("missing_resolution_reply_packet_channel" not in waste_warnings),
        severity="fail",
        observed=("missing_resolution_reply_packet_channel" in waste_warnings),
        expected=False,
    )

    add_check(
        name="no_missing_packet_projection_warning",
        ok=("missing_resolution_reply_packet_projection" not in waste_warnings),
        severity="fail",
        observed=("missing_resolution_reply_packet_projection" in waste_warnings),
        expected=False,
    )

    add_check(
        name="no_missing_recap_warning",
        ok=("resolution_packet_missing_recap" not in waste_warnings),
        severity="fail",
        observed=("resolution_packet_missing_recap" in waste_warnings),
        expected=False,
    )

    add_check(
        name="no_missing_technique_contribution_warning",
        ok=("resolution_packet_missing_technique_contribution" not in waste_warnings),
        severity="fail",
        observed=("resolution_packet_missing_technique_contribution" in waste_warnings),
        expected=False,
    )

    add_check(
        name="no_missing_final_forcing_warning",
        ok=("resolution_packet_missing_final_forcing" not in waste_warnings),
        severity="fail",
        observed=("resolution_packet_missing_final_forcing" in waste_warnings),
        expected=False,
    )

    add_check(
        name="no_missing_present_state_line_warning",
        ok=("resolution_packet_missing_present_state_line" not in waste_warnings),
        severity="fail",
        observed=("resolution_packet_missing_present_state_line" in waste_warnings),
        expected=False,
    )

    add_check(
        name="no_missing_post_commit_summary_warning",
        ok=("resolution_packet_missing_post_commit_summary" not in waste_warnings),
        severity="warn",
        observed=("resolution_packet_missing_post_commit_summary" in waste_warnings),
        expected=False,
    )

    add_check(
        name="no_missing_cta_warning",
        ok=("resolution_packet_missing_cta" not in waste_warnings),
        severity="fail",
        observed=("resolution_packet_missing_cta" in waste_warnings),
        expected=False,
    )

    add_check(
        name="rollout_mode_is_packet_first",
        ok=str(turn_header.get("rollout_mode") or "").strip() == "solving_resolution_packet_v1",
        severity="warn",
        observed=turn_header.get("rollout_mode"),
        expected="solving_resolution_packet_v1",
    )

    add_check(
        name="resolution_coverage_event_present",
        ok=bool(resolution_cov),
        severity="fail",
        observed=bool(resolution_cov),
        expected=True,
    )

    ok_n = sum(1 for c in checks if c["ok"] is True)
    warn_n = sum(1 for c in checks if c["ok"] is False and c["severity"] == "warn")
    fail_n = sum(1 for c in checks if c["ok"] is False and c["severity"] != "warn")
    total_n = len(checks)

    score_pct = round((ok_n / total_n) * 100.0, 1) if total_n > 0 else 0.0

    overall = "PASS"
    if fail_n > 0:
        overall = "FAIL"
    elif warn_n > 0:
        overall = "WARN"

    return {
        "overall": overall,
        "score_pct": score_pct,
        "summary": {
            "ok": ok_n,
            "warn": warn_n,
            "fail": fail_n,
            "total": total_n,
        },
        "packet_stats": {
            "resolution_packet_len": resolution_packet_len,
            "continuity_len": continuity_len,
            "coverage_present": bool(resolution_cov),
            "recap_max_beats": resolution_cov.get("resolution_packet_recap_max_beats"),
            "compact_mode": resolution_cov.get("resolution_packet_compact_mode"),
            "overlay_variant": resolution_cov.get("resolution_overlay_variant"),
        },
        "checks": checks,
        "notes": [
            "This scorecard audits whether the resolution turn is packet-first, commit-authorized, compact, present-state, and next-step ready.",
            "FAIL means structural stage leakage or missing core resolution packet truth. WARN means soft drift or budget pressure.",
        ],
    }




def extract_demand_supply_turn_report(turn_events: List[Event]) -> Dict[str, Any]:
    rr = extract_reply_request_details(turn_events)
    rr_turn_ctx = rr.get("turn_ctx") or {}
    reply_block = extract_model_call_block(turn_events, channel="REPLY", tick_id=2)

    demand = _extract_reply_demand_resolved(turn_events)
    plan = _extract_reply_assembly_plan(turn_events)
    projected = _extract_reply_projected_channels(turn_events)
    waste = _extract_reply_waste_audit(turn_events)
    setup_coverage = _extract_setup_reply_coverage(turn_events)
    confrontation_coverage = _extract_confrontation_reply_coverage(turn_events)
    resolution_coverage = _extract_resolution_reply_coverage(turn_events)
    prompt_surface = _extract_prompt_surface_from_reply_block(reply_block)

    binder_ev = _pick_last(turn_events, "INTENT_BINDER_V1")
    binder_payload = _event_payload_or_raw(binder_ev)

    projected_facts_ev = _pick_last(turn_events, "REPLY_PROJECTED_FACTS_V1")
    projected_facts_payload = _event_payload_or_raw(projected_facts_ev)

    detour_launched_ev = _pick_last(turn_events, "DETOUR_SOLVER_QUERY_LAUNCHED_V1")
    detour_succeeded_ev = _pick_last(turn_events, "DETOUR_SOLVER_QUERY_SUCCEEDED_V1")
    detour_failed_ev = _pick_last(turn_events, "DETOUR_SOLVER_QUERY_FAILED_V1")
    detour_reply_build_ev = _pick_last(turn_events, "DETOUR_SOLVER_REPLY_BUILD_V1")
    evidence_manifest_ev = _pick_last(turn_events, "TICK2_EVIDENCE_MANIFEST")

    evidence_payload = _event_payload_or_raw(evidence_manifest_ev)
    detour_launched_payload = _event_payload_or_raw(detour_launched_ev)

    def _none_if_nullish(v: Any) -> Any:
        if isinstance(v, str) and v.strip().lower() in {"", "null", "none"}:
            return None
        return v

    turn_id = (
        rr_turn_ctx.get("turn_id")
        or demand.get("turn_id")
        or plan.get("turn_id")
        or projected.get("turn_id")
        or waste.get("turn_id")
        or binder_payload.get("turn_id")
    )

    story_stage = (
        rr_turn_ctx.get("story", {}).get("stage")
        if isinstance(rr_turn_ctx.get("story"), dict)
        else demand.get("story_stage")
    ) or binder_payload.get("story_stage")

    turn_header = {
        "turn_id": turn_id,
        "tick_id": 2,
        "phase": rr_turn_ctx.get("phase") or demand.get("phase"),
        "story_stage": story_stage,
        "demand_category": demand.get("category") or plan.get("demand_category"),
        "setup_profile": (
            setup_coverage.get("setup_profile")
            or demand.get("setup_profile")
        ),
        "confrontation_proof_profile": (
            confrontation_coverage.get("confrontation_proof_profile")
            or demand.get("confrontation_proof_profile")
        ),
        "resolution_profile": (
            resolution_coverage.get("resolution_profile")
        ),
        "rollout_mode": plan.get("rollout_mode") or waste.get("rollout_mode"),
        "pending_before": rr_turn_ctx.get("pending_before"),
        "pending_after": rr_turn_ctx.get("pending_after"),
        "decision_kind": (rr.get("decision") or {}).get("decision_kind"),
        "user_text": rr_turn_ctx.get("user_text"),
    }

    supply_rows = _build_selected_supply_summary(
        demand=demand,
        plan=plan,
        projected=projected,
        waste=waste,
    )

    projected_order = [
        c.get("channel")
        for c in (projected.get("channels") or [])
        if isinstance(c, dict) and c.get("channel")
    ]

    prompt_surface = dict(prompt_surface)
    prompt_surface["selected_prompt_modules"] = plan.get("selected_prompt_modules") or []
    prompt_surface["developer_prompt_variant"] = _infer_developer_prompt_variant(demand, plan)
    prompt_surface["projected_user_channels_order"] = projected_order
    prompt_surface["projected_user_body_len"] = waste.get("total_dynamic_chars")
    prompt_surface["detour_mode"] = projected_facts_payload.get("detour_mode")
    prompt_surface["solver_backed_detour_mode"] = projected_facts_payload.get("solver_backed_detour_mode")
    prompt_surface["projected_fact_types"] = projected_facts_payload.get("fact_types")
    prompt_surface["projected_fact_count"] = projected_facts_payload.get("fact_count")

    payload_cards: List[Dict[str, Any]] = []
    for row in supply_rows:
        if not (row.get("selected") or row.get("projected")):
            continue
        if _as_int(row.get("payload_len"), 0) <= 0:
            continue
        payload_cards.append({
            "channel": row.get("channel_name"),
            "payload_len": row.get("payload_len"),
            "payload_sha12": row.get("payload_sha12"),
            "receipt": _receipt_for_channel(row.get("channel_name") or "", row.get("payload")),
            "payload": row.get("payload"),
        })

    selected_prompt_modules = plan.get("selected_prompt_modules") or []
    demand_category = demand.get("category") or plan.get("demand_category")

    if demand_category == "SOLVING_SETUP":
        doctrine_module = None
        if "SETUP_LENS_FIRST_RULES" in selected_prompt_modules:
            doctrine_module = "SETUP_LENS_FIRST_RULES"
        elif "SETUP_PATTERN_FIRST_RULES" in selected_prompt_modules:
            doctrine_module = "SETUP_PATTERN_FIRST_RULES"

        system_prompt_order = [
            "BASE_PERSONA",
            "BASE_JSON_OUTPUT",
            "NO_INVENTION_RULES",
            "GRID_TRUTH_RULES",
        ]
        if doctrine_module:
            system_prompt_order.append(doctrine_module)
        system_prompt_order.append("CTA_ENDING_RULES")
        system_prompt_order.append("strictSolvingSetupAppendixV1()")
    else:
        system_prompt_order = list(selected_prompt_modules)

    projected_payloads_in_order: List[Dict[str, Any]] = []
    payload_by_channel = {c.get("channel"): c for c in payload_cards if isinstance(c, dict) and c.get("channel")}

    seen_channels: List[str] = []
    for ch in projected_order + [c.get("channel") for c in payload_cards if c.get("channel")]:
        if ch and ch not in seen_channels:
            seen_channels.append(ch)

    for ch in seen_channels:
        c = payload_by_channel.get(ch) or {}
        projected_payloads_in_order.append({
            "channel": ch,
            "payload_len": c.get("payload_len"),
            "payload_sha12": c.get("payload_sha12"),
            "payload": c.get("payload"),
        })

    what_llm_sees = {
        "system_prompt": {
            "assembled_order": system_prompt_order,
            "text": prompt_surface.get("system_prompt_text"),
            "len": prompt_surface.get("system_prompt_len"),
        },
        "developer_prompt": {
            "variant": prompt_surface.get("developer_prompt_variant"),
            "text": prompt_surface.get("developer_prompt_text"),
            "len": prompt_surface.get("developer_prompt_len"),
        },
        "user_side_projected_prompt": {
            "projected_channels_order": projected_order,
            "assembled_user_message_text": prompt_surface.get("user_message_text"),
            "len": prompt_surface.get("user_message_len"),
            "projected_payloads": projected_payloads_in_order,
        },
    }

    verdict = _build_reality_vs_contract_verdict(demand, plan, supply_rows)
    dominance = _build_waste_dominance_metrics(
        demand=demand,
        supply_rows=supply_rows,
        prompt_surface=prompt_surface,
        waste=waste,
    )

    confrontation_scorecard = _build_confrontation_scorecard(
        turn_header=turn_header,
        demand=demand,
        plan=plan,
        supply_rows=supply_rows,
        verdict=verdict,
        dominance=dominance,
        confrontation_cov=confrontation_coverage,
        waste=waste,
    )

    resolution_scorecard = _build_resolution_scorecard(
        turn_header=turn_header,
        demand=demand,
        plan=plan,
        supply_rows=supply_rows,
        verdict=verdict,
        dominance=dominance,
        resolution_cov=resolution_coverage,
        waste=waste,
    )

    detour_header = {
        "detour_question_class": _none_if_nullish(binder_payload.get("detour_question_class")),
        "detour_kind": _none_if_nullish(binder_payload.get("detour_kind")),
        "detour_mode": bool(projected_facts_payload.get("detour_mode")),
        "solver_backed_detour_mode": bool(projected_facts_payload.get("solver_backed_detour_mode")),
        "heuristic_detour_mode": bool(projected_facts_payload.get("heuristic_detour_mode")),
        "solver_query_launched": detour_launched_ev is not None,
        "solver_query_succeeded": detour_succeeded_ev is not None,
        "solver_query_failed": detour_failed_ev is not None,
        "solver_query_kind": _none_if_nullish(detour_launched_payload.get("query_kind")),
        "evidence_manifest_kind": _none_if_nullish(evidence_payload.get("query_kind")),
        "projected_fact_types": projected_facts_payload.get("fact_types") or [],
    }

    detour_demand_summary = {
        "top_intent_type": _none_if_nullish(binder_payload.get("top_intent_type")),
        "detour_question_class": _none_if_nullish(binder_payload.get("detour_question_class")),
        "requested_cell": _none_if_nullish(binder_payload.get("requested_cell")),
        "requested_digit": _none_if_nullish(binder_payload.get("requested_digit")),
        "requested_technique": _none_if_nullish(binder_payload.get("requested_technique")),
        "route_following": bool(binder_payload.get("route_following")),
        "route_control": bool(binder_payload.get("route_control")),
        "detour_kind": _none_if_nullish(binder_payload.get("detour_kind")),
    }

    detour_contract_view = {
        "selected_prompt_modules": plan.get("selected_prompt_modules") or [],
        "selected_channels": plan.get("selected_channels") or [],
        "forbidden_channels": plan.get("forbidden_channels") or [],
        "rollout_mode": plan.get("rollout_mode"),
        "solver_backed_packets_projected": projected_facts_payload.get("solver_backed_packet_types") or [],
        "heuristic_detour_packets_projected": projected_facts_payload.get("heuristic_detour_packet_types") or [],
    }

    detour_visual_mode = {
        "required_overlay_frame_ids": projected_facts_payload.get("required_overlay_frame_ids") or [],
        "applied_overlay_frame_ids": projected_facts_payload.get("applied_overlay_frame_ids") or [],
        "derived_overlay_mode": projected_facts_payload.get("derived_overlay_mode"),
        "focus_atom_index": projected_facts_payload.get("focus_atom_index"),
        "story_stage_when_paused": projected_facts_payload.get("story_stage_when_paused"),
    }

    solver_query_timeline = {
        "launched": (
            {
                "query_kind": detour_launched_payload.get("query_kind"),
                "focus_kind": detour_launched_payload.get("focus_kind"),
                "cell": detour_launched_payload.get("cell"),
                "digit": detour_launched_payload.get("digit"),
                "technique": detour_launched_payload.get("technique"),
            }
            if detour_launched_ev
            else None
        ),
        "succeeded": (
            _event_payload_or_raw(detour_succeeded_ev)
            if detour_succeeded_ev
            else None
        ),
        "failed": (
            _event_payload_or_raw(detour_failed_ev)
            if detour_failed_ev
            else None
        ),
        "reply_build": (
            _event_payload_or_raw(detour_reply_build_ev)
            if detour_reply_build_ev
            else None
        ),
    }

    short_conclusion_lines: List[str] = []
    short_conclusion_lines.append(
        f"The demand was {turn_header.get('demand_category') or 'UNKNOWN'}"
        + (f" / {turn_header.get('setup_profile')}" if turn_header.get("setup_profile") else "")
        + (f" / {turn_header.get('confrontation_proof_profile')}" if turn_header.get("confrontation_proof_profile") else "")
        + (f" / {turn_header.get('resolution_profile')}" if turn_header.get("resolution_profile") else "")
        + "."
    )
    if verdict.get("required_channels_satisfied") is True:
        short_conclusion_lines.append("Required channels were satisfied.")
    else:
        missing = verdict.get("missing_required_channels") or []
        short_conclusion_lines.append(f"Required channels were missing: {missing}.")
    if waste.get("waste_ratio") is not None:
        short_conclusion_lines.append(f"Waste ratio was {waste.get('waste_ratio')}.")
    if detour_header.get("detour_question_class"):
        short_conclusion_lines.append(
            f"Detour mode was active for {detour_header.get('detour_question_class')}."
        )

    conversation_transcript = extract_turn_conversation_transcript(turn_events)

    return {
        "turn_header": turn_header,
        "conversation_transcript": conversation_transcript,
        "demand_summary": demand,
        "contract_view": plan,
        "selected_supply_summary": supply_rows,
        "payload_receipt_cards": payload_cards,
        "what_llm_sees": what_llm_sees,
        "prompt_surface": prompt_surface,
        "reply_waste_audit": waste,
        "reality_vs_contract_verdict": verdict,
        "waste_dominance_metrics": dominance,
        "setup_reply_coverage": setup_coverage,
        "confrontation_reply_coverage": confrontation_coverage,
        "resolution_reply_coverage": resolution_coverage,
        "confrontation_scorecard": confrontation_scorecard,
        "resolution_scorecard": resolution_scorecard,
        "detour_header": detour_header,
        "detour_demand_summary": detour_demand_summary,
        "detour_contract_view": detour_contract_view,
        "detour_visual_mode": detour_visual_mode,
        "solver_query_timeline": solver_query_timeline,
        "short_conclusion": " ".join(short_conclusion_lines),
    }


# -------------------------
# Render: Markdown
# -------------------------


def render_md_report(
    session_id: str,
    start_ts: Optional[str],
    end_ts: Optional[str],
    transcript: List[Dict[str, Any]],
    per_turn: List[Dict[str, Any]],
) -> str:
    lines: List[str] = []

    lines.append("# Sudoku Companion — Session Report (Telemetry-derived)")
    lines.append("")
    lines.append(f"- session_id: `{session_id}`")
    lines.append(f"- start: `{start_ts or ''}`")
    lines.append(f"- end: `{end_ts or ''}`")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1) Spoken transcript (User + Assistant, start → end)")
    lines.append("")

    for it in transcript:
        ts = it.get("ts_iso") or ""
        speaker = it.get("speaker") or ""
        text = it.get("text") or ""
        prefix = f"{ts} " if ts else ""
        lines.append(f"{prefix}{speaker}: {text}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 2) Turn-by-turn reconstruction (truth-first)")
    lines.append("")

    def _render_preview_box(title: str, box: Dict[str, Any]) -> None:
        lines.append(f"**{_md_escape(title)}**")
        if "parsed" in box:
            lines.append("```json")
            lines.append(_json_pretty(box["parsed"]))
            lines.append("```")
        else:
            raw = box.get("raw") or ""
            lines.append("```")
            lines.append(_md_escape(raw))
            lines.append("```")
        lines.append("")

    def _render_profile_snapshot_compact(snap: Dict[str, Any]) -> None:
        tlen = snap.get("user_tally_len")
        alen = snap.get("assistant_tally_len")
        rmlen = snap.get("relationship_memory_len")
        rlen = snap.get("recent_turns_len")
        suffix = f"user_len={tlen} assistant_len={alen} relationship_len={rmlen} recent_len={rlen}"

        lines.append(f"<details><summary><b>PROFILE_SNAPSHOT_AFTER_TICK1</b> — {suffix}</summary>")
        lines.append("")
        lines.append("_Compact view (previews)_")
        lines.append("")

        _render_preview_box(
            "UserTally (preview)",
            snap.get("user_tally_preview_box") or {"raw": snap.get("user_tally_preview") or ""},
        )
        _render_preview_box(
            "AssistantTally (preview)",
            snap.get("assistant_tally_preview_box") or {"raw": snap.get("assistant_tally_preview") or ""},
        )
        _render_preview_box(
            "RelationshipMemory (preview)",
            snap.get("relationship_memory_preview_box") or {"raw": snap.get("relationship_memory_preview") or ""},
        )
        _render_preview_box(
            "Recent turns (preview)",
            snap.get("recent_turns_preview_box") or {"raw": snap.get("recent_turns_preview") or ""},
        )

        lines.append("</details>")
        lines.append("")

    def _render_grid_facts_snapshots(grid_pack: Dict[str, Any]) -> None:
        if not isinstance(grid_pack, dict):
            return

        stages = grid_pack.get("stages") or {}
        if not isinstance(stages, dict) or not stages:
            return

        lines.append("### Grid facts snapshots")
        lines.append("")

        warnings = grid_pack.get("warnings") or []
        if isinstance(warnings, list) and warnings:
            for w in warnings:
                if isinstance(w, str) and w.strip():
                    lines.append(f"- ⚠️ {w}")
            lines.append("")

        preferred_order = ["PRE_TICK1", "POST_TICK1", "POST_DECISION", "POST_APPLY"]
        ordered = [k for k in preferred_order if k in stages] + [k for k in stages.keys() if k not in preferred_order]

        for st in ordered:
            snap = stages.get(st) or {}
            curated = snap.get("curated")
            raw_preview = snap.get("raw_preview") or ""

            meta = f"turn={snap.get('turn_id')} tick={snap.get('tick_id')} req={snap.get('policy_req_seq')} seq={snap.get('seq')}"
            lines.append(f"<details><summary><b>{_md_escape(st)}</b> — {meta}</summary>")
            lines.append("")

            if isinstance(curated, dict):
                lines.append("```json")
                lines.append(_json_pretty(curated))
                lines.append("```")
            else:
                lines.append("```")
                lines.append(_md_escape(raw_preview or "grid facts snapshot present but not parsed as JSON"))
                lines.append("```")

            lines.append("</details>")
            lines.append("")

    def _render_solving_truth_section(
        solve_pack: Dict[str, Any],
        grid_facts_pack: Dict[str, Any],
        packet_ready_pack: Dict[str, Any],
    ) -> None:
        if not isinstance(solve_pack, dict) or not solve_pack:
            return

        meta = solve_pack.get("meta") or {}
        step_curated = solve_pack.get("curated")
        if not isinstance(step_curated, dict):
            return

        match = solve_step_grid_matches_displayed(grid_facts_pack, solve_pack)
        if match is not None:
            step_curated = dict(step_curated)
            step_curated["grid_before_matches_displayed81"] = bool(match)

        suffix = (
            f"turn={meta.get('turn_id')} tick={meta.get('tick_id')} "
            f"step_id={meta.get('step_id')} grid_hash12={meta.get('grid_hash12')} "
            f"sha12={meta.get('step_json_sha12')} len={meta.get('step_json_len')}"
        )

        lines.append("### Solving truth")
        lines.append("")

        lines.append(f"<details><summary><b>Engine step essentials</b> — SOLVE_STEP_JSON_READY — {suffix}</summary>")
        lines.append("")
        lines.append("```json")
        lines.append(_json_pretty(step_curated))
        lines.append("```")
        lines.append("</details>")
        lines.append("")

        if isinstance(packet_ready_pack, dict) and packet_ready_pack:
            meta2 = packet_ready_pack.get("meta") or {}
            suffix2 = (
                f"turn={meta2.get('turn_id')} tick={meta2.get('tick_id')} "
                f"step_id={meta2.get('step_id')} grid_hash12={meta2.get('grid_hash12')} "
                f"sha12={meta2.get('packet_payload_sha12')} len={meta2.get('packet_payload_len')}"
            )

            tech_info = packet_ready_pack.get("technique_info")
            narrative_truth_v2 = packet_ready_pack.get("narrative_truth_v2")
            narrative_truth_from_atoms = packet_ready_pack.get("narrative_truth_from_atoms")
            nav = packet_ready_pack.get("narrative_atoms_v1")

            if isinstance(narrative_truth_v2, dict) and narrative_truth_v2:
                story = _curate_narrative_truth_story_v2(
                    narrative_truth_v2,
                    nav if isinstance(nav, dict) else None,
                )

                lines.append(f"<details><summary><b>Solving story</b> — trigger → bridge → final resolution — {suffix2}</summary>")
                lines.append("")
                lines.append("```json")
                lines.append(_json_pretty(story))
                lines.append("```")
                lines.append("</details>")
                lines.append("")

            if isinstance(tech_info, dict) and tech_info:
                lines.append(f"<details><summary><b>Technique truth</b> — packet.step.technique_info — {suffix2}</summary>")
                lines.append("")
                lines.append("```json")
                lines.append(_json_pretty(tech_info))
                lines.append("```")
                lines.append("</details>")
                lines.append("")

            if isinstance(narrative_truth_v2, dict) and narrative_truth_v2:
                lines.append(f"<details><summary><b>Narrative truth story</b> — curated from packet.evidence.narrative_truth_v2 — {suffix2}</summary>")
                lines.append("")
                lines.append("```json")
                lines.append(_json_pretty(_curate_narrative_truth_story_v2(
                    narrative_truth_v2,
                    nav if isinstance(nav, dict) else None,
                )))
                lines.append("```")
                lines.append("</details>")
                lines.append("")

                lines.append(f"<details><summary><b>Narrative truth v2 (raw)</b> — packet.evidence.narrative_truth_v2 — {suffix2}</summary>")
                lines.append("")
                lines.append("```json")
                lines.append(_json_pretty(narrative_truth_v2))
                lines.append("```")
                lines.append("</details>")
                lines.append("")

            if isinstance(narrative_truth_from_atoms, dict) and narrative_truth_from_atoms:
                lines.append(f"<details><summary><b>Legacy narrative truth (derived from atoms)</b> — reconstructed audit view — {suffix2}</summary>")
                lines.append("")
                lines.append("```json")
                lines.append(_json_pretty(narrative_truth_from_atoms))
                lines.append("```")
                lines.append("</details>")
                lines.append("")

            if isinstance(nav, dict) and nav:
                consistency = packet_ready_pack.get("truth_atoms_consistency")
                if isinstance(consistency, dict) and consistency:
                    lines.append(f"<details><summary><b>Truth ↔ atoms consistency</b> — v2 truth vs atoms vs overlays — {suffix2}</summary>")
                    lines.append("")
                    lines.append("```json")
                    lines.append(_json_pretty(consistency))
                    lines.append("```")
                    lines.append("</details>")
                    lines.append("")

                lines.append(f"<details><summary><b>Narrative atoms (raw)</b> — packet.evidence.narrative_atoms_v1 — {suffix2}</summary>")
                lines.append("")
                lines.append("```json")
                lines.append(_json_pretty(nav))
                lines.append("```")
                lines.append("</details>")
                lines.append("")

                frames = packet_ready_pack.get("reconstructed_overlay_frames") or []
                if isinstance(frames, list) and frames:
                    lines.append(f"<details><summary><b>Derived overlay frames</b> — reconstructed from narrative atoms — {suffix2}</summary>")
                    lines.append("")
                    for f in frames:
                        fid = _safe_get_dict(f.get("meta")).get("frame_id")
                        lines.append(f"#### {fid}")
                        lines.append("```json")
                        lines.append(_json_pretty(f))
                        lines.append("```")
                        lines.append("")
                    lines.append("</details>")
                    lines.append("")

    def _render_contract_block(title: str, contract_obj: Dict[str, Any], meta_suffix: str = "") -> None:
        if not isinstance(contract_obj, dict) or not contract_obj:
            return
        meta = meta_suffix.strip()
        if meta:
            lines.append(f"<details><summary><b>{_md_escape(title)}</b> — {meta}</summary>")
        else:
            lines.append(f"<details><summary><b>{_md_escape(title)}</b></summary>")
        lines.append("")
        lines.append("```json")
        lines.append(_json_pretty(contract_obj))
        lines.append("```")
        lines.append("</details>")
        lines.append("")

    def _render_turn_context(turn_ctx_pack: Dict[str, Any]) -> None:
        if not isinstance(turn_ctx_pack, dict) or not turn_ctx_pack:
            return
        meta = turn_ctx_pack.get("meta") or {}
        curated = turn_ctx_pack.get("curated")
        parsed = turn_ctx_pack.get("parsed")

        suffix = f"turn={meta.get('turn_id')} tick={meta.get('tick_id')} req={meta.get('policy_req_seq')} sha12={meta.get('turn_ctx_sha12')}"
        lines.append(f"<details><summary><b>TURN_CONTEXT_V1</b> — {suffix}</summary>")
        lines.append("")

        if isinstance(curated, dict) and curated:
            lines.append("_Curated view_")
            lines.append("")
            lines.append("```json")
            lines.append(_json_pretty(curated))
            lines.append("```")
            lines.append("")

        if isinstance(parsed, dict) and parsed:
            lines.append("_Full payload_")
            lines.append("")
            lines.append("```json")
            lines.append(_json_pretty(parsed))
            lines.append("```")
        else:
            raw_preview = turn_ctx_pack.get("raw_preview") or ""
            lines.append("```")
            lines.append(_md_escape(raw_preview or "TurnContext present but not parsed as JSON"))
            lines.append("```")

        lines.append("</details>")
        lines.append("")

    def _render_agenda_audit_events(pack: Dict[str, Any]) -> None:
        if not isinstance(pack, dict) or not pack:
            return
        evs = pack.get("events") if isinstance(pack.get("events"), list) else []
        counts = pack.get("counts") if isinstance(pack.get("counts"), dict) else {}

        lines.append("### Agenda audit events")
        lines.append("")

        if counts:
            counts_str = ", ".join([f"{k}={counts.get(k)}" for k in sorted(counts.keys())])
            lines.append(f"- counts: {counts_str}")
            lines.append("")

        if not evs:
            lines.append("_No agenda audit events found in this turn._")
            lines.append("")
            return

        for it in evs:
            et = it.get("type")
            tick = it.get("tick_id")
            seq = it.get("seq")
            ts = it.get("ts_iso") or ""
            header = f"{et} (tick={tick}, seq={seq})"
            if ts:
                header += f" @ {ts}"
            lines.append(f"<details><summary><b>{_md_escape(header)}</b></summary>")
            lines.append("")
            lines.append("```json")
            lines.append(_json_pretty(it.get("payload") or {}))
            lines.append("```")
            lines.append("</details>")
            lines.append("")

    for turn in per_turn:
        tid = turn["turn_id"]
        lines.append(f"## Turn {tid}")
        lines.append("")

        ps = (turn.get("memory_snapshots") or {}).get("profile_snapshot_after_tick1") or {}
        if ps:
            lines.append("### Memory snapshots")
            lines.append("")
            _render_profile_snapshot_compact(ps)

        grid_pack = turn.get("grid_facts_snapshots") or {}
        if grid_pack:
            _render_grid_facts_snapshots(grid_pack)

        solve_pack = turn.get("solve_step_json_ready") or {}
        packet_ready = turn.get("solving_step_packet_ready") or {}
        _render_solving_truth_section(solve_pack, grid_pack, packet_ready)

        tc = turn.get("turn_context_v1") or {}
        if tc:
            lines.append("### Turn context")
            lines.append("")
            _render_turn_context(tc)

        agenda_audit = turn.get("agenda_audit_events") or {}
        _render_agenda_audit_events(agenda_audit)

        lines.append("### Contract coverage")
        for w in (turn.get("contract_warnings") or []):
            lines.append(f"- ⚠️ {w}")
        lines.append("")

        env = turn.get("tick1_intent_envelope_expected")
        if env:
            meta = f"turn={tid} tick=1 req={env.get('ids', {}).get('policy_req_seq')} | tag={env.get('tag')}"
            _render_contract_block("TICK1_INTENT_ENVELOPE (LLM → App)", env, meta)

        app_plan = turn.get("app_plan_expected")
        if app_plan:
            meta = f"turn={tid} tick=1 req={app_plan.get('ids', {}).get('policy_req_seq')} | tag={app_plan.get('tag')}"
            _render_contract_block("APP_PLAN_V1 (App planner output)", app_plan, meta)

        if turn.get("meaning_block"):
            b = turn["meaning_block"]
            ids = b.get("ids", {})
            req = b.get("req", {})
            res = b.get("res", {})
            user_text = turn.get("user_text") or ""

            lines.append("### MEANING call (tick 1)")
            lines.append(
                f"- ids: model_call_id={ids.get('model_call_id')} toolplan_id={ids.get('toolplan_id')} correlation_id={ids.get('correlation_id')} policy_req_seq={ids.get('policy_req_seq')}"
            )
            lines.append(
                f"- req: req_id={req.get('req_id')} payload_sha12={req.get('payload_sha12')} payload_len={req.get('payload_len')} payload_kind={req.get('payload_kind')}"
            )
            lines.append(
                f"- res: http_code={res.get('http_code')} dt_ms={res.get('dt_ms')} response_sha12={res.get('response_sha12')} response_len={res.get('response_len')} parse_ok={res.get('parse_ok')} parse_errors={res.get('parse_errors')} payload_kind={res.get('payload_kind')}"
            )
            lines.append(f"- user_text: {user_text}")
            lines.append("")

            payload_out = b.get("payload_out", {}).get("payload_text")
            if isinstance(payload_out, str) and payload_out.strip():
                lines.append("#### MODEL_PAYLOAD_OUT.payload")
                lines.append("```json")
                lines.append(_md_escape(payload_out))
                lines.append("```")
                lines.append("")

            payload_in = b.get("payload_in", {}).get("response_text")
            if isinstance(payload_in, str) and payload_in.strip():
                lines.append("#### MODEL_PAYLOAD_IN.response")
                lines.append("```json")
                lines.append(_md_escape(payload_in))
                lines.append("```")
                lines.append("")

        if turn.get("reply_block"):
            b = turn["reply_block"]
            ids = b.get("ids", {})
            req = b.get("req", {})
            res = b.get("res", {})

            decision_kind = turn.get("decision_kind")
            decision_summary = turn.get("decision_summary")

            lines.append("### REPLY call (tick 2)")
            lines.append(
                f"- ids: model_call_id={ids.get('model_call_id')} toolplan_id={ids.get('toolplan_id')} correlation_id={ids.get('correlation_id')} policy_req_seq={ids.get('policy_req_seq')}"
            )
            lines.append(
                f"- req: req_id={req.get('req_id')} payload_sha12={req.get('payload_sha12')} payload_len={req.get('payload_len')} payload_kind={req.get('payload_kind')}"
            )
            lines.append(
                f"- res: http_code={res.get('http_code')} dt_ms={res.get('dt_ms')} response_sha12={res.get('response_sha12')} response_len={res.get('response_len')} parse_ok={res.get('parse_ok')} parse_errors={res.get('parse_errors')} payload_kind={res.get('payload_kind')}"
            )
            lines.append(f"- decision_kind: {decision_kind} | decision_summary: {decision_summary}")

            tp = turn.get("toolplan_tools") or {}
            if tp:
                lines.append("- toolplan_tools:")
                lines.append(f"  - operation_tools: {tp.get('operation_tools', [])}")
                lines.append(f"  - control_tools: {tp.get('control_tools', [])}")
                note = tp.get("note") or ""
                srcp = tp.get("source_path")
                if note:
                    lines.append(f"  - note: {note}")
                if srcp:
                    lines.append(f"  - source_path: {srcp}")

            lines.append("")

            payload_out = b.get("payload_out", {}).get("payload_text")
            if isinstance(payload_out, str) and payload_out.strip():
                lines.append("#### MODEL_PAYLOAD_OUT.payload")
                lines.append("```json")
                lines.append(_md_escape(payload_out))
                lines.append("```")
                lines.append("")

            payload_in = b.get("payload_in", {}).get("response_text")
            if isinstance(payload_in, str) and payload_in.strip():
                lines.append("#### MODEL_PAYLOAD_IN.response")
                lines.append("```json")
                lines.append(_md_escape(payload_in))
                lines.append("```")
                lines.append("")

        t2 = turn.get("tick2_request_contract_expected")
        if t2:
            meta = f"turn={tid} tick=2 req={t2.get('ids', {}).get('policy_req_seq')} | tag={t2.get('tag')}"
            payload_only = t2.get("payload") if isinstance(t2.get("payload"), dict) else t2
            _render_contract_block("TICK2_REQUEST_CONTRACT (sent to LLM)", payload_only, meta)

    return "\n".join(lines).rstrip() + "\n"




def render_demand_supply_md_report(
    session_id: str,
    start_ts: Optional[str],
    end_ts: Optional[str],
    turns: List[Dict[str, Any]],
) -> str:
    lines: List[str] = []

    lines.append("# Sudoku Companion — Demand / Supply Report")
    lines.append("")
    lines.append(f"- session_id: `{session_id}`")
    lines.append(f"- start: `{start_ts or ''}`")
    lines.append(f"- end: `{end_ts or ''}`")
    lines.append("")
    lines.append("This report shows, per turn, the chain:")
    lines.append("")
    lines.append("**Demand → Contract → What the LLM sees → Prompt Surface**")
    lines.append("")
    lines.append("---")
    lines.append("")

    for turn in turns:
        hdr = turn.get("turn_header") or {}
        transcript = turn.get("conversation_transcript") or []
        demand = turn.get("demand_summary") or {}
        contract = turn.get("contract_view") or {}
        llm_view = turn.get("what_llm_sees") or {}
        prompt_surface = turn.get("prompt_surface") or {}
        verdict = turn.get("reality_vs_contract_verdict") or {}
        dominance = turn.get("waste_dominance_metrics") or {}
        setup_cov = turn.get("setup_reply_coverage") or {}
        confrontation_cov = turn.get("confrontation_reply_coverage") or {}
        resolution_cov = turn.get("resolution_reply_coverage") or {}
        confrontation_scorecard = turn.get("confrontation_scorecard") or {}
        resolution_scorecard = turn.get("resolution_scorecard") or {}
        waste = turn.get("reply_waste_audit") or {}
        detour_header = turn.get("detour_header") or {}
        detour_demand = turn.get("detour_demand_summary") or {}
        detour_contract = turn.get("detour_contract_view") or {}
        detour_visual = turn.get("detour_visual_mode") or {}
        solver_timeline = turn.get("solver_query_timeline") or {}

        tid = hdr.get("turn_id")
        title = (
            f"## Turn {tid} — "
            f"{hdr.get('demand_category') or 'UNKNOWN'}"
            + (f" / {hdr.get('setup_profile')}" if hdr.get("setup_profile") else "")
            + (f" / {hdr.get('confrontation_proof_profile')}" if hdr.get("confrontation_proof_profile") else "")
            + (f" / {hdr.get('resolution_profile')}" if hdr.get("resolution_profile") else "")
            + (f" — rollout={hdr.get('rollout_mode')}" if hdr.get("rollout_mode") else "")
        )
        lines.append(title)
        lines.append("")

        lines.append("### Turn header")
        lines.append("")
        lines.append("```json")
        lines.append(_json_pretty(hdr))
        lines.append("```")
        lines.append("")

        if transcript:
            lines.append("### Conversation transcript")
            lines.append("")
            for row in transcript:
                speaker = row.get("speaker") or ""
                text = row.get("text") or ""
                if speaker and text:
                    lines.append(f"{speaker}: {text}")
                    lines.append("")
        else:
            lines.append("### Conversation transcript")
            lines.append("")
            lines.append("_No turn-local conversation transcript was available._")
            lines.append("")

        if detour_header.get("detour_question_class") or detour_header.get("solver_backed_detour_mode") or detour_header.get("heuristic_detour_mode"):
            lines.append("### Detour header")
            lines.append("")
            lines.append("```json")
            lines.append(_json_pretty(detour_header))
            lines.append("```")
            lines.append("")

        lines.append("### Demand summary")
        lines.append("")
        lines.append("```json")
        lines.append(_json_pretty(demand))
        lines.append("```")
        lines.append("")

        if detour_demand.get("detour_question_class") or detour_demand.get("detour_kind"):
            lines.append("### Detour demand summary")
            lines.append("")
            lines.append("```json")
            lines.append(_json_pretty(detour_demand))
            lines.append("```")
            lines.append("")

        lines.append("### Contract view")
        lines.append("")
        lines.append("```json")
        lines.append(_json_pretty(contract))
        lines.append("```")
        lines.append("")

        if detour_contract.get("selected_channels") or detour_contract.get("solver_backed_packets_projected"):
            lines.append("### Detour contract view")
            lines.append("")
            lines.append("```json")
            lines.append(_json_pretty(detour_contract))
            lines.append("```")
            lines.append("")

        lines.append("### What the LLM sees")
        lines.append("")

        system_prompt = llm_view.get("system_prompt") or {}
        lines.append("#### System prompt")
        lines.append("")
        lines.append("```json")
        lines.append(_json_pretty({
            "assembled_order": system_prompt.get("assembled_order") or [],
            "len": system_prompt.get("len"),
        }))
        lines.append("```")
        lines.append("")
        if system_prompt.get("text"):
            lines.append("```text")
            lines.append(system_prompt.get("text") or "")
            lines.append("```")
            lines.append("")
        else:
            lines.append("_System prompt text was not available._")
            lines.append("")

        developer_prompt = llm_view.get("developer_prompt") or {}
        lines.append("#### Developer prompt")
        lines.append("")
        lines.append("```json")
        lines.append(_json_pretty({
            "variant": developer_prompt.get("variant"),
            "len": developer_prompt.get("len"),
        }))
        lines.append("```")
        lines.append("")
        if developer_prompt.get("text"):
            lines.append("```text")
            lines.append(developer_prompt.get("text") or "")
            lines.append("```")
            lines.append("")
        else:
            lines.append("_Developer prompt text was not available._")
            lines.append("")

        user_prompt = llm_view.get("user_side_projected_prompt") or {}
        lines.append("#### User-side projected prompt")
        lines.append("")
        lines.append("```json")
        lines.append(_json_pretty({
            "projected_channels_order": user_prompt.get("projected_channels_order") or [],
            "len": user_prompt.get("len"),
        }))
        lines.append("```")
        lines.append("")
        if user_prompt.get("assembled_user_message_text"):
            lines.append("##### Assembled user message")
            lines.append("")
            lines.append("```text")
            lines.append(user_prompt.get("assembled_user_message_text") or "")
            lines.append("```")
            lines.append("")
        else:
            lines.append("_Assembled user message text was not available._")
            lines.append("")

        projected_payloads = user_prompt.get("projected_payloads") or []
        if projected_payloads:
            lines.append("##### Projected channel payloads")
            lines.append("")
            for c in projected_payloads:
                ch = c.get("channel")
                plen = c.get("payload_len")
                sha = c.get("payload_sha12")
                lines.append(f"###### {ch} — len={plen} sha12={sha}")
                lines.append("")
                payload = c.get("payload")
                if isinstance(payload, (dict, list)):
                    lines.append("```json")
                    lines.append(_json_pretty(payload))
                    lines.append("```")
                else:
                    lines.append("```text")
                    lines.append("" if payload is None else str(payload))
                    lines.append("```")
                lines.append("")
        else:
            lines.append("_No projected user-side payloads were available._")
            lines.append("")

        lines.append("### Prompt surface")
        lines.append("")
        lines.append("```json")
        lines.append(_json_pretty(prompt_surface))
        lines.append("```")
        lines.append("")

        if detour_visual.get("required_overlay_frame_ids") or detour_visual.get("applied_overlay_frame_ids") or detour_visual.get("derived_overlay_mode"):
            lines.append("### Overlay / visual mode")
            lines.append("")
            lines.append("```json")
            lines.append(_json_pretty(detour_visual))
            lines.append("```")
            lines.append("")

        if any(v for v in solver_timeline.values()):
            lines.append("### Solver query timeline")
            lines.append("")
            lines.append("```json")
            lines.append(_json_pretty(solver_timeline))
            lines.append("```")
            lines.append("")

        lines.append("### Reality vs contract verdict")
        lines.append("")
        lines.append("```json")
        lines.append(_json_pretty(verdict))
        lines.append("```")
        lines.append("")

        lines.append("### Waste dominance metrics")
        lines.append("")
        lines.append("```json")
        lines.append(_json_pretty(dominance))
        lines.append("```")
        lines.append("")

        if setup_cov:
            lines.append("### Setup reply coverage")
            lines.append("")
            lines.append("```json")
            lines.append(_json_pretty(setup_cov))
            lines.append("```")
            lines.append("")

        if confrontation_cov:
            lines.append("### Confrontation reply coverage")
            lines.append("")
            lines.append("```json")
            lines.append(_json_pretty(confrontation_cov))
            lines.append("```")
            lines.append("")

        if confrontation_scorecard:
            lines.append("### Confrontation scorecard")
            lines.append("")
            lines.append("```json")
            lines.append(_json_pretty(confrontation_scorecard))
            lines.append("```")
            lines.append("")

        if resolution_cov:
            lines.append("### Resolution reply coverage")
            lines.append("")
            lines.append("```json")
            lines.append(_json_pretty(resolution_cov))
            lines.append("```")
            lines.append("")

        if resolution_scorecard:
            lines.append("### Resolution scorecard")
            lines.append("")
            lines.append("```json")
            lines.append(_json_pretty(resolution_scorecard))
            lines.append("```")
            lines.append("")

        if waste:
            lines.append("### Reply waste audit")
            lines.append("")
            lines.append("```json")
            lines.append(_json_pretty(waste))
            lines.append("```")
            lines.append("")

        short_conclusion = turn.get("short_conclusion")
        if short_conclusion:
            lines.append("### Short conclusion")
            lines.append("")
            lines.append(short_conclusion)
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"



# -------------------------
# Build structured audit JSON
# -------------------------

def build_audit(events: List[Event]) -> Dict[str, Any]:
    session_id = find_session_id(events)
    start_ts, end_ts = compute_time_bounds(events)
    transcript = extract_spoken_transcript(events)

    turns = group_by_turn(events)
    per_turn: List[Dict[str, Any]] = []

    for tid, tev in turns.items():
        user_text = ""
        for e in tev:
            if e.type == "USER_SAY":
                user_text = e.get("text", "") or ""
                break

        rr = extract_reply_request_details(tev)
        decision_kind = None
        decision_summary = None
        rr_req = None
        rr_turn_ctx = {}
        if rr:
            rr_req = rr.get("reply_request_v1")
            rr_turn_ctx = rr.get("turn_ctx") or {}
            decision = rr.get("decision", {}) or {}
            decision_kind = decision.get("decision_kind")
            decision_summary = decision.get("summary")

        turn_context_v1 = extract_turn_context_v1(tev)
        meaning_block = extract_model_call_block(tev, channel="MEANING", tick_id=1)
        reply_block = extract_model_call_block(tev, channel="REPLY", tick_id=2)

        has_meaning = bool(
            meaning_block.get("payload_out", {}).get("payload_text")
            or meaning_block.get("payload_in", {}).get("response_text")
        )
        has_reply = bool(
            reply_block.get("payload_out", {}).get("payload_text")
            or reply_block.get("payload_in", {}).get("response_text")
        )

        profile_after_tick1 = extract_profile_snapshot_after_tick1(tev)
        grid_facts_pack = extract_grid_facts_snapshots_by_stage(tev)
        grid_present = bool(grid_facts_pack)

        toolplan_planned_v1 = extract_toolplan_planned_v1(tev)
        toolplan_tools = extract_toolplan_tools_from_reply_request(rr_req)

        solving_step_packet_ready = extract_solving_step_packet_ready(tev)
        packet_technique_info = solving_step_packet_ready.get("technique_info") if isinstance(solving_step_packet_ready, dict) else None
        nav_for_turn = solving_step_packet_ready.get("narrative_atoms_v1") if isinstance(solving_step_packet_ready, dict) else None

        solve_step_pack = extract_solve_step_json_ready(tev, packet_technique_info=packet_technique_info)

        if isinstance(solving_step_packet_ready, dict) and solving_step_packet_ready:
            truth_v2_for_turn = solving_step_packet_ready.get("narrative_truth_v2") if isinstance(solving_step_packet_ready.get("narrative_truth_v2"), dict) else None

            if isinstance(nav_for_turn, dict) and nav_for_turn:
                derived_truth_from_atoms = _derive_narrative_truth_from_atoms(
                    nav=nav_for_turn,
                    solve_step_pack=solve_step_pack,
                    truth_v2=truth_v2_for_turn,
                )
                reconstructed_frames = reconstruct_overlay_frames_from_atoms(nav_for_turn, style="full")
            else:
                derived_truth_from_atoms = None
                reconstructed_frames = []

            solving_step_packet_ready["narrative_truth_from_atoms"] = derived_truth_from_atoms
            solving_step_packet_ready["reconstructed_overlay_frames"] = reconstructed_frames
            solving_step_packet_ready["truth_atoms_consistency"] = build_truth_atoms_consistency_report(
                truth_v2=truth_v2_for_turn,
                narrative_atoms_v1=nav_for_turn,
                overlay_frames=reconstructed_frames,
            )

        ev_tick2 = _extract_contract_event(tev, "TICK2_REQUEST_CONTRACT", tick=2)
        tick2_payload = ev_tick2.get("payload") if (ev_tick2 and isinstance(ev_tick2.get("payload"), dict)) else None

        inferred_state = _infer_turn_state_from_anywhere(
            turn_events=tev,
            rr_turn_ctx=rr_turn_ctx,
            tick2_contract=tick2_payload,
            turn_context_v1=turn_context_v1,
        )

        tick1_env_expected, env_warnings = _shape_tick1_intent_envelope_expected(
            tev=tev,
            user_text=user_text,
            meaning_block=meaning_block,
            fallback_turn_state=inferred_state,
        )

        ev_plan = _extract_contract_event(tev, "APP_PLAN_V1", tick=1)
        app_plan_expected = _shape_app_plan_expected(
            ev=ev_plan,
            tick1_env=tick1_env_expected,
            inferred_state_before=inferred_state,
            grid_facts_present=grid_present,
        )

        tick2_expected = _shape_tick2_request_contract_expected(ev_tick2)

        contract_warnings: List[str] = []
        contract_warnings.extend(env_warnings)

        if not turn_context_v1:
            contract_warnings.append("missing_contract_event: TURN_CONTEXT_V1 (tick 1 binder input)")
        if not ev_plan:
            contract_warnings.append("missing_contract_event: APP_PLAN_V1 (tick 1)")
        if not ev_tick2:
            contract_warnings.append("missing_contract_event: TICK2_REQUEST_CONTRACT (tick 2)")

        agenda_audit_events = extract_agenda_audit_events(tev)

        def _compact_fields_profile(snap: Dict[str, Any]) -> Dict[str, Any]:
            if not snap:
                return {}
            ut = snap.get("user_tally_preview_box", {})
            at = snap.get("assistant_tally_preview_box", {})
            rt = snap.get("recent_turns_preview_box", {})
            return {
                "user_tally_preview": ut.get("parsed") if "parsed" in ut else ut.get("raw"),
                "assistant_tally_preview": at.get("parsed") if "parsed" in at else at.get("raw"),
                "recent_turns_preview": rt.get("parsed") if "parsed" in rt else rt.get("raw"),
            }

        per_turn.append({
            "turn_id": tid,
            "user_text": user_text,
            "decision_kind": decision_kind,
            "decision_summary": decision_summary,
            "reply_request_v1": rr.get("reply_request_v1") if rr else None,
            "grid_facts_snapshots": grid_facts_pack or None,
            "solve_step_json_ready": solve_step_pack or None,
            "solving_step_packet_ready": solving_step_packet_ready or None,
            "turn_context_v1": turn_context_v1 or None,
            "toolplan_tools": toolplan_tools,
            "toolplan_planned_v1": toolplan_planned_v1 or None,
            "meaning_block": meaning_block if has_meaning else None,
            "reply_block": reply_block if has_reply else None,
            "memory_snapshots": {
                "profile_snapshot_after_tick1": profile_after_tick1 or None,
            },
            "memory_compact": {
                "profile_snapshot_after_tick1": _compact_fields_profile(profile_after_tick1) if profile_after_tick1 else None,
            },
            "agenda_audit_events": agenda_audit_events,
            "contract_warnings": contract_warnings,
            "tick1_intent_envelope_expected": tick1_env_expected,
            "app_plan_expected": app_plan_expected,
            "tick2_request_contract_expected": tick2_expected,
            "events_count": len(tev),
        })

    md = render_md_report(
        session_id=session_id,
        start_ts=start_ts,
        end_ts=end_ts,
        transcript=transcript,
        per_turn=per_turn,
    )

    audit = {
        "audit_version": "telemetry_audit_v15_resolution_scorecard",
        "session_id": session_id,
        "start": start_ts,
        "end": end_ts,
        "counts": {
            "events": len(events),
            "turns": len(per_turn),
            "transcript_items": len(transcript),
        },
        "transcript": transcript,
        "turns": per_turn,
        "md_render": md,
    }
    return audit


# -------------------------
# Profile snapshot extraction
# -------------------------

def extract_profile_snapshot_after_tick1(turn_events: List[Event]) -> Dict[str, Any]:
    ev = _pick_last(turn_events, "PROFILE_SNAPSHOT_AFTER_TICK1", tick=1)
    source_tag = "PROFILE_SNAPSHOT_AFTER_TICK1"

    # Fallback for branches where the app emits the same data only via TURN_SUMMARY_SNAPSHOT.
    if not ev:
        ev = _pick_last(turn_events, "TURN_SUMMARY_SNAPSHOT")
        source_tag = "TURN_SUMMARY_SNAPSHOT"

    if not ev:
        return {}

    out: Dict[str, Any] = {
        "tag": source_tag,
        "turn_id": ev.get("turn_id"),
        "tick_id": ev.get("tick_id"),
        "policy_req_seq": ev.get("policy_req_seq"),
        "correlation_id": ev.get("correlation_id"),
        "model_call_id": ev.get("model_call_id"),
        "toolplan_id": ev.get("toolplan_id"),
        "user_tally_sha12": ev.get("user_tally_sha12"),
        "user_tally_len": ev.get("user_tally_len"),
        "user_tally_preview": ev.get("user_tally_preview") or ev.get("user_tally_json"),
        "assistant_tally_sha12": ev.get("assistant_tally_sha12"),
        "assistant_tally_len": ev.get("assistant_tally_len"),
        "assistant_tally_preview": ev.get("assistant_tally_preview") or ev.get("assistant_tally_json"),
        "relationship_memory_sha12": ev.get("relationship_memory_sha12"),
        "relationship_memory_len": ev.get("relationship_memory_len"),
        "relationship_memory_preview": ev.get("relationship_memory_preview") or ev.get("relationship_memory_json"),
        "recent_turns_sha12": ev.get("recent_turns_sha12"),
        "recent_turns_len": ev.get("recent_turns_len"),
        "recent_turns_preview": ev.get("recent_turns_preview") or ev.get("recent_turns_json"),
        "user_delta_present": ev.get("user_delta_present"),
        "assistant_delta_present": ev.get("assistant_delta_present"),
    }

    out["user_tally_preview_box"] = _parse_preview_json_or_raw(out.get("user_tally_preview"))
    out["assistant_tally_preview_box"] = _parse_preview_json_or_raw(out.get("assistant_tally_preview"))
    out["relationship_memory_preview_box"] = _parse_preview_json_or_raw(out.get("relationship_memory_preview"))
    out["recent_turns_preview_box"] = _parse_preview_json_or_raw(out.get("recent_turns_preview"))
    return out


# -------------------------
# CLI
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="Generate a telemetry-derived audit report (.md + .json)")
    ap.add_argument(
        "--in",
        dest="in_dir",
        required=True,
        help="Folder containing one or more telemetry files (.jsonl/.json) for the same session",
    )
    ap.add_argument(
        "--out",
        dest="out_base",
        required=True,
        help="Output base path (no extension). Produces <out>.md and <out>.json",
    )
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_base = Path(args.out_base)

    events = load_events_from_folder(in_dir)
    audit = build_audit(events)

    out_md = out_base.with_suffix(".md")
    out_json = out_base.with_suffix(".json")
    out_demand_supply_md = out_base.parent / f"{out_base.name}.demand_supply.md"

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_demand_supply_md.parent.mkdir(parents=True, exist_ok=True)

    md_render = audit.get("md_render")
    if not isinstance(md_render, str):
        raise TypeError(
            f"build_audit/render_md_report produced invalid md_render: "
            f"expected str, got {type(md_render).__name__}"
        )

    out_md.write_text(md_render, encoding="utf-8")

    audit_json = dict(audit)
    audit_json.pop("md_render", None)
    out_json.write_text(_json_pretty(audit_json), encoding="utf-8")

    # Third report: demand / supply turn-by-turn contract report
    session_id = audit.get("session_id") or "unknown_session"
    start_ts = audit.get("start")
    end_ts = audit.get("end")

    demand_supply_turns: List[Dict[str, Any]] = []
    turns_map = group_by_turn(events)
    for _, tev in turns_map.items():
        rep = extract_demand_supply_turn_report(tev)
        hdr = rep.get("turn_header") or {}
        if not hdr.get("turn_id"):
            continue
        if not (
            rep.get("demand_summary")
            or rep.get("contract_view")
            or rep.get("selected_supply_summary")
            or rep.get("reply_waste_audit")
        ):
            continue
        demand_supply_turns.append(rep)

    demand_supply_md = render_demand_supply_md_report(
        session_id=session_id,
        start_ts=start_ts,
        end_ts=end_ts,
        turns=demand_supply_turns,
    )
    out_demand_supply_md.write_text(demand_supply_md, encoding="utf-8")

    print(f"[telemetry_audit] wrote: {out_md}")
    print(f"[telemetry_audit] wrote: {out_json}")
    print(f"[telemetry_audit] wrote: {out_demand_supply_md}")


if __name__ == "__main__":
    main()