#!/usr/bin/env python3
# Conversation Quality Scorer (CQS v1)
#
# Reads a telemetry JSONL file and outputs:
#  - reconstructed transcript (user/assistant turns)
#  - mechanical metrics (turn-taking, latencies, repairs, loops, structure)
#  - a heuristic Conversation Quality Score (0-100)
#
# Note: Semantic categories (mutual understanding / coherence) are hard to score
# reliably without an LLM judge. This script outputs a conservative proxy score
# and provides a JSON payload you can feed to an LLM judge later.

from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                # Ignore malformed lines (e.g., logcat prefixes)
                continue
    out.sort(key=lambda x: x.get("ts_epoch_ms", x.get("ts_ms", 0)) or 0)
    return out


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def is_greeting(s: str) -> bool:
    s = s.lower()
    return any(w in s for w in ["hello", "hi ", "hi!", "hey", "good morning", "good afternoon", "good evening", "salut", "bonjour"])


def is_closing(s: str) -> bool:
    s = s.lower()
    return any(w in s for w in ["bye", "goodbye", "see you", "talk later", "à bientôt", "au revoir", "take care"])


def looks_like_repair(s: str) -> bool:
    s = s.lower()
    return any(
        w in s
        for w in [
            "sorry",
            "i didn't catch",
            "could you repeat",
            "can you say that again",
            "do you mean",
            "pardon",
            "je n'ai pas compris",
            "répète",
        ]
    )


@dataclass
class Turn:
    speaker: str  # "user" | "assistant"
    text: str
    t_start: Optional[int] = None  # ms epoch
    t_end: Optional[int] = None  # ms epoch
    row_id: Optional[int] = None  # ASR row id
    speak_req_id: Optional[int] = None  # assistant speak request id
    reply_to_row_id: Optional[int] = None  # assistant reply correlation
    engine: Optional[str] = None


def build_turns(events: List[Dict[str, Any]]) -> Tuple[List[Turn], Dict[str, Any]]:
    turns: List[Turn] = []

    # TTS times by speak_req_id (preferred correlation)
    tts_times_by_req: Dict[int, Dict[str, int]] = {}

    for e in events:
        typ = e.get("type", "")
        t = e.get("ts_epoch_ms")
        if t is None:
            continue
        if typ == "TTS_START":
            rid = e.get("speak_req_id")
            if rid is not None:
                tts_times_by_req.setdefault(int(rid), {})["start"] = int(t)
        if typ == "TTS_DONE":
            rid = e.get("speak_req_id")
            if rid is not None:
                tts_times_by_req.setdefault(int(rid), {})["end"] = int(t)

    # User turns: prefer USER_SAY (normalized, conductor-accepted)
    for e in events:
        if e.get("type") == "USER_SAY":
            turns.append(
                Turn(
                    speaker="user",
                    text=norm_text(e.get("text", "")),
                    t_end=e.get("ts_epoch_ms"),
                    row_id=e.get("row_id"),
                )
            )

    # Fallback: use ASR_FINAL if USER_SAY not present
    if not any(t.speaker == "user" for t in turns):
        for e in events:
            if e.get("type") == "ASR_FINAL":
                turns.append(
                    Turn(
                        speaker="user",
                        text=norm_text(e.get("text", "")),
                        t_end=e.get("ts_epoch_ms"),
                        row_id=e.get("row_id"),
                    )
                )

    # Assistant turns: ASSISTANT_SAY
    for e in events:
        if e.get("type") == "ASSISTANT_SAY":
            turns.append(
                Turn(
                    speaker="assistant",
                    text=norm_text(e.get("text", "")),
                    t_start=e.get("ts_epoch_ms"),
                    speak_req_id=e.get("speak_req_id"),
                    reply_to_row_id=e.get("reply_to_row_id") or e.get("reply_to_row"),
                    engine=e.get("engine"),
                )
            )

    # Attach TTS times when correlated
    for t in turns:
        if t.speaker != "assistant":
            continue
        if t.speak_req_id is not None and int(t.speak_req_id) in tts_times_by_req:
            d = tts_times_by_req[int(t.speak_req_id)]
            t.t_start = d.get("start", t.t_start)
            t.t_end = d.get("end", t.t_end)

    # Sort turns by their best timestamp
    def turn_key(x: Turn) -> int:
        if x.speaker == "assistant":
            return int(x.t_start or 0)
        return int(x.t_end or 0)

    turns.sort(key=turn_key)

    diag = {
        "assistant_say_count": sum(1 for e in events if e.get("type") == "ASSISTANT_SAY"),
        "user_say_count": sum(1 for e in events if e.get("type") == "USER_SAY"),
        "asr_final_count": sum(1 for e in events if e.get("type") == "ASR_FINAL"),
        "tts_start_count": sum(1 for e in events if e.get("type") == "TTS_START"),
        "tts_done_count": sum(1 for e in events if e.get("type") == "TTS_DONE"),
    }
    return turns, diag


def compute_metrics(turns: List[Turn], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not events:
        return {}

    times = [e.get("ts_epoch_ms") for e in events if e.get("ts_epoch_ms") is not None]
    if not times:
        return {}

    t0 = int(min(times))
    t1 = int(max(times))
    duration_ms = max(0, t1 - t0)

    u = [t for t in turns if t.speaker == "user" and t.text]
    a = [t for t in turns if t.speaker == "assistant" and t.text]

    balance_ratio = (len(u) / len(a)) if len(a) > 0 else None

    # user->assistant latency
    lat_u2a: List[int] = []
    for ut in u:
        if ut.t_end is None:
            continue
        candidates = [at for at in a if (at.t_start or 0) >= ut.t_end]
        if candidates:
            at = min(candidates, key=lambda x: x.t_start or 0)
            if at.t_start is not None:
                lat_u2a.append(int(at.t_start) - int(ut.t_end))

    # assistant->user latency: assistant end -> ASR_START
    asr_starts = [int(e["ts_epoch_ms"]) for e in events if e.get("type") == "ASR_START" and e.get("ts_epoch_ms") is not None]
    lat_a2u: List[int] = []
    for at in a:
        if at.t_end is None:
            continue
        starts = [t for t in asr_starts if t >= int(at.t_end)]
        if starts:
            lat_a2u.append(min(starts) - int(at.t_end))

    repair_assistant = sum(1 for at in a if looks_like_repair(at.text))
    asr_errors = sum(1 for e in events if e.get("type") in ["ASR_ERROR", "ASR_ERROR_SOFT"])

    assistant_loops = 0
    for x, y in zip(a, a[1:]):
        if x.text and y.text and x.text == y.text:
            assistant_loops += 1

    opening_present = bool(a and is_greeting(a[0].text))
    closing_present = bool(a and is_closing(a[-1].text))
    alignment_present_proxy = False
    for at in a[:2]:
        s = at.text.lower()
        if any(w in s for w in ["how can i help", "what would you like", "what are we doing", "let's", "we will"]):
            alignment_present_proxy = True

    return {
        "session_duration_ms": duration_ms,
        "total_turns": (len(u) + len(a)),
        "user_turns": len(u),
        "assistant_turns": len(a),
        "balance_ratio_user_over_assistant": balance_ratio,
        "latency_user_to_assistant_ms_avg": (statistics.mean(lat_u2a) if lat_u2a else None),
        "latency_user_to_assistant_ms_p50": (statistics.median(lat_u2a) if lat_u2a else None),
        "latency_assistant_to_user_ms_avg": (statistics.mean(lat_a2u) if lat_a2u else None),
        "repair_assistant_count": repair_assistant,
        "asr_error_count": asr_errors,
        "assistant_loop_repeat_count": assistant_loops,
        "opening_present": opening_present,
        "alignment_present_proxy": alignment_present_proxy,
        "closing_present": closing_present,
    }


def score_from_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    # Heuristic-only score (no semantic judge)
    u = int(metrics.get("user_turns") or 0)
    a = int(metrics.get("assistant_turns") or 0)

    # 1) Two-way exchange (0–20)
    two_way = 0
    if u >= 1 and a >= 1:
        two_way = 10
    if u >= 2 and a >= 2:
        two_way = 15
    if u >= 3 and a >= 3:
        two_way = 20

    br = metrics.get("balance_ratio_user_over_assistant")
    if br is not None:
        try:
            brf = float(br)
            if brf < 0.3 or brf > 3.0:
                two_way = max(0, two_way - 5)
        except Exception:
            pass

    # 2) Mutual understanding (0–25) — proxy (weak)
    mutual = 12 if (u >= 1 and a >= 1) else 0
    mutual -= min(8, int(metrics.get("assistant_loop_repeat_count") or 0) * 4)
    mutual -= min(6, int(metrics.get("asr_error_count") or 0) * 2)
    mutual = max(0, min(25, mutual))

    # 3) Coherence & continuity (0–20) — proxy
    coh = 10 if a >= 1 else 0
    coh -= min(10, int(metrics.get("assistant_loop_repeat_count") or 0) * 5)
    coh = max(0, min(20, coh))

    # 4) Repair behavior (0–15) — proxy
    rep = 0
    if int(metrics.get("repair_assistant_count") or 0) > 0:
        rep = 8
    if int(metrics.get("repair_assistant_count") or 0) <= 2 and u >= 1 and a >= 1:
        rep = 12
    rep -= min(6, int(metrics.get("asr_error_count") or 0) * 2)
    rep = max(0, min(15, rep))

    # 5) Clarity & helpfulness (0–10) — unknown => midpoint if any assistant
    cla = 0
    if a >= 1:
        cla = 5
    if a >= 2:
        cla = 6

    # 6) Structure & closure (0–10)
    st = 0
    if metrics.get("opening_present"):
        st += 3
    if metrics.get("alignment_present_proxy"):
        st += 3
    if metrics.get("closing_present"):
        st += 4
    st = min(10, st)

    total = max(0, min(100, two_way + mutual + coh + rep + cla + st))

    return {
        "ConversationQualityScore": total,
        "subscores": {
            "two_way_exchange_0_20": two_way,
            "mutual_understanding_proxy_0_25": mutual,
            "coherence_proxy_0_20": coh,
            "repair_proxy_0_15": rep,
            "clarity_proxy_0_10": cla,
            "structure_proxy_0_10": st,
        },
        "note": "Heuristic score only. For faithful Mutual Understanding/Coherence, plug an LLM judge over the transcript.",
    }


def analyze(path: str) -> Dict[str, Any]:
    events = load_jsonl(path)
    turns, diag = build_turns(events)
    metrics = compute_metrics(turns, events)
    score = score_from_metrics(metrics)
    transcript = [asdict(t) for t in turns if t.text]
    return {"diagnostics": diag, "metrics": metrics, "score": score, "transcript": transcript}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", help="Telemetry JSONL file path")
    ap.add_argument("--out", help="Write full JSON report to this file")
    ap.add_argument("--print_transcript", action="store_true", help="Print reconstructed transcript")
    args = ap.parse_args()

    report = analyze(args.jsonl)

    if args.print_transcript:
        for t in report["transcript"]:
            print(f'[{t.get("speaker")}] {t.get("text")}')
        print()

    print("ConversationQualityScore:", report["score"]["ConversationQualityScore"])
    print("Subscores:", json.dumps(report["score"]["subscores"], ensure_ascii=False))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print("Wrote:", args.out)


if __name__ == "__main__":
    main()
