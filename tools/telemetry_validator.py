#!/usr/bin/env python3
"""
Telemetry Completeness Validator for CQS v1

Usage:
  python telemetry_validator.py path/to/telemetry.jsonl

What it checks (for "100% telemetry"):
- USER_SAY events exist (user final text captured)
- ASSISTANT_SAY includes speak_req_id (+ ideally reply_to_row_id)
- TTS_START and TTS_DONE include speak_req_id (+ ideally reply_to_row_id)
- TURN_PAIR edges exist (row_id <-> speak_req_id)
- ASR row evaluation emitted at least once (CONVTEL_ASR_ROW_EVAL)
- convo_session_id present on key event families (USER_SAY / ASSISTANT_SAY / TTS_*)
"""

import json
import sys
from collections import Counter

REQUIRED = {
    "USER_SAY": ["row_id", "text"],
    "ASSISTANT_SAY": ["speak_req_id", "text"],
    "TTS_START": ["speak_req_id"],
    "TTS_DONE": ["speak_req_id"],
}
RECOMMENDED_FIELDS = {
    "ASSISTANT_SAY": ["reply_to_row_id", "convo_session_id"],
    "USER_SAY": ["convo_session_id"],
    "TTS_START": ["reply_to_row_id", "convo_session_id"],
    "TTS_DONE": ["reply_to_row_id", "convo_session_id"],
}
RECOMMENDED_EVENTS = ["TURN_PAIR", "CONVTEL_ASR_ROW_EVAL"]

def load_jsonl(path: str):
    events = []
    bad_lines = 0
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except Exception:
                bad_lines += 1
    return events, bad_lines

def field_presence(events, etype, field):
    evs = [e for e in events if e.get("type") == etype]
    if not evs:
        return False
    return any(field in e and e.get(field) is not None for e in evs)

def main():
    if len(sys.argv) < 2:
        print("Usage: python telemetry_validator.py telemetry.jsonl")
        sys.exit(2)
    path = sys.argv[1]
    events, bad_lines = load_jsonl(path)
    if not events:
        print("No parseable events found.")
        sys.exit(1)

    types = Counter(e.get("type") for e in events)
    print(f"Loaded events: {len(events)}   (bad/unparseable lines: {bad_lines})")
    print("Top event types:", ", ".join([f"{t}={c}" for t,c in types.most_common(12)]))

    print("\n== REQUIRED EVENT/FIELD CHECKS ==")
    ok_all = True
    for et, fields in REQUIRED.items():
        count = types.get(et, 0)
        if count == 0:
            ok_all = False
            print(f"[FAIL] {et}: missing (count=0)")
            continue
        missing = [f for f in fields if not field_presence(events, et, f)]
        if missing:
            ok_all = False
            print(f"[FAIL] {et}: present (count={count}) but missing fields: {missing}")
        else:
            print(f"[OK]   {et}: present (count={count}) and required fields ok")

    print("\n== RECOMMENDED CHECKS (for easier reconstruction) ==")
    for et in RECOMMENDED_EVENTS:
        if types.get(et, 0) > 0:
            print(f"[OK]   {et}: present (count={types.get(et)})")
        else:
            print(f"[WARN] {et}: missing (count=0)")

    for et, fields in RECOMMENDED_FIELDS.items():
        if types.get(et, 0) == 0:
            continue
        missing = [f for f in fields if not field_presence(events, et, f)]
        if missing:
            print(f"[WARN] {et}: missing recommended fields: {missing}")
        else:
            print(f"[OK]   {et}: recommended fields present")

    # Quick joinability signal
    print("\n== JOINABILITY SIGNAL ==")
    speak_ids = sum(1 for e in events if "speak_req_id" in e and e.get("speak_req_id") is not None)
    reply_ids = sum(1 for e in events if "reply_to_row_id" in e and e.get("reply_to_row_id") is not None)
    convo_ids = sum(1 for e in events if "convo_session_id" in e and e.get("convo_session_id") is not None)
    print(f"events with speak_req_id: {speak_ids}")
    print(f"events with reply_to_row_id: {reply_ids}")
    print(f"events with convo_session_id: {convo_ids}")

    print("\n== RESULT ==")
    if ok_all:
        print("✅ Telemetry has the REQUIRED pieces for CQS v1 reconstruction/scoring.")
    else:
        print("❌ Telemetry is NOT yet complete for full CQS v1 scoring. See FAIL items above.")

if __name__ == "__main__":
    main()
