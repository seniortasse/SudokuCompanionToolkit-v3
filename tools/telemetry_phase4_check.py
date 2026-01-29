#!/usr/bin/env python3
import argparse, json, sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# ------------------------------- IO ---------------------------------

def load_events(path: str) -> List[Dict[str, Any]]:
    events = []
    bad = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue

            # Try direct JSON first
            obj = None
            try:
                obj = json.loads(line)
            except Exception:
                # Try to parse from first '{' (logcat prefix case)
                j = line.find("{")
                if j >= 0:
                    try:
                        obj = json.loads(line[j:])
                    except Exception:
                        obj = None

            if obj is None:
                bad += 1
                # Print first few bad lines for debugging, then stay quiet
                if bad <= 5:
                    print(f"[warn] skipping non-JSON line {i}: {line[:120]}", file=sys.stderr)
                continue

            # Only keep telemetry-like objects with ts_epoch_ms + type
            if "ts_epoch_ms" not in obj or "type" not in obj:
                continue

            obj["_line"] = i
            events.append(obj)

    events.sort(key=lambda e: (e["ts_epoch_ms"], e.get("_line", 0)))
    if bad:
        print(f"[info] skipped {bad} non-JSON lines.", file=sys.stderr)
    return events



def ctx(events: List[Dict[str, Any]], idx: int, n: int = 8) -> str:
    lo = max(0, idx - n)
    hi = min(len(events), idx + n + 1)
    out = []
    for k in range(lo, hi):
        e = events[k]
        out.append(f"{k:04d} line={e.get('_line')} ts={e.get('ts_iso','?')} type={e.get('type')} "
                   f"tts_id={e.get('tts_id','-')} serial={e.get('serial','-')} source={e.get('source','-')} "
                   f"reason={e.get('reason','-')} state={e.get('state','-')}")
    return "\n".join(out)

# -------------------------- interval helpers -------------------------

@dataclass
class Interval:
    start: int
    end: int
    meta: Dict[str, Any]

def build_tts_active(events: List[Dict[str, Any]]) -> List[Interval]:
    start_by_id: Dict[int, int] = {}
    intervals: List[Interval] = []
    for e in events:
        t = e["ts_epoch_ms"]
        if e.get("type") == "TTS_START":
            start_by_id[e["tts_id"]] = t
        if e.get("type") == "TTS_STOP" and e.get("source") in ("ended", "manual_stop"):
            tid = e.get("tts_id")
            if tid in start_by_id:
                intervals.append(Interval(start_by_id.pop(tid), t, {"tts_id": tid}))
    return intervals

def build_asr_active(events: List[Dict[str, Any]]) -> List[Interval]:
    start_by_serial: Dict[int, int] = {}
    intervals: List[Interval] = []
    for e in events:
        t = e["ts_epoch_ms"]
        if e.get("type") == "ASR_START":
            start_by_serial[e["serial"]] = t
        if e.get("type") == "ASR_STOP":
            s = e.get("serial")
            if s in start_by_serial:
                intervals.append(Interval(start_by_serial.pop(s), t, {"serial": s}))
    return intervals

def overlaps(a: Interval, b: Interval) -> bool:
    return max(a.start, b.start) < min(a.end, b.end)

# ----------------------------- assertions ----------------------------

def fail(msg: str, events: List[Dict[str, Any]], idx: int) -> int:
    e = events[idx]
    print("\n❌ FIRST FAILURE")
    print(msg)
    print(f"At: ts={e.get('ts_iso','?')} line={e.get('_line')} type={e.get('type')}")
    print("\nContext:\n" + ctx(events, idx))
    return 1

def check_phase4(events: List[Dict[str, Any]],
                 ui_start_to_tts_start_ms: int = 0,
                 ui_stop_after_tts_stop_ms: int = 250) -> int:
    """
    Phase-4 gates (core A/B/C + mandatory E).
    Defaults enforce STRICT truthfulness:
      - UI_BARS_START must be at/after TTS_START (0ms early tolerance).
    """

    # Build active windows for overlap checks
    tts_intervals = build_tts_active(events)
    asr_intervals = build_asr_active(events)

    # A1/A2: No overlap between TTS_ACTIVE and ASR_ACTIVE
    for i, ti in enumerate(tts_intervals):
        for aj, ai in enumerate(asr_intervals):
            if overlaps(ti, ai):
                # find a representative event index near overlap start
                overlap_ts = max(ti.start, ai.start)
                idx = next(k for k,e in enumerate(events) if e["ts_epoch_ms"] >= overlap_ts)
                return fail(f"A (no overlap) violated: TTS_ACTIVE(tts_id={ti.meta['tts_id']}) overlaps "
                            f"ASR_ACTIVE(serial={ai.meta['serial']}).", events, idx)

    # E1: UI_BARS_START must not precede actual playback (TTS_START)
    # We match each UI_BARS_START to the next TTS_START.
    for idx, e in enumerate(events):
        if e.get("type") == "UI_BARS_START" and e.get("source") == "tts_azure":
            ui_ts = e["ts_epoch_ms"]
            nxt = None
            for j in range(idx, len(events)):
                if events[j].get("type") == "TTS_START":
                    nxt = events[j]
                    break
            if nxt is None:
                return fail("E1 violated: UI_BARS_START occurred but no subsequent TTS_START exists.", events, idx)

            delta = nxt["ts_epoch_ms"] - ui_ts
            # STRICT truthfulness default: delta must be <= 0  (no early start)
            if delta > ui_start_to_tts_start_ms:
                return fail(f"E1 violated: UI_BARS_START is {delta}ms before TTS_START(tts_id={nxt.get('tts_id')}).",
                            events, idx)

    # E2: UI_BARS_STOP should happen soon after terminal TTS_STOP
    # (We allow a small delay due to main-thread/UI scheduling.)
    for idx, e in enumerate(events):
        if e.get("type") == "TTS_STOP" and e.get("source") in ("ended", "manual_stop"):
            stop_ts = e["ts_epoch_ms"]
            found = False
            for j in range(idx, len(events)):
                ej = events[j]
                if ej.get("type") == "UI_BARS_STOP":
                    found = True
                    delay = ej["ts_epoch_ms"] - stop_ts
                    if delay > ui_stop_after_tts_stop_ms:
                        return fail(f"E2 violated: UI_BARS_STOP is too late ({delay}ms) after TTS_STOP.", events, idx)
                    break
                # stop scanning if we hit next TTS_START; then it's missing
                if ej.get("type") == "TTS_START":
                    break
            if not found:
                return fail("E2 violated: terminal TTS_STOP occurred but no UI_BARS_STOP followed before next TTS_START.",
                            events, idx)

    print("✅ Phase-4 checks PASSED (A overlap + mandatory E bars truthfulness).")
    return 0

# -------------------------------- main --------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--phase4", action="store_true", help="Run Phase-4 checks (A + mandatory E).")
    ap.add_argument("--ui-early-ms", type=int, default=0,
                    help="How many ms UI_BARS_START is allowed to precede TTS_START. Default 0 for strict truthfulness.")
    args = ap.parse_args()

    events = load_events(args.file)
    if args.phase4:
        return check_phase4(events, ui_start_to_tts_start_ms=args.ui_early_ms)

    print("No check selected. Use --phase4.")
    return 2

if __name__ == "__main__":
    sys.exit(main())