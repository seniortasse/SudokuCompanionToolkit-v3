from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from time import perf_counter


def now_text() -> str:
    """
    Return a compact timestamp for live production logs.
    """

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_seconds(seconds: float) -> str:
    """
    Format elapsed seconds for console progress.
    """

    value = float(seconds)
    if value < 60:
        return f"{value:.1f}s"

    minutes = int(value // 60)
    remaining = value - (minutes * 60)
    return f"{minutes}m {remaining:.1f}s"


def print_progress(phase: str, message: str) -> None:
    """
    Print one live, flush-safe progress line.

    Example:
        [2026-04-29 17:10:03] [LOG 3/1008] BK-CL9-DW-B01/fr L1-003 OK | 5.8s
    """

    phase_text = str(phase or "INFO").strip().upper()
    message_text = str(message or "").strip()

    print(
        f"[{now_text()}] [{phase_text}] {message_text}",
        file=sys.stdout,
        flush=True,
    )


@dataclass(frozen=True)
class ProgressTimer:
    """
    Small helper for elapsed-time reporting.
    """

    started_at: float

    @classmethod
    def start(cls) -> "ProgressTimer":
        return cls(started_at=perf_counter())

    def elapsed_seconds(self) -> float:
        return perf_counter() - self.started_at

    def elapsed_text(self) -> str:
        return format_seconds(self.elapsed_seconds())