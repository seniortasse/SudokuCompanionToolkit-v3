from __future__ import annotations


def build_record_id(*, layout_short: str, ordinal: int) -> str:
    layout = str(layout_short).strip().upper().replace(" ", "-")
    return f"REC-{layout}-{int(ordinal):08d}"