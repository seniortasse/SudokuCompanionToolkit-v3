from __future__ import annotations

CANDIDATE_STATUS_AVAILABLE = "available"
CANDIDATE_STATUS_ASSIGNED = "assigned"
CANDIDATE_STATUS_ARCHIVED = "archived"
CANDIDATE_STATUS_REJECTED = "rejected"
CANDIDATE_STATUS_DUPLICATE_BLOCKED = "duplicate_blocked"

ALL_CANDIDATE_STATUSES = {
    CANDIDATE_STATUS_AVAILABLE,
    CANDIDATE_STATUS_ASSIGNED,
    CANDIDATE_STATUS_ARCHIVED,
    CANDIDATE_STATUS_REJECTED,
    CANDIDATE_STATUS_DUPLICATE_BLOCKED,
}

DEFAULT_CANDIDATE_STATUS = CANDIDATE_STATUS_AVAILABLE


def normalize_candidate_status(value: str | None) -> str:
    status = str(value or DEFAULT_CANDIDATE_STATUS).strip().lower()
    if status not in ALL_CANDIDATE_STATUSES:
        raise ValueError(f"Unsupported candidate status: {value}")
    return status