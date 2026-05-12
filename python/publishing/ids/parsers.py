from __future__ import annotations

from typing import Dict

from .validators import is_valid_friendly_puzzle_id


def parse_friendly_puzzle_id(value: str) -> Dict[str, str]:
    """
    Parse ids like:
        CL9-DW-B01-L1-001
        CL9-TC-B03-L2-120

    Returns:
        {
            "library_short": "CL9",
            "aisle_short": "DW",
            "book_code": "B01",
            "section_code": "L1",
            "ordinal": "001",
        }
    """
    raw = str(value).strip().upper()
    if not is_valid_friendly_puzzle_id(raw):
        raise ValueError(f"Invalid friendly puzzle id: {value}")

    parts = raw.split("-")
    if len(parts) < 5:
        raise ValueError(f"Invalid friendly puzzle id: {value}")

    library_short = parts[0]
    aisle_short = parts[1]
    book_code = parts[2]
    section_code = "-".join(parts[3:-1])
    ordinal = parts[-1]

    return {
        "library_short": library_short,
        "aisle_short": aisle_short,
        "book_code": book_code,
        "section_code": section_code,
        "ordinal": ordinal,
    }