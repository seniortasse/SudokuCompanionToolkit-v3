from __future__ import annotations


def _norm_token(value: str) -> str:
    return str(value).strip().upper().replace(" ", "-")


def build_library_id(short_code: str) -> str:
    return f"LIB-{_norm_token(short_code)}"


def build_aisle_id(short_code: str) -> str:
    return f"AIS-{_norm_token(short_code)}"


def build_book_id(library_short: str, aisle_short: str, book_number: int) -> str:
    return f"BK-{_norm_token(library_short)}-{_norm_token(aisle_short)}-B{int(book_number):02d}"


def build_section_id(section_code: str) -> str:
    return f"SEC-{_norm_token(section_code)}"


def build_local_puzzle_code(section_code: str, ordinal: int) -> str:
    normalized_section = _norm_token(section_code)
    return f"{normalized_section}-{int(ordinal):03d}"


def build_puzzle_uid(
    library_short: str,
    aisle_short: str,
    book_number: int,
    section_code: str,
    ordinal: int,
) -> str:
    book_id = build_book_id(library_short, aisle_short, book_number)
    local_code = build_local_puzzle_code(section_code, ordinal)
    return f"PZ-{book_id[3:]}-{local_code}"


def build_puzzle_uid_for_book(
    *,
    book_id: str,
    section_code: str,
    ordinal: int,
) -> str:
    local_code = build_local_puzzle_code(section_code, ordinal)
    normalized_book_id = str(book_id).strip().upper()
    if normalized_book_id.startswith("BK-"):
        normalized_book_id = normalized_book_id[3:]
    return f"PZ-{normalized_book_id}-{local_code}"


def build_friendly_puzzle_id(
    library_short: str,
    aisle_short: str,
    book_number: int,
    section_code: str,
    ordinal: int,
) -> str:
    return (
        f"{_norm_token(library_short)}-"
        f"{_norm_token(aisle_short)}-"
        f"B{int(book_number):02d}-"
        f"{build_local_puzzle_code(section_code, ordinal)}"
    )


def build_pattern_id(library_short: str, ordinal: int) -> str:
    return f"PAT-{_norm_token(library_short)}-{int(ordinal):04d}"