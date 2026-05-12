from __future__ import annotations

import re

_RECORD_RE = re.compile(r"^REC-[A-Z0-9]+(?:-[A-Z0-9]+)*-\d{8}$")
_LIBRARY_RE = re.compile(r"^LIB-[A-Z0-9]+(?:-[A-Z0-9]+)*$")
_AISLE_RE = re.compile(r"^AIS-[A-Z0-9]+(?:-[A-Z0-9]+)*$")
_BOOK_RE = re.compile(r"^BK-[A-Z0-9]+-[A-Z0-9]+-B\d{2}$")
_SECTION_RE = re.compile(r"^SEC-[A-Z0-9]+(?:-[A-Z0-9]+)*$")
_LOCAL_PUZZLE_RE = re.compile(r"^[A-Z0-9]+(?:-[A-Z0-9]+)*-\d{3}$")
_FRIENDLY_PUZZLE_RE = re.compile(r"^[A-Z0-9]+-[A-Z0-9]+-B\d{2}-[A-Z0-9]+(?:-[A-Z0-9]+)*-\d{3}$")
_PUZZLE_UID_RE = re.compile(r"^PZ-[A-Z0-9]+-[A-Z0-9]+-B\d{2}-[A-Z0-9]+(?:-[A-Z0-9]+)*-\d{3}$")
_PATTERN_RE = re.compile(r"^PAT-[A-Z0-9]+-\d{4}$")


def is_valid_record_id(value: str) -> bool:
    return bool(_RECORD_RE.fullmatch(str(value).strip().upper()))


def is_valid_library_id(value: str) -> bool:
    return bool(_LIBRARY_RE.fullmatch(str(value).strip().upper()))


def is_valid_aisle_id(value: str) -> bool:
    return bool(_AISLE_RE.fullmatch(str(value).strip().upper()))


def is_valid_book_id(value: str) -> bool:
    return bool(_BOOK_RE.fullmatch(str(value).strip().upper()))


def is_valid_section_id(value: str) -> bool:
    return bool(_SECTION_RE.fullmatch(str(value).strip().upper()))


def is_valid_local_puzzle_code(value: str) -> bool:
    return bool(_LOCAL_PUZZLE_RE.fullmatch(str(value).strip().upper()))


def is_valid_friendly_puzzle_id(value: str) -> bool:
    return bool(_FRIENDLY_PUZZLE_RE.fullmatch(str(value).strip().upper()))


def is_valid_puzzle_uid(value: str) -> bool:
    return bool(_PUZZLE_UID_RE.fullmatch(str(value).strip().upper()))


def is_valid_pattern_id(value: str) -> bool:
    return bool(_PATTERN_RE.fullmatch(str(value).strip().upper()))