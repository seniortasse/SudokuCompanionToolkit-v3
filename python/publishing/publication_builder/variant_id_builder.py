from __future__ import annotations

from typing import Any, Dict, Optional


_CHANNEL_TOKEN_MAP = {
    "amazon_kdp_paperback": "KDP",
    "kdp": "KDP",
}

_PAPER_TOKEN_MAP = {
    "white_bw": "BW",
    "cream_bw": "BW",
    "white_color": "COLOR",
    "premium_color": "COLOR",
}

_LOCALE_TOKEN_MAP = {
    "en": "EN",
    "english": "EN",
    "fr": "FR",
    "french": "FR",
    "de": "DE",
    "german": "DE",
    "it": "IT",
    "italian": "IT",
    "es": "ES",
    "spanish": "ES",
    "pt": "PT",
    "portuguese": "PT",
    "nl": "NL",
    "dutch": "NL",
}


def _normalize_locale_token(value: Optional[str]) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return "EN"

    base = raw.split("-")[0].split("_")[0]
    if base in _LOCALE_TOKEN_MAP:
        return _LOCALE_TOKEN_MAP[base]

    return base.upper()


def _channel_token(channel_id: Optional[str]) -> str:
    raw = str(channel_id or "").strip().lower()
    if raw in _CHANNEL_TOKEN_MAP:
        return _CHANNEL_TOKEN_MAP[raw]
    if not raw:
        return "PUB"
    return raw.upper().replace("-", "_")


def _paper_token(paper_type: Optional[str]) -> str:
    raw = str(paper_type or "").strip().lower()
    if raw in _PAPER_TOKEN_MAP:
        return _PAPER_TOKEN_MAP[raw]
    if not raw:
        return "BW"
    return raw.upper().replace("-", "_")


def _layout_token(puzzles_per_page: Optional[int]) -> str:
    count = int(puzzles_per_page or 0)
    if count <= 0:
        return "XUP"
    return f"{count}UP"


def _trim_token(format_id: Optional[str]) -> str:
    raw = str(format_id or "").strip().lower()

    known = {
        "amazon_kdp_paperback_8_5x11_bw": "8511",
        "amazon_kdp_paperback_8_5x11_color": "8511",
        "amazon_kdp_paperback_8_5x11": "8511",
        "kdp_8_5x11_bw": "8511",
        "kdp_8_5x11_color": "8511",
        "kdp_8_5x11": "8511",
        "8_5x11": "8511",
        "8.5x11": "8511",
        "8511": "8511",
    }
    if raw in known:
        return known[raw]

    if "8_5x11" in raw or "8.5x11" in raw:
        return "8511"

    cleaned = (
        raw.replace("amazon_kdp_paperback_", "")
        .replace("kdp_", "")
        .replace("_bw", "")
        .replace("_color", "")
        .replace("_", "")
        .replace(".", "")
        .replace("x", "")
    )
    return cleaned.upper() or "TRIM"


def _base_book_id(base_spec: Dict[str, Any]) -> str:
    return str(base_spec.get("book_id") or "").strip()


def build_variant_publication_id(
    *,
    base_spec: Dict[str, Any],
    locale: Optional[str],
    puzzles_per_page: Optional[int],
    explicit_publication_id: Optional[str] = None,
) -> str:
    if explicit_publication_id:
        return str(explicit_publication_id).strip()

    book_id = _base_book_id(base_spec)
    channel_id = str(base_spec.get("channel_id") or "").strip()
    format_id = str(base_spec.get("format_id") or "").strip()
    paper_type = str(base_spec.get("paper_type") or "").strip()

    return "-".join(
        [
            "PUB",
            book_id,
            _channel_token(channel_id),
            _normalize_locale_token(locale),
            _layout_token(puzzles_per_page),
            _trim_token(format_id),
            _paper_token(paper_type),
        ]
    )