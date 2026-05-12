from __future__ import annotations

from typing import Dict, List

from python.publishing.schemas.page_types import (
    FEATURES_PAGE,
    PROMO_PAGE,
    RULES_PAGE,
    SECTION_HIGHLIGHTS_PAGE,
    SECTION_OPENER_PAGE,
    SECTION_PATTERN_GALLERY_PAGE,
    TOC_PAGE,
    TUTORIAL_PAGE,
    WARMUP_PAGE,
    WELCOME_PAGE,
)


_FRONT_MATTER_PAGE_SPECS: Dict[str, dict] = {
    WELCOME_PAGE: {
        "page_type": WELCOME_PAGE,
        "template_id": "welcome_page_basic",
        "show_page_number": False,
        "page_number_style": None,
    },
    FEATURES_PAGE: {
        "page_type": FEATURES_PAGE,
        "template_id": "features_page_basic",
        "show_page_number": False,
        "page_number_style": None,
    },
    TOC_PAGE: {
        "page_type": TOC_PAGE,
        "template_id": "toc_page_basic",
        "show_page_number": False,
        "page_number_style": None,
    },
    RULES_PAGE: {
        "page_type": RULES_PAGE,
        "template_id": "rules_page_basic",
        "show_page_number": False,
        "page_number_style": None,
    },
    TUTORIAL_PAGE: {
        "page_type": TUTORIAL_PAGE,
        "template_id": "tutorial_page_basic",
        "show_page_number": False,
        "page_number_style": None,
        "payload": {
            "tutorial_code": "TUT-1",
            "tutorial_title": "Sudoku Tutorial 1: Hidden Singles",
            "body_title": "Procedure to find a hidden single",
            "bullets": [
                "Choose a house: row, column, or box.",
                "Follow one candidate digit through that house.",
                "Rule out impossible cells using row, column, and box conflicts.",
                "When only one seat survives, place the digit.",
            ],
            "footer_note": "A hidden single is often the cleanest doorway into a puzzle.",
        },
    },
    WARMUP_PAGE: {
        "page_type": WARMUP_PAGE,
        "template_id": "warmup_page_basic",
        "show_page_number": False,
        "page_number_style": None,
        "payload": {
            "warmup_code": "WU-1",
            "warmup_title": "Warm-up 1: Hidden Singles",
            "body": "Try a few short drills before starting the main book progression.",
            "prompts": [
                "Find one hidden single in a row.",
                "Find one hidden single in a column.",
                "Find one hidden single in a box.",
            ],
        },
    },
}

_SECTION_PRELUDE_PAGE_SPECS: Dict[str, dict] = {
    SECTION_OPENER_PAGE: {
        "page_type": SECTION_OPENER_PAGE,
        "template_id": "section_opener_basic",
        "show_page_number": False,
        "page_number_style": None,
    },
    SECTION_HIGHLIGHTS_PAGE: {
        "page_type": SECTION_HIGHLIGHTS_PAGE,
        "template_id": "section_highlights_basic",
        "show_page_number": False,
        "page_number_style": None,
    },
    SECTION_PATTERN_GALLERY_PAGE: {
        "page_type": SECTION_PATTERN_GALLERY_PAGE,
        "template_id": "section_pattern_gallery_basic",
        "show_page_number": False,
        "page_number_style": None,
    },
}


_FRONT_MATTER_PROFILES: Dict[str, List[dict]] = {
    "minimal_front_matter": [],
    "classic_with_welcome": [
        dict(_FRONT_MATTER_PAGE_SPECS[WELCOME_PAGE]),
    ],
    "classic_with_tutorials": [
        dict(_FRONT_MATTER_PAGE_SPECS[WELCOME_PAGE]),
        dict(_FRONT_MATTER_PAGE_SPECS[TOC_PAGE]),
        dict(_FRONT_MATTER_PAGE_SPECS[RULES_PAGE]),
        dict(_FRONT_MATTER_PAGE_SPECS[TUTORIAL_PAGE]),
        dict(_FRONT_MATTER_PAGE_SPECS[WARMUP_PAGE]),
    ],
    "classic_full_companion": [
        dict(_FRONT_MATTER_PAGE_SPECS[WELCOME_PAGE]),
        dict(_FRONT_MATTER_PAGE_SPECS[TOC_PAGE]),
        dict(_FRONT_MATTER_PAGE_SPECS[RULES_PAGE]),
        dict(_FRONT_MATTER_PAGE_SPECS[TUTORIAL_PAGE]),
        dict(_FRONT_MATTER_PAGE_SPECS[WARMUP_PAGE]),
    ],
}


_END_MATTER_PROFILES: Dict[str, List[dict]] = {
    "none": [],
    "review_request": [
        {
            "page_type": PROMO_PAGE,
            "template_id": "review_request_basic",
            "show_page_number": False,
            "page_number_style": None,
            "payload": {
                "title": "Enjoying the Challenge?",
                "body": (
                    "If this book sharpened your mind and gave you a satisfying challenge, "
                    "please consider leaving a short review and sharing it with other Sudoku fans."
                ),
            },
        }
    ],
    "upsell_series": [
        {
            "page_type": PROMO_PAGE,
            "template_id": "upsell_series_basic",
            "show_page_number": False,
            "page_number_style": None,
            "payload": {
                "title": "Continue the Journey",
                "body": (
                    "Explore more volumes in the Sudoku Companion library, with new difficulty bands, "
                    "fresh layouts, and additional solving support."
                ),
            },
        }
    ],
}


def get_front_matter_profile(profile_id: str) -> List[dict]:
    if profile_id not in _FRONT_MATTER_PROFILES:
        known = ", ".join(sorted(_FRONT_MATTER_PROFILES.keys()))
        raise KeyError(f"Unknown front matter profile '{profile_id}'. Known profiles: {known}")
    return [dict(item) for item in _FRONT_MATTER_PROFILES[profile_id]]


def get_end_matter_profile(profile_id: str) -> List[dict]:
    if profile_id not in _END_MATTER_PROFILES:
        known = ", ".join(sorted(_END_MATTER_PROFILES.keys()))
        raise KeyError(f"Unknown end matter profile '{profile_id}'. Known profiles: {known}")
    return [dict(item) for item in _END_MATTER_PROFILES[profile_id]]


def get_front_matter_page_spec(page_type: str) -> dict:
    key = str(page_type).strip()
    if key not in _FRONT_MATTER_PAGE_SPECS:
        known = ", ".join(sorted(_FRONT_MATTER_PAGE_SPECS.keys()))
        raise KeyError(f"Unknown front matter page type '{page_type}'. Known page types: {known}")
    return dict(_FRONT_MATTER_PAGE_SPECS[key])


def get_section_prelude_page_spec(page_type: str) -> dict:
    key = str(page_type).strip()
    if key not in _SECTION_PRELUDE_PAGE_SPECS:
        known = ", ".join(sorted(_SECTION_PRELUDE_PAGE_SPECS.keys()))
        raise KeyError(f"Unknown section prelude page type '{page_type}'. Known page types: {known}")
    return dict(_SECTION_PRELUDE_PAGE_SPECS[key])