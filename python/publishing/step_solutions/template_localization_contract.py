from __future__ import annotations

import re
from typing import Dict, Sequence, Set


LOCALIZATION_MASTER_SCHEMA_VERSION = "step_solution_localization_master.v1"

CANONICAL_TEMPLATE_SHEETS: Sequence[str] = (
    "Headers",
    "Messages",
    "Names",
    "Keywords",
)

SUPPORTED_LOCALIZATION_LOCALES: Sequence[str] = (
    "en",
    "fr",
    "de",
    "it",
    "es",
)

PLACEHOLDER_RE = re.compile(r"\{[^{}]*\}")

# These English strings are acceptable in localized files because they are
# internationally recognized Sudoku technique names or unavoidable placeholders.
ALLOWED_ENGLISH_TERMS_BY_LOCALE: Dict[str, Set[str]] = {
    "fr": {
        "X-Wing",
        "Swordfish",
        "Jellyfish",
        "Y-Wing",
        "XY-Chain",
        "XY-Ring",
        "Remote Pairs",
        "Pointing",
        "Claiming",
    },
    "de": {
        "X-Wing",
        "Swordfish",
        "Jellyfish",
        "Y-Wing",
        "XY-Chain",
        "XY-Ring",
        "Pointing",
        "Claiming",
    },
    "it": {
        "X-Wing",
        "Swordfish",
        "Jellyfish",
        "Y-Wing",
        "XY-Chain",
        "XY-Ring",
        "Pointing",
        "Claiming",
    },
    "es": {
        "X-Wing",
        "Swordfish",
        "Jellyfish",
        "Y-Wing",
        "XY-Chain",
        "XY-Ring",
        "Pointing",
        "Claiming",
    },
}


# These are English words that can naturally appear inside non-English words
# or as intentionally retained international Sudoku technique vocabulary.
#
# The auditor should not flag these by substring. It should only flag them when
# they are clear standalone English leakage, and even then only outside
# placeholders/machine keys.
ALLOWED_SUBSTRING_FALSE_POSITIVES_BY_LOCALE: Dict[str, Set[str]] = {
    "fr": set(),
    "de": set(),
    "it": set(),
    "es": {
        # Spanish words:
        #   celda/celdas are not "cell"
        #   caja is not "box"
        #   columna is not "column"
        #   horizontalmente/verticalmente are valid Spanish
        "cell",
        "box",
        "column",
        "horizontal",
        "vertical",
    },
}

ENGLISH_LEAKAGE_TERMS: Sequence[str] = (
    # Phrases first. These are strong signals.
    "Looking at",
    "There is",
    "The other cells",
    "cannot contain",
    "because",
    "Therefore",
    "can be removed",
    "is the only missing value",
    "single position",

    # Single words. These require safer word-boundary matching.
    "Cells",
    "cell",
    "row",
    "column",
    "box",
    "contains",
    "contain",
    "candidates",
    "candidate",
    "positions",
    "position",
    "horizontal",
    "vertical",
    "Easy",
    "Medium",
    "Hard",
)

# Known accidental cross-language leakage terms we want to catch.
FOREIGN_LEAKAGE_TERMS_BY_LOCALE: Dict[str, Sequence[str]] = {
    "fr": (),
    "de": (
        "Facile",
        "Moyen",
        "Difficile",
        "Fácil",
        "Difícil",
        "Réduction",
        "Griffe",
    ),
    "it": (
        "Réduction",
        "Griffe",
        "Facile",
        "Moyen",
        "Difícil",
        "Fácil",
    ),
    "es": (
        "Réduction",
        "Griffe",
        "Moyen",
        "Difficile",
        "Facile",
    ),
}


def extract_placeholders(text: str) -> Set[str]:
    """
    Extract placeholders from a template string.

    Examples:
        "{dim}, {char}, {cell}" -> {"{dim}", "{char}", "{cell}"}
        "So {}. " -> {"{}"}
    """

    return set(PLACEHOLDER_RE.findall(str(text or "")))


def normalize_template_text(value: object) -> str:
    """
    Normalize a workbook cell value for comparison/reporting.
    """

    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def allowed_english_terms_for_locale(locale: str) -> Set[str]:
    return set(ALLOWED_ENGLISH_TERMS_BY_LOCALE.get(str(locale).lower(), set()))




def allowed_substring_false_positives_for_locale(locale: str) -> Set[str]:
    return set(
        ALLOWED_SUBSTRING_FALSE_POSITIVES_BY_LOCALE.get(
            str(locale).lower(),
            set(),
        )
    )


def strip_placeholders(text: str) -> str:
    """
    Remove placeholders before natural-language leakage checks.

    Without this, strings like:
        {cell}
        {box}
        {row}
    would be falsely flagged as English leakage.
    """

    return PLACEHOLDER_RE.sub(" ", str(text or ""))