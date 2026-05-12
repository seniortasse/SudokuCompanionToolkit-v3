from __future__ import annotations

from typing import Dict

from python.publishing.distribution.metadata_models import ImprintMetadata


_IMPRINTS: Dict[str, ImprintMetadata] = {
    "sudoku_companion": ImprintMetadata(
        imprint_id="sudoku_companion",
        imprint_name="Sudoku Companion",
        publisher_name="Sudoku Companion",
        website="",
        contact_email="",
    ),
    "contextionary": ImprintMetadata(
        imprint_id="contextionary",
        imprint_name="Contextionary",
        publisher_name="Contextionary",
        website="",
        contact_email="",
    ),
}


def get_imprint_metadata(imprint_id: str) -> ImprintMetadata:
    try:
        return _IMPRINTS[imprint_id]
    except KeyError as exc:
        known = ", ".join(sorted(_IMPRINTS.keys()))
        raise KeyError(f"Unknown imprint '{imprint_id}'. Known imprints: {known}") from exc