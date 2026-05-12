"""
Publishing subsystem for Sudoku Companion.

This package owns canonical content schemas, ids, validation,
pattern library ingestion, puzzle record building, catalog assembly,
book building, publication packaging, app catalog export, and PDF rendering.
"""

__all__ = [
    "schemas",
    "ids",
    "pattern_library",
    "puzzle_catalog",
    "difficulty",
    "app_catalog_builder",
    "book_builder",
    "publication_builder",
    "print_specs",
    "interior_builder",
    "interior_templates",
    "cover_builder",
    "distribution",
    "pdf_renderer",
    "cleanup",
    "qc",
]