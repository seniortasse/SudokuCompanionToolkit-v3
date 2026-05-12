from __future__ import annotations

"""
Step-by-step solution package generation for Sudoku publishing.

This package owns the book/language solution-export pipeline:

- commercial/user-facing puzzle codes such as L-1-1
- commercial image names such as B01-L-1-1_step1.png
- package folders under datasets/sudoku_books/classic9/step_solution_packages
- internal manifest files for validation and traceability

The first implementation phase is intentionally infrastructure-only.
Generation of Excel logs, images, and sudokuIndexFile.csv is added in later phases.
"""


__all__ = [
    "apply_localization_seed",
    "book_loader",
    "csv_index_writer",
    "excel_image_exporter",
    "full_book_readiness",
    "identity",
    "locale_templates",
    "localization_master",
    "localization_translation_seed",
    "log_generator",
    "models",
    "package_exporter",
    "package_manifest",
    "paths",
    "puzzle_instance_adapter",
    "runtime_qa",
    "template_auditor",
    "template_generator",
    "template_localization_contract",
    "template_reader",
]