from .book_manifest_builder import BuiltBook, build_book_from_spec
from .book_spec_loader import BookSpec, BookSectionSpec, load_book_spec
from .dedupe import build_reuse_blocklist, filter_reusable_puzzles
from .ordering import order_section_puzzles
from .puzzle_selector import select_puzzles_for_section
from .section_allocator import allocate_sections
from .book_spec_loader import BookSectionSpec, BookSpec, load_book_spec
from .capacity_analyzer import analyze_book_capacity, explain_capacity_failure
from .dw_presets import build_dw_preset_book_spec, list_dw_presets

__all__ = [
    "BuiltBook",
    "build_book_from_spec",
    "BookSpec",
    "BookSectionSpec",
    "load_book_spec",
    "build_reuse_blocklist",
    "filter_reusable_puzzles",
    "order_section_puzzles",
    "select_puzzles_for_section",
    "allocate_sections",
    "analyze_book_capacity",
    "explain_capacity_failure",
    "build_dw_preset_book_spec",
    "list_dw_presets",
]