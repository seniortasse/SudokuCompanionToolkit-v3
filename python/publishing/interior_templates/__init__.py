from .front_matter_profiles import (
    get_end_matter_profile,
    get_front_matter_page_spec,
    get_front_matter_profile,
    get_section_prelude_page_spec,
)
from .template_registry import (
    resolve_end_matter_profile,
    resolve_front_matter_page_spec,
    resolve_front_matter_profile,
    resolve_section_prelude_page_spec,
)

__all__ = [
    "get_front_matter_profile",
    "get_end_matter_profile",
    "get_front_matter_page_spec",
    "get_section_prelude_page_spec",
    "resolve_front_matter_profile",
    "resolve_end_matter_profile",
    "resolve_front_matter_page_spec",
    "resolve_section_prelude_page_spec",
]