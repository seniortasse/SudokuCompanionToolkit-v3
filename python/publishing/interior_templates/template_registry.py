from __future__ import annotations

from python.publishing.interior_templates.front_matter_profiles import (
    get_end_matter_profile,
    get_front_matter_page_spec,
    get_front_matter_profile,
    get_section_prelude_page_spec,
)


def resolve_front_matter_profile(profile_id: str):
    return get_front_matter_profile(profile_id)


def resolve_end_matter_profile(profile_id: str):
    return get_end_matter_profile(profile_id)


def resolve_front_matter_page_spec(page_type: str):
    return get_front_matter_page_spec(page_type)


def resolve_section_prelude_page_spec(page_type: str):
    return get_section_prelude_page_spec(page_type)