from __future__ import annotations

from python.publishing.schemas.models import MarginProfile, PrintFormatSpec


def get_margin_profile(format_spec: PrintFormatSpec, mirrored: bool) -> MarginProfile:
    profile_id = f"{format_spec.format_id}__{'mirrored' if mirrored else 'flat'}"
    return MarginProfile(
        profile_id=profile_id,
        mirrored=mirrored,
        inside_margin_in=format_spec.inside_margin_in,
        outside_margin_in=format_spec.outside_margin_in,
        top_margin_in=format_spec.top_margin_in,
        bottom_margin_in=format_spec.bottom_margin_in,
        bleed_in=format_spec.bleed_in,
        safe_margin_in=format_spec.safe_margin_in,
    )