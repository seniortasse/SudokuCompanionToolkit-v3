from __future__ import annotations

from typing import Dict, List

from python.publishing.schemas.models import ChannelPreset


_CHANNELS: Dict[str, ChannelPreset] = {
    "amazon_kdp_paperback": ChannelPreset(
        channel_id="amazon_kdp_paperback",
        vendor="amazon_kdp",
        binding_type="paperback",
        description="Amazon KDP paperback production channel.",
        default_format_ids=[
            "amazon_kdp_paperback_6x9_bw",
            "amazon_kdp_paperback_8_5x11_bw",
            "amazon_kdp_paperback_8_5x11_color",
        ],
    ),
    "generic_pdf_print": ChannelPreset(
        channel_id="generic_pdf_print",
        vendor="generic",
        binding_type="paperback",
        description="Generic print-ready PDF channel for local or manual print workflows.",
        default_format_ids=[
            "amazon_kdp_paperback_8_5x11_bw",
        ],
    ),
}


def get_channel_preset(channel_id: str) -> ChannelPreset:
    try:
        return _CHANNELS[channel_id]
    except KeyError as exc:
        known = ", ".join(sorted(_CHANNELS.keys()))
        raise KeyError(f"Unknown channel '{channel_id}'. Known channels: {known}") from exc


def list_channel_presets() -> List[ChannelPreset]:
    return [_CHANNELS[key] for key in sorted(_CHANNELS.keys())]