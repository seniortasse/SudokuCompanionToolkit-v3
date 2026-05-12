from __future__ import annotations


_PAPER_THICKNESS_BY_TYPE = {
    "white_bw": 0.002252,
    "cream_bw": 0.0025,
    "white_color": 0.002347,
    "premium_color": 0.002347,
    "standard_color": 0.002347,
}


def compute_spine_width_in(page_count: int, paper_type: str, channel_id: str) -> float:
    if page_count <= 0:
        return 0.0

    thickness = _PAPER_THICKNESS_BY_TYPE.get(paper_type, _PAPER_THICKNESS_BY_TYPE["white_bw"])
    return round(page_count * thickness, 6)