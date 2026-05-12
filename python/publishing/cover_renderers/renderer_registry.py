from __future__ import annotations

from python.publishing.cover_renderers.annual_arena_blue_multigrid_v1 import (
    AnnualArenaBlueMultiGridV1Renderer,
)
from python.publishing.cover_renderers.annual_expert_gauge_background_v1 import (
    AnnualExpertGaugeBackgroundV1Renderer,
)
from python.publishing.cover_renderers.annual_emerald_easy_hard_background_v1 import (
    AnnualEmeraldEasyHardBackgroundV1Renderer,
)
from python.publishing.cover_renderers.base_renderer import BaseCoverRenderer
from python.publishing.cover_renderers.minimal_white_grid_v1 import (
    MinimalWhiteGridV1Renderer,
)

from python.publishing.cover_renderers.annual_aurora_ascent_background_v1 import (
    AnnualAuroraAscentBackgroundV1Renderer,
)


def get_cover_renderer(renderer_key: str) -> BaseCoverRenderer:
    renderers: dict[str, BaseCoverRenderer] = {
        AnnualArenaBlueMultiGridV1Renderer.renderer_key: AnnualArenaBlueMultiGridV1Renderer(),
        AnnualExpertGaugeBackgroundV1Renderer.renderer_key: AnnualExpertGaugeBackgroundV1Renderer(),
        AnnualEmeraldEasyHardBackgroundV1Renderer.renderer_key: AnnualEmeraldEasyHardBackgroundV1Renderer(),
        MinimalWhiteGridV1Renderer.renderer_key: MinimalWhiteGridV1Renderer(),
        AnnualAuroraAscentBackgroundV1Renderer.renderer_key: AnnualAuroraAscentBackgroundV1Renderer(),
    }

    try:
        return renderers[renderer_key]
    except KeyError as exc:
        available = ", ".join(sorted(renderers.keys()))
        raise KeyError(
            f"No cover renderer registered for renderer_key={renderer_key!r}. "
            f"Available renderers: {available}"
        ) from exc