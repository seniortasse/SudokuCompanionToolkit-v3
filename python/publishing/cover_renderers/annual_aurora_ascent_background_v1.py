from __future__ import annotations

from pathlib import Path

from PIL import Image

from python.publishing.cover_designs.models import ResolvedCoverDesignContext
from python.publishing.cover_renderers.annual_expert_gauge_background_v1 import (
    AnnualExpertGaugeBackgroundV1Renderer,
    _get_background_variables,
    _resolve_asset_path,
)
from python.publishing.cover_renderers.base_renderer import CoverRenderResult


class AnnualAuroraAscentBackgroundV1Renderer(AnnualExpertGaugeBackgroundV1Renderer):
    """
    Aurora Ascent Easy-to-Expert renderer.

    Front:
      - Uses a complete baked front-cover image.
      - Does not redraw front text.
      - Does not redraw Sudoku digits.

    Back/spine:
      - Reuses the proven generated back/spine logic from
        AnnualExpertGaugeBackgroundV1Renderer.
      - The Aurora Ascent identity comes from cover_design.json palette variables.
    """

    renderer_key = "annual_aurora_ascent_background_v1"

    def render_front_cover(
        self,
        context: ResolvedCoverDesignContext,
        out_dir: str | Path,
        width_px: int = 2550,
        height_px: int = 3300,
    ) -> CoverRenderResult:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        background = _get_background_variables(context)
        asset_path = str(
            background.get(
                "asset_path",
                "assets/backgrounds/cover_easy_to_expert_3.png",
            )
        )
        background_path = _resolve_asset_path(context, asset_path)

        img = Image.open(background_path).convert("RGBA")

        # The source artwork is the complete 8.5 x 11 front cover.
        # Direct resize preserves the full baked design and avoids accidental cropping.
        img = img.resize((width_px, height_px), Image.Resampling.LANCZOS)

        output_file = out_path / "front_cover.png"
        img.convert("RGB").save(output_file, quality=95)

        return CoverRenderResult(
            front_cover_png=output_file,
            width_px=width_px,
            height_px=height_px,
            renderer_key=self.renderer_key,
        )