from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from python.publishing.cover_designs.models import ResolvedCoverDesignContext


@dataclass(frozen=True)
class CoverRenderResult:
    front_cover_png: Path
    width_px: int
    height_px: int
    renderer_key: str
    back_cover_png: Path | None = None
    spine_cover_png: Path | None = None


class BaseCoverRenderer:
    renderer_key: str = ""

    def render_front_cover(
        self,
        context: ResolvedCoverDesignContext,
        out_dir: str | Path,
        width_px: int = 2550,
        height_px: int = 3300,
    ) -> CoverRenderResult:
        raise NotImplementedError

    def render_back_cover(
        self,
        context: ResolvedCoverDesignContext,
        out_dir: str | Path,
        geometry: Any,
        width_px: int = 2550,
        height_px: int = 3300,
    ) -> Path | None:
        """Optionally render a custom back-panel image for full-wrap covers.

        Existing cover renderers can ignore this hook. When a renderer returns
        None, the generic full-wrap back panel remains in use.
        """
        return None

    def render_spine_cover(
        self,
        context: ResolvedCoverDesignContext,
        out_dir: str | Path,
        geometry: Any,
        width_px: int = 120,
        height_px: int = 3300,
    ) -> Path | None:
        """Optionally render a custom spine-panel image for full-wrap covers.

        Existing cover renderers can ignore this hook. When a renderer returns
        None, the generic full-wrap spine remains in use.
        """
        return None