from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CoverDesignCatalogEntry:
    cover_design_id: str
    name: str
    family: str
    renderer_key: str
    status: str
    design_dir: str
    supported_trim_sizes: list[str] = field(default_factory=list)
    supported_channels: list[str] = field(default_factory=list)
    default_palette_id: str | None = None
    default_texture_id: str | None = None
    preview_asset: str | None = None


@dataclass(frozen=True)
class CoverDesignRecord:
    cover_design_id: str
    name: str
    family: str
    renderer_key: str
    status: str
    description: str = ""
    identity: dict[str, Any] = field(default_factory=dict)
    supported_outputs: dict[str, Any] = field(default_factory=dict)
    supported_trim_sizes: list[str] = field(default_factory=list)
    supported_channels: list[str] = field(default_factory=list)
    editable_variables: dict[str, Any] = field(default_factory=dict)
    default_variables: dict[str, Any] = field(default_factory=dict)
    layout_regions: dict[str, Any] = field(default_factory=dict)
    assets: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CoverDesignAssignment:
    cover_design_id: str
    variables: dict[str, Any] = field(default_factory=dict)
    assignment_source: str = "manual"


@dataclass(frozen=True)
class ResolvedCoverDesignContext:
    cover_design_id: str
    name: str
    family: str
    renderer_key: str
    status: str
    variables: dict[str, Any]
    editable_variables: dict[str, Any]
    identity: dict[str, Any]
    layout_regions: dict[str, Any]
    assets: dict[str, Any]
    design_dir: Path
    catalog_path: Path
    assignment_source: str = "manual"