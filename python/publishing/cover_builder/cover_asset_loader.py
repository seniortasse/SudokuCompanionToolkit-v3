from __future__ import annotations

from pathlib import Path
from typing import Optional


def resolve_cover_asset_path(publication_dir: Path, asset_value: str | None) -> Optional[Path]:
    raw = str(asset_value or "").strip()
    if not raw:
        return None

    candidate = Path(raw)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    local = publication_dir / raw
    if local.exists():
        return local

    return None