from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(dict(result[key]), value)
        else:
            result[key] = value
    return result


def load_cover_variant_preset(
    *,
    preset_path: Path | None,
    locale: str,
    layout: str,
) -> dict[str, Any]:
    if preset_path is None:
        return {}

    data = json.loads(preset_path.read_text(encoding="utf-8"))

    default_cover = dict(data.get("default") or {})
    variants = list(data.get("variants") or [])

    locale_norm = str(locale).lower()
    layout_norm = str(layout).lower()

    for item in variants:
        if str(item.get("locale", "")).lower() != locale_norm:
            continue
        if str(item.get("layout", "")).lower() != layout_norm:
            continue

        selected = dict(item)
        selected.pop("locale", None)
        selected.pop("layout", None)
        return selected

    selected = dict(default_cover)
    selected.pop("locale", None)
    selected.pop("layout", None)
    return selected