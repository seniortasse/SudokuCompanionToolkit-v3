from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional


_ALLOWED_ROOTS = {
    "metadata",
    "editorial_copy",
    "features_page_config",
    "section_preview_config",
    "ecosystem_config",
    "cover_design",
    "solution_section_config",
}


def load_locale_pack(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Locale pack not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Locale pack must be a JSON object: {path}")
    return data


def infer_locale_pack_path(
    *,
    base_spec_path: Path,
    book_id: str,
    locale: Optional[str],
) -> Optional[Path]:
    raw = str(locale or "").strip().lower()
    if not raw:
        return None

    return (
        base_spec_path.resolve().parent.parent
        / "publication_locales"
        / f"{book_id}.{raw}.json"
    )


def _deep_merge(base: Any, overlay: Any) -> Any:
    if isinstance(base, dict) and isinstance(overlay, dict):
        out = deepcopy(base)
        for key, value in overlay.items():
            if key in out:
                out[key] = _deep_merge(out[key], value)
            else:
                out[key] = deepcopy(value)
        return out

    return deepcopy(overlay)


def apply_locale_pack_overrides(
    spec: Dict[str, Any],
    *,
    locale_pack: Dict[str, Any],
) -> Dict[str, Any]:
    out = deepcopy(spec)

    for root in _ALLOWED_ROOTS:
        if root in locale_pack:
            existing = out.get(root, {})
            out[root] = _deep_merge(existing, locale_pack[root])

    locale_code = str(locale_pack.get("locale") or "").strip().lower()
    language_label = str(locale_pack.get("language_label") or "").strip()

    layout = dict(out.get("layout_config") or {})
    metadata = dict(out.get("metadata") or {})

    if locale_code:
        layout["language"] = locale_code
        metadata["locale"] = locale_code

    if language_label:
        metadata["language"] = language_label

    out["layout_config"] = layout
    out["metadata"] = metadata
    return out