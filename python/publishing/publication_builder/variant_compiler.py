from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

from python.publishing.publication_builder.layout_presets import (
    list_layout_preset_ids,
    resolve_layout_preset,
)
from python.publishing.publication_builder.spec_overrides import (
    apply_publication_spec_overrides,
    load_publication_spec_dict,
)
from python.publishing.publication_builder.translation_overlay import (
    apply_locale_pack_overrides,
    infer_locale_pack_path,
    load_locale_pack,
)
from python.publishing.publication_builder.variant_id_builder import build_variant_publication_id
from python.publishing.publication_builder.variant_models import PublicationVariantRequest


_LANGUAGE_LABEL_MAP = {
    "en": "English",
    "english": "English",
    "de": "German",
    "german": "German",
    "fr": "French",
    "french": "French",
    "it": "Italian",
    "italian": "Italian",
    "es": "Spanish",
    "spanish": "Spanish",
    "pt": "Portuguese",
    "portuguese": "Portuguese",
    "nl": "Dutch",
    "dutch": "Dutch",
}


def _resolve_language_label(locale: Optional[str], language: Optional[str], fallback: str) -> str:
    if language:
        return str(language)

    raw = str(locale or "").strip()
    if not raw:
        return fallback

    key = raw.lower()
    if key in _LANGUAGE_LABEL_MAP:
        return _LANGUAGE_LABEL_MAP[key]

    return raw


def compile_publication_variant_spec(
    request: PublicationVariantRequest,
) -> Dict[str, Any]:
    base_spec_path = Path(request.base_spec_path)
    base_spec = load_publication_spec_dict(base_spec_path)
    out = deepcopy(base_spec)

    locale_code = str(request.locale or "").strip().lower()

    locale_pack_path = request.locale_pack_path
    if locale_pack_path is None:
        inferred = infer_locale_pack_path(
            base_spec_path=base_spec_path,
            book_id=str(base_spec["book_id"]),
            locale=locale_code,
        )
        locale_pack_path = inferred

    locale_pack = None
    if locale_pack_path is not None and Path(locale_pack_path).exists():
        locale_pack = load_locale_pack(Path(locale_pack_path))
        out = apply_locale_pack_overrides(out, locale_pack=locale_pack)
    elif locale_code and locale_code not in {"en", "english"}:
        raise FileNotFoundError(
            f"No locale pack found for locale={locale_code!r}. "
            f"Expected: {locale_pack_path}"
        )

    resolved_preset = resolve_layout_preset(
        layout_preset_id=request.layout_preset_id,
        puzzles_per_page=request.puzzles_per_page,
    )

    if request.layout_preset_id and resolved_preset is None:
        raise ValueError(
            f"Unsupported layout preset '{request.layout_preset_id}'. "
            f"Supported presets: {', '.join(list_layout_preset_ids())}"
        )

    resolved_puzzles_per_page = (
        int(resolved_preset.puzzles_per_page)
        if resolved_preset is not None
        else request.puzzles_per_page
    )

    resolved_rows = (
        int(request.rows)
        if request.rows is not None
        else (int(resolved_preset.rows) if resolved_preset is not None else None)
    )
    resolved_cols = (
        int(request.cols)
        if request.cols is not None
        else (int(resolved_preset.cols) if resolved_preset is not None else None)
    )

    out = apply_publication_spec_overrides(
        out,
        layout_preset_id=(resolved_preset.preset_id if resolved_preset is not None else None),
        puzzles_per_page=resolved_puzzles_per_page,
        rows=resolved_rows,
        cols=resolved_cols,
        font_family=request.font_family,
    )

    current_language = str((out.get("layout_config") or {}).get("language") or "en")
    language_label = _resolve_language_label(
        locale_code or current_language,
        request.language or (str(locale_pack.get("language_label")) if locale_pack else None),
        str((out.get("metadata") or {}).get("language") or "English"),
    )

    layout = dict(out.get("layout_config") or {})
    metadata = dict(out.get("metadata") or {})

    if locale_code:
        layout["language"] = locale_code
        metadata["locale"] = locale_code

    metadata["language"] = language_label

    out["layout_config"] = layout
    out["metadata"] = metadata

    out["publication_id"] = build_variant_publication_id(
        base_spec=base_spec,
        locale=locale_code or request.language,
        puzzles_per_page=resolved_puzzles_per_page,
        explicit_publication_id=request.publication_id,
    )

    effective_locale = str(
        locale_code
        or layout.get("language")
        or metadata.get("locale")
        or ""
    ).strip().lower()

    normalized_locale = effective_locale.split("-")[0].split("_")[0] if effective_locale else ""

    base_editorial_copy = dict(base_spec.get("editorial_copy") or {})
    base_ending = dict(base_editorial_copy.get("ending") or {})
    base_qr_path = str(base_ending.get("support_qr_image_path") or "").strip()


    # Force QR image path to follow the compiled book + locale.
    # Example: BK-CL9-DW-B04 + en -> assets/qr/solutionB04.en.png
    book_code = ""
    raw_book_id = str(out.get("book_id") or base_spec.get("book_id") or "").strip().upper()
    for part in reversed([p for p in raw_book_id.split("-") if p]):
        if part.startswith("B") and part[1:].isdigit():
            book_code = part
            break

    if book_code and normalized_locale:
        editorial_copy = dict(out.get("editorial_copy") or {})
        ending = dict(editorial_copy.get("ending") or {})
        ending["support_qr_image_path"] = f"assets/qr/solution{book_code}.{normalized_locale}.png"
        editorial_copy["ending"] = ending
        out["editorial_copy"] = editorial_copy

    if base_qr_path and normalized_locale and normalized_locale not in {"en", "english"}:
        editorial_copy = dict(out.get("editorial_copy") or {})
        ending = dict(editorial_copy.get("ending") or {})

        # If the locale pack already supplied an explicit QR path, keep it.
        # Example: assets/qr/solutionB02.de.png
        existing_qr_path = str(ending.get("support_qr_image_path") or "").strip()
        if not existing_qr_path or existing_qr_path == base_qr_path:
            qr_path = Path(base_qr_path)

            # Base English specs may already use a localized filename such as
            # solutionB02.en.png. When compiling German, French, Spanish, or
            # Italian, we want solutionB02.de.png, not solutionB02.en.de.png.
            stem = qr_path.stem
            for suffix_locale in ("en", "english", "fr", "de", "es", "it"):
                suffix = f".{suffix_locale}"
                if stem.lower().endswith(suffix):
                    stem = stem[: -len(suffix)]
                    break

            localized_qr_path = qr_path.with_name(
                f"{stem}.{normalized_locale}{qr_path.suffix}"
            ).as_posix()

            ending["support_qr_image_path"] = localized_qr_path
            editorial_copy["ending"] = ending
            out["editorial_copy"] = editorial_copy

    return out