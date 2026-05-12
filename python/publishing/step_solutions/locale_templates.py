from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

from python.publishing.step_solutions.models import SUPPORTED_STEP_SOLUTION_LOCALES


DEFAULT_SOLUTION_TEMPLATES_ROOT = Path(
    "datasets/sudoku_books/classic9/solution_templates"
)

VISUAL_TEMPLATE_FILENAME = "Template.xlsx"
MESSAGE_TEMPLATE_FILENAME = "Template_Messages.xlsx"


_LOCALE_ALIASES: Dict[str, str] = {
    "en": "en",
    "eng": "en",
    "english": "en",
    "fr": "fr",
    "fre": "fr",
    "fra": "fr",
    "french": "fr",
    "de": "de",
    "deu": "de",
    "ger": "de",
    "german": "de",
    "it": "it",
    "ita": "it",
    "italian": "it",
    "es": "es",
    "spa": "es",
    "sp": "es",
    "spanish": "es",
}

_LEGACY_LANGUAGE_DIR_BY_LOCALE: Dict[str, str] = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "es": "Spanish",
}


@dataclass(frozen=True)
class StepSolutionTemplatePaths:
    """
    Canonical template paths for one step-solution locale.

    visual_template_path:
        Shared workbook visual layout used to create the user_logs workbook.

    message_template_path:
        Locale-specific Template_Messages.xlsx used to generate localized
        solving narratives.
    """

    locale: str
    templates_root: Path
    visual_template_path: Path
    message_template_path: Path

    def to_dict(self) -> Dict[str, str]:
        payload = asdict(self)
        return {key: str(value) for key, value in payload.items()}


@dataclass(frozen=True)
class StepSolutionTemplateStatus:
    """
    Validation status for one locale's canonical templates.
    """

    locale: str
    visual_template_path: Path
    message_template_path: Path
    visual_template_exists: bool
    message_template_exists: bool

    @property
    def ok(self) -> bool:
        return self.visual_template_exists and self.message_template_exists

    def to_dict(self) -> Dict[str, object]:
        return {
            "locale": self.locale,
            "visual_template_path": str(self.visual_template_path),
            "message_template_path": str(self.message_template_path),
            "visual_template_exists": self.visual_template_exists,
            "message_template_exists": self.message_template_exists,
            "ok": self.ok,
        }


def normalize_step_solution_locale(locale: str) -> str:
    """
    Normalize a user-provided locale/language value to the internal locale code.

    Supported canonical locale codes:
        en, fr, de, it, es

    Accepted aliases include:
        english -> en
        french  -> fr
        german  -> de
        italian -> it
        spanish -> es
        sp      -> es
    """

    key = str(locale or "").strip().lower().replace("_", "-")
    key = key.split("-", 1)[0]

    normalized = _LOCALE_ALIASES.get(key)
    if not normalized:
        raise ValueError(
            f"Unsupported step-solution locale {locale!r}. "
            f"Supported locales: {', '.join(SUPPORTED_STEP_SOLUTION_LOCALES)}."
        )

    if normalized not in SUPPORTED_STEP_SOLUTION_LOCALES:
        raise ValueError(
            f"Locale alias {locale!r} normalized to {normalized!r}, "
            "but that locale is not registered as supported."
        )

    return normalized


def supported_step_solution_locales() -> List[str]:
    """
    Return the canonical supported locales in stable order.
    """

    return list(SUPPORTED_STEP_SOLUTION_LOCALES)


def visual_template_path(
    templates_root: Path = DEFAULT_SOLUTION_TEMPLATES_ROOT,
) -> Path:
    """
    Resolve the canonical shared visual workbook template path.
    """

    return Path(templates_root) / "visual" / VISUAL_TEMPLATE_FILENAME


def message_template_path(
    locale: str,
    templates_root: Path = DEFAULT_SOLUTION_TEMPLATES_ROOT,
) -> Path:
    """
    Resolve the canonical locale-specific Template_Messages.xlsx path.
    """

    normalized_locale = normalize_step_solution_locale(locale)
    return (
        Path(templates_root)
        / "messages"
        / normalized_locale
        / MESSAGE_TEMPLATE_FILENAME
    )


def resolve_solution_template_paths(
    locale: str,
    templates_root: Path = DEFAULT_SOLUTION_TEMPLATES_ROOT,
    require_exists: bool = True,
) -> StepSolutionTemplatePaths:
    """
    Resolve the canonical visual and message templates for one locale.

    This is the function later workflow phases should call.
    """

    normalized_locale = normalize_step_solution_locale(locale)

    paths = StepSolutionTemplatePaths(
        locale=normalized_locale,
        templates_root=Path(templates_root),
        visual_template_path=visual_template_path(templates_root),
        message_template_path=message_template_path(
            normalized_locale,
            templates_root,
        ),
    )

    if require_exists:
        missing = []
        if not paths.visual_template_path.exists():
            missing.append(str(paths.visual_template_path))
        if not paths.message_template_path.exists():
            missing.append(str(paths.message_template_path))

        if missing:
            raise FileNotFoundError(
                "Missing step-solution template file(s):\n"
                + "\n".join(f"  - {path}" for path in missing)
                + "\n\nRun:\n"
                + "  python -m python.publishing.workflows.prepare_step_solution_templates\n"
                + "or copy the templates into datasets/sudoku_books/classic9/solution_templates/."
            )

    return paths


def ensure_solution_template_directories(
    templates_root: Path = DEFAULT_SOLUTION_TEMPLATES_ROOT,
    locales: Optional[Iterable[str]] = None,
) -> None:
    """
    Create the canonical template folder structure.
    """

    root = Path(templates_root)
    (root / "visual").mkdir(parents=True, exist_ok=True)

    for locale in locales or SUPPORTED_STEP_SOLUTION_LOCALES:
        normalized_locale = normalize_step_solution_locale(locale)
        (root / "messages" / normalized_locale).mkdir(parents=True, exist_ok=True)


def template_status_for_locale(
    locale: str,
    templates_root: Path = DEFAULT_SOLUTION_TEMPLATES_ROOT,
) -> StepSolutionTemplateStatus:
    """
    Return existence status for one locale's canonical templates.
    """

    normalized_locale = normalize_step_solution_locale(locale)
    visual_path = visual_template_path(templates_root)
    message_path = message_template_path(normalized_locale, templates_root)

    return StepSolutionTemplateStatus(
        locale=normalized_locale,
        visual_template_path=visual_path,
        message_template_path=message_path,
        visual_template_exists=visual_path.exists(),
        message_template_exists=message_path.exists(),
    )


def build_template_status_report(
    templates_root: Path = DEFAULT_SOLUTION_TEMPLATES_ROOT,
    locales: Optional[Iterable[str]] = None,
) -> Dict[str, object]:
    """
    Build a small validation report for all canonical step-solution templates.
    """

    normalized_locales = [
        normalize_step_solution_locale(locale)
        for locale in (locales or SUPPORTED_STEP_SOLUTION_LOCALES)
    ]

    statuses = [
        template_status_for_locale(locale, templates_root)
        for locale in normalized_locales
    ]

    return {
        "templates_root": str(Path(templates_root)),
        "visual_template_filename": VISUAL_TEMPLATE_FILENAME,
        "message_template_filename": MESSAGE_TEMPLATE_FILENAME,
        "locales": [status.to_dict() for status in statuses],
        "ok": all(status.ok for status in statuses),
    }


def legacy_template_source_paths(
    legacy_root: Path = Path("python/step-by-step_solutions"),
) -> Mapping[str, Path]:
    """
    Return the active legacy template source paths selected for migration.

    The selected sources intentionally ignore older/root duplicates such as:
        - Template_Messages.xlsx
        - Template_Messages - EN.xlsx
        - Template_Messages_ORIGINAL ENGLISH.xlsx
    """

    root = Path(legacy_root)

    paths: Dict[str, Path] = {
        "visual": root / VISUAL_TEMPLATE_FILENAME,
    }

    for locale, language_dir in _LEGACY_LANGUAGE_DIR_BY_LOCALE.items():
        paths[locale] = (
            root
            / "Template Messages"
            / language_dir
            / MESSAGE_TEMPLATE_FILENAME
        )

    return paths


def copy_legacy_templates_to_canonical(
    legacy_root: Path = Path("python/step-by-step_solutions"),
    templates_root: Path = DEFAULT_SOLUTION_TEMPLATES_ROOT,
    overwrite: bool = False,
) -> Dict[str, object]:
    """
    Copy the active legacy templates into the canonical dataset template layout.

    This is a safe bootstrap utility:
        - creates missing canonical folders
        - copies selected active legacy templates
        - does not delete legacy files
        - does not overwrite existing canonical files unless overwrite=True
    """

    ensure_solution_template_directories(templates_root)

    sources = legacy_template_source_paths(legacy_root)
    copied: List[Dict[str, str]] = []
    skipped: List[Dict[str, str]] = []
    missing_sources: List[str] = []

    visual_dst = visual_template_path(templates_root)
    visual_src = sources["visual"]

    _copy_one_template(
        src=visual_src,
        dst=visual_dst,
        overwrite=overwrite,
        copied=copied,
        skipped=skipped,
        missing_sources=missing_sources,
    )

    for locale in SUPPORTED_STEP_SOLUTION_LOCALES:
        src = sources[locale]
        dst = message_template_path(locale, templates_root)

        _copy_one_template(
            src=src,
            dst=dst,
            overwrite=overwrite,
            copied=copied,
            skipped=skipped,
            missing_sources=missing_sources,
        )

    return {
        "legacy_root": str(Path(legacy_root)),
        "templates_root": str(Path(templates_root)),
        "overwrite": overwrite,
        "copied": copied,
        "skipped": skipped,
        "missing_sources": missing_sources,
        "status_report": build_template_status_report(templates_root),
    }


def _copy_one_template(
    src: Path,
    dst: Path,
    overwrite: bool,
    copied: List[Dict[str, str]],
    skipped: List[Dict[str, str]],
    missing_sources: List[str],
) -> None:
    if not src.exists():
        missing_sources.append(str(src))
        return

    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() and not overwrite:
        skipped.append(
            {
                "source": str(src),
                "destination": str(dst),
                "reason": "destination_exists",
            }
        )
        return

    shutil.copy2(src, dst)
    copied.append(
        {
            "source": str(src),
            "destination": str(dst),
        }
    )