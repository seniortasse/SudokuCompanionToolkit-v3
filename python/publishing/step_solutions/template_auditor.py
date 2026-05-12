from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from python.publishing.step_solutions.locale_templates import (
    DEFAULT_SOLUTION_TEMPLATES_ROOT,
    message_template_path,
    normalize_step_solution_locale,
)
from python.publishing.step_solutions.template_localization_contract import (
    CANONICAL_TEMPLATE_SHEETS,
    ENGLISH_LEAKAGE_TERMS,
    FOREIGN_LEAKAGE_TERMS_BY_LOCALE,
    allowed_english_terms_for_locale,
    allowed_substring_false_positives_for_locale,
    extract_placeholders,
    strip_placeholders,
)
from python.publishing.step_solutions.template_reader import (
    StepSolutionTemplateWorkbook,
    read_step_solution_template_workbook,
)


@dataclass(frozen=True)
class TemplateAuditIssue:
    locale: str
    severity: str
    issue_type: str
    sheet: str
    key: str
    message: str
    row: Optional[int] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TemplateAuditLocaleReport:
    locale: str
    template_path: Path
    issue_count: int = 0
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    issues: List[TemplateAuditIssue] = field(default_factory=list)

    def add_issue(self, issue: TemplateAuditIssue) -> None:
        self.issues.append(issue)
        self.issue_count += 1
        if issue.severity == "error":
            self.error_count += 1
        elif issue.severity == "warning":
            self.warning_count += 1
        else:
            self.info_count += 1

    def ok(self, strict: bool = False) -> bool:
        if self.error_count:
            return False

        if strict:
            high_risk_warning_types = {
                "empty_message_translation",
                "empty_technique_name",
                "empty_keyword_values",
                "rating_still_english",
            }
            if any(
                issue.severity == "warning"
                and issue.issue_type in high_risk_warning_types
                for issue in self.issues
            ):
                return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        issue_type_counts = Counter(issue.issue_type for issue in self.issues)
        severity_counts = Counter(issue.severity for issue in self.issues)

        return {
            "locale": self.locale,
            "template_path": str(self.template_path),
            "issue_count": self.issue_count,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "issue_type_counts": dict(sorted(issue_type_counts.items())),
            "severity_counts": dict(sorted(severity_counts.items())),
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass
class TemplateAuditReport:
    english_template_path: Path
    templates_root: Path
    strict: bool
    locales: List[TemplateAuditLocaleReport]

    def ok(self) -> bool:
        return all(locale_report.ok(strict=self.strict) for locale_report in self.locales)

    def to_dict(self) -> Dict[str, Any]:
        all_issues = [
            issue
            for locale_report in self.locales
            for issue in locale_report.issues
        ]
        issue_type_counts = Counter(issue.issue_type for issue in all_issues)
        severity_counts = Counter(issue.severity for issue in all_issues)

        return {
            "schema_version": "step_solution_template_audit.v1",
            "english_template_path": str(self.english_template_path),
            "templates_root": str(self.templates_root),
            "strict": self.strict,
            "ok": self.ok(),
            "summary": {
                "locale_count": len(self.locales),
                "issue_count": sum(report.issue_count for report in self.locales),
                "error_count": sum(report.error_count for report in self.locales),
                "warning_count": sum(report.warning_count for report in self.locales),
                "info_count": sum(report.info_count for report in self.locales),
                "issue_type_counts": dict(sorted(issue_type_counts.items())),
                "severity_counts": dict(sorted(severity_counts.items())),
            },
            "locales": [report.to_dict() for report in self.locales],
        }


def audit_step_solution_templates(
    locales: Sequence[str],
    templates_root: Path = DEFAULT_SOLUTION_TEMPLATES_ROOT,
    english_template_path: Optional[Path] = None,
    strict: bool = False,
) -> TemplateAuditReport:
    """
    Audit localized Template_Messages.xlsx workbooks against English.

    This does not modify any workbook.
    """

    templates_root = Path(templates_root)
    english_path = Path(english_template_path) if english_template_path else message_template_path(
        "en",
        templates_root,
    )

    english = read_step_solution_template_workbook(
        path=english_path,
        locale="en",
    )

    locale_reports: List[TemplateAuditLocaleReport] = []

    for locale_value in locales:
        locale = normalize_step_solution_locale(locale_value)
        if locale == "en":
            continue

        localized_path = message_template_path(locale, templates_root)
        locale_report = TemplateAuditLocaleReport(
            locale=locale,
            template_path=localized_path,
        )

        try:
            localized = read_step_solution_template_workbook(
                path=localized_path,
                locale=locale,
            )
            _audit_one_locale(
                english=english,
                localized=localized,
                report=locale_report,
            )
        except Exception as exc:
            locale_report.add_issue(
                TemplateAuditIssue(
                    locale=locale,
                    severity="error",
                    issue_type="template_read_failed",
                    sheet="",
                    key="",
                    message=str(exc),
                )
            )

        locale_reports.append(locale_report)

    return TemplateAuditReport(
        english_template_path=english_path,
        templates_root=templates_root,
        strict=strict,
        locales=locale_reports,
    )


def write_template_audit_report(
    report: TemplateAuditReport,
    output_dir: Path = Path("runs/publishing/classic9/step_solution_template_audit"),
) -> Dict[str, Path]:
    """
    Write JSON and text summary reports.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "template_audit_report.json"
    summary_path = output_dir / "template_audit_summary.txt"

    json_path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary_path.write_text(
        build_template_audit_summary_text(report),
        encoding="utf-8",
    )

    return {
        "json": json_path,
        "summary": summary_path,
    }


def build_template_audit_summary_text(report: TemplateAuditReport) -> str:
    lines: List[str] = []

    payload = report.to_dict()
    summary = payload["summary"]

    lines.append("Step Solution Template Audit")
    lines.append("=" * 35)
    lines.append(f"English template: {report.english_template_path}")
    lines.append(f"Templates root:   {report.templates_root}")
    lines.append(f"Strict mode:      {report.strict}")
    lines.append(f"Overall OK:       {report.ok()}")
    lines.append("")
    lines.append(
        "Totals: "
        f"{summary['issue_count']} issues, "
        f"{summary['error_count']} errors, "
        f"{summary['warning_count']} warnings, "
        f"{summary['info_count']} info"
    )

    issue_type_counts = summary.get("issue_type_counts") or {}
    if issue_type_counts:
        lines.append("")
        lines.append("Issue types:")
        for issue_type, count in sorted(issue_type_counts.items()):
            lines.append(f"  {issue_type}: {count}")

    lines.append("")

    for locale_report in report.locales:
        lines.append(f"[{locale_report.locale}] {locale_report.template_path}")
        locale_payload = locale_report.to_dict()
        issue_type_counts = locale_payload.get("issue_type_counts") or {}

        lines.append(
            f"  issues={locale_report.issue_count}, "
            f"errors={locale_report.error_count}, "
            f"warnings={locale_report.warning_count}, "
            f"info={locale_report.info_count}, "
            f"ok={locale_report.ok(strict=report.strict)}"
        )

        if issue_type_counts:
            compact = ", ".join(
                f"{issue_type}={count}"
                for issue_type, count in sorted(issue_type_counts.items())
            )
            lines.append(f"  issue_types: {compact}")

        for issue in locale_report.issues[:60]:
            row = f" row={issue.row}" if issue.row is not None else ""
            key = f" key={issue.key}" if issue.key else ""
            sheet = f" sheet={issue.sheet}" if issue.sheet else ""
            lines.append(
                f"  - {issue.severity.upper()} {issue.issue_type}{sheet}{key}{row}: {issue.message}"
            )

        remaining = len(locale_report.issues) - 60
        if remaining > 0:
            lines.append(f"  ... {remaining} more issue(s) in JSON report")

        lines.append("")

    return "\n".join(lines)


def _audit_one_locale(
    english: StepSolutionTemplateWorkbook,
    localized: StepSolutionTemplateWorkbook,
    report: TemplateAuditLocaleReport,
) -> None:
    _audit_required_sheets(english, localized, report)
    _audit_messages(english, localized, report)
    _audit_names(english, localized, report)
    _audit_keywords(english, localized, report)
    _audit_headers(english, localized, report)
    _audit_text_leakage(localized, report)


def _audit_required_sheets(
    english: StepSolutionTemplateWorkbook,
    localized: StepSolutionTemplateWorkbook,
    report: TemplateAuditLocaleReport,
) -> None:
    for sheet_name in CANONICAL_TEMPLATE_SHEETS:
        if sheet_name not in localized.sheet_names:
            report.add_issue(
                TemplateAuditIssue(
                    locale=report.locale,
                    severity="error",
                    issue_type="missing_sheet",
                    sheet=sheet_name,
                    key="",
                    message=f"Missing required sheet {sheet_name!r}.",
                )
            )

    for sheet_name in localized.sheet_names:
        if sheet_name not in english.sheet_names:
            report.add_issue(
                TemplateAuditIssue(
                    locale=report.locale,
                    severity="info",
                    issue_type="extra_sheet",
                    sheet=sheet_name,
                    key="",
                    message=f"Localized workbook has extra sheet {sheet_name!r}.",
                )
            )


def _audit_messages(
    english: StepSolutionTemplateWorkbook,
    localized: StepSolutionTemplateWorkbook,
    report: TemplateAuditLocaleReport,
) -> None:
    english_by_key = english.message_by_primary_key()
    localized_by_key = localized.message_by_primary_key()

    for key, english_record in english_by_key.items():
        localized_record = localized_by_key.get(key)
        if localized_record is None:
            report.add_issue(
                TemplateAuditIssue(
                    locale=report.locale,
                    severity="error",
                    issue_type="missing_message_row",
                    sheet="Messages",
                    key=key,
                    row=english_record.row,
                    message="Localized workbook is missing this message row.",
                )
            )
            continue

        if len(localized_record.fragments) != len(english_record.fragments):
            report.add_issue(
                TemplateAuditIssue(
                    locale=report.locale,
                    severity="error",
                    issue_type="message_fragment_count_mismatch",
                    sheet="Messages",
                    key=key,
                    row=localized_record.row,
                    message=(
                        "Message fragment count differs from English. "
                        f"English={len(english_record.fragments)}, "
                        f"Localized={len(localized_record.fragments)}."
                    ),
                    details={
                        "english_fragments": english_record.fragments,
                        "localized_fragments": localized_record.fragments,
                    },
                )
            )

        english_placeholders = set(english_record.placeholders)
        localized_placeholders = set(localized_record.placeholders)
        if english_placeholders != localized_placeholders:
            report.add_issue(
                TemplateAuditIssue(
                    locale=report.locale,
                    severity="error",
                    issue_type="placeholder_mismatch",
                    sheet="Messages",
                    key=key,
                    row=localized_record.row,
                    message="Localized message placeholders do not match English.",
                    details={
                        "missing": sorted(english_placeholders - localized_placeholders),
                        "extra": sorted(localized_placeholders - english_placeholders),
                        "english_placeholders": sorted(english_placeholders),
                        "localized_placeholders": sorted(localized_placeholders),
                    },
                )
            )

        if not localized_record.fragments:
            report.add_issue(
                TemplateAuditIssue(
                    locale=report.locale,
                    severity="warning",
                    issue_type="empty_message_translation",
                    sheet="Messages",
                    key=key,
                    row=localized_record.row,
                    message="Localized message row has no translated fragments.",
                )
            )

    for key, localized_record in localized_by_key.items():
        if key not in english_by_key:
            report.add_issue(
                TemplateAuditIssue(
                    locale=report.locale,
                    severity="info",
                    issue_type="extra_message_row",
                    sheet="Messages",
                    key=key,
                    row=localized_record.row,
                    message="Localized workbook has a message row not present in English.",
                )
            )


def _audit_names(
    english: StepSolutionTemplateWorkbook,
    localized: StepSolutionTemplateWorkbook,
    report: TemplateAuditLocaleReport,
) -> None:
    english_by_key = english.name_by_key()
    localized_by_key = localized.name_by_key()

    for key, english_record in english_by_key.items():
        localized_record = localized_by_key.get(key)
        if localized_record is None:
            report.add_issue(
                TemplateAuditIssue(
                    locale=report.locale,
                    severity="error",
                    issue_type="missing_name_row",
                    sheet="Names",
                    key=key,
                    row=english_record.row,
                    message="Localized workbook is missing this technique name row.",
                )
            )
            continue

        if not localized_record.name:
            report.add_issue(
                TemplateAuditIssue(
                    locale=report.locale,
                    severity="warning",
                    issue_type="empty_technique_name",
                    sheet="Names",
                    key=key,
                    row=localized_record.row,
                    message="Technique name is empty.",
                )
            )

        if localized_record.rating in {"Easy", "Medium", "Hard"} and report.locale != "en":
            report.add_issue(
                TemplateAuditIssue(
                    locale=report.locale,
                    severity="warning",
                    issue_type="rating_still_english",
                    sheet="Names",
                    key=key,
                    row=localized_record.row,
                    message=f"Rating is still English: {localized_record.rating!r}.",
                )
            )

        english_placeholders = set(english_record.placeholders)
        localized_placeholders = set(localized_record.placeholders)
        if english_placeholders != localized_placeholders:
            report.add_issue(
                TemplateAuditIssue(
                    locale=report.locale,
                    severity="error",
                    issue_type="placeholder_mismatch",
                    sheet="Names",
                    key=key,
                    row=localized_record.row,
                    message="Technique name placeholders do not match English.",
                    details={
                        "missing": sorted(english_placeholders - localized_placeholders),
                        "extra": sorted(localized_placeholders - english_placeholders),
                    },
                )
            )


def _audit_keywords(
    english: StepSolutionTemplateWorkbook,
    localized: StepSolutionTemplateWorkbook,
    report: TemplateAuditLocaleReport,
) -> None:
    english_by_key = english.keyword_by_key()
    localized_by_key = localized.keyword_by_key()

    for key, english_record in english_by_key.items():
        localized_record = localized_by_key.get(key)
        if localized_record is None:
            report.add_issue(
                TemplateAuditIssue(
                    locale=report.locale,
                    severity="error",
                    issue_type="missing_keyword_row",
                    sheet="Keywords",
                    key=key,
                    row=english_record.row,
                    message="Localized workbook is missing this keyword row.",
                )
            )
            continue

        if not localized_record.values:
            report.add_issue(
                TemplateAuditIssue(
                    locale=report.locale,
                    severity="warning",
                    issue_type="empty_keyword_values",
                    sheet="Keywords",
                    key=key,
                    row=localized_record.row,
                    message="Keyword row has no localized values.",
                )
            )

        english_placeholders = set(english_record.placeholders)
        localized_placeholders = set(localized_record.placeholders)
        if english_placeholders != localized_placeholders:
            report.add_issue(
                TemplateAuditIssue(
                    locale=report.locale,
                    severity="error",
                    issue_type="placeholder_mismatch",
                    sheet="Keywords",
                    key=key,
                    row=localized_record.row,
                    message="Keyword placeholders do not match English.",
                    details={
                        "missing": sorted(english_placeholders - localized_placeholders),
                        "extra": sorted(localized_placeholders - english_placeholders),
                    },
                )
            )


def _audit_headers(
    english: StepSolutionTemplateWorkbook,
    localized: StepSolutionTemplateWorkbook,
    report: TemplateAuditLocaleReport,
) -> None:
    english_records = english.headers
    localized_records = localized.headers

    if len(localized_records) < len(english_records):
        report.add_issue(
            TemplateAuditIssue(
                locale=report.locale,
                severity="warning",
                issue_type="fewer_header_rows",
                sheet="Headers",
                key="",
                message=(
                    f"Localized Headers sheet has fewer non-empty rows than English. "
                    f"English={len(english_records)}, Localized={len(localized_records)}."
                ),
            )
        )

    english_placeholders = set().union(
        *(set(record.placeholders) for record in english_records)
    ) if english_records else set()
    localized_placeholders = set().union(
        *(set(record.placeholders) for record in localized_records)
    ) if localized_records else set()

    missing_required = english_placeholders - localized_placeholders
    if missing_required:
        report.add_issue(
            TemplateAuditIssue(
                locale=report.locale,
                severity="warning",
                issue_type="header_placeholder_missing",
                sheet="Headers",
                key="",
                message="Localized headers are missing placeholders used by English.",
                details={
                    "missing": sorted(missing_required),
                    "english_placeholders": sorted(english_placeholders),
                    "localized_placeholders": sorted(localized_placeholders),
                },
            )
        )


def _audit_text_leakage(
    localized: StepSolutionTemplateWorkbook,
    report: TemplateAuditLocaleReport,
) -> None:
    allowed = allowed_english_terms_for_locale(report.locale)
    allowed_substrings = allowed_substring_false_positives_for_locale(report.locale)
    foreign_terms = list(FOREIGN_LEAKAGE_TERMS_BY_LOCALE.get(report.locale, ()))

    for sheet_name, cells in localized.raw_cells.items():
        for cell in cells:
            _audit_cell_english_leakage(
                cell_value=cell.value,
                sheet=sheet_name,
                row=cell.row,
                column=cell.column,
                report=report,
                allowed_terms=allowed,
                allowed_substrings=allowed_substrings,
            )
            _audit_cell_foreign_leakage(
                cell_value=cell.value,
                sheet=sheet_name,
                row=cell.row,
                column=cell.column,
                report=report,
                foreign_terms=foreign_terms,
            )


def _audit_cell_english_leakage(
    cell_value: str,
    sheet: str,
    row: int,
    column: int,
    report: TemplateAuditLocaleReport,
    allowed_terms: Set[str],
    allowed_substrings: Set[str],
) -> None:
    original_text = str(cell_value or "")
    if not original_text:
        return

    # Avoid noisy reports for machine keys such as:
    #   singles-1
    #   boxed-triplets
    #   boxed-triplets\nboxed-quads
    if _is_probably_machine_key_blob(original_text):
        return

    # Ignore workbook maintainer comments. They are not generated story text.
    if original_text.strip().lower().startswith("remark:"):
        return

    # Remove placeholders so {cell}, {box}, {row}, etc. are not treated as
    # English prose.
    text = strip_placeholders(original_text)
    if not text.strip():
        return

    hits = []
    for term in ENGLISH_LEAKAGE_TERMS:
        if term in allowed_terms:
            continue

        term_lower = term.lower()
        if term_lower in allowed_substrings:
            continue

        if _contains_english_leakage_term(text, term):
            hits.append(term)

    if not hits:
        return

    report.add_issue(
        TemplateAuditIssue(
            locale=report.locale,
            severity="warning",
            issue_type="possible_english_leakage",
            sheet=sheet,
            key="",
            row=row,
            message=f"Possible English text remains: {', '.join(sorted(set(hits)))}.",
            details={
                "column": column,
                "value": original_text,
                "scanned_value": text,
            },
        )
    )


def _audit_cell_foreign_leakage(
    cell_value: str,
    sheet: str,
    row: int,
    column: int,
    report: TemplateAuditLocaleReport,
    foreign_terms: Sequence[str],
) -> None:
    text = str(cell_value or "")
    if not text:
        return

    hits = [term for term in foreign_terms if term and term in text]
    if not hits:
        return

    report.add_issue(
        TemplateAuditIssue(
            locale=report.locale,
            severity="warning",
            issue_type="possible_foreign_language_leakage",
            sheet=sheet,
            key="",
            row=row,
            message=f"Possible wrong-language text remains: {', '.join(sorted(set(hits)))}.",
            details={
                "column": column,
                "value": text,
            },
        )
    )


def _contains_english_leakage_term(text: str, term: str) -> bool:
    """
    Match English leakage terms safely.

    Multi-word phrases are matched case-insensitively as phrases.
    Single words are matched with word boundaries so "column" is not found
    inside unrelated words and placeholders have already been stripped.
    """

    if not text or not term:
        return False

    if " " in term:
        return term.lower() in text.lower()

    pattern = re.compile(
        r"(?<![A-Za-zÀ-ÖØ-öø-ÿ])"
        + re.escape(term)
        + r"(?![A-Za-zÀ-ÖØ-öø-ÿ])",
        flags=re.IGNORECASE,
    )
    return bool(pattern.search(text))


def _is_probably_machine_key_blob(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return False

    parts = [part.strip() for part in value.splitlines() if part.strip()]
    if not parts:
        return False

    return all(_is_probably_machine_key_only(part) for part in parts)


def _is_probably_machine_key_only(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789-_")
    return value.lower() == value and all(char in allowed for char in value)