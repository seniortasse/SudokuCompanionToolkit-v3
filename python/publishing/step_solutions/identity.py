from __future__ import annotations

import re
from dataclasses import dataclass


_INTERNAL_PUZZLE_CODE_RE = re.compile(r"^L(?P<level>\d+)-(?P<number>\d+)$")
_EXTERNAL_PUZZLE_CODE_RE = re.compile(r"^L-(?P<level>\d+)-(?P<number>\d+)$")
_BOOK_ID_RE = re.compile(r"^BK-[A-Z0-9]+-[A-Z0-9]+-(?P<code>B\d+)$")


@dataclass(frozen=True)
class StepSolutionIdentity:
    """
    Central identity mapping for a puzzle inside a step-solution package.

    Example:
        book_id:                  BK-CL9-DW-B01
        commercial_book_code:     B01
        internal_puzzle_code:     L1-001
        external_puzzle_code:     L-1-1
        commercial_problem_id:    B01-L-1-1
    """

    book_id: str
    commercial_book_code: str
    internal_puzzle_code: str
    external_puzzle_code: str
    commercial_problem_id: str


def internal_to_external_puzzle_code(local_puzzle_code: str) -> str:
    """
    Convert platform/internal puzzle code to external/user-facing code.

    Examples:
        L1-001 -> L-1-1
        L1-010 -> L-1-10
        L2-034 -> L-2-34
        L3-120 -> L-3-120
    """

    value = str(local_puzzle_code or "").strip()
    match = _INTERNAL_PUZZLE_CODE_RE.match(value)
    if not match:
        raise ValueError(
            "Invalid internal puzzle code. Expected format like 'L1-001'; "
            f"got {local_puzzle_code!r}."
        )

    level = int(match.group("level"))
    number = int(match.group("number"))
    return f"L-{level}-{number}"


def external_to_internal_puzzle_code(external_puzzle_code: str, number_width: int = 3) -> str:
    """
    Convert external/user-facing puzzle code back to platform/internal code.

    Examples:
        L-1-1  -> L1-001
        L-1-10 -> L1-010
        L-2-34 -> L2-034
    """

    value = str(external_puzzle_code or "").strip()
    match = _EXTERNAL_PUZZLE_CODE_RE.match(value)
    if not match:
        raise ValueError(
            "Invalid external puzzle code. Expected format like 'L-1-1'; "
            f"got {external_puzzle_code!r}."
        )

    level = int(match.group("level"))
    number = int(match.group("number"))
    return f"L{level}-{number:0{number_width}d}"


def book_id_to_commercial_code(book_id: str) -> str:
    """
    Convert platform book id to commercial book code.

    Examples:
        BK-CL9-DW-B01 -> B01
        BK-CL9-DW-B02 -> B02
    """

    value = str(book_id or "").strip()
    match = _BOOK_ID_RE.match(value)
    if not match:
        raise ValueError(
            "Invalid book id. Expected format like 'BK-CL9-DW-B01'; "
            f"got {book_id!r}."
        )
    return match.group("code")


def make_commercial_problem_id(book_id: str, local_puzzle_code: str) -> str:
    """
    Build the external problem id used by commercial consumers.

    Example:
        BK-CL9-DW-B01 + L1-001 -> B01-L-1-1
    """

    commercial_book_code = book_id_to_commercial_code(book_id)
    external_puzzle_code = internal_to_external_puzzle_code(local_puzzle_code)
    return f"{commercial_book_code}-{external_puzzle_code}"


def make_step_solution_identity(book_id: str, local_puzzle_code: str) -> StepSolutionIdentity:
    """
    Build the full identity mapping for one puzzle.
    """

    commercial_book_code = book_id_to_commercial_code(book_id)
    external_puzzle_code = internal_to_external_puzzle_code(local_puzzle_code)
    commercial_problem_id = f"{commercial_book_code}-{external_puzzle_code}"

    return StepSolutionIdentity(
        book_id=book_id,
        commercial_book_code=commercial_book_code,
        internal_puzzle_code=local_puzzle_code,
        external_puzzle_code=external_puzzle_code,
        commercial_problem_id=commercial_problem_id,
    )


def make_user_log_filename(local_puzzle_code: str) -> str:
    """
    User-facing Excel log filename.

    Example:
        L1-001 -> L-1-1_user_logs.xlsx
    """

    external_puzzle_code = internal_to_external_puzzle_code(local_puzzle_code)
    return f"{external_puzzle_code}_user_logs.xlsx"


def make_localized_commercial_problem_id(
    book_id: str,
    local_puzzle_code: str,
    locale: str = "",
) -> str:
    """
    Build the image-safe commercial problem id.

    The CSV Problem ID remains language-neutral:
        BK-CL9-DW-B01 + L1-001 -> B01-L-1-1

    Image filenames include locale because all images may be uploaded into
    one shared online-platform folder:
        BK-CL9-DW-B01 + en + L1-001 -> B01-EN-L-1-1
        BK-CL9-DW-B01 + fr + L1-001 -> B01-FR-L-1-1
    """

    identity = make_step_solution_identity(
        book_id=book_id,
        local_puzzle_code=local_puzzle_code,
    )

    locale_token = _locale_filename_token(locale)
    if not locale_token:
        return identity.commercial_problem_id

    return (
        f"{identity.commercial_book_code}-"
        f"{locale_token}-"
        f"{identity.external_puzzle_code}"
    )


def make_answer_image_filename(
    book_id: str,
    local_puzzle_code: str,
    locale: str = "",
) -> str:
    """
    User-facing answer image filename.

    Examples:
        BK-CL9-DW-B01 + L1-001      -> B01-L-1-1_answer.png
        BK-CL9-DW-B01 + en + L1-001 -> B01-EN-L-1-1_answer.png
        BK-CL9-DW-B01 + fr + L1-001 -> B01-FR-L-1-1_answer.png
    """

    problem_id = make_localized_commercial_problem_id(
        book_id=book_id,
        local_puzzle_code=local_puzzle_code,
        locale=locale,
    )
    return f"{problem_id}_answer.png"


def make_step_image_filename(
    book_id: str,
    local_puzzle_code: str,
    step_number: int,
    locale: str = "",
) -> str:
    """
    User-facing step image filename.

    Examples:
        BK-CL9-DW-B01 + L1-001           -> B01-L-1-1_step1.png
        BK-CL9-DW-B01 + en + L1-001 + 1 -> B01-EN-L-1-1_step1.png
        BK-CL9-DW-B01 + fr + L1-001 + 1 -> B01-FR-L-1-1_step1.png
    """

    if int(step_number) < 1:
        raise ValueError(f"Step number must be >= 1; got {step_number!r}.")

    problem_id = make_localized_commercial_problem_id(
        book_id=book_id,
        local_puzzle_code=local_puzzle_code,
        locale=locale,
    )
    return f"{problem_id}_step{int(step_number)}.png"


def _locale_filename_token(locale: str = "") -> str:
    """
    Normalize locale for image filenames.

    Examples:
        en -> EN
        fr -> FR
        de -> DE
    """

    value = str(locale or "").strip()
    if not value:
        return ""

    return value.upper()