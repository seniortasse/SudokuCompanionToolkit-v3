from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from python.publishing.cleanup.delete_models import DeletePlan


DEFAULT_SINGLE_CONFIRM = "YES"
DEFAULT_BATCH_CONFIRM = "DELETE-BATCH"
DEFAULT_DELETE_ALL_BOOKS_CONFIRM = "DELETE-ALL-BOOKS"
DEFAULT_DELETE_ALL_PUZZLES_CONFIRM = "DELETE-ALL-PUZZLES"
DEFAULT_DELETE_ALL_CANDIDATES_CONFIRM = "DELETE-ALL-CANDIDATES"


@dataclass(frozen=True)
class ConfirmationSpec:
    required_token: str
    prompt: str
    reason: str


def confirmation_token_for_plan(
    plan: DeletePlan,
    *,
    is_batch: bool = False,
    is_delete_all: bool = False,
) -> ConfirmationSpec:
    target_type = plan.target.target_type

    if is_delete_all:
        if target_type == "book":
            token = DEFAULT_DELETE_ALL_BOOKS_CONFIRM
        elif target_type == "puzzle_record":
            token = DEFAULT_DELETE_ALL_PUZZLES_CONFIRM
        elif target_type == "candidate_jsonl":
            token = DEFAULT_DELETE_ALL_CANDIDATES_CONFIRM
        else:
            token = "DELETE-ALL"
        return ConfirmationSpec(
            required_token=token,
            prompt=f"Type {token} to proceed: ",
            reason="delete_all",
        )

    if is_batch:
        return ConfirmationSpec(
            required_token=DEFAULT_BATCH_CONFIRM,
            prompt=f"Type {DEFAULT_BATCH_CONFIRM} to proceed: ",
            reason="batch",
        )

    return ConfirmationSpec(
        required_token=DEFAULT_SINGLE_CONFIRM,
        prompt=f"Type {DEFAULT_SINGLE_CONFIRM} to proceed: ",
        reason="single",
    )


def is_confirmation_satisfied(
    *,
    plan: DeletePlan,
    provided_token: Optional[str] = None,
    yes: bool = False,
    is_batch: bool = False,
    is_delete_all: bool = False,
) -> bool:
    if yes:
        return True

    spec = confirmation_token_for_plan(
        plan,
        is_batch=is_batch,
        is_delete_all=is_delete_all,
    )
    return str(provided_token or "").strip() == spec.required_token


def prompt_for_confirmation(
    *,
    plan: DeletePlan,
    is_batch: bool = False,
    is_delete_all: bool = False,
) -> bool:
    spec = confirmation_token_for_plan(
        plan,
        is_batch=is_batch,
        is_delete_all=is_delete_all,
    )
    typed = input(spec.prompt).strip()
    return typed == spec.required_token