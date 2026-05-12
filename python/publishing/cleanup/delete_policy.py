from __future__ import annotations

from typing import List

from python.publishing.cleanup.delete_models import (
    DeleteAction,
    DeleteBlocker,
    DeletePolicyDecision,
    DeletePlan,
)


def _collect_blockers_for_action(plan: DeletePlan, *, action: str) -> List[DeleteBlocker]:
    blockers: List[DeleteBlocker] = list(plan.blockers)

    for dep in plan.dependencies:
        if action == DeleteAction.ARCHIVE and dep.blocks_archive:
            blockers.append(
                DeleteBlocker(
                    code="DEPENDENCY_BLOCKS_ARCHIVE",
                    message=dep.detail or f"{dep.dependency_type} blocks archive",
                    path=dep.path,
                    target_id=dep.reference_id,
                )
            )
        elif action in (DeleteAction.DELETE, DeleteAction.CASCADE_DELETE, DeleteAction.RAW_DELETE) and dep.blocks_hard_delete:
            blockers.append(
                DeleteBlocker(
                    code="DEPENDENCY_BLOCKS_DELETE",
                    message=dep.detail or f"{dep.dependency_type} blocks delete",
                    path=dep.path,
                    target_id=dep.reference_id,
                )
            )

    return blockers


def decide_record_delete(plan: DeletePlan) -> DeletePolicyDecision:
    requested = plan.requested_action

    if requested == DeleteAction.ARCHIVE:
        blockers = _collect_blockers_for_action(plan, action=DeleteAction.ARCHIVE)
        if blockers:
            return DeletePolicyDecision(
                outcome="BLOCK",
                action=DeleteAction.ARCHIVE,
                summary="Puzzle record archive is blocked by active downstream dependencies or state inconsistencies.",
                warnings=list(plan.warnings),
                blockers=blockers,
            )
        return DeletePolicyDecision(
            outcome="ALLOW",
            action=DeleteAction.ARCHIVE,
            summary="Puzzle record can be archived safely.",
            warnings=list(plan.warnings),
            blockers=[],
        )

    blockers = _collect_blockers_for_action(plan, action=DeleteAction.DELETE)
    if blockers:
        archive_blockers = _collect_blockers_for_action(plan, action=DeleteAction.ARCHIVE)
        if not archive_blockers:
            warnings = list(plan.warnings)
            warnings.append("Hard delete is blocked, but archive remains available.")
            return DeletePolicyDecision(
                outcome="ALLOW_WITH_WARNINGS",
                action=DeleteAction.ARCHIVE,
                summary="Hard delete is blocked. Archive is the safest allowed action.",
                warnings=warnings,
                blockers=blockers,
            )

        return DeletePolicyDecision(
            outcome="BLOCK",
            action=DeleteAction.DELETE,
            summary="Puzzle record hard delete is blocked.",
            warnings=list(plan.warnings),
            blockers=blockers,
        )

    return DeletePolicyDecision(
        outcome="ALLOW",
        action=DeleteAction.DELETE,
        summary="Puzzle record can be hard-deleted safely.",
        warnings=list(plan.warnings),
        blockers=[],
    )


def decide_book_delete(
    plan: DeletePlan,
    *,
    cascade_publications: bool = False,
    cascade_publication_specs: bool = False,
) -> DeletePolicyDecision:
    blockers: List[DeleteBlocker] = list(plan.blockers)
    warnings: List[str] = list(plan.warnings)

    for dep in plan.dependencies:
        if dep.dependency_type == "publication_package":
            if cascade_publications:
                warnings.append(
                    f"Publication package will need cascading removal: {dep.reference_id or dep.path}"
                )
            else:
                blockers.append(
                    DeleteBlocker(
                        code="PUBLICATION_PACKAGE_EXISTS",
                        message=dep.detail or "Book still has publication packages",
                        path=dep.path,
                        target_id=dep.reference_id,
                    )
                )
        elif dep.dependency_type == "publication_spec":
            if cascade_publication_specs:
                warnings.append(
                    f"Publication spec will need cascading removal: {dep.path}"
                )
            else:
                blockers.append(
                    DeleteBlocker(
                        code="PUBLICATION_SPEC_EXISTS",
                        message=dep.detail or "Book still has publication specs",
                        path=dep.path,
                        target_id=dep.reference_id,
                    )
                )

    if blockers:
        return DeletePolicyDecision(
            outcome="BLOCK",
            action=DeleteAction.DELETE,
            summary="Book delete is blocked by downstream publication artifacts or integrity issues.",
            warnings=warnings,
            blockers=blockers,
        )

    final_action = DeleteAction.CASCADE_DELETE if (cascade_publications or cascade_publication_specs) else DeleteAction.DELETE

    if final_action == DeleteAction.CASCADE_DELETE:
        warnings.append("Cascade deletion will be required for all selected downstream publication artifacts.")

    return DeletePolicyDecision(
        outcome="ALLOW",
        action=final_action,
        summary="Book can be deleted safely under the selected policy.",
        warnings=warnings,
        blockers=[],
    )


def decide_publication_delete(
    plan: DeletePlan,
    *,
    cascade_publication_specs: bool = False,
) -> DeletePolicyDecision:
    blockers: List[DeleteBlocker] = list(plan.blockers)
    warnings: List[str] = list(plan.warnings)

    for dep in plan.dependencies:
        if dep.dependency_type == "publication_spec":
            if cascade_publication_specs:
                warnings.append(
                    f"Publication spec will need cascading removal: {dep.path}"
                )
            else:
                warnings.append(
                    f"Publication spec will be retained unless explicitly cascaded: {dep.path}"
                )

    if blockers:
        return DeletePolicyDecision(
            outcome="BLOCK",
            action=DeleteAction.DELETE,
            summary="Publication delete is blocked by integrity issues.",
            warnings=warnings,
            blockers=blockers,
        )

    final_action = DeleteAction.CASCADE_DELETE if cascade_publication_specs else DeleteAction.DELETE
    if final_action == DeleteAction.CASCADE_DELETE:
        warnings.append("Cascade deletion will be required for the selected publication spec files.")

    return DeletePolicyDecision(
        outcome="ALLOW",
        action=final_action,
        summary="Publication can be deleted safely under the selected policy.",
        warnings=warnings,
        blockers=[],
    )


def decide_publication_spec_delete(plan: DeletePlan) -> DeletePolicyDecision:
    blockers = _collect_blockers_for_action(plan, action=DeleteAction.DELETE)
    warnings = list(plan.warnings)

    if blockers:
        return DeletePolicyDecision(
            outcome="BLOCK",
            action=DeleteAction.DELETE,
            summary="Publication spec delete is blocked.",
            warnings=warnings,
            blockers=blockers,
        )

    return DeletePolicyDecision(
        outcome="ALLOW",
        action=DeleteAction.DELETE,
        summary="Publication spec delete is allowed.",
        warnings=warnings,
        blockers=[],
    )



def decide_candidate_jsonl_delete(plan: DeletePlan) -> DeletePolicyDecision:
    blockers = _collect_blockers_for_action(plan, action=DeleteAction.RAW_DELETE)
    warnings = list(plan.warnings)

    if blockers:
        return DeletePolicyDecision(
            outcome="BLOCK",
            action=DeleteAction.RAW_DELETE,
            summary="Raw candidate delete is blocked.",
            warnings=warnings,
            blockers=blockers,
        )

    canonical_matches = [
        dep
        for dep in plan.dependencies
        if dep.dependency_type == "canonical_record_match"
    ]
    if canonical_matches:
        warnings.append(
            "Selected raw candidates appear to match canonical puzzle records. Deleting raw provenance may still be acceptable, but historical traceability will be reduced."
        )
        return DeletePolicyDecision(
            outcome="ALLOW_WITH_WARNINGS",
            action=DeleteAction.RAW_DELETE,
            summary="Raw candidate delete is allowed, but linked canonical provenance appears to exist.",
            warnings=warnings,
            blockers=[],
        )

    return DeletePolicyDecision(
        outcome="ALLOW",
        action=DeleteAction.RAW_DELETE,
        summary="Raw candidate delete is allowed.",
        warnings=warnings,
        blockers=[],
    )