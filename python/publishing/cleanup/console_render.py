from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from python.publishing.cleanup.delete_models import DeletePlan, DeletePolicyDecision


def _indent(lines: List[str], prefix: str = "  ") -> List[str]:
    return [f"{prefix}{line}" for line in lines]


def _trim_list(items: List[str], *, limit: int = 20) -> List[str]:
    if len(items) <= limit:
        return items
    head = items[:limit]
    head.append(f"... ({len(items) - limit} more)")
    return head


def _section(title: str, body_lines: List[str]) -> List[str]:
    if not body_lines:
        return [f"{title}: none"]
    return [f"{title}:"] + _indent(body_lines)


def render_delete_plan(plan: DeletePlan) -> str:
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("DELETE PREVIEW")
    lines.append("=" * 72)
    lines.append(f"Target type:       {plan.target.target_type}")
    lines.append(f"Target id:         {plan.target.target_id}")
    lines.append(f"Requested action:  {plan.requested_action}")
    lines.append(f"Allowed actions:   {', '.join(plan.allowed_actions) if plan.allowed_actions else 'none'}")

    if plan.target.display_name and plan.target.display_name != plan.target.target_id:
        lines.append(f"Display name:      {plan.target.display_name}")

    if plan.blockers:
        lines.extend(_section(
            "Plan blockers",
            [f"[{item.code}] {item.message}" + (f" ({item.path})" if item.path else "") for item in plan.blockers],
        ))

    if plan.dependencies:
        dep_lines = []
        for dep in plan.dependencies:
            flags = []
            if dep.blocks_hard_delete:
                flags.append("blocks_delete")
            if dep.blocks_archive:
                flags.append("blocks_archive")
            suffix = f" [{' | '.join(flags)}]" if flags else ""
            detail = dep.detail or dep.dependency_type
            location = f" ({dep.path})" if dep.path else ""
            dep_lines.append(f"{dep.dependency_type}: {detail}{location}{suffix}")
        lines.extend(_section("Dependencies", _trim_list(dep_lines, limit=30)))

    if plan.warnings:
        lines.extend(_section("Warnings", _trim_list(list(plan.warnings), limit=20)))

    if plan.notes:
        lines.extend(_section("Notes", _trim_list(list(plan.notes), limit=20)))

    if plan.files_to_update:
        lines.extend(_section("Files to update", _trim_list(list(plan.files_to_update), limit=30)))

    if plan.files_to_delete:
        lines.extend(_section("Files to delete", _trim_list(list(plan.files_to_delete), limit=30)))

    if plan.metadata:
        metadata_lines = []
        for key in sorted(plan.metadata.keys()):
            value = plan.metadata[key]
            if isinstance(value, (str, int, float, bool)) or value is None:
                metadata_lines.append(f"{key}: {value}")
            elif isinstance(value, list):
                metadata_lines.append(f"{key}: list[{len(value)}]")
            elif isinstance(value, dict):
                metadata_lines.append(f"{key}: object[{len(value)} keys]")
            else:
                metadata_lines.append(f"{key}: {type(value).__name__}")
        lines.extend(_section("Metadata", _trim_list(metadata_lines, limit=30)))

    return "\n".join(lines)


def render_policy_decision(decision: DeletePolicyDecision) -> str:
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("DELETE DECISION")
    lines.append("=" * 72)
    lines.append(f"Outcome:  {decision.outcome}")
    lines.append(f"Action:   {decision.action}")
    lines.append(f"Summary:  {decision.summary}")

    if decision.warnings:
        lines.extend(_section("Warnings", _trim_list(list(decision.warnings), limit=20)))

    if decision.blockers:
        blocker_lines = []
        for item in decision.blockers:
            suffix = f" ({item.path})" if item.path else ""
            blocker_lines.append(f"[{item.code}] {item.message}{suffix}")
        lines.extend(_section("Blockers", _trim_list(blocker_lines, limit=30)))

    return "\n".join(lines)


def render_snapshot_summary(snapshot_dir: Path) -> str:
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("BACKUP SNAPSHOT")
    lines.append("=" * 72)
    lines.append(f"Snapshot dir: {snapshot_dir}")
    lines.append("A pre-mutation backup snapshot has been prepared.")
    return "\n".join(lines)


def render_report_summary(report_path: Path, *, stage: str) -> str:
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("DELETE REPORT")
    lines.append("=" * 72)
    lines.append(f"Stage:       {stage}")
    lines.append(f"Report path: {report_path}")
    return "\n".join(lines)