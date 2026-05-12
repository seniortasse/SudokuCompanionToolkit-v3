from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from python.publishing.cleanup.delete_models import DeletePlan, DeletePolicyDecision


DEFAULT_REPORT_DIR = Path("runs/publishing/delete_reports")


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _normalize_path(path: Path) -> str:
    return str(path).replace("/", "\\")


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def build_report_path(
    *,
    plan: DeletePlan,
    report_dir: Path = DEFAULT_REPORT_DIR,
    stage: str,
    label: Optional[str] = None,
) -> Path:
    safe_stage = _safe_name(stage)
    safe_target_type = _safe_name(plan.target.target_type)
    safe_target_id = _safe_name(plan.target.target_id)
    suffix = f"_{_safe_name(label)}" if label else ""
    filename = f"{_utc_stamp()}__{safe_stage}__{safe_target_type}__{safe_target_id}{suffix}.json"
    return report_dir / filename


def build_report_payload(
    *,
    plan: DeletePlan,
    decision: Optional[DeletePolicyDecision] = None,
    stage: str,
    snapshot_dir: Optional[Path] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "generated_utc": _utc_stamp(),
        "stage": stage,
        "plan": plan.to_dict(),
    }

    if decision is not None:
        payload["decision"] = decision.to_dict()

    if snapshot_dir is not None:
        payload["snapshot_dir"] = _normalize_path(snapshot_dir)

    if extra:
        payload["extra"] = dict(extra)

    return payload


def write_delete_report(
    *,
    plan: DeletePlan,
    decision: Optional[DeletePolicyDecision] = None,
    report_dir: Path = DEFAULT_REPORT_DIR,
    stage: str,
    snapshot_dir: Optional[Path] = None,
    label: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    report_path = build_report_path(
        plan=plan,
        report_dir=report_dir,
        stage=stage,
        label=label,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)

    payload = build_report_payload(
        plan=plan,
        decision=decision,
        stage=stage,
        snapshot_dir=snapshot_dir,
        extra=extra,
    )
    report_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report_path