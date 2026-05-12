from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from python.publishing.cleanup.delete_models import DeletePlan


DEFAULT_BACKUP_ROOT = Path("runs/publishing/backups")


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _normalize_path(path: Path) -> str:
    return str(path).replace("/", "\\")


def _dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _safe_relpath(path: Path, *, base: Path) -> Path:
    try:
        return path.resolve().relative_to(base.resolve())
    except Exception:
        drive = str(path.drive).replace(":", "") if getattr(path, "drive", "") else "root"
        parts = [part for part in path.parts if part not in (path.anchor, "/", "\\")]
        return Path("_external") / drive / Path(*parts)


def _remove_existing_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def collect_snapshot_source_paths(plan: DeletePlan) -> List[Path]:
    candidates = _dedupe_preserve_order([
        *plan.files_to_update,
        *plan.files_to_delete,
    ])

    result: List[Path] = []
    for item in candidates:
        path = Path(item)
        if path.exists():
            result.append(path)
    return result


def build_snapshot_dir(
    *,
    plan: DeletePlan,
    backup_root: Path = DEFAULT_BACKUP_ROOT,
    label: Optional[str] = None,
) -> Path:
    safe_target_id = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in plan.target.target_id)
    safe_target_type = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in plan.target.target_type)
    safe_action = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in plan.requested_action)
    suffix = f"_{label}" if label else ""
    return backup_root / f"{_utc_stamp()}__{safe_target_type}__{safe_target_id}__{safe_action}{suffix}"


def write_backup_snapshot(
    *,
    plan: DeletePlan,
    backup_root: Path = DEFAULT_BACKUP_ROOT,
    project_root: Path = Path("."),
    label: Optional[str] = None,
) -> Path:
    snapshot_dir = build_snapshot_dir(plan=plan, backup_root=backup_root, label=label)
    files_dir = snapshot_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)

    copied_files: List[Dict[str, Any]] = []
    missing_files: List[str] = []

    for source_path in collect_snapshot_source_paths(plan):
        if not source_path.exists():
            missing_files.append(_normalize_path(source_path))
            continue

        rel_path = _safe_relpath(source_path, base=project_root)
        dest_path = files_dir / rel_path

        if source_path.is_dir():
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            if dest_path.exists():
                shutil.rmtree(dest_path)
            shutil.copytree(source_path, dest_path)

            copied_files.append(
                {
                    "source_path": _normalize_path(source_path),
                    "backup_path": _normalize_path(dest_path),
                    "relative_path": str(rel_path).replace("/", "\\"),
                    "path_type": "directory",
                    "size_bytes": None,
                }
            )
        else:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, dest_path)

            copied_files.append(
                {
                    "source_path": _normalize_path(source_path),
                    "backup_path": _normalize_path(dest_path),
                    "relative_path": str(rel_path).replace("/", "\\"),
                    "path_type": "file",
                    "size_bytes": source_path.stat().st_size,
                }
            )

    manifest = {
        "snapshot_created_utc": _utc_stamp(),
        "target": plan.target.to_dict(),
        "requested_action": plan.requested_action,
        "files_considered": _dedupe_preserve_order([
            *plan.files_to_update,
            *plan.files_to_delete,
        ]),
        "copied_files": copied_files,
        "missing_files": missing_files,
        "project_root": _normalize_path(project_root),
        "snapshot_dir": _normalize_path(snapshot_dir),
        "metadata": dict(plan.metadata),
    }
    (snapshot_dir / "snapshot_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return snapshot_dir


def restore_backup_snapshot(
    *,
    snapshot_dir: Path,
    project_root: Path = Path("."),
) -> List[str]:
    manifest_path = snapshot_dir / "snapshot_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Snapshot manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    restored_paths: List[str] = []

    for item in manifest.get("copied_files", []):
        relative_path = Path(str(item["relative_path"]).replace("\\", "/"))
        backup_path = snapshot_dir / "files" / relative_path
        restore_path = project_root / relative_path
        path_type = str(item.get("path_type") or "file").strip().lower()

        if not backup_path.exists():
            raise FileNotFoundError(f"Backup payload missing for restore: {backup_path}")

        if path_type == "directory":
            restore_path.parent.mkdir(parents=True, exist_ok=True)
            _remove_existing_path(restore_path)
            shutil.copytree(backup_path, restore_path)
        else:
            restore_path.parent.mkdir(parents=True, exist_ok=True)
            if restore_path.exists() and restore_path.is_dir():
                shutil.rmtree(restore_path)
            shutil.copy2(backup_path, restore_path)

        restored_paths.append(_normalize_path(restore_path))

    return restored_paths