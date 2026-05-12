from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional


@dataclass(frozen=True)
class DeleteTarget:
    target_type: str
    target_id: str
    display_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DeleteDependency:
    dependency_type: str
    path: str
    detail: str = ""
    reference_id: Optional[str] = None
    blocks_hard_delete: bool = False
    blocks_archive: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DeleteBlocker:
    code: str
    message: str
    path: Optional[str] = None
    target_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DeletePlan:
    target: DeleteTarget
    requested_action: str
    allowed_actions: List[str] = field(default_factory=list)
    dependencies: List[DeleteDependency] = field(default_factory=list)
    blockers: List[DeleteBlocker] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    files_to_update: List[str] = field(default_factory=list)
    files_to_delete: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target.to_dict(),
            "requested_action": self.requested_action,
            "allowed_actions": list(self.allowed_actions),
            "dependencies": [item.to_dict() for item in self.dependencies],
            "blockers": [item.to_dict() for item in self.blockers],
            "warnings": list(self.warnings),
            "files_to_update": list(self.files_to_update),
            "files_to_delete": list(self.files_to_delete),
            "notes": list(self.notes),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DeletePlan":
        return cls(
            target=DeleteTarget(**dict(data["target"])),
            requested_action=str(data["requested_action"]),
            allowed_actions=list(data.get("allowed_actions", [])),
            dependencies=[DeleteDependency(**dict(item)) for item in data.get("dependencies", [])],
            blockers=[DeleteBlocker(**dict(item)) for item in data.get("blockers", [])],
            warnings=list(data.get("warnings", [])),
            files_to_update=list(data.get("files_to_update", [])),
            files_to_delete=list(data.get("files_to_delete", [])),
            notes=list(data.get("notes", [])),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class DeletePolicyDecision:
    outcome: str
    action: str
    summary: str
    warnings: List[str] = field(default_factory=list)
    blockers: List[DeleteBlocker] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outcome": self.outcome,
            "action": self.action,
            "summary": self.summary,
            "warnings": list(self.warnings),
            "blockers": [item.to_dict() for item in self.blockers],
        }


class DeleteAction:
    ARCHIVE = "archive"
    DELETE = "delete"
    CASCADE_DELETE = "cascade_delete"
    RAW_DELETE = "raw_delete"