from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from python.publishing.pattern_library.pattern_identity import build_canonical_mask_signature
from python.publishing.schemas.models import PatternRecord


@dataclass
class PatternRegistry:
    library_id: str
    patterns: List[PatternRecord] = field(default_factory=list)

    def by_id(self) -> Dict[str, PatternRecord]:
        return {pattern.pattern_id: pattern for pattern in self.patterns}

    def by_mask(self) -> Dict[str, PatternRecord]:
        return {pattern.mask81: pattern for pattern in self.patterns}

    def by_canonical_mask(self) -> Dict[str, PatternRecord]:
        return {pattern.canonical_mask_signature: pattern for pattern in self.patterns}

    def get_next_ordinal(self) -> int:
        max_ordinal = 0
        for pattern in self.patterns:
            try:
                max_ordinal = max(max_ordinal, int(pattern.pattern_id.split("-")[-1]))
            except Exception:
                continue
        return max_ordinal + 1

    def find_by_mask(self, mask81: str) -> Optional[PatternRecord]:
        return self.by_mask().get(mask81)

    def find_by_canonical_mask(self, mask81: str) -> Optional[PatternRecord]:
        signature = build_canonical_mask_signature(mask81)
        return self.by_canonical_mask().get(signature)

    def find_by_id(self, pattern_id: str) -> Optional[PatternRecord]:
        return self.by_id().get(pattern_id)

    def add_or_skip_duplicate_mask(self, pattern: PatternRecord) -> bool:
        existing = self.find_by_canonical_mask(pattern.mask81)
        if existing is not None:
            return False
        self.patterns.append(pattern)
        return True

    def archive_pattern(self, pattern_id: str) -> bool:
        for idx, pattern in enumerate(self.patterns):
            if pattern.pattern_id == pattern_id:
                self.patterns[idx] = PatternRecord.from_dict(
                    {
                        **pattern.to_dict(),
                        "status": "archived",
                    }
                )
                return True
        return False

    def upsert_pattern(self, pattern: PatternRecord) -> None:
        for idx, existing in enumerate(self.patterns):
            if existing.pattern_id == pattern.pattern_id:
                self.patterns[idx] = pattern
                return
        self.patterns.append(pattern)

    def sort_in_place(self) -> None:
        self.patterns.sort(key=lambda p: p.pattern_id)

    def to_dict(self) -> Dict[str, object]:
        return {
            "library_id": self.library_id,
            "pattern_count": len(self.patterns),
            "patterns": [pattern.to_dict() for pattern in self.patterns],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "PatternRegistry":
        patterns = [PatternRecord.from_dict(x) for x in data.get("patterns", [])]
        return cls(
            library_id=str(data["library_id"]),
            patterns=patterns,
        )


def load_registry(registry_path: Path) -> PatternRegistry:
    if not registry_path.exists():
        raise FileNotFoundError(f"Pattern registry not found: {registry_path}")
    data = json.loads(registry_path.read_text(encoding="utf-8"))
    return PatternRegistry.from_dict(data)


def save_registry(registry: PatternRegistry, registry_path: Path) -> None:
    registry.sort_in_place()
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(registry.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )