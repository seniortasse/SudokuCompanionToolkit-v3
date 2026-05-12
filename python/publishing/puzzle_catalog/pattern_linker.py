from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from python.publishing.schemas.models import PatternRecord


@dataclass
class PatternLookup:
    by_id: Dict[str, PatternRecord]
    by_mask: Dict[str, List[PatternRecord]]
    by_family_id: Dict[str, List[PatternRecord]]

    def find_by_mask(self, mask81: str) -> Optional[PatternRecord]:
        matches = self.by_mask.get(str(mask81), [])
        return matches[0] if matches else None

    def find_by_id(self, pattern_id: str) -> Optional[PatternRecord]:
        return self.by_id.get(str(pattern_id))

    def find_by_family_id(self, family_id: str) -> List[PatternRecord]:
        return list(self.by_family_id.get(str(family_id), []))


def load_pattern_lookup(patterns_dir: Path) -> PatternLookup:
    registry_path = patterns_dir / "registry.json"
    if not registry_path.exists():
        raise FileNotFoundError(f"Pattern registry not found: {registry_path}")

    data = json.loads(registry_path.read_text(encoding="utf-8"))
    patterns = [PatternRecord.from_dict(item) for item in data.get("patterns", [])]

    active_patterns = [pattern for pattern in patterns if pattern.status == "active"]

    by_family_id: Dict[str, List[PatternRecord]] = {}
    by_mask: Dict[str, List[PatternRecord]] = {}

    for pattern in active_patterns:
        if pattern.family_id is not None:
            by_family_id.setdefault(str(pattern.family_id), []).append(pattern)

        by_mask.setdefault(str(pattern.mask81), []).append(pattern)

    return PatternLookup(
        by_id={str(pattern.pattern_id): pattern for pattern in active_patterns},
        by_mask=by_mask,
        by_family_id=by_family_id,
    )