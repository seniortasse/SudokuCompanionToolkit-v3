from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .puzzle_record_builder import GeneratorCandidate


def iter_candidates_from_jsonl(path: Path) -> Iterator[GeneratorCandidate]:
    if not path.exists():
        raise FileNotFoundError(f"Generator input JSONL not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc

            yield GeneratorCandidate.from_dict(payload)