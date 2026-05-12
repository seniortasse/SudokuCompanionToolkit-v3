from __future__ import annotations

import json
from pathlib import Path

from python.publishing.schemas.models import PublicationSpec


def load_publication_spec(path: Path) -> PublicationSpec:
    if not path.exists():
        raise FileNotFoundError(f"Publication spec not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    return PublicationSpec.from_dict(data)