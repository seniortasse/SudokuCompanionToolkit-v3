from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import yaml

class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def load_yaml(path: str | Path) -> DotDict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return DotDict(data)

def merge_overrides(cfg: Dict[str, Any], **overrides) -> Dict[str, Any]:
    for k, v in overrides.items():
        if v is None:
            continue
        cfg[k] = v
    return cfg