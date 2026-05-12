from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, Iterable, Mapping, Optional, Sequence

from python.publishing.pattern_library.pattern_registry import PatternRegistry


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _default_stats() -> Dict[str, object]:
    return {
        "generation_attempts": 0,
        "successful_candidates": 0,
        "rejected_requests": 0,
        "unique_successes": 0,
        "human_solvable_successes": 0,
        "total_weight_sum": 0,
        "weight_sample_count": 0,
        "avg_weight": None,
        "min_weight": None,
        "max_weight": None,
        "technique_histogram_aggregate": {},
        "success_rate": None,
        "unique_rate": None,
        "human_solvable_rate": None,
        "last_run_id": None,
        "last_run_at": None,
        "last_attempt_count": 0,
        "last_success_count": 0,
        "last_rejected_count": 0,
    }


def _safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _safe_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _merge_histogram(target: Dict[str, int], source: Mapping[str, object]) -> Dict[str, int]:
    out = {str(k): _safe_int(v) for k, v in dict(target or {}).items()}
    for key, value in dict(source or {}).items():
        name = str(key)
        out[name] = out.get(name, 0) + _safe_int(value)
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def record_production_outcomes(
    *,
    registry: PatternRegistry,
    requests: Sequence[object],
    candidates: Sequence[Mapping[str, object]],
    rejected: Sequence[Mapping[str, object]],
    run_id: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> Dict[str, object]:
    run_timestamp = timestamp or _now_iso()

    attempts_by_pattern_id: Dict[str, int] = defaultdict(int)
    success_by_pattern_id: Dict[str, int] = defaultdict(int)
    rejected_by_pattern_id: Dict[str, int] = defaultdict(int)
    unique_by_pattern_id: Dict[str, int] = defaultdict(int)
    human_by_pattern_id: Dict[str, int] = defaultdict(int)
    weight_sums_by_pattern_id: Dict[str, int] = defaultdict(int)
    weight_counts_by_pattern_id: Dict[str, int] = defaultdict(int)
    min_weight_by_pattern_id: Dict[str, int] = {}
    max_weight_by_pattern_id: Dict[str, int] = {}
    histogram_by_pattern_id: Dict[str, Dict[str, int]] = defaultdict(dict)

    for request in requests:
        pattern = getattr(request, "pattern", None)
        pattern_id = getattr(pattern, "pattern_id", None) if pattern is not None else None
        if pattern_id:
            attempts_by_pattern_id[str(pattern_id)] += 1

    for item in candidates:
        pattern_id = item.get("pattern_id")
        if not pattern_id:
            continue

        pattern_id = str(pattern_id)
        success_by_pattern_id[pattern_id] += 1

        if _safe_bool(item.get("is_unique"), default=False):
            unique_by_pattern_id[pattern_id] += 1

        if _safe_bool(item.get("is_human_solvable"), default=False):
            human_by_pattern_id[pattern_id] += 1

        weight = item.get("weight")
        if weight is not None:
            weight_int = _safe_int(weight)
            weight_sums_by_pattern_id[pattern_id] += weight_int
            weight_counts_by_pattern_id[pattern_id] += 1

            if pattern_id not in min_weight_by_pattern_id:
                min_weight_by_pattern_id[pattern_id] = weight_int
            else:
                min_weight_by_pattern_id[pattern_id] = min(min_weight_by_pattern_id[pattern_id], weight_int)

            if pattern_id not in max_weight_by_pattern_id:
                max_weight_by_pattern_id[pattern_id] = weight_int
            else:
                max_weight_by_pattern_id[pattern_id] = max(max_weight_by_pattern_id[pattern_id], weight_int)

        histogram_by_pattern_id[pattern_id] = _merge_histogram(
            histogram_by_pattern_id.get(pattern_id, {}),
            item.get("technique_histogram") or {},
        )

    for item in rejected:
        pattern_id = item.get("pattern_id")
        if pattern_id:
            rejected_by_pattern_id[str(pattern_id)] += 1

    touched_pattern_ids = set(attempts_by_pattern_id.keys()) | set(success_by_pattern_id.keys()) | set(rejected_by_pattern_id.keys())

    updated_patterns = 0

    for pattern in registry.patterns:
        pattern_id = str(pattern.pattern_id)
        if pattern_id not in touched_pattern_ids:
            continue

        stats = dict(pattern.production_stats or {})
        merged = _default_stats()
        merged.update(stats)

        merged["generation_attempts"] = _safe_int(merged.get("generation_attempts")) + attempts_by_pattern_id.get(pattern_id, 0)
        merged["successful_candidates"] = _safe_int(merged.get("successful_candidates")) + success_by_pattern_id.get(pattern_id, 0)
        merged["rejected_requests"] = _safe_int(merged.get("rejected_requests")) + rejected_by_pattern_id.get(pattern_id, 0)
        merged["unique_successes"] = _safe_int(merged.get("unique_successes")) + unique_by_pattern_id.get(pattern_id, 0)
        merged["human_solvable_successes"] = _safe_int(merged.get("human_solvable_successes")) + human_by_pattern_id.get(pattern_id, 0)

        merged["total_weight_sum"] = _safe_int(merged.get("total_weight_sum")) + weight_sums_by_pattern_id.get(pattern_id, 0)
        merged["weight_sample_count"] = _safe_int(merged.get("weight_sample_count")) + weight_counts_by_pattern_id.get(pattern_id, 0)

        total_weight_sum = _safe_int(merged.get("total_weight_sum"))
        weight_sample_count = _safe_int(merged.get("weight_sample_count"))
        if weight_sample_count > 0:
            merged["avg_weight"] = round(total_weight_sum / weight_sample_count, 3)

        existing_min = merged.get("min_weight")
        existing_max = merged.get("max_weight")

        if pattern_id in min_weight_by_pattern_id:
            new_min = min_weight_by_pattern_id[pattern_id]
            merged["min_weight"] = new_min if existing_min is None else min(_safe_int(existing_min), new_min)

        if pattern_id in max_weight_by_pattern_id:
            new_max = max_weight_by_pattern_id[pattern_id]
            merged["max_weight"] = new_max if existing_max is None else max(_safe_int(existing_max), new_max)

        merged["technique_histogram_aggregate"] = _merge_histogram(
            dict(merged.get("technique_histogram_aggregate") or {}),
            histogram_by_pattern_id.get(pattern_id, {}),
        )

        attempts = _safe_int(merged.get("generation_attempts"))
        successes = _safe_int(merged.get("successful_candidates"))
        unique_successes = _safe_int(merged.get("unique_successes"))
        human_successes = _safe_int(merged.get("human_solvable_successes"))

        merged["success_rate"] = round(successes / attempts, 4) if attempts > 0 else None
        merged["unique_rate"] = round(unique_successes / successes, 4) if successes > 0 else None
        merged["human_solvable_rate"] = round(human_successes / successes, 4) if successes > 0 else None

        merged["last_run_id"] = run_id
        merged["last_run_at"] = run_timestamp
        merged["last_attempt_count"] = attempts_by_pattern_id.get(pattern_id, 0)
        merged["last_success_count"] = success_by_pattern_id.get(pattern_id, 0)
        merged["last_rejected_count"] = rejected_by_pattern_id.get(pattern_id, 0)

        pattern.production_stats = merged
        updated_patterns += 1

    return {
        "updated_patterns": updated_patterns,
        "touched_pattern_ids": sorted(touched_pattern_ids),
        "run_id": run_id,
        "run_timestamp": run_timestamp,
    }