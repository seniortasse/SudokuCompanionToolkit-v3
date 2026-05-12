from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from python.publishing.pattern_library.pattern_stats import record_production_outcomes
from python.publishing.pattern_library.pattern_store import (
    load_pattern_store,
    rebuild_compiled_pattern_artifacts,
)
from python.publishing.puzzle_catalog.pattern_production import (
    build_production_requests,
    parse_legacy_output_workbook,
    run_legacy_pattern_generator,
    select_patterns_from_catalog,
    write_candidates_jsonl,
    write_pattern_requests_workbook,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class PatternProductionJobSpec:
    job_id: str
    description: str
    patterns_dir: str
    output_jsonl: str
    generator_root: str
    append: bool
    pattern_ids: List[str]
    family_ids: List[str]
    tags_any: List[str]
    max_patterns: Optional[int]
    min_clue_count: Optional[int]
    max_clue_count: Optional[int]
    min_aesthetic_score: Optional[float]
    min_print_score: Optional[float]
    min_legibility_score: Optional[float]
    count: int
    charset: str
    weight_min: Optional[int]
    weight_max: Optional[int]
    technique_count_min: Optional[int]
    technique_count_max: Optional[int]
    required_techniques: List[str]
    excluded_techniques: List[str]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PatternProductionJobSpec":
        selector = dict(data.get("selector") or {})
        production = dict(data.get("production") or {})
        output = dict(data.get("output") or {})

        job_id = str(data.get("job_id") or "").strip()
        if not job_id:
            raise ValueError("job_id is required")

        count = production.get("count")
        if count is None:
            raise ValueError("production.count is required")

        patterns_dir = str(output.get("patterns_dir") or "datasets/sudoku_books/classic9/patterns")
        output_jsonl = str(output.get("output_jsonl") or "runs/publishing/classic9/puzzle_generation/candidates.jsonl")
        generator_root = str(output.get("generator_root") or "python/puzzle_generator/pattern_sudoku_all_sizes")

        return cls(
            job_id=job_id,
            description=str(data.get("description") or ""),
            patterns_dir=patterns_dir,
            output_jsonl=output_jsonl,
            generator_root=generator_root,
            append=bool(output.get("append", True)),
            pattern_ids=[str(x) for x in list(selector.get("pattern_ids") or [])],
            family_ids=[str(x) for x in list(selector.get("family_ids") or [])],
            tags_any=[str(x) for x in list(selector.get("tags_any") or [])],
            max_patterns=selector.get("max_patterns"),
            min_clue_count=selector.get("min_clue_count"),
            max_clue_count=selector.get("max_clue_count"),
            min_aesthetic_score=selector.get("min_aesthetic_score"),
            min_print_score=selector.get("min_print_score"),
            min_legibility_score=selector.get("min_legibility_score"),
            count=int(count),
            charset=str(production.get("charset") or "123456789"),
            weight_min=production.get("weight_min"),
            weight_max=production.get("weight_max"),
            technique_count_min=production.get("technique_count_min"),
            technique_count_max=production.get("technique_count_max"),
            required_techniques=[str(x) for x in list(production.get("required_techniques") or [])],
            excluded_techniques=[str(x) for x in list(production.get("excluded_techniques") or [])],
        )


def load_pattern_production_job(job_path: Path) -> PatternProductionJobSpec:
    payload = json.loads(Path(job_path).read_text(encoding="utf-8"))
    return PatternProductionJobSpec.from_dict(payload)


def run_pattern_production_job(job_spec: PatternProductionJobSpec) -> Dict[str, Any]:
    patterns_dir = Path(job_spec.patterns_dir)
    output_jsonl = Path(job_spec.output_jsonl)
    generator_root = Path(job_spec.generator_root)

    registry = load_pattern_store(patterns_dir)
    selected_patterns = select_patterns_from_catalog(
        registry=registry,
        pattern_ids=job_spec.pattern_ids,
        family_ids=job_spec.family_ids,
        tags_any=job_spec.tags_any,
        min_clue_count=job_spec.min_clue_count,
        max_clue_count=job_spec.max_clue_count,
        min_aesthetic_score=job_spec.min_aesthetic_score,
        min_print_score=job_spec.min_print_score,
        min_legibility_score=job_spec.min_legibility_score,
        max_patterns=job_spec.max_patterns,
    )

    if not selected_patterns:
        raise ValueError(f"No active patterns matched the selector for job {job_spec.job_id}")

    requests = build_production_requests(
        patterns=selected_patterns,
        count=job_spec.count,
    )

    run_dir = output_jsonl.parent
    run_dir.mkdir(parents=True, exist_ok=True)

    request_workbook = run_dir / f"{job_spec.job_id}__input.xlsx"
    write_pattern_requests_workbook(
        requests=requests,
        workbook_path=request_workbook,
    )

    request_lookup = {request.request_id: request.pattern for request in requests}

    output_workbook, stdout_text, stderr_text = run_legacy_pattern_generator(
        generator_root=generator_root,
        input_workbook=request_workbook,
        charset=job_spec.charset,
        weight_min=job_spec.weight_min,
        weight_max=job_spec.weight_max,
        technique_count_min=job_spec.technique_count_min,
        technique_count_max=job_spec.technique_count_max,
        required_techniques=job_spec.required_techniques,
        excluded_techniques=job_spec.excluded_techniques,
    )

    candidates, rejected = parse_legacy_output_workbook(
        output_workbook=output_workbook,
        request_lookup=request_lookup,
    )

    write_candidates_jsonl(
        candidates=candidates,
        output_jsonl=output_jsonl,
        append=job_spec.append,
    )

    stats_summary = record_production_outcomes(
        registry=registry,
        requests=requests,
        candidates=candidates,
        rejected=rejected,
        run_id=job_spec.job_id,
        timestamp=_now_iso(),
    )
    artifact_paths = rebuild_compiled_pattern_artifacts(registry, patterns_dir)

    report = {
        "timestamp": _now_iso(),
        "job_id": job_spec.job_id,
        "description": job_spec.description,
        "patterns_dir": str(patterns_dir),
        "generator_root": str(generator_root),
        "output_jsonl": str(output_jsonl),
        "request_workbook": str(request_workbook),
        "output_workbook": str(output_workbook),
        "selected_pattern_count": len(selected_patterns),
        "selected_pattern_ids": [p.pattern_id for p in selected_patterns],
        "request_count": len(requests),
        "candidate_count": len(candidates),
        "rejected_count": len(rejected),
        "pattern_stats_updated": int(stats_summary["updated_patterns"]),
        "pattern_stats_touched_ids": list(stats_summary["touched_pattern_ids"]),
        "catalog_path": str(patterns_dir / "pattern_catalog.jsonl"),
        "registry_path": str(artifact_paths["registry"]),
    }

    report_path = run_dir / f"{job_spec.job_id}__report.json"
    stdout_path = run_dir / f"{job_spec.job_id}__stdout.txt"
    stderr_path = run_dir / f"{job_spec.job_id}__stderr.txt"
    rejected_path = run_dir / f"{job_spec.job_id}__rejected.json"

    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    stdout_path.write_text(stdout_text, encoding="utf-8")
    stderr_path.write_text(stderr_text, encoding="utf-8")
    rejected_path.write_text(json.dumps(rejected, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "job_id": job_spec.job_id,
        "patterns_dir": patterns_dir,
        "output_jsonl": output_jsonl,
        "request_workbook": request_workbook,
        "output_workbook": output_workbook,
        "report_path": report_path,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "rejected_path": rejected_path,
        "selected_patterns": selected_patterns,
        "requests": requests,
        "candidates": candidates,
        "rejected": rejected,
        "artifact_paths": artifact_paths,
        "stats_summary": stats_summary,
    }