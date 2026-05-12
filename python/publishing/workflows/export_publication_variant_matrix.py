from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and export a full locale × layout publication variant matrix."
    )
    parser.add_argument("--book-dir", required=True, help="Path to the built book directory.")
    parser.add_argument("--base-spec", required=True, help="Path to the canonical base publication spec JSON file.")
    parser.add_argument("--locales", nargs="+", required=True, help="Locale tokens such as de fr it es en.")
    parser.add_argument("--layouts", nargs="+", required=True, help="Layout preset ids such as 1up 2up 4up 6up 12up.")
    parser.add_argument(
        "--solution-layout",
        default=None,
        help="Optional independent solution layout preset id applied to every variant, e.g. 12up.",
    )
    parser.add_argument(
        "--solution-booklet",
        default=None,
        help=(
            "Optional solution booklet layout preset id applied to every variant, e.g. 12up. "
            "Each exported bundle receives solution_booklet.pdf."
        ),
    )
    parser.add_argument("--output-publications-dir", required=True, help="Directory where built publication packages are written.")
    parser.add_argument("--output-bundles-dir", required=True, help="Directory where exported bundles are written.")
    parser.add_argument("--output-compiled-specs-dir", default=None, help="Optional directory for resolved compiled specs.")
    parser.add_argument("--summary-path", default=None, help="Optional explicit path for the matrix summary JSON.")
    parser.add_argument("--stop-on-first-failure", action="store_true", help="Abort the matrix run after the first failed variant.")

    parser.add_argument(
        "--cover-design-id",
        default=None,
        help="Optional cover design id override passed to every variant bundle.",
    )
    parser.add_argument(
        "--cover-variables-json",
        default=None,
        help="Optional JSON file containing cover_design.variables overrides passed to every variant bundle.",
    )
    parser.add_argument(
        "--cover-preset-file",
        default=None,
        help="Optional JSON file declaring cover choices by locale/layout.",
    )
    parser.add_argument(
        "--skip-generated-cover",
        action="store_true",
        help="Keep legacy/exporter cover.pdf instead of replacing it with the cover-design pipeline output.",
    )

    return parser.parse_args()


def _normalize_tokens(values: List[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        token = str(value or "").strip()
        if not token:
            continue
        lowered = token.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        result.append(token)
    return result


def _default_summary_path(*, output_bundles_dir: Path) -> Path:
    return output_bundles_dir / f"variant_matrix_summary__{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"


def _run_variant(
    *,
    book_dir: Path,
    base_spec: Path,
    locale: str,
    layout: str,
    output_publications_dir: Path,
    output_bundles_dir: Path,
    output_compiled_specs_dir: Path | None,
    cover_design_id: str | None,
    cover_variables_json: Path | None,
    cover_preset_file: Path | None,
    skip_generated_cover: bool,
    solution_layout: str | None,
    solution_booklet: str | None,
) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        "-m",
        "python.publishing.workflows.export_publication_variant_bundle",
        "--book-dir",
        str(book_dir),
        "--base-spec",
        str(base_spec),
        "--locale",
        str(locale),
        "--layout",
        str(layout),
        "--output-publications-dir",
        str(output_publications_dir),
        "--output-bundles-dir",
        str(output_bundles_dir),
    ]

    if output_compiled_specs_dir is not None:
        cmd.extend(["--output-compiled-specs-dir", str(output_compiled_specs_dir)])

    if cover_design_id:
        cmd.extend(["--cover-design-id", str(cover_design_id)])

    if cover_variables_json is not None:
        cmd.extend(["--cover-variables-json", str(cover_variables_json)])

    if cover_preset_file is not None:
        cmd.extend(["--cover-preset-file", str(cover_preset_file)])

    if skip_generated_cover:
        cmd.append("--skip-generated-cover")

    if solution_layout:
        cmd.extend(["--solution-layout", str(solution_layout)])

    if solution_booklet:
        cmd.extend(["--solution-booklet", str(solution_booklet)])

    started_at = _now_iso()
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    finished_at = _now_iso()

    stdout = result.stdout or ""
    stderr = result.stderr or ""

    publication_dir = None
    bundle_dir = None
    resolved_publication_id = None

    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("Resolved publication id:"):
            resolved_publication_id = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Publication dir:"):
            publication_dir = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Bundle dir:"):
            bundle_dir = stripped.split(":", 1)[1].strip()

    return {
        "locale": locale,
        "layout": layout,
        "solution_layout": solution_layout,
        "solution_booklet": solution_booklet,
        "cover_design_id": cover_design_id,
        "cover_variables_json": str(cover_variables_json) if cover_variables_json else None,
        "cover_preset_file": str(cover_preset_file) if cover_preset_file else None,
        "skip_generated_cover": skip_generated_cover,
        "status": "built" if result.returncode == 0 else "failed",
        "returncode": int(result.returncode),
        "started_at": started_at,
        "finished_at": finished_at,
        "publication_id": resolved_publication_id,
        "publication_dir": publication_dir,
        "bundle_dir": bundle_dir,
        "stdout": stdout,
        "stderr": stderr,
    }


def main() -> int:
    args = _parse_args()

    book_dir = Path(args.book_dir).resolve()
    base_spec = Path(args.base_spec).resolve()
    output_publications_dir = Path(args.output_publications_dir).resolve()
    output_bundles_dir = Path(args.output_bundles_dir).resolve()
    output_compiled_specs_dir = Path(args.output_compiled_specs_dir).resolve() if args.output_compiled_specs_dir else None
    cover_variables_json = Path(args.cover_variables_json).resolve() if args.cover_variables_json else None
    cover_preset_file = Path(args.cover_preset_file).resolve() if args.cover_preset_file else None

    locales = _normalize_tokens(args.locales)
    layouts = _normalize_tokens(args.layouts)

    output_bundles_dir.mkdir(parents=True, exist_ok=True)
    if output_compiled_specs_dir is not None:
        output_compiled_specs_dir.mkdir(parents=True, exist_ok=True)

    summary_path = Path(args.summary_path).resolve() if args.summary_path else _default_summary_path(output_bundles_dir=output_bundles_dir)

    print("=" * 72)
    print("export_publication_variant_matrix.py starting")
    print("=" * 72)
    print(f"Book dir:                {book_dir}")
    print(f"Base spec:               {base_spec}")
    print(f"Locales:                 {' '.join(locales)}")
    print(f"Layouts:                 {' '.join(layouts)}")
    if args.solution_layout:
        print(f"Solution layout:         {args.solution_layout}")
    if args.solution_booklet:
        print(f"Solution booklet:        {args.solution_booklet}")
    print(f"Output publications dir: {output_publications_dir}")
    print(f"Output bundles dir:      {output_bundles_dir}")
    if args.cover_design_id:
        print(f"Cover design override:   {args.cover_design_id}")
    if cover_variables_json:
        print(f"Cover variables JSON:    {cover_variables_json}")
    if cover_preset_file:
        print(f"Cover preset file:       {cover_preset_file}")
    if args.skip_generated_cover:
        print("Generated cover:         skipped")
    print(f"Summary path:            {summary_path}")
    print("=" * 72)

    combinations = [(locale, layout) for locale in locales for layout in layouts]
    total = len(combinations)

    results: List[Dict[str, Any]] = []

    for idx, (locale, layout) in enumerate(combinations, start=1):
        print("-" * 72)
        print(f"[{idx}/{total}] Running locale={locale} layout={layout}")
        print("-" * 72)

        result = _run_variant(
            book_dir=book_dir,
            base_spec=base_spec,
            locale=locale,
            layout=layout,
            output_publications_dir=output_publications_dir,
            output_bundles_dir=output_bundles_dir,
            output_compiled_specs_dir=output_compiled_specs_dir,
            cover_design_id=args.cover_design_id,
            cover_variables_json=cover_variables_json,
            cover_preset_file=cover_preset_file,
            skip_generated_cover=bool(args.skip_generated_cover),
            solution_layout=args.solution_layout,
            solution_booklet=args.solution_booklet,
        )
        results.append(result)

        if result["stdout"]:
            print(result["stdout"].rstrip())
        if result["status"] == "failed" and result["stderr"]:
            print(result["stderr"].rstrip())

        print(f"Result:                  {result['status']}")
        if result.get("publication_id"):
            print(f"Publication id:          {result['publication_id']}")
        if result.get("bundle_dir"):
            print(f"Bundle dir:              {result['bundle_dir']}")

        if result["status"] == "failed" and args.stop_on_first_failure:
            print("Stopping early because --stop-on-first-failure was set.")
            break

    built = [r for r in results if r["status"] == "built"]
    failed = [r for r in results if r["status"] == "failed"]

    summary = {
        "generated_at": _now_iso(),
        "book_dir": str(book_dir),
        "base_spec": str(base_spec),
        "output_publications_dir": str(output_publications_dir),
        "output_bundles_dir": str(output_bundles_dir),
        "output_compiled_specs_dir": str(output_compiled_specs_dir) if output_compiled_specs_dir else None,
        "cover_design_id": args.cover_design_id,
        "cover_variables_json": str(cover_variables_json) if cover_variables_json else None,
        "cover_preset_file": str(cover_preset_file) if cover_preset_file else None,
        "skip_generated_cover": bool(args.skip_generated_cover),
        "requested_locales": locales,
        "requested_layouts": layouts,
        "requested_solution_layout": args.solution_layout,
        "requested_solution_booklet": args.solution_booklet,
        "total_requested": total,
        "built_count": len(built),
        "failed_count": len(failed),
        "results": results,
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 72)
    print("Matrix summary")
    print("=" * 72)
    print(f"Requested:               {total}")
    print(f"Built:                   {len(built)}")
    print(f"Failed:                  {len(failed)}")
    print(f"Summary JSON:            {summary_path}")
    print("=" * 72)
    print("export_publication_variant_matrix.py completed")
    print("=" * 72)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())