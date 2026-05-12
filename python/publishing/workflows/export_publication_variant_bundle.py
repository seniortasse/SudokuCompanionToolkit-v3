from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from python.publishing.publication_builder.spec_overrides import write_publication_spec_dict
from python.publishing.publication_builder.variant_compiler import compile_publication_variant_spec
from python.publishing.publication_builder.variant_models import PublicationVariantRequest
from python.publishing.publication_builder.publication_package_builder import build_publication_package
from python.publishing.workflows.validate_publication import validate_publication_dir
from python.publishing.pdf_renderer.compat import patch_hashlib_usedforsecurity

from python.publishing.publication_builder.cover_variant_presets import (
    load_cover_variant_preset,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile, build, validate, and export a publication bundle in one command."
    )
    parser.add_argument("--book-dir", required=True, help="Path to the built book directory.")
    parser.add_argument("--base-spec", required=True, help="Path to the canonical base publication spec JSON file.")
    parser.add_argument("--locale", required=True, help="Locale token such as en, fr, de, it, es.")
    parser.add_argument("--layout", required=True, help="Layout preset id such as 1up, 2up, 4up, 6up, or 12up.")
    parser.add_argument(
        "--solution-layout",
        default=None,
        help="Optional independent solution layout preset id such as 1up, 2up, 4up, 6up, or 12up.",
    )
    parser.add_argument(
        "--solution-booklet",
        default=None,
        help=(
            "Optional solution booklet layout preset id such as 12up. "
            "When supplied, the exported bundle receives solution_booklet.pdf. "
            "The booklet keeps front matter, solution pages, and end matter, "
            "but skips puzzle problem sections."
        ),
    )
    parser.add_argument("--output-publications-dir", required=True, help="Directory where built publication packages are written.")
    parser.add_argument("--output-bundles-dir", required=True, help="Directory where exported bundles are written.")
    parser.add_argument("--language", default=None, help="Optional language label override for metadata.language.")
    parser.add_argument("--locale-pack", default=None, help="Optional explicit locale pack JSON path.")
    parser.add_argument(
        "--output-compiled-specs-dir",
        default=None,
        help="Optional directory for resolved compiled specs. Defaults to sibling publication_specs_compiled/ next to the base spec.",
    )
    parser.add_argument("--publication-id", default=None, help="Optional explicit publication id override.")

    parser.add_argument(
        "--cover-design-id",
        default=None,
        help="Optional cover design id override for the generated bundle cover.",
    )
    parser.add_argument(
        "--cover-variables-json",
        default=None,
        help="Optional JSON file containing cover_design.variables overrides.",
    )

    parser.add_argument(
        "--cover-preset-file",
        default=None,
        help="Optional JSON file declaring cover choices by locale/layout.",
    )
    parser.add_argument(
        "--skip-generated-cover",
        action="store_true",
        help="Keep the legacy exported cover.pdf instead of replacing it with the cover-design pipeline output.",
    )


    return parser.parse_args()


def _default_compiled_specs_dir(base_spec_path: Path) -> Path:
    return base_spec_path.parent.parent / "publication_specs_compiled"


def _compiled_spec_filename(*, book_id: str, locale: str, layout: str, base_spec_path: Path) -> str:
    stem = base_spec_path.stem
    return f"{stem}.{str(locale).lower()}.{str(layout).lower()}.compiled.json"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(dict(result[key]), value)
        else:
            result[key] = value
    return result


def _apply_cover_overrides_to_spec(
    spec: dict[str, Any],
    *,
    cover_design_id: str | None,
    cover_variables_json: str | None,
) -> dict[str, Any]:
    if not cover_design_id and not cover_variables_json:
        return spec

    updated = dict(spec)
    cover_design = dict(updated.get("cover_design") or {})

    if cover_design_id:
        cover_design["cover_design_id"] = cover_design_id

    if cover_variables_json:
        override_variables = json.loads(Path(cover_variables_json).read_text(encoding="utf-8"))
        existing_variables = dict(cover_design.get("variables") or {})
        cover_design["variables"] = _deep_merge(existing_variables, override_variables)

    updated["cover_design"] = cover_design
    return updated


def _apply_solution_layout_to_spec(spec: dict[str, Any], *, layout_preset_id: str) -> dict[str, Any]:
    from python.publishing.publication_builder.layout_presets import resolve_layout_preset

    solution_preset = resolve_layout_preset(layout_preset_id=layout_preset_id)
    if solution_preset is None:
        raise ValueError(f"unsupported solution layout preset {layout_preset_id!r}")

    updated = dict(spec)
    layout_cfg = dict(updated.get("layout_config") or {})
    layout_cfg["solution_puzzles_per_page"] = int(solution_preset.puzzles_per_page)
    layout_cfg["solution_rows"] = int(solution_preset.rows)
    layout_cfg["solution_cols"] = int(solution_preset.cols)

    updated["layout_config"] = layout_cfg
    updated["solution_page_template"] = str(solution_preset.solution_page_template)
    return updated


def _booklet_spec_filename(*, compiled_spec_path: Path) -> str:
    return f"{compiled_spec_path.stem}.solution_booklet.json"


def _solution_booklet_publication_id(publication_id: str) -> str:
    base = str(publication_id or "").strip()
    if not base:
        return "SOLUTION-BOOKLET"
    if base.endswith("-SOLUTION-BOOKLET"):
        return base
    return f"{base}-SOLUTION-BOOKLET"



def _replace_bundle_cover_with_generated_cover(
    *,
    publication_spec_path: Path,
    book_dir: Path,
    bundle_dir: Path,
    interior_pdf: Path,
    cover_design_id: str | None,
    cover_variables_json: str | None,
) -> None:
    generated_dir = bundle_dir / "cover_design_generated"

    cmd = [
        sys.executable,
        "-m",
        "python.publishing.workflows.generate_cover_full_wrap",
        "--publication-spec",
        str(publication_spec_path),
        "--book-dir",
        str(book_dir),
        "--interior-pdf",
        str(interior_pdf),
        "--out-dir",
        str(generated_dir),
    ]

    if cover_design_id:
        cmd.extend(["--cover-design-id", str(cover_design_id)])

    if cover_variables_json:
        cmd.extend(["--cover-variables-json", str(cover_variables_json)])

    result = subprocess.run(cmd, check=False, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout.rstrip())

    if result.returncode != 0:
        if result.stderr:
            print(result.stderr.rstrip())
        raise RuntimeError(
            f"generate_cover_full_wrap CLI exited with code {result.returncode}"
        )

    generated_cover_pdf = generated_dir / "cover.pdf"
    final_cover_pdf = bundle_dir / "cover.pdf"

    if not generated_cover_pdf.exists():
        raise RuntimeError(f"Generated cover PDF not found: {generated_cover_pdf}")

    final_cover_pdf.write_bytes(generated_cover_pdf.read_bytes())


def main() -> int:
    args = _parse_args()

    book_dir = Path(args.book_dir).resolve()
    base_spec_path = Path(args.base_spec).resolve()
    output_publications_dir = Path(args.output_publications_dir).resolve()
    output_bundles_dir = Path(args.output_bundles_dir).resolve()
    locale_pack_path = Path(args.locale_pack).resolve() if args.locale_pack else None
    
    cover_variables_json = str(Path(args.cover_variables_json).resolve()) if args.cover_variables_json else None
    cover_preset_file = Path(args.cover_preset_file).resolve() if args.cover_preset_file else None

    print("=" * 72)
    print("export_publication_variant_bundle.py starting")
    print("=" * 72)
    print(f"Book dir:                {book_dir}")
    print(f"Base spec:               {base_spec_path}")
    print(f"Locale:                  {args.locale}")
    print(f"Layout:                  {args.layout}")
    if args.solution_layout:
        print(f"Solution layout:         {args.solution_layout}")
    if args.solution_booklet:
        print(f"Solution booklet:        {args.solution_booklet}")
    print(f"Output publications dir: {output_publications_dir}")
    print(f"Output bundles dir:      {output_bundles_dir}")
    if args.cover_design_id:
        print(f"Cover design override:   {args.cover_design_id}")
    if cover_preset_file:
        print(f"Cover preset file:       {cover_preset_file}")
    if cover_variables_json:
        print(f"Cover variables JSON:    {cover_variables_json}")
    print("=" * 72)

    patch_hashlib_usedforsecurity()

    request = PublicationVariantRequest(
        base_spec_path=base_spec_path,
        locale=args.locale,
        language=args.language,
        locale_pack_path=locale_pack_path,
        layout_preset_id=args.layout,
        publication_id=args.publication_id,
    )

    try:
        resolved_spec = compile_publication_variant_spec(request)
    except Exception as exc:
        print(f"ERROR during compile step: {exc}")
        return 1

    cover_preset = load_cover_variant_preset(
        preset_path=cover_preset_file,
        locale=args.locale,
        layout=args.layout,
    )

    preset_cover_design_id = cover_preset.get("cover_design_id")
    preset_variables = dict(cover_preset.get("variables") or {})

    if preset_cover_design_id or preset_variables:
        existing_cover = dict(resolved_spec.get("cover_design") or {})
        existing_variables = dict(existing_cover.get("variables") or {})

        if preset_cover_design_id:
            existing_cover["cover_design_id"] = str(preset_cover_design_id)

        existing_cover["variables"] = _deep_merge(existing_variables, preset_variables)
        resolved_spec["cover_design"] = existing_cover

    if args.solution_layout:
        try:
            resolved_spec = _apply_solution_layout_to_spec(
                resolved_spec,
                layout_preset_id=str(args.solution_layout),
            )
        except ValueError as exc:
            print(f"ERROR: {exc}")
            return 1

    resolved_spec = _apply_cover_overrides_to_spec(
        resolved_spec,
        cover_design_id=args.cover_design_id,
        cover_variables_json=cover_variables_json,
    )

    compiled_specs_dir = (
        Path(args.output_compiled_specs_dir).resolve()
        if args.output_compiled_specs_dir
        else _default_compiled_specs_dir(base_spec_path)
    )
    compiled_specs_dir.mkdir(parents=True, exist_ok=True)

    compiled_spec_path = compiled_specs_dir / _compiled_spec_filename(
        book_id=str(resolved_spec.get("book_id") or ""),
        locale=str(args.locale),
        layout=str(args.layout),
        base_spec_path=base_spec_path,
    )

    try:
        write_publication_spec_dict(resolved_spec, compiled_spec_path)
    except Exception as exc:
        print(f"ERROR while writing compiled spec: {exc}")
        return 1

    print(f"Compiled spec:           {compiled_spec_path}")
    print(f"Resolved publication id: {resolved_spec.get('publication_id')}")
    print("-" * 72)

    try:
        publication_dir, _package = build_publication_package(
            book_dir=book_dir,
            publication_spec_path=compiled_spec_path,
            output_publications_dir=output_publications_dir,
        )
    except Exception as exc:
        print(f"ERROR during build step: {exc}")
        return 1

    print(f"Publication dir:         {publication_dir}")
    print("-" * 72)

    try:
        qc_summary = validate_publication_dir(publication_dir=publication_dir)
    except Exception as exc:
        print(f"ERROR during validation step: {exc}")
        return 1

    print(f"QC errors:               {qc_summary.get('error_count')}")
    print(f"QC warnings:             {qc_summary.get('warning_count')}")

    qc_errors = list(qc_summary.get("errors") or [])
    qc_warnings = list(qc_summary.get("warnings") or [])

    if qc_errors:
        print("-" * 72)
        print("QC error details:")
        for err in qc_errors[:50]:
            print(f"- {err}")
        if len(qc_errors) > 50:
            print(f"- ... {len(qc_errors) - 50} more QC errors omitted")

    if qc_warnings:
        print("-" * 72)
        print("QC warning details:")
        for warning in qc_warnings[:50]:
            print(f"- {warning}")
        if len(qc_warnings) > 50:
            print(f"- ... {len(qc_warnings) - 50} more QC warnings omitted")

    if int(qc_summary.get("error_count", 0)) > 0:
        print("ERROR: validation failed; bundle export aborted.")
        return 1

    bundle_dir = output_bundles_dir / publication_dir.name

    export_cmd = [
        sys.executable,
        "-m",
        "python.publishing.workflows.export_publication_bundle",
        "--publication-dir",
        str(publication_dir),
        "--output-dir",
        str(bundle_dir),
    ]

    try:
        result = subprocess.run(export_cmd, check=False, capture_output=True, text=True)
    except Exception as exc:
        print(f"ERROR during export step: {exc}")
        return 1

    if result.stdout:
        print(result.stdout.rstrip())
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr.rstrip())
        print(f"ERROR during export step: export_publication_bundle CLI exited with code {result.returncode}")
        return 1
    

    solution_booklet_pdf = None

    if args.solution_booklet:
        try:
            booklet_spec = _apply_solution_layout_to_spec(
                resolved_spec,
                layout_preset_id=str(args.solution_booklet),
            )
        except ValueError as exc:
            print(f"ERROR: {exc}")
            return 1

        # The solution booklet is a standalone interior-only publication:
        # - no cover
        # - solutions forced on
        # - puzzle problem sections skipped
        # - same front matter and end matter as the main book
        booklet_spec["include_cover"] = False
        booklet_spec["include_solutions"] = True
        booklet_spec["publication_id"] = _solution_booklet_publication_id(
            str(resolved_spec.get("publication_id") or "")
        )

        booklet_spec_path = compiled_specs_dir / _booklet_spec_filename(
            compiled_spec_path=compiled_spec_path
        )

        try:
            write_publication_spec_dict(booklet_spec, booklet_spec_path)
        except Exception as exc:
            print(f"ERROR while writing solution booklet spec: {exc}")
            return 1

        print(f"Solution booklet spec:   {booklet_spec_path}")

        try:
            booklet_publication_dir, _booklet_package = build_publication_package(
                book_dir=book_dir,
                publication_spec_path=booklet_spec_path,
                output_publications_dir=output_publications_dir,
                skip_puzzle_sections=True,
            )
        except Exception as exc:
            print(f"ERROR during solution booklet build step: {exc}")
            return 1

        solution_booklet_pdf = bundle_dir / "solution_booklet.pdf"
        temp_booklet_bundle_dir = bundle_dir / "_solution_booklet_export"

        booklet_export_cmd = [
            sys.executable,
            "-m",
            "python.publishing.workflows.export_publication_bundle",
            "--publication-dir",
            str(booklet_publication_dir),
            "--output-dir",
            str(temp_booklet_bundle_dir),
            "--interior-filename",
            "solution_booklet.pdf",
        ]

        try:
            booklet_result = subprocess.run(
                booklet_export_cmd,
                check=False,
                capture_output=True,
                text=True,
            )
        except Exception as exc:
            print(f"ERROR during solution booklet PDF export step: {exc}")
            return 1

        if booklet_result.stdout:
            print(booklet_result.stdout.rstrip())

        if booklet_result.returncode != 0:
            if booklet_result.stderr:
                print(booklet_result.stderr.rstrip())
            print(
                "ERROR during solution booklet PDF export step: "
                f"export_publication_bundle CLI exited with code {booklet_result.returncode}"
            )
            return 1

        temp_solution_booklet_pdf = temp_booklet_bundle_dir / "solution_booklet.pdf"

        if not temp_solution_booklet_pdf.exists():
            print(
                "ERROR during solution booklet PDF export step: "
                f"expected PDF was not produced: {temp_solution_booklet_pdf}"
            )
            return 1

        try:
            shutil.copy2(temp_solution_booklet_pdf, solution_booklet_pdf)
            shutil.rmtree(temp_booklet_bundle_dir, ignore_errors=True)
        except Exception as exc:
            print(f"ERROR while copying solution booklet PDF into bundle: {exc}")
            return 1

        print(f"Solution booklet PDF:    {solution_booklet_pdf}")

    if not args.skip_generated_cover:
        interior_pdf = bundle_dir / "interior.pdf"

        if not interior_pdf.exists():
            print(f"ERROR: cannot generate catalog cover because interior PDF was not found: {interior_pdf}")
            return 1

        try:
            _replace_bundle_cover_with_generated_cover(
                publication_spec_path=compiled_spec_path,
                book_dir=book_dir,
                bundle_dir=bundle_dir,
                interior_pdf=interior_pdf,
                cover_design_id=args.cover_design_id,
                cover_variables_json=cover_variables_json,
            )
        except Exception as exc:
            print(f"ERROR during generated cover replacement step: {exc}")
            return 1

        print(f"Generated cover PDF:     {bundle_dir / 'cover.pdf'}")

    print("-" * 72)
    print(f"Bundle dir:              {bundle_dir}")
    print(f"Interior PDF:            {bundle_dir / 'interior.pdf'}")
    print(f"Cover PDF:               {bundle_dir / 'cover.pdf'}")
    if solution_booklet_pdf:
        print(f"Solution booklet PDF:    {solution_booklet_pdf}")
    if (bundle_dir / "previews").exists():
        print(f"Previews dir:            {bundle_dir / 'previews'}")
    else:
        print("Previews dir:            skipped")
    print("=" * 72)
    print("export_publication_variant_bundle.py completed successfully")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())