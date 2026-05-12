from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
    }


def _validate_geometry(geometry: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    required = [
        "trim_width_in",
        "trim_height_in",
        "bleed_in",
        "spine_width_in",
        "full_width_in",
        "full_height_in",
        "back_x_in",
        "spine_x_in",
        "front_x_in",
        "trim_y_in",
    ]

    for key in required:
        if key not in geometry:
            errors.append(f"Missing geometry field: {key}")

    if errors:
        return errors

    if float(geometry["trim_width_in"]) <= 0:
        errors.append("trim_width_in must be positive.")
    if float(geometry["trim_height_in"]) <= 0:
        errors.append("trim_height_in must be positive.")
    if float(geometry["bleed_in"]) < 0:
        errors.append("bleed_in cannot be negative.")
    if float(geometry["spine_width_in"]) < 0:
        errors.append("spine_width_in cannot be negative.")

    expected_front_x = (
        float(geometry["back_x_in"])
        + float(geometry["trim_width_in"])
        + float(geometry["spine_width_in"])
    )
    actual_front_x = float(geometry["front_x_in"])

    if abs(expected_front_x - actual_front_x) > 0.001:
        errors.append(
            f"front_x_in mismatch: expected {expected_front_x:.4f}, got {actual_front_x:.4f}"
        )

    return errors


def _build_validation_report(full_wrap_dir: Path) -> dict[str, Any]:
    manifest_path = full_wrap_dir / "generated_full_wrap_cover_assets.json"

    report: dict[str, Any] = {
        "status": "unknown",
        "errors": [],
        "warnings": [],
        "inputs": {
            "full_wrap_dir": str(full_wrap_dir),
        },
        "files": {},
        "geometry": None,
        "puzzle_art": None,
    }

    if not manifest_path.exists():
        report["status"] = "failed"
        report["errors"].append(f"Missing manifest: {manifest_path}")
        return report

    manifest = _load_json(manifest_path)
    report["manifest"] = manifest

    cover_pdf = Path(manifest.get("cover_pdf", ""))
    front_png = Path(manifest.get("front_cover_png", ""))
    back_png_raw = manifest.get("back_cover_png")
    spine_png_raw = manifest.get("spine_cover_png")
    back_png = Path(back_png_raw) if back_png_raw else None
    spine_png = Path(spine_png_raw) if spine_png_raw else None
    context_json = Path(manifest.get("cover_design_context_json", ""))
    puzzle_report_json = Path(manifest.get("cover_puzzle_art_report_json", ""))
    geometry_json = Path(manifest.get("geometry_json", ""))

    report["files"] = {
        "cover_pdf": _file_info(cover_pdf),
        "front_cover_png": _file_info(front_png),
        "back_cover_png": _file_info(back_png) if back_png is not None else {
            "path": None,
            "exists": False,
            "size_bytes": 0,
            "optional": True,
        },
        "spine_cover_png": _file_info(spine_png) if spine_png is not None else {
            "path": None,
            "exists": False,
            "size_bytes": 0,
            "optional": True,
        },
        "cover_design_context_json": _file_info(context_json),
        "cover_puzzle_art_report_json": _file_info(puzzle_report_json),
        "geometry_json": _file_info(geometry_json),
    }

    for label, info in report["files"].items():
        if info.get("optional") and not info["exists"]:
            continue

        if not info["exists"]:
            report["errors"].append(f"Missing expected output file: {label} -> {info['path']}")
        elif info["size_bytes"] <= 0:
            report["errors"].append(f"Output file is empty: {label} -> {info['path']}")

    if geometry_json.exists():
        geometry_payload = _load_json(geometry_json)
        geometry = dict(geometry_payload.get("geometry") or {})
        report["geometry"] = geometry
        report["errors"].extend(_validate_geometry(geometry))

    if puzzle_report_json.exists():
        puzzle_report = _load_json(puzzle_report_json)
        report["puzzle_art"] = puzzle_report

        puzzle_status = puzzle_report.get("status")
        if puzzle_status == "failed":
            report["errors"].append("Puzzle-art report status is failed.")
        elif puzzle_status == "ok_with_warnings":
            report["warnings"].append("Puzzle-art report has warnings.")

        for warning in puzzle_report.get("warnings") or []:
            report["warnings"].append(f"puzzle_art: {warning}")

        for error in puzzle_report.get("errors") or []:
            report["errors"].append(f"puzzle_art: {error}")

    report["status"] = "failed" if report["errors"] else (
        "ok_with_warnings" if report["warnings"] else "ok"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect and validate generated cover full-wrap assets."
    )
    parser.add_argument(
        "--full-wrap-dir",
        required=True,
        help="Directory produced by generate_cover_full_wrap.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional validation report JSON path. Defaults to <full-wrap-dir>/cover_validation_report.json.",
    )

    args = parser.parse_args()

    full_wrap_dir = Path(args.full_wrap_dir)
    out_path = Path(args.out) if args.out else full_wrap_dir / "cover_validation_report.json"

    report = _build_validation_report(full_wrap_dir)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 72)
    print("Cover preview / validation report")
    print("=" * 72)
    print(f"status:     {report['status']}")
    print(f"report:     {out_path}")

    files = report.get("files") or {}
    for label in (
        "cover_pdf",
        "front_cover_png",
        "back_cover_png",
        "spine_cover_png",
        "cover_puzzle_art_report_json",
        "geometry_json",
    ):
        info = files.get(label)
        if info:
            print(f"{label}: {info['path']}")

    if report["warnings"]:
        print("warnings:")
        for warning in report["warnings"]:
            print(f"  - {warning}")

    if report["errors"]:
        print("errors:")
        for error in report["errors"]:
            print(f"  - {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()