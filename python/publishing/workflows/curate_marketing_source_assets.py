from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


DEFAULT_BOOK_ID = "BK-CL9-DW-B01"
DEFAULT_CONFIG = Path(
    "datasets/sudoku_books/classic9/marketing_specs/"
    "BK-CL9-DW-B01.marketing_assets.phase_a2_curation.json"
)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _sha12(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _source_group_dir(a1_root: Path, source_group: str) -> Path:
    base = a1_root / "01_extracted_assets"
    mapping = {
        "cover": base / "cover",
        "page_renders": base / "page_renders",
        "pattern_tiles": base / "pattern_tiles",
        "puzzle_grids": base / "puzzle_grids",
    }
    if source_group not in mapping:
        raise ValueError(f"Unknown source_group: {source_group}")
    return mapping[source_group]


def _glob_files(root: Path, pattern: str) -> List[Path]:
    if not root.exists():
        return []

    # Path.glob supports bracket patterns like page_00[5-9]_*.png.
    matches = sorted(p for p in root.glob(pattern) if p.is_file())
    if matches:
        return matches

    # Fallback for ordinary fnmatch against filenames.
    all_files = sorted(p for p in root.rglob("*") if p.is_file())
    return [p for p in all_files if fnmatch.fnmatch(p.name, pattern)]


def _pick_file(
    files: Sequence[Path],
    *,
    pick_index: Optional[int] = None,
) -> Optional[Path]:
    if not files:
        return None

    if pick_index is None:
        return files[0]

    # Spec uses 1-based indexing because it is easier to read in JSON.
    idx = int(pick_index) - 1
    if idx < 0:
        idx = 0
    if idx >= len(files):
        idx = len(files) - 1
    return files[idx]


def _pick_collection(
    files: Sequence[Path],
    *,
    limit: int,
    selection_mode: str,
) -> List[Path]:
    if limit <= 0 or not files:
        return []

    files = list(files)

    if len(files) <= limit:
        return files

    if selection_mode == "first":
        return files[:limit]

    if selection_mode == "last":
        return files[-limit:]

    if selection_mode == "even_spread":
        if limit == 1:
            return [files[0]]
        indexes = []
        for i in range(limit):
            idx = round(i * (len(files) - 1) / (limit - 1))
            indexes.append(idx)

        out: List[Path] = []
        seen = set()
        for idx in indexes:
            if idx not in seen:
                seen.add(idx)
                out.append(files[idx])
        return out

    return files[:limit]


def _copy_asset(
    src: Path,
    dst: Path,
    *,
    force: bool,
    record: Dict[str, Any],
) -> Dict[str, Any]:
    _ensure_dir(dst.parent)

    if dst.exists() and not force:
        pass
    else:
        shutil.copy2(src, dst)

    out = dict(record)
    out.update(
        {
            "source_path": str(src),
            "output_path": str(dst),
            "sha12": _sha12(dst),
            "bytes": dst.stat().st_size,
        }
    )
    return out


def _copy_single_selector(
    selector: Dict[str, Any],
    *,
    a1_root: Path,
    output_root: Path,
    force: bool,
) -> Dict[str, Any]:
    source_group = str(selector["source_group"])
    pattern = str(selector["match"])
    pick_index = selector.get("pick_index")

    src_dir = _source_group_dir(a1_root, source_group)
    matches = _glob_files(src_dir, pattern)
    picked = _pick_file(matches, pick_index=pick_index)

    base_record = {
        "kind": "curated_asset",
        "id": selector.get("id"),
        "source_group": source_group,
        "match": pattern,
        "pick_index": pick_index,
        "role": selector.get("role"),
        "reason": selector.get("reason"),
        "matches_found": len(matches),
    }

    if picked is None:
        base_record.update(
            {
                "status": "missing",
                "source_dir": str(src_dir),
                "output_path": str(output_root / str(selector.get("output", ""))),
            }
        )
        return base_record

    dst = output_root / str(selector["output"])
    copied = _copy_asset(picked, dst, force=force, record=base_record)
    copied["status"] = "ok"
    return copied


def _copy_collection_selector(
    selector: Dict[str, Any],
    *,
    a1_root: Path,
    output_root: Path,
    force: bool,
) -> Dict[str, Any]:
    source_group = str(selector["source_group"])
    pattern = str(selector["match"])
    limit = int(selector.get("limit") or 0)
    selection_mode = str(selector.get("selection_mode") or "first")
    output_dir = output_root / str(selector["output_dir"])

    src_dir = _source_group_dir(a1_root, source_group)
    matches = _glob_files(src_dir, pattern)
    selected = _pick_collection(matches, limit=limit, selection_mode=selection_mode)

    records: List[Dict[str, Any]] = []
    _ensure_dir(output_dir)

    for idx, src in enumerate(selected, start=1):
        dst = output_dir / f"{idx:02d}_{src.name}"
        record = {
            "kind": "curated_collection_item",
            "collection_id": selector.get("id"),
            "source_group": source_group,
            "match": pattern,
            "selection_mode": selection_mode,
            "role": selector.get("role"),
            "ordinal": idx,
        }
        records.append(_copy_asset(src, dst, force=force, record=record))

    return {
        "kind": "curated_collection",
        "id": selector.get("id"),
        "source_group": source_group,
        "match": pattern,
        "output_dir": str(output_dir),
        "role": selector.get("role"),
        "limit": limit,
        "selection_mode": selection_mode,
        "matches_found": len(matches),
        "selected_count": len(selected),
        "items": records,
        "status": "ok" if selected else "missing",
    }


def _require_pillow() -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Pillow is required for Phase A2 contact sheets. "
            "Install it with: python -m pip install pillow"
        ) from exc


def _font(size: int, bold: bool = False):
    from PIL import ImageFont

    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)

    return ImageFont.load_default()


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = str(value or "#FFFFFF").strip().lstrip("#")
    if len(value) != 6:
        return (255, 255, 255)
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


def _make_thumbnail(src: Path, size: int):
    from PIL import Image

    img = Image.open(src).convert("RGB")
    img.thumbnail((size, size))
    canvas = Image.new("RGB", (size, size), "white")
    x = (size - img.width) // 2
    y = (size - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def _draw_wrapped_text(draw: Any, xy: tuple[int, int], text: str, font: Any, fill: Any, max_width: int, line_gap: int = 4) -> int:
    words = str(text).split()
    lines: List[str] = []
    current = ""

    for word in words:
        trial = word if not current else current + " " + word
        bbox = draw.textbbox((0, 0), trial, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)

    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((0, 0), line, font=font)
        y += (bbox[3] - bbox[1]) + line_gap

    return y


def _collect_banner_preview_assets(output_root: Path, banner_id: str) -> List[Path]:
    banner_dir = output_root / banner_id
    if not banner_dir.exists():
        return []
    return sorted(
        p
        for p in banner_dir.rglob("*.png")
        if p.is_file()
    )


def _create_contact_sheet(
    *,
    output_root: Path,
    banner_sets: Sequence[Dict[str, Any]],
    review_cfg: Dict[str, Any],
    manifest: Dict[str, Any],
) -> Optional[Path]:
    _require_pillow()

    from PIL import Image, ImageDraw

    thumb = int(review_cfg.get("thumb_size_px") or 180)
    bg = _hex_to_rgb(str(review_cfg.get("background") or "#F4F7FB"))
    title = str(review_cfg.get("title") or "Phase A2 Curated Marketing Assets")

    margin = 32
    gap = 20
    label_h = 64
    title_h = 76
    cols = 4
    tile_w = thumb
    row_w = margin * 2 + cols * tile_w + (cols - 1) * gap

    banner_blocks: List[Dict[str, Any]] = []
    total_h = title_h + margin

    for banner in banner_sets:
        banner_id = str(banner["banner_id"])
        assets = _collect_banner_preview_assets(output_root, banner_id)
        rows = max(1, (len(assets) + cols - 1) // cols)
        block_h = label_h + rows * thumb + max(0, rows - 1) * gap + margin
        banner_blocks.append(
            {
                "banner": banner,
                "assets": assets,
                "rows": rows,
                "block_h": block_h,
            }
        )
        total_h += block_h

    # Shared section.
    shared_assets = sorted((output_root / "_shared").glob("*.png"))
    shared_rows = max(1, (len(shared_assets) + cols - 1) // cols)
    shared_block_h = label_h + shared_rows * thumb + max(0, shared_rows - 1) * gap + margin
    total_h += shared_block_h

    img = Image.new("RGB", (row_w, total_h), bg)
    draw = ImageDraw.Draw(img)

    title_font = _font(28, bold=True)
    h_font = _font(20, bold=True)
    small_font = _font(13, bold=False)
    muted = (70, 80, 95)
    navy = (8, 53, 104)

    y = margin
    draw.text((margin, y), title, font=title_font, fill=navy)
    y += title_h

    def draw_asset_grid(section_title: str, section_subtitle: str, assets: Sequence[Path], y: int) -> int:
        draw.text((margin, y), section_title, font=h_font, fill=navy)
        y += 28
        y = _draw_wrapped_text(
            draw,
            (margin, y),
            section_subtitle,
            small_font,
            muted,
            row_w - margin * 2,
            line_gap=2,
        )
        y += 12

        for idx, asset in enumerate(assets):
            row = idx // cols
            col = idx % cols
            x = margin + col * (thumb + gap)
            yy = y + row * (thumb + gap)
            try:
                tile = _make_thumbnail(asset, thumb)
                img.paste(tile, (x, yy))
                draw.rectangle([x, yy, x + thumb - 1, yy + thumb - 1], outline=(210, 215, 225), width=1)
            except Exception:
                draw.rectangle([x, yy, x + thumb - 1, yy + thumb - 1], outline=(210, 80, 80), width=2)
                draw.text((x + 8, yy + 8), "Preview error", font=small_font, fill=(140, 40, 40))

        rows = max(1, (len(assets) + cols - 1) // cols)
        return y + rows * thumb + max(0, rows - 1) * gap + margin

    y = draw_asset_grid(
        "_shared",
        "Shared assets reused across the campaign.",
        shared_assets,
        y,
    )

    for block in banner_blocks:
        banner = block["banner"]
        y = draw_asset_grid(
            str(banner["banner_id"]),
            f"{banner.get('banner_type', '')} • {banner.get('role', '')} • {banner.get('headline', '')}",
            block["assets"],
            y,
        )

    review_dir = output_root / "_review"
    _ensure_dir(review_dir)
    out_path = review_dir / "phase_a2_curated_assets_contact_sheet.png"
    img.save(out_path)

    manifest["contact_sheet"] = str(out_path)
    return out_path


def run(args: argparse.Namespace) -> Dict[str, Any]:
    config_path = Path(args.config)
    config = _read_json(config_path)

    book_id = str(args.book_id or config.get("book_id") or DEFAULT_BOOK_ID)
    a1_root = Path(args.a1_root or config.get("a1_root"))
    output_root = Path(args.out_root or config.get("output_root"))

    if args.clean and output_root.exists():
        shutil.rmtree(output_root)

    _ensure_dir(output_root)
    _ensure_dir(output_root / "_reports")
    _ensure_dir(output_root / "_review")

    records: List[Dict[str, Any]] = []

    records.append(
        {
            "kind": "phase_a2_run",
            "book_id": book_id,
            "config_path": str(config_path),
            "a1_root": str(a1_root),
            "output_root": str(output_root),
            "force": bool(args.force),
            "clean": bool(args.clean),
            "dry_run": bool(args.dry_run),
        }
    )

    if args.dry_run:
        report = {
            "ok": True,
            "dry_run": True,
            "phase": "A2",
            "book_id": book_id,
            "config_path": str(config_path),
            "a1_root": str(a1_root),
            "output_root": str(output_root),
            "shared_asset_count": len(config.get("shared_assets") or []),
            "banner_count": len(config.get("banner_asset_sets") or []),
        }
        _write_json(output_root / "_reports" / "phase_a2_dry_run_report.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report

    # Keep a copy of the curation recipe beside the outputs.
    shutil.copy2(config_path, output_root / "_reports" / config_path.name)

    # Shared assets.
    for selector in config.get("shared_assets") or []:
        records.append(
            _copy_single_selector(
                selector,
                a1_root=a1_root,
                output_root=output_root,
                force=bool(args.force),
            )
        )

    # Banner-specific assets.
    banner_sets = list(config.get("banner_asset_sets") or [])
    for banner in banner_sets:
        banner_id = str(banner["banner_id"])
        banner_dir = output_root / banner_id
        _ensure_dir(banner_dir)

        banner_manifest: Dict[str, Any] = {
            "banner_id": banner_id,
            "banner_type": banner.get("banner_type"),
            "role": banner.get("role"),
            "headline": banner.get("headline"),
            "assets": [],
            "collections": [],
        }

        for selector in banner.get("assets") or []:
            record = _copy_single_selector(
                selector,
                a1_root=a1_root,
                output_root=output_root,
                force=bool(args.force),
            )
            records.append(record)
            banner_manifest["assets"].append(record)

        for selector in banner.get("collections") or []:
            record = _copy_collection_selector(
                selector,
                a1_root=a1_root,
                output_root=output_root,
                force=bool(args.force),
            )
            records.append(record)
            banner_manifest["collections"].append(record)

        _write_json(banner_dir / "_banner_curated_manifest.json", banner_manifest)

    missing = [r for r in records if r.get("status") == "missing"]
    copied_asset_count = sum(1 for r in records if r.get("kind") == "curated_asset" and r.get("status") == "ok")
    copied_collection_item_count = 0
    for r in records:
        if r.get("kind") == "curated_collection":
            copied_collection_item_count += len(r.get("items") or [])

    report: Dict[str, Any] = {
        "ok": len(missing) == 0,
        "phase": "A2",
        "book_id": book_id,
        "a1_root": str(a1_root),
        "output_root": str(output_root),
        "records": records,
        "summary": {
            "total_records": len(records),
            "curated_single_assets": copied_asset_count,
            "curated_collection_items": copied_collection_item_count,
            "missing_records": len(missing),
            "banner_count": len(banner_sets),
        },
    }

    review_cfg = dict(config.get("review_contact_sheets") or {})
    if review_cfg.get("enabled", True):
        contact_sheet = _create_contact_sheet(
            output_root=output_root,
            banner_sets=banner_sets,
            review_cfg=review_cfg,
            manifest=report,
        )
        if contact_sheet:
            report["summary"]["contact_sheet"] = str(contact_sheet)

    _write_json(output_root / "_reports" / "phase_a2_curated_asset_manifest.json", report)

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"[OK] Phase A2 curated assets written to: {output_root}")
    print(f"[OK] Manifest: {output_root / '_reports' / 'phase_a2_curated_asset_manifest.json'}")

    if missing:
        print("[WARN] Some selectors did not find assets:", file=sys.stderr)
        for item in missing:
            print(
                f"  - {item.get('id')} | group={item.get('source_group')} | match={item.get('match')}",
                file=sys.stderr,
            )

    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Curate Phase A1 marketing source assets into banner-specific folders "
            "for the Sudoku marketing banner pipeline."
        )
    )

    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Phase A2 curation config JSON.",
    )
    parser.add_argument(
        "--book-id",
        default=None,
        help=f"Book id. Defaults to config book_id or {DEFAULT_BOOK_ID}.",
    )
    parser.add_argument(
        "--a1-root",
        default=None,
        help="Phase A1 output root. Defaults to config a1_root.",
    )
    parser.add_argument(
        "--out-root",
        default=None,
        help="Phase A2 curated output root. Defaults to config output_root.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing curated files.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the Phase A2 output folder before curating.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and paths without copying files.",
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        run(args)
        return 0
    except Exception as exc:
        print(f"[ERROR] Phase A2 marketing asset curation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())