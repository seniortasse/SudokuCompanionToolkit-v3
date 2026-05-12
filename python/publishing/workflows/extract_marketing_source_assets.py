from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_BOOK_ID = "BK-CL9-DW-B01"
DEFAULT_MARKETING_SPEC = Path(
    "datasets/sudoku_books/classic9/marketing_specs/"
    "BK-CL9-DW-B01.marketing_assets.phase_a1.json"
)
DEFAULT_PATTERNS_DIR = Path("datasets/sudoku_books/classic9/patterns")
DEFAULT_BOOKS_ROOT = Path("datasets/sudoku_books/classic9/books")
DEFAULT_PUBLICATION_SPECS_DIR = Path("datasets/sudoku_books/classic9/publication_specs")
DEFAULT_EXPORT_BUNDLES_ROOT = Path("exports/sudoku_books/bundles")
DEFAULT_OUTPUT_ROOT = Path("runs/marketing/classic9")


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


def _safe_slug(value: str) -> str:
    out = []
    for ch in str(value).strip().lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in (" ", "-", "_", ".", "/", "\\"):
            out.append("_")
    slug = "".join(out)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "asset"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _copy_if_present(
    src: Optional[Path],
    dst: Path,
    *,
    force: bool,
    manifest: List[Dict[str, Any]],
    role: str,
) -> Optional[Path]:
    if not src:
        return None
    src = Path(src)
    if not src.exists():
        return None

    _ensure_dir(dst.parent)
    if dst.exists() and not force:
        pass
    else:
        shutil.copy2(src, dst)

    manifest.append(
        {
            "kind": "copied_source",
            "role": role,
            "source_path": str(src),
            "output_path": str(dst),
            "sha12": _sha12(dst),
            "bytes": dst.stat().st_size,
        }
    )
    return dst


def _find_first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path and path.exists():
            return path
    return None


def _discover_publication_spec(book_id: str) -> Optional[Path]:
    candidates = [
        DEFAULT_PUBLICATION_SPECS_DIR / f"{book_id}.kdp.6up.8_5x11.json",
        DEFAULT_PUBLICATION_SPECS_DIR / f"{book_id}.kdp.6up.8_5x11_bw.json",
        DEFAULT_PUBLICATION_SPECS_DIR / f"{book_id}.json",
    ]
    found = _find_first_existing(candidates)
    if found:
        return found

    if DEFAULT_PUBLICATION_SPECS_DIR.exists():
        matches = sorted(DEFAULT_PUBLICATION_SPECS_DIR.glob(f"**/*{book_id}*.json"))
        if matches:
            return matches[0]
    return None


def _discover_interior_pdf(book_id: str) -> Optional[Path]:
    if not DEFAULT_EXPORT_BUNDLES_ROOT.exists():
        return None

    patterns = [
        f"**/*{book_id}*interior*.pdf",
        f"**/{book_id}*/interior*.pdf",
        "**/interior*.pdf",
        "**/*Interior*.pdf",
    ]

    for pattern in patterns:
        matches = sorted(DEFAULT_EXPORT_BUNDLES_ROOT.glob(pattern))
        if matches:
            return matches[0]
    return None


def _discover_cover_image(book_id: str) -> Optional[Path]:
    search_roots = [
        DEFAULT_EXPORT_BUNDLES_ROOT,
        Path("runs/publishing/classic9"),
        Path("datasets/sudoku_books/classic9/publications"),
    ]

    patterns = [
        f"**/*{book_id}*front*.png",
        f"**/*{book_id}*cover*.png",
        "**/cover_design_generated/*front*.png",
        "**/previews/*front*.png",
        "**/*front_cover*.png",
        "**/*cover_front*.png",
        "**/*front*.jpg",
        "**/*cover*.jpg",
    ]

    for root in search_roots:
        if not root.exists():
            continue
        for pattern in patterns:
            matches = sorted(root.glob(pattern))
            if matches:
                return matches[0]
    return None


def _require_pillow():
    try:
        from PIL import Image, ImageDraw, ImageFont  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Pillow is required for marketing asset extraction. "
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


def _fit_cover_image(src: Path, dst: Path, *, max_height: int = 1400) -> None:
    _require_pillow()
    from PIL import Image

    img = Image.open(src).convert("RGB")
    if img.height > max_height:
        ratio = max_height / float(img.height)
        img = img.resize((int(img.width * ratio), max_height), Image.LANCZOS)
    _ensure_dir(dst.parent)
    img.save(dst)


def _render_pdf_pages(
    pdf_path: Path,
    selected_pages: Sequence[Dict[str, Any]],
    out_dir: Path,
    *,
    dpi: int,
    force: bool,
    manifest: List[Dict[str, Any]],
) -> None:
    try:
        import fitz  # PyMuPDF
    except Exception as exc:
        raise RuntimeError(
            "PyMuPDF is required to render PDF pages. "
            "Install it with: python -m pip install pymupdf"
        ) from exc

    if not pdf_path.exists():
        raise FileNotFoundError(f"Interior PDF not found: {pdf_path}")

    _ensure_dir(out_dir)

    doc = fitz.open(str(pdf_path))
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    for page_spec in selected_pages:
        page_number = int(page_spec["page"])
        slug = _safe_slug(page_spec.get("slug") or f"page_{page_number:03d}")

        if page_number < 1 or page_number > len(doc):
            manifest.append(
                {
                    "kind": "pdf_page_render_skipped",
                    "reason": "page_out_of_range",
                    "source_path": str(pdf_path),
                    "page": page_number,
                    "pdf_page_count": len(doc),
                    "slug": slug,
                }
            )
            continue

        out_path = out_dir / f"page_{page_number:03d}_{slug}.png"
        if out_path.exists() and not force:
            pass
        else:
            page = doc.load_page(page_number - 1)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            pix.save(str(out_path))

        manifest.append(
            {
                "kind": "pdf_page_render",
                "source_path": str(pdf_path),
                "output_path": str(out_path),
                "page": page_number,
                "slug": slug,
                "role": page_spec.get("role"),
                "reason": page_spec.get("reason"),
                "dpi": dpi,
                "sha12": _sha12(out_path),
                "bytes": out_path.stat().st_size,
            }
        )

    doc.close()


def _normalize_grid_string(value: Any) -> Optional[str]:
    if isinstance(value, str):
        s = value.strip()
        if len(s) == 81 and all(ch in "0123456789." for ch in s):
            return "".join("." if ch in "0." else ch for ch in s)

    if isinstance(value, list) and len(value) == 81:
        out = []
        for item in value:
            if item in (None, "", 0, ".", "0"):
                out.append(".")
            elif isinstance(item, int) and 1 <= item <= 9:
                out.append(str(item))
            elif isinstance(item, str) and item.strip() in list("123456789"):
                out.append(item.strip())
            else:
                return None
        return "".join(out)

    return None


def _normalize_mask_string(value: Any) -> Optional[str]:
    if isinstance(value, str):
        s = value.strip()
        if len(s) == 81 and all(ch in "01.#xX*" for ch in s):
            return "".join("1" if ch in "1#xX*" else "0" for ch in s)
        if len(s) == 81 and all(ch in "0123456789." for ch in s):
            return "".join("0" if ch in "0." else "1" for ch in s)

    if isinstance(value, list) and len(value) == 81:
        out = []
        for item in value:
            if isinstance(item, bool):
                out.append("1" if item else "0")
            elif isinstance(item, int):
                out.append("1" if item else "0")
            elif isinstance(item, str):
                out.append("0" if item.strip() in ("", "0", ".", "false", "False") else "1")
            else:
                return None
        return "".join(out)

    return None


def _walk_values(obj: Any) -> Iterable[Any]:
    yield obj
    if isinstance(obj, dict):
        for value in obj.values():
            yield from _walk_values(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _walk_values(value)


def _find_grid81(obj: Any) -> Optional[str]:
    preferred_keys = [
        "givens81",
        "grid81",
        "puzzle81",
        "puzzle",
        "givens",
        "initial_grid",
        "grid",
    ]

    if isinstance(obj, dict):
        for key in preferred_keys:
            if key in obj:
                normalized = _normalize_grid_string(obj[key])
                if normalized:
                    return normalized

    for value in _walk_values(obj):
        normalized = _normalize_grid_string(value)
        if normalized:
            return normalized

    return None


def _find_mask81(obj: Any) -> Optional[str]:
    preferred_keys = [
        "mask81",
        "givens_mask81",
        "pattern_mask81",
        "mask",
        "givens_mask",
        "pattern_mask",
        "cells",
        "pattern",
    ]

    if isinstance(obj, dict):
        for key in preferred_keys:
            if key in obj:
                normalized = _normalize_mask_string(obj[key])
                if normalized:
                    return normalized

    for value in _walk_values(obj):
        normalized = _normalize_mask_string(value)
        if normalized:
            return normalized

    return None


def _draw_sudoku_grid(
    draw: Any,
    *,
    x0: int,
    y0: int,
    size: int,
    thin: int = 2,
    thick: int = 5,
    line_fill: Tuple[int, int, int] = (20, 20, 20),
) -> None:
    cell = size / 9.0
    for i in range(10):
        width = thick if i % 3 == 0 else thin
        x = int(round(x0 + i * cell))
        y = int(round(y0 + i * cell))
        draw.line([(x, y0), (x, y0 + size)], fill=line_fill, width=width)
        draw.line([(x0, y), (x0 + size, y)], fill=line_fill, width=width)


def _render_pattern_tile(mask81: str, dst: Path, *, size: int) -> None:
    _require_pillow()
    from PIL import Image, ImageDraw

    margin = max(12, int(size * 0.07))
    grid_size = size - 2 * margin
    cell = grid_size / 9.0

    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)

    for idx, ch in enumerate(mask81):
        if ch == "1":
            r = idx // 9
            c = idx % 9
            x0 = int(round(margin + c * cell))
            y0 = int(round(margin + r * cell))
            x1 = int(round(margin + (c + 1) * cell))
            y1 = int(round(margin + (r + 1) * cell))
            draw.rectangle([x0, y0, x1, y1], fill=(18, 18, 18))

    _draw_sudoku_grid(
        draw,
        x0=margin,
        y0=margin,
        size=grid_size,
        thin=max(1, size // 180),
        thick=max(2, size // 70),
    )

    _ensure_dir(dst.parent)
    img.save(dst)


def _render_puzzle_grid(grid81: str, dst: Path, *, size: int) -> None:
    _require_pillow()
    from PIL import Image, ImageDraw

    margin = max(24, int(size * 0.07))
    grid_size = size - 2 * margin
    cell = grid_size / 9.0

    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)

    _draw_sudoku_grid(
        draw,
        x0=margin,
        y0=margin,
        size=grid_size,
        thin=max(1, size // 220),
        thick=max(3, size // 95),
    )

    digit_font = _font(max(18, int(cell * 0.58)), bold=False)

    for idx, ch in enumerate(grid81):
        if ch == ".":
            continue

        r = idx // 9
        c = idx % 9
        cx = margin + (c + 0.5) * cell
        cy = margin + (r + 0.5) * cell

        bbox = draw.textbbox((0, 0), ch, font=digit_font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text(
            (int(cx - tw / 2), int(cy - th / 2 - cell * 0.03)),
            ch,
            fill=(10, 10, 10),
            font=digit_font,
        )

    _ensure_dir(dst.parent)
    img.save(dst)


def _pick_even_spread(items: Sequence[Path], limit: int) -> List[Path]:
    if limit <= 0:
        return []
    if len(items) <= limit:
        return list(items)
    if limit == 1:
        return [items[0]]

    indexes = []
    for i in range(limit):
        idx = round(i * (len(items) - 1) / (limit - 1))
        indexes.append(idx)

    seen = set()
    out = []
    for idx in indexes:
        if idx not in seen:
            seen.add(idx)
            out.append(items[idx])
    return out


def _render_pattern_tiles(
    patterns_dir: Path,
    out_dir: Path,
    *,
    tile_size: int,
    max_tiles: int,
    preferred_pattern_ids: Sequence[str],
    force: bool,
    manifest: List[Dict[str, Any]],
) -> None:
    if not patterns_dir.exists():
        manifest.append(
            {
                "kind": "pattern_tiles_skipped",
                "reason": "patterns_dir_not_found",
                "source_path": str(patterns_dir),
            }
        )
        return

    _ensure_dir(out_dir)

    all_files = sorted(
        p
        for p in patterns_dir.glob("**/*.json")
        if p.is_file()
        and not p.name.startswith("_")
        and p.name.lower() not in {"registry.json", "index.json"}
    )

    selected_files: List[Path] = []

    if preferred_pattern_ids:
        by_stem = {p.stem: p for p in all_files}
        for pattern_id in preferred_pattern_ids:
            found = by_stem.get(pattern_id)
            if found:
                selected_files.append(found)

    if not selected_files:
        selected_files = all_files[:max_tiles]
    else:
        selected_files = selected_files[:max_tiles]

    count = 0
    for path in selected_files:
        try:
            payload = _read_json(path)
        except Exception as exc:
            manifest.append(
                {
                    "kind": "pattern_tile_skipped",
                    "reason": "json_read_error",
                    "source_path": str(path),
                    "error": str(exc),
                }
            )
            continue

        mask81 = _find_mask81(payload)
        if not mask81:
            manifest.append(
                {
                    "kind": "pattern_tile_skipped",
                    "reason": "no_81_cell_mask_found",
                    "source_path": str(path),
                }
            )
            continue

        pattern_id = str(payload.get("pattern_id") or path.stem)
        name = str(payload.get("name") or pattern_id)
        family_name = str(payload.get("family_name") or payload.get("family_id") or "")

        slug = _safe_slug(pattern_id)
        out_path = out_dir / f"{slug}_{tile_size}x{tile_size}.png"

        if out_path.exists() and not force:
            pass
        else:
            _render_pattern_tile(mask81, out_path, size=tile_size)

        manifest.append(
            {
                "kind": "pattern_tile",
                "source_path": str(path),
                "output_path": str(out_path),
                "pattern_id": pattern_id,
                "name": name,
                "family_name": family_name,
                "tile_size_px": tile_size,
                "sha12": _sha12(out_path),
                "bytes": out_path.stat().st_size,
            }
        )
        count += 1

    manifest.append(
        {
            "kind": "pattern_tile_summary",
            "patterns_dir": str(patterns_dir),
            "candidate_json_files": len(all_files),
            "rendered_tiles": count,
            "requested_max_tiles": max_tiles,
        }
    )


def _render_puzzle_grids(
    book_dir: Path,
    out_dir: Path,
    *,
    grid_size: int,
    max_grids: int,
    force: bool,
    manifest: List[Dict[str, Any]],
) -> None:
    puzzles_dir = book_dir / "puzzles"
    if not puzzles_dir.exists():
        manifest.append(
            {
                "kind": "puzzle_grids_skipped",
                "reason": "book_puzzles_dir_not_found",
                "source_path": str(puzzles_dir),
            }
        )
        return

    _ensure_dir(out_dir)

    all_files = sorted(p for p in puzzles_dir.glob("REC-*.json") if p.is_file())
    selected_files = _pick_even_spread(all_files, max_grids)

    count = 0
    for ordinal, path in enumerate(selected_files, start=1):
        try:
            payload = _read_json(path)
        except Exception as exc:
            manifest.append(
                {
                    "kind": "puzzle_grid_skipped",
                    "reason": "json_read_error",
                    "source_path": str(path),
                    "error": str(exc),
                }
            )
            continue

        grid81 = _find_grid81(payload)
        if not grid81:
            manifest.append(
                {
                    "kind": "puzzle_grid_skipped",
                    "reason": "no_81_cell_grid_found",
                    "source_path": str(path),
                }
            )
            continue

        record_id = str(payload.get("record_id") or path.stem)
        puzzle_id = str(
            payload.get("puzzle_id")
            or payload.get("local_puzzle_code")
            or payload.get("local_id")
            or ""
        )
        difficulty = str(payload.get("difficulty") or payload.get("puzzle_difficulty") or "")

        out_path = out_dir / f"{ordinal:02d}_{_safe_slug(record_id)}_{grid_size}x{grid_size}.png"

        if out_path.exists() and not force:
            pass
        else:
            _render_puzzle_grid(grid81, out_path, size=grid_size)

        manifest.append(
            {
                "kind": "puzzle_grid",
                "source_path": str(path),
                "output_path": str(out_path),
                "record_id": record_id,
                "puzzle_id": puzzle_id,
                "difficulty": difficulty,
                "grid_size_px": grid_size,
                "sha12": _sha12(out_path),
                "bytes": out_path.stat().st_size,
            }
        )
        count += 1

    manifest.append(
        {
            "kind": "puzzle_grid_summary",
            "book_dir": str(book_dir),
            "candidate_record_files": len(all_files),
            "rendered_grids": count,
            "requested_max_grids": max_grids,
        }
    )


def _make_dirs(out_root: Path) -> Dict[str, Path]:
    dirs = {
        "source": out_root / "00_source",
        "extracted": out_root / "01_extracted_assets",
        "reports": out_root / "_reports",
        "cover": out_root / "01_extracted_assets" / "cover",
        "page_renders": out_root / "01_extracted_assets" / "page_renders",
        "pattern_tiles": out_root / "01_extracted_assets" / "pattern_tiles",
        "puzzle_grids": out_root / "01_extracted_assets" / "puzzle_grids",
    }
    for path in dirs.values():
        _ensure_dir(path)
    return dirs


def _resolve_out_root(config: Dict[str, Any], book_id: str, cli_out_root: Optional[Path]) -> Path:
    if cli_out_root:
        return cli_out_root
    configured = config.get("output_root")
    if configured:
        return Path(str(configured))
    return DEFAULT_OUTPUT_ROOT / book_id


def run(args: argparse.Namespace) -> Dict[str, Any]:
    config_path = Path(args.config)
    config = _read_json(config_path)

    book_id = str(args.book_id or config.get("book_id") or DEFAULT_BOOK_ID)
    out_root = _resolve_out_root(config, book_id, Path(args.out_root) if args.out_root else None)
    book_dir = Path(args.book_dir) if args.book_dir else DEFAULT_BOOKS_ROOT / book_id
    patterns_dir = Path(args.patterns_dir) if args.patterns_dir else DEFAULT_PATTERNS_DIR

    publication_spec = (
        Path(args.publication_spec)
        if args.publication_spec
        else _discover_publication_spec(book_id)
    )
    interior_pdf = Path(args.interior_pdf) if args.interior_pdf else _discover_interior_pdf(book_id)
    cover_image = Path(args.cover_image) if args.cover_image else _discover_cover_image(book_id)

    dirs = _make_dirs(out_root)
    manifest: List[Dict[str, Any]] = []

    manifest.append(
        {
            "kind": "phase_a1_run",
            "book_id": book_id,
            "config_path": str(config_path),
            "output_root": str(out_root),
            "book_dir": str(book_dir),
            "patterns_dir": str(patterns_dir),
            "publication_spec": str(publication_spec) if publication_spec else None,
            "interior_pdf": str(interior_pdf) if interior_pdf else None,
            "cover_image": str(cover_image) if cover_image else None,
            "force": bool(args.force),
        }
    )

    if args.dry_run:
        report = {
            "ok": True,
            "dry_run": True,
            "book_id": book_id,
            "config_path": str(config_path),
            "output_root": str(out_root),
            "resolved_paths": {
                "book_dir": str(book_dir),
                "patterns_dir": str(patterns_dir),
                "publication_spec": str(publication_spec) if publication_spec else None,
                "interior_pdf": str(interior_pdf) if interior_pdf else None,
                "cover_image": str(cover_image) if cover_image else None,
            },
        }
        _write_json(dirs["reports"] / "phase_a1_dry_run_report.json", report)
        return report

    # Copy durable source spec/config references.
    _copy_if_present(
        config_path,
        dirs["source"] / config_path.name,
        force=args.force,
        manifest=manifest,
        role="marketing_extraction_config",
    )

    if config.get("copy_source_files", {}).get("publication_spec", True):
        _copy_if_present(
            publication_spec,
            dirs["source"] / "publication_spec.json",
            force=args.force,
            manifest=manifest,
            role="publication_spec",
        )

    # Cover.
    if cover_image and cover_image.exists():
        copied_cover = _copy_if_present(
            cover_image,
            dirs["source"] / f"cover_source{cover_image.suffix.lower()}",
            force=args.force,
            manifest=manifest,
            role="cover_source",
        )
        normalized_cover = dirs["cover"] / "cover_front_normalized.png"
        if normalized_cover.exists() and not args.force:
            pass
        else:
            _fit_cover_image(copied_cover or cover_image, normalized_cover)

        manifest.append(
            {
                "kind": "cover_normalized",
                "source_path": str(copied_cover or cover_image),
                "output_path": str(normalized_cover),
                "sha12": _sha12(normalized_cover),
                "bytes": normalized_cover.stat().st_size,
            }
        )
    else:
        manifest.append(
            {
                "kind": "cover_skipped",
                "reason": "cover_image_not_found",
                "hint": "Pass --cover-image if auto-discovery cannot find the front cover.",
            }
        )

    # PDF page renders.
    selected_pages = list(config.get("selected_pdf_pages") or [])
    if interior_pdf and interior_pdf.exists() and selected_pages:
        _render_pdf_pages(
            interior_pdf,
            selected_pages,
            dirs["page_renders"],
            dpi=int(args.page_dpi),
            force=bool(args.force),
            manifest=manifest,
        )
    else:
        manifest.append(
            {
                "kind": "pdf_page_renders_skipped",
                "reason": "interior_pdf_missing_or_no_selected_pages",
                "interior_pdf": str(interior_pdf) if interior_pdf else None,
                "selected_page_count": len(selected_pages),
                "hint": "Pass --interior-pdf if auto-discovery cannot find the book interior PDF.",
            }
        )

    # Pattern tiles.
    pattern_cfg = config.get("pattern_tiles") or {}
    if pattern_cfg.get("enabled", True):
        _render_pattern_tiles(
            patterns_dir,
            dirs["pattern_tiles"],
            tile_size=int(args.pattern_tile_size or pattern_cfg.get("tile_size_px") or 220),
            max_tiles=int(args.max_pattern_tiles or pattern_cfg.get("max_tiles") or 24),
            preferred_pattern_ids=list(pattern_cfg.get("preferred_pattern_ids") or []),
            force=bool(args.force),
            manifest=manifest,
        )

    # Puzzle grids.
    puzzle_cfg = config.get("puzzle_grids") or {}
    if puzzle_cfg.get("enabled", True):
        _render_puzzle_grids(
            book_dir,
            dirs["puzzle_grids"],
            grid_size=int(args.puzzle_grid_size or puzzle_cfg.get("grid_size_px") or 600),
            max_grids=int(args.max_puzzle_grids or puzzle_cfg.get("max_grids") or 12),
            force=bool(args.force),
            manifest=manifest,
        )

    report = {
        "ok": True,
        "phase": "A1",
        "book_id": book_id,
        "output_root": str(out_root),
        "asset_manifest_path": str(dirs["reports"] / "phase_a1_asset_manifest.json"),
        "records": manifest,
        "summary": {
            "total_records": len(manifest),
            "cover_assets": sum(1 for r in manifest if r.get("kind", "").startswith("cover")),
            "page_renders": sum(1 for r in manifest if r.get("kind") == "pdf_page_render"),
            "pattern_tiles": sum(1 for r in manifest if r.get("kind") == "pattern_tile"),
            "puzzle_grids": sum(1 for r in manifest if r.get("kind") == "puzzle_grid"),
        },
    }

    _write_json(dirs["reports"] / "phase_a1_asset_manifest.json", report)

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"[OK] Phase A1 assets written to: {out_root}")
    print(f"[OK] Manifest: {dirs['reports'] / 'phase_a1_asset_manifest.json'}")

    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract Phase A1 marketing source assets for a Sudoku book: "
            "cover image, selected PDF page renders, pattern tiles, puzzle grids, and manifest."
        )
    )

    parser.add_argument(
        "--config",
        default=str(DEFAULT_MARKETING_SPEC),
        help="Marketing extraction config JSON.",
    )
    parser.add_argument(
        "--book-id",
        default=None,
        help=f"Book id. Defaults to config book_id or {DEFAULT_BOOK_ID}.",
    )
    parser.add_argument(
        "--publication-spec",
        default=None,
        help="Optional explicit publication spec path.",
    )
    parser.add_argument(
        "--interior-pdf",
        default=None,
        help="Optional explicit interior PDF path.",
    )
    parser.add_argument(
        "--cover-image",
        default=None,
        help="Optional explicit front cover image path.",
    )
    parser.add_argument(
        "--book-dir",
        default=None,
        help="Optional explicit built book directory.",
    )
    parser.add_argument(
        "--patterns-dir",
        default=None,
        help="Optional explicit pattern catalog directory.",
    )
    parser.add_argument(
        "--out-root",
        default=None,
        help="Optional explicit output root. Defaults to config output_root.",
    )

    parser.add_argument(
        "--page-dpi",
        type=int,
        default=180,
        help="DPI for rendered PDF pages.",
    )
    parser.add_argument(
        "--pattern-tile-size",
        type=int,
        default=None,
        help="Pattern tile size in pixels. Defaults to config value.",
    )
    parser.add_argument(
        "--puzzle-grid-size",
        type=int,
        default=None,
        help="Puzzle grid render size in pixels. Defaults to config value.",
    )
    parser.add_argument(
        "--max-pattern-tiles",
        type=int,
        default=None,
        help="Maximum number of pattern tiles to render. Defaults to config value.",
    )
    parser.add_argument(
        "--max-puzzle-grids",
        type=int,
        default=None,
        help="Maximum number of puzzle grids to render. Defaults to config value.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing generated files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve paths and write a dry-run report without rendering assets.",
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        run(args)
        return 0
    except Exception as exc:
        print(f"[ERROR] Phase A1 marketing asset extraction failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())