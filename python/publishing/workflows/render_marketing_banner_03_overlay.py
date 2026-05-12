from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_CONFIG = Path(
    "datasets/sudoku_books/classic9/marketing_specs/"
    "BK-CL9-DW-B01.marketing.phase_c2_banner_03_overlay.json"
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


def _hex_to_rgba(value: str, alpha: int = 255) -> Tuple[int, int, int, int]:
    value = str(value or "#FFFFFF").strip().lstrip("#")
    if len(value) != 6:
        return (255, 255, 255, alpha)
    return (
        int(value[0:2], 16),
        int(value[2:4], 16),
        int(value[4:6], 16),
        alpha,
    )


def _require_pillow() -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont  # noqa: F401
    except Exception as exc:
        raise RuntimeError("Pillow is required. Install with: python -m pip install pillow") from exc


def _font_from_candidates(candidates: Sequence[str], size: int):
    from PIL import ImageFont

    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _font(config: Dict[str, Any], size: int, *, bold: bool, role: Optional[str] = None):
    font_candidates = config.get("font_candidates") or {}
    candidates: List[str] = []
    if role:
        candidates.extend(font_candidates.get(role) or [])
    candidates.extend(font_candidates.get("bold" if bold else "regular") or [])
    candidates.extend([
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ])
    return _font_from_candidates(candidates, size)


def _load_rgba(path: Path):
    from PIL import Image

    if not path.exists():
        raise FileNotFoundError(str(path))
    return Image.open(path).convert("RGBA")


def _fit_cover(src: Any, size: Tuple[int, int]):
    from PIL import Image

    target_w, target_h = size
    src_w, src_h = src.size
    scale = max(target_w / src_w, target_h / src_h)
    resized = src.resize((int(src_w * scale), int(src_h * scale)), Image.LANCZOS)
    x = (resized.width - target_w) // 2
    y = (resized.height - target_h) // 2
    return resized.crop((x, y, x + target_w, y + target_h))


def _cfg_first(cfg: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in cfg and cfg[key] is not None:
            return cfg[key]
    return default


def _resolve_box(cfg: Dict[str, Any], *, fallback_h: int = 24) -> List[int]:
    if "box" in cfg and cfg["box"] is not None:
        box = cfg["box"]
        return [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
    return [
        int(cfg.get("x", 0)),
        int(cfg.get("y", 0)),
        int(cfg.get("w", 0)),
        int(cfg.get("h", fallback_h)),
    ]


def _text_size(draw: Any, text: str, font: Any) -> Tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _fit_font_to_width(
    draw: Any,
    lines: Sequence[str],
    *,
    config: Dict[str, Any],
    start_size: int,
    max_width: int,
    bold: bool,
    min_size: int = 10,
    role: Optional[str] = None,
) -> Any:
    for size in range(int(start_size), int(min_size) - 1, -1):
        font = _font(config, size, bold=bold, role=role)
        if all(_text_size(draw, line, font)[0] <= max_width for line in lines):
            return font
    return _font(config, int(min_size), bold=bold, role=role)


def _wrap_text(draw: Any, text: str, font: Any, max_width: int) -> List[str]:
    out: List[str] = []
    for raw_line in str(text).split("\n"):
        words = raw_line.split()
        if not words:
            out.append("")
            continue
        current = ""
        for word in words:
            trial = word if not current else current + " " + word
            if _text_size(draw, trial, font)[0] <= max_width:
                current = trial
            else:
                if current:
                    out.append(current)
                current = word
        if current:
            out.append(current)
    return out


def _draw_text_with_shadow(
    draw: Any,
    xy: Tuple[int, int],
    text: str,
    *,
    font: Any,
    fill: Tuple[int, int, int, int],
    shadow: bool,
    shadow_fill: Tuple[int, int, int, int] = (0, 18, 45, 145),
    shadow_offset: Tuple[int, int] = (2, 2),
) -> None:
    x, y = xy
    if shadow:
        draw.text((x + shadow_offset[0], y + shadow_offset[1]), text, font=font, fill=shadow_fill)
    draw.text((x, y), text, font=font, fill=fill)


def _draw_text_layer(draw: Any, config: Dict[str, Any], layer: Dict[str, Any], text: str) -> Dict[str, Any]:
    box = _resolve_box(layer, fallback_h=int(layer.get("font_size", 18)) + 8)
    x, y, w, h = box
    pad_l = int(layer.get("padding_left", 0))
    pad_r = int(layer.get("padding_right", 0))
    pad_t = int(layer.get("padding_top", 0))
    pad_b = int(layer.get("padding_bottom", 0))
    inner_w = max(1, w - pad_l - pad_r)
    inner_h = max(1, h - pad_t - pad_b)

    start_size = int(layer.get("font_size", 18))
    min_size = int(layer.get("min_font_size", 10))
    bold = bool(layer.get("bold", True))
    role = str(layer.get("font_role") or "bold")
    line_gap = int(layer.get("line_gap", 2))
    align = str(layer.get("align", "center")).lower()
    valign = str(layer.get("valign", "center")).lower()
    shadow = bool(layer.get("shadow", False))

    raw_lines = str(text).split("\n")
    font = _fit_font_to_width(
        draw,
        raw_lines,
        config=config,
        start_size=start_size,
        max_width=inner_w,
        bold=bold,
        min_size=min_size,
        role=role,
    )
    if bool(layer.get("wrap", False)):
        lines = _wrap_text(draw, text, font, inner_w)
    else:
        lines = raw_lines

    line_metrics = []
    total_h = 0
    for line in lines:
        tw, th = _text_size(draw, line, font)
        line_metrics.append((line, tw, th))
        total_h += th
    total_h += line_gap * max(0, len(lines) - 1)

    if valign == "top":
        yy = y + pad_t
    elif valign == "bottom":
        yy = y + pad_t + inner_h - total_h
    else:
        yy = y + pad_t + (inner_h - total_h) // 2

    fill = _hex_to_rgba(layer.get("color", "#FFFFFF"), int(layer.get("alpha", 255)))
    for line, tw, th in line_metrics:
        if align == "left":
            xx = x + pad_l
        elif align == "right":
            xx = x + pad_l + inner_w - tw
        else:
            xx = x + pad_l + (inner_w - tw) // 2
        _draw_text_with_shadow(draw, (xx, yy), line, font=font, fill=fill, shadow=shadow)
        yy += th + line_gap

    return {"key": layer.get("key"), "box": box, "font_size_used": getattr(font, "size", None)}


def _resolve_layer_text(config: Dict[str, Any], layer: Dict[str, Any]) -> str:
    copy = config.get("copy") or {}
    if "text" in layer:
        return str(layer["text"])
    key = str(layer.get("copy_key") or "")
    value = copy
    for part in key.split("."):
        if not part:
            continue
        if isinstance(value, dict):
            value = value.get(part, "")
        else:
            value = ""
    return str(value)


def _resolve_required_files(config: Dict[str, Any]) -> List[Path]:
    assets_dir = Path(config["assets_dir"])
    return [assets_dir / config["background"]]


def render_banner(config: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    _require_pillow()
    from PIL import ImageDraw

    assets_dir = Path(config["assets_dir"])
    export = config["exports"]["main"]
    width = int(export["width_px"])
    height = int(export["height_px"])

    bg_path = assets_dir / config["background"]
    img = _fit_cover(_load_rgba(bg_path), (width, height))
    draw = ImageDraw.Draw(img)

    drawn_layers = []
    for layer in config.get("text_layers", []):
        text = _resolve_layer_text(config, layer)
        if text:
            drawn_layers.append(_draw_text_layer(draw, config, layer, text))

    return img.convert("RGB"), {"background": str(bg_path), "text_layers": drawn_layers}


def run(args: argparse.Namespace) -> Dict[str, Any]:
    config_path = Path(args.config)
    config = _read_json(config_path)

    out_dir = Path(args.out_dir or config["output_dir"])
    reports_dir = out_dir / "_reports"
    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    _ensure_dir(out_dir)
    _ensure_dir(reports_dir)

    if args.dry_run:
        required = _resolve_required_files(config)
        missing = [str(p) for p in required if not p.exists()]
        report = {
            "ok": not missing,
            "dry_run": True,
            "banner_id": config["banner_id"],
            "config_path": str(config_path),
            "output_dir": str(out_dir),
            "expected_output": str(out_dir / config["exports"]["main"]["filename"]),
            "required_files_checked": [str(p) for p in required],
            "missing_files": missing,
        }
        _write_json(reports_dir / "phase_c2_banner_03_overlay_dry_run_report.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report

    img, render_report = render_banner(config)
    out_path = out_dir / config["exports"]["main"]["filename"]
    img.save(out_path)
    shutil.copy2(config_path, reports_dir / config_path.name)

    report = {
        "ok": True,
        "phase": config.get("phase", "C2"),
        "banner_id": config["banner_id"],
        "banner_type": config["banner_type"],
        "book_id": config["book_id"],
        "campaign_id": config["campaign_id"],
        "config_path": str(config_path),
        "output_dir": str(out_dir),
        "outputs": {
            "main": {
                "path": str(out_path),
                "width_px": img.width,
                "height_px": img.height,
                "sha12": _sha12(out_path),
            }
        },
        "render": render_report,
    }
    _write_json(reports_dir / "phase_c2_banner_03_overlay_report.json", report)
    print(json.dumps(report["outputs"], ensure_ascii=False, indent=2))
    print(f"[OK] Phase C2 Banner 3 overlay written to: {out_path}")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render Banner 3 text overlay onto a locked background.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Phase C2 Banner 3 overlay config JSON.")
    parser.add_argument("--out-dir", default=None, help="Optional output directory. Defaults to config output_dir.")
    parser.add_argument("--clean", action="store_true", help="Delete existing output directory before rendering.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve paths without rendering.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
        return 0
    except Exception as exc:
        print(f"[ERROR] Banner 3 overlay render failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
