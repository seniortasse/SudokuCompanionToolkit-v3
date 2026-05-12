from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from PIL import Image, ImageDraw

from python.publishing.schemas.models import PatternRecord


CELL_SIZE = 48
GRID_SIZE = 9
PADDING = 16


def _mask_to_rows(mask81: str) -> List[str]:
    mask81 = str(mask81).strip()
    if len(mask81) != 81:
        raise ValueError("mask81 must be exactly 81 characters long")
    return [mask81[i:i + 9] for i in range(0, 81, 9)]


def render_pattern_preview(pattern: PatternRecord, output_path: Path) -> Path:
    rows = _mask_to_rows(pattern.mask81)

    board_px = GRID_SIZE * CELL_SIZE
    image_size = board_px + PADDING * 2

    image = Image.new("RGB", (image_size, image_size), "white")
    draw = ImageDraw.Draw(image)

    # Fill active cells.
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if rows[r][c] == "1":
                x0 = PADDING + c * CELL_SIZE
                y0 = PADDING + r * CELL_SIZE
                x1 = x0 + CELL_SIZE
                y1 = y0 + CELL_SIZE
                draw.rectangle([x0, y0, x1, y1], fill="black")

    # Grid lines.
    for i in range(GRID_SIZE + 1):
        width = 3 if i % 3 == 0 else 1
        x = PADDING + i * CELL_SIZE
        y = PADDING + i * CELL_SIZE
        draw.line([(x, PADDING), (x, PADDING + board_px)], fill="gray", width=width)
        draw.line([(PADDING, y), (PADDING + board_px, y)], fill="gray", width=width)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def render_pattern_previews(
    patterns: Iterable[PatternRecord],
    previews_dir: Path,
) -> List[Path]:
    previews_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    for pattern in patterns:
        output_path = previews_dir / f"{pattern.pattern_id}.png"
        written.append(render_pattern_preview(pattern, output_path))

    return written