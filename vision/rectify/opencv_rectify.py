from __future__ import annotations

from typing import Any

"""OpenCV-based rectification: detect the Sudoku grid, perform perspective warp, optional line removal, and crop 81 cell tiles. This normalizes arbitrary phone photos to a clean board canvas."""


# opencv_rectify.py
import json
from pathlib import Path

import cv2
import numpy as np


def largest_quadrilateral(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thr = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5
    )
    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_area = None, 0
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > best_area:
                best = approx
                best_area = area
    return best


def order_pts(pts):
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def warp(img, quad, size=900):
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(order_pts(quad), dst)
    return cv2.warpPerspective(img, M, (size, size))


def remove_grid_lines(gray):
    """Optionally suppress grid lines via morphological ops and/or adaptive thresholding so digits stand out."""
    inv = 255 - gray
    horiz = cv2.morphologyEx(
        inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    )
    vert = cv2.morphologyEx(inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25)))
    lines = cv2.bitwise_or(horiz, vert)
    cleaned = 255 - cv2.subtract(inv, lines)
    return cleaned


def extract_cells(board_gray, outdir, center_margin=6):
    H, W = board_gray.shape[:2]
    step = H // 9
    out = []
    Path(outdir).mkdir(parents=True, exist_ok=True)
    for r in range(9):
        for c in range(9):
            y0 = r * step
            x0 = c * step
            y1 = y0 + step
            x1 = x0 + step
            crop = board_gray[
                y0 + center_margin : y1 - center_margin, x0 + center_margin : x1 - center_margin
            ]
            p = Path(outdir) / f"r{r+1}c{c+1}.png"
            cv2.imwrite(str(p), crop)
            out.append({"r": r + 1, "c": c + 1, "path": str(p)})
    return out


def process(image_path: str, out_dir: str, tile_size: int = 64) -> dict[str, Any]:
    """High-level rectification pipeline.

    Steps:
      1) Read `image_path`, detect outer grid, warp to a square canvas.
      2) Suppress grid lines to emphasize digits.
      3) Extract 81 cell crops into `out_dir/cells/` (size = `tile_size`).
      4) Write:
         - out_dir/board_warped.png
         - out_dir/board_clean.png
         - out_dir/cells/...
         - out_dir/cells.json  (metadata from `extract_cells`)

    Returns:
      Dict with key paths so callers can find artifacts.

    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image_path))
    assert img is not None, f"Cannot read image: {image_path}"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    quad = largest_quadrilateral(gray)
    if quad is None:
        raise RuntimeError("No grid found")

    # Warp to a canonical square (e.g., 900px)
    warped = warp(img, quad, 900)  # BGR
    wgray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # Remove/suppress grid lines (so digits pop)
    clean = remove_grid_lines(wgray)

    # Save board images
    warped_path = out / "board_warped.png"
    clean_path = out / "board_clean.png"
    cv2.imwrite(str(warped_path), warped)
    cv2.imwrite(str(clean_path), clean)

    # Extract 81 cell crops
    cells_dir = out / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    cells = extract_cells(clean, cells_dir)  # must be JSON-serializable

    # Persist cell metadata/paths
    cells_json = out / "cells.json"
    with cells_json.open("w", encoding="utf-8") as f:
        json.dump(cells, f, indent=2)

    return {
        "warped": str(warped_path),
        "clean": str(clean_path),
        "cells_dir": str(cells_dir),
        "cells_json": str(cells_json),
        "cells_count": len(cells),
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python opencv_rectify.py <image_path> <export_dir>")
        sys.exit(1)
    print(process(sys.argv[1], sys.argv[2]))
