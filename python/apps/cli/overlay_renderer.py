from __future__ import annotations

from types_sudoku import Move

"""Rendering utilities to draw move highlights over the rectified board image (rows/cols/boxes/digits). Produces overlay_move_XX.jpg artifacts used by storyboard and animations."""


# overlay_renderer.py
# Render visual overlays for solver moves on a 900x900 warped board image.
import re

from PIL import Image, ImageDraw, ImageFont

CELL = 100  # 900/9
W = H = 900


def parse_cell(key):
    m = re.match(r"r(\d+)c(\d+)", key)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def cell_rect(r, c, pad=2):
    x0 = (c - 1) * CELL + pad
    y0 = (r - 1) * CELL + pad
    x1 = c * CELL - pad
    y1 = r * CELL - pad
    return (x0, y0, x1, y1)


def row_rect(r, pad=0):
    return (0 + pad, (r - 1) * CELL + pad, W - pad, r * CELL - pad)


def col_rect(c, pad=0):
    return ((c - 1) * CELL + pad, 0 + pad, c * CELL - pad, H - pad)


def box_rect(b, pad=2):
    # b = 1..9 left->right, top->bottom
    br = (b - 1) // 3
    bc = (b - 1) % 3
    x0 = bc * 3 * CELL + pad
    y0 = br * 3 * CELL + pad
    x1 = (bc + 1) * 3 * CELL - pad
    y1 = (br + 1) * 3 * CELL - pad
    return (x0, y0, x1, y1)


def load_font(size):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except:
        return ImageFont.load_default()


def draw_move(board_image_path: str, move: Move, out_path: str) -> None:
    """Render a single move overlay. Inputs: base board image path, move dict (technique, type, cell/digit/eliminate, highlights), output path. Draws colored shapes/text and saves a JPEG."""
    im = Image.open(board_image_path).convert("RGBA").resize((W, H))
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    # Highlights structure
    hl = move.get("highlights", {})

    # Row/Column/Box area highlights
    if "row" in hl and isinstance(hl["row"], str) and hl["row"].startswith("r"):
        r = int(hl["row"][1:])
        d.rectangle(row_rect(r), fill=(255, 255, 0, 64))
    if "col" in hl and isinstance(hl["col"], str) and hl["col"].startswith("c"):
        c = int(hl["col"][1:])
        d.rectangle(col_rect(c), fill=(0, 255, 255, 64))
    if "box" in hl and isinstance(hl["box"], str) and hl["box"].startswith("b"):
        b = int(hl["box"][1:])
        d.rectangle(box_rect(b), outline=(255, 0, 255, 255), width=6)

    # In-box / in-line helper cells
    for key in hl.get("in_box", []):
        rc = parse_cell(key)
        if rc:
            d.rectangle(cell_rect(*rc), fill=(255, 165, 0, 96))
    for key in hl.get("in_line", []):
        rc = parse_cell(key)
        if rc:
            d.rectangle(cell_rect(*rc), fill=(173, 216, 230, 96))

    # Main cells (targets)
    for key in hl.get("cells", []):
        rc = parse_cell(key)
        if rc:
            d.rectangle(
                cell_rect(*rc), fill=(144, 238, 144, 128), outline=(0, 128, 0, 255), width=3
            )

    # Eliminations
    if move.get("type") == "elimination":
        elim_digit = move.get("digit")
        for key in move.get("eliminate", []):
            rc = parse_cell(key)
            if not rc:
                continue
            # draw a red X and small '-d'
            x0, y0, x1, y1 = cell_rect(*rc, pad=12)
            d.line((x0, y0, x1, y1), fill=(255, 0, 0, 200), width=4)
            d.line((x0, y1, x1, y0), fill=(255, 0, 0, 200), width=4)
            cx = (x0 + x1) // 2
            cy = (y0 + y1) // 2
            f = load_font(22)
            d.text((cx, cy + 18), f"-{elim_digit}", fill=(255, 0, 0, 220), font=f, anchor="mm")

    # Placements
    if move.get("type", "placement") == "placement":
        cell_key = move.get("cell")
        rc = parse_cell(cell_key) if cell_key else None
        if rc:
            x0, y0, x1, y1 = cell_rect(*rc)
            cx = (x0 + x1) // 2
            cy = (y0 + y1) // 2
            f = load_font(64)
            d.text((cx, cy), str(move.get("digit", "")), fill=(0, 128, 0, 255), font=f, anchor="mm")

    # Title box
    title = f"{move.get('technique','?')}: "
    if move.get("type", "placement") == "placement":
        title += f"{move.get('cell','')} = {move.get('digit','')}"
    else:
        title += f"eliminate {move.get('digit','')} from {len(move.get('eliminate',[]))} cell(s)"
    ftitle = load_font(28)
    pad = 10
    tw, th = d.textbbox((0, 0), title, font=ftitle)[2:]
    box = (pad, pad, pad + tw + 20, pad + th + 20)
    d.rectangle(box, fill=(0, 0, 0, 160))
    d.text((pad + 10, pad + 10), title, fill=(255, 255, 255, 255), font=ftitle)

    out = Image.alpha_composite(im, overlay).convert("RGB")
    out.save(out_path)
    return out_path
