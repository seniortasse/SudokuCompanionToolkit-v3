# tools/label_corners.py
# Corner labeler for Sudoku grids
# - Image and UI separated in dedicated panes (no overlay on the image)
# - 'n' (next) auto-saves if 4 points + mode are set
# - ENTER saves and auto-advances when complete
# - Autosave on quit if current is complete
# - Keeps/updates labels.jsonl (idempotent), skips already-labeled images by default
#
# Run:
#   python .\tools\label_corners.py --root .\datasets\sudoku_corners_real
#
# labels.jsonl line example:
# {"file_name": "IMG_0001.jpg", "corners": [[x,y],... TL,TR,BR,BL], "mode": "warped_inside"}

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

VALID_MODES = ["warped_inside", "straight_strict", "curved_inside", "curved_partial"]
MODE_KEYS = {
    ord("w"): "warped_inside",
    ord("s"): "straight_strict",
    ord("c"): "curved_inside",
    ord("p"): "curved_partial",
}

WINDOW_NAME = "Corner Labeler"

# ----------------------------- utils --------------------------------- #
def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def load_labels(labels_path: Path) -> Dict[str, dict]:
    """Return dict[file_name] -> record"""
    if not labels_path.exists():
        return {}
    out = {}
    with labels_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                fname = rec.get("file_name") or rec.get("file") or rec.get("image") or rec.get("path")
                if fname:
                    out[fname] = rec
            except Exception:
                pass
    return out

def write_labels(labels_path: Path, records: Dict[str, dict]):
    ensure_dir(labels_path)
    with labels_path.open("w", encoding="utf-8") as f:
        for _, rec in records.items():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def order_corners_tl_tr_br_bl(corners_np: np.ndarray) -> np.ndarray:
    c = np.asarray(corners_np, dtype=np.float32)
    idx_y = np.argsort(c[:, 1])
    top = c[idx_y[:2]]
    bot = c[idx_y[2:]]
    top = top[np.argsort(top[:, 0])]
    bot = bot[np.argsort(bot[:, 0])]
    tl, tr = top[0], top[1]
    bl, br = bot[0], bot[1]
    return np.stack([tl, tr, br, bl], axis=0)

# ----------------------------- state --------------------------------- #
class State:
    def __init__(self, root: Path):
        self.root = root
        self.img_dir = root / "images"
        self.labels_path = root / "labels.jsonl"

        assert self.img_dir.is_dir(), f"Images directory not found: {self.img_dir}"

        self.images = sorted([p.name for p in self.img_dir.iterdir()
                              if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]])
        assert len(self.images) > 0, f"No images found in {self.img_dir}"

        self.labels = load_labels(self.labels_path)
        self.index = self._first_unlabeled_index()

        # current editable fields
        self.points: List[Tuple[int, int]] = []  # raw pixel coords on original image
        self.mode: Optional[str] = None
        self.status: str = "Click 4 corners (any order). Choose mode: [S]traight, [W]arped, [C]urved-in, [P]artial."

        # rendering
        self.zoom = 1.0
        self.image_cached = None  # (img_bgr, orig_w, orig_h)
        self.saved_since_change = False

        # UI layout
        self.PAD = 12
        self.IMG_W = 900   # image pane width
        self.IMG_H = 900   # image pane height (max)
        self.SIDEBAR_W = 360
        self.FOOTER_H = 90

    # -------------- navigation -------------- #
    def _first_unlabeled_index(self) -> int:
        for i, fname in enumerate(self.images):
            if fname not in self.labels:
                return i
        return 0  # all labeled; start at first for possible edits

    def current_name(self) -> str:
        return self.images[self.index]

    def current_path(self) -> Path:
        return self.img_dir / self.current_name()

    def advance(self):
        self.index = (self.index + 1) % len(self.images)
        self.load_current_into_editor()

    def back(self):
        self.index = (self.index - 1) % len(self.images)
        self.load_current_into_editor()

    def load_current_into_editor(self):
        # load image
        fp = self.current_path()
        img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if img is None:
            self.status = f"Cannot read image: {fp}"
            self.image_cached = None
            return
        h, w = img.shape[:2]
        self.image_cached = (img, w, h)

        # restore existing labels if present
        rec = self.labels.get(self.current_name())
        if rec is not None:
            pts = rec.get("corners", None)
            if pts and len(pts) == 4:
                self.points = [(int(round(x)), int(round(y))) for (x, y) in pts]
            else:
                self.points = []
            self.mode = rec.get("mode", None)
            self.saved_since_change = True
            self.status = "Loaded existing labels (edit as needed)."
        else:
            self.points = []
            self.mode = None
            self.saved_since_change = False
            self.status = "New image. Click 4 corners + choose mode."

    # -------------- save logic -------------- #
    def can_save(self) -> bool:
        return (len(self.points) == 4) and (self.mode is not None)

    def save_current(self):
        if not self.can_save():
            self.status = "Cannot save: need 4 points + mode."
            return False

        # order corners (TL, TR, BR, BL) before saving
        arr = np.array(self.points, dtype=np.float32)
        ord4 = order_corners_tl_tr_br_bl(arr).tolist()

        rec = {
            "file_name": self.current_name(),
            "corners": ord4,
            "mode": self.mode,
        }
        self.labels[self.current_name()] = rec
        write_labels(self.labels_path, self.labels)
        self.saved_since_change = True
        self.status = f"Saved {self.current_name()}"
        return True

# ----------------------------- mouse --------------------------------- #
def on_mouse(event, x, y, flags, param):
    state: State = param
    # Map from UI coords to image coords happens in draw(); we store native px already.
    # Here, we accept clicks only inside the image pane rectangle.
    # We'll compute pane rect identical to draw() to test membership.
    PAD = state.PAD
    IMG_W = state.IMG_W
    IMG_H = state.IMG_H
    SIDEBAR_W = state.SIDEBAR_W
    FOOTER_H = state.FOOTER_H

    # full window size
    win_w = PAD + IMG_W + PAD + SIDEBAR_W + PAD
    win_h = PAD + IMG_H + PAD + FOOTER_H + PAD

    # image pane rect
    img_x0 = PAD
    img_y0 = PAD
    img_x1 = PAD + IMG_W
    img_y1 = PAD + IMG_H

    if event == cv2.EVENT_LBUTTONDOWN and state.image_cached is not None:
        if img_x0 <= x < img_x1 and img_y0 <= y < img_y1:
            # translate to image space (consider padding + scaling)
            img, iw, ih = state.image_cached
            disp = fit_to_box(img, (IMG_W, IMG_H))
            dw, dh = disp.shape[1], disp.shape[0]
            # center within pane
            offx = img_x0 + (IMG_W - dw) // 2
            offy = img_y0 + (IMG_H - dh) // 2
            if offx <= x < offx + dw and offy <= y < offy + dh:
                # scale back to original pixels
                sx = iw / dw
                sy = ih / dh
                px = int(round((x - offx) * sx))
                py = int(round((y - offy) * sy))
                if len(state.points) < 4:
                    state.points.append((px, py))
                    state.saved_since_change = False
                    state.status = f"Point {len(state.points)}/4 added."
                else:
                    state.status = "Already 4 points; press 'r' to reset or BACKSPACE to remove last."

# ----------------------------- drawing -------------------------------- #
def fit_to_box(img_bgr: np.ndarray, box_wh: Tuple[int, int]) -> np.ndarray:
    """Return resized image that fits in (W,H) keeping aspect."""
    box_w, box_h = box_wh
    h, w = img_bgr.shape[:2]
    s = min(box_w / max(1, w), box_h / max(1, h))
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    if (nw, nh) == (w, h):
        return img_bgr.copy()
    return cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)

def draw_text(panel, text, org, scale=0.7, color=(240, 240, 240), thick=2):
    cv2.putText(panel, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def draw(state: State) -> np.ndarray:
    PAD = state.PAD
    IMG_W = state.IMG_W
    IMG_H = state.IMG_H
    SIDEBAR_W = state.SIDEBAR_W
    FOOTER_H = state.FOOTER_H

    # window/canvas
    win_w = PAD + IMG_W + PAD + SIDEBAR_W + PAD
    win_h = PAD + IMG_H + PAD + FOOTER_H + PAD
    canvas = np.zeros((win_h, win_w, 3), np.uint8)

    # colors
    col_bg = (18, 18, 18)
    col_sidebar = (28, 28, 28)
    col_footer = (28, 28, 28)
    col_white = (240, 240, 240)
    col_green = (60, 200, 60)
    col_red = (60, 60, 220)
    col_yellow = (60, 220, 220)

    # fill backgrounds
    canvas[:, :] = col_bg
    # image pane area
    img_x0, img_y0 = PAD, PAD
    img_x1, img_y1 = PAD + IMG_W, PAD + IMG_H
    # sidebar area
    sb_x0, sb_y0 = img_x1 + PAD, PAD
    sb_x1, sb_y1 = sb_x0 + SIDEBAR_W, img_y1
    canvas[sb_y0:sb_y1, sb_x0:sb_x1] = col_sidebar
    # footer area
    ft_x0, ft_y0 = PAD, img_y1 + PAD
    ft_x1, ft_y1 = win_w - PAD, ft_y0 + FOOTER_H
    canvas[ft_y0:ft_y1, ft_x0:ft_x1] = col_footer

    # draw image (centered in its pane) + annotations
    if state.image_cached is not None:
        img, iw, ih = state.image_cached
        disp = fit_to_box(img, (IMG_W, IMG_H))
        dw, dh = disp.shape[1], disp.shape[0]
        offx = img_x0 + (IMG_W - dw) // 2
        offy = img_y0 + (IMG_H - dh) // 2
        canvas[offy:offy+dh, offx:offx+dw] = disp

        # draw points/lines scaled to display
        sx = dw / iw
        sy = dh / ih
        pts_disp = [(int(offx + int(round(px * sx))), int(offy + int(round(py * sy))))
                    for (px, py) in state.points]
        for i, (x, y) in enumerate(pts_disp):
            cv2.circle(canvas, (x, y), 6, col_green, -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, (x, y), 10, (0, 0, 0), 2, lineType=cv2.LINE_AA)
            draw_text(canvas, f"{i+1}", (x+8, y-8), 0.6, col_green, 2)
        if len(pts_disp) == 4:
            # draw polygon
            for a in range(4):
                p0 = pts_disp[a]
                p1 = pts_disp[(a+1) % 4]
                cv2.line(canvas, p0, p1, col_yellow, 2, cv2.LINE_AA)

    # sidebar content
    y = sb_y0 + 30
    draw_text(canvas, f"Image {state.index+1}/{len(state.images)}", (sb_x0+16, y), 0.8, col_white, 2); y += 34
    draw_text(canvas, f"File: {state.current_name()}", (sb_x0+16, y), 0.6, col_white, 2); y += 24
    draw_text(canvas, "Mode:", (sb_x0+16, y), 0.8, col_white, 2); y += 32
    mode_str = state.mode if state.mode else "(none)"
    draw_text(canvas, f"{mode_str}", (sb_x0+16, y), 0.8, col_yellow if state.mode else (120,120,120), 2); y += 36
    draw_text(canvas, f"Points: {len(state.points)}/4", (sb_x0+16, y), 0.8, col_white, 2); y += 36

    y += 8
    draw_text(canvas, "Set Mode:", (sb_x0+16, y), 0.7, col_white, 2); y += 28
    draw_text(canvas, "[S] straight_strict", (sb_x0+28, y), 0.6, col_white, 2); y += 22
    draw_text(canvas, "[W] warped_inside", (sb_x0+28, y), 0.6, col_white, 2); y += 22
    draw_text(canvas, "[C] curved_inside", (sb_x0+28, y), 0.6, col_white, 2); y += 22
    draw_text(canvas, "[P] curved_partial", (sb_x0+28, y), 0.6, col_white, 2); y += 30

    draw_text(canvas, "Edit:", (sb_x0+16, y), 0.7, col_white, 2); y += 28
    draw_text(canvas, "[Left click] add point", (sb_x0+28, y), 0.6, col_white, 2); y += 22
    draw_text(canvas, "[BACKSPACE] undo point", (sb_x0+28, y), 0.6, col_white, 2); y += 22
    draw_text(canvas, "[R] reset image", (sb_x0+28, y), 0.6, col_white, 2); y += 30

    draw_text(canvas, "Save / Nav:", (sb_x0+16, y), 0.7, col_white, 2); y += 28
    draw_text(canvas, "[ENTER] save + (auto next)", (sb_x0+28, y), 0.6, col_white, 2); y += 22
    draw_text(canvas, "[N] next (saves if complete)", (sb_x0+28, y), 0.6, col_white, 2); y += 22
    draw_text(canvas, "[B] previous image", (sb_x0+28, y), 0.6, col_white, 2); y += 22
    draw_text(canvas, "[Q] quit (autosaves if complete)", (sb_x0+28, y), 0.6, col_white, 2); y += 22

    # footer/status
    draw_text(canvas, state.status, (ft_x0+16, ft_y0+56), 0.8, col_white, 2)

    return canvas

# ----------------------------- main ---------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Dataset root containing images/ and labels.jsonl")
    args = ap.parse_args()

    root = Path(args.root)
    st = State(root)
    st.load_current_into_editor()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse, st)

    while True:
        panel = draw(st)
        cv2.imshow(WINDOW_NAME, panel)
        key = cv2.waitKey(20) & 0xFFFF

        if key == 0xFFFF:
            continue

        # mode keys
        if key in MODE_KEYS:
            st.mode = MODE_KEYS[key]
            st.saved_since_change = False
            st.status = f"Mode set: {st.mode}"
            continue

        # ENTER => save (if complete) + auto-advance
        if key in (13, 10):
            if st.can_save():
                st.save_current()
                st.advance()
            else:
                st.status = "Cannot save: need 4 points + mode."
            continue

        # Next => auto-save if complete, else just next
        if key in (ord("n"), ord("N")):
            if st.can_save():
                st.save_current()
                st.status = "Saved and next."
            else:
                st.status = "Next (not saved: need 4 points + mode)."
            st.advance()
            continue

        # Previous image
        if key in (ord("b"), ord("B")):
            st.back()
            continue

        # Reset points
        if key in (ord("r"), ord("R")):
            st.points = []
            st.saved_since_change = False
            st.status = "Points cleared. Click 4 corners."
            continue

        # Undo last point
        if key == 8:  # Backspace
            if st.points:
                st.points.pop()
                st.saved_since_change = False
                st.status = f"Removed last point. {len(st.points)}/4 remain."
            else:
                st.status = "No points to remove."
            continue

        # Quit (autosave if complete & unsaved)
        if key in (ord("q"), ord("Q"), 27):  # q or ESC
            if st.can_save() and not st.saved_since_change:
                st.save_current()
                st.status = "Autosaved current before exit."
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()