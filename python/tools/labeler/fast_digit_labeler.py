
"""
Fast digit labeler for Sudoku tiles — with auto-skip & stable naming.

Key features
- Recurses --src and finds cells across many boards (e.g., demo_export/<board>/cells/r#c#.png).
- Deterministic order: board-by-board, within each board r1c1..r9c9.
- Auto-skip: if a tile was already labeled before, it won't be shown again.
- Manual split (t/v) overrides --val-every for the current tile.
- Destination names are stable & unique: <board>_<r#c#>.png (board = top-level under --src).

Keyboard
- 0..9 : label current tile (0 = blank)
- t    : set split to train (for future labels)
- v    : set split to val   (for future labels)
- s    : skip (do not copy)
- b    : back (previous tile)
- q    : quit

Outputs
- Copies files into <out_root>/<split>/<digit>/<board>_<r#c#>.png
- Ledger at <out_root>/ledger.json to remember which files were labeled (for auto-skip).
"""

import argparse
import json
import re
import shutil
from pathlib import Path
from PIL import Image, ImageTk
import tkinter as tk

SUPP = (".png", ".jpg", ".jpeg")
TILE_RE = re.compile(r"r(\d+)c(\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)

def rel_board_name(src_root: Path, p: Path) -> str:
    """
    Board name = first path component under --src.
    Example: src/demo_export, path: demo_export/real_train_27/cells/r1c1.png -> board='real_train_27'
    """
    rel = p.relative_to(src_root)
    parts = rel.parts
    return parts[0] if len(parts) > 1 else rel.stem

def list_tiles_grouped(src_root: Path):
    """
    Return a sorted list of tiles across boards as (board, r, c, path).
    Sorted by board name (asc), then r=1..9, c=1..9.
    Non-matching filenames fall back to alphabetical order at the end of each board.
    """
    all_files = [p for p in src_root.rglob("*") if p.suffix.lower() in SUPP]
    grouped = {}
    fallback = {}
    for p in all_files:
        board = rel_board_name(src_root, p)
        m = TILE_RE.search(p.name)
        if m:
            r = int(m.group(1)); c = int(m.group(2))
            grouped.setdefault(board, []).append((r, c, p))
        else:
            fallback.setdefault(board, []).append(p)

    ordered = []
    for board in sorted(set(list(grouped.keys()) + list(fallback.keys()))):
        if board in grouped:
            grouped[board].sort(key=lambda t: (t[0], t[1]))
            ordered.extend([(board, r, c, p) for (r,c,p) in grouped[board]])
        if board in fallback:
            for p in sorted(fallback[board]):
                ordered.append((board, 999, 999, p))
    return ordered

class Ledger:
    """Simple JSON-backed ledger to remember labeled files by absolute source path."""
    def __init__(self, out_root: Path):
        self.path = out_root / "ledger.json"
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self.data = {}
        else:
            self.data = {}

    def has(self, src_abs: Path) -> bool:
        return str(src_abs.resolve()) in self.data

    def add(self, src_abs: Path, split: str, label: int, dest_rel: str):
        self.data[str(src_abs.resolve())] = {"split": split, "label": int(label), "dest": dest_rel}
        self.commit()

    def commit(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

class App:
    def __init__(self, src: Path, out_root: Path, val_every: int = 0):
        self.src = src
        self.out_root = out_root
        self.val_every = int(val_every)
        self.tiles = list_tiles_grouped(src)   # [(board, r, c, path), ...]
        self.ledger = Ledger(out_root)

        # Filter out already-labeled tiles (auto-skip)
        self.index_map = [i for i, (_, _, _, p) in enumerate(self.tiles) if not self.ledger.has(p)]
        self.idx_pos = 0  # position within index_map

        self.split = "train"  # default split
        self.manual_split_this_tile = False
        self.count_labeled = 0  # only counts newly labeled in this session

        self.root = tk.Tk()
        self.canvas = tk.Label(self.root)
        self.canvas.pack()
        self.msg = tk.Label(self.root, text="Keys: 0..9 label, s skip, b back, t=train, v=val, q quit", font=("Arial", 12))
        self.msg.pack()

        self.root.bind("<Key>", self.on_key)
        self.show()

    def current(self):
        if self.idx_pos < 0 or self.idx_pos >= len(self.index_map):
            return None
        i = self.index_map[self.idx_pos]
        return self.tiles[i]

    def _title(self, extra=""):
        n_total = len(self.index_map)
        pos = self.idx_pos + 1 if n_total else 0
        auto = f" (auto-val every {self.val_every})" if self.val_every else ""
        return f"Labeler — split={self.split.upper()}{auto} — {pos}/{n_total}{extra}"

    def show(self):
        cur = self.current()
        if cur is None:
            self.canvas.configure(text="Done! No unlabeled tiles left.")
            self.root.title("Done")
            return
        board, r, c, p = cur
        im = Image.open(p).convert("L").resize((256,256), Image.Resampling.NEAREST)
        self.tkimg = ImageTk.PhotoImage(im)
        self.canvas.configure(image=self.tkimg)
        self.root.title(self._title(f" — {board} r{r}c{c}"))

        # reset manual flag per tile
        self.manual_split_this_tile = False

    def on_key(self, ev):
        k = ev.char
        if k in "0123456789":
            lab = int(k)
            # Decide split for THIS tile
            split = self.split
            if self.val_every and not self.manual_split_this_tile:
                # Use 1-based counting so the 10th, 20th, ... goes to val
                if ((self.count_labeled + 1) % self.val_every) == 0:
                    split = "val"
            self.save(lab, split)
            self.count_labeled += 1
            self.idx_pos += 1
            self.show()
        elif k == "s":
            self.idx_pos += 1; self.show()
        elif k == "b":
            self.idx_pos = max(0, self.idx_pos - 1); self.show()
        elif k == "t":
            self.split = "train"; self.manual_split_this_tile = True; self.root.title(self._title(" — (force TRAIN)"))
        elif k == "v":
            self.split = "val";   self.manual_split_this_tile = True; self.root.title(self._title(" — (force VAL)"))
        elif k == "q":
            self.root.destroy()

    def save(self, label: int, split: str):
        cur = self.current()
        if cur is None:
            return
        board, r, c, p = cur
        # Stable destination filename: <board>_r{r}c{c}<ext>
        ext = p.suffix.lower()
        base_name = f"{board}_r{r}c{c}{ext}"
        dest_dir = self.out_root / split / str(label)
        dest_dir.mkdir(parents=True, exist_ok=True)
        out = dest_dir / base_name
        i = 1
        while out.exists():
            out = dest_dir / f"{board}_r{r}c{c}_{i}{ext}"
            i += 1
        shutil.copy2(p, out)

        # Record in ledger
        dest_rel = str(out.relative_to(self.out_root))
        self.ledger.add(p, split, label, dest_rel)

    def run(self):
        """Start the Tkinter event loop."""
        self.root.mainloop()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Folder containing many boards (recurses).")
    ap.add_argument("--out", required=True, help="Output root (will create train/val/0..9).")
    ap.add_argument("--val-every", type=int, default=0, help="Send every Nth labeled sample to val (0=off).")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    App(src, out, val_every=args.val_every).run()

if __name__ == "__main__":
    main()
