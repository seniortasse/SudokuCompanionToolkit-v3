import json, os, glob
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

"""
Sudoku Cell Labeler
-------------------
- Open a folder with PNGs (e.g., r1c1.png ... r9c9.png).
- Labels: given_digit (0..9), solution_digit (0..9), candidates list.
- Output JSONL compatible with your trainer.

Keys (fast workflow)
- Right / n : next image
- Left  / b : previous image
- g then digit : set GIVEN to that digit (0 clears), exit mode
- s then digit : set SOLUTION to that digit (0 clears), exit mode
- digits (no mode): toggle CANDIDATE digits (0 clears all candidates)
- Esc: cancel entry mode
- Ctrl+S: save JSONL now
"""

JSONL_NAME = "cells_real_labeled.jsonl"

class Labeler:
    def __init__(self, master):
        self.master = master
        self.master.title("Sudoku Cell Labeler")
        self.canvas = tk.Canvas(master, width=256, height=256, bg="gray20", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.status = tk.StringVar()
        tk.Label(master, textvariable=self.status, anchor="w").pack(fill="x")

        self.img_paths = []
        self.idx = 0
        self.scale = 4  # 64 -> 256
        self.tkimg = None

        # path(str) -> dict(path, given_digit, solution_digit, candidates, source)
        self.meta = {}

        # current record values
        self.given = 0
        self.solution = 0
        self.candidates = set()

        # entry mode: None | "given" | "solution"
        self.entry_mode = None

        # autosave toggle
        self.autosave = False
        self.current_folder = None

        self._bind_keys()
        self._menu()

    # ------------------ UI wiring ------------------

    def _menu(self):
        menubar = tk.Menu(self.master)

        filem = tk.Menu(menubar, tearoff=0)
        filem.add_command(label="Open folder...", command=self.open_folder)
        filem.add_command(label="Save JSONL", command=self.save_jsonl, accelerator="Ctrl+S")
        filem.add_checkbutton(label="Autosave on navigate", onvalue=True, offvalue=False,
                              variable=tk.BooleanVar(value=self.autosave), command=self.toggle_autosave)
        filem.add_separator()
        filem.add_command(label="Quit", command=self.master.quit)
        menubar.add_cascade(label="File", menu=filem)

        self.master.config(menu=menubar)
        self.master.bind_all("<Control-s>", lambda e: self.save_jsonl())

    def _bind_keys(self):
        # navigation
        self.master.bind("<Right>", lambda e: self.next_img())
        self.master.bind("n",      lambda e: self.next_img())
        self.master.bind("<Left>", lambda e: self.prev_img())
        self.master.bind("b",      lambda e: self.prev_img())

        # enter modes
        self.master.bind("g", self.enter_given_mode)
        self.master.bind("s", self.enter_solution_mode)

        # optional cycling with Shift+G / Shift+S (kept for convenience)
        self.master.bind("G", lambda e: self._cycle_given(-1))
        self.master.bind("S", lambda e: self._cycle_solution(-1))

        # digits (0..9) are context-sensitive
        for d in "0123456789":
            self.master.bind(d, self.handle_digit)

        # exit mode
        self.master.bind("<Escape>", lambda e: self.exit_entry_mode())

    # ------------------ Folder & IO ------------------

    def open_folder(self):
        folder = filedialog.askdirectory(title="Pick image folder")
        if not folder:
            return
        self.current_folder = Path(folder)
        self.img_paths = sorted(glob.glob(os.path.join(folder, "*.png")))
        if not self.img_paths:
            messagebox.showerror("No PNG", "No .png files found in that folder.")
            return

        # reset index and load existing JSONL if present
        self.idx = 0
        self.meta = {}
        self._maybe_load_existing_jsonl()
        self.load_current()

    def _maybe_load_existing_jsonl(self):
        """Preload labels if JSONL exists in the folder."""
        jsonl_path = self.current_folder / JSONL_NAME
        if jsonl_path.exists():
            try:
                with jsonl_path.open("r", encoding="utf-8") as f:
                    for ln in f:
                        ln = ln.strip()
                        if not ln:
                            continue
                        rec = json.loads(ln)
                        p = rec.get("path", "")
                        if not p:
                            continue
                        # normalize to absolute path for lookup
                        abs_p = str((self.current_folder / Path(p).name).resolve())
                        # accept only images that exist in this folder
                        if abs_p in map(lambda x: str(Path(x).resolve()), self.img_paths):
                            self.meta[abs_p] = {
                                "path": abs_p.replace("\\", "/"),
                                "given_digit": int(rec.get("given_digit", 0)),
                                "solution_digit": int(rec.get("solution_digit", 0)),
                                "candidates": [int(x) for x in rec.get("candidates", [])],
                                "source": rec.get("source", "real"),
                            }
            except Exception as e:
                messagebox.showwarning("JSONL load", f"Could not load existing {JSONL_NAME}:\n{e}")

    def save_jsonl(self):
        if not self.img_paths:
            return
        self._persist_current()
        folder = Path(self.img_paths[0]).parent
        out = folder / JSONL_NAME
        try:
            with out.open("w", encoding="utf-8") as f:
                for p in self.img_paths:
                    key = str(Path(p).resolve())
                    rec = self.meta.get(key)
                    if not rec:
                        # unlabeled → write defaults so you can label progressively
                        rec = {
                            "path": key.replace("\\", "/"),
                            "given_digit": 0,
                            "solution_digit": 0,
                            "candidates": [],
                            "source": "real",
                        }
                    f.write(json.dumps(rec) + "\n")
            self.status.set(f"Saved → {out}")
        except Exception as e:
            messagebox.showerror("Save JSONL", f"Failed to save:\n{e}")

    def toggle_autosave(self):
        # reflect the menu checkbox programmatically
        self.autosave = not self.autosave
        self._refresh_status()

    # ------------------ Image/record handling ------------------

    def load_current(self):
        if not self.img_paths:
            return
        p = self.img_paths[self.idx]
        im = Image.open(p).convert("L").resize((64*self.scale, 64*self.scale), Image.NEAREST)
        self.tkimg = ImageTk.PhotoImage(im)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tkimg)

        # draw 3×3 guide for candidates
        s = 64 * self.scale
        m = int(0.18 * 64) * self.scale
        x0 = m; y0 = m; x1 = s - m; y1 = s - m
        for i in range(1, 3):
            xi = x0 + i * (x1 - x0) / 3
            yi = y0 + i * (y1 - y0) / 3
            self.canvas.create_line(xi, y0, xi, y1, fill="#66ff66", width=1)
            self.canvas.create_line(x0, yi, x1, yi, fill="#66ff66", width=1)

        # load existing labels or defaults
        key = str(Path(p).resolve())
        rec = self.meta.get(key, {"given_digit":0, "solution_digit":0, "candidates":[]})
        self.given = int(rec["given_digit"])
        self.solution = int(rec["solution_digit"])
        self.candidates = set(int(x) for x in rec["candidates"])

        # reset mode on load
        self.entry_mode = None
        self._refresh_status()

    def _persist_current(self):
        if not self.img_paths:
            return
        p = str(Path(self.img_paths[self.idx]).resolve())
        self.meta[p] = {
            "path": p.replace("\\", "/"),
            "given_digit": self.given,
            "solution_digit": self.solution,
            "candidates": sorted(self.candidates),
            "source": "real",
        }

    def next_img(self):
        if not self.img_paths:
            return
        self._persist_current()
        if self.autosave:
            self.save_jsonl()
        self.idx = min(len(self.img_paths)-1, self.idx+1)
        self.load_current()

    def prev_img(self):
        if not self.img_paths:
            return
        self._persist_current()
        if self.autosave:
            self.save_jsonl()
        self.idx = max(0, self.idx-1)
        self.load_current()

    # ------------------ Entry modes & key handling ------------------

    def enter_given_mode(self, *_):
        self.entry_mode = "given"
        self._refresh_status()

    def enter_solution_mode(self, *_):
        self.entry_mode = "solution"
        self._refresh_status()

    def exit_entry_mode(self):
        self.entry_mode = None
        self._refresh_status()

    def _cycle_given(self, step):
        self.given = (self.given + step) % 10
        self._refresh_status()

    def _cycle_solution(self, step):
        self.solution = (self.solution + step) % 10
        self._refresh_status()

    def handle_digit(self, event):
        d = int(event.char)

        if self.entry_mode == "given":
            self.given = d  # 0 clears GIVEN
            self.exit_entry_mode()
            return

        if self.entry_mode == "solution":
            self.solution = d  # 0 clears SOL
            self.exit_entry_mode()
            return

        # no entry mode → toggle candidates
        if d == 0:
            self.candidates.clear()
        else:
            if d in self.candidates:
                self.candidates.remove(d)
            else:
                self.candidates.add(d)
        self._refresh_status()

    # ------------------ Status ------------------

    def _refresh_status(self):
        name = Path(self.img_paths[self.idx]).name if self.img_paths else ""
        cand = "".join(str(d) for d in sorted(self.candidates)) or "-"
        mode = "" if not self.entry_mode else f"  [MODE: {self.entry_mode.upper()}]"
        autos = "ON" if self.autosave else "OFF"
        self.status.set(
            f"[{self.idx+1}/{len(self.img_paths)}] {name} | "
            f"GIVEN:{self.given}  SOL:{self.solution}  CAND:{cand}{mode}  | Autosave:{autos}  "
            "(Right/N next, Left/B back, g→digit, s→digit, digits toggle CAND, 0 clears, Esc cancels, Ctrl+S save)"
        )

# ------------------ main ------------------

if __name__ == "__main__":
    root = tk.Tk()
    app = Labeler(root)
    root.mainloop()