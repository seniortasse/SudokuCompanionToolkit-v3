# tools/mine_hard_examples.py
# ------------------------------------------------------------
# Mine "hard" real examples by running inference with a checkpoint
# and collecting rows where heads (given/solution/candidates)
# make a mistake. Optional merged manifest with oversampling.
# ------------------------------------------------------------

from __future__ import annotations
import argparse, json, os
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Your model
from python.vision.models.cell_net import CellNet

# ---------- I/O helpers ----------

def ensure_abs(p: str | Path) -> str:
    return str(Path(p).expanduser().resolve())

def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                out.append(json.loads(ln))
    return out

def write_jsonl(path: str | Path, rows: List[Dict[str, Any]]):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

# ---------- Dataset / transforms ----------

class CenterSquare:
    def __call__(self, im: Image.Image) -> Image.Image:
        w, h = im.size
        if w == h: return im
        side = min(w, h)
        l = (w - side) // 2
        t = (h - side) // 2
        return im.crop((l, t, l + side, t + side))

class CenterFrac:
    def __init__(self, frac: float = 1.0):
        self.frac = float(frac)
    def __call__(self, im: Image.Image) -> Image.Image:
        if self.frac >= 0.999: return im
        w, h = im.size
        side = int(min(w, h) * self.frac)
        side = max(1, side)
        l = (w - side) // 2
        t = (h - side) // 2
        return im.crop((l, t, l + side, t + side))

class CellsDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]], img: int = 96, inner_crop: float = 1.0):
        self.rows = rows
        self.tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            CenterSquare(),
            CenterFrac(inner_crop),
            transforms.Resize((img, img)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    def __len__(self) -> int: return len(self.rows)
    def __getitem__(self, idx: int):
        r = self.rows[idx]
        path = ensure_abs(r["path"])
        im = Image.open(path).convert("L")
        x = self.tf(im)
        given = int(r.get("given_digit", 0))
        solution = int(r.get("solution_digit", 0))
        cands = r.get("candidates", []) or []
        y_c = torch.zeros(10, dtype=torch.float32)
        for d in cands:
            if 0 <= int(d) <= 9:
                y_c[int(d)] = 1.0
        return x, given, solution, y_c, idx

# ---------- Mining logic ----------

@torch.no_grad()
def mine_hard(
    manifest_path: str,
    ckpt_path: str,
    out_hard_jsonl: str,
    device: str = "cpu",
    batch: int = 128,
    img: int = 96,
    inner_crop: float = 1.0,
    cand_thr: float = 0.15,
    only_solution: bool = False,
    only_given: bool = False,
    only_candidates: bool = False,
) -> Dict[str, int]:
    rows = read_jsonl(manifest_path)
    ds = CellsDataset(rows, img=img, inner_crop=inner_crop)
    ld = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=(0 if os.name == "nt" else 2), pin_memory=True)

    dev = torch.device(device)
    model = CellNet(num_classes=10).to(dev)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    out_rows: List[Dict[str, Any]] = []

    stats = {
        "N": len(rows),
        "given_err_nonzero": 0,
        "given_fp_on_zero": 0,
        "solution_err_nonzero": 0,
        "solution_fp_on_zero": 0,
        "cand_fn": 0,
        "cand_fp": 0,
        "any_hard": 0,
    }

    # head filter function
    def keep_row(given_err_nz, given_fp_z, sol_err_nz, sol_fp_z, cand_fn, cand_fp) -> bool:
        any_given = (given_err_nz or given_fp_z)
        any_sol   = (sol_err_nz   or sol_fp_z)
        any_cand  = (cand_fn      or cand_fp)
        if only_solution:
            return any_sol
        if only_given:
            return any_given
        if only_candidates:
            return any_cand
        # default: any error from any head
        return any_given or any_sol or any_cand

    for x, y_g, y_s, y_c, idx in ld:
        x = x.to(dev, non_blocking=True)
        out = model(x)
        lg = out["logits_given"].cpu()
        ls = out["logits_solution"].cpu()
        lc = out["logits_candidates"].cpu()

        pg = lg.argmax(dim=1)                # (B,)
        ps = ls.argmax(dim=1)                # (B,)
        pc = (torch.sigmoid(lc) > cand_thr)  # (B, 10) bool

        for i in range(x.size(0)):
            row = rows[idx[i].item()]
            g = int(y_g[i].item())
            s = int(y_s[i].item())
            c_label = (y_c[i] > 0.5).nonzero(as_tuple=False).flatten().tolist()
            c_pred  = (pc[i]).nonzero(as_tuple=False).flatten().tolist()

            g_pred = int(pg[i].item())
            s_pred = int(ps[i].item())

            given_err_nz = (g != 0 and g_pred != g)
            given_fp_z   = (g == 0 and g_pred != 0)

            sol_err_nz = (s != 0 and s_pred != s)
            sol_fp_z   = (s == 0 and s_pred != 0)

            set_label = set(c_label)
            set_pred  = set(c_pred)
            cand_fn   = int(len(set_label - set_pred) > 0)   # missed true candidates
            cand_fp   = int(len(set_pred  - set_label) > 0)  # predicted extra candidates

            if keep_row(given_err_nz, given_fp_z, sol_err_nz, sol_fp_z, cand_fn, cand_fp):
                r = dict(row)
                r["_hard_info"] = {
                    "given": {"y": g, "y_pred": g_pred, "err_nonzero": int(given_err_nz), "fp_on_zero": int(given_fp_z)},
                    "solution": {"y": s, "y_pred": s_pred, "err_nonzero": int(sol_err_nz), "fp_on_zero": int(sol_fp_z)},
                    "candidates": {"y": sorted(set_label), "y_pred": sorted(set_pred), "fn": cand_fn, "fp": cand_fp},
                }
                out_rows.append(r)

            stats["given_err_nonzero"] += int(given_err_nz)
            stats["given_fp_on_zero"]  += int(given_fp_z)
            stats["solution_err_nonzero"] += int(sol_err_nz)
            stats["solution_fp_on_zero"]  += int(sol_fp_z)
            stats["cand_fn"] += int(cand_fn)
            stats["cand_fp"] += int(cand_fp)
            stats["any_hard"] += int(given_err_nz or given_fp_z or sol_err_nz or sol_fp_z or cand_fn or cand_fp)

    write_jsonl(out_hard_jsonl, out_rows)
    return stats

def merge_with_oversample(base_manifest: str, hard_manifest: str, out_merged: str, oversample: int = 3):
    base_rows = read_jsonl(base_manifest)
    hard_rows = read_jsonl(hard_manifest)
    merged: List[Dict[str, Any]] = []
    merged.extend(base_rows)
    for r in hard_rows:
        for _ in range(max(1, int(oversample))):
            merged.append(r)
    write_jsonl(out_merged, merged)

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="JSONL to scan")
    ap.add_argument("--ckpt", required=True, help="Checkpoint .pt file (model_state or raw state_dict)")
    ap.add_argument("--out-hard", default="datasets/cells/cell_interpreter/real_hard.jsonl")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--img", type=int, default=96)
    ap.add_argument("--inner-crop", type=float, default=1.0)
    ap.add_argument("--cand-thr", type=float, default=0.15)

    # Filters
    ap.add_argument("--only-solution-errors", action="store_true", dest="only_solution",
                    help="Keep rows where the solution head errs (non-zero mismatch or FP-on-zero)")
    ap.add_argument("--only-given-errors", action="store_true", dest="only_given",
                    help="Keep rows where the given head errs (non-zero mismatch or FP-on-zero)")
    ap.add_argument("--only-candidate-errors", action="store_true", dest="only_candidates",
                    help="Keep rows where candidate head errs (FN or FP)")

    # Optional merge
    ap.add_argument("--merge-into", default="", help="If set, write merged manifest (base + oversampled hard)")
    ap.add_argument("--oversample", type=int, default=3)

    args = ap.parse_args()

    stats = mine_hard(
        manifest_path=args.manifest,
        ckpt_path=args.ckpt,
        out_hard_jsonl=args.out_hard,
        device=args.device,
        batch=args.batch,
        img=args.img,
        inner_crop=args.inner_crop,
        cand_thr=args.cand_thr,
        only_solution=args.only_solution,
        only_given=args.only_given,
        only_candidates=args.only_candidates,
    )

    print("\n[mining-stats]")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"\n[done] wrote hard examples to: {args.out_hard}")

    if args.merge_into:
        merge_with_oversample(
            base_manifest=args.manifest,
            hard_manifest=args.out_hard,
            out_merged=args.merge_into,
            oversample=args.oversample,
        )
        print(f"[merge] wrote merged manifest to: {args.merge_into} (oversample={args.oversample}x)")

if __name__ == "__main__":
    main()