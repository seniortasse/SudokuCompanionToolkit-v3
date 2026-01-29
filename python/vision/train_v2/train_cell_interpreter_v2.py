from __future__ import annotations
import argparse, json, os, random, math, csv
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from python.vision.models.cell_net import CellNet
from .samplers import Pools, HeadAwareBatchSampler
from .losses import build_losses
from .metrics import eval_heads, scan_candidate_thresholds

# ----------------- Seed & helpers -----------------
def _parse_thr_grid(s: str) -> list[float]:
    return [float(t.strip()) for t in s.split(",") if t.strip()]

def set_seed(seed: int = 1337):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_abs(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())

def _ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)
def _to_uint8(img01: np.ndarray) -> np.ndarray:
    return (np.clip(img01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

# ----------------- Data -----------------
class CenterSquare:
    def __call__(self, im: Image.Image) -> Image.Image:
        w, h = im.size
        if w == h: return im
        side = min(w, h); l = (w - side) // 2; t = (h - side) // 2
        return im.crop((l, t, l + side, t + side))

class CenterFrac:
    def __init__(self, frac: float = 1.0): self.frac = float(frac)
    def __call__(self, im: Image.Image) -> Image.Image:
        if self.frac >= 0.999: return im
        w, h = im.size; side = int(min(w, h) * self.frac)
        if side <= 0: return im
        l = (w - side) // 2; t = (h - side) // 2
        return im.crop((l, t, l + side, t + side))

class JsonlCellList:
    def __init__(self, manifest_path: str):
        self.rows = []
        mpath = ensure_abs(manifest_path)
        with open(mpath, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln: continue
                obj = json.loads(ln)
                p = ensure_abs(obj["path"])
                given = int(obj.get("given_digit", 0))
                sol   = int(obj.get("solution_digit", 0))
                cand  = obj.get("candidates", [])
                src   = obj.get("source", "")
                self.rows.append((p, given, sol, cand, src))
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx: int): return self.rows[idx]

class CellDataset(Dataset):
    def __init__(self, cell_list: JsonlCellList, img_size: int = 64, inner_crop: float = 1.0, train: bool = True):
        self.cell_list = cell_list; self.train = train
        aug = []
        common = [
            transforms.Grayscale(num_output_channels=1),
            CenterSquare(),
            CenterFrac(inner_crop),
            transforms.Resize((img_size, img_size)),
        ]
        if train:
            aug = [
                transforms.RandomAffine(degrees=6, translate=(0.08,0.08), scale=(0.9,1.1), fill=255),
                transforms.ColorJitter(brightness=0.08, contrast=0.08),
            ]
        self.tf = transforms.Compose(common + aug + [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    def __len__(self): return len(self.cell_list)

    def __getitem__(self, idx: int):
        path, given, sol, cand, _ = self.cell_list[idx]
        im = Image.open(path).convert("L")
        x = self.tf(im)
        y_g = torch.tensor(given, dtype=torch.long)
        y_s = torch.tensor(sol, dtype=torch.long)
        y_c = torch.zeros(10, dtype=torch.float32)
        for d in (cand or []):
            d = int(d)
            if 0 <= d <= 9: y_c[d] = 1.0
        return x, y_g, y_s, y_c

# ----------------- Param freezing helpers -----------------
def _split_params_by_head(model: nn.Module):
    groups = {"given": [], "solution": [], "candidates": [], "trunk": []}
    for name, p in model.named_parameters():
        lname = name.lower()
        if "given" in lname:
            groups["given"].append((name, p))
        elif "solution" in lname:
            groups["solution"].append((name, p))
        elif ("candidate" in lname or "candidates" in lname
              or "cand_" in lname or lname.startswith("cand") or "candhead" in lname):
            groups["candidates"].append((name, p))
        else:
            groups["trunk"].append((name, p))
    return groups

def _apply_freezing(groups, train_heads: set[str], freeze_trunk: bool):
    to_optimize = []
    def toggle(params, do_train: bool):
        for _, p in params:
            p.requires_grad = bool(do_train)
            if do_train: to_optimize.append(p)

    toggle(groups["given"], "given" in train_heads)
    toggle(groups["solution"], "solution" in train_heads)
    toggle(groups["candidates"], "candidates" in train_heads)
    toggle(groups["trunk"], not freeze_trunk)

    def count(params): return sum(p.numel() for _, p in params)
    def count_trainable(params): return sum(p.numel() for _, p in params if p.requires_grad)
    print("[trainable-params]")
    for k in ["trunk", "given", "solution", "candidates"]:
        total = count(groups[k]); trainable = count_trainable(groups[k])
        print(f"  {k:<11}: {trainable:,}/{total:,} params trainable")
    return to_optimize

# ----------------- Train utils -----------------
def train_one_epoch(
    model: nn.Module, loader: DataLoader, device: torch.device, opt: torch.optim.Optimizer,
    ce_g: nn.Module, ce_s: nn.Module, bce_c: nn.Module,
    w_given: float, w_solution: float, w_cand: float, show_pbar: bool = True,
) -> Tuple[float, float, float, float]:
    model.train()
    tl = tg = ts = tc = 0.0; n = 0

    iterator = loader
    if show_pbar:
        total = None
        try: total = len(loader)
        except Exception: total = None
        iterator = tqdm(loader, total=total, desc="train", ncols=88, leave=False, dynamic_ncols=True)

    for batch in iterator:
        x, y_g, y_s, y_c = batch
        x = x.to(device, non_blocking=True)
        y_g = y_g.to(device, non_blocking=True)
        y_s = y_s.to(device, non_blocking=True)
        y_c = y_c.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        out = model(x)
        lg = out["logits_given"]; ls = out["logits_solution"]; lc = out["logits_candidates"]

        Lg = ce_g(lg, y_g); Ls = ce_s(ls, y_s); Lc = bce_c(lc, y_c)
        L  = w_given * Lg + w_solution * Ls + w_cand * Lc
        L.backward(); opt.step()

        bs = x.size(0)
        tl += L.item()*bs; tg += Lg.item()*bs; ts += Ls.item()*bs; tc += Lc.item()*bs; n += bs

        if show_pbar:
            iterator.set_postfix({
                "loss": f"{tl/max(1,n):.4f}",
                "G":    f"{tg/max(1,n):.4f}",
                "S":    f"{ts/max(1,n):.4f}",
                "C":    f"{tc/max(1,n):.4f}",
            })
    return tl/max(1,n), tg/max(1,n), ts/max(1,n), tc/max(1,n)

# --------------- Extra diagnostics: candidate behavior on empty cells ---------------
@torch.no_grad()
def candidate_empty_stats(model: nn.Module, loader: DataLoader, device: torch.device, cand_thr: float) -> dict:
    model.eval()
    num_empty = 0; total_pred_digits = 0; per_digit_pred = np.zeros(10, dtype=np.int64)
    for x, _, _, y_c in loader:
        x = x.to(device, non_blocking=True); y_c = y_c.to(device, non_blocking=True)
        probs = torch.sigmoid(model(x)["logits_candidates"])
        preds = (probs >= cand_thr)  # (B,10)
        is_empty = (y_c.sum(dim=1) == 0.0)
        if not is_empty.any(): continue
        empty_preds = preds[is_empty]
        num_empty += empty_preds.size(0)
        total_pred_digits += int(empty_preds.sum().item())
        per_digit_pred += empty_preds.sum(dim=0).cpu().numpy().astype(np.int64)
    if num_empty == 0:
        return {"num_empty": 0, "avg_pred_per_empty": 0.0, "per_digit_pred_rate": [0.0]*10}
    return {
        "num_empty": int(num_empty),
        "avg_pred_per_empty": float(total_pred_digits / float(num_empty)),
        "per_digit_pred_rate": (per_digit_pred / float(num_empty)).tolist(),
    }

# ----------------- CAND heatmap helpers (visuals) -----------------
@torch.no_grad()
def _extract_cand_maps(model, x: torch.Tensor):
    out = model(x)
    return out["logits_candidates"], out.get("cand_maps", None)

def _grid_save_gray(pil_images, cols, out_path: Path, pad=2):
    if not pil_images: return
    w, h = pil_images[0].size; rows = math.ceil(len(pil_images)/cols)
    W = cols*w + (cols+1)*pad; H = rows*h + (rows+1)*pad
    canvas = Image.new("L", (W, H), 255)
    for i, im in enumerate(pil_images):
        r = i // cols; c = i % cols
        x = pad + c*(w+pad); y = pad + r*(h+pad)
        canvas.paste(im, (x, y))
    canvas.save(out_path)

@torch.no_grad()
def _summarize_maps_batch(maps: torch.Tensor, y_c: torch.Tensor, prob_thr: float=0.5, topk: int=5):
    B, C, H, W = maps.shape
    sig  = torch.sigmoid(maps)
    flat = maps.view(B, C, -1)
    sigf = sig.view(B, C, -1)

    y   = (y_c > 0.5).bool()
    maxv = flat.max(dim=2).values
    k = min(topk, H*W)
    topk_vals = flat.topk(k, dim=2).values.mean(dim=2)
    hot = (sigf >= prob_thr).float().mean(dim=2)

    def _agg(mask: torch.Tensor):
        out = {"max": [], "mean_topk": [], "frac_hot": [], "count": int(mask.sum().item())}
        for d in range(C):
            sel = mask[:, d]
            if sel.any():
                out["max"].append(float(maxv[sel, d].mean().item()))
                out["mean_topk"].append(float(topk_vals[sel, d].mean().item()))
                out["frac_hot"].append(float(hot[sel, d].mean().item()))
            else:
                out["max"].extend([0.0]); out["mean_topk"].extend([0.0]); out["frac_hot"].extend([0.0])
        return out

    pos_stats = _agg(y); neg_stats = _agg(~y)

    pooled = sigf.amax(dim=2)  # (B, C)
    pred_mask = (pooled >= prob_thr)
    co = torch.zeros(C, C, dtype=torch.long)
    for i in range(B):
        active = torch.nonzero(pred_mask[i], as_tuple=False).view(-1)
        for a in active:
            for b in active:
                co[a, b] += 1

    return {"pos": pos_stats, "neg": neg_stats, "coact": co.tolist()}

@torch.no_grad()
def _save_cand_maps_visuals(model, loader, device, out_dir: Path, n_images: int = 16, title_prefix: str = ""):
    _ensure_dir(out_dir)
    saved = 0
    try: font = ImageFont.load_default()
    except Exception: font = None

    for (x, _, _, _) in loader:
        x = x.to(device)
        lc, maps = _extract_cand_maps(model, x)
        if maps is None:
            print("[cand-viz] Model did not return cand_maps; visuals skipped.")
            return
        probs = torch.sigmoid(maps)
        up = torch.nn.functional.interpolate(probs, size=(x.size(-2), x.size(-1)),
                                             mode="bilinear", align_corners=False)
        B = maps.size(0)
        for i in range(B):
            if saved >= n_images: return
            cell01 = (x[i].detach().cpu().squeeze(0)*0.5 + 0.5).clamp(0,1).numpy()
            base = Image.fromarray(_to_uint8(cell01), mode="L").resize((128,128), Image.BICUBIC)
            tiles = [base]
            for d in range(10):
                h = up[i, d].detach().cpu().numpy()
                hm = Image.fromarray(_to_uint8(h), mode="L").resize((128,128), Image.BICUBIC)
                if font:
                    draw = ImageDraw.Draw(hm); draw.text((4,4), str(d), fill=0, font=font)
                tiles.append(hm)
            out = out_dir / f"{title_prefix}_candmaps_{saved:04d}.png"
            _grid_save_gray(tiles, cols=4, out_path=out); saved += 1

# ----------------- NEW: Candidate confusion matrix -----------------
# ----------------- NEW: class-head (given/solution) confusion matrix -----------------
@torch.no_grad()
def cls_confusion_matrix(model: nn.Module, loader: DataLoader, device: torch.device, head: str):
    """
    head ∈ {'solution','given'}.
    Builds a 10x10 confusion (rows=true digit, cols=pred digit), *non-zero only*.
    Returns (counts[10,10], row_totals[10], row_norm[10,10]).
    """
    assert head in {"solution", "given"}
    model.eval()
    counts = np.zeros((10, 10), dtype=np.int64)
    row_tot = np.zeros(10, dtype=np.int64)

    for x, y_g, y_s, _ in loader:
        x = x.to(device, non_blocking=True)
        if head == "solution":
            logits = model(x)["logits_solution"]
            y_true = y_s
        else:
            logits = model(x)["logits_given"]
            y_true = y_g

        # non-zero only
        y_true = y_true.to(torch.long)
        mask = (y_true > 0)
        if not mask.any():  # batch may have no positives for this head
            continue

        pred = torch.argmax(logits, dim=1)
        t = y_true[mask].cpu().numpy()
        p = pred[mask].cpu().numpy()
        for tt, pp in zip(t, p):
            counts[int(tt), int(pp)] += 1
            row_tot[int(tt)] += 1

    row_norm = np.zeros_like(counts, dtype=np.float64)
    for d in range(10):
        if row_tot[d] > 0:
            row_norm[d] = counts[d] / float(row_tot[d])
    return counts, row_tot, row_norm


def _print_confusion_cls(row_norm: np.ndarray, row_tot: np.ndarray, label: str):
    print(f"  [{label}] row=TRUE digit, cols=PRED digit (non-zero only); totals in []")
    print("       " + "  ".join([f"{d:>4d}" for d in range(10)]))
    for d in range(10):
        row = " ".join(f"{row_norm[d, j]:4.2f}" for j in range(10))
        print(f"    {d:>2d}: {row}  [{int(row_tot[d])}]")


@torch.no_grad()
def cand_confusion_matrix(model: nn.Module, loader: DataLoader, device: torch.device, thr: float):
    """
    For each true digit d (label==1), we accumulate a row of predicted positives (0..9).
    Returns counts[10,10], row_totals[10], row_normalized[10,10].
    """
    model.eval()
    counts = np.zeros((10,10), dtype=np.int64)
    row_tot = np.zeros(10, dtype=np.int64)
    for x, _, _, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)  # (B,10)
        probs = torch.sigmoid(model(x)["logits_candidates"])
        preds = (probs >= thr)               # (B,10) bool

        # For each sample, for each true digit, add the full pred row
        true_idx = (y > 0.5)
        for b in range(true_idx.size(0)):
            tr = true_idx[b]
            if tr.any():
                pred_row = preds[b].to(torch.int64).cpu().numpy()  # (10,)
                for d in torch.nonzero(tr, as_tuple=False).view(-1).tolist():
                    counts[d] += pred_row
                    row_tot[d] += 1

    row_norm = np.zeros_like(counts, dtype=np.float64)
    for d in range(10):
        if row_tot[d] > 0:
            row_norm[d] = counts[d] / float(row_tot[d])
    return counts, row_tot, row_norm

def _print_confusion(row_norm: np.ndarray, row_tot: np.ndarray, label="cand@thr"):
    # compact human-readable printout
    header = "    " + " ".join([f"{d:>5d}" for d in range(10)])
    print(f"  [{label}] row=TRUE digit, cols=PRED pos rate (per-true); totals in []")
    print("      " + " 0    1    2    3    4    5    6    7    8    9")
    for d in range(10):
        row = " ".join(f"{row_norm[d, j]:5.2f}" for j in range(10))
        print(f"    {d}: {row}  [{int(row_tot[d])}]")

# ----------------- Warm-start: strict shape filter -----------------
def _warm_start_strict_shapes(model: nn.Module, ckpt_path: str):
    """
    Load only checkpoint tensors whose names AND shapes match the current model.
    Useful when heads changed (e.g., deeper candidate head) but trunk/other heads should warm-start.
    """
    print(f"[warm-start] from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state", ckpt)

    msd = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in state.items():
        if k in msd and msd[k].shape == v.shape:
            filtered[k] = v
        else:
            skipped.append(k)

    msg = model.load_state_dict(filtered, strict=False)
    print(f"[warm-start] loaded keys: {len(filtered)} | skipped (shape/name): {len(skipped)}")
    # Uncomment for verbosity:
    # for k in skipped: print("   - skip:", k)
    print(f"[warm-start] load_state_dict msg: {msg}")

# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-manifest", type=str, required=True)
    ap.add_argument("--val-manifest", type=str, required=True)
    ap.add_argument("--val-manifest-aux", type=str, default="")
    ap.add_argument("--img", type=int, default=96)
    ap.add_argument("--inner-crop", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--lr", type=float, default=7.5e-05)

    # batch recipe
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--k-g", type=int, default=32)
    ap.add_argument("--k-s", type=int, default=32)
    ap.add_argument("--k-c", type=int, default=32)
    ap.add_argument("--k-e", type=int, default=24)
    ap.add_argument("--k-m", type=int, default=8)
    ap.add_argument("--k-hard", type=int, default=0)
    ap.add_argument("--steps-per-epoch", type=int, default=0)

    # head loss weights
    ap.add_argument("--w-given", type=float, default=0.6)
    ap.add_argument("--w-solution", type=float, default=0.8)
    ap.add_argument("--w-cand", type=float, default=3.5)

    # loss shapes
    ap.add_argument("--cw0-given", type=float, default=0.25)
    ap.add_argument("--cw0-solution", type=float, default=0.25)
    ap.add_argument("--cand-pos-weight", type=float, default=4.0)
    ap.add_argument("--focal-gamma", type=float, default=2.0)

    # cand threshold
    ap.add_argument("--cand-thr", type=float, default=0.25)
    ap.add_argument("--thr-grid", type=str, default="0.05,0.10,0.15,0.20,0.25,0.30,0.35")

    # I/O
    ap.add_argument("--save-dir", type=str, default="runs/cell_interpreter_v2")
    ap.add_argument("--model-out", type=str, default="models/cell_interpreter/best_cell_net_v2.pt")
    ap.add_argument("--warm-start", type=str, default="")

    # UI / logging
    ap.add_argument("--no-pbar", action="store_true")

    # ---- CAND HEATMAP DEBUG ----
    ap.add_argument("--cand-debug", action="store_true",
                    help="Collect extra per-digit heatmap stats and save visualizations.")
    ap.add_argument("--cand-viz-per-epoch", type=int, default=16)
    ap.add_argument("--cand-viz-from", type=str, default="val", choices=["val","train"])
    ap.add_argument("--cand-peak-prob", type=float, default=0.50)
    ap.add_argument("--cand-topk", type=int, default=5)
    ap.add_argument("--cand-pool", type=str, default="max", choices=["max","lse","topk"])
    ap.add_argument("--cand-lse-alpha", type=float, default=5.0)

    # training control
    ap.add_argument("--train-heads", type=str, default="all",
                    help="Comma list from {given,solution,candidates}, or 'all' or 'none'.")
    ap.add_argument("--freeze-trunk", action="store_true")

    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    model_out = Path(args.model_out); model_out.parent.mkdir(parents=True, exist_ok=True)

    print("[config]", vars(args))

    # data
    train_cells = JsonlCellList(args.train_manifest)
    val_cells   = JsonlCellList(args.val_manifest)
    val_cells_aux = JsonlCellList(args.val_manifest_aux) if args.val_manifest_aux else None

    train_ds = CellDataset(train_cells, img_size=args.img, inner_crop=args.inner_crop, train=True)
    val_ds   = CellDataset(val_cells,   img_size=args.img, inner_crop=args.inner_crop, train=False)
    val_aux_ds = CellDataset(val_cells_aux, img_size=args.img, inner_crop=args.inner_crop, train=False) if val_cells_aux else None

    pools = Pools(train_cells.rows); print("[pools]", pools)

    steps = (args.steps_per_epoch if args.steps_per_epoch > 0 else None)
    batch_sampler = HeadAwareBatchSampler(
        pools=pools, dataset_size=len(train_ds), batch_size=args.batch,
        k_g=args.k_g, k_s=args.k_s, k_c=args.k_c, k_e=args.k_e, k_m=args.k_m,
        steps_per_epoch=steps, hard_idx=None, k_hard=args.k_hard,
    )
    train_ld = DataLoader(train_ds, batch_sampler=batch_sampler,
                          num_workers=(0 if os.name == "nt" else 2), pin_memory=True)
    val_ld   = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                          num_workers=(0 if os.name == "nt" else 2), pin_memory=True)
    val_aux_ld = (DataLoader(val_aux_ds, batch_size=args.batch, shuffle=False,
                             num_workers=(0 if os.name == "nt" else 2), pin_memory=True)
                  if val_cells_aux else None)

    # model
    model = CellNet(num_classes=10, lse_tau=0.40).to(device)
    if args.warm_start:
        _warm_start_strict_shapes(model, args.warm_start)

    # losses
    ce_g, ce_s, bce_c = build_losses(
        device=device, w0_given=args.cw0_given, w0_solution=args.cw0_solution,
        cand_pos_weight=args.cand_pos_weight, focal_gamma=args.focal_gamma,
    )

    # choose heads to train / freezing
    raw = args.train_heads.strip().lower()
    if raw in ("all", "", "default"):
        train_heads = {"given", "solution", "candidates"}
    elif raw in ("none", "off"):
        train_heads = set()
    else:
        train_heads = {h.strip() for h in raw.split(",") if h.strip() in {"given", "solution", "candidates"}}
    print(f"[train-heads] {sorted(list(train_heads))} ; freeze_trunk={args.freeze_trunk}")

    groups = _split_params_by_head(model)
    to_optimize = _apply_freezing(groups, train_heads=train_heads, freeze_trunk=args.freeze_trunk)
    if not to_optimize:
        print("[warning] No parameters selected for optimization (all frozen). Training will be a no-op.")
    opt = torch.optim.Adam(to_optimize, lr=args.lr)

    best_score = -1.0
    best_operating_thr = args.cand_thr

    # --------------------------- training loop ---------------------------
    for ep in range(1, args.epochs + 1):
        print(f"\n[epoch {ep}/{args.epochs}]")
        trL, trLg, trLs, trLc = train_one_epoch(
            model, train_ld, device, opt, ce_g, ce_s, bce_c,
            args.w_given, args.w_solution, args.w_cand,
            show_pbar=(not args.no_pbar),
        )
        print(f"  train_loss={trL:.4f} (given={trLg:.4f}, solution={trLs:.4f}, cand={trLc:.4f})")

        # primary eval on REAL (val-manifest)
        metrics = eval_heads(model, val_ld, device, cand_thr=args.cand_thr)
        acc_g, acc_s = metrics["acc_given"], metrics["acc_solution"]
        acc_g_nz, acc_s_nz = metrics["acc_given_non0"], metrics["acc_solution_non0"]
        f1_all = metrics["f1_candidates"]; f1_ne = metrics["f1_candidates_nonempty"]
        f1_all_avg = float(f1_all.mean()); f1_ne_avg = float(f1_ne.mean())

        print(f"  val_acc_given={acc_g:.4f} (non0={acc_g_nz:.4f})  "
              f"val_acc_solution={acc_s:.4f} (non0={acc_s_nz:.4f})  "
              f"val_f1_candidates_avg={f1_all_avg:.4f}  "
              f"val_f1_candidates_nonempty_avg={f1_ne_avg:.4f}")

        # extra diagnostics: candidate behavior on empty cells at current cand_thr
        empty_stats = candidate_empty_stats(model, val_ld, device, cand_thr=args.cand_thr)
        print(f"  [empty-cells] num_empty={empty_stats['num_empty']}  "
              f"avg_pred_per_empty={empty_stats['avg_pred_per_empty']:.3f}")
        print("                per-digit avg preds on empty (0-9): "
              + ", ".join(f"{v:.3f}" for v in empty_stats["per_digit_pred_rate"]))

        # threshold scan on REAL (for candidates)
        thr_list = _parse_thr_grid(args.thr_grid)
        best_thr, scan_scores = scan_candidate_thresholds(model, val_ld, device, thr_list)
        print("  [thr-scan] cand F1_nonempty by thr:", " ".join(f"{t:.2f}:{scan_scores[t]:.3f}" for t in thr_list))
        print(f"  [thr-scan] best_thr={best_thr:.2f}")

        # optional aux eval (SYNTH, etc.)
        aux_acc_s_nz = 0.0
        aux_f1_ne_avg = 0.0
        if val_aux_ld is not None:
            m_aux = eval_heads(model, val_aux_ld, device, cand_thr=best_thr)
            aux_acc_s_nz = m_aux["acc_solution_non0"]
            aux_f1_ne_avg = float(m_aux["f1_candidates_nonempty"].mean())
            print(f"  [aux] val_acc_given={m_aux['acc_given']:.4f} (non0={m_aux['acc_given_non0']:.4f})  "
                  f"val_acc_solution={m_aux['acc_solution']:.4f} (non0={m_aux['acc_solution_non0']:.4f})  "
                  f"val_f1_candidates_nonempty_avg={aux_f1_ne_avg:.4f}")

        # ---------------- CAND HEATMAP DEBUG (per-epoch) ----------------
        if args.cand_debug:
            debug_dir = save_dir / "cand_debug"
            viz_dir   = save_dir / "cand_viz"
            _ensure_dir(debug_dir); _ensure_dir(viz_dir)

            # (A) Visual tiles
            if args.cand_viz_per_epoch > 0:
                viz_ld = val_ld if args.cand_viz_from == "val" else train_ld
                _save_cand_maps_visuals(model, viz_ld, device, viz_dir, args.cand_viz_per_epoch, f"ep{ep:03d}")

            # (B) Aggregate spatial stats
            model.eval()
            agg_pos_max = np.zeros(10); agg_neg_max = np.zeros(10)
            agg_pos_tk  = np.zeros(10); agg_neg_tk  = np.zeros(10)
            agg_pos_hot = np.zeros(10); agg_neg_hot = np.zeros(10)
            cnt_pos = np.zeros(10, dtype=np.int64); cnt_neg = np.zeros(10, dtype=np.int64)
            coact_tot = np.zeros((10,10), dtype=np.int64); n_batches = 0

            cand_maps_available = True
            for (xv, _, _, ycv) in val_ld:
                xv = xv.to(device); ycv = ycv.to(device)
                lc, maps = _extract_cand_maps(model, xv)
                if maps is None:
                    cand_maps_available = False; break
                stats = _summarize_maps_batch(maps, ycv, prob_thr=args.cand_peak_prob, topk=args.cand_topk)
                pos, neg = stats["pos"], stats["neg"]
                coact_tot += np.array(stats["coact"], dtype=np.int64)

                agg_pos_max += np.array(pos["max"]); agg_neg_max += np.array(neg["max"])
                agg_pos_tk  += np.array(pos["mean_topk"]); agg_neg_tk  += np.array(neg["mean_topk"])
                agg_pos_hot += np.array(pos["frac_hot"]);  agg_neg_hot += np.array(neg["frac_hot"])
                cnt_pos     += pos["count"];              cnt_neg     += neg["count"]
                n_batches   += 1

            if cand_maps_available and n_batches > 0:
                debug_payload = {
                    "epoch": ep,
                    "cand_peak_prob": args.cand_peak_prob,
                    "topk": args.cand_topk,
                    "val": {
                        "pos": {"count": int(cnt_pos.sum()),
                                "max_avg": (agg_pos_max / n_batches).tolist(),
                                "mean_topk_avg": (agg_pos_tk / n_batches).tolist(),
                                "frac_hot_avg": (agg_pos_hot / n_batches).tolist()},
                        "neg": {"count": int(cnt_neg.sum()),
                                "max_avg": (agg_neg_max / n_batches).tolist(),
                                "mean_topk_avg": (agg_neg_tk / n_batches).tolist(),
                                "frac_hot_avg": (agg_neg_hot / n_batches).tolist()},
                        "coactivation_counts": coact_tot.tolist(),
                    },
                    "thr_scan": { "best_thr": best_thr, "grid": args.thr_grid }
                }
                (debug_dir / f"cand_debug_ep{ep:03d}.json").write_text(json.dumps(debug_payload, indent=2))
                print("  [cand-debug] wrote", (debug_dir / f"cand_debug_ep{ep:03d}.json").name)
            elif not cand_maps_available:
                print("  [cand-debug] cand_maps not exposed by model; debug stats/visuals skipped.")

        # ---------------- Candidate Head Confusion Matrix (REAL) ----------------
        cm_counts, cm_row_tot, cm_row_norm = cand_confusion_matrix(model, val_ld, device, thr=best_thr)
        _print_confusion(cm_row_norm, cm_row_tot, label=f"cand@thr={best_thr:.2f}")
        # save CSV + JSON for later analysis
        cm_dir = save_dir / "cand_confusion"; _ensure_dir(cm_dir)
        np.savetxt(cm_dir / f"confusion_ep{ep:03d}_counts.csv", cm_counts, fmt="%d", delimiter=",")
        np.savetxt(cm_dir / f"confusion_ep{ep:03d}_rownorm.csv", cm_row_norm, fmt="%.6f", delimiter=",")
        (cm_dir / f"confusion_ep{ep:03d}.json").write_text(json.dumps({
            "epoch": ep, "thr": best_thr,
            "counts": cm_counts.tolist(),
            "row_totals": cm_row_tot.tolist(),
            "row_normalized": cm_row_norm.tolist(),
        }, indent=2), encoding="utf-8")

        # ---------------- Candidate Head Confusion Matrix (AUX, if any) ----------------
        if val_aux_ld is not None:
            cm_counts_a, cm_row_tot_a, cm_row_norm_a = cand_confusion_matrix(model, val_aux_ld, device, thr=best_thr)
            _print_confusion(cm_row_norm_a, cm_row_tot_a, label=f"cand@thr={best_thr:.2f} [AUX]")
            cm_aux_dir = save_dir / "cand_confusion_aux"; _ensure_dir(cm_aux_dir)
            np.savetxt(cm_aux_dir / f"confusion_ep{ep:03d}_counts.csv", cm_counts_a, fmt="%d", delimiter=",")
            np.savetxt(cm_aux_dir / f"confusion_ep{ep:03d}_rownorm.csv", cm_row_norm_a, fmt="%.6f", delimiter=",")
            (cm_aux_dir / f"confusion_ep{ep:03d}.json").write_text(json.dumps({
                "epoch": ep, "thr": best_thr,
                "counts": cm_counts_a.tolist(),
                "row_totals": cm_row_tot_a.tolist(),
                "row_normalized": cm_row_norm_a.tolist(),
            }, indent=2), encoding="utf-8")

        # ---------------- Solution Head Confusion (REAL + AUX) ----------------
        solC, solRT, solRN = cls_confusion_matrix(model, val_ld, device, head="solution")
        _print_confusion_cls(solRN, solRT, label="solution (REAL)")
        sol_dir = save_dir / "solution_confusion"; _ensure_dir(sol_dir)
        np.savetxt(sol_dir / f"confusion_ep{ep:03d}_counts.csv", solC, fmt="%d", delimiter=",")
        np.savetxt(sol_dir / f"confusion_ep{ep:03d}_rownorm.csv", solRN, fmt="%.6f", delimiter=",")
        (sol_dir / f"confusion_ep{ep:03d}.json").write_text(json.dumps({
            "epoch": ep,
            "counts": solC.tolist(),
            "row_totals": solRT.tolist(),
            "row_normalized": solRN.tolist(),
        }, indent=2), encoding="utf-8")

        if val_aux_ld is not None:
            solCa, solRTa, solRNa = cls_confusion_matrix(model, val_aux_ld, device, head="solution")
            _print_confusion_cls(solRNa, solRTa, label="solution (AUX)")
            sol_aux_dir = save_dir / "solution_confusion_aux"; _ensure_dir(sol_aux_dir)
            np.savetxt(sol_aux_dir / f"confusion_ep{ep:03d}_counts.csv", solCa, fmt="%d", delimiter=",")
            np.savetxt(sol_aux_dir / f"confusion_ep{ep:03d}_rownorm.csv", solRNa, fmt="%.6f", delimiter=",")
            (sol_aux_dir / f"confusion_ep{ep:03d}.json").write_text(json.dumps({
                "epoch": ep,
                "counts": solCa.tolist(),
                "row_totals": solRTa.tolist(),
                "row_normalized": solRNa.tolist(),
            }, indent=2), encoding="utf-8")

        # ---------------- Given Head Confusion (REAL + AUX) ----------------
        givC, givRT, givRN = cls_confusion_matrix(model, val_ld, device, head="given")
        _print_confusion_cls(givRN, givRT, label="given (REAL)")
        giv_dir = save_dir / "given_confusion"; _ensure_dir(giv_dir)
        np.savetxt(giv_dir / f"confusion_ep{ep:03d}_counts.csv", givC, fmt="%d", delimiter=",")
        np.savetxt(giv_dir / f"confusion_ep{ep:03d}_rownorm.csv", givRN, fmt="%.6f", delimiter=",")
        (giv_dir / f"confusion_ep{ep:03d}.json").write_text(json.dumps({
            "epoch": ep,
            "counts": givC.tolist(),
            "row_totals": givRT.tolist(),
            "row_normalized": givRN.tolist(),
        }, indent=2), encoding="utf-8")

        if val_aux_ld is not None:
            givCa, givRTa, givRNa = cls_confusion_matrix(model, val_aux_ld, device, head="given")
            _print_confusion_cls(givRNa, givRTa, label="given (AUX)")
            giv_aux_dir = save_dir / "given_confusion_aux"; _ensure_dir(giv_aux_dir)
            np.savetxt(giv_aux_dir / f"confusion_ep{ep:03d}_counts.csv", givCa, fmt="%d", delimiter=",")
            np.savetxt(giv_aux_dir / f"confusion_ep{ep:03d}_rownorm.csv", givRNa, fmt="%.6f", delimiter=",")
            (giv_aux_dir / f"confusion_ep{ep:03d}.json").write_text(json.dumps({
                "epoch": ep,
                "counts": givCa.tolist(),
                "row_totals": givRTa.tolist(),
                "row_normalized": givRNa.tolist(),
            }, indent=2), encoding="utf-8")

        # ---------------- Model selection score ----------------
        # Core REAL signal: we care most about nonzero solution + candidate quality on REAL.
        core_real = (
            acc_s_nz + f1_ne_avg           # dominant: nonzero solution & cand@nonempty (REAL)
            + 0.5 * (acc_s + f1_all_avg)   # softer: overall solution + cand on REAL
            + 0.25 * acc_g_nz              # sanity: given nonzero accuracy
        )

        # AUX (SYNTH) is only a soft nudge, never decisive alone
        core_aux = 0.0
        if val_aux_ld is not None:
            core_aux = 0.3 * (aux_acc_s_nz + aux_f1_ne_avg)

        # Penalty if candidates are too chatty on empty REAL cells
        avg_pred_per_empty = float(empty_stats["avg_pred_per_empty"])
        penalty_empty = 0.1 * min(avg_pred_per_empty, 0.5)

        score = core_real + core_aux - penalty_empty

        print(
            f"  [score] real_core={core_real:.4f}  aux_core={core_aux:.4f}  "
            f"penalty_empty={penalty_empty:.4f}  total={score:.4f}"
        )

        if score > best_score:
            best_score = score

            # Recompute thr scan for the checkpoint we’re actually saving,
            # and store that thr as the "best" for these weights.
            thr_list = _parse_thr_grid(args.thr_grid)
            best_thr_for_best, best_scan_scores = scan_candidate_thresholds(model, val_ld, device, thr_list)
            best_operating_thr = best_thr_for_best

            torch.save({"model_state": model.state_dict()}, model_out)
            best_pt_path = (save_dir / "best.pt"); best_pt_path.write_bytes(model_out.read_bytes())

            sidecar_txt  = save_dir / "best_operating_thr.txt"
            sidecar_json = save_dir / "best_operating_thr.json"
            sidecar_txt.write_text(f"{best_thr_for_best:.6f}\n", encoding="utf-8")
            sidecar_json.write_text(json.dumps({"best_thr": best_thr_for_best, "scores": best_scan_scores}, indent=2),
                                    encoding="utf-8")

            model_thr_txt  = Path(str(model_out) + ".thr.txt")
            model_thr_json = Path(str(model_out) + ".thr.json")
            model_thr_txt.write_text(f"{best_thr_for_best:.6f}\n", encoding="utf-8")
            model_thr_json.write_text(json.dumps({"best_thr": best_thr_for_best, "scores": best_scan_scores}, indent=2),
                                      encoding="utf-8")

            print(f"  [best] updated score={best_score:.4f}  (saved to {model_out})")
            print(f"  [best] sidecar best_operating_thr={best_thr_for_best:.2f}  "
                  f"(written to {sidecar_txt.name}, {model_thr_txt.name})")

        # per-epoch metrics json (compact)
        payload = {
            "epoch": ep,
            "train": {"loss": trL, "given": trLg, "solution": trLs, "cand": trLc},
            "val": {
                "acc_given": acc_g,
                "acc_solution": acc_s,
                "acc_given_non0": acc_g_nz,
                "acc_solution_non0": acc_s_nz,
                "f1_cand_avg": f1_all_avg,
                "f1_cand_nonempty_avg": f1_ne_avg,
                "best_thr_scan": {"best_thr": best_thr, "scores": scan_scores},
                "score": score,
            },
            "val_empty": empty_stats,
        }
        (save_dir / f"metrics_ep{ep:03d}.json").write_text(json.dumps(payload, indent=2))

    print("\n[done]")
    print(f"Best model weights saved to: {model_out}")
    print(f"Recommended cand threshold (real-val scan): {best_operating_thr:.2f}")

if __name__ == "__main__":
    main()