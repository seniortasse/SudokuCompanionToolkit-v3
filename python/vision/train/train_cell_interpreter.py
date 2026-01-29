"""
Cell Interpreter Trainer — Sudoku Companion
-------------------------------------------

WHY
----
We want a single CNN that, given a rectified Sudoku cell image, can interpret:
    - Book givens (printed digits)
    - Solver's main digit (big handwritten)
    - Candidate digits (small scribbles)

WHAT
-----
This script trains CellNet on JSONL-backed cell datasets with the schema:

    {
      "path": "path/to/image.png",
      "given_digit": 0..9,        # 0 = no given
      "solution_digit": 0..9,     # 0 = no solution
      "candidates": [digit,...],  # list of digits 0..9
      "source": "synth"           # optional tag
    }

Outputs:
    - Checkpoints in runs/cell_interpreter/
    - Best model: models/cell_interpreter/best_cell_net.pt

HOW (high-level)
----------------
1) Deterministic seeding.
2) JSONL → Dataset that returns:
       x: [1,H,W] normalized tensor
       y_given: int 0..9
       y_solution: int 0..9
       y_cand: FloatTensor[10] multi-hot {0,1}.
3) CellNet with three heads.
4) Loss = w_given * CE(given) + w_solution * CE(solution) + w_cand * BCE(candidates).
5) Validation metrics:
       - accuracy for given and solution heads
       - per-digit F1 for candidates
       - confusion matrices for given / solution (printed once at the end).
6) Optional TensorBoard logging (--log-dir).
"""

from __future__ import annotations
import argparse, os, json, random, sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from python.vision.models.cell_net import CellNet

# TensorBoard (optional)
try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:  # pragma: no cover
    SummaryWriter = None


# ----------------- Seed & helpers -----------------


def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_abs(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def format_bar(done: int, total: int, width: int = 30) -> str:
    """
    Simple ASCII progress bar: [#####.....] done/total
    """
    if total <= 0:
        return "[?]"
    frac = done / total
    filled = int(round(frac * width))
    filled = max(0, min(width, filled))
    bar = "#" * filled + "." * (width - filled)
    return f"[{bar}] {done}/{total}"


# ----------------- Data -----------------


class CenterSquare:
    def __call__(self, im: Image.Image) -> Image.Image:
        w, h = im.size
        if w == h:
            return im
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        return im.crop((left, top, left + side, top + side))


class CenterFrac:
    def __init__(self, frac: float = 1.0):
        self.frac = float(frac)

    def __call__(self, im: Image.Image) -> Image.Image:
        if self.frac >= 0.999:
            return im
        w, h = im.size
        side = int(min(w, h) * self.frac)
        if side <= 0:
            return im
        left = (w - side) // 2
        top = (h - side) // 2
        return im.crop((left, top, left + side, top + side))


class JsonlCellList:
    """
    Reads a JSONL manifest with one object per line:

        {
          "path": "path/to/img.png",
          "given_digit": 0..9,
          "solution_digit": 0..9,
          "candidates": [digit,...],
          "source": "synth"     # optional
        }
    """

    def __init__(self, manifest_path: str):
        self.rows = []
        mpath = ensure_abs(manifest_path)
        with open(mpath, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                obj = json.loads(ln)
                p = ensure_abs(obj["path"])
                given = int(obj.get("given_digit", 0))
                sol = int(obj.get("solution_digit", 0))
                cand = obj.get("candidates", [])
                src = obj.get("source", "")
                self.rows.append((p, given, sol, cand, src))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        return self.rows[idx]


class CellDataset(Dataset):
    """
    Wraps JsonlCellList and provides torch Tensors.

    Returns:
        x:          FloatTensor [1,H,W] normalized
        y_given:    LongTensor scalar (0..9)
        y_solution: LongTensor scalar (0..9)
        y_cand:     FloatTensor [10] multi-hot {0,1}
    """

    def __init__(
        self,
        cell_list: JsonlCellList,
        img_size: int = 64,
        inner_crop: float = 1.0,
        train: bool = True,
    ):
        self.cell_list = cell_list
        self.img_size = img_size
        self.inner_crop = inner_crop
        self.train = train

        aug: List[transforms.Compose] = []

        # Base: grayscale + center square + inner crop + resize
        common = [
            transforms.Grayscale(num_output_channels=1),
            CenterSquare(),
            CenterFrac(inner_crop),
            transforms.Resize((img_size, img_size)),
        ]

        if train:
            # Light augmentation: rotations, translations, scale jitter
            aug = [
                transforms.RandomAffine(
                    degrees=6,
                    translate=(0.08, 0.08),
                    scale=(0.9, 1.1),
                    fill=255,
                )
            ]

        self.tf = transforms.Compose(
            common
            + aug
            + [
                transforms.ToTensor(),  # [1,H,W] in [0,1]
                transforms.Normalize((0.5,), (0.5,)),  # -> [-1,1]
            ]
        )

    def __len__(self) -> int:
        return len(self.cell_list)

    def __getitem__(self, idx: int):
        path, given_digit, solution_digit, candidates, _src = self.cell_list[idx]
        im = Image.open(path).convert("L")
        x = self.tf(im)  # [1,H,W]

        # labels
        y_given = torch.tensor(given_digit, dtype=torch.long)
        y_solution = torch.tensor(solution_digit, dtype=torch.long)

        y_cand = torch.zeros(10, dtype=torch.float32)
        for d in candidates or []:
            d_int = int(d)
            if 0 <= d_int <= 9:
                y_cand[d_int] = 1.0

        return x, y_given, y_solution, y_cand


# ----------------- Metrics -----------------


@torch.no_grad()
def eval_heads(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cand_thr: float,   # threshold for candidate predictions
) -> dict:
    """
    Evaluate all three heads on the given loader.

    Returns dict with (both overall and 'non-zero' subsets):
        acc_given, acc_solution
        acc_given_non0, acc_solution_non0
        f1_candidates[10], f1_candidates_nonempty[10]
        tp, fp, fn
        tp_nonempty, fp_nonempty, fn_nonempty
        cm_given[10,10], cm_solution[10,10]
    """
    import numpy as np
    import torch

    model.eval()

    total = 0
    correct_given = 0
    correct_solution = 0

    # --- Non-zero (non-empty) tracking ---
    total_given_non0 = 0
    correct_given_non0 = 0
    total_solution_non0 = 0
    correct_solution_non0 = 0

    # For candidates: collect TP, FP, FN per digit (overall)
    tp = np.zeros(10, dtype=np.int64)
    fp = np.zeros(10, dtype=np.int64)
    fn = np.zeros(10, dtype=np.int64)

    # And for rows that actually have any candidates (non-empty rows)
    tp_ne = np.zeros(10, dtype=np.int64)
    fp_ne = np.zeros(10, dtype=np.int64)
    fn_ne = np.zeros(10, dtype=np.int64)

    # Confusion matrices for given / solution: [true, pred]
    cm_given = np.zeros((10, 10), dtype=np.int64)
    cm_solution = np.zeros((10, 10), dtype=np.int64)

    for x, y_given, y_solution, y_cand in loader:
        x = x.to(device, non_blocking=True)
        y_given = y_given.to(device, non_blocking=True)
        y_solution = y_solution.to(device, non_blocking=True)
        y_cand = y_cand.to(device, non_blocking=True)

        out = model(x)
        logits_given = out["logits_given"]
        logits_solution = out["logits_solution"]
        logits_candidates = out["logits_candidates"]

        pred_given = logits_given.argmax(1)
        pred_solution = logits_solution.argmax(1)

        # overall accuracies
        correct_given     += int((pred_given == y_given).sum().item())
        correct_solution  += int((pred_solution == y_solution).sum().item())
        bs = x.size(0)
        total += bs

        # --- update non-zero accuracies ---
        # given != 0
        mask_g_non0 = (y_given != 0)
        if mask_g_non0.any():
            total_given_non0    += int(mask_g_non0.sum().item())
            correct_given_non0  += int(((pred_given == y_given) & mask_g_non0).sum().item())
        # solution != 0
        mask_s_non0 = (y_solution != 0)
        if mask_s_non0.any():
            total_solution_non0   += int(mask_s_non0.sum().item())
            correct_solution_non0 += int(((pred_solution == y_solution) & mask_s_non0).sum().item())

        # confusion matrices
        y_g_cpu = y_given.cpu().numpy()
        p_g_cpu = pred_given.cpu().numpy()
        y_s_cpu = y_solution.cpu().numpy()
        p_s_cpu = pred_solution.cpu().numpy()
        for t, p in zip(y_g_cpu, p_g_cpu):
            if 0 <= t < 10 and 0 <= p < 10:
                cm_given[t, p] += 1
        for t, p in zip(y_s_cpu, p_s_cpu):
            if 0 <= t < 10 and 0 <= p < 10:
                cm_solution[t, p] += 1

        # candidates: sigmoid + threshold
        probs_cand = torch.sigmoid(logits_candidates)
        pred_cand = (probs_cand >= cand_thr).float()

        y_true = y_cand.cpu().numpy().astype(bool)  # [B,10]
        y_pred = pred_cand.cpu().numpy().astype(bool)

        # accumulate per-digit tp/fp/fn (overall)
        for d in range(10):
            tp[d] += int(np.logical_and(y_true[:, d],  y_pred[:, d]).sum())
            fp[d] += int(np.logical_and(~y_true[:, d], y_pred[:, d]).sum())
            fn[d] += int(np.logical_and(y_true[:, d],  ~y_pred[:, d]).sum())

        # accumulate only for rows with any true candidate (non-empty)
        row_has_any_true = y_true.any(axis=1)  # [B]
        if row_has_any_true.any():
            yt_ne = y_true[row_has_any_true]
            yp_ne = y_pred[row_has_any_true]
            for d in range(10):
                tp_ne[d] += int(np.logical_and(yt_ne[:, d],  yp_ne[:, d]).sum())
                fp_ne[d] += int(np.logical_and(~yt_ne[:, d], yp_ne[:, d]).sum())
                fn_ne[d] += int(np.logical_and(yt_ne[:, d],  ~yp_ne[:, d]).sum())

    # --- overall accuracies ---
    acc_given    = correct_given    / max(1, total)
    acc_solution = correct_solution / max(1, total)

    # --- non-zero accuracies ---
    acc_given_non0    = correct_given_non0    / max(1, total_given_non0)
    acc_solution_non0 = correct_solution_non0 / max(1, total_solution_non0)

    # --- F1 for candidates (overall + non-empty rows) ---
    def f1_from_counts(tp_v, fp_v, fn_v):
        f1 = np.zeros(10, dtype=np.float32)
        for d in range(10):
            precision = tp_v[d] / max(1, tp_v[d] + fp_v[d])
            recall    = tp_v[d] / max(1, tp_v[d] + fn_v[d])
            f1[d] = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return f1

    f1_all = f1_from_counts(tp, fp, fn)
    f1_ne  = f1_from_counts(tp_ne, fp_ne, fn_ne)

    return {
        "acc_given": acc_given,
        "acc_solution": acc_solution,
        "acc_given_non0": acc_given_non0,
        "acc_solution_non0": acc_solution_non0,
        "f1_candidates": f1_all,
        "f1_candidates_nonempty": f1_ne,
        "tp": tp, "fp": fp, "fn": fn,
        "tp_nonempty": tp_ne, "fp_nonempty": fp_ne, "fn_nonempty": fn_ne,
        "cm_given": cm_given,
        "cm_solution": cm_solution,
        "total": total,
        "total_given_non0": total_given_non0,
        "total_solution_non0": total_solution_non0,
    }



# ----------------- Training -----------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    opt: torch.optim.Optimizer,
    ce_given: nn.Module,
    ce_solution: nn.Module,
    bce_cand: nn.Module,
    w_given: float,
    w_solution: float,
    w_cand: float,
    epoch: int,
    num_epochs: int,
    use_progress_bar: bool = True,
) -> Tuple[float, float, float, float]:
    """
    Train for a single epoch.

    Returns:
        (avg_total_loss, avg_given_loss, avg_solution_loss, avg_cand_loss)
    """
    model.train()
    total_loss = 0.0
    total_given_loss = 0.0
    total_solution_loss = 0.0
    total_cand_loss = 0.0
    n = 0

    num_batches = len(loader)
    # update progress every N batches
    progress_step = max(1, num_batches // 20)

    for batch_idx, (x, y_given, y_solution, y_cand) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y_given = y_given.to(device, non_blocking=True)
        y_solution = y_solution.to(device, non_blocking=True)
        y_cand = y_cand.to(device, non_blocking=True)

        opt.zero_grad()
        out = model(x)

        logits_given = out["logits_given"]  # [B,10]
        logits_solution = out["logits_solution"]  # [B,10]
        logits_candidates = out["logits_candidates"]  # [B,10]

        loss_given = ce_given(logits_given, y_given)
        loss_solution = ce_solution(logits_solution, y_solution)
        loss_cand = bce_cand(logits_candidates, y_cand)

        loss = w_given * loss_given + w_solution * loss_solution + w_cand * loss_cand
        loss.backward()
        opt.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_given_loss += loss_given.item() * bs
        total_solution_loss += loss_solution.item() * bs
        total_cand_loss += loss_cand.item() * bs
        n += bs

        # simple progress bar on stdout
        if use_progress_bar and ((batch_idx + 1) % progress_step == 0 or (batch_idx + 1) == num_batches):
            bar = format_bar(batch_idx + 1, num_batches)
            msg = f"\r  [train] epoch {epoch}/{num_epochs} {bar}"
            sys.stdout.write(msg)
            sys.stdout.flush()

    if use_progress_bar:
        sys.stdout.write("\n")
        sys.stdout.flush()

    return (
        total_loss / max(1, n),
        total_given_loss / max(1, n),
        total_solution_loss / max(1, n),
        total_cand_loss / max(1, n),
    )


def save_ckpt(model: nn.Module, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    torch.save({"model_state": model.state_dict()}, path)
    print(f"[save] {path}")


# ----------------- Main -----------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train-manifest",
        type=str,
        required=True,
        help="JSONL path for training cells",
    )
  

    ap.add_argument(
        "--val-manifest",
        type=str,
        default="",
        help="Primary JSONL path for validation cells (used for best model selection).",
    )
    ap.add_argument(
        "--val-manifest-aux",
        type=str,
        default="",
        help="Auxiliary JSONL path for a second validation set (metrics only; does not affect best model selection).",
    )


    ap.add_argument("--img", type=int, default=64, help="Input size (H=W)")
    ap.add_argument(
        "--inner-crop",
        type=float,
        default=1.0,
        help="Center-crop fraction before resize (e.g., 0.9)",
    )
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    ap.add_argument(
        "--save-dir",
        type=str,
        default="runs/cell_interpreter",
        help="Directory for checkpoints and logs",
    )
    ap.add_argument(
        "--model-out",
        type=str,
        default="models/cell_interpreter/best_cell_net.pt",
        help="Path to save best model weights",
    )
    ap.add_argument(
        "--warm-start",
        type=str,
        default="",
        help="Path to .pt checkpoint to initialize weights",
    )
    ap.add_argument(
        "--w-given",
        type=float,
        default=1.0,
        help="Loss weight for given head",
    )
    ap.add_argument(
        "--w-solution",
        type=float,
        default=1.0,
        help="Loss weight for solution head",
    )
    ap.add_argument(
        "--w-cand",
        type=float,
        default=1.0,
        help="Loss weight for candidates head",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=(0 if os.name == "nt" else 2),
        help="DataLoader workers",
    )
    ap.add_argument(
        "--log-dir",
        type=str,
        default="",
        help="Optional TensorBoard log directory (if empty, TB logging is disabled)",
    )

    ap.add_argument("--cand-thr", type=float, default=0.50,
                help="Threshold for candidate predictions during validation.")

    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    save_dir = Path(args.save_dir)
    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)

    print("[config]", vars(args))
    print(f"[eval] candidate threshold = {args.cand_thr}")

    # TensorBoard writer (optional)
    writer: Optional[SummaryWriter]
    if args.log_dir and SummaryWriter is not None:
        writer = SummaryWriter(log_dir=args.log_dir)
        print(f"[tb] Logging enabled at: {args.log_dir}")
    elif args.log_dir and SummaryWriter is None:
        writer = None
        print("[tb] WARNING: torch.utils.tensorboard not available; logging disabled.")
    else:
        writer = None

    # Datasets
    train_cells = JsonlCellList(args.train_manifest)
    train_ds = CellDataset(
        train_cells,
        img_size=args.img,
        inner_crop=args.inner_crop,
        train=True,
    )

    train_ld = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    if args.val_manifest:
        val_cells = JsonlCellList(args.val_manifest)
        val_ds = CellDataset(
            val_cells,
            img_size=args.img,
            inner_crop=args.inner_crop,
            train=False,
        )
        val_ld = DataLoader(
            val_ds,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )
    else:
        val_ld = None


    # Auxiliary validation (optional)
    if args.val_manifest_aux:
        val_cells_aux = JsonlCellList(args.val_manifest_aux)
        val_ds_aux = CellDataset(
            val_cells_aux,
            img_size=args.img,
            inner_crop=args.inner_crop,
            train=False,
        )
        val_ld_aux = DataLoader(
            val_ds_aux,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )
    else:
        val_ld_aux = None

    # Model
    model = CellNet(num_classes=10).to(device)

    if args.warm_start:
        ckpt = torch.load(args.warm_start, map_location="cpu")
        state = ckpt.get("model_state", ckpt)
        msg = model.load_state_dict(state, strict=False)
        print(f"[warm-start] loaded from {args.warm_start}: {msg}")

    # Losses
    ce_given = nn.CrossEntropyLoss()
    ce_solution = nn.CrossEntropyLoss()
    bce_cand = nn.BCEWithLogitsLoss()

    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_score = -1.0

    for ep in range(1, args.epochs + 1):
        print(f"\n[epoch {ep}/{args.epochs}]")

        train_loss, loss_g, loss_s, loss_c = train_one_epoch(
            model,
            train_ld,
            device,
            opt,
            ce_given,
            ce_solution,
            bce_cand,
            args.w_given,
            args.w_solution,
            args.w_cand,
            epoch=ep,
            num_epochs=args.epochs,
            use_progress_bar=True,
        )
        print(
            f"  train_loss={train_loss:.4f} "
            f"(given={loss_g:.4f}, solution={loss_s:.4f}, cand={loss_c:.4f})"
        )

        if writer is not None:
            writer.add_scalar("train/loss_total", train_loss, ep)
            writer.add_scalar("train/loss_given", loss_g, ep)
            writer.add_scalar("train/loss_solution", loss_s, ep)
            writer.add_scalar("train/loss_candidates", loss_c, ep)

        if val_ld is not None:
            
            metrics = eval_heads(model, val_ld, device, cand_thr=args.cand_thr)

            acc_g         = metrics["acc_given"]
            acc_s         = metrics["acc_solution"]
            acc_g_nz      = metrics["acc_given_non0"]
            acc_s_nz      = metrics["acc_solution_non0"]

            f1_all        = metrics["f1_candidates"]
            f1_all_avg    = float(f1_all.mean())

            f1_ne         = metrics["f1_candidates_nonempty"]
            f1_ne_avg     = float(f1_ne.mean())

            

            print(
                f"  val_acc_given={acc_g:.4f} (non0={acc_g_nz:.4f})  "
                f"val_acc_solution={acc_s:.4f} (non0={acc_s_nz:.4f})  "
                f"val_f1_candidates_avg={f1_all_avg:.4f}  "
                f"val_f1_candidates_nonempty_avg={f1_ne_avg:.4f}"
            )

            print("  val_f1_candidates_per_digit:", " ".join(f"{d}:{f1_all[d]:.3f}" for d in range(10)))
            print("  val_f1_candidates_nonempty_per_digit:", " ".join(f"{d}:{f1_ne[d]:.3f}" for d in range(10)))

            # TensorBoard (if enabled)
            if writer is not None:
                writer.add_scalar("val/acc_given", acc_g, ep)
                writer.add_scalar("val/acc_solution", acc_s, ep)
                writer.add_scalar("val/acc_given_non0", acc_g_nz, ep)
                writer.add_scalar("val/acc_solution_non0", acc_s_nz, ep)
                writer.add_scalar("val/f1_candidates_avg", f1_all_avg, ep)
                writer.add_scalar("val/f1_candidates_nonempty_avg", f1_ne_avg, ep)
                for d in range(10):
                    writer.add_scalar(f"val/f1_cand_{d}", float(f1_all[d]), ep)
                    writer.add_scalar(f"val/f1_cand_nonempty_{d}", float(f1_ne[d]), ep)



            # Simple combined score to track best model
            #val_score = acc_g + acc_s + f1_c
            val_score = acc_g + acc_s + 0.5 * (acc_g_nz + acc_s_nz) + f1_ne_avg
            if val_score > best_val_score:
                best_val_score = val_score
                save_ckpt(model, save_dir, "best.pt")
                torch.save({"model_state": model.state_dict()}, model_out)
                print(f"  [best] updated best_val_score={best_val_score:.4f}")



            # --- Auxiliary validation metrics (does not affect best model) ---



            if val_ld_aux is not None:
                metrics_aux = eval_heads(model, val_ld_aux, device, cand_thr=args.cand_thr)

                acc_g_aux    = metrics_aux["acc_given"]
                acc_s_aux    = metrics_aux["acc_solution"]
                acc_g_nz_aux = metrics_aux["acc_given_non0"]
                acc_s_nz_aux = metrics_aux["acc_solution_non0"]

                f1_all_aux     = metrics_aux["f1_candidates"]
                f1_all_aux_avg = float(f1_all_aux.mean())
                f1_ne_aux      = metrics_aux["f1_candidates_nonempty"]
                f1_ne_aux_avg  = float(f1_ne_aux.mean())

                print(
                    f"  [aux] val_acc_given={acc_g_aux:.4f} (non0={acc_g_nz_aux:.4f})  "
                    f"val_acc_solution={acc_s_aux:.4f} (non0={acc_s_nz_aux:.4f})  "
                    f"val_f1_candidates_avg={f1_all_aux_avg:.4f}  "
                    f"val_f1_candidates_nonempty_avg={f1_ne_aux_avg:.4f}"
                )
                print("  [aux] val_f1_candidates_per_digit:", " ".join(f"{d}:{f1_all_aux[d]:.3f}" for d in range(10)))
                print("  [aux] val_f1_candidates_nonempty_per_digit:", " ".join(f"{d}:{f1_ne_aux[d]:.3f}" for d in range(10)))

                if writer is not None:
                    writer.add_scalar("val_aux/acc_given", acc_g_aux, ep)
                    writer.add_scalar("val_aux/acc_solution", acc_s_aux, ep)
                    writer.add_scalar("val_aux/acc_given_non0", acc_g_nz_aux, ep)
                    writer.add_scalar("val_aux/acc_solution_non0", acc_s_nz_aux, ep)
                    writer.add_scalar("val_aux/f1_candidates_avg", f1_all_aux_avg, ep)
                    writer.add_scalar("val_aux/f1_candidates_nonempty_avg", f1_ne_aux_avg, ep)
                    for d in range(10):
                        writer.add_scalar(f"val_aux/f1_cand_{d}", float(f1_all_aux[d]), ep)
                        writer.add_scalar(f"val_aux/f1_cand_nonempty_{d}", float(f1_ne_aux[d]), ep)




        else:
            # No val set: just keep last epoch
            save_ckpt(model, save_dir, f"epoch_{ep:03d}.pt")
            torch.save({"model_state": model.state_dict()}, model_out)

    if writer is not None:
        writer.close()

    print("\n[done]")
    print(f"Best model weights saved to: {model_out}")

    # Final detailed eval on validation with the *best* model (confusion matrices etc.)
    if val_ld is not None and model_out.exists():
        print("\n[final eval] loading best model and computing detailed metrics...")
        ckpt = torch.load(model_out, map_location=device)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state, strict=False)

        metrics = eval_heads(model, val_ld, device, cand_thr=args.cand_thr)

        acc_g = metrics["acc_given"]
        acc_s = metrics["acc_solution"]
        f1_c_arr = metrics["f1_candidates"]
        cm_given = metrics["cm_given"]
        cm_solution = metrics["cm_solution"]

        acc_g_nz = metrics["acc_given_non0"]
        acc_s_nz = metrics["acc_solution_non0"]
        f1_ne    = metrics["f1_candidates_nonempty"]
        f1_ne_avg = float(f1_ne.mean())

        print(f"  final_val_acc_given={acc_g:.4f}")
        print(f"  final_val_acc_solution={acc_s:.4f}")
        print(
            "  final_val_f1_candidates_per_digit: "
            + " ".join(f"{d}:{f1_c_arr[d]:.3f}" for d in range(10))
        )


        print(f"  final_val_acc_given_non0={acc_g_nz:.4f}")
        print(f"  final_val_acc_solution_non0={acc_s_nz:.4f}")
        print("  final_val_f1_candidates_nonempty_per_digit: " + " ".join(f"{d}:{f1_ne[d]:.3f}" for d in range(10)))

        # Confusion matrices (given / solution)
        def print_confusion(name: str, cm: np.ndarray):
            print(f"\n  Confusion matrix ({name}) [true x pred]:")
            header = "      " + " ".join(f"{d:4d}" for d in range(10))
            print(header)
            for t in range(10):
                row = " ".join(f"{cm[t, p]:4d}" for p in range(10))
                print(f"    {t:2d}: {row}")

        print_confusion("given", cm_given)
        print_confusion("solution", cm_solution)



        # --- Final eval on auxiliary validation (if provided) ---
        if val_ld_aux is not None:
            print("\n[final eval — aux] computing metrics on auxiliary validation set...")
            metrics_aux = eval_heads(model, val_ld_aux, device, cand_thr=args.cand_thr)

            acc_g_aux = metrics_aux["acc_given"]
            acc_s_aux = metrics_aux["acc_solution"]
            f1_c_aux_arr = metrics_aux["f1_candidates"]
            cm_given_aux = metrics_aux["cm_given"]
            cm_solution_aux = metrics_aux["cm_solution"]

            # Non-zero / non-empty additions
            acc_g_nz_aux = metrics_aux["acc_given_non0"]
            acc_s_nz_aux = metrics_aux["acc_solution_non0"]
            f1_ne_aux = metrics_aux["f1_candidates_nonempty"]
            f1_ne_aux_avg = float(f1_ne_aux.mean())

            print(f"  aux_final_val_acc_given={acc_g_aux:.4f}")
            print(f"  aux_final_val_acc_solution={acc_s_aux:.4f}")
            print(
                "  aux_final_val_f1_candidates_per_digit: "
                + " ".join(f"{d}:{f1_c_aux_arr[d]:.3f}" for d in range(10))
            )

            # New non-zero summaries
            print(f"  aux_final_val_acc_given_non0={acc_g_nz_aux:.4f}")
            print(f"  aux_final_val_acc_solution_non0={acc_s_nz_aux:.4f}")
            print("  aux_final_val_f1_candidates_nonempty_per_digit: "
                  + " ".join(f"{d}:{f1_ne_aux[d]:.3f}" for d in range(10)))
            print(f"  aux_final_val_f1_candidates_nonempty_avg={f1_ne_aux_avg:.4f}")

            print_confusion("given (aux)", cm_given_aux)
            print_confusion("solution (aux)", cm_solution_aux)



if __name__ == "__main__":
    main()