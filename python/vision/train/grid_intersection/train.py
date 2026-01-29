import os, time, json, argparse, random
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F  # noqa: F401 (kept for parity/logging)

from python.vision.train.grid_intersection.data import DataCfg, make_loaders
from python.vision.train.grid_intersection.losses import loss_5map
from python.vision.train.grid_intersection.eval import eval_val_miniset
from python.vision.train.grid_intersection.postproc import PPConfig
from python.vision.train.grid_intersection.metrics import EvalConfig

# Model must output 6 channels: [A,H,V,J,Ox,Oy]
from python.vision.models.grid_intersection_net import GridIntersectionNet


# -------- CPU threading: pick reasonable defaults (override via env vars) --------
def _int_env(name: str, default: int) -> int:
    try:
        v = int(os.environ.get(name, "").strip())
        return v if v > 0 else default
    except Exception:
        return default


_CPU_MAIN_THREADS = _int_env("OMP_NUM_THREADS", 8)
_CPU_MKL_THREADS  = _int_env("MKL_NUM_THREADS", 8)
torch.set_num_threads(_CPU_MAIN_THREADS)
torch.set_num_interop_threads(min(2, max(1, _CPU_MAIN_THREADS // 4)))


# ---------------- Determinism helpers ----------------
def set_global_determinism(seed: int = 0):
    """
    Sets global seeds and cuDNN flags to aim for reproducibility.
    NOTE: For full determinism with DataLoader workers > 0, also seed workers.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Optional YAML config loader
try:
    from python.vision.train.grid_intersection.config import load_yaml, merge_overrides
    HAVE_YAML = True
except Exception:
    HAVE_YAML = False


def save_ckpt(model, opt, epoch, outdir, tag):
    ck = {"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict()}
    torch.save(ck, str(outdir / f"{tag}.pt"))


def _infer_base_ch_from_ckpt(ckpt_path, map_location="cpu"):
    """
    Peek into a checkpoint to infer the backbone width (base_ch).
    Returns an int (e.g., 24/32/48/64) or None if it cannot be inferred.
    Safe to call before model construction.
    """
    try:
        ck = torch.load(ckpt_path, map_location=map_location)
        sd = ck.get("model", ck)  # allow raw state_dict too
        w = sd.get("enc1.block.0.conv.weight", None)
        if w is not None and isinstance(w, torch.Tensor) and w.ndim == 4:
            return int(w.shape[0])  # first block out_channels tracks base_ch
    except Exception:
        pass
    return None


def _print_resolved_cfg(cfg: dict):
    view = {
        "train_manifest": cfg.get("train_manifest"),
        "val_manifest": cfg.get("val_manifest"),
        "image_size": cfg.get("image_size"),
        "batch": cfg.get("batch"),
        "epochs": cfg.get("epochs"),
        "lr": cfg.get("lr"),
        "weight_decay": cfg.get("weight_decay"),
        "outdir": cfg.get("outdir"),
        "device": cfg.get("device"),
        "seed": cfg.get("seed"),
        "model": cfg.get("model"),
        "resume": cfg.get("resume"),
        "config_file": cfg.get("config_file"),
        "num_workers": cfg.get("num_workers"),
        "log_every": cfg.get("log_every"),
        "eval_overlay_max": cfg.get("eval_overlay_max"),
        "eval_save_overlays": cfg.get("eval_save_overlays"),
        "subpixel": cfg.get("subpixel"),
        "eval_only": cfg.get("eval_only", False),
        "eval_epoch": cfg.get("eval_epoch"),
        "cpu_threads": {
            "OMP_NUM_THREADS": _CPU_MAIN_THREADS,
            "MKL_NUM_THREADS": _CPU_MKL_THREADS,
            "torch.num_threads": torch.get_num_threads(),
            "torch.num_interop_threads": torch.get_num_interop_threads(),
        },
        "cudnn": {
            "deterministic": torch.backends.cudnn.deterministic,
            "benchmark": torch.backends.cudnn.benchmark,
        },
    }
    print("[train] resolved config:", json.dumps(view, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()

    # ---- Config (YAML optional) ----
    ap.add_argument("--config", default=None, help="YAML config path (optional)")

    # ---- Data (CLI overrides; None means 'not provided') ----
    ap.add_argument("--train_manifest", default=None, help="Path to train JSONL")
    ap.add_argument("--val_manifest",   default=None, help="Path to val JSONL")
    ap.add_argument("--image_size", type=int, default=None)

    # ---- Training ----
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight_decay", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None, help="Random seed (deterministic mode)")

    # ---- I/O ----
    ap.add_argument("--outdir", default=None, help="Run output directory")
    ap.add_argument("--resume", default=None, help="Checkpoint to resume from")
    ap.add_argument("--device", default=None, help='Preferred device: "cuda" or "cpu"')

    # ---- Model ----
    ap.add_argument("--in_channels", type=int, default=None)
    ap.add_argument("--out_channels", type=int, default=None)
    ap.add_argument("--base_ch", type=int, default=None)

    # ---- Dataloader workers (Windows-friendly default = 0) ----
    ap.add_argument("--num_workers", type=int, default=None)

    # ---- Logging / Eval verbosity ----
    ap.add_argument("--log_every", type=int, default=None, help="print batch log every N steps (0 = off)")
    ap.add_argument("--eval_overlay_max", type=int, default=None, help="max overlays saved in val mini-pass")
    ap.add_argument("--eval_save_overlays", action="store_true", help="save overlays during val mini-pass")
    ap.add_argument("--no_eval_save_overlays", action="store_true", help="disable overlays during val mini-pass")

    # ---- Loss tweak
    ap.add_argument("--w_J", type=float, default=None, help="Extra weight multiplier for J head loss (1.0 = no change)")

    # ---- Subpixel decoder for eval
    ap.add_argument(
        "--subpixel",
        type=str,
        default="quadfit",
        choices=["none", "quadfit", "softargmax"],
        help="Sub-pixel decoder for junctions during eval."
    )

    # ---- Eval-only convenience
    ap.add_argument("--eval_only", action="store_true",
                    help="Skip training and run a single validation mini-pass. Best used with --resume.")
    ap.add_argument("--eval_epoch", type=int, default=None,
                    help="Epoch index to stamp on eval outputs (defaults to resumed epoch or 1).")

    # ---- Extra decode controls (Phase A sweep)
    ap.add_argument("--softargmax_temp", type=float, default=None,
                    help="Softargmax temperature for J decode (only used when --subpixel=softargmax).")
    ap.add_argument("--tj", type=float, default=None, help="J logit temperature (divides logits[:,3] before sigmoid)")
    ap.add_argument("--j_conf", type=float, default=None, help="junction conf threshold")
    ap.add_argument("--j_topk", type=int,   default=None, help="junction NMS top-k")

    args = ap.parse_args()

    # -------- 1) Start with empty cfg + optional YAML --------
    cfg = {}

    if args.config:
        if not HAVE_YAML:
            raise RuntimeError(
                "Requested --config but YAML support isn't available. "
                "Make sure python/vision/train/grid_intersection/config.py exists and PyYAML is installed."
            )
        y = dict(load_yaml(args.config))
        cfg.update(y)

    # -------- 2) Apply CLI overrides --------
    flat_overrides = {
        "train_manifest": args.train_manifest,
        "val_manifest": args.val_manifest,
        "image_size": args.image_size,
        "batch": args.batch,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "outdir": args.outdir,
        "device": args.device,
        "resume": args.resume,
        "config_file": args.config,
        "num_workers": args.num_workers,
        "log_every": args.log_every,
        "eval_overlay_max": args.eval_overlay_max,
        "seed": args.seed,
        "w_J": args.w_J,
        "subpixel": args.subpixel,
        "softargmax_temp": args.softargmax_temp,
        "eval_only": args.eval_only,
        "eval_epoch": args.eval_epoch,
    }

    cfg = merge_overrides(cfg, **flat_overrides) if HAVE_YAML else {k: v for k, v in flat_overrides.items() if v is not None}

    # -------- 3) Defaults --------
    cfg.setdefault("image_size", 768)
    cfg.setdefault("batch", 8)
    cfg.setdefault("epochs", 60)
    cfg.setdefault("lr", 2e-4)
    cfg.setdefault("weight_decay", 1e-4)
    cfg.setdefault("device", "cuda")          # preferred, will auto-fallback to CPU
    cfg.setdefault("num_workers", 0)          # Windows-friendly
    cfg.setdefault("log_every", 50)
    cfg.setdefault("eval_overlay_max", 64)
    cfg.setdefault("seed", 0)
    cfg.setdefault("w_J", 1.0)
    cfg.setdefault("subpixel", "quadfit")
    cfg.setdefault("softargmax_temp", 0.5)

    if cfg.get("eval_save_overlays") is None:
        cfg["eval_save_overlays"] = not bool(args.no_eval_save_overlays)

    # Model sub-structure defaults/overrides
    model_cfg = dict(cfg.get("model", {}))
    model_cfg.setdefault("in_channels", 1)
    model_cfg.setdefault("out_channels", 6)
    model_cfg.setdefault("base_ch", 64)  # default backbone width
    if args.in_channels is not None:
        model_cfg["in_channels"] = args.in_channels
    if args.out_channels is not None:
        model_cfg["out_channels"] = args.out_channels
    if args.base_ch is not None:
        model_cfg["base_ch"] = args.base_ch
    cfg["model"] = model_cfg

    # -------- 3.5) If resuming and no explicit --base_ch, infer base_ch from ckpt (CPU-safe) --------
    # This ensures the instantiated model matches the checkpoint you’re evaluating.
    if cfg.get("resume") and os.path.exists(cfg["resume"]) and (args.base_ch is None):
        inferred = _infer_base_ch_from_ckpt(cfg["resume"], map_location="cpu")
        if inferred is not None:
            cfg["model"]["base_ch"] = int(inferred)
            print(f"[train] inferred base_ch={inferred} from checkpoint", flush=True)

    if not cfg.get("outdir"):
        cfg["outdir"] = f"runs/grid_intersection/gi_{cfg['image_size']}"

    # ---- Determinism (set seeds & cuDNN flags) ----
    set_global_determinism(int(cfg["seed"]))

    # -------- 4) Validate required paths --------
    if not cfg.get("train_manifest") or not cfg.get("val_manifest"):
        raise SystemExit(
            "train.py: train/val manifests must be provided via YAML --config or CLI "
            "--train_manifest/--val_manifest."
        )

    out_root = Path(cfg["outdir"])
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "preds_val").mkdir(exist_ok=True)
    (out_root / "tensorboard").mkdir(exist_ok=True)

    # -------- 5) Persist the effective config (now includes inferred base_ch if any) --------
    effective_cfg = {
        "train_manifest": cfg["train_manifest"],
        "val_manifest": cfg["val_manifest"],
        "image_size": cfg["image_size"],
        "batch": cfg["batch"],
        "epochs": cfg["epochs"],
        "lr": cfg["lr"],
        "weight_decay": cfg["weight_decay"],
        "outdir": cfg["outdir"],
        "device": cfg["device"],
        "seed": cfg["seed"],
        "model": cfg["model"],
        "resume": cfg.get("resume"),
        "config_file": cfg.get("config_file"),
        "num_workers": cfg.get("num_workers"),
        "log_every": cfg.get("log_every"),
        "eval_overlay_max": cfg.get("eval_overlay_max"),
        "eval_save_overlays": cfg.get("eval_save_overlays"),
        "subpixel": cfg.get("subpixel"),
        "eval_only": cfg.get("eval_only", False),
        "eval_epoch": cfg.get("eval_epoch"),
        "cpu_threads": {
            "OMP_NUM_THREADS": _CPU_MAIN_THREADS,
            "MKL_NUM_THREADS": _CPU_MAIN_THREADS,
            "torch.num_threads": torch.get_num_threads(),
            "torch.num_interop_threads": torch.get_num_interop_threads(),
        },
        "cudnn": {
            "deterministic": torch.backends.cudnn.deterministic,
            "benchmark": torch.backends.cudnn.benchmark,
        },
    }
    _print_resolved_cfg(effective_cfg)
    try:
        import yaml
        (out_root / "config_used.yaml").write_text(
            yaml.safe_dump(effective_cfg, sort_keys=False), encoding="utf-8"
        )
    except Exception:
        (out_root / "config_used.json").write_text(
            json.dumps(effective_cfg, indent=2), encoding="utf-8"
        )

    # -------- 6) Data --------
    dcfg = DataCfg(
        train_manifest=cfg["train_manifest"],
        val_manifest=cfg["val_manifest"],
        image_size=cfg["image_size"],
        grayscale=True,
    )
    train_ld, val_ld = make_loaders(dcfg, cfg["batch"], num_workers=cfg["num_workers"])
    print(f"[train] dataset sizes: train={len(train_ld.dataset)} val={len(val_ld.dataset)}", flush=True)
    steps_per_epoch = len(train_ld)
    print(f"[train] steps per epoch: {steps_per_epoch}", flush=True)

    # -------- 7) Device + Model --------
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    print(f"[train] using device: {device.type}", flush=True)

    model = GridIntersectionNet(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        base_ch=cfg["model"]["base_ch"],
    ).to(device)

    # -------- 8) Checkpoint resume (model only) --------
    start_epoch = 1
    best_iou = -float("inf")
    best_mje = float("inf")

    if cfg.get("resume") and os.path.exists(cfg["resume"]):
        ck = torch.load(cfg["resume"], map_location=device)
        model.load_state_dict(ck["model"], strict=True)
        start_epoch = ck.get("epoch", 0) + 1
        print(f"[train] resumed from {cfg['resume']} at epoch {start_epoch}", flush=True)

    # -------- 9) Eval-only early exit (BEFORE building optimizer/scheduler) --------
    pp_cfg = PPConfig()
    ev_cfg = EvalConfig()

    if cfg.get("eval_only", False):
        resumed_epoch = max(1, start_epoch - 1)  # epoch the ckpt corresponds to
        eval_epoch = int(cfg["eval_epoch"]) if cfg.get("eval_epoch") is not None else resumed_epoch
        preds_dir = out_root / f"preds_val_epoch{eval_epoch:02d}"
        print(f"[val]   Eval-only mode: writing to {preds_dir}", flush=True)

        metrics = eval_val_miniset(
            model, val_ld, device, preds_dir, eval_epoch, pp_cfg, ev_cfg,
            save_overlays=bool(cfg["eval_save_overlays"]),
            overlay_max=int(cfg["eval_overlay_max"]),
            subpixel=cfg.get("subpixel", "quadfit"),
            softargmax_temp=float(cfg.get("softargmax_temp", 0.5)),
            tj=float(args.tj) if args.tj is not None else None,
            j_conf=float(args.j_conf) if args.j_conf is not None else None,
            j_topk=int(args.j_topk) if args.j_topk is not None else None,
        )

        # quick console summary (normalized + finite rates + AP@2 finite)
        iou_A = np.array([m.get("A_IoU", np.nan) for m in metrics], dtype=np.float64)
        iou_H = np.array([m.get("H_IoU", np.nan) for m in metrics], dtype=np.float64)
        iou_V = np.array([m.get("V_IoU", np.nan) for m in metrics], dtype=np.float64)
        val_iou_mean = float(np.nanmean(np.concatenate([
            iou_A[~np.isnan(iou_A)],
            iou_H[~np.isnan(iou_H)],
            iou_V[~np.isnan(iou_V)]
        ]))) if ((~np.isnan(iou_A)).any() or (~np.isnan(iou_H)).any() or (~np.isnan(iou_V)).any()) else float("nan")

        jmje_vals      = np.array([m.get("J_MJE", np.nan) for m in metrics], dtype=np.float64)
        jmje_norm_vals = np.array([m.get("J_MJE_norm", np.nan) for m in metrics], dtype=np.float64)
        p6  = np.array([m.get("J_MJE<= 6px",  np.nan) for m in metrics], dtype=np.float64)
        p8  = np.array([m.get("J_MJE<= 8px",  np.nan) for m in metrics], dtype=np.float64)
        p10 = np.array([m.get("J_MJE<= 10px", np.nan) for m in metrics], dtype=np.float64)
        ap2 = np.array([m.get("J_AP@2px_finite", np.nan) for m in metrics], dtype=np.float64)
        pj100 = np.array([m.get("pred_J_eq_100", np.nan) for m in metrics], dtype=np.float64)

        val_j_mje       = float(np.nanmean(jmje_vals))      if (~np.isnan(jmje_vals)).any()      else float("nan")
        val_j_mje_norm  = float(np.nanmean(jmje_norm_vals)) if (~np.isnan(jmje_norm_vals)).any() else float("nan")
        r6 = float(np.nanmean(p6))   if (~np.isnan(p6)).any()   else float("nan")
        r8 = float(np.nanmean(p8))   if (~np.isnan(p8)).any()   else float("nan")
        r10 = float(np.nanmean(p10)) if (~np.isnan(p10)).any()  else float("nan")
        ap2m = float(np.nanmean(ap2)) if (~np.isnan(ap2)).any() else float("nan")
        pj100m = float(np.nanmean(pj100)) if (~np.isnan(pj100)).any() else float("nan")

        print(
            f"[val]   Eval-only: IoU_mean={val_iou_mean:.4f}  "
            f"J_MJE={val_j_mje:.4f}  J_MJE_norm={val_j_mje_norm:.4f}  "
            f"AP@2px_finite={ap2m:.3f}  "
            f"J_MJE<=8px={r8:.3f}  pred_J==100={pj100m:.3f}",
            flush=True
        )
        return

    # -------- 10) Optimizer / Scheduler (not reached in eval-only) --------
    opt = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg["epochs"]))

    # If we resumed AND we’re training, optionally restore optimizer too (if present)
    if cfg.get("resume") and os.path.exists(cfg["resume"]):
        try:
            if "opt" in ck and not cfg.get("eval_only", False):
                opt.load_state_dict(ck["opt"])
        except Exception:
            pass

    best_iou = -float("inf")
    best_mje = float("inf")

    # -------- 11) Training loop --------
    log_path = out_root / "train.log"
    with open(log_path, "a", encoding="utf-8") as LOG:
        w_J = float(cfg.get("w_J", 1.0))
        for epoch in range(start_epoch, cfg["epochs"] + 1):
            print(f"\n[train] === Epoch {epoch}/{cfg['epochs']} START ===", flush=True)
            model.train()
            t0 = time.time()
            acc_loss = 0.0
            steps = 0
            nonfinite_abort = False

            for step, (img, y, rec) in enumerate(train_ld, start=1):
                t_batch0 = time.time()
                img = img.to(device, non_blocking=True)
                y   = y.to(device, non_blocking=True)

                logits = model(img)  # (N,6,H,W)
                probs  = torch.sigmoid(logits[:, 0:4])  # A,H,V,J

                # --- progress/ETA logging ---
                if cfg["log_every"] and (step % cfg["log_every"] == 0 or step == 1 or step == steps_per_epoch):
                    elapsed = time.time() - t0
                    avg_step = elapsed / step
                    eta_epoch = (steps_per_epoch - step) * avg_step
                    sps = (img.shape[0] * step) / max(elapsed, 1e-6)  # samples/sec averaged
                    msg = (
                        f"[train][ep {epoch}] {step}/{steps_per_epoch} "
                        f"loss={acc_loss/max(1,steps):.4f} "
                        f"avg_step={avg_step:.2f}s "
                        f"ETA_epoch={eta_epoch/60:.1f}m "
                        f"sps={sps:.1f}"
                    )
                    print(msg, flush=True)

                pred = torch.cat([probs, logits[:, 4:6]], dim=1)

                comps = {}
                try:
                    loss, comps = loss_5map(pred, y, lambda_o=0.5, w_J=w_J)
                except TypeError:
                    loss, comps = loss_5map(pred, y, lambda_o=0.5)
                    j_keys = ["L_J", "loss_J", "bce_J", "j_bce", "J_bce"]
                    j_term = None
                    if isinstance(comps, dict):
                        for k in j_keys:
                            if k in comps:
                                j_val = comps[k]
                                if torch.is_tensor(j_val):
                                    j_term = j_val
                                else:
                                    j_term = torch.as_tensor(j_val, device=loss.device, dtype=loss.dtype)
                                break
                    if j_term is not None and abs(w_J - 1.0) > 1e-6:
                        loss = loss + (w_J - 1.0) * j_term

                if not torch.isfinite(loss):
                    print(f"[train][FATAL] non-finite loss at epoch {epoch} step {step}: {float(loss.detach().cpu())}", flush=True)
                    nonfinite_abort = True
                    break

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                acc_loss += float(loss.item())
                steps += 1

                if cfg["log_every"] and step % cfg["log_every"] == 0:
                    dtb = time.time() - t_batch0
                    sps = img.shape[0] / max(dtb, 1e-6)
                    msg = f"[train][ep {epoch}] step {step}/{steps_per_epoch} loss={acc_loss/steps:.4f} sps={sps:.1f}"
                    print(msg, flush=True)

            if nonfinite_abort:
                print("[train] Aborting epoch due to non-finite loss.", flush=True)
                break

            scheduler.step()
            train_loss = acc_loss / max(1, steps)
            dt = time.time() - t0
            print(f"[train] Epoch {epoch} END: train_loss={train_loss:.4f} time={dt:.1f}s", flush=True)

            # ---- Per-epoch VAL on a miniset ----
            print(f"[val]   Epoch {epoch}: running eval mini-pass...", flush=True)
            preds_dir = out_root / f"preds_val_epoch{epoch:02d}"

            metrics = eval_val_miniset(
                model, val_ld, device, preds_dir, epoch, pp_cfg, ev_cfg,
                save_overlays=bool(cfg["eval_save_overlays"]),
                overlay_max=int(cfg["eval_overlay_max"]),
                subpixel=cfg.get("subpixel", "quadfit"),
                softargmax_temp=float(cfg.get("softargmax_temp", 0.5)),
                tj=float(args.tj) if args.tj is not None else None,
                j_conf=float(args.j_conf) if args.j_conf is not None else None,
                j_topk=int(args.j_topk) if args.j_topk is not None else None,
            )

            # ---- Selection = mean IoU over A/H/V (higher is better)
            iou_A = np.array([m.get("A_IoU", np.nan) for m in metrics], dtype=np.float64)
            iou_H = np.array([m.get("H_IoU", np.nan) for m in metrics], dtype=np.float64)
            iou_V = np.array([m.get("V_IoU", np.nan) for m in metrics], dtype=np.float64)

            val_iou_mean = float(np.nanmean(np.concatenate([
                iou_A[~np.isnan(iou_A)],
                iou_H[~np.isnan(iou_H)],
                iou_V[~np.isnan(iou_V)]
            ]))) if (
                (~np.isnan(iou_A)).any() or
                (~np.isnan(iou_H)).any() or
                (~np.isnan(iou_V)).any()
            ) else float("nan")

            # ---- Also report richer J metrics
            jmje_vals      = np.array([m.get("J_MJE", np.nan) for m in metrics], dtype=np.float64)
            jmje_norm_vals = np.array([m.get("J_MJE_norm", np.nan) for m in metrics], dtype=np.float64)
            p6  = np.array([m.get("J_MJE<= 6px",  np.nan) for m in metrics], dtype=np.float64)
            p8  = np.array([m.get("J_MJE<= 8px",  np.nan) for m in metrics], dtype=np.float64)
            p10 = np.array([m.get("J_MJE<= 10px", np.nan) for m in metrics], dtype=np.float64)
            ap2 = np.array([m.get("J_AP@2px_finite", np.nan) for m in metrics], dtype=np.float64)
            pj100 = np.array([m.get("pred_J_eq_100", np.nan) for m in metrics], dtype=np.float64)

            val_j_mje       = float(np.nanmean(jmje_vals))      if (~np.isnan(jmje_vals)).any()      else float("nan")
            val_j_mje_norm  = float(np.nanmean(jmje_norm_vals)) if (~np.isnan(jmje_norm_vals)).any() else float("nan")
            r6 = float(np.nanmean(p6))   if (~np.isnan(p6)).any()   else float("nan")
            r8 = float(np.nanmean(p8))   if (~np.isnan(p8)).any()   else float("nan")
            r10 = float(np.nanmean(p10)) if (~np.isnan(p10)).any()  else float("nan")
            ap2m = float(np.nanmean(ap2)) if (~np.isnan(ap2)).any() else float("nan")
            pj100m = float(np.nanmean(pj100)) if (~np.isnan(pj100)).any() else float("nan")

            print(
                f"[val]   Epoch {epoch}: IoU_mean={val_iou_mean:.4f}  "
                f"J_MJE={val_j_mje:.4f}  J_MJE_norm={val_j_mje_norm:.4f}  "
                f"AP@2px_finite={ap2m:.3f}  "
                f"J_MJE<=8px={r8:.3f}  pred_J==100={pj100m:.3f}",
                flush=True
            )

            # ---- Save last
            save_ckpt(model, opt, epoch, out_root, "last")

            # ---- Save best by IoU (selection)
            if not np.isnan(val_iou_mean) and val_iou_mean > best_iou:
                best_iou = val_iou_mean
                save_ckpt(model, opt, epoch, out_root, "best_iou")
                print(f"[ckpt]  Epoch {epoch}: new best IoU_mean={best_iou:.4f} -> saved best_iou.pt", flush=True)

            # ---- Also track best MJE for reference
            if not np.isnan(val_j_mje) and val_j_mje < best_mje:
                best_mje = val_j_mje
                save_ckpt(model, opt, epoch, out_root, "best_mje")
                print(f"[ckpt]  Epoch {epoch}: new best J_MJE={best_mje:.4f} -> saved best_mje.pt", flush=True)

            # per-epoch scalar log (JSON line)
            line = json.dumps({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_IoU_mean": val_iou_mean,
                "val_J_MJE": val_j_mje,
                "val_J_MJE_norm": val_j_mje_norm,
                "val_J_MJE_le_8px": r8,
                "val_J_AP2_finite": ap2m,
                "val_predJ_eq_100": pj100m,
                "lr": scheduler.get_last_lr()[0],
                "time_sec": dt
            })
            LOG.write(line + "\n"); LOG.flush()
            print(line, flush=True)

    print(f"[train] training complete. Best IoU_mean: {best_iou:.4f} | Best J_MJE: {best_mje:.4f}", flush=True)


if __name__ == "__main__":
    main()