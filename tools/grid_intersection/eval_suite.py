"""
Batch evaluation + overlay dump for the 5-map grid-line model (CPU-friendly).

- Loads a checkpoint (.pt) and a manifest (jsonl).
- Runs inference over the whole manifest.
- Builds 10×10 junction structures for BOTH prediction and GT so J_MJE is finite.
- Saves per-image overlays and aggregates metrics into JSON + Markdown.

Outputs go to:
  <outdir>/
    preds_manifest/
      <stem>_polylines.jpg
      <stem>_A.png / _H.png / _V.png   (if --save_heatmaps)
      <stem>.preds.npz                 (A,H,V,J,O + juncs + polylines + J_pred_struct + J_gt_struct)
    eval_report.json
    eval_report.md
"""

from __future__ import annotations  # must be the first non-docstring statement

# --- repo path bootstrap (so this script runs from any cwd) ---
import sys
from pathlib import Path

# this file: <repo>/tools/grid_intersection/eval_suite.py
# repo root is two levels up from here (parents[2] = <repo>)
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
# --------------------------------------------------------------

import argparse, json, os, math, time
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import cv2
import torch

from python.vision.models.grid_intersection_net import GridIntersectionNet
from python.vision.train.grid_intersection.postproc import (
    PPConfig, bin_maps, junction_nms, cluster_grid, assemble_polylines
)
from python.vision.train.grid_intersection.metrics import (
    EvalConfig, evaluate_case, summarize_to_json_md
)


# -------------------------
# Utilities
# -------------------------
def read_jsonl(p: str) -> List[Dict]:
    recs = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def _resize_nn(m: np.ndarray, S: int) -> np.ndarray:
    """Resize label map to SxS using nearest-neighbor if needed."""
    if m.ndim == 3 and m.shape[-1] == 1:
        m = m[..., 0]
    if m.shape[0] != S or m.shape[1] != S:
        return cv2.resize(m, (S, S), interpolation=cv2.INTER_NEAREST)
    return m


def _to_uint8_255(arr: np.ndarray) -> np.uint8:
    """Robustly convert float [0..1] or uint8 to uint8[0..255]."""
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0) * 255.0
        return arr.astype(np.uint8)
    return arr


def _colorize(h: np.ndarray) -> np.ndarray:
    """Apply Turbo/JET colormap to a float [0,1] heatmap."""
    h8 = _to_uint8_255(h)
    cmap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
    return cv2.applyColorMap(h8, cmap)


def _flatten_grid_to_100(grid: np.ndarray) -> Optional[np.ndarray]:
    """
    Accepts grid returned by cluster_grid (10x10x2 or nested list of points).
    Returns (100,2) float32 in row-major, or None on failure.
    """
    try:
        flat = np.array([[float(grid[r][c][0]), float(grid[r][c][1])] for r in range(10) for c in range(10)],
                        dtype=np.float32)
        if flat.shape == (100, 2) and np.all(np.isfinite(flat)):
            return flat
    except Exception:
        pass
    return None


def _precheck_mje(pred_100: np.ndarray, gt_100: np.ndarray) -> float:
    """
    Cheap 1–1 row-major MJE precheck used as a fallback when the strict metric yields NaN/inf.
    Both inputs must be (100,2) with finite floats.
    """
    if not isinstance(pred_100, np.ndarray) or not isinstance(gt_100, np.ndarray):
        return float("nan")
    if pred_100.shape != (100, 2) or gt_100.shape != (100, 2):
        return float("nan")
    if not (np.all(np.isfinite(pred_100)) and np.all(np.isfinite(gt_100))):
        return float("nan")
    d = pred_100 - gt_100
    return float(np.mean(np.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1])))


def _build_struct_from_jmap(J01: np.ndarray, S: int, pp_cfg: PPConfig) -> Optional[np.ndarray]:
    """
    From a float [0,1] junction heatmap, run NMS + cluster_grid to produce a (100,2) lattice.
    """
    J8 = _to_uint8_255(_resize_nn(J01, S))
    pts = junction_nms(J8, pp_cfg)  # (N,3) -> x,y,score
    _, _, grid = cluster_grid(pts, pp_cfg, S, S)
    return _flatten_grid_to_100(grid)


def _fmt_metrics(m: Dict[str, Any]) -> str:
    """Compact, safe metric preview for logs."""
    keys = ["IoU_A", "IoU_H", "IoU_V", "IoU_mean", "J_MJE", "J_MJE_norm", "J_MJE<=8px", "J_AP@2px_finite"]
    parts = []
    for k in keys:
        v = m.get(k, None)
        if v is None:
            continue
        if isinstance(v, float):
            parts.append(f"{k}={v:.4f}")
        else:
            parts.append(f"{k}={v}")
    return " ".join(parts) if parts else "(no-metrics)"


# -------------------------
# Main
# -------------------------
@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Path to jsonl (train/val/real)")
    ap.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    ap.add_argument("--outdir", required=True, help="Output run dir (will create)")

    # runtime / model
    ap.add_argument("--device", default="cpu", help='Preferred device: "cpu" or "cuda" (falls back to cpu)')
    ap.add_argument("--image_size", type=int, default=768)
    ap.add_argument("--base_ch", type=int, default=64, help="Model base channels (match training)")

    # decode & thresholds (align with your sweep)
    ap.add_argument("--tj", type=float, default=0.70, help="J logit temperature: sigmoid(logit/tj)")
    ap.add_argument("--j_conf", type=float, default=0.05, help="junction NMS conf threshold")
    ap.add_argument("--j_topk", type=int,   default=150,  help="junction NMS top-k cap")

    # outputs
    ap.add_argument("--save_npz", action="store_true", help="Save raw preds to .npz per case")
    ap.add_argument("--save_heatmaps", action="store_true", help="Also save color A/H/V tiles")
    ap.add_argument("--overlay_max", type=int, default=999999, help="Cap overlays for huge sets")
    ap.add_argument("--log_every", type=int, default=10, help="Per-case progress logging cadence")
    ap.add_argument("--quiet", action="store_true", help="Reduce per-case logs")
    args = ap.parse_args()

    t0 = time.time()

    out_root = Path(args.outdir)
    pred_dir = out_root / "preds_manifest"
    pred_dir.mkdir(parents=True, exist_ok=True)

    # ----- Model / device -----
    want_cuda = (args.device.strip().lower() == "cuda")
    device = torch.device("cuda" if (want_cuda and torch.cuda.is_available()) else "cpu")
    model = GridIntersectionNet(in_channels=1, out_channels=6, base_ch=args.base_ch).to(device)

    print("[eval_suite] === CONFIG ===", flush=True)
    cfg_echo = {
        "manifest": args.manifest,
        "checkpoint": args.checkpoint,
        "outdir": str(out_root),
        "device": device.type,
        "image_size": args.image_size,
        "base_ch": args.base_ch,
        "tj": args.tj,
        "j_conf": args.j_conf,
        "j_topk": args.j_topk,
        "save_npz": bool(args.save_npz),
        "save_heatmaps": bool(args.save_heatmaps),
        "overlay_max": args.overlay_max,
        "log_every": args.log_every,
        "quiet": bool(args.quiet),
    }
    print(json.dumps(cfg_echo, indent=2), flush=True)

    print("[eval_suite] Loading checkpoint...", flush=True)
    ck = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    print("[eval_suite] Checkpoint loaded.", flush=True)

    # ----- Configs -----
    pp_cfg = PPConfig()
    # align NMS knobs with CLI
    if hasattr(pp_cfg, "j_topk"):     pp_cfg.j_topk     = max(int(args.j_topk), 1)
    if hasattr(pp_cfg, "j_topk_cap"): pp_cfg.j_topk_cap = max(int(args.j_topk), int(getattr(pp_cfg, "j_topk_cap", 180)))
    if hasattr(pp_cfg, "j_topk_min"): pp_cfg.j_topk_min = min(int(args.j_topk), int(getattr(pp_cfg, "j_topk_min", 120)))
    if hasattr(pp_cfg, "j_conf"):     pp_cfg.j_conf     = float(args.j_conf)

    ev_cfg = EvalConfig()

    # ----- Data -----
    print("[eval_suite] Reading manifest...", flush=True)
    recs = read_jsonl(args.manifest)
    total = len(recs)
    if total == 0:
        print("[eval_suite] Manifest is empty. Nothing to do.", flush=True)
        return
    print(f"[eval_suite] {total} records found.", flush=True)

    case_metrics: List[Dict[str, float]] = []
    n_overlay = 0
    n_errors = 0

    # Running averages
    acc_times: List[float] = []

    # ----- Main loop -----
    for idx, r in enumerate(recs, start=1):
        case_t0 = time.time()
        img_path = r["image_path"]
        lab_path = r.get("label_path", None)

        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[skip] ({idx}/{total}) unreadable image: {img_path}", flush=True)
                continue

            # Resize to square model size
            S = args.image_size
            inp = cv2.resize(img, (S, S), interpolation=cv2.INTER_AREA)
            x = torch.from_numpy(inp[None, None].astype(np.float32) / 255.0).to(device)

            # Forward
            logits = model(x)  # (1,6,S,S)
            # Temperature on J logit BEFORE sigmoid (aligns with your sweep)
            A = torch.sigmoid(logits[:, 0:1])[0, 0].cpu().numpy().astype(np.float32)
            H = torch.sigmoid(logits[:, 1:2])[0, 0].cpu().numpy().astype(np.float32)
            V = torch.sigmoid(logits[:, 2:3])[0, 0].cpu().numpy().astype(np.float32)
            J = torch.sigmoid(logits[:, 3] / float(max(args.tj, 1e-6)))[0].cpu().numpy().astype(np.float32)
            Ox = logits[0, 4].cpu().numpy()
            Oy = logits[0, 5].cpu().numpy()
            mag = np.sqrt(Ox * Ox + Oy * Oy) + 1e-6
            O = np.stack([Ox / mag, Oy / mag], axis=2).astype(np.float32)

            # Postproc to get lattice + polylines (PRED STRUCT = 10x10)
            A_bin, H_bin, V_bin = bin_maps(_to_uint8_255(A), _to_uint8_255(H), _to_uint8_255(V), pp_cfg)
            jpts = junction_nms(_to_uint8_255(J), pp_cfg)  # (N,3)
            _, _, grid_pred = cluster_grid(jpts, pp_cfg, S, S)
            J_pred_struct = _flatten_grid_to_100(grid_pred)  # (100,2) or None

            # Pred dict for metrics
            pred = {
                "A": _to_uint8_255(A),
                "H": _to_uint8_255(H),
                "V": _to_uint8_255(V),
                # IMPORTANT: metrics expect structured 100x2; fall back to empty if not available
                "J_pts": (J_pred_struct if J_pred_struct is not None else np.zeros((0, 2), np.float32)),
                # lines for overlay only
                "H_lines": [],  # not required by evaluate_case; we'll draw below
                "V_lines": [],
            }

            # Also assemble polylines for visualization
            H_lines, V_lines = assemble_polylines(grid_pred, A, O, pp_cfg)

            # ----- Ground truth (maps + GT STRUCT = 10x10) -----
            gt: Dict[str, Any] = {}
            has_gt_maps = False
            J_gt_struct: Optional[np.ndarray] = None
            if lab_path and os.path.exists(lab_path):
                with np.load(lab_path) as lab:
                    for k in ["A", "H", "V"]:
                        if k in lab:
                            gt[k] = _to_uint8_255(_resize_nn(lab[k], S))
                            has_gt_maps = True

                    # Prefer GT J heatmap if present; otherwise try GT points then cluster
                    if "J" in lab:
                        J_gt = lab["J"].astype(np.float32)
                        J_gt = _resize_nn(J_gt, S)
                        J_gt = np.clip(J_gt, 0.0, 1.0)
                        J_gt_struct = _build_struct_from_jmap(J_gt, S, pp_cfg)
                    elif "J_pts" in lab:
                        pts = np.array(lab["J_pts"]).reshape(-1, 2)
                        # synthesize a heatmap is overkill; instead cluster directly:
                        pts3 = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float32)])  # (N,3)
                        _, _, grid_gt = cluster_grid(pts3, pp_cfg, S, S)
                        J_gt_struct = _flatten_grid_to_100(grid_gt)

                    if J_gt_struct is None:
                        # Last resort: derive from H/V maps if available
                        if "J" in lab:
                            J_gt_struct = _build_struct_from_jmap(lab["J"].astype(np.float32), S, pp_cfg)

                    if J_gt_struct is not None:
                        gt["J_pts"] = J_gt_struct
                    else:
                        gt["J_pts"] = np.zeros((0, 2), np.float32)

            # ----- Metrics (with fallback if strict J_MJE is NaN) -----
            metrics = evaluate_case(pred, gt, ev_cfg)

            # Build a robust diagonal normalizer if needed (from GT lattice)
            if J_gt_struct is not None and J_pred_struct is not None:
                # Fallback if metric-side J_MJE came back NaN/inf
                jmje = float(metrics.get("J_MJE", float("nan")))
                if (not np.isfinite(jmje)):
                    fallback = _precheck_mje(J_pred_struct, J_gt_struct)
                    if np.isfinite(fallback):
                        metrics["J_MJE"] = float(fallback)
                        metrics["J_MJE_fallback"] = 1.0

                # Provide J_MJE_norm if missing
                if not np.isfinite(float(metrics.get("J_MJE_norm", float("nan")))):
                    gx_min = float(np.min(J_gt_struct[:, 0])); gx_max = float(np.max(J_gt_struct[:, 0]))
                    gy_min = float(np.min(J_gt_struct[:, 1])); gy_max = float(np.max(J_gt_struct[:, 1]))
                    dx = max(1e-6, gx_max - gx_min); dy = max(1e-6, gy_max - gy_min)
                    diag = math.hypot(dx, dy)
                    jmje_final = float(metrics.get("J_MJE", float("nan")))
                    if np.isfinite(jmje_final) and diag > 1e-6:
                        metrics["J_MJE_norm"] = float(jmje_final / diag)

                # Add thresholded flags if missing (so epoch means aren't NaN)
                for thr in (6, 8, 10):
                    key = f"J_MJE<= {thr}px"
                    if key not in metrics:
                        jmje_final = float(metrics.get("J_MJE", float("nan")))
                        metrics[key] = 1.0 if (np.isfinite(jmje_final) and jmje_final <= float(thr)) else 0.0

            case_metrics.append(metrics)

            # Save overlays (polylines + optional heatmaps)
            stem = Path(img_path).stem
            if n_overlay < args.overlay_max:
                color = cv2.cvtColor(inp, cv2.COLOR_GRAY2BGR)
                poly_img = color.copy()
                for pl in (H_lines + V_lines):
                    for a, b in zip(pl[:-1], pl[1:]):
                        cv2.line(poly_img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imwrite(str(pred_dir / f"{stem}_polylines.jpg"), poly_img)

                if args.save_heatmaps:
                    cv2.imwrite(str(pred_dir / f"{stem}_A.png"), _colorize(A))
                    cv2.imwrite(str(pred_dir / f"{stem}_H.png"), _colorize(H))
                    cv2.imwrite(str(pred_dir / f"{stem}_V.png"), _colorize(V))

                n_overlay += 1

            # Save raw preds as NPZ
            if args.save_npz:
                np.savez_compressed(
                    str(pred_dir / f"{stem}.preds.npz"),
                    A=A, H=H, V=V, J=J, O=O,
                    jpts=np.array(jpts, np.float32, copy=False),
                    H_lines=np.array([np.array(pl, np.float32) for pl in H_lines], dtype=object),
                    V_lines=np.array([np.array(pl, np.float32) for pl in V_lines], dtype=object),
                    J_pred_struct=(J_pred_struct if J_pred_struct is not None else np.zeros((0, 2), np.float32)),
                    J_gt_struct=(J_gt_struct if J_gt_struct is not None else np.zeros((0, 2), np.float32)),
                )

            # ---- progress / running stats ----
            dt = time.time() - case_t0
            acc_times.append(dt)
            avg_t = float(np.mean(acc_times))
            eta = (total - idx) * avg_t

            if not args.quiet and (idx == 1 or idx % max(1, args.log_every) == 0 or idx == total):
                msg = [
                    f"[eval_suite] Case {idx}/{total}",
                    f"file={Path(img_path).name}",
                    f"gt_maps={'yes' if ('A' in gt or 'H' in gt or 'V' in gt) else 'no'}",
                    f"j_pred={len(jpts)}",
                    f"H_lines={len(H_lines)} V_lines={len(V_lines)}",
                    f"time={dt:.2f}s avg={avg_t:.2f}s ETA={eta/60.0:.1f}m",
                    _fmt_metrics(metrics),
                ]
                print(" | ".join(msg), flush=True)

        except Exception as e:
            n_errors += 1
            print(f"[eval_suite][ERROR] ({idx}/{total}) file={img_path} -> {repr(e)}", flush=True)

    # ----- Aggregate reports -----
    js, md = summarize_to_json_md(case_metrics)
    (out_root / "eval_report.json").write_text(js, encoding="utf-8")
    (out_root / "eval_report.md").write_text(md, encoding="utf-8")

    # Final summary to stdout
    dur = time.time() - t0
    mean_spc = float(np.mean(acc_times)) if acc_times else float("nan")
    summary = {
        "cases": len(case_metrics),
        "errors": n_errors,
        "duration_sec": dur,
        "mean_seconds_per_case": mean_spc,
        "overlays_written": n_overlay,
        "pred_dir": str(pred_dir),
        "report_json": str(out_root / "eval_report.json"),
        "report_md": str(out_root / "eval_report.md"),
    }
    print("[eval_suite] === SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[eval_suite] wrote {len(case_metrics)} cases → {out_root}", flush=True)


if __name__ == "__main__":
    main()