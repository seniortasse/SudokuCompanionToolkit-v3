#!/usr/bin/env python3
"""
inspect_cellnet_tflite.py
---------------------------------
Inspect a CellNet TFLite model on one image or a JSONL manifest.

Features:
- Prints input/output tensor shapes and names.
- Runs inference on one image or many (via JSONL manifest).
- Outputs raw logits, post-processed probabilities, predictions, and candidate bitmasks.
- Saves per-sample CSV (and optional JSONL) for batch runs.

Requirements:
- Python 3.8+
- Pillow, numpy
- Either tensorflow (2.x) OR tflite-runtime

Example usages:
  # Single image
  python tools/inspect_cellnet_tflite.py ^
    --tflite exports/cell_interpreter_fp32.tflite ^
    --image datasets/cells/cell_interpreter/phone_capture_5/cell_0_0.png

  # Single image with inner-crop and different size
  python tools/inspect_cellnet_tflite.py ^
    --tflite exports/cell_interpreter_fp32.tflite ^
    --image datasets/cells/cell_interpreter/phone_capture_5/cell_0_0.png ^
    --img 64 --inner-crop 1.0 --cand-thr 0.58

  # Batch via JSONL manifest (paths inside JSONL are relative to --root if provided)
  python tools/inspect_cellnet_tflite.py ^
    --tflite exports/cell_interpreter_fp32.tflite ^
    --manifest datasets/cells/cell_interpreter/phone_capture_5/cells_real_labeled.jsonl ^
    --root . ^
    --out-dir runs/inspect_phone_capture_5 ^
    --cand-thr 0.58
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
from PIL import Image

def try_load_tflite_interpreter():
    try:
        from tensorflow.lite.python.interpreter import Interpreter  # type: ignore
        return Interpreter, "tensorflow.lite"
    except Exception:
        try:
            from tflite_runtime.interpreter import Interpreter  # type: ignore
            return Interpreter, "tflite_runtime"
        except Exception as e2:
            print("[ERR] Could not import TFLite interpreter. Install either:")
            print("      pip install 'tensorflow==2.16.*'  # or compatible 2.x")
            print("   or pip install tflite-runtime        # pick wheel for your OS/arch")
            raise

def center_square_crop(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))

def inner_crop_fraction(img: Image.Image, frac: float) -> Image.Image:
    assert 0.0 < frac <= 1.0
    s = min(img.size)
    new_s = max(1, int(round(s * frac)))
    left = (img.size[0] - new_s) // 2
    top  = (img.size[1] - new_s) // 2
    return img.crop((left, top, left + new_s, top + new_s))

def preprocess_to_nhwc(path: Path, img_size: int, crop_frac: float) -> np.ndarray:
    img = Image.open(path).convert("L")
    img = center_square_crop(img)
    if crop_frac < 1.0:
        img = inner_crop_fraction(img, crop_frac)
    img = img.resize((img_size, img_size), Image.BICUBIC)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - 0.5) / 0.5
    x = x.reshape(1, img_size, img_size, 1)
    return x

def softmax_row(v: np.ndarray) -> np.ndarray:
    v = v - np.max(v, axis=1, keepdims=True)
    e = np.exp(v)
    return e / np.sum(e, axis=1, keepdims=True)

def sigmoid(v: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-v))

def run_single_image(interpreter, x: np.ndarray, cand_thr: float):
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()
    out = interpreter.get_output_details()

    # If model expects NCHW (rare), transpose
    in_shape = inp[0]["shape"]
    if tuple(in_shape) == (1, 1, x.shape[1], x.shape[2]):
        x_feed = np.transpose(x, (0,3,1,2))
    else:
        x_feed = x

    interpreter.set_tensor(inp[0]["index"], x_feed.astype(np.float32))
    interpreter.invoke()

    logits_candidates      = interpreter.get_tensor(out[0]["index"])
    logits_solution   = interpreter.get_tensor(out[1]["index"])
    logits_given = interpreter.get_tensor(out[2]["index"])

    p_given    = softmax_row(logits_given)[0]
    p_solution = softmax_row(logits_solution)[0]
    p_cand     = sigmoid(logits_candidates)[0]

    pred_given    = int(np.argmax(p_given))
    pred_solution = int(np.argmax(p_solution))

    cand_mask_bits = 0
    cand_on = []
    for d in range(10):
        if p_cand[d] >= cand_thr:
            cand_mask_bits |= (1 << d)
            cand_on.append(d)

    return {
        "input_shape": in_shape.tolist(),
        "output_details": [{
            "index": out[i]["index"], "name": out[i].get("name",""), "shape": out[i]["shape"].tolist()
        } for i in range(len(out))],
        "logits_given":      logits_given[0].tolist(),
        "logits_solution":   logits_solution[0].tolist(),
        "logits_candidates": logits_candidates[0].tolist(),
        "p_given":    p_given.tolist(),
        "p_solution": p_solution.tolist(),
        "p_candidates": p_cand.tolist(),
        "pred_given": pred_given,
        "pred_solution": pred_solution,
        "cand_mask_bits": cand_mask_bits,
        "cand_on": cand_on,
    }

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def write_json(p: Path, obj):
    with p.open("w", encoding="utf-8") as f:
        import json
        json.dump(obj, f, indent=2)

def write_csv(p: Path, rows: List[dict]):
    if not rows:
        return
    import csv
    def row_to_flat(r):
        flat = {
            "path": r.get("path",""),
            "pred_given": r["pred_given"],
            "pred_solution": r["pred_solution"],
            "cand_mask_bits": r["cand_mask_bits"],
            "cand_on": " ".join(map(str, r["cand_on"])),
        }
        for i, v in enumerate(r["p_given"]):      flat[f"p_given_{i}"] = v
        for i, v in enumerate(r["p_solution"]):   flat[f"p_solution_{i}"] = v
        for i, v in enumerate(r["p_candidates"]): flat[f"p_cand_{i}"] = v
        for i, v in enumerate(r["logits_given"]):      flat[f"logits_given_{i}"] = v
        for i, v in enumerate(r["logits_solution"]):   flat[f"logits_solution_{i}"] = v
        for i, v in enumerate(r["logits_candidates"]): flat[f"logits_cand_{i}"] = v
        if "label_given" in r:      flat["label_given"] = r["label_given"]
        if "label_solution" in r:   flat["label_solution"] = r["label_solution"]
        if "label_candidates" in r: flat["label_candidates"] = " ".join(map(str, r["label_candidates"]))
        return flat
    flat_rows = [row_to_flat(r) for r in rows]
    fieldnames = list(flat_rows[0].keys())
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in flat_rows:
            w.writerow(r)

def parse_args():
    ap = argparse.ArgumentParser("Inspect a CellNet TFLite model")
    ap.add_argument("--tflite", required=True, type=Path, help="Path to .tflite model")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", type=Path, help="Single cell image path")
    g.add_argument("--manifest", type=Path, help="JSONL manifest (one JSON per line) with 'path' and optional labels")
    ap.add_argument("--root", type=Path, default=Path("."), help="Root added to relative manifest paths")
    ap.add_argument("--img", type=int, default=64, help="Model input size (square)")
    ap.add_argument("--inner-crop", type=float, default=1.0, help="Inner center-crop fraction in (0,1]; 1.0 = no-op")
    ap.add_argument("--cand-thr", type=float, default=0.58, help="Candidate threshold after sigmoid")
    ap.add_argument("--out-dir", type=Path, default=None, help="Output folder for batch results")
    ap.add_argument("--save-json", action="store_true", help="Also save JSONL for batch runs")
    return ap.parse_args()

def main():
    args = parse_args()
    Interpreter, backend = try_load_tflite_interpreter()
    print(f"[info] Using backend: {backend}")
    interpreter = Interpreter(model_path=str(args.tflite))
    print(f"[load] TFLite model: {args.tflite}")

    if args.image:
        x = preprocess_to_nhwc(args.image, args.img, args.inner_crop)
        print(f"[input] shape={x.shape} dtype={x.dtype} min={float(x.min()):.4f} max={float(x.max()):.4f} mean={float(x.mean()):.4f}")
        res = run_single_image(interpreter, x, args.cand_thr)
        res["path"] = str(args.image)
        print("\n[input details] shape:", res["input_shape"])
        print("[output details]:")
        for od in res["output_details"]:
            print(f"  idx={od['index']} name='{od['name']}' shape={od['shape']}")
        print("\n[raw logits]")
        print("  given     :", np.array(res['logits_given']))
        print("  solution  :", np.array(res['logits_solution']))
        print("  candidates:", np.array(res['logits_candidates']))
        print("\n[post-processed]")
        print("  p_given    :", np.array(res['p_given']))
        print("  p_solution :", np.array(res['p_solution']))
        print("  p_candidates:", np.array(res['p_candidates']))
        print("  pred_given    :", res['pred_given'])
        print("  pred_solution :", res['pred_solution'])
        print(f"  cand_mask_bits: {res['cand_mask_bits']}  (on: {res['cand_on']})")
        return

    # Batch via manifest
    rows = []
    n = 0
    with args.manifest.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rel = Path(obj["path"])
            p = rel if rel.is_absolute() else (args.root / rel)
            x = preprocess_to_nhwc(p, args.img, args.inner_crop)
            res = run_single_image(interpreter, x, args.cand_thr)
            res["path"] = str(p)
            if "given_digit" in obj:    res["label_given"] = int(obj["given_digit"])
            if "solution_digit" in obj: res["label_solution"] = int(obj["solution_digit"])
            if "candidates" in obj:     res["label_candidates"] = list(map(int, obj["candidates"]))
            rows.append(res)
            n += 1
            if n % 50 == 0:
                print(f"[progress] processed {n} samples...")

    if args.out_dir is None:
        args.out_dir = Path("runs/inspect_cellnet_tflite")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "per_cell.csv"
    write_csv(csv_path, rows)
    print(f"[ok] Wrote CSV: {csv_path} ({len(rows)} rows)")
    if args.save_json:
        json_path = args.out_dir / "per_cell.jsonl"
        with json_path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"[ok] Wrote JSONL: {json_path}")

if __name__ == "__main__":
    main()
