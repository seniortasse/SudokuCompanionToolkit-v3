import argparse
from pathlib import Path
import sys

import torch
from python.vision.models.unet_lite import UNetLite

def require(pkgs):
    missing = []
    for p in pkgs:
        try:
            __import__(p)
        except Exception:
            missing.append(p)
    if missing:
        print(
            "\n[export] Missing deps: " + ", ".join(missing) +
            "\nInstall e.g.: pip install onnx onnx-tf 'tensorflow-cpu==2.14.*'\n",
            file=sys.stderr
        )
        sys.exit(2)

def representative_gen(calib_dir, img_size):
    """Yield NHWC float32 batches for INT8 PTQ."""
    import cv2, numpy as np
    p = Path(calib_dir)
    imgs = sorted([*p.rglob("*.png"), *p.rglob("*.jpg"), *p.rglob("*.jpeg")])[:500]
    for ip in imgs:
        im = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        if im is None: 
            continue
        im = cv2.resize(im, (img_size, img_size), interpolation=cv2.INTER_AREA)
        x = (im.astype("float32") / 255.0)[None, :, :, None]  # NHWC1
        yield [x]

def main():
    ap = argparse.ArgumentParser(description="Export intersections UNetLite to TFLite (FP16 or INT8).")
    ap.add_argument("--ckpt", required=True, help="Path to PyTorch checkpoint (.pt)")
    ap.add_argument("--out",  required=True, help="Output folder (models/)")
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--fp16", action="store_true", help="Export FP16 TFLite")
    ap.add_argument("--int8", action="store_true", help="Export INT8 TFLite (post-training quant)")
    ap.add_argument("--calib-dir", default=None, help="Images for INT8 calibration (required if --int8)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load PyTorch model
    model = UNetLite(in_ch=1, out_ch=1, base=args.base)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()

    # 2) Export ONNX
    dummy = torch.randn(1, 1, args.img_size, args.img_size)
    onnx_path = out_dir / "intersections.onnx"
    torch.onnx.export(
        model, dummy, onnx_path.as_posix(),
        input_names=["img"], output_names=["logits"],
        dynamic_axes=None, opset_version=args.opset
    )
    print(f"[export] Saved ONNX: {onnx_path}")

    # 3) Convert ONNX -> TF SavedModel (via onnx-tf)
    require(["onnx", "onnx_tf", "tensorflow"])
    import onnx, onnx_tf.backend as backend
    model_onnx = onnx.load(onnx_path.as_posix())
    tf_rep = backend.prepare(model_onnx)              # converts NCHW graph to NHWC ops
    saved_model_dir = out_dir / "intersections_tf"
    tf_rep.export_graph(saved_model_dir.as_posix())
    print(f"[export] Saved TF SavedModel: {saved_model_dir}")

    # 4) TF SavedModel -> TFLite
    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir.as_posix())
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_name = None
    if args.fp16:
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
        tflite_name = "intersections_fp16.tflite"

    if args.int8:
        if args.calib_dir is None:
            print("[export] --int8 requires --calib-dir with sample images.", file=sys.stderr)
            sys.exit(2)
        converter.representative_dataset = lambda: representative_gen(args.calib_dir, args.img_size)
        converter.inference_input_type  = tf.uint8
        converter.inference_output_type = tf.uint8
        tflite_model = converter.convert()
        tflite_name = "intersections_int8.tflite"

    if not (args.fp16 or args.int8):
        # default: plain FP32 (kept small enough) — rarely needed
        tflite_model = converter.convert()
        tflite_name = "intersections_fp32.tflite"

    tflite_path = out_dir / tflite_name
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"[export] Saved TFLite: {tflite_path}")

if __name__ == "__main__":
    main()