import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn

THIS = Path(__file__).resolve()
ROOT = THIS.parents[2]  # repo root (…/python/vision)
sys.path.append(str(ROOT))

from python.vision.models.unet_lite import UNetLite  # adapt if your package name differs


def to_onnx(model, onnx_path: Path, img_size=128):
    model.eval()
    dummy = torch.randn(1, 1, img_size, img_size)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=None,
        opset_version=13,
        do_constant_folding=True
    )
    print(f"[OK] ONNX saved to {onnx_path}")


def try_tflite_from_onnx(onnx_path: Path, saved_model_dir: Path, tflite_path_fp32: Path,
                         tflite_path_int8: Path = None, calib_dir: Path = None):
    """
    Optional path: requires onnx-tf + tensorflow installed.
    - Converts ONNX -> TF SavedModel -> TFLite
    - INT8 PTQ requires calib images (grayscale 128x128 PNGs recommended)
    """
    try:
        from onnx_tf.backend import prepare as onnx_to_tf
        import onnx
        import tensorflow as tf
    except Exception as e:
        print("[WARN] TensorFlow or onnx-tf not installed. Skipping TFLite export.")
        print("       Install: pip install tensorflow onnx onnx-tf")
        return

    # ONNX -> TF
    saved_model_dir = Path(saved_model_dir)
    saved_model_dir.mkdir(parents=True, exist_ok=True)
    model_onnx = onnx.load(str(onnx_path))
    tf_rep = onnx_to_tf(model_onnx, strict=False)
    tf_rep.export_graph(str(saved_model_dir))
    print(f"[OK] TF SavedModel at {saved_model_dir}")

    # SavedModel -> TFLite FP32
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    tflite_model = converter.convert()
    tflite_path_fp32.parent.mkdir(parents=True, exist_ok=True)
    tflite_path_fp32.write_bytes(tflite_model)
    print(f"[OK] TFLite FP32 at {tflite_path_fp32}")

    # SavedModel -> TFLite INT8 (PTQ)
    if tflite_path_int8 is not None and calib_dir is not None and calib_dir.exists():
        def rep_ds():
            imgs = sorted(list(Path(calib_dir).glob("*.png")))
            for p in imgs[:512]:
                import cv2
                im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if im is None:
                    continue
                im = (im.astype(np.float32) / 255.0)[None, :, :, None]  # NHWC
                yield [im]

        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = rep_ds
        # Use float input/output for simplicity; runtime dequant is handled internally
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                                               tf.lite.OpsSet.TFLITE_BUILTINS]
        try:
            tflite_int8 = converter.convert()
            tflite_path_int8.parent.mkdir(parents=True, exist_ok=True)
            tflite_path_int8.write_bytes(tflite_int8)
            print(f"[OK] TFLite INT8 at {tflite_path_int8}")
        except Exception as e:
            print(f"[WARN] INT8 conversion failed: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to .pt checkpoint (dict with key 'model')")
    ap.add_argument("--onnx", required=True, help="Output ONNX path")
    ap.add_argument("--img-size", type=int, default=128)

    # Optional TF/TFLite
    ap.add_argument("--saved-model", default=None, help="Output TF SavedModel dir")
    ap.add_argument("--tflite-fp32", default=None, help="Output TFLite float32 path")
    ap.add_argument("--tflite-int8", default=None, help="Output TFLite int8 path")
    ap.add_argument("--calib-dir", default=None, help="Dir with 128x128 grayscale PNGs for PTQ")
    args = ap.parse_args()

    ckpt = torch.load(args.model, map_location="cpu")
    sd = ckpt["model"] if "model" in ckpt else ckpt
    m = UNetLite(in_ch=1, base=16, out_ch=1)
    m.load_state_dict(sd)
    m.eval()

    onnx_path = Path(args.onnx)
    to_onnx(m, onnx_path, img_size=args.img_size)

    # Optional TFLite path
    if args.saved_model and args.tflite-fp32:
        saved_model_dir = Path(args.saved_model)
        tflite_fp32 = Path(args.tflite-fp32)
        tflite_int8 = Path(args.tflite-int8) if args.tflite_int8 else None
        calib_dir = Path(args.calib_dir) if args.calib_dir else None
        try_tflite_from_onnx(onnx_path, saved_model_dir, tflite_fp32, tflite_int8, calib_dir)


if __name__ == "__main__":
    main()