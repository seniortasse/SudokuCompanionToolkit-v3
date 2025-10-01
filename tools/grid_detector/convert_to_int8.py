# tools/convert_to_int8.py
import argparse, os, glob, random, sys, traceback
import numpy as np
from PIL import Image
import tensorflow as tf

SIZE = 640

def letterbox_640_rgb(img: Image.Image, fill=(114,114,114)):
    img = img.convert("RGB")
    w, h = img.size
    scale = min(SIZE / w, SIZE / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    if (w, h) != (nw, nh):
        img = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (SIZE, SIZE), fill)
    pad_x = (SIZE - nw) // 2
    pad_y = (SIZE - nh) // 2
    canvas.paste(img, (pad_x, pad_y))
    arr = np.asarray(canvas, dtype=np.uint8)
    return arr

def build_rep_dataset(calib_dir, max_samples):
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.JPG","*.JPEG","*.PNG","*.BMP")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(calib_dir, e)))
    if not files:
        raise SystemExit(f"No images found in {calib_dir}")
    random.shuffle(files)
    files = files[:max_samples]
    def gen():
        for fp in files:
            img_u8 = letterbox_640_rgb(Image.open(fp))
            img_f32 = (img_u8.astype(np.float32) / 255.0)
            yield [np.expand_dims(img_f32, axis=0)]
    return gen

def save_model(tflite_model: bytes, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(tflite_model)

def try_full_int8_io(converter):
    # Full INT8 everything (may fail with NCHW-ish convs)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8
    # If layout/per-channel quant causes trouble on some graphs, disabling per-channel
    # sometimes helps. It’s an undocumented knob but exists on TF 2.12.
    if hasattr(converter, "_experimental_disable_per_channel"):
        converter._experimental_disable_per_channel = False
    return converter.convert()

def try_full_int8_float_io(converter):
    # Keep INT8 ops/weights, but expose float32 I/O (Quantize/Dequantize added)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.float32
    converter.inference_output_type = tf.float32
    if hasattr(converter, "_experimental_disable_per_channel"):
        converter._experimental_disable_per_channel = False
    return converter.convert()

def try_dynamic_range(converter):
    # No representative dataset; int8 weights only, float activations & I/O
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Remove rep dataset if present
    converter.representative_dataset = None
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.inference_input_type  = tf.float32
    converter.inference_output_type = tf.float32
    return converter.convert()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saved-model", required=True)
    ap.add_argument("--calib-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=200)
    args = ap.parse_args()

    sm = args.saved_model
    if not (os.path.isdir(sm) and os.path.exists(os.path.join(sm, "saved_model.pb"))):
        raise SystemExit(f"SavedModel not found at: {sm}")

    print(f"[INFO] Using SavedModel: {sm}")
    print(f"[INFO] Calibration images: {args.calib_dir}")
    print(f"[INFO] Samples: {args.samples}")
    print(f"[INFO] Output: {args.out}")

    # Base converter
    converter = tf.lite.TFLiteConverter.from_saved_model(sm)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = build_rep_dataset(args.calib_dir, args.samples)

    # Try modes in order
    # 1) Full INT8 w/ INT8 I/O
    try:
        print("[TRY ] Full INT8 (INT8 I/O)")
        tflm = try_full_int8_io(converter)
        save_model(tflm, args.out)
        print("[OK  ] Wrote FULL-INT8 model (INT8 I/O) →", args.out)
        return
    except Exception as e:
        print("[FAIL] Full INT8 (INT8 I/O):", e.__class__.__name__, str(e).splitlines()[-1])

    # 2) Full INT8 internal, float32 I/O
    try:
        print("[TRY ] Full INT8 (float32 I/O)")
        tflm = try_full_int8_float_io(converter)
        out2 = os.path.splitext(args.out)[0] + "_floatIO.tflite"
        save_model(tflm, out2)
        print("[OK  ] Wrote FULL-INT8 (float I/O) →", out2)
        print("       This runs nearly as fast as full INT8 on device and avoids NCHW/NHWC issues.")
        return
    except Exception as e:
        print("[FAIL] Full INT8 (float32 I/O):", e.__class__.__name__, str(e).splitlines()[-1])

    # 3) Dynamic range
    try:
        print("[TRY ] Dynamic-range quantization")
        tflm = try_dynamic_range(tf.lite.TFLiteConverter.from_saved_model(sm))
        out3 = os.path.splitext(args.out)[0] + "_dynamic.tflite"
        save_model(tflm, out3)
        print("[OK  ] Wrote dynamic-range model →", out3)
        return
    except Exception as e:
        print("[FAIL] Dynamic-range:", e.__class__.__name__, str(e).splitlines()[-1])
        print("[ABORT] All modes failed.")
        traceback.print_exc()
        sys.exit(2)

if __name__ == "__main__":
    main()