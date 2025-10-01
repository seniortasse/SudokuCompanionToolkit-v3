# tools/tflite_convert_int8.py
import argparse, pathlib, random
import numpy as np
import tensorflow as tf
from PIL import Image

def representative_dataset_gen(img_dir, size=640, num_samples=100, want_uint8_io=False):
    p = pathlib.Path(img_dir)
    paths = list(p.glob("*.jpg")) + list(p.glob("*.png")) + list(p.glob("*.jpeg"))
    random.shuffle(paths)
    for img_path in paths[:num_samples]:
        img = Image.open(img_path).convert("RGB").resize((size, size), Image.BILINEAR)
        arr = np.asarray(img)
        if want_uint8_io:
            arr = arr.astype(np.uint8)                 # 0..255 for uint8 I/O
        else:
            arr = (arr.astype(np.float32) / 255.0)     # 0..1 for fp32 I/O
        arr = np.expand_dims(arr, axis=0)              # BHWC
        yield [arr]

class Wrapped(tf.Module):
    def __init__(self, core, want_uint8_io: bool):
        super().__init__()
        self.core = core
        self.want_uint8_io = want_uint8_io

    @tf.function(input_signature=[tf.TensorSpec(shape=[1, None, None, 3],
                                               dtype=tf.uint8, name="images")])
    def __call_uint8__(self, images):
        # uint8 [0..255] -> float32 [0..1]
        x = tf.cast(images, tf.float32) / 255.0
        y = self.core(x)                  # <-- NO 'training=' kwarg
        return y

    @tf.function(input_signature=[tf.TensorSpec(shape=[1, None, None, 3],
                                               dtype=tf.float32, name="images")])
    def __call_fp32__(self, images):
        x = tf.cast(images, tf.float32)   # already [0..1]
        y = self.core(x)                  # <-- NO 'training=' kwarg
        return y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saved_model_dir", required=True)
    ap.add_argument("--out_tflite", required=True)
    ap.add_argument("--rep_data_dir", required=True)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--uint8_io", action="store_true",
                    help="Export full INT8 (uint8 input/output). Default: fp32 I/O.")
    args = ap.parse_args()

    # 1) Load SavedModel (may have no signatures)
    core = tf.saved_model.load(args.saved_model_dir)

    # 2) Wrap & trace to get a ConcreteFunction with fixed spatial dims
    wrapper = Wrapped(core, args.uint8_io)
    if args.uint8_io:
        concrete = wrapper.__call_uint8__.get_concrete_function(
            tf.TensorSpec([1, args.size, args.size, 3], tf.uint8, name="images"))
    else:
        concrete = wrapper.__call_fp32__.get_concrete_function(
            tf.TensorSpec([1, args.size, args.size, 3], tf.float32, name="images"))

    # 3) Converter from the concrete function
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], wrapper)

    # 4) Quantization + calibration
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset_gen(
        args.rep_data_dir, size=args.size, want_uint8_io=args.uint8_io
    )

    if args.uint8_io:
        # Full INT8: quantized ops + uint8 I/O
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    # else: keep fp32 I/O (still INT8 under the hood thanks to Optimize.DEFAULT + rep data)

    # 5) Convert and save
    tflite_model = converter.convert()
    pathlib.Path(args.out_tflite).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_tflite, "wb") as f:
        f.write(tflite_model)
    print(f"[ok] wrote {args.out_tflite}")

if __name__ == "__main__":
    main()