# convert_to_tflite.py
# Convert the trained Keras model to fully quantized INT8 TFLite using a representative dataset.
# Usage:
#   python convert_to_tflite.py --keras checkpoints/sudoku_cell_model.keras \
#       --manifests synthetic_cells/train_manifest.jsonl synthetic_cells/val_manifest.jsonl \
#       --out checkpoints/sudoku_cell_model_int8.tflite

import argparse
import json
import os

import tensorflow as tf


def iter_image_paths(manifests, max_samples=1000):
    seen = 0
    for m in manifests:
        with open(m, encoding="utf-8") as f:
            for line in f:
                if seen >= max_samples:
                    return
                obj = json.loads(line)
                yield obj["path"]
                seen += 1


def representative_dataset_gen(manifests, max_samples=1000):
    for p in iter_image_paths(manifests, max_samples):
        raw = tf.io.read_file(p)
        img = tf.io.decode_png(raw, channels=1)
        img = tf.image.resize(img, (64, 64))
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.expand_dims(img, 0)  # [1,64,64,1]
        yield [img]


def main(args):
    model = tf.keras.models.load_model(args.keras, compile=False)
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: representative_dataset_gen(
        args.manifests, args.max_samples
    )
    # Full-integer quantization
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.uint8
    conv.inference_output_type = tf.uint8
    tfl = conv.convert()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "wb").write(tfl)
    print("Wrote", args.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--keras", required=True)
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_samples", type=int, default=1000)
    args = ap.parse_args()
    main(args)
