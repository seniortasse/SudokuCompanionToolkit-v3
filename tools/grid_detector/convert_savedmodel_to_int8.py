# tools/convert_savedmodel_to_int8.py
import argparse, os, glob
from PIL import Image
import numpy as np
import tensorflow as tf

INPUT_SIZE = 640

def letterbox(img: Image.Image, size=INPUT_SIZE):
    w, h = img.size
    scale = min(size / w, size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2

    # Resize and paste onto gray canvas
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new('RGB', (size, size), (114,114,114))
    canvas.paste(img_resized, (pad_x, pad_y))
    arr = np.asarray(canvas).astype(np.float32) / 255.0   # [H,W,3] in [0,1]
    return arr

def rep_data_gen(img_dir, max_samples):
    paths = sorted(glob.glob(os.path.join(img_dir, '*')))
    count = 0
    for p in paths:
        if max_samples and count >= max_samples: break
        try:
            img = Image.open(p).convert('RGB')
            arr = letterbox(img)  # [640,640,3] float32
            arr = np.expand_dims(arr, 0)  # [1,640,640,3]
            yield [arr]
            count += 1
        except Exception:
            continue

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--savedmodel', required=True)
    ap.add_argument('--calib_dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--samples', type=int, default=200)
    ap.add_argument('--int8_io', action='store_true',
                    help='Force INT8 input/output types (otherwise leave runtime-flexible).')
    args = ap.parse_args()

    converter = tf.lite.TFLiteConverter.from_saved_model(args.savedmodel)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: rep_data_gen(args.calib_dir, args.samples)

    # Require full INT8 kernels
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.experimental_new_quantizer = True

    if args.int8_io:
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    # else: leave as float input/output for drop-in compatibility (weights & kernels int8)

    tflite_model = converter.convert()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'wb') as f:
        f.write(tflite_model)
    print(f"Wrote {args.out}")

if __name__ == '__main__':
    main()