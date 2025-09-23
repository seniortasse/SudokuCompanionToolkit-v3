import argparse, pathlib, tensorflow as tf

ap = argparse.ArgumentParser()
ap.add_argument("--saved_model_dir", required=True)
ap.add_argument("--out_tflite", required=True)
ap.add_argument("--size", type=int, default=640)
args = ap.parse_args()

core = tf.saved_model.load(args.saved_model_dir)

class Wrapped(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec([1, args.size, args.size, 3], tf.float32, name="images")])
    def __call__(self, images):
        x = tf.cast(images, tf.float32)  # assume [0,1]
        return core(x)

w = Wrapped()
concrete = w.__call__.get_concrete_function(tf.TensorSpec([1, args.size, args.size, 3], tf.float32, name="images"))
conv = tf.lite.TFLiteConverter.from_concrete_functions([concrete], w)
# no optimizations
tfl = conv.convert()
path = pathlib.Path(args.out_tflite)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_bytes(tfl)
print("[ok] wrote", path)