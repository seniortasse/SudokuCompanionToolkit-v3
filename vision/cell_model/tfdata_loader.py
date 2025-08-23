# tfdata_loader.py
# Utilities to build tf.data pipelines from the synthetic JSONL manifests.
# Produces inputs: float32 images [B,64,64,1] in [0,1]
# and targets: dict with keys 'type', 'digit', 'notes' as tensors.

import tensorflow as tf
import json

IMG_SIZE = 64

def _parse_jsonl(line):
    ex = tf.io.decode_json_example(line) if False else None
    # We can't rely on decode_json_example; use py_function
    def py_parse(s):
        obj = json.loads(s.numpy().decode("utf-8"))
        return (obj["path"], obj["type_onehot"], obj["digit_onehot"], obj["notes"])
    path, t_type, t_digit, t_notes = tf.py_function(
        func=py_parse, inp=[line], Tout=[tf.string, tf.int32, tf.int32, tf.int32]
    )
    path.set_shape(())
    t_type.set_shape((4,))
    t_digit.set_shape((10,))
    t_notes.set_shape((9,))
    return path, {"type": tf.cast(t_type, tf.float32),
                  "digit": tf.cast(t_digit, tf.float32),
                  "notes": tf.cast(t_notes, tf.float32)}

def _load_image(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE), method="bilinear")
    return img

def _augment(img):
    # Simple, fast augmentations in TF
    img = tf.image.random_brightness(img, max_delta=0.15)
    img = tf.image.random_contrast(img, lower=0.7, upper=1.3)
    # Small rotations +/- 10 deg
    try:
        import tensorflow_addons as tfa
        angle = tf.random.uniform([], minval=-10.0, maxval=10.0) * 3.14159265/180.0
        img = tfa.image.rotate(img, angles=angle, interpolation="BILINEAR", fill_mode="constant", fill_value=1.0)
    except Exception:
        pass
    # Add light Gaussian noise
    noise = tf.random.normal(tf.shape(img), mean=0.0, stddev=0.03)
    img = tf.clip_by_value(img + noise, 0.0, 1.0)
    return img

def make_dataset(jsonl_path, batch_size=64, shuffle=True, augment=False, repeat=False):
    ds = tf.data.TextLineDataset(jsonl_path)
    if shuffle:
        ds = ds.shuffle(4096, reshuffle_each_iteration=True)
    ds = ds.map(_parse_jsonl, num_parallel_calls=tf.data.AUTOTUNE)
    def map_img(path, targets):
        img = _load_image(path)
        if augment:
            img = _augment(img)
        return img, targets
    ds = ds.map(map_img, num_parallel_calls=tf.data.AUTOTUNE)
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
