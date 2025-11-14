import os
import sys
import torch
import onnx
import tensorflow as tf

# If cnn_small.py is alongside this script:
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
from cnn_small import CNN28  # <-- uses your CNN28 definition

# You can also: python tools/digit_classifier/export_digit_fp32.py --ckpt path/to.pt
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="python/vision/train/checkpoints/best.pt")
ap.add_argument("--outdir", default="models")
ap.add_argument("--name", default="digit_cnn")
ap.add_argument("--opset", type=int, default=13)
args = ap.parse_args()

CKPT = args.ckpt
OUTDIR = args.outdir
NAME = args.name
OPSET = args.opset

ONNX_OUT = os.path.join(OUTDIR, f"{NAME}.onnx")
SAVED_MODEL_DIR = os.path.join(OUTDIR, f"{NAME}_saved_model")
TFLITE_OUT = os.path.join(OUTDIR, f"{NAME}_fp32.tflite")

os.makedirs(OUTDIR, exist_ok=True)

# 1) Load PyTorch model
m = CNN28(num_classes=10)
state = torch.load(CKPT, map_location="cpu")

# unwrap common wrappers
if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]
elif isinstance(state, dict) and "model" in state:
    state = state["model"]

# strip "module." prefix if saved with DataParallel
if isinstance(state, dict):
    state = { (k.replace("module.", "")) : v for k, v in state.items() }

missing, unexpected = m.load_state_dict(state, strict=False)
if missing:
    print("[WARN] Missing keys:", missing)
if unexpected:
    print("[WARN] Unexpected keys:", unexpected)

m.eval()

# 2) Export to ONNX (NCHW 1x1x28x28)
dummy = torch.randn(1, 1, 28, 28)
torch.onnx.export(
    m, dummy, ONNX_OUT,
    input_names=["input"], output_names=["logits"],
    opset_version=OPSET, dynamic_axes=None
)
print("Wrote:", ONNX_OUT)

# 3) ONNX -> TF SavedModel
# Prefer onnx-tf via python API; if not installed, give a useful hint.
try:
    from onnx_tf.backend import prepare
except Exception as e:
    print("\n[ERROR] onnx-tf not installed or incompatible.\n"
          "Install a compatible version, e.g.:\n"
          "  pip install onnx onnx-tf\n"
          "and ensure TensorFlow is available.\n")
    raise

onnx_model = onnx.load(ONNX_OUT)
tf_rep = prepare(onnx_model)
if os.path.exists(SAVED_MODEL_DIR):
    # clean out target dir if it exists
    import shutil
    shutil.rmtree(SAVED_MODEL_DIR)
tf_rep.export_graph(SAVED_MODEL_DIR)
print("Wrote SavedModel:", SAVED_MODEL_DIR)

# 4) TF -> TFLite (FP32)
converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
tflite_model = converter.convert()
with open(TFLITE_OUT, "wb") as f:
    f.write(tflite_model)
print("Wrote:", TFLITE_OUT)