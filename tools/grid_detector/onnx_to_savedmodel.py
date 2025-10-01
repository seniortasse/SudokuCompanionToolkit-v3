import sys, pathlib, onnx
from onnx_tf.backend import prepare

if len(sys.argv) != 3:
    print("Usage: python tools\\onnx_to_savedmodel.py <in.onnx> <out_dir>")
    sys.exit(1)

in_path = pathlib.Path(sys.argv[1])
out_dir = pathlib.Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

print(f"Loading ONNX: {in_path}")
model = onnx.load(str(in_path))
print("Preparing TF graph...")
tf_rep = prepare(model, device='CPU')  # uses your installed TensorFlow
print(f"Exporting SavedModel → {out_dir}")
tf_rep.export_graph(str(out_dir))
print("Done.")