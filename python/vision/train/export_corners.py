# --- make <repo-root>/python importable so "vision.*" works ---
import sys
from pathlib import Path
_THIS_FILE = Path(__file__).resolve()
# parents: [0]=.../train, [1]=.../vision, [2]=.../python
_PY_ROOT = _THIS_FILE.parents[2]  # <-- this is .../<repo-root>/python
if str(_PY_ROOT) not in sys.path:
    sys.path.insert(0, str(_PY_ROOT))
# --------------------------------------------------------------

import argparse
from pathlib import Path
import torch
from vision.models.corner_unet_lite import CornerUNetLite

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="runs/corners/best.pt")
    ap.add_argument("--outdir", default="exports", help="output folder")
    ap.add_argument("--img", type=int, default=128)
    ap.add_argument("--base", type=int, default=24)
    args = ap.parse_args()

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    # Load
    model = CornerUNetLite(in_ch=1, base=args.base).eval()
    ck = torch.load(args.ckpt, map_location="cpu")
    sd = ck.get("state_dict", ck)
    model.load_state_dict(sd, strict=False)

    # TorchScript (recommended for Android via PyTorch Mobile)
    ex = torch.randn(1,1,args.img,args.img)
    script = torch.jit.trace(model, ex)
    ts_path = out/"corner_unet_lite.ptl"
    script.save(str(ts_path))
    print(f"Saved TorchScript: {ts_path}")

    # Optional: ONNX export (can be converted to TFLite via TF if desired)
    try:
        import torch.onnx
        onnx_path = out/"corner_unet_lite.onnx"
        torch.onnx.export(model, ex, str(onnx_path), input_names=["input"], output_names=["heatmaps"],
                          opset_version=13, dynamic_axes={"input":{0:"B"}, "heatmaps":{0:"B"}})
        print(f"Saved ONNX: {onnx_path}")
    except Exception as e:
        print(f"ONNX export skipped ({e})")

if __name__ == "__main__":
    main()
