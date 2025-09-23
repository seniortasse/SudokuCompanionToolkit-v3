# vision/train/export_torchscript.py
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
import torch.nn as nn

# Make sure the repo root is on sys.path (works when run with -m or plain python)
_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from vision.models.cnn_small import CNN28  # 28x28 grayscale classifier

class WrappedCNN(nn.Module):
    """
    Thin wrapper so the scripted module has a clean forward signature:
      input:  float32 tensor [N, 1, H, W] (H=W=28 by default)
      output: logits [N, 10]  (or probabilities if --softmax is used)
    """
    def __init__(self, core: nn.Module, softmax: bool = False, temperature: float = 1.0):
        super().__init__()
        self.core = core
        self.softmax = softmax
        self.temperature = float(temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N,1,28,28], dtype float32 normalized like training
        logits = self.core(x)
        if self.temperature != 1.0:
            logits = logits / self.temperature
        if self.softmax:
            return torch.softmax(logits, dim=1)
        return logits

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to checkpoint (.pt) with state_dict")
    ap.add_argument("--out", required=True, help="Output TorchScript file (.ptl)")
    ap.add_argument("--img", type=int, default=28, help="Input side (default 28)")
    ap.add_argument("--softmax", action="store_true", help="Export to return probabilities instead of logits")
    ap.add_argument("--temperature", type=float, default=1.0, help="Optional calibration temperature")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    device = args.device
    model = CNN28().to(device)
    ck = torch.load(args.model, map_location="cpu")
    sd = ck.get("state_dict", ck)
    model.load_state_dict(sd, strict=False)
    model.eval()

    wrapped = WrappedCNN(model, softmax=args.softmax, temperature=args.temperature).to(device)
    wrapped.eval()

    # Dynamic batch tracing: example input with batch=4 (batch dimension will stay dynamic)
    ex = torch.randn(4, 1, args.img, args.img, dtype=torch.float32, device=device)

    # Use scripting (safer for control flow); falls back to trace if needed
    try:
        ts = torch.jit.script(wrapped)
    except Exception:
        ts = torch.jit.trace(wrapped, ex)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ts.save(str(out_path))

    # Quick sanity run in-Python
    with torch.no_grad():
        y = ts(ex.cpu())
    print(f"Exported TorchScript to {out_path}")
    print(f"Sanity forward OK: input {tuple(ex.shape)} -> output {tuple(y.shape)}")

if __name__ == "__main__":
    main()