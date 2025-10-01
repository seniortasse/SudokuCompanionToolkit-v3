import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


class TinyCornerNet(nn.Module):
    """
    TinyCornerNet: 4 heatmap logits at input resolution.
    - enc1..enc4 (stride=2) then either unet_head (up4..up1 with skips) or plain up stack.
    - Training added XY coords by concatenation to input before enc1.
    - For TFLite BUILTINS, we support coords-as-input to avoid tf.Range in graph.
    """
    def __init__(self, in_ch_gray: int = 1, out_ch: int = 4, base: int = 24,
                 use_coordconv: bool = True, unet_head: bool = True,
                 coords_as_input: bool = True):
        super().__init__()
        self.unet_head = unet_head
        self.coords_as_input = coords_as_input
        # Effective input channels:
        # - If coords are supplied as input, expect [gray,x,y] => 3
        # - Else, we'll create x,y inside forward and concat => 1 (+2 inside)
        eff_in = (in_ch_gray + 2) if coords_as_input else (in_ch_gray + (2 if use_coordconv else 0))

        c1, c2, c3, c4 = base, base * 2, base * 3, base * 4

        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(eff_in, c1, 3, 2, 1), nn.BatchNorm2d(c1), nn.ReLU(inplace=True))  # 128->64
        self.enc2 = nn.Sequential(nn.Conv2d(c1,     c2, 3, 2, 1), nn.BatchNorm2d(c2), nn.ReLU(inplace=True))  # 64->32
        self.enc3 = nn.Sequential(nn.Conv2d(c2,     c3, 3, 2, 1), nn.BatchNorm2d(c3), nn.ReLU(inplace=True))  # 32->16
        self.enc4 = nn.Sequential(nn.Conv2d(c3,     c4, 3, 2, 1), nn.BatchNorm2d(c4), nn.ReLU(inplace=True))  # 16->8

        if unet_head:
            self.up4 = nn.Sequential(nn.ConvTranspose2d(c4,   c3, 2, 2), nn.ReLU(inplace=True))       # 8->16
            self.up3 = nn.Sequential(nn.ConvTranspose2d(c3*2, c2, 2, 2), nn.ReLU(inplace=True))       # 16->32
            self.up2 = nn.Sequential(nn.ConvTranspose2d(c2*2, c1, 2, 2), nn.ReLU(inplace=True))       # 32->64
            self.up1 = nn.Sequential(nn.ConvTranspose2d(c1*2, c1, 2, 2), nn.ReLU(inplace=True))       # 64->128
            self.head = nn.Conv2d(c1, out_ch, 1)
        else:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(c4, c3, 2, 2), nn.ReLU(inplace=True),
                nn.ConvTranspose2d(c3, c2, 2, 2), nn.ReLU(inplace=True),
                nn.ConvTranspose2d(c2, c1, 2, 2), nn.ReLU(inplace=True),
                nn.ConvTranspose2d(c1, c1, 2, 2), nn.ReLU(inplace=True),
            )
            self.head = nn.Conv2d(c1, out_ch, 1)

        self.use_coordconv = use_coordconv  # kept for meta logging only

    @staticmethod
    def _coord_channels(B: int, H: int, W: int, device):
        xs = torch.linspace(0.0, 1.0, W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)
        ys = torch.linspace(0.0, 1.0, H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
        return xs, ys

    def forward(self, x):  # NCHW
        # coords_as_input=True: expect x: [N,3,H,W] = [gray, xs, ys]
        # coords_as_input=False: x: [N,1,H,W], we will concat xs,ys here
        if not self.coords_as_input:
            B, _, H, W = x.shape
            xs, ys = self._coord_channels(B, H, W, x.device)
            x = torch.cat([x, xs, ys], dim=1)  # [N,3,H,W]

        e1 = self.enc1(x)   # 64
        e2 = self.enc2(e1)  # 32
        e3 = self.enc3(e2)  # 16
        e4 = self.enc4(e3)  # 8

        if self.unet_head:
            u = self.up4(e4)                  # 16
            u = torch.cat([u, e3], dim=1)
            u = self.up3(u)                   # 32
            u = torch.cat([u, e2], dim=1)
            u = self.up2(u)                   # 64
            u = torch.cat([u, e1], dim=1)
            u = self.up1(u)                   # 128
        else:
            u = self.up(e4)                   # 128

        y = self.head(u)                      # [N,4,128,128]
        return y


class WrapperNHWC(nn.Module):
    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core
    def forward(self, x_nhwc):
        x = x_nhwc.permute(0, 3, 1, 2)
        y = self.core(x)
        y = y.permute(0, 2, 3, 1)
        return y


def load_ckpt(ckpt_path: Path) -> Dict[str, Any]:
    try:
        sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except TypeError:
        sd = torch.load(str(ckpt_path), map_location="cpu")
    return sd

def build_from_sd(sd: Dict[str, Any], coords_as_input: bool) -> Tuple[nn.Module, int, int, bool]:
    meta = sd.get("meta", {}) if isinstance(sd, dict) else {}
    img_size = int(meta.get("img_size", 128))
    base     = int(meta.get("base", 24))
    use_cc   = bool(meta.get("use_coordconv", True))
    unet     = bool(meta.get("unet_head", True))

    state = sd.get("model", sd)
    model = TinyCornerNet(in_ch_gray=1, out_ch=4, base=base,
                          use_coordconv=use_cc, unet_head=unet,
                          coords_as_input=coords_as_input)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {len(missing)} (first 12): {missing[:12]}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {len(unexpected)} (first 12): {unexpected[:12]}")
    model.eval()
    return model, img_size, base, unet

def export_onnx_nhwc(model: nn.Module, out_path: Path, img_size: int, opset: int, ch: int):
    model.eval()
    wrapped = WrapperNHWC(model).eval()
    dummy = torch.randn(1, img_size, img_size, ch, dtype=torch.float32)
    with torch.no_grad():
        y = wrapped(dummy)
        assert tuple(y.shape) == (1, img_size, img_size, 4), f"Unexpected out shape: {tuple(y.shape)}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped, (dummy,), str(out_path),
        input_names=["input_nhwc"], output_names=["logits_nhwc"],
        opset_version=opset,
        dynamic_axes={"input_nhwc": {0: "batch"}, "logits_nhwc": {0: "batch"}},
        do_constant_folding=True,
    )
    print(f"[OK] ONNX saved → {out_path.resolve()}")

def parse_args():
    p = argparse.ArgumentParser("Export Sudoku corner model (TinyCornerNet) to NHWC ONNX")
    p.add_argument("--ckpt", type=Path, default=Path("runs/level_6_unet_ft9/best.pt"))
    p.add_argument("--out",  type=Path, default=Path("models/corner_heatmaps_nhwc.onnx"))
    p.add_argument("--opset", type=int, default=11)  # <= 11 for onnx-tf on TF 2.10.x
    p.add_argument("--coords-as-input", action="store_true", default=True,
                   help="Expect [gray,x,y] as input channels; removes tf.Range from graph.")
    return p.parse_args()

def main():
    args = parse_args()
    print(f"[INFO] Loading checkpoint: {args.ckpt}")
    sd = load_ckpt(args.ckpt)
    model, img_size, base, unet = build_from_sd(sd, coords_as_input=args.coords_as_input)
    ch = 3 if args.coords_as_input else 1
    print(f"[INFO] Rebuilt TinyCornerNet: img_size={img_size} base={base} unet_head={unet} coords_as_input={args.coords_as_input}")

    with torch.no_grad():
        nx = torch.randn(1, ch, img_size, img_size)
        ny = model(nx)
        print(f"[INFO] NCHW forward shape: {tuple(ny.shape)} (expect [1,4,{img_size},{img_size}])")

    export_onnx_nhwc(model, args.out, img_size=img_size, opset=args.opset, ch=ch)
    print("[DONE] Export complete. Continue ONNX→TF→TFLite.")

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    sys.exit(main())