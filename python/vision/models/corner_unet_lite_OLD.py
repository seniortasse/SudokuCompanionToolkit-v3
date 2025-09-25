"""
Corner Heatmaps — Mobile U-Net (Lite) for Sudoku Grid Corners

This module defines a tiny U-Net-like model that predicts 4 corner heatmaps
(TL, TR, BL, BR) from a grayscale input. It also provides:
  • softargmax_2d: differentiable coordinate readout (B,4,2) in pixel space
  • CornerLoss   : L = MSE(heatmaps) + λ * L1(coords) with temperature τ

Notes
-----
• Depthwise separable convs keep params/MACs low (mobile-friendly).
• Model is resolution-agnostic; it’s typically trained at 128×128.
• softargmax_2d is used for training/analysis; on-device you can argmax peaks.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CornerUNetLite", "softargmax_2d", "CornerLoss"]


# ──────────────────────────────────────────────────────────────────────────────
# 1) BUILDING BLOCKS
# ──────────────────────────────────────────────────────────────────────────────
def dw_conv(c_in: int, c_out: int, stride: int = 1) -> nn.Sequential:
    """
    Depthwise separable conv block:
      depthwise 3×3 (groups=c_in) → BN → ReLU6 → pointwise 1×1 → BN → ReLU6
    """
    return nn.Sequential(
        nn.Conv2d(c_in, c_in, 3, stride, 1, groups=c_in, bias=False),
        nn.BatchNorm2d(c_in), nn.ReLU6(inplace=True),
        nn.Conv2d(c_in, c_out, 1, 1, 0, bias=False),
        nn.BatchNorm2d(c_out), nn.ReLU6(inplace=True),
    )


# ──────────────────────────────────────────────────────────────────────────────
# 2) MODEL — CornerUNetLite
# ──────────────────────────────────────────────────────────────────────────────
class CornerUNetLite(nn.Module):
    """
    Mobile-friendly U-Net-like model for 4 corner heatmaps.

    Input  : (B, 1, H, W)  e.g., H=W=128
    Output : (B, 4, H, W)  TL, TR, BL, BR heatmaps
    """
    def __init__(self, in_ch: int = 1, base: int = 24, out_ch: int = 4):
        super().__init__()
        # Encoder
        self.e1 = dw_conv(in_ch,      base,   stride=1)  # 128×
        self.e2 = dw_conv(base,       base*2, stride=2)  # 64×
        self.e3 = dw_conv(base*2,     base*4, stride=2)  # 32×
        self.e4 = dw_conv(base*4,     base*4, stride=2)  # 16×

        # Decoder (upsample → concat skip → depthwise block)
        self.d3 = dw_conv(base*4 + base*4, base*4)       # 32×
        self.d2 = dw_conv(base*4 + base*2, base*2)       # 64×
        self.d1 = dw_conv(base*2 + base,   base)         # 128×

        # 1×1 conv head to map to 4 heatmaps
        self.head = nn.Conv2d(base, out_ch, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.e1(x)                              # 128×
        e2 = self.e2(e1)                             # 64×
        e3 = self.e3(e2)                             # 32×
        e4 = self.e4(e3)                             # 16×

        # Decoder
        u3 = F.interpolate(e4, scale_factor=2, mode="bilinear", align_corners=False)
        d3 = self.d3(torch.cat([u3, e3], dim=1))     # 32×

        u2 = F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=False)
        d2 = self.d2(torch.cat([u2, e2], dim=1))     # 64×

        u1 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False)
        d1 = self.d1(torch.cat([u1, e1], dim=1))     # 128×

        return self.head(d1)                         # (B,4,H,W)


# ──────────────────────────────────────────────────────────────────────────────
# 3) UTILITIES — Softargmax
# ──────────────────────────────────────────────────────────────────────────────
def softargmax_2d(heatmaps: torch.Tensor, eps: float = 1e-6, tau: float = 1.0):
    """
    Differentiable argmax over a 2D heatmap with optional temperature.

    Args:
        heatmaps: (B,4,H,W) unnormalized logits (not probabilities)
        eps:      numerical stability
        tau:      temperature (>0). Smaller -> sharper distribution; larger -> smoother.

    Returns:
        (B,4,2) coordinates in (x,y) pixels, 0..W-1 / 0..H-1
    """
    if tau <= 0:
        raise ValueError("softargmax_2d: tau must be > 0")

    B, C, H, W = heatmaps.shape
    logits = heatmaps.view(B * C, H * W) / tau
    prob = torch.softmax(logits, dim=1) + eps        # (B*C, H*W)

    # coordinate grids
    ys = torch.arange(H, device=heatmaps.device, dtype=prob.dtype)
    xs = torch.arange(W, device=heatmaps.device, dtype=prob.dtype)
    ys = ys.view(H, 1).expand(H, W).reshape(-1)      # (H*W)
    xs = xs.view(1, W).expand(H, W).reshape(-1)      # (H*W)

    y = (prob * ys).sum(dim=1)
    x = (prob * xs).sum(dim=1)
    return torch.stack([x, y], dim=1).view(B, C, 2)


# ──────────────────────────────────────────────────────────────────────────────
# 4) LOSS — Heatmap regression + coordinate penalty (with τ)
# ──────────────────────────────────────────────────────────────────────────────
# corner_unet_lite.py

class CornerLoss(nn.Module):
    def __init__(self, lambda_coord: float = 0.1, softargmax_tau: float = 1.0,
                 pos_weight: float = 25.0):  # <— NEW: tune 10–50
        super().__init__()
        self.lambda_coord = float(lambda_coord)
        self.softargmax_tau = float(softargmax_tau)
        self.pos_weight = float(pos_weight)
        self.mse = nn.MSELoss(reduction="mean")               # keep available
        self._bce = None                                      # lazy init

    def forward(self, pred_heat: torch.Tensor, gt_heat: torch.Tensor, gt_xy_px: torch.Tensor):
        # 1) Heatmap loss — use BCE-with-logits + pos_weight to fight class imbalance
        if self._bce is None or self._bce.pos_weight.device != pred_heat.device:
            self._bce = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(self.pos_weight, device=pred_heat.device)
            )
        lh = self._bce(pred_heat, gt_heat)                    # << instead of MSE

        # 2) Coordinate penalty via τ-scaled softargmax
        pred_xy = softargmax_2d(pred_heat, tau=self.softargmax_tau)  # (B,4,2)
        lc = F.l1_loss(pred_xy, gt_xy_px, reduction="mean")

        return lh + self.lambda_coord * lc, {"L_heat": lh.detach(), "L_coord": lc.detach()}