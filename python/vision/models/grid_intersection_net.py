"""
GridIntersectionNet — lightweight U‑Net for 5‑map grid-line model
Outputs 6 channels in fixed order: [A, H, V, J, Ox, Oy]

Dependencies: torch (and torchvision is NOT required).

Suggested use:
    from python.vision.models.grid_intersection_net import GridIntersectionNet
    model = GridIntersectionNet(in_channels=1, out_channels=6, base_ch=64)
    y = model(torch.randn(2,1,768,768))  # -> (2,6,768,768)

Notes
-----
• First 4 channels (A/H/V/J) are trained with sigmoid + BCE+Dice.
• Last 2 channels (Ox/Oy) are raw vectors; normalize in loss/postproc.
• Designed to be small but accurate; you can scale `base_ch`.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------ Blocks ---------------------------------

class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(in_ch, out_ch, 3, 1, 1),
            ConvBNAct(out_ch, out_ch, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Upsample + skip + DoubleConv. Uses bilinear upsampling to avoid checkerboard artifacts."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if sizes mismatch due to odd dims
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        if diffY != 0 or diffX != 0:
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# --------------------------- Main network ------------------------------

class GridIntersectionNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 6,
                 base_ch: int = 64,
                 dropout_p: float = 0.0):
        """
        Parameters
        ----------
        in_channels : number of input channels (1 for grayscale).
        out_channels: must be 6 → [A,H,V,J,Ox,Oy].
        base_ch     : width multiplier; 64 is a good default. Use 32 for mobile.
        dropout_p   : optional dropout at bottleneck.
        """
        super().__init__()
        assert out_channels == 6, "Output channels must be 6: [A,H,V,J,Ox,Oy]"

        ch1, ch2, ch3, ch4, ch5 = base_ch, base_ch*2, base_ch*4, base_ch*8, base_ch*16

        # Encoder
        self.enc1 = DoubleConv(in_channels, ch1)
        self.down1 = Down(ch1, ch2)
        self.down2 = Down(ch2, ch3)
        self.down3 = Down(ch3, ch4)

        # Bottleneck
        self.pool = nn.MaxPool2d(2)
        self.center = DoubleConv(ch4, ch5)
        self.drop = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()

        # Decoder
        self.up3 = Up(ch5, ch4, ch4)
        self.up2 = Up(ch4, ch3, ch3)
        self.up1 = Up(ch3, ch2, ch2)
        self.up0 = Up(ch2, ch1, ch1)

        # Heads
        self.out = nn.Conv2d(ch1, out_channels, kernel_size=1)

        self._init_weights()

    # ------------------------ forward ------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)          # HxW
        e2 = self.down1(e1)        # H/2 x W/2
        e3 = self.down2(e2)        # H/4
        e4 = self.down3(e3)        # H/8

        # Bottleneck
        c = self.pool(e4)          # H/16
        c = self.center(c)
        c = self.drop(c)

        # Decoder with skips
        d3 = self.up3(c, e4)       # H/8
        d2 = self.up2(d3, e3)      # H/4
        d1 = self.up1(d2, e2)      # H/2
        d0 = self.up0(d1, e1)      # H

        logits = self.out(d0)      # (N,6,H,W)
        return logits

    # -------------------- initialization ---------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ---------------------- utilities ------------------------
    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ------------------------------ Smoke test -----------------------------
if __name__ == "__main__":
    model = GridIntersectionNet(in_channels=1, out_channels=6, base_ch=64)
    x = torch.randn(1,1,512,512)
    y = model(x)
    print("out:", y.shape, "params(M):", model.count_params()/1e6)
