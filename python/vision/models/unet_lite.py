import torch
import torch.nn as nn
import torch.nn.functional as F


class DSConv(nn.Module):
    """
    Depthwise-separable conv: DWConv -> BN -> SiLU -> PWConv -> BN -> SiLU
    """
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None):
        super().__init__()
        if p is None:
            p = k // 2
        self.dw = nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False)
        self.dw_bn = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.act(self.dw_bn(self.dw(x)))
        x = self.act(self.pw_bn(self.pw(x)))
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = nn.Sequential(
            DSConv(in_ch, out_ch),
            DSConv(out_ch, out_ch),
        )

    def forward(self, x):
        x = self.pool(x)
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # in_ch here is channels after concatenation
        self.block = nn.Sequential(
            DSConv(in_ch, out_ch),
            DSConv(out_ch, out_ch),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class UNetLite(nn.Module):
    """
    Tiny UNet-like backbone for 1x128x128 -> 1x128x128 heatmap.
    Base widths: 16 -> 32 -> 48 -> 64 (~0.3–0.5M params).
    """
    def __init__(self, in_ch=1, base=16, out_ch=1):
        super().__init__()
        b = base
        # Encoder
        self.enc1 = nn.Sequential(DSConv(in_ch, b), DSConv(b, b))
        self.enc2 = Down(b, b * 2)
        self.enc3 = Down(b * 2, b * 3)
        self.enc4 = Down(b * 3, b * 4)
        # Bottleneck
        self.mid = nn.Sequential(DSConv(b * 4, b * 4), DSConv(b * 4, b * 4))
        # Decoder (note concatenation increases channels)
        self.up3 = Up(b * 4 + b * 3, b * 3)
        self.up2 = Up(b * 3 + b * 2, b * 2)
        self.up1 = Up(b * 2 + b, b)
        # Head
        self.head = nn.Conv2d(b, out_ch, 1)

    def forward(self, x):
        s1 = self.enc1(x)         # b
        s2 = self.enc2(s1)        # 2b
        s3 = self.enc3(s2)        # 3b
        s4 = self.enc4(s3)        # 4b
        x  = self.mid(s4)         # 4b
        x  = self.up3(x, s3)      # 3b
        x  = self.up2(x, s2)      # 2b
        x  = self.up1(x, s1)      # b
        x  = self.head(x)         # 1
        return x


if __name__ == "__main__":
    m = UNetLite()
    x = torch.randn(1, 1, 128, 128)
    y = m(x)
    print("out:", y.shape)  # (1, 1, 128, 128)