"""
U-Net Encoder-Decoder для генерации бас-спектрограммы.
Использует GroupNorm вместо BatchNorm (совместимо с DirectML/AMD GPU).

Вход:  (B, 1, n_mels, T)
Выход: (B, 1, n_mels, T)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def norm(channels):
    """GroupNorm — работает на любом устройстве включая DirectML."""
    groups = min(8, channels)
    while channels % groups != 0:
        groups //= 2
    return nn.GroupNorm(groups, channels)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            norm(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            norm(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


def match_size(x, target):
    if x.shape[2] != target.shape[2] or x.shape[3] != target.shape[3]:
        x = x[:, :, :target.shape[2], :target.shape[3]]
    return x


class AccompanimentModel(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()
        c = base_channels

        self.enc1 = ConvBlock(1,     c)
        self.enc2 = ConvBlock(c,     c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.enc4 = ConvBlock(c * 4, c * 8)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(c * 8, c * 16, 3, padding=1, bias=False),
            norm(c * 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(c * 16, c * 8, 3, padding=1, bias=False),
            norm(c * 8),
            nn.ReLU(inplace=True),
        )

        self.dec4 = DeconvBlock(c * 8  + c * 8, c * 4, dropout=0.1)
        self.dec3 = DeconvBlock(c * 4  + c * 4, c * 2, dropout=0.1)
        self.dec2 = DeconvBlock(c * 2  + c * 2, c)
        self.dec1 = DeconvBlock(c      + c,      1)

        self.out_act = nn.Tanh()

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        b = self.bottleneck(e4)

        b  = match_size(b,  e4)
        d4 = self.dec4(torch.cat([b,  e4], dim=1))
        d4 = match_size(d4, e3)
        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        d3 = match_size(d3, e2)
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d2 = match_size(d2, e1)
        d1 = self.dec1(torch.cat([d2, e1], dim=1))

        d1 = F.interpolate(d1, size=x.shape[2:], mode='bilinear', align_corners=False)
        return self.out_act(d1)


if __name__ == "__main__":
    model = AccompanimentModel(base_channels=32)
    x = torch.randn(2, 1, 128, 87)
    y = model(x)
    print(f"Вход: {x.shape} -> Выход: {y.shape}")
    print(f"Параметров: {sum(p.numel() for p in model.parameters()):,}")
