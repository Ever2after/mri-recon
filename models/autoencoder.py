import math
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Weight init (paper-style)
# -------------------------
def init_paper_leakyrelu(module: nn.Module, negative_slope: float = 0.2) -> None:
    """
    Paper: zero-mean Gaussian with std = 1/sqrt(n_l*(1+a^2)), where a is leaky slope.
    (This is LeakyReLU-friendly He-style normal init.)
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
            std = 1.0 / math.sqrt(fan_in * (1.0 + negative_slope**2))
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


# -------------------------
# Building blocks
# -------------------------
class ConvBNLReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, negative_slope: float = 0.2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ConvBN(nn.Module):
    """For the final latent conv in encoder: BN but NO activation (paper: all conv except final use LeakyReLU)."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class EncoderASI(nn.Module):
    """
    Paper encoder:
      - Block1: (Conv 32 + BN + LReLU) x2 -> AvgPool2d(2)
      - Block2: (Conv 64 + BN + LReLU) x2 -> AvgPool2d(2)
      - Then: Conv 128 + BN + LReLU, then final Conv 128 (NO activation), output is latent tensor
    """
    def __init__(self, in_ch: int = 1, negative_slope: float = 0.2):
        super().__init__()
        self.b1_1 = ConvBNLReLU(in_ch, 32, negative_slope)
        self.b1_2 = ConvBNLReLU(32, 32, negative_slope)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.b2_1 = ConvBNLReLU(32, 64, negative_slope)
        self.b2_2 = ConvBNLReLU(64, 64, negative_slope)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.c3_1 = ConvBNLReLU(64, 128, negative_slope)
        # final latent layer: no activation
        self.c3_2 = ConvBN(128, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.b1_1(x)
        x = self.b1_2(x)
        x = self.pool1(x)

        x = self.b2_1(x)
        x = self.b2_2(x)
        x = self.pool2(x)

        x = self.c3_1(x)
        z = self.c3_2(x)
        return z


class DecoderASI(nn.Module):
    """
    Paper decoder (reverse of encoder):
      - Two blocks: (Conv + LReLU) x2 -> BN -> Nearest-neighbor Upsample(2)
      - After each upsample, number of kernels is halved (128 -> 64 -> 32)
      - Then: Conv 32, Conv 1 with sigmoid
    """
    def __init__(self, out_ch: int = 1, negative_slope: float = 0.2):
        super().__init__()
        # latent channels = 128
        self.d1_1 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True)
        self.d1_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(128)

        self.d2_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=True)
        self.d2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(64)

        self.d3_1 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=True)
        self.d3_2 = nn.Conv2d(32, out_ch, kernel_size=3, padding=1, bias=True)

        self.act = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.out_act = nn.Sigmoid()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Block 1 (128 -> 128), then upsample
        x = self.act(self.d1_1(z))
        x = self.act(self.d1_2(x))
        x = self.bn1(x)
        x = self.up(x)

        # Block 2 (128 -> 64), then upsample
        x = self.act(self.d2_1(x))  # halve channels after upsample (done via this conv)
        x = self.act(self.d2_2(x))
        x = self.bn2(x)
        x = self.up(x)

        # Final convs (64 -> 32 -> 1) + sigmoid
        x = self.act(self.d3_1(x))
        x = self.d3_2(x)
        x = self.out_act(x)
        return x


class ASIAutoencoder(nn.Module):
    """
    Full CAE used in the paper.
    Notes:
      - Input H and W should be divisible by 4 during training (two 2x2 pools). Paper also mentions
        test images don't need to match training patch size, but still the pooling/upsample require /4 compatibility
        unless you pad.
    """
    def __init__(self, in_ch: int = 1, out_ch: int = 1, negative_slope: float = 0.2, apply_paper_init: bool = True):
        super().__init__()
        self.encoder = EncoderASI(in_ch=in_ch, negative_slope=negative_slope)
        self.decoder = DecoderASI(out_ch=out_ch, negative_slope=negative_slope)

        if apply_paper_init:
            init_paper_leakyrelu(self, negative_slope=negative_slope)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


# -------------------------
# Latent interpolation helper
# -------------------------
@torch.no_grad()
def latent_convex_mix(z0: torch.Tensor, z1: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    z_alpha = (1-alpha) z0 + alpha z1, alpha in [0,1]
    """
    a = float(alpha)
    return (1.0 - a) * z0 + a * z1
