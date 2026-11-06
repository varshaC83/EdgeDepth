"""EdgeDepthNet: a lightweight RGB-guided depth upsampling / completion CNN.

Inputs (all at the target HR resolution):
  * RGB guide                         (3 ch, ImageNet-normalised)
  * initial depth D0 / mu             (1 ch) bicubic upsampling (low-res input) or
                                       push-pull fill (sparse input) of the observation,
                                       divided by its mean mu (scale normalisation)
  * measurement mask M                (1 ch)
  * scale code log2(s)/3              (1 ch) so one model serves 2x, 4x, 8x and sparse input

Architecture:
  two encoders (RGB, depth) over 4 scales (1, 1/2, 1/4, 1/8)
  -> depth-gated fusion at each scale: RGB features are multiplied by a gate
     computed from depth+RGB features, so RGB edges are only let through
     where the depth branch supports a discontinuity (suppresses texture copying)
  -> U-Net decoder with skip connections
  -> heads: depth residual, boundary logit, 8-neighbour affinities
  -> D_coarse = D0 + residual
  -> learned edge-aware refinement: T steps of convolutional spatial
     propagation (CSPN, Cheng et al. 2018) with the predicted affinities.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1), nn.LeakyReLU(0.1, inplace=True))


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, 1, 1)
        self.c2 = nn.Conv2d(c, c, 3, 1, 1)

    def forward(self, x):
        return x + self.c2(F.leaky_relu(self.c1(x), 0.1))


class GatedFusion(nn.Module):
    """f = fd + Conv([fd, sigmoid(Wg [fd, fr]) * fr])."""

    def __init__(self, c):
        super().__init__()
        self.gate = nn.Conv2d(2 * c, c, 1)
        self.mix = conv(2 * c, c)

    def forward(self, fd, fr):
        g = torch.sigmoid(self.gate(torch.cat([fd, fr], 1)))
        return fd + self.mix(torch.cat([fd, g * fr], 1)), g


class CSPN(nn.Module):
    """Convolutional spatial propagation with 8-neighbour normalised affinities."""

    def __init__(self, iters: int = 6):
        super().__init__()
        self.iters = iters

    def forward(self, d, aff):
        k = torch.tanh(aff)
        k = k / (k.abs().sum(1, keepdim=True) + 1e-6)  # sum |k_i| <= 1 -> stable
        k0 = 1 - k.sum(1, keepdim=True)
        offs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for _ in range(self.iters):
            p = F.pad(d, (1, 1, 1, 1), mode="replicate")
            H, W = d.shape[-2:]
            out = k0 * d
            for i, (dy, dx) in enumerate(offs):
                out = out + k[:, i:i + 1] * p[:, :, 1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
            d = out
        return d


class EdgeDepthNet(nn.Module):
    def __init__(self, base: int = 16, cspn_iters: int = 6):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 6
        self.r1 = nn.Sequential(conv(3, c1), conv(c1, c1))
        self.d1 = nn.Sequential(conv(3, c1), conv(c1, c1))
        self.r2, self.d2 = conv(c1, c2, 2), conv(c1, c2, 2)
        self.r3, self.d3 = conv(c2, c3, 2), conv(c2, c3, 2)
        self.f1, self.f2, self.f3 = GatedFusion(c1), GatedFusion(c2), GatedFusion(c3)
        self.e2 = conv(c2, c2)
        self.e3 = conv(c3, c3)
        self.down4 = conv(c3, c4, 2)
        self.bott = nn.Sequential(ResBlock(c4), ResBlock(c4))
        self.u3 = conv(c4 + c3, c3)
        self.u2 = conv(c3 + c2, c2)
        self.u1 = nn.Sequential(conv(c2 + c1, c1), conv(c1, c1))
        self.head_res = nn.Conv2d(c1, 1, 3, 1, 1)
        self.head_edge = nn.Conv2d(c1, 1, 3, 1, 1)
        self.head_aff = nn.Conv2d(c1, 8, 3, 1, 1)
        for h in (self.head_res, self.head_aff):  # start as identity: D = D0, CSPN = no-op
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)
        self.cspn = CSPN(cspn_iters)

    @staticmethod
    def _up(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, rgb, d0n, mask, scode):
        dep_in = torch.cat([d0n, mask, scode], 1)
        r1, d1 = self.r1(rgb), self.d1(dep_in)
        f1, g1 = self.f1(d1, r1)
        r2, d2 = self.r2(r1), self.d2(f1)
        f2, _ = self.f2(d2, r2)
        f2 = self.e2(f2)
        r3, d3 = self.r3(r2), self.d3(f2)
        f3, _ = self.f3(d3, r3)
        f3 = self.e3(f3)
        b = self.bott(self.down4(f3))
        x = self.u3(torch.cat([self._up(b, f3), f3], 1))
        x = self.u2(torch.cat([self._up(x, f2), f2], 1))
        x = self.u1(torch.cat([self._up(x, f1), f1], 1))
        coarse = d0n + self.head_res(x)
        refined = self.cspn(coarse, self.head_aff(x))
        return {"coarse": coarse, "refined": refined, "edge_logit": self.head_edge(x), "gate1": g1}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def scale_code(eff_scale: float) -> float:
    return math.log2(max(eff_scale, 1.0)) / 3.0
