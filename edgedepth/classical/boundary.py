"""Boundary-aware refinement: joint weighted-median snapping at depth discontinuities.

Interpolation and most filters produce "mixed" depths (flying pixels) across
an occlusion boundary: values halfway between foreground and background that
belong to neither surface. A weighted *median* (unlike a weighted mean) always
returns one of the neighbour values, so inside a narrow band around detected
discontinuities each pixel is snapped to the surface whose pixels are most
similar in colour and closest in space. Outside the band the depth is left
untouched, which prevents oversmoothing and texture copying in flat regions.
"""
from __future__ import annotations

import cv2
import numpy as np

from .edges import discontinuity_probability


def boundary_band(depth: np.ndarray, prob: np.ndarray | None = None, thresh: float = 0.5, width: int = 2) -> np.ndarray:
    p = discontinuity_probability(depth) if prob is None else prob
    band = (p > thresh).astype(np.uint8)
    if width > 0:
        band = cv2.dilate(band, np.ones((2 * width + 1, 2 * width + 1), np.uint8))
    return band > 0


def weighted_median_refine(depth: np.ndarray, rgb: np.ndarray, band: np.ndarray, radius: int = 3,
                           sigma_c: float = 0.08, sigma_s: float = 2.0, sigma_d: float | None = None) -> np.ndarray:
    """Replace each band pixel by the joint weighted median of its (2r+1)^2 neighbourhood.

    Weights: w_q = exp(-|I_p-I_q|^2 / 2 sigma_c^2) * exp(-|p-q|^2 / 2 sigma_s^2)
    (optionally * exp(-|D_p-D_q|^2/2 sigma_d^2)). Neighbours that are themselves
    in the band get half weight so that pure-surface pixels dominate the vote.
    """
    H, W = depth.shape
    ys, xs = np.nonzero(band)
    if len(ys) == 0:
        return depth.copy()
    G = rgb.astype(np.float32) / 255.0
    r = radius
    Dp = np.pad(depth, r, mode="edge")
    Gp = np.pad(G, ((r, r), (r, r), (0, 0)), mode="edge")
    Bp = np.pad(band.astype(np.float32), r, mode="edge")
    offs = [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1)]
    vals = np.empty((len(ys), len(offs)), np.float32)
    wts = np.empty((len(ys), len(offs)), np.float32)
    gc = G[ys, xs]
    dc = depth[ys, xs]
    for k, (dy, dx) in enumerate(offs):
        yy, xx = ys + r + dy, xs + r + dx
        v = Dp[yy, xx]
        w = np.exp(-((Gp[yy, xx] - gc) ** 2).sum(-1) / (2 * sigma_c ** 2)) * np.exp(-(dy * dy + dx * dx) / (2 * sigma_s ** 2))
        if sigma_d is not None:
            w *= np.exp(-((v - dc) ** 2) / (2 * sigma_d ** 2))
        w *= 1.0 - 0.5 * Bp[yy, xx]
        vals[:, k] = v
        wts[:, k] = w
    order = np.argsort(vals, axis=1)
    sv = np.take_along_axis(vals, order, 1)
    sw = np.take_along_axis(wts, order, 1)
    cw = np.cumsum(sw, 1)
    idx = (cw < 0.5 * cw[:, -1:]).sum(1)
    out = depth.copy()
    out[ys, xs] = sv[np.arange(len(ys)), np.minimum(idx, sv.shape[1] - 1)]
    return out


def boundary_refine(depth: np.ndarray, rgb: np.ndarray, prob: np.ndarray | None = None, thresh: float = 0.5,
                    width: int = 1, radius: int = 3) -> np.ndarray:
    band = boundary_band(depth, prob, thresh, width)
    return weighted_median_refine(depth, rgb, band, radius=radius)
