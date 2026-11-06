"""Depth noise estimation and edge-preserving noise removal."""
from __future__ import annotations

import cv2
import numpy as np

from .holes import push_pull_fill


def estimate_noise_sigma(values: np.ndarray, mask: np.ndarray) -> float:
    """Robust estimate of additive depth noise std (metres) — Immerkaer (1996).

    The 3x3 kernel [[1,-2,1],[-2,4,-2],[1,-2,1]] cancels planes (and slanted
    planes), leaving mostly noise; sigma = sqrt(pi/2)/6 * mean|response|.
    Only the 50 % smoothest valid pixels are used so depth edges do not inflate
    the estimate. At coarse low-resolution grids (8x) scene structure still
    leaks into the response, which is why the decision threshold is scale
    dependent (see config.NOISE_REL_THRESH).
    """
    if mask.sum() < 25 or min(values.shape) < 5:
        return 0.0
    f = push_pull_fill(values, mask).astype(np.float32)
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    resp = cv2.filter2D(f, -1, k, borderType=cv2.BORDER_REFLECT)
    gy, gx = np.gradient(f)
    g = np.hypot(gx, gy) / np.maximum(f, 1e-3)
    sel = mask > 0
    sel[[0, -1], :] = False
    sel[:, [0, -1]] = False
    if sel.sum() < 25:
        return 0.0
    sel &= g <= np.percentile(g[sel], 50)
    return float(np.sqrt(np.pi / 2) / 6.0 * np.mean(np.abs(resp[sel])))


def denoise_depth(values: np.ndarray, mask: np.ndarray, sigma: float | None = None,
                  median_ksize: int = 3) -> np.ndarray:
    """Remove noise from a depth map while keeping discontinuities.

    1. median filter (removes outliers / salt-and-pepper, keeps step edges),
    2. bilateral filter whose range sigma is tied to the estimated noise level,
       so depth steps much larger than the noise are not smoothed.
    Missing pixels stay missing (holes are handled separately).
    """
    if sigma is None:
        sigma = estimate_noise_sigma(values, mask)
    filled = push_pull_fill(values, mask).astype(np.float32)
    out = cv2.medianBlur(filled, median_ksize)
    if sigma > 1e-4:
        out = cv2.bilateralFilter(out, d=5, sigmaColor=float(3.0 * sigma), sigmaSpace=1.5)
    return np.where(mask > 0, out, 0).astype(np.float32)
