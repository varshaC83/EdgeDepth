"""Missing-depth detection and hole filling."""
from __future__ import annotations

import cv2
import numpy as np


def detect_missing(depth: np.ndarray, min_depth: float = 0.1, max_depth: float = 20.0) -> np.ndarray:
    """Return a boolean map of *missing / invalid* depth values.

    A value is missing if it is 0, NaN/inf, or outside the plausible sensor
    range [min_depth, max_depth] metres.
    """
    d = np.asarray(depth, dtype=np.float32)
    bad = ~np.isfinite(d) | (d < min_depth) | (d > max_depth)
    return bad


def push_pull_fill(values: np.ndarray, mask: np.ndarray, max_levels: int = 12) -> np.ndarray:
    """Fill missing values with multi-scale normalised convolution ("push-pull").

    Push: build a pyramid where each coarser pixel is the mask-weighted mean of
    its 2x2 children. Pull: walk back up and replace missing fine pixels with
    the bilinearly upsampled coarser estimate. Every hole gets a smooth,
    boundary-consistent value; observed pixels are left untouched.
    """
    v = np.where(mask > 0, values, 0).astype(np.float32)
    m = (mask > 0).astype(np.float32)
    if m.all():
        return v.copy()
    if not m.any():
        return v.copy()
    pyr = [(v, m)]
    for _ in range(max_levels):
        v, m = pyr[-1]
        if min(v.shape) <= 1 or m.min() > 0:
            break
        h, w = v.shape
        ph, pw = h % 2, w % 2
        vp = np.pad(v * m, ((0, ph), (0, pw)))
        mp = np.pad(m, ((0, ph), (0, pw)))
        vs = vp.reshape(vp.shape[0] // 2, 2, vp.shape[1] // 2, 2).sum(axis=(1, 3))
        ms = mp.reshape(mp.shape[0] // 2, 2, mp.shape[1] // 2, 2).sum(axis=(1, 3))
        pyr.append((np.where(ms > 0, vs / np.maximum(ms, 1e-6), 0).astype(np.float32), np.minimum(ms, 1.0)))
    filled = pyr[-1][0]
    if pyr[-1][1].min() == 0:  # pathological: fill remaining with mean
        filled = np.where(pyr[-1][1] > 0, filled, filled[pyr[-1][1] > 0].mean() if pyr[-1][1].any() else 0)
    for v, m in reversed(pyr[:-1]):
        up = cv2.resize(filled, (v.shape[1], v.shape[0]), interpolation=cv2.INTER_LINEAR)
        filled = m * v + (1 - m) * up
    return filled.astype(np.float32)


def fill_holes(values: np.ndarray, mask: np.ndarray, method: str = "push_pull") -> np.ndarray:
    """Fill holes with one of: ``push_pull`` (default), ``telea`` (OpenCV inpainting), ``nearest``."""
    if method == "push_pull":
        return push_pull_fill(values, mask)
    if method == "nearest":
        from scipy.ndimage import distance_transform_edt

        if mask.all():
            return values.astype(np.float32).copy()
        _, (iy, ix) = distance_transform_edt(mask == 0, return_indices=True)
        return values[iy, ix].astype(np.float32)
    if method == "telea":
        lo, hi = float(values[mask > 0].min()), float(values[mask > 0].max())
        filled = cv2.inpaint(((values - lo) / max(hi - lo, 1e-6)).astype(np.float32), (mask == 0).astype(np.uint8), 5, cv2.INPAINT_TELEA)
        return (filled * (hi - lo) + lo).astype(np.float32)
    raise ValueError(f"unknown hole filling method {method}")


def hole_distance(mask: np.ndarray) -> np.ndarray:
    """Distance (pixels) from every pixel to the nearest valid measurement."""
    from scipy.ndimage import distance_transform_edt

    if mask.all():
        return np.zeros(mask.shape, np.float32)
    return distance_transform_edt(mask == 0).astype(np.float32)
