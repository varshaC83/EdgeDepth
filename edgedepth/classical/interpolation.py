"""Interpolation baselines: nearest neighbour, bilinear and bicubic.

Low-resolution input: classic image resampling on the regular LR grid
(OpenCV, half-pixel-centre convention, identical to the degradation model).
Sparse input: scattered-data interpolation of the valid samples (nearest
sample, Delaunay-linear, Clough-Tocher cubic), the natural generalisation of
the same three kernels to irregular samples.

Holes in the LR grid are filled first (shared preprocessing) so that missing
values are not interpolated as zeros.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator
from scipy.ndimage import distance_transform_edt

from ..degradation import Observation
from .holes import fill_holes

_CV = {"nearest": cv2.INTER_NEAREST, "bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC}


def _sparse_interp(values: np.ndarray, mask: np.ndarray, kind: str) -> np.ndarray:
    h, w = values.shape
    _, (iy, ix) = distance_transform_edt(mask == 0, return_indices=True)
    nearest = values[iy, ix].astype(np.float32)
    if kind == "nearest":
        return nearest
    ys, xs = np.nonzero(mask > 0)
    if len(ys) < 4:
        return nearest
    pts = np.stack([xs, ys], 1).astype(np.float64)
    vals = values[ys, xs].astype(np.float64)
    gy, gx = np.mgrid[0:h, 0:w]
    q = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float64)
    interp = LinearNDInterpolator(pts, vals) if kind == "bilinear" else CloughTocher2DInterpolator(pts, vals)
    out = interp(q).reshape(h, w)
    out = np.where(np.isfinite(out), out, nearest)  # outside the convex hull
    return out.astype(np.float32)


def interpolate(obs: Observation, kind: str = "bicubic", filled_lr: np.ndarray | None = None) -> np.ndarray:
    """Upsample an observation to the target grid with the chosen kernel."""
    if kind not in _CV:
        raise ValueError(f"unknown interpolation {kind}")
    H, W = obs.target_hw
    if obs.mode == "sparse":
        return _sparse_interp(obs.values, obs.mask, kind)
    lr = filled_lr if filled_lr is not None else fill_holes(obs.values, obs.mask)
    out = cv2.resize(lr, (W, H), interpolation=_CV[kind])
    if kind == "bicubic":
        # bicubic can overshoot below zero at very strong steps; depth must stay positive
        out = np.maximum(out, float(lr[lr > 0].min()) if (lr > 0).any() else 0.0)
    return out.astype(np.float32)


def upsample_mask(obs: Observation) -> np.ndarray:
    """Measurement mask on the HR grid (nearest neighbour)."""
    H, W = obs.target_hw
    if obs.mask.shape == (H, W):
        return obs.mask.astype(np.float32)
    return cv2.resize(obs.mask.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
