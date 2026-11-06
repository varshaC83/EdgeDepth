"""Synthetic degradation: low resolution, sparsity, sensor noise and missing depth.

An :class:`Observation` is what every reconstruction method receives. All
methods get exactly the same observation, which makes comparisons fair.

* ``mode="lowres"``: ``values`` is an (H/s, W/s) grid, ``scale`` = s.
* ``mode="sparse"``: ``values`` is an (H, W) grid with only a fraction of
  pixels valid; ``scale`` = 1.

``mask`` is 1 where a measurement exists, 0 where it is missing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class DegradationConfig:
    scale: int = 4                 # 2, 4 or 8 (lowres mode)
    mode: str = "lowres"           # "lowres" | "sparse"
    sparse_density: float = 0.05   # fraction of HR pixels kept in sparse mode
    noise_level: float = 0.0       # multiplier of the Kinect axial-noise model (0 = no noise)
    missing_rate: float = 0.0      # fraction of measurements randomly dropped
    hole_blobs: int = 0            # number of large structured holes (e.g. specular / absorbing surfaces)
    edge_dropout: float = 0.0      # probability of dropping measurements on depth discontinuities (sensor shadows)
    seed: int = 0


@dataclass
class Observation:
    values: np.ndarray             # float32 metres, 0 = missing
    mask: np.ndarray               # float32 {0,1}
    scale: int                     # upsampling factor to reach the target grid (1 for sparse)
    target_hw: Tuple[int, int]
    mode: str = "lowres"
    meta: dict = field(default_factory=dict)

    @property
    def effective_scale(self) -> float:
        """Average spacing between measurements in HR pixels (used as model conditioning)."""
        if self.mode == "sparse":
            density = max(float(self.mask.mean()), 1e-4)
            return float(np.clip(1.0 / np.sqrt(density), 1.0, 16.0))
        return float(self.scale)


def kinect_sigma(z: np.ndarray) -> np.ndarray:
    """Axial noise std (m) of a structured-light sensor vs depth (Nguyen et al., 3DIMPVT 2012)."""
    return 0.0012 + 0.0019 * (z - 0.4) ** 2


def masked_area_downsample(depth: np.ndarray, scale: int) -> Tuple[np.ndarray, np.ndarray]:
    """Box-average a depth map by ``scale`` ignoring zeros (models sensor integration)."""
    h, w = depth.shape
    hh, ww = h // scale, w // scale
    d = depth[: hh * scale, : ww * scale].reshape(hh, scale, ww, scale)
    m = (d > 0).astype(np.float32)
    s = (d * m).sum(axis=(1, 3))
    c = m.sum(axis=(1, 3))
    lr = np.where(c > 0, s / np.maximum(c, 1), 0).astype(np.float32)
    return lr, (c > 0).astype(np.float32)


def _discontinuity_mask(d: np.ndarray, rel_thresh: float = 0.05) -> np.ndarray:
    pad = np.pad(d, 1, mode="edge")
    diffs = [np.abs(pad[1:-1, 2:] - d), np.abs(pad[1:-1, :-2] - d), np.abs(pad[2:, 1:-1] - d), np.abs(pad[:-2, 1:-1] - d)]
    jump = np.max(diffs, axis=0) / np.maximum(d, 1e-3)
    return (jump > rel_thresh) & (d > 0)


def degrade(gt_hr: np.ndarray, cfg: DegradationConfig) -> Observation:
    """Create a degraded observation from a ground-truth HR depth map (metres)."""
    rng = np.random.default_rng(cfg.seed)
    h, w = gt_hr.shape
    if cfg.mode == "sparse":
        valid = gt_hr > 0
        keep = (rng.random((h, w)) < cfg.sparse_density) & valid
        values = np.where(keep, gt_hr, 0).astype(np.float32)
        mask = keep.astype(np.float32)
        scale = 1
    elif cfg.mode == "lowres":
        if h % cfg.scale or w % cfg.scale:
            raise ValueError(f"HR size {h}x{w} must be divisible by the scale {cfg.scale}")
        values, mask = masked_area_downsample(gt_hr, cfg.scale)
        scale = cfg.scale
    else:
        raise ValueError(f"unknown mode {cfg.mode}")

    # Sensor noise (depth-dependent, Kinect-like)
    if cfg.noise_level > 0:
        sig = kinect_sigma(values) * cfg.noise_level
        values = np.where(mask > 0, values + rng.normal(size=values.shape).astype(np.float32) * sig, 0)
        values = np.maximum(values, 0).astype(np.float32)
        mask = (values > 0).astype(np.float32) * mask

    # Missing measurements: random dropout, structured blobs, edge shadows
    drop = np.zeros(values.shape, dtype=bool)
    if cfg.missing_rate > 0:
        drop |= rng.random(values.shape) < cfg.missing_rate
    if cfg.hole_blobs > 0:
        lh, lw = values.shape
        blob = np.zeros((lh, lw), np.uint8)
        for _ in range(cfg.hole_blobs):
            cy, cx = rng.integers(0, lh), rng.integers(0, lw)
            ay, ax = rng.integers(max(2, lh // 20), max(3, lh // 7)), rng.integers(max(2, lw // 20), max(3, lw // 7))
            cv2.ellipse(blob, (int(cx), int(cy)), (int(ax), int(ay)), float(rng.uniform(0, 180)), 0, 360, 1, -1)
        drop |= blob.astype(bool)
    if cfg.edge_dropout > 0:
        edges = _discontinuity_mask(values)
        edges = cv2.dilate(edges.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        drop |= edges & (rng.random(values.shape) < cfg.edge_dropout)
    values = np.where(drop, 0, values).astype(np.float32)
    mask = np.where(drop, 0, mask).astype(np.float32)

    return Observation(values=values, mask=mask, scale=scale, target_hw=(h, w), mode=cfg.mode,
                       meta={"config": cfg.__dict__.copy()})


def observation_from_lowres(lr_depth: np.ndarray, target_hw: Tuple[int, int]) -> Observation:
    """Wrap a *real* low-resolution depth map (no ground truth) as an observation.

    The LR map is resized to exactly (H/s, W/s) where s is inferred from the
    target size, so every method reconstructs on the same grid.
    """
    H, W = target_hw
    lh, lw = lr_depth.shape
    s = int(round(max(H / lh, W / lw)))
    s = int(min([2, 4, 8, 16], key=lambda c: abs(c - s))) if s > 1 else 1
    if s == 1:
        mask = (lr_depth > 0).astype(np.float32)
        return Observation(lr_depth.astype(np.float32), mask, 1, (H, W), mode="sparse" if mask.mean() < 0.5 else "lowres")
    gh, gw = H // s, W // s
    if (lh, lw) != (gh, gw):
        valid = (lr_depth > 0).astype(np.uint8)
        lr_depth = cv2.resize(lr_depth, (gw, gh), interpolation=cv2.INTER_NEAREST)
        valid = cv2.resize(valid, (gw, gh), interpolation=cv2.INTER_NEAREST)
        lr_depth = lr_depth * valid
    mask = (lr_depth > 0).astype(np.float32)
    return Observation(lr_depth.astype(np.float32), mask, s, (H, W), mode="lowres")


def observation_from_sensor(depth_hr: np.ndarray, scale: int) -> Observation:
    """Simulate a low-res sensor from a *raw* HR depth with holes (no clean GT)."""
    lr, m = masked_area_downsample(depth_hr, scale)
    return Observation(lr, m, scale, depth_hr.shape, mode="lowres")
