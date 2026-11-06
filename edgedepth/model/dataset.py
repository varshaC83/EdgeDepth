"""On-the-fly training samples: crop -> degrade -> network inputs + targets."""
from __future__ import annotations

import glob
import os
from typing import List, Tuple

import cv2
import numpy as np
import torch

from ..classical.edges import depth_discontinuities
from ..degradation import DegradationConfig, degrade
from ..io_utils import load_depth, load_rgb
from .inference import make_inputs


def load_split(root: str, split: str) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    items = []
    for p in sorted(glob.glob(os.path.join(root, split, "*_rgb.png"))):
        items.append((os.path.basename(p)[:-8], load_rgb(p), load_depth(p.replace("_rgb.png", "_depth.png"))))
    if not items:
        raise FileNotFoundError(f"no frames in {root}/{split}; run scripts/prepare_dataset.py first")
    return items


def random_config(rng: np.random.Generator) -> DegradationConfig:
    """Random degradation covering all supported input types."""
    sparse = rng.random() < 0.2
    return DegradationConfig(
        scale=int(rng.choice([2, 4, 8])),
        mode="sparse" if sparse else "lowres",
        sparse_density=float(rng.uniform(0.01, 0.1)),
        noise_level=float(rng.uniform(0.3, 2.5)) if rng.random() < 0.5 else 0.0,
        missing_rate=float(rng.uniform(0.0, 0.15)) if rng.random() < 0.5 else 0.0,
        hole_blobs=int(rng.integers(1, 4)) if rng.random() < 0.3 else 0,
        edge_dropout=float(rng.uniform(0.2, 0.8)) if rng.random() < 0.3 else 0.0,
        seed=int(rng.integers(0, 2 ** 31)),
    )


def targets(gt: np.ndarray):
    edge = depth_discontinuities(gt, thin=False).astype(np.float32)
    band = (cv2.dilate(edge, np.ones((5, 5), np.uint8)) > 0).astype(np.float32)
    return edge, band


def sample_batch(frames, batch: int, crop_hw=(128, 160), rng=None):
    rng = rng or np.random.default_rng()
    cols = {k: [] for k in ("rgb", "d0", "m", "s", "mu", "gt", "valid", "edge", "band")}
    ch, cw = crop_hw
    for _ in range(batch):
        _, rgb, gt = frames[int(rng.integers(len(frames)))]
        H, W = gt.shape
        y = int(rng.integers(0, (H - ch) // 8 + 1)) * 8
        x = int(rng.integers(0, (W - cw) // 8 + 1)) * 8
        rgb_c, gt_c = rgb[y:y + ch, x:x + cw].copy(), gt[y:y + ch, x:x + cw].copy()
        if rng.random() < 0.5:
            rgb_c, gt_c = rgb_c[:, ::-1].copy(), gt_c[:, ::-1].copy()
        # photometric jitter on the guide, global depth rescale (sensor/scene range invariance)
        a, b = rng.uniform(0.7, 1.3), rng.uniform(-25, 25)
        rgb_c = np.clip(rgb_c.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
        gt_c = (gt_c * rng.uniform(0.7, 1.4)).astype(np.float32)
        obs = degrade(gt_c, random_config(rng))
        t_rgb, t_d0, t_m, t_s, mu = make_inputs(rgb_c, obs)
        edge, band = targets(gt_c)
        cols["rgb"].append(t_rgb)
        cols["d0"].append(t_d0)
        cols["m"].append(t_m)
        cols["s"].append(t_s)
        cols["mu"].append(torch.tensor(mu).view(1, 1, 1, 1))
        cols["gt"].append(torch.from_numpy(gt_c)[None, None])
        cols["valid"].append(torch.from_numpy((gt_c > 0).astype(np.float32))[None, None])
        cols["edge"].append(torch.from_numpy(edge)[None, None])
        cols["band"].append(torch.from_numpy(band)[None, None])
    return {k: torch.cat(v, 0).float() for k, v in cols.items()}
