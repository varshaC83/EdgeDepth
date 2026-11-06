"""Model input preparation and inference (CPU)."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, Optional

import cv2
import numpy as np
import torch

from .. import config as C
from ..classical.holes import push_pull_fill
from ..classical.interpolation import interpolate, upsample_mask
from ..degradation import Observation
from .network import IMAGENET_MEAN, IMAGENET_STD, EdgeDepthNet, scale_code

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def initial_depth(obs: Observation) -> np.ndarray:
    """D0 fed to the network: bicubic (low-res input) or push-pull fill (sparse input)."""
    if obs.mode == "sparse":
        return push_pull_fill(obs.values, obs.mask)
    return interpolate(obs, "bicubic")


def make_inputs(rgb: np.ndarray, obs: Observation, d0: Optional[np.ndarray] = None):
    """Return network tensors (batch of 1) plus the normalisation constant mu."""
    d0 = initial_depth(obs) if d0 is None else d0
    m = upsample_mask(obs)
    mu = float(d0[d0 > 0].mean()) if (d0 > 0).any() else 1.0
    t_rgb = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
    t_rgb = (t_rgb - IMAGENET_MEAN) / IMAGENET_STD
    t_d0 = torch.from_numpy(d0 / mu)[None, None].float()
    t_m = torch.from_numpy(m)[None, None].float()
    t_s = torch.full_like(t_m, scale_code(obs.effective_scale))
    return t_rgb, t_d0, t_m, t_s, mu


@lru_cache(maxsize=2)
def load_model(path: str = C.CHECKPOINT) -> Optional[EdgeDepthNet]:
    """Load the trained checkpoint once (cached). Returns None if it is missing."""
    full = path if os.path.isabs(path) else os.path.join(ROOT, path)
    if not os.path.exists(full):
        return None
    ck = torch.load(full, map_location="cpu", weights_only=False)
    model = EdgeDepthNet(**ck.get("model_args", {}))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    return model


def model_info(path: str = C.CHECKPOINT) -> Dict:
    full = path if os.path.isabs(path) else os.path.join(ROOT, path)
    if not os.path.exists(full):
        return {}
    ck = torch.load(full, map_location="cpu", weights_only=False)
    return {k: v for k, v in ck.items() if k not in ("state_dict", "optimizer")}


@torch.no_grad()
def predict(rgb: np.ndarray, obs: Observation, model: Optional[EdgeDepthNet] = None) -> Optional[Dict[str, np.ndarray]]:
    """Run EdgeDepthNet. Returns dict with 'refined', 'coarse' depth (m) and 'edge_prob', or None."""
    model = model or load_model()
    if model is None:
        return None
    t_rgb, t_d0, t_m, t_s, mu = make_inputs(rgb, obs)
    H, W = obs.target_hw
    # pad to a multiple of 8 for the 3 stride-2 stages
    ph, pw = (-H) % 8, (-W) % 8
    if ph or pw:
        pad = lambda t: torch.nn.functional.pad(t, (0, pw, 0, ph), mode="replicate")  # noqa: E731
        t_rgb, t_d0, t_m, t_s = map(pad, (t_rgb, t_d0, t_m, t_s))
    dev = next(model.parameters()).device
    out = model(*(t.to(dev) for t in (t_rgb, t_d0, t_m, t_s)))
    out = {k: v.detach().cpu() for k, v in out.items()}
    res = {
        "refined": np.maximum(out["refined"][0, 0, :H, :W].numpy() * mu, 1e-3).astype(np.float32),
        "coarse": np.maximum(out["coarse"][0, 0, :H, :W].numpy() * mu, 1e-3).astype(np.float32),
        "edge_prob": torch.sigmoid(out["edge_logit"])[0, 0, :H, :W].numpy().astype(np.float32),
        "rgb_gate": out["gate1"][0].mean(0)[:H, :W].numpy().astype(np.float32),
    }
    return res


def resize_like(x: np.ndarray, hw) -> np.ndarray:
    return cv2.resize(x, (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)
