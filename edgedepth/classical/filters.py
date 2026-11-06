"""Edge-aware RGB-guided refinement: joint bilateral upsampling and guided filtering."""
from __future__ import annotations

import cv2
import numpy as np

from .. import config as C
from ..degradation import Observation
from .holes import fill_holes
from .interpolation import interpolate


def _guide(rgb: np.ndarray) -> np.ndarray:
    return rgb.astype(np.float32) / 255.0


def joint_bilateral_upsample(obs: Observation, rgb: np.ndarray, sigma_s: float | None = None,
                             sigma_r: float = C.JBU_SIGMA_R, radius: int | None = None) -> np.ndarray:
    """Joint Bilateral Upsampling (Kopf et al., SIGGRAPH 2007).

        D(p) = sum_q  D_lr(q) m(q) f_s(||p_lr - q||) f_r(||I(p) - I_lr(q)||)  /  normaliser

    where q runs over the low-resolution measurements around p, f_s and f_r
    are Gaussians, I is the HR RGB guide and I_lr its area-downsampled copy,
    and m(q) is the measurement mask (missing values get zero weight). For
    sparse input the same filter runs on the HR grid over the valid samples
    (a joint bilateral normalised convolution).
    """
    H, W = obs.target_hw
    G = _guide(rgb)
    num = np.zeros((H, W), np.float32)
    den = np.zeros((H, W), np.float32)
    inv2r = 1.0 / (2 * sigma_r ** 2)

    if obs.mode == "lowres" and obs.scale > 1:
        s = obs.scale
        h, w = obs.values.shape
        r = radius if radius is not None else 2
        sig = sigma_s if sigma_s is not None else C.JBU_SIGMA_S
        G_lr = cv2.resize(G, (w, h), interpolation=cv2.INTER_AREA)
        cy = (np.arange(H, dtype=np.float32) + 0.5) / s - 0.5
        cx = (np.arange(W, dtype=np.float32) + 0.5) / s - 0.5
        by, bx = np.rint(cy).astype(int), np.rint(cx).astype(int)
        for dy in range(-r, r + 1):
            qy = np.clip(by + dy, 0, h - 1)
            wy = np.exp(-((qy - cy) ** 2) / (2 * sig ** 2))[:, None]
            for dx in range(-r, r + 1):
                qx = np.clip(bx + dx, 0, w - 1)
                wx = np.exp(-((qx - cx) ** 2) / (2 * sig ** 2))[None, :]
                Gq = G_lr[qy[:, None], qx[None, :]]
                wr = np.exp(-((G - Gq) ** 2).sum(-1) * inv2r)
                m = obs.mask[qy[:, None], qx[None, :]]
                wgt = wy * wx * wr * m
                num += wgt * obs.values[qy[:, None], qx[None, :]]
                den += wgt
    else:
        density = max(float(obs.mask.mean()), 1e-4)
        r = radius if radius is not None else int(np.clip(np.ceil(1.6 / np.sqrt(density)), 2, 14))
        sig = sigma_s if sigma_s is not None else r / 2.0
        V = np.pad(obs.values * obs.mask, r)
        M = np.pad(obs.mask, r)
        Gp = np.pad(G, ((r, r), (r, r), (0, 0)))
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                ws = np.exp(-(dy * dy + dx * dx) / (2 * sig ** 2))
                sl = (slice(r + dy, r + dy + H), slice(r + dx, r + dx + W))
                m = M[sl]
                if not m.any():
                    continue
                wr = np.exp(-((G - Gp[sl]) ** 2).sum(-1) * inv2r)
                wgt = ws * wr * m
                num += wgt * V[sl]
                den += wgt
    fallback = interpolate(obs, "bilinear")
    out = np.where(den > 1e-6, num / np.maximum(den, 1e-12), fallback)
    return out.astype(np.float32)


def _box(x: np.ndarray, r: int) -> np.ndarray:
    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), normalize=True, borderType=cv2.BORDER_REFLECT)


def guided_filter_color(I: np.ndarray, p: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Colour guided filter (He, Sun & Tang, TPAMI 2013).

    Inside each window w_k the output is a local linear model of the guide,
    q_i = a_k^T I_i + b_k, with
        a_k = (Sigma_k + eps U)^-1 cov_k(I, p),   b_k = mean_k(p) - a_k^T mean_k(I).
    Edges of the guide transfer to q, while eps controls how strong an RGB
    edge must be before it is copied (larger eps = smoother, less texture copy).
    """
    I = I.astype(np.float32)
    p = p.astype(np.float32)
    H, W, _ = I.shape
    mI = np.stack([_box(I[..., c], r) for c in range(3)], -1)
    mp = _box(p, r)
    mIp = np.stack([_box(I[..., c] * p, r) for c in range(3)], -1)
    cov_Ip = mIp - mI * mp[..., None]
    S = np.empty((H, W, 3, 3), np.float32)
    for i in range(3):
        for j in range(i, 3):
            v = _box(I[..., i] * I[..., j], r) - mI[..., i] * mI[..., j]
            S[..., i, j] = v
            S[..., j, i] = v
    S += eps * np.eye(3, dtype=np.float32)
    a = np.linalg.solve(S.reshape(-1, 3, 3), cov_Ip.reshape(-1, 3, 1)).reshape(H, W, 3)
    b = mp - (a * mI).sum(-1)
    ma = np.stack([_box(a[..., c], r) for c in range(3)], -1)
    mb = _box(b, r)
    return ((ma * I).sum(-1) + mb).astype(np.float32)


def guided_filter_upsample(obs: Observation, rgb: np.ndarray, radius: int | None = None,
                           eps: float = C.GF_EPS, init: np.ndarray | None = None) -> np.ndarray:
    """Guided-filter depth refinement: bilinear upsampling followed by an RGB-guided filter.

    The window radius grows with the upsampling factor (radius and eps were
    selected on the validation split, see edgedepth/config.py).
    """
    p = init if init is not None else interpolate(obs, "bilinear")
    r = radius if radius is not None else int(max(1, round(obs.effective_scale * C.GF_RADIUS_MULT)))
    q = guided_filter_color(_guide(rgb), p, r, eps)
    lo = float(p.min())
    return np.maximum(q, lo * 0.5).astype(np.float32)


def joint_bilateral_filter(depth: np.ndarray, rgb: np.ndarray, d: int = 7, sigma_r: float = 0.08,
                           sigma_s: float = 2.0) -> np.ndarray:
    """Joint bilateral *filter* on an HR depth map (no resolution change)."""
    obs = Observation(depth.astype(np.float32), (depth > 0).astype(np.float32), 1, depth.shape, mode="sparse")
    return joint_bilateral_upsample(obs, rgb, sigma_s=sigma_s, sigma_r=sigma_r, radius=d // 2)


__all__ = ["joint_bilateral_upsample", "guided_filter_upsample", "guided_filter_color", "joint_bilateral_filter", "fill_holes"]
