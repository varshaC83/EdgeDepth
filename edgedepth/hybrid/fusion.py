r"""Hybrid depth reconstruction by cue-driven adaptive fusion.

Candidates
----------
Each "expert" k produces a full HR depth map D_k from the SAME observation:
    bicubic         - smooth, unbiased in flat regions, blurs boundaries
    guided_filter   - RGB-guided, sharp edges, risk of texture copying
    joint_bilateral - RGB-guided with measurement mask, robust to holes, can copy texture
    deep_learning   - EdgeDepthNet (learned prior: holes, noise, boundaries)
(without a trained model the hybrid falls back to the three classical experts).

Cues (per pixel p)
------------------
    B_dl  boundary probability predicted by the network (0 if no network)
    B_cl  classical discontinuity probability 1-exp(-(|grad log D|/t)^2) of the expert consensus
    V     local relative depth variation  std_5x5(D0) / mean_5x5(D0)
    E     RGB edge strength (Sobel, normalised)
    H     missing-measurement indicator 1 - G_sigma * M  (no nearby measurements)
    T_k   texture-copy score  E (1 - B_cl) |D_k - D_bicubic| / D_bicubic   (RGB edge, no depth edge,
          yet candidate departs from the smooth baseline -> probably copied texture)
    R_k   data-consistency residual |down_s(D_k) - D_lr| / D_lr on measured LR pixels
    A_k   deviation from the robust consensus |D_k - median_j D_j| / median_j D_j

Fusion rule
-----------
    z_k(p) = beta_k0 + sum_f beta_kf * phi_kf(p)                (linear in the cues)
    w_k(p) = exp(z_k(p)) / sum_j exp(z_j(p))                    (softmax gating)
    D_hyb(p) = sum_k w_k(p) D_k(p)

This is a per-pixel mixture of experts: the weights are not fixed but move
toward the expert that is expected to be most reliable for the local
configuration of cues. The coefficients beta (a few dozen numbers) are fitted
on the VALIDATION split by minimising the L1 error of D_hyb (scripts/fit_fusion.py),
so every coefficient is interpretable: e.g. a positive beta for (deep_learning, H)
means "in holes trust the network more".

Edge-aware back-projection (data consistency)
--------------------------------------------
The fused map is then made consistent with the measurements: for t = 1..T
    r_t = M ⊙ (D_lr - down_s(D_t))                       residual on measured LR pixels
    D_{t+1} = D_t + JTU(r_t; RGB, D_t)                   joint *trilateral* upsampling of r_t
where JTU weights LR residuals by spatial distance, RGB similarity and log-depth
similarity to the current estimate, so corrections never leak across depth
edges or into holes (M = 0). This is classical iterative back-projection made
edge-aware; it fixes the systematic bias the network leaves w.r.t. the observed data.

An optional weighted-median step can additionally snap mixed depths at
discontinuities to one side of the edge (sharper edges, higher edge F1, but on
NYU validation it increases RMSE, so it is off by default).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

from .. import config as C
from ..classical.boundary import boundary_band, weighted_median_refine
from ..classical.denoise import denoise_depth, estimate_noise_sigma
from ..classical.edges import discontinuity_probability, rgb_edge_strength
from ..classical.filters import guided_filter_upsample, joint_bilateral_upsample
from ..classical.holes import fill_holes
from ..classical.interpolation import interpolate, upsample_mask
from ..degradation import Observation

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATURES = ["B_dl", "B_cl", "V", "E", "H", "T_k", "R_k", "A_k"]
CLASSICAL = ["bicubic", "guided_filter", "joint_bilateral"]
WITH_DL = CLASSICAL + ["deep_learning"]
REL_SCALE = 20.0  # relative errors (~5%) mapped to O(1) feature values


@dataclass
class FusionParams:
    candidates: List[str]
    beta: np.ndarray                      # (K, 1 + F)
    features: List[str] = field(default_factory=lambda: list(FEATURES))
    fitted_on: str = "default (uniform)"

    @classmethod
    def uniform(cls, candidates):
        return cls(list(candidates), np.zeros((len(candidates), 1 + len(FEATURES)), np.float32))

    def to_dict(self):
        return {"candidates": self.candidates, "features": self.features, "beta": np.asarray(self.beta).tolist(),
                "fitted_on": self.fitted_on}

    @classmethod
    def from_dict(cls, d):
        return cls(d["candidates"], np.asarray(d["beta"], np.float32), d.get("features", list(FEATURES)), d.get("fitted_on", ""))


def load_fusion_params(with_dl: bool, path: str = C.FUSION_PARAMS, variant: str = "full") -> FusionParams:
    full = path if os.path.isabs(path) else os.path.join(ROOT, path)
    key = ("with_dl" if with_dl else "classical") + ("" if variant == "full" else f"_{variant}")
    if os.path.exists(full):
        with open(full) as f:
            d = json.load(f)
        if key in d:
            return FusionParams.from_dict(d[key])
    return FusionParams.uniform(WITH_DL if with_dl else CLASSICAL)


def _area_down(d: np.ndarray, s: int) -> np.ndarray:
    H, W = d.shape
    return d[: H // s * s, : W // s * s].reshape(H // s, s, W // s, s).mean(axis=(1, 3))


def adaptive_preprocess(obs: Observation, rel_noise_thresh: Optional[float] = None):
    """Estimate the noise level; denoise the observation only if it is noisy.

    Returns (observation_for_classical_experts, info).
    """
    sigma = estimate_noise_sigma(obs.values, obs.mask) if obs.mode == "lowres" else 0.0
    med = float(np.median(obs.values[obs.mask > 0])) if obs.mask.any() else 1.0
    if rel_noise_thresh is None:
        rel_noise_thresh = C.NOISE_REL_THRESH.get(obs.scale, max(C.NOISE_REL_THRESH.values()))
    if obs.mode != "lowres":
        return obs, {"noise_sigma_est_m": 0.0, "rel_noise": 0.0, "threshold": rel_noise_thresh, "denoised": False}
    info = {"noise_sigma_est_m": sigma, "rel_noise": sigma / max(med, 1e-3), "threshold": rel_noise_thresh, "denoised": False}
    if obs.mode == "lowres" and sigma / max(med, 1e-3) > rel_noise_thresh:
        v = denoise_depth(obs.values, obs.mask, sigma)
        info["denoised"] = True
        return Observation(v, obs.mask.copy(), obs.scale, obs.target_hw, obs.mode, dict(obs.meta)), info
    return obs, info


def compute_candidates(rgb: np.ndarray, obs: Observation, use_dl: bool = True, model=None) -> Dict:
    """Run every expert once. Returns {'cands': {name: D}, 'dl': dict|None, 'obs_c': obs, 'pre': info}."""
    obs_c, info = adaptive_preprocess(obs)
    filled = fill_holes(obs_c.values, obs_c.mask) if obs_c.mode == "lowres" else None
    cands = {
        "bicubic": interpolate(obs_c, "bicubic", filled_lr=filled),
        "guided_filter": guided_filter_upsample(obs_c, rgb),
        "joint_bilateral": joint_bilateral_upsample(obs_c, rgb),
    }
    dl = None
    if use_dl:
        from ..model.inference import predict

        dl = predict(rgb, obs, model)  # the network gets the raw observation (it learned to denoise)
        if dl is not None:
            cands["deep_learning"] = dl["refined"]
    return {"cands": cands, "dl": dl, "obs_c": obs_c, "pre": info}


def compute_cues(rgb: np.ndarray, obs: Observation, cands: Dict[str, np.ndarray], names: List[str],
                 dl: Optional[Dict] = None) -> Dict[str, np.ndarray]:
    """Return shared cues and the per-candidate feature tensor phi (K, F, H, W)."""
    H, W = obs.target_hw
    stack = np.stack([cands[n] for n in names], 0)
    cons = np.median(stack, 0)
    bic = cands["bicubic"]
    d0 = interpolate(obs, "bilinear")
    mean = cv2.blur(d0, (5, 5))
    var = np.maximum(cv2.blur(d0 * d0, (5, 5)) - mean * mean, 0)
    V = np.clip(np.sqrt(var) / np.maximum(mean, 1e-3) * REL_SCALE, 0, 3)
    E = rgb_edge_strength(rgb)
    B_cl = discontinuity_probability(cons)
    B_dl = dl["edge_prob"] if dl is not None else np.zeros((H, W), np.float32)
    m_hr = upsample_mask(obs)
    sig = max(1.0, obs.effective_scale)
    H_miss = 1.0 - np.clip(cv2.GaussianBlur(m_hr, (0, 0), sig) / max(cv2.GaussianBlur(np.ones_like(m_hr), (0, 0), sig).max(), 1e-6), 0, 1)
    s, lr, valid = obs.scale, obs.values, obs.mask > 0
    phi = np.zeros((len(names), len(FEATURES), H, W), np.float32)
    for k, n in enumerate(names):
        D = cands[n]
        T = E * (1 - B_cl) * np.clip(np.abs(D - bic) / np.maximum(bic, 1e-3) * REL_SCALE, 0, 3)
        A = np.clip(np.abs(D - cons) / np.maximum(cons, 1e-3) * REL_SCALE, 0, 3)
        if obs.mode == "lowres" and obs.mask.any():
            r = np.where(valid, np.abs(_area_down(D, s) - lr) / np.maximum(lr, 1e-3), 0)
            R = cv2.resize(np.clip(r * REL_SCALE, 0, 3).astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
        else:
            r = np.where(obs.mask > 0, np.abs(D - obs.values) / np.maximum(obs.values, 1e-3), 0)
            R = cv2.dilate(np.clip(r * REL_SCALE, 0, 3).astype(np.float32), np.ones((3, 3), np.uint8))
        phi[k] = np.stack([B_dl, B_cl, V, E, H_miss, T, R, A], 0)
    return {"phi": phi, "stack": stack, "B_dl": B_dl, "B_cl": B_cl, "V": V, "E": E, "H": H_miss, "consensus": cons}


def fuse(stack: np.ndarray, phi: np.ndarray, params: FusionParams):
    """Softmax mixture of experts. stack (K,H,W), phi (K,F,H,W) -> fused (H,W), weights (K,H,W)."""
    beta = np.asarray(params.beta, np.float32)
    z = beta[:, 0][:, None, None] + np.einsum("kf,kfhw->khw", beta[:, 1:], phi)
    z = z - z.max(0, keepdims=True)
    w = np.exp(z)
    w /= w.sum(0, keepdims=True)
    return (w * stack).sum(0).astype(np.float32), w.astype(np.float32)



def _joint_trilateral_residual(r: np.ndarray, mask: np.ndarray, obs: Observation, rgb: np.ndarray, d: np.ndarray,
                               sigma_d: float) -> np.ndarray:
    """Spread residuals r (on the observation grid) to HR with spatial x RGB x log-depth weights."""
    H, W = obs.target_hw
    G = rgb.astype(np.float32) / 255.0
    L = np.log(np.maximum(d, 1e-3)).astype(np.float32)
    num = np.zeros((H, W), np.float32)
    den = np.zeros((H, W), np.float32)
    inv_r = 1.0 / (2 * C.JBU_SIGMA_R ** 2)
    inv_d = 1.0 / (2 * sigma_d ** 2)
    if obs.mode == "lowres":
        s = obs.scale
        h, w = r.shape
        G_lr = cv2.resize(G, (w, h), interpolation=cv2.INTER_AREA)
        L_lr = np.log(np.maximum(_area_down(d, s), 1e-3)).astype(np.float32)
        cy = (np.arange(H, dtype=np.float32) + 0.5) / s - 0.5
        cx = (np.arange(W, dtype=np.float32) + 0.5) / s - 0.5
        by, bx = np.rint(cy).astype(int), np.rint(cx).astype(int)
        sig = C.JBU_SIGMA_S
        for dy in range(-2, 3):
            qy = np.clip(by + dy, 0, h - 1)
            wy = np.exp(-((qy - cy) ** 2) / (2 * sig ** 2))[:, None]
            for dx in range(-2, 3):
                qx = np.clip(bx + dx, 0, w - 1)
                wx = np.exp(-((qx - cx) ** 2) / (2 * sig ** 2))[None, :]
                Q = (qy[:, None], qx[None, :])
                wgt = wy * wx * mask[Q] * np.exp(-((G - G_lr[Q]) ** 2).sum(-1) * inv_r) * np.exp(-((L - L_lr[Q]) ** 2) * inv_d)
                num += wgt * r[Q]
                den += wgt
    else:
        density = max(float(mask.mean()), 1e-4)
        rad = int(np.clip(np.ceil(1.6 / np.sqrt(density)), 2, 14))
        sig = rad / 2.0
        Rp, Mp = np.pad(r, rad), np.pad(mask, rad)
        Gp = np.pad(G, ((rad, rad), (rad, rad), (0, 0)), mode="edge")
        Lp = np.pad(L, rad, mode="edge")
        for dy in range(-rad, rad + 1):
            for dx in range(-rad, rad + 1):
                sl = (slice(rad + dy, rad + dy + H), slice(rad + dx, rad + dx + W))
                m = Mp[sl]
                if not m.any():
                    continue
                wgt = m * np.exp(-(dy * dy + dx * dx) / (2 * sig ** 2)) * np.exp(-((G - Gp[sl]) ** 2).sum(-1) * inv_r) \
                    * np.exp(-((L - Lp[sl]) ** 2) * inv_d)
                num += wgt * Rp[sl]
                den += wgt
    return np.where(den > 1e-6, num / np.maximum(den, 1e-12), 0).astype(np.float32)


def backprojection_gain(pre: Dict) -> float:
    """alpha(q) = clip(1 - (q - q0)/span, 0, 1), q = estimated relative noise / scale threshold."""
    q = pre.get("rel_noise", 0.0) / max(pre.get("threshold", 1.0), 1e-9)
    return float(np.clip(1.0 - (q - C.BP_GAIN_Q0) / C.BP_GAIN_SPAN, 0.0, 1.0))


def edge_aware_backprojection(d: np.ndarray, obs: Observation, rgb: np.ndarray, iters: int = C.BP_ITERS,
                              sigma_d: float = C.BP_SIGMA_D) -> np.ndarray:
    """Iteratively enforce consistency with the measurements using edge-aware residual upsampling."""
    d = d.astype(np.float32).copy()
    for _ in range(iters):
        if obs.mode == "lowres":
            r = np.where(obs.mask > 0, obs.values - _area_down(d, obs.scale), 0).astype(np.float32)
        else:
            r = np.where(obs.mask > 0, obs.values - d, 0).astype(np.float32)
        d = np.maximum(d + _joint_trilateral_residual(r, obs.mask, obs, rgb, d, sigma_d), 1e-3)
    return d


def hybrid_reconstruct(rgb: np.ndarray, obs: Observation, use_dl: bool = True, model=None,
                       params: Optional[FusionParams] = None, backprojection: bool = True,
                       boundary_snapping: bool = False, precomputed: Optional[Dict] = None, return_details: bool = False):
    """Primary EdgeDepth method: experts -> cue-driven fusion -> edge-aware back-projection
    (-> optional weighted-median boundary snapping). Returns the depth map (and details if requested)."""
    pc = precomputed or compute_candidates(rgb, obs, use_dl=use_dl, model=model)
    has_dl = "deep_learning" in pc["cands"] and use_dl
    params = params or load_fusion_params(has_dl)
    names = [n for n in params.candidates if n in pc["cands"]]
    if len(names) != len(params.candidates):  # model missing -> classical fallback parameters
        params = load_fusion_params(False)
        names = params.candidates
    cues = compute_cues(rgb, obs, pc["cands"], names, pc["dl"] if has_dl else None)
    fused, w = fuse(cues["stack"], cues["phi"], params)
    # back-projection uses the RAW observation (denoised values are not measurements),
    # with a gain that shrinks as the estimated measurement noise grows
    alpha = backprojection_gain(pc["pre"]) if backprojection else 0.0
    out = fused + alpha * (edge_aware_backprojection(fused, obs, rgb) - fused) if alpha > 0 else fused
    band = np.zeros(fused.shape, bool)
    if boundary_snapping:
        prob = np.maximum(cues["B_dl"], discontinuity_probability(out))
        band = boundary_band(out, prob, C.BREF_THRESH, C.BREF_WIDTH)
        out = weighted_median_refine(out, rgb, band, radius=C.BREF_RADIUS)
    if not return_details:
        return out
    return out, {"weights": {n: w[i] for i, n in enumerate(names)}, "fused_before_refine": fused, "band": band,
                 "correction": out - fused, "bp_gain": alpha,
                 "cues": {k: cues[k] for k in ("B_dl", "B_cl", "V", "E", "H")}, "candidates": pc["cands"],
                 "preprocess": pc["pre"], "used_dl": has_dl, "params": params}
