"""Unified entry point: every method takes (rgb, observation) and returns an HR depth map.

All methods receive the identical observation and output at the identical
target resolution, so comparisons are fair. Missing low-res values are filled
with the same push-pull hole filling before interpolation-based methods.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional

import numpy as np

from . import config as C
from .classical.boundary import boundary_band, weighted_median_refine
from .classical.edges import discontinuity_probability
from .classical.filters import guided_filter_upsample, joint_bilateral_upsample
from .classical.interpolation import interpolate
from .degradation import Observation
from .hybrid.fusion import hybrid_reconstruct, load_fusion_params
from .metrics import all_metrics, measure
from .model.inference import load_model, predict


class ModelUnavailable(RuntimeError):
    pass


def _dl(rgb, obs, key="refined"):
    out = predict(rgb, obs)
    if out is None:
        raise ModelUnavailable("Trained checkpoint not found (checkpoints/edgedepthnet.pt). "
                               "Deep learning is disabled; the hybrid uses its classical fallback.")
    return out[key]


def _dl_bref(rgb, obs):
    out = predict(rgb, obs)
    if out is None:
        raise ModelUnavailable("checkpoint missing")
    d = out["refined"]
    prob = np.maximum(out["edge_prob"], discontinuity_probability(d))
    return weighted_median_refine(d, rgb, boundary_band(d, prob, C.BREF_THRESH, C.BREF_WIDTH), radius=C.BREF_RADIUS)


METHOD_FUNCS: Dict[str, Callable[[np.ndarray, Observation], np.ndarray]] = {
    "nearest": lambda rgb, obs: interpolate(obs, "nearest"),
    "bilinear": lambda rgb, obs: interpolate(obs, "bilinear"),
    "bicubic": lambda rgb, obs: interpolate(obs, "bicubic"),
    "joint_bilateral": lambda rgb, obs: joint_bilateral_upsample(obs, rgb),
    "guided_filter": lambda rgb, obs: guided_filter_upsample(obs, rgb),
    "deep_learning": _dl,
    "hybrid": lambda rgb, obs: hybrid_reconstruct(rgb, obs, use_dl=True),
}

# Variants used only by the ablation study
def _dl_bp(rgb, obs):
    from .hybrid.fusion import adaptive_preprocess, backprojection_gain, edge_aware_backprojection

    d = _dl(rgb, obs)
    a = backprojection_gain(adaptive_preprocess(obs)[1])
    return d + a * (edge_aware_backprojection(d, obs, rgb) - d) if a > 0 else d


ABLATION_FUNCS: Dict[str, Callable] = {
    "A1_interpolation_only": lambda rgb, obs: interpolate(obs, "bicubic"),
    "A2_interpolation_guided_filter": lambda rgb, obs: guided_filter_upsample(obs, rgb),
    "A3_dl_only_backbone": lambda rgb, obs: _dl(rgb, obs, "coarse"),
    "A4a_dl_learned_refinement(CSPN)": _dl,
    "A4b_dl_cspn_edge_aware_backprojection": _dl_bp,
    "A4c_dl_cspn_weighted_median_snapping": _dl_bref,
    "A5_full_hybrid": lambda rgb, obs: hybrid_reconstruct(rgb, obs, use_dl=True),
    "X1_hybrid_without_backprojection": lambda rgb, obs: hybrid_reconstruct(rgb, obs, use_dl=True, backprojection=False),
    "X2_hybrid_constant_weights(no_cues)": lambda rgb, obs: hybrid_reconstruct(rgb, obs, use_dl=True, params=load_fusion_params(True, variant="nocues")),
    "X3_uniform_average_of_experts": lambda rgb, obs: hybrid_reconstruct(rgb, obs, use_dl=True, params=load_fusion_params(True, variant="uniform"), backprojection=False),
    "X4_hybrid_classical_only(fallback)": lambda rgb, obs: hybrid_reconstruct(rgb, obs, use_dl=False),
    "X5_hybrid_plus_weighted_median_snapping": lambda rgb, obs: hybrid_reconstruct(rgb, obs, use_dl=True, boundary_snapping=True),
}


def model_available() -> bool:
    return load_model() is not None


def run_method(name: str, rgb: np.ndarray, obs: Observation) -> np.ndarray:
    fn = METHOD_FUNCS.get(name) or ABLATION_FUNCS.get(name)
    if fn is None:
        raise KeyError(f"unknown method {name}")
    out = fn(rgb, obs)
    H, W = obs.target_hw
    if out.shape != (H, W):
        raise RuntimeError(f"{name} returned {out.shape}, expected {(H, W)}")
    return out.astype(np.float32)


def run_methods(rgb: np.ndarray, obs: Observation, methods: Iterable[str], gt: Optional[np.ndarray] = None,
                track_memory: bool = True) -> Dict[str, Dict]:
    """Run several methods; return {name: {'depth', 'latency_ms', 'memory_mb', 'metrics' | None, 'error'}}."""
    results = {}
    for m in methods:
        try:
            depth, ms, mb = measure(run_method, m, rgb, obs, track_memory=track_memory)
            results[m] = {"depth": depth, "latency_ms": ms, "memory_mb": mb,
                          "metrics": all_metrics(depth, gt) if gt is not None else None, "error": None}
        except ModelUnavailable as e:
            results[m] = {"depth": None, "latency_ms": None, "memory_mb": None, "metrics": None, "error": str(e)}
    return results
