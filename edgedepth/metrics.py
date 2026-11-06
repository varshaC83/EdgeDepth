"""Evaluation metrics for depth reconstruction.

All reference-based metrics are computed only on pixels where the ground
truth is valid (> 0). Depth is in metres.

* RMSE, MAE                 : metres (lower is better)
* PSNR                      : dB, peak = 10 m (NYU sensor range)
* SSIM                      : scikit-image SSIM, data_range = GT max - GT min
                              (missing GT pixels are set to 0 in both maps)
* B-RMSE (boundary RMSE)    : RMSE inside a 2-px band around GT depth discontinuities
                              (borders of missing GT regions are not counted as edges)
* Edge F1 (boundary accuracy): F1 between predicted and GT discontinuity maps,
                              2-px matching tolerance
* Edge sharpness            : mean |grad log D_pred| / mean |grad log D_gt| on GT
                              discontinuities (1.0 = as sharp as GT, <1 blurred)
"""
from __future__ import annotations

import time
import tracemalloc
from typing import Callable, Dict

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from .classical.edges import depth_discontinuities, depth_gradient_rel
from .io_utils import MAX_DEPTH_M

EDGE_THRESH = 0.03
METRIC_KEYS = ["rmse", "mae", "psnr", "ssim", "b_rmse", "edge_f1", "edge_sharpness"]
METRIC_LABELS = {
    "rmse": "RMSE (m) ↓", "mae": "MAE (m) ↓", "psnr": "PSNR (dB) ↑", "ssim": "SSIM ↑",
    "b_rmse": "Boundary RMSE (m) ↓", "edge_f1": "Edge F1 ↑", "edge_sharpness": "Edge sharpness (→1)",
    "latency_ms": "Latency (ms) ↓", "memory_mb": "Peak memory (MB) ↓",
}


def _valid(gt: np.ndarray) -> np.ndarray:
    return gt > 0


def rmse(pred, gt):
    v = _valid(gt)
    return float(np.sqrt(np.mean((pred[v] - gt[v]) ** 2)))


def mae(pred, gt):
    v = _valid(gt)
    return float(np.mean(np.abs(pred[v] - gt[v])))


def psnr(pred, gt, peak: float = MAX_DEPTH_M):
    v = _valid(gt)
    mse = np.mean((pred[v] - gt[v]) ** 2)
    return float(10 * np.log10(peak ** 2 / max(mse, 1e-12)))


def ssim(pred, gt):
    v = _valid(gt)
    g = np.where(v, gt, 0)
    p = np.where(v, pred, 0)
    rng = float(g[v].max() - g[v].min()) or 1.0
    return float(structural_similarity(g, p, data_range=rng))


def _gt_edges(gt):
    """GT discontinuities, excluding borders of missing GT regions (holes are not edges)."""
    valid_core = cv2.erode(_valid(gt).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    return depth_discontinuities(np.maximum(gt, 1e-3), EDGE_THRESH) & valid_core


def gt_edge_band(gt, width: int = 2):
    e = _gt_edges(gt).astype(np.uint8)
    return cv2.dilate(e, np.ones((2 * width + 1, 2 * width + 1), np.uint8)) > 0


def boundary_rmse(pred, gt, width: int = 2):
    band = gt_edge_band(gt, width) & _valid(gt)
    if band.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((pred[band] - gt[band]) ** 2)))


def edge_f1(pred, gt, tol: int = 2):
    eg = _gt_edges(gt)
    valid_core = cv2.erode(_valid(gt).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    ep = depth_discontinuities(np.maximum(pred, 1e-3), EDGE_THRESH) & valid_core
    if eg.sum() == 0:
        return float("nan")
    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    eg_d = cv2.dilate(eg.astype(np.uint8), k) > 0
    ep_d = cv2.dilate(ep.astype(np.uint8), k) > 0
    precision = (ep & eg_d).sum() / max(ep.sum(), 1)
    recall = (eg & ep_d).sum() / max(eg.sum(), 1)
    return float(2 * precision * recall / max(precision + recall, 1e-12))


def edge_sharpness(pred, gt):
    eg = _gt_edges(gt)
    if eg.sum() == 0:
        return float("nan")
    gp = depth_gradient_rel(np.maximum(pred, 1e-3))
    gg = depth_gradient_rel(np.maximum(gt, 1e-3))
    return float(gp[eg].mean() / max(gg[eg].mean(), 1e-9))


def all_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)
    if pred.shape != gt.shape:
        raise ValueError(f"prediction {pred.shape} and ground truth {gt.shape} differ in size")
    return {"rmse": rmse(pred, gt), "mae": mae(pred, gt), "psnr": psnr(pred, gt), "ssim": ssim(pred, gt),
            "b_rmse": boundary_rmse(pred, gt), "edge_f1": edge_f1(pred, gt), "edge_sharpness": edge_sharpness(pred, gt)}


def no_reference_stats(pred: np.ndarray, rgb: np.ndarray | None = None) -> Dict[str, float]:
    """Statistics that need no ground truth (shown when GT is unavailable)."""
    from .classical.edges import canny_edges

    d = depth_discontinuities(np.maximum(pred, 1e-3), EDGE_THRESH)
    out = {"depth_min_m": float(pred.min()), "depth_max_m": float(pred.max()),
           "discontinuity_px_pct": float(d.mean() * 100)}
    if rgb is not None:
        rgb_e = canny_edges(rgb)
        k = np.ones((5, 5), np.uint8)
        # fraction of depth edges that coincide with an RGB edge (alignment) and
        # fraction of RGB edges reproduced as depth edges (texture-copy risk proxy)
        out["depth_edges_on_rgb_edges_pct"] = float(((cv2.dilate(rgb_e.astype(np.uint8), k) > 0) & d).sum() / max(d.sum(), 1) * 100)
        out["rgb_edges_copied_pct"] = float(((cv2.dilate(d.astype(np.uint8), k) > 0) & rgb_e).sum() / max(rgb_e.sum(), 1) * 100)
    return out


def measure(fn: Callable, *args, track_memory: bool = True, **kwargs):
    """Run ``fn`` and return (result, latency_ms, peak_memory_mb).

    Memory = max(peak Python/NumPy heap allocations from tracemalloc,
    peak RSS increase sampled every 2 ms), which also captures PyTorch CPU
    tensors. Values are approximate and machine-dependent.
    """
    import threading

    import psutil

    proc = psutil.Process()
    base_rss = proc.memory_info().rss
    peak = [0]
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            peak[0] = max(peak[0], proc.memory_info().rss - base_rss)
            stop.wait(0.002)

    th = None
    if track_memory:
        tracemalloc.start()
        th = threading.Thread(target=sampler, daemon=True)
        th.start()
    t0 = time.perf_counter()
    try:
        out = fn(*args, **kwargs)
    finally:
        dt = (time.perf_counter() - t0) * 1000
        mem = 0.0
        if track_memory:
            stop.set()
            th.join()
            _, tm_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            mem = max(tm_peak, peak[0]) / 2 ** 20
    return out, dt, mem
