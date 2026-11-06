"""Loading, validation and visualisation helpers.

Depth inside EdgeDepth is always float32 metres with 0 meaning "missing".
"""
from __future__ import annotations

import io
from typing import Optional, Tuple

import cv2
import numpy as np

MAX_DEPTH_M = 10.0  # NYU Depth V2 sensor range; used for normalisation and PSNR peak


class InputError(ValueError):
    """Raised for invalid user input (shown verbatim in the UI)."""


def _decode(data: bytes, flags: int) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, flags)
    if img is None:
        raise InputError("Could not decode the uploaded image. Use PNG/JPG (RGB) or 16-bit PNG / .npy (depth).")
    return img


def load_rgb(src) -> np.ndarray:
    """Return an RGB uint8 image (H, W, 3) from a path or raw bytes."""
    if isinstance(src, (bytes, bytearray)):
        img = _decode(bytes(src), cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            raise InputError(f"Could not read RGB image {src}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_depth(src, units_per_metre: float = 1000.0, filename: str = "") -> np.ndarray:
    """Return depth in metres (float32, 0 = missing) from a path / bytes.

    Supports 16-bit PNG (any units, set ``units_per_metre``), 8-bit PNG
    (interpreted as 0..255 -> 0..MAX_DEPTH_M, a *relative* depth), and .npy
    float arrays already in metres.
    """
    name = filename or (str(src) if not isinstance(src, (bytes, bytearray)) else "")
    if name.lower().endswith(".npy"):
        arr = np.load(io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src, allow_pickle=False)
        d = np.asarray(arr, dtype=np.float32)
        if d.ndim == 3:
            d = d[..., 0]
    else:
        if isinstance(src, (bytes, bytearray)):
            raw = _decode(bytes(src), cv2.IMREAD_UNCHANGED)
        else:
            raw = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise InputError(f"Could not read depth map {src}")
        if raw.ndim == 3:
            raw = raw[..., 0]
        if raw.dtype == np.uint8:
            d = raw.astype(np.float32) / 255.0 * MAX_DEPTH_M
        else:
            d = raw.astype(np.float32) / float(units_per_metre)
    if d.ndim != 2:
        raise InputError("Depth map must be single-channel.")
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    d[d < 0] = 0
    if (d > 0).sum() < 16:
        raise InputError("Depth map has (almost) no valid values. Check the depth units setting.")
    return d.astype(np.float32)


def save_depth_png(depth_m: np.ndarray) -> bytes:
    d = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
    ok, buf = cv2.imencode(".png", d)
    return buf.tobytes()


def fit_to_multiple(rgb: np.ndarray, depth: Optional[np.ndarray], multiple: int = 8, max_side: int = 640):
    """Resize (only down) so both sides are <= max_side and divisible by ``multiple``.

    Returns (rgb, depth). Depth uses nearest-neighbour so no values are mixed
    across discontinuities.
    """
    h, w = rgb.shape[:2]
    s = min(1.0, max_side / max(h, w))
    nh, nw = max(multiple, int(h * s) // multiple * multiple), max(multiple, int(w * s) // multiple * multiple)
    if (nh, nw) != (h, w):
        rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        if depth is not None:
            depth = cv2.resize(depth, (nw, nh), interpolation=cv2.INTER_NEAREST)
    return rgb, depth


def depth_range(*depths: np.ndarray) -> Tuple[float, float]:
    vals = np.concatenate([d[d > 0].ravel() for d in depths if d is not None and (d > 0).any()])
    if vals.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(vals, [1, 99])
    if hi - lo < 1e-3:
        hi = lo + 1e-3
    return float(lo), float(hi)


def colorize(depth: np.ndarray, vmin: float, vmax: float, cmap: int = cv2.COLORMAP_TURBO,
             missing_color=(30, 30, 30)) -> np.ndarray:
    """Map depth (metres) to an RGB uint8 image with a FIXED range (fair comparison)."""
    norm = np.clip((depth - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    img = cv2.applyColorMap((255 - norm * 255).astype(np.uint8), cmap)  # near = warm
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img[depth <= 0] = missing_color
    return img


def colorize_error(err: np.ndarray, vmax: float) -> np.ndarray:
    norm = np.clip(err / max(vmax, 1e-6), 0, 1)
    img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def nearest_display(img: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    """Upscale a low-res visualisation with nearest neighbour so pixels stay visible."""
    return cv2.resize(img, (size_hw[1], size_hw[0]), interpolation=cv2.INTER_NEAREST)
