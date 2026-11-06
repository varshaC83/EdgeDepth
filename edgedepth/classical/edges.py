"""Edge detection on RGB guidance and depth-discontinuity detection."""
from __future__ import annotations

import cv2
import numpy as np


def to_gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0


def sobel_magnitude(img: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude of a single-channel float image."""
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return np.hypot(gx, gy)


def rgb_edge_strength(rgb: np.ndarray) -> np.ndarray:
    """Normalised (0..1) colour edge strength: max Sobel magnitude over RGB channels."""
    f = cv2.GaussianBlur(rgb.astype(np.float32) / 255.0, (3, 3), 0)
    mag = np.max([sobel_magnitude(f[..., c]) for c in range(3)], axis=0)
    p = np.percentile(mag, 99) + 1e-6
    return np.clip(mag / p, 0, 1).astype(np.float32)


def canny_edges(rgb: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """Colour Canny: Canny on each RGB channel (auto thresholds from the channel
    median) combined with OR, so iso-luminant colour boundaries are not lost."""
    out = np.zeros(rgb.shape[:2], bool)
    for c in range(3):
        ch = cv2.GaussianBlur(np.ascontiguousarray(rgb[..., c]), (5, 5), 0)
        v = float(np.median(ch))
        lo, hi = int(max(10, (1 - sigma) * v)), int(min(255, max(30, (1 + sigma) * v)))
        out |= cv2.Canny(ch, lo, max(hi, lo + 1)) > 0
    return out


def depth_gradient_rel(depth: np.ndarray) -> np.ndarray:
    """Relative depth gradient |grad log D| per pixel (scale-invariant)."""
    logd = np.log(np.maximum(depth, 1e-3)).astype(np.float32)
    gx = cv2.Sobel(logd, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(logd, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return np.hypot(gx, gy)


def depth_discontinuities(depth: np.ndarray, rel_thresh: float = 0.03, thin: bool = True) -> np.ndarray:
    """Binary map of depth discontinuities (occlusion boundaries).

    A pixel is a discontinuity if the relative depth change to a neighbour
    exceeds ``rel_thresh`` (3% of depth per pixel by default). Smooth slanted
    surfaces (floor, walls) change far less than this per pixel, so they are
    not flagged. Optional non-maximum thinning keeps 1-pixel-wide contours.
    """
    g = depth_gradient_rel(depth)
    edges = (g > rel_thresh) & (depth > 0)
    if thin:
        dil = cv2.dilate(g, np.ones((3, 3), np.uint8))
        edges &= g >= dil * 0.999
        # keep continuity: add back pixels strongly above threshold
        edges |= g > 3 * rel_thresh
    return edges


def discontinuity_probability(depth: np.ndarray, rel_thresh: float = 0.03) -> np.ndarray:
    """Soft (0..1) discontinuity likelihood, 1 - exp(-(g/t)^2)."""
    g = depth_gradient_rel(depth)
    return (1.0 - np.exp(-(g / rel_thresh) ** 2)).astype(np.float32)


def texture_edges(rgb: np.ndarray, depth: np.ndarray, rel_thresh: float = 0.03) -> np.ndarray:
    """RGB edges that do NOT coincide with a depth discontinuity (pure texture).

    These are the places where RGB-guided methods are at risk of copying
    texture into the depth map.
    """
    rgb_e = canny_edges(rgb)
    dd = cv2.dilate(depth_discontinuities(depth, rel_thresh, thin=False).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    return rgb_e & ~dd
