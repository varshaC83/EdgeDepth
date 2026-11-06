"""Controlled failure-case experiments (unseen NYU test frames).

F1  extremely sparse input     : density 10% -> 0.2%
F2  large missing region       : a rectangle covering ~20% of the LR map (incl. object boundaries)
F3  excessive noise            : Kinect noise model x1 .. x8 at 4x
F4  texture copying            : flat slanted-plane ground truth + real, highly textured RGB
F5  incorrect boundaries       : RGB/depth misalignment of 0..8 px (calibration error) at 8x

    python scripts/failure_cases.py
Outputs results/failure_cases.json and results/figures/failure_*.png
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edgedepth import METHOD_LABELS  # noqa: E402
from edgedepth.degradation import DegradationConfig, Observation, degrade  # noqa: E402
from edgedepth.io_utils import colorize, colorize_error, depth_range  # noqa: E402
from edgedepth.metrics import all_metrics  # noqa: E402
from edgedepth.model.dataset import load_split  # noqa: E402
from edgedepth.pipeline import run_method  # noqa: E402

M = ["bicubic", "joint_bilateral", "guided_filter", "deep_learning", "hybrid"]


def mean_metrics(frames, make_obs, rgb_fn=None, gt_fn=None):
    acc = {m: [] for m in M}
    for i, (_, rgb, gt) in enumerate(frames):
        g = gt_fn(gt, i) if gt_fn else gt
        r = rgb_fn(rgb, i) if rgb_fn else rgb
        obs = make_obs(g, i)
        for m in M:
            acc[m].append(all_metrics(run_method(m, r, obs), g))
    return {m: {k: float(np.nanmean([a[k] for a in v])) for k in v[0]} for m, v in acc.items()}


def save_strip(path, title, rgb, gt, outs, err_max=0.3):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("dark_background")
    lo, hi = depth_range(gt)
    n = 2 + len(outs)
    fig, ax = plt.subplots(2, n, figsize=(2.5 * n, 4.2))
    ax[0, 0].imshow(rgb)
    ax[0, 0].set_title("RGB", fontsize=8)
    ax[0, 1].imshow(colorize(gt, lo, hi))
    ax[0, 1].set_title("Ground truth", fontsize=8)
    ax[1, 0].axis("off")
    ax[1, 1].axis("off")
    for j, (m, d) in enumerate(outs.items()):
        ax[0, j + 2].imshow(colorize(d, lo, hi))
        ax[0, j + 2].set_title(METHOD_LABELS.get(m, m), fontsize=7)
        ax[1, j + 2].imshow(colorize_error(np.abs(d - gt), err_max))
        ax[1, j + 2].set_title(f"RMSE {np.sqrt(np.mean((d - gt) ** 2)):.3f} m", fontsize=7)
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    test = load_split("data/nyu", "test")
    frames = test[::3]  # 8 frames keep runtime moderate
    res = {"frames": [n for n, _, _ in frames]}

    # F1 sparsity
    res["F1_sparsity"] = {}
    for dens in (0.10, 0.05, 0.02, 0.01, 0.005, 0.002):
        res["F1_sparsity"][str(dens)] = mean_metrics(frames, lambda g, i, d=dens: degrade(g, DegradationConfig(mode="sparse", sparse_density=d, seed=40000 + i)))
        print("F1", dens, {m: round(v["rmse"], 4) for m, v in res["F1_sparsity"][str(dens)].items()}, flush=True)

    # F2 large missing region (inside vs outside the hole)
    def big_hole(g, i, s=4):
        obs = degrade(g, DegradationConfig(scale=s, seed=41000 + i))
        h, w = obs.values.shape
        y0, x0 = h // 3, w // 3
        obs.values[y0:y0 + h // 2, x0:x0 + int(w * 0.4)] = 0
        obs.mask[y0:y0 + h // 2, x0:x0 + int(w * 0.4)] = 0
        return obs

    f2 = {m: {"inside": [], "outside": []} for m in M}
    for i, (_, rgb, gt) in enumerate(frames):
        obs = big_hole(gt, i)
        hole = cv2.resize((obs.mask == 0).astype(np.uint8), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        for m in M:
            d = run_method(m, rgb, obs)
            f2[m]["inside"].append(float(np.sqrt(np.mean((d[hole] - gt[hole]) ** 2))))
            f2[m]["outside"].append(float(np.sqrt(np.mean((d[~hole] - gt[~hole]) ** 2))))
    res["F2_large_hole_rmse"] = {m: {k: float(np.mean(v)) for k, v in d.items()} for m, d in f2.items()}
    print("F2", res["F2_large_hole_rmse"], flush=True)
    name, rgb, gt = frames[1]
    obs = big_hole(gt, 1)
    save_strip("results/figures/failure_F2_large_hole.png", "F2: large missing region (4x, ~20% of LR map removed)", rgb, gt,
               {m: run_method(m, rgb, obs) for m in M})

    # F3 noise
    res["F3_noise"] = {}
    for nl in (1, 2, 4, 8):
        res["F3_noise"][str(nl)] = mean_metrics(frames, lambda g, i, n=nl: degrade(g, DegradationConfig(scale=4, noise_level=n, seed=42000 + i)))
        print("F3", nl, {m: round(v["rmse"], 4) for m, v in res["F3_noise"][str(nl)].items()}, flush=True)
    obs = degrade(gt, DegradationConfig(scale=4, noise_level=8, seed=42001))
    save_strip("results/figures/failure_F3_noise_x8.png", "F3: excessive noise (Kinect model x8, 4x)", rgb, gt, {m: run_method(m, rgb, obs) for m in M})

    # F4 texture copying: plane GT
    def plane(g, i):
        H, W = g.shape
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        return (2.0 + 1.5 * yy / H + 0.5 * xx / W).astype(np.float32)

    def textured(rgb, i):
        t = cv2.resize(rgb, None, fx=0.5, fy=0.5)
        t = np.tile(t, (2, 2, 1))[: rgb.shape[0], : rgb.shape[1]]
        return np.ascontiguousarray(t)

    from edgedepth.classical.edges import canny_edges

    acc = {m: [] for m in M}
    on_e = {m: [] for m in M}
    off_e = {m: [] for m in M}
    for i, (_, rgb, gt) in enumerate(frames):
        g, r = plane(gt, i), textured(rgb, i)
        obs = degrade(g, DegradationConfig(scale=8, seed=43000 + i))
        edges = cv2.dilate(canny_edges(r).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        for m in M:
            d = run_method(m, r, obs)
            e = np.abs(d - g)
            acc[m].append(all_metrics(d, g)["rmse"])
            on_e[m].append(float(e[edges].mean()))
            off_e[m].append(float(e[~edges].mean()))
    # texture-copy ratio: mean error ON RGB texture edges / mean error OFF them (1 = error unrelated to texture)
    res["F4_texture_copy"] = {m: {"rmse": float(np.mean(acc[m])), "mae_on_rgb_edges_mm": 1000 * float(np.mean(on_e[m])),
                                  "mae_off_rgb_edges_mm": 1000 * float(np.mean(off_e[m])),
                                  "texture_copy_ratio": float(np.mean(on_e[m]) / max(np.mean(off_e[m]), 1e-9))} for m in M}
    print("F4", res["F4_texture_copy"], flush=True)
    g, r = plane(gt, 0), textured(rgb, 0)
    obs = degrade(g, DegradationConfig(scale=8, seed=43000))
    save_strip("results/figures/failure_F4_texture_copy.png", "F4: texture copying (planar depth, textured RGB, 8x)", r, g,
               {m: run_method(m, r, obs) for m in M}, err_max=0.05)

    # F5 misalignment
    res["F5_misalignment"] = {}
    for sh in (0, 2, 4, 8):
        shift = lambda rgb, i, s=sh: np.ascontiguousarray(np.roll(rgb, s, axis=1))  # noqa: E731
        res["F5_misalignment"][str(sh)] = mean_metrics(frames, lambda g, i: degrade(g, DegradationConfig(scale=8, seed=44000 + i)), rgb_fn=shift)
        print("F5", sh, {m: (round(v["b_rmse"], 4), round(v["edge_f1"], 3)) for m, v in res["F5_misalignment"][str(sh)].items()}, flush=True)
    obs = degrade(gt, DegradationConfig(scale=8, seed=44001))
    r8 = np.ascontiguousarray(np.roll(rgb, 8, axis=1))
    save_strip("results/figures/failure_F5_misaligned_rgb.png", "F5: RGB shifted 8 px relative to depth (8x)", r8, gt, {m: run_method(m, r8, obs) for m in M})

    # F1 figure
    obs = degrade(gt, DegradationConfig(mode="sparse", sparse_density=0.002, seed=40001))
    save_strip("results/figures/failure_F1_sparse_0.2pct.png", "F1: extremely sparse input (0.2% of pixels)", rgb, gt, {m: run_method(m, rgb, obs) for m in M})

    with open("results/failure_cases.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
