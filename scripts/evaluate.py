"""Benchmark all methods on the unseen NYU TEST split (+ cross-dataset frames).

    python scripts/evaluate.py            # full benchmark (~10-20 min on 2 CPU cores)
    python scripts/evaluate.py --quick    # 6 test frames

Protocol (identical inputs for every method):
  * test frames: data/nyu/test (scenes basement_3, study_9; never used for
    training, tuning or fusion fitting)
  * conditions:  clean            - box-downsampled GT
                 noisy_missing    - + Kinect noise model (x1), 5% random dropout,
                                    50% dropout on depth edges
                 sparse_5pct      - 5% random HR samples (depth completion)
  * scales 2x, 4x, 8x for the low-res conditions; degradation seed fixed per frame
  * latency measured without memory tracking; peak memory measured in a
    separate pass on 4 frames (tracemalloc + RSS sampling)
Outputs: results/per_image.csv, results/summary.json, results/figures/*.png
"""
import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edgedepth import METHOD_LABELS, METHODS  # noqa: E402
from edgedepth.degradation import DegradationConfig, degrade, observation_from_sensor  # noqa: E402
from edgedepth.io_utils import load_depth, load_rgb  # noqa: E402
from edgedepth.metrics import METRIC_KEYS, all_metrics, measure  # noqa: E402
from edgedepth.model.dataset import load_split  # noqa: E402
from edgedepth.pipeline import model_available, run_method  # noqa: E402

CONDITIONS = {
    "clean": dict(),
    "noisy_missing": dict(noise_level=1.0, missing_rate=0.05, edge_dropout=0.5),
}


def cases_for(gt, i):
    for s in (2, 4, 8):
        for c, kw in CONDITIONS.items():
            yield f"{s}x", c, DegradationConfig(scale=s, seed=20000 + i, **kw)
    yield "sparse", "sparse_5pct", DegradationConfig(mode="sparse", sparse_density=0.05, seed=30000 + i)


def aggregate(rows, keys=("scale", "condition", "method")):
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for g, rs in groups.items():
        d = dict(zip(keys, g))
        d["n"] = len(rs)
        for m in METRIC_KEYS + ["latency_ms"]:
            vals = np.array([r[m] for r in rs if r.get(m) is not None and np.isfinite(r[m])], float)
            d[m] = float(vals.mean()) if len(vals) else None
            d[m + "_std"] = float(vals.std()) if len(vals) else None
        out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "figures"), exist_ok=True)

    test = load_split("data/nyu", "test")
    if args.quick:
        test = test[::4]
    methods = [m for m in METHODS if m != "deep_learning" or model_available()]
    print("methods:", methods, " frames:", len(test))
    rows = []
    t0 = time.time()
    for i, (name, rgb, gt) in enumerate(test):
        for scale, cond, cfg in cases_for(gt, i):
            obs = degrade(gt, cfg)
            for m in methods:
                t = time.perf_counter()
                d = run_method(m, rgb, obs)
                lat = (time.perf_counter() - t) * 1000
                rows.append({"frame": name, "scale": scale, "condition": cond, "method": m, "latency_ms": lat, **all_metrics(d, gt)})
        print(f"{i + 1}/{len(test)} {name}  {time.time() - t0:.0f}s", flush=True)

    # Peak memory pass (separate, so tracking overhead does not affect latency)
    mem = {}
    for name, rgb, gt in test[:4]:
        obs = degrade(gt, DegradationConfig(scale=4, seed=1))
        for m in methods:
            _, _, mb = measure(run_method, m, rgb, obs, track_memory=True)
            mem.setdefault(m, []).append(mb)
    mem = {m: float(np.mean(v)) for m, v in mem.items()}

    # Cross-dataset generalisation (raw sensor depth as reference, holes excluded)
    cross = []
    for p in sorted(glob.glob("data/samples/*_rgb.png")):
        sid = os.path.basename(p)[:-8]
        if sid.startswith("nyu_"):
            continue
        rgb = load_rgb(p)
        ref = load_depth(p.replace("_rgb.png", "_depth.png"))
        for s in (4, 8):
            obs = observation_from_sensor(ref, s)
            for m in methods:
                d = run_method(m, rgb, obs)
                cross.append({"sample": sid, "scale": f"{s}x", "method": m, **all_metrics(d, ref)})

    with open(os.path.join(args.out, "per_image.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = {
        "protocol": {"test_frames": [n for n, _, _ in test], "conditions": {k: v for k, v in CONDITIONS.items()},
                     "sparse_density": 0.05, "resolution": list(test[0][2].shape), "methods": methods,
                     "machine": f"{os.cpu_count()} CPU cores, no GPU", "quick": args.quick},
        "by_scale_condition": aggregate(rows),
        "by_scale": aggregate(rows, ("scale", "method")),
        "overall": aggregate(rows, ("method",)),
        "memory_mb_4x": mem,
        "cross_dataset": aggregate([{**r, "condition": r["sample"]} for r in cross], ("scale", "method")),
        "cross_dataset_per_sample": aggregate([{**r, "condition": r["sample"]} for r in cross], ("condition", "scale", "method")),
        "elapsed_s": time.time() - t0,
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    # Console table
    for sc in ("2x", "4x", "8x", "sparse"):
        for cond in list(CONDITIONS) + ["sparse_5pct"]:
            sel = [r for r in summary["by_scale_condition"] if r["scale"] == sc and r["condition"] == cond]
            if not sel:
                continue
            print(f"\n== {sc} {cond}")
            for r in sel:
                print(f"  {METHOD_LABELS[r['method']]:30s} RMSE {r['rmse']:.4f} MAE {r['mae']:.4f} PSNR {r['psnr']:.2f} "
                      f"SSIM {r['ssim']:.4f} B-RMSE {r['b_rmse']:.4f} EdgeF1 {r['edge_f1']:.3f} Sharp {r['edge_sharpness']:.3f} "
                      f"{r['latency_ms']:.0f}ms")
    print("memory", mem)
    make_figures(summary, test, methods, args.out)


def make_figures(summary, test, methods, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from edgedepth.io_utils import colorize, colorize_error, depth_range

    plt.style.use("dark_background")
    # RMSE / B-RMSE vs scale
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key in zip(axes, ["rmse", "b_rmse", "edge_f1"]):
        for m in methods:
            ys = [next(r[key] for r in summary["by_scale_condition"] if r["scale"] == s and r["condition"] == "noisy_missing" and r["method"] == m) for s in ("2x", "4x", "8x")]
            ax.plot([2, 4, 8], ys, marker="o", label=METHOD_LABELS[m], lw=2.5 if m == "hybrid" else 1.2)
        ax.set_xscale("log", base=2)
        ax.set_xticks([2, 4, 8])
        ax.set_xticklabels(["2x", "4x", "8x"])
        ax.set_title({"rmse": "RMSE (m)", "b_rmse": "Boundary RMSE (m)", "edge_f1": "Edge F1"}[key] + " - noisy + missing")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figures", "metrics_vs_scale.png"), dpi=110)
    plt.close(fig)

    # Qualitative grid on one test frame at 8x noisy
    name, rgb, gt = test[min(1, len(test) - 1)]
    obs = degrade(gt, DegradationConfig(scale=8, seed=20001, **CONDITIONS["noisy_missing"]))
    outs = {m: run_method(m, rgb, obs) for m in methods}
    lo, hi = depth_range(gt)
    n = len(methods) + 2
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 4.4))
    import cv2

    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("RGB guide", fontsize=8)
    axes[0, 1].imshow(colorize(cv2.resize(obs.values, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST), lo, hi))
    axes[0, 1].set_title("8x LR input", fontsize=8)
    axes[1, 0].imshow(colorize(gt, lo, hi))
    axes[1, 0].set_title("Ground truth", fontsize=8)
    axes[1, 1].axis("off")
    for j, m in enumerate(methods):
        axes[0, j + 2].imshow(colorize(outs[m], lo, hi))
        axes[0, j + 2].set_title(METHOD_LABELS[m], fontsize=7)
        axes[1, j + 2].imshow(colorize_error(np.abs(outs[m] - gt), 0.3))
        axes[1, j + 2].set_title(f"|err| RMSE {np.sqrt(np.mean((outs[m] - gt) ** 2)):.3f}", fontsize=7)
    for a in axes.ravel():
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figures", "qualitative_8x.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
