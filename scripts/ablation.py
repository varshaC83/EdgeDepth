"""Ablation study on the unseen NYU TEST split.

Variants (all receive identical observations):
  A1  interpolation only (bicubic)
  A2  interpolation + guided filtering
  A3  deep learning only (EdgeDepthNet backbone output, before learned refinement)
  A4a deep learning + learned edge-aware refinement (CSPN) = the 'deep_learning' method
  A4b A4a + classical boundary-aware weighted-median refinement
  A5  complete hybrid (cue-driven fusion + boundary refinement)
  X1  hybrid without the boundary refinement step
  X2  hybrid with constant learned weights (no cues -> tests adaptivity)
  X3  plain average of the experts (no learned weights, no refinement)
  X4  hybrid without the network (classical fallback)

    python scripts/ablation.py
Outputs results/ablation.json and results/ablation.csv
"""
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edgedepth.degradation import DegradationConfig, degrade  # noqa: E402
from edgedepth.metrics import METRIC_KEYS, all_metrics  # noqa: E402
from edgedepth.model.dataset import load_split  # noqa: E402
from edgedepth.pipeline import ABLATION_FUNCS, run_method  # noqa: E402

CONDS = {"clean": dict(), "noisy_missing": dict(noise_level=1.0, missing_rate=0.05, edge_dropout=0.5)}


def main():
    test = load_split("data/nyu", "test")
    rows = []
    t0 = time.time()
    for i, (name, rgb, gt) in enumerate(test):
        for s in (2, 4, 8):
            for c, kw in CONDS.items():
                obs = degrade(gt, DegradationConfig(scale=s, seed=20000 + i, **kw))
                for v in ABLATION_FUNCS:
                    t = time.perf_counter()
                    d = run_method(v, rgb, obs)
                    rows.append({"frame": name, "scale": f"{s}x", "condition": c, "variant": v,
                                 "latency_ms": (time.perf_counter() - t) * 1000, **all_metrics(d, gt)})
        print(f"{i + 1}/{len(test)} {time.time() - t0:.0f}s", flush=True)
    agg = {}
    for r in rows:
        for key in [(r["variant"], "all"), (r["variant"], r["scale"])]:
            agg.setdefault(key, []).append(r)
    out = []
    for (v, sc), rs in agg.items():
        d = {"variant": v, "scale": sc, "n": len(rs)}
        for m in METRIC_KEYS + ["latency_ms"]:
            d[m] = float(np.nanmean([r[m] for r in rs]))
        out.append(d)
    with open("results/ablation.json", "w") as f:
        json.dump({"protocol": "NYU test split, scales 2/4/8 x {clean, noisy_missing}, identical seeds", "rows": out}, f, indent=1)
    with open("results/ablation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    for d in out:
        if d["scale"] == "all":
            print(f"{d['variant']:40s} RMSE {d['rmse']:.4f} MAE {d['mae']:.4f} B-RMSE {d['b_rmse']:.4f} F1 {d['edge_f1']:.3f} "
                  f"SSIM {d['ssim']:.4f} {d['latency_ms']:.0f}ms")


if __name__ == "__main__":
    main()
