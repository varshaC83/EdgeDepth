"""Tune joint-bilateral and guided-filter hyper-parameters on the VALIDATION split.

The test split is never touched. Selection criterion: mean RMSE over scales
{2,4,8} and two degradation conditions (clean, noisy+missing). Results are
written to results/classical_tuning.json and the chosen values are the
defaults in edgedepth/config.py.
"""
import glob
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edgedepth.classical import guided_filter_upsample, joint_bilateral_upsample  # noqa: E402
from edgedepth.degradation import DegradationConfig, degrade  # noqa: E402
from edgedepth.io_utils import load_depth, load_rgb  # noqa: E402
from edgedepth.metrics import rmse  # noqa: E402

CONDS = {"clean": {}, "noisy_missing": dict(noise_level=1.0, missing_rate=0.05, edge_dropout=0.5)}


def main():
    names = sorted(glob.glob("data/nyu/val/*_rgb.png"))
    data = [(load_rgb(n), load_depth(n.replace("_rgb", "_depth"))) for n in names]
    cases = []
    for i, (rgb, gt) in enumerate(data):
        for s in (2, 4, 8):
            for c, kw in CONDS.items():
                cases.append((rgb, gt, degrade(gt, DegradationConfig(scale=s, seed=1000 + i, **kw)), s, c))
    out = {"jbu": [], "gf": []}
    for sr, ss in itertools.product([0.05, 0.1, 0.2, 0.4], [0.35, 0.5, 0.8]):
        e = {}
        for rgb, gt, obs, s, c in cases:
            e.setdefault(f"{s}x_{c}", []).append(rmse(joint_bilateral_upsample(obs, rgb, sigma_s=ss, sigma_r=sr), gt))
        e = {k: float(np.mean(v)) for k, v in e.items()}
        out["jbu"].append({"sigma_r": sr, "sigma_s": ss, "mean_rmse": float(np.mean(list(e.values()))), **e})
        print("jbu", out["jbu"][-1])
    for eps, rm in itertools.product([1e-5, 1e-4, 1e-3, 1e-2], [0.25, 0.5, 1.0]):
        e = {}
        for rgb, gt, obs, s, c in cases:
            r = int(max(1, round(obs.effective_scale * rm)))
            e.setdefault(f"{s}x_{c}", []).append(rmse(guided_filter_upsample(obs, rgb, radius=r, eps=eps), gt))
        e = {k: float(np.mean(v)) for k, v in e.items()}
        out["gf"].append({"eps": eps, "radius_mult": rm, "mean_rmse": float(np.mean(list(e.values()))), **e})
        print("gf", out["gf"][-1])
    best = {k: min(v, key=lambda r: r["mean_rmse"]) for k, v in out.items()}
    out["best"] = best
    os.makedirs("results", exist_ok=True)
    with open("results/classical_tuning.json", "w") as f:
        json.dump(out, f, indent=1)
    print("BEST", json.dumps(best, indent=1))


if __name__ == "__main__":
    main()
