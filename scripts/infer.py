"""Command-line inference.

Real low-resolution input (no ground truth):
    python scripts/infer.py --rgb rgb.png --depth lowres_depth.png --units 1000 --method hybrid --out hr.png

Simulate from a high-resolution ground truth and report metrics:
    python scripts/infer.py --rgb rgb.png --depth gt.png --simulate --scale 8 --noise 1 --missing 0.05 --out hr.png

The output is a 16-bit PNG in millimetres (+ a colour preview *_vis.png).
"""
import argparse
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edgedepth import METHODS  # noqa: E402
from edgedepth.degradation import DegradationConfig, degrade, observation_from_lowres  # noqa: E402
from edgedepth.io_utils import colorize, depth_range, fit_to_multiple, load_depth, load_rgb, save_depth_png  # noqa: E402
from edgedepth.metrics import all_metrics, measure  # noqa: E402
from edgedepth.pipeline import run_method  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb", required=True)
    ap.add_argument("--depth", required=True)
    ap.add_argument("--units", type=float, default=1000.0, help="depth units per metre (1000 = mm)")
    ap.add_argument("--method", default="hybrid", choices=METHODS)
    ap.add_argument("--simulate", action="store_true", help="treat --depth as HR ground truth and degrade it")
    ap.add_argument("--scale", type=int, default=4, choices=[2, 4, 8])
    ap.add_argument("--sparse", type=float, default=0.0, help="sparse density (0 = low-res mode)")
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--missing", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="edgedepth_out.png")
    a = ap.parse_args()

    rgb = load_rgb(a.rgb)
    depth = load_depth(a.depth, a.units, a.depth)
    gt = None
    if a.simulate:
        if depth.shape != rgb.shape[:2]:
            rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA)
        rgb, gt = fit_to_multiple(rgb, depth)
        cfg = DegradationConfig(scale=a.scale, mode="sparse" if a.sparse > 0 else "lowres", sparse_density=a.sparse or 0.05,
                                noise_level=a.noise, missing_rate=a.missing, seed=a.seed)
        obs = degrade(gt, cfg)
    else:
        rgb, _ = fit_to_multiple(rgb, None)
        obs = observation_from_lowres(depth, rgb.shape[:2])
    out, ms, mb = measure(run_method, a.method, rgb, obs)
    with open(a.out, "wb") as f:
        f.write(save_depth_png(out))
    lo, hi = depth_range(out)
    cv2.imwrite(os.path.splitext(a.out)[0] + "_vis.png", cv2.cvtColor(colorize(out, lo, hi), cv2.COLOR_RGB2BGR))
    report = {"method": a.method, "output": a.out, "size": list(out.shape), "scale": obs.scale, "latency_ms": round(ms, 1),
              "peak_memory_mb": round(mb, 1)}
    if gt is not None:
        report["metrics"] = {k: round(v, 5) for k, v in all_metrics(out, gt).items()}
    else:
        report["metrics"] = "unavailable (no ground truth)"
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
