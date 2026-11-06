"""Build the bundled demo samples (data/samples) used by the Streamlit app.

* NYU test-split frames (unseen during training) copied from data/nyu/test.
* Cross-dataset RGB-D frames from the Open3D sample-data releases on GitHub
  (TUM RGB-D, SUN RGB-D, Redwood, and one official NYU frame). These are
  different sensors/scenes and are used for generalisation checks.

All depth maps are re-encoded as 16-bit PNG in millimetres, RGB as PNG,
resized so the longest side is <= 640 and both sides are divisible by 8.

Usage:
    python scripts/prepare_samples.py --o3d /tmp/ds/o3d
Download the four zips first from
https://github.com/isl-org/open3d_downloads/releases/tag/20220201-data
(SampleTUMRGBDImage, SampleSUNRGBDImage, SampleRedwoodRGBDImages).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import cv2
import numpy as np


def fit8(rgb, depth, max_side=640):
    h, w = depth.shape
    s = min(1.0, max_side / max(h, w))
    nh, nw = int(h * s) // 8 * 8, int(w * s) // 8 * 8
    if (nh, nw) != (h, w):
        rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        depth = cv2.resize(depth, (nw, nh), interpolation=cv2.INTER_NEAREST)
    return rgb, depth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--o3d", default="/tmp/ds/o3d")
    ap.add_argument("--nyu", default="data/nyu/test")
    ap.add_argument("--out", default="data/samples")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    meta = []

    # NYU unseen test frames (a few representative ones)
    for name in ["basement_3_0001", "basement_3_0006", "study_9_0003", "study_9_0010"]:
        for suf in ("rgb", "depth"):
            src = os.path.join(args.nyu, f"{name}_{suf}.png")
            if os.path.exists(src):
                shutil.copy(src, os.path.join(args.out, f"nyu_{name}_{suf}.png"))
        if os.path.exists(os.path.join(args.nyu, f"{name}_rgb.png")):
            meta.append({"id": f"nyu_{name}", "dataset": "NYU Depth V2 (test split, unseen scene)", "gt": "dense filled"})

    def add(id_, dataset, rgb_path, depth_path, scale_to_mm, decode=None, note=""):
        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        d = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if rgb is None or d is None:
            print("skip", id_)
            return
        if decode:
            d = decode(d)
        d = (d.astype(np.float64) * scale_to_mm)
        d[(d < 0) | (d > 65535)] = 0
        d = d.astype(np.uint16)
        if rgb.shape[:2] != d.shape:
            rgb = cv2.resize(rgb, (d.shape[1], d.shape[0]), interpolation=cv2.INTER_AREA)
        rgb, d = fit8(rgb, d)
        cv2.imwrite(os.path.join(args.out, f"{id_}_rgb.png"), rgb)
        cv2.imwrite(os.path.join(args.out, f"{id_}_depth.png"), d)
        meta.append({"id": id_, "dataset": dataset, "gt": note, "holes_pct": round(float((d == 0).mean() * 100), 2)})

    o = args.o3d
    # TUM: 5000 units per metre
    add("tum_fr1", "TUM RGB-D (Kinect v1)", f"{o}/SampleTUMRGBDImage/TUM_color.png",
        f"{o}/SampleTUMRGBDImage/TUM_depth.png", 1000 / 5000, note="raw sensor depth with holes")
    # SUN RGB-D: 16-bit with 3-bit circular shift, millimetres
    add("sun_rgbd", "SUN RGB-D", f"{o}/SampleSUNRGBDImage/SUN_color.jpg", f"{o}/SampleSUNRGBDImage/SUN_depth.png", 1.0,
        decode=lambda d: ((d >> 3) | (d << 13)).astype(np.uint16), note="raw sensor depth with holes")
    # (The Open3D NYU sample stores raw 11-bit Kinect disparity, not metric depth, so it is not used.)
    # Redwood: millimetres
    add("redwood_0", "Redwood (PrimeSense)", f"{o}/SampleRedwoodRGBDImages/color/00000.jpg",
        f"{o}/SampleRedwoodRGBDImages/depth/00000.png", 1.0, note="raw sensor depth with holes")

    with open(os.path.join(args.out, "samples.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
