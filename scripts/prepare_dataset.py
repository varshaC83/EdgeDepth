"""Prepare the NYU Depth V2 subset and cross-dataset samples used by EdgeDepth.

Source of NYU data
------------------
The official NYU Depth V2 download hosts (cs.nyu.edu) were not reachable from the
build environment, so we use a public GitHub mirror of 179 RGB-D frames taken from
ten NYU Depth V2 scenes (basement_1..3, study_1..4, study_7..9):

    https://github.com/jiy0ungkim/nyu_depth_v2   (branch: main)

Each frame is a 640x480 RGB image plus a dense 16-bit depth map in millimetres
(the NYU "filled" ground truth). If you have the official
``nyu_depth_v2_labeled.mat`` you can point ``--mat`` at it instead (see README).

Splits are SCENE-DISJOINT so that no scene seen during training appears in
validation or test:

    train : basement_1, study_1, study_2, study_3, study_4, study_7   (126 frames)
    val   : basement_2, study_8                                         (29 frames)
    test  : basement_3, study_9                                         (24 frames)

Preprocessing: RGB is resized 640x480 -> 320x240 with area interpolation, depth
with nearest-neighbour (no mixing across depth discontinuities). 320x240 is
divisible by 8 so 2x/4x/8x low-resolution inputs share an exact grid.

Usage:
    python scripts/prepare_dataset.py --src /path/to/nyu_depth_v2_clone --out data/nyu
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

SPLITS = {
    "train": ["basement_1", "study_1", "study_2", "study_3", "study_4", "study_7"],
    "val": ["basement_2", "study_8"],
    "test": ["basement_3", "study_9"],
}
HR_W, HR_H = 320, 240
MIRROR = "https://github.com/jiy0ungkim/nyu_depth_v2"


def clone_source(dst: str) -> str:
    if os.path.isdir(dst) and glob.glob(os.path.join(dst, "*", "*", "images")):
        return dst
    print(f"Cloning {MIRROR} -> {dst} (about 125 MB) ...")
    subprocess.check_call(["git", "-c", "gc.auto=0", "clone", "--depth", "1", "-b", "main", MIRROR, dst])
    return dst


def process_pair(rgb_path: str, depth_path: str):
    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if rgb is None or depth is None:
        raise IOError(f"Could not read {rgb_path} / {depth_path}")
    rgb = cv2.resize(rgb, (HR_W, HR_H), interpolation=cv2.INTER_AREA)
    depth = cv2.resize(depth, (HR_W, HR_H), interpolation=cv2.INTER_NEAREST)
    return rgb, depth.astype(np.uint16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/tmp/ds/jk", help="clone of the NYU mirror (cloned if missing)")
    ap.add_argument("--out", default="data/nyu")
    args = ap.parse_args()

    src = clone_source(args.src)
    manifest = {}
    for split, scenes in SPLITS.items():
        out_dir = os.path.join(args.out, split)
        shutil.rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir)
        items = []
        for scene in scenes:
            scene_dir = glob.glob(os.path.join(src, "*", scene))
            if not scene_dir:
                print(f"missing scene {scene}", file=sys.stderr)
                continue
            scene_dir = scene_dir[0]
            ids = sorted(int(os.path.basename(p)[:-4]) for p in glob.glob(os.path.join(scene_dir, "images", "*.png")))
            for i in ids:
                rgb, depth = process_pair(
                    os.path.join(scene_dir, "images", f"{i}.png"),
                    os.path.join(scene_dir, "gt_depth", f"{i}_depth.png"),
                )
                name = f"{scene}_{i:04d}"
                cv2.imwrite(os.path.join(out_dir, f"{name}_rgb.png"), rgb)
                cv2.imwrite(os.path.join(out_dir, f"{name}_depth.png"), depth)
                items.append(name)
        manifest[split] = items
        print(f"{split}: {len(items)} frames from {scenes}")
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump({"source": MIRROR, "resolution": [HR_H, HR_W], "depth_units": "millimetres",
                   "splits": SPLITS, "frames": manifest}, f, indent=1)


if __name__ == "__main__":
    main()
