import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from ui.common import ROOT, depth_vis, hero, list_samples, load_json, load_sample, lr_vis, method_options, show_images
from edgedepth import METHOD_LABELS
from edgedepth.degradation import DegradationConfig, degrade
from edgedepth.io_utils import colorize_error, depth_range
from edgedepth.classical.edges import canny_edges
from edgedepth.pipeline import model_available, run_methods

SCENARIOS = {
    "Extremely sparse input (0.3% of pixels)": "sparse",
    "Large missing region (20% of the LR map)": "hole",
    "Excessive noise (Kinect model ×8)": "noise",
    "Texture copying (flat plane + textured RGB)": "texture",
    "Incorrect boundaries (RGB shifted 8 px)": "misalign",
}
WHY = {
    "sparse": "With 0.3 % samples the average gap between measurements is ~18 px: thin structures and small objects fall between samples, so no method has evidence for them. Interpolators produce Voronoi/triangle artefacts; the network and hybrid hallucinate plausible but wrong surfaces from RGB.",
    "hole": "Inside a large hole there is no depth evidence at all. Push-pull/bicubic propagate a smooth membrane from the hole border; the network uses RGB context but cannot recover absolute depth of objects that are entirely inside the hole, so errors concentrate there.",
    "noise": "At ×8 noise the per-sample σ (several cm at 3–5 m) exceeds many real depth steps. The noise estimator triggers denoising, but small discontinuities become indistinguishable from noise and are smoothed away; residual noise is still amplified by bicubic overshoot.",
    "texture": "The true depth is a smooth plane but the RGB is strongly textured. Any method that lets RGB similarity steer depth (JBU most of all, and the RGB term of the hybrid's back-projection) concentrates its error on texture edges — the texture-copy ratio quantifies it. The errors are millimetres here because the plane is smooth, but on real scenes the same mechanism produces ripples. The network's depth gate keeps its ratio near 1, yet the network itself is less accurate on this out-of-distribution plane than bicubic.",
    "misalign": "RGB-guided methods assume pixel-aligned colour and depth. A calibration/synchronisation error moves every RGB edge, so JBU, guided filter, the network and the hybrid all snap depth boundaries to the wrong place (boundary RMSE and edge F1 degrade), while unguided bicubic is unaffected.",
}


def _scenario(kind, rgb, gt, seed):
    if kind == "sparse":
        return rgb, gt, degrade(gt, DegradationConfig(mode="sparse", sparse_density=0.003, seed=seed))
    if kind == "hole":
        obs = degrade(gt, DegradationConfig(scale=4, seed=seed))
        h, w = obs.values.shape
        obs.values[h // 3:h // 3 + h // 2, w // 3:w // 3 + int(w * 0.4)] = 0
        obs.mask[h // 3:h // 3 + h // 2, w // 3:w // 3 + int(w * 0.4)] = 0
        return rgb, gt, obs
    if kind == "noise":
        return rgb, gt, degrade(gt, DegradationConfig(scale=4, noise_level=8, seed=seed))
    if kind == "texture":
        H, W = gt.shape
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        plane = (2.0 + 1.5 * yy / H + 0.5 * xx / W).astype(np.float32)
        t = cv2.resize(rgb, None, fx=0.5, fy=0.5)
        t = np.ascontiguousarray(np.tile(t, (2, 2, 1))[:H, :W])
        return t, plane, degrade(plane, DegradationConfig(scale=8, seed=seed))
    if kind == "misalign":
        return np.ascontiguousarray(np.roll(rgb, 8, axis=1)), gt, degrade(gt, DegradationConfig(scale=8, seed=seed))
    raise ValueError(kind)


def render():
    hero("Failure Analysis", "Real, reproducible limitations of every method — including ours.")
    nyu = [s for s in list_samples() if s.startswith("nyu_")]
    c1, c2 = st.columns(2)
    scen = c1.selectbox("Failure scenario", list(SCENARIOS))
    sid = c2.selectbox("Test frame", nyu or list(list_samples()))
    kind = SCENARIOS[scen]
    if st.button("▶ Run failure case", type="primary", width="stretch"):
        rgb, gt = load_sample(sid)
        r, g, obs = _scenario(kind, rgb, gt, 11)
        methods = [m for m in method_options(model_available()) if m in ("bicubic", "joint_bilateral", "guided_filter", "deep_learning", "hybrid")]
        with st.spinner("running …"):
            res = run_methods(r, obs, methods, gt=g, track_memory=False)
        st.session_state["fail"] = (kind, r, g, obs, res)
    if "fail" in st.session_state and st.session_state["fail"][0] == kind:
        _, r, g, obs, res = st.session_state["fail"]
        lo, hi = depth_range(g)
        show_images([("RGB guide", r), ("Input (nearest view)", lr_vis(obs, lo, hi)), ("Ground truth", depth_vis(g, lo, hi))], cols=4)
        ok = {m: x for m, x in res.items() if x["depth"] is not None}
        emax = 0.05 if kind == "texture" else 0.3
        show_images([(f"{METHOD_LABELS[m]} · RMSE {x['metrics']['rmse']:.3f}", depth_vis(x["depth"], lo, hi)) for m, x in ok.items()], cols=5)
        show_images([(f"|error| {METHOD_LABELS[m]}", colorize_error(np.abs(x["depth"] - g), emax)) for m, x in ok.items()], cols=5)
        rows = []
        for m, x in ok.items():
            row = {"Method": METHOD_LABELS[m], **{k: x["metrics"][k] for k in ("rmse", "b_rmse", "edge_f1", "ssim")}}
            if kind == "texture":
                e = np.abs(x["depth"] - g)
                ed = cv2.dilate(canny_edges(r).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
                row["Texture-copy ratio (err on RGB edges / elsewhere)"] = float(e[ed].mean() / max(e[~ed].mean(), 1e-9))
            rows.append(row)
        st.dataframe(pd.DataFrame(rows).style.format(precision=4), width="stretch", hide_index=True)
    st.markdown(f"**Why it fails.** {WHY[kind]}")

    f = load_json("failure_cases.json")
    if f:
        st.markdown("### Quantitative failure study (8 unseen test frames, scripts/failure_cases.py)")
        t1, t2, t3, t4, t5 = st.tabs(["F1 sparsity", "F2 large hole", "F3 noise", "F4 texture copy", "F5 misalignment"])
        with t1:
            st.dataframe(pd.DataFrame({d: {METHOD_LABELS[m]: v["rmse"] for m, v in r.items()} for d, r in f["F1_sparsity"].items()}).rename(
                columns=lambda d: f"{float(d) * 100:g}% density").style.format(precision=4), width="stretch")
            st.caption("RMSE (m). Error grows sharply below ~1% density for every method.")
        with t2:
            st.dataframe(pd.DataFrame({METHOD_LABELS[m]: v for m, v in f["F2_large_hole_rmse"].items()}).T.style.format(precision=4), width="stretch")
            st.caption("RMSE (m) inside vs outside the removed region.")
        with t3:
            st.dataframe(pd.DataFrame({f"noise ×{n}": {METHOD_LABELS[m]: v["rmse"] for m, v in r.items()} for n, r in f["F3_noise"].items()}).style.format(precision=4), width="stretch")
        with t4:
            st.dataframe(pd.DataFrame({METHOD_LABELS[m]: v for m, v in f["F4_texture_copy"].items()}).T.style.format(precision=3), width="stretch")
            st.caption("Ground truth is a plane. Texture-copy ratio = mean error on RGB texture edges / mean error elsewhere; > 1 means the error follows the RGB texture.")
        with t5:
            st.dataframe(pd.DataFrame({f"shift {s}px": {METHOD_LABELS[m]: v["b_rmse"] for m, v in r.items()} for s, r in f["F5_misalignment"].items()}).style.format(precision=4), width="stretch")
            st.caption("Boundary RMSE (m) at 8× as the RGB image is shifted against the depth.")
        figs = sorted(p for p in os.listdir(os.path.join(ROOT, "results", "figures")) if p.startswith("failure_"))
        if figs:
            pick = st.selectbox("Saved figure", figs)
            st.image(os.path.join(ROOT, "results", "figures", pick), width="stretch")
