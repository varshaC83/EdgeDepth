import streamlit as st

from ui.common import card, hero, load_json
from edgedepth.pipeline import model_available

WORKFLOW = """
digraph G {
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#151B26", color="#4FD1C5", fontcolor="#E6EDF3", fontname="Helvetica", fontsize=11];
  edge [color="#7F8EA3"];
  rgb [label="RGB guide\\n(HR)"]; dep [label="Low-res / sparse\\nnoisy depth"];
  pre [label="Missing-depth detection\\nnoise estimate → denoise\\nhole filling (push-pull)"];
  cls [label="Classical experts\\nbicubic · guided filter · JBU"];
  dl  [label="EdgeDepthNet\\nRGB/depth encoders\\ngated fusion · CSPN"];
  cue [label="Cues: boundary prob., RGB edges,\\ndepth variation, holes,\\ntexture-copy, data residual"];
  fus [label="Adaptive softmax\\nmixture of experts", fillcolor="#123b3a"];
  ref [label="Edge-aware back-projection\\n(joint trilateral residual,\\ndata consistency)"];
  out [label="Dense HR depth\\n+ metrics", fillcolor="#123b3a"];
  rgb -> cls; rgb -> dl; dep -> pre -> cls; dep -> dl; cls -> cue; dl -> cue; rgb -> cue;
  cls -> fus; dl -> fus; cue -> fus -> ref -> out;
}
"""


def render():
    hero("EdgeDepth", "RGB-guided, edge-preserving depth map upsampling and refinement · Computer Vision Hackathon 2026 · Problem P4")
    if not model_available():
        st.warning("Trained checkpoint not found: deep learning is disabled and the hybrid runs its classical fallback.")

    st.markdown("### Problem")
    st.markdown(
        "Depth sensors (Kinect, ToF, LiDAR) deliver depth that is **low resolution, sparse, noisy and full of holes**, "
        "while the RGB camera next to them is sharp. Naive upsampling blurs object boundaries and creates *flying pixels* "
        "between foreground and background; naive RGB guidance copies colour texture into flat surfaces. "
        "EdgeDepth reconstructs a dense, high-resolution depth map that keeps depth discontinuities sharp **without** texture copying."
    )
    c = st.columns(3)
    with c[0]:
        card("Objectives", "Upsample 2×/4×/8× low-res or sparse depth · fill missing values · suppress noise · "
             "preserve boundaries and discontinuities · avoid texture copying · measure everything honestly.")
    with c[1]:
        card("Approach", "Classical CV (interpolation, JBU, guided filter, edges, discontinuities, denoising, hole filling) + "
             "a 0.7M-parameter RGB-guided CNN, combined by a cue-driven per-pixel mixture of experts and refined by edge-aware back-projection.")
    with c[2]:
        card("Evaluation", "RMSE, MAE, PSNR, SSIM, boundary RMSE, edge F1, edge sharpness, latency and memory on unseen, "
             "scene-disjoint NYU Depth V2 test frames plus TUM / SUN RGB-D / Redwood frames.")

    st.markdown("### System workflow")
    st.graphviz_chart(WORKFLOW, width="stretch")

    st.markdown("### Key features")
    f = st.columns(4)
    feats = [
        ("7 reconstruction methods", "Nearest, bilinear, bicubic, joint bilateral, guided filter, deep learning, hybrid — same input, same output size."),
        ("Realistic degradations", "Box-downsampling, Kinect depth-dependent noise, random / blob / edge-shadow holes, sparse sampling."),
        ("Explainable hybrid", "Per-pixel fusion weights and cue maps are visualised; coefficients are fitted on the validation split."),
        ("Honest evaluation", "Metrics only when ground truth exists; all benchmark numbers are produced by scripts in the repo."),
    ]
    for col, (t, b) in zip(f, feats):
        with col:
            card(t, b)

    s = load_json("summary.json")
    if s:
        st.markdown("### Headline results (unseen NYU test split, mean over 2×/4×/8× and all conditions)")
        rows = {r["method"]: r for r in s["overall"]}
        if "hybrid" in rows and "bicubic" in rows:
            h, b = rows["hybrid"], rows["bicubic"]
            m = st.columns(4)
            m[0].metric("Hybrid RMSE", f"{h['rmse']:.4f} m", f"{(h['rmse'] - b['rmse']) / b['rmse'] * 100:+.1f}% vs bicubic", delta_color="inverse")
            m[1].metric("Hybrid boundary RMSE", f"{h['b_rmse']:.4f} m", f"{(h['b_rmse'] - b['b_rmse']) / b['b_rmse'] * 100:+.1f}% vs bicubic", delta_color="inverse")
            m[2].metric("Hybrid edge F1", f"{h['edge_f1']:.3f}", f"{h['edge_f1'] - b['edge_f1']:+.3f} vs bicubic")
            m[3].metric("Hybrid SSIM", f"{h['ssim']:.4f}", f"{h['ssim'] - b['ssim']:+.4f} vs bicubic")
            st.caption("Numbers are read from results/summary.json produced by scripts/evaluate.py — see the Evaluation Dashboard.")
    st.markdown("### How to use")
    st.markdown("1. **Depth Reconstruction** — pick a sample or upload RGB + depth, choose factor, method and degradation, run.\n"
                "2. **Method Comparison** — all methods side by side with one colour scale, zoom and boundary profile.\n"
                "3. **Evaluation Dashboard** — live metrics for your input and the full benchmark / ablation.\n"
                "4. **Technical Architecture** and **Failure Analysis** — how it works and where it breaks.")
