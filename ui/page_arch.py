import json
import os

import numpy as np
import pandas as pd
import streamlit as st

from ui.common import ROOT, hero, load_json
from ui.page_home import WORKFLOW
from edgedepth import METHOD_LABELS
from edgedepth import config as C
from edgedepth.hybrid.fusion import FEATURES
from edgedepth.model.inference import model_info

NET = """
digraph N {
  rankdir=TB; bgcolor="transparent"; nodesep=0.25; ranksep=0.3;
  node [shape=box, style="rounded,filled", fillcolor="#151B26", color="#4FD1C5", fontcolor="#E6EDF3", fontname="Helvetica", fontsize=10];
  edge [color="#7F8EA3"];
  subgraph cluster_r { label="RGB encoder"; fontcolor="#9FB3C8"; color="#2C3A4F";
    r1 [label="2×Conv 3→16 @1"]; r2 [label="Conv s2 →32 @1/2"]; r3 [label="Conv s2 →64 @1/4"]; r1->r2->r3; }
  subgraph cluster_d { label="Depth encoder [D0/μ, mask, scale code]"; fontcolor="#9FB3C8"; color="#2C3A4F";
    d1 [label="2×Conv 3→16 @1"]; d2 [label="Conv s2 →32 @1/2"]; d3 [label="Conv s2 →64 @1/4"]; }
  f1 [label="Gated fusion @1\\nf = f_d + Conv[f_d, σ(W[f_d,f_r])·f_r]", fillcolor="#123b3a"];
  f2 [label="Gated fusion @1/2", fillcolor="#123b3a"]; f3 [label="Gated fusion @1/4", fillcolor="#123b3a"];
  b  [label="Conv s2 →96 @1/8 + 2 ResBlocks"];
  u3 [label="Up + skip @1/4"]; u2 [label="Up + skip @1/2"]; u1 [label="Up + skip @1"];
  h  [label="Heads: residual · boundary logit · 8 affinities"];
  c  [label="D_coarse = D0 + residual"]; s [label="CSPN ×6 (learned edge-aware propagation)", fillcolor="#123b3a"]; o [label="D_dl × μ"];
  r1->f1; d1->f1; f1->d2; r2->f2; d2->f2; f2->d3; r3->f3; d3->f3; f3->b->u3->u2->u1->h->c->s->o;
  f3->u3 [style=dashed]; f2->u2 [style=dashed]; f1->u1 [style=dashed];
}
"""


def render():
    hero("Technical Architecture", "How the classical modules, the network and the hybrid fusion fit together.")
    st.markdown("### Overall data flow")
    st.graphviz_chart(WORKFLOW, width="stretch")
    st.markdown("""
**Data flow.** Input RGB *I* (H×W) and an observation *O* (an H/s×W/s grid or sparse H×W samples, 0 = missing).
1. *Missing-depth detection* marks zeros/invalid values; the *noise estimator* (median residual + MAD) decides whether to denoise (median + range-adaptive bilateral).
2. *Hole filling* (push-pull normalised convolution) fills the LR grid so interpolation never sees zeros.
3. Every expert produces a full H×W depth map from the same observation.
4. Cues are computed and the softmax mixture fuses the experts.
5. Edge-aware back-projection makes the fused map consistent with the measurements (residuals are spread by a joint
   trilateral kernel: space × RGB × log-depth, so corrections stop at depth edges).
""")
    st.markdown("### Classical computer-vision modules")
    st.markdown(f"""
| Module | Implementation (edgedepth/classical) |
|---|---|
| Nearest / bilinear / bicubic | `interpolation.py` — OpenCV resampling on the LR grid; scattered nearest / Delaunay-linear / Clough-Tocher-cubic for sparse input |
| Joint bilateral (JBU) | `filters.py` — Kopf et al. 2007, vectorised over the LR window; σ_s = {C.JBU_SIGMA_S} LR px, σ_r = {C.JBU_SIGMA_R} (validation-tuned) |
| Guided filter | `filters.py` — colour guided filter (He et al.) on bilinear depth, r = max(1, round(s/4)), ε = {C.GF_EPS} |
| Edge detection | `edges.py` — Sobel colour gradient, auto-threshold Canny |
| Depth discontinuities | `edges.py` — \\|∇ log D\\| > 0.03 (scale-invariant), NMS thinning, soft probability 1 − exp(−(g/t)²) |
| Noise removal | `denoise.py` — MAD noise estimate, 3×3 median + bilateral with σ_r = 3σ̂ |
| Missing detection / hole filling | `holes.py` — range check; push-pull pyramid (default), Telea inpainting, nearest |
| Boundary snapping (optional) | `boundary.py` — joint weighted median in a band around discontinuities |
| Edge-aware back-projection | `hybrid/fusion.py` — iterative residual correction with a joint trilateral kernel |
""")
    st.latex(r"D_{JBU}(p)=\frac{\sum_{q} D_{lr}(q)\,m(q)\,e^{-\|p_\downarrow-q\|^2/2\sigma_s^2}\,e^{-\|I(p)-I_\downarrow(q)\|^2/2\sigma_r^2}}{\sum_q m(q)\,e^{-\|p_\downarrow-q\|^2/2\sigma_s^2}\,e^{-\|I(p)-I_\downarrow(q)\|^2/2\sigma_r^2}}")
    st.latex(r"q_i=\bar a_k^\top I_i+\bar b_k,\quad a_k=(\Sigma_k+\epsilon U)^{-1}\mathrm{cov}_k(I,p),\quad b_k=\bar p_k-a_k^\top\bar\mu_k")

    st.markdown("### Deep learning model — EdgeDepthNet")
    st.graphviz_chart(NET, width="stretch")
    info = model_info()
    if info:
        c = st.columns(4)
        c[0].metric("Parameters", f"{info.get('params', 0):,}")
        c[1].metric("Best iteration", f"{info.get('iter', '–')}")
        c[2].metric("Val mean RMSE", f"{info.get('val_rmse', {}).get('mean', float('nan')):.4f} m")
        c[3].metric("Training time (CPU)", f"{info.get('train_time_min', '–')} min")
        st.caption(f"Trained on: {info.get('trained_on', '')} · {info.get('train_frames', '')} frames")
    else:
        st.warning("No checkpoint found — the app runs the classical fallback.")
    st.markdown("""
* **Inputs:** RGB, initial depth D0 (bilinear of the hole-filled LR grid, or push-pull fill of sparse samples) divided by its mean μ (scale invariance across sensors), measurement mask, and a scale code log₂(s)/3 — one model serves 2×, 4×, 8× and sparse input.
* **Depth-gated RGB fusion:** RGB features pass through a sigmoid gate predicted from depth+RGB features, so colour edges reach the decoder only where the depth branch supports a discontinuity — the architectural defence against texture copying.
* **Residual learning + CSPN:** the decoder predicts a residual on D0, then 6 steps of convolutional spatial propagation with learned 8-neighbour affinities refine the result along edges.
* **Loss:** edge-weighted L1 (×3 inside a 2-px band around GT discontinuities) on refined and coarse outputs + gradient L1 + boundary BCE.
* **Training data:** random 128×160 crops of scene-disjoint NYU train frames, random factor {2,4,8} or 1–10 % sparse, Kinect noise, random / blob / edge-shadow holes, colour jitter and depth-scale jitter.
""")
    st.markdown("### Hybrid fusion — the primary method")
    st.latex(r"z_k(p)=\beta_{k0}+\sum_{f}\beta_{kf}\,\phi_{kf}(p),\qquad w_k(p)=\frac{e^{z_k(p)}}{\sum_j e^{z_j(p)}},\qquad D_{hyb}(p)=\sum_k w_k(p)\,D_k(p)")
    st.markdown("""
Experts *k*: bicubic, guided filter, JBU, EdgeDepthNet. Cue vector φ_k(p):
`B_dl` network boundary probability · `B_cl` classical discontinuity probability · `V` local relative depth variation ·
`E` RGB edge strength · `H` missing-measurement indicator · `T_k` texture-copy score E·(1−B_cl)·|D_k−D_bic|/D_bic ·
`R_k` LR data-consistency residual |↓D_k − D_lr|/D_lr · `A_k` deviation from the median consensus.

Why not average? An average inherits the blur of bicubic at edges and the texture of the RGB-guided filters in flat areas.
The softmax gate is a *mixture of experts*: it is convex (the output never leaves the range of the experts),
differentiable (β is fitted by minimising L1 on validation pixels) and interpretable (each β says how a cue moves trust).

**Edge-aware back-projection (data consistency).** The fused map is corrected towards the measurements, 3 iterations:
""")
    st.latex(r"r_t=M\odot\big(D_{lr}-\downarrow_s D_t\big),\qquad D_{t+1}(p)=D_t(p)+\frac{\sum_q r_t(q)\,M(q)\,f_s\,f_{rgb}\,e^{-(\log D_t(p)-\log \downarrow_s D_t(q))^2/2\sigma_d^2}}{\sum_q M(q)\,f_s\,f_{rgb}\,e^{-(\cdots)^2/2\sigma_d^2}}")
    st.markdown("""
The log-depth term (σ_d = 0.05) prevents a correction measured on the background from leaking onto the foreground,
and M = 0 in holes means no correction is invented there. The network leaves a small systematic bias relative to the
measurements; this classical step removes it. Because raw measurements also carry noise, the correction is scaled by a
gain α = clip(1 − (q − 1)/2.5, 0, 1), where q is the estimated noise level relative to the scale threshold (fitted on
validation): full correction for clean data, none for very noisy data. An optional weighted-median *snapping* step (toggle on the
reconstruction page) sharpens edges further but raises RMSE on NYU validation, so it is off by default.
""")
    fp_path = os.path.join(ROOT, C.FUSION_PARAMS)
    if os.path.exists(fp_path):
        fp = json.load(open(fp_path))
        for key, title in (("with_dl", "Fitted coefficients β (with network)"), ("classical", "Fitted coefficients β (classical fallback)")):
            if key in fp:
                beta = np.asarray(fp[key]["beta"])
                df = pd.DataFrame(beta, columns=["bias"] + FEATURES, index=[METHOD_LABELS[c] for c in fp[key]["candidates"]])
                st.markdown(f"**{title}** — fitted on: {fp[key].get('fitted_on', '')}")
                st.dataframe(df.style.format(precision=2).background_gradient(cmap="coolwarm", axis=None), width="stretch")
    rep = load_json("fusion_fit_report.json")
    if rep:
        st.caption("Validation-pixel MAE of each expert vs. the fused output: " + json.dumps(
            {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "mae_each_expert_on_samples"} for k, v in rep.items()}))
