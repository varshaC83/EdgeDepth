import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ui.common import depth_vis, hero, input_selector, lr_vis, method_options, metrics_frame, range_for, show_images
from edgedepth import METHOD_LABELS
from edgedepth.io_utils import colorize_error
from edgedepth.pipeline import model_available, run_methods


def render():
    hero("Method Comparison", "All seven methods on the identical input, at the identical output size, with one shared colour scale.")
    model_ok = model_available()
    bundle = input_selector("cmp")
    if bundle is None:
        return
    methods = method_options(model_ok)
    if not model_ok:
        st.warning("Deep-learning checkpoint missing — comparing the remaining methods (hybrid uses its classical fallback).")
    if st.button("▶ Run all methods", type="primary", width="stretch"):
        with st.spinner("Reconstructing with every method …"):
            res = run_methods(bundle.rgb, bundle.obs, methods, gt=bundle.gt, track_memory=True)
        st.session_state["cmp_results"] = {"bundle": bundle, "res": res}
        st.session_state["bundle"] = bundle
    data = st.session_state.get("cmp_results")
    if not data:
        st.info("Press **Run all methods**.")
        return
    b, res = data["bundle"], data["res"]
    lo, hi = range_for(b)
    H, W = b.obs.target_hw
    st.caption(f"Output {W}×{H} for every method · colour range {lo:.2f}–{hi:.2f} m (shared) · "
               f"input {'%d× low-res' % b.obs.scale if b.obs.mode == 'lowres' else 'sparse %.2f%%' % (b.obs.mask.mean() * 100)}")
    top = [("RGB guide", b.rgb), ("Input (nearest view)", lr_vis(b.obs, lo, hi))]
    if b.gt is not None:
        top.append(("Ground truth", depth_vis(b.gt, lo, hi)))
    show_images(top, cols=4)
    ok = {m: r for m, r in res.items() if r["depth"] is not None}
    show_images([(f"{METHOD_LABELS[m]} · {r['latency_ms']:.0f} ms", depth_vis(r["depth"], lo, hi)) for m, r in ok.items()], cols=4)
    if b.gt is not None:
        st.markdown("##### Absolute error maps (0 – 0.3 m)")
        show_images([(f"{METHOD_LABELS[m]} · RMSE {r['metrics']['rmse']:.3f}", colorize_error(np.abs(r["depth"] - b.gt), 0.3)) for m, r in ok.items()], cols=4)

    st.markdown("##### Metrics")
    df = metrics_frame(res, b.gt is not None)
    st.dataframe(df.style.format(precision=4), width="stretch", hide_index=True)
    if b.gt is None:
        st.caption("Reference metrics unavailable (no ground truth for real low-resolution input).")

    st.markdown("##### Zoom on a region (boundary detail)")
    c1, c2, c3 = st.columns(3)
    zy = c1.slider("Centre row", 0, H - 1, H // 2, key="z_y")
    zx = c2.slider("Centre column", 0, W - 1, W // 2, key="z_x")
    zs = c3.select_slider("Window", [32, 48, 64, 96], value=64, key="z_s")
    y0, x0 = int(np.clip(zy - zs // 2, 0, H - zs)), int(np.clip(zx - zs // 2, 0, W - zs))
    crop = lambda im: np.ascontiguousarray(np.repeat(np.repeat(im[y0:y0 + zs, x0:x0 + zs], 4, 0), 4, 1))  # noqa: E731
    z = [("RGB", crop(b.rgb))] + ([("GT", crop(depth_vis(b.gt, lo, hi)))] if b.gt is not None else [])
    z += [(METHOD_LABELS[m], crop(depth_vis(r["depth"], lo, hi))) for m, r in ok.items()]
    show_images(z, cols=5)

    st.markdown("##### Depth profile along the selected row (behaviour at discontinuities)")
    fig = go.Figure()
    xs = np.arange(W)
    if b.gt is not None:
        fig.add_trace(go.Scatter(x=xs, y=b.gt[zy], name="Ground truth", line=dict(color="white", width=3)))
    for m, r in ok.items():
        fig.add_trace(go.Scatter(x=xs, y=r["depth"][zy], name=METHOD_LABELS[m],
                                 line=dict(width=3 if m == "hybrid" else 1.3, dash=None if m in ("hybrid", "deep_learning") else "dot")))
    fig.add_vrect(x0=x0, x1=x0 + zs, fillcolor="#4FD1C5", opacity=0.08, line_width=0)
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="column (px)", yaxis_title="depth (m)",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, width="stretch")
