"""EdgeDepth Streamlit application.

    streamlit run app.py
"""
import os
import sys

import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

st.set_page_config(page_title="EdgeDepth · RGB-guided depth upsampling", page_icon="🧊", layout="wide",
                   initial_sidebar_state="auto")

from ui import page_arch, page_compare, page_eval, page_failure, page_home, page_reconstruct  # noqa: E402
from ui.common import CSS  # noqa: E402
from edgedepth.pipeline import model_available  # noqa: E402

PAGES = {
    "🏠 Home": page_home,
    "🧩 Depth Reconstruction": page_reconstruct,
    "🆚 Method Comparison": page_compare,
    "📊 Evaluation Dashboard": page_eval,
    "🏗️ Technical Architecture": page_arch,
    "⚠️ Failure Analysis": page_failure,
}

st.markdown(CSS, unsafe_allow_html=True)
with st.sidebar:
    st.markdown("## 🧊 EdgeDepth")
    st.caption("P4 · Edge-preserving depth map upsampling and refinement")
    page = st.radio("Navigate", list(PAGES), label_visibility="collapsed")
    st.divider()
    ok = model_available()
    st.markdown(f"**Model:** {'✅ EdgeDepthNet loaded' if ok else '⚠️ checkpoint missing → classical fallback'}")
    st.caption("CPU inference · outputs at the RGB resolution (≤ 640 px side)")

try:
    PAGES[page].render()
except Exception as e:  # last-resort error state: never show a raw traceback to judges
    st.error(f"Something went wrong: {e}")
    st.caption("Try another input or reload the page. Details were logged to the server console.")
    import traceback

    traceback.print_exc()
