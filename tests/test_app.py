"""Headless UI tests with Streamlit's AppTest: every page renders and every main button works.

    python -m pytest -q tests/test_app.py
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from streamlit.testing.v1 import AppTest  # noqa: E402

PAGES = ["🏠 Home", "🧩 Depth Reconstruction", "🆚 Method Comparison", "📊 Evaluation Dashboard", "🏗️ Technical Architecture", "⚠️ Failure Analysis"]


def app():
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=180)
    at.run()
    return at


def goto(at, page):
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    goto(app(), page)


def test_reconstruction_runs_with_metrics():
    at = goto(app(), PAGES[1])
    at.button[0].click().run()
    assert not at.exception and not at.error
    labels = [m.label for m in at.metric]
    assert "Processing time" in labels and any("RMSE" in l for l in labels)


def test_reconstruction_every_method_and_sparse():
    at = goto(app(), PAGES[1])
    at.radio(key="rec_mode").set_value("Sparse samples").run()
    for m in at.selectbox[1].options:
        at.selectbox[1].set_value(m).run()
        at.button[0].click().run()
        assert not at.exception and not at.error, m


def test_comparison_and_dashboard():
    at = goto(app(), PAGES[2])
    at.button[0].click().run()
    assert not at.exception and not at.error
    assert len(at.dataframe) >= 1
    goto(at, PAGES[3])
    assert len(at.dataframe) >= 1


def test_failure_cases_run():
    at = goto(app(), PAGES[5])
    for s in at.selectbox[0].options:
        at.selectbox[0].set_value(s).run()
        at.button[0].click().run()
        assert not at.exception and not at.error, s
