"""Unit tests for degradation, classical modules, model, hybrid and metrics.

    python -m pytest -q tests
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from edgedepth import METHODS  # noqa: E402
from edgedepth.classical import (boundary_refine, canny_edges, denoise_depth, depth_discontinuities, detect_missing,  # noqa: E402
                                 estimate_noise_sigma, fill_holes, guided_filter_color, interpolate, joint_bilateral_upsample,
                                 push_pull_fill, sobel_magnitude, texture_edges)
from edgedepth.degradation import DegradationConfig, degrade, masked_area_downsample, observation_from_lowres  # noqa: E402
from edgedepth.io_utils import InputError, load_depth  # noqa: E402
from edgedepth.metrics import all_metrics, edge_f1, mae, psnr, rmse  # noqa: E402
from edgedepth.pipeline import run_method  # noqa: E402


def step_scene(H=64, W=80):
    """Two planes with a vertical depth step, RGB edge aligned with the step."""
    gt = np.full((H, W), 2.0, np.float32)
    gt[:, W // 2:] = 4.0
    rgb = np.zeros((H, W, 3), np.uint8)
    rgb[:, : W // 2] = (200, 60, 60)
    rgb[:, W // 2:] = (40, 80, 200)
    return rgb, gt


def test_degradation_shapes_and_mask():
    _, gt = step_scene()
    for s in (2, 4, 8):
        obs = degrade(gt, DegradationConfig(scale=s, missing_rate=0.2, seed=1))
        assert obs.values.shape == (64 // s, 80 // s)
        assert np.all(obs.values[obs.mask == 0] == 0)
        assert 0.6 < obs.mask.mean() < 0.95
    sp = degrade(gt, DegradationConfig(mode="sparse", sparse_density=0.05, seed=1))
    assert sp.values.shape == gt.shape and 0.02 < sp.mask.mean() < 0.09


def test_area_downsample_ignores_zeros():
    d = np.array([[1, 0], [3, 0]], np.float32)
    lr, m = masked_area_downsample(d, 2)
    assert lr[0, 0] == pytest.approx(2.0) and m[0, 0] == 1


def test_noise_is_depth_dependent_and_estimated():
    gt = np.full((128, 128), 4.0, np.float32)
    obs = degrade(gt, DegradationConfig(scale=2, noise_level=2.0, seed=3))
    est = estimate_noise_sigma(obs.values, obs.mask)
    true = 2.0 * (0.0012 + 0.0019 * 3.6 ** 2)
    assert 0.5 * true < est < 1.6 * true
    den = denoise_depth(obs.values, obs.mask)
    assert np.std(den - 4.0) < np.std(obs.values - 4.0)


def test_hole_filling_and_missing_detection():
    _, gt = step_scene()
    m = np.ones_like(gt)
    m[10:20, 10:30] = 0
    v = gt * m
    assert detect_missing(v)[15, 15] and not detect_missing(v)[0, 0]
    for meth in ("push_pull", "nearest", "telea"):
        f = fill_holes(v, m, meth)
        assert np.all(f[10:20, 10:30] > 1.5) and np.allclose(f[m > 0], gt[m > 0], atol=1e-3 if meth != "telea" else 0.05)
    assert np.allclose(push_pull_fill(gt, np.ones_like(gt)), gt)


def test_interpolations_differ_and_have_target_size():
    rgb, gt = step_scene()
    obs = degrade(gt, DegradationConfig(scale=4, seed=0))
    outs = {k: interpolate(obs, k) for k in ("nearest", "bilinear", "bicubic")}
    for o in outs.values():
        assert o.shape == gt.shape
    assert not np.allclose(outs["nearest"], outs["bilinear"]) and not np.allclose(outs["bilinear"], outs["bicubic"])
    # nearest keeps only original values
    assert set(np.unique(outs["nearest"])) <= set(np.unique(obs.values))


def test_edges_and_discontinuities():
    rgb, gt = step_scene()
    dd = depth_discontinuities(gt)
    assert dd[:, 38:42].any() and not dd[:, :30].any()
    assert canny_edges(rgb)[:, 36:44].any()
    assert sobel_magnitude(gt)[:, 39:41].max() > 0
    assert not texture_edges(rgb, gt)[:, 36:44].any()  # RGB edge coincides with the depth edge


def test_rgb_guidance_sharpens_step():
    rgb, gt = step_scene()
    obs = degrade(gt, DegradationConfig(scale=8, seed=0))
    bil = interpolate(obs, "bilinear")
    jbu = joint_bilateral_upsample(obs, rgb)
    assert rmse(jbu, gt) < rmse(bil, gt)
    gf = guided_filter_color(rgb.astype(np.float32) / 255, bil, 2, 1e-4)
    assert gf.shape == gt.shape and rmse(gf, gt) < rmse(bil, gt)
    br = boundary_refine(bil, rgb)
    assert rmse(br, gt) < rmse(bil, gt)


def test_metrics_basic():
    _, gt = step_scene()
    assert rmse(gt, gt) == 0 and mae(gt, gt) == 0 and psnr(gt, gt) > 100
    m = all_metrics(gt, gt)
    assert m["ssim"] == pytest.approx(1.0) and m["edge_f1"] == pytest.approx(1.0) and m["edge_sharpness"] == pytest.approx(1.0)
    pred = gt + 0.1
    assert rmse(pred, gt) == pytest.approx(0.1, abs=1e-5)
    # GT holes are ignored
    gt2 = gt.copy()
    gt2[:5] = 0
    assert rmse(np.where(gt2 > 0, gt2, 99), gt2) == 0
    blurred = np.convolve(gt[0], np.ones(9) / 9, mode="same")
    assert edge_f1(np.tile(blurred, (gt.shape[0], 1)).astype(np.float32), gt) <= 1.0


@pytest.mark.parametrize("method", METHODS)
def test_every_method_runs_on_all_inputs(method):
    rgb, gt = step_scene(64, 80)
    for cfg in (DegradationConfig(scale=2, seed=0), DegradationConfig(scale=8, noise_level=1, missing_rate=0.1, seed=0),
                DegradationConfig(mode="sparse", sparse_density=0.05, seed=0)):
        out = run_method(method, rgb, degrade(gt, cfg))
        assert out.shape == gt.shape and np.isfinite(out).all() and out.min() > 0


def test_methods_are_distinct():
    rgb, gt = step_scene(64, 80)
    obs = degrade(gt, DegradationConfig(scale=4, noise_level=1, seed=2))
    outs = [run_method(m, rgb, obs) for m in METHODS]
    for i in range(len(outs)):
        for j in range(i + 1, len(outs)):
            assert not np.allclose(outs[i], outs[j]), (METHODS[i], METHODS[j])


def test_real_lowres_wrapper():
    lr = np.full((30, 40), 2.0, np.float32)
    obs = observation_from_lowres(lr, (240, 320))
    assert obs.scale == 8 and obs.values.shape == (30, 40)


def test_bad_inputs():
    with pytest.raises(InputError):
        load_depth(b"not an image", 1000.0, "x.png")
    import cv2

    ok, buf = cv2.imencode(".png", np.zeros((10, 10), np.uint16))
    with pytest.raises(InputError):
        load_depth(buf.tobytes(), 1000.0, "zeros.png")


def test_backprojection_gain_and_consistency():
    from edgedepth.hybrid.fusion import backprojection_gain, edge_aware_backprojection, _area_down

    assert backprojection_gain({"rel_noise": 0.0, "threshold": 0.002}) == 1.0
    assert backprojection_gain({"rel_noise": 0.02, "threshold": 0.002}) == 0.0
    g = [backprojection_gain({"rel_noise": q * 0.002, "threshold": 0.002}) for q in (0.5, 1.5, 2.5, 3.5)]
    assert all(a >= b for a, b in zip(g, g[1:]))
    rgb, gt = step_scene()
    obs = degrade(gt, DegradationConfig(scale=4, seed=0))
    d0 = interpolate(obs, "bilinear") + 0.2  # biased estimate
    d1 = edge_aware_backprojection(d0, obs, rgb)
    r0 = np.abs(_area_down(d0, 4) - obs.values).mean()
    r1 = np.abs(_area_down(d1, 4) - obs.values).mean()
    assert r1 < 0.2 * r0 and rmse(d1, gt) < rmse(d0, gt)


def test_hybrid_details_and_fallback():
    from edgedepth.hybrid.fusion import hybrid_reconstruct

    rgb, gt = step_scene()
    obs = degrade(gt, DegradationConfig(scale=4, noise_level=1, seed=0))
    out, det = hybrid_reconstruct(rgb, obs, return_details=True)
    w = np.stack(list(det["weights"].values()))
    assert np.allclose(w.sum(0), 1, atol=1e-4) and 0 <= det["bp_gain"] <= 1
    out_cl = hybrid_reconstruct(rgb, obs, use_dl=False)
    assert out_cl.shape == gt.shape and np.isfinite(out_cl).all()
