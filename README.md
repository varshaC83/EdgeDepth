---
title: EdgeDepth
emoji: 🧊
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: RGB-guided edge-preserving depth upsampling (P4)
---

# EdgeDepth — RGB-Guided Edge-Preserving Depth Map Upsampling and Refinement

**Computer Vision Hackathon 2026 · Problem P4 · Team `<TeamID>`**

EdgeDepth turns low-resolution or sparse, noisy, hole-ridden depth maps into dense high-resolution depth
maps that keep object boundaries and depth discontinuities sharp, using the RGB image as a guide without
copying its texture. Its primary method is a **hybrid reconstruction** that fuses classical interpolation,
edge-aware filters and a lightweight CNN with a per-pixel, cue-driven mixture of experts.

* Live app: `<public URL — see Deployment>`
* Evaluation file: [`eval_P4_TeamID.md`](eval_P4_TeamID.md) · Design document: [`docs/design_document.md`](docs/design_document.md)
* Architecture: [`docs/architecture.md`](docs/architecture.md) ([Mermaid source](docs/architecture.mmd)) · Ablation: [`docs/ablation_study.md`](docs/ablation_study.md) · Failure analysis: [`docs/failure_analysis.md`](docs/failure_analysis.md)
* Demo script: [`docs/demo_script.md`](docs/demo_script.md) · Viva Q&A: [`docs/viva_qa.md`](docs/viva_qa.md)

## Problem

Depth sensors deliver depth at low resolution with noise, missing values (absorbing / specular surfaces,
sensor shadows at occlusions) and blurred boundaries, while the paired RGB camera is sharp. Plain
upsampling blurs boundaries and creates *flying pixels*; naive RGB guidance copies colour texture into flat
surfaces. P4 asks for 2×/4×/8× upsampling with hole filling, denoising and boundary preservation, compared
across interpolation, classical refinement, deep learning and hybrid methods.

## Features

* **7 methods, one interface** (`edgedepth.pipeline.run_method`): nearest, bilinear, bicubic, joint bilateral
  upsampling, guided filter, EdgeDepthNet (CNN), hybrid. Identical input, identical output size.
* **Classical CV modules**: interpolation (grid + scattered), JBU, colour guided filter, Sobel / colour Canny,
  scale-invariant depth-discontinuity detection, Immerkær noise estimation + edge-preserving denoising,
  missing-depth detection, push-pull hole filling, weighted-median boundary snapping.
* **EdgeDepthNet** (0.72 M parameters): dual RGB/depth encoders, depth-gated RGB fusion (anti texture copying),
  U-Net decoder, residual + boundary + affinity heads, CSPN refinement. One model for 2×, 4×, 8× and sparse input.
  Trained on CPU on scene-disjoint NYU Depth V2 frames.
* **Hybrid fusion**: softmax mixture of 4 experts driven by 8 per-pixel cues (boundary probability, RGB edges,
  depth variation, missing data, texture-copy score, data-consistency residual, consensus deviation), fitted on
  the validation split, followed by an edge-aware (joint trilateral) back-projection that enforces consistency with
  the raw measurements; classical fallback when the checkpoint is missing.
* **Realistic degradations**: box-downsampling, Kinect depth-dependent noise, random / blob / edge-shadow holes,
  sparse sampling.
* **Streamlit app** (dark, responsive): Home, Depth Reconstruction, Method Comparison (shared colour scale,
  zoom, boundary profile), Evaluation Dashboard (live + benchmark + ablation + mini-benchmark), Technical
  Architecture, Failure Analysis. Upload your own RGB-D pair or real low-res depth.
* **Honest evaluation**: reference metrics only when ground truth exists; every number in the docs is generated
  from `results/*.json` by `scripts/make_report.py`.

## Results (unseen NYU Depth V2 test split)

Mean over 24 test frames × {2×, 4×, 8×} × {clean, noisy + missing} plus 5 % sparse input:

<!-- AUTO:overall -->
| Method | RMSE (m) ↓ | MAE (m) ↓ | PSNR (dB) ↑ | SSIM ↑ | Boundary RMSE (m) ↓ | Edge F1 ↑ | Edge sharpness (→1) | Latency (ms) ↓ |
|---|---|---|---|---|---|---|---|---|
| Nearest neighbour | 0.1125 | 0.0476 | 41.91 | 0.8872 | 0.2643 | 0.782 | 0.509 | 1 |
| Bilinear | 0.0964 | 0.0433 | 43.38 | 0.9269 | 0.2264 | 0.597 | 0.408 | 5 |
| Bicubic | 0.0921 | 0.0413 | 43.87 | 0.9259 | 0.2157 | 0.658 | 0.476 | 6 |
| Joint bilateral (JBU) | 0.0986 | 0.0413 | 43.07 | 0.9235 | 0.2383 | 0.768 | 0.463 | 86 |
| Guided filter | 0.0976 | 0.0446 | 43.18 | 0.9264 | 0.2290 | 0.746 | 0.395 | 26 |
| Deep learning (EdgeDepthNet) | 0.0690 | 0.0289 | 45.92 | 0.9520 | 0.1659 | 0.870 | 0.698 | 137 |
| Hybrid (fusion + edge-aware refinement) | **0.0650** | **0.0264** | **46.61** | **0.9548** | **0.1567** | **0.880** | 0.713 | 549 |
<!-- /AUTO:overall -->

Per upsampling factor (mean of clean and noisy + missing):

<!-- AUTO:by_scale -->
| Method | RMSE 2x | RMSE 4x | RMSE 8x | B-RMSE 2x | B-RMSE 4x | B-RMSE 8x | Edge F1 2x | Edge F1 4x | Edge F1 8x |
|---|---|---|---|---|---|---|---|---|---|
| Nearest neighbour | 0.0627 | 0.1045 | 0.1628 | 0.1589 | 0.2530 | 0.3534 | 0.933 | 0.845 | 0.532 |
| Bilinear | 0.0548 | 0.0910 | 0.1455 | 0.1403 | 0.2216 | 0.3158 | 0.878 | 0.636 | 0.204 |
| Bicubic | 0.0514 | 0.0857 | 0.1381 | 0.1308 | 0.2074 | 0.3027 | 0.917 | 0.695 | 0.294 |
| Joint bilateral (JBU) | 0.0552 | 0.0911 | 0.1431 | 0.1427 | 0.2291 | 0.3261 | 0.918 | 0.792 | 0.608 |
| Guided filter | 0.0577 | 0.0918 | 0.1463 | 0.1481 | 0.2234 | 0.3163 | 0.889 | 0.761 | 0.558 |
| Deep learning (EdgeDepthNet) | 0.0389 | 0.0610 | 0.1008 | 0.0975 | 0.1495 | 0.2334 | 0.948 | 0.893 | 0.769 |
| Hybrid (fusion + edge-aware refinement) | 0.0360 | 0.0577 | 0.0964 | 0.0902 | 0.1417 | 0.2233 | 0.954 | 0.895 | 0.782 |
<!-- /AUTO:by_scale -->

Latency and memory (CPU, 320×240 output):

<!-- AUTO:efficiency -->
| Method | Mean latency at 4× (ms) | Peak memory at 4× (MB, approx.) |
|---|---|---|
| Nearest neighbour | 0 | 0.6 |
| Bilinear | 0 | 0.6 |
| Bicubic | 0 | 0.6 |
| Joint bilateral (JBU) | 50 | 4.8 |
| Guided filter | 21 | 16.2 |
| Deep learning (EdgeDepthNet) | 153 | 2.3 |
| Hybrid (fusion + edge-aware refinement) | 383 | 20.7 |
<!-- /AUTO:efficiency -->

**Key findings** (details in the [design document](docs/design_document.md#12-results-discussion) and
[ablation](docs/ablation_study.md)):
* The hybrid has the lowest RMSE, MAE and boundary RMSE in **every** factor/condition, including 5 % sparse input and
  unseen sensors (TUM, SUN RGB-D, Redwood); overall −29 % RMSE and −27 % boundary RMSE vs bicubic.
* Gains come from the learned RGB–depth network (−22 % RMSE vs bicubic), its CSPN refinement (−6 %) and the classical
  edge-aware back-projection (−5 %). With the network available the learned gate selects it almost exclusively, so the
  cue-driven fusion itself adds accuracy only in the classical fallback (−3.5 % vs bicubic) — reported honestly.
* JBU and the guided filter sharpen edges but do not beat bicubic on RMSE on NYU (texture edges ≠ depth edges).
* Measured failure modes: < 1 % sparsity, large holes, noise ×8, texture copying (reduced, not eliminated), RGB–depth
  misalignment — see [failure analysis](docs/failure_analysis.md).

Detailed per-condition tables, cross-dataset results (TUM RGB-D, SUN RGB-D, Redwood) and the ablation are in
[`docs/results_tables.md`](docs/results_tables.md) and the [evaluation file](eval_P4_TeamID.md).

## Technology stack

Python 3.11 · NumPy · OpenCV · SciPy · scikit-image · PyTorch (CPU) · Streamlit · Plotly · Matplotlib ·
NYU Depth V2 (+ TUM RGB-D, SUN RGB-D, Redwood samples) · Docker / Hugging Face Spaces.

## Installation

```bash
git clone <your-repo-url> EdgeDepth && cd EdgeDepth
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                        # CPU PyTorch wheels
```

## Run the app (one command)

```bash
streamlit run app.py                 # http://localhost:8501
# or with Docker
docker build -t edgedepth . && docker run -p 7860:7860 edgedepth   # http://localhost:7860
```

The repository ships the trained checkpoint (`checkpoints/edgedepthnet.pt`, ~3 MB), the fitted fusion
coefficients, the 24 unseen test frames and cross-dataset samples, so the app and the evaluation run
out of the box. If the checkpoint is removed, the app shows a warning and the hybrid uses its classical
fallback.

## Reproduce everything

```bash
python scripts/prepare_dataset.py        # clone the NYU mirror, build scene-disjoint train/val/test (320x240)
python scripts/prepare_samples.py        # demo samples (needs the Open3D sample zips, see script docstring)
python scripts/tune_classical.py         # JBU / guided-filter parameters on VAL      (~3 min)
python scripts/train.py --iters 5000     # EdgeDepthNet on CPU                       (~60 min, 2 cores)
python scripts/fit_fusion.py             # hybrid fusion coefficients on VAL         (~7 min)
python scripts/evaluate.py               # benchmark on TEST -> results/summary.json  (~5 min)
python scripts/ablation.py               # ablation on TEST  -> results/ablation.json
python scripts/failure_cases.py          # failure study     -> results/failure_cases.json
python scripts/make_report.py            # regenerate every table in the docs + eval file
python -m pytest -q tests                # unit + headless UI tests
python scripts/infer.py --rgb a.png --depth lr.png --out hr.png   # command-line inference
```

GPU users (Colab/Kaggle): `python scripts/train.py --device cuda --iters 20000 --batch 16`. The shipped
checkpoint was trained on CPU only (see `results/training_log.json`).

## Architecture

```
app.py, ui/            Streamlit front-end (6 sections)
edgedepth/
  degradation.py       Observation, low-res / sparse / noise / holes simulation
  classical/           interpolation, filters (JBU, guided), edges, denoise, holes, boundary refinement
  model/               EdgeDepthNet, losses, dataset, inference
  hybrid/fusion.py     experts, cues, softmax mixture, edge-aware back-projection, fallback
  pipeline.py          run_method / run_methods (+ ablation variants)
  metrics.py           RMSE, MAE, PSNR, SSIM, boundary RMSE, edge F1, sharpness, latency, memory
scripts/               data prep, tuning, training, fusion fitting, evaluation, ablation, failures, report
docs/                  design document, architecture (+ Mermaid), ablation, failure analysis, demo, viva
results/               JSON/CSV results and figures produced by the scripts
checkpoints/           edgedepthnet.pt, fusion_params.json
data/                  nyu/test (unseen), samples (demo + cross-dataset)
```

See [`docs/architecture.md`](docs/architecture.md) for diagrams of the system, data flow, network and fusion.

## Deployment

The app is a standard Docker Space (front matter at the top of this README). To publish on Hugging Face Spaces:

1. Create a Space → SDK **Docker** → blank.
2. `git remote add space https://huggingface.co/spaces/<user>/edgedepth && git push space main`
   (binary files ≤ 10 MB need no LFS; the checkpoint is ~3 MB).
3. The container listens on port 7860 and exposes `/_stcore/health`. Cold start ≈ 1–2 min; a full hybrid
   reconstruction takes < 2 s on the free 2-vCPU tier.

Alternatives: Streamlit Community Cloud (point it at `app.py`; `requirements.txt` already uses CPU wheels),
Render / Railway (use the Dockerfile; `$PORT` is honoured). No secrets or API keys are used.

## Limitations

* Trained on 126 frames from 6 NYU scenes (CPU-only budget); generalisation to outdoor LiDAR (KITTI) is not
  evaluated. Middlebury/KITTI servers were not reachable from the build environment.
* Assumes pixel-aligned RGB and depth; misalignment degrades every RGB-guided method (failure case F5).
* Below ~1 % sparse density and inside large holes there is no evidence; the network fills in plausible but
  possibly wrong geometry (F1, F2).
* NYU "ground truth" is itself inpainted in regions the Kinect missed, so metrics there measure agreement with
  a colourisation-based fill, not true geometry.
* Latency/memory are measured on a 2-core CPU and vary by machine.

## Ethics

Depth maps of indoor spaces can reveal private room layouts; the app processes uploads in memory and stores
nothing. Upsampled depth is an *estimate* — hallucinated surfaces in holes look plausible — so it must not be
used for safety-critical decisions (e.g. robot navigation, medical or legal measurement) without validation
against real sensor data. The datasets used are public research datasets with their own licences
(NYU Depth V2, TUM RGB-D, SUN RGB-D, Redwood).

## References

Kopf et al., *Joint Bilateral Upsampling*, SIGGRAPH 2007 · He, Sun, Tang, *Guided Image Filtering*, TPAMI 2013 ·
Silberman et al., *NYU Depth V2*, ECCV 2012 · Cheng, Wang, Yang, *CSPN*, ECCV 2018 · Nguyen, Izadi, Lovell,
*Modeling Kinect Sensor Noise*, 3DIMPVT 2012 · Jacobs et al., *Adaptive mixtures of local experts*, 1991.
