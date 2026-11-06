# EdgeDepth — Design Document

**RGB-Guided Edge-Preserving Depth Map Upsampling and Refinement**
Computer Vision Hackathon 2026 · Problem P4 · Team `<TeamID>`

---

## 1. Introduction

Commodity depth sensors (structured light, time of flight, stereo, LiDAR) produce depth maps that are much
coarser than the colour images captured beside them, and that contain noise and missing values. Many
downstream tasks — 3-D reconstruction, AR occlusion, robot grasping, bokeh rendering — need depth at the RGB
resolution with geometrically correct object boundaries. EdgeDepth is a complete system that upsamples and
refines such depth maps using the RGB image as a guide, and that measures, explains and visualises how well it
does so.

## 2. Problem definition

Given an RGB image *I* ∈ ℝ^{H×W×3} and a depth observation *O* that is either

* a low-resolution grid of size (H/s)×(W/s), s ∈ {2, 4, 8}, or
* a sparse set of samples on the H×W grid,

where some values are missing (0) and the rest are corrupted by sensor noise, estimate a dense depth map
*D* ∈ ℝ^{H×W} (metres) that minimises the error to the true depth while keeping depth discontinuities sharp
and correctly placed, and without transferring RGB texture into smooth surfaces.

Challenges (from the P4 brief): missing values, sensor noise, foreground/background discontinuities,
texture-copying artefacts, inaccurate boundaries, multiple upsampling factors.

## 3. Objectives

| # | Objective (P4 requirement) | Where it is met |
|---|---|---|
| O1 | Nearest / bilinear / bicubic baselines | `classical/interpolation.py` |
| O2 | Joint bilateral filtering, guided filtering | `classical/filters.py` |
| O3 | Edge detection, boundary-aware refinement, depth-discontinuity detection | `classical/edges.py`, `classical/boundary.py` |
| O4 | Noise removal and hole filling | `classical/denoise.py`, `classical/holes.py` |
| O5 | Depth upsampling network with RGB guidance, adapted on depth data | `model/`, `scripts/train.py` |
| O6 | Hybrid combining interpolation + edge-aware filtering + DL; boundary-preserving refinement; no oversmoothing / texture copying | `hybrid/fusion.py` |
| O7 | 2×, 4×, 8× | all methods; one network for all factors |
| O8 | Compare interpolation, classical refinement, DL, hybrid on unseen data | `scripts/evaluate.py`, app sections C/D |
| O9 | Show behaviour at boundaries and discontinuities | boundary RMSE, edge F1, sharpness, profiles, zoom |
| O10 | Metrics: RMSE, MAE, PSNR, SSIM, edge/boundary accuracy, per-factor, latency, memory | `metrics.py`, `results/` |
| O11 | Deployed application | `app.py`, `Dockerfile` |

## 4. Proposed methodology

EdgeDepth treats every reconstruction method as an *expert* that maps the same observation to an H×W depth
map. Classical experts are strong in particular regimes (bicubic in smooth regions, RGB-guided filters at
colour-aligned edges), a learned expert brings a data-driven prior for holes, noise and boundaries, and none is
best everywhere. The hybrid therefore estimates, *per pixel*, which expert to trust from observable cues, fuses
them with a softmax mixture, and finally enforces consistency with the actual measurements through an edge-aware
(joint trilateral) back-projection, so that corrections never cross depth edges or enter holes.

## 5. System architecture

See [`architecture.md`](architecture.md) for the diagrams (Mermaid source in [`architecture.mmd`](architecture.mmd)).

```
RGB ─────────────┬───────────────► JBU, guided filter, EdgeDepthNet, cues
Observation ─► missing detection ─► noise estimate ─► (denoise) ─► hole filling ─► experts
experts + cues ─► softmax mixture ─► edge-aware back-projection (raw measurements) ─► D_hybrid ─► metrics / UI
```

Components: I/O and validation (`io_utils.py`), degradation simulator (`degradation.py`), classical modules
(`classical/`), network (`model/`), hybrid (`hybrid/`), orchestration (`pipeline.py`), metrics (`metrics.py`),
Streamlit front-end (`app.py`, `ui/`), scripts (`scripts/`).

## 6. Classical CV implementation

**Interpolation.** For low-res input the LR grid is first hole-filled, then resampled with OpenCV
(`INTER_NEAREST`, `INTER_LINEAR`, `INTER_CUBIC`; half-pixel-centre convention, identical to the degradation
model). Bicubic overshoot below the minimum valid depth is clamped. For sparse input the same three kernels are
generalised to scattered data: nearest valid sample (Euclidean distance transform), Delaunay piecewise-linear,
and Clough–Tocher C¹ cubic (SciPy); outside the convex hull nearest is used.

**Joint bilateral upsampling** (Kopf et al. 2007):

  D(p) = Σ_q D_lr(q) m(q) exp(−‖p↓ − q‖²/2σ_s²) exp(−‖I(p) − I↓(q)‖²/2σ_r²) / Σ_q (same weights)

with q over a 5×5 LR neighbourhood, I↓ the area-downsampled guide and m the measurement mask (missing values get
zero weight, so JBU also fills small holes). For sparse input the filter runs on the HR grid with a radius
adapted to the density. σ_s = 0.5 LR px and σ_r = 0.2 were selected by grid search on the validation split
(`scripts/tune_classical.py`, `results/classical_tuning.json`).

**Guided filter** (He et al. 2013, colour version): q_i = ā_kᵀ I_i + b̄_k with
a_k = (Σ_k + εU)⁻¹ cov_k(I, p), b_k = p̄_k − a_kᵀ μ_k, applied to the bilinear upsampling *p*. A 3×3 linear
system is solved per pixel. Validation-selected: r = max(1, round(s/4)), ε = 1e-5.

**Edge detection.** Sobel colour gradient magnitude (max over channels, normalised to the 99th percentile) and a
colour Canny (per-channel Canny with median-based thresholds, OR-combined, so iso-luminant colour edges are not
lost — a bug we found with a unit test).

**Depth-discontinuity detection.** g = ‖∇ log D‖ (Sobel/8), i.e. the relative depth change per pixel, which is
invariant to absolute distance; edges = g > 0.03 with non-maximum thinning; soft probability 1 − exp(−(g/0.03)²).
Texture edges = RGB Canny edges farther than 2 px from any depth discontinuity.

**Noise removal.** σ̂ from Immerkær's estimator (√(π/2)/6 · mean |L ∗ D| with the 3×3 kernel
[[1,−2,1],[−2,4,−2],[1,−2,1]], over the 50 % smoothest valid pixels); denoising =
3×3 median followed by a bilateral filter with σ_r = 3σ̂ (steps ≫ noise survive). Missing pixels stay missing.

**Missing-depth detection and hole filling.** Missing = 0, non-finite or outside [0.1, 20] m. Push-pull
normalised convolution: a mask-weighted 2×2 pyramid is built until no hole remains, then coarse estimates are
bilinearly pulled back only into missing pixels. Telea inpainting and nearest fill are available alternatives.

**Boundary-aware snapping (optional).** In a band (1 px dilation) where the discontinuity probability exceeds 0.5, each
pixel is replaced by the joint weighted median of its 7×7 neighbourhood with weights
exp(−‖I_p − I_q‖²/2·0.08²)·exp(−‖p − q‖²/2·2²), band neighbours down-weighted by ½. Unlike a mean, the median
returns an existing value from one side of the edge, eliminating flying pixels.

## 7. Deep learning architecture

**EdgeDepthNet** (`model/network.py`, 723,882 parameters):

* Inputs at H×W: ImageNet-normalised RGB; D0/μ where D0 = bicubic of the hole-filled LR grid (or push-pull fill of
  sparse samples) and μ = mean(D0) (scale normalisation across sensors and scenes); measurement mask;
  scale code log₂(s)/3 (s = 1/√density for sparse).
* RGB and depth encoders at 1, 1/2, 1/4 (16/32/64 channels), **depth-gated fusion** at each level
  f = f_d + Conv([f_d, σ(W_g[f_d, f_r]) ⊙ f_r]), 1/8 bottleneck (96 channels, 2 residual blocks), U-Net decoder.
* Heads: residual r (zero-initialised), boundary logit, 8 affinities. D_coarse = D0/μ + r.
* **CSPN** (6 iterations, zero-initialised = identity): D ← κ₀D + Σ_{i=1..8} κ_i shift_i(D),
  κ_i = tanh(a_i)/Σ|tanh(a_j)|, κ₀ = 1 − Σκ_i. Output D_dl = μ · D_ref.

**Training** (`scripts/train.py`, CPU, 2 threads): random 128×160 crops of train frames with horizontal flip,
colour jitter and global depth scaling ×[0.7, 1.4]; random degradation per crop (factor 2/4/8 or 1–10 % sparse
with p = 0.2; Kinect noise ×0.3–2.5 with p = 0.5; dropout up to 15 % with p = 0.5; 1–3 blob holes with p = 0.3;
edge dropout with p = 0.3). Loss:

  L = L1_w(D_ref) + RMSE_w(D_ref) + 0.5·L1_w(D_coarse) + L_grad(D_ref) + 0.1·BCE(edge)

with weights 3 inside a 2-px band around GT discontinuities. Adam, lr 1e-3, cosine decay, batch 8, gradient
clipping 1.0. Every 250 iterations the model is validated on the full validation frames (2×/4×/8×, clean and
noisy) and the best checkpoint is kept. The shipped checkpoint is from iteration 5000 (the last one — validation RMSE
was still decreasing), mean validation RMSE 0.0427 m, 51.6 min of CPU training. Curve: `results/training_log.json`
(also plotted in the app's Evaluation Dashboard).

A first training run (lr 2e-3, bilinear D0, L1-only loss, random CSPN initialisation) was stopped after 1000
iterations because its validation RMSE had barely moved (0.0603 m at iteration 250, 0.0595 m at 1000) and at 2× its randomly initialised CSPN made the coarse output worse; its log is kept in
`results/train_stdout_run1_aborted.txt`. The fixes (bicubic D0, identity initialisation, RMSE term) are the ones
described above.

## 8. Hybrid reconstruction

**Experts** k ∈ {bicubic, guided filter, JBU, EdgeDepthNet}. In the hybrid path the classical experts receive
the denoised observation when the noise test below fires; the network always receives the raw observation (it
learned to denoise).

**Cues** φ_k(p) ∈ ℝ⁸ (relative quantities are scaled by 20 and clipped to [0, 3]):

| Cue | Definition | Intended effect |
|---|---|---|
| B_dl | sigmoid of the network boundary head | where real edges are |
| B_cl | 1 − exp(−(‖∇log C‖/0.03)²), C = median of experts | classical edge evidence |
| V | std₅ₓ₅(D0)/mean₅ₓ₅(D0) | planar vs. complex region |
| E | normalised Sobel colour gradient | RGB edge strength |
| H | 1 − (G_σ * M)/max, σ = s | absence of nearby measurements |
| T_k | E (1 − B_cl) |D_k − D_bic|/D_bic | texture-copy suspicion for expert k |
| R_k | |↓_s D_k − D_lr|/D_lr on measured LR pixels | disagreement with measurements |
| A_k | |D_k − C|/C | outlier expert |

**Fusion.** z_k(p) = β_k0 + Σ_f β_kf φ_kf(p), w_k(p) = softmax_k z_k(p), D_fused(p) = Σ_k w_k(p) D_k(p).

*Mathematical reasoning.* (i) The output is a convex combination, so it never leaves [min_k D_k, max_k D_k] — the
fusion cannot invent overshoots. (ii) If the experts' errors e_k at p were known, the optimal convex weights put
all mass on argmin |e_k|; the softmax of a linear function of cues is the maximum-entropy (multinomial-logit)
model of "which expert is best given the cues", i.e. a mixture of experts (Jacobs et al. 1991) with a linear
gating network. (iii) Because the mixture is differentiable in β, we fit β by minimising
mean |D_fused − D_gt| + 10⁻⁴‖β_cues‖² over ~1.7 M validation pixels (29 frames × 10 degradations × 6 000 pixels,
boundary pixels over-sampled ×3) with Adam (`scripts/fit_fusion.py`). The 36 parameters cannot overfit that many
samples, and the test split is never used. (iv) A plain average (all β = 0) is a special case; the ablation
compares against it and against constant learned weights (no cues) to show the value of adaptivity.

**Edge-aware back-projection.** The fused map D_0 = D_fused is corrected for T = 3 iterations:

  r_t(q) = M(q) (D_lr(q) − ↓_s D_t(q)),
  D_{t+1}(p) = D_t(p) + Σ_q r_t(q) M(q) f_s(p,q) f_rgb(p,q) f_d(p,q) / Σ_q M(q) f_s f_rgb f_d,

with f_s, f_rgb the JBU kernels and f_d = exp(−(log D_t(p) − log ↓_s D_t(q))²/2·0.05²). This is classical
iterative back-projection (Irani & Peleg 1991) with a joint *trilateral* upsampling of the residual: the log-depth
term keeps a correction measured on one surface from leaking onto another across a discontinuity, the RGB term
aligns corrections with object boundaries, and M = 0 prevents invented corrections inside holes. It always uses the
raw (not denoised) observation, because only measured values may act as constraints. On validation it removed a
systematic residual bias of the network (see §11). For sparse input the same update runs on the HR grid.

*Noise-adaptive gain.* Raw measurements are only as trustworthy as their noise: on validation, the full
correction helped up to Kinect-noise ×1 but hurt from ×2–×4 on (it re-injects noise). The applied correction is
therefore D = D_fused + α (D_BP − D_fused) with α = clip(1 − (q − 1)/2.5, 0, 1), where q is the estimated relative
noise divided by the scale threshold below. The two constants were read off a validation sweep of α ∈ {0, .25, .5,
.75, 1} against q (optimal α fell monotonically from 1 at q < 1 to 0 at q > 3.5); the test split was not used.

**Optional snapping.** A joint weighted median in a 1-px band around discontinuities (§6) can be enabled in the UI.
It raises edge F1 but on NYU validation it increased RMSE (a pixel snapped to the wrong side costs a full depth
step), so it is off by default — see ablation X5.

**Noise-adaptive pre-processing.** The Immerkær estimator (3×3 Laplacian-difference kernel, smoothest 50 % of
pixels) gives σ̂; the classical experts receive a denoised observation when σ̂/median(D) exceeds a scale-dependent
threshold (0.22 %, 0.35 %, 0.85 % for 2×/4×/8×, set on validation). The threshold grows with s because scene
structure leaks into the estimate on coarse grids: at 8× clean and noisy inputs overlap, a limitation we measured
and document.

**Fallback.** Without a checkpoint `hybrid_reconstruct` uses a β set fitted for the three classical experts.

The fitted coefficients are shown in the app (Technical Architecture) and stored in
`checkpoints/fusion_params.json`. Interpretation of the fitted values is given in §11.

## 9. Dataset

* **NYU Depth V2** (Silberman et al. 2012). The official servers were unreachable from our build environment;
  we used a public GitHub mirror with 179 RGB-D frames from 10 NYU scenes, each 640×480 RGB + dense 16-bit depth
  in millimetres (NYU's filled ground truth). Downsampled to 320×240 (RGB: area; depth: nearest).
* **Scene-disjoint split**: train 126 frames (basement_1, study_1/2/3/4/7) · val 29 (basement_2, study_8) ·
  test 24 (basement_3, study_9). `data/nyu/manifest.json` lists every frame.
* **Cross-dataset frames** (never used for training or tuning): TUM RGB-D fr1 (Kinect v1), SUN RGB-D, Redwood
  (PrimeSense), from the Open3D sample-data releases, converted to millimetres; their raw depth has 13–19 % holes,
  which are excluded from metrics.
* KITTI and Middlebury servers were also unreachable; this is listed as a limitation.

## 10. Experimental methodology

* **Degradations** (`degradation.py`): masked box-downsampling (sensor integration); Kinect axial noise
  σ(z) = k(0.0012 + 0.0019(z − 0.4)²) m (Nguyen et al. 2012); random dropout; elliptical blob holes; dropout on
  depth edges (occlusion shadows); sparse random sampling.
* **Benchmark protocol** (`scripts/evaluate.py`): 24 test frames × {2×, 4×, 8×} × {clean, noisy + missing
  (k = 1, 5 % dropout, 50 % edge dropout)} + 5 % sparse = 168 observations, fixed seeds, identical for all methods.
  All methods output 320×240.
* **Metrics** (`metrics.py`): RMSE, MAE (m); PSNR (peak 10 m); SSIM (data range = GT range); boundary RMSE (2-px
  band around GT discontinuities); edge F1 (2-px tolerance); edge sharpness (ratio of mean ‖∇log D‖ on GT edges);
  latency (wall clock, no tracking); peak memory (tracemalloc ∪ RSS sampling, separate pass, 4×).
* **Tuning discipline**: classical hyper-parameters, checkpoint selection and fusion β use only the validation
  split.
* **Hardware**: 2 CPU cores, 8 GB RAM, no GPU.

## 11. Evaluation and results

### 11.1 Overall (mean over all 168 test observations)

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

### 11.2 Performance across upsampling factors

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

### 11.3 Per condition

<!-- AUTO:detailed -->
**2x, clean** (n = 24 frames)

| Method | RMSE (m) ↓ | MAE (m) ↓ | PSNR (dB) ↑ | SSIM ↑ | Boundary RMSE (m) ↓ | Edge F1 ↑ | Edge sharpness (→1) | Latency (ms) ↓ |
|---|---|---|---|---|---|---|---|---|
| Nearest neighbour | 0.0491 | 0.0152 | 48.33 | 0.9764 | 0.1258 | 0.949 | 0.809 | 0 |
| Bilinear | 0.0404 | 0.0139 | 49.98 | 0.9833 | 0.1044 | 0.894 | 0.698 | 0 |
| Bicubic | 0.0347 | 0.0125 | 51.32 | 0.9870 | 0.0893 | 0.938 | 0.810 | 0 |
| Joint bilateral (JBU) | 0.0427 | 0.0144 | 49.54 | 0.9817 | 0.1107 | 0.904 | 0.700 | 48 |
| Guided filter | 0.0452 | 0.0165 | 48.95 | 0.9786 | 0.1172 | 0.916 | 0.655 | 22 |
| Deep learning (EdgeDepthNet) | 0.0291 | 0.0113 | 52.75 | 0.9888 | 0.0719 | 0.960 | 0.932 | 165 |
| Hybrid (fusion + edge-aware refinement) | **0.0259** | **0.0085** | **53.99** | **0.9929** | **0.0651** | **0.968** | 0.939 | 371 |

**2x, noisy + missing** (n = 24 frames)

| Method | RMSE (m) ↓ | MAE (m) ↓ | PSNR (dB) ↑ | SSIM ↑ | Boundary RMSE (m) ↓ | Edge F1 ↑ | Edge sharpness (→1) | Latency (ms) ↓ |
|---|---|---|---|---|---|---|---|---|
| Nearest neighbour | 0.0763 | 0.0299 | 44.56 | 0.9394 | 0.1920 | 0.916 | 0.593 | 1 |
| Bilinear | 0.0692 | 0.0268 | 45.43 | 0.9557 | 0.1761 | 0.863 | 0.526 | 1 |
| Bicubic | 0.0681 | 0.0265 | 45.57 | 0.9536 | 0.1722 | 0.896 | 0.596 | 1 |
| Joint bilateral (JBU) | 0.0676 | 0.0220 | 45.59 | 0.9598 | 0.1747 | 0.932 | 0.666 | 51 |
| Guided filter | 0.0703 | 0.0277 | 45.29 | 0.9551 | 0.1790 | 0.862 | 0.501 | 22 |
| Deep learning (EdgeDepthNet) | 0.0487 | 0.0177 | 48.25 | 0.9750 | 0.1231 | 0.935 | 0.816 | 105 |
| Hybrid (fusion + edge-aware refinement) | **0.0461** | **0.0171** | **48.86** | **0.9760** | **0.1153** | **0.940** | 0.823 | 371 |

**4x, clean** (n = 24 frames)

| Method | RMSE (m) ↓ | MAE (m) ↓ | PSNR (dB) ↑ | SSIM ↑ | Boundary RMSE (m) ↓ | Edge F1 ↑ | Edge sharpness (→1) | Latency (ms) ↓ |
|---|---|---|---|---|---|---|---|---|
| Nearest neighbour | 0.0852 | 0.0311 | 43.58 | 0.9290 | 0.2176 | 0.918 | 0.581 | 0 |
| Bilinear | 0.0691 | 0.0282 | 45.38 | 0.9529 | 0.1777 | 0.676 | 0.447 | 0 |
| Bicubic | 0.0607 | 0.0252 | 46.44 | 0.9577 | 0.1546 | 0.744 | 0.522 | 0 |
| Joint bilateral (JBU) | 0.0760 | 0.0297 | 44.61 | 0.9481 | 0.1946 | 0.801 | 0.463 | 49 |
| Guided filter | 0.0706 | 0.0294 | 45.18 | 0.9510 | 0.1813 | 0.843 | 0.438 | 21 |
| Deep learning (EdgeDepthNet) | 0.0460 | 0.0185 | 48.69 | 0.9730 | 0.1139 | 0.919 | 0.798 | 89 |
| Hybrid (fusion + edge-aware refinement) | **0.0430** | **0.0164** | **49.37** | **0.9769** | **0.1077** | **0.922** | 0.797 | 337 |

**4x, noisy + missing** (n = 24 frames)

| Method | RMSE (m) ↓ | MAE (m) ↓ | PSNR (dB) ↑ | SSIM ↑ | Boundary RMSE (m) ↓ | Edge F1 ↑ | Edge sharpness (→1) | Latency (ms) ↓ |
|---|---|---|---|---|---|---|---|---|
| Nearest neighbour | 0.1238 | 0.0547 | 40.46 | 0.8728 | 0.2884 | 0.771 | 0.399 | 0 |
| Bilinear | 0.1129 | 0.0508 | 41.31 | 0.9084 | 0.2654 | 0.596 | 0.315 | 0 |
| Bicubic | 0.1108 | 0.0486 | 41.46 | 0.9052 | 0.2602 | 0.646 | 0.361 | 1 |
| Joint bilateral (JBU) | 0.1061 | 0.0406 | 41.73 | 0.9141 | 0.2636 | 0.782 | 0.444 | 51 |
| Guided filter | 0.1129 | 0.0512 | 41.30 | 0.9092 | 0.2656 | 0.680 | 0.310 | 22 |
| Deep learning (EdgeDepthNet) | 0.0760 | 0.0301 | 44.33 | 0.9473 | 0.1850 | 0.866 | 0.659 | 218 |
| Hybrid (fusion + edge-aware refinement) | **0.0723** | **0.0292** | **44.79** | **0.9475** | **0.1757** | **0.867** | 0.665 | 429 |

**8x, clean** (n = 24 frames)

| Method | RMSE (m) ↓ | MAE (m) ↓ | PSNR (dB) ↑ | SSIM ↑ | Boundary RMSE (m) ↓ | Edge F1 ↑ | Edge sharpness (→1) | Latency (ms) ↓ |
|---|---|---|---|---|---|---|---|---|
| Nearest neighbour | 0.1366 | 0.0623 | 39.52 | 0.8284 | 0.3247 | 0.588 | 0.329 | 0 |
| Bilinear | 0.1159 | 0.0561 | 41.09 | 0.8982 | 0.2752 | 0.248 | 0.238 | 0 |
| Bicubic | 0.1045 | 0.0505 | 41.96 | 0.9009 | 0.2525 | 0.363 | 0.280 | 0 |
| Joint bilateral (JBU) | 0.1221 | 0.0573 | 40.68 | 0.8964 | 0.2893 | 0.648 | 0.295 | 54 |
| Guided filter | 0.1177 | 0.0581 | 40.97 | 0.8977 | 0.2771 | 0.660 | 0.242 | 21 |
| Deep learning (EdgeDepthNet) | 0.0827 | 0.0361 | 43.42 | 0.9333 | 0.1996 | 0.817 | 0.581 | 133 |
| Hybrid (fusion + edge-aware refinement) | **0.0782** | **0.0335** | **44.00** | **0.9355** | **0.1887** | **0.823** | 0.591 | 392 |

**8x, noisy + missing** (n = 24 frames)

| Method | RMSE (m) ↓ | MAE (m) ↓ | PSNR (dB) ↑ | SSIM ↑ | Boundary RMSE (m) ↓ | Edge F1 ↑ | Edge sharpness (→1) | Latency (ms) ↓ |
|---|---|---|---|---|---|---|---|---|
| Nearest neighbour | 0.1890 | 0.0986 | 36.84 | 0.7832 | 0.3820 | 0.475 | 0.222 | 0 |
| Bilinear | 0.1750 | 0.0930 | 37.61 | 0.8591 | 0.3564 | 0.161 | 0.165 | 0 |
| Bicubic | 0.1716 | 0.0885 | 37.78 | 0.8533 | 0.3528 | 0.224 | 0.190 | 0 |
| Joint bilateral (JBU) | 0.1641 | 0.0774 | 38.08 | 0.8545 | 0.3628 | 0.567 | 0.273 | 50 |
| Guided filter | 0.1749 | 0.0938 | 37.61 | 0.8614 | 0.3555 | 0.456 | 0.167 | 23 |
| Deep learning (EdgeDepthNet) | 0.1188 | 0.0564 | 40.39 | **0.9016** | 0.2672 | 0.721 | 0.471 | 146 |
| Hybrid (fusion + edge-aware refinement) | **0.1147** | **0.0543** | **40.79** | 0.9016 | **0.2579** | **0.742** | 0.488 | 395 |

**Sparse input, 5 % of pixels** (n = 24 frames)

| Method | RMSE (m) ↓ | MAE (m) ↓ | PSNR (dB) ↑ | SSIM ↑ | Boundary RMSE (m) ↓ | Edge F1 ↑ | Edge sharpness (→1) | Latency (ms) ↓ |
|---|---|---|---|---|---|---|---|---|
| Nearest neighbour | 0.1270 | 0.0413 | 40.06 | 0.8812 | 0.3193 | 0.856 | 0.632 | 4 |
| Bilinear | 0.0922 | 0.0343 | 42.88 | 0.9310 | 0.2295 | 0.743 | 0.471 | 32 |
| Bicubic | 0.0946 | 0.0376 | 42.55 | 0.9236 | 0.2281 | 0.794 | 0.572 | 42 |
| Joint bilateral (JBU) | 0.1116 | 0.0474 | 41.28 | 0.9095 | 0.2727 | 0.741 | 0.402 | 297 |
| Guided filter | 0.0914 | 0.0353 | 42.96 | 0.9321 | 0.2273 | 0.806 | 0.449 | 52 |
| Deep learning (EdgeDepthNet) | 0.0814 | 0.0321 | 43.61 | 0.9447 | 0.2002 | 0.871 | 0.633 | 106 |
| Hybrid (fusion + edge-aware refinement) | **0.0745** | **0.0258** | **44.46** | **0.9531** | **0.1865** | **0.896** | 0.688 | 1550 |
<!-- /AUTO:detailed -->

### 11.4 Efficiency

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

### 11.5 Cross-dataset generalisation

<!-- AUTO:cross -->
**8× upsampling, mean over TUM RGB-D / SUN RGB-D / Redwood frames**

| Method | RMSE (m) ↓ | MAE (m) ↓ | PSNR (dB) ↑ | SSIM ↑ | Boundary RMSE (m) ↓ | Edge F1 ↑ | Edge sharpness (→1) |
|---|---|---|---|---|---|---|---|
| Nearest neighbour | 0.1317 | 0.0397 | 38.02 | 0.9387 | 0.4747 | 0.556 | 0.325 |
| Bilinear | 0.1162 | 0.0348 | 39.06 | 0.9577 | 0.4231 | 0.302 | 0.205 |
| Bicubic | 0.1097 | 0.0343 | 39.56 | 0.9542 | 0.3984 | 0.440 | 0.236 |
| Joint bilateral (JBU) | 0.1219 | 0.0358 | 38.73 | 0.9575 | 0.4419 | 0.631 | 0.247 |
| Guided filter | 0.1180 | 0.0363 | 38.97 | 0.9575 | 0.4250 | 0.637 | 0.216 |
| Deep learning (EdgeDepthNet) | 0.0859 | 0.0246 | 42.09 | 0.9754 | 0.3220 | 0.861 | 0.537 |
| Hybrid (fusion + edge-aware refinement) | **0.0804** | **0.0224** | 42.69 | 0.9763 | **0.3024** | **0.865** | 0.562 |

**4× upsampling**

| Method | RMSE (m) ↓ | MAE (m) ↓ | PSNR (dB) ↑ | SSIM ↑ | Boundary RMSE (m) ↓ | Edge F1 ↑ | Edge sharpness (→1) |
|---|---|---|---|---|---|---|---|
| Nearest neighbour | 0.0901 | 0.0209 | 41.37 | 0.9728 | 0.3564 | **0.944** | 0.520 |
| Bilinear | 0.0809 | 0.0197 | 42.22 | 0.9757 | 0.3215 | 0.724 | 0.380 |
| Bicubic | 0.0759 | 0.0196 | 42.76 | 0.9749 | 0.2957 | 0.789 | 0.443 |
| Joint bilateral (JBU) | 0.0840 | 0.0204 | 41.92 | 0.9751 | 0.3340 | 0.826 | 0.388 |
| Guided filter | 0.0824 | 0.0205 | 42.08 | 0.9753 | 0.3262 | 0.849 | 0.376 |
| Deep learning (EdgeDepthNet) | 0.0578 | 0.0147 | 45.50 | 0.9859 | 0.2278 | 0.933 | 0.721 |
| Hybrid (fusion + edge-aware refinement) | **0.0561** | **0.0133** | 45.78 | 0.9866 | **0.2248** | 0.932 | 0.738 |
<!-- /AUTO:cross -->

### 11.6 Ablation

See [`ablation_study.md`](ablation_study.md).

<!-- AUTO:ablation -->
| Variant | RMSE ↓ | MAE ↓ | PSNR ↑ | SSIM ↑ | B-RMSE ↓ | Edge F1 ↑ | Sharpness | Latency (ms) |
|---|---|---|---|---|---|---|---|---|
| A1_interpolation_only | 0.0917 | 0.0420 | 44.09 | 0.9263 | 0.2136 | 0.635 | 0.460 | 0 |
| A2_interpolation_guided_filter | 0.0986 | 0.0461 | 43.22 | 0.9255 | 0.2293 | 0.736 | 0.386 | 21 |
| A3_dl_only_backbone | 0.0712 | 0.0319 | 45.63 | 0.9418 | 0.1693 | 0.846 | 0.655 | 95 |
| A4a_dl_learned_refinement(CSPN) | 0.0669 | 0.0283 | 46.31 | 0.9532 | 0.1601 | 0.870 | 0.709 | 93 |
| A4b_dl_cspn_edge_aware_backprojection | 0.0634 | 0.0265 | 46.97 | 0.9550 | 0.1518 | 0.877 | 0.718 | 262 |
| A4c_dl_cspn_weighted_median_snapping | 0.0758 | 0.0299 | 44.86 | 0.9480 | 0.1851 | 0.880 | 0.682 | 137 |
| A5_full_hybrid | 0.0634 | **0.0265** | 46.97 | 0.9551 | 0.1517 | 0.877 | 0.717 | 337 |
| X1_hybrid_without_backprojection | 0.0668 | 0.0283 | 46.31 | 0.9532 | 0.1601 | 0.869 | 0.708 | 171 |
| X2_hybrid_constant_weights(no_cues) | **0.0634** | 0.0265 | **46.97** | **0.9551** | **0.1517** | 0.877 | 0.717 | 338 |
| X3_uniform_average_of_experts | 0.0889 | 0.0423 | 44.41 | 0.9370 | 0.2059 | 0.757 | 0.468 | 172 |
| X4_hybrid_classical_only(fallback) | 0.0885 | 0.0387 | 44.57 | 0.9333 | 0.2099 | 0.799 | 0.538 | 258 |
| X5_hybrid_plus_weighted_median_snapping | 0.0720 | 0.0279 | 45.39 | 0.9502 | 0.1759 | **0.887** | 0.687 | 398 |

**2x** (clean + noisy/missing)

| Variant | RMSE ↓ | MAE ↓ | PSNR ↑ | SSIM ↑ | B-RMSE ↓ | Edge F1 ↑ |
|---|---|---|---|---|---|---|
| A1_interpolation_only | 0.0514 | 0.0195 | 48.44 | 0.9703 | 0.1308 | 0.917 |
| A2_interpolation_guided_filter | 0.0577 | 0.0221 | 47.12 | 0.9669 | 0.1481 | 0.889 |
| A3_dl_only_backbone | 0.0441 | 0.0173 | 49.46 | 0.9754 | 0.1114 | 0.940 |
| A4a_dl_learned_refinement(CSPN) | 0.0389 | 0.0145 | 50.50 | 0.9819 | 0.0975 | 0.948 |
| A4b_dl_cspn_edge_aware_backprojection | 0.0360 | **0.0128** | **51.42** | **0.9844** | **0.0902** | **0.955** |
| A4c_dl_cspn_weighted_median_snapping | 0.0529 | 0.0168 | 47.70 | 0.9744 | 0.1358 | 0.945 |
| A5_full_hybrid | 0.0360 | 0.0128 | 51.42 | 0.9844 | 0.0902 | 0.954 |
| X1_hybrid_without_backprojection | 0.0389 | 0.0145 | 50.51 | 0.9819 | 0.0975 | 0.947 |
| X2_hybrid_constant_weights(no_cues) | **0.0360** | 0.0128 | 51.42 | 0.9844 | 0.0902 | 0.954 |
| X3_uniform_average_of_experts | 0.0504 | 0.0193 | 48.52 | 0.9729 | 0.1285 | 0.894 |
| X4_hybrid_classical_only(fallback) | 0.0504 | 0.0181 | 48.89 | 0.9740 | 0.1284 | 0.939 |
| X5_hybrid_plus_weighted_median_snapping | 0.0503 | 0.0152 | 48.21 | 0.9770 | 0.1294 | 0.948 |

**4x** (clean + noisy/missing)

| Variant | RMSE ↓ | MAE ↓ | PSNR ↑ | SSIM ↑ | B-RMSE ↓ | Edge F1 ↑ |
|---|---|---|---|---|---|---|
| A1_interpolation_only | 0.0857 | 0.0369 | 43.95 | 0.9314 | 0.2074 | 0.695 |
| A2_interpolation_guided_filter | 0.0918 | 0.0403 | 43.24 | 0.9301 | 0.2234 | 0.761 |
| A3_dl_only_backbone | 0.0659 | 0.0283 | 45.79 | 0.9484 | 0.1603 | 0.860 |
| A4a_dl_learned_refinement(CSPN) | 0.0610 | 0.0243 | 46.51 | 0.9601 | 0.1495 | 0.893 |
| A4b_dl_cspn_edge_aware_backprojection | 0.0577 | **0.0228** | 47.08 | 0.9622 | 0.1417 | 0.895 |
| A4c_dl_cspn_weighted_median_snapping | 0.0693 | 0.0258 | 45.33 | 0.9551 | 0.1729 | 0.905 |
| A5_full_hybrid | 0.0577 | 0.0228 | 47.08 | 0.9622 | 0.1417 | 0.895 |
| X1_hybrid_without_backprojection | 0.0609 | 0.0243 | 46.51 | 0.9602 | 0.1494 | 0.892 |
| X2_hybrid_constant_weights(no_cues) | **0.0577** | 0.0228 | **47.08** | **0.9622** | **0.1417** | 0.895 |
| X3_uniform_average_of_experts | 0.0846 | 0.0380 | 44.17 | 0.9392 | 0.2034 | 0.754 |
| X4_hybrid_classical_only(fallback) | 0.0829 | 0.0339 | 44.39 | 0.9394 | 0.2039 | 0.833 |
| X5_hybrid_plus_weighted_median_snapping | 0.0655 | 0.0241 | 45.91 | 0.9578 | 0.1636 | **0.909** |

**8x** (clean + noisy/missing)

| Variant | RMSE ↓ | MAE ↓ | PSNR ↑ | SSIM ↑ | B-RMSE ↓ | Edge F1 ↑ |
|---|---|---|---|---|---|---|
| A1_interpolation_only | 0.1381 | 0.0695 | 39.87 | 0.8771 | 0.3027 | 0.294 |
| A2_interpolation_guided_filter | 0.1463 | 0.0760 | 39.29 | 0.8795 | 0.3163 | 0.558 |
| A3_dl_only_backbone | 0.1035 | 0.0500 | 41.63 | 0.9014 | 0.2362 | 0.738 |
| A4a_dl_learned_refinement(CSPN) | 0.1008 | 0.0463 | 41.91 | 0.9175 | 0.2334 | 0.769 |
| A4b_dl_cspn_edge_aware_backprojection | 0.0965 | 0.0439 | 42.39 | 0.9185 | 0.2234 | 0.782 |
| A4c_dl_cspn_weighted_median_snapping | 0.1051 | 0.0471 | 41.54 | 0.9145 | 0.2464 | 0.791 |
| A5_full_hybrid | 0.0964 | 0.0439 | 42.39 | 0.9185 | 0.2233 | 0.782 |
| X1_hybrid_without_backprojection | 0.1007 | 0.0462 | 41.91 | 0.9175 | 0.2333 | 0.768 |
| X2_hybrid_constant_weights(no_cues) | **0.0964** | **0.0439** | **42.40** | **0.9186** | **0.2233** | 0.782 |
| X3_uniform_average_of_experts | 0.1319 | 0.0695 | 40.53 | 0.8990 | 0.2858 | 0.622 |
| X4_hybrid_classical_only(fallback) | 0.1321 | 0.0642 | 40.44 | 0.8866 | 0.2974 | 0.626 |
| X5_hybrid_plus_weighted_median_snapping | 0.1002 | 0.0445 | 42.06 | 0.9160 | 0.2348 | **0.805** |
<!-- /AUTO:ablation -->

## 12. Results discussion

*Numbers quoted here are from `results/summary.json` / `results/ablation.json` at the time of writing; the tables
above are regenerated automatically.*

1. **The hybrid is the most accurate method in every tested condition.** Averaged over all 168 test observations it
   reduces RMSE from 0.0921 m (bicubic, best interpolation) to 0.0650 m (−29 %), MAE from 0.0413 to 0.0264 m (−36 %),
   boundary RMSE from 0.216 to 0.157 m (−27 %) and raises edge F1 from 0.66 to 0.88. It is also better than the network
   alone (RMSE 0.0690 m) at every factor and condition, including 5 % sparse input (0.0745 vs 0.0814 m) and the unseen
   TUM / SUN RGB-D / Redwood sensors (8×: 0.0804 vs 0.0859 m; bicubic 0.1097 m).
2. **Classical RGB-guided filters do not beat bicubic on RMSE on NYU.** JBU and the guided filter raise edge F1
   (sharper edges) but their RMSE is slightly worse than bicubic: NYU's RGB contains many texture and shading edges
   that are not depth edges, and the guided filter copies them. They help most under noise and holes (JBU has the
   lowest MAE of the classical methods in the noisy condition because its mask weighting ignores missing values).
3. **Where the gains come from (ablation).** The learned RGB–depth fusion is the largest single gain (A1 → A3:
   RMSE 0.0917 → 0.0712), learned CSPN refinement adds 6 % (→ 0.0669), and the classical edge-aware back-projection
   another 5 % (→ 0.0634) together with lower boundary RMSE. Weighted-median snapping raises edge F1 (0.887) but
   costs RMSE (0.0720), confirming the validation decision to keep it optional.
4. **The adaptive fusion is decisive only without the network.** With the trained network available the fitted gate
   puts ≈ 99 % of the weight on EdgeDepthNet everywhere (its intercept is ≈ 6 logits above the others and the cue
   coefficients are ≈ 0); constant weights (X2) give identical results. We report this honestly: on NYU-like data the
   network is better than every classical expert in every cue configuration we can observe, so the gate learns to
   select it. In the classical fallback the same fusion is genuinely adaptive — it excludes the guided filter, trusts
   JBU less where it disagrees with the measurements (β_R = −0.91) and more inside holes (β_H = +0.09), and beats the
   best classical expert (X4: 0.0885 m vs 0.0917 m). A uniform average (X3, 0.0889 m) is much worse than the fitted
   fusion with the network, which is why the brief's warning against unjustified averaging is well founded.
5. **Upsampling factor.** All errors grow with the factor; the relative advantage of the hybrid over bicubic is
   30 % at 2×, 33 % at 4× and 30 % at 8× (RMSE). Edge F1 shows the clearest separation at 8× (0.78 vs 0.29), where
   interpolation can no longer represent the boundaries at all.
6. **Cost.** For low-resolution input the hybrid takes ≈ 0.38 s per 320×240 frame at 4× on 2 CPU cores (four
   experts + cues + three back-projection iterations; network alone ≈ 0.15 s) and ≈ 21 MB of extra memory; a 640×480
   frame takes ≈ 1.2–1.4 s. Sparse input is slower (≈ 1.4 s at 320×240, ≈ 5.4 s at 640×480) because the sparse JBU and
   back-projection use wide windows — still inside the 10 s interactive budget, but the slowest path.

## 13. Limitations

* **Data scale.** 126 training frames from 6 indoor NYU scenes, trained for 51 min on CPU. The validation curve
  was still improving when training stopped; a GPU run on the full NYU raw set would very likely improve the network.
* **Fusion gate.** The linear-softmax gate collapses to the network when the network is available (see §12.4); a
  non-linear gate or experts with complementary errors would be needed for the fusion itself to add accuracy.
* **Noise estimation at 8×.** On coarse grids scene structure and noise cannot be separated reliably, so the
  noise-adaptive gain and the denoising decision are less accurate at 8×.
* **Extreme conditions.** Very high noise (×8 the Kinect model, outside the training range) makes the network and
  hybrid worse than the guided filter (F3); below ~1 % sparse density and inside large holes errors grow sharply
  (F1, F2); RGB–depth misalignment shifts boundaries for every RGB-guided method (F5).
* **Texture copying is reduced, not eliminated.** On a synthetic plane with textured RGB, JBU's error is 1.66× larger
  on texture edges than elsewhere and the hybrid's 1.32× (F4) — the RGB term of the back-projection lets some texture
  through.
* **Ground truth.** NYU depth is itself inpainted where the Kinect failed; KITTI and Middlebury were not evaluated
  because their servers were unreachable from the build environment.
* **Hardware dependence.** Latency and memory are measured on one 2-core CPU machine.

## 14. Ethics

Indoor depth maps can reveal private layouts; the application keeps uploads in memory only and stores nothing.
Reconstructed depth in holes is a plausible guess, not a measurement — it must not be used for safety-critical
decisions without validation. Public datasets are used under their respective licences; the NYU frames were
obtained from a public mirror of the official dataset.

## 15. Future scope

* Train on the full NYU Depth V2 (≈ 47 k raw frames) and KITTI depth completion on a GPU; evaluate on Middlebury.
* Learn the fusion gate as a small CNN on the cue maps (non-linear gating) and add uncertainty outputs.
* Self-calibration of RGB–depth misalignment (estimate a flow field before guidance).
* Temporal consistency for video depth streams; quantised (INT8) network for mobile deployment.
