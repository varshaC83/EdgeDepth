<USN_1>,<USN_2>,<USN_3>,<USN_4>

# Evaluation — P4: Edge-Preserving Depth Map Upsampling and Refinement (Team <TeamID>)

Primary method: **Hybrid** (cue-driven fusion of bicubic, guided filter, JBU and EdgeDepthNet + noise-adaptive edge-aware back-projection). All numbers below were produced by `scripts/evaluate.py`, `scripts/ablation.py` and `scripts/failure_cases.py` on the unseen, scene-disjoint NYU Depth V2 test split (24 frames, 320×240), on 2 CPU cores, no GPU. Regenerate with `python scripts/make_report.py`.

Conditions: *clean* = box-downsampled GT; *noisy + missing* = Kinect noise model ×1, 5 % random dropout, 50 % dropout on depth edges; *sparse* = 5 % random samples. Averages are over all frames and conditions unless stated.

## Headline (hybrid vs bicubic, mean over 2×/4×/8× and all conditions)

| Metric | Bicubic | Hybrid | Change |
|---|---|---|---|
| RMSE (m) | 0.0921 | 0.0650 | -29.5 % |
| MAE (m) | 0.0413 | 0.0264 | -36.2 % |
| PSNR (dB) | 43.87 | 46.61 | +6.3 % |
| SSIM | 0.9259 | 0.9548 | +3.1 % |
| Boundary RMSE (m) | 0.2157 | 0.1567 | -27.3 % |
| Edge F1 | 0.658 | 0.880 | +33.7 % |

## 1. RMSE and MAE (depth accuracy)

Mean over all test frames, scales and conditions:

| Method | RMSE (m) ↓ | MAE (m) ↓ |
|---|---|---|
| Nearest neighbour | 0.1125 | 0.0476 |
| Bilinear | 0.0964 | 0.0433 |
| Bicubic | 0.0921 | 0.0413 |
| Joint bilateral (JBU) | 0.0986 | 0.0413 |
| Guided filter | 0.0976 | 0.0446 |
| Deep learning (EdgeDepthNet) | 0.0690 | 0.0289 |
| Hybrid (fusion + edge-aware refinement) | **0.0650** | **0.0264** |

## 2. PSNR and SSIM

PSNR peak = 10 m; SSIM data range = per-frame GT range.

| Method | PSNR (dB) ↑ | SSIM ↑ |
|---|---|---|
| Nearest neighbour | 41.91 | 0.8872 |
| Bilinear | 43.38 | 0.9269 |
| Bicubic | 43.87 | 0.9259 |
| Joint bilateral (JBU) | 43.07 | 0.9235 |
| Guided filter | 43.18 | 0.9264 |
| Deep learning (EdgeDepthNet) | 45.92 | 0.9520 |
| Hybrid (fusion + edge-aware refinement) | **46.61** | **0.9548** |

## 3. Edge preservation and boundary accuracy

Boundary RMSE = RMSE within 2 px of GT depth discontinuities; Edge F1 = F1 of discontinuity maps (2-px tolerance); edge sharpness = mean |∇log D| on GT edges relative to GT (1.0 ideal).

| Method | Boundary RMSE (m) ↓ | Edge F1 ↑ | Edge sharpness (→1) |
|---|---|---|---|
| Nearest neighbour | 0.2643 | 0.782 | 0.509 |
| Bilinear | 0.2264 | 0.597 | 0.408 |
| Bicubic | 0.2157 | 0.658 | 0.476 |
| Joint bilateral (JBU) | 0.2383 | 0.768 | 0.463 |
| Guided filter | 0.2290 | 0.746 | 0.395 |
| Deep learning (EdgeDepthNet) | 0.1659 | 0.870 | 0.698 |
| Hybrid (fusion + edge-aware refinement) | **0.1567** | **0.880** | 0.713 |

## 4. Performance across upsampling factors

Mean of clean and noisy + missing conditions:

| Method | RMSE 2x | RMSE 4x | RMSE 8x | B-RMSE 2x | B-RMSE 4x | B-RMSE 8x | Edge F1 2x | Edge F1 4x | Edge F1 8x |
|---|---|---|---|---|---|---|---|---|---|
| Nearest neighbour | 0.0627 | 0.1045 | 0.1628 | 0.1589 | 0.2530 | 0.3534 | 0.933 | 0.845 | 0.532 |
| Bilinear | 0.0548 | 0.0910 | 0.1455 | 0.1403 | 0.2216 | 0.3158 | 0.878 | 0.636 | 0.204 |
| Bicubic | 0.0514 | 0.0857 | 0.1381 | 0.1308 | 0.2074 | 0.3027 | 0.917 | 0.695 | 0.294 |
| Joint bilateral (JBU) | 0.0552 | 0.0911 | 0.1431 | 0.1427 | 0.2291 | 0.3261 | 0.918 | 0.792 | 0.608 |
| Guided filter | 0.0577 | 0.0918 | 0.1463 | 0.1481 | 0.2234 | 0.3163 | 0.889 | 0.761 | 0.558 |
| Deep learning (EdgeDepthNet) | 0.0389 | 0.0610 | 0.1008 | 0.0975 | 0.1495 | 0.2334 | 0.948 | 0.893 | 0.769 |
| Hybrid (fusion + edge-aware refinement) | 0.0360 | 0.0577 | 0.0964 | 0.0902 | 0.1417 | 0.2233 | 0.954 | 0.895 | 0.782 |

Full per-condition tables:

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

## 5. Processing latency and memory consumption

| Method | Mean latency at 4× (ms) | Peak memory at 4× (MB, approx.) |
|---|---|---|
| Nearest neighbour | 0 | 0.6 |
| Bilinear | 0 | 0.6 |
| Bicubic | 0 | 0.6 |
| Joint bilateral (JBU) | 50 | 4.8 |
| Guided filter | 21 | 16.2 |
| Deep learning (EdgeDepthNet) | 153 | 2.3 |
| Hybrid (fusion + edge-aware refinement) | 383 | 20.7 |

Latency is wall-clock on the CPU above (no GPU); memory = max(tracemalloc peak, sampled RSS increase).

EdgeDepthNet: 723,882 parameters, best checkpoint at iteration 5000 (val mean RMSE 0.0427 m), CPU training time 51.6 min.

## Additional: cross-dataset generalisation (unseen sensors)

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

## Additional: ablation (test split)

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
