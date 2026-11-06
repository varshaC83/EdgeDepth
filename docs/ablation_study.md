# EdgeDepth — Ablation study

## Experimental setup

* **Data**: the 24 unseen NYU Depth V2 test frames (scenes `basement_3`, `study_9`), 320×240.
* **Inputs**: for every frame, factors 2×, 4×, 8× × conditions {clean, noisy + missing (Kinect noise ×1,
  5 % random dropout, 50 % dropout on depth edges)} = 6 observations per frame, generated with fixed seeds
  (`20000 + frame index`). Every variant receives exactly the same observation objects.
* **Variants** (`edgedepth/pipeline.py::ABLATION_FUNCS`, script `scripts/ablation.py`):

| ID | Variant | What it isolates |
|---|---|---|
| A1 | Interpolation only (bicubic on the hole-filled LR grid) | baseline |
| A2 | Interpolation + guided filtering | value of classical RGB-guided refinement |
| A3 | Deep learning only — EdgeDepthNet backbone output *D_coarse* (before CSPN) | value of the learned RGB-depth fusion |
| A4a | Deep learning + learned edge-aware refinement (CSPN) — the `deep_learning` method | value of learned propagation |
| A4b | A4a + classical edge-aware back-projection | value of classical edge-aware refinement on the DL output |
| A4c | A4a + classical weighted-median boundary snapping | value of boundary snapping on the DL output |
| A5 | **Complete hybrid** (4 experts, cue-driven fusion, edge-aware back-projection) | full system |
| X1 | Hybrid without the back-projection (fusion only) | contribution of the data-consistency step |
| X2 | Hybrid with constant learned weights (intercepts only, no cues) + back-projection | contribution of *adaptivity* |
| X3 | Plain average of the 4 experts, no refinement | "just averaging" baseline the brief warns against |
| X4 | Hybrid without the network (classical fallback: 3 experts + fusion + back-projection) | contribution of the deep expert |
| X5 | Full hybrid + weighted-median snapping | the optional UI setting |

* **Metrics**: as in the main evaluation (RMSE, MAE, PSNR, SSIM, boundary RMSE, edge F1, edge sharpness,
  latency). All values are means over 24 frames × 6 observations = 144 cases.
* A3 and A4a come from the *same* trained network (deep supervision on both outputs), so the comparison
  isolates the CSPN stage rather than two different trainings.

## Results (mean over all 144 test cases)

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

## Quantitative observations

*Numbers from `results/ablation.json` (mean of 144 cases).*

* **A1 → A2 (guided filtering on interpolation): worse RMSE, better edges.** RMSE 0.0917 → 0.0986 m, edge F1
  0.635 → 0.736. The guided filter transfers RGB edges — including texture edges — so boundary *detection* improves
  while accuracy drops. Classical RGB guidance alone is not enough on NYU.
* **A1 → A3 (learned RGB–depth fusion): the largest gain.** RMSE −22 % (0.0712 m), boundary RMSE 0.2136 → 0.1693 m,
  edge F1 0.846. The depth-gated network uses RGB where it is consistent with depth.
* **A3 → A4a (learned CSPN refinement): −6 % RMSE** (0.0669 m), SSIM 0.9418 → 0.9532, edge F1 0.870. Learned
  propagation removes residual noise along surfaces without crossing edges.
* **A4a → A4b (classical edge-aware back-projection): −5 % RMSE** (0.0634 m), boundary RMSE 0.1601 → 0.1518 m.
  A purely classical step corrects the network's systematic disagreement with the measurements.
* **A4a → A4c / A5 → X5 (weighted-median snapping): edges up, accuracy down.** Edge F1 rises to 0.880–0.887 but RMSE
  rises to 0.0720–0.0758 m: when the median picks the wrong side of an edge the error is the full depth step.
* **A5 (full hybrid) = A4b = X2 to four decimals.** With the trained network present, the fitted gate assigns ≈ 99 %
  weight to it everywhere, so cue-driven weights (A5) and constant weights (X2) coincide. The value of the hybrid over
  the network alone comes from the back-projection (X1 → A5: 0.0668 → 0.0634 m).
* **X3 (uniform average): 0.0889 m.** Averaging four experts is 40 % worse than the fitted fusion — averaging without
  justification is clearly harmful because three of the four experts are much less accurate.
* **X4 (classical fallback): 0.0885 m vs 0.0917 m for the best single classical method** and the best edge F1 of all
  classical variants (0.799). Without the network the cue-driven fusion is genuinely adaptive (see the coefficients in
  the app): it drops the guided filter, distrusts JBU where it disagrees with the measurements and trusts it more in
  holes.

**Contribution summary (RMSE, mean over 2×/4×/8× × clean/noisy):** learned fusion network −22 %, CSPN −6 %,
edge-aware back-projection −5 %; total −31 % vs bicubic. Adaptive gating contributes only in the classical fallback
(−3.5 % vs bicubic there).
