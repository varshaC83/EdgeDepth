# EdgeDepth — Technical viva preparation

Answers refer to the code in this repository. Numbers quoted come from
`results/summary.json`, `results/ablation.json` and `results/failure_cases.json`;
re-check them after any re-run.

## A. Problem and data

**Q1. What exactly is the input and output?**
An RGB image (H×W) and a depth observation that is either an (H/s)×(W/s) grid for s ∈ {2,4,8} or
sparse H×W samples, with 0 marking missing values. Output: a dense H×W depth map in metres. Every
method outputs exactly H×W (asserted in `pipeline.run_method`).

**Q2. Which dataset, and how did you avoid leakage?**
NYU Depth V2: 179 frames from 10 scenes (public GitHub mirror, the official host was unreachable from our
build machine). Splits are *scene-disjoint*: train 6 scenes / 126 frames, val 2 scenes / 29, test 2 scenes / 24.
Classical hyper-parameters, the model checkpoint and the fusion coefficients were selected on val; test was
touched only by `evaluate.py`, `ablation.py`, `failure_cases.py`.

**Q3. How do you generate low-resolution inputs?**
Masked box-averaging (`masked_area_downsample`): each LR pixel is the mean of the valid HR pixels in its s×s
block, which models the integration of a real low-res sensor. Then optional Kinect noise
σ(z)=k(1.2 mm + 1.9 mm·(z−0.4)²), random dropout, elliptical blob holes and dropout on depth edges (sensor shadows).

**Q4. Why is the evaluation resolution 320×240?**
Divisible by 8 so 2×/4×/8× share an exact grid; keeps CPU latency within the 10 s judging budget. Uploaded
images up to 640 px on the long side are processed at native resolution.

## B. Classical methods

**Q5. Difference between nearest, bilinear and bicubic?**
Kernel support and smoothness: nearest copies the closest sample (blocky, no new values); bilinear
interpolates linearly from 2×2 samples (C⁰, blurs edges); bicubic uses a 4×4 cubic kernel (C¹, sharper but
overshoots at steps — we clamp to the minimum valid depth). For sparse input we use the scattered-data
equivalents: nearest sample, Delaunay-linear, Clough-Tocher cubic.

**Q6. Explain joint bilateral upsampling.**
Each HR pixel is a normalised weighted mean of nearby LR measurements; weight = spatial Gaussian (distance in
LR pixels) × range Gaussian on the *RGB* difference between the HR pixel and the LR sample's colour × the
measurement mask. Colour edges therefore stop depth from bleeding across object boundaries. σ_s = 0.5 LR px,
σ_r = 0.2 were chosen on val (`results/classical_tuning.json`).

**Q7. Explain the guided filter. Why can it copy texture?**
It assumes the output is locally a linear function of the guide: q = aᵀI + b in every window, with
a = (Σ + εU)⁻¹cov(I,p). Where the guide varies, a ≠ 0 and the guide's variation — including pure texture —
appears in q. ε controls the threshold; the val-optimal ε = 1e-5 transfers edges strongly, which is exactly
why it copies texture in failure case F4.

**Q8. How do you detect depth discontinuities?**
Sobel on log-depth gives |∇log D| ≈ relative depth change per pixel, which is scale invariant (a 5 cm step
at 1 m and 50 cm at 10 m look alike). Threshold 0.03 + non-maximum thinning → binary map; 1−exp(−(g/0.03)²)
gives a soft probability used as a fusion cue.

**Q9. How does push-pull hole filling work?**
Push: build a pyramid where each coarse pixel is the mask-weighted mean of its 2×2 children, until every pixel
is valid. Pull: upsample coarse estimates and use them only where the finer level is missing. It is O(N),
parameter-free and fills holes of any size smoothly while keeping observed values unchanged.

**Q10. How is noise removed without destroying edges?**
Noise σ is estimated with Immerkær's method (a 3×3 kernel that cancels planes; mean absolute response on the
smoothest pixels). Then a 3×3 median (removes outliers, preserves steps) and a bilateral filter with range
σ_r = 3σ̂ (steps much larger than the noise are kept). In the hybrid this is applied to the classical experts only
if σ̂/median depth exceeds a scale-dependent threshold (0.22 % / 0.35 % / 0.85 % at 2× / 4× / 8×). At 8× the
estimator cannot fully separate scene structure from noise — a measured limitation.

## C. Deep learning

**Q11. Describe EdgeDepthNet.**
Two encoders (RGB; depth = D0/μ, mask, scale code) over 1, 1/2, 1/4, a 1/8 bottleneck with two residual
blocks, depth-gated fusion at each level, a U-Net decoder, and three heads: residual on D0, boundary logit,
8 affinities. D_coarse = D0 + residual is refined by six CSPN propagation steps. 723,882 parameters,
~0.1 s per 320×240 frame on CPU.

**Q12. Why residual learning on an interpolated D0?**
The network only has to learn the high-frequency correction (edges, holes, noise), which is far easier with
126 training frames than predicting absolute depth; with a zero-initialised residual head training starts
exactly at the bicubic baseline.

**Q13. What is the depth-gated fusion and why?**
RGB features are multiplied by a sigmoid gate computed from depth+RGB features before being mixed in. The
network can learn to close the gate on RGB edges that have no depth support (texture) and open it at real
boundaries — the learned counterpart of the texture-copy cue.

**Q14. What is CSPN?**
Convolutional Spatial Propagation (Cheng et al., ECCV 2018): repeated D ← κ₀D + Σᵢκᵢ·shiftᵢ(D) with
predicted affinities κᵢ normalised so Σ|κᵢ| ≤ 1 (stable). It diffuses depth along surfaces but not across
predicted edges — a learned anisotropic diffusion. We zero-initialise it so it starts as identity.

**Q15. Loss function?**
Edge-weighted L1 + edge-weighted RMSE on the refined depth, 0.5 × L1 on the coarse depth (deep supervision),
gradient L1 (penalises blurred / shifted edges) and 0.1 × BCE for boundaries. Pixels within 2 px of a GT
discontinuity get weight 3. L1 alone tends to sharp but sometimes misplaced edges, which RMSE punishes;
the RMSE term balances that.

**Q16. How was it trained without a GPU?**
On 2 CPU cores: random 128×160 crops, batch 8, Adam 1e-3 with cosine decay, ~0.55 s/iteration; validation on
full val frames every 250 iterations, best checkpoint kept (`results/training_log.json`). The same script runs
on Colab/Kaggle with a GPU.

**Q17. One model for 2×, 4×, 8× and sparse — how?**
Training samples a random factor or density per crop; the scale code channel log₂(s)/3 (s = 1/√density for
sparse) tells the network the expected measurement spacing.

## D. Hybrid fusion

**Q18. Write down the fusion rule.**
z_k(p) = β_k0 + Σ_f β_kf φ_kf(p); w_k = softmax_k(z_k); D(p) = Σ_k w_k(p) D_k(p). Experts: bicubic, guided
filter, JBU, EdgeDepthNet. Followed by 3 iterations of edge-aware back-projection:
D ← D + JTU(M ⊙ (D_lr − ↓D)), where JTU spreads LR residuals with spatial × RGB × log-depth weights.

**Q19. What are the cues and why each one?**
B_dl network boundary probability (where are edges?), B_cl classical discontinuity probability of the
consensus, V local depth variation (planar vs complex), E RGB edge strength, H missing-measurement indicator
(no data nearby → trust the learned prior), T_k texture-copy score (RGB edge without depth edge where the
expert departs from bicubic), R_k LR data residual (does the expert agree with the measurements?), A_k
deviation from the median consensus (outlier expert).

**Q20. Why a softmax mixture and not an average or a hard switch?**
A plain average inherits bicubic blur at edges and RGB texture in flat areas (variant X3 in the ablation). A
hard switch creates seams. The softmax is convex (never leaves the experts' range), smooth, differentiable so β
can be fitted by gradient descent, and each β is interpretable.

**Q21. How were β fitted and could that overfit?**
36 parameters (4 experts × (1 bias + 8 cues)), fitted on 1.74 M sampled validation pixels (29 frames × 10
degradations × 6000 px) by minimising L1 with a small L2 penalty. With so few parameters and that many samples
overfitting is negligible, and the test split was never used.

**Q21b. Your ablation shows constant weights give the same result as the cue-driven fusion. So is the fusion useless?**
With the trained network present, yes for accuracy: the fitted gate gives the network ≈ 99 % weight everywhere
because it beats every classical expert in every cue configuration we observed (validation MAE 0.030 m vs ≥ 0.049 m).
We report that rather than hide it. The hybrid's gain over the network comes from the classical edge-aware
back-projection (X1 → A5: RMSE 0.0668 → 0.0634 m). The fusion matters when the network is unavailable or weak: in
the classical fallback it drops the guided filter, distrusts JBU where it disagrees with the measurements and beats the
best classical method (0.0885 vs 0.0917 m). A non-linear gate or more complementary experts would be the next step.

**Q22. What does the back-projection do and why is it edge-aware?**
It measures how far the current estimate, downsampled the same way as the sensor, is from the actual LR
measurements, and spreads that residual back to HR. A plain bilinear spread would smear a background correction
onto the foreground; the joint trilateral kernel only spreads it to pixels of similar colour *and* similar current
depth, and never from missing measurements. It removes a systematic bias the network leaves. Its gain shrinks with
the estimated noise (α = clip(1 − (q − 1)/2.5, 0, 1)) because at high noise the measurements are worse than the
network's estimate — we found this on validation after the failure study showed the un-gated version hurt at ×4/×8 noise.

**Q22b. You also implemented a weighted-median snapping — why is it off by default?**
A median returns a value from one side of an edge, so it removes flying pixels and raises edge F1; but when it
picks the wrong side the error is a full depth step, which RMSE punishes. On validation it increased RMSE, so it
is an optional UI toggle and ablation X5 reports its effect on test.

**Q23. What happens when the checkpoint is missing?**
`load_model` returns None, the UI shows a warning, deep learning is hidden, and `hybrid_reconstruct` switches to
the classical β set fitted for the 3 classical experts (variant X4 in the ablation).

## E. Metrics

**Q24. Define RMSE, MAE and PSNR as you compute them.**
Over valid GT pixels: RMSE = √mean((D−G)²), MAE = mean|D−G| (metres). PSNR = 10 log₁₀(10²/MSE) with a fixed
10 m peak (NYU sensor range), so values are comparable across frames.

**Q25. SSIM on depth?**
scikit-image SSIM with data_range = GT max − GT min per frame; measures local structure (means, variances,
covariance) rather than absolute error.

**Q26. How do you measure edge preservation and boundary accuracy?**
(1) Boundary RMSE: RMSE in a 2-px band around GT discontinuities. (2) Edge F1: discontinuity maps of prediction
and GT (same detector), precision/recall with 2-px tolerance. (3) Edge sharpness: mean |∇log D| of the
prediction over GT edge pixels divided by that of GT (1 = as sharp as GT, < 1 blurred).

**Q27. Why can nearest-neighbour have a high edge F1 but a bad RMSE?**
Its staircase produces many strong discontinuities, some close enough to true edges to count as matches, but
they are in the wrong place and the values are blocky. That is why we report several complementary metrics.

**Q28. How do you measure latency and memory?**
Latency: wall-clock per call, measured without tracking overhead. Memory: max of tracemalloc peak (NumPy) and
peak RSS increase sampled every 2 ms (captures PyTorch), in a separate pass. Both are machine dependent.

## F. Design decisions and limitations

**Q29. Why not a large transformer depth model?**
Deployment target is a free CPU host with a 10 s response budget, and only 126 training frames were available;
a 0.7 M-parameter CNN trains on CPU in about an hour and runs in ~0.1 s.

**Q30. Biggest limitations?**
Small training set (10 NYU scenes) → limited diversity; reliance on RGB–depth alignment (F5); extremely sparse
input (< 1 %) and large holes have no evidence to recover (F1, F2); ground truth itself is NYU's filled depth,
whose inpainted regions are not real measurements.

**Q31. Ethics?**
Depth of indoor scenes can reveal room layouts; the app processes uploads in memory only and stores nothing.
Upsampled depth is an estimate — it must not be used for safety-critical decisions (e.g. robot collision
avoidance) without real sensor validation, since hallucinated surfaces in holes look plausible.
