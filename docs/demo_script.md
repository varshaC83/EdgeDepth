# EdgeDepth — 3-minute demo script

Live on the public URL. Total ≈ 3:00. Keep the sidebar open on a laptop; on a phone use the ☰ menu.
Numbers on screen are computed live; the benchmark numbers come from `results/summary.json`.

| Time | Screen | What to say |
|---|---|---|
| 0:00–0:20 | **Home** | "Depth sensors give low-resolution, noisy depth with holes; the RGB camera beside them is sharp. Problem P4: upsample 2×, 4×, 8× while keeping object boundaries sharp and *without* copying RGB texture into the depth." |
| 0:20–0:35 | Home → workflow diagram | "Motivation: robotics, AR and 3-D capture need dense depth with correct edges — blurred edges create flying pixels in the 3-D reconstruction. Our approach has three layers: classical CV, a lightweight CNN, and a hybrid that fuses them per pixel." |
| 0:35–1:05 | **Depth Reconstruction** → sample `nyu_study_9_0003`, Low resolution, factor **8×**, noise 1, missing 5 %, edge dropout 0.5, method **Bicubic** → Run; then **Joint bilateral** → Run | "Classical side: hole detection and push-pull filling, noise estimation and denoising, Sobel/Canny and depth-discontinuity detection, then interpolation or RGB-guided filters. Bicubic blurs the boundary; joint bilateral sharpens it using colour similarity — this is an unseen test scene." |
| 1:05–1:35 | same page, method **Hybrid** → Run, open *Hybrid internals* | "EdgeDepthNet: 0.7 M parameters, separate RGB and depth encoders, a depth-controlled gate on RGB features against texture copying, and CSPN refinement; trained on CPU on scene-disjoint NYU scenes. The hybrid fuses four experts with a softmax whose weights depend on eight cues — here are the weight maps and cues. Finally an edge-aware back-projection pulls the result back to the actual measurements — the correction map shows it acts along surfaces and stops at depth edges." |
| 1:35–2:05 | **Method Comparison** → Run all methods; move the *Centre row* slider over an object edge | "Same input, same output size, one colour scale. The profile shows the depth step: bilinear and bicubic ramp across it, the hybrid follows the ground-truth step. Metrics table: RMSE, boundary RMSE, edge F1, latency." |
| 2:05–2:25 | **Evaluation Dashboard** → Benchmark tab, metric RMSE, then Boundary RMSE | "On the full unseen test split, across 2×, 4×, 8×, clean and noisy, the hybrid gives the lowest RMSE and boundary RMSE [read the two numbers from the chart]. The ablation shows each component's contribution." |
| 2:25–2:50 | **Failure Analysis** → *Texture copying* → Run; then *Incorrect boundaries* | "Honest limitations: on a flat plane with textured RGB the guided filter prints texture into the depth — our gate and texture cue suppress most of it. If RGB and depth are misaligned by 8 px, every RGB-guided method puts edges in the wrong place, including ours." |
| 2:50–3:00 | Depth Reconstruction → *Upload my own* (judge's input) | "Final outcome: a deployed, CPU-only system that improves accuracy and boundary preservation over interpolation, handles noise, holes and sparse input, and runs on any uploaded RGB-D pair — happy to try yours." |

## Handling a judge's unseen input
* RGB + full-resolution depth → choose *High-res ground-truth depth → simulate* → metrics appear.
* RGB + real low-res depth → choose *Real low-resolution / sparse depth* → factor is detected, metrics are
  labelled unavailable, no-reference statistics are shown.
* Depth units: 1000 for millimetre PNGs (NYU, Redwood, Azure Kinect), 5000 for TUM; 8-bit PNGs are treated as
  relative depth 0–10 m; `.npy` must already be in metres.

## Backup plan
If the hosted URL is cold, open it 2 minutes before the slot (first load ≈ 1–2 min). Local fallback:
`docker run -p 7860:7860 edgedepth` or `streamlit run app.py`.
