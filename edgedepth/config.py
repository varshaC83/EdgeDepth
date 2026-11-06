"""Default hyper-parameters.

Classical filter parameters were selected on the VALIDATION split by
``scripts/tune_classical.py`` (see results/classical_tuning.json); the test
split was never used for tuning.
"""

# Joint bilateral upsampling (sigma_s in LR pixels, sigma_r on RGB in [0,1])
JBU_SIGMA_S = 0.5
JBU_SIGMA_R = 0.2

# Guided filter: radius = max(1, round(scale * GF_RADIUS_MULT)), regulariser eps
GF_RADIUS_MULT = 0.25
GF_EPS = 1e-5

# Relative noise level (sigma / median depth) above which the hybrid denoises the
# observation before the classical experts. Scale dependent because scene
# structure leaks into the noise estimate on coarse grids; set on VAL.
NOISE_REL_THRESH = {2: 0.0022, 4: 0.0035, 8: 0.0085}

# Edge-aware back-projection (data-consistency refinement of the fused depth), tuned on VAL
BP_ITERS = 3
BP_SIGMA_D = 0.05      # range sigma on log-depth (relative depth difference ~5%)
# Noise-adaptive gain of the back-projection correction (measurements are only
# trusted as much as their estimated noise allows), fitted on VAL:
#   q = (estimated relative noise) / NOISE_REL_THRESH[scale]
#   alpha = clip(1 - (q - BP_GAIN_Q0) / BP_GAIN_SPAN, 0, 1)
BP_GAIN_Q0 = 1.0
BP_GAIN_SPAN = 2.5

# Boundary-aware weighted-median snapping (optional; raises edge F1 but also RMSE on NYU val)
BREF_THRESH = 0.5
BREF_WIDTH = 1
BREF_RADIUS = 3

# Working resolution limit for uploaded images (keeps CPU latency < ~10 s)
MAX_SIDE = 640

CHECKPOINT = "checkpoints/edgedepthnet.pt"
FUSION_PARAMS = "checkpoints/fusion_params.json"
