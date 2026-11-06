"""Fit the hybrid fusion coefficients (beta) on the VALIDATION split.

For every validation frame and a set of degradations (2x/4x/8x low-res,
clean / noisy+missing / large holes, and 5% sparse) we compute all experts
and cues, sample pixels (boundary pixels over-sampled 3x), and minimise

    mean_p | sum_k softmax_k(z(p)) D_k(p) - D_gt(p) |  +  1e-4 ||beta_cues||^2

with Adam. Four parameter sets are written to checkpoints/fusion_params.json:
    with_dl          : 4 experts (bicubic, guided filter, JBU, EdgeDepthNet), all cues
    classical        : 3 classical experts (fallback when no checkpoint), all cues
    with_dl_nocues   : 4 experts, intercepts only (constant learned weights)  [ablation]
    with_dl_uniform  : plain average                                          [ablation]
The test split is never used.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edgedepth.degradation import DegradationConfig, degrade  # noqa: E402
from edgedepth.hybrid.fusion import CLASSICAL, FEATURES, WITH_DL, FusionParams, compute_candidates, compute_cues  # noqa: E402
from edgedepth.metrics import gt_edge_band  # noqa: E402
from edgedepth.model.dataset import load_split  # noqa: E402

CONDS = [
    dict(), dict(noise_level=1.0, missing_rate=0.05, edge_dropout=0.5), dict(noise_level=2.0, hole_blobs=3, missing_rate=0.1),
]


def collect(frames, names, use_dl, rng, per_img=6000):
    X, Y, D = [], [], []
    for i, (_, rgb, gt) in enumerate(frames):
        cfgs = [DegradationConfig(scale=s, seed=5000 + i * 10 + j, **c) for s in (2, 4, 8) for j, c in enumerate(CONDS)]
        cfgs.append(DegradationConfig(mode="sparse", sparse_density=0.05, seed=7000 + i))
        for cfg in cfgs:
            obs = degrade(gt, cfg)
            pc = compute_candidates(rgb, obs, use_dl=use_dl)
            cues = compute_cues(rgb, obs, pc["cands"], names, pc["dl"])
            band = gt_edge_band(gt) & (gt > 0)
            p = (gt > 0).astype(np.float64) * (1 + 2 * band)
            p /= p.sum()
            idx = rng.choice(gt.size, size=per_img, replace=False, p=p.ravel())
            H, W = gt.shape
            yy, xx = np.unravel_index(idx, (H, W))
            X.append(cues["phi"][:, :, yy, xx].transpose(2, 0, 1))  # (N, K, F)
            D.append(cues["stack"][:, yy, xx].T)                     # (N, K)
            Y.append(gt[yy, xx])
        print(f"  frame {i + 1}/{len(frames)}", flush=True)
    return np.concatenate(X), np.concatenate(D), np.concatenate(Y)


def fit(X, D, Y, use_cues=True, iters=2500, batch=200_000, seed=0):
    """Minimise L1 of the softmax mixture with Adam (mini-batches of pixels).

    Intercepts are initialised from the experts' mean errors (log 1/MAE), so the
    optimisation starts from a sensible constant-weight mixture.
    """
    g = torch.Generator().manual_seed(seed)
    X, D, Y = (torch.from_numpy(a).float() for a in (X, D, Y))
    N, K, F = X.shape
    mae_k = (D - Y[:, None]).abs().mean(0)
    beta = torch.zeros(K, 1 + F)
    beta[:, 0] = torch.log(1.0 / mae_k) * 2
    beta[:, 0] -= beta[:, 0].mean()
    beta.requires_grad_(True)
    opt = torch.optim.Adam([beta], lr=0.03)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)

    def mix(xb, db):
        z = beta[:, 0].expand(xb.shape[0], K) + ((xb * beta[:, 1:]).sum(-1) if use_cues else 0)
        return (torch.softmax(z, 1) * db).sum(1)

    for it in range(iters):
        idx = torch.randint(0, N, (min(batch, N),), generator=g)
        loss = (mix(X[idx], D[idx]) - Y[idx]).abs().mean() + 1e-4 * (beta[:, 1:] ** 2).sum()
        opt.zero_grad()
        loss.backward()
        if not use_cues:
            beta.grad[:, 1:] = 0
        opt.step()
        sched.step()
    with torch.no_grad():
        mae = float((mix(X, D) - Y).abs().mean())
    return beta.detach().numpy(), mae


def main():
    rng = np.random.default_rng(0)
    val = load_split("data/nyu", "val")
    out = {}
    path = "checkpoints/fusion_params.json"
    if os.path.exists(path):
        out = json.load(open(path))
    report = {}
    for key, names, use_dl in (("with_dl", WITH_DL, True), ("classical", CLASSICAL, False)):
        print(f"collecting {key} ...", flush=True)
        X, D, Y = collect(val, names, use_dl, rng)
        mae_each = {n: float(np.abs(D[:, k] - Y).mean()) for k, n in enumerate(names)}
        beta, mae = fit(X, D, Y, True)
        out[key] = FusionParams(names, beta, FEATURES, "NYU val split (scene-disjoint), sampled pixels").to_dict()
        report[key] = {"pixels": int(len(Y)), "mae_each_expert_on_samples": mae_each, "mae_fused_on_samples": mae}
        if use_dl:
            beta0, mae0 = fit(X, D, Y, False)
            out["with_dl_nocues"] = FusionParams(names, beta0, FEATURES, "intercepts only").to_dict()
            out["with_dl_uniform"] = FusionParams.uniform(names).to_dict()
            report["with_dl_nocues"] = {"mae_fused_on_samples": mae0}
            report["with_dl_uniform"] = {"mae_fused_on_samples": float(np.abs(D.mean(1) - Y).mean())}
        print(json.dumps(report, indent=1), flush=True)
    os.makedirs("checkpoints", exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    with open("results/fusion_fit_report.json", "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
