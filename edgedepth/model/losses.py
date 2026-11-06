"""Training losses: edge-weighted L1, gradient (edge-aware) loss and boundary BCE."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def gradient_l1(pred, gt, valid):
    dx_p, dx_g = pred[..., :, 1:] - pred[..., :, :-1], gt[..., :, 1:] - gt[..., :, :-1]
    dy_p, dy_g = pred[..., 1:, :] - pred[..., :-1, :], gt[..., 1:, :] - gt[..., :-1, :]
    vx = valid[..., :, 1:] * valid[..., :, :-1]
    vy = valid[..., 1:, :] * valid[..., :-1, :]
    return ((dx_p - dx_g).abs() * vx).sum() / vx.sum().clamp(min=1) + ((dy_p - dy_g).abs() * vy).sum() / vy.sum().clamp(min=1)


def edgedepth_loss(out, gt, valid, edge_gt, band_gt, mu, w_coarse=0.5, w_grad=1.0, w_edge=0.1, w_rmse=1.0, band_weight=2.0):
    """Total loss (depth in metres).

    L = L1_w(D_ref) + RMSE_w(D_ref) + 0.5 L1_w(D_coarse) + L_grad(D_ref) + 0.1 BCE(edge)
    where the _w terms weight pixels within 2 px of a GT discontinuity by (1 + band_weight).
    The L1 term keeps edges sharp (median-like), the RMSE term penalises large
    boundary errors that dominate RMSE.
    """
    ref = out["refined"] * mu
    coarse = out["coarse"] * mu
    w = valid * (1 + band_weight * band_gt)
    l_ref = (w * (ref - gt).abs()).sum() / w.sum().clamp(min=1)
    l_rmse = torch.sqrt((w * (ref - gt) ** 2).sum() / w.sum().clamp(min=1) + 1e-8)
    l_coarse = (w * (coarse - gt).abs()).sum() / w.sum().clamp(min=1)
    l_grad = gradient_l1(ref, gt, valid)
    pos = edge_gt.mean().clamp(min=1e-3)
    l_edge = F.binary_cross_entropy_with_logits(out["edge_logit"], edge_gt, pos_weight=((1 - pos) / pos).clamp(max=30))
    total = l_ref + w_rmse * l_rmse + w_coarse * l_coarse + w_grad * l_grad + w_edge * l_edge
    return total, {k: float(v.detach()) for k, v in {"l1_ref": l_ref, "rmse_ref": l_rmse, "l1_coarse": l_coarse, "grad": l_grad, "edge_bce": l_edge}.items()}
