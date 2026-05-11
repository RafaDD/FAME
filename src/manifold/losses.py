from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .model import LinearProjection, ManifoldSpine


def spine_loss(z: torch.Tensor, mu_at_t: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(z, mu_at_t.detach() if not mu_at_t.requires_grad else mu_at_t)


def inspire_loss(
    z: torch.Tensor,
    spine: "ManifoldSpine",
    c: torch.Tensor,
    t: torch.Tensor,
    edge_pairs: torch.Tensor,
    eps: float = 1e-2,
    margin: float = 0.5,
    eta: float = 0.1,
) -> torch.Tensor:
    if edge_pairs.shape[0] == 0:
        return z.new_zeros(())

    src_idx = edge_pairs[:, 0]
    tgt_idx = edge_pairs[:, 1]

    z_src = z[src_idx]
    z_tgt = z[tgt_idx]

    vec_micro = z_tgt - z_src

    t = t[src_idx]
    c = c[src_idx]
    t_fwd = (t + eps).clamp(max=1.0)
    t_bwd = (t - eps).clamp(min=0.0)
    step = (t_fwd - t_bwd).unsqueeze(-1)

    mu_fwd = spine(c, t_fwd)
    mu_bwd = spine(c, t_bwd)
    tangent_vecs = (mu_fwd - mu_bwd) / step

    cos = F.cosine_similarity(vec_micro, tangent_vecs, dim=-1, eps=1e-8)  # (E,)

    margin_term = F.relu(margin - cos).mean()
    dist_term = eta * (vec_micro.pow(2).sum(dim=-1)).mean()

    return margin_term + dist_term


def vanguard_loss(
    z: torch.Tensor,
    mu_at_t: torch.Tensor,
    spine: "ManifoldSpine",
    c: torch.Tensor,
    t: torch.Tensor,
    weights: torch.Tensor,
    delta_vg: float = 0.5,
    eps: float = 1e-2,
) -> torch.Tensor:
    w_sum = weights.sum()
    if w_sum < 1e-8:
        return z.new_zeros(())

    t_fwd = (t + eps).clamp(max=1.0)
    t_bwd = (t - eps).clamp(min=0.0)
    step = (t_fwd - t_bwd).unsqueeze(-1)

    mu_fwd = spine(c, t_fwd)
    mu_bwd = spine(c, t_bwd)
    tangent_vecs = (mu_fwd - mu_bwd) / step

    residual = z - mu_at_t

    cos = F.cosine_similarity(residual, tangent_vecs, dim=-1, eps=1e-8)

    weights_norm = (weights - torch.median(weights)) / (weights.max() - torch.median(weights))
    delta_vg = delta_vg + (1 - delta_vg) * weights_norm
    hinge = F.relu(delta_vg - cos)

    negative_cos = F.relu(cos)

    weights_mask = (weights >= 2).float()

    return (weights_mask * hinge).mean() + ((1 - weights_mask) * negative_cos).mean()

@dataclass
class LossBreakdown:
    total: float
    spine: float
    inspire: float
    vanguard: float

    def __str__(self) -> str:
        return (
            f"total={self.total:.4f}  "
            f"spine={self.spine:.4f}  "
            f"inspire={self.inspire:.4f}  "
            f"vanguard={self.vanguard:.4f}"
        )


def total_loss(
    l_spine: torch.Tensor,
    l_inspire: torch.Tensor,
    l_vanguard: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.5,
    nu: float = 0.2,
) -> tuple[torch.Tensor, LossBreakdown]:
    loss = (
        alpha * l_spine
        + beta * l_inspire
        + nu * l_vanguard
    )

    breakdown = LossBreakdown(
        total=loss.item(),
        spine=l_spine.item(),
        inspire=l_inspire.item(),
        vanguard=l_vanguard.item(),
    )
    return loss, breakdown
