from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_mlp(
    in_dim: int,
    hidden_dims: List[int],
    out_dim: int,
    layer_norm: bool = True,
    activation: str = "relu",
) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        if activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif activation == "tanh":
            layers.append(nn.Tanh())
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class SpatiotemporalMapper(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Optional[List[int]] = None,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]
        # input = [x (d), t (1)]
        self.time_emb = nn.Linear(1, out_dim)
        self.net = _build_mlp(in_dim + out_dim, hidden_dims, out_dim, layer_norm=layer_norm)
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

        t_col = t.unsqueeze(-1)          # (N, 1)
        t_emb = self.time_emb(t_col)
        inp = torch.cat([x, t_emb], dim=-1)  # (N, d+d')
        return self.net(inp)


class ManifoldSpine(nn.Module):
    def __init__(
        self,
        n_clusters: int,
        out_dim: int,
        e_dim: int = 32,
        hidden_dims: Optional[List[int]] = None,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128]
        self.topic_emb = nn.Embedding(n_clusters, e_dim)
        self.time_emb = nn.Linear(1, out_dim)

        self.net = _build_mlp(e_dim + out_dim, hidden_dims, out_dim, layer_norm=layer_norm, activation='tanh')
        self.n_clusters = n_clusters
        self.out_dim = out_dim
        self.e_dim = e_dim

    def forward(
        self,
        k_indices: torch.Tensor,
        t_values: torch.Tensor,
    ) -> torch.Tensor:
        e = self.topic_emb(k_indices)
        t_col = t_values.unsqueeze(-1)
        t_emb = self.time_emb(t_col)
        inp = torch.cat([e, t_emb], dim=-1)
        return self.net(inp)

    def at_time(self, k: int, t: float, device: torch.device | str = "cpu") -> torch.Tensor:
        k_idx = torch.tensor([k], dtype=torch.long, device=device)
        t_val = torch.tensor([t], dtype=torch.float32, device=device)
        return self.forward(k_idx, t_val).squeeze(0)

    def tangent(self, k: int, t: float, device: torch.device | str = "cpu") -> torch.Tensor:
        t_val = torch.tensor([t], dtype=torch.float32, device=device, requires_grad=True)
        k_idx = torch.tensor([k], dtype=torch.long, device=device)
        mu = self.forward(k_idx, t_val).squeeze(0)

        t_prev = torch.tensor([t - 5e-2], dtype=torch.float32, device=device, requires_grad=True)
        mu_prev = self.forward(k_idx, t_prev).squeeze(0)
        delta_mu = mu - mu_prev
        delta_t = t_val - t_prev
        tangent = delta_mu / delta_t
        return tangent


class LinearProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, d) → (N, d')"""
        return self.proj(x)


class NaiveModel(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Optional[List[int]] = None,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]
        self.net = _build_mlp(in_dim, hidden_dims, 1, layer_norm=layer_norm)
        self.in_dim = in_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out.squeeze(-1)
