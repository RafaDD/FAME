from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from .model import ManifoldSpine


def frontier_scores(
    z: torch.Tensor,
    spine: ManifoldSpine,
    times: torch.Tensor,
    cluster_ids: torch.Tensor,
    arxiv_ids: List[str],
    t_now: float = 1.0,
    recent_fraction: float = 0.15,
) -> Tuple[Dict[str, float], List[Tuple[str, float]]]:
    device = z.device
    spine = spine.to(device)
    spine.eval()

    K = spine.n_clusters
    N = z.shape[0]
    t_threshold = t_now - recent_fraction

    tangents: List[torch.Tensor] = []
    for k in range(K):
        tv = spine.tangent(k=k, t=t_now, device=device)
        norm = tv.norm().clamp(min=1e-12)
        tangents.append(tv / norm)
    tangent_stack = torch.stack(tangents, dim=0)

    k_idx = torch.arange(K, device=device)
    t_now_vec = torch.full((K,), t_now, dtype=torch.float32, device=device)
    with torch.no_grad():
        mu_now = spine(k_idx, t_now_vec)

    scores_list: List[float] = []
    for i in range(N):
        if times[i].item() < t_threshold:
            scores_list.append(-math.inf)
            continue
        k = cluster_ids[i].item()
        delta_z = z[i] - mu_now[k]
        score = F.cosine_similarity(delta_z, tangent_stack[k], dim=-1, eps=1e-8).item()
        scores_list.append(float(score))

    scores_dict = {aid: s for aid, s in zip(arxiv_ids, scores_list)}
    ranked = sorted(
        [(aid, s) for aid, s in scores_dict.items() if not math.isinf(s)],
        key=lambda x: x[1],
        reverse=True,
    )
    return scores_dict, ranked


def frontier_scores_at_publication_time(
    z: torch.Tensor,
    spine: ManifoldSpine,
    times: torch.Tensor,
    cluster_ids: torch.Tensor,
    arxiv_ids: List[str],
    t_now: float = 1.0,
    recent_fraction: float = 0.15,
) -> Tuple[Dict[str, float], List[Tuple[str, float]]]:
    device = z.device
    spine = spine.to(device)
    spine.eval()

    N = z.shape[0]
    t_threshold = t_now - recent_fraction

    with torch.no_grad():
        mu_at_paper_time = spine(cluster_ids, times)

    scores_list: List[float] = []
    for i in range(N):
        if times[i].item() < t_threshold:
            scores_list.append(-math.inf)
            continue
        k = cluster_ids[i].item()
        t_i = times[i].item()
        tv = spine.tangent(k=k, t=t_i, device=device)
        norm = tv.norm().clamp(min=1e-12)
        tangent_unit = tv / norm
        delta_z = z[i] - mu_at_paper_time[i]
        score = F.cosine_similarity(
            delta_z.unsqueeze(0), tangent_unit.unsqueeze(0), dim=-1, eps=1e-8
        ).item()
        scores_list.append(float(score))

    scores_dict = {aid: s for aid, s in zip(arxiv_ids, scores_list)}
    ranked = sorted(
        [(aid, s) for aid, s in scores_dict.items() if not math.isinf(s)],
        key=lambda x: x[1],
        reverse=True,
    )
    return scores_dict, ranked
