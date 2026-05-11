from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from arxiv_agent.cache_store import load_topic_data
from arxiv_agent.topic_embedding import cluster_embeddings, cosine_similarity


@dataclass
class ManifoldDataset:

    embeddings: torch.Tensor
    times: torch.Tensor
    cluster_ids: torch.Tensor
    edge_pairs: torch.Tensor
    impact_weights: torch.Tensor
    n_clusters: int
    embed_dim: int
    arxiv_ids: List[str]
    cluster_centroids: torch.Tensor
    id_to_idx: Dict[str, int] = field(default_factory=dict)

    cluster_labels: List[str] = field(default_factory=list)

    @property
    def n_papers(self) -> int:
        return self.embeddings.shape[0]

    @property
    def n_edges(self) -> int:
        return self.edge_pairs.shape[0]


def _compute_impact_scores(
    papers: list,
    edge_pairs: List[Tuple[int, int]],
    impact_path: Optional[Path] = None,
) -> List[float]:
    N = len(papers)

    raw = json.loads(impact_path.read_text(encoding="utf-8"))

    impact_by_id: Dict[str, Dict] = {}
    for entry in raw:
        arxiv_id = entry.get("arxiv_id")
        if arxiv_id:
            impact_by_id[arxiv_id] = entry

    stars_log: List[float] = []
    alt_log: List[float] = []
    citation_log: List[float] = []
    influential_citation_log: List[float] = []
    for p in papers:
        info = impact_by_id.get(getattr(p, "arxiv_id", ""), {})
        stars = float(info.get("github_stars", 0.0) or 0.0)
        alt = float(info.get("altmetric_score", 0.0) or 0.0)
        citation_count = float(info.get("citation_count", 0.0) or 0.0)
        influential_citation_count = float(info.get("influential_citation_count", 0.0) or 0.0)

        stars_log.append(math.log(1.0 + stars))
        alt_log.append(math.log(1.0 + alt) / math.log(4))
        citation_log.append(math.log(1.0 + citation_count) / math.log(8))
        influential_citation_log.append(math.log(1.0 + 10 * influential_citation_count) / math.log(8))

    alpha_stars = 1
    alpha_alt = 1
    alpha_citation = 1
    alpha_influential_citation = 1
    external_impact = [
        alpha_stars * s + alpha_alt * a + alpha_citation * c + alpha_influential_citation * ic
        for s, a, c, ic in zip(stars_log, alt_log, citation_log, influential_citation_log)
    ]

    scores = external_impact
    print(f"90th percentile: {np.percentile(scores, 90)}")
    print(f"50th percentile: {np.percentile(scores, 50)}")
    print(f"10th percentile: {np.percentile(scores, 10)}")

    return scores


def load_manifold_dataset(
    data_dir: str | Path,
    requested_k: Optional[int] = None,
    kmeans_seed: int = 42,
    device: str = "cpu",
) -> ManifoldDataset:
    data_dir = Path(data_dir)
    cache = load_topic_data(data_dir)
    if cache is None:
        raise FileNotFoundError(
            f"Could not load topic data from {data_dir!r}. "
            "Ensure papers.json and embeddings.npz are present."
        )

    papers = cache.papers
    raw_embeddings: List[List[float]] = cache.embeddings  # list[list[float]]

    assignments, centroids, topic_clusters = cluster_embeddings(
        papers=papers,
        embeddings=raw_embeddings,
        requested_k=requested_k,
        seed=kmeans_seed,
    )
    n_clusters = len(centroids)

    id_to_idx: Dict[str, int] = {p.arxiv_id: i for i, p in enumerate(papers)}

    timestamps = [p.published.timestamp() for p in papers]
    t_min = min(timestamps)
    t_max = max(timestamps)
    t_range = (t_max - t_min) if (t_max - t_min) > 0 else 1.0
    times_norm = [(t - t_min) / t_range for t in timestamps]
    times_norm = [t ** 2 for t in times_norm]

    edge_list: List[Tuple[int, int]] = []
    for edge in cache.inspiration_edges:
        if not edge.is_inspired:
            continue
        src_idx = id_to_idx.get(edge.source_arxiv_id)
        tgt_idx = id_to_idx.get(edge.target_arxiv_id)
        if src_idx is None or tgt_idx is None:
            continue
        edge_list.append((tgt_idx, src_idx))

    impact_path = data_dir / "impact.json"
    weights = _compute_impact_scores(papers, edge_list, impact_path)

    emb_tensor = torch.tensor(raw_embeddings, dtype=torch.float32, device=device)
    times_tensor = torch.tensor(times_norm, dtype=torch.float32, device=device)
    cluster_tensor = torch.tensor(assignments, dtype=torch.long, device=device)
    centroid_tensor = torch.tensor(centroids, dtype=torch.float32, device=device)
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

    if edge_list:
        edge_tensor = torch.tensor(edge_list, dtype=torch.long, device=device)
    else:
        edge_tensor = torch.zeros((0, 2), dtype=torch.long, device=device)

    return ManifoldDataset(
        embeddings=emb_tensor,
        times=times_tensor,
        cluster_ids=cluster_tensor,
        edge_pairs=edge_tensor,
        impact_weights=weight_tensor,
        n_clusters=n_clusters,
        embed_dim=emb_tensor.shape[1],
        arxiv_ids=[p.arxiv_id for p in papers],
        cluster_centroids=centroid_tensor,
        id_to_idx=id_to_idx,
        cluster_labels=[tc.label for tc in topic_clusters],
    )


def build_manifold_dataset_from_papers(
    data_dir: str | Path,
    papers: list,
    embeddings: List[List[float]],
    inspiration_edges: list,
    requested_k: Optional[int] = None,
    kmeans_seed: int = 42,
    device: str = "cpu",
) -> Tuple[ManifoldDataset, float, float, List[List[float]]]:
    from datetime import timezone

    def _ts(p):
        pt = getattr(p, "published", None)
        if pt is None:
            return 0.0
        if pt.tzinfo is None:
            pt = pt.replace(tzinfo=timezone.utc)
        return pt.timestamp()

    assignments, centroids, topic_clusters = cluster_embeddings(
        papers=papers,
        embeddings=embeddings,
        requested_k=requested_k,
        seed=kmeans_seed,
    )
    n_clusters = len(centroids)

    id_to_idx: Dict[str, int] = {p.arxiv_id: i for i, p in enumerate(papers)}

    timestamps = [_ts(p) for p in papers]
    t_min = min(timestamps)
    t_max = max(timestamps)
    t_range = (t_max - t_min) if (t_max - t_min) > 0 else 1.0
    times_norm = [((t - t_min) / t_range) ** 2 for t in timestamps]

    edge_list: List[Tuple[int, int]] = []
    for edge in inspiration_edges:
        if not edge.is_inspired:
            continue
        src_idx = id_to_idx.get(edge.source_arxiv_id)
        tgt_idx = id_to_idx.get(edge.target_arxiv_id)
        if src_idx is None or tgt_idx is None:
            continue
        edge_list.append((tgt_idx, src_idx))

    impact_path = data_dir / "impact.json"
    weights = _compute_impact_scores(papers, edge_list, impact_path=impact_path)

    emb_tensor = torch.tensor(embeddings, dtype=torch.float32, device=device)
    times_tensor = torch.tensor(times_norm, dtype=torch.float32, device=device)
    cluster_tensor = torch.tensor(assignments, dtype=torch.long, device=device)
    centroid_tensor = torch.tensor(centroids, dtype=torch.float32, device=device)
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

    if edge_list:
        edge_tensor = torch.tensor(edge_list, dtype=torch.long, device=device)
    else:
        edge_tensor = torch.zeros((0, 2), dtype=torch.long, device=device)

    dataset = ManifoldDataset(
        embeddings=emb_tensor,
        times=times_tensor,
        cluster_ids=cluster_tensor,
        edge_pairs=edge_tensor,
        impact_weights=weight_tensor,
        n_clusters=n_clusters,
        embed_dim=emb_tensor.shape[1],
        arxiv_ids=[p.arxiv_id for p in papers],
        cluster_centroids=centroid_tensor,
        id_to_idx=id_to_idx,
        cluster_labels=[tc.label for tc in topic_clusters],
    )
    return dataset, t_min, t_max, centroids
