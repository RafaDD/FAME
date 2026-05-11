from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple, Any

from config import cfg
from arxiv_agent.topic_embedding import assign_to_centroids
from manifold.dataset import build_manifold_dataset_from_papers
from manifold.trainer import ManifoldTrainer, NaiveTrainer, NaiveTrainingConfig, TrainingConfig
from manifold.analysis import frontier_scores_at_publication_time
from utils.evaluation import evaluate_ideas
from schemas import GeneratedIdea


def _ensure_tz(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def get_llm_eval_model_key() -> str:
    if not os.getenv("OPENAI_API_KEY"):
        return "fallback"
    return cfg.llm.eval_model or os.getenv("OPENAI_EVAL_MODEL", "gpt-4.1")


def run_manifold_pred(
    args: argparse.Namespace,
    data_dir: Path,
    train_papers: list,
    train_embeddings: list,
    eval_papers: list,
    eval_embeddings: list,
    all_edges: list,
) -> Dict[str, Any]:
    import torch

    dataset, t_min, t_max, centroids = build_manifold_dataset_from_papers(
        data_dir=data_dir,
        papers=train_papers,
        embeddings=train_embeddings,
        inspiration_edges=all_edges,
        requested_k=args.cluster_count,
        kmeans_seed=args.seed,
        device=args.device,
    )
    print(
        f"[predict_frontier] dataset: {dataset.n_papers} papers, "
        f"{dataset.n_clusters} clusters, {dataset.n_edges} inspiration edges",
        flush=True,
    )

    mc = cfg.manifold
    config = TrainingConfig(
        out_dim=args.out_dim,
        spine_e_dim=args.spine_e_dim,
        mapper_hidden=mc.mapper_hidden,
        spine_hidden=mc.spine_hidden,
        layer_norm=mc.layer_norm,
        lr=args.lr,
        weight_decay=args.weight_decay,
        n_epochs=args.epochs,
        log_every=args.log_every,
        alpha=args.alpha,
        beta=args.beta,
        nu=args.nu,
        inspire_margin=mc.inspire_margin,
        inspire_eta=mc.inspire_eta,
        delta_vg=mc.delta_vg,
        vanguard_eps=mc.vanguard_eps,
    )

    device = args.device
    trainer = ManifoldTrainer(dataset=dataset, config=config, device=device)
    print("[predict_frontier] training manifold model …", flush=True)
    trainer.train()

    t_range = (t_max - t_min) if (t_max - t_min) > 0 else 1.0
    fr_dict: Dict[str, float] = {}
    fr_ranked: list = []

    if eval_papers:
        eval_timestamps = [_ensure_tz(p.published).timestamp() for p in eval_papers]
        eval_times_norm = [((t - t_min) / t_range) ** 2 for t in eval_timestamps]
        eval_cluster_ids = assign_to_centroids(eval_embeddings, centroids)
        eval_emb_tensor = torch.tensor(eval_embeddings, dtype=torch.float32).to(device)
        eval_t_tensor = torch.tensor(eval_times_norm, dtype=torch.float32).to(device)
        eval_c_tensor = torch.tensor(eval_cluster_ids, dtype=torch.long).to(device)
        trainer.mapper.eval()
        trainer.mapper.to(device)
        with torch.no_grad():
            z_eval = trainer.mapper(eval_emb_tensor, eval_t_tensor)
        t_now_eval = max(eval_times_norm) if eval_times_norm else 1.0
        recent_fraction = t_now_eval - min(eval_times_norm) + 1e-6 if eval_times_norm else 1.0
        fr_dict, fr_ranked = frontier_scores_at_publication_time(
            z=z_eval,
            spine=trainer.spine,
            times=eval_t_tensor,
            cluster_ids=eval_c_tensor,
            arxiv_ids=[p.arxiv_id for p in eval_papers],
            t_now=t_now_eval,
            recent_fraction=recent_fraction,
        )

    return {
        "fr_dict": fr_dict,
        "fr_ranked": fr_ranked,
    }


def run_naive_pred(
    args: argparse.Namespace,
    data_dir: Path,
    train_papers: list,
    train_embeddings: list,
    eval_papers: list,
    eval_embeddings: list,
    all_edges: list,
) -> Dict[str, float]:
    import torch

    dataset, _, _, _ = build_manifold_dataset_from_papers(
        data_dir=data_dir,
        papers=train_papers,
        embeddings=train_embeddings,
        inspiration_edges=all_edges,
        requested_k=args.cluster_count,
        kmeans_seed=args.seed,
        device=args.device,
    )
    print(
        f"[predict_frontier] naive dataset: {dataset.n_papers} train papers",
        flush=True,
    )

    config = NaiveTrainingConfig(
        hidden_dims=[512, 256],
        layer_norm=True,
        lr=1e-4,
        weight_decay=1e-4,
        n_epochs=50,
        log_every=args.log_every,
    )

    trainer = NaiveTrainer(dataset=dataset, config=config, device=args.device)
    print("[predict_frontier] training naive model …", flush=True)
    trainer.train()

    naive_dict: Dict[str, float] = {}
    if eval_papers:
        device = args.device
        eval_emb_tensor = torch.tensor(eval_embeddings, dtype=torch.float32).to(device)
        trainer.model.eval()
        with torch.no_grad():
            pred_weights = trainer.model(eval_emb_tensor).cpu().numpy()
        for i, p in enumerate(eval_papers):
            naive_dict[p.arxiv_id] = float(pred_weights[i])

    return naive_dict


def run_llm_pred(
    eval_papers: list,
    topic_slug: str,
    hint_papers: list[tuple[str, float]] | None = None,
) -> Tuple[Dict[str, float], str]:
    cfg.pipeline.topic = topic_slug  # needed for evaluate_ideas prompt
    eval_ideas: list[GeneratedIdea] = [
        GeneratedIdea(title=p.title, abstract=p.abstract or "") for p in eval_papers
    ]
    llm_potential_by_id: Dict[str, float] = {}
    if eval_ideas:
        print("[predict_frontier] evaluating eval papers via LLM-as-judge …", flush=True)
        eval_ideas = evaluate_ideas(
            cfg,
            eval_ideas,
            hint_papers=hint_papers,
        )
        llm_potential_by_id = {
            p.arxiv_id: float(eval_ideas[i].potential_impact_score or 0.0)
            for i, p in enumerate(eval_papers)
        }
    model_key = get_llm_eval_model_key()
    return llm_potential_by_id, model_key
