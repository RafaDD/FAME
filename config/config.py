from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional



@dataclass
class LLMConfig:
    """Controls which LLM / embedding models are used across the pipeline."""

    model: Optional[str] = None                        # idea generation model
    eval_model: Optional[str] = None                   # idea evaluation model
    compare_model: Optional[str] = None                # inspiration / cluster model
    embedding_model: str = "text-embedding-3-small"    # paper embedding model



@dataclass
class PipelineConfig:
    """Controls paper retrieval, clustering and idea generation."""

    topic: str = ""

    seed: Optional[int] = 42

    max_results: int = 1
    since_year: Optional[int] = None
    deterministic_arxiv: bool = False      # disable random batch sampling

    cluster_count: Optional[int] = 5

    inspiration_top_k: int = 5

    analyze_clusters: bool = False         # LLM topic analysis per cluster



@dataclass
class ManifoldConfig:
    """Controls manifold model architecture, training and post-training analysis."""

    device: str = "cuda:0"

    epochs: int = 800
    lr: float = 1e-4
    weight_decay: float = 1e-5
    log_every: int = 50

    alpha: float = 1.0    # L_spine
    beta: float = 0.1     # L_inspire
    nu: float = 0.3       # L_vanguard

    out_dim: int = 128
    spine_e_dim: int = 128
    mapper_hidden: List[int] = field(default_factory=lambda: [256, 256])
    spine_hidden: List[int] = field(default_factory=lambda: [256, 256])
    layer_norm: bool = True

    inspire_margin: float = 0.5
    inspire_eta: float = 0.01

    delta_vg: float = 0.0 
    vanguard_eps: float = 1e-4

    frontier_recent_fraction: float = 0.4



@dataclass
class AppConfig:
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    manifold: ManifoldConfig = field(default_factory=ManifoldConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)


cfg = AppConfig()



def apply_args_to_config(base_cfg: AppConfig, args: argparse.Namespace) -> AppConfig:
    """Return a deep copy of *base_cfg* with CLI argument overrides applied."""
    config = deepcopy(base_cfg)
    p = config.pipeline
    m = config.manifold
    lc = config.llm

    p.topic = args.topic
    p.max_results = args.max_results
    p.since_year = getattr(args, "since_year", p.since_year)
    p.seed = getattr(args, "seed", p.seed)
    p.cluster_count = getattr(args, "cluster_count", p.cluster_count)
    p.inspiration_top_k = getattr(args, "inspiration_top_k", p.inspiration_top_k)
    p.analyze_clusters = getattr(args, "analyze_clusters", p.analyze_clusters)
    p.deterministic_arxiv = getattr(args, "deterministic_arxiv", p.deterministic_arxiv)

    m.epochs = getattr(args, "manifold_epochs", m.epochs)
    m.out_dim = getattr(args, "manifold_out_dim", m.out_dim)

    if getattr(args, "model", None):
        lc.model = args.model
    if getattr(args, "eval_model", None):
        lc.eval_model = args.eval_model
    if getattr(args, "compare_model", None):
        lc.compare_model = args.compare_model
    if getattr(args, "embedding_model", None):
        lc.embedding_model = args.embedding_model

    return config
