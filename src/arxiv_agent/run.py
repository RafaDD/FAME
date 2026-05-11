from __future__ import annotations
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from dotenv import load_dotenv

from arxiv_agent.arxiv_client import search_arxiv
from arxiv_agent.cache_store import load_topic_data, save_topic_data, topic_data_dir
from arxiv_agent.progress import _progress
from arxiv_agent.inspiration import detect_inspiration_edges
from arxiv_agent.topic_embedding import (
    analyze_cluster_topics,
    cluster_embeddings,
    compute_topic_affinities,
    embed_papers,
)
from schemas import InspirationEdge, InspirationGraphSummary, Paper, RankedPaper, TopicCluster

if TYPE_CHECKING:
    from config import AppConfig


@dataclass
class ArxivAgentResult:
    topic: str
    topic_dir: Path
    ranked_papers: List[RankedPaper]          # all papers, fully scored (not yet top-k)
    topic_clusters: List[TopicCluster]
    centroids: List[List[float]]
    inspiration_edges: List[InspirationEdge]
    inspiration_summary: Optional[InspirationGraphSummary]
    embedding_model: str
    used_embedding_fallback: bool
    tsne_papers: List[Paper]                  # deep-copy of all papers for UMAP visualization


def run_arxiv_agent(config: "AppConfig") -> ArxivAgentResult:
    load_dotenv()

    pc = config.pipeline
    lc = config.llm

    seed = pc.seed
    if seed is not None:
        random.seed(seed)
        try:
            import numpy as np
            np.random.seed(seed)
        except ImportError:
            pass
        _progress(f"Random seed set to {seed}.")

    embedding_model = lc.embedding_model
    used_embedding_fallback = False
    topic_clusters: List[TopicCluster] = []
    inspiration_edges: List[InspirationEdge] = []
    inspiration_summary: Optional[InspirationGraphSummary] = None
    centroids: List[List[float]] = []

    t_dir = topic_data_dir(pc.topic)

    if pc.max_results > 0:
        _progress(f"Searching arXiv for topic: {pc.topic!r} (max_results={pc.max_results})")
        fetched_papers = search_arxiv(
            topic=pc.topic,
            max_results=pc.max_results,
            since_year=pc.since_year,
            randomize_batch=not pc.deterministic_arxiv,
            seed=seed,
        )
        _progress(f"Retrieved {len(fetched_papers)} papers from arXiv.")

    if not fetched_papers:
        raise ValueError("No arXiv papers found for this topic.")

    existing = load_topic_data(t_dir)
    existing_ids: set[str] = set()
    existing_papers: list = []
    existing_embeddings: List[List[float]] = []

    if existing is not None:
        _progress(f"Loaded {len(existing.papers)} existing papers from {t_dir}.")
        existing_papers = existing.papers
        existing_embeddings = existing.embeddings
        existing_ids = {p.arxiv_id for p in existing_papers}
        inspiration_edges = existing.inspiration_edges
        inspiration_summary = existing.inspiration_summary
        used_embedding_fallback = existing.used_embedding_fallback
    else:
        _progress("No existing topic store found — starting fresh.")

    new_papers = [p for p in fetched_papers if p.arxiv_id not in existing_ids]
    _progress(
        f"{len(new_papers)} new papers to embed "
        f"({len(fetched_papers) - len(new_papers)} duplicates skipped)."
    )

    new_embeddings: List[List[float]] = []
    if new_papers:
        _progress(f"Embedding {len(new_papers)} new papers using {embedding_model}...")
        new_embeddings, new_fallback = embed_papers(papers=new_papers, model=embedding_model)
        used_embedding_fallback = used_embedding_fallback or new_fallback

    all_papers = existing_papers + new_papers
    all_embeddings = existing_embeddings + new_embeddings

    if not all_papers:
        raise ValueError("No papers available after merging.")

    _progress(f"Merged dataset: {len(all_papers)} papers total.")

    for paper, emb in zip(all_papers, all_embeddings):
        paper.embedding = emb

    _progress("Clustering all merged paper embeddings into topic groups...")
    assignments, centroids, topic_clusters = cluster_embeddings(
        papers=all_papers,
        embeddings=all_embeddings,
        requested_k=pc.cluster_count,
        seed=seed,
    )

    if pc.analyze_clusters:
        _progress("Analyzing cluster topics with LLM...")
        topic_clusters = analyze_cluster_topics(
            topic=pc.topic, clusters=topic_clusters, model=lc.compare_model
        )
    else:
        _progress("Skipping LLM cluster analysis (set analyze_clusters=True to enable).")

    ranked_all: List[RankedPaper] = []
    for paper, emb, cluster_id in zip(all_papers, all_embeddings, assignments):
        affinities = compute_topic_affinities(emb, centroids)
        paper.topic_affinities = affinities
        ranked_all.append(RankedPaper(paper=paper, score=0.0, cluster_id=cluster_id))

    tsne_all_papers = [paper.model_copy(deep=True) for paper in all_papers]

    existing_edge_pairs: set[tuple[str, str]] = {
        (e.source_arxiv_id, e.target_arxiv_id) for e in inspiration_edges
    }

    if new_papers:
        _progress(
            f"Detecting inspiration edges for {len(new_papers)} new papers "
            "against merged ranked list..."
        )
        new_inspiration = detect_inspiration_edges(
            ranked_papers=ranked_all,
            model=lc.compare_model,
            top_k_per_later_paper=max(1, pc.inspiration_top_k),
            similarity_threshold=0.55,
            filter_source_ids={p.arxiv_id for p in new_papers},
        )
        for edge in new_inspiration.edges:
            pair = (edge.source_arxiv_id, edge.target_arxiv_id)
            if pair not in existing_edge_pairs:
                inspiration_edges.append(edge)
                existing_edge_pairs.add(pair)
        inspiration_summary = new_inspiration.summary
    elif inspiration_edges:
        _progress("No new papers — reusing existing inspiration edges.")
    else:
        _progress("No inspiration edges available and no new papers to detect from.")


    _progress(f"Saving merged topic data to {t_dir}...")
    papers_for_store = [paper.model_copy(deep=True) for paper in all_papers]
    save_topic_data(
        t_dir,
        papers=papers_for_store,
        embeddings=all_embeddings,
        inspiration_edges=inspiration_edges,
        inspiration_summary=inspiration_summary,
        used_embedding_fallback=used_embedding_fallback,
    )

    return ArxivAgentResult(
        topic=pc.topic,
        topic_dir=t_dir,
        ranked_papers=ranked_all,
        topic_clusters=topic_clusters,
        centroids=centroids,
        inspiration_edges=inspiration_edges,
        inspiration_summary=inspiration_summary,
        embedding_model=embedding_model,
        used_embedding_fallback=used_embedding_fallback,
        tsne_papers=tsne_all_papers,
    )
