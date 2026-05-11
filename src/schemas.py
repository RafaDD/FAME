"""Shared Pydantic schemas used across all modules of the IdeaGen pipeline."""
from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class Paper(BaseModel):
    arxiv_id: str
    title: str
    abstract: str
    authors: List[str]
    published: datetime
    categories: List[str]
    url: str
    embedding: Optional[List[float]] = None
    topic_affinities: List[float] = Field(default_factory=list)


class RankedPaper(BaseModel):
    paper: Paper
    score: float
    cluster_id: Optional[int] = None
    paper_score: Optional["PaperScore"] = None
    manifold_frontier_score: Optional[float] = None


class GeneratedIdea(BaseModel):
    title: str
    abstract: str
    experiment: Optional[str] = None
    novelty_score: Optional[int] = None
    feasibility_score: Optional[int] = None
    interestingness_score: Optional[int] = None
    potential_impact_score: Optional[int] = None
    evaluation_notes: Optional[str] = None
    manifold_frontier_score: Optional[float] = None


class PaperScore(BaseModel):
    citations_score: float = 0.0
    time_score: float = 0.0
    inspire_score: float = 0.0
    final_score: float = 0.0


class TopicCluster(BaseModel):
    cluster_id: int
    label: str
    member_arxiv_ids: List[str] = Field(default_factory=list)
    member_titles: List[str] = Field(default_factory=list)
    centroid_norm: float = 0.0
    analysis: Optional[str] = None


class InspirationEdge(BaseModel):
    source_arxiv_id: str
    target_arxiv_id: str
    is_inspired: bool
    confidence: float = 0.0
    rationale: str = ""
    similarity: float = 0.0
    time_delta_days: int = 0


class InspirationGraphSummary(BaseModel):
    total_candidates: int = 0
    judged_edges: int = 0
    inspired_edges: int = 0


class PaperImpact(BaseModel):
    arxiv_id: str
    github_url: Optional[str] = None
    github_stars: Optional[int] = None
    hf_models_count: int = 0
    hf_datasets_count: int = 0
    hf_total_downloads: int = 0
    hf_total_likes: int = 0
    citation_count: int = 0
    influential_citation_count: int = 0
    altmetric_score: Optional[float] = None

