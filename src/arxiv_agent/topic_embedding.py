from __future__ import annotations

import hashlib
import json
import math
import os
import random
from typing import List, Optional

from openai import OpenAI

from schemas import Paper, TopicCluster


def _norm(vec: List[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


def _normalize(vec: List[float]) -> List[float]:
    n = _norm(vec)
    if n <= 1e-12:
        return vec[:]
    return [x / n for x in vec]


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    return sum(a * b for a, b in zip(vec_a, vec_b))


def _vector_add(acc: List[float], value: List[float]) -> None:
    for i, v in enumerate(value):
        acc[i] += v


def _vector_div(value: List[float], denom: float) -> List[float]:
    if denom <= 0:
        return value[:]
    return [x / denom for x in value]


def _paper_text(paper: Paper) -> str:
    return f"{paper.title}\n\n{paper.abstract}".strip()


def _fallback_embedding(text: str, dim: int = 128) -> List[float]:
    vec = [0.0] * dim
    for token in text.lower().split():
        h = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(h[:2], "big") % dim
        sign = 1.0 if (h[2] % 2 == 0) else -1.0
        vec[idx] += sign
    return _normalize(vec)


def embed_papers(
    papers: List[Paper],
    model: str,
) -> tuple[List[List[float]], bool]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not papers:
        return [], False
    if not api_key:
        return [_fallback_embedding(_paper_text(p)) for p in papers], True

    inputs = [_paper_text(paper) for paper in papers]
    try:
        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(model=model, input=inputs)
        embeddings = [_normalize(list(item.embedding)) for item in response.data]
        if len(embeddings) != len(papers):
            raise ValueError("Embedding response length mismatch.")
        return embeddings, False
    except Exception as exc:
        print(f"[arxiv_agent] Embedding API failed: {exc}. Using deterministic fallback.", flush=True)
        return [_fallback_embedding(text) for text in inputs], True


def embed_texts(
    texts: List[str],
    model: str,
) -> tuple[List[List[float]], bool]:
    if not texts:
        return [], False
    api_key = os.getenv("OPENAI_API_KEY")
    cleaned_texts = [text.strip() for text in texts]
    if not api_key:
        return [_fallback_embedding(text) for text in cleaned_texts], True

    try:
        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(model=model, input=cleaned_texts)
        embeddings = [_normalize(list(item.embedding)) for item in response.data]
        if len(embeddings) != len(cleaned_texts):
            raise ValueError("Embedding response length mismatch.")
        return embeddings, False
    except Exception as exc:
        print(f"[arxiv_agent] Embedding API failed: {exc}. Using deterministic fallback.", flush=True)
        return [_fallback_embedding(text) for text in cleaned_texts], True


def _choose_cluster_count(n_items: int, requested_k: Optional[int]) -> int:
    if n_items <= 1:
        return 1
    if requested_k is not None:
        return max(1, min(requested_k, n_items))
    heuristic = int(round(math.sqrt(n_items)))
    return max(2, min(heuristic, min(8, n_items)))


def _kmeans_assign(embeddings: List[List[float]], centroids: List[List[float]]) -> List[int]:
    assignments: list[int] = []
    for emb in embeddings:
        best_idx = 0
        best_score = -1e9
        for idx, centroid in enumerate(centroids):
            score = cosine_similarity(emb, centroid)
            if score > best_score:
                best_score = score
                best_idx = idx
        assignments.append(best_idx)
    return assignments


def assign_to_centroids(
    embeddings: List[List[float]],
    centroids: List[List[float]],
) -> List[int]:
    return _kmeans_assign(embeddings, centroids)


def _kmeans_recompute(
    embeddings: List[List[float]],
    assignments: List[int],
    k: int,
    seed: Optional[int],
) -> List[List[float]]:
    dim = len(embeddings[0])
    buckets: list[list[List[float]]] = [[] for _ in range(k)]
    for emb, assign in zip(embeddings, assignments):
        buckets[assign].append(emb)

    rng = random.Random(seed)
    new_centroids: list[list[float]] = []
    for bucket in buckets:
        if not bucket:
            new_centroids.append(embeddings[rng.randrange(len(embeddings))][:])
            continue
        centroid = [0.0] * dim
        for emb in bucket:
            _vector_add(centroid, emb)
        centroid = _vector_div(centroid, float(len(bucket)))
        new_centroids.append(_normalize(centroid))
    return new_centroids


def cluster_embeddings(
    papers: List[Paper],
    embeddings: List[List[float]],
    requested_k: Optional[int] = None,
    max_iter: int = 20,
    seed: Optional[int] = None,
) -> tuple[List[int], List[List[float]], List[TopicCluster]]:
    if not papers:
        return [], [], []
    if len(embeddings) != len(papers):
        raise ValueError("Embeddings and papers size mismatch.")

    k = _choose_cluster_count(len(papers), requested_k)
    if k == 1:
        centroid = _normalize(
            _vector_div(
                [sum(values) for values in zip(*embeddings)],
                float(len(embeddings)),
            )
        )
        clusters = [
            TopicCluster(
                cluster_id=0,
                label="topic_0",
                member_arxiv_ids=[paper.arxiv_id for paper in papers],
                centroid_norm=_norm(centroid),
            )
        ]
        return [0] * len(papers), [centroid], clusters

    rng = random.Random(seed)
    centroid_indices = rng.sample(range(len(embeddings)), k=k)
    centroids = [embeddings[i][:] for i in centroid_indices]
    assignments: list[int] = [-1] * len(embeddings)

    for _ in range(max_iter):
        new_assignments = _kmeans_assign(embeddings, centroids)
        if new_assignments == assignments:
            break
        assignments = new_assignments
        centroids = _kmeans_recompute(embeddings, assignments, k, seed)

    cluster_members: list[list[str]] = [[] for _ in range(k)]
    cluster_titles: list[list[str]] = [[] for _ in range(k)]
    for paper, assign in zip(papers, assignments):
        cluster_members[assign].append(paper.arxiv_id)
        cluster_titles[assign].append(paper.title)

    topic_clusters: list[TopicCluster] = []
    for idx, centroid in enumerate(centroids):
        topic_clusters.append(
            TopicCluster(
                cluster_id=idx,
                label=f"topic_{idx}",
                member_arxiv_ids=cluster_members[idx],
                member_titles=cluster_titles[idx],
                centroid_norm=_norm(centroid),
            )
        )
    return assignments, centroids, topic_clusters


def compute_topic_affinities(embedding: List[float], centroids: List[List[float]]) -> List[float]:
    if not centroids:
        return []
    sims = [max(-1.0, min(1.0, cosine_similarity(embedding, centroid))) for centroid in centroids]
    shifted = [sim + 1.0 for sim in sims]
    total = sum(shifted)
    if total <= 1e-12:
        return [1.0 / len(shifted)] * len(shifted)
    return [value / total for value in shifted]


def analyze_cluster_topics(
    topic: str,
    clusters: List[TopicCluster],
    model: Optional[str] = None,
) -> List[TopicCluster]:
    if not clusters:
        return clusters
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        for cluster in clusters:
            cluster.analysis = "Fallback cluster summary based on titles (LLM unavailable)."
        return clusters

    chosen_model = model or os.getenv("OPENAI_CLUSTER_MODEL", "gpt-4.1-mini")
    client = OpenAI(api_key=api_key)
    for cluster in clusters:
        titles = cluster.member_titles[:8]
        prompt = f"""
You summarize a paper cluster for topic '{topic}'.
Given paper titles, return strict JSON:
{{
  "label": "short cluster topic label (3-8 words)",
  "analysis": "1-2 sentences describing the shared topic"
}}

Titles:
{json.dumps(titles, indent=2)}
"""
        try:
            response = client.responses.create(model=chosen_model, input=prompt, temperature=0.2)
            text = response.output_text.strip()
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError("Invalid cluster analysis JSON.")
            payload = json.loads(text[start : end + 1])
            cluster.label = str(payload.get("label", cluster.label)).strip() or cluster.label
            cluster.analysis = (
                str(payload.get("analysis", "")).strip()
                or "LLM returned no cluster analysis."
            )
        except Exception as exc:
            print(f"[arxiv_agent] Cluster topic analysis failed: {exc}. Using fallback label.", flush=True)
            cluster.analysis = "Fallback cluster summary based on titles."
    return clusters
