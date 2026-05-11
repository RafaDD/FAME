from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from openai import OpenAI

from schemas import InspirationEdge, InspirationGraphSummary, RankedPaper
from arxiv_agent.topic_embedding import cosine_similarity
from arxiv_agent.progress import _progress

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


@dataclass
class InspirationRunResult:
    edges: List[InspirationEdge]
    summary: InspirationGraphSummary
    incoming_inspire_weight: Dict[str, float]


def _extract_json_payload(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = [line for line in stripped.splitlines() if not line.strip().startswith("```")]
        stripped = "\n".join(lines).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(stripped[start : end + 1])
        raise


_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")

def _normalize_arxiv_id(value: str) -> str:
    raw = (value or "").strip().lower()
    if not raw:
        return ""
    if "/" in raw:
        raw = raw.rsplit("/", maxsplit=1)[-1]
    if raw.startswith("arxiv:"):
        raw = raw.split(":", maxsplit=1)[-1]
    match = _ARXIV_ID_RE.search(raw)
    if match:
        return match.group(1)
    return raw.split("v", maxsplit=1)[0]


def _citation_id_set(paper: object) -> set[str]:
    citation_fields = (
        "citation_arxiv_ids",
        "cited_arxiv_ids",
        "reference_arxiv_ids",
        "citations",
        "references",
    )
    vals: list[str] = []
    for field in citation_fields:
        value = getattr(paper, field, None)
        if isinstance(value, list):
            vals.extend(str(v) for v in value if v)
            break
    return {cid for cid in (_normalize_arxiv_id(v) for v in vals) if cid}


def _build_candidates(
    ranked_papers: List[RankedPaper],
    top_k: int,
    min_time_delta_days: int,
    similarity_threshold: float,
    filter_source_ids: Optional[set[str]] = None,
) -> tuple[list[tuple[RankedPaper, RankedPaper, float, int]], int]:
    candidates: list[tuple[RankedPaper, RankedPaper, float, int]] = []
    total_time_ordered_pairs = 0

    for later in ranked_papers:
        if filter_source_ids is not None and later.paper.arxiv_id not in filter_source_ids:
            continue
        cited_ids = _citation_id_set(later.paper)
        pair_buffer: list[tuple[RankedPaper, RankedPaper, float, int]] = []
        for earlier in ranked_papers:
            if later.paper.arxiv_id == earlier.paper.arxiv_id:
                continue
            if later.paper.published <= earlier.paper.published:
                continue
            earlier_id = _normalize_arxiv_id(earlier.paper.arxiv_id)
            if earlier_id not in cited_ids:
                pass
            days = (later.paper.published - earlier.paper.published).days
            if days < min_time_delta_days:
                continue
            total_time_ordered_pairs += 1
            if not later.paper.embedding or not earlier.paper.embedding:
                continue
            similarity = cosine_similarity(later.paper.embedding, earlier.paper.embedding)
            if similarity < similarity_threshold:
                continue
            topic_score = 0.0
            if later.paper.topic_affinities and earlier.paper.topic_affinities:
                topic_score = sum(
                    a * b for a, b in zip(later.paper.topic_affinities, earlier.paper.topic_affinities)
                )
            combined = 0.8 * similarity + 0.2 * topic_score
            pair_buffer.append((later, earlier, combined, days))
        pair_buffer.sort(key=lambda item: item[2], reverse=True)
        candidates.extend(pair_buffer[: max(0, top_k)])
    return candidates, total_time_ordered_pairs


def _heuristic_judge(similarity: float) -> tuple[bool, float, str]:
    if similarity >= 0.85:
        return True, 0.85, "High semantic similarity indicates likely inspiration."
    if similarity >= 0.72:
        return True, 0.7, "Moderate-high semantic overlap suggests partial inspiration."
    return False, 0.3, "Low semantic overlap does not support inspiration."


def detect_inspiration_edges(
    ranked_papers: List[RankedPaper],
    model: Optional[str] = None,
    top_k_per_later_paper: int = 3,
    confidence_threshold: float = 0.6,
    min_time_delta_days: int = 60,
    similarity_threshold: float = 0.6,
    filter_source_ids: Optional[set[str]] = None,
) -> InspirationRunResult:
    candidates, all_time_ordered_pairs = _build_candidates(
        ranked_papers,
        top_k=top_k_per_later_paper,
        min_time_delta_days=max(0, min_time_delta_days),
        similarity_threshold=max(-1.0, min(1.0, similarity_threshold)),
        filter_source_ids=filter_source_ids,
    )
    if not candidates:
        return InspirationRunResult(
            edges=[],
            summary=InspirationGraphSummary(
                total_candidates=all_time_ordered_pairs,
                judged_edges=0,
                inspired_edges=0,
            ),
            incoming_inspire_weight={},
        )

    api_key = os.getenv("OPENAI_API_KEY")
    chosen_model = model or os.getenv("OPENAI_INSPIRE_MODEL", "gpt-4.1-mini")
    _progress(f"using {chosen_model} for inspiration detection")
    client = OpenAI(api_key=api_key) if api_key else None
    edges: list[InspirationEdge] = []
    incoming_weight: dict[str, float] = {}

    iterator = candidates
    if tqdm is not None:
        iterator = tqdm(candidates, desc="Inspiration checks", unit="pair")

    for later, earlier, similarity, time_delta_days in iterator:
        is_inspired = False
        confidence = 0.0
        rationale = ""

        if client is None:
            is_inspired, confidence, rationale = _heuristic_judge(similarity)
        else:
            prompt = f"""
You judge whether a later paper is inspired by an earlier paper.
Return strict JSON:
{{
  "is_inspired": true,
  "confidence": 0.0,
  "rationale": "1-2 sentence explanation"
}}

Later paper A:
- arXiv ID: {later.paper.arxiv_id}
- Title: {later.paper.title}
- Abstract: {later.paper.abstract}

Earlier paper B:
- arXiv ID: {earlier.paper.arxiv_id}
- Title: {earlier.paper.title}
- Abstract: {earlier.paper.abstract}

Additional signal:
- cosine_similarity: {similarity:.4f}
- time_delta_days(A_minus_B): {time_delta_days}
"""
            try:
                response = client.responses.create(model=chosen_model, input=prompt, temperature=0.1)
                payload = _extract_json_payload(response.output_text.strip())
                is_inspired = bool(payload.get("is_inspired", False))
                confidence = float(payload.get("confidence", 0.0))
                rationale = str(payload.get("rationale", "")).strip()
            except Exception as exc:
                print(f"[arxiv_agent] Inspiration LLM judge failed: {exc}. Using heuristic.", flush=True)
                is_inspired, confidence, rationale = _heuristic_judge(similarity)

        edge = InspirationEdge(
            source_arxiv_id=later.paper.arxiv_id,
            target_arxiv_id=earlier.paper.arxiv_id,
            is_inspired=is_inspired,
            confidence=max(0.0, min(1.0, confidence)),
            rationale=rationale or "No rationale provided.",
            similarity=max(-1.0, min(1.0, similarity)),
            time_delta_days=max(0, time_delta_days),
        )
        edges.append(edge)
        if edge.is_inspired and edge.confidence >= confidence_threshold:
            incoming_weight[edge.target_arxiv_id] = (
                incoming_weight.get(edge.target_arxiv_id, 0.0) + edge.confidence
            )

    inspired_edges = sum(
        1 for edge in edges if edge.is_inspired and edge.confidence >= confidence_threshold
    )
    return InspirationRunResult(
        edges=edges,
        summary=InspirationGraphSummary(
            total_candidates=all_time_ordered_pairs,
            judged_edges=len(edges),
            inspired_edges=inspired_edges,
        ),
        incoming_inspire_weight=incoming_weight,
    )
