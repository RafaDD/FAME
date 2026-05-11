from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from schemas import InspirationEdge, InspirationGraphSummary, Paper

_PAPERS_FILE = "papers.json"
_EMBEDDINGS_FILE = "embeddings.npz"
_INSPIRATION_FILE = "inspiration.json"


@dataclass
class TopicCache:
    papers: list[Paper]
    embeddings: list[list[float]]
    inspiration_edges: list[InspirationEdge]
    inspiration_summary: Optional[InspirationGraphSummary]
    used_embedding_fallback: bool


def topic_data_dir(topic: str) -> Path:
    slug = "_".join(topic.strip().lower().split())[:50]
    return Path("data") / slug


def load_topic_data(topic_dir: Path) -> Optional[TopicCache]:
    papers_path = topic_dir / _PAPERS_FILE
    embeddings_path = topic_dir / _EMBEDDINGS_FILE
    inspiration_path = topic_dir / _INSPIRATION_FILE

    if not papers_path.exists() or not embeddings_path.exists():
        return None

    try:
        papers_raw = json.loads(papers_path.read_text(encoding="utf-8"))
        papers = [Paper(**item) for item in papers_raw]
    except Exception:
        return None

    try:
        npz = np.load(str(embeddings_path), allow_pickle=False)
        stored_ids: list[str] = npz["arxiv_ids"].tolist()
        matrix: np.ndarray = npz["embeddings"].astype(np.float32)
        if matrix.ndim != 2 or len(stored_ids) != matrix.shape[0]:
            return None
        id_to_row = {aid: i for i, aid in enumerate(stored_ids)}
        embeddings: list[list[float]] = []
        for paper in papers:
            row_idx = id_to_row.get(paper.arxiv_id)
            if row_idx is None:
                return None
            embeddings.append(matrix[row_idx].tolist())
    except Exception:
        return None

    edges: list[InspirationEdge] = []
    summary: Optional[InspirationGraphSummary] = None
    if inspiration_path.exists():
        try:
            insp_raw = json.loads(inspiration_path.read_text(encoding="utf-8"))
            edges = [InspirationEdge(**e) for e in insp_raw.get("edges", [])]
            summary_obj = insp_raw.get("summary")
            if isinstance(summary_obj, dict):
                summary = InspirationGraphSummary(**summary_obj)
        except Exception:
            edges = []
            summary = None

    used_fallback = False
    meta_path = topic_dir / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            used_fallback = bool(meta.get("used_embedding_fallback", False))
        except Exception:
            pass

    return TopicCache(
        papers=papers,
        embeddings=embeddings,
        inspiration_edges=edges,
        inspiration_summary=summary,
        used_embedding_fallback=used_fallback,
    )


def save_topic_data(
    topic_dir: Path,
    *,
    papers: list[Paper],
    embeddings: list[list[float]],
    inspiration_edges: list[InspirationEdge],
    inspiration_summary: Optional[InspirationGraphSummary],
    used_embedding_fallback: bool,
) -> None:
    topic_dir.mkdir(parents=True, exist_ok=True)

    papers_payload = []
    for paper in papers:
        d = paper.model_dump(mode="json")
        d["embedding"] = None
        d["topic_affinities"] = []
        papers_payload.append(d)
    (topic_dir / _PAPERS_FILE).write_text(
        json.dumps(papers_payload, separators=(",", ":")),
        encoding="utf-8",
    )

    arxiv_ids = np.array([p.arxiv_id for p in papers])
    matrix = np.array(embeddings, dtype=np.float32)
    np.savez_compressed(str(topic_dir / _EMBEDDINGS_FILE), arxiv_ids=arxiv_ids, embeddings=matrix)

    (topic_dir / _INSPIRATION_FILE).write_text(
        json.dumps(
            {
                "edges": [
                    {k: v for k, v in e.model_dump(mode="json").items() if k != "rationale"}
                    for e in inspiration_edges
                ],
                "summary": inspiration_summary.model_dump(mode="json") if inspiration_summary else None,
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )

    (topic_dir / "meta.json").write_text(
        json.dumps({"used_embedding_fallback": used_embedding_fallback}, separators=(",", ":")),
        encoding="utf-8",
    )
