from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List

from dotenv import load_dotenv
from tqdm import tqdm

from arxiv_agent.cache_store import load_topic_data, topic_data_dir
from impact_agent.github_detection import detect_github_impact
from impact_agent.semantic_scholar import fetch_semantic_scholar_metrics
from impact_agent.altmetric import fetch_altmetric_score
from impact_agent.impact_store import load_impact_data, save_impact_data
from schemas import PaperImpact

if TYPE_CHECKING:
    from config import AppConfig


def _progress(message: str) -> None:
    print(f"[impact_agent] {message}", flush=True)


@dataclass
class ImpactAgentResult:

    topic: str
    topic_dir: Path
    impacts: List[PaperImpact]


def run_impact_agent(config: "AppConfig") -> ImpactAgentResult:
    load_dotenv()

    pc = config.pipeline

    topic_dir = topic_data_dir(pc.topic)
    cache = load_topic_data(topic_dir)
    if cache is None or not cache.papers:
        raise ValueError(
            f"No cached papers found for topic {pc.topic!r} in {topic_dir}. "
            "Run the arxiv agent first."
        )

    existing = load_impact_data(topic_dir)
    _progress(
        f"Loaded {len(existing)} existing impact entries; "
        f"processing {len(cache.papers)} papers..."
    )

    impacts: List[PaperImpact] = []

    for paper in tqdm(cache.papers, desc="Computing impact metrics"):
        prev = existing.get(paper.arxiv_id)
        impact = PaperImpact(
            arxiv_id=paper.arxiv_id,
            github_url=prev.github_url if prev else None,
            github_stars=prev.github_stars if prev else 0,
            citation_count=prev.citation_count if prev else 0,
            influential_citation_count=(
                prev.influential_citation_count if prev else 0
            ),
            altmetric_score=prev.altmetric_score if prev else 0,
        )

        try:
            gh_url, gh_stars = detect_github_impact(
                arxiv_id=paper.arxiv_id,
                title=paper.title,
                abstract=paper.abstract,
                pdf_url=paper.url,
            )
            if gh_url:
                impact.github_url = gh_url
            if gh_stars is not None:
                impact.github_stars = gh_stars
        except Exception as exc:
            _progress(f"GitHub detection failed for {paper.arxiv_id}: {exc}")

        try:
            c_count, ic_count, = fetch_semantic_scholar_metrics(paper.arxiv_id)
            impact.citation_count = c_count
            impact.influential_citation_count = ic_count
        except Exception as exc:
            _progress(f"Semantic Scholar fetch failed for {paper.arxiv_id}: {exc}")

        try:
            score = fetch_altmetric_score(paper.title)
            if score is not None:
                impact.altmetric_score = score
        except Exception as exc:
            _progress(f"Altmetric fetch failed for {paper.arxiv_id}: {exc}")

        impacts.append(impact)

    _progress(f"Saving impact data for {len(impacts)} papers to {topic_dir}...")
    save_impact_data(topic_dir, impacts)

    return ImpactAgentResult(topic=pc.topic, topic_dir=topic_dir, impacts=impacts)

