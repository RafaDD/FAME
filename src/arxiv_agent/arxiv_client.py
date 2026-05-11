from __future__ import annotations

from datetime import datetime
import random
from typing import List, Optional
from xml.etree import ElementTree as ET

import httpx

from schemas import Paper

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def build_query(topic: str, since_year: Optional[int] = None) -> str:
    cleaned = " ".join(topic.strip().split())
    query = f'all:"{cleaned}"'
    if since_year:
        query += f" AND submittedDate:[{since_year}01010000 TO 300001010000]"
    return query


def _get_text(node: Optional[ET.Element], default: str = "") -> str:
    return node.text.strip() if node is not None and node.text else default


def parse_arxiv_response(xml_text: str) -> List[Paper]:
    root = ET.fromstring(xml_text)
    papers: List[Paper] = []
    seen_ids: set[str] = set()

    for entry in root.findall("atom:entry", ATOM_NS):
        id_text = _get_text(entry.find("atom:id", ATOM_NS))
        arxiv_id = id_text.rsplit("/", maxsplit=1)[-1]
        if not arxiv_id or arxiv_id in seen_ids:
            continue
        seen_ids.add(arxiv_id)

        title = " ".join(_get_text(entry.find("atom:title", ATOM_NS)).split())
        abstract = " ".join(_get_text(entry.find("atom:summary", ATOM_NS)).split())
        published_raw = _get_text(entry.find("atom:published", ATOM_NS))
        published = datetime.fromisoformat(published_raw.replace("Z", "+00:00"))

        authors = [
            _get_text(author.find("atom:name", ATOM_NS))
            for author in entry.findall("atom:author", ATOM_NS)
            if _get_text(author.find("atom:name", ATOM_NS))
        ]
        categories = [
            cat.attrib.get("term", "")
            for cat in entry.findall("atom:category", ATOM_NS)
            if cat.attrib.get("term")
        ]
        paper_url = ""
        for link in entry.findall("atom:link", ATOM_NS):
            if link.attrib.get("rel") == "alternate":
                paper_url = link.attrib.get("href", "")
                break
        if not paper_url:
            paper_url = id_text

        papers.append(
            Paper(
                arxiv_id=arxiv_id,
                title=title,
                abstract=abstract,
                authors=authors,
                published=published,
                categories=categories,
                url=paper_url,
            )
        )
    return papers


def search_arxiv(
    topic: str,
    max_results: int = 25,
    since_year: Optional[int] = None,
    timeout: float = 20.0,
    randomize_batch: bool = True,
    seed: Optional[int] = None,
) -> List[Paper]:
    fetch_count = max_results * 3 if randomize_batch else max_results
    params = {
        "search_query": build_query(topic, since_year),
        "start": 0,
        "max_results": fetch_count,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    with httpx.Client(timeout=timeout) as client:
        response = client.get(ARXIV_API_URL, params=params)
        response.raise_for_status()
    papers = parse_arxiv_response(response.text)
    if not randomize_batch or len(papers) <= max_results:
        return papers[:max_results]
    rng = random.Random(seed)
    return rng.sample(papers, k=max_results)
