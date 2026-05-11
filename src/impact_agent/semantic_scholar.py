from __future__ import annotations

import os
from time import sleep
from typing import Tuple

import requests


def fetch_semantic_scholar_metrics(arxiv_id: str) -> Tuple[int, int]:
    endpoint = (
        f"https://api.semanticscholar.org/graph/v1/paper/ARXIV:{arxiv_id[:10]}?fields=citationCount,influentialCitationCount"
    )

    headers: dict[str, str] = {}
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key

    max_try = 8
    for i in range(max_try):
        try:
            response = requests.get(endpoint, headers=headers).json()

            c = response.get("citationCount")
            ic = response.get("influentialCitationCount")

            citation_count = int(c)
            influential_count = int(ic)

            return citation_count, influential_count

        except Exception:
            sleep(5)
            continue

    return 0, 0

