from __future__ import annotations

import re
from typing import Optional, Tuple
import requests

GITHUB_REGEX = re.compile(
    r"https?://github\.com/[\w\-]+/[\w\-.]+",
    re.IGNORECASE,
)


def _extract_github_from_text(text: str) -> Optional[str]:
    match = GITHUB_REGEX.search(text or "")
    if not match:
        return None
    # Strip trailing punctuation that may be attached in prose
    url = match.group(0).rstrip(").,;\"'")
    return url



def detect_github_impact(
    _arxiv_id: str,
    _title: str,
    abstract: str,
    _pdf_url: Optional[str] = None,
) -> Tuple[Optional[str], Optional[int]]:
    url = _extract_github_from_text(abstract)
    if not url:
        return None, None
    owner, repo = url.split("/")[-2:]

    response = requests.get(f"https://api.github.com/repos/{owner}/{repo}")
    if response.status_code == 200:
        return url, response.json().get("stargazers_count")

    return url, None
    

