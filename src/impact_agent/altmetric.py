from __future__ import annotations

import os
from typing import Optional

from altmetric.explorer import api


def fetch_altmetric_score(title: str) -> Optional[int]:

    API_KEY = os.getenv("ALTMETRIC_API_KEY")
    API_SECRET = os.getenv("ALTMETRIC_SECRET")

    try:
        client = api.Client('https://www.altmetric.com/explorer/api', API_KEY, API_SECRET)
        response = client.get_research_outputs(title=title, scope='all')

        paper_data = list(response.data)[0]["attributes"]

        return paper_data["altmetric-score"]

    except Exception as exc:
        return None 