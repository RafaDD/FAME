from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from schemas import PaperImpact


_IMPACT_FILE = "impact.json"


def _impact_path(topic_dir: Path) -> Path:
    return topic_dir / _IMPACT_FILE


def load_impact_data(topic_dir: Path) -> Dict[str, PaperImpact]:
    path = _impact_path(topic_dir)
    if not path.exists():
        return {}

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    impacts: Dict[str, PaperImpact] = {}
    if isinstance(raw, list):
        for item in raw:
            try:
                impact = PaperImpact(**item)
            except Exception:
                continue
            impacts[impact.arxiv_id] = impact
    elif isinstance(raw, dict):
        for aid, item in raw.items():
            if not isinstance(item, dict):
                continue
            try:
                impact = PaperImpact(arxiv_id=aid, **item)
            except Exception:
                continue
            impacts[impact.arxiv_id] = impact

    return impacts


def save_impact_data(topic_dir: Path, impacts: List[PaperImpact]) -> None:
    topic_dir.mkdir(parents=True, exist_ok=True)
    payload = [impact.model_dump(mode="json") for impact in impacts]
    _impact_path(topic_dir).write_text(
        json.dumps(payload, separators=(",", ":")),
        encoding="utf-8",
    )

